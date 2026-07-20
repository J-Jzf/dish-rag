"""理解菜谱结构的解析器。

解析器遵循 PDF 中的菜谱字段结构，而不是按字符长度粗暴切分。它会识别
菜谱标题行、固定字段标签、有序步骤和基础元数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dish_rag.models import Recipe


TITLE_RE = re.compile(r"(?m)^(?P<id>\d{3})\s+(?P<name>.+?)\s*$")
STEP_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*")

FIELD_LABELS = {
    "基础信息": "basic_info",
    "原材料": "ingredients",
    "详细做法": "steps",
    "口味特点": "taste",
    "适合人群": "audience",
    "饮食标签": "diet_tags",
    "过敏原提示": "allergens",
    "厨具与替换": "equipment_substitutions",
    "保存与复热": "storage",
}

LABEL_RE = re.compile("|".join(re.escape(label) for label in FIELD_LABELS))


@dataclass
class RecipeBlock:
    """字段级解析前的一道菜原始文本块。"""

    recipe_id: str
    name: str
    text: str
    page_start: int
    page_end: int


def parse_recipes_from_pages(pages: list[str]) -> list[Recipe]:
    """从逐页文本中解析所有菜谱。"""

    combined_parts: list[str] = []
    for page_no, page_text in enumerate(pages, start=1):
        # 页码标记让我们在合并文本后，仍然能追溯每道菜出自哪一页。
        combined_parts.append(f"\n[[PAGE:{page_no}]]\n{page_text}")
    combined = "\n".join(combined_parts)
    blocks = _find_recipe_blocks(combined)
    return [_parse_block(block) for block in blocks]


def _find_recipe_blocks(combined_text: str) -> list[RecipeBlock]:
    """按 `001 菜名` 这种标题格式寻找菜谱边界。"""

    matches = list(TITLE_RE.finditer(combined_text))
    page_markers = [
        (marker.start(), int(marker.group(1)))
        for marker in re.finditer(r"\[\[PAGE:(\d+)]]", combined_text)
    ]
    blocks: list[RecipeBlock] = []
    for index, match in enumerate(matches):
        recipe_id = match.group("id")
        # 附录可能也会出现 001 这样的示例；这里只接受正式的 001-230 范围，
        # 避免把说明文字误当成菜谱。
        if not recipe_id.isdigit() or not (1 <= int(recipe_id) <= 230):
            continue

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(combined_text)
        text = combined_text[start:end].strip()
        pages = _pages_for_span(page_markers, start, end)
        blocks.append(
            RecipeBlock(
                recipe_id=recipe_id,
                name=match.group("name").strip(),
                text=re.sub(r"\[\[PAGE:\d+]]", "", text).strip(),
                page_start=min(pages),
                page_end=max(pages),
            )
        )
    return blocks


def _pages_for_span(page_markers: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """返回一个菜谱文本跨度涉及的 PDF 页码。

    标题通常出现在页码标记之后，所以即使当前文本跨度内部没有页码标记，
    标题前最近的页码标记也属于这道菜。
    """

    preceding = [page for position, page in page_markers if position <= start]
    pages = [page for position, page in page_markers if start <= position < end]
    if preceding:
        pages.insert(0, preceding[-1])
    return pages or [1]


def _parse_block(block: RecipeBlock) -> Recipe:
    """把一道菜原始文本块转换成标准 JSON 结构。"""

    fields = _extract_fields(block.text)
    basic = fields.get("basic_info", "")
    basic_parts = [part.strip() for part in basic.split("｜")]

    ingredients = _split_semicolon_list(fields.get("ingredients", ""))
    steps = _split_steps(fields.get("steps", ""))
    equipment, substitutions = _split_equipment_and_substitutions(
        fields.get("equipment_substitutions", "")
    )

    recipe = Recipe(
        recipe_id=block.recipe_id,
        name=block.name,
        aliases=_make_aliases(block.name),
        page_start=block.page_start,
        page_end=block.page_end,
        cuisine=_safe_part(basic_parts, 0),
        category=_safe_part(basic_parts, 1),
        cooking_method=_safe_part(basic_parts, 2),
        difficulty=_safe_part(basic_parts, 3),
        time=_safe_part(basic_parts, 4),
        serving=_safe_part(basic_parts, 5),
        ingredients=ingredients,
        steps=steps,
        taste=fields.get("taste", ""),
        audience=fields.get("audience", ""),
        diet_tags=_split_comma_list(fields.get("diet_tags", "")),
        allergens=_split_comma_list(fields.get("allergens", "")),
        equipment=equipment,
        substitutions=substitutions,
        storage=fields.get("storage", ""),
        raw_text=block.text,
    )
    recipe.parse_confidence, recipe.parse_warnings = _score_recipe(recipe)
    return recipe


def _extract_fields(text: str) -> dict[str, str]:
    """从菜谱文本块中抽取 PDF 的固定字段。"""

    matches = list(LABEL_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(0)
        key = FIELD_LABELS[label]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        # 合并 PDF 抽取造成的换行，同时保留可读空格。
        fields[key] = re.sub(r"\s+", " ", value)
    return fields


def _split_semicolon_list(value: str) -> list[str]:
    """用中文分号切分原材料这类字段。"""

    return [item.strip(" ；;") for item in re.split(r"[；;]", value) if item.strip(" ；;")]


def _split_comma_list(value: str) -> list[str]:
    """用常见中英文分隔符切分标签类字段。"""

    return [item.strip(" 、，,;；") for item in re.split(r"[、，,;；]", value) if item.strip()]


def _split_steps(value: str) -> list[str]:
    """把带中文序号的做法拆成有序步骤列表。"""

    parts = [part.strip(" ；;") for part in STEP_RE.split(value) if part.strip(" ；;")]
    return parts


def _split_equipment_and_substitutions(value: str) -> tuple[list[str], str]:
    """把第一个句号前的厨具和句号后的替换建议分开。"""

    if "。" in value:
        equipment_text, substitutions = value.split("。", 1)
    else:
        equipment_text, substitutions = value, ""
    return _split_comma_list(equipment_text), substitutions.strip()


def _safe_part(parts: list[str], index: int) -> str:
    """安全读取列表项，避免抛出 IndexError。"""

    return parts[index] if index < len(parts) else ""


def _make_aliases(name: str) -> list[str]:
    """为精确匹配生成简单别名。"""

    aliases = {name.strip()}
    # 有些菜名会在同一标题行里同时包含英文名和中文名。
    if " " in name:
        aliases.add(name.split(" ")[0].strip())
        aliases.add(name.split(" ")[-1].strip())
    return sorted(alias for alias in aliases if alias)


def _score_recipe(recipe: Recipe) -> tuple[float, list[str]]:
    """计算可解释的解析置信度。"""

    warnings: list[str] = []
    checks = {
        "基础信息": bool(recipe.cuisine and recipe.category and recipe.cooking_method),
        "原材料": len(recipe.ingredients) >= 3,
        "详细做法": len(recipe.steps) >= 3,
        "口味特点": bool(recipe.taste),
        "适合人群": bool(recipe.audience),
        "饮食标签": bool(recipe.diet_tags),
        "过敏原提示": bool(recipe.allergens),
        "厨具与替换": bool(recipe.equipment),
        "保存与复热": bool(recipe.storage),
    }
    for label, passed in checks.items():
        if not passed:
            warnings.append(f"字段缺失或较弱：{label}")
    score = sum(1 for passed in checks.values() if passed) / len(checks)
    return round(score, 3), warnings
