"""导出已解析菜谱、chunk、Markdown 和人工验收报告。"""

import json
from pathlib import Path

from dish_rag.models import Recipe, RecipeChunk


def write_recipes_json(recipes: list[Recipe], output_path: Path) -> None:
    """写出标准结构化菜谱事实。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [recipe.model_dump(mode="json") for recipe in recipes]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_chunks_jsonl(chunks: list[RecipeChunk], output_path: Path) -> None:
    """把可检索 chunk 写成 JSONL。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [chunk.model_dump_json() for chunk in chunks]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recipes_markdown(recipes: list[Recipe], output_path: Path) -> None:
    """写出方便人工阅读的 Markdown 菜谱稿。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# 菜谱结构化抽取稿", ""]
    for recipe in recipes:
        lines.extend(
            [
                f"## {recipe.recipe_id} {recipe.name}",
                "",
                f"- PDF 页码：{recipe.page_start}-{recipe.page_end}",
                f"- 解析置信度：{recipe.parse_confidence}",
                f"- 基础信息：{recipe.cuisine}｜{recipe.category}｜{recipe.cooking_method}｜{recipe.difficulty}｜{recipe.time}",
                "",
                "### 原材料",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in recipe.ingredients)
        lines.extend(["", "### 详细做法", ""])
        lines.extend(f"{index}. {step}" for index, step in enumerate(recipe.steps, start=1))
        lines.extend(
            [
                "",
                "### 其他字段",
                "",
                f"- 口味特点：{recipe.taste}",
                f"- 适合人群：{recipe.audience}",
                f"- 饮食标签：{'、'.join(recipe.diet_tags)}",
                f"- 过敏原提示：{'、'.join(recipe.allergens)}",
                f"- 厨具：{'、'.join(recipe.equipment)}",
                f"- 替换：{recipe.substitutions}",
                f"- 保存与复热：{recipe.storage}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_review_report(
    recipes: list[Recipe],
    output_path: Path,
    low_confidence_threshold: float,
) -> None:
    """为低置信菜谱写出 HITL 人工验收清单。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    low_confidence = [
        recipe for recipe in recipes if recipe.parse_confidence < low_confidence_threshold
    ]
    lines = [
        "# 人工数据验收清单",
        "",
        f"- 总菜谱数：{len(recipes)}",
        f"- 低置信阈值：{low_confidence_threshold}",
        f"- 待复核菜谱数：{len(low_confidence)}",
        "",
        "## 低置信菜谱",
        "",
    ]
    if not low_confidence:
        lines.append("当前没有低于阈值的菜谱。")
    for recipe in low_confidence:
        lines.extend(
            [
                f"### [ ] {recipe.recipe_id} {recipe.name}",
                "",
                f"- PDF 页码：{recipe.page_start}-{recipe.page_end}",
                f"- 解析置信度：{recipe.parse_confidence}",
                f"- 警告：{'；'.join(recipe.parse_warnings)}",
                "- 人工结论：",
                "- 需要修正：",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
