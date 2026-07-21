"""菜谱的结构化 chunk 切分。"""

from dish_rag.models import Recipe, RecipeChunk, RecipeField


def chunk_recipe(recipe: Recipe) -> list[RecipeChunk]:
    """把一道结构化菜谱 Recipe 拆成多个可检索的 RecipeChunk。"""

    chunks: list[RecipeChunk] = []
    base_metadata = { # 每个 chunk 都会带上的通用元数据（不会被embedding）。
        "cuisine": recipe.cuisine,
        "category": recipe.category,
        "cooking_method": recipe.cooking_method,
        "difficulty": recipe.difficulty,
        "time": recipe.time,
        "allergens": recipe.allergens,
        "diet_tags": recipe.diet_tags,
    }

    def add(
        field: RecipeField,
        text: str,
        step_no: int | None = None,
        extra_metadata: dict[str, object] | None = None,
    ) -> None:
        """当文本非空时追加一个 chunk。"""
        # field 表示这个 chunk 属于哪个字段（如原料、步骤）；text 这个 chunk 的文本内容；step_no 仅在 field 为步骤时有效，表示这是第几步。

        if not text.strip():
            return # 如果文本是空的，就不创建 chunk。
        suffix = f"step_{step_no:02d}" if step_no else field.value # 生成 chunk id 的后缀，步骤号或字段名。
        metadata = base_metadata | {"recipe_name": recipe.name}
        if extra_metadata:
            metadata |= extra_metadata
        chunks.append(
            RecipeChunk(
                chunk_id=f"{recipe.recipe_id}:{suffix}", # 唯一id
                recipe_id=recipe.recipe_id, # 菜肴id
                recipe_name=recipe.name, # 菜肴名称
                field=field, # chunk 属于哪个字段（如原料、步骤）
                text=text.strip(), # chunk 的正文文本内容（只有这部分会被embedding）
                page=recipe.page_start, # chunk 所在的 PDF 页码，用于溯源
                step_no=step_no,
                metadata=metadata, # 合并元数据
            )
        )

    # 以下这些才是在真正创建 chunk：
    add(RecipeField.BASIC_INFO, _basic_text(recipe))
    add(RecipeField.INGREDIENTS, "；".join(recipe.ingredients))
    total_steps = len(recipe.steps)
    for index, step in enumerate(recipe.steps, start=1): # 每一个做法步骤单独创建一个 chunk，因为用户问：“下一步”“第二步”
        add(
            RecipeField.STEPS,
            f"第{index}步：{step}",
            step_no=index,
            extra_metadata={
                "is_step_chunk": True,
                "previous_step_no": index - 1 if index > 1 else None,
                "next_step_no": index + 1 if index < total_steps else None,
                "total_steps": total_steps,
            },
        )
    add(RecipeField.TASTE, recipe.taste)
    add(RecipeField.AUDIENCE, recipe.audience)
    add(RecipeField.DIET_TAGS, "、".join(recipe.diet_tags))
    add(RecipeField.ALLERGENS, "、".join(recipe.allergens))
    add(RecipeField.EQUIPMENT, "、".join(recipe.equipment))
    add(RecipeField.SUBSTITUTIONS, recipe.substitutions)
    add(RecipeField.STORAGE, recipe.storage)
    return chunks


def chunk_recipes(recipes: list[Recipe]) -> list[RecipeChunk]:
    """为所有已解析（每道菜的）菜谱创建 chunk。"""

    chunks: list[RecipeChunk] = []
    for recipe in recipes:
        chunks.extend(chunk_recipe(recipe))
    return chunks


def _basic_text(recipe: Recipe) -> str:
    """把基础元数据渲染成可检索文本。"""

    return "｜".join(
        part
        for part in [
            recipe.cuisine,
            recipe.category,
            recipe.cooking_method,
            recipe.difficulty,
            recipe.time,
            recipe.serving,
        ]
        if part
    )
