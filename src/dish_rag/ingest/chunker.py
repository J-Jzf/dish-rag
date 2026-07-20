"""菜谱的结构化 chunk 切分。"""

from dish_rag.models import Recipe, RecipeChunk, RecipeField


def chunk_recipe(recipe: Recipe) -> list[RecipeChunk]:
    """按菜谱字段和步骤创建 chunk。"""

    chunks: list[RecipeChunk] = []
    base_metadata = {
        "cuisine": recipe.cuisine,
        "category": recipe.category,
        "cooking_method": recipe.cooking_method,
        "difficulty": recipe.difficulty,
        "time": recipe.time,
        "allergens": recipe.allergens,
        "diet_tags": recipe.diet_tags,
    }

    def add(field: RecipeField, text: str, step_no: int | None = None) -> None:
        """当文本非空时追加一个 chunk。"""

        if not text.strip():
            return
        suffix = f"step_{step_no:02d}" if step_no else field.value
        chunks.append(
            RecipeChunk(
                chunk_id=f"{recipe.recipe_id}:{suffix}",
                recipe_id=recipe.recipe_id,
                recipe_name=recipe.name,
                field=field,
                text=text.strip(),
                page=recipe.page_start,
                step_no=step_no,
                metadata=base_metadata | {"recipe_name": recipe.name},
            )
        )

    add(RecipeField.BASIC_INFO, _basic_text(recipe))
    add(RecipeField.INGREDIENTS, "；".join(recipe.ingredients))
    for index, step in enumerate(recipe.steps, start=1):
        add(RecipeField.STEPS, step, step_no=index)
    add(RecipeField.TASTE, recipe.taste)
    add(RecipeField.AUDIENCE, recipe.audience)
    add(RecipeField.DIET_TAGS, "、".join(recipe.diet_tags))
    add(RecipeField.ALLERGENS, "、".join(recipe.allergens))
    add(RecipeField.EQUIPMENT, "、".join(recipe.equipment))
    add(RecipeField.SUBSTITUTIONS, recipe.substitutions)
    add(RecipeField.STORAGE, recipe.storage)
    return chunks


def chunk_recipes(recipes: list[Recipe]) -> list[RecipeChunk]:
    """为所有已解析菜谱创建 chunk。"""

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
