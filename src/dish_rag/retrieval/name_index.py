"""菜名精确匹配和模糊匹配。"""

from dish_rag.models import Recipe
from dish_rag.storage.sqlite_store import SQLiteStore


class RecipeNameIndex:
    """基于 SQLite 别名表的菜名解析器。（依赖 SQLite 里的 aliases 表）"""

    def __init__(self, store: SQLiteStore) -> None:
        """把解析器绑定到事实库。"""

        self.store = store

    def exact(self, name: str) -> Recipe | None:
        """返回精确别名匹配结果。"""

        return self.store.find_alias(name)

    def similar(self, name: str, limit: int = 5) -> list[Recipe]:
        """返回用于 HITL 消歧的相似菜谱。"""
        # 输入一个菜名，返回最多 5 个相似菜谱。用于 HITL。

        rows = self.store.search_alias_like(name, limit=limit)# 返回若干项，每一项是(alias, recipe_id)
        recipes: list[Recipe] = [] # 保存最终返回的 Recipe 对象
        seen: set[str] = set() # 记录已经加入过的 recipe_id，避免重复。
        for _, recipe_id in rows:
            if recipe_id in seen:
                continue
            recipe = self.store.get_recipe(recipe_id) # 根据 recipe_id 从 SQLite recipes 表取完整菜谱。
            if recipe:
                recipes.append(recipe)
                seen.add(recipe.recipe_id)
        return recipes
