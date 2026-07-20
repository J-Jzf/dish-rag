"""SQLite 事实源存储。

Qdrant 保存可搜索索引；本模块保存标准菜谱事实、chunk、别名、用户记忆
和审计 trace。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from dish_rag.models import Recipe, RecipeChunk


class SQLiteStore:
    """带显式建表逻辑的轻量 SQLite 封装。"""

    def __init__(self, path: Path) -> None:
        """为指定数据库路径创建存储对象。"""

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """打开 SQLite 连接，并让查询结果支持按字段名读取。"""

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def migrate(self) -> None:
        """在表不存在时创建所需表。"""

        with self.connect() as connection:
            connection.executescript(
                """
                create table if not exists recipes (
                    recipe_id text primary key,
                    name text not null,
                    page_start integer not null,
                    page_end integer not null,
                    payload_json text not null
                );

                create table if not exists chunks (
                    chunk_id text primary key,
                    recipe_id text not null,
                    field text not null,
                    step_no integer,
                    page integer not null,
                    text text not null,
                    payload_json text not null,
                    foreign key(recipe_id) references recipes(recipe_id)
                );

                create table if not exists aliases (
                    alias text primary key,
                    recipe_id text not null,
                    normalized_alias text not null,
                    foreign key(recipe_id) references recipes(recipe_id)
                );

                create table if not exists long_term_memory (
                    user_id text not null,
                    memory_key text not null,
                    memory_value text not null,
                    updated_at text default current_timestamp,
                    primary key(user_id, memory_key)
                );

                create table if not exists turn_traces (
                    trace_id integer primary key autoincrement,
                    thread_id text not null,
                    created_at text default current_timestamp,
                    payload_json text not null
                );
                """
            )

    def upsert_recipes(self, recipes: Iterable[Recipe]) -> None:
        """写入或替换菜谱事实和别名。"""

        with self.connect() as connection: # with 结束时会自动提交或关闭连接。
            for recipe in recipes: # 遍历每一道菜
                connection.execute( # 向 recipes 表写入一条菜谱记录
                    """
                    insert into recipes(recipe_id, name, page_start, page_end, payload_json)
                    values (?, ?, ?, ?, ?)
                    on conflict(recipe_id) do update set
                        name=excluded.name,
                        page_start=excluded.page_start,
                        page_end=excluded.page_end,
                        payload_json=excluded.payload_json
                    """,
                    # 写入的字段是菜谱编号、菜名、开始/结束页码、完整 Recipe JSON
                    # on conflict： 如果菜谱编号已存在，则更新菜名、页码和 JSON 而不是报错
                    (
                        recipe.recipe_id, # 此处给 SQL 里的 ? 填值
                        recipe.name,
                        recipe.page_start,
                        recipe.page_end,
                        recipe.model_dump_json(),
                    ),
                )
                for alias in recipe.aliases:
                    connection.execute( # 向 aliases 表写入每个别名（或英文的大小写归一化等）
                        """
                        insert into aliases(alias, recipe_id, normalized_alias)
                        values (?, ?, ?)
                        on conflict(alias) do update set
                            recipe_id=excluded.recipe_id,
                            normalized_alias=excluded.normalized_alias
                        """,
                        (alias, recipe.recipe_id, normalize_name(alias)),
                    )

    def upsert_chunks(self, chunks: Iterable[RecipeChunk]) -> None:
        """写入或替换 chunk 事实。"""

        with self.connect() as connection:
            for chunk in chunks:
                connection.execute(
                    """
                    insert into chunks(chunk_id, recipe_id, field, step_no, page, text, payload_json)
                    values (?, ?, ?, ?, ?, ?, ?)
                    on conflict(chunk_id) do update set
                        recipe_id=excluded.recipe_id,
                        field=excluded.field,
                        step_no=excluded.step_no,
                        page=excluded.page,
                        text=excluded.text,
                        payload_json=excluded.payload_json
                    """,
                    (
                        chunk.chunk_id,
                        chunk.recipe_id,
                        str(chunk.field),
                        chunk.step_no,
                        chunk.page,
                        chunk.text,
                        chunk.model_dump_json(),
                    ),
                )

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        """按编号返回一道菜谱。"""

        with self.connect() as connection:
            row = connection.execute(
                "select payload_json from recipes where recipe_id = ?",
                (recipe_id,),
            ).fetchone()
        return Recipe.model_validate_json(row["payload_json"]) if row else None

    def get_chunk(self, chunk_id: str) -> RecipeChunk | None:
        """按 chunk id 返回一个 chunk。"""

        with self.connect() as connection:
            row = connection.execute(
                "select payload_json from chunks where chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return RecipeChunk.model_validate_json(row["payload_json"]) if row else None

    def all_chunks(self) -> list[RecipeChunk]:
        """加载全部 chunk，用于本地 BM25 兜底和离线评测。"""

        with self.connect() as connection:
            rows = connection.execute("select payload_json from chunks").fetchall()
        return [RecipeChunk.model_validate_json(row["payload_json"]) for row in rows]

    def find_alias(self, name: str) -> Recipe | None:
        """通过别名表解析用户输入的菜名。"""

        normalized = normalize_name(name)
        with self.connect() as connection:
            row = connection.execute(
                "select recipe_id from aliases where normalized_alias = ?",
                (normalized,),
            ).fetchone()
        return self.get_recipe(row["recipe_id"]) if row else None

    def search_alias_like(self, name: str, limit: int = 5) -> list[tuple[str, str]]:
        """返回模糊匹配的别名候选，用于 HITL 建议。"""

        with self.connect() as connection:
            rows = connection.execute("select alias, recipe_id from aliases").fetchall()
        candidates = [(row["alias"], row["recipe_id"]) for row in rows]
        return _rank_aliases(name, candidates, limit)

    def save_memory(self, user_id: str, key: str, value: str) -> None:
        """持久化用户长期偏好或过敏记忆。"""

        with self.connect() as connection:
            connection.execute(
                """
                insert into long_term_memory(user_id, memory_key, memory_value)
                values (?, ?, ?)
                on conflict(user_id, memory_key) do update set
                    memory_value=excluded.memory_value,
                    updated_at=current_timestamp
                """,
                (user_id, key, value),
            )

    def load_memory(self, user_id: str) -> dict[str, str]:
        """加载指定用户的长期记忆。"""

        with self.connect() as connection:
            rows = connection.execute(
                "select memory_key, memory_value from long_term_memory where user_id = ?",
                (user_id,),
            ).fetchall()
        return {row["memory_key"]: row["memory_value"] for row in rows}

    def append_trace(self, thread_id: str, payload: dict[str, Any]) -> None:
        """持久化一轮可观测 trace。"""

        with self.connect() as connection:
            connection.execute(
                "insert into turn_traces(thread_id, payload_json) values (?, ?)",
                (thread_id, json.dumps(payload, ensure_ascii=False)),
            )


def normalize_name(value: str) -> str:
    """归一化菜名，便于精确匹配。"""

    return "".join(value.lower().split())


def _rank_aliases(
    query: str,
    candidates: list[tuple[str, str]],
    limit: int,
) -> list[tuple[str, str]]:
    """优先用 RapidFuzz 排序别名；不可用时退化为子串打分。"""

    try:
        from rapidfuzz import fuzz

        ranked = sorted(
            candidates,
            key=lambda item: fuzz.WRatio(query, item[0]),
            reverse=True,
        )
    except Exception:
        normalized = normalize_name(query)
        ranked = sorted(
            candidates,
            key=lambda item: int(normalized in normalize_name(item[0])),
            reverse=True,
        )
    return ranked[:limit]
