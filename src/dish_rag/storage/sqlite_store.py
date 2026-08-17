"""SQLite 事实源存储。

Qdrant 保存可搜索索引；本模块保存标准菜谱事实、chunk、别名、用户记忆
和审计 trace。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from dish_rag.models import MemoryOperation, Recipe, RecipeChunk, UserMemoryItem, UserMemorySnapshot


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

                create table if not exists user_preferences (
                    user_id text not null,
                    canonical text not null,
                    phrases_json text not null,
                    first_seen_at text default current_timestamp,
                    last_seen_at text default current_timestamp,
                    primary key(user_id, canonical)
                );

                create table if not exists user_restrictions (
                    user_id text not null,
                    canonical text not null,
                    phrases_json text not null,
                    first_seen_at text default current_timestamp,
                    last_seen_at text default current_timestamp,
                    primary key(user_id, canonical)
                );

                create table if not exists turn_traces (
                    trace_id integer primary key autoincrement,
                    thread_id text not null,
                    created_at text default current_timestamp,
                    payload_json text not null
                );

                create table if not exists system_metadata (
                    metadata_key text primary key,
                    metadata_value text not null
                );

                insert or ignore into system_metadata(metadata_key, metadata_value)
                values ('knowledge_base_version', '0');

                create table if not exists semantic_cache_entries (
                    cache_id text primary key,
                    cache_scope text not null,
                    knowledge_base_version text not null,
                    action_type text not null,
                    recipe_id_filter text not null,
                    memory_fingerprint text not null,
                    result_limit integer not null,
                    query_text text not null,
                    hits_json text not null,
                    created_at text default current_timestamp,
                    last_accessed_at text default current_timestamp,
                    hit_count integer not null default 0
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

    def load_user_memory(self, user_id: str) -> UserMemorySnapshot:
        """加载结构化偏好和忌口；普通偏好按最近确认时间排序。"""

        with self.connect() as connection:
            preferences = connection.execute(
                "select canonical, phrases_json, first_seen_at, last_seen_at from user_preferences where user_id = ? order by last_seen_at desc",
                (user_id,),
            ).fetchall()
            restrictions = connection.execute(
                "select canonical, phrases_json, first_seen_at, last_seen_at from user_restrictions where user_id = ? order by last_seen_at desc",
                (user_id,),
            ).fetchall()
        snapshot = UserMemorySnapshot(
            preferences=[_memory_item_from_row(row) for row in preferences],
            restrictions=[_memory_item_from_row(row) for row in restrictions],
        )
        if snapshot.preferences or snapshot.restrictions:
            return snapshot
        legacy = self.load_memory(user_id).get("preference_latest")
        if not legacy:
            return snapshot
        try:
            constraints = json.loads(legacy).get("preserved_constraints", [])
        except (json.JSONDecodeError, AttributeError):
            return snapshot
        return self.apply_memory_operations(
            user_id,
            [MemoryOperation(operation="add_restriction", canonical=str(value), phrase=str(value)) for value in constraints],
        )

    def apply_memory_operations(self, user_id: str, operations: Iterable[MemoryOperation]) -> UserMemorySnapshot:
        """应用已校验的记忆操作，并把不同普通偏好裁剪为最近十条。"""

        with self.connect() as connection:
            for operation in operations:
                table = "user_restrictions" if operation.operation.endswith("restriction") else "user_preferences"
                if operation.operation == "remove_restriction":
                    connection.execute(
                        "delete from user_restrictions where user_id = ? and canonical = ?",
                        (user_id, operation.canonical),
                    )
                    continue
                row = connection.execute(
                    f"select phrases_json from {table} where user_id = ? and canonical = ?",
                    (user_id, operation.canonical),
                ).fetchone()
                phrases = json.loads(row["phrases_json"]) if row else []
                if operation.phrase and operation.phrase not in phrases:
                    phrases.append(operation.phrase)
                connection.execute(
                    f"""insert into {table}(user_id, canonical, phrases_json)
                        values (?, ?, ?)
                        on conflict(user_id, canonical) do update set
                            phrases_json=excluded.phrases_json,
                            last_seen_at=current_timestamp""",
                    (user_id, operation.canonical, json.dumps(phrases, ensure_ascii=False)),
                )
            connection.execute(
                """delete from user_preferences where user_id = ? and canonical not in (
                    select canonical from user_preferences where user_id = ? order by last_seen_at desc limit 10
                )""",
                (user_id, user_id),
            )
        return self.load_user_memory(user_id)

    def append_trace(self, thread_id: str, payload: dict[str, Any]) -> None:
        """持久化一轮可观测 trace。"""

        with self.connect() as connection:
            connection.execute(
                "insert into turn_traces(thread_id, payload_json) values (?, ?)",
                (thread_id, json.dumps(payload, ensure_ascii=False)),
            )

    def knowledge_base_version(self) -> str:
        """返回当前结构化事实/索引构建版本，用于失效旧语义缓存。"""

        with self.connect() as connection:
            row = connection.execute(
                "select metadata_value from system_metadata where metadata_key = 'knowledge_base_version'"
            ).fetchone()
        return row["metadata_value"] if row else "0"

    def bump_knowledge_base_version(self) -> str:
        """知识库重建完成后递增版本，使旧缓存不再参与命中。"""

        with self.connect() as connection:
            row = connection.execute(
                "select metadata_value from system_metadata where metadata_key = 'knowledge_base_version'"
            ).fetchone()
            previous = int(row["metadata_value"]) if row else 0
            version = str(previous + 1)
            connection.execute(
                """insert into system_metadata(metadata_key, metadata_value) values (?, ?)
                   on conflict(metadata_key) do update set metadata_value = excluded.metadata_value""",
                ("knowledge_base_version", version),
            )
        return version

    def save_semantic_cache_entry(
        self,
        cache_id: str,
        *,
        cache_scope: str,
        knowledge_base_version: str,
        action_type: str,
        recipe_id_filter: str,
        memory_fingerprint: str,
        result_limit: int,
        query_text: str,
        hits: list[dict[str, Any]],
    ) -> None:
        """持久化语义缓存元数据及可复用的 RetrievalHit 列表。"""

        with self.connect() as connection:
            connection.execute(
                """insert into semantic_cache_entries(
                    cache_id, cache_scope, knowledge_base_version, action_type,
                    recipe_id_filter, memory_fingerprint, result_limit, query_text, hits_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(cache_id) do update set
                    hits_json=excluded.hits_json,
                    last_accessed_at=current_timestamp""",
                (
                    cache_id,
                    cache_scope,
                    knowledge_base_version,
                    action_type,
                    recipe_id_filter,
                    memory_fingerprint,
                    result_limit,
                    query_text,
                    json.dumps(hits, ensure_ascii=False),
                ),
            )

    def load_semantic_cache_entry(
        self,
        cache_id: str,
        *,
        cache_scope: str,
        knowledge_base_version: str,
        action_type: str,
        recipe_id_filter: str,
        memory_fingerprint: str,
        result_limit: int,
    ) -> list[dict[str, Any]] | None:
        """只有元数据完全一致时才读取缓存结果，避免向量近似命中越界复用。"""

        with self.connect() as connection:
            row = connection.execute(
                """select hits_json from semantic_cache_entries where cache_id = ?
                   and cache_scope = ? and knowledge_base_version = ? and action_type = ?
                   and recipe_id_filter = ? and memory_fingerprint = ? and result_limit = ?""",
                (
                    cache_id,
                    cache_scope,
                    knowledge_base_version,
                    action_type,
                    recipe_id_filter,
                    memory_fingerprint,
                    result_limit,
                ),
            ).fetchone()
            if row:
                connection.execute(
                    """update semantic_cache_entries
                       set last_accessed_at=current_timestamp, hit_count=hit_count + 1
                       where cache_id = ?""",
                    (cache_id,),
                )
        return json.loads(row["hits_json"]) if row else None


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


def _memory_item_from_row(row: sqlite3.Row) -> UserMemoryItem:
    return UserMemoryItem(
        canonical=row["canonical"],
        phrases=json.loads(row["phrases_json"]),
        first_seen_at=row["first_seen_at"] or "",
        last_seen_at=row["last_seen_at"] or "",
    )
