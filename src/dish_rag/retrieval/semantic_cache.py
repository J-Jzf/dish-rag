"""仅复用稳定查询 action 的检索结果语义缓存。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol
from uuid import uuid4

from dish_rag.llm.opai import EmbeddingClient
from dish_rag.models import RetrievalHit, UserMemorySnapshot
from dish_rag.storage.sqlite_store import SQLiteStore


CACHEABLE_ACTION_TYPES = frozenset({"recipe_lookup", "field_lookup"})


@dataclass(frozen=True)
class SemanticCacheContext:
    """除 Query 向量外，必须完全一致的缓存隔离条件。"""

    cache_scope: str
    knowledge_base_version: str
    action_type: str
    recipe_id_filter: str
    memory_fingerprint: str
    result_limit: int


class SemanticCacheIndex(Protocol):
    def find(
        self,
        vector: list[float],
        context: SemanticCacheContext,
        threshold: float,
    ) -> str | None: ...

    def upsert(
        self,
        cache_id: str,
        vector: list[float],
        context: SemanticCacheContext,
    ) -> None: ...


class SemanticCache:
    """SQLite 元数据/结果 + Qdrant Query 向量组成的保守语义缓存。"""

    def __init__(
        self,
        store: SQLiteStore,
        index: SemanticCacheIndex,
        embeddings: EmbeddingClient,
        threshold: float = 0.85,
    ) -> None:
        self.sqlite_store = store
        self.index = index
        self.embeddings = embeddings
        self.threshold = threshold

    def embed_query(self, query: str) -> list[float]:
        """复用一次 Query embedding，避免缓存命中/未命中重复向量化。"""

        return self.embeddings.embed_query(query)

    def lookup(
        self,
        *,
        query: str,
        action_type: str,
        filters: dict[str, object] | None,
        user_memory: UserMemorySnapshot,
        limit: int,
        cache_scope: str = "runtime",
    ) -> list[RetrievalHit] | None:
        """按 Query 语义和完整上下文读取可复用检索结果。"""

        vector = self.embed_query(query)
        return self.lookup_embedding(
            vector=vector,
            action_type=action_type,
            filters=filters,
            user_memory=user_memory,
            limit=limit,
            cache_scope=cache_scope,
        )

    def lookup_embedding(
        self,
        *,
        vector: list[float],
        action_type: str,
        filters: dict[str, object] | None,
        user_memory: UserMemorySnapshot,
        limit: int,
        cache_scope: str = "runtime",
    ) -> list[RetrievalHit] | None:
        """使用调用方已生成的 embedding 查询，未命中时可供 Hybrid 继续复用。"""

        context = self.context_for(
            action_type=action_type,
            filters=filters,
            user_memory=user_memory,
            limit=limit,
            cache_scope=cache_scope,
        )
        if context is None:
            return None
        cache_id = self.index.find(vector, context, self.threshold)
        if not cache_id:
            return None
        payload = self.sqlite_store.load_semantic_cache_entry(
            cache_id,
            cache_scope=context.cache_scope,
            knowledge_base_version=context.knowledge_base_version,
            action_type=context.action_type,
            recipe_id_filter=context.recipe_id_filter,
            memory_fingerprint=context.memory_fingerprint,
            result_limit=context.result_limit,
        )
        if payload is None:
            return None
        return [
            RetrievalHit.model_validate(item).model_copy(update={"source": "semantic_cache"})
            for item in payload
        ]

    def store(
        self,
        *,
        query: str,
        action_type: str,
        filters: dict[str, object] | None,
        user_memory: UserMemorySnapshot,
        limit: int,
        hits: list[RetrievalHit],
        cache_scope: str = "runtime",
    ) -> None:
        """为允许的 action 保存非空检索结果。"""

        vector = self.embed_query(query)
        self.store_embedding(
            query=query,
            vector=vector,
            action_type=action_type,
            filters=filters,
            user_memory=user_memory,
            limit=limit,
            hits=hits,
            cache_scope=cache_scope,
        )

    def store_embedding(
        self,
        *,
        query: str,
        vector: list[float],
        action_type: str,
        filters: dict[str, object] | None,
        user_memory: UserMemorySnapshot,
        limit: int,
        hits: list[RetrievalHit],
        cache_scope: str = "runtime",
    ) -> None:
        """保存调用方已生成 embedding 的检索结果。"""

        context = self.context_for(
            action_type=action_type,
            filters=filters,
            user_memory=user_memory,
            limit=limit,
            cache_scope=cache_scope,
        )
        if context is None or not hits:
            return
        cache_id = uuid4().hex
        self.sqlite_store.save_semantic_cache_entry(
            cache_id,
            cache_scope=context.cache_scope,
            knowledge_base_version=context.knowledge_base_version,
            action_type=context.action_type,
            recipe_id_filter=context.recipe_id_filter,
            memory_fingerprint=context.memory_fingerprint,
            result_limit=context.result_limit,
            query_text=query,
            hits=[hit.model_dump(mode="json") for hit in hits],
        )
        self.index.upsert(cache_id, vector, context)

    def context_for(
        self,
        *,
        action_type: str,
        filters: dict[str, object] | None,
        user_memory: UserMemorySnapshot,
        limit: int,
        cache_scope: str = "runtime",
    ) -> SemanticCacheContext | None:
        """构造安全的缓存上下文；不在白名单中的 action 一律绕过。"""

        if action_type not in CACHEABLE_ACTION_TYPES:
            return None
        recipe_id = str((filters or {}).get("recipe_id") or "")
        memory_payload = {
            "preferences": sorted(item.canonical for item in user_memory.preferences),
            "restrictions": sorted(item.canonical for item in user_memory.restrictions),
        }
        memory_fingerprint = hashlib.sha256(
            json.dumps(memory_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return SemanticCacheContext(
            cache_scope=cache_scope,
            knowledge_base_version=self.sqlite_store.knowledge_base_version(),
            action_type=action_type,
            recipe_id_filter=recipe_id,
            memory_fingerprint=memory_fingerprint,
            result_limit=limit,
        )
