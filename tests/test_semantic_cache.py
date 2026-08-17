from dish_rag.agent.nodes import AgentNodes
from dish_rag.models import CookingState, QueryRewrite, RetrievalHit, TurnTrace, UserMemoryItem, UserMemorySnapshot
from dish_rag.retrieval.semantic_cache import SemanticCache, SemanticCacheContext
from dish_rag.storage.qdrant_store import QdrantSemanticCacheIndex
from dish_rag.storage.sqlite_store import SQLiteStore


class _FakeEmbeddings:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2]


class _FakeCacheIndex:
    def __init__(self):
        self.entries: list[tuple[str, list[float], SemanticCacheContext]] = []

    def find(self, vector, context, threshold):
        for cache_id, _vector, saved_context in self.entries:
            if saved_context == context:
                return cache_id
        return None

    def upsert(self, cache_id, vector, context):
        self.entries.append((cache_id, vector, context))


def _hit(recipe_id: str = "001") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{recipe_id}:steps",
        recipe_id=recipe_id,
        recipe_name="宫保鸡丁",
        field="steps",
        text="第 1 步：处理鸡肉。",
        page=3,
        step_no=1,
        score=0.9,
        source="rerank",
        filters={"recipe_id": recipe_id},
    )


def _cache(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    return SemanticCache(store, _FakeCacheIndex(), _FakeEmbeddings())


def test_semantic_cache_reuses_recipe_lookup_hits_with_same_context(tmp_path):
    cache = _cache(tmp_path)
    memory = UserMemorySnapshot(
        preferences=[UserMemoryItem(canonical="高蛋白")],
        restrictions=[UserMemoryItem(canonical="不吃花生")],
    )

    cache.store(
        query="宫保鸡丁怎么做",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=memory,
        limit=8,
        hits=[_hit()],
    )

    hits = cache.lookup(
        query="教我做宫保鸡丁",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=memory,
        limit=8,
    )

    assert hits is not None
    assert [hit.chunk_id for hit in hits] == ["001:steps"]
    assert hits[0].source == "semantic_cache"


def test_semantic_cache_does_not_reuse_when_effective_memory_changes(tmp_path):
    cache = _cache(tmp_path)
    cache.store(
        query="宫保鸡丁怎么做",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=UserMemorySnapshot(),
        limit=8,
        hits=[_hit()],
    )

    hits = cache.lookup(
        query="教我做宫保鸡丁",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=UserMemorySnapshot(restrictions=[UserMemoryItem(canonical="不吃花生")]),
        limit=8,
    )

    assert hits is None


def test_semantic_cache_only_allows_recipe_and_field_lookup(tmp_path):
    cache = _cache(tmp_path)

    cache.store(
        query="推荐高蛋白菜",
        action_type="recommendation",
        filters={},
        user_memory=UserMemorySnapshot(),
        limit=8,
        hits=[_hit()],
    )

    assert cache.lookup(
        query="推荐高蛋白菜",
        action_type="recommendation",
        filters={},
        user_memory=UserMemorySnapshot(),
        limit=8,
    ) is None


class _CacheHit:
    def __init__(self, hits):
        self.hits = hits
        self.lookups: list[dict] = []

    def embed_query(self, query):
        return [0.1, 0.2]

    def lookup_embedding(self, **kwargs):
        self.lookups.append(kwargs)
        return self.hits

    def store_embedding(self, **kwargs):
        raise AssertionError("cache hit must not execute Hybrid retrieval or write a new cache entry")


class _NeverRetrieve:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("semantic-cache hit must skip Hybrid retrieval and rerank")


def test_recipe_lookup_cache_hit_skips_hybrid_retrieval_but_keeps_existing_hits_shape(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    cache = _CacheHit([_hit()])
    nodes = AgentNodes(chat=None, retriever=_NeverRetrieve(), store=store, semantic_cache=cache)

    result = nodes.retrieve(
        {
            "query_rewrite": QueryRewrite(
                raw_query="宫保鸡丁怎么做",
                rewritten_query="宫保鸡丁怎么做",
                intent="recipe_lookup",
                needs_retrieval=True,
            ),
            "cooking_state": CookingState(),
            "user_memory": UserMemorySnapshot(),
            "trace": TurnTrace(raw_query="宫保鸡丁怎么做"),
        }
    )

    assert [hit.chunk_id for hit in result["hits"]] == ["001:steps"]
    assert cache.lookups[0]["action_type"] == "recipe_lookup"


class _CacheWriteFailure:
    def embed_query(self, query):
        return [0.1, 0.2]

    def lookup_embedding(self, **kwargs):
        return None

    def store_embedding(self, **kwargs):
        raise RuntimeError("cache index unavailable")


class _RecordingRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, limit=8, filters=None, dense_vector=None):
        self.calls += 1
        return [_hit()]


def test_cache_write_failure_keeps_already_retrieved_hits_without_running_hybrid_twice(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    retriever = _RecordingRetriever()
    nodes = AgentNodes(chat=None, retriever=retriever, store=store, semantic_cache=_CacheWriteFailure())

    result = nodes.retrieve(
        {
            "query_rewrite": QueryRewrite(
                raw_query="宫保鸡丁怎么做",
                rewritten_query="宫保鸡丁怎么做",
                intent="recipe_lookup",
                needs_retrieval=True,
            ),
            "cooking_state": CookingState(),
            "user_memory": UserMemorySnapshot(),
            "trace": TurnTrace(raw_query="宫保鸡丁怎么做"),
        }
    )

    assert [hit.chunk_id for hit in result["hits"]] == ["001:steps"]
    assert retriever.calls == 1


def test_semantic_cache_uses_a_separate_local_qdrant_collection(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    cache = SemanticCache(
        store,
        QdrantSemanticCacheIndex("", "", "semantic_cache_test", tmp_path / "qdrant"),
        _FakeEmbeddings(),
    )

    cache.store(
        query="宫保鸡丁怎么做",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=UserMemorySnapshot(),
        limit=5,
        hits=[_hit()],
    )

    hits = cache.lookup(
        query="宫保鸡丁怎么做",
        action_type="recipe_lookup",
        filters={"recipe_id": "001"},
        user_memory=UserMemorySnapshot(),
        limit=5,
    )

    assert hits is not None
    assert hits[0].source == "semantic_cache"


def test_factory_reuses_main_qdrant_client_for_semantic_cache(tmp_path):
    shared_client = object()
    retriever = type(
        "Retriever",
        (),
        {"embeddings": _FakeEmbeddings(), "qdrant": type("Index", (), {"client": shared_client})()},
    )()
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    settings = Settings(qdrant_path=tmp_path / "qdrant")

    with patch("dish_rag.factory.QdrantSemanticCacheIndex") as cache_index:
        make_semantic_cache(settings, store, retriever)

    assert cache_index.call_args.kwargs["client"] is shared_client
from unittest.mock import patch

from dish_rag.config import Settings
from dish_rag.factory import make_semantic_cache
