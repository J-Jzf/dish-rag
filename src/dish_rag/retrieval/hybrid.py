"""高层检索流水线。"""

from __future__ import annotations

from dish_rag.llm.opai import EmbeddingClient, RerankClient, SparseBM25Encoder
from dish_rag.models import Recipe, RetrievalHit
from dish_rag.retrieval.bm25 import LocalBM25
from dish_rag.retrieval.name_index import RecipeNameIndex
from dish_rag.storage.qdrant_store import QdrantRecipeIndex
from dish_rag.storage.sqlite_store import SQLiteStore


class HybridRetriever:
    """组合菜名精确匹配、Qdrant 混合检索、本地 BM25 和 rerank。"""

    def __init__(
        self,
        store: SQLiteStore,
        name_index: RecipeNameIndex,
        qdrant: QdrantRecipeIndex,
        embeddings: EmbeddingClient,
        sparse_encoder: SparseBM25Encoder,
        reranker: RerankClient,
    ) -> None:
        """注入检索所需的全部依赖。"""

        self.store = store
        self.name_index = name_index
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.local_bm25 = LocalBM25(store.all_chunks())

    def exact_recipe(self, candidate_name: str) -> Recipe | None:
        """在不做语义替换的前提下解析菜名。"""

        return self.name_index.exact(candidate_name)

    def similar_recipes(self, candidate_name: str, limit: int = 5) -> list[Recipe]:
        """返回供 HITL 人工选择的相似菜谱候选。"""

        return self.name_index.similar(candidate_name, limit=limit)

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        """执行混合检索，并对融合后的候选进行 rerank。"""

        qdrant_hits: list[RetrievalHit] = []
        try:
            dense = self.embeddings.embed_query(query)
            sparse = self.sparse_encoder.encode_query(query)
            qdrant_hits = self.qdrant.hybrid_search(dense, sparse, limit=limit, filters=filters)
        except Exception:
            # 本地开发时 Qdrant 可能暂时不可用；BM25 兜底能保证 CLI 探索和测试仍可用。
            qdrant_hits = []

        bm25_hits = self.local_bm25.search(query, limit=limit)
        merged = _dedupe_hits(qdrant_hits + bm25_hits)
        if not merged:
            return []

        documents = [hit.text for hit in merged]
        reranked = self.reranker.rerank(query, documents)
        final_hits: list[RetrievalHit] = []
        for doc_index, score in reranked[:limit]:
            hit = merged[doc_index]
            final_hits.append(hit.model_copy(update={"score": score, "source": "rerank"}))
        return final_hits


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """对相同 chunk id 只保留最高分命中。"""

    by_id: dict[str, RetrievalHit] = {}
    for hit in hits:
        old = by_id.get(hit.chunk_id)
        if old is None or hit.score > old.score:
            by_id[hit.chunk_id] = hit
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)
