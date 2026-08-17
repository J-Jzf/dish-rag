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
        dense_vector: list[float] | None = None,
    ) -> list[RetrievalHit]:
        """执行混合检索，并对融合后的候选（去重后）进行 rerank。"""

        qdrant_hits: list[RetrievalHit] = [] # 先准备空的 Qdrant 命中列表
        try:
            dense = dense_vector or self.embeddings.embed_query(query)
            sparse = self.sparse_encoder.encode_query(query)
            qdrant_hits = self.qdrant.hybrid_search(dense, sparse, limit=limit, filters=filters) # 不用显式传入 Qdrant 中 chunk 的 embedding。因为 chunk 的 embedding 已经在 ingest 阶段写进 Qdrant 了。
        except Exception:
            # 本地开发时 Qdrant 可能暂时不可用；BM25 兜底能保证 CLI 探索和测试仍可用。
            qdrant_hits = []

        bm25_hits = self.local_bm25.search(query, limit=limit, filters=filters) # 为了不让整个程序崩溃，后面继续用本地 BM25 兜底；有菜名过滤时，本地 BM25 也必须限制在同一道菜内。
        merged = _dedupe_hits(qdrant_hits + bm25_hits) # 把 Qdrant 结果和本地 BM25 结果合并，然后去重。
        if not merged:
            return []

        documents = [hit.text for hit in merged] # 取出去重后的候选 chunk 文本，供 rerank 使用。
        reranked = self.reranker.rerank(query, documents) # 按openai调用 rerank 模型，输入（query+候选chunk文本列表），输出（候选chunk文本的索引+分数）列表
        # 让一个专门的排序模型重新判断“query 和每个 chunk 到底有多匹配”。常见的rerank做法是cross-encoder / reranker，模型直接同时读取 query + chunk，输出一个相关性分数。
        final_hits: list[RetrievalHit] = []
        for doc_index, score in reranked[:limit]: # 只取 rerank 后前 limit 条
            hit = merged[doc_index]
            final_hits.append(hit.model_copy(update={"score": score, "source": "rerank"})) # 更新score，改成改成 rerank 分数。
        return final_hits


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """对相同 chunk id 只保留最高分命中。负责去重。"""

    by_id: dict[str, RetrievalHit] = {}
    for hit in hits:
        old = by_id.get(hit.chunk_id)
        if old is None or hit.score > old.score:
            by_id[hit.chunk_id] = hit
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)
