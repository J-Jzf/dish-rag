"""Qdrant 稠密向量和稀疏索引适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dish_rag.models import RecipeChunk, RetrievalHit


class QdrantRecipeIndex:
    """面向菜谱 chunk 的 Qdrant 稠密+稀疏混合索引。"""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        local_path: Path | None = None,
    ) -> None:
        """创建 Qdrant 客户端封装。"""

        from qdrant_client import QdrantClient

        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        else:
            # 嵌入式 Qdrant 适合本地学习和小型菜谱索引，
            # 因为不需要额外启动独立服务进程。
            assert local_path is not None
            local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(local_path))

    def recreate_collection(self, dense_size: int) -> None:
        """创建包含命名稠密向量和稀疏向量的 collection。"""

        from qdrant_client import models

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    def upsert_chunks(
        self,
        chunks: Iterable[RecipeChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[object],
    ) -> None:
        """上传 chunk原文（pyload.text中）、稠密 embedding 和 BM25 稀疏向量。"""

        from qdrant_client import models

        points: list[models.PointStruct] = [] # 每个 Qdrant point 对应一个 chunk（相当于SQL的每一行）
        for index, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=_stable_point_id(chunk.chunk_id),
                    vector={
                        "dense": dense_vectors[index], # chunk 文本的稠密向量，用于语义相似度检索。
                        "bm25": _to_sparse_vector(sparse_vectors[index]), # chunk 文本的 BM25 稀疏向量，用于关键词检索。
                    },
                    # `payload.text`：chunk 原文，用于命中后展示和溯源。
                    # 业务元数据，用于过滤和引用。
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "recipe_id": chunk.recipe_id,
                        "recipe_name": chunk.recipe_name,
                        "field": str(chunk.field),
                        "page": chunk.page,
                        "step_no": chunk.step_no,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: object,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        """用稠密和稀疏预检索，再通过倒数排名融合搜索 Qdrant。"""

        from qdrant_client import models

        qdrant_filter = _build_filter(filters or {})
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=qdrant_filter,
                    limit=limit * 3,
                ),
                models.Prefetch(
                    query=_to_sparse_vector(sparse_vector),
                    using="bm25",
                    filter=qdrant_filter,
                    limit=limit * 3,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        hits: list[RetrievalHit] = []
        for point in points:
            payload = point.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=payload.get("chunk_id", ""),
                    recipe_id=payload.get("recipe_id", ""),
                    recipe_name=payload.get("recipe_name", ""),
                    field=payload.get("field", ""),
                    text=payload.get("text", ""),
                    page=int(payload.get("page", 0)),
                    score=float(point.score or 0.0),
                    source="fusion",
                    filters=filters or {},
                )
            )
        return hits


def _stable_point_id(chunk_id: str) -> int:
    """把 chunk id 转成稳定的无符号整数 point id。"""

    import hashlib

    digest = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _to_sparse_vector(vector: object):
    """把 FastEmbed 稀疏输出转成 Qdrant 的 SparseVector 模型。"""

    from qdrant_client import models

    if isinstance(vector, models.SparseVector):
        return vector
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is None or values is None:
        return vector
    return models.SparseVector(indices=list(indices), values=list(values))


def _build_filter(filters: dict[str, object]):
    """根据等值元数据过滤条件构建 Qdrant filter。"""

    if not filters:
        return None

    from qdrant_client import models

    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in filters.items()
        if value not in (None, "")
    ]
    return models.Filter(must=conditions) if conditions else None
