"""本地 BM25 兜底检索器和评测检索器。"""

from rank_bm25 import BM25Okapi

from dish_rag.models import RecipeChunk, RetrievalHit


class LocalBM25:
    """基于 SQLite chunk 的简单 BM25 索引。"""

    def __init__(self, chunks: list[RecipeChunk]) -> None:
        """构建本地 BM25 索引。"""

        self.chunks = chunks
        self.tokenized = [_tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.tokenized) if chunks else None

    def search(self, query: str, limit: int = 10) -> list[RetrievalHit]:
        """用 BM25 搜索本地 chunk。"""

        if not self.index:
            return []
        scores = self.index.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
        hits: list[RetrievalHit] = []
        for index, score in ranked:
            chunk = self.chunks[index]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    recipe_id=chunk.recipe_id,
                    recipe_name=chunk.recipe_name,
                    field=str(chunk.field),
                    text=chunk.text,
                    page=chunk.page,
                    step_no=chunk.step_no,
                    score=float(score),
                    source="bm25",
                )
            )
        return hits


def _tokenize(text: str) -> list[str]:
    """用字符 bigram 方式为中文文本分词。"""

    compact = "".join(text.lower().split())
    if len(compact) <= 2:
        return [compact] if compact else []
    # 对这个小型中文菜谱库来说，字符 bigram 足够可用，也能减少演示依赖。
    # 生产系统可以换成 jieba 或领域分词器，不需要改变检索器 API。
    return [compact[index : index + 2] for index in range(len(compact) - 1)]
