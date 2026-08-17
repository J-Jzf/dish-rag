import json
from pathlib import Path

from dish_rag.eval.cache import run_cache_eval
from dish_rag.models import RetrievalHit


def _hit(recipe_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{recipe_id}:steps",
        recipe_id=recipe_id,
        recipe_name=f"菜谱{recipe_id}",
        field="steps",
        text="步骤说明",
        page=1,
        score=0.9,
        source="rerank",
    )


class _FakeRetriever:
    def __init__(self, by_query: dict[str, list[RetrievalHit]]):
        self.by_query = by_query
        self.calls: list[str] = []

    def retrieve(self, query, limit=5, filters=None, dense_vector=None):
        self.calls.append(query)
        return self.by_query[query][:limit]


class _FakeSemanticCache:
    def __init__(self, reused_hits: list[RetrievalHit]):
        self.reused_hits = reused_hits
        self.stored = False

    def embed_query(self, query):
        return [0.1, 0.2]

    def lookup_embedding(self, **kwargs):
        return self.reused_hits if self.stored else None

    def store_embedding(self, **kwargs):
        self.stored = True


def _write_pairs(tmp_path: Path, pairs: list[dict]) -> Path:
    path = tmp_path / "cache_pairs.jsonl"
    path.write_text("\n".join(json.dumps(pair, ensure_ascii=False) for pair in pairs), encoding="utf-8")
    return path


def test_cache_eval_reports_hit_timing_and_correct_reuse(tmp_path):
    eval_file = _write_pairs(
        tmp_path,
        [
            {
                "query_a": "宫保鸡丁怎么做？",
                "query_b": "教我做宫保鸡丁。",
                "intent": "recipe_lookup",
                "expected_recipe_ids": ["001"],
            }
        ],
    )
    retriever = _FakeRetriever(
        {
            "宫保鸡丁怎么做？": [_hit("001")],
            "教我做宫保鸡丁。": [_hit("001")],
        }
    )

    report = run_cache_eval(
        retriever,
        _FakeSemanticCache([_hit("001")]),
        eval_file,
        limit=5,
    )

    assert report["pairs"] == 1
    assert report["cache_hit_rate"] == 1.0
    assert report["correct_reuse_rate"] == 1.0
    assert report["incorrect_reuse_rate"] == 0.0
    assert report["cache_hit_average_ms"] >= 0.0
    assert report["hybrid_average_ms"] >= 0.0
    assert retriever.calls == ["宫保鸡丁怎么做？", "教我做宫保鸡丁。"]


def test_cache_eval_only_counts_wrong_reuse_when_fresh_hybrid_would_be_correct(tmp_path):
    eval_file = _write_pairs(
        tmp_path,
        [
            {
                "query_a": "宫保鸡丁怎么做？",
                "query_b": "教我做宫保鸡丁。",
                "intent": "recipe_lookup",
                "expected_recipe_ids": ["001"],
            }
        ],
    )
    retriever = _FakeRetriever(
        {
            "宫保鸡丁怎么做？": [_hit("001")],
            "教我做宫保鸡丁。": [_hit("001")],
        }
    )

    report = run_cache_eval(
        retriever,
        _FakeSemanticCache([_hit("002")]),
        eval_file,
        limit=5,
    )

    assert report["correct_reuse_rate"] == 0.0
    assert report["incorrect_reuse_rate"] == 1.0
    assert report["baseline_failures"] == 0

