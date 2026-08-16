import json
from pathlib import Path

from rich.console import Console

from dish_rag.cli import render_retrieval_failures
from dish_rag.eval.offline import run_retrieval_eval
from dish_rag.models import RetrievalHit


class _FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits
        self.limits: list[int] = []

    def retrieve(self, query: str, limit: int):
        self.limits.append(limit)
        return self.hits[:limit]


def _hit(recipe_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{recipe_id}:steps",
        recipe_id=recipe_id,
        recipe_name=f"菜谱{recipe_id}",
        field="steps",
        text=f"菜谱{recipe_id}的步骤",
        page=1,
        score=score,
        source="rerank",
    )


def _write_cases(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")
    return path


def test_retrieval_eval_uses_top_five_by_default_and_accepts_rank_three_match(tmp_path):
    retriever = _FakeRetriever(
        [
            _hit("002", 0.9),
            _hit("003", 0.8),
            _hit("001", 0.7),
            _hit("004", 0.6),
            _hit("005", 0.5),
            _hit("006", 0.4),
        ]
    )
    eval_file = _write_cases(tmp_path, [{"query": "宫保鸡丁怎么做", "expected_recipe_ids": ["001"]}])

    report = run_retrieval_eval(retriever, eval_file)

    assert retriever.limits == [5]
    assert report["recall_at_k"] == 1.0
    assert report["failures"] == []


def test_retrieval_eval_reports_query_and_top_k_for_missing_recipe(tmp_path):
    retriever = _FakeRetriever([_hit("002", 0.9), _hit("003", 0.8)])
    eval_file = _write_cases(tmp_path, [{"query": "目标菜谱未命中", "expected_recipe_ids": ["001"]}])

    report = run_retrieval_eval(retriever, eval_file, limit=5)

    assert report["failures"] == [
        {
            "query": "目标菜谱未命中",
            "expected_recipe_ids": ["001"],
            "actual_top_k": [
                {"recipe_id": "002", "recipe_name": "菜谱002", "score": 0.9},
                {"recipe_id": "003", "recipe_name": "菜谱003", "score": 0.8},
            ],
            "rank": None,
        }
    ]


def test_render_retrieval_failures_includes_query_expected_top_k_and_rank():
    table = render_retrieval_failures(
        [
            {
                "query": "目标菜谱未命中",
                "expected_recipe_ids": ["001"],
                "actual_top_k": [
                    {"recipe_id": "002", "recipe_name": "麻婆豆腐", "score": 0.9},
                ],
                "rank": None,
            }
        ],
        limit=5,
    )
    console = Console(record=True, width=160)
    console.print(table)
    rendered = console.export_text()

    assert "FAIL@5" in rendered
    assert "目标菜谱未命中" in rendered
    assert "001" in rendered
    assert "002 麻婆豆腐 (0.900)" in rendered
    assert "未命中" in rendered
