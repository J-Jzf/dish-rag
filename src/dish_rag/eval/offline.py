"""离线检索评测工具。"""

import json
from pathlib import Path

from dish_rag.retrieval.hybrid import HybridRetriever


def run_retrieval_eval(
    retriever: HybridRetriever,
    eval_file: Path,
    limit: int = 5,
) -> dict[str, object]:
    """基于 JSONL 测试集计算 recall@k 和 precision@k。"""

    cases = [
        json.loads(line)
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recipe_cases = [case for case in cases if case.get("expected_recipe_ids")]
    if not recipe_cases:
        return {"cases": 0, "recall_at_k": 0.0, "precision_at_k": 0.0, "failures": []}

    recall_hits = 0
    precision_sum = 0.0
    failures: list[dict[str, object]] = []
    for case in recipe_cases:
        expected = set(case["expected_recipe_ids"])
        hits = retriever.retrieve(case["query"], limit=limit)
        returned = [hit.recipe_id for hit in hits]
        returned_set = set(returned)
        first_expected_rank = next(
            (index for index, recipe_id in enumerate(returned, start=1) if recipe_id in expected),
            None,
        )
        if first_expected_rank is not None:
            recall_hits += 1
        else:
            failures.append(
                {
                    "query": case["query"],
                    "expected_recipe_ids": case["expected_recipe_ids"],
                    "actual_top_k": [
                        {
                            "recipe_id": hit.recipe_id,
                            "recipe_name": hit.recipe_name,
                            "score": hit.score,
                        }
                        for hit in hits
                    ],
                    "rank": first_expected_rank,
                }
            )
        precision_sum += len(expected & returned_set) / max(1, len(returned))

    return {
        "cases": len(recipe_cases),
        "recall_at_k": recall_hits / len(recipe_cases),
        "precision_at_k": precision_sum / len(recipe_cases),
        "failures": failures,
    }
