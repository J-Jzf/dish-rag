"""语义缓存离线评测：比较一对同义 Query 的冷检索与缓存复用。"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dish_rag.models import RetrievalHit, UserMemorySnapshot
from dish_rag.retrieval.hybrid import HybridRetriever
from dish_rag.retrieval.semantic_cache import SemanticCache


def run_cache_eval(
    retriever: HybridRetriever,
    cache: SemanticCache,
    eval_file: Path,
    limit: int = 5,
) -> dict[str, object]:
    """评测缓存命中、复用正确性和跳过 Hybrid/rerank 后的耗时节省。"""

    pairs = [
        json.loads(line)
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pairs:
        return _empty_report()

    scope = f"eval-{uuid4().hex}"
    hybrid_times: list[float] = []
    cache_hit_times: list[float] = []
    cache_hits = 0
    correct_reuses = 0
    incorrect_reuses = 0
    baseline_failures = 0
    failures: list[dict[str, object]] = []

    for pair in pairs:
        query_a = str(pair["query_a"])
        query_b = str(pair["query_b"])
        action_type = str(pair.get("intent", "recipe_lookup"))
        expected_recipe_ids = [str(value) for value in pair.get("expected_recipe_ids", [])]
        expected_fields = [str(value) for value in pair.get("expected_fields", [])]
        filters = {"recipe_id": expected_recipe_ids[0]} if len(expected_recipe_ids) == 1 else {}
        memory = UserMemorySnapshot()

        # Query A 固定为冷启动：写入隔离的本轮评测 scope，为 Query B 预热。
        vector_a = cache.embed_query(query_a)
        cache.lookup_embedding(
            vector=vector_a,
            action_type=action_type,
            filters=filters,
            user_memory=memory,
            limit=limit,
            cache_scope=scope,
        )
        started = perf_counter()
        cold_hits = retriever.retrieve(
            query_a,
            limit=limit,
            filters=filters,
            dense_vector=vector_a,
        )
        hybrid_times.append(perf_counter() - started)
        cache.store_embedding(
            query=query_a,
            vector=vector_a,
            action_type=action_type,
            filters=filters,
            user_memory=memory,
            limit=limit,
            hits=cold_hits,
            cache_scope=scope,
        )

        # Query B 是实际缓存路径；命中后不会执行 Hybrid 或 rerank。
        vector_b = cache.embed_query(query_b)
        started = perf_counter()
        cached_hits = cache.lookup_embedding(
            vector=vector_b,
            action_type=action_type,
            filters=filters,
            user_memory=memory,
            limit=limit,
            cache_scope=scope,
        )
        cache_elapsed = perf_counter() - started

        # 额外执行一次 Query B 的原始 Hybrid，仅作为离线正确性和性能基线。
        started = perf_counter()
        baseline_hits = retriever.retrieve(
            query_b,
            limit=limit,
            filters=filters,
            dense_vector=vector_b,
        )
        baseline_elapsed = perf_counter() - started
        hybrid_times.append(baseline_elapsed)
        baseline_correct = _matches_expected(baseline_hits, expected_recipe_ids, expected_fields)
        if not baseline_correct:
            baseline_failures += 1

        if cached_hits is None:
            failures.append(
                {
                    "query_a": query_a,
                    "query_b": query_b,
                    "kind": "cache_miss",
                }
            )
            continue

        cache_hits += 1
        cache_hit_times.append(cache_elapsed)
        cache_correct = _matches_expected(cached_hits, expected_recipe_ids, expected_fields)
        if cache_correct:
            correct_reuses += 1
            continue
        if baseline_correct:
            incorrect_reuses += 1
            kind = "incorrect_reuse"
        else:
            kind = "baseline_failure"
        failures.append(
            {
                "query_a": query_a,
                "query_b": query_b,
                "kind": kind,
                "expected_recipe_ids": expected_recipe_ids,
                "expected_fields": expected_fields,
                "cached_recipe_ids": [hit.recipe_id for hit in cached_hits],
                "baseline_recipe_ids": [hit.recipe_id for hit in baseline_hits],
            }
        )

    pair_count = len(pairs)
    hybrid_average_ms = _average_ms(hybrid_times)
    cache_hit_average_ms = _average_ms(cache_hit_times)
    time_saved_ms = max(0.0, hybrid_average_ms - cache_hit_average_ms)
    time_saved_percent = (time_saved_ms / hybrid_average_ms * 100) if hybrid_average_ms else 0.0
    return {
        "pairs": pair_count,
        "cache_hit_rate": cache_hits / pair_count,
        "correct_reuse_rate": correct_reuses / cache_hits if cache_hits else 0.0,
        "incorrect_reuse_rate": incorrect_reuses / cache_hits if cache_hits else 0.0,
        "baseline_failures": baseline_failures,
        "hybrid_average_ms": hybrid_average_ms,
        "cache_hit_average_ms": cache_hit_average_ms,
        "time_saved_ms": time_saved_ms,
        "time_saved_percent": time_saved_percent,
        "failures": failures,
    }


def _matches_expected(
    hits: list[RetrievalHit],
    expected_recipe_ids: list[str],
    expected_fields: list[str],
) -> bool:
    expected_recipes = set(expected_recipe_ids)
    expected_field_set = {_normalize_field(value) for value in expected_fields}
    for hit in hits:
        if hit.recipe_id not in expected_recipes:
            continue
        if not expected_field_set or _normalize_field(str(hit.field)) in expected_field_set:
            return True
    return False


def _normalize_field(value: str) -> str:
    return value.lower().replace("recipefield.", "")


def _average_ms(values: list[float]) -> float:
    return sum(values) / len(values) * 1000 if values else 0.0


def _empty_report() -> dict[str, object]:
    return {
        "pairs": 0,
        "cache_hit_rate": 0.0,
        "correct_reuse_rate": 0.0,
        "incorrect_reuse_rate": 0.0,
        "baseline_failures": 0,
        "hybrid_average_ms": 0.0,
        "cache_hit_average_ms": 0.0,
        "time_saved_ms": 0.0,
        "time_saved_percent": 0.0,
        "failures": [],
    }
