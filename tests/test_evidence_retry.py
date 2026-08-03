import pytest

from dish_rag.agent.nodes import route_after_judge
from dish_rag.models import EvidenceJudgeResult, QueryRewrite


def _state(*, relevant=True, sufficient=True, confidence=0.8, retry_count=0):
    return {
        "query_rewrite": QueryRewrite(raw_query="测试", needs_retrieval=True),
        "evidence_judge": EvidenceJudgeResult(
            relevant=relevant,
            sufficient=sufficient,
            confidence=confidence,
        ),
        "evidence_retry_count": retry_count,
    }


@pytest.mark.parametrize(
    "judge_values",
    [
        {"relevant": False},
        {"sufficient": False},
        {"confidence": 0.549},
    ],
)
def test_evidence_judge_triggers_one_retry(judge_values):
    assert route_after_judge(_state(**judge_values)) == "retry_evidence"


def test_evidence_judge_does_not_retry_after_one_retry():
    state = _state(relevant=False, sufficient=False, confidence=0.1, retry_count=1)

    assert route_after_judge(state) == "update_cooking_state"


def test_confidence_boundary_055_is_accepted():
    assert route_after_judge(_state(confidence=0.55)) == "update_cooking_state"


def test_non_retrieval_turn_does_not_trigger_evidence_retry():
    state = _state(sufficient=False)
    state["query_rewrite"] = QueryRewrite(raw_query="你好", needs_retrieval=False)

    assert route_after_judge(state) == "update_cooking_state"
