"""LangGraph 状态类型。"""

from typing import Any, TypedDict

from dish_rag.models import CookingState, EvidenceJudgeResult, QueryRewrite, RetrievalHit, TurnTrace


class DishAgentState(TypedDict, total=False):
    """在图节点之间传递的可变状态。"""

    thread_id: str
    user_id: str
    user_query: str
    query_rewrite: QueryRewrite
    hits: list[RetrievalHit]
    evidence_judge: EvidenceJudgeResult
    answer: str
    citations: list[dict[str, Any]]
    cooking_state: CookingState
    trace: TurnTrace
    needs_hitl: bool
    hitl_candidates: list[dict[str, Any]]
    selected_recipe_id: str | None
    memory: dict[str, str]
