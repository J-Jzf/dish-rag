"""LangGraph 状态类型。"""

from typing import Any, TypedDict

from dish_rag.models import (
    ActionResult,
    CookingState,
    EvidenceJudgeResult,
    IntentPlan,
    QueryRewrite,
    RetrievalHit,
    TurnTrace,
    UserMemorySnapshot,
)

# 一轮对话在图里流动时携带的“共享数据包”
class DishAgentState(TypedDict, total=False): 
    """在图节点之间传递的可变状态。"""
    # TypedDict 表示：这是一个“带类型标注的字典”（实际运行时它还是普通 dict）。
    # total=False 表示：这些字段不是每个时候都必须存在。

    thread_id: str # 当前对话线程 ID，用于 LangGraph checkpoint 区分不同对话。
    user_id: str
    user_query: str
    query_rewrite: QueryRewrite # Query 处理结果，包括了原始问题、补全问题、重写问题、意图、实体、约束等。
    intent_plan: IntentPlan # 当前轮由 LLM 识别出的有序多意图动作计划。
    current_action_index: int # 当前正在执行的动作下标。
    action_results: list[ActionResult] # 已执行动作的独立结果，供最终合并回答。
    hits: list[RetrievalHit] # 检索命中结果列表。由 retrieve 节点生成。
    evidence_judge: EvidenceJudgeResult
    evidence_retry_count: int # Evidence Judge 触发的重检索次数，每轮最多 1 次。
    answer: str # 最终回答文本。由 answer 节点生成。
    citations: list[dict[str, Any]] # 最终引用信息。
    cooking_state: CookingState # 当前烹饪状态。由 checkpoint 保存，同一个 thread_id 后续会继续用。
    trace: TurnTrace # 本轮可观测 trace。记录调试信息。
    needs_hitl: bool # 是否需要人工介入
    hitl_candidates: list[dict[str, Any]] # HITL 候选列表
    selected_recipe_id: str | None # 用户从 HITL 候选里选择的菜谱 ID。如果用户没选，就是 None。
    memory: dict[str, str] # 长期记忆（例如过敏原、口感偏好），由 start_trace 从 SQLite 里加载。
    user_memory: UserMemorySnapshot
