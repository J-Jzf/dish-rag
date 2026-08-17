"""菜谱、chunk、trace 和烹饪状态的数据模型。"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RecipeField(str, Enum):
    """允许引用回 PDF 原文的菜谱字段。"""

    BASIC_INFO = "basic_info"
    INGREDIENTS = "ingredients"
    STEPS = "steps"
    TASTE = "taste"
    AUDIENCE = "audience"
    DIET_TAGS = "diet_tags"
    ALLERGENS = "allergens"
    EQUIPMENT = "equipment"
    SUBSTITUTIONS = "substitutions"
    STORAGE = "storage"


class Citation(BaseModel):
    """从生成文本指回原始菜谱证据的精确引用信息。"""

    recipe_id: str
    recipe_name: str
    page: int
    field: RecipeField | str
    step_no: int | None = None
    chunk_id: str | None = None


class Recipe(BaseModel):
    """从 PDF 中解析出的一道结构化菜谱。"""

    recipe_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    cuisine: str = ""
    category: str = ""
    cooking_method: str = ""
    difficulty: str = ""
    time: str = ""
    serving: str = ""
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    taste: str = ""
    audience: str = ""
    diet_tags: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    substitutions: str = ""
    storage: str = ""
    raw_text: str = ""
    parse_confidence: float = 0.0
    parse_warnings: list[str] = Field(default_factory=list)


class RecipeChunk(BaseModel):
    """由结构化字段或单个步骤派生出的可检索单元。"""

    chunk_id: str
    recipe_id: str
    recipe_name: str
    field: RecipeField | str
    text: str
    page: int
    step_no: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Intent(str, Enum):
    """LangGraph 路由使用的用户高层意图。"""

    RECIPE_LOOKUP = "recipe_lookup"
    FIELD_LOOKUP = "field_lookup"
    COOKING_START = "cooking_start"
    COOKING_NAVIGATION = "cooking_navigation"
    RECOMMENDATION = "recommendation"
    PREFERENCE_UPDATE = "preference_update"
    CHITCHAT = "chitchat"
    UNSAFE_OR_REFUSAL = "unsafe_or_refusal"
    NEED_CLARIFICATION = "need_clarification"


class QueryRewrite(BaseModel):
    """为调试、评测和溯源保留的多阶段 Query 形态。"""

    raw_query: str
    completed_query: str = ""
    rewritten_query: str = ""
    intent: Intent | str = Intent.RECIPE_LOOKUP
    recipe_entities: list[str] = Field(default_factory=list)
    preserved_constraints: list[str] = Field(default_factory=list)
    removed_or_weakened_constraints: list[str] = Field(default_factory=list)
    recommendation_count: int = Field(default=5, ge=1)
    needs_retrieval: bool = True


class IntentAction(BaseModel):
    """LLM 为单轮用户输入规划的一项可顺序执行动作。"""

    intent: Intent | str
    completed_query: str = ""
    recipe_entities: list[str] = Field(default_factory=list)
    recommendation_count: int = Field(default=5, ge=1)
    needs_retrieval: bool = True
    preserved_constraints: list[str] = Field(default_factory=list)


class IntentPlan(BaseModel):
    """一轮输入的有序动作计划；顺序已由 LLM 按依赖关系确定。"""

    actions: list[IntentAction] = Field(default_factory=list)


class UserMemoryItem(BaseModel):
    """用户的一条已归并长期偏好或忌口。"""

    canonical: str
    phrases: list[str] = Field(default_factory=list)
    first_seen_at: str = ""
    last_seen_at: str = ""


class UserMemorySnapshot(BaseModel):
    """供记忆归并与通用提示词使用的用户长期记忆快照。"""

    preferences: list[UserMemoryItem] = Field(default_factory=list)
    restrictions: list[UserMemoryItem] = Field(default_factory=list)


class MemoryOperation(BaseModel):
    """记忆归并模型返回的一项确定性持久化操作。"""

    operation: Literal["add_preference", "merge_preference", "add_restriction", "remove_restriction"]
    canonical: str
    phrase: str


class RetrievalHit(BaseModel):
    """精确匹配、Qdrant、BM25 或 rerank 返回的一条命中结果。"""

    chunk_id: str
    recipe_id: str
    recipe_name: str
    field: str
    text: str
    page: int
    step_no: int | None = None
    score: float
    source: Literal["exact", "qdrant_dense", "qdrant_sparse", "bm25", "fusion", "rerank", "semantic_cache"]
    filters: dict[str, Any] = Field(default_factory=dict)


class EvidenceJudgeResult(BaseModel): # 是一个 Pydantic 数据模型，用来保存 Evidence Judge 的输出。
    """判断证据是否足以支持回答的量化结果。"""

    relevant: bool # 证据是否相关
    sufficient: bool # 证据是否足够回答
    confidence: float # 置信度，通常是 0 到 1
    reasons: list[str] = Field(default_factory=list) # 判断理由列表。默认是空列表。
    missing: list[str] = Field(default_factory=list) # 缺失信息列表。比如“缺少步骤”“缺少原材料”。


class ActionResult(BaseModel):
    """一项 intent action 执行完成后的可聚合结果。"""

    action_index: int
    intent: Intent | str
    query: str = ""
    recommendation_count: int = 0
    hits: list[RetrievalHit] = Field(default_factory=list)
    evidence_judge: EvidenceJudgeResult | None = None
    evidence_retry_count: int = 0
    answer_hint: str = ""
    citations: list[Citation] = Field(default_factory=list)


class CookingState(BaseModel):
    """每个 thread 的烹饪状态；一个 thread 最多跟踪一道正在做的菜。"""

    active_recipe_id: str | None = None
    active_recipe_name: str | None = None
    current_step_no: int = 0
    total_steps: int = 0
    last_action: str = "" # 上一次状态动作，可能是start、next、prev、repeat、next_from_matched_step、repeat_matched_step等


class TurnTrace(BaseModel):
    """每一轮对话输出的可观测信息。"""

    raw_query: str
    parsed_intent: str = ""
    completed_query: str = ""
    rewritten_query: str = ""
    recipe_entities: list[str] = Field(default_factory=list)
    recommendation_count: int = 0
    evidence_retry_count: int = 0
    action_results: list[ActionResult] = Field(default_factory=list)
    qdrant_hits: list[RetrievalHit] = Field(default_factory=list)
    evidence_judge: EvidenceJudgeResult | None = None
    final_citations: list[Citation] = Field(default_factory=list)
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
