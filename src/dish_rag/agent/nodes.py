"""LangGraph 节点实现。"""

from __future__ import annotations

import json
from typing import Any

from langgraph.types import interrupt

from dish_rag.agent import prompts
from dish_rag.agent.state import DishAgentState
from dish_rag.llm.opai import ChatClient
from dish_rag.models import (
    Citation,
    CookingState,
    EvidenceJudgeResult,
    Intent,
    QueryRewrite,
    RetrievalHit,
    TurnTrace,
)
from dish_rag.retrieval.hybrid import HybridRetriever
from dish_rag.safety import check_refusal, detect_constraint_loss, extract_user_constraints
from dish_rag.storage.sqlite_store import SQLiteStore


class AgentNodes:
    """图节点及其共享依赖的容器。"""

    def __init__(self, chat: ChatClient, retriever: HybridRetriever, store: SQLiteStore) -> None:
        """绑定模型、检索器和存储依赖。"""

        self.chat = chat
        self.retriever = retriever
        self.store = store

    def start_trace(self, state: DishAgentState) -> DishAgentState:
        """初始化 trace，并加载长期记忆。"""

        cooking_state = state.get("cooking_state") or CookingState()
        memory = self.store.load_memory(state.get("user_id", "default"))
        trace = TurnTrace(
            raw_query=state["user_query"],
            state_before=cooking_state.model_dump(),
        )
        return {**state, "cooking_state": cooking_state, "memory": memory, "trace": trace}

    def classify_intent(self, state: DishAgentState) -> DishAgentState:
        """识别意图，并构造补全后的 Query。"""

        safety = check_refusal(state["user_query"])
        if not safety.allowed:
            rewrite = QueryRewrite(
                raw_query=state["user_query"],
                intent=Intent.UNSAFE_OR_REFUSAL,
                needs_retrieval=False,
            )
            trace = _update_trace(state["trace"], parsed_intent=str(rewrite.intent), notes=[safety.reason])
            return {**state, "query_rewrite": rewrite, "trace": trace}

        payload = self.chat.complete_json(
            prompts.INTENT_SYSTEM,
            prompts.INTENT_USER.format(
                query=state["user_query"],
                cooking_state=state["cooking_state"].model_dump(),
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
            ),
        )
        rewrite = QueryRewrite(
            raw_query=state["user_query"],
            completed_query=payload.get("completed_query") or state["user_query"],
            intent=payload.get("intent", Intent.RECIPE_LOOKUP),
            recipe_entities=payload.get("recipe_entities", []),
            needs_retrieval=bool(payload.get("needs_retrieval", True)),
            preserved_constraints=payload.get("preserved_constraints", []),
        )
        trace = _update_trace(
            state["trace"],
            parsed_intent=str(rewrite.intent),
            completed_query=rewrite.completed_query,
        )
        return {**state, "query_rewrite": rewrite, "trace": trace}

    def rewrite_query(self, state: DishAgentState) -> DishAgentState:
        """在保留用户限制条件的前提下重写 Query。"""

        rewrite = state["query_rewrite"]
        if not rewrite.needs_retrieval:
            return state

        constraints = extract_user_constraints(rewrite.raw_query)
        payload = self.chat.complete_json(
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER.format(
                raw_query=rewrite.raw_query,
                completed_query=rewrite.completed_query,
                recipe_entities=rewrite.recipe_entities,
                constraints=constraints,
            ),
        )
        rewritten_query = payload.get("rewritten_query") or rewrite.completed_query
        missing_constraints = detect_constraint_loss(constraints, rewritten_query)
        rewrite = rewrite.model_copy(
            update={
                "rewritten_query": rewritten_query,
                "preserved_constraints": payload.get("preserved_constraints", constraints),
                "removed_or_weakened_constraints": missing_constraints
                or payload.get("removed_or_weakened_constraints", []),
            }
        )
        if rewrite.removed_or_weakened_constraints:
            rewrite = rewrite.model_copy(update={"intent": Intent.NEED_CLARIFICATION})
        trace = _update_trace(state["trace"], rewritten_query=rewrite.rewritten_query)
        return {**state, "query_rewrite": rewrite, "trace": trace}

    def retrieve(self, state: DishAgentState) -> DishAgentState:
        """执行菜名精确匹配和混合检索。"""

        rewrite = state["query_rewrite"]
        if not rewrite.needs_retrieval:
            return {**state, "hits": []}

        # 精确菜名解析不做语义替换。
        for entity in rewrite.recipe_entities:
            exact = self.retriever.exact_recipe(entity)
            if exact:
                filters = {"recipe_id": exact.recipe_id}
                hits = self.retriever.retrieve(rewrite.rewritten_query, filters=filters)
                trace = _update_trace(state["trace"], qdrant_hits=hits)
                return {**state, "hits": hits, "trace": trace}

        # 如果用户明确说了菜名但没有精确别名，暂停图执行并进入 HITL。
        if rewrite.recipe_entities:
            candidates = self.retriever.similar_recipes(rewrite.recipe_entities[0])
            if candidates:
                hitl_candidates = [
                    {"recipe_id": recipe.recipe_id, "name": recipe.name, "page": recipe.page_start}
                    for recipe in candidates
                ]
                return {
                    **state,
                    "needs_hitl": True,
                    "hitl_candidates": hitl_candidates,
                    "hits": [],
                }

        hits = self.retriever.retrieve(rewrite.rewritten_query)
        trace = _update_trace(state["trace"], qdrant_hits=hits)
        return {**state, "hits": hits, "trace": trace}

    def hitl_recipe_choice(self, state: DishAgentState) -> DishAgentState:
        """暂停图执行，让用户选择是否使用相似菜。"""

        selected = interrupt(
            {
                "message": "菜谱库中没有精确菜名，请选择是否使用相似菜。",
                "candidates": state.get("hitl_candidates", []),
                "allow_none": True,
            }
        )
        selected_recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        return {**state, "selected_recipe_id": selected_recipe_id, "needs_hitl": False}

    def retrieve_selected_recipe(self, state: DishAgentState) -> DishAgentState:
        """用户选择 HITL 候选后，再按该菜谱过滤检索。"""

        recipe_id = state.get("selected_recipe_id")
        if not recipe_id:
            return {**state, "hits": []}
        hits = self.retriever.retrieve(
            state["query_rewrite"].rewritten_query,
            filters={"recipe_id": recipe_id},
        )
        trace = _update_trace(state["trace"], qdrant_hits=hits)
        return {**state, "hits": hits, "trace": trace}

    def judge_evidence(self, state: DishAgentState) -> DishAgentState:
        """判断检索证据是否足够回答。"""

        if not state.get("hits"):
            judge = EvidenceJudgeResult(
                relevant=False,
                sufficient=False,
                confidence=0.0,
                missing=["没有检索到可引用证据"],
            )
            trace = _update_trace(state["trace"], evidence_judge=judge)
            return {**state, "evidence_judge": judge, "trace": trace}

        payload = self.chat.complete_json(
            prompts.JUDGE_SYSTEM,
            prompts.JUDGE_USER.format(
                query=state["query_rewrite"].rewritten_query,
                evidence=_format_hits(state["hits"]),
            ),
        )
        judge = EvidenceJudgeResult(
            relevant=bool(payload.get("relevant", False)),
            sufficient=bool(payload.get("sufficient", False)),
            confidence=float(payload.get("confidence", 0.0)),
            reasons=payload.get("reasons", []),
            missing=payload.get("missing", []),
        )
        trace = _update_trace(state["trace"], evidence_judge=judge)
        return {**state, "evidence_judge": judge, "trace": trace}

    def update_cooking_state(self, state: DishAgentState) -> DishAgentState:
        """更新当前 thread 的烹饪进度。"""

        cooking_state = state["cooking_state"].model_copy()
        intent = str(state["query_rewrite"].intent)
        hits = state.get("hits", [])
        if intent == Intent.COOKING_START and hits:
            recipe = self.store.get_recipe(hits[0].recipe_id)
            if recipe:
                cooking_state.active_recipe_id = recipe.recipe_id
                cooking_state.active_recipe_name = recipe.name
                cooking_state.current_step_no = 1
                cooking_state.total_steps = len(recipe.steps)
                cooking_state.last_action = "start"
        elif intent == Intent.COOKING_NAVIGATION:
            cooking_state = _navigate_steps(cooking_state, state["user_query"])
        trace = _update_trace(state["trace"], state_after=cooking_state.model_dump())
        return {**state, "cooking_state": cooking_state, "trace": trace}

    def answer(self, state: DishAgentState) -> DishAgentState:
        """生成最终可溯源回答。"""

        rewrite = state["query_rewrite"]
        if rewrite.intent == Intent.UNSAFE_OR_REFUSAL:
            answer = "这个请求涉及高风险或医疗化内容，我不能按这个方向提供做法。"
            return {**state, "answer": answer}
        if rewrite.intent == Intent.NEED_CLARIFICATION:
            answer = "我需要先确认限制条件：你刚才的约束不能被改写或省略，请再说明一次你要保留的禁忌或口味要求。"
            return {**state, "answer": answer}
        if not state.get("hits"):
            answer = "菜谱库里没有找到可追溯证据。我不会自动替换成相似菜；你可以选择一个相似菜或换个菜名。"
            return {**state, "answer": answer}

        answer = self.chat.complete_text(
            prompts.ANSWER_SYSTEM,
            prompts.ANSWER_USER.format(
                query=rewrite.rewritten_query,
                evidence=_format_hits(state["hits"]),
                cooking_state=state["cooking_state"].model_dump(),
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
            ),
        )
        citations = [_citation_from_hit(hit).model_dump() for hit in state["hits"][:4]]
        trace = _update_trace(state["trace"], final_citations=[Citation(**item) for item in citations])
        return {**state, "answer": answer, "citations": citations, "trace": trace}

    def persist_trace(self, state: DishAgentState) -> DishAgentState:
        """把 trace 保存到 SQLite，便于调试和审计。"""

        trace = state.get("trace")
        if trace:
            self.store.append_trace(state.get("thread_id", "default"), trace.model_dump(mode="json"))
        return state


def route_after_retrieve(state: DishAgentState) -> str:
    """精确菜名失败但有候选时，路由到 HITL。"""

    return "hitl_recipe_choice" if state.get("needs_hitl") else "judge_evidence"


def _navigate_steps(cooking_state: CookingState, query: str) -> CookingState:
    """把自然语言导航指令应用到烹饪状态机。"""

    next_state = cooking_state.model_copy()
    if not next_state.active_recipe_id:
        return next_state
    if "上一步" in query or "回到" in query:
        next_state.current_step_no = max(1, next_state.current_step_no - 1)
        next_state.last_action = "previous"
    elif "重复" in query or "再说" in query:
        next_state.last_action = "repeat"
    else:
        next_state.current_step_no = min(next_state.total_steps, next_state.current_step_no + 1)
        next_state.last_action = "next"
    return next_state


def _format_hits(hits: list[RetrievalHit]) -> str:
    """把检索命中渲染成给模型看的紧凑证据文本。"""

    lines: list[str] = []
    for hit in hits:
        lines.append(
            f"[PDF p.{hit.page}｜{hit.recipe_id} {hit.recipe_name}｜{hit.field}｜score={hit.score:.3f}] {hit.text}"
        )
    return "\n".join(lines)


def _citation_from_hit(hit: RetrievalHit) -> Citation:
    """从检索命中构建引用信息。"""

    return Citation(
        recipe_id=hit.recipe_id,
        recipe_name=hit.recipe_name,
        page=hit.page,
        field=hit.field,
        step_no=None,
        chunk_id=hit.chunk_id,
    )


def _update_trace(trace: TurnTrace, **updates: Any) -> TurnTrace:
    """返回一个已更新指定字段的新 trace。"""

    return trace.model_copy(update={key: value for key, value in updates.items() if value is not None})
