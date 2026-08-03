"""LangGraph 节点实现。"""

from __future__ import annotations

import json
from typing import Any

from langgraph.types import interrupt

from dish_rag.agent import prompts
from dish_rag.agent.state import DishAgentState
from dish_rag.llm.opai import ChatClient
from dish_rag.models import (
    ActionResult,
    Citation,
    CookingState,
    EvidenceJudgeResult,
    Intent,
    IntentAction,
    IntentPlan,
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
        """初始化 trace，并加载当前用户长期记忆，记录本轮开始前的烹饪状态。"""
        # 这是每轮对话的“开场记录节点”

        cooking_state = state.get("cooking_state") or CookingState() # 从 state 里取 cooking_state；没有就新建 CookingState()
        memory = self.store.load_memory(state.get("user_id", "default")) # 从 SQLite long_term_memory 里加载用户长期记忆（过敏原、偏好等）
        trace = TurnTrace(
            raw_query=state["user_query"],
            state_before=cooking_state.model_dump(),
        )
        return {
            **state,
            "cooking_state": cooking_state,
            "memory": memory,
            "evidence_retry_count": 0,
            "trace": trace,
        }
    # 所有的return的意思是：返回一个新的 state，保留旧状态（**state），但覆盖相应字段。

    def classify_intent(self, state: DishAgentState) -> DishAgentState:
        """识别意图，并构造补全后的 Query。"""

        safety = check_refusal(state["user_query"]) # 先走安全检查
        if not safety.allowed: # 如果不允许回答
            rewrite = QueryRewrite(
                raw_query=state["user_query"],
                intent=Intent.UNSAFE_OR_REFUSAL,
                needs_retrieval=False,
            )
            plan = IntentPlan(
                actions=[
                    IntentAction(
                        intent=Intent.UNSAFE_OR_REFUSAL,
                        completed_query=state["user_query"],
                        needs_retrieval=False,
                    )
                ]
            )
            trace = _update_trace(state["trace"], parsed_intent=_intent_value(rewrite.intent), notes=[safety.reason])
            return {
                **state,
                "query_rewrite": rewrite,
                "intent_plan": plan,
                "current_action_index": 0,
                "action_results": [],
                "trace": trace,
            }

        # 如果安全，就调用 LLM
        payload = self.chat.complete_json(
            prompts.INTENT_SYSTEM,
            prompts.INTENT_USER.format(
                query=state["user_query"],
                cooking_state=state["cooking_state"].model_dump(),
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
            ),
        )
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list):
            raw_actions = [payload]
        actions = [_intent_action_from_payload(item, state["user_query"]) for item in raw_actions if isinstance(item, dict)]
        if not actions:
            actions = [
                IntentAction(
                    intent=Intent.NEED_CLARIFICATION,
                    completed_query=state["user_query"],
                    needs_retrieval=False,
                )
            ]
        plan = IntentPlan(actions=actions)
        rewrite = _query_rewrite_from_action(state["user_query"], plan.actions[0])
        is_recommendation = _intent_value(rewrite.intent) == Intent.RECOMMENDATION.value
        trace = _update_trace(
            state["trace"],
            parsed_intent=_intent_value(rewrite.intent),
            completed_query=rewrite.completed_query,
            recipe_entities=rewrite.recipe_entities,
            recommendation_count=rewrite.recommendation_count if is_recommendation else 0,
        )
        return {
            **state,
            "query_rewrite": rewrite,
            "intent_plan": plan,
            "current_action_index": 0,
            "action_results": [],
            "trace": trace,
        }

    def prepare_action(self, state: DishAgentState) -> DishAgentState:
        """装载当前动作，并在需要时先执行会影响后续动作的副作用。"""

        plan = state.get("intent_plan") or IntentPlan()
        action_index = state.get("current_action_index", 0)
        if action_index >= len(plan.actions):
            return state

        action = plan.actions[action_index]
        rewrite = _query_rewrite_from_action(state["user_query"], action)
        memory = dict(state.get("memory", {}))
        trace = _update_trace(
            state["trace"],
            parsed_intent=_intent_value(rewrite.intent),
            completed_query=rewrite.completed_query,
            recipe_entities=rewrite.recipe_entities,
            recommendation_count=(
                rewrite.recommendation_count
                if _intent_value(rewrite.intent) == Intent.RECOMMENDATION.value
                else 0
            ),
        )

        if _intent_value(rewrite.intent) == Intent.PREFERENCE_UPDATE.value:
            memory_value = json.dumps(
                {
                    "raw_query": rewrite.raw_query,
                    "completed_query": rewrite.completed_query,
                    "preserved_constraints": rewrite.preserved_constraints,
                },
                ensure_ascii=False,
            )
            self.store.save_memory(state.get("user_id", "default"), "preference_latest", memory_value)
            memory["preference_latest"] = memory_value
            trace = _update_trace(
                trace,
                notes=[*trace.notes, "已在当前动作中写入 SQLite long_term_memory。"],
            )

        return {
            **state,
            "query_rewrite": rewrite,
            "memory": memory,
            "hits": [],
            "evidence_judge": None,
            "evidence_retry_count": 0,
            "needs_hitl": False,
            "hitl_candidates": [],
            "selected_recipe_id": None,
            "trace": trace,
        }

    def rewrite_query(self, state: DishAgentState) -> DishAgentState:
        """在保留用户限制条件的前提下重写 Query。"""
        # 把用户问题改写成更适合检索的 query

        rewrite = state["query_rewrite"]
        if not rewrite.needs_retrieval: # 如果不需要检索，就直接返回
            return state

        constraints = extract_user_constraints(rewrite.raw_query) # 先规则抽取用户限制
        payload = self.chat.complete_json( # 然后调用 LLM 重写 query
            prompts.REWRITE_SYSTEM,
            prompts.REWRITE_USER.format(
                raw_query=rewrite.raw_query,
                completed_query=rewrite.completed_query,
                recipe_entities=rewrite.recipe_entities,
                intent=_intent_value(rewrite.intent),
                recommendation_count=rewrite.recommendation_count,
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
                constraints=constraints,
            ),
        )
        rewritten_query = payload.get("rewritten_query") or rewrite.completed_query
        missing_constraints = detect_constraint_loss(constraints, rewritten_query)
        rewrite = rewrite.model_copy(
            update={
                "rewritten_query": rewritten_query,
                "preserved_constraints": _as_string_list(payload.get("preserved_constraints", constraints)),
                "removed_or_weakened_constraints": missing_constraints
                or _as_string_list(payload.get("removed_or_weakened_constraints", [])),
            }
        )
        if rewrite.removed_or_weakened_constraints: # 如果丢了限制，就把 intent 改成 need_clarification
            rewrite = rewrite.model_copy(update={"intent": Intent.NEED_CLARIFICATION})
        trace = _update_trace(state["trace"], rewritten_query=rewrite.rewritten_query)
        return {**state, "query_rewrite": rewrite, "trace": trace}

    def retrieve(self, state: DishAgentState) -> DishAgentState:
        """执行菜名精确匹配和混合检索。"""

        rewrite = state["query_rewrite"]
        if not rewrite.needs_retrieval: # 第一步，如果不需要检索
            return {**state, "hits": []}

        if (
            _intent_value(rewrite.intent) == Intent.COOKING_NAVIGATION.value
            and not rewrite.recipe_entities
            and not state["cooking_state"].active_recipe_id # 第二步，处理烹饪导航缺状态的情况
        ):
            trace = _update_trace(
                state["trace"],
                qdrant_hits=[],
                notes=[*state["trace"].notes, "烹饪导航缺少当前菜谱状态和显式菜名，跳过全库检索并等待澄清。"],
            )
            return {**state, "hits": [], "trace": trace} 
        # 它本身不会暂停图执行。它只是让后续节点看到“没有 hits、trace 里有说明”，后续“answer()”根据状态返回输出的话术，**图的这一轮还是会继续走完**。

        if state.get("evidence_retry_count", 0) > 0 and state.get("selected_recipe_id"):
            hits = self.retriever.retrieve(
                rewrite.rewritten_query,
                filters={"recipe_id": state["selected_recipe_id"]},
            )
            trace = _update_trace(state["trace"], qdrant_hits=hits)
            return {**state, "hits": hits, "trace": trace}

        if _intent_value(rewrite.intent) == Intent.RECOMMENDATION.value:
            candidate_limit = max(rewrite.recommendation_count * 4, 8)
            candidates = self.retriever.retrieve(rewrite.rewritten_query, limit=candidate_limit)
            hits = _distinct_recipe_hits(candidates, rewrite.recommendation_count)
            trace = _update_trace(state["trace"], qdrant_hits=hits)
            return {**state, "hits": hits, "trace": trace}

        # 第三步，如果 LLM 识别出了菜名
        # 精确菜名解析不做语义替换。
        for entity in rewrite.recipe_entities:
            exact = self.retriever.exact_recipe(entity) # 先查 SQLite aliases 做精确匹配。
            if exact:
                filters = {"recipe_id": exact.recipe_id}
                hits = self.retriever.retrieve(rewrite.rewritten_query, filters=filters) # 如果命中，就只在这道菜里 hybrid 检索
                trace = _update_trace(state["trace"], qdrant_hits=hits)
                return {**state, "hits": hits, "trace": trace}

        # 第四步，如果有菜名但精确匹配失败
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

        active_recipe_id = state["cooking_state"].active_recipe_id
        
        # 第五步，如果是 cooking_navigation 且 checkpoint 里有当前菜
        if _intent_value(rewrite.intent) == Intent.COOKING_NAVIGATION.value and active_recipe_id:
            hits = self.retriever.retrieve(
                rewrite.rewritten_query,
                filters={"recipe_id": active_recipe_id},
            )
            trace = _update_trace(state["trace"], qdrant_hits=hits)
            return {**state, "hits": hits, "trace": trace}

        # 最后普通情况，全库 hybrid 检索。
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
        """LangGraph 里的证据判断节点。判断检索证据是否足够回答。"""

        if (
            not state.get("hits") # 没有检索命中（hits 是检索命中结果列表，每个 hit 基本对应一个命中的 chunk）
            and _intent_value(state["query_rewrite"].intent) == Intent.COOKING_NAVIGATION.value # 当前意图是 cooking_navigation（查下一步）
            and state["cooking_state"].active_recipe_id # checkpoint 里已经知道正在做哪道菜
        ):
            judge = EvidenceJudgeResult(
                relevant=True,
                sufficient=True,
                confidence=1.0,
                reasons=["烹饪导航可直接使用 checkpoint 中的当前菜谱和步骤状态"],
            )
            trace = _update_trace(state["trace"], evidence_judge=judge) # 把 judge 写入当前 state 和 trace。
            return {**state, "evidence_judge": judge, "trace": trace}

        if not state.get("hits"): # 没有检索命中，无法判断证据充分性。
            judge = EvidenceJudgeResult(
                relevant=False,
                sufficient=False,
                confidence=0.0,
                missing=["没有检索到可引用证据"],
            )
            trace = _update_trace(state["trace"], evidence_judge=judge)
            return {**state, "evidence_judge": judge, "trace": trace}

        # 如果有 hits，就调用 LLM 做 Evidence Judge。
        payload = self.chat.complete_json(
            prompts.JUDGE_SYSTEM,
            prompts.JUDGE_USER.format(
                query=state["query_rewrite"].rewritten_query, # 重写后的问题
                evidence=_format_hits(state["hits"]), # 检索命中的 chunk 证据（不是只返回纯 chunk.text，而是格式化成[PDF页码｜菜谱编号 菜名｜字段｜score=分数] chunk文本）
            ),
        )
        # 把 LLM 返回的 JSON 转成标准 EvidenceJudgeResult，如果某些字段没返回，就用默认值（False 或 0.0）
        judge = EvidenceJudgeResult(
            relevant=bool(payload.get("relevant", False)),
            sufficient=bool(payload.get("sufficient", False)),
            confidence=float(payload.get("confidence", 0.0)),
            reasons=_as_string_list(payload.get("reasons", [])),
            missing=_as_string_list(payload.get("missing", [])),
        )
        # 最后同样写入 state 和 trace
        trace = _update_trace(state["trace"], evidence_judge=judge)
        return {**state, "evidence_judge": judge, "trace": trace}

    def retry_evidence(self, state: DishAgentState) -> DishAgentState:
        """根据 Evidence Judge 的缺失项重写 Query，准备唯一一次重检索。"""

        rewrite = state["query_rewrite"]
        judge = state.get("evidence_judge")
        retry_count = state.get("evidence_retry_count", 0) + 1
        constraints = extract_user_constraints(rewrite.raw_query)
        payload = self.chat.complete_json(
            prompts.RETRY_REWRITE_SYSTEM,
            prompts.RETRY_REWRITE_USER.format(
                raw_query=rewrite.raw_query,
                completed_query=rewrite.completed_query,
                rewritten_query=rewrite.rewritten_query,
                constraints=constraints,
                judge=judge.model_dump_json() if judge else "{}",
                evidence=_format_hits(state.get("hits", [])),
            ),
        )
        rewritten_query = (
            payload.get("rewritten_query")
            or rewrite.rewritten_query
            or rewrite.completed_query
            or rewrite.raw_query
        )
        rewrite = rewrite.model_copy(update={"rewritten_query": rewritten_query})
        trace = _update_trace(
            state["trace"],
            rewritten_query=rewritten_query,
            evidence_retry_count=retry_count,
            notes=[
                *state["trace"].notes,
                "Evidence Judge 触发第 1 次重检索。",
            ],
        )
        return {
            **state,
            "query_rewrite": rewrite,
            "evidence_retry_count": retry_count,
            "hits": [],
            "needs_hitl": False,
            "trace": trace,
        }

    def capture_action_result(self, state: DishAgentState) -> DishAgentState:
        """在下一项动作开始前，保存当前动作的独立结果。"""

        rewrite = state["query_rewrite"]
        intent = _intent_value(rewrite.intent)
        answer_hint = ""
        citations: list[Citation] = []

        if intent == Intent.PREFERENCE_UPDATE.value:
            answer_hint = "我记下了这项偏好，后续推荐或食材建议会将它作为限制参考。"
        elif intent == Intent.UNSAFE_OR_REFUSAL.value:
            answer_hint = "这个请求涉及高风险或医疗化内容，我不能按这个方向提供做法。"
        elif intent == Intent.NEED_CLARIFICATION.value:
            answer_hint = "我需要先确认缺失的信息或必须保留的限制，请你补充说明。"
        elif intent == Intent.COOKING_NAVIGATION.value:
            navigation_answer = _build_cooking_navigation_answer(state, self.store)
            if navigation_answer:
                answer_hint, raw_citations = navigation_answer
                citations = [Citation(**item) for item in raw_citations]

        if not citations:
            citation_hits = (
                state.get("hits", [])
                if intent == Intent.RECOMMENDATION.value
                else state.get("hits", [])[:4]
            )
            citations = [_citation_from_hit(hit) for hit in citation_hits]

        result = ActionResult(
            action_index=state.get("current_action_index", 0),
            intent=rewrite.intent,
            query=rewrite.rewritten_query or rewrite.completed_query,
            recommendation_count=(
                rewrite.recommendation_count if intent == Intent.RECOMMENDATION.value else 0
            ),
            hits=state.get("hits", []),
            evidence_judge=state.get("evidence_judge"),
            evidence_retry_count=state.get("evidence_retry_count", 0),
            answer_hint=answer_hint,
            citations=citations,
        )
        action_results = [*state.get("action_results", []), result]
        trace = _update_trace(state["trace"], action_results=action_results)
        return {
            **state,
            "action_results": action_results,
            "current_action_index": state.get("current_action_index", 0) + 1,
            "trace": trace,
        }

    def update_cooking_state(self, state: DishAgentState) -> DishAgentState:
        """更新当前 thread 的烹饪进度。"""

        cooking_state = state["cooking_state"].model_copy()
        intent = _intent_value(state["query_rewrite"].intent)
        hits = state.get("hits", [])
        notes = list(state["trace"].notes)
        if intent == Intent.COOKING_START.value and hits: # 如果意图是 cooking_start 并且有 hits，就取第一个命中菜谱
            recipe = self.store.get_recipe(hits[0].recipe_id)
            if recipe:
                cooking_state.active_recipe_id = recipe.recipe_id
                cooking_state.active_recipe_name = recipe.name
                cooking_state.current_step_no = 1
                cooking_state.total_steps = len(recipe.steps)
                cooking_state.last_action = "start"
                notes.append("开始做菜：初始化当前菜谱和步骤状态。")
        elif intent == Intent.RECIPE_LOOKUP.value and hits:
            recipe = self.store.get_recipe(hits[0].recipe_id)
            if recipe:
                same_recipe = cooking_state.active_recipe_id == recipe.recipe_id
                cooking_state.active_recipe_id = recipe.recipe_id
                cooking_state.active_recipe_name = recipe.name
                cooking_state.total_steps = len(recipe.steps)
                cooking_state.current_step_no = cooking_state.current_step_no if same_recipe and cooking_state.current_step_no else 1
                cooking_state.last_action = "recipe_lookup_active"
                notes.append("菜谱查询命中菜名：把该菜设为当前 thread 正在做的菜。")
        elif intent == Intent.FIELD_LOOKUP.value and hits:
            step_hit = next((hit for hit in hits if hit.step_no is not None), None)
            if step_hit:
                recipe = self.store.get_recipe(step_hit.recipe_id)
                if recipe:
                    cooking_state.active_recipe_id = recipe.recipe_id
                    cooking_state.active_recipe_name = recipe.name
                    cooking_state.total_steps = len(recipe.steps)
                    cooking_state.current_step_no = step_hit.step_no
                    cooking_state.last_action = "field_lookup_step_active"
                    notes.append(f"步骤字段查询命中步骤 {step_hit.step_no}：同步为当前步骤。")
        elif intent == Intent.COOKING_NAVIGATION.value: # 如果意图是cooking_navigation
            cooking_state, note = _apply_cooking_navigation(
                cooking_state,
                hits,
                state["user_query"],
                self.store,
            )
            if note:
                notes.append(note)
        trace = _update_trace(state["trace"], state_after=cooking_state.model_dump(), notes=notes)
        return {**state, "cooking_state": cooking_state, "trace": trace}

    def answer_actions(self, state: DishAgentState) -> DishAgentState:
        """将多个已完成动作合并为一次最终回答。"""

        results = state.get("action_results", [])
        citations = _dedupe_citations(
            citation for result in results for citation in result.citations
        )
        has_evidence = any(result.hits for result in results)
        if not has_evidence:
            answer = "\n\n".join(result.answer_hint for result in results if result.answer_hint)
            if not answer:
                answer = "菜谱库里没有找到可追溯证据。"
            trace = _update_trace(state["trace"], final_citations=citations)
            return {
                **state,
                "answer": answer,
                "citations": [citation.model_dump() for citation in citations],
                "trace": trace,
            }

        answer = self.chat.complete_text(
            prompts.MULTI_ACTION_ANSWER_SYSTEM,
            prompts.MULTI_ACTION_ANSWER_USER.format(
                action_results=_format_action_results(results),
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
            ),
        )
        if any(
            result.evidence_judge is not None and not result.evidence_judge.sufficient
            for result in results
            if result.hits
        ):
            answer = f"{answer.rstrip()}\n\n证据不足"
        trace = _update_trace(state["trace"], final_citations=citations)
        return {
            **state,
            "answer": answer,
            "citations": [citation.model_dump() for citation in citations],
            "trace": trace,
        }

    def answer(self, state: DishAgentState) -> DishAgentState:
        """生成最终可溯源回答。"""

        if state.get("action_results"):
            return self.answer_actions(state)

        rewrite = state["query_rewrite"]
        intent = _intent_value(rewrite.intent)
        if intent == Intent.UNSAFE_OR_REFUSAL.value:
            answer = "这个请求涉及高风险或医疗化内容，我不能按这个方向提供做法。"
            return {**state, "answer": answer}
        if intent == Intent.NEED_CLARIFICATION.value:
            answer = "我需要先确认限制条件：你刚才的约束不能被改写或省略，请再说明一次你要保留的禁忌或口味要求。"
            return {**state, "answer": answer}
        if intent == Intent.PREFERENCE_UPDATE.value:
            memory_value = json.dumps(
                {
                    "raw_query": rewrite.raw_query,
                    "completed_query": rewrite.completed_query,
                    "preserved_constraints": rewrite.preserved_constraints,
                },
                ensure_ascii=False,
            )
            self.store.save_memory(state.get("user_id", "default"), "preference_latest", memory_value)
            trace = _update_trace(
                state["trace"],
                notes=[*state["trace"].notes, "已把用户偏好/限制写入 SQLite long_term_memory。"],
            )
            answer = "我记下了。涉及食材建议或替换时，我会把这个偏好作为参考，并在关键场景再次向你确认。"
            return {**state, "answer": answer, "trace": trace}
        if not state.get("hits"):
            navigation_answer = _build_cooking_navigation_answer(state, self.store)
            if navigation_answer:
                answer, citations = navigation_answer
                trace = _update_trace(
                    state["trace"],
                    final_citations=[Citation(**item) for item in citations],
                )
                return {**state, "answer": answer, "citations": citations, "trace": trace}
            answer = "菜谱库里没有找到可追溯证据。我不会自动替换成相似菜；你可以选择一个相似菜或换个菜名。"
            return {**state, "answer": answer}

        navigation_answer = _build_cooking_navigation_answer(state, self.store)
        if navigation_answer:
            answer, citations = navigation_answer
            trace = _update_trace(
                state["trace"],
                final_citations=[Citation(**item) for item in citations],
            )
            return {**state, "answer": answer, "citations": citations, "trace": trace}

        answer = self.chat.complete_text(
            prompts.ANSWER_SYSTEM,
            prompts.ANSWER_USER.format(
                query=rewrite.rewritten_query,
                intent=intent,
                recommendation_count=rewrite.recommendation_count,
                evidence=_format_hits(state["hits"]),
                cooking_state=state["cooking_state"].model_dump(),
                memory=json.dumps(state.get("memory", {}), ensure_ascii=False),
            ),
        )
        judge = state.get("evidence_judge")
        if judge is not None and not judge.sufficient:
            answer = f"{answer.rstrip()}\n\n证据不足"
        citation_hits = state["hits"] if intent == Intent.RECOMMENDATION.value else state["hits"][:4]
        citations = [_citation_from_hit(hit).model_dump() for hit in citation_hits]
        trace = _update_trace(state["trace"], final_citations=[Citation(**item) for item in citations])
        return {**state, "answer": answer, "citations": citations, "trace": trace}

    def persist_trace(self, state: DishAgentState) -> DishAgentState:
        """把 trace 保存到 SQLite，便于调试和审计。"""

        trace = state.get("trace")
        if trace:
            self.store.append_trace(state.get("thread_id", "default"), trace.model_dump(mode="json"))
        return state


def route_after_action(state: DishAgentState) -> str:
    """继续执行下一项动作，或在澄清/拒答后终止本轮后续动作。"""

    results = state.get("action_results", [])
    if results and _intent_value(results[-1].intent) in {
        Intent.UNSAFE_OR_REFUSAL.value,
        Intent.NEED_CLARIFICATION.value,
    }:
        return "answer"
    plan = state.get("intent_plan") or IntentPlan()
    return "prepare_action" if state.get("current_action_index", 0) < len(plan.actions) else "answer"


def route_after_retrieve(state: DishAgentState) -> str:
    """精确菜名失败但有候选时，路由到 HITL。"""

    return "hitl_recipe_choice" if state.get("needs_hitl") else "judge_evidence"


def route_after_judge(state: DishAgentState) -> str:
    """证据不合格时最多触发一次重写和重检索。"""

    if state.get("evidence_retry_count", 0) >= 1:
        return "update_cooking_state"
    rewrite = state.get("query_rewrite")
    if rewrite is None or not rewrite.needs_retrieval:
        return "update_cooking_state"
    judge = state.get("evidence_judge")
    if judge and (
        not judge.relevant
        or not judge.sufficient
        or judge.confidence < 0.55
    ):
        return "retry_evidence"
    return "update_cooking_state"


def _apply_cooking_navigation(
    cooking_state: CookingState,
    hits: list[RetrievalHit],
    query: str,
    store: SQLiteStore,
) -> tuple[CookingState, str]:
    """根据 checkpoint 状态或 hybrid 命中的步骤推进烹饪状态。"""

    next_state = cooking_state.model_copy()
    step_hit = next((hit for hit in hits if hit.step_no is not None), None)
    if step_hit:
        recipe = store.get_recipe(step_hit.recipe_id)
        if recipe:
            next_state.active_recipe_id = recipe.recipe_id
            next_state.active_recipe_name = recipe.name
            next_state.total_steps = len(recipe.steps)
            if _asks_previous(query):
                next_state.current_step_no = max(1, step_hit.step_no - 1)
                next_state.last_action = "previous_from_matched_step"
            elif _asks_repeat(query):
                next_state.current_step_no = step_hit.step_no
                next_state.last_action = "repeat_matched_step"
            else:
                next_state.current_step_no = min(next_state.total_steps, step_hit.step_no + 1)
                next_state.last_action = (
                    "completed" if step_hit.step_no >= next_state.total_steps else "next_from_matched_step"
                )
            return (
                next_state,
                f"烹饪导航：hybrid 命中步骤 {step_hit.step_no}，同步当前步骤为 {next_state.current_step_no}。",
            )

    if not next_state.active_recipe_id:
        return next_state, "烹饪导航：没有当前菜谱状态，也没有命中步骤，无法推进。"

    if _asks_previous(query):
        next_state.current_step_no = max(1, next_state.current_step_no - 1)
        next_state.last_action = "previous"
    elif _asks_repeat(query):
        next_state.last_action = "repeat"
    else:
        next_state.current_step_no = min(next_state.total_steps, next_state.current_step_no + 1)
        next_state.last_action = "completed" if cooking_state.current_step_no >= next_state.total_steps else "next"
    return next_state, f"烹饪导航：基于 checkpoint 推进到步骤 {next_state.current_step_no}。"


def _build_cooking_navigation_answer(
    state: DishAgentState,
    store: SQLiteStore,
) -> tuple[str, list[dict[str, Any]]] | None:
    """根据 CookingState.current_step_no 回答当前应执行的步骤。"""

    rewrite = state["query_rewrite"]
    if _intent_value(rewrite.intent) != Intent.COOKING_NAVIGATION.value:
        return None

    cooking_state = state["cooking_state"]
    if not cooking_state.active_recipe_id:
        answer = "我需要先确认你正在做哪道菜。请先说菜名，例如“我要做宫保鸡丁”，或者直接问“宫保鸡丁调好碗汁了接下来做什么”。"
        return answer, []

    recipe = store.get_recipe(cooking_state.active_recipe_id)
    if not recipe:
        return None

    if cooking_state.last_action == "completed":
        return "已完成", []

    step_no = cooking_state.current_step_no
    if step_no < 1:
        step_no = 1
    if step_no > len(recipe.steps):
        answer = f"{recipe.name} 的菜谱一共 {len(recipe.steps)} 步，当前已经到最后之后了，没有更多步骤。"
        return answer, []

    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    state_before = state["trace"].state_before
    include_prior_steps = bool(
        state["query_rewrite"].recipe_entities
        and state_before.get("active_recipe_id") != recipe.recipe_id
        and state["cooking_state"].last_action == "next_from_matched_step"
        and step_no > 2
    )
    if include_prior_steps:
        lines.append("你问到的这个操作之前，菜谱里已经有这些步骤：")
        for step_text in recipe.steps[: step_no - 2]:
            lines.append(f"- {step_text}")
        lines.append("")

    label = "当前步骤"
    if cooking_state.last_action in {"next", "next_from_matched_step"}:
        label = "下一步"
    elif cooking_state.last_action in {"previous", "previous_from_matched_step"}:
        label = "上一步"
    elif cooking_state.last_action in {"repeat", "repeat_matched_step"}:
        label = "重复这一步"

    current_hit = RetrievalHit(
        chunk_id=f"{recipe.recipe_id}:step_{step_no:02d}",
        recipe_id=recipe.recipe_id,
        recipe_name=recipe.name,
        field="RecipeField.STEPS",
        text=recipe.steps[step_no - 1],
        page=recipe.page_start,
        step_no=step_no,
        score=1.0,
        source="exact",
        filters={"recipe_id": recipe.recipe_id, "step_no": step_no},
    )
    lines.append(f"{label}是：{current_hit.text}")
    citations.append(_citation_from_hit(current_hit).model_dump())

    return "\n".join(lines), citations


def _asks_previous(query: str) -> bool:
    """判断用户是否在请求回到前一步。"""

    return "上一步" in query or "回到" in query or "前一步" in query


def _asks_repeat(query: str) -> bool:
    """判断用户是否在请求重复当前步骤。"""

    return "重复" in query or "再说" in query or "重新说" in query


def _intent_action_from_payload(payload: dict[str, Any], raw_query: str) -> IntentAction:
    """将 LLM 的单项动作 JSON 规范为可执行动作。"""

    intent = payload.get("intent", Intent.NEED_CLARIFICATION)
    action = IntentAction(
        intent=intent,
        completed_query=payload.get("completed_query") or raw_query,
        recipe_entities=_as_string_list(payload.get("recipe_entities", [])),
        recommendation_count=_as_recommendation_count(payload.get("recommendation_count")),
        needs_retrieval=bool(payload.get("needs_retrieval", True)),
        preserved_constraints=_as_string_list(payload.get("preserved_constraints", [])),
    )
    if _intent_value(action.intent) == Intent.RECOMMENDATION.value:
        return action.model_copy(update={"needs_retrieval": True})
    if _intent_value(action.intent) in {
        Intent.PREFERENCE_UPDATE.value,
        Intent.CHITCHAT.value,
        Intent.UNSAFE_OR_REFUSAL.value,
        Intent.NEED_CLARIFICATION.value,
    }:
        return action.model_copy(update={"needs_retrieval": False})
    return action


def _query_rewrite_from_action(raw_query: str, action: IntentAction) -> QueryRewrite:
    """把当前动作转换为现有检索节点继续使用的 QueryRewrite。"""

    return QueryRewrite(
        raw_query=raw_query,
        completed_query=action.completed_query or raw_query,
        intent=action.intent,
        recipe_entities=action.recipe_entities,
        recommendation_count=action.recommendation_count,
        needs_retrieval=action.needs_retrieval,
        preserved_constraints=action.preserved_constraints,
    )


def _intent_value(intent: Intent | str) -> str:
    """统一取得 intent 的字符串值，兼容 Enum 和普通字符串。"""

    return intent.value if isinstance(intent, Intent) else str(intent)


def _as_string_list(value: object) -> list[str]:
    """把 LLM 可能返回的字典、字符串或列表统一成字符串列表。"""

    if value in (None, ""):
        return []
    if isinstance(value, dict):
        return _as_string_list(list(value.values()))
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _as_recommendation_count(value: object) -> int:
    """将 LLM 返回的推荐数量规范为正整数；未提供或非法时默认 5。"""

    try:
        count = int(value)
    except (TypeError, ValueError):
        return 5
    return count if count >= 1 else 5


def _distinct_recipe_hits(hits: list[RetrievalHit], count: int) -> list[RetrievalHit]:
    """从 rerank 后的 chunk 命中中保留每道菜的首个命中，保证推荐菜谱不重复。"""

    selected: list[RetrievalHit] = []
    recipe_ids: set[str] = set()
    for hit in hits:
        if hit.recipe_id in recipe_ids:
            continue
        selected.append(hit)
        recipe_ids.add(hit.recipe_id)
        if len(selected) >= count:
            break
    return selected


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
        step_no=hit.step_no,
        chunk_id=hit.chunk_id,
    )


def _format_action_results(results: list[ActionResult]) -> str:
    """将顺序动作、直接答复提示和证据统一提供给最终回答模型。"""

    sections: list[str] = []
    for result in results:
        sections.append(
            "\n".join(
                [
                    f"动作 {result.action_index + 1}：{_intent_value(result.intent)}",
                    f"问题：{result.query}",
                    f"推荐数量：{result.recommendation_count or '无'}",
                    f"重检索次数：{result.evidence_retry_count}",
                    f"直接结果：{result.answer_hint or '无'}",
                    "证据：",
                    _format_hits(result.hits) or "无",
                ]
            )
        )
    return "\n\n".join(sections)


def _dedupe_citations(citations: Any) -> list[Citation]:
    """按 chunk 与步骤去重，保留动作原有的引用顺序。"""

    result: list[Citation] = []
    seen: set[tuple[str | None, str, int | None]] = set()
    for citation in citations:
        key = (citation.chunk_id, citation.recipe_id, citation.step_no)
        if key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


def _update_trace(trace: TurnTrace, **updates: Any) -> TurnTrace:
    """返回一个已更新指定字段的新 trace。"""

    return trace.model_copy(update={key: value for key, value in updates.items() if value is not None})
