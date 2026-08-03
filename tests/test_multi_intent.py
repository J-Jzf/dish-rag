from dish_rag.agent.nodes import AgentNodes, _distinct_recipe_hits, route_after_action
from dish_rag.models import CookingState, Intent, IntentAction, IntentPlan, RetrievalHit, TurnTrace


def test_intent_plan_keeps_llm_dependency_order():
    plan = IntentPlan.model_validate(
        {
            "actions": [
                {"intent": "preference_update", "needs_retrieval": False},
                {
                    "intent": "recommendation",
                    "recommendation_count": 3,
                    "needs_retrieval": True,
                },
            ]
        }
    )

    assert [action.intent for action in plan.actions] == [
        Intent.PREFERENCE_UPDATE,
        Intent.RECOMMENDATION,
    ]
    assert plan.actions[1].recommendation_count == 3


def test_distinct_recommendation_hits_does_not_overwrite_navigation_recipe():
    hits = [
        RetrievalHit(
            chunk_id=f"{recipe_id}:audience",
            recipe_id=recipe_id,
            recipe_name=f"菜谱{recipe_id}",
            field="audience",
            text=f"菜谱{recipe_id}的说明",
            page=1,
            score=1.0 - index / 10,
            source="rerank",
        )
        for index, recipe_id in enumerate(["001", "001", "002", "003"])
    ]

    result = _distinct_recipe_hits(hits, count=3)

    assert [hit.recipe_id for hit in result] == ["001", "002", "003"]


def test_intent_action_defaults_recommendation_count_to_five():
    action = IntentAction(intent=Intent.RECOMMENDATION, needs_retrieval=True)

    assert action.recommendation_count == 5


class _MemoryStore:
    def __init__(self):
        self.saved = []

    def save_memory(self, user_id, key, value):
        self.saved.append((user_id, key, value))


def test_preference_action_updates_in_turn_memory_before_next_recommendation_action():
    store = _MemoryStore()
    nodes = AgentNodes(chat=None, retriever=None, store=store)
    state = {
        "thread_id": "kitchen-002",
        "user_id": "user-001",
        "user_query": "我不吃花生，推荐 3 道高蛋白菜",
        "cooking_state": CookingState(),
        "memory": {},
        "trace": TurnTrace(raw_query="我不吃花生，推荐 3 道高蛋白菜"),
        "intent_plan": IntentPlan(
            actions=[
                IntentAction(
                    intent=Intent.PREFERENCE_UPDATE,
                    completed_query="记录不吃花生",
                    preserved_constraints=["不吃花生"],
                    needs_retrieval=False,
                ),
                IntentAction(
                    intent=Intent.RECOMMENDATION,
                    completed_query="推荐 3 道不含花生的高蛋白菜",
                    recommendation_count=3,
                    preserved_constraints=["不吃花生"],
                    needs_retrieval=True,
                ),
            ]
        ),
        "current_action_index": 0,
        "action_results": [],
    }

    preference_state = nodes.prepare_action(state)
    completed_state = nodes.capture_action_result(preference_state)
    recommendation_state = nodes.prepare_action(completed_state)

    assert store.saved[0][0:2] == ("user-001", "preference_latest")
    assert "不吃花生" in recommendation_state["memory"]["preference_latest"]
    assert recommendation_state["query_rewrite"].intent == Intent.RECOMMENDATION
    assert recommendation_state["query_rewrite"].recommendation_count == 3
    assert route_after_action(completed_state) == "prepare_action"
