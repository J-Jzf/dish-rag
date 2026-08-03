import pytest

from dish_rag.agent.nodes import AgentNodes
from dish_rag.models import CookingState, QueryRewrite, RetrievalHit, TurnTrace
from dish_rag.storage.sqlite_store import SQLiteStore


class _IntentChat:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user):
        return self.payload


class _RecordingRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.limits = []

    def retrieve(self, query, limit=8, filters=None):
        self.limits.append(limit)
        return self.hits


def _nodes(tmp_path, chat, retriever=None):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    return AgentNodes(chat, retriever, store)


@pytest.mark.parametrize(
    ("payload_count", "expected_count"),
    [
        (3, 3),
        (None, 5),
    ],
)
def test_classify_recommendation_uses_requested_count_or_default(tmp_path, payload_count, expected_count):
    payload = {
        "intent": "recommendation",
        "completed_query": "推荐适合健身爱好者的高蛋白菜",
        "recipe_entities": [],
        "needs_retrieval": False,
        "preserved_constraints": [],
    }
    if payload_count is not None:
        payload["recommendation_count"] = payload_count

    result = _nodes(tmp_path, _IntentChat(payload)).classify_intent(
        {
            "user_query": "推荐几道适合健身的高蛋白菜",
            "cooking_state": CookingState(),
            "memory": {},
            "trace": TurnTrace(raw_query="推荐几道适合健身的高蛋白菜"),
        }
    )

    assert result["query_rewrite"].intent == "recommendation"
    assert result["query_rewrite"].recommendation_count == expected_count
    assert result["query_rewrite"].needs_retrieval is True


def test_recommendation_retrieval_returns_requested_distinct_recipes(tmp_path):
    hits = [
        RetrievalHit(
            chunk_id=f"{recipe_id}:audience",
            recipe_id=recipe_id,
            recipe_name=f"菜谱{recipe_id}",
            field="audience",
            text=f"菜谱{recipe_id}的适合人群说明",
            page=1,
            score=1.0 - index / 10,
            source="rerank",
        )
        for index, recipe_id in enumerate(["001", "001", "002", "003", "004"])
    ]
    retriever = _RecordingRetriever(hits)
    nodes = _nodes(tmp_path, _IntentChat({}), retriever)

    result = nodes.retrieve(
        {
            "query_rewrite": QueryRewrite(
                raw_query="推荐 3 道高蛋白菜",
                rewritten_query="推荐适合健身的高蛋白菜",
                intent="recommendation",
                recommendation_count=3,
                needs_retrieval=True,
            ),
            "cooking_state": CookingState(),
            "trace": TurnTrace(raw_query="推荐 3 道高蛋白菜"),
        }
    )

    assert retriever.limits == [12]
    assert [hit.recipe_id for hit in result["hits"]] == ["001", "002", "003"]
