from dish_rag.agent.nodes import AgentNodes
from dish_rag.models import CookingState, EvidenceJudgeResult, QueryRewrite, RetrievalHit, TurnTrace
from dish_rag.storage.sqlite_store import SQLiteStore


class _RecordingChat:
    def __init__(self):
        self.complete_text_calls = 0

    def complete_text(self, system: str, user: str) -> str:
        self.complete_text_calls += 1
        return "generated answer"


def test_answer_includes_llm_evidence_judge_reasons_when_evidence_is_insufficient(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    chat = _RecordingChat()
    nodes = AgentNodes(chat, retriever=None, store=store)
    hit = RetrievalHit(
        chunk_id="001:steps",
        recipe_id="001",
        recipe_name="recipe",
        field="steps",
        text="partial evidence",
        page=3,
        score=0.1,
        source="bm25",
    )
    state = {
        "user_query": "how to cook it",
        "query_rewrite": QueryRewrite(
            raw_query="how to cook it",
            rewritten_query="how to cook it",
            intent="recipe_lookup",
        ),
        "hits": [hit],
        "evidence_judge": EvidenceJudgeResult(
            relevant=True,
            sufficient=False,
            confidence=0.2,
            reasons=["matched chunks do not cover the cooking method"],
            missing=["complete steps"],
        ),
        "cooking_state": CookingState(),
        "trace": TurnTrace(raw_query="how to cook it"),
    }

    result = nodes.answer(state)

    assert chat.complete_text_calls == 1
    assert "matched chunks do not cover the cooking method" in result["answer"]
    assert "complete steps" in result["answer"]
