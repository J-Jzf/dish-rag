from dish_rag.agent.nodes import AgentNodes
from dish_rag.models import CookingState, EvidenceJudgeResult, QueryRewrite, RetrievalHit, TurnTrace
from dish_rag.storage.sqlite_store import SQLiteStore


class _RecordingChat:
    def __init__(self):
        self.complete_text_calls = 0

    def complete_text(self, system: str, user: str) -> str:
        self.complete_text_calls += 1
        return "基于当前证据生成的回答"


def test_answer_adds_warning_after_generation_when_evidence_is_insufficient(tmp_path):
    store = SQLiteStore(tmp_path / "dish_rag.sqlite3")
    store.migrate()
    chat = _RecordingChat()
    nodes = AgentNodes(chat, retriever=None, store=store)

    hit = RetrievalHit(
        chunk_id="001:steps",
        recipe_id="001",
        recipe_name="宫保鸡丁",
        field="steps",
        text="证据不足的步骤片段",
        page=3,
        score=0.1,
        source="bm25",
    )
    state = {
        "user_query": "宫保鸡丁怎么做？",
        "query_rewrite": QueryRewrite(
            raw_query="宫保鸡丁怎么做？",
            rewritten_query="宫保鸡丁怎么做？",
            intent="recipe_lookup",
        ),
        "hits": [hit],
        "evidence_judge": EvidenceJudgeResult(
            relevant=True,
            sufficient=False,
            confidence=0.2,
            missing=["缺少完整步骤"],
        ),
        "cooking_state": CookingState(),
        "trace": TurnTrace(raw_query="宫保鸡丁怎么做？"),
    }

    result = nodes.answer(state)

    assert chat.complete_text_calls == 1
    assert result["answer"] == "基于当前证据生成的回答\n\n证据不足"
