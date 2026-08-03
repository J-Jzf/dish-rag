"""LangGraph 图组装。"""

from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from dish_rag.agent.nodes import AgentNodes, route_after_judge, route_after_retrieve
from dish_rag.agent.state import DishAgentState


def build_graph(nodes: AgentNodes, checkpoint_path: Path):
    """编译带 SQLite checkpoint 的菜谱 Agent 图。"""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # 当前版本的 SqliteSaver.from_conn_string() 返回上下文管理器；
    # graph.compile() 需要真正的 BaseCheckpointSaver 实例，所以这里显式创建连接。
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    graph = StateGraph(DishAgentState)
    # 定义图的节点顺序
    graph.add_node("start_trace", nodes.start_trace)
    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("hitl_recipe_choice", nodes.hitl_recipe_choice)
    graph.add_node("retrieve_selected_recipe", nodes.retrieve_selected_recipe)
    graph.add_node("judge_evidence", nodes.judge_evidence)
    graph.add_node("retry_evidence", nodes.retry_evidence)
    graph.add_node("update_cooking_state", nodes.update_cooking_state)
    graph.add_node("answer", nodes.answer)
    graph.add_node("persist_trace", nodes.persist_trace)
    # 定义图的边
    graph.add_edge(START, "start_trace")
    graph.add_edge("start_trace", "classify_intent")
    graph.add_edge("classify_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "hitl_recipe_choice": "hitl_recipe_choice",
            "judge_evidence": "judge_evidence",
        },
    )
    graph.add_edge("hitl_recipe_choice", "retrieve_selected_recipe")
    graph.add_edge("retrieve_selected_recipe", "judge_evidence")
    graph.add_conditional_edges(
        "judge_evidence",
        route_after_judge,
        {
            "retry_evidence": "retry_evidence",
            "update_cooking_state": "update_cooking_state",
        },
    )
    graph.add_edge("retry_evidence", "retrieve")
    graph.add_edge("update_cooking_state", "answer")
    graph.add_edge("answer", "persist_trace")
    graph.add_edge("persist_trace", END)

    return graph.compile(checkpointer=checkpointer)
