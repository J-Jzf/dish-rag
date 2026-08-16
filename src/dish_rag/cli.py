"""菜谱 RAG 项目的命令行入口。"""

from pathlib import Path
from typing import Annotated
import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="菜谱 Agentic RAG CLI") # 创建一个命令行应用对象。
# help 是这个命令行工具的说明文字，运行python main.py --help命令时会显示。
console = Console() # 创建一个 Rich 的终端输出对象，用来更漂亮地打印内容，比如彩色表格、面板、格式化文本。比普通 print() 更强。


def render_retrieval_failures(failures: list[dict[str, object]], limit: int) -> Table:
    """将 Recall@K 未命中的检索样例渲染为终端表格。"""

    table = Table(title=f"Retrieval Failures (FAIL@{limit})", show_lines=True)
    table.add_column("Query", overflow="fold")
    table.add_column("Expected Recipe IDs")
    table.add_column(f"Actual Top-{limit}", overflow="fold")
    table.add_column("Expected Rank")
    for failure in failures:
        actual_top_k = failure["actual_top_k"]
        actual_text = "\n".join(
            f"{hit['recipe_id']} {hit['recipe_name']} ({hit['score']:.3f})"
            for hit in actual_top_k
        ) or "无命中"
        rank = failure["rank"]
        table.add_row(
            str(failure["query"]),
            ", ".join(failure["expected_recipe_ids"]),
            actual_text,
            str(rank) if rank is not None else "未命中",
        )
    return table


@app.command() # 装饰器 把下面这个函数注册成一个命令行命令。使得可以python main.py ingest
def ingest(
    index_qdrant: Annotated[ # index_qdrant参数名，表示“是否写入 Qdrant 索引”。Annotated 是 Python 的类型标注工具。
        bool, # 参数类型是布尔值
        typer.Option(help="是否同时写入 Qdrant hybrid 索引。"), # 这个选项在 --help 里显示的说明
    ] = True, # 默认值是 True，也就是默认会写入 Qdrant 索引。
) -> None: # 函数返回值是 None
    """构建 Markdown、JSON、SQLite，并可选写入 Qdrant 索引。"""

    from dish_rag.config import get_settings
    from dish_rag.ingest.pipeline import run_ingest

    settings = get_settings(Path.cwd())
    result = run_ingest(settings, index_qdrant=index_qdrant)
    console.print(result)


@app.command()
def search(query: str, limit: int = 8) -> None:
    """执行检索，并打印命中项、分数和字段。"""

    from dish_rag.config import get_settings
    from dish_rag.factory import make_retriever, make_store

    settings = get_settings(Path.cwd())
    store = make_store(settings)
    retriever = make_retriever(settings, store)
    hits = retriever.retrieve(query, limit=limit)
    for hit in hits:
        console.print(
            f"{hit.score:.3f} {hit.source} "
            f"[p.{hit.page} {hit.recipe_id} {hit.recipe_name} {hit.field}] {hit.text}"
        )


@app.command()
def chat(
    query: str,
    thread_id: str = "default", # 对话线程 ID，默认是 "default"，也可以自己起名为类似--thread-id kitchen-001，一条对话线（当前的代码实现的版本）维护一个菜品的状态
    user_id: str = "default", # 用户 ID，默认是 "default"，主要用于长期记忆，比如用户偏好、过敏信息。
) -> None:
    """运行一轮 LangGraph 对话。"""

    from dish_rag.config import get_settings
    from dish_rag.factory import make_graph
    from dish_rag.models import TurnTrace
    from dish_rag.observability import render_trace

    settings = get_settings(Path.cwd())
    graph = make_graph(settings) # 编译图
    result = graph.invoke( # 执行一轮 LangGraph
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "user_query": query,
        },
        config={"configurable": {"thread_id": thread_id}}, # 这是给 LangGraph checkpoint 用的，LangGraph 会用这个 thread_id 保存/读取 checkpoint。
    )
    if "__interrupt__" in result: # 这里处理 HITL
        console.print(result["__interrupt__"])
        return
    console.print(result.get("answer", "")) # 这里处理 HITL
    if result.get("trace"):
        trace = result["trace"] # 取出 trace。trace 是本轮执行过程记录，如识别意图、重写query、检索命中、证据判断、状态变化
        render_trace(trace if isinstance(trace, TurnTrace) else TurnTrace.model_validate(trace)) # trace 数据模型


@app.command()
def eval_retrieval(
    eval_file: Path = Path("configs/eval_queries.jsonl"),
    limit: int = 5,
) -> None:
    """运行离线检索评测指标。"""

    from dish_rag.config import get_settings
    from dish_rag.eval.offline import run_retrieval_eval
    from dish_rag.factory import make_retriever, make_store

    settings = get_settings(Path.cwd())
    store = make_store(settings)
    retriever = make_retriever(settings, store)
    report = run_retrieval_eval(retriever, eval_file, limit=limit)
    console.print({key: value for key, value in report.items() if key != "failures"})
    if report["failures"]:
        console.print(render_retrieval_failures(report["failures"], limit))


@app.command()
def qdrant_preview(
    limit: int = 2,
    with_vectors: Annotated[
        bool,
        typer.Option(help="是否打印向量内容；默认不打印，因为向量通常很长。"),
    ] = False,
) -> None:
    """打印 Qdrant collection 信息和前几条 point。"""

    from dish_rag.config import get_settings
    from dish_rag.storage.qdrant_store import QdrantRecipeIndex

    settings = get_settings(Path.cwd())
    qdrant = QdrantRecipeIndex(
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection,
        settings.qdrant_path,
    )

    collections = qdrant.client.get_collections()
    console.print("[bold]Collections[/bold]")
    console.print(collections)

    collection_info = qdrant.client.get_collection(collection_name=settings.qdrant_collection)
    console.print("[bold]Collection Info[/bold]")
    console.print(collection_info)

    points, next_page = qdrant.client.scroll(
        collection_name=settings.qdrant_collection,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )
    preview = {
        "collection": settings.qdrant_collection,
        "next_page": str(next_page) if next_page else None,
        "points": [
            {
                "id": point.id,
                "payload": point.payload,
                "vector": point.vector if with_vectors else "未打印；需要时加 --with-vectors",
            }
            for point in points
        ],
    }
    console.print_json(json.dumps(preview, ensure_ascii=False, default=str))


if __name__ == "__main__":
    app()
