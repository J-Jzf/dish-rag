"""菜谱 RAG 项目的命令行入口。"""

from pathlib import Path
from typing import Annotated
import json

import typer
from rich.console import Console

app = typer.Typer(help="菜谱 Agentic RAG CLI") # 创建一个命令行应用对象。
# help 是这个命令行工具的说明文字，运行python main.py --help命令时会显示。
console = Console() # 创建一个 Rich 的终端输出对象，用来更漂亮地打印内容，比如彩色表格、面板、格式化文本。比普通 print() 更强。


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
    thread_id: str = "default",
    user_id: str = "default",
) -> None:
    """运行一轮 LangGraph 对话。"""

    from dish_rag.config import get_settings
    from dish_rag.factory import make_graph
    from dish_rag.models import TurnTrace
    from dish_rag.observability import render_trace

    settings = get_settings(Path.cwd())
    graph = make_graph(settings)
    result = graph.invoke(
        {
            "thread_id": thread_id,
            "user_id": user_id,
            "user_query": query,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    if "__interrupt__" in result:
        console.print(result["__interrupt__"])
        return
    console.print(result.get("answer", ""))
    if result.get("trace"):
        trace = result["trace"]
        render_trace(trace if isinstance(trace, TurnTrace) else TurnTrace.model_validate(trace))


@app.command()
def eval_retrieval(
    eval_file: Path = Path("configs/eval_queries.jsonl"),
    limit: int = 8,
) -> None:
    """运行离线检索评测指标。"""

    from dish_rag.config import get_settings
    from dish_rag.eval.offline import run_retrieval_eval
    from dish_rag.factory import make_retriever, make_store

    settings = get_settings(Path.cwd())
    store = make_store(settings)
    retriever = make_retriever(settings, store)
    metrics = run_retrieval_eval(retriever, eval_file, limit=limit)
    console.print(metrics)


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
