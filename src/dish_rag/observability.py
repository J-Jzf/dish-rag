"""结构化可观测性辅助函数。

Agent 每一轮会返回普通回答和机器可读 trace。本模块保证 CLI、测试和
未来 Web 前端使用同一套 trace 记录格式。
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dish_rag.models import TurnTrace


def trace_to_json(trace: TurnTrace) -> str:
    """序列化一轮 trace，并保留中文字符。"""

    return trace.model_dump_json(indent=2)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """向审计文件追加一行 JSON。"""

    # 在这里创建父目录，调用方就不用在每个命令里提前准备输出目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_trace(trace: TurnTrace) -> None:
    """在终端里格式化展示最重要的 trace 字段。"""

    console = Console()
    table = Table(title="RAG Turn Trace", show_lines=True)
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Raw Query", trace.raw_query)
    table.add_row("Intent", trace.parsed_intent)
    table.add_row("Completed Query", trace.completed_query)
    table.add_row("Rewritten Query", trace.rewritten_query)
    table.add_row("State Before", json.dumps(trace.state_before, ensure_ascii=False))
    table.add_row("State After", json.dumps(trace.state_after, ensure_ascii=False))

    if trace.evidence_judge:
        table.add_row("Evidence Judge", trace.evidence_judge.model_dump_json())

    console.print(table)

    # 命中结果可能比较长，所以单独放在一个紧凑面板里展示。
    hits = [
        {
            "chunk_id": hit.chunk_id,
            "recipe": f"{hit.recipe_id} {hit.recipe_name}",
            "field": hit.field,
            "score": hit.score,
            "source": hit.source,
            "filters": hit.filters,
        }
        for hit in trace.qdrant_hits
    ]
    console.print(Panel(json.dumps(hits, ensure_ascii=False, indent=2), title="Search Hits"))
