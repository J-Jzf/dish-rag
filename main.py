"""菜谱 RAG 项目的直接运行入口。

在项目根目录运行命令，例如：

    python main.py ingest --no-index-qdrant
    python main.py search "宫保鸡丁怎么做"
    python main.py chat "我要做宫保鸡丁" --thread-id kitchen-001
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# 项目源码放在 `src/` 目录下。把它加入 `sys.path` 后，
# 就可以直接运行 `main.py`，不用把项目安装成 Python 包。
sys.path.insert(0, str(SRC_DIR))

from dish_rag.cli import app  # noqa: E402


if __name__ == "__main__":
    app()
