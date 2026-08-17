"""配置加载。

所有运行时密钥都从环境变量或 `.env` 读取；源码里不硬编码模型密钥、
Qdrant 凭据或本地绝对路径。
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """入库、检索和 Agent 共用的类型化配置。"""

    # `env_file` 让本地开发可以使用 `.env`；
    # 生产环境也可以通过进程管理器直接注入真实环境变量。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 对话模型配置。接口按 OpenAI-compatible 形式调用。
    llm_model_id: str = Field(default="", alias="LLM_MODEL_ID")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")

    # 向量模型配置。单独配置是因为 embedding 常常和聊天模型使用不同服务。
    embedding_model_id: str = Field(default="", alias="EMBEDDING_MODEL_ID")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")

    # Rerank 模型配置。不同 rerank 服务的返回格式略有差异，
    # 具体兼容逻辑放在 wrapper 模块里。
    rerank_model_id: str = Field(default="", alias="RERANK_MODEL_ID")
    rerank_api_key: str = Field(default="", alias="RERANK_API_KEY")
    rerank_base_url: str = Field(default="", alias="RERANK_BASE_URL")

    # Qdrant 是搜索索引；下面的 SQLite 才是事实源。
    # 默认使用嵌入式本地 Qdrant。只有连接远程服务时才填写 `QDRANT_URL`。
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="dish_recipes", alias="QDRANT_COLLECTION")
    qdrant_path: Path = Field(default=Path("var/qdrant"), alias="QDRANT_PATH")
    semantic_cache_threshold: float = Field(default=0.85, alias="SEMANTIC_CACHE_THRESHOLD")

    # SQLite 保存菜谱原始事实和 LangGraph checkpoint。
    sqlite_path: Path = Field(default=Path("var/dish_rag.sqlite3"), alias="SQLITE_PATH")
    langgraph_checkpoint_db: Path = Field(
        default=Path("var/langgraph_checkpoints.sqlite3"),
        alias="LANGGRAPH_CHECKPOINT_DB",
    )

    # 入库路径默认相对于项目根目录。
    pdf_path: Path = Field(default=Path("data/菜谱_230道.pdf"), alias="PDF_PATH")
    build_dir: Path = Field(default=Path("build"), alias="BUILD_DIR")
    low_confidence_threshold: float = Field(default=0.95, alias="LOW_CONFIDENCE_THRESHOLD")


@lru_cache(maxsize=1) # 缓存装饰器 第一次执行时会真正读取 .env、解析配置、处理路径；之后如果用同样参数再调用，就直接返回上一次的结果，不再重复读取。
def get_settings(project_root: Path | None = None) -> Settings:
    """加载一次配置，并把相对路径归一化到项目根目录下。"""

    # 显式调用 `load_dotenv`，让 IDE 启动和终端启动时读取配置的行为一致。
    root = project_root or Path.cwd()
    load_dotenv(root / ".env")

    # `load_dotenv` 执行后，Pydantic 再从环境变量和 `.env` 中读取配置。
    settings = Settings()

    # 提前归一化路径，后续调用方就不用反复拼路径。
    for attr in ("sqlite_path", "langgraph_checkpoint_db", "pdf_path", "build_dir", "qdrant_path"):
        value = getattr(settings, attr)
        if not value.is_absolute():
            setattr(settings, attr, root / value)

    return settings
