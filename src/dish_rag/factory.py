"""CLI 命令使用的对象组装函数。"""

from dish_rag.agent.graph import build_graph
from dish_rag.agent.nodes import AgentNodes
from dish_rag.config import Settings
from dish_rag.llm.opai import ChatClient, EmbeddingClient, RerankClient, SparseBM25Encoder
from dish_rag.retrieval.hybrid import HybridRetriever
from dish_rag.retrieval.name_index import RecipeNameIndex
from dish_rag.storage.qdrant_store import QdrantRecipeIndex
from dish_rag.storage.sqlite_store import SQLiteStore


def make_store(settings: Settings) -> SQLiteStore:
    """创建并迁移 SQLite 事实库。"""

    store = SQLiteStore(settings.sqlite_path)
    store.migrate()
    return store


def make_retriever(settings: Settings, store: SQLiteStore) -> HybridRetriever:
    """创建 search 命令和图节点共用的混合检索器。"""

    return HybridRetriever(
        store=store,
        name_index=RecipeNameIndex(store),
        qdrant=QdrantRecipeIndex(
            settings.qdrant_url,
            settings.qdrant_api_key,
            settings.qdrant_collection,
            settings.qdrant_path,
        ),
        embeddings=EmbeddingClient(
            settings.embedding_model_id,
            settings.embedding_api_key,
            settings.embedding_base_url,
        ),
        sparse_encoder=SparseBM25Encoder(),
        reranker=RerankClient(
            settings.rerank_model_id,
            settings.rerank_api_key,
            settings.rerank_base_url,
        ),
    )


def make_graph(settings: Settings):
    """创建已经编译好的 LangGraph 应用。"""

    store = make_store(settings)
    retriever = make_retriever(settings, store)
    chat = ChatClient(settings.llm_model_id, settings.llm_api_key, settings.llm_base_url)
    nodes = AgentNodes(chat, retriever, store)
    return build_graph(nodes, settings.langgraph_checkpoint_db)
