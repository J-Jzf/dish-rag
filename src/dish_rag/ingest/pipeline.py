"""端到端知识库构建流水线。"""

from pathlib import Path

from dish_rag.config import Settings
from dish_rag.ingest.chunker import chunk_recipes
from dish_rag.ingest.exporter import (
    write_chunks_jsonl,
    write_recipes_json,
    write_recipes_markdown,
    write_review_report,
)
from dish_rag.ingest.parser import parse_recipes_from_pages
from dish_rag.ingest.pdf_extract import extract_pages, write_page_markdown
from dish_rag.llm.opai import EmbeddingClient, SparseBM25Encoder
from dish_rag.storage.qdrant_store import QdrantRecipeIndex
from dish_rag.storage.sqlite_store import SQLiteStore


def run_ingest(settings: Settings, index_qdrant: bool = True) -> dict[str, Path | int]:
    """解析菜谱 PDF、导出文件，并写入本地存储。"""

    build_dir = settings.build_dir
    pages = extract_pages(settings.pdf_path)
    write_page_markdown(pages, build_dir / "markdown" / "pdf_pages.md")

    recipes = parse_recipes_from_pages(pages) # 返回“每道菜”对象列表
    chunks = chunk_recipes(recipes) # 返回“每个 chunk”对象列表

    write_recipes_json(recipes, build_dir / "recipes.json")
    write_chunks_jsonl(chunks, build_dir / "chunks.jsonl")
    write_recipes_markdown(recipes, build_dir / "markdown" / "recipes.md")
    write_review_report(
        recipes,
        build_dir / "review" / "low_confidence_recipes.md",
        settings.low_confidence_threshold,
    )

    store = SQLiteStore(settings.sqlite_path)
    store.migrate()
    store.upsert_recipes(recipes)
    store.upsert_chunks(chunks) # 把 chunk 原文写入 SQLite 数据库

    if index_qdrant and settings.embedding_model_id:
        _index_qdrant(settings, chunks)

    return {
        "pages": len(pages),
        "recipes": len(recipes),
        "chunks": len(chunks),
        "recipes_json": build_dir / "recipes.json",
        "chunks_jsonl": build_dir / "chunks.jsonl",
        "review_report": build_dir / "review" / "low_confidence_recipes.md",
        "sqlite": settings.sqlite_path,
    }


def _index_qdrant(settings: Settings, chunks) -> None:
    """构建稠密向量和稀疏向量，并写入 Qdrant。"""

    texts = [chunk.text for chunk in chunks] # 取出所有 chunk 的正文文本
    embeddings = EmbeddingClient(
        settings.embedding_model_id,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )
    # 向量方向的匹配
    dense_vectors = embeddings.embed_texts(texts) # 把 chunk 文本转成稠密向量，每个 chunk 对应一个
    # “关键词匹配 + 权重打分”（有无重合的词，重合越多、越重要，分数越高）
    sparse_vectors = SparseBM25Encoder().encode_documents(texts) # 把 chunk 文本转成 BM25 稀疏向量（有indices和values，只记录“出现过且有检索意义的词项位置及其权重”），每个 chunk 对应一个

    qdrant = QdrantRecipeIndex( # 创建 Qdrant 索引客户端
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection,
        settings.qdrant_path,
    )
    qdrant.recreate_collection(dense_size=len(dense_vectors[0])) # 重新创建 Qdrant collection，指定稠密向量维度
    qdrant.upsert_chunks(chunks, dense_vectors, sparse_vectors)
