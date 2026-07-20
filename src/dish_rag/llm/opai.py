"""OpenAI-compatible 的对话、向量和 rerank 客户端。

项目统一调用这些 wrapper，避免把 SDK 调用散落在 Agent 代码里。这样以后
更换模型服务端点时，LangGraph 图结构不用跟着改。
"""

from __future__ import annotations

from typing import Any

import httpx
from openai import OpenAI


class ChatClient:
    """OpenAI-compatible Chat Completions 的轻量封装。"""

    def __init__(self, model_id: str, api_key: str, base_url: str) -> None:
        """创建对话客户端。"""

        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """让模型返回 JSON 对象，并解析成字典。"""

        import json

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def complete_text(self, system: str, user: str) -> str:
        """让模型返回普通文本。"""

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class EmbeddingClient:
    """OpenAI-compatible embedding 客户端。"""

    def __init__(self, model_id: str, api_key: str, base_url: str) -> None:
        """创建向量客户端。"""

        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把一个或多个文本转成向量。"""

        response = self.client.embeddings.create(model=self.model_id, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """把单条 Query 转成向量。"""

        return self.embed_texts([query])[0]


class SparseBM25Encoder:
    """用于 Qdrant 稀疏向量的 FastEmbed BM25 编码器。"""

    def __init__(self) -> None:
        """延迟加载 FastEmbed 的 BM25 模型。"""

        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def encode_documents(self, texts: list[str]) -> list[object]:
        """把文档编码成可写入 Qdrant 的稀疏向量。"""

        return list(self.model.embed(texts))

    def encode_query(self, text: str) -> object:
        """把单条 Query 编码成 Qdrant 稀疏检索向量。"""

        return list(self.model.query_embed(text))[0]


class RerankClient:
    """兼容性较强的 HTTP rerank 客户端。

    不同服务商的 JSON 返回结构略有不同。本方法会尝试兼容常见格式；
    如果没有配置 rerank 端点，就按原始顺序返回一个兜底排序。
    """

    def __init__(self, model_id: str, api_key: str, base_url: str) -> None:
        """创建 rerank 适配器。"""

        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """返回按相关性排序的 `(文档下标, 分数)`。"""

        if not self.base_url or not self.model_id or not documents:
            return [(index, 1.0 / (index + 1)) for index in range(len(documents))]

        payload = {"model": self.model_id, "query": query, "documents": documents}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=30) as client:
            response = client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results") or data.get("data") or []
        ranked: list[tuple[int, float]] = []
        for result in results:
            index = result.get("index", result.get("document_index", 0))
            score = result.get("relevance_score", result.get("score", 0.0))
            ranked.append((int(index), float(score)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)
