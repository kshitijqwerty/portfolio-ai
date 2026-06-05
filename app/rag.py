"""
RAG retrieval layer.

Qdrant collection schema:
  - collection: "portfolio"
  - vectors:    384-d float32, cosine distance
  - payload:    {"text": str, "section": str, "title": str}

RAM note: all-MiniLM-L6-v2 is ~22 MB when loaded via sentence-transformers.
We load it once at startup and keep it in memory for the lifetime of the
uvicorn worker. The embedding model is far smaller than the LLM itself.
"""

from typing import Optional
import uuid

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings


class RAGEngine:
    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        # Ensure collection exists at startup
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if settings.qdrant_collection not in names:
            await self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()

    def load_embedder(self) -> None:
        self._embedder = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )

    def load_reranker(self) -> None:
        if settings.rerank_enabled:
            self._reranker = CrossEncoder(
                settings.rerank_model,
                device="cpu",
            )

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        if not self._reranker or not results:
            return results
        pairs = [(query, r["text"]) for r in results]
        scores = self._reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["score"] = float(s)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text, normalize_embeddings=True)

    async def index_chunks(self, chunks: list[dict]) -> None:
        """Index a list of chunks into Qdrant.

        Each chunk: {"id": str, "text": str, "section": str, "title": str}
        """
        if not self._client:
            raise RuntimeError("Qdrant client not connected")
        if not self._embedder:
            raise RuntimeError("Embedder not loaded")

        points = []
        for chunk in chunks:
            vec = self._embedder.encode(chunk["text"], normalize_embeddings=True)
            points.append({
                # Qdrant rejects negative integers; UUID is clean and idempotent
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"])),
                "vector": vec.tolist(),
                "payload": {
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "title": chunk["title"],
                    "links": chunk.get("links", {}),
                },
            })

        await self._client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

    async def _query(self, embedding: np.ndarray, limit: int) -> list[dict]:
        response = await self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding.tolist(),
            limit=limit,
        )
        return [
            {
                "text": r.payload["text"],
                "section": r.payload["section"],
                "title": r.payload["title"],
                "links": r.payload.get("links", {}),
                "score": r.score,
            }
            for r in response.points
        ]

    async def retrieve(self, question_embedding: np.ndarray) -> list[dict]:
        if not self._client:
            raise RuntimeError("Qdrant client not connected")
        return await self._query(question_embedding, settings.rag_top_k)

    async def diverse_retrieve(self, question_embedding: np.ndarray) -> list[dict]:
        """Fetch a larger pool then select top result per section for breadth."""
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        results = await self._query(question_embedding, settings.rag_diverse_pool_size)
        if not results:
            return results

        seen_sections: set[str] = set()
        diverse: list[dict] = []
        remaining: list[dict] = []

        for r in results:
            sec = r["section"]
            if sec not in seen_sections:
                diverse.append(r)
                seen_sections.add(sec)
            else:
                remaining.append(r)

        slots_left = settings.rag_top_k - len(diverse)
        if slots_left > 0:
            diverse.extend(remaining[:slots_left])

        return diverse[:settings.rag_top_k]

    def format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c['section']}"
            if c.get("title"):
                header += f" — {c['title']}"
            text = c["text"]
            links = c.get("links", {})
            if links:
                link_str = "; ".join(f"{k}: {v}" for k, v in links.items())
                text += f"\nLinks: {link_str}"
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)
