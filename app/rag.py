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

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings


class RAGEngine:
    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None

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
                "id": hash(chunk["id"]),
                "vector": vec.tolist(),
                "payload": {
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "title": chunk["title"],
                },
            })

        await self._client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

    async def retrieve(self, question_embedding: np.ndarray) -> list[dict]:
        if not self._client:
            raise RuntimeError("Qdrant client not connected")

        results = await self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=question_embedding.tolist(),
            limit=settings.rag_top_k,
        )

        return [
            {
                "text": r.payload["text"],
                "section": r.payload["section"],
                "title": r.payload["title"],
                "score": r.score,
            }
            for r in results
        ]

    def format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c['section']}"
            if c.get("title"):
                header += f" — {c['title']}"
            parts.append(f"{header}\n{c['text']}")
        return "\n\n".join(parts)
