"""
Semantic cache layer.

Cache key design:
  We quantise the embedding vector to 2 decimal places before hashing.
  This buckets semantically similar questions together so that
  "what did you build at CARS24" and "CARS24 work?" resolve to the
  *same* cache entry even though the raw text differs.

  Steps:
    1. Embed the question with all-MiniLM-L6-v2 → 384-dim float32 vector
    2. np.round(_, 2)     → coarse quantisation (0.01 buckets)
    3. SHA256(.tobytes()) → fixed-length hex digest
    4. Prepend "chat:"    → Redis key

  RAM note: we keep this small — no question history, no hot reload.
  A single 384-elem float32 vector is ~1.5 KB for the duration of
  the cache-key computation; negligible.
"""

import hashlib
import json
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

from app.config import settings


class SemanticCache:
    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()

    def _make_key(self, embedding: np.ndarray) -> str:
        quantised = np.round(embedding, 2)
        digest = hashlib.sha256(quantised.tobytes()).hexdigest()
        return f"{settings.redis_cache_prefix}{digest}"

    async def get(self, embedding: np.ndarray) -> Optional[dict]:
        key = self._make_key(embedding)
        raw = await self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, embedding: np.ndarray, data: dict, ttl: Optional[int] = None) -> None:
        key = self._make_key(embedding)
        await self._client.setex(
            key,
            ttl or settings.redis_ttl_seconds,
            json.dumps(data),
        )
