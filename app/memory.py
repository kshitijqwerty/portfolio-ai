"""
Conversation memory — stores last N Q&A exchanges per session in Redis.

Key space:
  session:<uuid> → JSON list of message dicts
    [{"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}]

TTL is refreshed on every access.
"""

from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis

from app.config import settings


class ConversationMemory:
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

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def get_history(self, session_id: str) -> list[dict]:
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return []
        history = json.loads(raw)
        await self._client.expire(self._key(session_id), settings.memory_session_ttl)
        return history

    async def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        key = self._key(session_id)
        raw = await self._client.get(key)
        history = json.loads(raw) if raw else []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        max_msgs = settings.memory_max_exchanges * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        await self._client.setex(key, settings.memory_session_ttl, json.dumps(history))
