"""
llama.cpp HTTP client with SSE streaming support.

llama.cpp server exposes an OpenAI-compatible /v1/chat/completions endpoint.
We use the streaming variant (stream=True) and yield tokens one by one.

RAM note: we use httpx for async HTTP. The client is stateless — the ~900 MB
RAM footprint lives in the llama.cpp server process, not in this Python process.
"""

from __future__ import annotations

from typing import AsyncGenerator

import httpx
import json

from app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self._base_url = f"http://{settings.llm_host}:{settings.llm_port}"
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120)

    async def close(self) -> None:
        await self._client.aclose()

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Send a chat completion request with stream=True and yield tokens
        as they arrive from the SSE stream.
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.1,
            "stop": ["<|im_end|>"],
        }

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token

    async def generate(self, prompt: str) -> str:
        """Non-streaming variant — collects the full response."""
        tokens = []
        async for token in self.generate_stream(prompt):
            tokens.append(token)
        return "".join(tokens)
