"""
llama.cpp HTTP client with SSE streaming support.

llama.cpp server exposes an OpenAI-compatible /v1/chat/completions endpoint.
We send full message arrays (system + history + user) and let the server
apply the chat template — this avoids double-wrapping and enables
conversation memory.

RAM note: we use httpx for async HTTP. The client is stateless — the
~filestore RAM footprint lives in the llama.cpp server process.
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

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Send a chat completion request with stream=True and yield tokens
        as they arrive from the SSE stream.

        `messages` must follow the OpenAI message format:
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "..."},
             {"role": "assistant", "content": "..."},
             {"role": "user", "content": "current question"}]

        The caller is responsible for including system prompt, conversation
        history, and the current user question.
        """
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "repeat_penalty": settings.repeat_penalty,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "min_p": settings.min_p,
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

    async def generate(self, messages: list[dict]) -> str:
        """Non-streaming variant — collects the full response."""
        tokens = []
        async for token in self.generate_stream(messages):
            tokens.append(token)
        return "".join(tokens)
