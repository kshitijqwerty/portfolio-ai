"""
FastAPI entrypoint for portfolio Q&A.

Lifespan-managed resources:
  - RAGEngine  (Qdrant client + sentence-transformers embedder)
  - SemanticCache (Redis async client)
  - LLMClient  (httpx client for llama.cpp)

SSE streaming protocol:
  data: {"token": "hello"}

  data: {"token": " world"}

  data: [DONE]
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.cache import SemanticCache
from app.config import settings
from app.llm import LLMClient
from app.prompt import build_prompt
from app.rag import RAGEngine

rag = RAGEngine()
cache = SemanticCache()
llm = LLMClient()


class ChatRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    rag.load_embedder()
    await rag.connect()
    await cache.connect()
    yield
    # Shutdown
    await rag.disconnect()
    await cache.disconnect()
    await llm.close()


app = FastAPI(title="portfolio-ai", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, settings.cors_origin_alt],
    allow_methods=["POST"],
    allow_headers=["*"],
)


async def _stream_cached(answer: str) -> AsyncGenerator[str, None]:
    """Yield cached answer token-by-token with 8 ms delay per token."""
    for token in _tokenize(answer):
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.008)
    yield "data: [DONE]\n\n"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace-aware tokenisation for SSE playback."""
    tokens: list[str] = []
    for word in text.split(" "):
        if tokens:
            tokens.append(" ")
        tokens.append(word)
    return tokens


@app.post("/chat")
async def chat(body: ChatRequest):
    question = body.question.strip()
    if not question:
        return StreamingResponse(
            _stream_cached("Please ask a question about Kshitij's portfolio."),
            media_type="text/event-stream",
        )

    embed = rag.embed(question)

    # 1. Check semantic cache
    cached = await cache.get(embed)
    if cached is not None:
        return StreamingResponse(
            _stream_cached(cached["answer"]),
            media_type="text/event-stream",
        )

    # 2. RAG retrieval
    chunks = await rag.retrieve(embed)
    context = rag.format_context(chunks) if chunks else ""

    # 3. Build prompt and stream from LLM
    prompt = build_prompt(question, context)

    async def generate():
        full_answer = ""
        async for token in llm.generate_stream(prompt):
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

        # 4. Write to cache asynchronously
        if chunks:
            asyncio.create_task(
                cache.set(embed, {
                    "answer": full_answer,
                    "chunks_used": [c["text"] for c in chunks],
                })
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
