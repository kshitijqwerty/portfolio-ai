"""
FastAPI entrypoint for portfolio Q&A with conversation memory.

Lifespan-managed resources:
  - RAGEngine       (Qdrant client + embedder + cross-encoder reranker)
  - SemanticCache   (Redis async client)
  - ConversationMemory (Redis async client for session history)
  - LLMClient       (httpx client for llama.cpp)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.cache import SemanticCache
from app.config import settings
from app.llm import LLMClient
from app.memory import ConversationMemory
from app.prompt import build_prompt
from app.query import is_vague_query, rewrite_query
from app.rag import RAGEngine

rag = RAGEngine()
cache = SemanticCache()
memory = ConversationMemory()
llm = LLMClient()


class ChatRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.load_embedder()
    rag.load_reranker()
    await rag.connect()
    await cache.connect()
    await memory.connect()
    yield
    await rag.disconnect()
    await cache.disconnect()
    await memory.disconnect()
    await llm.close()


app = FastAPI(title="portfolio-ai", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, settings.cors_origin_alt],
    allow_methods=["POST"],
    allow_headers=["*"],
    allow_credentials=True,
)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in text.split(" "):
        if tokens:
            tokens.append(" ")
        tokens.append(word)
    return tokens


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    # Session resolution
    session_id = request.cookies.get("session_id")
    is_new = not session_id
    if is_new:
        session_id = str(uuid.uuid4())

    history = await memory.get_history(session_id)

    question = body.question.strip()
    if not question:
        return StreamingResponse(
            _stream_cached("Please ask a question about Kshitij's portfolio."),
            media_type="text/event-stream",
            headers={"Set-Cookie": _cookie(session_id)} if is_new else None,
        )

    # Rewrite for better retrieval, then embed
    search_query = rewrite_query(question)
    embed = rag.embed(search_query)

    # Skip cache when there's conversation history (context matters)
    if not history:
        cached = await cache.get(embed)
        if cached is not None:
            return StreamingResponse(
                _stream_cached(cached["answer"]),
                media_type="text/event-stream",
                headers={"Set-Cookie": _cookie(session_id)} if is_new else None,
            )

    # RAG retrieval
    vague = is_vague_query(question)
    if vague:
        chunks = await rag.diverse_retrieve(embed)
    else:
        chunks = await rag.retrieve(embed)

    # Rerank with cross-encoder
    if chunks:
        chunks = rag.rerank(search_query, chunks)

    # Relevance guard (skip for vague questions)
    if not chunks:
        return StreamingResponse(
            _stream_cached(NO_RESULT),
            media_type="text/event-stream",
            headers={"Set-Cookie": _cookie(session_id)} if is_new else None,
        )
    if not vague and chunks[0]["score"] < settings.rag_min_score:
        return StreamingResponse(
            _stream_cached(NO_RESULT),
            media_type="text/event-stream",
            headers={"Set-Cookie": _cookie(session_id)} if is_new else None,
        )

    # Ensure Summary chunk is present for vague questions
    if vague and not any(c["section"] == "Summary" for c in chunks):
        sum_embed = rag.embed("Kshitij Gupta professional summary background overview")
        sum_chunks = await rag.retrieve(sum_embed)
        for c in sum_chunks:
            if c["section"] == "Summary":
                chunks.insert(0, c)
                chunks = chunks[: settings.rag_top_k]
                break

    context = rag.format_context(chunks)
    system_prompt = build_prompt(context)

    # Build message array: system + history + current question
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    async def generate():
        full_answer = ""

        # Emit session_id as first event for new sessions
        if is_new:
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"

        async for token in llm.generate_stream(messages):
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield "data: [DONE]\n\n"

        # Save exchange to memory (fire-and-forget)
        asyncio.create_task(memory.add_exchange(session_id, question, full_answer))

        # Cache only first interaction (no history)
        if not history and chunks:
            asyncio.create_task(
                cache.set(embed, {
                    "answer": full_answer,
                    "chunks_used": [c["text"] for c in chunks],
                })
            )

    resp = StreamingResponse(generate(), media_type="text/event-stream")
    if is_new:
        resp.set_cookie(
            key="session_id",
            value=session_id,
            max_age=settings.memory_session_ttl,
            httponly=True,
            samesite="lax",
        )
    return resp


NO_RESULT = (
    "I don't have information about that in my knowledge base. "
    "Try asking about Kshitij's experience, skills, or projects."
)


def _cookie(session_id: str) -> str:
    return (
        f"session_id={session_id}; "
        f"Max-Age={settings.memory_session_ttl}; "
        f"Path=/; HttpOnly; SameSite=Lax"
    )


async def _stream_cached(answer: str) -> AsyncGenerator[str, None]:
    for token in _tokenize(answer):
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.008)
    yield "data: [DONE]\n\n"
