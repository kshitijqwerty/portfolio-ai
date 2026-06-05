#!/usr/bin/env python3
"""
End-to-end smoke test for the portfolio Q&A pipeline.

Checks:
  1. Redis is reachable
  2. Qdrant is reachable and has indexed data
  3. llama.cpp server is reachable and can generate
  4. Full pipeline: embed → retrieve → prompt → generate

Usage:  python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from app.cache import SemanticCache
from app.config import settings
from app.llm import LLMClient
from app.prompt import build_prompt
from app.rag import RAGEngine


async def main() -> None:
    errors = 0

    # 1. Redis
    print("[1/4] Checking Redis ...", end=" ")
    try:
        cache = SemanticCache()
        await cache.connect()
        # Quick ping via set/get
        await cache._client.set("health:test", "ok", ex=5)
        val = await cache._client.get("health:test")
        assert val == "ok", "Redis readback failed"
        print("OK")
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # 2. Qdrant
    print("[2/4] Checking Qdrant ...", end=" ")
    try:
        rag = RAGEngine()
        rag.load_embedder()
        await rag.connect()
        collections = await rag._client.get_collections()
        names = [c.name for c in collections.collections]
        if settings.qdrant_collection not in names:
            print(f"WARN — collection '{settings.qdrant_collection}' not found; no data indexed")
        else:
            count = await rag._client.count(collection_name=settings.qdrant_collection)
            print(f"OK — {count.count} vectors in '{settings.qdrant_collection}'")
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # 3. llama.cpp
    print("[3/4] Checking llama.cpp ...", end=" ")
    try:
        llm = LLMClient()
        tokens = []
        async for token in llm.generate_stream("Hi."):
            tokens.append(token)
            break  # Just one token to verify connectivity
        if tokens:
            print(f"OK — generated '{tokens[0]}'")
        else:
            print("WARN — empty response")
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # 4. Full pipeline
    print("[4/4] Full pipeline (embed → retrieve → generate) ...")
    try:
        question = "What did Kshitij work on at CARS24?"
        embed = rag.embed(question)
        chunks = await rag.retrieve(embed)
        context = rag.format_context(chunks)
        prompt = build_prompt(question, context)

        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Context chunks: {len(chunks)}")
        print(f"  Generating answer ...")

        full = ""
        async for token in llm.generate_stream(prompt):
            full += token
        print(f"  Answer: {full[:200]}...")
        print(" OK")
    except Exception as e:
        print(f"  FAIL — {e}")
        errors += 1

    # Cleanup
    await cache.disconnect()
    await rag.disconnect()
    await llm.close()

    print(f"\n{'All checks passed!' if errors == 0 else f'{errors} check(s) failed.'}")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
