#!/usr/bin/env python3
"""
Index resume chunks into Qdrant.

Usage:  python scripts/index_resume.py

Reads data/resume.json, embeds each chunk with all-MiniLM-L6-v2,
and upserts into the Qdrant "portfolio" collection.

Safe to re-run: upsert is idempotent (same chunk id → same point id).
"""

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from app.rag import RAGEngine


async def main() -> None:
    rag = RAGEngine()
    rag.load_embedder()
    await rag.connect()

    chunks_path = Path("data/resume.json")
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run from project root.", file=sys.stderr)
        sys.exit(1)

    with open(chunks_path) as f:
        chunks = json.load(f)

    print(f"Indexing {len(chunks)} chunks into Qdrant ...")
    await rag.index_chunks(chunks)

    # Verify by searching for a sample query
    test_query = "What did Kshitij build at CARS24?"
    embedding = rag.embed(test_query)
    results = await rag.retrieve(embedding)
    print(f"\nVerification search for: {test_query}")
    print(f"Top-{len(results)} results:")
    for r in results:
        print(f"  [{r['score']:.3f}] ({r['section']}) {r['title']}")
        print(f"       {r['text'][:100]}...")

    await rag.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
