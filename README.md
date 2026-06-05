# portfolio-ai

Self-hosted portfolio Q&A system for [kgup.me](https://kgup.me). Answers natural language questions about Kshitij Gupta's professional experience using fully offline inference — runs entirely on your own hardware with no cloud dependencies.

## Architecture

```
                        Cloudflare Tunnel
                              │
                         (public)
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │                   Docker Compose                      │
  │  ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
  │  │  Redis    │   │  Qdrant  │   │  llama.cpp       │  │
  │  │  alpine   │   │  :6333   │   │  built from src  │  │
  │  └────┬─────┘   └────┬─────┘   └────────┬─────────┘  │
  │       │              │                   │            │
  │       └──────┬───────┘                   │            │
  │              │                           │            │
  │       ┌──────▼───────────────────────────▼─────┐      │
  │       │   FastAPI (uvicorn) — python:3.11-slim │      │
  │       │       /chat  SSE endpoint              │      │
  │       └────────────────────────────────────────┘      │
  │                       │                               │
  │              127.0.0.1:8000                            │
  └───────────────────────────────────────────────────────┘
```

### Pipeline

```
User question
     │
     ▼
  Embed (all-MiniLM-L6-v2) ───→ Redis semantic cache
     │                              │
     ▼                              │
  Qdrant top-3 RAG                  │
     │                              │
     ▼                              │
  Build prompt (ChatML)             │
     │                              │
     ▼                              │
  llama.cpp inference ──────────────┤
     │                              │
     ▼                              ▼
  SSE stream to browser ───→ Cache write (async)
```

## Prerequisites

- Linux with Docker and `docker compose` plugin
- ~2 GB available RAM
- ~1.5 GB free disk
- `curl`, `git` installed

## Setup

```bash
# 1. Clone and enter the repository
git clone https://github.com/kshitijqwerty/portfolio-ai.git
cd portfolio-ai

# 2. Copy environment
cp .env.example .env
# Edit CLOUDFLARE_TOKEN in .env if using Cloudflare Tunnel

# 3. Download model(s)
make install-models
# Downloads Qwen2-0.5B GGUF into models/

# 4. Build Docker images
make docker-build
# Builds llama.cpp (from source) + app image

# 5. Start all services
make docker-up

# 6. Index resume chunks into Qdrant
make docker-index

# 7. Smoke test
make docker-test
```

### Auto-start on boot

```bash
sudo make deploy-systemd
# Installs deploy/docker-compose.service and enables it
```

### Cloudflare Tunnel

```bash
# On the host (not in Docker)
cloudflared tunnel login
cloudflared tunnel create portfolio-ai
# Configure DNS in cloudflare dashboard
cloudflared tunnel run portfolio-ai
```

## API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What did Kshitij build at CARS24?"}'
```

Response is an SSE stream:

```
data: {"token": "Kshitij"}
data: {"token": " built"}
data: {"token": " an"}
...
data: [DONE]
```

## Design Decisions

### Semantic Cache Key

Cache key = `SHA256(round(embedding, 2).tobytes())`. Quantising to 2 decimal places before hashing buckets semantically similar questions together. "What did you build at CARS24" and "CARS24 work?" will produce nearly identical embeddings, and after rounding they hash to the same key. TTL is 7 days.

### Model Selection

Qwen2-0.5B-Instruct Q4_K_M or SmolLM2-360M-Instruct Q4_K_M. Both fit within the available memory with room for the KV cache (2048 context window).
