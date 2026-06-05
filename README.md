# portfolio-ai

Self-hosted portfolio Q&A system for [kgup.me](https://kgup.me). Answers natural language questions about Kshitij Gupta's professional experience using fully offline inference on a Proxmox LXC container (Intel i3-8100T, ~2.5 GB RAM).

## Architecture

```
                        Cloudflare Tunnel
                              │
                         (public)
                              ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    LXC Container                          │
  │                    Docker Compose                         │
  │  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐  │
  │  │  Redis    │   │  Qdrant  │   │  llama.cpp           │  │
  │  │  alpine   │   │  :6333   │   │  built from source   │  │
  │  │  ~50 MB  │   │ ~150 MB  │   │  ~900 MB             │  │
  │  └────┬─────┘   └────┬─────┘   └─────────┬────────────┘  │
  │       │              │                    │               │
  │       └──────┬───────┘                    │               │
  │              │                            │               │
  │       ┌──────▼────────────────────────────▼──────┐        │
  │       │   FastAPI (uvicorn) — python:3.11-slim   │        │
  │       │         /chat  SSE endpoint              │        │
  │       │         ~120 MB                          │        │
  │       └──────────────────────────────────────────┘        │
  │                       │                                   │
  │              127.0.0.1:8000                                │
  └───────────────────────────────────────────────────────────┘
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

### RAM Budget

| Component   | Budget  |
|-------------|---------|
| llama.cpp   | ~900 MB |
| Qdrant      | ~150 MB |
| Redis       |  ~50 MB |
| FastAPI     | ~120 MB |
| OS + LXC    | ~400 MB |
| **Total**   | 1.62 GB |

## Prerequisites

- Ubuntu 24.04 LXC with Docker installed
- 2.5 GB available RAM
- ~1.5 GB free disk for models + Docker images
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
# Builds llama.cpp (from source, ~2 min on i3-8100T) + app image

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

### Docker image sizes

| Image | Size | Notes |
|-------|------|-------|
| `redis:alpine` | ~5 MB | Off the shelf |
| `qdrant/qdrant` | ~180 MB | Off the shelf |
| `portfolio-ai-llamacpp` | ~200 MB | Multi-stage: build deps stripped |
| `portfolio-ai-app` | ~1.1 GB | Includes model cache + deps |

### Model Selection

Qwen2-0.5B-Instruct Q4_K_M (~350 MB) or SmolLM2-360M-Instruct Q4_K_M (~230 MB). Both fit comfortably in the 900 MB llama.cpp budget with room for the KV cache (2048 context window).
