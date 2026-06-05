#!/usr/bin/env bash
# One-shot deploy script for portfolio-ai.
#
# Usage:  curl -fsSL https://raw.githubusercontent.com/kshitijqwerty/portfolio-ai/main/scripts/deploy_lxc.sh | bash
#
# Or copy and paste directly into the server terminal.
#
# This script:
#   1. Installs Docker (official convenience script) if missing
#   2. Installs build deps (git, curl, build-essential)
#   3. Clones the repo
#   4. Downloads the GGUF model
#   5. Builds Docker images (llama.cpp from source)
#   6. Starts all containers
#   7. Indexes resume into Qdrant
#   8. Runs the smoke test

set -euo pipefail

REPO="https://github.com/kshitijqwerty/portfolio-ai.git"
BRANCH="main"
INSTALL_DIR="/opt/portfolio-ai"

echo "=== portfolio-ai Deploy ==="

# ─── 1. System dependencies ────────────────────────────────────────────────
echo "[1/7] Installing system dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl git build-essential >/dev/null

# ─── 2. Docker ──────────────────────────────────────────────────────────────
echo "[2/7] Ensuring Docker is installed ..."
if ! command -v docker &>/dev/null; then
    echo "  Docker not found — installing via convenience script ..."
    curl -fsSL https://get.docker.com | sudo bash
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. (You may need to re-login for group changes.)"
else
    echo "  Docker already installed."
fi

# ─── 3. Clone repo ─────────────────────────────────────────────────────────
echo "[3/7] Cloning repository ..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  $INSTALL_DIR already exists — pulling latest ..."
    cd "$INSTALL_DIR"
    git pull origin "$BRANCH"
else
    sudo mkdir -p "$(dirname "$INSTALL_DIR")"
    sudo git clone --branch "$BRANCH" "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# ─── 4. Download model ─────────────────────────────────────────────────────
echo "[4/7] Downloading GGUF model (Qwen2-0.5B Q4_K_M, ~350 MB) ..."
make install-models

# ─── 5. Copy env ────────────────────────────────────────────────────────────
echo "[5/7] Setting up .env ..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example — edit CLOUDFLARE_TOKEN if needed."
fi

# ─── 6. Build & start Docker containers ────────────────────────────────────
echo "[6/7] Building Docker images (llama.cpp from source — ~2 min) ..."
make docker-build
echo "  Starting containers ..."
make docker-up

echo "  Waiting for services to be ready ..."
sleep 5

# ─── 7. Index & smoke test ─────────────────────────────────────────────────
echo "[7/7] Indexing resume into Qdrant ..."
make docker-index

echo "  Running smoke test ..."
make docker-test

echo ""
echo "=== Deploy complete ==="
echo ""
echo "  API:   http://127.0.0.1:8000"
echo "  Chat:  curl -X POST http://127.0.0.1:8000/chat"
echo "         -H 'Content-Type: application/json'"
echo "         -d '{\"question\":\"What did Kshitij build at CARS24?\"}'"
echo ""
echo "  Logs:  docker compose logs -f"
echo ""
echo "Optional — auto-start on boot:"
echo "  sudo make deploy-systemd"
