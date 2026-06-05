FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Build deps for sentence-transformers + numpy
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST so sentence-transformers doesn't pull
# the full CUDA package (~2 GB). The default torch wheel includes cuDNN,
# NCCN, and 100+ MB of GPU libs — all dead weight on a CPU-only LXC.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install Python deps first (cache-friendly layer ordering)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-cache the embedding model at build time so startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
