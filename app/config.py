from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # llama.cpp — default to Docker service name
    llm_host: str = "llamacpp"
    llm_port: int = 8080
    llm_model: str = "models/model.gguf"

    # Qdrant — default to Docker service name
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "portfolio"

    # Redis — default to Docker service name
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_ttl_seconds: int = 604800  # 7 days
    redis_cache_prefix: str = "chat:"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG
    rag_top_k: int = 5
    rag_diverse_pool_size: int = 15  # fetch N, then pick diverse subset
    rag_min_score: float = 0.3       # skip LLM if top chunk scores below this

    # Cross-encoder reranking
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Conversation memory
    memory_max_exchanges: int = 3
    memory_session_ttl: int = 3600  # 1 hour cookie + Redis TTL

    # Sampling
    temperature: float = 0.1
    repeat_penalty: float = 1.15
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    max_tokens: int = 512

    # CORS
    cors_origin: str = "https://kgup.me"
    cors_origin_alt: str = "http://localhost:5173"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Cloudflare
    cloudflare_token: str = ""


settings = Settings()
