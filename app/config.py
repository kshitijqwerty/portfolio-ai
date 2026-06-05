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
    rag_top_k: int = 3

    # Sampling (prevents gibberish from 0.5B models)
    temperature: float = 0.1
    repeat_penalty: float = 1.15
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    max_tokens: int = 512

    # RAG relevance — skip LLM if top chunk scores below this
    rag_min_score: float = 0.3

    # CORS
    cors_origin: str = "https://kgup.me"
    cors_origin_alt: str = "http://localhost:5173"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Cloudflare
    cloudflare_token: str = ""


settings = Settings()
