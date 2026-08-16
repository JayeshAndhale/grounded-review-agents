from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Everything provider-specific lives here so the rest
    of the codebase never hardcodes a model name or key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_model_cheap: str = "groq/llama-3.1-8b-instant"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "./data/chroma"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 5
    max_revision_loops: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()