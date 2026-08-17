"""Central config. Everything provider-specific lives here so the rest of
the codebase never hardcodes a model name, key, or client.
"""

from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Everything provider-specific lives here so the rest
    of the codebase never hardcodes a model name or key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    llm_model: str = "groq/openai/gpt-oss-120b"
    llm_model_cheap: str = "groq/openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "./data/chroma"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 5
    max_revision_loops: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm(tier: Literal["strong", "cheap"] = "strong") -> ChatGroq:
    """Tiered LLM client. Cached per-tier so nodes share one client
    instance instead of rebuilding it on every call.

    Strips the litellm-style 'groq/' prefix from the configured model name:
    Settings stores it that way because litellm (in requirements.txt) uses
    provider-prefixed routing, but the actual client here is langchain-groq's
    ChatGroq, which wants the bare model name.

    Returns the plain client, unwrapped by retry - retry has to be applied
    after .with_structured_output() at the call site (see with_backoff
    below), since RunnableRetry doesn't proxy that method.
    """
    settings = get_settings()
    model_name = settings.llm_model if tier == "strong" else settings.llm_model_cheap
    model_name = model_name.removeprefix("groq/")
    return ChatGroq(model=model_name, api_key=settings.groq_api_key)


def with_backoff(runnable):
    """Apply the project's standard exponential-backoff retry policy.
    Call this last, after .with_structured_output() if you're using it -
    RunnableRetry only implements the generic Runnable interface.
    """
    return runnable.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)