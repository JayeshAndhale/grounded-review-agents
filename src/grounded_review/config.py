"""Central config. Everything provider-specific lives here so the rest of
the codebase never hardcodes a model name, key, or client.
"""

from functools import lru_cache
from typing import Literal

from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Everything provider-specific lives here so the rest
    of the codebase never hardcodes a model name or key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    cerebras_api_key: str = ""

    # Prefix before the first '/' selects the provider ('groq' or
    # 'cerebras'); everything after it is that provider's own model id,
    # passed through unmodified. Strong tier moved to Cerebras after
    # hitting Groq's daily TPD ceiling on openai/gpt-oss-120b during
    # Phase 5 testing - same model weights, different inference backend.
    llm_model: str = "cerebras/gpt-oss-120b"
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
def get_llm(tier: Literal["strong", "cheap"] = "strong") -> ChatGroq | ChatCerebras:
    """Tiered LLM client. Cached per-tier so nodes share one client
    instance instead of rebuilding it on every call.

    Dispatches to the right provider based on the prefix before the first
    '/' in the configured model string - 'groq' or 'cerebras' - so
    switching a tier's provider is a one-line config change, not a code
    change. Do NOT hardcode llama-3.3-70b on Cerebras: it was scheduled
    for deprecation Feb 16 2026, the same failure shape as the Groq
    deprecation hit in Phase 3. gpt-oss-120b is the current stable choice.
    """
    settings = get_settings()
    model_string = settings.llm_model if tier == "strong" else settings.llm_model_cheap
    provider, model_name = model_string.split("/", 1)

    if provider == "groq":
        return ChatGroq(model=model_name, api_key=settings.groq_api_key)
    elif provider == "cerebras":
        return ChatCerebras(model=model_name, api_key=settings.cerebras_api_key)
    else:
        raise ValueError(f"Unknown LLM provider prefix: {provider!r} in {model_string!r}")


def with_backoff(runnable):
    """Apply the project's standard exponential-backoff retry policy.

    Tuned for bulk evaluation workloads: stop_after_attempt=8 with a wait
    ceiling of 60s per attempt, worst-case ~3 minutes cumulative, to bridge
    brief provider-side rate-limit dips rather than crash the whole run.
    Call this last, after .with_structured_output() if you're using it -
    RunnableRetry only implements the generic Runnable interface.
    """
    return runnable.with_retry(
        stop_after_attempt=8,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 2, "max": 60, "exp_base": 2, "jitter": 3},
    )