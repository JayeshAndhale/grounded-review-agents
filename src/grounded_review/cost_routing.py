# src/grounded_review/cost_routing.py
"""Provider-failover wrapper around get_llm(). Sits ABOVE with_backoff(),
not inside it — with_backoff() retries the SAME provider for transient
dips (confirmed working live in tonight's trace: a TPM 429 recovered on
retry:attempt:2). This module is for the case backoff can't fix: Groq's
daily token cap (TPD), confirmed in earlier sessions to be a multi-minute-
or-longer wall that no amount of short backoff bridges. When that happens,
route to Cerebras instead of continuing to hammer Groq.

Deliberately NOT merged into get_llm() itself: get_llm() is @lru_cache'd
per-tier and that caching is what makes it cheap to call repeatedly across
every node - it must keep returning a stable client. Provider-failover
logic needs to react to live failure state, which is incompatible with a
cached singleton. This wrapper is called at each invoke() site instead,
same pattern as with_backoff() itself.
"""

from typing import Any

from grounded_review.config import Settings, get_settings

# Substrings that identify a provider-level rate-limit failure, as opposed
# to a genuine application error (bad prompt, malformed schema, etc.) that
# failing over to a different provider would NOT fix and should not mask.
RATE_LIMIT_MARKERS = ("rate_limit_exceeded", "429", "tokens per day", "TPD")


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def invoke_with_failover(primary_tier: str, messages: list, **invoke_kwargs) -> Any:
    """Invoke the given tier's LLM; on a rate-limit-shaped failure (after
    with_backoff's own retries are exhausted), fail over to the OTHER
    provider for this one call, rather than raising.

    Deliberately does NOT retry the failover call again on a second
    failure - if the fallback provider also rate-limits, that's a real
    "both providers are out" condition that should surface as an error,
    not loop silently. Matches the project's existing stance (see
    evaluation/metrics.py's ResultStore) that a hard failure should be
    visible, not swallowed.
    """
    from grounded_review.config import get_llm, with_backoff

    settings = get_settings()
    primary_model_string = (
        settings.llm_model if primary_tier == "strong" else settings.llm_model_cheap
    )
    primary_provider = primary_model_string.split("/", 1)[0]

    try:
        llm = with_backoff(get_llm(primary_tier))
        return llm.invoke(messages, **invoke_kwargs)
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise  # a real error - failing over would hide the actual bug

        fallback_provider = "cerebras" if primary_provider == "groq" else "groq"
        fallback_client = _build_fallback_client(settings, fallback_provider, primary_tier)
        if fallback_client is None:
            raise  # no working fallback configured - surface the original error

        print(
            f"[cost_routing] {primary_provider} rate-limited on {primary_tier} tier - "
            f"failing over to {fallback_provider} for this call"
        )
        return with_backoff(fallback_client).invoke(messages, **invoke_kwargs)


def _build_fallback_client(settings: Settings, provider: str, tier: str):
    """Build a one-off client for the fallback provider. Not cached via
    get_llm()'s @lru_cache, since this is an exceptional-path client, not
    the steady-state one - reuses whatever model id is already configured
    for that provider if available, else returns None (no silent
    guessing at an unconfigured model)."""
    from langchain_cerebras import ChatCerebras
    from langchain_groq import ChatGroq

    if provider == "groq" and settings.groq_api_key:
        # Fall back to Groq's cheap-tier model - safest default, since we
        # don't have a second "strong" model string configured for Groq
        # once the primary strong-tier assignment has moved off it.
        _, model_name = settings.llm_model_cheap.split("/", 1)
        return ChatGroq(model=model_name, api_key=settings.groq_api_key)

    if provider == "cerebras" and settings.cerebras_api_key:
        _, model_name = settings.llm_model.split("/", 1) if "cerebras" in settings.llm_model else (None, None)
        if model_name is None:
            return None  # cerebras isn't the configured strong-tier provider - nothing to fall back to yet
        return ChatCerebras(model=model_name, api_key=settings.cerebras_api_key)

    return None