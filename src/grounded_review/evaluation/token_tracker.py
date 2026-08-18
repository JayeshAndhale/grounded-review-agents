"""Token usage tracking for cost accounting (Phase 5).

Groq is free-tier for this project - no real dollar cost to report - but
the CV claim ("cutting cost per manuscript by X%") needs an honest,
defensible number, not a guess. This reads ACTUAL usage the API already
returns on every response, rather than hand-estimating with tiktoken, and
multiplies by Groq's published per-token pricing to produce a clearly
labeled ESTIMATE.
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Groq's published per-1M-token pricing (USD) for the models this project
# uses, as of Phase 5. Provider pricing changes without notice - same
# lesson as the model-deprecation bug in Phase 3 - so confirm against
# https://groq.com/pricing before citing a cost figure anywhere final.
PRICING_PER_MILLION_TOKENS = {
    "strong": {"input": 0.15, "output": 0.60},   # openai/gpt-oss-120b
    "cheap": {"input": 0.075, "output": 0.30},   # openai/gpt-oss-20b
}


@dataclass
class TierUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost_usd(self, tier: str) -> float:
        rates = PRICING_PER_MILLION_TOKENS[tier]
        return (self.input_tokens / 1_000_000) * rates["input"] + (
            self.output_tokens / 1_000_000
        ) * rates["output"]


class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates token usage per model tier across an entire graph run.

    Attach via graph.invoke(state, config={"callbacks": [tracker]}) -
    LangChain fires callback events for every underlying LLM call inside
    any node automatically, so no node code needs to know this exists.

    Tier is read from the 'tier:strong'/'tier:cheap' tag set in
    config.py's get_llm(). An untagged call is bucketed as 'unknown'
    rather than silently dropped or guessed at - a real gap should show
    up as a visible number, not vanish.
    """

    def __init__(self) -> None:
        self.usage: dict[str, TierUsage] = {
            "strong": TierUsage(),
            "cheap": TierUsage(),
            "unknown": TierUsage(),
        }

    def _tier_from_tags(self, tags: list[str] | None) -> str:
        for tag in tags or []:
            if tag == "tier:strong":
                return "strong"
            if tag == "tier:cheap":
                return "cheap"
        return "unknown"

    def on_llm_end(
        self, response: LLMResult, *, run_id, parent_run_id=None, tags=None, **kwargs: Any
    ) -> None:
        tier = self._tier_from_tags(tags)
        prompt_tokens = 0
        completion_tokens = 0

        # Groq's OpenAI-compatible responses populate llm_output["token_usage"].
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            # Fallback: usage_metadata living directly on the AIMessage.
            for generation in response.generations:
                for gen in generation:
                    message = getattr(gen, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None) if message else None
                    if usage_metadata:
                        prompt_tokens += usage_metadata.get("input_tokens", 0)
                        completion_tokens += usage_metadata.get("output_tokens", 0)

        bucket = self.usage[tier]
        bucket.input_tokens += prompt_tokens
        bucket.output_tokens += completion_tokens
        bucket.calls += 1

    def summary(self) -> dict:
        """Per-tier token counts and estimated USD cost, labeled as an
        estimate since this project runs on Groq's free tier."""
        result: dict = {}
        total_cost = 0.0
        for tier in ("strong", "cheap"):
            u = self.usage[tier]
            cost = u.cost_usd(tier)
            total_cost += cost
            result[tier] = {
                "calls": u.calls,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "estimated_cost_usd": round(cost, 5),
            }
        if self.usage["unknown"].calls:
            result["unknown"] = {
                "calls": self.usage["unknown"].calls,
                "note": "Untagged LLM call detected - check get_llm() tagging in config.py",
            }
        result["total_estimated_cost_usd"] = round(total_cost, 5)
        return result