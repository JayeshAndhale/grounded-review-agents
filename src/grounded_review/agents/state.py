"""LangGraph state schema.

Every node reads this object and returns a dict of the fields it changed.
LangGraph merges that dict back into state before calling the next node.
Because this is a Pydantic model, not a TypedDict, a node that hands back
malformed data fails at the model boundary immediately, instead of silently
corrupting state that a node three hops downstream trusts blindly.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ResearchNote(BaseModel):
    """One retrieved-and-summarised piece of evidence, still tied to its source.

    The research agent produces a list of these per sub-topic in the plan.
    `chunk_id` is what makes Phase 4 verification possible at all — without
    it, the writer's citations are prose that only *looks* sourced.
    """

    chunk_id: str
    arxiv_id: str
    section: str
    summary: str = Field(description="Cheap-model summary of the chunk, in its own words")


class Critique(BaseModel):
    """Structured reviewer output — deliberately not free text.

    `approved` is what the graph's conditional edge branches on. A boolean
    field means routing doesn't cost an extra LLM call to interpret feedback.
    """

    approved: bool
    feedback: str


class ClaimVerdict(BaseModel):
    """One grounding verdict, checked against the chunk's original text
    (not the research-note summary, which is itself a paraphrase one step
    removed from the source).

    Three-way verdict rather than binary: 'partially_supported' captures
    claims that add specifics — a number, a generalisation — the source
    chunk doesn't actually give, which a binary pass/fail would hide.
    """

    claim_text: str
    chunk_id: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    explanation: str = Field(description="What the chunk does or doesn't say, relative to the claim")


class ReviewState(BaseModel):
    """The single object passed between every node in the graph."""

    # --- input, set once before the graph runs ---
    topic: str
    arxiv_ids: list[str]

    # --- scheduler writes this ---
    plan: list[str] = Field(default_factory=list, description="Sub-topics to research, in order")

    # --- research writes this ---
    research_notes: list[ResearchNote] = Field(default_factory=list)

    # --- writer writes this: prose with inline [[chunk_id]] citation markers ---
    draft: str = ""

    # --- reviewer writes this (coherence/completeness only) ---
    critique: Critique | None = None
    revision_count: int = 0

    # --- verifier writes this (grounding accuracy only) ---
    verification_results: list[ClaimVerdict] = Field(default_factory=list)
    coverage_flags: list[str] = Field(
        default_factory=list,
        description="Factual-looking sentences with no [[chunk_id]] marker - informational, does not gate the loop", 
    )
    verification_revision_count: int = 0

    # --- routing: which check sent the draft back to the writer ---
    next_check: Literal["reviewer", "verifier"] = "reviewer"