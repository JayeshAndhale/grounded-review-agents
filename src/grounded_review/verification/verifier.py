"""Grounding verifier: checks each cited claim in a draft against the
exact source text of the chunk it cites - not the research-note summary,
which is itself one paraphrase removed from the source.

Two separate signals, deliberately not merged into one judgment:
- verification_results: LLM-graded accuracy of claims that ARE cited
- coverage_flags: cheap heuristic catching factual-looking sentences with
  NO citation marker at all. Informational only - see ReviewState's
  docstring for why this doesn't gate the revision loop.
"""

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from grounded_review.agents.state import ClaimVerdict, ReviewState
from grounded_review.config import get_llm, with_backoff
from grounded_review.retrieval.vector_store import PaperStore

_store = PaperStore()

Verdict = Literal["supported", "partially_supported", "unsupported"]

# Marker sits at the end of the sentence it supports, e.g. "text [[id]]."
# Heuristic, same spirit as the Phase 1 section-heading regexes: assumes no
# other period falls between the sentence start and the marker. A sentence
# with an internal abbreviation period (e.g. "e.g.") could split early -
# acceptable for a sampling-based check, not worth over-engineering against
# without evidence it actually happens.
CITATION_PATTERN = re.compile(r"([^.\n]*?)\[\[([^\]]+)\]\]\.?")

# Cheap, mechanical signal that a sentence looks factual: a number, a
# percentage, or comparative/quantitative language. Doesn't need an LLM -
# this is noticing, not reasoning.
FACTUAL_SIGNAL = re.compile(
    r"\d|%|\b(higher|lower|greater|significant|outperform|exceed|"
    r"compared to|versus|majority|approximately)\b",
    re.IGNORECASE,
)


def extract_citations(draft: str) -> list[tuple[str, str]]:
    """Pull (claim_text, chunk_id) pairs from the writer's inline markers."""
    pairs = []
    for match in CITATION_PATTERN.finditer(draft):
        claim = match.group(1).strip()
        chunk_id = match.group(2).strip()
        if claim:
            pairs.append((claim, chunk_id))
    return pairs


def flag_uncited_claims(draft: str) -> list[str]:
    """Sentences that look factual but carry no citation marker at all."""
    flagged = []
    sentences = re.split(r"(?<=[.!?])\s+", draft)
    for sentence in sentences:
        if "[[" in sentence:
            continue
        if FACTUAL_SIGNAL.search(sentence) and len(sentence.strip()) > 15:
            flagged.append(sentence.strip())
    return flagged


VERIFIER_SYSTEM_PROMPT = """\
You are the grounding verification agent. You will be given one claim from
a manuscript draft and the exact source text of the chunk it cited.

Judge whether the source text supports the claim:
- "supported": the source directly states or clearly implies the claim.
- "partially_supported": the source is relevant, but the claim adds a
  specific number, comparison, or generalisation the source does not
  actually give.
- "unsupported": the source does not support the claim, or contradicts it.

Judge only whether THIS source text supports THIS claim - do not use
outside knowledge of the papers or topic. Give a one-sentence explanation
naming what the source does or doesn't say.
"""


class VerdictOutput(BaseModel):
    """What the LLM actually produces - claim_text and chunk_id are already
    known to us, so the model only supplies the judgment itself."""

    verdict: Verdict
    explanation: str


def verify_claim(claim_text: str, chunk_id: str, chunk_text: str | None) -> ClaimVerdict:
    """Grade one claim against its source. chunk_text=None means the
    chunk_id didn't resolve in the store - graded unsupported without
    spending an LLM call on a source that doesn't exist."""
    if chunk_text is None:
        return ClaimVerdict(
            claim_text=claim_text,
            chunk_id=chunk_id,
            verdict="unsupported",
            explanation="Cited chunk_id does not exist in the vector store.",
        )

    llm = with_backoff(get_llm("strong").with_structured_output(VerdictOutput))
    human = f"Claim:\n{claim_text}\n\nSource text:\n{chunk_text}"
    result = llm.invoke(
        [SystemMessage(content=VERIFIER_SYSTEM_PROMPT), HumanMessage(content=human)],
        config={"tags": ["tier:strong"]},
    )
    return ClaimVerdict(
        claim_text=claim_text,
        chunk_id=chunk_id,
        verdict=result.verdict,
        explanation=result.explanation,
    )

def all_claims_supported(results: list[ClaimVerdict]) -> bool:
    """A draft with zero citations is unsupported, not trivially passing -
    all(...) over an empty list is True in Python. Shared by verifier_node
    and graph.py's routing so the rule lives in exactly one place.
    """
    return bool(results) and all(v.verdict == "supported" for v in results)
def verifier_node(state: ReviewState) -> dict:
    """Graph node: verify every cited claim in the current draft.

    An empty citation list is treated as a grounding failure, not a pass -
    all(...) over an empty list is trivially True in Python, which would
    otherwise let a completely uncited draft sail through unchecked.
    """
    citations = extract_citations(state.draft)
    chunk_ids = list({cid for _, cid in citations})
    chunk_texts = _store.get_chunks(chunk_ids)

    results = [verify_claim(claim, cid, chunk_texts.get(cid)) for claim, cid in citations]
    coverage = flag_uncited_claims(state.draft)

    all_supported = all_claims_supported(results)
    update: dict = {"verification_results": results, "coverage_flags": coverage}
    if not all_supported:
        update["next_check"] = "verifier"
        update["verification_revision_count"] = state.verification_revision_count + 1
    return update