"""The five-node LangGraph pipeline:
scheduler -> research -> writer -> reviewer -> verifier -> end,
with two independently-capped loops back to the writer:
  - reviewer rejects (coherence)  -> writer -> reviewer again
  - verifier rejects (grounding)  -> writer -> verifier again (skips reviewer)
A coherent draft only reaches the verifier once; grounding revisions never
re-trigger a coherence check, since the coherence question was already
settled - see state.next_check.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from grounded_review.agents.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SCHEDULER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    format_research_notes,
)
from grounded_review.agents.state import Critique, ResearchNote, ReviewState
from grounded_review.agents.tools import search_papers
from grounded_review.config import get_llm, get_settings, with_backoff
from grounded_review.verification.verifier import all_claims_supported, verifier_node


class SchedulerPlan(BaseModel):
    """Structured output for the scheduler - not part of ReviewState itself,
    just the shape we force the LLM into before unpacking into state.plan."""

    sub_topics: list[str] = Field(min_length=3, max_length=6)


def scheduler_node(state: ReviewState) -> dict:
    llm = with_backoff(get_llm("strong").with_structured_output(SchedulerPlan))
    human = f"Review topic: {state.topic}\nPapers in scope (arXiv IDs): {', '.join(state.arxiv_ids)}"
    result = llm.invoke([SystemMessage(content=SCHEDULER_SYSTEM_PROMPT), HumanMessage(content=human)], config={"tags": ["tier:strong"]})
    return {"plan": result.sub_topics}


def research_node(state: ReviewState) -> dict:
    cheap_llm = with_backoff(get_llm("cheap"))
    notes: list[ResearchNote] = []
    for sub_topic in state.plan:
        chunks = search_papers.invoke({"query": sub_topic})
        for chunk in chunks:
            prompt = [
                SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                HumanMessage(content=f"Sub-topic: {sub_topic}\n\nPassage:\n{chunk['text']}"),
            ]
            summary = cheap_llm.invoke(prompt, config={"tags": ["tier:cheap"]}).content
            notes.append(
                ResearchNote(
                    chunk_id=chunk["chunk_id"],
                    arxiv_id=chunk["arxiv_id"],
                    section=chunk["section"],
                    summary=summary,
                )
            )
    return {"research_notes": notes}


def writer_node(state: ReviewState) -> dict:
    """Handles three cases: first draft, reviewer-rejected revision, and
    verifier-rejected revision - distinguished by which critique is 'live'
    given next_check, since both critique and verification_results can be
    populated at once (reviewer approved, then verifier ran and rejected)."""
    llm = with_backoff(get_llm("strong"))

    if state.next_check == "verifier" and state.verification_results:
        problems = "\n".join(
            f"- \"{v.claim_text}\" ({v.verdict}): {v.explanation}"
            for v in state.verification_results
            if v.verdict != "supported"
        )
        human = (
            f"Previous draft:\n{state.draft}\n\n"
            f"The following cited claims failed grounding verification - "
            f"reword or remove them so every claim is actually supported by its cited source:\n{problems}"
        )
    elif state.critique is not None and not state.critique.approved:
        human = (
            f"Previous draft:\n{state.draft}\n\n"
            f"Reviewer feedback - address this directly:\n{state.critique.feedback}"
        )
    else:
        plan_text = "\n".join(f"- {t}" for t in state.plan)
        human = f"Plan:\n{plan_text}\n\nResearch notes:\n{format_research_notes(state.research_notes)}"

    result = llm.invoke([SystemMessage(content=WRITER_SYSTEM_PROMPT), HumanMessage(content=human)], config={"tags": ["tier:strong"]})
    return {"draft": result.content}


def reviewer_node(state: ReviewState) -> dict:
    llm = with_backoff(get_llm("strong").with_structured_output(Critique))
    plan_text = "\n".join(f"- {t}" for t in state.plan)
    human = f"Plan:\n{plan_text}\n\nDraft:\n{state.draft}"
    critique = llm.invoke([SystemMessage(content=REVIEWER_SYSTEM_PROMPT), HumanMessage(content=human)], config={"tags": ["tier:strong"]})

    update: dict = {"critique": critique}
    if not critique.approved:
        update["revision_count"] = state.revision_count + 1
    return update


def route_after_writer(state: ReviewState) -> str:
    """Every writer revision returns to whichever check sent it back -
    reviewer by default (first pass, or a reviewer rejection), verifier
    only when a grounding failure triggered the revision."""
    return state.next_check


def route_after_review(state: ReviewState) -> str:
    """Coherence gate. Approved drafts move on to grounding checks;
    rejected drafts loop to the writer until approved or capped - a
    coherence-capped draft ships unapproved and skips verification
    entirely, since there's no point grounding-checking a draft that's
    about to be restructured anyway.
    """
    settings = get_settings()
    if state.critique and state.critique.approved:
        return "verifier"
    if state.revision_count >= settings.max_revision_loops:
        return END
    return "writer"


def route_after_verification(state: ReviewState) -> str:
    """Grounding gate, independent cap from the reviewer's loop."""
    settings = get_settings()
    if all_claims_supported(state.verification_results):
        return END
    if state.verification_revision_count >= settings.max_revision_loops:
        return END
    return "writer"


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("research", research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("verifier", verifier_node)

    builder.add_edge(START, "scheduler")
    builder.add_edge("scheduler", "research")
    builder.add_edge("research", "writer")

    builder.add_conditional_edges(
        "writer", route_after_writer, {"reviewer": "reviewer", "verifier": "verifier"}
    )
    builder.add_conditional_edges(
        "reviewer", route_after_review, {"writer": "writer", "verifier": "verifier", END: END}
    )
    builder.add_conditional_edges(
        "verifier", route_after_verification, {"writer": "writer", END: END}
    )

    return builder.compile()