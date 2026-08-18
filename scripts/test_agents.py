"""End-to-end smoke test for the Phase 3+4 agent graph.

Not a unit test - makes real Groq calls against two already-ingested papers
(DS-1000, BigCodeBench) and inspects the result structurally. LLM output is
non-deterministic, so this checks shape (did each stage produce what the
next stage needs), not exact content.
"""

from collections import Counter

from rich.console import Console
from rich.panel import Panel

from grounded_review.agents.graph import build_graph
from grounded_review.agents.state import ReviewState

console = Console()

TOPIC = "How do current benchmarks evaluate code generation from large language models?"
ARXIV_IDS = ["2211.11501", "2406.15877"]  # DS-1000, BigCodeBench - ingested in Phase 2


def main():
    graph = build_graph()
    initial_state = ReviewState(topic=TOPIC, arxiv_ids=ARXIV_IDS)

    console.print(Panel(f"Topic: {TOPIC}\nPapers: {', '.join(ARXIV_IDS)}", title="Running graph"))
    final_state = graph.invoke(initial_state)
    # graph.invoke() returns a dict, not a ReviewState - re-validate for typed access
    result = ReviewState.model_validate(final_state)

    console.rule("Plan")
    for i, sub_topic in enumerate(result.plan, 1):
        console.print(f"{i}. {sub_topic}")
    assert 3 <= len(result.plan) <= 6, f"expected 3-6 sub-topics, got {len(result.plan)}"

    console.rule("Research notes")
    console.print(f"{len(result.research_notes)} notes retrieved")
    assert result.research_notes, "research produced no notes"

    console.rule("Draft")
    console.print(result.draft)

    console.rule("Reviewer (coherence)")
    console.print(f"Approved: {result.critique.approved if result.critique else 'N/A'}")
    console.print(f"Revision count: {result.revision_count}")
    if result.critique:
        console.print(f"Feedback: {result.critique.feedback}")

    console.rule("Verifier (grounding)")
    console.print(f"Verification revision count: {result.verification_revision_count}")
    if result.verification_results:
        counts = Counter(v.verdict for v in result.verification_results)
        console.print(
            f"{len(result.verification_results)} claims checked - "
            f"supported: {counts['supported']}, "
            f"partially_supported: {counts['partially_supported']}, "
            f"unsupported: {counts['unsupported']}"
        )
        unresolved = [v for v in result.verification_results if v.verdict != "supported"]
        if unresolved:
            console.print("[yellow]Claims that did not pass:[/yellow]")
            for v in unresolved:
                console.print(f"  [{v.verdict}] \"{v.claim_text}\" ({v.chunk_id})")
                console.print(f"    -> {v.explanation}")
    else:
        console.print("[red]No cited claims found - draft may be entirely uncited.[/red]")

    console.rule("Coverage flags (informational only - does not gate the loop)")
    console.print(f"{len(result.coverage_flags)} factual-looking sentences with no citation marker")
    for flag in result.coverage_flags:
        console.print(f"  - {flag}")


if __name__ == "__main__":
    main()