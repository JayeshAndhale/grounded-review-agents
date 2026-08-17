"""End-to-end smoke test for the Phase 3 agent graph.

Not a unit test - makes real Groq calls against two already-ingested papers
(DS-1000, BigCodeBench) and inspects the result structurally. LLM output is
non-deterministic, so this checks shape (did each stage produce what the
next stage needs), not exact content.
"""

import re

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

    console.rule("Grounding sanity check (existence only, not Phase 4's semantic check)")
    known_chunk_ids = {n.chunk_id for n in result.research_notes}
    cited_ids = set(re.findall(r"\[\[([^\]]+)\]\]", result.draft))
    unknown = cited_ids - known_chunk_ids
    console.print(f"Citations in draft: {len(cited_ids)}")
    console.print(f"Unknown chunk_ids (writer fabricated an ID): {len(unknown)}")
    if unknown:
        console.print(f"[red]{unknown}[/red]")

    console.rule("Reviewer")
    console.print(f"Approved: {result.critique.approved if result.critique else 'N/A'}")
    console.print(f"Revision count: {result.revision_count}")
    if result.critique:
        console.print(f"Feedback: {result.critique.feedback}")


if __name__ == "__main__":
    main()