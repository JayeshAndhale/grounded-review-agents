"""Smoke test for TokenUsageTracker: confirms the tier:strong/tier:cheap
tags set in config.py's get_llm() actually survive being wrapped by
.with_structured_output() and .with_retry() at each node's call site.

If this passes with zero 'unknown' calls, the tracker is safe to rely on
in Phase 5's evaluation harness. If not, that's a real finding to fix
before any cost number gets trusted - not something to guess around.
"""

from rich.console import Console
from rich.panel import Panel

from grounded_review.agents.graph import build_graph
from grounded_review.agents.state import ReviewState
from grounded_review.evaluation.token_tracker import TokenUsageTracker

console = Console()

TOPIC = "How do current benchmarks evaluate code generation from large language models?"
ARXIV_IDS = ["2211.11501", "2406.15877"]  # DS-1000, BigCodeBench - already ingested


def main():
    graph = build_graph()
    initial_state = ReviewState(topic=TOPIC, arxiv_ids=ARXIV_IDS)
    tracker = TokenUsageTracker()

    console.print(Panel(f"Topic: {TOPIC}\nPapers: {', '.join(ARXIV_IDS)}", title="Running graph with token tracking"))
    final_state = graph.invoke(initial_state, config={"callbacks": [tracker]})
    result = ReviewState.model_validate(final_state)

    console.rule("Run outcome (sanity check - not the point of this script)")
    console.print(f"Reviewer approved: {result.critique.approved if result.critique else 'N/A'}")
    console.print(f"Revision count: {result.revision_count}, verification revision count: {result.verification_revision_count}")

    summary = tracker.summary()

    console.rule("Token usage by tier")
    for tier in ("strong", "cheap"):
        stats = summary[tier]
        console.print(
            f"[bold]{tier}[/bold]: {stats['calls']} calls, "
            f"{stats['input_tokens']} input tokens, {stats['output_tokens']} output tokens, "
            f"~${stats['estimated_cost_usd']} estimated"
        )

    console.rule("Propagation check - this is what this script actually verifies")
    unknown = summary.get("unknown")
    if unknown:
        console.print(
            f"[red]FAIL: {unknown['calls']} call(s) landed in 'unknown' - "
            f"tier tags did not propagate through .with_structured_output()/.with_retry(). "
            f"Cost numbers below are NOT trustworthy until this is fixed.[/red]"
        )
    else:
        console.print("[green]PASS: every LLM call was correctly attributed to strong or cheap. Tagging works.[/green]")

    console.rule("Total estimated cost")
    console.print(f"${summary['total_estimated_cost_usd']} (estimate based on published Groq pricing; project runs on free tier)")

    assert not unknown, "Fix tag propagation before trusting this tracker in the evaluation harness."


if __name__ == "__main__":
    main()