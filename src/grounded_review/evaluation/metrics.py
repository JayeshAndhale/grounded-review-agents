"""Evaluation harness: runs the benchmark matrix (topic × condition × run),
checkpointing after every individual run so the harness survives crashing
mid-matrix — expected given confirmed daily token-budget limits (§7/§8 of
the handoff), not a hypothetical edge case.

Baseline vs treatment is NOT two different graphs sharing logic by
coincidence — baseline is the identical scheduler->research->writer->
reviewer graph (Phase 3), with verification run as a standalone post-hoc
scoring pass over the final draft's citations, never wired into the loop.
Wiring the verifier into baseline's loop would make it stop being a
baseline - the entire point is measuring what an unverified pipeline
produces, then grading it after the fact with the same rubric treatment
uses live.

A failed run (any exception - TPD wall, provider error, etc.) is
checkpointed as status="failed" and is NOT retried automatically on the
next run_all() call. A TPD wall is not a transient blip (see handoff §7:
short backoff cannot bridge daily-quota exhaustion) - retrying immediately
just re-hits the same wall and burns more quota finding that out. Clearing
a failed run for retry is an explicit, separate action (retry_failed()),
never implicit.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from grounded_review.agents.graph import (
    reviewer_node,
    research_node,
    route_after_review,
    scheduler_node,
    writer_node,
)
from grounded_review.agents.state import ReviewState
from grounded_review.evaluation.benchmark import BENCHMARK_TOPICS, BenchmarkTopic
from grounded_review.evaluation.token_tracker import TokenUsageTracker
from grounded_review.verification.verifier import (
    all_claims_supported,
    extract_citations,
    flag_uncited_claims,
    verify_claim,
)
from grounded_review.retrieval.vector_store import PaperStore

Condition = Literal["baseline", "treatment"]

CHECKPOINT_PATH = Path("data/evaluation/checkpoint.jsonl")
RUNS_PER_CONDITION = 3

_store = PaperStore()


# ---------------------------------------------------------------------------
# Baseline graph: identical Phase 3 nodes, verifier deliberately NOT wired
# in as a graph node. Reuses the exact same node functions as the full
# graph (imported above) rather than reimplementing them, so "baseline" and
# "treatment" can never silently drift into testing different writer/
# reviewer logic - only the presence/absence of the verification loop
# differs, which is the one variable this experiment is measuring.
# ---------------------------------------------------------------------------

def build_baseline_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("research", research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "scheduler")
    builder.add_edge("scheduler", "research")
    builder.add_edge("research", "writer")
    builder.add_edge("writer", "reviewer")

    # route_after_review normally routes an approved draft to "verifier" -
    # that target doesn't exist in this graph, so remap "verifier" -> END.
    # Approval still ends the run; only the destination changes.
    def route_baseline(state: ReviewState) -> str:
        target = route_after_review(state)
        return END if target == "verifier" else target

    builder.add_conditional_edges(
        "reviewer", route_baseline, {"writer": "writer", END: END}
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Post-hoc scoring: identical grading logic to verifier_node, called once
# over a finished draft instead of gating a revision loop. This is what
# keeps baseline and treatment's numbers apples-to-apples - same rubric,
# different only in whether it influenced generation.
# ---------------------------------------------------------------------------

def score_draft(draft: str) -> dict:
    citations = extract_citations(draft)
    chunk_ids = list({cid for _, cid in citations})
    chunk_texts = _store.get_chunks(chunk_ids)

    results = [verify_claim(claim, cid, chunk_texts.get(cid)) for claim, cid in citations]
    coverage = flag_uncited_claims(draft)

    total = len(results)
    supported = sum(1 for v in results if v.verdict == "supported")
    partial = sum(1 for v in results if v.verdict == "partially_supported")
    unsupported = sum(1 for v in results if v.verdict == "unsupported")

    return {
        "total_citations": total,
        "supported": supported,
        "partially_supported": partial,
        "unsupported": unsupported,
        "supported_rate": round(supported / total, 4) if total else 0.0,
        "coverage_flag_count": len(coverage),
        "fully_grounded": all_claims_supported(results),
    }


@dataclass
class RunResult:
    topic_name: str
    condition: Condition
    run_number: int
    status: Literal["completed", "failed"]
    # --- populated on success ---
    scoring: dict = field(default_factory=dict)
    reviewer_revision_count: int = 0
    verifier_revision_count: int = 0
    token_summary: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # --- populated on failure ---
    error: str = ""
    error_type: str = ""


class ResultStore:
    """Append-only JSONL checkpoint. Line-per-run means a crash mid-matrix
    loses at most the one in-flight run, never the runs already recorded -
    no read-modify-write of one big file that a crash could corrupt."""

    def __init__(self, path: Path = CHECKPOINT_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _key(self, topic_name: str, condition: str, run_number: int) -> str:
        return f"{topic_name}::{condition}::{run_number}"

    def completed_keys(self) -> set[str]:
        """Every (topic, condition, run) already attempted - completed OR
        failed. Both count as 'don't touch on a plain resume' per the
        explicit no-auto-retry decision."""
        if not self.path.exists():
            return set()
        keys = set()
        with open(self.path) as f:
            for line in f:
                record = json.loads(line)
                keys.add(self._key(record["topic_name"], record["condition"], record["run_number"]))
        return keys

    def append(self, result: RunResult) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def all_results(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f]

    def rewrite_without_failed(self) -> int:
        """Drop every 'failed' record so those (topic, condition, run)
        combinations become eligible for a fresh attempt. Explicit,
        separate action - never called implicitly from run_all()."""
        records = self.all_results()
        kept = [r for r in records if r["status"] != "failed"]
        dropped = len(records) - len(kept)
        with open(self.path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        return dropped


def run_single(topic: BenchmarkTopic, condition: Condition, run_number: int) -> RunResult:
    """Execute exactly one (topic, condition, run) cell of the matrix."""
    tracker = TokenUsageTracker()
    start = time.monotonic()
    initial_state = ReviewState(topic=topic.topic_prompt, arxiv_ids=topic.arxiv_ids)

    try:
        if condition == "baseline":
            graph = build_baseline_graph()
            final = ReviewState.model_validate(
                graph.invoke(initial_state, config={"callbacks": [tracker]})
            )
            scoring = score_draft(final.draft)
            verifier_revision_count = 0  # baseline never runs the loop
        else:
            from grounded_review.agents.graph import build_graph
            graph = build_graph()
            final = ReviewState.model_validate(
                graph.invoke(initial_state, config={"callbacks": [tracker]})
            )
            scoring = score_draft(final.draft)  # same rubric, applied to the already-verified draft
            verifier_revision_count = final.verification_revision_count

        return RunResult(
            topic_name=topic.name,
            condition=condition,
            run_number=run_number,
            status="completed",
            scoring=scoring,
            reviewer_revision_count=final.revision_count,
            verifier_revision_count=verifier_revision_count,
            token_summary=tracker.summary(),
            elapsed_seconds=round(time.monotonic() - start, 1),
        )
    except Exception as e:
        return RunResult(
            topic_name=topic.name,
            condition=condition,
            run_number=run_number,
            status="failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_seconds=round(time.monotonic() - start, 1),
        )


def run_all(store: ResultStore | None = None) -> None:
    """Iterate the full topic x condition x run matrix, skipping anything
    already attempted (completed or failed). Safe to interrupt (Ctrl-C,
    TPD crash) and re-invoke - each cell is checkpointed independently."""
    store = store or ResultStore()
    done = store.completed_keys()

    for topic in BENCHMARK_TOPICS:
        for condition in ("baseline", "treatment"):
            for run_number in range(1, RUNS_PER_CONDITION + 1):
                key = f"{topic.name}::{condition}::{run_number}"
                if key in done:
                    continue

                print(f"Running {key}...")
                result = run_single(topic, condition, run_number)
                store.append(result)

                if result.status == "failed":
                    print(f"  FAILED ({result.error_type}): {result.error}")
                    print("  Stopping this run_all() call - resolve before resuming.")
                    return
                else:
                    print(f"  OK — supported_rate={result.scoring['supported_rate']}, "
                          f"elapsed={result.elapsed_seconds}s")