"""In-memory job store and background execution for the review-generation API."""

import time
from threading import Lock

from grounded_review.agents.graph import build_graph  # your Phase 3/4 full graph
from grounded_review.evaluation.metrics import build_baseline_graph, score_draft
from grounded_review.observability.tracer import GraphTracer
from grounded_review.evaluation.token_tracker import TokenUsageTracker  # adjust import path if named differently

from .models import JobRecord, TraceSummary

_jobs: dict[str, JobRecord] = {}
_lock = Lock()


def create_job(arxiv_ids: list[str], graph: str) -> JobRecord:
    record = JobRecord(arxiv_ids=arxiv_ids, graph=graph)
    with _lock:
        _jobs[record.job_id] = record
    return record


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _jobs.get(job_id)


def _save(record: JobRecord) -> None:
    with _lock:
        _jobs[record.job_id] = record


def _build_trace_summary(tracer: GraphTracer, tracker: TokenUsageTracker, wall_clock: float) -> TraceSummary:
    # Adjust attribute names to whatever GraphTracer/TokenUsageTracker actually expose —
    # this assumes tracker has per-tier totals and tracer has a flat span list with a "node" field.
    revision_loops = max(0, sum(1 for s in tracer.spans if s.node == "writer") - 1)
    return TraceSummary(
        total_tokens=tracker.total_tokens,
        tokens_by_tier=tracker.tokens_by_tier,
        wall_clock_seconds=wall_clock,
        span_count=len(tracer.spans),
        revision_loops=revision_loops,
    )


def _run_review(job_id: str, arxiv_ids: list[str], graph_choice: str) -> None:
    record = get_job(job_id)
    if record is None:
        return  # shouldn't happen, but don't crash a background thread on it

    record.status = "running"
    _save(record)

    tracer = GraphTracer()
    tracker = TokenUsageTracker()
    start = time.monotonic()

    try:
        graph = build_baseline_graph() if graph_choice == "baseline" else build_graph()

        # Adjust initial state shape to match your actual state schema (agents/state.py)
        result = graph.invoke(
            {"arxiv_ids": arxiv_ids},
            config={"callbacks": [tracer, tracker]},
        )

        manuscript = result["manuscript"]  # adjust key to your actual state field name
        grounding = score_draft(manuscript)  # adjust signature if score_draft needs more args

        record.manuscript = manuscript
        record.grounding_score = grounding
        record.trace = _build_trace_summary(tracer, tracker, time.monotonic() - start)
        record.status = "completed"

    except Exception as exc:
        record.error = str(exc)
        record.status = "failed"

    _save(record)