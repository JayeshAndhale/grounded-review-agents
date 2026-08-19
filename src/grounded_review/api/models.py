"""In-memory job store and background execution for the review-generation API.

Mirrors evaluation/metrics.py::run_single() deliberately - same initial-state
construction, same graph.invoke(..., callbacks=[tracker]) call, same
ReviewState.model_validate() + score_draft() sequence - so the API path and
the evaluation-harness path can never silently diverge in how a run is
actually executed. Only the job-record bookkeeping around it is new.
"""

import time
from threading import Lock

from grounded_review.agents.graph import build_graph
from grounded_review.agents.state import ReviewState
from grounded_review.evaluation.metrics import build_baseline_graph, score_draft
from grounded_review.evaluation.token_tracker import TokenUsageTracker

from .models import JobRecord, TraceSummary

_jobs: dict[str, JobRecord] = {}
_lock = Lock()


def create_job(topic: str, arxiv_ids: list[str], graph: str) -> JobRecord:
    record = JobRecord(topic=topic, arxiv_ids=arxiv_ids, graph=graph)
    with _lock:
        _jobs[record.job_id] = record
    return record


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _jobs.get(job_id)


def _save(record: JobRecord) -> None:
    with _lock:
        _jobs[record.job_id] = record


def _run_review(job_id: str, topic: str, arxiv_ids: list[str], graph_choice: str) -> None:
    record = get_job(job_id)
    if record is None:
        return  # shouldn't happen, but don't crash a background thread on it

    record.status = "running"
    _save(record)

    tracker = TokenUsageTracker()
    start = time.monotonic()

    try:
        initial_state = ReviewState(topic=topic, arxiv_ids=arxiv_ids)
        graph = build_baseline_graph() if graph_choice == "baseline" else build_graph()

        final = ReviewState.model_validate(
            graph.invoke(initial_state, config={"callbacks": [tracker]})
        )
        grounding = score_draft(final.draft)

        record.manuscript = final.draft
        record.grounding_score = grounding
        record.trace = TraceSummary(
            wall_clock_seconds=round(time.monotonic() - start, 1),
            reviewer_revision_count=final.revision_count,
            verifier_revision_count=final.verification_revision_count,
            token_summary=tracker.summary(),
        )
        record.status = "completed"

    except Exception as exc:
        record.error = str(exc)
        record.status = "failed"

    _save(record)