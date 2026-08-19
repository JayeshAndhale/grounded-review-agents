"""Structural/timing trace of a graph run: which nodes executed, in what
order, how long each took, and whether it succeeded — the complement to
token_tracker.py's cost-per-tier view. Same attachment mechanism
(BaseCallbackHandler via config={"callbacks": [...]}) as TokenUsageTracker,
kept as a separate class rather than merged into it, matching the existing
single-responsibility split between grading (metrics.py) and cost
(token_tracker.py).

NEEDS VERIFICATION AGAINST A REAL RUN: LangGraph's exact chain-naming and
nesting behavior through the callback system hasn't been inspected yet —
the node-detection logic here is a best-effort guess based on documented
callback behavior, not confirmed output. First real run's trace.json should
be read manually before this is trusted for anything beyond a rough view.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

TRACE_DIR = Path("data/traces")

# LangGraph node execution surfaces as an on_chain_start/on_chain_end pair.
# Internal LangChain machinery (prompt formatting, output parsing, retry
# wrapping) also fires chain events — those are NOT node-level spans and
# would clutter the trace if not filtered out. Best available signal:
# node-level spans are top-level (no parent_run_id) OR their serialized
# name matches a known node name. Both checks kept, in case one is
# unreliable in practice — verify against real output, don't assume.
KNOWN_NODE_NAMES = {"scheduler", "research", "writer", "reviewer", "verifier"}


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start_ts: float
    end_ts: float | None = None
    status: str = "running"  # "running" | "ok" | "error"
    error: str = ""
    kind: str = "chain"  # "chain" | "llm"
    tags: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.end_ts is None:
            return None
        return round((self.end_ts - self.start_ts) * 1000, 1)


class GraphTracer(BaseCallbackHandler):
    """Records a span tree for one graph.invoke() call. Attach fresh per
    run — like TokenUsageTracker, this is stateful and NOT safe to reuse
    across multiple runs."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.spans: dict[str, Span] = {}
        self._order: list[str] = []

    def _record_start(self, run_id, parent_run_id, name: str, kind: str, tags: list[str] | None):
        span = Span(
            span_id=str(run_id),
            name=name,
            parent_id=str(parent_run_id) if parent_run_id else None,
            start_ts=time.monotonic(),
            kind=kind,
            tags=tags or [],
        )
        self.spans[span.span_id] = span
        self._order.append(span.span_id)

    def _record_end(self, run_id, status: str, error: str = ""):
        span = self.spans.get(str(run_id))
        if span is None:
            return  # a span we didn't see start — don't crash the run over a trace gap
        span.end_ts = time.monotonic()
        span.status = status
        span.error = error

    # --- chain (node) events ---

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name", "unknown_chain")
        self._record_start(run_id, parent_run_id, name, "chain", tags)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._record_end(run_id, "ok")

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record_end(run_id, "error", str(error))

    # --- llm events (nested under whichever node called them) ---

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, **kwargs):
        name = (serialized or {}).get("name", "llm_call")
        self._record_start(run_id, parent_run_id, name, "llm", tags)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._record_end(run_id, "ok")

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._record_end(run_id, "error", str(error))

    # --- output ---

    def node_level_spans(self) -> list[Span]:
        """Spans whose name matches a known graph node — the filtered view
        for a human-readable timeline. Falls back to ALL top-level chain
        spans if no known node name matched anything, so a naming mismatch
        degrades to 'show everything' rather than 'show nothing' silently."""
        known = [s for s in self.spans.values() if s.name in KNOWN_NODE_NAMES]
        if known:
            return sorted(known, key=lambda s: s.start_ts)
        top_level = [s for s in self.spans.values() if s.parent_id is None and s.kind == "chain"]
        return sorted(top_level, key=lambda s: s.start_ts)

    def summary(self) -> dict:
        nodes = self.node_level_spans()
        return {
            "run_id": self.run_id,
            "total_spans_captured": len(self.spans),
            "node_spans": [
                {"name": s.name, "duration_ms": s.duration_ms, "status": s.status, "error": s.error}
                for s in nodes
            ],
        }

    def save(self, path: Path | None = None) -> Path:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = path or (TRACE_DIR / f"{self.run_id}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "spans": [asdict(s) for s in self.spans.values()],
                    "summary": self.summary(),
                },
                f,
                indent=2,
            )
        return out_path