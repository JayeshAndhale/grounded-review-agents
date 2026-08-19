# scripts/test_tracer.py
from grounded_review.agents.graph import build_graph
from grounded_review.agents.state import ReviewState
from grounded_review.observability.tracer import GraphTracer

def main():
    graph = build_graph()
    state = ReviewState(
        topic="How do current benchmarks evaluate code generation from large language models?",
        arxiv_ids=["2211.11501", "2406.15877"],
    )
    tracer = GraphTracer()
    graph.invoke(state, config={"callbacks": [tracer]})

    print(f"Captured {len(tracer.spans)} total spans")
    for s in tracer.node_level_spans():
        print(f"  {s.name}: {s.duration_ms}ms ({s.status})")

    path = tracer.save()
    print(f"Saved trace to {path}")

if __name__ == "__main__":
    main()