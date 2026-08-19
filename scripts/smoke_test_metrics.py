# scripts/smoke_test_metrics.py
from rich.console import Console
from rich.panel import Panel

from grounded_review.evaluation.benchmark import BENCHMARK_TOPICS
from grounded_review.evaluation.metrics import ResultStore, run_single

console = Console()

def main():
    topic = next(t for t in BENCHMARK_TOPICS if t.name == "code_generation_benchmarks")
    store = ResultStore()

    console.print(Panel(f"Topic: {topic.name}\nCondition: baseline\nRun: 1", title="Smoke test — single cell"))

    result = run_single(topic, "baseline", run_number=1)

    console.rule("Result")
    console.print(f"Status: {result.status}")
    if result.status == "failed":
        console.print(f"[red]{result.error_type}: {result.error}[/red]")
    else:
        console.print(f"Scoring: {result.scoring}")
        console.print(f"Reviewer revisions: {result.reviewer_revision_count}")
        console.print(f"Elapsed: {result.elapsed_seconds}s")
        console.print(f"Strong-tier calls: {result.token_summary.get('strong', {}).get('calls')}")

    store.append(result)
    console.rule("Checkpoint")
    console.print(f"Written to {store.path}")
    console.print(f"Completed keys now: {store.completed_keys()}")

if __name__ == "__main__":
    main()