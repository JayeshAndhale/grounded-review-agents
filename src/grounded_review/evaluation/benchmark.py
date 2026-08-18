"""Evaluation benchmark: a small set of topic-clustered papers, plus known
invalid IDs to exercise the Phase 1 refusal path.

Clustered by topic, not just concatenated, so the scheduler's cross-cutting
decomposition and retrieval scoping get genuinely exercised - a flat pile of
unrelated papers would let every review degenerate into per-paper summaries
by default, the exact failure mode the scheduler prompt exists to prevent.
"""

from dataclasses import dataclass


@dataclass
class BenchmarkTopic:
    name: str
    topic_prompt: str  # fed to ReviewState.topic
    arxiv_ids: list[str]


BENCHMARK_TOPICS: list[BenchmarkTopic] = [
    BenchmarkTopic(
        name="code_generation_benchmarks",
        topic_prompt="How do current benchmarks evaluate code generation from large language models?",
        arxiv_ids=["2211.11501", "2406.15877"],  # DS-1000, BigCodeBench - already ingested (Phase 2)
    ),
    BenchmarkTopic(
        name="retrieval_augmented_generation",
        topic_prompt="How has retrieval-augmented generation evolved as an approach to grounding language model outputs?",
        arxiv_ids=[
            "2005.11401",  # Lewis et al. - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
            "2002.08909",  # Guu et al. - REALM: Retrieval-Augmented Language Model Pre-Training
            "2004.04906",  # Karpukhin et al. - Dense Passage Retrieval for Open-Domain Question Answering
            "2310.11511",  # Asai et al. - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
        ],
    ),
    BenchmarkTopic(
        name="hallucination_detection",
        topic_prompt="What methods have been proposed to detect and reduce hallucination in large language model outputs?",
        arxiv_ids=[
            "2202.03629",  # Ji et al. - Survey of Hallucination in Natural Language Generation
            "2303.08896",  # Manakul et al. - SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
            "2305.14251",  # Min et al. - FActScore: Fine-grained Atomic Evaluation of Factual Precision
            "2309.11495",  # Dhuliawala et al. - Chain-of-Verification Reduces Hallucination in LLMs
        ],
    ),
]

# Known-invalid IDs to confirm PaperNotFoundError / normalize_arxiv_id's
# ValueError fire cleanly rather than crashing the harness mid-run. Mix of
# malformed (fails the regex) and well-formed-but-unreal (fails at fetch).
INVALID_ARXIV_IDS = [
    "9999.99999",       # well-formed pattern, does not exist on arXiv
    "not-an-arxiv-id",  # fails normalize_arxiv_id's regex entirely
    "1234.5",           # too few digits after the dot - fails the regex
]


def all_arxiv_ids() -> list[str]:
    """Every real paper ID across all topics, deduplicated - the full
    ingestion list for setting up the benchmark's vector store."""
    seen: set[str] = set()
    ids: list[str] = []
    for topic in BENCHMARK_TOPICS:
        for aid in topic.arxiv_ids:
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
    return ids