from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .fusion import rrf
from .metrics import evaluate
from .packing import pack
from .records import Query


@dataclass(frozen=True)
class PipelineConfig:
    retrieve_k: int = 30
    rerank_k: int = 10
    evidence_chars: int = 12000
    fusion_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0


class Pipeline:
    def __init__(self, lexical, dense=None, reranker=None, generator=None, config=None):
        self.lexical = lexical
        self.dense = dense
        self.reranker = reranker
        self.generator = generator
        self.config = config or PipelineConfig()

    def run(self, query: Query) -> dict:
        timing = {}

        started = perf_counter()
        lexical_hits = self.lexical.search(query.text, self.config.retrieve_k)
        timing["lexical_ms"] = (perf_counter() - started) * 1000

        dense_hits = []
        if self.dense is not None:
            started = perf_counter()
            dense_hits = self.dense.search(query.text, self.config.retrieve_k)
            timing["dense_ms"] = (perf_counter() - started) * 1000

        started = perf_counter()
        rankings = [lexical_hits] + ([dense_hits] if dense_hits else [])
        weights = [self.config.lexical_weight] + ([self.config.dense_weight] if dense_hits else [])
        candidates = rrf(rankings, self.config.fusion_k, weights)
        timing["fusion_ms"] = (perf_counter() - started) * 1000

        final_hits = candidates[:self.config.rerank_k]
        if self.reranker is not None:
            started = perf_counter()
            final_hits = self.reranker.rerank(query.text, candidates[:self.config.retrieve_k], self.config.rerank_k)
            timing["rerank_ms"] = (perf_counter() - started) * 1000

        started = perf_counter()
        evidence = pack(final_hits, self.config.evidence_chars)
        timing["packing_ms"] = (perf_counter() - started) * 1000

        generation = None
        if self.generator is not None:
            started = perf_counter()
            generation = self.generator.generate(query.text, evidence.text)
            timing["generation_ms"] = (perf_counter() - started) * 1000

        metrics = None
        if query.relevant:
            metrics = evaluate([x.id for x in final_hits], set(query.relevant), self.config.rerank_k)

        return {
            "query_id": query.id,
            "query": query.text,
            "lexical_ids": [x.id for x in lexical_hits],
            "dense_ids": [x.id for x in dense_hits],
            "final_ids": [x.id for x in final_hits],
            "evidence_ids": list(evidence.ids),
            "evidence_chars": evidence.chars,
            "timing_ms": timing,
            "metrics": metrics,
            "generation": generation,
        }
