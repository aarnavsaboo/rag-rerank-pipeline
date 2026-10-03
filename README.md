# rag-rerank-pipeline

A modular retrieval pipeline for experimenting with multi-stage RAG on local models.

The repository keeps retrieval stages explicit: lexical retrieval, embedding retrieval, rank fusion, optional diversity selection, optional cross-encoder reranking, evidence packing and local generation. Every stage emits inspectable records and timing data so a more complicated pipeline can be compared against a simpler baseline.

It is designed for workflow experiments rather than as a large application framework.

## Pipeline

```text
documents
   |
   v
chunk records
   |
   +---------------------+
   |                     |
   v                     v
BM25 retrieval      dense retrieval
   |                     |
   +----------+----------+
              |
              v
       reciprocal-rank fusion
              |
              v
         candidate pool
              |
        +-----+------+
        |            |
        v            v
     reranker     MMR diversity
        |            |
        +-----+------+
              |
              v
        evidence packer
              |
              v
        local generator
              |
              v
       answer + trace row
```

## Design goals

- each stage can be disabled independently
- every query produces stage-by-stage timing
- retrieval and generation results are stored separately
- batch runs use a declarative manifest
- expensive reranking can be limited to a small candidate pool
- evidence packing obeys a configurable token/character budget
- local generators are adapters rather than hard-coded dependencies
- pipeline runs remain reproducible from JSONL artifacts

## Example

```bash
python -m rag_rerank_pipeline build examples/corpus.jsonl --out runs/index.json

python -m rag_rerank_pipeline run \
  configs/pipeline.example.json \
  examples/queries.jsonl \
  --out runs/results.jsonl

python -m rag_rerank_pipeline report runs/results.jsonl
```

## Experiments this supports

- lexical-only baseline
- dense-only baseline
- BM25 + embeddings with reciprocal-rank fusion
- top-50 retrieve → top-10 rerank
- MMR before or after reranking
- small vs large reranker
- reranker latency vs retrieval gain
- evidence budget sweeps
- local generation with different small models
- generation with and without reranked evidence
- query-level failure analysis

## Output model

A result record contains query ID, rankings from each stage, selected evidence IDs, per-stage elapsed time, generation metadata and retrieval metrics when labels are present.

That means a final answer can be traced back through the exact ranked candidates that produced it without hiding the pipeline behind a framework object.

## Repository layout

- `retrieval.py` — lightweight BM25
- `dense.py` — embedding adapter and cosine ranking
- `fusion.py` — rank fusion
- `rerank.py` — optional cross-encoder adapter
- `diversity.py` — MMR selection
- `packing.py` — evidence-budget packing
- `generation.py` — local model adapters
- `pipeline.py` — stage orchestration and timings
- `batch.py` — JSONL workflow runner
- `metrics.py` — Recall/MRR/nDCG
- `report.py` — grouped run summaries
- `configs/` — experiment manifests

Maintained by **Aarnav Saboo**.
