from argparse import ArgumentParser
from pathlib import Path
import json

from .dense import DenseIndex, SentenceTransformerEmbedder
from .generation import OllamaGenerator
from .io import load_documents, load_queries, write_jsonl
from .pipeline import Pipeline, PipelineConfig
from .report import summarize
from .rerank import CrossEncoderReranker
from .retrieval import BM25


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("queries")
    run.add_argument("--corpus", required=True)
    run.add_argument("--out", required=True)

    report = sub.add_parser("report")
    report.add_argument("path")

    args = parser.parse_args()

    if args.cmd == "report":
        print(json.dumps(summarize(read_rows(args.path)), indent=2))
        return

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    documents = load_documents(args.corpus)
    queries = load_queries(args.queries)

    lexical = BM25(documents)

    dense = None
    embedding_model = cfg.get("embedding_model")
    if embedding_model:
        embedder = SentenceTransformerEmbedder(embedding_model)
        dense = DenseIndex(documents, embedder)

    reranker = None
    reranker_model = cfg.get("reranker_model")
    if reranker_model:
        reranker = CrossEncoderReranker(reranker_model)

    generator = None
    generator_model = cfg.get("generator_model")
    if generator_model:
        generator = OllamaGenerator(
            generator_model,
            cfg.get("generator_endpoint", "http://127.0.0.1:11434"),
        )

    pipeline_config = PipelineConfig(
        retrieve_k=int(cfg.get("retrieve_k", 30)),
        rerank_k=int(cfg.get("rerank_k", 10)),
        evidence_chars=int(cfg.get("evidence_chars", 12000)),
        fusion_k=int(cfg.get("fusion_k", 60)),
        lexical_weight=float(cfg.get("lexical_weight", 1.0)),
        dense_weight=float(cfg.get("dense_weight", 1.0)),
    )
    pipeline = Pipeline(
        lexical=lexical,
        dense=dense,
        reranker=reranker,
        generator=generator,
        config=pipeline_config,
    )
    rows = [pipeline.run(query) for query in queries]
    write_jsonl(args.out, rows)
    print(json.dumps({"queries":len(rows),"out":args.out}))


if __name__ == "__main__":
    main()
