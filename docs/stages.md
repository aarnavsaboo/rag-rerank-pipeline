# Stage notes

The pipeline separates candidate generation from expensive scoring.

A broad first-stage retriever can optimize recall. A smaller reranker pool can then spend more compute on ordering the candidates. Evidence packing is performed only after the final ranking so generation does not receive passages that were already discarded.

Every additional stage should earn its latency. The most important comparison is often not between two complicated pipelines, but between the complicated pipeline and a lexical-only or dense-only baseline.
