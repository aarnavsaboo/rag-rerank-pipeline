from math import sqrt
from typing import Protocol

from .records import Document, Hit


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def _norm(v: list[float]) -> list[float]:
    n = sqrt(sum(x*x for x in v))
    return [x / max(n, 1e-12) for x in v]


class DenseIndex:
    def __init__(self, documents: list[Document], embedder: Embedder):
        self.documents = documents
        self.embedder = embedder
        self.vectors = [_norm(x) for x in embedder.encode([d.text for d in documents])]

    def search(self, query: str, k: int = 20) -> list[Hit]:
        q = _norm(self.embedder.encode([query])[0])
        rows = []
        for doc, vector in zip(self.documents, self.vectors):
            score = sum(a*b for a,b in zip(q, vector))
            rows.append(Hit(doc.id, doc.text, score, doc.source))
        return sorted(rows, key=lambda x: (-x.score, x.id))[:k]
