from collections import Counter
from math import log
import re

from .records import Document, Hit


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = .75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(x.text)) for x in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter(term for counts in self.counts for term in counts)
        n = len(documents)
        self.idf = {term: log(1 + (n - freq + .5) / (freq + .5)) for term, freq in df.items()}

    def search(self, query: str, k: int = 20) -> list[Hit]:
        terms = set(tokenize(query))
        rows = []
        for doc, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / max(self.avg_len, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                rows.append(Hit(doc.id, doc.text, score, doc.source))
        return sorted(rows, key=lambda x: (-x.score, x.id))[:k]
