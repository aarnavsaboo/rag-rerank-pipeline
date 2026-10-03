from math import sqrt

from .records import Hit


def _cosine(a: list[float], b: list[float]) -> float:
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(x*x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x*y for x,y in zip(a,b)) / (na*nb)


def mmr(
    hits: list[Hit],
    vectors: dict[str, list[float]],
    k: int,
    lambda_mult: float = .7,
) -> list[Hit]:
    if not hits:
        return []
    selected: list[Hit] = []
    remaining = list(hits)
    max_score = max(abs(x.score) for x in hits) or 1.0
    while remaining and len(selected) < k:
        def utility(hit: Hit):
            relevance = hit.score / max_score
            redundancy = max(
                (_cosine(vectors[hit.id], vectors[x.id]) for x in selected),
                default=0.0,
            )
            return lambda_mult * relevance - (1 - lambda_mult) * redundancy
        chosen = max(remaining, key=lambda x: (utility(x), x.id))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
