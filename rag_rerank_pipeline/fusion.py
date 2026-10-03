from collections import defaultdict

from .records import Hit


def rrf(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    score = defaultdict(float)
    hit_by_id = {}
    for ranking, weight in zip(rankings, weights):
        for rank, hit in enumerate(ranking, 1):
            score[hit.id] += weight / (k + rank)
            hit_by_id[hit.id] = hit
    return [
        Hit(hit_by_id[doc_id].id, hit_by_id[doc_id].text, value, hit_by_id[doc_id].source)
        for doc_id, value in sorted(score.items(), key=lambda x: (-x[1], x[0]))
    ]
