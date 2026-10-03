from math import log2


def evaluate(ranking: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    top = ranking[:k]
    recall = 0.0 if not relevant else len(set(top) & relevant) / len(relevant)
    mrr = next((1 / i for i, item in enumerate(top, 1) if item in relevant), 0.0)
    dcg = sum(1 / log2(i + 1) for i,item in enumerate(top,1) if item in relevant)
    ideal = sum(1 / log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {"recall": recall, "mrr": mrr, "ndcg": 0.0 if ideal == 0 else dcg / ideal}
