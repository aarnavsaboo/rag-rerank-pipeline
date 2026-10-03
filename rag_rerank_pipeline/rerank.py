from .records import Hit


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.model_name = model
        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, hit.text) for hit in hits]).tolist()
        rows = [
            Hit(hit.id, hit.text, float(score), hit.source)
            for hit, score in zip(hits, scores)
        ]
        return sorted(rows, key=lambda x: (-x.score, x.id))[:k]
