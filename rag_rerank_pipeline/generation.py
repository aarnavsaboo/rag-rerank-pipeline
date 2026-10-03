from urllib.request import Request, urlopen
import json


class OllamaGenerator:
    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434"):
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def generate(self, query: str, evidence: str, max_tokens: int = 256) -> dict:
        prompt = (
            "Answer the question using the supplied evidence. "
            "If the evidence is insufficient, say so.\n\n"
            f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{query}"
        )
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }).encode()
        req = Request(
            self.endpoint + "/api/generate",
            data=body,
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urlopen(req, timeout=600) as response:
            row = json.load(response)
        return {
            "text": row.get("response", ""),
            "prompt_tokens": row.get("prompt_eval_count"),
            "output_tokens": row.get("eval_count"),
            "total_duration_ns": row.get("total_duration"),
        }
