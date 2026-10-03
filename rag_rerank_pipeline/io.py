from pathlib import Path
import json

from .records import Document, Query


def load_documents(path: str) -> list[Document]:
    return [
        Document(str(row["id"]), str(row["text"]), str(row.get("source","")))
        for row in (
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]


def load_queries(path: str) -> list[Query]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(Query(str(row["id"]), str(row["text"]), tuple(row.get("relevant", []))))
    return rows


def write_jsonl(path: str, rows: list[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
        encoding="utf-8",
    )
