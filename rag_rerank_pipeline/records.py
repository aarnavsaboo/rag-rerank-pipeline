from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    score: float
    source: str = ""
