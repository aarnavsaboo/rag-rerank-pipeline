from dataclasses import dataclass

from .records import Hit


@dataclass(frozen=True)
class PackedEvidence:
    text: str
    ids: tuple[str, ...]
    chars: int


def pack(hits: list[Hit], max_chars: int = 12000) -> PackedEvidence:
    sections = []
    ids = []
    used = 0
    for index, hit in enumerate(hits, 1):
        block = f"[{index}] source={hit.id}\n{hit.text.strip()}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 80:
                sections.append(block[:remaining])
                ids.append(hit.id)
                used += remaining
            break
        sections.append(block)
        ids.append(hit.id)
        used += len(block)
    return PackedEvidence("\n".join(sections), tuple(ids), used)
