"""Load documents and split them into section-aware chunks with metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.doc} > {self.section}"


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on '## ' headings; documents without headings become one section."""
    parts = re.split(r"(?m)^##\s+(.+)$", text)
    if len(parts) == 1:
        return [("Document", text.strip())]
    out = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].strip()
        if body:
            out.append((parts[i].strip(), body))
    return out


def window(words: list[str], size: int, overlap: int) -> list[str]:
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def load_chunks(docs_dir: Path = DOCS_DIR, max_words: int = 180, overlap: int = 40) -> list[Chunk]:
    """Section-aware chunking: keep each heading's body intact unless it is long,
    then fall back to overlapping word windows. The heading is prepended to every
    chunk so the retriever sees the topic even in a mid-section window."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        text = _read(path)
        title_match = re.search(r"(?m)^#\s+(.+)$", text)
        doc = title_match.group(1).split(":")[-1].strip() if title_match else path.stem
        for section, body in split_sections(text):
            body = re.sub(r"(?m)^>.*$", "", body).strip()  # drop disclaimer blockquotes
            for j, piece in enumerate(window(body.split(), max_words, overlap)):
                chunks.append(Chunk(id=f"{path.stem}::{section}::{j}", doc=doc, section=section,
                                    text=f"{section}. {piece}", meta={"file": path.name}))
    return chunks
