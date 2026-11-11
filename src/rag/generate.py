"""Grounded answer generation with citations.

Uses any OpenAI-compatible chat endpoint (OpenAI, Azure OpenAI, or a local
Ollama server via OPENAI_BASE_URL=http://localhost:11434/v1). With no key, it
falls back to an extractive answer built from the retrieved text, so the
pipeline remains testable offline.
"""
from __future__ import annotations

import os
import re

from .ingest import Chunk
from .retrieve import tokenize

SYSTEM_PROMPT = """You are a helpful assistant answering questions about an auto insurance policy.
Answer ONLY from the numbered context passages. Cite every factual claim with its passage
number in square brackets, e.g. [2]. If the context does not contain the answer, reply exactly:
"I can't find that in the policy documents." Do not use outside knowledge. Be concise."""

REFUSAL = "I can't find that in the policy documents."


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation}) {c.text}" for i, (c, _) in enumerate(hits, start=1))


def llm_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}],
    )
    return resp.choices[0].message.content.strip()


AMOUNT_CUES = re.compile(r"\b(how much|deductible|cost|limit|maximum|pay|paid|refund|discount|price)\b", re.I)
GENERIC = {"policy", "insurance", "covered", "cover", "coverage", "car", "vehicle", "my", "i", "do", "does",
           "what", "how", "much", "can", "will", "get", "is", "am", "there"}


def extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Offline fallback (no LLM): pick the sentences that best match the question,
    preferring higher-ranked passages, and add the passage's amount sentence for
    'how much'-style questions. Refuses when no passage shares a content word."""
    q = set(tokenize(question)) - GENERIC
    scored = []
    for i, (chunk, _) in enumerate(hits[:3], start=1):
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                scored.append((overlap + 1.0 / i, i, sent.strip()))
    if not scored:
        return REFUSAL
    scored.sort(key=lambda t: -t[0])
    picked = [scored[0]]
    if AMOUNT_CUES.search(question) and not re.search(r"[$%]|\d", picked[0][2]):
        src = picked[0][1]
        for sent in re.split(r"(?<=[.!?])\s+", hits[src - 1][0].text):
            if re.search(r"[$%]\s?\d|\d+\s?%|\$\d", sent):
                picked.append((0, src, sent.strip()))
                break
    if len(picked) == 1 and len(scored) > 1 and scored[1][2] != picked[0][2]:
        picked.append(scored[1])
    return " ".join(f"{s} [{i}]" for _, i, s in picked)


def answer(question: str, hits: list[tuple[Chunk, float]]) -> dict:
    mode = "llm" if os.getenv("OPENAI_API_KEY") else "extractive"
    text = llm_answer(question, hits) if mode == "llm" else extractive_answer(question, hits)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
    return {"answer": text, "mode": mode,
            "sources": [hits[n - 1][0].citation for n in cited]}
