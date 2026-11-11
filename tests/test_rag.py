import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import BM25Retriever, HybridRetriever, load_chunks  # noqa: E402
from rag.generate import REFUSAL, extractive_answer  # noqa: E402
from rag.ingest import split_sections, window  # noqa: E402


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


def test_chunks_have_metadata(chunks):
    assert len(chunks) > 20
    assert all(c.doc and c.section and c.text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_disclaimer_not_indexed(chunks):
    assert not any("fictional insurer" in c.text for c in chunks)


def test_split_sections():
    secs = split_sections("# T\n\n## A\nalpha\n## B\nbeta")
    assert secs == [("A", "alpha"), ("B", "beta")]


def test_window_overlap():
    w = window([str(i) for i in range(10)], size=4, overlap=2)
    assert w[0] == "0 1 2 3" and w[1] == "2 3 4 5"


def test_bm25_finds_exact_term(chunks):
    top = BM25Retriever(chunks).search("OCF-1 application form", k=1)[0][0]
    assert top.section == "Applying for Benefits"


def test_hybrid_returns_k_unique(chunks):
    hits = HybridRetriever(chunks, backend="lsa").search("windshield deductible", k=4)
    assert len(hits) == 4 and len({c.id for c, _ in hits}) == 4


def test_extractive_refuses_without_evidence(chunks):
    hits = BM25Retriever(chunks).search("best pizza in Toronto", k=3)
    assert extractive_answer("best pizza in Toronto", hits) == REFUSAL
