"""Retrieval evaluation: hit rate@k and MRR for BM25 vs dense vs hybrid,
broken down by question type. Writes reports/retrieval_eval.json/.md.

Run: python eval/evaluate.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag import BM25Retriever, DenseRetriever, HybridRetriever, answer, load_chunks  # noqa: E402

K = 3
qs = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
in_scope = [q for q in qs if q["section"]]
chunks = load_chunks()
retrievers = [BM25Retriever(chunks), DenseRetriever(chunks), HybridRetriever(chunks)]


def rank_of(hits, section):
    for r, (c, _) in enumerate(hits, start=1):
        if c.section == section:
            return r
    return None


results = {}
for ret in retrievers:
    by_type = {}
    for q in in_scope:
        r = rank_of(ret.search(q["q"], k=10), q["section"])
        by_type.setdefault(q["type"], []).append(r)
    all_ranks = [r for v in by_type.values() for r in v]
    row = {"hit@1": sum(r == 1 for r in all_ranks) / len(all_ranks),
           f"hit@{K}": sum(r is not None and r <= K for r in all_ranks) / len(all_ranks),
           "mrr": sum(1 / r for r in all_ranks if r) / len(all_ranks)}
    for t, ranks in by_type.items():
        row[f"hit@{K}_{t}"] = sum(r is not None and r <= K for r in ranks) / len(ranks)
    results[ret.name] = {k: round(v, 3) for k, v in row.items()}

# End-to-end check of the answer step on the hybrid retriever
hyb = retrievers[2]
e2e, refusals = [], []
for q in qs:
    out = answer(q["q"], hyb.search(q["q"], k=3))
    if q["answer_contains"]:
        e2e.append(q["answer_contains"].lower() in out["answer"].lower())
    else:
        refusals.append(out["answer"].startswith("I can't find"))
results["answer_step"] = {"mode": out["mode"],
                          "answer_contains_expected": round(sum(e2e) / len(e2e), 3),
                          "out_of_scope_refused": f"{sum(refusals)}/{len(refusals)}"}

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "retrieval_eval.json").write_text(json.dumps(results, indent=2))
print(f"Dense backend: {retrievers[1].backend}; chunks: {len(chunks)}; questions: {len(in_scope)} in-scope + {len(qs)-len(in_scope)} out-of-scope\n")
cols = ["hit@1", f"hit@{K}", "mrr", f"hit@{K}_keyword", f"hit@{K}_paraphrase"]
md = "| Retriever | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
for name in ("bm25", "dense", "hybrid"):
    md += f"| {name} | " + " | ".join(f"{results[name][c]:.2f}" for c in cols) + " |\n"
(ROOT / "reports" / "retrieval_eval.md").write_text(md)
print(md)
print(results["answer_step"])
