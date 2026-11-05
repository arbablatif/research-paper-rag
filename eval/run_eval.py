"""Evaluate the RAG system on eval/qa.json.

Metrics (deterministic, no LLM judge, so the only API calls are the 15 answers):
  - answer accuracy     : answerable Qs where ALL expected key facts appear in the answer
  - citation correctness: answerable Qs where some citation matches an expected (doc, page±1)
  - citation grounding  : share of all emitted citations that point to a retrieved passage
  - refusal accuracy    : unanswerable Qs that were correctly refused
  - false refusals      : answerable Qs that were wrongly refused

Usage:  python eval/run_eval.py [--mock] [--out eval/results.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.config import settings  # noqa: E402
from src.rag.llm import LLM  # noqa: E402
from src.rag.pipeline import RAG, ingest  # noqa: E402
from src.rag.store import VectorStore  # noqa: E402


def norm(s: str) -> str:
    """Lowercase, unify dashes, strip punctuation except . so numbers like 0.9 survive."""
    s = s.lower().replace(r"\mu", "mu").replace("$", "").replace("−", "-").replace("–", "-").replace("—", "-").replace("β", "beta")
    s = re.sub(r"[^\w\s.\-]", " ", s)
    return re.sub(r"\s+", " ", s)


def facts_present(answer: str, facts: list) -> bool:
    """Each fact is a string or a list of acceptable alternatives."""
    a = norm(answer)
    return all(any(norm(alt) in a for alt in (f if isinstance(f, list) else [f])) for f in facts)


def citation_ok(cites: list[dict], expected: list[dict]) -> bool:
    return any(c["doc"] == e["doc"] and abs(c["page"] - e["page"]) <= 1
               for c in cites for e in expected)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = ap.parse_args()

    qa = json.loads((ROOT / "eval" / "qa.json").read_text(encoding="utf-8"))
    store = VectorStore()
    ingest(settings.papers_dir, store)
    rag = RAG(store, LLM("mock" if args.mock else None))

    rows = []
    for item in qa:
        res = rag.answer(item["question"])
        answerable = item["answerable"]
        row = {
            "id": item["id"], "question": item["question"], "answerable": answerable,
            "answer": res["answer"], "refused": res["refused"],
            "citations": [{"doc": c["doc"], "page": c["page"], "grounded": c["grounded"]}
                          for c in res["citations"]],
            "top_hit": f"{res['hits'][0]['doc']} p.{res['hits'][0]['page']} "
                       f"({res['hits'][0]['score']:.2f})" if res["hits"] else None,
        }
        if answerable:
            row["correct"] = (not res["refused"]) and facts_present(res["answer"], item["facts"])
            row["citation_correct"] = citation_ok(res["citations"], item["sources"])
        else:
            row["correct"] = res["refused"]
        rows.append(row)
        flag = "OK " if row["correct"] else "BAD"
        cite = "" if not answerable else (" cite:OK" if row["citation_correct"] else " cite:BAD")
        print(f"[{flag}]{cite} {item['id']}: {item['question'][:70]}")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    all_cites = [c for r in rows for c in r["citations"]]
    summary = {
        "provider": rag.llm.provider, "model": rag.llm.model, "n": len(rows),
        "answer_accuracy": sum(r["correct"] for r in ans) / len(ans),
        "citation_correctness": sum(r["citation_correct"] for r in ans) / len(ans),
        "citation_grounding": (sum(c["grounded"] for c in all_cites) / len(all_cites)) if all_cites else 0.0,
        "refusal_accuracy": sum(r["correct"] for r in una) / len(una) if una else None,
        "false_refusals": sum(r["refused"] for r in ans),
        "overall_accuracy": sum(r["correct"] for r in rows) / len(rows),
    }
    print("\n" + json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
