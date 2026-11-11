"""Evaluate nanobot hybrid RAG against the AcmeTech enterprise test set.

Usage:
  python scripts/eval_acmetech_rag.py --dataset D:\\agent\\acmetech-enterprise-rag-dataset-main
  python scripts/eval_acmetech_rag.py --dataset ... --skip-ingest
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.webui.knowledge_api import open_knowledge_store


def _parse_sources(raw: str | None) -> list[str]:
    if not raw or raw.strip().upper() == "N/A":
        return []
    parts = re.split(r"\s*\+\s*", raw.strip())
    out: list[str] = []
    for part in parts:
        name = Path(part.replace("\\", "/")).name.lower()
        if name:
            out.append(name)
    return out


def _hit_filenames(hits) -> set[str]:
    return {(h.filename or "").lower() for h in hits if h.filename}


def ingest_all(store, docs_root: Path) -> dict:
    pdfs = sorted(docs_root.rglob("*.pdf"))
    # Clear existing docs for a clean eval corpus.
    for doc in list(store.list_documents()):
        store.delete_document(doc["id"])
    ok = 0
    errors: list[dict[str, str]] = []
    t0 = time.time()
    for path in pdfs:
        try:
            store.add_document(path.name, path.read_bytes())
            ok += 1
            print(f"  indexed {ok}/{len(pdfs)} {path.relative_to(docs_root)}")
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)[:300]})
            print(f"  FAIL {path.name}: {exc}")
    return {
        "indexed": ok,
        "total": len(pdfs),
        "errors": errors,
        "seconds": round(time.time() - t0, 1),
        "vector_enabled": store.vector_enabled,
    }


def evaluate(store, questions: list[dict], *, k: int = 5) -> dict:
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_cat: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict] = []

    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        qtype = q.get("question_type", "unknown")
        cat = q.get("category", "unknown")
        expected = _parse_sources(q.get("source_document"))
        hits = store.search(q["question"], k=k)
        found = _hit_filenames(hits)
        ranked = [(h.filename or "").lower() for h in hits]

        if qtype == "negative":
            # For negatives, "good" retrieval is empty or low-confidence noise.
            # We report whether any hit was returned; generation faithfulness is separate.
            any_hit = bool(hits)
            by_type[qtype]["total"] += 1
            by_type[qtype]["returned_hits"] += int(any_hit)
            by_cat[cat]["total"] += 1
            rows.append(
                {
                    "id": qid,
                    "type": qtype,
                    "category": cat,
                    "any_hit": any_hit,
                    "top_files": ranked[:k],
                }
            )
        else:
            recall = bool(expected) and all(doc in found for doc in expected)
            recall_any = bool(expected) and any(doc in found for doc in expected)
            # MRR over first expected doc appearance
            mrr = 0.0
            if expected:
                for rank, name in enumerate(ranked, start=1):
                    if name in expected:
                        mrr = 1.0 / rank
                        break
            by_type[qtype]["total"] += 1
            by_type[qtype]["recall_all"] += int(recall)
            by_type[qtype]["recall_any"] += int(recall_any)
            by_type[qtype]["mrr_sum"] += mrr
            by_cat[cat]["total"] += 1
            by_cat[cat]["recall_all"] += int(recall)
            by_cat[cat]["recall_any"] += int(recall_any)
            rows.append(
                {
                    "id": qid,
                    "type": qtype,
                    "category": cat,
                    "expected": expected,
                    "recall_all": recall,
                    "recall_any": recall_any,
                    "mrr": mrr,
                    "top_files": ranked[:k],
                    "sources": [h.source for h in hits[:k]],
                }
            )
        if i % 25 == 0 or i == len(questions):
            print(f"  evaluated {i}/{len(questions)}")

    summary = {
        "k": k,
        "vector_enabled": store.vector_enabled,
        "search_mode": store.search_mode,
        "by_type": {},
        "by_category": {},
    }
    for qtype, c in by_type.items():
        total = max(c["total"], 1)
        if qtype == "negative":
            summary["by_type"][qtype] = {
                "total": c["total"],
                "returned_hits_rate": round(c["returned_hits"] / total, 3),
                "note": "lower is better for negative (less spurious retrieval)",
            }
        else:
            summary["by_type"][qtype] = {
                "total": c["total"],
                "recall_all@k": round(c["recall_all"] / total, 3),
                "recall_any@k": round(c["recall_any"] / total, 3),
                "mrr": round(c["mrr_sum"] / total, 3),
            }
    for cat, c in by_cat.items():
        total = max(c["total"], 1)
        if c["recall_all"] or c["recall_any"]:
            summary["by_category"][cat] = {
                "total": c["total"],
                "recall_all@k": round(c["recall_all"] / total, 3),
                "recall_any@k": round(c["recall_any"] / total, 3),
            }
        else:
            summary["by_category"][cat] = {"total": c["total"]}

    # Overall grounded (exclude negative)
    grounded = [r for r in rows if r["type"] != "negative"]
    if grounded:
        summary["overall_grounded"] = {
            "total": len(grounded),
            "recall_all@k": round(sum(r["recall_all"] for r in grounded) / len(grounded), 3),
            "recall_any@k": round(sum(r["recall_any"] for r in grounded) / len(grounded), 3),
            "mrr": round(sum(r["mrr"] for r in grounded) / len(grounded), 3),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\agent\acmetech-enterprise-rag-dataset-main"),
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=0, help="Override tools.knowledge.chunkSize")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=-1,
        help="Override tools.knowledge.chunkOverlap (-1 keeps config)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Eval workspace (default: ~/.nanobot/workspace or *_c{chunk})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="DashVector collection name (reuse one collection across chunk-size A/Bs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".nanobot" / "eval" / "acmetech_rag_eval.json",
    )
    args = parser.parse_args()

    dataset = args.dataset
    docs_root = dataset / "AcmeTech"
    testset = dataset / "AcmeTech_RAG_TestSet.json"
    if not docs_root.is_dir() or not testset.is_file():
        raise SystemExit(f"dataset incomplete under {dataset}")

    cfg = load_config()
    kb = cfg.tools.knowledge
    if args.chunk_size > 0:
        kb.chunk_size = int(args.chunk_size)
    if args.chunk_overlap >= 0:
        kb.chunk_overlap = int(args.chunk_overlap)
    # Prefer an explicit collection; otherwise reuse a shared eval collection.
    # (DashVector free tier often limits to 2 collections.)
    parser_collection = getattr(args, "collection", "") or ""
    if parser_collection:
        kb.dashvector_collection = parser_collection
    else:
        base_coll = kb.dashvector_collection or "nanobot_enterprise_docs"
        kb.dashvector_collection = f"{base_coll}_eval"

    if args.workspace is not None:
        workspace = args.workspace
    elif args.chunk_size > 0:
        workspace = Path.home() / ".nanobot" / f"workspace_eval_c{kb.chunk_size}"
    else:
        workspace = Path.home() / ".nanobot" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    store = open_knowledge_store(workspace, cfg.tools)
    print(
        f"store vector_enabled={store.vector_enabled} mode={store.search_mode} "
        f"workspace={workspace} chunk_size={store.chunk_size} "
        f"overlap={store.chunk_overlap} collection={kb.dashvector_collection}"
    )

    ingest_report = None
    if not args.skip_ingest:
        print("Ingesting AcmeTech PDFs…")
        ingest_report = ingest_all(store, docs_root)
        print("ingest:", json.dumps(ingest_report, ensure_ascii=False))
        # reopen in case collection metadata changed
        store = open_knowledge_store(workspace, cfg.tools)

    questions = json.loads(testset.read_text(encoding="utf-8"))["questions"]
    print(f"Evaluating {len(questions)} questions (k={args.k})…")
    result = evaluate(store, questions, k=args.k)
    payload = {
        "dataset": str(dataset),
        "chunk_size": store.chunk_size,
        "chunk_overlap": store.chunk_overlap,
        "collection": kb.dashvector_collection,
        "workspace": str(workspace),
        "ingest": ingest_report,
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
