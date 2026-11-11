"""LLM-as-judge end-to-end eval on AcmeTech RAG test set.

Pipeline per question:
  1) hybrid retrieve (kb store)
  2) generate grounded answer from contexts (DeepSeek)
  3) judge: correctness / faithfulness / abstention (DeepSeek)

Usage:
  python scripts/eval_acmetech_llm_judge.py
  python scripts/eval_acmetech_llm_judge.py --limit 20
  python scripts/eval_acmetech_llm_judge.py --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from nanobot.config.loader import load_config
from nanobot.webui.knowledge_api import open_knowledge_store

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _contexts_block(hits: list[Any], *, max_chars: int = 3500) -> str:
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        crumb = f" | {h.breadcrumb}" if h.breadcrumb else ""
        piece = f"[{i}] {h.filename}{crumb}\n{h.snippet}"
        if used + len(piece) > max_chars and parts:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts) if parts else "(no retrieved context)"


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


async def _chat(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def generate_answer(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    contexts: str,
) -> str:
    system = (
        "You are AcmeTech's enterprise knowledge assistant. "
        "Answer ONLY using the provided context. "
        "If the context is insufficient, reply exactly: "
        '"Not covered in the knowledge base." '
        "Be concise and factual. Do not invent policies, names, or numbers."
    )
    user = f"Context:\n{contexts}\n\nQuestion: {question}\n\nAnswer:"
    return await _chat(client, model=model, system=system, user=user, max_tokens=350)


async def judge_answer(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
    contexts: str,
    question_type: str,
) -> dict[str, Any]:
    if question_type == "negative":
        system = (
            "You are a strict RAG evaluator. Return ONLY JSON. "
            "The question should NOT be answerable from AcmeTech KB. "
            'Score abstention=1 if the answer refuses / says not covered / unknown; '
            "else 0. Also faithfulness=1 if no fabricated concrete company facts, else 0."
        )
        user = (
            f"Question: {question}\n"
            f"Model answer: {answer}\n"
            "Return JSON: "
            '{"abstention":0|1,"faithfulness":0|1,"correctness":0|1,"reason":"..."}'
        )
    else:
        system = (
            "You are a strict RAG evaluator. Return ONLY JSON with integer 0/1 scores. "
            "correctness=1 if the model answer captures the key facts of ground_truth "
            "(paraphrase OK; minor wording differences OK). "
            "faithfulness=1 if all claims in the answer are supported by the retrieved context "
            "(or are harmless refusals). Penalize invented numbers/names/policies."
        )
        user = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Retrieved context:\n{contexts}\n\n"
            f"Model answer: {answer}\n"
            "Return JSON: "
            '{"correctness":0|1,"faithfulness":0|1,"reason":"..."}'
        )
    raw = await _chat(client, model=model, system=system, user=user, max_tokens=250)
    data = _parse_json(raw)
    out = {
        "correctness": int(data.get("correctness", 0) or 0),
        "faithfulness": int(data.get("faithfulness", 0) or 0),
        "abstention": int(data.get("abstention", 0) or 0),
        "reason": str(data.get("reason") or "")[:300],
        "raw": raw[:500],
    }
    if question_type == "negative":
        # For negatives, correctness mirrors successful abstention.
        out["correctness"] = out["abstention"]
    return out


async def eval_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    store,
    model: str,
    q: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    async with sem:
        hits = await asyncio.to_thread(lambda: store.search(q["question"], k=k))
        contexts = _contexts_block(hits)
        answer = await generate_answer(
            client, model=model, question=q["question"], contexts=contexts
        )
        verdict = await judge_answer(
            client,
            model=model,
            question=q["question"],
            ground_truth=q.get("ground_truth_answer") or "",
            answer=answer,
            contexts=contexts,
            question_type=q.get("question_type") or "single_hop",
        )
        return {
            "id": q["id"],
            "question_type": q.get("question_type"),
            "category": q.get("category"),
            "difficulty": q.get("difficulty"),
            "question": q["question"],
            "ground_truth": q.get("ground_truth_answer"),
            "source_document": q.get("source_document"),
            "top_files": [h.filename for h in hits],
            "answer": answer,
            **verdict,
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    api_key = cfg.get_api_key()
    api_base = cfg.get_api_base() or "https://api.deepseek.com"
    model = cfg.agents.defaults.model or "deepseek-chat"
    if not api_key:
        raise SystemExit("No LLM API key in nanobot config")

    workspace = Path.home() / ".nanobot" / "workspace"
    store = open_knowledge_store(workspace, cfg.tools)
    if not store.list_documents():
        raise SystemExit("Knowledge store empty — run scripts/eval_acmetech_rag.py first")

    testset = json.loads(args.dataset.joinpath("AcmeTech_RAG_TestSet.json").read_text(encoding="utf-8"))
    questions = testset["questions"]
    if args.types:
        allow = set(args.types)
        questions = [q for q in questions if q.get("question_type") in allow]
    if args.limit:
        questions = questions[: args.limit]

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    print(
        f"LLM-as-judge: n={len(questions)} model={model} base={api_base} "
        f"k={args.k} concurrency={args.concurrency} vector={store.vector_enabled}"
    )

    t0 = time.time()
    tasks = [
        eval_one(sem, client, store, model, q, k=args.k) for q in questions
    ]
    rows: list[dict[str, Any]] = []
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        row = await coro
        rows.append(row)
        if i % 10 == 0 or i == len(tasks):
            print(f"  done {i}/{len(tasks)} ({row['id']} c={row['correctness']} f={row['faithfulness']})")

    rows.sort(key=lambda r: r["id"])
    by_type: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["question_type"] or "unknown"].append(r)

    for qtype, items in grouped.items():
        n = len(items) or 1
        by_type[qtype] = {
            "total": len(items),
            "correctness": round(sum(x["correctness"] for x in items) / n, 3),
            "faithfulness": round(sum(x["faithfulness"] for x in items) / n, 3),
        }
        if qtype == "negative":
            by_type[qtype]["abstention"] = round(sum(x["abstention"] for x in items) / n, 3)

    grounded = [r for r in rows if r["question_type"] != "negative"]
    gn = len(grounded) or 1
    summary = {
        "model": model,
        "k": args.k,
        "n": len(rows),
        "seconds": round(time.time() - t0, 1),
        "overall_grounded": {
            "total": len(grounded),
            "correctness": round(sum(r["correctness"] for r in grounded) / gn, 3),
            "faithfulness": round(sum(r["faithfulness"] for r in grounded) / gn, 3),
        },
        "by_type": by_type,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\agent\acmetech-enterprise-rag-dataset-main"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--types",
        nargs="*",
        default=[],
        help="Optional filter: single_hop multi_hop negative",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".nanobot" / "eval" / "acmetech_llm_judge.json",
    )
    args = parser.parse_args()

    payload = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== LLM-AS-JUDGE SUMMARY ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
