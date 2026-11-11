"""Evaluate enterprise SkillRouter against a routing JSONL (RouteBench-style).

Metrics:
  - P@1 / Recall@k / MRR for positive cases
  - Abstention accuracy for negative cases (no expect[])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanobot.agent.skills import SkillRouter, SkillRouterConfig, SkillsLoader
from nanobot.config.loader import load_config


def load_cases(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(json.loads(text))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path.home() / ".nanobot" / "eval" / "skill_router" / "acmetech_routing_eval.jsonl",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".nanobot" / "eval" / "skill_router" / "acmetech_routing_report.json",
    )
    parser.add_argument("--mode", choices=["hybrid", "lexical", "vector"], default="hybrid")
    args = parser.parse_args()

    cfg = load_config()
    workspace = Path.home() / ".nanobot" / "workspace"
    loader = SkillsLoader(workspace, disabled_skills=set(cfg.agents.defaults.disabled_skills))
    router_cfg = SkillRouterConfig(
        enable=True,
        top_k=args.k,
        search_mode=args.mode,
        two_stage=True,
        category_top_k=2,
    )
    router = SkillRouter(
        loader,
        router_cfg,
        knowledge_cfg=cfg.tools.knowledge,
    )

    cases = load_cases(args.eval)
    pos = [c for c in cases if c.get("expect")]
    neg = [c for c in cases if not c.get("expect")]

    p_at_1 = 0
    recall = 0
    mrr = 0.0
    rows = []
    for case in pos:
        hits = router.route(case["prompt"], top_k=args.k)
        names = [h.name for h in hits]
        expect = list(case["expect"])
        hit_set = set(names)
        ok_any = any(e in hit_set for e in expect)
        ok_all = all(e in hit_set for e in expect)
        rank = 0
        for i, name in enumerate(names, start=1):
            if name in expect:
                rank = i
                break
        if names and names[0] in expect:
            p_at_1 += 1
        if ok_any:
            recall += 1
        if rank:
            mrr += 1.0 / rank
        rows.append(
            {
                "id": case["id"],
                "expect": expect,
                "got": names,
                "p_at_1": bool(names and names[0] in expect),
                "recall_any": ok_any,
                "recall_all": ok_all,
                "mrr": (1.0 / rank) if rank else 0.0,
            }
        )

    abstain_ok = 0
    neg_rows = []
    for case in neg:
        hits = router.route(case["prompt"], top_k=args.k)
        # Abstention heuristic: either no hits, or top score below soft threshold.
        ok = (not hits) or (hits[0].score < 0.08)
        abstain_ok += int(ok)
        neg_rows.append({"id": case["id"], "got": [h.name for h in hits], "abstain": ok})

    npos = max(len(pos), 1)
    nneg = max(len(neg), 1)
    summary = {
        "mode": args.mode,
        "k": args.k,
        "skills_indexed": len(loader.catalog_records()),
        "positive": {
            "total": len(pos),
            "p_at_1": round(p_at_1 / npos, 3),
            "recall_any@k": round(recall / npos, 3),
            "mrr": round(mrr / npos, 3),
        },
        "negative": {
            "total": len(neg),
            "abstention_accuracy": round(abstain_ok / nneg, 3),
            "note": "heuristic: empty route or top score < 0.08",
        },
    }
    payload = {"summary": summary, "positive_rows": rows, "negative_rows": neg_rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
