"""Generate enterprise-grade AcmeTech skill catalog + routing eval set.

Writes:
  <workspace>/skills/<category>/<skill>/SKILL.md
  <workspace>/../eval/skill_router/acmetech_routing_eval.jsonl
  <workspace>/../eval/skill_router/acmetech_categories.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")

CATEGORY_MAP = {
    "HR": "hr",
    "Engineering": "engineering",
    "Security": "security",
    "Product": "product",
    "Finance": "finance",
    "Customer_Support": "customer_support",
    "Legal": "legal",
}

CATEGORY_WHEN = {
    "hr": "employee lifecycle, leave, benefits, handbook, probation, performance",
    "engineering": "architecture, APIs, coding standards, deployment, engineering handbook",
    "security": "security policy, access control, incident response, disaster recovery",
    "product": "product overview, user guide, FAQ, release notes",
    "finance": "expense, procurement, travel, purchase approval",
    "customer_support": "support SOP, SLA, escalation tiers, ticket severity",
    "legal": "NDA, privacy, compliance, SOC2, legal review",
}


def slugify(name: str) -> str:
    base = Path(name).stem.lower()
    return _SLUG_RE.sub("-", base).strip("-")


def skill_markdown(doc: dict, category: str) -> str:
    skill_name = slugify(doc["filename"])
    title = doc.get("title") or skill_name
    headings = doc.get("section_headings") or []
    heading_blob = "; ".join(headings[:12])
    keywords = [skill_name.replace("-", " "), category.replace("_", " "), doc.get("doc_id", "")]
    keywords += [h.split(". ", 1)[-1] for h in headings[:8]]
    keywords = [k for k in keywords if k]

    when = (
        f"Use when the user asks about {title} topics"
        f" ({CATEGORY_WHEN.get(category, category)})."
    )
    when_not = (
        "Do not use for unrelated departments; pick the matching category skill instead. "
        "Do not invent policy numbers — retrieve with kb_search."
    )
    metadata = {
        "nanobot": {
            "category": category,
            "department": doc.get("category") or category,
            "doc_id": doc.get("doc_id"),
            "version": doc.get("version"),
            "source_document": doc.get("path") or doc.get("filename"),
            "keywords": keywords[:24],
            "when_to_use": when,
            "when_not_to_use": when_not,
            "requires_kb": True,
            "enterprise": True,
        }
    }
    desc = (
        f"AcmeTech enterprise playbook for {title} ({doc.get('doc_id', '')}). "
        f"Covers: {heading_blob}. Retrieve authoritative text via kb_search."
    )
    body = f"""# {title}

Enterprise skill for **{doc.get('category', category)}** · `{doc.get('doc_id', '')}` · {doc.get('version', '')}

## When to use
{when}

## When NOT to use
{when_not}

## Procedure
1. Call `kb_search` first. Keep the tool `query` close to the user's original wording; the runtime also anchors on the user turn and fuses results. Only rewrite after a weak/empty hit.
2. Prefer snippets from `{Path(doc.get('filename', '')).name}` and cite as `filename#chunkN`.
3. Answer with concrete facts (numbers, owners, timelines). Cite section headings when possible.
4. If retrieval is empty, low-confidence, or conflicting, say the knowledge base does not cover it — do not invent policy.

## Key sections
{chr(10).join(f'- {h}' for h in headings)}

## Related
See sibling skills in category `{category}` and use `kb_search` for cross-document (multi-hop) questions.
"""
    meta_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return (
        "---\n"
        f"name: {skill_name}\n"
        f"description: {json.dumps(desc, ensure_ascii=False)}\n"
        f"metadata: {meta_json}\n"
        "---\n\n"
        f"{body}"
    )


def map_testset_to_routing_cases(questions: list[dict], docs: list[dict]) -> list[dict]:
    by_file = {Path(d["filename"]).name.lower(): slugify(d["filename"]) for d in docs}
    cases: list[dict] = []
    for q in questions:
        qtype = q.get("question_type")
        raw = q.get("source_document") or ""
        if qtype == "negative":
            cases.append(
                {
                    "id": q["id"],
                    "prompt": q["question"],
                    "expect": [],
                    "reject": [],
                    "tags": ["negative", "abstention"],
                    "category": "general",
                }
            )
            continue
        expect: list[str] = []
        for part in re.split(r"\s*\+\s*", str(raw)):
            name = Path(part.replace("\\", "/")).name.lower()
            skill = by_file.get(name)
            if skill:
                expect.append(skill)
        # de-dupe preserve order
        seen = set()
        expect = [x for x in expect if not (x in seen or seen.add(x))]
        cat = CATEGORY_MAP.get(q.get("category", ""), "")
        cases.append(
            {
                "id": q["id"],
                "prompt": q["question"],
                "expect": expect,
                "reject": [],
                "tags": [qtype or "single_hop", q.get("difficulty") or ""],
                "category": cat,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\agent\acmetech-enterprise-rag-dataset-main"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / ".nanobot" / "workspace",
    )
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "AcmeTech_Manifest.json").read_text(encoding="utf-8"))
    testset = json.loads((args.dataset / "AcmeTech_RAG_TestSet.json").read_text(encoding="utf-8"))
    docs = manifest["documents"]

    skills_root = args.workspace / "skills"
    written = []
    for doc in docs:
        dept = doc.get("category") or "General"
        category = CATEGORY_MAP.get(dept, slugify(dept))
        skill_name = slugify(doc["filename"])
        dest_dir = skills_root / category / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        content = skill_markdown(doc, category)
        (dest_dir / "SKILL.md").write_text(content, encoding="utf-8")
        written.append({"name": skill_name, "category": category, "path": str(dest_dir / "SKILL.md")})

    eval_dir = Path.home() / ".nanobot" / "eval" / "skill_router"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cases = map_testset_to_routing_cases(testset["questions"], docs)
    eval_path = eval_dir / "acmetech_routing_eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    cats = {
        "company": manifest.get("company"),
        "categories": sorted({CATEGORY_MAP.get(d["category"], slugify(d["category"])) for d in docs}),
        "skills": written,
        "num_skills": len(written),
        "num_routing_cases": len(cases),
    }
    (eval_dir / "acmetech_categories.json").write_text(
        json.dumps(cats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"skills": len(written), "routing_cases": len(cases), "eval": str(eval_path)}, indent=2))


if __name__ == "__main__":
    main()
