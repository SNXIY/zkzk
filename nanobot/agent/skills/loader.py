"""Skills loader for agent capabilities (progressive + hierarchical catalogs)."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import yaml

# Builtin product skills (flat). Enterprise catalogs live under workspace/skills/<category>/.
BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"

_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)


class SkillsLoader:
    """Loader for agent skills (``SKILL.md``), including nested category folders."""

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()
        self._path_index: dict[str, Path] = {}

    def _skill_entries_from_dir(
        self,
        base: Path,
        source: str,
        *,
        skip_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        # Nested enterprise layout: skills/<category>/<skill>/SKILL.md
        # Flat layout: skills/<skill>/SKILL.md
        for skill_file in sorted(base.rglob("SKILL.md")):
            if not skill_file.is_file():
                continue
            skill_dir = skill_file.parent
            try:
                rel = skill_dir.relative_to(base)
            except ValueError:
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            if name in seen:
                continue
            seen.add(name)
            category = rel.parts[0] if len(rel.parts) >= 2 else ""
            entry = {
                "name": name,
                "path": str(skill_file),
                "source": source,
                "category": category,
            }
            entries.append(entry)
            self._path_index[name] = skill_file
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        self._path_index = {}
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        workspace_names = {entry["name"] for entry in skills}
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(
                    self.builtin_skills, "builtin", skip_names=workspace_names
                )
            )

        if self.disabled_skills:
            skills = [s for s in skills if s["name"] not in self.disabled_skills]

        if filter_unavailable:
            return [
                skill
                for skill in skills
                if self._check_requirements(self._get_skill_meta(skill["name"]))
            ]
        return skills

    def list_categories(self) -> dict[str, list[str]]:
        """Return category -> skill names (empty-string key = uncategorized)."""
        grouped: dict[str, list[str]] = {}
        for entry in self.list_skills(filter_unavailable=False):
            cat = entry.get("category") or self.get_skill_category(entry["name"]) or ""
            grouped.setdefault(cat, []).append(entry["name"])
        return grouped

    def load_skill(self, name: str) -> str | None:
        if name in self._path_index and self._path_index[name].is_file():
            return self._path_index[name].read_text(encoding="utf-8")
        # Refresh index then retry; also support legacy flat lookup.
        self.list_skills(filter_unavailable=False)
        path = self._path_index.get(name)
        if path and path.is_file():
            return path.read_text(encoding="utf-8")
        roots = [self.workspace_skills]
        if self.builtin_skills:
            roots.append(self.builtin_skills)
        for root in roots:
            candidate = root / name / "SKILL.md"
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(
        self,
        exclude: set[str] | None = None,
        *,
        only: set[str] | None = None,
        group_by_category: bool = False,
    ) -> str:
        """Build progressive-loading skill directory (optionally filtered / grouped)."""
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        lines: list[str] = []
        if group_by_category:
            by_cat: dict[str, list[dict[str, str]]] = {}
            for entry in all_skills:
                name = entry["name"]
                if exclude and name in exclude:
                    continue
                if only is not None and name not in only:
                    continue
                cat = entry.get("category") or self.get_skill_category(name) or "general"
                by_cat.setdefault(cat, []).append(entry)
            for cat in sorted(by_cat):
                lines.append(f"### {cat}")
                for entry in by_cat[cat]:
                    lines.append(self._summary_line(entry))
            return "\n".join(lines)

        for entry in all_skills:
            name = entry["name"]
            if exclude and name in exclude:
                continue
            if only is not None and name not in only:
                continue
            lines.append(self._summary_line(entry))
        return "\n".join(lines)

    def build_category_directory(self) -> str:
        """Compact department/category map for two-stage routing prompts."""
        grouped = self.list_categories()
        if not grouped:
            return ""
        lines = ["### Skill categories"]
        for cat in sorted(grouped):
            label = cat or "general"
            names = ", ".join(sorted(grouped[cat]))
            lines.append(f"- **{label}**: {names}")
        return "\n".join(lines)

    def _summary_line(self, entry: dict[str, str]) -> str:
        skill_name = entry["name"]
        meta = self._get_skill_meta(skill_name)
        available = self._check_requirements(meta)
        desc = self._get_skill_description(skill_name)
        cat = entry.get("category") or self.get_skill_category(skill_name)
        prefix = f"[{cat}] " if cat else ""
        if available:
            return f"- **{skill_name}** — {prefix}{desc}  `{entry['path']}`"
        missing = self._get_missing_requirements(meta)
        suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
        return f"- **{skill_name}** — {prefix}{desc}{suffix}  `{entry['path']}`"

    def catalog_records(self) -> list[dict]:
        """Structured records for the skill router index."""
        records: list[dict] = []
        for entry in self.list_skills(filter_unavailable=False):
            name = entry["name"]
            fm = self.get_skill_metadata(name) or {}
            nano = self._parse_nanobot_metadata(fm.get("metadata"))
            category = (
                str(nano.get("category") or entry.get("category") or fm.get("category") or "")
            ).strip()
            department = str(nano.get("department") or category).strip()
            keywords = nano.get("keywords") or fm.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            desc = str(fm.get("description") or name)
            when = str(nano.get("when_to_use") or "")
            when_not = str(nano.get("when_not_to_use") or "")
            # Explicit metadata only — do not infer from free-text "kb_search" mentions.
            if "requires_kb" in nano:
                requires_kb = bool(nano.get("requires_kb"))
            elif "requiresKb" in nano:
                requires_kb = bool(nano.get("requiresKb"))
            elif "requires_kb" in fm:
                requires_kb = bool(fm.get("requires_kb"))
            else:
                requires_kb = bool(nano.get("enterprise"))
            index_text = " ".join(
                part
                for part in [
                    name.replace("-", " "),
                    desc,
                    category.replace("_", " "),
                    department.replace("_", " "),
                    " ".join(keywords),
                    when,
                    when_not,
                ]
                if part
            )
            records.append(
                {
                    "name": name,
                    "path": entry["path"],
                    "source": entry["source"],
                    "category": category,
                    "department": department,
                    "description": desc,
                    "keywords": keywords,
                    "when_to_use": when,
                    "when_not_to_use": when_not,
                    "index_text": index_text,
                    "always": bool(nano.get("always") or fm.get("always")),
                    "requires_kb": requires_kb,
                }
            )
        return records

    def get_skill_category(self, name: str) -> str:
        fm = self.get_skill_metadata(name) or {}
        nano = self._parse_nanobot_metadata(fm.get("metadata"))
        return str(nano.get("category") or fm.get("category") or "").strip()

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return ", ".join(
            [f"CLI: {command_name}" for command_name in required_bins if not shutil.which(command_name)]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        requires = self._get_skill_meta(name).get("requires", {})
        bins = [str(value) for value in requires.get("bins", [])]
        env = [str(value) for value in requires.get("env", [])]
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not shutil.which(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

    def _get_skill_description(self, name: str) -> str:
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return str(meta["description"])
        return name

    def _strip_frontmatter(self, content: str) -> str:
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end() :].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict:
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        payload = data.get("nanobot", data.get("openclaw", {}))
        return payload if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict:
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]

    def get_skill_metadata(self, name: str) -> dict | None:
        content = self.load_skill(name)
        if not content or not content.startswith("---"):
            return None
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return None
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        metadata: dict[str, object] = {}
        for key, value in parsed.items():
            metadata[str(key)] = value
        return metadata
