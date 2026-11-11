"""Install / remove workspace skills from WebUI uploads."""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import yaml

from nanobot.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader, _STRIP_SKILL_FRONTMATTER

_SAFE_SKILL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_UNCOMPRESSED = 32 * 1024 * 1024
_MAX_FILES = 200


class SkillInstallError(ValueError):
    """Raised when a skill package cannot be installed."""


def install_workspace_skill(
    workspace: Path,
    *,
    filename: str,
    data: bytes,
) -> dict[str, Any]:
    """Install a skill zip / .skill / SKILL.md into ``<workspace>/skills``."""
    if not data:
        raise SkillInstallError("empty upload")
    if len(data) > _MAX_UNCOMPRESSED:
        raise SkillInstallError("upload too large")

    workspace = Path(workspace)
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    lower = (filename or "").lower()
    with tempfile.TemporaryDirectory(prefix="nanobot-skill-") as tmp:
        tmp_path = Path(tmp)
        extracted = tmp_path / "extracted"
        extracted.mkdir()

        if lower.endswith(".md") or (filename or "").upper() == "SKILL.MD":
            name = _name_from_skill_md(data)
            skill_dir = extracted / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_bytes(data)
        else:
            _extract_zip(data, extracted)
            skill_dir = _find_skill_dir(extracted)

        skill_dir = _validate_skill_dir(skill_dir)
        name = skill_dir.name
        if not _SAFE_SKILL.match(name):
            raise SkillInstallError(f"invalid skill name: {name}")

        dest = skills_root / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_dir, dest)

    loader = SkillsLoader(workspace)
    detail = {
        "name": name,
        "description": "",
        "source": "workspace",
        "available": True,
        "unavailable_reason": "",
    }
    meta = loader.get_skill_metadata(name) or {}
    desc = meta.get("description")
    if isinstance(desc, str):
        detail["description"] = desc.strip()
    available, reason = loader.get_skill_availability(name)
    detail["available"] = available
    detail["unavailable_reason"] = reason
    return {"skill": detail}


def delete_workspace_skill(workspace: Path, name: str) -> dict[str, Any]:
    """Delete a workspace skill. Builtin skills cannot be removed."""
    if not name or "/" in name or "\\" in name or not _SAFE_SKILL.match(name):
        raise SkillInstallError("invalid skill name")
    workspace = Path(workspace)
    loader = SkillsLoader(workspace)
    entries = loader.list_skills(filter_unavailable=False)
    entry = next((item for item in entries if item["name"] == name), None)
    if entry is None:
        raise LookupError("skill not found")
    if entry.get("source") != "workspace":
        raise SkillInstallError("builtin skills cannot be deleted")
    dest = workspace / "skills" / name
    if not dest.exists() or not dest.is_dir():
        raise LookupError("skill not found")
    # Extra safety: never delete under builtin tree
    if _is_under(dest, BUILTIN_SKILLS_DIR):
        raise SkillInstallError("refusing to delete builtin skill")
    shutil.rmtree(dest)
    return {"deleted": True, "name": name}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _extract_zip(data: bytes, dest: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > _MAX_FILES:
                raise SkillInstallError("too many files in skill archive")
            total = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise SkillInstallError("archive contains unsafe paths")
                total += max(info.file_size, 0)
                if total > _MAX_UNCOMPRESSED:
                    raise SkillInstallError("uncompressed skill archive too large")
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise SkillInstallError("invalid zip / .skill archive") from exc


def _find_skill_dir(root: Path) -> Path:
    direct = root / "SKILL.md"
    if direct.exists():
        # Archive contained SKILL.md at top level — wrap with folder name from frontmatter
        name = _name_from_skill_md(direct.read_bytes())
        wrapped = root / name
        if not wrapped.exists():
            wrapped.mkdir()
            for child in list(root.iterdir()):
                if child.name == name:
                    continue
                target = wrapped / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
        return wrapped

    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").exists()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer a single top-level skill folder when archive has nested junk
        top = [p for p in candidates if p.parent == root]
        if len(top) == 1:
            return top[0]
        raise SkillInstallError("archive contains multiple skills; upload one skill at a time")
    # Nested one level
    for child in root.iterdir():
        if child.is_dir():
            nested = [p for p in child.rglob("SKILL.md")]
            if len(nested) == 1:
                return nested[0].parent
    raise SkillInstallError("SKILL.md not found in upload")


def _validate_skill_dir(skill_dir: Path) -> Path:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise SkillInstallError("SKILL.md missing")
    content = skill_md.read_text(encoding="utf-8")
    meta = _parse_frontmatter(content)
    name = str(meta.get("name") or skill_dir.name).strip()
    desc = str(meta.get("description") or "").strip()
    if not name:
        raise SkillInstallError("skill frontmatter missing name")
    if not _SAFE_SKILL.match(name):
        raise SkillInstallError(f"invalid skill name in frontmatter: {name}")
    if not desc:
        raise SkillInstallError("skill frontmatter missing description")
    # Normalize directory name to frontmatter name
    if skill_dir.name != name:
        renamed = skill_dir.parent / name
        if renamed.exists() and renamed != skill_dir:
            raise SkillInstallError(f"skill name conflict: {name}")
        skill_dir.rename(renamed)
        return renamed
    return skill_dir


def _name_from_skill_md(data: bytes) -> str:
    content = data.decode("utf-8", errors="replace")
    meta = _parse_frontmatter(content)
    name = str(meta.get("name") or "").strip()
    if not name or not _SAFE_SKILL.match(name):
        raise SkillInstallError("SKILL.md must declare a valid name in frontmatter")
    return name


def _parse_frontmatter(content: str) -> dict[str, Any]:
    if not content.startswith("---"):
        raise SkillInstallError("SKILL.md must start with YAML frontmatter")
    match = _STRIP_SKILL_FRONTMATTER.match(content)
    if not match:
        raise SkillInstallError("invalid SKILL.md frontmatter")
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillInstallError("invalid SKILL.md YAML frontmatter") from exc
    if not isinstance(parsed, dict):
        raise SkillInstallError("skill frontmatter must be a mapping")
    return parsed
