"""Skill router configuration (enterprise progressive skill loading)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nanobot.config_base import Base


class SkillRouterConfig(Base):
    """Route user turns to a Top-K skill subset before prompt injection.

    Enterprise pattern: large skill catalogs must not all enter context.
    Two-stage routing first picks departments/categories, then skills.
    """

    enable: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    category_top_k: int = Field(default=2, ge=1, le=10)
    inject_full_top_n: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Load full SKILL.md bodies for the top-N routed skills",
    )
    search_mode: Literal["hybrid", "lexical", "vector"] = "hybrid"
    two_stage: bool = True
    include_category_directory: bool = True
    include_unrouted_names: bool = False
    min_score: float = Field(default=0.01, ge=0.0)
    # When True, always-skills stay fully injected; router only filters the rest.
    preserve_always: bool = True
