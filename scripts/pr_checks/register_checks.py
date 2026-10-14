"""Static-check identifiers, metadata, and intermediate finding model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    """A detailed finding produced inside one static checker."""

    check: str
    message: str
    location: str = ""
    module: str = ""
    symbol: str = ""
    file: str = ""
    line: int = 0


API_RST_MISSING = "api_rst_missing"
API_RST_STALE = "api_rst_stale"
DOCSTRING_COMPLETENESS = "docstring_completeness"
ARGS_CONSISTENCY = "args_consistency"
ENV_VARS_CONSISTENCY = "env_vars_consistency"
SUPPORTED_ARCH = "supported_arch_consistency"
QUICKREF_PATHS = "quickref_paths_exist"
SKILL_REFS = "skill_refs_exist"
DOCS_INDEX_REFS = "docs_index_refs_exist"


@dataclass(frozen=True)
class CheckMeta:
    """Human-readable metadata for one emitted check slug."""

    slug: str
    title: str
    desc: str


CHECKS: tuple[CheckMeta, ...] = (
    CheckMeta(
        API_RST_MISSING,
        "API ↔ RST Coverage (Missing)",
        "Public APIs decorated with @flashinfer_api but absent from docs/api/*.rst.",
    ),
    CheckMeta(
        API_RST_STALE,
        "API ↔ RST Coverage (Stale)",
        "Symbols listed in docs/api/*.rst but no longer public APIs.",
    ),
    CheckMeta(
        DOCSTRING_COMPLETENESS,
        "Docstring Completeness",
        "Public APIs need a docstring and a Parameters/Args section when applicable.",
    ),
    CheckMeta(
        ARGS_CONSISTENCY,
        "Args Consistency",
        "Function signature arguments must be documented.",
    ),
    CheckMeta(
        ENV_VARS_CONSISTENCY,
        "Env Vars Consistency",
        "Runtime FLASHINFER_* environment variables must be documented.",
    ),
    CheckMeta(
        SUPPORTED_ARCH,
        "Supported Architecture Consistency",
        "Supported-architecture claims must agree across sources.",
    ),
    CheckMeta(
        QUICKREF_PATHS,
        "CLAUDE.md Quick-Ref Paths",
        "Paths referenced in CLAUDE.md must exist.",
    ),
    CheckMeta(
        SKILL_REFS,
        "Skill References",
        "Paths referenced by .claude skills must exist.",
    ),
    CheckMeta(
        DOCS_INDEX_REFS,
        "Docs Index References",
        "docs/index.rst toctree entries must resolve.",
    ),
)


def get_check(slug: str) -> CheckMeta:
    """Look up check metadata by slug."""
    for check in CHECKS:
        if check.slug == slug:
            return check
    raise KeyError(f"Unknown check slug: {slug!r}")


__all__ = (
    "Finding",
    "CheckMeta",
    "CHECKS",
    "API_RST_MISSING",
    "API_RST_STALE",
    "DOCSTRING_COMPLETENESS",
    "ARGS_CONSISTENCY",
    "ENV_VARS_CONSISTENCY",
    "SUPPORTED_ARCH",
    "QUICKREF_PATHS",
    "SKILL_REFS",
    "DOCS_INDEX_REFS",
    "get_check",
)
