#!/usr/bin/env python3
"""
FlashInfer cross-source consistency checks (B-series).

Each check compares two authoritative sources to detect drift between code
and documentation. Pure static analysis: grep / AST / filesystem, no GPU,
no imports of flashinfer.

Checks:
  env_vars_consistency        — CLAUDE.md FLASHINFER_* env var table vs
                                real os.environ/getenv reads in code.
  supported_arch_consistency  — CLAUDE.md "Supported GPU Architectures"
                                line vs `is_sm*_supported()` predicates and
                                `supported_major_versions=[...]` lists in
                                the JIT layer.
  quickref_paths_exist        — every file path mentioned in CLAUDE.md
                                code blocks / tables really exists in repo.
  skill_refs_exist            — file paths inside .claude/skills/*/SKILL.md
                                still exist in the codebase.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .register_checks import (
    DOCS_INDEX_REFS,
    ENV_VARS_CONSISTENCY,
    QUICKREF_PATHS,
    SKILL_REFS,
    SUPPORTED_ARCH,
    Finding,
    get_check,
)
from .configure_checks import (
    CLAUDE_MD,
    CSRC_DIR,
    DOCS_DIR,
    FLASHINFER_PKG,
    FLASHINFER_ROOT,
    OUTPUT_DIR,
    SKILLS_DIR,
)

# Section ordering for this script's MD output.
CHECK_ORDER = (
    ENV_VARS_CONSISTENCY,
    SUPPORTED_ARCH,
    QUICKREF_PATHS,
    SKILL_REFS,
    DOCS_INDEX_REFS,
)


# ---------------------------------------------------------------------------
# env_vars_consistency
# ---------------------------------------------------------------------------

# Known env vars that are macros / build constants, NOT runtime env vars.
# These appear with FLASHINFER_ prefix but aren't user-settable env vars.
_ENV_VAR_EXCLUSIONS = {
    "FLASHINFER_AVAILABLE",
    "FLASHINFER_CUDA_CALL",
    "FLASHINFER_CUDA_VERSION",
    "FLASHINFER_DATA",
    "FLASHINFER_DLL",
    "FLASHINFER_ENABLE_BF16",
    "FLASHINFER_ENABLE_F16",
    "FLASHINFER_ENABLE_F32",
    "FLASHINFER_ENABLE_FP4_E2M1",
    "FLASHINFER_ENABLE_FP8_E4M3",
    "FLASHINFER_ENABLE_FP8_E5M2",
    "FLASHINFER_ENABLE_FP8_E8M0",
    "FLASHINFER_ENABLE_PROFILER",
    "FLASHINFER_ERROR",
    "FLASHINFER_INCLUDE_DIR",
    "FLASHINFER_LOG_INFO",
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
    "FLASHINFER_B12X_FORCE_MOE_W4A16",
    "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
    "FLASHINFER_B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS",
    "FLASHINFER_CHECK",
    "FLASHINFER_MAMBA_ENABLE_SM100",
    "FLASHINFER_MAMBA_ENABLE_SM90",
    "FLASHINFER_WARN",
    # Module-level constants exported by jit_env (read from os.environ
    # via helpers, end-user usually sees the canonical name in docs):
    "FLASHINFER_BASE_DIR",
    "FLASHINFER_CACHE_DIR",
    "FLASHINFER_SRC_DIR",
    "FLASHINFER_WORKSPACE_DIR",
    # nvcc plumbing — derived from launcher var but not user-set directly:
    "FLASHINFER_EXTRA_CFLAGS",
    "FLASHINFER_EXTRA_CUDAFLAGS",
    "FLASHINFER_EXTRA_LDFLAGS",
}

# Match env var reads of all common shapes:
#   os.environ.get("X")            os.environ["X"]
#   os.getenv("X")                 "X" in os.environ
#   std::getenv("X")               getenv("X")
#   env_var_name = "X"             # name-then-use indirect read
_ENV_VAR_READ_RE = re.compile(
    r"""(?xs)
    (?:
        (?:os\.environ\.get|os\.getenv|std::getenv|getenv)
            \(\s*['"](FLASHINFER_[A-Z][A-Z_0-9]*)
      | os\.environ\[\s*['"](FLASHINFER_[A-Z][A-Z_0-9]*)
      | ['"](FLASHINFER_[A-Z][A-Z_0-9]*)['"]\s+in\s+os\.environ
    )
    """
)
_ENV_VAR_DOCREF_RE = re.compile(r"FLASHINFER_[A-Z][A-Z_0-9]*")


def collect_env_vars_in_code() -> set[str]:
    found: set[str] = set()
    for root in (FLASHINFER_PKG, CSRC_DIR):
        if not root.exists():
            continue
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if fp.suffix not in (".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"):
                continue
            try:
                text = fp.read_text("utf-8", "replace")
            except Exception:
                continue
            for m in _ENV_VAR_READ_RE.finditer(text):
                # Any of the alternation groups may have matched.
                name = m.group(1) or m.group(2) or m.group(3)
                if name:
                    found.add(name)
    return found - _ENV_VAR_EXCLUSIONS


def collect_env_vars_in_claude_md() -> set[str]:
    if not CLAUDE_MD.exists():
        return set()
    text = CLAUDE_MD.read_text("utf-8", "replace")
    return set(_ENV_VAR_DOCREF_RE.findall(text)) - _ENV_VAR_EXCLUSIONS


def check_env_vars_consistency() -> list[Finding]:
    code = collect_env_vars_in_code()
    doc = collect_env_vars_in_claude_md()
    out: list[Finding] = []
    missing_in_doc = sorted(code - doc)
    for v in missing_in_doc:
        out.append(
            Finding(
                check=ENV_VARS_CONSISTENCY,
                location="CLAUDE.md",
                message=f"env var read in code but not documented: {v}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# supported_arch_consistency
# ---------------------------------------------------------------------------

# Single canonical sentence in CLAUDE.md:
#   "FlashInfer supports NVIDIA SM75, SM80, SM86, SM89, SM90, SM103, SM110,
#    SM120, and SM121."
_CLAUDE_ARCH_LINE_RE = re.compile(
    r"FlashInfer supports NVIDIA ((?:SM\d+[a-z]?(?:, and | and |, )?)+)"
)
_SM_TOKEN_RE = re.compile(r"SM(\d+)[a-z]?")
_SUPPORTED_MAJOR_RE = re.compile(r"supported_major_versions\s*=\s*\[([^\]]*)\]")
_IS_SM_PREDICATE_RE = re.compile(r"def\s+(is_sm(\d+)([a-z])?_supported)\s*\(")


def collect_arches_doc() -> set[int]:
    if not CLAUDE_MD.exists():
        return set()
    text = CLAUDE_MD.read_text("utf-8", "replace")
    m = _CLAUDE_ARCH_LINE_RE.search(text)
    if not m:
        return set()
    fragment = m.group(1)
    return {int(x) for x in _SM_TOKEN_RE.findall(fragment)}


def collect_arches_code() -> tuple[set[int], set[int]]:
    """Returns (set of SM compute capabilities derived from is_sm* predicates,
    set of CUDA major versions from supported_major_versions lists).

    Predicates ending in `Nx_supported` (literal lowercase `x`) are family
    predicates (e.g. `is_sm12x_supported`) — treated as a major version, not
    a full compute capability.
    """
    sm_caps: set[int] = set()
    major_caps: set[int] = set()
    for fp in FLASHINFER_PKG.rglob("*.py"):
        try:
            text = fp.read_text("utf-8", "replace")
        except Exception:
            continue
        for m in _IS_SM_PREDICATE_RE.finditer(text):
            digits = int(m.group(2))
            suffix = m.group(3)
            if suffix == "x":
                # is_smNx_supported — N is a major (e.g. 12 -> 12x family)
                major_caps.add(digits)
            else:
                sm_caps.add(digits)
        for m in _SUPPORTED_MAJOR_RE.finditer(text):
            body = m.group(1).strip()
            if not body:
                continue
            for tok in body.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    major_caps.add(int(tok))
    return sm_caps, major_caps


def check_supported_arch() -> list[Finding]:
    doc_archs = collect_arches_doc()
    sm_caps, major_caps = collect_arches_code()
    out: list[Finding] = []
    if not doc_archs:
        out.append(
            Finding(
                check=SUPPORTED_ARCH,
                location="CLAUDE.md",
                message="Could not find 'FlashInfer supports NVIDIA ...' arch line",
            )
        )
        return out

    # Every SM cap that has an is_sm*_supported predicate should appear in doc.
    for cap in sorted(sm_caps):
        if cap not in doc_archs:
            out.append(
                Finding(
                    check=SUPPORTED_ARCH,
                    location=f"flashinfer/utils.py (is_sm{cap}*_supported)",
                    message=f"SM{cap} has is_sm{cap}*_supported() predicate but is "
                    f"not in CLAUDE.md supported-arch list",
                )
            )
    # Every major in supported_major_versions[] should be reflected by at least
    # one SM in the doc list (compute_cap.major == that major).
    doc_majors = {cap // 10 for cap in doc_archs}
    for mj in sorted(major_caps):
        if mj not in doc_majors:
            out.append(
                Finding(
                    check=SUPPORTED_ARCH,
                    location="flashinfer/jit/* (supported_major_versions)",
                    message=f"Major SM{mj}x referenced in supported_major_versions=[] "
                    f"but no SM{mj}? appears in CLAUDE.md arch list",
                )
            )
    return out


# ---------------------------------------------------------------------------
# quickref_paths_exist
# ---------------------------------------------------------------------------

# File-like tokens inside backticks: e.g. `flashinfer/aot.py`, `tests/foo`.
_FILELIKE_RE = re.compile(
    r"`([a-zA-Z_][a-zA-Z0-9_./-]*(?:/[a-zA-Z0-9_.][a-zA-Z0-9_./-]*)+)`"
)

# Path basenames that are obvious tutorial placeholders (e.g. "create
# flashinfer/new_op.py" demonstrates the PATTERN, not a real file).
_PLACEHOLDER_PATH_TOKENS = (
    "new_op",
    "some_kernel",
    "scale.cuh",
    "scale.cu",
    "scale_jit_binding.cu",
    "scale.py",
    "bench_scale.py",
    "test_scale.py",
    "my_script.py",
    "mylog.txt",
)

# Paths that are documented to be generated/created at build time, not present
# in a clean checkout. CLAUDE.md explicitly calls these out as build artifacts.
_BUILD_ARTIFACT_PATHS = {
    "flashinfer/_build_meta.py",  # generated from version.txt
    "flashinfer/data/cutlass",  # editable-install symlink
    "flashinfer/data/csrc",  # editable-install symlink
    "flashinfer/data/include",  # editable-install symlink
}


def _is_placeholder_path(path: str) -> bool:
    if any(tok in path for tok in _PLACEHOLDER_PATH_TOKENS):
        return True
    if path in _BUILD_ARTIFACT_PATHS:
        return True
    return False


def check_quickref_paths_exist() -> list[Finding]:
    if not CLAUDE_MD.exists():
        return []
    text = CLAUDE_MD.read_text("utf-8", "replace")
    out: list[Finding] = []
    seen: set[str] = set()
    for m in _FILELIKE_RE.finditer(text):
        path = m.group(1)
        # Skip URLs, glob-like patterns, env var demo strings
        if path.startswith(("http", "/")):
            continue
        if any(ch in path for ch in ("*", "?")):
            continue
        # Skip generic placeholder paths like "path/to/new_op.py"
        if path.startswith("path/to/") or path.startswith("path/"):
            continue
        if _is_placeholder_path(path):
            continue
        if path.endswith("/"):
            path = path[:-1]
        if path in seen:
            continue
        seen.add(path)
        target = FLASHINFER_ROOT / path
        if target.exists():
            continue
        # Also accept paths that resolve relative to package
        if (FLASHINFER_PKG.parent / path).exists():
            continue
        out.append(
            Finding(
                check=QUICKREF_PATHS,
                location="CLAUDE.md",
                message=f"Referenced path does not exist: `{path}`",
            )
        )
    return out


# ---------------------------------------------------------------------------
# skill_refs_exist
# ---------------------------------------------------------------------------

_SKILL_FILELIKE_RE = _FILELIKE_RE


def check_skill_refs_exist() -> list[Finding]:
    out: list[Finding] = []
    if not SKILLS_DIR.exists():
        return out

    for skill_md in sorted(SKILLS_DIR.rglob("*.md")):
        rel = skill_md.relative_to(FLASHINFER_ROOT)
        try:
            text = skill_md.read_text("utf-8", "replace")
        except Exception:
            continue

        for m in _SKILL_FILELIKE_RE.finditer(text):
            path = m.group(1)
            if path.startswith(("http", "/")):
                continue
            if any(ch in path for ch in ("*", "?")):
                continue
            if path.startswith("path/to/") or path.startswith("path/"):
                continue
            if _is_placeholder_path(path):
                continue
            target = FLASHINFER_ROOT / path
            if target.exists() or (FLASHINFER_PKG.parent / path).exists():
                continue
            out.append(
                Finding(
                    check=SKILL_REFS,
                    location=str(rel),
                    message=f"Referenced path does not exist: `{path}`",
                )
            )
    return out


# ---------------------------------------------------------------------------
# docs_index_refs_exist
# ---------------------------------------------------------------------------

_TOCTREE_HEADER_RE = re.compile(r"^\s*\.\.\s+toctree::\s*$", re.MULTILINE)


def _collect_sphinx_gallery_dirs() -> set[str]:
    """Return the set of ``gallery_dirs`` declared in ``docs/conf.py``.

    Sphinx-Gallery generates these directories at build time from
    ``examples_dirs`` ``.py`` files — they don't exist in source control.
    A static toctree-target check that walks the filesystem would (and
    previously did) flag every entry under them as missing, so we have to
    treat them as a build-time allowlist.

    ``docs/conf.py`` imports the installed flashinfer package, so we can't
    ``exec`` it from this tool. Instead, we ``ast.parse`` it and pluck the
    module-level ``sphinx_gallery_conf = {...}`` literal, accepting either a
    single string ``gallery_dirs`` value (Sphinx-Gallery default) or a list
    of strings.
    """
    conf = DOCS_DIR / "conf.py"
    if not conf.is_file():
        return set()
    try:
        tree = ast.parse(conf.read_text("utf-8", "replace"), filename=str(conf))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "sphinx_gallery_conf"
            for t in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values, strict=False):
            if not (isinstance(k, ast.Constant) and k.value == "gallery_dirs"):
                continue
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.add(v.value)
            elif isinstance(v, (ast.List, ast.Tuple)):
                for elt in v.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
    return out


def _parse_toctree_entries(rst_text: str) -> list[tuple[int, str]]:
    """Yield (line_no, entry_path) for every leaf entry under any toctree
    directive in the given .rst text.

    A toctree block is the directive line, optional ":option:" lines, then
    indented entries until a blank line followed by a less-indented line.
    """
    lines = rst_text.splitlines()
    entries: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if _TOCTREE_HEADER_RE.match(lines[i]):
            i += 1
            # Skip ":opt:" lines and blank lines until first content line
            while i < len(lines) and (
                not lines[i].strip() or lines[i].lstrip().startswith(":")
            ):
                i += 1
            # Collect indented non-empty lines
            base_indent: int | None = None
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    # blank line — end of toctree block in RST
                    break
                ind = len(line) - len(line.lstrip(" "))
                if base_indent is None:
                    base_indent = ind
                if ind < base_indent:
                    break
                entry = line.strip()
                # Skip lines that contain spaces (e.g. captions, options)
                # but accept slash-separated paths.
                if entry and not entry.startswith(":"):
                    entries.append((i + 1, entry))
                i += 1
        else:
            i += 1
    return entries


def check_docs_index_refs() -> list[Finding]:
    index = DOCS_DIR / "index.rst"
    if not index.exists():
        return []
    text = index.read_text("utf-8", "replace")
    gallery_dirs = _collect_sphinx_gallery_dirs()
    out: list[Finding] = []
    for line_no, entry in _parse_toctree_entries(text):
        # Entries can refer to .rst files relative to docs/, optionally without
        # the .rst suffix; or to nested files like "tutorials/recursive_attention"
        candidates = [
            DOCS_DIR / f"{entry}.rst",
            DOCS_DIR / entry / "index.rst",
            DOCS_DIR / entry,
        ]
        if any(p.exists() for p in candidates):
            continue
        # Exempt anything under a Sphinx-Gallery ``gallery_dirs`` root — the
        # generated ``index.rst`` and per-example pages don't exist in
        # source control; ``sphinx_gallery.gen_gallery`` materialises them
        # during the docs build.
        if any(
            entry == gd or entry.startswith(gd.rstrip("/") + "/") for gd in gallery_dirs
        ):
            continue
        out.append(
            Finding(
                check=DOCS_INDEX_REFS,
                location=f"docs/index.rst:{line_no}",
                message=f"toctree entry has no matching .rst file: `{entry}`",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]) if len(argv) > 1 else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    findings: list[Finding] = []
    findings += check_env_vars_consistency()
    findings += check_supported_arch()
    findings += check_quickref_paths_exist()
    findings += check_skill_refs_exist()
    findings += check_docs_index_refs()

    by_check: dict[str, int] = {}
    for f in findings:
        by_check[f.check] = by_check.get(f.check, 0) + 1

    width = max(len(get_check(c).title) for c in CHECK_ORDER)
    print("Cross-source consistency check\n")
    for c in CHECK_ORDER:
        print(f"  {get_check(c).title:<{width}}  {by_check.get(c, 0):>4} fail")
    print()

    for c in CHECK_ORDER:
        sample = [f for f in findings if f.check == c][:8]
        if sample:
            print(f"=== {get_check(c).title} samples ===")
            for f in sample:
                print(f"  [fail] {f.location}: {f.message}")

    out_json = out_dir / "flashinfer_cross_source_check.json"
    out_json.write_text(
        json.dumps(
            {
                "summary": by_check,
                "findings": [asdict(f) for f in findings],
            },
            indent=2,
        )
    )
    print(f"\nWrote: {out_json}")

    out_md = out_dir / "flashinfer_cross_source_check.md"
    lines = [
        "# FlashInfer Cross-Source Consistency Check",
        "",
        "Static comparison between code, CLAUDE.md, and `.claude/skills/` content.",
        "",
        "## Summary",
        "",
        "| Check | Fail | Description |",
        "|---|---:|---|",
    ]
    for c in CHECK_ORDER:
        meta = get_check(c)
        lines.append(f"| {meta.title} | {by_check.get(c, 0)} | {meta.desc} |")

    for c in CHECK_ORDER:
        grp = [f for f in findings if f.check == c]
        if not grp:
            continue
        lines += ["", f"## {get_check(c).title} ({len(grp)} findings)", ""]
        for f in grp:
            lines.append(f"- **FAIL** `{f.location}` — {f.message}")
    out_md.write_text("\n".join(lines))
    print(f"Wrote: {out_md}")

    return 0 if not findings else 1


def run_main(argv: list[str]) -> int:
    try:
        return main(argv)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(run_main(sys.argv))
