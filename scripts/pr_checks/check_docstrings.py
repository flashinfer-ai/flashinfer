#!/usr/bin/env python3
"""
FlashInfer documentation static checks.

Extends check_api_docs.py (the API <-> docs/api/*.rst check) with two
higher-ROI checks:

  docstring_completeness
      Every @flashinfer_api function has a non-empty docstring containing
      a "Parameters" (or "Args") section.

  args_consistency
      The Parameters section in the docstring lists exactly the same
      argument names as the function signature (ignoring self / *args /
      **kwargs).

Pure AST + text parsing, no imports of flashinfer, no GPU needed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .inspect_sources import func_arg_names, iter_decorated_functions
from .register_checks import (
    ARGS_CONSISTENCY,
    DOCSTRING_COMPLETENESS,
    Finding,
    get_check,
)
from .configure_checks import FLASHINFER_PKG, FLASHINFER_ROOT, OUTPUT_DIR

# Module exclusions.
SKIP_MODULES = {"flashinfer.testing"}

# Section ordering for this script's MD output (matches CHECKS section numbers).
CHECK_ORDER = (DOCSTRING_COMPLETENESS, ARGS_CONSISTENCY)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class FuncRecord:
    module: str
    name: str
    file: str
    line: int
    docstring: str | None = None
    sig_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Docstring parsing (NumPy / Sphinx-style)
# ---------------------------------------------------------------------------

_SECTION_TITLES = (
    "Parameters",
    "Args",
    "Arguments",
    "Returns",
    "Yields",
    "Raises",
    "Examples",
    "Example",
    "Notes",
    "Note",
    "Warning",
    "Warnings",
    "See Also",
    "References",
    "Attributes",
)
_SECTION_TITLE_GROUP = "|".join(re.escape(t) for t in _SECTION_TITLES)

# Match three section header dialects:
#   NumPy-style:   "Parameters\n----------"  (dashes or = underline)
#   Google-style:  "Parameters:"             (trailing colon, single line)
#   The Google-style spelling is what trtllm_ar.py / mhc.py / kda_decode.py
#   in the flashinfer repo actually use, and previously failed §2.2 because
#   the strict NumPy regex didn't match.
_SECTION_RE = re.compile(
    rf"^[ \t]*(?P<title>{_SECTION_TITLE_GROUP})\s*(?:"
    rf":\s*$"
    rf"|\n[ \t]*[\-=]{{3,}}\s*$"
    r")",
    re.MULTILINE,
)

# Sphinx field-list form: ":param x:", ":parameter x:", ":arg x:".
# When a docstring uses *only* this form (no header at all), we still want
# to treat it as having an Args section and extract names.
_SPHINX_PARAM_RE = re.compile(
    r"^[ \t]*:(?:param|parameter|arg|argument)\s+(?:[\w\[\],\s]+\s+)?(\w+)\s*:",
    re.MULTILINE,
)

# Delegation-style docstrings: "See :class:`X`", "See :meth:`Y`", etc.
# When the body only points the reader at another symbol, requiring a
# Parameters section is a false positive — e.g. a wrapper ``__init__`` whose
# class-level docstring already documents every constructor arg. We accept
# any inline reference role.
_DELEGATE_RE = re.compile(
    r"\bSee\s+:(?:class|meth|func|obj|attr|data|mod)?:`[^`]+`",
)


def split_sections(doc: str) -> dict[str, str]:
    """Split a NumPy- or Google-style docstring into sections by title.

    NumPy headers use a separator-underline line below the title; Google
    headers terminate with a single trailing colon. Both are recognised here
    so checks don't false-fire on the Google variant.
    """
    if not doc:
        return {}
    matches = list(_SECTION_RE.finditer(doc))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group("title")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        sections[title] = doc[start:end]
    return sections


def is_delegated_docstring(doc: str | None) -> bool:
    """True iff *doc* delegates to another documented symbol.

    A docstring like::

        See :class:`BatchAttention` for the meaning of each parameter.

    is the intended way to document a thin wrapper (e.g. a subclass
    ``__init__``). Sphinx renders the cross-ref and the reader follows it, so
    requiring a local ``Parameters`` section would be wrong.
    """
    if not doc:
        return False
    return _DELEGATE_RE.search(doc) is not None


def has_sphinx_param_fields(doc: str | None) -> bool:
    if not doc:
        return False
    return _SPHINX_PARAM_RE.search(doc) is not None


def sphinx_param_names(doc: str | None) -> list[str]:
    if not doc:
        return []
    return [m.group(1) for m in _SPHINX_PARAM_RE.finditer(doc)]


# NumPy-style ``name : type`` (space on both sides of the colon). The
# negative lookahead ``(?!\w+`?:)`` rejects narrative starts that mimic the
# pattern, e.g. ``The :meth:`X.y```. The acceptable type-token prefix
# covers every shape we see in flashinfer docstrings:
#   - identifier / quoted string / collection / generic literal
#   - leading backtick (``name: \`\`Constant\`\``` style)
#   - leading brace (NumPy enumeration ``backend : {"cuda", "cute-dsl"}``)
#
# The leading capture is a comma-separated identifier list — NumPy-style
# allows ``num_qo_heads, num_kv_heads, head_dim : int`` as a single header
# documenting three parameters at once. Previously only the first name was
# captured, so the remaining two were reported as "missing from docs". We
# now accept the joined header and split on comma after the match.
_PARAM_HEADER_WITH_TYPE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*:\s*(?!\w+`?:)[A-Za-z_\[\(\{\"'`]"
)
_PARAM_HEADER_NO_TYPE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*$"
)

# Google-style ``name: description`` — the colon is *flush* against the
# identifier (no space on the left), distinguishing it from the NumPy form.
# The description is then any non-empty content (lets us pick up
# ``name: \`\`Constant\`\`...``, ``name: see :class:\`X\``` etc.). Used at the
# section base-indent only — see ``doc_param_names`` below.
_PARAM_HEADER_GOOGLE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+\S")

# Google-style dash bullet for one parameter: "- name: description".
# Used by flashinfer/comm/trtllm_ar.py and friends.
_PARAM_HEADER_GOOGLE_BULLET_RE = re.compile(r"^-\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")


def doc_param_names(params_section: str) -> list[str]:
    """Extract parameter names from a Parameters / Args section.

    Recognises four on-line styles:

    * NumPy single-name : ``name : type``
    * NumPy multi-name  : ``a, b, c : type``  → expanded to ``[a, b, c]``
    * Google flush-colon: ``name: description``  (no space before ``:``)
    * Google dash bullet at any indent: ``- name: description``

    Heuristic to avoid false positives:
    - Compute the minimum non-blank indentation in the section (param-header indent)
    - Only lines at that minimum indent are candidates for header-style matches
    - A NumPy header requires the type token to start with a letter, ``[``,
      ``(``, quote, or backtick (rejects ``The :meth:`...``` narrative starts
      that mimic ``word:``).
    """
    raw_lines = params_section.splitlines()
    non_empty_indents = [len(l) - len(l.lstrip(" ")) for l in raw_lines if l.strip()]
    if not non_empty_indents:
        return []
    indent = min(non_empty_indents)

    out: list[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        stripped = line.lstrip(" ").rstrip()
        line_indent = len(line) - len(line.lstrip(" "))

        if line_indent == indent:
            m = (
                _PARAM_HEADER_WITH_TYPE_RE.match(stripped)
                or _PARAM_HEADER_NO_TYPE_RE.match(stripped)
                or _PARAM_HEADER_GOOGLE_RE.match(stripped)
            )
            if m:
                for name in m.group(1).split(","):
                    name = name.strip()
                    if name:
                        out.append(name)
                continue
        bullet = _PARAM_HEADER_GOOGLE_BULLET_RE.match(stripped)
        if bullet:
            out.append(bullet.group(1))
    return out


# ---------------------------------------------------------------------------
# Collect FuncRecords
# ---------------------------------------------------------------------------


def collect_records(pkg_root: Path) -> list[FuncRecord]:
    records: list[FuncRecord] = []
    for py_file, mod, node in iter_decorated_functions(
        pkg_root, "flashinfer_api", scope="all"
    ):
        if any(
            mod == skipped or mod.startswith(f"{skipped}.") for skipped in SKIP_MODULES
        ):
            continue
        records.append(
            FuncRecord(
                module=mod,
                name=node.name,
                file=str(py_file.relative_to(FLASHINFER_ROOT)),
                line=node.lineno,
                docstring=ast.get_docstring(node),
                sig_args=func_arg_names(node),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _finding(rec: FuncRecord, check: str, message: str) -> Finding:
    """Helper: build a per-symbol Finding from a FuncRecord (fills file/line/module/symbol)."""
    return Finding(
        check=check,
        message=message,
        module=rec.module,
        symbol=rec.name,
        file=rec.file,
        line=rec.line,
    )


def check_docstring_completeness(rec: FuncRecord) -> list[Finding]:
    out: list[Finding] = []
    if not rec.docstring or not rec.docstring.strip():
        out.append(_finding(rec, DOCSTRING_COMPLETENESS, "Missing docstring"))
        return out
    if is_delegated_docstring(rec.docstring):
        return out
    sections = split_sections(rec.docstring)
    has_params_section = bool(
        sections.get("Parameters") or sections.get("Args") or sections.get("Arguments")
    )
    if not has_params_section and has_sphinx_param_fields(rec.docstring):
        has_params_section = True
    if rec.sig_args and not has_params_section:
        out.append(
            _finding(
                rec,
                DOCSTRING_COMPLETENESS,
                "Missing 'Parameters' / 'Args' section in docstring",
            )
        )
    return out


def check_args_consistency(rec: FuncRecord) -> list[Finding]:
    if not rec.docstring:
        return []
    if is_delegated_docstring(rec.docstring):
        return []
    sections = split_sections(rec.docstring)
    params_text = (
        sections.get("Parameters") or sections.get("Args") or sections.get("Arguments")
    )
    if params_text is None and not has_sphinx_param_fields(rec.docstring):
        return []
    doc_args = doc_param_names(params_text) if params_text is not None else []
    doc_args += sphinx_param_names(rec.docstring)
    sig_args = rec.sig_args
    missing_in_doc = [a for a in sig_args if a not in doc_args]

    out: list[Finding] = []
    if missing_in_doc:
        out.append(
            _finding(
                rec,
                ARGS_CONSISTENCY,
                f"Args in signature but not documented: {missing_in_doc}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]) if len(argv) > 1 else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(FLASHINFER_PKG)

    findings: list[Finding] = []
    for rec in records:
        findings += check_docstring_completeness(rec)
        findings += check_args_consistency(rec)

    by_check: dict[str, int] = {}
    for f in findings:
        by_check[f.check] = by_check.get(f.check, 0) + 1

    print(f"Scanned {len(records)} @flashinfer_api functions")
    print()
    width = max(len(get_check(c).title) for c in CHECK_ORDER)
    for c in CHECK_ORDER:
        print(f"  {get_check(c).title:<{width}}  {by_check.get(c, 0):>4} fail")
    print()

    for c in CHECK_ORDER:
        sample = [f for f in findings if f.check == c][:5]
        if sample:
            print(f"=== {get_check(c).title} samples ===")
            for f in sample:
                print(f"  [fail] {f.module}.{f.symbol}  ({f.file}:{f.line})")
                print(f"          {f.message}")

    out_json = out_dir / "docstring_checks.json"
    out_json.write_text(
        json.dumps(
            {
                "summary": by_check,
                "total_records": len(records),
                "findings": [asdict(f) for f in findings],
            },
            indent=2,
        )
    )
    print(f"\nWrote: {out_json}")

    out_md = out_dir / "docstring_checks.md"
    lines = [
        "# FlashInfer Documentation Static Check",
        "",
        f"- Scanned: **{len(records)}** `@flashinfer_api` functions",
        f"- Source: `flashinfer/` package at `{FLASHINFER_ROOT}`",
        "",
        "## Summary",
        "",
        "| Check | Fail | Description |",
        "|---|---:|---|",
    ]
    for c in CHECK_ORDER:
        meta = get_check(c)
        lines.append(f"| {meta.title} | {by_check.get(c, 0)} | {meta.desc} |")

    mod_fail = Counter(f.module for f in findings)
    lines += [
        "",
        "## Top 10 modules by failure count",
        "",
        "| Module | Fails |",
        "|---|---:|",
    ]
    for mod, n in mod_fail.most_common(10):
        lines.append(f"| `{mod}` | {n} |")

    for c in CHECK_ORDER:
        grp = [f for f in findings if f.check == c]
        if not grp:
            continue
        meta = get_check(c)
        lines += ["", f"## {meta.title} — {meta.desc} ({len(grp)} findings)", ""]
        for f in grp:
            lines.append(
                f"- **FAIL** `{f.module}.{f.symbol}` ({f.file}:{f.line}) — {f.message}"
            )
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
