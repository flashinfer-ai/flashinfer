#!/usr/bin/env python3
"""
FlashInfer documentation gap checker.
Detects MISSING (@flashinfer_api in code but absent from .rst) and
STALE (in .rst but no @flashinfer_api in code) entries.
"""

import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .inspect_sources import (
    collect_deprecated_symbols,
    collect_module_alias_exports,
    iter_decorated_functions,
)
from .configure_checks import DOCS_API_DIR, FLASHINFER_PKG, FLASHINFER_ROOT, OUTPUT_DIR


def _detect_version() -> str:
    """Resolve the version tag to embed in per-axis JSON / Markdown output.

    Resolution order:

    1. ``DOC_CHECK_VERSION`` env var — set by ``run_doc_check.sh`` so the
       per-axis filename matches the aggregate report filename
       (``flashinfer_<base>_<short_sha>_doc_check_result.{md,html}``).
    2. ``${FLASHINFER_SRC}/version.txt`` — the wheel version of the checked
       out source. Useful when invoking this script directly (no shell
       driver) — keeps artefacts aligned with the real checkout, instead
       of the previously hard-coded ``0.6.11``.
    3. ``unknown`` — last-resort fallback.
    """
    env_v = os.environ.get("DOC_CHECK_VERSION", "").strip()
    if env_v:
        return env_v
    vf = FLASHINFER_ROOT / "version.txt"
    if vf.is_file():
        try:
            v = vf.read_text(encoding="utf-8").strip()
            if v:
                return v
        except Exception:
            pass
    return "unknown"


VERSION = _detect_version()

# Modules whose MISSING entries should be skipped (top-level re-exports everything)
SKIP_MISSING_MODULES = {"flashinfer"}
# Modules whose STALE entries should be skipped
SKIP_STALE_MODULES = {"flashinfer.testing"}
# Per-module symbol ignore list
DOC_IGNORE = {
    "flashinfer.fused_moe": {"cute_dsl_fused_moe_nvfp4"},
}


# ---------------------------------------------------------------------------
# Step 1: collect all @flashinfer_api-decorated function names via AST
# ---------------------------------------------------------------------------
def collect_flashinfer_api_functions(pkg_root: Path) -> dict[str, set[str]]:
    """Returns ``{module_dotted_name: set(function_names)}``.

    Only **module-level** decorated functions are collected here — class
    methods like ``BatchAttention.plan`` are already picked up by ``..
    autoclass:: BatchAttention :members:`` in the .rst, and reporting them as
    MISSING against the module's own .rst entry would be a false positive
    (the previous ``ast.walk`` based implementation conflated method names
    like ``__init__/plan/run`` with module-level functions of the same name).

    On top of that, we resolve ``from .sub import x as y`` re-exports so a
    ``.rst`` line documenting ``flashinfer.comm.vllm_all_reduce`` matches the
    ``def all_reduce`` (decorated in ``flashinfer/comm/vllm_ar.py``) re-exported
    by ``flashinfer/comm/__init__.py`` as ``vllm_all_reduce``.
    """
    result: dict[str, set[str]] = {}
    for _py_file, mod, node in iter_decorated_functions(
        pkg_root, "flashinfer_api", scope="module"
    ):
        result.setdefault(mod, set()).add(node.name)
    for mod, names in collect_module_alias_exports(pkg_root, "flashinfer_api").items():
        result.setdefault(mod, set()).update(names)
    return result


# ---------------------------------------------------------------------------
# Step 2: collect documented symbols from docs/api/*.rst
# ---------------------------------------------------------------------------
_AUTOSUMMARY_RE = re.compile(r"^\s{0,8}(\w+)\s*$")
_CURRENTMOD_RE = re.compile(r"\.\.\s+currentmodule::\s+(\S+)")
_AUTOCLASS_RE = re.compile(r"\.\.\s+autoclass::\s+(\S+)")
_AUTOFUNCTION_RE = re.compile(r"\.\.\s+autofunction::\s+(\S+)")
_IN_AUTOSUMMARY = re.compile(r"\.\.\s+autosummary::")


def parse_rst_symbols(rst_file: Path) -> dict[str, set[str]]:
    """Returns {module_dotted_name: set(symbol_names)} parsed from one rst file."""
    lines = rst_file.read_text(encoding="utf-8", errors="replace").splitlines()
    modules: dict[str, set[str]] = {}
    current_module = ""
    in_autosummary = False
    in_toctree_options = False

    for line in lines:
        # .. currentmodule::
        m = _CURRENTMOD_RE.match(line)
        if m:
            current_module = m.group(1)
            in_autosummary = False
            continue

        # .. autosummary::
        if _IN_AUTOSUMMARY.match(line):
            in_autosummary = True
            in_toctree_options = True
            continue

        # .. autoclass:: ClassName  or  .. autofunction:: name
        m = _AUTOCLASS_RE.match(line) or _AUTOFUNCTION_RE.match(line)
        if m:
            sym = m.group(1).split(".")[-1]
            if current_module:
                modules.setdefault(current_module, set()).add(sym)
            continue

        if in_autosummary:
            stripped = line.strip()
            if not stripped:
                # blank line — could be between directive options and entries
                continue
            if stripped.startswith(":"):
                # directive option like :toctree: or :nosignatures:
                in_toctree_options = True
                continue
            if stripped.startswith(".."):
                in_autosummary = False
                continue
            # plain symbol entry
            if in_toctree_options and not stripped.startswith(":"):
                in_toctree_options = False
            if re.match(r"^[A-Za-z_]\w*$", stripped):
                if current_module:
                    modules.setdefault(current_module, set()).add(stripped)

    return modules


def collect_documented_symbols(docs_api: Path) -> dict[str, set[str]]:
    """Merge all rst files into {module: set(symbols)}."""
    all_docs: dict[str, set[str]] = {}
    for rst_file in sorted(docs_api.glob("*.rst")):
        for mod, syms in parse_rst_symbols(rst_file).items():
            all_docs.setdefault(mod, set()).update(syms)
    return all_docs


# ---------------------------------------------------------------------------
# Step 3: compare and report
# ---------------------------------------------------------------------------
def run_check(
    api_by_module: dict[str, set[str]],
    doc_symbols: dict[str, set[str]],
    deprecated_by_module: dict[str, set[str]] | None = None,
):
    all_api_funcs: set[str] = set()
    for funcs in api_by_module.values():
        all_api_funcs.update(funcs)

    all_deprecated_funcs: set[str] = set()
    if deprecated_by_module:
        for funcs in deprecated_by_module.values():
            all_deprecated_funcs.update(funcs)

    # Pool of every documented symbol across all .rst files (used to absolve
    # MISSING entries when the .rst uses a different ``.. currentmodule::``
    # than the importing module).
    all_doc_syms: set[str] = set()
    for syms in doc_symbols.values():
        all_doc_syms.update(syms)

    results = []

    for mod in sorted(set(api_by_module) | set(doc_symbols)):
        doc_syms = doc_symbols.get(mod, set())

        # Gather @flashinfer_api functions exported by this module.
        #
        # Previously this also pulled in every function from every submodule
        # (``candidate_mod.startswith(mod + ".")``) and surfaced the *original*
        # name. That conflicted with ``__init__.py`` aliases — for example
        # ``flashinfer.comm.vllm_ar.all_reduce`` is re-exported as
        # ``vllm_all_reduce``, so ``all_reduce`` is NOT actually accessible as
        # ``flashinfer.comm.all_reduce`` and reporting it as MISSING was a
        # false positive. The alias-export collector already adds the renamed
        # symbol under the importing module, so the exact-match lookup here
        # is sufficient.
        mod_api: set[str] = api_by_module.get(mod, set()).copy()

        ignore = DOC_IGNORE.get(mod, set())

        # Functions explicitly marked deprecated (``@deprecated`` decorator or
        # ``Deprecated:`` docstring section) are exempt from STALE because
        # their continued .rst presence is a deliberate "documented for
        # back-compat" decision — the .rst already explains the deprecation
        # and points readers at the replacement.
        deprecated = (
            deprecated_by_module.get(mod, set()) if deprecated_by_module else set()
        )

        missing: set[str] = set()
        if mod not in SKIP_MISSING_MODULES:
            # A symbol exposed by ``mod`` (e.g. re-exported via
            # ``__init__.py``) is still considered documented when *some*
            # .rst file lists it under any ``.. currentmodule::`` — that is
            # how PR-6 documents the CuTe-DSL kernels under their canonical
            # submodule path (``flashinfer.quantization.kernels.*``) instead
            # of the conditional re-export in the public ``quantization``
            # facade, avoiding a docs-build dependency on the optional
            # ``cutlass`` stack.
            missing = (mod_api - doc_syms - all_doc_syms) - ignore

        stale: set[str] = set()
        if mod not in SKIP_STALE_MODULES:
            # Stale: in rst but has no @flashinfer_api anywhere AND not
            # explicitly deprecated; skip PascalCase (classes).
            stale = {
                s
                for s in (doc_syms - all_api_funcs - all_deprecated_funcs - deprecated)
                if s and not s[0].isupper()
            } - ignore

        results.append(
            {
                "module": mod,
                "documented": sorted(doc_syms),
                "api_decorated": sorted(mod_api),
                "deprecated": sorted(deprecated),
                "missing": sorted(missing),
                "stale": sorted(stale),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(results):
    total_missing = sum(len(r["missing"]) for r in results)
    total_stale = sum(len(r["stale"]) for r in results)

    print(f"\n{'=' * 70}")
    print(f"FlashInfer {VERSION} Documentation Gap Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    print(f"  Total MISSING : {total_missing}")
    print(f"  Total STALE   : {total_stale}")
    print(f"{'=' * 70}\n")

    for r in results:
        if not r["missing"] and not r["stale"]:
            continue
        print(f"Module: {r['module']}")
        if r["missing"]:
            print(
                f"  MISSING ({len(r['missing'])}) — in code (@flashinfer_api) but absent from .rst:"
            )
            for s in r["missing"]:
                print(f"    + {s}")
        if r["stale"]:
            print(
                f"  STALE ({len(r['stale'])}) — in .rst but no @flashinfer_api in code:"
            )
            for s in r["stale"]:
                print(f"    - {s}")
        print()

    if total_missing == 0 and total_stale == 0:
        print("  All documented modules are in sync with code.")


def write_json(results, out_path: Path):
    total_missing = sum(len(r["missing"]) for r in results)
    total_stale = sum(len(r["stale"]) for r in results)
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(),
        "summary": {"total_missing": total_missing, "total_stale": total_stale},
        "modules": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"JSON report saved: {out_path}")


def write_markdown(results, out_path: Path):
    total_missing = sum(len(r["missing"]) for r in results)
    total_stale = sum(len(r["stale"]) for r in results)
    lines = [
        f"# FlashInfer {VERSION} Documentation Gap Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| MISSING (code has @flashinfer_api, absent from .rst) | {total_missing} |",
        f"| STALE (in .rst, no @flashinfer_api in code) | {total_stale} |",
        "",
    ]

    for r in results:
        if not r["missing"] and not r["stale"]:
            continue
        lines.append(f"## `{r['module']}`")
        if r["missing"]:
            lines.append(f"\n### MISSING ({len(r['missing'])})")
            lines.append(
                "Functions decorated with `@flashinfer_api` but absent from `.rst`:\n"
            )
            for s in r["missing"]:
                lines.append(f"- `{s}`")
        if r["stale"]:
            lines.append(f"\n### STALE ({len(r['stale'])})")
            lines.append("Entries in `.rst` without `@flashinfer_api` in code:\n")
            for s in r["stale"]:
                lines.append(f"- `{s}`")
        lines.append("")

    if total_missing == 0 and total_stale == 0:
        lines.append("All documented modules are in sync with code.")

    out_path.write_text("\n".join(lines))
    print(f"Markdown report saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Step 1: Scanning @flashinfer_api decorators...")
    api_by_module = collect_flashinfer_api_functions(FLASHINFER_PKG)
    total_api = sum(len(v) for v in api_by_module.values())
    print(
        f"  Found {total_api} @flashinfer_api functions across {len(api_by_module)} modules"
    )

    print("Step 2: Parsing docs/api/*.rst files...")
    doc_symbols = collect_documented_symbols(DOCS_API_DIR)
    total_doc = sum(len(v) for v in doc_symbols.values())
    print(f"  Found {total_doc} documented symbols across {len(doc_symbols)} modules")

    print("Step 3: Comparing...")
    deprecated_by_module = collect_deprecated_symbols(FLASHINFER_PKG)
    total_dep = sum(len(v) for v in deprecated_by_module.values())
    print(f"  Found {total_dep} deprecated symbols (will be exempted from STALE)")
    results = run_check(api_by_module, doc_symbols, deprecated_by_module)

    print_report(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(results, OUTPUT_DIR / f"api_rst_{VERSION}_{ts}.json")
    write_markdown(results, OUTPUT_DIR / f"api_rst_{VERSION}_{ts}.md")

    total_issues = sum(len(r["missing"]) + len(r["stale"]) for r in results)
    return 0 if total_issues == 0 else 1


def run_main() -> int:
    try:
        return main()
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(run_main())
