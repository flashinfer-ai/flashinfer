"""Shared AST helpers for the flashinfer_document_check tooling.

The two checker scripts that walk the flashinfer source tree
(check_api_docs.py and check_docstrings.py) used to each carry their
own near-duplicate copy of these helpers — the check_docstrings version
was strictly more correct (handles arbitrarily nested attribute chains in
decorator calls), so we lift that one into shared code and have both
scripts import it.

Behavioral notes:
- ``decorator_call_name`` returns the **full dotted attribute chain** of a
  decorator expression (Name/Attribute/Call). For ``@a.b.c()`` it returns
  ``"a.b.c"``. Match on suffix (``endswith("flashinfer_api")``) or use the
  ``is_decorated_with`` convenience wrapper.
- ``iter_decorated_functions`` is a generator yielding
  ``(py_file, module_dotted_name, FunctionDef)`` tuples — the canonical way
  to scan all ``@flashinfer_api`` functions in the flashinfer package.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator


def decorator_call_name(dec: ast.expr) -> str:
    """Return the dotted attr name of a decorator.

    Handles ``ast.Name`` (``@foo``), ``ast.Attribute`` (``@a.b.c``), and
    ``ast.Call`` wrappers around either of those (``@foo()``, ``@a.b.c()``).
    Returns ``""`` for anything else (lambdas, subscripts, etc.).
    """
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        chain: list[str] = []
        cur: ast.expr = target
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.append(cur.id)
        return ".".join(reversed(chain))
    return ""


def is_decorated_with(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    decorator: str,
) -> bool:
    """True iff *node* carries a decorator whose dotted name ends with *decorator*.

    Matching on suffix lets ``@flashinfer_api`` and
    ``@flashinfer.utils.flashinfer_api`` both register positive.
    """
    for dec in node.decorator_list:
        name = decorator_call_name(dec)
        if name == decorator or name.endswith("." + decorator):
            return True
    return False


def func_arg_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return positional + keyword-only arg names, excluding self/cls."""
    args: list[str] = []
    a = fn.args
    for x in a.posonlyargs + a.args + a.kwonlyargs:
        if x.arg in ("self", "cls"):
            continue
        args.append(x.arg)
    return args


def module_name(py_file: Path, pkg_root: Path) -> str:
    """Convert a .py file path to a dotted module name relative to pkg_root's parent.

    e.g. for ``pkg_root=/.../flashinfer/flashinfer`` and
    ``py_file=/.../flashinfer/flashinfer/fused_moe/__init__.py`` returns
    ``"flashinfer.fused_moe"``.
    """
    rel = py_file.relative_to(pkg_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# Sphinx ``.. deprecated::`` directive or Google/NumPy ``Deprecated`` section.
# Either header form (with or without underline) is enough to flag the symbol
# as intentionally deprecated, alongside the AST-level ``@deprecated`` check.
_DEPRECATED_DOCSTRING_RE = re.compile(
    r"(?:"
    r"\.\.\s+deprecated::"  # Sphinx directive
    r"|Deprecated\s*:\s*$"  # Google-style "Deprecated:"
    r"|Deprecated\s*\n\s*[-=]{3,}"  # NumPy-style "Deprecated\n----"
    r"|Deprecated\b"  # leading-line "Deprecated pointer-based ..."
    r")",
    re.MULTILINE,
)


def is_function_deprecated(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff *fn* is explicitly marked deprecated.

    Matches either an ``@deprecated(...)``-flavoured decorator (suffix match,
    so ``typing.deprecated`` / ``warnings.deprecated`` / project-local
    ``deprecated`` helpers all count) or a docstring that opens with a
    Sphinx ``.. deprecated::`` directive, a Google ``Deprecated:`` section,
    or a NumPy ``Deprecated\\n----`` block.

    Used by the API↔.rst checker to exempt deprecated-but-still-documented
    helpers from the STALE bucket: the .rst entry is a deliberate
    "documented for back-compat" decision, not an out-of-sync stub.
    """
    for dec in fn.decorator_list:
        name = decorator_call_name(dec)
        if name == "deprecated" or name.endswith(".deprecated"):
            return True
    doc = ast.get_docstring(fn)
    if doc and _DEPRECATED_DOCSTRING_RE.match(doc.lstrip()):
        return True
    return False


def collect_deprecated_symbols(
    pkg_root: Path,
) -> dict[str, set[str]]:
    """Return ``{module_dotted_name: set(symbol_name, ...)}`` for every
    top-level function across *pkg_root* whose AST trips
    :func:`is_function_deprecated`.

    Mirrors the shape of :func:`collect_module_alias_exports` so callers can
    union the two and treat the result as "documented-but-not-currently-fresh
    symbols we accept as documented".
    """
    out: dict[str, set[str]] = {}
    for py_file in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(
                py_file.read_text("utf-8", "replace"), filename=str(py_file)
            )
        except SyntaxError:
            continue
        mod = module_name(py_file, pkg_root)
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and is_function_deprecated(node):
                out.setdefault(mod, set()).add(node.name)
    return out


def iter_decorated_functions(
    pkg_root: Path,
    decorator: str = "flashinfer_api",
    scope: str = "module",
) -> Iterator[tuple[Path, str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield (py_file, module_dotted_name, fn_node) for every function in
    *pkg_root* decorated with *decorator* (suffix match).

    The *scope* arg controls which functions are reported:

    - ``"module"`` (default) — only module-level (top-level) functions. Used by
      the API↔.rst checker: ``.. autoclass:: BatchAttention :members:`` already
      pulls in every decorated method, so reporting class methods as MISSING
      against the module-level .rst would be a false positive.
    - ``"all"`` — every decorated FunctionDef anywhere in the AST, including
      class methods. Used by the docstring / args checks which need to look at
      method docstrings too. Class methods are reported under a synthetic
      module path ``mod.ClassName`` so they don't collide with module-level
      functions sharing the same name.

    Skips files with syntax errors silently — matches existing checker behavior.
    """
    if scope not in ("module", "all"):
        raise ValueError(f"unknown scope: {scope!r}")

    for py_file in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(
                py_file.read_text("utf-8", "replace"), filename=str(py_file)
            )
        except SyntaxError:
            continue
        mod = module_name(py_file, pkg_root)

        if scope == "module":
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if is_decorated_with(node, decorator):
                    yield py_file, mod, node
        else:
            for top in tree.body:
                if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_decorated_with(top, decorator):
                        yield py_file, mod, top
                elif isinstance(top, ast.ClassDef):
                    cls_mod = f"{mod}.{top.name}"
                    for sub in top.body:
                        if isinstance(
                            sub, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and is_decorated_with(sub, decorator):
                            yield py_file, cls_mod, sub


def collect_module_alias_exports(
    pkg_root: Path,
    decorator: str = "flashinfer_api",
) -> dict[str, set[str]]:
    """Resolve ``from .submod import name as alias`` re-exports.

    A `.rst` file may document ``flashinfer.comm.vllm_all_reduce`` whose
    actual definition lives in ``flashinfer.comm.vllm_ar`` as ``def all_reduce``
    (decorated with ``@flashinfer_api``). We follow ``ast.ImportFrom`` chains
    inside any ``__init__.py`` (and other modules) to surface the *aliased*
    name under the *importing* module's dotted path.

    Returns ``{importing_module_dotted: set(alias_name, ...)}`` containing
    aliases that ultimately resolve to a ``@decorator``-decorated top-level
    function. Re-export chains are resolved to a fixed point, so a symbol can
    pass through multiple package facades before being surfaced. The result
    has the same shape as :func:`collect_flashinfer_api_functions`.
    """
    available: dict[str, set[str]] = {}
    for py_file in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(
                py_file.read_text("utf-8", "replace"), filename=str(py_file)
            )
        except SyntaxError:
            continue
        mod = module_name(py_file, pkg_root)
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and is_decorated_with(node, decorator):
                available.setdefault(mod, set()).add(node.name)

    pkg_dotted = pkg_root.name
    raw_imports: list[tuple[str, str, list[ast.alias]]] = []
    for py_file in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(
                py_file.read_text("utf-8", "replace"), filename=str(py_file)
            )
        except SyntaxError:
            continue
        importer_mod = module_name(py_file, pkg_root)
        importer_parts = importer_mod.split(".")
        is_package = py_file.name == "__init__.py"
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            level = node.level or 0
            target_mod: str
            if level > 0:
                base_length = len(importer_parts) - level + (1 if is_package else 0)
                base_parts = importer_parts[: max(0, base_length)]
                if not base_parts or base_parts[0] != pkg_dotted:
                    continue
                tail = [node.module] if node.module else []
                target_mod = (
                    ".".join(base_parts + tail) if tail else ".".join(base_parts)
                )
            else:
                if not node.module or not node.module.startswith(pkg_dotted):
                    continue
                target_mod = node.module
            raw_imports.append((importer_mod, target_mod, node.names))

    aliases: dict[str, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for importer_mod, target_mod, names in raw_imports:
            for alias in names:
                if alias.name == "*" or alias.name not in available.get(
                    target_mod, set()
                ):
                    continue
                surfaced = alias.asname or alias.name
                target_names = available.setdefault(importer_mod, set())
                if surfaced in target_names:
                    continue
                target_names.add(surfaced)
                aliases.setdefault(importer_mod, set()).add(surfaced)
                changed = True
    return aliases


__all__ = (
    "decorator_call_name",
    "is_decorated_with",
    "is_function_deprecated",
    "func_arg_names",
    "module_name",
    "iter_decorated_functions",
    "collect_module_alias_exports",
    "collect_deprecated_symbols",
)
