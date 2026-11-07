"""Decide whether a pull request can change the AOT jit-cache provider wheels.

The ``AOT Build Import`` lanes of ``pr-test.yml`` build every jit-cache provider
wheel (45-70 minutes each). A pull request that cannot change what those builds
compile or import does not need them. A false skip (AOT broken, lanes skipped)
is worse than an unneeded run, so this is an allow-list of what is safe to skip.
A changed file is **ignorable** only if it

* matches ``SKIP_CI_PATTERNS`` (``pr-test.yml``'s workflow ``env``: the files
  that skip the whole run when nothing else changed), so a PR that skips all
  of CI always skips AOT too, from one shared list;
* matches the AOT-only patterns below (tests, other workflows, test tooling):
  files that still need the GPU lanes but not the provider builds; or
* is a ``flashinfer/**/*.py`` file that ``flashinfer/aot.py`` cannot reach.

Anything else, including a path nobody classified, runs AOT.

Reachability of ``flashinfer/**/*.py`` is computed on the checked-out tree, not
kept as a hand-written list: the import closure of ``flashinfer/aot.py``
(``import`` and ``from ... import`` statements at any nesting depth, so lazy
imports inside functions count, and ``from pkg import name`` also follows
``pkg.name``). The root ``flashinfer/__init__.py`` is always treated as relevant
rather than traced, because the build imports it first. A file that calls
``importlib.import_module`` / ``__import__`` pulls in its whole subtree when it
is a package ``__init__.py`` (a lazy export loader, whose targets are not
visible statically). Dynamic imports in other modules are not expanded: the
ones in the closure resolve runtime kernels, tuning configs and log targets,
not AOT specs, and expanding them pulls in the whole package. A deleted or
renamed ``.py`` always counts: its importers' closure no longer exists on the
tree being read.

Usage::

    SKIP_CI_PATTERNS=... aot_scope.py --base <sha> --head <sha> [--root .]

``SKIP_CI_PATTERNS`` is an extended regular expression searched (unanchored,
as ``grep -E`` does in ``pr-test.yml``) in each path. Unset, only the AOT-only
patterns apply, which can only make the gate run AOT more often.

Prints ``run=true|false`` and a reason on separate lines, and exits 0. Any
error (bad revisions, unreadable file) prints ``run=true``: a gate that cannot
decide must not skip.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

ENTRY = "flashinfer/aot.py"
ROOT_INIT = "flashinfer/__init__.py"

# AOT-only additions to SKIP_CI_PATTERNS: paths that still need the GPU lanes
# but cannot influence the provider wheels or the checks run on them. Docs,
# Markdown, licenses, benchmarks and docker/ come from SKIP_CI_PATTERNS (the AOT
# lanes use the image ci/docker-tags.yml names, so docker/ alone cannot change
# them; a tag bump changes ci/ and runs AOT).
_AOT_ONLY_IGNORABLE = re.compile(
    r"""^(
        tests/
      | examples/
      | profiler/
      | \.claude/
      | \.devcontainer/
      | scripts/pr_checks/
      | scripts/test_sharding/
      | scripts/tests/
      | scripts/task_jit_run_tests_[^/]+
      | \.github/(?!workflows/pr-test\.yml$|actions/)
      | \.pre-commit-config\.yaml$
      | \.gitignore$
      | \.gitattributes$
      | LICENSE[^/]*$
      | pytest\.ini$
    )""",
    re.VERBOSE,
)


def _module_file(root: Path, dotted: str) -> Optional[str]:
    base = root.joinpath(*dotted.split("."))
    if (base / "__init__.py").is_file():
        return (base / "__init__.py").relative_to(root).as_posix()
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py").relative_to(root).as_posix()
    return None


def _imports(root: Path, rel: str) -> tuple[set[str], bool]:
    """Return (flashinfer files this file imports, whether it imports dynamically)."""
    tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
    package = list(PurePosixPath(rel).parent.parts)
    found: set[str] = set()
    dynamic = False
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            called = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            dynamic = dynamic or called in {"import_module", "__import__"}
        for name in names:
            if name == "flashinfer" or name.startswith("flashinfer."):
                target = _module_file(root, name)
                # The root package initializer is relevant unconditionally.
                if target and target != ROOT_INIT:
                    found.add(target)
    return found, dynamic


def aot_closure(root: Path) -> set[str]:
    """Every ``flashinfer/**/*.py`` file the AOT entry point can execute."""
    seen: set[str] = set()
    pending = [ENTRY]
    while pending:
        rel = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        imported, dynamic = _imports(root, rel)
        pending.extend(imported - seen)
        if dynamic and PurePosixPath(rel).name == "__init__.py":
            here = root / PurePosixPath(rel).parent
            pending.extend(p.relative_to(root).as_posix() for p in here.glob("**/*.py"))
    return seen


def decide(
    changed: Iterable[tuple[str, str]],
    root: Path,
    skip_ci_patterns: str = "",
) -> tuple[bool, str]:
    """Return (run_aot, reason) for ``(status, path)`` entries of a diff."""
    shared = re.compile(skip_ci_patterns) if skip_ci_patterns else None
    closure: Optional[set[str]] = None
    for status, path in changed:
        if (shared and shared.search(path)) or _AOT_ONLY_IGNORABLE.match(path):
            continue
        if path.startswith("flashinfer/") and path.endswith(".py"):
            if path in (ENTRY, ROOT_INIT) or path.startswith("flashinfer/jit/"):
                return True, f"{path}: AOT registration or JIT spec code"
            if not status.startswith(("M", "A")):
                return True, f"{path}: deleted or renamed Python module"
            if closure is None:
                closure = aot_closure(root)
            if path in closure:
                return True, f"{path}: imported by {ENTRY}"
            continue
        return True, f"{path}: can affect the provider wheels (or is unclassified)"
    return False, "no changed file can affect the provider wheels"


def _changed_files(base: str, head: str, root: Path) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode()
    parts = [p for p in out.split("\0") if p]
    return list(zip(parts[0::2], parts[1::2], strict=True))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args(argv)
    try:
        changed = _changed_files(args.base, args.head, args.root)
        run, reason = decide(changed, args.root, os.environ.get("SKIP_CI_PATTERNS", ""))
    except Exception as exc:  # noqa: BLE001 - an undecidable gate must run AOT
        run, reason = True, f"could not decide ({type(exc).__name__}: {exc})"
    print(f"run={'true' if run else 'false'}")
    print(f"reason={reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
