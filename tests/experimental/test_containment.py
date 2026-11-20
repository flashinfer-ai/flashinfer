"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Containment tests for the experimental packages.

``flashinfer/experimental/README.md`` is the normative policy for everything
under ``flashinfer/experimental/``; its "Containment rules" and "Relaxations
vs. the stable-API checklist" sections say how core -- everything outside that
directory -- may reach the experimental packages.  Three of those rules are
checked here, one test each:

1. ``flashinfer/aot.py`` registers no experimental module.  Experimental
   features are JIT-only and never ship in the pre-built packages, so the file
   imports nothing from ``flashinfer.experimental`` and every ``gen_*``
   generator that ``gen_all_modules`` calls comes from outside that tree.
2. Core imports ``flashinfer.experimental`` only on demand, through deferred
   (function-local) imports.  The one sanctioned module-level import is the
   lightweight ``*.support`` module that defines an ``@experimental_backend``
   checker; imports under ``if TYPE_CHECKING:`` never run and are not counted.
   Every other module-level import is a runtime violation unless ``ALLOWLIST``
   names it.
3. ``import flashinfer`` loads exactly the experimental modules that sanctioned
   support import pulls in, and nothing else.

``ALLOWLIST`` is a burn-down list, not a sanction: it names the module-level
imports core still had when this test landed, so the test is green while they
are removed one by one.  An entry that no longer matches any statement fails
the test until it is deleted, so the list only shrinks.

CPU-only; no GPU or JIT compilation required.
"""

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "flashinfer"
EXPERIMENTAL_ROOT = PACKAGE_ROOT / "experimental"
AOT_FILE = PACKAGE_ROOT / "aot.py"
EXPERIMENTAL_PACKAGE = "flashinfer.experimental"

# Module-level runtime imports of flashinfer.experimental that core still has:
# (repo-relative file, imported module as the statement spells it, why it is
# still there).  Keyed by file and module rather than by line so that an
# unrelated edit does not break the test; the failure message reports the
# current line.  Delete the entry together with the import.
ALLOWLIST = (
    (
        "flashinfer/mega_moe_v3.py",
        ".experimental.mega_moe_v3.runtime",
        "shim re-exporting the runtime symbols of the MegaMoE v3 package",
    ),
    (
        "flashinfer/moe_ep/__init__.py",
        "..experimental.cake_w8a8_megamoe_ep16.cake_config",
        "EP16 MegaMoE registration of the sm103 Cake backend (config)",
    ),
    (
        "flashinfer/moe_ep/__init__.py",
        "..experimental.cake_w8a8_megamoe_ep16.cake_weights",
        "EP16 MegaMoE registration of the sm103 Cake backend (weights)",
    ),
    (
        "flashinfer/moe_ep/backends/mega/kernel/sm103/__init__.py",
        "flashinfer.experimental.cake_w8a8_megamoe_ep16",
        "the only backend of the sm103 MegaMoE kernel path registers here",
    ),
)

# What ``import flashinfer`` may load from the experimental tree: the nvfp4
# per-token support module that flashinfer/gemm/gemm_base.py imports to
# register its checker, and the two packages on the way to it.
EXPECTED_MODULES_AFTER_IMPORT = [
    "flashinfer.experimental",
    "flashinfer.experimental.cake_nvfp4_per_token",
    "flashinfer.experimental.cake_nvfp4_per_token.support",
]


def _is_experimental(module: str) -> bool:
    return module == EXPERIMENTAL_PACKAGE or module.startswith(
        EXPERIMENTAL_PACKAGE + "."
    )


def _package_of(path: Path) -> list[str]:
    """Parts of the package the relative imports of ``path`` resolve against."""
    return list(path.relative_to(REPO_ROOT).parts[:-1])


def _resolve(module: str, level: int, package: list[str]) -> str | None:
    """Absolute name of ``module`` written with ``level`` leading dots.

    ``None`` when the dots climb above the top-level package.
    """
    if level == 0:
        return module
    if level > len(package):
        return None
    base = package[: len(package) - (level - 1)]
    return ".".join([*base, module] if module else base)


def _imported_modules(node: ast.Import | ast.ImportFrom, package: list[str]):
    """``(bound name, module as written, absolute module, target)`` per alias.

    ``target`` is the most specific module the alias can load.  For
    ``from M import name`` it is ``M.name``: statically ``name`` may be a
    submodule of ``M`` rather than an attribute, so ``from flashinfer import
    experimental`` reaches the tree and ``from ..experimental.pkg import
    support`` names the support module.  Otherwise it is the module itself.
    """
    if isinstance(node, ast.Import):
        return [
            (
                alias.asname or alias.name.partition(".")[0],
                alias.name,
                alias.name,
                alias.name,
            )
            for alias in node.names
        ]
    dots = "." * node.level
    if node.module:
        module = _resolve(node.module, node.level, package)
        return [
            (
                alias.asname or alias.name,
                dots + node.module,
                module,
                None if module is None else f"{module}.{alias.name}",
            )
            for alias in node.names
        ]
    # ``from . import a, b`` names modules of the package the dots point at.
    modules = [_resolve(alias.name, node.level, package) for alias in node.names]
    return [
        (alias.asname or alias.name, dots + alias.name, module, module)
        for alias, module in zip(node.names, modules, strict=True)
    ]


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(tree: ast.Module):
    """``(import statement, type_checking_only)`` for every import that runs at load.

    Descends into every block that executes when the module loads -- if / for /
    while / with / try / match blocks and class bodies -- but not into function
    bodies, which run when called.  Statements in the body of
    ``if TYPE_CHECKING:`` are reported with ``type_checking_only=True``; its
    ``else`` branch is ordinary runtime code.
    """

    def walk(nodes, type_checking_only):
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield node, type_checking_only
            elif isinstance(node, ast.If):
                guarded = type_checking_only or _is_type_checking(node.test)
                yield from walk(node.body, guarded)
                yield from walk(node.orelse, type_checking_only)
            elif not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.expr)
            ):
                # Class bodies, try / with / loop / match blocks, except
                # handlers and match cases: their statements run at load.  An
                # expression holds no import statement, and a lambda is one.
                yield from walk(ast.iter_child_nodes(node), type_checking_only)

    yield from walk(tree.body, False)


@dataclass(frozen=True)
class _ExperimentalImport:
    file: str  # repo-relative
    line: int
    written: str  # the module as the statement spells it
    statement: str
    type_checking_only: bool
    support: bool  # every name it imports is the sanctioned *.support module

    @property
    def runtime_violation(self) -> bool:
        return not self.type_checking_only and not self.support


def _core_files() -> list[Path]:
    """Every flashinfer/**/*.py outside flashinfer/experimental/, in a fixed order."""
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if not path.is_relative_to(EXPERIMENTAL_ROOT)
    )


def _scan_core() -> list[_ExperimentalImport]:
    """Module-level imports of flashinfer.experimental in core, in file order."""
    found = []
    for path in _core_files():
        source = path.read_text(encoding="utf-8")
        # An import statement of the experimental tree spells "experimental";
        # skipping the files that do not saves parsing tens of MB of kernels.
        if "experimental" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        package = _package_of(path)
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node, type_checking_only in _module_level_imports(tree):
            # One record per module the statement names.  It is sanctioned only
            # when every name imported from it is the lightweight *.support
            # module (README "Containment rules"): the module itself, or a
            # submodule the statement imports as a name.
            support: dict[str, bool] = {}
            for _, written, module, target in _imported_modules(node, package):
                if target is None or not _is_experimental(target):
                    continue
                sanctioned = module.endswith(".support") or target.endswith(".support")
                support[written] = support.get(written, True) and sanctioned
            for written, sanctioned in sorted(support.items()):
                found.append(
                    _ExperimentalImport(
                        rel,
                        node.lineno,
                        written,
                        ast.unparse(node),
                        type_checking_only,
                        sanctioned,
                    )
                )
    return found


def test_aot_registers_no_experimental_module():
    """flashinfer/aot.py builds the pre-built packages; experimental stays out.

    README "Relaxations vs. the stable-API checklist": AOT registration is
    prohibited because experimental features are JIT-only and never ship in the
    pre-built packages; a graduating feature registers for AOT as part of
    graduation.  Two consequences are checked: the file imports nothing from
    ``flashinfer.experimental``, at module level or inside a function, and every
    ``gen_*`` generator that ``gen_all_modules`` calls -- directly or through the
    helpers defined in the file -- is defined in aot.py or imported from a module
    outside the experimental tree.
    """
    tree = ast.parse(AOT_FILE.read_text(encoding="utf-8"), filename=str(AOT_FILE))
    package = _package_of(AOT_FILE)
    rel = AOT_FILE.relative_to(REPO_ROOT).as_posix()

    problems = []
    origins: dict[str, str] = {}  # name bound by an import -> absolute module
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = _imported_modules(node, package)
            origins.update((name, module) for name, _, module, _ in bound if module)
            if any(target and _is_experimental(target) for *_, target in bound):
                problems.append(f"{rel}:{node.lineno}: {ast.unparse(node)}")

    local_defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "gen_all_modules" in local_defs, f"{rel} defines no gen_all_modules"
    pending, seen, generators = ["gen_all_modules"], {"gen_all_modules"}, {}
    while pending:
        for node in ast.walk(local_defs[pending.pop()]):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if not name.lstrip("_").startswith("gen_"):
                continue
            if name not in local_defs:
                generators.setdefault(name, node.lineno)
            elif name not in seen:
                seen.add(name)
                pending.append(name)
    for name, line in sorted(generators.items()):
        origin = origins.get(name)
        if origin is None:
            problems.append(f"{rel}:{line}: {name} is neither defined nor imported")
        elif _is_experimental(origin):
            problems.append(f"{rel}:{line}: {name} comes from {origin}")

    assert not problems, (
        "flashinfer/aot.py reaches the experimental tree:\n" + "\n".join(problems)
    )


def test_core_has_no_module_level_experimental_import():
    """Core reaches flashinfer.experimental through deferred imports only.

    README "Containment rules": core must not import from
    ``flashinfer.experimental`` at module level, with one narrow exception --
    the lightweight support module that defines an ``@experimental_backend``
    checker -- and kernels are reached only through a deferred (function-local)
    import inside a thin entry point.  Imports inside ``if TYPE_CHECKING:`` are
    exempt as well: type checkers read them, the interpreter never runs them.

    Every other module-level import of ``flashinfer.experimental`` is reported
    as ``file:line: statement`` unless ``ALLOWLIST`` names its (file, module)
    pair; "module level" includes ``if`` / ``try`` / ``with`` / loop / ``match``
    blocks and class bodies, which all run at load.  An ``ALLOWLIST`` entry that
    matches no statement fails the test too, in a section of its own, so that
    removing an import also removes it from the burn-down list.
    """
    runtime = [found for found in _scan_core() if found.runtime_violation]
    allowed = {(file, module) for file, module, _ in ALLOWLIST}
    present = {(found.file, found.written) for found in runtime}

    sections = []
    unlisted = [
        found for found in runtime if (found.file, found.written) not in allowed
    ]
    if unlisted:
        sections.append(
            "module-level imports of flashinfer.experimental in core; defer each "
            "into the function that uses it:\n"
            + "\n".join(f"{f.file}:{f.line}: {f.statement}" for f in unlisted)
        )
    stale = [entry for entry in ALLOWLIST if entry[:2] not in present]
    if stale:
        sections.append(
            "ALLOWLIST entries that match no statement any more; delete them:\n"
            + "\n".join(
                f"{file}: {module} -- {reason}" for file, module, reason in stale
            )
        )
    if sections:
        pytest.fail("\n\n".join(sections), pytrace=False)


def test_import_flashinfer_loads_only_the_sanctioned_support_module():
    """``import flashinfer`` loads no experimental kernel.

    README "Containment rules": ``flashinfer/__init__.py`` must not eagerly
    import ``flashinfer.experimental``, and the deferred imports of the thin
    entry points keep ``import flashinfer`` from ever loading experimental
    kernels.  Of the module-level paths into the tree, ``import flashinfer``
    reaches only the sanctioned support import of ``flashinfer/gemm/gemm_base.py``
    (no ``ALLOWLIST`` file is on its import graph), so a fresh interpreter that
    imports flashinfer has exactly ``EXPECTED_MODULES_AFTER_IMPORT`` in
    ``sys.modules``.  Runs in a subprocess so that whatever this test session
    already imported does not count.
    """
    code = (
        "import json, sys\n"
        "import flashinfer\n"
        "loaded = [m for m in sys.modules if m == 'flashinfer.experimental'"
        " or m.startswith('flashinfer.experimental.')]\n"
        "print(json.dumps(sorted(loaded)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"import flashinfer exited with {proc.returncode}:\n{proc.stderr}",
            pytrace=False,
        )
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == EXPECTED_MODULES_AFTER_IMPORT, (
        "import flashinfer loaded a different set of experimental modules than "
        f"the sanctioned support import predicts:\nloaded:   {loaded}\n"
        f"expected: {EXPECTED_MODULES_AFTER_IMPORT}"
    )
