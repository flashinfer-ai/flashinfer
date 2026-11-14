# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Every Cake package imports with ``flashinfer`` blocked.

A build that vendors a Cake package copies its directory into its own tree and
imports it under its own name, without flashinfer (#6553).  The renames of
#8928 must keep that working
(https://github.com/flashinfer-ai/flashinfer/issues/8928#issuecomment-6680200530):
every backend is self-contained.  This is the one check that imports every
package of ``PACKAGES`` (``tests/cake/test_helpers/cake_package.py``); family suites
do not repeat it.  It reads the table when it runs, so a rename PR updates that
table and nothing here; a row whose directory does not exist fails.

For each distinct import path, a fresh interpreter launched outside the
repository:

1. blocks flashinfer: ``sys.modules["flashinfer"] = None`` and a meta-path
   finder that raises ``ImportError`` for ``flashinfer`` and ``flashinfer.*``;
2. drops from ``sys.path`` the working directory and every entry at or above a
   directory it loads (the repository root, from ``PYTHONPATH`` or a ``.pth``
   file), and keeps the others, so torch and tvm_ffi still import.  An editable
   install's finder still maps its other top-level names (``b12x``,
   ``build_backend``, ``build_utils``) to the checkout; this check does not block
   them, so a package must not import them;
3. loads the package directory where it is, without a copy, as the top-level
   package ``vendored_<directory>``, the name the vendored gate of
   ``scripts/check_cake_build_manifests.py`` gives its copy, so a relative
   import that climbs out of the directory (``from ..other_package import x``)
   fails;
4. imports every module (``*.py``, in subdirectories too) of the directory, one
   by one: the backend suites import them (``cake_package(key, "cake_backend")``),
   and several ``__init__.py`` import nothing.  A module an earlier one imported
   is not run again.  When an ``__init__.py`` fails, its package stays registered
   unexecuted, so its other modules are still checked;
5. fails if ``sys.modules`` holds a ``flashinfer`` or ``flashinfer.*`` entry
   other than ``None`` at the end.

A package that needs a sibling package (a KCP package and its single-rank
package) declares it in its README and in its ``build.json`` as
``common.requires.packages``: a list of sibling directory names.  The check
then loads the package and its siblings as children of one parent package,
``vendored_<directory>.<directory>`` beside ``vendored_<directory>.<sibling>``,
as the vendored gate lays them out, so ``from ..<sibling> import x`` resolves.
A sibling loads when a module imports it, in any order, as from the gate's
namespace directory; a module that names the sibling builds the name from its
own parent package (``f"{parent}.{sibling}" if parent else sibling``).  Without
that key, the check reads the ``requires`` of the package's row in
``scripts/check_cake_build_manifests.py`` (full import paths); the two must
name the same packages, and ``build.json`` takes precedence here.  A sibling
must be a package of ``PACKAGES`` with the same parent package and an existing
directory, so the import names the same package in the repository and in a
vendoring build.  No directory that holds packages goes on ``sys.path``.

To fix a failure, make the module import nothing outside its own directory
when it loads: move a flashinfer import into the function that needs it,
guarded the way ``cake_build._jit_toolchain`` guards the JIT fallback, keep a
copy of what it uses, or declare the sibling package it must ship with.

``ALLOWLIST`` is a burn-down list, not a sanction: it names, per package, the
modules that failed when this test landed.  Any other failing module fails the
test, and so does a flashinfer module left in ``sys.modules``.  A listed module
must still fail: once it imports cleanly, or once ``PACKAGES`` no longer names
its package, the test fails until the entry is updated.  So the family PR that
fixes a module (or declares the sibling it imports) deletes that module from
the entry in the same change, and the entry once it lists none; a rename PR
moves the entry with its package.  The list only shrinks.

CI runs this file in the unit-test run over ``tests/``; it lives in
``tests/cpu/infra`` because that run does not recurse into ``tests/experimental``.
The ``experimental-tests`` block of a PR body accepts only
``tests/experimental/``, so a PR that changes ``PACKAGES``, a package directory
or a sibling declaration names this file when it asks for CI:
``@flashinfer-bot run tests/cpu/cake/test_cake_packages_standalone.py``.

CPU-only: the children run with ``CUDA_VISIBLE_DEVICES`` empty, as on a
CPU-only consumer, and write no bytecode into the repository.  This module
imports nothing from flashinfer; the children need only the test environment
(torch, tvm_ffi).
"""

from collections.abc import Collection
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from tests.cake.test_helpers.cake_package import PACKAGES

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_CHECKER = REPO_ROOT / "scripts" / "check_cake_build_manifests.py"

# Modules that do not import with flashinfer blocked yet, per package:
# (import path, module stems, why); ``__init__`` is the package itself and
# ``sub.module`` a module in a subdirectory.  Delete a stem together with the
# import it names, and the entry once it names none.
ALLOWLIST = (
    (
        "flashinfer.experimental.cake_stepfun_moe",
        ("__init__",),
        "__init__.py decorates its entry points with flashinfer's "
        "flashinfer_experimental_api ('from ...api_logging import')",
    ),
)


def _manifest_checker() -> ModuleType:
    """``scripts/check_cake_build_manifests.py``, loaded by its path (it imports
    only the standard library)."""
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", MANIFEST_CHECKER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # its dataclasses resolve their annotations through sys.modules
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


# Import path of a package -> the ``requires`` of its row in
# scripts/check_cake_build_manifests.py: its siblings when its build.json does
# not declare common.requires.packages.
CHECKER_REQUIRES: dict[str, tuple[str, ...]] = {
    package.name: package.requires
    for package in _manifest_checker().PACKAGES
    if package.requires
}

# Where a package declares its siblings: first choice, fallback.
BUILD_JSON_SIBLINGS = "build.json common.requires.packages"
CHECKER_SIBLINGS = "the requires of its row in scripts/check_cake_build_manifests.py"
# Marks the child's result line on stdout.
RESULT_PREFIX = "cake-standalone-result: "
# Seconds a child may take; one that hangs fails its package, not the CI job.
CHILD_TIMEOUT = 600
# Lines kept per failing module (traceback) or child (stderr) in a message.
TAIL_LINES = 12
FIX = (
    "Import nothing outside the package directory when a module loads: move a "
    "flashinfer import into the function that needs it (guarded like "
    "cake_build._jit_toolchain), keep a copy of what it uses, or declare the "
    "sibling package it must ship with in its README and in "
    f"{BUILD_JSON_SIBLINGS}."
)

# Runs in a fresh interpreter; argv[1] is the JSON layout of ``_layout``.
_CHILD = """
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import traceback
import types

layout = json.loads(sys.argv[1])
cwd = os.path.realpath(os.getcwd())
directories = [layout["directory"]] + [d for _, d in layout["siblings"]]
loaded = [os.path.realpath(directory) for directory in directories]


def exposes(entry):
    # the working directory, a loaded directory or a directory above one
    entry = os.path.realpath(entry or os.curdir)
    return entry == cwd or any(os.path.commonpath([entry, d]) == entry for d in loaded)


def package_spec(name, directory):
    return importlib.util.spec_from_file_location(
        name,
        os.path.join(directory, "__init__.py"),
        submodule_search_locations=[directory],
    )


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] == "flashinfer":
            raise ImportError(f"flashinfer is blocked: {name}")
        return None


siblings = dict(layout["siblings"])


class Siblings(importlib.abc.MetaPathFinder):
    # a declared sibling loads when a module imports it, in any order
    def find_spec(self, name, path=None, target=None):
        if name in siblings:
            return package_spec(name, siblings[name])
        return None


sys.path[:] = [entry for entry in sys.path if not exposes(entry)]
sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None  # every import of flashinfer now fails
parent = layout["parent"]
if parent:  # the package and its siblings, children of one parent package
    vendor = types.ModuleType(parent)
    vendor.__path__ = []  # no directory: only the declared siblings resolve
    sys.modules[parent] = vendor
    sys.meta_path.insert(1, Siblings())
failed = []


def register(name, module):
    sys.modules[name] = module
    owner, _, attribute = name.rpartition(".")
    if owner in sys.modules:  # a namespace directory may have no module yet
        setattr(sys.modules[owner], attribute, module)


def load(name, directory):
    spec = package_spec(name, directory)
    module = importlib.util.module_from_spec(spec)
    register(name, module)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # an unexecuted package instead, so its modules still import one by one
        register(name, importlib.util.module_from_spec(spec))
        raise


def attempt(stem, name, action, *arguments):
    try:
        action(*arguments)
    except (Exception, SystemExit) as exc:
        failed.append(
            {
                "module": name,
                "stem": stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(traceback.format_exception(exc)),
            }
        )


package = layout["package"]
attempt("__init__", package, load, package, layout["directory"])
for stem, relative in layout["modules"]:
    name = f"{package}.{stem}"
    if name in sys.modules:  # imported cleanly: a failed import removes it
        continue
    if os.path.basename(relative) == "__init__.py":
        directory = os.path.join(layout["directory"], os.path.dirname(relative))
        attempt(stem, name, load, name, directory)
    else:
        attempt(stem, name, importlib.import_module, name)
# the None entry for flashinfer blocks it; any other flashinfer entry leaks
leaked = sorted(
    name
    for name, module in sys.modules.items()
    if name.partition(".")[0] == "flashinfer"
    and (module is not None or name != "flashinfer")
)
print(layout["prefix"] + json.dumps({"failed": failed, "leaked": leaked}))
"""


def _keys_by_path() -> dict[str, tuple[str, ...]]:
    """Each distinct import path of ``PACKAGES`` -> the keys that name it."""
    keys: dict[str, list[str]] = {}
    for key, path in PACKAGES.items():
        keys.setdefault(path, []).append(key)
    return {path: tuple(names) for path, names in keys.items()}


KEYS = _keys_by_path()


def _directory(path: str) -> Path:
    return REPO_ROOT.joinpath(*path.split("."))


def _allowed(path: str) -> tuple[set[str], str]:
    """The module stems ``ALLOWLIST`` names for ``path``, and why."""
    for listed, stems, reason in ALLOWLIST:
        if listed == path:
            return set(stems), reason
    return set(), ""


def _siblings(
    path: str,
    directory: Path,
    known: Collection[str],
    checker_requires: dict[str, tuple[str, ...]],
) -> list[Path]:
    """The directories of the sibling packages of ``path`` (in ``directory``):
    the directory names of its build.json ``common.requires.packages``, resolved
    in its parent package, when build.json has that key, else the import paths
    of ``checker_requires[path]``.  A name that is not a directory name, or a
    sibling that is not a package of ``known`` with the same parent package or
    has no ``__init__.py``, fails the test."""
    parent = path.rpartition(".")[0]
    declared, source = checker_requires.get(path, ()), CHECKER_SIBLINGS
    manifest = directory / "build.json"
    if manifest.is_file():
        common = json.loads(manifest.read_text()).get("common") or {}
        requires = common.get("requires") or {}
        if "packages" in requires:
            names = requires["packages"]
            listed = isinstance(names, list) and all(
                isinstance(name, str) and name.isidentifier() for name in names
            )
            if not listed:
                pytest.fail(
                    f"{path}: {BUILD_JSON_SIBLINGS} is {names!r}, not a list of "
                    "sibling directory names",
                    pytrace=False,
                )
            declared = tuple(f"{parent}.{name}" if parent else name for name in names)
            source = BUILD_JSON_SIBLINGS
    siblings, problems = [], []
    for other in declared:
        sibling = directory.parent / other.rpartition(".")[2]
        if other.rpartition(".")[0] != parent:
            problems.append(f"{other} is not in the parent package {parent}")
        elif not (sibling / "__init__.py").is_file():
            problems.append(
                f"{other} has no {sibling.name}/__init__.py beside {directory.name}/"
            )
        elif other not in known:
            problems.append(f"{other} is not a package of PACKAGES")
        else:
            siblings.append(sibling)
    if problems:
        pytest.fail(
            f"{path} declares its sibling packages in {source}:\n"
            + "\n".join(f"- {problem}" for problem in problems),
            pytrace=False,
        )
    return siblings


def _modules(directory: Path) -> list[list[str]]:
    """``[stem, path relative to directory]`` of every module of ``directory``
    but its ``__init__.py``: ``sub/inner.py`` is ``sub.inner`` and
    ``sub/__init__.py`` is ``sub``; sorted, so a subpackage precedes its
    modules."""
    modules = []
    for file in directory.rglob("*.py"):
        relative = file.relative_to(directory)
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append([".".join(parts), relative.as_posix()])
    return sorted(modules)


def _layout(directory: Path, siblings: list[Path]) -> dict[str, Any]:
    """What a child loads: ``directory`` as the top-level package
    ``vendored_<name>``, or, with ``siblings``, as ``vendored_<name>.<name>``
    beside ``vendored_<name>.<sibling name>``, the layout of the vendored gate
    of scripts/check_cake_build_manifests.py; a sibling loads when a module
    imports it."""
    neutral = f"vendored_{directory.name}"
    parent = neutral if siblings else ""
    return {
        "parent": parent,
        "package": f"{parent}.{directory.name}" if parent else neutral,
        "directory": str(directory),
        "siblings": [[f"{parent}.{other.name}", str(other)] for other in siblings],
        "modules": _modules(directory),
        "prefix": RESULT_PREFIX,
    }


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-TAIL_LINES:])


def _run_child(layout: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """What a fresh interpreter loading ``layout`` reports: the modules that fail
    to import (``failed``) and the flashinfer modules loaded anyway (``leaked``).
    A child that hangs, prints no result or exits non-zero fails the test."""
    # -B: no __pycache__ beside the package sources
    command = [sys.executable, "-B", "-c", _CHILD, json.dumps(layout)]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # a CPU-only consumer
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        pytest.fail(
            f"the child interpreter loading {layout['directory']} did not finish "
            f"in {CHILD_TIMEOUT} s:\n{_tail(stderr)}",
            pytrace=False,
        )
    _, found, record = result.stdout.rpartition(RESULT_PREFIX)
    if result.returncode or not found:
        state = "after its result" if found else "without a result"
        pytest.fail(
            f"the child interpreter loading {layout['directory']} exited "
            f"{result.returncode} {state}:\n{_tail(result.stderr)}",
            pytrace=False,
        )
    return json.loads(record.partition("\n")[0])


def _describe(layout: dict[str, Any], failed: list[dict[str, str]]) -> str:
    """Where the child loaded the package, and each failure with its traceback."""
    if layout["parent"]:
        siblings = ", ".join(name for name, _ in layout["siblings"])
        lines = [f"loaded as {layout['package']} beside {siblings}:"]
    else:
        lines = [f"loaded as the top-level package {layout['package']}:"]
    for failure in failed:
        lines.append(f"- {failure['module']}: {failure['error']}")
        tail = failure["traceback"].splitlines()[-TAIL_LINES:]
        lines.extend(f"    {line}" for line in tail)
    return "\n".join(lines)


@pytest.mark.cpu_only
@pytest.mark.parametrize("path", sorted(KEYS))
def test_package_imports_without_flashinfer(path: str, tmp_path: Path) -> None:
    """Every module of the package imports with flashinfer blocked, except the
    modules ``ALLOWLIST`` names, which must still fail, and no flashinfer module
    is loaded."""
    directory = _directory(path)
    relative = directory.relative_to(REPO_ROOT)
    keys = ", ".join(KEYS[path])
    if not directory.is_dir():
        pytest.fail(
            f"PACKAGES names {path} ({keys}) but {relative} does not exist",
            pytrace=False,
        )
    if not (directory / "__init__.py").is_file():
        pytest.fail(
            f"PACKAGES names {path} ({keys}) but {relative}/__init__.py does not exist",
            pytrace=False,
        )
    layout = _layout(directory, _siblings(path, directory, KEYS, CHECKER_REQUIRES))
    result = _run_child(layout, tmp_path)
    failed = result["failed"]
    stems, reason = _allowed(path)
    sections = []
    if result["leaked"]:
        sections.append(
            f"{path} ({keys}) left flashinfer modules in sys.modules with flashinfer "
            f"blocked: {', '.join(result['leaked'])}; a module loads or registers "
            "them past the import system, which ALLOWLIST does not cover"
        )
    unlisted = [failure for failure in failed if failure["stem"] not in stems]
    if unlisted:
        sections.append(f"{path} ({keys}) {_describe(layout, unlisted)}\n{FIX}")
    fixed = sorted(stems - {failure["stem"] for failure in failed})
    if fixed:
        sections.append(
            f"ALLOWLIST names {', '.join(fixed)} of {path}, which import with "
            "flashinfer blocked now or no longer exist; delete them from the entry"
        )
    if sections:
        pytest.fail("\n\n".join(sections), pytrace=False)
    if failed:
        pytest.xfail(
            f"ALLOWLIST: {reason}\n{path} ({keys}) {_describe(layout, failed)}"
        )


@pytest.mark.cpu_only
def test_allowlist_names_packages() -> None:
    """``PACKAGES`` names a package, and every ``ALLOWLIST`` entry names a
    module of an import path of ``PACKAGES`` that no other entry names."""
    problems = []
    if not KEYS:
        problems.append("PACKAGES in tests/cake/test_helpers/cake_package.py is empty")
    paths = [path for path, _, _ in ALLOWLIST]
    stale = sorted(set(paths) - set(KEYS))
    if stale:
        problems.append(
            f"stale ALLOWLIST entries {stale}: PACKAGES names no such import path; "
            "delete them, or follow the rename"
        )
    repeated = sorted({path for path in paths if paths.count(path) > 1})
    if repeated:
        problems.append(f"ALLOWLIST names {repeated} more than once; merge them")
    empty = sorted(path for path, stems, _ in ALLOWLIST if not stems)
    if empty:
        problems.append(f"ALLOWLIST entries {empty} name no module; delete them")
    if problems:
        pytest.fail("\n".join(problems), pytrace=False)


# The self-tests below lay out src/probe beside src/sibling (value = 1) and
# src/other, which takes value from src/sibling.
_FROM_SIBLING = "from ..sibling import value\n"
_UNBLOCK = "import sys\n\nsys.modules.pop('flashinfer')\nimport flashinfer\n"
_REGISTER = (
    "import sys\nimport types\n\n"
    "sys.modules['flashinfer.jit'] = types.ModuleType('flashinfer.jit')\n"
)
# fails when it runs a second time, as a registry that rejects duplicates does
_RUN_ONCE = (
    "import builtins\n\n"
    "assert not hasattr(builtins, 'probe_ops')\n"
    "builtins.probe_ops = 1\n"
)


def _build_json(*siblings: str) -> str:
    return json.dumps({"common": {"requires": {"packages": list(siblings)}}})


# a build.json without common.requires.packages
_NO_PACKAGES_KEY = json.dumps(
    {"common": {"requires": {"cuda_toolkit": {"min": "12.8"}}}}
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("files", "requires", "errors"),
    [
        (
            {"module.py": "import flashinfer.jit\n"},
            (),
            {"module": "ModuleNotFoundError"},
        ),
        ({"module.py": _FROM_SIBLING}, (), {"module": "ImportError"}),
        ({"module.py": _FROM_SIBLING}, ("src.sibling",), {}),
        (
            {"module.py": _FROM_SIBLING, "build.json": _NO_PACKAGES_KEY},
            ("src.sibling",),
            {},
        ),
        ({"module.py": _FROM_SIBLING, "build.json": _build_json("sibling")}, (), {}),
        (
            {"module.py": _FROM_SIBLING, "build.json": _build_json()},
            ("src.sibling",),
            {"module": "ImportError"},
        ),
        (
            {
                "module.py": "from ..other import value\n",
                "build.json": _build_json("other", "sibling"),
            },
            (),
            {},
        ),
        (
            {"__init__.py": "import flashinfer\n", "module.py": _FROM_SIBLING},
            (),
            {"__init__": "ModuleNotFoundError", "module": "ImportError"},
        ),
        (
            {
                "sub/__init__.py": "import flashinfer\n",
                "sub/inner.py": "",
                "flat/inner.py": "from flashinfer import jit\n",
            },
            (),
            {"sub": "ModuleNotFoundError", "flat.inner": "ModuleNotFoundError"},
        ),
        ({"__init__.py": "from . import ops\n", "ops/__init__.py": _RUN_ONCE}, (), {}),
        (
            {
                "module.py": "import sibling\n",
                "other.py": "import marker\n",
                "build.json": _build_json("sibling"),
            },
            (),
            {"module": "ModuleNotFoundError", "other": "ModuleNotFoundError"},
        ),
        ({"module.py": _UNBLOCK}, (), {"module": "ImportError"}),
        ({"module.py": _REGISTER}, (), {"flashinfer.jit": "leaked"}),
    ],
    ids=[
        "flashinfer-import",
        "relative-import-above-the-package",
        "sibling-in-the-checker-requires",
        "checker-when-build-json-lacks-the-key",
        "sibling-in-build-json",
        "build-json-before-the-checker",
        "siblings-in-any-order",
        "modules-of-a-failing-package",
        "nested-modules",
        "subpackage-imported-once",
        "no-sys-path-to-the-cwd-or-above-the-package",
        "flashinfer-unblocked-by-a-module",
        "flashinfer-module-registered-by-hand",
    ],
)
def test_child_rejects_imports_outside_the_package(
    files: dict[str, str],
    requires: tuple[str, ...],
    errors: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child fails a flashinfer import, also after a module drops the
    ``None`` entry, an import above the package and an import from the working
    directory or from a ``PYTHONPATH`` entry above the package; resolves a
    sibling declared in build.json or, without that key, in the checker's
    ``requires``, whatever the order of the siblings; still imports the modules
    of a failing package and of subdirectories, each once; and reports a
    flashinfer module registered by hand, so a pass above is not vacuous."""
    source = tmp_path / "src"
    package = source / "probe"
    for relative, text in {"__init__.py": "", **files}.items():
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_text(text)
    for name, text in {"sibling": "value = 1\n", "other": _FROM_SIBLING}.items():
        (source / name).mkdir()
        (source / name / "__init__.py").write_text(text)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "marker.py").write_text("")
    # the directory above the packages on PYTHONPATH, as a checkout on
    # PYTHONPATH puts the repository root there
    paths = [str(source), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, paths)))
    known = {"src.probe", "src.sibling", "src.other"}
    siblings = _siblings("src.probe", package, known, {"src.probe": requires})
    layout = _layout(package, siblings)
    result = _run_child(layout, cwd)
    found = {
        failure["stem"]: failure["error"].partition(":")[0]
        for failure in result["failed"]
    }
    found.update(dict.fromkeys(result["leaked"], "leaked"))
    assert found == errors, _describe(layout, result["failed"])


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("manifest", "requires", "message"),
    [
        (
            _build_json("missing"),
            (),
            f"{BUILD_JSON_SIBLINGS}:\n"
            "- src.missing has no missing/__init__.py beside probe/",
        ),
        (
            _build_json("stray"),
            (),
            f"{BUILD_JSON_SIBLINGS}:\n- src.stray is not a package of PACKAGES",
        ),
        (
            "",
            ("other.stray",),
            f"{CHECKER_SIBLINGS}:\n- other.stray is not in the parent package src",
        ),
        (
            json.dumps({"common": {"requires": {"packages": "stray"}}}),
            (),
            f"{BUILD_JSON_SIBLINGS} is 'stray', not a list of sibling directory names",
        ),
        (
            _build_json("src/stray"),
            (),
            f"{BUILD_JSON_SIBLINGS} is ['src/stray'], not a list of sibling "
            "directory names",
        ),
    ],
    ids=[
        "missing-directory",
        "not-in-packages",
        "other-parent-in-the-checker",
        "not-a-list",
        "path-not-a-directory-name",
    ],
)
def test_declared_sibling_must_be_a_package(
    manifest: str, requires: tuple[str, ...], message: str, tmp_path: Path
) -> None:
    """A declared sibling fails the package when build.json gives no list of
    directory names, its directory does not exist, ``PACKAGES`` does not name it
    or its parent package differs."""
    package = tmp_path / "src" / "probe"
    stray = tmp_path / "src" / "stray"
    for directory in (package, stray):
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("")
    if manifest:
        (package / "build.json").write_text(manifest)
    with pytest.raises(pytest.fail.Exception) as failure:
        _siblings("src.probe", package, {"src.probe"}, {"src.probe": requires})
    assert message in str(failure.value)
