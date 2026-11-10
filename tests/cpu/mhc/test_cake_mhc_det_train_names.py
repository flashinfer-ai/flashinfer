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

Names of the deterministic mHC training module ``flashinfer.cake_mhc_det_train``
(FlashInfer issue #8928): every name exported by the module and by its backend
package ``flashinfer.experimental.cake.cake_mhc_det_train`` but the ALL_CAPS
constants carries the mode token ``det`` (``Det`` in a class name), no public
callable takes a ``deterministic`` keyword (the ``forward`` / ``backward`` of
the autograd ``Function`` classes included), the former modules and names are
gone (no module, no alias, no top-level name), and the package is wired at its
path (test-helper key, package data, pre-commit exclude,
manifest checker row).
"""

import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer

MODE = "det"
MODULE = "flashinfer.cake_mhc_det_train"
BACKEND = "flashinfer.experimental.cake.cake_mhc_det_train"
KEY = "cake.cake_mhc_det_train"
REPO = Path(__file__).resolve().parents[3]
# The former modules of the mHC training modes (FlashInfer 0.7.3's package names, the backend paths before
# #8928, the former non-deterministic module): importing them fails.
REMOVED_MODULES = (
    "flashinfer.cake_mhc_train",
    "flashinfer.cake_mhc_train_backend",
    "flashinfer.cake_mhc_train_head_backend",
    "flashinfer.cake_mhc_det_train_backend",
    "flashinfer.cake_mhc_det_train_head_backend",
    "flashinfer.experimental.cake.cake_mhc_train",
    "flashinfer.experimental.cake.cake_mhc_train_backend",
)
# The former names: their prefixes (0.7.3's names without the mode word, the classes and head names before
# #8928) and the generic route helpers.
REMOVED_PREFIXES = (
    "cake_mhc_train_",
    "is_cake_mhc_train_",
    "CakeMHCTrain",
    "CakeMHCDetTrain",
    "cake_mhc_det_train_head",
    "is_cake_mhc_det_train_head",
)
REMOVED_NAMES = ("route_key", "route_modules")

_CHILD = r"""
import importlib, json, sys

out = {}
for name in sys.argv[1:]:
    try:
        imported = importlib.import_module(name)
    except ImportError:
        out[name] = None
    else:
        # a namespace package left by a stale __pycache__ directory has no origin: it holds no code
        spec = getattr(imported, "__spec__", None)
        out[name] = getattr(spec, "origin", None)
print(json.dumps(out))
"""


def _mode_tokens(name: str) -> set[str]:
    """The mode words of ``name``: ``det`` / ``nondet`` as ``_``-separated tokens of a snake_case name, ``Det`` /
    ``Nondet`` as words of a CamelCase one (a substring test would read ``nondet`` as ``det`` too)."""
    if name[:1].isupper():
        return {word.lower() for word in re.findall(r"(Nondet|Det)(?=[A-Z]|$)", name)}
    return set(re.findall(r"(?:^|_)(det|nondet)(?=_|$)", name))


@pytest.mark.cpu_only
@pytest.mark.parametrize("module_name", (MODULE, BACKEND))
def test_every_name_carries_the_mode(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__all__, module_name
    without = [
        n for n in module.__all__ if not n.isupper() and _mode_tokens(n) != {MODE}
    ]
    assert not without, (module_name, without)


@pytest.mark.cpu_only
@pytest.mark.parametrize("module_name", (MODULE, BACKEND))
def test_no_public_callable_takes_a_mode_keyword(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            functions = (obj.forward, obj.backward)
        elif callable(obj):
            functions = (obj,)
        else:
            assert name.isupper(), name  # only the constants are not callable
            continue
        for function in functions:
            parameters = inspect.signature(function).parameters.values()
            assert all(p.name != "deterministic" for p in parameters), name
            assert all(p.kind is not p.VAR_KEYWORD for p in parameters), name


@pytest.mark.cpu_only
def test_former_modules_fail_at_import() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *REMOVED_MODULES],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == dict.fromkeys(REMOVED_MODULES), out


@pytest.mark.cpu_only
def test_former_names_are_gone() -> None:
    module = importlib.import_module(MODULE)
    backend = importlib.import_module(BACKEND)
    for owner in (module, backend):
        leftovers = [
            n
            for n in dir(owner)
            if n.startswith(REMOVED_PREFIXES) or n in REMOVED_NAMES
        ]
        assert not leftovers, (owner.__name__, leftovers)
    leftovers = [n for n in dir(flashinfer) if n.startswith(REMOVED_PREFIXES)]
    assert not leftovers, leftovers


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    """The test-helper key, the build manifest's package, the package data, the
    pre-commit exclude of the generated sources and the manifest checker's row name the package at its path; no
    key, package data, exclude, checker row or module file is left at the former path."""
    from tests.cake.test_helpers.cake_package import PACKAGES

    backend = importlib.import_module(BACKEND)
    directory = Path(backend.__file__).resolve().parent
    if directory != REPO.joinpath(*BACKEND.split(".")).resolve():
        pytest.skip(f"{BACKEND} is imported from {directory}, outside this checkout")
    former = "flashinfer.cake_mhc_det_train_backend"
    assert PACKAGES[KEY] == BACKEND and former not in PACKAGES.values()
    manifest = json.loads((directory / "build.json").read_text())
    assert manifest["package"] == BACKEND
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    assert package_data[BACKEND] == ["csrc/**", "build.json"]
    assert former not in package_data
    pre_commit = (REPO / ".pre-commit-config.yaml").read_text()
    # pre-commit skips every generated source of the package (a per-package entry or the generic #8928 rule), still
    # checks its Python modules; the former tree holds no sources (the shared exclude may still list its path)
    exclude = re.compile(
        next(ln for ln in pre_commit.splitlines() if ln.startswith("exclude:"))
        .split(":", 1)[1]
        .strip()
    )
    root, package_dir = REPO.resolve(), directory.resolve()
    sources = [
        p.relative_to(root).as_posix()
        for p in sorted((package_dir / "csrc").rglob("*"))
        if p.is_file()
    ]
    assert sources and all(exclude.search(p) for p in sources), [
        p for p in sources if not exclude.search(p)
    ]
    assert not exclude.search(
        (package_dir / "cake_jit.py").relative_to(root).as_posix()
    )
    assert not list((REPO / "flashinfer/cake_mhc_det_train_backend/csrc").glob("*"))
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", REPO / "scripts" / "check_cake_build_manifests.py"
    )
    checker = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_cake_build_manifests", checker)
    spec.loader.exec_module(checker)  # the checker never imports flashinfer
    (entry,) = [p for p in checker.PACKAGES if p.name == BACKEND]
    assert entry.path == "flashinfer/experimental/cake/cake_mhc_det_train"
    assert former not in [p.name for p in checker.PACKAGES]
    assert checker.suite_key(REPO, entry) == KEY
    assert not list(REPO.joinpath(*former.split(".")).glob("*.py"))
