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

Names of the non-deterministic mHC training head module
``flashinfer.cake_mhc_head_nondet_train`` (FlashInfer issue #8928): every name
exported by the module and by its backend package
``flashinfer.experimental.cake.cake_mhc_head_nondet_train`` but the ALL_CAPS
constants carries the mode token ``nondet`` (``Nondet`` in a class name), no
public callable takes a ``deterministic`` keyword (the ``forward`` /
``backward`` of the autograd ``Function`` class included), the former mHC
modules and names are gone, the top level of ``flashinfer`` has no mHC
``nondet`` name, the module imports its backend on first use only, the backend
never imports the deterministic head package, its registry (records and
routes) is the deterministic head's under the library stem
``cake_mhc_train_head_*``, and the package is wired at its path (test-helper
key, package data, pre-commit exclude, manifest checker row).
"""

import ast
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

MODE = "nondet"
MODULE = "flashinfer.cake_mhc_head_nondet_train"
BACKEND = "flashinfer.experimental.cake.cake_mhc_head_nondet_train"
DET_BACKEND = "flashinfer.experimental.cake.cake_mhc_head_det_train"
KEY = "cake.cake_mhc_head_nondet_train"
REPO = Path(__file__).resolve().parents[3]
# The library stems: the deterministic head's, and this package's (the stem of the non-deterministic mHC family).
DET_STEM = "cake_mhc_det_train_head_"
STEM = "cake_mhc_train_head_"
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
def test_the_module_mirrors_the_deterministic_head() -> None:
    """The module exports the deterministic head module's names with ``nondet`` in place of ``det``, and each
    entry point takes the same parameters."""
    module = importlib.import_module(MODULE)
    det = importlib.import_module("flashinfer.cake_mhc_head_det_train")

    def renamed(name: str) -> str:
        return name.replace("HeadDet", "HeadNondet").replace("head_det", "head_nondet")

    assert sorted(module.__all__) == sorted(renamed(n) for n in det.__all__)
    for name in det.__all__:
        if name.isupper() or inspect.isclass(getattr(det, name)):
            continue
        assert inspect.signature(getattr(module, renamed(name))) == inspect.signature(
            getattr(det, name)
        ), name


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
def test_former_names_are_gone_and_no_top_level_name() -> None:
    module = importlib.import_module(MODULE)
    backend = importlib.import_module(BACKEND)
    for owner in (module, backend, flashinfer):
        leftovers = [
            n
            for n in dir(owner)
            if n.startswith(REMOVED_PREFIXES) or n in REMOVED_NAMES
        ]
        assert not leftovers, (owner.__name__, leftovers)
    # the mode has no top-level name: its module is imported by its full name (importing it binds the submodule
    # attribute ``flashinfer.cake_mhc_head_nondet_train``, which is the module, not a re-exported name)
    leftovers = [
        n
        for n in dir(flashinfer)
        if "mhc" in n.lower()
        and MODE in n.lower()
        and not inspect.ismodule(getattr(flashinfer, n))
    ]
    assert not leftovers, leftovers
    lazy = [
        n for n in flashinfer._LAZY_EXPORTS if "mhc" in n.lower() and MODE in n.lower()
    ]
    assert not lazy, lazy


_LAZY_CHILD = r"""
import json, sys

module_name, backend, det_backend = sys.argv[1:4]

import flashinfer

out = {
    "flashinfer": [m for m in (module_name, backend) if m in sys.modules],
    "top_level": sorted(n for n in dir(flashinfer) if "mhc" in n.lower() and "nondet" in n.lower()),
}
import flashinfer.cake_mhc_head_nondet_train as module

module.cake_mhc_head_nondet_train_forward
out["entry_point"] = backend in sys.modules
module.ARCHES
out["lazy_attribute"] = backend in sys.modules
out["det_backend"] = det_backend in sys.modules
print(json.dumps(out))
"""


@pytest.mark.cpu_only
def test_the_backend_is_imported_on_first_use() -> None:
    """In a fresh interpreter, ``import flashinfer`` imports neither this module nor its backend and exposes no mHC
    ``nondet`` name; importing the module and reading an entry point import no backend; the first lazily resolved
    attribute (``ARCHES``) imports it; and the backend's modules never import the deterministic head package."""
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_CHILD, MODULE, BACKEND, DET_BACKEND],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {
        "flashinfer": [],
        "top_level": [],
        "entry_point": False,
        "lazy_attribute": True,
        "det_backend": False,
    }, out


def _backend_dir() -> Path:
    backend = importlib.import_module(BACKEND)
    directory = Path(backend.__file__).resolve().parent
    if directory != REPO.joinpath(*BACKEND.split(".")).resolve():
        pytest.skip(f"{BACKEND} is imported from {directory}, outside this checkout")
    return directory


@pytest.mark.cpu_only
def test_the_backend_imports_nothing_of_the_det_head() -> None:
    """No module of the package names the deterministic head package in an import, and no relative import climbs
    out of the package directory."""
    directory = _backend_dir()
    found = []
    for path in sorted(directory.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
                if node.level > 1:
                    found.append(
                        f"{path.name}:{node.lineno}: climbs {node.level} levels"
                    )
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            if any("cake_mhc_head_det_train" in name for name in names):
                found.append(f"{path.name}:{node.lineno}: imports {names}")
    assert not found, found


def _renamed(value):
    """``value`` (a registry literal of the deterministic head) with its library stem replaced by this package's."""
    if isinstance(value, str):
        return value.replace(DET_STEM, STEM)
    if isinstance(value, dict):
        return {_renamed(k): _renamed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_renamed(v) for v in value)
    return value


@pytest.mark.cpu_only
def test_the_registry_is_the_det_head_registry_renamed() -> None:
    """Every record of the registry (launch, grid rule, argument plan, logical map, architectures, hidden size), the
    route of every (architecture, role, hidden size), the small-problem routes and every host constant are the
    deterministic head's, with the library stem ``cake_mhc_det_train_head_`` replaced by ``cake_mhc_train_head_``:
    the same programs, launched on the same architectures, under this package's own library names."""
    ours = importlib.import_module(f"{BACKEND}.cake_jit")
    det = importlib.import_module(f"{DET_BACKEND}.cake_jit")
    modules, routes, small_routes = ours.MODULES, ours.ROUTES, ours.SMALL_ROUTES
    assert modules == _renamed(det.MODULES)
    assert routes == _renamed(det.ROUTES)
    assert small_routes == _renamed(det.SMALL_ROUTES)
    assert all(record["library"].startswith(STEM) for record in modules.values()), (
        modules
    )
    for arch, roles in routes.items():
        for role, by_hidden in roles.items():
            for hidden, name in by_hidden.items():
                for rows in (
                    1,
                    ours.small_elements(arch, role) // hidden,
                    ours.small_elements(arch, role) // hidden + 1,
                ):
                    assert ours.route_module(arch, role, hidden, rows) == _renamed(
                        det.route_module(arch, role, hidden, rows)
                    )
                assert ours.route_module(arch, role, hidden, 1 << 20) == name
                record = modules[name]
                assert record["role"] == role and arch in record["arches"]
                assert record["hidden"] in (0, hidden)
    for name in (
        "ARCHES",
        "ARCH_BY_CAPABILITY",
        "STREAMS",
        "HIDDEN_SIZES",
        "MAX_TOKENS",
        "MAX_VECTORS",
        "SMALL_ELEMENTS",
        "SMALL_ELEMENTS_BY_ROUTE",
    ):
        assert getattr(ours, name) == getattr(det, name), name


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    """The test-helper key, the build manifest's package and libraries, the package data,
    the pre-commit exclude of the generated sources and the manifest checker's row name the package
    at its path."""
    from tests.cake.test_helpers.cake_package import PACKAGES

    directory = _backend_dir()
    assert PACKAGES[KEY] == BACKEND
    manifest = json.loads((directory / "build.json").read_text())
    assert manifest["package"] == BACKEND
    loader = importlib.import_module(f"{BACKEND}.cake_jit")
    libraries = {entry["name"] for entry in manifest["libraries"]}
    assert libraries == {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        for arch in record["arches"]
    }
    assert all(name.startswith(STEM) for name in libraries), libraries
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    assert package_data[BACKEND] == ["csrc/**", "build.json"]
    pre_commit = (REPO / ".pre-commit-config.yaml").read_text()
    # pre-commit skips every generated source of the package (the generic #8928 rule), still checks its Python
    # modules
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
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", REPO / "scripts" / "check_cake_build_manifests.py"
    )
    checker = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_cake_build_manifests", checker)
    spec.loader.exec_module(checker)  # the checker never imports flashinfer
    (entry,) = [p for p in checker.PACKAGES if p.name == BACKEND]
    assert entry.path == "flashinfer/experimental/cake/cake_mhc_head_nondet_train"
    assert checker.suite_key(REPO, entry) == KEY
