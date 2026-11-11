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

Names of the deterministic mHC training head module
``flashinfer.cake_mhc_head_det_train`` (FlashInfer issue #8928): every name
exported by the module and by its backend package
``flashinfer.experimental.cake.cake_mhc_head_det_train`` but the ALL_CAPS
constants carries the mode token ``det`` (``Det`` in a class name), no public
callable takes a ``deterministic`` keyword (the ``forward`` / ``backward`` of
the autograd ``Function`` classes included), the former modules and names are
gone (no module, no alias, no top-level name), the module imports its backend
on first use only, the registry routes each architecture and hidden size to its
own libraries (``ROUTES[arch][role][hidden]``, the small-problem libraries of
``SMALL_ROUTES``; ``build.json`` builds exactly the records' architectures),
and the package is wired at its path (test-helper key, package data,
pre-commit exclude, manifest checker row).
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
MODULE = "flashinfer.cake_mhc_head_det_train"
BACKEND = "flashinfer.experimental.cake.cake_mhc_head_det_train"
KEY = "cake.cake_mhc_head_det_train"
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
# The registry's records: library -> (role, architectures, CTA threads, BF16 values per vector, the hidden size the
# build is for, 0 for any).  On sm_100a and sm_103a a 64-bit library per hidden size: the forward in 256-thread CTAs
# on sm_100a and 128-thread CTAs on sm_103a, the backward in 512-thread CTAs (384 on sm_100a at
# 4096, 256 on sm_100a at 5120 and 7168), the accumulating backward in 768-thread CTAs.  On sm_107a the hidden-4096
# forward and writing backward move 256-bit vectors in 128-thread CTAs and the accumulating backward is the L2
# bulk-reduce library (128-bit vectors, 256-thread CTAs).  The unsuffixed libraries take any hidden size: 128-bit
# forward and 256-bit backward vectors in 256-thread CTAs, for a small problem on sm_107a (and the backward on
# sm_100a / sm_103a).
_LW64 = ("sm_100a", "sm_103a")
RECORDS = {
    "cake_mhc_det_train_head_fwd": ("head_fwd", ("sm_107a",), 256, 8, 0),
    **{
        f"cake_mhc_det_train_head_fwd__lw64_t128_h{h}": (
            "head_fwd",
            ("sm_103a",),
            128,
            4,
            h,
        )
        for h in (4096, 5120, 7168)
    },
    **{
        f"cake_mhc_det_train_head_fwd__lw64_t256_h{h}": (
            "head_fwd",
            ("sm_100a",),
            256,
            4,
            h,
        )
        for h in (4096, 5120, 7168)
    },
    "cake_mhc_det_train_head_fwd__lw256_t128_h4096": (
        "head_fwd",
        ("sm_107a",),
        128,
        16,
        4096,
    ),
    "cake_mhc_det_train_head_bwd": ("head_bwd", (*_LW64, "sm_107a"), 256, 16, 0),
    "cake_mhc_det_train_head_bwd__lw64_t384_h4096": (
        "head_bwd",
        ("sm_100a",),
        384,
        4,
        4096,
    ),
    **{
        f"cake_mhc_det_train_head_bwd__lw64_t512_h{h}": (
            "head_bwd",
            ("sm_103a",),
            512,
            4,
            h,
        )
        for h in (4096, 5120, 7168)
    },
    "cake_mhc_det_train_head_bwd__lw64_t256_h5120": (
        "head_bwd",
        ("sm_100a",),
        256,
        4,
        5120,
    ),
    "cake_mhc_det_train_head_bwd__lw64_t256_h7168": (
        "head_bwd",
        ("sm_100a",),
        256,
        4,
        7168,
    ),
    "cake_mhc_det_train_head_bwd__lw256_t128_h4096": (
        "head_bwd",
        ("sm_107a",),
        128,
        16,
        4096,
    ),
    "cake_mhc_det_train_head_bwd__acc": ("head_bwd_acc", ("sm_107a",), 256, 16, 0),
    **{
        f"cake_mhc_det_train_head_bwd__acc_lw64_t768_h{h}": (
            "head_bwd_acc",
            _LW64,
            768,
            4,
            h,
        )
        for h in (4096, 5120, 7168)
    },
    "cake_mhc_det_train_head_bwd__acc_l2red_lw128_t256_h4096": (
        "head_bwd_acc",
        ("sm_107a",),
        256,
        8,
        4096,
    ),
}
# The dynamic shared memory of a record's launch: the L2 bulk-reduce backward's segment of 256 16-byte vectors.
SMEM = {"cake_mhc_det_train_head_bwd__acc_l2red_lw128_t256_h4096": 4096}
ROLES = ("head_fwd", "head_bwd", "head_bwd_acc")

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


_LAZY_CHILD = r"""
import json, sys

backend = sys.argv[1]

import flashinfer

out = {"flashinfer": backend in sys.modules}
import flashinfer.cake_mhc_head_det_train as module

module.cake_mhc_head_det_train_forward
out["entry_point"] = backend in sys.modules
module.ARCHES
out["lazy_attribute"] = backend in sys.modules
print(json.dumps(out))
"""


@pytest.mark.cpu_only
def test_the_backend_is_imported_on_first_use() -> None:
    """In a fresh interpreter, ``import flashinfer`` (which re-exports the head's entry points) and reading an entry
    point import no backend; the first lazily resolved attribute (``ARCHES``) imports it."""
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_CHILD, BACKEND],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"flashinfer": False, "entry_point": False, "lazy_attribute": True}


@pytest.mark.cpu_only
def test_the_registry_routes_each_architecture_and_hidden_size_to_its_library() -> None:
    """The records are the libraries of :data:`RECORDS` (role, architectures, hidden size, a ``vectors`` grid rule
    whose thread count is the CTA's, ``kernel`` / ``library`` the record name, ``kernel_symbol``
    ``kernel_<library>``).  ``ROUTES[arch][role]`` maps every hidden size the architecture serves to the record of
    the role delivered for the architecture and built for that hidden size, or built for any where none is;
    ``SMALL_ROUTES`` sends a call of at most ``small_elements(arch, role)`` values per stream (``SMALL_ELEMENTS``,
    T 7168 at hidden 4096 for the sm_107a forward) to the unsuffixed library of its role (the writing backward on
    sm_100a and sm_103a, every role on sm_107a); :func:`route_module` follows both and refuses any other call, and a
    record refuses a hidden size it is not built for.  Every route stays inside the kernels' ``hv <= MAX_VECTORS``
    and its largest grid inside 31 bits."""
    loader = importlib.import_module(f"{BACKEND}.cake_jit")
    assert set(loader.MODULES) == set(RECORDS)
    for name, (role, arches, threads, elems, hidden) in RECORDS.items():
        record = loader.MODULES[name]
        assert record["role"] == role and tuple(record["arches"]) == arches, name
        assert record["hidden"] == hidden, name
        assert record["grid_rule"] == {
            "kind": "vectors",
            "threads": threads,
            "vector_elems": elems,
        }, name
        assert tuple(record["launch"]["block"]) == (threads, 1, 1), name
        assert record["launch"]["dynamic_smem_bytes"] == SMEM.get(name, 0), name
        assert record["kernel"] == record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"

    def built(arch: str, role: str, hidden: int) -> str:
        names = [
            name
            for name, (r, arches, _t, _e, h) in RECORDS.items()
            if r == role and arch in arches and h == hidden
        ]
        assert len(names) <= 1, names
        return names[0] if names else built(arch, role, 0)

    assert loader.HIDDEN_SIZES == {
        "sm_100a": (4096, 5120, 7168),
        "sm_103a": (4096, 5120, 7168),
        "sm_107a": (4096,),
    }
    routes, small_routes = loader.ROUTES, loader.SMALL_ROUTES
    assert routes == {
        arch: {
            role: {
                hidden: built(arch, role, hidden)
                for hidden in loader.supported_hidden_sizes(arch)
            }
            for role in ROLES
        }
        for arch in loader.ARCHES
    }
    small = "cake_mhc_det_train_head_bwd"
    assert small_routes == {
        "sm_100a": {"head_bwd": small},
        "sm_103a": {"head_bwd": small},
        "sm_107a": {
            "head_fwd": "cake_mhc_det_train_head_fwd",
            "head_bwd": small,
            "head_bwd_acc": "cake_mhc_det_train_head_bwd__acc",
        },
    }
    assert loader.SMALL_ELEMENTS == 2048 * 4096
    assert loader.SMALL_ELEMENTS_BY_ROUTE == {"sm_107a": {"head_fwd": 7168 * 4096}}
    for arch in loader.ARCHES:
        for role in ROLES:
            for hidden in loader.supported_hidden_sizes(arch):
                largest = loader.small_elements(arch, role) // hidden
                routed = loader.ROUTES[arch][role][hidden]
                to_small = loader.SMALL_ROUTES.get(arch, {}).get(role, routed)
                assert loader.route_module(arch, role, hidden, largest + 1) == routed
                assert loader.route_module(arch, role, hidden, largest) == to_small
                assert loader.route_module(arch, role, hidden, 1) == to_small
                for name in {routed, to_small}:
                    record = loader.MODULES[name]
                    hv = loader.vectors_per_token(record, hidden)
                    assert 1 <= hv <= loader.MAX_VECTORS, (arch, role, hidden, hv)
                    (grid_x, _, _) = loader.launch_grid(
                        record, loader.MAX_TOKENS, hidden
                    )
                    assert grid_x < 2**31, (arch, role, hidden, grid_x)
    for arch, role, hidden in (
        ("sm_90a", "head_fwd", 4096),
        ("sm_100a", "head_mean", 4096),
        ("sm_100a", "head_fwd", 8192),
        ("sm_107a", "head_bwd", 5120),
    ):
        with pytest.raises(ValueError, match="no exported"):
            loader.route_module(arch, role, hidden, 16384)
    with pytest.raises(ValueError, match="is built for hidden 4096, got 5120"):
        loader.vectors_per_token(
            loader.MODULES["cake_mhc_det_train_head_bwd__lw64_t384_h4096"], 5120
        )


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    """The test-helper key, the build manifest's package and libraries, the package data,
    the pre-commit exclude of the generated sources and the manifest checker's row name the package at
    its path; no key, package data, exclude, checker row or module file is left at the former path."""
    from tests.cake.test_helpers.cake_package import PACKAGES

    backend = importlib.import_module(BACKEND)
    directory = Path(backend.__file__).resolve().parent
    if directory != REPO.joinpath(*BACKEND.split(".")).resolve():
        pytest.skip(f"{BACKEND} is imported from {directory}, outside this checkout")
    former = "flashinfer.cake_mhc_det_train_head_backend"
    assert PACKAGES[KEY] == BACKEND and former not in PACKAGES.values()
    manifest = json.loads((directory / "build.json").read_text())
    assert manifest["package"] == BACKEND
    loader = importlib.import_module(f"{BACKEND}.cake_jit")
    assert {entry["name"] for entry in manifest["libraries"]} == {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        for arch in record["arches"]
    }
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
    assert not list(
        (REPO / "flashinfer/cake_mhc_det_train_head_backend/csrc").glob("*")
    )
    spec = importlib.util.spec_from_file_location(
        "check_cake_build_manifests", REPO / "scripts" / "check_cake_build_manifests.py"
    )
    checker = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_cake_build_manifests", checker)
    spec.loader.exec_module(checker)  # the checker never imports flashinfer
    (entry,) = [p for p in checker.PACKAGES if p.name == BACKEND]
    assert entry.path == "flashinfer/experimental/cake/cake_mhc_head_det_train"
    assert former not in [p.name for p in checker.PACKAGES]
    assert checker.suite_key(REPO, entry) == KEY
    assert not list(REPO.joinpath(*former.split(".")).glob("*.py"))
