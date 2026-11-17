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

Public names of the Cake deterministic GDN training routes (FlashInfer #8928).

The facades ``flashinfer.cake_gdn_det_train`` and ``flashinfer.cake_gdn_det_train_kcp`` sit at the top level and their
backend packages at ``flashinfer.experimental.cake.cake_gdn_det_train[_kcp]``.  Every name the facades and the backend
packages export carries the mode word ``det`` as a token, right after the op (constants keep their ALL_CAPS names
and are not exported); no entry binds a ``deterministic`` keyword (``inspect.signature(obj).bind_partial(
deterministic=False)`` raises ``TypeError``, the autograd ``Function.forward`` / ``backward`` included; the KCP
entries that take the option fields through ``**`` declare them in their signatures, so no signature has a ``**``
parameter); the former module paths do not import, and no namespace (the ``flashinfer`` top level, the facades, the
backend packages and their modules) binds a former name.  Importing a facade loads no backend module until first use
(checked in fresh interpreters).  The mode-word and keyword checks, and one check of each facade with no exemption
(``assert_cake_mode_module``), are the shared ones of ``tests/cake/test_helpers/cake_mode_names.py``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
from tests.cake.test_helpers.cake_mode_names import (
    assert_cake_mode_module,
    assert_mode_names,
    assert_no_deterministic_keyword,
)

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_gdn_det_train"
KCP_FACADE = "flashinfer.cake_gdn_det_train_kcp"
PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train"
KCP_PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train_kcp"
SUBMODULES = ("cake_backend", "cake_build", "cake_chain", "cake_jit", "cake_plan")
KCP_SUBMODULES = (
    "cake_backend",
    "cake_build",
    "cake_bytes",
    "cake_fake_backend",
    "cake_jit",
    "cake_kcp",
    "cake_kcp_plan",
    "cake_l2n",
    "cake_options",
    "cake_tables",
)
REMOVED_MODULES = (
    "flashinfer.experimental.cake_gdn_det_train",
    *(f"flashinfer.experimental.cake_gdn_det_train.{m}" for m in SUBMODULES),
    "flashinfer.experimental.cake_gdn_det_train_kcp",
    *(f"flashinfer.experimental.cake_gdn_det_train_kcp.{m}" for m in KCP_SUBMODULES),
    "flashinfer.gdn_train",
    "flashinfer.experimental.cake_gdn_train",
)

FACADE_ALL = ("CakeGDNDetFunction", "cake_gdn_det", "is_cake_gdn_det_train_supported")
PACKAGE_ALL = (
    "CakeGDNDetFunction",
    "cake_gdn_det",
    "cake_gdn_det_train_backward",
    "cake_gdn_det_train_forward",
)
KCP_FACADE_ALL = (
    "CakeGDNDetKcpContext",
    "CakeGDNDetKcpFunction",
    "CakeGDNDetKcpOptions",
    "CakeGDNDetKcpPartition",
    "CakeGDNDetKcpRecomputeHandle",
    "CakeGDNDetKcpSaved",
    "CakeGDNDetKcpWorkspace",
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_build_context",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_packed_qkv_views",
    "cake_gdn_det_train_kcp_partition",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_workspace_bytes",
    "is_cake_gdn_det_train_kcp_supported",
)
KCP_PACKAGE_ALL = (
    "CakeGDNDetKcpCall",
    "CakeGDNDetKcpContext",
    "CakeGDNDetKcpFragment",
    "CakeGDNDetKcpFunction",
    "CakeGDNDetKcpOptions",
    "CakeGDNDetKcpPackedQKVSplitFunction",
    "CakeGDNDetKcpPartition",
    "CakeGDNDetKcpRecomputeHandle",
    "CakeGDNDetKcpSaved",
    "CakeGDNDetKcpWorkspace",
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_build_context",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_packed_qkv_views",
    "cake_gdn_det_train_kcp_partition",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_workspace_bytes",
)
# (former name, name) of the single-rank route.
SINGLE_RANK_RENAMES = (
    ("chunk_gated_delta_rule_det", "cake_gdn_det"),
    ("ChunkGatedDeltaRuleDetFunction", "CakeGDNDetFunction"),
    ("gdn_chunk_det_forward", "cake_gdn_det_train_forward"),
    ("gdn_chunk_det_backward", "cake_gdn_det_train_backward"),
)
# (former name, name) of the KCP route.
KCP_RENAMES = (
    ("chunk_gdn_det_kcp", "cake_gdn_det_kcp"),
    ("ChunkGDNDetFunctionKcp", "CakeGDNDetKcpFunction"),
    ("chunk_gdn_det_fwd_kcp", "cake_gdn_det_train_kcp_forward"),
    ("chunk_gdn_det_bwd_kcp", "cake_gdn_det_train_kcp_backward"),
    ("partition_det_kcp", "cake_gdn_det_train_kcp_partition"),
    ("build_context_det_kcp", "cake_gdn_det_train_kcp_build_context"),
    ("packed_qkv_views_det_kcp", "cake_gdn_det_train_kcp_packed_qkv_views"),
    ("PackedQKVSplitDetKcp", "CakeGDNDetKcpPackedQKVSplitFunction"),
    ("gdn_det_workspace_bytes_kcp", "cake_gdn_det_train_kcp_workspace_bytes"),
    ("gdn_det_saved_bytes_kcp", "cake_gdn_det_train_kcp_saved_bytes"),
    ("gdn_det_comm_bytes_kcp", "cake_gdn_det_train_kcp_comm_bytes"),
    ("gdn_det_workspace_layout_kcp", "cake_gdn_det_train_kcp_workspace_layout"),
    ("gdn_det_saved_layout_kcp", "cake_gdn_det_train_kcp_saved_layout"),
    ("is_gdn_det_train_kcp_supported", "is_cake_gdn_det_train_kcp_supported"),
    ("OptionsDetKcp", "CakeGDNDetKcpOptions"),
    ("ContextDetKcp", "CakeGDNDetKcpContext"),
    ("PartitionDetKcp", "CakeGDNDetKcpPartition"),
    ("FragmentDetKcp", "CakeGDNDetKcpFragment"),
    ("WorkspaceDetKcp", "CakeGDNDetKcpWorkspace"),
    ("SavedDetKcp", "CakeGDNDetKcpSaved"),
    ("RecomputeHandleDetKcp", "CakeGDNDetKcpRecomputeHandle"),
    ("CallDetKcp", "CakeGDNDetKcpCall"),
)
# The no-det names of before the det rename (removed earlier; they stay absent).
NO_DET_NAMES = (
    "chunk_gated_delta_rule",
    "ChunkGatedDeltaRuleFunction",
    "gdn_chunk_forward",
    "gdn_chunk_backward",
    "chunk_gdn_kcp",
    "chunk_gdn_kcp_fwd",
    "chunk_gdn_kcp_bwd",
    "kcp_partition",
    "build_kcp_context",
    "ChunkGDNKcpFunction",
    "KcpContext",
    "KcpOptions",
    "KcpPartition",
    "KcpRecomputeHandle",
    "KcpSaved",
    "KcpWorkspace",
)
FORMER_NAMES = (
    tuple(old for old, _ in SINGLE_RANK_RENAMES)
    + tuple(old for old, _ in KCP_RENAMES)
    + NO_DET_NAMES
)
# Former Cake names that FlashInfer's own (non-Cake) stable API also uses at the top level: there the name must bind
# that API's object, never a Cake one.
STABLE_TOP_LEVEL = {"chunk_gated_delta_rule": "flashinfer.gdn_prefill"}
# The facade functions that call the package function of their name (the shared check ties the package object to
# the facade name through these pairs).
FORWARDS = ("cake_gdn_det",)
KCP_FORWARDS = (
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_partition",
    "cake_gdn_det_train_kcp_build_context",
    "cake_gdn_det_train_kcp_packed_qkv_views",
    "cake_gdn_det_train_kcp_workspace_bytes",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_comm_bytes",
)


def test_the_shared_check_reads_the_mode_word_by_token() -> None:
    # the shared mode-name check (tests/cake/test_helpers/cake_mode_names.py) passes the det names this suite exports and
    # fails a nondet name and a word that only starts with "det"
    module = importlib.import_module(KCP_PACKAGE)
    assert_mode_names(
        module,
        "det",
        names=[
            "CakeGDNDetKcpPackedQKVSplitFunction",
            "is_cake_gdn_det_train_kcp_supported",
        ],
    )
    for name in (
        "cake_gdn_nondet_kcp",
        "CakeGDNNondetKcpSaved",
        "detach_state",
        "Determinant",
    ):
        with pytest.raises(AssertionError):
            assert_mode_names(module, "det", names=[name], stem=False)


def _namespaces() -> dict[str, object]:
    out = {
        "flashinfer": flashinfer,
        FACADE: importlib.import_module(FACADE),
        KCP_FACADE: importlib.import_module(KCP_FACADE),
        PACKAGE: importlib.import_module(PACKAGE),
        KCP_PACKAGE: importlib.import_module(KCP_PACKAGE),
    }
    for sub in SUBMODULES:
        out[f"{PACKAGE}.{sub}"] = importlib.import_module(f"{PACKAGE}.{sub}")
    for sub in KCP_SUBMODULES:
        out[f"{KCP_PACKAGE}.{sub}"] = importlib.import_module(f"{KCP_PACKAGE}.{sub}")
    return out


# Untracked build leftovers a former package directory may keep after the move (ignored by git, never code).
LEFTOVER_ENTRIES = ("__pycache__", "aot")


def _namespace_leftover(name: str):
    """The module at ``name`` when it imports as an empty namespace package (a leftover untracked directory at a
    former package path), else ``None``."""
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:  # a parent path does not import
        return None
    if (
        spec is not None
        and spec.origin in (None, "namespace")
        and spec.submodule_search_locations
    ):
        return importlib.import_module(name)
    return None


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_former_module_paths_do_not_import(name: str) -> None:
    # A clean tree has no module at a former path.  A leftover untracked directory at a former package path (``aot/``
    # or ``__pycache__`` from an earlier build) makes the path import as an empty namespace package: then it binds no
    # former name and every former submodule path still raises ``ModuleNotFoundError``.
    sys.modules.pop(name, None)
    leftover = _namespace_leftover(name)
    if leftover is None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
        return
    for former in FORMER_NAMES:
        assert not hasattr(leftover, former), (name, former)
    for sub in sorted({*SUBMODULES, *KCP_SUBMODULES}):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{name}.{sub}")


def test_former_package_directories_are_gone() -> None:
    # gone, or holding only untracked build leftovers: no module, manifest or source
    experimental = Path(flashinfer.__file__).parent / "experimental"
    for name in ("cake_gdn_det_train", "cake_gdn_det_train_kcp", "cake_gdn_train"):
        directory = experimental / name
        if directory.exists():
            kept = sorted(
                p.name for p in directory.iterdir() if p.name not in LEFTOVER_ENTRIES
            )
            assert kept == [], (name, kept)
    assert (experimental / "cake" / "cake_gdn_det_train" / "build.json").is_file()
    assert (experimental / "cake" / "cake_gdn_det_train_kcp" / "build.json").is_file()


@pytest.mark.parametrize("name", FORMER_NAMES)
def test_no_namespace_binds_a_former_name(name: str) -> None:
    for module_name, module in _namespaces().items():
        if module_name == "flashinfer" and name in STABLE_TOP_LEVEL:
            # absent when that API's import is skipped
            obj = getattr(module, name, None)
            assert obj is None or obj.__module__ == STABLE_TOP_LEVEL[name], name
            continue
        assert not hasattr(module, name), (module_name, name)
        assert name not in getattr(module, "__all__", ()), (module_name, name)


@pytest.mark.parametrize(
    "module_name, names",
    [
        (FACADE, FACADE_ALL),
        (PACKAGE, PACKAGE_ALL),
        (KCP_FACADE, KCP_FACADE_ALL),
        (KCP_PACKAGE, KCP_PACKAGE_ALL),
    ],
)
def test_exports_are_the_mode_word_names(module_name: str, names: tuple) -> None:
    module = importlib.import_module(module_name)
    assert sorted(module.__all__) == sorted(names)
    # the mode word as a token, right after the op (the shared check)
    assert_mode_names(module, "det")
    for name in module.__all__:
        assert hasattr(module, name), name


@pytest.mark.parametrize(
    "module_name, renames",
    [
        (PACKAGE, SINGLE_RANK_RENAMES),
        (f"{PACKAGE}.cake_chain", SINGLE_RANK_RENAMES),
        (KCP_PACKAGE, KCP_RENAMES[:8] + KCP_RENAMES[14:]),
    ],
)
def test_renamed_objects_carry_their_new_names(module_name: str, renames) -> None:
    module = importlib.import_module(module_name)
    for _, new in renames:
        if not hasattr(module, new):  # layouts live in cake_bytes only
            continue
        obj = getattr(module, new)
        assert obj.__name__ == new, (module_name, new)
        assert obj.__qualname__ == new, (module_name, new)


def test_the_byte_layouts_carry_their_new_names() -> None:
    gb = importlib.import_module(f"{KCP_PACKAGE}.cake_bytes")
    for new in (
        "cake_gdn_det_train_kcp_workspace_layout",
        "cake_gdn_det_train_kcp_saved_layout",
        "cake_gdn_det_train_kcp_workspace_bytes",
        "cake_gdn_det_train_kcp_saved_bytes",
        "cake_gdn_det_train_kcp_comm_bytes",
    ):
        assert getattr(gb, new).__name__ == new
        assert new in gb.__all__


def _callables(module) -> list[tuple[str, object]]:
    out = []
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            for attr in ("__init__", "forward", "backward"):
                fn = obj.__dict__.get(attr)
                if fn is not None:
                    out.append((f"{name}.{attr}", getattr(fn, "__func__", fn)))
        elif callable(obj):
            out.append((name, obj))
    return out


@pytest.mark.parametrize("module_name", [FACADE, PACKAGE, KCP_FACADE, KCP_PACKAGE])
def test_no_entry_takes_a_deterministic_parameter(module_name: str) -> None:
    # the shared check on every exported entry, with no exemption: the entries that take the option keywords through
    # ``**knobs`` / ``**kw`` declare them in their signatures, so no signature has a ``**`` parameter
    module = importlib.import_module(module_name)
    assert_no_deterministic_keyword(module)
    found = _callables(module)
    assert found
    for name, fn in found:
        sig = inspect.signature(fn)
        assert "deterministic" not in sig.parameters, (module_name, name)
        assert all(p.kind is not p.VAR_KEYWORD for p in sig.parameters.values()), (
            module_name,
            name,
        )


@pytest.mark.parametrize(
    "facade, package, submodules, forwards",
    [
        (FACADE, PACKAGE, SUBMODULES, FORWARDS),
        (KCP_FACADE, KCP_PACKAGE, KCP_SUBMODULES, KCP_FORWARDS),
    ],
    ids=["single-rank", "kcp"],
)
def test_the_facade_follows_the_shared_mode_name_rules(
    facade: str, package: str, submodules, forwards
) -> None:
    # the shared check of one facade with no exemption: the mode word right after the op, no ``deterministic`` at any
    # signature, each object bound under its facade name only (the flashinfer top level, the backend package and its
    # modules), and the former module paths and names failing at import in a fresh interpreter
    module = importlib.import_module(facade)
    pkg = importlib.import_module(package)
    assert_cake_mode_module(
        module,
        "det",
        namespaces=[
            flashinfer,
            pkg,
            *(importlib.import_module(f"{package}.{sub}") for sub in submodules),
        ],
        forwards=[(name, getattr(pkg, name)) for name in forwards],
        old_modules=REMOVED_MODULES,
        old_names=[(m, name) for m in (facade, package) for name in FORMER_NAMES],
        cwd=Path(__file__).resolve().parents[2],
    )


def test_the_forwarded_keywords_refuse_deterministic() -> None:
    options = importlib.import_module(f"{KCP_PACKAGE}.cake_options")
    assert "deterministic" not in options.OPTION_NAMES
    with pytest.raises(TypeError, match="deterministic"):
        options.make_options(deterministic=False)
    with pytest.raises(TypeError, match="deterministic"):
        options.make_options({"deterministic": True})
    gb = importlib.import_module(f"{KCP_PACKAGE}.cake_bytes")
    for name in (
        "cake_gdn_det_train_kcp_comm_bytes",
        "cake_gdn_det_train_kcp_saved_bytes",
        "cake_gdn_det_train_kcp_workspace_bytes",
    ):
        sig = inspect.signature(getattr(gb, name))
        assert not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()), name
        with pytest.raises(TypeError):
            sig.bind_partial(deterministic=False)


def test_the_support_queries_return_a_bool_without_a_device() -> None:
    facade = importlib.import_module(FACADE)
    kcp_facade = importlib.import_module(KCP_FACADE)
    for query in (
        facade.is_cake_gdn_det_train_supported,
        kcp_facade.is_cake_gdn_det_train_kcp_supported,
    ):
        assert query("cpu") is False
        assert isinstance(query(), bool)


def _skipped_det_imports(path: Path) -> list[int]:
    """Lines of ``path`` that import a GDN det module through ``pytest.importorskip`` (a literal path naming
    ``gdn_det``, or a computed one): such a test skips instead of failing when the module moves."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        func = getattr(node, "func", None)
        if isinstance(node, ast.Call) and getattr(func, "attr", None) == "importorskip":
            arg = node.args[0] if node.args else None
            literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            if not literal or "gdn_det" in arg.value:
                out.append(node.lineno)
    return out


def test_det_tests_import_the_det_packages_without_a_skip() -> None:
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        (root / "tests" / "experimental").glob("test_cake_gdn_det_train*.py")
    )
    files.append(root / "benchmarks" / "bench_cake_gdn_det_train_kcp.py")
    assert len(files) > 8
    offenders = {p.name: lines for p in files if (lines := _skipped_det_imports(p))}
    assert not offenders, offenders


# Fresh interpreter: importing the facades (or flashinfer) loads no backend module; the backend loads on first
# attribute access of a lazy type.
_LAZY_CHECK = """
import importlib.util
import sys

{imports}
backend = [m for m in sys.modules if m.startswith(("flashinfer.experimental.cake.cake_gdn_det_train",
                                                   "flashinfer.experimental.cake_gdn_det_train"))]
assert not backend, backend
for old in ("flashinfer.experimental.cake_gdn_det_train", "flashinfer.experimental.cake_gdn_det_train_kcp"):
    # absent, or an empty namespace package (a leftover untracked directory of an earlier build)
    spec = importlib.util.find_spec(old)
    assert spec is None or (spec.origin in (None, "namespace") and spec.submodule_search_locations), old
import flashinfer
for old in {former!r}:
    if old in {stable!r}:
        assert getattr(getattr(flashinfer, old, None), "__module__", {stable!r}[old]) == {stable!r}[old], old
    else:
        assert not hasattr(flashinfer, old), old
import flashinfer.cake_gdn_det_train as facade
import flashinfer.cake_gdn_det_train_kcp as kcp_facade
assert facade.CakeGDNDetFunction.__name__ == "CakeGDNDetFunction"
assert kcp_facade.CakeGDNDetKcpFunction.__name__ == "CakeGDNDetKcpFunction"
assert "flashinfer.experimental.cake.cake_gdn_det_train" in sys.modules
assert "flashinfer.experimental.cake.cake_gdn_det_train_kcp" in sys.modules
print("GDN_DET_NAMES_OK")
"""

IMPORTS = {
    "flashinfer": "import flashinfer",
    "facade": f"import {FACADE}",
    "kcp_facade": f"import {KCP_FACADE}",
    "facade_names": f"from {FACADE} import cake_gdn_det, is_cake_gdn_det_train_supported\n"
    f"from {KCP_FACADE} import cake_gdn_det_kcp, cake_gdn_det_train_kcp_partition",
}


@pytest.mark.parametrize("form", sorted(IMPORTS))
def test_facade_imports_load_no_backend_until_first_use(form: str) -> None:
    code = _LAZY_CHECK.format(
        imports=IMPORTS[form], former=FORMER_NAMES, stable=STABLE_TOP_LEVEL
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
    assert "GDN_DET_NAMES_OK" in r.stdout, (form, r.stdout, r.stderr)
