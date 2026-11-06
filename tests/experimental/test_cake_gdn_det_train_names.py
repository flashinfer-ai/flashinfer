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
packages export carries the mode word ``det`` as a word (split on ``_`` and at CamelCase boundaries; constants keep
their ALL_CAPS names and are not exported); no entry binds a ``deterministic`` keyword
(``inspect.signature(obj).bind_partial(deterministic=False)`` raises ``TypeError``, the autograd ``Function.forward``
included; the entries whose ``**`` keywords are the KCP option fields or the byte queries' keyword-only arguments
refuse it where those keywords are bound); the former module paths do not import, and no namespace (the ``flashinfer`` top level, the
facades, the backend packages and their modules) binds a former name.  Importing a facade loads no backend module until
first use (checked in fresh interpreters).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_gdn_det_train"
KCP_FACADE = "flashinfer.cake_gdn_det_train_kcp"
PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train"
KCP_PACKAGE = "flashinfer.experimental.cake.cake_gdn_det_train_kcp"
SUBMODULES = ("cake_backend", "cake_build", "cake_chain", "cake_jit")
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
# Entries whose ``**`` keywords are the fields of ``CakeGDNDetKcpOptions`` (``make_options`` refuses an unknown name
# with ``TypeError``) or the keyword-only arguments of the byte queries of ``cake_bytes``: ``bind_partial`` accepts any
# keyword there, so ``test_the_forwarded_keywords_refuse_deterministic`` checks where those keywords are bound.
VAR_KEYWORD_ENTRIES = (
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_workspace_bytes",
)
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _words(name: str) -> list[str]:
    """The words of ``name``, lower-cased: split on ``_`` and at CamelCase boundaries (an acronym run is one word)."""
    return [w.lower() for part in name.split("_") for w in _WORD.findall(part)]


def test_the_word_split_finds_the_mode_word_by_token() -> None:
    assert _words("CakeGDNDetKcpPackedQKVSplitFunction") == [
        "cake",
        "gdn",
        "det",
        "kcp",
        "packed",
        "qkv",
        "split",
        "function",
    ]
    assert _words("is_cake_gdn_det_train_kcp_supported") == [
        "is",
        "cake",
        "gdn",
        "det",
        "train",
        "kcp",
        "supported",
    ]
    for name in (
        "cake_gdn_nondet_kcp",
        "CakeGDNNondetKcpSaved",
        "detach_state",
        "Determinant",
    ):
        assert "det" not in _words(name), name


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


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_former_module_paths_do_not_import(name: str) -> None:
    sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_former_package_directories_are_gone() -> None:
    experimental = Path(flashinfer.__file__).parent / "experimental"
    for name in ("cake_gdn_det_train", "cake_gdn_det_train_kcp", "cake_gdn_train"):
        assert not (experimental / name).exists(), name
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
    for name in module.__all__:
        assert "det" in _words(name) or name.isupper(), name
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
    module = importlib.import_module(module_name)
    found = _callables(module)
    assert found
    var_keyword = []
    for name, fn in found:
        sig = inspect.signature(fn)
        assert "deterministic" not in sig.parameters, (module_name, name)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            var_keyword.append(name)
            continue
        with pytest.raises(TypeError):
            sig.bind_partial(deterministic=False)
    want = VAR_KEYWORD_ENTRIES if module_name in (KCP_FACADE, KCP_PACKAGE) else ()
    assert sorted(var_keyword) == sorted(want), module_name


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
    assert importlib.util.find_spec(old) is None, old
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
