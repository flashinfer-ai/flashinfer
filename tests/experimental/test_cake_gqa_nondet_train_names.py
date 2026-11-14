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

Names of the non-deterministic GQA HD256 training module ``flashinfer.cake_gqa_nondet_train`` (#8928).

The public names are the facade's ``__all__`` and the backend objects it forwards to or re-exports
(``flashinfer.experimental.cake.cake_gqa_nondet_train``): each carries the mode (``nondet``) as a token, the
former names and package paths are gone (no alias), and no entry point takes a ``deterministic`` keyword.  The rules
shared by every #8928 family are checked with ``tests.cake.test_helpers.cake_mode_names``; the checks here are this
family's own.  Nothing reads torch's deterministic mode: every output repeats bit for bit, so the package raises no
alert.  Internal helpers and build identities (program and library stems ``cake_gqa_train_<hex>``, kernel symbols,
csrc file names) keep their names.
"""

import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer
import flashinfer.cake_gqa_nondet_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "flashinfer.experimental.cake.cake_gqa_nondet_train"
ENTRIES = (
    "cake_gqa_nondet",
    "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_workspace",
    "cake_gqa_nondet_train_workspace_bytes",
)
# The ``typing.Callable`` alias the facade re-exports: a type, not an entry point.
TYPE_ALIASES = ("CakeGQANondetDkvAccSource",)
# Backend classes and types the facade re-exports (resolved on first access).
REEXPORTED = (
    "CakeGQANondetDkvAccSource",
    "CakeGQANondetFunction",
    "CakeGQANondetWorkspace",
)
# The backend objects the facade forwards to or re-exports (``cake_backend``, also exported by the package).
FORWARDED = (
    "cake_gqa_nondet",
    "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_workspace_bytes",
) + REEXPORTED
# Module paths that no longer import: the former facade and the former backend package.
REMOVED_MODULES = (
    "flashinfer.gqa_train",
    "flashinfer.experimental.cake_gqa_train.cake_backend",
    "flashinfer.experimental.cake_gqa_train.cake_jit",
)
# The former facade entries (``flashinfer.gqa_train``).
REMOVED_ENTRIES = (
    "varlen_gqa_attention",
    "varlen_gqa_attention_fwd",
    "varlen_gqa_attention_bwd",
    "gqa_train_workspace",
    "gqa_train_workspace_bytes",
)
# Former backend names that neither ``cake_backend`` nor the package defines or exports (no aliases).
REMOVED_NAMES = (
    "varlen_gqa_attention",
    "gqa_attention_fwd",
    "gqa_attention_bwd",
    "VarlenGQAAttentionFunction",
    "Workspace",
    "DkvAccSource",
    "workspace_bytes",
)


def _backend(module_name=None):
    return importlib.import_module(
        BACKEND if module_name is None else f"{BACKEND}.{module_name}"
    )


def _fresh(code: str):
    """``code`` run in a fresh interpreter; returns the JSON it prints last."""
    repo = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_paths_do_not_import(name):
    """A former module path does not import (a stale ``__pycache__`` or ``aot`` directory can still make the bare
    package path import as a namespace package, so modules of it are imported)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_former_entries_are_gone():
    """No former entry is an attribute of the facade, the backend package or ``flashinfer`` (fresh interpreter)."""
    for name in REMOVED_ENTRIES:
        assert not hasattr(public, name), name
    for name in REMOVED_NAMES:
        for module in (_backend(), _backend("cake_backend")):
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in module.__all__, (module.__name__, name)
    code = (
        "import json\n"
        "import flashinfer\n"
        "names = set(dir(flashinfer)) | set(getattr(flashinfer, '__all__', ()))\n"
        f"former = {REMOVED_ENTRIES!r} + ('gqa_train',)\n"
        "print(json.dumps(sorted(n for n in names if n in former or 'gqa_train' in n)))\n"
    )
    assert _fresh(code) == []


def test_facade_all_is_the_entries_and_reexports():
    """The facade's ``__all__`` is exactly its five entries and three re-exports; each entry is a stable API (no
    experimental marking) named after the module."""
    assert sorted(public.__all__) == sorted(ENTRIES + REEXPORTED)
    for name in ENTRIES:
        fn = getattr(public, name)
        assert name.startswith("cake_gqa_nondet"), name
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name, name
    backend = _backend("cake_backend")
    for name in REEXPORTED:
        assert getattr(public, name) is getattr(backend, name), name


def test_the_shared_mode_rules_hold():
    """The #8928 rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names
    carry ``nondet`` after the op, no entry point takes ``deterministic``, ``flashinfer``, the backend package and
    ``cake_backend`` bind each facade object and each backend object a facade entry calls only under the facade's
    name, and the former module paths and names fail at import in a fresh interpreter.

    These are the checks of ``assert_cake_mode_module``, called one by one so that the keyword check leaves out
    ``CakeGQANondetDkvAccSource``: a ``typing.Callable`` alias has typing's ``(*args, **kwargs)`` signature, but it
    is a type, not an entry point (calling it raises ``TypeError``).  ``test_no_entry_takes_a_mode_keyword`` covers
    the entry points by their own signatures."""
    package, backend = _backend(), _backend("cake_backend")
    old_names = [("flashinfer", "gqa_train")]
    old_names += [
        ("flashinfer.cake_gqa_nondet_train", name) for name in REMOVED_ENTRIES
    ]
    old_names += [
        (module.__name__, name)
        for module in (package, backend)
        for name in REMOVED_NAMES
    ]
    assert_mode_names(public)
    assert_no_deterministic_keyword(
        [getattr(public, name) for name in public.__all__ if name not in TYPE_ALIASES]
    )
    assert_same_object_same_name(
        public,
        (flashinfer, package, backend),
        forwards=[
            (name, getattr(backend, name)) for name in FORWARDED if name in ENTRIES
        ],
    )
    assert_old_names_fail(
        REMOVED_MODULES + ("flashinfer.experimental.cake_gqa_train",),
        old_names,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_public_names_carry_the_mode():
    """Every backend object the facade forwards to or re-exports carries ``nondet`` after the op (the shared rule,
    applied to the backend's names); every package ``__all__`` name is such an object or an ALL_CAPS constant, and
    the backend's internal helpers (``default_workspace``, ``load``, ``record``, ...) are not package attributes."""
    backend = _backend("cake_backend")
    package = _backend()
    assert_mode_names(package, names=FORWARDED)
    for name in FORWARDED:
        assert name in backend.__all__ and name in package.__all__, name
        assert getattr(package, name) is getattr(backend, name), name
    for name in package.__all__:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in FORWARDED, name
    internal = [
        name
        for name in backend.__all__
        if name not in FORWARDED and not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
    ]
    assert "default_workspace" in internal
    for name in internal:
        assert not hasattr(package, name), name


def test_no_entry_takes_a_mode_keyword():
    """No facade entry, forwarded backend callable or autograd ``Function.forward`` accepts a ``deterministic``
    keyword (``inspect.signature(fn).bind_partial(deterministic=False)`` is a ``TypeError``: no such parameter, no
    ``**kwargs``); passing it to a facade entry is a ``TypeError`` from argument binding, before any work."""
    backend = _backend("cake_backend")
    fns = {f"facade.{name}": getattr(public, name) for name in ENTRIES}
    for name in FORWARDED:
        obj = getattr(backend, name)
        if name == "CakeGQANondetFunction":
            fns[f"backend.{name}.forward"] = obj.forward
        elif name == "CakeGQANondetWorkspace" or inspect.isfunction(obj):
            fns[f"backend.{name}"] = obj
    autograd = [
        cls
        for cls in vars(backend).values()
        if inspect.isclass(cls)
        and issubclass(cls, torch.autograd.Function)
        and cls.__module__ == backend.__name__
    ]
    assert autograd == [backend.CakeGQANondetFunction]
    assert len(fns) == 11
    for label, fn in fns.items():
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values())
    t = torch.zeros(4, 2, 256, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    calls = {
        "cake_gqa_nondet": (t, t, t, cu, cu),
        "cake_gqa_nondet_train_forward": (t, t, t, cu, cu),
        "cake_gqa_nondet_train_backward": (t, t, t, t, t, t, cu, cu),
        "cake_gqa_nondet_train_workspace": (),
        "cake_gqa_nondet_train_workspace_bytes": (4, 2, 1),
    }
    for name, args in calls.items():
        for value in (True, False):
            with pytest.raises(
                TypeError, match="unexpected keyword argument 'deterministic'"
            ):
                getattr(public, name)(*args, deterministic=value)


def test_nothing_reads_torch_mode():
    """The facade and the backend never read torch's deterministic mode and never raise its alert: every sum has one
    fixed order, so there is nothing to alert about (the backend suite checks the strict and ``warn_only`` modes on a
    device)."""
    reads = re.compile(r"torch\.(?:are|is)_deterministic_algorithms_\w*enabled\(")
    assert not reads.search(inspect.getsource(public))
    for module_name in ("cake_backend", "cake_jit", None):
        source = inspect.getsource(_backend(module_name))
        assert not reads.search(source), module_name
        assert "alert_not_deterministic" not in source, module_name
        assert "_alert_nondeterministic" not in source, module_name


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the moved package; program and library stems keep their build
    identity."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_gqa_train"
    path = Path(jit.__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    manifest = json.loads(path.read_text())
    assert manifest["package"] == BACKEND
    assert all(
        entry["name"].startswith("cake_gqa_train_") for entry in manifest["libraries"]
    )
