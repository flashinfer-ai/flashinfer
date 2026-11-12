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

Names of the deterministic LM-head training module ``flashinfer.cake_lm_head_det_train``.

The public names are the facade's ``__all__`` and the backend objects it forwards to
(``flashinfer.experimental.cake.cake_lm_head_det_train``): each carries the mode (``det``), the
former names and package paths are gone (no alias), and no entry point takes a ``deterministic``
keyword or reads torch's deterministic mode.  Internal helpers and build identities (program and
library stems ``cake_lm_head_det_loss_<hex>``) keep their names.
"""

import importlib
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import flashinfer.cake_lm_head_det_train as public
from flashinfer.api_logging import ExperimentalWarning
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs

pytestmark = pytest.mark.cpu_only

BACKEND = "flashinfer.experimental.cake.cake_lm_head_det_train"
ENTRIES = (
    "cake_lm_head_det_train_logprob",
    "cake_lm_head_det_train_loss",
    "cake_lm_head_det_train_preload",
    "cake_lm_head_det_train_support_kl",
    "cake_lm_head_det_train_support_logprob",
)
# The backend objects the facade forwards to, by module (each defined in that module).
FORWARDED = {
    "cake_backend": (
        "cake_lm_head_det_train_loss",
        "cake_lm_head_det_train_logprob",
        "cake_lm_head_det_train_logprob_temperature",
        "cake_lm_head_det_train_logprob_keep",
        "CakeLMHeadDetLossFunction",
        "CakeLMHeadDetLogprobFunction",
        "CakeLMHeadDetLogprobTemperatureFunction",
        "CakeLMHeadDetLogprobKeepFunction",
    ),
    "cake_backend_opt_in": (
        "cake_lm_head_det_train_loss",
        "cake_lm_head_det_train_logprob",
        "cake_lm_head_det_train_logprob_temperature",
        "cake_lm_head_det_train_preload",
        "CakeLMHeadDetLossFunction",
        "CakeLMHeadDetLogprobFunction",
        "CakeLMHeadDetLogprobTemperatureFunction",
    ),
    "cake_support": (
        "cake_lm_head_det_train_support_logprob",
        "cake_lm_head_det_train_support_kl",
        "CakeLMHeadDetSupportFunction",
    ),
}
# Package paths that no longer import: the deterministic package before this module's name and its alias
# package before the det naming.
REMOVED_MODULES = (
    "flashinfer.experimental.cake_lm_head_det_loss.cake_backend",
    "flashinfer.experimental.cake_lm_head_det_loss.cake_support",
    "flashinfer.experimental.cake_lm_head_loss.cake_backend",
)
# The former entries of the deterministic facade (``flashinfer.chunked_lm_head`` before this module).
REMOVED_ENTRIES = (
    "chunked_lm_head_det_loss",
    "chunked_lm_head_det_logprob",
    "chunked_lm_head_det_preload",
    "chunked_lm_head_det_support_logprob",
    "chunked_lm_head_det_support_kl",
)
# Former names that the backend modules no longer define or export (no aliases): the names before the det
# naming, then the det names before this module's name.
REMOVED_NAMES = {
    "cake_backend": (
        "chunked_lm_head_loss",
        "chunked_lm_head_logprob",
        "prepare_lm_head_loss",
        "validate_lm_head_inputs",
        "lm_head_loss_workspace_size",
        "LmHeadLossRunner",
        "ChunkedLmHeadLossFunction",
        "ChunkedLmHeadLogprobFunction",
        "load_cake_lm_head_loss_module",
        "chunked_lm_head_det_loss",
        "chunked_lm_head_det_logprob",
        "chunked_lm_head_det_logprob_temperature",
        "chunked_lm_head_det_logprob_keep",
        "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadDetLogprobFunction",
        "ChunkedLmHeadDetLogprobTemperatureFunction",
        "ChunkedLmHeadDetLogprobKeepFunction",
    ),
    "cake_backend_opt_in": (
        "chunked_lm_head_loss",
        "chunked_lm_head_logprob",
        "prepare_lm_head_loss",
        "validate_lm_head_inputs",
        "LmHeadLossRunner",
        "ChunkedLmHeadLossFunction",
        "ChunkedLmHeadLogprobFunction",
        "load_cake_lm_head_loss_module",
        "chunked_lm_head_det_loss",
        "chunked_lm_head_det_logprob",
        "chunked_lm_head_det_logprob_temperature",
        "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadDetLogprobFunction",
        "ChunkedLmHeadDetLogprobTemperatureFunction",
        "preload",
    ),
    "cake_support": (
        "chunked_lm_head_support_logprob",
        "chunked_lm_head_support_kl",
        "SupportLogprobFunction",
        "chunked_lm_head_det_support_logprob",
        "chunked_lm_head_det_support_kl",
        "SupportDetLogprobFunction",
    ),
    "cake_jit": ("load_cake_lm_head_loss_module",),
}
H_HOST, V_HOST = 256, 512


def _tokens(name: str) -> list:
    """The lowercase words of ``name``: split on ``_``, then on CamelCase (``CakeLMHeadDetSupportFunction`` ->
    cake, lm, head, det, support, function)."""
    words = []
    for part in name.split("_"):
        words += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", part)
    return [w.lower() for w in words]


def _autograd_functions(module) -> list:
    """The ``torch.autograd.Function`` classes defined in ``module``."""
    return [
        (name, value)
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, torch.autograd.Function)
        and value.__module__ == module.__name__
    ]


def _backend(module_name):
    return importlib.import_module(f"{BACKEND}.{module_name}")


def _fresh(code: str) -> dict:
    """``code`` run in a fresh interpreter (no module of this process imported); returns the JSON it prints."""
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


def _host_call():
    X = torch.zeros(4, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(4, dtype=torch.int64)
    return X, W, labels


def _support_args():
    X, W, _ = _host_call()
    ids = torch.zeros(4, dtype=torch.int64)
    offsets = torch.arange(5, dtype=torch.int64)
    return X, W, ids, offsets


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_package_paths_do_not_import(name):
    """A module of a former package path does not import (a stale ``__pycache__`` or ``aot`` directory can still
    make the bare package path import as a namespace package, so a module is imported)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_former_entries_fail_at_import():
    """In a fresh interpreter, no former deterministic entry and no entry of this module imports from
    ``flashinfer.chunked_lm_head`` (a removed module; its non-deterministic entries are now
    ``flashinfer.cake_lm_head_nondet_train``)."""
    names = REMOVED_ENTRIES + ENTRIES
    code = (
        "import json\n"
        "result = {}\n"
        f"for name in {names!r}:\n"
        "    try:\n"
        "        exec(f'from flashinfer.chunked_lm_head import {name}')\n"
        "        result[name] = 'imported'\n"
        "    except ImportError as exc:\n"
        "        result[name] = type(exc).__name__\n"
        "print(json.dumps(result))\n"
    )
    result = _fresh(code)
    assert len(result) == 10
    assert set(result.values()) <= {"ImportError", "ModuleNotFoundError"}, result


def test_former_names_are_gone_from_the_backend():
    """No backend module defines or exports a former name (no aliases)."""
    assert sum(len(names) for names in REMOVED_NAMES.values()) == 39
    assert len({name for names in REMOVED_NAMES.values() for name in names}) == 24
    for module_name, names in REMOVED_NAMES.items():
        module = _backend(module_name)
        for name in names:
            assert not hasattr(module, name), (module_name, name)
            assert name not in getattr(module, "__all__", ()), (module_name, name)


def test_facade_all_is_the_five_entries():
    """The facade's ``__all__`` is exactly its five entries, each named after the module and each a stable API
    (no experimental marking)."""
    assert sorted(public.__all__) == list(ENTRIES)
    for name in public.__all__:
        fn = getattr(public, name)
        assert name.startswith("cake_lm_head_det_train_"), name
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name, name


def test_mode_word_is_a_token():
    """The token split finds the mode word only as a whole word."""
    assert _tokens("CakeLMHeadDetSupportFunction") == [
        "cake",
        "lm",
        "head",
        "det",
        "support",
        "function",
    ]
    assert _tokens("cake_lm_head_det_train_loss") == [
        "cake",
        "lm",
        "head",
        "det",
        "train",
        "loss",
    ]
    assert "det" not in _tokens("cake_lm_head_detail_loss")
    assert "det" not in _tokens("DetailFunction")


def test_public_names_carry_the_mode():
    """Every name of the facade's ``__all__`` and every backend object the facade forwards to carries the mode
    word ``det`` as a token (split on ``_`` and CamelCase); ``cake_support.__all__`` lists only those objects and
    ALL_CAPS constants."""
    count = 0
    for name in public.__all__:
        assert "det" in _tokens(name), name
        count += 1
    for module_name, names in FORWARDED.items():
        module = _backend(module_name)
        for name in names:
            obj = getattr(module, name)
            assert obj.__module__ == module.__name__, (module_name, name)
            assert "det" in _tokens(name), (module_name, name)
            count += 1
    assert count == 23
    support = _backend("cake_support")
    for name in support.__all__:
        assert name.isupper() or "det" in _tokens(name), name
    assert set(FORWARDED["cake_support"]) <= set(support.__all__)


def test_no_entry_takes_a_mode_keyword():
    """No facade entry, forwarded backend entry or autograd ``Function.forward`` of the backend accepts a
    ``deterministic`` keyword: ``inspect.signature(fn).bind_partial(deterministic=False)`` is a ``TypeError``
    (no such parameter, no ``**kwargs``); passing ``deterministic`` (``True`` or ``False``) to a facade entry is a
    ``TypeError`` from argument binding (host tensors suffice: it raises before any work)."""
    fns = {name: getattr(public, name) for name in ENTRIES}
    for module_name, names in FORWARDED.items():
        module = _backend(module_name)
        for name in names:
            obj = getattr(module, name)
            fns[f"{module_name}.{name}"] = obj.forward if inspect.isclass(obj) else obj
    assert len(fns) == 23
    classes = 0
    for module_name in ("cake_backend", "cake_backend_opt_in", "cake_support"):
        for name, cls in _autograd_functions(_backend(module_name)):
            fns[f"{module_name}.{name}.forward"] = cls.forward
            classes += 1
    # the opt-in module's two automatic-form log-probability subclasses and its one-chunk function among them
    assert classes == 11
    for label, fn in fns.items():
        try:
            inspect.signature(fn).bind_partial(deterministic=False)
        except TypeError:
            pass
        else:
            pytest.fail(f"{label} accepts a deterministic keyword")
        params = inspect.signature(fn).parameters
        assert "deterministic" not in params, label
        assert all(p.kind is not p.VAR_KEYWORD for p in params.values()), label
    X, W, labels = _host_call()
    sup = _support_args()
    calls = {
        "cake_lm_head_det_train_loss": ((X, W, labels), dict(loss_div=1.0)),
        "cake_lm_head_det_train_logprob": ((X, W, labels), {}),
        "cake_lm_head_det_train_preload": ((), {}),
        "cake_lm_head_det_train_support_logprob": (sup, dict(k_cap=3)),
        "cake_lm_head_det_train_support_kl": (sup + (torch.zeros(4),), dict(k_cap=3)),
    }
    for name, (args, kwargs) in calls.items():
        for value in (True, False):
            with pytest.raises(
                TypeError, match="unexpected keyword argument 'deterministic'"
            ):
                getattr(public, name)(*args, **kwargs, deterministic=value)


def test_backend_reads_no_torch_mode():
    """No function or autograd ``Function.forward`` of the backend modules takes a ``deterministic`` parameter,
    and none of them reads torch's deterministic mode."""
    seen = 0
    for name in ("cake_backend", "cake_backend_opt_in", "cake_support"):
        module = _backend(name)
        assert "are_deterministic_algorithms_enabled" not in inspect.getsource(module)
        for attr, value in vars(module).items():
            if getattr(value, "__module__", None) != module.__name__:
                continue
            if inspect.isclass(value) and issubclass(value, torch.autograd.Function):
                fn = value.forward
            elif inspect.isfunction(value):
                fn = value
            else:
                continue
            seen += 1
            assert "deterministic" not in inspect.signature(fn).parameters, (
                f"{name}.{attr}"
            )
    assert seen > 0


@pytest.mark.parametrize("warn_only", [False, True])
def test_torch_deterministic_mode_raises_no_alert(warn_only):
    """With ``torch.use_deterministic_algorithms(True)`` (also ``warn_only``), a host-level call through the
    backend's reference path computes the loss and both gradients with no alert and no warning."""
    inp = make_inputs(37, seed=20261106, H=H_HOST, V=V_HOST, device=torch.device("cpu"))
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    was = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", ExperimentalWarning)
            loss = _backend("cake_backend").cake_lm_head_det_train_loss(
                X,
                W,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=16,
                backend="reference",
            )
            loss.backward()
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    assert X.grad is not None and W.grad is not None


def test_package_identity_and_build_identity():
    """``build.json`` names the moved package; the program and library stems keep their build identity."""
    path = Path(_backend("cake_build").__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    assert json.loads(path.read_text())["package"] == BACKEND
    assert _backend("cake_jit").library_stem("x") == "cake_lm_head_det_loss_x"


def test_flashinfer_reexports_no_former_name():
    """In a fresh interpreter, ``flashinfer`` exports no name of the former deterministic facade or package."""
    code = (
        "import json\n"
        "import flashinfer\n"
        "names = set(dir(flashinfer)) | set(getattr(flashinfer, '__all__', ()))\n"
        "print(json.dumps(sorted(n for n in names if 'chunked_lm_head' in n or 'cake_lm_head_det_loss' in n)))\n"
    )
    assert _fresh(code) == []
