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

Tests of the public surface of the facade ``flashinfer.cake_gdn_nondet_train`` of the non-deterministic GDN
chunked-training package ``flashinfer.experimental.cake.cake_gdn_nondet_train`` (the package's own names are tested in
its backend suite ``test_cake_gdn_nondet_train.py``).  On the CPU: importing the facade does not import the package
before first use; the exact facade names, each with the word ``nondet``; the facade is a stable API (``@flashinfer_api``
entries, not marked experimental, a first call warns of nothing); the old names and the old module path gone; no
``deterministic`` keyword (passing one is a ``TypeError``); the support query returns a ``bool``.  On a device: the
facade runs unchanged under PyTorch's deterministic algorithms, strict and ``warn_only`` (no alert, no warning, no
fallback to the deterministic package), to the bits of the default mode.

The suite imports the facade and the old module path by name and the packages through the shared alias helper.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import math
import re
import subprocess
import sys
import warnings

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

FACADE = "flashinfer.cake_gdn_nondet_train"
PACKAGE = "flashinfer.experimental.cake.cake_gdn_nondet_train"
# The module path before the rename (FlashInfer #8928): nothing imports from it any more.
OLD_MODULE = "flashinfer.experimental.cake.cake_gdn_train"
HEAD_DIM = 128

# The facade's ``__all__``.  Every name carries the mode word ``nondet`` as a word of its own (see ``_name_tokens``:
# ``det`` is a substring of ``nondet``, so a substring check cannot tell the modes apart).
FACADE_NAMES = (
    "CakeGDNNondetFunction",
    "cake_gdn_nondet",
    "is_cake_gdn_nondet_train_supported",
)
# The names before the rename and the deterministic package's names: the facade holds none of them.
OLD_NAMES = (
    "chunk_gated_delta_rule",
    "ChunkGatedDeltaRuleFunction",
    "gdn_chunk_forward",
    "gdn_chunk_backward",
)
DET_NAMES = (
    "cake_gdn_det",
    "CakeGDNDetFunction",
    "cake_gdn_det_train_forward",
    "cake_gdn_det_train_backward",
)
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
# One word of a name part: an acronym run before a capitalized word, a capitalized or lower-case word, the rest of an
# acronym run, digits.
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _facade():
    return importlib.import_module(FACADE)


def _name_tokens(name: str) -> list[str]:
    """The lower-cased words of ``name``, split on ``_`` and on CamelCase boundaries, an acronym run as one word
    (``CakeGDNNondetFunction``: ``cake``, ``gdn``, ``nondet``, ``function``)."""
    return [word.lower() for part in name.split("_") for word in _WORD.findall(part)]


def _callables(module) -> dict:
    """The callables of ``module.__all__``, an autograd Function by its ``forward`` and ``backward`` (and by its
    ``apply`` when that signature names its parameters)."""
    var_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    found = {}
    for name in module.__all__:
        obj = getattr(module, name)
        if isinstance(obj, type):
            found[f"{name}.forward"] = obj.forward
            found[f"{name}.backward"] = obj.backward
            # ``torch.autograd.Function.apply`` takes ``*args, **kwargs``: its signature binds any keyword, so the
            # check reads ``forward``, the parameters ``apply`` passes on, and ``apply`` only if it names them.  Some
            # torch builds expose ``apply`` as a builtin without a signature (``ValueError``): it names nothing then
            try:
                params = list(inspect.signature(obj.apply).parameters.values())
            except ValueError:
                params = None
            if params is not None and all(p.kind not in var_kinds for p in params):
                found[f"{name}.apply"] = obj.apply
        else:
            found[name] = obj
    return found


# --------------------------------------------------------------------------- the names and the mode (no device)


def test_importing_the_facade_does_not_import_the_package() -> None:
    # a fresh interpreter: this module may already have imported the package
    code = (
        "import sys\n"
        f"import {FACADE} as facade\n"
        f"assert {PACKAGE!r} not in sys.modules, 'the facade import imported the package'\n"
        "assert 'CakeGDNNondetFunction' in dir(facade)\n"
        f"assert {PACKAGE!r} not in sys.modules, 'dir() imported the package'\n"
        "cls = facade.CakeGDNNondetFunction\n"
        f"assert {PACKAGE!r} in sys.modules\n"
        "print('ok', cls.__qualname__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok CakeGDNNondetFunction"


def test_the_facade_names_are_the_nondet_names() -> None:
    # the split itself: an acronym run is one word, and ``det`` is a word of a det name only
    for name, words in (
        ("CakeGDNNondetFunction", "cake gdn nondet function"),
        ("is_cake_gdn_nondet_train_supported", "is cake gdn nondet train supported"),
        ("CakeGDNDetFunction", "cake gdn det function"),
    ):
        assert _name_tokens(name) == words.split(), name
    facade = _facade()
    assert sorted(facade.__all__) == sorted(FACADE_NAMES)
    for name in facade.__all__:
        tokens = _name_tokens(name)
        assert "nondet" in tokens and "det" not in tokens, (name, tokens)
        assert getattr(facade, name) is not None, name
    # the facade resolves the package's Function; the entries keep their names
    cake_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
    assert facade.CakeGDNNondetFunction is cake_chain.CakeGDNNondetFunction
    cls = facade.CakeGDNNondetFunction
    assert cls.__name__ == cls.__qualname__ == "CakeGDNNondetFunction"
    entry = facade.cake_gdn_nondet
    assert entry.__name__ == entry.__qualname__ == "cake_gdn_nondet"


def test_the_facade_holds_no_old_or_det_name() -> None:
    facade = _facade()
    for name in OLD_NAMES + DET_NAMES:
        assert not hasattr(facade, name), name
        assert name not in facade.__all__, name
        assert name not in dir(facade), name


def test_the_old_module_path_does_not_import() -> None:
    # A clean tree has no module at the old path.  A leftover untracked directory there (``aot/`` or ``__pycache__``
    # from an earlier build) makes the path import as an empty namespace package: then it must hold no code, no
    # module of the package and none of the old names.
    spec = importlib.util.find_spec(OLD_MODULE)
    if spec is None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(OLD_MODULE)
        return
    assert spec.origin is None, spec  # a namespace package, not a package with code
    old = importlib.import_module(OLD_MODULE)
    for name in OLD_NAMES:
        assert not hasattr(old, name), name
    for sub in ("cake_chain", "cake_jit", "cake_backend", "cake_build"):
        assert importlib.util.find_spec(f"{OLD_MODULE}.{sub}") is None, sub


def test_the_facade_is_a_stable_api() -> None:
    # real ``@flashinfer_api`` defs of the facade module, not marked by the experimental-API decorator (it sets
    # ``is_experimental`` and ``experimental_feature`` and puts a ``.. warning::`` banner first in the docstring)
    facade = _facade()
    for name in ("cake_gdn_nondet", "is_cake_gdn_nondet_train_supported"):
        entry = getattr(facade, name)
        assert inspect.unwrap(entry).__module__ == FACADE, name
        assert not getattr(entry, "is_experimental", False), name
        assert not hasattr(entry, "experimental_feature"), name
        assert not entry.__doc__.lstrip().startswith(".. warning::"), name
    # a fresh interpreter, since the experimental-API decorator warns on the first call of a process only: the first
    # call, with a keyword the entry does not take, fails at its signature on host operands and warns of nothing
    code = (
        "import warnings\n"
        "import torch\n"
        f"import {FACADE} as facade\n"
        f"q = torch.zeros(1, 64, 1, {HEAD_DIM}, dtype=torch.bfloat16)\n"
        f"v = torch.zeros(1, 64, 2, {HEAD_DIM}, dtype=torch.bfloat16)\n"
        "g, beta = torch.zeros(1, 64, 2), torch.zeros(1, 64, 2)\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    try:\n"
        "        facade.cake_gdn_nondet(q, q, v, g, beta, deterministic=False)\n"
        "    except TypeError as exc:\n"
        "        assert 'deterministic' in str(exc), exc\n"
        "    else:\n"
        "        raise AssertionError('the entry took deterministic=')\n"
        "assert not caught, [(w.category.__name__, str(w.message)) for w in caught]\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_no_facade_entry_takes_a_deterministic_parameter() -> None:
    # every callable of the facade's ``__all__``: binding the keyword fails at the signature
    facade = _facade()
    callables = _callables(facade)
    assert {"cake_gdn_nondet", "CakeGDNNondetFunction.forward",
            "is_cake_gdn_nondet_train_supported"} <= set(callables)  # fmt: skip
    for name, fn in callables.items():
        try:
            inspect.signature(fn).bind_partial(deterministic=False)
        except TypeError as exc:
            assert "deterministic" in str(exc), (name, exc)
        else:
            pytest.fail(f"{name} binds deterministic=")
    # the facade entry copies the package entry's signature and forwards every argument
    cake_chain = cake_package("cake_gdn_nondet_train", "cake_chain")
    ours = inspect.signature(facade.cake_gdn_nondet).parameters
    theirs = inspect.signature(cake_chain.cake_gdn_nondet).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in theirs.values()
    ]


def test_a_deterministic_keyword_is_a_type_error() -> None:
    # host operands: the call fails at its signature, before any check or launch
    q = torch.zeros(1, 64, 1, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 64, 2, HEAD_DIM, dtype=torch.bfloat16)
    g, beta = torch.zeros(1, 64, 2), torch.zeros(1, 64, 2)
    entry = _facade().cake_gdn_nondet
    for value in (True, False):
        with pytest.raises(TypeError, match="deterministic"):
            entry(q, q, v, g, beta, deterministic=value)


def test_the_support_query_returns_a_bool() -> None:
    facade = _facade()
    got = facade.is_cake_gdn_nondet_train_supported()
    assert isinstance(got, bool)
    assert facade.is_cake_gdn_nondet_train_supported("cpu") is False
    if not torch.cuda.is_available():
        assert got is False
    else:
        cake_backend = cake_package("cake_gdn_nondet_train", "cake_backend")
        capability = tuple(torch.cuda.get_device_capability())
        assert got is (cake_backend.arch_for_capability(capability) is not None)


# --------------------------------------------------------------------------- the facade on a device


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cake_backend = cake_package("cake_gdn_nondet_train", "cake_backend")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"compute capability {capability} is not an exported target")
    return torch.device("cuda", index)


@contextlib.contextmanager
def _deterministic_algorithms(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def _inputs(seq_lens, heads, device, *, seed) -> dict:
    """Packed ``[1, T, ...]`` operands drawn on the host."""
    num_heads, num_v_heads = heads
    total = sum(seq_lens)
    gen = torch.Generator().manual_seed(seed)
    out = {
        "q": torch.randn(1, total, num_heads, HEAD_DIM, generator=gen).bfloat16(),
        "k": torch.randn(1, total, num_heads, HEAD_DIM, generator=gen).bfloat16(),
        "v": torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen).bfloat16(),
        "g": torch.rand(1, total, num_v_heads, generator=gen) * -0.5,
        "beta": torch.rand(1, total, num_v_heads, generator=gen).bfloat16(),
        "do": torch.randn(1, total, num_v_heads, HEAD_DIM, generator=gen).bfloat16(),
    }
    return {name: t.to(device).contiguous() for name, t in out.items()}


def _run(entry, inputs, seq_lens) -> dict:
    """Forward and backward through ``entry`` on the host-table route; returns the output and the input gradients."""
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + int(n))
    o, _ = entry(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(HEAD_DIM),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=inputs["q"].device),
        cu_seqlens_cpu=torch.tensor(cu, dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )
    (o.float() * inputs["do"].float()).sum().backward()
    torch.cuda.synchronize()
    return {
        "o": o.detach(),
        **{f"d{n}": leaves[n].grad for n in ("q", "k", "v", "g", "beta")},
    }


def test_the_facade_runs_unchanged_under_torch_deterministic_algorithms(
    monkeypatch,
) -> None:
    # a fixed-order non-deterministic entry under PyTorch's deterministic algorithms, strict and warn_only: the
    # facade's forward and backward run with no alert, no warning and no fallback, to the bits of the default mode
    device = _device_or_skip()

    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(
        cake_package("cake_gdn_det_train", "cake_backend"), "launch_stage", refuse
    )
    facade = _facade()
    seq_lens, heads = (300, 77, 640), (2, 6)
    inputs = _inputs(seq_lens, heads, device, seed=269)
    # the default mode first: the reference bits
    want = _run(facade.cake_gdn_nondet, inputs, seq_lens)
    for warn_only in (False, True):
        with (
            _deterministic_algorithms(True, warn_only=warn_only),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            got = _run(facade.cake_gdn_nondet, inputs, seq_lens)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert not user_warnings, (warn_only, [str(w.message) for w in user_warnings])
        for name in ("o",) + GRAD_NAMES:
            assert got[name] is not None, (warn_only, name)
            assert torch.equal(got[name], want[name]), (warn_only, name)
