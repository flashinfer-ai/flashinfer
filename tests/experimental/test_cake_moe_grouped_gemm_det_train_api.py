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

flashinfer public-API tests of the deterministic Cake MoE grouped GEMM module.

The entry points of the module ``flashinfer.cake_moe_grouped_gemm_det_train``
(``cake_moe_grouped_gemm_det_train_{forward,dgrad,wgrad}``, the
``cake_moe_grouped_gemm_det_train_prepare_*`` builders and the autograd entry
``cake_moe_grouped_gemm_det``) and the entry point
``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` hand off to the backend
package ``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``.  This suite
covers the module's entry points on CPU -- stable APIs with no experimental marking and
no warning, the signatures and docstrings they share with the package functions, the
two types resolved from the package, the support query, and the deferred import that
keeps the package out of ``import`` of the module -- and checks the stable route on a GPU against the per-group fp64 reference (including the
rows past ``m_indptr[-1]``, which belong to no group) and against the options the Cake
backend does not take, and that a non-reentrant checkpoint of the module's
``cake_moe_grouped_gemm_det`` with early stop on skips the recomputed forward GEMM.
On CPU it also checks the deterministic-only contract: no entry takes a
``deterministic`` argument (``TypeError``, also a stale positional mode of
``CakeMoEGroupedGemmDetFunction.apply``), and torch's deterministic mode changes
nothing for them (no alert, no warning).  The removed names and paths are covered by
``test_cake_moe_grouped_gemm_det_train_names.py``.
The backend tests of the package live in ``test_cake_moe_grouped_gemm_det_train.py``,
which imports the package only through ``tests.cake.test_helpers.cake_package``.
"""

import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import flashinfer.api_logging as api_logging
import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as pkg
import flashinfer.cake_moe_grouped_gemm_det_train as facade
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
    cake_backend as cb,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ATOL = RTOL = 1e-2  # bf16 outputs

# Group size lists: an empty group, a one-row group and sizes that are not tile
# multiples (the same lists as the backend suite).
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]

# The entry points of flashinfer.cake_moe_grouped_gemm_det_train; the package
# functions they hand off to carry the same names.
ENTRY_POINTS = (
    "cake_moe_grouped_gemm_det_train_forward",
    "cake_moe_grouped_gemm_det_train_dgrad",
    "cake_moe_grouped_gemm_det_train_wgrad",
    "cake_moe_grouped_gemm_det_train_prepare_forward",
    "cake_moe_grouped_gemm_det_train_prepare_dgrad",
    "cake_moe_grouped_gemm_det_train_prepare_wgrad",
    "cake_moe_grouped_gemm_det",
)
# The types of the module, resolved from the package on first access: name -> submodule.
LAZY_TYPES = {
    "CakeMoEGroupedGemmDetFunction": "cake_autograd",
    "CakeMoEGroupedGemmDetLaunch": "cake_backend",
}


# ---------------------------------------------------------------------------
# The module's entry points (CPU only)
# ---------------------------------------------------------------------------


def _run_cpu_subprocess(code):
    """Run ``code`` in a fresh interpreter with no visible GPU; its last stdout line.

    A subprocess keeps the warning state and the set of imported modules
    independent of whatever this test session has already done.
    """
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc.stdout.strip().splitlines()[-1]


def test_facade_entry_points_are_stable_handoffs():
    """``flashinfer.cake_moe_grouped_gemm_det_train.<name>`` is a stable FlashInfer
    API (``@flashinfer_api``; no experimental marking) with the signature and the
    docstring of the package function of the same name, and is listed in the
    module's ``__all__``.  The two types of the module are the package's classes,
    resolved on first access."""
    assert "flashinfer_experimental_api" not in Path(facade.__file__).read_text()
    for name in ENTRY_POINTS:
        entry, impl = getattr(facade, name), getattr(pkg, name)
        assert name in facade.__all__, name
        assert entry is not impl, name
        assert not getattr(entry, "is_experimental", False), name
        assert not hasattr(entry, "experimental_feature"), name
        assert inspect.signature(entry) == inspect.signature(impl), name
        assert entry.__doc__ == impl.__doc__, name
        assert not hasattr(impl, "is_experimental"), name
    for name, submodule in LAZY_TYPES.items():
        assert name in facade.__all__ and name in dir(facade), name
        module = importlib.import_module(f"{pkg.__name__}.{submodule}")
        assert getattr(facade, name) is getattr(module, name) is getattr(pkg, name)


_NO_WARNING = """
import warnings

import torch

import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as pkg
import flashinfer.cake_moe_grouped_gemm_det_train as facade

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


# bfloat16 CPU operands pass the dtype and shape checks and fail the device
# check, so every call raises the host-side ValueError and touches no kernel.
offs = torch.tensor([2, 4], dtype=torch.int32)
operands = {
    "cake_moe_grouped_gemm_det_train_forward": (bf16(4, 64), bf16(2, 256, 64), offs),
    "cake_moe_grouped_gemm_det_train_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    "cake_moe_grouped_gemm_det_train_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_moe_grouped_gemm_det_train_prepare_forward": (
        bf16(4, 64),
        bf16(2, 256, 64),
        offs,
    ),
    "cake_moe_grouped_gemm_det_train_prepare_dgrad": (
        bf16(4, 64),
        bf16(2, 64, 256),
        offs,
    ),
    "cake_moe_grouped_gemm_det_train_prepare_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_moe_grouped_gemm_det": (bf16(4, 64), bf16(2, 256, 64), offs),
}


def warnings_of(fn, args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("CPU operands must be rejected on the host")
    return rec


for name in pkg.__all__:  # load the submodules outside the recorded calls
    getattr(pkg, name)

for name, args in operands.items():
    entry, impl = getattr(facade, name), getattr(pkg, name)
    for fn in (entry, entry, impl):  # the first call, a second call, the package
        rec = warnings_of(fn, args)
        assert rec == [], (name, [f"{w.category.__name__}: {w.message}" for w in rec])
print("OK")
"""


def test_facade_entry_points_do_not_warn():
    """Each entry point hands CPU operands to the package function, which raises the
    host-side ``ValueError``, with no warning: no ``ExperimentalWarning`` on the
    first call or later, and none from the package function.  Runs in a fresh
    interpreter, so each call there is the first of its entry point."""
    assert _run_cpu_subprocess(_NO_WARNING) == "OK"


def test_importing_the_facade_does_not_import_the_package():
    """``flashinfer.cake_moe_grouped_gemm_det_train`` reaches the package through
    deferred imports only (experimental containment): a fresh interpreter that
    imports it (and ``flashinfer.grouped_mm``) has the facade loaded and no module of
    ``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``."""
    code = (
        "import json, sys\n"
        "import flashinfer.grouped_mm\n"
        "import flashinfer.cake_moe_grouped_gemm_det_train\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if 'cake_moe_grouped' in m)))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == [
        "flashinfer.cake_moe_grouped_gemm_det_train"
    ]


_SUPPORT_OFF_CUDA = """
import json
import sys

import torch

import flashinfer.cake_moe_grouped_gemm_det_train as facade

assert not torch.cuda.is_available()
query = facade.is_cake_moe_grouped_gemm_det_train_supported
devices = (None, "cuda", torch.device("cuda", 0), "cpu")
answers = [query()] + [query(device) for device in devices]
loaded = sorted(m for m in sys.modules if m.startswith(facade._PACKAGE))
print(json.dumps([answers, loaded]))
"""


def test_support_query_is_false_off_cuda():
    """``is_cake_moe_grouped_gemm_det_train_supported`` is a stable FlashInfer API
    (``@flashinfer_api``; not marked experimental) that is ``False`` without CUDA
    (the default device, an explicit CUDA device, a CPU device) and then imports
    none of the package; a device that is not CUDA is ``False`` on any host."""
    answers, loaded = json.loads(_run_cpu_subprocess(_SUPPORT_OFF_CUDA))
    assert answers == [False] * 5 and loaded == []
    query = facade.is_cake_moe_grouped_gemm_det_train_supported
    for device in ("cpu", torch.device("cpu"), "meta"):
        assert query(device) is False, device
    assert not getattr(query, "is_experimental", False)
    assert "is_cake_moe_grouped_gemm_det_train_supported" in facade.__all__
    assert list(inspect.signature(query).parameters) == ["device"]
    assert inspect.signature(query).parameters["device"].default is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_support_query_on_this_device():
    """On a CUDA device the support query is the backend's program query of every
    operation (``cake_jit.OPS``), for the default device and an explicit index; an
    index past the visible devices is ``False``."""
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import cake_jit

    query = facade.is_cake_moe_grouped_gemm_det_train_supported
    device = torch.device("cuda", torch.cuda.current_device())
    want = all(cb.generated_program_available(device, op) for op in cake_jit.OPS)
    assert query() is want and query(device) is want
    assert query(torch.device("cuda", torch.cuda.device_count())) is False


# ---------------------------------------------------------------------------
# The deterministic-only contract (CPU)
# ---------------------------------------------------------------------------


def _cpu_operands():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    w = torch.zeros(2, 256, 64, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)
    return x, w, offs


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("entry", ["package", "facade"])
def test_det_wrapper_refuses_a_mode_argument(entry, value):
    """``cake_moe_grouped_gemm_det(..., deterministic=...)`` raises ``TypeError`` for either
    value (no silent acceptance of ``False``), as does a fourth positional argument, in
    the package and in the module ``flashinfer.cake_moe_grouped_gemm_det_train``."""
    fn = (
        pkg.cake_moe_grouped_gemm_det
        if entry == "package"
        else facade.cake_moe_grouped_gemm_det
    )
    x, w, offs = _cpu_operands()
    with pytest.raises(TypeError, match="deterministic"):
        fn(x, w, offs, deterministic=value)
    with pytest.raises(TypeError):
        fn(x, w, offs, value)


@pytest.mark.parametrize("op", ["forward", "dgrad"])
def test_det_functions_refuse_a_mode_keyword(op):
    """The one-shot functions and the prepared-launch builders of the package take no
    ``deterministic`` keyword either (``TypeError``)."""
    x, w, offs = _cpu_operands()
    prefix = "cake_moe_grouped_gemm_det_train"
    for fn in (getattr(pkg, f"{prefix}_{op}"), getattr(cb, f"{prefix}_prepare_{op}")):
        with pytest.raises(TypeError, match="deterministic"):
            fn(x, w, offs, deterministic=True)
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_moe_grouped_gemm_det_train_wgrad(x, x, offs, deterministic=False)
    with pytest.raises(TypeError, match="deterministic"):
        cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            x, x, offs, deterministic=False
        )


@pytest.mark.parametrize("stale", [True, False, 0, "deterministic"])
def test_direct_apply_refuses_a_stale_mode_argument(stale):
    """``CakeMoEGroupedGemmDetFunction.apply(x, w, offs, True)`` / ``(..., False)`` (the former
    numerical-mode argument) raises ``TypeError`` in ``forward`` before anything runs,
    instead of binding the value to the internal deferred-launch slot; five arguments
    exceed the arity."""
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_autograd,
    )

    x, w, offs = _cpu_operands()
    x.requires_grad_(True)
    with pytest.raises(TypeError, match="takes no fourth argument"):
        cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, stale)
    with pytest.raises(TypeError):
        cake_autograd.CakeMoEGroupedGemmDetFunction.apply(
            x, w, offs, stale, cake_autograd._DEFER_FWD
        )


def _with_main_grad(w):
    """``w`` requiring grad with a valid fp32 ``main_grad`` (the fuse flag's forward check passes)."""
    w.requires_grad_(True)
    w.main_grad = torch.zeros(w.shape, dtype=torch.float32)
    return w


_DET_MODE_CALLS = {
    "cake_moe_grouped_gemm_det_train_forward": lambda x,
    w,
    offs: pkg.cake_moe_grouped_gemm_det_train_forward(x, w, offs),
    "cake_moe_grouped_gemm_det_train_dgrad": lambda x,
    w,
    offs: pkg.cake_moe_grouped_gemm_det_train_dgrad(
        torch.zeros(4, 256, dtype=torch.bfloat16), w, offs
    ),
    "cake_moe_grouped_gemm_det_train_wgrad": lambda x,
    w,
    offs: pkg.cake_moe_grouped_gemm_det_train_wgrad(
        torch.zeros(4, 256, dtype=torch.bfloat16), x, offs, out_dtype=torch.float32
    ),
    "cake_moe_grouped_gemm_det_train_prepare_wgrad": lambda x, w, offs: (
        cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            torch.zeros(4, 256, dtype=torch.bfloat16), x, offs
        )
    ),
    "cake_moe_grouped_gemm_det": lambda x, w, offs: pkg.cake_moe_grouped_gemm_det(
        x, w.requires_grad_(True), offs
    ),
    "cake_moe_grouped_gemm_det_train_wgrad_accumulate": lambda x,
    w,
    offs: pkg.cake_moe_grouped_gemm_det_train_wgrad(
        torch.zeros(4, 256, dtype=torch.bfloat16),
        x,
        offs,
        out=torch.zeros(2, 256, 64),
        accumulate=True,
    ),
    "cake_moe_grouped_gemm_det_fuse": lambda x, w, offs: pkg.cake_moe_grouped_gemm_det(
        x, _with_main_grad(w), offs, fuse_wgrad_accumulation=True
    ),
}


@pytest.mark.parametrize("name", list(_DET_MODE_CALLS))
def test_torch_deterministic_mode_changes_nothing_for_det_entries(name):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) a
    deterministic entry raises no alert and no ``UserWarning``, also with
    ``accumulate=True`` / ``fuse_wgrad_accumulation=True``: CPU operands reach the same
    host-side ``ValueError`` (the device check) as with the mode off."""

    def outcome():
        x, w, offs = _cpu_operands()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as info:
                _DET_MODE_CALLS[name](x, w, offs)
        return str(info.value), [w for w in rec if w.category is UserWarning]

    want, warned = outcome()
    assert warned == []
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        for warn_only in (False, True):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            got, warned = outcome()
            assert got == want, (warn_only, got)
            assert warned == [], (warn_only, [str(w.message) for w in warned])
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


# ---------------------------------------------------------------------------
# The stable route grouped_mm_bf16(..., backend="cake") (GPU)
# ---------------------------------------------------------------------------


def _require_program(op, out_dtype=None):
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    if not cb.generated_program_available(device, op, out_dtype):
        pytest.skip(
            f"no generated ragged grouped GEMM program for {op} registered for {arch}"
        )
    return device


def _offs(sizes, device):
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def _inputs(sizes, n, k, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(len(sizes), n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(sum_m, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    return x, w, g, _offs(sizes, device)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return list(zip([0] + ends[:-1], ends, strict=True))


def _reference_fwd(x, w, sizes):
    """Per-group fp64 reference of the forward projection ``Y = X @ W[e].T`` in fp32."""
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32, device=x.device)
    for e, (s, t) in enumerate(_bounds(sizes)):
        if t > s:
            out[s:t] = (x[s:t].double() @ w[e].double().T).float()
    return out


def _assert_bf16_close(out, ref):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


def test_grouped_mm_bf16_cake_backend_matches_reference():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E6, 1024, 512, device)
    out = grouped_mm_bf16(x, w, _m_indptr(RAGGED_E6, device), backend="cake")
    # naming the backend is the opt-in: the (API, backend) pair has warned once
    assert ("grouped_mm_bf16", "cake") in api_logging._WARNED_EXPERIMENTAL_BACKENDS
    assert out.shape == (sum(RAGGED_E6), 1024) and out.dtype == torch.bfloat16
    _assert_bf16_close(out, _reference_fwd(x, w, RAGGED_E6))
    # rows past m_indptr[-1] belong to no group and are left untouched
    padded_x = torch.cat([x, torch.zeros(37, 512, dtype=x.dtype, device=device)])
    out = torch.full(
        (sum(RAGGED_E6) + 37, 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    grouped_mm_bf16(padded_x, w, _m_indptr(RAGGED_E6, device), out=out, backend="cake")
    _assert_bf16_close(out[: sum(RAGGED_E6)], _reference_fwd(x, w, RAGGED_E6))
    assert torch.isnan(out[sum(RAGGED_E6) :]).all()


def test_grouped_mm_bf16_cake_backend_rejects_unsupported_options():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E4, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E4, device)
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, out_dtype=torch.float32, backend="cake")
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, backend="cake", tactic=0)


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_facade_autograd_checkpoint_early_stop_skips_the_forward_launch(
    early_stop, monkeypatch
):
    """``flashinfer.cake_moe_grouped_gemm_det_train.cake_moe_grouped_gemm_det`` hands off to the package's wrapper,
    which launches its forward after autograd saved the inputs: a non-reentrant
    checkpoint of one GEMM launches the forward once with early stop on (the
    recomputation stops before it) and twice with it off, with the bits of the plain
    call for ``Y``, ``dX`` and ``dW``."""
    device = _require_program("fwd")
    _require_program("dgrad")
    _require_program("wgrad", torch.bfloat16)
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_autograd,
    )

    x0, w0, g, offs = _inputs(RAGGED_E4, 512, 256, device, seed=55)
    launches = []
    fwd_op = cake_autograd._fwd_op

    def counted(*args):
        launches.append(args[3].shape)
        fwd_op(*args)

    monkeypatch.setattr(cake_autograd, "_fwd_op", counted)

    def gemm(x, w):
        return facade.cake_moe_grouped_gemm_det(x, w, offs)

    def run(checkpointed):
        launches.clear()
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            if checkpointed:
                y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
            else:
                y = gemm(x, w)
            y.backward(g)
        torch.cuda.synchronize()
        return y.detach(), x.grad, w.grad, len(launches)

    *plain, n0 = run(False)
    *checkpointed, n1 = run(True)
    assert n0 == 1
    assert n1 == (1 if early_stop else 2)
    for a, b in zip(checkpointed, plain, strict=True):
        assert torch.equal(a, b)
    _assert_bf16_close(plain[0], _reference_fwd(x0, w0, RAGGED_E4))
