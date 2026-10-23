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

flashinfer public-API tests of the Cake MoE grouped GEMM package.

The experimental entry points ``grouped_gemm_fwd``, ``grouped_gemm_dgrad``,
``grouped_gemm_wgrad`` and ``cake_grouped_mm`` of ``flashinfer.grouped_mm`` (the
thin core module ``flashinfer.grouped_mm.cake``) and the stable entry point
``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` hand off to
``flashinfer.experimental.cake_moe_grouped_gemm``.  This suite covers the core
entry points on CPU -- the experimental-API marking, the signatures and docstrings
they share with the package functions, the once-only ``ExperimentalWarning`` that
the package functions never emit, and the deferred import that keeps the package
out of ``import flashinfer.grouped_mm`` -- and checks the stable route on a GPU
against the per-group fp64 reference (including the rows past ``m_indptr[-1]``,
which belong to no group) and against the options the Cake backend does not take.
The backend tests of the package live in ``test_cake_moe_grouped_gemm.py``, which
imports the package only through ``tests.test_helpers.cake_package``.
"""

import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.api_logging as api_logging
import flashinfer.experimental.cake_moe_grouped_gemm as pkg
import flashinfer.grouped_mm as grouped_mm
from flashinfer.experimental.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ATOL = RTOL = 1e-2  # bf16 outputs

# Group size lists: an empty group, a one-row group and sizes that are not tile
# multiples (the same lists as the backend suite).
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]

# The experimental entry points of flashinfer.grouped_mm.cake; the package
# functions they hand off to carry the same names.
ENTRY_POINTS = (
    "grouped_gemm_fwd",
    "grouped_gemm_dgrad",
    "grouped_gemm_wgrad",
    "cake_grouped_mm",
)


# ---------------------------------------------------------------------------
# Core entry points (CPU only)
# ---------------------------------------------------------------------------


def _run_cpu_subprocess(code):
    """Run ``code`` in a fresh interpreter with no visible GPU; its last stdout line.

    A subprocess keeps the once-only warning flags and the set of imported
    modules independent of whatever this test session has already done.
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


def test_core_entry_points_are_the_experimental_api():
    """``flashinfer.grouped_mm.<name>`` is the experimental API of the package
    function of the same name: marked with the feature
    ``cake_moe_grouped_gemm.<name>``, carrying the package function's signature
    and its docstring behind the experimental banner, and exported by
    ``flashinfer.grouped_mm``; the package function itself is unmarked."""
    for name in ENTRY_POINTS:
        entry, impl = getattr(grouped_mm, name), getattr(pkg, name)
        assert name in grouped_mm.__all__, name
        assert entry.is_experimental is True, name
        assert entry.experimental_feature == f"cake_moe_grouped_gemm.{name}", name
        assert inspect.signature(entry) == inspect.signature(impl), name
        assert entry.__doc__.startswith(".. warning::"), name
        assert entry.__doc__.endswith(impl.__doc__), name
        assert not hasattr(impl, "is_experimental"), name


_WARN_ONCE = """
import warnings

import torch

import flashinfer.api_logging as api_logging
import flashinfer.experimental.cake_moe_grouped_gemm as pkg
import flashinfer.grouped_mm as grouped_mm

assert not torch.cuda.is_available()


def bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


# bfloat16 CPU operands pass the dtype and shape checks and fail the device
# check, so every call raises the host-side ValueError and touches no kernel.
offs = torch.tensor([2, 4], dtype=torch.int32)
operands = {
    "grouped_gemm_fwd": (bf16(4, 64), bf16(2, 256, 64), offs),
    "grouped_gemm_dgrad": (bf16(4, 64), bf16(2, 64, 256), offs),
    "grouped_gemm_wgrad": (bf16(4, 256), bf16(4, 64), offs),
    "cake_grouped_mm": (bf16(4, 64), bf16(2, 256, 64), offs),
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
    entry, impl = getattr(grouped_mm, name), getattr(pkg, name)
    [w] = warnings_of(entry, args)  # first call: exactly one warning
    assert w.category is api_logging.ExperimentalWarning, (name, w.category)
    message = str(w.message)
    assert f"'cake_moe_grouped_gemm.{name}' is an experimental" in message, message
    assert warnings_of(entry, args) == [], name  # second call: silent
    assert warnings_of(impl, args) == [], name  # the package function: never
print("OK")
"""


def test_core_entry_points_warn_once_and_the_package_functions_do_not():
    """Each entry point emits ``flashinfer.api_logging.ExperimentalWarning`` once,
    on its first call, before the handoff raises the host-side ``ValueError`` of
    CPU operands; a second call is silent, and the package functions never warn.
    Runs in a subprocess so the once-only flags do not depend on test order."""
    assert _run_cpu_subprocess(_WARN_ONCE) == "OK"


def test_importing_grouped_mm_does_not_import_the_package():
    """``flashinfer.grouped_mm`` reaches the package through deferred imports only
    (experimental containment): a fresh interpreter that imports it has the thin
    module ``flashinfer.grouped_mm.cake`` loaded and no module of
    ``flashinfer.experimental.cake_moe_grouped_gemm``."""
    code = (
        "import json, sys\n"
        "import flashinfer.grouped_mm\n"
        "print(json.dumps(sorted(m for m in sys.modules"
        " if m == 'flashinfer.grouped_mm.cake'"
        " or m.startswith('flashinfer.experimental.cake_moe_grouped_gemm'))))\n"
    )
    assert json.loads(_run_cpu_subprocess(code)) == ["flashinfer.grouped_mm.cake"]


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
