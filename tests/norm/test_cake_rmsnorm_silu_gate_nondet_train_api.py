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

Tests of the flashinfer public surface of the Cake SiLU-gated RMSNorm training
package with the non-deterministic weight gradient (``nondet``): the
``flashinfer.cake_rmsnorm_silu_gate_nondet_train`` facade, its re-export from
the ``flashinfer`` top level, and its interplay with
``torch.use_deterministic_algorithms``.

The mode is part of every name.  Every entry point of the facade and of the
standalone backend ``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``
runs only the non-deterministic kernels; no entry point takes a
``deterministic`` keyword (passing one is a ``TypeError``), and no call runs
the deterministic package ``flashinfer.cake_rmsnorm_silu_gate_det_train``.

Every output of the package repeats bit for bit, except a BF16 weight
gradient of the fused backward (``bwd_bf16_fused``, a band of
``BACKWARD_BANDS`` past the small-row bands on every architecture, on sm_107a
between two repeatable bands; an FP32 weight gradient runs no fused
backward) and of
the row-claims backwards
(``bwd_bf16_rc``, ``bwd_bf16_rc_mid``, where a band runs them), whose CTAs
claim 16-row blocks at run time and add FP64 atomics (grid kind
``row_claims``), so its last bits may vary from run to run; there every
launch's ``dw`` is checked against the budgets below instead of the first
launch's bits.  The fused backward (:func:`_fused`) runs the row arithmetic
of the deterministic package's BF16 backward program (``DET_FUSED_DIGEST``),
so its ``dx`` and ``dz`` are the deterministic package's bits at every grid
while that package routes that program on the device
(:func:`assert_det_bits_where_fused`); its ``dw`` need not be.  The
repeatable backward (``bwd_bf16_rep``, :func:`_rep`, the last BF16 band on
every architecture and on sm_107a also the band before the fused one) is that
program itself, so its ``dx``, ``dz`` and ``dw``
(accumulated too) are the deterministic package's bits.  In every other
band the backward writes ``dx`` and ``dz`` with this package's one row
arithmetic; in every band they are held to FP64 (within one BF16 ulp plus
1/64 ulp of the tensor's largest finite magnitude,
:func:`assert_dx_dz_within_fp64`) on every path checked here, and outside the
fused and the repeatable bands they need not be the deterministic package's
bits.  Nor is
``y``: the forward computes ``rstd`` and the SiLU gate with approximate FP32
intrinsics.  Nor, near rounding boundaries, is ``dw``: the chunked backwards
add the chunk partials of the weight gradient of the deterministic package
at FlashInfer ``2e989556`` exactly, as integers, and the fused and the
row-claims backwards add FP32 runs of rows in FP64.  So every output is also
compared with the deterministic package through an error budget against the
backend suite's FP64 reference (:func:`assert_error_within_det`):
on the same inputs, the relative L2 error of an output and its largest error
relative to the reference's largest magnitude are each at most
``DET_ERROR_RATIO`` (1.1) times the deterministic package's, and 0 where the
deterministic package's is 0; an FP32 weight gradient is held instead to the
deterministic package's within the backend suite's weight-gradient budget
(its 1.1x bound is against the deterministic package of FlashInfer
``2e989556``, whose FP32 ``dw`` this package's repeats bit for bit except
where the exact sum lies within that package's FP64 sum's error of a rounding
boundary; the in-tree deterministic package's FP32 ``dw`` need not be these
bits).  The deterministic outputs come from that
package's public forward and backward only.  This module checks:

* the surface without a device: the exports (one autograd function, the
  backend's class), the facade's sources (no import of the deterministic
  package), the signature of every entry point (no ``deterministic``
  parameter), the ``TypeError`` for a ``deterministic`` keyword of any value
  from every facade and backend entry point under each
  ``torch.use_deterministic_algorithms`` setting (nothing launched), that the
  facade runs the backend with the same arguments (the backend replaced by
  recorders), the autograd function in both directions (its backward naming
  itself for the alert), the support query and the
  workspace reset, the backend's standalone form (no import of another
  flashinfer package; a copy under another name imports with flashinfer
  blocked), and the strides the backend passes for one-head and other inputs
  (a single head's stride as 128, so packed one-head rows pass the kernels'
  packed-rows test);
* on a device: the alert of an autograd node whose backward runs the fused
  backward after ``torch.use_deterministic_algorithms(True)`` was
  enabled (exact texts: a ``RuntimeError`` before the launch, or a
  ``UserWarning`` under ``warn_only=True`` and the launch; none for a node
  whose weight needs no gradient or in any other band, the small-row bands
  among them); no call warns or
  raises in the bands whose outputs repeat their bits, strict or with
  ``warn_only=True`` (the forward, the backward into a BF16 and an FP32
  weight gradient, autograd and non-reentrant checkpointing); the
  ``TypeError`` for a mode keyword on real tensors before
  any launch; the entry points launch only the backend's modules; the
  non-deterministic backward refuses an FP32 workspace (the deterministic
  package's) naming both modes' workspaces, also in a call with no rows with
  the error of a call with rows, and the deterministic backward refuses the
  ``torch.uint8`` workspace of this package with its own FP32 workspace
  requirement; on the backend suite's accuracy matrix (weight-gradient terms
  that cancel per column included) and paths (accumulation,
  ``compute_grad_weight=False``, a reset
  workspace with nonzero bytes reused across shapes, two streams, a side
  stream, CUDA Graph replays, autograd and non-reentrant checkpointing) every
  output repeats bit for bit (an accumulated ``dw`` from the same starting
  value too) but a BF16 ``dw`` in the fused and the row-claims bands, ``dx``
  and ``dz`` lie within one BF16 ulp of FP64, ``y``, ``dx``, ``dz`` and
  ``dw`` lie within the error budget (``y`` also within one BF16 ulp of FP64,
  :func:`assert_y_within_fp64`), and ``dw`` within the weight-gradient budget of
  the deterministic ``dw`` (in the fused and the row-claims bands on every
  launch); in the fused band (the accumulation and
  ``compute_grad_weight=False`` paths also at its first rows, and grids
  forced to 1, 3 and 7 CTAs) ``dx`` and ``dz`` are the deterministic
  package's bits; so do
  they around the ends of the forward's ``FORWARD_BANDS`` bands (packed rows)
  and ``STRIDED_FORWARD_BANDS`` bands (other layouts) and at the first and
  the last rows of each ``BACKWARD_BANDS`` band of this device (the
  small-row backwards, one cluster each, the fused backward, the static
  backwards, the row-claims backwards and the claimed backward), each call
  launching ``module_for_rows``' module, and where a chunked backward's
  chunks grow by a block; so do strided inputs (``dy``, ``x`` and ``z`` not
  of one packed row stride) in the bands of the row-claims, the claimed and
  the fused backwards (``LAYOUT_BAND_CASES``), launching the strides of their
  layout; gate inputs at and past the packed
  gate's range (around FP32 ``expf``'s overflow too), in the forward and the
  backward, and infinite upstream gradients (in the last row too, with one
  row, half a block and one row short of a whole block in the last row block
  of the band's program) give the FP64 reference's
  ``NaN``, ``+inf`` and ``-inf`` positions in every band and finite elements
  within the error budget (in the fused band ``dx`` and ``dz`` the
  deterministic package's bits, ``NaN`` payloads included); one-head inputs
  of several layouts one row past the start of
  each module that warms rows in L2 and of the fused backward give the same
  bits for every layout (a
  BF16 ``dw`` of the fused and the row-claims backwards within the budgets
  instead),
  passing the packed-rows test exactly when their rows follow one another;
  and weight gradients in the subnormal range (identical rows whose FP32 terms
  underflow) lie within the weight-gradient budget widened by the underflow
  allowance the package documents (``dx`` and ``dz`` within one BF16 ulp of
  FP64, ``dx`` with an absolute floor below BF16's smallest subnormal).

The numerical contract of the non-deterministic kernels against the FP64
reference is ``test_cake_rmsnorm_silu_gate_nondet_train.py``'s; its data
generators, references and budgets are reused here.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy
import pytest
import torch
import torch.utils.checkpoint

import flashinfer
import flashinfer.cake_rmsnorm_silu_gate_det_train as deterministic_facade
import flashinfer.cake_rmsnorm_silu_gate_nondet_train as facade
import flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train as backend
from flashinfer.cake_rmsnorm_silu_gate_nondet_train import (
    CakeRMSNormSiLUGateNondetFunction,
    cake_rmsnorm_silu_gate_nondet,
    cake_rmsnorm_silu_gate_nondet_train_backward,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_nondet_train_forward,
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace,
    is_cake_rmsnorm_silu_gate_nondet_train_supported,
)
from flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train import (
    cake_jit as det_loader,
)
from flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train import (
    cake_jit as loader,
)
from tests.norm import test_cake_rmsnorm_silu_gate_nondet_train as suite

D = 128
EPS = 1e-6
# The entry points of the facade and of the backend package (the same names;
# the autograd function is the backend's class).
ENTRY_NAMES = (
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "cake_rmsnorm_silu_gate_nondet",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
)
# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))
MODE_IDS = ["true", "false", "none", "zero", "one", "string", "numpy_false"]
BACKEND_DIR = Path(backend.__file__).parent
# The bound of an output's errors against the FP64 reference, as a multiple of
# the deterministic package's errors on the same inputs.
DET_ERROR_RATIO = 1.1
# The non-deterministic backward's refusal of an FP32 workspace (the
# deterministic package's), naming both modes' workspace dtypes and factories
# (the backend suite asserts the same text).
REFUSAL = (
    "workspace is FP32, a workspace of the deterministic mode "
    "(flashinfer.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
    "non-deterministic backward needs its own zero-filled uint8 workspace "
    f"({backend.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
)

nd_forward = cake_rmsnorm_silu_gate_nondet_train_forward
nd_backward = cake_rmsnorm_silu_gate_nondet_train_backward
nd_workspace = cake_rmsnorm_silu_gate_nondet_train_backward_workspace
nd_workspace_bytes = cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
nd_reset = cake_rmsnorm_silu_gate_nondet_train_reset_workspace
nd_gated_rmsnorm = cake_rmsnorm_silu_gate_nondet
# The deterministic package, through its public forward and backward only.
det_forward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
det_backward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward


def assert_y_within_fp64(y, x, z, w, eps, *, what):
    """``y`` NaN where the FP64 forward is NaN, its infinity where it is
    infinite and elsewhere within one BF16 ulp of it plus a floor of 1/64 ulp of
    its row's largest finite ungated magnitude ``|x * rstd * w|``, times the
    gate's magnitude where that exceeds one.  The deterministic package's
    forward computes its gate and ``rstd`` with approximate instructions, so its
    ``y`` need not be this package's bits; each package holds its own ``y`` to
    FP64."""
    xd, zd, wd = (t.double().reshape(-1, D) for t in (x, z, w))
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    ungated = xd * rstd * wd
    scale = torch.where(ungated.isfinite(), ungated.abs(), 0.0).amax(
        dim=-1, keepdim=True
    )
    silu = zd * torch.sigmoid(zd)
    want = ungated * silu
    got = y.detach().double().reshape(-1, D)
    assert torch.equal(got.isnan(), want.isnan()), f"{what}: NaN elements"
    inf = want.isinf()
    assert torch.equal(got.isinf(), inf), f"{what}: infinite elements"
    assert torch.equal(got[inf], want[inf]), f"{what}: infinite elements"
    floor = scale * silu.abs().clamp_min(1.0) * suite.BF16_RTOL / 64
    err = (got - want).abs()
    bad = want.isfinite() & ~(err <= want.abs() * suite.BF16_RTOL + floor)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond one BF16 ulp "
        f"of FP64; worst err {err[bad].max().item():.3e}"
    )


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture
def determinism_setting():
    """Restore the caller's ``torch.use_deterministic_algorithms`` setting
    after the test; the test starts with it off."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _cpu_tensors():
    x = torch.zeros(4, 2, D, dtype=torch.bfloat16)
    w = torch.ones(D, dtype=torch.bfloat16)
    return x, w


def _inputs(T: int, H: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, z, dy = (
        torch.randn(T, H, D, device="cuda", generator=g).bfloat16() for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    return x, z, w, dy


def _stub_kernels(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the backend's entry points and its forward and backward bodies
    by recorders (no device needed)."""
    calls: list = []
    for name in (*ENTRY_NAMES, "_forward", "_backward"):
        monkeypatch.setattr(
            backend, name, lambda *a, _name=name, **k: calls.append((_name, a, k))
        )
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: calls.append(("run", name))
    )
    return calls


def _mode_calls(module, ws) -> dict:
    """One call of every entry point of ``module`` (the facade or the backend)
    on CPU operands, taking extra keywords: ``entry -> call``."""
    x, w = _cpu_tensors()
    return {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward(x, x, x, w, EPS, workspace=ws, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **k),
        "cake_rmsnorm_silu_gate_nondet": lambda **k: module.cake_rmsnorm_silu_gate_nondet(x, x, w, EPS, **k),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **k: module.is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0", **k),
    }  # fmt: skip


def fp64_errors(out, ref, mask=None) -> tuple[float, float]:
    """``(rel_l2, max_rel)`` of ``out`` against the FP64 reference ``ref``, in
    FP64, over the elements ``mask`` picks (all without one):
    ``||out - ref||_2 / ||ref||_2`` and ``max|out - ref| / max|ref|`` (a zero
    reference divides by 1)."""
    ref = ref.double().reshape(-1)
    out = out.double().reshape(-1)
    if mask is not None:
        mask = mask.reshape(-1)
        ref, out = ref[mask], out[mask]
    diff = out - ref
    l2 = ref.norm().item() or 1.0
    top = ref.abs().max().item() or 1.0
    return diff.norm().item() / l2, diff.abs().max().item() / top


def assert_error_within_det(what, got, det, ref, *, mask=None, mag=None) -> None:
    """The error budget against the deterministic package: ``got``'s
    ``rel_l2`` and ``max_rel`` against the FP64 reference ``ref``
    (:func:`fp64_errors`) are each at most ``DET_ERROR_RATIO`` times those of
    ``det``, the deterministic package's output on the same inputs, so 0
    where ``det``'s is 0.  An FP32 ``got`` (an FP32 weight gradient: every
    other output is BF16) is held instead within the backend suite's
    weight-gradient budget of ``det``, ``|got - det| <= |det| * 2^-7 + mag *
    2^-20`` (``suite.assert_dw_within``; ``mag`` the sum of the terms'
    magnitudes per column, which such a call must pass).  ``mask`` picks the
    elements compared."""
    if got.dtype == torch.float32:
        assert mag is not None, f"{what}: an FP32 weight gradient needs mag"
        if mask is not None:
            got, det, mag = got[mask], det[mask], mag[mask]
        suite.assert_dw_within(
            got, det, mag, what=f"{what} vs the deterministic package"
        )
        return
    errors = zip(
        ("rel_l2", "max_rel"),
        fp64_errors(got, ref, mask),
        fp64_errors(det, ref, mask),
        strict=True,
    )
    for metric, nondet_error, det_error in errors:
        ratio = nondet_error / det_error if det_error else float("inf")
        assert nondet_error <= DET_ERROR_RATIO * det_error, (
            f"{what}: {metric} vs FP64 {nondet_error:.6e}, the deterministic "
            f"package's {det_error:.6e}: ratio {ratio:.4f} > {DET_ERROR_RATIO}"
        )


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bits as integers (a ``NaN`` equals its own bits)."""
    ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.detach().view(ints[t.element_size()])


def assert_same_bits(what, got, want) -> None:
    """Run to run: ``got`` is ``want``'s bits."""
    assert got.dtype == want.dtype and got.shape == want.shape, (
        what,
        got.dtype,
        want.dtype,
        tuple(got.shape),
        tuple(want.shape),
    )
    assert torch.equal(_bits(got), _bits(want)), f"{what}: not the same bits"


def assert_same_classes(what, got, ref) -> None:
    """``got`` has the FP64 reference's classes: ``NaN`` exactly where it is
    ``NaN`` (any sign and payload), ``+inf`` and ``-inf`` exactly where it is."""
    got = got.detach().double().reshape(-1)
    ref = ref.double().reshape(-1)
    for name, cls in (
        ("NaN", torch.isnan),
        ("+inf", torch.isposinf),
        ("-inf", torch.isneginf),
    ):
        assert torch.equal(cls(got), cls(ref)), f"{what}: {name} positions vs FP64"


def assert_dx_dz_within_fp64(what, got, want, *, floor=0.0) -> None:
    """``got`` (``dx`` or ``dz``, or a sum of them) is ``NaN`` where the FP64
    reference ``want`` is ``NaN``, its infinity where it is infinite, and
    elsewhere within one BF16 ulp of it plus 1/64 ulp of its largest finite
    magnitude (cancellation in ``dx``), or ``floor`` where that is larger.
    The deterministic package's backward need not give this package's
    ``dx`` and ``dz`` bits, so each package holds its own to FP64."""
    got = got.detach().double().reshape(-1)
    want = want.double().reshape(-1)
    assert torch.equal(got.isnan(), want.isnan()), f"{what}: NaN elements"
    inf = want.isinf()
    assert torch.equal(got.isinf(), inf), f"{what}: infinite elements"
    assert torch.equal(got[inf], want[inf]), f"{what}: infinite elements"
    finite = want.isfinite()
    scale = want[finite].abs().max().item() if bool(finite.any()) else 0.0
    bound = want.abs() * suite.BF16_RTOL + max(scale * suite.BF16_RTOL / 64, floor)
    err = (got - want).abs()
    bad = finite & ~(err <= bound)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond one BF16 ulp "
        f"of FP64; worst err {err[bad].max().item():.3e}"
    )


def assert_dw_repeats_within_det(what, dw, first, det, ref, mag, *, varies) -> None:
    """``dw`` of a launch whose first launch gave ``first``: ``first``'s bits
    where its band repeats them; where it may vary in its last bits from run
    to run (``varies``: a BF16 ``dw`` of the fused or a row-claims backward,
    ``suite._varies``) within the error budget of the deterministic
    package's ``det`` against the FP64 reference ``ref``
    (:func:`assert_error_within_det`) and within the weight-gradient budget
    of ``det`` (``suite.assert_dw_within``, ``mag`` the sum of the terms'
    magnitudes) instead."""
    if not varies:
        assert_same_bits(what, dw, first)
        return
    assert dw.dtype == first.dtype and dw.shape == first.shape, what
    assert_error_within_det(what, dw, det, ref)
    suite.assert_dw_within(dw, det, mag, what=f"{what} vs the deterministic package")


# The name digest of the deterministic package's BF16 backward program whose
# row arithmetic the fused backward runs (the same build of the same row
# operations and block schedule; its weight gradient is summed exactly there
# and with FP64 atomics here).
DET_FUSED_DIGEST = "8a6cb16beff15b6ffbfe"
# The name digest of the deterministic package's look-ahead BF16 backward
# program (its long role, which it runs on packed rows from its
# ``min_rows_per_sm`` rows per SM on): the same build with the same bits, the
# program the repeatable backward's long role runs.
DET_LONG_DIGEST = "fef9d1cc2cc267219992"


def _fused(name: str) -> bool:
    """Whether the backward module ``name`` is the fused backward
    (``bwd_bf16_fused``, on every architecture; grid kind ``row_claims``):
    the row arithmetic of the deterministic package's BF16 backward program
    (``DET_FUSED_DIGEST``), so its ``dx`` and ``dz`` are that package's bits
    on the same inputs and device at every grid; its CTAs claim 16-row blocks
    and add the weight gradient with FP64 atomics, so its ``dw`` may vary in
    its last bits."""
    return loader.MODULES[name]["role"] == "bwd_bf16_fused"


def _rep(name: str) -> bool:
    """Whether the backward module ``name`` is the repeatable backward
    (``bwd_bf16_rep``, on every architecture; grid kind
    ``claimed_row_blocks``) or its long role (``bwd_bf16_rep_long``, on
    packed rows from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on): the
    deterministic package's BF16 backward program (``DET_FUSED_DIGEST``) or
    its look-ahead program of the same bits (``DET_LONG_DIGEST``) under this
    package's name, whose CTAs claim 16-row blocks and sum the weight
    gradient exactly in fixed point, so its ``dx``, ``dz`` and ``dw``
    (accumulated too) are the deterministic package's bits on the same
    inputs and device, and repeat them."""
    return loader.MODULES[name]["role"] in ("bwd_bf16_rep", "bwd_bf16_rep_long")


def _det_routes_same_program() -> bool:
    """Whether the deterministic package's BF16 backward on the current
    device is the program of ``DET_FUSED_DIGEST`` and its long role, where it
    routes one, the program of ``DET_LONG_DIGEST`` (the same bits); once that
    package ships another BF16 backward, the ``dx`` and ``dz`` bits of the
    two packages need not agree."""
    arch = loader.device_arch(torch.cuda.current_device())
    routes = det_loader.ROUTES[arch]
    long = routes.get("bwd_bf16" + det_loader.LONG, f"_{DET_LONG_DIGEST}")
    same = routes["bwd_bf16"].endswith(f"_{DET_FUSED_DIGEST}")
    return same and long.endswith(f"_{DET_LONG_DIGEST}")


def assert_det_bits_where_fused(name: str, got, det) -> None:
    """Where the backward module ``name`` is the fused or the repeatable
    backward (:func:`_fused`, :func:`_rep`) and the deterministic package
    routes the program whose row arithmetic it runs
    (:func:`_det_routes_same_program`), ``dx`` and ``dz`` of ``got`` (``dx``,
    ``dz`` and, when given, ``dw``) are the bits of the deterministic
    package's output of the same call in ``det``, and so is the repeatable
    backward's ``dw``; the fused backward's ``dw`` and the other backwards'
    outputs need not be that package's bits, so nothing more is checked."""
    if not (_fused(name) or _rep(name)) or not _det_routes_same_program():
        return
    assert len(got) == len(det) and 2 <= len(got) <= 3, (len(got), len(det))
    which = "repeatable" if _rep(name) else "fused"
    count = len(got) if _rep(name) else 2
    for what, g, d in zip(("dx", "dz", "dw")[:count], got, det, strict=False):
        assert_same_bits(
            f"{what} of the {which} backward vs the deterministic package", g, d
        )


def _assert_refusal(error: BaseException) -> None:
    """The refusal of an FP32 workspace is :data:`REFUSAL`, naming both
    modes' workspaces."""
    assert str(error) == REFUSAL, str(error)


# --------------------------------------------------------------------------- the surface (no device)


def test_top_level_exports_are_the_facade() -> None:
    assert flashinfer.cake_rmsnorm_silu_gate_nondet is cake_rmsnorm_silu_gate_nondet
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_forward
        is cake_rmsnorm_silu_gate_nondet_train_forward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_backward
        is cake_rmsnorm_silu_gate_nondet_train_backward
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
    )
    # one non-deterministic autograd function: the backend's class, reached
    # through the facade and the top level
    assert (
        CakeRMSNormSiLUGateNondetFunction is backend.CakeRMSNormSiLUGateNondetFunction
    )
    assert (
        flashinfer.CakeRMSNormSiLUGateNondetFunction
        is CakeRMSNormSiLUGateNondetFunction
    )
    # the det names stay the det package's objects
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det
        is deterministic_facade.cake_rmsnorm_silu_gate_det
    )
    assert (
        flashinfer.cake_rmsnorm_silu_gate_det_train_forward
        is deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
    )


def test_facade_imports_no_deterministic_package() -> None:
    """The facade never runs the deterministic package: its source imports
    nothing from it."""
    tree = ast.parse(Path(facade.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        assert not any("silu_gate_det" in name for name in names), (node.lineno, names)


@pytest.mark.parametrize("namespace", ["facade", "backend"])
def test_no_entry_point_takes_a_mode_keyword(namespace: str) -> None:
    """The mode is in the name (``nondet``): no entry point of the facade or
    the backend package has a ``deterministic`` parameter, and the autograd
    function takes no fifth input."""
    module = {"facade": facade, "backend": backend}[namespace]
    assert set(ENTRY_NAMES) | {"CakeRMSNormSiLUGateNondetFunction"} == set(
        module.__all__
    ) - {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"}
    for name in ENTRY_NAMES:
        parameters = inspect.signature(getattr(module, name)).parameters
        assert "deterministic" not in parameters, (namespace, name)
    assert list(
        inspect.signature(CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]


@pytest.mark.parametrize("value", MODES, ids=MODE_IDS)
def test_a_mode_keyword_is_a_type_error(
    value, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=<any value>`` is a ``TypeError`` from every facade and
    backend entry point before any call reaches a kernel, under each
    ``torch.use_deterministic_algorithms`` setting; so is a fifth input of the
    autograd function (the backend's bodies recorded: no device needed)."""
    x, w = _cpu_tensors()
    ws = torch.zeros(64, dtype=torch.uint8)
    launched: list = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for module in (facade, backend):
            for call in _mode_calls(module, ws).values():
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, x, w, EPS, value)
    assert launched == []
    assert not bool(ws.any())


def test_facade_runs_the_backend_with_the_same_arguments(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """Each facade entry point calls the backend entry point of the same name
    with the same arguments and nothing else (stubbed)."""
    calls = _stub_kernels(monkeypatch)
    x, w = _cpu_tensors()
    ws = torch.zeros(64, dtype=torch.uint8)
    cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, out=x)
    cake_rmsnorm_silu_gate_nondet_train_backward(
        x, x, x, w, EPS, workspace=ws, compute_grad_weight=False
    )
    cake_rmsnorm_silu_gate_nondet(x, x, w, EPS)
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0")
    names = [c[0] for c in calls]
    assert names == [
        "cake_rmsnorm_silu_gate_nondet_train_forward",
        "cake_rmsnorm_silu_gate_nondet_train_backward",
        "cake_rmsnorm_silu_gate_nondet",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        "is_cake_rmsnorm_silu_gate_nondet_train_supported",
    ], names
    assert calls[0][1:] == ((x, x, w, EPS), {"out": x})
    assert calls[1][1] == (x, x, x, w, EPS)
    assert calls[1][2] == {
        "grad_x": None,
        "grad_z": None,
        "grad_weight": None,
        "accumulate_grad_weight": False,
        "compute_grad_weight": False,
        "workspace": ws,
    }
    assert calls[2][1:] == ((x, x, w, EPS), {})
    assert calls[3][1:] == ((7, "cuda:0"), {})
    assert calls[4][1:] == ((7, "cuda:0"), {})
    assert calls[5][1:] == ((ws,), {})
    assert calls[6][1:] == (("cuda:0",), {})


def test_autograd_function_runs_the_backend_in_both_directions(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """:class:`CakeRMSNormSiLUGateNondetFunction` (the backend's class, also an
    attribute of the facade) runs the backend's forward and backward bodies;
    its backward names the autograd backward for the alert (``entry``).
    Neither stubbed body warns under ``torch.use_deterministic_algorithms(True)``
    (bodies stubbed: no device needed)."""
    calls: list = []

    def forward(x, z, w, eps, *, out):
        calls.append(("fwd", out))
        return x * 1

    def backward(dy, x, z, w, eps, **kwargs):
        calls.append(("bwd", kwargs))
        return dy * 1, dy * 1, w * 1

    monkeypatch.setattr(backend, "_forward", forward)
    monkeypatch.setattr(backend, "_backward", backward)
    x, w = _cpu_tensors()
    leaves = [t.clone().requires_grad_(True) for t in (x, x, w)]
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = CakeRMSNormSiLUGateNondetFunction.apply(*leaves, EPS)
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        y.backward(torch.ones_like(y))
    assert calls == [
        ("fwd", None),
        (
            "bwd",
            {
                "grad_x": None,
                "grad_z": None,
                "grad_weight": None,
                "accumulate_grad_weight": False,
                "compute_grad_weight": True,
                "workspace": None,
                "entry": "CakeRMSNormSiLUGateNondetFunction.backward",
            },
        ),
    ]


def test_support_query_ignores_torch_determinism(determinism_setting) -> None:
    torch.use_deterministic_algorithms(True)
    cpu = torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False
        assert backend.is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False


def test_reset_workspace_passes_through(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The facade's reset resets through the backend whatever
    ``torch.use_deterministic_algorithms`` says (a fill has no reduction); the
    backend refuses an FP32 buffer (checked on the device)."""
    resets: list = []
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        lambda workspace, **k: resets.append((workspace, k)),
    )
    ws = torch.ones(8, dtype=torch.uint8)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    assert resets == [(ws, {})] * 3


def test_backend_package_imports_no_other_flashinfer_package() -> None:
    """The backend package is standalone: no source file imports from a parent
    package or another flashinfer module, and every relative import names a
    module of the package.  The one exception is the shared loader
    ``cake_build.py``, whose JIT fallback imports ``flashinfer.jit`` inside a
    function when no prebuilt library exists."""
    sources = sorted(BACKEND_DIR.glob("*.py"))
    assert {p.name for p in sources} == {"__init__.py", "cake_build.py", "cake_jit.py"}
    jit_fallback = []
    for path in sources:
        text = path.read_text()
        assert "import_module(" not in text and "__import__(" not in text, path.name
        tree = ast.parse(text)
        in_function = {
            id(node)
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(function)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                names = (
                    [node.module.split(".")[0]]
                    if node.module
                    else [alias.name for alias in node.names]
                )
                for name in names:
                    assert (BACKEND_DIR / f"{name}.py").is_file(), (path.name, name)
                continue
            if isinstance(node, ast.ImportFrom):
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module == "flashinfer" or module.startswith("flashinfer."):
                    assert path.name == "cake_build.py" and id(node) in in_function, (
                        path.name,
                        node.lineno,
                        module,
                    )
                    assert module.startswith("flashinfer.jit."), (path.name, module)
                    jit_fallback.append(module)
    assert jit_fallback, "the cake_build JIT fallback import was not found"


_VENDORED_CHILD = r"""
import importlib, importlib.abc, json, sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError("flashinfer is blocked: " + name)
        return None


sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None
sys.path.insert(0, sys.argv[1])
import torch

package = importlib.import_module(sys.argv[2])
importlib.import_module(sys.argv[2] + ".cake_jit")
importlib.import_module(sys.argv[2] + ".cake_build")
x = torch.zeros(2, 128, dtype=torch.bfloat16)
out = {
    "supported_cpu": package.is_cake_rmsnorm_silu_gate_nondet_train_supported(
        torch.device("cpu")
    ),
    "all": sorted(package.__all__),
}
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0], deterministic=False)
except TypeError as error:
    out["mode_keyword"] = str(error)
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0])
except ValueError as error:
    out["cpu"] = str(error)
out["flashinfer"] = sorted(
    name for name, module in sys.modules.items()
    if module is not None and (name == "flashinfer" or name.startswith("flashinfer."))
)
print(json.dumps(out))
"""


def test_vendored_copy_imports_with_flashinfer_blocked(tmp_path: Path) -> None:
    """The backend package copied under another top-level name (the vendored
    gate's first step, without its sources) imports, answers its support query,
    takes no mode keyword and refuses a CPU tensor in an interpreter where every
    ``flashinfer`` import raises; no flashinfer module is loaded."""
    neutral = "vendored_cake_rmsnorm_silu_gate_nondet_train"
    shutil.copytree(
        BACKEND_DIR,
        tmp_path / neutral,
        ignore=shutil.ignore_patterns("aot", "__pycache__", "csrc", "build.json"),
    )
    run = subprocess.run(
        [sys.executable, "-c", _VENDORED_CHILD, str(tmp_path), neutral],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-8:])
    assert run.returncode == 0, tail
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["flashinfer"] == [], out["flashinfer"]
    assert out["supported_cpu"] is False
    assert out["all"] == sorted(backend.__all__)
    assert "deterministic" in out["mode_keyword"], out["mode_keyword"]
    assert "CUDA" in out["cpu"], out["cpu"]


# --------------------------------------------------------------------------- on the device


@requires_device
def test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """A node of :func:`cake_rmsnorm_silu_gate_nondet` is the backend's
    :class:`CakeRMSNormSiLUGateNondetFunction`.  When
    ``torch.use_deterministic_algorithms(True)`` is enabled between its forward
    and its backward, the backward of a node in the first rows of the fused
    band ``bwd_bf16_fused`` (``suite._band_shape``; every architecture)
    raises PyTorch's alert, naming the autograd backward and the
    deterministic autograd function, immediately before the launch of the
    fused backward, whose weight-gradient sum's order depends on scheduling;
    under ``warn_only=True`` it warns and runs it.  A node there whose weight
    needs no gradient, and a node in the first rows of every other band of a
    BF16 weight gradient on this device whose module repeats its bits
    (``suite._varies``: the small-row bands of ``BAND_ROLES``, the short
    backward on every architecture and the exact tiny backward on sm_100a
    and sm_107a, each summing on chip in a fixed order; a static or a claimed
    backward where a band runs one, adding its partials exactly), does not
    alert.  The launches are recorded and skipped (``run_module``
    stubbed)."""
    launched: list[str] = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    arch = loader.device_arch(torch.cuda.current_device())
    q = f"{backend.__name__}.CakeRMSNormSiLUGateNondetFunction.backward"
    d = f"{deterministic_facade.__name__}.CakeRMSNormSiLUGateDetFunction"
    alerting = suite._band_shape("bwd_bf16_fused")
    # every architecture runs the fused backward in a band of its own
    assert alerting is not None, arch
    bands = suite._device_bands("bwd_bf16")
    quiet_bands = [
        band
        for band, _, _ in bands
        if not suite._varies(loader.route_module(arch, band))
    ]
    quiet = [suite._band_shape(band) for band in quiet_bands]
    assert None not in quiet, (arch, quiet_bands, quiet)
    assert any(band in loader.BAND_ROLES for band in quiet_bands), quiet_bands

    def node(shape, weight_grad=True):
        x, z, w, _ = _inputs(*shape, 21)
        y = cake_rmsnorm_silu_gate_nondet(
            x.requires_grad_(True),
            z.requires_grad_(True),
            w.requires_grad_(weight_grad),
            EPS,
        )
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        return y

    def module(shape):
        return loader.module_for_rows(
            arch, "bwd_bf16", shape[0] * shape[1], _sm_count()
        )

    for band, shape in zip(quiet_bands, quiet, strict=True):
        assert module(shape) == loader.route_module(arch, band), (band, shape)
    assert module(alerting) == loader.route_module(arch, "bwd_bf16_fused")
    y = node(alerting)
    launched.clear()
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError) as info:
        y.backward(torch.ones_like(y))
    assert str(info.value) == (
        f"{q} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'. You can call {d} instead, "
        "or use the 'warn_only=True' option, if that's acceptable for your "
        "application."
    )
    assert launched == []
    torch.use_deterministic_algorithms(False)
    y = node(alerting)
    launched.clear()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        y.backward(torch.ones_like(y))
    assert [str(r.message) for r in record if backend.__name__ in str(r.message)] == [
        f"{q} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'; the "
        f"non-deterministic kernels run. {d} is the deterministic one."
    ]
    assert launched == [module(alerting)]
    no_alert = [(alerting, False)] + [(shape, True) for shape in quiet]
    for shape, weight_grad in no_alert:
        torch.use_deterministic_algorithms(False)
        y = node(shape, weight_grad)
        launched.clear()
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y.backward(torch.ones_like(y))
        assert launched == [module(shape)], (shape, weight_grad, launched)


@requires_device
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_call_alerts_under_deterministic_algorithms(
    warn_only, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or with
    ``warn_only=True``, every call in the bands whose outputs repeat their
    bits runs with no warning and no error: the forward and the backward
    into a BF16 and an FP32 weight gradient at the first rows of each
    ``BACKWARD_BANDS`` band and of each long role's packed rows of this
    device (``_band_modules``, ``_long_band_modules``; for a BF16
    weight gradient every band but the fused band, whose backward alerts:
    :func:`test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled`),
    and the autograd backward, with the weight gradient, of
    :func:`cake_rmsnorm_silu_gate_nondet`, of
    :class:`CakeRMSNormSiLUGateNondetFunction` and of a non-reentrant
    checkpoint at the most rows of those BF16 calls.  The functional calls
    launch ``module_for_rows``' modules."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    calls = []
    varying = set()
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        bands = _band_modules(arch, role, sms) + _long_band_modules(arch, role, sms)
        for band, name, first, _ in bands:
            if suite._varies(name):
                varying.add(name)
                continue
            assert loader.module_for_rows(arch, role, first, sms) == name, (role, band)
            calls.append((first, dtype, name))
    # the fused band of this device (a BF16 weight gradient on every
    # architecture; an FP32 weight gradient runs none) alerts
    routed = {loader.ROUTES[arch][role] for role in suite.FUSED_ROLES[arch]}
    assert routed <= varying, (arch, varying, routed)
    autograd_rows = max(rows for rows, dtype, _ in calls if dtype == torch.bfloat16)
    inputs = {rows: suite.make(rows, 1, layout="flat", seed=21) for rows, _, _ in calls}
    x, z, w, dy = inputs[autograd_rows]
    leaves = [[t.clone().requires_grad_(True) for t in (x, z, w)] for _ in range(3)]
    launched = _spy_launches(monkeypatch)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for rows, dtype, _ in calls:
            xr, zr, wr, dyr = inputs[rows]
            nd_forward(xr, zr, wr, EPS)
            nd_backward(
                dyr,
                xr,
                zr,
                wr,
                EPS,
                grad_weight=torch.empty(D, dtype=dtype, device=xr.device),
            )
        nd_gated_rmsnorm(*leaves[0], EPS).backward(dy)
        CakeRMSNormSiLUGateNondetFunction.apply(*leaves[1], EPS).backward(dy)
        torch.utils.checkpoint.checkpoint(
            nd_gated_rmsnorm, *leaves[2], EPS, use_reentrant=False
        ).backward(dy)
        torch.cuda.synchronize()
    assert [str(c.message) for c in caught] == []
    for leaf_set in leaves:
        assert all(t.grad is not None for t in leaf_set)
    want = []
    for rows, _, name in calls:
        want += [loader.module_for_rows(arch, "fwd", rows, sms), name]
    assert launched[: len(want)] == want, launched


@requires_device
def test_a_mode_keyword_is_a_type_error_on_the_device(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a device too, ``deterministic=<any value>`` is a ``TypeError`` from
    every facade entry point before any launch or change of the workspace; so
    is a fifth input of the autograd function.  The same calls without the
    keyword run."""
    launched = _spy_launches(monkeypatch)
    T, H = 129, 48
    x, z, w, dy = _inputs(T, H, 11)
    device = x.device
    ws = nd_workspace(T * H, device)
    runs = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, workspace=ws, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
    assert set(runs) == set(ENTRY_NAMES)
    for run in runs.values():
        for value in (True, False):
            with pytest.raises(TypeError, match="deterministic"):
                run(deterministic=value)
    for value in (True, False):
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launched == []
    torch.cuda.synchronize()
    assert not bool(ws.any())
    for run in runs.values():
        run()
    assert launched, "the entry points launched nothing"


@requires_device
def test_nondet_entry_points_run_this_package(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``nondet`` entry points launch the backend's forward and backward
    modules that ``module_for_rows`` picks: ``y``, ``dx``, ``dz`` and ``dw``
    repeat bit for bit (``dw`` where its band repeats it,
    :func:`assert_dw_repeats_within_det`) and lie within the error budget of
    the deterministic package's outputs, ``dx`` and ``dz`` within one BF16
    ulp of FP64, ``dw`` (of each call) also within one BF16 rounding plus the FP32
    accumulation budget of the deterministic ``dw``; where the backward is
    the fused backward (at these rows on every architecture) ``dx`` and
    ``dz`` are the deterministic package's bits.  The two workspace kinds
    do not substitute, and a refusal launches nothing: this backward refuses
    an FP32 tensor (the deterministic package's workspace), naming both modes'
    workspaces, and the deterministic backward refuses this package's
    ``torch.uint8`` workspace (``ValueError``)."""
    launched = _spy_launches(monkeypatch)
    T, H = 257, 48
    x, z, w, dy = _inputs(T, H, 12)
    y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS)
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    assert launched == [
        loader.module_for_rows(arch, "fwd", T * H, sms),
        loader.module_for_rows(arch, "bwd_bf16", T * H, sms),
    ], launched
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    assert_y_within_fp64(y, x, z, w, EPS, what="y")
    again = nd_backward(dy, x, z, w, EPS)
    for what, got, want in zip(("dx", "dz"), again[:2], (dx, dz), strict=True):
        assert_same_bits(what, got, want)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS)
    assert_dx_dz_within_fp64("dx", dx, dx64)
    assert_dx_dz_within_fp64("dz", dz, dz64)
    outputs = zip(
        ("y", "dx", "dz", "dw"),
        (y, dx, dz, dw),
        (det_forward(x, z, w, EPS), dx_d, dz_d, dw_d),
        (y64, dx64, dz64, dw64),
        strict=True,
    )
    for what, got, det, ref in outputs:
        assert_error_within_det(what, got, det, ref)
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + EPS)
    )
    mag = terms.abs().reshape(-1, D).sum(0)
    assert_dw_repeats_within_det(
        "dw", again[2], dw, dw_d, dw64, mag, varies=suite._varies(launched[1])
    )
    for got in (dw, again[2]):
        err = (got.double() - dw_d.double()).abs()
        assert (err <= dw_d.double().abs() * 2.0**-7 + mag * 2.0**-20).all()
    assert_det_bits_where_fused(launched[1], (dx, dz, dw), (dx_d, dz_d, dw_d))
    # an FP32 buffer (the deterministic package's workspace) is not a
    # workspace of the non-deterministic backward, nor the other way round
    launched.clear()
    fp32_workspace = torch.zeros(T * H, dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError) as info:
        cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, workspace=fp32_workspace
        )
    _assert_refusal(info.value)
    # the deterministic backward refuses this package's workspace, naming both
    # modes' workspaces
    with pytest.raises(
        ValueError,
        match=re.escape(
            "workspace is uint8, a workspace of the non-deterministic mode "
            "(flashinfer.cake_rmsnorm_silu_gate_nondet_train."
            "cake_rmsnorm_silu_gate_nondet_train_backward_workspace); the "
            "deterministic backward needs its own zero-filled FP32 workspace "
            "(flashinfer.cake_rmsnorm_silu_gate_det_train."
            "cake_rmsnorm_silu_gate_det_train_backward_workspace)"
        ),
    ):
        det_backward(dy, x, z, w, EPS, workspace=nd_workspace(T * H, x.device))
    assert launched == []


# Grids of the fused backward: the host's (``None``) and grids forced to 1, 3
# and 7 CTAs (``backward_ctas`` replaced), so that every CTA claims many 16-row
# blocks.
FUSED_GRID_CTAS = [None, 1, 3, 7]


@requires_device
@pytest.mark.parametrize("rows_at", ["first", "claims"])
@pytest.mark.parametrize("ctas", FUSED_GRID_CTAS, ids=["host", "1", "3", "7"])
def test_fused_backward_dx_dz_are_the_deterministic_bits_at_every_grid(
    ctas, rows_at, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fused backward (``bwd_bf16_fused``, every architecture) at the
    fewest whole tokens of 48 heads in its band (``rows_at="first"``) and at
    the first rows past one wave of its host grid as tokens of one head
    (``"claims"``: more 16-row blocks than 4 CTAs per SM, which the CTAs claim
    at run time), on the host's grid and on grids forced to 1, 3 and 7 CTAs:
    on every one of ``suite.REPEATS`` launches ``dx`` and ``dz`` are the bits
    of the deterministic package's BF16 backward of the same inputs (the
    program of ``DET_FUSED_DIGEST``, whose row arithmetic the fused backward
    runs; skipped once that package routes another program); ``dw`` lies
    within the backend suite's budget of the FP64 reference and of the
    deterministic ``dw``, and an accumulated ``dw`` within that of the FP64
    reference, with no launch held to another's bits (the FP64 additions
    follow the blocks each CTA claimed; the count of distinct ``dw`` bit
    patterns is printed); the workspace is zero after every launch."""
    if not _det_routes_same_program():
        pytest.skip("the deterministic package routes another BF16 backward")
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    found = suite._device_band("bwd_bf16_fused")
    assert found is not None, arch
    first, last = found
    T, H = (-(-first // 48), 48) if rows_at == "first" else (64 * sms + 1, 1)
    rows = T * H
    assert first <= rows and (last is None or rows <= last), (rows, found)
    name = loader.module_for_rows(arch, "bwd_bf16", rows, sms)
    assert _fused(name), (rows, name)
    x, z, w, dy = suite.make(T, H, seed=31)
    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS)
    dw64 = suite.fp64_reference(dy, x, z, w, EPS)[3]
    mag = suite.magnitudes(dy, x, z, EPS)
    ws = nd_workspace(rows, x.device)
    if ctas is not None:
        monkeypatch.setattr(backend, "backward_ctas", lambda *_args: ctas)
    launched = _spy_launches(monkeypatch)
    patterns = set()
    for k in range(suite.REPEATS):
        dx, dz, dw = nd_backward(dy, x, z, w, EPS, workspace=ws)
        what = f"launch {k} at {ctas or 'the host'} CTAs"
        assert_same_bits(f"dx {what} vs the deterministic package", dx, dx_d)
        assert_same_bits(f"dz {what} vs the deterministic package", dz, dz_d)
        suite.assert_dw_within(dw, dw64, mag, what=f"dw {what} vs FP64")
        suite.assert_dw_within(
            dw, dw_d, mag, what=f"dw {what} vs the deterministic package"
        )
        main = torch.full((D,), 0.25, dtype=torch.bfloat16, device=x.device)
        nd_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        suite.assert_dw_within(
            main, 0.25 + dw64, mag, what=f"accumulated dw {what} vs FP64"
        )
        torch.cuda.synchronize()
        assert not bool(ws.any()), f"{what}: the workspace is not zero"
        patterns.add(tuple(_bits(dw).tolist()))
    print(
        f"FUSED_GRID arch={arch} sms={sms} rows={rows} ctas={ctas or 'host'} "
        f"launches={suite.REPEATS} distinct_dw={len(patterns)}"
    )
    assert launched == [name] * (2 * suite.REPEATS), launched


@requires_device
@pytest.mark.parametrize("shape", [(0, D), (0, 48, D)], ids=["0x128", "0x48x128"])
def test_backward_with_no_rows_refuses_the_deterministic_workspace(
    shape: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call with no rows refuses an FP32 tensor (the deterministic package's
    workspace) with the ``ValueError`` of a call with rows, naming both modes'
    workspaces, before it returns (``grad_weight`` keeps its value).  With
    this package's ``torch.uint8`` workspace it returns zero-sized ``dx`` and
    ``dz`` and a zero ``dw``, or keeps ``grad_weight`` when accumulating;
    nothing is launched and the workspace stays zero."""
    launched = _spy_launches(monkeypatch)
    empty = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    x, z, w, dy = _inputs(65, 48, 14)
    fp32_workspace = torch.zeros(65 * 48, dtype=torch.float32, device=x.device)
    ws = nd_workspace(0, x.device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(0, x.device)
    for module in (facade, backend):
        backward = module.cake_rmsnorm_silu_gate_nondet_train_backward
        with pytest.raises(ValueError) as with_rows:
            backward(dy, x, z, w, EPS, workspace=fp32_workspace)
        _assert_refusal(with_rows.value)
        main = torch.full((D,), 1.5, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError) as no_rows:
            backward(
                empty,
                empty,
                empty,
                w,
                EPS,
                grad_weight=main,
                workspace=fp32_workspace,
            )
        assert str(no_rows.value) == str(with_rows.value), str(no_rows.value)
        assert torch.equal(main, torch.full_like(main, 1.5))
        dx, dz, dw = backward(empty, empty, empty, w, EPS, workspace=ws)
        assert dx.shape == dz.shape == empty.shape and dx.numel() == dz.numel() == 0
        assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
        _, _, dw = backward(
            empty,
            empty,
            empty,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5))
        _, _, dw = backward(empty, empty, empty, w, EPS, grad_weight=main, workspace=ws)
        assert dw is main and torch.equal(main, torch.zeros_like(main))
    torch.cuda.synchronize()
    assert launched == []
    assert not bool(ws.any())


# --------------------------------------------------------------------------- within the deterministic package's error

# The backend suite's accuracy matrix: (T, H), the generator's keywords, eps and
# the weight-gradient dtype.
MATRIX = [
    *(
        pytest.param((T, H), {}, EPS, torch.bfloat16, id=f"shape-{T}x{H}")
        for T, H in suite.SHAPES
    ),
    *(
        pytest.param((65, 48), {"layout": v, "seed": 1}, EPS, torch.bfloat16, id=v)
        for v in suite.LAYOUTS
    ),
    *(
        pytest.param((37, 48), {"data": v, "seed": 2}, EPS, torch.bfloat16, id=v)
        for v in suite.EDGE_DATA
    ),
    *(
        pytest.param((19, 48), {"seed": 3}, eps, torch.bfloat16, id=f"eps-{eps}")
        for eps in suite.EPSILONS
    ),
    *(
        pytest.param((T, 48), {"seed": 4}, EPS, torch.bfloat16, id=f"issue-{T}")
        for T in suite.ISSUE_TOKENS
    ),
    *(
        pytest.param((T, H), {"seed": 5}, EPS, torch.float32, id=f"fp32-{T}x{H}")
        for T, H in suite.FP32_SHAPES
    ),
]


@requires_device
@pytest.mark.parametrize("shape,make_kwargs,eps,dw_dtype", MATRIX)
def test_accuracy_matrix_within_the_deterministic_error(
    shape, make_kwargs, eps, dw_dtype, determinism_setting
) -> None:
    """``y`` repeats bit for bit and, on every one of ``suite.REPEATS``
    backward launches, ``dx``, ``dz`` and ``dw`` are the first launch's bits
    (``dw`` where its band repeats it; in the fused and the row-claims bands
    each launch's ``dw`` lies within the budgets below instead,
    :func:`assert_dw_repeats_within_det`), ``dx`` and ``dz`` within one BF16
    ulp of FP64, and in the fused band ``dx`` and ``dz`` the
    deterministic package's bits (:func:`assert_det_bits_where_fused`); each
    output lies within
    the error budget of the deterministic package's output against the FP64
    reference (``y`` also within one BF16 ulp of FP64), and ``dw`` within the
    weight-gradient budget of the deterministic ``dw``."""
    x, z, w, dy = suite.make(*shape, **make_kwargs)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, eps)
    y = nd_forward(x, z, w, eps)
    assert_same_bits("y", nd_forward(x, z, w, eps), y)
    assert_error_within_det("y", y, det_forward(x, z, w, eps), y64)
    assert_y_within_fp64(y, x, z, w, eps, what="y")
    mag = suite.magnitudes(dy, x, z, eps)
    varies = suite._dw_varies(x.numel() // D, dw_dtype)

    def out():
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    dets = det_backward(dy, x, z, w, eps, grad_weight=out())
    first = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, eps, grad_weight=out())
        assert got[2].dtype == dw_dtype
        if first is None:
            first = got
        for what, g, f in zip(("dx", "dz"), got[:2], first[:2], strict=True):
            assert_same_bits(what, g, f)
        assert_dw_repeats_within_det(
            "dw", got[2], first[2], dets[2], dw64, mag, varies=varies
        )
        assert_dx_dz_within_fp64("dx", got[0], dx64)
        assert_dx_dz_within_fp64("dz", got[1], dz64)
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref, mag=mag)
    suite.assert_dw_within(
        first[2], dets[2], mag, what="dw vs the deterministic package"
    )
    assert_det_bits_where_fused(
        suite._backward_module(x.numel() // D, dw_dtype), first, dets
    )


@requires_device
@pytest.mark.parametrize(
    "band",
    [None, "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
    ids=["1031x48", "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
)
def test_accumulation_within_the_deterministic_error(band, determinism_setting) -> None:
    """Two microbatches of ``T x H`` rows (1031 tokens of 48 heads; for
    ``band`` the fewest whole tokens of 48 heads in that BF16 band, the
    fused, the repeatable or the long role's packed rows,
    ``suite._band_shape``, on every architecture; an FP32 weight gradient
    runs its own band there) accumulated into an FP32 and a BF16 buffer: the
    same starting buffer gives the same bits (a BF16 buffer where its band
    repeats a BF16 ``dw``; in the fused and the row-claims bands each buffer
    instead meets the checks that follow), within two roundings of the
    deterministic accumulation and within the error budget of the
    deterministic package's against the FP64 sum.  Autograd accumulation into
    ``.grad`` over two microbatches gives the sum of the two nodes' weight
    gradients (recorded by a hook), each the functional backward's bits (in
    the fused and the row-claims bands each within the budgets of
    :func:`assert_dw_repeats_within_det` instead), and twice the functional
    backward's ``dx`` and ``dz`` bits, ``x.grad`` and ``z.grad`` within one
    BF16 ulp of twice the FP64 ``dx`` and ``dz`` and within the error budget
    of twice the deterministic package's; in the fused band the functional
    backward's ``dx`` and ``dz`` are the deterministic package's bits."""
    shape = (1031, 48) if band is None else suite._band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    x, z, w, dy = suite.make(T, H, seed=6)
    x2, z2, _, dy2 = suite.make(T, H, seed=7)
    rows = T * H
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    dw64b = suite.fp64_reference(dy2, x2, z2, w, EPS)[3]
    mag = suite.magnitudes(dy, x, z, EPS) + suite.magnitudes(dy2, x2, z2, EPS)
    first = 0.25 + dw64
    want = first + dw64b
    # Two roundings into the accumulator's dtype, each within its relative unit.
    bound = (first.abs() + want.abs()) * suite.BF16_RTOL + mag * suite.DW_MAG_TOL
    for dtype in (torch.float32, torch.bfloat16):
        varies = suite._dw_varies(rows, dtype)
        mains = []
        for _ in range(2):
            main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
            for args in ((dy, x, z), (dy2, x2, z2)):
                nd_backward(
                    *args, w, EPS, grad_weight=main, accumulate_grad_weight=True
                )
            mains.append(main)
        if not varies:
            assert_same_bits(f"{dtype} accumulator", mains[1], mains[0])
        main_reference = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        for args in ((dy, x, z), (dy2, x2, z2)):
            det_backward(
                *args, w, EPS, grad_weight=main_reference, accumulate_grad_weight=True
            )
        for main in mains if varies else mains[:1]:
            assert (
                (main.double() - main_reference.double()).abs() <= 2 * bound
            ).all(), dtype
            assert_error_within_det(
                f"{dtype} accumulator", main, main_reference, want, mag=mag
            )
        module = suite._backward_module(rows, dtype)
        if _rep(module) and _det_routes_same_program():
            assert_same_bits(
                f"{dtype} accumulator of the repeatable backward vs the "
                "deterministic package",
                mains[0],
                main_reference,
            )
    # autograd accumulation into .grad over two microbatches
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    node_dws: list[torch.Tensor] = []
    wa.register_hook(lambda grad: node_dws.append(grad.detach().clone()))
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    assert len(node_dws) == 2, len(node_dws)
    dx, dz, dw = nd_backward(dy, x, z, w, EPS)
    assert_same_bits("x.grad", xa.grad, dx + dx)
    assert_same_bits("z.grad", za.grad, dz + dz)
    assert_same_bits("w.grad", wa.grad, node_dws[0] + node_dws[1])
    dx_d, dz_d, dw_d = det_backward(dy, x, z, w, EPS)
    assert_det_bits_where_fused(
        suite._backward_module(rows), (dx, dz, dw), (dx_d, dz_d, dw_d)
    )
    for k, node_dw in enumerate(node_dws):
        assert_dw_repeats_within_det(
            f"node {k}: dw",
            node_dw,
            dw,
            dw_d,
            dw64,
            suite.magnitudes(dy, x, z, EPS),
            varies=suite._dw_varies(rows),
        )
    assert_dx_dz_within_fp64("x.grad", xa.grad, 2 * dx64)
    assert_dx_dz_within_fp64("z.grad", za.grad, 2 * dz64)
    assert_error_within_det("x.grad", xa.grad, dx_d + dx_d, 2 * dx64)
    assert_error_within_det("z.grad", za.grad, dz_d + dz_d, 2 * dz64)


@requires_device
@pytest.mark.parametrize(
    "band",
    [None, "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
    ids=["257x48", "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
)
def test_compute_grad_weight_false_within_the_deterministic_error(
    band,
    determinism_setting,
) -> None:
    """Without the weight gradient (with this package's workspace or none,
    and through autograd with a weight that needs no gradient), ``dx`` and
    ``dz`` are the bits of a backward with the weight gradient, within one
    BF16 ulp of FP64 and within the deterministic package's error budget, at
    257 tokens of 48 heads and, for ``band``, at the fewest whole tokens of 48
    heads in that BF16 band, the fused, the repeatable or the long role's
    packed rows (``suite._band_shape``; a call without the
    weight gradient takes the BF16 bands; every architecture); where the
    backward is the fused backward they are the deterministic package's
    bits."""
    shape = (257, 48) if band is None else suite._band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    x, z, w, dy = suite.make(T, H, seed=12)
    _, dx64, dz64, _ = suite.fp64_reference(dy, x, z, w, EPS)
    dx_d, dz_d, _ = det_backward(dy, x, z, w, EPS, compute_grad_weight=False)
    want_dx, want_dz, _ = nd_backward(dy, x, z, w, EPS)
    assert_det_bits_where_fused(
        suite._backward_module(T * H), (want_dx, want_dz), (dx_d, dz_d)
    )
    ws = nd_workspace(T * H, x.device)
    for workspace in (ws, None):
        dx, dz, dw = nd_backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
        )
        assert dw is None
        assert_same_bits("dx", dx, want_dx)
        assert_same_bits("dz", dz, want_dz)
        assert_dx_dz_within_fp64("dx", dx, dx64)
        assert_dx_dz_within_fp64("dz", dz, dz64)
    assert_error_within_det("dx", want_dx, dx_d, dx64)
    assert_error_within_det("dz", want_dz, dz_d, dz64)
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    nd_gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert_same_bits("x.grad", xa.grad, want_dx)
    assert_same_bits("z.grad", za.grad, want_dz)
    assert_dx_dz_within_fp64("x.grad", xa.grad, dx64)
    assert_dx_dz_within_fp64("z.grad", za.grad, dz64)


@requires_device
def test_reset_workspace_then_reuse_across_shapes(determinism_setting) -> None:
    """A workspace of the architecture's bytes (``suite.WORKSPACE_BYTES``)
    with nonzero bytes in every buffer (the integer windows, the FP64 column
    sums, the class words, the arrival ticket and the claim counter) zeroed by
    ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, then reused
    without a reset over the backend suite's sequence of shapes, then the
    first rows of each ``BACKWARD_BANDS`` band and of each long role's packed
    rows of this device none of them runs (``suite._range_shape``; so every
    banded backward program of the architecture among them, the fused
    backward and the repeatable backward's long role too), weight-gradient
    dtypes and accumulation: every
    output repeats bit for bit (an accumulated ``dw`` from the same starting
    value too; a BF16 ``dw`` in the fused and the row-claims bands instead
    meets the checks that follow on each call), ``dx`` and
    ``dz`` lie within one BF16 ulp of FP64, ``dx``, ``dz`` and ``dw`` lie
    within the error budget of the deterministic package's,
    ``dw`` within the budget of the deterministic ``dw`` and of the FP64
    reference, in the fused band ``dx`` and ``dz`` are the deterministic
    package's bits, and the workspace is zero after every call."""
    device = torch.device("cuda")
    ws = nd_workspace(1, device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(1, device)
    assert nd_workspace_bytes(1, device) == nd_workspace_bytes(10**7, device)
    arch = loader.device_arch(torch.cuda.current_device())
    assert ws.numel() == suite.WORKSPACE_BYTES[arch], (arch, ws.numel())
    generator = torch.Generator(device="cuda").manual_seed(70)
    ws.copy_(
        torch.randint(
            1, 256, ws.shape, dtype=torch.uint8, device=device, generator=generator
        )
    )
    assert bool(ws.ne(0).all())
    nd_reset(ws)
    torch.cuda.synchronize()
    assert not bool(ws.any())
    cases = [
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
        (129, 48, torch.float32, False),
    ]
    sms = _sm_count()
    ran = {
        (_role_of(dtype), _band_of(arch, _role_of(dtype), T * H, sms))
        for T, H, dtype, _ in cases
    }
    for dtype in (torch.bfloat16, torch.float32):
        role = _role_of(dtype)
        bands = _band_modules(arch, role, sms) + _long_band_modules(arch, role, sms)
        for band, _, first, last in bands:
            if (role, (band, first)) not in ran:
                T, H = suite._range_shape(first, last)
                cases.append((T, H, dtype, False))
    for i, (T, H, dtype, accumulate) in enumerate(cases):
        x, z, w, dy = suite.make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0
        case = f"T={T} H={H} {dtype}"
        varies = suite._dw_varies(T * H, dtype)

        def gw(start=start, dtype=dtype, accumulate=accumulate):
            if accumulate or dtype != torch.bfloat16:
                return torch.full((D,), start, dtype=dtype, device=device)
            return None

        runs = [
            nd_backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=gw(),
                accumulate_grad_weight=accumulate,
                workspace=ws,
            )
            for _ in range(2)
        ]
        torch.cuda.synchronize()
        assert not bool(ws.any()), case
        for what, first, again in zip(("dx", "dz", "dw"), *runs, strict=True):
            if what != "dw" or not varies:
                assert_same_bits(f"{what} {case}", again, first)
        dx, dz, dw = runs[0]
        assert dw.dtype == dtype
        dets = det_backward(
            dy, x, z, w, EPS, grad_weight=gw(), accumulate_grad_weight=accumulate
        )
        mag = suite.magnitudes(dy, x, z, EPS)
        _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
        assert_dx_dz_within_fp64(f"dx {case}", dx, dx64)
        assert_dx_dz_within_fp64(f"dz {case}", dz, dz64)
        want = dw64 + start
        for what, got, det, ref in zip(
            ("dx", "dz"), (dx, dz), dets[:2], (dx64, dz64), strict=True
        ):
            assert_error_within_det(f"{what} {case}", got, det, ref)
        assert_det_bits_where_fused(
            loader.module_for_rows(arch, _role_of(dtype), T * H, sms), runs[0], dets
        )
        for _, _, dw in runs if varies else runs[:1]:
            assert dw.dtype == dtype
            assert_error_within_det(f"dw {case}", dw, dets[2], want, mag=mag)
            suite.assert_dw_within(dw, want, mag, what=f"dw {case} vs FP64")
            suite.assert_dw_within(
                dw, dets[2], mag, what=f"dw {case} vs the deterministic package"
            )


@requires_device
def test_two_streams_repeat_the_current_stream_bits(determinism_setting) -> None:
    """Backwards on two streams, each with its own workspace, five times over:
    every ``dx``, ``dz`` and ``dw`` is the bits of the same backward on the
    current stream (``dw`` where its band repeats it; in a fused or row-claims band
    within the budgets of :func:`assert_dw_repeats_within_det` instead),
    which lies within the error budget of the deterministic
    package's, and every ``dx`` and ``dz`` within one BF16 ulp of FP64."""
    cases = [suite.make(4097, 48, seed=30), suite.make(1031, 16, seed=31)]
    refs = [nd_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    dets = [det_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    dxz64s, dw64s, mags = [], [], []
    for (x, z, w, dy), ref, det_out in zip(cases, refs, dets, strict=True):
        _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
        dxz64s.append((dx64, dz64))
        dw64s.append(dw64)
        mags.append(suite.magnitudes(dy, x, z, EPS))
        for what, got, det, want in zip(
            ("dx", "dz", "dw"), ref, det_out, (dx64, dz64, dw64), strict=True
        ):
            assert_error_within_det(what, got, det, want)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [nd_workspace(1, torch.device("cuda")) for _ in streams]
    outputs: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                outputs[i].append(
                    nd_backward(dy, x, z, w, EPS, workspace=workspaces[i])
                )
    torch.cuda.synchronize()
    for i in range(2):
        varies = suite._dw_varies(cases[i][0].numel() // D)
        for got in outputs[i]:
            for what, g, want in zip(("dx", "dz"), got[:2], refs[i][:2], strict=True):
                assert_same_bits(f"stream {i}: {what}", g, want)
            assert_dw_repeats_within_det(
                f"stream {i}: dw",
                got[2],
                refs[i][2],
                dets[i][2],
                dw64s[i],
                mags[i],
                varies=varies,
            )
            assert_dx_dz_within_fp64(f"stream {i}: dx", got[0], dxz64s[i][0])
            assert_dx_dz_within_fp64(f"stream {i}: dz", got[1], dxz64s[i][1])


@requires_device
def test_side_stream_repeats_the_current_stream_bits(determinism_setting) -> None:
    """The forward and the backward (with this package's workspace and
    without one) on a side stream give the bits of the same calls on the
    current stream (``dw`` where its band repeats it; in a fused or row-claims band
    within the budgets of :func:`assert_dw_repeats_within_det` instead),
    which lie within the error budget of the deterministic
    package's; ``dx`` and ``dz`` lie within one BF16 ulp of FP64."""
    x, z, w, dy = suite.make(2049, 48, seed=9)
    want_y = nd_forward(x, z, w, EPS)
    want = nd_backward(dy, x, z, w, EPS)
    dets = det_backward(dy, x, z, w, EPS)
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert_error_within_det("y", want_y, det_forward(x, z, w, EPS), y64)
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), want, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref)
    ws = nd_workspace(x.shape[0] * x.shape[1], x.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = nd_forward(x, z, w, EPS)
        got = nd_backward(dy, x, z, w, EPS, workspace=ws)
        fresh = nd_backward(dy, x, z, w, EPS)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert_same_bits("y", y, want_y)
    assert_y_within_fp64(y, x, z, w, EPS, what="y")
    varies = suite._dw_varies(x.numel() // D)
    mag = suite.magnitudes(dy, x, z, EPS)
    for outputs in (got, fresh):
        for what, g, w_ in zip(("dx", "dz"), outputs[:2], want[:2], strict=True):
            assert_same_bits(what, g, w_)
        assert_dw_repeats_within_det(
            "dw", outputs[2], want[2], dets[2], dw64, mag, varies=varies
        )
        assert_dx_dz_within_fp64("dx", outputs[0], dx64)
        assert_dx_dz_within_fp64("dz", outputs[1], dz64)


@requires_device
def test_cuda_graph_replay_repeats_the_eager_bits(determinism_setting) -> None:
    """Replays of a captured forward and backward (this package's workspace,
    a BF16 ``dw``) on new inputs give the bits of eager calls on those inputs
    (``dw`` where its band repeats it; in a fused or row-claims band within the
    budgets of :func:`assert_dw_repeats_within_det` instead), which lie
    within the error budget of the deterministic package's; ``dx``
    and ``dz`` lie within one BF16 ulp of FP64."""
    T, H = 300, 48
    x, z, w, dy = suite.make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    ws = nd_workspace(T * H, x.device)

    def step():
        nd_forward(xs, zs, w, EPS, out=y)
        nd_backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    names = ("y", "dx", "dz", "dw")
    varies = suite._dw_varies(T * H)
    for k in range(3):
        xn, zn, _, dyn = suite.make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        eager = (nd_forward(xn, zn, w, EPS), *nd_backward(dyn, xn, zn, w, EPS))
        for what, got, want in zip(names[:3], (y, dx, dz), eager[:3], strict=True):
            assert_same_bits(f"replay {k}: {what}", got, want)
        assert_y_within_fp64(y, xn, zn, w, EPS, what=f"y, replay {k}")
        dets = (det_forward(xn, zn, w, EPS), *det_backward(dyn, xn, zn, w, EPS))
        refs = suite.fp64_reference(dyn, xn, zn, w, EPS)
        assert_dx_dz_within_fp64(f"replay {k}: dx", dx, refs[1])
        assert_dx_dz_within_fp64(f"replay {k}: dz", dz, refs[2])
        for what, got, det, ref in zip(names, eager, dets, refs, strict=True):
            assert_error_within_det(f"replay {k}: {what}", got, det, ref)
        assert_dw_repeats_within_det(
            f"replay {k}: dw",
            dw,
            eager[3],
            dets[3],
            refs[3],
            suite.magnitudes(dyn, xn, zn, EPS),
            varies=varies,
        )


@requires_device
def test_autograd_and_checkpoint_repeat_the_functional_bits(
    determinism_setting,
) -> None:
    """Autograd through :func:`cake_rmsnorm_silu_gate_nondet`, through
    :class:`CakeRMSNormSiLUGateNondetFunction` and through a non-reentrant
    checkpoint gives the functional entry points' bits (``y``, ``dx``, ``dz``
    and a BF16 ``dw``; ``dw`` where its band repeats it, in a fused or row-claims band
    within the budgets of :func:`assert_dw_repeats_within_det` instead),
    which lie within the error budget of the deterministic
    package's outputs, ``dx`` and ``dz`` within one BF16 ulp of FP64, ``dw``
    also within the
    weight-gradient budget of the deterministic ``dw``."""
    x, z, w, dy = suite.make(513, 48, seed=8)
    want = (nd_forward(x, z, w, EPS), *nd_backward(dy, x, z, w, EPS))
    dets = (det_forward(x, z, w, EPS), *det_backward(dy, x, z, w, EPS))
    names = ("y", "dx", "dz", "dw")
    refs = suite.fp64_reference(dy, x, z, w, EPS)
    for what, got, det, ref in zip(names, want, dets, refs, strict=True):
        assert_error_within_det(what, got, det, ref)
    mag = suite.magnitudes(dy, x, z, EPS)
    suite.assert_dw_within(
        want[3], dets[3], mag, what="dw vs the deterministic package"
    )
    varies = suite._dw_varies(513 * 48)

    def check(y, xg, zg, wg, what):
        assert_y_within_fp64(y, x, z, w, EPS, what=f"{what} y")
        assert wg.dtype == torch.bfloat16, what
        for name, got, w_ in zip(names[:3], (y, xg, zg), want[:3], strict=True):
            assert_same_bits(f"{what}: {name}", got, w_)
        assert_dw_repeats_within_det(
            f"{what}: dw", wg, want[3], dets[3], refs[3], mag, varies=varies
        )
        assert_dx_dz_within_fp64(f"{what}: dx", xg, refs[1])
        assert_dx_dz_within_fp64(f"{what}: dz", zg, refs[2])

    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = nd_gated_rmsnorm(xa, za, wa, EPS)
    assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")
    xf, zf, wf = (t.clone().requires_grad_(True) for t in (x, z, w))
    yf = CakeRMSNormSiLUGateNondetFunction.apply(xf, zf, wf, EPS)
    yf.backward(dy)
    check(yf, xf.grad, zf.grad, wf.grad, "autograd function")
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        nd_gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")


# --------------------------------------------------------------------------- the forward's bands and the backward's programs


def _spy_launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _role_of(dtype: torch.dtype) -> str:
    """The backward role of a ``dtype`` weight gradient: ``bwd_f32`` for
    FP32, ``bwd_bf16`` for BF16."""
    return "bwd_f32" if dtype == torch.float32 else "bwd_bf16"


def _band_role_dtype(band: str) -> torch.dtype:
    """The weight-gradient dtype of the backward band role ``band``
    (``bwd_f32...`` FP32, ``bwd_bf16...`` BF16)."""
    return torch.float32 if band.startswith("bwd_f32") else torch.bfloat16


def _band_modules(
    arch: str, role: str, sms: int
) -> list[tuple[str, str, int, int | None]]:
    """``(band role, module, first rows, last rows or None)`` of each
    ``loader.BACKWARD_BANDS`` band of the backward ``role`` (``bwd_bf16`` or
    ``bwd_f32``) on ``arch`` with ``sms`` SMs that runs at least one row
    count, in order (``suite._backward_bands``): the band's routed module
    over the rows past the ends of the bands before it up to its own end
    (``suite._band_end``); the last band has no last row.  A band role may
    hold more than one band (on sm_107a the BF16 repeatable role runs on both
    sides of the fused band)."""
    return [
        (band, loader.route_module(arch, band), first, last)
        for band, first, last in suite._backward_bands(arch, role, sms)
    ]


def _long_band_modules(
    arch: str, role: str, sms: int
) -> list[tuple[str, str, int, int | None]]:
    """``(long role, module, first rows, last rows or None)`` of each long
    role of the backward ``role`` on ``arch`` with ``sms`` SMs
    (``suite._long_bands``): the routed module of the long role over the
    packed rows of its band from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM
    on."""
    return [
        (band, loader.route_module(arch, band), first, last)
        for band, first, last in suite._long_bands(arch, role, sms)
    ]


def _band_occurrence(
    arch: str, role: str, sms: int, band: str, nth: int = 1
) -> tuple[str, int, int | None] | None:
    """``(module, first rows, last rows or None)`` of the ``nth`` band of the
    band role ``band`` among :func:`_band_modules`, or ``None`` where the
    role has fewer such bands on ``arch`` with ``sms`` SMs."""
    found = [
        (module, first, last)
        for name, module, first, last in _band_modules(arch, role, sms)
        if name == band
    ]
    return found[nth - 1] if len(found) >= nth else None


def _band_of(arch: str, role: str, rows: int, sms: int) -> tuple[str, int]:
    """``(band role, first rows)`` of the band that runs ``rows`` packed rows
    of the backward ``role`` on ``arch`` with ``sms`` SMs: a long role's
    (:func:`_long_band_modules`) where it holds them, else the
    ``loader.BACKWARD_BANDS`` band (:func:`_band_modules`) that holds them."""
    bands = _long_band_modules(arch, role, sms) + _band_modules(arch, role, sms)
    for band, _, first, last in bands:
        if first <= rows and (last is None or rows <= last):
            return band, first
    raise AssertionError(f"no {role} band of {arch} holds {rows} rows")


def _start_rows(arch: str, name: str, sms: int, packed: bool = True) -> int:
    """The fewest rows the routed module ``name`` runs for on ``arch`` with
    ``sms`` SMs: for a forward module the first row of its first
    ``FORWARD_BANDS`` band (``STRIDED_FORWARD_BANDS`` where the rows are not
    ``packed``; one more than the previous band's ``last * SMs``), for a
    backward module the first row of its ``BACKWARD_BANDS`` band
    (:func:`_band_modules`), for a long role's module the first of its
    packed rows (:func:`_long_band_modules`)."""
    role = loader.MODULES[name]["role"]
    if role.startswith("bwd"):
        call_role = "bwd_f32" if role.startswith("bwd_f32") else "bwd_bf16"
        bands = _band_modules(arch, call_role, sms)
        for _, module, first, _ in bands + _long_band_modules(arch, call_role, sms):
            if module == name:
                return first
        raise AssertionError(f"{arch} has no backward band of {role}")
    first = 1
    table = loader.FORWARD_BANDS if packed else loader.STRIDED_FORWARD_BANDS
    for last, band in table[arch]:
        if band == role:
            return first
        assert last is not None, (arch, role)
        first = int(last) * sms + 1
    raise AssertionError(f"{arch} has no forward band of {role}")


# Forward inputs around and past the ends of the forward's bands: (tokens,
# heads, layout); ``heads=None`` takes ``last * SMs + tokens`` rows of one
# head, ``last`` the rows per SM of the architecture's ``FORWARD_BANDS`` band
# ``i`` of ``layout`` ``flat<i>`` (packed rows: one flat [rows, 128] tensor) or
# of its ``STRIDED_FORWARD_BANDS`` band ``i`` of ``layout`` ``strided<i>``
# (the projection layout, whose ``z`` rows are not packed); ``token<i>`` takes
# one token of that many heads past ``FORWARD_BANDS`` band ``i`` (packed rows
# whose token stride is passed as 0); skipped where the architecture has no
# such band with an end.
FORWARD_CASES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(13, 7, "contiguous", id="13x7"),
    pytest.param(13, 48, "projection", id="13x48-projection"),
    pytest.param(21, 48, "head_major", id="21x48-head_major"),
    pytest.param(0, None, "flat0", id="band0-end"),
    pytest.param(1, None, "flat0", id="band0-end+1"),
    pytest.param(0, None, "flat1", id="band1-end"),
    pytest.param(1, None, "flat1", id="band1-end+1"),
    pytest.param(0, None, "strided0", id="strided-band0-end"),
    pytest.param(1, None, "strided0", id="strided-band0-end+1"),
    pytest.param(0, None, "strided1", id="strided-band1-end"),
    pytest.param(1, None, "strided1", id="strided-band1-end+1"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(4097, 48, "contiguous", id="4097x48"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(6000, 32, "contiguous", id="6000x32"),
    pytest.param(0, None, "flat2", id="band2-end"),
    pytest.param(1, None, "flat2", id="band2-end+1"),
    pytest.param(0, None, "flat3", id="band3-end"),
    pytest.param(1, None, "flat3", id="band3-end+1"),
    pytest.param(0, None, "flat4", id="band4-end"),
    pytest.param(1, None, "flat4", id="band4-end+1"),
    pytest.param(0, None, "flat5", id="band5-end"),
    pytest.param(1, None, "flat5", id="band5-end+1"),
    pytest.param(1, None, "token0", id="one-token-band0-end+1"),
    pytest.param(1, None, "token1", id="one-token-band1-end+1"),
    pytest.param(1, None, "token2", id="one-token-band2-end+1"),
    pytest.param(1, None, "token3", id="one-token-band3-end+1"),
    pytest.param(1, None, "token4", id="one-token-band4-end+1"),
    pytest.param(1, None, "token5", id="one-token-band5-end+1"),
    pytest.param(0, None, "strided2", id="strided-band2-end"),
    pytest.param(1, None, "strided2", id="strided-band2-end+1"),
    pytest.param(0, None, "strided3", id="strided-band3-end"),
    pytest.param(1, None, "strided3", id="strided-band3-end+1"),
    pytest.param(0, None, "strided4", id="strided-band4-end"),
    pytest.param(1, None, "strided4", id="strided-band4-end+1"),
    pytest.param(7685, 48, "contiguous", id="7685x48"),
    pytest.param(7685, 48, "projection", id="7685x48-projection"),
    pytest.param(7685, 48, "head_major", id="7685x48-head_major"),
    pytest.param(9500, 48, "contiguous", id="9500x48"),
    pytest.param(9500, 48, "projection", id="9500x48-projection"),
    pytest.param(9500, 48, "head_major", id="9500x48-head_major"),
    pytest.param(16384, 32, "contiguous", id="16384x32"),
    pytest.param(8192, 48, "contiguous", id="8192x48"),
    pytest.param(8192, 48, "projection", id="8192x48-projection"),
    pytest.param(8192, 48, "head_major", id="8192x48-head_major"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
    pytest.param(32737, 32, "contiguous", id="32737x32"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", FORWARD_CASES)
def test_forward_around_its_bands_within_the_deterministic_error(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On both sides of the ends of the forward's bands and past them, the
    forward launches ``module_for_rows``' module: the module of the first
    ``FORWARD_BANDS`` band of the architecture (``STRIDED_FORWARD_BANDS``
    band where the rows are not packed) with ``rows <= last * SMs`` (the
    tiny, short, pipelined, mid, long, extra-long and plain forwards, each
    where the architecture's bands name it).  ``y`` repeats bit for bit and lies
    within the error budget of the deterministic package's ``y`` and within
    one BF16 ulp of FP64; where a pipelined forward (a deterministic forward
    program with a gate that does not flush) runs, ``y`` is the deterministic
    package's bits (no ``z`` here reaches the deterministic gate's flush)."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    routes = loader.ROUTES[arch]
    if heads is None:
        kind = next(k for k in ("strided", "flat", "token") if layout.startswith(k))
        i = int(layout[len(kind) :])
        table = (
            loader.STRIDED_FORWARD_BANDS if kind == "strided" else loader.FORWARD_BANDS
        )
        if i >= len(table[arch]) or table[arch][i][0] is None:
            pytest.skip(f"{arch} has no {kind} forward band {i} with an end")
        rows = int(table[arch][i][0]) * sms + tokens
        if kind == "token":
            tokens, heads, layout = 1, rows, "contiguous"
        else:
            tokens, heads = rows, 1
            layout = "projection" if kind == "strided" else "flat"
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=21)
    rows = x.numel() // D
    packed = suite._packed(tokens, heads, layout)
    bands = (loader.FORWARD_BANDS if packed else loader.STRIDED_FORWARD_BANDS)[arch]
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    name = loader.module_for_rows(arch, "fwd", rows, sms, packed=packed)
    assert launched == [name] * 2, launched
    want = next(
        routes[band] for last, band in bands if last is None or rows <= last * sms
    )
    assert name == want, (arch, rows, sms)
    y64 = suite.fp64_reference(dy, x, z, w, EPS)[0]
    y_det = det_forward(x, z, w, EPS)
    if loader.MODULES[name]["role"] in ("fwd_pipe", "fwd_pipe_packed"):
        assert_same_bits("y of the pipelined forward", y, y_det)
    assert_error_within_det("y", y, y_det, y64)
    assert_y_within_fp64(y, x, z, w, EPS, what="y")


# Gate inputs where the deterministic package's gate flushes to 0
# (``rcp.approx.ftz`` of a value past 2^126, from about -87.34 down) and FP32
# ``expf`` does not overflow (above about -88.72): the sigmoid is a nonzero
# FP32 subnormal there, and ``y`` a nonzero BF16 value of ordinary rows.
GATE_TAIL = (-87.5, -88.0, -88.5)


@requires_device
@pytest.mark.parametrize("band", range(8))
@pytest.mark.parametrize("kind", ["flat", "strided"])
def test_forward_keeps_the_gate_tail_in_every_band(
    kind, band, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the first row count of every forward band (``FORWARD_BANDS``, or
    ``STRIDED_FORWARD_BANDS`` where the rows are not packed), with every ``z``
    in :data:`GATE_TAIL`, the forward launches the band's module and ``y`` is
    within one BF16 ulp of the FP64 forward plus half the smallest BF16
    subnormal, with no floor: no forward program flushes the gate to 0 where
    FP32 keeps it."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    table = (
        loader.STRIDED_FORWARD_BANDS if kind == "strided" else loader.FORWARD_BANDS
    )[arch]
    if band >= len(table):
        pytest.skip(f"{arch} has {len(table)} {kind} forward bands")
    # The band's first row count; two rows in the first strided band, where one row
    # is packed whatever its strides.
    rows = int(table[band - 1][0]) * sms + 1 if band else 1 + (kind == "strided")
    x, z, w, _ = suite.make(
        rows, 1, layout="projection" if kind == "strided" else "flat", seed=23
    )
    tail = torch.tensor(GATE_TAIL, dtype=z.dtype, device=z.device)
    index = torch.arange(z.numel(), device=z.device) % len(GATE_TAIL)
    z.copy_(tail[index].view(z.shape))
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert launched == [loader.ROUTES[arch][table[band][1]]], (arch, rows, launched)
    xd, zd, wd = (t.double().reshape(-1, D) for t in (x, z, w))
    rstd = torch.rsqrt(xd.square().mean(dim=-1, keepdim=True) + EPS)
    want = xd * rstd * wd * (zd * torch.sigmoid(zd))
    err = (y.double().reshape(-1, D) - want).abs()
    bad = err > want.abs() * suite.BF16_RTOL + 2.0**-134
    assert not bad.any(), (
        f"{arch} {kind} band {band} ({rows} rows): {int(bad.sum())} elements "
        f"of {bad.numel()} off FP64, the largest error {float(err.max()):.3e}"
    )


# Gate inputs at and past the edges of the packed gate's range: below -41,
# around FP32 expf's overflow (|z| from 87.5 to 89), subnormal and zero,
# infinite and NaN, beside saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
    -89.0,
    -88.5,
    -88.0,
    -87.5,
    -42.0,
    -41.0,
    -40.0,
    -1e-30,
    -1e-39,
    -0.0,
    0.0,
    1e-39,
    1e-30,
    40.0,
    87.5,
    88.0,
    89.0,
    100.0,
    1e4,
    float("inf"),
    float("-inf"),
    float("nan"),
]
# Forward inputs (``x`` and ``z`` packed rows, or ``z`` the gate columns of a
# wider projection) the tiny or the short forward runs, and those at the end
# of the short forward's band on sm_100a and sm_103a, 12 rows per SM, on 148 SMs (37x48, 253x7, 55x32) and on
# 152 SMs (38x48, 260x7, 57x32), and on sm_107a, 59 rows per SM, on 212 SMs
# (260x48, 1786x7, 390x32), and one token past it (261x48, 1787x7, 391x32),
# with forward inputs at and past 48 rows per SM on 148 and 152 SMs (148x48,
# 1014x7, 222x32, 152x48, 1042x7, 228x32 and one token more), which the tiny
# or the pipelined forward runs there.
GATE_EDGE_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(13, 48, id="13x48"),
    pytest.param(30, 48, id="30x48"),
    pytest.param(31, 48, id="31x48"),
    pytest.param(33, 48, id="33x48"),
    pytest.param(37, 48, id="37x48"),
    pytest.param(38, 48, id="38x48"),
    pytest.param(13, 7, id="13x7"),
    pytest.param(217, 7, id="217x7"),
    pytest.param(253, 7, id="253x7"),
    pytest.param(260, 7, id="260x7"),
    pytest.param(55, 32, id="55x32"),
    pytest.param(57, 32, id="57x32"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(148, 48, id="148x48"),
    pytest.param(149, 48, id="149x48"),
    pytest.param(152, 48, id="152x48"),
    pytest.param(153, 48, id="153x48"),
    pytest.param(1014, 7, id="1014x7"),
    pytest.param(1015, 7, id="1015x7"),
    pytest.param(1042, 7, id="1042x7"),
    pytest.param(1043, 7, id="1043x7"),
    pytest.param(222, 32, id="222x32"),
    pytest.param(223, 32, id="223x32"),
    pytest.param(228, 32, id="228x32"),
    pytest.param(229, 32, id="229x32"),
    pytest.param(260, 48, id="260x48"),
    pytest.param(261, 48, id="261x48"),
    pytest.param(1786, 7, id="1786x7"),
    pytest.param(1787, 7, id="1787x7"),
    pytest.param(390, 32, id="390x32"),
    pytest.param(391, 32, id="391x32"),
]


def _gate_edge_inputs(tokens: int, heads: int, columns: int = D):
    """``(x, z, w, dy)`` of ``suite.make`` with gate inputs from
    ``GATE_EDGE_VALUES`` in a quarter of the elements of the first
    ``columns`` columns of two of every three rows (so the packed gate's
    range check fails for some threads of a row and holds for others) and
    ordinary ones in the rest."""
    x, z, w, dy = suite.make(tokens, heads, seed=23)
    g = torch.Generator(device=z.device).manual_seed(24)
    values = torch.tensor(GATE_EDGE_VALUES, device=z.device).bfloat16()
    pick = torch.randint(
        0, len(GATE_EDGE_VALUES), z.shape, device=z.device, generator=g
    )
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(tokens * heads, device=z.device).view(tokens, heads, 1)
    edge &= rows % 3 != 0
    edge &= torch.arange(D, device=z.device) < columns
    z.copy_(torch.where(edge, values[pick], z))
    return x, z, w, dy


@requires_device
@pytest.mark.parametrize("layout", ["contiguous", "projection"])
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_CASES)
def test_forward_gate_edges_keep_the_fp64_classes(
    tokens, heads, layout, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With gate inputs at and past the packed gate's range in two of every
    three rows (a quarter of their elements, so the range check fails for some
    threads of a row and holds for others) and ordinary ones in the rest, the
    forward launches ``module_for_rows``' module and ``y`` repeats bit for bit,
    has the FP64 reference's ``NaN``, ``+inf`` and ``-inf`` positions, and its
    finite elements lie within the error budget of the deterministic
    package's."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    x, z, w, dy = _gate_edge_inputs(tokens, heads)
    if layout == "projection":
        wide = torch.zeros(tokens, 2 * heads * D + 16, dtype=z.dtype, device=z.device)
        view = wide[:, heads * D : 2 * heads * D].view(tokens, heads, D)
        view.copy_(z)
        z = view
    launched = _spy_launches(monkeypatch)
    y = nd_forward(x, z, w, EPS)
    assert_same_bits("y", nd_forward(x, z, w, EPS), y)
    packed = suite._packed(tokens, heads, layout)
    name = loader.module_for_rows(arch, "fwd", tokens * heads, sms, packed=packed)
    assert launched == [name] * 2, launched
    y64 = suite.fp64_reference(dy, x, z, w, EPS)[0]
    assert_same_classes("y", y, y64)
    assert_error_within_det(
        "finite y", y, det_forward(x, z, w, EPS), y64, mask=torch.isfinite(y64)
    )
    assert_y_within_fp64(y, x, z, w, EPS, what="y")


# Backward inputs with the forward's gate edges (``_gate_edge_inputs``) in the
# first GATE_EDGE_BACKWARD_COLUMNS columns, so every output keeps finite
# elements (``dw`` in the other columns): 48 and 240 rows (the first
# small-row band of each role: for a BF16 weight gradient the short
# backward), 288 rows (the exact tiny backward of a BF16
# weight gradient on sm_100a and sm_107a, the fused backward on sm_103a),
# then 1008, 6192, 49488, 49536 and 245808 rows in the bands past them (the
# fused, the static and the claimed backwards, as the architecture's
# ``BACKWARD_BANDS`` and the SM count place them: every one of them in the
# fused band of a BF16 weight gradient).
GATE_EDGE_BACKWARD_COLUMNS = 96
GATE_EDGE_BACKWARD_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(5, 48, id="5x48"),
    pytest.param(6, 48, id="6x48"),
    pytest.param(21, 48, id="21x48"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(1031, 48, id="1031x48"),
    pytest.param(1032, 48, id="1032x48"),
    pytest.param(5121, 48, id="5121x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_gate_edges_keep_the_fp64_classes(
    tokens, heads, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the forward's gate edges in the first
    ``GATE_EDGE_BACKWARD_COLUMNS`` columns (``_gate_edge_inputs``) the
    backward launches ``module_for_rows``' module, every output repeats bit
    for bit (but a BF16 ``dw`` in the fused and the row-claims bands, which
    meets the checks that follow on each call instead), ``dx`` and ``dz`` lie
    within one BF16 ulp of FP64, ``dx``,
    ``dz`` and ``dw`` have the FP64 reference's ``NaN``, ``+inf`` and
    ``-inf`` positions in every band, their finite
    elements (each has some) lie within the error budget of the
    deterministic package's, in the fused band ``dx`` and ``dz`` are the
    deterministic package's bits (``NaN`` payloads included), and the
    workspace is zero after every call."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = _gate_edge_inputs(tokens, heads, GATE_EDGE_BACKWARD_COLUMNS)
    rows = tokens * heads
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    assert not bool(ws.any()), "the workspace is not zero"
    again = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    assert not bool(ws.any()), "the workspace is not zero"
    name = loader.module_for_rows(arch, role, rows, sms)
    assert launched == [name] * 2, launched
    assert got[2].dtype == dw_dtype
    varies = suite._varies(name)
    for what, g, a in zip(("dx", "dz", "dw"), got, again, strict=True):
        if what != "dw" or not varies:
            assert_same_bits(what, a, g)
    assert_dx_dz_within_fp64("dx", got[0], dx64)
    assert_dx_dz_within_fp64("dz", got[1], dz64)
    assert_det_bits_where_fused(name, got, dets)
    checked = [
        ("dx", got[0], dets[0], dx64),
        ("dz", got[1], dets[1], dz64),
        ("dw", got[2], dets[2], dw64),
    ]
    if varies:
        checked.append(("again: dw", again[2], dets[2], dw64))
    for what, g, det, ref in checked:
        assert_same_classes(what, g, ref)
        finite = torch.isfinite(ref)
        assert bool(finite.any()), f"{what}: the FP64 reference has no finite element"
        assert_error_within_det(f"finite {what}", g, det, ref, mask=finite, mag=mag)


def _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch) -> None:
    """``suite.REPEATS`` backward launches of module ``name`` into a
    ``dw_dtype`` weight gradient with this package's workspace, then two
    accumulating ones from the same starting value: every output is the
    first launch's bits (the accumulated ``dw`` too; where a BF16 ``dw`` may
    vary, in the fused and the row-claims bands, every launch's ``dw`` and
    accumulated ``dw`` meets the budgets that follow instead), ``dx`` and
    ``dz`` within one BF16 ulp of FP64, and the workspace is zero after every
    call; ``dx``, ``dz``, ``dw`` and the accumulated ``dw`` lie
    within the error budget of the deterministic package's, ``dw`` within the
    weight-gradient budget of the deterministic ``dw`` and the accumulated one
    within that of the FP64 reference; for the fused backward ``dx`` and
    ``dz`` are the deterministic package's bits."""
    rows = x.numel() // D
    mag = suite.magnitudes(dy, x, z, EPS)
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    varies = suite._varies(name)

    def out(value=None):
        if value is None:
            return torch.empty(D, dtype=dw_dtype, device=x.device)
        return torch.full((D,), value, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    acc_d = det_backward(
        dy, x, z, w, EPS, grad_weight=out(0.25), accumulate_grad_weight=True
    )[2]
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    first = None
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first is None:
            first = got
        for what, g, f in zip(("dx", "dz"), got[:2], first[:2], strict=True):
            assert_same_bits(what, g, f)
        assert_dw_repeats_within_det(
            "dw", got[2], first[2], dets[2], dw64, mag, varies=varies
        )
        assert_dx_dz_within_fp64("dx", got[0], dx64)
        assert_dx_dz_within_fp64("dz", got[1], dz64)
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz", "dw"), first, dets, (dx64, dz64, dw64), strict=True
    ):
        assert_error_within_det(what, got, det, ref, mag=mag)
    suite.assert_dw_within(
        first[2], dets[2], mag, what="dw vs the deterministic package"
    )
    assert_det_bits_where_fused(name, first, dets)
    acc = [
        nd_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=out(0.25),
            accumulate_grad_weight=True,
            workspace=ws,
        )[2]
        for _ in range(2)
    ]
    assert not bool(ws.any()), "the workspace is not zero"
    if not varies:
        assert_same_bits("accumulated dw", acc[1], acc[0])
    for acc_dw in acc if varies else acc[:1]:
        assert_error_within_det("accumulated dw", acc_dw, acc_d, 0.25 + dw64, mag=mag)
        suite.assert_dw_within(acc_dw, 0.25 + dw64, mag, what="accumulated dw vs FP64")
    if _rep(name) and _det_routes_same_program():
        assert_same_bits(
            "accumulated dw of the repeatable backward vs the deterministic package",
            acc[0],
            acc_d,
        )
    assert launched == [name] * (suite.REPEATS + 2), launched


# Every backward band role of ``loader.BACKWARD_BANDS`` on any architecture,
# in table order (the weight-gradient dtype from the role,
# :func:`_band_role_dtype`); a test of a band this device does not route
# skips.
BACKWARD_BAND_ROLES = list(
    dict.fromkeys(
        band
        for roles in loader.BACKWARD_BANDS.values()
        for bands in roles.values()
        for _, band in bands
    )
)
# The grid kinds of the modules each band role routes on the architectures
# that route it.
_BAND_KINDS = {
    band: {
        loader.MODULES[routes[band]]["grid_rule"]["kind"]
        for routes in loader.ROUTES.values()
        if band in routes
    }
    for band in BACKWARD_BAND_ROLES
}


def _band_nths() -> dict[tuple[str, int], bool]:
    """``(band role, n) -> whether it has an end`` of the ``n``-th band of each
    band role in a role's table on some architecture (a band role may hold
    more than one band: on sm_107a the BF16 repeatable role runs on both sides
    of the fused band); a band has an end (``last``) where some architecture
    gives it one."""
    nths: dict[tuple[str, int], bool] = {}
    for roles in loader.BACKWARD_BANDS.values():
        for bands in roles.values():
            seen: dict[str, int] = {}
            for last, band in bands:
                seen[band] = seen.get(band, 0) + 1
                key = (band, seen[band])
                nths[key] = nths.get(key, False) or last is not None
    return nths


_BAND_NTHS = _band_nths()
# The ``(band role, n)`` bands in table order, and those with an end (``last``)
# on some architecture (every band but the last one of a role).
BACKWARD_BAND_NTHS = [
    (band, nth) for band in BACKWARD_BAND_ROLES for b, nth in _BAND_NTHS if b == band
]
_BANDS_WITH_AN_END = {key for key, has_end in _BAND_NTHS.items() if has_end}


def _band_id(band: str, nth: int) -> str:
    """The test id of the ``nth`` band of the band role ``band``: the role
    for the first one, the role and ``nth`` for a later one."""
    return band if nth == 1 else f"{band}{nth}"


# The edges of the backward's bands on this device (:func:`_band_modules`):
# (band role, nth, edge).  ``first`` and ``last`` are a band's first and last rows
# (``last`` where the band has an end); for a band of a chunked backward
# (grid kind ``chunks``) ``chunks`` is the program's last rows of chunks of
# ``k`` blocks, ``rows_per_block * chunk_limit * k`` for the fewest ``k`` in
# its band, and ``chunks+1`` its first rows of ``k + 1`` blocks, where its
# band holds them.
BACKWARD_BAND_EDGES = [
    pytest.param(band, nth, edge, id=f"{_band_id(band, nth)}-{edge}")
    for band, nth in BACKWARD_BAND_NTHS
    for edge in ("first", "last", "chunks", "chunks+1")
    if (edge != "last" or (band, nth) in _BANDS_WITH_AN_END)
    and (not edge.startswith("chunks") or "chunks" in _BAND_KINDS[band])
]
# Each band of each band role (``BACKWARD_BAND_NTHS``) as ``(band, nth)``.
BACKWARD_BAND_CASES = [
    pytest.param(band, nth, id=_band_id(band, nth)) for band, nth in BACKWARD_BAND_NTHS
]


@requires_device
@pytest.mark.parametrize("band,nth,edge", BACKWARD_BAND_EDGES)
def test_backward_band_edges_within_the_deterministic_error(
    band, nth, edge, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the first and the last rows of each ``BACKWARD_BANDS`` band on this
    device's SM count (:func:`_band_modules`: the small-row backwards, the
    fused backward, the static backwards, the row-claims backwards and the
    claimed backward, each where the architecture routes it) and where a
    chunked program's chunks
    grow by a block (``backward_chunks``): ``module_for_rows`` runs the
    band's program from its first to its last rows only, and
    ``_check_backward`` holds on flat [rows, 128] inputs into the band's
    weight-gradient dtype."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _band_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    if edge == "first":
        rows = first
    elif edge == "last":
        if last is None:
            pytest.skip(f"the {band} band of {arch} has no end")
        rows = last
    else:
        rule = loader.MODULES[name]["grid_rule"]
        if rule["kind"] != "chunks":
            pytest.skip(f"the {band} program of {arch} has no chunks")
        span = int(rule["rows_per_block"]) * int(rule["chunk_limit"])
        k = -(-first // span)
        rows = k * span + (edge == "chunks+1")
        if last is not None and rows > last:
            pytest.skip(f"the {band} band ({first}-{last} rows) has no such edge")
        assert loader.backward_chunks(name, rows)[0] == k + (edge == "chunks+1")
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    if edge == "first" and first > 1:
        assert loader.module_for_rows(arch, role, rows - 1, sms) != name, rows
    if edge == "last":
        assert loader.module_for_rows(arch, role, rows + 1, sms) != name, rows
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=22)
    _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch)


# Backward inputs of every program and layout: (tokens, heads, layout).
BACKWARD_SHAPES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(5, 48, "projection", id="5x48-projection"),
    pytest.param(13, 7, "head_major", id="13x7-head_major"),
    pytest.param(255, 1, "flat", id="255x1"),
    pytest.param(256, 1, "flat", id="256x1"),
    pytest.param(257, 1, "flat", id="257x1"),
    pytest.param(1031, 48, "contiguous", id="1031x48"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(2701, 48, "contiguous", id="2701x48"),
    pytest.param(3091, 48, "projection", id="3091x48-projection"),
    pytest.param(3157, 48, "head_major", id="3157x48-head_major"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(5300, 32, "contiguous", id="5300x32"),
    pytest.param(9600, 48, "contiguous", id="9600x48"),
    pytest.param(25000, 48, "contiguous", id="25000x48"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_shapes_within_the_deterministic_error(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs of several shapes and layouts: ``module_for_rows`` runs one of
    the role's programs (the modules of its bands, :func:`_band_modules`, or
    on packed rows of its long roles, :func:`_long_band_modules`) and
    ``_check_backward`` holds."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=24)
    rows = x.numel() // D
    packed = suite._backward_packed(tokens, heads, layout)
    name = loader.module_for_rows(arch, role, rows, sms, packed=packed)
    bands = _band_modules(arch, role, sms)
    if packed:
        bands += _long_band_modules(arch, role, sms)
    programs = {module for _, module, _, _ in bands}
    assert name in programs, (name, programs)
    _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch)


# The backend suite's ``BULK_BAND_CASES`` (strided and packed inputs in the
# bands of the row-claims, the claimed and the fused backwards) and the same
# layouts in the fused band (``bwd_bf16_fused``, every architecture) where the
# suite does not list it; a case of a band this device does not route skips.
FUSED_BANDS = ("bwd_bf16_fused",)
BULK_LAYOUTS = ("projection", "head_major", "one_token", "contiguous")
LAYOUT_BAND_CASES = [
    *suite.BULK_BAND_CASES,
    *(
        pytest.param(band, layout, id=f"{band}-{layout}")
        for band in FUSED_BANDS
        if band not in {case.values[0] for case in suite.BULK_BAND_CASES}
        for layout in BULK_LAYOUTS
    ),
]


@requires_device
@pytest.mark.parametrize("band,layout", LAYOUT_BAND_CASES)
def test_layouts_in_the_row_claims_and_claimed_backward_bands_within_the_deterministic_error(
    band,
    layout,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LAYOUT_BAND_CASES`` (strided and packed inputs in the bands of the
    row-claims, the claimed and the fused backwards,
    ``suite._bulk_band_shape``): ``module_for_rows`` runs the band's program,
    the launch passes the strides of the layout, which pass the kernels'
    packed-rows test (``_packed``: ``dy``, ``x`` and ``z`` of one packed row
    stride, the rows the sm_100a and sm_103a programs warm with a bulk
    prefetch) for packed rows of more than one token only, and
    ``_check_backward`` holds (on the other layouts those programs warm the
    rows with a prefetch per thread, and so does the fused program, which
    warms packed rows with a bulk prefetch on every architecture; the fused
    backward gives the deterministic package's ``dx`` and ``dz`` bits)."""
    shape = suite._bulk_band_shape(band, layout)
    if shape is None:
        pytest.skip(f"this device routes no {band} band with a {layout} input")
    tokens, heads, layout = shape
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=26)
    rows = x.numel() // D
    packed = layout == "contiguous" and tokens > 1
    assert suite._backward_packed(tokens, heads, layout) == packed, layout
    name = loader.module_for_rows(arch, role, rows, sms, packed=packed)
    assert name == loader.ROUTES[arch][band], (rows, name)
    ws = nd_workspace(rows, x.device)
    launched = _spy_strides(monkeypatch)
    nd_backward(
        dy,
        x,
        z,
        w,
        EPS,
        grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device),
        workspace=ws,
    )
    assert [n for n, _ in launched] == [name], launched
    assert _packed(launched[0][1]) == packed, (layout, launched[0][1])
    assert not bool(ws.any()), "the workspace is not zero"
    _check_backward(x, z, w, dy, dw_dtype, name, monkeypatch)


@requires_device
@pytest.mark.parametrize("band,nth", BACKWARD_BAND_CASES)
def test_backward_with_infinite_grad_out_keeps_the_fp64_classes(
    band, nth, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in rows that are not the last, in each
    ``BACKWARD_BANDS`` band of this device (:func:`_band_modules`: a band's
    last rows, 4097 rows past the first of the last band, which has no end),
    into the band's weight-gradient dtype: the
    weight gradient of the columns holding them is not finite and has the
    FP64 reference's class (an infinity its sign, a ``NaN`` a ``NaN``), every
    other column lies within the error budget of the deterministic package's
    and within the weight-gradient budget of the deterministic ``dw``, and
    ``dx`` and ``dz`` have the FP64 reference's ``NaN``, ``+inf`` and ``-inf``
    positions, their other elements within one BF16 ulp of FP64 and within
    the error budget,
    on every repeat the same bits (but a BF16 ``dw`` in a fused or row-claims band,
    whose every launch meets the weight gradient's checks instead), in the
    fused band ``dx`` and ``dz`` the deterministic package's bits (``NaN``
    payloads included), with the workspace zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _band_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    rows = first + 4097 if last is None else last
    assert loader.module_for_rows(arch, role, rows, sms) == name
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=27)
    # (row, column, sign): +inf twice in column 3, -inf in column 40, +inf and
    # -inf in column 77; no infinite term in the last row
    for r, c, sign in (
        (0, 3, 1.0),
        (rows // 2, 3, 1.0),
        (17, 40, -1.0),
        (rows // 3, 77, 1.0),
        (rows // 3 + 5, 77, -1.0),
    ):
        dy[r, c] = sign * float("inf")
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert torch.equal(~torch.isfinite(dw64), cols), "FP64 dw columns"
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    varies = suite._varies(name)
    first_run = None
    dws = []
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz"), got[:2], first_run[:2], strict=True):
            assert_same_bits(what, g, f)
        if not varies:
            assert_same_bits("dw", got[2], first_run[2])
        dws.append(got[2])
        assert_dx_dz_within_fp64("dx", got[0], dx64)
        assert_dx_dz_within_fp64("dz", got[1], dz64)
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz"), first_run[:2], dets[:2], (dx64, dz64), strict=True
    ):
        assert_same_classes(what, got, ref)
        assert_error_within_det(
            f"finite {what}", got, det, ref, mask=torch.isfinite(ref)
        )
    assert_det_bits_where_fused(name, first_run, dets)
    dw_d = dets[2]
    for dw in dws if varies else dws[:1]:
        assert_same_classes("dw", dw, dw64)
        assert_error_within_det(
            "finite dw", dw, dw_d, dw64, mask=torch.isfinite(dw64), mag=mag
        )
        suite.assert_dw_within(
            dw[~cols], dw_d[~cols], mag[~cols], what="finite dw columns"
        )
    assert launched == [name] * suite.REPEATS, launched


def _block_rows(name: str) -> int:
    """The rows of one row block of the backward module ``name``, from its
    record: ``rows_per_block`` of its grid rule for a chunked backward (grid
    kind ``chunks``: the static and the claimed backwards, 8) and for the
    repeatable backward (``claimed_row_blocks``, 16), ``rows_per_cta`` for
    the fused and a row-claims backward (``row_claims``, 16), and for a
    small-row backward (``cluster``) its launch block's threads over the
    teams of its CTA over 16 threads a row (``suite._cluster_block_rows``):
    16 rows for the short backward (one 256-thread team a CTA), 8 for the
    exact tiny backwards (4 teams of the deterministic backward's 128-thread
    CTA in a 512-thread CTA, each team its own 8-row block)."""
    rule = loader.MODULES[name]["grid_rule"]
    if rule["kind"] in ("chunks", "claimed_row_blocks"):
        return int(rule["rows_per_block"])
    if rule["kind"] == "row_claims":
        return int(rule["rows_per_cta"])
    assert rule["kind"] == "cluster", (name, rule)
    return suite._cluster_block_rows(name)


def _last_block_rows(block: int, first: int, last: int | None) -> dict[str, int]:
    """Three row counts of a program of ``block``-row blocks over ``first``
    to ``last`` rows that are not a multiple of its block: ``rem1`` one row
    in the last block and ``half`` half a block in it (``block // 2`` rows: 8
    rows of a 16-row block, whose padded rows must not turn an infinite
    ``grad_out`` in the last row into a ``NaN`` weight gradient), each the
    smallest such count from ``2 * block + 1`` rows and ``first``, and
    ``remmax`` one row short of a whole block (the largest up to ``last``, or
    4096 rows past ``rem1`` for the last band, which has no last row)."""
    lo = max(first, 2 * block + 1)
    low = lo + (1 - lo) % block
    half = lo + (block // 2 - lo) % block
    high = low + 4096 + block - 2 if last is None else last - (last + 1) % block
    return {"rem1": low, "half": half, "remmax": high}


@requires_device
@pytest.mark.parametrize("count", ["rem1", "half", "remmax"])
@pytest.mark.parametrize("band,nth", BACKWARD_BAND_CASES)
def test_backward_with_infinite_grad_out_in_the_last_row_keeps_the_fp64_classes(
    count, band, nth, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in the last row, at row counts that are not a
    multiple of the program's row block (``_last_block_rows``: one row, half
    a block and one row short of a whole block in the last block), in each
    ``BACKWARD_BANDS`` band of this device (:func:`_band_modules`) into the
    band's weight-gradient dtype: ``+inf`` in column 3, ``-inf`` in
    column 40, and ``+inf`` in column 77 beside ``-inf`` in the row before.
    The weight gradient of those columns has the FP64 reference's class
    (``+inf``, ``-inf`` and ``NaN``) in every band, every other column lies
    within the error budget of the deterministic package's and within the
    weight-gradient budget of the deterministic ``dw``, and ``dx`` and ``dz``
    have the FP64 reference's ``NaN``, ``+inf`` and ``-inf`` positions and
    their other elements within one BF16 ulp of FP64, on every repeat the
    same bits (but a BF16 ``dw`` in a fused or row-claims band,
    whose every launch meets the weight gradient's checks instead), in the
    fused band ``dx`` and ``dz`` and in the repeatable band ``dx``, ``dz``
    and ``dw`` the deterministic package's bits (``NaN`` payloads included),
    with the workspace zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _band_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    block = _block_rows(name)
    rows = _last_block_rows(block, first, last)[count]
    if not first <= rows <= (last if last is not None else rows):
        pytest.skip(f"{band} ({first}-{last} rows) has no such row count")
    assert rows % block != 0, rows
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=28)
    # (row, column, sign of the weight-gradient term): a term dy * silu(z) * x
    # * rstd has dy's sign times that of z * x, so dy takes the product
    for r, c, sign in (
        (rows - 1, 3, 1.0),
        (rows - 1, 40, -1.0),
        (rows - 1, 77, 1.0),
        (rows - 2, 77, -1.0),
    ):
        zx = float(z[r, c]) * float(x[r, c])
        assert zx != 0.0, (r, c)
        dy[r, c] = math.copysign(float("inf"), sign * zx)
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    _, dx64, dz64, dw64 = suite.fp64_reference(dy, x, z, w, EPS)
    assert torch.equal(~torch.isfinite(dw64), cols), "FP64 dw columns"
    assert bool(torch.isposinf(dw64[3])), dw64[3]
    assert bool(torch.isneginf(dw64[40])), dw64[40]
    assert bool(torch.isnan(dw64[77])), dw64[77]
    mag = suite.magnitudes(dy, x, z, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    dets = det_backward(dy, x, z, w, EPS, grad_weight=out())
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    varies = suite._varies(name)
    first_run = None
    dws = []
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz"), got[:2], first_run[:2], strict=True):
            assert_same_bits(what, g, f)
        if not varies:
            assert_same_bits("dw", got[2], first_run[2])
        dws.append(got[2])
        assert_dx_dz_within_fp64("dx", got[0], dx64)
        assert_dx_dz_within_fp64("dz", got[1], dz64)
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got, det, ref in zip(
        ("dx", "dz"), first_run[:2], dets[:2], (dx64, dz64), strict=True
    ):
        assert_same_classes(f"{what} rows={rows}", got, ref)
        assert_error_within_det(
            f"finite {what}", got, det, ref, mask=torch.isfinite(ref)
        )
    assert_det_bits_where_fused(name, first_run, dets)
    dw_d = dets[2]
    for dw in dws if varies else dws[:1]:
        assert_same_classes(f"dw rows={rows}", dw, dw64)
        assert_error_within_det(
            "finite dw", dw, dw_d, dw64, mask=torch.isfinite(dw64), mag=mag
        )
        suite.assert_dw_within(
            dw[~cols], dw_d[~cols], mag[~cols], what="finite dw columns"
        )
    assert launched == [name] * suite.REPEATS, launched


# --------------------------------------------------------------------------- one head: packed rows


def _packed(strides: dict) -> bool:
    """The kernels' packed-rows test over the strides of one launch: every
    input's head stride is 128 and its token stride ``heads * 128``.  The mid
    and the extra-long forward warm row blocks in L2 with a bulk prefetch only
    when it holds, and so do the row-claims mid and the claimed backward of
    sm_100a and sm_103a (``dy``, ``x`` and ``z`` of one packed row stride)."""
    heads = int(strides["heads"])
    inputs = [k[: -len("_head")] for k in strides if k.endswith("_head")]
    return bool(inputs) and all(
        strides[f"{n}_head"] == D and strides[f"{n}_tok"] == heads * D for n in inputs
    )


def _spy_strides(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """``(module, {"heads", "<input>_tok", "<input>_head"})`` of every launch."""
    launched: list[tuple[str, dict]] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        strides = {
            k: int(v)
            for k, v in values.items()
            if k == "heads" or k.endswith(("_tok", "_head"))
        }
        launched.append((name, strides))
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


# One-head layouts of a [rows, 128] input: (passes the packed-rows test, read
# in place).  ``three_d`` [rows, 1, 128], ``flat`` [rows, 128], ``transposed``
# the transpose of [1, rows, 128] (token stride 128, head stride rows * 128),
# ``first_of_two`` the first head of [rows, 2, 128] (token stride 256: rows
# that do not follow one another), ``misaligned`` [rows, 128] at a storage
# offset of 8 elements (copied once by the host).
ONE_HEAD_LAYOUTS = {
    "three_d": (True, True),
    "flat": (True, True),
    "transposed": (True, True),
    "first_of_two": (False, True),
    "misaligned": (True, False),
}


def _one_head(base: torch.Tensor, layout: str) -> torch.Tensor:
    """``base`` ([rows, 128]) as a one-head input of ``layout``."""
    rows = base.shape[0]
    if layout == "three_d":
        return base.clone().view(rows, 1, D)
    if layout == "flat":
        return base.clone()
    if layout == "transposed":
        return base.clone().view(1, rows, D).transpose(0, 1)
    if layout == "first_of_two":
        wide = torch.zeros(rows, 2, D, dtype=base.dtype, device=base.device)
        wide[:, 0] = base
        return wide[:, :1]
    if layout == "misaligned":
        buf = torch.zeros(rows * D + 8, dtype=base.dtype, device=base.device)
        out = buf[8:].view(rows, D)
        out.copy_(base)
        return out
    raise ValueError(layout)


@pytest.mark.parametrize("layout", list(ONE_HEAD_LAYOUTS))
def test_one_head_input_passes_the_packed_rows_test_when_its_rows_follow_one_another(
    layout,
) -> None:
    """Without a device: a one-head input's head stride is passed as 128 (it is
    never dereferenced), so the kernels see packed rows exactly when its token
    stride is 128 (or the host copied it), as for ``[T, H, 128]`` inputs."""
    base = torch.randn(7, D).bfloat16()
    t = _one_head(base, layout)
    view, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    packed, in_place = ONE_HEAD_LAYOUTS[layout]
    assert head == D
    assert tok == (2 * D if layout == "first_of_two" else D)
    assert _packed({"heads": 1, "x_tok": tok, "x_head": head}) == packed
    assert (view.data_ptr() == t.data_ptr()) == in_place
    assert torch.equal(view.reshape(-1, D), base)


@pytest.mark.parametrize(
    "shape,strides,want",
    [
        pytest.param((1, 1, D), None, (0, D), id="1x1"),
        pytest.param((1, D), None, (0, D), id="1-flat"),
        pytest.param((D,), None, (0, D), id="one-row"),
        pytest.param((1, 2, D), None, (0, D), id="1x2"),
        pytest.param((5, 2, D), None, (2 * D, D), id="5x2"),
        pytest.param((5, 3, D), None, (3 * D, D), id="5x3"),
        pytest.param((5, 2, D), (4 * D, D, 1), (4 * D, D), id="5x2-projection"),
        pytest.param((5, 2, D), (D, 5 * D, 1), (D, 5 * D), id="5x2-head_major"),
    ],
)
def test_kernel_view_strides(shape, strides, want) -> None:
    """Without a device: a single token's stride is passed as 0, a single
    head's as 128, any other dimension's as it is."""
    if strides is None:
        t = torch.zeros(shape, dtype=torch.bfloat16)
    else:
        t = torch.empty_strided(shape, strides, dtype=torch.bfloat16)
    _, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    assert (tok, head) == want


# The modules that warm rows in L2, by role: the mid and the extra-long
# forward (row blocks of packed rows; one row past the start of their
# ``STRIDED_FORWARD_BANDS`` band, where the one-head layouts with packed rows
# run the forward of that row count's ``FORWARD_BANDS`` band), the
# row-claims mid backward (sm_100a and sm_103a: thread 0 warms the next block
# of packed rows), the claimed backward (on sm_100a and sm_103a a CTA's
# next block, with a bulk prefetch per tensor where the rows are packed; on
# sm_107a the rows of the block two ahead); and the fused backward (every
# architecture: thread 0 warms the next block of a claim with a bulk prefetch
# per tensor where the rows are packed, else each thread its own row), whose
# bits must not depend on the layout either; so do the repeatable backward
# and its long role (which runs the packed layouts only; a layout whose rows
# do not follow one another runs the repeatable backward, the same bits).
# The one-head inputs take one row past the start of each one's band
# (``_start_rows``); a module that no band of this device runs (a forward
# module: no ``STRIDED_FORWARD_BANDS`` band) skips.
ONE_HEAD_ROUTES = [
    pytest.param("fwd", "fwd_mid", id="fwd_mid"),
    pytest.param("fwd", "fwd_xlong", id="fwd_xlong"),
    pytest.param("bwd_bf16", "bwd_bf16_rc_mid", id="bwd_bf16_rc_mid"),
    pytest.param("bwd_bf16", "bwd_bf16_mid", id="bwd_bf16_claimed"),
    pytest.param("bwd_f32", "bwd_f32_mid", id="bwd_f32_claimed"),
    pytest.param("bwd_bf16", "bwd_bf16_fused", id="bwd_bf16_fused"),
    pytest.param("bwd_bf16", "bwd_bf16_rep", id="bwd_bf16_rep"),
    pytest.param("bwd_bf16", "bwd_bf16_rep_long", id="bwd_bf16_rep_long"),
]


@requires_device
@pytest.mark.parametrize("role,band", ONE_HEAD_ROUTES)
def test_one_head_inputs_give_the_same_bits_for_every_layout(
    role, band, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row past the start of the mid and the extra-long forward (their
    strided bands) and of the row-claims mid, the claimed, the fused and the
    repeatable backwards and the long role's packed rows: a one-head input of
    every layout (``ONE_HEAD_LAYOUTS``) launches ``module_for_rows``' module
    (the module of its layout's packing) with head stride 128, passes the
    packed-rows test exactly when its rows follow one another (or the host
    copied it), and returns the same
    ``y``, or ``dx``, ``dz`` and ``dw``, bits for every layout launching the
    same module and every repeat (but a BF16 ``dw`` of the fused and the
    row-claims mid backwards, which lies within the budgets of
    :func:`assert_dw_repeats_within_det` on every launch instead), with the
    workspace zero after every call, ``dx`` and ``dz`` within one BF16 ulp of
    FP64; each module's outputs lie within the error
    budget of the deterministic package's on the flat rows, ``dw`` also within
    the weight-gradient budget of the deterministic ``dw``, and the fused
    backward's ``dx`` and ``dz`` are the deterministic package's bits on the
    flat rows (the repeatable backward's and its long role's ``dw`` too)."""
    arch = loader.device_arch(torch.cuda.current_device())
    name = loader.ROUTES[arch].get(band)
    if name is None:
        pytest.skip(f"{arch} routes no {band} module")
    sms = _sm_count()
    if role != "fwd" and name not in {
        module
        for _, module, _, _ in _band_modules(arch, role, sms)
        + _long_band_modules(arch, role, sms)
    }:
        pytest.skip(f"no band of {arch} on {sms} SMs runs {band}")
    strided_bands = {b for _, b in loader.STRIDED_FORWARD_BANDS[arch]}
    if role == "fwd" and band not in strided_bands:
        pytest.skip(f"no band of {arch} of other layouts runs {band}")
    strided = role == "fwd"
    rows = _start_rows(arch, name, sms, packed=not strided) + 1
    assert loader.module_for_rows(arch, role, rows, sms, packed=not strided) == name
    x0, z0, w, dy0 = suite.make(rows, 1, layout="flat", seed=25)
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16
    y64, dx64, dz64, dw64 = suite.fp64_reference(dy0, x0, z0, w, EPS)

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x0.device)

    if role == "fwd":
        assert_y_within_fp64(nd_forward(x0, z0, w, EPS), x0, z0, w, EPS, what="flat y")
        names = ("y",)
        dets = (det_forward(x0, z0, w, EPS),)
        refs = (y64,)
        calls = 2
    else:
        names = ("dx", "dz", "dw")
        dets = det_backward(dy0, x0, z0, w, EPS, grad_weight=out())
        refs = (dx64, dz64, dw64)
        calls = suite.REPEATS
        ws = nd_workspace(rows, x0.device)
        mag = suite.magnitudes(dy0, x0, z0, EPS)
    varies = role != "fwd" and suite._varies(name)
    launched = _spy_strides(monkeypatch)
    wants: dict[str, tuple] = {}
    for layout, (packed, _) in ONE_HEAD_LAYOUTS.items():
        x, z, dy = (_one_head(t, layout) for t in (x0, z0, dy0))
        module = loader.module_for_rows(arch, role, rows, sms, packed=packed)
        want = wants.get(module)
        launched.clear()
        for _ in range(calls):
            if role == "fwd":
                got = (nd_forward(x, z, w, EPS).reshape(rows, D),)
            else:
                dx, dz, dw = nd_backward(
                    dy, x, z, w, EPS, grad_weight=out(), workspace=ws
                )
                got = (dx.reshape(rows, D), dz.reshape(rows, D), dw)
                assert_dx_dz_within_fp64(f"{layout}: dx", got[0], dx64)
                assert_dx_dz_within_fp64(f"{layout}: dz", got[1], dz64)
                assert not bool(ws.any()), (layout, "the workspace is not zero")
            if want is None:
                want = wants[module] = got
            for what, g, w_ in zip(names, got, want, strict=True):
                if what == "dw":
                    assert_dw_repeats_within_det(
                        f"{layout}: dw", g, w_, dets[2], dw64, mag, varies=varies
                    )
                else:
                    assert_same_bits(f"{layout}: {what}", g, w_)
        assert [n for n, _ in launched] == [module] * calls, (layout, launched)
        for _, strides in launched:
            assert strides["heads"] == 1, (layout, strides)
            heads = [v for k, v in strides.items() if k.endswith("_head")]
            assert heads and all(v == D for v in heads), (layout, strides)
            assert _packed(strides) == packed, (layout, strides)
    assert name in wants, (name, sorted(wants))
    for want in wants.values():
        for what, got, det, ref in zip(names, want, dets, refs, strict=True):
            assert_error_within_det(
                what, got, det, ref, mag=mag if role != "fwd" else None
            )
        if role != "fwd":
            suite.assert_dw_within(
                want[2], dets[2], mag, what="dw vs the deterministic package"
            )
            assert_det_bits_where_fused(name, want, dets)


# Identical rows whose weight-gradient terms ``dy * silu(z) * x * rstd`` are
# below FP32's smallest subnormal, 2^-149 (``x`` 1, ``z`` 2^-133, the smallest
# BF16 subnormal, so ``silu(z)`` is 2^-134 exactly; ``dy`` 2^-16 .. 2^-14 by
# column): each term rounds to 0 or 2^-149 in FP32, the same in every row, so
# the error grows with the rows.  Row counts of the small-row band (16 rows)
# and past it (4096, 65536 and 262144 rows: the fused, the static and the
# claimed backwards, as the architecture's ``BACKWARD_BANDS`` and the SM count
# place them; every one of them in the fused band of a BF16 weight gradient).
SUBNORMAL_SHAPES = [
    pytest.param(16, 1, id="16x1"),
    pytest.param(4096, 1, id="4096x1"),
    pytest.param(2048, 32, id="2048x32"),
    pytest.param(8192, 32, id="8192x32"),
]


def _underflowing_rows(T: int, H: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    col = torch.arange(D, device=device, dtype=torch.float64)
    x = torch.ones(T, H, D, device=device).bfloat16()
    z = torch.full((T, H, D), 2.0**-133, device=device).bfloat16()
    scale = torch.where(
        col < 64, 2.0**-16 * (1 + col / 64), 2.0**-15 * (1 + (col - 64) / 64)
    )
    dy = scale.float().bfloat16().view(1, 1, D).expand(T, H, D).contiguous()
    w = torch.ones(D, device=device).bfloat16()
    return x, z, w, dy


@requires_device
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize("T,H", SUBNORMAL_SHAPES)
def test_subnormal_weight_gradient_within_the_floored_budget(
    T, H, dw_dtype, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weight gradients in the subnormal range, where the documented budget
    ``|dw64| * 2^-7 + sum|terms| * 2^-20`` (for FP32 normal gates, terms and
    sums) does not hold for either package: the backward launches
    ``module_for_rows``' module, every output repeats bit for bit (but a BF16
    ``dw`` of the fused or a row-claims backward, held to the floored budget
    below on every launch instead), ``dx`` and
    ``dz`` lie within one BF16 ulp of FP64 (``dx``, whose FP64 values lie
    below BF16's smallest subnormal, with an absolute floor of ``2^-134``),
    ``dx``, ``dz`` and a BF16 ``dw`` lie within
    the error budget of the deterministic package's, and every ``dw`` lies
    within that budget widened by the
    documented underflow allowance, ``(1 + |x * rstd|) * 2^-150`` per row
    plus half the dtype's smallest subnormal, with the workspace zero; the
    fused backward's ``dx`` and ``dz`` are the deterministic package's
    bits."""
    x, z, w, dy = _underflowing_rows(T, H)
    rows = T * H
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    name = loader.module_for_rows(arch, role, rows, _sm_count())
    dets = det_backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device)
    )
    _, dx64, dz64, _ = suite.fp64_reference(dy, x, z, w, EPS)
    xd, zd = x.double(), z.double()
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    terms = (dy.double() * zd * torch.sigmoid(zd) * n).reshape(rows, D)
    dw64 = terms.sum(0)
    smallest = 2.0**-133 if dw_dtype == torch.bfloat16 else 2.0**-149
    allowance = (1 + n.abs().max().item()) * rows * 2.0**-150 + smallest / 2
    budget = dw64.abs() * suite.BF16_RTOL + terms.abs().sum(0) * suite.DW_MAG_TOL
    assert bool(((dets[2].double() - dw64).abs() > budget).any()), (
        "the inputs stay within the normal-range budget"
    )
    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    varies = suite._varies(name)
    first = None
    for _ in range(suite.REPEATS):
        dw = torch.empty(D, dtype=dw_dtype, device=x.device)
        dx, dz, _ = nd_backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
        if first is None:
            first = (dx, dz, dw)
        for what, g, f in zip(("dx", "dz", "dw"), (dx, dz, dw), first, strict=True):
            if what != "dw" or not varies:
                assert_same_bits(what, g, f)
        # dx's FP64 values lie below BF16's smallest subnormal (2^-133)
        assert_dx_dz_within_fp64("dx", dx, dx64, floor=2.0**-134)
        assert_dx_dz_within_fp64("dz", dz, dz64)
        excess = (dw.double() - dw64).abs() - budget - allowance
        assert not bool((excess > 0).any()), (
            f"dw beyond the floored budget by {excess.max().item():.3e}"
        )
        assert not bool(ws.any()), "the workspace is not zero"
    # an FP32 dw is held to the floored budget above, as FlashInfer main's
    # suite holds it: the weight-gradient budget against the deterministic dw
    # does not hold below the normal range
    outputs = ("dx", "dz", "dw") if dw_dtype == torch.bfloat16 else ("dx", "dz")
    for what, got, det, ref in zip(
        outputs, first, dets, (dx64, dz64, dw64), strict=False
    ):
        assert_error_within_det(what, got, det, ref)
    assert_det_bits_where_fused(name, first, dets)
    assert launched == [name] * suite.REPEATS, launched
