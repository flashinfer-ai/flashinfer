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

Backend suite of the Cake SiLU-gated RMSNorm training package of the
non-deterministic family
(``flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train``, the backend of
:mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`):
``RMSNorm(x) * w * SiLU(z)``.

``y``, ``dx`` and ``dz`` must lie within one BF16 ulp of an independent FP64
reference (plus an absolute floor of 1/64 ulp of the tensor's largest
magnitude), and the weight gradient within the deterministic suite's budget,
``|dw - dw64| <= |dw64| * 2^-7 + sum|terms| * 2^-20``; each output is also
checked against the explicit eager operation (FP32 arithmetic, one final cast
to BF16, PyTorch autograd for the gradients) with the deterministic suite's
bounds: ``y``, ``dx`` and ``dz`` within one BF16 ulp of the eager output plus
the floor of the FP64 reference's largest magnitude, ``dw`` within
``|dw_eager| * 2^-7 + sum|terms| * 2^-20``.  ``y``, ``dx``, ``dz`` and an
FP32 weight gradient (accumulated or not) must be the same bits on every
launch and on every path: streams, CUDA Graph replays, autograd,
``compute_grad_weight`` (``False`` the bits of the BF16 bands' module), grids
smaller than the work (the static, the claimed, the row-claims, the fused
and the repeatable backwards at 1, 3 and 7 CTAs; the repeatable backward's
weight gradient the bits of its grid, whose chunks follow it), a workspace
reused across shapes; so must a BF16 weight gradient of every backward but
the fused backward (``bwd_bf16_fused``, which runs a BF16 weight gradient's
rows in a band of ``BACKWARD_BANDS`` past the small-row bands on every
architecture, on sm_107a between two repeatable bands) and the row-claims backwards (``_rc``
and ``_rc_mid``, :func:`_varies`), whose CTAs claim 16-row blocks at run time
and add FP32 runs of rows with FP64 atomics,
so that their BF16 weight gradient may vary in its last bits from run to run:
there every launch's BF16 weight gradient, alone and accumulated, lies within
the budgets above instead.  The matrix covers dynamic token and head
counts (``H = 48`` and small or irregular row counts), ``[T, H, 128]`` strided
and flattened layouts (also in the bands of the claimed, the row-claims, the
fused and the repeatable backwards, whose sm_100a and sm_103a claimed and
row-claims mid programs and fused and repeatable programs on every
architecture warm packed rows with a bulk prefetch and other layouts with a
prefetch per thread),
near-zero variance, saturated gates, weight-gradient terms that cancel per
column, FP32 and BF16 weight gradients,
accumulation, ``compute_grad_weight=False``, the small-row backwards (one
cluster each, in the rows of their ``BACKWARD_BANDS`` bands: the short
backward of a BF16 weight gradient and the exact tiny backwards; no workspace made
or touched, a caller workspace checked and left as it is, the first row count
past them on the backward), empty work, a workspace reused across shapes (zero after every
call), a workspace with nonzero scratch returned to zero by
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, two streams with their own
workspaces, CUDA Graph capture and replay (captured calls without a
workspace: one workspace per capture and capture stream, zero-filled outside
the graph, no fill kernel in it; two such graphs replayed at the same time on
two streams; the fill inside the graph past the graph-workspace limit), the
autograd wrapper (non-reentrant checkpointing included), the state a
checkpointed step keeps (FlashInfer issue #7039), the registry's routes (a
module for every call, which the support query requires; none of the programs
the package no longer ships), bands, chunks and grids (the first and the last
rows of every band of ``BACKWARD_BANDS`` run the band's program; packed rows
of the last repeatable band from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on
run its long role, ``bwd_bf16_rep_long``, the bits of the repeatable
backward), and the mode
policy: the mode is in the name, so no entry point takes a ``deterministic``
keyword (passing one, of any value, is a ``TypeError`` before any launch); under
``torch.use_deterministic_algorithms(True)`` a backward that computes the
weight gradient on the fused or a row-claims backward (grid kind
``row_claims``) raises PyTorch's alert immediately
before its launch (a ``RuntimeError``; with ``warn_only=True`` a
``UserWarning``, after which it runs; a CUDA Graph warns when it is captured
and not when it is replayed), and every other call (the other bands,
``compute_grad_weight=False``, no rows, the forward, the workspace entry
points, the reset and the support query) runs without an alert or a warning
(strict and ``warn_only``, CUDA Graph capture included) and returns the bits
it returns with the setting off; the deterministic package's FP32 workspace is
refused naming both modes' workspaces.

This suite exercises the package alone, named once through
``tests.cake.test_helpers.cake_package`` (so a vendored copy runs it unchanged), and
imports nothing else from flashinfer.  The comparisons against the
deterministic package (each output's FP64 error at most 1.1 times the
deterministic package's on the same inputs), the facade and the ``flashinfer``
top level are ``test_cake_rmsnorm_silu_gate_nondet_train_api.py``'s.
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path

import numpy
import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_rmsnorm_silu_gate_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")
forward = pkg.cake_rmsnorm_silu_gate_nondet_train_forward
backward = pkg.cake_rmsnorm_silu_gate_nondet_train_backward
workspace_for = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace
workspace_bytes = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
reset_workspace = pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace
gated_rmsnorm = pkg.cake_rmsnorm_silu_gate_nondet

D = 128
EPS = 1e-6
# One BF16 unit in the last place relative to the value (8 significant bits).
BF16_RTOL = 2.0**-7
# The FP32 accumulation term of the weight-gradient budget, relative to the sum
# of the terms' magnitudes (the deterministic suite's).
DW_MAG_TOL = 2.0**-20
REPEATS = 10
DET = "flashinfer.cake_rmsnorm_silu_gate_det_train"
# PyTorch's alert of a backward under its deterministic algorithms.
ALERT = "does not have a deterministic implementation"
# The workspace of a backward module by its slots, ``(buffer, byte offset,
# elements)``: the u64 windows of the exact weight-gradient sum (slots x 7
# windows x 128 columns; on the sparse programs of sm_107a the last slot holds
# the occupancy words), the FP32 class words (128) and the u32 arrival ticket
# and claim counter, each buffer 16-byte aligned; and its bytes.
WORKSPACE_LAYOUTS = {
    4: (("dw_win", 0, 4 * 7 * 128), ("dw_spec", 28672, 128), ("dw_done", 29184, 2)),
    5: (("dw_win", 0, 5 * 7 * 128), ("dw_spec", 35840, 128), ("dw_done", 36352, 2)),
}
MODULE_WORKSPACE_BYTES = {4: 29192, 5: 36360}
# The slots of each static and claimed backward module, by architecture and
# role: on sm_100a and sm_103a 4 for the static backward of an FP32 weight
# gradient and the claimed (``_mid``) backwards; on sm_107a 5 for the static
# backwards and 4 for the claimed backwards.
BACKWARD_SLOTS = {
    "sm_100a": {"bwd_f32": 4, "bwd_bf16_mid": 4, "bwd_f32_mid": 4},
    "sm_103a": {"bwd_f32": 4, "bwd_bf16_mid": 4, "bwd_f32_mid": 4},
    "sm_107a": {"bwd_bf16": 5, "bwd_f32": 5, "bwd_bf16_mid": 4, "bwd_f32_mid": 4},
}
# The torch dtype of each workspace buffer's view.
WORKSPACE_DTYPES = {
    "dw_win": torch.uint64,
    "dw_sums": torch.float64,
    "dw_spec": torch.float32,
    "dw_done": torch.uint32,
    "acc": torch.uint64,
    "flags": torch.uint32,
    "done": torch.uint32,
}
# The row-claims backwards of a BF16 weight gradient (the base and the mid
# backward whose CTAs claim 16-row blocks at run time) and their grid rules:
# at most 4 (``_rc``) and 5 (``_rc_mid``) CTAs per SM; and their workspace: 512
# FP64 column sums (4 slots of 128 columns) and the u32 arrival ticket and
# claim counter, 4104 bytes.
ROW_CLAIMS_GRID_RULES = {
    "bwd_bf16_rc": {"ctas_per_sm": 4, "kind": "row_claims", "rows_per_cta": 16},
    "bwd_bf16_rc_mid": {"ctas_per_sm": 5, "kind": "row_claims", "rows_per_cta": 16},
}
ROW_CLAIMS_LAYOUT = (("dw_sums", 0, 4 * D), ("dw_done", 4096, 2))
ROW_CLAIMS_WORKSPACE_BYTES = 4104
# The fused backward of a BF16 weight gradient on every architecture
# (``bwd_bf16_fused``: the row arithmetic of the deterministic package's BF16
# backward program, the claimed 16-row block schedule, the weight gradient
# added with FP64 atomics; an FP32 weight gradient runs no fused backward) and
# its grid rule: at most 4 CTAs per SM claiming 16-row blocks, the grid kind
# and the rule of ``bwd_bf16_rc``; and its workspace: the row-claims layout,
# 4104 bytes.
FUSED_ROLES = {
    "sm_100a": ("bwd_bf16_fused",),
    "sm_103a": ("bwd_bf16_fused",),
    "sm_107a": ("bwd_bf16_fused",),
}
FUSED_GRID_RULE = {"ctas_per_sm": 4, "kind": "row_claims", "rows_per_cta": 16}
FUSED_LAYOUT = ROW_CLAIMS_LAYOUT
FUSED_WORKSPACE_BYTES = ROW_CLAIMS_WORKSPACE_BYTES
# The repeatable backward of a BF16 weight gradient on every architecture
# (``bwd_bf16_rep``: the deterministic package's fused backward program under
# this package's name, the fused backward's schedule and row arithmetic with
# the weight gradient summed exactly in fixed point, so every output repeats
# its bits; an FP32 weight gradient runs no repeatable backward) and its long
# role (``bwd_bf16_rep_long``: the deterministic package's look-ahead program
# of the same build, the same bits, which runs packed rows from
# ``LONG_ROWS_PER_SM`` rows per SM on) and their grid rule: at most 4 CTAs per
# SM claiming 16-row blocks; and their workspace: 2304 u64 fixed-point digits
# (``acc``), a u32 flag word a column (``flags``) and the u32 arrival ticket
# and claim counter (``done``), 18952 bytes.
REP_ROLES = {
    "sm_100a": ("bwd_bf16_rep", "bwd_bf16_rep_long"),
    "sm_103a": ("bwd_bf16_rep", "bwd_bf16_rep_long"),
    "sm_107a": ("bwd_bf16_rep", "bwd_bf16_rep_long"),
}
# The rows per SM from which the repeatable backward's long role runs packed
# rows of its band (``LONG_BACKWARD_ROWS_PER_SM``): 2560 on sm_100a and
# sm_103a, 1536 on sm_107a, where only the last repeatable band reaches them.
LONG_ROWS_PER_SM = {"sm_100a": 2560, "sm_103a": 2560, "sm_107a": 1536}
REP_GRID_RULE = {"ctas_per_sm": 4, "kind": "claimed_row_blocks", "rows_per_block": 16}
REP_LAYOUT = (("acc", 0, 18 * D), ("flags", 18432, D), ("done", 18944, 2))
REP_WORKSPACE_BYTES = 18952
# The bytes of the backward workspace of each architecture, the largest of its
# backward modules' (one zero-filled workspace serves them all, each module
# laying its buffers over the first bytes): 29192 on sm_100a and sm_103a, 36360
# on sm_107a; and the stride of the device's reserve of capture workspaces
# (256-byte aligned).
WORKSPACE_BYTES = {"sm_100a": 29192, "sm_103a": 29192, "sm_107a": 36360}
RESERVE_STRIDE = {"sm_100a": 29440, "sm_103a": 29440, "sm_107a": 36608}

# The accuracy matrix; test_cake_rmsnorm_silu_gate_nondet_train_api.py runs the same
# cases against the deterministic package.
SHAPES = [
    (1, 1),
    (1, 48),
    (3, 48),
    (5, 48),
    (13, 48),
    (17, 7),
    (255, 2),
    (1031, 48),
    (4097, 16),
    (2, 64),
    (33, 1),
]
LAYOUTS = ["flat", "projection", "head_major", "misaligned"]
EDGE_DATA = ["near_zero", "saturated", "dw_cancel"]
EPSILONS = [1e-5, 1e-3]
ISSUE_TOKENS = [30193, 32737, 32388]
FP32_SHAPES = [(7, 48), (1031, 48), (32737, 48)]


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def deterministic_algorithms_off():
    """Run with ``torch.use_deterministic_algorithms`` off (tests that need it
    on turn it on); restore the caller's setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The module names this package's registry launches, in order."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _routes() -> dict[str, str]:
    return loader.ROUTES[loader.device_arch(torch.cuda.current_device())]


def _module(role: str, rows: int, *, packed: bool = True) -> str:
    """The module ``module_for_rows`` runs for ``role`` and ``rows`` rows on
    this device (``packed``: the forward's inputs hold packed rows)."""
    device_index = torch.cuda.current_device()
    return loader.module_for_rows(
        loader.device_arch(device_index),
        role,
        rows,
        loader.device_sm_count(device_index),
        packed=packed,
    )


def _packed(T: int, H: int, layout: str = "contiguous") -> bool:
    """Whether ``make(T, H, layout=layout)``'s ``x`` and ``z`` hold packed
    rows (row ``n`` at element ``128 n``) as the forward reads them: the
    stride of a single token or a single head is never read, and the host
    copies a misaligned input to packed rows."""
    if layout in ("contiguous", "flat", "misaligned"):
        return True
    if layout == "projection":
        return T == 1
    if layout == "head_major":
        return T == 1 or H == 1
    raise ValueError(layout)


def _backward_packed(T: int, H: int, layout: str = "contiguous") -> bool:
    """Whether the backward kernels read ``make(T, H, layout=layout)``'s
    ``dy``, ``x`` and ``z`` as packed rows (head stride 128, token stride
    ``H * 128``; a single head's stride passed as 128), the test that picks a
    long role (``LONG_BACKWARD_ROWS_PER_SM``): a single token's stride is
    passed as 0, which is not packed, and the host copies a misaligned input
    to packed rows."""
    tokens = T * H if layout == "flat" else T
    if tokens == 1:
        return False
    if layout in ("contiguous", "flat", "misaligned"):
        return True
    if layout == "projection":
        return False
    if layout == "head_major":
        return H == 1
    raise ValueError(layout)


def _forward_module(rows: int, packed: bool = True) -> str:
    """The forward module ``module_for_rows`` runs for ``rows`` rows on this
    device (the program of the architecture's ``FORWARD_BANDS`` band, of its
    ``STRIDED_FORWARD_BANDS`` band where the rows are not ``packed``)."""
    return _module("fwd", rows, packed=packed)


def _backward_module(
    rows: int, dw_dtype: torch.dtype = torch.bfloat16, *, packed: bool = True
) -> str:
    """The backward module ``module_for_rows`` runs for ``rows`` rows and a
    ``dw_dtype`` weight gradient on this device: the program of the role's
    first ``BACKWARD_BANDS`` band that holds the rows (a small-row backward,
    a static, a row-claims, a fused, the repeatable or the claimed backward),
    that band's long role where the rows are ``packed`` (the backward's test,
    :func:`_backward_packed`) and reach its ``LONG_BACKWARD_ROWS_PER_SM``."""
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    return _module(role, rows, packed=packed)


def _chunks(rows: int) -> int:
    """The chunks of the backward with a workspace for ``rows`` rows: 8-row
    blocks, at most 8192 chunks of the fewest consecutive blocks each."""
    blocks = -(-rows // 8)
    return -(-blocks // max(1, -(-blocks // 8192)))


def _band_end(last, sm_count: int) -> int | None:
    """The most rows of a ``BACKWARD_BANDS`` band whose ``last`` is ``last``
    on ``sm_count`` SMs: ``n`` for ``(ROWS, n)`` (a count of rows), the most
    rows with ``rows <= last * sm_count`` for a number (rows per SM), ``None``
    for the last band, which has no end."""
    if last is None:
        return None
    if isinstance(last, tuple):
        assert len(last) == 2 and last[0] == loader.ROWS, last
        return int(last[1])
    return math.floor(last * sm_count)


def _backward_bands(
    arch: str, role: str, sm_count: int
) -> list[tuple[str, int, int | None]]:
    """``(band role, first rows, last rows or None)`` of each
    ``BACKWARD_BANDS`` band of the backward ``role`` (``bwd_bf16`` or
    ``bwd_f32``) on ``arch`` with ``sm_count`` SMs that runs at least one row
    count, in order: a band runs the rows past the ends of the bands before it
    up to its own end (:func:`_band_end`); the last band has no end."""
    bands = []
    covered = 0
    for last, band in loader.BACKWARD_BANDS[arch][role]:
        end = _band_end(last, sm_count)
        if end is None or end > covered:
            bands.append((band, covered + 1, end))
        if end is None:
            break
        covered = max(covered, end)
    return bands


def _long_bands(
    arch: str, role: str, sm_count: int
) -> list[tuple[str, int, int | None]]:
    """``(long role, first rows, last rows or None)`` of each band of
    :func:`_backward_bands` whose role names a cut in
    ``LONG_BACKWARD_ROWS_PER_SM[arch]`` that its rows reach: the long role
    (the band's role and ``LONG``) runs the band's packed rows from ``cut *
    sm_count`` rows (rows per SM) on, up to the band's end; other rows of the
    band run its role."""
    cuts = loader.LONG_BACKWARD_ROWS_PER_SM.get(arch, {})
    found = []
    for band, first, last in _backward_bands(arch, role, sm_count):
        if band in cuts:
            start = max(first, cuts[band] * sm_count)
            if last is None or start <= last:
                found.append((band + loader.LONG, start, last))
    return found


def _device_bands(role: str) -> list[tuple[str, int, int | None]]:
    """:func:`_backward_bands` of the backward ``role`` on this device."""
    device_index = torch.cuda.current_device()
    return _backward_bands(
        loader.device_arch(device_index), role, loader.device_sm_count(device_index)
    )


def _device_long_bands(role: str) -> list[tuple[str, int, int | None]]:
    """:func:`_long_bands` of the backward ``role`` on this device."""
    device_index = torch.cuda.current_device()
    return _long_bands(
        loader.device_arch(device_index), role, loader.device_sm_count(device_index)
    )


def _device_band(band: str) -> tuple[int, int | None] | None:
    """``(first rows, last rows or None)`` of the ``BACKWARD_BANDS`` band of
    the backward role ``band`` (``bwd_bf16_rc``, for one) on this device
    (:func:`_device_bands`), or of the long role ``band``
    (``bwd_bf16_rep_long``: its packed rows, :func:`_device_long_bands`);
    ``None`` where this device routes no such band."""
    role = "bwd_f32" if band.startswith("bwd_f32") else "bwd_bf16"
    for name, first, last in _device_bands(role) + _device_long_bands(role):
        if name == band:
            return first, last
    return None


def _band_tokens(band: str, heads: int = 48) -> int | None:
    """The fewest tokens of ``heads`` heads whose rows lie in the
    ``BACKWARD_BANDS`` band of the backward role ``band`` (``bwd_bf16_rc``,
    for one) on this device; ``None`` where this device routes no such band
    or no such count of rows lies in it."""
    found = _device_band(band)
    if found is None:
        return None
    first, last = found
    tokens = -(-first // heads)
    return tokens if last is None or tokens * heads <= last else None


def _band_shape(band: str) -> tuple[int, int] | None:
    """``(T, H)`` of the fewest rows in the ``BACKWARD_BANDS`` band of the
    backward role ``band`` on this device as tokens of 48 heads, else (a
    band narrower than that) the band's first rows as tokens of one head;
    ``None`` where this device routes no such band.  The first band of the
    role where it has more than one (on sm_107a the BF16 repeatable role
    runs on both sides of the fused band; :func:`_range_shape` takes any); a
    long role's band (:func:`_device_band`) holds packed rows, which these
    shapes are."""
    tokens = _band_tokens(band)
    if tokens is not None:
        return tokens, 48
    tokens = _band_tokens(band, heads=1)
    return None if tokens is None else (tokens, 1)


def _range_shape(first: int, last: int | None) -> tuple[int, int]:
    """``(T, H)`` of the fewest rows from ``first`` to ``last`` (``None``: no
    end) as tokens of 48 heads, else (a range narrower than that) ``first``
    as tokens of one head."""
    tokens = -(-first // 48)
    if last is None or tokens * 48 <= last:
        return tokens, 48
    return first, 1


def _device_band_first(rows: int, dw_dtype: torch.dtype) -> tuple[str, int]:
    """``(role, first rows)`` of the band that runs ``rows`` packed rows of a
    ``dw_dtype`` weight gradient on this device: a long role's
    (:func:`_device_long_bands`) where it holds them, else the band of
    :func:`_device_bands` that holds them."""
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    for _band, first, last in _device_long_bands(role) + _device_bands(role):
        if first <= rows and (last is None or rows <= last):
            return role, first
    raise AssertionError(f"no {role} band of this device holds {rows} rows")


def _varies(name: str) -> bool:
    """Whether the BF16 weight gradient of backward module ``name`` may vary
    in its last bits from run to run: the fused backward's and a row-claims
    backward's (``_fused``, ``_rc`` and ``_rc_mid``, grid kind
    ``row_claims``), whose CTAs claim 16-row blocks at run time and add the
    FP32 runs of their rows with FP64 atomics.  Every other backward repeats
    its bits: the small-row backwards sum on chip in a fixed order, the static
    and the claimed backwards add their partials exactly, as integers, and so
    does the repeatable backward its chunks' terms, as fixed-point digits."""
    return loader.MODULES[name]["grid_rule"]["kind"] == "row_claims"


def _rep(name: str) -> bool:
    """Whether backward module ``name`` is the repeatable backward (the
    deterministic package's fused backward program, ``_rep``, or its
    look-ahead program, ``_rep_long``): its weight gradient's bits are those
    of the rows and the grid (its chunks' rows follow the grid), so a forced
    grid may give other bits than the host's."""
    return loader.MODULES[name]["grid_rule"]["kind"] == "claimed_row_blocks"


def _dw_varies(rows: int, dw_dtype: torch.dtype = torch.bfloat16) -> bool:
    """Whether the ``dw_dtype`` weight gradient of ``rows`` rows on this
    device may vary in its last bits from run to run (:func:`_varies` of the
    module ``module_for_rows`` runs; never for an FP32 weight gradient)."""
    return _varies(_backward_module(rows, dw_dtype))


def _expected_layout(arch: str, role: str) -> tuple:
    """The workspace layout ``(buffer, byte offset, elements)`` of the
    backward module that ``role`` routes on ``arch``: ``ROW_CLAIMS_LAYOUT``
    for a row-claims backward, ``FUSED_LAYOUT`` for a fused backward,
    ``REP_LAYOUT`` for the repeatable backward, the layout of its slots
    (``BACKWARD_SLOTS``, ``WORKSPACE_LAYOUTS``) for a static or a claimed
    backward, none for a small-row backward (one cluster)."""
    if role in ROW_CLAIMS_GRID_RULES:
        return ROW_CLAIMS_LAYOUT
    if role in FUSED_ROLES[arch]:
        return FUSED_LAYOUT
    if role in REP_ROLES[arch]:
        return REP_LAYOUT
    if role in BACKWARD_SLOTS[arch]:
        return WORKSPACE_LAYOUTS[BACKWARD_SLOTS[arch][role]]
    return ()


def fp64_reference(dy, x, z, w, eps):
    """An independent FP64 forward and closed-form backward from the BF16 inputs:
    ``(y, dx, dz, dw)`` in FP64 (unrounded)."""
    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    n = xd * rstd
    s = torch.sigmoid(zd)
    silu = zd * s
    y = n * wd * silu
    a = gd * silu
    dz = gd * n * wd * s * (1.0 + zd * (1.0 - s))
    dn = a * wd
    dx = rstd * dn - xd * rstd**3 * (dn * xd).mean(dim=-1, keepdim=True)
    dw = (a * n).reshape(-1, D).sum(dim=0)
    return y, dx, dz, dw


def eager(x, z, w, eps):
    """The issue's reference operation in FP32, one final cast (the
    deterministic suite's)."""
    xf, zf, wf = x.float(), z.float(), w.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (((xf * rstd) * wf) * torch.nn.functional.silu(zf)).to(x.dtype)


def eager_grads(dy, x, z, w, eps):
    """``(y, dx, dz, dw)`` of :func:`eager` and its PyTorch autograd backward
    for ``dy`` (BF16 inputs, BF16 gradients)."""
    xr = x.detach().clone().requires_grad_(True)
    zr = z.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    y = eager(xr, zr, wr, eps)
    y.backward(dy)
    return y.detach(), xr.grad, zr.grad, wr.grad


def magnitudes(dy, x, z, eps):
    """``sum |a * n|`` per column: the scale of the weight gradient's rounding."""
    terms = (dy.double() * torch.nn.functional.silu(z.double())) * (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + eps)
    )
    return terms.abs().reshape(-1, D).sum(0)


def assert_bf16_close(got, want, *, what, scale=None):
    """``got`` (BF16) within one BF16 ulp of ``want`` plus an absolute floor of
    1/64 ulp of the tensor's largest magnitude (cancellation in ``dx``)."""
    want = want.double()
    floor = (scale if scale is not None else want.abs().max().item()) * BF16_RTOL / 64
    err = (got.double() - want).abs()
    bound = want.abs() * BF16_RTOL + floor
    bad = err > bound
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} elements beyond one BF16 ulp; "
        f"worst err {err.max().item():.3e}"
    )


def assert_dw_within(dw, want, mag, *, what):
    """The weight-gradient budget: ``|dw - want| <= |want| * 2^-7 + mag * 2^-20``."""
    err = (dw.double() - want.double()).abs()
    bound = want.double().abs() * BF16_RTOL + mag * DW_MAG_TOL
    bad = err > bound
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} of {bad.numel()} columns beyond the budget; "
        f"worst excess {(err - bound).max().item():.3e}"
    )


def assert_dw_repeats(dw, first, want, mag, *, varies, what):
    """A weight gradient ``dw`` of a call that ran before as ``first``: the
    same bits where its band repeats them; where it may vary in its last bits
    (``varies``: a row-claims band, :func:`_varies`) within the budget of the
    reference ``want`` (:func:`assert_dw_within`) instead."""
    if varies:
        assert_dw_within(dw, want, mag, what=what)
    else:
        assert torch.equal(dw, first), f"{what}: not the bits of the first call"


def make(T, H, *, layout="contiguous", data="normal", seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=g).bfloat16()

    if layout == "contiguous":
        x, z, dy = randn(T, H, D), randn(T, H, D), randn(T, H, D)
    elif layout == "flat":
        x, z, dy = randn(T * H, D), randn(T * H, D), randn(T * H, D)
    elif layout == "projection":
        # z (and dy) as the gate columns of wider projections, read in place.
        x = randn(T, H, D)
        z = randn(T, 2 * H * D + 16)[:, H * D : 2 * H * D].view(T, H, D)
        dy = randn(T, 2 * H * D + 16)[:, : H * D].view(T, H, D)
    elif layout == "head_major":
        x = randn(H, T, D).permute(1, 0, 2)
        z = randn(H, T, D).permute(1, 0, 2)
        dy = randn(T, H, D)
    elif layout == "misaligned":
        # Storage offsets of 8 elements: materialized once by the host.
        x = randn(T * H * D + 8)[8:].view(T, H, D)
        z = randn(T * H * D + 8)[8:].view(T, H, D)
        dy = randn(T * H * D + 8)[8:].view(T, H, D)
    else:
        raise ValueError(layout)
    if data == "near_zero":
        flat = torch.arange(T * H, device=device).view(T, H)
        xv = x.view(T, H, D)
        xv[flat % 3 == 0] = 0
        xv[flat % 3 == 1] = 2.0**-60
    elif data == "dw_cancel":
        # Rows in pairs with the same x and z and opposite dy (but for a small
        # perturbation): each column's weight-gradient terms cancel to a small
        # remainder of their magnitudes.
        xv, zv, gv = x.reshape(-1, D), z.reshape(-1, D), dy.reshape(-1, D)
        pairs = xv.shape[0] // 2
        xv[1 : 2 * pairs : 2] = xv[0 : 2 * pairs : 2]
        zv[1 : 2 * pairs : 2] = zv[0 : 2 * pairs : 2]
        small = (2.0**-6 * torch.randn(pairs, D, device=device, generator=g)).bfloat16()
        gv[1 : 2 * pairs : 2] = -gv[0 : 2 * pairs : 2] + small
    elif data == "saturated":
        values = torch.tensor(
            [-1e4, -100.0, -90.0, -30.0, -10.0, 10.0, 30.0, 90.0, 100.0, 1e4],
            device=device,
        ).bfloat16()
        pick = torch.randint(0, 10, z.shape, device=device, generator=g)
        z.copy_(
            torch.where(
                torch.rand(z.shape, device=device, generator=g) < 0.5, values[pick], z
            )
        )
    w = (1.0 + 0.5 * torch.randn(D, device=device, generator=g)).bfloat16()
    return x, z, w, dy


def check_eager(y, dx, dz, dw, x, z, w, dy, eps):
    """``y``, ``dx``, ``dz`` and ``dw`` against the explicit eager operation
    (:func:`eager_grads`) with the deterministic suite's bounds: one BF16 ulp
    of the eager output plus 1/64 ulp of the FP64 reference's largest
    magnitude, and for ``dw`` ``|dw_eager| * 2^-7 + sum|terms| * 2^-20``."""
    y64, dx64, dz64, _ = fp64_reference(dy, x, z, w, eps)
    mag = magnitudes(dy, x, z, eps)
    ye, dxe, dze, dwe = eager_grads(dy, x, z, w, eps)
    assert_bf16_close(y, ye, what="y vs eager", scale=y64.abs().max().item())
    assert_bf16_close(dz, dze, what="dz vs eager", scale=dz64.abs().max().item())
    assert_bf16_close(dx, dxe, what="dx vs eager", scale=dx64.abs().max().item())
    assert_dw_within(dw, dwe, mag, what="dw vs eager")


def check_all(x, z, w, dy, eps, *, dw_dtype=torch.bfloat16, repeats=REPEATS):
    """``repeats`` forward and backward launches against the FP64 reference
    and the eager operation (:func:`check_eager`, the first launch),
    ``y``, ``dx`` and ``dz`` the same bits on every launch, and so the weight
    gradient, also when accumulated into a nonzero buffer, where its band
    repeats it (:func:`_dw_varies`); in a row-claims band every launch's
    weight gradient, alone and accumulated, lies within the FP64 budget and
    the eager operation's instead; returns ``(y, dx, dz, dw)`` of the first
    launch."""
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, eps)
    mag = magnitudes(dy, x, z, eps)
    varies = _dw_varies(x.numel() // D, dw_dtype)
    dw_eager = eager_grads(dy, x, z, w, eps)[3] if varies else None
    repeated = (
        ("y", "dx", "dz") if varies else ("y", "dx", "dz", "dw", "accumulated dw")
    )

    def out(fill=None):
        if fill is not None:
            return torch.full((D,), fill, dtype=dw_dtype, device=x.device)
        return (
            None
            if dw_dtype == torch.bfloat16
            else torch.empty(D, dtype=dw_dtype, device=x.device)
        )

    first = None
    for i in range(repeats):
        y = forward(x, z, w, eps)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape
        dx, dz, dw = backward(dy, x, z, w, eps, grad_weight=out())
        assert dx.dtype == dz.dtype == torch.bfloat16 and dw.dtype == dw_dtype
        assert dx.shape == dz.shape == x.shape and dw.shape == (D,)
        main = out(0.25)
        backward(dy, x, z, w, eps, grad_weight=main, accumulate_grad_weight=True)
        if first is None:
            first = y, dx, dz, dw, main
            assert_bf16_close(y, y64, what="y vs FP64")
            assert_bf16_close(dz, dz64, what="dz vs FP64")
            assert_bf16_close(dx, dx64, what="dx vs FP64")
            assert_dw_within(dw, dw64, mag, what="dw vs FP64")
            assert_dw_within(main, 0.25 + dw64, mag, what="accumulated dw vs FP64")
            check_eager(y, dx, dz, dw, x, z, w, dy, eps)
        elif varies:
            assert_dw_within(dw, dw64, mag, what=f"dw vs FP64, launch {i}")
            assert_dw_within(dw, dw_eager, mag, what=f"dw vs eager, launch {i}")
            assert_dw_within(
                main, 0.25 + dw64, mag, what=f"accumulated dw vs FP64, launch {i}"
            )
        for what, got, want in zip(
            ("y", "dx", "dz", "dw", "accumulated dw"),
            (y, dx, dz, dw, main),
            first,
            strict=True,
        ):
            if what in repeated:
                assert torch.equal(got, want), f"{what}: launch {i} vs the first launch"
    return first[:4]


@pytest.mark.parametrize("T,H", SHAPES)
def test_forward_backward_shapes(T, H):
    x, z, w, dy = make(T, H)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts(layout):
    x, z, w, dy = make(65, 48, layout=layout, seed=1)
    check_all(x, z, w, dy, EPS)


# Inputs in the bands of the backwards past the small-row and the static bands
# (band, layout), at the fewest tokens of 64 heads in the band on this device
# (``_bulk_band_shape``; ``one_token``: one token of as many heads as the
# band's first rows).  On sm_100a and sm_103a the claimed backwards (``_mid``)
# and the row-claims mid backward (``bwd_bf16_rc_mid``) warm a block's rows in
# L2 with one bulk prefetch per tensor only where dy, x and z share one packed
# row stride (head stride 128, token stride H * 128); the other layouts here
# (``z`` and ``dy`` the gate columns of wider projections, ``x`` and ``z``
# head-major) fall back to a prefetch per thread.
# One token of many heads (its token stride passed as 0) and contiguous tokens
# are packed rows.  The row-claims backward (``bwd_bf16_rc``) warms no rows;
# the fused and the repeatable backwards (``bwd_bf16_fused``,
# ``bwd_bf16_rep``) warm the next block of a claim on every architecture (one
# bulk prefetch per tensor for packed rows, else a prefetch per thread); all
# run the same checks.  The repeatable backward's long role
# (``bwd_bf16_rep_long``, which looks ahead along a claim) runs packed rows
# only: its band takes the contiguous input alone.
BULK_BANDS = (
    "bwd_bf16_rc",
    "bwd_bf16_rc_mid",
    "bwd_bf16_mid",
    "bwd_f32_mid",
    "bwd_bf16_fused",
    "bwd_bf16_rep",
    "bwd_bf16_rep_long",
)
BULK_BAND_CASES = [
    pytest.param(band, layout, id=f"{band}-{layout}")
    for band in BULK_BANDS
    for layout in ("projection", "head_major", "one_token", "contiguous")
]


def _bulk_band_shape(band: str, layout: str) -> tuple[int, int, str] | None:
    """``(T, H, make's layout)`` of the ``BULK_BAND_CASES`` case ``(band,
    layout)`` on this device: the fewest tokens (at least 2) of 64 heads whose
    rows lie in the ``BACKWARD_BANDS`` band of the backward role ``band`` (or
    of its long role, :func:`_device_band`), or for ``one_token`` one
    contiguous token of the band's first rows of heads; ``None`` where this
    device routes no such band or no such shape lies in it (a long role's
    band holds packed rows only, :func:`_backward_packed`)."""
    found = _device_band(band)
    if found is None:
        return None
    first, last = found
    if layout == "one_token":
        T, H, layout = 1, first, "contiguous"
    else:
        T, H = max(2, -(-first // 64)), 64
        if last is not None and last < T * H:
            return None
    if band.endswith(loader.LONG) and not _backward_packed(T, H, layout):
        return None
    return T, H, layout


@pytest.mark.parametrize("band,layout", BULK_BAND_CASES)
def test_layouts_in_the_row_claims_and_claimed_backward_bands(band, layout, launches):
    """``BULK_BAND_CASES``: the band's program runs the rows
    (``module_for_rows``), within the FP64 budgets and the eager operation's,
    ``y``, ``dx`` and ``dz`` (and ``dw`` where the band repeats it) the same
    bits on every launch (``check_all``), and ``dx`` and ``dz`` the bits of the
    same rows packed as ``[T * H, 1, 128]`` (which the bulk prefetch warms on
    sm_100a and sm_103a; sm_107a's programs take no bulk prefetch and run the
    same checks), so is ``dw`` where the band repeats it, else within the FP64
    budget."""
    shape = _bulk_band_shape(band, layout)
    if shape is None:
        pytest.skip(f"this device routes no {band} band with a {layout} input")
    T, H, layout = shape
    rows = T * H
    dw_dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    name = _backward_module(rows, dw_dtype, packed=_backward_packed(T, H, layout))
    assert name == _routes()[band], (rows, name)
    x, z, w, dy = make(T, H, layout=layout, seed=90)
    _, dx, dz, dw = check_all(x, z, w, dy, EPS, dw_dtype=dw_dtype)
    fwd = _forward_module(rows, _packed(T, H, layout))
    assert launches == [fwd, name, name] * REPEATS, launches
    gw = None
    if dw_dtype != torch.bfloat16:
        gw = torch.empty(D, dtype=dw_dtype, device=x.device)
    packed = [t.contiguous().view(rows, 1, D) for t in (dy, x, z)]
    again = backward(*packed, w, EPS, grad_weight=gw)
    for what, a, b in zip(("dx", "dz"), again[:2], (dx, dz), strict=True):
        assert torch.equal(a.reshape(-1, D), b.reshape(-1, D)), (what, layout)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    assert_dw_repeats(
        again[2], dw, dw64, mag, varies=_varies(name), what=f"packed dw ({layout})"
    )


@pytest.mark.parametrize("data", EDGE_DATA)
def test_edge_values(data):
    x, z, w, dy = make(37, 48, data=data, seed=2)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("eps", EPSILONS)
def test_epsilon(eps):
    x, z, w, dy = make(19, 48, seed=3)
    check_all(x, z, w, dy, eps)


@pytest.mark.parametrize("T", ISSUE_TOKENS)
def test_issue_shapes(T):
    x, z, w, dy = make(T, 48, seed=4)
    check_all(x, z, w, dy, EPS)


@pytest.mark.parametrize("T,H", FP32_SHAPES)
def test_fp32_weight_gradient(T, H):
    x, z, w, dy = make(T, H, seed=5)
    check_all(x, z, w, dy, EPS, dw_dtype=torch.float32)


def test_weight_gradient_accumulation():
    """Two backwards accumulating into one FP32 or BF16 ``grad_weight``: every
    repeat within the budget of two roundings, and the same bits on every
    repeat but for a BF16 accumulator in a row-claims band
    (:func:`_dw_varies`); autograd's ``.grad`` over two microbatches the sums
    of the functional backward's gradients (``dw`` where its band repeats it;
    in a row-claims band within the budget of two BF16 weight gradients, as
    everywhere)."""
    x, z, w, dy = make(1031, 48, seed=6)
    x2, z2, _, dy2 = make(1031, 48, seed=7)
    varies = _dw_varies(1031 * 48)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    dw64b = fp64_reference(dy2, x2, z2, w, EPS)[3]
    mag = magnitudes(dy, x, z, EPS) + magnitudes(dy2, x2, z2, EPS)
    first = 0.25 + dw64
    want = first + dw64b
    # Two roundings into the accumulator's dtype, each within its relative unit.
    bound = (first.abs() + want.abs()) * BF16_RTOL + mag * DW_MAG_TOL
    for dtype in (torch.float32, torch.bfloat16):
        results = []
        for _ in range(3):
            main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
            for args in ((dy, x, z), (dy2, x2, z2)):
                backward(*args, w, EPS, grad_weight=main, accumulate_grad_weight=True)
            results.append(main)
        for main in results:
            assert ((main.double() - want).abs() <= bound).all(), dtype
        if dtype == torch.float32:
            assert torch.allclose(main.double(), want, rtol=1e-5, atol=1e-3)
        if dtype == torch.float32 or not varies:
            for other in results[1:]:
                assert torch.equal(other, results[0]), dtype
    # autograd accumulation into .grad over two microbatches: the sums of the
    # functional backward's gradients
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    gated_rmsnorm(xa, za, wa, EPS).backward(dy)
    dx1, dz1, dw1 = backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, dx1 + dx1)
    assert torch.equal(za.grad, dz1 + dz1)
    if not varies:
        assert torch.equal(wa.grad, dw1 + dw1)
    # Two BF16 weight gradients within budget, summed in BF16 by autograd.
    mag1 = magnitudes(dy, x, z, EPS)
    err = (wa.grad.double() - 2 * dw64).abs()
    assert (err <= (2 * dw64).abs() * 2.0**-6 + mag1 * 2.0**-18).all()


def test_compute_grad_weight_false_touches_no_weight_state(monkeypatch):
    x, z, w, dy = make(257, 48, seed=12)
    # dx and dz do not depend on whether the weight gradient is computed
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(257 * 48, x.device)
    main = torch.full((D,), 3.0, dtype=torch.float32, device="cuda")
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 3.0))
    assert not bool(ws.any())
    # without a workspace: the launch still takes the arrival ticket (and the
    # claimed and the row-claims backwards claim their chunks or blocks), on
    # the stream's workspace of the BF16 bands' module (its buffers, left zero)
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _backward_module(257 * 48),
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    views = pkg._STREAM_WORKSPACES[key]
    arch = loader.device_arch(torch.cuda.current_device())
    layout = _expected_layout(arch, loader.MODULES[key[2]]["role"])
    assert layout and sorted(views) == sorted(b for b, _, _ in layout), key
    # (through bytes: uint64 tensors take few operations)
    assert not any(bool(view.view(torch.uint8).any()) for view in views.values())
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert w.grad is None


# The band roles of the backwards that run on a workspace: the static
# backwards (the call roles), the row-claims backwards of a BF16 weight
# gradient, the claimed backwards and the fused and repeatable backwards, and
# the repeatable backward's long role (on its packed rows).
GRID_BANDS = (
    "bwd_bf16",
    "bwd_f32",
    "bwd_bf16_rc",
    "bwd_bf16_rc_mid",
    "bwd_bf16_mid",
    "bwd_f32_mid",
    "bwd_bf16_fused",
    "bwd_bf16_rep",
    "bwd_bf16_rep_long",
)


@pytest.mark.parametrize("ctas", [1, 3, 7])
@pytest.mark.parametrize("band", GRID_BANDS)
def test_backward_at_grids_smaller_than_the_chunks(band, ctas, launches, monkeypatch):
    """The host's grid forced to ``ctas`` CTAs for the fewest rows of the
    ``BACKWARD_BANDS`` band of ``band`` (or of its long role, packed rows) on
    this device (:func:`_band_shape`; skipped where this device routes no
    such band), so every CTA runs many
    chunks or row blocks: ``dx`` and ``dz`` the bits of the host's grid; the
    weight gradient of the band's dtype, alone and accumulated, within the
    FP64 budget, and the bits of the host's grid where the band repeats them
    (:func:`assert_dw_repeats`; within the budget only where it may vary; the
    repeatable backward's, whose chunks follow the grid, the bits of a second
    launch at the same grid instead);
    ``compute_grad_weight=False`` (the BF16 bands' module) leaves
    ``grad_weight`` as it is, ``dx`` and ``dz`` that module's bits at the
    host's grid; the workspace is zero after every launch."""
    shape = _band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    rows = T * H
    dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    name = _backward_module(rows, dtype)
    assert name == _routes()[band], (rows, name)
    varies = _varies(name)
    x, z, w, dy = make(T, H, seed=12)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    mag = magnitudes(dy, x, z, EPS)
    dw_d = torch.empty(D, dtype=dtype, device="cuda")
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS, grad_weight=dw_d)
    main_d = torch.full((D,), 0.25, dtype=dtype, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main_d, accumulate_grad_weight=True)
    # compute_grad_weight=False runs the BF16 bands' module, whose dx and dz
    # are another program's where that is not the band's module (an FP32
    # band's rows in the fused band): the bits of its call at the host's grid
    dx_n, dz_n = dx_d, dz_d
    if _backward_module(rows) != name:
        dx_n, dz_n, _ = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    ws = workspace_for(rows, x.device)
    monkeypatch.setattr(pkg, "backward_ctas", lambda name, rows, sm_count: ctas)
    launches.clear()
    what = f"{band} at {ctas} CTAs"
    dw = torch.empty(D, dtype=dtype, device="cuda")
    dx, dz, _ = backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what
    if _rep(name):
        # the bits of the forced grid: a second launch at it
        dw_d = torch.empty(D, dtype=dtype, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=dw_d, workspace=ws)
        main_d = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main_d,
            accumulate_grad_weight=True,
            workspace=ws,
        )
    assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"dw {what}")
    assert_dw_within(dw, dw64, mag, what=f"dw {what}")
    assert not bool(ws.any()), what
    main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
    backward(
        dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True, workspace=ws
    )
    assert_dw_repeats(
        main, main_d, 0.25 + dw64, mag, varies=varies, what=f"accumulated dw {what}"
    )
    assert_dw_within(main, 0.25 + dw64, mag, what=f"accumulated dw {what}")
    assert not bool(ws.any()), what
    main = torch.full((D,), 3.0, dtype=dtype, device="cuda")
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert dw is None
    assert torch.equal(dx, dx_n) and torch.equal(dz, dz_n), what
    assert torch.equal(main, torch.full_like(main, 3.0)), what
    assert not bool(ws.any()), what
    # a call with compute_grad_weight=False runs the BF16 bands' module
    # whatever the dtype of grad_weight (it writes no weight gradient)
    repeats = 4 if _rep(name) else 2
    assert launches == [name] * repeats + [_backward_module(rows)], launches


@pytest.mark.parametrize("shape", [(0, 48, D), (5, 0, D), (0, D)])
def test_empty_work_keeps_the_accumulator(shape, launches):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    ws = workspace_for(0, x.device)
    assert forward(x, x, w).shape == x.shape
    for dtype in (torch.bfloat16, torch.float32):
        main = torch.full((D,), 1.5, dtype=dtype, device="cuda")
        dx, dz, dw = backward(
            x, x, x, w, grad_weight=main, accumulate_grad_weight=True, workspace=ws
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5)), dtype
        assert dx.shape == dz.shape == x.shape
        backward(x, x, x, w, grad_weight=main)
        assert torch.equal(main, torch.zeros_like(main)), dtype
    _, _, dw = backward(x, x, x, w)
    assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
    assert launches == []
    assert not bool(ws.any())


def test_workspace_reused_across_shapes_without_reset():
    """One zero-filled byte workspace of the architecture's
    ``WORKSPACE_BYTES`` (the largest of its backward modules' workspaces, each
    module's layout over the first bytes: a static or a claimed backward's by
    its slots, a row-claims backward's ``ROW_CLAIMS_LAYOUT``, a fused
    backward's ``FUSED_LAYOUT``, the repeatable backward's and its long
    role's ``REP_LAYOUT``) serves every
    backward module and row count: each backward on it is within the FP64
    budgets, ``dx`` and ``dz`` the bits of the same call on the stream's own
    workspace, so is the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`), and leaves it zero.  The shapes run every
    ``BACKWARD_BANDS`` band of both weight-gradient dtypes on this device
    and every long role's packed rows (the fewest rows of each band no
    listed shape runs, :func:`_range_shape`), the small-row backwards among
    them."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    assert ws.dtype == torch.uint8 and ws.is_contiguous() and not bool(ws.any())
    assert ws.data_ptr() % 16 == 0
    arch = loader.device_arch(torch.cuda.current_device())
    nbytes = workspace_bytes(1, device)
    assert nbytes == WORKSPACE_BYTES[arch], (arch, nbytes)
    assert ws.numel() == nbytes == workspace_bytes(10**7, device)
    largest = 0
    for role, name in loader.ROUTES[arch].items():
        if not role.startswith("bwd"):
            continue
        layout = _expected_layout(arch, role)
        assert loader.workspace_layout(name) == layout, role
        module_bytes = loader.workspace_bytes(name)
        if role in loader.BAND_ROLES:
            assert layout == () and module_bytes == 0, role
        elif role in ROW_CLAIMS_GRID_RULES:
            assert module_bytes == ROW_CLAIMS_WORKSPACE_BYTES, role
        elif role in FUSED_ROLES[arch]:
            assert module_bytes == FUSED_WORKSPACE_BYTES, role
        elif role in REP_ROLES[arch]:
            assert module_bytes == REP_WORKSPACE_BYTES, role
        else:
            slots = BACKWARD_SLOTS[arch][role]
            assert module_bytes == MODULE_WORKSPACE_BYTES[slots], role
        largest = max(largest, module_bytes)
    assert largest == nbytes, (arch, largest, nbytes)
    cases = [
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
        (129, 48, torch.float32, False),
    ]
    bands = {
        band
        for role in ("bwd_bf16", "bwd_f32")
        for band, _, _ in _device_bands(role) + _device_long_bands(role)
    }
    listed = {_device_band_first(T * H, dtype) for T, H, dtype, _ in cases}
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        for _band, first, last in _device_bands(role) + _device_long_bands(role):
            if (role, first) not in listed:
                T, H = _range_shape(first, last)
                cases.append((T, H, dtype, False))
    ran = set()
    for i, (T, H, dtype, accumulate) in enumerate(cases):
        name = _backward_module(T * H, dtype)
        ran.add(loader.MODULES[name]["role"])
        x, z, w, dy = make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0
        gw = (
            torch.full((D,), start, dtype=dtype, device=device)
            if accumulate or dtype != torch.bfloat16
            else None
        )
        dx, dz, dw = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=gw,
            accumulate_grad_weight=accumulate,
            workspace=ws,
        )
        torch.cuda.synchronize()
        assert not bool(ws.any()), (T, H, dtype)
        _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
        assert_bf16_close(dx, dx64, what=f"dx T={T} H={H}")
        assert_bf16_close(dz, dz64, what=f"dz T={T} H={H}")
        assert dw.dtype == dtype
        want = dw64 + start
        mag = magnitudes(dy, x, z, EPS)
        assert_dw_within(dw, want, mag, what=f"dw T={T} H={H} {dtype}")
        # the same call on the stream's workspace: the same bits (the weight
        # gradient within the budget where its band may vary it)
        gw2 = None if gw is None else torch.full_like(gw, start)
        again = backward(
            dy, x, z, w, EPS, grad_weight=gw2, accumulate_grad_weight=accumulate
        )
        for a, b in zip(again[:2], (dx, dz), strict=True):
            assert torch.equal(a, b), (T, H, dtype)
        assert_dw_repeats(
            again[2],
            dw,
            want,
            mag,
            varies=_varies(name),
            what=f"dw on the stream's workspace T={T} H={H} {dtype}",
        )
    assert ran == bands, (arch, ran, bands)


def _unfit_workspaces(nbytes: int) -> list[torch.Tensor]:
    """Buffers that are not a backward workspace of ``nbytes`` bytes: too
    small, strided, misaligned, of another dtype."""
    big = torch.zeros(nbytes + 64, dtype=torch.uint8, device="cuda")
    return [
        torch.zeros(nbytes - 1, dtype=torch.uint8, device="cuda"),
        torch.zeros(2 * nbytes, dtype=torch.uint8, device="cuda")[::2],
        big[1 : 1 + nbytes],
        big[8 : 8 + nbytes],
        torch.zeros(nbytes, dtype=torch.float64, device="cuda"),
        torch.zeros(nbytes, dtype=torch.int8, device="cuda"),
    ]


def test_workspace_errors():
    x, z, w, dy = make(65, 48, seed=13)
    nbytes = workspace_for(1, x.device).numel()
    # an FP32 buffer (the deterministic package's kind)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            backward(dy, x, z, w, EPS, workspace=bad)
        assert not bool(bad.any())


@pytest.mark.parametrize("role", ["bwd_bf16", "bwd_f32"])
def test_workspace_errors_in_every_band(role, launches):
    """In every ``BACKWARD_BANDS`` band of ``role`` on this device (the
    small-row, the static, the row-claims and the claimed backwards, whatever
    part of the workspace the band's module uses) and on the packed rows of
    its long roles (:func:`_device_long_bands`), a caller workspace that is
    not a contiguous ``torch.uint8`` buffer of at least ``workspace_bytes``
    bytes (one byte short, strided, misaligned, of another dtype:
    :func:`_unfit_workspaces`) raises ``ValueError`` naming ``uint8``,
    launches nothing and is left as it is."""
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16
    nbytes = workspace_for(1, torch.device("cuda")).numel()
    bands = _device_bands(role)
    assert bands, role
    for band, first, last in bands + _device_long_bands(role):
        shape = _range_shape(first, last)
        x, z, w, dy = make(*shape, seed=17)
        assert _backward_module(shape[0] * shape[1], dw_dtype) == _routes()[band]
        for bad in _unfit_workspaces(nbytes):
            gw = torch.empty(D, dtype=dw_dtype, device="cuda")
            with pytest.raises(ValueError, match="uint8"):
                backward(dy, x, z, w, EPS, grad_weight=gw, workspace=bad)
            torch.cuda.synchronize()
            assert not bool(bad.any()), (band, bad.dtype, bad.numel())
    assert launches == [], launches


def test_reset_workspace_clears_nonzero_scratch():
    """A caller workspace whose integer windows, class words, arrival ticket
    and claim counter hold garbage (written by something other than the
    backward) is zeroed by ``reset_workspace``; backward launches at two
    different shapes and weight-gradient dtypes then run on it in sequence,
    within the FP64 budget and the bits of the same calls on a fresh
    workspace (the weight gradient within the budget instead where its band
    may vary it, :func:`assert_dw_repeats`), and leave it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    g = torch.Generator(device="cuda").manual_seed(60)
    ws.copy_(
        torch.randint(1, 256, ws.shape, device=device, generator=g).to(torch.uint8)
    )
    # a stale arrival ticket and a stale claim count, no byte of them zero, in
    # the layout of each backward module of the architecture
    arch = loader.device_arch(torch.cuda.current_device())
    offsets = {
        offset
        for role in loader.ROUTES[arch]
        if role.startswith("bwd")
        for buffer, offset, _ in _expected_layout(arch, role)
        if buffer == "dw_done"
    }
    assert offsets, arch
    for offset in sorted(offsets):
        ws[offset : offset + 8].view(torch.int32).fill_(0x03030303)
    assert bool(ws.ne(0).all())
    reset_workspace(ws)
    torch.cuda.synchronize()
    assert not bool(ws.any())
    # 149 x 7 = 1043 rows: past every small-row backward's band, so both calls
    # use the workspace
    for i, (T, H, dtype) in enumerate(
        ((1031, 48, torch.bfloat16), (149, 7, torch.float32))
    ):
        x, z, w, dy = make(T, H, seed=61 + i)
        gw = (
            None
            if dtype == torch.bfloat16
            else torch.empty(D, dtype=dtype, device=device)
        )
        dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=gw, workspace=ws)
        _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
        assert_bf16_close(dx, dx64, what=f"dx T={T} H={H}")
        assert_bf16_close(dz, dz64, what=f"dz T={T} H={H}")
        assert dw.dtype == dtype
        mag = magnitudes(dy, x, z, EPS)
        assert_dw_within(dw, dw64, mag, what=f"dw T={T} H={H} {dtype}")
        gw2 = None if gw is None else torch.empty_like(gw)
        fresh = backward(
            dy, x, z, w, EPS, grad_weight=gw2, workspace=workspace_for(1, device)
        )
        for a, b in zip(fresh[:2], (dx, dz), strict=True):
            assert torch.equal(a, b), (T, H)
        assert_dw_repeats(
            fresh[2],
            dw,
            dw64,
            mag,
            varies=_dw_varies(T * H, dtype),
            what=f"dw on a fresh workspace T={T} H={H} {dtype}",
        )
        torch.cuda.synchronize()
        assert not bool(ws.any()), (T, H)


def test_reset_workspace_of_a_zero_workspace_is_a_no_op(launches):
    """Resetting a zero workspace keeps it zero and in place, launches no module
    and does not synchronize the host; the next backward is unaffected."""
    x, z, w, dy = make(129, 48, seed=62)
    ws = workspace_for(129 * 48, x.device)
    ptr = ws.data_ptr()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reset_workspace(ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert ws.data_ptr() == ptr and not bool(ws.any())
    assert launches == []
    _, _, dw = backward(dy, x, z, w, EPS, workspace=ws)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    assert_dw_within(dw, dw64, mag, what="dw after a reset of a zero workspace")
    torch.cuda.synchronize()
    assert not bool(ws.any())


def test_reset_workspace_errors():
    nbytes = workspace_for(1, torch.device("cuda")).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        reset_workspace(fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            reset_workspace(bad)
        assert not bool(bad.any())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(nbytes, dtype=torch.uint8))
    ws = workspace_for(1, torch.device("cuda"))
    # the mode is in the name: a mode keyword is a TypeError
    for value in (False, True, None):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
                ws, deterministic=value
            )


def test_two_streams_with_their_own_workspaces():
    cases = [make(4097, 48, seed=30), make(1031, 16, seed=31)]
    # dx and dz of one launch on the current stream
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    references = [
        (fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS))
        for x, z, w, dy in cases
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [workspace_for(1, torch.device("cuda")) for _ in streams]
    outputs: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                outputs[i].append(backward(dy, x, z, w, EPS, workspace=workspaces[i]))
    torch.cuda.synchronize()
    for i in range(2):
        dx_d, dz_d, dw_d = refs[i]
        dw64, mag = references[i]
        varies = _dw_varies(cases[i][0].numel() // D)
        assert_dw_within(dw_d, dw64, mag, what=f"stream {i} dw")
        for dx, dz, dw in outputs[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"stream {i} dw")
        assert not bool(workspaces[i].any()), i


def test_side_stream_and_no_host_sync():
    x, z, w, dy = make(2049, 48, seed=9)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    ws = workspace_for(x.shape[0] * x.shape[1], x.device)
    # Load both modules before the sync-checked region; the bits of y, dx and dz
    # on the current stream.
    want_y = forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS, workspace=ws)
    assert_dw_within(dw_d, dw64, mag, what="dw")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            y = forward(x, z, w, EPS)
            got = backward(dy, x, z, w, EPS, workspace=ws)
            fresh = backward(dy, x, z, w, EPS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(y, want_y)
    varies = _dw_varies(x.numel() // D)
    for dx, dz, dw in (got, fresh):
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
        assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what="side stream dw")
    assert not bool(ws.any())


def test_cuda_graph_capture_and_replay():
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    ws = workspace_for(T * H, x.device)

    def step():
        forward(xs, zs, w, EPS, out=y)
        backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
        # FP32 main-grad accumulation with the workspace allocated in the graph
        backward(dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    main.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    total = torch.zeros(D, dtype=torch.float64, device="cuda")
    total_mag = torch.zeros(D, dtype=torch.float64, device="cuda")
    main_e = torch.zeros(D, dtype=torch.float32, device="cuda")
    varies = _dw_varies(T * H)
    for k in range(3):
        xn, zn, _, dyn = make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        # the replay against eager launches on the same inputs: the same bits
        # (the BF16 weight gradient within the budget where its band may vary
        # it)
        assert torch.equal(y, forward(xn, zn, w, EPS)), k
        dx_e, dz_e, dw_e = backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_e) and torch.equal(dz, dz_e), k
        dw64, mag = fp64_reference(dyn, xn, zn, w, EPS)[3], magnitudes(dyn, xn, zn, EPS)
        assert_dw_repeats(dw, dw_e, dw64, mag, varies=varies, what=f"replay {k} dw")
        backward(dyn, xn, zn, w, EPS, grad_weight=main_e, accumulate_grad_weight=True)
        assert torch.equal(main, main_e), k
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not bool(ws.any()), k
        total += dw64
        total_mag += mag
        # three FP32 roundings of the accumulator at most, plus the sums' budget
        err = (main.double() - total).abs()
        assert (err <= total.abs() * 2.0**-21 + total_mag * DW_MAG_TOL).all(), k


def _kept_graph() -> torch.cuda.CUDAGraph:
    try:
        return torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        pytest.skip("listing graph nodes needs torch.cuda.CUDAGraph(keep_graph=True)")


def _graph_node_types(graph: torch.cuda.CUDAGraph) -> list[str]:
    """The node types of a graph captured with ``keep_graph=True``."""
    driver = pytest.importorskip("cuda.bindings.driver")
    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(handle)
    _, nodes, count = driver.cuGraphGetNodes(handle, count)
    return [
        driver.cuGraphNodeGetType(node)[1].name.replace("CU_GRAPH_NODE_TYPE_", "")
        for node in nodes[:count]
    ]


@pytest.fixture
def reserve(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fresh reserve and capture tables; the reserve of the current device,
    filled and seen ready by eager calls without a workspace."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    # 1056 rows: past every small-row backward band, a backward with a workspace
    x, z, w, dy = make(22, 48, seed=69)
    backward(dy, x, z, w, EPS)  # loads the module, fills the reserve
    torch.cuda.synchronize()
    backward(dy, x, z, w, EPS)  # sees the fill complete
    torch.cuda.synchronize()
    state = pkg._RESERVES[torch.cuda.current_device()]
    assert len(state["ready"]) == pkg._RESERVE_SIZE and not state["pending"]
    return state


def test_capture_without_workspace_takes_a_reserve_workspace(reserve):
    """Backwards without a caller workspace captured in one graph on one
    stream share one zero-filled workspace taken from the reserve: the graph
    holds the backward kernels only (no fill kernel, memset or event wait), for
    both weight-gradient dtypes; every replay matches eager launches and leaves
    the workspace zero; eager calls do not touch it."""
    pytest.importorskip("cuda.bindings.runtime")
    T, H = 129, 48
    x, z, w, dy = make(T, H, seed=70)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    torch.cuda.synchronize()
    main.zero_()
    ready = list(reserve["ready"])

    def step():
        outs = [backward(dy, x, z, w, EPS) for _ in range(2)]
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        return outs

    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    ws = taken[0]
    varies = _dw_varies(T * H)
    for k in range(3):
        backward(dy, x, z, w, EPS)  # an eager call between replays
        graph.replay()
        torch.cuda.synchronize()
        for dx, dz, dw in outs:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
            assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"replay {k}")
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not bool(ws.any()), k
        err = (main.double() - (k + 1) * dw64).abs()
        assert (err <= (k + 1) * (dw64.abs() * 2.0**-21 + mag * DW_MAG_TOL)).all(), k


def test_graphs_without_workspace_replay_together_on_two_streams(reserve):
    """Two graphs, each capturing a backward without a caller workspace, own
    two reserve workspaces and replay at the same time on two streams, the
    first replays right after the captures with no host synchronization."""
    cases = [make(4097, 48, seed=71), make(1031, 16, seed=72)]
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    budgets = [
        (fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS))
        for x, z, w, dy in cases
    ]
    torch.cuda.synchronize()
    graphs, outs, taken = [], [], []
    for x, z, w, dy in cases:
        ready = list(reserve["ready"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append(backward(dy, x, z, w, EPS))
        graphs.append(graph)
        taken += [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 2 and taken[0].data_ptr() != taken[1].data_ptr()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results: list[list] = [[], []]
    for _ in range(5):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                graphs[i].replay()
                results[i].append(tuple(t.clone() for t in outs[i]))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i in range(2):
        dx_d, dz_d, dw_d = refs[i]
        dw64, mag = budgets[i]
        varies = _dw_varies(cases[i][0].numel() // D)
        assert_dw_within(dw_d, dw64, mag, what=f"graph {i} dw")
        for dx, dz, dw in results[i]:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
            assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"graph {i} dw")
        assert not bool(taken[i].any()), i


def test_captured_calls_without_capture_ids_take_one_workspace_each(
    reserve, monkeypatch
):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays stay correct and leave them zero."""
    monkeypatch.setattr(pkg, "_cudart", lambda: None)
    x, z, w, dy = make(129, 48, seed=74)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    torch.cuda.synchronize()
    ready = list(reserve["ready"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [backward(dy, x, z, w, EPS) for _ in range(3)]
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 3
    varies = _dw_varies(129 * 48)
    for k in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for dx, dz, dw in outs:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
            assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"replay {k}")
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
        assert not any(bool(ws.any()) for ws in taken), k


def test_capture_with_no_ready_workspace_fills_in_the_graph(monkeypatch):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a memset
    or fill kernel before the backward kernel) and stays correct."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types in (["MEMSET", "KERNEL"], ["KERNEL", "KERNEL"]), types
    varies = _dw_varies(129 * 48)
    for k in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
        assert_dw_repeats(dw, dw_d, dw64, mag, varies=varies, what=f"replay {k}")
        assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")


def test_autograd_matches_functional_and_checkpoint(launches):
    x, z, w, dy = make(513, 48, seed=8)
    # the functional entry points' bits
    y_ref = forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    assert_dw_within(dw_d, dw64, mag, what="dw vs FP64")
    varies = _dw_varies(x.numel() // D)

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        assert_dw_repeats(wg, dw_d, dw64, mag, varies=varies, what=what)

    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = gated_rmsnorm(xa, za, wa, EPS)
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")

    # Saved state: only the inputs, no FP32 temporary.
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel(), t.data_ptr()))
        return t

    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        gated_rmsnorm(xb, zb, wb, EPS)
    assert sorted(saved) == sorted(
        [
            (torch.bfloat16, x.numel(), xb.data_ptr()),
            (torch.bfloat16, z.numel(), zb.data_ptr()),
            (torch.bfloat16, D, wb.data_ptr()),
        ]
    )

    # Non-reentrant activation checkpointing replays the forward, in the same
    # mode, in the backward.
    launches.clear()
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")
    fwd = _forward_module(x.numel() // D)
    assert launches == [fwd, fwd, _backward_module(x.numel() // D)], launches


def test_nondeterministic_mode_launches_this_package(launches):
    x, z, w, dy = make(129, 48, seed=14)
    forward(x, z, w, EPS)
    backward(dy, x, z, w, EPS)
    backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=torch.float32, device="cuda")
    )
    fwd = _forward_module(x.numel() // D)
    rows = 129 * 48
    bf16, f32 = _backward_module(rows), _backward_module(rows, torch.float32)
    assert launches == [fwd, bf16, f32], launches
    for name in launches:
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
    arch = loader.device_arch(torch.cuda.current_device())
    for name, role in ((bf16, "bwd_bf16"), (f32, "bwd_f32")):
        bands = {band for _, band in loader.BACKWARD_BANDS[arch][role]}
        assert loader.MODULES[name]["role"] in bands, (name, role)
    # A mode keyword is a TypeError before any launch.
    launches.clear()
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_forward(
            x, z, w, EPS, deterministic=False
        )
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, deterministic=False
        )
    assert launches == []


def test_empty_and_errors():
    x = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    assert forward(x, x, w).shape == x.shape
    dx, dz, dw = backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        forward(good.float(), good, w)
    with pytest.raises(ValueError):
        forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        backward(good, good, good, w, accumulate_grad_weight=True)
    with pytest.raises(ValueError):
        backward(
            good,
            good,
            good,
            w,
            grad_weight=torch.empty(D, dtype=torch.float16, device="cuda"),
        )


# Programs this package no longer ships: no record, ``build.json`` library or
# source file names them.  The second static backwards of sm_100a and sm_103a
# (9e8bf507c9563bb24193, 348592b55aa04975cc33) and of sm_107a (3 slots:
# f5a44b8ba8bccb590488 of a BF16, de4ac875a4e07bb547fd of an FP32 weight
# gradient), the static backward of a BF16 weight gradient on sm_100a and
# sm_103a (7e9f1c0cadbc18c390d2), the forwards without the split gate
# (c7919260b39b69e4e646, 83fe9e25ce0d01cc3ab1, c6151f64f22ae65a48eb,
# cf7a87977aaad1a9822f, 3cb6aa31e140d9b13052, 04bb67629537ef270bd4), and the
# static backward of an FP32 weight gradient on sm_100a and sm_103a whose
# last CTA reads its windows by fetching adds (f2915438ea54b487d93f; the
# build that reads them by exchanges ships, with the same bits).
REMOVED_PROGRAMS = tuple(
    f"cake_rmsnorm_silu_gate_train_{digest}"
    for digest in (
        "9e8bf507c9563bb24193",
        "348592b55aa04975cc33",
        "7e9f1c0cadbc18c390d2",
        "f5a44b8ba8bccb590488",
        "de4ac875a4e07bb547fd",
        "c7919260b39b69e4e646",
        "83fe9e25ce0d01cc3ab1",
        "c6151f64f22ae65a48eb",
        "cf7a87977aaad1a9822f",
        "3cb6aa31e140d9b13052",
        "04bb67629537ef270bd4",
        "f2915438ea54b487d93f",
    )
)


def test_registry_records_reference_the_build_manifest():
    """Every architecture routes the forward, the roles of its forward bands
    and the roles of its ``BACKWARD_BANDS`` bands, one module a role, and
    besides them only backward roles of a BF16 weight gradient that no band
    runs (the long role of each band role ``LONG_BACKWARD_ROWS_PER_SM`` names,
    which runs in its place on long packed rows; programs kept routed for
    measurement; the call role ``bwd_bf16`` on sm_107a only: no static
    backward of a BF16 weight gradient ships on sm_100a and sm_103a); every
    module record names its ``build.json`` library (schema ``cake-build/1``),
    whose entries carry the package's kernel and binding sources and the
    precise-arithmetic flags; no record, library or source file names a
    program the package no longer ships (``REMOVED_PROGRAMS``).  The static
    and the claimed backward records cut their rows into chunks by one grid
    rule (``CHUNK_GRID_RULE``), launch as the static backward of an FP32
    weight gradient of their architecture, ship on the architectures that
    route them and carry the workspace buffers of their slots
    (``BACKWARD_SLOTS``); the row-claims backward records claim 16-row blocks
    by their grid rule (``ROW_CLAIMS_GRID_RULES``; the fused backward's
    ``FUSED_GRID_RULE``), launch 256-thread CTAs, take the FP64 column sums
    and the two words (``ROW_CLAIMS_LAYOUT``) and no chunks, and ship on the
    architectures that route them (the fused backward on every architecture,
    ``FUSED_ROLES``); the repeatable backward records claim 16-row blocks by
    their grid rule (``REP_GRID_RULE``), launch 256-thread CTAs, take the
    fixed-point digits, the flag words and the two words (``REP_LAYOUT``) and
    no chunks, and ship on every architecture (``REP_ROLES``; the long role's
    record launches as its band role's, with the same arguments); the
    small-row backward records launch one cluster and take no workspace."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        routed = set(loader.ROUTES[arch])
        forward_bands = {
            band
            for table in (loader.FORWARD_BANDS, loader.STRIDED_FORWARD_BANDS)
            for _, band in table[arch]
        }
        assert forward_bands <= set(FORWARD_ROLES), (arch, forward_bands)
        assert set(loader.BACKWARD_BANDS[arch]) == {"bwd_bf16", "bwd_f32"}, arch
        backward_bands = {
            band for bands in loader.BACKWARD_BANDS[arch].values() for _, band in bands
        }
        banded = {"fwd"} | forward_bands | backward_bands
        assert banded <= routed, (arch, banded - routed)
        unbanded = routed - banded
        assert all(role.startswith("bwd_bf16") for role in unbanded), (arch, unbanded)
        assert not unbanded & set(loader.BAND_ROLES), (arch, unbanded)
        cuts = loader.LONG_BACKWARD_ROWS_PER_SM[arch]
        assert cuts == {"bwd_bf16_rep": LONG_ROWS_PER_SM[arch]}, (arch, cuts)
        assert set(cuts) <= backward_bands, (arch, cuts)
        longs = {band + loader.LONG for band in cuts}
        assert longs <= unbanded, (arch, longs - unbanded)
        assert "bwd_f32" in routed, arch
        assert ("bwd_bf16" in routed) == (arch == "sm_107a"), arch
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            assert record["role"] == role and arch in record["arches"], (arch, role)
            assert loader.library_name(name, arch) in libraries, (arch, role, name)
    routed = {name for by_role in loader.ROUTES.values() for name in by_role.values()}
    assert set(loader.MODULES) == routed
    sources = [path.name for path in (package / "csrc").iterdir()]
    for program in REMOVED_PROGRAMS:
        assert program not in loader.MODULES, program
        assert not [lib for lib in libraries if lib.startswith(program)], program
        assert not [s for s in sources if s.startswith(program)], program
    for name, record in loader.MODULES.items():
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False
        clustered = record["grid_rule"]["kind"] == "cluster"
        assert (record["launch"]["cluster"] != (1, 1, 1)) == clustered, name
        plan = [key for _kind, key in record["arg_plan"]]
        if record["role"] in ("fwd", "fwd_tiny", "fwd_short"):
            assert record["grid_rule"]["kind"] == "row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in (
            "fwd_xlong",
            "fwd_long",
            "fwd_mid",
            "fwd_pipe",
            "fwd_pipe_packed",
        ):
            assert record["grid_rule"]["kind"] == "persistent_row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in loader.BAND_ROLES:
            assert clustered
            assert record["launch"]["cluster"] == (record["grid_rule"]["cluster"], 1, 1)
            assert {"dw", "weight_grad"} <= set(plan), name
            assert not {"dw_win", "dw_sums", "dw_spec", "dw_done"} & set(plan), name
            assert record.get("workspace") is None
            assert loader.workspace_bytes(name) == 0
        elif record["grid_rule"]["kind"] == "row_claims":
            role = record["role"]
            fused = role in FUSED_ROLES[record["arches"][0]]
            rule = FUSED_GRID_RULE if fused else ROW_CLAIMS_GRID_RULES.get(role)
            assert record["grid_rule"] == rule, name
            assert record["launch"]["block"] == (256, 1, 1)
            assert {"dw_sums", "dw_done", "dw", "weight_grad"} <= set(plan), name
            assert not {"dw_win", "dw_spec", "blocks_per_chunk"} & set(plan), name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            if fused:
                assert routed_on == [a for a in loader.ARCHES if role in FUSED_ROLES[a]]
            assert record["workspace"] == {"dw_done": 2, "dw_sums": 4 * D}, name
            assert loader.workspace_layout(name) == ROW_CLAIMS_LAYOUT, name
            assert loader.workspace_bytes(name) == ROW_CLAIMS_WORKSPACE_BYTES, name
            for arch in record["arches"]:
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
        elif record["grid_rule"]["kind"] == "claimed_row_blocks":
            role = record["role"]
            assert record["grid_rule"] == REP_GRID_RULE, name
            assert record["launch"]["block"] == (256, 1, 1)
            assert {"acc", "flags", "done", "dw", "weight_grad"} <= set(plan), name
            assert not {
                "dw_win",
                "dw_sums",
                "dw_spec",
                "dw_done",
                "blocks_per_chunk",
            } & set(plan), name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            assert routed_on == [a for a in loader.ARCHES if role in REP_ROLES[a]]
            if role.endswith(loader.LONG):
                band = role[: -len(loader.LONG)]
                twin = loader.MODULES[loader.ROUTES[routed_on[0]][band]]
                for key in ("arches", "grid_rule", "launch", "arg_plan", "workspace"):
                    assert record[key] == twin[key], (name, key)
            assert record["workspace"] == {"acc": 18 * D, "done": 2, "flags": D}, name
            assert loader.workspace_layout(name) == REP_LAYOUT, name
            assert loader.workspace_bytes(name) == REP_WORKSPACE_BYTES, name
            for arch in record["arches"]:
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
        else:
            role = record["role"]
            assert record["grid_rule"] == CHUNK_GRID_RULE, name
            assert record["launch"]["block"] == (128, 1, 1)
            assert {
                "dw_win",
                "dw_spec",
                "dw_done",
                "dw",
                "weight_grad",
                "blocks_per_chunk",
            } <= set(plan), name
            assert "dw_sums" not in plan, name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            for arch in record["arches"]:
                slots = BACKWARD_SLOTS[arch][role]
                assert record["workspace"] == {
                    "dw_done": 2,
                    "dw_spec": D,
                    "dw_win": slots * 7 * D,
                }, (name, arch)
                assert loader.workspace_layout(name) == WORKSPACE_LAYOUTS[slots]
                assert loader.workspace_bytes(name) == MODULE_WORKSPACE_BYTES[slots]
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
                static = loader.MODULES[loader.ROUTES[arch]["bwd_f32"]]
                assert record["launch"] == static["launch"], (name, arch)
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


# (rows, SMs, blocks per chunk, chunks, CTAs) of the static and the claimed
# backwards: 8-row blocks, at most 8192 chunks of the fewest consecutive
# blocks each (the chunks of the deterministic backward at FlashInfer 2e989556),
# at most 8 CTAs per SM.
BACKWARD_GRIDS = [
    (1, 148, 1, 1, 1),
    (48, 148, 1, 6, 6),
    (6191, 148, 1, 774, 774),
    (64 * 148, 148, 1, 1184, 1184),  # every chunk has a CTA of one wave
    (64 * 148 + 1, 148, 1, 1185, 1184),  # a chunk more than one wave
    (49488, 148, 1, 6186, 1184),
    (49489, 148, 1, 6187, 1184),
    (8 * 8192, 148, 1, 8192, 1184),
    (8 * 8192 + 1, 148, 2, 4097, 1184),  # two blocks a chunk
    (245760, 148, 4, 7680, 1184),
    (245761, 148, 4, 7681, 1184),
    (32737 * 48, 152, 24, 8185, 1216),
    (1440000, 160, 22, 8182, 1280),
    (64 * 212, 212, 1, 1696, 1696),
    (64 * 212 + 1, 212, 1, 1697, 1696),
    (98352, 212, 2, 6147, 1696),
    (98353, 212, 2, 6148, 1696),
    (64 * 216, 216, 1, 1728, 1728),
    (64 * 216 + 1, 216, 1, 1729, 1728),
    (2**31 - 1, 216, 32768, 8192, 1728),
]


@pytest.mark.parametrize(
    ("rows", "sm_count", "per_chunk", "chunks", "ctas"), BACKWARD_GRIDS
)
def test_backward_cta_rule(rows, sm_count, per_chunk, chunks, ctas):
    """The static and the claimed backwards (every routed module whose grid
    rule is ``chunks``): ``blocks_per_chunk = max(1, ceil(blocks / 8192))``
    blocks a chunk, ``ceil(blocks / blocks_per_chunk)`` chunks, ``min(chunks,
    8 * SMs)`` CTAs (each running chunks ``b``, ``b + grid``, ... in the static
    backwards; claiming runs of chunks in the claimed ones), so a smaller grid
    still covers every row; each backward role runs the module of the
    ``BACKWARD_BANDS`` band that holds the rows (:func:`_band_module`)."""
    assert _chunks(rows) == chunks, rows
    for arch in loader.ARCHES:
        chunked = {
            name
            for name in loader.ROUTES[arch].values()
            if loader.MODULES[name]["grid_rule"]["kind"] == "chunks"
        }
        assert chunked, arch
        for name in sorted(chunked):
            record = loader.MODULES[name]
            assert loader.backward_chunks(name, rows) == (per_chunk, chunks)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas == min(chunks, 8 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas, 1, 1)
        for role in ("bwd_bf16", "bwd_f32"):
            want = _band_module(arch, role, rows, sm_count)
            got = loader.module_for_rows(arch, role, rows, sm_count)
            assert got == want, (arch, role, rows)


# (rows, SMs, 16-row blocks, CTAs at 4 CTAs per SM, CTAs at 5 CTAs per SM) of
# the row-claims backwards.
ROW_CLAIMS_GRIDS = [
    (1, 148, 1, 1, 1),
    (16, 148, 1, 1, 1),
    (17, 148, 2, 2, 2),
    (11 * 48, 148, 33, 33, 33),
    (16 * 592, 148, 592, 592, 592),  # a block a CTA of one wave of 4 per SM
    (16 * 592 + 1, 148, 593, 592, 593),
    (16 * 740, 148, 740, 592, 740),  # a block a CTA of one wave of 5 per SM
    (16 * 740 + 1, 148, 741, 592, 740),
    (1031 * 48, 148, 3093, 592, 740),
    (4097 * 48, 152, 12291, 608, 760),
    (300 * 48, 212, 900, 848, 900),
    (2**31 - 1, 216, 2**27, 864, 1080),
]


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "ctas5"), ROW_CLAIMS_GRIDS
)
def test_row_claims_backward_cta_rule(rows, sm_count, blocks, ctas4, ctas5):
    """The row-claims backwards (every routed module whose grid rule is
    ``row_claims`` but the fused backward's, :func:`test_fused_backward_cta_rule`):
    ``min(ceil(rows / 16), ctas_per_sm * SMs)`` CTAs, at most
    one CTA a 16-row block (past one wave the CTAs claim the blocks from the
    claim counter at run time), a grid of that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            rule = record["grid_rule"]
            if rule["kind"] != "row_claims" or role in FUSED_ROLES[arch]:
                continue
            found.add(role)
            assert rule == ROW_CLAIMS_GRID_RULES[role], (arch, role)
            want = {4: ctas4, 5: ctas5}[rule["ctas_per_sm"]]
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == want == min(blocks, rule["ctas_per_sm"] * sm_count), (
                name,
                rows,
            )
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (want, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    assert found == set(ROW_CLAIMS_GRID_RULES), found


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "_ctas5"), ROW_CLAIMS_GRIDS
)
def test_fused_backward_cta_rule(rows, sm_count, blocks, ctas4, _ctas5):
    """The fused backward (``FUSED_ROLES``: ``bwd_bf16_fused`` on every
    architecture, grid kind ``row_claims``): ``min(ceil(rows / 16), 4 * SMs)``
    CTAs (``ROW_CLAIMS_GRIDS`` at 4 CTAs per SM; past one wave the CTAs
    claim the blocks at run time), a grid of that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            if role not in FUSED_ROLES[arch]:
                continue
            found.add((arch, role))
            assert record["grid_rule"] == FUSED_GRID_RULE, (arch, role)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas4 == min(blocks, 4 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas4, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    assert found == {(a, r) for a, roles in FUSED_ROLES.items() for r in roles}


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "_ctas5"), ROW_CLAIMS_GRIDS
)
def test_repeatable_backward_cta_rule(rows, sm_count, blocks, ctas4, _ctas5):
    """The repeatable backward (``REP_ROLES``: ``bwd_bf16_rep`` and its long
    role on every architecture, grid kind ``claimed_row_blocks``):
    ``min(ceil(rows / 16), 4 * SMs)`` CTAs (``ROW_CLAIMS_GRIDS`` at 4 CTAs
    per SM; past one wave the CTAs claim the blocks at run time), a grid of
    that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            if record["grid_rule"]["kind"] != "claimed_row_blocks":
                continue
            found.add((arch, role))
            assert record["grid_rule"] == REP_GRID_RULE, (arch, role)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas4 == min(blocks, 4 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas4, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    assert found == {(a, r) for a, roles in REP_ROLES.items() for r in roles}


# The forward bands of each architecture for packed rows: (last rows per SM,
# role) in increasing rows per SM (``None``: no end), a call of ``rows`` rows on
# ``sm_count`` SMs running the first band's role with ``rows <= last *
# sm_count`` (cuts that follow the checkpointed training step at each rows per
# SM; the tiny forward ends where beside a kernel holding 16 SMs it still ends
# no later than the deterministic package's forward).  The forward's band
# roles, the call role ``fwd`` among them (the plain forward, the last band of
# other layouts); the backward's are ``BACKWARD_BANDS``.
FORWARD_BANDS = {
    "sm_100a": (
        (38, "fwd_tiny"),
        (85, "fwd_pipe"),
        (160, "fwd_mid"),
        (400, "fwd_long"),
        (520, "fwd_xlong"),
        (6500, "fwd_pipe"),
        (None, "fwd_mid"),
    ),
    "sm_103a": (
        (36, "fwd_tiny"),
        (240, "fwd_mid"),
        (600, "fwd_xlong"),
        (6000, "fwd_pipe"),
        (8192, "fwd_mid"),
        (None, "fwd_pipe"),
    ),
    "sm_107a": (
        (65, "fwd_tiny"),
        (175, "fwd_mid"),
        (280, "fwd_xlong"),
        (None, "fwd_pipe_packed"),
    ),
}
# The forward bands of inputs whose rows are not packed (the tiny forward
# reads packed rows only), and of packed rows from PACKED_ROWS on, as
# FORWARD_BANDS.
STRIDED_FORWARD_BANDS = {
    "sm_100a": (
        (12, "fwd_short"),
        (140, "fwd_mid"),
        (400, "fwd_long"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_103a": (
        (12, "fwd_short"),
        (250, "fwd_mid"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_107a": (
        (59, "fwd_short"),
        (480, "fwd_mid"),
        (None, "fwd"),
    ),
}
FORWARD_ROLES = (
    "fwd",
    "fwd_tiny",
    "fwd_short",
    "fwd_mid",
    "fwd_long",
    "fwd_xlong",
    "fwd_pipe",
    "fwd_pipe_packed",
)
# Packed rows from this many on run by STRIDED_FORWARD_BANDS (a packed row's
# element offset no longer fits 32 bits).
PACKED_ROWS = 16777216
# The one grid rule of the static and the claimed backwards (no cut of their
# own: ``BACKWARD_BANDS`` gives each its rows).
CHUNK_GRID_RULE = {
    "chunk_limit": 8192,
    "ctas_per_sm": 8,
    "kind": "chunks",
    "rows_per_block": 8,
}


def _short_end(arch, sm_count):
    """The most rows the short forward runs on ``arch`` by
    ``STRIDED_FORWARD_BANDS``."""
    (last, role) = STRIDED_FORWARD_BANDS[arch][0]
    assert role == "fwd_short"
    return last * sm_count


def _tiny_end(arch, sm_count):
    """The most rows the tiny forward runs on ``arch`` by ``FORWARD_BANDS``."""
    (last, role) = FORWARD_BANDS[arch][0]
    assert role == "fwd_tiny"
    return last * sm_count


def _band_module(arch, role, rows, sm_count, packed=True):
    """The module of the backward ``role`` (``bwd_bf16`` or ``bwd_f32``) for
    ``rows`` rows on ``arch`` with ``sm_count`` SMs: the routed module of the
    band of :func:`_backward_bands` whose first and last rows hold ``rows``
    (the last band has no end), or where the rows are ``packed`` the routed
    module of the long role of :func:`_long_bands` that holds them."""
    longs = _long_bands(arch, role, sm_count) if packed else []
    for band, first, last in longs + _backward_bands(arch, role, sm_count):
        if first <= rows and (last is None or rows <= last):
            return loader.ROUTES[arch][band]
    raise AssertionError(f"no {role} band of {arch} holds {rows} rows")


def _tier_module(arch, role, rows, sm_count, packed=True):
    """The module of ``role`` for ``rows`` rows: the forward's by
    ``FORWARD_BANDS`` (by ``STRIDED_FORWARD_BANDS`` where the rows are not
    ``packed`` or are ``PACKED_ROWS`` or more), the module of the first forward
    band whose rows per SM hold ``rows``; a backward role's by
    :func:`_band_module` (its long roles on ``packed`` rows)."""
    if role == "fwd":
        packed = packed and rows < PACKED_ROWS
        table = FORWARD_BANDS if packed else STRIDED_FORWARD_BANDS
        for last, band in table[arch]:
            if last is None or rows <= last * sm_count:
                return loader.ROUTES[arch][band]
        return loader.route_module(arch, role)
    return _band_module(arch, role, rows, sm_count, packed)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_forward_and_backward_modules_run_in_their_bands(sm_count):
    """``module_for_rows``: the forward's module of the first
    ``FORWARD_BANDS`` band (``STRIDED_FORWARD_BANDS`` band where the rows are
    not packed) with ``rows <= last * sm_count`` (rows per SM; the tiny, short,
    pipelined, mid, long, extra-long and plain forwards, each where the
    architecture's bands name it), a backward role's module of the first
    ``BACKWARD_BANDS`` band whose ``last`` holds (:func:`_band_module`).  The
    backward bands of each architecture and role: the small-row bands (a
    ``BAND_ROLES`` role each, whose module is one cluster), each ending at a
    count of rows ``(ROWS, n)`` with ``n`` at most its module's ``max_rows``,
    and the bands of rows per SM, each running a static, a row-claims (a BF16
    weight gradient on sm_100a and sm_103a only), a fused (``FUSED_ROLES``), a
    repeatable (``REP_ROLES``) or a claimed backward by its grid rule, the
    ends of each kind in increasing order; last, with no end, the repeatable
    backward of a BF16 weight gradient (after the fused band) and the claimed
    backward (``<role>_mid``) of an FP32 one; packed rows of the last
    repeatable band from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on
    (``LONG_ROWS_PER_SM``) run its long role (:func:`_long_bands`), other
    rows the band's role.  No grid rule carries a band's
    rows per SM or a chunks-per-SM cut (the pipelined forwards'
    ``persistent_from_rows_per_sm`` shapes their grid only): the bands and
    the long roles' cuts are host plan only.  Packed rows of
    ``PACKED_ROWS`` or more run by ``STRIDED_FORWARD_BANDS``."""
    assert loader.PACKED_ROWS == PACKED_ROWS == 2**31 // D
    assert loader.FORWARD_BANDS == FORWARD_BANDS
    assert loader.STRIDED_FORWARD_BANDS == STRIDED_FORWARD_BANDS
    clustered = {
        loader.MODULES[name]["role"]
        for routes in loader.ROUTES.values()
        for name in routes.values()
        if loader.MODULES[name]["grid_rule"]["kind"] == "cluster"
    }
    assert set(loader.BAND_ROLES) == clustered, (loader.BAND_ROLES, clustered)
    assert loader.ROWS == "rows"
    for gone in ("ONE_WAVE", "MAX_ROWS", "_claims"):
        assert not hasattr(loader, gone), gone
    assert set(loader.BACKWARD_BANDS) == set(loader.ARCHES)
    for arch in loader.ARCHES:
        forward = {"fwd"}
        for table, first in (
            (FORWARD_BANDS, "fwd_tiny"),
            (STRIDED_FORWARD_BANDS, "fwd_short"),
        ):
            bands = table[arch]
            lasts = [last for last, _ in bands[:-1]]
            assert lasts == sorted(set(lasts)) and bands[-1][0] is None, arch
            assert bands[0][1] == first, arch
            forward |= {band for _, band in bands}
            for _, band in bands:
                record = loader.MODULES[loader.ROUTES[arch][band]]
                assert record["role"] == band
                row_blocks = band in ("fwd", "fwd_tiny", "fwd_short")
                kind = "row_blocks" if row_blocks else None
                assert record["grid_rule"]["kind"] == (kind or "persistent_row_blocks")
        routed = {role for role in loader.ROUTES[arch] if role.startswith("fwd")}
        assert routed == forward, arch
        assert set(loader.BACKWARD_BANDS[arch]) == {"bwd_bf16", "bwd_f32"}, arch
        for role, bands in loader.BACKWARD_BANDS[arch].items():
            # the small-row bands (counts of rows) and the bands of rows per SM,
            # each kind's ends increasing; the last with no end the repeatable
            # backward of a BF16 weight gradient, after the fused band, and the
            # claimed backward of an FP32 one
            final = {"bwd_bf16": "bwd_bf16_rep", "bwd_f32": "bwd_f32_mid"}[role]
            assert bands[-1] == (None, final), (arch, role, bands[-1])
            if role == "bwd_bf16":
                assert bands[-2][1] == "bwd_bf16_fused", (arch, bands)
                assert type(bands[-2][0]) in (int, float), (arch, bands)
            counts, per_sm = [], []
            for last, band in bands:
                assert band == role or band.startswith(role + "_"), (arch, band)
                record = loader.MODULES[loader.ROUTES[arch][band]]
                assert record["role"] == band
                rule = record["grid_rule"]
                if band in loader.BAND_ROLES:
                    assert rule["kind"] == "cluster", (arch, band)
                    assert isinstance(last, tuple) and len(last) == 2, (arch, band)
                    assert last[0] == loader.ROWS, (arch, band, last)
                    assert last[1] <= int(rule["max_rows"]), (arch, band, last)
                    counts.append(last[1])
                    continue
                if last is not None:
                    assert type(last) in (int, float), (arch, band, last)
                    per_sm.append(last)
                if band in ROW_CLAIMS_GRID_RULES:
                    assert arch != "sm_107a", (arch, band)
                    assert rule == ROW_CLAIMS_GRID_RULES[band], (arch, band)
                elif band in FUSED_ROLES[arch]:
                    assert rule == FUSED_GRID_RULE, (arch, band)
                elif band in REP_ROLES[arch]:
                    assert rule == REP_GRID_RULE, (arch, band)
                else:
                    assert rule == CHUNK_GRID_RULE, (arch, band)
            assert counts == sorted(set(counts)), (arch, role, counts)
            assert per_sm == sorted(set(per_sm)), (arch, role, per_sm)
            ends = [end for _, _, end in _backward_bands(arch, role, sm_count)]
            assert ends[:-1] == sorted(set(ends[:-1])), (arch, role, ends)
            # the long role of the last repeatable band of a BF16 weight
            # gradient, from its cut on; an FP32 weight gradient has none
            longs = _long_bands(arch, role, sm_count)
            if role == "bwd_f32":
                assert longs == [], (arch, longs)
                continue
            cut = LONG_ROWS_PER_SM[arch] * sm_count
            assert longs == [("bwd_bf16_rep_long", cut, None)], (arch, longs)
            assert _backward_bands(arch, role, sm_count)[-1][1] < cut, arch
        for name in loader.ROUTES[arch].values():
            rule = loader.MODULES[name]["grid_rule"]
            assert not {"min_rows_per_sm", "max_rows_per_sm"} & set(rule), name
            assert "more_chunks_than_per_sm" not in rule, name
        for role, name in loader.ROUTES[arch].items():
            assert loader.MODULES[name]["role"] == role
        edges = {1, 48, 255, 256, 257, 8 * 8192, 8 * 8192 + 1, 2**31 - 1}
        for role in loader.BACKWARD_BANDS[arch]:
            for _, _, end in _backward_bands(arch, role, sm_count)[:-1]:
                edges |= {end - 1, end, end + 1}
        for table in (FORWARD_BANDS, STRIDED_FORWARD_BANDS):
            for bands in table.values():
                for last, _ in bands[:-1]:
                    edges |= {last * sm_count + d for d in (-1, 0, 1)}
        for start in (10, 12, 59, 64, LONG_ROWS_PER_SM[arch]):
            edges |= {start * sm_count - 1, start * sm_count, start * sm_count + 1}
        edges |= {PACKED_ROWS - 1, PACKED_ROWS, PACKED_ROWS + 1}
        for role in loader.ROLES:
            for rows in sorted(edges):
                want = _tier_module(arch, role, rows, sm_count)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == want, (arch, role, rows)
                want = _tier_module(arch, role, rows, sm_count, packed=False)
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == want, (arch, role, rows, "strided")
    # the forward on sm_100a: packed rows the tiny forward up to 38 rows per
    # SM, the pipelined forward up to 85, the mid forward up to 160, the long
    # forward up to 400, the extra-long forward up to 520, the pipelined
    # forward up to 6500, the mid forward past it; other layouts the short
    # forward up to 12 rows per SM, the mid forward up to 140, the long
    # forward up to 400, the extra-long forward up to 700, the plain forward
    # past it
    routes = loader.ROUTES["sm_100a"]
    for rows, packed, strided in (
        (1, "fwd_tiny", "fwd_short"),
        (12 * sm_count, "fwd_tiny", "fwd_short"),
        (12 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (38 * sm_count, "fwd_tiny", "fwd_mid"),
        (38 * sm_count + 1, "fwd_pipe", "fwd_mid"),
        (85 * sm_count, "fwd_pipe", "fwd_mid"),
        (85 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (140 * sm_count + 1, "fwd_mid", "fwd_long"),
        (160 * sm_count, "fwd_mid", "fwd_long"),
        (160 * sm_count + 1, "fwd_long", "fwd_long"),
        (400 * sm_count, "fwd_long", "fwd_long"),
        (400 * sm_count + 1, "fwd_xlong", "fwd_xlong"),
        (520 * sm_count, "fwd_xlong", "fwd_xlong"),
        (520 * sm_count + 1, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count + 1, "fwd_pipe", "fwd"),
        (6500 * sm_count, "fwd_pipe", "fwd"),
        (6500 * sm_count + 1, "fwd_mid", "fwd"),
        (PACKED_ROWS - 1, "fwd_mid", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_100a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_100a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    # the forward on sm_103a: packed rows the tiny forward up to 36 rows per
    # SM, the mid forward up to 240, the extra-long forward up to 600, the
    # pipelined forward up to 6000, the mid forward up to 8192, the pipelined
    # forward past it; other layouts the short forward up to 12, the mid
    # forward up to 250, the extra-long forward up to 700, the plain forward
    # past it (sm_103a routes no long forward)
    routes = loader.ROUTES["sm_103a"]
    for rows, packed, strided in (
        (1, "fwd_tiny", "fwd_short"),
        (12 * sm_count, "fwd_tiny", "fwd_short"),
        (12 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (36 * sm_count, "fwd_tiny", "fwd_mid"),
        (36 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (240 * sm_count, "fwd_mid", "fwd_mid"),
        (240 * sm_count + 1, "fwd_xlong", "fwd_mid"),
        (250 * sm_count, "fwd_xlong", "fwd_mid"),
        (250 * sm_count + 1, "fwd_xlong", "fwd_xlong"),
        (600 * sm_count, "fwd_xlong", "fwd_xlong"),
        (600 * sm_count + 1, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count + 1, "fwd_pipe", "fwd"),
        (6000 * sm_count, "fwd_pipe", "fwd"),
        (6000 * sm_count + 1, "fwd_mid", "fwd"),
        (8192 * sm_count, "fwd_mid", "fwd"),
        (8192 * sm_count + 1, "fwd_pipe", "fwd"),
        (PACKED_ROWS - 1, "fwd_pipe", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_103a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_103a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    assert "fwd_long" not in routes
    # the forward on sm_107a: packed rows the tiny forward up to 65 rows per
    # SM, the mid forward up to 175, the extra-long forward up to 280, the
    # pipelined forward of packed rows past it; other layouts the short
    # forward up to 59, the mid forward up to 480, the plain forward past it
    # (sm_107a routes neither a long forward nor the pipelined forward of
    # other layouts)
    routes = loader.ROUTES["sm_107a"]
    for rows, packed, strided in (
        (1, "fwd_tiny", "fwd_short"),
        (10 * sm_count + 1, "fwd_tiny", "fwd_short"),
        (59 * sm_count, "fwd_tiny", "fwd_short"),
        (59 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (65 * sm_count, "fwd_tiny", "fwd_mid"),
        (65 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (175 * sm_count, "fwd_mid", "fwd_mid"),
        (175 * sm_count + 1, "fwd_xlong", "fwd_mid"),
        (280 * sm_count, "fwd_xlong", "fwd_mid"),
        (280 * sm_count + 1, "fwd_pipe_packed", "fwd_mid"),
        (480 * sm_count, "fwd_pipe_packed", "fwd_mid"),
        (480 * sm_count + 1, "fwd_pipe_packed", "fwd"),
        (65536 * 48, "fwd_pipe_packed", "fwd"),
        (PACKED_ROWS - 1, "fwd_pipe_packed", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_107a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_107a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    assert "fwd_long" not in routes and "fwd_pipe" not in routes


def _check_backward_bands(arch: str, sm_count: int) -> None:
    """Every ``BACKWARD_BANDS`` band of ``arch`` runs rows on ``sm_count`` SMs
    (:func:`_backward_bands`), and ``module_for_rows`` runs its routed module
    at its first and at its last rows (the last band, which has no end, at
    ``INT32_MAX`` rows) that are not packed, and so on packed rows below its
    long role's first rows (:func:`_band_module`); neighbouring bands run
    different modules, and the row after a band's end runs the next band's
    module, so the bands cover every row count without a gap; a band of a
    count of rows, ``(ROWS, n)``, ends at most at its cluster module's
    ``max_rows``.  A long role (:func:`_long_bands`) runs packed rows at its
    first and at its last rows, not the row before them, and its band's
    role runs the same rows when they are not packed."""
    for role in ("bwd_bf16", "bwd_f32"):
        table = loader.BACKWARD_BANDS[arch][role]
        bands = _backward_bands(arch, role, sm_count)
        assert [band for band, _, _ in bands] == [band for _, band in table], (
            arch,
            role,
            sm_count,
            bands,
        )
        assert bands[0][1] == 1 and bands[-1][2] is None, (arch, role, bands)
        for i, (band, first, last) in enumerate(bands):
            name = loader.ROUTES[arch][band]
            assert loader.MODULES[name]["role"] == band, (arch, band)
            for rows in (first, loader.INT32_MAX if last is None else last):
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == name, (arch, role, band, rows)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                want = _band_module(arch, role, rows, sm_count)
                assert got == want, (arch, role, band, rows, "packed")
            if last is None:
                continue
            after = loader.ROUTES[arch][bands[i + 1][0]]
            assert after != name, (arch, role, band, bands[i + 1][0])
            got = loader.module_for_rows(arch, role, last + 1, sm_count, packed=False)
            assert got == after, (arch, role, band, last + 1)
        for band, first, last in _long_bands(arch, role, sm_count):
            name = loader.ROUTES[arch][band]
            assert loader.MODULES[name]["role"] == band, (arch, band)
            short = loader.ROUTES[arch][band[: -len(loader.LONG)]]
            for rows in (first, loader.INT32_MAX if last is None else last):
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == name, (arch, role, band, rows)
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == short, (arch, role, band, rows, "strided")
            got = loader.module_for_rows(arch, role, first - 1, sm_count)
            assert got != name, (arch, role, band, first - 1)
        for last, band in table:
            if isinstance(last, tuple):
                rule = loader.MODULES[loader.ROUTES[arch][band]]["grid_rule"]
                assert rule["kind"] == "cluster", (arch, band)
                assert last[1] <= int(rule["max_rows"]), (arch, band, last)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_backward_bands_run_their_modules_from_their_first_to_their_last_rows(
    sm_count,
):
    """``BACKWARD_BANDS`` of every architecture on ``sm_count`` SMs
    (:func:`_check_backward_bands`); the call role ``bwd_bf16`` has no module
    on sm_100a and sm_103a (``route_module`` raises ``ValueError``): there the
    bands of a BF16 weight gradient cover every row count without it."""
    for arch in loader.ARCHES:
        _check_backward_bands(arch, sm_count)
    for arch in ("sm_100a", "sm_103a"):
        assert "bwd_bf16" not in loader.ROUTES[arch], arch
        with pytest.raises(ValueError, match="no exported"):
            loader.route_module(arch, "bwd_bf16")


def test_backward_bands_run_their_modules_on_this_device(launches):
    """``BACKWARD_BANDS`` on this device's architecture and SM count
    (:func:`_check_backward_bands`, :func:`_device_bands`), and a backward of
    the first and of the last rows of every band (``T`` tokens of one head,
    packed rows; the last band at its first rows) and of the first rows of
    every long role (:func:`_device_long_bands`) launches the band's module,
    ``dx``, ``dz`` and the weight gradient of the band's dtype within the
    FP64 budgets."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    _check_backward_bands(arch, sm_count)
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        assert _device_bands(role) == _backward_bands(arch, role, sm_count)
        assert _device_long_bands(role) == _long_bands(arch, role, sm_count)
        for band, first, last in _device_bands(role) + _device_long_bands(role):
            name = loader.ROUTES[arch][band]
            for rows in sorted({first} | ({last} if last is not None else set())):
                x, z, w, dy = make(rows, 1, seed=91)
                launches.clear()
                gw = torch.empty(D, dtype=dtype, device="cuda")
                dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=gw)
                assert launches == [name], (band, rows, launches)
                what = f"{band} at {rows} rows"
                _, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
                assert_bf16_close(dx, dx64, what=f"dx {what}")
                assert_bf16_close(dz, dz64, what=f"dz {what}")
                mag = magnitudes(dy, x, z, EPS)
                assert_dw_within(dw, dw64, mag, what=f"dw {what}")


# SM counts on which the host routes the repeatable backward's long role:
# small MIG-like instances, the products' counts and larger ones.
LONG_SM_COUNTS = (1, 2, 7, 14, 16, 24, 33, 66, 148, 152, 160, 212, 216)


@pytest.mark.parametrize("sm_count", LONG_SM_COUNTS)
def test_backward_runs_the_long_role_on_packed_rows_only(sm_count):
    """The host (no launch): a backward of a BF16 weight gradient runs the
    repeatable backward's long role (``bwd_bf16_rep_long``) from
    ``LONG_ROWS_PER_SM * SMs`` rows on when ``dy``, ``x`` and ``z`` all have
    head stride 128 and token stride ``H * 128`` as ``_kernel_view`` passes
    them (``_backward_packed``; also with one head, whose head stride is
    passed as 128, and at the kernels' row limit), and the repeatable
    backward a row fewer, for a single token (token stride 0), a token
    stride of 0 in any one tensor, a gate or a ``dy`` sliced from a
    projection, head-major rows and a head stride other than 128; an FP32
    weight gradient runs the module of its band whatever the layout."""
    assert loader.LONG == "_long"
    cuts = {arch: {"bwd_bf16_rep": LONG_ROWS_PER_SM[arch]} for arch in loader.ARCHES}
    assert cuts == loader.LONG_BACKWARD_ROWS_PER_SM
    H = 48
    row = H * D
    packed = (row, D, row, D, row, D)
    one_head = (D, D, D, D, D, D)
    for arch in loader.ARCHES:
        rep = loader.ROUTES[arch]["bwd_bf16_rep"]
        long = loader.ROUTES[arch]["bwd_bf16_rep_long"]
        start = LONG_ROWS_PER_SM[arch] * sm_count
        for rows, heads, strides, is_packed, want in (
            (start - 1, H, packed, True, rep),
            (start, H, packed, True, long),
            (start + 1, H, packed, True, long),
            (loader.INT32_MAX, H, packed, True, long),
            (start - 1, 1, one_head, True, rep),
            (start, 1, one_head, True, long),
            (start, start, (0, D, 0, D, 0, D), False, rep),
            (start, H, (0, D, row, D, row, D), False, rep),
            (start, H, (row, D, 0, D, row, D), False, rep),
            (start, H, (row, D, row, D, 0, D), False, rep),
            (start, H, (row, D, row, D, 2 * row + 16, D), False, rep),
            (start, H, (2 * row + 16, D, row, D, row, D), False, rep),
            (start, H, (row, D, D, 65 * D, D, 65 * D), False, rep),
            (start, H, (row, D, row, D, row, 2 * D), False, rep),
        ):
            what = (arch, rows, heads, strides)
            assert pkg._backward_packed(heads, *strides) == is_packed, what
            got = loader.module_for_rows(
                arch, "bwd_bf16", rows, sm_count, packed=is_packed
            )
            assert got == want, what
            f32 = loader.module_for_rows(
                arch, "bwd_f32", rows, sm_count, packed=is_packed
            )
            assert f32 == _band_module(arch, "bwd_f32", rows, sm_count), what
            assert not loader.MODULES[f32]["role"].endswith(loader.LONG), what


def _forcing(band: str):
    """A ``module_for_rows`` that names the routed module of the backward band
    role ``band`` for every call of a BF16 weight gradient's role
    (``bwd_bf16``, ``compute_grad_weight=False`` among them) and the host's
    module for every other call."""
    plan = loader.module_for_rows

    def module(arch, role, rows, sm_count, *, packed=True):
        if role == "bwd_bf16":
            return loader.ROUTES[arch][band]
        return plan(arch, role, rows, sm_count, packed=packed)

    return module


def _rep_calls(dy, x, z, w, ws) -> list[torch.Tensor]:
    """The backward calls of the long-role tests, all of the role
    ``bwd_bf16``: a BF16 weight gradient, one accumulated onto 0.5 (on
    ``ws``) and ``compute_grad_weight=False``.  Their ``dx``, ``dz`` and
    weight gradients."""
    acc = torch.full((D,), 0.5, dtype=torch.bfloat16, device=x.device)
    got = []
    for kwargs in (
        {},
        {"grad_weight": acc, "accumulate_grad_weight": True, "workspace": ws},
        {"compute_grad_weight": False},
    ):
        dx, dz, dw = backward(dy, x, z, w, EPS, **kwargs)
        got += [dx, dz] if dw is None else [dx, dz, dw]
    return got


# Packed shapes on which the repeatable backward's long role is compared with
# the repeatable backward: flat rows a row below, at and a row above the long
# role's first rows, and the issue's 32737 x 48 rows (claims far past the
# look-ahead), each on this device's grid and on the grids of
# ``LONG_BIT_SM_COUNTS`` SMs (``device_sm_count`` patched: the first rows of
# the long role follow the SMs).
LONG_BIT_SHAPES = ("long-1", "long", "long+1", "32737x48")
LONG_BIT_SM_COUNTS = (1, 16, 66)


@pytest.mark.parametrize("shape", LONG_BIT_SHAPES)
def test_long_role_gives_the_bits_of_the_repeatable_backward(shape, launches):
    """On this device's grid and on the grids of ``LONG_BIT_SM_COUNTS`` SMs,
    for ``LONG_BIT_SHAPES``: the host runs the long role
    (``bwd_bf16_rep_long``) on packed rows from ``LONG_ROWS_PER_SM * SMs``
    rows on and the repeatable backward a row fewer; forced through
    ``module_for_rows`` (:func:`_forcing`), the long role gives the bits of
    the repeatable backward on the same packed inputs and grid, and so do the
    host's calls: ``dx``, ``dz``, the BF16 weight gradient, accumulated or
    not, and ``compute_grad_weight=False`` (:func:`_rep_calls`); the
    workspace is zero after the calls."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    rep, long = routes["bwd_bf16_rep"], routes["bwd_bf16_rep_long"]
    for sm_count in (loader.device_sm_count(device_index), *LONG_BIT_SM_COUNTS):
        start = LONG_ROWS_PER_SM[arch] * sm_count
        if shape.startswith("long"):
            rows = start + {"long-1": -1, "long": 0, "long+1": 1}[shape]
            x, z, w, dy = make(rows, 1, layout="flat", seed=46)
        else:
            T, H = (int(v) for v in shape.split("x"))
            rows = T * H
            x, z, w, dy = make(T, H, seed=46)
        assert (rows >= start) == (shape != "long-1"), (shape, sm_count, rows)
        ws = workspace_for(rows, x.device)
        runs = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "device_sm_count", lambda index, count=sm_count: count)
            launches.clear()
            host = _rep_calls(dy, x, z, w, ws)
            want = long if rows >= start else rep
            assert launches == [want] * 3, (shape, sm_count, launches)
            for band in ("bwd_bf16_rep", "bwd_bf16_rep_long"):
                patch.setattr(pkg, "module_for_rows", _forcing(band))
                launches.clear()
                runs.append(_rep_calls(dy, x, z, w, ws))
                assert launches == [routes[band]] * 3, (band, launches)
        torch.cuda.synchronize()
        assert not bool(ws.any()), (shape, sm_count)
        what = f"{shape} ({rows} rows) on the grids of {sm_count} SMs"
        for i, (a, b, c) in enumerate(zip(runs[1], runs[0], host, strict=True)):
            assert torch.equal(a, b), f"{what}: output {i} of the long role"
            assert torch.equal(c, b), f"{what}: output {i} of the host's module"


def test_capture_runs_the_long_role_and_the_repeatable_backward(reserve, launches):
    """Backwards of this device's first rows of the long role
    (``LONG_ROWS_PER_SM * SMs`` flat packed rows) captured in one graph on one
    stream: a BF16 weight gradient on a caller workspace runs the long role
    (``bwd_bf16_rep_long``), an FP32 one accumulated without a workspace the
    module of its band, and a BF16 one whose ``grad_out`` is strided (read in
    place, so not packed) the repeatable backward on the reserve workspace
    the FP32 call took.  The graph holds the three backward kernels only (no
    fill kernel, memset or event wait).  Every replay on new inputs gives the
    bits of eager calls with the BF16 calls forced to the repeatable backward
    (:func:`_forcing`; the accumulated weight gradient those of the same
    accumulation run eagerly) and leaves the caller workspace and the reserve
    workspace zero."""
    pytest.importorskip("cuda.bindings.runtime")
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    rows = LONG_ROWS_PER_SM[arch] * loader.device_sm_count(device_index)
    xs, zs, w, dys = make(rows, 1, layout="flat", seed=76)
    # grad_out read in place with token stride 2 * 128 + 16, so not packed
    wide = torch.zeros(rows, 2 * D + 16, dtype=torch.bfloat16, device=xs.device)
    dy_strided = wide[:, :D]
    dy_strided.copy_(dys)
    main = torch.zeros(D, dtype=torch.float32, device=xs.device)
    ws = workspace_for(rows, xs.device)

    def step():
        return (
            backward(dys, xs, zs, w, EPS, workspace=ws),
            backward(
                dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True
            ),
            backward(dy_strided, xs, zs, w, EPS),
        )

    f32 = _module("bwd_f32", rows)
    want = [routes["bwd_bf16_rep_long"], f32, routes["bwd_bf16_rep"]]
    launches.clear()
    step()  # loads the modules before the capture
    torch.cuda.synchronize()
    assert launches == want, launches
    assert not bool(ws.any())
    main.zero_()
    ready = list(reserve["ready"])
    launches.clear()
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    assert launches == want, launches
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [t for t in ready if all(t is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    eager_main = torch.zeros_like(main)
    forced = [routes["bwd_bf16_rep"], f32, routes["bwd_bf16_rep"]]
    for k in range(3):
        xn, zn, _, dyn = make(rows, 1, layout="flat", seed=77 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        dy_strided.copy_(dyn)
        graph.replay()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "module_for_rows", _forcing("bwd_bf16_rep"))
            launches.clear()
            refs = (
                backward(dyn, xn, zn, w, EPS),
                backward(
                    dyn,
                    xn,
                    zn,
                    w,
                    EPS,
                    grad_weight=eager_main,
                    accumulate_grad_weight=True,
                ),
                backward(dy_strided, xn, zn, w, EPS),
            )
            assert launches == forced, launches
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(outs, refs, strict=True)):
            for what, a, b in zip(("dx", "dz", "dw"), got, ref, strict=True):
                assert torch.equal(a, b), f"replay {k}: {what} of call {i}"
        assert torch.equal(main, eager_main), k
        assert not bool(ws.any()), k
        assert not bool(taken[0].any()), k


def test_support_query_needs_a_route_for_every_call(monkeypatch):
    """``routes_every_call``: every exported architecture routes a module for
    every call (the role of every band of its ``FORWARD_BANDS``,
    ``STRIDED_FORWARD_BANDS`` and ``BACKWARD_BANDS``, and the long role of
    every band role of its ``LONG_BACKWARD_ROWS_PER_SM``; a role whose bands
    end without a ``None`` band, its own module), so the support query is
    true on this device; an architecture that routes no module for one band's
    role or one long role, or whose bands of a role without a module of its
    own end, or that is not exported, does not."""
    assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    assert not loader.routes_every_call("sm_90a")
    routes = loader.ROUTES
    for arch in loader.ARCHES:
        assert loader.routes_every_call(arch), arch
        band_roles = {
            band
            for table in (loader.FORWARD_BANDS, loader.STRIDED_FORWARD_BANDS)
            for _, band in table[arch]
        }
        band_roles |= {
            band for bands in loader.BACKWARD_BANDS[arch].values() for _, band in bands
        }
        longs = {band + loader.LONG for band in loader.LONG_BACKWARD_ROWS_PER_SM[arch]}
        assert longs == {"bwd_bf16_rep_long"}, (arch, longs)
        for role in sorted(band_roles | longs):
            fewer = {a: dict(by_role) for a, by_role in routes.items()}
            del fewer[arch][role]
            monkeypatch.setattr(loader, "ROUTES", fewer)
            assert not loader.routes_every_call(arch), (arch, role)
            monkeypatch.setattr(loader, "ROUTES", routes)
    # the bands of a BF16 weight gradient without their last (``None``) band:
    # sm_100a routes no module of the call role past them, sm_107a its static
    # backward
    bands = {a: dict(by_role) for a, by_role in loader.BACKWARD_BANDS.items()}
    for arch in loader.ARCHES:
        bands[arch]["bwd_bf16"] = bands[arch]["bwd_bf16"][:-1]
    monkeypatch.setattr(loader, "BACKWARD_BANDS", bands)
    for arch in loader.ARCHES:
        own = "bwd_bf16" in loader.ROUTES[arch]
        assert loader.routes_every_call(arch) == own, arch
    assert not loader.routes_every_call("sm_100a")
    assert loader.routes_every_call("sm_107a")


# (rows, SMs, CTAs) of the long forward: 32-row blocks, the fewest whole waves
# of 3 CTAs per SM whose CTAs walk at most 6 blocks, never more CTAs than blocks.
LONG_FORWARD_GRIDS = [
    (1, 148, 1),
    (32 * 444, 148, 444),  # a block per CTA of one wave
    (32 * 444 * 6, 148, 444),  # one wave, 6 blocks a CTA
    (32 * 444 * 6 + 1, 148, 888),  # a block more: two waves
    (140 * 148 + 1, 148, 444),  # B200 strided long band's first row: 648 blocks
    (160 * 148 + 1, 148, 444),  # B200 packed long band's first row: 741 blocks
    (400 * 148, 148, 444),  # its last row: 1850 blocks, 1 wave
    (2400 * 148, 148, 2220),  # 11100 blocks, 5 waves
    (32737 * 48, 148, 8436),  # 49106 blocks, 19 waves
    (600 * 152, 152, 912),  # GB300 packed extra-long band's last row: 2 waves
    (700 * 152, 152, 912),  # GB300 strided extra-long band's last row: 2 waves
    (2400 * 152, 152, 2280),  # 11400 blocks, 5 waves
    (32737 * 48, 152, 8208),  # 18 waves of 456
    (1740 * 212, 212, 2544),  # 11528 blocks, 4 waves
    (8192 * 48, 212, 2544),  # 12288 blocks, 4 waves of 636
    (2**31 - 1, 148, 25192 * 444),  # 2**26 blocks, 25192 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), LONG_FORWARD_GRIDS)
def test_long_forward_grid_is_whole_waves_of_at_most_six_blocks_a_cta(
    rows, sm_count, ctas
):
    name = loader.ROUTES["sm_100a"]["fwd_long"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "ctas_per_sm": 3,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 6,
        "rows_per_block": 32,
    }
    # shipped on every architecture, routed on sm_100a only
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert "fwd_long" not in loader.ROUTES["sm_103a"]
    assert "fwd_long" not in loader.ROUTES["sm_107a"]
    # the extra-long forward launches the same grid
    xlong = loader.MODULES[loader.ROUTES["sm_107a"]["fwd_xlong"]]
    assert loader.launch_grid(xlong, rows=rows, sm_count=sm_count) == (ctas, 1, 1)
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 32), 3 * sm_count
    assert -(-blocks // ctas) <= 6
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 6 < blocks
    # the grid is the long forward's whatever ``ctas`` the call passes
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    # the forward: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert fwd["launch"]["block"] == (64, 1, 1)
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 8),
        1,
        1,
    )


def test_extra_long_forward_is_the_long_forwards_grid():
    # the extra-long forward: the long forward's grid rule (32-row blocks,
    # whole waves of 3 CTAs per SM walking at most 6 blocks), 256 threads, no
    # PDL, routed on every architecture (FORWARD_BANDS gives it its rows)
    name = loader.ROUTES["sm_103a"]["fwd_xlong"]
    assert loader.ROUTES["sm_107a"]["fwd_xlong"] == name
    assert loader.ROUTES["sm_100a"]["fwd_xlong"] == name
    record = loader.MODULES[name]
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert record["role"] == "fwd_xlong"
    long_record = loader.MODULES[loader.ROUTES["sm_100a"]["fwd_long"]]
    assert record["grid_rule"] == long_record["grid_rule"]
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["use_pdl"] is False
    for rows, sm_count in (
        (400 * 148 + 1, 148),
        (520 * 148, 148),
        (700 * 148, 148),
        (240 * 152 + 1, 152),
        (600 * 152, 152),
        (700 * 152, 152),
        (3000 * 152, 152),
        (32737 * 48, 152),
        (2**31 - 1, 152),
        (175 * 212 + 1, 212),
        (280 * 212, 212),
        (1740 * 212, 212),
        (32737 * 48, 212),
        (2**31 - 1, 216),
    ):
        assert loader.launch_grid(
            record, rows=rows, sm_count=sm_count
        ) == loader.launch_grid(long_record, rows=rows, sm_count=sm_count)


# (rows, SMs, CTAs) of the mid forward: 16-row blocks, the fewest whole waves
# of 6 CTAs per SM whose CTAs walk at most 2 blocks, never more CTAs than blocks.
MID_FORWARD_GRIDS = [
    (1, 148, 1),
    (16 * 888, 148, 888),  # a block per CTA of one wave
    (16 * 888 * 2, 148, 888),  # one wave, 2 blocks a CTA
    (16 * 888 * 2 + 1, 148, 1776),  # a block more: two waves
    (12 * 148 + 1, 148, 112),  # B200 strided mid band's first row: 112 blocks
    (85 * 148 + 1, 148, 787),  # B200 packed mid band's first row: 787 blocks
    (100 * 148 + 1, 148, 888),  # 926 blocks, one wave
    (140 * 148, 148, 888),  # B200 strided mid band's last row: 1295 blocks
    (160 * 148, 148, 888),  # B200 packed mid band's last row: 1480 blocks
    (976 * 148, 148, 5328),  # 9028 blocks, 6 waves
    (2400 * 148 - 1, 148, 11544),  # 22200 blocks, 13 waves
    (4097 * 48, 148, 6216),  # 12291 blocks, 7 waves of 888
    (6500 * 148 + 1, 148, 30192),  # B200 last packed mid band: 34 waves
    (36 * 152 + 1, 152, 343),  # GB300 first packed mid band's first row
    (240 * 152, 152, 1824),  # GB300 first packed mid band's last row: 2 waves
    (250 * 152, 152, 1824),  # GB300 strided mid band's last row: 2 waves
    (976 * 152, 152, 5472),  # 9272 blocks, 6 waves
    (4097 * 48, 152, 6384),  # 12291 blocks, 7 waves of 912
    (6000 * 152 + 1, 152, 29184),  # GB300 second packed mid band's first row
    (8192 * 152, 152, 39216),  # its last row: 77824 blocks, 43 waves
    (59 * 212 + 1, 212, 782),  # R200 strided mid band's first row: 782 blocks
    (65 * 212 + 1, 212, 862),  # R200 packed mid band's first row: 862 blocks
    (100 * 212 + 1, 212, 1272),  # 1326 blocks, one wave
    (175 * 212, 212, 1272),  # R200 packed mid band's last row: 2319 blocks
    (200 * 212, 212, 2544),  # 2650 blocks, 2 waves
    (480 * 212, 212, 3816),  # R200 strided mid band's last row: 3 waves
    (800 * 212 + 1, 212, 6360),  # 10601 blocks, 5 waves
    (1740 * 212, 212, 12720),  # 23055 blocks, 10 waves
    (65536 * 48, 212, 99216),  # 196608 blocks, 78 waves
    (2**31 - 1, 148, 67109712),  # 2**27 blocks, 75574 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), MID_FORWARD_GRIDS)
def test_mid_forward_grid_is_whole_waves_of_at_most_two_blocks_a_cta(
    rows, sm_count, ctas
):
    # each architecture's mid forward module (warming later blocks in L2) has
    # this grid rule
    records = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        record = loader.MODULES[loader.ROUTES[arch]["fwd_mid"]]
        assert arch in record["arches"]
        assert record["grid_rule"] == {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "rows_per_block": 16,
        }
        assert record["launch"]["block"] == (128, 1, 1)
        records.append(record)
    record = records[0]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    for other in records[1:]:
        assert loader.launch_grid(other, rows=rows, sm_count=sm_count) == grid
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 16), 6 * sm_count
    assert -(-blocks // ctas) <= 2
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 2 < blocks
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid


# (rows, SMs, CTAs) of the pipelined forwards: one CTA per 16-row block below
# 8192 rows per SM, from there the fewest whole waves of 6 CTAs per SM whose
# CTAs walk at most 2 blocks, never more CTAs than blocks.
PIPE_FORWARD_GRIDS = [
    (1, 148, 1),
    (38 * 148 + 1, 148, 352),  # B200 first pipelined band's first row
    (85 * 148, 148, 787),  # its last row: 787 blocks
    (520 * 148 + 1, 148, 4811),  # B200 second pipelined band's first row
    (6500 * 148, 148, 60125),  # its last row: 60125 blocks
    (8192 * 148 - 1, 148, 75776),  # below 8192 rows per SM: a CTA a block
    (8192 * 148, 148, 38184),  # from 8192 rows per SM: 75776 blocks, 43 waves
    (32737 * 48, 148, 49728),  # 98211 blocks, 56 waves of 888
    (600 * 152 + 1, 152, 5701),  # GB300 first pipelined band's first row
    (6000 * 152, 152, 57000),  # its last row: 57000 blocks
    (8192 * 152 + 1, 152, 39216),  # GB300 last band's first row: 43 waves
    (280 * 212 + 1, 212, 3711),  # R200 pipelined band's first row: 3711 blocks
    (32737 * 48, 212, 98211),  # 98211 blocks, below 8192 rows per SM
    (8192 * 212, 212, 54696),  # 108544 blocks, 43 waves of 1272
    (65536 * 48, 212, 99216),  # 196608 blocks, 78 waves of 1272
    (2**24 - 1, 212, 525336),  # R200 pipelined band's last row: 413 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), PIPE_FORWARD_GRIDS)
def test_pipelined_forward_grid_is_a_block_a_cta_then_whole_waves(rows, sm_count, ctas):
    # every pipelined forward module (the deterministic package's grid rule)
    records = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for role in ("fwd_pipe", "fwd_pipe_packed"):
            name = loader.ROUTES[arch].get(role)
            if name is None:
                continue
            record = loader.MODULES[name]
            assert arch in record["arches"]
            assert record["grid_rule"] == {
                "ctas_per_sm": 6,
                "kind": "persistent_row_blocks",
                "max_blocks_per_cta": 2,
                "persistent_from_rows_per_sm": 8192,
                "rows_per_block": 16,
            }
            assert record["launch"]["block"] == (128, 1, 1)
            records.append(record)
    assert len(records) == 3
    grid = loader.launch_grid(records[0], rows=rows, sm_count=sm_count)
    for other in records[1:]:
        assert loader.launch_grid(other, rows=rows, sm_count=sm_count) == grid
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 16), 6 * sm_count
    if rows < 8192 * sm_count:
        assert ctas == blocks
    else:
        assert -(-blocks // ctas) <= 2
        assert ctas == blocks or ctas % wave == 0
        assert ctas == blocks or (ctas - wave) * 2 < blocks


# (rows, SMs) of the short forward: one CTA per 16-row block, up to 12 rows per
# SM on sm_100a and sm_103a and 59 on sm_107a (each row count is checked on
# every architecture: past an architecture's band the next band's forward runs
# it), so at most the SMs less 16 CTAs on sm_100a and sm_103a, and on sm_107a,
# at the 212 and 216 SMs listed (the bound needs 205 or more,
# SHORT_FORWARD_MIN_SMS), at most one wave of the 4 CTAs per SM its launch
# bounds give on the SMs less 16.
SHORT_FORWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48: 3 blocks
    (13 * 7, 148),
    (21 * 48, 148),  # 6.8 rows per SM on B200
    (10 * 148, 148),  # B200 at 10 rows per SM: 93 blocks
    (10 * 152, 152),  # GB300 at 10 rows per SM: 95 blocks
    (12 * 148, 148),  # B200 at 12 rows per SM: 111 blocks
    (12 * 152, 152),  # GB300 at 12 rows per SM: 114 blocks
    (10 * 212, 212),  # R200 at 10 rows per SM: 133 blocks
    (30 * 212, 212),  # R200 at 30 rows per SM: 398 blocks
    (1786 * 7, 212),  # T 1786, H 7: 782 blocks, 58.97 rows per SM
    (59 * 212, 212),  # R200 at 59 rows per SM: 782 blocks
    (59 * 216, 216),  # 216-SM R200 at 59 rows per SM: 797 blocks
]
# The most CTAs of the short forward's grid on sm_count SMs, from
# SHORT_FORWARD_MIN_SMS SMs (64 on sm_100a and sm_103a, 205 on sm_107a; on
# fewer SMs, as on a small MIG instance, the grid at the end of the
# architecture's band has more): at most the SMs less 16 on sm_100a and
# sm_103a, one wave of 4 CTAs per SM on the SMs less 16 on sm_107a.
SHORT_FORWARD_MAX_CTAS = {
    "sm_100a": lambda sm_count: sm_count - 16,
    "sm_103a": lambda sm_count: sm_count - 16,
    "sm_107a": lambda sm_count: 4 * (sm_count - 16),
}
# The fewest SMs on which the short forward's grid at the end of the
# architecture's band keeps SHORT_FORWARD_MAX_CTAS: ceil(12 * SMs / 16) <= SMs -
# 16 from 64 SMs on sm_100a and sm_103a, ceil(59 * SMs / 16) <= 4 * (SMs - 16)
# from 205 SMs on sm_107a (212 and 216 keep it; a small MIG instance does not).
SHORT_FORWARD_MIN_SMS = {"sm_100a": 64, "sm_103a": 64, "sm_107a": 205}


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_FORWARD_GRIDS)
def test_short_forward_grid_is_one_cta_a_block_beside_16_held_sms(rows, sm_count):
    name = loader.ROUTES["sm_100a"]["fwd_short"]
    routed = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["fwd_short"] == name
        end = _short_end(arch, sm_count)
        want = loader.ROUTES[arch][STRIDED_FORWARD_BANDS[arch][1][1]]
        if rows <= end:
            routed.append(arch)
            want = name
        got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
        assert got == want
    assert routed, (rows, sm_count)
    record = loader.MODULES[name]
    assert record["grid_rule"] == {"kind": "row_blocks", "rows_per_cta": 16}
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
    assert grid == (-(-rows // 16), 1, 1)
    for arch in routed:
        assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, grid)
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
    # the plain forward: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count) == (
        -(-rows // 8),
        1,
        1,
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_short_forward_grid_bound_holds_from_its_fewest_sms(arch):
    name = loader.ROUTES[arch]["fwd_short"]
    record = loader.MODULES[name]
    for sm_count in range(1, 1025):
        rows = _short_end(arch, sm_count)
        got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
        assert got == name
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
        fits = grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count)
        assert fits == (sm_count >= SHORT_FORWARD_MIN_SMS[arch]), (
            arch,
            sm_count,
            grid,
        )


def test_short_forward_grid_at_its_bands_end_keeps_the_bound_on_this_device():
    """On the running device, the short forward's grid at the end of the
    architecture's band (the most rows it runs) keeps SHORT_FORWARD_MAX_CTAS; skipped
    where the device has fewer SMs than the bound needs."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    name = loader.ROUTES[arch].get("fwd_short")
    if name is None:
        pytest.skip(f"{arch} routes no fwd_short module")
    if sm_count < SHORT_FORWARD_MIN_SMS[arch]:
        pytest.skip(
            f"{sm_count} SMs: the short forward's grid bound on {arch} needs "
            f"{SHORT_FORWARD_MIN_SMS[arch]} SMs or more"
        )
    rows = _short_end(arch, sm_count)
    assert loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False) == name
    assert loader.module_for_rows(arch, "fwd", rows + 1, sm_count, packed=False) != name
    grid = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sm_count)
    assert grid == (-(-rows // 16), 1, 1)
    assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, sm_count, grid)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_tiny_forward_runs_packed_rows_one_cta_an_8_row_block(sm_count):
    """The tiny forward, routed on every architecture for packed rows only: one
    128-thread CTA per 8-row block whatever the SMs, reading no stride (its
    arguments are ``x``, ``z``, ``w``, ``y``, the rows and ``eps``)."""
    name = loader.ROUTES["sm_100a"]["fwd_tiny"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {"kind": "row_blocks", "rows_per_cta": 8}
    assert record["launch"]["block"] == (128, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert [key for _kind, key in record["arg_plan"]] == [
        "x",
        "z",
        "w",
        "y",
        "rows",
        "eps",
        "grid_x",
        "grid_y",
        "grid_z",
    ]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["fwd_tiny"] == name
        end = _tiny_end(arch, sm_count)
        for rows in (1, 7, 8, 9, end):
            assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
            got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
            assert got != name, (arch, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
            assert grid == (-(-rows // 8), 1, 1)
            assert (
                loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7) == grid
            )
        assert loader.module_for_rows(arch, "fwd", end + 1, sm_count) != name


# (T, H) whose rows both the tiny forward (packed rows) and the short forward
# (other layouts) run on 148 SMs or more.
TINY_AND_SHORT_SHAPES = [(1, 1), (2, 48), (8, 1), (13, 7), (255, 2), (37, 48)]


@pytest.mark.parametrize(("T", "H"), TINY_AND_SHORT_SHAPES)
def test_tiny_forward_gives_the_short_forwards_bits(T, H, launches):
    """The tiny forward has the short forward's arithmetic: on the same rows,
    ``z`` once packed (the tiny forward) and once the gate columns of a wider
    projection (the short forward), ``y`` is the same bits."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    rows, end = T * H, _short_end(arch, loader.device_sm_count(device_index))
    if rows > end:
        pytest.skip(f"{rows} rows are past the short forward's band on {arch}")
    x, z, w, _ = make(max(T, 2), H, layout="projection", seed=90)
    x, z = x[:T], z[:T]
    y_short = forward(x, z, w, EPS)
    y_tiny = forward(x, z.contiguous(), w, EPS)
    routes = _routes()
    if T == 1:
        # a single token's stride is never read: both calls read packed rows
        assert launches == [routes["fwd_tiny"]] * 2, launches
    else:
        assert launches == [routes["fwd_short"], routes["fwd_tiny"]], launches
    assert torch.equal(y_short, y_tiny)


# (rows, SMs) of the small-row backwards: one cluster whatever the rows and the
# SMs, up to each program's ``max_rows``.
SHORT_BACKWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48
    (13 * 7, 152),
    (5 * 48, 148),
    (256, 148),
    (257, 160),
    (512, 152),
    (1024, 160),
    (1024, 212),
]
# The threads of one row in every backward program (16 lanes a row).
ROW_LANES = 16
# role -> (threads a CTA, teams a CTA, CTAs of the cluster) of each small-row
# backward (``BAND_ROLES``), one cluster each, whose CTAs each run their
# teams' row blocks: the short backward of a BF16 weight gradient (on every
# architecture; one 256-thread team a CTA, CTA ``r`` running the 16-row blocks
# ``r``, ``r + 16``, ...) and the exact tiny backwards (``_t512``: 4 teams of
# the 128-thread CTA of the deterministic backward at 2e989556 a CTA, team ``q``
# of CTA ``r`` running the 8-row block ``r + 16 q``).  A row block holds
# :func:`_cluster_block_rows` rows.
SMALL_ROW_BACKWARDS = {
    "bwd_bf16_short": (256, 1, 16),
    "bwd_bf16_t512": (512, 4, 16),
    "bwd_f32_t512": (512, 4, 16),
}


def _cluster_block_rows(name: str) -> int:
    """The rows of one row block of the small-row backward module ``name``
    (grid kind ``cluster``), derived from its record: the threads of its
    launch block over the teams of its role's CTA (``SMALL_ROW_BACKWARDS``),
    over ``ROW_LANES`` threads a row: 256 / 1 / 16 = 16 rows for the short
    backward, 512 / 4 / 16 = 8 rows (a team's block) for the exact tiny
    backwards."""
    record = loader.MODULES[name]
    assert record["grid_rule"]["kind"] == "cluster", (name, record["grid_rule"])
    threads = int(record["launch"]["block"][0])
    teams = SMALL_ROW_BACKWARDS[record["role"]][1]
    assert threads % (teams * ROW_LANES) == 0, (name, threads, teams)
    return threads // teams // ROW_LANES


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_BACKWARD_GRIDS)
def test_small_row_backwards_are_one_cluster(rows, sm_count):
    """Every small-row backward (the module each ``BAND_ROLES`` role routes,
    ``SMALL_ROW_BACKWARDS``): one module a role, shipped for exactly the
    architectures that route it; one cluster of its grid rule's ``cluster``
    CTAs of its role's threads whatever the rows and the SMs, up to its
    rule's ``max_rows``, which is one row block (:func:`_cluster_block_rows`)
    for each team of the cluster, so up to it no team runs more than one
    block; no workspace and no chunks."""
    assert set(loader.BAND_ROLES) == set(SMALL_ROW_BACKWARDS), loader.BAND_ROLES
    for role, (threads, teams, ctas) in SMALL_ROW_BACKWARDS.items():
        routed_on = [arch for arch in loader.ARCHES if role in loader.ROUTES[arch]]
        names = {loader.ROUTES[arch][role] for arch in routed_on}
        assert len(names) == 1, (role, routed_on, names)
        (name,) = names
        record = loader.MODULES[name]
        assert record["role"] == role
        assert record["arches"] == routed_on, (name, record["arches"])
        rule = record["grid_rule"]
        assert rule["kind"] == "cluster", name
        assert set(rule) == {"cluster", "kind", "max_rows"}, name
        assert int(rule["cluster"]) == ctas, name
        assert record["launch"]["block"] == (threads, 1, 1), name
        assert record["launch"]["cluster"] == (ctas, 1, 1), name
        block = _cluster_block_rows(name)
        assert int(rule["max_rows"]) == ctas * teams * block, (name, rule, block)
        assert loader.workspace_layout(name) == (), name
        assert loader.workspace_bytes(name) == 0, name
        if rows > int(rule["max_rows"]):
            continue
        assert loader.backward_ctas(name, rows, sm_count) == ctas
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=ctas)
        assert grid == (ctas, 1, 1)
        assert -(-rows // block) <= ctas * teams, (name, rows)
        with pytest.raises(ValueError, match="chunks"):
            loader.backward_chunks(name, rows)


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_BACKWARD_GRIDS)
def test_short_backward_is_one_cluster_up_to_its_max_rows(rows, sm_count):
    """The short backward (``bwd_bf16_short``): one module, routed on the
    architectures whose BF16 ``BACKWARD_BANDS`` start with its band, a count
    of rows ``(ROWS, n)`` with ``n`` at most its rule's ``max_rows`` (no FP32
    band runs it); ``module_for_rows`` runs it for 1 to ``n`` rows whatever
    the SMs and the next band's module (:func:`_band_module`) from ``n + 1``
    rows; one cluster of 16 CTAs of 256 threads, 6144 bytes of dynamic shared
    memory each, so that up to ``max_rows`` no CTA runs more than one 16-row
    block (:func:`_cluster_block_rows`); no workspace."""
    role = "bwd_bf16_short"
    routed_on = [arch for arch in loader.ARCHES if role in loader.ROUTES[arch]]
    assert routed_on, role
    (name,) = {loader.ROUTES[arch][role] for arch in routed_on}
    record = loader.MODULES[name]
    rule = record["grid_rule"]
    most = int(rule["max_rows"])
    assert rule == {"cluster": 16, "kind": "cluster", "max_rows": most}, rule
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["cluster"] == (16, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 6144
    assert record["arches"] == routed_on, record["arches"]
    block = _cluster_block_rows(name)
    assert block == 16, block
    for arch in loader.ARCHES:
        banded = [band for _, band in loader.BACKWARD_BANDS[arch]["bwd_bf16"]]
        assert (role in banded) == (arch in routed_on), (arch, banded)
        assert role not in [band for _, band in loader.BACKWARD_BANDS[arch]["bwd_f32"]]
    for arch in routed_on:
        last, band = loader.BACKWARD_BANDS[arch]["bwd_bf16"][0]
        assert band == role and isinstance(last, tuple), (arch, last, band)
        assert last[0] == loader.ROWS and 0 < int(last[1]) <= most, (arch, last)
        end = int(last[1])
        got = loader.module_for_rows(arch, "bwd_bf16", rows, sm_count)
        if rows <= end:
            assert got == name, (arch, rows)
        else:
            assert got != name, (arch, rows)
        after = loader.module_for_rows(arch, "bwd_bf16", end + 1, sm_count)
        assert after != name, arch
        assert after == _band_module(arch, "bwd_bf16", end + 1, sm_count), arch
    if rows > most:
        return
    assert loader.backward_ctas(name, rows, sm_count) == 16
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=16)
    assert grid == (16, 1, 1)
    assert -(-rows // block) <= grid[0]
    assert loader.workspace_layout(name) == () and loader.workspace_bytes(name) == 0


# Small-row backward shapes (T, H, layout): T * H rows in the bands in every
# layout, and row counts past them.
SHORT_BACKWARD_SHAPES = [
    (1, 1, "contiguous"),
    (1, 48, "contiguous"),
    (2, 48, "contiguous"),
    (5, 48, "contiguous"),
    (13, 7, "contiguous"),
    (8, 32, "contiguous"),
    (16, 16, "contiguous"),
    (256, 1, "flat"),
    (5, 48, "projection"),
    (13, 7, "head_major"),
    (2, 48, "misaligned"),
    (257, 1, "flat"),
    (37, 7, "contiguous"),
    (513, 1, "flat"),
    (621, 1, "flat"),
    (1025, 1, "flat"),
]


def _short_backward(rows: int, dw_dtype: torch.dtype) -> tuple[str, str | None]:
    """``(the module module_for_rows runs, that module where it is a small-row
    backward or None)`` for ``rows`` rows and a ``dw_dtype`` weight gradient on
    this device."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    name = loader.module_for_rows(
        arch, role, rows, loader.device_sm_count(device_index)
    )
    small = loader.MODULES[name]["role"] in loader.BAND_ROLES
    return name, (name if small else None)


@pytest.mark.parametrize("T,H,layout", SHORT_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_short_backward_dw_is_the_same_bits_and_takes_no_workspace(
    T, H, layout, dw_dtype, launches, monkeypatch
):
    """Within the FP64 budgets and the same bits on every launch
    (``check_all``), the module ``module_for_rows`` names launched; on a
    small-row backward (the rows of its ``BACKWARD_BANDS`` band) no stream
    workspace is made, a caller workspace holding anything is left as it is (the results
    the same bits) and one that is not a workspace is refused."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    x, z, w, dy = make(T, H, layout=layout, seed=80)
    name, short = _short_backward(T * H, dw_dtype)
    check_all(x, z, w, dy, EPS, dw_dtype=dw_dtype, repeats=5)
    fwd = _forward_module(T * H, _packed(T, H, layout))
    assert launches == [fwd, name, name] * 5, launches
    if short is None:
        return
    assert pkg._STREAM_WORKSPACES == {}

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    want = backward(dy, x, z, w, EPS, grad_weight=out())
    ws = workspace_for(T * H, x.device)
    ws.fill_(7)
    got = backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    torch.cuda.synchronize()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    assert bool((ws == 7).all()), "the small-row backward wrote the workspace"
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=ws.float())
    with pytest.raises(ValueError, match="uint8"):
        backward(dy, x, z, w, EPS, workspace=ws[1:])
    assert pkg._STREAM_WORKSPACES == {}


@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_short_backward_accumulates_and_skips_the_weight_gradient(dw_dtype):
    """On the small-row backward of 240 rows of a ``dw_dtype`` weight
    gradient (the short backward of a BF16 weight gradient, else the exact
    tiny backward): ``accumulate_grad_weight`` adds
    to the accumulator before the one rounding (within the budget, the same
    bits on every launch), ``compute_grad_weight=False`` leaves
    ``grad_weight`` as it is, and ``dx`` and ``dz`` are the same bits either
    way."""
    x, z, w, dy = make(5, 48, seed=81)
    _name, short = _short_backward(5 * 48, dw_dtype)
    if short is None:
        pytest.skip("this architecture routes no small-row backward at 240 rows")
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    first = None
    for _ in range(3):
        main = torch.full((D,), 0.25, dtype=dw_dtype, device="cuda")
        dx, dz, dw = backward(
            dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
        )
        assert dw is main
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
        first = main if first is None else first
        assert torch.equal(main, first)
    assert_dw_within(main, 0.25 + dw64, mag, what=f"accumulated dw {dw_dtype}")
    main = torch.full((D,), 3.0, dtype=dw_dtype, device="cuda")
    dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False)
    torch.cuda.synchronize()
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    assert torch.equal(main, torch.full_like(main, 3.0))


def test_calls_without_workspace_share_one_buffer_per_stream():
    """Calls without a workspace keep one per (device, stream, backward
    module): the module's buffers (``_expected_layout``) as views of their
    dtypes (``WORKSPACE_DTYPES``), zero after the calls."""
    # 1043 rows: past every small-row backward band, a backward with a workspace
    x, z, w, dy = make(149, 7)
    for _ in range(2):
        backward(dy, x, z, w, 1e-6)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    arch = loader.device_arch(device)
    stream = torch.cuda.current_stream().cuda_stream
    held = {k: v for k, v in pkg._STREAM_WORKSPACES.items() if k[1] == stream}
    assert (device, stream, _backward_module(149 * 7)) in held, list(held)
    for key, views in held.items():
        layout = _expected_layout(arch, loader.MODULES[key[2]]["role"])
        assert sorted(views) == sorted(b for b, _, _ in layout), key
        for buffer, _, elements in layout:
            assert views[buffer].dtype == WORKSPACE_DTYPES[buffer], (key, buffer)
            assert views[buffer].numel() == elements, (key, buffer)
        for view in views.values():
            assert not bool(view.view(torch.uint8).any())
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        backward(dy, x, z, w, 1e-6)
    side.synchronize()
    assert any(k[1] == side.cuda_stream for k in pkg._STREAM_WORKSPACES)


def test_checkpointed_steps_keep_only_the_stream_workspace(monkeypatch):
    """FlashInfer issue #7039: a non-reentrant checkpointed forward keeps nothing
    alive but its output, and once the first step has run, every checkpointed
    forward and backward leaves ``torch.cuda.memory_allocated()`` unchanged.  The
    buffers kept across steps are the workspace of the backward without a
    caller workspace, one per (device, stream, backward module) of
    ``workspace_bytes`` bytes, and the device's reserve of zero-filled
    workspaces for CUDA Graph captures (one allocation).  The first step's
    kept bytes are requested bytes, because the sizes of the cached blocks
    that serve a request depend on earlier tests."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    # 1056 rows: past every small-row backward band (a backward with a
    # workspace), and small enough for the caching allocator's small pool, whose
    # blocks are the requested sizes rounded up to 512 bytes.
    x, z, w, dy = make(22, 48, seed=63)
    leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
    for leaf in leaves:
        leaf.grad = torch.zeros_like(leaf)

    def checkpointed():
        return torch.utils.checkpoint.checkpoint(
            gated_rmsnorm, *leaves, EPS, use_reentrant=False
        )

    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    checkpointed().backward(dy)  # loads the modules, allocates the workspace
    torch.cuda.synchronize()
    kept = torch.cuda.memory_stats()["requested_bytes.all.current"] - before
    nbytes = workspace_bytes(1, x.device)
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _backward_module(22 * 48),
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    for view in pkg._STREAM_WORKSPACES[key].values():
        assert view.untyped_storage().nbytes() == nbytes
    assert nbytes == WORKSPACE_BYTES[loader.device_arch(x.device.index)], nbytes
    (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
    reserved = block.untyped_storage().nbytes()
    # 256-byte aligned workspaces: 29440 bytes each (about 1.8 MiB in all) on
    # sm_100a and sm_103a, 36608 bytes each (about 2.2 MiB) on sm_107a
    stride = RESERVE_STRIDE[loader.device_arch(x.device.index)]
    assert stride == -(-nbytes // 256) * 256, (stride, nbytes)
    assert reserved == pkg._RESERVE_SIZE * stride, reserved
    assert nbytes + reserved <= kept < nbytes + reserved + 512, (kept, nbytes)
    base = torch.cuda.memory_allocated()
    for step in range(3):
        y = checkpointed()
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - base - y.untyped_storage().nbytes()
        assert retained == 0, (step, retained)
        y.backward(dy)
        del y
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == base, step
    assert list(pkg._STREAM_WORKSPACES) == [key]


# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))


def test_no_entry_point_takes_a_mode_keyword(launches):
    """The mode is in the name (``nondet``): no entry point of the package has a
    ``deterministic`` parameter, and passing one is a ``TypeError`` before any
    launch, whatever its value and PyTorch's deterministic-algorithms setting;
    the autograd function takes no fifth input."""
    x, z, w, dy = make(5, 48, seed=64)
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    calls = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
    assert set(calls) | {"CakeRMSNormSiLUGateNondetFunction"} == set(pkg.__all__) - {
        "ARCHES",
        "SUPPORTED_COMPUTE_CAPABILITIES",
    }
    for name in calls:
        assert "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
    assert list(
        inspect.signature(pkg.CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for call in calls.values():
            for value in MODES:
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        for value in MODES:
            with pytest.raises(TypeError):
                pkg.CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launches == []
    torch.cuda.synchronize()
    assert not bool(ws.any())


def _forward_rows() -> dict[str, tuple[int, bool]]:
    """One row count of each forward module this device routes, the first row
    of the module's first ``FORWARD_BANDS`` band, else (a module only other
    layouts run) of its first ``STRIDED_FORWARD_BANDS`` band, at least 2 rows
    so that a projection layout's rows are not packed: ``{module: (rows,
    packed)}``."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    found: dict[str, tuple[int, bool]] = {}
    for packed, table in (
        (True, loader.FORWARD_BANDS),
        (False, loader.STRIDED_FORWARD_BANDS),
    ):
        start = 1 if packed else 2
        for last, _role in table[arch]:
            found.setdefault(_forward_module(start, packed), (start, packed))
            if last is not None:
                start = last * sm_count + 1
    return found


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_entry_point_alerts_under_deterministic_algorithms(warn_only, launches):
    """Every launch but the weight gradient of the fused or a row-claims
    backward repeats bit for bit, so under
    ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, those calls run without an
    alert and return the bits they return with the setting off, launching the
    same modules: the forward on every forward module this device routes; at
    the fewest rows of every ``BACKWARD_BANDS`` band of this device and of
    every long role's packed rows (:func:`_range_shape`) an FP32 weight
    gradient (accumulation included),
    ``compute_grad_weight=False`` (the BF16 bands' module, the fused
    backward included) and the autograd backward of a weight that needs no
    gradient, and where the band repeats a BF16 weight gradient
    (:func:`_dw_varies`) the BF16 weight gradient (caller workspace and none)
    and the autograd backward with it, also of a node built before the
    setting was enabled, and non-reentrant checkpointing; no rows; the
    workspace entry points, the reset and the support query.  Every band's
    module of this device is among the launches."""
    forward_rows = _forward_rows()
    assert len(forward_rows) >= 3, forward_rows
    ranges = [
        (band, first, last)
        for role in ("bwd_bf16", "bwd_f32")
        for band, first, last in _device_bands(role) + _device_long_bands(role)
    ]
    bands = [band for band, _, _ in ranges]
    shapes = list(dict.fromkeys(_range_shape(first, last) for _, first, last in ranges))
    cases = [make(T, H, seed=69 + i) for i, (T, H) in enumerate(shapes)]
    repeats = [not _dw_varies(T * H) for T, H in shapes]
    assert any(repeats), shapes
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    inputs = [
        make(rows, 1, layout="contiguous" if packed else "projection", seed=72)
        for rows, packed in forward_rows.values()
    ]
    ws = workspace_for(1, empty.device)

    def leaves(x, z, w, weight_grad=True):
        return [
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        ]

    def run(nodes):
        out = [forward(x, z, w, EPS) for x, z, w, _ in inputs]
        for (x, z, w, dy), bf16 in zip(cases, repeats, strict=True):
            if bf16:
                out += backward(dy, x, z, w, EPS)
            out += backward(dy, x, z, w, EPS, grad_weight=torch.empty(D, device="cuda"))
            main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
            out += backward(
                dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
            )
            out += backward(dy, x, z, w, EPS, compute_grad_weight=False)[:2]
            if bf16:
                out += backward(dy, x, z, w, EPS, workspace=ws)
        out += backward(empty, empty, empty, cases[0][2], EPS)
        nbytes = WORKSPACE_BYTES[loader.device_arch(empty.device.index)]
        assert workspace_bytes(1, empty.device) == nbytes
        assert workspace_for(1, empty.device).numel() == nbytes
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(empty.device)
        for (x, z, w, dy), node in zip(cases, nodes, strict=True):
            if node is not None:
                y, grads = node
                y.backward(dy)
                out += [y, *(t.grad for t in grads)]
                ts = leaves(x, z, w)
                yc = torch.utils.checkpoint.checkpoint(
                    gated_rmsnorm, *ts, EPS, use_reentrant=False
                )
                yc.backward(dy)
                out += [yc, *(t.grad for t in ts)]
            ts = leaves(x, z, w, weight_grad=False)
            gated_rmsnorm(*ts, EPS).backward(dy)
            out += [ts[0].grad, ts[1].grad]
            assert ts[2].grad is None
        return out

    def nodes():
        built = []
        for (x, z, w, _), bf16 in zip(cases, repeats, strict=True):
            grads = leaves(x, z, w) if bf16 else None
            built.append((gated_rmsnorm(*grads, EPS), grads) if bf16 else None)
        return built

    want = run(nodes())
    torch.cuda.synchronize()
    want_launches = list(launches)
    launches.clear()
    built = nodes()  # with the setting off
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run(built)
        torch.cuda.synchronize()
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), i
    assert launches == want_launches, launches
    modules = set(launches)
    assert set(forward_rows) <= modules, forward_rows
    for T, H in shapes:
        assert {
            _backward_module(T * H),
            _backward_module(T * H, torch.float32),
        } <= modules, (T, H)
    roles = {loader.MODULES[name]["role"] for name in modules}
    assert set(bands) <= roles, (bands, roles)
    assert not bool(ws.any())


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_graph_capture_under_deterministic_algorithms(warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, a CUDA Graph captures the
    forward and the backward (caller workspace and none, BF16 and FP32 weight
    gradients) without an alert at the fewest rows of the short backward's
    band (:func:`_band_shape`), whose BF16 weight gradient repeats its bits;
    its replays return the bits of eager calls, which lie within the FP64
    reference's budgets, the workspace zero after each."""
    shape = _band_shape("bwd_bf16_short")
    assert shape is not None
    T, H = shape
    assert not _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=75)
    ws = workspace_for(T * H, x.device)
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.empty(D, dtype=torch.float32, device="cuda")

    def step():
        forward(x, z, w, EPS, out=y)
        backward(dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws)
        backward(dy, x, z, w, EPS, grad_weight=main)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    want = [t.clone() for t in (y, dx, dz, dw, main)]
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
    mag = magnitudes(dy, x, z, EPS)
    assert_bf16_close(want[0], y64, what="y vs FP64")
    assert_bf16_close(want[1], dx64, what="dx vs FP64")
    assert_bf16_close(want[2], dz64, what="dz vs FP64")
    assert_dw_within(want[3], dw64, mag, what="dw vs FP64")
    assert_dw_within(want[4], dw64, mag, what="FP32 dw vs FP64")
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for k in range(3):
            for t in (y, dx, dz, dw, main):
                t.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            for i, (a, b) in enumerate(zip((y, dx, dz, dw, main), want, strict=True)):
                assert torch.equal(a, b), (k, i)
            assert not bool(ws.any()), k


# The bands whose backward of a weight gradient alerts under PyTorch's
# deterministic algorithms (the fused and the row-claims backwards; the
# row-claims backwards where a band runs them), and the rows of each band a
# test runs: its first, its last (a band with an end) and the first rows past
# one wave of claims (more 16-row blocks than 4 CTAs per SM, a band without
# an end).
ALERT_BAND_CASES = [
    pytest.param(band, edge, id=f"{band}-{edge}")
    for band in ("bwd_bf16_fused", "bwd_bf16_rc", "bwd_bf16_rc_mid")
    for edge in ("first", "last", "claims")
]


@pytest.mark.parametrize("band,edge", ALERT_BAND_CASES)
def test_backward_alerts_under_deterministic_algorithms(band, edge, launches):
    """Under ``torch.use_deterministic_algorithms(True)`` the paths that launch
    the order-varying weight-gradient sum of the fused or a row-claims
    backward (the first rows of the band ``band`` on this device and its last
    or, in a band without an end, the first rows past one wave of claims, as
    tokens of one head; skipped where this device routes no such band or the
    band has no such rows) -- the backward
    with ``compute_grad_weight=True`` (also accumulating) and the autograd
    backward when ``weight`` needs a gradient, also for a node built before
    the mode was enabled -- raise PyTorch's alert immediately before the
    launch, which never runs; under ``warn_only=True`` they warn and run the
    non-deterministic backward (never a fallback): ``dx`` and ``dz`` the bits
    of the call with the mode off, ``dw`` within the FP64 budget.  The
    forward, the workspace entry points, the reset, the support query, the
    backward without the weight gradient (on the same row-claims module) and
    with an FP32 one, and the autograd backward of a weight that needs no
    gradient run without a warning."""
    found = _device_band(band)
    if found is None:
        pytest.skip(f"this device routes no {band} band")
    first, last = found
    if edge == "first":
        rows = first
    elif edge == "last":
        if last is None:
            pytest.skip(f"the {band} band has no end")
        rows = last
    else:
        rows = 64 * loader.device_sm_count(torch.cuda.current_device()) + 1
        if rows < first or (last is not None and rows > last):
            pytest.skip(f"the {band} band holds no rows past one wave of claims")
    name = _backward_module(rows)
    assert name == _routes()[band] and _varies(name), (rows, name)
    x, z, w, dy = make(rows, 1, seed=65)
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(1, x.device)

    def leaves(weight_grad=True):
        return (
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        )

    node = gated_rmsnorm(*leaves(), EPS)  # built with the mode off
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forward(x, z, w, EPS)
        assert workspace_bytes(1, x.device) == workspace_for(1, x.device).numel()
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(x.device)
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert dw is None
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
        f32 = torch.empty(D, dtype=torch.float32, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
        xa, za, wa = leaves(weight_grad=False)
        gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        assert xa.grad is not None and wa.grad is None
    launches.clear()
    with pytest.raises(RuntimeError, match=ALERT):
        backward(dy, x, z, w, EPS)
    main = torch.full((D,), 0.25, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=ALERT):
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
    with pytest.raises(RuntimeError, match=ALERT):
        node.backward(dy)
    assert launches == []
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 0.25))
    assert not bool(ws.any())
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match=ALERT):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    assert_dw_within(dw, dw64, mag, what="warn_only dw")
    xa, za, wa = leaves()
    y = gated_rmsnorm(xa, za, wa, EPS)
    with pytest.warns(UserWarning, match=ALERT):
        y.backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert_dw_within(wa.grad, dw64, mag, what="warn_only autograd dw")
    fwd = _forward_module(rows)
    assert launches == [name, fwd, name], launches


def _alert(entry: str, deterministic_entry: str, *, warn_only: bool) -> str:
    """The exact text of the alert of ``entry`` of this package."""
    q = f"{pkg.__name__}.{entry}"
    d = f"{DET}.{deterministic_entry}"
    if warn_only:
        return (
            f"{q} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; the "
            f"non-deterministic kernels run. {d} is the deterministic one."
        )
    return (
        f"{q} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'. You can call {d} instead, "
        "or use the 'warn_only=True' option, if that's acceptable for your "
        "application."
    )


def test_alert_texts_name_the_entry_point_and_the_deterministic_one(launches):
    """At the fewest rows of the fused band (:func:`_band_shape`), the
    alert of the functional backward names it and the deterministic
    backward; the autograd backward's names the autograd backward and the
    deterministic autograd function (strict: ``RuntimeError`` before the
    launch; ``warn_only``: one ``UserWarning`` a call, then the launch)."""
    shape = _band_shape("bwd_bf16_fused")
    assert shape is not None
    T, H = shape
    assert _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=66)

    def node():
        torch.use_deterministic_algorithms(False)
        return gated_rmsnorm(
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(True),
            EPS,
        )

    functional = (
        "cake_rmsnorm_silu_gate_nondet_train_backward",
        "cake_rmsnorm_silu_gate_det_train_backward",
    )
    autograd = (
        "CakeRMSNormSiLUGateNondetFunction.backward",
        "CakeRMSNormSiLUGateDetFunction",
    )
    torch.use_deterministic_algorithms(True)
    launches.clear()
    with pytest.raises(RuntimeError) as info:
        backward(dy, x, z, w, EPS)
    assert str(info.value) == _alert(*functional, warn_only=False)
    y = node()
    torch.use_deterministic_algorithms(True)
    launches.clear()
    with pytest.raises(RuntimeError) as info:
        y.backward(dy)
    assert _alert(*autograd, warn_only=False) in str(info.value), str(info.value)
    assert launches == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        backward(dy, x, z, w, EPS)
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [_alert(*functional, warn_only=True)], got
    y = node()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        y.backward(dy)
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [_alert(*autograd, warn_only=True)], got
    bwd = _routes()["bwd_bf16_fused"]
    assert launches == [bwd, _forward_module(T * H), bwd], launches


def test_graph_capture_alerts_at_capture_and_replays_without_a_check():
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)``
    capturing a backward that computes the weight gradient at the fewest rows
    of the fused band (:func:`_band_shape`) warns once (the check runs on
    the host, at capture); the replays run the non-deterministic kernel
    without a warning, ``dx`` and ``dz`` the bits of the eager call and within
    the FP64 budget, ``dw`` within the budget, the workspace zero after
    each."""
    shape = _band_shape("bwd_bf16_fused")
    assert shape is not None
    T, H = shape
    assert _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=72)
    ws = workspace_for(T * H, x.device)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")

    def step():
        backward(dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the module before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dx_e, dz_e = dx.clone(), dz.clone()
    torch.use_deterministic_algorithms(True, warn_only=True)
    graph = torch.cuda.CUDAGraph()
    with pytest.warns(UserWarning) as record, torch.cuda.graph(graph):
        step()
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [
        _alert(
            "cake_rmsnorm_silu_gate_nondet_train_backward",
            "cake_rmsnorm_silu_gate_det_train_backward",
            warn_only=True,
        )
    ], got
    dw64, mag = fp64_reference(dy, x, z, w, EPS)[3], magnitudes(dy, x, z, EPS)
    _, dx64, dz64, _ = fp64_reference(dy, x, z, w, EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for k in range(3):
            dw.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            assert_dw_within(dw, dw64, mag, what=f"replay {k} dw")
            assert_bf16_close(dx, dx64, what=f"replay {k} dx")
            assert_bf16_close(dz, dz64, what=f"replay {k} dz")
            assert torch.equal(dx, dx_e) and torch.equal(dz, dz_e), k
            assert not bool(ws.any()), k


def test_deterministic_workspace_is_refused_naming_both_modes():
    """The backward (past the short backward and on it, which checks a caller
    workspace too) and the reset refuse an FP32 tensor, the deterministic
    package's workspace, naming both modes' workspaces, and leave it as it is;
    any other unfit buffer gets the byte-workspace requirement."""
    device = torch.device("cuda")
    nbytes = workspace_for(1, device).numel()
    fp32 = torch.zeros(8192, dtype=torch.float32, device="cuda")
    want = (
        "workspace is FP32, a workspace of the deterministic mode "
        f"({DET}.cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
        "non-deterministic backward needs its own zero-filled uint8 workspace "
        f"({pkg.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
    )
    for T in (65, 5):
        x, z, w, dy = make(T, 48, seed=73)
        for dtype in (torch.bfloat16, torch.float32):
            with pytest.raises(ValueError) as info:
                backward(
                    dy,
                    x,
                    z,
                    w,
                    EPS,
                    grad_weight=torch.empty(D, dtype=dtype, device="cuda"),
                    workspace=fp32,
                )
            assert str(info.value) == want, (T, dtype)
    with pytest.raises(ValueError) as info:
        reset_workspace(fp32)
    assert str(info.value) == want
    torch.cuda.synchronize()
    assert not bool(fp32.any())
    x, z, w, dy = make(65, 48, seed=74)
    device = x.device
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError) as info:
            backward(dy, x, z, w, EPS, workspace=bad)
        assert str(info.value) == (
            f"workspace must be a contiguous torch.uint8 tensor on {device} with at "
            f"least {nbytes} bytes starting at a 16-byte aligned address, zero "
            "before its first use "
            "(cake_rmsnorm_silu_gate_nondet_train_backward_workspace(rows, device), "
            "or cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace))"
        )
