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

Candidate-support LM-head log-probabilities and the detached-teacher forward KL
(flashinfer-ai/flashinfer#7636): the projection ``z = X @ W^T`` normalized over a caller-supplied
ragged candidate set per token (sampling-mask replay, restricted-support self-distillation).

Contract
--------
* ``X [T, H]`` BF16 (rows contiguous, leading stride ``>= H``), ``W [V, H]`` BF16 contiguous;
  ``H`` and ``V`` multiples of 256 and ``V <= SUPPORT_V_MAX``.
* ``support_ids [M]`` int32 / int64 and ``support_offsets [T + 1]`` int32 / int64: the candidates of
  row ``t`` are ``support_ids[offsets[t]:offsets[t + 1]]``, ``K_t = offsets[t + 1] - offsets[t]``,
  ``offsets[0] = 0``, nondecreasing, ``offsets[T] = M``, ``0 <= K_t <= k_cap``, ids in ``[0, V)`` and
  unique inside a row (any order; ids repeat freely across rows).  ``M <= 2^31 - 2^16``.
* Per row ``t`` with ``z_j = f32(bf16(X_t . W_{v_j}))`` (BF16 RN of an FP32 accumulation) and the
  temperature ``tau_t``: ``q_j = RN(z_j / tau_t)``, ``lse_t = log sum_j exp(q_j)`` over the row's
  candidates, ``logp_j = q_j - lse_t`` (FP32, the fixed sequence of :data:`CANONICAL_SEQUENCE`).
  Rows with ``K_t <= 1`` have ``logp = 0`` and ``KL = 0`` exactly (selects, not arithmetic).
* Outputs (FP32): the packed ``logp [M]`` in caller order, the selected ``logp [T]`` of the labels
  (``-100``: 0 with zero gradient; any other label must be one of the row's candidates; a label on an
  empty row is an error), the unreduced forward ``KL_t = sum_j Q_j (lq_j - logp_j)`` over the entries
  with ``Q_j = exp(lq_j) > 0`` for a detached FP32 teacher ``lq [M]`` aligned with ``support_ids``.
* Gradients to ``X`` and ``W`` only (BF16 ``dz`` at the support, FP32 accumulation, one cast each);
  the teacher, the temperature, the support metadata and the labels are constants.  A teacher or a
  temperature tensor with ``requires_grad=True`` is a ``ValueError``.
* Temperature: a finite positive Python number (or 0-dim CPU tensor) or an FP32 ``[T]`` tensor on
  ``X``'s device, validated by :func:`cake_backend._resolve_temperature` (strict on every row).
  ``None`` / ``1.0`` / ``ones(T)`` are bitwise equal and a scalar ``s`` equals ``full((T,), s)`` bitwise:
  every form becomes one FP32 ``[T]`` row vector read by the same code.
* Invalid support (offsets, cap, id range, duplicates, labels) is detected by one validation stage
  (:meth:`SupportReferenceEngine.support_check`, the device kernel K1 of the cake backend) that forms
  a deterministic int64 error word (:func:`support_error_word`).  ``check_inputs=False`` (default) on
  CUDA queues ``torch._assert_async(word == NO_ERROR)``: no host synchronization, and a violation ends
  the CUDA context like an out-of-range PyTorch index.  ``check_inputs=True`` (and every CPU call)
  reads the word back once and raises ``ValueError`` naming the rule and the row.

``backend="reference"`` runs every stage with PyTorch operators at the kernels' rounding
boundaries: the row stages follow :data:`CANONICAL_SEQUENCE` operation by operation (FP32
``div.rn``, exact ``fma``, the 32-lane strided sums with the xor butterfly), the projection and the
gradient products run as BF16 x BF16 products with FP32 accumulation per token chunk of
``chunk_size`` rows (a development aid: it materializes a ``[chunk_size, V]`` block and is not
allocation-free).

``backend="cake"`` runs the generated kernels (SM100 / SM103, CUDA tensors), with no ``[T, V]`` or
``[chunk, V]`` tensor in the sparse form:

* the validation stage (int32 copies of int64 ids / offsets, clamped), then the logits at the support
  ``z [M]`` (BF16): sparse = a gather GEMM over tiles of 128 consecutive pairs (the pairs' ``W`` rows
  against their tokens' ``X`` rows), dense = the shipped logits GEMM per token chunk and vocabulary
  block (``W[v0:v1]`` views) and a gather of the support's columns;
* the forward row stage (one warp per row) and the backward row stage, which writes ``dz [M]`` (BF16);
* ``dX``: sparse = one CTA per token and 1024 columns adding its pairs' ``W`` rows in pair order (rows
  of fewer than two candidates write +0 without reading ``W``), dense = a scatter of ``dz`` into the
  chunk's ``[chunk, vb]`` block and the shipped dX GEMM per block, blocks added in ascending order in
  FP32, one cast per chunk;
* ``dW``: sparse = the unique keys ``(v << 32) | p`` of the pairs (a pair of a row no upstream reaches
  takes a sentinel row), sorted, then one CTA per key range adding the ``X`` rows of each id's pairs in
  pair order (a long segment, an id most tokens share, is cut into parts summed in part order); the
  BF16 ``dW`` is written directly (ids in no support stay +0) up to
  :data:`SUPPORT_DW_GLOBAL_MAX_PAIRS` pairs, above that pair ranges of ``chunk_size * ceil(M / T)``
  pairs accumulate into an FP32 ``[V, H]`` sum cast once; dense = the shipped weight-gradient GEMM per
  block, accumulated in chunk order.

The pass form of each pass is a host rule on host-known sizes (:data:`SUPPORT_DENSE_MIN_MEAN_K`):
deterministic, no synchronization.  Every sum has one owner and one order (no atomics on values), so
reruns are bitwise equal; the bits depend on the pass form, ``chunk_size``, the dense vocabulary block
and the sparse split.  The device bytes of a step by bucket: :func:`support_memory_report`.  A token
of two or more candidates that no upstream reaches gets ``dX = +0`` for finite ``W`` (``+0`` times its
rows), as the reference backend does.
"""

from __future__ import annotations

import functools
import math
import operator
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch.autograd.function import once_differentiable

from .cake_backend import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GEOMETRY,
    IGNORE_INDEX,
    Geometry,
    _assert_on_device,
    _device_constants,
    _eager_launch,
    _ffi_stream_context,
    _mm_fp32,
    _resolve_temperature,
    _temperature_rows,
    arch_for,
    gemm_scalars,
    record_abi,
    record_for,
)
from .cake_jit import MODULES, select_module

BACKENDS = ("cake", "reference")
SUPPORT_INDEX_DTYPES = (torch.int32, torch.int64)
# Largest vocabulary of the validation stage's per-warp shared-memory bitmap (V bits = 192 KiB at the limit).
SUPPORT_V_MAX = 1536 * 1024
# Packed pair indices (label positions, CSC keys) are int32 in the kernels, whose 32-wide loops over a row's pairs
# step past the end of the row: ``M <= 2^31 - 2^16`` keeps every such index and the pair tile count in int32.
SUPPORT_M_MAX = 2**31 - 2**16
LANES = 32  # lanes of the row stages' warp: lane l takes the row's entries l, l + 32, ... in ascending order
_BUTTERFLY = (
    16,
    8,
    4,
    2,
    1,
)  # xor offsets of the warp sum, in this order (``val += shfl_xor(val, offset)``)
_LANE_BUDGET = 1 << 24  # FP32 slots of one lane-fold block of the reference engine

CANONICAL_SEQUENCE = r"""
row t, entries j = offsets[t] .. offsets[t + 1] - 1 in caller order, i = j - offsets[t], z_j = f32(z_bf16[j]):
  q_j    = div.rn(z_j, tau_t)
  m_t    = max_j q_j                                         (exact; -inf on an empty row)
  s_t    = lane_sum_j expf(q_j - m_t)
  lse_t  = (K_t == 0) ? +0 : m_t + logf(s_t)
  logp_j = (K_t <= 1) ? +0 : q_j - lse_t
  sel_t  = (lpos_t < 0) ? +0 : logp[lpos_t]
  Q_j    = expf(lq_j)
  kl_t   = (K_t <= 1) ? +0 : lane_fold_j [acc = (Q_j > 0) ? fma(Q_j, lq_j - logp_j, acc) : acc]
backward, upstreams g [M] (packed), d [T] (selected), e [T] (KL); an absent upstream is skipped:
  u_j    = +0 ; u_j = g_j (if g) ; u_j = (j == lpos_t) ? u_j + d_t : u_j (if d) ; u_j = fma(-e_t, Q_j, u_j) (if e)
  U_t    = lane_sum_j u_j
  p_j    = expf(div.rn(z_j, tau_t) - lse_t)
  act_t  = K_t >= 2 and (g or e or (d and lpos_t >= 0))
  dz_j   = act_t ? bf16_rn(div.rn(fma(-p_j, U_t, u_j), tau_t)) : +0
lane_sum / lane_fold: lane l = i % 32 starts at +0 and combines its entries i = l, l + 32, ... in ascending i;
  then val += shfl_xor(val, o) for o = 16, 8, 4, 2, 1 (lane 0's value).  FP32 throughout, RN, no FTZ,
  expf / logf the precise calls, no contraction other than the explicit fma.
"""

# ---------------------------------------------------------------------------
# Error word of the validation stage
# ---------------------------------------------------------------------------

NO_ERROR = 2**63 - 1  # the word of a valid support (atomicMin identity)
ERROR_BAND_SHIFT = 61
ERROR_ROW_SHIFT = 8
# Band 0: the offsets themselves (the pair -> row mapping is undefined while one of them holds), so any
# band-0 violation wins the minimum over every content violation.  Band 1: the content of a valid layout.
ERR_OFFSETS_FIRST = 0x01  # row 0: offsets[0] != 0
ERR_OFFSETS_DECREASING = 0x02  # row t: offsets[t + 1] < offsets[t]
ERR_OFFSETS_LAST = 0x03  # row T: offsets[T] != M
ERR_OVER_CAP = 0x10  # row t: K_t > k_cap
ERR_ID_RANGE = 0x11  # row t: an id outside [0, V)
ERR_DUPLICATE_ID = 0x12  # row t: an id twice
ERR_LABEL_EMPTY_ROW = 0x13  # row t: K_t == 0 and label != -100
ERR_LABEL_NOT_IN_SUPPORT = (
    0x14  # row t: K_t > 0, label != -100 and not one of the row's ids
)
SUPPORT_ERRORS: dict[int, tuple[int, str]] = {
    ERR_OFFSETS_FIRST: (0, "support_offsets[0] must be 0"),
    ERR_OFFSETS_DECREASING: (
        0,
        "support_offsets must be nondecreasing (support_offsets[row + 1] < support_offsets[row])",
    ),
    ERR_OFFSETS_LAST: (0, "support_offsets[T] must equal M = support_ids.numel()"),
    ERR_OVER_CAP: (
        1,
        "the row holds more than k_cap candidates (a recorded support is never trimmed)",
    ),
    ERR_ID_RANGE: (1, "the row holds a candidate id outside [0, V)"),
    ERR_DUPLICATE_ID: (1, "the row holds the same candidate id twice"),
    ERR_LABEL_EMPTY_ROW: (
        1,
        "the row has an empty support but a label other than -100",
    ),
    ERR_LABEL_NOT_IN_SUPPORT: (
        1,
        "the row's label is neither -100 nor one of its candidate ids",
    ),
}


def support_error_word(code: int, row: int) -> int:
    """The int64 error word of one violation: ``band << 61 | row << 8 | code``; the validation stage
    reports the minimum over every violation it finds (order-free, hence deterministic)."""
    band = SUPPORT_ERRORS[code][0]
    return (band << ERROR_BAND_SHIFT) | (int(row) << ERROR_ROW_SHIFT) | int(code)


def decode_support_error(word: int) -> tuple[int, int]:
    """``(code, row)`` of an error word other than :data:`NO_ERROR`."""
    word = int(word)
    return word & 0xFF, (word >> ERROR_ROW_SHIFT) & (
        (1 << (ERROR_BAND_SHIFT - ERROR_ROW_SHIFT)) - 1
    )


def describe_support_error(word: int) -> str:
    code, row = decode_support_error(word)
    rule = SUPPORT_ERRORS.get(code, (None, f"unknown rule code {code:#x}"))[1]
    return f"invalid candidate support at row {row}: {rule} (error word {int(word):#x})"


_ASSERT_MESSAGE = (
    "invalid candidate support (support_offsets / support_ids / labels); call with check_inputs=True for "
    "the rule and the row"
)

# ---------------------------------------------------------------------------
# Host validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportProblem:
    """Validated geometry and options of one call."""

    num_rows: int  # T
    hidden: int  # H
    vocab: int  # V
    num_pairs: int  # M
    k_cap: int
    chunk: int  # token chunk of the reference engine; the cake backend's dW pair range is chunk * ceil(M / T)
    has_labels: bool
    has_teacher: bool


def _index_tensor(t, name: str, numel: Optional[int] = None) -> None:
    if not isinstance(t, torch.Tensor) or t.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
    if t.dtype not in SUPPORT_INDEX_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {t.dtype}")
    if numel is not None and int(t.numel()) != numel:
        raise ValueError(f"{name} must have {numel} entries, got {int(t.numel())}")


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def validate_support_inputs(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    teacher_logprobs: Optional[torch.Tensor] = None,
    *,
    k_cap: int,
    chunk_size: Optional[int] = None,
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> SupportProblem:
    """Host checks (shapes, dtypes, devices, limits; no value read, no synchronization).  The value rules
    of the support run in the validation stage (:meth:`SupportReferenceEngine.support_check`)."""
    for name, t in (("X", X), ("W", W)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
    if X.ndim != 2 or X.dtype != torch.bfloat16:
        raise ValueError("X must be a BF16 [T, H] tensor")
    if W.ndim != 2 or W.dtype != torch.bfloat16:
        raise ValueError("W must be a BF16 [V, H] tensor")
    if not W.is_contiguous():
        raise ValueError("W must be contiguous")
    T, H = (int(s) for s in X.shape)
    V, w_hidden = (int(s) for s in W.shape)
    if w_hidden != H:
        raise ValueError(
            f"W has {w_hidden} columns, X has {H}: the hidden sizes differ"
        )
    h_multiple = math.lcm(
        geometry.hidden_multiple, geometry.dx_item_cols, geometry.dw_item_cols
    )
    v_multiple = math.lcm(geometry.vocab_multiple, geometry.logits_item_cols)
    if H < 1 or H % h_multiple:
        raise ValueError(f"H must be a positive multiple of {h_multiple}, got {H}")
    if V < 1 or V % v_multiple:
        raise ValueError(f"V must be a positive multiple of {v_multiple}, got {V}")
    if V > SUPPORT_V_MAX:
        raise ValueError(
            f"V must not exceed {SUPPORT_V_MAX} (the validation stage's shared-memory bitmap), got {V}"
        )
    if X.stride(1) != 1:
        raise ValueError("X rows must be contiguous (stride(1) == 1)")
    if T > 1 and X.stride(0) < H:
        raise ValueError("X must be row-major with a leading stride of at least H")
    _index_tensor(support_ids, "support_ids")
    _index_tensor(support_offsets, "support_offsets", T + 1)
    M = int(support_ids.numel())
    if M > SUPPORT_M_MAX:
        raise ValueError(
            f"support_ids must have at most 2^31 - 2^16 = {SUPPORT_M_MAX} entries, got {M}"
        )
    k_cap = _positive_int(k_cap, "k_cap")
    if T == 0 and M:
        raise ValueError("support_ids must be empty when X has no rows")
    if T * k_cap < M:
        raise ValueError(
            f"support_ids has {M} entries, more than T * k_cap = {T * k_cap}: a row holds more than k_cap "
            "candidates (a recorded support is never trimmed)"
        )
    chunk = (
        DEFAULT_CHUNK_SIZE
        if chunk_size is None
        else _positive_int(chunk_size, "chunk_size")
    )
    if labels is not None:
        if (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or int(labels.numel()) != T
        ):
            raise ValueError("labels must be a [T] tensor")
        if labels.dtype != torch.int64:
            raise ValueError("labels must be int64 (-100 marks ignored rows)")
    if teacher_logprobs is not None:
        t = teacher_logprobs
        if not isinstance(t, torch.Tensor) or t.ndim != 1 or int(t.numel()) != M:
            raise ValueError(
                "teacher_logprobs must be an FP32 [M] tensor aligned with support_ids"
            )
        if t.dtype != torch.float32:
            raise ValueError(f"teacher_logprobs must be FP32, got {t.dtype}")
        if t.requires_grad:
            raise ValueError(
                "teacher_logprobs is a constant and must not require grad; pass teacher_logprobs.detach()"
            )
    device = X.device
    for name, t in (
        ("W", W),
        ("support_ids", support_ids),
        ("support_offsets", support_offsets),
        ("labels", labels),
        ("teacher_logprobs", teacher_logprobs),
    ):
        if t is not None and t.device != device:
            raise ValueError(
                f"{name} must be on the device of X ({device}), got {t.device}"
            )
    return SupportProblem(
        num_rows=T,
        hidden=H,
        vocab=V,
        num_pairs=M,
        k_cap=k_cap,
        chunk=chunk,
        has_labels=labels is not None,
        has_teacher=teacher_logprobs is not None,
    )


# ---------------------------------------------------------------------------
# Temperature (the package's shared resolver, :func:`cake_backend._resolve_temperature`)
# ---------------------------------------------------------------------------


def _support_tau_rows(form, T: int, device: torch.device) -> torch.Tensor:
    """The FP32 ``[T]`` row vector every stage reads, from a resolved temperature ``form``
    (:func:`cake_backend._resolve_temperature`): ``full((T,), s)`` for a scalar (``s = 1.0`` for the default
    path ``None``), the caller's contiguous tensor otherwise.  One code path for every form, so ``None``,
    ``1.0`` and ``ones(T)`` are bitwise equal and a scalar ``s`` equals ``full((T,), s)`` bitwise."""
    tau = _temperature_rows(form, None, T, device)
    if tau is None:
        return torch.full((T,), 1.0, dtype=torch.float32, device=device)
    return tau


# ---------------------------------------------------------------------------
# Canonical FP32 primitives (the row stages' operations, emulated exactly)
# ---------------------------------------------------------------------------


def div_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``div.rn.f32(a, b)``: IEEE FP32 division, round to nearest even.  Evaluated in FP64 and rounded once
    to FP32, which is exact for division (53 >= 2 * 24 + 2), independent of how PyTorch lowers FP32 ``/``."""
    return (a.double() / b.double()).float()


def fma_rn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``fma.rn.f32(a, b, c)``: ``a * b + c`` rounded once to FP32 (no intermediate rounding).

    ``a * b`` is exact in FP64; the FP64 sum is rounded to odd (TwoSum error term, one ulp step toward it
    when the rounded sum is even), and a round-to-odd value with 53 >= 24 + 2 bits rounds to the correctly
    rounded FP32 result."""
    a, b, c = a.double(), b.double(), c.double()
    p = a * b
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    even = (s.view(torch.int64) & 1) == 0
    step = (err != 0) & even & torch.isfinite(s)
    toward = torch.where(
        err > 0, torch.full_like(s, math.inf), torch.full_like(s, -math.inf)
    )
    return torch.where(step, torch.nextafter(s, toward), s).float()


# ---------------------------------------------------------------------------
# Row layout and the lane-ordered folds
# ---------------------------------------------------------------------------


@dataclass
class SupportRows:
    """The pair -> row mapping of a support (offsets clamped to ``[0, M]``, ids clamped to ``[0, V)``: in
    bounds for any input; exact for a valid support)."""

    num_rows: int
    num_pairs: int
    offsets: torch.Tensor  # int64 [T + 1], clamped
    ids: torch.Tensor  # int64 [M], clamped
    row: torch.Tensor  # int64 [M]: the row of each pair
    local: torch.Tensor  # int64 [M]: the pair's index inside its row
    counts: torch.Tensor  # int64 [T]: K_t of the clamped offsets
    steps: int  # ceil(k_max / 32), k_max = min(k_cap, V, M): the lane-fold depth

    @classmethod
    def build(
        cls,
        ids: torch.Tensor,
        offsets: torch.Tensor,
        *,
        num_rows: int,
        vocab: int,
        k_cap: int,
    ) -> "SupportRows":
        T, M = int(num_rows), int(ids.numel())
        off = offsets.to(torch.int64).clamp(0, M)
        idc = ids.to(torch.int64).clamp(0, vocab - 1)
        counts = (off[1:] - off[:-1]).clamp_min(0)
        k_max = max(1, min(int(k_cap), int(vocab), M))
        steps = -(-k_max // LANES)
        if M:
            j = torch.arange(M, dtype=torch.int64, device=ids.device)
            row = torch.searchsorted(off[1:].contiguous(), j, right=True).clamp_max(
                T - 1
            )
            local = (j - off[row]).clamp(0, steps * LANES - 1)
        else:
            row = local = torch.zeros(0, dtype=torch.int64, device=ids.device)
        return cls(T, M, off, idc, row, local, counts, steps)

    def blocks(self) -> list[tuple[int, int]]:
        per = max(1, _LANE_BUDGET // (self.steps * LANES))
        return [
            (r0, min(self.num_rows, r0 + per)) for r0 in range(0, self.num_rows, per)
        ]


def _butterfly_sum(acc: torch.Tensor) -> torch.Tensor:
    lanes = torch.arange(LANES, device=acc.device)
    for offset in _BUTTERFLY:
        acc = acc + acc[:, lanes ^ offset]
    return acc[:, 0]


def lane_fold(
    rows: SupportRows,
    values: tuple[torch.Tensor, ...],
    combine,
    present: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP32 ``[T]``: per row, lane ``l`` starts at ``+0`` and folds ``acc = combine(acc, *values_j)`` over its
    entries ``i = l, l + 32, ...`` in ascending order (``present_j`` False: the entry is skipped), then the
    xor butterfly sum of the 32 lanes (:data:`CANONICAL_SEQUENCE`)."""
    T, M = rows.num_rows, rows.num_pairs
    device = rows.row.device
    out = torch.zeros(T, dtype=torch.float32, device=device)
    if M == 0:
        return out
    steps = rows.steps
    lanes = rows.local % LANES
    step = rows.local // LANES
    for r0, r1 in rows.blocks():
        n = r1 - r0
        size = n * steps * LANES
        inblk = (rows.row >= r0) & (rows.row < r1)
        if present is not None:
            inblk = inblk & present
        slot = torch.where(
            inblk, ((rows.row - r0) * steps + step) * LANES + lanes, size
        )
        bufs = []
        for v in values:
            buf = torch.zeros(size + 1, dtype=torch.float32, device=device)
            buf.index_put_((slot,), v.float())
            bufs.append(buf[:size].view(n, steps, LANES))
        pres = torch.zeros(size + 1, dtype=torch.bool, device=device)
        pres.index_put_((slot,), torch.ones_like(slot, dtype=torch.bool))
        pres = pres[:size].view(n, steps, LANES)
        acc = torch.zeros(n, LANES, dtype=torch.float32, device=device)
        for s in range(steps):
            acc = torch.where(pres[:, s], combine(acc, *(b[:, s] for b in bufs)), acc)
        out[r0:r1] = _butterfly_sum(acc)
    return out


def _add(acc: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return acc + v


def _fma_acc(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return fma_rn(a, b, acc)


def lane_sum(rows: SupportRows, values: torch.Tensor, present=None) -> torch.Tensor:
    return lane_fold(rows, (values,), _add, present)


# ---------------------------------------------------------------------------
# Reference engine: every stage of the support path with PyTorch operators
# ---------------------------------------------------------------------------


@dataclass
class SupportCheck:
    word: torch.Tensor  # int64 [1]: NO_ERROR or the minimum violation word
    lpos: Optional[
        torch.Tensor
    ]  # int32 [T]: packed index of the row's label, -1 for -100 (labels given)


def _blk_rows(rows: SupportRows, r0: int, r1: int):
    inblk = (rows.row >= r0) & (rows.row < r1)
    return inblk, torch.where(inblk, rows.row - r0, r1 - r0)


class SupportReferenceEngine:
    """PyTorch implementation of the support stages at the kernels' rounding boundaries.

    ``support_check`` is the validation stage K1 (the error word and the label positions, no host
    synchronization); ``support_logits`` the projection at the support (BF16 RN of an FP32 accumulation);
    ``row_finalize`` / ``row_grad`` the FP32 row stages F2 / G of :data:`CANONICAL_SEQUENCE`, operation by
    operation; ``grad_x`` / ``grad_w`` the gradient products (BF16 ``dz`` x BF16 operand, FP32 accumulation,
    one cast; token chunks in ascending order, the weight gradient accumulated across chunks in FP32)."""

    @staticmethod
    def support_check(
        ids: torch.Tensor,
        offsets: torch.Tensor,
        labels: Optional[torch.Tensor],
        *,
        num_rows: int,
        vocab: int,
        k_cap: int,
    ) -> SupportCheck:
        T, M = int(num_rows), int(ids.numel())
        device = ids.device
        i64 = dict(dtype=torch.int64, device=device)
        none = torch.full((), NO_ERROR, **i64)
        band1 = 1 << ERROR_BAND_SHIFT
        off = offsets.to(torch.int64)
        rows_t = torch.arange(T, **i64)
        words = [
            torch.where(
                off[0] != 0,
                torch.full((), support_error_word(ERR_OFFSETS_FIRST, 0), **i64),
                none,
            ),
            torch.where(
                off[T] != M,
                torch.full((), support_error_word(ERR_OFFSETS_LAST, T), **i64),
                none,
            ),
        ]
        k_raw = off[1:] - off[:-1]
        row_word = rows_t << ERROR_ROW_SHIFT
        if T:
            words.append(
                torch.where(k_raw < 0, row_word | ERR_OFFSETS_DECREASING, none).amin()
            )
            words.append(
                torch.where(
                    k_raw > int(k_cap), band1 | row_word | ERR_OVER_CAP, none
                ).amin()
            )
        rows = SupportRows.build(ids, offsets, num_rows=T, vocab=vocab, k_cap=k_cap)
        id64 = ids.to(torch.int64)
        if M:
            pair_word = band1 | (rows.row << ERROR_ROW_SHIFT)
            bad = (id64 < 0) | (id64 >= vocab)
            words.append(torch.where(bad, pair_word | ERR_ID_RANGE, none).amin())
        if M >= 2:
            # one key per (row, id): an id that repeats inside a row repeats its key (an out-of-range id is
            # already reported with a smaller code of the same row, so its key may alias)
            key, _ = torch.sort((rows.row << 32) | id64.clamp(0, (1 << 32) - 1))
            dup = key[1:] == key[:-1]
            words.append(
                torch.where(
                    dup,
                    band1 | ((key[1:] >> 32) << ERROR_ROW_SHIFT) | ERR_DUPLICATE_ID,
                    none,
                ).amin()
            )
        lpos = None
        if labels is not None and T:
            lab = labels.to(torch.int64)
            need = lab != IGNORE_INDEX
            if M:
                j = torch.arange(M, **i64)
                match = id64 == lab[rows.row]
                first = torch.full((T,), M, **i64).scatter_reduce(
                    0, rows.row, torch.where(match, j, M), "amin", include_self=True
                )
            else:
                first = torch.full((T,), M, **i64)
            found = first < M
            words.append(
                torch.where(
                    need & (rows.counts == 0),
                    band1 | row_word | ERR_LABEL_EMPTY_ROW,
                    none,
                ).amin()
            )
            words.append(
                torch.where(
                    need & (rows.counts > 0) & ~found,
                    band1 | row_word | ERR_LABEL_NOT_IN_SUPPORT,
                    none,
                ).amin()
            )
            lpos = torch.where(need & found, first, -1).to(torch.int32)
        elif labels is not None:
            lpos = torch.zeros(0, dtype=torch.int32, device=device)
        word = torch.stack(words).amin().reshape(1)
        return SupportCheck(word=word, lpos=lpos)

    @staticmethod
    def support_logits(
        X: torch.Tensor, W: torch.Tensor, rows: SupportRows, chunk: int
    ) -> torch.Tensor:
        """BF16 ``z [M]``: ``bf16(X_t . W_{v_j})`` per pair (FP32 accumulation, one RN), token chunk by chunk."""
        z = torch.zeros(rows.num_pairs, dtype=torch.bfloat16, device=X.device)
        if rows.num_pairs == 0:
            return z
        for r0 in range(0, rows.num_rows, chunk):
            r1 = min(rows.num_rows, r0 + chunk)
            zblk = _mm_fp32(X[r0:r1], W.t()).to(torch.bfloat16)
            inblk, rr = _blk_rows(rows, r0, r1)
            z = torch.where(inblk, zblk[rr.clamp_max(r1 - r0 - 1), rows.ids], z)
        return z

    @staticmethod
    def row_finalize(
        z: torch.Tensor,
        tau: torch.Tensor,
        rows: SupportRows,
        lpos: Optional[torch.Tensor],
        teacher: Optional[torch.Tensor],
        *,
        packed: bool,
        selected: bool,
        kl: bool,
    ) -> dict[str, Optional[torch.Tensor]]:
        """``lse [T]`` and the requested outputs (``packed [M]``, ``selected [T]``, ``kl [T]``)."""
        T, M = rows.num_rows, rows.num_pairs
        device = z.device
        row = rows.row
        q = div_rn(z.float(), tau[row])
        m = torch.full(
            (T,), -math.inf, dtype=torch.float32, device=device
        ).scatter_reduce(0, row, q, "amax", include_self=True)
        s = lane_sum(rows, torch.exp(q - m[row]))
        zero_t = torch.zeros(T, dtype=torch.float32, device=device)
        lse = torch.where(rows.counts > 0, m + torch.log(s), zero_t)
        multi = rows.counts >= 2
        logp = torch.where(multi[row], q - lse[row], torch.zeros_like(q))
        out: dict[str, Optional[torch.Tensor]] = dict(
            lse=lse, packed=None, selected=None, kl=None
        )
        if packed:
            out["packed"] = logp
        if selected:
            lp = lpos.to(torch.int64)
            out["selected"] = (
                torch.where(lp >= 0, logp[lp.clamp(0, M - 1)], zero_t)
                if M
                else zero_t.clone()
            )
        if kl:
            lq = teacher.float()
            Q = torch.exp(lq)
            acc = lane_fold(rows, (Q, lq - logp), _fma_acc, present=Q > 0)
            out["kl"] = torch.where(multi, acc, zero_t)
        return out

    @staticmethod
    def row_grad(
        z: torch.Tensor,
        tau: torch.Tensor,
        lse: torch.Tensor,
        rows: SupportRows,
        lpos: Optional[torch.Tensor],
        teacher: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        d: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """BF16 ``dz [M]`` from the upstream gradients of the packed (``g [M]``), selected (``d [T]``) and KL
        (``e [T]``) outputs; ``None`` = absent (skipped, never added as 0)."""
        M = rows.num_pairs
        device = z.device
        row = rows.row
        tau_p = tau[row]
        p = torch.exp(div_rn(z.float(), tau_p) - lse[row])
        u = torch.zeros(M, dtype=torch.float32, device=device)
        if g is not None:
            u = g.float().clone()
        lp = lpos.to(torch.int64) if lpos is not None else None
        if d is not None:
            is_label = lp[row] == torch.arange(M, dtype=torch.int64, device=device)
            u = torch.where(is_label, u + d.float()[row], u)
        if e is not None:
            u = fma_rn(-e.float()[row], torch.exp(teacher.float()), u)
        U = lane_sum(rows, u)
        active = rows.counts >= 2
        if g is None and e is None:
            active = active & (lp >= 0) if d is not None else torch.zeros_like(active)
        dz = div_rn(fma_rn(-p, U[row], u), tau_p).to(torch.bfloat16)
        return torch.where(active[row], dz, torch.zeros_like(dz))

    @staticmethod
    def _dz_block(
        dz: torch.Tensor, rows: SupportRows, r0: int, r1: int, vocab: int
    ) -> torch.Tensor:
        """BF16 ``[r1 - r0, V]``: the chunk's ``dz`` at its support, zeros elsewhere."""
        blk = torch.zeros(r1 - r0 + 1, vocab, dtype=torch.bfloat16, device=dz.device)
        _, rr = _blk_rows(rows, r0, r1)
        blk.index_put_((rr, rows.ids), dz)
        return blk[: r1 - r0]

    @staticmethod
    def grad_x(
        dz: torch.Tensor, rows: SupportRows, W: torch.Tensor, chunk: int
    ) -> torch.Tensor:
        T, V, H = rows.num_rows, int(W.shape[0]), int(W.shape[1])
        dX = torch.zeros(T, H, dtype=torch.bfloat16, device=W.device)
        if rows.num_pairs == 0:
            return dX
        for r0 in range(0, T, chunk):
            r1 = min(T, r0 + chunk)
            blk = SupportReferenceEngine._dz_block(dz, rows, r0, r1, V)
            dX[r0:r1] = _mm_fp32(blk, W).to(torch.bfloat16)
        return dX

    @staticmethod
    def grad_w(
        dz: torch.Tensor, rows: SupportRows, X: torch.Tensor, vocab: int, chunk: int
    ) -> torch.Tensor:
        T, H = rows.num_rows, int(X.shape[1])
        if rows.num_pairs == 0:
            return torch.zeros(vocab, H, dtype=torch.bfloat16, device=X.device)
        acc = None
        for r0 in range(0, T, chunk):
            r1 = min(T, r0 + chunk)
            blk = SupportReferenceEngine._dz_block(dz, rows, r0, r1, vocab)
            product = _mm_fp32(blk.t(), X[r0:r1])
            acc = (
                product if acc is None else acc.add_(product)
            )  # chunk order, one FP32 add per element
        return acc.to(torch.bfloat16)


ENGINES = {"reference": SupportReferenceEngine}


def _check_word(word: torch.Tensor, check_inputs: bool) -> None:
    """``ValueError`` (one readback: ``check_inputs=True`` or a host tensor) or an asynchronous device
    assertion queued ahead of every stage that indexes with the support."""
    if check_inputs or word.device.type != "cuda":
        value = int(word.item())
        if value != NO_ERROR:
            raise ValueError(describe_support_error(value))
    else:
        _assert_on_device(word[0] == NO_ERROR, _ASSERT_MESSAGE)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportOutputs:
    """Which outputs one call forms (``SupportDetLogprobFunction`` returns them in this order)."""

    packed: bool
    selected: bool
    kl: bool

    def names(self) -> tuple[str, ...]:
        return tuple(n for n in ("packed", "selected", "kl") if getattr(self, n))


# ---------------------------------------------------------------------------
# Cake backend: the generated support kernels and the shipped GEMMs
# ---------------------------------------------------------------------------

# The validation stage, one program per (ids dtype, offsets dtype).
SUPPORT_CHECK_STAGES: dict[tuple[torch.dtype, torch.dtype], str] = {
    (torch.int32, torch.int32): "support_check_ids_i32_off_i32",
    (torch.int32, torch.int64): "support_check_ids_i32_off_i64",
    (torch.int64, torch.int32): "support_check_ids_i64_off_i32",
    (torch.int64, torch.int64): "support_check_ids_i64_off_i64",
}
# Every generated support program the cake backend launches (registered in ``cake_jit`` beside the chunked LM-head
# programs, one translation unit per architecture each).
SUPPORT_STAGES = (
    *SUPPORT_CHECK_STAGES.values(),
    "support_row_finalize",
    "support_row_grad",
    "support_gather_bf16",
    "support_scatter_dz_bf16",
    "support_logits_gather_gemm",
    "support_sga_dx",
    "support_sga_dw",
    "support_sga_dw_f32_add",
    "support_sga_combine",
    "support_sga_combine_f32_add",
)
# The shipped GEMMs the dense pass form runs on row-block views ``W[v0:v1]`` (the call geometry is a launch scalar,
# so one record serves every block width).
SUPPORT_DENSE_STAGES = (
    "gemm_logits_nostats",
    "gemm_dx",
    "gemm_dw_acc",
    "gemm_dw_cast_bf16",
)
# Bits of the row finalize's ``outputs`` and of the row gradient's ``upstreams``.
OUTPUT_BITS = {"packed": 1, "selected": 2, "kl": 4}
# The three passes with a pass form: the forward logits ``F``, the input gradient ``X`` and the weight gradient ``W``.
PASSES = ("F", "X", "W")
PATHS = ("auto", "sparse", "dense")
DW_MODES = ("auto", "global", "chunked")
# Dispatch: pass ``P`` takes the dense form iff ``M >= K_P* x T`` (the mean support size reaches ``K_P*``), a host
# rule on host-known sizes (no synchronization, deterministic).  Per-architecture constants calibrated on B200
# (sm_100a) and GB300 (sm_103a): the thresholds that minimize the summed step time (forward + backward, relative to
# the best pass forms of each problem) over a calibration set of uniform, Zipf-distributed and group-shared supports
# at H = 6144, V = 154880, T = 8192 / 16384 and mean support sizes 32 to 2048, and a recorded top-p sampling mask.
# The rule cannot see how much the rows share their candidates.
SUPPORT_DENSE_MIN_MEAN_K: dict[str, dict[str, int]] = {
    "sm_100a": {"F": 512, "X": 1029, "W": 508},
    "sm_103a": {"F": 508, "X": 1029, "W": 508},
}
# Global weight-gradient mode (one pair range, BF16 dW stored directly, no FP32 [V, H] accumulator) up to this many
# pairs; above it the pairs are cut into ranges of ``chunk_size * ceil(M / T)`` pairs accumulated in FP32.
SUPPORT_DW_GLOBAL_MAX_PAIRS = 2**23
# Sparse dW: CTA key range of the segment gather-accumulate and the split of a long segment (a row many tokens share)
# into parts of at most ``split`` pairs, each with an FP32 partial row of H floats (2 ceil(n / split) rows).  The split
# is the smallest power of two >= SGA_SPLIT_MIN whose partial rows fit SGA_PARTIAL_BUDGET_BYTES.
SGA_SPAN = 128
SGA_SPLIT_MIN = 256
SGA_PARTIAL_BUDGET_BYTES = 128 * 2**20
# Measured device bytes per key of ``torch.sort`` on the int64 CSC keys, its outputs included (values and indices).
SORT_BYTES_PER_KEY = 40.25
# CSC row of a structurally zero pair (an inactive row's pairs): any row >= V sorts last and ends the walk.
SENTINEL_ROW = 2**31 - 1
# The validation stage takes the cap as int32; K_t <= M < 2^31, so a larger cap is exactly "no cap".
K1_K_CAP_MAX = 2**31 - 1
# Dense form: vocabulary block of about this many columns (a multiple of 256) and token chunks of at most
# DENSE_CHUNK_MAX rows (the shipped GEMMs' chunk bound).
DENSE_VOCAB_BLOCK_TARGET = 16384
DENSE_CHUNK_MAX = 65535


@dataclass(frozen=True)
class SupportKnobs:
    """Backend-level knobs of the cake backend (not public; tests and measurements force them): the pass form of each
    pass (``"auto"`` = :data:`SUPPORT_DENSE_MIN_MEAN_K`), the weight-gradient mode, the dense vocabulary block and the
    sparse dW split.  They enter :class:`SupportPlan` and, through its launch scalars, every remembered binding."""

    path: tuple[tuple[str, str], ...] = (("F", "auto"), ("X", "auto"), ("W", "auto"))
    dw_mode: str = "auto"
    vocab_block: Optional[int] = None
    dw_split: Optional[int] = None


def _resolve_knobs(path, dw_mode, vocab_block, dw_split) -> SupportKnobs:
    if isinstance(path, str):
        forms = {p: path for p in PASSES}
    elif isinstance(path, dict):
        unknown = sorted(set(path) - set(PASSES))
        if unknown:
            raise ValueError(f"path keys must be among {PASSES}, got {unknown}")
        forms = {p: path.get(p, "auto") for p in PASSES}
    else:
        raise ValueError(
            f"path must be one of {PATHS} or a dict over {PASSES}, got {path!r}"
        )
    for p, form in forms.items():
        if form not in PATHS:
            raise ValueError(f"path[{p!r}] must be one of {PATHS}, got {form!r}")
    if dw_mode not in DW_MODES:
        raise ValueError(f"dw_mode must be one of {DW_MODES}, got {dw_mode!r}")
    if vocab_block is not None:
        vocab_block = _positive_int(vocab_block, "vocab_block")
        if vocab_block % 256:
            raise ValueError(
                f"vocab_block must be a multiple of 256, got {vocab_block}"
            )
    if dw_split is not None:
        dw_split = _positive_int(dw_split, "dw_split")
        if dw_split > K1_K_CAP_MAX:
            raise ValueError(f"dw_split must be below 2^31, got {dw_split}")
    return SupportKnobs(
        tuple((p, forms[p]) for p in PASSES), dw_mode, vocab_block, dw_split
    )


DEFAULT_KNOBS = SupportKnobs()


@dataclass(frozen=True)
class SupportPlan:
    """Host decisions of one cake-backend call (pure functions of host-known sizes and the knobs): the pass forms, the
    weight-gradient mode and pair range, the dense vocabulary block and token chunk, the sparse dW split override."""

    arch: str
    forward: str  # "sparse" | "dense"
    dx: str
    dw: str
    dw_mode: str  # "global" | "chunked"
    dw_pairs: int  # pairs per weight-gradient range (M in global mode)
    vocab_block: int
    dense_chunk: int
    dw_split: Optional[int]


def _dense_vocab_block(vocab: int) -> int:
    blocks = -(-vocab // DENSE_VOCAB_BLOCK_TARGET)
    width = -(-vocab // blocks)
    return min(vocab, -(-width // 256) * 256)


def _dense_dx_dw_admissible(problem: SupportProblem, geometry: Geometry) -> bool:
    """The dense dX / dW form runs the shipped GEMMs with ``N = H``: H must hold whole work items of both."""
    return problem.hidden % math.lcm(geometry.dx_item_cols, geometry.dw_item_cols) == 0


@functools.lru_cache(maxsize=256)
def make_support_plan(
    problem: SupportProblem, arch: str, geometry: Geometry, knobs: SupportKnobs
) -> SupportPlan:
    """The :class:`SupportPlan` of ``problem`` on ``arch`` (remembered per argument tuple: a pure function)."""
    T, M = problem.num_rows, problem.num_pairs
    table = SUPPORT_DENSE_MIN_MEAN_K[arch]
    forms = dict(knobs.path)
    dense_ok = {"F": True, "X": _dense_dx_dw_admissible(problem, geometry)}
    dense_ok["W"] = dense_ok["X"]
    chosen: dict[str, str] = {}
    for p in PASSES:
        form = forms[p]
        if form == "auto":
            form = "dense" if dense_ok[p] and table[p] * T <= M else "sparse"
        elif form == "dense" and not dense_ok[p]:
            raise ValueError(
                f"the dense {p} pass needs H to be a multiple of "
                f"{math.lcm(geometry.dx_item_cols, geometry.dw_item_cols)} on {arch}, got H = {problem.hidden}"
            )
        chosen[p] = form
    dw_mode = knobs.dw_mode
    if dw_mode == "auto":
        dw_mode = "global" if M <= SUPPORT_DW_GLOBAL_MAX_PAIRS else "chunked"
    if dw_mode == "global":
        dw_pairs = max(M, 1)
    else:
        dw_pairs = max(1, problem.chunk * -(-M // max(T, 1)))
    vocab_block = (
        _dense_vocab_block(problem.vocab)
        if knobs.vocab_block is None
        else min(knobs.vocab_block, problem.vocab)
    )
    return SupportPlan(
        arch=arch,
        forward=chosen["F"],
        dx=chosen["X"],
        dw=chosen["W"],
        dw_mode=dw_mode,
        dw_pairs=dw_pairs,
        vocab_block=vocab_block,
        dense_chunk=max(1, min(problem.chunk, DENSE_CHUNK_MAX)),
        dw_split=knobs.dw_split,
    )


def support_split(n_keys: int, hidden: int, override: Optional[int] = None) -> int:
    """The sparse dW split of a range of ``n_keys`` keys: ``override``, else the smallest power of two
    ``>= SGA_SPLIT_MIN`` whose ``2 ceil(n_keys / split)`` FP32 partial rows of ``hidden`` floats fit
    :data:`SGA_PARTIAL_BUDGET_BYTES`.  A smaller split spreads a long segment (a row most tokens share) over more
    CTAs; the bits of ``dW`` depend on it (a fixed function of the sizes)."""
    if override is not None:
        return int(override)
    split = SGA_SPLIT_MIN
    while 8 * hidden * -(-max(n_keys, 1) // split) > SGA_PARTIAL_BUDGET_BYTES:
        split *= 2
    return split


@dataclass(frozen=True, eq=False)
class _Cake:
    """The device, the registered record of its architecture and the record's geometry."""

    device: torch.device
    index: int
    module: str
    record: dict
    geometry: Geometry
    arch: str


def _cake_context(device: torch.device) -> _Cake:
    if device.type != "cuda":
        raise ValueError(
            "backend='cake' runs on CUDA tensors of a compute capability 10.0 / 10.3 device; use backend='reference'"
        )
    module, record = record_for(device)
    record_abi(record)
    missing = [s for s in SUPPORT_STAGES + SUPPORT_DENSE_STAGES if s not in record]
    if missing:
        raise NotImplementedError(
            f"the candidate-support programs {missing} are not registered for {record['arch']} in this checkout"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _Cake(
        device,
        int(index),
        module,
        record,
        Geometry.from_record(record),
        str(record["arch"]),
    )


def support_program_available(device: Optional[torch.device] = None) -> bool:
    """True when this checkout registers every program of the cake backend for ``device``."""
    arch = arch_for(device)
    if arch is None or arch not in SUPPORT_DENSE_MIN_MEAN_K:
        return False
    try:
        record = MODULES[select_module(arch)]
    except NotImplementedError:
        return False
    return all(s in record for s in SUPPORT_STAGES + SUPPORT_DENSE_STAGES)


@functools.lru_cache(maxsize=None)
def _dummies(index: int) -> dict:
    """One-element operands of the pointers a launch never touches (an absent output / upstream / label array)."""
    device = torch.device("cuda", index)
    return {
        torch.int32: torch.full((1,), -1, dtype=torch.int32, device=device),
        torch.int64: torch.zeros((1,), dtype=torch.int64, device=device),
        torch.float32: torch.zeros((16,), dtype=torch.float32, device=device),
    }


def _or_dummy(cx: _Cake, t: Optional[torch.Tensor], dtype) -> torch.Tensor:
    return _dummies(cx.index)[dtype] if t is None else t


def _run(cx: _Cake, stage: str, values: dict) -> None:
    """Launch one stage of the record on the current stream (bound by the package's eager binding; the templated
    launch is remembered per stage, device and launch scalars)."""
    launch, arguments = _eager_launch(
        cx.record, cx.module, stage, values, cx.device, cx.geometry
    )
    with _ffi_stream_context(cx.index):
        if launch.prepare is not None:
            launch.prepare(*arguments)
        launch.entry(*arguments)


def _x_operand(X: torch.Tensor) -> torch.Tensor:
    """``X`` as the kernels read it: rows 16-byte aligned with a row pitch that is a multiple of 8 elements (the TMA
    and 16-byte-load rule); otherwise one contiguous copy (``2 T H`` bytes)."""
    T, H = (int(s) for s in X.shape)
    ld = int(X.stride(0))
    if T and (ld < H or ld % 8 or X.data_ptr() % 16):
        return torch.empty((T, H), dtype=X.dtype, device=X.device).copy_(X)
    return X


def _x_copied(X: torch.Tensor) -> bool:
    T, H = (int(s) for s in X.shape)
    ld = int(X.stride(0))
    return bool(T and (ld < H or ld % 8 or X.data_ptr() % 16))


def _rows_operand(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(tensor, ld_rows)`` of a row-major ``[R, H]`` operand of the segment gather-accumulate kernels, which take a
    contiguous tensor and the row stride: the tensor itself, or the contiguous 1-D span its rows cover."""
    R, H = (int(s) for s in t.shape)
    if t.is_contiguous():
        return t, H
    ld = int(t.stride(0))
    return t.as_strided(((R - 1) * ld + H,), (1,), t.storage_offset()), ld


def _cake_check(
    cx: _Cake,
    problem: SupportProblem,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    labels: Optional[torch.Tensor],
):
    """The validation stage: ``(word [1], lpos [T] | None, offsets32 [T + 1], the int32 ids the later stages read)``.
    The int32 copies are clamped (offsets to ``[0, M]``, int64 ids to ``[0, V - 1]``), an int32 caller array is read
    as it is (every later stage clamps it again; the sparse dW keys drop an id outside ``[0, V)``)."""
    T, M, V = problem.num_rows, problem.num_pairs, problem.vocab
    device = cx.device
    word = torch.full((1,), NO_ERROR, dtype=torch.int64, device=device)
    off32 = torch.empty((T + 1,), dtype=torch.int32, device=device)
    lpos = (
        torch.empty((T,), dtype=torch.int32, device=device)
        if labels is not None
        else None
    )
    wide_ids = ids.dtype == torch.int64
    ids32 = (
        torch.empty((M,), dtype=torch.int32, device=device) if wide_ids and M else None
    )
    _run(
        cx,
        SUPPORT_CHECK_STAGES[(ids.dtype, offsets.dtype)],
        dict(
            ids=ids if M else _dummies(cx.index)[ids.dtype],
            offsets=offsets,
            labels=_or_dummy(cx, labels, torch.int64),
            word=word,
            lpos=_or_dummy(cx, lpos, torch.int32),
            offsets32=off32,
            ids32=_or_dummy(cx, ids32, torch.int32),
            T=T,
            M=M,
            V=V,
            k_cap=min(problem.k_cap, K1_K_CAP_MAX),
            has_labels=int(labels is not None),
        ),
    )
    ids_rows = ids32 if wide_ids else ids
    return word, lpos, off32, ids_rows


def _gemm(
    cx: _Cake,
    stage: str,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    stats: torch.Tensor,
    ws: torch.Tensor,
    hidden: int,
    vocab: int,
    rows_c: int,
    first_chunk: int = 1,
) -> None:
    values = dict(
        A=A,
        B=B,
        C=C,
        STATS_OUT=stats,
        WS=ws,
        **gemm_scalars(
            stage,
            cx.geometry,
            hidden=hidden,
            vocab=vocab,
            rows_c=rows_c,
            first_chunk=first_chunk,
        ),
    )
    _run(cx, stage, values)


def _blocks(total: int, width: int) -> list[tuple[int, int]]:
    return [(v0, min(width, total - v0)) for v0 in range(0, total, width)]


def _dense_logits(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    Xop: torch.Tensor,
    W: torch.Tensor,
    ids_rows: torch.Tensor,
    off32: torch.Tensor,
    z: torch.Tensor,
) -> None:
    """Dense forward: per token chunk and vocabulary block (both ascending), the shipped logits GEMM writes the BF16
    block ``zblk`` and the gather copies the support's logits out of it (each pair's id lies in one block)."""
    T, H, V, M = problem.num_rows, problem.hidden, problem.vocab, problem.num_pairs
    C, vb = min(plan.dense_chunk, T), plan.vocab_block
    flat = torch.empty((C * vb,), dtype=torch.bfloat16, device=cx.device)
    f32 = _device_constants(cx.index)["f32_dummy"]
    for r0, rows_c in _blocks(T, C):
        A = Xop[r0 : r0 + rows_c]
        for v0, width in _blocks(V, vb):
            blk = flat[: rows_c * width].view(rows_c, width)
            _gemm(
                cx,
                "gemm_logits_nostats",
                A,
                W[v0 : v0 + width],
                blk,
                stats=f32,
                ws=f32,
                hidden=H,
                vocab=width,
                rows_c=rows_c,
            )
            _run(
                cx,
                "support_gather_bf16",
                dict(
                    zblk=blk,
                    ids=ids_rows,
                    offsets=off32,
                    z=z,
                    t0=r0,
                    rows=rows_c,
                    v0=v0,
                    vb=width,
                    ldc=width,
                    M=M,
                    V=V,
                ),
            )


def _dense_backward(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    Xop: Optional[torch.Tensor],
    W: torch.Tensor,
    ids_rows: torch.Tensor,
    off32: torch.Tensor,
    dz: torch.Tensor,
    need_dx: bool,
    need_dw: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Dense backward, chunk-outer: per token chunk and vocabulary block (ascending) the scatter builds the BF16
    gradient block ``dzblk`` (``dz`` at the support, +0 elsewhere); the shipped dX GEMM writes the block's FP32
    product, added into the chunk's FP32 ``dX_c`` in block order (one cast per chunk); the shipped weight-gradient
    GEMM stores / accumulates ``dW_acc[v0:v1]`` in chunk order and the last chunk's GEMM writes the BF16 ``dW`` from
    its epilogue (``cast(dW_acc + tile)``, or ``cast(tile)`` for one chunk)."""
    T, H, V, M = problem.num_rows, problem.hidden, problem.vocab, problem.num_pairs
    device = cx.device
    C, vb = min(plan.dense_chunk, T), plan.vocab_block
    consts = _device_constants(cx.index)
    f32 = consts["f32_dummy"]
    chunks = _blocks(T, C)
    multi = len(chunks) > 1
    flat = torch.empty((C * vb,), dtype=torch.bfloat16, device=device)
    dX = torch.empty((T if need_dx else 0, H), dtype=torch.bfloat16, device=device)
    dxc = torch.empty((C if need_dx else 0, H), dtype=torch.float32, device=device)
    tmp = torch.empty((C if need_dx else 0, H), dtype=torch.float32, device=device)
    dW = torch.empty((V if need_dw else 0, H), dtype=torch.bfloat16, device=device)
    dw_acc = torch.empty(
        (V if need_dw and multi else 0, H), dtype=torch.float32, device=device
    )
    for ci, (r0, rows_c) in enumerate(chunks):
        last = ci == len(chunks) - 1
        for bi, (v0, width) in enumerate(_blocks(V, vb)):
            blk = flat[: rows_c * width].view(rows_c, width)
            _run(
                cx,
                "support_scatter_dz_bf16",
                dict(
                    dz=dz,
                    ids=ids_rows,
                    offsets=off32,
                    dzblk=blk,
                    t0=r0,
                    rows=rows_c,
                    v0=v0,
                    vb=width,
                    ldc=width,
                    M=M,
                    V=V,
                ),
            )
            if need_dx:
                out = dxc[:rows_c] if bi == 0 else tmp[:rows_c]
                _gemm(
                    cx,
                    "gemm_dx",
                    blk,
                    W[v0 : v0 + width],
                    out,
                    stats=f32,
                    ws=f32,
                    hidden=H,
                    vocab=width,
                    rows_c=rows_c,
                )
                if bi:
                    dxc[:rows_c].add_(
                        tmp[:rows_c]
                    )  # block order, one FP32 add per element
            if need_dw:
                assert Xop is not None
                B = Xop[r0 : r0 + rows_c]
                if not last:
                    _gemm(
                        cx,
                        "gemm_dw_acc",
                        blk,
                        B,
                        dw_acc[v0 : v0 + width],
                        stats=f32,
                        ws=f32,
                        hidden=H,
                        vocab=width,
                        rows_c=rows_c,
                        first_chunk=int(ci == 0),
                    )
                else:
                    _gemm(
                        cx,
                        "gemm_dw_cast_bf16",
                        blk,
                        B,
                        dW[v0 : v0 + width],
                        stats=consts["unit_scale"],
                        ws=dw_acc[v0 : v0 + width] if multi else f32,
                        hidden=H,
                        vocab=width,
                        rows_c=rows_c,
                        first_chunk=int(not multi),
                    )
        if need_dx:
            dX[r0 : r0 + rows_c].copy_(dxc[:rows_c])  # one round to BF16
    return (dX if need_dx else None), (dW if need_dw else None)


def _row_active(
    off32: torch.Tensor, lpos: Optional[torch.Tensor], selected_only: bool
) -> torch.Tensor:
    """Rows whose ``dz`` can be nonzero: ``K_t >= 2`` and an upstream reaches the row (the selected upstream alone
    reaches only the rows with a label)."""
    active = (off32[1:] - off32[:-1]) >= 2
    if selected_only:
        assert lpos is not None
        active &= lpos >= 0
    return active


def _sga_dw_range(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    rows: torch.Tensor,
    ld_rows: int,
    ids_rows: torch.Tensor,
    off32: torch.Tensor,
    dz: torch.Tensor,
    active: torch.Tensor,
    out: torch.Tensor,
    mode: str,
    p0: int,
    p1: int,
) -> None:
    """One sparse dW pass over the pairs ``[p0, p1)``: the unique CSC keys ``(v << 32) | p`` (a pair of an inactive
    row or with an id outside ``[0, V)`` takes :data:`SENTINEL_ROW`), sorted (unique keys: one order for any correct
    sort), then the segment gather-accumulate and, when a segment can be longer than the split, the ordered sum of
    its parts."""
    T, H, V, M = problem.num_rows, problem.hidden, problem.vocab, problem.num_pairs
    n = p1 - p0
    p = torch.arange(p0, p1, dtype=torch.int32, device=cx.device)
    row = torch.searchsorted(off32[1:], p, right=True).clamp_(max=T - 1)
    keep = active[row]
    del row
    keys = ids_rows[p0:p1].to(torch.int64)
    keys.masked_fill_(~keep | (keys < 0) | (keys >= V), SENTINEL_ROW)
    del keep
    keys.bitwise_left_shift_(32).bitwise_or_(p)
    del p
    keys = torch.sort(keys).values
    split = support_split(n, H, plan.dw_split)
    partial = torch.empty(
        (2 * -(-n // split), H), dtype=torch.float32, device=cx.device
    )
    suffix = "" if mode == "bf16" else "_f32_add"
    _run(
        cx,
        "support_sga_dw" + suffix,
        dict(
            rows=rows,
            scale=dz,
            keys=keys,
            offsets=off32,
            out=out,
            partial=partial,
            ld_rows=ld_rows,
            H=H,
            V=V,
            M=M,
            T=T,
            n_keys=n,
            split=split,
            span=SGA_SPAN,
        ),
    )
    if n > split:
        _run(
            cx,
            "support_sga_combine" + suffix,
            dict(
                keys=keys,
                partial=partial,
                out=out,
                H=H,
                V=V,
                n_keys=n,
                split=split,
                span=SGA_SPAN,
            ),
        )


def _sparse_dw(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    Xop: torch.Tensor,
    ids_rows: torch.Tensor,
    off32: torch.Tensor,
    dz: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Sparse dW.  Global mode: ``dW`` zero-filled (ids in no support keep +0), then one BF16 pass over every pair.
    Chunked mode: an FP32 ``dW_acc`` zero-filled once, one FP32 add per element per pair range in range order, one
    cast at the end."""
    V, H, M = problem.vocab, problem.hidden, problem.num_pairs
    rows, ld = _rows_operand(Xop)
    if plan.dw_mode == "global":
        # +0 written as FP32 zero words (H % 256 == 0: each row holds whole words): the same bits as a BF16 zero
        # fill, at about twice its rate
        dW = torch.empty((V, H), dtype=torch.bfloat16, device=cx.device)
        dW.view(torch.float32).zero_()
        _sga_dw_range(
            cx, plan, problem, rows, ld, ids_rows, off32, dz, active, dW, "bf16", 0, M
        )
        return dW
    acc = torch.zeros((V, H), dtype=torch.float32, device=cx.device)
    for p0 in range(0, M, plan.dw_pairs):
        _sga_dw_range(
            cx,
            plan,
            problem,
            rows,
            ld,
            ids_rows,
            off32,
            dz,
            active,
            acc,
            "f32_add",
            p0,
            min(M, p0 + plan.dw_pairs),
        )
    return acc.to(torch.bfloat16)


def _cake_forward(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    outputs: SupportOutputs,
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    labels: Optional[torch.Tensor],
    teacher: Optional[torch.Tensor],
    tau: torch.Tensor,
    check_inputs: bool,
):
    """``(results, saved)``: the requested outputs and ``lse``, and the packed state the backward reads."""
    T, H, V, M = problem.num_rows, problem.hidden, problem.vocab, problem.num_pairs
    device = cx.device
    ids = support_ids.contiguous()
    offsets = support_offsets.contiguous()
    lab = labels.contiguous() if outputs.selected and labels is not None else None
    word, lpos, off32, ids_rows = _cake_check(cx, problem, ids, offsets, lab)
    _check_word(word, check_inputs)
    teacher_c = teacher.contiguous() if outputs.kl and teacher is not None else None
    if (
        M == 0
    ):  # every row is empty: lse = 0, every output 0; no launch reads the support
        zero_t = torch.zeros((T,), dtype=torch.float32, device=device)
        res = dict(
            lse=zero_t,
            packed=torch.zeros((0,), dtype=torch.float32, device=device),
            selected=zero_t.clone(),
            kl=zero_t.clone(),
        )
        z = torch.zeros((0,), dtype=torch.bfloat16, device=device)
        return res, (ids_rows, off32, lpos, teacher_c, z, zero_t)
    z = torch.empty((M,), dtype=torch.bfloat16, device=device)
    Xop = _x_operand(X)
    if plan.forward == "sparse":
        _run(
            cx,
            "support_logits_gather_gemm",
            dict(W=W, X=Xop, z=z, ids=ids_rows, offsets=off32, M=M, T=T, V=V, H=H),
        )
    else:
        _dense_logits(cx, plan, problem, Xop, W, ids_rows, off32, z)
    del Xop
    lse = torch.empty((T,), dtype=torch.float32, device=device)
    packed = (
        torch.empty((M,), dtype=torch.float32, device=device)
        if outputs.packed
        else None
    )
    selected = (
        torch.empty((T,), dtype=torch.float32, device=device)
        if outputs.selected
        else None
    )
    kl = torch.empty((T,), dtype=torch.float32, device=device) if outputs.kl else None
    bits = sum(OUTPUT_BITS[n] for n in outputs.names())
    _run(
        cx,
        "support_row_finalize",
        dict(
            z=z,
            tau=tau,
            offsets=off32,
            lpos=_or_dummy(cx, lpos, torch.int32),
            teacher=_or_dummy(cx, teacher_c, torch.float32),
            lse=lse,
            packed=_or_dummy(cx, packed, torch.float32),
            selected=_or_dummy(cx, selected, torch.float32),
            kl=_or_dummy(cx, kl, torch.float32),
            T=T,
            M=M,
            outputs=bits,
        ),
    )
    res = dict(lse=lse, packed=packed, selected=selected, kl=kl)
    return res, (ids_rows, off32, lpos, teacher_c, z, lse)


def _upstream(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.to(torch.float32).contiguous()


def _cake_backward(
    cx: _Cake,
    plan: SupportPlan,
    problem: SupportProblem,
    X: torch.Tensor,
    W: torch.Tensor,
    saved,
    tau: torch.Tensor,
    g: Optional[torch.Tensor],
    d: Optional[torch.Tensor],
    e: Optional[torch.Tensor],
    need_dx: bool,
    need_dw: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    T, H, V, M = problem.num_rows, problem.hidden, problem.vocab, problem.num_pairs
    device = cx.device
    ids_rows, off32, lpos, teacher, z, lse = saved
    dX: Optional[torch.Tensor] = None
    dW: Optional[torch.Tensor] = None
    if M == 0:  # no pair: zero gradients, no launch
        if need_dx:
            dX = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
        if need_dw:
            dW = torch.zeros((V, H), dtype=torch.bfloat16, device=device)
        return dX, dW
    g, d, e = _upstream(g), _upstream(d), _upstream(e)
    dz = torch.empty((M,), dtype=torch.bfloat16, device=device)
    bits = (
        (OUTPUT_BITS["packed"] if g is not None else 0)
        | (OUTPUT_BITS["selected"] if d is not None else 0)
        | (OUTPUT_BITS["kl"] if e is not None else 0)
    )
    _run(
        cx,
        "support_row_grad",
        dict(
            z=z,
            tau=tau,
            lse=lse,
            offsets=off32,
            lpos=_or_dummy(cx, lpos if d is not None else None, torch.int32),
            teacher=_or_dummy(cx, teacher if e is not None else None, torch.float32),
            g=_or_dummy(cx, g, torch.float32),
            d=_or_dummy(cx, d, torch.float32),
            e=_or_dummy(cx, e, torch.float32),
            dz=dz,
            T=T,
            M=M,
            upstreams=bits,
        ),
    )
    del g, d, e
    dense_x = need_dx and plan.dx == "dense"
    dense_w = need_dw and plan.dw == "dense"
    Xop = _x_operand(X) if need_dw else None
    if dense_x or dense_w:
        dX, dW = _dense_backward(
            cx, plan, problem, Xop, W, ids_rows, off32, dz, dense_x, dense_w
        )
    if need_dx and not dense_x:
        dX = torch.empty((T, H), dtype=torch.bfloat16, device=device)
        _run(
            cx,
            "support_sga_dx",
            dict(
                rows=W,
                scale=dz,
                ids=ids_rows,
                offsets=off32,
                out=dX,
                ld_rows=H,
                H=H,
                V=V,
                M=M,
                T=T,
                min_pairs=2,
            ),
        )
    if need_dw and not dense_w:
        assert Xop is not None
        active = _row_active(off32, lpos, bits == OUTPUT_BITS["selected"])
        dW = _sparse_dw(cx, plan, problem, Xop, ids_rows, off32, dz, active)
    return dX, dW


def support_memory_report(
    T: int,
    H: int,
    V: int,
    M: int,
    *,
    outputs: tuple[str, ...] = ("packed",),
    ids_dtype: torch.dtype = torch.int64,
    chunk_size: Optional[int] = None,
    arch: str = "sm_100a",
    path: Union[str, dict] = "auto",
    dw_mode: str = "auto",
    vocab_block: Optional[int] = None,
    dw_split: Optional[int] = None,
    need_dx: bool = True,
    need_dw: bool = True,
    x_copy: bool = False,
) -> dict:
    """Device bytes of one cake-backend training step (forward + backward) by bucket, from host-known sizes and the
    plan the call would take on ``arch`` (an upper bound per bucket: every temporary of a phase counted at once).

    * ``packed_state``: the saved ``O(M)`` / ``O(T)`` state (``z`` 2 B and, for int64 ids, the int32 ids 4 B per pair;
      int32 offsets, ``lse``, ``tau`` and, with labels, ``lpos`` 4 B per row) -- counted with the packed outputs.
    * ``outputs``: the requested outputs (``packed`` 4 B per pair, ``selected`` / ``kl`` 4 B per row).
    * ``forward_workspace``: the dense logits block ``[C, vb]`` BF16 (dense forward only), plus ``X``'s copy when its
      rows are not 16-byte aligned (``x_copy``).
    * ``backward_workspace``: ``dz`` 2 B per pair; dense dX / dW: the gradient block ``[C, vb]`` BF16 and, for dX, two
      FP32 ``[C, H]`` blocks; sparse dW: per pair range of ``n`` pairs the pair index (4 B), the row index (8 B), the
      keys (8 B), :data:`SORT_BYTES_PER_KEY` measured ``torch.sort`` bytes per key, the row activity (1 B per row) and
      ``2 ceil(n / split)`` FP32 partial rows; ``X``'s copy (``x_copy``).
    * ``weight_gradient_storage``: the FP32 ``dW_acc [V, H]`` (sparse chunked mode, or dense with more than one token
      chunk).
    * ``gradients``: BF16 ``dX`` and ``dW``.
    * ``plan``: the pass forms, the dW mode, the pair range, the split, the vocabulary block and the dense chunk.
    """
    if arch not in SUPPORT_DENSE_MIN_MEAN_K:
        raise ValueError(f"arch must be one of {sorted(SUPPORT_DENSE_MIN_MEAN_K)}")
    names = tuple(outputs)
    if not names or any(n not in OUTPUT_BITS for n in names):
        raise ValueError(f"outputs must name some of {tuple(OUTPUT_BITS)}")
    geometry = Geometry.from_record(MODULES[select_module(arch)])
    chunk = (
        DEFAULT_CHUNK_SIZE
        if chunk_size is None
        else _positive_int(chunk_size, "chunk_size")
    )
    problem = SupportProblem(
        num_rows=int(T),
        hidden=int(H),
        vocab=int(V),
        num_pairs=int(M),
        k_cap=max(1, int(M)),
        chunk=chunk,
        has_labels="selected" in names,
        has_teacher="kl" in names,
    )
    plan = make_support_plan(
        problem, arch, geometry, _resolve_knobs(path, dw_mode, vocab_block, dw_split)
    )
    T, H, V, M = int(T), int(H), int(V), int(M)
    C = min(plan.dense_chunk, max(T, 1))
    vb = plan.vocab_block
    x_bytes = 2 * T * H if x_copy else 0
    packed_state = (
        2 * M
        + (4 * M if ids_dtype == torch.int64 else 0)
        + 4 * (T + 1)
        + 8 * T
        + (4 * T if "selected" in names else 0)
    )
    out_bytes = (
        (4 * M if "packed" in names else 0)
        + (4 * T if "selected" in names else 0)
        + (4 * T if "kl" in names else 0)
    )
    forward_ws = (2 * C * vb if plan.forward == "dense" and M else 0) + x_bytes
    backward_ws = 2 * M if M and (need_dx or need_dw) else 0
    dense_x = need_dx and plan.dx == "dense" and M
    dense_w = need_dw and plan.dw == "dense" and M
    if dense_x or dense_w:
        backward_ws += 2 * C * vb + (8 * C * H if dense_x else 0)
    split: Optional[int] = None
    if need_dw and M and not dense_w:
        n = min(plan.dw_pairs, M)
        split = support_split(n, H, plan.dw_split)
        backward_ws += (
            math.ceil((4 + 8 + 8 + SORT_BYTES_PER_KEY) * n) + T + 8 * H * -(-n // split)
        )
    if need_dw and M:
        backward_ws += x_bytes
    multi_chunk = T > C
    dw_storage = (
        4 * V * H
        if need_dw
        and M
        and ((not dense_w and plan.dw_mode == "chunked") or (dense_w and multi_chunk))
        else 0
    )
    grads = (2 * T * H if need_dx else 0) + (2 * V * H if need_dw else 0)
    report = dict(
        packed_state=packed_state,
        outputs=out_bytes,
        forward_workspace=forward_ws,
        backward_workspace=backward_ws,
        weight_gradient_storage=dw_storage,
        gradients=grads,
        sort_bytes=math.ceil(SORT_BYTES_PER_KEY * min(plan.dw_pairs, M))
        if need_dw and M and not dense_w
        else 0,
        plan=dict(
            forward=plan.forward,
            dx=plan.dx,
            dw=plan.dw,
            dw_mode=plan.dw_mode,
            dw_pairs=plan.dw_pairs,
            dw_split=split,
            vocab_block=vb,
            dense_chunk=plan.dense_chunk,
        ),
    )
    report["total"] = (
        packed_state + out_bytes + forward_ws + backward_ws + dw_storage + grads
    )
    return report


class SupportDetLogprobFunction(torch.autograd.Function):
    """One forward serves every requested output; gradients to ``X`` and ``W`` only."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        teacher,
        tau,
        problem: SupportProblem,
        outputs: SupportOutputs,
        check_inputs: bool,
        backend: str,
        cake,
    ):
        ctx.set_materialize_grads(False)
        ctx.problem, ctx.outputs, ctx.backend, ctx.cake = (
            problem,
            outputs,
            backend,
            cake,
        )
        T, M = problem.num_rows, problem.num_pairs
        device = X.device
        if T == 0:
            ctx.save_for_backward(X)
            empty = dict(
                packed=torch.zeros(M, dtype=torch.float32, device=device),
                selected=torch.zeros(0, dtype=torch.float32, device=device),
                kl=torch.zeros(0, dtype=torch.float32, device=device),
            )
            return tuple(empty[n] for n in outputs.names())
        if backend == "cake":
            cx, plan = cake
            res, saved = _cake_forward(
                cx,
                plan,
                problem,
                outputs,
                X,
                W,
                support_ids,
                support_offsets,
                labels,
                teacher,
                tau,
                check_inputs,
            )
            ctx.save_for_backward(X, W, *saved, tau)
            return tuple(res[n] for n in outputs.names())
        engine = ENGINES[backend]
        chk = engine.support_check(
            support_ids,
            support_offsets,
            labels if outputs.selected else None,
            num_rows=T,
            vocab=problem.vocab,
            k_cap=problem.k_cap,
        )
        _check_word(chk.word, check_inputs)
        rows = SupportRows.build(
            support_ids,
            support_offsets,
            num_rows=T,
            vocab=problem.vocab,
            k_cap=problem.k_cap,
        )
        z = engine.support_logits(X, W, rows, problem.chunk)
        res = engine.row_finalize(
            z,
            tau,
            rows,
            chk.lpos,
            teacher,
            packed=outputs.packed,
            selected=outputs.selected,
            kl=outputs.kl,
        )
        ctx.save_for_backward(
            X, W, rows.ids, rows.offsets, chk.lpos, teacher, z, res["lse"], tau
        )
        return tuple(res[n] for n in outputs.names())

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        problem, outputs = ctx.problem, ctx.outputs
        up = dict(zip(outputs.names(), grads, strict=True))
        g, d, e = up.get("packed"), up.get("selected"), up.get("kl")
        rest = (None,) * 10
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        if (g is None and d is None and e is None) or not (need_dx or need_dw):
            return (None, None) + rest
        T, H, V = problem.num_rows, problem.hidden, problem.vocab
        if T == 0:
            (X,) = ctx.saved_tensors
            dX = (
                torch.zeros(0, H, dtype=torch.bfloat16, device=X.device)
                if need_dx
                else None
            )
            dW = (
                torch.zeros(V, H, dtype=torch.bfloat16, device=X.device)
                if need_dw
                else None
            )
            return (dX, dW) + rest
        X, W, ids, offsets, lpos, teacher, z, lse, tau = ctx.saved_tensors
        if ctx.backend == "cake":
            cx, plan = ctx.cake
            dX, dW = _cake_backward(
                cx,
                plan,
                problem,
                X,
                W,
                (ids, offsets, lpos, teacher, z, lse),
                tau,
                g,
                d,
                e,
                need_dx,
                need_dw,
            )
            return (dX, dW) + rest
        engine = ENGINES[ctx.backend]
        rows = SupportRows.build(ids, offsets, num_rows=T, vocab=V, k_cap=problem.k_cap)
        dz = engine.row_grad(z, tau, lse, rows, lpos, teacher, g, d, e)
        dX = engine.grad_x(dz, rows, W, problem.chunk) if need_dx else None
        dW = engine.grad_w(dz, rows, X, V, problem.chunk) if need_dw else None
        return (dX, dW) + rest


def _support_call(
    X,
    W,
    support_ids,
    support_offsets,
    labels,
    teacher,
    *,
    k_cap,
    temperature,
    chunk_size,
    check_inputs,
    backend,
    outputs: SupportOutputs,
    path="auto",
    dw_mode="auto",
    vocab_block=None,
    dw_split=None,
) -> dict[str, torch.Tensor]:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if not isinstance(check_inputs, bool):
        raise ValueError("check_inputs must be a bool")
    problem = validate_support_inputs(
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        teacher,
        k_cap=k_cap,
        chunk_size=chunk_size,
    )
    resolved = _resolve_temperature(
        temperature, problem.num_rows, X.device, check=check_inputs
    )
    knobs = _resolve_knobs(path, dw_mode, vocab_block, dw_split)
    tau = _support_tau_rows(resolved, problem.num_rows, X.device)
    if problem.num_rows == 0:
        # no row, so no validation stage: the one offsets rule left (offsets[0] == 0) is checked here
        _check_first_offset(support_offsets, check_inputs)
    cake = None
    if backend == "cake":
        cx = _cake_context(X.device)
        cake = (cx, make_support_plan(problem, cx.arch, cx.geometry, knobs))
    result = SupportDetLogprobFunction.apply(
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        teacher,
        tau,
        problem,
        outputs,
        check_inputs,
        backend,
        cake,
    )
    return dict(zip(outputs.names(), result, strict=True))


def _check_first_offset(offsets: torch.Tensor, check_inputs: bool) -> None:
    """``offsets[0] == 0`` of a call without rows: one readback and ``ValueError`` (``check_inputs=True`` or a host
    tensor), else an asynchronous device assertion."""
    first = offsets[0]
    if check_inputs or first.device.type != "cuda":
        if int(first) != 0:
            raise ValueError(
                describe_support_error(support_error_word(ERR_OFFSETS_FIRST, 0))
            )
    else:
        _assert_on_device(first == 0, _ASSERT_MESSAGE)


def chunked_lm_head_det_support_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    backend: str = "cake",
    path: Union[str, dict] = "auto",
    dw_mode: str = "auto",
    vocab_block: Optional[int] = None,
    dw_split: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable FP32 log-probabilities of ``z = X @ W^T`` normalized over each row's candidates.

    ``labels is None``: the packed ``logp [M]`` aligned with ``support_ids`` (caller order).  Otherwise the
    selected ``logp [T]`` of the int64 ``labels`` (``-100``: 0 with zero gradient; any other label must be
    one of the row's candidates).  Rows with ``K_t <= 1`` give 0.  ``k_cap``: the caller's candidate cap
    (``K_t > k_cap`` is an error, never trimmed).  ``temperature``: a finite positive number or an FP32
    ``[T]`` tensor (default 1).  ``chunk_size``: the token chunk (default 4096); the weight gradient's
    rounding depends on it.  ``check_inputs``: read the validation word back and raise ``ValueError``
    (default: an asynchronous device assertion, no synchronization).  Under ``torch.no_grad()`` the call
    only runs the forward (teacher-target preparation).  See the module docstring for the contract.

    Backend-level knobs of ``backend="cake"`` (not part of the public API; tests and measurements force them):
    ``path`` (``"auto"`` / ``"sparse"`` / ``"dense"``, or a dict over the passes ``"F"`` / ``"X"`` / ``"W"``),
    ``dw_mode`` (``"auto"`` / ``"global"`` / ``"chunked"``), ``vocab_block`` (the dense form's vocabulary block, a
    multiple of 256) and ``dw_split`` (the sparse weight gradient's segment split).  The bits of ``dX`` / ``dW``
    depend on the pass form, ``chunk_size``, ``vocab_block`` and ``dw_split`` (fixed functions of the sizes when
    left at their defaults); reruns with the same arguments are bitwise equal.
    """
    outputs = SupportOutputs(
        packed=labels is None, selected=labels is not None, kl=False
    )
    res = _support_call(
        X,
        W,
        support_ids,
        support_offsets,
        labels,
        None,
        k_cap=k_cap,
        temperature=temperature,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        backend=backend,
        outputs=outputs,
        path=path,
        dw_mode=dw_mode,
        vocab_block=vocab_block,
        dw_split=dw_split,
    )
    return res["packed"] if labels is None else res["selected"]


def chunked_lm_head_det_support_kl(
    X: torch.Tensor,
    W: torch.Tensor,
    support_ids: torch.Tensor,
    support_offsets: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    k_cap: int,
    temperature: Union[None, float, torch.Tensor] = 1.0,
    return_logprobs: bool = False,
    chunk_size: Optional[int] = None,
    check_inputs: bool = False,
    backend: str = "cake",
    path: Union[str, dict] = "auto",
    dw_mode: str = "auto",
    vocab_block: Optional[int] = None,
    dw_split: Optional[int] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Unreduced FP32 forward KL ``[T]``, ``KL_t = sum_j Q_j (lq_j - logp_j)`` over the row's candidates with
    ``Q_j = exp(lq_j) > 0``, for the detached FP32 teacher log-probabilities ``teacher_logprobs [M]``
    (``lq``, aligned with ``support_ids``; ``-inf`` allowed: ``Q = 0``).  Student and teacher share the
    support.  Rows with ``K_t <= 1`` give 0.  ``return_logprobs=True`` also returns the differentiable
    packed student ``logp [M]``: ``(kl, logp)``.  Weighting, reduction and any ``tau ** 2`` factor belong to
    the caller.  Other arguments (and the backend-level knobs) as in :func:`chunked_lm_head_det_support_logprob`.
    """
    outputs = SupportOutputs(packed=bool(return_logprobs), selected=False, kl=True)
    res = _support_call(
        X,
        W,
        support_ids,
        support_offsets,
        None,
        teacher_logprobs,
        k_cap=k_cap,
        temperature=temperature,
        chunk_size=chunk_size,
        check_inputs=check_inputs,
        backend=backend,
        outputs=outputs,
        path=path,
        dw_mode=dw_mode,
        vocab_block=vocab_block,
        dw_split=dw_split,
    )
    return (res["kl"], res["packed"]) if return_logprobs else res["kl"]


# The names before the deterministic (``det``) naming, kept as aliases: each is
# the canonical object above.
SupportLogprobFunction = SupportDetLogprobFunction
chunked_lm_head_support_logprob = chunked_lm_head_det_support_logprob
chunked_lm_head_support_kl = chunked_lm_head_det_support_kl

__all__ = [
    "CANONICAL_SEQUENCE",
    "NO_ERROR",
    "SUPPORT_ERRORS",
    "SUPPORT_M_MAX",
    "SUPPORT_V_MAX",
    "SupportDetLogprobFunction",
    "SupportLogprobFunction",
    "SupportReferenceEngine",
    "SupportRows",
    "chunked_lm_head_det_support_kl",
    "chunked_lm_head_det_support_logprob",
    "chunked_lm_head_support_kl",
    "chunked_lm_head_support_logprob",
    "decode_support_error",
    "describe_support_error",
    "support_error_word",
    "support_memory_report",
    "support_program_available",
    "validate_support_inputs",
]
