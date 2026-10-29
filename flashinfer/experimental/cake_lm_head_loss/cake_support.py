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
  unique inside a row (any order; ids repeat freely across rows).  ``M < 2^31``.
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
allocation-free).  ``backend="cake"`` is the generated-kernel path (not built in this revision).
"""

from __future__ import annotations

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
    _mm_fp32,
    _resolve_temperature,
    _temperature_rows,
)

BACKENDS = ("cake", "reference")
SUPPORT_INDEX_DTYPES = (torch.int32, torch.int64)
# Largest vocabulary of the validation stage's per-warp shared-memory bitmap (V bits = 192 KiB at the limit).
SUPPORT_V_MAX = 1536 * 1024
# ``M < 2^31``: packed pair indices (label positions, CSC keys) are int32 in the kernels.
SUPPORT_M_MAX = 2**31 - 1
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
        raise ValueError(f"support_ids must have fewer than 2^31 entries, got {M}")
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
    """Which outputs one call forms (``SupportLogprobFunction`` returns them in this order)."""

    packed: bool
    selected: bool
    kl: bool

    def names(self) -> tuple[str, ...]:
        return tuple(n for n in ("packed", "selected", "kl") if getattr(self, n))


class SupportLogprobFunction(torch.autograd.Function):
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
    ):
        ctx.set_materialize_grads(False)
        ctx.problem, ctx.outputs, ctx.backend = problem, outputs, backend
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
        rest = (None,) * 9
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
    tau = _support_tau_rows(resolved, problem.num_rows, X.device)
    if backend == "cake":
        raise NotImplementedError(
            "the cake backend of the candidate-support entry points is not built in this revision; "
            "use backend='reference'"
        )
    result = SupportLogprobFunction.apply(
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
    )
    return dict(zip(outputs.names(), result, strict=True))


def chunked_lm_head_support_logprob(
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
    )
    return res["packed"] if labels is None else res["selected"]


def chunked_lm_head_support_kl(
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
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Unreduced FP32 forward KL ``[T]``, ``KL_t = sum_j Q_j (lq_j - logp_j)`` over the row's candidates with
    ``Q_j = exp(lq_j) > 0``, for the detached FP32 teacher log-probabilities ``teacher_logprobs [M]``
    (``lq``, aligned with ``support_ids``; ``-inf`` allowed: ``Q = 0``).  Student and teacher share the
    support.  Rows with ``K_t <= 1`` give 0.  ``return_logprobs=True`` also returns the differentiable
    packed student ``logp [M]``: ``(kl, logp)``.  Weighting, reduction and any ``tau ** 2`` factor belong to
    the caller.  Other arguments as in :func:`chunked_lm_head_support_logprob`.
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
    )
    return (res["kl"], res["packed"]) if return_logprobs else res["kl"]


__all__ = [
    "CANONICAL_SEQUENCE",
    "NO_ERROR",
    "SUPPORT_ERRORS",
    "SUPPORT_V_MAX",
    "SupportLogprobFunction",
    "SupportReferenceEngine",
    "SupportRows",
    "chunked_lm_head_support_kl",
    "chunked_lm_head_support_logprob",
    "decode_support_error",
    "describe_support_error",
    "support_error_word",
    "validate_support_inputs",
]
