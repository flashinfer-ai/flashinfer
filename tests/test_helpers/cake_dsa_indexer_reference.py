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

Independent reference for the DSA indexer top-k contract
(flashinfer-ai/flashinfer#5676): input generators, an FP64 / FP32 scorer, the
exact global selection with the issue's tie rule, the accumulation-error
bound and the judgement of a result against the reference.  Shared by
``tests/experimental/test_cake_dsa_indexer.py`` and
``benchmarks/bench_cake_dsa_indexer.py``.

Nothing here knows how the kernels partition queries or keys.

Scoring.  For every visible (query, key) pair the FP64 score is treated as
exact and an independent FP32 scorer that follows the generated program's
documented reduction model is evaluated beside it.  With
``order="sequential"`` the dot product ``x_h`` is accumulated with elementwise
operations ``(((q0 k0 + q1 k1) + q2 k2) + ...)``: the products of two BF16
values are exact in FP32 and FP64, so each addition is the only rounding and
the bits do not depend on how the operands were chunked.  The head reduction
is the program's: ``r_h = fl(x_h + |x_h|)`` (exactly ``2 relu(x_h)``; ``+0.0``
for every ``x_h <= 0`` including ``-0.0``), ``w'_h = fl(w_h * softmax_scale)``,
four interleaved FMA chains ``c_m = fma(r_h, w'_h, c_m)`` over ``h = m, m + 4,
..., m + 28`` starting from ``+0.0`` (the FMA is emulated as the exact FP64
product-sum rounded once to FP32), then ``s = fl(fl(fl(c_0 + c_2) + fl(c_1 +
c_3)) * 0.5)``.  Because every chain starts from ``+0.0`` and
``fma(+0, w', +0) = +0`` for either sign of ``w'``, a row whose head terms are
all zero scores ``+0.0``: ``-0.0`` cannot arise from this reduction (the
program's ``zero_sign_policy = "positive_accumulator"``).  ``order="matmul"``
uses ``torch.matmul`` (TF32 disabled for FP32) for the dot products: much
faster, but its bits depend on the operand shapes, so it is used only for
bound-based comparisons; ``order="auto"`` picks ``sequential`` up to
``SEQUENTIAL_PAIR_LIMIT`` visible pairs.

Accumulation bound.  Two independent FP32 evaluations of one score differ by
at most ``gamma_n * A`` with ``n = 2D + 4H + 4``, ``u = 2**-24``, ``gamma_n = n u
/ (1 - n u)`` and ``A = softmax_scale * sum_h |w_h| sum_d |q_hd k_jd|``
(Higham: a sum or dot product of ``m`` terms evaluated in any order has
forward error at most ``gamma_m(u) * sum |terms|``; one evaluation spends at
most ``D`` roundings on the dot product, one on the scale, none on the
monotone 1-Lipschitz relu, ``H`` on the head products and ``H`` on the head
sum, plus one spare: ``D + 2H + 2``; two evaluations give ``2D + 4H + 4``.
A tensor-core accumulation that truncates instead of rounding has unit
roundoff ``2u``, and ``gamma_m(2u) <= gamma_2m(u)``, so the same ``n`` covers
the kernel-versus-FP64 comparison).

Judgement.  Structure (shapes, dtypes, padding, uniqueness, ascending order,
visibility and segment membership) is always exact.  Returned score bits must
lie within the bound of both the FP64 score and the FP32 scorer
(``exact_bits=True``: bitwise equal to the FP32 scorer, modulo the documented
zero-sign policy).  Selected id sets must agree with the reference wherever
the score separation exceeds the numerical uncertainty: in every differing
row each key the result lacks must have an error interval ``[s - b, s + b]``
that overlaps the interval of every key the result holds instead; a lacking
key whose lower bound exceeds the upper bound of a held key is a clearly
better key and fails the row.  ``exact_bits=True`` makes any id difference a
failure.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

NUM_HEADS = 32
HEAD_DIM = 128
DEFAULT_TOP_K = 2048
DEFAULT_SCALE = HEAD_DIM**-0.5
EXACT_SCALE = 2.0**-3
TRAINER_KEY_ROW_STRIDE = 704
PAD_ID = -1
PAD_SCORE = float("-inf")
FP32_UNIT_ROUNDOFF = 2.0**-24
SEQUENTIAL_PAIR_LIMIT = 1 << 25
CHUNK_BYTES = 1 << 30
INT64_MAX = (1 << 63) - 1
ZERO_SIGN_POLICIES = ("positive_accumulator", "ieee_sum")
_GEN_ROWS = 1 << 17


# ---------------------------------------------------------------------------
# Bound
# ---------------------------------------------------------------------------


def accumulation_terms(num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM) -> int:
    """``n = 2D + 4H + 4`` (module docstring)."""
    return 2 * head_dim + 4 * num_heads + 4


def gamma_bound(n: int, unit_roundoff: float = FP32_UNIT_ROUNDOFF) -> float:
    """Higham's ``gamma_n = n u / (1 - n u)``."""
    nu = n * unit_roundoff
    if nu >= 1.0:
        raise ValueError(f"gamma_n undefined for n u >= 1 (n = {n})")
    return nu / (1.0 - nu)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def cumulative(lengths) -> list[int]:
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


def effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, ratio: int) -> list[int]:
    """Per-segment offset: the supplied value, else ``Lk - Lq`` for ``ratio == 1``, else ``0``."""
    if q_causal_offsets is not None:
        if len(q_causal_offsets) != len(seg_q_len):
            raise ValueError("one offset per segment is required")
        return [int(o) for o in q_causal_offsets]
    if int(ratio) == 1:
        return [int(lk) - int(lq) for lq, lk in zip(seg_q_len, seg_k_len, strict=True)]
    return [0 for _ in seg_q_len]


def visible_scalar(offset: int, position: int, ratio: int, num_keys: int) -> int:
    """Visible keys of one query by the issue's rule (Python ``//`` is a true floor)."""
    return max(0, min(int(num_keys), (int(offset) + int(position) + 1) // int(ratio)))


@dataclass
class IndexerInputs:
    """Operator inputs plus the host segment metadata they were built from."""

    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    top_k: int
    softmax_scale: float
    seg_q_len: list[int]
    seg_k_len: list[int]
    q_causal_offsets: Optional[torch.Tensor] = None
    ratio: int = 1
    label: str = ""

    @property
    def num_queries(self) -> int:
        return int(self.q.shape[0])

    @property
    def num_keys(self) -> int:
        return int(self.k.shape[0])

    @property
    def num_segments(self) -> int:
        return len(self.seg_q_len)

    @property
    def device(self) -> torch.device:
        return self.q.device

    def offsets(self) -> list[int]:
        explicit = (
            None if self.q_causal_offsets is None else self.q_causal_offsets.tolist()
        )
        return effective_offsets(self.seg_q_len, self.seg_k_len, explicit, self.ratio)

    def kwargs(self) -> dict[str, Any]:
        return dict(
            top_k=self.top_k,
            softmax_scale=self.softmax_scale,
            q_causal_offsets=self.q_causal_offsets,
            ratio=self.ratio,
        )

    def query_starts(self) -> list[int]:
        return cumulative(self.seg_q_len)

    def key_starts(self) -> list[int]:
        return cumulative(self.seg_k_len)


def visible_rows(inputs: IndexerInputs, device=None) -> torch.Tensor:
    """int64 ``[T]`` visible-key count per query row from the host metadata only."""
    device = inputs.device if device is None else device
    out = torch.zeros(inputs.num_queries, dtype=torch.int64, device=device)
    starts = inputs.query_starts()
    for s, off in enumerate(inputs.offsets()):
        lq, lk = inputs.seg_q_len[s], inputs.seg_k_len[s]
        if lq == 0:
            continue
        u = torch.arange(lq, dtype=torch.int64, device=device)
        vis = torch.div(u + (off + 1), inputs.ratio, rounding_mode="floor").clamp_(
            min=0, max=lk
        )
        out[starts[s] : starts[s] + lq] = vis
    return out


def segment_of_rows(inputs: IndexerInputs, device=None) -> torch.Tensor:
    """int64 ``[T]`` segment index of every query row."""
    device = inputs.device if device is None else device
    out = torch.empty(inputs.num_queries, dtype=torch.int64, device=device)
    starts = inputs.query_starts()
    for s, lq in enumerate(inputs.seg_q_len):
        if lq:
            out[starts[s] : starts[s] + lq] = s
    return out


def build_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    seg_q_len,
    seg_k_len,
    *,
    top_k: int,
    softmax_scale: float,
    q_causal_offsets=None,
    ratio: int = 1,
    label: str = "",
) -> IndexerInputs:
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    if len(seg_q_len) != len(seg_k_len):
        raise ValueError("query and key segment lists must have the same length")
    if sum(seg_q_len) != q.shape[0] or sum(seg_k_len) != k.shape[0]:
        raise ValueError("segment lengths must sum to the tensor row counts")
    device = q.device
    offsets = None
    if q_causal_offsets is not None:
        offsets = torch.tensor(
            [int(o) for o in q_causal_offsets], dtype=torch.int64, device=device
        )
    return IndexerInputs(
        q=q,
        k=k,
        w=w,
        cu_seqlens_q=torch.tensor(
            cumulative(seg_q_len), dtype=torch.int32, device=device
        ),
        cu_seqlens_k=torch.tensor(
            cumulative(seg_k_len), dtype=torch.int32, device=device
        ),
        top_k=int(top_k),
        softmax_scale=float(softmax_scale),
        seg_q_len=seg_q_len,
        seg_k_len=seg_k_len,
        q_causal_offsets=offsets,
        ratio=int(ratio),
        label=label,
    )


def _gaussian_bf16(shape, gen, device) -> torch.Tensor:
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    for r0 in range(0, shape[0], _GEN_ROWS):
        r1 = min(shape[0], r0 + _GEN_ROWS)
        out[r0:r1].copy_(
            torch.randn(
                (r1 - r0, *shape[1:]), generator=gen, device=device, dtype=torch.float32
            )
        )
    return out


def make_random_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
    device="cuda",
    peaked: bool = False,
    ratio: int = 1,
    q_causal_offsets=None,
    softmax_scale: Optional[float] = None,
    k_row_stride: Optional[int] = None,
    hot_keys_per_segment: int = 16,
    label: str = "",
) -> IndexerInputs:
    """Seeded iid Gaussian BF16 ``q`` / ``k`` (unit RMS) and signed FP32 ``w ~ N(0, 1) * H**-0.5``.

    ``peaked=True`` adds a shared +-1 direction to every query and twice that
    direction to a few keys per segment, so those keys dominate most rows
    (heavy-tailed scores).  ``k_row_stride`` returns ``k`` as the leading
    ``[:, :128]`` view of a wider packed ``[Tkv, k_row_stride]`` tensor.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    T, Tkv = sum(seg_q_len), sum(seg_k_len)
    q = _gaussian_bf16((T, NUM_HEADS, HEAD_DIM), gen, device)
    w = torch.randn(
        (T, NUM_HEADS), generator=gen, device=device, dtype=torch.float32
    ) * (NUM_HEADS**-0.5)
    if k_row_stride is None:
        k = _gaussian_bf16((Tkv, HEAD_DIM), gen, device)
    else:
        if k_row_stride < HEAD_DIM:
            raise ValueError("k_row_stride must be >= 128")
        k = _gaussian_bf16((Tkv, int(k_row_stride)), gen, device)[:, :HEAD_DIM]
    if peaked:
        direction = (
            torch.randint(0, 2, (HEAD_DIM,), generator=gen, device=device) * 2 - 1
        ).to(torch.bfloat16)
        q.add_(direction.view(1, 1, HEAD_DIM))
        k_start = 0
        for lk in seg_k_len:
            n_hot = min(int(hot_keys_per_segment), lk)
            if n_hot:
                hot = torch.randperm(lk, generator=gen, device=device)[:n_hot] + k_start
                k[hot] = (
                    k[hot].float() + 2.0 * direction.float().view(1, HEAD_DIM)
                ).to(torch.bfloat16)
            k_start += lk
    return build_inputs(
        q,
        k,
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=DEFAULT_SCALE if softmax_scale is None else softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        label=label,
    )


# -- exactly representable cases ----------------------------------------------------------------------------------
# Key ``j`` carries two integer levels ``up_j`` and ``down_j`` in ``[0, 255]``: ``up`` is spread over dims 0..7 and
# ``down`` over dims 64..71 as small integers (<= 32, exact in BF16).  Query head 0 holds the amplitude ``a`` on dims
# 0..7 and head 1 holds ``a`` on dims 64..71 (``a`` in {1, 2, 4}); every other head is zero.  With ``softmax_scale =
# 1/8`` the head-0 logit is ``a * up_j`` and the head-1 logit ``a * down_j`` (integers <= 1020), so with weights
# ``w_0``, ``w_1`` in {+-1} the score ``(w_0 * a * up_j + w_1 * a * down_j) / 8`` is exact in FP32 under every
# reduction order and every other head contributes a signed zero only.
#
# Every case has its historical small geometry (the default) and a scaled geometry (``num_queries`` queries at the
# tail of ``num_keys`` keys, one segment) built by the same rule, so a call runs long enough for benchmark timing to
# resolve small differences; the scaled ``distinct`` case separates up to 65535 keys exactly through the head-1
# weight ``2**-8`` (score ``a * (up_j + down_j / 256) / 8``, a multiple of ``2**-11`` below ``2**10``).

UP_DIMS = slice(0, 8)
DOWN_DIMS = slice(64, 72)


def _split_level(level: int, parts: int = 8) -> list[int]:
    base, rem = divmod(int(level), parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def mixed_weights() -> list[float]:
    w = [0.25 if h % 2 else -0.25 for h in range(NUM_HEADS)]
    w[0], w[1] = 1.0, -1.0
    return w


def negative_weights() -> list[float]:
    w = [-0.25] * NUM_HEADS
    w[0], w[1] = -1.0, -1.0
    return w


def distinct_weights() -> list[float]:
    """Mixed weights with head 1 at ``2**-8``: distinct ``(up, down)`` pairs give distinct exact scores."""
    w = mixed_weights()
    w[1] = 2.0**-8
    return w


def _scaled_geometry(num_queries, num_keys) -> Optional[tuple[int, int]]:
    """``(lq, lk)`` of the scaled single-segment geometry, or ``None`` for a case's historical geometry."""
    if num_queries is None and num_keys is None:
        return None
    if num_queries is None or num_keys is None:
        raise ValueError("num_queries and num_keys must be given together")
    lq, lk = int(num_queries), int(num_keys)
    if lq < 1 or lk < 8 or lq > lk:
        raise ValueError(
            "scaled exact cases need 1 <= num_queries <= num_keys and num_keys >= 8"
        )
    return lq, lk


def _scaled_top_k(lk: int) -> int:
    return min(DEFAULT_TOP_K, lk // 4)


def make_exact_inputs(
    segments,
    *,
    top_k: int,
    device="cuda",
    label: str,
    q_causal_offsets=None,
) -> IndexerInputs:
    """Build an exactly representable problem from per-segment specs.

    Each spec: ``lq`` (queries), ``up`` (list of key levels), optional ``down``
    (same length), optional ``weights`` (list of 32 floats or ``row ->
    list``), optional ``amp`` (list per row or ``row -> int``; default cycles
    1, 2, 4).
    """
    device = torch.device(device)
    seg_q_len = [int(s["lq"]) for s in segments]
    seg_k_len = [len(s["up"]) for s in segments]
    cu_q, cu_k = cumulative(seg_q_len), cumulative(seg_k_len)
    q = torch.zeros((cu_q[-1], NUM_HEADS, HEAD_DIM), dtype=torch.float32)
    w = torch.zeros((cu_q[-1], NUM_HEADS), dtype=torch.float32)
    k = torch.zeros((cu_k[-1], HEAD_DIM), dtype=torch.float32)
    for s, spec in enumerate(segments):
        up = [int(v) for v in spec["up"]]
        down = [int(v) for v in spec.get("down", [0] * len(up))]
        if len(down) != len(up):
            raise ValueError("up and down must have the same length")
        for j, (u_level, d_level) in enumerate(zip(up, down, strict=True)):
            if not (0 <= u_level <= 255 and 0 <= d_level <= 255):
                raise ValueError("levels must be integers in [0, 255]")
            k[cu_k[s] + j, UP_DIMS] = torch.tensor(
                _split_level(u_level), dtype=torch.float32
            )
            k[cu_k[s] + j, DOWN_DIMS] = torch.tensor(
                _split_level(d_level), dtype=torch.float32
            )
        weights = spec.get("weights", mixed_weights())
        amp = spec.get("amp")
        for u in range(seg_q_len[s]):
            t = cu_q[s] + u
            a = (
                amp(u)
                if callable(amp)
                else (amp[u] if amp is not None else (1, 2, 4)[u % 3])
            )
            q[t, 0, UP_DIMS] = float(a)
            q[t, 1, DOWN_DIMS] = float(a)
            row_w = weights(u) if callable(weights) else weights
            w[t] = torch.tensor(list(row_w), dtype=torch.float32)
    return build_inputs(
        q.to(torch.bfloat16).to(device),
        k.to(torch.bfloat16).to(device),
        w.to(device),
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=EXACT_SCALE,
        q_causal_offsets=q_causal_offsets,
        label=label,
    )


def exact_closed_form(inputs: IndexerInputs) -> Callable[[int, int], float]:
    """``score(row, local_key_id)`` of an exact problem as a Python float, from the input tensors alone."""
    q = inputs.q.float().cpu()
    k = inputs.k.float().cpu()
    w = inputs.w.cpu()
    up = k[:, UP_DIMS].sum(dim=1).tolist()
    down = k[:, DOWN_DIMS].sum(dim=1).tolist()
    amp = q[:, 0, 0].tolist()
    w0 = w[:, 0].tolist()
    w1 = w[:, 1].tolist()
    seg = segment_of_rows(inputs, device="cpu").tolist()
    k_starts = inputs.key_starts()
    scale = inputs.softmax_scale

    def score(row: int, local_id: int) -> float:
        j = k_starts[seg[row]] + int(local_id)
        a = amp[row]
        term0 = w0[row] * max(scale * a * up[j], 0.0)
        term1 = w1[row] * max(scale * a * down[j], 0.0)
        return term0 + term1

    return score


def _shuffled_levels(count: int, seed: int) -> list[int]:
    levels = list(range(1, count + 1))
    random.Random(seed).shuffle(levels)
    return levels


def _case_distinct(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 8, "up": _shuffled_levels(96, 1)},
                {"lq": 40, "up": _shuffled_levels(40, 2)},
            ],
            top_k=24,
            device=device,
            label="exact_distinct",
        )
    lq, lk = geometry
    if lk > 65535:
        raise ValueError("the scaled distinct case separates at most 65535 keys")
    # distinct (up, down) pairs: level = up * 256 + down
    levels = _shuffled_levels(lk, 1)
    return make_exact_inputs(
        [
            {
                "lq": lq,
                "up": [level // 256 for level in levels],
                "down": [level % 256 for level in levels],
                "weights": distinct_weights(),
            }
        ],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_distinct",
    )


def _case_cutoff_ties(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        up = [j // 3 + 1 for j in range(48)]
        return make_exact_inputs(
            [{"lq": 7, "up": up}, {"lq": 30, "up": up[:30]}],
            top_k=10,
            device=device,
            label="exact_cutoff_ties",
        )
    lq, lk = geometry
    group = -(-lk // 255)  # equal-level groups of this size keep every level <= 255
    top_k = _scaled_top_k(lk)
    if (lk - top_k) % group == 0:
        # the selection boundary of the row seeing every key must fall inside a tie group
        top_k += 1
    return make_exact_inputs(
        [{"lq": lq, "up": [j // group + 1 for j in range(lk)]}],
        top_k=top_k,
        device=device,
        label="exact_cutoff_ties",
    )


def _case_all_equal(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        return make_exact_inputs(
            [{"lq": 6, "up": [5] * 50}, {"lq": 30, "up": [5] * 30}],
            top_k=16,
            device=device,
            label="exact_all_equal",
        )
    lq, lk = geometry
    return make_exact_inputs(
        [{"lq": lq, "up": [5] * lk}],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_all_equal",
    )


def _case_signed_zeros(device, *, num_queries=None, num_keys=None):
    """Zero-score keys (six of every eight ids) with the selection boundary inside the zero group of every row.

    Even rows use all-negative head weights (an IEEE sequential head sum of their zero terms would be ``-0.0``;
    the program's positive-accumulator reduction gives ``+0.0``, so the bit check tells the two models apart),
    odd rows mixed-sign weights.  Segment 0 rows see 49..64 keys, segment 1 rows 21..40; with
    ``top_k = 12`` every mixed-weight row selects its <= 8 positive keys plus some but not all zeros, and every
    all-negative row selects 12 of >= 15 zeros, so the id-descending tie rule and the zero bits are exercised on
    every row.  The scaled geometry keeps the pattern (levels cycle through 1..255) with ``top_k = num_keys // 4``
    (at most 2048), which lies between the <= 1/8 positive keys and the >= 3/4 zeros of every row as long as the
    first row sees enough keys; a geometry breaking that is rejected.
    """
    geometry = _scaled_geometry(num_queries, num_keys)
    n = 64 if geometry is None else geometry[1]
    up, down = [0] * n, [0] * n
    for j in range(n):
        if j % 8 == 6:
            down[j] = (j // 8) % 255 + 1
        elif j % 8 == 7:
            up[j] = (j // 8) % 255 + 1

    def weights(u):
        return negative_weights() if u % 2 == 0 else mixed_weights()

    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 16, "up": up, "down": down, "weights": weights},
                {"lq": 20, "up": up[:40], "down": down[:40], "weights": weights},
            ],
            top_k=12,
            device=device,
            label="exact_signed_zeros",
        )
    lq, lk = geometry
    top_k = _scaled_top_k(lk)
    for visible in (lk - lq + 1, lk):  # the rows seeing the fewest and the most keys
        positive, negative = visible // 8, (visible + 1) // 8
        if not positive < top_k < visible - negative:
            raise ValueError(
                "signed_zeros needs #positive < top_k < #positive + #zeros on every row"
            )
    return make_exact_inputs(
        [{"lq": lq, "up": up, "down": down, "weights": weights}],
        top_k=top_k,
        device=device,
        label="exact_signed_zeros",
    )


def _case_negative(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    n = 100 if geometry is None else geometry[1]
    up, down = [0] * n, [0] * n
    for j in range(n):
        if j % 2 == 0:
            up[j] = (j // 2) % 37 + 1
        else:
            down[j] = (j // 2) % 29 + 1

    def weights(u):
        return negative_weights() if u % 3 == 0 else mixed_weights()

    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 12, "up": up, "down": down, "weights": weights},
                {"lq": 36, "up": up[:36], "down": down[:36], "weights": weights},
            ],
            top_k=40,
            device=device,
            label="exact_negative",
        )
    lq, lk = geometry
    return make_exact_inputs(
        [{"lq": lq, "up": up, "down": down, "weights": weights}],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_negative",
    )


def _case_few_winners_large_tie(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    lq, lk = (6, 4096) if geometry is None else geometry
    if lk < 128:
        raise ValueError("few_winners_large_tie needs at least 128 keys")
    up = [1] * lk
    # five winners at the relative positions of the 4096-key geometry (ids 3, 1000, 2047, 2048 and 4095 there)
    for j, level in (
        (3, 50),
        (lk // 4 - 24, 40),
        (lk // 2 - 1, 30),
        (lk // 2, 20),
        (lk - 1, 10),
    ):
        up[j] = level
    return make_exact_inputs(
        [{"lq": lq, "up": up}],
        top_k=2048 if geometry is None else _scaled_top_k(lk),
        device=device,
        label="exact_few_winners_large_tie",
    )


EXACT_CASES: dict[str, Callable[..., IndexerInputs]] = {
    "distinct": _case_distinct,
    "cutoff_ties": _case_cutoff_ties,
    "all_equal": _case_all_equal,
    "signed_zeros": _case_signed_zeros,
    "negative": _case_negative,
    "few_winners_large_tie": _case_few_winners_large_tie,
}


def make_exact_case(
    name: str,
    device="cuda",
    *,
    num_queries: Optional[int] = None,
    num_keys: Optional[int] = None,
) -> IndexerInputs:
    """An exactly representable case in its historical geometry or, with both sizes, as one segment of
    ``num_queries`` queries at the tail of ``num_keys`` keys built by the same rule."""
    return EXACT_CASES[name](device, num_queries=num_queries, num_keys=num_keys)


# -- packed causality ---------------------------------------------------------------------------------------------
PACKED_GEOMETRIES: dict[str, dict[str, Any]] = {
    "equal_lengths": dict(
        seg_q_len=[256, 384, 320], seg_k_len=[256, 384, 320], top_k=128
    ),
    "unequal_lengths": dict(
        seg_q_len=[48, 160, 90], seg_k_len=[900, 120, 90], top_k=128
    ),
    "positive_offsets": dict(
        seg_q_len=[100, 70], seg_k_len=[600, 450], top_k=96, q_causal_offsets=[7, 123]
    ),
    "negative_offsets": dict(
        seg_q_len=[100, 70], seg_k_len=[180, 260], top_k=96, q_causal_offsets=[-5, -40]
    ),
    "ratio2": dict(seg_q_len=[480, 251], seg_k_len=[240, 126], top_k=96, ratio=2),
    "ratio2_offsets": dict(
        seg_q_len=[320, 90],
        seg_k_len=[380, 60],
        top_k=96,
        ratio=2,
        q_causal_offsets=[150, -9],
    ),
    "empty_segments": dict(
        seg_q_len=[0, 60, 0, 72], seg_k_len=[50, 0, 0, 220], top_k=64
    ),
    "singletons": dict(seg_q_len=[1, 1, 2], seg_k_len=[1, 6, 1], top_k=16),
    "short_rows": dict(seg_q_len=[40, 12], seg_k_len=[90, 280], top_k=256),
    "k1": dict(seg_q_len=[260, 90], seg_k_len=[260, 800], top_k=1),
    "k4096": dict(seg_q_len=[12, 20], seg_k_len=[4600, 4200], top_k=4096),
}


def make_packed_inputs(
    name: str,
    device="cuda",
    *,
    seed: int = 100,
    peaked: bool = False,
    extra_segments: int = 0,
    extra_segment_queries: int = 1024,
    extra_segment_keys: int = 8192,
) -> IndexerInputs:
    """A packed geometry, optionally followed by ``extra_segments`` iid segments of ``extra_segment_queries``
    queries over ``extra_segment_keys`` keys (same ``top_k`` and ``ratio``; each appended segment sees all of its
    keys from its last query on, i.e. offset ``ratio * keys - queries``) so a call runs long enough for benchmark
    timing to resolve small differences while the geometry's own segments stay exactly as declared."""
    spec = dict(PACKED_GEOMETRIES[name])
    seg_q_len = list(spec.pop("seg_q_len"))
    seg_k_len = list(spec.pop("seg_k_len"))
    extra_segments = int(extra_segments)
    if extra_segments < 0:
        raise ValueError("extra_segments must be >= 0")
    if extra_segments:
        lq, lk = int(extra_segment_queries), int(extra_segment_keys)
        if lq < 1 or lk < 1:
            raise ValueError("extra segments need at least one query and one key")
        ratio = int(spec.get("ratio", 1))
        offsets = spec.get("q_causal_offsets")
        if offsets is not None or ratio != 1:  # the ratio-1 default already is lk - lq
            spec["q_causal_offsets"] = (
                effective_offsets(seg_q_len, seg_k_len, offsets, ratio)
                + [ratio * lk - lq] * extra_segments
            )
        seg_q_len += [lq] * extra_segments
        seg_k_len += [lk] * extra_segments
    return make_random_inputs(
        seg_q_len,
        seg_k_len,
        device=device,
        seed=seed,
        peaked=peaked,
        label=f"packed_{name}",
        **spec,
    )


def make_boundary_inputs(
    lq: int, lk: int, *, top_k: int, device="cuda", seed: int = 7
) -> IndexerInputs:
    """One segment of ``lq`` queries at the tail of ``lk`` keys (default offset ``lk - lq``)."""
    return make_random_inputs(
        [lq],
        [lk],
        top_k=top_k,
        device=device,
        seed=seed,
        label=f"boundary_{lq}x{lk}_k{top_k}",
    )


def make_nonfinite_inputs(
    device="cuda", *, seed: int = 11, top_k: int = 64
) -> IndexerInputs:
    """Three segments: [0] finite, [1] NaN queries and infinite keys, [2] BF16-max values whose products overflow FP32."""
    inputs = make_random_inputs(
        [200, 64, 40],
        [200, 260, 40],
        top_k=top_k,
        device=device,
        seed=seed,
        label="nonfinite",
    )
    q, k = inputs.q, inputs.k
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    q[q_starts[1] + 3] = float("nan")
    q[q_starts[1] + 7, 5] = float("nan")
    k[k_starts[1] + 5] = float("inf")
    k[k_starts[1] + 9, 17] = float("-inf")
    big = torch.finfo(torch.bfloat16).max
    q[q_starts[2] : q_starts[2] + 8] = big
    k[k_starts[2] : k_starts[2] + 8] = big
    k[k_starts[2] + 8 : k_starts[2] + 12] = -big
    return inputs


def clean_rows(inputs: IndexerInputs) -> torch.Tensor:
    """Rows of the finite segment of :func:`make_nonfinite_inputs`."""
    return torch.arange(0, inputs.seg_q_len[0], dtype=torch.int64, device=inputs.device)


def dirty_rows(inputs: IndexerInputs) -> torch.Tensor:
    """Rows of the non-finite segments of :func:`make_nonfinite_inputs`."""
    return torch.arange(
        inputs.seg_q_len[0], inputs.num_queries, dtype=torch.int64, device=inputs.device
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fp32_ieee_matmul():
    """True-FP32 GEMMs (no TF32, no reduced-precision reductions); restores the previous policy."""
    prev_precision = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul = torch.backends.cuda.matmul
    prev_bf16 = getattr(matmul, "allow_bf16_reduced_precision_reduction", None)
    prev_fp16 = getattr(matmul, "allow_fp16_reduced_precision_reduction", None)
    try:
        torch.set_float32_matmul_precision("highest")
        matmul.allow_tf32 = False
        if prev_bf16 is not None:
            matmul.allow_bf16_reduced_precision_reduction = False
        if prev_fp16 is not None:
            matmul.allow_fp16_reduced_precision_reduction = False
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        matmul.allow_tf32 = prev_tf32
        if prev_bf16 is not None:
            matmul.allow_bf16_reduced_precision_reduction = prev_bf16
        if prev_fp16 is not None:
            matmul.allow_fp16_reduced_precision_reduction = prev_fp16


def _relu_plus_zero(x: torch.Tensor) -> torch.Tensor:
    """``x if x > 0 else +0.0`` (also for ``-0.0``)."""
    return torch.where(x > 0, x, torch.zeros((), dtype=x.dtype, device=x.device))


def _dot_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_d a[..., d] * b[..., d]`` accumulated in order ``d = 0 .. D-1`` with elementwise ops (broadcasting)."""
    acc = a[..., 0] * b[..., 0]
    for d in range(1, a.shape[-1]):
        acc = acc + a[..., d] * b[..., d]
    return acc


def _heads_sequential(terms: torch.Tensor) -> torch.Tensor:
    """``((t_0 + t_1) + t_2) + ...`` over dimension 1 (IEEE signed zeros)."""
    acc = terms.select(1, 0).clone()
    for h in range(1, terms.shape[1]):
        acc = acc + terms.select(1, h)
    return acc


def _logits(
    q: torch.Tensor, k: torch.Tensor, order: str, *, batched: bool
) -> torch.Tensor:
    """``[c, H, D] x [n, D] -> [c, H, n]`` (``batched=False``) or ``[c, H, D] x [c, n, D] -> [c, H, n]``."""
    if order == "sequential":
        kk = k[:, None, :, :] if batched else k[None, None, :, :]
        return _dot_sequential(q[:, :, None, :], kk)
    if order != "matmul":
        raise ValueError(f"order must be 'sequential' or 'matmul', got {order!r}")
    kt = k.transpose(1, 2) if batched else k.t()
    if q.dtype == torch.float32:
        with fp32_ieee_matmul():
            return torch.matmul(q, kt)
    return torch.matmul(q, kt)


def _fma_fp32(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``fl32(c + a * b)``: the product of two FP32 values is exact in FP64 and the
    FP64 sum is rounded once to FP32 (a double rounding differs from the fused
    operation only when the FP64 sum lands on an FP32 halfway point; the
    bit-exact cases are exactly representable and never do, the bound-based
    checks absorb it)."""
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(
        torch.float32
    )


def _heads_program_model(relu2: torch.Tensor, w_scaled: torch.Tensor) -> torch.Tensor:
    """The generated program's head reduction (module docstring): four interleaved
    FMA chains over ``h = m, m + 4, ...`` from ``+0.0``, then
    ``fl(fl(fl(c_0 + c_2) + fl(c_1 + c_3)) * 0.5)``.  ``relu2`` is ``[c, H, n]``
    (``fl(x + |x|)``), ``w_scaled`` ``[c, H, 1]`` (``fl(w * scale)``), FP32."""
    num_heads = relu2.shape[1]
    chains = []
    for m in range(4):
        acc = torch.zeros_like(relu2.select(1, 0))  # +0.0
        for h in range(m, num_heads, 4):
            acc = _fma_fp32(acc, relu2.select(1, h), w_scaled.select(1, h))
        chains.append(acc)
    return ((chains[0] + chains[2]) + (chains[1] + chains[3])) * 0.5


def _finish_scores(logits64, logits32, w_rows, scale: float):
    w64 = w_rows.to(torch.float64).unsqueeze(-1)
    s64 = _heads_sequential(w64 * _relu_plus_zero(logits64 * scale))
    relu2 = (
        logits32 + logits32.abs()
    )  # fl(x + |x|): +0.0 for x <= 0 (also -0.0), NaN for NaN and -inf
    w_scaled = (
        w_rows * torch.tensor(scale, dtype=torch.float32, device=w_rows.device)
    ).unsqueeze(-1)
    s32 = _heads_program_model(relu2, w_scaled)
    return s64, s32


def score_block(
    q_rows,
    w_rows,
    k_rows,
    scale: float,
    *,
    order: str = "sequential",
    with_bound: bool = False,
):
    """Scores of every (query, key) pair: ``q_rows [c, H, D]`` bf16, ``w_rows [c, H]`` fp32, ``k_rows [n, D]`` bf16.

    Returns ``(s64 [c, n], s32 [c, n], A64 [c, n] or None)`` where ``A`` is the bound base of the module docstring.
    """
    q64, k64 = q_rows.to(torch.float64), k_rows.to(torch.float64)
    q32, k32 = q_rows.to(torch.float32), k_rows.to(torch.float32)
    s64, s32 = _finish_scores(
        _logits(q64, k64, order, batched=False),
        _logits(q32, k32, order, batched=False),
        w_rows,
        scale,
    )
    bound_base = None
    if with_bound:
        abs_logits = torch.matmul(
            q64.abs(), k64.abs().t()
        )  # [c, H, n]; any order is fine for a bound
        bound_base = (
            torch.bmm(w_rows.to(torch.float64).abs().unsqueeze(1), abs_logits).squeeze(
                1
            )
            * scale
        )
    return s64, s32, bound_base


def score_selected(
    inputs: IndexerInputs, ids: torch.Tensor, *, rows=None, order: str = "sequential"
):
    """Scores at a ``[R, K]`` table of segment-local ids (``-1`` = padding -> ``-inf`` / ``-inf`` / ``0``).

    Returns ``(s64, s32, A64)`` of shape ``[R, K]`` for the rows ``rows`` (default: all rows).
    """
    device = inputs.device
    if rows is None:
        rows = torch.arange(inputs.num_queries, dtype=torch.int64, device=device)
    R, K = int(rows.numel()), int(ids.shape[1])
    s64 = torch.full((R, K), PAD_SCORE, dtype=torch.float64, device=device)
    s32 = torch.full((R, K), PAD_SCORE, dtype=torch.float32, device=device)
    a64 = torch.zeros((R, K), dtype=torch.float64, device=device)
    if R == 0 or K == 0 or inputs.num_keys == 0:
        return s64, s32, a64
    k_starts = torch.tensor(inputs.key_starts()[:-1], dtype=torch.int64, device=device)
    seg = segment_of_rows(inputs)
    per_row = K * (HEAD_DIM * 12 + NUM_HEADS * 40 + 64)
    chunk = max(1, min(R, CHUNK_BYTES // per_row))
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        rr = rows[r0:r1]
        local = ids[r0:r1].to(torch.int64)
        valid = local >= 0
        global_ids = k_starts[seg[rr]].unsqueeze(1) + local.clamp_min(0)
        qb = inputs.q.index_select(0, rr)
        wb = inputs.w.index_select(0, rr)
        kb = inputs.k.index_select(0, global_ids.reshape(-1)).reshape(
            r1 - r0, K, HEAD_DIM
        )
        q64, k64 = qb.to(torch.float64), kb.to(torch.float64)
        l64 = _logits(q64, k64, order, batched=True)
        l32 = _logits(qb.to(torch.float32), kb.to(torch.float32), order, batched=True)
        b64, b32 = _finish_scores(l64, l32, wb, inputs.softmax_scale)
        abs_logits = torch.matmul(q64.abs(), k64.abs().transpose(1, 2))
        base = (
            torch.bmm(wb.to(torch.float64).abs().unsqueeze(1), abs_logits).squeeze(1)
            * inputs.softmax_scale
        )
        s64[r0:r1] = torch.where(valid, b64, s64[r0:r1])
        s32[r0:r1] = torch.where(valid, b32, s32[r0:r1])
        a64[r0:r1] = torch.where(valid, base, a64[r0:r1])
    return s64, s32, a64


def score_pairs(
    inputs: IndexerInputs,
    rows: torch.Tensor,
    ids: torch.Tensor,
    *,
    order: str = "sequential",
):
    """Scores of arbitrary ``(row, local id)`` pairs (int64 ``[n]`` each): ``(s64 [n], s32 [n], A64 [n])``."""
    s64, s32, a64 = score_selected(inputs, ids.view(-1, 1), rows=rows, order=order)
    return s64[:, 0], s32[:, 0], a64[:, 0]


def pair_bound(inputs: IndexerInputs, bound_base: torch.Tensor) -> torch.Tensor:
    """``gamma_n * A`` for a tensor of bound bases ``A``."""
    return bound_base * gamma_bound(
        accumulation_terms(inputs.q.shape[1], inputs.q.shape[2])
    )


def count_visible_pairs(inputs: IndexerInputs) -> int:
    return int(visible_rows(inputs, device="cpu").sum().item())


def resolve_order(inputs: IndexerInputs, order: str) -> str:
    if order in ("sequential", "matmul"):
        return order
    if order != "auto":
        raise ValueError(
            f"order must be 'auto', 'sequential' or 'matmul', got {order!r}"
        )
    return (
        "sequential"
        if count_visible_pairs(inputs) <= SEQUENTIAL_PAIR_LIMIT
        else "matmul"
    )


# ---------------------------------------------------------------------------
# Exact selection
# ---------------------------------------------------------------------------


@dataclass
class ReferenceSelection:
    """Reference output (ids ascending per row; padding ``-1`` / ``-inf``)."""

    inputs: IndexerInputs
    indices: torch.Tensor  # int32 [T, K]
    scores: torch.Tensor  # fp32 [T, K]: the independent FP32 scorer at the selected ids
    scores_fp64: torch.Tensor  # fp64 [T, K]: the exact scores
    visible: torch.Tensor  # int64 [T]
    selected_count: torch.Tensor  # int64 [T] = min(K, visible)
    order: str
    chunks: dict[str, int] = field(default_factory=dict)


def _ranking_key(scores: torch.Tensor) -> torch.Tensor:
    """``-0.0 -> +0.0``; NaN ranks lowest (``-inf``); everything else by value.

    The NaN placement is the reference's own selection convention (finite inputs
    are the contract's domain); it is not a guarantee of the generated program.
    """
    return torch.nan_to_num(
        scores + 0.0, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )


def _lexicographic_top(
    rank: torch.Tensor, ids: torch.Tensor, keep: int
) -> torch.Tensor:
    """Permutation ``[c, keep]`` selecting the top entries by (rank desc, id desc): two stable sorts."""
    _, by_id = torch.sort(ids, dim=1, descending=True, stable=True)
    _, by_rank = torch.sort(rank.gather(1, by_id), dim=1, descending=True, stable=True)
    return by_id.gather(1, by_rank)[:, :keep]


def plan_chunks(
    inputs: IndexerInputs, chunk_bytes: int = CHUNK_BYTES
) -> tuple[int, int]:
    """(query chunk, key chunk) under a byte budget (fp64 + fp32 logits and terms per pair)."""
    K = inputs.top_k
    max_lk = max(inputs.seg_k_len, default=1)
    key_chunk = max(1, min(max_lk, max(2 * K, 8192)))
    per_pair = NUM_HEADS * 40 + 64
    query_chunk = max(1, min(512, chunk_bytes // max(1, key_chunk * per_pair)))
    return query_chunk, key_chunk


def select_reference(
    inputs: IndexerInputs,
    *,
    order: str = "auto",
    query_chunk: Optional[int] = None,
    key_chunk: Optional[int] = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> ReferenceSelection:
    """Exact global selection per row with the issue's tie rule, memory-bounded over keys.

    Ranking uses the FP64 score.  ``query_chunk`` / ``key_chunk`` change only
    the work partition: with ``order="sequential"`` the result is bitwise
    identical for every partition, with ``"matmul"`` only up to the BLAS
    rounding of the dot products (bound-based comparisons only).
    """
    order = resolve_order(inputs, order)
    device = inputs.device
    T, K = inputs.num_queries, inputs.top_k
    cq, ck = plan_chunks(inputs, chunk_bytes)
    if query_chunk is not None:
        cq = max(1, int(query_chunk))
    if key_chunk is not None:
        ck = max(1, int(key_chunk))
    out_ids = torch.full((T, K), PAD_ID, dtype=torch.int32, device=device)
    out_s32 = torch.full((T, K), PAD_SCORE, dtype=torch.float32, device=device)
    out_s64 = torch.full((T, K), PAD_SCORE, dtype=torch.float64, device=device)
    visible = visible_rows(inputs)
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    for s in range(inputs.num_segments):
        q0, lq = q_starts[s], inputs.seg_q_len[s]
        k0, lk = k_starts[s], inputs.seg_k_len[s]
        if lq == 0 or lk == 0:
            continue
        keep = min(K, lk)
        for t0 in range(q0, q0 + lq, cq):
            t1 = min(q0 + lq, t0 + cq)
            c = t1 - t0
            vis_rows = visible[t0:t1]
            j_end = int(
                vis_rows.max().item()
            )  # visibility is monotone within a segment: the last row sees most
            if j_end == 0:
                continue
            qb, wb = inputs.q[t0:t1], inputs.w[t0:t1]
            best_rank = torch.empty((c, 0), dtype=torch.float64, device=device)
            best_id = torch.empty((c, 0), dtype=torch.int64, device=device)
            best_s32 = torch.empty((c, 0), dtype=torch.float32, device=device)
            best_s64 = torch.empty((c, 0), dtype=torch.float64, device=device)
            for j0 in range(0, j_end, ck):
                j1 = min(j_end, j0 + ck)
                s64, s32, _ = score_block(
                    qb,
                    wb,
                    inputs.k[k0 + j0 : k0 + j1],
                    inputs.softmax_scale,
                    order=order,
                )
                ids = (
                    torch.arange(j0, j1, dtype=torch.int64, device=device)
                    .unsqueeze(0)
                    .expand(c, -1)
                )
                seen = ids < vis_rows.unsqueeze(1)
                rank = torch.where(
                    seen, _ranking_key(s64), torch.full_like(s64, float("-inf"))
                )
                ids = torch.where(seen, ids, torch.full_like(ids, -1))
                cand_rank = torch.cat([best_rank, rank], dim=1)
                cand_id = torch.cat([best_id, ids], dim=1)
                perm = _lexicographic_top(
                    cand_rank, cand_id, min(keep, cand_rank.shape[1])
                )
                best_rank = cand_rank.gather(1, perm)
                best_id = cand_id.gather(1, perm)
                best_s32 = torch.cat([best_s32, s32], dim=1).gather(1, perm)
                best_s64 = torch.cat([best_s64, s64], dim=1).gather(1, perm)
            # ascending ids, padding (id -1) last
            key = torch.where(
                best_id >= 0, best_id, torch.full_like(best_id, INT64_MAX)
            )
            key_sorted, perm = torch.sort(key, dim=1)
            pad = key_sorted == INT64_MAX
            n = key_sorted.shape[1]
            out_ids[t0:t1, :n] = torch.where(
                pad, torch.full_like(key_sorted, PAD_ID), key_sorted
            ).to(torch.int32)
            out_s32[t0:t1, :n] = torch.where(
                pad, torch.full_like(best_s32, PAD_SCORE), best_s32.gather(1, perm)
            )
            out_s64[t0:t1, :n] = torch.where(
                pad, torch.full_like(best_s64, PAD_SCORE), best_s64.gather(1, perm)
            )
    return ReferenceSelection(
        inputs=inputs,
        indices=out_ids,
        scores=out_s32,
        scores_fp64=out_s64,
        visible=visible,
        selected_count=visible.clamp(max=K),
        order=order,
        chunks={"query_chunk": cq, "key_chunk": ck},
    )


def bruteforce_rows(inputs: IndexerInputs) -> tuple[list[list[int]], list[list[float]]]:
    """Pure-Python per-row selection (FP64 scores from torch, ranking in Python) for tiny problems."""
    q = inputs.q.to(torch.float64).cpu()
    k = inputs.k.to(torch.float64).cpu()
    w = inputs.w.to(torch.float64).cpu()
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    offsets = inputs.offsets()
    ids_out: list[list[int]] = []
    scores_out: list[list[float]] = []
    for s in range(inputs.num_segments):
        for u in range(inputs.seg_q_len[s]):
            t = q_starts[s] + u
            vis = visible_scalar(offsets[s], u, inputs.ratio, inputs.seg_k_len[s])
            scored = []
            for j in range(vis):
                logit = (q[t] * k[k_starts[s] + j]).sum(dim=1) * inputs.softmax_scale
                terms = w[t] * torch.clamp(logit, min=0.0)
                score = float(terms.sum())
                scored.append((score + 0.0, j, score))
            scored.sort(key=lambda e: (e[0], e[1]), reverse=True)
            chosen = sorted(scored[: inputs.top_k], key=lambda e: e[1])
            row_ids = [j for _, j, _ in chosen] + [PAD_ID] * (
                inputs.top_k - len(chosen)
            )
            row_scores = [sc for _, _, sc in chosen] + [PAD_SCORE] * (
                inputs.top_k - len(chosen)
            )
            ids_out.append(row_ids)
            scores_out.append(row_scores)
    return ids_out, scores_out


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------


@dataclass
class Judgement:
    passed: bool
    failures: list[str]
    rows: int
    rows_identical: int
    rows_within_uncertainty: int
    max_fp64_error_over_bound: float
    max_fp32_error_over_bound: float
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        text = (
            f"{head}: rows={self.rows} identical={self.rows_identical} within_uncertainty={self.rows_within_uncertainty} "
            f"max|err64|/bound={self.max_fp64_error_over_bound:.3g} max|err32|/bound={self.max_fp32_error_over_bound:.3g}"
        )
        if self.failures:
            text += "\n  " + "\n  ".join(self.failures[:20])
            if len(self.failures) > 20:
                text += f"\n  ... {len(self.failures) - 20} more"
        return text


def structure_failures(
    inputs: IndexerInputs,
    indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    rows=None,
    finite_domain: bool = True,
) -> list[str]:
    """Exact structural checks: shapes, dtypes, padding, uniqueness, ascending order, visibility."""
    failures: list[str] = []
    T, K = inputs.num_queries, inputs.top_k
    if tuple(indices.shape) != (T, K):
        failures.append(f"indices shape {tuple(indices.shape)} != {(T, K)}")
    if tuple(scores.shape) != (T, K):
        failures.append(f"scores shape {tuple(scores.shape)} != {(T, K)}")
    if indices.dtype != torch.int32:
        failures.append(f"indices dtype {indices.dtype} != int32")
    if scores.dtype != torch.float32:
        failures.append(f"scores dtype {scores.dtype} != float32")
    if failures:
        return failures
    device = indices.device
    visible = visible_rows(inputs, device)
    selected = visible.clamp(max=K)
    if rows is not None:
        rows = rows.to(device=device, dtype=torch.int64)
        indices, scores, visible, selected = (
            indices[rows],
            scores[rows],
            visible[rows],
            selected[rows],
        )
    if indices.shape[0] == 0:
        return failures
    slot = torch.arange(K, dtype=torch.int64, device=device).unsqueeze(0)
    valid = slot < selected.unsqueeze(1)
    ids = indices.to(torch.int64)

    def first(mask: torch.Tensor, what: str) -> None:
        if bool(mask.any()):
            r, c = mask.nonzero()[0].tolist()
            row = int(rows[r].item()) if rows is not None else r
            failures.append(
                f"{what} at row {row} slot {c} (id={int(ids[r, c])}, score={float(scores[r, c])!r})"
            )

    first(
        valid & ((ids < 0) | (ids >= visible.unsqueeze(1))),
        "invisible or out-of-segment id in a valid slot",
    )
    first((~valid) & (ids != PAD_ID), "padding slot without id -1")
    first((~valid) & ~torch.isneginf(scores), "padding slot without -inf score")
    if K > 1:
        first(
            valid[:, 1:] & ~(ids[:, 1:] > ids[:, :-1]),
            "ids not strictly ascending (duplicate or unsorted)",
        )
    if finite_domain:
        first(valid & ~torch.isfinite(scores), "non-finite score in a valid slot")
    return failures


def judge(
    result,
    reference: ReferenceSelection,
    *,
    exact_bits: bool = False,
    zero_sign_policy: str = "positive_accumulator",
    rows=None,
    bound=None,
) -> Judgement:
    """Judge ``result = (indices, scores)`` against the reference under the near-tie rule (module docstring).

    ``bound``: ``None`` -> ``gamma_n * A`` per pair; a float -> uniform absolute bound.  ``exact_bits=True``
    demands id equality and bitwise equality with the FP32 scorer (exactly representable problems).  The scorer
    follows the positive-accumulator reduction of the generated program (module docstring), so that is the only
    ``zero_sign_policy`` it can judge bit-exactly; a program declaring another policy needs its own model.
    """
    if zero_sign_policy not in ZERO_SIGN_POLICIES:
        raise ValueError(
            f"zero_sign_policy must be one of {ZERO_SIGN_POLICIES}, got {zero_sign_policy!r}"
        )
    if zero_sign_policy != "positive_accumulator":
        raise NotImplementedError(
            f"the reference models the positive-accumulator reduction; no model for {zero_sign_policy!r}"
        )
    inputs, order = reference.inputs, reference.order
    indices, scores = result
    failures = structure_failures(inputs, indices, scores, rows=rows)
    if failures:
        return Judgement(
            False, failures, 0, 0, 0, math.inf, math.inf, {"stage": "structure"}
        )
    device = indices.device
    T, K = inputs.num_queries, inputs.top_k
    rows = (
        torch.arange(T, dtype=torch.int64, device=device)
        if rows is None
        else rows.to(device=device, dtype=torch.int64)
    )
    R = int(rows.numel())
    got_ids = indices.index_select(0, rows).to(torch.int64)
    got_scores = scores.index_select(0, rows)
    ref_ids = reference.indices.index_select(0, rows).to(torch.int64)
    selected = reference.selected_count.index_select(0, rows)
    slot = torch.arange(K, dtype=torch.int64, device=device).unsqueeze(0)
    valid = slot < selected.unsqueeze(1)

    def bound_of(base: torch.Tensor) -> torch.Tensor:
        return (
            pair_bound(inputs, base)
            if bound is None
            else torch.full_like(base, float(bound))
        )

    # 1. returned score bits against the exact score and the independent FP32 scorer
    s64, s32, base = score_selected(inputs, got_ids, rows=rows, order=order)
    b = bound_of(base)
    got64 = got_scores.to(torch.float64)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    err64 = torch.where(valid, (got64 - s64).abs(), zero)
    err32 = torch.where(valid, (got64 - s32.to(torch.float64)).abs(), zero)
    safe = torch.where(b > 0, b, torch.ones_like(b))
    inf = torch.full_like(b, math.inf)
    ratio64 = torch.where(b > 0, err64 / safe, torch.where(err64 > 0, inf, zero))
    ratio32 = torch.where(b > 0, err32 / safe, torch.where(err32 > 0, inf, zero))
    max64 = float(ratio64.max().item()) if R else 0.0
    max32 = float(ratio32.max().item()) if R else 0.0
    for mask, what in (
        (valid & (err64 > b), "score outside the bound of the exact FP64 score"),
        (valid & (err32 > b), "score outside the bound of the FP32 scorer"),
    ):
        if bool(mask.any()):
            r, c = mask.nonzero()[0].tolist()
            failures.append(
                f"{what} at row {int(rows[r])} slot {c}: id={int(got_ids[r, c])} got={float(got_scores[r, c]):.9g} "
                f"exact={float(s64[r, c]):.9g} fp32={float(s32[r, c]):.9g} bound={float(b[r, c]):.3g}"
            )
    if exact_bits:
        expected32 = s32  # the program model never produces -0.0; zero bits are compared as computed
        got_bits = got_scores.contiguous().view(torch.int32)
        exp_bits = expected32.contiguous().view(torch.int32)
        mismatch = valid & (got_bits != exp_bits)
        if bool(mismatch.any()):
            r, c = mismatch.nonzero()[0].tolist()
            failures.append(
                f"score bits differ from the FP32 scorer at row {int(rows[r])} slot {c}: id={int(got_ids[r, c])} "
                f"got=0x{int(got_bits[r, c]) & 0xFFFFFFFF:08x} ({float(got_scores[r, c])!r}) "
                f"expected=0x{int(exp_bits[r, c]) & 0xFFFFFFFF:08x} ({float(expected32[r, c])!r}) [{zero_sign_policy}]"
            )

    # 2. selected id sets
    identical = (got_ids == ref_ids).all(dim=1)
    rows_identical = int(identical.sum().item())
    differing = (~identical).nonzero().squeeze(1)
    rows_within = 0
    details: dict[str, Any] = {"differing_rows": int(differing.numel()), "order": order}
    if differing.numel():
        if exact_bits:
            r = int(differing[0])
            failures.append(
                f"selected ids differ (exact case) at row {int(rows[r])}: got={got_ids[r, : min(K, 16)].tolist()} "
                f"reference={ref_ids[r, : min(K, 16)].tolist()}"
            )
        gd, rd, vd = got_ids[differing], ref_ids[differing], valid[differing]
        # Row-unique tags make membership a flat isin: tag = differing-row index << 32 | id.
        tag_g = (differing.unsqueeze(1) << 32) | gd.clamp_min(0)
        tag_r = (differing.unsqueeze(1) << 32) | rd.clamp_min(0)
        lacking = vd & ~torch.isin(tag_r, tag_g[vd])
        held = vd & ~torch.isin(tag_g, tag_r[vd])
        details["lacking_pairs"] = int(lacking.sum().item())
        details["held_instead_pairs"] = int(held.sum().item())
        if details["lacking_pairs"] != details["held_instead_pairs"]:
            failures.append(
                "internal: lacking / held counts differ although the structure passed"
            )
        l_row, l_slot = lacking.nonzero(as_tuple=True)
        h_row, h_slot = held.nonzero(as_tuple=True)
        l64, _, l_base = score_pairs(
            inputs, rows[differing[l_row]], rd[l_row, l_slot], order=order
        )
        h64, _, h_base = score_pairs(
            inputs, rows[differing[h_row]], gd[h_row, h_slot], order=order
        )
        lower = l64 - bound_of(
            l_base
        )  # lacking keys: optimistic lower bound of their exact score
        upper = h64 + bound_of(h_base)  # held keys: pessimistic upper bound
        n = int(differing.numel())
        row_lower = torch.full((n,), float("-inf"), dtype=torch.float64, device=device)
        row_upper = torch.full((n,), float("inf"), dtype=torch.float64, device=device)
        row_lower = row_lower.scatter_reduce(
            0, l_row, lower, reduce="amax", include_self=True
        )
        row_upper = row_upper.scatter_reduce(
            0, h_row, upper, reduce="amin", include_self=True
        )
        violated = row_lower > row_upper
        rows_within = int((~violated).sum().item())
        row_l = torch.full(
            (n,), float("-inf"), dtype=torch.float64, device=device
        ).scatter_reduce(0, l_row, l64, reduce="amax", include_self=True)
        row_h = torch.full(
            (n,), float("inf"), dtype=torch.float64, device=device
        ).scatter_reduce(0, h_row, h64, reduce="amin", include_self=True)
        details["max_lacking_minus_held_exact"] = (
            float((row_l - row_h).max().item()) if n else 0.0
        )
        details["max_bound_at_disagreements"] = (
            float(torch.cat([bound_of(l_base), bound_of(h_base)]).max().item())
            if l64.numel()
            else 0.0
        )
        if bool(violated.any()):
            v = int(violated.nonzero()[0])
            r_abs = int(rows[differing[v]])
            li = (l_row == v).nonzero().squeeze(1)
            hi = (h_row == v).nonzero().squeeze(1)
            li = li[lower[li].argmax()]
            hi = hi[upper[hi].argmin()]
            failures.append(
                f"clearly better key lacking at row {r_abs}: reference id {int(rd[l_row[li], l_slot[li]])} exact "
                f"{float(l64[li]):.9g} (bound {float(bound_of(l_base)[li]):.3g}) ranks above held id "
                f"{int(gd[h_row[hi], h_slot[hi]])} exact {float(h64[hi]):.9g} (bound {float(bound_of(h_base)[hi]):.3g}); "
                f"{int(violated.sum())} row(s) violate the near-tie rule"
            )
    return Judgement(
        not failures, failures, R, rows_identical, rows_within, max64, max32, details
    )


def same_bits(a, b) -> bool:
    """Bitwise identity of two ``(indices, scores)`` results."""
    ia, sa = a
    ib, sb = b
    if tuple(ia.shape) != tuple(ib.shape) or tuple(sa.shape) != tuple(sb.shape):
        return False
    return bool(torch.equal(ia, ib)) and bool(
        torch.equal(
            sa.contiguous().view(torch.int32), sb.contiguous().view(torch.int32)
        )
    )


__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_SCALE",
    "DEFAULT_TOP_K",
    "EXACT_CASES",
    "EXACT_SCALE",
    "FP32_UNIT_ROUNDOFF",
    "HEAD_DIM",
    "NUM_HEADS",
    "PACKED_GEOMETRIES",
    "PAD_ID",
    "PAD_SCORE",
    "SEQUENTIAL_PAIR_LIMIT",
    "TRAINER_KEY_ROW_STRIDE",
    "ZERO_SIGN_POLICIES",
    "IndexerInputs",
    "Judgement",
    "ReferenceSelection",
    "accumulation_terms",
    "bruteforce_rows",
    "build_inputs",
    "clean_rows",
    "count_visible_pairs",
    "cumulative",
    "dirty_rows",
    "effective_offsets",
    "exact_closed_form",
    "fp32_ieee_matmul",
    "gamma_bound",
    "judge",
    "make_boundary_inputs",
    "make_exact_case",
    "make_exact_inputs",
    "make_nonfinite_inputs",
    "make_packed_inputs",
    "make_random_inputs",
    "mixed_weights",
    "negative_weights",
    "pair_bound",
    "plan_chunks",
    "resolve_order",
    "same_bits",
    "score_block",
    "score_pairs",
    "score_selected",
    "segment_of_rows",
    "select_reference",
    "structure_failures",
    "visible_rows",
    "visible_scalar",
]
