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
"""

"""Benchmark the experimental Cake DSA sparse-attention training kernels (SM100 / SM103 / SM107).

Times the forward, the backward and the whole training step (forward +
backward, wrapper-inclusive) of ``flashinfer.experimental.cake_dsa_train`` on
the GLM-5.2 geometry (64 query heads, 512 latent + 64 rope, 512 value
dimensions, top-k 2048, BF16) over eighteen representative rows: one causal
document of 4k .. 128k tokens, context-parallel tails (4k queries over 64k /
128k keys), packed context-parallel all-gather-KV problems (32k queries over
256k keys as 1 .. 32 equal documents, and eight skewed documents) and
CP-rank spreads (32k queries over 64k / 128k / 192k keys), plus the GLM-5.2
trainer rows of issue #5675 (``packed_glm_a`` -- the recorded eight-segment
batch, ``packed_glm_b``, ``packed_glm_a_s704``, ``tail_2123x67923``,
``doc_16231``, ``doc_4095``, ``doc_4097``).  Optional baselines when
importable: FlashMLA sparse forward + cuDNN frontend sparse attention backward
(``flash_mla``, ``cudnn.DSA``) and the FA sparse-MLA kernels
(``flash_attn.cute.flash_attn_varlen_func`` with ``gather_kv_indices``).

Layouts (issue #5675).  Every row can run in the trainer's packed / strided
layouts (``--q-layout``, ``--kv-stride``, ``--dkv-acc``, ``--dkv-dst-map``;
the GLM-5.2 rows default to them and ``--preset`` selects one of them): the
canonical tensors of ``make_inputs`` are copied INTO a pre-absorption query
buffer ``[T, 64, 256]`` whose ``192:256`` channels are ``q_rope`` (head stride
256) and into a packed ``[Tkv, 576 | 704]`` KV buffer (latent columns
``0:512``, rope ``512:576``), and the backward accumulates dK/dV into a
caller-owned fp32 ``dkv_acc [Tkv, 576]``, optionally through a destination-row
map.  The ``cake`` arm then calls the public varlen entry
``dsa_sparse_attention_varlen(..., causal=True, dkv_acc=, dkv_dst_map=)`` with
the views as they are (no copies) through autograd; the baseline arms run the
contiguous copies / cat and the ``index_add_`` / ``+=`` accumulation into
``dkv_acc`` their kernels need inside the timed step.  ``dkv_acc`` is
allocated inside the timed step of every arm, so it counts in time and in the
peak memory alike.

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and
a cold L2 between iterations (per-iteration GPU span); medians over
``--steps`` iterations.  ``--accuracy`` adds the relative-L2 comparison against
the chunked FP64 reference (iid 4k x 4k and peaked-attention cases; with a
destination map the reference dK/dV is mapped the same way).

``--host-us`` measures the host side of the eager entry points (``forward``,
``backward``, the public ``dsa_sparse_attention`` forward / autograd backward /
step) as wall-clock microseconds per call with the GPU running asynchronously,
alternating the binding cache off / on inside one process (``--host-rounds``
rounds of ``--host-calls`` calls each), and checks that both paths produce the
same results and the same kernel-only time.

Usage::

    python benchmarks/bench_cake_dsa_train.py [--rows doc_4096 ...] [--arms cake,flashmla_cudnn,fa4]
        [--steps 20] [--accuracy] [--host-us] [--json out.json]
    python benchmarks/bench_cake_dsa_train.py --preset packed_glm_a --arms cake,flashmla_cudnn,fa4
    python benchmarks/bench_cake_dsa_train.py --rows doc_4096 --q-layout pre256 --kv-stride 704 \\
        --dkv-acc --dkv-dst-map perm --accuracy
"""

import argparse
import json
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.dsa_sparse_attention import (  # noqa: E402
    dsa_sparse_attention,
    dsa_sparse_attention_varlen,
)
from flashinfer.experimental.cake_dsa_train import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_dsa_train_reference import (  # noqa: E402
    D_LATENT,
    D_QK,
    D_ROPE,
    DEFAULT_SCALE,
    DEFAULT_TOPK,
    NUM_HEADS,
    calibrate_beta,
    make_inputs,
    reference_fp64,
    rel_l2,
    rel_l2_rows,
)

# name -> (query lengths per document, key lengths per document)
ROWS: dict[str, tuple[list, list]] = {}
for _n in (4096, 8192, 16384, 32768, 65536, 131072):
    ROWS[f"doc_{_n}"] = ([_n], [_n])
for _s in (65536, 131072):
    ROWS[f"cptail_4k_{_s}"] = ([4096], [_s])
for _N in (1, 2, 4, 8, 16, 32):
    ROWS[f"packed_N{_N}"] = ([32768 // _N] * _N, [262144 // _N] * _N)
_RATIOS = [1, 1, 2, 2, 4, 4, 8, 10]
ROWS["skewed8"] = ([1024 * r for r in _RATIOS], [8192 * r for r in _RATIOS])
for _s in (65536, 131072, 196608):
    ROWS[f"spread_32k_{_s}"] = ([32768], [_s])
# GLM-5.2 trainer rows (issue #5675).  packed_glm_a is the recorded batch: eight
# segments alternating a full document and the tail of a long prefix,
#   cu_seqlens_q = [0, 1777, 3532, 5655, 7765, 9888, 11998, 14121, 16231]
#   cu_seqlens_k = [0, 1777, 58619, 60742, 128665, 130788, 198711, 200834, 268757]
# (T 16231, Tkv 268757, max_seqlen_q 2123, max_seqlen_k 67923); packed_glm_b is its
# deterministic sibling (T 16172, Tkv 267520,
#   cu_seqlens_q = [0, 1770, 3518, 5634, 7736, 9852, 11954, 14070, 16172]
#   cu_seqlens_k = [0, 1770, 58390, 60506, 128100, 130216, 197810, 199926, 267520]).
GLM_A = (
    [1777, 1755, 2123, 2110, 2123, 2110, 2123, 2110],
    [1777, 56842, 2123, 67923, 2123, 67923, 2123, 67923],
)
GLM_B = (
    [1770, 1748, 2116, 2102, 2116, 2102, 2116, 2102],
    [1770, 56620, 2116, 67594, 2116, 67594, 2116, 67594],
)
ROWS["packed_glm_a"] = GLM_A
ROWS["packed_glm_b"] = GLM_B
ROWS["packed_glm_a_s704"] = GLM_A  # the same batch with KV row stride 704 (ROW_LAYOUTS)
ROWS["tail_2123x67923"] = ([2123], [67923])  # the short-query / long-key tail segment
ROWS["doc_16231"] = ([16231], [16231])  # one document at the recorded token count
ROWS["doc_4095"] = ([4095], [4095])  # tile / chunk boundary
ROWS["doc_4097"] = ([4097], [4097])


# GB200 trainer-trace rows (round 2 of issue #5675): the recorded dK/dV destination buffers have FEWER rows than the
# KV buffer (268,757 key rows into 260,611 fp32 rows for batch a, 267,520 into 259,412 for batch b -- 3.03 % of the key
# rows share a destination), and the backward runs in 4096-query chunks over the whole packed key row.
GLM_DST_ROWS = {"glm_a": (GLM_A[1], 260611), "glm_b": (GLM_B[1], 259412)}


def dst_map_rows(seq_k, dst_rows):
    """Destination row of every key row (Python ints, ``sum(seq_k)`` long) of the overlap map that reproduces a recorded
    destination-row count: segment ``i > 0`` aliases its first ``d_i`` key rows onto the previous segment's last ``d_i``
    destination rows (``sum d_i = sum(seq_k) - dst_rows``, split over segments ``1..N-1`` in proportion to their key
    counts, largest remainder), fresh rows follow in natural order."""
    seq_k = [int(x) for x in seq_k]
    S, dups = sum(seq_k), sum(seq_k) - int(dst_rows)
    if dups < 0 or (dups and len(seq_k) < 2):
        raise ValueError(
            f"dst_map_rows: {dst_rows} destination rows for {S} key rows in {len(seq_k)} segment(s)"
        )
    weight = sum(seq_k[1:])
    raw = [dups * lk / weight for lk in seq_k[1:]] if weight else []
    d = [0] + [int(x) for x in raw]
    for i in sorted(
        range(1, len(seq_k)), key=lambda i: raw[i - 1] - int(raw[i - 1]), reverse=True
    )[: dups - sum(d)]:
        d[i] += 1
    for i in range(1, len(seq_k)):
        if d[i] > min(seq_k[i], seq_k[i - 1]):
            raise ValueError(
                f"dst_map_rows: segment {i} would alias {d[i]} rows, more than min({seq_k[i]}, {seq_k[i - 1]})"
            )
    out, next_row = [], 0
    for lk, di in zip(seq_k, d, strict=True):
        out.extend(
            range(next_row - di, next_row)
        )  # aliased onto the previous segment's last di rows
        out.extend(range(next_row, next_row + lk - di))
        next_row += lk - di
    assert next_row == int(dst_rows) and len(out) == S
    return out


def bwd_chunk(seq_q, seq_k, q0, q1):
    """Segment lists ``(seq_q, seq_k)`` of one backward chunk of a packed batch: the chunk's queries ``[q0, q1)`` of the
    packed query row over the FULL key row (the recorded trainer runs the backward of a 16k-query batch in 4096-query
    chunks).  Documents outside the chunk keep their key rows as 0-query segments; a document the chunk splits becomes
    ``(n, keys up to its last own key)`` + ``(0, the rest)``."""
    seq_q, seq_k = [int(x) for x in seq_q], [int(x) for x in seq_k]
    out_q, out_k, cu = [], [], 0
    for lq, lk in zip(seq_q, seq_k, strict=True):
        lo, hi = (
            max(q0, cu),
            min(q1, cu + lq),
        )  # the chunk's queries of this document (packed positions)
        n = max(0, hi - lo)
        if n == 0:
            out_q.append(0)
            out_k.append(lk)
        else:
            own_last = (lk - lq) + (
                hi - cu
            )  # keys up to the last own key of the chunk's queries
            out_q.append(n)
            out_k.append(own_last)
            if own_last < lk:
                out_q.append(0)
                out_k.append(lk - own_last)
        cu += lq
    if sum(out_q) != q1 - q0 or sum(out_k) != sum(seq_k):
        raise ValueError(
            f"bwd_chunk: queries [{q0}, {q1}) do not fit the batch ({sum(seq_q)} queries)"
        )
    return out_q, out_k


ROWS["packed_glm_a_dstmap"] = (
    GLM_A  # the recorded batch a with its dK/dV destination map (ROW_LAYOUTS)
)
ROWS["packed_glm_b_dstmap"] = GLM_B
ROWS["packed_glm_a_s704_dstmap"] = GLM_A
ROWS["packed_glm_b_s704_dstmap"] = GLM_B
ROWS["chunk_4096x268757"] = bwd_chunk(
    *GLM_A, 0, 4096
)  # the recorded 4096-query backward chunks
ROWS["chunk_3943x268757"] = bwd_chunk(*GLM_A, 12288, 16231)
ROWS["chunk_3884x267520"] = bwd_chunk(*GLM_B, 12288, 16172)

Q_LAYOUTS = ("packed576", "pre256")
KV_STRIDES = (D_QK, 704)
DST_MAPS = ("none", "identity", "perm", "glm_a", "glm_b")
DEFAULT_LAYOUT = dict(
    q_layout="packed576", kv_stride=D_QK, dkv_acc=False, dkv_dst_map="none"
)
TRAINER_LAYOUT = dict(
    q_layout="pre256", kv_stride=D_QK, dkv_acc=True, dkv_dst_map="none"
)
# rows that default to the trainer layout (explicit layout options still override)
ROW_LAYOUTS = {
    "packed_glm_a": TRAINER_LAYOUT,
    "packed_glm_b": TRAINER_LAYOUT,
    "packed_glm_a_s704": {**TRAINER_LAYOUT, "kv_stride": 704},
    "tail_2123x67923": TRAINER_LAYOUT,
    "doc_16231": TRAINER_LAYOUT,
    "doc_4095": TRAINER_LAYOUT,
    "doc_4097": TRAINER_LAYOUT,
    "packed_glm_a_dstmap": {**TRAINER_LAYOUT, "dkv_dst_map": "glm_a"},
    "packed_glm_b_dstmap": {**TRAINER_LAYOUT, "dkv_dst_map": "glm_b"},
    "packed_glm_a_s704_dstmap": {
        **TRAINER_LAYOUT,
        "kv_stride": 704,
        "dkv_dst_map": "glm_a",
    },
    "packed_glm_b_s704_dstmap": {
        **TRAINER_LAYOUT,
        "kv_stride": 704,
        "dkv_dst_map": "glm_b",
    },
    "chunk_4096x268757": {**TRAINER_LAYOUT, "dkv_dst_map": "glm_a"},
    "chunk_3943x268757": {**TRAINER_LAYOUT, "dkv_dst_map": "glm_a"},
    "chunk_3884x267520": {**TRAINER_LAYOUT, "dkv_dst_map": "glm_b"},
}
# --preset: one GLM-5.2 row in the trainer layout
PRESETS = {
    name: dict(rows=[name], **ROW_LAYOUTS[name])
    for name in ("packed_glm_a", "packed_glm_b", "tail_2123x67923")
}

ACCURACY_CASES = {
    "iid_4k": dict(
        seq_q=[4096], seq_k=[4096], self_including=False, target_self_weight=None
    ),
    "peaked_053_4k": dict(
        seq_q=[4096], seq_k=[4096], self_including=True, target_self_weight=0.53
    ),
    "peaked_099_4k": dict(
        seq_q=[4096], seq_k=[4096], self_including=True, target_self_weight=0.99
    ),
}
SEED = 1701


def median_ms(fn, steps):
    times = bench_gpu_time(
        fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=True, cold_l2_cache=True
    )
    return float(statistics.median(times))


def gib(nbytes):
    return nbytes / 2**30


# ---------------------------------------------------------------------------
# Layouts (issue #5675)
# ---------------------------------------------------------------------------


def _fill_randn_bf16(view, gen, chunk_rows=4096):
    """Fill a (possibly strided) bf16 view with N(0,1) values drawn in fp32 row chunks."""
    for r0 in range(0, view.shape[0], chunk_rows):
        r1 = min(view.shape[0], r0 + chunk_rows)
        view[r0:r1].copy_(
            torch.randn(r1 - r0, *view.shape[1:], device=view.device, generator=gen)
        )


class Layout:
    """The trainer's packed / strided layouts of one problem.

    Built from the canonical contiguous ``make_inputs`` tensors by copying INTO
    the layout buffers (never-read filler from a separate generator), so every
    arm sees the same values whatever the layout.

    * ``q_layout="pre256"``: ``q_rope`` is the ``192:256`` channel slice of the
      pre-absorption query buffer ``q256 [T, 64, 256]`` (head stride 256, token
      stride 16384; no contiguous copy), ``q_latent`` stays the contiguous
      absorbed ``[T, 64, 512]`` and the KV rows are one packed ``kv_buf
      [Tkv, kv_stride]`` (latent = columns ``0:512``, rope = ``512:576``).
      ``"packed576"``: the contiguous per-component tensors (``kv_stride=704``
      alone still packs the KV rows).
    * ``kv_stride``: 576, or 704 with never-read filler in columns ``576:704``
      (a frozen indexer key stored alongside).
    * ``dkv_acc``: the caller provides a zeroed fp32 ``dkv_acc [Tkv, 576]`` and
      the step's dK/dV lands there (latent columns ``0:512``, rope ``512:576``);
      ``dkv_dst_map`` ``"identity"`` (an explicit arange map) or ``"perm"`` (a
      seed-derived permutation in which source rows 2i and 2i+1, i < 8, share
      one destination row) maps every key row to its destination row.
    """

    def __init__(
        self,
        inp,
        *,
        q_layout="packed576",
        kv_stride=D_QK,
        dkv_acc=False,
        dkv_dst_map="none",
        seed=SEED,
    ):
        kv_stride = int(kv_stride)
        if q_layout not in Q_LAYOUTS:
            raise ValueError(f"q_layout must be one of {Q_LAYOUTS}, got {q_layout!r}")
        if kv_stride not in KV_STRIDES:
            raise ValueError(f"kv_stride must be one of {KV_STRIDES}, got {kv_stride}")
        if dkv_dst_map not in DST_MAPS:
            raise ValueError(
                f"dkv_dst_map must be one of {DST_MAPS}, got {dkv_dst_map!r}"
            )
        if dkv_dst_map != "none" and not dkv_acc:
            raise ValueError("a dkv destination map needs the caller-owned dkv_acc")
        self.q_layout, self.kv_stride = q_layout, kv_stride
        self.dkv_acc, self.dkv_dst_map = bool(dkv_acc), dkv_dst_map
        self.trainer = (q_layout, kv_stride, self.dkv_acc, dkv_dst_map) != (
            "packed576",
            D_QK,
            False,
            "none",
        )
        self.total_q, self.total_k = inp.total_q, inp.total_k
        self.dkv_rows = (
            inp.total_k
        )  # rows of dkv_acc (fewer than the key rows with a recorded map)
        device = inp.q_latent.device
        self.q_latent, self.q_rope = inp.q_latent, inp.q_rope
        self.kv_latent, self.k_rope = inp.kv_latent, inp.k_rope
        self.q256 = self.kv_buf = self.dst_map = None
        filler = torch.Generator(device=device)
        filler.manual_seed(int(seed) * 7919 + 17)
        if q_layout == "pre256":
            self.q256 = torch.empty(
                inp.total_q, NUM_HEADS, 256, dtype=torch.bfloat16, device=device
            )
            _fill_randn_bf16(self.q256[:, :, : 256 - D_ROPE], filler)
            self.q256[:, :, 256 - D_ROPE :].copy_(inp.q_rope)
            self.q_rope = self.q256[:, :, 256 - D_ROPE :]
        if q_layout == "pre256" or kv_stride != D_QK:
            self.kv_buf = torch.empty(
                inp.total_k, kv_stride, dtype=torch.bfloat16, device=device
            )
            self.kv_buf[:, :D_LATENT].copy_(inp.kv_latent)
            self.kv_buf[:, D_LATENT:D_QK].copy_(inp.k_rope)
            if kv_stride > D_QK:
                _fill_randn_bf16(self.kv_buf[:, D_QK:], filler)
            self.kv_latent = self.kv_buf[:, :D_LATENT]
            self.k_rope = self.kv_buf[:, D_LATENT:D_QK]
        if dkv_dst_map == "identity":
            self.dst_map = torch.arange(inp.total_k, dtype=torch.int32, device=device)
        elif dkv_dst_map == "perm":
            gm = torch.Generator(device=device)
            gm.manual_seed(int(seed) * 31 + 5)
            perm = torch.randperm(inp.total_k, device=device, generator=gm)
            ndup = min(8, inp.total_k // 2)
            perm[1 : 2 * ndup : 2] = perm[0 : 2 * ndup : 2]
            self.dst_map = perm.to(torch.int32).contiguous()
        elif dkv_dst_map in GLM_DST_ROWS:
            seq_k, rows = GLM_DST_ROWS[dkv_dst_map]
            if sum(seq_k) != inp.total_k:
                raise ValueError(
                    f"dkv_dst_map {dkv_dst_map!r} is defined for {sum(seq_k)} key rows, this row has {inp.total_k}"
                )
            self.dst_map = torch.tensor(
                dst_map_rows(seq_k, rows), dtype=torch.int32, device=device
            )
            self.dkv_rows = rows

    def describe(self):
        return dict(
            q_layout=self.q_layout,
            kv_stride=self.kv_stride,
            dkv_acc=self.dkv_acc,
            dkv_dst_map=self.dkv_dst_map,
            trainer=self.trainer,
            q_rope_stride=list(self.q_rope.stride()),
            kv_latent_stride=list(self.kv_latent.stride()),
        )

    def new_dkv_acc(self):
        """The caller's zeroed fp32 dK/dV buffer.

        Allocated inside the timed step (the trainer allocates and zeroes it per
        step), so it counts in the step time and in the peak memory of every arm.
        """
        return torch.zeros(
            self.dkv_rows, D_QK, dtype=torch.float32, device=self.q_latent.device
        )

    def accumulate(self, dkv_acc, dkv_latent, dk_rope):
        """Glue of a kernel that produces its own natural-layout dK/dV: fp32-accumulate
        it into the caller's packed ``dkv_acc`` (latent columns ``0:512``, rope
        ``512:576``), through the destination map when given (``index_add_``, fp32
        atomics, duplicated destinations allowed) or row for row (a fused
        upcast-add, ``+= x.float()`` without the temporary)."""
        if self.dst_map is None:
            dkv_acc[:, :D_LATENT].add_(dkv_latent)
            dkv_acc[:, D_LATENT:D_QK].add_(dk_rope)
        else:
            dkv_acc[:, :D_LATENT].index_add_(0, self.dst_map, dkv_latent.float())
            dkv_acc[:, D_LATENT:D_QK].index_add_(0, self.dst_map, dk_rope.float())
        return dkv_acc

    def map_reference(self, ref):
        """Reference dK/dV rows -> the destination rows of ``dkv_acc`` (fp64
        ``index_add_`` through the map; duplicated destinations sum).  In place."""
        if self.dst_map is None:
            return ref
        idx = self.dst_map.long()
        for key in ("dkv_latent", "dk_rope"):
            t = ref[key]
            ref[key] = torch.zeros(
                (self.dkv_rows,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device
            ).index_add_(0, idx, t)
        return ref


def _dkv_views(dkv_acc):
    """Gradient entries of a packed fp32 ``dkv_acc``: the buffer and its column views."""
    return dict(
        dkv_acc=dkv_acc,
        dkv_latent=dkv_acc[:, :D_LATENT],
        dk_rope=dkv_acc[:, D_LATENT:D_QK],
    )


def resolve_layout(row, args):
    """Layout of one run: the row's own default (the GLM-5.2 rows are trainer-layout
    rows), then ``--preset``, then the explicit layout options."""
    layout = dict(ROW_LAYOUTS.get(row, DEFAULT_LAYOUT))
    if args.preset:
        layout.update({k: v for k, v in PRESETS[args.preset].items() if k != "rows"})
    for key in ("q_layout", "kv_stride", "dkv_dst_map"):
        value = getattr(args, key)
        if value is not None:
            layout[key] = value
    if args.dkv_acc:
        layout["dkv_acc"] = True
    return layout


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


class ArmCake:
    """The Cake kernels.

    Default layout: the prepared runner (``prepare_dsa_train``, direct launches
    into caller buffers).  Trainer layout: the public varlen entry
    ``dsa_sparse_attention_varlen(..., causal=True, dkv_acc=, dkv_dst_map=)``
    through autograd, fed the layout views as they are (head-stride-256
    ``q_rope`` slice, packed KV rows) and the caller's fp32 ``dkv_acc``
    allocated inside the timed forward (it is passed at forward time and saved
    for the backward); the backward is ``torch.autograd.grad`` of ``out`` with
    respect to ``q_latent`` and the ``[T, 64, 256]`` query buffer, so ``dq_rope``
    flows back into that buffer's gradient exactly as in the trainer.
    """

    name = "cake"

    def __init__(self, inp, layout):
        self.inp, self.layout = inp, layout
        device = inp.q_latent.device
        self.backward_available = cake_backend.generated_program_available(
            device, backward=True
        )
        self.runner = None
        if not layout.trainer:
            self.runner = cake_backend.prepare_dsa_train(
                inp.q_latent,
                inp.q_rope,
                inp.kv_latent,
                inp.k_rope,
                inp.idx_global,
                topk_length=inp.topk_length,
                dout=inp.dout if self.backward_available else None,
                softmax_scale=DEFAULT_SCALE,
                backward=self.backward_available,
            )
            return
        self.q_latent = layout.q_latent.detach().requires_grad_()
        if layout.q256 is not None:
            self.q_buf = layout.q256.detach().requires_grad_()
            self.q_rope = self.q_buf[:, :, 256 - D_ROPE :]  # slice of the leaf
        else:
            self.q_buf = layout.q_rope.detach().requires_grad_()
            self.q_rope = self.q_buf
        if layout.dkv_acc:  # dK/dV lands in the caller's buffer: no KV leaves
            self.kv_leaves = ()
            self.kv_latent, self.k_rope = layout.kv_latent, layout.k_rope
        elif layout.kv_buf is not None:
            kv_buf = layout.kv_buf.detach().requires_grad_()
            self.kv_leaves = (kv_buf,)
            self.kv_latent, self.k_rope = kv_buf[:, :D_LATENT], kv_buf[:, D_LATENT:D_QK]
        else:
            self.kv_leaves = (
                layout.kv_latent.detach().requires_grad_(),
                layout.k_rope.detach().requires_grad_(),
            )
            self.kv_latent, self.k_rope = self.kv_leaves

    def versions(self):
        if self.runner is not None:
            return dict(
                module=self.runner.module_name,
                abi=self.runner.abi,
                stages=list(self.runner.stages),
                entry="prepare_dsa_train",
            )
        module_name, record = cake_backend.record_for(self.inp.q_latent.device)
        stages = getattr(cake_backend, "registered_stages", lambda _m: ())(module_name)
        return dict(
            module=module_name,
            abi=cake_backend.record_abi(record),
            stages=list(stages),
            entry="dsa_sparse_attention_varlen",
            layout=self.layout.describe(),
        )

    def forward(self):
        if self.runner is not None:
            out, lse, _ = self.runner.forward()
            return dict(out=out, lse=lse)
        inp, layout = self.inp, self.layout
        dkv_acc = layout.new_dkv_acc() if layout.dkv_acc else None
        out, lse = dsa_sparse_attention_varlen(
            self.q_latent,
            self.q_rope,
            self.kv_latent,
            self.k_rope,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            softmax_scale=DEFAULT_SCALE,
            return_lse=True,
            dkv_acc=dkv_acc,
            dkv_dst_map=layout.dst_map,
        )
        return dict(out=out, lse=lse, dkv_acc=dkv_acc)

    def backward(self, st):
        if self.runner is not None:
            dq_latent, dq_rope, dkv_latent, dk_rope = self.runner.backward()
            return dict(
                dq_latent=dq_latent,
                dq_rope=dq_rope,
                dkv_latent=dkv_latent,
                dk_rope=dk_rope,
            )
        grads = torch.autograd.grad(
            st["out"],
            (self.q_latent, self.q_buf, *self.kv_leaves),
            self.inp.dout,
            retain_graph=True,
        )
        dq_latent, dq_buf = grads[0], grads[1]
        dq_rope = (
            dq_buf[:, :, 256 - D_ROPE :] if self.layout.q256 is not None else dq_buf
        )
        res = dict(dq_latent=dq_latent, dq_rope=dq_rope)
        if self.layout.dkv_acc:
            res.update(_dkv_views(st["dkv_acc"]))
        elif len(self.kv_leaves) == 1:
            res.update(
                dkv_latent=grads[2][:, :D_LATENT], dk_rope=grads[2][:, D_LATENT:D_QK]
            )
        else:
            res.update(dkv_latent=grads[2], dk_rope=grads[3])
        return res

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


class ArmFlashMLACudnn:
    """FlashMLA sparse forward (packed q / kv, global indices) + cuDNN frontend DSA backward.

    Both take one packed bf16 ``q [T, 64, 576]`` and ``kv [S, 576]``.  Default
    layout: packed once at construction.  Trainer layout: packed inside the
    timed step (the glue the trainer runs for these kernels): ``q = cat(latent,
    rope slice)``; the packed KV rows at stride 576 are consumed directly, at
    stride 704 copied to a contiguous ``[S, 576]``; cuDNN's bf16 ``dkv`` is
    upcast-accumulated into the caller's fp32 ``dkv_acc`` (``index_add_`` with a
    destination map, a fused upcast-add otherwise).
    """

    name = "flashmla_cudnn"

    def __init__(self, inp, layout):
        import flash_mla
        from cudnn import DSA

        self.inp, self.layout = inp, layout
        self.flash_mla = flash_mla
        self.DSA = DSA
        device = inp.q_latent.device
        if not layout.trainer:
            self.q = torch.cat([inp.q_latent, inp.q_rope], dim=-1)
            self.kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1)
            self.dq = torch.empty_like(self.q)
            self.dkv = torch.zeros_like(self.kv)
        self.sink = torch.full(
            (NUM_HEADS,), float("-inf"), dtype=torch.float32, device=device
        )
        # cuDNN: valid slots first + topk_length; sentinels replaced by 0 (ignored past topk_length)
        self.idx_cudnn = inp.idx_global.clamp_min(0).contiguous()

    def versions(self):
        import cudnn

        return dict(
            flash_mla=getattr(self.flash_mla, "__file__", "?"),
            cudnn_frontend=cudnn.__version__,
            torch_cudnn=torch.backends.cudnn.version(),
        )

    def _packed(self):
        if not self.layout.trainer:
            return self.q, self.kv
        lay = self.layout
        q = torch.cat([lay.q_latent, lay.q_rope], dim=-1)  # gathers the strided slice
        if lay.kv_buf is None:
            kv = torch.cat([lay.kv_latent, lay.k_rope], dim=-1)
        elif lay.kv_stride == D_QK:
            kv = lay.kv_buf  # the packed [S, 576] rows are this kernel's kv layout
        else:
            kv = lay.kv_buf[:, :D_QK].contiguous()  # row stride 704 -> [S, 576]
        return q, kv

    def forward(self):
        q, kv = self._packed()
        out, max_logits, lse = self.flash_mla.flash_mla_sparse_fwd(
            q,
            kv.unsqueeze(1),
            self.inp.idx_global.unsqueeze(1),
            DEFAULT_SCALE,
            D_LATENT,
        )
        return dict(q=q, kv=kv, out=out, lse=lse, max_logits=max_logits)

    def backward(self, st):
        if not self.layout.trainer:
            dq, dkv = self.dq, self.dkv
            dkv.zero_()
        else:
            dq, dkv = torch.empty_like(st["q"]), torch.zeros_like(st["kv"])  # bf16
        self.DSA.sparse_attention_backward_wrapper(
            st["q"],
            st["kv"],
            st["out"],
            self.inp.dout,
            st["lse"],
            self.sink,
            self.idx_cudnn,
            softmax_scale=DEFAULT_SCALE,
            topk_length=self.inp.topk_length,
            dq=dq,
            dkv=dkv,
        )
        res = dict(dq_latent=dq[..., :D_LATENT], dq_rope=dq[..., D_LATENT:])
        if self.layout.dkv_acc:
            dkv_acc = self.layout.accumulate(
                self.layout.new_dkv_acc(), dkv[:, :D_LATENT], dkv[:, D_LATENT:]
            )
            res.update(_dkv_views(dkv_acc))
        else:
            res.update(dkv_latent=dkv[:, :D_LATENT], dk_rope=dkv[:, D_LATENT:])
        return res

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


class ArmFA4:
    """FA sparse-MLA kernels: varlen entry with per-document gather indices, recompute-P backward.

    They take contiguous per-component tensors.  Default layout: the inputs are
    the autograd leaves (set once).  Trainer layout: the contiguous copies the
    kernels need -- ``q_rope [T, 64, 64]`` from the ``[T, 64, 256]`` slice,
    ``k_rope`` / ``kv_latent [S, 1, d]`` from the packed rows -- are made inside
    the timed forward (the glue the trainer runs for these kernels) and kept
    for the backward like any saved activation; the bf16 dK/dV they return are
    upcast-accumulated into the two column ranges of the caller's fp32
    ``dkv_acc``.
    """

    name = "fa4"

    def __init__(self, inp, layout, token_chunk=4096):
        from flash_attn.cute import flash_attn_varlen_func

        self.inp, self.layout = inp, layout
        self.fn = flash_attn_varlen_func
        self.token_chunk = token_chunk
        if not layout.trainer:
            self.q_rope = inp.q_rope.detach().requires_grad_()
            self.q_latent = inp.q_latent.detach().requires_grad_()
            self.k_rope = inp.k_rope.detach().unsqueeze(1).contiguous().requires_grad_()
            self.kv_latent = (
                inp.kv_latent.detach().unsqueeze(1).contiguous().requires_grad_()
            )

    def versions(self):
        import cutlass
        import flash_attn.cute as fc

        return dict(cutlass_dsl=cutlass.__version__, fa_file=fc.__file__)

    def _leaves(self):
        if not self.layout.trainer:
            return (self.q_rope, self.k_rope, self.kv_latent, self.q_latent)
        lay = self.layout
        return (
            lay.q_rope.detach().contiguous().requires_grad_(),
            lay.k_rope.detach().unsqueeze(1).contiguous().requires_grad_(),
            lay.kv_latent.detach().unsqueeze(1).contiguous().requires_grad_(),
            lay.q_latent.detach().requires_grad_(),
        )

    def forward(self):
        inp = self.inp
        leaves = self._leaves()
        q_rope, k_rope, kv_latent, q_latent = leaves
        out, lse = self.fn(
            q_rope,
            k_rope,
            kv_latent,
            qv=q_latent,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=DEFAULT_SCALE,
            causal=True,
            gather_kv_indices=inp.idx_local,
            pack_gqa=True,
            gather_bwd_recompute_p=True,
            gather_bwd_token_chunk=self.token_chunk,
            return_lse=True,
        )
        st = dict(out=out, lse=lse)
        if self.layout.trainer:
            st["leaves"] = leaves  # the copies, saved for the backward
        return st

    def backward(self, st):
        leaves = st.get("leaves") or (
            self.q_rope,
            self.k_rope,
            self.kv_latent,
            self.q_latent,
        )
        dq_rope, dk_rope, dkv_latent, dq_latent = torch.autograd.grad(
            st["out"], leaves, self.inp.dout, retain_graph=True
        )
        res = dict(dq_latent=dq_latent, dq_rope=dq_rope)
        if self.layout.dkv_acc:
            dkv_acc = self.layout.new_dkv_acc()
            res.update(
                _dkv_views(
                    self.layout.accumulate(dkv_acc, dkv_latent[:, 0], dk_rope[:, 0])
                )
            )
        else:
            res.update(dkv_latent=dkv_latent[:, 0], dk_rope=dk_rope[:, 0])
        return res

    def outputs(self, st):
        lse = st["lse"]
        if lse.shape[0] != self.inp.total_q:  # (nheads, total_q) layout
            lse = lse.transpose(0, 1)
        return dict(out=st["out"], lse=lse)


ARMS = {
    ArmCake.name: ArmCake,
    ArmFlashMLACudnn.name: ArmFlashMLACudnn,
    ArmFA4.name: ArmFA4,
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_perf(arm, steps):
    """Median forward, backward and step milliseconds plus the peak memory above the inputs.

    ``dkv_acc`` (trainer layout) is allocated inside the timed step of every arm,
    so the peak counts it for every arm alike; each arm additionally holds what
    its own path materialises (the baselines their bf16 dK/dV, the contiguous
    copies and the fp32 upcasts of the ``index_add_`` glue).
    """
    st = arm.forward()
    torch.cuda.synchronize()
    fwd_ms = median_ms(arm.forward, steps)
    bwd_ms = None
    step_ms = None
    if not isinstance(arm, ArmCake) or arm.backward_available:
        bwd_ms = median_ms(lambda: arm.backward(st), steps)

        def full():
            s = arm.forward()
            arm.backward(s)

        step_ms = median_ms(full, steps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    s = arm.forward()
    if bwd_ms is not None:
        arm.backward(s)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return dict(
        fwd_ms=fwd_ms, bwd_ms=bwd_ms, step_ms=step_ms, peak_above_inputs_gib=gib(peak)
    )


def measure_accuracy(arm, inp, layout):
    st = arm.forward()
    grads = None
    try:
        grads = arm.backward(st)
    except NotImplementedError as exc:
        grads = dict(error=str(exc))
    torch.cuda.synchronize()
    ref = reference_fp64(
        inp.q_latent,
        inp.q_rope,
        inp.kv_latent,
        inp.k_rope,
        inp.idx_global,
        dout=inp.dout,
        own_key=inp.own_key,
    )
    layout.map_reference(ref)  # dK/dV in the destination rows when a map is used
    outs = arm.outputs(st)
    valid = torch.isfinite(ref["lse"])
    record = dict(
        self_weight=ref["self_weight"],
        out_rel_l2=rel_l2(outs["out"], ref["out"]),
        lse_max_abs=float((outs["lse"].double()[valid] - ref["lse"][valid]).abs().max())
        if valid.any()
        else 0.0,
        layout=layout.describe(),
    )
    if "error" in grads:
        record["backward"] = grads["error"]
    else:
        for name in ("dq_latent", "dq_rope", "dkv_latent", "dk_rope"):
            record[f"{name}_rel_l2"] = rel_l2(grads[name], ref[name])
        record["dq_latent_row_p99"] = float(
            rel_l2_rows(grads["dq_latent"], ref["dq_latent"]).quantile(0.99)
        )
        record["nan"] = bool(
            any(
                torch.isnan(g).any().item()
                for g in grads.values()
                if torch.is_tensor(g)
            )
        )
        if layout.dst_map is not None and "dkv_acc" in grads:
            hit = torch.zeros(
                layout.dkv_rows, dtype=torch.bool, device=layout.dst_map.device
            )
            hit[layout.dst_map.long()] = True
            record["dst_rows_unused"] = int((~hit).sum().item())
            record["unused_dst_rows_zero"] = bool(
                (grads["dkv_acc"][~hit] == 0).all().item()
            )
    return record


def _arm_or_error(cls, inp, layout):
    try:
        return cls(inp, layout), None
    except (
        Exception
    ) as exc:  # optional dependency missing or arm unavailable on this device
        return None, f"{type(exc).__name__}: {exc}"


def run_perf(args, results):
    steps_for = lambda inp: max(
        args.min_steps,
        # the largest problems (doc_131072 is exactly 2**34 query x key pairs) take --steps-128k
        args.steps if inp.total_q * inp.total_k < 2**34 else args.steps_128k,
    )
    for row in args.rows:
        seq_q, seq_k = ROWS[row]
        inp = make_inputs(
            seq_q, seq_k, seed=SEED, topk=DEFAULT_TOPK, device=args.device
        )
        layout = Layout(inp, **resolve_layout(row, args))
        entry = dict(
            row=row,
            total_q=inp.total_q,
            total_k=inp.total_k,
            num_docs=len(seq_q),
            cu_seqlens_q=inp.cu_seqlens_q.tolist(),
            cu_seqlens_k=inp.cu_seqlens_k.tolist(),
            inputs_gib=gib(inp.bytes_inputs()),
            layout=layout.describe(),
            arms={},
        )
        if layout.trainer:
            print(f"{row:22s} layout {json.dumps(layout.describe())}", flush=True)
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, layout)
            if arm is None:
                entry["arms"][name] = dict(error=error)
                print(f"{row:22s} {name:15s} unavailable: {error}", flush=True)
                continue
            try:
                perf = measure_perf(arm, steps_for(inp))
                perf["versions"] = arm.versions()
            except Exception as exc:
                perf = dict(
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            entry["arms"][name] = perf
            fmt = lambda v: "   n/a" if v is None else f"{v:8.3f}"
            if "error" in perf:
                print(f"{row:22s} {name:15s} failed: {perf['error']}", flush=True)
            else:
                print(
                    f"{row:22s} {name:15s} fwd {fmt(perf['fwd_ms'])} ms  bwd {fmt(perf['bwd_ms'])} ms  "
                    f"step {fmt(perf['step_ms'])} ms  peak {perf['peak_above_inputs_gib']:.2f} GiB",
                    flush=True,
                )
            del arm
            torch.cuda.empty_cache()
        results["rows"].append(entry)
        del inp, layout
        torch.cuda.empty_cache()


def run_accuracy(args, results):
    for case, spec in ACCURACY_CASES.items():
        beta = 0.0
        if spec["target_self_weight"] is not None:
            beta = calibrate_beta(
                spec["target_self_weight"], seed=SEED, device=args.device
            )
        inp = make_inputs(
            spec["seq_q"],
            spec["seq_k"],
            seed=SEED,
            topk=DEFAULT_TOPK,
            self_including=spec["self_including"],
            beta=beta,
            device=args.device,
        )
        layout = Layout(inp, **resolve_layout(None, args))
        entry = dict(case=case, beta=beta, layout=layout.describe(), arms={})
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, layout)
            if arm is None:
                entry["arms"][name] = dict(error=error)
                continue
            try:
                entry["arms"][name] = measure_accuracy(arm, inp, layout)
            except Exception as exc:
                entry["arms"][name] = dict(error=f"{type(exc).__name__}: {exc}")
            print(
                f"{case:16s} {name:15s} {json.dumps(entry['arms'][name], default=str)}",
                flush=True,
            )
            del arm
            torch.cuda.empty_cache()
        results["accuracy"].append(entry)
        del inp, layout
        torch.cuda.empty_cache()


def _host_us_per_call(fn, calls):
    """Wall-clock microseconds per call of ``fn`` (GPU asynchronous; results dropped as they come)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    return (time.perf_counter() - t0) / calls * 1e6


def measure_host_path(inp, *, calls, rounds, kernel_steps):
    """Host microseconds per call of the eager entry points with the binding cache off / on.

    Each round measures every entry point in both modes back to back (off
    first), so the two modes see the same process state; the medians over the
    rounds are reported.  Also checks that both modes give the same results
    (``out``, ``lse``, ``dq_*`` bitwise; ``dkv_*`` within the ``red.global``
    run-to-run spread) and reports the CUPTI kernel-only medians of both.
    """
    cache = cake_backend.BINDING_CACHE
    was_enabled = cache.enabled
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    backward_available = cake_backend.generated_program_available(
        inp.q_latent.device, backward=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        dsa_sparse_attention(*args)  # the experimental banner fires once per process
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    saved = cake_backend.forward(*args)
    torch.cuda.synchronize()

    def eager_forward():
        return cake_backend.forward(*args)

    def eager_backward():
        return cake_backend.backward(*args, saved[0], saved[2], saved[1], inp.dout)

    def public_forward():
        return dsa_sparse_attention(*leaves, inp.idx_global)

    graph_out = public_forward()

    def public_backward():  # autograd backward of one retained graph (the saved forward outputs are fixed)
        return torch.autograd.grad(graph_out, leaves, inp.dout, retain_graph=True)

    def autograd_step():
        out = dsa_sparse_attention(*leaves, inp.idx_global)
        return torch.autograd.grad(out, leaves, inp.dout)

    entry_points = [("forward", eager_forward), ("public_forward", public_forward)]
    if backward_available:
        entry_points += [
            ("backward", eager_backward),
            ("public_backward", public_backward),
            ("autograd_step", autograd_step),
        ]
    samples = {name: {"off": [], "on": []} for name, _ in entry_points}
    lookups = {
        name: [] for name, _ in entry_points
    }  # (hits, misses) of the measured cache-on calls
    try:
        for _ in range(rounds):
            for mode in ("off", "on"):
                cache.enabled = mode == "on"
                for name, fn in entry_points:
                    fn()  # the first call of a mode binds (a miss); measured calls follow
                    hits, misses = cache.hits, cache.misses
                    samples[name][mode].append(_host_us_per_call(fn, calls))
                    if mode == "on":
                        lookups[name].append((cache.hits - hits, cache.misses - misses))
        # same results from both paths
        cache.enabled = False
        off_fwd = eager_forward()
        off_bwd = eager_backward() if backward_available else None
        cache.enabled = True
        on_fwd = eager_forward()
        on_bwd = eager_backward() if backward_available else None
        torch.cuda.synchronize()
        same = dict(
            out=torch.equal(off_fwd[0], on_fwd[0]),
            lse=torch.equal(off_fwd[1], on_fwd[1]),
        )
        if backward_available:
            same.update(
                dq_latent=torch.equal(off_bwd[0], on_bwd[0]),
                dq_rope=torch.equal(off_bwd[1], on_bwd[1]),
                dkv_latent_rel_l2=rel_l2(off_bwd[2], on_bwd[2]),
                dk_rope_rel_l2=rel_l2(off_bwd[3], on_bwd[3]),
            )
        del off_fwd, off_bwd, on_fwd, on_bwd
        # kernel-only time of both paths (CUPTI, per-iteration GPU span)
        kernel_ms = {}
        try:
            for mode in ("off", "on"):
                cache.enabled = mode == "on"
                kernel_ms[mode] = dict(forward=median_ms(eager_forward, kernel_steps))
                if backward_available:
                    kernel_ms[mode]["backward"] = median_ms(
                        eager_backward, kernel_steps
                    )
                    kernel_ms[mode]["autograd_step"] = median_ms(
                        autograd_step, kernel_steps
                    )
        except Exception as exc:  # the host figures stand on their own when CUPTI tracing is unavailable
            kernel_ms["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cache.enabled = was_enabled
    host_us = {
        name: {
            mode: dict(
                median=float(statistics.median(v)),
                min=float(min(v)),
                rounds=[float(x) for x in v],
            )
            for mode, v in modes.items()
        }
        for name, modes in samples.items()
    }
    for name, hm in lookups.items():
        host_us[name]["on"]["lookups"] = dict(
            hits=sum(h for h, _ in hm), misses=sum(m for _, m in hm)
        )
    return dict(
        calls=calls,
        rounds=rounds,
        host_us=host_us,
        same_results=same,
        kernel_ms=kernel_ms,
        cache=dict(
            hits=cache.hits,
            misses=cache.misses,
            bindings=len(cache),
            owned_bytes=cache.owned_bytes,
        ),
    )


def run_host_path(args, results):
    for row in args.rows:
        seq_q, seq_k = ROWS[row]
        inp = make_inputs(
            seq_q, seq_k, seed=SEED, topk=DEFAULT_TOPK, device=args.device
        )
        try:
            entry = measure_host_path(
                inp,
                calls=args.host_calls,
                rounds=args.host_rounds,
                kernel_steps=max(args.min_steps, args.steps_128k),
            )
        except Exception as exc:
            entry = dict(
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
            )
            print(f"{row:22s} host path failed: {entry['error']}", flush=True)
        else:
            for name, modes in entry["host_us"].items():
                off, on = modes["off"]["median"], modes["on"]["median"]
                lk = modes["on"]["lookups"]
                print(
                    f"{row:22s} {name:16s} host us/call  cache off {off:8.1f}  cache on {on:8.1f}  ({off / on:5.2f}x)"
                    f"  lookups hit {lk['hits']} miss {lk['misses']}",
                    flush=True,
                )
            print(
                f"{row:22s} same results: {json.dumps(entry['same_results'], default=str)}",
                flush=True,
            )
            for mode, ms in entry["kernel_ms"].items():
                print(
                    f"{row:22s} kernel-only ms (cache {mode}): {json.dumps(ms)}",
                    flush=True,
                )
        entry["row"] = row
        results["host_path"].append(entry)
        del inp
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        default=None,
        choices=list(ROWS),
        help="rows to run (default: every row, or the --preset row)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="a GLM-5.2 row in the trainer layout (pre256 q_rope slice, packed KV rows, "
        "caller-owned fp32 dkv_acc); explicit layout options override its layout",
    )
    parser.add_argument(
        "--q-layout",
        choices=Q_LAYOUTS,
        default=None,
        help="packed576: contiguous q_latent / q_rope; pre256: q_rope = the 192:256 slice "
        "of a [T, 64, 256] buffer and packed KV rows (default: the row's own layout)",
    )
    parser.add_argument(
        "--kv-stride",
        type=int,
        choices=KV_STRIDES,
        default=None,
        help="row stride of the packed KV buffer (704 = 128 never-read filler columns)",
    )
    parser.add_argument(
        "--dkv-acc",
        action="store_true",
        help="accumulate dK/dV into a caller-owned fp32 [Tkv, 576] buffer allocated "
        "inside the timed step",
    )
    parser.add_argument(
        "--dkv-dst-map",
        choices=DST_MAPS,
        default=None,
        help="destination-row map of dkv_acc: identity (explicit arange), perm (a "
        "permutation with duplicated destinations) or glm_a / glm_b (the recorded "
        "trainer maps: 268,757 / 267,520 key rows onto 260,611 / 259,412 fp32 rows; "
        "only for the rows of that batch); needs --dkv-acc",
    )
    parser.add_argument("--arms", default="cake,flashmla_cudnn,fa4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--steps-128k", type=int, default=13, help="iterations for the largest problems"
    )
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument(
        "--host-us",
        action="store_true",
        help="host microseconds per call, binding cache off / on",
    )
    parser.add_argument("--host-calls", type=int, default=20)
    parser.add_argument("--host-rounds", type=int, default=3)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    if args.rows is None:
        args.rows = PRESETS[args.preset]["rows"] if args.preset else list(ROWS)
    # every layout the run will build (one per perf row, plus the accuracy layout) must carry dkv_acc when a
    # destination map is set -- Layout() raises otherwise, which would abort the run after the first rows
    planned = [(row, resolve_layout(row, args)) for row in args.rows]
    if args.accuracy:
        planned.append(("accuracy", resolve_layout(None, args)))
    for name, layout in planned:
        if layout.get("dkv_dst_map", "none") != "none" and not layout.get("dkv_acc"):
            parser.error(
                f"--dkv-dst-map needs dkv_acc, but the layout of {name!r} resolves without it "
                "(add --dkv-acc or pick rows / a preset that enable it)"
            )
    device = torch.device(args.device)
    if device.index is None:  # torch >= 2.13 requires an index here
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    results = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        layout_options=dict(
            preset=args.preset,
            q_layout=args.q_layout,
            kv_stride=args.kv_stride,
            dkv_acc=args.dkv_acc,
            dkv_dst_map=args.dkv_dst_map,
        ),
        rows=[],
        accuracy=[],
        host_path=[],
    )
    if not args.no_perf:
        run_perf(args, results)
    if args.accuracy:
        run_accuracy(args, results)
    if args.host_us:
        run_host_path(args, results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
