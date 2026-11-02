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
step) and of a prepared runner as wall-clock microseconds per call with the GPU
running asynchronously (``--host-rounds`` rounds of ``--host-calls`` calls
each, the order of the entry points rotated every round and recorded per
round), and checks that both paths produce the same results and the same
kernel-only time.

NoPE rows (GLM-5.3-Flash: shared latent K/V, zero RoPE, softmax scale 1/16, pooled
selection).  The ``nope_*`` rows (``--preset nope`` runs them all) are the zigzag-CP16
rank-0 packed batches at the observed token counts (``nope_glm53_a`` 16263 queries over
269564 keys, ``nope_glm53_b`` 14928 over 247211), their 4096-query backward chunks,
single documents of 4k .. 64k tokens, a selection-width sweep (``K_slots`` 128 .. 1024),
context-parallel tails and short rows; the indices follow the pool structure (``P``
four-token pools plus the open pool's causal tail, ``K_slots = round_up(4 P + 3, 128)``,
2176 at the model's ``P = 512``) or an iid causal top-k (``nope_glm53_a_iid``).  Arms:
``cake_nope`` (the native zero-RoPE entry), ``cake_rope_bridge`` (the previous Cake
version: the RoPE entry fed zero rope operands, with the zero fills, the index conversion
and the ``dkv_acc[:, :512]`` slice inside the timed step), ``fa4_nope`` (the FA sparse-MLA
kernels through autograd, ``k is v`` and an explicit scale), ``fa4_nope_a2`` (the same
kernels through their private forward / sparse-MLA backward entries with a preallocated
fp32 ``dv``; the faster FA form per row is the baseline) and ``flashmla_cudnn_nope`` (the
512 / 512 geometry when the packages accept it).  Every NoPE arm reports the training
forward, the no-grad forward, the whole-row and the chunked backward, the step, the
recompute step (two forwards + backward), the peak memory, the workspace and -- for the
Cake arms -- the planner's backward key passes (``key_passes``: the whole row and each
4096-query chunk); ``--accuracy-nope`` adds the FP64 comparison of ``out`` / ``dq`` / the
combined ``dkv``.  Protocol of the NoPE rows: ``dout`` is the recorded trainer layout
(``head_major``: strides ``[512, Tq * 512, 1]``; ``--nope-dout-layout contiguous`` is the
control), consumed by every arm as the same view, so a contiguous copy (FA's own, the Cake
backend's dense copy, the cuDNN arm's explicit one) is inside the timed backward; the fp32
``dkv`` accumulator of every arm is zero-filled inside the timed backward -- once per step
and once per recompute step, never inside a forward.  ``--check-nope-arms`` instantiates
every requested NoPE arm on a small row and runs each entry once before the perf rows (a
signature drift shows up there, not after the timings).

Usage::

    python benchmarks/bench_cake_dsa_train.py [--rows doc_4096 ...] [--arms cake,flashmla_cudnn,fa4]
        [--steps 20] [--accuracy] [--host-us] [--json out.json]
    python benchmarks/bench_cake_dsa_train.py --preset packed_glm_a --arms cake,flashmla_cudnn,fa4
    python benchmarks/bench_cake_dsa_train.py --rows doc_4096 --q-layout pre256 --kv-stride 704 \\
        --dkv-acc --dkv-dst-map perm --accuracy
    python benchmarks/bench_cake_dsa_train.py --preset nope --check-nope-arms \\
        --arms cake_nope,cake_rope_bridge,fa4_nope,fa4_nope_a2 --json nope.json
"""

import argparse
import dataclasses
import inspect
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
from flashinfer.experimental.cake_dsa_train import (  # noqa: E402
    cake_backend,
    cake_jit,
)
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_dsa_train_reference import (  # noqa: E402
    D_LATENT,
    D_QK,
    D_ROPE,
    DEFAULT_NOPE_SCALE,
    DEFAULT_NOPE_TOPK,
    DEFAULT_SCALE,
    DEFAULT_TOPK,
    NUM_HEADS,
    calibrate_beta,
    calibrate_beta_nope,
    compacted_twin,
    make_inputs,
    make_nope_inputs,
    nope_pools,
    reference_fp64,
    reference_fp64_nope,
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
# (T 16231, Tkv 268757, max_seqlen_q 2123, max_seqlen_k 67923); packed_glm_b is the
# second recorded batch (T 16172, Tkv 267520, max_seqlen_q 2027, max_seqlen_k 64853,
#   cu_seqlens_q = [0, 2027, 4043, 6070, 8086, 10113, 12129, 14156, 16172]
#   cu_seqlens_k = [0, 2027, 66880, 68907, 133760, 135787, 200640, 202667, 267520]).
GLM_A = (
    [1777, 1755, 2123, 2110, 2123, 2110, 2123, 2110],
    [1777, 56842, 2123, 67923, 2123, 67923, 2123, 67923],
)
GLM_B = (
    [2027, 2016, 2027, 2016, 2027, 2016, 2027, 2016],
    [2027, 64853, 2027, 64853, 2027, 64853, 2027, 64853],
)
ROWS["packed_glm_a"] = GLM_A
ROWS["packed_glm_b"] = GLM_B
ROWS["packed_glm_a_s704"] = GLM_A  # the same batch with KV row stride 704 (ROW_LAYOUTS)
ROWS["tail_2123x67923"] = ([2123], [67923])  # the short-query / long-key tail segment
ROWS["doc_16231"] = ([16231], [16231])  # one document at the recorded token count
ROWS["doc_4095"] = ([4095], [4095])  # tile / chunk boundary
ROWS["doc_4097"] = ([4097], [4097])
# the same rows with ``dout`` in the recorded trainer layout (head-major: strides ``[512, Tq * 512, 1]``, the view an
# output projection hands back); the contiguous rows above are their controls.  Issue #7041 ask 2: the rope backward
# reads this view in place (its delta stage carries the strides); a package that copies it pays the copy here.
ROWS["packed_glm_a_hm"] = GLM_A
ROWS["doc_16231_hm"] = ([16231], [16231])
ROW_DOUT_LAYOUT = {"packed_glm_a_hm": "head_major", "doc_16231_hm": "head_major"}


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


# ---------------------------------------------------------------------------
# NoPE rows: the GLM-5.3-Flash geometry (shared latent K/V, softmax scale 1/16, pooled
# selection, K_slots = round_up(4 * P + 3, 128))
# ---------------------------------------------------------------------------
# GLM-5.3-Flash packs with zigzag context parallelism (CP16): a document of length L is cut
# into 32 chunks of c = ceil(L / 32) tokens (the last one r = L - 31 c); rank 0 holds chunk 0
# (dense-causal inside its own window: seq_q = seq_k = c) and chunk 31 (seq_q = r over the
# whole prefix, seq_k = L), so Skv - Tq = 31 * sum(c).  The observed pairs 16263 / 269564 and
# 14928 / 247211 satisfy it; the per-document (c, r) splits below are one choice that
# reproduces them (the trainer's real cu_seqlens are not public).
GLM53_A_DOCS = ((1802, 1783), (2123, 2103), (2123, 2103), (2123, 2103))
GLM53_B_DOCS = ((1873, 1859), (1873, 1859), (1873, 1858), (1874, 1859))


def zigzag_rank0(docs):
    """Rank-0 segment lists ``(seq_q, seq_k)`` of zigzag-CP16 documents given as ``(c, r)``."""
    seq_q, seq_k = [], []
    for c, r in docs:
        length = 31 * c + r
        if -(-length // 32) != c or not 1 <= r <= c:
            raise ValueError(
                f"({c}, {r}) is not a zigzag split of a {length}-token document"
            )
        seq_q += [c, r]
        seq_k += [c, length]
    return seq_q, seq_k


GLM53_A = zigzag_rank0(GLM53_A_DOCS)  # Tq 16263, Skv 269564, 8 segments
GLM53_B = zigzag_rank0(GLM53_B_DOCS)  # Tq 14928, Skv 247211, 8 segments
NOPE_CHUNK = 4096  # the trainer's backward query-chunk policy


NOPE_DOUT_LAYOUTS = ("head_major", "contiguous")


NOPE_TWINS = (None, "stable")
NOPE_HOLE_PATTERNS = (None, "every_other")


def _nope_row(
    seq_q,
    seq_k,
    k_slots=DEFAULT_NOPE_TOPK,
    index="pooled",
    pools=None,
    dout_layout="head_major",
    twin=None,
    holes=None,
):
    """One NoPE row: document lengths, the physical slot width, the index structure
    (``pooled``: P four-token pools plus the causal tail of the open pool; ``iid``: random
    causal top-k; the hole layouts of issue #7041 -- ``pooled_fixed_tail``: the pooled draw
    with the open pool's tail at the fixed column ``4 P`` and ``-1`` between the last filled
    pool and that column, ``repro_gap``: the issue's rows, 1024 front keys, a gap to column
    2048, four tail keys, ``repro_compact_issue``: their descending-sorted twin), the pool
    capacity (default: the width's recorded capacity), the ``dout`` layout (default
    ``head_major``, the recorded trainer strides ``[512, Tq * 512, 1]``;
    ``--nope-dout-layout`` overrides it for every row), ``holes="every_other"``: every odd
    slot of the draw is invalidated after it is drawn (:func:`_plant_holes`; a hole row at
    any slot width, below the one-launch route's slot limit included), and ``twin="stable"``:
    the row's indices are the stable compaction of the draw (valid keys first in ascending
    slot order, the same keys per row: the baseline-side control of a hole row,
    ``compacted_twin``; applied after the planted holes)."""
    if pools is None and index in ("pooled", "pooled_fixed_tail"):
        pools = nope_pools(k_slots)
    if dout_layout not in NOPE_DOUT_LAYOUTS:
        raise ValueError(
            f"dout_layout must be one of {NOPE_DOUT_LAYOUTS}, got {dout_layout!r}"
        )
    if twin not in NOPE_TWINS:
        raise ValueError(f"twin must be one of {NOPE_TWINS}, got {twin!r}")
    if holes not in NOPE_HOLE_PATTERNS:
        raise ValueError(f"holes must be one of {NOPE_HOLE_PATTERNS}, got {holes!r}")
    return dict(
        seq_q=[int(x) for x in seq_q],
        seq_k=[int(x) for x in seq_k],
        k_slots=int(k_slots),
        index=index,
        pools=pools,
        dout_layout=dout_layout,
        twin=twin,
        holes=holes,
    )


def _plant_holes(inp, pattern):
    """``inp`` with holes planted in every row after the draw.  ``"every_other"`` invalidates the odd slots of
    the per-document and the global rows alike: the valid keys of the even slots stay where they are, so a
    position-preserving kernel scans each row to its last valid slot (about twice the valid count) while a
    compacting one scans the valid count.  The lengths become the new valid counts and the index kind gains the
    suffix ``+every_other``; ``compacted_twin`` of the result is the row's baseline-side control."""
    if pattern != "every_other":
        raise ValueError(f"unknown hole pattern {pattern!r}")
    local, glob = inp.idx_local.clone(), inp.idx_global.clone()
    local[:, 1::2] = -1
    glob[:, 1::2] = -1
    return dataclasses.replace(
        inp,
        idx_local=local,
        idx_global=glob,
        topk_length=(local >= 0).sum(-1).to(torch.int32).contiguous(),
        index=inp.index + "+every_other",
    )


# name -> NoPE row; the chunk rows are the trainer's 4096-query backward chunks of the packed
# batch over the full key row (``bwd_chunk``; the split 4096 chunks have nine segments).
NOPE_ROWS: dict[str, dict] = {
    "nope_glm53_a": _nope_row(*GLM53_A),
    "nope_glm53_b": _nope_row(*GLM53_B),
    "nope_glm53_a_iid": _nope_row(*GLM53_A, index="iid"),
    "nope_chunk_4096x269564": _nope_row(*bwd_chunk(*GLM53_A, 0, 4096)),
    "nope_chunk_3975x269564": _nope_row(*bwd_chunk(*GLM53_A, 12288, 16263)),
    "nope_chunk_1927x269564": _nope_row(*bwd_chunk(*GLM53_A, 14336, 16263)),
    "nope_tail_2048x65536": _nope_row([2048], [65536]),
    "nope_tail_4096x131072": _nope_row([4096], [131072]),
    "nope_tail_1024x16384_k512": _nope_row([1024], [16384], k_slots=512),
    "nope_short_1024_k128": _nope_row([1024], [1024], k_slots=128),
    "nope_short_4096_k512": _nope_row([4096], [4096], k_slots=512),
    # the two tables straddling the one-launch route's slot limit (524,288 and 1,048,576 slots; the tiny rows
    # above are 131,072 and 524,288)
    "nope_short_2048_k256": _nope_row([2048], [2048], k_slots=256),
    "nope_short_4096_k256": _nope_row([4096], [4096], k_slots=256),
    "nope_glm53_a_p256": _nope_row(*GLM53_A, k_slots=1152, pools=256),
}
for _n in (4096, 16384, 65536):
    NOPE_ROWS[f"nope_doc_{_n}"] = _nope_row([_n], [_n])
for _k in (128, 256, 512, 1024):
    NOPE_ROWS[f"nope_doc_16384_k{_k}"] = _nope_row([16384], [16384], k_slots=_k)
# Hole rows of issue #7041 (the rows above are valid-first: the no-regression set).  Every hole row has its stable
# twin (``_compact_stable`` / ``_compact``: the same keys per row, valid-first) as the baseline-side control: the
# gate is t_cand(row) / t_base(twin).  Selector assumptions: ``pooled_fixed_tail`` writes the ascending four-token
# pools from column 0 (512 pools x 4 tokens in columns 0..2047 at K_slots 2176), ``-1`` where a document has fewer
# pools, and the open pool's causal tail at column 2048 only when the position is not the last token of its pool
# (``pos % 4 != 3``); ``repro_gap`` is the issue's construction (position ``pos = seq_k - seq_q + i``: keys
# ``pos - 4 - j`` for j < 1024 in columns 0..1023, keys ``pos - j`` for j < 4 at columns 2048..2051, ``-1`` elsewhere).
# R1: the issue's 16k document.  R2: a 4-document 262k pack as context-parallel rank 0 sees it (each document a
# 2048-token head segment and a 2048-token tail of its 65536-token prefix).  R2': the recorded GLM-5.3 batch with the
# fixed-column tail.  R3: 264 documents of 992 tokens (528 segments of 31 queries: the pooled selector fills at most
# 248 pools per document, so three quarters of the slots are the gap to column 2048).
NOPE_ROWS["nope_repro_16k_gap"] = _nope_row([16384], [16384], index="repro_gap")
NOPE_ROWS["nope_repro_16k_compact_stable"] = _nope_row(
    [16384], [16384], index="repro_gap", twin="stable"
)
NOPE_ROWS["nope_repro_16k_compact_issue"] = _nope_row(
    [16384], [16384], index="repro_compact_issue"
)
NOPE_ROWS["nope_pack4_262k_rank0_fixedtail"] = _nope_row(
    [2048, 2048] * 4, [2048, 65536] * 4, index="pooled_fixed_tail"
)
NOPE_ROWS["nope_pack4_262k_rank0_fixedtail_compact"] = _nope_row(
    [2048, 2048] * 4, [2048, 65536] * 4, index="pooled_fixed_tail", twin="stable"
)
NOPE_ROWS["nope_glm53_a_fixedtail"] = _nope_row(*GLM53_A, index="pooled_fixed_tail")
NOPE_ROWS["nope_glm53_a_fixedtail_compact"] = _nope_row(
    *GLM53_A, index="pooled_fixed_tail", twin="stable"
)
NOPE_ROWS["nope_docs264x992_rank0_fixedtail"] = _nope_row(
    [31, 31] * 264, [31, 992] * 264, index="pooled_fixed_tail"
)
NOPE_ROWS["nope_docs264x992_rank0_fixedtail_compact"] = _nope_row(
    [31, 31] * 264, [31, 992] * 264, index="pooled_fixed_tail", twin="stable"
)
# A small hole row (131,072 slots: below the slot limit, the one-launch position-preserving route) and its stable
# twin: every odd slot of the pooled draw is -1, so the row scans about twice its valid keys position-preserving
# while the twin scans its valid count.  Graded against the previous version's own time (``--parity-rows``), not
# against the twin.
NOPE_ROWS["nope_short_1024_k128_holes"] = _nope_row(
    [1024], [1024], k_slots=128, holes="every_other"
)
NOPE_ROWS["nope_short_1024_k128_holes_compact_stable"] = _nope_row(
    [1024], [1024], k_slots=128, holes="every_other", twin="stable"
)
NOPE_HOLE_ROWS = tuple(
    name
    for name, spec in NOPE_ROWS.items()
    if (spec["index"] in ("pooled_fixed_tail", "repro_gap") or spec.get("holes"))
    and spec["twin"] is None
)
NOPE_ARMS_DEFAULT = (
    "cake_nope,cake_rope_bridge,fa4_nope,fa4_nope_a2,flashmla_cudnn_nope"
)

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
    "packed_glm_a_hm": TRAINER_LAYOUT,
    "doc_16231_hm": TRAINER_LAYOUT,
}
# --preset: one GLM-5.2 row in the trainer layout
PRESETS = {
    name: dict(rows=[name], **ROW_LAYOUTS[name])
    for name in ("packed_glm_a", "packed_glm_b", "tail_2123x67923")
}
PRESETS["nope"] = dict(rows=list(NOPE_ROWS))  # every NoPE row with the NoPE arms

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
                stages=list(self.runner.stages),
                entry="prepare_dsa_train",
            )
        module_name, _record = cake_backend.record_for(self.inp.q_latent.device)
        return dict(
            module=module_name,
            stages=list(cake_jit.registered_stages()),
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
# NoPE arms
# ---------------------------------------------------------------------------


def _glue_compacts():
    """Whether this package's torch glue can compact (``offset_gather_kv_indices(..., compact=True)``, issue #7041);
    an older package keeps the position-preserving rows (its own behaviour, what the baseline arm measures)."""
    return (
        "compact" in inspect.signature(cake_backend.offset_gather_kv_indices).parameters
    )


def _globalize(local, cu_q, cu_k):
    """Global rows and lengths of per-document rows through the torch glue, in the form the package's entries
    produce on device for a table of this size: the stable compaction with the valid counts where they run the
    compacting stage, else the position-preserving rows with the last valid slot + 1."""
    mode = getattr(cake_backend, "nope_varlen_index_mode", None)
    compact = _glue_compacts() and (
        mode is None or mode(int(local.shape[0]), int(local.shape[1])) == "index_prep"
    )
    kw = dict(compact=True) if compact else {}
    return cake_backend.offset_gather_kv_indices(
        local, cu_q, cu_k, causal=True, return_topk_length=True, **kw
    )


def _chunk_bounds(inp, chunk):
    """Query sub-ranges ``(q0, q1, cu_seqlens_q, cu_seqlens_k, num_segments)`` of the
    backward chunk policy (``bwd_chunk`` over the full key row), as device tensors."""
    device = inp.q.device
    bounds = []
    for q0 in range(0, inp.total_q, chunk):
        q1 = min(q0 + chunk, inp.total_q)
        seq_q, seq_k = bwd_chunk(inp.seq_q, inp.seq_k, q0, q1)
        cu_q = [0] + torch.tensor(seq_q).cumsum(0).tolist()
        cu_k = [0] + torch.tensor(seq_k).cumsum(0).tolist()
        bounds.append(
            (
                q0,
                q1,
                torch.tensor(cu_q, dtype=torch.int32, device=device),
                torch.tensor(cu_k, dtype=torch.int32, device=device),
                len(seq_q),
            )
        )
    return bounds


def _bridge_dkv_direct(device, num_queries, num_kv):
    """The shipped record's direct-accumulation decision for a bridge call, passed explicitly
    so the binding does not depend on the environment."""
    _, record = cake_backend.record_for(device)
    stages = cake_jit.registered_stages()
    policy = cake_backend.DkvDirectPolicy.from_record(record)
    return bool(
        policy is not None
        and cake_backend.record_direct_stages(record, stages)
        and policy.wants(num_queries, num_kv)
    )


def _planner_key_passes(inp, chunk, program=None):
    """The FlashInfer planner's backward key-pass decisions for a row (host side, no launch):
    the pass count of the whole-row backward, of each ``chunk``-query backward chunk, the
    policy's token chunk and the architecture rule it applied.  ``program=None`` is the RoPE
    record (the bridge arm), else the named program's record (the native arm)."""
    try:
        kw = {} if program is None else dict(program=program)
        _, record = cake_backend.record_for(inp.q.device, **kw)
        stages = tuple(
            cake_jit.registered_stages(program)
            if program is not None
            else cake_jit.registered_stages()
        )
        arch = cake_backend.arch_for(inp.q.device)
        plan = lambda T, segments: int(
            cake_backend.plan_key_passes(
                record,
                stages,
                T,
                inp.total_k,
                inp.topk,
                None,
                num_segments=segments,
                arch=arch,
                **kw,
            )
        )
        facts = dict(
            whole_row=plan(inp.total_q, inp.num_segments),
            chunks=[
                plan(q1 - q0, segments)
                for q0, q1, _cu_q, _cu_k, segments in _chunk_bounds(inp, chunk)
            ],
        )
        policy = cake_backend.KeyPassPolicy.from_record(record)
        if policy is not None:
            facts["token_chunk"] = int(policy.token_chunk(inp.topk))
            facts["rule"] = dict(arch=arch, **policy.rule_for(arch))
        return facts
    except Exception as exc:  # noqa: BLE001 -- the column is informational
        return dict(error=f"{type(exc).__name__}: {exc}")


class ArmCakeNope:
    """The native zero-RoPE entry (``dsa_sparse_attention_nope_varlen``) through autograd.

    The caller's fp32 ``dkv_acc [S, 512]`` is allocated in the forward (the entry binds it
    for the backward) and zero-filled inside the timed backward, once per step (where the
    FA arm zero-fills its fp32 ``dv``), and receives the combined dK + dV; ``dq`` is the
    autograd gradient of ``q``.  ``backward_chunks`` runs the eager backward over the
    4096-query chunks of the row into one accumulate buffer.
    """

    name = "cake_nope"

    def __init__(self, inp, chunk=NOPE_CHUNK):
        from flashinfer.dsa_sparse_attention import dsa_sparse_attention_nope_varlen
        from flashinfer.experimental.cake_dsa_train import cake_jit

        program = getattr(cake_jit, "NOPE_PROGRAM", None)
        if program is None or not cake_backend.generated_program_available(
            inp.q.device, backward=True, program=program
        ):
            raise RuntimeError("skipped: nope stems not installed")
        try:  # the record may be registered before the device libraries are built
            from flashinfer.experimental.cake_dsa_train import cake_build

            arch = cake_backend.arch_for(inp.q.device)
            installed = cake_build.libraries_by_name()
            missing = [
                stage
                for stage in cake_jit.registered_stages(program)
                if cake_jit.library_name(stage, arch, program) not in installed
            ]
        except Exception:  # noqa: BLE001 -- an older manifest API: let the first launch decide
            missing = []
        if missing:
            raise RuntimeError("skipped: nope stems not installed")
        self.program = program
        self.fn = dsa_sparse_attention_nope_varlen
        self.inp, self.chunk = inp, chunk
        self.q = inp.q.detach().requires_grad_()
        self.eager = None

    def versions(self):
        return dict(module=self.program, entry="dsa_sparse_attention_nope_varlen")

    def _call(self, q, kv, dkv_acc):
        inp = self.inp
        return self.fn(
            q,
            kv,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            softmax_scale=DEFAULT_NOPE_SCALE,
            return_lse=True,
            dkv_acc=dkv_acc,
        )

    def forward(self):
        dkv_acc = torch.empty(
            self.inp.total_k, D_LATENT, dtype=torch.float32, device=self.q.device
        )
        out, lse = self._call(self.q, self.inp.kv, dkv_acc)
        return dict(out=out, lse=lse, dkv_acc=dkv_acc)

    def forward_nograd(self):
        out, lse = self._call(self.inp.q, self.inp.kv, None)
        return dict(out=out, lse=lse)

    def backward(self, st):
        st[
            "dkv_acc"
        ].zero_()  # the accumulator's zero fill: once per step, inside the backward
        (dq,) = torch.autograd.grad(
            st["out"], (self.q,), self.inp.dout, retain_graph=True
        )
        return dict(dq=dq, dkv=st["dkv_acc"])

    def issue_step(self):
        """The issue's step (#7041): forward + ``autograd.grad`` to ``q`` and ``kv`` with a fresh ``dkv``."""
        kv = self._kv_leaf()
        out, _ = self._call(self.q, kv, None)
        return torch.autograd.grad(out, (self.q, kv), self.inp.dout)

    def issue_forward(self):
        with torch.no_grad():
            return self._call(self.inp.q, self.inp.kv, None)

    def _kv_leaf(self):
        leaf = getattr(self, "_kv_leaf_cache", None)
        if leaf is None:
            leaf = self._kv_leaf_cache = self.inp.kv.detach().requires_grad_()
        return leaf

    def key_passes(self):
        return _planner_key_passes(self.inp, self.chunk, program=self.program)

    def backward_chunks(self, st):
        """Eager backward over the query chunks into one accumulate buffer (``None`` when the
        eager NoPE entries are not available).  The eager forward and the per-chunk global
        rows and lengths are prepared once, on the first call: the timed region is the
        chunked backward alone, so that a package whose torch glue does more per call (the
        compacting glue of #7041) is compared on its kernels; the glue's own cost is the
        index-glue table."""
        fwd = getattr(cake_backend, "forward_nope", None)
        bwd = getattr(cake_backend, "backward_nope", None)
        if fwd is None or bwd is None:
            return None
        inp = self.inp
        if self.eager is None:
            idx, length = _globalize(inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k)
            out, lse, o_lo = fwd(
                inp.q, inp.kv, idx, topk_length=length, softmax_scale=DEFAULT_NOPE_SCALE
            )
            chunks = []
            for q0, q1, cu_q, cu_k, segments in _chunk_bounds(inp, self.chunk):
                idx_c, len_c = _globalize(inp.idx_local[q0:q1], cu_q, cu_k)
                chunks.append((q0, q1, idx_c, len_c, segments))
            self.eager = dict(out=out, lse=lse, o_lo=o_lo, chunks=chunks)
        e = self.eager
        dkv_acc = torch.zeros(
            inp.total_k, D_LATENT, dtype=torch.float32, device=inp.q.device
        )
        for q0, q1, idx_c, len_c, segments in e["chunks"]:
            bwd(
                inp.q[q0:q1],
                inp.kv,
                idx_c,
                e["out"][q0:q1],
                e["o_lo"][q0:q1],
                e["lse"][q0:q1],
                inp.dout[q0:q1],
                topk_length=len_c,
                softmax_scale=DEFAULT_NOPE_SCALE,
                dkv_acc=dkv_acc,
                num_segments=segments,
            )
        return dict(dkv=dkv_acc)

    def workspace(self):
        size = getattr(cake_backend, "dsa_nope_train_workspace_size", None)
        if size is None:
            return None
        inp = self.inp
        return int(
            size(
                inp.total_q,
                inp.total_k,
                inp.topk,
                inp.q.device,
                backward=True,
                num_segments=inp.num_segments,
            )
        )

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


class ArmCakeRopeBridge:
    """The previous Cake version: the RoPE entry driven through a zero-RoPE bridge.

    Inside the timed forward: ``q_rope0 [T, 64, 64]`` and ``k_rope0 [S, 64]`` zero fills, the
    varlen index conversion of the public entry, ``softmax_scale = 1/16``; the caller's fp32
    ``dkv_acc [S, 576]`` is allocated in the forward and zero-filled inside the timed backward
    (once per step); the backward is the autograd gradient of ``q`` (the dead ``dq_rope``
    output is still written) and ``dkv = dkv_acc[:, :512]`` (a view; a consumer that needs
    contiguous rows pays a copy).  ``dkv_direct`` is passed as the record's own
    size rule evaluates.  ``backward_chunks`` uses the eager entries over the 4096-query
    chunks.
    """

    name = "cake_rope_bridge"

    def __init__(self, inp, chunk=NOPE_CHUNK):
        self.inp, self.chunk = inp, chunk
        self.q = inp.q.detach().requires_grad_()
        self.direct = _bridge_dkv_direct(inp.q.device, inp.total_q, inp.total_k)
        self.eager = None

    def versions(self):
        module_name, _record = cake_backend.record_for(self.inp.q.device)
        return dict(
            module=module_name,
            entry="dsa_sparse_attention_varlen (zero-RoPE bridge)",
            dkv_direct=self.direct,
        )

    def _zeros(self):
        inp = self.inp
        device = inp.q.device
        q_rope0 = torch.zeros(
            inp.total_q, NUM_HEADS, D_ROPE, dtype=torch.bfloat16, device=device
        )
        k_rope0 = torch.zeros(inp.total_k, D_ROPE, dtype=torch.bfloat16, device=device)
        return q_rope0, k_rope0

    def forward(self):
        inp = self.inp
        q_rope0, k_rope0 = self._zeros()
        dkv_acc = torch.empty(
            inp.total_k, D_QK, dtype=torch.float32, device=inp.q.device
        )
        out, lse = dsa_sparse_attention_varlen(
            self.q,
            q_rope0,
            inp.kv,
            k_rope0,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            softmax_scale=DEFAULT_NOPE_SCALE,
            return_lse=True,
            dkv_acc=dkv_acc,
            dkv_direct=self.direct,
        )
        return dict(out=out, lse=lse, dkv_acc=dkv_acc)

    def forward_nograd(self):
        return (
            self.forward()
        )  # the same kernels: the forward always writes its o_lo residual

    def backward(self, st):
        st[
            "dkv_acc"
        ].zero_()  # the accumulator's zero fill: once per step, inside the backward
        (dq,) = torch.autograd.grad(
            st["out"], (self.q,), self.inp.dout, retain_graph=True
        )
        return dict(dq=dq, dkv=st["dkv_acc"][:, :D_LATENT])

    def key_passes(self):
        return _planner_key_passes(self.inp, self.chunk, program=None)

    def backward_chunks(self, st):
        inp = self.inp
        if self.eager is None:
            q_rope0, k_rope0 = self._zeros()
            idx, length = cake_backend.offset_gather_kv_indices(
                inp.idx_local,
                inp.cu_seqlens_q,
                inp.cu_seqlens_k,
                causal=True,
                return_topk_length=True,
            )
            out, lse, o_lo = cake_backend.forward(
                inp.q,
                q_rope0,
                inp.kv,
                k_rope0,
                idx,
                topk_length=length,
                softmax_scale=DEFAULT_NOPE_SCALE,
            )
            bounds = [
                (
                    q0,
                    q1,
                    cu_q,
                    cu_k,
                    segments,
                    _bridge_dkv_direct(inp.q.device, q1 - q0, inp.total_k),
                )
                for q0, q1, cu_q, cu_k, segments in _chunk_bounds(inp, self.chunk)
            ]
            self.eager = dict(
                out=out,
                lse=lse,
                o_lo=o_lo,
                q_rope0=q_rope0,
                k_rope0=k_rope0,
                bounds=bounds,
            )
        e = self.eager
        dkv_acc = torch.zeros(
            inp.total_k, D_QK, dtype=torch.float32, device=inp.q.device
        )
        for q0, q1, cu_q, cu_k, segments, direct in e["bounds"]:
            idx_c, len_c = cake_backend.offset_gather_kv_indices(
                inp.idx_local[q0:q1], cu_q, cu_k, causal=True, return_topk_length=True
            )
            cake_backend.backward(
                inp.q[q0:q1],
                e["q_rope0"][q0:q1],
                inp.kv,
                e["k_rope0"],
                idx_c,
                e["out"][q0:q1],
                e["o_lo"][q0:q1],
                e["lse"][q0:q1],
                inp.dout[q0:q1],
                topk_length=len_c,
                softmax_scale=DEFAULT_NOPE_SCALE,
                dkv_acc=dkv_acc,
                num_segments=segments,
                dkv_direct=direct,
            )
        return dict(dkv=dkv_acc[:, :D_LATENT])

    def workspace(self):
        inp = self.inp
        return int(
            cake_backend.dsa_train_workspace_size(
                inp.total_q,
                inp.total_k,
                inp.topk,
                inp.q.device,
                backward=True,
                num_segments=inp.num_segments,
                dkv_acc=True,
                dkv_direct=self.direct,
            )
        )

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


class ArmFA4Nope:
    """FA sparse-MLA kernels on the NoPE geometry: ``k is v`` (the shared latent, so the
    kernels take the latent query as ``qv``), explicit ``softmax_scale = 1/16``,
    per-document gather indices, recompute-P backward in 4096-token chunks.  The bf16
    ``dkv`` they return is upcast-added into a fresh fp32 ``[S, 512]`` accumulator inside
    the timed backward (the trainer's accumulation); the gather width must be a multiple of
    128, so other widths are ``-1``-padded inside the timed forward (correctness widths
    only; every performance row is a multiple of 128).
    """

    name = "fa4_nope"

    def __init__(self, inp, token_chunk=NOPE_CHUNK):
        from flash_attn.cute import flash_attn_varlen_func

        self.fn = flash_attn_varlen_func
        self.inp, self.token_chunk = inp, token_chunk
        self.q = inp.q.detach().requires_grad_()
        self.kv3 = inp.kv.detach().unsqueeze(1).requires_grad_()  # [S, 1, 512]
        self.q_d, self.kv3_d = inp.q.detach(), inp.kv.detach().unsqueeze(1)

    def versions(self):
        import cutlass
        import flash_attn.cute as fc

        return dict(cutlass_dsl=cutlass.__version__, fa_file=fc.__file__)

    def _indices(self):
        idx = self.inp.idx_local
        pad = (-idx.shape[1]) % 128
        if pad:
            idx = torch.cat([idx, idx.new_full((idx.shape[0], pad), -1)], dim=1)
        return idx

    def _call(self, q, kv3):
        inp = self.inp
        return self.fn(
            q,
            kv3,
            kv3,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=DEFAULT_NOPE_SCALE,
            causal=True,
            gather_kv_indices=self._indices(),
            pack_gqa=True,
            gather_bwd_recompute_p=True,
            gather_bwd_token_chunk=self.token_chunk,
            return_lse=True,
        )

    def forward(self):
        out, lse = self._call(self.q, self.kv3)
        return dict(out=out, lse=lse)

    def forward_nograd(self):
        # the inference configuration is selected by the inputs' requires_grad flags
        out, lse = self._call(self.q_d, self.kv3_d)
        return dict(out=out, lse=lse)

    def backward(self, st):
        dq, dkv = torch.autograd.grad(
            st["out"], (self.q, self.kv3), self.inp.dout, retain_graph=True
        )
        acc = torch.zeros(
            self.inp.total_k, D_LATENT, dtype=torch.float32, device=dq.device
        )
        acc.add_(dkv[:, 0])
        return dict(dq=dq, dkv=acc)

    def backward_chunks(self, st):
        return None  # the backward chunks internally (gather_bwd_token_chunk)

    def workspace(self):
        return None

    def outputs(self, st):
        lse = st["lse"]
        if lse.shape[0] != self.inp.total_q:
            lse = lse.transpose(0, 1)
        return dict(out=st["out"], lse=lse)


class ArmFA4NopeA2(ArmFA4Nope):
    """The second FA form: the private ``_flash_attn_fwd`` (training configuration through
    grad-requiring inputs; returns the ``o_lo`` residual) and ``_flash_attn_bwd_sparse_mla``
    with a preallocated fp32 ``dv [S, 1, 512]`` zero-filled inside the timed backward (the
    kernels' own fp32 accumulation, no bf16 round trip; ``recompute_p``, 4096-token chunks).
    A strided ``dout`` is made contiguous by the entry itself, inside the timed call.  The
    faster of ``fa4_nope`` and ``fa4_nope_a2`` per row is the FA baseline."""

    name = "fa4_nope_a2"

    def __init__(self, inp, token_chunk=NOPE_CHUNK):
        super().__init__(inp, token_chunk)
        from flash_attn.cute.interface import (
            _flash_attn_bwd_sparse_mla,
            _flash_attn_fwd,
        )

        self.ffwd, self.fbwd = _flash_attn_fwd, _flash_attn_bwd_sparse_mla

    def versions(self):
        return dict(
            super().versions(),
            form="_flash_attn_fwd(None, None, kv3, qv=q) + _flash_attn_bwd_sparse_mla(dv=zeros(S, 1, 512, f32), o_lo)",
        )

    def _fwd(self, q, kv3):
        inp = self.inp
        out, lse, _p, _row_max, o_lo = self.ffwd(
            None,
            None,
            kv3,
            qv=q,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=DEFAULT_NOPE_SCALE,
            causal=True,
            pack_gqa=True,
            return_lse=True,
            gather_kv_indices=self._indices(),
            gather_bwd_recompute_p=True,
        )
        return dict(out=out, lse=lse, o_lo=o_lo)

    def forward(self):
        return self._fwd(self.q, self.kv3)

    def forward_nograd(self):
        return self._fwd(self.q_d, self.kv3_d)

    def backward(self, st):
        inp = self.inp
        dv = torch.zeros(
            inp.total_k, 1, D_LATENT, dtype=torch.float32, device=self.q_d.device
        )
        _dq, _dk, dv, dqv, _dqk = self.fbwd(
            None,
            None,
            self.kv3_d,
            self.q_d,
            st["out"],
            inp.dout,
            st["lse"],
            None,
            None,
            self._indices(),
            softmax_scale=DEFAULT_NOPE_SCALE,
            causal=True,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            recompute_p=True,
            token_chunk=self.token_chunk,
            o_lo=st["o_lo"],
            dv=dv,
        )
        return dict(dq=dqv, dkv=dv[:, 0])


class ArmFlashMLACudnnNope:
    """FlashMLA sparse forward + cuDNN frontend DSA backward on the 512 / 512 geometry
    (``d_qk = d_v = 512``, global indices); cuDNN's bf16 ``dkv`` is upcast-added into a fresh
    fp32 accumulator inside the timed backward, and a strided ``dout`` is copied to a
    contiguous array there too (a counted cost).  Unavailable packages or a refused geometry
    are recorded as the arm's error."""

    name = "flashmla_cudnn_nope"

    def __init__(self, inp):
        import flash_mla
        from cudnn import DSA

        if getattr(inp, "has_holes", False):
            raise RuntimeError(
                "skipped: hole rows (cuDNN DSA takes dense top-k rows; -1 slots before valid keys are undefined)"
            )
        self.inp = inp
        self.flash_mla, self.DSA = flash_mla, DSA
        device = inp.q.device
        self.sink = torch.full(
            (NUM_HEADS,), float("-inf"), dtype=torch.float32, device=device
        )
        self.idx_cudnn = inp.idx_global.clamp_min(0).contiguous()

    def versions(self):
        import cudnn

        return dict(
            flash_mla=getattr(self.flash_mla, "__file__", "?"),
            cudnn_frontend=cudnn.__version__,
        )

    def forward(self):
        inp = self.inp
        out, max_logits, lse = self.flash_mla.flash_mla_sparse_fwd(
            inp.q,
            inp.kv.unsqueeze(1),
            inp.idx_global.unsqueeze(1),
            DEFAULT_NOPE_SCALE,
            D_LATENT,
        )
        return dict(out=out, lse=lse, max_logits=max_logits)

    def backward(self, st):
        inp = self.inp
        dq, dkv = torch.empty_like(inp.q), torch.zeros_like(inp.kv)
        dout = inp.dout if inp.dout.is_contiguous() else inp.dout.contiguous()
        self.DSA.sparse_attention_backward_wrapper(
            inp.q,
            inp.kv,
            st["out"],
            dout,
            st["lse"],
            self.sink,
            self.idx_cudnn,
            softmax_scale=DEFAULT_NOPE_SCALE,
            topk_length=inp.topk_length,
            dq=dq,
            dkv=dkv,
        )
        acc = torch.zeros(inp.total_k, D_LATENT, dtype=torch.float32, device=dq.device)
        acc.add_(dkv)
        return dict(dq=dq, dkv=acc)

    def backward_chunks(self, st):
        return None

    def workspace(self):
        return None

    def outputs(self, st):
        return dict(out=st["out"], lse=st["lse"])


NOPE_ARMS = {
    ArmCakeNope.name: ArmCakeNope,
    ArmCakeRopeBridge.name: ArmCakeRopeBridge,
    ArmFA4Nope.name: ArmFA4Nope,
    ArmFA4NopeA2.name: ArmFA4NopeA2,
    ArmFlashMLACudnnNope.name: ArmFlashMLACudnnNope,
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


def issue_median_ms(fn, reps=10):
    """The timing protocol of issue #7041: ``reps`` timed calls, each after a synchronize and a 20M-cycle
    ``torch.cuda._sleep``, CUDA events around the call, ``sorted()[reps // 2]``."""
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda._sleep(20_000_000)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return float(times[len(times) // 2])


def measure_perf_nope(arm, inp, steps, *, issue_protocol=False):
    """Median milliseconds of a NoPE arm: the training forward, the no-grad forward (detached
    inputs), the whole-row backward, the chunked backward (the 4096-query policy into one
    accumulate buffer; ``None`` where the arm chunks internally), the step (forward +
    backward) and the recompute step (two forwards + backward, activation checkpointing),
    plus the peak memory above the inputs, the workspace, the planner's ``key_passes``
    (Cake arms: the whole row and each chunk) and the row's geometry columns.
    ``issue_protocol`` adds the issue's own measurement (#7041: forward + ``autograd.grad``
    to ``q`` and ``kv`` with a fresh ``dkv``, and the no-grad forward, under
    :func:`issue_median_ms`) for the arms that expose ``issue_step``.
    """
    st = arm.forward()
    torch.cuda.synchronize()
    perf = dict(
        Tq=inp.total_q,
        Skv=inp.total_k,
        K_slots=inp.topk,
        valid_keys=inp.valid_keys,
        chunk_policy=NOPE_CHUNK,
        fwd_ms=median_ms(arm.forward, steps),
    )
    if hasattr(arm, "forward_nograd"):
        perf["fwd_nograd_ms"] = median_ms(arm.forward_nograd, steps)
    perf["bwd_ms"] = median_ms(lambda: arm.backward(st), steps)
    perf["bwd_chunks_ms"] = (
        median_ms(lambda: arm.backward_chunks(st), steps)
        if arm.backward_chunks(st) is not None
        else None
    )

    def step():
        arm.backward(arm.forward())

    def recompute_step():
        arm.forward()
        arm.backward(arm.forward())

    perf["step_ms"] = median_ms(step, steps)
    perf["recompute_step_ms"] = median_ms(recompute_step, steps)
    if issue_protocol and hasattr(arm, "issue_step"):
        arm.issue_step()  # the kv-leaf graph: plans before the timing
        perf["issue_fwd_bwd_ms"] = issue_median_ms(arm.issue_step)
        perf["issue_fwd_ms"] = issue_median_ms(arm.issue_forward)
    perf["workspace_bytes"] = arm.workspace()
    perf["key_passes"] = arm.key_passes() if hasattr(arm, "key_passes") else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    arm.backward(arm.forward())
    torch.cuda.synchronize()
    perf["peak_above_inputs_gib"] = gib(torch.cuda.max_memory_allocated() - base)
    return perf


def _nope_arm_or_error(cls, inp):
    """A NoPE arm over ``inp`` (the NoPE arms take no layout), or the error that makes it unavailable."""
    try:
        return cls(inp), None
    except (
        Exception
    ) as exc:  # optional dependency missing, entry not shipped, or arm unavailable here
        text = str(exc)
        if text.startswith("skipped:"):
            return None, text
        return None, f"{type(exc).__name__}: {text}"


def run_perf_nope_row(row, args, results):
    spec = NOPE_ROWS[row]
    inp = make_nope_inputs(
        spec["seq_q"],
        spec["seq_k"],
        seed=SEED,
        topk=spec["k_slots"],
        index=spec["index"],
        pools=spec["pools"],
        dout_layout=args.nope_dout_layout or spec["dout_layout"],
        device=args.device,
    )
    if spec.get("holes"):
        inp = _plant_holes(inp, spec["holes"])
    if spec.get("twin") == "stable":
        inp = compacted_twin(inp)
    entry = dict(
        row=row,
        total_q=inp.total_q,
        total_k=inp.total_k,
        num_docs=len(spec["seq_q"]),
        cu_seqlens_q=inp.cu_seqlens_q.tolist(),
        cu_seqlens_k=inp.cu_seqlens_k.tolist(),
        K_slots=inp.topk,
        index=inp.index,
        pools=inp.pools,
        valid_keys=inp.valid_keys,
        twin=spec.get("twin"),
        holes=spec.get("holes"),
        dout_layout=inp.dout_layout,
        dout_strides=list(inp.dout.stride()),
        inputs_gib=gib(inp.bytes_inputs()),
        arms={},
    )
    # the hole geometry: valid keys, the sum of last-valid + 1 (what a position-preserving kernel scans), the sum
    # of the valid counts (what a compacting kernel scans), the rows with a hole before a valid slot
    entry.update(inp.hole_stats())
    print(
        f"{row:26s} geometry T {inp.total_q} S {inp.total_k} K {inp.topk} index {inp.index} "
        f"valid {entry['valid_keys']} scan(pos) {entry['sum_last_valid_plus_one']} scan(compact) {entry['sum_counts']} "
        f"hole rows {entry['rows_with_holes_frac']:.3f}",
        flush=True,
    )
    steps = max(
        args.min_steps,
        args.steps if inp.total_q * inp.total_k < 2**34 else args.steps_128k,
    )
    arms = [a for a in args.arms if a in NOPE_ARMS] or NOPE_ARMS_DEFAULT.split(",")
    for name in arms:
        arm, error = _nope_arm_or_error(NOPE_ARMS[name], inp)
        if arm is None:
            entry["arms"][name] = dict(error=error)
            print(f"{row:26s} {name:19s} unavailable: {error}", flush=True)
            continue
        try:
            perf = measure_perf_nope(
                arm, inp, steps, issue_protocol=args.issue_protocol
            )
            perf["versions"] = arm.versions()
        except Exception as exc:
            perf = dict(
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        entry["arms"][name] = perf
        fmt = lambda v: "   n/a" if v is None else f"{v:8.3f}"
        if "error" in perf:
            print(f"{row:26s} {name:19s} failed: {perf['error']}", flush=True)
        else:
            kp = perf.get("key_passes")
            kp_text = (
                f"  key_passes {kp['whole_row']} (chunks {kp['chunks']})"
                if isinstance(kp, dict) and "whole_row" in kp
                else ""
            )
            issue_text = (
                f"  issue fwd+bwd {fmt(perf['issue_fwd_bwd_ms'])} ms fwd {fmt(perf['issue_fwd_ms'])} ms"
                if "issue_fwd_bwd_ms" in perf
                else ""
            )
            print(
                f"{row:26s} {name:19s} fwd {fmt(perf['fwd_ms'])} ms  bwd {fmt(perf['bwd_ms'])} ms  "
                f"step {fmt(perf['step_ms'])} ms  recompute {fmt(perf['recompute_step_ms'])} ms  "
                f"peak {perf['peak_above_inputs_gib']:.2f} GiB{kp_text}{issue_text}",
                flush=True,
            )
        del arm
        torch.cuda.empty_cache()
    results["rows"].append(entry)
    del inp
    torch.cuda.empty_cache()


def run_check_nope_arms(args, results):
    """Instantiate every requested NoPE arm on a small row (two documents, ``K_slots`` 128,
    the row ``dout`` layout) and run each entry once -- forward, backward, chunked backward,
    workspace, key passes, versions -- so a signature drift between the bench arms and the
    backend shows before the perf rows, not after their timings.  Records ``ok`` (with the
    output shapes / dtypes), ``skipped: ...`` or the error per arm."""
    inp = make_nope_inputs(
        [200, 312],
        [456, 712],
        seed=SEED,
        topk=128,
        index="iid",
        dout_layout=args.nope_dout_layout or "head_major",
        device=args.device,
    )
    arms = [a for a in args.arms if a in NOPE_ARMS] or NOPE_ARMS_DEFAULT.split(",")
    report = {}
    for name in arms:
        arm, error = _nope_arm_or_error(NOPE_ARMS[name], inp)
        if arm is None:
            report[name] = dict(status=error)
            print(f"check {name:19s} {error}", flush=True)
            continue
        try:
            st = arm.forward()
            grads = arm.backward(st)
            chunks = arm.backward_chunks(st)
            nograd = arm.forward_nograd() if hasattr(arm, "forward_nograd") else None
            torch.cuda.synchronize()
            report[name] = dict(
                status="ok",
                out=f"{tuple(st['out'].shape)} {st['out'].dtype}",
                dq=f"{tuple(grads['dq'].shape)} {grads['dq'].dtype}",
                dkv=f"{tuple(grads['dkv'].shape)} {grads['dkv'].dtype}",
                backward_chunks=None if chunks is None else "ok",
                forward_nograd=None if nograd is None else "ok",
                workspace_bytes=arm.workspace(),
                key_passes=arm.key_passes() if hasattr(arm, "key_passes") else None,
                versions=arm.versions(),
                dout_strides=list(inp.dout.stride()),
            )
            print(
                f"check {name:19s} ok {json.dumps(report[name], default=str)}",
                flush=True,
            )
        except Exception as exc:
            report[name] = dict(
                status=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
            )
            print(f"check {name:19s} FAILED {type(exc).__name__}: {exc}", flush=True)
        del arm
        torch.cuda.empty_cache()
    results["check_nope_arms"] = report
    return report


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
        if row in NOPE_ROWS:
            run_perf_nope_row(row, args, results)
            continue
        seq_q, seq_k = ROWS[row]
        dout_layout = args.dout_layout or ROW_DOUT_LAYOUT.get(row, "contiguous")
        inp = make_inputs(
            seq_q,
            seq_k,
            seed=SEED,
            topk=DEFAULT_TOPK,
            dout_layout=dout_layout,
            device=args.device,
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
            dout_layout=dout_layout,
            dout_strides=list(inp.dout.stride()),
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


NOPE_ACCURACY_CASES = {
    "nope_iid_4k": dict(index="iid", self_including=False, target_self_weight=None),
    "nope_pooled_4k": dict(
        index="pooled", self_including=False, target_self_weight=None
    ),
    "nope_peaked_053_4k": dict(
        index="iid", self_including=True, target_self_weight=0.53
    ),
    "nope_peaked_099_4k": dict(
        index="iid", self_including=True, target_self_weight=0.99
    ),
}


def _row_normalised(x, ref, atol=2e-2, rtol=1e-2):
    """Row-normalised comparison of ``x`` with the FP64 ``ref`` over the vectors of its last
    axis (the (token, head) rows of ``dq``): the p99 and maximum over rows of
    ``max_k |x - ref| / max(RMS(ref row), 1e-3)`` and the number of elements outside the band
    ``|x - ref| <= atol * max(RMS(ref row), 1e-3) + rtol * |ref|`` -- the per-row form the
    kernel's own tests use for ``dq`` in place of a flat elementwise band (the bf16 rounding of
    dS bounds the error per row, not per element).  Report only."""
    x2 = x.detach().double().reshape(-1, x.shape[-1])
    r2 = ref.detach().double().reshape(-1, ref.shape[-1])
    diff = (x2 - r2).abs_()
    den = r2.pow(2).mean(1, keepdim=True).sqrt_().clamp_min_(1e-3)
    row_max = (diff / den).max(1).values
    exceed = diff > (atol * den + rtol * r2.abs())
    # rows whose reference RMS sits below the 1e-3 clamp (a single-key row has dq == 0
    # exactly): their exceedances measure the clamp, not the kernel, so they are counted apart
    degenerate = r2.pow(2).mean(1) < 1e-6
    return dict(
        row_norm_max_p99=float(row_max.quantile(0.99)),
        row_norm_max_max=float(row_max.max()),
        band_exceed_count=int(exceed.sum()),
        elements=int(diff.numel()),
        rows_rms_below_clamp=int(degenerate.sum()),
        band_exceed_count_rows_rms_ge_clamp=int(exceed[~degenerate].sum()),
        band=dict(atol=atol, rtol=rtol, denominator="max(RMS(ref row), 1e-3)"),
    )


def measure_accuracy_nope(arm, inp):
    """Relative-L2 of a NoPE arm against the chunked FP64 reference (``out``, ``dq``, the
    combined ``dkv``), the ``lse`` maximum absolute error over finite rows, the p99 of the
    per-row ``dq`` rel-L2 and the row-normalised ``dq`` metrics (:func:`_row_normalised`),
    each beside the bf16 emulation floor of the same quantity."""
    st = arm.forward()
    grads = arm.backward(st)
    torch.cuda.synchronize()
    ref = reference_fp64_nope(
        inp.q,
        inp.kv,
        inp.idx_global,
        dout=inp.dout,
        softmax_scale=DEFAULT_NOPE_SCALE,
        own_key=inp.own_key,
    )
    outs = arm.outputs(st)
    valid = torch.isfinite(ref["lse"])
    return dict(
        self_weight=ref["self_weight"],
        out_rel_l2=rel_l2(outs["out"], ref["out"]),
        lse_max_abs=float((outs["lse"].double()[valid] - ref["lse"][valid]).abs().max())
        if valid.any()
        else 0.0,
        dq_rel_l2=rel_l2(grads["dq"], ref["dq"]),
        dkv_rel_l2=rel_l2(grads["dkv"], ref["dkv"]),
        dq_row_p99=float(rel_l2_rows(grads["dq"], ref["dq"]).quantile(0.99)),
        dq_row_normalised=_row_normalised(grads["dq"], ref["dq"]),
        dkv_row_p99=float(rel_l2_rows(grads["dkv"], ref["dkv"]).quantile(0.99)),
        bf16_floor=dict(
            out=rel_l2(ref["out_emu"], ref["out"]),
            dq=rel_l2(ref["dq_emu"], ref["dq"]),
            dq_row_p99=float(rel_l2_rows(ref["dq_emu"], ref["dq"]).quantile(0.99)),
            dq_row_normalised=_row_normalised(ref["dq_emu"], ref["dq"]),
            dkv=rel_l2(ref["dkv_emu"], ref["dkv"]),
            dkv_row_p99=float(rel_l2_rows(ref["dkv_emu"], ref["dkv"]).quantile(0.99)),
        ),
        nan=bool(
            any(
                torch.isnan(g).any().item()
                for g in grads.values()
                if torch.is_tensor(g)
            )
        ),
    )


def run_accuracy_nope(args, results):
    arms = [a for a in args.arms if a in NOPE_ARMS] or NOPE_ARMS_DEFAULT.split(",")
    for case, spec in NOPE_ACCURACY_CASES.items():
        beta = 0.0
        if spec["target_self_weight"] is not None:
            beta = calibrate_beta_nope(
                spec["target_self_weight"],
                seed=SEED,
                index=spec["index"],
                device=args.device,
            )
        inp = make_nope_inputs(
            [4096],
            [4096],
            seed=SEED,
            topk=DEFAULT_NOPE_TOPK,
            index=spec["index"],
            self_including=spec["self_including"],
            beta=beta,
            device=args.device,
        )
        entry = dict(
            case=case, beta=beta, K_slots=inp.topk, valid_keys=inp.valid_keys, arms={}
        )
        for name in arms:
            arm, error = _nope_arm_or_error(NOPE_ARMS[name], inp)
            if arm is None:
                entry["arms"][name] = dict(error=error)
                continue
            try:
                entry["arms"][name] = measure_accuracy_nope(arm, inp)
            except Exception as exc:
                entry["arms"][name] = dict(error=f"{type(exc).__name__}: {exc}")
            print(
                f"{case:20s} {name:19s} {json.dumps(entry['arms'][name], default=str)}",
                flush=True,
            )
            del arm
            torch.cuda.empty_cache()
        results["accuracy_nope"].append(entry)
        del inp
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
    """Host microseconds per call of the eager entry points against the prepared runner.

    Each round measures every entry point back to back, so they see the same
    process state; the medians over the rounds are reported.  Also checks that
    the eager path and the runner give the same results (``out``, ``lse``,
    ``dq_*`` bitwise; ``dkv_*`` within the ``red.global`` run-to-run spread)
    and reports the CUPTI kernel-only medians of both.
    """
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
    backward_available = cake_backend.generated_program_available(
        inp.q_latent.device, backward=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        dsa_sparse_attention(*args)  # the experimental banner fires once per process
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    saved = cake_backend.forward(*args)
    runner = cake_backend.prepare_dsa_train(
        *args,
        dout=inp.dout if backward_available else None,
        backward=backward_available,
    )
    torch.cuda.synchronize()

    def eager_forward():
        return cake_backend.forward(*args)

    def eager_backward():
        return cake_backend.backward(*args, saved[0], saved[2], saved[1], inp.dout)

    def runner_forward():
        return runner.forward()

    def runner_backward():
        return runner.backward()

    def public_forward():
        return dsa_sparse_attention(*leaves, inp.idx_global)

    graph_out = public_forward()

    def public_backward():  # autograd backward of one retained graph (the saved forward outputs are fixed)
        return torch.autograd.grad(graph_out, leaves, inp.dout, retain_graph=True)

    def autograd_step():
        out = dsa_sparse_attention(*leaves, inp.idx_global)
        return torch.autograd.grad(out, leaves, inp.dout)

    entry_points = [
        ("forward", eager_forward),
        ("runner_forward", runner_forward),
        ("public_forward", public_forward),
    ]
    if backward_available:
        entry_points += [
            ("backward", eager_backward),
            ("runner_backward", runner_backward),
            ("public_backward", public_backward),
            ("autograd_step", autograd_step),
        ]
    samples = {name: [] for name, _ in entry_points}
    orders = []
    for r in range(rounds):
        # rotate the measurement order every round so no entry point always runs
        # first (cache state and clock drift would otherwise favour one position)
        start = r % len(entry_points)
        order = entry_points[start:] + entry_points[:start]
        orders.append([name for name, _ in order])
        for name, fn in order:
            fn()  # warm the entry point; measured calls follow
            samples[name].append(_host_us_per_call(fn, calls))
    # same results from both paths
    e_fwd = eager_forward()
    r_fwd = runner_forward()
    e_bwd = eager_backward() if backward_available else None
    r_bwd = runner_backward() if backward_available else None
    torch.cuda.synchronize()
    same = dict(
        out=torch.equal(e_fwd[0], r_fwd[0]), lse=torch.equal(e_fwd[1], r_fwd[1])
    )
    if backward_available:
        same.update(
            dq_latent=torch.equal(e_bwd[0], r_bwd[0]),
            dq_rope=torch.equal(e_bwd[1], r_bwd[1]),
            dkv_latent_rel_l2=rel_l2(e_bwd[2], r_bwd[2]),
            dk_rope_rel_l2=rel_l2(e_bwd[3], r_bwd[3]),
        )
    del e_fwd, r_fwd, e_bwd, r_bwd
    # kernel-only time of both paths (CUPTI, per-iteration GPU span)
    kernel_ms = {}
    try:
        for mode, fwd, bwd in (
            ("eager", eager_forward, eager_backward),
            ("runner", runner_forward, runner_backward),
        ):
            kernel_ms[mode] = dict(forward=median_ms(fwd, kernel_steps))
            if backward_available:
                kernel_ms[mode]["backward"] = median_ms(bwd, kernel_steps)
        if backward_available:
            kernel_ms["eager"]["autograd_step"] = median_ms(autograd_step, kernel_steps)
    except (
        Exception
    ) as exc:  # the host figures stand on their own when CUPTI tracing is unavailable
        kernel_ms["error"] = f"{type(exc).__name__}: {exc}"
    host_us = {
        name: dict(
            median=float(statistics.median(v)),
            min=float(min(v)),
            rounds=[float(x) for x in v],
        )
        for name, v in samples.items()
    }
    return dict(
        calls=calls,
        rounds=rounds,
        order_per_round=orders,
        host_us=host_us,
        same_results=same,
        kernel_ms=kernel_ms,
    )


INDEX_GLUE_ROWS = ("packed_glm_a", "doc_16231", "tail_2123x67923")


def _kernel_table(fn):
    """Per-kernel launch table of one call of ``fn`` (torch profiler, CUDA activities; memcpy / memset excluded):
    ``[{name, calls, total_us}]`` sorted by time."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    table = {}
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.name.startswith(
            ("Memcpy", "Memset")
        ):
            continue
        t = table.setdefault(e.name, dict(name=e.name, calls=0, total_us=0.0))
        t["calls"] += 1
        t["total_us"] += (
            float(e.device_time) if hasattr(e, "device_time") else float(e.cuda_time)
        )
    return sorted(table.values(), key=lambda t: -t["total_us"])


def run_index_glue(args, results):
    """Index preparation alone on the rope rows (issue #7041 ask 3): the torch glue
    (``offset_gather_kv_indices(..., return_topk_length=True)``, the path the rope varlen entry
    replaces), the compacting twin (``compact=True``) and the on-device stage
    (``nope_globalize_indices(..., compact=True)``) -- CUPTI kernel-only medians, host
    microseconds per call and the per-launch kernel table of each; the stage must agree
    bitwise with the twin."""
    rows = [r for r in (args.index_glue_rows or INDEX_GLUE_ROWS) if r in ROWS]
    compacts = _glue_compacts()
    globalize = getattr(cake_backend, "nope_globalize_indices", None)
    stage_ok = (
        globalize is not None
        and compacts
        and cake_backend.nope_index_prep_available(torch.device(args.device))
    )
    for row in rows:
        seq_q, seq_k = ROWS[row]
        inp = make_inputs(
            seq_q, seq_k, seed=SEED, topk=DEFAULT_TOPK, device=args.device
        )
        local, cu_q, cu_k = inp.idx_local, inp.cu_seqlens_q, inp.cu_seqlens_k
        variants = {
            "glue": lambda: cake_backend.offset_gather_kv_indices(
                local, cu_q, cu_k, causal=True, return_topk_length=True
            )
        }
        if compacts:
            variants["glue_compact"] = lambda: cake_backend.offset_gather_kv_indices(
                local, cu_q, cu_k, causal=True, return_topk_length=True, compact=True
            )
        if stage_ok:
            variants["stage_compact"] = lambda: globalize(
                local, cu_q, cu_k, causal=True, compact=True
            )
        entry = dict(
            row=row,
            total_q=inp.total_q,
            total_k=inp.total_k,
            K_slots=inp.topk,
            variants={},
        )
        outputs = {}
        for name, fn in variants.items():
            outputs[name] = fn()
            torch.cuda.synchronize()
            rec = dict(
                median_ms=median_ms(fn, args.steps),
                host_us_per_call=statistics.median(
                    _host_us_per_call(fn, args.host_calls)
                    for _ in range(args.host_rounds)
                ),
                kernels=_kernel_table(fn),
            )
            rec["launches"] = sum(k["calls"] for k in rec["kernels"])
            entry["variants"][name] = rec
            print(
                f"{row:22s} {name:14s} {rec['median_ms']:8.3f} ms  host {rec['host_us_per_call']:8.1f} us/call  "
                f"launches {rec['launches']}",
                flush=True,
            )
            for k in rec["kernels"][:12]:
                print(
                    f"{'':22s}   {k['calls']:4d} x {k['total_us']:9.1f} us  {k['name'][:90]}",
                    flush=True,
                )
        if "stage_compact" in outputs and "glue_compact" in outputs:
            a, b = outputs["stage_compact"], outputs["glue_compact"]
            entry["stage_equals_twin"] = bool(
                torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            )
            print(f"{row:22s} stage == twin: {entry['stage_equals_twin']}", flush=True)
        results["index_glue"].append(entry)
        del inp, outputs
        torch.cuda.empty_cache()


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
            for name, stats in entry["host_us"].items():
                print(
                    f"{row:22s} {name:16s} host us/call  median {stats['median']:8.1f}  min {stats['min']:8.1f}",
                    flush=True,
                )
            print(
                f"{row:22s} same results: {json.dumps(entry['same_results'], default=str)}",
                flush=True,
            )
            for mode, ms in entry["kernel_ms"].items():
                print(
                    f"{row:22s} kernel-only ms ({mode}): {json.dumps(ms)}", flush=True
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
        choices=list(ROWS) + list(NOPE_ROWS),
        help="rows to run (default: every RoPE row, or the --preset rows); the nope_* rows "
        "run the NoPE arms",
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
    parser.add_argument(
        "--arms",
        default="cake,flashmla_cudnn,fa4",
        help="RoPE arms (cake, flashmla_cudnn, fa4) and / or NoPE arms (cake_nope, "
        "cake_rope_bridge, fa4_nope, flashmla_cudnn_nope); nope_* rows run the NoPE arms "
        f"named here, else all of them ({NOPE_ARMS_DEFAULT})",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--steps-128k", type=int, default=13, help="iterations for the largest problems"
    )
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument(
        "--accuracy-nope",
        action="store_true",
        help="NoPE accuracy cases (iid / pooled / peaked 4k x 4k at K_slots 2176, scale 1/16) "
        "against the FP64 reference, for the NoPE arms",
    )
    parser.add_argument(
        "--nope-dout-layout",
        choices=NOPE_DOUT_LAYOUTS,
        default=None,
        help="dout layout of every NoPE row: head_major (the rows' default, the recorded "
        "trainer strides [512, Tq*512, 1]) or contiguous (the control)",
    )
    parser.add_argument(
        "--dout-layout",
        choices=NOPE_DOUT_LAYOUTS,
        default=None,
        help="dout layout of every RoPE row (default: contiguous, or head_major for the *_hm rows)",
    )
    parser.add_argument(
        "--issue-protocol",
        action="store_true",
        help="NoPE rows: add the measurement of issue #7041 (forward + autograd.grad to q and kv, "
        "10 reps after a 20M-cycle _sleep, sorted()[5]) beside the CUPTI medians",
    )
    parser.add_argument(
        "--index-glue",
        action="store_true",
        help="index preparation alone on the rope rows: the torch glue vs its compacting twin vs the "
        "on-device stage (CUPTI medians, host us/call, per-launch kernel table)",
    )
    parser.add_argument(
        "--index-glue-rows",
        nargs="*",
        default=None,
        choices=list(ROWS),
        help=f"rows for --index-glue (default {', '.join(INDEX_GLUE_ROWS)})",
    )
    parser.add_argument(
        "--check-nope-arms",
        action="store_true",
        help="before the perf rows, instantiate every requested NoPE arm on a small row and "
        "run each entry once (forward, backward, chunked backward, workspace, key passes)",
    )
    parser.add_argument(
        "--host-us",
        action="store_true",
        help="host microseconds per call, eager entry points and prepared runner",
    )
    parser.add_argument("--host-calls", type=int, default=20)
    parser.add_argument("--host-rounds", type=int, default=3)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(args.arms) - set(ARMS) - set(NOPE_ARMS))
    if unknown:
        parser.error(
            f"unknown arms {unknown}; choose from {sorted(ARMS)} and {sorted(NOPE_ARMS)}"
        )
    if args.rows is None:
        args.rows = PRESETS[args.preset]["rows"] if args.preset else list(ROWS)
    # every layout the run will build (one per perf row, plus the accuracy cases) must carry dkv_acc when a
    # destination map is set, and a recorded glm_* map must match the key-row count of the inputs it is applied
    # to -- Layout() raises otherwise, which would abort the run after rows (or the whole perf run) have been timed
    planned = [
        (row, resolve_layout(row, args), sum(ROWS[row][1]))
        for row in args.rows
        if row in ROWS  # the NoPE rows take no layout options
    ]
    if args.accuracy:
        accuracy_layout = resolve_layout(None, args)
        planned += [
            (f"accuracy case {case!r}", accuracy_layout, sum(spec["seq_k"]))
            for case, spec in ACCURACY_CASES.items()
        ]
    for name, layout, total_k in planned:
        dst_map = layout.get("dkv_dst_map", "none")
        if dst_map != "none" and not layout.get("dkv_acc"):
            parser.error(
                f"--dkv-dst-map needs dkv_acc, but the layout of {name!r} resolves without it "
                "(add --dkv-acc or pick rows / a preset that enable it)"
            )
        if dst_map in GLM_DST_ROWS and sum(GLM_DST_ROWS[dst_map][0]) != total_k:
            parser.error(
                f"--dkv-dst-map {dst_map} is recorded for {sum(GLM_DST_ROWS[dst_map][0])} key rows, "
                f"but {name} has {total_k} (the glm_* maps fit only the rows of their batch; "
                "the accuracy cases use 4096-key inputs)"
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
        nope_options=dict(
            dout_layout=args.nope_dout_layout or "rows' default (head_major)",
            chunk_policy=NOPE_CHUNK,
            accumulator="fp32 dkv zero-filled inside the timed backward, once per step",
        ),
        rows=[],
        accuracy=[],
        accuracy_nope=[],
        host_path=[],
        index_glue=[],
    )
    if args.check_nope_arms:
        run_check_nope_arms(args, results)
    if not args.no_perf:
        run_perf(args, results)
    if args.accuracy:
        run_accuracy(args, results)
    if args.accuracy_nope:
        run_accuracy_nope(args, results)
    if args.host_us:
        run_host_path(args, results)
    if args.index_glue:
        run_index_glue(args, results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
