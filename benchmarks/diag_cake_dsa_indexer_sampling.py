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

"""Diagnose the sampled first threshold of the experimental Cake DSA indexer top-k backend.

The scan kernel of ``flashinfer.experimental.cake_dsa_indexer`` streams the visible key
prefix of every work unit (a block of ``block_q`` consecutive queries of one segment) in
128-key tiles.  A unit whose stream would overflow its candidate buffer first walks one tile
of every ``stride`` (``n_sample`` sample tiles); at the last sample tile each query picks
``tau_0``, the ``r'``-th largest of its buffered sample keys, as its first threshold
(``r = ceil(K c / visible)`` is the expected sample rank of the K-th key and
``r' = r + r * sample_shift_permille / 1000 + 24`` its conservative shift).  The rest of the
walk keeps only keys at or above ``tau_0``.  The pass is exact when every query ends with
``min(K, visible)`` keys in its buffer; a unit with any query below that count has failed its
verdict and is streamed again without sample tiles.  The results are bitwise identical either
way; only the time changes, and a rank of a context-parallel call whose units fail more often
than its neighbours' takes longer than they do.

This script replays that decision on the host from the actual inputs -- it scores every
(query, visible key) pair, composes the kernel's 64-bit order key (score bits, then id) and
runs the sample select exactly as the shipped programs do -- and prints, for one call:

* the plan the backend picks for the architecture (program, ``block_q``, CTA pairs,
  ``cand_cap``, ``check_period``, ``sample_tiles_max``, ``sample_shift_permille``,
  ``far_sample``, the sampled-fit form, the tile order and the rank statistic it implies);
* units total / sampled / failed, the share of streamed key tiles that are streamed twice,
  and the same per segment;
* the distribution of failures over the persistent CTAs under the static round-robin
  assignment (unit ``u`` on CTA ``u mod G``).  The programs that draw their units from a
  global ticket word (``TICKET_TAIL_PROGRAMS`` of the installed package; every sm_100a and
  sm_103a scan program once that change ships) assign units dynamically, so for them this
  per-CTA view is an approximation and is labelled as such;
* one ``SUMMARY`` line per call, suitable for pasting into an issue, and with ``--csv`` one row
  per unit (``--json`` keeps everything).

Inputs are one of:

* ``--load FILE``: a ``torch.save`` dict with ``q`` ``[T, 32, 128]`` bf16, ``k`` ``[Tkv, 128]``
  bf16, ``w`` ``[T, 32]`` fp32, ``cu_seqlens_q`` / ``cu_seqlens_k`` int32 ``[S + 1]``, and
  optionally ``q_causal_offsets`` (int64 ``[S]``), ``top_k`` (default 2048), ``ratio``
  (default 1), ``max_seqlen_k``, ``softmax_scale`` (default ``128 ** -0.5``) -- one rank's
  arguments to ``dsa_indexer_topk`` saved as they were passed;
* ``--row NAME``: a row of ``benchmarks/bench_cake_dsa_indexer.py`` (``S4sink``, ``Z1``, ``P1``,
  ...), built with that script's generators and seeds;
* ``--cp16 MIX --dist DIST --rank R``: one rank of a 16-way context-parallel zigzag split of a
  synthetic document mix (``doc262k``: one 262144-token document; ``packed11``: eleven
  documents of 4096 .. 65536 tokens, lengths printed) with iid Gaussian (``iid``),
  attention-sink plus local-band (``sink``) or topic-block (``topic``) score statistics;
  ``--save FILE`` writes the rank's inputs in the ``--load`` format.

``--knob NAME=VALUE`` overrides a plan knob (``sample_tiles_max``, ``check_period``,
``sample_shift_permille``, ``far_sample``, ``candidate_multiplier``, ``grid_ctas``) the way the
backend's keyword arguments do; ``--arch`` / ``--num-sms`` plan for another architecture or SM
count (the replay itself needs no GPU: ``--device cpu`` works, slowly).  ``--time`` (GPU only)
also runs the kernel on the inputs with the default plan and with ``sample_tiles_max=0`` --
the benchmark script's CUPTI timer, three paired phases (AB, BA, interleaved) -- and reports
both medians and their ratio next to the predicted failures.

Approximations, stated once here: the replay's ``tau_0`` is the exact ``r'``-th largest
sample key, while the kernel's radix select returns the lower edge of a kept set of
``r' .. r' + 64`` entries (a ``tau_0`` at most a few ranks lower), so the replayed failure
count is a slight upper bound; the scores are fp32 accumulations of exact bf16 products in
PyTorch's summation order, not the kernel's, which can flip a verdict only when ``tau_0`` and
the K-th key are within an ulp; the per-CTA attribution assumes the static assignment.

Usage::

    python benchmarks/diag_cake_dsa_indexer_sampling.py --row S4sink
    python benchmarks/diag_cake_dsa_indexer_sampling.py --load rank3.pt --csv rank3_units.csv
    python benchmarks/diag_cake_dsa_indexer_sampling.py --cp16 packed11 --dist topic --rank 3 --time
    python benchmarks/diag_cake_dsa_indexer_sampling.py --row Z1 --arch sm_100a --num-sms 148 --device cpu
"""

import argparse
import csv
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_cake_dsa_indexer as bcdi  # noqa: E402  (benchmarks/ is the script directory)
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.experimental.cake_dsa_indexer import cake_backend  # noqa: E402
from tests.test_helpers import cake_dsa_indexer_reference as ref  # noqa: E402

TILE = 128
NUM_HEADS = ref.NUM_HEADS
HEAD_DIM = ref.HEAD_DIM
SAMPLE_RANK_PAD = 24  # the kernel's SAMPLE_RANK_PAD
SAMPLE_FIT_TILES = 8  # the sampled-fit form samples at most eight tiles
DEFAULT_NUM_SMS = {"sm_100a": 148, "sm_103a": 160, "sm_107a": 212}
KNOBS = (
    "sample_tiles_max",
    "check_period",
    "sample_shift_permille",
    "far_sample",
    "candidate_multiplier",
    "grid_ctas",
)
CP_PARTS = 16
PACKED11_DOCS = (
    4096,
    6144,
    8192,
    12288,
    16384,
    20480,
    24576,
    32768,
    40960,
    49152,
    65536,
)
CP16_MIXES = {"doc262k": (262144,), "packed11": PACKED11_DOCS}
TOPIC_SEED = 7912
IID_SEED = 7620


# ---------------------------------------------------------------------------
# Plan, tile order and the rank statistic of the shipped programs
# ---------------------------------------------------------------------------


def tile_order_desc(arch: str, physical_kind: str) -> bool:
    """Tile order of the architecture's program: sm_100a walks descending (the diagonal tile
    first) except its three-warpgroup ``l6`` program; sm_103a walks ascending everywhere;
    sm_107a walks ascending on its wide program only."""
    if arch == "sm_100a":
        return physical_kind != "l6"
    if arch == "sm_103a":
        return False
    if arch == "sm_107a":
        return physical_kind != "wide"
    raise ValueError(f"unknown architecture {arch!r}")


def statistic_form(arch: str, physical_kind: str, desc: bool, far_sample: int) -> str:
    """Which sample entries the rank statistic ranks.

    ``diag``: a descending program leaves the unit's diagonal tile (tile ``n_tiles - 1``) out
    of both the count and the visible span (that tile is always a sample tile and holds the
    hottest keys of a causal row); ``near``: an ascending program leaves out its last sample
    tile ``(n_sample - 1) * stride`` -- always on the sm_103a gate-warpgroup wide program, under
    the ``far_sample`` knob on the other ascending programs; ``plain``: every sample entry.
    """
    if desc:
        return "diag"
    if arch == "sm_103a" and physical_kind == "wide":
        return "near"
    return "near" if int(far_sample) else "plain"


def sample_plan(n_tiles, cand_cap, check_period, sample_tiles_max, top_k, fit):
    """``(stride, n_sample)`` of a unit of ``n_tiles`` key tiles, as the kernel plans it."""
    max_tiles = min(sample_tiles_max, cand_cap // TILE - check_period)
    stride, n_sample = 1, 0
    if max_tiles >= 2 and n_tiles * TILE + TILE > cand_cap:
        stride = -(-n_tiles // max_tiles)
        if stride >= 2:
            n_sample = -(-n_tiles // stride)
    if fit and max_tiles >= 2 and n_tiles * TILE + TILE <= cand_cap:
        if n_tiles * (TILE // 2) > top_k:
            cap_fit = min(max_tiles, SAMPLE_FIT_TILES)
            if cap_fit >= 2:
                stride = -(-n_tiles // cap_fit)
                if stride >= 2:
                    n_sample = -(-n_tiles // stride)
    return stride, n_sample


def parse_knobs(items):
    knobs = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"--knob expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if name not in KNOBS:
            raise SystemExit(f"unknown knob {name!r}; known: {', '.join(KNOBS)}")
        knobs[name] = int(value)
    return knobs


def host_plan(inputs, arch, num_sms, knobs, max_seqlen_k=None):
    plan = cake_backend.plan_dsa_indexer_topk(
        inputs.num_queries,
        inputs.num_keys,
        inputs.num_segments,
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        arch=arch,
        num_sms=num_sms,
        max_seqlen_k=max_seqlen_k,
        **knobs,
    )
    desc = tile_order_desc(arch, plan.physical_kind)
    form = statistic_form(arch, plan.physical_kind, desc, int(plan.far_sample))
    program = cake_backend.program_for(arch, plan.scan_key)
    dynamic = arch in getattr(cake_backend, "TICKET_TAIL_PROGRAMS", {}).get(program, ())
    return dict(
        arch=arch,
        num_sms=int(num_sms),
        scan_key=plan.scan_key,
        program=program,
        kind=plan.kind,
        physical_kind=plan.physical_kind,
        block_q=int(plan.block_q),
        pair=bool(plan.pair),
        n_split=int(plan.n_split),
        grid_ctas=int(plan.grid_ctas),
        cand_cap=int(plan.cand_cap),
        check_period=int(plan.check_period),
        sample_tiles_max=int(plan.sample_tiles_max),
        sample_shift_permille=int(plan.sample_shift_permille),
        far_sample=int(plan.far_sample),
        sample_fit=bool(plan.sample_fit),
        tile_order="desc" if desc else "asc",
        statistic=form,
        dynamic_tickets=bool(dynamic),
        workspace_bytes=int(plan.workspace_bytes),
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def cp_zigzag_rank(length: int, rank: int, parts: int = CP_PARTS):
    """Rank ``rank`` of a ``parts``-way zigzag split of one ``length``-token document: query
    chunks ``rank`` and ``2 parts - 1 - rank`` of ``2 parts`` chunks of ``c`` tokens, each
    attending its causal prefix (``seg_q = [c, c]``, ``seg_k = [(rank + 1) c, (2 parts - rank) c]``,
    the offsets by the ``Lk - Lq`` rule)."""
    c = length // (2 * parts)
    return [c, c], [(rank + 1) * c, (2 * parts - rank) * c]


def cp16_geometry(mix: str, rank: int):
    seg_q, seg_k, docs = [], [], []
    for length in CP16_MIXES[mix]:
        sq, sk = cp_zigzag_rank(length, rank)
        seg_q += sq
        seg_k += sk
        docs += [length, length]
    return seg_q, seg_k, docs


def topic_blocks(gen, length, device, n_topics, lo=128, hi=2048):
    """Topic id per document position: blocks of ``U(lo, hi)`` tokens, never two equal in a row."""
    ids = torch.empty(length, dtype=torch.int64, device=device)
    pos, prev = 0, -1
    while pos < length:
        n = int(torch.randint(lo, hi + 1, (1,), generator=gen, device=device))
        t = int(torch.randint(0, n_topics, (1,), generator=gen, device=device))
        if t == prev:
            t = (t + 1) % n_topics
        ids[pos : pos + n] = t
        pos += n
        prev = t
    return ids


def make_topic_inputs(
    mix,
    rank,
    device,
    *,
    top_k=2048,
    seed=TOPIC_SEED,
    noise=0.5,
    topic=1.0,
    broad_frac=0.3,
    broad_scale=0.3,
    label="",
):
    """Topic-block statistics on the zigzag rank: every document is a sequence of blocks of
    128 .. 2048 tokens drawn from 3 .. 6 topics (the same blocks on every rank: seeded per
    document); a key carries its block's topic vector (a random +-1 direction in dims 64 ..
    127), a query the vector of its own block in every head, 30 % of the queries attend broadly
    (vector x 0.3); Gaussian noise in dims 0 .. 63 orders the keys within a topic.  A query then
    scores high on every key of its topic, near or far, and a sample whose tiles land in more
    same-topic blocks than the row's average over-estimates the density of high keys."""
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed) * 100 + int(rank))
    seg_q, seg_k, docs = cp16_geometry(mix, rank)
    code = HEAD_DIM // 2
    T, Tkv = sum(seg_q), sum(seg_k)
    q = torch.randn((T, NUM_HEADS, HEAD_DIM), generator=gen, device=device) * noise
    k = torch.randn((Tkv, HEAD_DIM), generator=gen, device=device) * noise
    q[:, :, code:] = 0
    k[:, code:] = 0
    w = (
        torch.randn((T, NUM_HEADS), generator=gen, device=device).abs()
        * NUM_HEADS**-0.5
    )
    q0 = k0 = 0
    lengths = list(CP16_MIXES[mix])
    for s, (lq, lk) in enumerate(zip(seg_q, seg_k, strict=True)):
        length = docs[s]
        dgen = torch.Generator(device=device).manual_seed(
            int(seed) * 1000 + lengths.index(length)
        )
        n_topics = int(torch.randint(3, 7, (1,), generator=dgen, device=device))
        vectors = (
            torch.randint(0, 2, (n_topics, code), generator=dgen, device=device) * 2 - 1
        ).float() * topic
        ids = topic_blocks(dgen, length, device, n_topics)
        offset = lk - lq
        k[k0 : k0 + lk, code:] = vectors[ids[:lk]]
        qv = vectors[ids[offset : offset + lq]]
        broad = torch.rand(lq, generator=gen, device=device) < broad_frac
        qv = torch.where(broad[:, None], qv * broad_scale, qv)
        q[q0 : q0 + lq, :, code:] = qv[:, None, :]
        q0 += lq
        k0 += lk
    return ref.build_inputs(
        q.bfloat16(),
        k.bfloat16(),
        w,
        seg_q,
        seg_k,
        top_k=top_k,
        softmax_scale=ref.DEFAULT_SCALE,
        ratio=1,
        label=label,
    )


def build_cp16_inputs(mix, dist, rank, device, top_k, label):
    seg_q, seg_k, _docs = cp16_geometry(mix, rank)
    if dist == "iid":
        return ref.make_random_inputs(
            seg_q, seg_k, top_k=top_k, seed=IID_SEED + rank, device=device, label=label
        )
    if dist == "sink":
        return ref.make_sink_inputs(
            seg_q, seg_k, top_k=top_k, seed=ref.SINK_SEED, device=device, label=label
        )
    if dist == "topic":
        return make_topic_inputs(mix, rank, device, top_k=top_k, label=label)
    raise SystemExit(f"unknown --dist {dist!r}")


def load_inputs(path, device, label):
    d = torch.load(path, map_location=device, weights_only=True)
    cu_q, cu_k = d["cu_seqlens_q"].tolist(), d["cu_seqlens_k"].tolist()
    seg_q = [b - a for a, b in zip(cu_q[:-1], cu_q[1:], strict=True)]
    seg_k = [b - a for a, b in zip(cu_k[:-1], cu_k[1:], strict=True)]
    offsets = d.get("q_causal_offsets")
    if offsets is not None:
        offsets = [int(v) for v in offsets.tolist()]
    inputs = ref.build_inputs(
        d["q"].to(device=device, dtype=torch.bfloat16),
        d["k"].to(device=device, dtype=torch.bfloat16),
        d["w"].to(device=device, dtype=torch.float32),
        seg_q,
        seg_k,
        top_k=int(d.get("top_k", ref.DEFAULT_TOP_K)),
        softmax_scale=float(d.get("softmax_scale", ref.DEFAULT_SCALE)),
        q_causal_offsets=offsets,
        ratio=int(d.get("ratio", 1)),
        label=label,
    )
    max_seqlen_k = d.get("max_seqlen_k")
    return inputs, (None if max_seqlen_k is None else int(max_seqlen_k))


def save_inputs(inputs, max_seqlen_k, path):
    d = dict(
        q=inputs.q.cpu(),
        k=inputs.k.cpu(),
        w=inputs.w.cpu(),
        cu_seqlens_q=inputs.cu_seqlens_q.cpu(),
        cu_seqlens_k=inputs.cu_seqlens_k.cpu(),
        top_k=int(inputs.top_k),
        ratio=int(inputs.ratio),
        softmax_scale=float(inputs.softmax_scale),
    )
    if inputs.q_causal_offsets is not None:
        d["q_causal_offsets"] = inputs.q_causal_offsets.cpu()
    if max_seqlen_k is not None:
        d["max_seqlen_k"] = int(max_seqlen_k)
    torch.save(d, path)


# ---------------------------------------------------------------------------
# Scores and the kernel's order key
# ---------------------------------------------------------------------------


def composite_keys(scores, ids):
    """The scan's 64-bit order key as a non-negative int64 (``key32 << 31 | id``): a monotone
    unsigned image of the fp32 score bits (both zeros equal, NaN below everything), then the id."""
    m = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    m = torch.where((m & 0x7FFFFFFF) == 0, torch.zeros_like(m), m)
    key32 = torch.where((m & 0x80000000) != 0, (~m) & 0xFFFFFFFF, m | 0x80000000)
    key32 = torch.where((m & 0x7FFFFFFF) > 0x7F800000, torch.zeros_like(key32), key32)
    return (key32 << 31) | ids.to(torch.int64)


def segment_scores(q, kt, w, scale, score_bytes):
    """fp32 ``[rows, Lk]`` scores ``sum_h w_h relu(scale q_h . k_j)`` of a block of query rows
    against ``kt`` (``[128, Lk]`` bf16): exact bf16 products, fp32 accumulation."""
    rows, heads, dim = q.shape
    lk = kt.shape[1]
    out = torch.empty((rows, lk), dtype=torch.float32, device=q.device)
    inner = max(1, score_bytes // max(1, heads * lk * 4))
    fp32_out = q.device.type == "cuda" and bcdi._fp32_mm_supported(q.device)
    for r0 in range(0, rows, inner):
        r1 = min(rows, r0 + inner)
        q2d = q[r0:r1].reshape((r1 - r0) * heads, dim)
        if fp32_out:
            logits = torch.mm(q2d, kt, out_dtype=torch.float32)
        else:
            logits = torch.matmul(q2d.float(), kt.float())
        logits = torch.relu_(logits.mul_(scale)).view(r1 - r0, heads, lk)
        out[r0:r1] = torch.einsum("qhk,qh->qk", logits, w[r0:r1].float())
        del logits
    return out


def r_shift(r, shift):
    """``r + trunc(r * shift / 1000) + 24`` with C integer division."""
    prod = r * shift
    trunc = torch.where(prod >= 0, prod // 1000, -((-prod) // 1000))
    return r + trunc + SAMPLE_RANK_PAD


# ---------------------------------------------------------------------------
# Unit enumeration
# ---------------------------------------------------------------------------


def visible_count(offset, u, lk, ratio):
    num = offset + u + 1
    return max(0, min(lk, num // ratio)) if num > 0 else 0


def segment_units(seg_index, lq, lk, offset, ratio, plan):
    """The units of one segment in the kernel's enumeration order (longest first): per unit the
    blocks it covers (two on a CTA-pair program), its tile count (the pair's first block's) and
    its split items ``(t_lo, n_item)``."""
    bq, pair, n_split = plan["block_q"], plan["pair"], plan["n_split"]
    nb = -(-lq // bq)

    def tiles_of(blk):
        last = min(blk * bq + bq - 1, lq - 1)
        return -(-visible_count(offset, last, lk, ratio) // TILE)

    units = []
    if pair:
        for rel in range((nb + 1) // 2):
            blk0 = nb - 1 - 2 * rel
            blocks = [blk0] + ([blk0 - 1] if blk0 - 1 >= 0 else [])
            units.append(
                dict(seg=seg_index, blocks=blocks, n_tiles=tiles_of(blk0), item=0)
            )
    else:
        for rel in range(nb):
            blk = nb - 1 - rel
            n_tiles = tiles_of(blk)
            for item in range(n_split):
                units.append(
                    dict(seg=seg_index, blocks=[blk], n_tiles=n_tiles, item=item)
                )
    for u in units:
        n_tiles, item = u["n_tiles"], u["item"]
        t_lo = (item * n_tiles) // n_split
        u["t_lo"] = t_lo
        u["n_item"] = ((item + 1) * n_tiles) // n_split - t_lo
    # the sm_103a gate-warpgroup wide program's sampled mode hands out a near-equal segment
    # (last block's tile count at most 5/4 of the first block's) in ascending block order
    order_t = False
    if units and plan["dynamic_tickets"] and plan["sample_tiles_max"] != 0:
        if plan["arch"] == "sm_103a" and plan["physical_kind"] == "wide":
            order_t = tiles_of(nb - 1) * 4 <= tiles_of(0) * 5
    return units[::-1] if order_t else units


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------


def replay(inputs, plan, args, device):
    """Replay the sample select and the verdict of every unit.  Returns the per-unit records
    (in enumeration order) and the per-query totals."""
    K = inputs.top_k
    bq, desc, form = plan["block_q"], plan["tile_order"] == "desc", plan["statistic"]
    shift = plan["sample_shift_permille"]
    offsets = inputs.offsets()
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    records = []
    unit_index = 0
    for s in range(inputs.num_segments):
        lq, lk = inputs.seg_q_len[s], inputs.seg_k_len[s]
        if lq == 0 or lk == 0:
            continue
        offset = offsets[s]
        units = segment_units(s, lq, lk, offset, inputs.ratio, plan)
        q0s, k0s = q_starts[s], k_starts[s]
        kt = inputs.k[k0s : k0s + lk].t().contiguous()
        vis_all = torch.tensor(
            [visible_count(offset, u, lk, inputs.ratio) for u in range(lq)],
            dtype=torch.int64,
            device=device,
        )
        ids = torch.arange(lk, device=device)
        # score consecutive query blocks in chunks; a unit's blocks are consecutive and the
        # chunk boundaries fall on unit boundaries
        rows_per_chunk = max(bq, (args.chunk_rows // bq) * bq)
        per_chunk = max(1, rows_per_chunk // (bq * (2 if plan["pair"] else 1)))
        # units of a segment are enumerated longest-first, i.e. from the last block down
        for c0 in range(0, len(units), per_chunk):
            chunk = units[c0 : c0 + per_chunk]
            blocks = sorted({b for u in chunk for b in u["blocks"]})
            lo_blk, hi_blk = blocks[0], blocks[-1] + 1
            r0, r1 = lo_blk * bq, min(lq, hi_blk * bq)
            vmax = int(vis_all[r1 - 1])
            if vmax == 0:
                for u in chunk:
                    records.append(_empty_record(u, unit_index, plan))
                    unit_index += 1
                continue
            scores = segment_scores(
                inputs.q[q0s + r0 : q0s + r1],
                kt[:, :vmax],
                inputs.w[q0s + r0 : q0s + r1],
                inputs.softmax_scale,
                args.score_bytes,
            )
            keys = composite_keys(scores, ids[None, :vmax].expand_as(scores))
            del scores
            valid = ids[None, :vmax] < vis_all[r0:r1, None]
            keys = torch.where(valid, keys, torch.full_like(keys, -1))
            # one padding row (all invalid) and one padding column for the gathers
            keys = torch.nn.functional.pad(keys, (0, 1, 0, 1), value=-1)
            n_rows = r1 - r0
            pad_row, pad_col = n_rows, vmax
            vis_rows = torch.cat([vis_all[r0:r1], vis_all.new_zeros(1)])
            # per unit: its query rows (relative to r0), its sample columns, its exclusions
            rows_q = []
            cols_u = []
            far_lo_u = []
            lo_keys_u = []
            hi_keys_u = []
            plans = []
            qrows = bq * (2 if plan["pair"] else 1)
            for u in chunk:
                n_item, t_lo = u["n_item"], u["t_lo"]
                stride, n_sample = sample_plan(
                    n_item,
                    plan["cand_cap"],
                    plan["check_period"],
                    plan["sample_tiles_max"],
                    K,
                    plan["sample_fit"],
                )
                plans.append((stride, n_sample))
                rr = []
                for blk in u["blocks"]:
                    rr += [
                        (blk * bq + i - r0) if blk * bq + i < lq else pad_row
                        for i in range(bq)
                    ]
                rr += [pad_row] * (qrows - len(rr))
                rows_q.append(rr)
                lo_keys = t_lo * TILE
                hi_keys = min(lk, (t_lo + n_item) * TILE)
                lo_keys_u.append(lo_keys)
                hi_keys_u.append(hi_keys)
                tiles = []
                if n_sample:
                    ds = [i * stride for i in range(n_sample)]
                    tiles = [(n_item - 1 - d) if desc else d for d in ds]
                if form == "diag" and n_sample:
                    far_lo = lo_keys + (n_item - 1) * TILE
                elif form == "near" and n_sample:
                    far_lo = lo_keys + (n_sample - 1) * stride * TILE
                else:
                    far_lo = 1 << 62
                far_lo_u.append(far_lo)
                cols = [lo_keys + t * TILE + j for t in tiles for j in range(TILE)]
                cols_u.append([c if c < vmax else pad_col for c in cols])
            U = len(chunk)
            smax = max(len(c) for c in cols_u)
            cols_t = torch.full(
                (U, max(1, smax)), pad_col, dtype=torch.int64, device=device
            )
            for i, c in enumerate(cols_u):
                if c:
                    cols_t[i, : len(c)] = torch.tensor(
                        c, dtype=torch.int64, device=device
                    )
            rows_t = torch.tensor(
                rows_q, dtype=torch.int64, device=device
            )  # [U, qrows]
            far_lo_t = torch.tensor(far_lo_u, dtype=torch.int64, device=device)
            lo_keys_t = torch.tensor(lo_keys_u, dtype=torch.int64, device=device)
            hi_keys_t = torch.tensor(hi_keys_u, dtype=torch.int64, device=device)
            # [U, qrows, smax] sample keys; -1 where invisible / padding
            sk = keys[rows_t[:, :, None], cols_t[:, None, :]]
            sv = sk >= 0
            far = sv & (cols_t[:, None, :] < far_lo_t[:, None, None])
            vis_q = vis_rows[rows_t]  # [U, qrows] visible keys of every query
            vis_c = (vis_q - lo_keys_t[:, None]).clamp(min=0)
            vis_c = torch.minimum(vis_c, (hi_keys_t - lo_keys_t)[:, None])
            live = (rows_t != pad_row) & (vis_c > 0)
            c_all = sv.sum(-1)
            c_far = far.sum(-1)
            n_x = c_all - c_far
            v_far = vis_c - n_x
            r = torch.where(
                v_far > 0,
                -(-(K * c_far) // v_far.clamp(min=1)),
                torch.zeros_like(c_far),
            )
            rp = r_shift(r, shift)
            fires = live & (v_far > 0) & (rp < c_far) & (rp >= 1) & (c_far > 0)
            order = torch.sort(
                torch.where(far, sk, torch.full_like(sk, -1)), dim=-1, descending=True
            ).values
            idx = (rp - 1).clamp(min=0, max=order.shape[-1] - 1)
            tau = order.gather(-1, idx[..., None])[..., 0]
            tau = torch.where(fires, tau, torch.full_like(tau, -1))
            # admitted keys at or above tau_0 over the item's whole visible range of each query
            col_range = ids[None, :vmax]
            in_item = (col_range >= lo_keys_t[:, None, None]) & (
                col_range < hi_keys_t[:, None, None]
            )
            krows = keys[rows_t][:, :, :vmax]  # [U, qrows, vmax]
            admitted = ((krows >= tau[..., None]) & (krows >= 0) & in_item).sum(-1)
            del krows
            need = torch.minimum(torch.full_like(vis_c, K), vis_c)
            fail_q = fires & (admitted < need)
            adm_over_k = torch.where(
                fires,
                admitted.double() / K,
                torch.full(admitted.shape, float("nan"), device=device),
            )
            for i, u in enumerate(chunk):
                stride, n_sample = plans[i]
                live_i, fires_i, fail_i = live[i], fires[i], fail_q[i]
                adm = adm_over_k[i][fires_i]
                rec = dict(
                    unit=unit_index,
                    segment=u["seg"],
                    blocks=list(u["blocks"]),
                    item=u["item"],
                    n_tiles=int(u["n_item"]),
                    stride=int(stride),
                    n_sample=int(n_sample),
                    queries=int(live_i.sum()),
                    fired=int(fires_i.sum()),
                    failed_queries=int(fail_i.sum()),
                    fail=bool(fail_i.any()),
                    adm_min_over_k=(float(adm.min()) if adm.numel() else None),
                    adm_max_over_k=(float(adm.max()) if adm.numel() else None),
                    c_mean=(
                        float(c_all[i][live_i].double().mean()) if n_sample else 0.0
                    ),
                    r_shift_max=(
                        int(rp[i][fires_i].max()) if bool(fires_i.any()) else 0
                    ),
                )
                records.append(rec)
                unit_index += 1
            del keys, valid, sk, sv, far, order
        del kt, vis_all
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def _empty_record(u, unit_index, plan):
    return dict(
        unit=unit_index,
        segment=u["seg"],
        blocks=list(u["blocks"]),
        item=u["item"],
        n_tiles=int(u["n_item"]),
        stride=1,
        n_sample=0,
        queries=0,
        fired=0,
        failed_queries=0,
        fail=False,
        adm_min_over_k=None,
        adm_max_over_k=None,
        c_mean=0.0,
        r_shift_max=0,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(records, plan, inputs):
    G = plan["grid_ctas"] // (2 if plan["pair"] else 1)
    sampled = [r for r in records if r["n_sample"] > 0]
    failed = [r for r in sampled if r["fail"]]
    tiles = sum(r["n_tiles"] for r in records)
    fail_tiles = sum(r["n_tiles"] for r in failed)
    per_seg = {}
    for r in records:
        seg = per_seg.setdefault(
            r["segment"], dict(units=0, sampled=0, failed=0, tiles=0, fail_tiles=0)
        )
        seg["units"] += 1
        seg["tiles"] += r["n_tiles"]
        if r["n_sample"] > 0:
            seg["sampled"] += 1
            if r["fail"]:
                seg["failed"] += 1
                seg["fail_tiles"] += r["n_tiles"]
    offsets = inputs.offsets()
    segments = []
    for s in range(inputs.num_segments):
        d = per_seg.get(s, dict(units=0, sampled=0, failed=0, tiles=0, fail_tiles=0))
        segments.append(
            dict(
                segment=s,
                Lq=inputs.seg_q_len[s],
                Lk=inputs.seg_k_len[s],
                offset=offsets[s],
                **d,
                restream_tile_pct=100.0 * d["fail_tiles"] / max(1, d["tiles"]),
            )
        )
    per_cta = [0] * G
    first_pos = []
    lists = [[] for _ in range(G)]
    for r in records:
        lists[r["unit"] % G].append(r)
    for cta, lst in enumerate(lists):
        flags = [r["fail"] for r in lst]
        per_cta[cta] = sum(flags)
        if any(flags):
            first_pos.append(flags.index(True) / max(1, len(lst)))
    hist = {"0": 0, "1": 0, "2": 0, "3-5": 0, "6+": 0}
    for n in per_cta:
        hist[
            "0"
            if n == 0
            else "1"
            if n == 1
            else "2"
            if n == 2
            else "3-5"
            if n <= 5
            else "6+"
        ] += 1
    return dict(
        units=len(records),
        sampled=len(sampled),
        failed=len(failed),
        failed_pct_of_sampled=100.0 * len(failed) / max(1, len(sampled)),
        tiles=tiles,
        fail_tiles=fail_tiles,
        restream_tile_pct=100.0 * fail_tiles / max(1, tiles),
        failed_queries=sum(r["failed_queries"] for r in failed),
        segments=segments,
        cta=dict(
            G=G,
            assignment="static u mod G (approximation: this program draws dynamic tickets)"
            if plan["dynamic_tickets"]
            else "static u mod G (exact for this program)",
            ctas_with_fail=sum(1 for n in per_cta if n),
            max_fails_per_cta=max(per_cta) if per_cta else 0,
            hist=hist,
            first_fail_pos_median=(statistics.median(first_pos) if first_pos else None),
            first_fail_pos_p90=(
                sorted(first_pos)[int(0.9 * (len(first_pos) - 1))]
                if first_pos
                else None
            ),
            fails_per_cta=per_cta,
        ),
    )


# ---------------------------------------------------------------------------
# Timing (GPU): the default plan against sample_tiles_max = 0 on the same inputs
# ---------------------------------------------------------------------------


def time_default_vs_off(inputs, label, max_seqlen_k, reps, warmup, knobs):
    bcdi.ROWS[label] = dict(
        seg_q=list(inputs.seg_q_len),
        seg_k=list(inputs.seg_k_len),
        top_k=inputs.top_k,
        ratio=inputs.ratio,
        offsets=inputs.offsets() if inputs.q_causal_offsets is not None else None,
        peaked=False,
        seed=0,
        reps=reps,
        note=label,
        dist="diag",
        bound_seqlen=max_seqlen_k is not None,
    )
    off = dict(knobs)
    off["sample_tiles_max"] = 0
    arms = {
        "default": bcdi.CakeArm(inputs, knobs=knobs, name="default"),
        "sampling_off": bcdi.CakeArm(inputs, knobs=off, name="sampling_off"),
    }
    samples = {a: {"AB": [], "BA": [], "IL": []} for a in arms}
    order = list(arms)
    for a in order:
        samples[a]["AB"] = bcdi.gpu_times(arms[a], reps, warmup)
    for a in reversed(order):
        samples[a]["BA"] = bcdi.gpu_times(arms[a], reps, warmup)
    for _ in range(reps):
        for a in order:
            samples[a]["IL"].extend(bcdi.gpu_times(arms[a], 1, 0))
    outputs = {}
    for a, arm in arms.items():
        out = arm()
        torch.cuda.synchronize()
        outputs[a] = (out[0].clone(), out[1].clone())
    result = {}
    for a in order:
        every = samples[a]["AB"] + samples[a]["BA"] + samples[a]["IL"]
        result[a] = dict(
            gpu_ms={phase: statistics.median(v) for phase, v in samples[a].items()},
            plan=arms[a].plan,
            workspace_bytes=arms[a].workspace_bytes,
        )
        result[a]["gpu_ms"]["all"] = statistics.median(every)
    result["ratio_default_over_off"] = (
        result["default"]["gpu_ms"]["all"] / result["sampling_off"]["gpu_ms"]["all"]
    )
    result["same_bits"] = ref.same_bits(outputs["default"], outputs["sampling_off"])
    result["timer"] = bcdi.timer_record()
    result["reps"] = reps
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_report(label, plan, agg, timing, docs):
    p = plan
    print(
        f"[{label}] plan: arch {p['arch']} program {p['scan_key']} ({p['program']}) block_q {p['block_q']} "
        f"pair {p['pair']} n_split {p['n_split']} grid {p['grid_ctas']} cand_cap {p['cand_cap']} "
        f"check_period {p['check_period']} sample_tiles_max {p['sample_tiles_max']} "
        f"shift {p['sample_shift_permille']} far_sample {p['far_sample']} sample_fit {p['sample_fit']} "
        f"tile order {p['tile_order']} statistic {p['statistic']} "
        f"{'dynamic tickets' if p['dynamic_tickets'] else 'static enumeration'} "
        f"workspace {p['workspace_bytes'] / 2**20:.1f} MiB"
    )
    if docs:
        print(f"[{label}] document lengths: {docs}")
    print(
        f"[{label}] units {agg['units']} sampled {agg['sampled']} failed {agg['failed']} "
        f"({agg['failed_pct_of_sampled']:.1f} % of sampled; {agg['failed_queries']} failing queries); "
        f"key tiles streamed twice: {agg['restream_tile_pct']:.1f} % ({agg['fail_tiles']} of {agg['tiles']})"
    )
    print(f"[{label}] per segment: seg Lq Lk offset | units sampled failed restream%")
    for s in agg["segments"]:
        print(
            f"[{label}]   {s['segment']:3d} {s['Lq']:7d} {s['Lk']:7d} {s['offset']:7d} | "
            f"{s['units']:6d} {s['sampled']:6d} {s['failed']:6d} {s['restream_tile_pct']:6.1f}"
        )
    c = agg["cta"]
    fp = (
        "-"
        if c["first_fail_pos_median"] is None
        else f"{c['first_fail_pos_median']:.2f} / {c['first_fail_pos_p90']:.2f}"
    )
    h = c["hist"]
    print(
        f"[{label}] per CTA ({c['assignment']}): G {c['G']}, CTAs with >= 1 failed unit {c['ctas_with_fail']}, "
        f"max failures on one CTA {c['max_fails_per_cta']}, histogram 0/1/2/3-5/6+ = "
        f"{h['0']}/{h['1']}/{h['2']}/{h['3-5']}/{h['6+']}, first failure at (median / p90 of the CTA's list) {fp}"
    )
    t = ""
    if timing:
        d, o = timing["default"]["gpu_ms"], timing["sampling_off"]["gpu_ms"]
        print(
            f"[{label}] time: default {d['all']:.3f} ms (AB {d['AB']:.3f} / BA {d['BA']:.3f} / IL {d['IL']:.3f}), "
            f"sampling off {o['all']:.3f} ms (AB {o['AB']:.3f} / BA {o['BA']:.3f} / IL {o['IL']:.3f}), "
            f"default / off {timing['ratio_default_over_off']:.4f}, bits identical {timing['same_bits']}, "
            f"timer {timing['timer']['name']}"
        )
        t = (
            f" default_ms={d['all']:.3f} off_ms={o['all']:.3f} "
            f"default/off={timing['ratio_default_over_off']:.4f} bits_identical={timing['same_bits']}"
        )
    print(
        f"SUMMARY {label} arch={p['arch']} program={p['scan_key']} statistic={p['statistic']} "
        f"units={agg['units']} sampled={agg['sampled']} failed={agg['failed']} "
        f"({agg['failed_pct_of_sampled']:.1f}%) restream_tiles={agg['restream_tile_pct']:.1f}% "
        f"ctas_with_fail={c['ctas_with_fail']}/{c['G']} max_fails_per_cta={c['max_fails_per_cta']}"
        f"{' (static approx.)' if p['dynamic_tickets'] else ''}{t}",
        flush=True,
    )


def write_csv(path, records, G):
    fields = [
        "unit",
        "cta_static",
        "segment",
        "blocks",
        "item",
        "n_tiles",
        "stride",
        "n_sample",
        "queries",
        "fired",
        "failed_queries",
        "fail",
        "adm_min_over_k",
        "adm_max_over_k",
        "c_mean",
        "r_shift_max",
    ]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in records:
            row = dict(r)
            row["cta_static"] = r["unit"] % G
            row["blocks"] = " ".join(str(b) for b in r["blocks"])
            row["fail"] = int(r["fail"])
            wr.writerow({k: row[k] for k in fields})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--load", help="torch.save dict of one rank's inputs")
    src.add_argument("--row", help="a row of benchmarks/bench_cake_dsa_indexer.py")
    src.add_argument(
        "--cp16",
        choices=sorted(CP16_MIXES),
        help="synthetic 16-way zigzag document mix",
    )
    parser.add_argument("--dist", choices=("iid", "sink", "topic"), default="sink")
    parser.add_argument(
        "--rank", type=int, default=0, help="zigzag rank 0 .. 15 (--cp16)"
    )
    parser.add_argument(
        "--top-k", type=int, default=ref.DEFAULT_TOP_K, help="top-k of a --cp16 row"
    )
    parser.add_argument("--save", help="write the built inputs in the --load format")
    parser.add_argument(
        "--label", help="name of the call in the report (default: derived)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--arch",
        choices=sorted(DEFAULT_NUM_SMS),
        help="plan for this architecture (default: the device's; required on CPU)",
    )
    parser.add_argument(
        "--num-sms", type=int, help="persistent grid (default: the device's SM count)"
    )
    parser.add_argument(
        "--knob", action="append", default=[], help="NAME=VALUE plan knob override"
    )
    parser.add_argument(
        "--statistic",
        choices=("auto", "plain", "near", "diag"),
        default="auto",
        help="rank statistic to replay (auto: the program's; the others for comparison)",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="also time default vs sample_tiles_max=0 (GPU)",
    )
    parser.add_argument(
        "--reps", type=int, default=10, help="timed iterations per phase"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--chunk-rows", type=int, default=192, help="query rows scored per chunk"
    )
    parser.add_argument(
        "--score-bytes",
        type=int,
        default=1 << 29,
        help="bytes of head logits per matmul",
    )
    parser.add_argument("--csv", help="per-unit rows")
    parser.add_argument(
        "--json", help="plan, totals, segments, CTA distribution, timing, units"
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore", ExperimentalWarning)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    knobs = parse_knobs(args.knob)
    docs = None
    max_seqlen_k = None
    if args.load:
        label = args.label or Path(args.load).stem
        inputs, max_seqlen_k = load_inputs(args.load, device, label)
    elif args.row:
        label = args.label or args.row
        if args.row not in bcdi.ROWS:
            raise SystemExit(f"unknown row {args.row!r}; known: {sorted(bcdi.ROWS)}")
        inputs = bcdi.build_inputs(args.row, device, False)
        row = bcdi.ROWS[args.row]
        max_seqlen_k = max(row["seg_k"]) if row["bound_seqlen"] else None
    else:
        if not 0 <= args.rank < CP_PARTS:
            raise SystemExit(f"--rank must be in 0 .. {CP_PARTS - 1}")
        label = args.label or f"{args.cp16}_{args.dist}_rank{args.rank}"
        inputs = build_cp16_inputs(
            args.cp16, args.dist, args.rank, device, args.top_k, label
        )
        docs = list(CP16_MIXES[args.cp16])
        max_seqlen_k = max(inputs.seg_k_len)
    if args.save:
        save_inputs(inputs, max_seqlen_k, args.save)
        print(f"[{label}] wrote {args.save}")
    if args.arch is None:
        arch = cake_backend.arch_for(device) if device.type == "cuda" else None
        if arch is None:
            raise SystemExit(
                "--arch is required when the device is not a supported GPU"
            )
    else:
        arch = args.arch
    if args.num_sms is not None:
        num_sms = args.num_sms
    elif device.type == "cuda":
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        num_sms = DEFAULT_NUM_SMS[arch]
    plan = host_plan(inputs, arch, num_sms, knobs, max_seqlen_k)
    if args.statistic != "auto":
        plan["statistic"] = args.statistic
    print(
        f"[{label}] T={inputs.num_queries} Tkv={inputs.num_keys} segments={inputs.num_segments} "
        f"top_k={inputs.top_k} ratio={inputs.ratio} device={device}",
        flush=True,
    )
    t0 = time.time()
    records = replay(inputs, plan, args, device)
    seconds = time.time() - t0
    agg = aggregate(records, plan, inputs)
    timing = None
    if args.time:
        if device.type != "cuda":
            raise SystemExit("--time needs a CUDA device")
        if (
            arch != cake_backend.arch_for(device)
            or num_sms != torch.cuda.get_device_properties(device).multi_processor_count
        ):
            raise SystemExit(
                "--time runs the kernel on this device: do not combine it with --arch / --num-sms"
            )
        timing = time_default_vs_off(
            inputs, label, max_seqlen_k, args.reps, args.warmup, knobs
        )
    print_report(label, plan, agg, timing, docs)
    print(f"[{label}] replay {seconds:.1f} s", flush=True)
    if args.csv:
        write_csv(args.csv, records, agg["cta"]["G"])
        print(f"[{label}] wrote {args.csv}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                dict(
                    label=label,
                    inputs=dict(
                        T=inputs.num_queries,
                        Tkv=inputs.num_keys,
                        segments=inputs.num_segments,
                        seg_q=list(inputs.seg_q_len),
                        seg_k=list(inputs.seg_k_len),
                        offsets=inputs.offsets(),
                        top_k=inputs.top_k,
                        ratio=inputs.ratio,
                        max_seqlen_k=max_seqlen_k,
                        docs=docs,
                    ),
                    knobs=knobs,
                    plan=plan,
                    totals={
                        k: v for k, v in agg.items() if k not in ("segments", "cta")
                    },
                    segments=agg["segments"],
                    cta=agg["cta"],
                    timing=timing,
                    replay_seconds=seconds,
                    device=str(device),
                    torch=torch.__version__,
                    units=records,
                ),
                indent=1,
            )
        )
        print(f"[{label}] wrote {args.json}")


if __name__ == "__main__":
    main()
