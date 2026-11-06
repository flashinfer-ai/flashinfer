# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The tensor-parallel protocol of the batch-invariant sampling class ``lse.row``
(``flashinfer.sampling.log_probs_tp_partials`` / ``log_probs_tp_merge`` / ``log_probs_tp`` / ``log_softmax_tp`` /
``softmax_tp``; ``flashinfer/batch_invariant/README.md``, Sampling: the tensor-parallel protocol).

Device-free: the wire format (``shards.ShardLayout``) is exactly the maximal in-shard sub-trees of the row trees; the
partials kernel's program, traced over every unit of every configuration, stores each of them as the canonical sub-tree
term; the merge kernel's program rebuilds the row term node for node; the two-round protocol is the spec's ``(m, s)`` as a
term identity and the one-round online merge is NOT the class; the production shard layouts (incl. Megatron's padded
vocabularies) are admitted and their widths pinned; the dispatcher picks members of the space; the registry entries carry
the class's fingerprint; the kernels' PTX uses only the declared forms. On an SM100 / SM103 / SM107 device with the CuTe
DSL (the ranks simulated on one device, ``tp_api.tp_simulate``: the all-gather is a stack, the all-reduce a sum): the
merged ``(m, s)`` of TP in {2, 4, 8} == the TP = 1 protocol's == the exact CPU model, for the production vocabularies
and Megatron's padded ones; every sharded output == the full-row kernel's bits; every configuration of the partials
space writes the same partials tensor; batch invariance per shard; masked / NaN / inf rows, a partial last slot, empty
shards, M = 0, every temperature form, CUDA-graph replay and PDL, the ``+0`` filler of ``log_probs_tp``, the error
contract.
"""

import glob
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.sampling import reference as R
from flashinfer.batch_invariant.sampling import shards
from flashinfer.batch_invariant.sampling import sterm as term
from flashinfer.batch_invariant.sampling import tp_api
from flashinfer.batch_invariant.sampling.api import LSE_ROW_ARCHS
from flashinfer.batch_invariant.sampling.schedule import (
    LANES,
    V,
    device_sm_count,
    max_vec_elems,
    padded_blocks,
)
from flashinfer.batch_invariant.sampling.shards import ShardLayout
from flashinfer.batch_invariant.sampling.spec import (
    LSESpec,
    max_reduction,
    sum_reduction,
)
from flashinfer.batch_invariant.sampling.tp_schedule import (
    TPConfig,
    all_valid_tp_configs,
    tp_config,
)

DT = [torch.bfloat16, torch.float16, torch.float32]
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
PROD_VOCABS = [32000, 32768, 128256, 151936, 152064, 262144]
TPS = [1, 2, 4, 8]
SMALL_NS = [8, 64, 100, 300, 1000, 2056, 4097]
#: Small tilings for the device suite: two shards of an odd-block vocabulary, four equal shards, a partial last slot, a
#: one-slot first shard, and eight shards of 32000.
SMALL = [
    (1000, [(0, 504), (504, 1000)]),
    (4096, shards.tp_shards(4096, 4)),
    (4097, [(0, 2048), (2048, 4097)]),
    (2056, [(0, 8), (8, 1032), (1032, 2056)]),
    (32000, shards.tp_shards(32000, 8)),
]
#: (dtype, N, bounds): every tiling for bf16, the first three for f16 / fp32.
SMALL_DT = [
    (dt, N, b) for dt in DT for (N, b) in (SMALL if dt == torch.bfloat16 else SMALL[:3])
]
PROD = [(N, tp) for N in PROD_VOCABS for tp in TPS]
#: Megatron's ``make_vocab_size_divisible_by=128`` paddings of 152064 / 128256.
PADDED = [(152576, 8), (129024, 8), (128512, 4)]

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), LSE_ROW_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def _ids(v):
    return str(v) if isinstance(v, int) else f"R{len(v)}"


# ----------------------------------------------------------------------------------------------- device-free: the layout


def _bounds_cases():
    out = []
    for N in SMALL_NS:
        for tp in (1, 2, 4):
            if N % tp == 0 and (N // tp) % V == 0:
                out.append((N, shards.tp_shards(N, tp)))
        s = N // V  # irregular slot-aligned tilings
        if s >= 3:
            out.append(
                (
                    N,
                    [
                        (0, 8),
                        (8, (s // 3) * 8),
                        ((s // 3) * 8, (s - 1) * 8),
                        ((s - 1) * 8, N),
                    ],
                )
            )
            out.append((N, [(0, (s - 1) * 8), ((s - 1) * 8, N)]))
    return out


@pytest.mark.parametrize("N,bounds", _bounds_cases(), ids=_ids)
def test_layout_positions_are_the_maximal_subtrees(N, bounds):
    """The dyadic layout lists exactly the positions ``shard_partials`` enumerates (its lane-level nodes aside, which only
    a whole-row shard without padding has), every piece is in-shard with a parent that is not, the pieces of a lane tile
    its block range, and the slot table is a bijection onto ``[0, P)``."""
    spec = LSESpec(N)
    red = spec.max_reduction(0)
    for lo, hi in bounds:
        lay = ShardLayout(lo, hi, N)
        ref = shards.shard_partials(red, spec.alg, lo, hi, N)
        ref_block = {k for k in ref if k[0] == "block"}
        for _, L, i in [k for k in ref if k[0] == "lane"]:
            for lane in range(i << L, (i + 1) << L):
                if lay.ranges[lane][0] < lay.ranges[lane][1]:
                    ref_block.add(("block", lane, lay.kb, 0))
        got = set(lay.positions())
        assert got == ref_block, (lo, hi, N, sorted(got ^ ref_block)[:6])
        for _, lane, L, i in got:
            assert lay.is_piece(lane, L, i)
        for lane in range(LANES):
            a, b = lay.ranges[lane]
            covered = sorted(
                x for (L, i) in lay.pieces[lane] for x in range(i << L, (i + 1) << L)
            )
            assert covered == list(range(a, b))
            sides = [(L, shards.piece_side(i)) for (L, i) in lay.pieces[lane]]
            assert len(sides) == len(set(sides))
        tab = lay.slot_table()
        slots = sorted(
            tab[lane][L][s]
            for lane in range(LANES)
            for L in range(lay.kb + 1)
            for s in range(2)
            if tab[lane][L][s] >= 0
        )
        assert slots == list(range(lay.P))


def test_dyadic_pieces_examples():
    assert shards.dyadic_pieces(297, 594) == [
        (0, 297),
        (1, 149),
        (2, 75),
        (4, 19),
        (6, 5),
        (7, 3),
        (6, 8),
        (4, 36),
        (1, 296),
    ]
    assert shards.dyadic_pieces(0, 297) == [(8, 0), (5, 8), (3, 36), (0, 296)]
    assert shards.dyadic_pieces(5, 5) == [] and shards.dyadic_pieces(0, 1024) == [
        (10, 0)
    ]
    assert shards.lane_block_range(19008, 38016, 0) == (75, 149)
    assert shards.lane_block_range(19008, 38016, 16) == (74, 148)


@pytest.mark.parametrize("N", PROD_VOCABS)
@pytest.mark.parametrize("tp", TPS)
def test_production_layouts_admitted_and_bounded(N, tp):
    """Every production shard is admitted; its width is a function of the shard alone and bounded by the tree's depth;
    the widths of the protocol are pinned (a change here is a change of the wire format)."""
    bounds = shards.tp_shards(N, tp)
    shards.check_bounds(bounds, N)
    Kp = padded_blocks(N)
    for lo, hi in bounds:
        lay = ShardLayout(lo, hi, N)
        assert LANES * (2 * Kp.bit_length() + 1) >= lay.P
        assert shards.partials_width(lo, hi, N) == lay.P
        assert lay.P >= LANES or hi - lo < LANES * V
    widths = [shards.partials_width(lo, hi, N) for lo, hi in bounds]
    pins = {
        (152064, 8): [104, 208, 264, 160, 248, 248, 128, 168],
        (128256, 8): [180, 192, 188, 180, 192, 204, 180, 192],
        (32768, 8): [32] * 8,
        (262144, 2): [32, 32],
        (151936, 4): [108, 228, 304, 188],
        (152064, 1): [128],
        (32768, 1): [32],
    }
    if (N, tp) in pins:
        assert widths == pins[(N, tp)], (N, tp, widths)
    assert tp_api.max_partials_width(bounds, N) == max(widths)


def test_megatron_padded_shards():
    """``make_vocab_size_divisible_by=128``: the padded vocabulary is the class's N; its equal shards are slot aligned."""
    assert shards.tp_shards_padded(152064, 8) == (152576, shards.tp_shards(152576, 8))
    assert shards.tp_shards_padded(128256, 8) == (129024, shards.tp_shards(129024, 8))
    assert shards.tp_shards_padded(128256, 4) == (128512, shards.tp_shards(128512, 4))
    assert shards.tp_shards_padded(151936, 2)[0] == 152064  # Qwen: padded to 152064
    for N, tp in PADDED:
        shards.check_bounds(shards.tp_shards(N, tp), N)


def test_bounds_validation_and_error_contract():
    """The layout rejects shards off the slot grid and tilings with gaps / overlaps; the entries raise
    :class:`BatchInvariantUnsupportedError` for them (and for a CPU tensor) before any kernel; the operators accept only
    ``batch_invariant=True`` and CUDA tensors."""
    with pytest.raises(ValueError):
        ShardLayout(4, 16, 32)
    with pytest.raises(ValueError):
        ShardLayout(0, 40, 32)
    for bad in ([(0, 8), (16, 32)], [(0, 8), (8, 24)], [(0, 12), (12, 32)]):
        with pytest.raises(ValueError):
            shards.check_bounds(bad, 32)
    shards.check_bounds([(0, 0), (0, 32), (32, 32)], 32)  # empty shards are admitted
    x = torch.zeros(2, 512, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        tp_api.log_probs_tp_partials_batch_invariant(x, 4, 516, 1024)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        tp_api.log_probs_tp_partials_batch_invariant(x, 0, 512, 1024)  # a CPU tensor
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        tp_api.max_partials_width([(0, 512), (520, 1024)], 1024)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        tp_api.log_probs_tp_merge_batch_invariant(
            torch.zeros(2, 2, 40), [(0, 512), (520, 1024)], 1024, kind="max"
        )
    with pytest.raises(ValueError, match="batch_invariant"):
        flashinfer.sampling.log_probs_tp_partials(
            x, 0, 512, 1024, batch_invariant=False
        )
    with pytest.raises(ValueError, match="batch_invariant"):
        flashinfer.sampling.log_probs_tp_merge(
            torch.zeros(2, 2, 40),
            [(0, 512), (512, 1024)],
            1024,
            kind="max",
            batch_invariant=False,
        )
    m = torch.zeros(2)
    for op in (
        lambda: flashinfer.sampling.log_probs_tp(
            x,
            torch.zeros(2, dtype=torch.int64),
            0,
            512,
            1024,
            m,
            m,
            batch_invariant=False,
        ),
        lambda: flashinfer.sampling.log_softmax_tp(
            x, 0, 512, 1024, m, m, batch_invariant=False
        ),
        lambda: flashinfer.sampling.softmax_tp(
            x, 0, 512, 1024, m, m, batch_invariant=False
        ),
    ):
        with pytest.raises(ValueError, match="batch_invariant"):
            op()
    with pytest.raises(ValueError, match="CUDA"):
        flashinfer.sampling.log_probs_tp_partials(x, 0, 512, 1024)
    with pytest.raises(ValueError, match="CUDA"):
        flashinfer.sampling.log_probs_tp_merge(
            torch.zeros(2, 2, 40), [(0, 1024)], 1024, kind="max"
        )


def test_registry_entries_carry_the_class_fingerprint():
    """The TP entries are the class ``lse.row`` with the class's fingerprint: the partials are its own sub-trees and the
    merge rebuilds its trees; the wire format is a layout, not a class."""
    for op in (
        "log_probs_tp_partials",
        "log_probs_tp_merge",
        "log_probs_tp",
        "log_softmax_tp",
        "softmax_tp",
    ):
        assert bi.CLASSES[op].name == "lse.row"
        for N in (32768, 152064):
            for temp in (False, True):
                assert bi.fingerprint(
                    op, N=N, dtype="bf16", temperature=temp
                ) == bi.fingerprint("log_probs", N=N, dtype="bf16", temperature=temp)
    assert bi.fingerprint(
        "log_probs_tp", N=152064, dtype="bf16", temperature=False
    ) != bi.fingerprint("log_probs_tp", N=152576, dtype="bf16", temperature=False)


# ------------------------------------------------------------------------------- device-free: the partials kernel's program


def test_pairwise_nodes_are_the_digit_reduce():
    alg = term.TermAlgebra()
    red = max_reduction(4096, lambda fp, i: fp.x(0, i))
    db = red.digit("block")
    vals = [alg.x(1, i) for i in range(8)]
    levels = shards.pairwise_nodes(db, alg, vals)
    assert (
        len(levels) == 4 and levels[-1][0] is db.reduce(alg, vals) and levels[0] == vals
    )
    for L in range(4):
        for i, v in enumerate(levels[L]):
            assert v is db.reduce(alg, vals[i << L : (i + 1) << L])
    vals2 = [alg.x(2, 0), None, None, alg.x(2, 3)]
    lv = shards.pairwise_nodes(db, alg, vals2)
    assert lv[1] == [alg.x(2, 0), alg.x(2, 3)] and lv[2][0] is db.combine(
        alg, alg.x(2, 0), alg.x(2, 3)
    )
    nodes = shards.aligned_group_nodes(db, alg, vals, 0)
    assert nodes[(3, 0)] is levels[3][0] and nodes[(1, 2)] is levels[1][1]
    nodes = shards.aligned_group_nodes(
        db, alg, vals, 1
    )  # first unit at an odd global index
    assert (1, 0) not in nodes and nodes[(1, 1)] is db.combine(alg, vals[1], vals[2])
    assert (2, 1) not in nodes and (2, 3) in nodes


def _layouts_for(N):
    out = []
    for tp in (1, 2, 4, 8):
        if N % tp == 0 and (N // tp) % V == 0:
            out += [ShardLayout(lo, hi, N) for lo, hi in shards.tp_shards(N, tp)]
    s = N // V
    if s >= 3:
        out.append(ShardLayout(8, (s - 1) * 8, N))
    return out


@pytest.mark.parametrize("N", [64, 300, 1000, 2056, 4097])
def test_partials_program_stores_the_canonical_subtrees(N):
    """Every configuration's program (chunk nodes, the in-CTA aligned groups, the cluster groups with a residue) stores
    exactly the layout's pieces as the canonical sub-tree terms, for both reductions."""
    spec = LSESpec(N)
    for lay in _layouts_for(N):
        cfgs = all_valid_tp_configs(lay, 2)
        assert cfgs, lay.describe()
        for cfg in cfgs:
            st = cfg.structure(lay)
            for red in (spec.max_reduction(0), spec.sum_reduction(0)):
                shards.verify_partials(red, spec.alg, lay, st)


def test_structure_alignment_and_cover():
    lay = ShardLayout(19008, 38016, 152064)  # rank 1 of 152064 / 8: blocks [74, 149)
    assert lay.block_span == (74, 149)
    st = shards.TPStructure.resolve(lay, 8, 2)
    assert (st.kpw, st.A_lo, st.Kv) == (8, 64, 128) and st.cta_residue == 1
    st = shards.TPStructure.resolve(lay, 4, 4)
    assert (st.kpw, st.A_lo, st.Kv) == (8, 64, 128) and st.cta_residue == 2
    st = shards.TPStructure.resolve(lay, 32, 1)
    assert (st.kpw, st.A_lo, st.Kv) == (8, 0, 256)
    with pytest.raises(ValueError):
        shards.TPStructure(lay, 1, 8, 1)


# ---------------------------------------------------------------------------------- device-free: the merge kernel's program


@pytest.mark.parametrize("N,bounds", _bounds_cases(), ids=_ids)
def test_merge_program_rebuilds_the_row_term(N, bounds):
    spec = LSESpec(N)
    for red in (spec.max_reduction(0), spec.sum_reduction(0)):
        shards.verify_merge(red, spec.alg, bounds, N)


@pytest.mark.parametrize(
    "N,tp",
    [
        (152064, 8),
        (128256, 8),
        (151936, 4),
        (262144, 8),
        (32768, 2),
        (32000, 8),
        (152576, 8),
        (129024, 8),
    ],
)
def test_merge_program_rebuilds_the_row_term_production(N, tp):
    """The production layouts: the merge of the shards' pieces is the row term (the max tree; the sum tree has the same
    shape), and the lane classes are few."""
    spec = LSESpec(N)
    bounds = shards.tp_shards(N, tp)
    shards.verify_merge(spec.max_reduction(0), spec.alg, bounds, N)
    layouts, _cls, progs = shards.merge_programs(bounds, N)
    assert len(progs) <= len(bounds) + 1, (N, tp, len(progs))
    n_loads = [sum(1 for op in ops if op[0] == "load") for ops, _ in progs]
    assert max(n_loads) == max(
        sum(len(lay.pieces[lane]) for lay in layouts) for lane in range(LANES)
    )


def test_end_to_end_two_round_protocol_is_the_spec():
    """Round 1 merges to the spec's maximum term; round 2's partials over ``ex2((x - m) c)`` with that ``m`` merge to the
    spec's sum term; the chains on them are the spec's outputs (term identity, not just equivalence)."""
    N, bounds = 1000, [(0, 248), (248, 504), (504, 752), (752, 1000)]
    spec = LSESpec(N, temperature=True)
    red_m = spec.max_reduction(0)
    parts = [
        shards.shard_block_partials(red_m, spec.alg, lo, hi, N) for lo, hi in bounds
    ]
    m = shards.merge_partials(red_m, spec.alg, parts, N)
    assert m is spec.m(0)
    red_s = sum_reduction(
        N,
        lambda fp, i: spec.alg.ex2(
            spec.alg.mul(spec.alg.add(spec.leaf(0, i), spec.alg.neg(m)), spec.c())
        ),
    )
    parts_s = [
        shards.shard_block_partials(red_s, spec.alg, lo, hi, N) for lo, hi in bounds
    ]
    s = shards.merge_partials(red_s, spec.alg, parts_s, N)
    assert s is spec.s(0)
    assert spec.lse(0) is spec.alg.mul(
        spec.alg.add(spec.alg.mul(m, spec.c()), spec.alg.op("lg2", s)),
        spec.alg.const(0.6931471805599453),
    )


def test_online_merge_is_not_the_class():
    """The protocol decision: a one-round merge of per-shard ``(m_r, s_r)`` with the online-softmax rescale builds a
    different term from the class's sum over the global maximum (so it cannot be bit-identical by construction) -- the
    second round is necessary."""
    N, bounds = 300, [(0, 152), (152, 300)]
    spec = LSESpec(N)
    alg = spec.alg
    m = spec.m(0)
    c = spec.c()
    shard_m, shard_s = [], []
    for lo, hi in bounds:
        red_m = max_reduction(N, lambda fp, i: fp.x(0, i) if lo <= i < hi else None)
        m_r = red_m.canonical(alg)
        red_s = sum_reduction(
            N,
            lambda fp, i, m_r=m_r: (
                fp.ex2(fp.mul(fp.add(fp.x(0, i), fp.neg(m_r)), c))
                if lo <= i < hi
                else None
            ),
        )
        shard_m.append(m_r)
        shard_s.append(red_s.canonical(alg))
    online = alg.add(
        alg.mul(shard_s[0], alg.ex2(alg.mul(alg.add(shard_m[0], alg.neg(m)), c))),
        alg.mul(shard_s[1], alg.ex2(alg.mul(alg.add(shard_m[1], alg.neg(m)), c))),
    )
    assert online is not spec.s(0)
    assert term.sexpr(online) != term.sexpr(spec.s(0))


# ---------------------------------------------------------------------------------------------- device-free: the space


@pytest.mark.parametrize("elem_bytes", [2, 4])
def test_tp_dispatcher_is_a_member_of_the_space(elem_bytes):
    """Every pick is a member of the enumerated space at every SM count, also under the narrow vector width a column view
    forces (``vec_max``); the description round-trips; the structure resolves."""
    for N in [1000, 4096, 32768, 128256, 152064, 262144, 152576]:
        for tp in (1, 2, 4, 8):
            if N % tp or (N // tp) % V:
                continue
            for lo, hi in shards.tp_shards(N, tp)[:: max(1, tp // 2)]:
                lay = ShardLayout(lo, hi, N)
                space = all_valid_tp_configs(lay, elem_bytes) + all_valid_tp_configs(
                    lay, elem_bytes, 1
                )
                for M in (1, 7, 128, 4096):
                    for sm in (148, 152, 204, 224):
                        cfg = tp_config(lay, M, elem_bytes, sm)
                        narrow = tp_config(lay, M, elem_bytes, sm, vec_max=1)
                        assert narrow in space and narrow.vec_elems == 1
                        assert cfg in space, (N, tp, lo, M, sm, cfg.describe())
                        assert TPConfig.parse(cfg.describe()) == cfg
                        cfg.structure(lay)


def test_tp_config_structure_threads_cover_the_shard():
    lay = ShardLayout(0, 19008, 152064)
    for cfg in all_valid_tp_configs(lay, 2):
        st = cfg.structure(lay)
        B_lo, B_hi = lay.block_span
        assert (
            st.A_lo <= B_lo
            and st.A_lo + st.Kv >= B_hi
            and st.A_lo % (st.kpw * st.W) == 0
        )
        assert cfg.threads_per_row == 32 * st.W and cfg.cluster == st.C


# --------------------------------------------------------------------------------------------- device-free: the IR pin
# (kind of kernel, dtype, shard (lo, hi, N) or bounds, config tuple, round / mode, per-row temperature)
IR_CASES = [
    (
        "partials",
        "bf16",
        (19008, 38016, 152064),
        (256, 1, 8, 2),
        "max",
        False,
    ),  # rank 1 of 152064 / 8, a 2-CTA cluster
    (
        "partials",
        "bf16",
        (19008, 38016, 152064),
        (128, 1, 8, 4),
        "sum",
        True,
    ),  # round 2: the ex2 chain, per-row T
    (
        "partials",
        "f16",
        (2048, 4097, 4097),
        (32, 1, 1, 1),
        "max",
        False,
    ),  # a partial last slot, one warp
    ("merge", None, tuple(shards.tp_shards(152064, 8)), None, "max", False),
    (
        "merge",
        None,
        ((0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)),
        None,
        "sum",
        False,
    ),  # one lane class
    ("outputs", "bf16", (19008, 38016, 152064), None, "logp", True),
]
# The admitted PTX forms: the `.rn` arithmetic incl. the mixed-precision forms the backend builds from an exact widening +
# a `.rn` op, the exact conversions and selects, the two declared MUFU instructions, and `max.bf16` / `max.f16`: the
# backend narrows `max.f32(widen(a), widen(b))` to the 16-bit max, value-identical (the exact widening is monotone) and
# different at most in the sign of a zero maximum and the NaN choice, neither of which reaches an output bit of the class.
PTX_FP_ALLOWED = re.compile(
    r"^(fma\.rn\.f32(\.bf16|\.f16)?|add\.rn\.f32(\.bf16|\.f16)?|sub\.rn\.f32(\.bf16|\.f16)?|mul\.rn\.f32|div\.rn\.f32"
    r"|cvt\.f32\.(bf16|f16)|cvt\.rn\.(bf16|f16)\.f32|cvt\.rn\.f32\.s32|cvt\.rn\.f32\.s64|cvt\.f32\.f16"
    r"|max\.(f32|bf16|f16)|min\.f32|abs\.f32|neg\.f32|mov\.(b32|f32)|selp\.f32|setp\.[a-z]+\.(f32|bf16|f16)"
    r"|ex2\.approx\.ftz\.f32|lg2\.approx\.ftz\.f32)$"
)
PTX_FP_LIKE = re.compile(
    r"^(fma|add|sub|mul|div|sqrt|rsqrt|rcp|ex2|lg2|cvt|max|min|abs|neg|selp|setp|fns|mad|sin|cos|tanh)\."
)


@pytest.mark.parametrize(
    "what,dtype,shard,cfg,which,trow", IR_CASES, ids=lambda v: str(v)[:40]
)
def test_tp_ir_uses_only_declared_forms(tmp_path, what, dtype, shard, cfg, which, trow):
    """Compile one kernel per code path with no GPU visible and check the PTX: the same admitted forms as the row kernels
    (``.rn`` arithmetic, the two MUFU instructions where the chains need them, ``max.f32``), the cluster exchange only in
    cluster members, no divisions except the per-row temperature's and the softmax's reciprocal, the merge kernel's five
    lane shuffles, no fast-math anywhere."""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_KEEP_IR="1",
        CUTE_DSL_KEEP="ptx",
        FLASHINFER_CUTE_DSL_DISABLE_CACHE="1",
        CUTE_DSL_ARCH=os.environ.get("BI_TEST_ARCH", "sm_100a"),
    )
    pre = (
        "from flashinfer.batch_invariant.sampling.shards import ShardLayout\n"
        "from flashinfer.batch_invariant.sampling.tp_schedule import TPConfig\n"
        "from flashinfer.batch_invariant.sampling.tp_kernel import get_merge_kernel, get_outputs_kernel, get_partials_kernel\n"
    )
    if what == "partials":
        code = (
            pre
            + f"get_partials_kernel({dtype!r}, ShardLayout{shard}, TPConfig{cfg}, {which!r}, {trow}, False)\n"
        )
    elif what == "merge":
        code = pre + f"get_merge_kernel({shard!r}, {shard[-1][1]}, {which!r}, False)\n"
    else:
        vec = 8 if dtype != "f32" else 4
        code = (
            pre
            + f"get_outputs_kernel({dtype!r}, ShardLayout{shard}, {which!r}, {trow}, {vec}, False)\n"
        )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    mlir = glob.glob(str(tmp_path / "*.mlir"))
    assert mlir, "no MLIR dumped"
    ir = Path(mlir[0]).read_text()
    assert not re.search(r"arith\.(addf|mulf|divf|subf|fmaf)\b", ir), (
        "arith float op present"
    )
    assert not re.search(r"\bmath\.", ir), "math-dialect op present"
    assert (
        "fastmath<" not in ir
        and "rsqrt" not in ir
        and "div.approx" not in ir
        and "rcp.approx" not in ir
    )
    ptxs = glob.glob(str(tmp_path / "*.ptx"))
    assert ptxs, "no PTX dumped"
    ptx = Path(ptxs[0]).read_text()
    ops = set()
    for line in ptx.splitlines():
        tok = line.strip().split()
        if not tok or tok[0].startswith(("//", ".", "@", "{", "}", "bra", "ret")):
            continue
        if PTX_FP_LIKE.match(tok[0].rstrip(";")):
            ops.add(tok[0].rstrip(";"))
    bad = sorted(
        o
        for o in ops
        if not PTX_FP_ALLOWED.match(o)
        and not re.search(r"\.(b|u|s)(8|16|32|64)(\.|$)", o)
        and not re.match(r"^(cvt\.rzi|mul\.(wide|lo|hi))", o)
    )
    assert not bad, f"PTX contains non-admitted floating-point forms: {bad}"
    assert not re.search(r"rcp\.|div\.approx|div\.full|sqrt\.", ptx)
    assert "ftz" not in ptx.replace("ex2.approx.ftz.f32", "").replace(
        "lg2.approx.ftz.f32", ""
    )
    if what == "partials":
        C = cfg[3]
        assert ("barrier.cluster" in ptx) == (C > 1)
        assert ("mapa.shared::cluster" in ptx) == (C > 1) and (
            "ld.shared::cluster" in ptx
        ) == (C > 1)
        if C > 1:
            assert (
                ptx.count("ld.shared::cluster.f32") == C
            )  # the C CTA roots, once (one round per launch)
        W = cfg[0] // 32
        assert ir.count("nvvm.barrier") == (
            1 if W > 1 else 0
        )  # one exchange: one barrier after the writes
        assert (
            ir.count("nvvm.shfl.sync") == 0
        )  # the lane tree is the merge's, never a shard's
        assert ("ex2.approx.ftz.f32" in ptx) == (
            which == "sum"
        ) and "lg2.approx" not in ptx
        if which == "max":
            assert re.search(r"max\.(f32|bf16|f16)", ptx) is not None
        else:
            assert "add.rn.f32" in ptx
        assert ptx.count("div.rn.f32") == (1 if trow else 0)
    elif what == "merge":
        n_classes = len(shards.merge_programs(list(shard), shard[-1][1])[2])
        # the lane butterfly; the warp-mapped form (several lane classes) has the staging and roots hand-offs
        assert ir.count("nvvm.shfl.sync") == 5 and ir.count("nvvm.barrier") == (
            2 if n_classes > 1 else 0
        ), n_classes
        assert "barrier.cluster" not in ptx and "mapa" not in ptx
        assert ("max.f32" in ptx) == (which == "max") and ("add.rn.f32" in ptx) == (
            which == "sum"
        )
        assert (
            "ex2.approx" not in ptx and "lg2.approx" not in ptx and "div.rn" not in ptx
        )
    else:
        assert ("lg2.approx.ftz.f32" in ptx) == (which != "probs")
        assert ("ex2.approx.ftz.f32" in ptx) == (which == "probs")
        assert ptx.count("div.rn.f32") == (1 if which == "probs" else 0) + (
            1 if trow else 0
        )
        assert ir.count("nvvm.shfl.sync") == 0 and "barrier" not in ptx


# ------------------------------------------------------------------------------------------------------- on a device


def logits(M, N, dtype, seed=0, kind="dense"):
    """``dense``: N(0, 3) logits (the sum partials carry hundreds of non-trivial terms); ``adv``: magnitudes over ~40
    binades with random signs, near-cancelling pairs, exact +0 / -0, subnormals and outliers, scaled into a logits-like
    range. Built on the CPU, then moved."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "dense":
        return (torch.randn(M, N, generator=g) * 3).to(dtype).cuda()
    exp_span = 20.0 if dtype != torch.float16 else 6.0
    e = (torch.rand(M, N, generator=g) * 2 - 1) * exp_span
    mant = torch.rand(M, N, generator=g) + 1.0
    sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
    x = sign * mant * torch.exp2(e)
    pairs = torch.rand(M, N // 2, generator=g) < 0.15
    idx = torch.arange(0, N - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, N // 2, generator=g)),
        x[:, idx + 1],
    )
    r = torch.rand(M, N, generator=g)
    outlier = 2.0**12 if dtype != torch.float16 else 2.0**4
    x = torch.where(r < 0.01, x * outlier, x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = torch.where((r >= 0.03) & (r < 0.035), x * 2.0**-130, x)
    x = (x.to(dtype).float() * 2.0**-10).to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def bits(t):
    t = t.detach().contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(bits(a.cpu()), bits(b.cpu()))


def same_or_nan(a, b) -> bool:
    """Bitwise equal, NaN payloads aside (out of scope: both sides NaN counts as equal)."""
    a, b = a.cpu(), b.cpu()
    if a.shape != b.shape:
        return False
    nan = torch.isnan(a) & torch.isnan(b)
    return torch.equal(bits(a)[~nan], bits(b)[~nan]) and torch.equal(
        torch.isnan(a), torch.isnan(b)
    )


def ids_for(M, N, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, N, (M,), generator=g).cuda()


# -- the session precompile: the device suite's kernels are compiled device-free in parallel subprocesses into the module
# cache the GPU process then loads from (one process compiles ~7-20 s per kernel; the suite needs about a thousand)
_WORKER_CODE = r"""
import ast, sys
from flashinfer.batch_invariant.sampling.shards import ShardLayout
from flashinfer.batch_invariant.sampling.tp_kernel import get_merge_kernel, get_outputs_kernel, get_partials_kernel
from flashinfer.batch_invariant.sampling.tp_schedule import TPConfig
bad = []
for s in ast.literal_eval(sys.argv[1]):
    try:
        if s[0] == "partials":
            _, ds, lo, hi, N, cfg, kind, trow, pdl = s
            get_partials_kernel(ds, ShardLayout(lo, hi, N), TPConfig(*cfg), kind, trow, pdl)
        elif s[0] == "merge":
            _, bounds, N, kind, pdl = s
            get_merge_kernel(tuple(tuple(b) for b in bounds), N, kind, pdl)
        else:
            _, ds, lo, hi, N, mode, trow, vec, pdl = s
            get_outputs_kernel(ds, ShardLayout(lo, hi, N), mode, trow, vec, pdl)
    except Exception as e:  # noqa: BLE001
        bad.append((s, repr(e)[:300]))
print("BAD", repr(bad))
"""
ALL_OUTPUTS = ("logp", "logprobs", "probs")


def _vec_for(N, lo, hi, eb, M):
    """The vector width ``tp_api._shard_rows`` admits for the column view ``x[:, lo:hi]`` of a contiguous ``(M, N)``."""
    vec = max_vec_elems(hi - lo, eb)
    while vec > 1 and (lo % vec or (M > 1 and N % vec)):
        vec //= 2
    return vec


def _protocol_specs(
    ds, N, bounds, M, eb, sm, temp_rows=False, outputs=ALL_OUTPUTS, pdl=False
):
    specs = []
    for lo, hi in bounds:
        if hi == lo:
            continue
        lay = ShardLayout(lo, hi, N)
        vec = _vec_for(N, lo, hi, eb, M)
        cfg = tp_config(lay, M, eb, sm, vec_max=vec)
        key = (cfg.threads_per_row, cfg.rows_per_cta, cfg.vec_elems, cfg.cluster)
        for kind in ("max", "sum"):
            specs.append(("partials", ds, lo, hi, N, key, kind, temp_rows, pdl))
        for mode in outputs:
            specs.append(("outputs", ds, lo, hi, N, mode, temp_rows, vec, pdl))
    for kind in ("max", "sum"):
        specs.append(("merge", tuple(tuple(b) for b in bounds), N, kind, pdl))
    return specs


def _suite_specs(sm):
    specs = []
    for dtype, N, bounds in SMALL_DT:
        ds, eb = _DTYPE_STR[dtype], dtype.itemsize
        for trow in (False, True):
            specs += _protocol_specs(ds, N, bounds, 5, eb, sm, trow)
    for N, tp in PROD + PADDED:
        specs += _protocol_specs(
            "bf16", N, shards.tp_shards(N, tp), 3, 2, sm, outputs=("logp",)
        )
    for tp in TPS:  # the TP degrees of 32768 at M = 16
        specs += _protocol_specs(
            "bf16", 32768, shards.tp_shards(32768, tp), 16, 2, sm, outputs=()
        )
    for N, bounds in SMALL[:4]:  # the whole partials space of the first two shards
        for lo, hi in bounds[:2]:
            lay = ShardLayout(lo, hi, N)
            vec = _vec_for(N, lo, hi, 2, 37)
            for cfg in all_valid_tp_configs(lay, 2, vec):
                key = (
                    cfg.threads_per_row,
                    cfg.rows_per_cta,
                    cfg.vec_elems,
                    cfg.cluster,
                )
                for kind in ("max", "sum"):
                    specs.append(
                        ("partials", "bf16", lo, hi, N, key, kind, False, False)
                    )
    for N, bounds in SMALL[:3]:  # batch invariance: M = 64, 1, 13
        for M in (64, 1, 13):
            specs += _protocol_specs("bf16", N, bounds, M, 2, sm)
    specs += _protocol_specs(
        "f32", 4097, [(0, 2048), (2048, 4097)], 6, 4, sm
    )  # masked rows
    specs += _protocol_specs(
        "bf16", 4096, shards.tp_shards(4096, 4), 64, 2, sm
    )  # the +0 filler
    specs += _protocol_specs(
        "bf16", 1024, [(0, 0), (0, 512), (512, 512), (512, 1024)], 4, 2, sm
    )  # empty shards
    for pdl in (False, True):  # graph replay and PDL, per-row temperatures
        specs += _protocol_specs(
            "bf16", 4096, shards.tp_shards(4096, 4), 5, 2, sm, True, pdl=pdl
        )
    specs += _protocol_specs(
        "bf16", 32000, shards.tp_shards(32000, 8), 7, 2, sm, outputs=()
    )  # the stats entry
    specs += _protocol_specs(
        "bf16", 4096, shards.tp_shards(4096, 4), 6, 2, sm
    )  # the operators
    return sorted(set(specs), key=repr)


@pytest.fixture(scope="session", autouse=True)
def _precompile_tp_kernels():
    if not _supported or os.environ.get("BI_TP_NO_PRECOMPILE") == "1":
        yield
        return
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env

    dev = torch.device("cuda")
    sm = device_sm_count(dev)
    # the workers compile without a device: they take the device's architecture explicitly, so the module directory
    # (FLASHINFER_CUDA_ARCH_LIST decides the workspace's architecture segment, CUTE_DSL_ARCH the kernel target) is the
    # one this process loads from
    archs = sorted(CompilationContext().TARGET_CUDA_ARCHS)
    major, minor = archs[0]
    specs = _suite_specs(sm)
    random.Random(0).shuffle(
        specs
    )  # spread the heavy (cluster) kernels over the workers
    workers = int(
        os.environ.get("BI_TP_PRECOMPILE_WORKERS", min(32, os.cpu_count() or 8))
    )
    workers = max(1, min(workers, len(specs)))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        CUTE_DSL_ARCH=f"sm_{major}{minor}",
        FLASHINFER_CUDA_ARCH_LIST=" ".join(f"{a}.{b}" for a, b in archs),
    )
    chunks = [specs[i::workers] for i in range(workers)]

    def run(chunk):
        r = subprocess.run(
            [sys.executable, "-c", _WORKER_CODE, repr(chunk)],
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if r.returncode != 0:
            return [(("worker",), r.stderr[-1500:])]
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("BAD ")]
        import ast

        return ast.literal_eval(tail[-1][4:]) if tail else [(("worker",), "no report")]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        bad = [b for res in ex.map(run, chunks) for b in res]
    print(
        f"[batch_invariant.sampling.tp] precompiled {len(specs)} kernels on {workers} workers in {time.time() - t0:.0f} s",
        flush=True,
    )
    assert not bad, bad[:3]
    module_dir = (
        jit_env.FLASHINFER_JIT_DIR
        / f"batch_invariant_sampling_tp_sm{major}{minor}_cute_dsl"
    )
    n_cached = len(list(module_dir.glob("*.o"))) if module_dir.exists() else 0
    print(
        f"[batch_invariant.sampling.tp] {n_cached} kernels in {module_dir}", flush=True
    )
    assert n_cached >= len(specs) // 2, (
        "the workers did not fill this process's module cache"
    )
    yield


def check_against_row_kernel(
    x, bounds, temp=None, ids=None, config=None, nan_ok=False, outputs=ALL_OUTPUTS
):
    """The protocol's (m, s) == the exact model's; every sharded output == the full-row kernel's bits."""
    M, N = x.shape
    if ids is None:
        ids = torch.arange(M, device="cuda") % N
    res = tp_api.tp_simulate(x, bounds, temp, ids, config=config, outputs=outputs)
    ref = R.lse_reference(x, temp, ids)
    eq = same_or_nan if nan_ok else same
    assert eq(res["m"], ref["m"]), "m is not the exact model's"
    assert eq(res["s"], ref["s"]), "s is not the exact model's"
    if "logp" in outputs:
        lp, lse = flashinfer.sampling.log_probs(x, ids, temp, return_lse=True)
        assert eq(res["logp"], lp), "logp (summed over the ranks) != the row kernel's"
        assert eq(res["lse"], lse), "lse != the row kernel's"
        assert eq(res["logp"], ref["logp"]) and eq(res["lse"], ref["lse"])
    if "logprobs" in outputs:
        assert eq(res["logprobs"], flashinfer.sampling.log_softmax(x, temp)), (
            "log_softmax columns"
        )
    if "probs" in outputs:
        assert eq(
            res["probs"], flashinfer.sampling.softmax(x, temp, batch_invariant=True)
        ), "softmax columns"
    return res


# --------------------------------------------------------------------------------------------------- TP invariance
@needs_device
@pytest.mark.parametrize(
    "dtype,N,bounds",
    SMALL_DT,
    ids=lambda v: _ids(v) if not isinstance(v, torch.dtype) else str(v)[6:],
)
def test_small_vocabularies(dtype, N, bounds):
    """Dense and adversarial rows, the three temperature forms: (m, s) == the exact model, every output == the row kernel."""
    for kind in ("dense", "adv"):
        x = logits(5, N, dtype, kind=kind)
        ids = ids_for(5, N)
        check_against_row_kernel(x, bounds, None, ids)
        check_against_row_kernel(x, bounds, 0.37, ids)
        temps = (
            (torch.rand(5, generator=torch.Generator().manual_seed(3)) * 1.5 + 0.25)
            .to(dtype)
            .cuda()
        )
        check_against_row_kernel(x, bounds, temps, ids)


@needs_device
@pytest.mark.parametrize("N,tp", PROD + PADDED)
def test_production_vocabularies(N, tp):
    """The production shards: TP = tp == TP = 1 == the exact model, the token log-probability == the row kernel's; the
    exchange width is the layout's."""
    bounds = shards.tp_shards(N, tp)
    x = logits(3, N, torch.bfloat16)
    ids = torch.tensor([0, N - 1, N // 2], device="cuda")
    res = check_against_row_kernel(x, bounds, None, ids, outputs=("logp",))
    one = tp_api.tp_simulate(x, [(0, N)], None, ids, outputs=())
    assert same(res["m"], one["m"]) and same(res["s"], one["s"]), "TP = tp != TP = 1"
    assert res["partials_max"].shape == (3, tp, tp_api.max_partials_width(bounds, N))
    xa = logits(3, N, torch.bfloat16, kind="adv")
    check_against_row_kernel(xa, bounds, None, ids, outputs=("logp",))


@needs_device
def test_tp_degrees_agree_with_each_other():
    N = 32768
    x = logits(16, N, torch.bfloat16, seed=3)
    stats = [
        tp_api.tp_simulate(x, shards.tp_shards(N, tp), None, None, outputs=())
        for tp in TPS
    ]
    for r in stats[1:]:
        assert same(r["m"], stats[0]["m"]) and same(r["s"], stats[0]["s"])


@needs_device
def test_operator_entries():
    """The public operators run the protocol: the merged statistics and the summed token log-probability equal the
    full-row ``log_probs``, the shard outputs the full row's columns."""
    N, bounds = 4096, shards.tp_shards(4096, 4)
    x = logits(6, N, torch.bfloat16, seed=4)
    ids = ids_for(6, N)
    Pmax = tp_api.max_partials_width(bounds, N)
    p1 = torch.stack(
        [
            flashinfer.sampling.log_probs_tp_partials(
                x[:, lo:hi], lo, hi, N, width=Pmax
            )
            for lo, hi in bounds
        ],
        dim=1,
    )
    m = flashinfer.sampling.log_probs_tp_merge(p1, bounds, N, kind="max")
    p2 = torch.stack(
        [
            flashinfer.sampling.log_probs_tp_partials(
                x[:, lo:hi], lo, hi, N, m=m, width=Pmax
            )
            for lo, hi in bounds
        ],
        dim=1,
    )
    s = flashinfer.sampling.log_probs_tp_merge(p2, bounds, N, kind="sum")
    lp_sum = None
    for lo, hi in bounds:
        lp, lse = flashinfer.sampling.log_probs_tp(
            x[:, lo:hi], ids, lo, hi, N, m, s, return_lse=True
        )
        lp_sum = lp if lp_sum is None else lp_sum + lp
        assert same(
            flashinfer.sampling.log_softmax_tp(x[:, lo:hi], lo, hi, N, m, s),
            flashinfer.sampling.log_softmax(x)[:, lo:hi],
        )
        assert same(
            flashinfer.sampling.softmax_tp(x[:, lo:hi], lo, hi, N, m, s),
            flashinfer.sampling.softmax(x, batch_invariant=True)[:, lo:hi],
        )
    row_lp, row_lse = flashinfer.sampling.log_probs(x, ids, return_lse=True)
    assert same(lp_sum, row_lp) and same(lse, row_lse)
    ref = R.lse_reference(x, None, ids)
    assert same(m, ref["m"]) and same(s, ref["s"])


# ------------------------------------------------------------------------------------------- configuration agreement
@needs_device
@pytest.mark.parametrize("N,bounds", SMALL[:4], ids=_ids)
def test_config_agreement_whole_space(N, bounds):
    """Every member of the partials space writes the identical partials tensor (the wire format is the shard's alone) in
    both rounds, so the merged statistics agree; the dispatcher's pick is a member."""
    x = logits(37, N, torch.bfloat16, seed=1)
    for lo, hi in bounds[:2]:
        lay = ShardLayout(lo, hi, N)
        xs = x[:, lo:hi]
        vec = tp_api._shard_rows(xs, lay)[
            2
        ]  # a view of the full row: its stride decides the admitted vector width
        cfgs = all_valid_tp_configs(lay, 2, vec)
        assert len(cfgs) >= 2, lay.describe()
        ref1 = tp_api.log_probs_tp_partials_batch_invariant(
            xs, lo, hi, N, config=cfgs[0]
        )
        m = (torch.randn(37, generator=torch.Generator().manual_seed(2)) * 3).cuda()
        ref2 = tp_api.log_probs_tp_partials_batch_invariant(
            xs, lo, hi, N, m=m, config=cfgs[0]
        )
        for c in cfgs[1:]:
            got1 = tp_api.log_probs_tp_partials_batch_invariant(xs, lo, hi, N, config=c)
            assert same(got1, ref1), f"max partials differ for {c.describe()}"
            got2 = tp_api.log_probs_tp_partials_batch_invariant(
                xs, lo, hi, N, m=m, config=c
            )
            assert same(got2, ref2), f"sum partials differ for {c.describe()}"
        assert tp_config(lay, 37, 2, device_sm_count(x.device), vec_max=vec) in cfgs
        for sm in (148, 152, 204, 224):
            assert tp_config(lay, 37, 2, sm, vec_max=vec) in cfgs


# ------------------------------------------------------------------------------------------------ batch invariance
@needs_device
@pytest.mark.parametrize("N,bounds", SMALL[:3], ids=_ids)
def test_batch_invariance_per_shard(N, bounds):
    """A row alone equals the row inside a batch, permuted and in a sub-batch: the partials of both rounds, the merged
    statistics and every sharded output."""
    xb = logits(64, N, torch.bfloat16, seed=2)
    ids = ids_for(64, N)
    full = tp_api.tp_simulate(xb, bounds, None, ids)
    for i in (0, 17, 63):
        one = tp_api.tp_simulate(xb[i : i + 1].clone(), bounds, None, ids[i : i + 1])
        for k in (
            "m",
            "s",
            "logp",
            "lse",
            "logprobs",
            "probs",
            "partials_max",
            "partials_sum",
        ):
            assert same(one[k][0], full[k][i]), (
                f"{k}: row {i} differs between M = 1 and M = 64"
            )
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1)).cuda()
    permuted = tp_api.tp_simulate(xb[perm].contiguous(), bounds, None, ids[perm])
    for k in ("m", "s", "logp", "partials_max"):
        assert same(permuted[k], full[k][perm]), f"{k}: permutation"
    sub = torch.randperm(64, generator=torch.Generator().manual_seed(7))[:13].cuda()
    subbatch = tp_api.tp_simulate(xb[sub].contiguous(), bounds, None, ids[sub])
    for k in ("m", "s", "logp", "logprobs"):
        assert same(subbatch[k], full[k][sub]), f"{k}: sub-batch"


# --------------------------------------------------------------------------------------------------------- edge cases
@needs_device
def test_masked_nan_inf_rows_and_partial_slot():
    N, bounds = 4097, [(0, 2048), (2048, 4097)]
    x = logits(6, N, torch.float32)
    x[0, 10:] = float("-inf")  # a masked row: the mass is on 10 tokens, all in rank 0
    x[1, :] = float("-inf")  # everything masked: m = -inf, every output NaN
    x[2, 2049] = float("nan")  # a NaN logit in rank 1
    x[3, 7] = float("inf")  # an infinite logit
    x[4, :2048] = float("-inf")  # rank 0 entirely masked
    x[5, 4096] = 40.0  # the partial last slot holds the maximum
    ids = torch.tensor([3, 0, 2049, 7, 3000, 4096], device="cuda")
    res = check_against_row_kernel(x, bounds, None, ids, nan_ok=True)
    assert bool((res["probs"][0, 10:] == 0).all()) and bool(
        torch.isinf(res["logprobs"][0, 10:]).all()
    )
    for r in (1, 2, 3):
        assert bool(torch.isnan(res["probs"][r]).all()) and bool(
            torch.isnan(res["lse"][r])
        )
    assert float(res["probs"][5, 4096]) > 0.99 and float(res["m"][5]) == 40.0


@needs_device
def test_logp_filler_is_exact_plus_zero():
    """Ranks without the token write +0 (so the SUM is exact), NaN for an id outside [0, N) on every rank, and no
    log-probability of the class is -0.0."""
    N, bounds = 4096, shards.tp_shards(4096, 4)
    x = logits(64, N, torch.bfloat16, seed=5)
    ids = ids_for(64, N)
    ids[0], ids[1] = -1, N
    res = tp_api.tp_simulate(x, bounds, None, ids)
    for lo, hi in bounds:
        lp, lse = tp_api.log_probs_tp_batch_invariant(
            x[:, lo:hi], ids, lo, hi, N, res["m"], res["s"], return_lse=True
        )
        inside = (ids >= lo) & (ids < hi)
        out_of_vocab = (ids < 0) | (ids >= N)
        filler = lp[~inside & ~out_of_vocab].cpu()
        assert torch.equal(bits(filler), torch.zeros(filler.numel(), dtype=torch.int32))
        assert bool(torch.isnan(lp[out_of_vocab]).all())
        assert same(lse, res["lse"])
    row = flashinfer.sampling.log_probs(x, ids)
    assert same_or_nan(res["logp"], row) and bool(torch.isnan(res["logp"][:2]).all())
    assert not bool((bits(res["logp"].cpu()) == bits(torch.tensor([-0.0]))).any())


@needs_device
def test_empty_shards_and_m_zero():
    N = 1024
    bounds = [(0, 0), (0, 512), (512, 512), (512, 1024)]
    x = logits(4, N, torch.bfloat16)
    ids = torch.arange(4, device="cuda") * 200
    check_against_row_kernel(x, bounds, None, ids)
    assert tp_api.max_partials_width(bounds, N) == shards.partials_width(0, 512, N)
    e = tp_api.log_probs_tp_partials_batch_invariant(x[:0, :512], 0, 512, N)
    assert e.shape == (0, shards.partials_width(0, 512, N))
    m = tp_api.log_probs_tp_merge_batch_invariant(
        torch.zeros(0, 4, 40, device="cuda"), bounds, N, kind="max"
    )
    assert m.shape == (0,)
    lp, lse = tp_api.log_probs_tp_batch_invariant(
        x[:0, :512], ids[:0], 0, 512, N, m, m, return_lse=True
    )
    assert lp.shape == (0,) and lse.shape == (0,)
    assert tp_api.log_softmax_tp_batch_invariant(
        x[:0, :512], 0, 512, N, m, m
    ).shape == (0, 512)
    y = logits(3, N, torch.bfloat16).view(3, 1, N)[:, :, :512]  # a 3-D shard
    p = tp_api.log_probs_tp_partials_batch_invariant(y, 0, 512, N)
    assert p.shape == (3, 1, shards.partials_width(0, 512, N))


@needs_device
def test_graph_replay_and_pdl():
    """R1: repeated launches, PDL on, and CUDA-graph replay give the same bits."""
    N, bounds = 4096, shards.tp_shards(4096, 4)
    x = logits(5, N, torch.bfloat16, seed=9)
    ids = ids_for(5, N)
    temps = (torch.rand(5, generator=torch.Generator().manual_seed(5)) + 0.5).cuda()
    ref = tp_api.tp_simulate(x, bounds, temps, ids)
    for _ in range(2):
        got = tp_api.tp_simulate(x, bounds, temps, ids)
        assert all(
            same(got[k], ref[k]) for k in ("m", "s", "logp", "lse", "logprobs", "probs")
        )
    got = tp_api.tp_simulate(x, bounds, temps, ids, enable_pdl=True)
    for k in ("m", "s", "logp", "lse", "logprobs", "probs"):
        assert same(got[k], ref[k]), k
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        tp_api.tp_simulate(x, bounds, temps, ids)
        torch.cuda.current_stream().synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            cap = tp_api.tp_simulate(x, bounds, temps, ids)
    torch.cuda.synchronize()
    for k in ("m", "s", "logp"):
        cap[k].zero_()
    g.replay()
    torch.cuda.synchronize()
    for k in ("m", "s", "logp", "logprobs"):
        assert same(cap[k], ref[k]), f"{k}: graph replay"


@needs_device
def test_stats_entry_matches_simulation():
    """``lse_row_tp_stats`` with the trainer's all-gather on every rank gives the simulation's (m, s), identical on every
    rank."""
    N, bounds = 32000, shards.tp_shards(32000, 8)
    x = logits(7, N, torch.bfloat16, seed=11)
    sim = tp_api.tp_simulate(x, bounds, 0.8, None, outputs=())

    def gather_for(r):
        calls = []

        def all_gather(
            p,
        ):  # the other ranks' partials of the same round come from the simulation's stacks (R1)
            key = "partials_max" if not calls else "partials_sum"
            calls.append(key)
            g = sim[key].clone()
            g[:, r, :] = p
            return g

        return all_gather

    for r, (lo, hi) in enumerate(bounds):
        m, s = tp_api.lse_row_tp_stats(
            x[:, lo:hi], lo, hi, N, bounds, 0.8, all_gather=gather_for(r)
        )
        assert same(m, sim["m"]) and same(s, sim["s"]), f"rank {r}"


@needs_device
def test_rejects_bad_shards_on_the_device():
    N = 1024
    x = logits(2, N, torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        tp_api.log_probs_tp_partials_batch_invariant(x[:, 4:516], 4, 516, N)
    with pytest.raises(ValueError):
        tp_api.log_probs_tp_partials_batch_invariant(x[:, :500], 0, 512, N)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        tp_api.log_probs_tp_merge_batch_invariant(
            torch.zeros(2, 2, 40, device="cuda"), [(0, 512), (520, 1024)], N, kind="max"
        )
    with pytest.raises(ValueError):
        tp_api.log_probs_tp_merge_batch_invariant(
            torch.zeros(2, 2, 1, device="cuda"), shards.tp_shards(N, 2), N, kind="max"
        )
    with pytest.raises(ValueError):
        tp_api.log_probs_tp_partials_batch_invariant(
            x[:, :512], 0, 512, N, config=TPConfig(1024, 1, 8, 16)
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        tp_api.log_probs_tp_partials_batch_invariant(
            x[:, :512].to(torch.int32), 0, 512, N
        )
