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

# Run-to-run bitwise repeatability of the non-deterministic ragged BF16 MoE grouped GEMM
# package (``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train``).
#
# The forward pass, the data gradient and the 256-wide k tile's weight gradient run
# copies of the deterministic package's programs.  The 512-wide k tile's weight gradient
# finishes a split tail in the same launch in a fixed order (``plan["ordered_tail"]``):
# unit ``s`` of split tail tile ``j`` stores its fp32 partial to slot
# ``j * tail_splits + s`` of the stream's ``tail_sums``, and the tile's last unit sums
# the slots from 0.0 in slot order, rounds once and stores (an ``_acc`` build adds the
# sum into ``out`` once).  Every output therefore repeats bit for bit for the same
# inputs, shapes, device SM count and configuration.  This suite checks that over the
# cases of the deterministic package's determinism suite
# (``test_cake_moe_grouped_gemm_det_train_determinism.py``: the 58 benchmark rows and
# its edge cases).
#
# Every case fixes its inputs (seeded from its label), prepares one launch and reuses
# it, so the plan, the caller-owned buffers and the current stream's scheduler / wave /
# tail workspaces stay the same over the repeats.  Before each of the first ``REPEATS``
# launches the output and the stream's kept ``tail_sums`` are filled with zero bytes,
# before each of the next ``REPEATS`` with ``FILL_BYTE`` (``tail_sums`` through
# ``base._poison_tail_sums``): no overwrite reads ``out``, and ``tail_sums`` has no rest
# state (each launch overwrites the slots it reads), so the bits must depend on the
# prior contents of neither.  Every launch's raw bits (an integer view, so NaN payloads
# and -0.0 compare exactly) must equal the first launch's, and a split tail's tickets
# rest at zero after the launches.
#
# Arms: ``planned`` launches the prepared launch as planned (for a data gradient whose
# plan probes the queue split, after its probe has completed, so every compared launch
# passes the probed split); ``even`` (data gradient only) launches the same prepared
# launch with the probed split disabled, so it passes the plan's even split, and its
# bits must also equal the planned arm's.
#
# Beyond the cases: every 512-wide k tile form forced onto a split-tail shape; sequences
# of 1 and 4 accumulate launches (``accumulate=True``: the fused route into fp32, the
# composed one into bf16) over the benchmark rows; autograd steps (plain, under a
# non-reentrant checkpoint, under torch's deterministic mode) and
# ``fuse_wgrad_accumulation`` micro-steps; CUDA graph replays against the eager bits;
# and two concurrent streams against a quiet one.  ``test_route_coverage`` checks that
# the cases reach the 256-wide k tile's tail reduce and the 512-wide k tile's split
# ordered tail for both output dtypes on the device (so the suite cannot pass on unsplit
# tails alone), and ``test_fresh_processes_give_the_same_bits`` compares output digests
# of a subset of the cases (split ordered tails and accumulates among them) across two
# fresh processes and this one.  The fused and the composed routes are not compared with
# each other here: they differ by design on an empty group's -0.0.

import contextlib
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

# The backend suite ``base`` names the package through ``cake_package`` only; nothing
# else of flashinfer is imported here.
from tests.experimental import test_cake_moe_grouped_gemm_nondet_train as base

pkg = base.pkg
cb = base.cb
cake_jit = base.cake_jit
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS

REPEATS = 10  # launches per pre-state (zero bytes, then FILL_BYTE)
RACE_REPEATS = 100  # the flashinfer-ai/flashinfer#6570 shape
# The byte of the second pre-state, the one ``base._poison_tail_sums`` writes (0x5A).
FILL_BYTE = base.GARBAGE_BYTE
SPOT_ROWS = 64
SPOT_COLS = 256
WORKER_TIMEOUT_S = 3600

# The 58 MoE projection rows of benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py:
# (N, K) of w ``[E, N, K]``; E=4 balanced / skewed / one-empty group sizes at two token
# counts for every op, the fp32 weight gradient of the balanced E=4 rows, and E=8/16/32
# balanced over 131072 rows.
BENCH_PROJ = base.BENCH_PROJ
E4_GROUPS = {
    112576: {
        "balanced": [28128, 28128, 28128, 28192],
        "skewed": [45024, 33760, 22528, 11264],
        "one_empty": [56288, 33760, 22528, 0],
    },
    136064: {
        "balanced": [34016, 34016, 34016, 34016],
        "skewed": [54432, 40832, 27200, 13600],
        "one_empty": [68032, 40832, 27200, 0],
    },
}
EXPERT_COUNTS = (8, 16, 32)
EXPERT_COUNT_SUM_M = 131072

# Edge cases, each run for fwd, dgrad and the bf16 / fp32 weight gradient.
# (N, K): the two MoE projections; ODD has K % 512 != 0 (the 256-wide k tile and
# the 256-column fwd / dgrad tiles); RACE is the #6570 reproduction shape.
ODD_PROJ = (1280, 1792)
RACE_PROJ = (1024, 6144)
EDGE_SIZES = {
    "one_group": [4000],
    "empty_groups": [0, 3000, 0, 0, 517, 4096, 0, 1],
    "one_nonempty": [0, 0, 5000, 0],
    "nonmult_m": [1, 255, 257, 1023, 4097, 63, 65],
    "all_empty": [0, 0, 0, 0],
    # Fewer rows per group than k512_min_rows_per_group: the 256-wide k tile.
    "e64_m2048": [2048] * 64,
    "e128_m512": [512] * 128,
}
ODD_SIZES = ("empty_groups", "nonmult_m")
RACE_SIZES = [32] * 256
# Weight gradient only: the 256-wide k tile (K % 512 != 0) over two long groups,
# with 84 / 112 / 224 output tiles, so that on every architecture's cluster count
# (74, 76, 106, 108) some case leaves a short last wave that is split over the
# clusters and summed by the tail reduce.
K256_TAIL_K = 1792
K256_TAIL_N_BLOCKS = (6, 8, 16)
K256_TAIL_SIZES = [8192, 8191]

OPS = ("fwd", "dgrad", "wgrad")
INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    op: str
    n: int
    k: int
    sizes: tuple[int, ...]
    out_dtype: torch.dtype
    repeats: int = REPEATS


def _case(label, op, nk, sizes, out_dtype=torch.bfloat16, repeats=REPEATS):
    n, k = nk
    tag = "f32" if out_dtype == torch.float32 else "bf16"
    return Case(
        label=f"{op}_{label}_{tag}",
        op=op,
        n=n,
        k=k,
        sizes=tuple(sizes),
        out_dtype=out_dtype,
        repeats=repeats,
    )


def bench_cases():
    cases = []
    for proj, nk in BENCH_PROJ.items():
        for op in OPS:
            for dists in E4_GROUPS.values():
                for dist, sizes in dists.items():
                    cases.append(
                        _case(f"{proj}_e4_m{sum(sizes)}_{dist}", op, nk, sizes)
                    )
    for proj, nk in BENCH_PROJ.items():
        for dists in E4_GROUPS.values():
            sizes = dists["balanced"]
            cases.append(
                _case(
                    f"{proj}_e4_m{sum(sizes)}_balanced",
                    "wgrad",
                    nk,
                    sizes,
                    out_dtype=torch.float32,
                )
            )
    for e in EXPERT_COUNTS:
        for proj, nk in BENCH_PROJ.items():
            for op in OPS:
                sizes = [EXPERT_COUNT_SUM_M // e] * e
                cases.append(
                    _case(f"{proj}_e{e}_m{EXPERT_COUNT_SUM_M}_balanced", op, nk, sizes)
                )
    assert len(cases) == 58
    return cases


def _edge_variants(label, nk, sizes, repeats=REPEATS):
    cases = [_case(label, "fwd", nk, sizes), _case(label, "dgrad", nk, sizes)]
    # The weight gradient addresses dW [E, N, K] with 32-bit element offsets.
    if len(sizes) * nk[0] * nk[1] <= int(HOST_PLAN_CONSTANTS["int32_elems"]):
        cases += [
            _case(label, "wgrad", nk, sizes, repeats=repeats),
            _case(label, "wgrad", nk, sizes, out_dtype=torch.float32, repeats=repeats),
        ]
    return cases


def edge_cases():
    cases = []
    for name, sizes in EDGE_SIZES.items():
        for proj, nk in BENCH_PROJ.items():
            cases += _edge_variants(f"edge_{name}_{proj}", nk, sizes)
    for name in ODD_SIZES:
        cases += _edge_variants(f"edge_{name}_odd", ODD_PROJ, EDGE_SIZES[name])
    cases += _edge_variants("race6570_e256_m32", RACE_PROJ, RACE_SIZES, RACE_REPEATS)
    for blocks in K256_TAIL_N_BLOCKS:
        nk = (256 * blocks, K256_TAIL_K)
        label = f"k256tail_n{nk[0]}"
        cases.append(_case(label, "wgrad", nk, K256_TAIL_SIZES))
        cases.append(
            _case(label, "wgrad", nk, K256_TAIL_SIZES, out_dtype=torch.float32)
        )
    return cases


CASES = bench_cases() + edge_cases()
CASE_BY_LABEL = {case.label: case for case in CASES}
assert len(CASE_BY_LABEL) == len(CASES)
ARMS = [
    (case.label, arm)
    for case in CASES
    for arm in (("planned", "even") if case.op == "dgrad" else ("planned",))
]

# Accumulate cases (``accumulate=True``): every weight-gradient row of the benchmark
# (``base.BENCH_WGRAD_ROWS``, the 512-wide k tile) of both projections, into an fp32
# ``out`` (the fused route) and a bf16 one (the composed route: no bf16 accumulate build
# ships).  Each runs ``ACC_SEQUENCES`` times a sequence of each length of ``ACC_STEPS``
# launches into one ``out`` from the same ``out0``; ``out``'s bits are compared after
# every launch where ``out`` has at most ``ACC_STEP_BITS_MAX_ELEMS`` elements, else
# after the first and the last launch of the sequence.
ACC_CASES = [
    _case(f"acc_{proj}_{label}", "wgrad", BENCH_PROJ[proj], sizes, out_dtype=dtype)
    for label, sizes in base.BENCH_WGRAD_ROWS
    for proj in BENCH_PROJ
    for dtype in (torch.float32, torch.bfloat16)
]
ACC_CASE_BY_LABEL = {case.label: case for case in ACC_CASES}
assert len(ACC_CASE_BY_LABEL) == len(ACC_CASES)
ACC_STEPS = (1, 4)
ACC_SEQUENCES = 3  # the first with zero-byte slots, the others with FILL_BYTE slots
ACC_STEP_BITS_MAX_ELEMS = 1 << 28
ACC_OUT0_SCALE = 0.25  # the starting out: normal values of this scale (no -0.0)

# The fresh-process digests: a subset of ``CASES`` (every weight-gradient benchmark
# case, the 512-wide k tile, whose ordered tail splits on most rows of every
# architecture's parts; the forward and data gradient of the E4 rows at 112576 rows;
# the 256-wide k tile's tail-reduce cases; the #6570 shape's weight gradients), and
# accumulate digests (``ACC_PREFIX`` + label: ``ACC_DIGEST_STEPS`` accumulate launches
# into an ``out0`` seeded from that label) of the balanced E4 weight-gradient cases.
XPROC_LABELS = tuple(
    case.label
    for case in CASES
    if case.label.startswith(
        ("wgrad_gate_up_", "wgrad_down_", "wgrad_k256tail_", "wgrad_race6570_")
    )
    or (case.op != "wgrad" and "_e4_m112576_" in case.label)
)
ACC_PREFIX = "acc:"
ACC_DIGEST_STEPS = 2
ACC_DIGEST_LABELS = tuple(
    ACC_PREFIX + case.label
    for case in CASES
    if case.label.startswith(("wgrad_gate_up_e4_", "wgrad_down_e4_"))
    and case.label.endswith(("_balanced_bf16", "_balanced_f32"))
)

# CUDA graph replay: the captured launches, the replays of a captured overwrite (each
# compared with the eager bits) and the replay counts of a captured accumulate (each
# compared with as many eager launches from the same ``out0``).
GRAPH_KINDS = (
    "fwd",
    "dgrad",
    "wgrad_bf16",
    "wgrad_f32",
    "wgrad_acc_f32",
    "wgrad_acc_bf16",
)
GRAPH_REPLAYS = 4
GRAPH_ACC_REPLAYS = (1, 3)

# Two concurrent streams: (output dtype, accumulate) of the launches and the iterations.
STREAM_CASES = [
    (torch.bfloat16, False),
    (torch.float32, False),
    (torch.float32, True),
    (torch.bfloat16, True),
]
STREAM_ITERATIONS = 16

# Autograd: plain steps compared with the first, and the ``fuse_wgrad_accumulation``
# arms (name, checkpoint, early stop).
AUTOGRAD_REPEATS = 3
MAIN_GRAD_ARMS = (
    ("plain", None, True),
    ("plain_again", None, True),
    ("checkpoint", "non_reentrant", True),
    ("checkpoint_no_early_stop", "non_reentrant", False),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device():
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    if cb.arch_for(device) is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    return device


def _require_program(case, device):
    out_dtype = case.out_dtype if case.op == "wgrad" else None
    if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
        pytest.skip(
            f"no generated program for {case.op} registered for {cb.arch_for(device)}"
        )


def _inputs(case, device):
    """Fixed inputs of ``case``: seeded from its label, so every process builds the same bits."""
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(case.label.encode()))
    sum_m, num_groups = sum(case.sizes), len(case.sizes)
    scale = 1.0 / math.sqrt(case.k)

    def randn(*shape, scale=1.0):
        t = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
        return t.mul_(scale).to(torch.bfloat16)

    offs = torch.tensor(case.sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    if case.op == "fwd":
        a = randn(sum_m, case.k, scale=scale)
        b = randn(num_groups, case.n, case.k, scale=scale)
        out = torch.empty(sum_m, case.n, dtype=torch.bfloat16, device=device)
    elif case.op == "dgrad":
        a = randn(sum_m, case.n)
        b = randn(num_groups, case.n, case.k, scale=scale)
        out = torch.empty(sum_m, case.k, dtype=torch.bfloat16, device=device)
    else:
        a = randn(sum_m, case.n)
        b = randn(sum_m, case.k, scale=scale)
        out = torch.empty(
            num_groups, case.n, case.k, dtype=case.out_dtype, device=device
        )
    return a, b, offs, out


def _prepare(case, a, b, offs, out, accumulate=False):
    if case.op == "fwd":
        return cb.cake_moe_grouped_gemm_nondet_train_prepare_forward(
            x=a, w=b, offs=offs, out=out
        )
    if case.op == "dgrad":
        return cb.cake_moe_grouped_gemm_nondet_train_prepare_dgrad(
            g=a, w=b, offs=offs, out=out
        )
    return cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
        g=a,
        x=b,
        offs=offs,
        out=out,
        out_dtype=case.out_dtype,
        num_groups=None,
        accumulate=accumulate,
    )


def _even_split(launch):
    """``launch`` with the probed queue split disabled: it passes the plan's even ``c0`` / ``c1``."""
    return dataclasses.replace(
        launch,
        launches=tuple(
            dataclasses.replace(item, split_slots=()) for item in launch.launches
        ),
    )


def _settle_probe(launch, device):
    """Launch once and wait, so the probe a probed launch starts has completed; return
    the split the next launches pass (None for an even plan or where the probe gave none)."""
    launch.launch()
    torch.cuda.synchronize(device)
    if launch.plan.get("queue_split") != "probed":
        return None
    return cb.stream_queue_split(device)


def _bits(out):
    return out.view(INT_VIEW[out.dtype])


def _plan_record(launch):
    keys = (
        "grid",
        "clusters",
        "c0",
        "c1",
        "queue_split",
        "tile_n",
        "tile_k",
        "tail_tile_k",
        "num_tail",
        "tail_splits",
        "ordered_tail",
        "unordered_tail",
        "tail_sum_elems",
        "tail_ticket_words",
        "raster_rows",
        "wave_mode",
        "launches",
        "accumulate",
    )
    return {key: launch.plan[key] for key in keys if key in launch.plan}


def _split_ordered(plan):
    """Whether a weight-gradient plan finishes a split tail in the same launch in slot order
    (the stream's ``tail_sums`` and ``tail_tickets`` are used)."""
    return bool(plan.get("ordered_tail")) and int(plan.get("tail_splits", 1)) > 1


def _tail_form(launch):
    """How a weight-gradient launch finishes its tail: ``"ordered_split"`` (the ordered
    one-launch tail, split), ``"ordered_whole"`` (that form, unsplit), ``"unordered"``
    (never planned by the shipped constants), ``"tail_reduce"`` (the 256-wide k tile's
    two-launch reduce) or ``"none"``."""
    plan = launch.plan
    if plan.get("unordered_tail"):
        return "unordered"
    if plan.get("ordered_tail"):
        return "ordered_split" if plan["tail_splits"] > 1 else "ordered_whole"
    return "tail_reduce" if "tail_reduce" in launch.stages else "none"


def _assert_split_ordered(launch):
    """``launch`` is a weight gradient whose plan finishes a split tail in the same launch in
    slot order: one launch, no tail reduce, ``tail_sums`` sized ``num_tail * tail_splits``
    slots of ``cluster_m x tail_tile_k`` floats, not order-varying."""
    plan = launch.plan
    assert plan["ordered_tail"] and not plan["unordered_tail"], plan
    assert plan["tail_splits"] > 1 and plan["launches"] == 1, plan
    assert plan["tail_reduce_grid"] is None, plan
    assert "tail_reduce" not in launch.stages, launch.stages
    assert plan["tail_sum_elems"] == (
        plan["num_tail"]
        * plan["tail_splits"]
        * int(HOST_PLAN_CONSTANTS["cluster_m"])
        * plan["tail_tile_k"]
    ), plan
    assert not launch.order_varying, plan


def _assert_acc_route(plan, out_dtype):
    """An fp32 ``out`` takes the fused route (or composes with ``"min_cuda"`` where its
    build's library cannot load here); a bf16 one composes (no bf16 accumulate build
    ships)."""
    if out_dtype == torch.float32:
        assert (
            plan["accumulate"] == "fused" or plan.get("accumulate_reason") == "min_cuda"
        ), plan
    else:
        assert plan["accumulate"] == "composed", plan


def _prefill_tail_sums(device, fill, stream=None):
    """Fill the kept split-tail slots of ``stream`` (the current stream by default) with
    ``fill`` bytes, on that stream: zero bytes here, ``FILL_BYTE`` through
    ``base._poison_tail_sums``.  Returns whether a kept slot buffer existed."""
    if fill == FILL_BYTE:
        return base._poison_tail_sums(device, stream)
    buffer = cb.stream_workspace("tail_sums", device, stream)
    if buffer is None:
        return False
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        buffer.view(torch.uint8).fill_(fill)
    return True


def _run_repeats(launch, out, repeats, device):
    """Launch ``2 * repeats`` times (zero bytes, then FILL_BYTE, in ``out`` and in the
    stream's kept ``tail_sums`` before each); return (reference bits of the first launch,
    launches, mismatching launches, worst count of differing elements, launches whose kept
    slots were prefilled)."""
    ref, launches, mismatches, worst, prefilled = None, 0, 0, 0, 0
    for fill in (0, FILL_BYTE):
        for _ in range(repeats):
            out.view(torch.uint8).fill_(fill)
            prefilled += int(_prefill_tail_sums(device, fill))
            launch.launch()
            bits = _bits(out)
            launches += 1
            if ref is None:
                ref = bits.clone()
            elif not torch.equal(bits, ref):
                mismatches += 1
                worst = max(worst, int((bits != ref).sum()))
    return ref, launches, mismatches, worst, prefilled


def _spot_check(case, a, b, out):
    """Guard against a vacuous pass: the last non-empty group's first block matches an fp64
    reference, and an empty group's weight gradient is exact zeros."""
    ends = [sum(case.sizes[: i + 1]) for i in range(len(case.sizes))]
    bounds = list(zip([0] + ends[:-1], ends, strict=True))
    nonempty = [e for e, (s, t) in enumerate(bounds) if t > s]
    if not nonempty:
        return
    e = nonempty[-1]
    s, t = bounds[e]
    if case.op == "fwd":
        r, c = min(t, s + SPOT_ROWS), min(case.n, SPOT_COLS)
        ref = a[s:r].double() @ b[e, :c].double().T
        got = out[s:r, :c]
    elif case.op == "dgrad":
        r, c = min(t, s + SPOT_ROWS), min(case.k, SPOT_COLS)
        ref = a[s:r].double() @ b[e][:, :c].double()
        got = out[s:r, :c]
    else:
        rn, ck = min(case.n, SPOT_COLS // 2), min(case.k, SPOT_COLS // 2)
        ref = a[s:t, :rn].double().T @ b[s:t, :ck].double()
        got = out[e, :rn, :ck]
        for empty in (i for i, (s0, t0) in enumerate(bounds) if t0 == s0):
            assert int(_bits(out[empty]).count_nonzero()) == 0, (
                f"group {empty} is empty but its weight gradient is not exact zeros"
            )
    torch.testing.assert_close(got.double(), ref, atol=3e-2, rtol=2e-2)


def _acc_out0(label, case, device):
    """The starting ``out`` of an accumulate: normal values of scale ``ACC_OUT0_SCALE``
    (no -0.0) seeded from ``label``, so every process builds the same bits."""
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(f"out0:{label}".encode()))
    out0 = torch.randn(
        len(case.sizes),
        case.n,
        case.k,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    return out0.mul_(ACC_OUT0_SCALE).to(case.out_dtype)


def _acc_spot_check(case, a, b, out0, out):
    """Guard against a vacuous pass of an accumulate: on the first block of the last group
    with rows ``out - out0`` matches the fp64 weight gradient, and an empty group keeps
    ``out0``'s bits (``out0`` holds no -0.0, so the composed ``out.add_`` keeps them too)."""
    bounds = base._bounds(case.sizes)
    nonempty = [e for e, (s, t) in enumerate(bounds) if t > s]
    assert nonempty, case.label
    e = nonempty[-1]
    s, t = bounds[e]
    rn, ck = min(case.n, SPOT_COLS // 2), min(case.k, SPOT_COLS // 2)
    ref = a[s:t, :rn].double().T @ b[s:t, :ck].double()
    got = out[e, :rn, :ck].double() - out0[e, :rn, :ck].double()
    torch.testing.assert_close(got, ref, atol=3e-2, rtol=2e-2)
    for empty in (i for i, (s0, t0) in enumerate(bounds) if t0 == s0):
        assert torch.equal(_bits(out[empty]), _bits(out0[empty])), (
            f"group {empty} is empty but the accumulate changed it"
        )


def _assert_added(out, out0, dws, what):
    """Guard against a vacuous pass: ``out`` is ``out0`` plus the weight gradients ``dws``
    (fp64 references) within the package's bf16 budget applied per term, ``1e-2 + 1e-2 *
    (|out0| + sum |dW|)``, on every element (an fp32 ``out`` meets it too)."""
    ref = out0.double()
    scale = out0.double().abs()
    for dw in dws:
        ref = ref + dw.double()
        scale = scale + dw.double().abs()
    err = (out.double() - ref).abs()
    bad = int((err > base.ATOL + base.RTOL * scale).sum())
    assert bad == 0, f"{what}: {bad} elements beyond the budget of out0 + dW"


def _assert_same_step_bits(got, want, what):
    """The outputs and leaf gradients of two ``base._step`` results have the same bits
    (None where None)."""
    pairs = zip((*got[0], *got[1]), (*want[0], *want[1]), strict=True)
    for index, (a, b) in enumerate(pairs):
        assert (a is None) == (b is None), (what, index)
        if a is None:
            continue
        assert a.dtype == b.dtype and a.shape == b.shape, (what, index)
        assert torch.equal(_bits(a), _bits(b)), (
            f"{what}: tensor {index} differs from the first step in "
            f"{int((_bits(a) != _bits(b)).sum())} elements"
        )


def _main_grad_steps(fn, x0, w0, mg0, grads, ckpt, early_stop, device):
    """Micro-steps of ``fn(x, w)`` (``fuse_wgrad_accumulation=True``) from
    ``w.main_grad = mg0.clone()``, plain (``ckpt`` None) or under a non-reentrant
    checkpoint, the stream's slots filled with ``FILL_BYTE`` before each; returns
    ``(w, x gradients)``."""
    w = w0.clone().requires_grad_(True)
    w.main_grad = mg0.clone()
    dxs = []
    for g in grads:
        base._poison_tail_sums(device)
        x = x0.clone().requires_grad_(True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            if ckpt is None:
                y = fn(x, w)
            else:
                y = torch.utils.checkpoint.checkpoint(fn, x, w, use_reentrant=False)
            y.backward(g)
        dxs.append(x.grad)
    torch.cuda.synchronize()
    return w, dxs


# ---------------------------------------------------------------------------
# Repeat-run determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "arm"), ARMS, ids=[f"{x}-{y}" for x, y in ARMS])
def test_repeat_launches_give_identical_bits(label, arm, record_property):
    case = CASE_BY_LABEL[label]
    device = _device()
    _require_program(case, device)
    a, b, offs, out = _inputs(case, device)
    launch = _prepare(case, a, b, offs, out)
    assert not launch.order_varying, launch.plan
    if case.op == "wgrad" and launch.plan["tile_k"] == 512:
        assert launch.plan["ordered_tail"] and not launch.plan["unordered_tail"], (
            launch.plan
        )
    split_tail = _split_ordered(launch.plan)
    probed = launch.plan.get("queue_split") == "probed"
    split = _settle_probe(launch, device)
    if probed and split is None and sum(case.sizes) > 0:
        pytest.fail(
            "the plan probes the queue split but the probe gave none on this device "
            "(no CUDA compiler, or it did not see every SM and both domains)"
        )
    planned_bits = None
    if arm == "even":
        # The planned launch after the settled probe passes the probed split.
        out.view(torch.uint8).fill_(0)
        launch.launch()
        planned_bits = _bits(out).clone()
        launch = _even_split(launch)
    ref, launches, mismatches, worst, prefilled = _run_repeats(
        launch, out, case.repeats, device
    )
    torch.cuda.synchronize(device)
    record = dict(
        case=label,
        op=case.op,
        arm=arm,
        arch=cb.arch_for(device),
        sm_count=cb.sm_count(device),
        record_name=launch.record_name,
        stages=list(launch.stages),
        plan=_plan_record(launch),
        probed_split=list(split) if split is not None else None,
        split_ordered_tail=split_tail,
        sizes_e=len(case.sizes),
        sum_m=sum(case.sizes),
        n=case.n,
        k=case.k,
        out_dtype=str(case.out_dtype).replace("torch.", ""),
        prestates=["zero", f"0x{FILL_BYTE:02x}"],
        launches=launches,
        slots_prefilled=prefilled,
        mismatches=mismatches,
        worst_diff_elems=worst,
    )
    if arm == "even" and probed:
        record["equal_to_probed"] = bool(torch.equal(ref, planned_bits))
    record_property("repeat", json.dumps(record, sort_keys=True))
    print("REPEAT " + json.dumps(record, sort_keys=True))
    assert mismatches == 0, (
        f"{mismatches} of {launches - 1} repeat launches differ from the first "
        f"(worst {worst} elements)"
    )
    if arm == "even" and probed:
        assert record["equal_to_probed"], (
            "the even split gives other bits than the probed split"
        )
    elif arm == "even":
        assert torch.equal(ref, planned_bits)
    if split_tail:
        # Every compared launch ran on prefilled slots; the tickets are back at rest.
        assert prefilled == launches, (prefilled, launches)
        base._assert_tail_at_rest(device)
    _spot_check(case, a, b, out)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("form", base.K512_FORMS)
def test_forced_forms_give_identical_bits(form, out_dtype, monkeypatch):
    """Every 512-wide k tile form the device's architecture serves (forced through the host
    plan constants, ``base._force_form``) on a ``K512_E3`` shape whose plan splits its
    ordered tail (``base._split_tail_shape``): the launches of both pre-states give
    identical bits, the result is within the budget of the fp64 reference with an
    exact-zero empty group, and the stream's tickets rest at zero."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    arch = base._arch(device)
    if not base._force_form(monkeypatch, form, arch):
        pytest.skip(f"no {form} weight-gradient build on {arch}")
    sizes = list(base.K512_E3)
    shape = base._split_tail_shape(device, sizes, base._form_ks(form))
    assert shape is not None, f"no candidate shape splits the {form} ordered tail"
    n, k = shape
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=21)
    launch = base.prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    assert launch.stages == (form,), (launch.stages, launch.plan)
    _assert_split_ordered(launch)
    launch.launch()  # the stream's slots and tickets exist before the compared launches
    _ref, launches, mismatches, worst, prefilled = _run_repeats(
        launch, launch.out, REPEATS, device
    )
    torch.cuda.synchronize(device)
    plan = launch.plan
    print(
        f"FORM form={form} arch={arch} dtype={out_dtype} n={n} k={k} "
        f"splits={plan['tail_splits']} num_tail={plan['num_tail']} "
        f"mismatches={mismatches} worst={worst}"
    )
    assert mismatches == 0, (
        f"{mismatches} of {launches - 1} repeat launches differ from the first "
        f"(worst {worst} elements)"
    )
    assert prefilled == launches, (prefilled, launches)
    base._assert_tail_at_rest(device)
    base._assert_wgrad_close(launch.out, base._reference("wgrad", g, x, sizes))
    base._assert_empty_groups_zero(launch.out, sizes)


@pytest.mark.parametrize("label", list(ACC_CASE_BY_LABEL))
def test_accumulate_sequences_give_identical_bits(label, record_property):
    """A prepared accumulate (``accumulate=True``, the fused route into fp32, the composed
    one into bf16) of a benchmark row, run ``ACC_SEQUENCES`` times as a sequence of 1 and
    of 4 launches into the same ``out`` from the same ``out0``, the stream's slots filled
    with zero bytes (first run) or ``FILL_BYTE`` (the others) before every launch: every
    run gives the first run's bits after each launch (after the first and the last where
    ``out`` is large), the 1-launch result is the first launch of the 4-launch sequence,
    the first block of ``out - out0`` matches the fp64 weight gradient, an empty group
    keeps ``out0``'s bits, and a split tail's tickets rest at zero."""
    case = ACC_CASE_BY_LABEL[label]
    device = _device()
    _require_program(case, device)
    a, b, offs, out = _inputs(case, device)
    out0 = _acc_out0(label, case, device)
    out.copy_(out0)
    launch = _prepare(case, a, b, offs, out, accumulate=True)
    plan = launch.plan
    assert plan["tile_k"] == 512 and plan["ordered_tail"], plan
    assert not plan["unordered_tail"] and not launch.order_varying, plan
    _assert_acc_route(plan, case.out_dtype)
    split_tail = _split_ordered(plan)
    launch.launch()  # the stream's slots and tickets exist before the compared launches
    every_step = out.numel() <= ACC_STEP_BITS_MAX_ELEMS
    firsts, mismatches, launches, prefilled = {}, [], 0, 0
    for steps in ACC_STEPS:
        refs = {}
        for run in range(ACC_SEQUENCES):
            fill = 0 if run == 0 else FILL_BYTE
            out.copy_(out0)
            for step in range(steps):
                prefilled += int(_prefill_tail_sums(device, fill))
                launch.launch()
                launches += 1
                if not (every_step or step in (0, steps - 1)):
                    continue
                bits = _bits(out)
                if run == 0:
                    refs[step] = bits.clone()
                elif not torch.equal(bits, refs[step]):
                    differ = int((bits != refs[step]).sum())
                    mismatches.append((steps, run, step, differ))
        if steps == 1:
            _acc_spot_check(case, a, b, out0, out)
        firsts[steps] = refs[0]
        del refs
    torch.cuda.synchronize(device)
    first_equal = bool(torch.equal(firsts[ACC_STEPS[0]], firsts[ACC_STEPS[-1]]))
    record = dict(
        case=label,
        arch=cb.arch_for(device),
        sm_count=cb.sm_count(device),
        stages=list(launch.stages),
        plan=_plan_record(launch),
        accumulate_reason=plan.get("accumulate_reason"),
        split_ordered_tail=split_tail,
        steps=list(ACC_STEPS),
        sequences=ACC_SEQUENCES,
        every_step=every_step,
        launches=launches,
        slots_prefilled=prefilled,
        mismatches=mismatches[:8],
        first_launch_equal=first_equal,
    )
    record_property("accumulate", json.dumps(record, sort_keys=True))
    print("ACCUMULATE " + json.dumps(record, sort_keys=True))
    assert not mismatches, (
        f"{len(mismatches)} compared states differ from the first run "
        f"(steps, run, step, elements): {mismatches[:4]}"
    )
    assert first_equal, (
        "the first launch of the 4-launch sequence differs from the 1-launch sequence"
    )
    if split_tail:
        assert prefilled == launches, (prefilled, launches)
        base._assert_tail_at_rest(device)


# ---------------------------------------------------------------------------
# Autograd and checkpoint replay
# ---------------------------------------------------------------------------


def test_autograd_steps_give_identical_bits(monkeypatch):
    """``cake_moe_grouped_gemm_nondet`` on a 512-wide k tile shape whose weight gradient
    splits its ordered tail (``base._tile_case``): plain steps (the stream's slots filled
    with ``FILL_BYTE`` before each), steps under a non-reentrant checkpoint with the early
    stop on and off, and a plain step under torch's deterministic mode (no launch alerts)
    all give the first plain step's ``Y``, ``dX`` and ``dW`` bits; ``dW`` is within the
    budget of the fp64 reference with an exact-zero empty group."""
    device = base._require_autograd_programs()
    base._fresh_stream_workspaces(monkeypatch)
    sizes, (n, k) = base._tile_case(device, "k512")
    x0, w0, g, offs = base._inputs(sizes, n, k, device, seed=13)
    _assert_split_ordered(base.prepare_wgrad(g, x0, offs))
    wrapper = base._wrapper(offs)
    first = base._step(wrapper, base._leaves(x0, w0), (g,))
    assert base._poison_tail_sums(device), "the weight gradient kept no tail slots"
    arms = [("plain", None, True)] * (AUTOGRAD_REPEATS - 1) + [
        ("checkpoint", "non_reentrant", True),
        ("checkpoint_no_early_stop", "non_reentrant", False),
    ]
    for name, ckpt, early_stop in arms:
        base._poison_tail_sums(device)
        got = base._step(
            wrapper, base._leaves(x0, w0), (g,), ckpt=ckpt, early_stop=early_stop
        )
        _assert_same_step_bits(got, first, name)
    base._poison_tail_sums(device)
    with base._torch_deterministic():
        got = base._step(wrapper, base._leaves(x0, w0), (g,))
    _assert_same_step_bits(got, first, "deterministic_mode")
    base._assert_tail_at_rest(device)
    dw = first[1][1]
    assert dw.dtype == torch.bfloat16
    base._assert_wgrad_close(dw, base._reference("wgrad", g, x0, sizes))
    base._assert_empty_groups_zero(dw, sizes)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_main_grad_accumulation_gives_identical_bits(out_dtype, monkeypatch):
    """``cake_moe_grouped_gemm_nondet(..., fuse_wgrad_accumulation=True)`` on the shape of
    ``test_autograd_steps_give_identical_bits``: two micro-steps from the same
    ``main_grad`` (fp32: the fused route, bf16: the composed one), plain twice and under a
    non-reentrant checkpoint with the early stop on and off, leave the same ``main_grad``
    and ``x.grad`` bits with ``w.grad`` None, and ``main_grad`` is within the budget of
    ``mg0 + dW1 + dW2``."""
    device = base._require_autograd_programs()
    base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, (n, k) = base._tile_case(device, "k512")
    x0, w0, g1, offs = base._inputs(sizes, n, k, device, seed=14)
    g2 = base._inputs(sizes, n, k, device, seed=15)[2]
    gen = torch.Generator(device=device).manual_seed(53)
    mg0 = (torch.randn(w0.shape, generator=gen, device=device) * ACC_OUT0_SCALE).to(
        out_dtype
    )
    planned = base.prepare_wgrad(g1, x0, offs, out=mg0.clone(), accumulate=True)
    _assert_split_ordered(planned)
    _assert_acc_route(planned.plan, out_dtype)
    del planned
    fn = lambda a, b: pkg.cake_moe_grouped_gemm_nondet(
        a, b, offs, fuse_wgrad_accumulation=True
    )
    first = None
    for name, ckpt, early_stop in MAIN_GRAD_ARMS:
        w, dxs = _main_grad_steps(fn, x0, w0, mg0, (g1, g2), ckpt, early_stop, device)
        assert w.grad is None, name
        got = (w.main_grad, *dxs)
        if first is None:
            first = got
            continue
        for index, (have, want) in enumerate(zip(got, first, strict=True)):
            assert torch.equal(_bits(have), _bits(want)), (
                f"{name}: tensor {index} (main_grad, dX1, dX2) differs from the first arm "
                f"in {int((_bits(have) != _bits(want)).sum())} elements"
            )
    base._assert_tail_at_rest(device)
    dws = [base._reference("wgrad", g, x0, sizes) for g in (g1, g2)]
    _assert_added(first[0], mg0, dws, "main_grad")


# ---------------------------------------------------------------------------
# CUDA graph replay and concurrent streams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_graph_replays_give_the_eager_bits(kind, monkeypatch):
    """A prepared launch captured into a CUDA graph (which takes workspaces of its own: a
    fresh scheduler, uninitialized tail slots, zeroed tickets, the plan's even queue split)
    replays the eager launch's bits: the forward and the data gradient over ``K512_E3``,
    and the weight gradient on a 512-wide k tile shape whose ordered tail splits,
    overwriting (``out`` and the graph's slots filled with ``FILL_BYTE`` before every
    replay) or accumulating (``r`` replays from ``out0`` give the bits of ``r`` eager
    launches from ``out0``; fp32: the fused route, bf16: the composed one)."""
    base._fresh_stream_workspaces(monkeypatch)
    op = kind.split("_")[0]
    accumulate = "_acc_" in kind
    out_dtype = torch.float32 if kind.endswith("_f32") else torch.bfloat16
    sizes = list(base.K512_E3)
    if op == "wgrad":
        device = base._require_program("wgrad", out_dtype)
        shape = base._split_tail_shape(device, sizes, base._form_ks("main_halves"))
        assert shape is not None, "no candidate shape splits the ordered tail"
        n, k = shape
    else:
        device = base._require_program(op)
        n, k = 1024, 512
    x, w, g, offs = base._inputs(sizes, n, k, device, seed=8)
    out0 = None
    if op == "fwd":
        launch = base.prepare_fwd(x, w, offs)
        ref64 = base._reference("fwd", x, w, sizes)
    elif op == "dgrad":
        launch = base.prepare_dgrad(g, w, offs)
        ref64 = base._reference("dgrad", g, w, sizes)
    else:
        ref64 = base._reference("wgrad", g, x, sizes)
        if accumulate:
            gen = torch.Generator(device=device).manual_seed(9)
            out0 = torch.randn(len(sizes), n, k, generator=gen, device=device)
            out0 = out0.mul_(ACC_OUT0_SCALE).to(out_dtype)
            launch = base.prepare_wgrad(g, x, offs, out=out0.clone(), accumulate=True)
            _assert_acc_route(launch.plan, out_dtype)
        else:
            launch = base.prepare_wgrad(g, x, offs, out_dtype=out_dtype)
        _assert_split_ordered(launch)
    out = launch.out
    _settle_probe(launch, device)
    eager = {}
    if out0 is None:
        out.view(torch.uint8).fill_(FILL_BYTE)
        base._poison_tail_sums(device)
        launch.launch()
        eager[1] = _bits(out).clone()
        if op == "wgrad":
            base._assert_wgrad_close(out, ref64)
            base._assert_empty_groups_zero(out, sizes)
        else:
            base._assert_bf16_close(out, ref64)
    else:
        for count in GRAPH_ACC_REPLAYS:
            out.copy_(out0)
            for _ in range(count):
                base._poison_tail_sums(device)
                launch.launch()
            eager[count] = _bits(out).clone()
            if count == 1:
                _assert_added(out, out0, [ref64], "eager accumulate")
    if op == "wgrad":
        base._assert_tail_at_rest(device)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()  # a launch on a stream outside the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    captured = []
    launch_workspace = cb._launch_workspace

    def recording(ws_kind, device_, device_index, keep, size=0):
        buffer = launch_workspace(ws_kind, device_, device_index, keep, size)
        captured.append((ws_kind, keep, buffer))
        return buffer

    monkeypatch.setattr(cb, "_launch_workspace", recording)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    monkeypatch.setattr(cb, "_launch_workspace", launch_workspace)
    kinds = [ws_kind for ws_kind, _keep, _buffer in captured]
    assert captured and not any(keep for _kind, keep, _buffer in captured), kinds
    # The graph's own tail slots (filled with garbage between replays) and tickets.
    slots = [buffer for ws_kind, _keep, buffer in captured if ws_kind == "tail_sums"]
    tickets = [b for ws_kind, _keep, b in captured if ws_kind == "tail_tickets"]
    assert bool(slots) == bool(tickets) == (op == "wgrad"), kinds
    differ = []
    if out0 is None:
        for replay in range(GRAPH_REPLAYS):
            out.view(torch.uint8).fill_(FILL_BYTE)
            for buffer in slots:
                buffer.view(torch.uint8).fill_(FILL_BYTE)
            graph.replay()
            if not torch.equal(_bits(out), eager[1]):
                differ.append((replay, int((_bits(out) != eager[1]).sum())))
    else:
        for count in GRAPH_ACC_REPLAYS:
            out.copy_(out0)
            for _ in range(count):
                for buffer in slots:
                    buffer.view(torch.uint8).fill_(FILL_BYTE)
                graph.replay()
            if not torch.equal(_bits(out), eager[count]):
                differ.append((count, int((_bits(out) != eager[count]).sum())))
    torch.cuda.synchronize()
    print(f"GRAPH kind={kind} plan={json.dumps(_plan_record(launch), sort_keys=True)}")
    assert not differ, (
        f"replays differ from the eager bits (replay, elements): {differ}"
    )
    for buffer in tickets:
        assert int(buffer.view(torch.int32).count_nonzero()) == 0, (
            "the graph's tail tickets are not at rest after the replays"
        )


@pytest.mark.parametrize(
    "out_dtype,accumulate",
    STREAM_CASES,
    ids=[f"{base.DTYPE_NAMES[d]}-{'acc' if a else 'ovw'}" for d, a in STREAM_CASES],
)
def test_two_concurrent_streams_give_the_quiet_stream_bits(
    out_dtype, accumulate, monkeypatch
):
    """Two launches of the same weight gradient (prepared twice, so each has its own
    ``out`` and descriptor slots) on a 512-wide k tile shape whose ordered tail splits,
    launched interleaved on two streams in flight together over ``STREAM_ITERATIONS``
    iterations while a long-running kernel stays resident on a third stream and copies
    and GEMMs run on a fourth, each stream's slots filled with ``FILL_BYTE`` before each
    of its launches: every result has the bits of an isolated launch on a quiet stream
    (overwrite, or accumulate from ``out0``), each stream keeps its own slots and tickets,
    and both streams' tickets rest at zero."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    sizes = list(base.K512_E3)
    shape = base._split_tail_shape(device, sizes, base._form_ks("main_halves"))
    assert shape is not None, "no candidate shape splits the ordered tail"
    n, k = shape
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=12)
    out0 = None
    if accumulate:
        gen = torch.Generator(device=device).manual_seed(10)
        out0 = torch.randn(len(sizes), n, k, generator=gen, device=device)
        out0 = out0.mul_(ACC_OUT0_SCALE).to(out_dtype)

    def prepared():
        if accumulate:
            launch = base.prepare_wgrad(g, x, offs, out=out0.clone(), accumulate=True)
            _assert_acc_route(launch.plan, out_dtype)
        else:
            launch = base.prepare_wgrad(g, x, offs, out_dtype=out_dtype)
        _assert_split_ordered(launch)
        return launch

    def reset(launch):
        if accumulate:
            launch.out.copy_(out0)
        else:
            launch.out.view(torch.uint8).fill_(FILL_BYTE)

    main = torch.cuda.current_stream(device)
    isolated = prepared()
    quiet = torch.cuda.Stream(device)
    quiet.wait_stream(main)
    with torch.cuda.stream(quiet):
        reset(isolated)
        isolated.launch()
    quiet.synchronize()
    ref = _bits(isolated.out).clone()
    ref64 = base._reference("wgrad", g, x, sizes)
    if accumulate:
        _assert_added(isolated.out, out0, [ref64], "quiet accumulate")
    else:
        base._assert_wgrad_close(isolated.out, ref64)
    launches = [prepared(), prepared()]
    assert launches[1].plan == launches[0].plan == isolated.plan
    src = torch.randn(1 << 24, device=device)
    dst = torch.empty_like(src)
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    streams = [torch.cuda.Stream(device) for _ in range(4)]
    for stream in streams:
        stream.wait_stream(main)
    with torch.cuda.stream(streams[2]):
        torch.cuda._sleep(base.RESIDENT_SLEEP_CYCLES)
    outs = [[], []]
    poisoned = [0, 0]
    for _ in range(STREAM_ITERATIONS):
        for index in range(2):
            with torch.cuda.stream(streams[index]):
                reset(launches[index])
                poisoned[index] += int(base._poison_tail_sums(device, streams[index]))
                outs[index].append(_bits(launches[index].launch()).clone())
        with torch.cuda.stream(streams[3]):
            dst.copy_(src)
            torch.mm(a, a, out=c)
    for stream in streams[:2]:
        stream.synchronize()
    # Whether the weight gradients finished while the resident kernel still ran (a
    # timing fact, reported only: the bits are what the test asserts).
    overlap = not streams[2].query()
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    print(
        f"STREAMS dtype={out_dtype} accumulate={accumulate} overlap={overlap} "
        f"splits={isolated.plan['tail_splits']} poisoned={poisoned}"
    )
    differ = [
        (index, it, int((bits != ref).sum()))
        for index in range(2)
        for it, bits in enumerate(outs[index])
        if not torch.equal(bits, ref)
    ]
    assert not differ, (
        f"{len(differ)} launches differ from the quiet stream's bits "
        f"(stream, iteration, elements): {differ[:4]}"
    )
    # Every launch after a stream's first ran on slots filled with garbage.
    assert min(poisoned) >= STREAM_ITERATIONS - 1, poisoned
    for kind in cb.SIZED_STREAM_WORKSPACES:
        buffers = [cb.stream_workspace(kind, device, s) for s in streams[:2]]
        assert buffers[0] is not None and buffers[1] is not None, kind
        assert buffers[0].data_ptr() != buffers[1].data_ptr(), kind
    for stream in streams[:2]:
        base._assert_tail_at_rest(device, stream)


# ---------------------------------------------------------------------------
# Coverage of the routes by the cases
# ---------------------------------------------------------------------------


def test_route_coverage(record_property):
    """The cases reach, on this device, the 256-wide k tile's tail reduce and the 512-wide
    k tile's split ordered one-launch tail for both output dtypes (so the repeat test runs
    split ordered tails and cannot pass on unsplit tails alone), every 512-wide k tile
    plan takes the ordered tail, no launch is order-varying, and (where the plan probes it)
    the probed queue split of the data gradient is reached."""
    device = _device()
    arch = cb.arch_for(device)
    if not any(cb.generated_program_available(device, op=op) for op in OPS):
        pytest.skip(f"no generated program registered for {arch}")
    seen = {}
    varying = []
    for case in CASES:
        out_dtype = case.out_dtype if case.op == "wgrad" else None
        if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
            continue
        a, b, offs, out = _inputs(case, device)
        launch = _prepare(case, a, b, offs, out)
        plan = launch.plan
        if launch.order_varying:
            varying.append(case.label)
        key = (
            case.op,
            str(case.out_dtype).replace("torch.", ""),
            plan.get("tile_k"),
            "+".join(launch.stages) or "none",
            plan.get("queue_split"),
            _tail_form(launch) if case.op == "wgrad" else None,
        )
        seen.setdefault(key, []).append(case.label)
        del a, b, offs, out, launch
    rows = sorted([*map(str, key), len(labels)] for key, labels in seen.items())
    summary = dict(arch=arch, sm_count=cb.sm_count(device), rows=rows)
    record_property("coverage", json.dumps(summary))
    print("COVERAGE " + json.dumps(summary))
    assert not varying, f"order-varying launches: {varying[:8]}"
    for key, labels in seen.items():
        if key[0] == "wgrad" and key[2] == 512:
            assert key[5] in ("ordered_split", "ordered_whole"), (key, labels[:4])
        elif key[0] == "wgrad":
            assert key[5] in ("tail_reduce", "none"), (key, labels[:4])
    for dtype in ("bfloat16", "float32"):
        assert any(
            key[0] == "wgrad"
            and key[1] == dtype
            and key[2] == 256
            and key[5] == "tail_reduce"
            for key in seen
        ), f"no case reaches the {dtype} wgrad k256 program with the tail reduce"
        assert any(
            key[0] == "wgrad"
            and key[1] == dtype
            and key[2] == 512
            and key[5] == "ordered_split"
            for key in seen
        ), (
            f"no case splits the {dtype} weight gradient's ordered k512 tail on {arch} "
            f"with {cb.sm_count(device)} SMs: the repeat test would pass vacuously"
        )
    assert any(key[0] == "fwd" for key in seen)
    assert any(key[0] == "dgrad" for key in seen)
    if arch in HOST_PLAN_CONSTANTS["domain_split_archs"]:
        assert any(key[0] == "dgrad" and key[4] == "probed" for key in seen)
        assert any(key[0] == "dgrad" and key[4] == "even" for key in seen)


# ---------------------------------------------------------------------------
# Across fresh processes
# ---------------------------------------------------------------------------


def _digest_cases(labels):
    """sha256 of every case's output after a settled planned launch (the output and the
    stream's slots filled with ``FILL_BYTE`` first), or for an ``ACC_PREFIX`` label after
    ``ACC_DIGEST_STEPS`` accumulate launches from its ``out0``, with its plan and the split
    the launch passed."""
    device = torch.device("cuda", 0)
    digests = {}
    for label in labels:
        accumulate = label.startswith(ACC_PREFIX)
        case = CASE_BY_LABEL[label.removeprefix(ACC_PREFIX)]
        out_dtype = case.out_dtype if case.op == "wgrad" else None
        if not cb.generated_program_available(device, op=case.op, out_dtype=out_dtype):
            continue
        a, b, offs, out = _inputs(case, device)
        if accumulate:
            out.copy_(_acc_out0(label, case, device))
            launch = _prepare(case, a, b, offs, out, accumulate=True)
            split = None
            for _ in range(ACC_DIGEST_STEPS):
                base._poison_tail_sums(device)
                launch.launch()
        else:
            launch = _prepare(case, a, b, offs, out)
            split = _settle_probe(launch, device)
            out.view(torch.uint8).fill_(FILL_BYTE)
            base._poison_tail_sums(device)
            launch.launch()
        torch.cuda.synchronize(device)
        data = out.contiguous().view(torch.uint8).reshape(-1).cpu().numpy()
        plan = _plan_record(launch)
        digests[label] = dict(
            sha256=hashlib.sha256(data).hexdigest(),
            plan={k: v for k, v in plan.items() if k not in ("c0", "c1")},
            stages=list(launch.stages),
            split=list(split) if split is not None else None,
        )
        del a, b, offs, out, launch, data
    return digests


def _worker_main(labels):
    print("DIGESTS_JSON " + json.dumps(_digest_cases(labels), sort_keys=True))


def _fresh_process_digests(labels):
    """Digests of ``labels`` from a new interpreter that imports this module by name."""
    here = Path(__file__).resolve()
    root = here.parents[2]
    module = ".".join(here.relative_to(root).with_suffix("").parts)
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r}); import importlib; "
        f"importlib.import_module({module!r})._worker_main(sys.argv[1:])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, *labels],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    lines = [x for x in proc.stdout.splitlines() if x.startswith("DIGESTS_JSON ")]
    assert len(lines) == 1, proc.stdout[-2000:]
    return json.loads(lines[0][len("DIGESTS_JSON ") :])


def test_fresh_processes_give_the_same_bits(record_property):
    """Two fresh processes (each with its own JIT load, stream workspaces and queue probe)
    and this one give the same output bits for every case of ``XPROC_LABELS`` and every
    accumulate of ``ACC_DIGEST_LABELS``; on this device a split ordered tail is among
    the overwrite cases and among the accumulates."""
    device = _device()
    labels = [*XPROC_LABELS, *ACC_DIGEST_LABELS]
    here = _digest_cases(labels)
    assert here, "no case has a generated program on this device"
    split = sorted(
        label
        for label, d in here.items()
        if d["plan"].get("ordered_tail") and d["plan"].get("tail_splits", 1) > 1
    )
    arch = cb.arch_for(device)
    assert any(not x.startswith(ACC_PREFIX) for x in split), (
        f"no digested overwrite splits its ordered tail on {arch}"
    )
    assert any(x.startswith(ACC_PREFIX) for x in split), (
        f"no digested accumulate splits its ordered tail on {arch}"
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    first = _fresh_process_digests(labels)
    second = _fresh_process_digests(labels)
    differ = []
    for label, d in here.items():
        for name, other in (("first", first), ("second", second)):
            o = other.get(label)
            if o is None or o["sha256"] != d["sha256"] or o["plan"] != d["plan"]:
                differ.append((label, name))
    splits = sorted(
        {
            json.dumps([d["split"] for d in (here[x], first[x], second[x])])
            for x in here
            if here[x]["split"] is not None and x in first and x in second
        }
    )
    summary = dict(
        arch=arch,
        cases=len(here),
        processes=3,
        split_ordered_tail_cases=len(split),
        differ=differ,
        probed_splits_here_first_second=splits,
    )
    record_property("fresh_processes", json.dumps(summary, sort_keys=True))
    print("XPROC " + json.dumps(summary, sort_keys=True))
    assert not differ, (
        f"{len(differ)} case digests differ across processes: {differ[:8]}"
    )
