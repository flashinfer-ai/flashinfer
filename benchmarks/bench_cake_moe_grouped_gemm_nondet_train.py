# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the non-deterministic ragged BF16 MoE grouped GEMM on SM100/SM103/SM107.

The 58 MoE projection rows of the deterministic package's benchmark: gate/up
(N=4096, K=6144) and down (N=6144, K=2048) projections, E=4 with balanced /
skewed / one-empty group distributions at two token counts, E=8/16/32 balanced,
and the fp32 weight gradient of the balanced E=4 rows; plus one checkpointed
training step per (projection, distribution) -- ``2 x forward + backward``
through the autograd wrapper under ``torch.utils.checkpoint`` (non-reentrant,
early stop off so that every arm recomputes its forward), both gradients included.
Opt-in autograd rows over the same distributions: ``step_es``, that step with
checkpoint early stop on (the torch default: an arm whose inputs are saved
before its GEMM launches, ``cake_det``, ``cake_nondet`` and ``torch._grouped_mm``,
skips the recomputed forward); ``block`` / ``block_es``, one checkpointed MoE expert
block with early stop off / on -- gate_up, ``silu(a) * b`` on its two halves,
down, with the gradients of ``x``, ``w1`` and ``w2`` -- in which only the
recompute of the down GEMM can be skipped.  The rate of these rows counts the
useful work only (forward, data and weight gradient of each GEMM); the ``step``
rows keep their earlier count, which includes the recomputed forward.  After every
other row: E=1 and E=32 at the two E=4 token counts (112,576 and 136,064 rows; E=32
in 32-row aligned groups, the last taking the remainder, as in the balanced E=4
rows), each projection, with fwd, dgrad, the bf16 wgrad and the step.

Arms: ``cake_det`` (the deterministic package,
``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``), ``cake_nondet``
(``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train``),
``torch._grouped_mm``, a per-group cuBLAS loop with host-known bounds and
``flashinfer.grouped_mm.grouped_mm_bf16`` (cuDNN; fwd / dgrad only); opt-in
``cake_nondet_inforward``, ``cake_nondet`` through a direct
``CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)``, whose forward GEMM launches
inside ``forward`` so that a checkpoint always recomputes it (autograd rows only; when
it runs with ``cake_nondet``, the JSON records the elements in which their outputs and
gradients differ -- none is expected: every output repeats bit for bit for the same
inputs, shapes, device SM count and configuration).  ``--census`` adds the
CUDA kernel count of one call per arm (``torch.profiler``).  Each row reports
the median time with a cold L2 between
iterations (and the 10th / 90th percentiles), the peak device memory one
preparation and launch takes beyond its inputs and output (the Cake arms'
kept stream workspaces included: they are dropped before the measurement) and
the memory it retains, the Cake plans (stages, tail split, the form of a split
tail: ``ordered_tail`` / ``unordered_tail``), the error of the ``cake_nondet`` output
against ``cake_det``, and
the ``cake_nondet`` / ``cake_det`` time ratio.  Consecutive rows run the arms in
alternating order (the JSON records it as ``arm_order``); an arm with no
program for the device is dropped.

Weight-gradient accumulation (opt-in ``--ops acc,bwd_acc,wgrad_acc``): the rows
of the step's distributions plus ``short16`` (E=32, 16 groups of 64-484 rows
beside 16 of 7918: the 512-wide k tile with short groups), each projection.
``acc`` is one checkpointed micro-step (non-reentrant, early stop off,
``2 x fwd + bwd``) whose weight gradient goes to the parameter's gradient or
``main_grad``; ``bwd_acc`` is the backward alone (the data gradient and the
weight gradient with its accumulation, replayed on one forward with
``retain_graph``); ``wgrad_acc`` is the weight gradient alone (prepared
launches).  Arms (the non-deterministic package unless named ``det_``):
``ovw`` (bf16 ``w.grad`` overwritten every micro-step), ``acc``
(AccumulateGrad: bf16 ``w.grad`` added into), ``mgB`` (Megatron's unfused
path: bf16 ``dW``, then the fp32 ``main_grad.add_``), ``mgF`` (fp32 ``dW``
into a temporary, then ``main_grad.add_``: the unfused route of this package),
``mgO`` (fp32 ``dW`` written over ``main_grad``: the lower bound of a fused
add), ``mgX`` (``fuse_wgrad_accumulation=True`` / ``accumulate=True`` into the
fp32 ``main_grad``, with Megatron's ``grad_added_to_main_grad`` placeholder),
``mgXn`` (``mgX`` without the attribute: no placeholder), ``mgX_zero``
(``mgX`` with ``zero_out_wgrad``), ``accX`` (the same into a bf16
``main_grad``: the composed route, a bf16 ``dW`` and the same-dtype add),
``det_acc`` (the deterministic package: its fp32 ``dW`` into a temporary, then
``main_grad.add_``, the deterministic weight gradient composed with the add; in
the step and the backward, through a Function running the deterministic
forward, data gradient and weight gradient), ``det_mgB`` (the
deterministic autograd wrapper's bf16 ``w.grad``, then the fp32
``main_grad.add_``; the deterministic bf16 ``dW`` and the add in
``wgrad_acc``) and ``det_mgX`` (the deterministic package's own accumulate:
``cake_moe_grouped_gemm_det(..., fuse_wgrad_accumulation=True)`` into the fp32
``main_grad`` with the ``grad_added_to_main_grad`` placeholder;
``cake_moe_grouped_gemm_det_train_prepare_wgrad(..., accumulate=True)`` in ``wgrad_acc``).
The peak memory includes the placeholder gradient and every temporary.  A
fused arm records ``plan["accumulate"]`` and, where the device serves an
accumulate build of its dtype, fails unless the row is fused
(``--allow-composed`` turns that into a note).  The JSON's ``acc_ratios``
holds ``mgX/mgF``, ``mgX/mgB``, ``mgX/mgO``, ``mgX/det_acc``,
``mgX/det_mgB``, ``accX/acc``, ``mgF/mgO``, ``det_mgX/det_acc``,
``det_mgX/det_mgB`` and ``mgX/det_mgX``.  The non-Cake baselines of
the accumulation rows (Transformer Engine, ``torch._grouped_mm``, per-expert
cuBLAS) are not arms of this benchmark.

Usage::

    python benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py [--ops fwd,wgrad,step] [--rows e4_m112576] [--arms cake_det,cake_nondet] [--cupti] [--json out.json]
    python benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py --ops step,step_es,block,block_es --arms cake_det,cake_nondet,cake_nondet_inforward,torch_grouped_mm [--census]
    python benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py --ops acc,bwd_acc,wgrad_acc --arms ovw,acc,mgB,mgF,mgO,mgX,mgXn,mgX_zero,accX,det_acc,det_mgB,det_mgX
"""

import argparse
import json
import math

import torch
import torch.profiler
import torch.utils.checkpoint

import flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train as pkg
from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
    cake_backend as cb,
)
import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as det_pkg
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
    cake_backend as det_cb,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train.cake_autograd import (
    cake_moe_grouped_gemm_det,
)
from flashinfer.testing import bench_gpu_time

PROJECTIONS = {"gate_up": (4096, 6144), "down": (6144, 2048)}  # (N, K)
BLOCK = "gate_up_down"  # the expert block's projections: gate_up, silu(a) * b, down
OPS = (
    "fwd",
    "dgrad",
    "wgrad",
    "step",
    "step_es",
    "block",
    "block_es",
    "acc",
    "bwd_acc",
    "wgrad_acc",
)
DEFAULT_OPS = OPS[:4]  # the earlier row set
AUTOGRAD_OPS = ("step", "step_es", "block", "block_es")
BLOCK_OPS = ("block", "block_es")
# The weight-gradient accumulation rows: a checkpointed micro-step, its backward alone, and the weight
# gradient alone.
ACC_OPS = ("acc", "bwd_acc", "wgrad_acc")
AUTOGRAD_NOTES = {
    "step": "checkpointed 2 x fwd + bwd",
    "step_es": "checkpointed step, early stop on (the recompute skips the GEMM only "
    "where its inputs are saved before it launches)",
    "block": "checkpointed gate_up, silu(a) * b, down: 2 x fwd + bwd",
    "block_es": "checkpointed gate_up, silu(a) * b, down, early stop on (gate_up is "
    "always recomputed; down only where its inputs are saved before it launches)",
}
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
# E=32 over EXPERT_COUNT_SUM_M rows, 16 short groups beside 16 long ones: an average of 4096 rows keeps the
# 512-wide k tile, whose accumulate epilogue then runs tiles of short mainloops.
ACC_SHORT_E32 = [64 + 28 * i for i in range(16)] + [7918] * 16
# The expert counts of the rows after every other row, at the token counts of E4_GROUPS.
RANGE_EXPERT_COUNTS = (1, 32)
ARMS = (
    "cake_det",
    "cake_nondet",
    "torch_grouped_mm",
    "per_expert_cublas",
    "cudnn_grouped_mm",
)
ACC_ARMS = (
    "ovw",
    "acc",
    "mgB",
    "mgF",
    "mgO",
    "mgX",
    "mgXn",
    "mgX_zero",
    "accX",
    "det_acc",
    "det_mgB",
    "det_mgX",
)
# The deterministic-package arms of the accumulation rows.
DET_ACC_ARMS = ("det_acc", "det_mgB", "det_mgX")
# The arms that add with the package's accumulate mode, and the dtype of their main_grad.
FUSED_ARMS = {
    "mgX": torch.float32,
    "mgXn": torch.float32,
    "mgX_zero": torch.float32,
    "accX": torch.bfloat16,
}
ACC_RATIOS = (
    ("mgX", "mgF"),
    ("mgX", "mgB"),
    ("mgX", "mgO"),
    ("mgX", "det_acc"),
    ("mgX", "det_mgB"),
    ("accX", "acc"),
    ("mgF", "mgO"),
    ("det_mgX", "det_acc"),
    ("det_mgX", "det_mgB"),
    ("mgX", "det_mgX"),
)
DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}
MIB = float(1 << 20)
# A fused arm on a row its device could fuse must be fused (``--allow-composed`` clears it).
REQUIRE_FUSED = True


class AccumulatePlanError(AssertionError):
    """A fused arm's row took the composed route on a device that serves the accumulate build."""


def aligned_balanced(sum_m, e, align=32):
    """``e`` groups of ``align``-row aligned size, the last taking the remainder (the
    rule of the balanced E=4 rows)."""
    size = (sum_m // e) // align * align
    return [size] * (e - 1) + [sum_m - size * (e - 1)]


def build_rows():
    rows = []

    def add(op, proj, sizes, out_dtype, group):
        if proj == BLOCK:
            # The block's output and input width: the hidden size.
            n, k = PROJECTIONS["down"][0], PROJECTIONS["gate_up"][1]
        else:
            n, k = PROJECTIONS[proj]
        dist = group if group in ("skewed", "one_empty", "short16") else "balanced"
        dtype_tag = "f32" if out_dtype == torch.float32 else "bf16"
        label = f"{op}_{proj}_e{len(sizes)}_m{sum(sizes)}_{dist}_{dtype_tag}"
        rows.append(
            dict(
                label=label,
                op=op,
                proj=proj,
                sizes=sizes,
                N=n,
                K=k,
                out_dtype=out_dtype,
            )
        )

    for proj in PROJECTIONS:
        for op in OPS[:3]:
            for dists in E4_GROUPS.values():
                for dist, sizes in dists.items():
                    add(op, proj, sizes, torch.bfloat16, dist)
    for proj in PROJECTIONS:
        for dists in E4_GROUPS.values():
            add("wgrad", proj, dists["balanced"], torch.float32, "balanced")
    for e in EXPERT_COUNTS:
        for proj in PROJECTIONS:
            for op in OPS[:3]:
                add(op, proj, [EXPERT_COUNT_SUM_M // e] * e, torch.bfloat16, "balanced")
    assert len(rows) == 58
    # One checkpointed training step per projection and group distribution.
    for proj in PROJECTIONS:
        for dists in E4_GROUPS.values():
            for dist, sizes in dists.items():
                add("step", proj, sizes, torch.bfloat16, dist)
        for e in EXPERT_COUNTS:
            add("step", proj, [EXPERT_COUNT_SUM_M // e] * e, torch.bfloat16, "balanced")
    assert len(rows) == 76
    # The step's distributions: the step with checkpoint early stop on, and one
    # checkpointed expert block with early stop off / on.
    step_dists = [
        (sizes, dist) for dists in E4_GROUPS.values() for dist, sizes in dists.items()
    ] + [([EXPERT_COUNT_SUM_M // e] * e, "balanced") for e in EXPERT_COUNTS]
    for proj in PROJECTIONS:
        for sizes, dist in step_dists:
            add("step_es", proj, sizes, torch.bfloat16, dist)
    for op in BLOCK_OPS:
        for sizes, dist in step_dists:
            add(op, BLOCK, sizes, torch.bfloat16, dist)
    # The weight-gradient accumulation rows: the step's distributions and short16.
    for op in ACC_OPS:
        for proj in PROJECTIONS:
            for sizes, dist in step_dists + [(ACC_SHORT_E32, "short16")]:
                add(op, proj, sizes, torch.bfloat16, dist)
    # E=1 and E=32 at both ends of the token range (the E=4 token counts): fwd, dgrad,
    # the bf16 weight gradient and the step per projection.  They come after every other
    # row, so the earlier rows keep their order and their arm order.
    for e in RANGE_EXPERT_COUNTS:
        for sum_m in E4_GROUPS:
            sizes = aligned_balanced(sum_m, e)
            for proj in PROJECTIONS:
                for op in OPS[:3] + ("step",):
                    add(op, proj, sizes, torch.bfloat16, "balanced")
    assert len(rows) == 204
    return rows


def row_flops(row):
    """The row's FLOPs.  An autograd row counts the forward, data and weight gradient
    of each GEMM (useful work, no recompute), except ``step``, which keeps its earlier
    count with the recomputed forward included."""
    sum_m = sum(row["sizes"])
    projs = ("gate_up", "down") if row["op"] in BLOCK_OPS else (row["proj"],)
    flops = sum(2.0 * sum_m * n * k for n, k in (PROJECTIONS[p] for p in projs))
    if row["op"] in ("step", "acc"):
        return flops * 4  # two forward passes, the data and the weight gradient
    if row["op"] == "bwd_acc":
        return flops * 2  # the data and the weight gradient
    if row["op"] in AUTOGRAD_OPS:
        return flops * 3
    return flops


def make_inputs(row, device, seed=0):
    torch.manual_seed(seed)
    sizes, n, k = row["sizes"], row["N"], row["K"]
    sum_m, e = sum(sizes), len(sizes)
    if row["op"] in BLOCK_OPS:
        # x [sum_m, H], w1 (gate_up) [E, 2 I, H], w2 (down) [E, H, I], g [sum_m, H].
        (n1, k1), (n2, k2) = PROJECTIONS["gate_up"], PROJECTIONS["down"]
        s1, s2 = 1.0 / math.sqrt(k1), 1.0 / math.sqrt(k2)
        x = torch.randn(sum_m, k1, dtype=torch.bfloat16, device=device).mul_(s1)
        w1 = torch.randn(e, n1, k1, dtype=torch.bfloat16, device=device).mul_(s1)
        w2 = torch.randn(e, n2, k2, dtype=torch.bfloat16, device=device).mul_(s2)
        g = torch.randn(sum_m, n2, dtype=torch.bfloat16, device=device)
        offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
            0, dtype=torch.int32
        )
        return dict(x=x, w1=w1, w2=w2, g=g, offs=offs, out=None)
    scale = 1.0 / math.sqrt(k)
    x = torch.randn(sum_m, k, dtype=torch.bfloat16, device=device).mul_(scale)
    w = torch.randn(e, n, k, dtype=torch.bfloat16, device=device).mul_(scale)
    g = torch.randn(sum_m, n, dtype=torch.bfloat16, device=device)
    offs = torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    op = row["op"]
    if op in ACC_OPS:
        # The accumulators the arms add into (kept across the timed calls, as a main_grad is).
        main_grad32 = torch.randn(e, n, k, dtype=torch.float32, device=device).mul_(
            1e-3
        )
        return dict(
            x=x,
            w=w,
            g=g,
            offs=offs,
            out=None,
            main_grad32=main_grad32,
            main_grad16=main_grad32.to(torch.bfloat16),
        )
    if op == "fwd":
        out = torch.empty(sum_m, n, dtype=torch.bfloat16, device=device)
    elif op == "dgrad":
        out = torch.empty(sum_m, k, dtype=torch.bfloat16, device=device)
    elif op == "wgrad":
        out = torch.empty(e, n, k, dtype=row["out_dtype"], device=device)
    else:
        out = None
    return dict(x=x, w=w, g=g, offs=offs, out=out)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return [
        (e, s, t) for e, (s, t) in enumerate(zip([0] + ends[:-1], ends, strict=True))
    ]


def _plan_note(launch):
    plan = launch.plan
    keys = (
        "tile_n",
        "tile_k",
        "num_tail",
        "tail_splits",
        "ordered_tail",
        "unordered_tail",
        "wave_mode",
    )
    shown = {key: plan[key] for key in keys if key in plan}
    return f"{launch.record_name} stages={'+'.join(launch.stages) or 'none'} {shown}"


def _checkpointed(region, t, names, early_stop):
    """A callable running ``region(*leaves, offs)`` under a non-reentrant checkpoint
    and its backward with ``t["g"]``, the leaves being ``t[name]`` requiring grad.  It
    returns the region's output and the leaves, whose ``.grad`` holds that call's
    gradients."""
    leaves = [t[name].detach().requires_grad_(True) for name in names]
    g, offs = t["g"], t["offs"]

    def run():
        for leaf in leaves:
            leaf.grad = None
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            y = torch.utils.checkpoint.checkpoint(
                region, *leaves, offs, use_reentrant=False
            )
            y.backward(g)
        return y, leaves

    return run


def _step(autograd_fn, t, early_stop=False):
    """A non-reentrant checkpointed ``y = autograd_fn(x, w, offs)`` and its backward
    (which replays the forward), with both gradients.

    Early stop off (the ``step`` rows): every arm recomputes its forward GEMM,
    ``2 x forward + backward``.  On (the torch default, the ``step_es`` rows): the
    recompute stops once the region saved its last tensor, so an arm whose inputs are
    saved before its GEMM launches (``torch._grouped_mm``, ``cake_moe_grouped_gemm_det``,
    ``cake_moe_grouped_gemm_nondet``) does one GEMM less, and one that launches inside
    ``Function.forward`` (``cake_nondet_inforward``) still recomputes it."""
    return _checkpointed(autograd_fn, t, ("x", "w"), early_stop)


def _block(mm, t, early_stop=False):
    """One checkpointed MoE expert block and its backward, with the gradients of ``x``,
    ``w1`` and ``w2``: ``a, b = mm(x, w1, offs).chunk(2, -1)`` (gate_up), then
    ``mm(silu(a) * b, w2, offs)`` (down).  SiLU and the product save tensors after the
    gate_up GEMM, so with early stop on only the recompute of down can be skipped."""

    def region(x, w1, w2, offs):
        a, b = mm(x, w1, offs).chunk(2, -1)
        return mm(torch.nn.functional.silu(a) * b, w2, offs)

    return _checkpointed(region, t, ("x", "w1", "w2"), early_stop)


def _autograd(mm, row, t):
    """The timed callable of an autograd row with ``mm(x, w, offs)`` as its GEMM."""
    op = row["op"]
    if op in BLOCK_OPS:
        return _block(mm, t, early_stop=op == "block_es")
    return _step(mm, t, early_stop=op == "step_es")


def _no_acc_rows(op):
    if op in ACC_OPS:
        raise NotImplementedError("the accumulation rows run the accumulation arms")


def _cake_arm(backend, row, t):
    op = row["op"]
    _no_acc_rows(op)
    deterministic = backend is det_cb
    if op in AUTOGRAD_OPS:
        if deterministic:
            fn = _autograd(
                lambda x, w, offs: cake_moe_grouped_gemm_det(x, w, offs), row, t
            )
        else:
            fn = _autograd(
                lambda x, w, offs: pkg.cake_moe_grouped_gemm_nondet(x, w, offs),
                row,
                t,
            )
        return fn, None, f"{AUTOGRAD_NOTES[op]} (autograd)"
    args = (t["x"], t["w"]) if op == "fwd" else (t["g"], t["w"])
    if op == "wgrad":
        args = (t["g"], t["x"])
    if deterministic:
        prepare = {
            "fwd": det_cb.cake_moe_grouped_gemm_det_train_prepare_forward,
            "dgrad": det_cb.cake_moe_grouped_gemm_det_train_prepare_dgrad,
            "wgrad": det_cb.cake_moe_grouped_gemm_det_train_prepare_wgrad,
        }[op]
        launch = prepare(*args, t["offs"], out=t["out"])
    else:
        prepare = {
            "fwd": cb.cake_moe_grouped_gemm_nondet_train_prepare_forward,
            "dgrad": cb.cake_moe_grouped_gemm_nondet_train_prepare_dgrad,
            "wgrad": cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad,
        }[op]
        launch = prepare(*args, t["offs"], out=t["out"])
    return launch.launch, launch, _plan_note(launch)


def cake_det_arm(row, t):
    return _cake_arm(det_cb, row, t)


def cake_nondet_arm(row, t):
    return _cake_arm(cb, row, t)


def cake_nondet_inforward_arm(row, t):
    """``cake_nondet`` through a direct ``CakeMoEGroupedGemmNondetFunction.apply``: the
    forward GEMM launches inside ``forward``, before autograd saves the inputs, as
    before the wrapper deferred it (the internal control of the autograd rows)."""
    op = row["op"]
    if op not in AUTOGRAD_OPS:
        raise NotImplementedError("the in-forward control has autograd rows only")
    _no_acc_rows(op)
    fn = _autograd(
        lambda x, w, offs: pkg.CakeMoEGroupedGemmNondetFunction.apply(x, w, offs),
        row,
        t,
    )
    return (
        fn,
        None,
        f"{AUTOGRAD_NOTES[op]} (direct CakeMoEGroupedGemmNondetFunction.apply)",
    )


def torch_grouped_mm_arm(row, t):
    op, fn = row["op"], torch._grouped_mm
    _no_acc_rows(op)
    if op in AUTOGRAD_OPS:

        def mm(a, b, o):
            return fn(a, b.transpose(1, 2), offs=o)

        return _autograd(mm, row, t), None, f"{AUTOGRAD_NOTES[op]} (torch autograd)"
    x, w, g, offs = t["x"], t["w"], t["g"], t["offs"]
    if op == "fwd":
        return (
            (lambda: fn(x, w.transpose(1, 2), offs=offs)),
            None,
            "x @ w.transpose(1, 2)",
        )
    if op == "dgrad":
        return (lambda: fn(g, w, offs=offs)), None, "g @ w"
    kwargs = {"out_dtype": torch.float32} if row["out_dtype"] == torch.float32 else {}
    try:
        probe = fn(g.t(), x, offs=offs, **kwargs)
        torch.cuda.synchronize()
        if tuple(probe.shape) != tuple(t["out"].shape):
            raise RuntimeError(f"output shape {tuple(probe.shape)}")
        del probe
        return (
            (lambda: fn(g.t(), x, offs=offs, **kwargs)),
            None,
            "g.t() @ x (transposed view accepted)",
        )
    except Exception as exc:  # noqa: BLE001 - the copy inside the timed callable is recorded
        note = f"g.t().contiguous() inside the timed callable ({type(exc).__name__})"
        return (lambda: fn(g.t().contiguous(), x, offs=offs, **kwargs)), None, note


def per_expert_cublas_arm(row, t):
    op = row["op"]
    _no_acc_rows(op)
    if op in AUTOGRAD_OPS:
        raise NotImplementedError("no autograd form of the per-group loop")
    x, w, g, out = t["x"], t["w"], t["g"], t["out"]
    bounds = _bounds(row["sizes"])
    fp32 = out.dtype == torch.float32

    def mm(a, b, dst):
        if fp32:
            torch.mm(a, b, out_dtype=torch.float32, out=dst)
        else:
            torch.matmul(a, b, out=dst)

    if op == "fwd":

        def run():
            for e, s, t_ in bounds:
                if t_ > s:
                    mm(x[s:t_], w[e].t(), out[s:t_])

    elif op == "dgrad":

        def run():
            for e, s, t_ in bounds:
                if t_ > s:
                    mm(g[s:t_], w[e], out[s:t_])

    else:

        def run():
            for e, s, t_ in bounds:
                if t_ > s:
                    mm(g[s:t_].t(), x[s:t_], out[e])
                else:
                    out[e].zero_()

    launches = sum(1 for _, s, t_ in bounds if t_ > s)
    return run, None, f"{launches} GEMM launches, host-known bounds"


def cudnn_grouped_mm_arm(row, t):
    from flashinfer.grouped_mm import grouped_mm_bf16

    op = row["op"]
    _no_acc_rows(op)
    if op == "wgrad" or op in AUTOGRAD_OPS:
        raise NotImplementedError(
            "grouped_mm_bf16 has no K-offset (weight gradient) form"
        )
    x, w, g, offs, out = t["x"], t["w"], t["g"], t["offs"], t["out"]
    m_indptr = torch.cat((offs.new_zeros(1), offs))
    a = x if op == "fwd" else g
    b = w if op == "fwd" else w.transpose(1, 2)

    def run():
        grouped_mm_bf16(a, b, m_indptr, out=out, out_dtype=out.dtype, tactic=-1)

    run()
    torch.cuda.synchronize()
    return run, None, "cuDNN heuristic plan (tactic=-1), W read in place"


class _MainGradMm(torch.autograd.Function):
    """The caller-side Function of the ``mgF`` / ``mgO`` / ``det_acc`` arms: the forward and data
    gradient, and the weight gradient written by the caller into ``w.main_grad`` (``mgF``: this
    package's fp32 ``dW`` into a temporary, then ``main_grad.add_``; ``mgO``: its fp32 ``dW`` written
    over ``main_grad``; ``det_acc``: the deterministic package's forward, data gradient and fp32
    ``dW`` into a temporary, then ``main_grad.add_``).  Returns no gradient for ``w``."""

    @staticmethod
    def forward(ctx, x, w, offs, mode):
        if mode == "det_acc":
            y = det_pkg.cake_moe_grouped_gemm_det_train_forward(x, w, offs)
        else:
            y = pkg.cake_moe_grouped_gemm_nondet_train_forward(x, w, offs)
        ctx.save_for_backward(x, w, offs)
        ctx.w, ctx.mode = w, mode
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w, offs = ctx.saved_tensors
        # Bind the CUDA context on the autograd thread (``cake_autograd``).
        torch.cuda.current_stream(x.device).query()
        g = grad_y.contiguous()
        det = ctx.mode == "det_acc"
        dx = None
        if ctx.needs_input_grad[0]:
            if det:
                dx = det_pkg.cake_moe_grouped_gemm_det_train_dgrad(g, w, offs)
            else:
                dx = pkg.cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs)
        main_grad = ctx.w.main_grad
        if det:
            main_grad.add_(
                det_pkg.cake_moe_grouped_gemm_det_train_wgrad(
                    g, x, offs, out_dtype=torch.float32
                )
            )
        elif ctx.mode == "mgF":
            main_grad.add_(
                pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
                    g, x, offs, out_dtype=torch.float32
                )
            )
        else:
            pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out=main_grad)
        return dx, None, None, None


def _accumulate_mode(t, main_grad):
    """``(plan["accumulate"], reason)`` of a weight gradient of the row's operands adding into
    ``main_grad``, from the host rules (nothing is prepared or allocated)."""
    arch = cb.arch_for(main_grad.device)
    c = cb.host_plan_constants()
    e, _n, k = (int(v) for v in main_grad.shape)
    sum_m = int(t["x"].shape[0])
    tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
    name = cb.select_module(
        "wgrad", arch, out_dtype=DTYPE_NAMES[main_grad.dtype], tile_k=tile_k
    )
    stage, _wave = cb.select_wgrad_stage(
        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=e
    )
    return cb.wgrad_accumulate_mode(
        name, stage, tile_k, arch, aligned=cb._accumulate_aligned(main_grad)
    )


def _accumulate_note(arm, row, t):
    """The plan note of a fused arm; raises ``AccumulatePlanError`` when the row composes on a
    device whose registry serves an accumulate build of the arm's dtype (also when that build
    cannot load here, ``"min_cuda"``), unless ``--allow-composed``."""
    main_grad = t["main_grad32" if FUSED_ARMS[arm] == torch.float32 else "main_grad16"]
    mode, reason = _accumulate_mode(t, main_grad)
    servable = cb.acc_stage_served(
        DTYPE_NAMES[main_grad.dtype], cb.arch_for(main_grad.device)
    )
    if REQUIRE_FUSED and mode != "fused" and servable:
        raise AccumulatePlanError(
            f"{row['label']} {arm}: plan['accumulate'] is {mode!r} ({reason}), the device serves "
            "the accumulate build"
        )
    return f"accumulate={mode}" + (f" ({reason})" if reason else "")


def _acc_autograd(arm, t, op):
    """The timed callable of ``arm`` on an autograd accumulation row, returning ``(y, leaves)``; the
    parameter's gradient goes where the arm sends it.  ``acc``: one checkpointed micro-step
    (non-reentrant, early stop off: ``2 x fwd + bwd``).  ``bwd_acc``: the backward alone, replayed
    with ``retain_graph`` on one forward made here (outside the timing)."""
    x = t["x"].detach().requires_grad_(True)
    w = t["w"].detach().requires_grad_(True)
    g, offs = t["g"], t["offs"]
    main_grad = t["main_grad16" if arm == "accX" else "main_grad32"]
    if arm in ("mgF", "mgO", "det_acc", "det_mgX") or arm in FUSED_ARMS:
        w.main_grad = main_grad
    if arm in ("mgX", "mgX_zero", "accX", "det_mgX"):
        w.grad_added_to_main_grad = False
        w.zero_out_wgrad = arm == "mgX_zero"
    if arm in ("ovw", "acc", "mgB"):

        def mm(a, b, o):
            return pkg.cake_moe_grouped_gemm_nondet(a, b, o)

    elif arm == "det_mgB":

        def mm(a, b, o):
            return cake_moe_grouped_gemm_det(a, b, o)

    elif arm == "det_mgX":

        def mm(a, b, o):
            return cake_moe_grouped_gemm_det(a, b, o, fuse_wgrad_accumulation=True)

    elif arm in ("mgF", "mgO", "det_acc"):

        def mm(a, b, o):
            return _MainGradMm.apply(a, b, o, arm)

    else:

        def mm(a, b, o):
            return pkg.cake_moe_grouped_gemm_nondet(
                a, b, o, fuse_wgrad_accumulation=True
            )

    y_once = mm(x, w, offs) if op == "bwd_acc" else None

    def run():
        x.grad = None
        if arm != "acc":
            # Megatron's DDP drops the placeholder; the overwrite arms start from no gradient.
            w.grad = None
        if y_once is not None:
            y_once.backward(g, retain_graph=True)
            y = y_once
        else:
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                y = torch.utils.checkpoint.checkpoint(
                    mm, x, w, offs, use_reentrant=False
                )
                y.backward(g)
        if arm in ("mgB", "det_mgB"):
            main_grad.add_(w.grad)
        return y, [x, w]

    if arm == "acc":
        run()  # every timed micro-step then adds into an existing w.grad
    return run


def _acc_wgrad(arm, t):
    """The weight gradient of ``arm`` alone, as prepared launches into the arm's accumulator."""
    g, x, offs = t["g"], t["x"], t["offs"]
    mg32, mg16 = t["main_grad32"], t["main_grad16"]

    def prepare(out, accumulate=False):
        return cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
            g, x, offs, out=out, accumulate=accumulate
        )

    if arm in ("mgXn", "mgX_zero"):
        raise NotImplementedError(
            "the placeholder-gradient arms have autograd rows only"
        )
    if arm == "det_mgX":
        # The deterministic package's accumulate: its fp32 dW into the launch's scratch,
        # then the add.
        return det_cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out=mg32, accumulate=True
        ).launch
    if arm in DET_ACC_ARMS:
        # The deterministic package's dW (fp32: the composed deterministic accumulate; bf16:
        # Megatron's unfused path) into a temporary, then the fp32 add into main_grad.
        dtype = torch.float32 if arm == "det_acc" else torch.bfloat16
        tmp = torch.empty(mg32.shape, dtype=dtype, device=mg32.device)
        det_launch = det_cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out=tmp
        )

        def run_det():
            det_launch.launch()
            mg32.add_(tmp)

        return run_det
    if arm == "ovw":
        return prepare(torch.empty_like(mg16)).launch
    if arm == "mgO":
        return prepare(mg32).launch
    if arm == "mgX":
        return prepare(mg32, accumulate=True).launch
    if arm == "accX":
        return prepare(mg16, accumulate=True).launch
    # Overwrite into a temporary, then the add: acc (bf16 += bf16), mgB (fp32 += bf16), mgF (fp32 += fp32).
    tmp = torch.empty_like(mg32 if arm == "mgF" else mg16)
    launch = prepare(tmp)
    target = mg16 if arm == "acc" else mg32

    def run():
        launch.launch()
        target.add_(tmp)

    return run


def _acc_arm(arm):
    def build(row, t):
        op = row["op"]
        if op not in ACC_OPS:
            raise NotImplementedError("the accumulation arms run the accumulation rows")
        note = _accumulate_note(arm, row, t) if arm in FUSED_ARMS else ""
        if op in ("acc", "bwd_acc"):
            what = "checkpointed 2 x fwd + bwd" if op == "acc" else "backward"
            return _acc_autograd(arm, t, op), None, f"{what} ({arm}) {note}".strip()
        return _acc_wgrad(arm, t), None, f"weight gradient ({arm}) {note}".strip()

    return build


ARM_BUILDERS = {
    "cake_det": cake_det_arm,
    "cake_nondet": cake_nondet_arm,
    "torch_grouped_mm": torch_grouped_mm_arm,
    "per_expert_cublas": per_expert_cublas_arm,
    "cudnn_grouped_mm": cudnn_grouped_mm_arm,
    "cake_nondet_inforward": cake_nondet_inforward_arm,
    **{arm: _acc_arm(arm) for arm in ACC_ARMS},
}
CAKE_BACKENDS = {
    "cake_det": det_cb,
    "cake_nondet": cb,
    "cake_nondet_inforward": cb,
    **dict.fromkeys(ACC_ARMS, cb),
    **dict.fromkeys(DET_ACC_ARMS, det_cb),
}


def _times_ms(fn, cupti):
    times = sorted(
        float(v) for v in bench_gpu_time(fn, cold_l2_cache=True, enable_cupti=cupti)
    )
    pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]  # noqa: E731
    return dict(median_ms=pick(0.5), p10_ms=pick(0.1), p90_ms=pick(0.9), n=len(times))


def _memory(arm, row, t, device):
    """(peak MiB, retained MiB) of building and running ``arm`` once on ``row`` beyond
    the inputs and output; a Cake arm's kept stream workspaces are dropped first, so its
    first launch allocates them again inside the measurement."""
    backend = CAKE_BACKENDS.get(arm)
    torch.cuda.synchronize(device)
    if backend is not None:
        backend._stream_workspaces.clear()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn, launch, _note = ARM_BUILDERS[arm](row, t)
    fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    retained = torch.cuda.memory_allocated(device) - base
    del fn, launch
    return peak / MIB, retained / MIB


def _error_vs_det(row, t, outs):
    """The ``cake_nondet`` output against ``cake_det``'s in the deterministic budget's unit
    (bf16: worst |d| / (1e-2 + 1e-2 |ref|); fp32: rel-L2 / 1e-4) and the count of
    differing elements."""
    if "cake_nondet" not in outs or "cake_det" not in outs:
        return None
    out, ref = outs["cake_nondet"].float(), outs["cake_det"].float()
    differing = int((outs["cake_nondet"] != outs["cake_det"]).sum())
    if row["op"] == "wgrad" and row["out_dtype"] == torch.float32:
        err = float((out - ref).norm() / ref.norm().clamp_min(1e-30)) / 1e-4
    else:
        err = float(((out - ref).abs() / (1e-2 + 1e-2 * ref.abs())).max())
    return dict(budget_fraction=err, differing_elements=differing)


def _autograd_outputs(fn):
    """The output and the leaf gradients of one call of an autograd row's callable."""
    y, leaves = fn()
    return [y.detach()] + [leaf.grad for leaf in leaves]


def _cake_nondet_vs_inforward(row, grads):
    """The count of elements in which the ``cake_nondet`` and ``cake_nondet_inforward``
    output and gradients differ (all must be equal: the 512-wide k tile's weight
    gradient sums its split tail in slot order, so it repeats bit for bit too; only a
    split tail planned through ``unordered_tail_archs``, empty in this release, could
    differ)."""
    if len(grads) < 2:
        return None
    names = ("out", "x.grad", "w1.grad", "w2.grad")
    if row["op"] not in BLOCK_OPS:
        names = ("y", "x.grad", "w.grad")
    pairs = zip(
        names, grads["cake_nondet"], grads["cake_nondet_inforward"], strict=True
    )
    return {name: int((a != b).sum()) for name, a, b in pairs}


def _kernel_census(fn, device):
    """The CUDA kernels (copies and memsets excluded) of one call of the warm ``fn``."""
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize(device)
    return sum(
        1
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memcpy", "Memset"))
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ops",
        default=",".join(DEFAULT_OPS),
        help=f"comma-separated subset of {','.join(OPS)}",
    )
    parser.add_argument(
        "--rows",
        default="",
        help="comma-separated substrings; a row runs when any matches its label",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARM_BUILDERS)}",
    )
    parser.add_argument(
        "--cupti", action="store_true", help="time with CUPTI instead of CUDA events"
    )
    parser.add_argument(
        "--census",
        action="store_true",
        help="count the CUDA kernels of one call per arm with torch.profiler after "
        "timing it (a separate run: the profiler may perturb later timings)",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="skip the peak-memory measurement"
    )
    parser.add_argument(
        "--allow-composed",
        action="store_true",
        help="record a fused arm's composed route on a device that serves its accumulate "
        "build instead of failing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="", help="write per-row results to this file")
    args = parser.parse_args()
    global REQUIRE_FUSED
    REQUIRE_FUSED = not args.allow_composed

    device = torch.device("cuda", 0)
    ops = {o.strip() for o in args.ops.split(",") if o.strip()}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    filters = [f.strip() for f in args.rows.split(",") if f.strip()]
    unknown = [a for a in arms if a not in ARM_BUILDERS]
    if unknown:
        parser.error(f"unknown arms {unknown}")
    if args.census and args.cupti:
        parser.error("--census profiles with torch.profiler; run it without --cupti")
    for arm, backend in CAKE_BACKENDS.items():
        if arm in arms and not backend.generated_program_available(device):
            print(
                f"# no generated {arm} program registered for this device "
                f"({backend.arch_for(device)}); the arm is skipped"
            )
            arms.remove(arm)

    results = []
    header = f"{'row':48s} {'sum_m':>7s} " + " ".join(f"{arm:>24s}" for arm in arms)
    print(header + f" {'cake/det':>8s}")
    for row in build_rows():
        if row["op"] not in ops or (
            filters and not any(f in row["label"] for f in filters)
        ):
            continue
        t = make_inputs(row, device, args.seed)
        flops = row_flops(row)
        record = dict(
            label=row["label"],
            op=row["op"],
            N=row["N"],
            K=row["K"],
            sizes=row["sizes"],
            out_dtype=str(row["out_dtype"]).replace("torch.", ""),
            arms={},
        )
        # Alternate the arm order between measured rows so that no arm always runs first.
        row_arms = arms if len(results) % 2 == 0 else arms[::-1]
        record["arm_order"] = list(row_arms)
        # An autograd row run by both cake_nondet and cake_nondet_inforward compares their results.
        check = row["op"] in AUTOGRAD_OPS and {
            "cake_nondet",
            "cake_nondet_inforward",
        } <= set(arms)
        cells, outs, grads = {}, {}, {}
        for arm in row_arms:
            try:
                fn, _launch, note = ARM_BUILDERS[arm](row, t)
                times = _times_ms(fn, args.cupti)
                ms = times["median_ms"]
                entry = dict(times, tflops=flops / ms / 1e9, note=note)
                if arm in CAKE_BACKENDS and t["out"] is not None:
                    fn()
                    torch.cuda.synchronize(device)
                    outs[arm] = t["out"].clone()
                if check and arm in ("cake_nondet", "cake_nondet_inforward"):
                    grads[arm] = _autograd_outputs(fn)
                    torch.cuda.synchronize(device)
                if args.census:
                    entry["kernels"] = _kernel_census(fn, device)
                del fn, _launch
                if not args.no_memory:
                    entry["peak_mib"], entry["retained_mib"] = _memory(
                        arm, row, t, device
                    )
                record["arms"][arm] = entry
                cells[arm] = f"{ms:9.4f}ms {flops / ms / 1e9:6.0f}T"
            except AccumulatePlanError:
                raise
            except Exception as exc:  # noqa: BLE001 - an unavailable arm is reported, not fatal
                record["arms"][arm] = dict(error=f"{type(exc).__name__}: {exc}"[:200])
                cells[arm] = f"{'n/a':>24s}"
        ratio = None
        det_ms = record["arms"].get("cake_det", {}).get("median_ms")
        nondet_ms = record["arms"].get("cake_nondet", {}).get("median_ms")
        if det_ms and nondet_ms:
            ratio = nondet_ms / det_ms
        record["cake_over_det"] = ratio
        if row["op"] in ACC_OPS:
            medians = {a: e.get("median_ms") for a, e in record["arms"].items()}
            record["acc_ratios"] = {
                f"{a}/{b}": medians[a] / medians[b]
                for a, b in ACC_RATIOS
                if medians.get(a) and medians.get(b)
            }
        record["cake_vs_det_output"] = _error_vs_det(row, t, outs)
        if check:
            record["cake_nondet_vs_inforward"] = _cake_nondet_vs_inforward(row, grads)
        results.append(record)
        print(
            f"{row['label']:48s} {sum(row['sizes']):7d} "
            + " ".join(f"{cells[arm]:>24s}" for arm in arms)
            + (f" {ratio:8.4f}" if ratio else f" {'n/a':>8s}")
        )
        if record.get("acc_ratios"):
            shown = " ".join(f"{k} {v:.4f}" for k, v in record["acc_ratios"].items())
            print(f"# {row['label']}: {shown}")
        differing = record.get("cake_nondet_vs_inforward") or {}
        if any(differing.values()):
            print(
                f"# {row['label']}: cake_nondet differs from cake_nondet_inforward {differing}"
            )
        del t, outs, grads
        torch.cuda.empty_cache()

    ratios = [r["cake_over_det"] for r in results if r["cake_over_det"]]
    if ratios:
        geo = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        slower = [r["label"] for r in results if (r["cake_over_det"] or 0) > 1.0]
        print(
            f"# cake_nondet / cake_det median-time ratio: geomean {geo:.4f} over {len(ratios)} rows; "
            f"{len(slower)} rows slower"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    arch=cb.arch_for(device),
                    sm_count=cb.sm_count(device),
                    cupti=args.cupti,
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    rows=results,
                ),
                f,
                indent=2,
            )
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
