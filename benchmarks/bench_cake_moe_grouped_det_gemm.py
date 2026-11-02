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

"""Benchmark the experimental ragged BF16 MoE grouped GEMM on SM100/SM103/SM107.

The 58 MoE projection rows of the design: gate/up (N=4096, K=6144) and down
(N=6144, K=2048) projections, E=4 with balanced / skewed / one-empty group
distributions at two token counts, E=8/16/32 balanced, and the fp32 weight
gradient of the balanced E=4 rows.  Each row times the prepared Cake launch
against ``torch._grouped_mm``, a per-group cuBLAS loop with host-known bounds
and ``flashinfer.grouped_mm.grouped_mm_bf16`` (cuDNN; fwd / dgrad only), all
with a cold L2 between iterations: the median time (the JSON adds the 10th /
90th percentiles), the arm order alternating between measured rows.

Plus one checkpointed training step per (projection, distribution), the
``step`` rows: ``y = mm(x, w, offs)`` under ``torch.utils.checkpoint``
(non-reentrant) and its backward with both gradients, through
``cake_grouped_det_mm`` and through ``torch._grouped_mm``'s autograd, with
checkpoint early stop off so that every arm recomputes its forward GEMM
(``2 x forward + backward``, the same work in every arm).  Opt-in autograd rows
over the same distributions: ``step_es``, that step with early stop on (the
torch default: an arm whose inputs are saved before its GEMM launches,
``cake`` and ``torch._grouped_mm``, skips the recomputed forward); ``block`` /
``block_es``, one checkpointed MoE expert block with early stop off / on --
gate_up, ``silu(a) * b`` on its two halves, down, with the gradients of ``x``,
``w1`` and ``w2`` -- in which only the recompute of the down GEMM can be
skipped.  The rate of the ``step`` rows counts the four GEMMs every arm runs;
that of the other autograd rows the useful work only (forward, data and
weight gradient of each GEMM).  Opt-in arm ``cake_inforward``: ``cake``
through a direct ``CakeGroupedDetMm.apply(x, w, offs, True)``, whose forward
GEMM launches inside ``forward`` so that a checkpoint always recomputes it
(autograd rows only; when it runs with ``cake``, the JSON records the elements
in which their outputs and gradients differ -- none).  ``--census`` adds the
CUDA kernel count of one call per arm (``torch.profiler``).

Usage::

    python benchmarks/bench_cake_moe_grouped_det_gemm.py [--ops fwd,wgrad,step] [--rows e4_m112576] [--cupti] [--json out.json]
    python benchmarks/bench_cake_moe_grouped_det_gemm.py --ops step,step_es,block,block_es --arms cake,cake_inforward,torch_grouped_mm [--census]
"""

import argparse
import json
import math

import torch
import torch.profiler
import torch.utils.checkpoint

from flashinfer.experimental.cake_moe_grouped_det_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_det_gemm.cake_autograd import (
    CakeGroupedDetMm,
    cake_grouped_det_mm,
)
from flashinfer.experimental.cake_moe_grouped_det_gemm.cake_backend import (
    prepare_grouped_det_gemm_dgrad,
    prepare_grouped_det_gemm_fwd,
    prepare_grouped_det_gemm_wgrad,
)
from flashinfer.testing import bench_gpu_time

PROJECTIONS = {"gate_up": (4096, 6144), "down": (6144, 2048)}  # (N, K)
BLOCK = "gate_up_down"  # the expert block's projections: gate_up, silu(a) * b, down
OPS = ("fwd", "dgrad", "wgrad", "step", "step_es", "block", "block_es")
DEFAULT_OPS = OPS[:4]
AUTOGRAD_OPS = OPS[3:]
BLOCK_OPS = ("block", "block_es")
AUTOGRAD_NOTES = {
    "step": "checkpointed 2 x fwd + bwd, early stop off",
    "step_es": "checkpointed step, early stop on (the recompute skips the GEMM only "
    "where its inputs are saved before it launches)",
    "block": "checkpointed gate_up, silu(a) * b, down: 2 x fwd + bwd",
    "block_es": "checkpointed gate_up, silu(a) * b, down, early stop on (gate_up is "
    "always recomputed; down only where its inputs are saved before it launches)",
}
# E=4 group sizes per token count (32-row aligned, last group takes the remainder).
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
ARMS = ("cake", "torch_grouped_mm", "per_expert_cublas", "cudnn_grouped_mm")


def build_rows():
    rows = []

    def add(op, proj, sizes, out_dtype, group):
        if proj == BLOCK:
            # The block's output and input width: the hidden size.
            n, k = PROJECTIONS["down"][0], PROJECTIONS["gate_up"][1]
        else:
            n, k = PROJECTIONS[proj]
        dist = group if group in ("skewed", "one_empty") else "balanced"
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
    # The checkpointed step and block rows over the step's distributions.
    step_dists = [
        (sizes, dist) for dists in E4_GROUPS.values() for dist, sizes in dists.items()
    ] + [([EXPERT_COUNT_SUM_M // e] * e, "balanced") for e in EXPERT_COUNTS]
    for op in ("step", "step_es"):
        for proj in PROJECTIONS:
            for sizes, dist in step_dists:
                add(op, proj, sizes, torch.bfloat16, dist)
    for op in BLOCK_OPS:
        for sizes, dist in step_dists:
            add(op, BLOCK, sizes, torch.bfloat16, dist)
    return rows


def row_flops(row):
    """The row's FLOPs.  ``step`` counts the four GEMMs every arm runs with early stop
    off (two forward passes, the data and the weight gradient); the other autograd rows
    the forward, data and weight gradient of each GEMM (useful work, no recompute)."""
    sum_m = sum(row["sizes"])
    projs = ("gate_up", "down") if row["op"] in BLOCK_OPS else (row["proj"],)
    flops = sum(2.0 * sum_m * n * k for n, k in (PROJECTIONS[p] for p in projs))
    if row["op"] == "step":
        return flops * 4
    if row["op"] in AUTOGRAD_OPS:
        return flops * 3
    return flops


def make_inputs(row, device, seed):
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


def _step(mm, t, early_stop=False):
    """A non-reentrant checkpointed ``y = mm(x, w, offs)`` and its backward (which
    replays the forward), with both gradients.

    Early stop off (the ``step`` rows): every arm recomputes its forward GEMM,
    ``2 x forward + backward``.  On (the torch default, the ``step_es`` rows): the
    recompute stops once the region saved its last tensor, so an arm whose inputs are
    saved before its GEMM launches (``torch._grouped_mm``, ``cake_grouped_det_mm``) does
    one GEMM less, and one that launches inside ``Function.forward``
    (``cake_inforward``) still recomputes it."""
    return _checkpointed(mm, t, ("x", "w"), early_stop)


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


def cake_arm(row, t):
    op = row["op"]
    if op in AUTOGRAD_OPS:
        fn = _autograd(lambda x, w, offs: cake_grouped_det_mm(x, w, offs), row, t)
        return fn, f"{AUTOGRAD_NOTES[op]} (cake_grouped_det_mm)"
    if op == "fwd":
        launch = prepare_grouped_det_gemm_fwd(t["x"], t["w"], t["offs"], out=t["out"])
    elif op == "dgrad":
        launch = prepare_grouped_det_gemm_dgrad(t["g"], t["w"], t["offs"], out=t["out"])
    else:
        launch = prepare_grouped_det_gemm_wgrad(t["g"], t["x"], t["offs"], out=t["out"])
    return launch.launch, f"{launch.record_name} plan={launch.plan}"


def cake_inforward_arm(row, t):
    """``cake`` through a direct ``CakeGroupedDetMm.apply``: the forward GEMM launches
    inside ``forward``, before autograd saves the inputs (the internal control of the
    autograd rows)."""
    op = row["op"]
    if op not in AUTOGRAD_OPS:
        raise NotImplementedError("the in-forward control has autograd rows only")
    fn = _autograd(lambda x, w, offs: CakeGroupedDetMm.apply(x, w, offs, True), row, t)
    return fn, f"{AUTOGRAD_NOTES[op]} (direct CakeGroupedDetMm.apply)"


def torch_grouped_mm_arm(row, t):
    op, fn = row["op"], torch._grouped_mm
    if op in AUTOGRAD_OPS:

        def mm(a, b, o):
            return fn(a, b.transpose(1, 2), offs=o)

        return _autograd(mm, row, t), f"{AUTOGRAD_NOTES[op]} (torch autograd)"
    x, w, g, offs = t["x"], t["w"], t["g"], t["offs"]
    if op == "fwd":
        return (lambda: fn(x, w.transpose(1, 2), offs=offs)), "x @ w.transpose(1, 2)"
    if op == "dgrad":
        return (lambda: fn(g, w, offs=offs)), "g @ w"
    kwargs = {"out_dtype": torch.float32} if row["out_dtype"] == torch.float32 else {}
    try:
        probe = fn(g.t(), x, offs=offs, **kwargs)
        torch.cuda.synchronize()
        if tuple(probe.shape) != tuple(t["out"].shape):
            raise RuntimeError(f"output shape {tuple(probe.shape)}")
        del probe
        return (
            lambda: fn(g.t(), x, offs=offs, **kwargs)
        ), "g.t() @ x (transposed view accepted)"
    except Exception as exc:  # noqa: BLE001 - the copy inside the timed callable is recorded
        note = f"g.t().contiguous() inside the timed callable ({type(exc).__name__})"
        return (lambda: fn(g.t().contiguous(), x, offs=offs, **kwargs)), note


def per_expert_cublas_arm(row, t):
    op = row["op"]
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
    return run, f"{launches} GEMM launches, host-known bounds"


def cudnn_grouped_mm_arm(row, t):
    from flashinfer.grouped_mm import grouped_mm_bf16

    op = row["op"]
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
    return run, "cuDNN heuristic plan (tactic=-1), W read in place"


ARM_BUILDERS = {
    "cake": cake_arm,
    "torch_grouped_mm": torch_grouped_mm_arm,
    "per_expert_cublas": per_expert_cublas_arm,
    "cudnn_grouped_mm": cudnn_grouped_mm_arm,
    "cake_inforward": cake_inforward_arm,
}


def _times_ms(fn, cupti):
    times = sorted(
        float(v) for v in bench_gpu_time(fn, cold_l2_cache=True, enable_cupti=cupti)
    )
    pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]  # noqa: E731
    return dict(median_ms=pick(0.5), p10_ms=pick(0.1), p90_ms=pick(0.9), n=len(times))


def _autograd_outputs(fn):
    """The output and the leaf gradients of one call of an autograd row's callable."""
    y, leaves = fn()
    return [y.detach()] + [leaf.grad for leaf in leaves]


def _cake_vs_inforward(row, grads):
    """The count of elements in which the ``cake`` and ``cake_inforward`` output and
    gradients differ (none: the same deterministic programs)."""
    if len(grads) < 2:
        return None
    names = ("out", "x.grad", "w1.grad", "w2.grad")
    if row["op"] not in BLOCK_OPS:
        names = ("y", "x.grad", "w.grad")
    pairs = zip(names, grads["cake"], grads["cake_inforward"], strict=True)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="", help="write per-row results to this file")
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    ops = {o.strip() for o in args.ops.split(",") if o.strip()}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    filters = [f.strip() for f in args.rows.split(",") if f.strip()]
    unknown = [a for a in arms if a not in ARM_BUILDERS]
    if unknown:
        parser.error(f"unknown arms {unknown}")
    if args.census and args.cupti:
        parser.error("--census profiles with torch.profiler; run it without --cupti")
    if {"cake", "cake_inforward"} & set(arms) and not cb.generated_program_available(
        device
    ):
        arch = cb.arch_for(device)
        print(
            f"# no generated program registered for this device ({arch}); the cake arms are skipped"
        )

    results = []
    header = f"{'row':48s} {'sum_m':>7s} " + " ".join(f"{arm:>18s}" for arm in arms)
    print(header)
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
            arms={},
        )
        # Alternate the arm order between measured rows so that no arm always runs first.
        row_arms = arms if len(results) % 2 == 0 else arms[::-1]
        record["arm_order"] = list(row_arms)
        # An autograd row run by both cake and cake_inforward compares their results.
        check = row["op"] in AUTOGRAD_OPS and {"cake", "cake_inforward"} <= set(arms)
        cells, grads = {}, {}
        for arm in row_arms:
            try:
                fn, note = ARM_BUILDERS[arm](row, t)
                times = _times_ms(fn, args.cupti)
                ms = times["median_ms"]
                entry = dict(**times, tflops=flops / ms / 1e9, note=note)
                if check and arm in ("cake", "cake_inforward"):
                    grads[arm] = _autograd_outputs(fn)
                    torch.cuda.synchronize(device)
                if args.census:
                    entry["kernels"] = _kernel_census(fn, device)
                record["arms"][arm] = entry
                cells[arm] = f"{ms:9.4f}ms {flops / ms / 1e9:6.0f}T"
                del fn
            except Exception as exc:  # noqa: BLE001 - an unavailable arm is reported, not fatal
                record["arms"][arm] = dict(error=f"{type(exc).__name__}: {exc}"[:200])
                cells[arm] = "n/a"
        if check:
            record["cake_vs_inforward"] = _cake_vs_inforward(row, grads)
        results.append(record)
        print(
            f"{row['label']:48s} {sum(row['sizes']):7d} "
            + " ".join(f"{cells[arm]:>18s}" for arm in arms)
        )
        differing = record.get("cake_vs_inforward") or {}
        if any(differing.values()):
            print(f"# {row['label']}: cake differs from cake_inforward {differing}")
        del t, grads
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    device=torch.cuda.get_device_name(device),
                    cupti=args.cupti,
                    rows=results,
                ),
                f,
                indent=2,
            )
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
