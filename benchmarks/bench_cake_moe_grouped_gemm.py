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

"""Benchmark the selectable-mode ragged BF16 MoE grouped GEMM on SM100/SM103/SM107.

The 58 MoE projection rows of the deterministic package's benchmark: gate/up
(N=4096, K=6144) and down (N=6144, K=2048) projections, E=4 with balanced /
skewed / one-empty group distributions at two token counts, E=8/16/32 balanced,
and the fp32 weight gradient of the balanced E=4 rows; plus one checkpointed
training step per (projection, distribution) -- ``2 x forward + backward``
through the autograd wrapper under ``torch.utils.checkpoint`` (non-reentrant,
early stop off so that every arm recomputes its forward), both gradients included.

Arms: ``cake_det`` (the deterministic package,
``flashinfer.experimental.cake_moe_grouped_det_gemm``), ``cake``
(``flashinfer.experimental.cake.cake_moe_grouped_gemm`` with
``deterministic=False``), ``torch._grouped_mm``, a per-group cuBLAS loop with
host-known bounds and ``flashinfer.grouped_mm.grouped_mm_bf16`` (cuDNN; fwd /
dgrad only).  Each row reports the median time with a cold L2 between
iterations (and the 10th / 90th percentiles), the peak device memory one
preparation and launch takes beyond its inputs and output (the Cake arms'
kept stream workspaces included: they are dropped before the measurement) and
the memory it retains, the Cake plans (stages, tail split, whether the split
tail is unordered), the error of the ``cake`` output against ``cake_det``, and
the ``cake`` / ``cake_det`` time ratio.  Consecutive rows run the arms in
alternating order (the JSON records it as ``arm_order``); an arm with no
program for the device is dropped.

Usage::

    python benchmarks/bench_cake_moe_grouped_gemm.py [--ops fwd,wgrad,step] [--rows e4_m112576] [--arms cake_det,cake] [--cupti] [--json out.json]
"""

import argparse
import json
import math

import torch
import torch.utils.checkpoint

from flashinfer.experimental.cake import cake_moe_grouped_gemm as pkg
from flashinfer.experimental.cake.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_det_gemm import cake_backend as det_cb
from flashinfer.experimental.cake_moe_grouped_det_gemm.cake_autograd import (
    cake_grouped_det_mm,
)
from flashinfer.testing import bench_gpu_time

PROJECTIONS = {"gate_up": (4096, 6144), "down": (6144, 2048)}  # (N, K)
OPS = ("fwd", "dgrad", "wgrad", "step")
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
ARMS = ("cake_det", "cake", "torch_grouped_mm", "per_expert_cublas", "cudnn_grouped_mm")
MIB = float(1 << 20)


def build_rows():
    rows = []

    def add(op, proj, sizes, out_dtype, group):
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
    # One checkpointed training step per projection and group distribution.
    for proj in PROJECTIONS:
        for dists in E4_GROUPS.values():
            for dist, sizes in dists.items():
                add("step", proj, sizes, torch.bfloat16, dist)
        for e in EXPERT_COUNTS:
            add("step", proj, [EXPERT_COUNT_SUM_M // e] * e, torch.bfloat16, "balanced")
    return rows


def make_inputs(row, device, seed):
    torch.manual_seed(seed)
    sizes, n, k = row["sizes"], row["N"], row["K"]
    sum_m, e = sum(sizes), len(sizes)
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


def _plan_note(launch):
    plan = launch.plan
    keys = (
        "tile_n",
        "tile_k",
        "num_tail",
        "tail_splits",
        "unordered_tail",
        "wave_mode",
    )
    shown = {key: plan[key] for key in keys if key in plan}
    return f"{launch.record_name} stages={'+'.join(launch.stages) or 'none'} {shown}"


def _step(autograd_fn, t):
    """``2 x forward + backward``: a non-reentrant checkpointed forward and its backward
    (which recomputes the forward), with both gradients.

    Checkpoint early stop is off, so every arm recomputes its forward GEMM.  With it
    on, the recompute of a built-in op such as ``torch._grouped_mm`` stops before its
    kernel (its saved tensors are its inputs) and that arm would do one GEMM less."""
    x = t["x"].detach().requires_grad_(True)
    w = t["w"].detach().requires_grad_(True)
    g, offs = t["g"], t["offs"]

    def run():
        x.grad = None
        w.grad = None
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            y = torch.utils.checkpoint.checkpoint(
                autograd_fn, x, w, offs, use_reentrant=False
            )
            y.backward(g)

    return run


def _cake_arm(backend, row, t):
    op = row["op"]
    deterministic = backend is det_cb
    if op == "step":
        if deterministic:
            fn = _step(lambda x, w, offs: cake_grouped_det_mm(x, w, offs), t)
        else:
            fn = _step(
                lambda x, w, offs: pkg.cake_grouped_mm(x, w, offs, deterministic=False),
                t,
            )
        return fn, None, "checkpointed 2 x fwd + bwd (autograd)"
    args = (t["x"], t["w"]) if op == "fwd" else (t["g"], t["w"])
    if op == "wgrad":
        args = (t["g"], t["x"])
    if deterministic:
        prepare = {
            "fwd": det_cb.prepare_grouped_det_gemm_fwd,
            "dgrad": det_cb.prepare_grouped_det_gemm_dgrad,
            "wgrad": det_cb.prepare_grouped_det_gemm_wgrad,
        }[op]
        launch = prepare(*args, t["offs"], out=t["out"])
    else:
        prepare = {
            "fwd": cb.prepare_grouped_gemm_fwd,
            "dgrad": cb.prepare_grouped_gemm_dgrad,
            "wgrad": cb.prepare_grouped_gemm_wgrad,
        }[op]
        launch = prepare(*args, t["offs"], out=t["out"], deterministic=False)
    return launch.launch, launch, _plan_note(launch)


def cake_det_arm(row, t):
    return _cake_arm(det_cb, row, t)


def cake_arm(row, t):
    return _cake_arm(cb, row, t)


def torch_grouped_mm_arm(row, t):
    op, fn = row["op"], torch._grouped_mm
    x, w, g, offs = t["x"], t["w"], t["g"], t["offs"]
    if op == "fwd":
        return (
            (lambda: fn(x, w.transpose(1, 2), offs=offs)),
            None,
            "x @ w.transpose(1, 2)",
        )
    if op == "dgrad":
        return (lambda: fn(g, w, offs=offs)), None, "g @ w"
    if op == "step":

        def mm(a, b, o):
            return fn(a, b.transpose(1, 2), offs=o)

        return _step(mm, t), None, "checkpointed 2 x fwd + bwd (torch autograd)"
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
    if op == "step":
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
    if op in ("wgrad", "step"):
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


ARM_BUILDERS = {
    "cake_det": cake_det_arm,
    "cake": cake_arm,
    "torch_grouped_mm": torch_grouped_mm_arm,
    "per_expert_cublas": per_expert_cublas_arm,
    "cudnn_grouped_mm": cudnn_grouped_mm_arm,
}
CAKE_BACKENDS = {"cake_det": det_cb, "cake": cb}


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
    """The ``cake`` output against ``cake_det``'s in the deterministic budget's unit
    (bf16: worst |d| / (1e-2 + 1e-2 |ref|); fp32: rel-L2 / 1e-4) and the count of
    differing elements."""
    if "cake" not in outs or "cake_det" not in outs:
        return None
    out, ref = outs["cake"].float(), outs["cake_det"].float()
    differing = int((outs["cake"] != outs["cake_det"]).sum())
    if row["op"] == "wgrad" and row["out_dtype"] == torch.float32:
        err = float((out - ref).norm() / ref.norm().clamp_min(1e-30)) / 1e-4
    else:
        err = float(((out - ref).abs() / (1e-2 + 1e-2 * ref.abs())).max())
    return dict(budget_fraction=err, differing_elements=differing)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ops",
        default=",".join(OPS),
        help="comma-separated subset of fwd,dgrad,wgrad,step",
    )
    parser.add_argument(
        "--rows",
        default="",
        help="comma-separated substrings; a row runs when any matches its label",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}",
    )
    parser.add_argument(
        "--cupti", action="store_true", help="time with CUPTI instead of CUDA events"
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="skip the peak-memory measurement"
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
        flops = 2.0 * sum(row["sizes"]) * row["N"] * row["K"]
        if row["op"] == "step":
            flops *= 4  # two forward passes, the data and the weight gradient
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
        cells, outs = {}, {}
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
                del fn, _launch
                if not args.no_memory:
                    entry["peak_mib"], entry["retained_mib"] = _memory(
                        arm, row, t, device
                    )
                record["arms"][arm] = entry
                cells[arm] = f"{ms:9.4f}ms {flops / ms / 1e9:6.0f}T"
            except Exception as exc:  # noqa: BLE001 - an unavailable arm is reported, not fatal
                record["arms"][arm] = dict(error=f"{type(exc).__name__}: {exc}"[:200])
                cells[arm] = f"{'n/a':>24s}"
        ratio = None
        det_ms = record["arms"].get("cake_det", {}).get("median_ms")
        cake_ms = record["arms"].get("cake", {}).get("median_ms")
        if det_ms and cake_ms:
            ratio = cake_ms / det_ms
        record["cake_over_det"] = ratio
        record["cake_vs_det_output"] = _error_vs_det(row, t, outs)
        results.append(record)
        print(
            f"{row['label']:48s} {sum(row['sizes']):7d} "
            + " ".join(f"{cells[arm]:>24s}" for arm in arms)
            + (f" {ratio:8.4f}" if ratio else f" {'n/a':>8s}")
        )
        del t, outs
        torch.cuda.empty_cache()

    ratios = [r["cake_over_det"] for r in results if r["cake_over_det"]]
    if ratios:
        geo = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        slower = [r["label"] for r in results if (r["cake_over_det"] or 0) > 1.0]
        print(
            f"# cake / cake_det median-time ratio: geomean {geo:.4f} over {len(ratios)} rows; "
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
