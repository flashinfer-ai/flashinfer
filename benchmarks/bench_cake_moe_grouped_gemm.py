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
with a cold L2 between iterations.

Usage::

    python benchmarks/bench_cake_moe_grouped_gemm.py [--ops fwd,wgrad] [--rows e4_m112576] [--cupti] [--json out.json]
"""

import argparse
import json
import math

import torch

from flashinfer.experimental.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
    prepare_grouped_gemm_dgrad,
    prepare_grouped_gemm_fwd,
    prepare_grouped_gemm_wgrad,
)
from flashinfer.testing import bench_gpu_time

PROJECTIONS = {"gate_up": (4096, 6144), "down": (6144, 2048)}  # (N, K)
OPS = ("fwd", "dgrad", "wgrad")
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
        for op in OPS:
            for dists in E4_GROUPS.values():
                for dist, sizes in dists.items():
                    add(op, proj, sizes, torch.bfloat16, dist)
    for proj in PROJECTIONS:
        for dists in E4_GROUPS.values():
            add("wgrad", proj, dists["balanced"], torch.float32, "balanced")
    for e in EXPERT_COUNTS:
        for proj in PROJECTIONS:
            for op in OPS:
                add(op, proj, [EXPERT_COUNT_SUM_M // e] * e, torch.bfloat16, "balanced")
    assert len(rows) == 58
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
    else:
        out = torch.empty(e, n, k, dtype=row["out_dtype"], device=device)
    return dict(x=x, w=w, g=g, offs=offs, out=out)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return [
        (e, s, t) for e, (s, t) in enumerate(zip([0] + ends[:-1], ends, strict=True))
    ]


def cake_arm(row, t):
    op = row["op"]
    if op == "fwd":
        launch = prepare_grouped_gemm_fwd(t["x"], t["w"], t["offs"], out=t["out"])
    elif op == "dgrad":
        launch = prepare_grouped_gemm_dgrad(t["g"], t["w"], t["offs"], out=t["out"])
    else:
        launch = prepare_grouped_gemm_wgrad(t["g"], t["x"], t["offs"], out=t["out"])
    return launch.launch, f"{launch.record_name} plan={launch.plan}"


def torch_grouped_mm_arm(row, t):
    op, fn = row["op"], torch._grouped_mm
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
    if op == "wgrad":
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
}


def _median_ms(fn, cupti):
    times = bench_gpu_time(fn, cold_l2_cache=True, enable_cupti=cupti)
    times = sorted(float(v) for v in times)
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ops", default=",".join(OPS), help="comma-separated subset of fwd,dgrad,wgrad"
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
    if "cake" in arms and not cb.generated_program_available(device):
        arch = cb.arch_for(device)
        print(
            f"# no generated program registered for this device ({arch}); the cake arm is skipped"
        )

    results = []
    header = f"{'row':46s} {'sum_m':>7s} " + " ".join(f"{arm:>18s}" for arm in arms)
    print(header)
    for row in build_rows():
        if row["op"] not in ops or (
            filters and not any(f in row["label"] for f in filters)
        ):
            continue
        t = make_inputs(row, device, args.seed)
        flops = 2.0 * sum(row["sizes"]) * row["N"] * row["K"]
        record = dict(
            label=row["label"],
            op=row["op"],
            N=row["N"],
            K=row["K"],
            sizes=row["sizes"],
            arms={},
        )
        cells = []
        for arm in arms:
            try:
                fn, note = ARM_BUILDERS[arm](row, t)
                ms = _median_ms(fn, args.cupti)
                record["arms"][arm] = dict(
                    median_ms=ms, tflops=flops / ms / 1e9, note=note
                )
                cells.append(f"{ms:9.4f}ms {flops / ms / 1e9:6.0f}T")
            except Exception as exc:  # noqa: BLE001 - an unavailable arm is reported, not fatal
                record["arms"][arm] = dict(error=f"{type(exc).__name__}: {exc}"[:200])
                cells.append(f"{'n/a':>18s}")
        results.append(record)
        print(
            f"{row['label']:46s} {sum(row['sizes']):7d} "
            + " ".join(f"{c:>18s}" for c in cells)
        )
        del t
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
