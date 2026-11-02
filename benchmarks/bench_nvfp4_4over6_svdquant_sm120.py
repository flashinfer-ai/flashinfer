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

SVDQuant linear with dynamic NVFP4 4over6 activations on SM12x (nvfp4_4over6_svdquant_linear_sm120) against
the same computation composed from public APIs: torch amax -> global scale and alpha -> fp4_quantize with the
4over6 recipe -> torch LoRA-down divided by alpha -> mm_nvfp4_svdquant. Also reported: the GEMM alone, the
quantize + LoRA-down prefix alone, and the prefix forced onto the unfused path. Each op is captured CALLS
times in a CUDA graph; the time per call is the median over REPS replays.

    python benchmarks/bench_nvfp4_4over6_svdquant_sm120.py [--m ...] [--nk 6144x4096 ...] [--backend cutlass-sm120]
"""

import argparse
import statistics

import torch

import flashinfer
from flashinfer import (
    NVFP44Over6Config,
    fp4_quantize,
    mm_nvfp4_svdquant,
    nvfp4_4over6_dynamic_quantize_lora_down_sm120,
    nvfp4_4over6_svdquant_linear_sm120,
)
from flashinfer.quantization.nvfp4_4over6_sm120 import (
    _default_rcp_quant_range,
    get_nvfp4_4over6_quantize_sm120_module,
)

CONFIG = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")


def time_us(fn, calls: int, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(times)


def bench_shape(m: int, n: int, k: int, rank: int, args) -> str:
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device).bfloat16()
    w = (torch.randn(n, k, device=device) * k**-0.5).bfloat16()
    w_gs = (448.0 * 6.0 / w.abs().amax().float()).reshape(1)
    w_fp4, w_sf = fp4_quantize(w, w_gs, 16, False, True, nvfp4_4over6=None)
    w_fp4 = w_fp4.view(torch.uint8).reshape(n, k // 2)
    w_sf = w_sf.view(torch.uint8).reshape(-1)
    wgs = 1.0 / w_gs
    lora_down = (torch.randn(rank, k, device=device) * k**-0.5).bfloat16()
    lora_up = (torch.randn(n, rank, device=device) * 0.05).bfloat16()
    rcp_q = _default_rcp_quant_range(CONFIG)
    backend = args.backend

    def composed():
        deq = x.abs().amax().float().clamp(min=1e-12) * rcp_q
        gs = (1.0 / deq).reshape(1)
        alpha = (deq * wgs).reshape(1)
        x_q, sf = fp4_quantize(x, gs, 16, False, True, nvfp4_4over6=CONFIG)
        down = ((x @ lora_down.t()).float() / alpha).bfloat16()
        return mm_nvfp4_svdquant(
            x_q.view(torch.uint8),
            w_fp4,
            sf.view(torch.uint8).reshape(-1),
            w_sf,
            alpha,
            down,
            lora_up,
            backend=backend,
        )

    x_q, sf, down, _, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
        x, lora_down, wgs, CONFIG
    )
    _, _, _, op = get_nvfp4_4over6_quantize_sm120_module()
    gs_buf, alpha_buf = torch.empty(1, device=device), torch.empty(1, device=device)
    ops = {
        "composed": composed,
        "linear": lambda: nvfp4_4over6_svdquant_linear_sm120(
            x,
            w_fp4,
            w_sf,
            wgs,
            lora_down,
            lora_up,
            nvfp4_4over6=CONFIG,
            backend=backend,
        ),
        "gemm": lambda: mm_nvfp4_svdquant(
            x_q, w_fp4, sf, w_sf, alpha, down, lora_up, backend=backend
        ),
        "prefix": lambda: op(x, lora_down, gs_buf, alpha_buf, wgs, rcp_q, 2, -1),
        "prefix_unfused": lambda: op(x, lora_down, gs_buf, alpha_buf, wgs, rcp_q, 2, 0),
    }
    if (
        args.autotune
    ):  # one tuned GEMM tactic per shape, shared by every arm (same mm problem)
        with flashinfer.autotune(True):
            ops["gemm"]()
        torch.cuda.synchronize()
    t = {}
    for name, fn in ops.items():
        try:
            t[name] = time_us(fn, args.calls, args.reps)
        except Exception as error:
            print(
                f"# {name} failed at M={m} N={n} K={k}: {type(error).__name__}: {error}"
            )
            t[name] = float("nan")
    path = int(
        get_nvfp4_4over6_quantize_sm120_module()[
            0
        ].nvfp4_4over6_dynamic_quantize_lora_down_sm120(
            x,
            gs_buf,
            alpha_buf,
            wgs,
            rcp_q,
            torch.zeros(8192, device=device),
            lora_down,
            torch.empty(64 * 64 * 64, device=device),
            torch.empty_like(x_q),
            torch.empty_like(sf),
            torch.empty_like(down),
            -1,
            0,
        )
    )
    return (
        f"{m:>6} {n:>6} {k:>6} | {t['composed']:>9.2f} {t['linear']:>8.2f} {t['composed'] / t['linear']:>6.2f}x | "
        f"{t['gemm']:>8.2f} {t['prefix']:>8.2f} {t['prefix_unfused']:>8.2f}  path {path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 4096]
    )
    parser.add_argument(
        "--nk",
        nargs="+",
        default=["6144x4096", "4096x4096", "24576x4096", "4096x12288"],
        help="N x K of the dense layers (default: Qwen3-8B QKV, O, gate/up, down)",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--backend", default="cutlass-sm120")
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune the mm_nvfp4_svdquant tactic per shape first",
    )
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    print(
        f"# {torch.cuda.get_device_name()}, recipe {CONFIG!r}, rank {args.rank}, mm backend {args.backend}"
        f"{' (autotuned)' if args.autotune else ''}, us per call"
    )
    print(
        f"{'M':>6} {'N':>6} {'K':>6} | {'composed':>9} {'linear':>8} {'speedup':>7} | "
        f"{'gemm':>8} {'prefix':>8} {'unfused':>8}  path"
    )
    for nk in args.nk:
        n, k = (int(v) for v in nk.split("x"))
        for m in args.m:
            print(bench_shape(m, n, k, args.rank, args), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
