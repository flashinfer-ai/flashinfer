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

"""Benchmark the experimental Kimi-K3 FP8_PB_WO projection GEMMs on SM100/SM103.

The 22 TP8 / TP1 projection families of ``nvidia/Kimi-K3-NVFP4`` (``q_proj``,
``fused_qkvg``, ``in_proj_qkvgfab``, ``f_a``, ``f_b``, ``b_proj``, ``kv_a``,
``fused_qkv_a``, ``q_b``, ``kv_b``, ``o_proj``) x ``M in {1, 8, 64, 256, 4096,
16384}`` are timed as CUDA-graph replays with CUPTI (``--cupti``, cupti-python
>= 13) or CUDA events and a cold L2 between iterations.  The same tensors are
also run through FlashInfer's existing chain ``per_token_group_quant_8bit`` +
``gemm_fp8_nt_groupwise`` (``cutlass`` sm1 / sm2, ``trtllm``, ``cutile``) on
the identical FP8_PB_WO weight; the fastest chain is the baseline of each row
and the per-row and geometric-mean speedups are printed.

Usage::

    python benchmarks/bench_cake_kimi_k3_fp8_projection.py [--cupti] [--rows tp8_q_proj_m1 ...]
"""

import argparse
import json
import math

import torch

from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
    AMAX_FLOOR,
    BLOCK,
    E4M3_MAX,
    PROJECTION_FAMILIES,
    n_padded_128,
    requant_weight_ue8m0,
)
from flashinfer.gemm import (
    allocate_kimi_k3_fp8_projection_workspace,
    gemm_fp8_nt_groupwise,
    prepare_kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection_weights,
)
from flashinfer.quantization import per_token_group_quant_8bit
from flashinfer.testing import bench_gpu_time

M_VALUES = (1, 8, 64, 256, 4096, 16384)
ROWS = {
    f"{tp}_{module}_m{m}": (tp, module, m)
    for tp, modules in PROJECTION_FAMILIES.items()
    for module in modules
    for m in M_VALUES
}
CHAINS = ("cutlass_sm1", "cutlass_sm2", "trtllm", "cutile")


def make_weight(n_valid, K, device, seed):
    """A serialized FP8_PB_WO weight: E4M3 ``[N_pad128, K]`` + FP32 128x128 block scales."""
    gen = torch.Generator(device=device).manual_seed(seed + 31 * n_valid + K)
    n128 = n_padded_128(n_valid)
    w = torch.randn((n128, K), device=device, generator=gen, dtype=torch.float32) * 0.02
    w[n_valid:] = 0.0
    wv = w.to(torch.bfloat16).view(n128 // BLOCK, BLOCK, K // BLOCK, BLOCK)
    amax = wv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(AMAX_FLOOR)
    sf = amax / E4M3_MAX
    w_q = (wv.float() * (1.0 / sf)).to(torch.float8_e4m3fn).view(n128, K)
    return w_q.contiguous(), sf.view(n128 // BLOCK, 1, K // BLOCK, 1).contiguous()


def _chain(name, x, weight, scale, n128, device):
    """One FlashInfer quantization + groupwise GEMM chain writing ``[M, N_pad128]`` (or None)."""
    sf = scale.reshape(n128 // BLOCK, -1).contiguous()
    out = torch.empty((x.shape[0], n128), dtype=torch.bfloat16, device=device)
    if name == "cutile":

        def run():
            q, s = per_token_group_quant_8bit(x, 128)
            gemm_fp8_nt_groupwise(
                q, weight, s, sf, scale_major_mode="K", out=out, backend="cutile"
            )

    elif name == "trtllm":

        def run():
            q, s = per_token_group_quant_8bit(x, 128, column_major_scales=True)
            gemm_fp8_nt_groupwise(
                q, weight, s, sf, scale_major_mode="MN", out=out, backend="trtllm"
            )

    else:
        mma_sm = 2 if name.endswith("sm2") else 1
        sf_mn = sf.t().contiguous()

        def run():
            q, s = per_token_group_quant_8bit(x, 128, column_major_scales=True)
            gemm_fp8_nt_groupwise(
                q, weight, s.t(), sf_mn, scale_major_mode="MN", mma_sm=mma_sm, out=out
            )

    try:
        run()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001 - an unavailable backend drops out of the baseline set
        print(
            f"  chain {name}: unavailable ({type(error).__name__}: {str(error)[:80]})"
        )
        return None
    return run


def _reference(x, weight, scale, n_valid):
    w2, s2 = requant_weight_ue8m0(weight, scale)
    n, k = w2.shape
    m = x.shape[0]
    xv = x.view(m, k // BLOCK, BLOCK)
    amax = xv.abs().float().amax(dim=2).clamp(AMAX_FLOOR)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX))
    sf = torch.pow(2.0, exp)
    a_q = (xv.float() / sf.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, k)
    a = (a_q.float().view(m, k // BLOCK, BLOCK) * sf.view(m, k // BLOCK, 1)).view(m, k)
    w = (
        w2.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * s2.view(n // BLOCK, 1, k // BLOCK, 1)
    ).view(n, k)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    for i in range(0, m, 4096):
        out[i : i + 4096] = a[i : i + 4096] @ w.T
    return out[:, :n_valid].to(torch.bfloat16)


def _median_us(fn, *, cupti):
    times = bench_gpu_time(
        fn, enable_cupti=cupti, cold_l2_cache=True, use_cuda_graph=True
    )
    times = sorted(times)
    return 1000.0 * float(times[len(times) // 2])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cupti", action="store_true", help="time with CUPTI instead of CUDA events"
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help=f"row names (default: all {len(ROWS)})"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="write the per-row results to this JSON file",
    )
    parser.add_argument("--seed", type=int, default=622)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    names = args.rows or list(ROWS)
    results = []
    for name in names:
        tp, module, M = ROWS[name]
        n_valid, K = PROJECTION_FAMILIES[tp][module]
        weight, scale = make_weight(n_valid, K, device, args.seed)
        gen = torch.Generator(device=device).manual_seed(args.seed + 7 * M + K)
        x = torch.randn((M, K), device=device, generator=gen, dtype=torch.float32).to(
            torch.bfloat16
        )
        out = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
        prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
        workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, M)
        runner = prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)
        runner()
        torch.cuda.synchronize()
        expected = _reference(x, weight, scale, n_valid)
        max_err = float((out.float() - expected.float()).abs().max())
        ours_us = _median_us(runner, cupti=args.cupti)
        chains = {}
        n128 = n_padded_128(n_valid)
        for chain in CHAINS:
            fn = _chain(chain, x, weight, scale, n128, device)
            if fn is not None:
                chains[chain] = _median_us(fn, cupti=args.cupti)
        best = (
            min(chains.items(), key=lambda item: item[1])
            if chains
            else (None, float("nan"))
        )
        speedup = best[1] / ours_us if chains else float("nan")
        route = runner.plan.route
        detail = ""
        if runner.plan.decode is not None:
            cfg = runner.plan.decode
            detail = f" t{cfg.tok} s{cfg.split}{' fused' if cfg.fused else ''}{' res' if cfg.resident else ''}"
        print(
            f"{name:28s} M={M:6d} N={n_valid:6d} K={K:6d} route={route}{detail:16s} ours={ours_us:9.2f} us"
            f"  best_chain={best[0]} {best[1]:9.2f} us  speedup={speedup:5.2f}x  max_err={max_err:.4g}"
        )
        results.append(
            dict(
                row=name,
                tp=tp,
                module=module,
                M=M,
                N=n_valid,
                K=K,
                route=route,
                kernels=list(runner.plan.kernels),
                ours_us=ours_us,
                chains_us=chains,
                best_chain=best[0],
                speedup=speedup,
                max_abs_err=max_err,
            )
        )
    measured = [r["speedup"] for r in results if math.isfinite(r["speedup"])]
    if measured:
        gm = math.exp(sum(math.log(v) for v in measured) / len(measured))
        wins = sum(v > 1.0 for v in measured)
        print(
            f"rows={len(measured)} faster_than_best_chain={wins} geomean_speedup={gm:.3f}x min={min(measured):.3f}x"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
