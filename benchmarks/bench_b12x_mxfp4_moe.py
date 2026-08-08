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

Report-only SM120/SM121 W4A4 fused-MoE benchmark.

The default shape follows a DeepSeek-style expert:
H=7168, I=2048, E=256, top-k=8, with token counts from decode to prefill.
MXFP4 and NVFP4 use the same packed E2M1 weight bytes, BF16 inputs, routing
IDs, and routing weights. Scale tensors use their native UE8M0/block-32 and
E4M3/block-16 layouts. This avoids a large temporary BF16 weight allocation
while keeping the performance comparison data-identical.

This script measures one rank and does not simulate expert-parallel dispatch.
For EP comparisons, pass the number of experts resident on that rank and choose
--tokens/--top-k so that tokens * top_k matches the routed pairs
actually processed by that rank. Under balanced routing this is approximately
the model top-k divided by the EP size, but it can be fractional or imbalanced;
the default --top-k 8 is a single-rank workload and is not EP-aware.

Usage:
    python benchmarks/bench_b12x_mxfp4_moe.py
    python benchmarks/bench_b12x_mxfp4_moe.py --tokens 1 8 32
"""

import argparse

import numpy as np
import torch

from flashinfer import B12xMoEWrapper
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
    select_sm120_moe_backend,
)
from flashinfer.testing.utils import bench_gpu_time


def _make_mma_scale_factors(
    num_experts: int,
    rows: int,
    columns: int,
    quant_mode: str,
) -> torch.Tensor:
    """Create native MMA-layout scale factors with decoded scale value 1."""
    sf_vec_size = 32 if quant_mode == "mxfp4" else 16
    m_tiles = (rows + 127) // 128
    k_tiles = ((columns + sf_vec_size - 1) // sf_vec_size + 3) // 4
    if quant_mode == "mxfp4":
        storage = torch.full(
            (num_experts, m_tiles, k_tiles, 32, 4, 4),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
    else:
        storage = torch.full(
            (num_experts, m_tiles, k_tiles, 32, 4, 4),
            1.0,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
    return storage.permute(3, 4, 1, 5, 2, 0)


def _make_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    generator = torch.Generator(device="cuda").manual_seed(2027)
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    alpha = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    return w1, w2, alpha


def _tflops(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    latency_ms: float,
) -> float:
    # Gated FC1 has gate+up projections (2 GEMMs); FC2 adds one GEMM.
    operations = 6 * num_tokens * top_k * hidden_size * intermediate_size
    return operations * 1e-9 / latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 8, 32, 128, 512, 2048],
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument(
        "--quant-modes",
        nargs="+",
        choices=["nvfp4", "mxfp4"],
        default=["nvfp4", "mxfp4"],
    )
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    if (
        any(num_tokens <= 0 for num_tokens in args.tokens)
        or args.hidden_size <= 0
        or args.intermediate_size <= 0
        or args.num_experts <= 0
        or args.top_k <= 0
    ):
        parser.error("tokens, dimensions, num-experts, and top-k must be positive")

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((12, 0), (12, 1)):
        raise RuntimeError(f"b12x MXFP4 MoE requires SM120/SM121, got SM{major}{minor}")
    if args.hidden_size % 128 or args.intermediate_size % 128:
        raise ValueError("hidden and intermediate sizes must be multiples of 128")

    torch.manual_seed(args.seed)
    w1, w2, alpha = _make_weights(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
    )
    scales = {
        quant_mode: (
            _make_mma_scale_factors(
                args.num_experts,
                2 * args.intermediate_size,
                args.hidden_size,
                quant_mode,
            ),
            _make_mma_scale_factors(
                args.num_experts,
                args.hidden_size,
                args.intermediate_size,
                quant_mode,
            ),
        )
        for quant_mode in args.quant_modes
    }
    print(
        "quant_mode,tokens,backend,latency_ms,tflops",
        flush=True,
    )
    for num_tokens in args.tokens:
        x = (
            torch.randn(
                num_tokens,
                args.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 100
        )
        token_ids = (
            torch.arange(num_tokens, device="cuda", dtype=torch.int32)[:, None]
            * args.top_k
            + torch.arange(args.top_k, device="cuda", dtype=torch.int32)[None, :]
        ) % args.num_experts
        token_weights = torch.full(
            (num_tokens, args.top_k),
            1.0 / args.top_k,
            dtype=torch.float32,
            device="cuda",
        )
        for quant_mode in args.quant_modes:
            w1_sf, w2_sf = scales[quant_mode]
            moe = B12xMoEWrapper(
                num_experts=args.num_experts,
                top_k=args.top_k,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                use_cuda_graph=True,
                max_num_tokens=num_tokens,
                quant_mode=quant_mode,
            )

            def run():
                return moe.run(
                    x=x,
                    w1_weight=w1,
                    w1_weight_sf=w1_sf,
                    w1_alpha=alpha,
                    fc2_input_scale=(alpha[:1] if quant_mode == "nvfp4" else None),
                    w2_weight=w2,
                    w2_weight_sf=w2_sf,
                    w2_alpha=alpha,
                    token_selected_experts=token_ids,
                    token_final_scales=token_weights,
                )

            run()
            measurements = bench_gpu_time(
                run,
                dry_run_time_ms=args.warmup_ms,
                repeat_time_ms=args.repeat_ms,
                use_cuda_graph=True,
                cold_l2_cache=False,
            )
            latency_ms = float(np.median(measurements))
            backend = select_sm120_moe_backend(
                num_tokens=num_tokens,
                num_topk=args.top_k,
                quant_mode=quant_mode,
            )
            achieved_tflops = _tflops(
                num_tokens,
                args.top_k,
                args.hidden_size,
                args.intermediate_size,
                latency_ms,
            )
            print(
                f"{quant_mode},{num_tokens},{backend},"
                f"{latency_ms:.6f},{achieved_tflops:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
