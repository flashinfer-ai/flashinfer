"""Measure pristine B12x W4A16/NVFP4 MoE CUDA-Graph latency on SM120."""

from __future__ import annotations

import argparse
import gc

import torch

from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper


PRESETS = {
    "qwen": {"hidden": 2048, "intermediate": 512},
    "joyai": {"hidden": 2048, "intermediate": 768},
}


def bench(fn, warmup: int, iterations: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    latency = start.elapsed_time(end) * 1000.0 / iterations
    del graph
    gc.collect()
    return latency


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *prefix, cols = weight.shape
    rows = prefix[-1]
    experts = prefix[0]
    packed_flat, swizzled = fp4_quantize(
        weight.reshape(-1, cols),
        torch.ones(1, dtype=torch.float32, device=weight.device),
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    packed = packed_flat.reshape(*prefix, cols // 2).contiguous()
    scales = convert_sf_to_mma_layout(
        swizzled,
        m=rows,
        k=cols,
        num_groups=experts,
        sf_vec_size=16,
    )
    return packed, scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--mode", choices=("w4a16", "nvfp4"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    if not 1 <= args.m <= 8:
        parser.error("m must be in [1, 8]")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    experts, topk = 64, 8
    hidden = PRESETS[args.preset]["hidden"]
    intermediate = PRESETS[args.preset]["intermediate"]
    x = (
        torch.randn(args.m, hidden, dtype=torch.bfloat16, device=device) * 0.1
    ).contiguous()
    w1 = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).contiguous()
    w2 = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).contiguous()
    w1, w1_sf = quantize_weight(w1)
    w2, w2_sf = quantize_weight(w2)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(args.m)]
    ).to(torch.int32)
    route_weights = torch.softmax(
        torch.randn(args.m, topk, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    alphas = torch.ones(experts, dtype=torch.float32, device=device)
    fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)
    wrapper = B12xMoEWrapper(
        num_experts=experts,
        top_k=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        quant_mode=args.mode,
        use_cuda_graph=True,
        max_num_tokens=args.m,
    )

    def run():
        return wrapper.run(
            x=x,
            w1_weight=w1,
            w1_weight_sf=w1_sf,
            w1_alpha=alphas,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=w2_sf,
            w2_alpha=alphas,
            token_selected_experts=ids,
            token_final_scales=route_weights,
        )

    latency = bench(run, args.warmup, args.iterations)
    print(
        f"preset={args.preset} mode={args.mode} m={args.m} "
        f"unique_experts={int(torch.unique(ids).numel())} graph_us={latency:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
