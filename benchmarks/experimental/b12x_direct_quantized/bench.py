"""Benchmark B12x Direct W4A16 against both B12x precision modes."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Callable

import torch

from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper
from flashinfer.fused_moe.b12x_direct_quantized import (
    prepare_b12x_direct_w4a16_scales,
    b12x_direct_nvfp4_fused_moe,
    b12x_direct_nvfp4_fused_moe_workspace,
    b12x_direct_w4a16_fused_moe,
    b12x_direct_w4a16_fused_moe_workspace,
)


PRESETS = {
    "qwen": {"hidden": 2048, "intermediate": 512},
    "joyai": {"hidden": 2048, "intermediate": 768},
}


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    """Capture one call and return mean CUDA Graph replay latency in us."""
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
    latency_us = start.elapsed_time(end) * 1000.0 / iterations
    del graph
    gc.collect()
    return latency_us


def _probe(fn: Callable[[], torch.Tensor]) -> None:
    """Run a candidate eagerly before creating a CUDA Graph for it."""
    fn()
    torch.cuda.synchronize()


def _quantize_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed FP4, B12x MMA scales, and Direct folded BF16 scales."""
    *prefix, cols = weight.shape
    rows = prefix[-1]
    experts = prefix[0]
    flat = weight.reshape(-1, cols)
    global_scale = torch.ones(1, dtype=torch.float32, device=weight.device)
    packed_flat, swizzled = fp4_quantize(
        flat,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    packed = packed_flat.reshape(*prefix, cols // 2).contiguous()
    mma_scales = convert_sf_to_mma_layout(
        swizzled,
        m=rows,
        k=cols,
        num_groups=experts,
        sf_vec_size=16,
    )
    alphas = torch.ones(experts, dtype=torch.float32, device=weight.device)
    direct_scales = prepare_b12x_direct_w4a16_scales(
        mma_scales,
        alphas,
        rows=rows,
        cols=cols,
    )
    return packed, mma_scales, direct_scales


def _make_b12x_wrapper(
    mode: str,
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    hidden: int,
    intermediate: int,
) -> B12xMoEWrapper:
    return B12xMoEWrapper(
        num_experts=num_experts,
        top_k=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        quant_mode=mode,
        use_cuda_graph=True,
        max_num_tokens=num_tokens,
    )


def _direct_latency(
    fn_factory: Callable[[int | None, int | None], Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    tune: bool,
) -> tuple[float, int | None, int | None]:
    candidates: list[tuple[int | None, int | None]] = [(None, None)]
    if tune:
        candidates = [
            (outputs, threads)
            for outputs in (1, 2, 4, 8)
            for threads in range(64, 1025, 64)
        ]
    best: tuple[float, int | None, int | None] = (float("inf"), None, None)
    for outputs, threads in candidates:
        try:
            candidate = fn_factory(outputs, threads)
            _probe(candidate)
            latency = _bench(
                candidate,
                min(warmup, 30) if tune else warmup,
                min(iterations, 200) if tune else iterations,
            )
        except RuntimeError:
            # High outputs-per-warp variants can exceed the register limit.
            continue
        if latency < best[0]:
            best = (latency, outputs, threads)
    if best[0] == float("inf"):
        raise RuntimeError("no valid Direct launch configuration")
    if tune:
        candidate = fn_factory(best[1], best[2])
        _probe(candidate)
        latency = _bench(candidate, warmup, iterations)
        best = (latency, best[1], best[2])
    return best


def _nvfp4_latency(
    fn_factory: Callable[[int | None, int | None], Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    tune: bool,
) -> tuple[float, int | None, int | None]:
    candidates: list[tuple[int | None, int | None]] = [(None, None)]
    if tune:
        candidates = [
            (outputs, threads)
            for outputs in (1, 2, 4, 8)
            for threads in range(64, 513, 64)
            if outputs * (threads // 32) >= 16
            and outputs * (threads // 32) % 16 == 0
            # Keep the product below the launch-resource limit.  A failed
            # launch during CUDA Graph capture poisons the capture state, so
            # such configurations cannot be safely treated as normal tuner
            # misses in this process.
            and outputs * threads <= 1024
        ]
    best: tuple[float, int | None, int | None] = (float("inf"), None, None)
    for outputs, threads in candidates:
        try:
            candidate = fn_factory(outputs, threads)
            _probe(candidate)
            latency = _bench(
                candidate,
                min(warmup, 30) if tune else warmup,
                min(iterations, 200) if tune else iterations,
            )
        except (RuntimeError, ValueError):
            continue
        if latency < best[0]:
            best = (latency, outputs, threads)
    if best[0] == float("inf"):
        raise RuntimeError("no valid Direct NVFP4 launch configuration")
    if tune:
        candidate = fn_factory(best[1], best[2])
        _probe(candidate)
        best = (
            _bench(candidate, warmup, iterations),
            best[1],
            best[2],
        )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="qwen")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--m", type=int)
    parser.add_argument("--tune-direct", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    if args.m is not None and not 1 <= args.m <= 8:
        parser.error("m must be in [1, 8]")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this benchmark requires SM120")

    torch.manual_seed(args.seed)
    hidden = PRESETS[args.preset]["hidden"]
    intermediate = PRESETS[args.preset]["intermediate"]
    experts, topk = args.num_experts, args.topk
    # A single-M invocation must consume RNG in the same order as the pristine
    # B12x baseline script; otherwise route uniqueness and weight tensors differ
    # even with the same seed, making the cross-worktree comparison unfair.
    max_tokens = int(args.m) if args.m is not None else 8
    device = torch.device("cuda")

    hidden_states = (
        torch.randn(max_tokens, hidden, dtype=torch.bfloat16, device=device) * 0.1
    ).contiguous()
    w1_bf16 = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).contiguous()
    w2_bf16 = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    ).contiguous()
    w1, w1_sf, w1_direct_sf = _quantize_weight(w1_bf16)
    w2, w2_sf, w2_direct_sf = _quantize_weight(w2_bf16)
    del w1_bf16, w2_bf16
    alphas = torch.ones(experts, dtype=torch.float32, device=device)
    activation_global_scale_value = 448.0
    b12x_fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(max_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(max_tokens, topk, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    direct_output = torch.empty_like(hidden_states)
    direct_workspace = b12x_direct_w4a16_fused_moe_workspace(
        max_tokens, topk, intermediate, device=device
    )

    rows: list[dict[str, object]] = []
    token_counts = range(1, max_tokens + 1) if args.m is None else (args.m,)
    for num_tokens in token_counts:
        x = hidden_states[:num_tokens]
        ids = topk_ids[:num_tokens]
        route_weights = topk_weights[:num_tokens]

        def direct_factory(outputs: int, threads: int):
            def run_direct() -> torch.Tensor:
                return b12x_direct_w4a16_fused_moe(
                    x,
                    ids,
                    route_weights,
                    w1,
                    w1_direct_sf,
                    w2,
                    w2_direct_sf,
                    output=direct_output[:num_tokens],
                    workspace=direct_workspace[: num_tokens * topk],
                    outputs_per_warp=outputs,
                    num_threads=threads,
                    skip_check=True,
                )

            return run_direct

        direct_us, outputs, threads = _direct_latency(
            direct_factory,
            warmup=args.warmup,
            iterations=args.iterations,
            tune=args.tune_direct,
        )
        row: dict[str, object] = {
            "preset": args.preset,
            "num_tokens": num_tokens,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_experts": experts,
            "topk": topk,
            "direct_w4a16_us": direct_us,
            "direct_outputs_per_warp": outputs,
            "direct_threads": threads,
        }

        nvfp4_workspace = b12x_direct_nvfp4_fused_moe_workspace(
            num_tokens, topk, hidden, intermediate, device=device
        )

        def nvfp4_factory(outputs: int, threads: int):
            def run_nvfp4() -> torch.Tensor:
                return b12x_direct_nvfp4_fused_moe(
                    x,
                    ids,
                    route_weights,
                    w1,
                    w1_direct_sf,
                    w2,
                    w2_direct_sf,
                    output=direct_output[:num_tokens],
                    workspace=nvfp4_workspace,
                    outputs_per_warp=outputs,
                    num_threads=threads,
                    hidden_global_encode_scale=activation_global_scale_value,
                    intermediate_global_encode_scale=activation_global_scale_value,
                    skip_check=True,
                )

            return run_nvfp4

        nvfp4_us, nvfp4_outputs, nvfp4_threads = _nvfp4_latency(
            nvfp4_factory,
            warmup=args.warmup,
            iterations=args.iterations,
            tune=args.tune_direct,
        )
        row["direct_nvfp4_us"] = nvfp4_us
        row["direct_nvfp4_outputs_per_warp"] = nvfp4_outputs
        row["direct_nvfp4_threads"] = nvfp4_threads

        mode_outputs: dict[str, torch.Tensor] = {}
        for mode in ("w4a16", "nvfp4"):
            try:
                wrapper = _make_b12x_wrapper(
                    mode,
                    num_tokens=num_tokens,
                    num_experts=experts,
                    topk=topk,
                    hidden=hidden,
                    intermediate=intermediate,
                )

                def run_b12x() -> torch.Tensor:
                    return wrapper.run(
                        x=x,
                        w1_weight=w1,
                        w1_weight_sf=w1_sf,
                        w1_alpha=alphas,
                        fc2_input_scale=b12x_fc2_input_scale,
                        w2_weight=w2,
                        w2_weight_sf=w2_sf,
                        w2_alpha=alphas,
                        token_selected_experts=ids,
                        token_final_scales=route_weights,
                    )

                latency = _bench(run_b12x, args.warmup, args.iterations)
                mode_outputs[mode] = run_b12x().clone()
                row[f"b12x_{mode}_us"] = latency
                row[f"direct_vs_{mode}_speedup"] = latency / direct_us
                if mode == "nvfp4":
                    row["direct_nvfp4_vs_b12x_speedup"] = latency / nvfp4_us
            except (RuntimeError, ValueError) as error:
                row[f"b12x_{mode}_us"] = ""
                row[f"direct_vs_{mode}_speedup"] = ""
                row[f"b12x_{mode}_error"] = str(error)

        direct_result = direct_factory(outputs, threads)().clone()
        if "w4a16" in mode_outputs:
            error = (direct_result.float() - mode_outputs["w4a16"].float()).abs()
            row["direct_vs_b12x_w4a16_max_abs"] = float(error.max())
            row["direct_vs_b12x_w4a16_mean_abs"] = float(error.mean())
        if "nvfp4" in mode_outputs:
            nvfp4_result = nvfp4_factory(nvfp4_outputs, nvfp4_threads)().clone()
            error = (nvfp4_result.float() - mode_outputs["nvfp4"].float()).abs()
            row["direct_vs_b12x_nvfp4_max_abs"] = float(error.max())
            row["direct_vs_b12x_nvfp4_mean_abs"] = float(error.mean())
        rows.append(row)
        print(row, flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
