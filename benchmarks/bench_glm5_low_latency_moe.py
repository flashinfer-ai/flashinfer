# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare the GLM5 low-latency MoE with the CUTLASS BF16 path.

Example::

    torchrun --nproc_per_node=8 benchmarks/bench_glm5_low_latency_moe.py \
      --dump-dir ~/dev/debug_output --warmup 20 --iterations 100

Use ``--baseline-only`` or ``--glm5-only`` to run the two paths in separate
compatible FlashInfer installations while retaining identical data loading,
correctness checks, timing, and rank aggregation.

The comparison uses the same TP8 checkpoint weights and times the same logical
operation: routing, expert-up, SwiGLU, expert-down, and the local routed/shared
reduction. The router GEMM and TP all-reduce are outside both timed regions. A
kernel-only CUTLASS number is also reported to show the cost of its externally
computed routing. The checkpoint's block-FP8 weights are dequantized to BF16
once before timing because CUTLASS block-FP8 MoE is not implemented on SM100.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

if "--baseline-only" in sys.argv:
    from flashinfer.fused_moe import cutlass_fused_moe
else:
    from flashinfer.fused_moe import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        Glm5LowLatencyConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        RoutingInputMode,
        RoutingMethodType,
        cutlass_fused_moe,
    )


_NUM_EXPERTS = 256
_TOP_K = 8
_ROUTED_SCALING_FACTOR = 2.5


def _one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one tensor matching {pattern!r} under {path}, got {matches}"
        )
    return matches[0]


def _load(path: Path, rank: int, layer: int, name: str, device) -> torch.Tensor:
    return torch.load(path / f"r{rank}_l{layer}_{name}.pt", map_location="cpu").to(
        device
    )


def _profile(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def _prepare_cutlass_weights(
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    routed_up_gate_weight: torch.Tensor,
    routed_up_gate_scale: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Dequantize block-FP8 weights and append the shared expert."""

    def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        expanded = scale.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
        return (weight.float() * expanded).to(torch.bfloat16)

    routed_up_gate_weight = dequantize(routed_up_gate_weight, routed_up_gate_scale)
    routed_down_weight = dequantize(routed_down_weight, routed_down_scale)
    shared_gate_up_weight = dequantize(shared_gate_up_weight, shared_gate_up_scale)
    shared_down_weight = dequantize(shared_down_weight, shared_down_scale)
    shared_gate, shared_up = shared_gate_up_weight.chunk(2, dim=0)
    return {
        "gemm1_weights": torch.cat(
            (routed_up_gate_weight, torch.cat((shared_up, shared_gate)).unsqueeze(0))
        ).contiguous(),
        "gemm2_weights": torch.cat(
            (routed_down_weight, shared_down_weight.unsqueeze(0))
        ).contiguous(),
    }


def _deepseek_routing(
    router_logits: torch.Tensor, routing_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = router_logits.sigmoid()
    selected = torch.topk(scores + routing_bias, _TOP_K, dim=-1).indices
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * _ROUTED_SCALING_FACTOR
    shared = torch.full_like(selected[:, :1], _NUM_EXPERTS)
    return (
        torch.cat((selected, shared), dim=-1).to(torch.int),
        torch.cat((weights, torch.ones_like(weights[:, :1])), dim=-1).float(),
    )


def _summarize(times_us: list[float]) -> tuple[float, float, float]:
    return statistics.mean(times_us), statistics.median(times_us), min(times_us)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="run only the CUTLASS BF16 baseline",
    )
    mode.add_argument(
        "--glm5-only",
        action="store_true",
        help="run only the GLM5 low-latency path",
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=1e-3,
        help="maximum allowed error against the saved PyTorch reference",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.max_abs_error <= 0:
        parser.error("--max-abs-error must be positive")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    dump_dir = args.dump_dir.expanduser()

    router_path = _one(dump_dir, f"r{rank}_l*_router_weight.pt")
    hidden_path = _one(dump_dir, f"r{rank}_l*_hidden_states.pt")
    weight_layer = int(router_path.name.split("_", 2)[1][1:])
    activation_layer = int(hidden_path.name.split("_", 2)[1][1:])
    hidden_states = _load(dump_dir, rank, activation_layer, "hidden_states", device)[
        : args.tokens
    ].contiguous()
    router_weight = _load(dump_dir, rank, weight_layer, "router_weight", device)
    routing_bias = _load(dump_dir, rank, weight_layer, "routing_bias", device)
    router_logits = torch.matmul(
        hidden_states.float(), router_weight.float().transpose(0, 1)
    ).contiguous()

    raw_weights = {
        "shared_gate_up_weight": _load(
            dump_dir, rank, weight_layer, "shared_gate_up_weight_org", device
        ),
        "shared_gate_up_scale": _load(
            dump_dir,
            rank,
            weight_layer,
            "shared_gate_up_weight_scale_org",
            device,
        ),
        "routed_up_gate_weight": _load(
            dump_dir, rank, weight_layer, "routed_w3_w1_weight", device
        ),
        "routed_up_gate_scale": _load(
            dump_dir,
            rank,
            weight_layer,
            "routed_w3_w1_weight_scaling_factor",
            device,
        ),
        "routed_down_weight": _load(
            dump_dir, rank, weight_layer, "routed_w2_weight", device
        ),
        "routed_down_scale": _load(
            dump_dir,
            rank,
            weight_layer,
            "routed_w2_weight_scaling_factor",
            device,
        ),
        "shared_down_weight": _load(
            dump_dir, rank, weight_layer, "shared_down_weight_org", device
        ),
        "shared_down_scale": _load(
            dump_dir,
            rank,
            weight_layer,
            "shared_down_weight_scale_org",
            device,
        ),
    }
    if not args.baseline_only:
        glm5_view = Glm5LowLatencyConfig.prepare_weights(**raw_weights)
        glm5_weight_pack = MoEWeightPack()
        glm5_weight_pack.prepare_for("glm5_low_latency", glm5_view)
        glm5_act_pack = MoEActivationPack(
            hidden_states_q=hidden_states,
            hidden_states_scale=None,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_logits=router_logits,
            routing_bias=routing_bias,
        )
        glm5_config = MoEConfig(
            routing=RoutingConfig(
                num_experts=_NUM_EXPERTS,
                top_k=_TOP_K,
                method=RoutingMethodType.MiniMax2,
                routed_scaling_factor=_ROUTED_SCALING_FACTOR,
            ),
            quant=QuantConfig(variant=QuantVariant.Glm5LowLatencyFp8),
            experts=ExpertConfig(
                intermediate_size=raw_weights["shared_down_weight"].shape[1],
                num_fused_shared_experts=1,
            ),
            backend=BackendOptions(candidates=(Glm5LowLatencyConfig(),)),
            execution=ExecutionConfig(tune_max_num_tokens=4),
        )
        glm5_layer = MoELayer(glm5_config, device=device)
    if not args.glm5_only:
        cutlass_weights = _prepare_cutlass_weights(**raw_weights)
        selected_experts, routing_weights = _deepseek_routing(
            router_logits, routing_bias
        )
    expected = _load(dump_dir, rank, weight_layer, "pytorch_ref_output", device)[
        : args.tokens
    ]

    if not args.glm5_only:
        cutlass_output = torch.empty_like(hidden_states)

    def run_glm5() -> torch.Tensor:
        return glm5_layer(glm5_act_pack, glm5_weight_pack)

    def run_cutlass_kernel(
        routed_experts: torch.Tensor | None = None,
        routed_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if routed_experts is None:
            routed_experts = selected_experts
        if routed_weights is None:
            routed_weights = routing_weights
        result = cutlass_fused_moe(
            hidden_states,
            routed_experts,
            routed_weights,
            cutlass_weights["gemm1_weights"],
            cutlass_weights["gemm2_weights"],
            torch.bfloat16,
            quant_scales=None,
            output=cutlass_output,
        )
        return result[0] if isinstance(result, list) else result

    def run_cutlass_pipeline() -> torch.Tensor:
        routed_experts, routed_weights = _deepseek_routing(router_logits, routing_bias)
        return run_cutlass_kernel(routed_experts, routed_weights)

    with torch.inference_mode():
        actual_outputs = {}
        if not args.glm5_only:
            actual_outputs["cutlass_bf16"] = run_cutlass_kernel().clone()
        if not args.baseline_only:
            actual_outputs["glm5_low_latency"] = run_glm5().clone()
        torch.cuda.synchronize(device)
        errors = {
            name: (actual.float() - expected.float()).abs().max().item()
            for name, actual in actual_outputs.items()
        }
        if len(actual_outputs) == 2:
            errors["paths"] = (
                (
                    actual_outputs["glm5_low_latency"].float()
                    - actual_outputs["cutlass_bf16"].float()
                )
                .abs()
                .max()
                .item()
            )
        print(
            f"rank={rank} correctness max_abs_error "
            + " ".join(f"{name}={error:.6e}" for name, error in errors.items()),
            flush=True,
        )
        for name in actual_outputs:
            if not errors[name] <= args.max_abs_error:
                raise AssertionError(
                    f"{name} max_abs_error={errors[name]:.6e} exceeds "
                    f"--max-abs-error={args.max_abs_error:.6e}"
                )

        timings = {}
        if not args.glm5_only:
            timings["cutlass_bf16_kernel"] = _summarize(
                _profile(run_cutlass_kernel, args.warmup, args.iterations)
            )
            timings["cutlass_bf16_pipeline"] = _summarize(
                _profile(run_cutlass_pipeline, args.warmup, args.iterations)
            )
        if not args.baseline_only:
            timings["glm5_low_latency"] = _summarize(
                _profile(run_glm5, args.warmup, args.iterations)
            )
    for name, (mean_us, median_us, min_us) in timings.items():
        print(
            f"rank={rank} tokens={args.tokens} path={name} mean_us={mean_us:.3f} "
            f"median_us={median_us:.3f} min_us={min_us:.3f}",
            flush=True,
        )

    if world_size > 1:
        names = tuple(timings)
        local_stats = torch.tensor(
            [value for name in names for value in timings[name]],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.empty_like(local_stats) for _ in range(world_size)]
        dist.all_gather(gathered, local_stats)
        if rank == 0:
            all_stats = torch.stack(gathered).cpu().reshape(world_size, len(names), 3)
            means = all_stats[:, :, 0].mean(dim=0)
            for index, name in enumerate(names):
                print(
                    f"all_ranks={world_size} tokens={args.tokens} path={name} "
                    f"mean_us={means[index].item():.3f} "
                    f"mean_rank_median_us={all_stats[:, index, 1].mean().item():.3f} "
                    f"rank_mean_range_us=[{all_stats[:, index, 0].min().item():.3f}, "
                    f"{all_stats[:, index, 0].max().item():.3f}]",
                    flush=True,
                )
            if len(names) == 3:
                glm5_index = names.index("glm5_low_latency")
                for baseline in (
                    "cutlass_bf16_kernel",
                    "cutlass_bf16_pipeline",
                ):
                    baseline_index = names.index(baseline)
                    speedup = means[baseline_index].item() / means[glm5_index].item()
                    print(
                        f"all_ranks={world_size} speedup_vs={baseline} "
                        f"glm5_low_latency={speedup:.3f}x",
                        flush=True,
                    )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
