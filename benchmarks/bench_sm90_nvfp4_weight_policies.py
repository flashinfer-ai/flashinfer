"""Single-GPU packed/folded/hot-folded NVFP4 policy sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch


def _csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("expected a nonempty integer CSV")
    return result


def _checkpoints(args, device):
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        quantize_bf16_to_nvfp4_checkpoint,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    w13 = torch.randn(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2 = torch.randn(
        args.experts,
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    result = (
        quantize_bf16_to_nvfp4_checkpoint(w13),
        quantize_bf16_to_nvfp4_checkpoint(w2),
    )
    del w13, w2
    return result


def _inputs(args, tokens: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(args.seed + tokens)
    x = torch.randn(
        tokens,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    popularity = (
        torch.arange(1, args.experts + 1, device=device).float().pow(-args.routing_skew)
    )
    ids = torch.multinomial(
        popularity.expand(tokens, -1),
        args.top_k,
        replacement=False,
        generator=generator,
    ).to(torch.int32)
    weights = torch.softmax(
        torch.randn(tokens, args.top_k, device=device, generator=generator), dim=-1
    )
    return x, ids, weights


def _layer(args, tokens, transformed, policy, hot_experts):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm90PushNvFp4MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=args.experts,
            max_tokens_per_rank=tokens,
            token_hidden_size=args.hidden,
        ),
        weights=None,
        backend=MegaConfig(
            megakernel=Sm90PushNvFp4MegaMoeConfig(
                intermediate_size=args.intermediate,
                top_k=args.top_k,
                weight_policy=policy,
                hot_expert_count=hot_experts if policy == "hot_folded" else 0,
                acknowledge_dual_residency=policy == "dual",
                payload_dtype="bf16",
                combine_dtype="bf16",
                grouped_combine=False,
            ),
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=transformed,
        ),
    )


def _forward(layer, inputs):
    from flashinfer.moe_ep import MoEEpTensors

    return layer(
        MoEEpTensors(
            hidden_states=inputs[0],
            topk_ids=inputs[1],
            topk_weights=inputs[2],
        )
    )


def main() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        estimate_residency,
        make_dual_weights_from_checkpoints,
        make_hot_folded_weights_from_checkpoints,
        make_transformed_weights_from_checkpoints,
    )
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=_csv, default=(64, 512, 2048))
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hot-experts", type=_csv, default=(0, 1, 2, 4, 8))
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--routing-skew", type=float, default=1.2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.experts <= 0 or args.top_k not in (1, 2, 4, 6, 8):
        raise ValueError("experts must be positive and top-k must be supported")
    if args.top_k > args.experts or any(
        not 0 <= hot <= args.experts for hot in args.hot_experts
    ):
        raise ValueError("top-k and hot expert counts exceed the expert count")
    if any(tokens <= 0 for tokens in args.tokens):
        raise ValueError("tokens must be positive")
    if args.hidden % 128 or args.intermediate % 128 or args.intermediate > 16384:
        raise ValueError("H/I must satisfy the SM90 W4A8 geometry")
    if not math.isfinite(args.routing_skew) or args.routing_skew < 0:
        raise ValueError("routing-skew must be finite and non-negative")
    if args.warmup <= 0 or args.iters <= 0:
        raise ValueError("iteration counts are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("benchmark requires SM90")
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError("folded-FP8 policy benchmarking requires CUDA 12.8+")
    checkpoints = _checkpoints(args, device)
    props = torch.cuda.get_device_properties(device)
    residency_rows = []
    for policy, hot_experts in (
        ("packed", 0),
        ("folded", args.experts),
        ("dual", args.experts),
        *(("hot_folded", hot) for hot in args.hot_experts if 0 < hot < args.experts),
    ):
        estimate = estimate_residency(
            args.experts,
            policy,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            hot_expert_count=hot_experts,
        )
        residency_rows.append(
            {
                "weight_policy": policy,
                "hot_experts": estimate.hot_experts,
                "packed_bytes": estimate.packed_bytes,
                "folded_bytes": estimate.folded_bytes,
                "resident_bytes": estimate.total_bytes,
            }
        )
    print(
        json.dumps(
            {
                "schema": "flashinfer.sm90-nvfp4.residency-matrix.v1",
                "experts": args.experts,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "rows": residency_rows,
            },
            sort_keys=True,
        )
    )
    policy_cases = [("packed", 0), ("folded", args.experts), ("dual", args.experts)]
    policy_cases.extend(
        ("hot_folded", hot) for hot in args.hot_experts if 0 < hot < args.experts
    )
    for policy, hot_experts in policy_cases:
        torch.cuda.synchronize(device)
        begin = time.perf_counter_ns()
        if policy == "packed":
            transformed = make_transformed_weights_from_checkpoints(
                *checkpoints,
                nvfp4_mode="w4a8",
                group_size=128,
                residual_scheme="generic",
            )
        elif policy == "dual":
            transformed = make_dual_weights_from_checkpoints(
                *checkpoints,
            )
        else:
            transformed = make_hot_folded_weights_from_checkpoints(
                *checkpoints,
                hot_experts=hot_experts,
            )
        torch.cuda.synchronize(device)
        conversion_ms = (time.perf_counter_ns() - begin) / 1e6
        estimate = estimate_residency(
            args.experts,
            policy,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            hot_expert_count=hot_experts,
        )
        for tokens in args.tokens:
            inputs = _inputs(args, tokens, device)
            layer = _layer(args, tokens, transformed, policy, hot_experts)
            for _ in range(args.warmup):
                _forward(layer, inputs)
            selector = dict(layer._workspace.runner._w4a8_engine.selection_provenance)
            samples = []
            for _ in range(args.iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward(layer, inputs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1e3)
            layer.destroy()
            actual_resident = getattr(
                transformed,
                "resident_bytes",
                estimate.total_bytes,
            )
            print(
                json.dumps(
                    {
                        "schema": "flashinfer.sm90-nvfp4.weight-policy.v1",
                        "weight_policy": policy,
                        "tokens_per_rank": tokens,
                        "grouped_gemm_rows": tokens * args.top_k,
                        "selector": selector,
                        "experts": args.experts,
                        "hot_experts": hot_experts,
                        "hot_route_fraction": float(
                            (inputs[1] < hot_experts).float().mean().item()
                        ),
                        "routing_skew": args.routing_skew,
                        "forward_us": statistics.median(samples),
                        "conversion_ms": conversion_ms,
                        "resident_bytes": int(actual_resident),
                        "estimated_resident_bytes": estimate.total_bytes,
                        "packed_bytes": estimate.packed_bytes,
                        "folded_bytes": estimate.folded_bytes,
                        "hidden": args.hidden,
                        "intermediate": args.intermediate,
                        "top_k": args.top_k,
                        "gpu": props.name,
                        "sm_count": props.multi_processor_count,
                        "cuda": torch.version.cuda,
                        "torch": torch.__version__,
                        "seed": args.seed,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    },
                    sort_keys=True,
                )
            )
            del layer, inputs
        del transformed
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
