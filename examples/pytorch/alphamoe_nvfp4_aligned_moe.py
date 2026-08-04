"""Run the pre-aligned AlphaMoE NVFP4 kernel on SM100 or SM103."""

import argparse

import torch

from flashinfer.fused_moe import alphamoe_nvfp4_aligned_moe


def _random_nvfp4(
    leading_shape: tuple[int, ...],
    columns: int,
    *,
    generator: torch.Generator,
    min_scale_exp: float,
    max_scale_exp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create packed E2M1 values and linear per-16 E4M3 scales."""

    packed = torch.randint(
        0,
        256,
        (*leading_shape, columns // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scale_exp = torch.rand(
        (*leading_shape, columns // 16),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    scales = torch.exp2(min_scale_exp + scale_exp * (max_scale_exp - min_scale_exp)).to(
        torch.float8_e4m3fn
    )
    return packed, scales.contiguous()


def _aligned_plan(
    topk_ids: torch.Tensor,
    *,
    block_m: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert token-local expert ids to the aligned-plan input ABI."""

    flat_ids = topk_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
    sentinel = int(flat_ids.numel())
    sorted_positions: list[int] = []
    block_experts: list[int] = []
    for expert in range(num_experts):
        positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().tolist()
        if not positions:
            continue
        padded = ((len(positions) + block_m - 1) // block_m) * block_m
        sorted_positions.extend(positions)
        sorted_positions.extend([sentinel] * (padded - len(positions)))
        block_experts.extend([expert] * (padded // block_m))
    return (
        torch.tensor(sorted_positions, dtype=torch.int32, device="cuda"),
        torch.tensor(block_experts, dtype=torch.int32, device="cuda"),
        torch.tensor([len(sorted_positions)], dtype=torch.int32, device="cuda"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--block-m", type=int, default=8)
    parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=28101)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this example requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(
            f"AlphaMoE NVFP4 requires exact SM100/SM103, got {capability}"
        )
    if args.hidden_size < 256 or args.hidden_size % 256:
        raise ValueError("hidden_size must be at least 256 and divisible by 256")
    if args.intermediate_size < 128 or args.intermediate_size % 128:
        raise ValueError("intermediate_size must be at least 128 and divisible by 128")
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if args.block_m < 8 or args.block_m % 8:
        raise ValueError("block_m must be at least 8 and divisible by 8")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    gemm1_out_size = 2 * args.intermediate_size
    hidden_states, hidden_states_scale = _random_nvfp4(
        (args.num_tokens,),
        args.hidden_size,
        generator=generator,
        min_scale_exp=-3.0,
        max_scale_exp=-2.0,
    )
    gemm1_weights, gemm1_weights_scale = _random_nvfp4(
        (args.num_experts, gemm1_out_size),
        args.hidden_size,
        generator=generator,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    gemm2_weights, gemm2_weights_scale = _random_nvfp4(
        (args.num_experts, args.hidden_size),
        args.intermediate_size,
        generator=generator,
        min_scale_exp=-5.0,
        max_scale_exp=-4.0,
    )
    logits = torch.randn(
        args.num_tokens,
        args.num_experts,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_ids = torch.topk(logits, args.top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            args.num_tokens,
            args.top_k,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ),
        dim=-1,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _aligned_plan(
        topk_ids,
        block_m=args.block_m,
        num_experts=args.num_experts,
    )

    kernel_args = (
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
    )
    out = torch.zeros(
        args.num_tokens, args.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    result = alphamoe_nvfp4_aligned_moe(
        *kernel_args,
        out,
        args.top_k,
        args.block_m,
        args.routed_scaling_factor,
    )
    torch.cuda.synchronize()
    assert result is None
    assert bool((out != 0).any()), "the zero-initialized accumulator was not updated"
    fresh_norm = out.float().norm().item()

    seeded = torch.ones_like(out)
    result = alphamoe_nvfp4_aligned_moe(
        *kernel_args,
        seeded,
        args.top_k,
        args.block_m,
        args.routed_scaling_factor,
    )
    torch.cuda.synchronize()
    assert result is None
    assert bool((seeded != 1).any()), "the seeded accumulator was not updated"
    print(
        f"AlphaMoE NVFP4 succeeded on sm_{capability[0]}{capability[1]}: "
        f"fresh ||out||={fresh_norm:.6f}, seeded ||out||={seeded.float().norm().item():.6f}"
    )


if __name__ == "__main__":
    main()
