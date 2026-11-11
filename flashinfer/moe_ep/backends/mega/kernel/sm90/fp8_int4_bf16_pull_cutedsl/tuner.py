# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Offline INT4 tuning with the shared Hopper activation/routing workload."""

from ...tuning import finish_sweep, run_tuning as _run_tuning
from ..fp8_mxfp4_bf16_pull_cutedsl.tuner import _create_canonical_inputs


def create_tuning_weights(*, local_experts, hidden, intermediate, rank, device, seed=0):
    import torch

    from ......kernel_src.sm90.pull_style_cutedsl_megakernel import Int4WeightPack

    generator = torch.Generator(device=device).manual_seed(0x494E5434 + seed + rank)

    def leg(rows, k):
        payload = torch.randint(
            256,
            (local_experts, rows, k // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scales = (
            torch.rand(
                local_experts, rows, k // 128, device=device, generator=generator
            )
            * 0.006
            + 0.002
        ).bfloat16()
        return payload, scales

    w13, s13 = leg(2 * intermediate, hidden)
    w2, s2 = leg(hidden, intermediate)
    return Int4WeightPack(w13, s13, w2, s2)


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg

    pkg.require_hopper_mxfp4_fused_tuning_device()
    candidates = pkg.hopper_int4_candidates(
        max_tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.num_experts,
        world_size=world_size,
        routing_profile=args.routing_profile,
    )
    y, l1, l2, workspace = _create_canonical_inputs(
        args,
        rank,
        world_size,
        max_tokens,
        max_tokens,
        candidates[0],
        create_weights=create_tuning_weights,
        preprocess_weights=pkg.preprocess_int4_mega_weights,
        allocate_workspace=pkg.get_symm_buffer_for_hopper_int4_mega_moe,
    )
    try:
        return finish_sweep(
            args,
            rank,
            max_tokens,
            max_tokens,
            workspace,
            y,
            l1,
            l2,
            candidates,
            pkg.autotune_hopper_int4_mega_moe,
            tune_kwargs={"gate_up_clamp": args.gate_up_clamp},
        )
    finally:
        workspace.destroy()


def run_tuning(args) -> int:
    from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg

    return _run_tuning(args, tune_one, pkg=pkg)
