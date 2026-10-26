# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline CLI adapter for Blackwell GenPhase's collective tuner."""

from __future__ import annotations

import dataclasses
import json
import os

from ...tuning import finish_sweep


def tune_one(args, rank, world_size, max_tokens):
    """Stage synthetic inputs and tune one capacity on every EP rank.

    ``args.live_tokens`` selects the workload bucket without shrinking the
    workspace. Schedule-only sweeps preserve the selected tile and cluster.
    """
    import torch

    from ......kernel_src.sm100 import next_cutedsl_megamoe as pkg
    from ......weights import MoEWeightPack
    from ......tensors import MoEEpTensors
    from .staging import stage_genphase_inputs
    from .weights import preprocess_mega_weights

    live = args.live_tokens if args.live_tokens is not None else max_tokens
    if not 0 <= live <= max_tokens:
        raise ValueError("--live-tokens must be between zero and --max-tokens")
    cfg = pkg.Sm100GenphaseMoeConfig(
        num_total_experts=args.num_experts,
        max_tokens_per_rank=max_tokens,
        num_topk=args.topk,
        hidden=args.hidden,
        intermediate=args.intermediate,
        rank=rank,
        world_size=world_size,
        gate_up_clamp=args.gate_up_clamp,
    )
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    constraints = None
    if args.sweep == "schedule":
        base = (
            json.loads(args.base_knobs)
            if args.base_knobs
            else pkg.resolve_genphase_knobs(cfg, sm_count=sms, live_tokens=live)[0]
        )
        constraints = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in base.items()
            if k != "schedule_policy"
        }
    candidates = pkg.sm100_genphase_candidates(
        cfg, sm_count=sms, live_tokens=live, constraints=constraints
    )
    workspace = pkg.Sm100GenphaseSymmBuffer(dataclasses.replace(cfg, **candidates[0]))
    try:
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 7 * rank)
        legs = ([], [])
        local_experts = args.num_experts // world_size
        for begin in range(0, local_experts, 8):
            count = min(8, local_experts - begin)
            w13 = (
                torch.randn(
                    count,
                    2 * args.intermediate,
                    args.hidden,
                    device="cuda",
                    generator=generator,
                )
                * args.hidden**-0.5
            ).to(torch.bfloat16)
            w2 = (
                torch.randn(
                    count,
                    args.hidden,
                    args.intermediate,
                    device="cuda",
                    generator=generator,
                )
                * args.intermediate**-0.5
            ).to(torch.bfloat16)
            transformed = preprocess_mega_weights(
                MoEWeightPack(w13=w13, w2=w2),
                intermediate_size=args.intermediate,
                hidden_size=args.hidden,
                kind=args.dtype,
                kernel_variant="genphase",
            )
            for parts, (weight, scale) in zip(legs, transformed, strict=True):
                parts.append((weight, scale.reshape(count, -1)))
            del w13, w2
        l1, l2 = (pkg.concatenate_genphase_weights(parts) for parts in legs)
        del legs, transformed, parts, weight, scale
        x = torch.randn(live, args.hidden, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        ids = (
            torch.rand(live, args.num_experts, device="cuda", generator=generator)
            .topk(args.topk, dim=-1)
            .indices.to(torch.int32)
        )
        weights = torch.rand(live, args.topk, device="cuda", generator=generator) + 0.5
        tensors = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
        stage_genphase_inputs(tensors, workspace, quantize_input=True)
        return finish_sweep(
            args,
            rank,
            max_tokens,
            live,
            workspace,
            None,
            l1,
            l2,
            candidates,
            pkg.autotune_sm100_genphase_mega_moe,
        )
    except BaseException:
        # A failed GPU rank cannot safely enter collective free with its peers.
        workspace = None
        raise
    finally:
        if workspace is not None:
            workspace.destroy()


def run_tuning(args):
    """Bootstrap an EP job, tune requested GenPhase capacities, and return zero.

    Requires MXFP8 E4M3, BF16 combine and deterministic reduction. A failed GPU
    worker relies on job teardown rather than entering collective cleanup.
    """
    import torch

    from ......config import BootstrapConfig
    from ......core.runtime import (
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
        mxfp8_cutedsl_runtime_requirements,
    )
    from ......kernel_src.sm100.next_cutedsl_megamoe import knob_cache_path

    if (
        args.dtype != "mxfp8_e4m3"
        or args.combine_dtype != "bf16"
        or args.allow_nondeterministic
    ):
        raise ValueError(
            "Blackwell GenPhase requires MXFP8 E4M3, BF16 combine and deterministic reduction"
        )
    rank, world_size = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if os.environ.get("MEGA_NO_DIST") == "1" and world_size != 1:
        raise ValueError("MEGA_NO_DIST=1 requires a single-rank job")
    runtime = None
    if world_size == 1:
        os.environ.setdefault("MEGA_NO_DIST", "1")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    else:
        bootstrap = BootstrapConfig(rank=rank, world_size=world_size)
        ensure_moe_ep_cuda_device(bootstrap)
        runtime = bootstrap_moe_ep_runtime(
            bootstrap, mxfp8_cutedsl_runtime_requirements(bootstrap)
        )
    try:
        for capacity in args.max_tokens:
            tune_one(args, rank, world_size, capacity)
        torch.cuda.synchronize()
    except BaseException:
        runtime = None
        raise
    finally:
        if runtime is not None:
            finalize_moe_ep_runtime(runtime)
    if rank == 0:
        print(
            f"[moe_ep-tune] done; cache: {knob_cache_path() or 'DISABLED'}", flush=True
        )
    return 0
