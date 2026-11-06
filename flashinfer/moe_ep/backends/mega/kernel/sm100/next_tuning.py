# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tuning for the standard and GenPhase Blackwell kernels."""

from __future__ import annotations

import dataclasses
import json
import os

from ..tuning import finish_sweep


def tune_one(args, rank, world_size, max_tokens):
    """Stage synthetic inputs and tune one capacity on every EP rank.

    ``args.live_tokens`` selects the workload bucket without shrinking the
    workspace. Schedule-only sweeps preserve the selected tile and cluster.
    """
    import torch

    from .....kernel_src.sm100 import next_cutedsl_megamoe as pkg
    from .....tensors import MoEEpTensors
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.kernel_helpers import (
        concatenate_block_scaled_weights,
    )
    from .mxfp8_mxfp8_bf16_cutedsl.staging import stage_genphase_inputs
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
        sm100_standard_candidates,
        sm100_schedule_candidates,
    )

    live = args.live_tokens if args.live_tokens is not None else max_tokens
    if not 0 <= live <= max_tokens:
        raise ValueError("--live-tokens must be between zero and --max-tokens")
    genphase = args.kernel_variant == "genphase"
    config_type = (
        pkg.Sm100GenphaseMoeConfig if genphase else pkg.Sm100BlockScaledMoeConfig
    )
    options = (
        dict(
            quant_kind=args.dtype,
            apply_routing_weights_before_fc2=args.apply_routing_weights_before_fc2,
        )
        if genphase
        else dict(
            quant_kind=args.dtype,
            combine_dtype=args.combine_dtype,
            activation=args.activation,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            apply_routing_weights_before_fc2=args.apply_routing_weights_before_fc2,
            use_custom_finalize=not args.do_finalize,
        )
    )
    cfg = config_type(
        num_total_experts=args.num_experts,
        max_tokens_per_rank=max_tokens,
        num_topk=args.topk,
        hidden=args.hidden,
        intermediate=args.intermediate,
        rank=rank,
        world_size=world_size,
        gate_up_clamp=args.gate_up_clamp,
        **options,
    )
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    from .....kernel_src.sm100.next_cutedsl_megamoe.shim.knob_cache import resolve_knobs

    constraints = None
    if args.sweep == "schedule":
        base = (
            json.loads(args.base_knobs)
            if args.base_knobs
            else resolve_knobs(cfg, sm_count=sms, live_tokens=live)[0]
        )
        base = {k: tuple(v) if isinstance(v, list) else v for k, v in base.items()}
        constraints = (
            {k: v for k, v in base.items() if k != "schedule_policy"}
            if genphase
            else None
        )
    candidates = (
        pkg.sm100_genphase_candidates(
            cfg, sm_count=sms, live_tokens=live, constraints=constraints
        )
        if genphase
        else (
            sm100_schedule_candidates(base, cfg)
            if args.sweep == "schedule"
            else sm100_standard_candidates(
                cfg, allow_nondeterministic=args.allow_nondeterministic
            )
        )
    )
    workspace_type = (
        pkg.Sm100GenphaseSymmBuffer if genphase else pkg.Sm100BlockScaledSymmBuffer
    )
    workspace = workspace_type(dataclasses.replace(cfg, **candidates[0]))
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
            transformed = pkg.preprocess_block_scaled_weights(
                w13, w2, quant_kind=args.dtype, intermediate_size=args.intermediate
            )
            for parts, (weight, scale) in zip(legs, transformed, strict=True):
                parts.append((weight, scale.reshape(count, -1)))
            del w13, w2
        l1, l2 = (concatenate_block_scaled_weights(parts) for parts in legs)
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
        if genphase and args.dtype != "nvfp4":
            stage_genphase_inputs(tensors, workspace, quantize_input=True)
        else:
            from .nvfp4_nvfp4_bf16_cutedsl import staging as nvfp4_staging
            from .mxfp8_mxfp8_bf16_cutedsl import staging as mxfp8_staging

            staging = nvfp4_staging if args.dtype == "nvfp4" else mxfp8_staging
            extra = (
                {"norm_const": args.input_norm_const}
                if args.dtype == "nvfp4"
                else {"kind": args.dtype}
            )
            width = ((cfg.hidden // cfg.sf_vec_size + 3) // 4) * 4
            staging.stage_mega_moe_inputs(
                x,
                weights,
                ids,
                workspace.x,
                workspace.x_sf[:, :width],
                workspace.topk_idx,
                workspace.topk_weights,
                **extra,
            )
            workspace.note_staged_tokens(live)
            if args.dtype == "nvfp4":
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
                    getattr(workspace, name).fill_(getattr(args, name))
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
            pkg.autotune_sm100_block_scaled_mega_moe,
            tune_kwargs={
                "force_tune": True,
                "install_winner": False,
                "allow_nondeterministic": args.allow_nondeterministic,
            },
        )
    except BaseException:
        # A failed GPU rank cannot safely enter collective free with its peers.
        workspace = None
        raise
    finally:
        if workspace is not None:
            workspace.destroy()


def run_tuning(args):
    """Bootstrap an EP job and persist fresh results for each requested capacity."""
    import torch

    from .....config import BootstrapConfig
    from .....core.runtime import (
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
        mxfp8_cutedsl_runtime_requirements,
        nvfp4_cutedsl_runtime_requirements,
    )
    from .....kernel_src.sm100.next_cutedsl_megamoe import knob_cache_path

    from flashinfer import autotune_v2
    from flashinfer.autotuner import AutoTuner

    if args.kernel_variant == "genphase" and (
        args.combine_dtype != "bf16"
        or args.allow_nondeterministic
        or not args.do_finalize
        or args.activation != "swiglu"
    ):
        raise ValueError(
            "Blackwell GenPhase requires SwiGLU and separate BF16 finalization"
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
            bootstrap,
            (
                nvfp4_cutedsl_runtime_requirements
                if args.dtype == "nvfp4"
                else mxfp8_cutedsl_runtime_requirements
            )(bootstrap),
        )
    try:
        with autotune_v2():
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
            f"[moe_ep-tune] done; MegaMoE cache: {knob_cache_path() or 'DISABLED'}; "
            f"FlashInfer cache: {AutoTuner.get()._active_managed_store.env_dir}",
            flush=True,
        )
    return 0
