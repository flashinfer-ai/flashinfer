# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tuning for the standard and GenPhase Blackwell kernels."""

from __future__ import annotations

import dataclasses
import json
import os

from ..tuning import finish_sweep, restage_skewed_routing


def _dummy_workload(args, rank, world_size, live):
    import torch

    from .....tensors import MoEEpTensors
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.kernel_helpers import (
        concatenate_block_scaled_weights,
        preprocess_block_scaled_weights,
    )

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
        transformed = preprocess_block_scaled_weights(
            w13, w2, quant_kind=args.dtype, intermediate_size=args.intermediate
        )
        for parts, (weight, scale) in zip(legs, transformed, strict=True):
            parts.append((weight, scale.reshape(count, -1)))
        del w13, w2
    prepared = tuple(concatenate_block_scaled_weights(parts) for parts in legs)
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
    return prepared, MoEEpTensors(x, ids, weights)


def _schedule_candidates(args, cfg, live, sms):
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.knob_cache import (
        resolve_knobs,
    )
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning import (
        sm100_genphase_candidates,
        sm100_schedule_candidates,
    )

    constraints = (
        {"reduce_topk_in_kernel": False}
        if args.kernel_variant == "auto"
        and cfg.kernel_variant == "inference"
        and not args.allow_nondeterministic
        else None
    )
    base = (
        json.loads(args.base_knobs)
        if args.base_knobs
        else resolve_knobs(
            cfg, sm_count=sms, live_tokens=live, constraints=constraints
        )[0]
    )
    if "cluster_shape_mnk" in base:
        raise ValueError(
            "cluster_shape_mnk belongs to the retired legacy tuning schema. "
            "Retune with native cluster_shape_mn=(M, N); renaming an old "
            "cache entry does not make it compatible."
        )
    base = {k: tuple(v) if isinstance(v, list) else v for k, v in base.items()}
    if cfg.kernel_variant == "genphase":
        return sm100_genphase_candidates(
            cfg,
            sm_count=sms,
            live_tokens=live,
            constraints={k: v for k, v in base.items() if k != "schedule_policy"},
        )
    return sm100_schedule_candidates(base, cfg)


def _variant_candidates(args, backend, workspace, live):
    if args.sweep == "default" and args.max_candidates is None:
        return None

    import torch

    from .variant_selection import _candidates

    eligible = backend._eligible(workspace.fleet)
    if args.sweep == "default":
        return {
            variant: _candidates(
                cfg, live, allow_nondeterministic=args.allow_nondeterministic
            )[: args.max_candidates]
            for variant, cfg in eligible.items()
        }

    # Use one set of schedule bases even when ranks have different cache files.
    candidates, error = None, None
    if backend.ep_rank == 0:
        try:
            sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            candidates = {
                variant: _schedule_candidates(args, cfg, live, sms)[
                    : args.max_candidates
                ]
                for variant, cfg in eligible.items()
            }
            if any(not choices for choices in candidates.values()):
                raise ValueError("no valid schedule candidates for an eligible variant")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    candidates, error = backend._gather((candidates, error))[0]
    if error is not None:
        raise ValueError(f"Cannot build Blackwell schedule candidates: {error}")
    return candidates


def _build_backend(args, rank, world_size, max_tokens):
    import torch

    from .....config import BootstrapConfig, FleetParams
    from .....core.kernel.registry import create_mega_kernel
    from .mxfp8_mxfp4_bf16_cutedsl.config import (
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .mxfp8_mxfp8_bf16_cutedsl.config import (
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from .nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    extra = {}
    if args.dtype == "nvfp4":
        config_type = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        extra = {
            "input_norm_const": args.input_norm_const,
            **{
                name: torch.full(
                    (args.num_experts // world_size,),
                    getattr(args, name),
                    device="cuda",
                    dtype=torch.float32,
                )
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
                if getattr(args, name) != 1.0
            },
        }
    elif args.dtype == "mxfp4_mxfp8":
        config_type = Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
    else:
        config_type = Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        extra = {"kind": args.dtype}
    config = config_type(
        intermediate_size=args.intermediate,
        top_k=args.topk,
        knobs="auto",
        kernel_variant=args.kernel_variant,
        combine_dtype=args.combine_dtype,
        activation=args.activation,
        situ_beta=args.situ_beta,
        situ_linear_beta=args.situ_linear_beta,
        gate_up_clamp=args.gate_up_clamp,
        enable_in_kernel_fc2_reduce=args.allow_nondeterministic,
        apply_routing_weights_before_fc2=args.apply_routing_weights_before_fc2,
        **extra,
    )
    bootstrap = BootstrapConfig(rank=rank, world_size=world_size)
    fleet = FleetParams(args.num_experts, max_tokens, args.hidden)
    backend = create_mega_kernel(config)
    backend.validate_init(bootstrap, fleet)
    return backend, bootstrap, fleet


def _sync_staged_routes(tensors, workspace):
    tensors.topk_ids.copy_(workspace.topk_idx[: tensors.num_tokens])
    tensors.topk_weights.copy_(workspace.topk_weights[: tensors.num_tokens])


def _tune_variants(args, rank, world_size, max_tokens, live):
    import torch

    from flashinfer.autotuner import AutoTuner
    from .variant_selection import NAMESPACE, _VariantProtocol

    backend, bootstrap, fleet = _build_backend(args, rank, world_size, max_tokens)
    weights, tensors = _dummy_workload(args, rank, world_size, live)
    output = torch.empty_like(tensors.hidden_states)
    workspace = backend.prepare_workspace(bootstrap, fleet)
    try:
        backend.stage_inputs(tensors, workspace, quantize_input=True)
        if args.skew is not None:
            staged = workspace.workspace
            restage_skewed_routing(
                staged, live, args.topk, args.num_experts, args.skew, args.seed + rank
            )
            _sync_staged_routes(tensors, staged)
        candidates = _variant_candidates(args, backend, workspace, live)
        record = AutoTuner.get().choose_with_protocol(
            NAMESPACE,
            _VariantProtocol(
                backend,
                workspace,
                weights,
                output,
                candidates=candidates,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                install_winner=False,
            ),
            force_tune=True,
        )
        winner = {name: record[name] for name in ("variant", "knobs")}
        if rank == 0:
            print(
                f"[moe_ep-tune] recorded winner for max_tokens={max_tokens}: "
                f"{json.dumps(winner, default=list)}",
                flush=True,
            )
    except BaseException:
        # A rank-local GPU failure must escape without a collective free.
        workspace = None
        raise
    finally:
        if workspace is not None:
            backend.destroy(workspace)
    return winner


def _forward_tuner(args, backend, tensors, output):
    from flashinfer.autotuner import AutoTuner
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning_protocol import (
        NAMESPACE,
        BlackwellTuningProtocol,
    )
    from .forward_tuning import ForwardExecution

    execution = ForwardExecution(backend, tensors, quantize_input=True, output=output)

    def tune(y, l1, l2, workspace, *, force_tune, **kwargs):
        if args.skew is not None:
            _sync_staged_routes(tensors, workspace)
        return AutoTuner.get().choose_with_protocol(
            NAMESPACE,
            BlackwellTuningProtocol(
                y,
                l1,
                l2,
                workspace,
                process_group=backend.ep_comm_group
                if backend.ep_world_size > 1
                else None,
                make_trial=execution.make_trial,
                execution_contract=execution.contract(),
                **kwargs,
            ),
            force_tune=force_tune,
        )

    return tune


def tune_one(args, rank, world_size, max_tokens):
    """Stage synthetic inputs and tune one capacity on every EP rank.

    ``args.live_tokens`` selects the workload bucket without shrinking the
    workspace. Schedule-only sweeps preserve the selected tile and cluster.
    """
    import torch

    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe import sources as pkg
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning import (
        sm100_standard_candidates,
    )

    live = args.live_tokens if args.live_tokens is not None else max_tokens
    if not 0 <= live <= max_tokens:
        raise ValueError("--live-tokens must be between zero and --max-tokens")
    if args.kernel_variant == "auto":
        return _tune_variants(args, rank, world_size, max_tokens, live)
    backend = None
    if args.do_finalize:
        backend, _, fleet = _build_backend(args, rank, world_size, max_tokens)
        cfg = backend._workspace_config(fleet)
    else:
        cfg = pkg.Sm100BlockScaledMoeConfig(
            num_total_experts=args.num_experts,
            max_tokens_per_rank=max_tokens,
            num_topk=args.topk,
            hidden=args.hidden,
            intermediate=args.intermediate,
            rank=rank,
            world_size=world_size,
            gate_up_clamp=args.gate_up_clamp,
            quant_kind=args.dtype,
            combine_dtype=args.combine_dtype,
            activation=args.activation,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            apply_routing_weights_before_fc2=args.apply_routing_weights_before_fc2,
            use_custom_finalize=True,
        )
    genphase = cfg.kernel_variant == "genphase"
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if args.sweep == "schedule":
        candidates = _schedule_candidates(args, cfg, live, sms)
    elif genphase:
        candidates = pkg.sm100_genphase_candidates(cfg, sm_count=sms, live_tokens=live)
    else:
        candidates = sm100_standard_candidates(
            cfg, allow_nondeterministic=args.allow_nondeterministic
        )
    workspace_type = (
        pkg.Sm100GenphaseSymmBuffer if genphase else pkg.Sm100BlockScaledSymmBuffer
    )
    workspace = workspace_type(dataclasses.replace(cfg, **candidates[0]))
    try:
        (l1, l2), tensors = _dummy_workload(args, rank, world_size, live)
        output = None
        tune = pkg.autotune_sm100_block_scaled_mega_moe
        if backend is not None:
            backend.stage_inputs(tensors, workspace, quantize_input=True)
            output = torch.empty_like(tensors.hidden_states)
            tune = _forward_tuner(args, backend, tensors, output)
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
                tensors.hidden_states,
                tensors.topk_weights,
                tensors.topk_ids,
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
            output,
            l1,
            l2,
            candidates,
            tune,
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
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import knob_cache_path

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
    persist = args.max_candidates is None
    try:
        with autotune_v2(persistent_cache=persist):
            for capacity in args.max_tokens:
                tune_one(args, rank, world_size, capacity)
        torch.cuda.synchronize()
    except BaseException:
        runtime = None
        raise
    finally:
        if runtime is not None:
            finalize_moe_ep_runtime(runtime)
    if rank == 0 and not persist:
        print("[moe_ep-tune] done; candidate-limited sweep results were not saved")
    elif rank == 0:
        print(
            f"[moe_ep-tune] done; MegaMoE cache: {knob_cache_path() or 'DISABLED'}; "
            f"FlashInfer cache: {AutoTuner.get()._active_managed_store.env_dir}",
            flush=True,
        )
    return 0
