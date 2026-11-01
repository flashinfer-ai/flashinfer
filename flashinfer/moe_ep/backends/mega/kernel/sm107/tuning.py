"""Offline tuning shared by the SM107 block-scaled backends.

The core runtime owns torch.distributed and NVSHMEM initialization.
Each candidate needs a new session because kernel knobs are fixed at
construction. ``MEGA_NO_DIST=1`` selects single-rank execution.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any, cast

from ..tuning import finish_sweep

if TYPE_CHECKING:
    from .automatic import RubinBackend


def _backend_module(quant_kind: str, name: str):
    backend = (
        "nvfp4_nvfp4_bf16_cutedsl"
        if quant_kind == "nvfp4"
        else "mxfp8_mxfp4_bf16_cutedsl"
        if quant_kind == "mxfp4_mxfp8"
        else "mxfp8_mxfp8_bf16_cutedsl"
    )
    return importlib.import_module(f".{backend}.{name}", __package__)


_WEIGHT_CHUNK_EXPERTS = 8  # bf16 generation + quantize peak-memory bound


def _dummy_transformed_weights(args, rank: int, world_size: int, quant_kind: str):
    """Random local expert slice in kernel layout, quantized chunk-wise."""
    import torch

    from .....weights import MoEWeightPack
    from .....kernel_src.sm107.next_cutedsl_megamoe import (
        concatenate_block_scaled_weights,
    )

    weights_mod = _backend_module(quant_kind, "weights")
    extra = {} if quant_kind in ("nvfp4", "mxfp4_mxfp8") else {"kind": quant_kind}

    experts_per_rank = args.num_experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 7 * rank)
    fc1_parts, fc2_parts = [], []
    for begin in range(0, experts_per_rank, _WEIGHT_CHUNK_EXPERTS):
        count = min(_WEIGHT_CHUNK_EXPERTS, experts_per_rank - begin)
        w13 = (
            torch.randn(
                count,
                2 * args.intermediate,
                args.hidden,
                device="cuda",
                dtype=torch.float32,
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
                dtype=torch.float32,
                generator=generator,
            )
            * args.intermediate**-0.5
        ).to(torch.bfloat16)
        (fc1_w, fc1_sf), (fc2_w, fc2_sf) = weights_mod.preprocess_mega_weights(
            MoEWeightPack(w13=w13, w2=w2),
            intermediate_size=args.intermediate,
            hidden_size=args.hidden,
            **extra,
        )
        fc1_parts.append((fc1_w, fc1_sf.reshape(count, -1)))
        fc2_parts.append((fc2_w, fc2_sf.reshape(count, -1)))
        del w13, w2
        torch.cuda.empty_cache()
    return (
        concatenate_block_scaled_weights(fc1_parts),
        concatenate_block_scaled_weights(fc2_parts),
    )


def _dummy_inputs(args, rank, live_tokens: int):
    """BF16 activations and near-uniform distinct top-k routing."""
    import torch

    from .....tensors import MoEEpTensors

    generator = torch.Generator(device="cuda").manual_seed(args.seed + 13 * rank)
    x = torch.randn(
        live_tokens,
        args.hidden,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    scores = torch.rand(
        live_tokens, args.num_experts, device="cuda", generator=generator
    )
    topk_ids = scores.topk(args.topk, dim=-1).indices.to(torch.int32)
    topk_weights = (
        torch.rand(live_tokens, args.topk, device="cuda", generator=generator) + 0.5
    )
    return MoEEpTensors(x, topk_ids, topk_weights)


def tune_one(
    args, rank: int, world_size: int, max_tokens: int, quant_kind: str
) -> dict:
    import json

    import torch

    from .....kernel_src.sm107.next_cutedsl_megamoe import (
        resolve_knobs,
        sm107_candidates,
        sm107_schedule_candidates,
    )

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")

    symm_buffer: Any = None
    try:
        transformed = _dummy_transformed_weights(args, rank, world_size, quant_kind)
        l1, l2 = transformed
        from .....config import BootstrapConfig, FleetParams
        from .....core.kernel.registry import create_mega_kernel
        from .forward_tuning import ForwardExecution
        from .nvfp4_nvfp4_bf16_cutedsl.config import (
            Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        )
        from .mxfp8_mxfp8_bf16_cutedsl.config import (
            Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        )
        from .mxfp8_mxfp4_bf16_cutedsl.config import (
            Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
        )

        config_type = (
            Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
            if quant_kind == "nvfp4"
            else Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
            if quant_kind == "mxfp4_mxfp8"
            else Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
        )
        extra = (
            {
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
            if quant_kind == "nvfp4"
            else {}
            if quant_kind == "mxfp4_mxfp8"
            else {"kind": quant_kind}
        )
        config = config_type(
            intermediate_size=args.intermediate,
            top_k=args.topk,
            knobs="auto",
            kernel_variant=args.kernel_variant,
            combine_dtype=args.combine_dtype,
            gate_up_clamp=args.gate_up_clamp,
            activation=args.activation,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            in_kernel_fc2_reduce=args.allow_nondeterministic,
            **extra,
        )
        backend = cast("RubinBackend", create_mega_kernel(config))
        backend.bind_ep_bootstrap(BootstrapConfig(rank=rank, world_size=world_size))
        fleet = FleetParams(args.num_experts, max_tokens, args.hidden)
        symm_buffer = backend._allocate_workspace(fleet)
        tensors = _dummy_inputs(args, rank, live_tokens)
        output = torch.empty(
            live_tokens, args.hidden, device="cuda", dtype=torch.bfloat16
        )
        backend.stage_inputs(tensors, symm_buffer, quantize_input=True)
        execution = ForwardExecution(
            backend, tensors, quantize_input=True, output=output
        )

        def tune_forward(y, l1, l2, workspace, **kwargs):
            # finish_sweep may have replaced the routing with a skewed workload.
            if args.skew is not None:
                tensors.topk_ids.copy_(workspace.topk_idx[:live_tokens])
                tensors.topk_weights.copy_(workspace.topk_weights[:live_tokens])
            return execution.tune(l1, l2, workspace, **kwargs)

        if args.sweep == "schedule":
            if args.base_knobs:
                base = json.loads(args.base_knobs)
                base = {
                    k: tuple(v) if isinstance(v, list) else v for k, v in base.items()
                }
            else:
                base, src = resolve_knobs(
                    dtype=quant_kind,
                    kernel_variant=args.kernel_variant,
                    combine_dtype=args.combine_dtype,
                    world_size=world_size,
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    num_experts=args.num_experts,
                    topk=args.topk,
                    max_tokens=max_tokens,
                    activation=args.activation,
                    situ_beta=args.situ_beta,
                    situ_linear_beta=args.situ_linear_beta,
                    gate_up_clamp=args.gate_up_clamp,
                )
                if rank == 0:
                    print(f"[moe_ep-tune] schedule sweep base ({src}): {base}")
            candidates = sm107_schedule_candidates(
                base, kernel_variant=args.kernel_variant
            )
        else:
            candidates = sm107_candidates(
                quant_kind,
                allow_in_kernel_fc2_reduce=args.allow_nondeterministic
                and args.combine_dtype == "bf16",
                kernel_variant=args.kernel_variant,
            )

        return finish_sweep(
            args,
            rank,
            max_tokens,
            live_tokens,
            symm_buffer,
            output,
            l1,
            l2,
            candidates,
            tune_forward,
            tune_kwargs={
                "force_tune": True,
                "install_winner": False,
                "allow_nondeterministic": args.allow_nondeterministic,
            },
        )
    except BaseException:
        # A rank-local GPU failure must escape to the job supervisor without
        # entering collective free while peers may still be running.
        symm_buffer = None
        raise
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def run_tuning(args, quant_kind: str) -> int:
    """Initialize the runtime and tune each token-capacity bucket."""
    import torch
    from flashinfer.autotuner import AutoTuner
    from flashinfer import autotune_v2

    if args.combine_dtype not in ("bf16", "nvfp4", "mxfp8"):
        raise SystemExit("SM107 supports bf16, nvfp4, or mxfp8 combine")
    if args.kernel_variant == "genphase" and args.combine_dtype != "bf16":
        raise SystemExit("GenPhase requires BF16 combine")
    from .validation import validate_input_norm_const

    for name in ("input_norm_const", "fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        validate_input_norm_const(getattr(args, name), name=name)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if os.environ.get("MEGA_NO_DIST") == "1" and world_size != 1:
        raise SystemExit("MEGA_NO_DIST=1 cannot be used in a multi-rank SM107 job")
    from .....kernel_src.sm107.next_cutedsl_megamoe import require_sm107_dsl

    require_sm107_dsl()
    runtime = None
    if os.environ.get("MEGA_NO_DIST") == "1" or world_size == 1:
        os.environ.setdefault("MEGA_NO_DIST", "1")
        rank, world_size = 0, 1
        torch.cuda.set_device(0)
    else:
        from .....config import BootstrapConfig
        from .....core.runtime import (
            bootstrap_moe_ep_runtime,
            ensure_moe_ep_cuda_device,
            sm107_block_scaled_runtime_requirements,
        )

        bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
        ensure_moe_ep_cuda_device(bootstrap)
        runtime = bootstrap_moe_ep_runtime(
            bootstrap, sm107_block_scaled_runtime_requirements(bootstrap)
        )
    try:
        with autotune_v2():
            for max_tokens in args.max_tokens:
                tune_one(args, rank, world_size, max_tokens, quant_kind)
        torch.cuda.synchronize()
    except BaseException:
        runtime = None
        raise
    finally:
        if runtime is not None:
            from .....core.runtime import finalize_moe_ep_runtime

            finalize_moe_ep_runtime(runtime)
    if rank == 0:
        from .....kernel_src.sm107.next_cutedsl_megamoe import knob_cache_path

        print(
            f"[moe_ep-tune] done; MegaMoE cache: {knob_cache_path() or 'DISABLED'}; "
            f"FlashInfer cache: {AutoTuner.get()._active_managed_store.env_dir}",
            flush=True,
        )
    return 0
