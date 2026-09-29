"""Offline tuning shared by the SM107 NVFP4 and MXFP8 backends.

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
    from .....kernel_src.sm107.next_cutedsl_megamoe import Sm107QuantKind


def _backend_module(quant_kind: str, name: str):
    backend = (
        "nvfp4_nvfp4_bf16_cutedsl"
        if quant_kind == "nvfp4"
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
    extra = {} if quant_kind == "nvfp4" else {"kind": quant_kind}

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


def _stage_dummy_inputs(args, rank, symm_buffer, live_tokens: int, quant_kind: str):
    """Random activations + near-uniform distinct top-k routing, staged."""
    import torch

    staging_mod = _backend_module(quant_kind, "staging")
    extra = {} if quant_kind == "nvfp4" else {"kind": quant_kind}

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
    staged = staging_mod.stage_mega_moe_inputs(
        x,
        topk_weights,
        topk_ids,
        symm_buffer.x,
        symm_buffer.x_sf,
        symm_buffer.topk_idx,
        symm_buffer.topk_weights,
        **extra,
    )
    symm_buffer.note_staged_tokens(staged)


def tune_one(
    args, rank: int, world_size: int, max_tokens: int, quant_kind: str
) -> dict:
    import json

    import torch

    from .....kernel_src.sm107.next_cutedsl_megamoe import (
        autotune_sm107_block_scaled_mega_moe,
        get_symm_buffer_for_sm107_block_scaled_mega_moe,
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
        # Candidates copy inputs from this session; its kernel is never launched.
        symm_buffer = get_symm_buffer_for_sm107_block_scaled_mega_moe(
            args.num_experts,
            max_tokens,
            args.topk,
            args.hidden,
            args.intermediate,
            rank,
            world_size,
            quant_kind=cast("Sm107QuantKind", quant_kind),
            gate_up_clamp=args.gate_up_clamp,
        )
        _stage_dummy_inputs(args, rank, symm_buffer, live_tokens, quant_kind)
        y = torch.empty(live_tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

        if args.sweep == "schedule":
            if args.base_knobs:
                base = json.loads(args.base_knobs)
                base = {
                    k: tuple(v) if isinstance(v, list) else v for k, v in base.items()
                }
            else:
                base, src = resolve_knobs(
                    dtype=quant_kind,
                    world_size=world_size,
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    num_experts=args.num_experts,
                    topk=args.topk,
                    max_tokens=max_tokens,
                )
                if rank == 0:
                    print(f"[moe_ep-tune] schedule sweep base ({src}): {base}")
            candidates = sm107_schedule_candidates(base)
        else:
            candidates = sm107_candidates(
                quant_kind,
                allow_in_kernel_fc2_reduce=args.allow_nondeterministic,
            )

        return finish_sweep(
            args,
            rank,
            max_tokens,
            live_tokens,
            symm_buffer,
            y,
            l1,
            l2,
            candidates,
            autotune_sm107_block_scaled_mega_moe,
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

    if args.combine_dtype != "bf16":
        raise SystemExit("the SM107 backends are wired for bf16 combine only")

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
            f"[moe_ep-tune] done; cache: {knob_cache_path() or 'DISABLED'}", flush=True
        )
    return 0
