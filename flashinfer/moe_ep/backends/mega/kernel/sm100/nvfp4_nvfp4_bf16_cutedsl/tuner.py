"""Offline knob tuner for the NVFP4 cutedsl mega kernel.

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim (``--dtype
nvfp4``).  Kernel-specific pieces (dummy input creation, candidate
enumeration, knob resolution, the autotune entry point) live here, next to
the backend that consumes the recorded winners; the sweep loop and shared
helpers come from ``backends/mega/kernel/tuning.py``.
"""

from __future__ import annotations

from typing import Any

from ...tuning import finish_sweep, run_tuning as _run_tuning, schedule_candidates


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    from ......kernel_src.cutedsl_megamoe import (
        COMBINE_FORMAT_NAMES,
        autotune_nvfp4_mega_moe,
        create_dummy_nvfp4_inputs,
        nvfp4_candidates,
        resolve_knobs,
    )

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    symm_buffer: Any = None
    try:
        y, l1, l2, symm_buffer = create_dummy_nvfp4_inputs(
            rank,
            world_size,
            args.num_experts,
            max_tokens,
            live_tokens,
            args.topk,
            args.hidden,
            2 * args.intermediate,
            activation=args.activation,
            gate_up_clamp=args.gate_up_clamp,
            combine_dtype=args.combine_dtype,
            seed=args.seed,
        )
        candidates = nvfp4_candidates(
            combine_format=COMBINE_FORMAT_NAMES[args.combine_dtype],
            allow_in_kernel_fc2_reduce=args.allow_nondeterministic,
        )

        if args.sweep == "schedule":
            import json

            if args.base_knobs:
                base = json.loads(args.base_knobs)
                base = {
                    k: tuple(v) if isinstance(v, list) else v for k, v in base.items()
                }
            else:
                base, src = resolve_knobs(
                    dtype=args.dtype,
                    world_size=world_size,
                    hidden=args.hidden,
                    intermediate=2 * args.intermediate,
                    num_experts=args.num_experts,
                    topk=args.topk,
                    max_tokens=max_tokens,
                    combine_dtype=args.combine_dtype,
                    activation=args.activation,
                )
                if rank == 0:
                    print(f"[moe_ep-tune] schedule sweep base ({src}): {base}")
            candidates = schedule_candidates(base)

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
            autotune_nvfp4_mega_moe,
        )
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def run_tuning(args) -> int:
    return _run_tuning(args, tune_one)
