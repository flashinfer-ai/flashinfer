"""Offline knob tuner for the SM90 pull-style FP8 mega kernel.

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim (``--dtype
sm90_fp8_e4m3`` / ``sm90_fp8_e5m2``).  Kernel-specific pieces (dummy input
creation, candidate enumeration, knob resolution, the autotune entry point)
live here, next to the backend that consumes the recorded winners; the sweep
loop and shared helpers come from ``backends/mega/kernel/tuning.py``.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from ...tuning import finish_sweep, run_tuning as _run_tuning, schedule_candidates


def _kind(args) -> Literal["fp8_e4m3", "fp8_e5m2"]:
    # CLI dtype "sm90_fp8_e4m3" -> shim kind "fp8_e4m3".
    kind = args.dtype.removeprefix("sm90_")
    if kind not in ("fp8_e4m3", "fp8_e5m2"):
        raise ValueError(f"unsupported SM90 FP8 dtype {args.dtype!r}")
    return cast(Literal["fp8_e4m3", "fp8_e5m2"], kind)


def tune_one(args, rank: int, world_size: int, max_tokens: int) -> dict:
    from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
        autotune_hopper_fp8_mega_moe,
        create_dummy_hopper_fp8_inputs,
        hopper_fp8_candidates,
        resolve_knobs,
    )

    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    symm_buffer: Any = None
    try:
        y, l1, l2, symm_buffer = create_dummy_hopper_fp8_inputs(
            rank,
            world_size,
            args.num_experts,
            max_tokens,
            live_tokens,
            args.topk,
            args.hidden,
            args.intermediate,
            kind=_kind(args),
            fp8_scale_mode=args.fp8_scale_mode,
            gate_up_clamp=args.gate_up_clamp,
            seed=args.seed,
        )
        candidates = hopper_fp8_candidates(
            fp8_scale_mode=args.fp8_scale_mode,
            max_tokens=max_tokens,
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
                    dtype=_kind(args),
                    fp8_scale_mode=args.fp8_scale_mode,
                    world_size=world_size,
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    num_experts=args.num_experts,
                    topk=args.topk,
                    max_tokens=max_tokens,
                    gate_up_clamp=args.gate_up_clamp,
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
            autotune_hopper_fp8_mega_moe,
            tune_kwargs={"gate_up_clamp": args.gate_up_clamp},
        )
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def run_tuning(args) -> int:
    from ......kernel_src.sm90 import pull_style_cutedsl_megakernel as pkg

    return _run_tuning(args, tune_one, pkg=pkg)
