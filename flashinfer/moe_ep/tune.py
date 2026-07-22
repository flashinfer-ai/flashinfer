"""Offline knob tuner for the cutedsl mega-MoE path.

Runs the collective autotune sweep OUTSIDE any serving engine and persists
the winners in the knob cache (see
``kernel_src/cutedsl_megamoe/shim/knob_cache.py``). After tuning, an engine
that constructs the mega layer with ``knobs=None`` (the default) resolves the
recorded winner with a pure dict lookup — no compiles, no collectives, no
timing on the hot path.

Run with the SAME EP world size, GPU model, and geometry as production.
Multi-rank (matches a 4-GPU EP deployment)::

    torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \\
        --dtype nvfp4 --hidden 7168 --intermediate 2048 \\
        --num-experts 256 --topk 8 --max-tokens 8 512 2048

Single-rank (no torchrun)::

    MEGA_NO_DIST=1 python -m flashinfer.moe_ep.tune --dtype nvfp4 ...

``--intermediate`` is the model's post-SwiGLU width (the
``*MegaMoeConfig.intermediate_size`` convention); the shim-level conversion
(NVFP4 sessions size fc1 as ``2 * intermediate``) is applied internally, so
recorded cache keys match engine-time lookups exactly.

Nondeterministic candidates (``in_kernel_fc2_reduce``) are EXCLUDED by
default; pass ``--allow-nondeterministic`` to sweep them (a recorded ikr
winner makes the engine's output accumulation order nondeterministic).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m flashinfer.moe_ep.tune",
        description="Offline cutedsl mega-MoE knob tuner (writes the knob cache).",
    )
    parser.add_argument(
        "--dtype", choices=("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"), default="nvfp4"
    )
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--intermediate",
        type=int,
        required=True,
        help="model post-SwiGLU intermediate size "
        "(*MegaMoeConfig.intermediate_size convention)",
    )
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        required=True,
        help="buffer capacities (tokens/rank) to tune, one "
        "sweep each — use the engine's actual buffer size(s)",
    )
    parser.add_argument(
        "--combine-dtype",
        choices=("bf16", "mxfp8", "nvfp4"),
        default="bf16",
        help="cross-rank combine wire (nvfp4 dtype only)",
    )
    parser.add_argument("--gate-up-clamp", type=float, default=None)
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="also sweep in_kernel_fc2_reduce candidates",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="truncate the candidate list (smoke testing)",
    )
    parser.add_argument(
        "--live-tokens",
        type=int,
        default=None,
        help="live token count to stage and time (default: the bucket size). "
        "Use a decode-like count (e.g. 256) to tune for decode steps while "
        "keeping the engine's buffer bucket; the cache entry is still keyed "
        "on --max-tokens, so write decode-tuned winners to a separate cache "
        "file (FLASHINFER_MOE_EP_KNOB_CACHE).",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=None,
        help="target per-launch expert-load skew (max-load/mean-load) for the "
        "tuning routing, e.g. 18 for the DSV4-measured mean. Default keeps "
        "the near-uniform random routing — which CANNOT discriminate "
        "skew-sensitive knobs (load_balance_mode, scheduling); pass the "
        "measured production ratio (FI_MOE_EP_LOAD_STATS cold run).",
    )
    parser.add_argument(
        "--sweep",
        choices=("default", "schedule"),
        default="default",
        help="'default' sweeps tile/flag_batch/token-back(/ikr); 'schedule' "
        "pins those from --base-knobs (or the current cache winner) and "
        "sweeps load_balance_mode x group_hint — the skew-sensitive axes.",
    )
    parser.add_argument(
        "--base-knobs",
        type=str,
        default=None,
        help="JSON knob dict used as the base for --sweep schedule "
        "(default: resolve the current cache/heuristic winner for this key)",
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _restage_skewed_routing(
    symm_buffer,
    num_tokens: int,
    topk: int,
    num_experts: int,
    target_ratio: float,
    seed: int,
) -> None:
    """Overwrite the staged routing with a skewed expert distribution.

    Expert popularity follows a power law tuned (bisection on the exponent)
    so the realized per-launch max/mean load ratio approximates
    ``target_ratio`` — matching cold-run production stats instead of the
    near-uniform default that hides skew-sensitive knob behavior.
    """
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)

    def realized(alpha: float) -> tuple:
        w = torch.arange(1, num_experts + 1, device="cuda", dtype=torch.float32)
        w = w.pow(-alpha)
        w = w[torch.randperm(num_experts, generator=g, device="cuda")]
        ids = torch.multinomial(
            w.expand(num_tokens, -1), topk, replacement=False, generator=g
        )
        counts = torch.bincount(ids.flatten(), minlength=num_experts).float()
        return float(counts.max() / counts.mean().clamp(min=1e-9)), ids

    lo, hi = 0.0, 3.0
    ids = None
    for _ in range(12):
        mid = (lo + hi) / 2
        ratio, ids = realized(mid)
        if ratio < target_ratio:
            lo = mid
        else:
            hi = mid
    assert ids is not None
    symm_buffer.topk_idx[:num_tokens].copy_(ids.to(torch.int64))
    symm_buffer.topk_idx[num_tokens:].fill_(-1)
    symm_buffer.topk_weights[:num_tokens].fill_(1.0 / topk)


def _tune_one(
    args: argparse.Namespace, rank: int, world_size: int, max_tokens: int
) -> dict:
    import json

    from .kernel_src.cutedsl_megamoe import (
        autotune_mxfp8_mega_moe,
        autotune_nvfp4_mega_moe,
        create_dummy_mxfp8_inputs,
        create_dummy_nvfp4_inputs,
    )
    from .kernel_src.cutedsl_megamoe.shim.autotune import (
        mxfp8_candidates,
        nvfp4_candidates,
    )
    from .kernel_src.cutedsl_megamoe.shim.nvfp4 import COMBINE_FORMAT_NAMES

    is_nvfp4 = args.dtype == "nvfp4"
    live_tokens = args.live_tokens if args.live_tokens is not None else max_tokens
    if live_tokens > max_tokens:
        raise SystemExit("--live-tokens must be <= --max-tokens")
    symm_buffer: Any = None
    try:
        if is_nvfp4:
            y, l1, l2, symm_buffer = create_dummy_nvfp4_inputs(
                rank,
                world_size,
                args.num_experts,
                max_tokens,
                live_tokens,
                args.topk,
                args.hidden,
                2 * args.intermediate,
                gate_up_clamp=args.gate_up_clamp,
                seed=args.seed,
            )
            candidates = nvfp4_candidates(
                combine_format=COMBINE_FORMAT_NAMES[args.combine_dtype],
                allow_in_kernel_fc2_reduce=args.allow_nondeterministic,
            )
            tune = autotune_nvfp4_mega_moe
        else:
            y, l1, l2, symm_buffer = create_dummy_mxfp8_inputs(
                rank,
                world_size,
                args.num_experts,
                max_tokens,
                live_tokens,
                args.topk,
                args.hidden,
                args.intermediate,
                kind=args.dtype,
                gate_up_clamp=args.gate_up_clamp,
                seed=args.seed,
            )
            candidates = mxfp8_candidates(
                in_kernel_fc2_reduce=args.allow_nondeterministic,
            )
            tune = autotune_mxfp8_mega_moe

        if args.sweep == "schedule":
            import json as _json

            from .kernel_src.cutedsl_megamoe import resolve_knobs

            if args.base_knobs:
                base = _json.loads(args.base_knobs)
                base = {
                    k: tuple(v) if isinstance(v, list) else v for k, v in base.items()
                }
            else:
                base, src = resolve_knobs(
                    dtype=args.dtype,
                    world_size=world_size,
                    hidden=args.hidden,
                    intermediate=(2 if is_nvfp4 else 1) * args.intermediate,
                    num_experts=args.num_experts,
                    topk=args.topk,
                    max_tokens=max_tokens,
                )
                if rank == 0:
                    print(f"[moe_ep-tune] schedule sweep base ({src}): {base}")
            candidates = [
                {**base, "load_balance_mode": lb, "group_hint": gh}
                for lb in ("atomic_counter", "static")
                for gh in (None, 128, 256, 512)
            ]

        if args.skew is not None:
            _restage_skewed_routing(
                symm_buffer,
                live_tokens,
                args.topk,
                args.num_experts,
                args.skew,
                args.seed + rank,
            )

        if args.max_candidates is not None:
            candidates = candidates[: args.max_candidates]
        if rank == 0:
            print(
                f"[moe_ep-tune] {args.dtype} max_tokens={max_tokens} "
                f"live_tokens={live_tokens}: {len(candidates)} candidates",
                flush=True,
            )

        winner = tune(
            y,
            l1,
            l2,
            symm_buffer,
            num_tokens=live_tokens,
            candidates=candidates,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )
        if rank == 0:
            print(
                f"[moe_ep-tune] recorded winner for max_tokens={max_tokens}: "
                f"{json.dumps(winner, default=list)}",
                flush=True,
            )
        return winner
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.combine_dtype != "bf16" and args.dtype != "nvfp4":
        print("--combine-dtype is only wired for --dtype nvfp4", file=sys.stderr)
        return 2

    import torch

    from .kernel_src.cutedsl_megamoe import finalize_dist, init_dist
    from .kernel_src.cutedsl_megamoe.shim.knob_cache import _cache_path

    rank, world_size = init_dist()
    try:
        for max_tokens in args.max_tokens:
            _tune_one(args, rank, world_size, max_tokens)
        torch.cuda.synchronize()
    finally:
        finalize_dist()
    if rank == 0:
        path = _cache_path()
        print(f"[moe_ep-tune] done; cache: {path or 'DISABLED'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
