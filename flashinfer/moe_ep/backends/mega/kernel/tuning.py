"""Shared offline-tuning helpers for the mega-kernel backend tuners.

Kernel-specific tuners live next to their backend
(``sm100/<name>/tuner.py``); :mod:`flashinfer.moe_ep.tune` is the CLI shim
that parses arguments and dispatches to them.  This module holds the pieces
every cutedsl tuner shares: the dist lifecycle + sweep loop
(:func:`run_tuning`), the timed-sweep tail (:func:`finish_sweep`), the
schedule-sweep candidate grid (:func:`schedule_candidates`), and the skewed
routing restage (:func:`restage_skewed_routing`).
"""

from __future__ import annotations

from typing import Any, Callable, List


def restage_skewed_routing(
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


def schedule_candidates(base: dict) -> List[dict]:
    """Expand a base knob dict into the --sweep schedule grid
    (load_balance_mode x group_hint — the skew-sensitive axes)."""
    return [
        {**base, "load_balance_mode": lb, "group_hint": gh}
        for lb in ("atomic_counter", "static")
        for gh in (None, 128, 256, 512)
    ]


def finish_sweep(
    args,
    rank: int,
    max_tokens: int,
    live_tokens: int,
    symm_buffer,
    y,
    l1,
    l2,
    candidates: List[dict],
    tune_fn: Callable,
) -> dict:
    """Common tail of a tuning sweep: optional skew restage, candidate
    truncation, the timed autotune call, and winner reporting."""
    import json

    if args.skew is not None:
        restage_skewed_routing(
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

    winner = tune_fn(
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


def run_tuning(args, tune_one: Callable[[Any, int, int, int], dict]) -> int:
    """Dist lifecycle + per-bucket sweep loop shared by the cutedsl tuners."""
    import torch

    from ....kernel_src.cutedsl_megamoe import (
        finalize_dist,
        init_dist,
        knob_cache_path,
    )

    rank, world_size = init_dist()
    try:
        for max_tokens in args.max_tokens:
            tune_one(args, rank, world_size, max_tokens)
        torch.cuda.synchronize()
    finally:
        finalize_dist()
    if rank == 0:
        path = knob_cache_path()
        print(f"[moe_ep-tune] done; cache: {path or 'DISABLED'}", flush=True)
    return 0
