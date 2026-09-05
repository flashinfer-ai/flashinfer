# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online (warmup-time) knob autotuning for the SM90 Hopper FP8 MegaMoE frontend.

Sibling-fork mirror of ``kernel_src/cutedsl_megamoe/shim/autotune.py``: times
a curated candidate knob set on the live problem and applies the winner to
the session's frontend, replacing the token-bucket heuristic table with a
measured choice.  Candidates are the heuristic table's per-bucket winners
plus the geometries that win neighbouring buckets — the configurations that
the kernel drop's four-rank sweep found competitive anywhere.

The tune is a COLLECTIVE operation: the mega kernel's dispatch/combine spans
all EP ranks, so every rank must call the autotune entry point in the same
iteration with the same candidate list.  Ranks compile and launch each
candidate in lockstep (barriers around compile and timing), and the winner
is agreed on by all-reducing per-candidate times with MAX (the slowest rank
is the real latency of a collective kernel) — the argmin index is then
identical everywhere.

Cost: one ``cute.compile`` per candidate.  Unlike the SM100 tree (minutes
per compile), the SM90 kernel compiles in seconds, so the default candidate
list finishes in about a minute.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch

from .tuner import default_knobs, is_valid


def _sweep_geometries() -> List[Dict[str, Any]]:
    """Every geometry that wins at least one bucket of the kernel drop's
    four-rank sweep: ``heuristic_config.HEURISTIC_CONFIGS``, both scale
    modes, deduplicated (16 entries today).  Derived from the table so a
    future table refresh updates the candidate set automatically."""
    from moe_hopper_fp8.heuristic_config import HEURISTIC_CONFIGS

    out: List[Dict[str, Any]] = []
    seen = set()
    for table in HEURISTIC_CONFIGS.values():
        for c in table.values():
            key = (c.swap_ab, c.pingpong, c.mma_tiler_mnk, c.cluster_shape_mnk)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "swap_ab": c.swap_ab,
                    "pingpong": c.pingpong,
                    "mma_tiler_mnk": tuple(c.mma_tiler_mnk),
                    "cluster_shape_mnk": tuple(c.cluster_shape_mnk),
                    "fp8_accum_mode": "1xacc",
                }
            )
    return out


def hopper_fp8_candidates(
    *,
    fp8_scale_mode: str = "per_tensor",
    max_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """Default candidate knob dicts: heuristic winner first, then every
    geometry that wins some bucket of the drop's sweep.  The heuristic
    winner leads so a tie keeps the established default."""
    out: List[Dict[str, Any]] = []
    seen = set()

    def _add(knobs: Dict[str, Any]) -> None:
        key = tuple(
            sorted(
                (k, tuple(v) if isinstance(v, tuple) else v) for k, v in knobs.items()
            )
        )
        if key not in seen and is_valid(knobs):
            seen.add(key)
            out.append(knobs)

    _add(default_knobs(max_tokens, fp8_scale_mode=fp8_scale_mode))
    # Each geometry is swept under both validated token-back modes (the
    # epi/reuse crossover is bucket-dependent; ``standalone_warps`` stays
    # out of the perf candidates until it has measured numbers).
    for geometry in _sweep_geometries():
        for token_back in ("epi_warps", "reuse_dispatch_warps"):
            _add({**geometry, "token_back_mode": token_back})
    return out


def autotune_knobs(
    frontend: Any,
    launch: Callable[[], None],
    candidates: List[Dict[str, Any]],
    *,
    label: str,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    on_winner: Optional[Callable[[Dict[str, Any], float], None]] = None,
) -> Dict[str, Any]:
    """Time each candidate on the live problem and apply the winner.

    ``frontend`` is a :class:`.hopper_fp8.MegaMoEHopperFp8Frontend` (must
    have ``apply_knobs``); ``launch`` is a zero-arg closure that runs one
    synchronized forward with the caller's real staged inputs.

    ``on_winner`` (optional) is called once with ``(winner, p50_seconds)``
    after the winner is applied — used to persist the result in the knob
    cache.  It runs on every rank; the callback decides who writes.

    COLLECTIVE: every EP rank must call this in the same iteration with the
    same ``candidates`` (order included).  Returns the winning knob dict.
    """
    if not candidates:
        raise ValueError("autotune_knobs needs a non-empty candidate list.")

    from .comm import ensure_not_capturing

    ensure_not_capturing("knobs='auto' collective autotune sweep")

    import torch.distributed as dist

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if collective else 0

    def _barrier() -> None:
        if collective:
            dist.barrier()

    scores: List[float] = []
    for knobs in candidates:
        # A candidate failure (ctor reject / compile error) is deterministic
        # across ranks -- same static problem, same knobs -- so scoring it
        # inf keeps the collective iteration aligned.
        try:
            frontend.apply_knobs(knobs)
            _barrier()
            for _ in range(warmup_iters):  # first launch compiles
                launch()
            _barrier()
            iters: List[float] = []
            for _ in range(timed_iters):  # launch() syncs internally
                t0 = time.perf_counter()
                launch()
                iters.append(time.perf_counter() - t0)
            scores.append(statistics.median(iters))
        except Exception as exc:  # noqa: BLE001 -- score-and-continue by design
            warnings.warn(
                f"[sm90-autotune] {label}: candidate {knobs} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            scores.append(math.inf)
        _barrier()

    t = torch.tensor(scores, dtype=torch.float64, device="cuda")
    if collective:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank = real latency
    best = int(torch.argmin(t).item())
    if not math.isfinite(float(t[best])):
        raise RuntimeError(
            f"[sm90-autotune] {label}: every candidate failed to compile/run."
        )
    winner = candidates[best]
    frontend.apply_knobs(winner)
    if on_winner is not None:
        on_winner(winner, float(t[best]))
    if rank == 0:
        ranked = sorted(zip(t.tolist(), candidates, strict=False), key=lambda kv: kv[0])
        summary = "\n".join(f"    {us * 1e6:10.1f} us  {knobs}" for us, knobs in ranked)
        print(
            f"[sm90-autotune] {label}: winner {winner} "
            f"({float(t[best]) * 1e6:.1f} us median, max across ranks) "
            f"out of {len(candidates)} candidates:\n{summary}",
            flush=True,
        )
    return winner


def autotune_hopper_fp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Autotune the SM90 FP8 mega session on the caller's staged inputs.

    Arguments mirror :func:`.hopper_fp8.hopper_fp8_mega_moe`; ``y`` is
    clobbered by the candidate launches.  Applies the winner and returns its
    knob dict; subsequent ``hopper_fp8_mega_moe`` calls on ``symm_buffer``
    reuse the winning compile.  COLLECTIVE -- see :func:`autotune_knobs`.
    """
    from .hopper_fp8 import hopper_fp8_mega_moe

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and topk reduce) complete.
        hopper_fp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    cfg = symm_buffer._frontend.config
    if candidates is None:
        candidates = hopper_fp8_candidates(
            fp8_scale_mode=cfg.fp8_scale_mode,
            max_tokens=cfg.num_tokens_per_rank,
        )

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            record_knobs(
                winner,
                dtype=cfg.kind,
                fp8_scale_mode=cfg.fp8_scale_mode,
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                p50_us=p50_s * 1e6,
                source="autotune",
            )

    return autotune_knobs(
        symm_buffer._frontend,
        launch,
        candidates,
        label="sm90_fp8_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
    )


__all__ = [
    "autotune_hopper_fp8_mega_moe",
    "autotune_knobs",
    "hopper_fp8_candidates",
]
