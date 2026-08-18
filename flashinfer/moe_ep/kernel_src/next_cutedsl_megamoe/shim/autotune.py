# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective knob autotuning for the SM107 block-scaled mega kernel.

SM100 counterpart: ``cutedsl_megamoe/shim/autotune.py``.  The structural
difference: the SM107 kernel bakes every knob into the session at
construction (``BlockScaledSwapAbMegaMoeKernel`` + its device workspaces are
built from the ImplDesc), so a candidate cannot be applied to a live session
via ``apply_knobs``.  Each candidate instead builds a fresh
:class:`.block_scaled.Sm107BlockScaledSymmBuffer`, copies the caller's staged
inputs into it, times it, and destroys it.  The caller's session is left
untouched; the winner is recorded in the knob cache (``.knob_cache``) for
pure-lookup engine starts.

The tune is a COLLECTIVE operation: dispatch/combine span all EP ranks, so
every rank must call the autotune entry point with the same candidate list
(order included).  Ranks build and launch each candidate in lockstep and the
winner is agreed on by all-reducing per-candidate times with MAX (the slowest
rank is the real latency of a collective kernel).

Cost: one ``cute.compile`` + one symmetric-heap workspace build per
candidate, paid offline (``python -m flashinfer.moe_ep.tune``).  Narrow
``candidates`` to trade quality for sweep time.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from typing import Any, Dict, List, Optional

# Knob keys = Sm107BlockScaledMoeConfig field names (the shim owns the dialect;
# the backends map their config field names onto these).
KNOB_KEYS = (
    "mma_tiler_mnk",
    "cluster_shape_mn",
    "fallback_cluster_shape_mn",
    "schedule_policy",
    "work_id_mode",
    "fc2_use_bulk",
    "fc2_tma_stages",
    "epi_flag_batches",
    "token_in_flag_batch",
    "token_back_mode",
    "reduce_topk_in_kernel",
)


def is_valid_sm107(knobs: Dict[str, Any], base_config: Any) -> bool:
    """``True`` if ``knobs`` is constructible on ``base_config``'s geometry.

    Pure host-side check — replays the shim config validation (solver rules)
    without touching CUDA.
    """
    import dataclasses

    unknown = set(knobs) - set(KNOB_KEYS)
    if unknown:
        raise ValueError(f"unknown SM107 knob keys: {sorted(unknown)}")
    try:
        dataclasses.replace(base_config, **knobs)
    except (ValueError, TypeError):
        return False
    return True


def sm107_candidates(
    quant_kind: str = "nvfp4",
    *,
    allow_in_kernel_fc2_reduce: bool = False,
) -> List[Dict[str, Any]]:
    """Default SM107 candidate knob dicts (tile x launch x epi x fc2-bulk).

    16 candidates (32 with the ikr axis), spanning the axes the upstream
    Rubin TS4B report showed to matter (see TUNING.md):

    - tile N 128 vs 256 (K fixed at the kind's 2x-mode depth),
    - uniform (2,1) grouped/grid-stride launch vs mixed-CGA (4,1)+(2,1)
      phase-interleave/atomic launch,
    - epi flag batches (1,4) vs (2,4),
    - FC2 bulk TMA (2 stages) on/off.

    ``token_in_flag_batch`` and the phase-interleave hint are skew-sensitive —
    sweep them with :func:`sm107_schedule_candidates` instead.  An ikr winner
    makes the output accumulation order nondeterministic; it stays opt-in.
    """
    tile_k = 256 if quant_kind == "nvfp4" else 128
    launches = (
        # (cluster, fallback, schedule_policy, work_id_mode)
        ((2, 1), None, ("grouped", None), "grid_stride"),
        ((4, 1), (2, 1), ("phase_interleave", None), "atomic_counter"),
    )
    out: List[Dict[str, Any]] = []
    for tile_n in (128, 256):
        for cluster, fallback, schedule, work_id in launches:
            for epi in ((1, 4), (2, 4)):
                for fc2_bulk in (False, True):
                    for ikr in (
                        (False, True) if allow_in_kernel_fc2_reduce else (False,)
                    ):
                        out.append(
                            {
                                "mma_tiler_mnk": (256, tile_n, tile_k),
                                "cluster_shape_mn": cluster,
                                "fallback_cluster_shape_mn": fallback,
                                "schedule_policy": schedule,
                                "work_id_mode": work_id,
                                "fc2_use_bulk": fc2_bulk,
                                "fc2_tma_stages": 2 if fc2_bulk else None,
                                "epi_flag_batches": epi,
                                "token_in_flag_batch": 1,
                                "token_back_mode": "epi_warps",
                                "reduce_topk_in_kernel": ikr,
                            }
                        )
    return out


def sm107_schedule_candidates(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a base knob dict into the skew-sensitive schedule grid
    (phase-interleave hint x token-in flag batch)."""
    out: List[Dict[str, Any]] = []
    for hint in (None, 3, 4, 6):
        for tif in (1, 4):
            out.append(
                {
                    **base,
                    "schedule_policy": ("phase_interleave", hint),
                    "work_id_mode": "atomic_counter",
                    "token_in_flag_batch": tif,
                }
            )
    return out


def autotune_sm107_block_scaled_mega_moe(
    y,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: Optional[int] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> Dict[str, Any]:
    """Time each candidate on ``symm_buffer``'s staged inputs; record the winner.

    ``symm_buffer`` provides the geometry and the staged activation / SF /
    routing payloads (its own compiled session is never launched or mutated).
    ``y`` is clobbered by candidate launches.  COLLECTIVE — every EP rank must
    call this with the same ``candidates``.  Returns the winning knob dict;
    rank 0 persists it via :func:`.knob_cache.record_knobs`.
    """
    import dataclasses

    import torch
    import torch.distributed as dist

    from .block_scaled import Sm107BlockScaledSymmBuffer, sm107_block_scaled_mega_moe
    from .comm import ensure_not_capturing

    ensure_not_capturing("SM107 collective autotune sweep")

    cfg = symm_buffer.config
    if candidates is None:
        candidates = sm107_candidates(cfg.quant_kind)
    if not candidates:
        raise ValueError("autotune needs a non-empty candidate list.")
    if num_tokens is None:
        num_tokens = symm_buffer.staged_tokens()
    if num_tokens is None:
        raise ValueError("num_tokens unset and no tokens staged on symm_buffer.")

    collective = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if collective else 0

    def _barrier() -> None:
        if collective:
            dist.barrier()

    scores: List[float] = []
    for knobs in candidates:
        # A candidate failure (config reject / compile error) is deterministic
        # across ranks — same static problem, same knobs — so scoring it inf
        # keeps the collective iteration aligned.
        trial = None
        try:
            trial_cfg = dataclasses.replace(cfg, **knobs)
            trial = Sm107BlockScaledSymmBuffer(trial_cfg)
            # Staged payload shapes depend only on the problem geometry (and
            # the padding knobs, which the sweep does not vary).
            trial.x.copy_(symm_buffer.x)
            trial.x_sf.copy_(symm_buffer.x_sf)
            trial.topk_idx.copy_(symm_buffer.topk_idx)
            trial.topk_weights.copy_(symm_buffer.topk_weights)
            trial.note_staged_tokens(num_tokens)
            _barrier()
            for _ in range(warmup_iters):  # first launch compiles
                sm107_block_scaled_mega_moe(
                    y,
                    transformed_l1,
                    transformed_l2,
                    trial,
                    num_tokens=num_tokens,
                    sync=True,
                )
            _barrier()
            iters: List[float] = []
            for _ in range(timed_iters):
                t0 = time.perf_counter()
                sm107_block_scaled_mega_moe(
                    y,
                    transformed_l1,
                    transformed_l2,
                    trial,
                    num_tokens=num_tokens,
                    sync=True,
                )
                iters.append(time.perf_counter() - t0)
            scores.append(statistics.median(iters))
        except Exception as exc:  # noqa: BLE001 -- score-and-continue by design
            warnings.warn(
                f"[sm107-autotune] candidate {knobs} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            scores.append(math.inf)
        finally:
            if trial is not None:
                trial.destroy()
        _barrier()

    t = torch.tensor(scores, dtype=torch.float64, device="cuda")
    if collective:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank = real latency
    best = int(torch.argmin(t).item())
    if not math.isfinite(float(t[best])):
        raise RuntimeError("[sm107-autotune] every candidate failed to compile/run.")
    winner = candidates[best]

    if rank == 0:
        from .knob_cache import record_knobs

        record_knobs(
            winner,
            dtype=cfg.quant_kind,
            world_size=cfg.world_size,
            hidden=cfg.hidden,
            intermediate=cfg.intermediate,
            num_experts=cfg.num_total_experts,
            topk=cfg.num_topk,
            max_tokens=cfg.max_tokens_per_rank,
            p50_us=float(t[best]) * 1e6,
            source="autotune",
        )
        ranked = sorted(zip(t.tolist(), candidates, strict=False), key=lambda kv: kv[0])
        summary = "\n".join(f"    {s * 1e6:10.1f} us  {k}" for s, k in ranked)
        print(
            f"[sm107-autotune] winner {winner} "
            f"({float(t[best]) * 1e6:.1f} us median, max across ranks) "
            f"out of {len(candidates)} candidates:\n{summary}",
            flush=True,
        )
    return winner


__all__ = [
    "KNOB_KEYS",
    "autotune_sm107_block_scaled_mega_moe",
    "is_valid_sm107",
    "sm107_candidates",
    "sm107_schedule_candidates",
]
