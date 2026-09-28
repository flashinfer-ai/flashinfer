# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective offline tuning for the SM107 block-scaled kernel.

Each candidate builds a session, copies the staged inputs, compiles and times
its kernel, then destroys the session. The caller's workspace stays unchanged;
the winner is saved in the knob cache for later backend construction.

All EP ranks must use the same candidates in the same order. Timing takes the
maximum rank duration for each sample, then the median across samples.
"""

from __future__ import annotations

import math
import statistics
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
    Rubin perf report showed to matter (see TUNING.md):

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
    process_group=None,
) -> Dict[str, Any]:
    """Validate candidates against a sampled Torch oracle, time, and cache.

    Every EP rank must participate. GPU allocation, compile, or launch errors
    abort the sweep: run distributed tuning under a worker supervisor with
    a timeout. Continuing after a rank-local CUDA fault is unsafe.
    """
    import dataclasses

    import torch
    import torch.distributed as dist

    from .block_scaled import Sm107BlockScaledSymmBuffer, sm107_block_scaled_mega_moe
    from .comm import ensure_not_capturing
    from .correctness import output_error, sampled_reference

    ensure_not_capturing("SM107 collective autotune sweep")
    cfg = symm_buffer.config
    candidates = sm107_candidates(cfg.quant_kind) if candidates is None else candidates
    if not candidates:
        raise ValueError("autotune needs a non-empty candidate list.")
    if warmup_iters < 1 or timed_iters < 1:
        raise ValueError("warmup_iters and timed_iters must both be positive.")
    num_tokens = symm_buffer.staged_tokens() if num_tokens is None else num_tokens
    if num_tokens is None or not 0 <= num_tokens <= cfg.max_tokens_per_rank:
        raise ValueError("stage a valid num_tokens before autotuning.")

    collective = cfg.world_size > 1
    if collective:
        if (
            not dist.is_initialized()
            or dist.get_world_size(process_group) != cfg.world_size
            or dist.get_rank(process_group) != cfg.rank
        ):
            raise ValueError(
                "autotune process_group must match the workspace EP rank and world size"
            )
        spec = dataclasses.asdict(cfg)
        spec.pop("rank")
        proposal = (spec, candidates, warmup_iters, timed_iters)
        proposals = [None] * cfg.world_size
        dist.all_gather_object(proposals, proposal, group=process_group)
        if any(p != proposal for p in proposals):
            raise ValueError(
                "all EP ranks must use identical autotune geometry, candidates, and iteration counts"
            )

    def barrier():
        if collective:
            dist.barrier(group=process_group)

    # This oracle uses actual transformed and staged bytes, with one expert
    # and 64 sampled rows in FP32 at a time. It is outside the timed window.
    indices, expected = sampled_reference(
        symm_buffer,
        transformed_l1,
        transformed_l2,
        num_tokens,
        process_group=process_group,
    )
    tolerance = 0.06 if cfg.quant_kind == "nvfp4" else 0.02
    scores = []
    errors = []
    for knobs in candidates:
        if not is_valid_sm107(knobs, cfg):
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        trial_cfg = dataclasses.replace(cfg, **knobs)
        reason = None
        try:
            # Geometry/resource checks (including the hardware-dependent
            # minimum phase-interleave hint) precede symmetric allocation.
            Sm107BlockScaledSymmBuffer._build_kernel(trial_cfg)
        except (ValueError, NotImplementedError) as exc:
            reason = str(exc)
        reasons = [reason]
        if collective:
            reasons = [None] * cfg.world_size
            dist.all_gather_object(reasons, reason, group=process_group)
        if any(reasons):
            if cfg.rank == 0:
                print(f"[sm107-autotune] rejected {knobs}: {reasons}", flush=True)
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        # No finally/collective free on failure: another rank may already be
        # inside a kernel or have a failed CUDA context. The supervisor must
        # terminate the whole job, then retry a candidate in a fresh job.
        trial = Sm107BlockScaledSymmBuffer(trial_cfg)
        trial.x.view(torch.uint8).copy_(symm_buffer.x.view(torch.uint8))
        trial.x_sf.view(torch.uint8).copy_(symm_buffer.x_sf.view(torch.uint8))
        trial.topk_idx.copy_(symm_buffer.topk_idx)
        trial.topk_weights.copy_(symm_buffer.topk_weights)
        trial.note_staged_tokens(num_tokens)
        barrier()
        for _ in range(warmup_iters):
            sm107_block_scaled_mega_moe(
                y, transformed_l1, transformed_l2, trial, num_tokens=num_tokens
            )
        torch.cuda.synchronize()
        error = torch.tensor(
            output_error(y, indices, expected), dtype=torch.float64, device="cuda"
        )
        if collective:
            dist.all_reduce(error, op=dist.ReduceOp.MAX, group=process_group)
        if float(error) > tolerance:
            scores.append(math.inf)
            errors.append(float(error))
            trial.destroy()
            barrier()
            continue

        samples = []
        for _ in range(timed_iters):
            barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            trial.launch(transformed_l1, transformed_l2)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)
        times = torch.tensor(samples, dtype=torch.float64, device="cuda")
        if collective:
            # Reduce each matched sample before taking its median.
            dist.all_reduce(times, op=dist.ReduceOp.MAX, group=process_group)
        error = torch.tensor(
            output_error(trial.output_activation[:num_tokens], indices, expected),
            dtype=torch.float64,
            device="cuda",
        )
        if collective:
            dist.all_reduce(error, op=dist.ReduceOp.MAX, group=process_group)
        scores.append(
            statistics.median(times.tolist()) if float(error) <= tolerance else math.inf
        )
        errors.append(float(error))
        trial.destroy()
        barrier()

    best = min(range(len(scores)), key=scores.__getitem__)
    if not math.isfinite(scores[best]):
        raise RuntimeError(
            "no SM107 candidate passed configuration and sampled correctness checks"
        )
    winner = candidates[best]
    if cfg.rank == 0:
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
            p50_us=scores[best],
            source="autotune-cuda-events-sampled-oracle",
            allow_nondeterministic=any(
                k.get("reduce_topk_in_kernel", cfg.reduce_topk_in_kernel)
                for k in candidates
            ),
            apply_topk_at_fc1=cfg.apply_topk_at_fc1,
        )
        for score, error, knobs in zip(scores, errors, candidates, strict=False):
            print(
                f"[sm107-autotune] {score:.2f} us p50(max rank), rel_l2={error:.5f}: {knobs}",
                flush=True,
            )
        print(f"[sm107-autotune] winner {winner}", flush=True)
    return winner


__all__ = [
    "KNOB_KEYS",
    "autotune_sm107_block_scaled_mega_moe",
    "is_valid_sm107",
    "sm107_candidates",
    "sm107_schedule_candidates",
]
