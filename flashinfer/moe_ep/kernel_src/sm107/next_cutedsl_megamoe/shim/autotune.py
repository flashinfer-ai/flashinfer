# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective tuning for the SM107 block-scaled kernel.

Each candidate builds a session, copies the staged inputs, compiles and times
its kernel, then destroys the session. Offline tuning leaves the caller's
workspace unchanged; online tuning installs and prepares the winner before execution.

All EP ranks must use the same candidates in the same order. Timing takes the
maximum rank duration for each sample, then the median across samples.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional


class NoPassingCandidate(RuntimeError):
    """All candidates were rejected after agreement across EP ranks."""


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

# Structural choices before expanding the optional in-kernel reduction axis.
_INFERENCE_CANDIDATE_ORDER = (
    29,
    25,
    19,
    27,
    31,
    21,
    23,
    18,
    17,
    15,
    7,
    22,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
    20,
    24,
    26,
    28,
    30,
)

_GENPHASE_CANDIDATE_ORDER = (8, 3, 0, 1, 2, 4, 5, 6, 7, 9, 10)


def _unique_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tuple-valued knobs and retain each choice's first occurrence."""
    from .knob_cache import _knobs_from_json

    unique: Dict[tuple, Dict[str, Any]] = {}
    for knobs in candidates:
        normalized = _knobs_from_json(knobs)
        unique.setdefault(tuple(sorted(normalized.items())), normalized)
    return list(unique.values())


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
        if getattr(base_config, "batch_invariant", False):
            from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
                validate_block_scaled_batch_invariant,
            )

            validate_block_scaled_batch_invariant(base_config, knobs, arch="sm_107")
        dataclasses.replace(base_config, **knobs)
    except (ValueError, TypeError):
        return False
    return True


def sm107_candidates(
    quant_kind: str = "nvfp4",
    *,
    allow_in_kernel_fc2_reduce: bool = False,
    kernel_variant: str = "inference",
    batch_invariant: bool = False,
) -> List[Dict[str, Any]]:
    """Default SM107 candidate knob dicts (tile x launch x epi x fc2-bulk).

    Inference uses 32 candidates (64 with the ikr axis), ordered for callers
    taking a prefix. The grid includes:

    - tile N 128 vs 256 at the kind's base K depth,
    - uniform (2,1) grouped/grid-stride launch vs mixed-CGA (4,1)+(2,1)
      phase-interleave/atomic launch,
    - epi flag batches (1,4) vs (2,4),
    - FC2 bulk TMA (2 stages) on/off.

    Additional candidates use uniform clusters (2,1) or (4,1) with
    phase-interleave/atomic launches and epi flag batches (2,4): N64 at the
    kind's base and doubled K depths, plus N128/N256 at the base K depth.
    FC2 bulk stores use one stage for N64 and two for the wider tiles.

    Default GenPhase candidates use cluster (4, 1) and require FC2 bulk stores.
    MXFP8 E4M3 and mixed FP4/FP8 additionally include the N64 decode tile,
    using one FC2 stage and deterministic separate top-k reduction.

    ``token_in_flag_batch`` and the phase-interleave hint are skew-sensitive —
    sweep them with :func:`sm107_schedule_candidates` instead.  An ikr winner
    makes the output accumulation order nondeterministic; it stays opt-in.
    """
    if batch_invariant:
        if (
            quant_kind
            not in ("nvfp4", "mxfp4", "mxfp4_mxfp8", "mxfp8_e4m3", "mxfp8_e5m2")
            or allow_in_kernel_fc2_reduce
        ):
            raise ValueError(
                "batch_invariant SM107 candidates require an admitted block-scaled kind without in-kernel reduction"
            )
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
            block_scaled_batch_invariant_default_knobs,
        )

        from types import SimpleNamespace

        base = block_scaled_batch_invariant_default_knobs(
            SimpleNamespace(quant_kind=quant_kind),
            "genphase" if kernel_variant == "genphase" else "standard",
            arch="sm_107",
        )
        if kernel_variant == "genphase":
            return [
                dict(base, work_id_mode=work)
                for work in ("grid_stride", "atomic_counter")
            ]
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.quantized_batch_invariant_tiles import (
            quantized_batch_invariant_tiles,
        )

        return [
            dict(base, mma_tiler_mnk=tile, token_back_mode=carrier, work_id_mode=work)
            for tile in quantized_batch_invariant_tiles(quant_kind, arch="sm_107")
            for carrier in ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
            for work in ("grid_stride", "atomic_counter")
        ]
    tile_k = 256 if quant_kind in ("nvfp4", "mxfp4") else 128
    launches = (
        # (cluster, fallback, schedule_policy, work_id_mode)
        ((2, 1), None, ("grouped", None), "grid_stride"),
        ((4, 1), (2, 1), ("phase_interleave", None), "atomic_counter"),
    )
    if kernel_variant == "genphase":
        launches = (
            ((4, 1), None, ("grouped", None), "grid_stride"),
            ((4, 1), None, ("phase_interleave", None), "atomic_counter"),
        )
    elif kernel_variant != "inference":
        raise ValueError(f"unsupported kernel_variant {kernel_variant!r}")
    out: List[Dict[str, Any]] = []
    for tile_n in (128, 256):
        for cluster, fallback, schedule, work_id in launches:
            for epi in ((1, 4), (2, 4)):
                for fc2_bulk in (
                    (True,) if kernel_variant == "genphase" else (False, True)
                ):
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
    if kernel_variant == "inference":
        for tile_n, depth in (
            (64, tile_k),
            (64, 2 * tile_k),
            (128, tile_k),
            (256, tile_k),
        ):
            for cluster in ((2, 1), (4, 1)):
                for fc2_bulk in (False, True):
                    for ikr in (
                        (False, True) if allow_in_kernel_fc2_reduce else (False,)
                    ):
                        out.append(
                            {
                                "mma_tiler_mnk": (256, tile_n, depth),
                                "cluster_shape_mn": cluster,
                                "fallback_cluster_shape_mn": None,
                                "schedule_policy": ("phase_interleave", None),
                                "work_id_mode": "atomic_counter",
                                "fc2_use_bulk": fc2_bulk,
                                "fc2_tma_stages": (1 if tile_n == 64 else 2)
                                if fc2_bulk
                                else None,
                                "epi_flag_batches": (2, 4),
                                "token_in_flag_batch": 1,
                                "token_back_mode": "epi_warps",
                                "reduce_topk_in_kernel": ikr,
                            }
                        )
    if kernel_variant == "genphase" and quant_kind in (
        "mxfp8_e4m3",
        "mxfp4_mxfp8",
    ):
        # These are additional candidates, not a replacement for wider tiles:
        # routing, cache residency and token count change the best pipeline.
        for schedule, work_id in (
            (("grouped", None), "grid_stride"),
            (("grouped", None), "atomic_counter"),
            (("phase_interleave", None), "atomic_counter"),
        ):
            out.append(
                {
                    "mma_tiler_mnk": (256, 64, 128),
                    "cluster_shape_mn": (4, 1),
                    "fallback_cluster_shape_mn": None,
                    "schedule_policy": schedule,
                    "work_id_mode": work_id,
                    "fc2_use_bulk": True,
                    "fc2_tma_stages": 1,
                    "epi_flag_batches": (1, 4),
                    "token_in_flag_batch": 1,
                    "token_back_mode": "epi_warps",
                    "reduce_topk_in_kernel": False,
                }
            )
    if kernel_variant == "inference":
        # Keep each structural choice's reduction variants adjacent.
        stride = 2 if allow_in_kernel_fc2_reduce else 1
        out = [
            candidate
            for index in _INFERENCE_CANDIDATE_ORDER
            for candidate in out[index * stride : (index + 1) * stride]
        ]
    else:
        # Only the eight wider-tile choices expand the reduction axis.
        stride = 2 if allow_in_kernel_fc2_reduce else 1
        base_count = 8 * stride
        groups = [out[i : i + stride] for i in range(0, base_count, stride)]
        groups.extend([candidate] for candidate in out[base_count:])
        out = [
            candidate
            for index in _GENPHASE_CANDIDATE_ORDER
            if index < len(groups)
            for candidate in groups[index]
        ]
    return out


def sm107_schedule_candidates(
    base: Dict[str, Any], *, kernel_variant: str = "inference"
) -> List[Dict[str, Any]]:
    """Expand a base knob dict into the skew-sensitive schedule grid
    (phase-interleave hint x token-in flag batch)."""
    out: List[Dict[str, Any]] = []
    for hint in (None, 3, 4, 6):
        for tif in (1,) if kernel_variant == "genphase" else (1, 4):
            out.append(
                {
                    **base,
                    "schedule_policy": ("phase_interleave", hint),
                    "work_id_mode": "atomic_counter",
                    "token_in_flag_batch": tif,
                }
            )
    return out


def _prepare_collectively(prepare, world_size, process_group):
    """Finish host preparation on every rank before launching collective kernels."""
    result, error = None, None
    try:
        result = prepare()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if world_size > 1:
        import torch.distributed as dist

        errors = [None] * world_size
        dist.all_gather_object(errors, error, group=process_group)
    if any(errors):
        result = None
        raise RuntimeError(f"MegaMoE preparation failed on EP ranks: {errors}")
    return result


def _sweep_sm107_block_scaled_mega_moe(
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
    measurement=None,
    make_trial=None,
) -> tuple[Dict[str, Any], float, int]:
    """Validate candidates against a sampled Torch oracle and time survivors.

    Every EP rank must participate. GPU allocation, compile, or launch errors
    abort the sweep: run distributed tuning under a worker supervisor with
    a timeout. Continuing after a rank-local CUDA fault is unsafe.

    ``make_trial`` can wrap each prepared workspace with a ``launch(l1, l2)``
    that includes input staging and output handling. The same launch is used
    for warmup, measurement and correctness checks.
    """
    import dataclasses

    import torch
    import torch.distributed as dist

    from .block_scaled import Sm107BlockScaledSymmBuffer, sm107_block_scaled_mega_moe
    from .comm import ensure_not_capturing
    from .correctness import output_error, sampled_reference
    from .tuning_measurement import CaptureUnsupported, measure_samples

    ensure_not_capturing("SM107 collective autotune sweep")
    cfg = symm_buffer.config
    candidates = _unique_candidates(
        sm107_candidates(
            cfg.quant_kind,
            kernel_variant=cfg.kernel_variant,
            batch_invariant=getattr(cfg, "batch_invariant", False),
        )
        if candidates is None
        else candidates
    )
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
        proposal = (spec, candidates, warmup_iters, timed_iters, make_trial is not None)
        proposals = [None] * cfg.world_size
        dist.all_gather_object(proposals, proposal, group=process_group)
        if any(p != proposal for p in proposals):
            raise ValueError(
                "all EP ranks must use identical autotune geometry, candidates, and iteration counts"
            )

    if y is None and make_trial is None:
        y = _prepare_collectively(
            lambda: torch.empty(
                num_tokens, cfg.hidden, dtype=torch.bfloat16, device=symm_buffer.device
            ),
            cfg.world_size,
            process_group,
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
    tolerance = 0.06 if cfg.quant_kind in ("nvfp4", "mxfp4") else 0.02
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
        trial = Sm107BlockScaledSymmBuffer(
            trial_cfg,
            **({"process_group": process_group} if process_group is not None else {}),
        )

        def prepare_trial():
            trial.x.view(torch.uint8).copy_(symm_buffer.x.view(torch.uint8))
            trial.x_sf.view(torch.uint8).copy_(symm_buffer.x_sf.view(torch.uint8))
            trial.topk_idx.copy_(symm_buffer.topk_idx)
            trial.topk_weights.copy_(symm_buffer.topk_weights)
            if cfg.quant_kind == "nvfp4":
                for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
                    # Copy the actual bound per-call scales, including defaults.
                    getattr(trial, name).copy_(symm_buffer._scalar_tensor(name))
            trial.note_staged_tokens(num_tokens)
            trial.prepare(transformed_l1, transformed_l2)
            return trial if make_trial is None else make_trial(trial)

        measured_trial = _prepare_collectively(
            prepare_trial,
            cfg.world_size,
            process_group,
        )
        barrier()
        for _ in range(warmup_iters):
            if make_trial is None:
                sm107_block_scaled_mega_moe(
                    y, transformed_l1, transformed_l2, trial, num_tokens=num_tokens
                )
            else:
                measured_trial.launch(transformed_l1, transformed_l2)
        torch.cuda.synchronize()
        actual = y if y is not None else trial.output_activation[:num_tokens]
        error = torch.tensor(
            output_error(actual, indices, expected), dtype=torch.float64, device="cuda"
        )
        if collective:
            dist.all_reduce(error, op=dist.ReduceOp.MAX, group=process_group)
        if float(error) > tolerance:
            scores.append(math.inf)
            errors.append(float(error))
            trial.destroy()
            barrier()
            continue

        try:
            samples = measure_samples(
                measured_trial,
                (transformed_l1, transformed_l2),
                timed_iters,
                measurement
                or {
                    "cuda_graph_profile_replays": None,
                    "cold_l2": False,
                    "timer": "native_events",
                },
                process_group,
            )
        except CaptureUnsupported as exc:
            if cfg.rank == 0:
                print(f"[sm107-autotune] rejected {knobs}: {exc}", flush=True)
            scores.append(math.inf)
            errors.append(math.inf)
            trial.destroy()
            barrier()
            continue
        times = torch.tensor(samples, dtype=torch.float64, device="cuda")
        if collective:
            # Reduce each matched sample before taking its median.
            dist.all_reduce(times, op=dist.ReduceOp.MAX, group=process_group)
        actual = (
            y
            if make_trial is not None and y is not None
            else trial.output_activation[:num_tokens]
        )
        error = torch.tensor(
            output_error(actual, indices, expected),
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
        raise NoPassingCandidate(
            "no SM107 candidate passed configuration and sampled correctness checks"
        )
    winner = candidates[best]
    if cfg.rank == 0:
        for score, error, knobs in zip(scores, errors, candidates, strict=False):
            print(
                f"[sm107-autotune] {score:.2f} us p50(max rank), rel_l2={error:.5f}: {knobs}",
                flush=True,
            )
        print(f"[sm107-autotune] winner {winner}", flush=True)
    return winner, scores[best], sum(math.isfinite(s) for s in scores)


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
    force_tune: bool = True,
    install_winner: bool = False,
    allow_nondeterministic: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Measure already-staged kernels through FlashInfer's autotuner.

    Explicit calls search afresh. Set ``force_tune=False`` to honor the shared
    context and ``install_winner=True`` to prepare the winner in this workspace.
    A disabled cache miss returns None. Layer and CLI tuning measure complete
    forwards through the same protocol with a staging-aware execution trial.
    Supplied candidates retain their order; repeated choices are measured once.
    """
    from .tuning_protocol import RubinTuningProtocol

    protocol = RubinTuningProtocol(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        candidates=candidates,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        process_group=process_group,
        install_winner=install_winner,
        allow_nondeterministic=allow_nondeterministic,
    )
    return _choose_protocol(protocol, force_tune=force_tune)


def _choose_protocol(protocol, *, force_tune):
    from flashinfer import autotune_v2
    from flashinfer.autotuner import AutoTuner
    from .tuning_protocol import NAMESPACE

    tuner = AutoTuner.get()
    if force_tune and not tuner._in_v2_context:
        store = tuner._active_managed_store
        with autotune_v2(
            cache_root=None if store is None else store.root,
            measurement_policy=tuner.protocol_context(NAMESPACE).measurement_policy,
        ):
            return tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
    return tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=force_tune)


__all__ = [
    "KNOB_KEYS",
    "autotune_sm107_block_scaled_mega_moe",
    "is_valid_sm107",
    "sm107_candidates",
    "sm107_schedule_candidates",
]
