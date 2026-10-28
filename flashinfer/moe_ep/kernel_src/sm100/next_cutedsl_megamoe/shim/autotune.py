# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective warm CUDA-graph tuning for Blackwell block-scaled kernels.

Each trial owns a separate workspace. Online selection installs and prepares
the winner in the caller's workspace; offline tuning only returns the knobs.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

from .tuning import sm100_genphase_candidates


def _sweep_sm100_mega_moe(
    y,
    transformed_l1,
    transformed_l2,
    symm_buffer,
    *,
    num_tokens=None,
    candidates=None,
    constraints=None,
    warmup_iters=8,
    timed_iters=32,
    calls_per_sample=8,
    process_group=None,
    measurement=None,
):
    """Check sampled Torch correctness and minimize median(max EP rank).

    All EP ranks must enter together outside capture. A CUDA/compile failure
    aborts the supervised worker job; collective cleanup after a rank-local
    GPU failure could deadlock peers. Timing excludes staging and weight prep.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.core.tuning import tuning_live_tokens
    from .comm import ensure_not_capturing
    from .correctness import output_error, sampled_reference
    from flashinfer.moe_ep.core.tuning_measurement import (
        CaptureUnsupported,
        measure_samples,
    )
    from .knob_cache import device_identity

    ensure_not_capturing("Blackwell collective autotune")
    cfg = symm_buffer.config
    workspace_type = type(symm_buffer)
    from .tuning import knob_keys, is_valid_sm100_knobs

    collective = cfg.world_size > 1
    if collective and (
        not dist.is_initialized()
        or dist.get_world_size(process_group) != cfg.world_size
        or dist.get_rank(process_group) != cfg.rank
    ):
        raise ValueError("autotune process_group must match the workspace EP group")
    num_tokens = symm_buffer.staged_tokens() if num_tokens is None else num_tokens
    valid = (
        type(num_tokens) is int
        and 0 <= num_tokens <= cfg.max_tokens_per_rank
        and all(
            type(n) is int and n > 0
            for n in (warmup_iters, timed_iters, calls_per_sample)
        )
    )
    hardware = device_identity()
    spec = dataclasses.asdict(cfg)
    spec.pop("rank")
    proposal = (
        spec,
        candidates,
        constraints,
        warmup_iters,
        timed_iters,
        calls_per_sample,
        hardware,
        valid,
    )
    proposals = [proposal]
    if collective:
        proposals = [None] * cfg.world_size
        dist.all_gather_object(proposals, proposal, group=process_group)
    if not valid or any(p != proposal for p in proposals):
        raise ValueError(
            "all EP ranks need matching autotune specifications and valid token/iteration counts"
        )
    live_tokens = tuning_live_tokens(num_tokens, cfg.max_tokens_per_rank, process_group)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    candidates = (
        sm100_genphase_candidates(
            cfg, sm_count=sm_count, live_tokens=live_tokens, constraints=constraints
        )
        if candidates is None
        else candidates
    )
    if not candidates:
        raise ValueError("autotune needs at least one candidate")
    # Partial overrides are valid constraints, but a recorded winner must be
    # complete. Reject on every rank before rank 0 can fail during publication.
    if any(set(knobs) != set(knob_keys(cfg)) for knobs in candidates):
        raise ValueError("autotune candidates must set every Blackwell knob")

    def barrier():
        """Keep candidate lifetimes and cache publication ordered across EP ranks."""
        if collective:
            dist.barrier(group=process_group)

    checked_tokens = num_tokens
    live_count = getattr(symm_buffer, "num_valid_tokens", None)
    if live_count is not None:
        checked_tokens = int(live_count.item())
        valid = 0 <= checked_tokens <= num_tokens
        valid_counts = [valid]
        if collective:
            valid_counts = [None] * cfg.world_size
            dist.all_gather_object(valid_counts, valid, group=process_group)
        if not all(valid_counts):
            raise ValueError("persistent live counts must be within staged tokens")
    indices, expected = sampled_reference(
        symm_buffer,
        transformed_l1,
        transformed_l2,
        checked_tokens,
        process_group=process_group,
    )

    def error(trial):
        """Return the maximum sampled output error across the EP group."""
        output = trial.output_activation[:checked_tokens]
        if getattr(cfg, "use_custom_finalize", False):
            partials = trial.custom_finalize_workspace()[:checked_tokens].float()
            valid = (trial.topk_idx[:checked_tokens] >= 0) & (
                trial.topk_idx[:checked_tokens] < cfg.num_total_experts
            )
            partials = partials.masked_fill(~valid.unsqueeze(-1), 0)
            if not cfg.apply_routing_weights_before_fc2:
                weights = trial.topk_weights[:checked_tokens].masked_fill(~valid, 0)
                partials = partials * weights.unsqueeze(-1)
            output = partials.sum(dim=1).to(torch.bfloat16)
        value = torch.tensor(
            output_error(output, indices, expected),
            dtype=torch.float64,
            device="cuda",
        )
        if collective:
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=process_group)
        return float(value)

    scores, errors = [], []
    for knobs in candidates:
        if not is_valid_sm100_knobs(knobs, cfg):
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        trial_cfg = dataclasses.replace(cfg, **knobs)
        reason = None
        try:
            workspace_type._build_kernel(trial_cfg)
        except (ValueError, NotImplementedError) as exc:
            reason = str(exc)
        reasons = [reason]
        if collective:
            reasons = [None] * cfg.world_size
            dist.all_gather_object(reasons, reason, group=process_group)
        if any(reasons):
            if cfg.rank == 0:
                print(f"[megamoe-autotune] rejected {knobs}: {reasons}", flush=True)
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        trial = workspace_type(
            trial_cfg,
            **(
                {"num_valid_tokens_tensor": live_count}
                if live_count is not None
                else {}
            ),
        )
        for name in ("x", "x_sf", "topk_idx", "topk_weights"):
            getattr(trial, name).view(torch.uint8).copy_(
                getattr(symm_buffer, name).view(torch.uint8)
            )
        for name in (
            "fc1_alpha",
            "fc2_alpha",
            "fc1_norm_const",
            "fc1_activation_per_token_scale",
        ):
            source = getattr(symm_buffer, name, None)
            if source is not None:
                getattr(trial, name).copy_(source)
        trial.note_staged_tokens(num_tokens)
        _prepare_collectively(
            lambda: trial.prepare(transformed_l1, transformed_l2),
            cfg.world_size,
            process_group,
        )
        barrier()
        trial.launch(transformed_l1, transformed_l2)
        torch.cuda.synchronize()
        measured_error = error(trial)
        score = math.inf
        if measured_error <= 0.02:
            settings = measurement or {
                "timer": "native_events",
                "cold_l2": False,
                "cuda_graph_profile_replays": 1,
                "calls_per_sample": calls_per_sample,
                "warmup_iters": warmup_iters,
            }
            if settings["cuda_graph_profile_replays"] is None:
                for _ in range(warmup_iters):
                    trial.launch(transformed_l1, transformed_l2)
                torch.cuda.synchronize()
            try:
                samples = measure_samples(
                    trial,
                    (transformed_l1, transformed_l2),
                    timed_iters,
                    settings,
                    process_group,
                )
            except CaptureUnsupported:
                trial.destroy()
                raise
            # A bad clock sample on one rank must invalidate the candidate,
            # even if another rank has a positive elapsed duration.
            samples = [t if math.isfinite(t) and t > 0 else math.inf for t in samples]
            times = torch.tensor(samples, dtype=torch.float64, device="cuda")
            if collective:
                dist.all_reduce(times, op=dist.ReduceOp.MAX, group=process_group)
            measured_error = error(trial)
            values = times.tolist()
            if measured_error <= 0.02 and all(math.isfinite(t) for t in values):
                score = statistics.median(values)
        scores.append(score)
        errors.append(measured_error)
        trial.destroy()
        barrier()
        if cfg.rank == 0:
            print(
                f"[megamoe-autotune] {score:.3f} us p50(max rank), rel_l2={measured_error:.5f}: {knobs}",
                flush=True,
            )
    best = min(range(len(scores)), key=scores.__getitem__)
    if not math.isfinite(scores[best]):
        raise RuntimeError("no Blackwell candidate passed correctness and admission")
    winner = dict(candidates[best])
    return winner, scores[best], sum(math.isfinite(score) for score in scores)


def _prepare_collectively(prepare, world_size, process_group):
    error = None
    try:
        prepare()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if world_size > 1:
        import torch.distributed as dist

        errors = [None] * world_size
        dist.all_gather_object(errors, error, group=process_group)
    if any(errors):
        raise RuntimeError(f"MegaMoE preparation failed on EP ranks: {errors}")


def autotune_sm100_block_scaled_mega_moe(
    y,
    transformed_l1,
    transformed_l2,
    symm_buffer,
    *,
    num_tokens=None,
    candidates=None,
    constraints=None,
    warmup_iters=8,
    timed_iters=32,
    calls_per_sample=8,
    process_group=None,
    force_tune=True,
    install_winner=False,
    allow_nondeterministic=None,
):
    """Select native Blackwell knobs through managed autotune v2.

    Explicit calls search afresh. Online selection honors the shared context
    and prepares the winner only when install_winner=True. GenPhase accepts
    geometry/hint constraints; standard callers can supply complete candidates.
    """
    from flashinfer import autotune_v2
    from flashinfer.autotuner import AutoTuner
    from .tuning_protocol import NAMESPACE, BlackwellTuningProtocol

    protocol = BlackwellTuningProtocol(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        candidates=candidates,
        constraints=constraints,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        calls_per_sample=calls_per_sample,
        process_group=process_group,
        install_winner=install_winner,
        allow_nondeterministic=allow_nondeterministic,
    )
    tuner = AutoTuner.get()
    if force_tune and not tuner._in_v2_context:
        store = tuner._active_managed_store
        with autotune_v2(
            cache_root=None if store is None else store.root,
            measurement_policy=tuner.protocol_context(NAMESPACE).measurement_policy,
        ):
            return tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
    return tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=force_tune)


# Retain the existing explicit GenPhase tuning entry point.
autotune_sm100_genphase_mega_moe = autotune_sm100_block_scaled_mega_moe
