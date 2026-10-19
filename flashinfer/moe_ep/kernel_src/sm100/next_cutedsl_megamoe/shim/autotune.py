# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective warm CUDA-graph tuning for Blackwell GenPhase.

Each trial owns a separate workspace. The caller's workspace is unchanged;
its backend may apply the returned knobs before its first captured launch.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

from .tuning import KNOB_KEYS, is_valid_sm100_genphase_knobs, sm100_genphase_candidates


def autotune_sm100_genphase_mega_moe(
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
):
    """Check sampled Torch correctness, minimize median(max EP rank), and cache.

    All EP ranks must enter together outside capture. A CUDA/compile failure
    aborts the supervised worker job; collective cleanup after a rank-local
    GPU failure could deadlock peers. Timing excludes staging and weight prep.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.core.tuning import tuning_live_tokens
    from .comm import ensure_not_capturing
    from .correctness import output_error, sampled_reference
    from .genphase import Sm100GenphaseSymmBuffer
    from .knob_cache import device_identity, record_knobs

    ensure_not_capturing("Blackwell GenPhase collective autotune")
    cfg = symm_buffer.config
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
    if any(set(knobs) != set(KNOB_KEYS) for knobs in candidates):
        raise ValueError("autotune candidates must set every GenPhase knob")

    def barrier():
        """Keep candidate lifetimes and cache publication ordered across EP ranks."""
        if collective:
            dist.barrier(group=process_group)

    indices, expected = sampled_reference(
        symm_buffer,
        transformed_l1,
        transformed_l2,
        num_tokens,
        process_group=process_group,
    )

    def error(trial):
        """Return the maximum sampled output error across the EP group."""
        value = torch.tensor(
            output_error(trial.output_activation[:num_tokens], indices, expected),
            dtype=torch.float64,
            device="cuda",
        )
        if collective:
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=process_group)
        return float(value)

    scores, errors = [], []
    for knobs in candidates:
        if not is_valid_sm100_genphase_knobs(knobs, cfg):
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        trial_cfg = dataclasses.replace(cfg, **knobs)
        reason = None
        try:
            Sm100GenphaseSymmBuffer._build_kernel(trial_cfg)
        except (ValueError, NotImplementedError) as exc:
            reason = str(exc)
        reasons = [reason]
        if collective:
            reasons = [None] * cfg.world_size
            dist.all_gather_object(reasons, reason, group=process_group)
        if any(reasons):
            if cfg.rank == 0:
                print(f"[genphase-autotune] rejected {knobs}: {reasons}", flush=True)
            scores.append(math.inf)
            errors.append(math.inf)
            continue
        trial = Sm100GenphaseSymmBuffer(trial_cfg)
        for name in ("x", "x_sf", "topk_idx", "topk_weights"):
            getattr(trial, name).view(torch.uint8).copy_(
                getattr(symm_buffer, name).view(torch.uint8)
            )
        trial.note_staged_tokens(num_tokens)
        barrier()
        trial.launch(transformed_l1, transformed_l2)
        torch.cuda.synchronize()
        measured_error = error(trial)
        score = math.inf
        if measured_error <= 0.02:
            graph = torch.cuda.CUDAGraph()
            barrier()
            with torch.cuda.graph(graph):
                for _ in range(calls_per_sample):
                    trial.launch(transformed_l1, transformed_l2)
            for _ in range(warmup_iters):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(timed_iters):
                barrier()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / calls_per_sample)
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
            # Drop captured pointer ownership before releasing symmetric storage.
            del graph
        scores.append(score)
        errors.append(measured_error)
        trial.destroy()
        barrier()
        if cfg.rank == 0:
            print(
                f"[genphase-autotune] {score:.3f} us p50(max rank), rel_l2={measured_error:.5f}: {knobs}",
                flush=True,
            )
    best = min(range(len(scores)), key=scores.__getitem__)
    if not math.isfinite(scores[best]):
        raise RuntimeError(
            "no Blackwell GenPhase candidate passed correctness and admission"
        )
    winner = dict(candidates[best])
    if cfg.rank == 0:
        record_knobs(
            winner, cfg, live_tokens=live_tokens, p50_us=scores[best], device=hardware
        )
        print(f"[genphase-autotune] winner {winner}", flush=True)
    # Every rank may construct a cache-backed layer immediately after warmup.
    barrier()
    return winner
