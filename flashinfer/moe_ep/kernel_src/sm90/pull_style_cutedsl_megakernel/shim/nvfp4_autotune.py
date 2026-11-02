# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective NV-only adapter over the existing lockstep MegaMoE tuner."""

from __future__ import annotations
from typing import Any


def autotune_hopper_nvfp4_mega_moe(
    y: Any,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: Any,
    *,
    num_tokens: int | None = None,
    gate_up_clamp: float | None = None,
    activation_clamp: float | None = None,
    candidates: list[dict] | None = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    process_group: Any = None,
) -> dict:
    """Measure the finite NV union collectively, never import MX scores.

    The prepared weight K permutation and explicit FC1 overlap override stay
    fixed for the entire session. Conversion/layout fields are complete in
    each candidate and in the persisted winner.
    """
    from .autotune import _autotune_hopper_mega_moe
    from .hopper_nvfp4 import (
        _NVFP4_TUNING_DTYPE_ID,
        _build_nvfp4_inputs,
        hopper_nvfp4_mega_moe,
    )
    from .mxfp4_tuner import require_hopper_mxfp4_fused_tuning_device
    from .nvfp4_tuner import (
        hopper_nvfp4_candidates,
        normalize_hopper_nvfp4_tactic,
        nvfp4_candidate_sha256,
        hopper_nvfp4_cache_provenance_sha256,
    )

    cfg = symm_buffer._frontend.config
    require_hopper_mxfp4_fused_tuning_device()
    full = hopper_nvfp4_candidates(
        cfg.num_tokens_per_rank,
        hidden=cfg.hidden,
        intermediate=cfg.intermediate,
        num_experts=cfg.num_total_experts,
        world_size=cfg.world_size,
        routing_profile=cfg.routing_profile,
        kperm=cfg.nvfp4_kperm,
    )
    if cfg.nvfp4_fc1_k256_overlap is True:
        full = [c for c in full if c["mma_tiler_mnk"][2] == 256 and not c["pingpong"]]
    selected = (
        full
        if candidates is None
        else [normalize_hopper_nvfp4_tactic(c) for c in candidates]
    )
    if not selected or any(c not in full for c in selected):
        raise ValueError(
            "NVFP4 autotune candidates must belong to the live NV candidate union"
        )
    if any(c in selected[:i] for i, c in enumerate(selected)):
        raise ValueError("NVFP4 autotune candidates must be unique")
    union_sha = nvfp4_candidate_sha256(selected)

    def record(_winner: dict, p50_s: float) -> None:
        requested = normalize_hopper_nvfp4_tactic(_winner)
        if requested not in selected:
            raise ValueError("NVFP4 autotune winner is outside the measured candidates")
        winner = normalize_hopper_nvfp4_tactic(symm_buffer._frontend.effective_tactic())
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            actual = symm_buffer._frontend.config
            record_knobs(
                winner,
                dtype=_NVFP4_TUNING_DTYPE_ID,
                fp8_scale_mode="mxfp4_hybrid",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                live_tokens=num_tokens
                if num_tokens is not None
                else cfg.num_tokens_per_rank,
                gate_up_clamp=actual.gate_up_clamp,
                routing_profile=cfg.routing_profile,
                tuning_provenance_sha256=hopper_nvfp4_cache_provenance_sha256(
                    routing_profile=cfg.routing_profile,
                    kperm=cfg.nvfp4_kperm,
                    overlap=cfg.nvfp4_fc1_k256_overlap,
                    max_tokens=cfg.num_tokens_per_rank,
                    hidden=cfg.hidden,
                    intermediate=cfg.intermediate,
                    num_experts=cfg.num_total_experts,
                    world_size=cfg.world_size,
                ),
                p50_us=p50_s * 1e6,
                source=f"autotune:sm90_nvfp4_fused:{union_sha}",
            )

    return _autotune_hopper_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        build_inputs=_build_nvfp4_inputs,
        launch_kernel=hopper_nvfp4_mega_moe,
        candidates=selected,
        on_winner=record,
        label="sm90_nvfp4_fused_mega",
        num_tokens=num_tokens,
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        process_group=process_group,
    )
