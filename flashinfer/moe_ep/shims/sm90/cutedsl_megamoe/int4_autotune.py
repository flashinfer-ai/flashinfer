# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective INT4 tuning over the shared Hopper compile/timing lifecycle."""

from __future__ import annotations

from typing import Any


def autotune_hopper_int4_mega_moe(
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
    """Measure INT4 tactics on every EP rank, apply and cache the winner."""
    from .autotune import _autotune_hopper_mega_moe
    from .hopper_int4 import (
        _INT4_TUNING_DTYPE_ID,
        _build_int4_inputs,
        hopper_int4_mega_moe,
    )
    from .int4_tuner import (
        hopper_int4_cache_provenance_sha256,
        hopper_int4_candidates,
        int4_candidate_sha256,
        normalize_hopper_int4_tactic,
    )
    from .mxfp4_tuner import require_hopper_mxfp4_fused_tuning_device

    cfg = symm_buffer._frontend.config
    require_hopper_mxfp4_fused_tuning_device()
    domain = dict(
        max_tokens=cfg.num_tokens_per_rank,
        hidden=cfg.hidden,
        intermediate=cfg.intermediate,
        num_experts=cfg.num_total_experts,
        world_size=cfg.world_size,
        routing_profile=cfg.routing_profile,
    )
    full = hopper_int4_candidates(**domain)
    selected = (
        full
        if candidates is None
        else [normalize_hopper_int4_tactic(c) for c in candidates]
    )
    if not selected or any(c not in full for c in selected):
        raise ValueError("INT4 autotune candidates must belong to the live INT4 domain")
    if any(c in selected[:i] for i, c in enumerate(selected)):
        raise ValueError("INT4 autotune candidates must be unique")

    def record(_winner: dict, p50_s: float) -> None:
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            frontend = symm_buffer._frontend
            record_knobs(
                frontend.effective_tactic(),
                dtype=_INT4_TUNING_DTYPE_ID,
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
                gate_up_clamp=frontend.config.gate_up_clamp,
                routing_profile=cfg.routing_profile,
                tuning_provenance_sha256=hopper_int4_cache_provenance_sha256(**domain),
                p50_us=p50_s * 1e6,
                source=f"autotune:sm90_int4_fused:{int4_candidate_sha256(selected)}",
            )

    return _autotune_hopper_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        build_inputs=_build_int4_inputs,
        launch_kernel=hopper_int4_mega_moe,
        candidates=selected,
        on_winner=record,
        label="sm90_int4_fused_mega",
        num_tokens=num_tokens,
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        process_group=process_group,
    )
