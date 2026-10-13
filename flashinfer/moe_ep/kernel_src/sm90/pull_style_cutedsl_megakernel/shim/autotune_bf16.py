# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Online (warmup-time) knob autotuning for the SM90 Hopper BF16 MegaMoE frontend.

Twin of :mod:`.autotune` (FP8): the candidate set is the BF16 heuristic
table's per-bucket winners plus every geometry that wins a neighbouring
bucket, timed in lockstep on all EP ranks by the shared
:func:`.autotune.autotune_knobs` collective (MAX all-reduce of per-candidate
medians, argmin identical everywhere).  Winners persist in the knob cache
under the BF16 key (:mod:`.tuner_bf16`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .autotune import autotune_knobs
from .tuner_bf16 import default_knobs, is_valid


def _sweep_geometries() -> List[Dict[str, Any]]:
    """Every geometry that wins at least one bucket of the kernel drop's
    four-rank sweep (``heuristic_config.HEURISTIC_CONFIGS``), deduplicated.
    Derived from the table so a table refresh updates the candidates."""
    from moe_hopper_bf16.heuristic_config import HEURISTIC_CONFIGS

    out: List[Dict[str, Any]] = []
    seen = set()
    for c in HEURISTIC_CONFIGS.values():
        # The scheduler companions (group_hint / tail_split_pairs) were tuned
        # with the geometry, so they travel with it.
        key = (
            c.swap_ab,
            c.pingpong,
            c.mma_tiler_mnk,
            c.cluster_shape_mnk,
            c.group_hint,
            c.tail_split_pairs,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "swap_ab": c.swap_ab,
                "pingpong": c.pingpong,
                "mma_tiler_mnk": tuple(c.mma_tiler_mnk),
                "cluster_shape_mnk": tuple(c.cluster_shape_mnk),
                "group_hint": c.group_hint,
                "tail_split_pairs": c.tail_split_pairs,
            }
        )
    return out


def hopper_bf16_candidates(*, max_tokens: int = 0) -> List[Dict[str, Any]]:
    """Default candidate knob dicts: heuristic winner first, then every
    geometry that wins some bucket of the drop's sweep, each under both
    validated token-back placements.  The heuristic winner leads so a tie
    keeps the established default."""
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

    _add(default_knobs(max_tokens))
    for geometry in _sweep_geometries():
        for token_back in ("epi_warps", "reuse_dispatch_warps"):
            _add({**geometry, "token_back_mode": token_back})
    return out


def autotune_hopper_bf16_mega_moe(
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
    """Autotune the SM90 BF16 mega session on the caller's staged inputs.

    Arguments mirror :func:`.hopper_bf16.hopper_bf16_mega_moe`; ``y`` is
    clobbered by the candidate launches.  Applies the winner and returns its
    knob dict; subsequent ``hopper_bf16_mega_moe`` calls on ``symm_buffer``
    reuse the winning compile.  COLLECTIVE -- see :func:`.autotune.autotune_knobs`.
    """
    from .hopper_bf16 import hopper_bf16_mega_moe

    def launch() -> None:
        # sync=True: the tune loop times launches with perf_counter, so the
        # call must block until the kernel (and topk reduce) complete.
        hopper_bf16_mega_moe(
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
        candidates = hopper_bf16_candidates(max_tokens=cfg.num_tokens_per_rank)

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Persist for future pure-lookup engine starts; rank 0 writes (the
        # winner is identical on all ranks after the all_reduce).
        if cfg.rank == 0:
            from .tuner_bf16 import record_knobs

            record_knobs(
                winner,
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
        label="sm90_bf16_mega",
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        on_winner=_record,
    )


__all__ = [
    "autotune_hopper_bf16_mega_moe",
    "hopper_bf16_candidates",
]
