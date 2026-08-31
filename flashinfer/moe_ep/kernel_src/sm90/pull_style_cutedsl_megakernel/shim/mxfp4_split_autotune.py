# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective autotuning for fixed-pointer Hopper MXFP4 Green split sessions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .autotune import autotune_knobs
from .hopper_mxfp4_split import (
    MegaMoEHopperMxfp4SplitSymmBuffer,
    _SPLIT_TUNING_IDENTITY,
)
from .mxfp4_tuner import (
    hopper_mxfp4_candidates,
    hopper_mxfp4_ordered_candidates,
    hopper_mxfp4_tuning_provenance,
    is_hopper_mxfp4_tactic_shape_compatible,
    require_hopper_mxfp4_tuning_device,
    validate_hopper_mxfp4_tactic,
)


class _SplitTacticAdapter:
    """Present fresh split sessions through autotune_knobs.apply_knobs."""

    def __init__(self, source: MegaMoEHopperMxfp4SplitSymmBuffer) -> None:
        if source._destroyed:
            raise RuntimeError("cannot autotune a destroyed split workspace")
        if source.session.captured:
            raise RuntimeError(
                "split collective autotune must run before the session is captured"
            )
        self._source = source
        self._current: Optional[MegaMoEHopperMxfp4SplitSymmBuffer] = None
        self._committed = False

    @property
    def current(self) -> MegaMoEHopperMxfp4SplitSymmBuffer:
        if self._current is None:
            raise RuntimeError("split autotune candidate is not allocated")
        return self._current

    def _destroy_current(self) -> None:
        if self._current is None:
            return
        current, self._current = self._current, None
        current.destroy()

    def apply_knobs(self, knobs: Dict[str, Any]) -> None:
        """Destroy the prior candidate and build one fresh fixed-pointer session."""

        tactic = validate_hopper_mxfp4_tactic(
            knobs,
            execution_mode="split",
        )
        self._destroy_current()
        source = self._source
        cfg = source.session.config

        from .hopper_mxfp4_split import (
            get_symm_buffer_for_hopper_mxfp4_split_mega_moe,
        )

        candidate: Optional[MegaMoEHopperMxfp4SplitSymmBuffer] = None
        try:
            candidate = get_symm_buffer_for_hopper_mxfp4_split_mega_moe(
                source.num_total_experts,
                source.num_max_tokens,
                source.num_topk,
                source.hidden,
                source.intermediate,
                source.rank,
                source.world_size,
                split_k1_mma_tiler_mnk=tactic["k1_mma_tiler_mnk"],
                split_k2_mma_tiler_mnk=tactic["k2_mma_tiler_mnk"],
                split_k1_cluster_shape_mnk=tactic["k1_cluster_shape_mnk"],
                split_k2_cluster_shape_mnk=tactic["k2_cluster_shape_mnk"],
                split_k1_group_hint=tactic["k1_group_hint"],
                split_k2_group_hint=tactic["k2_group_hint"],
                split_k1_num_sched_stages=tactic["k1_num_sched_stages"],
                split_k2_num_sched_stages=tactic["k2_num_sched_stages"],
                split_k1_sm_count=tactic["k1_sm_count"],
                split_k2_sm_count=tactic["k2_sm_count"],
                split_counter_epoch_banks=tactic["counter_epoch_banks"],
                split_graph_variant=tactic["graph_variant"],
                clc_bundle_size=cfg.clc_bundle_size,
                flag_batch=cfg.flag_batch,
                epi_flag_batch=cfg.epi_flag_batch,
                gate_up_clamp=cfg.gate_up_clamp,
                split_enable_iket=tactic["enable_iket"],
                process_group=source.session._process_group,
                routing_profile=cfg.routing_profile,
            )
            candidate.x.copy_(source.x)
            candidate.x_sf.copy_(source.x_sf)
            candidate.topk_idx.copy_(source.topk_idx)
            candidate.topk_weights.copy_(source.topk_weights)
            staged = getattr(source.topk_idx, "_sm90_mxfp4_staged_tokens", None)
            if staged is not None:
                candidate.topk_idx._sm90_mxfp4_staged_tokens = staged
            self._current = candidate
        except BaseException:
            if candidate is not None:
                candidate.destroy()
            raise

    def commit(self) -> None:
        """Transfer the measured winner's resources into the caller workspace."""

        winner = self.current
        source = self._source
        # The last source->winner staging copies must complete before freeing
        # the source symmetric allocations.
        torch.cuda.synchronize()
        source.destroy()

        source.x = winner.x
        source.x_sf = winner.x_sf
        source.topk_idx = winner.topk_idx
        source.topk_weights = winner.topk_weights
        source.output_activation = winner.output_activation
        source._session = winner._session
        source._sym_roots = winner._sym_roots
        source._destroyed = False

        # Disarm the temporary owner without destroying the transferred
        # session or symmetric roots. The original workspace owns them now.
        winner._sym_roots = []
        winner._destroyed = True
        self._current = None
        self._committed = True

    def close(self) -> None:
        if not self._committed:
            self._destroy_current()


def autotune_hopper_mxfp4_split_mega_moe(
    y: torch.Tensor,
    transformed_l1: Any,
    transformed_l2: Any,
    symm_buffer: MegaMoEHopperMxfp4SplitSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    warmup_iters: int = 3,
    timed_iters: int = 10,
    process_group: Any = None,
) -> Dict[str, Any]:
    """Tune the compact split union using fresh fixed-pointer sessions.

    Each candidate owns fresh symmetric buffers, compiled K1/K2 roles, Green
    contexts, graph executables, SM partition, counter bank, and graph variant.
    The common tuner supplies rank-local median then all-rank MAX scoring.
    """

    if y is None:
        raise ValueError("split autotune requires a caller output tensor")
    cfg = symm_buffer.session.config
    require_hopper_mxfp4_tuning_device()
    if process_group is None:
        process_group = symm_buffer.session._process_group
    n = cfg.num_tokens_per_rank if num_tokens is None else num_tokens
    if candidates is None:
        candidates = hopper_mxfp4_ordered_candidates(
            cfg.num_tokens_per_rank,
            execution_mode="split",
            hidden=cfg.hidden,
            intermediate=cfg.intermediate,
            routing_profile=cfg.routing_profile,
        )
    else:
        candidates = [
            validate_hopper_mxfp4_tactic(candidate, execution_mode="split")
            for candidate in candidates
        ]
        frozen_candidates = hopper_mxfp4_candidates(
            execution_mode="split",
            routing_profile=cfg.routing_profile,
        )
        outside_union = [
            candidate for candidate in candidates if candidate not in frozen_candidates
        ]
        if outside_union:
            raise ValueError(
                "supplied MXFP4 split autotune candidate is outside the "
                "frozen manifest candidate union"
            )
        if any(
            candidate in candidates[:index]
            for index, candidate in enumerate(candidates)
        ):
            raise ValueError("supplied MXFP4 split autotune candidates must be unique")
        candidates = [
            candidate
            for candidate in candidates
            if is_hopper_mxfp4_tactic_shape_compatible(
                candidate,
                execution_mode="split",
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
            )
        ]
        if not candidates:
            raise ValueError(
                "no supplied MXFP4 split autotune candidate supports "
                f"hidden={cfg.hidden}, intermediate={cfg.intermediate}"
            )

    adapter = _SplitTacticAdapter(symm_buffer)

    def launch() -> None:
        from .hopper_mxfp4_split import hopper_mxfp4_split_mega_moe

        hopper_mxfp4_split_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            adapter.current,
            num_tokens=n,
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
            sync=True,
        )

    def _commit_and_record(winner: Dict[str, Any], p50_s: float) -> None:
        # Commit is collective resource lifecycle and therefore runs on every
        # rank. Only rank zero mutates the shared persistent cache file.
        adapter.commit()
        if cfg.rank == 0:
            from .knob_cache import record_knobs

            provenance = hopper_mxfp4_tuning_provenance(
                execution_mode="split",
                routing_profile=cfg.routing_profile,
            )
            manifest_sha256 = provenance.get("manifest_sha256")
            if manifest_sha256 is None:
                manifest_sha256 = provenance["runtime_manifest_sha256"]
            record_knobs(
                winner,
                dtype=_SPLIT_TUNING_IDENTITY,
                fp8_scale_mode="mxfp4_hybrid",
                world_size=cfg.world_size,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                num_experts=cfg.num_total_experts,
                topk=cfg.num_topk,
                max_tokens=cfg.num_tokens_per_rank,
                gate_up_clamp=cfg.gate_up_clamp,
                routing_profile=cfg.routing_profile,
                p50_us=p50_s * 1e6,
                source=(f"autotune:sm90_mxfp4_split:{manifest_sha256}"),
            )

    try:
        return autotune_knobs(
            adapter,
            launch,
            candidates,
            label="sm90_mxfp4_green_split_mega",
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            on_winner=_commit_and_record,
            process_group=process_group,
            expected_world_size=cfg.world_size,
        )
    finally:
        adapter.close()


__all__ = ["autotune_hopper_mxfp4_split_mega_moe"]
