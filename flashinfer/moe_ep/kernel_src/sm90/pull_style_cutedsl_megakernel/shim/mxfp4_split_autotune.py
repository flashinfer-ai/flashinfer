# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Collective autotuning for fixed-pointer Hopper MXFP4 Green split sessions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch

from .autotune import autotune_knobs
from .hopper_mxfp4_split import (
    MegaMoEHopperMxfp4SplitSession,
    MegaMoEHopperMxfp4SplitSymmBuffer,
    _SPLIT_TUNING_IDENTITY,
)
from .mxfp4_tuner import (
    hopper_mxfp4_cache_provenance_sha256,
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
        self._retired_session: Optional[MegaMoEHopperMxfp4SplitSession] = None
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

    @staticmethod
    def _make_candidate(
        source: MegaMoEHopperMxfp4SplitSymmBuffer,
        tactic: Dict[str, Any],
    ) -> MegaMoEHopperMxfp4SplitSymmBuffer:
        cfg = source.session.config
        candidate_cfg = replace(
            cfg,
            k1_mma_tiler_mnk=tactic["k1_mma_tiler_mnk"],
            k2_mma_tiler_mnk=tactic["k2_mma_tiler_mnk"],
            k1_cluster_shape_mnk=tactic["k1_cluster_shape_mnk"],
            k2_cluster_shape_mnk=tactic["k2_cluster_shape_mnk"],
            k1_group_hint=tactic["k1_group_hint"],
            k2_group_hint=tactic["k2_group_hint"],
            k1_num_sched_stages=tactic["k1_num_sched_stages"],
            k2_num_sched_stages=tactic["k2_num_sched_stages"],
            k1_sm_count=tactic["k1_sm_count"],
            k2_sm_count=tactic["k2_sm_count"],
            counter_epoch_banks=tactic["counter_epoch_banks"],
            graph_variant=tactic["graph_variant"],
            enable_iket=tactic["enable_iket"],
        )
        candidate_session = MegaMoEHopperMxfp4SplitSession(
            candidate_cfg,
            process_group=source.session._process_group,
        )
        return MegaMoEHopperMxfp4SplitSymmBuffer(
            num_total_experts=source.num_total_experts,
            num_max_tokens=source.num_max_tokens,
            num_topk=source.num_topk,
            hidden=source.hidden,
            intermediate=source.intermediate,
            rank=source.rank,
            world_size=source.world_size,
            x=source.x,
            x_sf=source.x_sf,
            topk_idx=source.topk_idx,
            topk_weights=source.topk_weights,
            output_activation=source.output_activation,
            _session=candidate_session,
            _sym_roots=[],
        )

    def apply_knobs(self, knobs: Dict[str, Any]) -> None:
        """Build a session owner that borrows the source fixed input tensors."""

        if self._current is not None:
            raise RuntimeError(
                "previous split candidate was not collectively discarded"
            )
        tactic = validate_hopper_mxfp4_tactic(
            knobs,
            execution_mode="split",
        )
        source = self._source
        self._current = self._make_candidate(source, tactic)

    def prepare(
        self,
        transformed_l1: Any,
        transformed_l2: Any,
        *,
        inputs: Any = None,
    ) -> None:
        """Compile the current candidate without entering its Green graphs."""

        from .hopper_mxfp4_split import _split_inputs

        current = self.current
        if inputs is None:
            inputs = _split_inputs(current, transformed_l1, transformed_l2)
        current.session.prepare_compile_only(inputs)

    def discard(self) -> None:
        self._destroy_current()

    def commit(self) -> None:
        """Transfer the measured winner's resources into the caller workspace."""

        winner = self.current
        source = self._source
        self._retired_session = source._session
        source._session = winner._session
        # The source now owns the winner session. The temporary buffer never
        # owned the borrowed input roots and must not destroy the session.
        winner._destroyed = True
        self._current = None
        self._committed = True

    def rollback(self) -> None:
        """Restore the source and collectively discard the staged winner."""

        if self._committed:
            retired = self._retired_session
            if retired is None:
                raise RuntimeError("split committed session has no retired source")
            winner_session = self._source._session
            self._source._session = retired
            self._retired_session = None
            self._committed = False
            winner_session.destroy()
            return
        self._destroy_current()

    def finalize_commit(self) -> None:
        """Release the retired source after every rank has committed."""

        if not self._committed or self._retired_session is None:
            raise RuntimeError("split winner is not pending finalization")
        self._retired_session.destroy()
        self._retired_session = None

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

    Every candidate borrows the caller's fixed input/output symmetric tensors
    and owns a fresh tactic-specific session: compiled K1/K2 roles, Green
    contexts, internal workspaces, graph executables, SM partition, counter
    bank, and graph variant. The common tuner supplies rank-local median then
    all-rank MAX scoring.
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

    preflight_inputs: Dict[str, Any] = {}

    def _preflight() -> None:
        from .comm import resolve_gate_up_clamp
        from .hopper_mxfp4_split import _split_inputs

        if symm_buffer._destroyed:
            raise RuntimeError("cannot autotune a destroyed split workspace")
        if not 0 <= n <= symm_buffer.num_max_tokens:
            raise ValueError("num_tokens is outside split workspace capacity")
        if tuple(y.shape) != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
            raise ValueError("split autotune output does not match the live problem")
        clamp = resolve_gate_up_clamp(
            gate_up_clamp=gate_up_clamp,
            activation_clamp=activation_clamp,
        )
        if clamp is not None and clamp != cfg.gate_up_clamp:
            raise ValueError(
                "gate_up_clamp is split compile identity; create a new session"
            )
        inputs = _split_inputs(symm_buffer, transformed_l1, transformed_l2)
        symm_buffer.session._input_validator._validate_inputs(
            inputs,
            num_tokens=inputs.activation.shape[0],
        )
        preflight_inputs["inputs"] = inputs

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

    def _record(winner: Dict[str, Any], p50_s: float) -> None:
        # Resource commit/finalize has already succeeded on every EP rank.
        # Only rank zero mutates the shared persistent cache file.
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
                tuning_provenance_sha256=hopper_mxfp4_cache_provenance_sha256(
                    execution_mode="split",
                    routing_profile=cfg.routing_profile,
                ),
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
            on_winner=_record,
            prepare_candidate=lambda: adapter.prepare(
                transformed_l1,
                transformed_l2,
                inputs=preflight_inputs["inputs"],
            ),
            discard_candidate=adapter.discard,
            commit_winner=adapter.commit,
            rollback_winner=adapter.rollback,
            finalize_winner=adapter.finalize_commit,
            preflight=_preflight,
            process_group=process_group,
            expected_world_size=cfg.world_size,
        )
    finally:
        adapter.close()


__all__ = ["autotune_hopper_mxfp4_split_mega_moe"]
