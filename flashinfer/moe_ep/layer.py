"""MoEEpLayer — public nn.Module for MoE Expert-Parallel.

Owns a lazy :class:`Fleet`, creates a fresh :class:`Handle` per forward
pass, runs ``dispatch → inner_compute → combine → complete`` in that
order.

Inner compute is the **identity** when no ``compute_config`` is supplied
(comm-only, backward compatible).  When a ``compute_config`` (a
:class:`flashinfer.fused_moe.api.MoEConfig`) and ``weights`` are given, the
dispatched per-expert tokens are run through the unified MoE compute API
(``flashinfer.fused_moe.MoELayer``) as a *per-expert grouped GEMM*.  EXPERT_MAJOR
runs a ``top_k=1`` pre-routed pack and ``combine`` applies the routing weights on
receive; RANK_MAJOR and HT instead drive the runner with the *received* per-token
routing at the real ``top_k`` (weighted, ``do_finalize=True``) and ``combine`` sums
the per-rank partials.  See :mod:`flashinfer.moe_ep._compute_bridge` for the layout
translation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
import torch.nn as nn

from .algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
)
from .config import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    DispatchOutput,
    FleetParams,
    HandleParams,
)

# from ..api_logging import flashinfer_api  # disabled per PR #3453 review
from .fleet import Fleet, create_fleet

if TYPE_CHECKING:
    from ..fused_moe.api import MoEConfig, MoEWeightPack
    from ..fused_moe.layer import MoELayer
    from .tensors import MoEEpTensors


class MoEEpLayer(nn.Module):
    """Backend-agnostic Expert-Parallel layer.

    Backend is selected via the ``backend`` argument (string name or
    ``NcclEpConfig`` / ``NvepConfig`` instance from
    :mod:`flashinfer.moe_ep.split_backends`).
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, object] = "nccl_ep",
        compute_config: Optional["MoEConfig"] = None,
        weights: Optional["MoEWeightPack"] = None,
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._fleet_knobs = list(fleet_knobs)
        self._backend = backend
        self._fleet: Fleet | None = None

        if (compute_config is None) != (weights is None):
            raise ValueError(
                "compute_config and weights must be supplied together "
                "(or both omitted for the comm-only identity path)."
            )
        if compute_config is not None:
            from ._validators import validate_compute_consistency

            validate_compute_consistency(fleet_params, bootstrap, compute_config)
        self._compute_config = compute_config
        self._weights = weights
        self._compute: "MoELayer | None" = None

        # Opt-in per-stage profiling. When True, forward() records CUDA events
        # around dispatch / compute / combine and stores their elapsed GPU time
        # (ms) in ``last_timings_ms`` after a device sync. Off by default (zero
        # overhead on the hot path). Used by benchmarks/bench_moe_ep.py.
        self.enable_timing = False
        self.last_timings_ms: dict[str, float] = {}

    def _ensure_fleet(self) -> Fleet:
        if self._fleet is None:
            self._fleet = create_fleet(
                self._bootstrap,
                self._fleet_params,
                self._fleet_knobs,
                backend=self._backend,
            )
        return self._fleet

    def _ensure_compute(self) -> "MoELayer":
        if self._compute is None:
            import dataclasses

            from ..fused_moe.layer import MoELayer
            from .config import EpAlgorithm, EpLayout

            cfg = self._compute_config
            received_routing = (
                self._fleet_params.layout is EpLayout.RANK_MAJOR
                or self._fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
            )
            if received_routing:
                # RANK_MAJOR and HT FLAT are token-major: each received token carries
                # its real top-k routing (received local topk_idx, -1 = non-local).
                # The runner runs at the model's top_k with do_finalize=True (the
                # per-token weighted pre-reduce across local experts); the bridge
                # masks non-local picks to weight 0.
                compute_cfg = cfg
            else:
                # EXPERT_MAJOR: each dispatched row is already assigned to exactly one
                # local expert, so the inner MoE kernel runs with top_k=1 regardless
                # of the model's real top-k (which lives in dispatch/combine).  The
                # bridge synthesizes top_k=1 routing (selected_experts [M, 1],
                # final_scales == 1) to match; the kernel's check_routing() asserts
                # expert_indices.size(1) == top_k.
                compute_cfg = dataclasses.replace(
                    cfg, routing=dataclasses.replace(cfg.routing, top_k=1)
                )
            self._compute = MoELayer(compute_cfg)
        return self._compute

    @staticmethod
    def _inner_compute_identity(expert_tensors: "torch.Tensor") -> "torch.Tensor":
        """Stub inner compute — passes dispatched tokens through unchanged.

        Used when no ``compute_config`` was supplied (comm-only path).
        """
        return expert_tensors

    def _inner_compute(self, d: "DispatchOutput") -> "torch.Tensor":
        """Grouped-GEMM expert FFN over the dispatched tokens.

        Bridges the 3D dispatch output to the token-major compute pack, runs
        ``flashinfer.fused_moe.MoELayer``, and reshapes back to the 3D combine
        layout.  EXPERT_MAJOR uses a top_k=1 pre-routed pack (combine owns the
        reweight); RANK_MAJOR and HT use the received per-token routing at the real
        top_k with do_finalize (this rank applies the weights).  See
        :mod:`flashinfer.moe_ep._compute_bridge`.
        """
        expert_tensors = d.expert_tensors
        if self._compute_config is None:
            return self._inner_compute_identity(expert_tensors)

        from ..fused_moe.api import QuantVariant
        from .config import EpAlgorithm, EpLayout

        is_nvfp4 = self._compute_config.quant.variant is QuantVariant.NVFP4
        offset = self._compute_config.experts.local_expert_offset
        # dim0/dim1 are the two leading dims of the 3D dispatch output; the reshape
        # back to the combine layout is identical for both LL layouts.
        dim0, dim1, _ = expert_tensors.shape

        is_ht = self._fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
        if is_ht or self._fleet_params.layout is EpLayout.RANK_MAJOR:
            # RANK_MAJOR recv [world, max_tokens_per_rank, hidden] and HT FLAT recv
            # [num_local_experts, max_tokens_per_rank, hidden] are both token-major:
            # each received row is a token carrying its received LOCAL topk_idx
            # (-1 = non-local). Drive compute from that routing, masked to local
            # experts, weighted (combine sums unweighted for both layouts).
            from ._compute_bridge import (
                build_activation_pack_rank_major,
                reshape_for_combine,
            )

            if d.recv_topk_idx is None or d.recv_topk_weights is None:
                raise RuntimeError(
                    f"{'HT' if is_ht else 'RANK_MAJOR'} compute requires the dispatch "
                    "to return recv_topk_idx / recv_topk_weights (the received "
                    "per-token routing); got None."
                )
            act_pack = build_activation_pack_rank_major(
                expert_tensors,
                d.recv_topk_idx,
                d.recv_topk_weights,
                num_local_experts=self._compute_config.experts.local_num_experts,
                local_expert_offset=offset,
                is_nvfp4=is_nvfp4,
            )
        else:
            # EXPERT_MAJOR (and HT's expert-major-shaped view): each padded row is
            # pre-assigned to one local expert (expert = row // cap).
            from ._compute_bridge import build_activation_pack, reshape_for_combine

            act_pack = build_activation_pack(
                expert_tensors,
                local_expert_offset=offset,
                is_nvfp4=is_nvfp4,
            )

        out_2d = self._ensure_compute()(act_pack, self._weights)
        return reshape_for_combine(out_2d, dim0, dim1)

    # @flashinfer_api  # disabled per PR #3453 review
    def forward(self, t: "MoEEpTensors") -> "torch.Tensor":
        fleet = self._ensure_fleet()
        handle_knobs: list[AlgoKnob] = [
            HandleAlgoKnobUserStream(stream=torch.cuda.current_stream().cuda_stream),
            HandleAlgoKnobTopKWeights(weights=t.topk_weights),
        ]
        handle = fleet.create_handle(
            HandleParams(topk_ids=t.topk_ids),
            algo_knobs=handle_knobs,
        )

        if not self.enable_timing:
            # try/finally so the handle lifecycle is always drained (complete()):
            # the fused compute or combine can raise after dispatch, and skipping
            # complete() would leave the EP handle in an incomplete state.
            try:
                d = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
                expert_out = self._inner_compute(d)
                c = handle.combine(
                    CombineInputParams(
                        x=[expert_out], out=torch.empty_like(t.hidden_states)
                    )
                )
            finally:
                handle.complete()
            return c.x

        # Profiling path: bracket each stage with CUDA events. EP ops + compute
        # all run on the current stream (HandleAlgoKnobUserStream), so events
        # recorded here capture their GPU time. dispatch host-syncs internally,
        # which doesn't affect on-stream event timing.
        ev = {
            k: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for k in ("dispatch", "compute", "combine")
        }
        try:
            ev["dispatch"][0].record()
            d = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
            ev["dispatch"][1].record()
            ev["compute"][0].record()
            expert_out = self._inner_compute(d)
            ev["compute"][1].record()
            ev["combine"][0].record()
            c = handle.combine(
                CombineInputParams(
                    x=[expert_out], out=torch.empty_like(t.hidden_states)
                )
            )
            ev["combine"][1].record()
        finally:
            handle.complete()
        torch.cuda.synchronize()
        self.last_timings_ms = {
            k: start.elapsed_time(end) for k, (start, end) in ev.items()
        }
        return c.x

    def destroy(self) -> None:
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
