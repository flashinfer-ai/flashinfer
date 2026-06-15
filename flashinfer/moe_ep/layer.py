"""MoEEpLayer — public nn.Module for MoE Expert-Parallel.

Owns a lazy :class:`Fleet`, creates a fresh :class:`Handle` per forward
pass, runs ``dispatch → inner_compute → combine → complete`` in that
order. Inner compute is the **identity** in this iteration; FP8 / FP4
quant routing through ``flashinfer.fused_moe.*`` lands in a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

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
    FleetParams,
    HandleParams,
)

# from ..api_logging import flashinfer_api  # disabled per PR #3453 review
from .fleet import Fleet, create_fleet

if TYPE_CHECKING:
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
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._fleet_knobs = list(fleet_knobs)
        self._backend = backend
        self._fleet: Fleet | None = None

    def _ensure_fleet(self) -> Fleet:
        if self._fleet is None:
            self._fleet = create_fleet(
                self._bootstrap,
                self._fleet_params,
                self._fleet_knobs,
                backend=self._backend,
            )
        return self._fleet

    @staticmethod
    def _inner_compute_identity(
        expert_tensors: "torch.Tensor", num_tokens: int
    ) -> "torch.Tensor":
        """Stub inner compute — passes dispatched tokens through unchanged.

        Real quant-aware routing (cutlass_fused_moe / trtllm_* / cute_dsl)
        lands in a follow-up that takes (expert_tensors, num_tokens,
        EpConfig.quant) and dispatches to the right kernel.
        """
        return expert_tensors

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
        d = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
        expert_out = self._inner_compute_identity(d.expert_tensors, d.num_tokens)
        c = handle.combine(
            CombineInputParams(
                x=[expert_out],
                out=torch.empty_like(t.hidden_states),
            )
        )
        handle.complete()
        return c.x

    def destroy(self) -> None:
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
