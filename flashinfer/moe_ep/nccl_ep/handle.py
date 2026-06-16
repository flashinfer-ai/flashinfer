"""NcclEpHandle — per-iteration handle over the ``nccl.ep`` v0.1.0 API.

LL EXPERT_MAJOR layout (``nccl.ep.Layout.EXPERT_MAJOR``), matching what the
compute bridge consumes:

* dispatch input  : 2D ``[num_tokens, hidden]`` bf16 (``DispatchInputs.tokens``)
* dispatch output : 3D ``[num_local_experts, max_tokens_per_rank * world_size, hidden]``
                    bf16 (``DispatchOutputs.tokens``; allocated by us, library fills)
* dispatch counts : 1D ``[num_local_experts]`` int32 written by the library into
                    ``LayoutInfo.expert_counters`` (per-expert received token count)
* combine input   : same 3D shape as dispatch output (after inner compute)
* combine output  : 2D ``[num_tokens, hidden]`` bf16 (``CombineOutputs.tokens``)
* combine weights  : 2D ``[num_tokens, top_k]`` fp32 routing weights, applied on the
                    receive side via ``CombineOutputs.topk_weights``

``topk_idx`` is bound into the handle at create time (int64 ``[num_tokens, top_k]``).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Sequence

from ..algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    _index_knobs,
)
from ..config import (
    CombineInputParams,
    CombineOutput,
    DispatchInputParams,
    DispatchOutput,
    HandleParams,
)
from ..handle import Handle

# from ...api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from .fleet import NcclEpFleet


class NcclEpHandle(Handle):
    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        fleet: "NcclEpFleet",
        params: HandleParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        import torch

        self._fleet = fleet
        self._ep = fleet.nccl_ep
        self._handle_knobs = _index_knobs(algo_knobs)
        self._stream = self._knob_stream()
        self._staged = HandleAlgoKnobSplitOperation in self._handle_knobs
        self._destroyed = False

        world_size = fleet.bootstrap.world_size
        self._num_local_experts = fleet.params.num_experts // world_size

        # topk_idx must be int64 [num_tokens, top_k] for the v0.1.0 API.
        topk_idx = params.topk_ids
        if topk_idx.dtype != torch.int64:
            topk_idx = topk_idx.to(torch.int64)
        self._topk_idx = topk_idx  # keepalive
        self._num_tokens_in = topk_idx.shape[0]
        self._topk_idx_t = self._ep.Tensor(topk_idx)

        # Per-expert received-count buffer; the library writes it at dispatch.
        self._recv_count_t = torch.zeros(
            self._num_local_experts, dtype=torch.int32, device="cuda"
        )

        self._handle = fleet.group.create_handle(
            self._ep.Layout.EXPERT_MAJOR,
            self._topk_idx_t,
            layout_info=None,  # LL mode forbids handle-time layout_info
            config=None,
            stream=self._stream,
        )

    # ----------------------------------------------------------------- knobs

    def _knob_stream(self) -> int:
        k = self._handle_knobs.get(HandleAlgoKnobUserStream)
        return int(k.stream) if k is not None else self._fleet.stream  # type: ignore[attr-defined]

    # ----------------------------------------------------------------- dispatch

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Send-and-recv dispatch of token tensors (LL EXPERT_MAJOR)."""
        import torch

        world_size = self._fleet.bootstrap.world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size

        x = params.x[0]  # MVP: single token tensor [num_tokens, hidden] bf16

        out_t = torch.empty(
            self._num_local_experts,
            max_per_rank * world_size,
            hidden,
            dtype=x.dtype,
            device=x.device,
        )

        inputs = self._ep.DispatchInputs(tokens=self._ep.Tensor(x))
        outputs = self._ep.DispatchOutputs(tokens=self._ep.Tensor(out_t))
        layout_info = self._ep.LayoutInfo(
            expert_counters=self._ep.Tensor(self._recv_count_t)
        )
        config = self._ep.DispatchConfig(
            send_only=int(self._staged),
            round_scales=0,
        )

        self._handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=config,
            stream=self._stream,
        )
        if self._staged:
            self._handle.complete(stream=self._stream)

        # Keepalives for combine() / async safety.
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_layout = layout_info
        self._dispatch_output_t = out_t

        # recv_count is written on self._stream; sync it before the host read.
        torch.cuda.ExternalStream(self._stream).synchronize()
        num_tokens = int(self._recv_count_t.sum().item())
        return DispatchOutput(expert_tensors=out_t, num_tokens=num_tokens)

    # ----------------------------------------------------------------- combine

    # @flashinfer_api  # disabled per PR #3453 review
    def combine(self, params: CombineInputParams) -> CombineOutput:
        """Gather expert outputs back to the originating ranks (LL EXPERT_MAJOR)."""
        import torch

        x = params.x[0]  # 3D [num_local_experts, max_per_rank * world, hidden]

        out_t = (
            params.out
            if params.out is not None
            else torch.empty(
                self._num_tokens_in,
                self._fleet.params.token_hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
        )

        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NcclEpHandle.combine requires HandleAlgoKnobTopKWeights set "
                "at handle creation; NCCL EP LL needs per-token weights to "
                "reweight on combine."
            )
        weights = tw.weights  # type: ignore[attr-defined]
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x))
        outputs = self._ep.CombineOutputs(
            tokens=self._ep.Tensor(out_t),
            topk_weights=self._ep.Tensor(weights),
        )
        config = self._ep.CombineConfig(send_only=int(self._staged))

        self._handle.combine(inputs, outputs, config=config, stream=self._stream)
        if self._staged:
            self._handle.complete(stream=self._stream)

        # Keepalives.
        self._combine_inputs = inputs
        self._combine_outputs = outputs
        self._combine_weights = weights
        return CombineOutput(x=out_t)

    # ----------------------------------------------------------------- complete

    # @flashinfer_api  # disabled per PR #3453 review
    def complete(self) -> None:
        """No-op in non-staged LL mode.

        ``nccl.ep.Handle.complete`` is only required when dispatch/combine ran
        with ``send_only=True`` (the :class:`HandleAlgoKnobSplitOperation` path),
        and is already issued inline there.
        """

    def destroy(self) -> None:
        if not self._destroyed:
            with contextlib.suppress(Exception):
                self._handle.destroy()
            self._destroyed = True

    def __del__(self) -> None:
        self.destroy()
