"""NcclEpHandle — per-iteration handle over the ``nccl.ep`` v0.1.0 API.

Supports three dispatch/combine I/O contracts, selected from FleetParams
(``algorithm`` + ``layout``):

**LL EXPERT_MAJOR** (``nccl.ep.Layout.EXPERT_MAJOR``), matching the compute bridge:

* dispatch input  : 2D ``[num_tokens, hidden]`` bf16 (``DispatchInputs.tokens``)
* dispatch output : 3D ``[num_local_experts, max_tokens_per_rank * world_size, hidden]``
                    bf16 (``DispatchOutputs.tokens``; allocated by us, library fills)
* dispatch counts : 1D ``[num_local_experts]`` int32 written by the library into
                    ``LayoutInfo.expert_counters`` (per-expert received token count)
* combine input   : same 3D shape as dispatch output (after inner compute)
* combine output  : 2D ``[num_tokens, hidden]`` bf16 (``CombineOutputs.tokens``)
* combine weights : 2D ``[num_tokens, top_k]`` fp32 routing weights, applied on the
                    receive side via ``CombineOutputs.topk_weights``

**LL RANK_MAJOR** (``nccl.ep.Layout.RANK_MAJOR``) — tokens grouped by source rank:

* dispatch input  : tokens + ``topk_weights`` (per-token routing weights)
* dispatch output : 3D ``[world_size, max_tokens_per_rank, hidden]`` bf16, plus the
                    received ``topk_weights`` / ``topk_idx`` (``[M, top_k]``)
* dispatch counts : 1D ``[world_size]`` int32 into ``LayoutInfo.src_rank_counters``
* combine input   : the 3D ``[world_size, max_tokens_per_rank, hidden]`` pre-reduced
                    expert output; combine takes NO ``topk_weights`` (the caller's
                    pre-reduce already applied them) and just sums across ranks
* combine output  : 2D ``[num_tokens, hidden]`` bf16

**HT FLAT** (``nccl.ep.Layout.FLAT``) — see ``_dispatch_ht`` / the combine branch.

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

        from ..config import EpAlgorithm, EpLayout

        self._fleet = fleet
        self._ep = fleet.nccl_ep
        self._handle_knobs = _index_knobs(algo_knobs)
        self._stream = self._knob_stream()
        self._staged = HandleAlgoKnobSplitOperation in self._handle_knobs
        self._destroyed = False

        # Honor the algorithm + layout carried in FleetParams. HT always uses the
        # FLAT receive layout ([num_recv_tokens, hidden] with topk_weights/topk_idx
        # exchanged in dispatch). LL uses one of two layouts:
        #   * EXPERT_MAJOR — recv [num_local_experts, max_tokens*world, hidden];
        #     each padded row pre-assigned to one expert; combine reweights on recv.
        #   * RANK_MAJOR   — recv [world, max_tokens_per_rank, hidden]; tokens
        #     grouped by source rank (carrying their topk_idx/weights); caller
        #     pre-reduces across local experts and combine sums across ranks.
        # The three paths have different dispatch/combine I/O contracts; we branch
        # on ``_is_ht`` / ``_is_rank_major`` below.
        self._is_ht = fleet.params.algorithm == EpAlgorithm.HIGH_THROUGHPUT
        self._is_rank_major = (not self._is_ht) and (
            fleet.params.layout is EpLayout.RANK_MAJOR
        )

        world_size = fleet.bootstrap.world_size
        self._world_size = world_size
        self._num_local_experts = fleet.params.num_experts // world_size

        # topk_idx must be int64 [num_tokens, top_k] for the v0.1.0 API.
        topk_idx = params.topk_ids
        if topk_idx.dtype != torch.int64:
            topk_idx = topk_idx.to(torch.int64)
        self._topk_idx = topk_idx  # keepalive
        self._num_tokens_in = topk_idx.shape[0]
        self._top_k = topk_idx.shape[1]
        self._topk_idx_t = self._ep.Tensor(topk_idx)

        # Per-source counter the library writes at dispatch (LL): EXPERT_MAJOR
        # gets per-local-expert recv counts [num_local_experts]; RANK_MAJOR gets
        # per-source-rank token counts [world].
        recv_count_len = world_size if self._is_rank_major else self._num_local_experts
        self._recv_count_t = torch.zeros(
            recv_count_len, dtype=torch.int32, device="cuda"
        )

        if self._is_ht:
            layout = self._ep.Layout.FLAT
        elif self._is_rank_major:
            layout = self._ep.Layout.RANK_MAJOR
        else:
            layout = self._ep.Layout.EXPERT_MAJOR
        self._handle = fleet.group.create_handle(
            layout,
            self._topk_idx_t,
            layout_info=None,  # max_tokens enabled → no handle-time layout_info
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
        """Send-and-recv dispatch of token tensors (LL EXPERT_MAJOR or HT FLAT)."""
        x = params.x[0]  # MVP: single token tensor [num_tokens, hidden] bf16
        if self._is_ht:
            return self._dispatch_ht(x)
        if self._is_rank_major:
            return self._dispatch_ll_rank_major(x)
        return self._dispatch_ll(x)

    def _dispatch_ll(self, x) -> DispatchOutput:
        import torch

        world_size = self._fleet.bootstrap.world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size

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
        config = self._ep.DispatchConfig(send_only=int(self._staged), round_scales=0)

        self._handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=config,
            stream=self._stream,
        )
        # Finish the dispatch (incl. the cross-rank RECEIVE) before the inner
        # compute reads the recv buffer. Required even in non-staged mode — the LL
        # dispatch is async, so without complete() the remote slots are still empty
        # and FFN(empty)=0 silently drops every remote-routed token's contribution.
        # Mirrors contrib/nccl_ep/ep_test.py, which calls complete() after dispatch
        # unconditionally.
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

    def _dispatch_ll_rank_major(self, x) -> DispatchOutput:
        """LL RANK_MAJOR dispatch.

        Recv buffer is ``[world, max_tokens_per_rank, hidden]`` (tokens grouped by
        source rank). Like HT FLAT, dispatch also returns the per-token received
        ``topk_weights`` / ``topk_idx``, and writes per-source-rank counts into
        ``LayoutInfo.src_rank_counters``. Returns the 3D recv tensor; the compute
        bridge pre-reduces across local experts and combine sums across ranks.
        """
        import torch

        world_size = self._world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size
        m = max_per_rank * world_size  # total recv slots [world, max_per_rank]

        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NcclEpHandle RANK_MAJOR dispatch requires HandleAlgoKnobTopKWeights "
                "(rank-major dispatch sends per-token routing weights)."
            )
        weights = tw.weights  # type: ignore[attr-defined]
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        # RANK_MAJOR mirrors the 3D rank-grouped token buffer: the library asserts
        # the received topk_weights / topk_idx are 3D [world, max_per_rank, top_k]
        # (nccl_ep.cc:2482 `recv_topk_weights->ndim == 3`), not flat [M, top_k].
        out_t = torch.empty(
            world_size, max_per_rank, hidden, dtype=x.dtype, device=x.device
        )
        out_w = torch.empty(
            world_size, max_per_rank, self._top_k, dtype=torch.float32, device=x.device
        )
        # RANK_MAJOR received topk_idx is int32 (nccl_ep.cc:2488
        # `recv_topk_idx->datatype == ncclInt32`); the bridge upcasts to int64.
        out_idx = torch.empty(
            world_size, max_per_rank, self._top_k, dtype=torch.int32, device=x.device
        )

        inputs = self._ep.DispatchInputs(
            tokens=self._ep.Tensor(x), topk_weights=self._ep.Tensor(weights)
        )
        outputs = self._ep.DispatchOutputs(
            tokens=self._ep.Tensor(out_t),
            topk_weights=self._ep.Tensor(out_w),
            topk_idx=self._ep.Tensor(out_idx),
        )
        layout_info = self._ep.LayoutInfo(
            src_rank_counters=self._ep.Tensor(self._recv_count_t)
        )
        config = self._ep.DispatchConfig(send_only=int(self._staged), round_scales=0)

        self._handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=config,
            stream=self._stream,
        )
        # Finish the dispatch (incl. cross-rank RECEIVE) before compute reads the
        # recv buffer — see _dispatch_ll for why this is unconditional.
        self._handle.complete(stream=self._stream)

        # Keepalives for combine() / async safety.
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_layout = layout_info
        self._dispatch_output_t = out_t
        self._dispatch_out_w = out_w
        self._dispatch_out_idx = out_idx

        # src_rank_counters is written on self._stream; sync before the host read.
        torch.cuda.ExternalStream(self._stream).synchronize()
        num_tokens = int(self._recv_count_t.sum().item())
        # Flatten the rank-grouped routing to the [M, top_k] view the compute
        # bridge consumes (row-major, matching out_t.reshape(M, hidden)).
        return DispatchOutput(
            expert_tensors=out_t,
            num_tokens=num_tokens,
            recv_topk_idx=out_idx.reshape(m, self._top_k),
            recv_topk_weights=out_w.reshape(m, self._top_k),
        )

    def _dispatch_ht(self, x) -> DispatchOutput:
        """HT FLAT dispatch: tokens + topk_weights in, [num_recv, H] + recv
        topk_weights/idx out.  num_recv = max_tokens_per_rank * num_local_experts
        (per-expert blocks), reshaped to the 3D [L, cap, H] view the compute
        bridge expects (cap = max_tokens_per_rank)."""
        import torch

        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size
        num_recv = max_per_rank * self._num_local_experts

        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NcclEpHandle HT dispatch requires HandleAlgoKnobTopKWeights "
                "(HT forward dispatch sends per-token routing weights)."
            )
        weights = tw.weights  # type: ignore[attr-defined]
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        # HT FLAT recv is token-major: each row is a received token carrying its
        # received LOCAL topk_idx (-1 = non-local) in out_idx. The compute bridge
        # masks non-local picks (and padding rows, whose topk_idx is -1) to weight 0,
        # so the recv buffer can be uninitialized (matches the RANK_MAJOR path).
        out_t = torch.empty(num_recv, hidden, dtype=x.dtype, device=x.device)
        out_w = torch.empty(num_recv, self._top_k, dtype=torch.float32, device=x.device)
        out_idx = torch.empty(num_recv, self._top_k, dtype=torch.int64, device=x.device)

        inputs = self._ep.DispatchInputs(
            tokens=self._ep.Tensor(x), topk_weights=self._ep.Tensor(weights)
        )
        outputs = self._ep.DispatchOutputs(
            tokens=self._ep.Tensor(out_t),
            topk_weights=self._ep.Tensor(out_w),
            topk_idx=self._ep.Tensor(out_idx),
        )
        config = self._ep.DispatchConfig(send_only=int(self._staged), round_scales=0)

        self._handle.dispatch(
            inputs, outputs, layout_info=None, config=config, stream=self._stream
        )
        # Finish the dispatch (incl. cross-rank RECEIVE) before compute reads the
        # recv buffer — unconditional, matching the LL paths and ep_test.py.
        self._handle.complete(stream=self._stream)

        # Keepalives.
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_output_t = out_t
        self._dispatch_out_w = out_w
        self._dispatch_out_idx = out_idx

        out_3d = out_t.view(self._num_local_experts, max_per_rank, hidden)
        return DispatchOutput(
            expert_tensors=out_3d,
            num_tokens=num_recv,
            recv_topk_idx=out_idx,
            recv_topk_weights=out_w,
        )

    # ----------------------------------------------------------------- combine

    # @flashinfer_api  # disabled per PR #3453 review
    def combine(self, params: CombineInputParams) -> CombineOutput:
        """Gather expert outputs back to the originating ranks (LL or HT)."""
        import torch

        x = params.x[0]  # 3D [num_local_experts, cap, hidden] from the bridge
        hidden = self._fleet.params.token_hidden_size
        out_t = (
            params.out
            if params.out is not None
            else torch.empty(
                self._num_tokens_in, hidden, dtype=x.dtype, device=x.device
            )
        )

        if self._is_ht:
            # HT FLAT combine: input is the 2D [num_recv, hidden] expert output;
            # FWD combine takes NO topk_weights (routing weights were captured at
            # dispatch). Flatten the bridge's 3D [L, cap, H] view back to 2D.
            x2d = x.reshape(-1, hidden)
            inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x2d))
            outputs = self._ep.CombineOutputs(tokens=self._ep.Tensor(out_t))
            config = self._ep.CombineConfig(send_only=int(self._staged))
            self._handle.combine(inputs, outputs, config=config, stream=self._stream)
            if self._staged:
                self._handle.complete(stream=self._stream)
            self._combine_inputs = inputs
            self._combine_outputs = outputs
            self._combine_x2d = x2d
            return CombineOutput(x=out_t)

        if self._is_rank_major:
            # LL RANK_MAJOR combine: input is the 3D [world, max_per_rank, hidden]
            # pre-reduced expert output. Routing weights were already applied by
            # the caller's pre-reduce, so combine takes NO topk_weights and simply
            # sums each token's contributions back across the ranks that held it.
            inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x))
            outputs = self._ep.CombineOutputs(tokens=self._ep.Tensor(out_t))
            config = self._ep.CombineConfig(send_only=int(self._staged))
            self._handle.combine(inputs, outputs, config=config, stream=self._stream)
            if self._staged:
                self._handle.complete(stream=self._stream)
            self._combine_inputs = inputs
            self._combine_outputs = outputs
            return CombineOutput(x=out_t)

        # LL EXPERT_MAJOR combine: weights applied on the receive side.
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
