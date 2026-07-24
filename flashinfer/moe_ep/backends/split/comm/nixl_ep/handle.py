"""NixlEpHandle — wraps the per-dispatch NIXL handle tuple."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .....algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    _index_knobs,
)
from .....config import (
    CombineInputParams,
    CombineOutput,
    DispatchInputParams,
    DispatchOutput,
    HandleParams,
)
from .....core.comm.handle import Handle
# from .....api_logging import flashinfer_api  # disabled per PR #3453 review

if TYPE_CHECKING:
    from .fleet import NixlEpFleet


class NixlEpHandle(Handle):
    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        fleet: "NixlEpFleet",
        params: HandleParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        self._fleet = fleet
        self._handle_knobs = _index_knobs(algo_knobs)
        self._staged = HandleAlgoKnobSplitOperation in self._handle_knobs
        # NIXL builds pick the index width at compile time (TOPK_IDX_BITS,
        # int32 by default since v1.3); callers hand us int64 (the nccl.ep
        # convention), so cast to whatever this build binds.
        import torch

        topk_t = getattr(fleet._nixl_ep, "topk_idx_t", torch.int64)
        if not isinstance(topk_t, torch.dtype):
            topk_t = torch.int64
        self._topk_ids = (
            params.topk_ids
            if params.topk_ids.dtype == topk_t
            else params.topk_ids.to(topk_t)
        )

        # Stream the Buffer kernels should run on: the user-stream knob wins,
        # else the fleet's bootstrap stream; 0 = leave the current stream alone.
        us = self._handle_knobs.get(HandleAlgoKnobUserStream)
        self._stream = (
            int(us.stream)  # type: ignore[attr-defined]
            if us is not None
            else int(fleet._bootstrap.stream or 0)
        )
        self._stream_obj = None  # lazily built torch.cuda.ExternalStream

        # State stashed by dispatch() for combine() + complete().
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None

    def _stream_ctx(self):
        """Context manager redirecting Buffer kernels to the bound stream.

        The NIXL ``Buffer`` API takes no stream argument (unlike nccl.ep) and
        launches on torch's current stream, so honoring
        :class:`HandleAlgoKnobUserStream` means swapping the current stream
        around each call. ``complete()`` runs under the same context so the
        event wait / recv hook targets that stream too.
        """
        if not self._stream:
            return contextlib.nullcontext()
        import torch

        if self._stream_obj is None:
            self._stream_obj = torch.cuda.ExternalStream(self._stream)
        return torch.cuda.stream(self._stream_obj)

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Forward to ``Buffer.low_latency_dispatch``."""
        x = params.x[0]  # MVP: single token tensor
        buf = self._fleet.buffer
        async_finish = not self._staged
        return_recv_hook = self._staged
        with self._stream_ctx():
            (
                recv_x,
                recv_count,
                handle,
                event,
                hook,
            ) = buf.low_latency_dispatch(
                x,
                self._topk_ids,
                self._fleet.params.max_tokens_per_rank,
                self._fleet.params.num_experts,
                use_fp8=self._fleet.use_fp8,
                round_scale=self._fleet.use_ue8m0,
                use_ue8m0=self._fleet.use_ue8m0,
                async_finish=async_finish,
                return_recv_hook=return_recv_hook,
            )
        self._nixl_handle = handle
        self._event = event
        self._recv_hook = hook
        # recv_x is (fp8_tensor, scales) tuple when use_fp8 — surface both.
        if isinstance(recv_x, tuple):
            expert_tensors, expert_scales = recv_x[0], recv_x[1]
        else:
            expert_tensors, expert_scales = recv_x, None
        # num_tokens is the per-expert row count of the recv buffer (same
        # semantics as nccl_ep LL EXPERT_MAJOR: max_tokens_per_rank * ranks).
        # Read it off the returned [num_local, rows, hidden] tensor; the
        # Buffer is sized to the fleet's rank capacity, not the live world.
        if expert_tensors.dim() == 3:
            num_tokens = expert_tensors.size(1)
        else:
            num_tokens = self._fleet.params.max_tokens_per_rank * self._fleet.capacity
        return DispatchOutput(
            expert_tensors=expert_tensors,
            num_tokens=num_tokens,
            expert_counts=recv_count,
            expert_scales=expert_scales,
        )

    # @flashinfer_api  # disabled per PR #3453 review
    def combine(self, params: CombineInputParams) -> CombineOutput:
        """Forward to ``Buffer.low_latency_combine``."""
        x = params.x[0]
        buf = self._fleet.buffer
        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NixlEpHandle.combine requires HandleAlgoKnobTopKWeights set "
                "at handle creation; NIXL needs the per-token weights to "
                "reweight on combine."
            )
        topk_weights = tw.weights  # type: ignore[attr-defined]
        out_t: Optional[Any] = params.out
        with self._stream_ctx():
            result = buf.low_latency_combine(
                x,
                self._topk_ids,
                topk_weights,
                self._nixl_handle,
                async_finish=not self._staged,
                zero_copy=False,
                return_recv_hook=self._staged,
                out=out_t,
            )
        # low_latency_combine returns (combined_x, event, hook).
        if isinstance(result, tuple):
            combined_x = result[0]
            self._event = result[1] if len(result) > 1 else None
            self._recv_hook = result[2] if len(result) > 2 else None
        else:
            combined_x = result
        return CombineOutput(x=combined_x)

    # @flashinfer_api  # disabled per PR #3453 review
    def complete(self) -> None:
        """Wait on the staged event or invoke the deferred recv hook."""
        with self._stream_ctx():
            if self._recv_hook is not None:
                self._recv_hook()
                self._recv_hook = None
            elif self._event is not None:
                self._event.current_stream_wait()

    def destroy(self) -> None:
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None
