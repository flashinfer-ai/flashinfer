"""NixlEpHandle — wraps the per-dispatch NIXL handle tuple."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from .....algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobTopKWeights,
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

        # HandleAlgoKnobUserStream is intentionally NOT honored by wrapping the
        # Buffer calls in a torch stream context. The NIXL Buffer takes no
        # stream argument and manages its own CUDA streams/events for the async
        # RDMA dispatch/combine completion; forcing its calls onto a foreign
        # (External) stream breaks that completion signaling and DEADLOCKS the
        # combine — rank 0 never pushes its combine data, so every peer times
        # out on "combine receive src_rank 0" (observed end-to-end on B200
        # DP8-EP decode). The Buffer must run on the natural current stream,
        # which is exactly the stream vLLM/callers have current when they call
        # dispatch/combine — so nothing is lost by ignoring the knob here.

        # State stashed by dispatch() for combine() + complete().
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Forward to ``Buffer.low_latency_dispatch``."""
        x = params.x[0]  # MVP: single token tensor
        buf = self._fleet.buffer
        # Always drive the Buffer with async_finish=False + a recv hook,
        # mirroring vLLM's proven native nixl_ep path. The MVP's
        # async_finish=True event does NOT guarantee the RDMA transfer has
        # landed — it deadlocks combine under load (rank 0 never finishes
        # sending, peers time out on "combine receive src_rank 0"). The hook
        # is the actual completion barrier; run it now for the synchronous
        # (non-staged) path, or defer it to complete() when staged (DBO).
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
            async_finish=False,
            return_recv_hook=True,
        )
        self._nixl_handle = handle
        if self._staged:
            self._event = event
            self._recv_hook = hook
        else:
            if hook is not None:
                hook()
            self._event = None
            self._recv_hook = None
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
        # async_finish=False + recv hook (see dispatch()): the async event
        # path is unreliable in the NIXL MVP and hangs combine under load.
        result = buf.low_latency_combine(
            x,
            self._topk_ids,
            topk_weights,
            self._nixl_handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=True,
            out=out_t,
        )
        # low_latency_combine returns (combined_x, event, hook).
        if isinstance(result, tuple):
            combined_x = result[0]
            event = result[1] if len(result) > 1 else None
            hook = result[2] if len(result) > 2 else None
        else:
            combined_x, event, hook = result, None, None
        if self._staged:
            self._event = event
            self._recv_hook = hook
        else:
            if hook is not None:
                hook()
            self._event = None
            self._recv_hook = None
        return CombineOutput(x=combined_x)

    # @flashinfer_api  # disabled per PR #3453 review
    def complete(self) -> None:
        """Wait on the staged event or invoke the deferred recv hook."""
        if self._recv_hook is not None:
            self._recv_hook()
            self._recv_hook = None
        elif self._event is not None:
            self._event.current_stream_wait()

    def destroy(self) -> None:
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None
