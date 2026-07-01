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
        self._topk_ids = params.topk_ids

        # State stashed by dispatch() for combine() + complete().
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Forward to ``Buffer.low_latency_dispatch``."""
        x = params.x[0]  # MVP: single token tensor
        buf = self._fleet.buffer
        async_finish = not self._staged
        return_recv_hook = self._staged
        (
            recv_x,
            _recv_count,
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
        # recv_x is (fp8_tensor, scales) tuple when use_fp8 — pick the data tensor.
        expert_tensors = recv_x[0] if isinstance(recv_x, tuple) else recv_x
        world = self._fleet._bootstrap.world_size
        num_local = self._fleet.params.num_experts // world
        num_tokens = num_local * self._fleet.params.max_tokens_per_rank * world
        return DispatchOutput(
            expert_tensors=expert_tensors,
            num_tokens=num_tokens,
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
        if self._recv_hook is not None:
            self._recv_hook()
            self._recv_hook = None
        elif self._event is not None:
            self._event.current_stream_wait()

    def destroy(self) -> None:
        self._nixl_handle = None
        self._event = None
        self._recv_hook = None
