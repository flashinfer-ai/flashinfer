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
        """Bind the handle to this step's routing at graph-stable addresses.

        When the caller's ``topk_ids`` dtype differs from the index width
        this build binds, the destination it is cast into is owned here
        rather than minted per call, and :meth:`update` funnels every later
        step through it. When no cast is needed there is no such buffer and
        the caller's tensor is bound as-is; there address stability comes
        instead from :meth:`update` rejecting an address change under
        capture -- a raise, not a funnel, and one that only fires while
        capturing, so an eager rebind between replays would still move the
        address out from under a recorded graph. Through
        :class:`MoEEpSplitGraphState` that cannot happen (its ``_check``
        rejects a rebound ``topk_ids`` first); a direct Handle-API caller is
        on its own.
        """
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
        self._topk_t = topk_t
        # When a cast is needed, own the destination instead of letting .to()
        # mint a fresh tensor per update: a CUDA graph binds whatever address
        # the dispatch kernel saw at capture, so a per-call allocation would
        # leave replays reading routing that update() no longer writes.
        self._topk_cast_buf = (
            None
            if params.topk_ids.dtype == topk_t
            else torch.empty_like(params.topk_ids, dtype=topk_t)
        )
        self._topk_ids = self._bind_topk(params.topk_ids)

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

    def _bind_topk(self, topk_ids):
        """Return the ids tensor the transport will read, at a stable address."""
        if self._topk_cast_buf is None:
            return topk_ids
        self._topk_cast_buf.copy_(topk_ids)
        return self._topk_cast_buf

    @staticmethod
    def _capturing() -> bool:
        """The single predicate deciding this handle's capture-mode behavior.

        Two things change under capture: send and recv use a combined launch
        (see :meth:`_use_hook`) and :meth:`update`
        promotes a routing-address change from a harmless rebind into an
        error. Both ask here so they can never disagree about which mode the
        handle is in.
        """
        import torch

        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()

    def _use_hook(self) -> bool:
        """Whether to drive the Buffer in ``return_recv_hook`` mode.

        Outside capture: yes, matching vLLM's native nixl_ep call convention.
        Note the hook does not itself wait for anything: it only defers
        launching ``EP_RECV_PHASE`` (``nixl_ep.cpp`` ~:1141 dispatch, ~:1242
        combine), which lets the caller interleave host work between send and
        recv.

        Under capture: no. ``return_recv_hook=False`` launches
        ``EP_SEND_PHASE | EP_RECV_PHASE`` as one kernel and joins it back to
        the compute stream inside the transport. A hook invoked during capture
        would also record its kernel launch, but the combined path needs no
        deferred host orchestration. Arrival is a device spin on a peer-written
        flag in either mode, so dropping the hook removes no waiting.

        ``async_finish`` stays False in both modes -- see ``dispatch()``.
        """
        return not self._capturing()

    def update(self, params: HandleParams) -> None:
        """Rebind to a new step's routing. See :meth:`Handle.update`.

        NIXL-EP needs no ``ncclEpUpdateHandle`` analogue: it has no separate
        handle-init step, LL dispatch recomputes routing inside the kernel
        from ``topk_idx`` on every call, and the recv buffers live in the
        Buffer's persistent RDMA arena rather than in per-iteration state. So
        a captured dispatch already re-reads routing from its bound address on
        every replay, and this method exists to guarantee the binding that
        makes that true -- the ids land at the address the graph captured --
        and to reject the shape changes a graph cannot express.
        """
        topk_ids = params.topk_ids
        if topk_ids.shape != self._topk_ids.shape:
            raise ValueError(
                f"NixlEpHandle.update cannot change the routing shape: handle "
                f"was created with {tuple(self._topk_ids.shape)}, got "
                f"{tuple(topk_ids.shape)}. The buffers and the per-token "
                "weights bound at creation still describe the original rows. "
                "Create a new handle instead."
            )
        if not topk_ids.is_cuda:
            raise ValueError(
                f"NixlEpHandle.update: topk_ids must be on the GPU, got "
                f"{topk_ids.device}."
            )
        if not topk_ids.is_contiguous():
            raise ValueError("NixlEpHandle.update: topk_ids must be contiguous.")
        bound = self._bind_topk(topk_ids)
        if (
            self._topk_cast_buf is None
            and bound.data_ptr() != self._topk_ids.data_ptr()
            and self._capturing()
        ):
            # No cast buffer to funnel through, so a different tensor means a
            # different address -- silently fine eagerly, silently WRONG once
            # captured, which is the case worth naming.
            raise ValueError(
                "NixlEpHandle.update: topk_ids moved to a different buffer "
                f"during capture (0x{self._topk_ids.data_ptr():x} -> "
                f"0x{bound.data_ptr():x}). A graph binds the address it "
                "saw, so replays would keep reading the old one; write the "
                "new ids into the registered tensor in place."
            )
        self._topk_ids = bound

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Forward to ``Buffer.low_latency_dispatch``."""
        x = params.x[0]  # MVP: single token tensor
        buf = self._fleet.buffer
        use_hook = self._use_hook()
        # async_finish is always False. Observed: with async_finish=True this
        # path deadlocked combine under sustained load (peers timing out on
        # "combine receive src_rank 0"), fixed in b848e0ae / PR #4139. The
        # mechanism the transport source supports is a missing JOIN rather than
        # RDMA-landing semantics: the pre-fix caller stashed dispatch's event in
        # self._event and combine() then overwrote it, so nothing ever ordered
        # the expert GEMM after dispatch. async_finish=False makes the transport
        # issue that join itself. Eagerly the recv hook then defers the recv
        # launch (not a wait -- see _use_hook()); run it now for the synchronous
        # path, or defer it to complete() when staged (DBO). Under capture there
        # is no hook at all.
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
            return_recv_hook=use_hook,
        )
        self._nixl_handle = handle
        if self._staged and use_hook:
            self._event = event
            self._recv_hook = hook
        else:
            # Not staged, or captured (where there is no hook to defer and the
            # kernel already waited): drain now / nothing to drain.
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
        use_hook = self._use_hook()
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
            return_recv_hook=use_hook,
            out=out_t,
        )
        # low_latency_combine returns (combined_x, event, hook).
        if isinstance(result, tuple):
            combined_x = result[0]
            event = result[1] if len(result) > 1 else None
            hook = result[2] if len(result) > 2 else None
        else:
            combined_x, event, hook = result, None, None
        if self._staged and use_hook:
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
