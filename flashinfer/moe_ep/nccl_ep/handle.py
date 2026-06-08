"""NcclEpHandle — per-iteration ncclEpHandle_t wrapper.

The C ABI's shape conventions for LL mode (matches
``3rdparty/nccl/contrib/nccl_ep/ep_test.py``):

* dispatch input  : 2D ``[num_tokens, hidden]`` bf16, tag TOKENS
* dispatch output : 3D ``[num_local_experts, max_tokens_per_rank * world_size, hidden]``
                    bf16, tag TOKENS (allocated by us, library fills)
* dispatch local  : 1D ``[num_local_experts]`` int32, tag
                    RECV_EXPERT_COUNTER_DEVICE — receives per-expert
                    token counts; required by the LL kernel.
* combine input   : same 3D shape as dispatch output (after inner compute)
* combine output  : 2D ``[num_tokens, hidden]`` bf16, tag TOKENS
* combine local   : 2D ``[num_tokens, top_k]`` fp32, tag TOPK_WEIGHTS

``ncclEpComplete`` runs after each dispatch and each combine — required
in HT mode regardless of send_only; harmless in LL mode and matches the
upstream ep_test pattern.
"""

from __future__ import annotations

import contextlib
import ctypes
from typing import TYPE_CHECKING, Sequence

from ..algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobNumReceivedTokens,
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
from .ndtensor import NDTensor, get_nccl_lib

if TYPE_CHECKING:
    from .fleet import NcclEpFleet


def _tag(name: str) -> int:
    from nccl_ep import ncclEpTensorTag_t  # type: ignore[import-not-found]

    return getattr(ncclEpTensorTag_t, f"NCCL_EP_TENSOR_TAG_{name}")


def _to_pointer_array(nds: Sequence[NDTensor]):
    """ctypes array of ncclNDTensor_t* for dispatch/combine I/O slots."""
    from nccl_ep import ncclNDTensor_t  # type: ignore[import-not-found]

    if not nds:
        return None, 0
    arr = (ncclNDTensor_t * len(nds))()
    for i, nd in enumerate(nds):
        arr[i] = nd.handle
    return arr, len(nds)


class NcclEpHandle(Handle):
    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        fleet: "NcclEpFleet",
        params: HandleParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        self._fleet = fleet
        self._handle_knobs = _index_knobs(algo_knobs)
        self._stream = self._knob_stream()
        self._staged = HandleAlgoKnobSplitOperation in self._handle_knobs
        self._destroyed = False
        # Cache the NCCL library handle so later calls (+ __del__) don't
        # re-resolve it; avoids interpreter-shutdown lookups.
        self._lib = get_nccl_lib()

        # NCCL EP handle is created with a topk-idx NDTensor and (optionally)
        # a RECV_EXPERT_COUNTER_HOST tensor in handle-local. Both are built
        # here once per handle.
        self._topk_idx_nd = NDTensor.from_torch(
            fleet.group, params.topk_ids, _tag("TOPK_IDX")
        )

        handle_local_nds = self._build_handle_local_tensors()
        self._handle_local_nds = handle_local_nds  # keepalive

        local_arr = [nd.handle for nd in handle_local_nds] if handle_local_nds else None
        self._handle = self._lib.ncclEpCreateHandle(
            fleet.group,
            self._topk_idx_nd.handle,
            None,  # config (reserved)
            ctypes.c_void_p(self._stream),
            local_tensors=local_arr,
            use_fp8=fleet.use_fp8,
        )

        # Pre-allocate the dispatch-local recv_count tensor (LL mode).
        # NCCL EP writes per-expert recv counts here during dispatch; combine
        # doesn't need it but we keep the storage alive across the handle.
        import torch

        world_size = fleet.bootstrap.world_size
        num_local_experts = fleet.params.num_experts // world_size
        self._recv_count_t = torch.zeros(
            num_local_experts, dtype=torch.int32, device="cuda"
        )
        self._recv_count_nd = NDTensor.from_torch(
            fleet.group, self._recv_count_t, _tag("RECV_EXPERT_COUNTER_DEVICE")
        )

    # ----------------------------------------------------------------- knobs

    def _knob_stream(self) -> int:
        k = self._handle_knobs.get(HandleAlgoKnobUserStream)
        return int(k.stream) if k is not None else self._fleet.stream  # type: ignore[attr-defined]

    def _build_handle_local_tensors(self) -> list[NDTensor]:
        """Local tensors passed at handle-create time.

        LL mode forbids handle-local tensors (RECV_EXPERT_COUNTER is a
        dispatch-local, not handle-local). HT mode optionally takes
        RECV_EXPERT_COUNTER_HOST when max_tokens_per_rank=NCCL_EP_AUTO.
        We honor :class:`HandleAlgoKnobNumReceivedTokens` only when
        user-set; default = empty.
        """
        out: list[NDTensor] = []
        nrt = self._handle_knobs.get(HandleAlgoKnobNumReceivedTokens)
        if nrt is not None:
            target = nrt.target  # type: ignore[attr-defined]
            out.append(
                NDTensor.from_torch(
                    self._fleet.group, target, _tag("RECV_EXPERT_COUNTER_HOST")
                )
            )
        return out

    # ----------------------------------------------------------------- dispatch

    # @flashinfer_api  # disabled per PR #3453 review
    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Send-and-recv dispatch of token tensors (LL mode)."""
        import torch

        from nccl_ep import ncclEpDispatchConfig_t  # type: ignore[import-not-found]

        world_size = self._fleet.bootstrap.world_size
        num_local_experts = self._fleet.params.num_experts // world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size

        x = params.x[0]  # MVP: single token tensor
        input_nd = NDTensor.from_torch(self._fleet.group, x, _tag("TOKENS"))

        # 3D dispatch output: [num_local_experts, max_per_rank * world_size, hidden].
        out_t = torch.empty(
            num_local_experts,
            max_per_rank * world_size,
            hidden,
            dtype=x.dtype,
            device=x.device,
        )
        output_nd = NDTensor.from_torch(self._fleet.group, out_t, _tag("TOKENS"))

        in_arr, in_n = _to_pointer_array([input_nd])
        out_arr, out_n = _to_pointer_array([output_nd])
        local_arr, local_n = _to_pointer_array([self._recv_count_nd])

        cfg = ncclEpDispatchConfig_t(round_scales=int(self._fleet.use_ue8m0))

        self._lib.ncclEpDispatch(
            self._handle,
            in_arr,
            in_n,
            out_arr,
            out_n,
            local_arr,
            local_n,
            int(self._staged),
            cfg,
            ctypes.c_void_p(self._stream),
        )
        # LL mode requires ncclEpComplete after dispatch.
        self._lib.ncclEpComplete(self._handle, None, ctypes.c_void_p(self._stream))

        # Keepalive for combine().
        self._dispatch_input_nd = input_nd
        self._dispatch_output_nd = output_nd
        self._dispatch_output_t = out_t
        # `_recv_count_t.sum().item()` is a host read that only synchronizes the
        # *current* PyTorch stream. The EP work was enqueued on self._stream
        # (which may be a caller-supplied HandleAlgoKnobUserStream different
        # from the current stream), so block on that stream first to ensure
        # ncclEpDispatch/Complete have populated _recv_count_t before we read.
        self._sync_stream()
        num_tokens = int(self._recv_count_t.sum().item())
        return DispatchOutput(expert_tensors=out_t, num_tokens=num_tokens)

    def _sync_stream(self) -> None:
        """Host-synchronize the CUDA stream the EP ops were enqueued on."""
        import torch

        torch.cuda.ExternalStream(self._stream).synchronize()

    # ----------------------------------------------------------------- combine

    # @flashinfer_api  # disabled per PR #3453 review
    def combine(self, params: CombineInputParams) -> CombineOutput:
        """Gather expert outputs back to the originating ranks (LL mode)."""
        import torch

        x = params.x[
            0
        ]  # expected: 3D [num_local_experts, max_per_rank * world, hidden]
        input_nd = NDTensor.from_torch(self._fleet.group, x, _tag("TOKENS"))

        # 2D combine output: [num_tokens, hidden]. num_tokens = topk_idx.size(0).
        num_tokens = self._topk_idx_nd.shape[0]
        out_t = (
            params.out
            if params.out is not None
            else torch.empty(
                num_tokens,
                self._fleet.params.token_hidden_size,
                dtype=x.dtype,
                device=x.device,
            )
        )
        output_nd = NDTensor.from_torch(self._fleet.group, out_t, _tag("TOKENS"))

        # Combine-local: topk_weights (fp32). Mandatory in LL mode.
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
        weights_nd = NDTensor.from_torch(
            self._fleet.group, weights, _tag("TOPK_WEIGHTS")
        )

        in_arr, in_n = _to_pointer_array([input_nd])
        out_arr, out_n = _to_pointer_array([output_nd])
        local_arr, local_n = _to_pointer_array([weights_nd])

        self._lib.ncclEpCombine(
            self._handle,
            in_arr,
            in_n,
            out_arr,
            out_n,
            local_arr,
            local_n,
            int(self._staged),
            None,  # config (reserved)
            ctypes.c_void_p(self._stream),
        )
        self._lib.ncclEpComplete(self._handle, None, ctypes.c_void_p(self._stream))

        # Keepalives.
        self._combine_input_nd = input_nd
        self._combine_output_nd = output_nd
        self._combine_weights_nd = weights_nd
        return CombineOutput(x=out_t)

    # ----------------------------------------------------------------- complete

    # @flashinfer_api  # disabled per PR #3453 review
    def complete(self) -> None:
        """No-op: ncclEpComplete is issued internally after dispatch/combine.

        Kept on the API for parity with the design's
        ``HandleAlgoKnobSplitOperation`` pattern, which is deferred for HT
        mode multi-handle pipelines.
        """

    def __del__(self) -> None:
        if not self._destroyed and self._handle is not None:
            with contextlib.suppress(Exception):
                self._lib.ncclEpHandleDestroy(self._handle)
            self._destroyed = True
