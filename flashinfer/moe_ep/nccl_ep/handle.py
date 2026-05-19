"""NcclEpHandle — per-iteration ncclEpHandle_t wrapper."""

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
from .ndtensor import NDTensor, get_nccl_lib

if TYPE_CHECKING:
    from .fleet import NcclEpFleet


# Tag constants imported lazily because nccl_ep load happens after the
# package import.
def _tag(name: str) -> int:
    from nccl_ep import ncclEpTensorTag_t  # type: ignore[import-not-found]

    return getattr(ncclEpTensorTag_t, f"NCCL_EP_TENSOR_TAG_{name}")


class NcclEpHandle(Handle):
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

        # Build the topk-idx NDTensor handle (borrows from the input tensor).
        self._topk_idx_nd = NDTensor.from_torch(
            fleet.group, params.topk_ids, _tag("TOPK_IDX")
        )

        local_tensors = self._build_handle_local_tensors()
        self._local_nds = local_tensors  # keepalives

        lib = get_nccl_lib()
        local_arr = self._build_handle_array(local_tensors)
        self._handle = lib.ncclEpCreateHandle(
            fleet.group,
            self._topk_idx_nd.handle,
            None,  # config (reserved)
            ctypes.c_void_p(self._stream),
            local_tensors=local_arr,
            use_fp8=fleet.use_fp8,
        )

    # ----------------------------------------------------------------- knobs

    def _knob_stream(self) -> int:
        k = self._handle_knobs.get(HandleAlgoKnobUserStream)
        return int(k.stream) if k is not None else self._fleet.stream  # type: ignore[attr-defined]

    def _build_handle_local_tensors(self) -> list[NDTensor]:
        """Local tensors passed at handle-create time.

        For HT mode, RECV_EXPERT_COUNTER may be required; for LL mode it
        must be None. We expose it via HandleAlgoKnobNumReceivedTokens —
        when set, the user-provided tensor becomes the counter target.
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

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _build_handle_array(nds: Sequence[NDTensor]):
        """Return a ctypes array of ncclNDTensor_t* or None for empty."""
        if not nds:
            return None
        return [nd.handle for nd in nds]

    @staticmethod
    def _to_pointer_array(nds: Sequence[NDTensor]):
        """ctypes array suitable for the dispatch/combine input/output args."""
        from nccl_ep import ncclNDTensor_t  # type: ignore[import-not-found]

        arr = (ncclNDTensor_t * len(nds))()
        for i, nd in enumerate(nds):
            arr[i] = nd.handle
        return arr, len(nds)

    # ----------------------------------------------------------------- dispatch

    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        """Send-and-recv dispatch of token tensors."""
        lib = get_nccl_lib()
        from nccl_ep import ncclEpDispatchConfig_t  # type: ignore[import-not-found]

        import torch

        inputs = [
            NDTensor.from_torch(self._fleet.group, t, _tag("TOKENS")) for t in params.x
        ]
        # Pre-allocate output torch tensors so as_torch() on the returned
        # NDTensors is the cheap borrowing path. Library-allocated mode
        # (data=nullptr) needs __cuda_array_interface__ round-trip which
        # doesn't work for bfloat16 via plain torch.as_tensor — borrowing
        # avoids the issue.
        out_torch = [
            torch.empty(
                self._fleet.params.max_tokens_per_rank,
                self._fleet.params.token_hidden_size,
                dtype=t.dtype,
                device=t.device,
            )
            for t in params.x
        ]
        outputs = [
            NDTensor.from_torch(self._fleet.group, ot, _tag("TOKENS"))
            for ot in out_torch
        ]
        in_arr, in_n = self._to_pointer_array(inputs)
        out_arr, out_n = self._to_pointer_array(outputs)
        local_nds: list[NDTensor] = []  # dispatch-local (SCALES in FP8 mode; deferred)
        local_arr, local_n = (
            (None, 0) if not local_nds else self._to_pointer_array(local_nds)
        )

        cfg = ncclEpDispatchConfig_t(round_scales=int(self._fleet.use_ue8m0))

        lib.ncclEpDispatch(
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

        # Recv count: if max_tokens_per_rank > 0, use it directly; otherwise
        # query via ncclEpHandleGetNumRecvTokens (HT auto-size mode).
        if self._fleet.params.max_tokens_per_rank > 0:
            n = self._fleet.params.max_tokens_per_rank
        else:
            n = lib.ncclEpHandleGetNumRecvTokens(self._handle)

        # Keep input/output NDTensors alive for the duration of this Handle
        # so the storage isn't freed before combine().
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        return DispatchOutput(expert_tensors=outputs[0].as_torch(), num_tokens=int(n))

    # ----------------------------------------------------------------- combine

    def combine(self, params: CombineInputParams) -> CombineOutput:
        """Gather expert outputs back to the originating ranks."""
        lib = get_nccl_lib()

        inputs = [
            NDTensor.from_torch(self._fleet.group, t, _tag("TOKENS")) for t in params.x
        ]
        out_t = params.out if params.out is not None else None
        if out_t is None:
            # Allocate a contiguous output tensor of the same shape as input[0].
            import torch

            out_t = torch.empty_like(params.x[0])

        outputs = [NDTensor.from_torch(self._fleet.group, out_t, _tag("TOKENS"))]
        in_arr, in_n = self._to_pointer_array(inputs)
        out_arr, out_n = self._to_pointer_array(outputs)

        # Combine-local tensors: topk_weights when provided.
        local_nds: list[NDTensor] = []
        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is not None:
            local_nds.append(
                NDTensor.from_torch(
                    self._fleet.group,
                    tw.weights,  # type: ignore[attr-defined]
                    _tag("TOPK_WEIGHTS"),
                )
            )
        local_arr, local_n = (
            (None, 0) if not local_nds else self._to_pointer_array(local_nds)
        )

        lib.ncclEpCombine(
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

        self._combine_inputs = inputs
        self._combine_outputs = outputs
        return CombineOutput(x=out_t)

    # ----------------------------------------------------------------- complete

    def complete(self) -> None:
        if not self._staged:
            return
        lib = get_nccl_lib()
        lib.ncclEpComplete(self._handle, None, ctypes.c_void_p(self._stream))

    def __del__(self) -> None:
        if not self._destroyed and self._handle is not None:
            with contextlib.suppress(Exception):
                lib = get_nccl_lib()
                lib.ncclEpHandleDestroy(self._handle)
            self._destroyed = True
