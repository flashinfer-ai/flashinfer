"""NcclEpHandle — per-iteration handle over the ``nccl.ep`` v0.1.0 API."""

from __future__ import annotations

import contextlib
import os as _os
from time import perf_counter as _pc
from typing import TYPE_CHECKING, Sequence

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

_HP = _os.environ.get("EP_PROFILE_HOST") == "1"
_HP_SKIP = int(_os.environ.get("EP_PROFILE_SKIP", "15"))
_hprof: "dict[str, list]" = {}
_FAST = _os.environ.get("NV_FI_EP_FAST_PATH") == "1"


def _hp(name, t0):
    if not _HP:
        return None
    now = _pc()
    _hprof.setdefault(name, []).append((now - t0) * 1e6)
    return now


if _HP:
    import atexit
    from statistics import median

    @atexit.register
    def _dump_hprof():
        if _os.environ.get("RANK", "0") != "0" or not _hprof:
            return
        groups: "dict[str, list]" = {}
        for k, samples in _hprof.items():
            groups.setdefault(k.split(".")[0], []).append((k, samples[_HP_SKIP:]))
        print(
            f"\n=== EP_PROFILE_HOST (median host wall µs/call, rank 0, "
            f"skip first {_HP_SKIP}) ===",
            flush=True,
        )
        for grp, items in groups.items():
            n0 = len(items[0][1])
            gtot = sum(median(s) for _, s in items if s)
            print(f"  [{grp}] {gtot:8.1f} us/call total ({n0} samples)", flush=True)
            for k, s in items:
                if s:
                    print(
                        f"      {k.split('.', 1)[1]:18s} {median(s):8.2f} us",
                        flush=True,
                    )


if TYPE_CHECKING:
    from .fleet import NcclEpFleet


class NcclEpHandle(Handle):
    def __init__(
        self,
        fleet: "NcclEpFleet",
        params: HandleParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        import torch

        from .....config import EpAlgorithm, EpLayout

        self._fleet = fleet
        self._ep = fleet.nccl_ep
        self._handle_knobs = _index_knobs(algo_knobs)
        self._stream = self._knob_stream()
        self._staged = HandleAlgoKnobSplitOperation in self._handle_knobs
        self._destroyed = False

        self._is_ht = fleet.params.algorithm == EpAlgorithm.HIGH_THROUGHPUT
        self._is_rank_major = (not self._is_ht) and (
            fleet.params.layout is EpLayout.RANK_MAJOR
        )

        world_size = fleet.bootstrap.world_size
        self._world_size = world_size
        self._num_local_experts = fleet.params.num_experts // world_size

        topk_idx = params.topk_ids
        if topk_idx.dtype != torch.int64:
            topk_idx = topk_idx.to(torch.int64)
        self._topk_idx = topk_idx
        self._num_tokens_in = topk_idx.shape[0]
        self._top_k = topk_idx.shape[1]
        self._topk_idx_t = self._ep.Tensor(topk_idx)

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
            layout_info=None,
            config=None,
            stream=self._stream,
        )

    def _knob_stream(self) -> int:
        k = self._handle_knobs.get(HandleAlgoKnobUserStream)
        return int(k.stream) if k is not None else self._fleet.stream  # type: ignore[attr-defined]

    def dispatch(self, params: DispatchInputParams) -> DispatchOutput:
        x = params.x[0]
        if self._is_ht:
            return self._dispatch_ht(x)
        if self._is_rank_major:
            return self._dispatch_ll_rank_major(x)
        return self._dispatch_ll(x)

    def _dispatch_ll(self, x) -> DispatchOutput:
        import torch

        _t = _pc() if _HP else None
        world_size = self._fleet.bootstrap.world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size

        out_t = getattr(self, "_ll_recv_buf", None) if _FAST else None
        if out_t is None:
            out_t = torch.empty(
                self._num_local_experts,
                max_per_rank * world_size,
                hidden,
                dtype=x.dtype,
                device=x.device,
            )
            if _FAST:
                self._ll_recv_buf = out_t
        _t = _hp("ll_disp.alloc", _t)

        cache = getattr(self, "_ll_disp_cache", None) if _FAST else None
        if cache is None:
            outputs = self._ep.DispatchOutputs(tokens=self._ep.Tensor(out_t))
            layout_info = self._ep.LayoutInfo(
                expert_counters=self._ep.Tensor(self._recv_count_t)
            )
            config = self._ep.DispatchConfig(
                send_only=int(self._staged), round_scales=0
            )
            if _FAST:
                self._ll_disp_cache = (outputs, layout_info, config)
        else:
            outputs, layout_info, config = cache
        inputs = self._ep.DispatchInputs(tokens=self._ep.Tensor(x))
        _t = _hp("ll_disp.build_ffi_objs", _t)

        self._handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=config,
            stream=self._stream,
        )
        _t = _hp("ll_disp.ffi_dispatch", _t)
        self._handle.complete(stream=self._stream)
        _t = _hp("ll_disp.ffi_complete", _t)

        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_layout = layout_info
        self._dispatch_output_t = out_t

        if _FAST:

            def _num_tokens():
                torch.cuda.ExternalStream(self._stream).synchronize()
                return int(self._recv_count_t.sum().item())

            _t = _hp("ll_disp.defer_count", _t)
            return DispatchOutput(expert_tensors=out_t, num_tokens=_num_tokens)

        torch.cuda.ExternalStream(self._stream).synchronize()
        _t = _hp("ll_disp.synchronize", _t)
        num_tokens = int(self._recv_count_t.sum().item())
        _t = _hp("ll_disp.recvcount_sum_item", _t)
        return DispatchOutput(expert_tensors=out_t, num_tokens=num_tokens)

    def _dispatch_ll_rank_major(self, x) -> DispatchOutput:
        import torch

        world_size = self._world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size
        m = max_per_rank * world_size

        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NcclEpHandle RANK_MAJOR dispatch requires HandleAlgoKnobTopKWeights "
                "(rank-major dispatch sends per-token routing weights)."
            )
        weights = tw.weights  # type: ignore[attr-defined]
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        out_t = torch.empty(
            world_size, max_per_rank, hidden, dtype=x.dtype, device=x.device
        )
        out_w = torch.empty(
            world_size, max_per_rank, self._top_k, dtype=torch.float32, device=x.device
        )
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
        self._handle.complete(stream=self._stream)

        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_layout = layout_info
        self._dispatch_output_t = out_t
        self._dispatch_out_w = out_w
        self._dispatch_out_idx = out_idx

        torch.cuda.ExternalStream(self._stream).synchronize()
        num_tokens = int(self._recv_count_t.sum().item())
        return DispatchOutput(
            expert_tensors=out_t,
            num_tokens=num_tokens,
            recv_topk_idx=out_idx.reshape(m, self._top_k),
            recv_topk_weights=out_w.reshape(m, self._top_k),
        )

    def _dispatch_ht(self, x) -> DispatchOutput:
        import torch

        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size
        world = self._fleet.params.num_experts // self._num_local_experts
        num_recv = max_per_rank * world

        tw = self._handle_knobs.get(HandleAlgoKnobTopKWeights)
        if tw is None:
            raise ValueError(
                "NcclEpHandle HT dispatch requires HandleAlgoKnobTopKWeights "
                "(HT forward dispatch sends per-token routing weights)."
            )
        weights = tw.weights  # type: ignore[attr-defined]
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)

        _t = _pc() if _HP else None
        cached = getattr(self, "_ht_recv_bufs", None)
        if cached is None or cached[0].shape[0] != num_recv:
            out_t = torch.empty(num_recv, hidden, dtype=x.dtype, device=x.device)
            out_w = torch.empty(
                num_recv, self._top_k, dtype=torch.float32, device=x.device
            )
            out_idx = torch.empty(
                num_recv, self._top_k, dtype=torch.int64, device=x.device
            )
            self._ht_recv_bufs = (out_t, out_w, out_idx)
        else:
            out_t, out_w, out_idx = cached
        _t = _hp("ht_disp.alloc_cached", _t)

        cache = getattr(self, "_ht_disp_cache", None) if _FAST else None
        if cache is None:
            outputs = self._ep.DispatchOutputs(
                tokens=self._ep.Tensor(out_t),
                topk_weights=self._ep.Tensor(out_w),
                topk_idx=self._ep.Tensor(out_idx),
            )
            config = self._ep.DispatchConfig(
                send_only=int(self._staged), round_scales=0
            )
            weights_t = self._ep.Tensor(weights)
            if _FAST:
                self._ht_disp_cache = (outputs, config, weights_t)
        else:
            outputs, config, weights_t = cache
        inputs = self._ep.DispatchInputs(
            tokens=self._ep.Tensor(x), topk_weights=weights_t
        )
        _t = _hp("ht_disp.build_ffi_objs", _t)

        self._handle.dispatch(
            inputs, outputs, layout_info=None, config=config, stream=self._stream
        )
        _t = _hp("ht_disp.ffi_dispatch", _t)
        self._handle.complete(stream=self._stream)
        _t = _hp("ht_disp.ffi_complete", _t)

        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_output_t = out_t
        self._dispatch_out_w = out_w
        self._dispatch_out_idx = out_idx

        out_3d = out_t.view(world, max_per_rank, hidden)
        return DispatchOutput(
            expert_tensors=out_3d,
            num_tokens=num_recv,
            recv_topk_idx=out_idx,
            recv_topk_weights=out_w,
        )

    def combine(self, params: CombineInputParams) -> CombineOutput:
        import torch

        x = params.x[0]
        hidden = self._fleet.params.token_hidden_size
        out_t = (
            params.out
            if params.out is not None
            else torch.empty(
                self._num_tokens_in, hidden, dtype=x.dtype, device=x.device
            )
        )

        if self._is_ht:
            x2d = x.reshape(-1, hidden)
            cache = getattr(self, "_ht_comb_cache", None) if _FAST else None
            if cache is None or cache[2] is not out_t:
                outputs = self._ep.CombineOutputs(tokens=self._ep.Tensor(out_t))
                config = self._ep.CombineConfig(send_only=int(self._staged))
                if _FAST:
                    self._ht_comb_cache = (outputs, config, out_t)
            else:
                outputs, config, _ = cache
            inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x2d))
            self._handle.combine(inputs, outputs, config=config, stream=self._stream)
            self._handle.complete(stream=self._stream)
            self._combine_inputs = inputs
            self._combine_outputs = outputs
            self._combine_x2d = x2d
            return CombineOutput(x=out_t)

        if self._is_rank_major:
            inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x))
            outputs = self._ep.CombineOutputs(tokens=self._ep.Tensor(out_t))
            config = self._ep.CombineConfig(send_only=int(self._staged))
            self._handle.combine(inputs, outputs, config=config, stream=self._stream)
            if self._staged:
                self._handle.complete(stream=self._stream)
            self._combine_inputs = inputs
            self._combine_outputs = outputs
            return CombineOutput(x=out_t)

        cache = getattr(self, "_ll_comb_cache", None) if _FAST else None
        if cache is None:
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
            weights_t = self._ep.Tensor(weights)
            config = self._ep.CombineConfig(send_only=int(self._staged))
            if _FAST:
                self._ll_comb_cache = (weights, weights_t, config)
        else:
            weights, weights_t, config = cache

        inputs = self._ep.CombineInputs(tokens=self._ep.Tensor(x))
        outputs = self._ep.CombineOutputs(
            tokens=self._ep.Tensor(out_t),
            topk_weights=weights_t,
        )

        self._handle.combine(inputs, outputs, config=config, stream=self._stream)
        if self._staged:
            self._handle.complete(stream=self._stream)

        self._combine_inputs = inputs
        self._combine_outputs = outputs
        self._combine_weights = weights
        return CombineOutput(x=out_t)

    def complete(self) -> None:
        """No-op in non-staged LL mode."""

    def destroy(self) -> None:
        if not self._destroyed:
            with contextlib.suppress(Exception):
                self._handle.destroy()
            self._destroyed = True

    def __del__(self) -> None:
        self.destroy()
