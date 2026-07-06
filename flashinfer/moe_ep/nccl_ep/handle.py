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
    HandleAlgoKnobNumReceivedTokens,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    _index_knobs,
)
from .._validators import MoEEpConfigError
from ..config import (
    CombineInputParams,
    CombineOutput,
    DispatchInputParams,
    DispatchOutput,
    HandleParams,
)
from ..handle import Handle

# from ...api_logging import flashinfer_api  # disabled per PR #3453 review

# --- EP_PROFILE_HOST: per-step host-wall burn-down of the dispatch/combine host
# path (zero overhead when unset). Accumulates host µs per labelled step and prints
# a grouped breakdown on rank 0 at exit. Used to attribute the FI host-call gap.
import os as _os
from time import perf_counter as _pc

_HP = _os.environ.get("EP_PROFILE_HOST") == "1"
# Drop the first N (cold/warmup) samples per step before summarizing.
_HP_SKIP = int(_os.environ.get("EP_PROFILE_SKIP", "15"))
_hprof: "dict[str, list]" = {}

# NV_FI_EP_FAST_PATH=1 enables the Python-side host-call optimizations (measured against
# the EP_PROFILE_HOST burn-down):
#   (1) cache the per-call FFI wrapper objects built over STABLE tensors (recv
#       buffer / counters / weights / configs); only the per-call input-token wrap
#       is rebuilt (it may alias a new tensor each call);
#   (2) cache the LL recv buffer instead of torch.empty() every dispatch.
# (The recv-count .sum().item() readback that used to dominate the LL host call is
# now removed unconditionally — DispatchOutput no longer carries num_tokens.)
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
    # @flashinfer_api  # disabled per PR #3453 review
    def __init__(
        self,
        fleet: "NcclEpFleet",
        params: HandleParams,
        algo_knobs: Sequence[AlgoKnob] = (),
    ) -> None:
        import torch

        from ..config import EpAlgorithm, EpLayout

        _t = _pc() if _HP else None
        self._fleet = fleet
        self._ep = fleet.nccl_ep
        # Cross-handle host-path cache (declared on NcclEpFleet). vLLM creates
        # a fresh Handle every MoE layer x step (routing binds at
        # create_handle), so per-handle caches never hit; anchoring them on the
        # long-lived Fleet makes the recv buffers, counter tensors and FFI
        # descriptor objects reusable across forwards. Tensor wrappers are
        # memoized by (data_ptr, dtype, shape), so an entry can only ever
        # describe the same memory layout it was built for; the dict is cleared
        # when it grows past a bound (entries are then rebuilt, which is always
        # safe — each handle only needs address stability within its own
        # lifetime).
        self._hot = fleet._hot_cache
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
        self._topk_idx_t = self._wrap(topk_idx)

        # Per-source counter the library writes at dispatch (LL): EXPERT_MAJOR
        # gets per-local-expert recv counts [num_local_experts]; RANK_MAJOR gets
        # per-source-rank token counts [world]. Fleet-cached; NOT re-zeroed
        # across forwards — the dispatch metadata fully overwrites every entry
        # (the same contract the NV_FI_EP_FAST_PATH per-handle reuse relies on).
        recv_count_len = world_size if self._is_rank_major else self._num_local_experts
        ck = ("recv_count", recv_count_len, topk_idx.device)
        self._recv_count_t = self._hot.get(ck)
        if self._recv_count_t is None:
            self._recv_count_t = torch.zeros(
                recv_count_len, dtype=torch.int32, device=topk_idx.device
            )
            self._hot[ck] = self._recv_count_t
        _t = _hp("hinit.setup", _t)

        if self._is_ht:
            layout = self._ep.Layout.FLAT
        elif self._is_rank_major:
            layout = self._ep.Layout.RANK_MAJOR
        else:
            layout = self._ep.Layout.EXPERT_MAJOR

        # GAP 3 (opt-in): expose HT's actual received-token count. When the caller
        # sets HandleAlgoKnobNumReceivedTokens, bind its target as
        # LayoutInfo.recv_total_counter so ncclEpCreateHandle -> ncclEpUpdateHandle
        # -> metadata preprocessing writes the actual total into it (nccl_ep.cc
        # ~2316-2357). This works in the static max_tokens mode too — only dynamic
        # *buffer sizing* is unsupported in v0.1, not the count readback — letting an
        # HT consumer trim compute to recv_x[:actual_recv] on the static buffer.
        # HT-only: LL rejects handle-time layout_info (nccl_ep.cc:2201), and without
        # the knob HT keeps the verified layout_info=None path unchanged.
        self._recv_total_target = None
        create_layout_info = None
        num_recv_knob = self._handle_knobs.get(HandleAlgoKnobNumReceivedTokens)
        if num_recv_knob is not None and self._is_ht:
            tgt = num_recv_knob.target  # type: ignore[attr-defined]
            if tgt.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "HandleAlgoKnobNumReceivedTokens.target must be int32 or int64, "
                    f"got {tgt.dtype}"
                )
            if tgt.numel() < 1:
                raise ValueError(
                    "HandleAlgoKnobNumReceivedTokens.target must have >= 1 element, "
                    f"got shape {tuple(tgt.shape)}"
                )
            self._recv_total_target = tgt
            create_layout_info = self._ep.LayoutInfo(
                recv_total_counter=self._ep.Tensor(tgt)
            )
        self._create_layout_info = create_layout_info  # keepalive across create

        self._handle = fleet.group.create_handle(
            layout,
            self._topk_idx_t,
            layout_info=create_layout_info,  # HT recv-count opt-in; None otherwise
            config=None,
            stream=self._stream,
        )
        _t = _hp("hinit.create_handle_c", _t)

    # ----------------------------------------------------------------- knobs

    def _knob_stream(self) -> int:
        k = self._handle_knobs.get(HandleAlgoKnobUserStream)
        return int(k.stream) if k is not None else self._fleet.stream  # type: ignore[attr-defined]

    # Only memoize wrappers of SMALL tensors: the wrapper keeps the torch tensor
    # alive, so caching wraps of large activations (e.g. 8k-token prefill inputs,
    # the [num_recv, hidden] combine views) pins GBs across allocator addresses
    # and OOMs at high --gpu-memory-utilization. Small tensors (weights, topk,
    # counters, decode-sized activations) are exactly the host-bound decode path
    # this cache exists for. 2 MiB * 256 entries caps pinning at 512 MiB worst
    # case (steady-state decode reuses a handful of addresses).
    _WRAP_MEMO_MAX_BYTES = 2 << 20
    _WRAP_MEMO_MAX_ENTRIES = 256

    def _wrap(self, t):
        """Memoized ``nccl.ep.Tensor`` wrapper (fleet-level, address-keyed).

        Building an FFI Tensor descriptor costs ~10us of host time; vLLM's
        allocator recycles workspace addresses across decode steps, so keying
        by (data_ptr, dtype, shape) hits almost always after warmup. A hit can
        never alias the wrong layout — a reused address with a different
        shape/dtype misses and builds a fresh wrapper. Large tensors are
        wrapped per call (see _WRAP_MEMO_MAX_BYTES).
        """
        if t.numel() * t.element_size() > self._WRAP_MEMO_MAX_BYTES:
            return self._ep.Tensor(t)
        hot = self._hot
        key = (t.data_ptr(), t.dtype, tuple(t.shape))
        w = hot.get(key)
        if w is None:
            if len(hot) > self._WRAP_MEMO_MAX_ENTRIES:
                hot.clear()
            w = self._ep.Tensor(t)
            hot[key] = w
        return w

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

        _t = _pc() if _HP else None
        world_size = self._fleet.bootstrap.world_size
        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size

        # Fleet-cached recv buffer (a fresh Handle is created every forward, so
        # per-handle caching never hits; the fleet persists).
        shape = (self._num_local_experts, max_per_rank * world_size, hidden)
        out_t = self._hot.get("ll_recv_buf")
        if (
            out_t is None
            or out_t.shape != shape
            or out_t.dtype != x.dtype
            or out_t.device != x.device
        ):
            out_t = torch.empty(*shape, dtype=x.dtype, device=x.device)
            self._hot["ll_recv_buf"] = out_t
        _t = _hp("ll_disp.alloc", _t)

        # Fleet-cached FFI descriptor objects over the STABLE tensors (recv
        # buffer / counters / config). Only the input-token wrap varies per call
        # (memoized by address in _wrap).
        cache = self._hot.get("ll_disp_ffi")
        if (
            cache is None
            or cache[0] is not out_t
            or cache[1] is not self._recv_count_t
            or cache[2] != self._staged
        ):
            outputs = self._ep.DispatchOutputs(tokens=self._ep.Tensor(out_t))
            layout_info = self._ep.LayoutInfo(
                expert_counters=self._ep.Tensor(self._recv_count_t)
            )
            config = self._ep.DispatchConfig(
                send_only=int(self._staged), round_scales=0
            )
            self._hot["ll_disp_ffi"] = (
                out_t,
                self._recv_count_t,
                self._staged,
                outputs,
                layout_info,
                config,
            )
        else:
            outputs, layout_info, config = cache[3], cache[4], cache[5]
        inputs = self._ep.DispatchInputs(tokens=self._wrap(x))
        _t = _hp("ll_disp.build_ffi_objs", _t)

        self._handle.dispatch(
            inputs,
            outputs,
            layout_info=layout_info,
            config=config,
            stream=self._stream,
        )
        _t = _hp("ll_disp.ffi_dispatch", _t)
        # Finish the dispatch (incl. the cross-rank RECEIVE) before the inner
        # compute reads the recv buffer. Required even in non-staged mode — the LL
        # dispatch is async, so without complete() the remote slots are still empty
        # and FFN(empty)=0 silently drops every remote-routed token's contribution.
        # Mirrors contrib/nccl_ep/ep_test.py, which calls complete() after dispatch
        # unconditionally.
        self._handle.complete(stream=self._stream)
        _t = _hp("ll_disp.ffi_complete", _t)

        # Keepalives for combine() / async safety.
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_layout = layout_info
        self._dispatch_output_t = out_t

        # The library wrote per-expert recv counts into self._recv_count_t (via the
        # LayoutInfo above). combine masks padding via the routing (not a count), so
        # we never read them host-side here; but we surface the device tensor for
        # consumers that want it (no forced sync unless they read it).
        return DispatchOutput(expert_tensors=out_t, expert_counts=self._recv_count_t)

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

        # src_rank_counters were written into self._recv_count_t (LayoutInfo above);
        # nothing consumes them, so we skip the device->host readback.
        # Flatten the rank-grouped routing to the [M, top_k] view the compute
        # bridge consumes (row-major, matching out_t.reshape(M, hidden)).
        return DispatchOutput(
            expert_tensors=out_t,
            recv_topk_idx=out_idx.reshape(m, self._top_k),
            recv_topk_weights=out_w.reshape(m, self._top_k),
            expert_counts=self._recv_count_t,  # per-source-rank counts [world]
        )

    def _dispatch_ht(self, x) -> DispatchOutput:
        """HT FLAT dispatch: tokens + topk_weights in, [num_recv, H] + recv
        topk_weights/idx out.  num_recv = max_tokens_per_rank * num_local_experts
        (per-expert blocks), reshaped to the 3D [L, cap, H] view the compute
        bridge expects (cap = max_tokens_per_rank)."""
        import torch

        max_per_rank = self._fleet.params.max_tokens_per_rank
        hidden = self._fleet.params.token_hidden_size
        # HT recv-buffer rows MUST equal the fleet's max_recv_tokens_per_rank (the
        # per-rank recv-slot budget the library sizes its internal HT staging
        # buffers to). Fleet sets that to max_tokens_per_rank * world_size; sizing
        # the dispatch output by max_tokens_per_rank * num_local_experts instead
        # overflows the library staging buffer whenever num_local_experts > world
        # (i.e. num_experts > world^2, e.g. 256 experts / 8 GPU → local_n=32),
        # surfacing as nccl_ep.cc:3269 'invalid argument' at combine (masked as a
        # 2nd-dispatch hang on the default stream). They coincide only when
        # num_local_experts == world (the rank-derived ep_test geometry), which is
        # why the 8-GPU/64-expert correctness test passed but the 256-expert
        # matrix did not. The uniform recv estimate is max_tokens_per_rank * top_k
        # (<= this budget for top_k <= world); we use the budget to stay byte-for-
        # byte consistent with the GroupConfig the library was created with.
        world = self._fleet.params.num_experts // self._num_local_experts
        num_recv = max_per_rank * world

        # The HT staging buffers (and this recv buffer) are sized to max_per_rank,
        # which the fleet clamps to the library's MAX_SUPPORTED_TOKENS_PER_RANK. A
        # forward that dispatches more than that per rank would overflow the staging
        # buffers (and previously hit a C++ abort at group-create for the un-clamped
        # value). Fail with an actionable error instead of corrupting memory.
        n_tokens = x.shape[0]
        if n_tokens > max_per_rank:
            raise MoEEpConfigError(
                f"nccl_ep HT dispatch received {n_tokens} tokens on this rank, "
                f"exceeding max_tokens_per_rank ({max_per_rank} = the library's "
                "MAX_SUPPORTED_TOKENS_PER_RANK). Reduce the per-forward token count "
                "per rank (e.g. vLLM --max-num-batched-tokens <= "
                f"{max_per_rank}), or use the low-latency algorithm."
            )

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
        #
        # Allocate the recv buffers ONCE and reuse them across dispatches: HT
        # switches to "cached" mode on the 2nd+ dispatch (handle state is reused,
        # cf. ep_bench.cu), which assumes the SAME output buffer pointers as the
        # first dispatch. Fresh torch.empty buffers each call gave the cached
        # dispatch new addresses and deadlocked the next collective.
        _t = _pc() if _HP else None
        cached = self._hot.get("ht_recv_bufs")
        if (
            cached is None
            or cached[0].shape[0] != num_recv
            or cached[1].shape[1] != self._top_k
            or cached[0].dtype != x.dtype
            or cached[0].device != x.device
        ):
            out_t = torch.empty(num_recv, hidden, dtype=x.dtype, device=x.device)
            out_w = torch.empty(
                num_recv, self._top_k, dtype=torch.float32, device=x.device
            )
            out_idx = torch.empty(
                num_recv, self._top_k, dtype=torch.int64, device=x.device
            )
            self._hot["ht_recv_bufs"] = (out_t, out_w, out_idx)
        else:
            out_t, out_w, out_idx = cached
        _t = _hp("ht_disp.alloc_cached", _t)

        # Fleet-cached output wraps (over the cached recv bufs) + config; the
        # per-call input-token and weights wraps go through the _wrap memo.
        cache = self._hot.get("ht_disp_ffi")
        if cache is None or cache[0] is not out_t or cache[1] != self._staged:
            outputs = self._ep.DispatchOutputs(
                tokens=self._ep.Tensor(out_t),
                topk_weights=self._ep.Tensor(out_w),
                topk_idx=self._ep.Tensor(out_idx),
            )
            config = self._ep.DispatchConfig(
                send_only=int(self._staged), round_scales=0
            )
            self._hot["ht_disp_ffi"] = (out_t, self._staged, outputs, config)
        else:
            outputs, config = cache[2], cache[3]
        inputs = self._ep.DispatchInputs(
            tokens=self._wrap(x), topk_weights=self._wrap(weights)
        )
        _t = _hp("ht_disp.build_ffi_objs", _t)

        self._handle.dispatch(
            inputs, outputs, layout_info=None, config=config, stream=self._stream
        )
        _t = _hp("ht_disp.ffi_dispatch", _t)
        # Finish the dispatch (incl. cross-rank RECEIVE) before compute reads the
        # recv buffer — unconditional, matching the LL paths and ep_test.py.
        self._handle.complete(stream=self._stream)
        _t = _hp("ht_disp.ffi_complete", _t)

        # Keepalives.
        self._dispatch_inputs = inputs
        self._dispatch_outputs = outputs
        self._dispatch_output_t = out_t
        self._dispatch_out_w = out_w
        self._dispatch_out_idx = out_idx

        # HT FLAT recv is token-major (num_recv = max_per_rank * world rows); view
        # it as [world, max_per_rank, hidden] — the same layout the RANK_MAJOR
        # compute bridge consumes (HT routes through build_activation_pack_rank_major).
        # Equals the old [num_local_experts, max_per_rank, hidden] only when
        # num_local_experts == world; combine flattens it back to 2D regardless.
        out_3d = out_t.view(world, max_per_rank, hidden)
        return DispatchOutput(
            expert_tensors=out_3d,
            recv_topk_idx=out_idx,
            recv_topk_weights=out_w,
            # Populated at handle-create when the caller opted in via
            # HandleAlgoKnobNumReceivedTokens; None otherwise. Lets an HT consumer
            # trim compute to recv_x[:recv_total_counter] on the static buffer.
            recv_total_counter=self._recv_total_target,
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

        _t = _pc() if _HP else None
        if self._is_ht:
            # HT FLAT combine: input is the 2D [num_recv, hidden] expert output;
            # FWD combine takes NO topk_weights (routing weights were captured at
            # dispatch). Flatten the bridge's 3D [L, cap, H] view back to 2D.
            x2d = x.reshape(-1, hidden)
            # (2) cache output wrap + config (guarded by out_t identity); rebuild
            # only the per-call input wrap (x2d is a fresh view each call).
            ck = ("ht_comb_cfg", self._staged)
            config = self._hot.get(ck)
            if config is None:
                config = self._ep.CombineConfig(send_only=int(self._staged))
                self._hot[ck] = config
            outputs = self._ep.CombineOutputs(tokens=self._wrap(out_t))
            inputs = self._ep.CombineInputs(tokens=self._wrap(x2d))
            _t = _hp("ht_comb.build_ffi_objs", _t)
            self._handle.combine(inputs, outputs, config=config, stream=self._stream)
            _t = _hp("ht_comb.ffi_combine", _t)
            # HT requires complete() after EVERY combine (unconditional, matching
            # ep_test.py's cached-mode pairing dispatch->complete->combine->complete).
            # The HT combine leaves a pending GIN/scan op on the handle; without
            # draining it here the next dispatch on this handle hangs (the repeated-
            # op deadlock seen across MoEEpLayer.forward iterations and the matrix
            # benchmark's measure loop). LL self-drains, so this is HT-specific.
            self._handle.complete(stream=self._stream)
            _t = _hp("ht_comb.ffi_complete", _t)
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

        # LL EXPERT_MAJOR combine: weights applied on the receive side. The
        # weights tensor changes every forward (per-step routing), but its
        # allocator address recycles across decode steps — the _wrap memo makes
        # the descriptor build ~free. The config is static per staged-mode.
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
        weights_t = self._wrap(weights)
        ck = ("ll_comb_cfg", self._staged)
        config = self._hot.get(ck)
        if config is None:
            config = self._ep.CombineConfig(send_only=int(self._staged))
            self._hot[ck] = config
        inputs = self._ep.CombineInputs(tokens=self._wrap(x))
        outputs = self._ep.CombineOutputs(
            tokens=self._wrap(out_t),
            topk_weights=weights_t,
        )
        _t = _hp("ll_comb.build_ffi_objs", _t)

        self._handle.combine(inputs, outputs, config=config, stream=self._stream)
        _t = _hp("ll_comb.ffi_combine", _t)
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
            _t = _pc() if _HP else None
            with contextlib.suppress(Exception):
                self._handle.destroy()
            self._destroyed = True
            _hp("hdestroy.destroy_c", _t)

    def __del__(self) -> None:
        self.destroy()
