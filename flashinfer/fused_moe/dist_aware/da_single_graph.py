"""CUDA graph helpers for DAKNNv2 MoE dispatch.

This module provides the precision-agnostic capture structure used to inject a
DAKNNv2 decision, conditional bodies, and routing work into an existing CUDA
graph capture. Precision-owned modules supply the body closures.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "DAInlineGraphInjector",
]


# Per-device caches. `inject()` is invoked from inside vLLM's CUDA graph
# capture; allocating a fresh side stream / pool handle each call accumulates
# ever-more resources and eventually puts the side stream into a state where
# `cudaStreamBeginCaptureToGraph` returns cudaErrorIllegalState (401). Reuse
# per device instead, which is safe: the inline SWITCH is always fully set up
# before the next `inject()` fires on the same device, so the side stream is
# guaranteed to be out of capture mode between calls.
_DA_INLINE_POOL_HANDLES: Dict[int, Any] = {}
_DA_INLINE_SIDE_STREAMS: Dict[int, "torch.cuda.Stream"] = {}
_DA_INLINE_ROUTING_STREAMS: Dict[int, "torch.cuda.Stream"] = {}


def _validated_capture_streams(
    *,
    device: torch.device,
    outer_stream: torch.cuda.Stream,
    side_stream: torch.cuda.Stream,
    routing_stream: torch.cuda.Stream,
    side_stream_supplied: bool = False,
    routing_stream_supplied: bool = False,
) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Validate the three capture streams, repairing only internal streams."""
    max_reacquire_attempts = 10
    outer = int(outer_stream.cuda_stream)
    for attempt in range(max_reacquire_attempts + 1):
        side_handle = int(side_stream.cuda_stream)
        routing_handle = int(routing_stream.cuda_stream)
        if len({outer, side_handle, routing_handle}) == 3:
            device_idx = int(device.index or 0)
            if not side_stream_supplied:
                _DA_INLINE_SIDE_STREAMS[device_idx] = side_stream
            if not routing_stream_supplied:
                _DA_INLINE_ROUTING_STREAMS[device_idx] = routing_stream
            return side_stream, routing_stream
        diagnostics = (
            f"outer={outer}, side={side_handle}, routing={routing_handle}, "
            f"reacquire_attempts={attempt}"
        )
        if side_stream_supplied and side_handle == outer:
            raise RuntimeError(
                "framework-supplied DA side_stream aliases the outer capture "
                f"stream; {diagnostics}"
            )
        if routing_stream_supplied and routing_handle == outer:
            raise RuntimeError(
                "framework-supplied DA routing_stream aliases the outer capture "
                f"stream; {diagnostics}"
            )
        if side_stream_supplied and routing_stream_supplied:
            raise RuntimeError(
                "framework-supplied DA auxiliary streams alias each other; "
                + diagnostics
            )
        if attempt == max_reacquire_attempts:
            raise RuntimeError(
                "unable to acquire three distinct DA capture streams; " + diagnostics
            )

        # Preserve supplied handles. If one supplied auxiliary aliases an
        # internal one, repair the internal counterpart.
        if side_handle == outer or (
            side_handle == routing_handle and routing_stream_supplied
        ):
            side_stream = torch.cuda.Stream(device=device)
        elif routing_handle == outer or side_handle == routing_handle:
            routing_stream = torch.cuda.Stream(device=device)
    raise AssertionError("unreachable stream-reacquisition state")


def capture_primitives(
    device: torch.device,
) -> Tuple[torch.cuda.Stream, torch.cuda.Stream, Any]:
    """Return stable per-device auxiliary streams and a graph memory pool."""
    device_idx = int(device.index or 0)
    side_stream = _DA_INLINE_SIDE_STREAMS.get(device_idx)
    if side_stream is None:
        side_stream = torch.cuda.Stream(device=device)
        _DA_INLINE_SIDE_STREAMS[device_idx] = side_stream
    routing_stream = _DA_INLINE_ROUTING_STREAMS.get(device_idx)
    if routing_stream is None:
        routing_stream = torch.cuda.Stream(device=device)
        _DA_INLINE_ROUTING_STREAMS[device_idx] = routing_stream
    pool_handle = _DA_INLINE_POOL_HANDLES.get(device_idx)
    if pool_handle is None:
        pool_handle = torch.cuda.graph_pool_handle()
        _DA_INLINE_POOL_HANDLES[device_idx] = pool_handle
    return side_stream, routing_stream, pool_handle


class DAInlineGraphInjector:
    """Inject a DA decision + SWITCH + per-tile MoE bodies into a torch
    stream-capture region that is already in progress.

    This class is meant to be called from inside a framework's
    ``torch.cuda.graph()`` capture (e.g. vLLM's piecewise graph capture). The emitted SWITCH node
    becomes part of *that* captured graph; no nested launches are ever
    involved, avoiding the capture-mode-forbidden ``cudaGraphLaunch``.

    Lifecycle::

        injector = DAInlineGraphInjector(ffi_moe_op)
        with injector.inject(
            topk_ids=scratch,
            tile_sizes=(8, 16, 32, 64, 128),
            num_tokens_bucket=bucket,
            num_local_experts=num_local,
            local_expert_offset=offset,
            top_k=top_k,
        ) as ctx:
            # num_bodies may exceed len(tile_sizes) when kNN exemplars
            # carry distinct (tile, config) pairs for the same tile.
            for i in range(ctx.num_bodies):
                with ctx.body(i):
                    capture_preallocated_precision_body(per_body_tactics[i])
        # On __exit__: SWITCH gets wired into the outer capture's dependency
        # chain so whatever vLLM captures next depends on SWITCH completion.

    Upload DAKNNv2 exemplars with ``da_upload_knn_exemplars`` before the first
    ``inject(bucket=...)`` so the decision kernel has selector state to read.
    """

    def __init__(self, ffi_moe_op: Any) -> None:
        self._ffi = ffi_moe_op

    @contextmanager
    def inject(
        self,
        *,
        selector_handle: int = 0,
        topk_ids: torch.Tensor,
        routing_input_mode: int,
        tile_sizes: Sequence[int],
        num_tokens_bucket: int,
        num_local_experts: int,
        local_expert_offset: int,
        top_k: int,
        expert_counts: Optional[torch.Tensor] = None,
        side_stream: Optional[torch.cuda.Stream] = None,
        routing_stream: Optional[torch.cuda.Stream] = None,
        pool_handle: Optional[Any] = None,
        side_stream_supplied: Optional[bool] = None,
        routing_stream_supplied: Optional[bool] = None,
    ):
        """Context manager that sets up the inline switch and tears it down."""
        if side_stream_supplied is None:
            side_stream_supplied = side_stream is not None
        if routing_stream_supplied is None:
            routing_stream_supplied = routing_stream is not None
        if not topk_ids.is_cuda:
            raise ValueError("topk_ids must be a CUDA tensor")
        if topk_ids.dtype != torch.int32:
            raise ValueError("topk_ids must have dtype torch.int32")
        if len(tile_sizes) == 0:
            raise ValueError("tile_sizes must be non-empty")
        if expert_counts is not None:
            if not expert_counts.is_cuda:
                raise ValueError("expert_counts must be a CUDA tensor")
            if expert_counts.dtype != torch.int32:
                raise ValueError("expert_counts must have dtype torch.int32")
            if expert_counts.ndim != 1:
                raise ValueError("expert_counts must be a 1D tensor")

        device_idx = topk_ids.device.index
        if torch.cuda.is_current_stream_capturing() and side_stream is None:
            raise RuntimeError(
                "active DA capture requires a pre-created side_stream from warmup"
            )
        if torch.cuda.is_current_stream_capturing() and routing_stream is None:
            raise RuntimeError(
                "active DA capture requires a pre-created routing_stream from warmup"
            )
        if torch.cuda.is_current_stream_capturing() and pool_handle is None:
            raise RuntimeError(
                "active DA capture requires a pre-created graph pool from warmup"
            )
        if side_stream is None:
            cached = _DA_INLINE_SIDE_STREAMS.get(device_idx)
            if cached is None:
                cached = torch.cuda.Stream(device=topk_ids.device)
                _DA_INLINE_SIDE_STREAMS[device_idx] = cached
            side_stream = cached
        if routing_stream is None:
            routing_stream = _DA_INLINE_ROUTING_STREAMS.get(device_idx)
            if routing_stream is None:
                routing_stream = torch.cuda.Stream(device=topk_ids.device)
                _DA_INLINE_ROUTING_STREAMS[device_idx] = routing_stream
        if pool_handle is None:
            cached_pool = _DA_INLINE_POOL_HANDLES.get(device_idx)
            if cached_pool is None:
                cached_pool = torch.cuda.graph_pool_handle()
                _DA_INLINE_POOL_HANDLES[device_idx] = cached_pool
            pool_handle = cached_pool

        if torch.cuda.is_current_stream_capturing():
            side_stream, routing_stream = _validated_capture_streams(
                device=topk_ids.device,
                outer_stream=torch.cuda.current_stream(topk_ids.device),
                side_stream=side_stream,
                routing_stream=routing_stream,
                side_stream_supplied=side_stream_supplied,
                routing_stream_supplied=routing_stream_supplied,
            )

        if expert_counts is None:
            ctx_id = int(
                self._ffi.da_inline_switch_begin_with_handle(
                    int(selector_handle),
                    topk_ids,
                    int(routing_input_mode),
                    int(num_tokens_bucket),
                    int(top_k),
                    int(num_local_experts),
                    int(local_expert_offset),
                    [int(t) for t in tile_sizes],
                    int(side_stream.cuda_stream),
                )
            )
        else:
            ctx_id = int(
                self._ffi.da_inline_switch_begin_from_counts_with_handle(
                    int(selector_handle),
                    topk_ids,
                    expert_counts,
                    int(routing_input_mode),
                    int(num_tokens_bucket),
                    int(top_k),
                    int(num_local_experts),
                    int(local_expert_offset),
                    [int(t) for t in tile_sizes],
                    int(side_stream.cuda_stream),
                )
            )

        num_bodies = int(self._ffi.da_inline_get_num_bodies(ctx_id))
        direct_mode = bool(self._ffi.da_inline_is_direct_mode(ctx_id))
        ctx = _DAInlineCtx(
            ffi=self._ffi,
            ctx_id=ctx_id,
            tile_sizes=tuple(int(t) for t in tile_sizes),
            num_bodies=num_bodies,
            side_stream=side_stream,
            routing_stream=routing_stream,
            device_idx=device_idx,
            pool_handle=pool_handle,
            direct_mode=direct_mode,
        )
        try:
            # Route allocations device-wide to our graph-private pool for
            # the duration of the body captures. This is how MoE internal
            # workspace (workspace_fc1/fc2, routing scratch) stays alive
            # across the outer graph's replays.
            torch._C._cuda_beginAllocateToPool(device_idx, pool_handle)
            try:
                yield ctx
            finally:
                torch._C._cuda_endAllocateToPool(device_idx, pool_handle)
            self._ffi.da_inline_switch_end(ctx_id)
        finally:
            self._ffi.da_inline_destroy(ctx_id)


@dataclass
class _DAInlineCtx:
    ffi: Any
    ctx_id: int
    tile_sizes: Tuple[int, ...]
    num_bodies: int
    side_stream: torch.cuda.Stream
    routing_stream: torch.cuda.Stream
    device_idx: int
    pool_handle: Any
    direct_mode: bool = False

    @contextmanager
    def routing_branch(self):
        """Capture routing-table construction into the outer graph.

        CUDA does not allow a node in a conditional body graph to depend
        directly on a node in the parent graph. The C++ side therefore joins
        this branch by adding its tail nodes as extra dependencies of the
        SWITCH node. That still overlaps routing-table construction with the
        selector, while keeping the selected body free to start FC1/FC2 with
        already-built metadata.

        In direct mode (single body, no SWITCH) this is a no-op: the caller's
        routing work runs on the outer stream as part of the normal capture.
        """
        if self.direct_mode:
            yield
            return
        self.ffi.da_inline_routing_begin_capture(
            self.ctx_id, int(self.routing_stream.cuda_stream)
        )
        with torch.cuda.stream(self.routing_stream):
            try:
                yield self.routing_stream
            finally:
                self.ffi.da_inline_routing_end_capture(self.ctx_id)

    @contextmanager
    def body(self, body_index: int):
        """Capture kernels launched on the side stream into body_graphs[i].

        In direct mode the single body is captured directly on the outer
        stream — no side stream, no SWITCH body subgraph.
        """
        if not (0 <= body_index < self.num_bodies):
            raise IndexError(
                f"body_index {body_index} out of range [0, {self.num_bodies})"
            )
        if self.direct_mode:
            yield
            return
        self.ffi.da_inline_body_begin_capture(self.ctx_id, int(body_index))
        with torch.cuda.stream(self.side_stream):
            try:
                yield self.side_stream
            finally:
                self.ffi.da_inline_body_end_capture(self.ctx_id)
