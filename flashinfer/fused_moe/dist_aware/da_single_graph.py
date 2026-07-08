"""CUDA graph helpers for DAKNNv2 MoE dispatch.

This module provides the capture-safe pieces used by
``flashinfer.fused_moe.core`` when it injects a DAKNNv2 decision + SWITCH +
TRT-LLM FP4 MoE bodies into an existing CUDA graph capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "capture_safe_trtllm_fp4_moe",
    "DAInlineGraphInjector",
]


def capture_safe_trtllm_fp4_moe(
    *,
    ffi_moe_op: Any,
    routing_input_mode: int,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    routing_logits: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    expert_weights: torch.Tensor,
    output: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    num_local_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    do_finalize: bool,
    enable_pdl: bool,
    activation_type: int,
    tactic: Sequence[int],
) -> None:
    """Capture-safe direct call to the TRT-LLM FP4 block-scale MoE launcher.

    This bypasses :func:`trtllm_fp4_block_scale_moe_op`'s Python wrapper so
    the call does **not** allocate tensors (``torch.empty``) or consult the
    AutoTuner — both of which would break ``cudaStreamBeginCaptureToGraph``
    capture. All tensor arguments must be pre-allocated persistent buffers
    whose addresses should be baked into the captured graph; ``tactic`` must
    be the pre-selected ``[tile_n, config]`` pair. Pass ``[-1, -1]`` to
    request the TRT-LLM default tactic.

    The call writes into ``output`` in-place; this function returns ``None``
    because callers capturing a graph only care about the kernel side-effects.

    See ``csrc/trtllm_fused_moe_kernel_launcher.cu`` for the C++ entry point
    (``trtllm_fp4_block_scale_moe`` in namespace scope, TVM-FFI exported).
    """
    tactic_list = [int(tactic[0]), int(tactic[1])]

    ffi_moe_op.trtllm_fp4_block_scale_moe(
        int(routing_input_mode),
        routing_logits,
        topk_ids,
        expert_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        None,
        int(num_experts),
        int(top_k),
        n_group,
        topk_group,
        int(intermediate_size),
        int(local_expert_offset),
        int(num_local_experts),
        routed_scaling_factor,
        int(routing_method_type),
        bool(do_finalize),
        bool(enable_pdl),
        int(activation_type),
        output,
        tactic_list,
        True,
        None,
    )


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
                    capture_safe_trtllm_fp4_moe(
                        ffi_moe_op=ffi_moe_op,
                        tactic=per_body_tactics[i], ...
                    )
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
        tile_sizes: Sequence[int],
        num_tokens_bucket: int,
        num_local_experts: int,
        local_expert_offset: int,
        top_k: int,
        expert_counts: Optional[torch.Tensor] = None,
        side_stream: Optional[torch.cuda.Stream] = None,
        routing_stream: Optional[torch.cuda.Stream] = None,
        pool_handle: Optional[Any] = None,
    ):
        """Context manager that sets up the inline switch and tears it down."""
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

        if expert_counts is None:
            ctx_id = int(
                self._ffi.da_inline_switch_begin_with_handle(
                    int(selector_handle),
                    topk_ids,
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
