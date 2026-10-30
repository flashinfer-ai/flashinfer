# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused bf16 -> quant + routing-repack staging (``src.inputs_process``).

One ``DataPreprocess`` launch replaces the multi-kernel torch staging path
(fp32 upcast + ~a dozen elementwise/reduce launches + per-tensor copies) that
`stage_mega_moe_inputs` used per forward: the kernel reads the bf16 row once,
quantizes it in smem (NVFP4 per-16 E4M3 scales or MXFP4/MXFP8 per-32 E8M0), and
repacks routing to the workspace's integer-ID/fp32 layout.

Compiled stagers are cached by geometry, quantization mode and tensor types
and layouts. Launch bindings are cached by data pointers, token count, scales
and stream. Compilation must happen during warmup, before graph capture.

NVFP4 uses the offline (calibrated constant) scale mode; the online amax mode
exists in the kernel but is not wired here.
"""

from __future__ import annotations

import dataclasses
from functools import cache
from typing import Any, Optional

import torch

from .comm import ensure_not_capturing

_QUANT_TYPES = ("nvfp4", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2")
_MAX_FUSED_TAIL_ELEMENTS_PER_THREAD = 16


@dataclasses.dataclass
class _CompiledStager:
    dp: Any
    compiled: Any = None
    launch_key: Optional[tuple] = None
    launch_args: Optional[tuple] = None
    launch_kwargs: Optional[dict] = None


# Explicit dict rather than functools.cache: a miss must run the
# ensure_not_capturing guard (a hit must not), and the entry is a mutable
# launch-args cache, not a pure function value.
_STAGERS: dict[tuple, _CompiledStager] = {}

# topk_idx_out.data_ptr() -> num_tokens staged by the last fused call into
# that buffer (tail-fill memoization; see the fill logic below).
_LAST_STAGED_N: dict[int, int] = {}

# Buffers that have ever been staged inside a CUDA graph capture: replays
# bypass host code, so the memo is permanently untrustworthy for them.
_GRAPH_CAPTURED_BUFFERS: set[int] = set()


@cache
def _int32_routing_stager_class(
    mxfp8_normal_scales: bool = False,
    num_experts: Optional[int] = None,
    fuse_tail_mask: bool = False,
):
    """Adapt routing and optionally apply the normal-FP32 MXFP8 scale floor."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from cutlass._mlir.dialects import cf
    from cutlass.cutlass_dsl import dsl_user_op
    from src.inputs_process import DataPreprocess, Fp32Max, cvt_f32_to_f8_to_f32

    @dsl_user_op
    def assert_valid_route(valid, *, loc=None, ip=None):
        cf.assert_(
            valid.ir_value(),
            "MegaMoE routes require unique expert IDs in [0, E), or -1, "
            "and finite scores for valid routes",
            loc=loc,
            ip=ip,
        )

    class Int32RoutingDataPreprocess(DataPreprocess):
        @cute.jit
        def _repack_routing(
            self,
            token_idx,
            tid,
            topk_idx,
            topk_weights,
            token_padding_info,
            topk_idx_output,
            topk_weights_output,
        ):
            """Store int32 routes and zero masked scores for one live token.

            Optional checks reject invalid routes before narrowing IDs. When
            enabled, live CTAs also mask tails within the per-thread work limit.
            """
            if tid < cutlass.Int32(self.topk):
                is_padding = False
                if cutlass.const_expr(token_padding_info is not None):
                    is_padding = token_padding_info[token_idx] != cutlass.Int32(0)
                if cutlass.const_expr(num_experts is not None):
                    if not is_padding:
                        # Check before narrowing: an invalid int64 ID must not
                        # wrap into an apparently valid int32 route.
                        expert = cutlass.Int64(topk_idx[token_idx, tid])
                        score = cutlass.Float32(topk_weights[token_idx, tid])
                        valid = (expert >= -1) & (expert < num_experts)
                        finite = (
                            (score == score)
                            & (score < float("inf"))
                            & (score > -float("inf"))
                        )
                        valid = valid & ((expert == -1) | finite)
                        for other in cutlass.range_constexpr(self.topk):
                            if other < tid:
                                valid = valid & ~(
                                    (expert >= 0)
                                    & (
                                        expert
                                        == cutlass.Int64(topk_idx[token_idx, other])
                                    )
                                )
                        assert_valid_route(valid)
                idx = (
                    cutlass.Int32(-1)
                    if is_padding
                    else cutlass.Int32(topk_idx[token_idx, tid])
                )
                weight = (
                    cutlass.Float32(0.0)
                    if idx == -1
                    else cutlass.Float32(topk_weights[token_idx, tid])
                )
                topk_idx_output[token_idx, tid] = idx
                topk_weights_output[token_idx, tid] = weight

            if cutlass.const_expr(fuse_tail_mask):
                # Share the tail among the live-row CTAs. No activation or
                # score is read for padding, and every captured call masks the
                # full tail even after a graph with a larger live batch.
                live = cutlass.Int64(topk_idx.shape[0])
                workers = live * self._threads_per_cta
                worker = cutlass.Int64(
                    token_idx
                ) * self._threads_per_cta + cutlass.Int64(tid)
                tail_elements = (
                    cutlass.Int64(topk_idx_output.shape[0]) - live
                ) * self.topk
                offset = worker
                # Tiny live batches in a large workspace use the parallel
                # standalone fill instead of serializing a long tail here.
                if tail_elements <= workers * _MAX_FUSED_TAIL_ELEMENTS_PER_THREAD:
                    while offset < tail_elements:
                        row = Int32(live + offset // self.topk)
                        col = Int32(offset % self.topk)
                        topk_idx_output[row, col] = Int32(-1)
                        offset += workers

    if not mxfp8_normal_scales:
        return Int32RoutingDataPreprocess

    class NormalScaleMxDataPreprocess(Int32RoutingDataPreprocess):
        def __init__(self, topk, hidden, quant_type):
            # MXFP4 shares the per-32 E8M0 launch layout with MXFP8.
            super().__init__(
                topk, hidden, "mxfp8_e4m3" if quant_type == "mxfp4" else quant_type
            )
            self.quant_type = quant_type
            if quant_type == "mxfp4":
                self.quant_dtype = cutlass.Float4E2M1FN
                self._mxfp8_data_rcp_limit = 1.0 / 6.0

        @cute.kernel
        def mxfp8_quant_and_process_impl(
            self,
            activation_bf16,
            topk_idx,
            topk_weights,
            token_padding_info,
            activation_quant,
            activation_sf,
            topk_idx_output,
            topk_weights_output,
        ):
            # Each lane stores 16 bytes: a full MXFP4 scale block or half an
            # MXFP8 block. Only MXFP8 needs a reduction between adjacent lanes.
            threads: cutlass.Constexpr[int] = self._threads_per_cta
            hidden: cutlass.Constexpr[int] = self.hidden
            load_vec: cutlass.Constexpr[int] = 8  # 8 bf16 = 16 B per cp.async
            quant_vec: cutlass.Constexpr[int] = 128 // self.quant_dtype.width
            chunks_per_scale: cutlass.Constexpr[int] = 32 // quant_vec
            load_rounds_per_quant_round: cutlass.Constexpr[int] = quant_vec // load_vec
            num_chunks: cutlass.Constexpr[int] = hidden // load_vec
            num_quant_chunks: cutlass.Constexpr[int] = hidden // quant_vec
            chunks_per_thread: cutlass.Constexpr[int] = (
                num_chunks + threads - 1
            ) // threads
            quant_chunks_per_thread: cutlass.Constexpr[int] = (
                num_quant_chunks + threads - 1
            ) // threads
            num_groups: cutlass.Constexpr[int] = (
                chunks_per_thread + load_rounds_per_quant_round - 1
            ) // load_rounds_per_quant_round
            token_idx = cute.arch.block_idx()[0]
            tid = cute.arch.thread_idx()[0]

            data_rcp_limit = Float32(self._mxfp8_data_rcp_limit)
            fp32_max = Fp32Max

            smem = cutlass.utils.SmemAllocator()
            smem_row = smem.allocate_tensor(
                cutlass.BFloat16,
                cute.make_layout(hidden),
                1024,
                swizzle=cute.make_swizzle(1, 4, 3),
            )
            g_chunks = cute.zipped_divide(activation_bf16[token_idx, None], (load_vec,))
            s_chunks = cute.zipped_divide(smem_row, (load_vec,))
            g2s_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                cutlass.BFloat16,
                num_bits_per_copy=load_vec * 16,
            )
            # Commit the loads for each output vector as one group.
            for i in cutlass.range_constexpr(chunks_per_thread):
                c = tid + Int32(i * threads)
                if c < Int32(num_chunks):
                    cute.copy(
                        g2s_atom,
                        self._mark_alignment(g_chunks[(None,), (c,)], 16),
                        s_chunks[(None,), (c,)],
                    )
                if cutlass.const_expr(
                    (i + 1) % load_rounds_per_quant_round == 0
                    or i == chunks_per_thread - 1
                ):
                    cute.arch.cp_async_commit_group()

            s_quant_chunk = cute.zipped_divide(smem_row, (quant_vec,))
            q_quant_chunk = cute.zipped_divide(
                activation_quant[token_idx, None], (quant_vec,)
            )
            lds_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.quant_dtype, num_bits_per_copy=128
            )
            for j in cutlass.range_constexpr(quant_chunks_per_thread):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(num_groups)
                cute.arch.sync_threads()
                quant_chunk_idx = tid + Int32(j * threads)
                values = cute.make_rmem_tensor((quant_vec,), Float32)
                local_absmax = Float32(0.0)
                if quant_chunk_idx < Int32(num_quant_chunks):
                    quant_chunk_bf16 = cute.make_rmem_tensor(
                        (quant_vec,), cutlass.BFloat16
                    )
                    cute.copy(
                        lds_atom,
                        s_quant_chunk[(None,), (quant_chunk_idx,)],
                        quant_chunk_bf16,
                    )
                    values.store(quant_chunk_bf16.load().to(Float32))
                    for i in cutlass.range_constexpr(quant_vec):
                        local_absmax = cute.arch.fmax(
                            local_absmax, cute.arch.fmax(values[i], -values[i])
                        )

                block_absmax = local_absmax
                if cutlass.const_expr(chunks_per_scale == 2):
                    paired_absmax = Float32(
                        cute.arch.shuffle_sync_bfly(local_absmax, offset=1)
                    )
                    block_absmax = cute.arch.fmax(local_absmax, paired_absmax)
                if quant_chunk_idx < Int32(num_quant_chunks):
                    # Per-32 E8M0 (power-of-2) scale, rounded up (cvt.rp) so the block
                    # fits the quantized range; rcp is exact for a power of two.
                    # The Rubin Torch quantizer clamps to normal FP32 scales.
                    unrounded_scale = cute.arch.fmax(
                        block_absmax * data_rcp_limit, Float32(2.0**-126)
                    )
                    scale_f32 = Float32(
                        cvt_f32_to_f8_to_f32(unrounded_scale, cutlass.Float8E8M0FNU)
                    )
                    sf_e8m0 = scale_f32.to(cutlass.Float8E8M0FNU)
                    acc_scale = cute.arch.fmin(
                        cute.arch.rcp_approx(scale_f32), fp32_max
                    )

                    scaled = cute.make_rmem_tensor((quant_vec,), Float32)
                    for i in cutlass.range_constexpr(quant_vec):
                        scaled[i] = values[i] * acc_scale
                    data = cute.make_rmem_tensor((quant_vec,), self.quant_dtype)
                    data.store(scaled.load().to(self.quant_dtype))

                    cute.copy(
                        store_atom,
                        data,
                        self._mark_alignment(
                            q_quant_chunk[(None,), (quant_chunk_idx,)], 16
                        ),
                    )
                    if quant_chunk_idx % Int32(chunks_per_scale) == Int32(0):
                        activation_sf[
                            token_idx, (0, quant_chunk_idx // Int32(chunks_per_scale))
                        ] = sf_e8m0

            self._repack_routing(
                token_idx,
                tid,
                topk_idx,
                topk_weights,
                token_padding_info,
                topk_idx_output,
                topk_weights_output,
            )

    return NormalScaleMxDataPreprocess


def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def note_staged_tokens(topk_idx_out: torch.Tensor, num_tokens: int) -> None:
    """Record a non-fused staging into ``topk_idx_out`` (tail-fill memo).

    The torch fallback paths stage live rows and re-mask the full tail; they
    must update the memo so a later fused call cannot skip a fill over rows
    the fallback left live. Unknown buffers default to "assume fully live"
    (prev_n = capacity), which is always safe. Captured writers also disable
    tail memoization because replays update routes without updating the memo.
    """
    ptr = topk_idx_out.data_ptr()
    if topk_idx_out.is_cuda and torch.cuda.is_current_stream_capturing():
        _GRAPH_CAPTURED_BUFFERS.add(ptr)
    _LAST_STAGED_N[ptr] = num_tokens


def staged_tokens(topk_idx_out: torch.Tensor) -> Optional[int]:
    """Live-token count of the last staging into this buffer, if known."""
    return _LAST_STAGED_N.get(topk_idx_out.data_ptr())


def forget_staged_tokens(topk_idx_out: torch.Tensor) -> None:
    """Evict a buffer from the staging memos at workspace teardown.

    The symmetric heap (and the caching allocator) reuse freed addresses, so
    a later workspace can land on this pointer and would otherwise inherit a
    stale live-count or graph-captured mark from the destroyed buffer.
    """
    ptr = topk_idx_out.data_ptr()
    _LAST_STAGED_N.pop(ptr, None)
    _GRAPH_CAPTURED_BUFFERS.discard(ptr)


def fused_quant_stage_supported(
    hidden_states: torch.Tensor, quant_type: str = "nvfp4"
) -> bool:
    """True when the fused kernel can take this activation tensor.

    The DSL launcher requires 16-byte-aligned data pointers; torch allocations
    satisfy this, but sliced/offset views may not — callers fall back to their
    torch staging path in that case.

    The fused kernel also stages the SF plane at its exact unpadded width, so
    ``hidden // sf_vec`` must be a multiple of 4 (the buffer's round-up-to-4
    SF padding must be zero): hidden % 64 for nvfp4 (sf_vec 16), hidden % 128
    for mxfp8 (sf_vec 32). 128-misaligned mxfp8 shapes (gpt-oss: 2880) fall
    back to torch staging rather than failing.
    """
    sf_align = 64 if quant_type == "nvfp4" else 128
    return hidden_states.data_ptr() % 16 == 0 and hidden_states.shape[1] % sf_align == 0


def fused_quant_stage(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_out: torch.Tensor,
    x_sf_out: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    quant_type: str,
    norm_const: Optional[float] = None,
    scale_multiplier: Optional[float] = None,
    mxfp8_normal_scales: bool = False,
    num_experts: Optional[int] = None,
    fuse_tail_mask: bool = False,
) -> None:
    """Quantize + stage one batch into the mega symm-buffer views.

    ``hidden_states`` is the live ``(n, hidden)`` bf16 batch; the ``*_out``
    tensors are the full-capacity buffer views (``x``/``x_sf``/``topk_idx``/
    ``topk_weights`` of a mega symm buffer, or plain CUDA tensors with the
    same dtypes). Rows ``[:n]`` are staged and the ``topk_idx`` capacity tail
    is re-masked to ``-1``, matching the torch staging path's contract.

    ``norm_const`` is the NVFP4 offline per-tensor scale (required for
    ``quant_type="nvfp4"``, rejected otherwise).
    ``scale_multiplier`` optionally supplies the rounded ``norm_const / 6``
    factor used by the caller's NVFP4 quantization contract.
    ``mxfp8_normal_scales`` clamps MXFP8 scales to the normal FP32 range,
    matching Rubin's Torch quantizer; other callers retain the full E8M0 range.
    MXFP4 always uses normal FP32 scales.
    ``num_experts`` optionally checks route bounds, uniqueness and finite
    unmasked scores in the same kernel, before narrowing IDs to int32.
    ``fuse_tail_mask`` additionally masks all trailing routing rows in that
    launch when the live CTAs have enough parallelism. Large tails retain a
    separate fill. It requires normal-scale MXFP4/MXFP8 with int32 route storage.
    """
    if quant_type not in _QUANT_TYPES:
        raise ValueError(
            f"quant_type must be one of {_QUANT_TYPES}, got {quant_type!r}"
        )
    normal_scales = mxfp8_normal_scales or quant_type == "mxfp4"
    is_nvfp4 = quant_type == "nvfp4"
    if is_nvfp4 and mxfp8_normal_scales:
        raise ValueError("mxfp8_normal_scales is only supported for MXFP8")
    if is_nvfp4 and norm_const is None:
        raise ValueError("nvfp4 staging requires an offline norm_const")
    if not is_nvfp4 and norm_const is not None:
        raise ValueError(
            f"{quant_type} staging is self-scaled; norm_const must be None"
        )
    if not is_nvfp4 and scale_multiplier is not None:
        raise ValueError("scale_multiplier is only supported for NVFP4")

    num_tokens, hidden = hidden_states.shape
    capacity = x_out.shape[0]
    if fuse_tail_mask and (
        not normal_scales
        or topk_idx_out.dtype != torch.int32
        or topk_idx_out.shape[0] != capacity
    ):
        raise ValueError(
            "fused tail masking requires normal-scale MXFP4/MXFP8 and full-capacity int32 routes"
        )
    if num_tokens == 0:
        # Nothing to quantize, but the staging contract still applies: rows a
        # previous batch left routed must be re-masked and the live-count memo
        # must record 0, or staged_tokens()/compute(output=None) would keep
        # reporting the previous batch.
        _mask_tail_and_note(topk_idx_out, num_tokens, capacity)
        return
    sf_vec = 16 if is_nvfp4 else 32
    # hidden // sf_vec must be a multiple of 4 so the buffer's round-up-to-4
    # SF padding is zero and the full-width view is the exact block count —
    # hidden % 64 (nvfp4) / % 128 (mxfp8). Callers gate on
    # fused_quant_stage_supported() and fall back to torch staging otherwise.
    if hidden % (4 * sf_vec) != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {4 * sf_vec} for the fused "
            f"{quant_type} stage (got {hidden}); use the torch staging path."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")
    topk = topk_ids.shape[1]
    if num_experts is not None and (
        num_experts <= 0 or topk > 32 or topk_idx_out.dtype != torch.int32
    ):
        raise ValueError(
            "fused route checks require positive num_experts and int32 top-k <= 32"
        )
    n_blocks = hidden // sf_vec
    if x_sf_out.shape[1] != n_blocks:
        raise ValueError(
            f"x_sf trailing dim ({x_sf_out.shape[1]}) must be {n_blocks} "
            f"for hidden={hidden}, {quant_type}."
        )

    # Rubin uses int32 route storage; the original workspace uses int64.
    # A compiled stager is specialized on these tensor types and layouts.
    tensor_signature = tuple(
        (t.dtype, t.device, tuple(t.stride()))
        for t in (
            hidden_states,
            topk_ids,
            topk_weights,
            x_out,
            x_sf_out,
            topk_idx_out,
            topk_weights_out,
        )
    )
    key = (
        topk,
        hidden,
        quant_type,
        tensor_signature,
        scale_multiplier is not None,
        mxfp8_normal_scales,
        num_experts,
        fuse_tail_mask,
    )
    stager = _STAGERS.get(key)
    if stager is None:
        ensure_not_capturing("fused staging construction")
        from src.inputs_process import DataPreprocess

        stager_class = (
            _int32_routing_stager_class(normal_scales, num_experts, fuse_tail_mask)
            if topk_idx_out.dtype == torch.int32 or normal_scales
            else DataPreprocess
        )
        stager = _CompiledStager(
            dp=stager_class(topk=topk, hidden=hidden, quant_type=quant_type)
        )
        _STAGERS[key] = stager

    import cuda.bindings.driver as cuda_driver

    stream = torch.cuda.current_stream().cuda_stream
    launch_key = (
        hidden_states.data_ptr(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        x_out.data_ptr(),
        x_sf_out.data_ptr(),
        topk_idx_out.data_ptr(),
        topk_weights_out.data_ptr(),
        num_tokens,
        capacity if fuse_tail_mask else None,
        norm_const,
        scale_multiplier,
        stream,
    )
    if stager.compiled is None or stager.launch_key != launch_key:
        import cutlass

        args = (
            _to_cute(hidden_states, 16),
            _to_cute(topk_ids, 4),
            _to_cute(topk_weights, 4),
            None,  # token_padding_info: only live rows are launched
            _to_cute(x_out[:num_tokens], 16),
            _to_cute(x_sf_out[:num_tokens], 4),
            _to_cute(topk_idx_out if fuse_tail_mask else topk_idx_out[:num_tokens], 4),
            _to_cute(topk_weights_out[:num_tokens], 4),
            cuda_driver.CUstream(stream),
        )
        kwargs = {"offline_norm_const": cutlass.Float32(norm_const)} if is_nvfp4 else {}
        if scale_multiplier is not None:
            kwargs["offline_scale_multiplier"] = cutlass.Float32(scale_multiplier)
        if stager.compiled is None:
            ensure_not_capturing("fused staging cute.compile")
            import cutlass.cute as cute

            # CuTe otherwise drops cf.assert in release builds. Route checks
            # are part of the input contract, including during graph replay.
            compile_options = (
                {"options": "--enable-assertions"} if num_experts is not None else {}
            )
            stager.compiled = cute.compile(
                stager.dp, *args, **kwargs, **compile_options
            )
        stager.launch_args = args
        stager.launch_kwargs = kwargs
        stager.launch_key = launch_key

    stager.compiled(*stager.launch_args, **stager.launch_kwargs)

    if fuse_tail_mask and (capacity - num_tokens) * topk <= (
        num_tokens * stager.dp._threads_per_cta * _MAX_FUSED_TAIL_ELEMENTS_PER_THREAD
    ):
        note_staged_tokens(topk_idx_out, num_tokens)
    else:
        _mask_tail_and_note(topk_idx_out, num_tokens, capacity)


def _mask_tail_and_note(
    topk_idx_out: torch.Tensor, num_tokens: int, capacity: int
) -> None:
    # Parity with the torch stager: rows beyond this batch must stay masked
    # (-1) so they cannot dispatch as live tokens. The buffer starts fully
    # masked and staging only overwrites [:n], so eagerly a fill is needed
    # ONLY for the [n, prev_n) rows a previous LARGER batch left routed —
    # memoized per buffer so same-or-growing batches skip the launch (all
    # layers share one buffer in an engine, so at most the first layer of a
    # shrinking step pays it).
    #
    # Under CUDA graphs the memo is USELESS AND DANGEROUS: graphs replay
    # device ops without host logic, and engines capture one graph per batch
    # size — a small-size graph replayed after a larger one must mask
    # everything the larger graph staged, and an EAGER call after any replay
    # cannot trust a host count the replays never updated. So every captured
    # graph bakes the full conservative fill, and a buffer that has ever been
    # captured permanently loses the memo optimization (eager fills fully).
    ptr = topk_idx_out.data_ptr()
    if torch.cuda.is_current_stream_capturing():
        _GRAPH_CAPTURED_BUFFERS.add(ptr)
    if ptr in _GRAPH_CAPTURED_BUFFERS:
        if num_tokens < capacity:
            topk_idx_out[num_tokens:capacity].fill_(-1)
    else:
        prev_n = _LAST_STAGED_N.get(ptr, capacity)
        if num_tokens < prev_n:
            topk_idx_out[num_tokens:prev_n].fill_(-1)
    # The memo always records the ACTUAL live count of this staging: it also
    # backs staged_tokens() / compute(output=None) view slicing, which must
    # see the real n even for captured buffers (each captured graph's view is
    # fixed at its own capture-time batch size).
    _LAST_STAGED_N[ptr] = num_tokens
