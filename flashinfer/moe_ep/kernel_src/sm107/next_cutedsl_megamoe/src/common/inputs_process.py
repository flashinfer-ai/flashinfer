# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused bf16 -> quantized-activation staging for MegaMoE.

One launch turns bf16 hidden states into the packed activation + block-scale
layout the MegaMoE GEMMs consume, and repacks the routing tensors (topk idx /
weights) with padding-token masking. The quant kind is fixed at construction:

    nvfp4        E2M1 data + per-16 E4M3 block scale + a per-tensor global scale
                 (``norm_const``). E4M3 block scales need that global scale to
                 stay in range; it is either an offline-calibrated constant or
                 computed online from the activation amax.
    mxfp8_e4m3   E4M3 data + per-32 E8M0 block scale. Self-contained: the
    mxfp8_e5m2   power-of-two E8M0 scale needs no global scale.

Padding tokens (``token_padding_info[token]`` truthy) get ``topk_idx = -1`` and
``topk_weights = 0`` so downstream dispatch skips them; their quantized rows are
still written but never read.
"""

from typing import Literal, Optional, Union


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64

from .megamoe_constants import (
    Fp32Max,
    Fp8E4M3FNMax,
    Fp8E4M3RcpLimit,
    Fp8E5M2RcpLimit,
    Mxfp8BlockSize,
    Nvfp4BlockSize,
    Nvfp4E2M1Max,
    Nvfp4E2M1RcpLimit,
)
from .moe_utils import cvt_f32_to_f8_to_f32
from .ptx_helpers import max_abs_bf16x2, red_min_relaxed_gpu_u32_from_f32_raw


class DataPreprocess:
    """Staging launcher pinned to one ``(topk, quant_type)``.

    ``__init__`` pins the static config (quant dtypes, scale-block size);
    ``__call__`` is the ``@cute.jit`` launcher that sizes the grid from the
    runtime token count and dispatches the quant kind's kernel. The caller owns
    the torch->cute conversion and the ``cute.compile`` / ``aot_compile``.
    """

    # One CTA per token; threads split that token's hidden row.
    _threads_per_cta: int = 128
    _amax_threads_per_cta: int = 128
    _amax_load_vec: int = 8  # 8 bf16 = 16 B per load

    def __init__(self, topk: int, hidden: int, quant_type: Literal["nvfp4", "mxfp8_e5m2", "mxfp8_e4m3"]) -> None:
        self.topk = int(topk)
        # The routing repack assigns one lane per topk slot, so topk cannot
        # exceed the CTA width.
        if self.topk > self._threads_per_cta:
            raise ValueError(
                f"topk ({self.topk}) must be <= threads_per_cta "
                f"({self._threads_per_cta}); the routing repack uses one lane "
                "per topk slot."
            )
        # Compile-time: the quant kernel stages the whole row into (static) smem.
        self.hidden = int(hidden)
        self.quant_type = quant_type
        self.is_nvfp4 = quant_type == "nvfp4"
        if quant_type == "nvfp4":
            self.sf_vec = Nvfp4BlockSize
            self.quant_dtype = cutlass.Float4E2M1FN
            self.sf_dtype = cutlass.Float8E4M3FN
        elif quant_type == "mxfp8_e4m3":
            self.sf_vec = Mxfp8BlockSize
            self.quant_dtype = cutlass.Float8E4M3FN
            self.sf_dtype = cutlass.Float8E8M0FNU
            self._mxfp8_data_rcp_limit = float(Fp8E4M3RcpLimit)  # 1 / 448
        elif quant_type == "mxfp8_e5m2":
            self.sf_vec = Mxfp8BlockSize
            self.quant_dtype = cutlass.Float8E5M2
            self.sf_dtype = cutlass.Float8E8M0FNU
            self._mxfp8_data_rcp_limit = float(Fp8E5M2RcpLimit)  # 1 / 57344
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type!r}")

        output_alignment_elements = 32 if self.is_nvfp4 else 16
        if self.hidden % output_alignment_elements != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must make each quantized row 16-byte aligned; "
                f"{quant_type} requires a multiple of {output_alignment_elements} elements."
            )

        # nvfp4 encode constants (see `nvfp4_quant_and_process_impl`):
        #   stored E4M3 block scale = block_amax * rcp_limit * norm_const
        #   norm_const at amax        = (E4M3_max * E2M1_max) / amax_tensor
        self._nvfp4_rcp_limit = float(Nvfp4E2M1RcpLimit)  # 1 / 6
        self._nvfp4_norm_numer = float(Fp8E4M3FNMax * Nvfp4E2M1Max)  # 448 * 6

    # -- launcher -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        # inputs
        activation_bf16: cute.Tensor,  # (token, hidden)
        topk_idx: cute.Tensor,  # (token, topk)
        topk_weights: cute.Tensor,  # (token, topk)
        token_padding_info: Optional[cute.Tensor],  # (token,) ; None => no padding
        # outputs
        activation_quant: cute.Tensor,  # (token, hidden) fp4/fp8 elements
        # (cute sees the real fp4 dtype, not
        # torch's pack2 uint8 -> full hidden)
        activation_sf: cute.Tensor,  # (token, ceil(hidden / sf_vec)) block scales,
        # rebuilt below into a (token, hidden)
        # broadcast view over the block dim
        topk_idx_output: cute.Tensor,  # (token, topk)
        topk_weights_output: cute.Tensor,  # (token, topk)
        cuda_stream: cuda.CUstream,
        # nvfp4 only: exactly one of the two is valid (see class docstring).
        #   online:  (1,) output buffer; staging seeds it with Fp32Max and
        #            min-reduces _nvfp4_norm_numer / token_amax into norm_const
        #            for the quantization launch.
        #   offline: runtime f32 scalar, read-only calibrated scale.
        online_norm_const: Optional[cute.Tensor] = None,
        offline_norm_const: Optional[cutlass.Float32] = None,
        # Optional pre-rounded norm_const / 6 for the NVFP4 block scale.
        offline_scale_multiplier: Optional[cutlass.Float32] = None,
    ) -> None:
        # Mode invariants resolve at trace time (the scale args are None-or-not
        # when the kernel is compiled).
        if cutlass.const_expr(offline_scale_multiplier is not None):
            if cutlass.const_expr(not self.is_nvfp4 or offline_norm_const is None):
                raise ValueError("offline_scale_multiplier requires NVFP4 offline normalization")
        if cutlass.const_expr(self.is_nvfp4):
            if cutlass.const_expr((online_norm_const is None) == (offline_norm_const is None)):
                raise ValueError(
                    "nvfp4 staging needs exactly one of `online_norm_const` "
                    "(output buffer, staging-zeroed) or `offline_norm_const` "
                    "(calibrated constant); got both or neither."
                )
        else:
            if cutlass.const_expr(online_norm_const is not None or offline_norm_const is not None):
                raise ValueError(
                    f"{self.quant_type} staging is self-scaled; do not pass `online_norm_const` / `offline_norm_const`."
                )

        num_tokens = activation_bf16.shape[0]
        grid = [num_tokens, 1, 1]
        block = [self._threads_per_cta, 1, 1]

        # Rebuild the block scales into the cute-idiomatic broadcast layout so
        # they index in the same (token, hidden) space as the data: the inner
        # sf_vec mode carries stride 0, so every element in a block maps to that
        # block's single scale entry.
        #   (token, ceil(hidden / sf_vec)) -> (token, (sf_vec, ceil(hidden / sf_vec)))
        #                                    stride (d_token, (0, d_block))
        num_sf_blocks = activation_sf.shape[1]
        activation_sf = cute.make_tensor(
            activation_sf.iterator,
            cute.make_layout(
                (activation_sf.shape[0], (self.sf_vec, num_sf_blocks)),
                stride=(activation_sf.stride[0], (0, activation_sf.stride[1])),
            ),
        )

        if cutlass.const_expr(not self.is_nvfp4):
            self.mxfp8_quant_and_process_impl(
                activation_bf16,
                topk_idx,
                topk_weights,
                token_padding_info,
                activation_quant,
                activation_sf,
                topk_idx_output,
                topk_weights_output,
            ).launch(grid=grid, block=block, stream=cuda_stream, min_blocks_per_mp=1)
            return

        # nvfp4: online derives norm_const in-band (seed -> reduce -> quant,
        # three launches so kernel boundaries stand in for a grid barrier
        # with no extra workspace); offline reads the caller's constant in one
        # launch. Perf-critical callers should calibrate offline.
        if cutlass.const_expr(online_norm_const is not None):
            self._init_online_scale_impl(online_norm_const).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=cuda_stream)
            self.nvfp4_amax_impl(activation_bf16, token_padding_info, online_norm_const).launch(
                grid=grid, block=[self._amax_threads_per_cta, 1, 1], stream=cuda_stream, min_blocks_per_mp=1
            )
            norm_const = online_norm_const
        else:
            norm_const = offline_norm_const

        self.nvfp4_quant_and_process_impl(
            activation_bf16,
            topk_idx,
            topk_weights,
            token_padding_info,
            activation_quant,
            activation_sf,
            topk_idx_output,
            topk_weights_output,
            norm_const,
            offline_scale_multiplier,
        ).launch(grid=grid, block=block, stream=cuda_stream, min_blocks_per_mp=1)

    # -- shared device helpers ------------------------------------------------

    @cute.jit
    def _mark_alignment(self, tensor: cute.Tensor, align_bytes: int) -> cute.Tensor:
        # Re-tag a tensor's pointer with a stronger alignment so vectorized
        # copies are legal (mirrors topk_reduce._mark_alignment).
        p = tensor.iterator
        return cute.make_tensor(cute.make_ptr(p.dtype, p.toint(), p.memspace, assumed_align=align_bytes), tensor.layout)

    @cute.jit
    def _repack_routing(
        self,
        token_idx,
        tid,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        token_padding_info: Optional[cute.Tensor],
        topk_idx_output: cute.Tensor,
        topk_weights_output: cute.Tensor,
    ) -> None:
        # Preserve the destination's integer route type; padding tokens are
        # masked to (-1, 0).
        topk: cutlass.Constexpr[int] = self.topk
        if tid < Int32(topk):
            is_padding = False
            if cutlass.const_expr(token_padding_info is not None):
                is_padding = token_padding_info[token_idx] != Int32(0)
            idx = Int64(-1) if is_padding else Int64(topk_idx[token_idx, tid])
            weight = Float32(0.0) if is_padding else Float32(topk_weights[token_idx, tid])
            topk_idx_output[token_idx, tid] = idx.to(topk_idx_output.element_type)
            topk_weights_output[token_idx, tid] = weight

    # -- kernels --------------------------------------------------------------

    @cute.kernel
    def _init_online_scale_impl(self, scale: cute.Tensor) -> None:
        # Seed the (1,) online scale slot with the atomic-min identity (a large
        # positive sentinel) before the amax reduction runs against it.
        scale[0] = Float32(Fp32Max)

    @cute.kernel
    def nvfp4_amax_impl(
        self, activation_bf16: cute.Tensor, token_padding_info: Optional[cute.Tensor], online_norm_const: cute.Tensor
    ) -> None:
        # One CTA per token reduces its (non-padding) row amax, converts it to a
        # per-token norm_const candidate, and atomic-mins it into the shared
        # slot. norm_const = numer / amax is monotone-decreasing in amax, so
        # min over tokens == numer / max amax == the global norm_const.
        threads: cutlass.Constexpr[int] = self._amax_threads_per_cta
        num_warps: cutlass.Constexpr[int] = threads // 32
        load_vec: cutlass.Constexpr[int] = self._amax_load_vec
        token_idx = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        hidden = activation_bf16.shape[1]

        smem = cutlass.utils.SmemAllocator()
        warp_partials = smem.allocate_array(Float32, num_warps)

        is_padding = False
        if cutlass.const_expr(token_padding_info is not None):
            is_padding = token_padding_info[token_idx] != Int32(0)

        if not is_padding:
            # 16 B vectorized loads: each thread strides over load_vec-wide chunks.
            a_chunks = cute.zipped_divide(activation_bf16[token_idx, None], (load_vec,))
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=load_vec * 16
            )
            num_chunks = hidden // load_vec
            # Stay in bf16: max.xorsign.abs.bf16x2 does per-lane abs-max on two
            # packed bf16, so no per-element fp32 convert. The junk xor-sign is
            # masked at the end; the single per-thread amax is converted once.
            acc_pair = Int32(0)
            c = tid
            while c < num_chunks:
                chunk = cute.make_rmem_tensor((load_vec,), cutlass.BFloat16)
                # Each chunk starts at c*16 B (16 B aligned); re-tag so the
                # 128-bit copy's static alignment check passes.
                cute.copy(load_atom, self._mark_alignment(a_chunks[(None,), (c,)], 16), chunk)
                pairs = cute.recast_tensor(chunk, Int32)
                for p in cutlass.range_constexpr(load_vec // 2):
                    acc_pair = max_abs_bf16x2(acc_pair, pairs[p])
                c += Int32(threads)

            # Fold the two bf16 lanes together, mask the junk sign, convert once.
            acc_pair = max_abs_bf16x2(acc_pair, acc_pair >> Int32(16))
            local_bits = cute.make_rmem_tensor((1,), Int32)
            local_bits[0] = acc_pair & Int32(0x7FFF)
            local_amax = Float32(cute.recast_tensor(local_bits, cutlass.BFloat16)[0])

            # Warp reduce -> num_warps STS partials -> warp 0 LDS -> warp reduce
            # -> atomic. num_warps (<= 32) partials fit in warp 0's lanes.
            warp_amax = cute.arch.warp_redux_sync(local_amax, "fmax", abs=True)
            warp_idx = tid // Int32(32)
            lane_idx = tid % Int32(32)
            if lane_idx == Int32(0):
                warp_partials[warp_idx] = warp_amax
            cute.arch.sync_threads()

            if warp_idx == Int32(0):
                partial = Float32(0.0)
                if lane_idx < Int32(num_warps):
                    partial = warp_partials[lane_idx]
                token_amax = cute.arch.warp_redux_sync(partial, "fmax", abs=True)
                if lane_idx == Int32(0):
                    # token_amax == 0 -> candidate +inf, harmless under min.
                    candidate = Float32(self._nvfp4_norm_numer) * cute.arch.rcp_approx(token_amax)
                    red_min_relaxed_gpu_u32_from_f32_raw(online_norm_const.iterator.toint(), candidate)

    @cute.kernel
    def nvfp4_quant_and_process_impl(
        self,
        activation_bf16: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        token_padding_info: Optional[cute.Tensor],
        activation_quant: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx_output: cute.Tensor,
        topk_weights_output: cute.Tensor,
        norm_const: Union[cute.Tensor, cutlass.Float32],  # online (1,) tensor or offline f32 scalar
        scale_multiplier: Optional[cutlass.Float32] = None,
    ) -> None:
        threads: cutlass.Constexpr[int] = self._threads_per_cta
        sf_vec: cutlass.Constexpr[int] = self.sf_vec
        hidden: cutlass.Constexpr[int] = self.hidden
        load_vec: cutlass.Constexpr[int] = 8  # 8 bf16 = 16 B per cp.async
        num_blocks: cutlass.Constexpr[int] = hidden // sf_vec
        num_chunks: cutlass.Constexpr[int] = hidden // load_vec
        chunks_per_thread: cutlass.Constexpr[int] = (num_chunks + threads - 1) // threads
        blocks_per_thread: cutlass.Constexpr[int] = (num_blocks + threads - 1) // threads
        token_idx = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]

        if cutlass.const_expr(isinstance(norm_const, cute.Tensor)):
            nc = Float32(norm_const[0])  # online: derived scale slot
        else:
            nc = norm_const  # offline: runtime f32 scalar
        rcp_limit = Float32(self._nvfp4_rcp_limit)
        fp32_max = Float32(Fp32Max)

        # Stage the whole bf16 row into swizzled smem via cp.async (LDGSTS):
        # coalesced 16 B loads (8 bf16/lane), all issued up front, no mbarrier.
        # The swizzle makes the later per-block LDS.128 bank-conflict free.
        smem = cutlass.utils.SmemAllocator()
        smem_row = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout(hidden), 1024, swizzle=cute.make_swizzle(1, 4, 3)
        )
        g_chunks = cute.zipped_divide(activation_bf16[token_idx, None], (load_vec,))
        s_chunks = cute.zipped_divide(smem_row, (load_vec,))
        g2s_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=load_vec * 16
        )
        # Issue every cp.async up front, committing a group every 2 load rounds
        # (= one quant round's worth of blocks). num_groups == blocks_per_thread
        # (nested ceil-div identity), so group j feeds quant round j 1:1.
        num_groups: cutlass.Constexpr[int] = (chunks_per_thread + 1) // 2
        for i in cutlass.range_constexpr(chunks_per_thread):
            c = tid + Int32(i * threads)
            if c < Int32(num_chunks):
                cute.copy(g2s_atom, self._mark_alignment(g_chunks[(None,), (c,)], 16), s_chunks[(None,), (c,)])
            if cutlass.const_expr(i % 2 == 1 or i == chunks_per_thread - 1):
                cute.arch.cp_async_commit_group()

        # Quantize each 16-block from smem; block amax is register-local, so no
        # cross-thread reduce. Read via LDS.128 to benefit from the swizzle.
        s_blk = cute.zipped_divide(smem_row, (sf_vec,))
        q_blk = cute.zipped_divide(activation_quant[token_idx, None], (sf_vec,))
        lds_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN, num_bits_per_copy=sf_vec * 4
        )
        for j in cutlass.range_constexpr(blocks_per_thread):
            # A dummy commit each round keeps the target group a fixed depth from
            # the tail, so the wait arg stays the constant `num_groups` (waits for
            # exactly group j; later groups keep loading underneath). Barrier
            # makes the cooperatively loaded smem region visible to all threads.
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(num_groups)
            cute.arch.sync_threads()
            b = tid + Int32(j * threads)
            if b < Int32(num_blocks):
                block_bf16 = cute.make_rmem_tensor((sf_vec,), cutlass.BFloat16)
                cute.copy(lds_atom, s_blk[(None,), (b,)], block_bf16)
                vals = cute.make_rmem_tensor((sf_vec,), Float32)
                vals.store(block_bf16.load().to(Float32))

                absmax = Float32(0.0)
                for i in cutlass.range_constexpr(sf_vec):
                    absmax = cute.arch.fmax(absmax, cute.arch.fmax(vals[i], -vals[i]))

                # Stored per-block E4M3 scale; .to() clamps to the E4M3 range.
                if cutlass.const_expr(scale_multiplier is not None):
                    sfc_e4m3 = (absmax * scale_multiplier).to(cutlass.Float8E4M3FN)
                else:
                    sfc_e4m3 = (absmax * rcp_limit * nc).to(cutlass.Float8E4M3FN)
                sfc_rt = Float32(sfc_e4m3)
                # Per-element encode scale; the mask zeros the sfc==0 (all-zero
                # block) case so its fp4 codes stay 0 instead of NaN.
                acc_scale = cute.arch.fmin(nc * cute.arch.rcp_approx(sfc_rt), fp32_max)
                acc_scale = acc_scale * cute.arch.fmin(sfc_rt * Float32(1.0e30), Float32(1.0))

                scaled = cute.make_rmem_tensor((sf_vec,), Float32)
                for i in cutlass.range_constexpr(sf_vec):
                    scaled[i] = vals[i] * acc_scale
                fp4 = cute.make_rmem_tensor((sf_vec,), cutlass.Float4E2M1FN)
                fp4.store(scaled.load().to(cutlass.Float4E2M1FN))

                cute.copy(store_atom, fp4, self._mark_alignment(q_blk[(None,), (b,)], sf_vec // 2))
                activation_sf[token_idx, (0, b)] = sfc_e4m3

        self._repack_routing(
            token_idx, tid, topk_idx, topk_weights, token_padding_info, topk_idx_output, topk_weights_output
        )

    @cute.kernel
    def mxfp8_quant_and_process_impl(
        self,
        activation_bf16: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        token_padding_info: Optional[cute.Tensor],
        activation_quant: cute.Tensor,
        activation_sf: cute.Tensor,
        topk_idx_output: cute.Tensor,
        topk_weights_output: cute.Tensor,
    ) -> None:
        # Adjacent lanes jointly quantize one 32-element scale block. Each lane
        # owns one contiguous 16-byte output vector, so every STG.128 is
        # coalesced across the warp. The final scale block may contain only the
        # even lane's 16 elements.
        threads: cutlass.Constexpr[int] = self._threads_per_cta
        hidden: cutlass.Constexpr[int] = self.hidden
        load_vec: cutlass.Constexpr[int] = 8  # 8 bf16 = 16 B per cp.async
        quant_vec: cutlass.Constexpr[int] = 16  # one coalesced 16 B fp8 store per lane
        quant_subrounds_per_stage: cutlass.Constexpr[int] = 2
        load_rounds_per_stage: cutlass.Constexpr[int] = quant_subrounds_per_stage * quant_vec // load_vec
        num_chunks: cutlass.Constexpr[int] = hidden // load_vec
        num_quant_chunks: cutlass.Constexpr[int] = hidden // quant_vec
        chunks_per_thread: cutlass.Constexpr[int] = (num_chunks + threads - 1) // threads
        full_quant_rounds: cutlass.Constexpr[int] = num_quant_chunks // threads
        tail_quant_chunks: cutlass.Constexpr[int] = num_quant_chunks % threads
        num_stages: cutlass.Constexpr[int] = (num_quant_chunks + quant_subrounds_per_stage * threads - 1) // (
            quant_subrounds_per_stage * threads
        )
        num_groups: cutlass.Constexpr[int] = (chunks_per_thread + load_rounds_per_stage - 1) // load_rounds_per_stage
        token_idx = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]

        data_rcp_limit = Float32(self._mxfp8_data_rcp_limit)
        fp32_max = Float32(Fp32Max)

        smem = cutlass.utils.SmemAllocator()
        smem_row = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout(hidden), 1024, swizzle=cute.make_swizzle(1, 4, 3)
        )
        g_chunks = cute.zipped_divide(activation_bf16[token_idx, None], (load_vec,))
        s_chunks = cute.zipped_divide(smem_row, (load_vec,))
        g2s_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=load_vec * 16
        )
        # Preserve the original 32-element-per-thread batch: each group feeds
        # two 16-element quant subrounds.
        for i in cutlass.range_constexpr(chunks_per_thread):
            c = tid + Int32(i * threads)
            if c < Int32(num_chunks):
                cute.copy(g2s_atom, self._mark_alignment(g_chunks[(None,), (c,)], 16), s_chunks[(None,), (c,)])
            if cutlass.const_expr((i + 1) % load_rounds_per_stage == 0 or i == chunks_per_thread - 1):
                cute.arch.cp_async_commit_group()

        s_quant_chunk = cute.zipped_divide(smem_row, (quant_vec,))
        q_quant_chunk = cute.zipped_divide(activation_quant[token_idx, None], (quant_vec,))
        lds_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.quant_dtype, num_bits_per_copy=128)
        # A stage waits once, then consumes two adjacent 16-element subrounds.
        # Compile-time subround selection keeps every complete subround
        # straight-line; only the final partial subround is predicated.
        for stage in cutlass.range_constexpr(num_stages):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(num_groups)
            cute.arch.sync_threads()

            for subround in cutlass.range_constexpr(quant_subrounds_per_stage):
                quant_round = stage * quant_subrounds_per_stage + subround
                quant_chunk_idx = tid + Int32(quant_round * threads)

                if cutlass.const_expr(quant_round < full_quant_rounds):
                    quant_chunk_bf16 = cute.make_rmem_tensor((quant_vec,), cutlass.BFloat16)
                    cute.copy(lds_atom, s_quant_chunk[(None,), (quant_chunk_idx,)], quant_chunk_bf16)
                    values = cute.make_rmem_tensor((quant_vec,), Float32)
                    values.store(quant_chunk_bf16.load().to(Float32))

                    local_absmax = Float32(0.0)
                    for i in cutlass.range_constexpr(quant_vec):
                        local_absmax = cute.arch.fmax(local_absmax, cute.arch.fmax(values[i], -values[i]))
                    paired_absmax = Float32(cute.arch.shuffle_sync_bfly(local_absmax, offset=1))
                    block_absmax = cute.arch.fmax(local_absmax, paired_absmax)

                    # Per-32 E8M0 (power-of-2) scale, rounded up (cvt.rp) so the
                    # block never overflows fp8; rcp is exact for a power of two.
                    scale_f32 = Float32(cvt_f32_to_f8_to_f32(block_absmax * data_rcp_limit, cutlass.Float8E8M0FNU))
                    sf_e8m0 = scale_f32.to(cutlass.Float8E8M0FNU)
                    acc_scale = cute.arch.fmin(cute.arch.rcp_approx(scale_f32), fp32_max)

                    scaled = cute.make_rmem_tensor((quant_vec,), Float32)
                    for i in cutlass.range_constexpr(quant_vec):
                        scaled[i] = values[i] * acc_scale
                    data = cute.make_rmem_tensor((quant_vec,), self.quant_dtype)
                    data.store(scaled.load().to(self.quant_dtype))

                    cute.copy(
                        store_atom, data, self._mark_alignment(q_quant_chunk[(None,), (quant_chunk_idx,)], quant_vec)
                    )
                    activation_sf[token_idx, (0, quant_chunk_idx // Int32(2))] = sf_e8m0

                elif cutlass.const_expr(quant_round == full_quant_rounds and tail_quant_chunks != 0):
                    # Separate locals avoid type joins with the full subround in CuTe DSL 4.6.
                    tail_values = cute.make_rmem_tensor((quant_vec,), Float32)
                    tail_local_absmax = Float32(0.0)
                    if tid < Int32(tail_quant_chunks):
                        tail_quant_chunk_bf16 = cute.make_rmem_tensor((quant_vec,), cutlass.BFloat16)
                        cute.copy(lds_atom, s_quant_chunk[(None,), (quant_chunk_idx,)], tail_quant_chunk_bf16)
                        tail_values.store(tail_quant_chunk_bf16.load().to(Float32))
                        for i in cutlass.range_constexpr(quant_vec):
                            tail_local_absmax = cute.arch.fmax(tail_local_absmax, cute.arch.fmax(tail_values[i], -tail_values[i]))

                    tail_paired_absmax = Float32(cute.arch.shuffle_sync_bfly(tail_local_absmax, offset=1))
                    tail_block_absmax = cute.arch.fmax(tail_local_absmax, tail_paired_absmax)
                    if tid < Int32(tail_quant_chunks):
                        tail_scale_f32 = Float32(cvt_f32_to_f8_to_f32(tail_block_absmax * data_rcp_limit, cutlass.Float8E8M0FNU))
                        tail_sf_e8m0 = tail_scale_f32.to(cutlass.Float8E8M0FNU)
                        tail_acc_scale = cute.arch.fmin(cute.arch.rcp_approx(tail_scale_f32), fp32_max)

                        tail_scaled = cute.make_rmem_tensor((quant_vec,), Float32)
                        for i in cutlass.range_constexpr(quant_vec):
                            tail_scaled[i] = tail_values[i] * tail_acc_scale
                        tail_data = cute.make_rmem_tensor((quant_vec,), self.quant_dtype)
                        tail_data.store(tail_scaled.load().to(self.quant_dtype))

                        cute.copy(
                            store_atom,
                            tail_data,
                            self._mark_alignment(q_quant_chunk[(None,), (quant_chunk_idx,)], quant_vec),
                        )
                        activation_sf[token_idx, (0, quant_chunk_idx // Int32(2))] = tail_sf_e8m0

        self._repack_routing(
            token_idx, tid, topk_idx, topk_weights, token_padding_info, topk_idx_output, topk_weights_output
        )
