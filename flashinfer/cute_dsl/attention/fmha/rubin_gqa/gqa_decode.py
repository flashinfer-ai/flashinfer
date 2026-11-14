# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Literal, Type, Tuple, cast
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)

import cutlass.utils as utils
from cutlass.cute.typing import (
    Float32,
    BFloat16,
    Float16,
    Int32,
    Int64,
    Int128,
    Boolean,
    Optional,
)

from .mask import AttentionMask, NoMask, DenseMask, CausalMask, SlidingWindowMask

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4  # log2(16)

# Math helpers
min_f32 = Float32(
    -3.4028234663852886e38
)  # lowest finite float32, prevent nans with masking
log2_e = math.log2(math.e)  # change exponential base
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")


# Arch queries
from cutlass import Arch
from cutlass.cutlass_dsl import BaseDSL

arch = BaseDSL._get_dsl().get_arch_enum()
resource_arch = f"sm_{arch.major}{arch.minor}"
make_trivial_tiled_mma = utils.sm100.make_trivial_tiled_mma
mma_k64_supported = arch.can_run_binary_built_for(Arch.sm_107f)
if mma_k64_supported:
    # The Rubin helper selects K=64 by default for rank-2 FP8 MMA shapes.
    make_trivial_tiled_mma = utils.sm107.make_trivial_tiled_mma


class GqaDecodeTcgen05:
    def __init__(
        self,
        headdim,
        grouped_head_tile,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode: Literal["kernel", "atomic", "none"] = "kernel",
        *,
        disable_persistence=False,
        reduction_splits: Optional[int] = 16,
        red_vec_bits: int = 64,
        red_d_ctas: int = 1,
        red_max_groups: int = 16,
        kv_stages_cap: int = 0,
    ):
        """
        Parameters
        ----------
        headdim
            Head dimension.
        grouped_head_tile
            Grouped heads per threadblock (GQA packing factor).
        prediction_tile
            Predicted tokens per threadblock.
        sequence_tile
            KV tokens per threadblock per loop iteration.
        reduction_mode
            Split-K reduction algorithm:
              - ``"kernel"``: deterministic kernel reduction with partial result workspace.
              - ``"atomic"``: cluster reduction with atomic adds, no workspace.
              - ``"none"``: no split-K, flash decoding disabled.
        reduction_splits
            ``kv_splits`` as a Python int, for ``"kernel"`` reduction only. The
            reducer is specialized at compile time.
        red_vec_bits
            Width in bits of the reduction kernel's per-thread partial-O vector.
        red_d_ctas
            Reduction CTAs per head; splits a head's head-dimension gather
            across that many CTAs.
        red_max_groups
            Upper bound on the reduction kernel's split groups. The group count
            is the largest divisor of ``reduction_splits`` not exceeding it.
        kv_stages_cap
            Maximum KV pipeline depth; zero leaves the depth uncapped.
        """
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == "kernel"
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_none_red = reduction_mode == "none" or reduction_mode is None
        self.is_persistent = self.do_none_red and not disable_persistence
        self.reduction_splits = reduction_splits
        self.red_vec_bits = red_vec_bits
        self.red_d_ctas = red_d_ctas
        self.red_max_groups = red_max_groups
        self.kv_stages_cap = kv_stages_cap
        self.threads_per_cta = 4 * warpgroup_threads

        assert headdim > 0 and headdim % 64 == 0
        assert grouped_head_tile * prediction_tile in (1, 2, 4, 8, 16, 32)
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert self.do_kernel_red ^ self.do_atomic_red ^ self.do_none_red
        if self.do_kernel_red:
            assert reduction_splits != None

    ##############################
    # Launch helpers
    ##############################
    # Runtime implementable check
    def can_implement(
        self,
        kv_splits,
        qo_shape_bshd,
        kv_shape_bshd,
        qkv_dtype,
        o_dtype,
        mask_config,
    ):
        b_k, s_q, h_q, d_q = qo_shape_bshd
        b_q, s_k, h_k, d_k = kv_shape_bshd

        if qkv_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")

        if not (d_q == d_k == self.headdim):
            raise ValueError(
                f"headdim_q({d_q}), headdim_k({d_k}) must be {self.headdim}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if s_q > self.sequence_tile:
            raise ValueError(
                f"seqlen_q({s_q}) must be at most tile_kv({self.sequence_tile}) for decode"
            )

        if 0 < s_k < s_q:
            raise ValueError(
                f"non-zero seqlen({s_k}) must be at least prediction({s_q})"
            )

        if b_k != b_q:
            raise ValueError(f"batches_k({b_k}) and batches_q({b_q}) mismatch")

        if self.do_atomic_red:
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits po2 <= 16, got {kv_splits}"
                )

            if o_dtype not in (Float32, BFloat16, Float16):
                raise TypeError(
                    f"atomic reduction requires (Float32, BFloat16, Float16) o_dtype, got {o_dtype}"
                )

        if self.do_none_red and kv_splits != 1:
            raise ValueError("KV splits must be 1 if flash decoding is disabled")

        if self.do_kernel_red and kv_splits != self.reduction_splits:
            raise ValueError(
                f"kv_splits={kv_splits} must match "
                f"compiled reduction_splits={self.reduction_splits}"
            )

        mask_config.can_implement(s_q, s_k, self.prediction_tile, self.sequence_tile)

    # Pack grouped heads with predicted tokens (s_q)
    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]
        # Batch + partial stride must be coalescible
        # to get 5 independent TMA modes (TMA limitation)
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k  # grouped heads
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    # Reorder and group modes for GEMM
    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)  # GEMM1 MKL
        dshb = (3, 1, 2, 0)  # GEMM2 MKL
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    # Pack, reorder, and group modes for GEMM workspace
    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    ##############################
    # Decode Kernel launch
    ##############################
    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        sink_h: Optional[cute.Tensor],
        mask_config,  # duck-typed AttentionMask, TVM FFI conversion breaks if we annotate with base class for now
        scale_s: Float32,
        scale_o: Float32,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        """
        Parameters
        ----------
        kv_splits
            Threadblocks per sequence (flash decoding).
        q_bshd, k_bshd, v_bshd
            Q/K/V tensors in BSHD logical view. Strides can be BHSD.
        o_bshd
            Output tensor. Must be zero initialized for atomic reduction.
        l_bsh
            Log-sum-exp output (Float32, log2 base). May be None.
        o_partial_bshd
            Partial O per kv split (kernel-red workspace).
        l_partial_bsh
            Partial ``colsum_p`` per kv split (kernel-red workspace).
        m_partial_bsh
            Partial ``colmax_s`` per kv split (kernel-red workspace).
        sink_h
            Pre-scaled attention sink logits per head
        scale_s
            Softmax scale.
        scale_o
            Output scale, applied in the reduction epilogue.
        mask_config
            Attention logit masking configuration.
        stream
            CUDA stream to launch on.
        enable_pdl
            Programmatic Dependent Launch. Runtime-dynamic — no recompile on
            toggle.
        """
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_bshd.dtype
        acc_dtype = Float32
        o_smem_dtype = o_partial_bshd.dtype if self.do_kernel_red else o_bshd.dtype
        assert k_bshd.dtype == v_bshd.dtype == mma_dtype

        # Block tile sets the granularity at which threadblocks consume work (BMM1/BMM2)
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are staged in smem
        mma_tile_m = 128
        mma_tile_k = 128 * 8 // mma_dtype.width
        # N-major 8b B in smem requires N multiple of 16
        min_mma_tile_n = 16 if mma_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m)
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0

        # GEMM1: (S_K, (H_R, S_Q), D, (H_K, B))
        tiled_mma_kq = make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        # GEMM2: (D, (H_R, S_Q), S_K, (H_K, B))
        tiled_mma_vp = make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.MN,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        ##############################
        # Calculate stage counts
        ##############################
        # Fixed stage counts
        self.p_stages = p_stages = 4  # smem P (BMM2 B)
        self.o_stages = o_stages = 2  # tmem O (BMM2 C)
        self.q_stages = q_stages = 2 if self.is_persistent else 1  # smem Q + O alias

        # Calculate tmem alloc
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols(resource_arch)
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages  # per-thread colsum
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages  # O
        max_s_stages = (tmem_capacity_cols - tmem_alloc_cols) // tmem_s_stage_cols
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages  # S
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # po2
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        # Calculate smem alloc
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        mk_stage_bits = mma_tile_m * mma_tile_k * mma_dtype.width
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        # tmem ptr
        smem_alloc_bits += Int32.width
        # CLC
        if cutlass.const_expr(self.is_persistent):
            smem_alloc_bits += mbarrier_bits * 2  # CLC full + empty
            smem_alloc_bits += Int128.width  # CLC response
        # colmax + colsum
        smem_alloc_bits += q_stages * blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_atomic_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
        # Q, O
        q_stage_bits = tiles_dk * mma_tile_n * mma_tile_k * mma_dtype.width
        o_stage_bits = tiles_dm * mma_tile_m * mma_tile_n * o_smem_dtype.width
        qo_stage_bits = max(q_stage_bits, o_stage_bits)  # alias QO stages
        smem_alloc_bits += q_stages * qo_stage_bits + mbarrier_bits  # 1 mbar for Q
        # S, P
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits  # o in tmem
        # Upper bound the padding. K/V ring, Q, P are 128B-aligned and
        # (sPT, sM, sL, sSink, sR) are 16B aligned.
        alignment_bits = (3 * 128 + 5 * 16) * 8
        # K, V
        smem_capacity_bytes = cutlass.memory.get_smem_capacity_in_bytes(resource_arch)
        remaining_bits = (smem_capacity_bytes * 8) - smem_alloc_bits - alignment_bits
        kv_stages = remaining_bits // (mk_stage_bits + pipe_stage_bits)
        if cutlass.const_expr(self.kv_stages_cap > 0):
            kv_stages = min(kv_stages, self.kv_stages_cap)

        ##############################
        # TMA creation
        ##############################
        h_k = k_bshd.shape[2]
        o_bshd_ = o_partial_bshd if self.do_kernel_red else o_bshd

        # ((h_g, s_q), d, (h_k, b))
        mQ_nkl = self.gemm_view(self.gqa_pack(q_bshd, h_k), True)
        mK_mkl = self.gemm_view(k_bshd, True)  # (s_k, d, (h_k, b))
        mV_mkl = self.gemm_view(v_bshd, False)  # (d, s_k, (h_k, b))
        # (d, (h_g, s_q), (h_k, b_partial))
        mO_mnl = self.gemm_view(self.gqa_pack(o_bshd_, h_k), False)

        # (MMA, MMA_M, MMA_K, stages)
        smem_layout_k = utils.sm100.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, mma_dtype, kv_stages
        )
        smem_layout_v = utils.sm100.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, mma_dtype, kv_stages
        )

        # Construct QO smem stage layout
        smem_layout_q = utils.sm100.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, mma_dtype, tiles_dk
        )
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM)
        smem_layout_o = cute.select(smem_layout_o, (0, 1, 2))

        # Alias QO stages
        # (MMA, MMA_N, MMA_K, #TILE_DK, q_stages)
        q_stage_stride = qo_stage_bits // mma_dtype.width
        smem_layout_q = cute.make_composed_layout(
            smem_layout_q.inner,
            smem_layout_q.offset,
            cute.make_layout(
                shape=(*smem_layout_q.outer.shape, q_stages),
                stride=(*smem_layout_q.outer.stride, q_stage_stride),
            ),
        )
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, q_stages)
        o_stage_stride = qo_stage_bits // o_smem_dtype.width
        smem_layout_o = cute.make_composed_layout(
            smem_layout_o.inner,
            smem_layout_o.offset,
            cute.make_layout(
                shape=(*smem_layout_o.outer.shape, q_stages),
                stride=(*smem_layout_o.outer.stride, o_stage_stride),
            ),
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

        # Construct multimode gmem tiler
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_mkl,
            cute.select(smem_layout_k, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV_mkl,
            cute.select(smem_layout_v, mma_modes),
            tma_tile_mnk,
            tiled_mma_vp,
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mO_mnl,
            cute.select(smem_layout_o, mode=[0, 1]),
            tma_tile_mnk[:2],
        )

        # GEMM view for LSE output
        # ((h_g, s_q), (h_k, b))
        mL_nl = None if l_bsh is None else self.gemm_view_bsh(l_bsh, h_k)
        assert l_bsh is None or l_bsh.dtype == acc_dtype

        # GEMM views for workspace tensors
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_kernel_red):
            assert (
                m_partial_bsh.dtype
                == l_partial_bsh.dtype
                == o_partial_bshd.dtype
                == acc_dtype
            )

            # ((h_g, s_q), (h_k, b), kv_splits)
            mM_partial_nl = self.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = self.gemm_view_bsh(l_partial_bsh, h_k)

        if cutlass.const_expr(sink_h is not None):
            assert sink_h.dtype == acc_dtype

            h_g = sink_h.shape[0] // h_k
            mSink = cute.make_tensor(
                sink_h.iterator,
                cute.make_layout((h_g, h_k), stride=(1, h_g)),
            )
        else:
            mSink = None

        ##############################
        # Launch kernel(s)
        ##############################
        scale_s_log2_e = scale_s * log2_e

        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(mQ_nkl.shape[2])  # l tiles
        grid_x = 1 if self.do_none_red else kv_splits
        grid = (grid_x, grid_y, grid_z)
        cluster_x = kv_splits if self.do_atomic_red else 1
        cluster = (cluster_x, 1, 1)

        self.decode(
            # MMA
            blk_tile_shpd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            o_smem_dtype,
            # Q
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            # K
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            # V
            smem_layout_v,
            tma_atom_v,
            tma_tensor_v,
            # O
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            # Rest
            mL_nl,
            mL_partial_nl,
            mM_partial_nl,
            mSink,
            mask_config,
            scale_s_log2_e,
            scale_o,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=cluster,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            self.launch_reduction(
                self.headdim,
                self.reduction_splits,
                self.red_vec_bits,
                self.red_d_ctas,
                self.red_max_groups,
                o_bshd,
                l_bsh,
                o_partial_bshd,
                l_partial_bsh,
                m_partial_bsh,
                sink_h,
                scale_o,
                stream,
                enable_pdl,
            )

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shpd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        mma_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        smem_layout_v: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: cute.CopyAtom,
        mO: cute.Tensor,
        # Rest
        mL: Optional[cute.Tensor],  # LSE output (Float32, log2 base)
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mSink: Optional[cute.Tensor],  # (h_g, h_k)
        mask_config: AttentionMask,
        scale_s_log2_e: Float32,
        scale_o: Float32,
    ):
        ##############################
        # Static variables
        ##############################
        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = cutlass.memory.SmemAllocator()

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        q_dtype = k_dtype = mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)  # multimode tiler
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)

        # Static control flow
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        do_none_red = self.do_none_red
        store_lse = mL is not None
        use_sink = mSink is not None
        is_persistent = self.is_persistent

        # Persistence is for multi-wave, split-K is for sub-wave
        assert not (is_persistent and not do_none_red), "persistence doesn't support split-KV"

        ##############################
        # Warp specialization
        ##############################
        # Warp assignments
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = tma_kv_warp_id if not is_persistent else None
        warpgroup_id += 1

        softmax_warpgroups = 2
        softmax_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + softmax_warpgroups)
        )
        warpgroup_id += softmax_warpgroups

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        # Register allocations
        use_reg_reconfig = blk_tile_n > 16
        max_sw_regs_per_wg_thread = 256  # CUDA limitation
        max_hw_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        mma_tma_regs = 64
        softmax_regs = 120
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            max_hw_regs_per_wg_thread - mma_tma_regs - softmax_regs * 2,
        )
        assert (
            mma_tma_regs + softmax_regs * 2 + correction_regs
        ) <= max_hw_regs_per_wg_thread

        # Read thread indices
        grid_dim = cute.arch.grid_dim()
        block_idx = cute.arch.block_idx()
        if cutlass.const_expr(do_none_red):
            kv_splits = 1
            kv_split_idx = 0
        else:
            kv_splits = grid_dim[0]
            kv_split_idx = block_idx[0]
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1  # warp 0 does all mbarrier inits for now

        # Unpack multimodes
        seqlen = mK.shape[0]
        grouped_heads, prediction = mQ.shape[0]
        heads_k, batches = tiles_hb = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)

        def get_coords(
            block_idx,
            tiles_hp = tiles_hp,
            tiles_hb = tiles_hb,
        ):
            kv_split_idx, idx_hp, idx_hb = block_idx
            coord_hp = cute.idx2crd(idx_hp, tiles_hp)
            coord_hb = cute.idx2crd(idx_hb, tiles_hb)
            return coord_hp, coord_hb
        coord_hp, coord_hb = get_coords(block_idx)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb

        # Runtime control flow
        coord_kv, tiles_s = mask_config.get_tiles_kv(
            prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
        )
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = min(2, self.s_stages - 1)  # MMA KQ iters to hide first softmax
        exit_early = not is_persistent and kv_split_idx >= tiles_s
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Initialize named barriers
        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        tma_threads = warp_threads
        # Shared KV pipeline requires ordering MMA
        mma_order_kq_nbar = nbar(1, tma_threads + tma_threads)
        mma_order_vp_nbar = nbar(2, tma_threads + tma_threads)
        sM_producer_nbar = nbar(3, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(4, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(5, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(7, softmax_threads + correction_threads)
        sM_final_nbar = nbar(9, correction_threads + reduction_threads)
        sL_final_nbar = nbar(10, correction_threads + reduction_threads)
        sO_final_nbar = nbar(11, correction_threads + tma_threads)
        sM_mutex_nbar = nbar(12, softmax_threads * softmax_warpgroups)

        # named barrier stage helper
        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        # Alias thread cooperatives
        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        cta_cooperative = thr_cg(self.threads_per_cta)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative

        # Initialize cluster colmax + colsum mbar (even if this split exits early)
        if cutlass.const_expr(do_atomic_red):
            reduction_mbars_ptr = smem.allocate_array(Int64, max_reduction_iters * 2)
            if warp_idx == init_warp:
                if lane_idx < max_reduction_iters * 2:
                    mbar_ptr = reduction_mbars_ptr + lane_idx
                    arrive_count = 1
                    expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                    cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()

        # Initialize Q load mbarrier
        q_stages = smem_layout_q.shape[-1]
        q_stage_layout = cute.select(smem_layout_q, (*mma_modes, 3))
        q_stage_bytes = cute.size_in_bytes(q_dtype, q_stage_layout)
        q_pipeline_ptr = smem.allocate_array(Int64, q_stages * 2)
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=q_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=q_stage_bytes,
            barrier_storage=q_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        # Initialize pipelines
        kv_stages = smem_layout_k.shape[-1]
        kv_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        kv_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=kv_stage_bytes,
            barrier_storage=kv_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=s_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=p_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=mma_group,
            consumer_group=correction_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        clc_response_ptr = None
        if cutlass.const_expr(is_persistent):
            clc_mbar_ptr = smem.allocate_array(Int64, 2)
            clc_response_ptr = smem.allocate_array(Int128)
            clc_producer, clc_consumer = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbar_ptr,
                num_stages=1,
                producer_group=tma_group,
                consumer_group=cta_cooperative,
                tx_count=(Int128.width // 8),
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()

        def query_next_work(
            clc_producer: pipeline.PipelineProducer,
            clc_response_ptr=clc_response_ptr,
        ):
            clc_producer.acquire()
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(
                    clc_producer.current_handle().barrier,
                    clc_response_ptr,
                )

        def get_next_work(
            clc_consumer: pipeline.PipelineConsumer,
            clc_response_ptr=clc_response_ptr,
        ):
            clc_consumer.wait()
            bidx, bidy, bidz, valid = cute.arch.clc_response(clc_response_ptr)
            cute.arch.fence_view_async_shared()
            clc_consumer.release()
            return (bidx, bidy, bidz), Boolean(valid)

        ##############################
        # Smem Tensor Allocation
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # Q, K, V
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAsV = cute.make_tensor(
            tAsK.iterator, smem_layout_v.outer
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        # Place Q, P, M, L on the second sub-bank (128-227KiB)
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, #TILE_DK, q_stages)

        # S
        # (MMA_MN, #MMA_M=1, #MMA_N=1, #TILE_SM, s_stages)
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N=1, #MMA_K=TILE_S/MMA_K, p_stages)
        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_vp, blk_tile_nm, mma_dtype, p_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
            (mma_tile_n, mma_tile_k)
        )  # (MMA_NK, #MMA_N=1, #MMA_K=MMA_TILE_K/MMA_K, #TILE_SK=TILE_S/MMA_TILE_K, p_stages)
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=TILE_S/MMA_K, p_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, #TILE_SM, p_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        # Reuse Q smem for O TMA store
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, q_stages)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        # (MMA, #MMA_M, #MMA_N, #TILE_DM, q_stages/o_stages)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape[:-1], o_stages))

        # M - colmax
        sM_layout = cute.make_layout((blk_tile_n, q_stages))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx, 0] = -Float32.inf
        init_warp += 1

        # L - colsum
        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1

        # Sink
        if cutlass.const_expr(use_sink and not do_kernel_red):
            cpasync_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
            )
            sSink_layout = cute.make_layout((blk_tile_hp,), stride=((1, 0),))
            sSink = smem.allocate_tensor(acc_dtype, sSink_layout, svector_align)
            sSink_lane = cute.local_tile(sSink[((None, 0),)], (1,), (lane_idx,))
            gSink = cute.local_tile(mSink, (blk_tile_h,), (coord_hg, coord_hk))
            gSink_lane = cute.local_tile(gSink, (1,), (lane_idx,))
            if cutlass.const_expr(not is_persistent):
                if warp_idx == reduction_warp_id and lane_idx < blk_tile_h:
                    cute.copy(cpasync_atom, gSink_lane, sSink_lane)
                    cute.arch.cp_async_commit_group()
        else:
            sSink = None

        # per-thread colsum
        # (MMA_MN, #MMA_M=1, #MMA_N=1, o_stages)
        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

        # R - cluster reduction buffers for colmax + colsum
        if cutlass.const_expr(do_atomic_red):
            sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
            sR = smem.allocate_tensor(acc_dtype, sR_layout, svector_align)

        ##############################
        # Sync
        ##############################
        # Ensure visibility of cluster mbarriers
        if cutlass.const_expr(do_atomic_red):
            cute.arch.cluster_wait()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()
        assert init_warp < (self.threads_per_cta // warp_threads), (
            f"used {init_warp + 1} init warps, {self.threads_per_cta // warp_threads} warps available"
        )

        ##############################
        # Tmem tensor allocation
        ##############################
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtL = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        assert tmem_offset <= tmem_alloc_cols, (
            f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
        )

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == tma_kv_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Apply block tiler and slice
            gK = cute.local_tile(
                mK, tiler=(blk_tile_s, blk_tile_d), coord=(None, 0, None)
            )  # (TILE_S, TILE_D, #TILE_S, (h_k, b))
            gV = cute.local_tile(
                mV, tiler=(blk_tile_d, blk_tile_s), coord=(0, None, None)
            )  # (TILE_D, TILE_S, #TILE_S, (h_k, b))

            # Apply MMA tiler and MMA partition
            gK_mma = cute.flat_divide(
                gK, (mma_tile_m, mma_tile_k)
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_SM, #TILE_DK, #TILE_S, (h_k, b))
            gV_mma = cute.flat_divide(
                gV, (mma_tile_m, mma_tile_k)
            )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK, #TILE_S, (h_k, b))
            tAgK = thrblk_mma_kq.partition_A(
                gK_mma
            )  # (MMA, #MMA_M, #MMA_K, #TILE_SM, #TILE_DK, #TILE_S, (h_k, b))
            tAgV = thrblk_mma_vp.partition_A(
                gV_mma
            )  # (MMA, #MMA_M, #MMA_K, #TILE_DM, #TILE_SK, #TILE_S, (h_k, b))

            # #TILE_SM=TILE_S/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_DK=TILE_D/MMA_TILE_K
            # #TILE_DM=TILE_D/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_SK=TILE_S/MMA_TILE_K

            # Example with TILE_S=MMA_TILE_M=128, TILE_H=MMA_TILE_N=8, MMA_TILE_K=64, TILE_D=512
            # BMM1: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4,
            #       TILE_SM=1, #TILE_HN=1, #TILE_DK=8, #TILE_S=S/128
            # BMM2: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4,
            #       #TILE_DM=4, #TILE_HN=1, #TILE_SK=2, #TILE_S=S/128

            # TMA partition
            # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
            tAsK_tma, tAgK_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsK, 0, 3),
                gmem_tensor=cute.group_modes(tAgK, 0, 3),
            )

            tAsV_tma, tAgV_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsV, 0, 3),
                gmem_tensor=cute.group_modes(tAgV, 0, 3),
            )

            cute.arch.griddepcontrol_wait()

            next_work_valid = True
            while next_work_valid:
                # Work tile slice
                tAgK_hb = tAgK_tma[None, None, None, None, coord_hb]
                tAgV_hb = tAgV_tma[None, None, None, None, coord_hb]
                tAgK_hb = cute.domain_offset((None, None, None, coord_kv), tAgK_hb)
                tAgV_hb = cute.domain_offset((None, None, None, coord_kv), tAgV_hb)

                # Sequence loop
                prefetch_tiles = prefetch_iters * kv_splits
                kv_token = kv_producer.try_acquire()
                for s in cutlass.range(kv_split_idx, prefetch_tiles + tiles_s, kv_splits):
                    # Load K
                    if s < tiles_s:
                        tAgK_s = tAgK_hb[None, None, None, s]
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for dk in cutlass.range_constexpr(tiles_dk):
                                kv_handle = kv_producer.acquire_and_advance(kv_token)
                                kv_token = kv_producer.try_acquire()
                                cute.copy(
                                    tma_atom_k,
                                    tAgK_s[None, sm, dk],
                                    tAsK_tma[None, kv_handle.index],
                                    tma_bar_ptr=kv_handle.barrier,
                                )

                    # Load V
                    if s >= prefetch_tiles:
                        tAgV_s = tAgV_hb[None, None, None, s - prefetch_tiles]
                        for sk in cutlass.range_constexpr(tiles_sk):
                            for dm in cutlass.range_constexpr(tiles_dm):
                                kv_handle = kv_producer.acquire_and_advance(kv_token)
                                kv_token = kv_producer.try_acquire()
                                cute.copy(
                                    tma_atom_v,
                                    tAgV_s[None, dm, sk],
                                    tAsV_tma[None, kv_handle.index],
                                    tma_bar_ptr=kv_handle.barrier,
                                )

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Request next work from CLC
                    query_next_work(clc_producer)
                    clc_producer.advance()

                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Update tile coordinates
                    (_, coord_p), (coord_hk, coord_b) = coord_hp, coord_hb = get_coords(block_idx)
                    coord_kv, tiles_s = mask_config.get_tiles_kv(
                        prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
                    )
                    iters_s = tiles_s

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == tma_qo_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Slice and partition Q
            # (TILE_H, TILE_D, tiles_hp, tile_hb)
            gQ = cute.local_tile(
                mQ,
                tiler=(blk_tile_hp, blk_tile_d),
                coord=(None, 0, None),
            )
            # Apply MMA tiler and MMA partition
            # (MMA_TILE_N, MMA_TILE_K, #TILE_HN=1, #TILE_DK, tiles_hp, tiles_hb)
            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, mma_tile_k))
            gQ_mma = gQ_mma[None, None, 0, None, None, None]
            # (MMA, #MMA_N, #MMA_K, #TILE_DK, tiles_hp, tiles_hb)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            # (TMA, #TILE_DK, tiles_hp, tiles_hb)
            tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            # Slice and partition O
            # (TILE_D, TILE_H, tiles_hp, h_k, b)
            gO = cute.local_tile(
                mO,
                tiler=(blk_tile_d, blk_tile_hp),
                coord=(0, None, (None, None)),
            )
            # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN=1, tiles_hp, h_k, b)
            gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
            gO_mma = gO_mma[None, None, None, 0, None, None, None]
            # (TMA, #TILE_DM, tiles_hp, h_k, b)
            sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )
            # Partial mode is coalesced with batches
            if cutlass.const_expr(do_kernel_red):
                coord_b = kv_split_idx * batches + coord_b

            cute.arch.griddepcontrol_wait()

            # Load Q
            tBgQ_hpb = tBgQ_tma[None, None, coord_hp, coord_hb]
            q_handle = q_producer.acquire_and_advance()
            tBsQ_stage = tBsQ_tma[None, None, q_handle.index]
            cute.copy(tma_atom_q, tBgQ_hpb, tBsQ_stage, tma_bar_ptr=q_handle.barrier)

            next_work_valid = True
            while next_work_valid:
                # Slice this work's O
                gO_hpb = gO_tma[None, None, coord_hp, coord_hk, coord_b]
                sO_stage = sO_tma[None, None, q_handle.index]
                sM_stage = sM[None, q_handle.index]

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Slice this work's LSE
                    if cutlass.const_expr(store_lse):
                        gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))

                    # Load this work's sink
                    if cutlass.const_expr(use_sink):
                        gSink = cute.local_tile(mSink, (blk_tile_h,), (coord_hg, coord_hk))
                        gSink_lane = cute.local_tile(gSink, (1,), (lane_idx,))
                        if lane_idx < blk_tile_h:
                            cute.copy(cpasync_atom, gSink_lane, sSink_lane)
                            cute.arch.cp_async_commit_group()

                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Update tile coordinates
                    (coord_hg, coord_p), (coord_hk, coord_b) = coord_hp, coord_hb = get_coords(block_idx)

                    # Next work tile slice
                    tBgQ_hpb = tBgQ_tma[None, None, coord_hp, coord_hb]

                    if next_work_valid:
                        # Wait for previous O store to complete (Q/O smem is aliased)
                        cute.arch.cp_async_bulk_wait_group(0)
                        with cute.arch.elect_one():
                            q_consumer.release() # MMA KQ did wait
                        q_consumer.advance()

                        # Load next work's Q
                        skip_acquire = True  # O store wait is an implicit acquire
                        q_handle = q_producer.acquire_and_advance(skip_acquire)
                        tBsQ_stage = tBsQ_tma[None, None, q_handle.index]
                        cute.copy(tma_atom_q, tBgQ_hpb, tBsQ_stage, tma_bar_ptr=q_handle.barrier)

                    # Wait for sink
                    if cutlass.const_expr(use_sink):
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_warp()

                    # Reduce colmax and colsum, maybe store lse
                    self.reduction_none(
                        lane_store_max,
                        lane_idx,
                        sM_final_nbar,
                        sL_final_nbar,
                        sM_stage,
                        sL,
                        gL if store_lse else None,
                        sSink,
                        scale_o,
                    )

                # Store O to gmem
                sO_final_nbar.arrive_and_wait()
                cute.copy(tma_atom_o, sO_stage, gO_hpb)
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.griddepcontrol_launch_dependents()

        ##############################
        # MMA KQ (BMM1) Dispatch
        ##############################
        elif warp_idx == mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            next_work_valid = True
            while next_work_valid:
                # Wait for Q
                q_handle = q_consumer.wait_and_advance()

                # Sequence loop
                for s in cutlass.range(iters_s):
                    s_token = s_producer.try_acquire()

                    mma_order_vp_nbar.arrive_and_wait()
                    k_token = kv_consumer.try_wait()

                    s_handle = s_producer.acquire_and_advance(s_token)
                    for sm in cutlass.range_constexpr(tiles_sm):
                        tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = kv_consumer.wait_and_advance(k_token)
                            is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                            if is_last_iter:
                                mma_order_kq_nbar.arrive()
                            else:
                                k_token = kv_consumer.try_wait()

                            mmas_k = cute.size(tAsK.shape[2])
                            for mma_k in cutlass.range_constexpr(mmas_k):
                                cute.gemm(
                                    tiled_mma_kq,
                                    tCtS[mma_dice + (sm, s_handle.index)],
                                    tAsK_desc[None, None, mma_k, k_handle.index],
                                    tBsQ_desc[None, None, mma_k, dk, q_handle.index],
                                    tCtS[mma_dice + (sm, s_handle.index)],
                                )
                                if dk == 0 and mma_k == 0:
                                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                    s_handle.commit()

                    # Advance for MMA VP
                    if s >= prefetch_iters:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()

                # Tail loop
                for s in cutlass.range_constexpr(prefetch_iters):
                    mma_order_vp_nbar.arrive_and_wait()
                    mma_order_kq_nbar.arrive()
                    if is_persistent and s < iters_s:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Persistence has no KV splits; refresh this Q tile's interval.
                    (_, coord_p), _ = get_coords(block_idx)
                    coord_kv, tiles_s = mask_config.get_tiles_kv(
                        prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
                    )
                    iters_s = tiles_s

        ##############################
        # MMA VP (BMM2) Dispatch
        ##############################
        elif warp_idx == mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors, order barrier
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            mma_order_vp_nbar.arrive()

            next_work_valid = True
            while next_work_valid:
                # Prefetch loop
                for s in cutlass.range_constexpr(prefetch_iters):
                    if s < iters_s:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                    mma_order_kq_nbar.arrive_and_wait()
                    mma_order_vp_nbar.arrive()

                # Sequence loop
                for s in cutlass.range(iters_s):
                    p_token = p_consumer.try_wait()
                    o_token = o_producer.try_acquire()

                    # Advance for MMA KQ
                    if s < iters_s - prefetch_iters:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                    mma_order_kq_nbar.arrive_and_wait()
                    v_token = kv_consumer.try_wait()

                    p_handle = p_consumer.wait_and_advance(p_token)
                    o_handle = o_producer.acquire_and_advance(o_token)
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_consumer.wait_and_advance(v_token)
                            is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                            if is_last_iter:
                                mma_order_vp_nbar.arrive()
                            else:
                                v_token = kv_consumer.try_wait()

                            mmas_k = cute.size(tAsV.shape[2])
                            for mma_k in cutlass.range_constexpr(mmas_k):
                                cute.gemm(
                                    tiled_mma_vp,
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                    tAsV_desc[None, None, mma_k, v_handle.index],
                                    tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                )
                            v_handle.release()
                    p_handle.release()
                    o_handle.commit()

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Persistence has no KV splits; refresh this Q tile's interval.
                    (_, coord_p), _ = get_coords(block_idx)
                    coord_kv, tiles_s = mask_config.get_tiles_kv(
                        prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
                    )
                    iters_s = tiles_s

            # Wait for signal to dealloc tmem, then dealloc
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        ##############################
        # Softmax Dispatch
        ##############################
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(softmax_regs)

            # Initialize for dual warpgroups
            softmax_phase = (warpgroup_idx - 1) % softmax_warpgroups
            tL_producer_nbar = with_phase(tL_producer_nbar, softmax_phase)
            tL_consumer_nbar = with_phase(tL_consumer_nbar, softmax_phase)
            sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
            sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
            if softmax_phase == 1:
                s_consumer.advance()
                p_producer.advance()
                sM_release_nbar.arrive()
            sM_stage = sM[None, 0]

            # Construct copy atom for S
            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            tmem_repeat_op_s = tcgen05.Repetition(tmem_repeat_op_s)
            tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_repeat_op_s)
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
            # Tile atom and slice
            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            # Partition S and P
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #CPY_SM, stages)
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)
            # Slice unused modes
            tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
            tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)

            # Construct copy atom for L
            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)
            # Tile atom and slice
            tCtL_stage = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_stage)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            # Partition L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tStL = thr_load_l.partition_S(tCtL_stage)
            tSrL_shape = thr_load_l.partition_D(tCtL_stage).shape
            # Slice unused modes
            # (CPY,)
            tStL = tStL[None, 0, 0, 0]
            tSrL_shape = tSrL_shape[:1]

            next_work_valid = True
            while next_work_valid:
                # Mask configuration loop args
                range_args = mask_config.get_range_args(
                    prediction,
                    seqlen,
                    blk_tile_p,
                    blk_tile_s,
                    coord_p,
                    coord_kv,
                    tiles_s,
                    iters_s,
                    kv_splits,
                    kv_split_idx,
                    softmax_warpgroups,
                    softmax_phase,
                )
                num_mask_phases = len(range_args)

                # Masking phase loop over all sequence tiles
                for mask_phase in cutlass.range_constexpr(num_mask_phases):
                    # Sequence tile loop per masking phase
                    start, stop, step, is_masked = range_args[mask_phase]
                    for coord_s in cutlass.range(start, stop, step):
                        s_token = s_consumer.try_wait()
                        p_token = p_producer.try_acquire()

                        # Load S from tmem and notify BMM1
                        s_handle = s_consumer.wait_and_advance(s_token)
                        tStS_s = tStS[None, None, s_handle.index]
                        tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                        cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                        cute.arch.fence_view_async_tmem_load()
                        s_handle.release()

                        # Apply mask
                        if cutlass.const_expr(is_masked):
                            masked = cute.make_rmem_tensor(
                                (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                            )
                            masked.store(tSrS_s.load().reshape(masked.shape))
                            offset_p = coord_p * blk_tile_p
                            offset_s = coord_s * blk_tile_s + warpgroup_tidx
                            for sm in cutlass.range_constexpr(tiles_sm):
                                for p in cutlass.range_constexpr(blk_tile_p):
                                    idx_q = offset_p + p
                                    idx_kv = offset_s + sm * mma_tile_m
                                    is_oob_kv = mask_config.is_oob_kv(
                                        idx_q, idx_kv, prediction, seqlen
                                    )
                                    mask = -Float32.inf if is_oob_kv else Float32(0)
                                    masked_p = masked[None, p, sm]
                                    masked_p.store(masked_p.load() + mask)
                            scores = masked.load().reshape((blk_tile_n, tiles_sm))
                        else:
                            scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))

                        # Reduce colmax in thread RF
                        rM = cute.make_rmem_tensor_like(sM_stage)
                        rM.store(
                            scores.reduce(
                                cute.ReductionOp.MAX,
                                # prevent nan accumulations with masking, see explanation below
                                init_val=min_f32,
                                reduction_profile=(None, 0),
                            )
                        )
                        """ Why using min finite float32 to prevent nan accumulation works:

                        Assume inputs are well formed, ie. S=QK can only be finite or -inf after mask.
                        3 cases to consider, -inf S -> -inf S, -inf S -> finite S, finite S -> -inf S.
                        First iteration suppose S is -inf after mask. Running max goes to min_f32,
                        P=exp(-inf - min_f32)=exp(-inf)=0, 0 is accumulated into O, O remains 0.
                        Second iteration suppose S is -inf again. prev_max = cur_max = min_f32,
                        correction rescale = exp(prev_max-new_max) = exp(0) = 1. O was zero, remains 0.
                        Third iteration suppose S is finite. Running max goes to S since prev_max is min finite.
                        Correction=exp(min_f32-S)=exp(-inf)=0 or exp(finite)=finite. O is still 0, so finite * 0 = 0.
                        P=exp(S - max_S)=positive finite, O accumulates first non-zero result.
                        Fourth iteration suppose S is -inf again. Running max stays finite and unchanged,
                        correction rescale is 1, P=exp(-inf-finite)=0, so O is unscaled and accumulates 0.
                        After we see the first finite S, all subsequent -inf S will not affect O/max/correction.
                        If finite S is never seen, then partial O remains 0 and will not contribute to final reduced O.
                        """

                        # Reduce colmax in warp RF
                        rM_lane = Float32(0)
                        for n in cutlass.range_constexpr(blk_tile_n):
                            rM[n] = warp_fmax(rM[n])  # warp reduction
                            # Avoid dynamic register indexing (creates spills)
                            if n == lane_idx:
                                rM_lane = rM[n]
                        rM_lane *= scale_s_log2_e  # apply scale

                        # Reduce colmax in smem
                        sM_acquire_nbar.arrive_and_wait()
                        sM_consumer_nbar.arrive_and_wait()
                        if lane_store_max:
                            smem_fmax(sM_stage.iterator + sM_stage.layout(lane_idx), rM_lane)

                        # Wait for colmax and load
                        sM_producer_nbar.arrive_and_wait()
                        colmax = sM_stage.load()
                        sM_release_nbar.arrive()

                        # Wait for empty P buffer
                        # Here so we can interleave ex2 with convert ops
                        p_handle = p_producer.acquire_and_advance(p_token)
                        tSsP_s = tSsP[None, None, p_handle.index]

                        # Compute online softmax
                        probs = exp2(scale_s_log2_e * scores - colmax)

                        # Store P to smem and notify BMM2
                        tSsP_s.store(probs.to(mma_dtype).reshape(tSsP_s.shape))
                        cute.arch.fence_view_async_shared()
                        p_handle.commit()

                        # Accumulate per-thread colsum
                        # TensorSSA.reduce doesn't elide the zero add
                        colsum = probs[None, 0]
                        for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                            colsum += probs[None, sm]
                        tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                        tSrL.store(colsum.reshape(tSrL.shape))

                        # Store per-thread colsum to tmem
                        tL_consumer_nbar.arrive_and_wait()
                        cute.copy(tmem_store_atom_l, tSrL, tStL)
                        cute.arch.fence_view_async_tmem_store()
                        tL_producer_nbar.arrive()

                        # Advance again for dual warpgroups
                        s_consumer.advance()
                        p_producer.advance()

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Which warpgroup goes first for next work
                    flip_phase = iters_s % softmax_warpgroups == 1
                    if flip_phase:
                        softmax_phase ^= 1

                    # Reset next sM for next work
                    q_consumer.advance()
                    qo_index = q_consumer.current_handle().index
                    sM_stage = sM[None, qo_index]
                    if softmax_phase == 0 and warpgroup_widx == 0 and lane_store_max:
                        sM_stage[lane_idx] = -Float32.inf

                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Update tile coordinates
                    (_, coord_p), _ = coord_hp, _ = get_coords(block_idx)
                    coord_kv, tiles_s = mask_config.get_tiles_kv(
                        prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
                    )
                    iters_s = tiles_s

        ##############################
        # Correction Dispatch
        ##############################
        elif warpgroup_idx == correction_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_increase(correction_regs)

            # Select copy atoms for O and L
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
            # Tile atoms and slice
            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
            # Partition O and L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, o_stages)
            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, o_stages)
            tOtL = thr_load_o.partition_S(tCtL)
            # Slice unused modes
            tOtO = tOtO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, o_stages)
            tOsO = tOsO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, q_stages)
            tOtL = tOtL[None, 0, 0, 0, None]  # (CPY, o_stages)

            # colsum load helper
            def colsum_load(
                phase,
                blk_tile_n=blk_tile_n,
                tOtL=tOtL,
                tOrO_shape=tOsO.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)

            # Initialize O and colsum
            tOrO_zero = cute.make_rmem_tensor(tOsO.shape[:-1], acc_dtype)
            tOrO_zero.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO_zero, tOtO[None, None, phase])
            cute.arch.fence_view_async_tmem_store()

            # Initialize consumer barriers
            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()
            sM_stage = sM[None, 0]

            next_work_valid = True
            while next_work_valid:
                # Initialize colsum in RF
                colsum_total = cute.make_rmem_tensor((blk_tile_n,), Float32)
                colsum_total.fill(Float32(0))

                # Load colmax of s-2, s-1
                sM_lane_prev_prev = sM_lane_prev = -Float32.inf
                for s in cutlass.range_constexpr(o_stages):
                    sM_lane_prev_prev = sM_lane_prev
                    if not (s == 1 and iters_s == 1):
                        sM_producer_nbar.arrive_and_wait()
                        if lane_store_max:
                            sM_lane_prev = sM_stage[lane_idx]
                        sM_consumer_nbar.arrive()

                # Sequence loop
                for s in cutlass.range(iters_s - o_stages):
                    # Load colsum of s-2
                    o_index = o_consumer.current_handle().index
                    colsum_s = colsum_load(o_index)

                    # Load colmax of s
                    sM_producer_nbar.arrive_and_wait()
                    if s == iters_s - o_stages - 1:  # Notify for final colmax
                        sM_final_nbar.arrive()
                    sM_lane = Float32(0)
                    if lane_store_max:
                        sM_lane = sM_stage[lane_idx]
                    sM_consumer_nbar.arrive()

                    # Wait for O of s-2
                    # Here so we can interleave shuffle_sync with correction muls
                    o_token = o_consumer.try_wait()
                    o_handle = o_consumer.wait_and_advance(o_token)

                    # Compute correction of s-2
                    correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                    correction = cute.make_rmem_tensor_like(sM_stage)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                    correction = correction.load()

                    # Correct O of s-2 and notify MMA VP
                    correction_o = correction.reshape(tOsO.shape[:1])
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOtO_dm = tOtO[None, dm, o_handle.index]
                        tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                        cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                        tOrO_dm.store(correction_o * tOrO_dm.load())
                        cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                    # Correct colsum of s-2
                    colsum_s *= correction

                    # Compute correction of s-1
                    correction_lane = exp2(sM_lane_prev - sM_lane)
                    correction = cute.make_rmem_tensor_like(sM_stage)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                    correction = correction.load()

                    # Correct total colsum and accumulate colsum s-2
                    colsum_total.store(correction * colsum_total.load() + colsum_s)

                    # Advance loop
                    sM_lane_prev_prev, sM_lane_prev = sM_lane_prev, sM_lane

                # Notify for final colmax if we didn't enter loop
                if iters_s <= o_stages:
                    sM_final_nbar.arrive()

                # Compute correction of s-1
                correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
                correction = cute.make_rmem_tensor_like(sM_stage)
                for n in cutlass.range_constexpr(blk_tile_n):
                    correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                correction = correction.load()

                # region: Correct and accumulate final colsum
                # Accumulate in thread RF
                prev_index = o_consumer.current_handle().index
                if iters_s == 1:
                    prev_index ^= 1
                colsum_final = colsum_total.load()
                colsum_final += colsum_load(prev_index ^ 1)
                if iters_s > 1:
                    colsum_final += colsum_load(prev_index) * correction
                # Reduce colsum in warp RF
                rL_lane = Float32(0.0)
                for n in cutlass.range_constexpr(blk_tile_n):
                    rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                    if n == lane_idx:
                        rL_lane = rL_n
                # Store partial colsum in smem and notify
                if lane_store_max:
                    sL[lane_idx, warpgroup_widx] = rL_lane
                # Wait to ensure reduction warp has reset sM_final_nbar
                sL_final_nbar.arrive_and_wait()
                # endregion: final colsum

                # region: Correct and accumulate final output
                # Load O of s-1, s
                tOrO = cute.make_rmem_tensor((*tOsO.shape[:-1], o_stages), acc_dtype)
                for s in cutlass.range_constexpr(o_stages):
                    if not (s == 0 and iters_s == 1):
                        o_handle = o_consumer.wait_and_advance()
                        tOtO_s = tOtO[None, None, o_handle.index]
                        tOrO_s = tOrO[None, None, s]
                        cute.copy(tmem_load_atom_o, tOtO_s, tOrO_s)
                        cute.arch.fence_view_async_tmem_load()
                        # Reset O tmem for next work
                        if cutlass.const_expr(is_persistent):
                            cute.copy(tmem_store_atom_o, tOrO_zero, tOtO_s)
                            cute.arch.fence_view_async_tmem_store()
                        o_handle.release()  # Final release signals tmem dealloc
                # reshape and accumulate
                tOrO_final = tOrO[None, None, 1].load()
                output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
                if iters_s > 1:
                    tOrO_prev = tOrO[None, None, 0].load()
                    output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
                    output_final += correction * output_prev
                # endregion: final output

                # Apply final normalization
                if cutlass.const_expr(do_atomic_red or do_none_red):
                    # final normalization stored in sM
                    sM_final_nbar.arrive_and_wait()
                    normalization = sM_stage.load()
                    output_final *= normalization

                # Slice smem QO stage corresponding to this work
                qo_index = q_consumer.current_handle().index
                tOsO_stage = tOsO[None, None, qo_index]

                # Store O to smem and notify
                # TODO: shuffle in tmem, transpose to smem
                tOsO_stage.store(output_final.to(o_dtype).reshape(tOsO_stage.shape))
                cute.arch.fence_view_async_shared()
                sO_final_nbar.arrive()

                next_work_valid = False
                if cutlass.const_expr(is_persistent):
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    # Persistence has no KV splits; refresh this Q tile's interval.
                    (_, coord_p), _ = get_coords(block_idx)
                    coord_kv, tiles_s = mask_config.get_tiles_kv(
                        prediction, seqlen, blk_tile_s, blk_tile_p, coord_p
                    )
                    iters_s = tiles_s

                    # Update colmax smem stage
                    q_consumer.advance()
                    qo_index = q_consumer.current_handle().index
                    sM_stage = sM[None, qo_index]

        ##############################
        # Reduction Dispatch
        ##############################
        if not is_persistent and warp_idx == reduction_warp_id:
            # Wait for sink
            if cutlass.const_expr(sSink is not None):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()

            if cutlass.const_expr(do_kernel_red):
                self.reduction_epilogue(
                    blk_tile_hp,
                    coord_hp,
                    coord_hb,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM[None, 0],
                    sL,
                    mM_partial,
                    mL_partial,
                )
            elif cutlass.const_expr(do_atomic_red):
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                self.reduction_cluster(
                    blk_tile_n,
                    kv_splits,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    reduction_mbars_ptr,
                    sM[None, 0],
                    sL,
                    sR,
                    gL,
                    sSink,
                    scale_o,
                )
            elif cutlass.const_expr(do_none_red):
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                self.reduction_none(
                    lane_store_max,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM[None, 0],
                    sL,
                    gL,
                    sSink,
                    scale_o,
                )

        return

    @staticmethod
    @cute.jit
    def reduction_none(
        lane_store_max: bool,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        gL: Optional[cute.Tensor],
        sSink: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        store_lse = gL is not None
        colmax = Float32(0)
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                colmax = sM[lane_idx]
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            if cutlass.const_expr(sSink is not None):
                colsum += exp2(log2_e * sSink[lane_idx] - colmax)
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive_and_wait()  # wait in case this is a persistent loop
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                gL[lane_idx] = colmax + cute.math.log2(colsum)

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        # get gmem colmax + colsum to store to
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        # tile predication
        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        # gmem predication
        grouped_heads, prediction = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        # Store partial colmax. global colmax is constructed in reduction kernel.
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            gM_partial[lane_idx] = sM[lane_idx]

        # Load partial colsum and reduce
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane

    @staticmethod
    @cute.jit
    def reduction_cluster(
        blk_tile_n: int,
        kv_splits: Int32,
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        reduction_mbars_ptr: cute.Pointer,
        sM: cute.Tensor,
        sL: cute.Tensor,
        sR: cute.Tensor,
        gL: Optional[cute.Tensor],
        sSink: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        acc_dtype = sM.dtype
        assert acc_dtype.width == 32, "DSMEM reduction requires 32-bit accumulators"
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        lane_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            acc_dtype,
            num_bits_per_copy=copy_vec_bits,
        )
        dsmem_store_r = cute.make_tiled_copy(
            lane_copy_atom,
            cute.make_ordered_layout(
                (dsmem_store_threads, dsmem_store_values), order=(1, 0)
            ),
            (blk_tile_n,),
        )
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)  # (CPY, #CPY)
        tRsL = thr_store_r.partition_S(sL)  # (CPY, #CPY, warpgroup_warps)
        tRsR = thr_store_r.partition_S(sR)  # (CPY, #CPY, max_red_iters, 2)
        payload_codec = cute.dsmem.B32PayloadCodec(
            [(f"value_{j}", Float32) for j in range(dsmem_store_values)]
        )

        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrL_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)

        # Wait for last colmax
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    local_mbar = reduction_mbars_ptr + i
                    cute.dsmem.store_to_peer(
                        tRsR_local.iterator,
                        tRrM_final,
                        payload_codec,
                        local_mbar,
                        peer_idx,
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])

        # Wait for last colsum
        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            # Warpgroup reduction
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()

            # Compute final correction and correct local colsum
            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    cute.dsmem.store_to_peer(
                        tRsR_local.iterator,
                        tRrL_local,
                        payload_codec,
                        local_mbar,
                        peer_idx,
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()

            if cutlass.const_expr(sSink is not None):
                tRsSink = thr_store_r.partition_S(sSink)  # (CPY, #CPY)
                tRrSink = tRsSink.load().reshape(tRrM_final.shape)
                sink_prob = exp2(log2_e * tRrSink - tRrM_final.load())
                colsum += sink_prob.reshape(colsum.shape)

            # Divide by final colsum and store
            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load() * scale_o)

            # Save final colsum for LSE
            if cutlass.const_expr(gL is not None):
                tRrL_final.store(colsum)

        # Notify for final correction
        sM_final_nbar.arrive()

        # Compute and store LSE
        if cutlass.const_expr(gL is not None):
            tRgL = thr_store_r.partition_D(gL)  # (CPY, #CPY=1)
            if kv_split_idx == 0 and is_reduction_lane:
                lse = tRrM_final.load() + cute.math.log2(tRrL_final.load())
                tRgL.store(lse)

    @staticmethod
    def setup_reduction_attrs(
        headdim: int,
        splits: int,
        red_vec_bits: int = 64,
        red_d_ctas: int = 1,
        red_max_groups: int = 16,
    ):
        """Compile-time attributes for the split-K reduction kernel."""
        assert headdim % red_d_ctas == 0, (headdim, red_d_ctas)
        assert red_vec_bits == 64 or red_vec_bits == 128
        d_per_blk = headdim // red_d_ctas
        d_per_thr = min(max(1, red_vec_bits // Float32.width), d_per_blk)
        assert d_per_blk % d_per_thr == 0
        thr_per_grp = d_per_blk // d_per_thr

        # Split-parallel groups. Each group owns a strided slice of the splits and
        # holds that slice in registers. Pick largest exact divisor of `splits`
        # so every group has the same static slice length.
        max_groups = min(red_max_groups, 1024 // thr_per_grp)
        split_groups = 1
        for g in range(1, max_groups + 1):
            if splits % g == 0:
                split_groups = g

        per_grp = splits // split_groups
        thr_per_blk = thr_per_grp * split_groups
        smem_floats = 2 + split_groups * (d_per_blk + 1)
        return (
            d_per_blk,  # Number of output d elements per threadblock
            d_per_thr,  # Number of output d elements per thread
            thr_per_grp,  # Number of threads per split-parallel group
            thr_per_blk,  # Threadblock size
            split_groups,  # Number of split-parallel groups
            per_grp,  # Number of splits each group reduces
            smem_floats * Float32.width // 8,  # Smem size in bytes
        )

    ##############################
    # Reduction Kernel launch
    ##############################
    @staticmethod
    @cute.jit
    def launch_reduction(
        headdim: int,  # Head dimension
        splits: int,  # Number of kv splits
        red_vec_bits: int,  # Size of each thread's partial O vector
        red_d_ctas: int,  # Number of CTAs in the head dim
        red_max_groups: int,  # Maximum number of split-parallel groups
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],  # LSE output (log2 base) or None
        o_partial_bshd: cute.Tensor,  # partial O per kv split
        l_partial_bsh: cute.Tensor,  # partial colsum_p per kv split
        m_partial_bsh: cute.Tensor,  # partial colmax_s per kv split
        sink_h: Optional[cute.Tensor],  # Pre-scaled sink logits per head
        scale_o: Float32,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        (
            d_per_blk,
            d_per_thr,
            thr_per_grp,
            thr_per_blk,
            split_groups,
            per_grp,
            smem_bytes,
        ) = GqaDecodeTcgen05.setup_reduction_attrs(
            headdim, splits, red_vec_bits, red_d_ctas, red_max_groups
        )
        _, b, s_q, h_q, d = o_partial_bshd.shape

        # Tile in headdim first
        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        l_hsb = reverse(l_bsh) if l_bsh is not None else None
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)

        d_blks = cute.ceil_div(d, d_per_blk)

        GqaDecodeTcgen05.reduction_kernel(
            (thr_per_grp, d_per_thr, d_per_blk),
            split_groups,
            per_grp,
            o_dhsb,
            l_hsb,
            o_partial_dhsb,
            l_partial_hsb,
            m_partial_hsb,
            sink_h,
            scale_o,
        ).launch(
            grid=[s_q, d_blks * h_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        split_groups: cutlass.Constexpr,
        per_grp: cutlass.Constexpr,
        o_dhsb: cute.Tensor,
        l_hsb: Optional[cute.Tensor],
        o_partial_dhsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        sink_h: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        thr_per_grp, d_per_thr, d_per_blk = tile_d
        d, h_q, s_q, b, splits = o_partial_dhsb.shape
        coord_s, coord_dh, coord_b = cute.arch.block_idx()
        d_blks = cute.ceil_div(d, d_per_blk)
        d_blk_idx, coord_h = cute.idx2crd(coord_dh, (d_blks, h_q))
        tidx, _, _ = cute.arch.thread_idx()

        # (d lane, split group) decomposition of the CTA
        d_tidx = tidx % thr_per_grp
        grp = tidx // thr_per_grp

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + d_tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b, None)
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o[:-1])
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_o)

        gM_partial = cute.local_tile(m_partial_hsb, (1,), coord_o[1:])
        gL_partial = cute.local_tile(l_partial_hsb, (1,), coord_o[1:])

        smem_ptr = cute.arch.get_dyn_smem(Float32)

        scalar_layout = cute.make_layout(1)
        use_sink = sink_h is not None
        if cutlass.const_expr(use_sink):
            gSink = cute.local_tile(sink_h, (1,), (coord_h,))
        sSink = cute.make_tensor(smem_ptr, scalar_layout)
        sMax = cute.make_tensor(smem_ptr + 1, scalar_layout)

        # Per-group staging for the cross-group combine
        sAcc = cute.make_tensor(
            smem_ptr + 2,
            cute.make_layout((d_per_blk, split_groups), stride=(1, d_per_blk)),
        )
        sSum = cute.make_tensor(
            smem_ptr + 2 + d_per_blk * split_groups,
            cute.make_layout(split_groups),
        )
        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((thr_per_grp, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(d_tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)  # (CPY, #CPY=1, splits)
        tCgO_partial = tCgO_partial[None, 0, None]  # (CPY, splits)
        tCgO = thr_copy.partition_D(gO)  # (CPY, #CPY=1)
        tCgO = tCgO[None, 0]  # (CPY)
        tCrO_final = cute.zeros_like(tCgO, Float32)

        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait
        cute.arch.griddepcontrol_wait()

        if tidx == 0:
            sMax[0] = -Float32.inf
            if cutlass.const_expr(use_sink):
                cute.copy(cpasync_atom, gSink, sSink)
        cute.arch.cp_async_commit_group()

        # This group's strided slice of the splits axis: elements
        # grp, grp + split_groups, grp + 2 * split_groups, ...
        split_slice = cute.make_layout(per_grp, stride=split_groups)
        gM_grp = cute.composition(
            cute.domain_offset(grp, gM_partial[0, None]), split_slice
        )
        gL_grp = cute.composition(
            cute.domain_offset(grp, gL_partial[0, None]), split_slice
        )
        gO_grp = cute.composition(
            cute.domain_offset((0, grp), tCgO_partial), (None, split_slice)
        )

        rM = gM_grp.load()
        rL = gL_grp.load()
        rO = gO_grp.load()

        # Compute colmax: per-thread max over its own slice, then one smem atomic
        # per warp. A serial scan over all splits dominates the kernel otherwise.
        mx_thr = rM.reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        mx_thr = warp_fmax(mx_thr)
        cute.arch.sync_threads()
        if tidx % warp_threads == 0:
            smem_fmax(sMax.iterator, mx_thr)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        max_final = sMax[0]

        sum_grp = Float32(0)
        for j in cutlass.range_constexpr(per_grp):
            l_j = rL[j]
            # rM[j] = -inf means that the split is empty
            if rM[j] > -Float32.inf:
                correction = exp2(rM[j] - max_final)
                sum_grp += correction * l_j
                o_j = rO[None, j]
                tCrO_final += correction * o_j

        # Combine the split groups
        if cutlass.const_expr(split_groups > 1):
            tCsAcc = thr_copy.partition_D(sAcc[None, grp])[None, 0]
            tCsAcc.store(tCrO_final)
            if d_tidx == 0:
                sSum[grp] = sum_grp
            cute.arch.sync_threads()

        cute.arch.griddepcontrol_launch_dependents()

        if grp == 0:
            sum_final = Float32(0)
            if cutlass.const_expr(use_sink):
                sum_final = exp2(log2_e * sSink[0] - max_final)
            sum_final += sum_grp
            if cutlass.const_expr(split_groups > 1):
                for g in cutlass.range_constexpr(1, split_groups):
                    sum_final += sSum[g]
                    tCrO_final += thr_copy.partition_D(sAcc[None, g])[None, 0].load()

            if not_oob_d:
                # For rows which are fully masked, store 0
                if sum_final > Float32(0):
                    tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o
                else:
                    tCrO_final = cute.zeros_like(tCgO, Float32)
                tCgO.store(tCrO_final.to(o_dhsb.dtype))

            if cutlass.const_expr(l_hsb is not None):
                if d_blk_idx == 0 and d_tidx == 0:
                    l_hsb[coord_h, coord_s, coord_b] = max_final + cute.math.log2(
                        sum_final
                    )

        return
