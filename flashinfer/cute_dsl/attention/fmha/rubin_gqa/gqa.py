# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
This kernel is based on fmha.py and gqa_decode.py. It is meant to supplement rather than replace them.

Key design differences from fmha.py
1. Temporal pipeline instead of spatial pipeline.
fmha.py will load 2 tiles of Q and each softmax warpgroup processes 1 of 2 tiles to produce 2 indpendent O tiles.
gqa.py loads 1 tile of Q and both softmax warpgroups stride over KV tiles to compute a partial O, then reduce
their partial Os in epilogue to produce 1 tile of O. This can give better occupancy for short seqlen Q problems.
It is also beneficial for cases where fully tmem resident Q is preferred but GPU doesnt have enough
tmem for 2 tiles of Q, e.g. fp8 MLA prefill on SM107. Note that fmha.py supports partial Q residency.
2. Support for M=256 2CTA MMA. This gives smem savings for a deeper KV pipeline and reduces TMA unit work per SM.

Key design differences from gqa_decode.py
1. GQA packing in gemm-M instead of gemm-N and support for M=64 1CTA MMA.
This allows gqa.py to support decode problems with large grouped heads and/or large speculation
e.g. 16 q heads per k head with MTP 3 gives grouped_heads * seqlen_q = 64 (seqlen_q = MTP + 1)
This covers a previous perf gap where gqa_decode.py is optimized for grouped_heads * seqlen_q <= 32
and fmha.py is optimized for seqlen_q >= 256. (Functional problem size coverage is the same).
GQA packing also reduces wasted computations for causal/sliding window mask problems
since each CTA processes less seqlen_q compared to non-GQA packed version.
2. Unified paged + non-paged KV implementation. gqa.py does not move paged implementation to a separate kernel.

Major functional gaps from fmha.py
1. no support for varlen Q or packed varlen KV. varlen KV is supported with paging or padding
2. no support for emulated ex2, important for SM100 perf
3. no support for skip softmax aka BLASST
4. support for split KV (fmha.py doesn't support). Currently only 2nd kernel reduction is supported
"""


import math
from typing import Literal, Type, Tuple, cast
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass import const_expr, range_constexpr
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
pass
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)


from cutlass.experimental.primitives import tcgen05_fence, Tcgen05Fence

from cutlass.cute.typing import (
    Float32,
    BFloat16,
    Float16,
    Float8E4M3FN,
    Float8E5M2,
    Int32,
    Int64,
    Int128,
    Boolean,
    Optional,
)



from .mask import AttentionMask, DenseMask, CausalMask, SlidingWindowMask, NoMask
from .gqa_decode import GqaDecodeTcgen05 as GqaDecode

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4  # log2(16)
debug = False

# Math helpers
min_f32 = Float32(
    -3.4028234663852886e38
)  # lowest finite float32, prevent nans with masking
log2_e = math.log2(math.e)  # change exponential base
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
warp_fmin = partial(cute.arch.warp_redux_sync, kind="fmin", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")

# Arch queries
arch = BaseDSL._get_dsl().get_arch_enum()
resource_arch = f"sm_{arch.major}{arch.minor}"
tmem_red_supported = (
    arch.can_run_binary_built_for(Arch.sm_103f)
    or arch.can_run_binary_built_for(Arch.sm_110f)
)
mma_k64_supported = arch.can_run_binary_built_for(Arch.sm_107f)

# fp16 softmax
fp16_softmax_supported = arch.can_run_binary_built_for(Arch.sm_107f)
from .helpers.fmha_helpers import (
    add_packed_f16x2_f32x2_f32x2,
    with_collector_op,
)

@cute.jit
def sub_f32_to_f16(a : cute.TensorSSA, b: Float32):
    # TODO: OCG should lower to this if it sees f32 add followed by f16 cvt
    assert a.dtype == Float32
    assert fp16_softmax_supported
    c = cute.make_rmem_tensor_like(a, Float16)
    for i in range_constexpr(0, cute.size(c), 2):
        c[i + 0], c[i + 1] = add_packed_f16x2_f32x2_f32x2(
            (a[i + 0], a[i + 1]), (-b, -b)
        )
    return c.load()

@cute.jit
def rcp_approx(a : cute.TensorSSA):
    assert a.dtype == Float32
    b = cute.make_rmem_tensor_like(a, Float32)
    for i in range_constexpr(cute.size(a.shape)):
        b[i] = cute.arch.rcp_approx(a[i])
    return b.load()


from cutlass._mlir.dialects import llvm

def warp_sync_with_values(tensor : cute.TensorSSA | cute.Tensor, *, loc=None, ip=None):
    """Materialize a register fragment before a warp sync.

    Pass every packed 32-bit word as an input to side-effecting inline assembly
    so LLVM/libNVVM cannot sink the fragment's computations past the warp sync.

    Use before a blocking pipeline acquire to keep softmax and conversion work
    ahead of the wait.
    """
    if isinstance(tensor, cute.Tensor):
        tensor = tensor.load(loc=loc, ip=ip)
    words = tensor.bitcast(Int32, loc=loc, ip=ip)

    operands = [
        words[i].ir_value(loc=loc, ip=ip)
        for i in range(cute.size(words.shape))
    ]

    llvm.inline_asm(
        res=None,
        operands_=operands,
        asm_string='bar.warp.sync  0xffffffff;\n',
        constraints=",".join(["r"] * len(operands) + ["~{memory}"]),
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _split_headdim(headdim: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(headdim, int):
        return headdim, headdim
    if len(headdim) != 2:
        raise ValueError("headdim must be an integer or a pair of integers")
    return headdim

# Aux fusion
def sigmoid_gate(o_f32 : cute.TensorSSA, aux : cute.TensorSSA):
    gate_values = aux.to(Float32)
    # exp(-abs(x)) never overflows; retain the small negative
    # tail directly rather than subtracting it from one.
    exp_neg_abs = exp2(-log2_e * cute.math.abs(gate_values))
    inv_denom = rcp_approx(1.0 + exp_neg_abs)
    sigmoid_values = cute.where(
        gate_values >= 0.0, inv_denom, exp_neg_abs * inv_denom
    )
    return o_f32 * sigmoid_values

class GqaTcgen05:
    """
    Grouped query attention with contiguous or paged KV for 5th-generation Tensor cores.

    Supports MQA and MHA as special cases of GQA (h_k==1 and h_q==h_k respectively).

    Also supports non-latent space MLA (uncompressed QKV), assuming the
    rope dim has already been concatenated to the uncompressed QK head dimension.
    QK dimensions must be multiples of 64; VO dimensions must be 64 or 128.
    Use headdim QK = headdim + ropedim, headdim VO = headdim.
    """

    def __init__(
        self,
        headdim: int | tuple[int, int],
        grouped_head_tile: int,
        sequence_qo_tile: int,
        page_size: Optional[int] = None,
        *,
        sequence_kv_tile: int = 128,
        kv_splits: int = 1,
        reduction_mode: Literal["kernel", "atomic", "none"] = "none",
        softmax_dtype=Float32,
        softmax_rescale_threshold: Optional[float] = 8.,
        softmax_quantization_scale: Optional[float] = None,
        use_softmax_fma=False,
        use_softmax_scale=True,
        disable_persistence=False,
        disable_mma_rowsum=False,
        aux_fn=None,
    ):
        """
        Parameters
        ----------
        headdim
            Q/K head dimension, or a ``(Q/K, V/O)`` head dimension pair.
        grouped_head_tile
            Grouped heads per threadblock (GQA packing factor). Use 1 to disable GQA packing.
            Doesn't need to match total problem grouped heads (ie can be more or less).
        page_size
            KV tokens per physical page (8, 16, 32, or 64), or None for
            contiguous KV storage.
        sequence_qo_tile
            QO tokens per threadblock.
        sequence_kv_tile
            KV tokens per threadblock per mainloop iteration.
        kv_splits
            Compile-time threadblocks per sequence (flash decoding).
        reduction_mode
            Split-KV reduction algorithm:
              - ``"kernel"``: deterministic kernel reduction with partial result workspace.
              - ``"atomic"``: cluster reduction with atomic adds, no workspace.
              - ``"none"``: no split-KV, flash decoding disabled.
        softmax_dtype
            The datatype to perform softmax computations in. Float32 or Float16.
            MMA accumulators are always Float32.
        softmax_rescale_threshold
            Online softmax conditional rescaling threshold as described in FA4.
            Threshold is with respect to scale_s_log2_e scaled logits.
            Use `None` to disable conditional rescaling.
        softmax_quantization_scale
            Apply a scale to softmax output before quantizing for PV MMA.
            Inverse is applied in final normalization. Useful for FP8 precision.
        use_softmax_fma
            Use fused multiply add when computing softmax logits (scale * QK - (scale * max(QK))).
            Less instructions but potentially unstable for very large logits due to rounding.
        use_softmax_scale
            Optionally disable applying scale_s_log2_e (Q was already prescaled).
            use_softmax_fma is ignored if this is disabled.
        aux_fn
            Binary aux fusion functor, applying O_f32 = aux_Fn(O_f32, Aux),
            where Aux is the same shape and type as O in gmem. Must accept TensorSSA args.
            Requires no split-KV reduction and kv_splits=1.
        """
        # Tiling config
        if page_size not in (None, 8, 16, 32, 64):
            raise ValueError("page_size must be None, 8, 16, 32, or 64")
        self.page_size = page_size
        self.headdim_qk, self.headdim_vo = _split_headdim(headdim)
        self.grouped_head_tile = grouped_head_tile
        self.sequence_qo_tile = sequence_qo_tile
        self.sequence_kv_tile = sequence_kv_tile
        blk_tile_m = grouped_head_tile * sequence_qo_tile

        # Split KV config
        self.kv_splits = kv_splits
        self.do_kernel_red = reduction_mode == "kernel"
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_none_red = reduction_mode == "none" or reduction_mode is None

        # Softmax config
        self.softmax_dtype = softmax_dtype
        self.rescale_threshold = softmax_rescale_threshold
        self.quantization_scale = softmax_quantization_scale
        self.use_softmax_fma = use_softmax_fma
        self.use_softmax_scale = use_softmax_scale

        # Kernel config
        self.is_persistent = self.do_none_red and not disable_persistence
        self.disable_mma_rowsum = disable_mma_rowsum
        self.threads_per_cta = 4 * warpgroup_threads
        self.aux_fn = aux_fn

        # Post-init validation
        if self.headdim_qk <= 0 or self.headdim_qk % 64 != 0:
            raise ValueError(
                f"Q/K head dimension must be a positive multiple of 64, got {self.headdim_qk}"
            )
        if self.headdim_vo not in (64, 128):
            raise ValueError(
                f"V/O head dimension must be 64 or 128, got {self.headdim_vo}"
            )
        if blk_tile_m not in (64, 128, 256):
            raise ValueError(
                f"grouped_head_tile * sequence_qo_tile must be 64, 128, or 256, got {blk_tile_m}"
            )
        if self.sequence_kv_tile not in (64, 128):
            raise ValueError(
                f"sequence_kv_tile must be 64 or 128, got {self.sequence_kv_tile}"
            )
        mma_ctas = 2 if blk_tile_m == 256 else 1
        if page_size is not None and self.sequence_kv_tile % (mma_ctas * page_size):
            raise ValueError("each CTA's KV tile must contain a whole number of pages")
        if not (self.do_kernel_red ^ self.do_atomic_red ^ self.do_none_red):
            raise ValueError(
                "reduction_mode must be 'kernel', 'atomic', 'none', or None, "
                f"got {reduction_mode!r}"
            )
        if self.aux_fn is not None and (not self.do_none_red or self.kv_splits != 1):
            raise ValueError("aux fusion requires reduction_mode='none' and kv_splits=1")
        if self.do_atomic_red:
            raise NotImplementedError("atomic reduction mode is not implemented")
        if not self.do_none_red and blk_tile_m > 128:
            raise ValueError(
                f"split-KV reduction requires tile M <= 128, got {blk_tile_m}"
            )
        if softmax_dtype not in (Float32, Float16):
            raise ValueError(
                f"softmax_dtype must be Float32 or Float16, got {softmax_dtype}"
            )
        if softmax_dtype == Float16 and not fp16_softmax_supported:
            raise ValueError(
                f"Float16 softmax is not supported on {resource_arch}"
            )
        # Rescale threshold and quantization scale have similar effects on the
        # distribution of P; for example, both can distribute P in (0, 256].
        if (
            softmax_rescale_threshold is not None
            and softmax_quantization_scale is not None
        ):
            raise ValueError(
                "softmax_rescale_threshold and softmax_quantization_scale "
                "cannot both be set"
            )

    # Runtime implementable check
    def can_implement(
        self,
        kv_splits,
        q_shape_bshd,
        k_shape_bshd,
        v_shape_bshd,
        qkv_dtype,
        o_dtype,
        mask_config,
        seqlen_kv=None,
    ):
        """Validate shapes and a shared KV length or one logical length per batch."""
        b_q, s_q, h_q, d_q = q_shape_bshd
        b_k, s_k, h_k, d_k = k_shape_bshd
        b_v, s_v, h_v, d_v = v_shape_bshd

        if qkv_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")

        if qkv_dtype not in (BFloat16, Float16, Float8E4M3FN, Float8E5M2):
            raise TypeError("unsupported QKV dtype")

        if d_q != self.headdim_qk or d_k != self.headdim_qk:
            raise ValueError(
                f"headdim_q({d_q}) and headdim_k({d_k}) must be "
                f"{self.headdim_qk}"
            )

        if d_v != self.headdim_vo:
            raise ValueError(
                f"headdim_v({d_v}) must be {self.headdim_vo}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if (b_k, s_k, h_k) != (b_v, s_v, h_v):
            raise ValueError(
                f"K shape prefix ({b_k}, {s_k}, {h_k}) and "
                f"V shape prefix ({b_v}, {s_v}, {h_v}) mismatch"
            )

        if self.page_size is not None:
            if s_k != self.page_size:
                raise ValueError(f"KV page size must be {self.page_size}, got {s_k}")
            if seqlen_kv is None:
                raise ValueError("paged KV requires a logical KV sequence length")
        else:
            if b_q != b_k:
                raise ValueError(f"batches_q({b_q}) and batches_k({b_k}) mismatch")
            if seqlen_kv is None:
                seqlen_kv = s_k

        if isinstance(seqlen_kv, (tuple, list)):
            if len(seqlen_kv) != b_q:
                raise ValueError("provide one logical KV sequence length per batch")
            seqlens = seqlen_kv
        else:
            seqlens = (seqlen_kv,)
        for seq in seqlens:
            if seq <= 0:
                raise ValueError("KV sequence lengths must be positive")
            if self.page_size is None and seq > s_k:
                raise ValueError("KV sequence length must fit the KV tensor")

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

        if self.do_kernel_red and kv_splits != self.kv_splits:
            raise ValueError(
                f"kernel reduction requires kv_splits == compiled kv_splits({self.kv_splits}), got {kv_splits}"
            )

        for seq in set(seqlens):
            mask_config.can_implement(
                s_q, seq, self.sequence_qo_tile, self.sequence_kv_tile
            )

    # --------------------------------------------------------------
    # Attention Kernel launch
    # --------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        seqlens: cute.Tensor | Int32 | None,
        table_offsets: Optional[cute.Tensor],
        page_table: Optional[cute.Tensor],
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
        aux_bshd: Optional[cute.Tensor] = None,
    ):
        """
        Parameters
        ----------
        seqlens, table_offsets, page_table
            Scalar or per-batch Int32 KV lengths. For contiguous KV, None uses
            the tensor sequence dimension; table_offsets and page_table must
            be None. For paged KV, page_table is a flattened Int32 array of
            logical-to-physical page indices. Per-batch lengths require Int32
            table_offsets; scalar lengths use consecutive equal-size tables.
        q_bshd
            Q tensor in BSHD logical view. Strides can be BHSD.
        k_bshd, v_bshd
            Contiguous KV tensors in BSHD logical view (strides may be BHSD),
            or paged tensors of shape (page_count, page_size, heads_k, headdim).
            Tokens beyond each logical KV length must be zero padded.
        o_bshd
            Output tensor. Must be zero initialized for atomic reduction.
        aux_bshd
            Optional aux tensor, with the same BSHD shape and dtype as O.
            Applies O = aux_fn(o, aux) in the softmax epilogue before conversion.
            Requires reduction_mode='none' and kv_splits=1.
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
        # --------------------------------------------------------------
        # Calculate tile sizes
        # --------------------------------------------------------------
        mma_dtype = q_bshd.dtype
        acc_dtype = Float32
        sO_dtype = o_partial_bshd.dtype if self.do_kernel_red else o_bshd.dtype
        assert k_bshd.dtype == v_bshd.dtype == mma_dtype
        if const_expr(self.aux_fn is not None):
            assert aux_bshd is not None, "aux tensor expected"
            assert self.do_none_red and self.kv_splits == 1, "aux fusion and split-KV are mutually exclusive"
            assert aux_bshd.dtype == o_bshd.dtype
            assert cute.rank(aux_bshd) == cute.rank(o_bshd) == 4
        is_paged = page_table is not None
        assert is_paged == (self.page_size is not None), "page_size and page_table must be set together"
        if const_expr(is_paged):
            assert seqlens is not None, "paged KV requires logical KV lengths"
            assert k_bshd.shape[1] == v_bshd.shape[1] == self.page_size
            assert isinstance(seqlens, cute.Tensor) == (table_offsets is not None)
            assert page_table.dtype == Int32
            if const_expr(table_offsets is not None):
                assert table_offsets.dtype == Int32
        else:
            assert table_offsets is None, "contiguous KV does not use page-table offsets"
        if const_expr(isinstance(seqlens, cute.Tensor)):
            assert seqlens.dtype == Int32

        # Block tile sets the granularity at which CTAs (or CTA pairs) consume work (BMM1/BMM2)
        blk_tile_kv = self.sequence_kv_tile
        blk_tile_qo = self.sequence_qo_tile
        blk_tile_hg = self.grouped_head_tile
        blk_tile_dq = self.headdim_qk
        blk_tile_do = self.headdim_vo
        blk_tile_hg_qo = (blk_tile_hg, blk_tile_qo)  # multimode tiler
        blk_tile_m = blk_tile_hg * blk_tile_qo  # linearized tiler
        # MMA tile sets the granularity at which TMAs + MMAs are staged in smem
        mma_tile_m = min(blk_tile_m, 256)
        mma_qk_tile_mnk = (mma_tile_m, blk_tile_kv, blk_tile_dq)
        mma_pv_tile_mnk = (mma_tile_m, blk_tile_do, blk_tile_kv)
        # CTAs per MMA
        mma_ctas = 2 if mma_tile_m == 256 else 1
        min_mma_n = 16 if mma_ctas == 2 else 8
        use_mma_k64 = mma_k64_supported and mma_dtype.width and mma_tile_m != 64
        min_mma_k_bits = 512 if use_mma_k64 else 256
        min_mma_k = min_mma_k_bits // mma_dtype.width

        # --------------------------------------------------------------
        # Calculate stage counts
        # --------------------------------------------------------------
        # Fixed stage counts
        self.tS_stages = tS_stages = 2  # tmem S
        self.tP_stages = tP_stages = 2  # tmem P
        self.tO_stages = tO_stages = 2  # tmem O
        self.sQ_stages = sQ_stages = 2 if self.is_persistent else 1  # smem Q + O alias

        # Calculate tmem alloc
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(resource_arch)
        tmem_pool_cols = self.tmem_alloc_cols
        tmem_pool_cols -= tS_stages * blk_tile_kv
        tmem_pool_cols -= tO_stages * blk_tile_do
        # tP aliased to tS
        no_alias_tP = 64 in (mma_tile_m, blk_tile_kv, blk_tile_do)
        tP_cols = tP_stages * blk_tile_kv * mma_dtype.width // 32
        self.alias_tP = not no_alias_tP
        tmem_pool_cols -= 0 if self.alias_tP else tP_cols
        # Q persistent in tmem
        # FIXME: compile crash with tQ and M=64
        no_use_tQ = mma_tile_m == 64
        tQ_cols = blk_tile_dq * mma_dtype.width // 32
        self.use_tQ = tmem_pool_cols >= tQ_cols and not no_use_tQ
        tmem_pool_cols -= tQ_cols if self.use_tQ else 0
        # MMA P1
        no_use_mma_p1 = self.disable_mma_rowsum
        no_use_mma_p1 |= mma_tile_m == 64 or mma_dtype.width > 8  # MMA limited
        self.use_mma_p1 = tmem_pool_cols >= min_mma_n and not no_use_mma_p1
        self.tP1_stages = 2 if tmem_pool_cols >= 2 * min_mma_n else 1
        tP1_cols = self.tP1_stages * min_mma_n
        tmem_pool_cols -= tP1_cols if self.use_mma_p1 else 0
        assert tmem_pool_cols >= 0 or mma_tile_m == 64, f"exceed tmem capacity by {-tmem_pool_cols} cols"

        # Calculate smem alloc
        smem_alloc_bits = 0
        # K leads V by two tiles; reserve another slot for page prefetch.
        if const_expr(is_paged):
            self.pt_stages = 4
            pages_kv = blk_tile_kv // self.page_size
            smem_alloc_bits += self.pt_stages * pages_kv * Int32.width
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        blk_m_per_cta = blk_tile_m // mma_ctas
        mma_m_per_cta = mma_tile_m // mma_ctas
        # tmem ptr
        smem_alloc_bits += Int32.width
        # softmax mbars
        smem_alloc_bits += mbarrier_bits * 3
        if const_expr(aux_bshd is not None):
            smem_alloc_bits += mbarrier_bits
        # CLC
        if const_expr(self.is_persistent):
            smem_alloc_bits += mbarrier_bits * 2  # CLC full + empty
            smem_alloc_bits += Int128.width  # CLC response
        # sQ, sO
        sQ_stage_elts = blk_tile_m * blk_tile_dq // mma_ctas
        sO_stage_elts = blk_tile_m * blk_tile_do // mma_ctas
        sQ_stage_bits = sQ_stage_elts * mma_dtype.width
        sO_stage_bits = sO_stage_elts * sO_dtype.width
        sQO_stage_bits = max(sQ_stage_bits, sO_stage_bits)  # alias QO stages
        smem_alloc_bits += sQ_stages * sQO_stage_bits
        smem_alloc_bits += sQ_stages * pipe_stage_bits
        # tS, tP
        smem_alloc_bits += tS_stages * pipe_stage_bits  # tS pipeline
        smem_alloc_bits += tP_stages * pipe_stage_bits  # tP pipeline
        # s1
        if const_expr(self.use_mma_p1):
            smem_alloc_bits += min_mma_n * min_mma_k_bits
        # sK, sV
        smem_capacity_bits = (
            cutlass.memory.get_smem_capacity_in_bytes(resource_arch) * 8
        )
        remaining_bits = smem_capacity_bits - smem_alloc_bits
        sK_stage_elts = blk_tile_kv * blk_tile_dq // mma_ctas
        sV_stage_elts = blk_tile_kv * blk_tile_do // mma_ctas
        sK_stage_bits = sK_stage_elts * mma_dtype.width
        sV_stage_bits = sV_stage_elts * mma_dtype.width
        sKV_stage_bits = max(sK_stage_bits, sV_stage_bits)
        kv_stages = remaining_bits // (sKV_stage_bits + pipe_stage_bits)
        remaining_bits -= kv_stages * (sKV_stage_bits + pipe_stage_bits)
        if const_expr(debug):
            print(f"kv stages {kv_stages} persistent {self.is_persistent}")
            print(f"blk_tile_m {blk_tile_m} mma_tile_m {mma_tile_m}")
            print(f"use_softmax_fma {self.use_softmax_fma} "
                  f"use_softmax_scale {self.use_softmax_scale} "
                  f"rescale threshold {self.rescale_threshold}")
            print(f"alias tP {self.alias_tP} use tQ {self.use_tQ} use mma p1 {self.use_mma_p1}")


        # --------------------------------------------------------------
        # TiledMma creation
        # --------------------------------------------------------------
        make_trivial_tiled_mma = (utils.sm107.make_trivial_tiled_mma
                                  if use_mma_k64 else
                                  utils.sm100.make_trivial_tiled_mma)

        # GEMM1: ((H_G, S_Q), S_K, D_Q, (H_K, B))
        cta_group = tcgen05.CtaGroup(mma_ctas)
        tiled_mma_qk = make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # Q
            OperandMajorMode.K,  # K
            acc_dtype,
            cta_group,
            (mma_tile_m, blk_tile_kv),
            a_source=(
                tcgen05.OperandSource.TMEM
                if self.use_tQ
                else tcgen05.OperandSource.SMEM
            ),
        )

        # GEMM2: ((H_G, S_Q), D_O, S_K, (H_K, B))
        tiled_mma_pv = make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,   # P
            OperandMajorMode.MN,  # V
            acc_dtype,
            cta_group,
            (mma_tile_m, blk_tile_do),
            a_source=tcgen05.OperandSource.TMEM,
        )

        # Calculate rowsum P by MMA with vector of 1s
        # For when we are more softmax limited than MMA limited
        mma_p1_tile_mnk = tiled_mma_p1 = None
        if const_expr(self.use_mma_p1):
            mma_p1_tile_mnk = (mma_tile_m, min_mma_n, min_mma_k)
            tiled_mma_p1 = make_trivial_tiled_mma(
                mma_dtype,
                mma_dtype,
                OperandMajorMode.K,  # P
                OperandMajorMode.K,  # 1s
                acc_dtype,
                cta_group,
                (mma_tile_m, min_mma_n),
                a_source=tcgen05.OperandSource.TMEM,
            )
            tiled_mma_pv = with_collector_op(
                tiled_mma_pv,
                tcgen05.CollectorOp.FILL,
                tcgen05.CollectorOp.DISCARD,
            )
            tiled_mma_p1 = with_collector_op(
                tiled_mma_p1,
                tcgen05.CollectorOp.LASTUSE,
                tcgen05.CollectorOp.DISCARD,
            )

        # --------------------------------------------------------------
        # TMA creation
        # --------------------------------------------------------------
        heads_k = h_k = k_bshd.shape[2]
        batches = b = q_bshd.shape[0]
        o_bshd_ = o_partial_bshd if self.do_kernel_red else o_bshd

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

        # ((h_g, s_q), d_qk, (h_k, b))
        mQ_mkl = GqaDecode.gemm_view(GqaDecode.gqa_pack(q_bshd, h_k), True)
        # (page_size, d, (h_k, pages)) or (s_k, d, (h_k, b))
        mK_nkl = GqaDecode.gemm_view(k_bshd, True)
        # (d, page_size, (h_k, pages)) or (d, s_k, (h_k, b))
        mV_nkl = GqaDecode.gemm_view(v_bshd, False)
        # ((h_g, s_q), d_vo, (h_k, b_partial))
        mO_mnl = GqaDecode.gemm_view(GqaDecode.gqa_pack(o_bshd_, h_k), True)

        # (MMA, #MMA_N, #MMA_K, kv_stages)
        sK_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_qk,
            mma_qk_tile_mnk,
            mma_dtype,
            kv_stages,
        )
        # (MMA, #MMA_N, #MMA_K, kv_stages)
        sV_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_pv,
            mma_pv_tile_mnk,
            mma_dtype,
            kv_stages,
        )

        # Re-stride to alias KV stages
        kv_stage_stride = sKV_stage_bits // mma_dtype.width
        sK_layout = cute.make_composed_layout(
            sK_layout.inner,
            sK_layout.offset,
            cute.make_layout(
                shape=sK_layout.outer.shape,
                stride=(*sK_layout.outer.stride[:-1], kv_stage_stride),
            ),
        )
        sV_layout = cute.make_composed_layout(
            sV_layout.inner,
            sV_layout.offset,
            cute.make_layout(
                shape=sV_layout.outer.shape,
                stride=(*sV_layout.outer.stride[:-1], kv_stage_stride),
            ),
        )

        if const_expr(page_table is not None):
            # Each CTA owns half of B's N mode for 2-CTA MMA. Subdivide those
            # local layouts into pages, preserving the aliased KV stage stride.
            sK_nk = cute.composition(
                sK_layout, cute.make_layout((blk_tile_kv // mma_ctas, blk_tile_dq, kv_stages))
            )
            sV_nk = cute.composition(
                sV_layout, cute.make_layout((blk_tile_do // mma_ctas, blk_tile_kv, kv_stages))
            )
            # ((PAGE, DQ), #PAGE_KV, 1, kv_stages)
            sK_pages = cute.tiled_divide(sK_nk, (self.page_size, blk_tile_dq))
            sK_pages = cute.select(sK_pages, [0, 1, 3])
            # ((DO/2, PAGE), 2, #PAGE_KV, kv_stages)
            sV_pages = cute.tiled_divide(sV_nk, (blk_tile_do // mma_ctas, self.page_size))
            sV_pages = cute.select(sV_pages, [0, 2, 3])
            tma_k_page = cpasync.make_tiled_tma_atom(
                tma_load_op, mK_nkl, sK_pages[0], (self.page_size, blk_tile_dq)
            )
            tma_v_page = cpasync.make_tiled_tma_atom(
                tma_load_op, mV_nkl, sV_pages[0], (blk_tile_do // mma_ctas, self.page_size)
            )
            # TMA views describe pages; pass the MMA layouts separately.
            tma_k = cpasync.TmaInfo(tma_k_page.atom, tma_k_page.tma_tensor, sK_pages)
            tma_v = cpasync.TmaInfo(tma_v_page.atom, tma_v_page.tma_tensor, sV_pages)

        else:
            tma_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK_nkl,
                sK_layout,
                (blk_tile_hg_qo, blk_tile_kv, blk_tile_dq),
                tiled_mma_qk,
            )
            tma_v = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV_nkl,
                sV_layout,
                (blk_tile_hg_qo, blk_tile_do, blk_tile_kv),
                tiled_mma_pv,
            )

        # Construct QO smem stage layout
        # (MMA, #MMA_M, #MMA_K, sQ_stages)
        sQ_layout = utils.sm100.make_smem_layout_a(
            tiled_mma_qk,
            (blk_tile_m, blk_tile_kv, blk_tile_dq),
            mma_dtype,
            sQ_stages
        )
        sO_layout = utils.sm100.make_smem_layout_c(
            tiled_mma_pv,
            (blk_tile_m, blk_tile_do, blk_tile_kv),
            sO_dtype,
            sQ_stages,
            is_m_major=False,
        )

        # Re-stride to alias QO stages
        # (MMA, #MMA_M, #MMA_K, sQ_stages)
        q_stage_stride = sQO_stage_bits // mma_dtype.width
        sQ_layout = cute.make_composed_layout(
            sQ_layout.inner,
            sQ_layout.offset,
            cute.make_layout(
                shape=sQ_layout.outer.shape,
                stride=(*sQ_layout.outer.stride[:-1], q_stage_stride),
            ),
        )
        # (MMA, #MMA_M, #MMA_N, sQ_stages)
        o_stage_stride = sQO_stage_bits // sO_dtype.width
        sO_layout = cute.make_composed_layout(
            sO_layout.inner,
            sO_layout.offset,
            cute.make_layout(
                shape=sO_layout.outer.shape,
                stride=(*sO_layout.outer.stride[:-1], o_stage_stride),
            ),
        )

        tma_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ_mkl,
            sQ_layout,
            (blk_tile_hg_qo, blk_tile_kv, blk_tile_dq),
            tiled_mma_qk,
        )
        tma_o = cute.nvgpu.make_tiled_tma_atom_C(
            tma_store_op,
            mO_mnl,
            sO_layout,
            (blk_tile_hg_qo, *mma_pv_tile_mnk[1:]),
            tiled_mma_pv,
        )

        tma_aux = None
        if const_expr(aux_bshd is not None):
            mAux = GqaDecode.gemm_view(GqaDecode.gqa_pack(aux_bshd, h_k), True)
            tma_aux = cute.nvgpu.make_tiled_tma_atom_C(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                mAux,
                sO_layout,
                (blk_tile_hg_qo, blk_tile_do, blk_tile_kv),
                tiled_mma_pv,
            )

        # GEMM view for LSE output
        # ((h_g, s_q), (h_k, b))
        mL_ml = None if l_bsh is None else GqaDecode.gemm_view_bsh(l_bsh, h_k)
        assert l_bsh is None or l_bsh.dtype == acc_dtype

        # GEMM views for workspace tensors
        mM_partial_ml = mL_partial_ml = None
        if const_expr(self.do_kernel_red):
            assert (
                m_partial_bsh.dtype
                == l_partial_bsh.dtype
                == o_partial_bshd.dtype
                == acc_dtype
            )

            # ((h_g, s_q), (h_k, b), kv_splits)
            mM_partial_ml = GqaDecode.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_ml = GqaDecode.gemm_view_bsh(l_partial_bsh, h_k)

        if const_expr(sink_h is not None):
            assert sink_h.dtype == acc_dtype

            h_g = sink_h.shape[0] // h_k
            mSink = cute.make_tensor(
                sink_h.iterator,
                cute.make_layout((h_g, h_k), stride=(1, h_g)),
            )
        else:
            mSink = None

        # --------------------------------------------------------------
        # Launch kernel(s)
        # --------------------------------------------------------------
        scale_s_log2_e = scale_s * log2_e

        kv_splits = 1 if self.do_none_red else self.kv_splits
        tiles_hg, tiles_qo = cute.ceil_div(mQ_mkl.shape[0], blk_tile_hg_qo)

        # Rasterize blocks in order of KV temporal locality
        grid_x = (mma_ctas, tiles_hg, tiles_qo)  # load same KV
        grid_y = kv_splits  # strided loads of same KV
        grid_z = (heads_k, batches)  # load different KV
        grid = cute.product_each((grid_x, grid_y, grid_z))

        cluster_x = mma_ctas
        cluster_y = kv_splits if self.do_atomic_red else 1
        cluster = (cluster_x, cluster_y, 1)

        self.attention(
            # MMA
            mma_qk_tile_mnk,
            mma_pv_tile_mnk,
            mma_p1_tile_mnk,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_p1,
            mma_dtype,
            sO_dtype,
            tma_q,
            tma_k,
            tma_v,
            tma_o,
            tma_aux,
            sK_layout,
            sV_layout,
            seqlens,
            table_offsets,
            page_table,
            # Rest
            mL_ml,
            mL_partial_ml,
            mM_partial_ml,
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

        if const_expr(self.do_kernel_red):
            GqaDecode.launch_reduction(
                self.headdim_vo,
                kv_splits,
                64,  # red_vec_bits
                1,  # red_d_ctas
                16,  # red_max_groups
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
    def attention(
        self,
        # MMA
        mma_qk_tile_mnk: cute.Tile,
        mma_pv_tile_mnk: cute.Tile,
        mma_p1_tile_mnk: Optional[cute.Tile],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_p1: Optional[cute.TiledMma],
        mma_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        tma_q: cpasync.TmaInfo,
        tma_k: cpasync.TmaInfo,
        tma_v: cpasync.TmaInfo,
        tma_o: cpasync.TmaInfo,
        tma_aux: Optional[cpasync.TmaInfo],
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        seqlens: cute.Tensor | Int32 | None,
        table_offsets: Optional[cute.Tensor],
        page_table: Optional[cute.Tensor],
        # Rest
        mL: Optional[cute.Tensor],  # LSE output (Float32, log2 base)
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mSink: Optional[cute.Tensor],  # (h_g, h_k)
        mask_config: AttentionMask,
        scale_s_log2_e: Float32,
        scale_o: Float32,
    ):
        # --------------------------------------------------------------
        # region  Prologue
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Codegen config
        # --------------------------------------------------------------
        # Alias types
        q_dtype = k_dtype = v_dtype = mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32

        # Shapes for MMA tile indexing
        # Block tile sets the granularity at which CTAs (or CTA-pairs) consume work (BMM1/BMM2)
        blk_tile_qo = self.sequence_qo_tile
        blk_tile_hg = self.grouped_head_tile
        blk_tile_hg_qo = (blk_tile_hg, blk_tile_qo)  # multimode tiler
        blk_tile_m = blk_tile_hg * blk_tile_qo  # linearized tiler
        # MMA tile sets the granularity at which TMAs + MMAs are staged in smem
        mma_tile_m, blk_tile_kv, blk_tile_dq = mma_qk_tile_mnk
        _mma_tile_m, blk_tile_do, _blk_tile_kv = mma_pv_tile_mnk
        assert mma_tile_m == _mma_tile_m
        assert blk_tile_kv == _blk_tile_kv
        # MMA tiles per block tile (spatial pipeline is mmas_m == 2)
        mmas_m = blk_tile_m // mma_tile_m
        assert mmas_m == 1, "spatial pipeline not supported, use fmha.py"
        # CTAs per MMA
        mma_ctas = 2 if mma_tile_m == 256 else 1
        cta_group = tcgen05.CtaGroup(mma_ctas)
        blk_m_per_cta = blk_tile_m // mma_ctas

        # Static control flow
        prefetch_iters_kv = 2
        rescale_threshold = self.rescale_threshold
        do_none_red = self.do_none_red
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        use_2cta_mma = mma_ctas == 2
        use_clusters = use_2cta_mma or do_atomic_red
        use_mma_p1 = self.use_mma_p1
        store_lse = mL is not None
        use_tQ = self.use_tQ
        use_sink = mSink is not None
        is_persistent = self.is_persistent
        is_paged = page_table is not None
        do_aux_load = self.aux_fn is not None

        # Persistence is for multi-wave, split-K is for sub-wave
        assert not (is_persistent and not do_none_red), "persistence doesn't support split-KV"

        # Warp assignments
        warpgroup_id = 0
        mma_warp_id = warpgroup_id * warpgroup_warps + 0
        tma_qo_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_k_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_v_warp_id = warpgroup_id * warpgroup_warps + 3
        tma_mma_warpgroup_id = warpgroup_id
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
        wg_reg_pool = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        mma_tma_regs = 32
        wg_reg_pool -= mma_tma_regs
        softmax_regs = 176 if self.softmax_dtype == Float16 else 184
        wg_reg_pool -= softmax_regs * softmax_warpgroups
        correction_regs = wg_reg_pool
        wg_reg_pool -= correction_regs
        assert wg_reg_pool >= 0, f"exceeded register budget by {-wg_reg_pool}"
        # --------------------------------------------------------------
        # endregion Codegen config
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Runtime config
        # --------------------------------------------------------------
        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = cutlass.memory.SmemAllocator()

        # Read thread indices
        grid_dim = cute.arch.grid_dim()
        block_idx = cute.arch.block_idx()
        if const_expr(do_none_red):
            kv_splits = 1
            kv_split_idx = 0
        else:
            kv_splits = grid_dim[1]
            kv_split_idx = block_idx[1]
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warpgroup_tidx // warp_threads
        init_warp = 1  # warp 0 does all pipeline inits for now

        # Cluster indices + layouts
        cluster_rank = cute.arch.block_idx_in_cluster() if use_clusters else 0
        mma_rank = cluster_rank % mma_ctas
        thrblk_mma_qk = tiled_mma_qk.get_slice(mma_rank)
        thrblk_mma_pv = tiled_mma_pv.get_slice(mma_rank)
        cta_layout_vmnk = cute.make_layout((mma_ctas, 1, 1, 1))

        # Unpack multimodes
        grouped_heads, seqlen_qo = tma_q.tma_tensor.shape[0]
        heads_k, batches = tma_q.tma_tensor.shape[2]
        tiles_hg, tiles_qo = cute.ceil_div(
            (grouped_heads, seqlen_qo), blk_tile_hg_qo
        )
        def get_coords(
            block_idx,
            mma_ctas = mma_ctas,
            tiles_hg = tiles_hg,
            tiles_qo = tiles_qo,
            heads_k = heads_k,
            batches = batches,
            mask_config = mask_config,
        ):
            bidx, kv_split_idx, idx_hk_b = block_idx
            idx_hg_qo = bidx // mma_ctas
            coord_hg, coord_qo = cute.idx2crd(
                idx_hg_qo, (tiles_hg, tiles_qo)
            )
            coord_hk, coord_b = cute.idx2crd(
                idx_hk_b, (heads_k, batches)
            )
            if const_expr(isinstance(mask_config, CausalMask)):
                coord_qo = tiles_qo - coord_qo - 1  # reverse


            # Work around a frontendnext bug: tuple assignment decomposes static
            # coordinate wrappers and incorrectly checks their internal _slot_key
            # tuple lengths. Explicit Int32 conversion keeps coordinates staged.
            # Keep coordinate leaves staged across persistent-loop updates.

            coord_hg = Int32(coord_hg)
            coord_qo = Int32(coord_qo)
            coord_hk = Int32(coord_hk)
            coord_b = Int32(coord_b)
            return (coord_hg, coord_qo), (coord_hk, coord_b)
        coord_hg_qo, coord_hk_b = get_coords(block_idx)
        coord_hg, coord_qo = coord_hg_qo
        coord_hk, coord_b = coord_hk_b

        # load var seqlen
        def get_seqlen(coord_b, seqlens=seqlens, tma_k=tma_k):
            if const_expr(isinstance(seqlens, cute.Tensor)):
                return seqlens[coord_b]
            if const_expr(seqlens is None):
                return tma_k.tma_tensor.shape[0]
            return seqlens
        seqlen_kv = get_seqlen(coord_b)

        # Runtime control flow
        coord_kv, tiles_kv = mask_config.get_tiles_kv(
            seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
        )
        iters_kv = cute.ceil_div(tiles_kv - kv_split_idx, kv_splits)
        iters_kv = max(iters_kv, prefetch_iters_kv)  # simplify synchronization
        cta_exit_early = kv_split_idx >= tiles_kv
        mma_exit_early = mma_rank == 1 and warp_idx == mma_warp_id
        exit_early = cta_exit_early or mma_exit_early

        # Prefetch TMA descriptor
        if warp_idx == init_warp and not cta_exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q.atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k.atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v.atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_o.atom)
            if const_expr(do_aux_load):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_aux.atom)
        init_warp += 1

        # Tmem Allocation
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp and not cta_exit_early:
            cute.arch.alloc_tmem(
                tmem_alloc_cols,
                tmem_ptr_smem_ptr,
                use_2cta_mma,
                arch=resource_arch,
            )
            cute.arch.relinquish_tmem_alloc_permit(use_2cta_mma)
        init_warp += 1

        # --------------------------------------------------------------
        # endregion  Runtime config
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Pipeline Init
        # --------------------------------------------------------------
        # Alias thread counts/cooperatives
        cg = lambda t: CooperativeGroup(Agent.Thread, t)
        mma_threads = tma_threads = warp_threads
        mma_elect_one = tma_elect_one = 1
        softmax_threads = correction_threads = warpgroup_threads
        total_softmax_threads = softmax_threads * softmax_warpgroups

        # Named barriers
        softmax_sync = nbar(1, total_softmax_threads).sync
        tma_order_k_nbar = nbar(2, tma_threads + tma_threads)
        tma_order_v_nbar = nbar(3, tma_threads + tma_threads)
        sO_final_nbar = nbar(4, total_softmax_threads + tma_threads)
        sM_final_nbar = nbar(5, total_softmax_threads + correction_threads)
        mma_tma_nbar = nbar(6, mma_threads + tma_threads)
        last_v_nbar = nbar(7, tma_threads * 2) if do_aux_load else None
        # 8 nbars, 1 per warp per phase (coalesce to 2 if more nbars needed)
        sM_producer_nbars = nbar(8, warp_threads * 2)
        # named barrier stage helper
        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        # Tmem deallocation mbarrier
        if const_expr(use_2cta_mma):
            tmem_dealloc_mbar = smem.allocate_array(Int64)
            if warp_idx == init_warp:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar, 1)
                    cute.arch.mbarrier_init_fence()
            init_warp += 1

        # softmax order mbarriers
        softmax_mbars = smem.allocate_array(Int64, 1)
        tSP_alias_mbar = softmax_mbars + 0

        if warp_idx == init_warp:
            if lane_idx < 1:
                cute.arch.mbarrier_init(softmax_mbars + lane_idx, softmax_threads)
                cute.arch.mbarrier_init_fence()
        init_warp += 1

        # Aux TMA completion (one local load per work tile).
        aux_mbar = None
        if const_expr(do_aux_load):
            aux_mbar = smem.allocate_array(Int64, 1)
            if warp_idx == init_warp:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(aux_mbar, 1)
                    cute.arch.mbarrier_init_fence()
            init_warp += 1

        # sQ pipeline
        sQ_stages = tma_q.smem_layout.shape[-1]
        sQ_stage_elts = blk_tile_m * blk_tile_dq  # across maybe 2CTA pair
        sQ_stage_bytes = sQ_stage_elts * q_dtype.width // 8
        sQ_mbarriers = smem.allocate_array(Int64, sQ_stages * 2)
        sQ_producer, sQ_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=sQ_stages,
            producer_group=cg(tma_elect_one),
            consumer_group=cg(mma_elect_one),  # TMA VO actually does the release
            tx_count=sQ_stage_bytes,
            barrier_storage=sQ_mbarriers,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # sKV pipeline
        sKV_stages = sK_layout.shape[-1]
        sK_stage_elts = blk_tile_kv * blk_tile_dq  # across maybe 2CTA pair
        sV_stage_elts = blk_tile_kv * blk_tile_do
        sK_stage_bytes = sK_stage_elts * k_dtype.width // 8
        sV_stage_bytes = sV_stage_elts * v_dtype.width // 8
        sKV_stage_bytes = max(sK_stage_bytes, sV_stage_bytes)
        sKV_mbarriers = smem.allocate_array(Int64, sKV_stages * 2)
        sKV_producer, sKV_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=sKV_stages,
            producer_group=cg(tma_elect_one),
            consumer_group=cg(mma_elect_one),
            tx_count=sKV_stage_bytes,
            barrier_storage=sKV_mbarriers,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        tS_stages = self.tS_stages
        tO_stages = self.tO_stages
        tS_mbarriers = smem.allocate_array(Int64, tS_stages * 2)
        tS_producer, tS_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=tS_stages,
            producer_group=cg(mma_elect_one),
            consumer_group=cg(softmax_threads * mma_ctas),
            barrier_storage=tS_mbarriers,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        assert tS_stages == tO_stages == softmax_warpgroups == 2

        tP_stages = self.tP_stages
        tP_mbarriers = smem.allocate_array(Int64, tP_stages * 2)
        tP_producer_arv_cnt = (softmax_threads + correction_threads) * mma_ctas
        # MMA P1 can use just 1 tmem buffer to support tmem limited configs (e.g. fp8 mla prefill)
        # So correction must arrive 1 stage ahead to signal completion of row sum load
        if const_expr(use_mma_p1 and self.tP1_stages == 1):
            tP_producer_arv_cnt += correction_threads * mma_ctas
        tP_producer, tP_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=tP_stages,
            producer_group=cg(tP_producer_arv_cnt),
            consumer_group=cg(mma_elect_one),
            barrier_storage=tP_mbarriers,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # tP pipeline is also used to track completion of tO
        assert tP_stages == tO_stages

        clc_response_ptr = None
        if const_expr(is_persistent):
            if const_expr(use_2cta_mma):
                # rank 1 mma warp exits early
                clc_threads = 2 * self.threads_per_cta - warp_threads
            else:
                clc_threads = self.threads_per_cta
            clc_mbar_ptr = smem.allocate_array(Int64, 2)
            clc_response_ptr = smem.allocate_array(Int128)
            clc_producer, clc_consumer = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbar_ptr,
                num_stages=1,
                producer_group=cg(tma_elect_one),
                consumer_group=cg(clc_threads),
                tx_count=(Int128.width // 8),
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            ).make_participants()

        def query_next_work(clc_producer : pipeline.PipelineProducer,
                            clc_response_ptr=clc_response_ptr):
            clc_producer.acquire()
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(
                    clc_producer.current_handle().barrier,
                    clc_response_ptr,
                )

        def get_next_work(clc_consumer : pipeline.PipelineConsumer,
                          clc_response_ptr=clc_response_ptr, cluster_rank=cluster_rank):
            clc_consumer.wait()
            bidx, bidy, bidz, valid = cute.arch.clc_response(clc_response_ptr)
            cute.arch.fence_view_async_shared()
            clc_consumer.release()
            bidx += cluster_rank
            return (bidx, bidy, bidz), Boolean(valid)


        if const_expr(use_clusters):
            cute.arch.cluster_arrive_relaxed()
        # --------------------------------------------------------------
        # endregion  Pipeline init
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Smem tensor alloc
        # --------------------------------------------------------------
        # TMA partition helper
        # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
        def tma_partition(atom, smem_tensor, gmem_tensor, group=(0,3)):
            mcast_coord = 0
            mcast_layout = cute.make_layout((1,))  # no TMA multicast
            return cute.nvgpu.cpasync.tma_partition(
                atom, mcast_coord, mcast_layout,
                cute.group_modes(smem_tensor, *group),
                cute.group_modes(gmem_tensor, *group),
            )

        # sK and sV alias the same allocation. The common stage stride does
        # not pad the last stage's cosize: reserve the larger full footprint
        # so V cannot overwrite Q/O when its head dimension exceeds K's.
        k_bytes = cute.cosize(sK_layout.outer) * k_dtype.width // 8
        v_bytes = cute.cosize(sV_layout.outer) * v_dtype.width // 8
        sKV_ptr = smem.allocate(max(k_bytes, v_bytes), stensor_align)
        sK_ptr = cute.recast_ptr(sKV_ptr, sK_layout.inner, dtype=k_dtype)
        # (MMA, #MMA_N, #MMA_K, kv_stages)
        tBsK = cute.make_tensor(sK_ptr, sK_layout.outer)
        sV_ptr = cute.recast_ptr(sKV_ptr, sV_layout.inner, dtype=v_dtype)
        # (MMA, #MMA_N, #MMA_K, kv_stages)
        tBsV = cute.make_tensor(sV_ptr, sV_layout.outer)
        # Place Q, O on the second sub-bank (128-227KiB)
        # sQ aliased to sO
        q_bytes = cute.cosize(tma_q.smem_layout.outer) * q_dtype.width // 8
        o_bytes = cute.cosize(tma_o.smem_layout.outer) * o_dtype.width // 8
        sQO_ptr = smem.allocate(max(q_bytes, o_bytes), stensor_align)
        # (MMA, #MMA_M, #MMA_K, sQ_stages)
        tAsQ = cute.make_tensor(
            cute.recast_ptr(sQO_ptr, tma_q.smem_layout.inner, dtype=q_dtype),
            tma_q.smem_layout.outer,
        )
        # (MMA, #MMA_M, #MMA_N, sQ_stages)
        tCsO = cute.make_tensor(
            cute.recast_ptr(sQO_ptr, tma_o.smem_layout.inner, dtype=o_dtype),
            tma_o.smem_layout.outer,
        )

        if const_expr(use_mma_p1):
            # (MMA_NK, #MMA_N=1, #MMA_K=1, 1)
            tBs1_layout = utils.sm100.make_smem_layout_b(
                tiled_mma_p1, mma_p1_tile_mnk, mma_dtype, 1
            )
            tBs1 = smem.allocate_tensor(
                mma_dtype,
                tBs1_layout.outer,
                stensor_align,
                tBs1_layout.inner
            )
            # 1-init 0th column
            if warp_idx == init_warp:
                col_zero = tBs1[(0, None), 0, 0, 0]
                vec_one = cute.make_rmem_tensor((32 // mma_dtype.width,), mma_dtype)
                vec_one.fill(mma_dtype(1))
                col_zero = cute.zipped_divide(col_zero, vec_one.shape)
                num_vec = cute.size(col_zero.shape[1])
                for i in range_constexpr(0, num_vec, warp_threads):
                    if i + lane_idx < num_vec:
                        col_zero[None, i + lane_idx].store(vec_one.load())
            init_warp += 1

        # sPT - page table lookup buffer
        pages_kv = page_copy_atom = None
        if const_expr(is_paged):
            # K loads all page indices; V consumes the same ring two tiles later.
            pages_kv = blk_tile_kv // self.page_size
            pt_stages = self.pt_stages
            assert pt_stages == prefetch_iters_kv + 2
            sPT = smem.allocate_tensor(Int32, cute.make_layout((pages_kv, pt_stages)), svector_align)
            page_copy_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), Int32, num_bits_per_copy=32)

        def get_page_info(coord_b, seqlen,
                          table_offsets=table_offsets,
                          page_size=self.page_size):
            page_count = cute.ceil_div(seqlen, page_size)
            if const_expr(table_offsets is not None):
                table_offset = table_offsets[coord_b]
            else:
                table_offset = coord_b * page_count
            return page_count, table_offset

        def prefetch_pages(tile, page_count, table_offset, dst,
                           page_table=page_table, pages_kv=pages_kv,
                           lane_idx=lane_idx, page_copy_atom=page_copy_atom):
            if lane_idx < pages_kv:
                logical_page = tile * pages_kv + lane_idx
                dst_lane = cute.local_tile(dst, (1,), (lane_idx,))
                if logical_page < page_count:
                    src_lane = cute.local_tile(page_table, (1,), (table_offset + logical_page,))
                    cute.copy(page_copy_atom, src_lane, dst_lane)
                else:
                    dst_lane[0] = -1  # TMA out-of-bounds zero fill
            cute.arch.cp_async_commit_group()

        # sM - vector stats (max, sum, correction)
        sM_layout = cute.make_layout((mma_tile_m, 1), stride=(1, 0))
        # we just want a partitoned layout that preserves 0 strides
        # (partition_shape_C doesn't accept stride info)
        fake_sM = cute.make_tensor(0, sM_layout)
        # (MMA, #MMA_M=1, #MMA_N=1)
        fake_tCsM = tiled_mma_qk.get_slice(0).partition_C(fake_sM)
        tCsM_layout = fake_tCsM[None, 0, 0].layout
        # (MMA, tO_stages)
        # FIXME: *_product should accept shape tilers
        tCsM_layout = cute.flat_product(tCsM_layout, cute.make_layout(tO_stages))
        tCsM = smem.allocate_tensor(acc_dtype, tCsM_layout, svector_align)

        # --------------------------------------------------------------
        # endregion  smem tensors
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Prologue sync
        # --------------------------------------------------------------
        # Ensure visibility of cluster mbarriers
        if const_expr(use_clusters):
            cute.arch.cluster_wait()
        # Ensure visibility of local mbarrier/smem inits and tmem alloc
        cute.arch.sync_threads()
        assert init_warp < (self.threads_per_cta // warp_threads), (
            f"used {init_warp + 1} init warps, {self.threads_per_cta // warp_threads} warps available"
        )
        # --------------------------------------------------------------
        # endregion  Prologue sync
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Tmem tensor alloc
        # --------------------------------------------------------------
        # Tmem buffer pool
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem = cutlass.memory.TmemBufferPool(tmem_ptr, tmem_alloc_cols)

        # Tmem copy atoms
        if const_expr(mma_tile_m == 64):
            tmem_ld_op = tcgen05.Ld16x32bx2Op
            tmem_st_op = tcgen05.St16x32bx2Op
            # Unlike Ld16x32bx2Op, LdRed16x32bx2Op requires specified split offset
            tmem_red_op = lambda r : tcgen05.LdRed16x32bx2Op(r, half_split_off=r.value)
            tmem_op_width = 64 # 32bx2
            tmem_row_lanes = 2 # x2 splits row across 2 half-warps
        else:
            tmem_ld_op = tcgen05.Ld32x32bOp
            tmem_st_op = tcgen05.St32x32bOp
            tmem_red_op = tcgen05.LdRed32x32bOp # defaults to max
            tmem_op_width = 32
            tmem_row_lanes = 1
        def make_tmem_atom(op, cols, dtype, op_width=tmem_op_width):
            repeat = cols * dtype.width // op_width
            repeat = min(repeat, 64)  # reg pressure too high for x128
            repeat = tcgen05.Repetition(repeat)
            return cute.make_copy_atom(op(repeat), dtype)
        make_tmem_ld_atom = partial(make_tmem_atom, tmem_ld_op)
        make_tmem_st_atom = partial(make_tmem_atom, tmem_st_op)
        make_tmem_red_atom = partial(make_tmem_atom, tmem_red_op)
        # Tmem tiled copy helper
        def tile_tmem_atom(atom, tensor, tidx=warpgroup_tidx):
            return tcgen05.make_tmem_copy(atom, tensor).get_slice(tidx)
        # tO
        # (MMA_MN, #MMA_M=1, #MMA_N=1, tO_stages)
        tCtO_shape = tiled_mma_pv.partition_shape_C(
            (mma_tile_m, blk_tile_do, tO_stages)
        )
        tCtO = tiled_mma_pv.make_fragment_C(tCtO_shape)
        # (MMA_MN, tO_stages)
        tCtO = tCtO[None, 0, 0, None]
        tCtO = tmem.allocate_tensor(tCtO.layout, acc_dtype)

        # tS
        # (MMA_MN, #MMA_M=1, #MMA_N=1, tS_stages)
        tCtS_shape = tiled_mma_qk.partition_shape_C(
            (mma_tile_m, blk_tile_kv, tS_stages)
        )
        tCtS = tiled_mma_qk.make_fragment_C(tCtS_shape)
        # (MMA_MN, tS_stages)
        tCtS = tCtS[None, 0, 0, None]
        tCtS = tmem.allocate_tensor(tCtS.layout, acc_dtype)

        # tP
        # (MMA_MK, #MMA_M=1, #MMA_K, tP_stages)
        tAtP_shape = tiled_mma_pv.partition_shape_A(
            (mma_tile_m, blk_tile_kv, tP_stages)
        )
        tAtP = tiled_mma_pv.make_fragment_A(tAtP_shape)

        # alias tP to tS
        if const_expr(self.alias_tP):
            tCtS_recast = cute.recast_tensor(tCtS, mma_dtype)
            tAtP = cute.make_tensor(
                tCtS_recast.iterator,
                cute.make_layout(
                    tAtP.shape,
                    stride=(*tAtP.stride[:-1], tCtS_recast.stride[-1])
                )
            )
        else:
            tAtP = tmem.allocate_tensor(tAtP.layout, mma_dtype)

        # tQ
        # (MMA_MK, #MMA_M, #MMA_K)
        if const_expr(use_tQ):
            tAtQ_shape = tiled_mma_qk.partition_shape_A(
                (blk_tile_m, blk_tile_dq)
            )
            tAtQ = tiled_mma_qk.make_fragment_A(tAtQ_shape)
            tAtQ = tmem.allocate_tensor(tAtQ.layout, mma_dtype)

        # tP1 - rowsum P computed with MMA
        if const_expr(use_mma_p1):
            # (MMA_MN, #MMA_M=1, #MMA_N=1, tP1_stages)
            tP1_stages = self.tP1_stages
            tCtP1_shape = tiled_mma_p1.partition_shape_C(
                (*mma_p1_tile_mnk[:2], tP1_stages)
            )
            tCtP1 = tiled_mma_p1.make_fragment_C(tCtP1_shape)
            tCtP1 = tmem.allocate_tensor(tCtP1.layout, acc_dtype)
            # (MMA_MN, tP1_stages)
            tCtP1 = tCtP1[None, 0, 0, None]
            # unconditionally index into 2 stages even if we have 1
            assert tCtP1.stride[-1] == 0 if tP1_stages == 1 else True
        # --------------------------------------------------------------
        # endregion  tmem tensors
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # endregion  Prologue
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  Exit early
        # --------------------------------------------------------------
        if exit_early:
            if const_expr(do_kernel_red):
                gM_partial = cute.local_tile(
                    mM_partial,
                    (blk_tile_hg_qo,),
                    (coord_hg_qo, coord_hk_b, kv_split_idx),
                )[(None, None),]
                gL_partial = cute.local_tile(
                    mL_partial,
                    (blk_tile_hg_qo,),
                    (coord_hg_qo, coord_hk_b, kv_split_idx),
                )[(None, None),]
                if cta_exit_early and tidx < blk_tile_m:
                    blk_idx_hg, blk_idx_qo = cute.idx2crd(tidx, blk_tile_hg_qo)
                    idx_hg = coord_hg * blk_tile_hg + blk_idx_hg
                    idx_qo = coord_qo * blk_tile_qo + blk_idx_qo
                    if idx_hg < grouped_heads and idx_qo < seqlen_qo:
                        gM_partial[blk_idx_hg, blk_idx_qo] = -Float32.inf
                        gL_partial[blk_idx_hg, blk_idx_qo] = Float32(0)

            # MMA rank 1 must dealloc before exiting
            if mma_exit_early and not cta_exit_early:
                cute.arch.setmaxregister_decrease(mma_tma_regs)

        # --------------------------------------------------------------
        # endregion  Exit early
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  TMA QO Dispatch
        # --------------------------------------------------------------
        elif warp_idx == tma_qo_warp_id:
            cute.arch.setmaxregister_decrease(mma_tma_regs)

            # region  partition gQ, gO
            # ((TILE_HG, TILE_QO), TILE_DQ, #TILE_HG_QO, (h_k, b))
            gQ = cute.local_tile(
                tma_q.tma_tensor,
                tiler=(blk_tile_hg_qo, blk_tile_dq),
                coord=(None, 0, None),
            )
            # (MMA, #MMA_M, #MMA_K, #TILE_HG_QO, (h_k, b))
            tAgQ = thrblk_mma_qk.partition_A(gQ)
            # (TMA, #TILE_HG_QO, (h_k, b))
            tAsQ_tma, tAgQ_tma = tma_partition(tma_q.atom, tAsQ, tAgQ)

            # ((TILE_HG, TILE_QO), TILE_DO, #TILE_HG_QO, (h_k, b_partial))
            gO = cute.local_tile(
                tma_o.tma_tensor,
                tiler=(blk_tile_hg_qo, blk_tile_do),
                coord=(None, 0, None),
            )
            # (MMA, #MMA_M, #MMA_N, #TILE_HG_QO, (h_k, b_partial))
            tCgO = thrblk_mma_pv.partition_C(gO)
            # (TMA, #TILE_HG_QO, (h_k, b_partial))
            tCsO_tma, tCgO_tma = tma_partition(tma_o.atom, tCsO, tCgO)
            tCsAux_tma = None
            if const_expr(do_aux_load):
                gAux = cute.local_tile(
                    tma_aux.tma_tensor,
                    tiler=(blk_tile_hg_qo, blk_tile_do),
                    coord=(None, 0, None),
                )
                tCgAux = thrblk_mma_pv.partition_C(gAux)
                tCsAux_tma, tCgAux_tma = tma_partition(tma_aux.atom, tCsO, tCgAux)

            def load_aux(
                aux_tile,
                stage,
                *,
                tma_aux=tma_aux,
                tCsAux_tma=tCsAux_tma,
                last_v_nbar=last_v_nbar,
                aux_mbar=aux_mbar,
                aux_bytes=blk_tile_m * blk_tile_do * out_dtype.width // (8 * mma_ctas),
            ):
                # Q's pipeline release protects the alias; this barrier
                # additionally orders aux behind the final V TMA issue.
                last_v_nbar.arrive_and_wait()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(aux_mbar, aux_bytes)
                cute.copy(
                    tma_aux.atom, aux_tile, tCsAux_tma[None, stage],
                    tma_bar_ptr=aux_mbar,
                )
            # O partial mode is coalesced with batches
            if const_expr(do_kernel_red):
                coord_partial = (None, None, (None, kv_split_idx * batches))
                tCgO_tma = cute.domain_offset(coord_partial, tCgO_tma)

            # Partition for sQ to tQ copy
            if const_expr(use_tQ):
                tQ_cp_op = tcgen05.Cp128x256bOp(cta_group)
                tQ_cp_atom = cute.make_copy_atom(tQ_cp_op, mma_dtype)
                tQ_cp_tiled = tcgen05.make_s2t_copy(tQ_cp_atom, tAtQ).get_slice(0)
                # (CPY, #CPY_MMA, #CPY_M, #CPY_K, sQ_stages)
                frg_sQ = tQ_cp_tiled.partition_S(tAsQ)
                frg_tQ = tQ_cp_tiled.partition_D(tAtQ)
                frg_sQ_desc = tcgen05.get_s2t_smem_desc_tensor(tQ_cp_atom, frg_sQ)
            # endregion partition gQ, gO

            # Work tile slice
            tAgQ_hb = tAgQ_tma[None, coord_hg_qo, coord_hk_b]
            tCgO_hb = tCgO_tma[None, coord_hg_qo, coord_hk_b]
            if const_expr(do_aux_load):
                tCgAux_hb = tCgAux_tma[None, coord_hg_qo, coord_hk_b]

            # Load first/current gQ
            sO_index = 0
            sQ_handle = sQ_producer.acquire_and_advance()
            tAsQ_stage = tAsQ_tma[None, sQ_handle.index]
            cute.arch.griddepcontrol_wait()
            cute.copy(
                tma_q.atom,
                tAgQ_hb,
                tAsQ_stage,
                tma_bar_ptr=sQ_handle.barrier,
            )
            if const_expr(use_tQ):
                # Copy sQ to tQ
                if mma_rank == 0:
                    sQ_consumer.wait_and_advance()
                    frg_sQ_stage = frg_sQ_desc[((None,) * 4) + (sQ_handle.index,)]
                    cute.copy(tQ_cp_atom, frg_sQ_stage, frg_tQ)
                    tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
                    mma_tma_nbar.arrive()

            # region  Persistent loop
            if const_expr(is_persistent):
                # Read CLC response from smem
                block_idx, next_work_valid = get_next_work(clc_consumer)
                clc_consumer.advance()

                # Acquire sQ of s+1
                sQ_handle = sQ_producer.acquire_and_advance()
                while next_work_valid:
                    # Update tile coordinates
                    coord_hg_qo, coord_hk_b = get_coords(block_idx)

                    # Load gQ of s+1
                    tAgQ_hb = tAgQ_tma[None, coord_hg_qo, coord_hk_b]
                    tAsQ_stage = tAsQ_tma[None, sQ_handle.index]
                    # Wait for O store to complete
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.copy(
                        tma_q.atom,
                        tAgQ_hb,
                        tAsQ_stage,
                        tma_bar_ptr=sQ_handle.barrier,
                    )
                    # Acquire sQ of s+2 to ensure tQ of s was consumed
                    next_sQ_handle = sQ_producer.acquire_and_advance()
                    if const_expr(use_tQ):
                        # Copy sQ to tQ
                        if mma_rank == 0:
                            sQ_consumer.wait_and_advance()
                            frg_sQ_s = frg_sQ_desc[((None,) * 4) + (sQ_handle.index,)]
                            cute.copy(tQ_cp_atom, frg_sQ_s, frg_tQ)
                            tcgen05_fence(Tcgen05Fence.BEFORE_THREAD_SYNC)
                            mma_tma_nbar.arrive()
                    sQ_handle = next_sQ_handle

                    # Store gO of s
                    if const_expr(do_aux_load):
                        # next_sQ_handle acquired this stage after the last QK.
                        load_aux(tCgAux_hb, sO_index)
                        tCgAux_hb = tCgAux_tma[None, coord_hg_qo, coord_hk_b]
                    tCsO_stage = tCsO_tma[None, sO_index]
                    sO_final_nbar.arrive_and_wait()
                    cute.copy(tma_o.atom, tCsO_stage, tCgO_hb)
                    cute.arch.cp_async_bulk_commit_group()
                    tCgO_hb = tCgO_tma[None, coord_hg_qo, coord_hk_b]
                    sO_index ^= 1

                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()
            # endregion  Persistent loop

            # Store last gO
            if const_expr(do_aux_load):
                # Drain the current Q stage even when there is no next tile.
                sQ_producer.acquire()
                load_aux(tCgAux_hb, sO_index)
            tCsO_stage = tCsO_tma[None, sO_index]
            sO_final_nbar.arrive_and_wait()
            cute.copy(tma_o.atom, tCsO_stage, tCgO_hb)
            cute.arch.griddepcontrol_launch_dependents()

            # Wait for signal to dealloc tmem, then dealloc
            if const_expr(use_2cta_mma):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar, cluster_rank ^ 1)
                cute.arch.mbarrier_wait(tmem_dealloc_mbar, phase=0)
            cute.arch.dealloc_tmem(
                tmem_ptr,
                tmem_alloc_cols,
                use_2cta_mma,
                arch=resource_arch,
            )
        # --------------------------------------------------------------
        # endregion TMA QO Dispatch
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  TMA K Dispatch
        # --------------------------------------------------------------
        elif warp_idx == tma_k_warp_id:
            cute.arch.setmaxregister_decrease(mma_tma_regs)

            if const_expr(is_paged):
                # region  Partition paged gK
                # (PAGE, DQ, heads_k, physical_pages)
                gK = cute.local_tile(
                    tma_k.tma_tensor,
                    tiler=(self.page_size, blk_tile_dq),
                    coord=(0, 0, None),
                )
                # ((PAGE, DQ), #PAGE, physical_pages)
                sK_tma_view = cute.make_tensor(tBsK.iterator, tma_k.smem_layout.outer)
                tBsK_tma, tBgK_tma = cpasync.tma_partition(
                    tma_k.atom, 0, cute.make_layout(1), sK_tma_view,
                    cute.group_modes(gK, 0, 2),
                )
                # endregion partition paged gK

                page_count, table_offset = get_page_info(coord_b, seqlen_kv)
                cute.arch.griddepcontrol_wait()

                # Persistent Loop
                next_work_valid = True
                while next_work_valid:
                    # Work tile slice
                    tBgK_hb = tBgK_tma[None, (coord_hk, None)]
                    pt_index = Int32(0)
                    # Also protects ring reuse after the previous persistent tile's
                    # final V load. K alone produces the full block's page indices.
                    tma_order_v_nbar.arrive_and_wait()
                    prefetch_pages(coord_kv + kv_split_idx, page_count, table_offset, sPT[None, 0])

                    # region Load paged gK loop
                    prefetch_tiles_kv = prefetch_iters_kv * kv_splits
                    # Match MMA/softmax, including the minimum prefetch iterations.
                    kv_stop = kv_split_idx + iters_kv * kv_splits
                    for coord_kv_ in cutlass.range(kv_split_idx, kv_stop, kv_splits):
                        if coord_kv_ != kv_split_idx:
                            tma_order_v_nbar.arrive_and_wait()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_warp()
                        sPT_cta = cute.local_tile(
                            sPT[None, pt_index], (pages_kv // mma_ctas,), (mma_rank,)
                        )
                        rPT = sPT_cta.load()
                        pt_next = 0 if pt_index == pt_stages - 1 else pt_index + 1
                        if coord_kv_ + kv_splits < kv_stop:
                            prefetch_pages(coord_kv + coord_kv_ + kv_splits, page_count,
                                        table_offset, sPT[None, pt_next])
                        sK_handle = sKV_producer.acquire_and_advance(expected_tx=sK_stage_bytes)
                        for page in range_constexpr(pages_kv // mma_ctas):
                            physical_page = rPT[page]
                            cute.copy(
                                tma_k.atom, tBgK_hb[None, physical_page],
                                tBsK_tma[None, page, sK_handle.index],
                                tma_bar_ptr=sK_handle.barrier,
                            )
                        pt_index = pt_next
                        tma_order_k_nbar.arrive()

                        # Advance for TMA V
                        if coord_kv_ >= prefetch_tiles_kv:
                            sKV_producer.advance()

                    # Request next work from CLC
                    if const_expr(is_persistent):
                        if cluster_rank == 0:
                            query_next_work(clc_producer)
                            clc_producer.advance()

                    # Tail gV loop
                    sKV_producer.advance(prefetch_iters_kv)
                    for _ in range_constexpr(prefetch_iters_kv):
                        tma_order_v_nbar.arrive_and_wait()
                        tma_order_k_nbar.arrive()
                    # endregion load paged gK loop

                    # Persistent loop update
                    next_work_valid = False
                    if const_expr(is_persistent):
                        # Read CLC response from smem
                        block_idx, next_work_valid = get_next_work(clc_consumer)
                        clc_consumer.advance()

                        if next_work_valid:
                            # Update tile coordinates
                            coord_hg_qo, coord_hk_b = get_coords(block_idx)
                            coord_hk, coord_b = coord_hk_b
                            _, coord_qo = coord_hg_qo
                            seqlen_kv = get_seqlen(coord_b)
                            page_count, table_offset = get_page_info(coord_b, seqlen_kv)
                            coord_kv, tiles_kv = mask_config.get_tiles_kv(
                                seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                            )
                            iters_kv = max(tiles_kv, prefetch_iters_kv)
            else:
                # region  Partition contig gK
                # (TILE_KV, TILE_DQ, #TILE_KV, (h_k, b))
                gK = cute.local_tile(
                    tma_k.tma_tensor,
                    tiler=(blk_tile_kv, blk_tile_dq),
                    coord=(None, 0, None),
                )
                # (MMA, #MMA_N, #MMA_K, #TILE_KV, (h_k, b))
                tBgK = thrblk_mma_qk.partition_B(gK)
                # (TMA, #TILE_KV, (h_k, b))
                tBsK_tma, tBgK_tma = tma_partition(tma_k.atom, tBsK, tBgK)
                # endregion partition contig gK

                cute.arch.griddepcontrol_wait()

                # Persistent Loop
                next_work_valid = True
                while next_work_valid:
                    # Work tile slice
                    tBgK_hb = tBgK_tma[None, None, coord_hk_b]
                    # Masking offset
                    tBgK_hb = cute.domain_offset((None, coord_kv), tBgK_hb)

                    # region Load contig gK loop
                    prefetch_tiles_kv = prefetch_iters_kv * kv_splits
                    # Match MMA/softmax, including the minimum prefetch iterations.
                    kv_stop = kv_split_idx + iters_kv * kv_splits
                    for coord_kv_ in cutlass.range(kv_split_idx, kv_stop, kv_splits):
                        sK_handle = sKV_producer.acquire_and_advance(expected_tx=sK_stage_bytes)
                        tma_order_v_nbar.arrive_and_wait()
                        cute.copy(
                            tma_k.atom,
                            tBgK_hb[None, coord_kv_],
                            tBsK_tma[None, sK_handle.index],
                            tma_bar_ptr=sK_handle.barrier,
                        )
                        tma_order_k_nbar.arrive()

                        # Advance for TMA V
                        if coord_kv_ >= prefetch_tiles_kv:
                            sKV_producer.advance()

                    # Request next work from CLC
                    if const_expr(is_persistent):
                        if cluster_rank == 0:
                            query_next_work(clc_producer)
                            clc_producer.advance()

                    # Tail gV loop
                    sKV_producer.advance(prefetch_iters_kv)
                    for _ in range_constexpr(prefetch_iters_kv):
                        tma_order_v_nbar.arrive_and_wait()
                        tma_order_k_nbar.arrive()
                    # endregion load contig gK loop

                    # Persistent loop update
                    next_work_valid = False
                    if const_expr(is_persistent):
                        # Read CLC response from smem
                        block_idx, next_work_valid = get_next_work(clc_consumer)
                        clc_consumer.advance()

                        if next_work_valid:
                            # Update tile coordinates
                            coord_hg_qo, coord_hk_b = get_coords(block_idx)
                            coord_hk, coord_b = coord_hk_b
                            _, coord_qo = coord_hg_qo
                            seqlen_kv = get_seqlen(coord_b)
                            coord_kv, tiles_kv = mask_config.get_tiles_kv(
                                seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                            )
                            iters_kv = max(tiles_kv, prefetch_iters_kv)
            if const_expr(is_persistent):
                if cluster_rank == 0:
                    # Every CLC consumer releases to CTA 0, including after
                    # the final invalid response. Keep its shared memory live
                    # until the peer CTA's final releases have completed.
                    clc_producer.tail()
        # --------------------------------------------------------------
        # endregion TMA K
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  TMA V Dispatch
        # --------------------------------------------------------------
        elif warp_idx == tma_v_warp_id:
            cute.arch.setmaxregister_decrease(mma_tma_regs)

            if const_expr(is_paged):
                # Barrier setup
                tma_order_v_nbar.arrive()

                # region  Partition paged gV
                # (DO / CTA, PAGE, heads_k, physical_pages)
                gV = cute.local_tile(
                    tma_v.tma_tensor,
                    tiler=(blk_tile_do // mma_ctas, self.page_size),
                    coord=(mma_rank, 0, None),
                )
                sV_tma_view = cute.make_tensor(tBsV.iterator, tma_v.smem_layout.outer)
                tBsV_tma, tBgV_tma = cpasync.tma_partition(
                    tma_v.atom, 0, cute.make_layout(1), sV_tma_view,
                    cute.group_modes(gV, 0, 2),
                )
                # endregion  partition paged gV

                # Persistent loop
                next_work_valid = True
                while next_work_valid:
                    # Work tile slice
                    tBgV_hb = tBgV_tma[None, (coord_hk, None)]
                    pt_index = Int32(0)

                    # region  Load paged gV loop
                    # Prefetch K loop
                    sKV_producer.advance(prefetch_iters_kv)
                    for _ in range_constexpr(prefetch_iters_kv):
                        tma_order_k_nbar.arrive_and_wait()
                        tma_order_v_nbar.arrive()

                    prefetch_tiles_kv = prefetch_iters_kv * kv_splits
                    # Match MMA/softmax, including the minimum prefetch iterations.
                    kv_stop = kv_split_idx + iters_kv * kv_splits
                    for coord_kv_ in cutlass.range(kv_split_idx, kv_stop, kv_splits):
                        # Advance for TMA K
                        if coord_kv_ < kv_stop - prefetch_tiles_kv:
                            sKV_producer.advance()

                        sV_handle = sKV_producer.acquire_and_advance(expected_tx=sV_stage_bytes)
                        tma_order_k_nbar.arrive_and_wait()
                        # K's named-barrier arrival publishes the completed page copy.
                        rPT = sPT[None, pt_index].load()
                        for page in range_constexpr(pages_kv):
                            cute.copy(
                                tma_v.atom, tBgV_hb[None, rPT[page]],
                                tBsV_tma[None, page, sV_handle.index],
                                tma_bar_ptr=sV_handle.barrier,
                            )
                        pt_index = 0 if pt_index == pt_stages - 1 else pt_index + 1
                        tma_order_v_nbar.arrive()
                    # endregion  load paged gV loop
                    if const_expr(do_aux_load):
                        last_v_nbar.arrive_and_wait()

                    # Persistent loop update
                    next_work_valid = False
                    if const_expr(is_persistent):
                        # Read CLC response from smem
                        block_idx, next_work_valid = get_next_work(clc_consumer)
                        clc_consumer.advance()

                        # Update tile coordinates
                        if next_work_valid:
                            coord_hg_qo, coord_hk_b = get_coords(block_idx)
                            coord_hk, coord_b = coord_hk_b
                            _, coord_qo = coord_hg_qo
                            seqlen_kv = get_seqlen(coord_b)
                            coord_kv, tiles_kv = mask_config.get_tiles_kv(
                                seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                            )
                            iters_kv = max(tiles_kv, prefetch_iters_kv)
            else:
                # Barrier setup
                tma_order_v_nbar.arrive()

                # region  Partition contig gV
                # (TILE_DO, TILE_KV, #TILE_KV, (h_k, b))
                gV = cute.local_tile(
                    tma_v.tma_tensor,
                    tiler=(blk_tile_do, blk_tile_kv),
                    coord=(0, None, None),
                )
                # (MMA, #MMA_N, #MMA_K, #TILE_KV, (h_k, b))
                tBgV = thrblk_mma_pv.partition_B(gV)
                # (TMA, #TILE_KV, (h_k, b))
                tBsV_tma, tBgV_tma = tma_partition(tma_v.atom, tBsV, tBgV)
                # endregion  partition contig gV

                # Persistent loop
                next_work_valid = True
                while next_work_valid:
                    # Work tile slice
                    tBgV_hb = tBgV_tma[None, None, coord_hk_b]
                    # Masking offset
                    tBgV_hb = cute.domain_offset((None, coord_kv), tBgV_hb)

                    # region  Load contig gV loop
                    # Prefetch K loop
                    sKV_producer.advance(prefetch_iters_kv)
                    for _ in range_constexpr(prefetch_iters_kv):
                        tma_order_k_nbar.arrive_and_wait()
                        tma_order_v_nbar.arrive()

                    prefetch_tiles_kv = prefetch_iters_kv * kv_splits
                    # Match MMA/softmax, including the minimum prefetch iterations.
                    kv_stop = kv_split_idx + iters_kv * kv_splits
                    for coord_kv_ in cutlass.range(kv_split_idx, kv_stop, kv_splits):
                        # Advance for TMA K
                        if coord_kv_ < kv_stop - prefetch_tiles_kv:
                            sKV_producer.advance()

                        sV_handle = sKV_producer.acquire_and_advance(expected_tx=sV_stage_bytes)
                        tma_order_k_nbar.arrive_and_wait()
                        cute.copy(
                            tma_v.atom,
                            tBgV_hb[None, coord_kv_],
                            tBsV_tma[None, sV_handle.index],
                            tma_bar_ptr=sV_handle.barrier,
                        )
                        tma_order_v_nbar.arrive()
                    # endregion  load contig gV loop
                    if const_expr(do_aux_load):
                        last_v_nbar.arrive_and_wait()

                    # Persistent loop update
                    next_work_valid = False
                    if const_expr(is_persistent):
                        # Read CLC response from smem
                        block_idx, next_work_valid = get_next_work(clc_consumer)
                        clc_consumer.advance()

                        if next_work_valid:
                            # Update tile coordinates
                            coord_hg_qo, coord_hk_b = get_coords(block_idx)
                            coord_hk, coord_b = coord_hk_b
                            _, coord_qo = coord_hg_qo
                            seqlen_kv = get_seqlen(coord_b)
                            coord_kv, tiles_kv = mask_config.get_tiles_kv(
                                seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                            )
                            iters_kv = max(tiles_kv, prefetch_iters_kv)
        # --------------------------------------------------------------
        # endregion TMA V
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region  MMA Dispatch
        # --------------------------------------------------------------
        elif warp_idx == mma_warp_id:
            cute.arch.setmaxregister_decrease(mma_tma_regs)

            # region  MMA QK, BMM1
            if const_expr(not use_tQ):
                tAsQ_desc = tiled_mma_qk.make_fragment_A(tAsQ)
            tBsK_desc = tiled_mma_qk.make_fragment_B(tBsK)
            def mma_qk(
                tAxQ : cute.Tensor,
                sKV_consumer : pipeline.PipelineConsumer,
                tS_producer : pipeline.PipelineProducer,
                skip_tS_acquire : bool,
                *,
                tiled_mma_qk = tiled_mma_qk,
                tCtS = tCtS,
                tBsK_desc = tBsK_desc,
            ):
                tS_index = tS_producer.current_handle().index
                sK_index = sKV_consumer.current_handle().index
                tCtS_s = tCtS[None, tS_index]
                tAxQ_s = tAxQ[None, 0, None]
                tBsK_s = tBsK_desc[None, 0, None, sK_index]
                pass
                sKV_consumer.wait()
                if const_expr(not skip_tS_acquire):
                    tS_producer.acquire()
                mmas_k = cute.size(tAxQ.shape[2])
                for mma_k in range_constexpr(mmas_k):
                    tiled_mma_qk = tiled_mma_qk.with_()  # copy, dont carry runtime state
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, mma_k > 0)
                    cute.gemm(
                        tiled_mma_qk,
                        tCtS_s,
                        tAxQ_s[None, mma_k],
                        tBsK_s[None, mma_k],
                        tCtS_s,
                    )
                    pass
                tS_producer.commit()
                sKV_consumer.release()
                pass
                sKV_consumer = sKV_consumer.clone()
                tS_producer = tS_producer.clone()
                sKV_consumer.advance()
                tS_producer.advance()
                return sKV_consumer, tS_producer
            # endregion MMA QK

            # region MMA PV, BMM2
            tBsV_desc = tiled_mma_pv.make_fragment_B(tBsV)
            if const_expr(use_mma_p1):
                tBs1_desc = tiled_mma_p1.make_fragment_B(tBs1)
                # slice unused modes
                tBs1_desc = tBs1_desc[None, 0, 0, 0]
            def mma_pv(
                sKV_consumer : pipeline.PipelineConsumer,
                tP_consumer : pipeline.PipelineConsumer,
                zero_tO : Boolean,
                *,
                tiled_mma_pv = tiled_mma_pv,
                tCtO = tCtO,
                tAtP = tAtP,
                tBsV_desc = tBsV_desc,
                tiled_mma_p1 = tiled_mma_p1,
                tCtP1 = tCtP1 if use_mma_p1 else None,
                tBs1_desc = tBs1_desc if use_mma_p1 else None,
            ):
                sV_index = sKV_consumer.current_handle().index
                tO_index = tP_consumer.current_handle().index
                tP_index = tO_index ^ 1
                tCtO_s = tCtO[None, tO_index]
                tAtP_s = tAtP[None, 0, None, tP_index]
                tBsV_s = tBsV_desc[None, 0, None, sV_index]
                if const_expr(tiled_mma_p1 is not None):
                    tCtP1_s = tCtP1[None, tO_index]
                sKV_consumer.wait()
                pass
                tP_consumer.wait()
                mmas_k = cute.size(tAtP.shape[2])
                for mma_k in range_constexpr(mmas_k):
                    accumulate_tO = not (mma_k == 0 and zero_tO)
                    tiled_mma_pv_ = tiled_mma_pv.with_()  # copy, dont carry runtime state
                    tiled_mma_pv_.set(tcgen05.Field.ACCUMULATE, accumulate_tO)
                    cute.gemm(
                        tiled_mma_pv_,
                        tCtO_s,
                        tAtP_s[None, mma_k],
                        tBsV_s[None, mma_k],
                        tCtO_s,
                    )
                    if const_expr(tiled_mma_p1 is not None):
                        tiled_mma_p1 = tiled_mma_p1.with_()  # copy, dont carry runtime state
                        tiled_mma_p1.set(tcgen05.Field.ACCUMULATE, mma_k > 0)
                        cute.gemm(
                            tiled_mma_p1,
                            tCtP1_s,
                            tAtP_s[None, mma_k],
                            tBs1_desc,
                            tCtP1_s,
                        )
                    pass
                tP_consumer.release()
                sKV_consumer.release()
                pass
                sKV_consumer = sKV_consumer.clone()
                tP_consumer = tP_consumer.clone()
                sKV_consumer.advance()
                tP_consumer.advance()
                return sKV_consumer, tP_consumer
            # endregion MMA PV

            next_work_valid = True
            while next_work_valid:
                # Wait for sQ
                if const_expr(use_tQ):
                    mma_tma_nbar.arrive_and_wait()
                    tcgen05_fence(Tcgen05Fence.AFTER_THREAD_SYNC)
                    tAxQ = tAtQ
                else:
                    sQ_consumer.wait()
                    sQ_index = sQ_consumer.current_handle().index
                    tAxQ = tAsQ_desc[None, None, None, sQ_index]

                # region  MMA loop
                for _ in range_constexpr(prefetch_iters_kv):
                    sKV_consumer, tS_producer = mma_qk(
                        tAxQ, sKV_consumer, tS_producer, skip_tS_acquire=False
                    )

                tS_producer.acquire()
                for kv in cutlass.range(iters_kv - prefetch_iters_kv):
                    # After the first PV we can skip subsequent tS acquire
                    # since tP wait implies tS was released
                    sKV_consumer, tS_producer = mma_qk(
                        tAxQ, sKV_consumer, tS_producer, skip_tS_acquire=True
                    )

                    sKV_consumer, tP_consumer = mma_pv(
                        sKV_consumer, tP_consumer, kv < tO_stages
                    )
                sQ_consumer.release()

                for kv_tail in range_constexpr(prefetch_iters_kv):
                    kv = kv_tail + iters_kv - prefetch_iters_kv
                    sKV_consumer, tP_consumer = mma_pv(
                        sKV_consumer, tP_consumer, kv < tO_stages
                    )
                # endregion MMA loop

                next_work_valid = False
                if const_expr(is_persistent):
                    sQ_consumer.advance()
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()

                    if next_work_valid:
                        # Update KV tile iterations
                        (_, coord_qo), (_, coord_b) = get_coords(block_idx)
                        seqlen_kv = get_seqlen(coord_b)
                        coord_kv, tiles_kv = mask_config.get_tiles_kv(
                            seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                        )
                        iters_kv = max(tiles_kv, prefetch_iters_kv)
        # --------------------------------------------------------------
        # endregion MMA
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region Softmax Dispatch
        # --------------------------------------------------------------
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Alloc registers
            cute.arch.setmaxregister_increase(softmax_regs)

            # Pipeline setup
            softmax_order = softmax_phase = (warpgroup_idx - 1) % softmax_warpgroups
            aux_phase = Int32(0)
            tP_alias_acquire = lambda : cute.arch.mbarrier_wait(tSP_alias_mbar, softmax_phase ^ 1)
            tS_alias_release = lambda : cute.arch.mbarrier_arrive(tSP_alias_mbar)
            sM_commit = with_phase(sM_producer_nbars, softmax_phase * warpgroup_warps + warpgroup_widx).arrive
            if softmax_phase == 1:
                tS_consumer.advance()
                tP_producer.advance()

            # region Partition tS, tP
            # Select copy atom
            tS_ld_atom = make_tmem_ld_atom(blk_tile_kv, acc_dtype)
            tS_red_atom = make_tmem_red_atom(blk_tile_kv, acc_dtype)
            # Tile atom and partition
            tCtS_mma = tCtS[None, softmax_phase]
            tS_ld_tiled = tile_tmem_atom(tS_ld_atom, tCtS_mma)
            # (CPY, #CPY_MMA)
            frg_tS = tS_ld_tiled.partition_S(tCtS_mma)
            frg_rS_shape = tS_ld_tiled.partition_D(tCtS_mma).shape
            elts_kv = cute.size(frg_rS_shape)
            num_tS_ld = cute.size(frg_rS_shape[1])  # tmem loads per accumulator (1 or 2)
            assert num_tS_ld in (1, 2)

            # Calculate masking offsets
            # ((TILE_HG, TILE_QO), TILE_KV)
            cQ = cute.make_identity_tensor((blk_tile_hg_qo, blk_tile_kv))
            # (MMA, #MMA_M, #MMA_N=1)
            tCcQ = thrblk_mma_qk.partition_C(cQ)
            # (CPY, #CPY_MMA, #MMA_M, #MMA_N=1)
            frg_cQ = tS_ld_tiled.partition_D(tCcQ)
            # local QK indices in block tile
            (blk_idx_hg, blk_idx_qo), blk_idx_kv = frg_cQ[0]
            idx_hg = coord_hg * blk_tile_hg + blk_idx_hg
            idx_qo = coord_qo * blk_tile_qo + blk_idx_qo
            not_oob_m = (idx_hg < grouped_heads) and (idx_qo < seqlen_qo)

            # Partition sM
            # (CPY, #CPY_MMA, tO_stages)
            frg_sM = tS_ld_tiled.partition_D(tCsM)
            # (CPY, tO_stages)
            frg_sM = frg_sM[None, 0, None]

            # Select copy atom
            tP_st_atom = make_tmem_st_atom(blk_tile_kv, mma_dtype)
            # Tile atom and partition
            tAtP_stage = tAtP[mma_dice + (0,)]
            tP_st_tiled = tile_tmem_atom(tP_st_atom, tAtP_stage)
            # (CPY, #CPY_MMA=1, #CPY_M=1, #CPY_K=1, tP_stages)
            frg_tP = tP_st_tiled.partition_D(tAtP)
            frg_rP_shape = tP_st_tiled.partition_S(tAtP).shape
            # Slice unused modes
            # (CPY,)
            assert cute.size(frg_rP_shape[1:-1]) == 1
            frg_tP = frg_tP[None, 0, 0, 0, softmax_phase ^ 1]
            frg_rP_shape = frg_rP_shape[:1]
            # endregion Partition tP

            # Max reduction alias
            ssa_max = partial(
                cute.TensorSSA.reduce,
                op=cute.ReductionOp.MAX,
                reduction_profile=0,  # scalar reduction
            )

            # region  Persistent loop
            next_work_valid = True
            while next_work_valid:
                # Mask configuration loop args
                range_args = mask_config.get_range_args(
                    seqlen_qo,
                    seqlen_kv,
                    blk_tile_qo,
                    blk_tile_kv,
                    coord_qo,
                    coord_kv,
                    tiles_kv,
                    iters_kv,
                    kv_splits,
                    kv_split_idx,
                    softmax_warpgroups,
                    softmax_order,
                )
                num_mask_phases = len(range_args)

                # Init total row sum + max
                vec_elts = 2  # intermediate RF values for rowsum
                phase_max = -Float32.inf
                applied_max = min_f32  # prevent nans with masking
                vec_sum = cute.full((vec_elts,), 0, acc_dtype)

                # --------------------------------------------------------------
                # region  Softmax loop
                # --------------------------------------------------------------
                # reordering for latency hiding
                if const_expr(self.alias_tP):
                    tP_alias_acquire()

                for mask_phase in range_constexpr(num_mask_phases):
                    start, stop, step, is_masked = range_args[mask_phase]
                    for coord_kv_ in cutlass.range(start, stop, step):
                        # Load S from tmem
                        tS_consumer.wait()
                        frg_rS = cute.make_rmem_tensor(frg_rS_shape, acc_dtype)
                        use_tmem_red = tmem_red_supported and not is_masked
                        if const_expr(use_tmem_red):
                            red_rS = cute.make_rmem_tensor(((1, 1), num_tS_ld), acc_dtype)
                            cute.copy(tS_red_atom, frg_tS, (frg_rS, red_rS))
                        else:
                            red_rS = frg_rS
                            cute.copy(tS_ld_atom, frg_tS, frg_rS)
                        cute.arch.fence_view_async_tmem_load()
                        tS_consumer.release()
                        if const_expr(self.alias_tP):
                            tS_alias_release()

                        # Apply mask
                        if const_expr(is_masked):
                            pass
                            # TODO: P2R before LDTM, R2P after LDTM
                            # global KV index of 0th elt in thread
                            idx_kv_0 = coord_kv_ * blk_tile_kv + blk_idx_kv
                            for kv in range_constexpr(elts_kv):
                                idx_kv = idx_kv_0 + kv
                                is_oob_kv = mask_config.is_oob_kv(
                                    idx_qo, idx_kv, seqlen_qo, seqlen_kv
                                )
                                frg_rS[kv] = -Float32.inf if is_oob_kv else frg_rS[kv]

                        # Calculate max
                        cur_max = ssa_max(red_rS.load(), init_val=applied_max)
                        if const_expr(tmem_row_lanes == 2):
                            cur_max_t16 = cute.arch.shuffle_sync_bfly(cur_max, warp_threads // 2)
                            cur_max = cute.arch.fmax(cur_max, cur_max_t16)

                        # Calculate correction rescale
                        delta = cur_max - applied_max
                        if const_expr(self.use_softmax_scale):
                            delta *= scale_s_log2_e
                        if const_expr(rescale_threshold is not None):
                            phase_max = cute.arch.fmax(phase_max, cur_max)
                            do_rescale = warp_fmax(delta) > rescale_threshold
                            applied_max = cur_max if do_rescale else applied_max
                            correction = exp2(-delta) if do_rescale else Float32(1)
                        else:
                            applied_max = phase_max = cur_max
                            correction = exp2(-delta)

                        # Handoff rescale to correction warps
                        frg_sM[0, softmax_phase] = correction
                        sM_commit()
                        pass
                        if const_expr(not use_mma_p1):
                            vec_sum *= correction

                        # Compute online softmax
                        if const_expr(self.softmax_dtype == Float16):
                            if const_expr(not use_mma_p1):
                                frg_rP_sum = cute.make_rmem_tensor(frg_rS_shape, Float16)
                            frg_rP = cute.make_rmem_tensor(frg_rP_shape, mma_dtype)
                            frg_rP_reshape = cute.composition(frg_rP, cute.make_layout(frg_rS_shape))
                            for ld in range_constexpr(num_tS_ld):
                                scores = frg_rS[None, ld].load()
                                scores = sub_f32_to_f16(scores, applied_max)
                                if const_expr(self.use_softmax_scale):
                                    scores *= Float16(scale_s_log2_e)
                                probs = exp2(scores)
                                if const_expr(self.quantization_scale is not None):
                                    probs *= Float16(self.quantization_scale)

                                # Convert P to mma type
                                frg_rP_reshape[None, ld].store(probs.to(mma_dtype))
                                if const_expr(not use_mma_p1):
                                    frg_rP_sum[None, ld].store(probs)
                                # {$nv-release-never begin}
                                # Reduce live ranges of cvt regs to promote reg reuse
                                # and eliminate MOV/spills in sass
                                if const_expr(ld < num_tS_ld - 1):
                                    pass
                                # {$nv-release-never end}

                            # Store P to tmem and notify BMM2
                            warp_sync_with_values(frg_rP) # dont reorder blocking acquire before softmax ops
                            pass
                            tP_producer.acquire()  # blocking retry loop
                            if const_expr(self.alias_tP):
                                tP_alias_acquire()
                            cute.copy(tP_st_atom, frg_rP, frg_tP)
                            cute.arch.fence_view_async_tmem_store()
                            tP_producer.commit()  # notify BMM2

                            # Reduce sum
                            if const_expr(not use_mma_p1):
                                pass
                                num_vec = elts_kv // vec_elts
                                vec_probs = frg_rP_sum.load().reshape((vec_elts, num_vec))
                                for v in range_constexpr(num_vec):
                                    vec_sum += vec_probs[None, v]
                        else:
                            assert self.softmax_dtype == Float32
                            assert not use_mma_p1, "mma rowsum only helpful for fp16 softmax"

                            frg_rP = cute.make_rmem_tensor(frg_rP_shape, mma_dtype)
                            frg_rP_reshape = cute.composition(frg_rP, cute.make_layout(frg_rS_shape))
                            for ld in range_constexpr(num_tS_ld):
                                scores = frg_rS[None, ld].load()
                                if const_expr(self.use_softmax_scale and self.use_softmax_fma):
                                    scores = scale_s_log2_e * scores - (scale_s_log2_e * applied_max)
                                else:
                                    scores = scores - applied_max
                                    if const_expr(self.use_softmax_scale):
                                        scores *= scale_s_log2_e
                                probs = exp2(scores)
                                if const_expr(self.quantization_scale is not None):
                                    # We can fold log2 scale into max if perf is important for this case
                                    probs *= self.quantization_scale

                                # Convert P to mma type
                                frg_rP_reshape[None, ld].store(probs.to(mma_dtype))

                                # Reduce sum
                                if const_expr(ld < num_tS_ld - 1):
                                    num_vec = elts_kv // num_tS_ld // vec_elts
                                    vec_probs = probs.reshape((vec_elts, num_vec))
                                    for v in range_constexpr(num_vec):
                                        vec_sum += vec_probs[None, v]
                                    pass

                            # Store P to tmem and notify BMM2
                            warp_sync_with_values(frg_rP) # dont reorder blocking acquire before softmax ops
                            pass
                            tP_producer.acquire()
                            if const_expr(self.alias_tP):
                                tP_alias_acquire()
                            cute.copy(tP_st_atom, frg_rP, frg_tP)
                            cute.arch.fence_view_async_tmem_store()
                            tP_producer.commit()  # notify BMM2

                            # Reduce sum (tail)
                            pass
                            num_vec = elts_kv // num_tS_ld // vec_elts
                            vec_probs = probs.reshape((vec_elts, num_vec))
                            for v in range_constexpr(num_vec):
                                vec_sum += vec_probs[None, v]

                        # Pingpong warpgroup advance
                        tS_consumer.advance(softmax_warpgroups)
                        tP_producer.advance(softmax_warpgroups)

                # Tail release
                if const_expr(self.alias_tP):
                    tS_alias_release()
                # --------------------------------------------------------------
                # endregion  Softmax loop
                # --------------------------------------------------------------

                # --------------------------------------------------------------
                # region  Softmax epilogue
                # --------------------------------------------------------------
                # Load sink
                if const_expr(do_none_red and use_sink):
                    # (TILE_HG)
                    gSink = cute.local_tile(
                        mSink, (blk_tile_hg,), (coord_hg, coord_hk)
                    )
                    sink = -Float32.inf
                    if not_oob_m:
                        sink = gSink[blk_idx_hg]

                # region  partition tO, sO
                # Select atoms
                tO_ld_atom = make_tmem_ld_atom(
                    # load both stages, half stage per warpgroup
                    blk_tile_do // softmax_warpgroups, acc_dtype
                )
                # Tile atom and partition
                tO_ld_tiled = tile_tmem_atom(tO_ld_atom, tCtO[None, 0])
                # (CPY, #CPY_MMA=2, tO_stages)
                frg_tO = tO_ld_tiled.partition_S(tCtO)
                # (CPY, #CPY_MMA=2, #MMA_M=1, #MMA_N=1, sO_stages)
                frg_sO = tO_ld_tiled.partition_D(tCsO)
                assert cute.size(frg_tO.shape[1]) == softmax_warpgroups
                assert cute.size(frg_sO.shape[2:-1]) == 1
                # (CPY, tO_stages)
                frg_tO = frg_tO[None, softmax_phase, None]
                # (CPY, sO_stages)
                sO_index = sQ_consumer.current_handle().index
                frg_sO = frg_sO[None, softmax_phase, 0, 0, sO_index]
                frg_rO_shape = (frg_sO.shape[0], tO_stages)

                # tM - vector stats alised to tO
                num_stat = 2 if self.use_mma_p1 else 4  # pass up to 4 32b stats per thread
                tM_cols = num_stat * tmem_row_lanes
                tM_ld_atom = make_tmem_ld_atom(tM_cols, acc_dtype)
                tM_st_atom = make_tmem_st_atom(tM_cols, acc_dtype)
                # Tile atom and partition
                tM_ld_tiled = tile_tmem_atom(tM_ld_atom, tCtO[None, 0])
                # (CPY=#STAT, #CPY_MMA, tO_stages)
                frg_tM = tM_ld_tiled.partition_S(tCtO)
                frg_rM_shape = tM_ld_tiled.partition_D(tCtO).shape[:2]
                num_tM_ld = cute.size(frg_rM_shape[1])
                assert cute.size(frg_rM_shape[0]) == num_stat
                # (CPY, tO_stages)
                cpy_phase = num_tM_ld // 2 * softmax_phase
                cpy_peer = num_tM_ld // 2 * (softmax_phase ^ 1)
                frg_tM_phase = frg_tM[None, cpy_phase, softmax_phase]
                frg_tM_peer = frg_tM[None, cpy_peer, softmax_phase ^ 1]
                # endregion  partition tO, sO

                # Wait for BMM2 completion
                tP_producer.acquire()  # this phases BMM2 finished

                # Load half of tO from this phase
                frg_rO = cute.make_rmem_tensor(frg_rO_shape, acc_dtype)
                cute.copy(tO_ld_atom, frg_tO[None, softmax_phase ^ 0], frg_rO[None, 0])
                cute.arch.fence_view_async_tmem_load()

                # Handoff this phase's max and sum
                frg_rM = cute.make_rmem_tensor(frg_rM_shape[:1], acc_dtype)
                frg_rM[0], frg_rM[1] = phase_max, applied_max
                if const_expr(not use_mma_p1):
                    phase_sum = vec_sum[0]
                    for i in range_constexpr(vec_elts - 1):
                        phase_sum += vec_sum[i+1]
                    frg_rM[2] = phase_sum
                cute.copy(tM_st_atom, frg_rM, frg_tM_phase)
                cute.arch.fence_view_async_tmem_store()
                softmax_sync()  # peer phases BMM2 finished

                # Load half of tO from peer phase
                cute.copy(tO_ld_atom, frg_tO[None, softmax_phase ^ 1], frg_rO[None, 1])

                # Load peer phase max and sum
                cute.copy(tM_ld_atom, frg_tM_peer, frg_rM)
                peer_phase_max, peer_applied_max = frg_rM[0], frg_rM[1]
                if const_expr(not use_mma_p1):
                    peer_phase_sum = frg_rM[2]
                else:
                    sM_final_nbar.arrive_and_wait()
                    phase_sum = frg_sM[0, softmax_phase]
                    peer_phase_sum = frg_sM[0, softmax_phase ^ 1]

                # Reduce output across both phases
                # Correct both phases to prevent lane divergence
                split_max = cute.arch.fmax(phase_max, peer_phase_max)
                delta_phase = applied_max - split_max
                delta_peer = peer_applied_max - split_max
                if const_expr(self.use_softmax_scale):
                    delta_phase *= scale_s_log2_e
                    delta_peer *= scale_s_log2_e
                correction_phase = exp2(delta_phase)
                correction_peer = exp2(delta_peer)
                # Calculate split sum
                split_sum = correction_phase * phase_sum
                split_sum += correction_peer * peer_phase_sum
                if const_expr(tmem_row_lanes == 2):
                    split_sum += cute.arch.shuffle_sync_bfly(split_sum, warp_threads // 2)
                # Calculate split output
                output = correction_phase * frg_rO[None, 0].load()
                output += correction_peer * frg_rO[None, 1].load()
                # Apply final normalization
                if const_expr(do_none_red):
                    final_max_log2 = split_max
                    if const_expr(self.use_softmax_scale):
                        final_max_log2 *= scale_s_log2_e
                    final_sum = split_sum
                    if const_expr(use_sink):
                        sink_prob = exp2(log2_e * sink - final_max_log2)
                        if const_expr(self.quantization_scale is not None):
                            sink_prob *= Float32(self.quantization_scale)
                        final_sum += sink_prob
                    normalization = cute.arch.rcp_approx(final_sum) * scale_o
                    output *= normalization
                elif const_expr(do_kernel_red):
                    # The shared reduction kernel expects unquantized partials.
                    if const_expr(self.quantization_scale is not None):
                        inv_quantization_scale = Float32(
                            1.0 / self.quantization_scale
                        )
                        split_sum *= inv_quantization_scale
                        output *= inv_quantization_scale

                if const_expr(do_aux_load):
                    cute.arch.mbarrier_wait(aux_mbar, aux_phase)
                    aux_phase ^= 1
                    aux_values = frg_sO.load()
                    output = self.aux_fn(output, aux_values)
                # Store output to smem
                frg_sO.store(output.to(out_dtype))
                cute.arch.fence_view_async_shared()
                sO_final_nbar.arrive()  # Notify O TMA store

                # Store LSE
                if const_expr(do_none_red and store_lse):
                    # (TILE_HG, TILE_QO)
                    gL = cute.local_tile(
                        mL, (blk_tile_hg_qo,), (coord_hg_qo, coord_hk_b)
                    )[(None,None),]
                    is_lse_writer = not_oob_m
                    if const_expr(tmem_row_lanes == 2):
                        is_lse_writer &= lane_idx < warp_threads // 2
                    if is_lse_writer and softmax_phase == 0:
                        lse = final_max_log2 + cute.math.log2(final_sum)
                        if const_expr(self.quantization_scale is not None):
                            lse -= Float32(math.log2(self.quantization_scale))
                        gL[blk_idx_hg, blk_idx_qo] = lse

                # Store partials to gmem
                if const_expr(do_kernel_red):
                    gM_partial = cute.local_tile(
                        mM_partial,
                        (blk_tile_hg_qo,),
                        (coord_hg_qo, coord_hk_b, kv_split_idx),
                    )[(None, None),]
                    gL_partial = cute.local_tile(
                        mL_partial,
                        (blk_tile_hg_qo,),
                        (coord_hg_qo, coord_hk_b, kv_split_idx),
                    )[(None, None),]

                    partial_max_log2 = split_max
                    if const_expr(self.use_softmax_scale):
                        partial_max_log2 *= scale_s_log2_e

                    is_partial_writer = not_oob_m
                    if const_expr(tmem_row_lanes == 2):
                        is_partial_writer &= lane_idx < warp_threads // 2
                    if is_partial_writer and softmax_phase == 0:
                        gM_partial[blk_idx_hg, blk_idx_qo] = partial_max_log2
                        gL_partial[blk_idx_hg, blk_idx_qo] = split_sum

                # --------------------------------------------------------------
                # endregion  Softmax epilogue
                # --------------------------------------------------------------

                # Persistent loop update
                next_work_valid = False
                if const_expr(is_persistent):
                    # Determine which phase goes next
                    softmax_order ^= iters_kv % 2
                    sQ_consumer.advance()
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()
                    if next_work_valid:
                        # Update tile coordinates and direct GMEM row indices.
                        coord_hg_qo, coord_hk_b = get_coords(block_idx)
                        coord_hg, coord_qo = coord_hg_qo
                        coord_hk, coord_b = coord_hk_b
                        idx_hg = coord_hg * blk_tile_hg + blk_idx_hg
                        idx_qo = coord_qo * blk_tile_qo + blk_idx_qo
                        not_oob_m = (idx_hg < grouped_heads) & (idx_qo < seqlen_qo)
                        seqlen_kv = get_seqlen(coord_b)
                        coord_kv, tiles_kv = mask_config.get_tiles_kv(
                            seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                        )
                        iters_kv = max(tiles_kv, prefetch_iters_kv)
                    # Complete epilogue before proceeding to next tile
                    softmax_sync()
            # endregion  Persistent loop
        # --------------------------------------------------------------
        # endregion softmax
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # region Correction Dispatch
        # --------------------------------------------------------------
        elif warpgroup_idx == correction_warpgroup_id:
            # Free registers
            cute.arch.setmaxregister_decrease(correction_regs)

            # Barrier setup
            if const_expr(use_mma_p1 and tP1_stages == 1):
                tP_producer.commit()

            # region  Partition tO, tP1
            # Select atoms
            st_per_ld = 1 if (mma_tile_m == 64 or blk_tile_do == 64) else 2
            cols_per_ld = blk_tile_do if (mma_tile_m == 64 or blk_tile_do == 64) else 64
            tO_ld_atom = make_tmem_ld_atom(cols_per_ld, acc_dtype)
            tO_st_atom = make_tmem_st_atom(cols_per_ld // st_per_ld, acc_dtype)
            # Tile atoms and partition
            tO_ld_tiled = tile_tmem_atom(tO_ld_atom, tCtO[None, 0])
            tO_st_tiled = tile_tmem_atom(tO_st_atom, tCtO[None, 0])
            # (CPY, #CPY_MMA, tO_stages)
            frg_tO_ld = tO_ld_tiled.partition_S(tCtO)
            frg_tO_st = tO_st_tiled.partition_D(tCtO)
            frg_rO_ld_shape = tO_ld_tiled.partition_D(tCtO).shape[:2]
            frg_rO_st_shape = tO_st_tiled.partition_S(tCtO).shape[:2]
            num_ld = cute.size(frg_rO_ld_shape[1])
            num_st = cute.size(frg_rO_st_shape[1])
            assert (num_ld, num_st) in ((1, 1), (2, 4))

            if const_expr(use_mma_p1):
                tP1_ld_atom = make_tmem_ld_atom(1, acc_dtype)
                tP1_ld_tiled = tile_tmem_atom(tP1_ld_atom, tCtP1[None, 0])
                # (CPY=1, #CPY, tP1_stages)
                frg_tP1 = tP1_ld_tiled.partition_S(tCtP1)
                # (CPY=1, tP1_stages)
                frg_tP1 = frg_tP1[None, 0, None]
                frg_rP1_shape = tP1_ld_tiled.partition_D(tCtP1).shape[:1]

            # Smem stats load helper
            # (CPY, tO_stages)
            frg_sM = tO_ld_tiled.partition_D(tCsM)[None, 0, None]
            def stat_load(
                smax_phase,
                frg_sM=frg_sM,
                nbars=sM_producer_nbars,
                widx=warpgroup_widx,
            ):
                with_phase(nbars, smax_phase * 4 + widx).arrive_and_wait()
                return frg_sM[0, smax_phase]
            # endregion  partition tO, tP1

            # region  Persistent loop
            next_work_valid = True
            while next_work_valid:

                # region  Correction loop
                # Load stats of s-2, s-1 (no correction)
                for _ in range_constexpr(tO_stages):
                    softmax_phase = tP_producer.current_handle().index
                    _ = stat_load(softmax_phase)
                    tP_handle = tP_producer.acquire_and_advance()
                    tP_handle.commit()

                row_sum = row_sum_prev = Float32(0)
                for _ in cutlass.range(iters_kv - tO_stages):
                    tP_handle = tP_producer.acquire_and_advance()
                    softmax_phase = tP_handle.index
                    if const_expr(use_mma_p1):
                        frg_rP1 = cute.make_rmem_tensor(frg_rP1_shape, acc_dtype)
                        frg_tP1_s = frg_tP1[None, softmax_phase]
                        cute.copy(tP1_ld_atom, frg_tP1_s, frg_rP1)
                        if const_expr(tP1_stages == 1):
                            cute.arch.fence_view_async_tmem_load()
                            tP_producer.commit()  # arrive 1 stage ahead

                    # Load correction of s (wrt phase max of s-2)
                    correction = stat_load(softmax_phase)
                    do_rescale = (rescale_threshold is None or
                                  warp_fmin(correction) < Float32(1))

                    # Correct O of s-2 and notify MMA VP
                    if do_rescale:
                        # rescale tO
                        frg_tO_ld_s = frg_tO_ld[None, None, softmax_phase]
                        frg_tO_st_s = frg_tO_st[None, None, softmax_phase]
                        frg_rO_ld = cute.make_rmem_tensor(frg_rO_ld_shape, acc_dtype)
                        frg_rO_st = cute.composition(frg_rO_ld, cute.make_layout(frg_rO_st_shape))
                        cute.copy(tO_ld_atom, frg_tO_ld_s[None, 0], frg_rO_ld[None, 0])
                        for st in range_constexpr(num_st):
                            frg_rO_st[None, st].store(correction * frg_rO_st[None, st].load())
                            cute.copy(tO_st_atom, frg_rO_st[None, st], frg_tO_st_s[None, st])
                            if st == 0 and st_per_ld == 2:
                                cute.copy(tO_ld_atom, frg_tO_ld_s[None, 1], frg_rO_ld[None, 1])
                        cute.arch.fence_view_async_tmem_store()
                    tP_handle.commit()

                    # accumulate rowsum P
                    if const_expr(use_mma_p1):
                        row_sum = (row_sum + frg_rP1[0]) * correction
                        row_sum, row_sum_prev = row_sum_prev, row_sum
                # endregion  correction loop

                # region  Correction epilogue
                if const_expr(use_mma_p1):
                    tP_tail = tP_producer.clone()
                    softmax_phase = tP_tail.current_handle().index
                    tP_tail.acquire_and_advance()
                    frg_rP1 = cute.make_rmem_tensor(frg_rP1_shape, acc_dtype)
                    frg_tP1_s = frg_tP1[None, softmax_phase]
                    cute.copy(tP1_ld_atom, frg_tP1_s, frg_rP1)
                    if const_expr(tP1_stages == 1):
                        cute.arch.fence_view_async_tmem_load()
                        tP_tail.commit()  # arrive 1 stage ahead
                    row_sum += frg_rP1[0]
                    frg_sM[0, softmax_phase] = row_sum

                    tP_tail.acquire_and_advance()
                    frg_tP1_s = frg_tP1[None, softmax_phase ^ 1]
                    cute.copy(tP1_ld_atom, frg_tP1_s, frg_rP1)
                    if const_expr(tP1_stages == 1):
                        cute.arch.fence_view_async_tmem_load()
                        tP_tail.commit()  # arrive 1 stage ahead
                    row_sum_prev += frg_rP1[0]
                    frg_sM[0, softmax_phase ^ 1] = row_sum_prev
                    sM_final_nbar.arrive()
                # endregion  Correction epilogue

                # Persistent loop update
                next_work_valid = False
                if const_expr(is_persistent):
                    # Read CLC response from smem
                    block_idx, next_work_valid = get_next_work(clc_consumer)
                    clc_consumer.advance()
                    if next_work_valid:
                        # Update KV tile iterations
                        (_, coord_qo), (_, coord_b) = get_coords(block_idx)
                        seqlen_kv = get_seqlen(coord_b)
                        coord_kv, tiles_kv = mask_config.get_tiles_kv(
                            seqlen_qo, seqlen_kv, blk_tile_kv, blk_tile_qo, coord_qo
                        )
                        iters_kv = max(tiles_kv, prefetch_iters_kv)
                # endregion  Persistent loop
        # --------------------------------------------------------------
        # endregion correction
        # --------------------------------------------------------------

        return
