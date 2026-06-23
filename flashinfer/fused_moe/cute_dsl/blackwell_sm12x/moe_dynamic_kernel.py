"""
MoEDynamicKernel — queue-driven routed NVFP4 MoE kernel for SM120.

A queue-driven fused control-plane kernel. It keeps the proven FC1 / SiLU /
quant / FC2 / scatter compute body, but replaces a resident-grid route/pack ->
compute barrier with a global ready-task queue.

Execution model
  Phase 0: cooperative init / clear scratch state
  Phase 1: all CTAs start as producers
           - claim routed (token, topk_slot) pairs from pair_head
           - append expert rows
           - write token_map + token_weights
           - quantize each routed token row into expert-major packed A + scales
           - publish one compute task per ready (expert, m_tile, slice_group)
             as soon as a tile is fully written
  Phase 2: CTAs that finish producing become consumers immediately
           - CTA leader pops one ready task into shared ctrl state
           - MMA warps run FC1 -> SiLU -> quant -> FC2 -> scatter for that task
           - DMA warp streams the corresponding FC1 / FC2 weights

This is intentionally conservative:
  - still one CTA per SM
  - the per-slice microkernel runs sequentially for a small grouped slice task
  - still one initial resident-grid barrier after init

Control-plane design
  - no global route/pack -> compute barrier
  - no resident-grid scheduler in the compute steady state
  - route/pack is warp-private instead of CTA-broadcast
  - compute work is driven by a global append-only ready-task queue

It spans the full M-tile set {16,32,64,128} x 128, so a single dynamic kernel
covers the decode and prefill bands.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
    T,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    fabs_f32,
    fmax_f32,
    fp8_e4m3_to_f32,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    quantize_block_fp4_fast,
    get_ptr_as_int64,
    st_global_f32,
    st_global_i32,
    shared_ptr_to_u32,
    st_shared_f32,
    st_shared_u8,
    st_global_u64,
    st_global_v4_u32,
    warp_reduce,
)
from flashinfer.fused_moe.cute_dsl.utils import (
    broadcast_f32_to_half2,
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    cp_async_u64_shared_global,
    e2m1x8_mul_residual_to_e4m3x8,
    e2m1x8_to_e4m3x8,
    ld_shared_u32,
    atomic_add_shared_i32,
    mxfp8_mma_m16n8k32_f32_e4m3,
    quantize_block_fp4,
    quantize_block_fp8_mx,
    st_shared_u32,
)
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel as DenseGemmKernel,
    _reshape_acc_to_mn,
)
from flashinfer.cute_dsl.utils import (
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from flashinfer.cute_dsl.fp4_common import scatter_add_v4_bf16x2
from .moe_activations import (
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)


_SF_VEC_SIZE = 16
_TASK_SLICE_CHUNK = 1
# w4a8 smem staging geometry: one 128-row x 64-byte packed-FP4 B tile per
# k-tile. Rows pad to 80 bytes: 16-aligned (cp.async.cg requires dst
# alignment = copy size) and 20*g mod 32 spreads the eight g-rows a lane
# quad touches across distinct bank groups.
_W4A8_B_ROW_PAD = 80
_W4A8_TMA_TILE_BYTES = 128 * 64  # one (128 n, 128 fp4-k) TMA box
_W4A8_B_BUF_BYTES = 128 * _W4A8_B_ROW_PAD


@cute.jit
def _w4a8_stage_b_tile(
    b_u32: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_u32_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    row_pad: cutlass.Constexpr = _W4A8_B_ROW_PAD,
):
    """cp.async one 128x64B packed-FP4 tile into smem rows.

    Unpadded (64B) rows XOR-swizzle the 16B chunk by the row's low bits —
    without it every consumer B load is a 4-8 way bank conflict."""
    for i in cutlass.range_constexpr((128 * 4 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(128 * 4):
            r = idx >> Int32(2)
            ch = idx & Int32(3)
            gaddr = get_ptr_as_int64(
                b_u32,
                base_idx + Int64(r) * Int64(row_u32_stride) + Int64(ch * 4),
            )
            cp_async4_shared_global(
                smem_base + r * Int32(row_pad) + ch * Int32(16), gaddr
            )


@cute.jit
def _w4a8_stage_a_tile(
    pa_u32: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_u32_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """cp.async one rows x 128B E4M3 A tile into smem rows.

    Chunks XOR-swizzle by the row's low bits: at a 128B row stride every
    lane of an A-fragment load hits one bank otherwise."""
    for i in cutlass.range_constexpr((rows * 8 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(rows * 8):
            r = idx >> Int32(3)
            ch = idx & Int32(7)
            gaddr = get_ptr_as_int64(
                pa_u32,
                base_idx + Int64(r) * Int64(row_u32_stride) + Int64(ch * 4),
            )
            cp_async4_shared_global(
                smem_base + (r << Int32(7)) + (ch << Int32(4)), gaddr
            )


@cute.jit
def _w4a8_stage_bytes4(
    t_u8: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr = 128,
):
    """cp.async 4 contiguous bytes per row (scale grids)."""
    for i in cutlass.range_constexpr((rows + tcnt - 1) // tcnt):
        r = tidx + Int32(i * tcnt)
        if r < Int32(rows):
            gaddr = get_ptr_as_int64(t_u8, base_idx + Int64(r) * Int64(row_stride))
            cp_async_u32_shared_global(smem_base + (r << Int32(2)), gaddr)


@cute.jit
def _w4a8_stage_bytes8(
    t_u8: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """cp.async 8 contiguous bytes per row for 128 rows (residual grids)."""
    for i in cutlass.range_constexpr((128 + tcnt - 1) // tcnt):
        r = tidx + Int32(i * tcnt)
        if r < Int32(128):
            gaddr = get_ptr_as_int64(t_u8, base_idx + Int64(r) * Int64(row_stride))
            cp_async_u64_shared_global(smem_base + (r << Int32(3)), gaddr)


_PRODUCER_PAIRS_PER_WARP = 2
_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0


class DynamicLaunchParams:
    """Minimal runtime launch state shared between host setup and kernel code."""

    def __init__(
        self,
        row_counts: cute.Tensor,
        gate_tile_cnt: Int32,
        *,
        loc=None,
    ):
        self.row_counts = row_counts
        self.gate_tile_cnt = gate_tile_cnt
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.row_counts, self.gate_tile_cnt]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.row_counts, self.gate_tile_cnt],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return DynamicLaunchParams(*(tuple(obj_list)), loc=self._loc)


@dsl_user_op
def _st_shared_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_shared_release_i32(addr, val, *, loc=None, ip=None):
    """CTA-scope release store: pairs with the consumers' acquire spin so
    the producer's prior writes (including async-proxy writes it observed
    via an mbarrier wait or cp.async drain) are visible before the flag."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.release.cta.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_global_u64(addr, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u64 $0, [$1];",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_global_acquire_i32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_global_release_i32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _spin_wait_global_eq_i32(addr, expected, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _spin_wait_shared_ge_i32(addr, value, *, loc=None, ip=None):
    """Spin until the shared i32 at ``addr`` is >= ``value`` (acquire)."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_ge_loop:\n"
        "  ld.acquire.cta.shared.s32 %val, [$0];\n"
        "  setp.lt.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_ge_loop;\n"
        "}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _threadfence(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _atomic_cas_global_i32(addr, compare, value, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class MoEDynamicKernelBackend:
    """Queue-driven first-pass dynamic MoE kernel."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        activation: str = "silu",
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
        deterministic_output: bool = False,
        swap_ab: bool = False,
        quant_recipe: str = "nvfp4",
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        activation = normalize_moe_activation(activation)
        if quant_recipe not in {"nvfp4", "w4a8_mx", "w4a8_nvfp4"}:
            raise ValueError(f"unsupported quant_recipe {quant_recipe!r}")
        if quant_recipe != "nvfp4" and activation == SWIGLUOAI_UNINTERLEAVE:
            raise NotImplementedError(
                "activation='swigluoai_uninterleave' is not supported by W4A8 MoE"
            )
        swiglu_limit = normalize_swiglu_limit_for_activation(activation, swiglu_limit)
        swiglu_alpha = normalize_swiglu_alpha_for_activation(activation, swiglu_alpha)
        swiglu_beta = normalize_swiglu_beta_for_activation(activation, swiglu_beta)
        self._dense_cls = DenseGemmKernel
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.fast_math = fast_math
        self.activation = activation
        self.is_gated = is_gated_moe_activation(activation)
        self.is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        # w4a8 recipes: E4M3 activations (dynamic per-32 UE8M0 block scales)
        # against the same packed-FP4 weight bytes, computed on the FP8
        # m16n8k32 block-scale MMA. "w4a8_nvfp4" additionally applies the
        # per-K/16 residual multipliers from the NVFP4 scale decomposition
        # during nibble expansion. The control plane, epilogue, and scatter
        # are shared with the nvfp4 recipe; operands are read directly from
        # global/shared memory (no A/SF TMA staging) in this bring-up shape.
        self.quant_recipe = quant_recipe
        self.is_w4a8 = quant_recipe != "nvfp4"
        self.w4a8_residual = quant_recipe == "w4a8_nvfp4"
        if self.is_w4a8 and swap_ab:
            raise ValueError("w4a8 recipes do not support swap_ab yet")
        if self.is_w4a8 and share_input_across_experts:
            raise ValueError(
                "w4a8 recipes do not support share_input_across_experts yet"
            )
        if self.is_w4a8 and dynamic_down_scale:
            raise ValueError(
                "w4a8 recipes are self-ranging; dynamic_down_scale does not apply"
            )
        self.dynamic_down_scale = dynamic_down_scale
        self.share_input_across_experts = share_input_across_experts
        self.deterministic_output = bool(deterministic_output)
        # swap_ab runs the gated FC1 with the intermediate (logical N) on the
        # MMA M-role (mirrors dense.py), so a sub-64 tile_n that divides a
        # non-128 per-shard n (e.g. 32 | 352) is legal and the gate-half base
        # rides the sub-128 M-atom SF slice instead of straddling N SF atoms.
        # FC2 stays in the normal orientation, so FC1 builds its own tiled_mma.
        self.swap_ab = bool(swap_ab) and self.is_gated
        # FC1 swap produce-tile width (intermediate cols per swapped MMA tile).
        # 32: for any 32-aligned n the gate-half base n%128 in {0,32,64,96}, so
        # offset+32 <= 128 always fits one 128-row SF atom; and tile_m=32 keeps
        # the base atom_shape (2,2,1)/4-warps, so FC1 and FC2 share warp count.
        self._fc1_int_tile = 32
        tile_k = sf_vec_size * 8
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        # Scale-factor tiles are 128-row atoms in hardware. For sub-128 MMA
        # tiles (e.g. tile_m=64) one SF atom backs several MMA tiles, so the
        # TMA atom + smem are built at max(128, tile) and the kernel offsets
        # into the shared block by `*_tiles_per_block` (mirrors dense.py).
        self.sa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sa_tiles_per_block = self.sa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.occupancy = 1
        # Per-tile atom layout / MMA-warp count (dense's table). Keep dynamic's
        # proven (2,2,1)/4-warp config for the 128 tile; smaller tiles need the
        # matching atom shape so the SF smem layout + V-map are consistent.
        _tm = mma_tiler_mn[0]
        if _tm == 128:
            self.atom_shape = (2, 2, 1)
            self.num_mma_warps = 4
        elif _tm == 16:
            self.atom_shape = (1, 2, 1)
            self.num_mma_warps = 2
        elif _tm == 32:
            self.atom_shape = (2, 2, 1)
            self.num_mma_warps = 4
        else:  # 64
            self.atom_shape = (4, 2, 1)
            self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        # w4a8 small tiles: two producer warps (the staging-issue bandwidth of
        # one warp limits the DRAM-bound bands; epochs split by parity).
        # NOTE: a 2-warp producer split (epochs by parity) was tried and
        # mis-corrupts (~cos 0.98 at m=64); kept dormant pending a root cause.
        self.num_dma_warps = 1
        self.threads_per_cta = (
            self.num_mma_warps + self.num_dma_warps
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.pass_gate_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.pass_final_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFA(self, sfa_tensor, tiled_mma)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        return self._dense_cls._thrfrg_SFB(self, sfb_tensor, tiled_mma)

    def _get_layoutSFA_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFA_TV(self, tiled_mma)

    def _get_layoutSFB_TV(self, tiled_mma):
        return self._dense_cls._get_layoutSFB_TV(self, tiled_mma)

    def _setup_attributes(self):
        import cutlass.utils.blackwell_helpers as sm120_utils

        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_layout = cute.make_layout(self.atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.mma_atom = cute.make_mma_atom(mma_op)
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_m_tiles = self.tile_shape_mnk[0] // (16 * self.atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (8 * self.atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // 64

        if cutlass.const_expr(self.swap_ab):
            # Swapped FC1: intermediate (self._fc1_int_tile wide) rides the MMA
            # M-role, tokens the N-role -- so a sub-64 int tile is legal and the
            # gate-half weight SF rides the sub-128 M-atom slice (dense.py
            # pattern). Same atom_shape -> identical warp count, so the CTA
            # thread/barrier structure is unchanged. FC2 keeps self.tiled_mma
            # (normal orientation over the 128-wide K contraction).
            #
            # The token N-role must be a multiple of the fixed sm120 N
            # permutation atom (8,2,2)=32; a smaller tile (tile_shape_mnk[0] is
            # 16 here) makes the MMA address phantom N-positions and scrambles
            # tokens. Round the token tile up to a 64 multiple (>=64, dense's FP4
            # swap_ab floor) and mask the padding with valid_rows. The 128-row
            # activation atom supplies the extra token slots.
            self._fc1_tok_tile = max(64, ((self.tile_shape_mnk[0] + 63) // 64) * 64)
            self.fc1_tile_shape_mnk = (
                self._fc1_int_tile,
                self._fc1_tok_tile,
                self.tile_shape_mnk[2],
            )
            fc1_perm = sm120_utils.get_permutation_mnk(
                self.fc1_tile_shape_mnk,
                self.sf_vec_size,
                False,
            )
            self.fc1_tiled_mma = cute.make_tiled_mma(
                mma_op,
                atom_layout,
                permutation_mnk=fc1_perm,
            )
            self.fc1_num_m_tiles = self.fc1_tile_shape_mnk[0] // (
                16 * self.atom_shape[0]
            )
            self.fc1_num_n_tiles = self.fc1_tile_shape_mnk[1] // (
                8 * self.atom_shape[1]
            )

        sfa_smem = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        self.ab_stage, self.epi_stage = self._dense_cls._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem,
            sfb_smem,
            self.epi_tile,
            cutlass.BFloat16,
            self.smem_capacity,
            self.occupancy,
        )
        # dense._compute_stages assumes a single B/SFB buffer, but the gated
        # path keeps two (sB+sB_up, sSFB+sSFB_up) plus a third mbar pipeline.
        # For sub-128 MMA tiles its smaller A inflates the suggested stage
        # count past what dynamic's doubled buffers fit, so cap to the real
        # per-stage footprint before rounding to a divisor of 32.
        nb = 2 if self.is_gated else 1
        n_pipe = 3 if self.is_gated else 2
        # sA smem is the 128-row atom for sub-128 tiles, so size from
        # sa_tile_shape_mk (= tile_m at tile_m>=128).
        a_bytes = (
            self.sa_tile_shape_mk[0]
            * self.sa_tile_shape_mk[1]
            * self.a_dtype.width
            // 8
        )
        b_bytes = (
            cute.size(cute.slice_(self.tile_shape_mnk, (0, None, None)))
            * self.b_dtype.width
            // 8
        )
        sfa_bytes = (
            cute.size(cute.filter_zeros(sfa_smem).shape) * self.sf_dtype.width // 8
        )
        sfb_bytes = (
            cute.size(cute.filter_zeros(sfb_smem).shape) * self.sf_dtype.width // 8
        )
        per_stage = a_bytes + nb * b_bytes + sfa_bytes + nb * sfb_bytes + n_pipe * 2 * 8
        fixed = (
            self.tile_shape_mnk[0] * self.tile_shape_mnk[1] * 2  # sC (bf16 epi)
            + 2 * (self.num_mma_warps + 1) * 32 * 4  # route caches
            + self.tile_shape_mnk[0] * 8  # scatter caches
            + 8 * 8
            + 1024
            + 8 * 1024  # ctrl/mbar/align slack
        )
        max_fit = max(1, (self.smem_capacity - fixed) // per_stage)
        self.ab_stage = min(self.ab_stage, max_fit)
        # ab_stage must divide k_tile_cnt (K/tile_K = 4096/128 = 32) evenly;
        # 32%3!=0 causes pipeline phase mismatch. Round down to nearest divisor.
        while self.ab_stage > 1 and 32 % self.ab_stage != 0:
            self.ab_stage -= 1
        if self.is_w4a8:
            # w4a8 repurposes the staging regions as fixed double buffers; a
            # larger ab_stage only inflates smem. Capping at 2 cuts the CTA
            # footprint to ~88KB so two blocks co-reside per SM (the barrier-
            # heavy mainloop needs the extra occupancy to hide latency).
            self.ab_stage = min(self.ab_stage, 2)
            # Staging layout/depth (see the kernel hoist block): python ints
            # must live on self so in-loop const_expr sees them as static.
            # Small tiles: gated FC1 runs FUSED (one k sweep, both gate+up
            # accumulators — 64 hot regs fit at tile_m<=32) with 16KB
            # gate+up B granules at depth 2; non-gated small tiles use
            # depth-4 single-pass staging. Tile 128 keeps two passes/depth 2.
            self.w4a8_small = self.tile_shape_mnk[0] <= 32
            self.w4a8_fused = self.w4a8_small and self.is_gated
            self.w4a8_depth = 2 if self.w4a8_fused else (4 if self.w4a8_small else 2)
            self.w4a8_b_pad = 64 if self.w4a8_small else _W4A8_B_ROW_PAD
            self.w4a8_depth_log2 = self.w4a8_depth.bit_length() - 1
            self.w4a8_fc1_windows_per_slice = (
                1 if self.w4a8_fused else (2 if self.is_gated else 1)
            )
            # B staged through TMA (probe-pinned SW64 layout: read chunk =
            # kb ^ ((n_in>>1)&3)) instead of cp.async; the flag/epoch
            # protocol is unchanged. Small tiles only (the dispatched path).
            self.w4a8_b_tma = self.w4a8_small
            # Fused small tiles also stage FC2 B_down in PAIRS (one 16KB
            # granule = two output tiles), halving the FC2 sync cadence.
            self.w4a8_fc2_pair = self.w4a8_fused
        self.epi_stage = 1
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._dense_cls._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            cutlass.BFloat16,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )
        # The A operand needs the 128-major swizzle atom for sub-128 MMA tiles
        # (the LdMatrix/MMA path); dense builds it at tile_m. Override with the
        # 128-atom layout (see _make_a_smem_layout). Identity when
        # tile_m>=128 (sa_tiles_per_block==1), so the 128 path is untouched.
        if cutlass.const_expr(self.sa_tiles_per_block > 1):
            self.a_smem_layout_staged = self._make_a_smem_layout(self.ab_stage)

        if cutlass.const_expr(self.swap_ab):
            # FC1 SF smem layouts built for the swapped 32-int tile MMA (the SF
            # smem must match the tiled_mma that reads it; the base 128-tile
            # layout does not, which is the 'Unexpected index' root cause). Only
            # the SF layouts are needed (A=weight uses the 128-atom a_smem; B=act
            # uses sA's layout). Same ab_stage/storage bytes as the base.
            (
                self.fc1_a_smem_layout_staged,
                _fc1_b_unused,
                self.fc1_sfa_smem_layout_staged,
                self.fc1_sfb_smem_layout_staged,
                _fc1_epi_unused,
            ) = self._dense_cls._make_smem_layouts(
                self.fc1_tile_shape_mnk,
                (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1]),
                self.a_dtype,
                self.a_layout,
                self.b_dtype,
                self.b_layout,
                self.ab_stage,
                cutlass.BFloat16,
                self.c_layout,
                self.epi_stage,
                self.sf_vec_size,
                self.fc1_tiled_mma,
            )

    def _make_a_smem_layout(self, ab_stage: int):
        import cutlass.utils.hopper_helpers as sm90_utils

        a_is_k_major = self.a_layout.is_k_major_a()
        a_major_mode_size = self.sa_tile_shape_mk[1 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.a_layout,
                self.a_dtype,
                a_major_mode_size,
            ),
            self.a_dtype,
        )
        return cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(self.sa_tile_shape_mk, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

    @cute.jit
    def _gated_activation_value(self, gate: cutlass.Float32, up: cutlass.Float32):
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if gate > limit:
                gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        sigmoid_arg = gate
        up_term = up
        if cutlass.const_expr(self.is_swigluoai):
            sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
            up_term = up + cutlass.Float32(self.swiglu_beta)
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-sigmoid_arg, fastmath=self.fast_math)
        )
        return gate * sigmoid * up_term

    @cute.jit
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        _threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                _st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_ready_tasks(
        self,
        task_tail: cute.Tensor,
        task_ready: cute.Tensor,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        gate_tile_cnt: Int32,
        slice_chunk: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        num_groups = (gate_tile_cnt + slice_chunk - Int32(1)) // slice_chunk
        start = atomic_add_global_i32(get_ptr_as_int64(task_tail, Int32(0)), num_groups)

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            slice_begin = g * slice_chunk
            slice_count = gate_tile_cnt - slice_begin
            if slice_count > slice_chunk:
                slice_count = slice_chunk
            task_expert[slot] = expert_idx
            task_m_tile[slot] = m_tile_idx
            task_slice_begin[slot] = slice_begin
            task_slice_count[slot] = slice_count
            task_valid_rows[slot] = valid_rows
            g += Int32(1)

        _threadfence()

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            _st_global_release_i32(get_ptr_as_int64(task_ready, slot), Int32(1))
            g += Int32(1)

    @cute.jit
    def _publish_deferred_tasks(
        self,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        gate_tile_cnt: Int32,
        slice_chunk: Int32,
        expert_idx: Int32,
        m_tile_idx: Int32,
        valid_rows: Int32,
    ):
        num_groups = (gate_tile_cnt + slice_chunk - Int32(1)) // slice_chunk
        start = m_tile_idx * num_groups

        g = Int32(0)
        while g < num_groups:
            slot = start + g
            slice_begin = g * slice_chunk
            slice_count = gate_tile_cnt - slice_begin
            if slice_count > slice_chunk:
                slice_count = slice_chunk
            task_expert[slot] = expert_idx
            task_m_tile[slot] = m_tile_idx
            task_slice_begin[slot] = slice_begin
            task_slice_count[slot] = slice_count
            task_valid_rows[slot] = valid_rows
            g += Int32(1)

    @cute.jit
    def __call__(
        self,
        a_input: cute.Tensor,  # [num_tokens, K] bf16
        topk_ids: cute.Tensor,  # [num_tokens * topk] int32
        topk_weights: cute.Tensor,  # [num_tokens * topk] float32
        packed_a: cute.Tensor,  # [rows_padded, K, 1] fp4x2 view for compute
        sfa_ptr: cute.Pointer,
        packed_a_storage: cute.Tensor,  # flat uint8 backing packed_a
        scale_storage: cute.Tensor,  # flat uint8 backing sfa_ptr
        barrier_count: cute.Tensor,  # [1] int32 (host-zeroed)
        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)
        pair_head: cute.Tensor,  # [1] int32
        producers_done_count: cute.Tensor,  # [1] int32
        all_work_published: cute.Tensor,  # [1] int32
        task_head: cute.Tensor,  # [1] int32
        task_tail: cute.Tensor,  # [1] int32
        task_ready: cute.Tensor,  # [max_tasks] int32
        task_expert: cute.Tensor,  # [max_tasks] int32
        task_m_tile: cute.Tensor,  # [max_tasks] int32
        task_slice_begin: cute.Tensor,  # [max_tasks] int32
        task_slice_count: cute.Tensor,  # [max_tasks] int32
        task_valid_rows: cute.Tensor,  # [max_tasks] int32
        tile_write_count: cute.Tensor,  # [E * max_m_tiles] int32
        b_w13: cute.Tensor,  # [w1_n, K, E] — gated packs [up, gate], relu2 is single FC1
        sfb_w13_ptr: cute.Pointer,  # scale factors for FC1 weights
        b_down: cute.Tensor,  # [K, I_tp, E]
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,  # expert row histogram [E]
        expert_write_rows: cute.Tensor,  # route/pack write cursors [E]
        expert_tile_base: cute.Tensor,  # compact physical-tile prefix [E + 1]
        input_global_scale: cute.Tensor,  # [E] per-expert FC1 input scale
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,  # [num_tokens, K]
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        *,
        # w4a8 recipe operands (placeholders under the nvfp4 recipe): plain
        # (unswizzled) UE8M0 K/32 weight scale grids and, for w4a8_nvfp4, the
        # per-K/16 E4M3 residual grids from the NVFP4 scale decomposition.
        sfb_w13_mx: cute.Tensor | None = None,  # [w1_n, K//32, E] uint8
        sfb_down_mx: cute.Tensor | None = None,  # [K, I_tp//32, E] uint8
        w13_residual: cute.Tensor | None = None,  # [w1_n, K//16, E] uint8
        down_residual: cute.Tensor | None = None,  # [K, I_tp//16, E] uint8
    ):
        self.a_dtype = packed_a.element_type
        self.b_dtype = b_w13.element_type
        self.sf_dtype = sfa_ptr.dtype
        self.a_layout = utils.LayoutEnum.from_tensor(packed_a)
        self.b_layout = utils.LayoutEnum.from_tensor(b_w13)
        # Dynamic never materializes the intermediate C tensor. Preserve the
        # original row-major epilogue layout without carrying a dead memref.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        self._setup_attributes()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            packed_a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # Single SF tensor for FC1 weights. Gated activation packs [up, gate]
        # along the N dimension; relu2 uses a single FC1 pass.
        sfb_w13_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_w13.shape, self.sf_vec_size
        )
        sfb_w13_tensor = cute.make_tensor(sfb_w13_ptr, sfb_w13_layout)

        # TMA descriptors
        tma_a, gA = self._dense_cls._make_tma_atoms_and_tensors(
            packed_a,
            self.a_smem_layout_staged,
            self.sa_tile_shape_mk,
            1,
        )
        tma_sfa, gSFA = self._dense_cls._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        # Single TMA descriptor over FC1 weights. Gated activation packs
        # [up, gate] across N; relu2 uses a single FC1 slice.
        tma_b_w13, gB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            b_w13,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_w13, gSFB_w13 = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_w13_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        # B_down TMA
        sfb_down_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_down.shape, self.sf_vec_size
        )
        sfb_down_tensor = cute.make_tensor(sfb_down_ptr, sfb_down_layout)
        tma_b_down, gB_down = self._dense_cls._make_tma_atoms_and_tensors(
            b_down,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_sfb_down, gSFB_down = self._dense_cls._make_tma_atoms_and_tensors(
            sfb_down_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.swap_ab):
            # Non-128-aligned n: the scheduler's slice count must CEIL so the
            # partial last intermediate slice is issued. The plain floor below
            # (b_w13.shape[0] // tile_n // 2) drops it -- e.g. n=352 -> 704//128
            # //2 = 2, but the kernel's internal tile count ceils to 3, so the
            # last 96-col slice is never computed (it silently contributes 0).
            n_int = Int32(b_w13.shape[0])
            if self.is_gated:
                n_int = n_int // Int32(2)
            gate_tile_cnt = (n_int + Int32(self.tile_shape_mnk[1]) - Int32(1)) // Int32(
                self.tile_shape_mnk[1]
            )
        else:
            gate_tile_cnt = Int32(b_w13.shape[0]) // Int32(self.tile_shape_mnk[1])
            if self.is_gated:
                gate_tile_cnt = gate_tile_cnt // Int32(2)
        launch_params = DynamicLaunchParams(row_counts, gate_tile_cnt)
        if cutlass.const_expr(self.is_w4a8):
            assert sfb_w13_mx is not None and sfb_down_mx is not None, (
                "w4a8 recipes require sfb_w13_mx and sfb_down_mx"
            )
            assert not self.w4a8_residual or (
                w13_residual is not None and down_residual is not None
            ), "w4a8_nvfp4 requires w13_residual and down_residual"
            assert self.ab_stage * self.sa_tile_shape_mk[0] * self.sa_tile_shape_mk[
                1
            ] // 2 >= (self.tile_shape_mnk[0] * self.tile_shape_mnk[2]), (
                "w4a8 needs ab_stage >= 2 to repurpose the sA staging region"
            )
        if cutlass.const_expr(sfb_w13_mx is None):
            sfb_w13_mx = row_counts  # unused placeholder under nvfp4
        if cutlass.const_expr(sfb_down_mx is None):
            sfb_down_mx = row_counts
        if cutlass.const_expr(w13_residual is None):
            w13_residual = sfb_w13_mx
        if cutlass.const_expr(down_residual is None):
            down_residual = sfb_down_mx
        # Raw u32 views over the packed-FP4 weight bytes for the w4a8 direct
        # global-load path (the TMA-derived gB tensors are coordinate tensors
        # and not byte-addressable).
        b_w13_u32 = cute.recast_tensor(b_w13, cutlass.Uint32)
        b_down_u32 = cute.recast_tensor(b_down, cutlass.Uint32)
        grid = (*self.cluster_shape_mn, max_active_clusters)
        self.kernel(
            a_input,
            topk_ids,
            topk_weights,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            tma_a,
            gA,
            tma_sfa,
            gSFA,
            tma_b_w13,
            gB_w13,
            tma_sfb_w13,
            gSFB_w13,
            tma_b_down,
            gB_down,
            tma_sfb_down,
            gSFB_down,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            # swap_ab FC1 objects (placeholders when off; only used under the
            # const_expr swap branch). cute requires region-local values, so the
            # fc1 tiled_mma + SF layouts are passed as params, not via self.
            self.fc1_tiled_mma if self.swap_ab else self.tiled_mma,
            self.fc1_sfa_smem_layout_staged
            if self.swap_ab
            else self.sfa_smem_layout_staged,
            self.fc1_sfb_smem_layout_staged
            if self.swap_ab
            else self.sfb_smem_layout_staged,
            launch_params,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights,
            sfb_w13_mx,
            sfb_down_mx,
            w13_residual,
            down_residual,
            b_w13_u32,
            b_down_u32,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready: cute.Tensor,
        task_expert: cute.Tensor,
        task_m_tile: cute.Tensor,
        task_slice_begin: cute.Tensor,
        task_slice_count: cute.Tensor,
        task_valid_rows: cute.Tensor,
        tile_write_count: cute.Tensor,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_b_w13: cute.CopyAtom,
        mB_w13: cute.Tensor,
        tma_sfb_w13: cute.CopyAtom,
        mSFB_w13: cute.Tensor,
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        sfa_smem_staged: cute.Layout,
        sfb_smem_staged: cute.Layout,
        epi_smem_staged: cute.ComposedLayout,
        fc1_tiled_mma: cute.TiledMma,
        fc1_sfa_smem_staged: cute.Layout,
        fc1_sfb_smem_staged: cute.Layout,
        launch_params: DynamicLaunchParams,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        sfb_w13_mx: cute.Tensor,
        sfb_down_mx: cute.Tensor,
        w13_residual: cute.Tensor,
        down_residual: cute.Tensor,
        b_w13_u32: cute.Tensor,
        b_down_u32: cute.Tensor,
    ):
        """Kernel entry point."""
        from cutlass.cute.nvgpu.warp.mma import Field as WarpField

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        _, _, gdim_z = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_b_w13)
            cpasync.prefetch_descriptor(tma_sfb_w13)
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cta_layout_mnk.get_flat_coord(cta_rank)

        a_smem_one = cute.slice_(a_smem_staged, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, 0))
        sfa_smem_one = cute.slice_(sfa_smem_staged, (None, None, 0))
        sfb_smem_one = cute.slice_(sfb_smem_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_one)
            + cute.size_in_bytes(self.b_dtype, b_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        )
        phase2_tma_copy_bytes = cute.size_in_bytes(
            self.b_dtype, b_smem_one
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
        # swap_ab with a mid-atom gate base loads a second weight atom (atom-hi)
        # into sB_up/sSFB_up during the gate pass, so the gate mbarrier expects
        # those extra bytes; the up/phase2 pipelines are unchanged.
        ml_tx_count = tma_copy_bytes
        if cutlass.const_expr(
            self.swap_ab and ((mB_w13.shape[0] // 2) % self.tile_shape_mnk[1]) != 0
        ):
            ml_tx_count = (
                tma_copy_bytes
                + cute.size_in_bytes(self.b_dtype, b_smem_one)
                + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)
            )

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            ctrl: cute.struct.MemRange[cutlass.Int32, 8]
            route_phys_rows: cute.struct.MemRange[
                cutlass.Int32, (self.num_mma_warps + 1) * 32
            ]
            route_expert_ids: cute.struct.MemRange[
                cutlass.Int32, (self.num_mma_warps + 1) * 32
            ]
            pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            up_pipeline_array: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            phase2_pipeline_array: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            scatter_tok_cache: cute.struct.MemRange[
                cutlass.Int32, self.tile_shape_mnk[0]
            ]
            scatter_weight_cache: cute.struct.MemRange[
                cutlass.Float32, self.tile_shape_mnk[0]
            ]
            # w4a8 producer-consumer staging flags: [0..1] FC1 ready epoch per
            # buffer, [2..3] FC1 done count per buffer, [4..5] FC2 ready,
            # [6..7] FC2 done.
            w4a8_pipe: cute.struct.MemRange[cutlass.Int32, 8]
            # One mbarrier per staging-buffer parity for w4a8 TMA-B staging
            # (arrive count 1; phases tracked in producer registers).
            w4a8_tma_mbar: cute.struct.MemRange[
                cutlass.Int64, self.w4a8_depth if self.is_w4a8 else 1
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sB_up: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_up: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_smem_staged)],
                self.buffer_align_bytes,
            ]
            reduce_scratch: cute.struct.MemRange[cutlass.Float32, 5]

        storage = smem.allocate(Storage)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        ml_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=ml_tx_count,
            barrier_storage=storage.pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        up_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=tma_copy_bytes,
            barrier_storage=storage.up_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )
        phase2_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=phase2_tma_copy_bytes,
            barrier_storage=storage.phase2_pipeline_array.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cutlass.const_expr(self.is_w4a8 and self.w4a8_b_tma):
            w4a8_tma_mbar_ptr = storage.w4a8_tma_mbar.data_ptr()
            if Int32(tidx) == Int32(0):
                for _mb in cutlass.range_constexpr(self.w4a8_depth):
                    cute.arch.mbarrier_init(w4a8_tma_mbar_ptr + _mb, Int32(1))
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        sA = storage.sA.get_tensor(a_smem_staged.outer, swizzle=a_smem_staged.inner)
        sB = storage.sB.get_tensor(b_smem_staged.outer, swizzle=b_smem_staged.inner)
        sB_up = storage.sB_up.get_tensor(
            b_smem_staged.outer, swizzle=b_smem_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_staged)
        sSFB_up = storage.sSFB_up.get_tensor(sfb_smem_staged)
        sC = storage.sC.get_tensor(
            epi_smem_staged.outer,
            swizzle=epi_smem_staged.inner,
        )
        sfa_base_addr = shared_ptr_to_u32(storage.sSFA.data_ptr())
        reduce_scratch_addr = shared_ptr_to_u32(storage.reduce_scratch.data_ptr())
        ctrl_base_addr = shared_ptr_to_u32(storage.ctrl.data_ptr())
        route_phys_rows_addr = shared_ptr_to_u32(storage.route_phys_rows.data_ptr())
        route_expert_ids_addr = shared_ptr_to_u32(storage.route_expert_ids.data_ptr())
        scatter_tok_base_addr = shared_ptr_to_u32(storage.scatter_tok_cache.data_ptr())
        scatter_weight_base_addr = shared_ptr_to_u32(
            storage.scatter_weight_cache.data_ptr()
        )

        num_tokens = Int32(a_input.shape[0])
        cols = Int32(a_input.shape[1])
        scatter_base = scatter_output.iterator.toint()
        row_counts = launch_params.row_counts
        num_experts = Int32(row_counts.shape[0])
        sf_blocks_per_row = cols // Int32(16)
        if cutlass.const_expr(self.is_w4a8):
            # E4M3 payload: one byte per element; UE8M0 scales per 32 elements.
            output_bytes_per_row = cols
            mx_blocks_per_row = cols // Int32(32)
        else:
            output_bytes_per_row = cols // Int32(2)
            mx_blocks_per_row = sf_blocks_per_row  # unused placeholder
        cols_u32 = cols // Int32(2)
        scatter_output_u32 = cute.recast_tensor(scatter_output, cutlass.Uint32)
        total_pairs = Int32(topk_ids.shape[0])
        num_topk = total_pairs // num_tokens
        flat_tid = Int32(bidz) * Int32(self.threads_per_cta) + Int32(tidx)
        flat_stride = Int32(gdim_z) * Int32(self.threads_per_cta)
        num_k_tiles = (cols + Int32(63)) // Int32(64)
        route_gate_tile_cnt = launch_params.gate_tile_cnt
        task_slice_chunk = Int32(_TASK_SLICE_CHUNK)
        full_tile_publish_enabled = Int32(0)

        # Phase 0: cooperative init — zero routing state, queue state, and output.
        task_capacity = Int32(task_ready.shape[0])
        tile_write_slots = Int32(tile_write_count.shape[0])
        i = flat_tid
        while i < num_experts:
            row_counts[i] = Int32(0)
            expert_write_rows[i] = Int32(0)
            i += flat_stride
        if flat_tid < num_experts + Int32(1):
            expert_tile_base[flat_tid] = Int32(0)

        scatter_rows = Int32(scatter_output.shape[0])
        scatter_total_u32 = scatter_rows * cols_u32
        scatter_vecs = scatter_total_u32 // Int32(4)
        zero_u32 = Uint32(0)
        zv = flat_tid
        while zv < scatter_vecs:
            st_global_v4_u32(
                scatter_base + Int64(zv) * Int64(16),
                zero_u32,
                zero_u32,
                zero_u32,
                zero_u32,
            )
            zv += flat_stride

        j = scatter_vecs * Int32(4) + flat_tid
        while j < scatter_total_u32:
            scatter_output_u32[j // cols_u32, j % cols_u32] = Uint32(0)
            j += flat_stride

        k = flat_tid
        while k < task_capacity:
            task_ready[k] = Int32(0)
            task_expert[k] = Int32(0)
            task_m_tile[k] = Int32(0)
            task_slice_begin[k] = Int32(0)
            task_slice_count[k] = Int32(0)
            task_valid_rows[k] = Int32(0)
            k += flat_stride

        if full_tile_publish_enabled > Int32(0):
            tw = flat_tid
            while tw < tile_write_slots:
                tile_write_count[tw] = Int32(0)
                tw += flat_stride

        if flat_tid == Int32(0):
            pair_head[Int32(0)] = Int32(0)
            producers_done_count[Int32(0)] = Int32(0)
            all_work_published[Int32(0)] = Int32(0)
            task_head[Int32(0)] = Int32(0)
            task_tail[Int32(0)] = Int32(0)

        cute.arch.sync_threads()
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 1: histogram routed rows per expert.
        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            atomic_add_global_i32(get_ptr_as_int64(row_counts, expert_id), Int32(1))
            hist_idx += flat_stride

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if flat_tid == Int32(0):
            tile_acc = Int32(0)
            expert_idx = Int32(0)
            while expert_idx < num_experts:
                expert_tile_base[expert_idx] = tile_acc
                rows = row_counts[expert_idx]
                tile_acc += (rows + Int32(self.tile_shape_mnk[0]) - Int32(1)) // Int32(
                    self.tile_shape_mnk[0]
                )
                expert_idx += Int32(1)
            expert_tile_base[num_experts] = tile_acc

        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        # Phase 2: warp-private route/pack producers into compact physical tiles.
        lane_id = Int32(tidx) & Int32(31)
        num_cta_warps = Int32(self.num_mma_warps + 1)
        producer_batch_pairs = num_cta_warps * Int32(_PRODUCER_PAIRS_PER_WARP)
        shared_input_gs_value = cutlass.Float32(0.0)
        if cutlass.const_expr(self.share_input_across_experts):
            shared_input_gs_value = input_global_scale[Int32(0)].to(cutlass.Float32)
        pair_idx = Int32(0)
        expert_id = Int32(0)
        token_idx = Int32(0)
        weight = cutlass.Float32(0.0)
        row = Int32(0)
        phys_tile = Int32(0)
        phys_row = Int32(0)
        produce_active = Int32(1)
        while produce_active > Int32(0):
            batch_base = Int32(0)
            if is_cta_leader > Int32(0):
                claim_count = producer_batch_pairs
                if cutlass.const_expr(self.share_input_across_experts):
                    claim_count = num_cta_warps
                batch_base = atomic_add_global_i32(
                    get_ptr_as_int64(pair_head, Int32(0)),
                    claim_count,
                )
                _st_shared_i32(ctrl_base_addr + Int32(28), batch_base)
            cute.arch.sync_threads()
            batch_base = _ld_shared_i32(ctrl_base_addr + Int32(28))
            producer_limit = total_pairs
            if cutlass.const_expr(self.share_input_across_experts):
                producer_limit = num_tokens
            if batch_base >= producer_limit:
                produce_active = Int32(0)
            else:
                if cutlass.const_expr(self.share_input_across_experts):
                    token_idx = batch_base + warp_idx
                    if token_idx < num_tokens:
                        route_slot_base = warp_idx * Int32(32)
                        if lane_id == Int32(0):
                            topk_slot = Int32(0)
                            while topk_slot < num_topk:
                                pair_idx = token_idx * num_topk + topk_slot
                                expert_id = topk_ids[pair_idx].to(Int32)
                                weight = topk_weights[pair_idx].to(cutlass.Float32)
                                row = atomic_add_global_i32(
                                    get_ptr_as_int64(expert_write_rows, expert_id),
                                    Int32(1),
                                )
                                phys_tile = expert_tile_base[expert_id] + row // Int32(
                                    self.tile_shape_mnk[0]
                                )
                                phys_row = phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                ) + row % Int32(self.tile_shape_mnk[0])
                                map_value = token_idx
                                if cutlass.const_expr(self.deterministic_output):
                                    map_value = pair_idx
                                st_global_i32(
                                    get_ptr_as_int64(token_map, phys_row), map_value
                                )
                                st_global_f32(
                                    get_ptr_as_int64(token_weights, phys_row), weight
                                )
                                slot = route_slot_base + topk_slot
                                _st_shared_i32(
                                    route_phys_rows_addr + slot * Int32(4), phys_row
                                )
                                _st_shared_i32(
                                    route_expert_ids_addr + slot * Int32(4), expert_id
                                )
                                topk_slot += Int32(1)
                        cute.arch.sync_warp()

                        gs_value = shared_input_gs_value
                        if num_topk == Int32(8):
                            route_output_base = cute.make_rmem_tensor((8,), Int32)
                            route_scale_base = cute.make_rmem_tensor((8,), Int32)
                            for cache_slot in cutlass.range_constexpr(8):
                                slot = route_slot_base + Int32(cache_slot)
                                phys_row = _ld_shared_i32(
                                    route_phys_rows_addr + slot * Int32(4)
                                )
                                route_output_base[cache_slot] = (
                                    phys_row * output_bytes_per_row
                                )
                                # scale_storage uses 128-row SF atoms: index by the
                                # 128-atom (phys_row>>7) + row-within-atom, not the
                                # MMA tile. Identity at tile_m==128.
                                sf_atom = phys_row >> Int32(7)
                                sf_row = phys_row & Int32(127)
                                route_scale_base[cache_slot] = (
                                    sf_atom * num_k_tiles * Int32(32 * 4 * 4)
                                    + (sf_row % Int32(32)) * Int32(4 * 4)
                                    + (sf_row // Int32(32)) * Int32(4)
                                )

                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = cutlass.Float32(
                                        a_input[
                                            token_idx, block_start + Int32(elem_idx)
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.is_gated and self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                k_tile_idx = sf_idx // Int32(4)
                                inner_k_idx = sf_idx % Int32(4)
                                scale_k_base = (
                                    k_tile_idx * Int32(32 * 4 * 4) + inner_k_idx
                                )
                                for cache_slot in cutlass.range_constexpr(8):
                                    output_offset = route_output_base[
                                        cache_slot
                                    ] + sf_idx * Int32(8)
                                    st_global_u64(
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )
                                    scale_storage[
                                        route_scale_base[cache_slot] + scale_k_base
                                    ] = scale_byte
                                sf_idx += Int32(32)
                        else:
                            sf_idx = lane_id
                            while sf_idx < sf_blocks_per_row:
                                block_start = sf_idx * Int32(16)
                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = cutlass.Float32(
                                        a_input[
                                            token_idx, block_start + Int32(elem_idx)
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.is_gated and self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, gs_value
                                    )

                                topk_slot = Int32(0)
                                while topk_slot < num_topk:
                                    slot = route_slot_base + topk_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + slot * Int32(4)
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row
                                        + sf_idx * Int32(8)
                                    )
                                    st_global_u64(
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )

                                    # scale_storage uses 128-row SF atoms
                                    # (tile_atom_to_shape_SF); index by the 128-atom
                                    # (phys_row>>7) + row-within-atom, NOT the MMA
                                    # tile (which may be 64). Identity at tile_m==128.
                                    k_tile_idx = sf_idx // Int32(4)
                                    sf_atom = phys_row >> Int32(7)
                                    sf_row = phys_row & Int32(127)
                                    outer_m_idx = sf_row % Int32(32)
                                    inner_m_idx = sf_row // Int32(32)
                                    inner_k_idx = sf_idx % Int32(4)
                                    scale_offset = (
                                        sf_atom * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + outer_m_idx * Int32(4 * 4)
                                        + inner_m_idx * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    topk_slot += Int32(1)
                                sf_idx += Int32(32)

                        if full_tile_publish_enabled > Int32(0):
                            cute.arch.sync_warp()
                            _threadfence()
                            cute.arch.sync_warp()

                            if lane_id == Int32(0):
                                topk_slot = Int32(0)
                                while topk_slot < num_topk:
                                    slot = route_slot_base + topk_slot
                                    phys_row = _ld_shared_i32(
                                        route_phys_rows_addr + slot * Int32(4)
                                    )
                                    expert_id = _ld_shared_i32(
                                        route_expert_ids_addr + slot * Int32(4)
                                    )
                                    phys_tile = phys_row // Int32(
                                        self.tile_shape_mnk[0]
                                    )
                                    completed = atomic_add_global_i32(
                                        get_ptr_as_int64(tile_write_count, phys_tile),
                                        Int32(1),
                                    ) + Int32(1)
                                    if completed == Int32(self.tile_shape_mnk[0]):
                                        self._publish_ready_tasks(
                                            task_tail,
                                            task_ready,
                                            task_expert,
                                            task_m_tile,
                                            task_slice_begin,
                                            task_slice_count,
                                            task_valid_rows,
                                            route_gate_tile_cnt,
                                            task_slice_chunk,
                                            expert_id,
                                            phys_tile,
                                            Int32(self.tile_shape_mnk[0]),
                                        )
                                    topk_slot += Int32(1)
                else:
                    warp_item = Int32(0)
                    while warp_item < Int32(_PRODUCER_PAIRS_PER_WARP):
                        pair_idx = batch_base + warp_idx + warp_item * num_cta_warps
                        expert_id = Int32(0)
                        token_idx = Int32(0)
                        weight = cutlass.Float32(0.0)
                        row = Int32(0)
                        phys_tile = Int32(0)
                        if pair_idx < total_pairs:
                            expert_id = topk_ids[pair_idx].to(Int32)
                            token_idx = pair_idx // num_topk
                            weight = topk_weights[pair_idx].to(cutlass.Float32)

                            if lane_id == Int32(0):
                                row = atomic_add_global_i32(
                                    get_ptr_as_int64(expert_write_rows, expert_id),
                                    Int32(1),
                                )
                                phys_tile = expert_tile_base[expert_id] + row // Int32(
                                    self.tile_shape_mnk[0]
                                )
                                phys_row = phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                ) + row % Int32(self.tile_shape_mnk[0])
                                map_value = token_idx
                                if cutlass.const_expr(self.deterministic_output):
                                    map_value = pair_idx
                                st_global_i32(
                                    get_ptr_as_int64(token_map, phys_row), map_value
                                )
                                st_global_f32(
                                    get_ptr_as_int64(token_weights, phys_row), weight
                                )

                            row = cute.arch.shuffle_sync(row, Int32(0))
                            phys_tile = cute.arch.shuffle_sync(phys_tile, Int32(0))
                            expert_id = cute.arch.shuffle_sync(expert_id, Int32(0))
                            token_idx = cute.arch.shuffle_sync(token_idx, Int32(0))

                            gs_value = input_global_scale[expert_id].to(cutlass.Float32)
                            if cutlass.const_expr(self.is_w4a8):
                                # w4a8: per-32 dynamic UE8M0 + E4M3 payload, no
                                # global scale. Payload stored plain row-major;
                                # scales stored plain [row, cols//32].
                                phys_row = phys_tile * Int32(
                                    self.tile_shape_mnk[0]
                                ) + row % Int32(self.tile_shape_mnk[0])
                                blk_idx = lane_id
                                while blk_idx < mx_blocks_per_row:
                                    block_start = blk_idx * Int32(32)
                                    values = cute.make_rmem_tensor(
                                        (32,), cutlass.Float32
                                    )
                                    block_max = cutlass.Float32(0.0)
                                    for elem_idx in cutlass.range_constexpr(32):
                                        value = cutlass.Float32(
                                            a_input[
                                                token_idx, block_start + Int32(elem_idx)
                                            ]
                                        )
                                        values[elem_idx] = value
                                        block_max = fmax_f32(block_max, fabs_f32(value))
                                    payload, mx_scale_byte = quantize_block_fp8_mx(
                                        values, block_max
                                    )
                                    output_offset = (
                                        phys_row * output_bytes_per_row + block_start
                                    )
                                    for pair_idx in cutlass.range_constexpr(4):
                                        packed64 = (
                                            Uint64(payload[pair_idx * 2 + 1])
                                            << Uint64(32)
                                        ) | Uint64(payload[pair_idx * 2])
                                        st_global_u64(
                                            get_ptr_as_int64(
                                                packed_a_storage,
                                                output_offset + Int32(pair_idx * 8),
                                            ),
                                            packed64,
                                        )
                                    scale_storage[
                                        phys_row * mx_blocks_per_row + blk_idx
                                    ] = Uint8(mx_scale_byte & Uint32(0xFF))
                                    blk_idx += Int32(32)
                            else:
                                sf_idx = lane_id
                                while sf_idx < sf_blocks_per_row:
                                    block_start = sf_idx * Int32(16)
                                    values = cute.make_rmem_tensor(
                                        (16,), cutlass.Float32
                                    )
                                    block_max = cutlass.Float32(0.0)
                                    for elem_idx in cutlass.range_constexpr(16):
                                        value = cutlass.Float32(
                                            a_input[
                                                token_idx, block_start + Int32(elem_idx)
                                            ]
                                        )
                                        values[elem_idx] = value
                                        block_max = fmax_f32(block_max, fabs_f32(value))
                                    packed64 = Uint64(0)
                                    scale_byte = Uint8(0)
                                    if self.is_gated and self.fast_math:
                                        packed64, scale_byte = quantize_block_fp4_fast(
                                            values, block_max, gs_value
                                        )
                                    else:
                                        packed64, scale_byte = quantize_block_fp4(
                                            values, block_max, gs_value
                                        )

                                    output_offset = (
                                        phys_tile * Int32(self.tile_shape_mnk[0])
                                        + row % Int32(self.tile_shape_mnk[0])
                                    ) * output_bytes_per_row + sf_idx * Int32(8)
                                    st_global_u64(
                                        get_ptr_as_int64(
                                            packed_a_storage, output_offset
                                        ),
                                        packed64,
                                    )

                                    k_tile_idx = sf_idx // Int32(4)
                                    inner_k_idx = sf_idx % Int32(4)
                                    # scale_storage uses 128-row SF atoms: index by the
                                    # 128-atom + row-within-atom, not the MMA tile.
                                    # Identity at tile_m==128.
                                    phys_row = phys_tile * Int32(
                                        self.tile_shape_mnk[0]
                                    ) + row % Int32(self.tile_shape_mnk[0])
                                    sf_atom = phys_row >> Int32(7)
                                    sf_row = phys_row & Int32(127)
                                    scale_offset = (
                                        sf_atom * num_k_tiles * Int32(32 * 4 * 4)
                                        + k_tile_idx * Int32(32 * 4 * 4)
                                        + (sf_row % Int32(32)) * Int32(4 * 4)
                                        + (sf_row // Int32(32)) * Int32(4)
                                        + inner_k_idx
                                    )
                                    scale_storage[scale_offset] = scale_byte
                                    sf_idx += Int32(32)

                            if full_tile_publish_enabled > Int32(0):
                                cute.arch.sync_warp()
                                # When the whole launch has fewer than one M-tile of routed
                                # rows, only the final partial-tile flush can publish work.
                                # Skip the per-row fence/counter path in that common micro case.
                                _threadfence()
                                cute.arch.sync_warp()

                                if lane_id == Int32(0):
                                    completed = atomic_add_global_i32(
                                        get_ptr_as_int64(tile_write_count, phys_tile),
                                        Int32(1),
                                    ) + Int32(1)
                                    if completed == Int32(self.tile_shape_mnk[0]):
                                        self._publish_ready_tasks(
                                            task_tail,
                                            task_ready,
                                            task_expert,
                                            task_m_tile,
                                            task_slice_begin,
                                            task_slice_count,
                                            task_valid_rows,
                                            route_gate_tile_cnt,
                                            task_slice_chunk,
                                            expert_id,
                                            phys_tile,
                                            Int32(self.tile_shape_mnk[0]),
                                        )
                        warp_item += Int32(1)

        cute.arch.sync_threads()
        # Conservative publish fence before the last-producer CTA flushes any
        # partial tiles. All producer threads in the CTA must have ordered
        # their global writes before lane 0 can publish work.
        _threadfence()
        cute.arch.sync_threads()

        if full_tile_publish_enabled == Int32(0):
            # Micro batches cannot fill a full M tile, so overlap is impossible.
            # Rendezvous once, publish the final partial tiles, then consume.
            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

            if is_cta_leader > Int32(0):
                expert_flush = Int32(bidz)
                while expert_flush < num_experts:
                    rows_remaining = row_counts[expert_flush]
                    m_tile_offset = Int32(0)
                    while rows_remaining > Int32(0):
                        valid_rows = rows_remaining
                        if valid_rows > Int32(self.tile_shape_mnk[0]):
                            valid_rows = Int32(self.tile_shape_mnk[0])
                        self._publish_deferred_tasks(
                            task_expert,
                            task_m_tile,
                            task_slice_begin,
                            task_slice_count,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            expert_tile_base[expert_flush] + m_tile_offset,
                            valid_rows,
                        )
                        rows_remaining -= Int32(self.tile_shape_mnk[0])
                        m_tile_offset += Int32(1)
                    expert_flush += Int32(gdim_z)

            if flat_tid == Int32(0):
                num_groups = (
                    route_gate_tile_cnt + task_slice_chunk - Int32(1)
                ) // task_slice_chunk
                st_global_i32(
                    get_ptr_as_int64(task_tail, Int32(0)),
                    expert_tile_base[num_experts] * num_groups,
                )

            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )
            if flat_tid == Int32(0):
                _st_global_release_i32(
                    get_ptr_as_int64(all_work_published, Int32(0)),
                    Int32(1),
                )
        elif is_cta_leader > Int32(0):
            prev_done = atomic_add_global_i32(
                get_ptr_as_int64(producers_done_count, Int32(0)),
                Int32(1),
            )
            if prev_done == Int32(gdim_z) - Int32(1):
                expert_flush = Int32(0)
                while expert_flush < num_experts:
                    rows = row_counts[expert_flush]
                    rem = rows % Int32(self.tile_shape_mnk[0])
                    if rem != Int32(0):
                        partial_m_tile = expert_tile_base[expert_flush] + rows // Int32(
                            self.tile_shape_mnk[0]
                        )
                        self._publish_ready_tasks(
                            task_tail,
                            task_ready,
                            task_expert,
                            task_m_tile,
                            task_slice_begin,
                            task_slice_count,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            expert_flush,
                            partial_m_tile,
                            rem,
                        )
                    expert_flush += Int32(1)
                _threadfence()
                _st_global_release_i32(
                    get_ptr_as_int64(all_work_published, Int32(0)),
                    Int32(1),
                )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
        # Single tiled view over concatenated w13 [2*I_tp, K, E].
        # W13 is packed as [up, gate] across the concatenated N dimension.
        # Up tiles: N-indices 0..gate_tile_cnt-1
        # Gate tiles: N-indices gate_tile_cnt..2*gate_tile_cnt-1
        gB_w13_tiled = cute.local_tile(
            mB_w13,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # SF tiles use the 128-row atom shape; for sub-128 MMA tiles one SF
        # block backs `sfa_tiles_per_block` MMA tiles (offset applied below).
        gSFA = cute.local_tile(mSFA, self.sfa_tile_shape_mk, (None, None, None))
        gSFB_w13_tiled = cute.local_tile(
            mSFB_w13, self.sfb_tile_shape_nk, (None, None, None)
        )
        # swap_ab gate feed: the gate half of w13 starts at row n, which for a
        # 32-aligned non-128-aligned n is mid-SF-atom (n % 128 in {32,64,96}).
        # A 128-row SF atom can only be TMA'd atom-aligned, so the gate's 128-int
        # slice is sourced from TWO adjacent 128-row atoms: atom-lo (the atom
        # holding row n, at within-atom 32-sub gate_lo_sub) and atom-hi (the
        # next). atom-lo loads into sB/sSFB (the existing gate buffers); atom-hi
        # loads into sB_up/sSFB_up (free during the gate pass; the up pass only
        # overwrites them after pass_gate_barrier). The consumer picks each
        # 32-int sub from lo/hi at the within-atom offset (const_expr below).
        gate_lo_off = (mB_w13.shape[0] // 2) // self.tile_shape_mnk[1]
        gate_lo_sub = (
            (mB_w13.shape[0] // 2) % self.tile_shape_mnk[1]
        ) // self._fc1_int_tile
        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord[1]
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord[0]

        tAsA, tAgA = cpasync.tma_partition(
            tma_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # Single w13 TMA partition (gate+up concatenated)
        tBsB_w13, tBgB_w13 = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsB_w13_up, _ = cpasync.tma_partition(
            tma_b_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_up, 0, 2),
            cute.group_modes(gB_w13_tiled, 0, 2),
        )
        tBsSFB_w13, tBgSFB_w13 = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsSFB_w13_up, _ = cpasync.tma_partition(
            tma_sfb_w13,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB_up, 0, 2),
            cute.group_modes(gSFB_w13_tiled, 0, 2),
        )
        tBsB_w13_up = cute.filter_zeros(tBsB_w13_up)
        tBsSFB_w13 = cute.filter_zeros(tBsSFB_w13)
        tBgSFB_w13 = cute.filter_zeros(tBgSFB_w13)
        tBsSFB_w13_up = cute.filter_zeros(tBsSFB_w13_up)

        # B_down TMA partitions
        gB_down = cute.local_tile(
            mB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_down = cute.local_tile(
            mSFB_down,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        tBsB_down, tBgB_down = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsB_down_up, _ = cpasync.tma_partition(
            tma_b_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_up, 0, 2),
            cute.group_modes(gB_down, 0, 2),
        )
        tBsSFB_down, tBgSFB_down = cpasync.tma_partition(
            tma_sfb_down,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_down, 0, 2),
        )
        tBsSFB_down = cute.filter_zeros(tBsSFB_down)
        tBgSFB_down = cute.filter_zeros(tBgSFB_down)

        # MMA fragment partitions
        # sA is the 128-major atom for sub-128 tiles; slice to the tile_m sub-tile
        # the V-map expects (offset applied per-task at consumption). Identity at
        # tile_m>=128.
        if cutlass.const_expr(self.sa_tiles_per_block > 1):
            sA_part = cute.local_tile(
                sA,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (Int32(0), 0, None),
            )
        else:
            sA_part = sA
        tCsA = thr_mma.partition_A(sA_part)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        # sSFA holds a full 128-row SF atom; for sub-128 MMA tiles slice it to the
        # tile_m-row sub-tile the V-map expects. Identity when
        # tile_m>=128 (sfa_tiles_per_block==1) so the 128 path is byte-identical.
        if cutlass.const_expr(self.sfa_tiles_per_block > 1):
            sSFA_part = cute.local_tile(
                sSFA,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (Int32(0), 0, None),
            )
        else:
            sSFA_part = sSFA
        tCrSFA = self._dense_cls._partition_fragment_SFA(
            self,  # type: ignore[arg-type]
            sSFA_part[None, None, 0],
            thr_mma,
            tidx,
        )
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFB = self._dense_cls._partition_fragment_SFB(
            self,  # type: ignore[arg-type]
            sSFB[None, None, 0],
            thr_mma,
            tidx,
        )

        tCsC_for_shape = thr_mma.partition_C(sC[None, None, 0])
        epi_m_scale = self.tile_shape_mnk[0] // self.epi_tile[0]
        sub_shape = tCsC_for_shape.shape[:3]
        acc_shape = (sub_shape[0], sub_shape[1] * epi_m_scale, sub_shape[2])
        gate_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        up_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA, mode=[3])
        fc1_k_tile_cnt = k_tile_cnt
        # Gated FC1 packs [up, gate] across N; relu2 has a single FC1 pass.
        intermediate_tile_cnt = cute.size(gB_w13_tiled, mode=[2])
        gate_tile_cnt = intermediate_tile_cnt
        if self.is_gated:
            gate_tile_cnt = intermediate_tile_cnt // Int32(2)
        output_tile_cnt = cute.size(gB_down, mode=[2])

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        up_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        up_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        phase2_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        phase2_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        num_k_blocks = cute.size(tCrA, mode=[2])

        atom_ld_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
            self.a_dtype,
        )
        atom_ld_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
            self.b_dtype,
        )
        smem_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
        smem_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
        atom_ld_SF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        smem_copy_SFA = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFA_TV(self, tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        smem_copy_SFB = cute.make_tiled_copy(
            atom_ld_SF,
            self._dense_cls._get_layoutSFB_TV(self, tiled_mma),  # type: ignore[arg-type]
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )

        thr_ld_A = smem_copy_A.get_slice(tidx)
        thr_ld_B = smem_copy_B.get_slice(tidx)
        csA = thr_ld_A.partition_S(sA_part)
        crA = thr_ld_A.retile(tCrA)
        csB = thr_ld_B.partition_S(sB)
        csB_up = thr_ld_B.partition_S(sB_up)
        crB = thr_ld_B.retile(tCrB)

        thr_ld_SFA = smem_copy_SFA.get_slice(tidx)
        thr_ld_SFB = smem_copy_SFB.get_slice(tidx)
        csSFA = thr_ld_SFA.partition_S(sSFA_part)
        crSFA = thr_ld_SFA.retile(tCrSFA)
        csSFB = thr_ld_SFB.partition_S(sSFB)
        csSFB_up = thr_ld_SFB.partition_S(sSFB_up)
        crSFB = thr_ld_SFB.retile(tCrSFB)

        if cutlass.const_expr(self.swap_ab):
            # Swapped FC1 (dense.py swap_ab pattern): the gate/up weight (sB /
            # sB_up) feeds MMA-A on the 32-wide intermediate M-role; the
            # activation (sA) feeds MMA-B on the token N-role. Fragment/copy
            # templates are built at intermediate sub-tile 0 of the shared
            # 128-row weight smem; the per-sub-tile 32-col slice + SF offset are
            # applied inside the FC1 loop. FC2 keeps the unswapped fragments
            # above. num_mma_warps matches, so warp partitioning is unchanged.
            thr_mma_fc1 = fc1_tiled_mma.get_slice(tidx)
            # Weight DATA: same fp4 dtype/atom as the activation, so re-view sB
            # through the activation's 128-row atom layout (sub-32 sliceable).
            sB_fc1 = storage.sB.get_tensor(
                a_smem_staged.outer, swizzle=a_smem_staged.inner
            )
            # SF smem must match the tiled_mma that reads it. Under swap the
            # weight SF takes the SFA role (FC1 SFA layout, 32-int) and the
            # activation SF the SFB role (FC1 SFB layout, 128-token) -- using the
            # base 128-tile SF layout here is the 'Unexpected index' root cause.
            sSFB_fc1 = storage.sSFB.get_tensor(fc1_sfa_smem_staged)
            # Activation SF keeps its ORIGINAL smem layout (as the producer wrote
            # it); only the SFB thread-partition (swapped mma) re-roles it. Re-
            # viewing it through fc1_sfb scrambles the token axis (dense partitions
            # sSFA directly under swap_ab, it does not relayout it).
            sSFA_fc1 = storage.sSFA.get_tensor(sfa_smem_staged)
            sB_up_fc1 = storage.sB_up.get_tensor(
                a_smem_staged.outer, swizzle=a_smem_staged.inner
            )
            sSFB_up_fc1 = storage.sSFB_up.get_tensor(fc1_sfa_smem_staged)
            sB_sub0 = cute.local_tile(
                sB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (None, 0, None)),
                (Int32(0), 0, None),
            )
            sSFB_sub0 = cute.local_tile(
                sSFB_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (None, 0, None)),
                (Int32(0), 0, None),
            )
            # Activation B-operand rides the padded token N-role; slice the
            # 128-row activation atom to fc1_tok_tile tokens (offset 0 -- the one
            # valid m-tile sits at the atom start, padding masked by valid_rows).
            sA_part_sw = cute.local_tile(
                sA,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (Int32(0), 0, None),
            )
            # MMA-A = weight sub-tile (data sB / SF sSFB->SFA);
            # MMA-B = activation (data sA_part_sw / SF sSFA->SFB).
            tCsA_sw = thr_mma_fc1.partition_A(sB_sub0)
            tCrA_sw = fc1_tiled_mma.make_fragment_A(tCsA_sw[None, None, None, 0])
            tCsB_sw = thr_mma_fc1.partition_B(sA_part_sw)
            tCrB_sw = fc1_tiled_mma.make_fragment_B(tCsB_sw[None, None, None, 0])
            tCrSFA_sw = self._dense_cls._partition_fragment_SFA(
                self,  # type: ignore[arg-type]
                sSFB_sub0[None, None, 0],
                thr_mma_fc1,
                tidx,
            )
            # Activation SF rides the same fc1_tok_tile-wide N-slice as the
            # activation data (sA_part_sw); slice the atom SF to match so the
            # SFB fragment and the per-task copy (which re-slices at the task's
            # token tile) agree in size.
            _sSFA_fc1_n = cute.local_tile(
                sSFA_fc1,
                cute.slice_(self.fc1_tile_shape_mnk, (0, None, None)),
                (Int32(0), 0, None),
            )
            tCrSFB_sw = self._dense_cls._partition_fragment_SFB(
                self,  # type: ignore[arg-type]
                _sSFA_fc1_n[None, None, 0],
                thr_mma_fc1,
                tidx,
            )
            # swapped acc: [int-32 M-role, token N-role]
            tCgC_sw = thr_mma_fc1.partition_C(
                cute.make_identity_tensor(
                    (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
                )
            )
            gate_acc_sw = cute.make_rmem_tensor(tCgC_sw.shape[:3], self.acc_dtype)
            up_acc_sw = cute.make_rmem_tensor(tCgC_sw.shape[:3], self.acc_dtype)

            # smem->rmem copy atoms for the swapped FC1 (A=weight uses b_dtype/
            # b_layout, B=activation uses a_dtype/a_layout; SF TV-layouts from
            # fc1_tiled_mma). Mirrors dense.py swap_ab (1118-1180).
            atom_ld_A_sw = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            atom_ld_B_sw = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_copy_A_sw = cute.make_tiled_copy_A(atom_ld_A_sw, fc1_tiled_mma)
            smem_copy_B_sw = cute.make_tiled_copy_B(atom_ld_B_sw, fc1_tiled_mma)
            smem_copy_SFA_sw = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFA_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(fc1_tiled_mma.permutation_mnk[0]),
                    cute.size(fc1_tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_copy_SFB_sw = cute.make_tiled_copy(
                atom_ld_SF,
                self._dense_cls._get_layoutSFB_TV(self, fc1_tiled_mma),  # type: ignore[arg-type]
                (
                    cute.size(fc1_tiled_mma.permutation_mnk[1]),
                    cute.size(fc1_tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_ld_A_sw = smem_copy_A_sw.get_slice(tidx)
            thr_ld_B_sw = smem_copy_B_sw.get_slice(tidx)
            thr_ld_SFA_sw = smem_copy_SFA_sw.get_slice(tidx)
            thr_ld_SFB_sw = smem_copy_SFB_sw.get_slice(tidx)
            crA_sw = thr_ld_A_sw.retile(tCrA_sw)
            crB_sw = thr_ld_B_sw.retile(tCrB_sw)
            crSFA_sw = thr_ld_SFA_sw.retile(tCrSFA_sw)
            crSFB_sw = thr_ld_SFB_sw.retile(tCrSFB_sw)

        # ===================================================================
        # Per-warp setup for the consumer steady state
        # ===================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ===================================================================
        # Consumer steady state: pop one ready task per CTA, then let
        # the MMA warps and DMA warp cooperate on that task.
        # ===================================================================
        if cutlass.const_expr(self.is_w4a8):
            # w4a8 operand views and smem addresses, hoisted out of the
            # consumer loop (loop state must be purely dynamic expressions).
            lane_c = Int32(tidx) & Int32(3)
            lane_g = (Int32(tidx) & Int32(31)) >> Int32(2)
            tCgC_id = thr_mma.partition_C(
                cute.make_identity_tensor(
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
                )
            )
            n_k32_per_tile = self.tile_shape_mnk[2] // 32
            a_u32_per_row = cols // Int32(4)
            a_mx_per_row = cols // Int32(32)
            pa_u32 = cute.recast_tensor(packed_a_storage, cutlass.Uint32)
            sa_flat_addr = shared_ptr_to_u32(storage.sA.data_ptr())
            # Double-buffered staging regions (repurposed; TMA is off under
            # w4a8): B tiles in sB/sB_up, SFB byte rows in sSFB, residual
            # byte rows in sSFB_up, A-scale byte rows in sSFA.
            w4a8_sb0 = shared_ptr_to_u32(storage.sB.data_ptr())
            # FC1 A tiles double-buffer through the sC region (32KB, provably
            # free during the FC1 mainloop: the epilogue writes it afterwards).
            # Staging depth: 4 buffers for small tiles (the DRAM-bound bands
            # need memory-level parallelism, and capacity allows it); 2 for
            # tile 128. Depth-4 layout: B unpadded 64B rows, two bufs per sB
            # region; A bufs in sA past the FC2-intermediate bytes; FC1
            # residual bufs in sC (dead until the epilogue rendezvous); FC2
            # residual bufs alias the (FC1-only) A-buf region.
            if cutlass.const_expr(self.w4a8_small):
                w4a8_sa0 = shared_ptr_to_u32(storage.sA.data_ptr()) + Int32(4096)
                w4a8_res0 = shared_ptr_to_u32(storage.sC.data_ptr())
                w4a8_res2 = w4a8_sa0
            else:
                w4a8_sa0 = shared_ptr_to_u32(storage.sC.data_ptr())
                w4a8_res0 = shared_ptr_to_u32(storage.sSFB_up.data_ptr())
                w4a8_res2 = shared_ptr_to_u32(storage.sSFB_up.data_ptr())
            w4a8_pipe_addr = shared_ptr_to_u32(storage.w4a8_pipe.data_ptr())
            w4a8_sb1 = shared_ptr_to_u32(storage.sB_up.data_ptr())
            w4a8_sfbb = shared_ptr_to_u32(storage.sSFB.data_ptr())
            w4a8_resb = shared_ptr_to_u32(storage.sSFB_up.data_ptr())

        # w4a8 TMA-B: per-mbarrier phase bits, owned by the producer warp.
        # Persist across tasks (mbarriers are init'd once; epochs reset per
        # task but phases must not).
        w4a8_mbar_phase = Int32(0)

        consumer_live = Int32(1)
        while consumer_live > Int32(0):
            if is_cta_leader > Int32(0):
                _st_shared_i32(ctrl_base_addr + Int32(0), Int32(0))  # has_task
                _st_shared_i32(ctrl_base_addr + Int32(4), Int32(0))  # done
                _st_shared_i32(ctrl_base_addr + Int32(28), Int32(0))  # claimed slot
                if full_tile_publish_enabled == Int32(0):
                    tail = _ld_global_acquire_i32(get_ptr_as_int64(task_tail, Int32(0)))
                    slot = atomic_add_global_i32(
                        get_ptr_as_int64(task_head, Int32(0)),
                        Int32(1),
                    )
                    if slot < tail:
                        _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                        _st_shared_i32(ctrl_base_addr + Int32(28), slot)
                        _st_shared_i32(ctrl_base_addr + Int32(8), task_expert[slot])
                        _st_shared_i32(ctrl_base_addr + Int32(12), task_m_tile[slot])
                        _st_shared_i32(
                            ctrl_base_addr + Int32(16), task_slice_begin[slot]
                        )
                        _st_shared_i32(
                            ctrl_base_addr + Int32(20), task_slice_count[slot]
                        )
                        _st_shared_i32(
                            ctrl_base_addr + Int32(24), task_valid_rows[slot]
                        )
                    else:
                        _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
                else:
                    head = _ld_global_acquire_i32(get_ptr_as_int64(task_head, Int32(0)))
                    tail = _ld_global_acquire_i32(get_ptr_as_int64(task_tail, Int32(0)))
                    if head < tail:
                        prev_head = _atomic_cas_global_i32(
                            get_ptr_as_int64(task_head, Int32(0)),
                            head,
                            head + Int32(1),
                        )
                        if prev_head == head:
                            _spin_wait_global_eq_i32(
                                get_ptr_as_int64(task_ready, head), Int32(0)
                            )
                            _st_shared_i32(ctrl_base_addr + Int32(0), Int32(1))
                            _st_shared_i32(ctrl_base_addr + Int32(28), head)
                            _st_shared_i32(ctrl_base_addr + Int32(8), task_expert[head])
                            _st_shared_i32(
                                ctrl_base_addr + Int32(12), task_m_tile[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(16), task_slice_begin[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(20), task_slice_count[head]
                            )
                            _st_shared_i32(
                                ctrl_base_addr + Int32(24), task_valid_rows[head]
                            )
                    else:
                        if _ld_global_acquire_i32(
                            get_ptr_as_int64(all_work_published, Int32(0))
                        ) > Int32(0):
                            _st_shared_i32(ctrl_base_addr + Int32(4), Int32(1))
            cute.arch.sync_threads()

            has_task = _ld_shared_i32(ctrl_base_addr + Int32(0))
            is_done = _ld_shared_i32(ctrl_base_addr + Int32(4))
            if has_task > Int32(0) and full_tile_publish_enabled > Int32(0):
                claimed_slot = _ld_shared_i32(ctrl_base_addr + Int32(28))
                _ld_global_acquire_i32(get_ptr_as_int64(task_ready, claimed_slot))
            if has_task > Int32(0):
                task_m_tile_idx_cache = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_valid_rows_cache = _ld_shared_i32(ctrl_base_addr + Int32(24))
                tile_m_base_cache = task_m_tile_idx_cache * Int32(
                    self.tile_shape_mnk[0]
                )
                cache_row = Int32(tidx)
                while cache_row < Int32(self.tile_shape_mnk[0]):
                    tok = Int32(0)
                    wv = cutlass.Float32(0.0)
                    if cache_row < task_valid_rows_cache:
                        global_row_cache = tile_m_base_cache + cache_row
                        tok = token_map[global_row_cache].to(Int32)
                        wv = token_weights[global_row_cache].to(cutlass.Float32)
                    _st_shared_i32(scatter_tok_base_addr + cache_row * Int32(4), tok)
                    st_shared_f32(scatter_weight_base_addr + cache_row * Int32(4), wv)
                    cache_row += Int32(self.threads_per_cta)
                if cutlass.const_expr(self.is_w4a8):
                    if Int32(tidx) < Int32(8):
                        _st_shared_i32(
                            w4a8_pipe_addr + Int32(tidx) * Int32(4), Int32(0)
                        )
                cute.arch.sync_threads()
            if has_task == Int32(0):
                if is_done > Int32(0):
                    consumer_live = Int32(0)
            elif warp_idx < self.num_mma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))
                task_valid_rows_val = _ld_shared_i32(ctrl_base_addr + Int32(24))

                alpha_value = alpha[task_expert_idx].to(cutlass.Float32)
                if cutlass.const_expr(self.is_w4a8):
                    # w4a8 activations are self-ranging: the calibrated input
                    # global scale is NOT applied at quantize time, so fold it
                    # out of the combined nvfp4 alpha to leave the pure weight
                    # dequant alpha (alpha_nvfp4 = 1/(gs_act * gs_w)).
                    alpha_value = alpha_value * input_global_scale[task_expert_idx].to(
                        cutlass.Float32
                    )
                valid_rows = task_valid_rows_val

                # In-loop SFA partition for sub-128 tiles. FC1 reads the
                # TMA-loaded activation SF, whose rows sit at offset
                # (task_m_tile_idx % sfa_tiles_per_block) within the shared
                # 128-row atom. (FC2 re-slices at offset 0 before phase B, since
                # its intermediate SF is quant-written to the atom's first half.)
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    _fc1_off = task_m_tile_idx % Int32(self.sfa_tiles_per_block)
                    _sA_il = cute.local_tile(
                        sA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (_fc1_off, 0, None),
                    )
                    tCrA = tiled_mma.make_fragment_A(
                        thr_mma.partition_A(_sA_il)[None, None, None, 0]
                    )
                    csA = thr_ld_A.partition_S(_sA_il)
                    crA = thr_ld_A.retile(tCrA)
                    _sSFA_il = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (_fc1_off, 0, None),
                    )
                    tCrSFA = self._dense_cls._partition_fragment_SFA(
                        self,  # type: ignore[arg-type]
                        _sSFA_il[None, None, 0],
                        thr_mma,
                        tidx,
                    )
                    csSFA = thr_ld_SFA.partition_S(_sSFA_il)
                    crSFA = thr_ld_SFA.retile(tCrSFA)

                _is_m_major = self.c_layout.is_m_major_c()
                copy_atom_r2s = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                    cutlass.BFloat16,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_Atom
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rGate = tiled_copy_r2s.retile(gate_acc)
                tRS_rUp = tiled_copy_r2s.retile(up_acc)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, cutlass.BFloat16
                )

                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rGate, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rGate, mode=[2])
                epi_buffer = Int32(0)

                down_alpha_value = down_alpha[task_expert_idx].to(cutlass.Float32)
                if cutlass.const_expr(self.is_w4a8):
                    down_alpha_value = down_alpha_value * global_scale[
                        task_expert_idx
                    ].to(cutlass.Float32)
                down_acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                MmaMPerEpiM = self.epi_tile[0] // mma_tile_m
                MmaNPerEpiN = self.epi_tile[1] // mma_tile_n
                fc2_m_tiles = cute.size(tCrA, mode=[1])
                fc2_n_tiles = cute.size(tCrB, mode=[1])

                fc1_m_tiles = cute.size(tCrA, mode=[1])
                fc1_n_tiles = cute.size(tCrB, mode=[1])
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    # ============================================================
                    # PHASE A: FC1 for this slice (gate/only pass, plus up for silu)
                    # ============================================================

                    if cutlass.const_expr(self.swap_ab):
                        # === Swapped FC1: produce this 128-int slice as n_sub x
                        # 32-int sub-tiles (weight on MMA M, tokens on N), then a
                        # transposed silu-store into sC[token,int] so the shared
                        # quant below packs it into sA exactly as the normal path.
                        n_sub = self.tile_shape_mnk[1] // self._fc1_int_tile
                        n_mt = gate_acc_sw.shape[1]
                        fz_crSFA_sw = cute.filter_zeros(crSFA_sw)
                        fz_crSFB_sw = cute.filter_zeros(crSFB_sw)
                        _nnt = cute.size(tCrB_sw, mode=[1])
                        # Accumulators carry the MMA m-tiles (a 32-int sub-tile
                        # spans n_mt MMA m16-tiles), the token n-tiles, and the
                        # n_sub index (which 32-int sub-tile of the 128-int slice).
                        # A Python list of tensors can't be tree-flattened as
                        # while-loop state, so use one rmem tensor each. n_sub is the
                        # OUTERMOST mode so a per-sub slice [None,:,:,_s] is compact
                        # ((atom),m_tile,n_tile) -- identical to dense's accumulator,
                        # which _reshape_acc_to_mn expects for the transposed store.
                        sw_gate = cute.make_rmem_tensor(
                            (gate_acc_sw.shape[0], n_mt, gate_acc_sw.shape[2], n_sub),
                            self.acc_dtype,
                        )
                        sw_up = cute.make_rmem_tensor(
                            (up_acc_sw.shape[0], n_mt, up_acc_sw.shape[2], n_sub),
                            self.acc_dtype,
                        )
                        sw_gate.fill(0.0)
                        sw_up.fill(0.0)
                        _int_k = (self._fc1_int_tile, self.tile_shape_mnk[2])
                        _sf_slice = cute.slice_(
                            self.fc1_tile_shape_mnk, (None, 0, None)
                        )
                        # task_m_tile_idx is a GLOBAL packed-tile index; this task's
                        # valid tokens sit at token m-tile _fc1_off within the 128-row
                        # activation atom. Select the fc1_tok_tile-wide N-slice that
                        # holds that m-tile (offset 0 when the atom is one such slice),
                        # and re-base the valid tokens to sC rows 0..valid_rows in the
                        # store via _tok_sub_off (the shared quant/scatter expect that).
                        _tiles_per_tok = self._fc1_tok_tile // self.tile_shape_mnk[0]
                        if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                            _fc1_off_sw = task_m_tile_idx % Int32(
                                self.sfa_tiles_per_block
                            )
                        else:
                            _fc1_off_sw = Int32(0)
                        _tok_tile_idx = _fc1_off_sw // Int32(_tiles_per_tok)
                        _tok_sub_off = (
                            _fc1_off_sw - _tok_tile_idx * Int32(_tiles_per_tok)
                        ) * Int32(self.tile_shape_mnk[0])
                        _tok_nslice = cute.slice_(
                            self.fc1_tile_shape_mnk, (0, None, None)
                        )
                        sA_part_sw_t = cute.local_tile(
                            sA, _tok_nslice, (_tok_tile_idx, 0, None)
                        )
                        sSFA_fc1_t = cute.local_tile(
                            sSFA_fc1, _tok_nslice, (_tok_tile_idx, 0, None)
                        )
                        _csB = thr_ld_B_sw.partition_S(sA_part_sw_t)
                        _fz_csSFB = cute.filter_zeros(
                            thr_ld_SFB_sw.partition_S(sSFA_fc1_t)
                        )

                        cons_state.reset_count()
                        for _kt in range(fc1_k_tile_cnt):
                            _pk = ml_pipeline.consumer_try_wait(cons_state)
                            ml_pipeline.consumer_wait(cons_state, _pk)
                            _i = cons_state.index
                            for _kb in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_copy_B_sw,
                                    _csB[None, None, _kb, _i],
                                    crB_sw[None, None, _kb],
                                )
                                cute.copy(
                                    smem_copy_SFB_sw,
                                    _fz_csSFB[None, None, _kb, _i],
                                    fz_crSFB_sw[None, None, _kb],
                                )
                                for _s in cutlass.range_constexpr(n_sub):
                                    # gate sub _s = atom-lo (sB/sSFB) at within-atom
                                    # sub gate_lo_sub+_s while that stays < n_sub,
                                    # else atom-hi (sB_up/sSFB_up). For a 128-aligned
                                    # forced swap (gate_lo_sub==0) every sub is in
                                    # atom-lo at offset _s (the verified n=384 path).
                                    if cutlass.const_expr(_s < n_sub - gate_lo_sub):
                                        _g_data = sB_fc1
                                        _g_sf = sSFB_fc1
                                        _g_sub = gate_lo_sub + _s
                                    else:
                                        _g_data = sB_up_fc1
                                        _g_sf = sSFB_up_fc1
                                        _g_sub = _s - (n_sub - gate_lo_sub)
                                    _csA_s = thr_ld_A_sw.partition_S(
                                        cute.local_tile(
                                            _g_data, _int_k, (Int32(_g_sub), 0, None)
                                        )
                                    )
                                    _csSFA_s = thr_ld_SFA_sw.partition_S(
                                        cute.local_tile(
                                            _g_sf, _sf_slice, (Int32(_g_sub), 0, None)
                                        )
                                    )
                                    cute.copy(
                                        smem_copy_A_sw,
                                        _csA_s[None, None, _kb, _i],
                                        crA_sw[None, None, _kb],
                                    )
                                    cute.copy(
                                        smem_copy_SFA_sw,
                                        cute.filter_zeros(_csSFA_s)[
                                            None, None, _kb, _i
                                        ],
                                        fz_crSFA_sw[None, None, _kb],
                                    )
                                    for _mt in cutlass.range_constexpr(n_mt):
                                        for _nt in cutlass.range_constexpr(_nnt):
                                            mma_atom.set(
                                                WarpField.SFA,
                                                tCrSFA_sw[None, _mt, _kb].iterator,
                                            )
                                            mma_atom.set(
                                                WarpField.SFB,
                                                tCrSFB_sw[None, _nt, _kb].iterator,
                                            )
                                            cute.gemm(
                                                mma_atom,
                                                sw_gate[None, _mt, _nt, _s],
                                                tCrA_sw[None, _mt, _kb],
                                                tCrB_sw[None, _nt, _kb],
                                                sw_gate[None, _mt, _nt, _s],
                                            )
                            ml_pipeline.consumer_release(cons_state)
                            cons_state.advance()
                        self.pass_gate_barrier.arrive_unaligned()

                        up_cons_state.reset_count()
                        for _kt in range(fc1_k_tile_cnt):
                            _pk = up_pipeline.consumer_try_wait(up_cons_state)
                            up_pipeline.consumer_wait(up_cons_state, _pk)
                            _i = up_cons_state.index
                            for _kb in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_copy_B_sw,
                                    _csB[None, None, _kb, _i],
                                    crB_sw[None, None, _kb],
                                )
                                cute.copy(
                                    smem_copy_SFB_sw,
                                    _fz_csSFB[None, None, _kb, _i],
                                    fz_crSFB_sw[None, None, _kb],
                                )
                                for _s in cutlass.range_constexpr(n_sub):
                                    _csA_s = thr_ld_A_sw.partition_S(
                                        cute.local_tile(
                                            sB_up_fc1, _int_k, (Int32(_s), 0, None)
                                        )
                                    )
                                    _csSFA_s = thr_ld_SFA_sw.partition_S(
                                        cute.local_tile(
                                            sSFB_up_fc1, _sf_slice, (Int32(_s), 0, None)
                                        )
                                    )
                                    cute.copy(
                                        smem_copy_A_sw,
                                        _csA_s[None, None, _kb, _i],
                                        crA_sw[None, None, _kb],
                                    )
                                    cute.copy(
                                        smem_copy_SFA_sw,
                                        cute.filter_zeros(_csSFA_s)[
                                            None, None, _kb, _i
                                        ],
                                        fz_crSFA_sw[None, None, _kb],
                                    )
                                    for _mt in cutlass.range_constexpr(n_mt):
                                        for _nt in cutlass.range_constexpr(_nnt):
                                            mma_atom.set(
                                                WarpField.SFA,
                                                tCrSFA_sw[None, _mt, _kb].iterator,
                                            )
                                            mma_atom.set(
                                                WarpField.SFB,
                                                tCrSFB_sw[None, _nt, _kb].iterator,
                                            )
                                            cute.gemm(
                                                mma_atom,
                                                sw_up[None, _mt, _nt, _s],
                                                tCrA_sw[None, _mt, _kb],
                                                tCrB_sw[None, _nt, _kb],
                                                sw_up[None, _mt, _nt, _s],
                                            )
                            up_pipeline.consumer_release(up_cons_state)
                            up_cons_state.advance()

                        sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
                        # transposed silu-store (dense.py swap_ab epilogue pattern):
                        # reshape the swapped acc (M-role=int, N-role=token) and the
                        # matching identity coords through _reshape_acc_to_mn so the
                        # MMA fragment's value layout is regrouped into a logical
                        # (M,N) grid -- a manual flat-element walk does NOT align the
                        # compact acc with the native coord layout. coord[0]=int,
                        # coord[1]=token; store sC[token, s*32+int].
                        _coord_sw = thr_mma_fc1.partition_C(
                            cute.make_identity_tensor(
                                (self.fc1_tile_shape_mnk[0], self.fc1_tile_shape_mnk[1])
                            )
                        )
                        _coord_mn = _reshape_acc_to_mn(_coord_sw, transpose=True)
                        # For a non-128-aligned n the last slice is partial; zero
                        # its tail int cols (global int >= n) so the FC2 (which
                        # contracts a full 128-int tile against TMA-zero-filled
                        # b_down) ignores them. global int base = slice*128.
                        _n_total = mB_w13.shape[0] // 2
                        _gslice = task_slice_begin_idx + slice_idx
                        _int_base = _gslice * Int32(self.tile_shape_mnk[1])
                        for _s in cutlass.range_constexpr(n_sub):
                            _g_mn = _reshape_acc_to_mn(
                                sw_gate[None, None, None, _s], transpose=True
                            )
                            _u_mn = _reshape_acc_to_mn(
                                sw_up[None, None, None, _s], transpose=True
                            )
                            for _am in cutlass.range_constexpr(
                                cute.size(_g_mn.shape[0])
                            ):
                                for _an in cutlass.range_constexpr(
                                    cute.size(_g_mn.shape[1])
                                ):
                                    _c = _coord_mn[_am, _an]
                                    _ir = _c[0]
                                    _tk = _c[1]
                                    _g = alpha_value * _g_mn[_am, _an]
                                    _u = alpha_value * _u_mn[_am, _an]
                                    _int_col = _s * self._fc1_int_tile + _ir
                                    _val = self._gated_activation_value(_g, _u).to(
                                        cutlass.BFloat16
                                    )
                                    if cutlass.const_expr(
                                        _n_total % self.tile_shape_mnk[1] != 0
                                    ):
                                        if _int_base + _int_col >= Int32(_n_total):
                                            _val = cutlass.BFloat16(0.0)
                                    _local_tk = _tk - _tok_sub_off
                                    if _local_tk >= Int32(0) and _local_tk < valid_rows:
                                        sC[_local_tk, _int_col, 0] = _val
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        # quant sC[token,int] -> sA (same packing as the normal path)
                        _epi_rows = valid_rows
                        if _epi_rows > Int32(self.epi_tile[0]):
                            _epi_rows = Int32(self.epi_tile[0])
                        _qgs = global_scale[task_expert_idx].to(cutlass.Float32)
                        _qi = Int32(tidx)
                        while _qi < _epi_rows * sf_blocks_per_row:
                            _lr = _qi // sf_blocks_per_row
                            row = _lr
                            _sfb = _qi - _lr * sf_blocks_per_row
                            _bs = _sfb * Int32(16)
                            values = cute.make_rmem_tensor((16,), cutlass.Float32)
                            block_max = cutlass.Float32(0.0)
                            for elem_idx in cutlass.range_constexpr(16):
                                value = cutlass.Float32(sC[_lr, _bs + elem_idx, 0])
                                values[elem_idx] = value
                                block_max = fmax_f32(block_max, fabs_f32(value))
                            packed64 = Uint64(0)
                            scale_byte = Uint8(0)
                            if self.is_gated and self.fast_math:
                                packed64, scale_byte = quantize_block_fp4_fast(
                                    values, block_max, _qgs
                                )
                            else:
                                packed64, scale_byte = quantize_block_fp4(
                                    values, block_max, _qgs
                                )
                            packed_base = _sfb << Int32(3)
                            dst_pcol = row & Int32(63)
                            xor_bits = ((dst_pcol >> Int32(1)) & Int32(0x3)) << Int32(4)
                            row_high = row >> Int32(6)
                            for byte_idx in cutlass.range_constexpr(8):
                                src_pcol = packed_base + Int32(byte_idx)
                                dst_row = ((src_pcol ^ xor_bits) << Int32(1)) + row_high
                                dst_flat = dst_row * packed_cols + dst_pcol
                                sA_u8[dst_flat] = Uint8(
                                    (packed64 >> Uint64(byte_idx * 8)) & Uint64(0xFF)
                                )
                            outer_m_idx = row % Int32(32)
                            inner_m_idx = row // Int32(32)
                            inner_k_idx = _sfb % Int32(4)
                            k_tile_idx = _sfb // Int32(4)
                            sf_raw_idx = (
                                k_tile_idx * Int32(32 * 4 * 4)
                                + outer_m_idx * Int32(4 * 4)
                                + inner_m_idx * Int32(4)
                                + inner_k_idx
                            )
                            st_shared_u8(sfa_base_addr + sf_raw_idx, scale_byte)
                            _qi += Int32(self.num_mma_warps * self.num_threads_per_warp)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(self.is_w4a8):
                        # ===== w4a8 FC1: direct-global operands, raw FP8 MMA =====
                        # Re-assert the MMA warps' register budget at the top of
                        # the w4a8 hot region: without a dominating setmaxnreg,
                        # ptxas register-targets for occupancy (observed: 40-reg
                        # compile, accumulators spilled around every QMMA).
                        cute.arch.setmaxregister_increase(self.mma_register_requirement)
                        # Probe-pinned addressing (tests/test_w4a8_fragment_probe):
                        # lane t (c=t%4, g=t/4) expands the packed-FP4 word for
                        # orig k [K0+8c, +8) into the (b0,b1) regs of the k32
                        # mxf8f6f4 atom; A regs are contiguous u32 loads; SFA row
                        # r is read from lanes 4*(r%8)+(r//8) byte 0, SFB col n
                        # from lane 4n byte 0.
                        m_tile_row_base = task_m_tile_idx * Int32(
                            self.tile_shape_mnk[0]
                        )
                        cur_slice = task_slice_begin_idx + slice_idx
                        up_n_tile = cur_slice
                        gate_n_tile = cur_slice
                        if cutlass.const_expr(self.is_gated):
                            gate_n_tile = cur_slice + gate_tile_cnt

                        # Accumulate into a FLAT rmem tensor (simple strides,
                        # constant indices): the nested partition_C layout of
                        # gate_acc/up_acc defeats register promotion under
                        # scalar element access (observed: 40-reg compile with
                        # GBs of local spill traffic). Results copy into the
                        # layout-native accumulators once per pass for the
                        # shared epilogue.
                        # Two passes (gate, then up) keep one 128-register
                        # accumulator hot at a time -- interleaving both blows the
                        # 232-register budget and thrashes spills. Per k32 block
                        # the B words for all n-atoms are loaded and expanded
                        # once into register arrays (the m-atom loop reuses
                        # them), batching the global loads back-to-back for
                        # memory-level parallelism.
                        # fc1_k_tile_cnt counts 128-wide FP4-position tiles over
                        # the byte-backed packed_a view (2K positions); the E4M3
                        # contraction advances 128 BYTES per iteration, so halve.
                        for _pass in cutlass.range_constexpr(
                            self.w4a8_fc1_windows_per_slice
                        ):
                            if cutlass.const_expr(_pass == 0):
                                pass_n_tile = gate_n_tile
                            else:
                                pass_n_tile = up_n_tile
                            # Accumulator as a flat tuple of SSA values:
                            # loop-carried across the dynamic k loop, so the
                            # MMAs read/write registers, never local memory.
                            w4a8_facc = tuple(
                                cutlass.Float32(0.0)
                                for _ in range(4 * fc1_m_tiles * fc1_n_tiles)
                            )
                            if cutlass.const_expr(self.w4a8_fused):
                                # Fused single sweep: the up accumulator rides
                                # the same k loop (64 hot regs at tile_m<=32).
                                w4a8_facc_u = tuple(
                                    cutlass.Float32(0.0)
                                    for _ in range(4 * fc1_m_tiles * fc1_n_tiles)
                                )
                            # ---- cp.async double-buffered operand staging ----
                            # Per 128-deep k-tile: the B tile (8.5KB padded), the
                            # SFB byte rows, the residual byte rows, and the
                            # A-scale byte rows are staged by the MMA warps while
                            # the previous tile computes. A payload stays as
                            # direct global loads (16 words per k32 block,
                            # covered by the MMA stream).
                            KT_fc1 = fc1_k_tile_cnt // 2
                            n_base_pass = pass_n_tile * Int32(self.tile_shape_mnk[1])
                            b_row_u32 = b_w13_u32.shape[1]
                            b_pass_base = Int64(task_expert_idx) * Int64(  # noqa: F841
                                b_w13_u32.shape[0] * b_row_u32
                            ) + Int64(n_base_pass) * Int64(b_row_u32)
                            sfb_row_b = sfb_w13_mx.shape[1]
                            sfb_pass_base = Int64(task_expert_idx) * Int64(  # noqa: F841
                                sfb_w13_mx.shape[0] * sfb_row_b
                            ) + Int64(n_base_pass) * Int64(sfb_row_b)
                            if cutlass.const_expr(self.w4a8_residual):
                                res_row_b = w13_residual.shape[1]
                                res_pass_base = Int64(task_expert_idx) * Int64(  # noqa: F841
                                    w13_residual.shape[0] * res_row_b
                                ) + Int64(n_base_pass) * Int64(res_row_b)
                            asc_pass_base = Int64(m_tile_row_base) * Int64(a_mx_per_row)  # noqa: F841

                            for _w4a8_kt in range(KT_fc1):
                                fc1_epoch = (
                                    slice_idx
                                    * Int32(
                                        self.w4a8_fc1_windows_per_slice * KT_fc1
                                        + (
                                            output_tile_cnt // 2
                                            if self.w4a8_fc2_pair
                                            else output_tile_cnt
                                        )
                                    )
                                    + Int32(_pass * KT_fc1)
                                    + _w4a8_kt
                                    + Int32(1)
                                )
                                par = (fc1_epoch - Int32(1)) & Int32(
                                    self.w4a8_depth - 1
                                )
                                _spin_wait_shared_ge_i32(
                                    w4a8_pipe_addr + (par << Int32(2)), fc1_epoch
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")

                                if cutlass.const_expr(self.w4a8_depth == 4):
                                    b_buf = (
                                        w4a8_sb0
                                        + (w4a8_sb1 - w4a8_sb0) * (par >> Int32(1))
                                        + (par & Int32(1)) * Int32(128 * 64)
                                    )
                                else:
                                    b_buf = w4a8_sb0 + (w4a8_sb1 - w4a8_sb0) * par
                                sa_buf = w4a8_sa0 + par * Int32(
                                    self.tile_shape_mnk[0] * 128
                                )
                                if cutlass.const_expr(self.w4a8_fused):
                                    sfb_buf = w4a8_sfbb + (par << Int32(10))
                                    res_buf = w4a8_res0 + (par << Int32(11))
                                else:
                                    sfb_buf = w4a8_sfbb + (par << Int32(9))
                                    res_buf = w4a8_res0 + (par << Int32(10))
                                # A-scale staging bufs: the sSFA region is
                                # tiny at small tiles (tile_m*8 per stage) and
                                # its base bytes carry the FC2 intermediate
                                # scales. At depth 4 the sSFB_up region is
                                # entirely free (residual bufs moved to
                                # sC/sA), so A-scale bufs live there.
                                if cutlass.const_expr(self.w4a8_small):
                                    asc_buf = w4a8_resb + par * Int32(
                                        self.tile_shape_mnk[0] * 4
                                    )
                                else:
                                    asc_buf = sfa_base_addr + (par << Int32(9))

                                n_in_arr = cute.make_rmem_tensor((fc1_n_tiles,), Int32)
                                sfb_words = cute.make_rmem_tensor(
                                    (fc1_n_tiles,), Uint32
                                )
                                res_w0 = cute.make_rmem_tensor((fc1_n_tiles,), Uint32)
                                res_w1 = cute.make_rmem_tensor((fc1_n_tiles,), Uint32)
                                if cutlass.const_expr(self.w4a8_fused):
                                    sfb_words_u = cute.make_rmem_tensor(
                                        (fc1_n_tiles,), Uint32
                                    )
                                    res_w0_u = cute.make_rmem_tensor(
                                        (fc1_n_tiles,), Uint32
                                    )
                                    res_w1_u = cute.make_rmem_tensor(
                                        (fc1_n_tiles,), Uint32
                                    )
                                for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                    _cnv = tCgC_id[(None, 0, _nt)]
                                    _cn = _cnv[0]
                                    n_in = _cn[1] - Int32(2) * lane_c + lane_g
                                    n_in_arr[_nt] = n_in
                                    sfb_words[_nt] = ld_shared_u32(
                                        sfb_buf + (n_in << Int32(2))
                                    )
                                    if cutlass.const_expr(self.w4a8_fused):
                                        sfb_words_u[_nt] = ld_shared_u32(
                                            sfb_buf + Int32(512) + (n_in << Int32(2))
                                        )
                                    if cutlass.const_expr(self.w4a8_residual):
                                        res_w0[_nt] = ld_shared_u32(
                                            res_buf + (n_in << Int32(3))
                                        )
                                        res_w1[_nt] = ld_shared_u32(
                                            res_buf + (n_in << Int32(3)) + Int32(4)
                                        )
                                        if cutlass.const_expr(self.w4a8_fused):
                                            res_w0_u[_nt] = ld_shared_u32(
                                                res_buf
                                                + Int32(1024)
                                                + (n_in << Int32(3))
                                            )
                                            res_w1_u[_nt] = ld_shared_u32(
                                                res_buf
                                                + Int32(1024)
                                                + (n_in << Int32(3))
                                                + Int32(4)
                                            )
                                asc_words = cute.make_rmem_tensor(
                                    (fc1_m_tiles,), Uint32
                                )
                                row_lo_arr = cute.make_rmem_tensor(
                                    (fc1_m_tiles,), Int32
                                )
                                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                    _c0v = tCgC_id[(None, _mt, 0)]
                                    _c0 = _c0v[0]
                                    row_lo_arr[_mt] = _c0[0]
                                    sel = _c0[0] + ((lane_c & Int32(1)) << Int32(3))
                                    asc_words[_mt] = ld_shared_u32(
                                        asc_buf + (sel << Int32(2))
                                    )

                                for _kb in cutlass.range_constexpr(n_k32_per_tile):
                                    kbg = _w4a8_kt * Int32(n_k32_per_tile) + Int32(_kb)
                                    a_word_base = kbg * Int32(8) + Int32(2) * lane_c  # noqa: F841
                                    b_lo = cute.make_rmem_tensor((fc1_n_tiles,), Uint32)
                                    b_hi = cute.make_rmem_tensor((fc1_n_tiles,), Uint32)
                                    sfb_arr = cute.make_rmem_tensor(
                                        (fc1_n_tiles,), Uint32
                                    )
                                    if cutlass.const_expr(self.w4a8_fused):
                                        b_lo_u = cute.make_rmem_tensor(
                                            (fc1_n_tiles,), Uint32
                                        )
                                        b_hi_u = cute.make_rmem_tensor(
                                            (fc1_n_tiles,), Uint32
                                        )
                                        sfb_arr_u = cute.make_rmem_tensor(
                                            (fc1_n_tiles,), Uint32
                                        )
                                    for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                        if cutlass.const_expr(self.w4a8_b_tma):
                                            # TMA layout (probe-pinned): chunk
                                            # = kb ^ ((row>>1)&3) over 64B rows.
                                            w4a8_bk = (
                                                Int32(_kb)
                                                ^ (
                                                    (n_in_arr[_nt] >> Int32(1))
                                                    & Int32(3)
                                                )
                                            ) << Int32(4)
                                        else:
                                            w4a8_bk = Int32(_kb * 16)
                                        w_pk = ld_shared_u32(
                                            b_buf
                                            + n_in_arr[_nt] * Int32(self.w4a8_b_pad)
                                            + w4a8_bk
                                            + (lane_c << Int32(2))
                                        )
                                        sfb_arr[_nt] = (
                                            sfb_words[_nt] >> Uint32(_kb * 8)
                                        ) & Uint32(0xFF)
                                        if cutlass.const_expr(self.w4a8_residual):
                                            if cutlass.const_expr(_kb < 2):
                                                res_word = res_w0[_nt]
                                            else:
                                                res_word = res_w1[_nt]
                                            res_b = (
                                                res_word
                                                >> (
                                                    Uint32((_kb * 2) % 4 * 8)
                                                    + (Uint32(lane_c) >> Uint32(1))
                                                    * Uint32(8)
                                                )
                                            ) & Uint32(0xFF)
                                            res_h2 = broadcast_f32_to_half2(
                                                fp8_e4m3_to_f32(res_b)
                                            )
                                            blo, bhi = e2m1x8_mul_residual_to_e4m3x8(
                                                w_pk, res_h2
                                            )
                                        else:
                                            blo, bhi = e2m1x8_to_e4m3x8(w_pk)
                                        b_lo[_nt] = blo
                                        b_hi[_nt] = bhi
                                        if cutlass.const_expr(self.w4a8_fused):
                                            w_pk_u = ld_shared_u32(
                                                b_buf
                                                + Int32(128 * 64)
                                                + n_in_arr[_nt] * Int32(self.w4a8_b_pad)
                                                + w4a8_bk
                                                + (lane_c << Int32(2))
                                            )
                                            sfb_arr_u[_nt] = (
                                                sfb_words_u[_nt] >> Uint32(_kb * 8)
                                            ) & Uint32(0xFF)
                                            if cutlass.const_expr(self.w4a8_residual):
                                                if cutlass.const_expr(_kb < 2):
                                                    res_word_u = res_w0_u[_nt]
                                                else:
                                                    res_word_u = res_w1_u[_nt]
                                                res_b_u = (
                                                    res_word_u
                                                    >> (
                                                        Uint32((_kb * 2) % 4 * 8)
                                                        + (Uint32(lane_c) >> Uint32(1))
                                                        * Uint32(8)
                                                    )
                                                ) & Uint32(0xFF)
                                                res_h2_u = broadcast_f32_to_half2(
                                                    fp8_e4m3_to_f32(res_b_u)
                                                )
                                                blo_u, bhi_u = (
                                                    e2m1x8_mul_residual_to_e4m3x8(
                                                        w_pk_u, res_h2_u
                                                    )
                                                )
                                            else:
                                                blo_u, bhi_u = e2m1x8_to_e4m3x8(w_pk_u)
                                            b_lo_u[_nt] = blo_u
                                            b_hi_u[_nt] = bhi_u
                                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                        row_lo = row_lo_arr[_mt]
                                        row_hi = row_lo + Int32(8)
                                        a_addr_lo = (
                                            sa_buf
                                            + (row_lo << Int32(7))
                                            + Int32(_kb * 32)
                                            + (lane_c << Int32(3))
                                        )
                                        a_addr_hi = (
                                            sa_buf
                                            + (row_hi << Int32(7))
                                            + Int32(_kb * 32)
                                            + (lane_c << Int32(3))
                                        )
                                        a0 = ld_shared_u32(a_addr_lo)
                                        a1 = ld_shared_u32(a_addr_hi)
                                        a2 = ld_shared_u32(a_addr_lo + Int32(4))
                                        a3 = ld_shared_u32(a_addr_hi + Int32(4))
                                        sfa_w = (
                                            asc_words[_mt] >> Uint32(_kb * 8)
                                        ) & Uint32(0xFF)
                                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                            _fi = ((_mt * fc1_n_tiles) + _nt) * 4
                                            d0, d1, d2, d3 = (
                                                mxfp8_mma_m16n8k32_f32_e4m3(
                                                    w4a8_facc[_fi],
                                                    w4a8_facc[_fi + 1],
                                                    w4a8_facc[_fi + 2],
                                                    w4a8_facc[_fi + 3],
                                                    a0,
                                                    a1,
                                                    a2,
                                                    a3,
                                                    b_lo[_nt],
                                                    b_hi[_nt],
                                                    sfa_w,
                                                    sfb_arr[_nt],
                                                )
                                            )
                                            w4a8_facc = (
                                                w4a8_facc[:_fi]
                                                + (d0, d1, d2, d3)
                                                + w4a8_facc[_fi + 4 :]
                                            )
                                            if cutlass.const_expr(self.w4a8_fused):
                                                u0, u1, u2, u3 = (
                                                    mxfp8_mma_m16n8k32_f32_e4m3(
                                                        w4a8_facc_u[_fi],
                                                        w4a8_facc_u[_fi + 1],
                                                        w4a8_facc_u[_fi + 2],
                                                        w4a8_facc_u[_fi + 3],
                                                        a0,
                                                        a1,
                                                        a2,
                                                        a3,
                                                        b_lo_u[_nt],
                                                        b_hi_u[_nt],
                                                        sfa_w,
                                                        sfb_arr_u[_nt],
                                                    )
                                                )
                                                w4a8_facc_u = (
                                                    w4a8_facc_u[:_fi]
                                                    + (u0, u1, u2, u3)
                                                    + w4a8_facc_u[_fi + 4 :]
                                                )
                                cute.arch.sync_warp()
                                if lane_c == Int32(0) and lane_g == Int32(0):
                                    atomic_add_shared_i32(
                                        w4a8_pipe_addr + Int32(16) + (par << Int32(2)),
                                        Int32(1),
                                    )
                            # Hand the pass result to the epilogue's
                            # layout-native accumulator (cold copy).
                            for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                    _fi = ((_mt * fc1_n_tiles) + _nt) * 4
                                    if cutlass.const_expr(_pass == 0):
                                        o_sl = gate_acc[(None, _mt, _nt)]
                                    else:
                                        o_sl = up_acc[(None, _mt, _nt)]
                                    o_sl[0] = w4a8_facc[_fi]
                                    o_sl[1] = w4a8_facc[_fi + 1]
                                    o_sl[2] = w4a8_facc[_fi + 2]
                                    o_sl[3] = w4a8_facc[_fi + 3]
                                    if cutlass.const_expr(self.w4a8_fused):
                                        u_sl = up_acc[(None, _mt, _nt)]
                                        u_sl[0] = w4a8_facc_u[_fi]
                                        u_sl[1] = w4a8_facc_u[_fi + 1]
                                        u_sl[2] = w4a8_facc_u[_fi + 2]
                                        u_sl[3] = w4a8_facc_u[_fi + 3]
                        self.pass_gate_barrier.arrive_unaligned()

                        # All MMA warps must be past their FC1 staging-buffer
                        # reads before sC is rewritten below: without the old
                        # per-k-tile CTA barriers, a fast warp's epilogue store
                        # could clobber a lagging warp's staged A tile in sC.
                        self.epilog_sync_barrier.arrive_and_wait()

                        # Activation into sC (same math as the nvfp4 path), then
                        # w4a8 quantize: E4M3 payload + UE8M0 scales into the
                        # repurposed sA/sSFA smem regions at plain layouts.
                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            epi_m_valid = valid_rows - Int32(epi_m) * Int32(
                                self.epi_tile[0]
                            )
                            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                            if epi_m_valid > Int32(0):
                                for mma_n_in_epi in cutlass.range_constexpr(
                                    MmaNPerEpiN
                                ):
                                    for mma_m_in_epi in cutlass.range_constexpr(
                                        MmaMPerEpiM
                                    ):
                                        mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                        mma_n = mma_n_in_epi
                                        tRS_rD_slice = tRS_rD[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                        if cutlass.const_expr(self.is_gated):
                                            up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                            for elem_idx in cutlass.range_constexpr(
                                                cute.size(tRS_rD_slice)
                                            ):
                                                g = alpha_value * gate_slice[elem_idx]
                                                u = alpha_value * up_slice[elem_idx]
                                                tRS_rD_slice[elem_idx] = (
                                                    self._gated_activation_value(g, u)
                                                )
                                        else:
                                            for elem_idx in cutlass.range_constexpr(
                                                cute.size(tRS_rD_slice)
                                            ):
                                                g = alpha_value * gate_slice[elem_idx]
                                                relu_g = fmax_f32(
                                                    g, cutlass.Float32(0.0)
                                                )
                                                tRS_rD_slice[elem_idx] = relu_g * relu_g

                                acc_vec = tRS_rD.load()
                                acc_vec = acc_vec.to(cutlass.BFloat16)
                                tRS_rD_out.store(acc_vec)
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                            epi_rows = epi_m_valid
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)
                            mx_blocks_tile = Int32(self.tile_shape_mnk[2] // 32)
                            quant_idx = Int32(tidx)
                            while quant_idx < epi_rows * mx_blocks_tile:
                                local_row = quant_idx // mx_blocks_tile
                                row = rows_offset + local_row
                                mx_block = quant_idx - local_row * mx_blocks_tile
                                block_start = mx_block * Int32(32)
                                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(32):
                                    value = cutlass.Float32(
                                        sC[
                                            local_row,
                                            block_start + elem_idx,
                                            epi_buffer,
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))
                                payload, mx_scale_byte = quantize_block_fp8_mx(
                                    values, block_max
                                )
                                pay_addr = (
                                    sa_flat_addr
                                    + row * Int32(self.tile_shape_mnk[2])
                                    + block_start
                                )
                                for word_idx in cutlass.range_constexpr(8):
                                    st_shared_u32(
                                        pay_addr + Int32(word_idx * 4),
                                        payload[word_idx],
                                    )
                                st_shared_u8(
                                    sfa_base_addr + row * mx_blocks_tile + mx_block,
                                    Uint8(mx_scale_byte & Uint32(0xFF)),
                                )
                                quant_idx += Int32(
                                    self.num_mma_warps * self.num_threads_per_warp
                                )

                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr((not self.swap_ab) and (not self.is_w4a8)):
                        # Gate GEMM (inlined to avoid @cute.jit pass-by-value for acc)
                        fz_crSFA = cute.filter_zeros(crSFA)
                        fz_crSFB = cute.filter_zeros(crSFB)
                        gate_acc.fill(0.0)
                        cons_state.reset_count()
                        peek = ml_pipeline.consumer_try_wait(cons_state)
                        ml_pipeline.consumer_wait(cons_state, peek)
                        csA_p = csA[None, None, None, cons_state.index]
                        csB_p = csB[None, None, None, cons_state.index]
                        csSFA_p = csSFA[None, None, None, cons_state.index]
                        csSFB_p = csSFB[None, None, None, cons_state.index]
                        cute.copy(smem_copy_A, csA_p[None, None, 0], crA[None, None, 0])
                        cute.copy(smem_copy_B, csB_p[None, None, 0], crB[None, None, 0])
                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                        cute.copy(
                            smem_copy_SFA,
                            fz_csSFA_p[None, None, 0],
                            fz_crSFA[None, None, 0],
                        )
                        cute.copy(
                            smem_copy_SFB,
                            fz_csSFB_p[None, None, 0],
                            fz_crSFB[None, None, 0],
                        )
                        for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_next = (
                                    0
                                    if k_block_idx + 1 == num_k_blocks
                                    else k_block_idx + 1
                                )
                                if k_block_idx == num_k_blocks - 1:
                                    ml_pipeline.consumer_release(cons_state)
                                    cons_state.advance()
                                    peek = ml_pipeline.consumer_try_wait(cons_state)
                                    csA_p = csA[None, None, None, cons_state.index]
                                    csB_p = csB[None, None, None, cons_state.index]
                                    csSFA_p = csSFA[None, None, None, cons_state.index]
                                    csSFB_p = csSFB[None, None, None, cons_state.index]
                                    fz_csSFA_p = cute.filter_zeros(csSFA_p)
                                    fz_csSFB_p = cute.filter_zeros(csSFB_p)
                                    ml_pipeline.consumer_wait(cons_state, peek)
                                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                    for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                        mma_atom.set(
                                            WarpField.SFA,
                                            tCrSFA[None, _mt, k_block_idx].iterator,
                                        )
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB[None, _nt, k_block_idx].iterator,
                                        )
                                        cute.gemm(
                                            mma_atom,
                                            gate_acc[None, _mt, _nt],
                                            tCrA[None, _mt, k_block_idx],
                                            tCrB[None, _nt, k_block_idx],
                                            gate_acc[None, _mt, _nt],
                                        )
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_next],
                                    crA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                                fz_csSFA_cur = cute.filter_zeros(
                                    csSFA[None, None, None, cons_state.index]
                                )
                                fz_csSFB_cur = cute.filter_zeros(
                                    csSFB[None, None, None, cons_state.index]
                                )
                                cute.copy(
                                    smem_copy_SFA,
                                    fz_csSFA_cur[None, None, k_next],
                                    fz_crSFA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_SFB,
                                    fz_csSFB_cur[None, None, k_next],
                                    fz_crSFB[None, None, k_next],
                                )
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_next = (
                                0
                                if k_block_idx + 1 == num_k_blocks
                                else k_block_idx + 1
                            )
                            if k_block_idx == num_k_blocks - 1:
                                ml_pipeline.consumer_release(cons_state)
                                cons_state.advance()
                            if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                                cute.copy(
                                    smem_copy_A,
                                    csA_p[None, None, k_next],
                                    crA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_B,
                                    csB_p[None, None, k_next],
                                    crB[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_SFA,
                                    fz_csSFA_p[None, None, k_next],
                                    fz_crSFA[None, None, k_next],
                                )
                                cute.copy(
                                    smem_copy_SFB,
                                    fz_csSFB_p[None, None, k_next],
                                    fz_crSFB[None, None, k_next],
                                )
                            for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA[None, _mt, k_block_idx].iterator,
                                    )
                                    mma_atom.set(
                                        WarpField.SFB,
                                        tCrSFB[None, _nt, k_block_idx].iterator,
                                    )
                                    cute.gemm(
                                        mma_atom,
                                        gate_acc[None, _mt, _nt],
                                        tCrA[None, _mt, k_block_idx],
                                        tCrB[None, _nt, k_block_idx],
                                        gate_acc[None, _mt, _nt],
                                    )
                        # Signal FC1 gate/only completion before the DMA warp
                        # reuses the shared A/gate buffers for the next pass.
                        self.pass_gate_barrier.arrive_unaligned()

                        if self.is_gated:
                            # Up GEMM (inlined, same pattern)
                            up_acc.fill(0.0)
                            up_cons_state.reset_count()
                            peek = up_pipeline.consumer_try_wait(up_cons_state)
                            up_pipeline.consumer_wait(up_cons_state, peek)
                            csA_p = csA[None, None, None, up_cons_state.index]
                            csB_p = csB_up[None, None, None, up_cons_state.index]
                            csSFA_p = csSFA[None, None, None, up_cons_state.index]
                            csSFB_p = csSFB_up[None, None, None, up_cons_state.index]
                            cute.copy(
                                smem_copy_A, csA_p[None, None, 0], crA[None, None, 0]
                            )
                            cute.copy(
                                smem_copy_B, csB_p[None, None, 0], crB[None, None, 0]
                            )
                            fz_csSFA_p = cute.filter_zeros(csSFA_p)
                            fz_csSFB_p = cute.filter_zeros(csSFB_p)
                            cute.copy(
                                smem_copy_SFA,
                                fz_csSFA_p[None, None, 0],
                                fz_crSFA[None, None, 0],
                            )
                            cute.copy(
                                smem_copy_SFB,
                                fz_csSFB_p[None, None, 0],
                                fz_crSFB[None, None, 0],
                            )
                            for _k_tile in range(0, fc1_k_tile_cnt - 1, 1, unroll=4):  # type: ignore[call-overload]
                                for k_block_idx in cutlass.range_constexpr(
                                    num_k_blocks
                                ):
                                    k_next = (
                                        0
                                        if k_block_idx + 1 == num_k_blocks
                                        else k_block_idx + 1
                                    )
                                    if k_block_idx == num_k_blocks - 1:
                                        up_pipeline.consumer_release(up_cons_state)
                                        up_cons_state.advance()
                                        peek = up_pipeline.consumer_try_wait(
                                            up_cons_state
                                        )
                                        csA_p = csA[
                                            None, None, None, up_cons_state.index
                                        ]
                                        csB_p = csB_up[
                                            None, None, None, up_cons_state.index
                                        ]
                                        csSFA_p = csSFA[
                                            None, None, None, up_cons_state.index
                                        ]
                                        csSFB_p = csSFB_up[
                                            None, None, None, up_cons_state.index
                                        ]
                                        fz_csSFA_p = cute.filter_zeros(csSFA_p)
                                        fz_csSFB_p = cute.filter_zeros(csSFB_p)
                                        up_pipeline.consumer_wait(up_cons_state, peek)
                                    for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                        for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                            mma_atom.set(
                                                WarpField.SFA,
                                                tCrSFA[None, _mt, k_block_idx].iterator,
                                            )
                                            mma_atom.set(
                                                WarpField.SFB,
                                                tCrSFB[None, _nt, k_block_idx].iterator,
                                            )
                                            cute.gemm(
                                                mma_atom,
                                                up_acc[None, _mt, _nt],
                                                tCrA[None, _mt, k_block_idx],
                                                tCrB[None, _nt, k_block_idx],
                                                up_acc[None, _mt, _nt],
                                            )
                                    cute.copy(
                                        smem_copy_A,
                                        csA_p[None, None, k_next],
                                        crA[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_B,
                                        csB_p[None, None, k_next],
                                        crB[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_SFA,
                                        fz_csSFA_p[None, None, k_next],
                                        fz_crSFA[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_SFB,
                                        fz_csSFB_p[None, None, k_next],
                                        fz_crSFB[None, None, k_next],
                                    )
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_next = (
                                    0
                                    if k_block_idx + 1 == num_k_blocks
                                    else k_block_idx + 1
                                )
                                if k_block_idx == num_k_blocks - 1:
                                    up_pipeline.consumer_release(up_cons_state)
                                    up_cons_state.advance()
                                if k_next > 0 and fc1_k_tile_cnt > Int32(0):
                                    cute.copy(
                                        smem_copy_A,
                                        csA_p[None, None, k_next],
                                        crA[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_B,
                                        csB_p[None, None, k_next],
                                        crB[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_SFA,
                                        fz_csSFA_p[None, None, k_next],
                                        fz_crSFA[None, None, k_next],
                                    )
                                    cute.copy(
                                        smem_copy_SFB,
                                        fz_csSFB_p[None, None, k_next],
                                        fz_crSFB[None, None, k_next],
                                    )
                                for _mt in cutlass.range_constexpr(fc1_m_tiles):
                                    for _nt in cutlass.range_constexpr(fc1_n_tiles):
                                        mma_atom.set(
                                            WarpField.SFA,
                                            tCrSFA[None, _mt, k_block_idx].iterator,
                                        )
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB[None, _nt, k_block_idx].iterator,
                                        )
                                        cute.gemm(
                                            mma_atom,
                                            up_acc[None, _mt, _nt],
                                            tCrA[None, _mt, k_block_idx],
                                            tCrB[None, _nt, k_block_idx],
                                            up_acc[None, _mt, _nt],
                                        )
                        # Activation + quant into sA
                        sA_u8 = cute.recast_tensor(sA[None, None, 0], cutlass.Uint8)
                        packed_cols = Int32(self.tile_shape_mnk[2] // 2)
                        sf_blocks_per_row = Int32(self.tile_shape_mnk[2] // 16)
                        gs_value = global_scale[task_expert_idx].to(cutlass.Float32)
                        if cutlass.const_expr(self.dynamic_down_scale):
                            fc2_down_alpha_value = down_alpha_value

                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            epi_m_valid = valid_rows - Int32(epi_m) * Int32(
                                self.epi_tile[0]
                            )
                            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                            if epi_m_valid > Int32(0):
                                for mma_n_in_epi in cutlass.range_constexpr(
                                    MmaNPerEpiN
                                ):
                                    for mma_m_in_epi in cutlass.range_constexpr(
                                        MmaMPerEpiM
                                    ):
                                        mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                        mma_n = mma_n_in_epi
                                        tRS_rD_slice = tRS_rD[
                                            (None, mma_m_in_epi, mma_n_in_epi)
                                        ]
                                        gate_slice = tRS_rGate[(None, mma_m, mma_n)]
                                        if self.is_gated:
                                            up_slice = tRS_rUp[(None, mma_m, mma_n)]
                                            for elem_idx in cutlass.range_constexpr(
                                                cute.size(tRS_rD_slice)
                                            ):
                                                g = alpha_value * gate_slice[elem_idx]
                                                u = alpha_value * up_slice[elem_idx]
                                                tRS_rD_slice[elem_idx] = (
                                                    self._gated_activation_value(g, u)
                                                )
                                        else:
                                            for elem_idx in cutlass.range_constexpr(
                                                cute.size(tRS_rD_slice)
                                            ):
                                                g = alpha_value * gate_slice[elem_idx]
                                                relu_g = fmax_f32(
                                                    g, cutlass.Float32(0.0)
                                                )
                                                tRS_rD_slice[elem_idx] = relu_g * relu_g

                                acc_vec = tRS_rD.load()
                                acc_vec = acc_vec.to(cutlass.BFloat16)
                                tRS_rD_out.store(acc_vec)
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rD_out,
                                    tRS_sD[(None, None, None, epi_buffer)],
                                )
                                cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])
                            epi_rows = epi_m_valid
                            if epi_rows > Int32(self.epi_tile[0]):
                                epi_rows = Int32(self.epi_tile[0])
                            if epi_rows < Int32(0):
                                epi_rows = Int32(0)
                            quant_gs_value = gs_value
                            if cutlass.const_expr(self.dynamic_down_scale):
                                if epi_rows > Int32(0):
                                    local_max = cutlass.Float32(0.0)
                                    scan_idx = Int32(tidx)
                                    scan_total = (
                                        epi_rows * sf_blocks_per_row * Int32(16)
                                    )
                                    while scan_idx < scan_total:
                                        sr = scan_idx // (sf_blocks_per_row * Int32(16))
                                        sc = scan_idx % (sf_blocks_per_row * Int32(16))
                                        local_max = fmax_f32(
                                            local_max,
                                            fabs_f32(
                                                cutlass.Float32(sC[sr, sc, epi_buffer])
                                            ),
                                        )
                                        scan_idx += Int32(
                                            self.num_mma_warps
                                            * self.num_threads_per_warp
                                        )
                                    warp_amax = warp_reduce(local_max, fmax_f32)
                                    lane_id = Int32(tidx) & Int32(31)
                                    if lane_id == Int32(0):
                                        st_shared_f32(
                                            reduce_scratch_addr + warp_idx * Int32(4),
                                            warp_amax,
                                        )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if warp_idx == 0:
                                        tile_amax = cutlass.Float32(0.0)
                                        if lane_id < Int32(self.num_mma_warps):
                                            tile_amax = ld_shared_f32(
                                                reduce_scratch_addr + lane_id * Int32(4)
                                            )
                                        tile_amax = warp_reduce(tile_amax, fmax_f32)
                                        if lane_id == Int32(0):
                                            st_shared_f32(
                                                reduce_scratch_addr, tile_amax
                                            )
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    tile_amax = ld_shared_f32(reduce_scratch_addr)
                                    tile_gs_value = cutlass.Float32(0.0)
                                    if tile_amax > cutlass.Float32(0.0):
                                        tile_gs_value = (
                                            cutlass.Float32(_FC2_TILE_RECIP_GS_NUM)
                                            / tile_amax
                                        )
                                    tile_gs_value = fmax_f32(
                                        tile_gs_value, cutlass.Float32(1.0e-12)
                                    )
                                    if tile_gs_value != cutlass.Float32(0.0):
                                        fc2_down_alpha_value = down_alpha_value * (
                                            gs_value / tile_gs_value
                                        )
                                    quant_gs_value = tile_gs_value
                                    self.epilog_sync_barrier.arrive_and_wait()
                            quant_idx = Int32(tidx)
                            while quant_idx < epi_rows * sf_blocks_per_row:
                                local_row = quant_idx // sf_blocks_per_row
                                row = rows_offset + local_row
                                sf_block = quant_idx - local_row * sf_blocks_per_row
                                block_start = sf_block * Int32(16)

                                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                                block_max = cutlass.Float32(0.0)
                                for elem_idx in cutlass.range_constexpr(16):
                                    value = cutlass.Float32(
                                        sC[
                                            local_row,
                                            block_start + elem_idx,
                                            epi_buffer,
                                        ]
                                    )
                                    values[elem_idx] = value
                                    block_max = fmax_f32(block_max, fabs_f32(value))

                                packed64 = Uint64(0)
                                scale_byte = Uint8(0)
                                if self.is_gated and self.fast_math:
                                    packed64, scale_byte = quantize_block_fp4_fast(
                                        values, block_max, quant_gs_value
                                    )
                                else:
                                    packed64, scale_byte = quantize_block_fp4(
                                        values, block_max, quant_gs_value
                                    )
                                packed_base = sf_block << Int32(3)
                                dst_pcol = row & Int32(63)
                                xor_bits = (
                                    (dst_pcol >> Int32(1)) & Int32(0x3)
                                ) << Int32(4)
                                row_high = row >> Int32(6)
                                for byte_idx in cutlass.range_constexpr(8):
                                    src_pcol = packed_base + Int32(byte_idx)
                                    dst_row = (
                                        (src_pcol ^ xor_bits) << Int32(1)
                                    ) + row_high
                                    dst_flat = dst_row * packed_cols + dst_pcol
                                    byte_val = Uint8(
                                        (packed64 >> Uint64(byte_idx * 8))
                                        & Uint64(0xFF)
                                    )
                                    sA_u8[dst_flat] = byte_val

                                outer_m_idx = row % Int32(32)
                                inner_m_idx = row // Int32(32)
                                inner_k_idx = sf_block % Int32(4)
                                k_tile_idx = sf_block // Int32(4)
                                sf_raw_idx = (
                                    k_tile_idx * Int32(32 * 4 * 4)
                                    + outer_m_idx * Int32(4 * 4)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                st_shared_u8(sfa_base_addr + sf_raw_idx, scale_byte)
                                quant_idx += Int32(
                                    self.num_mma_warps * self.num_threads_per_warp
                                )

                        cute.arch.fence_proxy("async.shared", space="cta")
                        # epilog_sync: MMA-only barrier. DMA warp doesn't need to wait
                        # for quant — it only loads B_down into sB (separate buffer).
                        # This allows DMA to prefetch B_down tiles earlier.
                        self.epilog_sync_barrier.arrive_and_wait()

                    # ============================================================
                    # PHASE B: Sweep ALL FC2 output tiles using cached sA
                    # No CTA-wide barrier needed here: gate is done with sB/sSFB
                    # (barrier at line 925 ensured that), up uses sB_up/sSFB_up,
                    # and DMA's B_down loads into sB/sSFB don't conflict with
                    # MMA's activation+quant on sC/sA/sSFA. The phase2_pipeline
                    # handles B_down availability for FC2 GEMM.
                    # ============================================================
                    scatter_N = Int32(scatter_output.shape[1])
                    lane_id = Int32(tidx) & Int32(31)
                    warp_in_tile = Int32(tidx) >> Int32(5)
                    warp_m_base = (warp_in_tile >> Int32(1)) * Int32(64)
                    warp_n_base = (warp_in_tile & Int32(1)) * Int32(64)

                    # FC2's intermediate A/SF were quant-written to the first
                    # half of the shared 128-row atom, so phase B reads offset 0
                    # (vs FC1's per-task offset). Re-slice both. Identity at
                    # tile_m==128.
                    if cutlass.const_expr(
                        (not self.is_w4a8) and self.sfa_tiles_per_block > 1
                    ):
                        _sA_p2 = cute.local_tile(
                            sA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (Int32(0), 0, None),
                        )
                        tCrA = tiled_mma.make_fragment_A(
                            thr_mma.partition_A(_sA_p2)[None, None, None, 0]
                        )
                        csA = thr_ld_A.partition_S(_sA_p2)
                        crA = thr_ld_A.retile(tCrA)
                        _sSFA_p2 = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (Int32(0), 0, None),
                        )
                        tCrSFA = self._dense_cls._partition_fragment_SFA(
                            self,  # type: ignore[arg-type]
                            _sSFA_p2[None, None, 0],
                            thr_mma,
                            tidx,
                        )
                        csSFA = thr_ld_SFA.partition_S(_sSFA_p2)
                        crSFA = thr_ld_SFA.retile(tCrSFA)

                    expert_idx = task_expert_idx

                    if cutlass.const_expr(not self.is_w4a8):
                        csA_phase2 = csA[None, None, None, 0]
                        csSFA_phase2 = csSFA[None, None, None, 0]

                        # Consume all output tiles continuously from phase2_pipeline.

                        # Hoist A-side register loads: sA is constant across all
                        # FC2 output tiles (quantized intermediate). Load crA and
                        # crSFA for all k-blocks once, reuse for all 32 tiles.
                        fz_crSFA_p2 = cute.filter_zeros(crSFA)
                        cute.copy(
                            smem_copy_A, csA_phase2[None, None, 0], crA[None, None, 0]
                        )
                        fz_csSFA_p2 = cute.filter_zeros(csSFA_phase2)
                        cute.copy(
                            smem_copy_SFA,
                            fz_csSFA_p2[None, None, 0],
                            fz_crSFA_p2[None, None, 0],
                        )
                        for _kb_pre in cutlass.range_constexpr(num_k_blocks - 1):
                            k_pre = _kb_pre + 1
                            cute.copy(
                                smem_copy_A,
                                csA_phase2[None, None, k_pre],
                                crA[None, None, k_pre],
                            )
                            cute.copy(
                                smem_copy_SFA,
                                fz_csSFA_p2[None, None, k_pre],
                                fz_crSFA_p2[None, None, k_pre],
                            )

                    # Seed FC1-epilogue scratch that the FC2 loop re-binds. With
                    # the non-swap FC1 wrapped in `const_expr(not swap_ab)`, these
                    # names don't reliably escape to here, so the dynamic FC2 loop
                    # would otherwise see a None->typed flip on its first trace.
                    # Values are overwritten inside FC2 (write-before-read).
                    k_next = 0
                    mma_m = 0
                    mma_n = 0
                    rows_offset = Int32(0)
                    local_row = Int32(0)
                    phase2_cons_state.reset_count()
                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        if cutlass.const_expr(not self.is_w4a8):
                            phase2_peek = phase2_pipeline.consumer_try_wait(
                                phase2_cons_state
                            )
                            phase2_pipeline.consumer_wait(
                                phase2_cons_state, phase2_peek
                            )
                            csB_phase2 = csB[None, None, None, phase2_cons_state.index]
                            csSFB_phase2 = csSFB[
                                None, None, None, phase2_cons_state.index
                            ]

                            # Only load B-side (B_down changes per output tile; A is hoisted)
                            cute.copy(
                                smem_copy_B,
                                csB_phase2[None, None, 0],
                                crB[None, None, 0],
                            )
                            f2 = cute.filter_zeros(csSFB_phase2)
                            f4 = cute.filter_zeros(crSFB)
                            cute.copy(
                                smem_copy_SFB, f2[None, None, 0], f4[None, None, 0]
                            )

                        if cutlass.const_expr(not self.is_w4a8):
                            down_acc.fill(0.0)
                        if cutlass.const_expr(self.is_w4a8):
                            # w4a8 FC2: A from the epilogue's plain smem region,
                            # B_down/scales/residuals cp.async double-buffered
                            # across output tiles, raw FP8 MMA.
                            w4a8_facc2 = tuple(
                                cutlass.Float32(0.0)
                                for _ in range(4 * fc2_m_tiles * fc2_n_tiles)
                            )
                            w4a8_fc2_windows = (
                                output_tile_cnt // 2
                                if self.w4a8_fc2_pair
                                else output_tile_cnt
                            )
                            fc2_epoch = (
                                slice_idx
                                * Int32(
                                    self.w4a8_fc1_windows_per_slice
                                    * (fc1_k_tile_cnt // 2)
                                    + w4a8_fc2_windows
                                )
                                + Int32(
                                    self.w4a8_fc1_windows_per_slice
                                    * (fc1_k_tile_cnt // 2)
                                )
                                + (
                                    (output_tile_idx >> Int32(1))
                                    if cutlass.const_expr(self.w4a8_fc2_pair)
                                    else output_tile_idx
                                )
                                + Int32(1)
                            )
                            par = (fc2_epoch - Int32(1)) & Int32(self.w4a8_depth - 1)
                            _spin_wait_shared_ge_i32(
                                w4a8_pipe_addr + (par << Int32(2)), fc2_epoch
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")

                            if cutlass.const_expr(self.w4a8_depth == 4):
                                b_buf = (
                                    w4a8_sb0
                                    + (w4a8_sb1 - w4a8_sb0) * (par >> Int32(1))
                                    + (par & Int32(1)) * Int32(128 * 64)
                                )
                            else:
                                b_buf = w4a8_sb0 + (w4a8_sb1 - w4a8_sb0) * par
                            if cutlass.const_expr(self.w4a8_fc2_pair):
                                w4a8_t_in_pair = output_tile_idx & Int32(1)
                                b_buf = b_buf + w4a8_t_in_pair * Int32(128 * 64)
                                sfb_buf = (
                                    w4a8_sfbb
                                    + (par << Int32(10))
                                    + (w4a8_t_in_pair << Int32(9))
                                )
                                res_buf = (
                                    w4a8_res2
                                    + (par << Int32(11))
                                    + (w4a8_t_in_pair << Int32(10))
                                )
                            else:
                                sfb_buf = w4a8_sfbb + (par << Int32(9))
                                res_buf = w4a8_res2 + (par << Int32(10))
                            mx_blocks_tile_p2 = Int32(self.tile_shape_mnk[2] // 32)

                            n_in_arr2 = cute.make_rmem_tensor((fc2_n_tiles,), Int32)
                            sfb_words2 = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                            res2_w0 = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                            res2_w1 = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                            for _nt in cutlass.range_constexpr(fc2_n_tiles):
                                _cnv = tCgC_id[(None, 0, _nt)]
                                _cn = _cnv[0]
                                n_in = _cn[1] - Int32(2) * lane_c + lane_g
                                n_in_arr2[_nt] = n_in
                                sfb_words2[_nt] = ld_shared_u32(
                                    sfb_buf + (n_in << Int32(2))
                                )
                                if cutlass.const_expr(self.w4a8_residual):
                                    res2_w0[_nt] = ld_shared_u32(
                                        res_buf + (n_in << Int32(3))
                                    )
                                    res2_w1[_nt] = ld_shared_u32(
                                        res_buf + (n_in << Int32(3)) + Int32(4)
                                    )
                            rowlo_arr2 = cute.make_rmem_tensor((fc2_m_tiles,), Int32)
                            for _mt in cutlass.range_constexpr(fc2_m_tiles):
                                _c0v = tCgC_id[(None, _mt, 0)]
                                _c0 = _c0v[0]
                                rowlo_arr2[_mt] = _c0[0]

                            for _kb in cutlass.range_constexpr(n_k32_per_tile):
                                b_lo = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                                b_hi = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                                sfb_arr = cute.make_rmem_tensor((fc2_n_tiles,), Uint32)
                                for _nt in cutlass.range_constexpr(fc2_n_tiles):
                                    if cutlass.const_expr(self.w4a8_b_tma):
                                        w4a8_bk2 = (
                                            Int32(_kb)
                                            ^ ((n_in_arr2[_nt] >> Int32(1)) & Int32(3))
                                        ) << Int32(4)
                                    else:
                                        w4a8_bk2 = Int32(_kb * 16)
                                    w_dn = ld_shared_u32(
                                        b_buf
                                        + n_in_arr2[_nt] * Int32(self.w4a8_b_pad)
                                        + w4a8_bk2
                                        + (lane_c << Int32(2))
                                    )
                                    sfb_arr[_nt] = (
                                        sfb_words2[_nt] >> Uint32(_kb * 8)
                                    ) & Uint32(0xFF)
                                    if cutlass.const_expr(self.w4a8_residual):
                                        if cutlass.const_expr(_kb < 2):
                                            res_word = res2_w0[_nt]
                                        else:
                                            res_word = res2_w1[_nt]
                                        res_b = (
                                            res_word
                                            >> (
                                                Uint32((_kb * 2) % 4 * 8)
                                                + (Uint32(lane_c) >> Uint32(1))
                                                * Uint32(8)
                                            )
                                        ) & Uint32(0xFF)
                                        res_h2 = broadcast_f32_to_half2(
                                            fp8_e4m3_to_f32(res_b)
                                        )
                                        blo, bhi = e2m1x8_mul_residual_to_e4m3x8(
                                            w_dn, res_h2
                                        )
                                    else:
                                        blo, bhi = e2m1x8_to_e4m3x8(w_dn)
                                    b_lo[_nt] = blo
                                    b_hi[_nt] = bhi
                                for _mt in cutlass.range_constexpr(fc2_m_tiles):
                                    row_lo = rowlo_arr2[_mt]
                                    row_hi = row_lo + Int32(8)
                                    a_addr_lo = (
                                        sa_flat_addr
                                        + row_lo * Int32(self.tile_shape_mnk[2])
                                        + Int32(_kb * 32)
                                        + Int32(8) * lane_c
                                    )
                                    a_addr_hi = (
                                        sa_flat_addr
                                        + row_hi * Int32(self.tile_shape_mnk[2])
                                        + Int32(_kb * 32)
                                        + Int32(8) * lane_c
                                    )
                                    a0 = ld_shared_u32(a_addr_lo)
                                    a1 = ld_shared_u32(a_addr_hi)
                                    a2 = ld_shared_u32(a_addr_lo + Int32(4))
                                    a3 = ld_shared_u32(a_addr_hi + Int32(4))
                                    sfa_w = Uint32(0)
                                    if lane_c < Int32(2):
                                        sf_row = row_lo
                                        if lane_c == Int32(1):
                                            sf_row = row_hi
                                        sf_idx = sf_row * mx_blocks_tile_p2 + Int32(_kb)
                                        sf_word = ld_shared_u32(
                                            sfa_base_addr
                                            + ((sf_idx >> Int32(2)) << Int32(2))
                                        )
                                        sfa_w = (
                                            sf_word
                                            >> (
                                                (Uint32(sf_idx) & Uint32(3)) * Uint32(8)
                                            )
                                        ) & Uint32(0xFF)
                                    for _nt in cutlass.range_constexpr(fc2_n_tiles):
                                        _fi = ((_mt * fc2_n_tiles) + _nt) * 4
                                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                            w4a8_facc2[_fi],
                                            w4a8_facc2[_fi + 1],
                                            w4a8_facc2[_fi + 2],
                                            w4a8_facc2[_fi + 3],
                                            a0,
                                            a1,
                                            a2,
                                            a3,
                                            b_lo[_nt],
                                            b_hi[_nt],
                                            sfa_w,
                                            sfb_arr[_nt],
                                        )
                                        w4a8_facc2 = (
                                            w4a8_facc2[:_fi]
                                            + (d0, d1, d2, d3)
                                            + w4a8_facc2[_fi + 4 :]
                                        )
                            for _mt in cutlass.range_constexpr(fc2_m_tiles):
                                for _nt in cutlass.range_constexpr(fc2_n_tiles):
                                    _fi = ((_mt * fc2_n_tiles) + _nt) * 4
                                    dn_sl = down_acc[(None, _mt, _nt)]
                                    dn_sl[0] = w4a8_facc2[_fi]
                                    dn_sl[1] = w4a8_facc2[_fi + 1]
                                    dn_sl[2] = w4a8_facc2[_fi + 2]
                                    dn_sl[3] = w4a8_facc2[_fi + 3]
                        if cutlass.const_expr(not self.is_w4a8):
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_next = (
                                    0
                                    if k_block_idx + 1 == num_k_blocks
                                    else k_block_idx + 1
                                )
                                if k_block_idx == num_k_blocks - 1:
                                    phase2_pipeline.consumer_release(phase2_cons_state)
                                    phase2_cons_state.advance()
                                if k_next > 0:
                                    # Only B-side for next k-block (A already in registers)
                                    cute.copy(
                                        smem_copy_B,
                                        csB_phase2[None, None, k_next],
                                        crB[None, None, k_next],
                                    )
                                    f2 = cute.filter_zeros(csSFB_phase2)
                                    f4 = cute.filter_zeros(crSFB)
                                    cute.copy(
                                        smem_copy_SFB,
                                        f2[None, None, k_next],
                                        f4[None, None, k_next],
                                    )
                                for _mt in cutlass.range_constexpr(fc2_m_tiles):
                                    for _nt in cutlass.range_constexpr(fc2_n_tiles):
                                        mma_atom.set(
                                            WarpField.SFA,
                                            tCrSFA[None, _mt, k_block_idx].iterator,
                                        )
                                        mma_atom.set(
                                            WarpField.SFB,
                                            tCrSFB[None, _nt, k_block_idx].iterator,
                                        )
                                        cute.gemm(
                                            mma_atom,
                                            down_acc[None, _mt, _nt],
                                            tCrA[None, _mt, k_block_idx],
                                            tCrB[None, _nt, k_block_idx],
                                            down_acc[None, _mt, _nt],
                                        )

                        # Scatter using precomputed metadata (no redundant gmem loads)
                        tile_n_base_cur = output_tile_idx * Int32(
                            self.tile_shape_mnk[1]
                        )
                        for epi_m in cutlass.range_constexpr(epi_rest_m):
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_n = mma_n_in_epi
                                    mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    down_epi_acc_slice = down_acc[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        if cutlass.const_expr(self.dynamic_down_scale):
                                            tRS_rD_slice[elem_idx] = (
                                                fc2_down_alpha_value
                                                * down_epi_acc_slice[elem_idx]
                                            )
                                        else:
                                            tRS_rD_slice[elem_idx] = (
                                                down_alpha_value
                                                * down_epi_acc_slice[elem_idx]
                                            )

                            acc_vec = tRS_rD.load()
                            acc_vec = acc_vec.to(cutlass.BFloat16)
                            tRS_rD_out.store(acc_vec)
                            epi_buffer = Int32(epi_m) % cute.size(tRS_sD, mode=[3])
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            # Vector scatter reads wider spans from sC than the
                            # scalar path, so wait for all MMA-warps' stores.
                            self.epilog_sync_barrier.arrive_and_wait()
                            rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])

                            # Per-warp scatter: each warp scatters its own quadrant
                            # of sC (64 M-rows × 64 N-cols). No cross-warp read
                            # dependencies, so no pre-scatter barrier is needed.
                            warp_epi_rows = valid_rows - rows_offset - warp_m_base
                            if warp_epi_rows > Int32(64):
                                warp_epi_rows = Int32(64)
                            if warp_epi_rows < Int32(0):
                                warp_epi_rows = Int32(0)

                            tile_vec_cols = Int32(64) // Int32(8)
                            vec_idx = lane_id
                            while vec_idx < warp_epi_rows * tile_vec_cols:
                                local_row = vec_idx // tile_vec_cols
                                local_vec_col = vec_idx - local_row * tile_vec_cols
                                local_col = warp_n_base + local_vec_col * Int32(8)
                                global_col = tile_n_base_cur + local_col
                                cached_row = rows_offset + warp_m_base + local_row
                                tok = ld_shared_i32_relaxed(
                                    scatter_tok_base_addr + cached_row * Int32(4)
                                )
                                wv = ld_shared_f32(
                                    scatter_weight_base_addr + cached_row * Int32(4)
                                )
                                sc_v0 = cutlass.Float32(
                                    sC[warp_m_base + local_row, local_col, epi_buffer]
                                )
                                sc_v1 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(1),
                                        epi_buffer,
                                    ]
                                )
                                sc_v2 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(2),
                                        epi_buffer,
                                    ]
                                )
                                sc_v3 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(3),
                                        epi_buffer,
                                    ]
                                )
                                sc_v4 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(4),
                                        epi_buffer,
                                    ]
                                )
                                sc_v5 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(5),
                                        epi_buffer,
                                    ]
                                )
                                sc_v6 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(6),
                                        epi_buffer,
                                    ]
                                )
                                sc_v7 = cutlass.Float32(
                                    sC[
                                        warp_m_base + local_row,
                                        local_col + Int32(7),
                                        epi_buffer,
                                    ]
                                )
                                if cutlass.const_expr(self.deterministic_output):
                                    scatter_output[tok, global_col] = cutlass.BFloat16(
                                        wv * sc_v0
                                    )
                                    scatter_output[tok, global_col + Int32(1)] = (
                                        cutlass.BFloat16(wv * sc_v1)
                                    )
                                    scatter_output[tok, global_col + Int32(2)] = (
                                        cutlass.BFloat16(wv * sc_v2)
                                    )
                                    scatter_output[tok, global_col + Int32(3)] = (
                                        cutlass.BFloat16(wv * sc_v3)
                                    )
                                    scatter_output[tok, global_col + Int32(4)] = (
                                        cutlass.BFloat16(wv * sc_v4)
                                    )
                                    scatter_output[tok, global_col + Int32(5)] = (
                                        cutlass.BFloat16(wv * sc_v5)
                                    )
                                    scatter_output[tok, global_col + Int32(6)] = (
                                        cutlass.BFloat16(wv * sc_v6)
                                    )
                                    scatter_output[tok, global_col + Int32(7)] = (
                                        cutlass.BFloat16(wv * sc_v7)
                                    )
                                else:
                                    scatter_add_v4_bf16x2(
                                        get_ptr_as_int64(
                                            scatter_output,
                                            tok * scatter_N + global_col,
                                        ),
                                        wv * sc_v0,
                                        wv * sc_v1,
                                        wv * sc_v2,
                                        wv * sc_v3,
                                        wv * sc_v4,
                                        wv * sc_v5,
                                        wv * sc_v6,
                                        wv * sc_v7,
                                    )
                                vec_idx += Int32(self.num_threads_per_warp)

                            # Post-scatter barrier: needed to ensure all warps
                            # finish scatter before next output tile's pipeline ops
                            # (pipeline consumer is collective across all MMA warps).
                            self.epilog_sync_barrier.arrive_and_wait()
                            if cutlass.const_expr(self.is_w4a8):
                                # All MMA warps are past their staging-buffer and
                                # sC reads (the barrier above orders them);
                                # release the buffer use in one bump — once per
                                # PAIR under fused FC2 pairing.
                                if cutlass.const_expr(self.w4a8_fc2_pair):
                                    if (output_tile_idx & Int32(1)) == Int32(1):
                                        if Int32(tidx) == Int32(0):
                                            atomic_add_shared_i32(
                                                w4a8_pipe_addr
                                                + Int32(16)
                                                + (par << Int32(2)),
                                                Int32(self.num_mma_warps),
                                            )
                                else:
                                    if Int32(tidx) == Int32(0):
                                        atomic_add_shared_i32(
                                            w4a8_pipe_addr
                                            + Int32(16)
                                            + (par << Int32(2)),
                                            Int32(self.num_mma_warps),
                                        )

                    # Signal that FC2/scatter no longer needs sA, so the DMA
                    # warp may start the next slice/task's FC1 loads.
                    self.pass_final_barrier.arrive_unaligned()
                    slice_idx += Int32(1)

            elif warp_idx < self.num_mma_warps + self.num_dma_warps:
                task_expert_idx = _ld_shared_i32(ctrl_base_addr + Int32(8))
                task_m_tile_idx = _ld_shared_i32(ctrl_base_addr + Int32(12))
                task_slice_begin_idx = _ld_shared_i32(ctrl_base_addr + Int32(16))
                task_slice_count_val = _ld_shared_i32(ctrl_base_addr + Int32(20))

                # gA and gSFA are tiled in 128-row atoms; a sub-128 MMA tile maps
                # to block (task_m_tile_idx // tiles_per_block). The within-block
                # half is selected in the fragment partition. Identity when
                # tile_m>=128 (tiles_per_block==1).
                sa_block_idx = task_m_tile_idx // Int32(self.sa_tiles_per_block)
                tAgA_mk = tAgA[(None, sa_block_idx, None, Int32(0))]
                sfa_block_idx = task_m_tile_idx // Int32(self.sfa_tiles_per_block)
                tAgSFA_mk = tAgSFA[(None, sfa_block_idx, None, Int32(0))]
                slice_idx = Int32(0)
                while slice_idx < task_slice_count_val:
                    intermediate_slice = task_slice_begin_idx + slice_idx

                    # FC1 producer slice. Gated activation packs [up, gate]
                    # across N; relu2 uses a single FC1 pass.
                    tBgB_w13_up_nk = tBgB_w13[
                        (None, intermediate_slice, None, task_expert_idx)
                    ]
                    tBgSFB_w13_up_nk = tBgSFB_w13[
                        (None, intermediate_slice, None, task_expert_idx)
                    ]
                    gate_slice_idx = intermediate_slice
                    if self.is_gated:
                        gate_slice_idx = intermediate_slice + gate_tile_cnt
                    if cutlass.const_expr(self.swap_ab):
                        # swap: atom-lo is the 128-row atom holding gate row n
                        # (n//128 + slice); the consumer reads the gate's 32-int
                        # subs from it at within-atom offset gate_lo_sub.
                        gate_slice_idx = intermediate_slice + Int32(gate_lo_off)
                    tBgB_w13_gate_nk = tBgB_w13[
                        (None, gate_slice_idx, None, task_expert_idx)
                    ]
                    tBgSFB_w13_gate_nk = tBgSFB_w13[
                        (None, gate_slice_idx, None, task_expert_idx)
                    ]
                    if cutlass.const_expr(self.swap_ab and gate_lo_sub > 0):
                        gate_hi_idx = intermediate_slice + Int32(gate_lo_off + 1)
                        tBgB_w13_gate_hi_nk = tBgB_w13[
                            (None, gate_hi_idx, None, task_expert_idx)
                        ]
                        tBgSFB_w13_gate_hi_nk = tBgSFB_w13[
                            (None, gate_hi_idx, None, task_expert_idx)
                        ]

                    # ---- FC1 gate pass ----
                    if cutlass.const_expr(self.is_w4a8):
                        # w4a8 producer: the DMA warp cp.asyncs each k-tile's
                        # operands into the staging buffers and publishes them
                        # via per-buffer ready epochs; the MMA warps consume by
                        # spinning on the epoch (no CTA barriers in the k loop)
                        # and acknowledge via per-buffer done counts. Epochs are
                        # monotonic per task, so buffer reuse across passes,
                        # FC2, and slices is ordered by the flags alone.
                        w4a8_KT = fc1_k_tile_cnt // 2
                        w4a8_passes = self.w4a8_fc1_windows_per_slice
                        m_row_base_p = task_m_tile_idx * Int32(self.tile_shape_mnk[0])
                        w4a8_pend_par = Int32(0)
                        w4a8_pend_epoch = Int32(0)
                        # With two producer warps, epochs split by parity:
                        # warp 0 stages odd epochs (pars 0,2), warp 1 even
                        # (pars 1,3) — disjoint buffers, independent FIFOs.
                        w4a8_warp_off = warp_idx - Int32(self.tma_load_warp_id)
                        asc_base_p = Int64(m_row_base_p) * Int64(a_mx_per_row)
                        cur_slice_p = task_slice_begin_idx + slice_idx
                        for _pp in cutlass.range_constexpr(w4a8_passes):
                            if cutlass.const_expr(_pp == 0):
                                p_n_tile = cur_slice_p
                                if cutlass.const_expr(self.is_gated):
                                    p_n_tile = cur_slice_p + gate_tile_cnt
                            else:
                                p_n_tile = cur_slice_p
                            if cutlass.const_expr(self.w4a8_fused):
                                # Fused: this window also stages the up half.
                                pu_n_tile = cur_slice_p
                                nu_base_p = pu_n_tile * Int32(self.tile_shape_mnk[1])
                            n_base_p = p_n_tile * Int32(self.tile_shape_mnk[1])
                            b_row_u32p = b_w13_u32.shape[1]
                            b_base_p = Int64(task_expert_idx) * Int64(
                                b_w13_u32.shape[0] * b_row_u32p
                            ) + Int64(n_base_p) * Int64(b_row_u32p)
                            sfb_row_p = sfb_w13_mx.shape[1]
                            sfb_base_p = Int64(task_expert_idx) * Int64(
                                sfb_w13_mx.shape[0] * sfb_row_p
                            ) + Int64(n_base_p) * Int64(sfb_row_p)
                            if cutlass.const_expr(self.w4a8_fused):
                                b_base_pu = Int64(task_expert_idx) * Int64(
                                    b_w13_u32.shape[0] * b_row_u32p
                                ) + Int64(nu_base_p) * Int64(b_row_u32p)
                                sfb_base_pu = Int64(task_expert_idx) * Int64(
                                    sfb_w13_mx.shape[0] * sfb_row_p
                                ) + Int64(nu_base_p) * Int64(sfb_row_p)
                            if cutlass.const_expr(self.w4a8_residual):
                                res_row_p = w13_residual.shape[1]
                                res_base_p = Int64(task_expert_idx) * Int64(
                                    w13_residual.shape[0] * res_row_p
                                ) + Int64(n_base_p) * Int64(res_row_p)
                                if cutlass.const_expr(self.w4a8_fused):
                                    res_base_pu = Int64(task_expert_idx) * Int64(
                                        w13_residual.shape[0] * res_row_p
                                    ) + Int64(nu_base_p) * Int64(res_row_p)
                            w4a8_uses = w4a8_passes * w4a8_KT + (
                                output_tile_cnt // 2
                                if self.w4a8_fc2_pair
                                else output_tile_cnt
                            )
                            for _pkt in range(w4a8_KT):
                                # Unified buffer-use epochs: gate tiles, up
                                # tiles, then FC2 tiles share one sequence, so
                                # every cross-phase buffer reuse is ordered by
                                # the same ready/done flags.
                                epoch = (
                                    slice_idx * Int32(w4a8_uses)
                                    + Int32(_pp * w4a8_KT)
                                    + _pkt
                                    + Int32(1)
                                )
                                par = (epoch - Int32(1)) & Int32(self.w4a8_depth - 1)
                                w4a8_owned = Int32(1)
                                if cutlass.const_expr(self.num_dma_warps > 1):
                                    w4a8_owned = Int32(
                                        ((epoch - Int32(1)) & Int32(1)) == w4a8_warp_off
                                    )
                                if w4a8_owned > Int32(0):
                                    need = (
                                        (epoch - Int32(1))
                                        >> Int32(self.w4a8_depth_log2)
                                    ) * Int32(self.num_mma_warps)
                                    _spin_wait_shared_ge_i32(
                                        w4a8_pipe_addr + Int32(16) + (par << Int32(2)),
                                        need,
                                    )
                                    if cutlass.const_expr(self.w4a8_b_tma):
                                        # B granule through TMA: one box per
                                        # 8KB half; granule bytes complete on
                                        # this par's mbarrier. Region/stage
                                        # selection mirrors the consumers'
                                        # b_buf math exactly.
                                        if lane_id == Int32(0):
                                            cute.arch.mbarrier_arrive_and_expect_tx(
                                                w4a8_tma_mbar_ptr + par,
                                                Int32(
                                                    2 * _W4A8_TMA_TILE_BYTES
                                                    if self.w4a8_fused
                                                    else _W4A8_TMA_TILE_BYTES
                                                ),
                                            )
                                        if cutlass.const_expr(self.w4a8_fused):
                                            if par == Int32(0):
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_gate_nk[(None, _pkt)],
                                                    tBsB_w13[(None, 0)],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_up_nk[(None, _pkt)],
                                                    tBsB_w13[(None, 1)],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                            else:
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_gate_nk[(None, _pkt)],
                                                    tBsB_w13_up[(None, 0)],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_up_nk[(None, _pkt)],
                                                    tBsB_w13_up[(None, 1)],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                        else:
                                            # Non-fused small (relu2): region
                                            # par>>1, stage par&1, single box.
                                            if (par >> Int32(1)) == Int32(0):
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_gate_nk[(None, _pkt)],
                                                    tBsB_w13[(None, par & Int32(1))],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                            else:
                                                cute.copy(
                                                    tma_b_w13,
                                                    tBgB_w13_gate_nk[(None, _pkt)],
                                                    tBsB_w13_up[(None, par & Int32(1))],
                                                    tma_bar_ptr=w4a8_tma_mbar_ptr + par,
                                                )
                                    else:
                                        if cutlass.const_expr(self.w4a8_depth == 4):
                                            w4a8_b_dst = (
                                                w4a8_sb0
                                                + (w4a8_sb1 - w4a8_sb0)
                                                * (par >> Int32(1))
                                                + (par & Int32(1)) * Int32(128 * 64)
                                            )
                                        else:
                                            w4a8_b_dst = (
                                                w4a8_sb0 + (w4a8_sb1 - w4a8_sb0) * par
                                            )
                                        _w4a8_stage_b_tile(
                                            b_w13_u32,
                                            w4a8_b_dst,
                                            b_base_p + Int64(_pkt) * Int64(16),
                                            Int32(b_row_u32p),
                                            Int32(lane_id),
                                            32,
                                            self.w4a8_b_pad,
                                        )
                                        if cutlass.const_expr(self.w4a8_fused):
                                            _w4a8_stage_b_tile(
                                                b_w13_u32,
                                                w4a8_b_dst + Int32(128 * 64),
                                                b_base_pu + Int64(_pkt) * Int64(16),
                                                Int32(b_row_u32p),
                                                Int32(lane_id),
                                                32,
                                                self.w4a8_b_pad,
                                            )
                                    _w4a8_stage_a_tile(
                                        pa_u32,
                                        w4a8_sa0
                                        + par * Int32(self.tile_shape_mnk[0] * 128),
                                        Int64(m_row_base_p) * Int64(a_u32_per_row)
                                        + Int64(_pkt) * Int64(32),
                                        a_u32_per_row,
                                        Int32(lane_id),
                                        32,
                                        self.tile_shape_mnk[0],
                                    )
                                    if cutlass.const_expr(self.w4a8_fused):
                                        w4a8_sfb_dst = w4a8_sfbb + (par << Int32(10))
                                    else:
                                        w4a8_sfb_dst = w4a8_sfbb + (par << Int32(9))
                                    _w4a8_stage_bytes4(
                                        sfb_w13_mx,
                                        w4a8_sfb_dst,
                                        sfb_base_p + Int64(_pkt) * Int64(4),
                                        Int32(sfb_row_p),
                                        Int32(lane_id),
                                        32,
                                    )
                                    if cutlass.const_expr(self.w4a8_fused):
                                        _w4a8_stage_bytes4(
                                            sfb_w13_mx,
                                            w4a8_sfb_dst + Int32(512),
                                            sfb_base_pu + Int64(_pkt) * Int64(4),
                                            Int32(sfb_row_p),
                                            Int32(lane_id),
                                            32,
                                        )
                                    if cutlass.const_expr(self.w4a8_residual):
                                        if cutlass.const_expr(self.w4a8_fused):
                                            w4a8_res_dst = w4a8_res0 + (
                                                par << Int32(11)
                                            )
                                        else:
                                            w4a8_res_dst = w4a8_res0 + (
                                                par << Int32(10)
                                            )
                                        _w4a8_stage_bytes8(
                                            w13_residual,
                                            w4a8_res_dst,
                                            res_base_p + Int64(_pkt) * Int64(8),
                                            Int32(res_row_p),
                                            Int32(lane_id),
                                            32,
                                        )
                                        if cutlass.const_expr(self.w4a8_fused):
                                            _w4a8_stage_bytes8(
                                                w13_residual,
                                                w4a8_res_dst + Int32(1024),
                                                res_base_pu + Int64(_pkt) * Int64(8),
                                                Int32(res_row_p),
                                                Int32(lane_id),
                                                32,
                                            )
                                    _w4a8_stage_bytes4(
                                        scale_storage,
                                        (
                                            w4a8_resb
                                            + par * Int32(self.tile_shape_mnk[0] * 4)
                                        )
                                        if cutlass.const_expr(self.w4a8_small)
                                        else (sfa_base_addr + (par << Int32(9))),
                                        asc_base_p + Int64(_pkt) * Int64(4),
                                        a_mx_per_row,
                                        Int32(lane_id),
                                        32,
                                        self.tile_shape_mnk[0],
                                    )
                                    cute.arch.cp_async_commit_group()
                                    # Deferred publish: drain to one outstanding
                                    # group, release-publish the PREVIOUS window;
                                    # the current window's copies overlap the
                                    # consumers' work on the previous one. Under
                                    # TMA-B the pending granule's mbarrier is
                                    # waited too (whole warp; phase bit per par).
                                    if cutlass.const_expr(self.w4a8_b_tma):
                                        if w4a8_pend_epoch > Int32(0):
                                            cute.arch.mbarrier_wait(
                                                w4a8_tma_mbar_ptr + w4a8_pend_par,
                                                phase=(w4a8_mbar_phase >> w4a8_pend_par)
                                                & Int32(1),
                                            )
                                            w4a8_mbar_phase = w4a8_mbar_phase ^ (
                                                Int32(1) << w4a8_pend_par
                                            )
                                    cute.arch.cp_async_wait_group(1)
                                    cute.arch.fence_proxy("async.shared", space="cta")
                                    if lane_id == Int32(0):
                                        if w4a8_pend_epoch > Int32(0):
                                            _st_shared_release_i32(
                                                w4a8_pipe_addr
                                                + (w4a8_pend_par << Int32(2)),
                                                w4a8_pend_epoch,
                                            )
                                    w4a8_pend_par = par
                                    w4a8_pend_epoch = epoch
                            # (The original pass_gate wait below still pairs
                            # with the MMA warps' arrive; buffer reuse across
                            # passes is ordered by the done-count epochs.)
                        # ---- FC2 B_down staging (per output tile) ----
                        dn_row_u32p = b_down_u32.shape[1]
                        dn_base_p = Int64(task_expert_idx) * Int64(
                            b_down_u32.shape[0] * dn_row_u32p
                        ) + Int64(cur_slice_p) * Int64(16)
                        dnsf_row_p = sfb_down_mx.shape[1]
                        dnsf_base_p = Int64(task_expert_idx) * Int64(
                            sfb_down_mx.shape[0] * dnsf_row_p
                        ) + Int64(cur_slice_p) * Int64(4)
                        if cutlass.const_expr(self.w4a8_residual):
                            dnres_row_p = down_residual.shape[1]
                            dnres_base_p = Int64(task_expert_idx) * Int64(
                                down_residual.shape[0] * dnres_row_p
                            ) + Int64(cur_slice_p) * Int64(8)
                        w4a8_fc2_step = 2 if self.w4a8_fc2_pair else 1
                        for _pt in range(0, output_tile_cnt, w4a8_fc2_step):
                            epoch2 = (
                                slice_idx * Int32(w4a8_uses)
                                + Int32(w4a8_passes * w4a8_KT)
                                + (
                                    (_pt >> Int32(1))
                                    if cutlass.const_expr(self.w4a8_fc2_pair)
                                    else _pt
                                )
                                + Int32(1)
                            )
                            par2 = (epoch2 - Int32(1)) & Int32(self.w4a8_depth - 1)
                            w4a8_owned2 = Int32(1)
                            if cutlass.const_expr(self.num_dma_warps > 1):
                                w4a8_owned2 = Int32(
                                    ((epoch2 - Int32(1)) & Int32(1)) == w4a8_warp_off
                                )
                            if w4a8_owned2 > Int32(0):
                                need2 = (
                                    (epoch2 - Int32(1)) >> Int32(self.w4a8_depth_log2)
                                ) * Int32(self.num_mma_warps)
                                _spin_wait_shared_ge_i32(
                                    w4a8_pipe_addr + Int32(16) + (par2 << Int32(2)),
                                    need2,
                                )
                                row_off_p = Int64(_pt) * Int64(128)
                                if cutlass.const_expr(self.w4a8_b_tma):
                                    if lane_id == Int32(0):
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            w4a8_tma_mbar_ptr + par2,
                                            Int32(
                                                2 * _W4A8_TMA_TILE_BYTES
                                                if self.w4a8_fc2_pair
                                                else _W4A8_TMA_TILE_BYTES
                                            ),
                                        )
                                    if cutlass.const_expr(self.w4a8_fc2_pair):
                                        if par2 == Int32(0):
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt,
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down[(None, 0)],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt + Int32(1),
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down[(None, 1)],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                        else:
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt,
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down_up[(None, 0)],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt + Int32(1),
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down_up[(None, 1)],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                    else:
                                        if (par2 >> Int32(1)) == Int32(0):
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt,
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down[(None, par2 & Int32(1))],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                        else:
                                            cute.copy(
                                                tma_b_down,
                                                tBgB_down[
                                                    (
                                                        None,
                                                        _pt,
                                                        cur_slice_p,
                                                        task_expert_idx,
                                                    )
                                                ],
                                                tBsB_down_up[(None, par2 & Int32(1))],
                                                tma_bar_ptr=w4a8_tma_mbar_ptr + par2,
                                            )
                                else:
                                    if cutlass.const_expr(self.w4a8_depth == 4):
                                        w4a8_b2_dst = (
                                            w4a8_sb0
                                            + (w4a8_sb1 - w4a8_sb0) * (par2 >> Int32(1))
                                            + (par2 & Int32(1)) * Int32(128 * 64)
                                        )
                                    else:
                                        w4a8_b2_dst = (
                                            w4a8_sb0 + (w4a8_sb1 - w4a8_sb0) * par2
                                        )
                                    _w4a8_stage_b_tile(
                                        b_down_u32,
                                        w4a8_b2_dst,
                                        dn_base_p + row_off_p * Int64(dn_row_u32p),
                                        Int32(dn_row_u32p),
                                        Int32(lane_id),
                                        32,
                                        self.w4a8_b_pad,
                                    )
                                    if cutlass.const_expr(self.w4a8_fc2_pair):
                                        row_off_p2 = row_off_p + Int64(128)
                                        _w4a8_stage_b_tile(
                                            b_down_u32,
                                            w4a8_b2_dst + Int32(128 * 64),
                                            dn_base_p + row_off_p2 * Int64(dn_row_u32p),
                                            Int32(dn_row_u32p),
                                            Int32(lane_id),
                                            32,
                                            self.w4a8_b_pad,
                                        )
                                if cutlass.const_expr(self.w4a8_fc2_pair):
                                    w4a8_sfb2_dst = w4a8_sfbb + (par2 << Int32(10))
                                else:
                                    w4a8_sfb2_dst = w4a8_sfbb + (par2 << Int32(9))
                                _w4a8_stage_bytes4(
                                    sfb_down_mx,
                                    w4a8_sfb2_dst,
                                    dnsf_base_p + row_off_p * Int64(dnsf_row_p),
                                    Int32(dnsf_row_p),
                                    Int32(lane_id),
                                    32,
                                )
                                if cutlass.const_expr(self.w4a8_fc2_pair):
                                    _w4a8_stage_bytes4(
                                        sfb_down_mx,
                                        w4a8_sfb2_dst + Int32(512),
                                        dnsf_base_p
                                        + (row_off_p + Int64(128)) * Int64(dnsf_row_p),
                                        Int32(dnsf_row_p),
                                        Int32(lane_id),
                                        32,
                                    )
                                if cutlass.const_expr(self.w4a8_residual):
                                    if cutlass.const_expr(self.w4a8_fc2_pair):
                                        w4a8_res2_dst = w4a8_res2 + (par2 << Int32(11))
                                    else:
                                        w4a8_res2_dst = w4a8_res2 + (par2 << Int32(10))
                                    _w4a8_stage_bytes8(
                                        down_residual,
                                        w4a8_res2_dst,
                                        dnres_base_p + row_off_p * Int64(dnres_row_p),
                                        Int32(dnres_row_p),
                                        Int32(lane_id),
                                        32,
                                    )
                                    if cutlass.const_expr(self.w4a8_fc2_pair):
                                        _w4a8_stage_bytes8(
                                            down_residual,
                                            w4a8_res2_dst + Int32(1024),
                                            dnres_base_p
                                            + (row_off_p + Int64(128))
                                            * Int64(dnres_row_p),
                                            Int32(dnres_row_p),
                                            Int32(lane_id),
                                            32,
                                        )
                                cute.arch.cp_async_commit_group()
                                if cutlass.const_expr(self.w4a8_b_tma):
                                    if w4a8_pend_epoch > Int32(0):
                                        cute.arch.mbarrier_wait(
                                            w4a8_tma_mbar_ptr + w4a8_pend_par,
                                            phase=(w4a8_mbar_phase >> w4a8_pend_par)
                                            & Int32(1),
                                        )
                                        w4a8_mbar_phase = w4a8_mbar_phase ^ (
                                            Int32(1) << w4a8_pend_par
                                        )
                                cute.arch.cp_async_wait_group(1)
                                cute.arch.fence_proxy("async.shared", space="cta")
                                if lane_id == Int32(0):
                                    if w4a8_pend_epoch > Int32(0):
                                        _st_shared_release_i32(
                                            w4a8_pipe_addr
                                            + (w4a8_pend_par << Int32(2)),
                                            w4a8_pend_epoch,
                                        )
                                w4a8_pend_par = par2
                                w4a8_pend_epoch = epoch2
                        if cutlass.const_expr(self.w4a8_b_tma):
                            if w4a8_pend_epoch > Int32(0):
                                cute.arch.mbarrier_wait(
                                    w4a8_tma_mbar_ptr + w4a8_pend_par,
                                    phase=(w4a8_mbar_phase >> w4a8_pend_par) & Int32(1),
                                )
                                w4a8_mbar_phase = w4a8_mbar_phase ^ (
                                    Int32(1) << w4a8_pend_par
                                )
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if lane_id == Int32(0):
                            if w4a8_pend_epoch > Int32(0):
                                _st_shared_release_i32(
                                    w4a8_pipe_addr + (w4a8_pend_par << Int32(2)),
                                    w4a8_pend_epoch,
                                )
                        w4a8_pend_epoch = Int32(0)
                    prod_state.reset_count()
                    for k_tile in range(  # type: ignore[call-overload]
                        0, fc1_k_tile_cnt if not self.is_w4a8 else 0, 1, unroll=4
                    ):
                        ml_pipeline.producer_acquire(prod_state)
                        cute.copy(
                            tma_a,
                            tAgA_mk[(None, k_tile)],
                            tAsA[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        cute.copy(
                            tma_sfa,
                            tAgSFA_mk[(None, k_tile)],
                            tAsSFA[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        cute.copy(
                            tma_b_w13,
                            tBgB_w13_gate_nk[(None, k_tile)],
                            tBsB_w13[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        cute.copy(
                            tma_sfb_w13,
                            tBgSFB_w13_gate_nk[(None, k_tile)],
                            tBsSFB_w13[(None, prod_state.index)],
                            tma_bar_ptr=ml_pipeline.producer_get_barrier(prod_state),
                        )
                        if cutlass.const_expr(self.swap_ab and gate_lo_sub > 0):
                            # atom-hi (the next 128-row atom) staged in sB_up/
                            # sSFB_up under ml_pipeline. Safe to share with the up
                            # pass: within a slice pass_gate_barrier orders the
                            # gate GEMM read before the up pass overwrite; across
                            # slices the next gate pass also rewrites sA (read by
                            # this slice's FC2), so the ml_pipeline sA dependency
                            # serializes it -- no extra buffer needed (verified
                            # bit-identical to a dedicated buffer). ml_tx_count
                            # accounts for the extra bytes.
                            cute.copy(
                                tma_b_w13,
                                tBgB_w13_gate_hi_nk[(None, k_tile)],
                                tBsB_w13_up[(None, prod_state.index)],
                                tma_bar_ptr=ml_pipeline.producer_get_barrier(
                                    prod_state
                                ),
                            )
                            cute.copy(
                                tma_sfb_w13,
                                tBgSFB_w13_gate_hi_nk[(None, k_tile)],
                                tBsSFB_w13_up[(None, prod_state.index)],
                                tma_bar_ptr=ml_pipeline.producer_get_barrier(
                                    prod_state
                                ),
                            )
                        ml_pipeline.producer_commit(prod_state)
                        prod_state.advance()

                    # Wait for the MMA warps to finish the FC1 gate/only pass
                    # before reusing sA/sB/sSFA/sSFB for up/FC2 staging. Under
                    # w4a8 the staging-flag epochs order every buffer reuse, so
                    # the producer only contributes its arrival and keeps
                    # streaming the up/FC2 tiles (the wait would idle it for
                    # the MMA warps' whole gate pass).
                    if cutlass.const_expr(self.is_w4a8):
                        self.pass_gate_barrier.arrive_unaligned()
                    else:
                        self.pass_gate_barrier.wait_unaligned()

                    if self.is_gated:
                        # ---- FC1 up pass ----
                        up_prod_state.reset_count()
                        for k_tile in range(  # type: ignore[call-overload]
                            0, fc1_k_tile_cnt if not self.is_w4a8 else 0, 1, unroll=4
                        ):
                            up_pipeline.producer_acquire(up_prod_state)
                            cute.copy(
                                tma_a,
                                tAgA_mk[(None, k_tile)],
                                tAsA[(None, up_prod_state.index)],
                                tma_bar_ptr=up_pipeline.producer_get_barrier(
                                    up_prod_state
                                ),
                            )
                            cute.copy(
                                tma_b_w13,
                                tBgB_w13_up_nk[(None, k_tile)],
                                tBsB_w13_up[(None, up_prod_state.index)],
                                tma_bar_ptr=up_pipeline.producer_get_barrier(
                                    up_prod_state
                                ),
                            )
                            cute.copy(
                                tma_sfa,
                                tAgSFA_mk[(None, k_tile)],
                                tAsSFA[(None, up_prod_state.index)],
                                tma_bar_ptr=up_pipeline.producer_get_barrier(
                                    up_prod_state
                                ),
                            )
                            cute.copy(
                                tma_sfb_w13,
                                tBgSFB_w13_up_nk[(None, k_tile)],
                                tBsSFB_w13_up[(None, up_prod_state.index)],
                                tma_bar_ptr=up_pipeline.producer_get_barrier(
                                    up_prod_state
                                ),
                            )
                            up_pipeline.producer_commit(up_prod_state)
                            up_prod_state.advance()

                    # ---- FC2 B_down loads: continuous pipeline ----
                    # No barrier needed: sB/sSFB are free (gate done, up uses
                    # sB_up/sSFB_up). phase2_pipeline handles data availability.
                    # intermediate_slice selects the K-tile of GEMM2 (FC1 output N-tile
                    # = GEMM2 K-tile since intermediate dim is the reduction dim).
                    # Load ALL FC2 tiles continuously once stage1 no longer needs
                    # the gate staging buffers.
                    phase2_prod_state.reset_count()
                    for output_tile_idx in range(  # type: ignore[call-overload]
                        0, output_tile_cnt if not self.is_w4a8 else 0, 1, unroll=4
                    ):
                        phase2_pipeline.producer_acquire(phase2_prod_state)
                        cute.copy(
                            tma_b_down,
                            tBgB_down[
                                (
                                    None,
                                    output_tile_idx,
                                    intermediate_slice,
                                    task_expert_idx,
                                )
                            ],
                            tBsB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        cute.copy(
                            tma_sfb_down,
                            tBgSFB_down[
                                (
                                    None,
                                    output_tile_idx,
                                    intermediate_slice,
                                    task_expert_idx,
                                )
                            ],
                            tBsSFB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        phase2_pipeline.producer_commit(phase2_prod_state)
                        phase2_prod_state.advance()

                    # Ensure MMA warps finish FC2/scatter before DMA starts the
                    # next slice/task's FC1 loads into shared A buffers.
                    self.pass_final_barrier.wait_unaligned()
                    slice_idx += Int32(1)

        if cutlass.const_expr(not self.is_w4a8):
            if warp_idx == self.tma_load_warp_id:
                ml_pipeline.producer_tail(prod_state)
                if self.is_gated:
                    up_pipeline.producer_tail(up_prod_state)
                phase2_pipeline.producer_tail(phase2_prod_state)
        return


__all__ = ["MoEDynamicKernelBackend"]
