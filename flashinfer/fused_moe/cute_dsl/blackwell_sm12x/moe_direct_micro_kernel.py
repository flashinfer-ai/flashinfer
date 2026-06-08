"""
MoEDirectMicroKernel - direct routed NVFP4/W4A16 micro MoE kernel for SM120/SM121.

Ported from the b12x kernel library to FlashInfer.

This direct micro backend is the low-latency path for very small routed decode
batches. Unlike the compact static/dynamic kernels, it performs the FC1 and FC2
work with warp-level direct dot products instead of staging full MMA tiles.
The W4A16 micro kernel subclasses this implementation with BF16 intermediate
storage enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Uint32

from cutlass.cutlass_dsl import Int32, Int64

from flashinfer.cute_dsl.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
)
from flashinfer.cute_dsl.fp4_common import (
    atomic_add_global_i32,
    cvt_e4m3_to_f32_via_f16,
    cvt_e4m3x4_to_f32x4,
    cvt_f32_to_e4m3,
    fmax_f32,
    fp4_dot4_sum,
    fp4_dot4_sum_f32acc,
    fp4_dot8_sum,
    fp4_dot8_sum_f32acc,
    get_ptr_as_int64,
    ld_global_acquire_i32,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    nvfp4_scale_from_amax,
    pack_f32x2_to_f16x2,
    quant_dequant_2,
    spin_wait_global_eq_i32,
    st_global_i32,
    st_global_release_i32,
    threadfence,
    warp_reduce,
)


_BLOCK_SIZE = 16
_FP8_E4M3_MAX = 448.0
_FC2_TILE_RECIP_GS_NUM = 6.0 * _FP8_E4M3_MAX
_NUM_WARPS = 16
_BLOCK_DIM = _NUM_WARPS * 32
_K_PER_CTA = 16
_MAX_DIRECT_K_SEGMENTS = 12


def _direct_k_segments_supported(k_segments: int) -> bool:
    return 1 <= int(k_segments) <= _MAX_DIRECT_K_SEGMENTS


def _align_up(value: int, align: int) -> int:
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _fc1_chunks_for_m(m: int, n: int) -> int:
    rows_per_warp = max(1, int(m))
    rows_per_chunk = max(_BLOCK_SIZE, _NUM_WARPS * rows_per_warp)
    chunks = max(1, int(n) // rows_per_chunk)
    while chunks > 1:
        i_chunk = int(n) // chunks
        if int(n) % chunks == 0 and i_chunk % _BLOCK_SIZE == 0:
            return chunks
        chunks -= 1
    return 1


@dataclass(frozen=True)
class _ShapeConfig:
    """Compile-time shape constants."""

    k_dim: int
    n: int
    two_n: int
    weight_E: int
    num_topk: int
    k_half: int
    n_half: int
    w1_sf_rows: int
    w1_sf_cols: int
    w2_sf_rows: int
    w2_sf_cols: int
    k_segments: int
    fc1_chunks: int
    fc1_chunks_per_block: int
    i_chunk: int
    rows_per_warp_fc1: int
    smem_xh_stride: int
    smem_xh_size: int
    inter_blocks: int
    inter_u32: int
    fc2_n_chunks: int


def _make_shape_config(
    *,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    weight_E: int,
    is_gated: bool = True,
) -> _ShapeConfig:
    k_half = k // 2
    n_half = n // 2
    two_n = 2 * n if is_gated else n
    w1_sf_rows = _align_up(two_n, 128)
    w1_sf_cols = _align_up(k // _BLOCK_SIZE, 4)
    w2_sf_rows = _align_up(k, 128)
    w2_sf_cols = _align_up(n // _BLOCK_SIZE, 4)
    k_segments = k // (32 * _BLOCK_SIZE)
    fc1_chunks = _fc1_chunks_for_m(m, n)
    fc1_chunks_per_block = 16  # chunks per 128-wide swizzle block
    i_chunk = n // fc1_chunks
    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
    smem_xh_stride = k_segments * (_BLOCK_SIZE // 2) + 1
    smem_xh_size = k_half + (k // (_BLOCK_SIZE * 8))
    inter_blocks = i_chunk // _BLOCK_SIZE
    fc2_n_chunks = (n // 2 + 127) // 128
    inter_u32 = fc2_n_chunks * 128 * num_topk
    return _ShapeConfig(
        k_dim=k,
        n=n,
        two_n=two_n,
        weight_E=weight_E,
        num_topk=num_topk,
        k_half=k_half,
        n_half=n_half,
        w1_sf_rows=w1_sf_rows,
        w1_sf_cols=w1_sf_cols,
        w2_sf_rows=w2_sf_rows,
        w2_sf_cols=w2_sf_cols,
        k_segments=k_segments,
        fc1_chunks=fc1_chunks,
        fc1_chunks_per_block=fc1_chunks_per_block,
        i_chunk=i_chunk,
        rows_per_warp_fc1=rows_per_warp_fc1,
        smem_xh_stride=smem_xh_stride,
        smem_xh_size=smem_xh_size,
        inter_blocks=inter_blocks,
        inter_u32=inter_u32,
        fc2_n_chunks=fc2_n_chunks,
    )


def _remake_shape_config_fc1(cfg: _ShapeConfig, fc1_chunks: int) -> _ShapeConfig:
    i_chunk = cfg.n // fc1_chunks
    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
    smem_xh_stride = cfg.k_segments * (_BLOCK_SIZE // 2) + 1
    smem_xh_size = cfg.k_half + (cfg.k_dim // (_BLOCK_SIZE * 8))
    inter_blocks = i_chunk // _BLOCK_SIZE
    fc2_n_chunks = (cfg.n // 2 + 127) // 128
    inter_u32 = fc2_n_chunks * 128 * cfg.num_topk
    return _ShapeConfig(
        k_dim=cfg.k_dim,
        n=cfg.n,
        two_n=cfg.two_n,
        weight_E=cfg.weight_E,
        num_topk=cfg.num_topk,
        k_half=cfg.k_half,
        n_half=cfg.n_half,
        w1_sf_rows=cfg.w1_sf_rows,
        w1_sf_cols=cfg.w1_sf_cols,
        w2_sf_rows=cfg.w2_sf_rows,
        w2_sf_cols=cfg.w2_sf_cols,
        k_segments=cfg.k_segments,
        fc1_chunks=fc1_chunks,
        fc1_chunks_per_block=cfg.fc1_chunks_per_block,
        i_chunk=i_chunk,
        rows_per_warp_fc1=rows_per_warp_fc1,
        smem_xh_stride=smem_xh_stride,
        smem_xh_size=smem_xh_size,
        inter_blocks=inter_blocks,
        inter_u32=inter_u32,
        fc2_n_chunks=fc2_n_chunks,
    )


@cute.jit
def _block_dot_hfma2(
    u_a: Uint32,
    u_b: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Float32:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    xh4 = Uint32(smem_xh[xh_base + Int32(4)])
    xh5 = Uint32(smem_xh[xh_base + Int32(5)])
    xh6 = Uint32(smem_xh[xh_base + Int32(6)])
    xh7 = Uint32(smem_xh[xh_base + Int32(7)])
    return fp4_dot8_sum(u_a, u_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)


@cute.jit
def _block_dot_hfma2_pair(
    up_a: Uint32,
    up_b: Uint32,
    gate_a: Uint32,
    gate_b: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Tuple[Float32, Float32]:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    xh4 = Uint32(smem_xh[xh_base + Int32(4)])
    xh5 = Uint32(smem_xh[xh_base + Int32(5)])
    xh6 = Uint32(smem_xh[xh_base + Int32(6)])
    xh7 = Uint32(smem_xh[xh_base + Int32(7)])
    up = fp4_dot8_sum(up_a, up_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)
    gate = fp4_dot8_sum(gate_a, gate_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)
    return up, gate


@cute.jit
def _block_dot_hfma2_f32acc(
    u_a: Uint32,
    u_b: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Float32:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    xh4 = Uint32(smem_xh[xh_base + Int32(4)])
    xh5 = Uint32(smem_xh[xh_base + Int32(5)])
    xh6 = Uint32(smem_xh[xh_base + Int32(6)])
    xh7 = Uint32(smem_xh[xh_base + Int32(7)])
    return fp4_dot8_sum_f32acc(u_a, u_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)


@cute.jit
def _block_dot_hfma2_pair_f32acc(
    up_a: Uint32,
    up_b: Uint32,
    gate_a: Uint32,
    gate_b: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Tuple[Float32, Float32]:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    xh4 = Uint32(smem_xh[xh_base + Int32(4)])
    xh5 = Uint32(smem_xh[xh_base + Int32(5)])
    xh6 = Uint32(smem_xh[xh_base + Int32(6)])
    xh7 = Uint32(smem_xh[xh_base + Int32(7)])
    up = fp4_dot8_sum_f32acc(up_a, up_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)
    gate = fp4_dot8_sum_f32acc(gate_a, gate_b, xh0, xh1, xh2, xh3, xh4, xh5, xh6, xh7)
    return up, gate


@cute.jit
def _block_dot4(
    u_val: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Float32:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    return fp4_dot4_sum(u_val, xh0, xh1, xh2, xh3)


@cute.jit
def _block_dot4_pair(
    up_val: Uint32,
    gate_val: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Tuple[Float32, Float32]:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    up = fp4_dot4_sum(up_val, xh0, xh1, xh2, xh3)
    gate = fp4_dot4_sum(gate_val, xh0, xh1, xh2, xh3)
    return up, gate


@cute.jit
def _block_dot4_f32acc(
    u_val: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Float32:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    return fp4_dot4_sum_f32acc(u_val, xh0, xh1, xh2, xh3)


@cute.jit
def _block_dot4_pair_f32acc(
    up_val: Uint32,
    gate_val: Uint32,
    smem_xh: cute.Tensor,
    xh_base: Int32,
) -> Tuple[Float32, Float32]:
    xh0 = Uint32(smem_xh[xh_base + Int32(0)])
    xh1 = Uint32(smem_xh[xh_base + Int32(1)])
    xh2 = Uint32(smem_xh[xh_base + Int32(2)])
    xh3 = Uint32(smem_xh[xh_base + Int32(3)])
    up = fp4_dot4_sum_f32acc(up_val, xh0, xh1, xh2, xh3)
    gate = fp4_dot4_sum_f32acc(gate_val, xh0, xh1, xh2, xh3)
    return up, gate


@cute.jit
def _token_publish_fc1_ready(
    barrier_count: cute.Tensor,
    barrier_epoch: cute.Tensor,
    token_idx: Int32,
    expected_epoch: Int32,
    chunks_per_token: Int32,
    is_cta_leader: Int32,
):
    if is_cta_leader > Int32(0):
        count_addr = get_ptr_as_int64(barrier_count, token_idx)
        epoch_addr = get_ptr_as_int64(barrier_epoch, token_idx)
        arrived = atomic_add_global_i32(count_addr, Int32(1))
        if arrived == chunks_per_token - Int32(1):
            st_global_i32(count_addr, Int32(0))
            st_global_release_i32(epoch_addr, expected_epoch + Int32(1))


@cute.jit
def _token_wait_fc1_ready(
    barrier_epoch: cute.Tensor,
    token_idx: Int32,
    expected_epoch: Int32,
    is_cta_leader: Int32,
):
    cute.arch.sync_threads()
    if is_cta_leader > Int32(0):
        epoch_addr = get_ptr_as_int64(barrier_epoch, token_idx)
        spin_wait_global_eq_i32(epoch_addr, expected_epoch)
    cute.arch.sync_threads()


class MoEDirectMicroKernel:
    """Decode-focused compact MoE kernel for SM120."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fast_math: bool = False,
        activation: str = "silu",
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        dynamic_down_scale: bool = False,
        w4a16_mode: bool = False,
    ):
        if activation not in {"silu", "relu2"}:
            raise ValueError(f"unsupported activation {activation!r}")
        self.sf_vec_size = sf_vec_size
        self.fast_math = fast_math
        self.activation = activation
        self.is_gated = activation == "silu"
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
        self.dynamic_down_scale = dynamic_down_scale
        self.w4a16_mode = w4a16_mode
        self._cfg = cast(_ShapeConfig, None)
        self.m_const = 0
        self.m1_fc2_onepass = False
        self.grid_x = 0

    @cute.jit
    def _fp4_dot4_for_math(
        self,
        u_packed: Uint32,
        x0: Uint32,
        x1: Uint32,
        x2: Uint32,
        x3: Uint32,
    ) -> Float32:
        if cutlass.const_expr(self.fast_math):
            return fp4_dot4_sum(u_packed, x0, x1, x2, x3)
        return fp4_dot4_sum_f32acc(u_packed, x0, x1, x2, x3)

    @cute.jit
    def _block_dot_hfma2_for_math(
        self,
        u_a: Uint32,
        u_b: Uint32,
        smem_xh: cute.Tensor,
        xh_base: Int32,
    ) -> Float32:
        if cutlass.const_expr(self.fast_math):
            return _block_dot_hfma2(u_a, u_b, smem_xh, xh_base)
        return _block_dot_hfma2_f32acc(u_a, u_b, smem_xh, xh_base)

    @cute.jit
    def _block_dot_hfma2_pair_for_math(
        self,
        up_a: Uint32,
        up_b: Uint32,
        gate_a: Uint32,
        gate_b: Uint32,
        smem_xh: cute.Tensor,
        xh_base: Int32,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(self.fast_math):
            return _block_dot_hfma2_pair(up_a, up_b, gate_a, gate_b, smem_xh, xh_base)
        return _block_dot_hfma2_pair_f32acc(
            up_a, up_b, gate_a, gate_b, smem_xh, xh_base
        )

    @cute.jit
    def _block_dot4_for_math(
        self,
        u_val: Uint32,
        smem_xh: cute.Tensor,
        xh_base: Int32,
    ) -> Float32:
        if cutlass.const_expr(self.fast_math):
            return _block_dot4(u_val, smem_xh, xh_base)
        return _block_dot4_f32acc(u_val, smem_xh, xh_base)

    @cute.jit
    def _block_dot4_pair_for_math(
        self,
        up_val: Uint32,
        gate_val: Uint32,
        smem_xh: cute.Tensor,
        xh_base: Int32,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(self.fast_math):
            return _block_dot4_pair(up_val, gate_val, smem_xh, xh_base)
        return _block_dot4_pair_f32acc(up_val, gate_val, smem_xh, xh_base)

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        if m not in (1, 2, 4, 8):
            return False
        if k <= 0 or k % (32 * _BLOCK_SIZE) != 0 or k % 128 != 0:
            return False
        if k // _BLOCK_SIZE > 32 * _MAX_DIRECT_K_SEGMENTS:
            return False
        if n <= 0 or n % _BLOCK_SIZE != 0:
            return False
        fc1_chunks = _fc1_chunks_for_m(m, n)
        if n % fc1_chunks != 0:
            return False
        i_chunk = n // fc1_chunks
        if i_chunk % _BLOCK_SIZE != 0:
            return False
        k_segments = k // (32 * _BLOCK_SIZE)
        return (
            _direct_k_segments_supported(k_segments)
            and 0 < num_topk <= 32
            and weight_E > 0
        )

    def configure(
        self,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
        *,
        max_active_ctas: int | None = None,
        device: torch.device | None = None,
    ):
        cfg = _make_shape_config(
            m=m, k=k, n=n, num_topk=num_topk, weight_E=weight_E, is_gated=self.is_gated
        )
        num_fc1_chunks = _fc1_chunks_for_m(m, n)
        if self.w4a16_mode and m > 1:
            # Keep one FC1 row per warp for W4A16 multi-token decode so the
            # direct kernel stays within the CUTLASS 4.5 launch resource limit.
            num_fc1_chunks = max(num_fc1_chunks, n // _BLOCK_SIZE)
        cfg = _remake_shape_config_fc1(cfg, num_fc1_chunks)

        fc1_tasks = m * cfg.num_topk * cfg.fc1_chunks
        w4a16_rowpair_fc2 = bool(self.w4a16_mode and m > 1 and cfg.fc2_n_chunks == 1)
        if m == 1:
            fc2_tasks = cfg.k_dim // (_K_PER_CTA * 2)
        elif w4a16_rowpair_fc2:
            fc2_tasks = (m * cfg.k_dim) // (_K_PER_CTA * 2)
        else:
            fc2_tasks = (m * cfg.k_dim) // (_K_PER_CTA * 4)
        if max_active_ctas is None:
            max_active_ctas = min(get_num_sm(device), get_max_active_clusters(1))
        if m == 1 or m == 2:
            grid_x = max(1, min(int(max_active_ctas), max(fc1_tasks, fc2_tasks)))
        elif num_fc1_chunks < 16:
            grid_x = max(1, min(int(max_active_ctas), fc2_tasks))
        else:
            grid_x = max(1, min(int(max_active_ctas), fc1_tasks, fc2_tasks))
        m1_fc2_onepass = bool(m == 1 and grid_x >= fc2_tasks)

        self._cfg = cfg
        self.m_const = m
        self.m1_fc2_onepass = m1_fc2_onepass
        self.grid_x = grid_x

    @cute.jit
    def _resident_grid_barrier(
        self,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        grid_x: Int32,
        is_cta_leader: Int32,
    ):
        cute.arch.sync_threads()
        threadfence()
        if is_cta_leader > Int32(0):
            barrier_count_addr = get_ptr_as_int64(barrier_count, Int32(0))
            barrier_epoch_addr = get_ptr_as_int64(barrier_epoch, Int32(0))
            old_epoch = ld_global_acquire_i32(barrier_epoch_addr)
            arrived = atomic_add_global_i32(barrier_count_addr, Int32(1))
            if arrived == grid_x - Int32(1):
                st_global_i32(barrier_count_addr, Int32(0))
                st_global_release_i32(barrier_epoch_addr, old_epoch + Int32(1))
            else:
                spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)
        cute.arch.sync_threads()

    @cute.jit
    def _m1_fc2_rowpair_narrow(
        self,
        fc2_task: Int32,
        warp_id: Int32,
        lane: Int32,
        w2_base_addr: Int64,
        w2s_base_addr: Int64,
        intermediate: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
    ):
        cfg = self._cfg
        k_chunk_off = fc2_task * Int32(_K_PER_CTA * 2)
        k_row0 = k_chunk_off + warp_id * Int32(2)
        k_row1 = k_row0 + Int32(1)

        lane_byte_off = Int64(lane) * Int64(4)
        sf_cols = Int32(cfg.w2_sf_cols)
        lane_cb = lane >> Int32(3)
        lane_mode_c = (lane >> Int32(1)) & Int32(3)
        bsf_byte_shift = lane_mode_c * Int32(8)
        out_acc0 = Float32(0.0)
        out_acc1 = Float32(0.0)

        for kk in cutlass.range_constexpr(cfg.num_topk):
            eid_addr = Int32(kk)
            eid = Int32(topk_ids[eid_addr])
            router_w = topk_weights[eid_addr]
            alpha_fc2 = w2_alphas[eid]
            scale_lane = alpha_fc2 * router_w

            ebase_w = Int64(eid) * Int64(cfg.k_dim * cfg.n_half)
            ebase_sf = Int64(eid) * Int64(cfg.w2_sf_rows * cfg.w2_sf_cols)

            row_rb0 = k_row0 >> Int32(7)
            row_mode_a0 = (k_row0 >> Int32(5)) & Int32(3)
            row_mode_32_0 = k_row0 & Int32(31)
            row_rb1 = k_row1 >> Int32(7)
            row_mode_a1 = (k_row1 >> Int32(5)) & Int32(3)
            row_mode_32_1 = k_row1 & Int32(31)

            kk_off = Int32(kk) * Int32(128)
            xh0 = Uint32(intermediate[kk_off + Int32(0 * 32) + lane])
            xh1 = Uint32(intermediate[kk_off + Int32(1 * 32) + lane])
            xh2 = Uint32(intermediate[kk_off + Int32(2 * 32) + lane])
            xh3 = Uint32(intermediate[kk_off + Int32(3 * 32) + lane])

            u_packed0 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row0) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off0 = (
                Int64(row_rb0) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_0) * Int64(16)
                + Int64(row_mode_a0) * Int64(4)
            )
            sf_word0 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off0)
            bsf_byte0 = (sf_word0 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f0 = cvt_e4m3_to_f32_via_f16(bsf_byte0)
            out_acc0 = (
                out_acc0
                + bsf_f0
                * self._fp4_dot4_for_math(u_packed0, xh0, xh1, xh2, xh3)
                * scale_lane
            )

            u_packed1 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row1) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off1 = (
                Int64(row_rb1) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_1) * Int64(16)
                + Int64(row_mode_a1) * Int64(4)
            )
            sf_word1 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off1)
            bsf_byte1 = (sf_word1 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f1 = cvt_e4m3_to_f32_via_f16(bsf_byte1)
            out_acc1 = (
                out_acc1
                + bsf_f1
                * self._fp4_dot4_for_math(u_packed1, xh0, xh1, xh2, xh3)
                * scale_lane
            )

        sum_warp0 = cute.arch.warp_reduction_sum(out_acc0)
        sum_warp1 = cute.arch.warp_reduction_sum(out_acc1)
        if lane == Int32(0):
            scatter_output[k_row0] = BFloat16(sum_warp0)
            scatter_output[k_row1] = BFloat16(sum_warp1)

    @cute.jit
    def _m1_fc2_rowpair_wide(
        self,
        fc2_task: Int32,
        warp_id: Int32,
        lane: Int32,
        w2_base_addr: Int64,
        w2s_base_addr: Int64,
        intermediate: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
    ):
        cfg = self._cfg
        k_chunk_off = fc2_task * Int32(_K_PER_CTA * 2)
        k_row0 = k_chunk_off + warp_id * Int32(2)
        k_row1 = k_row0 + Int32(1)

        lane_byte_off = Int64(lane) * Int64(4)
        n_u32_per_expert = Int32(cfg.fc2_n_chunks * 128)
        sf_cols = Int32(cfg.w2_sf_cols)
        num_cb = sf_cols >> Int32(2)
        lane_cb = lane >> Int32(3)
        lane_mode_c = (lane >> Int32(1)) & Int32(3)
        bsf_byte_shift = lane_mode_c * Int32(8)
        out_acc0 = Float32(0.0)
        out_acc1 = Float32(0.0)

        for kk in cutlass.range_constexpr(cfg.num_topk):
            eid_addr = Int32(kk)
            eid = Int32(topk_ids[eid_addr])
            router_w = topk_weights[eid_addr]
            alpha_fc2 = w2_alphas[eid]
            scale_lane = alpha_fc2 * router_w

            ebase_w = Int64(eid) * Int64(cfg.k_dim * cfg.n_half)
            ebase_sf = Int64(eid) * Int64(cfg.w2_sf_rows * cfg.w2_sf_cols)

            row_rb0 = k_row0 >> Int32(7)
            row_mode_a0 = (k_row0 >> Int32(5)) & Int32(3)
            row_mode_32_0 = k_row0 & Int32(31)
            row_rb1 = k_row1 >> Int32(7)
            row_mode_a1 = (k_row1 >> Int32(5)) & Int32(3)
            row_mode_32_1 = k_row1 & Int32(31)

            for nc in cutlass.range_constexpr(cfg.fc2_n_chunks):
                chunk_base = Int32(nc) * Int32(128)
                kk_off = Int32(kk) * n_u32_per_expert + chunk_base
                xh0 = Uint32(intermediate[kk_off + Int32(0 * 32) + lane])
                xh1 = Uint32(intermediate[kk_off + Int32(1 * 32) + lane])
                xh2 = Uint32(intermediate[kk_off + Int32(2 * 32) + lane])
                xh3 = Uint32(intermediate[kk_off + Int32(3 * 32) + lane])

                cb_idx = Int32(nc) * Int32(4) + lane_cb
                w_valid = Int32(1) if cb_idx < num_cb else Int32(0)
                u_packed0 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row0) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off0 = (
                    Int64(row_rb0) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_0) * Int64(16)
                    + Int64(row_mode_a0) * Int64(4)
                )
                sf_word0 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off0)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte0 = (sf_word0 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f0 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte0)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc0 = (
                    out_acc0
                    + bsf_f0
                    * self._fp4_dot4_for_math(u_packed0, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

                u_packed1 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row1) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off1 = (
                    Int64(row_rb1) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_1) * Int64(16)
                    + Int64(row_mode_a1) * Int64(4)
                )
                sf_word1 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off1)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte1 = (sf_word1 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f1 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte1)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc1 = (
                    out_acc1
                    + bsf_f1
                    * self._fp4_dot4_for_math(u_packed1, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

        sum_warp0 = cute.arch.warp_reduction_sum(out_acc0)
        sum_warp1 = cute.arch.warp_reduction_sum(out_acc1)
        if lane == Int32(0):
            scatter_output[k_row0] = BFloat16(sum_warp0)
            scatter_output[k_row1] = BFloat16(sum_warp1)

    @cute.jit
    def _m2_fc2_rowpair_narrow(
        self,
        fc2_task: Int32,
        warp_id: Int32,
        lane: Int32,
        w2_base_addr: Int64,
        w2s_base_addr: Int64,
        intermediate: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
    ):
        cfg = self._cfg
        rows_per_cta = Int32(_K_PER_CTA * 2)
        linear_row_base = fc2_task * rows_per_cta
        t = linear_row_base // Int32(cfg.k_dim)
        k_chunk_off = linear_row_base - t * Int32(cfg.k_dim)
        k_row0 = k_chunk_off + warp_id * Int32(2)
        k_row1 = k_row0 + Int32(1)

        lane_byte_off = Int64(lane) * Int64(4)
        token_inter_base = t * Int32(cfg.inter_u32)
        sf_cols = Int32(cfg.w2_sf_cols)
        lane_cb = lane >> Int32(3)
        lane_mode_c = (lane >> Int32(1)) & Int32(3)
        bsf_byte_shift = lane_mode_c * Int32(8)
        out_acc0 = Float32(0.0)
        out_acc1 = Float32(0.0)

        row_rb0 = k_row0 >> Int32(7)
        row_mode_a0 = (k_row0 >> Int32(5)) & Int32(3)
        row_mode_32_0 = k_row0 & Int32(31)
        row_rb1 = k_row1 >> Int32(7)
        row_mode_a1 = (k_row1 >> Int32(5)) & Int32(3)
        row_mode_32_1 = k_row1 & Int32(31)

        for kk in cutlass.range_constexpr(cfg.num_topk):
            eid_addr = t * Int32(cfg.num_topk) + Int32(kk)
            eid = Int32(topk_ids[eid_addr])
            router_w = topk_weights[eid_addr]
            alpha_fc2 = w2_alphas[eid]
            scale_lane = alpha_fc2 * router_w

            ebase_w = Int64(eid) * Int64(cfg.k_dim * cfg.n_half)
            ebase_sf = Int64(eid) * Int64(cfg.w2_sf_rows * cfg.w2_sf_cols)

            kk_off = token_inter_base + Int32(kk) * Int32(128)
            xh0 = Uint32(intermediate[kk_off + Int32(0 * 32) + lane])
            xh1 = Uint32(intermediate[kk_off + Int32(1 * 32) + lane])
            xh2 = Uint32(intermediate[kk_off + Int32(2 * 32) + lane])
            xh3 = Uint32(intermediate[kk_off + Int32(3 * 32) + lane])

            u_packed0 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row0) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off0 = (
                Int64(row_rb0) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_0) * Int64(16)
                + Int64(row_mode_a0) * Int64(4)
            )
            sf_word0 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off0)
            bsf_byte0 = (sf_word0 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f0 = cvt_e4m3_to_f32_via_f16(bsf_byte0)
            out_acc0 = (
                out_acc0
                + bsf_f0
                * self._fp4_dot4_for_math(u_packed0, xh0, xh1, xh2, xh3)
                * scale_lane
            )

            u_packed1 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row1) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off1 = (
                Int64(row_rb1) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_1) * Int64(16)
                + Int64(row_mode_a1) * Int64(4)
            )
            sf_word1 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off1)
            bsf_byte1 = (sf_word1 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f1 = cvt_e4m3_to_f32_via_f16(bsf_byte1)
            out_acc1 = (
                out_acc1
                + bsf_f1
                * self._fp4_dot4_for_math(u_packed1, xh0, xh1, xh2, xh3)
                * scale_lane
            )

        sum_warp0 = cute.arch.warp_reduction_sum(out_acc0)
        sum_warp1 = cute.arch.warp_reduction_sum(out_acc1)
        if lane == Int32(0):
            out_base = t * Int32(cfg.k_dim)
            scatter_output[out_base + k_row0] = BFloat16(sum_warp0)
            scatter_output[out_base + k_row1] = BFloat16(sum_warp1)

    @cute.jit
    def _m2_fc2_rowquad_narrow(
        self,
        fc2_task: Int32,
        warp_id: Int32,
        lane: Int32,
        w2_base_addr: Int64,
        w2s_base_addr: Int64,
        intermediate: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
    ):
        cfg = self._cfg
        rows_per_cta = Int32(_K_PER_CTA * 4)
        linear_row_base = fc2_task * rows_per_cta
        t = linear_row_base // Int32(cfg.k_dim)
        k_chunk_off = linear_row_base - t * Int32(cfg.k_dim)
        k_row0 = k_chunk_off + warp_id * Int32(4)
        k_row1 = k_row0 + Int32(1)
        k_row2 = k_row0 + Int32(2)
        k_row3 = k_row0 + Int32(3)

        lane_byte_off = Int64(lane) * Int64(4)
        token_inter_base = t * Int32(cfg.inter_u32)
        sf_cols = Int32(cfg.w2_sf_cols)
        lane_cb = lane >> Int32(3)
        lane_mode_c = (lane >> Int32(1)) & Int32(3)
        bsf_byte_shift = lane_mode_c * Int32(8)
        out_acc0 = Float32(0.0)
        out_acc1 = Float32(0.0)
        out_acc2 = Float32(0.0)
        out_acc3 = Float32(0.0)

        row_rb0 = k_row0 >> Int32(7)
        row_mode_a0 = (k_row0 >> Int32(5)) & Int32(3)
        row_mode_32_0 = k_row0 & Int32(31)
        row_rb1 = k_row1 >> Int32(7)
        row_mode_a1 = (k_row1 >> Int32(5)) & Int32(3)
        row_mode_32_1 = k_row1 & Int32(31)
        row_rb2 = k_row2 >> Int32(7)
        row_mode_a2 = (k_row2 >> Int32(5)) & Int32(3)
        row_mode_32_2 = k_row2 & Int32(31)
        row_rb3 = k_row3 >> Int32(7)
        row_mode_a3 = (k_row3 >> Int32(5)) & Int32(3)
        row_mode_32_3 = k_row3 & Int32(31)

        for kk in cutlass.range_constexpr(cfg.num_topk):
            eid_addr = t * Int32(cfg.num_topk) + Int32(kk)
            eid = Int32(topk_ids[eid_addr])
            router_w = topk_weights[eid_addr]
            alpha_fc2 = w2_alphas[eid]
            scale_lane = alpha_fc2 * router_w

            ebase_w = Int64(eid) * Int64(cfg.k_dim * cfg.n_half)
            ebase_sf = Int64(eid) * Int64(cfg.w2_sf_rows * cfg.w2_sf_cols)

            kk_off = token_inter_base + Int32(kk) * Int32(128)
            xh0 = Uint32(intermediate[kk_off + Int32(0 * 32) + lane])
            xh1 = Uint32(intermediate[kk_off + Int32(1 * 32) + lane])
            xh2 = Uint32(intermediate[kk_off + Int32(2 * 32) + lane])
            xh3 = Uint32(intermediate[kk_off + Int32(3 * 32) + lane])

            u_packed0 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row0) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off0 = (
                Int64(row_rb0) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_0) * Int64(16)
                + Int64(row_mode_a0) * Int64(4)
            )
            sf_word0 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off0)
            bsf_byte0 = (sf_word0 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f0 = cvt_e4m3_to_f32_via_f16(bsf_byte0)
            out_acc0 = (
                out_acc0
                + bsf_f0
                * self._fp4_dot4_for_math(u_packed0, xh0, xh1, xh2, xh3)
                * scale_lane
            )

            u_packed1 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row1) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off1 = (
                Int64(row_rb1) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_1) * Int64(16)
                + Int64(row_mode_a1) * Int64(4)
            )
            sf_word1 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off1)
            bsf_byte1 = (sf_word1 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f1 = cvt_e4m3_to_f32_via_f16(bsf_byte1)
            out_acc1 = (
                out_acc1
                + bsf_f1
                * self._fp4_dot4_for_math(u_packed1, xh0, xh1, xh2, xh3)
                * scale_lane
            )

            u_packed2 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row2) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off2 = (
                Int64(row_rb2) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_2) * Int64(16)
                + Int64(row_mode_a2) * Int64(4)
            )
            sf_word2 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off2)
            bsf_byte2 = (sf_word2 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f2 = cvt_e4m3_to_f32_via_f16(bsf_byte2)
            out_acc2 = (
                out_acc2
                + bsf_f2
                * self._fp4_dot4_for_math(u_packed2, xh0, xh1, xh2, xh3)
                * scale_lane
            )

            u_packed3 = ld_global_nc_u32(
                w2_base_addr
                + ebase_w
                + Int64(k_row3) * Int64(cfg.n_half)
                + lane_byte_off
            )
            bsf_off3 = (
                Int64(row_rb3) * Int64(sf_cols * 128)
                + Int64(lane_cb) * Int64(512)
                + Int64(row_mode_32_3) * Int64(16)
                + Int64(row_mode_a3) * Int64(4)
            )
            sf_word3 = ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off3)
            bsf_byte3 = (sf_word3 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
            bsf_f3 = cvt_e4m3_to_f32_via_f16(bsf_byte3)
            out_acc3 = (
                out_acc3
                + bsf_f3
                * self._fp4_dot4_for_math(u_packed3, xh0, xh1, xh2, xh3)
                * scale_lane
            )

        sum_warp0 = cute.arch.warp_reduction_sum(out_acc0)
        sum_warp1 = cute.arch.warp_reduction_sum(out_acc1)
        sum_warp2 = cute.arch.warp_reduction_sum(out_acc2)
        sum_warp3 = cute.arch.warp_reduction_sum(out_acc3)
        if lane == Int32(0):
            out_base = t * Int32(cfg.k_dim)
            scatter_output[out_base + k_row0] = BFloat16(sum_warp0)
            scatter_output[out_base + k_row1] = BFloat16(sum_warp1)
            scatter_output[out_base + k_row2] = BFloat16(sum_warp2)
            scatter_output[out_base + k_row3] = BFloat16(sum_warp3)

    @cute.jit
    def _m2_fc2_rowquad_wide(
        self,
        fc2_task: Int32,
        warp_id: Int32,
        lane: Int32,
        w2_base_addr: Int64,
        w2s_base_addr: Int64,
        intermediate: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
    ):
        cfg = self._cfg
        rows_per_cta = Int32(_K_PER_CTA * 4)
        linear_row_base = fc2_task * rows_per_cta
        t = linear_row_base // Int32(cfg.k_dim)
        k_chunk_off = linear_row_base - t * Int32(cfg.k_dim)
        k_row0 = k_chunk_off + warp_id * Int32(4)
        k_row1 = k_row0 + Int32(1)
        k_row2 = k_row0 + Int32(2)
        k_row3 = k_row0 + Int32(3)

        lane_byte_off = Int64(lane) * Int64(4)
        n_u32_per_expert = Int32(cfg.fc2_n_chunks * 128)
        token_inter_base = t * Int32(cfg.inter_u32)
        sf_cols = Int32(cfg.w2_sf_cols)
        num_cb = sf_cols >> Int32(2)
        lane_cb = lane >> Int32(3)
        lane_mode_c = (lane >> Int32(1)) & Int32(3)
        bsf_byte_shift = lane_mode_c * Int32(8)
        out_acc0 = Float32(0.0)
        out_acc1 = Float32(0.0)
        out_acc2 = Float32(0.0)
        out_acc3 = Float32(0.0)

        row_rb0 = k_row0 >> Int32(7)
        row_mode_a0 = (k_row0 >> Int32(5)) & Int32(3)
        row_mode_32_0 = k_row0 & Int32(31)
        row_rb1 = k_row1 >> Int32(7)
        row_mode_a1 = (k_row1 >> Int32(5)) & Int32(3)
        row_mode_32_1 = k_row1 & Int32(31)
        row_rb2 = k_row2 >> Int32(7)
        row_mode_a2 = (k_row2 >> Int32(5)) & Int32(3)
        row_mode_32_2 = k_row2 & Int32(31)
        row_rb3 = k_row3 >> Int32(7)
        row_mode_a3 = (k_row3 >> Int32(5)) & Int32(3)
        row_mode_32_3 = k_row3 & Int32(31)

        for kk in cutlass.range_constexpr(cfg.num_topk):
            eid_addr = t * Int32(cfg.num_topk) + Int32(kk)
            eid = Int32(topk_ids[eid_addr])
            router_w = topk_weights[eid_addr]
            alpha_fc2 = w2_alphas[eid]
            scale_lane = alpha_fc2 * router_w

            ebase_w = Int64(eid) * Int64(cfg.k_dim * cfg.n_half)
            ebase_sf = Int64(eid) * Int64(cfg.w2_sf_rows * cfg.w2_sf_cols)

            for nc in cutlass.range_constexpr(cfg.fc2_n_chunks):
                chunk_base = Int32(nc) * Int32(128)
                kk_off = token_inter_base + Int32(kk) * n_u32_per_expert + chunk_base
                xh0 = Uint32(intermediate[kk_off + Int32(0 * 32) + lane])
                xh1 = Uint32(intermediate[kk_off + Int32(1 * 32) + lane])
                xh2 = Uint32(intermediate[kk_off + Int32(2 * 32) + lane])
                xh3 = Uint32(intermediate[kk_off + Int32(3 * 32) + lane])

                cb_idx = Int32(nc) * Int32(4) + lane_cb
                w_valid = Int32(1) if cb_idx < num_cb else Int32(0)
                u_packed0 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row0) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off0 = (
                    Int64(row_rb0) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_0) * Int64(16)
                    + Int64(row_mode_a0) * Int64(4)
                )
                sf_word0 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off0)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte0 = (sf_word0 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f0 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte0)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc0 = (
                    out_acc0
                    + bsf_f0
                    * self._fp4_dot4_for_math(u_packed0, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

                u_packed1 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row1) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off1 = (
                    Int64(row_rb1) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_1) * Int64(16)
                    + Int64(row_mode_a1) * Int64(4)
                )
                sf_word1 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off1)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte1 = (sf_word1 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f1 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte1)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc1 = (
                    out_acc1
                    + bsf_f1
                    * self._fp4_dot4_for_math(u_packed1, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

                u_packed2 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row2) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off2 = (
                    Int64(row_rb2) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_2) * Int64(16)
                    + Int64(row_mode_a2) * Int64(4)
                )
                sf_word2 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off2)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte2 = (sf_word2 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f2 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte2)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc2 = (
                    out_acc2
                    + bsf_f2
                    * self._fp4_dot4_for_math(u_packed2, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

                u_packed3 = (
                    ld_global_nc_u32(
                        w2_base_addr
                        + ebase_w
                        + Int64(k_row3) * Int64(cfg.n_half)
                        + Int64(chunk_base)
                        + lane_byte_off
                    )
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_off3 = (
                    Int64(row_rb3) * Int64(sf_cols * 128)
                    + Int64(cb_idx) * Int64(512)
                    + Int64(row_mode_32_3) * Int64(16)
                    + Int64(row_mode_a3) * Int64(4)
                )
                sf_word3 = (
                    ld_global_nc_u32(w2s_base_addr + ebase_sf + bsf_off3)
                    if w_valid > Int32(0)
                    else Uint32(0)
                )
                bsf_byte3 = (sf_word3 >> Uint32(bsf_byte_shift)) & Uint32(0xFF)
                bsf_f3 = (
                    cvt_e4m3_to_f32_via_f16(bsf_byte3)
                    if w_valid > Int32(0)
                    else Float32(0.0)
                )
                out_acc3 = (
                    out_acc3
                    + bsf_f3
                    * self._fp4_dot4_for_math(u_packed3, xh0, xh1, xh2, xh3)
                    * scale_lane
                )

        sum_warp0 = cute.arch.warp_reduction_sum(out_acc0)
        sum_warp1 = cute.arch.warp_reduction_sum(out_acc1)
        sum_warp2 = cute.arch.warp_reduction_sum(out_acc2)
        sum_warp3 = cute.arch.warp_reduction_sum(out_acc3)
        if lane == Int32(0):
            out_base = t * Int32(cfg.k_dim)
            scatter_output[out_base + k_row0] = BFloat16(sum_warp0)
            scatter_output[out_base + k_row1] = BFloat16(sum_warp1)
            scatter_output[out_base + k_row2] = BFloat16(sum_warp2)
            scatter_output[out_base + k_row3] = BFloat16(sum_warp3)

    @cute.kernel
    def kernel(
        self,
        a_input: cute.Tensor,
        w1_weights: cute.Tensor,
        w1_scales: cute.Tensor,
        w1_alphas: cute.Tensor,
        input_gs: cute.Tensor,
        down_input_scale: cute.Tensor,
        intermediate: cute.Tensor,
        w2_weights: cute.Tensor,
        w2_scales: cute.Tensor,
        w2_alphas: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        scatter_output: cute.Tensor,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        m_val: Int32,
    ):
        cfg = self._cfg
        bidx_x, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        gdim_x, _, _ = cute.arch.grid_dim()
        is_cta_leader = Int32(1) if Int32(tidx) == Int32(0) else Int32(0)
        m1_epoch0 = Int32(0)
        if cutlass.const_expr(self.m_const == 1):
            m1_epoch0 = ld_global_acquire_i32(get_ptr_as_int64(barrier_epoch, Int32(0)))

        if cutlass.const_expr(cfg.k_segments == 2):
            smem_xh_ptr = cute.arch.alloc_smem(Uint32, 2 * cfg.smem_xh_size)
            smem_xh = cute.make_tensor(
                smem_xh_ptr, cute.make_layout(2 * cfg.smem_xh_size)
            )
        else:
            smem_xh_ptr = cute.arch.alloc_smem(Uint32, cfg.smem_xh_size)
            smem_xh = cute.make_tensor(smem_xh_ptr, cute.make_layout(cfg.smem_xh_size))
        smem_int_ptr = cute.arch.alloc_smem(Float32, cfg.i_chunk)
        smem_int = cute.make_tensor(smem_int_ptr, cute.make_layout(cfg.i_chunk))
        reduce_scratch_ptr = cute.arch.alloc_smem(Float32, _NUM_WARPS)
        reduce_scratch = cute.make_tensor(
            reduce_scratch_ptr, cute.make_layout(_NUM_WARPS)
        )

        warp_id = tidx // Int32(32)
        lane = tidx % Int32(32)
        w1_base_addr = w1_weights.iterator.toint()
        w1s_base_addr = w1_scales.iterator.toint()

        # ===================================================================
        # PHASE 1: FC1 over route-order tasks
        # ===================================================================
        fc1_task_count = Int32(self.m_const * cfg.num_topk * cfg.fc1_chunks)
        fc1_task = Int32(bidx_x)
        if cutlass.const_expr(cfg.k_segments == 2):
            buf_idx = Int32(0)
            # Pre-loop: quantize first task into buf[0]
            if fc1_task < fc1_task_count:
                route_idx_0 = fc1_task // Int32(cfg.fc1_chunks)
                t0 = route_idx_0 // Int32(cfg.num_topk)
                eid_addr_0 = t0 * Int32(cfg.num_topk) + (
                    route_idx_0 - t0 * Int32(cfg.num_topk)
                )
                gs_fc1_0 = input_gs[Int32(topk_ids[eid_addr_0])]
                in_blk = tidx
                while in_blk < Int32(cfg.k_dim // _BLOCK_SIZE):
                    x_base = t0 * Int32(cfg.k_dim) + in_blk * Int32(_BLOCK_SIZE)
                    pad_off = in_blk // Int32(8)
                    phys_base = in_blk * Int32(_BLOCK_SIZE // 2) + pad_off
                    if cutlass.const_expr(self.w4a16_mode):
                        for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                            v0 = Float32(a_input[x_base + Int32(i * 2)])
                            v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                            smem_xh[phys_base + Int32(i)] = pack_f32x2_to_f16x2(v0, v1)
                    else:
                        blk_peak = Float32(0.0)
                        for i in cutlass.range_constexpr(_BLOCK_SIZE):
                            v = Float32(a_input[x_base + Int32(i)])
                            abs_v = v
                            if v < Float32(0.0):
                                abs_v = -v
                            if abs_v > blk_peak:
                                blk_peak = abs_v
                        q_scale = nvfp4_scale_from_amax(blk_peak, gs_fc1_0)
                        if q_scale > Float32(_FP8_E4M3_MAX):
                            q_scale = Float32(_FP8_E4M3_MAX)
                        sf_val = cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))
                        eff_scale = Float32(0.0)
                        if gs_fc1_0 != Float32(0.0):
                            eff_scale = sf_val / gs_fc1_0
                        if eff_scale < Float32(1e-30):
                            eff_scale = Float32(1e-30)
                        for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                            v0 = Float32(a_input[x_base + Int32(i * 2)])
                            v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                            f0, f1 = quant_dequant_2(v0, v1, sf_val, eff_scale)
                            smem_xh[phys_base + Int32(i)] = pack_f32x2_to_f16x2(f0, f1)
                    in_blk += Int32(_BLOCK_DIM)
                cute.arch.sync_threads()
        else:
            prev_t = Int32(-1)
        while fc1_task < fc1_task_count:
            route_idx = fc1_task // Int32(cfg.fc1_chunks)
            chunk_idx = fc1_task - route_idx * Int32(cfg.fc1_chunks)
            t = route_idx // Int32(cfg.num_topk)
            k_idx = route_idx - t * Int32(cfg.num_topk)
            i_chunk_off = chunk_idx * Int32(cfg.i_chunk)

            eid_addr = t * Int32(cfg.num_topk) + k_idx
            eid = Int32(topk_ids[eid_addr])
            alpha_fc1 = w1_alphas[eid]
            gs_fc1 = input_gs[eid]
            gs_fc2 = down_input_scale[eid]

            # ---- Input quantization ----
            if cutlass.const_expr(cfg.k_segments != 2):
                need_quant = Int32(1)
                if cutlass.const_expr(self.share_input_across_experts):
                    need_quant = Int32(1) if t != prev_t else Int32(0)
                if need_quant > Int32(0):
                    in_blk = tidx
                    while in_blk < Int32(cfg.k_dim // _BLOCK_SIZE):
                        x_base = t * Int32(cfg.k_dim) + in_blk * Int32(_BLOCK_SIZE)
                        pad_off = in_blk // Int32(8)
                        phys_base = in_blk * Int32(_BLOCK_SIZE // 2) + pad_off
                        if cutlass.const_expr(self.w4a16_mode):
                            for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                                v0 = Float32(a_input[x_base + Int32(i * 2)])
                                v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                                smem_xh[phys_base + Int32(i)] = pack_f32x2_to_f16x2(
                                    v0, v1
                                )
                        else:
                            blk_peak = Float32(0.0)
                            for i in cutlass.range_constexpr(_BLOCK_SIZE):
                                v = Float32(a_input[x_base + Int32(i)])
                                abs_v = v
                                if v < Float32(0.0):
                                    abs_v = -v
                                if abs_v > blk_peak:
                                    blk_peak = abs_v
                            q_scale = nvfp4_scale_from_amax(blk_peak, gs_fc1)
                            if q_scale > Float32(_FP8_E4M3_MAX):
                                q_scale = Float32(_FP8_E4M3_MAX)
                            sf_val = cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))
                            eff_scale = Float32(0.0)
                            if gs_fc1 != Float32(0.0):
                                eff_scale = sf_val / gs_fc1
                            if eff_scale < Float32(1e-30):
                                eff_scale = Float32(1e-30)
                            for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                                v0 = Float32(a_input[x_base + Int32(i * 2)])
                                v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                                f0, f1 = quant_dequant_2(v0, v1, sf_val, eff_scale)
                                smem_xh[phys_base + Int32(i)] = pack_f32x2_to_f16x2(
                                    f0, f1
                                )
                        in_blk += Int32(_BLOCK_DIM)
                    if cutlass.const_expr(self.share_input_across_experts):
                        prev_t = t
                cute.arch.sync_threads()

            # ---- FC1 weight load + dot product ----
            ebase_w = Int64(eid) * Int64(cfg.two_n) * Int64(cfg.k_half)
            ebase_sf = Int64(eid) * Int64(cfg.w1_sf_rows * cfg.w1_sf_cols)
            thread_byte_off = Int64(lane) * Int64(cfg.k_half // 32)
            xh_buf_base = Int32(0)
            if cutlass.const_expr(cfg.k_segments == 2):
                xh_buf_base = buf_idx * Int32(cfg.smem_xh_size)
            lane_seg_base = lane * Int32(cfg.k_segments)
            lane_pad_base = lane_seg_base // Int32(8)
            xh_base_t = (
                xh_buf_base + lane_seg_base * Int32(_BLOCK_SIZE // 2) + lane_pad_base
            )

            for r_iter in cutlass.range_constexpr(cfg.rows_per_warp_fc1):
                i_local = warp_id * Int32(cfg.rows_per_warp_fc1) + Int32(r_iter)
                i = i_chunk_off + i_local

                if cutlass.const_expr(self.is_gated):
                    up_byte_addr = (
                        w1_base_addr
                        + ebase_w
                        + Int64(i) * Int64(cfg.k_half)
                        + thread_byte_off
                    )
                    row_g = Int32(cfg.n) + i
                    rb_u = i >> Int32(7)
                    mode_a_u = (i >> Int32(5)) & Int32(3)
                    mode_32_u = i & Int32(31)
                    bsf_base_u = Int64(rb_u) * Int64(cfg.w1_sf_cols * 128) + Int64(
                        mode_32_u * Int32(16) + mode_a_u * Int32(4)
                    )
                    rb_g = row_g >> Int32(7)
                    mode_a_g = (row_g >> Int32(5)) & Int32(3)
                    mode_32_g = row_g & Int32(31)
                    bsf_base_g = Int64(rb_g) * Int64(cfg.w1_sf_cols * 128) + Int64(
                        mode_32_g * Int32(16) + mode_a_g * Int32(4)
                    )
                else:
                    row_g = i
                    rb_g = row_g >> Int32(7)
                    mode_a_g = (row_g >> Int32(5)) & Int32(3)
                    mode_32_g = row_g & Int32(31)
                    bsf_base_g = Int64(rb_g) * Int64(cfg.w1_sf_cols * 128) + Int64(
                        mode_32_g * Int32(16) + mode_a_g * Int32(4)
                    )
                gate_byte_addr = (
                    w1_base_addr
                    + ebase_w
                    + Int64(row_g) * Int64(cfg.k_half)
                    + thread_byte_off
                )
                col_blk_off = Int64(lane) * Int64((cfg.k_segments // 4) * 512)

                if cutlass.const_expr(cfg.k_segments == 8):
                    if cutlass.const_expr(self.is_gated):
                        uw_a0, uw_a1, uw_a2, uw_a3 = ld_global_nc_v4_u32(up_byte_addr)
                        uw_b0, uw_b1, uw_b2, uw_b3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(16)
                        )
                        uw_c0, uw_c1, uw_c2, uw_c3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(32)
                        )
                        uw_d0, uw_d1, uw_d2, uw_d3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(48)
                        )

                        bsf_addr_u_a = (
                            w1s_base_addr + ebase_sf + bsf_base_u + col_blk_off
                        )
                        bsf_addr_u_b = bsf_addr_u_a + Int64(512)
                        sf_u0, sf_u1, sf_u2, sf_u3 = cvt_e4m3x4_to_f32x4(
                            ld_global_nc_u32(bsf_addr_u_a)
                        )
                        sf_u4, sf_u5, sf_u6, sf_u7 = cvt_e4m3x4_to_f32x4(
                            ld_global_nc_u32(bsf_addr_u_b)
                        )

                    gw_a0, gw_a1, gw_a2, gw_a3 = ld_global_nc_v4_u32(gate_byte_addr)
                    gw_b0, gw_b1, gw_b2, gw_b3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(16)
                    )
                    gw_c0, gw_c1, gw_c2, gw_c3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(32)
                    )
                    gw_d0, gw_d1, gw_d2, gw_d3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(48)
                    )

                    bsf_addr_g_a = w1s_base_addr + ebase_sf + bsf_base_g + col_blk_off
                    bsf_addr_g_b = bsf_addr_g_a + Int64(512)
                    sf_g0, sf_g1, sf_g2, sf_g3 = cvt_e4m3x4_to_f32x4(
                        ld_global_nc_u32(bsf_addr_g_a)
                    )
                    sf_g4, sf_g5, sf_g6, sf_g7 = cvt_e4m3x4_to_f32x4(
                        ld_global_nc_u32(bsf_addr_g_b)
                    )

                    if cutlass.const_expr(not self.is_gated):
                        partial_gate = (
                            sf_g0
                            * self._block_dot_hfma2_for_math(
                                gw_a0, gw_a1, smem_xh, xh_base_t + Int32(0)
                            )
                            + sf_g1
                            * self._block_dot_hfma2_for_math(
                                gw_a2, gw_a3, smem_xh, xh_base_t + Int32(8)
                            )
                            + sf_g2
                            * self._block_dot_hfma2_for_math(
                                gw_b0, gw_b1, smem_xh, xh_base_t + Int32(16)
                            )
                            + sf_g3
                            * self._block_dot_hfma2_for_math(
                                gw_b2, gw_b3, smem_xh, xh_base_t + Int32(24)
                            )
                            + sf_g4
                            * self._block_dot_hfma2_for_math(
                                gw_c0, gw_c1, smem_xh, xh_base_t + Int32(32)
                            )
                            + sf_g5
                            * self._block_dot_hfma2_for_math(
                                gw_c2, gw_c3, smem_xh, xh_base_t + Int32(40)
                            )
                            + sf_g6
                            * self._block_dot_hfma2_for_math(
                                gw_d0, gw_d1, smem_xh, xh_base_t + Int32(48)
                            )
                            + sf_g7
                            * self._block_dot_hfma2_for_math(
                                gw_d2, gw_d3, smem_xh, xh_base_t + Int32(56)
                            )
                        )
                    elif cutlass.const_expr(self.is_gated):
                        dot_u0, dot_g0 = self._block_dot_hfma2_pair_for_math(
                            uw_a0, uw_a1, gw_a0, gw_a1, smem_xh, xh_base_t + Int32(0)
                        )
                        dot_u1, dot_g1 = self._block_dot_hfma2_pair_for_math(
                            uw_a2, uw_a3, gw_a2, gw_a3, smem_xh, xh_base_t + Int32(8)
                        )
                        dot_u2, dot_g2 = self._block_dot_hfma2_pair_for_math(
                            uw_b0, uw_b1, gw_b0, gw_b1, smem_xh, xh_base_t + Int32(16)
                        )
                        dot_u3, dot_g3 = self._block_dot_hfma2_pair_for_math(
                            uw_b2, uw_b3, gw_b2, gw_b3, smem_xh, xh_base_t + Int32(24)
                        )
                        dot_u4, dot_g4 = self._block_dot_hfma2_pair_for_math(
                            uw_c0, uw_c1, gw_c0, gw_c1, smem_xh, xh_base_t + Int32(32)
                        )
                        dot_u5, dot_g5 = self._block_dot_hfma2_pair_for_math(
                            uw_c2, uw_c3, gw_c2, gw_c3, smem_xh, xh_base_t + Int32(40)
                        )
                        dot_u6, dot_g6 = self._block_dot_hfma2_pair_for_math(
                            uw_d0, uw_d1, gw_d0, gw_d1, smem_xh, xh_base_t + Int32(48)
                        )
                        dot_u7, dot_g7 = self._block_dot_hfma2_pair_for_math(
                            uw_d2, uw_d3, gw_d2, gw_d3, smem_xh, xh_base_t + Int32(56)
                        )
                        partial_up = (
                            sf_u0 * dot_u0
                            + sf_u1 * dot_u1
                            + sf_u2 * dot_u2
                            + sf_u3 * dot_u3
                            + sf_u4 * dot_u4
                            + sf_u5 * dot_u5
                            + sf_u6 * dot_u6
                            + sf_u7 * dot_u7
                        )
                        partial_gate = (
                            sf_g0 * dot_g0
                            + sf_g1 * dot_g1
                            + sf_g2 * dot_g2
                            + sf_g3 * dot_g3
                            + sf_g4 * dot_g4
                            + sf_g5 * dot_g5
                            + sf_g6 * dot_g6
                            + sf_g7 * dot_g7
                        )
                elif cutlass.const_expr(cfg.k_segments == 6):
                    xh_off0 = Int32(0)
                    xh_off1 = Int32(8) + (
                        (lane_seg_base + Int32(1)) // Int32(8) - lane_pad_base
                    )
                    xh_off2 = Int32(16) + (
                        (lane_seg_base + Int32(2)) // Int32(8) - lane_pad_base
                    )
                    xh_off3 = Int32(24) + (
                        (lane_seg_base + Int32(3)) // Int32(8) - lane_pad_base
                    )
                    xh_off4 = Int32(32) + (
                        (lane_seg_base + Int32(4)) // Int32(8) - lane_pad_base
                    )
                    xh_off5 = Int32(40) + (
                        (lane_seg_base + Int32(5)) // Int32(8) - lane_pad_base
                    )

                    scale_pair_off = Int64(lane_seg_base // Int32(4)) * Int64(512)
                    scale_lane_mod = lane_seg_base % Int32(4)

                    if cutlass.const_expr(self.is_gated):
                        uw_a0, uw_a1, uw_a2, uw_a3 = ld_global_nc_v4_u32(up_byte_addr)
                        uw_b0, uw_b1, uw_b2, uw_b3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(16)
                        )
                        uw_c0, uw_c1, uw_c2, uw_c3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(32)
                        )

                        sf_word_u_a = ld_global_nc_u32(
                            w1s_base_addr + ebase_sf + bsf_base_u + scale_pair_off
                        )
                        sf_word_u_b = ld_global_nc_u32(
                            w1s_base_addr
                            + ebase_sf
                            + bsf_base_u
                            + scale_pair_off
                            + Int64(512)
                        )
                        sf_u0 = Float32(0.0)
                        sf_u1 = Float32(0.0)
                        sf_u2 = Float32(0.0)
                        sf_u3 = Float32(0.0)
                        sf_u4 = Float32(0.0)
                        sf_u5 = Float32(0.0)
                        if scale_lane_mod == Int32(0):
                            sf_u0 = cvt_e4m3_to_f32_via_f16(sf_word_u_a & Uint32(0xFF))
                            sf_u1 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_a >> Uint32(8)) & Uint32(0xFF)
                            )
                            sf_u2 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_a >> Uint32(16)) & Uint32(0xFF)
                            )
                            sf_u3 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_a >> Uint32(24)) & Uint32(0xFF)
                            )
                            sf_u4 = cvt_e4m3_to_f32_via_f16(sf_word_u_b & Uint32(0xFF))
                            sf_u5 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_b >> Uint32(8)) & Uint32(0xFF)
                            )
                        else:
                            sf_u0 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_a >> Uint32(16)) & Uint32(0xFF)
                            )
                            sf_u1 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_a >> Uint32(24)) & Uint32(0xFF)
                            )
                            sf_u2 = cvt_e4m3_to_f32_via_f16(sf_word_u_b & Uint32(0xFF))
                            sf_u3 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_b >> Uint32(8)) & Uint32(0xFF)
                            )
                            sf_u4 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_b >> Uint32(16)) & Uint32(0xFF)
                            )
                            sf_u5 = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u_b >> Uint32(24)) & Uint32(0xFF)
                            )

                    gw_a0, gw_a1, gw_a2, gw_a3 = ld_global_nc_v4_u32(gate_byte_addr)
                    gw_b0, gw_b1, gw_b2, gw_b3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(16)
                    )
                    gw_c0, gw_c1, gw_c2, gw_c3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(32)
                    )

                    sf_word_g_a = ld_global_nc_u32(
                        w1s_base_addr + ebase_sf + bsf_base_g + scale_pair_off
                    )
                    sf_word_g_b = ld_global_nc_u32(
                        w1s_base_addr
                        + ebase_sf
                        + bsf_base_g
                        + scale_pair_off
                        + Int64(512)
                    )
                    sf_g0 = Float32(0.0)
                    sf_g1 = Float32(0.0)
                    sf_g2 = Float32(0.0)
                    sf_g3 = Float32(0.0)
                    sf_g4 = Float32(0.0)
                    sf_g5 = Float32(0.0)
                    if scale_lane_mod == Int32(0):
                        sf_g0 = cvt_e4m3_to_f32_via_f16(sf_word_g_a & Uint32(0xFF))
                        sf_g1 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_a >> Uint32(8)) & Uint32(0xFF)
                        )
                        sf_g2 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_a >> Uint32(16)) & Uint32(0xFF)
                        )
                        sf_g3 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_a >> Uint32(24)) & Uint32(0xFF)
                        )
                        sf_g4 = cvt_e4m3_to_f32_via_f16(sf_word_g_b & Uint32(0xFF))
                        sf_g5 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_b >> Uint32(8)) & Uint32(0xFF)
                        )
                    else:
                        sf_g0 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_a >> Uint32(16)) & Uint32(0xFF)
                        )
                        sf_g1 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_a >> Uint32(24)) & Uint32(0xFF)
                        )
                        sf_g2 = cvt_e4m3_to_f32_via_f16(sf_word_g_b & Uint32(0xFF))
                        sf_g3 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_b >> Uint32(8)) & Uint32(0xFF)
                        )
                        sf_g4 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_b >> Uint32(16)) & Uint32(0xFF)
                        )
                        sf_g5 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g_b >> Uint32(24)) & Uint32(0xFF)
                        )

                    if cutlass.const_expr(not self.is_gated):
                        partial_gate = (
                            sf_g0
                            * self._block_dot_hfma2_for_math(
                                gw_a0, gw_a1, smem_xh, xh_base_t + xh_off0
                            )
                            + sf_g1
                            * self._block_dot_hfma2_for_math(
                                gw_a2, gw_a3, smem_xh, xh_base_t + xh_off1
                            )
                            + sf_g2
                            * self._block_dot_hfma2_for_math(
                                gw_b0, gw_b1, smem_xh, xh_base_t + xh_off2
                            )
                            + sf_g3
                            * self._block_dot_hfma2_for_math(
                                gw_b2, gw_b3, smem_xh, xh_base_t + xh_off3
                            )
                            + sf_g4
                            * self._block_dot_hfma2_for_math(
                                gw_c0, gw_c1, smem_xh, xh_base_t + xh_off4
                            )
                            + sf_g5
                            * self._block_dot_hfma2_for_math(
                                gw_c2, gw_c3, smem_xh, xh_base_t + xh_off5
                            )
                        )
                    elif cutlass.const_expr(self.is_gated):
                        dot_u0, dot_g0 = self._block_dot_hfma2_pair_for_math(
                            uw_a0, uw_a1, gw_a0, gw_a1, smem_xh, xh_base_t + xh_off0
                        )
                        dot_u1, dot_g1 = self._block_dot_hfma2_pair_for_math(
                            uw_a2, uw_a3, gw_a2, gw_a3, smem_xh, xh_base_t + xh_off1
                        )
                        dot_u2, dot_g2 = self._block_dot_hfma2_pair_for_math(
                            uw_b0, uw_b1, gw_b0, gw_b1, smem_xh, xh_base_t + xh_off2
                        )
                        dot_u3, dot_g3 = self._block_dot_hfma2_pair_for_math(
                            uw_b2, uw_b3, gw_b2, gw_b3, smem_xh, xh_base_t + xh_off3
                        )
                        dot_u4, dot_g4 = self._block_dot_hfma2_pair_for_math(
                            uw_c0, uw_c1, gw_c0, gw_c1, smem_xh, xh_base_t + xh_off4
                        )
                        dot_u5, dot_g5 = self._block_dot_hfma2_pair_for_math(
                            uw_c2, uw_c3, gw_c2, gw_c3, smem_xh, xh_base_t + xh_off5
                        )
                        partial_up = (
                            sf_u0 * dot_u0
                            + sf_u1 * dot_u1
                            + sf_u2 * dot_u2
                            + sf_u3 * dot_u3
                            + sf_u4 * dot_u4
                            + sf_u5 * dot_u5
                        )
                        partial_gate = (
                            sf_g0 * dot_g0
                            + sf_g1 * dot_g1
                            + sf_g2 * dot_g2
                            + sf_g3 * dot_g3
                            + sf_g4 * dot_g4
                            + sf_g5 * dot_g5
                        )
                elif cutlass.const_expr(cfg.k_segments == 12):
                    xh_off0 = Int32(0)
                    xh_off1 = Int32(8) + (
                        (lane_seg_base + Int32(1)) // Int32(8) - lane_pad_base
                    )
                    xh_off2 = Int32(16) + (
                        (lane_seg_base + Int32(2)) // Int32(8) - lane_pad_base
                    )
                    xh_off3 = Int32(24) + (
                        (lane_seg_base + Int32(3)) // Int32(8) - lane_pad_base
                    )
                    xh_off4 = Int32(32) + (
                        (lane_seg_base + Int32(4)) // Int32(8) - lane_pad_base
                    )
                    xh_off5 = Int32(40) + (
                        (lane_seg_base + Int32(5)) // Int32(8) - lane_pad_base
                    )
                    xh_off6 = Int32(48) + (
                        (lane_seg_base + Int32(6)) // Int32(8) - lane_pad_base
                    )
                    xh_off7 = Int32(56) + (
                        (lane_seg_base + Int32(7)) // Int32(8) - lane_pad_base
                    )
                    xh_off8 = Int32(64) + (
                        (lane_seg_base + Int32(8)) // Int32(8) - lane_pad_base
                    )
                    xh_off9 = Int32(72) + (
                        (lane_seg_base + Int32(9)) // Int32(8) - lane_pad_base
                    )
                    xh_off10 = Int32(80) + (
                        (lane_seg_base + Int32(10)) // Int32(8) - lane_pad_base
                    )
                    xh_off11 = Int32(88) + (
                        (lane_seg_base + Int32(11)) // Int32(8) - lane_pad_base
                    )

                    if cutlass.const_expr(self.is_gated):
                        uw_a0, uw_a1, uw_a2, uw_a3 = ld_global_nc_v4_u32(up_byte_addr)
                        uw_b0, uw_b1, uw_b2, uw_b3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(16)
                        )
                        uw_c0, uw_c1, uw_c2, uw_c3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(32)
                        )
                        uw_d0, uw_d1, uw_d2, uw_d3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(48)
                        )
                        uw_e0, uw_e1, uw_e2, uw_e3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(64)
                        )
                        uw_f0, uw_f1, uw_f2, uw_f3 = ld_global_nc_v4_u32(
                            up_byte_addr + Int64(80)
                        )

                        bsf_addr_u_a = (
                            w1s_base_addr + ebase_sf + bsf_base_u + col_blk_off
                        )
                        bsf_addr_u_b = bsf_addr_u_a + Int64(512)
                        bsf_addr_u_c = bsf_addr_u_a + Int64(1024)
                        sf_u0, sf_u1, sf_u2, sf_u3 = cvt_e4m3x4_to_f32x4(
                            ld_global_nc_u32(bsf_addr_u_a)
                        )
                        sf_u4, sf_u5, sf_u6, sf_u7 = cvt_e4m3x4_to_f32x4(
                            ld_global_nc_u32(bsf_addr_u_b)
                        )
                        sf_u8, sf_u9, sf_u10, sf_u11 = cvt_e4m3x4_to_f32x4(
                            ld_global_nc_u32(bsf_addr_u_c)
                        )

                    gw_a0, gw_a1, gw_a2, gw_a3 = ld_global_nc_v4_u32(gate_byte_addr)
                    gw_b0, gw_b1, gw_b2, gw_b3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(16)
                    )
                    gw_c0, gw_c1, gw_c2, gw_c3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(32)
                    )
                    gw_d0, gw_d1, gw_d2, gw_d3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(48)
                    )
                    gw_e0, gw_e1, gw_e2, gw_e3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(64)
                    )
                    gw_f0, gw_f1, gw_f2, gw_f3 = ld_global_nc_v4_u32(
                        gate_byte_addr + Int64(80)
                    )

                    bsf_addr_g_a = w1s_base_addr + ebase_sf + bsf_base_g + col_blk_off
                    bsf_addr_g_b = bsf_addr_g_a + Int64(512)
                    bsf_addr_g_c = bsf_addr_g_a + Int64(1024)
                    sf_g0, sf_g1, sf_g2, sf_g3 = cvt_e4m3x4_to_f32x4(
                        ld_global_nc_u32(bsf_addr_g_a)
                    )
                    sf_g4, sf_g5, sf_g6, sf_g7 = cvt_e4m3x4_to_f32x4(
                        ld_global_nc_u32(bsf_addr_g_b)
                    )
                    sf_g8, sf_g9, sf_g10, sf_g11 = cvt_e4m3x4_to_f32x4(
                        ld_global_nc_u32(bsf_addr_g_c)
                    )

                    if cutlass.const_expr(not self.is_gated):
                        partial_gate = (
                            sf_g0
                            * self._block_dot_hfma2_for_math(
                                gw_a0, gw_a1, smem_xh, xh_base_t + xh_off0
                            )
                            + sf_g1
                            * self._block_dot_hfma2_for_math(
                                gw_a2, gw_a3, smem_xh, xh_base_t + xh_off1
                            )
                            + sf_g2
                            * self._block_dot_hfma2_for_math(
                                gw_b0, gw_b1, smem_xh, xh_base_t + xh_off2
                            )
                            + sf_g3
                            * self._block_dot_hfma2_for_math(
                                gw_b2, gw_b3, smem_xh, xh_base_t + xh_off3
                            )
                            + sf_g4
                            * self._block_dot_hfma2_for_math(
                                gw_c0, gw_c1, smem_xh, xh_base_t + xh_off4
                            )
                            + sf_g5
                            * self._block_dot_hfma2_for_math(
                                gw_c2, gw_c3, smem_xh, xh_base_t + xh_off5
                            )
                            + sf_g6
                            * self._block_dot_hfma2_for_math(
                                gw_d0, gw_d1, smem_xh, xh_base_t + xh_off6
                            )
                            + sf_g7
                            * self._block_dot_hfma2_for_math(
                                gw_d2, gw_d3, smem_xh, xh_base_t + xh_off7
                            )
                            + sf_g8
                            * self._block_dot_hfma2_for_math(
                                gw_e0, gw_e1, smem_xh, xh_base_t + xh_off8
                            )
                            + sf_g9
                            * self._block_dot_hfma2_for_math(
                                gw_e2, gw_e3, smem_xh, xh_base_t + xh_off9
                            )
                            + sf_g10
                            * self._block_dot_hfma2_for_math(
                                gw_f0, gw_f1, smem_xh, xh_base_t + xh_off10
                            )
                            + sf_g11
                            * self._block_dot_hfma2_for_math(
                                gw_f2, gw_f3, smem_xh, xh_base_t + xh_off11
                            )
                        )
                    elif cutlass.const_expr(self.is_gated):
                        dot_u0, dot_g0 = self._block_dot_hfma2_pair_for_math(
                            uw_a0, uw_a1, gw_a0, gw_a1, smem_xh, xh_base_t + xh_off0
                        )
                        dot_u1, dot_g1 = self._block_dot_hfma2_pair_for_math(
                            uw_a2, uw_a3, gw_a2, gw_a3, smem_xh, xh_base_t + xh_off1
                        )
                        dot_u2, dot_g2 = self._block_dot_hfma2_pair_for_math(
                            uw_b0, uw_b1, gw_b0, gw_b1, smem_xh, xh_base_t + xh_off2
                        )
                        dot_u3, dot_g3 = self._block_dot_hfma2_pair_for_math(
                            uw_b2, uw_b3, gw_b2, gw_b3, smem_xh, xh_base_t + xh_off3
                        )
                        dot_u4, dot_g4 = self._block_dot_hfma2_pair_for_math(
                            uw_c0, uw_c1, gw_c0, gw_c1, smem_xh, xh_base_t + xh_off4
                        )
                        dot_u5, dot_g5 = self._block_dot_hfma2_pair_for_math(
                            uw_c2, uw_c3, gw_c2, gw_c3, smem_xh, xh_base_t + xh_off5
                        )
                        dot_u6, dot_g6 = self._block_dot_hfma2_pair_for_math(
                            uw_d0, uw_d1, gw_d0, gw_d1, smem_xh, xh_base_t + xh_off6
                        )
                        dot_u7, dot_g7 = self._block_dot_hfma2_pair_for_math(
                            uw_d2, uw_d3, gw_d2, gw_d3, smem_xh, xh_base_t + xh_off7
                        )
                        dot_u8, dot_g8 = self._block_dot_hfma2_pair_for_math(
                            uw_e0, uw_e1, gw_e0, gw_e1, smem_xh, xh_base_t + xh_off8
                        )
                        dot_u9, dot_g9 = self._block_dot_hfma2_pair_for_math(
                            uw_e2, uw_e3, gw_e2, gw_e3, smem_xh, xh_base_t + xh_off9
                        )
                        dot_u10, dot_g10 = self._block_dot_hfma2_pair_for_math(
                            uw_f0, uw_f1, gw_f0, gw_f1, smem_xh, xh_base_t + xh_off10
                        )
                        dot_u11, dot_g11 = self._block_dot_hfma2_pair_for_math(
                            uw_f2, uw_f3, gw_f2, gw_f3, smem_xh, xh_base_t + xh_off11
                        )
                        partial_up = (
                            sf_u0 * dot_u0
                            + sf_u1 * dot_u1
                            + sf_u2 * dot_u2
                            + sf_u3 * dot_u3
                            + sf_u4 * dot_u4
                            + sf_u5 * dot_u5
                            + sf_u6 * dot_u6
                            + sf_u7 * dot_u7
                            + sf_u8 * dot_u8
                            + sf_u9 * dot_u9
                            + sf_u10 * dot_u10
                            + sf_u11 * dot_u11
                        )
                        partial_gate = (
                            sf_g0 * dot_g0
                            + sf_g1 * dot_g1
                            + sf_g2 * dot_g2
                            + sf_g3 * dot_g3
                            + sf_g4 * dot_g4
                            + sf_g5 * dot_g5
                            + sf_g6 * dot_g6
                            + sf_g7 * dot_g7
                            + sf_g8 * dot_g8
                            + sf_g9 * dot_g9
                            + sf_g10 * dot_g10
                            + sf_g11 * dot_g11
                        )
                elif cutlass.const_expr(cfg.k_segments == 2):
                    if cutlass.const_expr(self.is_gated):
                        uw_a0, uw_a1, uw_a2, uw_a3 = ld_global_nc_v4_u32(up_byte_addr)
                    gw_a0, gw_a1, gw_a2, gw_a3 = ld_global_nc_v4_u32(gate_byte_addr)

                    col_blk_off_2 = Int64(lane // Int32(2)) * Int64(512)
                    sf_shift0 = Uint32((lane % Int32(2)) * Int32(16))
                    sf_shift1 = sf_shift0 + Uint32(8)

                    if cutlass.const_expr(self.is_gated):
                        sf_word_u = ld_global_nc_u32(
                            w1s_base_addr + ebase_sf + bsf_base_u + col_blk_off_2
                        )
                        sf_u0 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_u >> sf_shift0) & Uint32(0xFF)
                        )
                        sf_u1 = cvt_e4m3_to_f32_via_f16(
                            (sf_word_u >> sf_shift1) & Uint32(0xFF)
                        )
                    sf_word_g = ld_global_nc_u32(
                        w1s_base_addr + ebase_sf + bsf_base_g + col_blk_off_2
                    )
                    sf_g0 = cvt_e4m3_to_f32_via_f16(
                        (sf_word_g >> sf_shift0) & Uint32(0xFF)
                    )
                    sf_g1 = cvt_e4m3_to_f32_via_f16(
                        (sf_word_g >> sf_shift1) & Uint32(0xFF)
                    )

                    seg_blk0 = lane * Int32(2)
                    seg_blk1 = seg_blk0 + Int32(1)
                    xh_base0 = xh_buf_base + seg_blk0 * Int32(8) + seg_blk0 // Int32(8)
                    xh_base1 = xh_buf_base + seg_blk1 * Int32(8) + seg_blk1 // Int32(8)

                    if cutlass.const_expr(not self.is_gated):
                        dot_g0 = self._block_dot_hfma2_for_math(
                            gw_a0, gw_a1, smem_xh, xh_base0
                        )
                        dot_g1 = self._block_dot_hfma2_for_math(
                            gw_a2, gw_a3, smem_xh, xh_base1
                        )
                        partial_gate = sf_g0 * dot_g0 + sf_g1 * dot_g1
                    elif cutlass.const_expr(self.is_gated):
                        dot_u0a, dot_g0a = self._block_dot4_pair_for_math(
                            uw_a0, gw_a0, smem_xh, xh_base0
                        )
                        dot_u0b, dot_g0b = self._block_dot4_pair_for_math(
                            uw_a1, gw_a1, smem_xh, xh_base0 + Int32(4)
                        )
                        dot_u1a, dot_g1a = self._block_dot4_pair_for_math(
                            uw_a2, gw_a2, smem_xh, xh_base1
                        )
                        dot_u1b, dot_g1b = self._block_dot4_pair_for_math(
                            uw_a3, gw_a3, smem_xh, xh_base1 + Int32(4)
                        )
                        partial_up = sf_u0 * (dot_u0a + dot_u0b) + sf_u1 * (
                            dot_u1a + dot_u1b
                        )
                        partial_gate = sf_g0 * (dot_g0a + dot_g0b) + sf_g1 * (
                            dot_g1a + dot_g1b
                        )
                else:
                    partial_up = Float32(0.0)
                    partial_gate = Float32(0.0)
                    for seg in cutlass.range_constexpr(cfg.k_segments):
                        seg_byte_off = Int64(seg * (_BLOCK_SIZE // 2))
                        scale_col = lane * Int32(cfg.k_segments) + Int32(seg)
                        sf_group_off = Int64(scale_col // Int32(4)) * Int64(512)
                        sf_shift = Uint32((scale_col % Int32(4)) * Int32(8))
                        xh_base = (
                            xh_buf_base
                            + scale_col * Int32(_BLOCK_SIZE // 2)
                            + scale_col // Int32(8)
                        )

                        gw0 = ld_global_nc_u32(gate_byte_addr + seg_byte_off)
                        gw1 = ld_global_nc_u32(gate_byte_addr + seg_byte_off + Int64(4))
                        sf_word_g = ld_global_nc_u32(
                            w1s_base_addr + ebase_sf + bsf_base_g + sf_group_off
                        )
                        sf_g = cvt_e4m3_to_f32_via_f16(
                            (sf_word_g >> sf_shift) & Uint32(0xFF)
                        )

                        if cutlass.const_expr(self.is_gated):
                            uw0 = ld_global_nc_u32(up_byte_addr + seg_byte_off)
                            uw1 = ld_global_nc_u32(
                                up_byte_addr + seg_byte_off + Int64(4)
                            )
                            sf_word_u = ld_global_nc_u32(
                                w1s_base_addr + ebase_sf + bsf_base_u + sf_group_off
                            )
                            sf_u = cvt_e4m3_to_f32_via_f16(
                                (sf_word_u >> sf_shift) & Uint32(0xFF)
                            )
                            dot_u, dot_g = self._block_dot_hfma2_pair_for_math(
                                uw0, uw1, gw0, gw1, smem_xh, xh_base
                            )
                            partial_up = partial_up + sf_u * dot_u
                            partial_gate = partial_gate + sf_g * dot_g
                        else:
                            partial_gate = (
                                partial_gate
                                + sf_g
                                * self._block_dot_hfma2_for_math(
                                    gw0,
                                    gw1,
                                    smem_xh,
                                    xh_base,
                                )
                            )
                # ---- Activation + intermediate quant ----
                gate_red = cute.arch.warp_reduction_sum(partial_gate) * alpha_fc1
                if cutlass.const_expr(self.is_gated):
                    up_red = cute.arch.warp_reduction_sum(partial_up) * alpha_fc1
                if lane == Int32(0):
                    if cutlass.const_expr(self.is_gated):
                        sigmoid = Float32(1.0) / (
                            Float32(1.0) + cute.math.exp(-gate_red, fastmath=False)
                        )
                        activated = sigmoid * gate_red * up_red
                    else:
                        relu_val = fmax_f32(gate_red, Float32(0.0))
                        activated = relu_val * relu_val
                    smem_int[i_local] = Float32(BFloat16(activated))

            # Look-ahead: quantize next task into other buffer (k_segments==2 only)
            if cutlass.const_expr(cfg.k_segments == 2):
                next_task = fc1_task + Int32(gdim_x)
                need_quant_next = Int32(0)
                t_next = t
                next_route = route_idx
                if next_task < fc1_task_count:
                    next_route = next_task // Int32(cfg.fc1_chunks)
                    t_next = next_route // Int32(cfg.num_topk)
                    need_quant_next = Int32(1)
                    if cutlass.const_expr(self.share_input_across_experts):
                        need_quant_next = Int32(1) if t_next != t else Int32(0)
                if need_quant_next > Int32(0):
                    next_eid_addr = t_next * Int32(cfg.num_topk) + (
                        next_route - t_next * Int32(cfg.num_topk)
                    )
                    gs_fc1_next = input_gs[Int32(topk_ids[next_eid_addr])]
                    next_buf_base = (Int32(1) - buf_idx) * Int32(cfg.smem_xh_size)
                    in_blk = tidx
                    while in_blk < Int32(cfg.k_dim // _BLOCK_SIZE):
                        x_base = t_next * Int32(cfg.k_dim) + in_blk * Int32(_BLOCK_SIZE)
                        pad_off = in_blk // Int32(8)
                        phys_base = in_blk * Int32(_BLOCK_SIZE // 2) + pad_off
                        if cutlass.const_expr(self.w4a16_mode):
                            for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                                v0 = Float32(a_input[x_base + Int32(i * 2)])
                                v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                                smem_xh[next_buf_base + phys_base + Int32(i)] = (
                                    pack_f32x2_to_f16x2(v0, v1)
                                )
                        else:
                            blk_peak = Float32(0.0)
                            for i in cutlass.range_constexpr(_BLOCK_SIZE):
                                v = Float32(a_input[x_base + Int32(i)])
                                abs_v = v
                                if v < Float32(0.0):
                                    abs_v = -v
                                if abs_v > blk_peak:
                                    blk_peak = abs_v
                            q_scale = nvfp4_scale_from_amax(blk_peak, gs_fc1_next)
                            if q_scale > Float32(_FP8_E4M3_MAX):
                                q_scale = Float32(_FP8_E4M3_MAX)
                            sf_val = cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))
                            eff_scale = Float32(0.0)
                            if gs_fc1_next != Float32(0.0):
                                eff_scale = sf_val / gs_fc1_next
                            if eff_scale < Float32(1e-30):
                                eff_scale = Float32(1e-30)
                            for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                                v0 = Float32(a_input[x_base + Int32(i * 2)])
                                v1 = Float32(a_input[x_base + Int32(i * 2 + 1)])
                                f0, f1 = quant_dequant_2(v0, v1, sf_val, eff_scale)
                                smem_xh[next_buf_base + phys_base + Int32(i)] = (
                                    pack_f32x2_to_f16x2(f0, f1)
                                )
                        in_blk += Int32(_BLOCK_DIM)

            cute.arch.sync_threads()

            gs_fc2_eff = gs_fc2
            if cutlass.const_expr(self.dynamic_down_scale):
                local_max = Float32(0.0)
                scan_idx = Int32(tidx)
                while scan_idx < Int32(cfg.i_chunk):
                    v = smem_int[scan_idx]
                    abs_v = v
                    if v < Float32(0.0):
                        abs_v = -v
                    local_max = fmax_f32(local_max, abs_v)
                    scan_idx += Int32(_BLOCK_DIM)
                warp_max = warp_reduce(local_max, fmax_f32)
                if lane == Int32(0):
                    reduce_scratch[warp_id] = warp_max
                cute.arch.sync_threads()
                tile_amax = Float32(0.0)
                if warp_id == Int32(0):
                    if lane < Int32(_NUM_WARPS):
                        tile_amax = reduce_scratch[lane]
                    tile_amax = warp_reduce(tile_amax, fmax_f32)
                    if lane == Int32(0):
                        reduce_scratch[Int32(0)] = tile_amax
                cute.arch.sync_threads()
                gs_fc2_eff = Float32(0.0)
                if reduce_scratch[Int32(0)] > Float32(0.0):
                    gs_fc2_eff = (
                        Float32(_FC2_TILE_RECIP_GS_NUM) / reduce_scratch[Int32(0)]
                    )
                gs_fc2_eff = fmax_f32(gs_fc2_eff, Float32(1.0e-12))
            fc2_rescale = Float32(1.0)
            if cutlass.const_expr(self.dynamic_down_scale):
                if gs_fc2_eff != Float32(0.0):
                    fc2_rescale = gs_fc2 / gs_fc2_eff
            if tidx < Int32(cfg.inter_blocks):
                mid_blk = tidx
                if cutlass.const_expr(self.w4a16_mode):
                    for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                        v0 = smem_int[mid_blk * Int32(_BLOCK_SIZE) + Int32(i * 2)]
                        v1 = smem_int[mid_blk * Int32(_BLOCK_SIZE) + Int32(i * 2 + 1)]
                        half_base = chunk_idx * Int32(
                            cfg.i_chunk // 2
                        ) + mid_blk * Int32(_BLOCK_SIZE // 2)
                        n_blk = half_base // Int32(128)
                        h_local = half_base - n_blk * Int32(128)
                        h_i = h_local + Int32(i)
                        packed_idx = (
                            t * Int32(cfg.inter_u32)
                            + k_idx * Int32(cfg.fc2_n_chunks * 128)
                            + n_blk * Int32(128)
                            + (h_i % Int32(4)) * Int32(32)
                            + (h_i // Int32(4))
                        )
                        intermediate[packed_idx] = pack_f32x2_to_f16x2(v0, v1)
                else:
                    blk_peak = Float32(0.0)
                    for i in cutlass.range_constexpr(_BLOCK_SIZE):
                        v = smem_int[mid_blk * Int32(_BLOCK_SIZE) + Int32(i)]
                        abs_v = v
                        if v < Float32(0.0):
                            abs_v = -v
                        if abs_v > blk_peak:
                            blk_peak = abs_v
                    q_scale = nvfp4_scale_from_amax(blk_peak, gs_fc2_eff)
                    if q_scale > Float32(_FP8_E4M3_MAX):
                        q_scale = Float32(_FP8_E4M3_MAX)
                    sf_val = cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))
                    eff_scale = Float32(0.0)
                    if gs_fc2_eff != Float32(0.0):
                        eff_scale = sf_val / gs_fc2_eff
                    if eff_scale < Float32(1e-30):
                        eff_scale = Float32(1e-30)
                    for i in cutlass.range_constexpr(_BLOCK_SIZE // 2):
                        v0 = smem_int[mid_blk * Int32(_BLOCK_SIZE) + Int32(i * 2)]
                        v1 = smem_int[mid_blk * Int32(_BLOCK_SIZE) + Int32(i * 2 + 1)]
                        f0, f1 = quant_dequant_2(v0, v1, sf_val, eff_scale)
                        f0 = f0 * fc2_rescale
                        f1 = f1 * fc2_rescale
                        half_base = chunk_idx * Int32(
                            cfg.i_chunk // 2
                        ) + mid_blk * Int32(_BLOCK_SIZE // 2)
                        n_blk = half_base // Int32(128)
                        h_local = half_base - n_blk * Int32(128)
                        h_i = h_local + Int32(i)
                        packed_idx = (
                            t * Int32(cfg.inter_u32)
                            + k_idx * Int32(cfg.fc2_n_chunks * 128)
                            + n_blk * Int32(128)
                            + (h_i % Int32(4)) * Int32(32)
                            + (h_i // Int32(4))
                        )
                        intermediate[packed_idx] = pack_f32x2_to_f16x2(f0, f1)

            cute.arch.sync_threads()
            if cutlass.const_expr(cfg.k_segments == 2):
                if need_quant_next > Int32(0):
                    buf_idx = Int32(1) - buf_idx
            fc1_task += Int32(gdim_x)

        if cutlass.const_expr(self.m_const == 1):
            _token_publish_fc1_ready(
                barrier_count,
                barrier_epoch,
                Int32(0),
                m1_epoch0,
                Int32(gdim_x),
                is_cta_leader,
            )
            _token_wait_fc1_ready(barrier_epoch, Int32(0), m1_epoch0, is_cta_leader)
        else:
            self._resident_grid_barrier(
                barrier_count, barrier_epoch, Int32(gdim_x), is_cta_leader
            )

        # ===================================================================
        # PHASE 2: FC2 output
        # ===================================================================
        w2_base_addr = w2_weights.iterator.toint()
        w2s_base_addr = w2_scales.iterator.toint()
        # ---- m==1 FC2 rowpair ----
        if cutlass.const_expr(self.m_const == 1):
            fc2_chunks_m1 = Int32(cfg.k_dim // (_K_PER_CTA * 2))
            if cutlass.const_expr(self.m1_fc2_onepass):
                fc2_task = Int32(bidx_x)
                if fc2_task < fc2_chunks_m1:
                    if cutlass.const_expr(cfg.fc2_n_chunks > 1):
                        self._m1_fc2_rowpair_wide(
                            fc2_task,
                            warp_id,
                            lane,
                            w2_base_addr,
                            w2s_base_addr,
                            intermediate,
                            w2_alphas,
                            topk_ids,
                            topk_weights,
                            scatter_output,
                        )
                    else:
                        self._m1_fc2_rowpair_narrow(
                            fc2_task,
                            warp_id,
                            lane,
                            w2_base_addr,
                            w2s_base_addr,
                            intermediate,
                            w2_alphas,
                            topk_ids,
                            topk_weights,
                            scatter_output,
                        )
            else:
                fc2_task = Int32(bidx_x)
                while fc2_task < fc2_chunks_m1:
                    if cutlass.const_expr(cfg.fc2_n_chunks > 1):
                        self._m1_fc2_rowpair_wide(
                            fc2_task,
                            warp_id,
                            lane,
                            w2_base_addr,
                            w2s_base_addr,
                            intermediate,
                            w2_alphas,
                            topk_ids,
                            topk_weights,
                            scatter_output,
                        )
                    else:
                        self._m1_fc2_rowpair_narrow(
                            fc2_task,
                            warp_id,
                            lane,
                            w2_base_addr,
                            w2s_base_addr,
                            intermediate,
                            w2_alphas,
                            topk_ids,
                            topk_weights,
                            scatter_output,
                        )
                    fc2_task += Int32(gdim_x)

        # ---- m>=2 FC2 rowquad ----
        else:
            fc2_task_count = Int32((self.m_const * cfg.k_dim) // (_K_PER_CTA * 4))
            if cutlass.const_expr(self.w4a16_mode and cfg.fc2_n_chunks == 1):
                fc2_task_count = Int32((self.m_const * cfg.k_dim) // (_K_PER_CTA * 2))
            fc2_task = Int32(bidx_x)
            while fc2_task < fc2_task_count:
                if cutlass.const_expr(self.w4a16_mode and cfg.fc2_n_chunks == 1):
                    self._m2_fc2_rowpair_narrow(
                        fc2_task,
                        warp_id,
                        lane,
                        w2_base_addr,
                        w2s_base_addr,
                        intermediate,
                        w2_alphas,
                        topk_ids,
                        topk_weights,
                        scatter_output,
                    )
                elif cutlass.const_expr(cfg.fc2_n_chunks > 1):
                    self._m2_fc2_rowquad_wide(
                        fc2_task,
                        warp_id,
                        lane,
                        w2_base_addr,
                        w2s_base_addr,
                        intermediate,
                        w2_alphas,
                        topk_ids,
                        topk_weights,
                        scatter_output,
                    )
                else:
                    self._m2_fc2_rowquad_narrow(
                        fc2_task,
                        warp_id,
                        lane,
                        w2_base_addr,
                        w2s_base_addr,
                        intermediate,
                        w2_alphas,
                        topk_ids,
                        topk_weights,
                        scatter_output,
                    )
                fc2_task += Int32(gdim_x)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w1_ptr: cute.Pointer,
        w1s_ptr: cute.Pointer,
        w1a_ptr: cute.Pointer,
        a1_ptr: cute.Pointer,
        a2_ptr: cute.Pointer,
        inter_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2s_ptr: cute.Pointer,
        w2a_ptr: cute.Pointer,
        tid_ptr: cute.Pointer,
        tw_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        barrier_count_ptr: cute.Pointer,
        barrier_epoch_ptr: cute.Pointer,
        m_val: Int32,
        grid_x: Int32,
        stream,
    ):
        cfg = self._cfg
        a_input = cute.make_tensor(
            x.iterator, cute.make_layout(Int32(m_val * cfg.k_dim))
        )
        w1_weights = cute.make_tensor(
            w1_ptr, cute.make_layout(Int64(cfg.weight_E * cfg.two_n * cfg.k_half))
        )
        w1_scales = cute.make_tensor(
            w1s_ptr,
            cute.make_layout(Int64(cfg.weight_E * cfg.w1_sf_rows * cfg.w1_sf_cols)),
        )
        w1_alphas = cute.make_tensor(w1a_ptr, cute.make_layout(Int32(cfg.weight_E)))
        input_gs = cute.make_tensor(a1_ptr, cute.make_layout(Int32(cfg.weight_E)))
        down_input_scale = cute.make_tensor(
            a2_ptr, cute.make_layout(Int32(cfg.weight_E))
        )
        intermediate = cute.make_tensor(
            inter_ptr, cute.make_layout(Int32(m_val * cfg.inter_u32))
        )
        w2_weights = cute.make_tensor(
            w2_ptr, cute.make_layout(Int64(cfg.weight_E * cfg.k_dim * cfg.n_half))
        )
        w2_scales = cute.make_tensor(
            w2s_ptr,
            cute.make_layout(Int64(cfg.weight_E * cfg.w2_sf_rows * cfg.w2_sf_cols)),
        )
        w2_alphas = cute.make_tensor(w2a_ptr, cute.make_layout(Int32(cfg.weight_E)))
        topk_ids_tensor = cute.make_tensor(
            tid_ptr, cute.make_layout(Int32(m_val * cfg.num_topk))
        )
        topk_weights_tensor = cute.make_tensor(
            tw_ptr, cute.make_layout(Int32(m_val * cfg.num_topk))
        )
        scatter_output_tensor = cute.make_tensor(
            out_ptr, cute.make_layout(Int32(m_val * cfg.k_dim))
        )
        barrier_slots = m_val * Int32(cfg.num_topk + 16)
        barrier_count = cute.make_tensor(
            barrier_count_ptr, cute.make_layout(barrier_slots)
        )
        barrier_epoch = cute.make_tensor(
            barrier_epoch_ptr, cute.make_layout(barrier_slots)
        )

        self.kernel(
            a_input,
            w1_weights,
            w1_scales,
            w1_alphas,
            input_gs,
            down_input_scale,
            intermediate,
            w2_weights,
            w2_scales,
            w2_alphas,
            topk_ids_tensor,
            topk_weights_tensor,
            scatter_output_tensor,
            barrier_count,
            barrier_epoch,
            m_val,
        ).launch(
            grid=(grid_x, Int32(1), Int32(1)),
            block=(_BLOCK_DIM, 1, 1),
            smem=0,
            stream=stream,
        )

    @staticmethod
    def launch(
        compiled_fn,
        *,
        x: torch.Tensor,
        w1_fp4: torch.Tensor,
        w1_blockscale: torch.Tensor,
        w1_alphas: torch.Tensor,
        a1_gscale: torch.Tensor,
        a2_gscale: torch.Tensor,
        inter_fp32: torch.Tensor,
        w2_fp4: torch.Tensor,
        w2_blockscale: torch.Tensor,
        w2_alphas: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
        barrier_count: torch.Tensor,
        barrier_epoch: torch.Tensor,
        m: int,
        grid_x: int,
    ):
        stream = current_cuda_stream()

        compiled_fn(
            x,
            w1_fp4.data_ptr(),
            w1_blockscale.view(torch.uint8).data_ptr(),
            w1_alphas.data_ptr(),
            a1_gscale.data_ptr(),
            a2_gscale.data_ptr(),
            inter_fp32.view(torch.uint32).data_ptr(),
            w2_fp4.data_ptr(),
            w2_blockscale.view(torch.uint8).data_ptr(),
            w2_alphas.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            barrier_count.data_ptr(),
            barrier_epoch.data_ptr(),
            Int32(m),
            Int32(grid_x),
            stream,
        )


__all__ = ["MoEDirectMicroKernel"]
