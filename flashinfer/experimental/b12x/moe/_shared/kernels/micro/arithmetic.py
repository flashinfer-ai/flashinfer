"""Packed arithmetic for small-batch routed MoE."""

from __future__ import annotations

from cutlass import Float32
from cutlass.cutlass_dsl import Int32
from typing import Tuple
from cutlass import Uint32
import cutlass.cute as cute
from b12x._lib.intrinsics.dot import fp4_dot4_sum_f32acc
from b12x._lib.intrinsics.dot import fp4_dot8_sum_f32acc


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
