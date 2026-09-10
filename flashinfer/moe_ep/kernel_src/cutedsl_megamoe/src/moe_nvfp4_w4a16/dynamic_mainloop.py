# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic routed-token BF16 MMA for the W4A16 MegaMoE pipeline.

Mirrors moe_nvfp4_swapab.dynamic_mainloop's runtime-N policy, operand casts,
and per-K atom issue loop. The non-block-scaled BF16 descriptor fields follow
flashinfer/cute_dsl/sparse/sm100_blk64/mma_sm100_desc.py. The caller must use
the matching non-leader TMA-B shift and valid-token epilogue mapping; changing
only the MMA descriptor is insufficient for a two-CTA kernel.
"""

from typing import Optional
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, builtin


def _align16(x):
    """Round Int32 SSA ``x`` up to a multiple of 16 (mask off bottom 4 bits)."""
    return (Int32(x) + Int32(15)) & Int32(-16)


@dsl_user_op
def compute_non_leader_cta_load_shift(
    *,
    valid_tokens_in_tile,  # Int32 SSA
    mma_tiler_n: int,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """Token offset that non-leader CTA's TMA-B read must shift by under 2cta.

    Under dynamic UMMA + 2cta:
      - MMA splits N at align16(valid) / 2
      - TMA static partition splits at mma_tiler_n / 2

    The result ∈ (-mma_tiler_n/2, 0]; apply via
    ``cute.domain_offset((shift, 0, 0), real_b)`` on non-leader CTA only.
    """
    return (_align16(valid_tokens_in_tile) >> Int32(1)) - Int32(mma_tiler_n // 2)


def _smem_desc_to_i64(smem_desc_value: ir.Value) -> ir.Value:
    """Bit-cast cute_nvgpu.smem_desc value -> i64."""
    i64_ty = ir.IntegerType.get_signless(64)
    return builtin.unrealized_conversion_cast([i64_ty], [smem_desc_value])


def _tmem_ptr_to_i32(tmem_ptr_value: ir.Value) -> ir.Value:
    """Bit-cast cute.ptr<tmem> -> i32."""
    i32_ty = ir.IntegerType.get_signless(32)
    return builtin.unrealized_conversion_cast([i32_ty], [tmem_ptr_value])


def _as_value(it) -> ir.Value:
    """Unwrap to underlying ir.Value (cute Pointer has .value)."""
    return it.value if hasattr(it, "value") else it


def bf16_static_idesc(umma_m: int) -> int:
    """BF16 A/B, FP32 accumulation, K-major; leave runtime N bits clear."""
    if umma_m not in (128, 256):
        raise ValueError("Dynamic BF16 MMA supports M128/M256 UMMA.")
    # Existing non-block-scaled descriptor: CFormat.F32 at4; BF16 at7/10;
    # N/8 at17 (runtime), M/16 at24. No negate, sparse, or saturation flags.
    return (1 << 4) | (1 << 7) | (1 << 10) | ((umma_m >> 4) << 24)


@dsl_user_op
def _issue_bf16_atom(
    *,
    cta_group: int,
    a_from_tmem: bool,
    accumulator,
    operand_a,
    operand_b,
    descriptor,
    accumulate,
    loc=None,
    ip=None,
):
    # The literal PTX approach mirrors the vendored dynamic-N helper and
    # avoids unstable private NVVM enum bindings. BF16 uses kind::f16.
    a_text = "[$1]" if a_from_tmem else "$1"
    a_constraint = "r" if a_from_tmem else "l"
    llvm.inline_asm(
        None,
        [accumulator, operand_a, operand_b, descriptor, accumulate],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        f"tcgen05.mma.cta_group::{cta_group}.kind::f16 "
        f"[$0], {a_text}, $2, $3, p;\n\t}}\n",
        f"r,{a_constraint},l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_dynamic_bf16_mma_tile(
    *,
    acc_tensor,
    a_frag_tile,
    b_frag_tile,
    k_tile_idx,
    valid_tokens_in_tile,
    mma_tiler_mnk: tuple,
    a_from_tmem: bool,
    loc=None,
    ip=None,
):
    """Issue BF16 K16 atoms with the existing align16 routed-token policy.

    The caller owns pipeline waits/commits, TMEM allocation, valid-token
    bounds, and the non-leader TMA-B shift. Only positive valid tiles may
    reach this function, and valid_tokens must not exceed the allocated N.
    """
    umma_m, allocated_n, tile_k = mma_tiler_mnk
    if allocated_n not in (64, 128, 256) or tile_k % 16:
        raise ValueError("Unsupported BF16 N/K geometry.")
    if (
        cute.size(a_frag_tile, mode=[1]) != 1
        or cute.size(b_frag_tile, mode=[1]) != 1
        or cute.size(a_frag_tile, mode=[2]) != tile_k // 16
        or cute.size(b_frag_tile, mode=[2]) != tile_k // 16
    ):
        raise ValueError("Dynamic BF16 MMA requires one M/N atom and K16 fragments.")
    descriptor = Int32(bf16_static_idesc(umma_m)) | (
        (_align16(valid_tokens_in_tile) >> Int32(3)) << Int32(17)
    )
    cta_group = 2 if umma_m == 256 else 1
    for inner_k in range(tile_k // 16):
        a_atom = a_frag_tile[(None, 0, inner_k)]
        b_atom = b_frag_tile[(None, 0, inner_k)]
        acc_atom = acc_tensor[(None, 0, 0)]
        a_value = _as_value(a_atom.iterator)
        a_operand = (
            _tmem_ptr_to_i32(a_value) if a_from_tmem else _smem_desc_to_i64(a_value)
        )
        accumulate = k_tile_idx != 0 if inner_k == 0 else True
        with cute.arch.elect_one():
            _issue_bf16_atom(
                cta_group=cta_group,
                a_from_tmem=a_from_tmem,
                accumulator=_tmem_ptr_to_i32(_as_value(acc_atom.iterator)),
                operand_a=a_operand,
                operand_b=_smem_desc_to_i64(_as_value(b_atom.iterator)),
                descriptor=descriptor.ir_value(),
                accumulate=Int32(Boolean(accumulate)).ir_value(),
                loc=loc,
                ip=ip,
            )


__all__ = [
    "bf16_static_idesc",
    "compute_non_leader_cta_load_shift",
    "issue_dynamic_bf16_mma_tile",
]
