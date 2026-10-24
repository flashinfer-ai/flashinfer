# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-N native MXFP4-weight x MXFP8-activation tcgen05 mainloop.

This module owns only instruction issue.  TMA, SMEM/TMEM layouts, pipelines,
barriers, and accumulator lifetime belong to the integrating MegaMoE kernel.

The kernel uses swap-AB orientation:

* instruction A: MXFP4 E2M1 weights, ``(M, K, L)``;
* instruction B: MXFP8 E4M3 activations, ``(N, K, L)``;
* SFA/SFB: E8M0, one scale per 32 K elements; and
* D: FP32 accumulator in TMEM.

Thus the instruction computes the transpose-oriented equivalent of the logical
activation x weight GEMM.  Keeping this distinction explicit is important:
``idesc.atype`` is E2M1 and ``idesc.btype`` is E4M3.

The regular CuTe tcgen05 wrapper builds a static instruction descriptor.  A
MegaMoE tile can contain fewer tokens than its configured N extent, so this
component emits literal PTX and fills ``idesc.n_dim`` from the runtime valid
token count.  All other descriptor fields are fixed by the contract above.
"""

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm
from cutlass.cutlass_dsl import Boolean, Int32, dsl_user_op


A_DTYPE = cutlass.Float4E2M1FN
B_DTYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
SF_VEC_SIZE = 32
UMMA_K = 32


# PTX ISA tcgen05.mma.kind::mxf8f6f4 instruction descriptor fields.
_BIT_B_SF_ID = 4
_BIT_A_FORMAT = 7
_BIT_B_FORMAT = 10
_BIT_A_MAJOR = 15
_BIT_B_MAJOR = 16
_BIT_N_DIM = 17
_BIT_SCALE_FORMAT = 23
_BIT_A_SF_LAYOUT = 26
_BIT_M_DIM = 27
_BIT_A_SF_ID = 29
_BIT_K_SIZE = 31

_FORMAT_E4M3 = 0
_FORMAT_E2M1 = 5
_SCALE_FORMAT_UE8M0 = 1
_SF_LAYOUT_32_LANE = 0
_K_SIZE_DENSE_32 = 0


def validate_mma_contract(
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_tiler_mnk: tuple[int, int, int],
) -> None:
    """Validate the compile-time contract consumed by the instruction issuer."""
    if a_dtype is not A_DTYPE:
        raise ValueError("swap-AB instruction A must be MXFP4 Float4E2M1FN weights")
    if b_dtype is not B_DTYPE:
        raise ValueError("swap-AB instruction B must be MXFP8 Float8E4M3FN activations")
    if sf_dtype is not SF_DTYPE:
        raise ValueError("MXFP8 x MXFP4 scale factors must be Float8E8M0FNU")
    if sf_vec_size != SF_VEC_SIZE:
        raise ValueError(f"MXFP8 x MXFP4 requires sf_vec_size={SF_VEC_SIZE}")

    mma_m, mma_n, mma_k = mma_tiler_mnk
    if mma_m not in (128, 256):
        raise ValueError(f"MMA M must be 128 (1 CTA) or 256 (2 CTA), got {mma_m}")
    if mma_n <= 0 or mma_n > 256 or mma_n % 16:
        raise ValueError(
            f"MMA N must be a positive multiple of 16 up to 256, got {mma_n}"
        )
    if mma_k <= 0 or mma_k % UMMA_K:
        raise ValueError(f"MMA K must be a positive multiple of {UMMA_K}, got {mma_k}")


def align_valid_tokens(valid_tokens: int) -> int:
    """Host-side mirror of the runtime valid-token alignment policy."""
    if valid_tokens < 0:
        raise ValueError(f"valid_tokens must be non-negative, got {valid_tokens}")
    return (valid_tokens + 15) & -16


def _align16(x):
    """Round an Int32 SSA value up to a multiple of 16."""
    return (Int32(x) + Int32(15)) & Int32(-16)


@dsl_user_op
def compute_non_leader_cta_load_shift(
    *,
    valid_tokens_in_tile,
    mma_tiler_n: int,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """Return the non-leader CTA B-load shift for a dynamic-N 2-CTA MMA.

    TMA partitions B at ``mma_tiler_n / 2``, while hardware partitions the
    issued tile at ``align16(valid_tokens) / 2``.  The caller applies this
    offset with ``cute.domain_offset`` to the non-leader CTA's activation
    tensor before constructing its TMA slice.
    """
    if mma_tiler_n <= 0 or mma_tiler_n > 256 or mma_tiler_n % 16:
        raise ValueError("mma_tiler_n must be a positive multiple of 16 up to 256")
    return (_align16(valid_tokens_in_tile) >> Int32(1)) - Int32(mma_tiler_n // 2)


def build_static_idesc_base(*, umma_m: int) -> int:
    """Build the static mxf8f6f4 descriptor for swap-AB MXFP4 x MXFP8.

    PTX reserves bits 24-25 for this MMA kind; unlike the mxf4nvf4 descriptor,
    M is a two-bit ``M >> 7`` field at bits 27-28.  This distinction is why the
    W4A4 descriptor builder cannot be reused.
    """
    if umma_m not in (128, 256):
        raise ValueError(f"UMMA M must be 128 or 256, got {umma_m}")

    desc = 0
    desc |= _FORMAT_E2M1 << _BIT_A_FORMAT
    desc |= _FORMAT_E4M3 << _BIT_B_FORMAT
    desc |= 0 << _BIT_A_MAJOR  # K-major
    desc |= 0 << _BIT_B_MAJOR  # K-major
    desc |= _SCALE_FORMAT_UE8M0 << _BIT_SCALE_FORMAT
    desc |= _SF_LAYOUT_32_LANE << _BIT_A_SF_LAYOUT
    desc |= (umma_m >> 7) << _BIT_M_DIM
    desc |= _K_SIZE_DENSE_32 << _BIT_K_SIZE
    return desc & 0xFFFFFFFF


def build_idesc_host(
    *,
    umma_m: int,
    valid_tokens_in_tile: int,
    sfa_tmem_addr: int = 0,
    sfb_tmem_addr: int = 0,
) -> int:
    """Host-side descriptor encoder used by focused contract tests."""
    aligned_n = align_valid_tokens(valid_tokens_in_tile)
    if aligned_n == 0 or aligned_n > 256:
        raise ValueError(f"aligned dynamic N must be in [16, 256], got {aligned_n}")
    desc = build_static_idesc_base(umma_m=umma_m)
    desc |= (aligned_n >> 3) << _BIT_N_DIM
    desc |= ((sfa_tmem_addr >> 30) & 0x3) << _BIT_A_SF_ID
    desc |= ((sfb_tmem_addr >> 30) & 0x3) << _BIT_B_SF_ID
    return desc & 0xFFFFFFFF


@dsl_user_op
def compute_idesc(
    *,
    static_base: int,
    n_dim_value,
    sfa_tmem_addr_i32,
    sfb_tmem_addr_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Int32:
    """OR runtime N and scale-factor data IDs into a static descriptor."""
    idesc = Int32(static_base) | (Int32(n_dim_value) << _BIT_N_DIM)
    # Shift first, then mask: Int32 ``>>`` is arithmetic, so shifting the
    # masked top bits sign-extends ones over the descriptor whenever bit 31
    # is set (scale-factor IDs 2 and 3, i.e. K steps 2 and 3 of a K tile).
    sfa_id = (Int32(sfa_tmem_addr_i32) >> Int32(30)) & Int32(0x3)
    sfb_id = (Int32(sfb_tmem_addr_i32) >> Int32(30)) & Int32(0x3)
    idesc = idesc | (sfa_id << Int32(_BIT_A_SF_ID))
    idesc = idesc | (sfb_id << Int32(_BIT_B_SF_ID))
    return idesc


def _smem_desc_to_i64(smem_desc_value: ir.Value) -> ir.Value:
    i64_ty = ir.IntegerType.get_signless(64)
    return builtin.unrealized_conversion_cast([i64_ty], [smem_desc_value])


def _tmem_ptr_to_i32(tmem_ptr_value: ir.Value) -> ir.Value:
    i32_ty = ir.IntegerType.get_signless(32)
    return builtin.unrealized_conversion_cast([i32_ty], [tmem_ptr_value])


def _as_value(value) -> ir.Value:
    return value.value if hasattr(value, "value") else value


@dsl_user_op
def _tcgen05_mma_mxf8f6f4_block_scale_block32(
    *,
    cta_group: int,
    d_tmem_i32,
    a_desc_i64,
    b_desc_i64,
    idesc_i32,
    enable_input_d_i32,
    sfa_tmem_i32,
    sfb_tmem_i32,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Emit native E2M1(A) x E4M3(B), UE8M0 block-32 tcgen05 MMA."""
    if cta_group not in (1, 2):
        raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")
    llvm.inline_asm(
        None,
        [
            d_tmem_i32,
            a_desc_i64,
            b_desc_i64,
            idesc_i32,
            enable_input_d_i32,
            sfa_tmem_i32,
            sfb_tmem_i32,
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $4, 0;\n\t"
        f"tcgen05.mma.cta_group::{cta_group}.kind::mxf8f6f4.block_scale.block32 "
        "[$0], $1, $2, $3, [$5], [$6], p;\n\t"
        "}\n",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_dynamic_block_scaled_mma_tile(
    *,
    acc_tensor,
    a_frag_tile,
    b_frag_tile,
    sfa_tensor,
    sfb_tensor,
    k_tile_idx,
    valid_tokens_in_tile,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_tiler_mnk: tuple[int, int, int],
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Issue one K tile of native dynamic-N MXFP4 x MXFP8 MMAs.

    Fragment tensors use ``(V, MN-count, K-count)`` and must be built from a
    tiled MMA matching the explicit dtype/SF contract.  Each K-count slice is
    one dense K=32 instruction and therefore consumes exactly one E8M0 scale
    from each scale-factor fragment.

    ``valid_tokens_in_tile`` must be in ``[1, mma_tiler_n]``.  It is rounded to
    16 before being encoded as ``idesc.n_dim = aligned_n >> 3``.  The caller
    owns the corresponding non-leader B-load shift for 2-CTA operation.
    """
    validate_mma_contract(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        mma_tiler_mnk=mma_tiler_mnk,
    )
    static_idesc_base = build_static_idesc_base(umma_m=mma_tiler_mnk[0])
    n_dim_value = _align16(valid_tokens_in_tile) >> Int32(3)
    num_k_inner = mma_tiler_mnk[2] // UMMA_K
    cta_group = 2 if mma_tiler_mnk[0] == 256 else 1

    for k_inner in range(num_k_inner):
        a_atom = a_frag_tile[(None, 0, k_inner)]
        b_atom = b_frag_tile[(None, 0, k_inner)]
        sfa_atom = sfa_tensor[(None, 0, k_inner)]
        sfb_atom = sfb_tensor[(None, 0, k_inner)]
        acc_atom = acc_tensor[(None, 0, 0)]

        operand_a = _smem_desc_to_i64(_as_value(a_atom.iterator))
        operand_b = _smem_desc_to_i64(_as_value(b_atom.iterator))
        operand_sfa_i32 = _tmem_ptr_to_i32(_as_value(sfa_atom.iterator))
        operand_sfb_i32 = _tmem_ptr_to_i32(_as_value(sfb_atom.iterator))
        operand_acc_i32 = _tmem_ptr_to_i32(_as_value(acc_atom.iterator))

        idesc = compute_idesc(
            static_base=static_idesc_base,
            n_dim_value=n_dim_value,
            sfa_tmem_addr_i32=operand_sfa_i32,
            sfb_tmem_addr_i32=operand_sfb_i32,
        )
        accum_flag = k_tile_idx != 0 if k_inner == 0 else True

        with cute.arch.elect_one():
            _tcgen05_mma_mxf8f6f4_block_scale_block32(
                cta_group=cta_group,
                d_tmem_i32=operand_acc_i32,
                a_desc_i64=operand_a,
                b_desc_i64=operand_b,
                idesc_i32=idesc.ir_value(),
                enable_input_d_i32=Int32(Boolean(accum_flag)).ir_value(),
                sfa_tmem_i32=operand_sfa_i32,
                sfb_tmem_i32=operand_sfb_i32,
            )
