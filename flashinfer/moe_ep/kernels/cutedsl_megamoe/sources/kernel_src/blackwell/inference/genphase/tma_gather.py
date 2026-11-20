"""Blackwell replacement for the Rubin TMA TILE_GATHER4 load used by the gen-phase FC1 gather.

The default uses ordinary tile-mode loads to retain the existing descriptor and toolchain path. One
gather4 writes four consecutive images of the descriptor's ``(1, tile_k)`` box, one per row coordinate, starting at
the destination address. The same bytes land in the same places when each row is loaded by an ordinary tile-mode
tensor load of that box at ``destination + i * box_bytes``: the box is one 128-byte swizzle atom, so every row image
is self-contained. The four loads are four TMA transactions instead of one, but they complete the same number of
transaction bytes on the same mbarrier, so the caller's expected-transaction accounting does not change.

``native_tma_gather4_load`` (MEGA_GENPHASE_NATIVE_GATHER4=1) issues the Rubin load unchanged; it assembled for sm_100a
and ran correctly on the GB200 nodes it was tried on.
"""

import os

import cutlass
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op

# L2 cache policy of the gather loads (MEGA_GENPHASE_GPOL: 0 none, 1 evict-first, 2 evict-last).
_GATHER_CACHE_POLICY = (0, 0x12F0000000000000, 0x14F0000000000000)[int(os.environ.get("MEGA_GENPHASE_GPOL", "0"))]


_CTA_GROUP_2_MBAR_MASK = 0xFEFFFFFF
GATHER_ROW_BYTES = 128


@dsl_user_op
def sm100_tma_gather4_load(
    tma_atom,
    smem_dst_ptr,
    mbar_ptr,
    col,
    r0,
    r1,
    r2,
    r3,
    *,
    use_cta_group_2: bool = False,
    mcast_mask=None,
    loc=None,
    ip=None,
):
    """Issue four single-row tile loads that together equal one Rubin TILE_GATHER4 load.

    Selects one of four PTX forms (the spellings CUTLASS uses for sm_100):
      {1CTA, 2CTA} x {no-mcast, multicast::cluster}.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        AddressSpace.generic,
        64,
    )
    desc_cute_ptr = _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
    desc_i64 = desc_cute_ptr.toint().ir_value(loc=loc, ip=ip)

    smem_dst_base = cutlass.Int32(smem_dst_ptr.toint())
    mbar_int = cutlass.Int32(mbar_ptr.toint())
    if use_cta_group_2:
        mbar_int = mbar_int & cutlass.Int32(_CTA_GROUP_2_MBAR_MASK)
    mbar_i32 = mbar_int.ir_value(loc=loc, ip=ip)
    col_i32 = cutlass.Int32(col).ir_value(loc=loc, ip=ip)
    cache_hint_i64 = cutlass.Int64(_GATHER_CACHE_POLICY).ir_value(loc=loc, ip=ip)

    use_mcast = mcast_mask is not None
    if use_mcast:
        mcast_mask_i16 = cutlass.Int16(mcast_mask).ir_value(loc=loc, ip=ip)

    cta_group_qualifier = ".cta_group::2" if use_cta_group_2 else ""
    if use_mcast:
        asm = (
            f"cp.async.bulk.tensor.2d{cta_group_qualifier}.shared::cluster.global"
            ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [$0], [$1, {$3, $4}], [$2], $5, $6;"
        )
        constraints = "r, l, r, r, r, h, l"
    else:
        asm = (
            f"cp.async.bulk.tensor.2d{cta_group_qualifier}.shared::cluster.global"
            ".mbarrier::complete_tx::bytes.L2::cache_hint"
            " [$0], [$1, {$3, $4}], [$2], $5;"
        )
        constraints = "r, l, r, r, r, l"

    for row_in_group, row in enumerate((r0, r1, r2, r3)):
        smem_dst_i32 = (smem_dst_base + cutlass.Int32(row_in_group * GATHER_ROW_BYTES)).ir_value(loc=loc, ip=ip)
        row_i32 = cutlass.Int32(row).ir_value(loc=loc, ip=ip)
        operands = [smem_dst_i32, desc_i64, mbar_i32, col_i32, row_i32]
        if use_mcast:
            operands.extend((mcast_mask_i16, cache_hint_i64))
        else:
            operands.append(cache_hint_i64)
        llvm.inline_asm(
            None,
            operands,
            asm,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@dsl_user_op
def native_tma_gather4_load(
    tma_atom,
    smem_dst_ptr,
    mbar_ptr,
    col,
    r0,
    r1,
    r2,
    r3,
    *,
    use_cta_group_2: bool = False,
    mcast_mask=None,
    loc=None,
    ip=None,
):
    """Issue one TMA TILE_GATHER4 load (the Rubin primitive, unchanged).

    Selects one of four PTX forms:
      {1CTA, 2CTA} x {no-mcast, multicast::cluster}.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(),
        AddressSpace.generic,
        64,
    )
    desc_cute_ptr = _cute_nvgpu_ir.get_tma_desc_addr(desc_ptr_ty, exec_atom, loc=loc, ip=ip)
    desc_i64 = desc_cute_ptr.toint().ir_value(loc=loc, ip=ip)

    smem_dst_i32 = cutlass.Int32(smem_dst_ptr.toint()).ir_value(loc=loc, ip=ip)
    mbar_int = cutlass.Int32(mbar_ptr.toint())
    if use_cta_group_2:
        mbar_int = mbar_int & cutlass.Int32(_CTA_GROUP_2_MBAR_MASK)
    mbar_i32 = mbar_int.ir_value(loc=loc, ip=ip)
    col_i32 = cutlass.Int32(col).ir_value(loc=loc, ip=ip)
    r0_i32 = cutlass.Int32(r0).ir_value(loc=loc, ip=ip)
    r1_i32 = cutlass.Int32(r1).ir_value(loc=loc, ip=ip)
    r2_i32 = cutlass.Int32(r2).ir_value(loc=loc, ip=ip)
    r3_i32 = cutlass.Int32(r3).ir_value(loc=loc, ip=ip)
    cache_hint_i64 = cutlass.Int64(_GATHER_CACHE_POLICY).ir_value(loc=loc, ip=ip)

    use_mcast = mcast_mask is not None
    if use_mcast:
        mcast_mask_i16 = cutlass.Int16(mcast_mask).ir_value(loc=loc, ip=ip)

    scope = "cluster" if use_cta_group_2 or use_mcast else "cta"
    multicast_qualifier = ".multicast::cluster" if use_mcast else ""
    cta_group_qualifier = ".cta_group::2" if use_cta_group_2 else ""
    asm = (
        f"cp.async.bulk.tensor.2d.shared::{scope}.global"
        f".tile::gather4.mbarrier::complete_tx::bytes{multicast_qualifier}"
        f".L2::cache_hint{cta_group_qualifier}"
        " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8"
    )
    operands = [
        smem_dst_i32,
        desc_i64,
        mbar_i32,
        col_i32,
        r0_i32,
        r1_i32,
        r2_i32,
        r3_i32,
    ]
    constraints = "r, l, r, r, r, r, r, r"
    if use_mcast:
        asm += ", $9;"
        operands.extend((mcast_mask_i16, cache_hint_i64))
        constraints += ", h, l"
    else:
        asm += ";"
        operands.append(cache_hint_i64)
        constraints += ", l"
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = ["native_tma_gather4_load", "sm100_tma_gather4_load"]
