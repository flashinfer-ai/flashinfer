# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for CuTe DSL 2D Gather4 TMA loads.

CuTe DSL 4.6 exposes the Gather4 dialect operation and lowering, but its
``cpasync`` module does not expose a corresponding high-level copy operation.
Keep the low-level dependency isolated here until that wrapper is public.
"""

from collections.abc import Sequence

import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass import Int16, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import tcgen05


_REQUIRED_GATHER4_IR_SYMBOLS = (
    "atom_make_non_exec_2d_gather4_tma_load",
    "GatherScatterTmaLoadEnum",
    "CopyAtomNonExec2DGather4TmaLoadType",
)


def _ensure_gather4_ir_support() -> None:
    """Require the low-level Gather4 API shipped by CuTe DSL 4.6 or newer."""
    missing = [
        name
        for name in _REQUIRED_GATHER4_IR_SYMBOLS
        if not hasattr(_cute_nvgpu_ir, name)
    ]
    if missing:
        raise RuntimeError(
            "HCA Gather4 requires nvidia-cutlass-dsl>=4.6.0; missing "
            + ", ".join(missing)
        )


class _CopyBulkTensor2DGather4G2SOp(cpasync.CopyBulkTensorTileG2SOp):
    """Local copy-op adapter for the public low-level Gather4 IR API."""

    def _get_description(self) -> str:
        return "cp.async GMEM -> SMEM bulk tensor Gather4 copy operation"

    def _to_ir(self):
        if self.cta_group == tcgen05.CtaGroup.ONE:
            return _cute_nvgpu_ir.GatherScatterTmaLoadEnum.sm_100
        if self.cta_group == tcgen05.CtaGroup.TWO:
            return _cute_nvgpu_ir.GatherScatterTmaLoadEnum.sm_100_2sm
        raise ValueError(f"unsupported CTA group for Gather4: {self.cta_group}")


@cute.jit
def make_gather4_2sm_tma_atom(
    gmem: cute.Tensor,
    smem_layout: cute.Layout,
    mma_tiler,
    tiled_mma: cute.TiledMma,
    gmem_coord_tensor: cute.Tensor,
):
    """Build a two-CTA Gather4 TMA atom for token-level absolute indices."""
    _ensure_gather4_ir_support()
    gather4_op = _CopyBulkTensor2DGather4G2SOp(cta_group=tcgen05.CtaGroup.TWO)
    ident = cute.make_identity_layout(gmem.shape)
    g_tile = cute.composition(ident, mma_tiler)
    cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
    cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
    cta_v_map = cute.select(cta_v_map, mode=[0, 2])
    cta_v_map = cute.zipped_divide(cta_v_map, (cta_mn, mma_tiler[1]))
    cta_v_map = cute.select(cta_v_map, mode=[0])

    smem_layout_ir = smem_layout.value if hasattr(smem_layout, "value") else smem_layout
    res = _cute_nvgpu_ir.atom_make_non_exec_2d_gather4_tma_load(
        gmem.value,
        gmem_coord_tensor.layout,
        smem_layout_ir,
        cta_v_map,
        gather4_op._to_ir(),
        num_multicast=1,
    )
    return cute.CopyAtom(
        gather4_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
    ), res[1]


@dsl_user_op
def gather4_tma_descriptor_address(
    tma_atom: cute.CopyAtom, *, loc=None, ip=None
) -> cute.Pointer:
    """Return the tensor-map pointer held by a Gather4 copy atom.

    The public ``cute.copy`` Gather4 interface receives its four indices from
    a GMEM coordinate tensor.  DSV4 CSA instead reads those indices from the
    W9 six-stage page-index SMEM ring, so the raw TMA instruction needs the
    descriptor address embedded in the atom.
    """
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    descriptor_ptr_type = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
    )
    return _cute_nvgpu_ir.get_tma_desc_addr(
        descriptor_ptr_type, exec_atom, loc=loc, ip=ip
    )


@dsl_user_op
def gather4_cta2_leader_mbarrier_address(
    completion_mbarrier: cute.Pointer, *, loc=None, ip=None
) -> Int32:
    """Map a local mbarrier to CTA rank zero in its threadblock cluster.

    The generated DSV4 source uses ``cuda_ptx::mapa(..., getLeadCtaRank())``
    before every CTA-group-2 Gather4.  A bit-mask formulation is not
    equivalent in CuTe DSL: its address-alignment analysis rounds the mask to
    a 1 KiB boundary.  Emit the source instruction directly instead.
    """
    local_addr = completion_mbarrier.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    leader_rank = Int32(0).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [local_addr, leader_rank],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def gather4_cta2_cluster_smem_address(
    local_smem: cute.Pointer, *, loc=None, ip=None
) -> Int32:
    """Encode a local SMEM pointer for a ``shared::cluster`` destination.

    A ``cp.async...shared::cluster`` operand is not a plain local-SMEM
    offset.  The generated source obtains this conversion through its typed
    cluster pointer before issuing TMA.  ``mapa`` with the issuing CTA's rank
    is the equivalent explicit form and, unlike passing the local offset
    through raw asm, preserves the CTA-rank bits for the rank-1 issuer.
    """
    local_addr = local_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [local_addr, Int32(cta_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def issue_gather4_tma_2cta(
    tma_descriptor,
    dst_smem: cute.Pointer,
    completion_mbarrier: cute.Pointer,
    head_dim_offset: Int32,
    page_indices: Sequence[Int32],
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one SM100 CTA2 Gather4 load using four SMEM-sourced indices.

    This is deliberately lower-level than ``cute.copy``.  The generated
    TRTLLM DSV4 kernel issues
    ``cp.async.bulk.tensor.2d...tile::gather4...cta_group::2`` with the
    coordinate tuple ``{head_dim, page0, page1, page2, page3}``, where the
    four page values were prefetched by W9 into shared memory.  CuTe's
    high-level Gather4 atom currently accepts a GMEM index tensor instead;
    using it would reintroduce the extra GMEM dependency that this adapter is
    intended to remove.
    """
    if len(page_indices) != 4:
        raise ValueError("SM100 Gather4 requires exactly four page-index coordinates")

    # The source passes its local SMEM destination directly to the
    # ``shared::cluster`` TMA instruction.  Cluster recipient selection comes
    # from the one-bit multicast mask below; only the completion mbarrier is
    # explicitly mapped to the lead CTA.
    dst_addr = dst_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    # Ordinary CuTe atoms expose a pointer here.  The DSV4 source map uses a
    # [H128, page-slot=1] box which the public Gather4 atom constructor cannot
    # form, so the port may instead supply a device address of a manually
    # encoded ``CUtensorMap`` descriptor.
    if hasattr(tma_descriptor, "toint"):
        descriptor_addr = tma_descriptor.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    else:
        descriptor_addr = Int64(tma_descriptor).ir_value(loc=loc, ip=ip)
    # Match the generated source: CTA-group-2 completion is routed to the
    # pair leader's full barrier, while the pipeline maps consumer wait to the
    # same CTA-group leader.
    barrier_addr = gather4_cta2_leader_mbarrier_address(
        completion_mbarrier, loc=loc, ip=ip
    ).ir_value(loc=loc, ip=ip)
    head_dim = Int32(head_dim_offset).ir_value(loc=loc, ip=ip)
    coords = [Int32(coord).ir_value(loc=loc, ip=ip) for coord in page_indices]
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cta_mask = (Int16(1) << Int16(cta_rank)).ir_value(loc=loc, ip=ip)

    llvm.inline_asm(
        None,
        [
            dst_addr,
            descriptor_addr,
            barrier_addr,
            head_dim,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            cta_mask,
        ],
        (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
            "mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 "
            "[$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;"
        ),
        # Match the generated CUDA wrapper's ``: \"memory\"`` clobber.  The
        # raw instruction asynchronously writes SMEM; ``has_side_effects``
        # alone prevents deletion but does not model that memory dependency
        # to LLVM's scheduler.
        "r,l,r,r,r,r,r,r,h,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
