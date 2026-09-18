# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw SM100 CTA-group-2 Gather4 TMA issue for the token-sparse MLA loaders.

The sparse K/V loaders feed the Gather4 instruction four page indices read
from the page-index SMEM ring.  ``cute.copy``'s Gather4 atom instead expects
a GMEM coordinate tensor, which would reintroduce the GMEM dependency the
ring removes, so the descriptor address, the leader-CTA mbarrier mapping and
the instruction itself are issued directly here.  The shared Gather4 atom
constructor stays in ``flashinfer.cute_dsl.attention.dsa.gather4_utils``.
"""

from collections.abc import Sequence

import cutlass.cute as cute
from cutlass import Int16, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def gather4_tma_descriptor_address(
    tma_atom: cute.CopyAtom, *, loc=None, ip=None
) -> cute.Pointer:
    """Return the tensor-map pointer held by a Gather4 copy atom.

    The raw Gather4 instruction takes the descriptor address directly; its four
    page indices come from the W9 page-index SMEM ring rather than from the
    GMEM coordinate tensor that ``cute.copy`` expects.
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
    """Map a local mbarrier address to CTA rank zero of the cluster.

    Every CTA-group-2 Gather4 signals the pair leader's barrier. A bit-mask
    formulation is not usable in CuTe DSL because its address-alignment
    analysis rounds the mask to a 1 KiB boundary, so ``mapa`` is emitted
    directly.
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

    Emits ``cp.async.bulk.tensor.2d...tile::gather4...cta_group::2`` with the
    coordinate tuple ``{head_dim, page0, page1, page2, page3}``. The page
    indices come from the W9 SMEM ring; ``cute.copy``'s Gather4 atom needs a
    GMEM index tensor, which would add the GMEM dependency this path removes.
    """
    if len(page_indices) != 4:
        raise ValueError("SM100 Gather4 requires exactly four page-index coordinates")

    # The destination is the local SMEM address; the one-bit multicast mask
    # below selects the recipient CTA. Only the completion mbarrier is mapped
    # to the lead CTA.
    dst_addr = dst_smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    # Accept either a CuTe descriptor pointer or the device address of a
    # manually encoded ``CUtensorMap``; the public Gather4 atom constructor
    # cannot form the [H128, page-slot=1] box.
    if hasattr(tma_descriptor, "toint"):
        descriptor_addr = tma_descriptor.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    else:
        descriptor_addr = Int64(tma_descriptor).ir_value(loc=loc, ip=ip)
    # CTA-group-2 completion signals the pair leader's full barrier, which is
    # also where the consumer waits.
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
        # ``~{memory}`` clobber: the instruction asynchronously writes SMEM;
        # ``has_side_effects`` alone prevents deletion but does not order it
        # against surrounding memory operations.
        "r,l,r,r,r,r,r,r,h,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
