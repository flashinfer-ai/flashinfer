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
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


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
def issue_gather4_tma_cta(
    tma_descriptor,
    dst_smem,
    completion_mbarrier,
    head_dim_offset: Int32,
    page_indices: Sequence,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one CTA-local Gather4 load (no cluster multicast).

    Used by the 1CTA token-sparse loader: the four row coordinates address a
    flat ``[row_bytes, N]`` pool map and completion lands on this CTA's own
    stage mbarrier.  ``dst_smem`` / ``completion_mbarrier`` accept a CuTe
    pointer or a ``cutlass.Array`` view of shared memory; ``tma_descriptor`` is
    a CuTe descriptor pointer or the GMEM address of an encoded ``CUtensorMap``.
    """
    if len(page_indices) != 4:
        raise ValueError("SM100 Gather4 requires exactly four row coordinates")

    def _smem_addr(obj):
        if hasattr(obj, "data_ptr") and not hasattr(obj, "toint"):
            obj = obj.data_ptr()
        return obj.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)

    dst_addr = _smem_addr(dst_smem)
    barrier_addr = _smem_addr(completion_mbarrier)
    if hasattr(tma_descriptor, "toint"):
        descriptor_addr = tma_descriptor.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    else:
        descriptor_addr = Int64(tma_descriptor).ir_value(loc=loc, ip=ip)
    head_dim = Int32(head_dim_offset).ir_value(loc=loc, ip=ip)
    coords = [Int32(coord).ir_value(loc=loc, ip=ip) for coord in page_indices]
    llvm.inline_asm(
        None,
        [dst_addr, descriptor_addr, barrier_addr, head_dim, *coords],
        (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
            "mbarrier::complete_tx::bytes "
            "[$0], [$1, {$3, $4, $5, $6, $7}], [$2];"
        ),
        "r,l,r,r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
