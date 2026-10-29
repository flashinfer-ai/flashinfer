# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blackwell TMA primitives shared by the MoE mainloops."""

import cutlass
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def tma_gather4(
    atom,
    destination,
    mbarrier,
    k,
    r0,
    r1,
    r2,
    r3,
    cta_group,
    cache_policy=0,
    mcast_mask=None,
    loc=None,
    ip=None,
):
    """Load four independently indexed rows into the native K-major layout."""
    executable = _cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    descriptor_type = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64
    )
    descriptor = _cute_nvgpu_ir.get_tma_desc_addr(
        descriptor_type, executable, loc=loc, ip=ip
    )
    barrier = cutlass.Int32(mbarrier.toint())
    if cta_group == 2:
        # Both CTAs' completion bytes belong to the leader's barrier.
        barrier = barrier & cutlass.Int32(-16777217)
    elif cta_group != 1:
        raise ValueError("Unsupported TMA CTA group")
    operands = [
        cutlass.Int32(destination.toint()).ir_value(loc=loc, ip=ip),
        descriptor.toint().ir_value(loc=loc, ip=ip),
        barrier.ir_value(loc=loc, ip=ip),
        *[
            cutlass.Int32(value).ir_value(loc=loc, ip=ip)
            for value in (k, r0, r1, r2, r3)
        ],
    ]
    constraints = "r,l,r,r,r,r,r,r"
    scope = "cluster" if cta_group == 2 or mcast_mask is not None else "cta"
    asm = (
        f"cp.async.bulk.tensor.2d.shared::{scope}.global.tile::gather4."
        "mbarrier::complete_tx::bytes"
    )
    arguments = "[$0], [$1, {$3, $4, $5, $6, $7}], [$2]"
    if mcast_mask is not None:
        asm += ".multicast::cluster"
        arguments += f", ${len(operands)}"
        operands.append(cutlass.Int16(mcast_mask).ir_value(loc=loc, ip=ip))
        constraints += ",h"
    asm += f".cta_group::{cta_group}.L2::cache_hint "
    arguments += f", ${len(operands)};"
    operands.append(cutlass.Int64(cache_policy).ir_value(loc=loc, ip=ip))
    constraints += ",l"
    asm += arguments
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
