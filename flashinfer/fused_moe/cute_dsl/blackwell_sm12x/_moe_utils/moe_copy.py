# Copyright (c) 2025 by FlashInfer team.
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
"""Shared token-map activation copies and TMA gather4 descriptors."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from .....utils import ceil_div


def can_use_tma_gather4(tile, token_map, sm_version):
    return (
        tuple(tile) == (64, 128, 128)
        and sm_version == "sm_120"
        and token_map is not None
        and token_map.numel() >= 4
        and hasattr(cpasync, "CopyBulkTensor2DGather4G2SOp")
    )


def make_gather4_index_tensor(gMap, K):
    return cute.make_tensor(
        gMap.iterator, cute.make_layout((cute.size(gMap, mode=[0]), K), stride=(1, 0))
    )


def make_gather4_tma_atom(gA, gMap, smem_layout, bk):
    box = (4, bk)
    stage_layout = cute.slice_(smem_layout, (None, None, 0))
    prefix = cute.slice_(cute.zipped_divide(stage_layout, box), (None, (0, 0)))
    prefix = cute.make_composed_layout(
        prefix.inner, prefix.offset, cute.flatten(prefix.outer)
    )
    indices = make_gather4_index_tensor(gMap, cute.size(gA, mode=[1]))
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensor2DGather4G2SOp(),
        gA,
        prefix,
        box,
        gmem_coord_tensor=indices,
        num_multicast=1,
    )


@cute.jit
def partition_gather4(atom, tensor_a, gMap, sA, begin, bk):
    box = (4, bk)
    gI = make_gather4_index_tensor(gMap, cute.size(tensor_a, mode=[1]))
    gA = cute.zipped_divide(cute.domain_offset((begin, 0), tensor_a), box)
    gI = cute.zipped_divide(cute.domain_offset((begin, 0), gI), box)
    return cpasync.tma_partition(
        atom,
        cutlass.Int32(0),
        cute.make_layout(1),
        cute.zipped_divide(sA, box),
        [gA, gI],
    )


@cute.jit
def gmem_to_smem_gather4(atom, tAgA, tAgI, tAsA, kt, stage, full, bk, rows, tokens):
    groups = ceil_div(rows, 4)
    full_groups = rows >> 2
    with cute.arch.elect_one():
        cute.arch.mbarrier_expect_tx(full, groups * 4 * bk)
    for group in cutlass.range(full_groups, unroll=1):
        cute.copy(
            atom,
            [tAgA[None, (group, kt)], tAgI[None, (group, kt)]],
            tAsA[None, (group, 0, stage)],
            tma_bar_ptr=full,
        )
    if (rows & 3) != 0:
        index = tAgI[None, (full_groups, kt)]
        assert cute.cosize(index.layout) == 4
        rI = cute.make_rmem_tensor(index.layout, cutlass.Int32)
        gIndex = cute.make_tensor(index.iterator, cute.make_layout(4))
        rIndex = cute.make_tensor(rI.iterator, cute.make_layout(4))
        rIndex.fill(cutlass.Int32(tokens))
        for row in cutlass.range_constexpr(4):
            if row < (rows & 3):
                rIndex[row] = gIndex[row]
        cute.copy(
            atom,
            [tAgA[None, (full_groups, kt)], rI],
            tAsA[None, (full_groups, 0, stage)],
            tma_bar_ptr=full,
        )


@cute.jit
def gmem_to_smem_tokens(atom, gA, gMap, target, begin, rows, kt):
    lane = cute.arch.lane_idx()
    K = cute.size(gA, mode=[1])
    k_tiles = ceil_div(K, 128)
    source_kt = cutlass.Int32(0)
    if kt < k_tiles:
        source_kt = kt
    source_layout = cute.make_layout(((16, 1),), stride=((1, 0),))
    pred = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Boolean)
    for part in cutlass.range_constexpr(cute.size(target, mode=[1])):
        row = (lane >> 3) + part * 4
        token = cutlass.Int32(0)
        if row < rows:
            token = gMap[begin + row]
        address = (
            gA.iterator + cutlass.Int64(token) * K + source_kt * 128 + (lane & 7) * 16
        )
        pointer = cute.make_ptr(
            cutlass.Float8E4M3FN,
            address.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        source = cute.make_tensor(pointer, source_layout)
        pred[0] = (row < rows) & (kt < k_tiles)
        cute.copy(atom, source, target[None, part], pred=pred)
