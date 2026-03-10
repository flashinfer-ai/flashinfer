"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Type, Union

import cutlass
import cutlass.cute as cute


def make_smem_layout_a_kind(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[cutlass.Numeric],
    num_stages: int,
    kind: cute.nvgpu.tcgen05.SmemLayoutAtomKind,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps with:
    1. Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int
    :param kind:         The kind of layout Atom
    :type kind:          SmemLayoutAtomKind

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = (
        tiled_mma.op.a_major_mode == cutlass.cute.nvgpu.tcgen05.OperandMajorMode.K
    )
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
        kind,
        a_dtype,
        loc=loc,
        ip=ip,
    )
    a_smem_layout_staged = cute.nvgpu.tcgen05.tile_to_mma_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged


def make_smem_layout_b_kind(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    b_dtype: Type[cutlass.Numeric],
    num_stages: int,
    kind: cute.nvgpu.tcgen05.SmemLayoutAtomKind,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps:
    1. Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    :param tiled_mma: The tiled MMA which is used to partition the B tensor.
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape.
    :type mma_tiler_mnk: cute.cute.Tile
    :param b_dtype: The element type for the B tensor.
    :type b_dtype: Type[Numeric]
    :param num_stages: The stage of the B tensor.
    :type num_stages: int
    :param kind:         The kind of layout Atom
    :type kind:          SmemLayoutAtomKind

    :return: SMEM layout for the B tensor.
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = (
        tiled_mma.op.b_major_mode == cutlass.cute.nvgpu.tcgen05.OperandMajorMode.K
    )
    b_smem_shape = tiled_mma.partition_shape_B(
        cute.dice(mma_tiler_mnk, (None, 1, 1), loc=loc, ip=ip)
    )
    b_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
        kind,
        b_dtype,
        loc=loc,
        ip=ip,
    )
    b_smem_layout_staged = cute.nvgpu.tcgen05.tile_to_mma_shape(
        b_smem_layout_atom,
        cute.append(b_smem_shape, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return b_smem_layout_staged


def make_smem_layout_epi_kind(
    epi_dtype: Type[cutlass.Numeric],
    epi_layout: cutlass.utils.LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    kind: cute.nvgpu.tcgen05.SmemLayoutAtomKind,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps:
    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensor's majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    :param epi_dtype: The element type for the epilog tensor.
    :type epi_dtype: Type[Numeric]
    :param epi_layout: The layout enum for the epilog tensor.
    :type epi_layout: LayoutEnum
    :param epi_tile: The epilogue tile shape.
    :type epi_tile: cute.cute.Tile
    :param epi_stage: The stage of the epilog tensor.
    :type epi_stage: int

    :return: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    epilog_shape = cute.product_each(
        cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip
    )

    c_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
        kind,
        epi_dtype,
        loc=loc,
        ip=ip,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        cute.append(epilog_shape, epi_stage, loc=loc, ip=ip),
        order=((1, 0, 2) if not epi_layout.is_n_major_c() else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return epi_smem_layout_staged
