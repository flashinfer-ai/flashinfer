# Copyright (c) 2025 by the b12x authors.
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

"""Block-scale storage layouts."""

from typing import Tuple
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import torch
from b12x._lib.utils.dtypes import ceil_div


def convert_sf_to_mma_layout(
    sf: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Convert scale factors from swizzled 2D layout to 6D MMA-compatible layout.

    This function converts scale factors produced by `fp4_quantize(..., is_sf_swizzled_layout=True)`
    to the 6D layout expected by CuteDSL grouped GEMM kernels.

    The swizzled scale factors from `fp4_quantize` have shape `(M, K/sf_vec_size)` but are
    stored in a swizzled pattern internally. This function reshapes them to the explicit
    6D MMA-compatible layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)` with the
    physical storage order `(num_groups, m_tiles, k_tiles, 32, 4, 4)`.

    Layout mapping (from linear (m, k) position):
        - m_tile = m // 128
        - outer_m = m % 32
        - inner_m = (m % 128) // 32
        - k_tile = k // 4
        - inner_k = k % 4
        - 6D position: (outer_m, inner_m, m_tile, inner_k, k_tile, group)

    Args:
        sf: Scale factor tensor from `fp4_quantize(..., is_sf_swizzled_layout=True)`.
            Shape: `(M, K/sf_vec_size)` or `(num_groups * M, K/sf_vec_size)`.
        m: The M dimension (rows) of the original matrix before quantization.
        k: The K dimension (columns) of the original matrix before quantization.
        num_groups: Number of groups (e.g., experts). Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Scale factors in 6D MMA layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)`.
        This is a strided view (not contiguous) with physical storage order
        `(num_groups, m_tiles, k_tiles, 32, 4, 4)`.

    Example:
        >>> # Quantize weight tensor
        >>> w_q, w_sf = fp4_quantize(weight, global_scale=gs, is_sf_swizzled_layout=True)
        >>> # Convert scale factors to MMA layout
        >>> w_sf_mma = convert_sf_to_mma_layout(w_sf, m=weight.shape[0], k=weight.shape[1])

    Note:
        - The input `sf` must be produced with `is_sf_swizzled_layout=True`.
        - M and K dimensions must be multiples of 128 and 64 respectively for proper alignment.
        - For grouped tensors (e.g., expert weights), reshape to `(num_groups * M, K)`
          before quantization, then use this function with the appropriate `num_groups`.
        - The returned tensor is a strided view, NOT contiguous. This is intentional as
          the CuteDSL kernel expects the specific physical memory layout.
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)

    # Verify input shape
    expected_elements = num_groups * m_tiles * k_tiles * 32 * 4 * 4
    actual_elements = sf.numel()
    if actual_elements != expected_elements:
        raise ValueError(
            f"Scale factor tensor has {actual_elements} elements, "
            f"expected {expected_elements} for m={m}, k={k}, num_groups={num_groups}"
        )

    # Reshape from flat 2D to 6D physical storage order
    # Physical storage: (num_groups, m_tiles, k_tiles, 32, 4, 4)
    sf_6d = sf.view(num_groups, m_tiles, k_tiles, 32, 4, 4)

    # Permute to MMA logical order: (32, 4, m_tiles, 4, k_tiles, num_groups)
    # This creates a strided view (non-contiguous), which is what the kernel expects
    sf_6d = sf_6d.permute(3, 4, 1, 5, 2, 0)

    return sf_6d  # Return strided view, NOT contiguous


def convert_sf_from_mma_layout(
    sf_6d: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Convert scale factors from 6D MMA layout back to 2D swizzled layout.

    This is the inverse of `convert_sf_to_mma_layout`.

    Args:
        sf_6d: Scale factors in 6D MMA layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)`.
               Can be either a strided view or contiguous.
        m: The M dimension (rows) of the original matrix.
        k: The K dimension (columns) of the original matrix.
        num_groups: Number of groups. Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Scale factors in 2D swizzled layout: `(num_groups * M_padded, K_padded/sf_vec_size)`.
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)

    # Permute from MMA logical order back to storage order
    # From: (32, 4, m_tiles, 4, k_tiles, num_groups)
    # To: (num_groups, m_tiles, k_tiles, 32, 4, 4)
    sf_storage = sf_6d.permute(5, 2, 4, 0, 1, 3).contiguous()

    # Reshape to 2D
    padded_m = m_tiles * 128
    padded_sf_k = k_tiles * 4
    sf_2d = sf_storage.reshape(num_groups * padded_m, padded_sf_k)

    return sf_2d


def get_mma_sf_shape(
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> Tuple[int, int, int, int, int, int]:
    """Get the 6D MMA-compatible scale factor shape.

    Args:
        m: The M dimension (rows) of the matrix.
        k: The K dimension (columns) of the matrix.
        num_groups: Number of groups. Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Shape tuple: (32, 4, m_tiles, 4, k_tiles, num_groups)
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)
    return (32, 4, m_tiles, 4, k_tiles, num_groups)


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    # The global scale atom always has four K groups.  A BK64 stage consumes
    # two groups but keeps the same row pitch inside shared memory.
    blk_elems = blk_mn * 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[0] must be divisible by 16"
    )

    # Scale-factor tiles are quantized in 128-row blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfa_tile_m = max(blk_mn, ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFB based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    # Keep the four-group atom pitch when a BK64 stage exposes two groups.
    blk_elems = blk_mn * 4

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    assert tile_shape_mnk[1] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[1] must be divisible by 16"
    )

    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    # Scale-factor tiles are quantized in 128-column blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfb_tile_n = max(blk_mn, ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFA_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeN, sSFA_shapeK)
    sSFA_stride = (sSF_strideN, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged
