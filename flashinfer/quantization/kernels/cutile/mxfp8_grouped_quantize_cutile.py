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

cuTile grouped MXFP8 quantization.

This module imports ``cuda.tile`` directly, mirroring the cuTile GEMM backends
in ``flashinfer/gemm/kernels/cutile``. Callers must gate on
``flashinfer.cutile.is_cuda_tile_available()`` before importing it so the
``cuda.tile`` dependency is only required when this backend is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct
import torch


BLOCK_M = 128
BLOCK_K = 128
MX_GROUP_K = 32
SCALE_COLS_PER_K_TILE = BLOCK_K // MX_GROUP_K
SCALE_BYTES_PER_TILE = BLOCK_M * SCALE_COLS_PER_K_TILE
MAX_GROUPS_FUSED = 2048


@dataclass(frozen=True)
class _PrefixSchedule:
    tile_offsets: torch.Tensor
    total_tiles: int


def mxfp8_grouped_quantize_cutile(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    """Launch grouped MXFP8 quantization with a persistent-prefix schedule.

    Args:
        input: Row-major flattened input tensor of shape ``[sum_m, padded_K]``.
        problem_sizes: Int32 tensor of shape ``[groups, >=3]``. Column 0 is
            the valid row count for each group.
        expert_offsets: Int32 row offsets into ``input`` and ``quant_output``.
        blockscale_offsets: Int32 row offsets into flattened scale storage.
        quant_output: Row-major output tensor with the same shape as ``input``.
        scale_factor: Contiguous uint8 scale tensor addressed as flat storage.
    """

    _check_common_args(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )

    prefix_schedule = build_mxfp8_grouped_quant_prefix_schedule(input, problem_sizes)
    mxfp8_grouped_quantize_cutile_with_prefix_schedule(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
        prefix_schedule,
    )


def build_mxfp8_grouped_quant_prefix_schedule(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
) -> _PrefixSchedule:
    """Build the device-side prefix table consumed by the persistent kernel."""

    num_groups = problem_sizes.size(0)
    if num_groups > MAX_GROUPS_FUSED:
        raise ValueError(
            f"num_groups={num_groups} exceeds MAX_GROUPS_FUSED={MAX_GROUPS_FUSED}."
        )
    if num_groups == 0:
        return _PrefixSchedule(
            torch.zeros(1, dtype=torch.int32, device=input.device),
            total_tiles=-1,
        )

    # Allocate the prefix scratch per call. The build kernel fully overwrites it
    # before the persistent kernel reads it on the same stream, so it needs no
    # zero-init and its lifetime is one call; it is only MAX_GROUPS_FUSED + 1
    # int32 entries (8 KB when MAX_GROUPS_FUSED is 2048), so caching would save
    # a negligible allocation.
    tile_offsets = torch.empty(
        MAX_GROUPS_FUSED + 1, dtype=torch.int32, device=input.device
    )

    _launch_prefix_offsets_into(input, problem_sizes, tile_offsets)
    return _PrefixSchedule(tile_offsets, total_tiles=-1)


def _launch_prefix_offsets_into(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    tile_offsets: torch.Tensor,
) -> None:
    if input.dim() != 2:
        raise ValueError("input must be a 2D tensor")
    if input.size(1) % BLOCK_K != 0:
        raise ValueError("input K dimension must align to 128")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if problem_sizes.dim() != 2 or problem_sizes.size(1) < 3:
        raise ValueError("problem_sizes must be a [groups, >=3] tensor")
    if problem_sizes.dtype != torch.int32:
        raise ValueError("problem_sizes dtype must be torch.int32")
    if not problem_sizes.is_cuda:
        raise ValueError("problem_sizes must be a CUDA tensor")
    if problem_sizes.device != input.device:
        raise ValueError("problem_sizes must live on the input device")

    if tile_offsets.dim() != 1:
        raise ValueError("tile_offsets must be a 1D tensor")
    if tile_offsets.dtype != torch.int32:
        raise ValueError("tile_offsets dtype must be torch.int32")
    if not tile_offsets.is_cuda:
        raise ValueError("tile_offsets must be a CUDA tensor")
    if tile_offsets.device != input.device:
        raise ValueError("tile_offsets must live on the input device")
    if tile_offsets.numel() < MAX_GROUPS_FUSED + 1:
        raise ValueError(
            f"tile_offsets must have at least {MAX_GROUPS_FUSED + 1} elements, "
            f"got {tile_offsets.numel()}."
        )
    if problem_sizes.size(0) > MAX_GROUPS_FUSED:
        raise ValueError(
            f"num_groups={problem_sizes.size(0)} exceeds "
            f"MAX_GROUPS_FUSED={MAX_GROUPS_FUSED}."
        )

    k_tiles = input.size(1) // BLOCK_K
    ct.launch(
        torch.cuda.current_stream(input.device),
        (1,),
        _mxfp8_build_prefix_offsets_kernel,
        (problem_sizes, tile_offsets, k_tiles),
    )


def mxfp8_grouped_quantize_cutile_with_prefix_schedule(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
    prefix_schedule: _PrefixSchedule,
    *,
    blocks_per_sm: int = 4,
) -> None:
    """Launch the persistent grouped MXFP8 kernel with a prebuilt schedule."""

    if blocks_per_sm <= 0:
        raise ValueError("blocks_per_sm must be positive")

    _check_common_args(
        input,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )
    if prefix_schedule.total_tiles == 0:
        return

    props = torch.cuda.get_device_properties(input.device)
    persistent_grid = props.multi_processor_count * blocks_per_sm
    if prefix_schedule.total_tiles > 0:
        num_blocks = max(1, min(prefix_schedule.total_tiles, persistent_grid))
    else:
        num_blocks = max(1, persistent_grid)

    ct.launch(
        torch.cuda.current_stream(input.device),
        (num_blocks,),
        _mxfp8_grouped_quant_cutile_persistent_prefix,
        (
            input,
            problem_sizes,
            expert_offsets,
            blockscale_offsets,
            prefix_schedule.tile_offsets,
            quant_output,
            scale_factor.reshape(-1),
        ),
    )


def _check_common_args(
    input: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    if input.dim() != 2:
        raise ValueError("input must be a 2D tensor")
    if input.size(1) % BLOCK_K != 0:
        raise ValueError("input K dimension must align to 128")
    if input.stride(1) != 1:
        raise ValueError("input must be row-major in the K dimension")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("input dtype must be torch.float16 or torch.bfloat16")

    if quant_output.shape != input.shape:
        raise ValueError("quant_output must have the same shape as input")
    if quant_output.dtype != torch.float8_e4m3fn:
        raise ValueError("quant_output dtype must be torch.float8_e4m3fn")
    if quant_output.stride(1) != 1:
        raise ValueError("quant_output must be row-major in the K dimension")
    if not quant_output.is_cuda:
        raise ValueError("quant_output must be a CUDA tensor")

    if problem_sizes.dim() != 2 or problem_sizes.size(1) < 3:
        raise ValueError("problem_sizes must be a [groups, >=3] tensor")
    if problem_sizes.dtype != torch.int32:
        raise ValueError("problem_sizes dtype must be torch.int32")
    if not problem_sizes.is_cuda:
        raise ValueError("problem_sizes must be a CUDA tensor")

    groups = problem_sizes.size(0)
    if expert_offsets.dim() != 1 or expert_offsets.size(0) != groups:
        raise ValueError("expert_offsets must be 1D with one entry per group")
    if blockscale_offsets.dim() != 1 or blockscale_offsets.size(0) != groups:
        raise ValueError("blockscale_offsets must be 1D with one entry per group")
    if expert_offsets.dtype != torch.int32 or blockscale_offsets.dtype != torch.int32:
        raise ValueError("expert_offsets and blockscale_offsets must be torch.int32")
    if not expert_offsets.is_cuda or not blockscale_offsets.is_cuda:
        raise ValueError("expert_offsets and blockscale_offsets must be CUDA tensors")

    if scale_factor.dtype != torch.uint8:
        raise ValueError("scale_factor dtype must be torch.uint8")
    if not scale_factor.is_cuda:
        raise ValueError("scale_factor must be a CUDA tensor")
    if not scale_factor.is_contiguous():
        raise ValueError(
            "scale_factor must be contiguous because cuTile writes it as flat "
            "block-scaled storage"
        )

    expected_device = input.device
    if (
        quant_output.device != expected_device
        or problem_sizes.device != expected_device
        or expert_offsets.device != expected_device
        or blockscale_offsets.device != expected_device
        or scale_factor.device != expected_device
    ):
        raise ValueError("all cuTile quant tensors must be on the same CUDA device")


@ct.function
def _float_to_e8m0_rp_u8(x):
    """Convert positive float32 scale values to E8M0 storage bytes."""

    min_e8m0 = 5.877471754111438e-39  # 2 ** -127
    safe_x = ct.maximum(x, min_e8m0)
    exponent = ct.ceil(ct.log2(safe_x)).astype(ct.int32) + 127
    exponent = ct.minimum(ct.maximum(exponent, 0), 254)
    return ct.where(x == 0.0, 0, exponent).astype(ct.uint8)


@ct.function
def _quantize_one_128x128_tile(
    input_view, output_view, scale_flat, mt, kt, blockscale_offset
):
    """Quantize one 128x128 tile and scatter its 128x4 scale bytes."""

    tile = ct.load(
        input_view,
        (mt, kt),
        shape=(BLOCK_M, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
        latency=8,
    )
    tile_f32 = tile.astype(ct.float32)
    tile_grouped = ct.reshape(tile_f32, (BLOCK_M, SCALE_COLS_PER_K_TILE, MX_GROUP_K))

    amax = ct.max(ct.abs(tile_grouped), axis=2)
    scale_float = amax / 448.0
    scale_byte = _float_to_e8m0_rp_u8(scale_float)
    reciprocal_scale = ct.where(
        amax != 0.0,
        ct.exp2(127.0 - scale_byte.astype(ct.float32)),
        0.0,
    )

    quant_grouped = tile_grouped * ct.expand_dims(reciprocal_scale, 2)
    quant_tile = ct.reshape(quant_grouped, (BLOCK_M, BLOCK_K)).astype(ct.float8_e4m3fn)
    ct.store(output_view, (mt, kt), quant_tile, latency=8)

    k_groups = input_view.shape[1] // MX_GROUP_K
    k_scale_tiles = k_groups // SCALE_COLS_PER_K_TILE
    base = (
        blockscale_offset * k_groups + (mt * k_scale_tiles + kt) * SCALE_BYTES_PER_TILE
    )

    row = ct.reshape(ct.arange(BLOCK_M, dtype=ct.int32), (BLOCK_M, 1))
    scale_col = ct.reshape(
        ct.arange(SCALE_COLS_PER_K_TILE, dtype=ct.int32),
        (1, SCALE_COLS_PER_K_TILE),
    )
    swizzled_offset = (row % 32) * 16 + (row // 32) * SCALE_COLS_PER_K_TILE + scale_col
    ct.scatter(
        scale_flat,
        base + swizzled_offset,
        scale_byte,
        check_bounds=False,
        latency=2,
    )


@ct.kernel(opt_level=3, occupancy=4)
def _mxfp8_grouped_quant_cutile_persistent_prefix(
    input: ct.IndexedWithInt64,
    problem_sizes,
    expert_offsets,
    blockscale_offsets,
    tile_offsets,
    quant_output: ct.IndexedWithInt64,
    scale_flat,
):
    groups = problem_sizes.shape[0]
    total_tiles = ct.load(tile_offsets, (groups,), shape=()).item()
    stride = ct.num_blocks(0)
    k_tiles = input.shape[1] // BLOCK_K

    tile_id = ct.bid(0)
    while tile_id < total_tiles:
        lo = 0
        hi = groups
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            mid_offset = ct.load(tile_offsets, (mid,), shape=()).item()
            if tile_id < mid_offset:
                hi = mid
            else:
                lo = mid

        g = lo
        group_tile0 = ct.load(tile_offsets, (g,), shape=()).item()
        local_tile = tile_id - group_tile0
        mt = local_tile // k_tiles
        kt = local_tile - mt * k_tiles

        m = ct.load(problem_sizes, (g, 0), shape=()).item()
        row0 = ct.load(expert_offsets, (g,), shape=()).item()
        blockscale_offset = ct.load(blockscale_offsets, (g,), shape=()).item()

        input_view = input.slice(axis=0, start=row0, stop=row0 + m)
        output_view = quant_output.slice(axis=0, start=row0, stop=row0 + m)
        _quantize_one_128x128_tile(
            input_view,
            output_view,
            scale_flat,
            mt,
            kt,
            blockscale_offset,
        )

        tile_id = tile_id + stride


@ct.kernel(opt_level=3)
def _mxfp8_build_prefix_offsets_kernel(
    problem_sizes,
    tile_offsets,
    k_tiles,
):
    m_tile = ct.load(
        problem_sizes,
        (0, 0),
        shape=(MAX_GROUPS_FUSED, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    m = ct.reshape(m_tile, (MAX_GROUPS_FUSED,))

    m_tiles = (m + BLOCK_M - 1) // BLOCK_M
    tile_counts = m_tiles * k_tiles

    cumsum_out = ct.cumsum(tile_counts, axis=0)

    leading_zero = ct.full((1,), 0, dtype=ct.int32)
    ct.store(tile_offsets, (0,), leading_zero)

    scatter_indices = ct.arange(MAX_GROUPS_FUSED, dtype=ct.int32) + 1
    ct.scatter(
        tile_offsets,
        scatter_indices,
        cumsum_out,
        check_bounds=False,
    )
