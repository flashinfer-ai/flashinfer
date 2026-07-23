# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""PDL-aware BF16 activation permutation for SM100 W4A16 MoE."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from flashinfer.utils import get_device_sm_count


_COPY_TILE_SIZE = 8192
_NUM_WARPS = 32
_ROWS_PER_PROGRAM = 8


@triton.jit
def _moe_permute_bf16_pdl_kernel(
    input,
    permuted_output,
    tile_idx_to_mn_limit,
    permuted_idx_to_expanded_idx,
    num_non_exiting_tiles,
    HIDDEN_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_groups = tl.cdiv(TILE_SIZE, ROWS_PER_PROGRAM)
    num_work_items = tl.load(num_non_exiting_tiles) * num_row_groups
    offsets_k = tl.arange(0, BLOCK_K)

    for work_idx in tl.range(pid, num_work_items, num_programs):
        tile_idx = work_idx // num_row_groups
        row_group = work_idx % num_row_groups
        tile_start = tile_idx * TILE_SIZE
        mn_limit = tl.load(tile_idx_to_mn_limit + tile_idx).to(tl.uint32)
        group_start = tile_start + row_group * ROWS_PER_PROGRAM

        for row_offset in range(ROWS_PER_PROGRAM):
            permuted_idx = (group_start + row_offset).to(tl.uint32)
            row_is_active = permuted_idx < mn_limit
            if TILE_SIZE % ROWS_PER_PROGRAM:
                row_is_active &= permuted_idx < tile_start + TILE_SIZE
            expanded_idx = tl.load(
                permuted_idx_to_expanded_idx + permuted_idx,
                mask=row_is_active,
                other=0,
            )
            token_idx = expanded_idx.to(tl.uint32) // TOP_K
            input_row = token_idx * HIDDEN_SIZE
            output_row = permuted_idx * HIDDEN_SIZE

            for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
                offsets = k_start + offsets_k
                column_mask = row_is_active & (offsets < HIDDEN_SIZE)
                values = tl.load(input + input_row + offsets, mask=column_mask)
                tl.store(
                    permuted_output + output_row + offsets,
                    values,
                    mask=column_mask,
                )


def moe_permute_bf16_pdl(
    input: torch.Tensor,
    permuted_output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    top_k: int,
    tile_size: int,
    ctas_per_sm: Tuple[int, int],
) -> None:
    """Permute BF16 activations while launching the dependent W4A16 GEMM early."""
    max_num_row_groups = triton.cdiv(
        max_num_permuted_tokens,
        _ROWS_PER_PROGRAM,
    )
    ctas_numerator, ctas_denominator = ctas_per_sm
    num_programs = min(
        max_num_row_groups,
        max(
            1,
            get_device_sm_count(input.device) * ctas_numerator // ctas_denominator,
        ),
    )
    _moe_permute_bf16_pdl_kernel[(num_programs,)](
        input,
        permuted_output,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        HIDDEN_SIZE=int(input.size(1)),
        TOP_K=top_k,
        TILE_SIZE=tile_size,
        BLOCK_K=_COPY_TILE_SIZE,
        ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
        num_warps=_NUM_WARPS,
        launch_pdl=True,
    )
