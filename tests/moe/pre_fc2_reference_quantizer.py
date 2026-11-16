"""
Copyright (c) 2025 by FlashInfer team.

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

"""Reference weighted NVFP4 quantizer for the pre-FC2 routing tests.

The library computes this in the gather GEMM1 epilogue. This standalone Triton
kernel takes the BF16 FC1 output instead and serves as the bitwise oracle.
"""

import torch
import triton
import triton.language as tl

from flashinfer.quantization.nvfp4_quantization_utils import FLOAT4_E2M1_MAX
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    NVFP4_BLOCK_SIZE,
    NVFP4_VALUES_PER_BYTE,
    NVFP4_SF_TILE_ROWS,
    NVFP4_SF_TILE_COLS,
    NVFP4_SF_ROW_GROUP_SIZE,
)

_BLOCK_SIZE = tl.constexpr(NVFP4_BLOCK_SIZE)
_VALUES_PER_BYTE = tl.constexpr(NVFP4_VALUES_PER_BYTE)
_PACKED_BYTES_PER_BLOCK = tl.constexpr(NVFP4_BLOCK_SIZE // NVFP4_VALUES_PER_BYTE)
_FP4_MAX = tl.constexpr(FLOAT4_E2M1_MAX)
_SF_TILE_ROWS = tl.constexpr(NVFP4_SF_TILE_ROWS)
_SF_TILE_COLS = tl.constexpr(NVFP4_SF_TILE_COLS)
_SF_ROW_GROUP_SIZE = tl.constexpr(NVFP4_SF_ROW_GROUP_SIZE)
_SF_ROW_GROUPS = tl.constexpr(NVFP4_SF_TILE_ROWS // NVFP4_SF_ROW_GROUP_SIZE)


@triton.jit
def _sf_offset(row, col, num_cols: tl.constexpr):
    """Map logical block-scale coordinates into the MMA 128x4 swizzle."""
    return (
        row // _SF_TILE_ROWS * (_SF_TILE_ROWS * num_cols)
        + col // _SF_TILE_COLS * (_SF_TILE_ROWS * _SF_TILE_COLS)
        + row % _SF_ROW_GROUP_SIZE * (_SF_ROW_GROUPS * _SF_TILE_COLS)
        + row % _SF_TILE_ROWS // _SF_ROW_GROUP_SIZE * _SF_TILE_COLS
        + col % _SF_TILE_COLS
    )


@triton.jit
def _clamped_relu2_weight_nvfp4_quant(
    Raw,
    Inverse,
    Weights,
    GlobalScale,
    Experts,
    LOCAL_EXPERT_OFFSET: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    Quantized,
    Scales,
    N: tl.constexpr,
    Clamp,
    BLOCKS: tl.constexpr,
    COL_PROGRAMS: tl.constexpr,
):
    """Fuse BF16-rounded clamped ReLU2, route weighting and block-16 NVFP4 packing.

    Each program handles one route and walks the column chunks
    ``program_id(1), program_id(1) + COL_PROGRAMS, ...``. Every block of 16
    values is computed independently, so the result does not depend on how the
    chunks are split across programs.
    """
    # BF16 intermediates and packed outputs can exceed signed 32-bit element offsets.
    route = tl.program_id(0).to(tl.int64)
    row = tl.load(Inverse + route).to(tl.int64)
    if row < 0:
        # Routes to other ranks' experts have no permuted row.
        return
    clamp = tl.load(Clamp)
    weight = tl.load(Weights + route)
    expert = tl.load(Experts + route) - LOCAL_EXPERT_OFFSET if NUM_SCALES > 1 else 0
    global_scale = tl.load(GlobalScale + expert)
    num_chunks = tl.cdiv(N // _BLOCK_SIZE, BLOCKS)
    for chunk in range(tl.program_id(1), num_chunks, COL_PROGRAMS):
        blocks = chunk * BLOCKS + tl.arange(0, BLOCKS)
        cols = blocks[:, None] * _BLOCK_SIZE + tl.arange(0, _BLOCK_SIZE)[None, :]
        # GEMM1 stores compact expanded rows; only the quantized output is padded.
        value = tl.load(Raw + route * N + cols, cols < N, 0).to(tl.float32)
        # Explicit PTX keeps the soft-clamp approximation fixed before BF16 rounding.
        value = clamp * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[value / clamp],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        value = value.to(tl.bfloat16).to(tl.float32)
        value = tl.maximum(value, 0.0)
        value = (value * value).to(tl.bfloat16).to(tl.float32)
        value = (value * weight).to(tl.bfloat16).to(tl.float32)
        amax = tl.max(tl.abs(value), 1)
        scale = (amax * (global_scale / _FP4_MAX)).to(tl.float8e4nv).to(tl.float32)
        tl.store(
            Scales + _sf_offset(row, blocks, N // _BLOCK_SIZE),
            scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
            blocks < N // _BLOCK_SIZE,
        )
        value = value * tl.where(scale > 0.0, global_scale / scale, 0.0)[:, None]
        pair = tl.reshape(value, (BLOCKS, _PACKED_BYTES_PER_BLOCK, _VALUES_PER_BYTE))
        lo = tl.sum(
            tl.where(tl.arange(0, _VALUES_PER_BYTE)[None, None, :] == 0, pair, 0.0), 2
        )
        hi = tl.sum(
            tl.where(tl.arange(0, _VALUES_PER_BYTE)[None, None, :] == 1, pair, 0.0), 2
        )
        # Hardware round-to-nearest, saturating conversion packs two E2M1 values.
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, $2, $1; cvt.u32.u8 $0, p; }",
            constraints="=r,f,f",
            args=[lo, hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
        dest = (
            row * (N // _VALUES_PER_BYTE)
            + blocks[:, None] * _PACKED_BYTES_PER_BLOCK
            + tl.arange(0, _PACKED_BYTES_PER_BLOCK)[None, :]
        )
        tl.store(Quantized + dest, packed, blocks[:, None] < N // _BLOCK_SIZE)


_BLOCKS_PER_PROGRAM = 64
_NUM_WARPS = 4


def quantize_reference(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    routing_weights: torch.Tensor,
    inverse: torch.Tensor,
    experts: torch.Tensor,
    local_expert_offset: int,
    clamp: torch.Tensor,
    out: torch.Tensor,
    out_sf: torch.Tensor,
) -> None:
    """Weight BF16 FC1 rows (one per route) and quantize them to permuted rows."""
    if inverse.numel() == 0:
        return
    _clamped_relu2_weight_nvfp4_quant[(inverse.numel(), 1)](
        input,
        inverse,
        routing_weights,
        global_scale,
        experts,
        local_expert_offset,
        global_scale.numel(),
        out,
        out_sf,
        input.shape[1],
        clamp,
        _BLOCKS_PER_PROGRAM,
        1,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
