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

"""TraceTemplates for GEMM operations."""

import torch

from ..template import Const, Tensor, TraceTemplate, Var


def _mm_reference(A, B):
    return torch.matmul(A, B.T)


def _mm_fp8_reference(A, B):
    """Dequantize FP8 block-scale inputs and compute C = A @ B.T.

    B is in TRT-LLM block layout [K//block_size, N, block_size] and is
    reshaped to [K, N] before the matmul.
    """
    K_div_bs, N, block_size = B.shape
    B_fp32 = B.reshape(K_div_bs * block_size, N).to(torch.float32)
    A_fp32 = A.to(torch.float32)
    return torch.matmul(A_fp32, B_fp32.T).to(torch.bfloat16)


def _mm_mxfp8_reference(A, B, a_descale, b_descale):
    """Dequantize MXFP8 inputs (block size 32) and compute C = A @ B.T.

    a_descale: [M, K//32] uint8 interpreted as float scale per block.
    b_descale: [K//32, N] uint8 interpreted as float scale per block.
    """
    M, K = A.shape
    _, N = B.shape
    block_size = 32
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    # Apply per-block scales along the K dimension.
    a_scale = a_descale.to(torch.float32).repeat_interleave(block_size, dim=1)  # [M, K]
    b_scale = b_descale.to(torch.float32).repeat_interleave(block_size, dim=0)  # [K, N]
    A_scaled = A_fp32 * a_scale
    B_scaled = B_fp32 * b_scale
    return torch.matmul(A_scaled, B_scaled.T).to(torch.bfloat16)


def _mm_fp4_reference(A, B, a_descale, b_descale, block_size=16):
    """Dequantize FP4 inputs and compute C = A @ B.T.

    A and B are fp4 e2m1fn values packed two-per-byte as uint8.
    a_descale: [M, K//block_size], b_descale: [K, N//block_size].
    The reference unpacks the nibbles and applies the block scales.
    """
    def _unpack_fp4(packed, rows, cols):
        # Each byte holds two fp4 nibbles (low nibble = first element).
        lo = (packed & 0x0F).to(torch.float32)
        hi = ((packed >> 4) & 0x0F).to(torch.float32)
        # Interleave low/high nibbles along the last dimension.
        out = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
        return out

    M, K_packed = A.shape
    K = K_packed * 2
    _, N_packed = B.shape
    N = N_packed * 2

    A_fp32 = _unpack_fp4(A, M, K)
    B_fp32 = _unpack_fp4(B, K, N)

    # Apply per-block scales.
    a_scale = a_descale.to(torch.float32).repeat_interleave(block_size, dim=1)  # [M, K]
    b_scale = b_descale.to(torch.float32).repeat_interleave(block_size, dim=1)  # [K, N]
    A_scaled = A_fp32 * a_scale
    B_scaled = B_fp32 * b_scale
    return torch.matmul(A_scaled, B_scaled.T).to(torch.bfloat16)


mm_bf16_trace = TraceTemplate(
    op_type="gemm_bf16",
    description="General matrix multiply (GEMM) C = A @ B.T.",
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "A": Tensor(["M", "K"], param="a"),
        "B": Tensor(
            ["K", "N"],
            param="b",
            description="Weight matrix in column-major layout (physical shape [K, N]).",
        ),
    },
    outputs={
        "C": Tensor(["M", "N"], dtype_from="a"),
    },
    tags=["status:verified"],
    reference=_mm_reference,
)

mm_fp8_trace = TraceTemplate(
    op_type="gemm_fp8",
    description=(
        "FP8 block-scale GEMM C = A @ B.T (TRT-LLM layout). "
        "A is [M, K] float8_e4m3fn; B is [K//block_size, N, block_size] float8_e4m3fn."
    ),
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "A": Tensor(["M", "K"], param="a"),
        "B": Tensor(
            ["K_div_block_size", "N", "block_size"],
            param="b",
            description="FP8 weight in TRT-LLM block layout [K//block_size, N, block_size].",
        ),
    },
    outputs={
        "C": Tensor(["M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_mm_fp8_reference,
)

# ── MXFP8 GEMM ───────────────────────────────────────────────────────────────

mm_mxfp8_trace = TraceTemplate(
    op_type="gemm_mxfp8",
    description=(
        "MXFP8 GEMM C = A @ B.T (MX block size 32). "
        "A and B are float8_e4m3fn; scale tensors use block size 32."
    ),
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "A": Tensor(
            ["M", "K"],
            param="a",
            description="Input A tensor, float8_e4m3fn.",
        ),
        "B": Tensor(
            ["K", "N"],
            param="b",
            description="Input B tensor, float8_e4m3fn, column-major.",
        ),
        "a_descale": Tensor(
            ["M", "K_div_32"],
            description="Block scale for A, shape [M, K//32], uint8.",
        ),
        "b_descale": Tensor(
            ["K_div_32", "N"],
            description="Block scale for B, shape [K//32, N], uint8.",
        ),
    },
    outputs={
        "C": Tensor(["M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:mxfp8"],
    reference=_mm_mxfp8_reference,
)

# ── FP4 GEMM ─────────────────────────────────────────────────────────────────

mm_fp4_trace = TraceTemplate(
    op_type="gemm_fp4",
    description=(
        "FP4 GEMM C = A @ B.T. "
        "A and B are fp4 (e2m1fn_x2 packed as uint8); scale tensors use block_size."
    ),
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
        "block_size": Const(description="FP4 quantization block size (16 for nvfp4, 32 for mxfp4)."),
    },
    inputs={
        "A": Tensor(
            ["M", "K"],
            param="a",
            description="Input A tensor, fp4 e2m1fn_x2 packed as uint8.",
        ),
        "B": Tensor(
            ["K", "N"],
            param="b",
            description="Input B tensor, fp4 e2m1fn_x2 packed as uint8, column-major.",
        ),
        "a_descale": Tensor(
            ["M", "K_div_block_size"],
            description="Block scale for A, shape [M, K//block_size], float8_e4m3fn or uint8.",
        ),
        "b_descale": Tensor(
            ["K", "N_div_block_size"],
            description="Block scale for B, shape [K, N//block_size], float8_e4m3fn or uint8.",
        ),
    },
    outputs={
        "C": Tensor(["M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:fp4"],
    reference=_mm_fp4_reference,
)
