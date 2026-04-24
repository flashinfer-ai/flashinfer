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

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _mm_reference(A, B):
    # B is physically [K, N] (column-major weight), so C = A @ B.
    return torch.matmul(A, B)


def _mm_fp8_reference(A, B):
    """Dequantize FP8 block-scale inputs and compute C = A @ B.

    B is in TRT-LLM block layout [K//block_size, N, block_size] and is
    reshaped to [K, N] before the matmul.
    """
    K_div_bs, N, block_size = B.shape
    B_fp32 = B.reshape(K_div_bs * block_size, N).to(torch.float32)
    A_fp32 = A.to(torch.float32)
    return torch.matmul(A_fp32, B_fp32).to(torch.bfloat16)


def _mm_mxfp8_reference(A, B, a_descale, b_descale):
    """Dequantize MXFP8 inputs (block size 32) and compute C = A @ B.

    a_descale: [M, K//32] uint8 interpreted as float scale per block.
    b_descale: [K//32, N] uint8 interpreted as float scale per block.
    """
    _, K = A.shape
    block_size = 32
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    # Apply per-block scales along the K dimension.
    a_scale = a_descale.to(torch.float32).repeat_interleave(block_size, dim=1)  # [M, K]
    b_scale = b_descale.to(torch.float32).repeat_interleave(block_size, dim=0)  # [K, N]
    A_scaled = A_fp32 * a_scale
    B_scaled = B_fp32 * b_scale
    return torch.matmul(A_scaled, B_scaled).to(torch.bfloat16)


def _mm_fp4_reference(A, B, a_descale, b_descale, block_size=16):
    """Dequantize FP4 inputs and compute C = A @ B.

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
    return torch.matmul(A_scaled, B_scaled).to(torch.bfloat16)


mm_bf16_trace = TraceTemplate(
    op_type="gemm_bf16",
    description="General matrix multiply (GEMM) C = A @ B (B is column-major [K, N]).",
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
        "FP8 block-scale GEMM C = A @ B (TRT-LLM layout). "
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
        "MXFP8 GEMM C = A @ B (MX block size 32). "
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
        "FP4 GEMM C = A @ B. "
        "A and B are fp4 (e2m1fn_x2 packed as uint8); scale tensors use block_size."
    ),
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
        "block_size": Const(
            description="FP4 quantization block size (16 for nvfp4, 32 for mxfp4)."
        ),
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
        "block_size": Scalar(
            "int32",
            description="FP4 quantization block size (16 for nvfp4, 32 for mxfp4).",
        ),
    },
    outputs={
        "C": Tensor(["M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:fp4"],
    reference=_mm_fp4_reference,
)


# ── Batched matmuls (BMM) ────────────────────────────────────────────────────


def _bmm_reference(A, B):
    """Batched matmul C[b] = A[b] @ B[b]."""
    return torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(A.dtype)


def _bmm_fp8_reference(A, B, A_scale, B_scale, dtype):
    """Reference per-tensor FP8 BMM: dequantize then matmul."""
    A_f = A.to(torch.float32) * A_scale.to(torch.float32)
    B_f = B.to(torch.float32) * B_scale.to(torch.float32)
    return torch.matmul(A_f, B_f).to(dtype)


def _bmm_mxfp8_reference(A, B, A_scale, B_scale, dtype):
    """Reference MXFP8 BMM (block size 32)."""
    block = 32
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    a_scale = A_scale.to(torch.float32).repeat_interleave(block, dim=-1)
    b_scale = B_scale.to(torch.float32).repeat_interleave(block, dim=-2)
    return torch.matmul(A_f * a_scale, B_f * b_scale).to(dtype)


bmm_bf16_trace = TraceTemplate(
    op_type="bmm_bf16",
    description="Batched matrix multiply C[b] = A[b] @ B[b] (bf16/fp16).",
    axes={
        "batch_size": Var(),
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "A": Tensor(["batch_size", "M", "K"]),
        "B": Tensor(["batch_size", "K", "N"]),
    },
    outputs={
        "C": Tensor(["batch_size", "M", "N"], dtype_from="A"),
    },
    tags=["status:verified"],
    reference=_bmm_reference,
)


bmm_fp8_trace = TraceTemplate(
    op_type="bmm_fp8",
    description=(
        "Per-tensor FP8 batched matmul. A and B are float8_e4m3fn; "
        "A_scale/B_scale are scalar tensors holding the dequant scales."
    ),
    axes={
        "batch_size": Var(),
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "A": Tensor(["batch_size", "M", "K"]),
        "B": Tensor(["batch_size", "K", "N"]),
        "A_scale": Tensor(
            ["scalar"], dtype="float32", description="Per-tensor dequant scale for A."
        ),
        "B_scale": Tensor(
            ["scalar"], dtype="float32", description="Per-tensor dequant scale for B."
        ),
        "dtype": Scalar("int32", description="Output dtype enum."),
    },
    outputs={
        "C": Tensor(["batch_size", "M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_bmm_fp8_reference,
)
bmm_fp8_trace.axes["scalar"] = Var(description="A/B scale tensor length (typically 1).")


bmm_mxfp8_trace = TraceTemplate(
    op_type="bmm_mxfp8",
    description=(
        "MXFP8 batched matmul (MX block size 32). A, B are float8_e4m3fn; "
        "A_scale/B_scale are uint8 block scales (block size 32 along K)."
    ),
    axes={
        "batch_size": Var(),
        "M": Var(),
        "N": Const(),
        "K": Const(),
        "K_div_32": Var(description="K // 32 (MX block count)."),
    },
    inputs={
        "A": Tensor(["batch_size", "M", "K"]),
        "B": Tensor(["batch_size", "K", "N"]),
        "A_scale": Tensor(
            ["batch_size", "M", "K_div_32"], description="MX block scales for A."
        ),
        "B_scale": Tensor(
            ["batch_size", "K_div_32", "N"], description="MX block scales for B."
        ),
        "dtype": Scalar("int32", description="Output dtype enum."),
    },
    outputs={
        "C": Tensor(["batch_size", "M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:mxfp8"],
    reference=_bmm_mxfp8_reference,
)


# ── tinygemm_bf16 (small bf16 GEMM with bias) ────────────────────────────────


@torch.no_grad()
def _tinygemm_bf16_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias=None,
    use_pdl: bool = False,
    **_unused,
) -> None:
    """Reference for tinygemm_bf16: out = input @ weight.T (+ bias). In-place."""
    a = input.to(torch.float32)
    w = weight.to(torch.float32)
    res = a @ w.T
    if bias is not None:
        res = res + bias.to(torch.float32).unsqueeze(0)
    out.copy_(res.to(out.dtype))


tinygemm_bf16_trace = TraceTemplate(
    op_type="gemm_bf16",
    name_prefix="tinygemm_bf16",
    description=(
        "SM90+ small-batch bf16 GEMM (F.linear-equivalent): "
        "out = input @ weight.T + bias. Optimized for tiny M (1-8 rows). "
        "Mutates ``out`` in place."
    ),
    axes={
        "M": Var(description="Number of rows in input (small)."),
        "N": Const(abbrev="n"),
        "K": Const(abbrev="k"),
    },
    inputs={
        "input": Tensor(["M", "K"]),
        "weight": Tensor(["N", "K"]),
        "out": Tensor(["M", "N"], description="In-place output buffer."),
        "bias": Tensor(["N"], optional=True),
        "use_pdl": Scalar("int32", optional=True),
    },
    outputs={
        "out": Tensor(["M", "N"], dtype_from="input"),
    },
    tags=["status:verified"],
    reference=_tinygemm_bf16_reference,
)


# ── fmha_v2_prefill_deepseek (separate Q/K/V variant of FMHA v2 prefill) ─────


@torch.no_grad()
def _fmha_v2_prefill_deepseek_reference(
    query,
    key,
    value,
    out,
    num_heads,
    head_dim,
    seq_len,
    scale_softmax,
    scale_bmm1=None,
    scale_bmm2=None,
    return_lse: bool = False,
    lse=None,
    **_unused,
) -> torch.Tensor:
    """Reference for fmha_v2_prefill_deepseek: per-batch causal SDPA on
    separate Q/K/V tensors, fixed seq_len, GQA-MHA with num_heads heads.
    Mutates ``out`` in-place; returns it.
    """
    B, S, H, D = query.shape
    s = float(scale_bmm1 or scale_softmax)
    b2 = float(scale_bmm2) if scale_bmm2 is not None else 1.0
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    # Per head, per batch: causal attention.
    for batch in range(B):
        for h in range(H):
            logits = q[batch, :, h] @ k[batch, :, h].T * s
            mask = torch.triu(torch.ones_like(logits) * float("-inf"), diagonal=1)
            logits = logits + mask
            attn = torch.softmax(logits, dim=-1)
            out[batch, :, h] = (attn @ v[batch, :, h] * b2).to(out.dtype)
    return out


fmha_v2_prefill_deepseek_trace = TraceTemplate(
    op_type="trtllm_paged",
    name_prefix="fmha_v2_prefill_deepseek",
    description=(
        "DeepSeek-specific FMHA v2 prefill: separate Q/K/V tensors, "
        "fixed seq_len, causal SDPA per batch. Mutates ``out`` in-place."
    ),
    axes={
        "batch_size": Var(),
        "seq_len": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "query": Tensor(["batch_size", "seq_len", "num_heads", "head_dim"]),
        "key": Tensor(["batch_size", "seq_len", "num_heads", "head_dim"]),
        "value": Tensor(["batch_size", "seq_len", "num_heads", "head_dim"]),
        "out": Tensor(
            ["batch_size", "seq_len", "num_heads", "head_dim"],
            description="In-place output buffer.",
        ),
        "num_heads": Scalar("int32"),
        "head_dim": Scalar("int32"),
        "seq_len": Scalar("int32"),
        "scale_softmax": Scalar("float32"),
        "scale_bmm1": Scalar("float32", optional=True),
        "scale_bmm2": Scalar("float32", optional=True),
    },
    outputs={
        "out": Tensor(
            ["batch_size", "seq_len", "num_heads", "head_dim"], dtype_from="query"
        ),
    },
    tags=["status:verified", "stage:prefill", "backend:trtllm"],
    reference=_fmha_v2_prefill_deepseek_reference,
)
