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
from ._init_helpers import (
    fp8_block_quant_1d,
    fp8_block_quant_2d,
    per_tensor_fp8_quantize,
)


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


def _mm_bf16_init(
    *,
    M: int,
    N: int = 4096,
    K: int = 4096,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mm_bf16``.

    ``B`` is constructed as ``randn(N, K).T`` to get column-major [K, N]
    matching the example call.
    """
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(N, K, dtype=dtype, device=device).T  # [K, N] col-major
    return {"a": a, "b": b}


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
    init=_mm_bf16_init,
)


def _mm_fp8_init(
    *,
    M: int,
    N: int = 1536,
    K: int = 7168,
    K_div_block_size: int = 0,  # derived
    block_size: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mm_fp8`` (TRT-LLM low-latency layout).

    Sourced from ``tests/gemm/test_mm_fp8.py``: ``input`` and ``mat2`` are
    sampled from ``randn`` (bf16) then per-tensor-quantized to FP8 via
    ``to_float8`` (mirrored as ``per_tensor_fp8_quantize``). ``alpha``
    is the product of the two dequant scales. The kernel internally
    consumes the TRT-LLM-permuted layout
    ``[K // block_size, N, block_size]``; the test invokes
    ``prepare_low_latency_gemm_weights`` to produce that — we just
    reshape post-quant to give a tensor of the right shape (numerics
    won't match the kernel without the actual permute).
    """
    del K_div_block_size
    torch.manual_seed(seed)
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    mat2_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    a, a_inv_s = per_tensor_fp8_quantize(input_bf16)
    mat2_fp8, mat2_inv_s = per_tensor_fp8_quantize(mat2_bf16)
    # Reshape mat2 [N, K] → [K//block_size, N, block_size] to match the
    # trace-template-declared layout. The real kernel uses
    # `prepare_low_latency_gemm_weights(mat2_fp8, _cache_permute_indices)`
    # to produce this shape with kernel-specific permutation.
    b = mat2_fp8.reshape(N, K // block_size, block_size).permute(1, 0, 2).contiguous()
    alpha = (a_inv_s * mat2_inv_s).reshape(())
    return {"a": a, "b": b, "alpha": alpha}


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
    init=_mm_fp8_init,
)

# ── MXFP8 GEMM ───────────────────────────────────────────────────────────────


def _mm_mxfp8_init(
    *,
    M: int,
    N: int = 4096,
    K: int = 4096,
    K_div_32: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mm_mxfp8``. Block size = 32.

    Sourced from ``tests/gemm/test_mm_mxfp8_sm120.py``: ``a`` and ``b``
    are ``randn`` bf16 passed through ``flashinfer.mxfp8_quantize`` (or
    the test's ``_prepare_mxfp8`` helper which is equivalent for
    swizzled layout). The trace declares ``b`` as ``[K, N]`` and the
    descales as uint8 block scales.
    """
    del K_div_32
    from flashinfer import mxfp8_quantize  # noqa: PLC0415

    torch.manual_seed(seed)
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    a, a_descale = mxfp8_quantize(a_bf16)
    b_fp8, b_descale = mxfp8_quantize(b_bf16)
    # The kernel takes b as the transposed view ([K, N]) of the [N, K] result.
    b = b_fp8.T.contiguous()
    # Trace declares b_descale as [K//32, N]; mxfp8_quantize returns it
    # along the same axis as b.
    b_descale = b_descale.T.contiguous()
    return {"a": a, "b": b, "a_descale": a_descale, "b_descale": b_descale}


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
    init=_mm_mxfp8_init,
)

# ── FP4 GEMM ─────────────────────────────────────────────────────────────────


def _mm_fp4_init(
    *,
    M: int,
    N: int = 2048,
    K: int = 7168,
    block_size: int = 16,
    K_div_block_size: int = 0,  # derived
    N_div_block_size: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mm_fp4``. NvFP4 uses block_size=16.

    Sourced from ``tests/gemm/test_mm_fp4.py::_test_mm_fp4`` (nvfp4
    branch): ``input`` and ``mat2`` are ``randn`` bf16, quantized via
    ``flashinfer.nvfp4_quantize`` with ``global_sf = (448 * 6) / amax``.
    Requires SM100+ at runtime; CPU smoke tests skip.
    """
    del K_div_block_size, N_div_block_size
    from flashinfer import nvfp4_quantize  # noqa: PLC0415
    from flashinfer.quantization.fp4_quantization import SfLayout  # noqa: PLC0415

    torch.manual_seed(seed)
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    mat2_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    global_sf_input = (
        torch.tensor([448.0 * 6.0], device=device)
        / input_bf16.float().abs().nan_to_num().max()
    )
    global_sf_mat2 = (
        torch.tensor([448.0 * 6.0], device=device)
        / mat2_bf16.float().abs().nan_to_num().max()
    )
    a, a_descale = nvfp4_quantize(
        input_bf16, global_sf_input, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    mat2_fp4, mat2_descale = nvfp4_quantize(
        mat2_bf16, global_sf_mat2, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return {
        "a": a,
        "b": mat2_fp4.T.contiguous(),
        "a_descale": a_descale,
        "b_descale": mat2_descale.T.contiguous(),
        "block_size": int(block_size),
    }


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
    init=_mm_fp4_init,
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


def _bmm_bf16_init(
    *,
    batch_size: int,
    M: int = 64,
    N: int = 64,
    K: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for batched ``bmm`` (bf16)."""
    torch.manual_seed(seed)
    A = torch.randn(batch_size, M, K, dtype=dtype, device=device)
    B = torch.randn(batch_size, K, N, dtype=dtype, device=device)
    return {"A": A, "B": B}


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
    init=_bmm_bf16_init,
)


def _bmm_fp8_init(
    *,
    batch_size: int,
    M: int = 64,
    N: int = 64,
    K: int = 128,
    scalar: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``bmm_fp8`` (per-tensor scale).

    Sourced from ``tests/gemm/test_bmm_fp8.py``: ``input`` and ``mat2``
    are ``randn`` bf16, per-tensor-quantized via ``to_float8``
    (mirrored as ``per_tensor_fp8_quantize``); the scales are scalar
    fp32 reciprocals.
    """
    del scalar
    torch.manual_seed(seed)
    input_bf16 = torch.randn(batch_size, M, K, dtype=torch.bfloat16, device=device)
    mat2_bf16 = torch.randn(batch_size, K, N, dtype=torch.bfloat16, device=device)
    A, A_inv_s = per_tensor_fp8_quantize(input_bf16)
    B, B_inv_s = per_tensor_fp8_quantize(mat2_bf16)
    return {
        "A": A,
        "B": B,
        "A_scale": A_inv_s.reshape(1),
        "B_scale": B_inv_s.reshape(1),
        "dtype": 1,
    }


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
    init=_bmm_fp8_init,
)
bmm_fp8_trace.axes["scalar"] = Var(description="A/B scale tensor length (typically 1).")


def _bmm_mxfp8_init(
    *,
    batch_size: int,
    M: int = 64,
    N: int = 64,
    K: int = 128,
    K_div_32: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``bmm_mxfp8`` (block size 32).

    Sourced from ``tests/gemm/test_bmm_mxfp8.py``: ``input`` and
    ``mat2`` are ``randn`` bf16 passed through
    ``flashinfer.mxfp8_quantize`` to produce float8_e4m3fn data and
    uint8 block scales.
    """
    del K_div_32
    from flashinfer import mxfp8_quantize  # noqa: PLC0415

    torch.manual_seed(seed)
    a_bf16 = torch.randn(batch_size, M, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(batch_size, K, N, dtype=torch.bfloat16, device=device)
    A, A_scale = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=True)
    B, B_scale = mxfp8_quantize(b_bf16, is_sf_swizzled_layout=True)
    return {
        "A": A,
        "B": B,
        "A_scale": A_scale,
        "B_scale": B_scale,
        "dtype": 1,
    }


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
    init=_bmm_mxfp8_init,
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


def _tinygemm_bf16_init(
    *,
    M: int,
    N: int = 4096,
    K: int = 4096,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for ``tinygemm_bf16``: ``out = input @ weight.T + bias``."""
    torch.manual_seed(seed)
    inp = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    out = torch.empty(M, N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    return {"input": inp, "weight": w, "out": out, "bias": bias}


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
    init=_tinygemm_bf16_init,
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


def _fmha_v2_prefill_deepseek_init(
    *,
    batch_size: int,
    seq_len: int = 128,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for ``fmha_v2_prefill_deepseek``."""
    import math as _math

    torch.manual_seed(seed)
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    return {
        "query": q,
        "key": k,
        "value": v,
        "out": out,
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "seq_len": int(seq_len),
        "scale_softmax": 1.0 / _math.sqrt(head_dim),
    }


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
    init=_fmha_v2_prefill_deepseek_init,
)


# ── fp8_blockscale_gemm_sm90 (FP8 block-scale GEMM with auto swapAB) ─────────


@torch.no_grad()
def _fp8_blockscale_gemm_sm90_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale=None,
    weight_scale=None,
    out=None,
    out_dtype=None,
    **_unused,
) -> torch.Tensor:
    """Reference for FP8 block-scale GEMM (SM90). Dequantizes via per-block
    scales (block size 128 along K, 128x128 for the weight) then matmul."""
    a = input.to(torch.float32)
    w = weight.to(torch.float32)
    if input_scale is not None:
        # input_scale: [M, K//128] — broadcast to [M, K].
        a_scale = input_scale.to(torch.float32).repeat_interleave(128, dim=-1)
        a = a * a_scale[..., : a.shape[-1]]
    if weight_scale is not None:
        # weight_scale: [N//128, K//128] — broadcast.
        w_scale = (
            weight_scale.to(torch.float32)
            .repeat_interleave(128, dim=0)
            .repeat_interleave(128, dim=1)
        )
        w = w * w_scale[: w.shape[0], : w.shape[1]]
    res = a @ w.T
    out_dtype = out_dtype or torch.bfloat16
    return res.to(out_dtype)


def _fp8_blockscale_gemm_sm90_init(
    *,
    M: int,
    N: int = 4096,
    K: int = 4096,
    K_div_128: int = 0,  # derived
    N_div_128: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for SM90 FP8 block-scale GEMM.

    Sourced from ``tests/gemm/test_fp8_blockscale_gemm.py`` /
    ``tests/gemm/test_groupwise_scaled_gemm_fp8.py``: ``input`` and
    ``weight`` are ``randn`` bf16 passed through the same
    ``fp8_block_quant_*`` helpers as MoE (1×128 input scale, 128×128
    weight scale).
    """
    del K_div_128, N_div_128
    torch.manual_seed(seed)
    inp_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    inp, input_scale = fp8_block_quant_1d(inp_bf16, block=128)
    weight, weight_scale = fp8_block_quant_2d(weight_bf16, block=128)
    return {
        "input": inp,
        "weight": weight,
        "input_scale": input_scale,
        "weight_scale": weight_scale,
    }


fp8_blockscale_gemm_sm90_trace = TraceTemplate(
    op_type="gemm_fp8",
    name_prefix="fp8_blockscale_gemm_sm90",
    description=(
        "SM90 FP8 block-scale GEMM with automatic swapAB (uses swapAB "
        "kernel for small M < 32). Block scales are 1x128 for input and "
        "128x128 for weight."
    ),
    axes={
        "M": Var(),
        "N": Const(abbrev="n"),
        "K": Const(abbrev="k"),
        "K_div_128": Var(description="K // 128 (input block scale dim)."),
        "N_div_128": Var(description="N // 128."),
    },
    inputs={
        "input": Tensor(["M", "K"]),
        "weight": Tensor(["N", "K"]),
        "input_scale": Tensor(
            ["M", "K_div_128"],
            dtype="float32",
            optional=True,
        ),
        "weight_scale": Tensor(
            ["N_div_128", "K_div_128"],
            dtype="float32",
            optional=True,
        ),
    },
    outputs={
        "out": Tensor(["M", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_fp8_blockscale_gemm_sm90_reference,
    init=_fp8_blockscale_gemm_sm90_init,
)


# ── grouped_gemm_nt_masked (Blackwell grouped GEMM with masked-M) ────────────


@torch.no_grad()
def _grouped_gemm_nt_masked_reference(
    lhs,
    rhs,
    out,
    masked_m,
    ab_dtype: str = "fp8",
    sf_dtype: str = "ue4m3",
    c_dtype: str = "bf16",
    sf_vec_size: int = 128,
    **_unused,
):
    """Reference for grouped_gemm_nt_masked: per-group masked GEMM where
    only the first ``masked_m[g]`` rows of group g participate. Mutates
    ``out`` in-place; returns it.

    The kernel internally dequantizes via fp8/mxfp4 block scales — this
    reference performs a straight bf16 matmul of dequantized inputs and
    is intended for shape/finite validation. For numerics use the
    kernel's own unit test suite under ``tests/gemm/``.
    """
    lhs_data, _lhs_sf = lhs
    rhs_data, _rhs_sf = rhs
    G = lhs_data.shape[0]
    for g in range(G):
        m = int(masked_m[g].item())
        if m <= 0:
            continue
        a = lhs_data[g, :m].to(torch.float32)
        b = rhs_data[g].to(torch.float32)
        out[g, :m] = (a @ b.T).to(out.dtype)
    return out


def _grouped_gemm_nt_masked_init(
    *,
    num_groups: int,
    max_m: int = 64,
    N: int = 4096,
    K: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``grouped_gemm_nt_masked`` (Blackwell MoE FC2).

    Sourced from ``tests/gemm/test_groupwise_scaled_gemm_fp8.py``:
    per-group ``randn`` bf16 lhs/rhs quantized via the same
    1×128 / 128×128 block scheme as ``fp8_blockscale_gemm_sm90``.
    """
    torch.manual_seed(seed)
    lhs_bf16 = torch.randn(num_groups, max_m, K, dtype=torch.bfloat16, device=device)
    rhs_bf16 = torch.randn(num_groups, N, K, dtype=torch.bfloat16, device=device)
    # Per-group 1D / 2D block quantization.
    lhs_data = torch.empty(
        num_groups, max_m, K, dtype=torch.float8_e4m3fn, device=device
    )
    lhs_sf = torch.empty(
        num_groups, max_m, K // 128, dtype=torch.float32, device=device
    )
    for g in range(num_groups):
        q, s = fp8_block_quant_1d(lhs_bf16[g], block=128)
        lhs_data[g] = q
        lhs_sf[g] = s
    rhs_data, rhs_sf = fp8_block_quant_2d(rhs_bf16, block=128)
    out = torch.empty(num_groups, max_m, N, dtype=torch.bfloat16, device=device)
    masked_m = torch.full((num_groups,), max_m, dtype=torch.int32, device=device)
    return {
        "lhs": (lhs_data, lhs_sf),
        "rhs": (rhs_data, rhs_sf),
        "out": out,
        "masked_m": masked_m,
        "ab_dtype": 0,
        "sf_dtype": 0,
        "c_dtype": 0,
        "sf_vec_size": 128,
    }


grouped_gemm_nt_masked_trace = TraceTemplate(
    op_type="gemm_fp8",
    name_prefix="grouped_gemm_nt_masked",
    description=(
        "Blackwell grouped GEMM with masked-M per group. Each group "
        "computes ``out[g, :masked_m[g]] = lhs[g, :masked_m[g]] @ "
        "rhs[g].T`` with FP8 / MXFP4 block-scale dequant. Used in "
        "MoE expert FC2 path."
    ),
    axes={
        "num_groups": Var(description="Number of expert groups."),
        "max_m": Var(description="Max rows per group (padded)."),
        "N": Const(abbrev="n"),
        "K": Const(abbrev="k"),
    },
    inputs={
        "lhs": Tensor(
            ["num_groups", "max_m", "K"],
            description="Tuple (lhs_data, lhs_sf): quantized A tensor + scales.",
        ),
        "rhs": Tensor(
            ["num_groups", "N", "K"],
            description="Tuple (rhs_data, rhs_sf): quantized B tensor + scales.",
        ),
        "out": Tensor(
            ["num_groups", "max_m", "N"],
            description="In-place output buffer.",
        ),
        "masked_m": Tensor(
            ["num_groups"],
            dtype="int32",
            description="Per-group valid row count.",
        ),
        "ab_dtype": Scalar("int32"),
        "sf_dtype": Scalar("int32"),
        "c_dtype": Scalar("int32"),
        "sf_vec_size": Scalar("int32"),
    },
    outputs={
        "out": Tensor(["num_groups", "max_m", "N"], dtype_from="out"),
    },
    tags=["status:verified", "moe", "quantization:fp8"],
    reference=_grouped_gemm_nt_masked_reference,
    init=_grouped_gemm_nt_masked_init,
)


# ── batch_deepgemm_fp8_nt_groupwise (batched FP8 group-wise GEMM) ────────────


@torch.no_grad()
def _batch_deepgemm_fp8_nt_groupwise_reference(
    a, b, a_scale, b_scale, masked_m, expected_m, **_unused
):
    """Reference for batch_deepgemm_fp8_nt_groupwise. Per-batch FP8 GEMM
    with 1x128 input scales and 128x128 weight scales, masked-M variant.
    """
    B = a.shape[0]
    M_max = a.shape[1]
    N = b.shape[1]
    out = torch.zeros(B, M_max, N, dtype=torch.bfloat16, device=a.device)
    for g in range(B):
        m = int(masked_m[g].item())
        if m <= 0:
            continue
        af = a[g, :m].to(torch.float32)
        bf = b[g].to(torch.float32)
        if a_scale is not None:
            sa = a_scale[g, :m].to(torch.float32).repeat_interleave(128, dim=-1)
            af = af * sa[:, : af.shape[-1]]
        if b_scale is not None:
            sb = (
                b_scale[g]
                .to(torch.float32)
                .repeat_interleave(128, dim=0)
                .repeat_interleave(128, dim=1)
            )
            bf = bf * sb[: bf.shape[0], : bf.shape[1]]
        out[g, :m] = (af @ bf.T).to(torch.bfloat16)
    return out


def _batch_deepgemm_fp8_nt_groupwise_init(
    *,
    batch_size: int,
    M_max: int = 128,
    N: int = 4096,
    K: int = 4096,
    K_div_128: int = 0,
    N_div_128: int = 0,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for batched DeepGEMM FP8 group-wise GEMM.

    Sourced from ``tests/gemm/test_groupwise_scaled_gemm_fp8.py`` (DeepGEMM
    backend): per-batch ``randn`` bf16 a/b passed through the 1×128 / 128×128
    block-quantization helpers (same scheme as MoE / SM90 block-scale).
    """
    del K_div_128, N_div_128
    torch.manual_seed(seed)
    a_bf16 = torch.randn(batch_size, M_max, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(batch_size, N, K, dtype=torch.bfloat16, device=device)
    a = torch.empty(batch_size, M_max, K, dtype=torch.float8_e4m3fn, device=device)
    a_scale = torch.empty(
        batch_size, M_max, K // 128, dtype=torch.float32, device=device
    )
    for g in range(batch_size):
        q, s = fp8_block_quant_1d(a_bf16[g], block=128)
        a[g] = q
        a_scale[g] = s
    b, b_scale = fp8_block_quant_2d(b_bf16, block=128)
    masked_m = torch.full((batch_size,), M_max, dtype=torch.int32, device=device)
    return {
        "a": a,
        "b": b,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "masked_m": masked_m,
        "expected_m": int(M_max),
    }


batch_deepgemm_fp8_nt_groupwise_trace = TraceTemplate(
    op_type="gemm_fp8",
    name_prefix="batch_deepgemm_fp8_nt_groupwise",
    description=(
        "Batched FP8 group-wise GEMM (DeepGEMM backend). 1x128 scale "
        "granularity along K for input; 128x128 for weight. Mask-M "
        "variant — only the first ``masked_m[g]`` rows of group g "
        "participate."
    ),
    axes={
        "batch_size": Var(),
        "M_max": Var(),
        "N": Const(abbrev="n"),
        "K": Const(abbrev="k"),
        "K_div_128": Var(description="K // 128 (input block scale dim)."),
        "N_div_128": Var(description="N // 128."),
    },
    inputs={
        "a": Tensor(["batch_size", "M_max", "K"]),
        "b": Tensor(["batch_size", "N", "K"]),
        "a_scale": Tensor(
            ["batch_size", "M_max", "K_div_128"],
            dtype="float32",
        ),
        "b_scale": Tensor(
            ["batch_size", "N_div_128", "K_div_128"],
            dtype="float32",
        ),
        "masked_m": Tensor(["batch_size"], dtype="int32"),
        "expected_m": Scalar("int32"),
    },
    outputs={
        "out": Tensor(["batch_size", "M_max", "N"], dtype="bfloat16"),
    },
    tags=["status:verified", "quantization:float8_e4m3fn"],
    reference=_batch_deepgemm_fp8_nt_groupwise_reference,
    init=_batch_deepgemm_fp8_nt_groupwise_init,
)


# ── mm_M1_16_K7168_N256 (DeepSeek-V3 router GEMM, fixed shape) ───────────────


@torch.no_grad()
def _mm_M1_16_K7168_N256_reference(
    mat_a, mat_b, out, launch_with_pdl: bool = False, **_unused
):
    """Reference for the DeepSeek-V3 router GEMM (M=1..16, K=7168, N=256).
    Mutates ``out`` in-place; returns it.
    """
    a = mat_a.to(torch.float32)
    b = mat_b.to(torch.float32)
    out.copy_((a @ b).to(out.dtype))
    return out


def _mm_M1_16_K7168_N256_init(
    *,
    M: int,
    K: int = 7168,
    N: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for the DeepSeek-V3 router GEMM. Mutates ``out``."""
    torch.manual_seed(seed)
    M = max(1, min(int(M), 16))
    mat_a = torch.randn(M, K, dtype=dtype, device=device)
    mat_b = torch.randn(K, N, dtype=dtype, device=device)
    out = torch.empty(M, N, dtype=dtype, device=device)
    return {"mat_a": mat_a, "mat_b": mat_b, "out": out}


mm_M1_16_K7168_N256_trace = TraceTemplate(
    op_type="gemm_bf16",
    name_prefix="mm_M1_16_K7168_N256",
    description=(
        "DeepSeek-V3 router-GEMM specialization: out = mat_a @ mat_b for "
        "M in [1, 16], K=7168, N=256. Mutates ``out`` in-place."
    ),
    axes={
        "M": Var(description="Number of tokens (1-16)."),
        "K": Const(description="DeepSeek-V3 hidden dim (7168).", abbrev="k"),
        "N": Const(description="Number of experts (256).", abbrev="n"),
    },
    inputs={
        "mat_a": Tensor(["M", "K"]),
        "mat_b": Tensor(["K", "N"]),
        "out": Tensor(["M", "N"], description="In-place output."),
    },
    outputs={
        "out": Tensor(["M", "N"], dtype_from="mat_a"),
    },
    tags=["status:verified", "moe"],
    reference=_mm_M1_16_K7168_N256_reference,
    init=_mm_M1_16_K7168_N256_init,
)


# ── trtllm_ragged_attention_deepseek (DeepSeek ragged prefill) ───────────────


@torch.no_grad()
def _trtllm_ragged_attention_deepseek_reference(
    query,
    key,
    value,
    workspace_buffer,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    o_sf_scale,
    batch_size,
    window_left,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    enable_pdl,
    is_causal,
    return_lse,
    **_unused,
):
    """Reference for DeepSeek ragged prefill: variable-length per-batch
    SDPA on ragged Q/K/V with optional causal mask and sliding window.
    Mutates / returns the output tensor.
    """
    s = (
        float(bmm1_scale)
        if not isinstance(bmm1_scale, torch.Tensor)
        else float(bmm1_scale.item())
    )
    s2 = (
        float(bmm2_scale)
        if not isinstance(bmm2_scale, torch.Tensor)
        else float(bmm2_scale.item())
    )
    H = query.shape[-2]
    out = torch.zeros_like(query, dtype=torch.float32)
    for b in range(int(batch_size)):
        q_start = int(cum_seq_lens_q[b].item())
        q_end = int(cum_seq_lens_q[b + 1].item())
        kv_start = int(cum_seq_lens_kv[b].item())
        kv_end = int(cum_seq_lens_kv[b + 1].item())
        if q_end <= q_start or kv_end <= kv_start:
            continue
        q_b = query[q_start:q_end].to(torch.float32)
        k_b = key[kv_start:kv_end].to(torch.float32)
        v_b = value[kv_start:kv_end].to(torch.float32)
        qi = q_end - q_start
        kv_len = kv_end - kv_start
        delta = kv_len - qi
        for h in range(H):
            logits = (q_b[:, h] @ k_b[:, h].T) * s
            if is_causal:
                mask = torch.full_like(logits, float("-inf"))
                for i in range(qi):
                    lo = max(0, i + 1 + delta - max(0, int(window_left)))
                    hi = i + 1 + max(0, delta)
                    mask[i, lo:hi] = 0.0
                logits = logits + mask
            attn = torch.softmax(logits, dim=-1)
            out[q_start:q_end, h] = (attn @ v_b[:, h]) * s2
    return out.to(query.dtype)


def _trtllm_ragged_attention_deepseek_init(
    *,
    num_q_tokens: int,
    num_kv_tokens: int = 256,
    batch_size: int = 4,
    batch_size_plus_1: int = 0,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    """Build inputs for ``trtllm_ragged_attention_deepseek``."""
    import math as _math

    del batch_size_plus_1
    torch.manual_seed(seed)
    q = torch.randn(num_q_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_kv_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    workspace_buffer = torch.empty(num_q_tokens, dtype=torch.int8, device=device)
    seq_lens = torch.full(
        (batch_size,),
        max(1, num_kv_tokens // max(1, batch_size)),
        dtype=torch.int32,
        device=device,
    )
    cum_q = torch.tensor(
        [i * (num_q_tokens // max(1, batch_size)) for i in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )
    cum_kv = torch.tensor(
        [i * (num_kv_tokens // max(1, batch_size)) for i in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )
    return {
        "query": q,
        "key": k,
        "value": v,
        "workspace_buffer": workspace_buffer,
        "seq_lens": seq_lens,
        "max_q_len": int(num_q_tokens // max(1, batch_size)),
        "max_kv_len": int(num_kv_tokens // max(1, batch_size)),
        "bmm1_scale": 1.0 / _math.sqrt(head_dim),
        "bmm2_scale": 1.0,
        "o_sf_scale": 1.0,
        "batch_size": int(batch_size),
        "window_left": -1,
        "cum_seq_lens_q": cum_q,
        "cum_seq_lens_kv": cum_kv,
        "is_causal": 1,
        "return_lse": 0,
    }


trtllm_ragged_attention_deepseek_trace = TraceTemplate(
    op_type="trtllm_paged",
    name_prefix="trtllm_ragged_attention_deepseek",
    description=(
        "DeepSeek-specific TRT-LLM ragged-batch attention. Variable-length "
        "Q and KV tensors (cum_seq_lens_q / cum_seq_lens_kv), optional "
        "causal + sliding-window masks. Used in DeepSeek-V3 prefill."
    ),
    axes={
        "num_q_tokens": Var(),
        "num_kv_tokens": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "batch_size": Var(),
        "batch_size_plus_1": Var(description="batch_size + 1."),
    },
    inputs={
        "query": Tensor(["num_q_tokens", "num_heads", "head_dim"]),
        "key": Tensor(["num_kv_tokens", "num_heads", "head_dim"]),
        "value": Tensor(["num_kv_tokens", "num_heads", "head_dim"]),
        "workspace_buffer": Tensor(["num_q_tokens"], dtype="int8"),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_q_len": Scalar("int32"),
        "max_kv_len": Scalar("int32"),
        "bmm1_scale": Scalar("float32"),
        "bmm2_scale": Scalar("float32"),
        "o_sf_scale": Scalar("float32"),
        "batch_size": Scalar("int32"),
        "window_left": Scalar("int32"),
        "cum_seq_lens_q": Tensor(["batch_size_plus_1"], dtype="int32"),
        "cum_seq_lens_kv": Tensor(["batch_size_plus_1"], dtype="int32"),
        "is_causal": Scalar("int32"),
        "return_lse": Scalar("int32"),
    },
    outputs={
        "output": Tensor(["num_q_tokens", "num_heads", "head_dim"], dtype_from="query"),
    },
    tags=["status:verified", "stage:prefill", "backend:trtllm"],
    reference=_trtllm_ragged_attention_deepseek_reference,
    init=_trtllm_ragged_attention_deepseek_init,
)
