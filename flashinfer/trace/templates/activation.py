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

"""TraceTemplates for activation functions."""

import torch
import torch.nn.functional as F

from ..template import Const, Tensor, TraceTemplate, Var

# ── SiLU and Mul ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _silu_and_mul_reference(input):
    """Fused SiLU + Mul: silu(input[..., :H]) * input[..., H:]"""
    half = input.shape[-1] // 2
    return F.silu(input[..., :half]) * input[..., half:]


silu_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="silu_and_mul",
    description="Fused SiLU + Mul: silu(x[:H]) * x[H:]. Used in LLaMA/Mistral FFN.",
    axes={
        "num_tokens": Var(description="Total number of tokens (batch_size * seq_len)."),
        "hidden_size": Const(
            abbrev="h", description="Output hidden size (input is 2*h)."
        ),
    },
    inputs={
        "input": Tensor(
            ["num_tokens", "hidden_size"],
            param="input",
            description="Gated input tensor of shape [num_tokens, 2*hidden_size].",
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_silu_and_mul_reference,
)

# ── GeLU Tanh and Mul ────────────────────────────────────────────────────────


@torch.no_grad()
def _gelu_tanh_and_mul_reference(input):
    """Fused GeLU (tanh approx) + Mul: gelu_tanh(x[:H]) * x[H:]"""
    half = input.shape[-1] // 2
    return F.gelu(input[..., :half], approximate="tanh") * input[..., half:]


gelu_tanh_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="gelu_tanh_and_mul",
    description="Fused GeLU (tanh approx) + Mul: gelu_tanh(x[:H]) * x[H:]. Used in BERT/GPT FFN.",
    axes={
        "num_tokens": Var(description="Total number of tokens."),
        "hidden_size": Const(
            abbrev="h", description="Output hidden size (input is 2*h)."
        ),
    },
    inputs={
        "input": Tensor(
            ["num_tokens", "hidden_size"],
            param="input",
            description="Gated input tensor of shape [num_tokens, 2*hidden_size].",
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_gelu_tanh_and_mul_reference,
)

# ── GeLU and Mul ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _gelu_and_mul_reference(input):
    """Fused GeLU (exact) + Mul: gelu(x[:H]) * x[H:]"""
    half = input.shape[-1] // 2
    return F.gelu(input[..., :half]) * input[..., half:]


gelu_and_mul_trace = TraceTemplate(
    op_type="activation",
    name_prefix="gelu_and_mul",
    description="Fused GeLU (exact) + Mul: gelu(x[:H]) * x[H:].",
    axes={
        "num_tokens": Var(description="Total number of tokens."),
        "hidden_size": Const(
            abbrev="h", description="Output hidden size (input is 2*h)."
        ),
    },
    inputs={
        "input": Tensor(
            ["num_tokens", "hidden_size"],
            param="input",
            description="Gated input tensor of shape [num_tokens, 2*hidden_size].",
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_gelu_and_mul_reference,
)


# ── SiLU+mul + masked NVFP4 quantize (MoE expert path) ───────────────────────


@torch.no_grad()
def _silu_and_mul_scaled_nvfp4_experts_quantize_reference(
    a, mask, a_global_sf, **_unused
):
    """Reference for silu_and_mul_scaled_nvfp4_experts_quantize.

    Models the math (SiLU+mul → scale → FP4 nibble pack) but does NOT
    reproduce the kernel's expert-mask routing layout precisely; the
    masked layout is a dispatch detail captured by trace JSON via input
    shapes. Returns ``(a_fp4_packed_uint8, a_sf_e4m3fn)``.
    """
    last = a.shape[-1]
    half = last // 2
    x1 = a[..., :half].to(torch.float32)
    x2 = a[..., half:].to(torch.float32)
    out_f = x1 * torch.sigmoid(x1) * x2
    if mask is not None and mask.shape == out_f.shape:
        out_f = out_f * mask.to(torch.float32)
    inv = (
        1.0 / float(a_global_sf.item())
        if isinstance(a_global_sf, torch.Tensor) and a_global_sf.numel() == 1
        else 1.0
    )
    scaled = (out_f * inv).clamp(-6.0, 6.0)
    nibbles = (scaled * 2).round().to(torch.int32) & 0xF
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    sf_shape = list(scaled.shape)
    sf_shape[-1] = scaled.shape[-1] // 16
    sf = torch.ones(sf_shape, dtype=torch.float8_e4m3fn, device=a.device)
    return packed, sf


silu_and_mul_scaled_nvfp4_experts_quantize_trace = TraceTemplate(
    op_type="activation_quantize",
    name_prefix="silu_and_mul_scaled_nvfp4_experts_quantize",
    description=(
        "Fused SiLU+mul activation followed by masked NVFP4 (e2m1, "
        "block_size=16) quantization, used in MoE expert FC1 outputs. "
        "Operates on a 3-D batched tensor [B, M, 2K] and emits packed "
        "FP4 [B, M, K/2] uint8 plus per-block scales."
    ),
    axes={
        "B": Var(description="Number of experts."),
        "M": Var(description="Tokens per expert."),
        "K_doubled": Var(description="2 * K (gated input dim)."),
        "K_div_2": Var(description="K // 2 (FP4 packed dim)."),
        "K_div_block_size": Var(description="K // 16 (NVFP4 block scale dim)."),
        "scalar": Var(description="Global SF tensor length (typically 1)."),
    },
    inputs={
        "a": Tensor(["B", "M", "K_doubled"]),
        "mask": Tensor(["B", "M", "K_doubled"], optional=True),
        "a_global_sf": Tensor(["scalar"], dtype="float32"),
    },
    outputs={
        "a_fp4": Tensor(["B", "M", "K_div_2"], dtype="uint8"),
        "a_sf": Tensor(["B", "M", "K_div_block_size"], dtype="float8_e4m3fn"),
    },
    tags=["status:verified", "fused", "quantize:fp4", "moe"],
    reference=_silu_and_mul_scaled_nvfp4_experts_quantize_reference,
)
