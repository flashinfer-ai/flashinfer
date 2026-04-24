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

"""TraceTemplates for normalization operations."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

# ── RMSNorm ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def _rmsnorm_reference(hidden_states, weight):
    """Root Mean Square Normalization. Epsilon is fixed at 1e-6."""
    EPS = 1e-6
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="rmsnorm",
    description="Root Mean Square Normalization. Epsilon is fixed at 1e-6.",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified"],
    reference=_rmsnorm_reference,
)

# ── Fused Add + RMSNorm ───────────────────────────────────────────────────────


@torch.no_grad()
def _fused_add_rmsnorm_reference(hidden_states, residual, weight):
    """Fused Add + RMSNorm. Epsilon is fixed at 1e-6."""
    EPS = 1e-6
    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


fused_add_rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="fused_add_rmsnorm",
    description="Fused Add + RMSNorm. Epsilon is fixed at 1e-6.",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "residual": Tensor(["batch_size", "hidden_size"]),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
        "residual": Tensor(
            ["batch_size", "hidden_size"],
            dtype_from="input",
            description="Updated residual (in-place: residual += hidden_states).",
        ),
    },
    tags=["status:verified", "fused"],
    reference=_fused_add_rmsnorm_reference,
)

# ── RMSNorm + FP8 Quantize ────────────────────────────────────────────────────


@torch.no_grad()
def _rmsnorm_quant_reference(hidden_states, weight, scale):
    """RMSNorm followed by per-tensor FP8 (e4m3fn) quantization.

    ``out = clamp(rmsnorm(input, weight) / scale, fp8_min, fp8_max).to(fp8_e4m3fn)``.
    Epsilon is fixed at 1e-6.
    """
    EPS = 1e-6
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    s = (
        scale.to(torch.float32).reshape(())
        if isinstance(scale, torch.Tensor)
        else float(scale)
    )
    y = y / s
    fp8_max = 448.0  # float8_e4m3fn max finite value
    y = y.clamp(-fp8_max, fp8_max)
    return y.to(torch.float8_e4m3fn)


rmsnorm_quant_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="rmsnorm_quant",
    description="RMSNorm + FP8 quantization. out = quantize(rmsnorm(input, weight), scale).",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "weight": Tensor(["hidden_size"]),
        "scale": Scalar(
            "float32", description="Per-tensor quantization scale, shape (1,)."
        ),
    },
    outputs={
        "out": Tensor(
            ["batch_size", "hidden_size"],
            description="Quantized output (dtype matches pre-allocated out tensor).",
        ),
    },
    tags=["status:verified", "quantization:fp8"],
    reference=_rmsnorm_quant_reference,
)

# ── Fused Add + RMSNorm + FP8 Quantize ───────────────────────────────────────


@torch.no_grad()
def _fused_add_rmsnorm_quant_reference(hidden_states, residual, weight, scale):
    """Fused Add + RMSNorm + FP8 quantize.

    ``residual' = hidden_states + residual``
    ``out = quantize(rmsnorm(residual', weight), scale)``
    Returns ``(out, residual')``.
    """
    EPS = 1e-6
    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    s = (
        scale.to(torch.float32).reshape(())
        if isinstance(scale, torch.Tensor)
        else float(scale)
    )
    y = y / s
    fp8_max = 448.0
    y = y.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return y, x.to(hidden_states.dtype)


fused_add_rmsnorm_quant_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="fused_add_rmsnorm_quant",
    description=(
        "Fused Add + RMSNorm + FP8 quantization. "
        "residual += input; out = quantize(rmsnorm(residual, weight), scale)."
    ),
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "residual": Tensor(["batch_size", "hidden_size"]),
        "weight": Tensor(["hidden_size"]),
        "scale": Scalar(
            "float32", description="Per-tensor quantization scale, shape (1,)."
        ),
    },
    outputs={
        "out": Tensor(
            ["batch_size", "hidden_size"],
            description="Quantized output (dtype matches pre-allocated out tensor).",
        ),
        "residual": Tensor(
            ["batch_size", "hidden_size"],
            dtype_from="input",
            description="Updated residual (in-place: residual += input).",
        ),
    },
    tags=["status:verified", "fused", "quantization:fp8"],
    reference=_fused_add_rmsnorm_quant_reference,
)

# ── Gemma RMSNorm ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _gemma_rmsnorm_reference(input, weight):
    """Gemma-style RMSNorm: out = rmsnorm(input) * (weight + 1). Epsilon fixed at 1e-6."""
    EPS = 1e-6
    x = input.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * inv_rms * (weight.to(torch.float32) + 1)).to(input.dtype)


gemma_rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="gemma_rmsnorm",
    description="Gemma-style RMSNorm: out = rmsnorm(x) * (weight + 1).",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "model:gemma"],
    reference=_gemma_rmsnorm_reference,
)

# ── Gemma Fused Add + RMSNorm ─────────────────────────────────────────────────


@torch.no_grad()
def _gemma_fused_add_rmsnorm_reference(input, residual, weight):
    """Gemma-style Fused Add + RMSNorm."""
    EPS = 1e-6
    x = input.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * inv_rms * (weight.to(torch.float32) + 1)).to(input.dtype)


gemma_fused_add_rmsnorm_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="gemma_fused_add_rmsnorm",
    description="Gemma-style Fused Add + RMSNorm: residual += input; out = gemma_rmsnorm(residual).",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "residual": Tensor(["batch_size", "hidden_size"]),
        "weight": Tensor(["hidden_size"]),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
        "residual": Tensor(
            ["batch_size", "hidden_size"],
            dtype_from="input",
            description="Updated residual (in-place: residual += input).",
        ),
    },
    tags=["status:verified", "fused", "model:gemma"],
    reference=_gemma_fused_add_rmsnorm_reference,
)

# ── LayerNorm ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _layernorm_reference(input, weight, bias):
    """Standard LayerNorm with gamma (weight) and beta (bias). Epsilon fixed at 1e-6."""
    EPS = 1e-6
    x = input.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + EPS)
    return (x_norm * weight.to(torch.float32) + bias.to(torch.float32)).to(input.dtype)


layernorm_trace = TraceTemplate(
    op_type="layernorm",
    name_prefix="layernorm",
    description="Standard LayerNorm with gamma and beta. Epsilon fixed at 1e-6.",
    axes={
        "batch_size": Var(),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "hidden_states": Tensor(["batch_size", "hidden_size"], param="input"),
        "weight": Tensor(
            ["hidden_size"], param="gemma", description="Scale (gamma) tensor, float32."
        ),
        "bias": Tensor(
            ["hidden_size"], param="beta", description="Bias (beta) tensor, float32."
        ),
    },
    outputs={
        "output": Tensor(["batch_size", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified"],
    reference=_layernorm_reference,
)


# ── Fused RMSNorm + SiLU ──────────────────────────────────────────────────────


@torch.no_grad()
def _fused_rmsnorm_silu_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    **_unused,
) -> torch.Tensor:
    """Fused RMSNorm followed by SiLU. ``out = SiLU(RMSNorm(input))``."""
    x = input.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    normed = (x * inv_rms) * weight.to(torch.float32)
    silu = normed * torch.sigmoid(normed)
    return silu.to(input.dtype)


fused_rmsnorm_silu_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="fused_rmsnorm_silu",
    description=(
        "Fused RMSNorm + SiLU activation: out = SiLU(RMSNorm(input, weight)). "
        "Optimized for SM100 WAN VAE decoder shapes."
    ),
    axes={
        "num_tokens": Var(description="Number of tokens (rows)."),
        "hidden_size": Const(abbrev="h"),
    },
    inputs={
        "input": Tensor(["num_tokens", "hidden_size"]),
        "weight": Tensor(["hidden_size"]),
        "eps": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(["num_tokens", "hidden_size"], dtype_from="input"),
    },
    tags=["status:verified", "fused"],
    reference=_fused_rmsnorm_silu_reference,
)
