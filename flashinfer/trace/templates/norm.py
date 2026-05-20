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


def _rmsnorm_init(
    *,
    batch_size: int,
    hidden_size: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.rmsnorm``.

    Sourced from ``tests/utils/test_norm.py::test_norm`` (contiguous path):
    ``input ~ randn(batch_size, hidden_size)``, ``weight ~ randn(hidden_size)``.
    Default ``hidden_size=4096`` matches Llama-3.1-8B.
    """
    torch.manual_seed(seed)
    return {
        "input": torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        ),
        "weight": torch.randn(hidden_size, dtype=torch.bfloat16, device=device),
    }


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
    init=_rmsnorm_init,
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


def _fused_add_rmsnorm_init(
    *,
    batch_size: int,
    hidden_size: int = 5120,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.fused_add_rmsnorm``.

    Sourced from ``tests/utils/test_norm.py::test_fused_add_rmsnorm``.
    Default ``hidden_size=5120`` matches Qwen3-14B (per the example call).
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    return {"input": x, "residual": residual, "weight": weight}


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
    init=_fused_add_rmsnorm_init,
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


def _rmsnorm_quant_init(
    *,
    batch_size: int,
    hidden_size: int = 7168,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.rmsnorm_quant``.

    Sourced from ``tests/utils/test_norm.py::test_norm_quant``. Default
    ``hidden_size=7168`` matches DeepSeek-V3 down_proj.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {"out": out, "input": x, "weight": w, "scale": scale}


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
    init=_rmsnorm_quant_init,
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


def _fused_add_rmsnorm_quant_init(
    *,
    batch_size: int,
    hidden_size: int = 7168,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.fused_add_rmsnorm_quant``.

    Sourced from ``tests/utils/test_norm.py::test_fused_add_rmsnorm_quant``.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    w = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {
        "out": out,
        "input": x,
        "residual": residual,
        "weight": w,
        "scale": scale,
    }


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
    init=_fused_add_rmsnorm_quant_init,
)

# ── Gemma RMSNorm ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _gemma_rmsnorm_reference(input, weight):
    """Gemma-style RMSNorm: out = rmsnorm(input) * (weight + 1). Epsilon fixed at 1e-6."""
    EPS = 1e-6
    x = input.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * inv_rms * (weight.to(torch.float32) + 1)).to(input.dtype)


def _gemma_rmsnorm_init(
    *,
    batch_size: int,
    hidden_size: int = 4608,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gemma_rmsnorm``.

    Sourced from ``tests/utils/test_norm.py::test_gemma_norm``. Default
    ``hidden_size=4608`` matches Gemma-2-27B.
    """
    torch.manual_seed(seed)
    return {
        "input": torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        ),
        "weight": torch.randn(hidden_size, dtype=torch.bfloat16, device=device),
    }


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
    init=_gemma_rmsnorm_init,
)

# ── Gemma Fused Add + RMSNorm ─────────────────────────────────────────────────


@torch.no_grad()
def _gemma_fused_add_rmsnorm_reference(input, residual, weight):
    """Gemma-style Fused Add + RMSNorm."""
    EPS = 1e-6
    x = input.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * inv_rms * (weight.to(torch.float32) + 1)).to(input.dtype)


def _gemma_fused_add_rmsnorm_init(
    *,
    batch_size: int,
    hidden_size: int = 4608,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.gemma_fused_add_rmsnorm``.

    Sourced from ``tests/utils/test_norm.py::test_gemma_fused_add_rmsnorm``.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    return {"input": x, "residual": residual, "weight": weight}


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
    init=_gemma_fused_add_rmsnorm_init,
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


def _layernorm_init(
    *,
    batch_size: int,
    hidden_size: int = 768,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.layernorm``.

    Sourced from ``tests/utils/test_norm.py::test_layernorm``. Default
    ``hidden_size=768`` matches GPT-2/BERT base. ``gemma`` (gamma) and
    ``beta`` are float32 as required by the API.
    """
    torch.manual_seed(seed)
    return {
        "input": torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        ),
        "gemma": torch.randn(hidden_size, dtype=torch.float32, device=device),
        "beta": torch.randn(hidden_size, dtype=torch.float32, device=device),
    }


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
    init=_layernorm_init,
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


def _fused_rmsnorm_silu_init(
    *,
    num_tokens: int,
    hidden_size: int = 4096,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for the fused RMSNorm + SiLU kernel.

    Sourced from ``tests/norm/test_fused_rmsnorm_silu.py``: input is
    ``randn * 5 + 5`` (positive-shifted, larger variance — matches WAN
    VAE decoder activations); weight is ``rand * 1.5 + 0.5`` (positive,
    in [0.5, 2.0]). Default ``hidden_size=4096``.
    """
    torch.manual_seed(seed)
    inp = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 5.0
        + 5.0
    )
    weight = torch.rand(hidden_size, dtype=torch.bfloat16, device=device) * 1.5 + 0.5
    return {"input": inp, "weight": weight}


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
    init=_fused_rmsnorm_silu_init,
)


# ── CuteDSL RMSNorm + FP4 Quantize (rmsnorm_fp4quant / add_rmsnorm_fp4quant) ─


@torch.no_grad()
def _rmsnorm_fp4quant_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    y_fp4=None,
    block_scale=None,
    global_scale=None,
    eps: float = 1e-6,
    block_size: int = 16,
    **_unused,
):
    """Reference for cute_dsl.rmsnorm_fp4quant: RMSNorm * weight, optional
    global scaling, then per-block FP4 quantization.

    Returns ``(y_fp4_packed_bytes, block_scale)``. The FP4 packing follows
    the e2m1_x2 convention (low nibble = first element); block scales use
    the FP8 e4m3 absmax / 6 mapping. Modeled at the math level only —
    the kernel uses a bespoke shuffled scale layout that this reference
    does not reproduce; correctness can still be verified via the
    dequantized round-trip rather than packed-byte equality.
    """
    x = input.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    y = (x * inv_rms) * weight.to(torch.float32)
    if global_scale is not None:
        y = y * float(global_scale.item())
    # Block-quantize: per-block absmax / 6.0, then quantize values to nearest E2M1.
    M, K = y.shape
    y_blocks = y.reshape(M, K // block_size, block_size)
    sf = (y_blocks.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
    sf_f = sf.to(torch.float32).clamp(min=1e-12)
    quant = (y_blocks / sf_f.unsqueeze(-1)).clamp(-6.0, 6.0)
    nibbles = (quant * 2).round().to(torch.int32) & 0xF
    nibbles = nibbles.reshape(M, K)
    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    if y_fp4 is not None:
        y_fp4.copy_(packed)
    if block_scale is not None:
        block_scale.copy_(sf)
    return packed, sf


@torch.no_grad()
def _add_rmsnorm_fp4quant_reference(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_fp4=None,
    block_scale=None,
    global_scale=None,
    eps: float = 1e-6,
    block_size: int = 16,
    **_unused,
):
    """Reference for cute_dsl.add_rmsnorm_fp4quant: residual+input then
    RMSNorm+FP4 quantize. Mutates ``residual`` in-place to hold
    (input + residual) — matching the kernel's prenorm semantics."""
    pre = input.to(torch.float32) + residual.to(torch.float32)
    residual.copy_(pre.to(residual.dtype))
    return _rmsnorm_fp4quant_reference(
        pre,
        weight,
        y_fp4=y_fp4,
        block_scale=block_scale,
        global_scale=global_scale,
        eps=eps,
        block_size=block_size,
    )


_RMSNORM_FP4_AXES: dict[str, Var | Const] = {
    "num_tokens": Var(),
    "hidden_size": Const(abbrev="h"),
    "hidden_div_2": Var(description="hidden_size // 2 (FP4 packed dim)."),
    "hidden_div_block_size": Var(description="hidden_size // block_size."),
}


_RMSNORM_FP4_INPUTS: dict[str, Tensor | Scalar] = {
    "input": Tensor(["num_tokens", "hidden_size"]),
    "weight": Tensor(["hidden_size"]),
    "global_scale": Tensor(
        ["scalar"],
        dtype="float32",
        optional=True,
        description="Optional per-tensor pre-quantization scale.",
    ),
    "eps": Scalar("float32", optional=True),
    "block_size": Scalar("int32", optional=True),
}


_RMSNORM_FP4_OUTPUTS: dict[str, Tensor | Scalar] = {
    "y_fp4": Tensor(
        ["num_tokens", "hidden_div_2"],
        dtype="uint8",
        description="Packed FP4 e2m1_x2 output.",
    ),
    "block_scale": Tensor(
        ["num_tokens", "hidden_div_block_size"],
        dtype="float8_e4m3fn",
        description="Per-block absmax-derived scale.",
    ),
}


def _rmsnorm_fp4quant_init(
    *,
    num_tokens: int,
    hidden_size: int = 4096,
    hidden_div_2: int = 0,  # derived
    hidden_div_block_size: int = 0,  # derived
    scalar: int = 1,
    block_size: int = 16,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for the CuTe-DSL ``rmsnorm_fp4quant``.

    Sourced from the cute_dsl FP4 norm test (and matching the
    ``_rmsnorm_fp4quant_reference`` math):
    ``global_scale = 448 * 6 / amax(rmsnorm(input) * weight)``, computed
    from the actual normed output rather than a constant. ``y_fp4`` and
    ``block_scale`` are pre-allocated output buffers.
    """
    del hidden_div_2, hidden_div_block_size  # derived from hidden_size/block_size
    torch.manual_seed(seed)
    inp = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    # Compute global_scale from the RMSNorm output amax.
    eps = 1e-6
    x_f32 = inp.to(torch.float32)
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_f32 * inv_rms) * weight.to(torch.float32)
    amax = y.abs().nan_to_num().max().clamp(min=1e-12)
    global_scale = (448.0 * 6.0 / amax).reshape(1).contiguous()
    y_fp4 = torch.empty(num_tokens, hidden_size // 2, dtype=torch.uint8, device=device)
    block_scale = torch.empty(
        num_tokens,
        hidden_size // block_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    return {
        "input": inp,
        "weight": weight,
        "y_fp4": y_fp4,
        "block_scale": block_scale,
        "global_scale": global_scale,
        "block_size": block_size,
    }


rmsnorm_fp4quant_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="rmsnorm_fp4quant",
    description=(
        "CuTe-DSL fused RMSNorm + FP4 (e2m1) quantize. y = RMSNorm(input) "
        "* weight, optionally scaled by ``global_scale``, then "
        "block-quantized to FP4 with per-block FP8 e4m3 scales."
    ),
    axes=_RMSNORM_FP4_AXES,
    inputs=_RMSNORM_FP4_INPUTS,
    outputs=_RMSNORM_FP4_OUTPUTS,
    tags=["status:verified", "fused", "quantize:fp4"],
    reference=_rmsnorm_fp4quant_reference,
    init=_rmsnorm_fp4quant_init,
)
rmsnorm_fp4quant_trace.axes["scalar"] = Var(
    description="global_scale tensor length (typically 1).",
)


def _add_rmsnorm_fp4quant_init(
    *,
    num_tokens: int,
    hidden_size: int = 4096,
    hidden_div_2: int = 0,
    hidden_div_block_size: int = 0,
    scalar: int = 1,
    block_size: int = 16,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for the CuTe-DSL ``add_rmsnorm_fp4quant``.

    Same as ``rmsnorm_fp4quant`` plus a residual buffer (mutated in-place
    with ``input + residual`` by the kernel). ``global_scale`` is
    computed from amax of ``rmsnorm(input + residual) * weight``.
    """
    del hidden_div_2, hidden_div_block_size
    torch.manual_seed(seed)
    inp = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(inp)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
    # Compute global_scale from the post-add-rmsnorm output amax.
    eps = 1e-6
    pre = inp.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (pre * inv_rms) * weight.to(torch.float32)
    amax = y.abs().nan_to_num().max().clamp(min=1e-12)
    global_scale = (448.0 * 6.0 / amax).reshape(1).contiguous()
    y_fp4 = torch.empty(num_tokens, hidden_size // 2, dtype=torch.uint8, device=device)
    block_scale = torch.empty(
        num_tokens,
        hidden_size // block_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    return {
        "input": inp,
        "residual": residual,
        "weight": weight,
        "y_fp4": y_fp4,
        "block_scale": block_scale,
        "global_scale": global_scale,
        "block_size": block_size,
    }


add_rmsnorm_fp4quant_trace = TraceTemplate(
    op_type="rmsnorm",
    name_prefix="add_rmsnorm_fp4quant",
    description=(
        "CuTe-DSL fused (residual + input) → RMSNorm → FP4 quantize. "
        "Mutates the residual buffer in-place with the prenorm sum."
    ),
    axes=_RMSNORM_FP4_AXES,
    inputs={
        "input": Tensor(["num_tokens", "hidden_size"]),
        "residual": Tensor(
            ["num_tokens", "hidden_size"],
            description="Mutated in-place with input + residual.",
        ),
        "weight": Tensor(["hidden_size"]),
        "global_scale": Tensor(
            ["scalar"],
            dtype="float32",
            optional=True,
        ),
        "eps": Scalar("float32", optional=True),
        "block_size": Scalar("int32", optional=True),
    },
    outputs=_RMSNORM_FP4_OUTPUTS,
    tags=["status:verified", "fused", "quantize:fp4"],
    reference=_add_rmsnorm_fp4quant_reference,
    init=_add_rmsnorm_fp4quant_init,
)
add_rmsnorm_fp4quant_trace.axes["scalar"] = Var(
    description="global_scale tensor length (typically 1).",
)
