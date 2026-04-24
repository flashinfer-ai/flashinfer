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

"""TraceTemplates for FP4 / FP8 quantization APIs."""

from typing import Dict, Optional, Tuple, Union

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_AxisT = Union[Var, Const]


# ── Reference helpers ────────────────────────────────────────────────────────

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # FP4 e2m1fn magnitudes


@torch.no_grad()
def _fp4_e2m1_quantize_block(
    block: torch.Tensor, amax_per_block: torch.Tensor
) -> torch.Tensor:
    """Round a float block to the nearest FP4 e2m1fn value and pack sign/magnitude.

    Returns an int64 tensor with values in [0, 15] matching the nibble codes
    used by ``_unpack_fp4_e2m1`` in moe.py: low 3 bits = magnitude index,
    high bit = sign.
    """
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=block.device)
    sign_bit = (block < 0).to(torch.int64) << 3
    mag = block.abs()
    # Nearest-magnitude index among the 8 e2m1 values.
    diffs = (mag.unsqueeze(-1) - values).abs()
    idx = diffs.argmin(dim=-1)
    return (idx | sign_bit) & 0x0F


@torch.no_grad()
def _pack_fp4_pairs(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit codes along the last axis into uint8 bytes.

    Low nibble = first element (matches _unpack_fp4_e2m1).
    """
    assert nibbles.shape[-1] % 2 == 0
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed


@torch.no_grad()
def _quantize_fp4_block_scale(
    input_tensor: torch.Tensor,
    block_size: int,
    use_ue8m0: bool,
    global_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference FP4 block-scale quantization.

    Returns ``(packed_uint8, scales)`` where ``scales`` has dtype
    ``float8_e4m3fn`` when ``use_ue8m0`` is False (NvFP4) and ``uint8``
    (UE8M0) otherwise (MXFP4).
    """
    M, K = input_tensor.shape
    assert K % block_size == 0
    x = input_tensor.to(torch.float32)
    blocks = x.reshape(M, K // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)  # [M, K/bs]
    # Per-block scale that maps amax to FP4 max magnitude (6.0).
    block_scale = amax / 6.0
    # Optional global scale factor applied before block scaling (NvFP4 path).
    if global_scale is not None:
        gs = global_scale.to(torch.float32).reshape(())
        block_scale = block_scale * gs
    if use_ue8m0:
        # Round scale to the nearest power of two and encode as UE8M0 (uint8).
        safe = torch.where(block_scale > 0, block_scale, torch.ones_like(block_scale))
        exp = torch.floor(torch.log2(safe)).to(torch.int64)
        exp = exp.clamp(-127, 128) + 127
        scales_raw = exp.to(torch.uint8)
        # Reconstruct the actual scale we quantized with for the packed values.
        actual_scale = torch.pow(
            torch.tensor(2.0, device=x.device), (exp - 127).to(torch.float32)
        )
    else:
        scales_raw = block_scale.to(torch.float8_e4m3fn)
        actual_scale = scales_raw.to(torch.float32)
    # Avoid division by zero for all-zero blocks.
    actual_scale = torch.where(
        actual_scale > 0,
        actual_scale,
        torch.ones_like(actual_scale),
    )
    # Broadcast block scale back to element granularity and quantize.
    scaled = blocks / actual_scale.unsqueeze(-1)
    nibbles = _fp4_e2m1_quantize_block(scaled, amax)
    nibbles = nibbles.reshape(M, K)
    packed = _pack_fp4_pairs(nibbles)
    return packed, scales_raw


@torch.no_grad()
def _quantize_mxfp8(
    input_tensor: torch.Tensor, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantization: fp8_e4m3fn values with UE8M0 per-32 scales."""
    M, K = input_tensor.shape
    assert K % block_size == 0
    x = input_tensor.to(torch.float32)
    blocks = x.reshape(M, K // block_size, block_size)
    amax = blocks.abs().amax(dim=-1)
    # fp8_e4m3fn max finite value is 448.0.
    block_scale = amax / 448.0
    safe = torch.where(block_scale > 0, block_scale, torch.ones_like(block_scale))
    exp = torch.floor(torch.log2(safe)).to(torch.int64)
    exp = exp.clamp(-127, 128) + 127
    scales_raw = exp.to(torch.uint8)
    actual_scale = torch.pow(
        torch.tensor(2.0, device=x.device), (exp - 127).to(torch.float32)
    )
    actual_scale = torch.where(
        actual_scale > 0, actual_scale, torch.ones_like(actual_scale)
    )
    scaled = blocks / actual_scale.unsqueeze(-1)
    quantized = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(M, K)
    return quantized, scales_raw


@torch.no_grad()
def _fp4_quantize_reference(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
    enable_pdl: Optional[bool] = None,
    backend: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference FP4 quantize. Produces packed uint8 + scales in LINEAR layout.

    The runtime API may return scales in a swizzled layout; consumers should
    dequantize before comparing.
    """
    packed, scales = _quantize_fp4_block_scale(
        input.reshape(-1, input.shape[-1]),
        block_size=int(sf_vec_size),
        use_ue8m0=bool(sf_use_ue8m0),
        global_scale=global_scale,
    )
    packed = packed.reshape(*input.shape[:-1], input.shape[-1] // 2)
    scales = scales.reshape(*input.shape[:-1], input.shape[-1] // int(sf_vec_size))
    return packed, scales


@torch.no_grad()
def _nvfp4_quantize_reference(
    a: torch.Tensor,
    a_global_sf: torch.Tensor,
    sfLayout=None,
    do_shuffle: bool = False,
    sf_vec_size: int = 16,
    enable_pdl: Optional[bool] = None,
    backend: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference NvFP4 quantize (block_size=16, fp8_e4m3fn scales)."""
    return _fp4_quantize_reference(
        a,
        global_scale=a_global_sf,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=False,
    )


@torch.no_grad()
def _mxfp4_quantize_reference(
    a: torch.Tensor,
    backend: str = "cuda",
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP4 quantize (block_size=32, UE8M0 scales)."""
    return _fp4_quantize_reference(
        a,
        global_scale=None,
        sf_vec_size=32,
        sf_use_ue8m0=True,
    )


@torch.no_grad()
def _mxfp8_quantize_reference(
    input: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
    backend: str = "cuda",
    sf_swizzle_layout=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantize (block_size=32, UE8M0 scales)."""
    return _quantize_mxfp8(
        input.reshape(-1, input.shape[-1]),
        block_size=int(alignment),
    )


# ── FP4 quantization (generic) ───────────────────────────────────────────────
# input [M, K]  →  (quantized [M, K/2] uint8 packed,  scales [variable])

_FP4_AXES: Dict[str, _AxisT] = {
    "M": Var(description="Number of rows."),
    "K": Const(abbrev="k", description="Number of input columns."),
    "K_packed": Var(
        description="Packed column dimension (K/2 for FP4, two values per uint8).",
    ),
    "num_scale_elems": Var(
        description="Total number of scale factor elements (layout-dependent)."
    ),
    "one": Var(description="Placeholder for shape [1] scalar tensors."),
}

fp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="fp4_quantize",
    description="Generic FP4 quantization: bf16/fp16 input → packed FP4 e2m1fn + block scales.",
    axes=_FP4_AXES,
    inputs={
        "input": Tensor(
            ["M", "K"],
            param="input",
            description="Input tensor, fp16/bf16/fp8_e4m3fn.",
        ),
        "global_scale": Tensor(
            ["one"],
            dtype="float32",
            optional=True,
            description="Optional per-tensor global scale (shape [1]).",
        ),
        "sf_vec_size": Scalar(
            "int32",
            optional=True,
            description="Scale-factor vector size (16 for NVFP4, 32 for MXFP4).",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output (two e2m1fn values per byte).",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="Block scale factors packed as uint8 bytes (layout-dependent shape).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:fp4"],
    reference=_fp4_quantize_reference,
)

# ── NVFP4 quantization ────────────────────────────────────────────────────────
nvfp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="nvfp4_quantize",
    description="NVFP4 quantization (sf_vec_size=16). Requires a per-tensor global scale.",
    axes=_FP4_AXES,
    inputs={
        "a": Tensor(["M", "K"], description="Input tensor, fp16/bf16/fp8_e4m3fn."),
        "a_global_sf": Tensor(
            ["one"],
            dtype="float32",
            description="Global scale factor, shape [1].",
        ),
        "sf_vec_size": Scalar(
            "int32",
            optional=True,
            description="Scale-factor vector size (fixed at 16 for NVFP4).",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="Block scale factors packed as uint8 bytes (layout-dependent shape).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:nvfp4"],
    reference=_nvfp4_quantize_reference,
)

# ── MXFP4 quantization ────────────────────────────────────────────────────────
mxfp4_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="mxfp4_quantize",
    description="MXFP4 quantization (sf_vec_size=32, UE8M0 scales). No global scale.",
    axes=_FP4_AXES,
    inputs={
        "a": Tensor(["M", "K"], description="Input tensor, fp16/bf16."),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K_packed"],
            dtype="uint8",
            description="Packed FP4 output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="UE8M0 block scale factors (1 byte per 32-element block).",
        ),
    },
    constraints=["K_packed == K // 2"],
    tags=["status:verified", "quantization:mxfp4"],
    reference=_mxfp4_quantize_reference,
)

# ── MXFP8 quantization ────────────────────────────────────────────────────────

mxfp8_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="mxfp8_quantize",
    description="MXFP8 quantization (block size 32, UE8M0 scales). Output is fp8_e4m3fn.",
    axes={
        "M": Var(description="Number of rows."),
        "K": Const(abbrev="k", description="Number of input columns."),
        "num_scale_elems": Var(
            description="Total number of scale factor elements (layout-dependent)."
        ),
    },
    inputs={
        "input": Tensor(
            ["M", "K"],
            param="input",
            description="Input tensor, fp16/bf16.",
        ),
    },
    outputs={
        "quantized": Tensor(
            ["M", "K"],
            dtype="float8_e4m3fn",
            description="MXFP8 quantized output.",
        ),
        "scales": Tensor(
            ["num_scale_elems"],
            dtype="uint8",
            description="UE8M0 block scale factors (1 byte per 32-element block).",
        ),
    },
    tags=["status:verified", "quantization:mxfp8"],
    reference=_mxfp8_quantize_reference,
)


# ── NVFP4 KV-cache quantize (linear block-scale layout) ──────────────────────


@torch.no_grad()
def _nvfp4_kv_quantize_reference(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for nvfp4_kv_quantize. NVFP4 (block_size=16) quantize
    with linear (un-swizzled) scale layout.
    """
    return _fp4_quantize_reference(
        input,
        global_scale=global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )


nvfp4_kv_quantize_trace = TraceTemplate(
    op_type="quantize_fp4",
    name_prefix="nvfp4_kv_quantize",
    description=(
        "NVFP4 (block_size=16) quantization for KV cache with linear "
        "block-scale layout. Requires SM100+ for the "
        "cvt.rn.satfinite.e2m1x2.f32 PTX instruction."
    ),
    axes={
        "M": Var(),
        "K": Const(abbrev="k"),
        "K_div_2": Var(description="K // 2 (FP4 packed dim)."),
        "K_div_16": Var(description="K // 16 (NVFP4 block scale dim)."),
        "scalar": Var(description="global_scale tensor length (typically 1)."),
    },
    inputs={
        "input": Tensor(["M", "K"]),
        "global_scale": Tensor(["scalar"], dtype="float32"),
    },
    outputs={
        "x_q": Tensor(["M", "K_div_2"], dtype="uint8"),
        "sf": Tensor(["M", "K_div_16"], dtype="float8_e4m3fn"),
    },
    tags=["status:verified", "quantization:fp4"],
    reference=_nvfp4_kv_quantize_reference,
)
