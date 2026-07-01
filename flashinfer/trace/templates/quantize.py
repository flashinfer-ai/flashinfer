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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_AxisT = Union[Var, Const]


def _bind_trace_init_dependencies(wrapper: Any, *dependencies: Any) -> Any:
    wrapper._trace_init_dependencies = dependencies
    return wrapper


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
    if global_scale is not None:
        scaled = (
            blocks
            * global_scale.to(torch.float32).reshape(())
            / actual_scale.unsqueeze(-1)
        )
    else:
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


def _fp4_quantize_init(
    *,
    M: int,
    K: int = 4096,
    K_packed: int = 0,
    num_scale_elems: int = 0,
    one: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.fp4_quantize`` (generic NvFP4 path).

    Sourced from ``tests/utils/test_fp4_quantize.py``: ``input`` is
    ``randn`` (bf16); ``global_scale`` is computed as
    ``FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax(input)`` per the test
    fixture. Default ``K=4096`` matches the example call.
    """
    del K_packed, num_scale_elems, one  # derived
    torch.manual_seed(seed)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    amax = inp.float().abs().nan_to_num().max().clamp(min=1e-12)
    global_scale = (448.0 * 6.0 / amax).reshape(1).contiguous()
    return {
        "input": inp,
        "global_scale": global_scale,
        "sf_vec_size": 16,
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
    init=_fp4_quantize_init,
)


def _nvfp4_quantize_init(
    *,
    M: int,
    K: int = 4096,
    K_packed: int = 0,
    num_scale_elems: int = 0,
    one: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.nvfp4_quantize``.

    Sourced from ``tests/utils/test_fp4_quantize.py`` /
    ``test_mm_fp4.py``: ``a_global_sf`` is computed from the input
    absmax as ``448 * 6 / amax(a)``.
    """
    del K_packed, num_scale_elems, one
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    amax = a.float().abs().nan_to_num().max().clamp(min=1e-12)
    return {
        "a": a,
        "a_global_sf": (448.0 * 6.0 / amax).reshape(1).contiguous(),
    }


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
    init=_nvfp4_quantize_init,
)


def _mxfp4_quantize_init(
    *,
    M: int,
    K: int = 4096,
    K_packed: int = 0,
    num_scale_elems: int = 0,
    one: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mxfp4_quantize`` (no global scale)."""
    del K_packed, num_scale_elems, one
    torch.manual_seed(seed)
    return {"a": torch.randn(M, K, dtype=torch.bfloat16, device=device)}


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
    init=_mxfp4_quantize_init,
)


def _mxfp8_quantize_init(
    *,
    M: int,
    K: int = 4096,
    num_scale_elems: int = 0,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.mxfp8_quantize``."""
    del num_scale_elems
    torch.manual_seed(seed)
    return {"input": torch.randn(M, K, dtype=torch.bfloat16, device=device)}


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
    init=_mxfp8_quantize_init,
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


def _nvfp4_kv_quantize_init(
    *,
    M: int,
    K: int = 128,
    K_div_2: int = 0,
    K_div_16: int = 0,
    scalar: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.nvfp4_kv_quantize``.

    Default ``K=128`` matches a typical KV head dim. ``global_scale`` is
    computed from input absmax (``448 * 6 / amax``) — same principled
    formula used by ``fp4_quantize`` / ``nvfp4_quantize``.

    Note: ``tests/utils/test_fp4_kv_quantization.py`` parametrizes over
    multiple ``global_scale`` values (0.5, 1.0) — the 1.0 case
    specifically guards against FP8 E4M3 block-scale underflow at very
    small KV head_dim. The amax-derived scale here matches the
    activation/weight quantize pipeline; callers who want the underflow
    guard can override with ``global_scale=torch.tensor([1.0])`` after
    calling ``init``.
    """
    del K_div_2, K_div_16, scalar
    torch.manual_seed(seed)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    amax = inp.float().abs().nan_to_num().max().clamp(min=1e-12)
    return {
        "input": inp,
        "global_scale": (448.0 * 6.0 / amax).reshape(1).contiguous(),
    }


def _nvfp4_kv_dequantize_paged_init(
    *,
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    k_head_dim: int = 128,
    v_head_dim: int = 128,
    k_packed_dim: int = 0,
    v_packed_dim: int = 0,
    k_scale_dim: int = 0,
    v_scale_dim: int = 0,
    num_pages: int = 0,
    page_size: int = 16,
    block_table_stride: int = 0,
    scalar: int = 1,
    kv_layout: str = "NHD",
    device: str = "cuda",
    seed: int = 0,
):
    """Build tuple-cache inputs for ``flashinfer.nvfp4_kv_dequantize_paged``."""
    del k_packed_dim, v_packed_dim, k_scale_dim, v_scale_dim, scalar
    torch.manual_seed(seed)
    k_packed = k_head_dim // 2
    v_packed = v_head_dim // 2
    k_scales_dim = k_head_dim // 16
    v_scales_dim = v_head_dim // 16
    pages_per_request = (max_seq_len + page_size - 1) // page_size
    if block_table_stride == 0:
        block_table_stride = pages_per_request
    if num_pages == 0:
        num_pages = batch_size * block_table_stride
    if kv_layout == "NHD":
        k_cache_shape = (num_pages, page_size, num_heads, k_packed)
        v_cache_shape = (num_pages, page_size, num_heads, v_packed)
        k_scales_shape = (num_pages, page_size, num_heads, k_scales_dim)
        v_scales_shape = (num_pages, page_size, num_heads, v_scales_dim)
    elif kv_layout == "HND":
        k_cache_shape = (num_pages, num_heads, page_size, k_packed)
        v_cache_shape = (num_pages, num_heads, page_size, v_packed)
        k_scales_shape = (num_pages, num_heads, page_size, k_scales_dim)
        v_scales_shape = (num_pages, num_heads, page_size, v_scales_dim)
    else:
        raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")

    k_cache = torch.randint(
        0,
        256,
        k_cache_shape,
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.randint(
        0,
        256,
        v_cache_shape,
        dtype=torch.uint8,
        device=device,
    )
    k_scales = torch.randint(
        1,
        120,
        k_scales_shape,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    v_scales = torch.randint(
        1,
        120,
        v_scales_shape,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    block_tables = (
        torch.arange(batch_size * block_table_stride, dtype=torch.int32, device=device)
        .reshape(batch_size, block_table_stride)
        .remainder(num_pages)
    )
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    output_k = torch.empty(
        (batch_size, max_seq_len, num_heads, k_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    output_v = torch.empty(
        (batch_size, max_seq_len, num_heads, v_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "paged_kv_cache": (k_cache, v_cache),
        "kv_cache_sf": (k_scales, v_scales),
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "k_scale": torch.tensor([1.0], dtype=torch.float32, device=device),
        "v_scale": torch.tensor([1.0], dtype=torch.float32, device=device),
        "output_k": output_k,
        "output_v": output_v,
        "kv_layout": kv_layout,
    }


def _nvfp4_kv_dequantize_paged_nhd_init(
    *,
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    k_head_dim: int = 128,
    v_head_dim: int = 128,
    k_packed_dim: int = 0,
    v_packed_dim: int = 0,
    k_scale_dim: int = 0,
    v_scale_dim: int = 0,
    num_pages: int = 0,
    page_size: int = 16,
    block_table_stride: int = 0,
    scalar: int = 1,
    kv_layout: str = "NHD",
    device: str = "cuda",
    seed: int = 0,
):
    del kv_layout
    return _nvfp4_kv_dequantize_paged_init(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        k_packed_dim=k_packed_dim,
        v_packed_dim=v_packed_dim,
        k_scale_dim=k_scale_dim,
        v_scale_dim=v_scale_dim,
        num_pages=num_pages,
        page_size=page_size,
        block_table_stride=block_table_stride,
        scalar=scalar,
        kv_layout="NHD",
        device=device,
        seed=seed,
    )


def _nvfp4_kv_dequantize_paged_hnd_init(
    *,
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    k_head_dim: int = 128,
    v_head_dim: int = 128,
    k_packed_dim: int = 0,
    v_packed_dim: int = 0,
    k_scale_dim: int = 0,
    v_scale_dim: int = 0,
    num_pages: int = 0,
    page_size: int = 16,
    block_table_stride: int = 0,
    scalar: int = 1,
    kv_layout: str = "HND",
    device: str = "cuda",
    seed: int = 0,
):
    del kv_layout
    return _nvfp4_kv_dequantize_paged_init(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        k_packed_dim=k_packed_dim,
        v_packed_dim=v_packed_dim,
        k_scale_dim=k_scale_dim,
        v_scale_dim=v_scale_dim,
        num_pages=num_pages,
        page_size=page_size,
        block_table_stride=block_table_stride,
        scalar=scalar,
        kv_layout="HND",
        device=device,
        seed=seed,
    )


_nvfp4_kv_dequantize_paged_nhd_init = _bind_trace_init_dependencies(
    _nvfp4_kv_dequantize_paged_nhd_init,
    _nvfp4_kv_dequantize_paged_init,
)
_nvfp4_kv_dequantize_paged_hnd_init = _bind_trace_init_dependencies(
    _nvfp4_kv_dequantize_paged_hnd_init,
    _nvfp4_kv_dequantize_paged_init,
)


def _make_nvfp4_kv_dequantize_paged_trace(
    *, kv_layout: str, name_prefix: str
) -> TraceTemplate:
    if kv_layout == "NHD":
        k_cache_dims = ["num_pages", "page_size", "num_heads", "k_packed_dim"]
        v_cache_dims = ["num_pages", "page_size", "num_heads", "v_packed_dim"]
        k_scales_dims = ["num_pages", "page_size", "num_heads", "k_scale_dim"]
        v_scales_dims = ["num_pages", "page_size", "num_heads", "v_scale_dim"]
    elif kv_layout == "HND":
        k_cache_dims = ["num_pages", "num_heads", "page_size", "k_packed_dim"]
        v_cache_dims = ["num_pages", "num_heads", "page_size", "v_packed_dim"]
        k_scales_dims = ["num_pages", "num_heads", "page_size", "k_scale_dim"]
        v_scales_dims = ["num_pages", "num_heads", "page_size", "v_scale_dim"]
    else:
        raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")

    init = (
        _nvfp4_kv_dequantize_paged_nhd_init
        if kv_layout == "NHD"
        else _nvfp4_kv_dequantize_paged_hnd_init
    )

    return TraceTemplate(
        op_type="dequantize_fp4",
        name_prefix=name_prefix,
        description=(
            "Gather and dequantize a paged NVFP4 KV cache through block tables "
            "into caller-owned contiguous K/V output buffers."
        ),
        axes={
            "batch_size": Var(),
            "max_seq_len": Var(),
            "num_heads": Const(abbrev="h"),
            "k_head_dim": Const(abbrev="dk"),
            "v_head_dim": Const(abbrev="dv"),
            "k_packed_dim": Var(description="k_head_dim // 2."),
            "v_packed_dim": Var(description="v_head_dim // 2."),
            "k_scale_dim": Var(description="k_head_dim // 16."),
            "v_scale_dim": Var(description="v_head_dim // 16."),
            "num_pages": Var(),
            "page_size": Const(abbrev="ps"),
            "block_table_stride": Var(),
            "scalar": Var(description="Global scale tensor length, normally 1."),
        },
        inputs={
            "paged_k_cache": Tensor(
                k_cache_dims,
                param="paged_kv_cache",
                tuple_idx=0,
            ),
            "paged_v_cache": Tensor(
                v_cache_dims,
                param="paged_kv_cache",
                tuple_idx=1,
            ),
            "k_scales": Tensor(
                k_scales_dims,
                param="kv_cache_sf",
                tuple_idx=0,
            ),
            "v_scales": Tensor(
                v_scales_dims,
                param="kv_cache_sf",
                tuple_idx=1,
            ),
            "block_tables": Tensor(["batch_size", "block_table_stride"], dtype="int32"),
            "seq_lens": Tensor(["batch_size"], dtype="int32"),
            "k_scale": Tensor(["scalar"], dtype="float32"),
            "v_scale": Tensor(["scalar"], dtype="float32"),
            "output_k": Tensor(
                ["batch_size", "max_seq_len", "num_heads", "k_head_dim"]
            ),
            "output_v": Tensor(
                ["batch_size", "max_seq_len", "num_heads", "v_head_dim"]
            ),
        },
        outputs={
            "output_k": Tensor(
                ["batch_size", "max_seq_len", "num_heads", "k_head_dim"],
                dtype_from="output_k",
            ),
            "output_v": Tensor(
                ["batch_size", "max_seq_len", "num_heads", "v_head_dim"],
                dtype_from="output_v",
            ),
        },
        constraints=[
            "k_head_dim == k_packed_dim * 2",
            "v_head_dim == v_packed_dim * 2",
            "k_head_dim == k_scale_dim * 16",
            "v_head_dim == v_scale_dim * 16",
            "block_table_stride * page_size >= max_seq_len",
            "scalar == 1",
        ],
        tags=["status:verified", "quantization:fp4"],
        init=init,
    )


_nvfp4_kv_dequantize_paged_nhd_trace = _make_nvfp4_kv_dequantize_paged_trace(
    kv_layout="NHD", name_prefix="nvfp4_kv_dequantize_paged"
)
_nvfp4_kv_dequantize_paged_hnd_trace = _make_nvfp4_kv_dequantize_paged_trace(
    kv_layout="HND", name_prefix="nvfp4_kv_dequantize_paged_hnd"
)


def nvfp4_kv_dequantize_paged_trace(**kwargs):
    """Return a layout-specific trace template for paged NVFP4 KV dequant."""
    kv_layout = kwargs.get("kv_layout", "NHD")
    if kv_layout == "NHD":
        return _nvfp4_kv_dequantize_paged_nhd_trace
    if kv_layout == "HND":
        return _nvfp4_kv_dequantize_paged_hnd_trace
    raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")


nvfp4_kv_dequantize_paged_trace.templates = [  # type: ignore[attr-defined]
    _nvfp4_kv_dequantize_paged_nhd_trace,
    _nvfp4_kv_dequantize_paged_hnd_trace,
]


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
    init=_nvfp4_kv_quantize_init,
)


# ── Grouped MXFP8 quantization (cuTile, SM100+) ──────────────────────────────


def _mxfp8_grouped_quantize_init(
    *,
    B,
    M,
    K=4096,
    padded_K=0,
    rm=0,
    rk=0,
    m32=32,
    m4=4,
    k4=4,
    device="cuda",
    seed=0,
):
    """Build inputs for ``flashinfer.mxfp8_grouped_quantize``.

    Every group uses the full ``M`` valid rows (``mask = M``) so the trace
    exercises the dense path with no uninitialized output rows; callers may
    shrink ``mask`` afterwards.
    """
    del padded_K, rm, rk, m32, m4, k4  # output-only / derived axes
    torch.manual_seed(seed)
    a = torch.randn(B, M, K, dtype=torch.bfloat16, device=device)
    mask = torch.full((B,), M, dtype=torch.int32, device=device)
    return {"a": a, "mask": mask}


mxfp8_grouped_quantize_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="mxfp8_grouped_quantize",
    description=(
        "Grouped MXFP8 quantization (cuTile, SM100+): [B, M, K] bf16/fp16 -> "
        "fp8_e4m3fn activations + UE8M0 block scales, laid out for the masked "
        "grouped GEMM. K is padded up to a multiple of 128 for the kernel."
    ),
    axes={
        "B": Var(description="Number of groups."),
        "M": Var(description="Rows per group."),
        "K": Const(abbrev="k", description="Input columns (divisible by 32)."),
        "padded_K": Var(description="K rounded up to a multiple of 128."),
        "rm": Var(description="padded_M // 128 (row scale tiles)."),
        "rk": Var(description="padded_K // 128 (column scale tiles)."),
        "m32": Var(description="MXFP8 scale swizzle dim (32)."),
        "m4": Var(description="MXFP8 scale swizzle dim (4)."),
        "k4": Var(description="MXFP8 scale swizzle dim (4)."),
    },
    inputs={
        "a": Tensor(["B", "M", "K"], description="Input tensor, fp16/bf16."),
        "mask": Tensor(
            ["B"],
            dtype="int32",
            description="Valid rows per group (int32).",
        ),
    },
    outputs={
        "x_q": Tensor(
            ["M", "padded_K", "B"],
            dtype="float8_e4m3fn",
            description="Quantized activations, permuted for the grouped GEMM.",
        ),
        "sf": Tensor(
            ["m32", "m4", "rm", "k4", "rk", "B"],
            dtype="uint8",
            description="UE8M0 swizzled block scales (1 byte per 32-element block).",
        ),
    },
    constraints=[
        "padded_K == ((K + 127) // 128) * 128",
        "rm == (M + 127) // 128",
        "rk == padded_K // 128",
        "m32 == 32",
        "m4 == 4",
        "k4 == 4",
    ],
    tags=["status:verified", "quantization:mxfp8"],
    init=_mxfp8_grouped_quantize_init,
)
