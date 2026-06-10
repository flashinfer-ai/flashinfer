"""
MXFP8 (FP8 E4M3 + UE8M0 per-row) layout helpers for cute sm120 GEMM entries.

These helpers produce input tensors compatible with the per-row INT32-packed
TMA-aligned MN-major scale layout. The current SM120 cute backend only exposes
`flashinfer.grouped_mm.grouped_mm_mxfp8_nt_groupwise_zero_padding` (MoE
zero-padding), which provides its own dedicated CUDA quantization helper
`quantize_mxfp8_for_zero_padding` — its input/output layout differs from the
per-row helpers below. These per-row helpers (per-token/per-block quantize,
transform, dequantize) are retained as building blocks for upstream tests and
future SM120 GEMM entries (dense / batched / masked / contiguous) that share
the per-row UE8M0 layout.

Distinct from the existing `flashinfer.quantization.mxfp8_quantize` (TRT-LLM-style
sfVecSize=32 swizzled layout, SM100+). These helpers target the cute sm120 backend
and use the DeepGEMM-compatible per-row layout.

Copyright (c) 2025 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..deep_gemm import get_col_major_tma_aligned_packed_tensor


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """UE8M0 ceil: round x to the next power of two."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


@flashinfer_api
def mxfp8_quantize_per_token(
    input: torch.Tensor,
    masked_m: Optional[torch.Tensor] = None,
    k_gran: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-token (1 x ``k_gran``) MXFP8 quantization with UE8M0 scaling.

    Quantizes ``input`` along the K dimension in blocks of ``k_gran`` elements,
    producing FP8 E4M3 values and INT32-packed UE8M0 scales in the kernel-required
    per-row TMA-aligned MN-major layout.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, dtype ``torch.bfloat16``. Supported shapes:

        - 2D ``(m, k)`` (dense Normal entry input)
        - 3D ``(b, m, k)`` (dense Batched / masked / MoE contiguous entry input)
    masked_m : Optional[torch.Tensor]
        Optional per-group valid row count for the masked path. Shape ``(b,)``,
        dtype ``torch.int32``. When provided, ``input`` must be 3D and this
        marks the masked semantic for the caller; the quantization itself
        always processes the full padded ``input``, and the consuming kernel
        skips invalid rows via ``masked_m`` logic.
    k_gran : int
        UE8M0 K-axis block granularity. Must be ``32`` or ``128``.

    Returns
    -------
    fp8 : torch.Tensor
        FP8 E4M3 quantized tensor, same shape as ``input``,
        dtype ``torch.float8_e4m3fn``.
    sf_packed : torch.Tensor
        INT32-packed UE8M0 scale tensor in per-row TMA-aligned MN-major layout.
        Shape ``(m, ceil(k / (k_gran * 4)))`` for 2D input or
        ``(b, m, ceil(k / (k_gran * 4)))`` for 3D input, dtype ``torch.int32``.
    """
    assert input.dtype == torch.bfloat16, f"input must be bfloat16; got {input.dtype}"
    assert k_gran in (32, 128), f"k_gran must be 32 or 128; got {k_gran}"
    assert input.size(-1) % k_gran == 0, (
        f"input.size(-1)={input.size(-1)} not divisible by k_gran={k_gran}"
    )

    if masked_m is not None:
        assert input.dim() == 3, "masked path requires 3D input shape (b, max_m, k)"
        assert masked_m.dim() == 1 and masked_m.size(0) == input.size(0), (
            f"masked_m shape {tuple(masked_m.shape)} must be (b,) = ({input.size(0)},)"
        )
        assert masked_m.dtype == torch.int32

    if input.dim() == 2:
        m, n = input.shape
        x_view = input.view(m, -1, k_gran)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf_fp32 = _ceil_to_ue8m0(x_amax / 448.0)
        fp8 = (x_view * (1.0 / sf_fp32.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    elif input.dim() == 3:
        g, m, n = input.shape
        x_view = input.view(g, m, -1, k_gran)
        x_amax = x_view.abs().float().amax(dim=3).view(g, m, -1).clamp(1e-4)
        sf_fp32 = _ceil_to_ue8m0(x_amax / 448.0)
        fp8 = (
            (x_view * (1.0 / sf_fp32.unsqueeze(3)))
            .to(torch.float8_e4m3fn)
            .view(g, m, n)
        )
    else:
        raise ValueError(f"input must be 2D or 3D; got {input.dim()}D")

    sf_packed = get_col_major_tma_aligned_packed_tensor(sf_fp32)
    return fp8, sf_packed


@flashinfer_api
def mxfp8_quantize_per_block(
    input: torch.Tensor,
    k_gran: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-block (``k_gran`` x ``k_gran``) MXFP8 quantization for B/weight matrices.

    Quantizes ``input`` into symmetric 2D blocks of size ``(k_gran, k_gran)``,
    producing FP8 E4M3 values and FP32 UE8M0 scales. The FP32 scales are NOT
    yet TMA-aligned — caller must chain ``mxfp8_transform_sf_layout`` to
    convert to the kernel-required INT32-packed per-row layout.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, dtype ``torch.bfloat16``. Supported shapes:

        - 2D ``(n, k)`` (dense Normal / MoE A-shared B weight)
        - 3D ``(b, n, k)`` (dense Batched / MoE per-expert B weight)
    k_gran : int
        Block size on both M and K axes. Must be ``32`` or ``128``.

    Returns
    -------
    fp8 : torch.Tensor
        FP8 E4M3 quantized tensor, same shape as ``input``,
        dtype ``torch.float8_e4m3fn``.
    sf : torch.Tensor
        FP32 UE8M0 scale tensor. Shape ``(ceil(n/k_gran), ceil(k/k_gran))``
        for 2D input or ``(b, ceil(n/k_gran), ceil(k/k_gran))`` for 3D input.
    """
    assert input.dtype == torch.bfloat16, f"input must be bfloat16; got {input.dtype}"
    assert k_gran in (32, 128), f"k_gran must be 32 or 128; got {k_gran}"
    assert input.dim() in (2, 3), f"input must be 2D or 3D; got {input.dim()}D"

    squeezed = input.dim() == 2
    if squeezed:
        input = input.unsqueeze(0)

    g, m, n = input.shape
    pad_m = _ceil_div(m, k_gran) * k_gran
    pad_n = _ceil_div(n, k_gran) * k_gran
    x_padded = torch.zeros((g, pad_m, pad_n), dtype=input.dtype, device=input.device)
    x_padded[:, :m, :n] = input

    x_view = x_padded.view(g, -1, k_gran, pad_n // k_gran, k_gran)
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_scaled = x_scaled.view_as(x_padded)[:, :m, :n].contiguous()
    sf = sf.view(g, pad_m // k_gran, pad_n // k_gran)

    if squeezed:
        x_scaled = x_scaled.squeeze(0)
        sf = sf.squeeze(0)

    return x_scaled, sf


@flashinfer_api
def mxfp8_transform_sf_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Union[Tuple[int, int, int], Tuple[int, int]],
    num_groups: Optional[int] = None,
    is_sfa: Optional[bool] = None,
) -> torch.Tensor:
    r"""Transform a scale-factor tensor into the kernel-required layout
    (INT32-packed, per-row, TMA-aligned, MN-major). DeepGEMM-aligned API.

    Parameters
    ----------
    sf : torch.Tensor
        Scale factor tensor. Two valid input forms:

        - ``torch.int32`` with per-row (``gran_mn=1``): already in kernel layout,
          returned as-is (fast-path, no-op).
        - ``torch.float`` with any ``gran_mn``: broadcast (if ``gran_mn > 1``)
          via index_select to per-row, then INT-packed and TMA-aligned via the
          DG building block.
    mn : int
        M (for A scale) or N (for B scale) dimension.
    k : int
        K dimension.
    recipe : Union[Tuple[int, int, int], Tuple[int, int]]
        Either:

        - 2-tuple ``(m_gran, k_gran)`` per-matrix recipe (preferred).
        - 3-tuple ``(m_gran_a, m_gran_b, k_gran)`` joint recipe (DG-style).
          Requires ``is_sfa`` to pick the correct ``m_gran``.
    num_groups : Optional[int]
        Number of groups for 3D scale tensors; ``None`` for 2D.
    is_sfa : Optional[bool]
        Required if ``recipe`` is a 3-tuple; ``True`` selects the A granularity,
        ``False`` selects the B granularity.

    Returns
    -------
    torch.Tensor
        INT32-packed UE8M0 scale tensor in per-row TMA-aligned MN-major layout.
    """
    if len(recipe) == 3:
        assert is_sfa is not None, (
            "3-tuple recipe requires is_sfa to select A or B granularity"
        )
        gran_mn = recipe[0] if is_sfa else recipe[1]
        gran_k = recipe[2]
    elif len(recipe) == 2:
        gran_mn, gran_k = recipe
    else:
        raise ValueError(f"recipe must be 2-tuple or 3-tuple, got length {len(recipe)}")

    assert gran_k in (32, 128), f"gran_k must be 32 or 128; got {gran_k}"

    if sf.dtype == torch.int32 and gran_mn == 1:
        return sf

    if sf.dtype == torch.float:
        if gran_mn != 1:
            sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // gran_mn)
        return get_col_major_tma_aligned_packed_tensor(sf)

    raise ValueError(
        f"Unsupported sf dtype={sf.dtype} with gran_mn={gran_mn}; "
        f"INT-packed sf must already be per-row (gran_mn=1)"
    )


@flashinfer_api
def mxfp8_dequantize_per_token(
    fp8: torch.Tensor,
    sf_packed: torch.Tensor,
    k_gran: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    r"""Dequantize FP8 + INT32-packed UE8M0 scales back to BF16/FP16.

    Reverse of ``mxfp8_quantize_per_token`` for upstream-test dequant-self
    correctness reference (testing fp8 arithmetic vs dequant pipeline
    consistency, calc_diff should be ~0).

    Parameters
    ----------
    fp8 : torch.Tensor
        FP8 E4M3 tensor of shape ``(m, n)`` or ``(b, m, n)``, dtype
        ``torch.float8_e4m3fn``.
    sf_packed : torch.Tensor
        INT32-packed UE8M0 scale tensor produced by
        ``mxfp8_quantize_per_token`` (per-row TMA-aligned MN-major layout).
    k_gran : int
        UE8M0 K-axis block granularity (must match the value used in
        ``mxfp8_quantize_per_token``).
    dtype : torch.dtype
        Output dtype, default ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        Dequantized tensor of the same shape as ``fp8``.
    """
    assert fp8.dtype == torch.float8_e4m3fn, (
        f"fp8 must be float8_e4m3fn; got {fp8.dtype}"
    )
    assert sf_packed.dtype == torch.int32, (
        f"sf_packed must be int32; got {sf_packed.dtype}"
    )
    assert k_gran in (32, 128), f"k_gran must be 32 or 128; got {k_gran}"

    # Unpack int32 → 4 UE8M0 (uint8) via bit-shift (stride-layout-agnostic)
    s0 = sf_packed & 0xFF
    s1 = (sf_packed >> 8) & 0xFF
    s2 = (sf_packed >> 16) & 0xFF
    s3 = (sf_packed >> 24) & 0xFF
    scales_uint8 = torch.stack([s0, s1, s2, s3], dim=-1)
    scales_uint8 = scales_uint8.view(*sf_packed.shape[:-1], -1)
    scales_float = torch.pow(2.0, scales_uint8.float() - 127.0)

    n = fp8.size(-1)
    num_blocks = _ceil_div(n, k_gran)
    scales_float = scales_float[..., :num_blocks]

    if fp8.dim() == 2:
        m, n_ = fp8.shape
        x_view = fp8.view(m, -1, k_gran).to(torch.float32)
        x_dequant = x_view * scales_float.unsqueeze(2)
        return x_dequant.view(m, n_).to(dtype)
    elif fp8.dim() == 3:
        b, m, n_ = fp8.shape
        x_view = fp8.view(b, m, -1, k_gran).to(torch.float32)
        x_dequant = x_view * scales_float.unsqueeze(3)
        return x_dequant.view(b, m, n_).to(dtype)
    else:
        raise ValueError(f"fp8 must be 2D or 3D; got {fp8.dim()}D")
