# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch-facing entry points of the NVFP4 quantizer.

:func:`nvfp4_quantize_batch_invariant` is what :func:`flashinfer.nvfp4_quantize` runs under ``batch_invariant=True``
(``backend="auto"``): the per-tensor class ``nvfp4.quant`` (the global scale is an input) or, with
``per_token_activation=True``, the per-token class ``nvfp4.quant.pertoken`` (the global scale is derived per row and
returned), with FlashInfer's output contract: ``fp4`` a ``uint8`` tensor of shape ``(*a.shape[:-1], K / 2)`` (two e2m1
codes per byte, element ``2j`` in the low nibble), ``sf`` the ``uint8`` e4m3 block scales of the chosen ``SfLayout`` as
a ``(rows, padded columns)`` view (rows padded to 128 / 8 for the swizzled layouts, columns to 4; no padding for the linear
layout) and, per token, the fp32 per-row scale. ``do_shuffle`` applies FlashInfer's own host-side row shuffles to the
linear-layout bytes, as its backends do (a placement). Every unsupported input raises
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.

:func:`nvfp4_quantize_rows` is the generic entry the tests drive (``config=`` to force a member of the enumerated space,
``sm_count=`` to ask the dispatcher for another part's pick, ``out=`` for preallocated outputs).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from ..support import check_device, unsupported
from .layout import padded_sf_cols, padded_sf_rows, sf_numel
from .mx_api import _rows, device_sm_count, layout_of
from .nvfp4_schedule import BLOCK, QuantConfig, default_config, num_blocks
from .nvfp4_spec import PER_TENSOR, PER_TOKEN, Variant

CLASS_NVFP4 = "nvfp4.quant"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
NVFP4_QUANT_ARCHS = ((10, 0), (10, 3), (10, 7))
#: The enumerated configuration space covers rows of up to this many input bytes (K <= 65536 for bf16 / f16, 32768 for fp32).
MAX_ROW_BYTES = 131072
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_SHUFFLE_TILE_M = (
    128  # FlashInfer's epilogue tile of the shuffled (TRT-LLM) weight layout
)
_SHUFFLE_ROWS = 32  # the row blocks FlashInfer's shuffle permutes for that tile (``get_shuffle_block_size``)


def _scale_tensor(scale, ref: torch.Tensor) -> torch.Tensor:
    """The global scale as a 1-element fp32 device tensor: a host float becomes a device fill (``torch.full``,
    CUDA-graph capturable); a tensor on another device is copied to the input's device, as the operators' own paths
    do (a copy is not capturable)."""
    if not isinstance(scale, torch.Tensor):
        return torch.full((1,), float(scale), dtype=torch.float32, device=ref.device)
    return scale.to(device=ref.device, dtype=torch.float32).reshape(-1)[:1].contiguous()


def nvfp4_quantize_rows(
    a: torch.Tensor,
    a_global_sf: Union[float, torch.Tensor],
    layout="128x4",
    *,
    per_token: bool = False,
    out: Optional[Tuple[torch.Tensor, ...]] = None,
    config: Optional[QuantConfig] = None,
    enable_pdl: bool = False,
    sm_count: Optional[int] = None,
):
    """Quantize ``a`` (bf16 / f16 / fp32, last dim ``K`` a multiple of 16) to NVFP4 (the fp32 input is the same class:
    the leaf is the fp32 value of the element after the exact widening, so the bf16 / f16 bits are unchanged).

    Returns ``(fp4, sf)``: ``fp4`` uint8 of shape ``(..., K / 2)`` and ``sf`` the flat uint8 e4m3 block scales in the
    layout ``layout`` names (``"128x4"``, ``"8x4"`` or ``"linear"``; padding bytes zero). With ``per_token=True`` a third
    tensor, the per-row fp32 scale, is returned and ``a_global_sf`` is the inverse base scale (TE: ``1 / (448 * 6)``);
    otherwise ``a_global_sf`` is the per-tensor global scale, an input (a host float or a 1-element fp32 device tensor).
    Bitwise identical for every ``config`` in :func:`nvfp4_schedule.all_valid_configs`, for every batch composition,
    every SM count and the three layouts."""
    from .nvfp4_kernel import get_compiled_kernel

    if a.dtype not in _DTYPE_STR:
        raise ValueError(f"input dtype {a.dtype} not supported (bf16 / f16 / fp32)")
    layout = layout_of(layout)
    K = a.shape[-1]
    num_blocks(K)
    x2 = _rows(a, K, "input")
    M = x2.shape[0]
    variant: Variant = PER_TOKEN if per_token else PER_TENSOR
    gs = _scale_tensor(a_global_sf, a)
    padded_M = padded_sf_rows(M, layout)
    nsf = sf_numel(M, K, BLOCK, layout)
    if out is None:
        fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=a.device)
        sf = torch.empty(nsf, dtype=torch.uint8, device=a.device)
    else:
        fp4, sf = out[0], out[1]
        fp4 = _rows(fp4, K // 2, "fp4 out")
        assert fp4.shape == (M, K // 2) and fp4.dtype == torch.uint8
        sf = sf.view(-1)
        assert sf.dtype == torch.uint8 and sf.is_contiguous() and sf.numel() >= nsf
        sf = sf[:nsf]
    pts = torch.empty(M, dtype=torch.float32, device=a.device) if per_token else gs
    elem_bytes = a.element_size()
    if config is None:
        config = default_config(
            K, M, elem_bytes, per_token, sm_count or device_sm_count(a.device), layout
        )
    else:
        config.validate(K, elem_bytes)
    if x2.data_ptr() % 16:
        # an under-aligned storage offset: the 16-byte vector loads need the base aligned (a copy is a placement)
        x2 = x2.clone(memory_format=torch.contiguous_format)
    assert x2.stride(0) % (16 // elem_bytes) == 0 or M == 1, (
        "input row stride must be a multiple of 16 bytes"
    )
    if fp4.data_ptr() % 16 or sf.data_ptr() % 16:
        raise ValueError("the fp4 and sf output buffers must be 16-byte aligned")
    assert fp4.stride(0) % 8 == 0 or M == 1, (
        "fp4 row stride must be a multiple of 8 bytes"
    )
    kernel = get_compiled_kernel(
        _DTYPE_STR[a.dtype], K, config, variant, bool(enable_pdl), layout
    )
    kernel(x2, fp4, sf, pts, gs, M, K, padded_M)
    fp4 = fp4.view(*a.shape[:-1], K // 2)
    if per_token:
        return fp4, sf, pts
    return fp4, sf


def nvfp4_quantize_batch_invariant(
    a: torch.Tensor,
    a_global_sf: Union[float, torch.Tensor],
    sfLayout=None,
    do_shuffle: bool = False,
    sf_vec_size: int = 16,
    enable_pdl: Optional[bool] = None,
    per_token_activation: bool = False,
    expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    nvfp4_4over6=None,
    *,
    config: Optional[QuantConfig] = None,
    sm_count: Optional[int] = None,
):
    """``flashinfer.nvfp4_quantize(a, a_global_sf, ..., backend="auto", batch_invariant=True)``: the class
    ``nvfp4.quant`` (``per_token_activation=False``: the global scale is the input ``a_global_sf``, a host float or a
    1-element fp32 device tensor) or ``nvfp4.quant.pertoken`` (``True``: ``a_global_sf`` is the inverse base scale, the
    row's global scale is derived from its amax and returned as the fp32 per-token scale).

    FlashInfer's contract is reproduced: ``fp4`` a ``uint8`` tensor ``(*a.shape[:-1], K / 2)``; ``sf`` the ``uint8``
    scales of ``sfLayout`` (``SfLayout.layout_128x4``: rows padded to 128 and columns to 4; ``layout_8x4``: rows to 8,
    columns to 4; ``layout_linear``: ``M x K / 16``) as a ``(rows, columns)`` view; per token also the ``(M,)`` fp32
    scale. ``do_shuffle`` (the 128x4 layout only): the class writes the linear-layout bytes and FlashInfer's own host
    shuffles (``shuffle_matrix_a`` / ``shuffle_matrix_sf_a``) permute them, exactly as the other backends compose them.
    ``enable_pdl=None`` follows the device. ``config`` / ``sm_count`` force a configuration / another part's dispatcher
    pick (members of the enumerated space: the bits never depend on them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class: a dtype
    other than bf16 / f16 / fp32, ``K`` not a multiple of 16 or a row beyond the enumerated space, ``sf_vec_size != 16``,
    an explicit 4over6 recipe (a scale-candidate search is another specification), ``expanded_idx_to_permuted_idx``
    (the MoE row map), ``out_scale`` (folded into the per-token scale by other backends), a layout outside the three,
    device, architecture; it never falls back."""
    from ...utils import device_support_pdl

    cls = CLASS_NVFP4 + (".pertoken" if per_token_activation else "")
    check_device(a.device, cls, NVFP4_QUANT_ARCHS)
    if a.dtype not in _DTYPE_STR:
        raise unsupported(cls, f"input dtype {a.dtype}", "bf16, f16 or fp32 inputs")
    if a.dim() < 1:
        raise unsupported(cls, f"a {a.dim()}-D input", "inputs of shape (..., K)")
    K = int(a.shape[-1])
    if K <= 0 or K % BLOCK:
        raise unsupported(cls, f"K = {K}", f"K a positive multiple of {BLOCK}")
    if K * a.element_size() > MAX_ROW_BYTES:
        raise unsupported(
            cls,
            f"K = {K} for {a.dtype}",
            f"rows of at most {MAX_ROW_BYTES} bytes (K <= {MAX_ROW_BYTES // a.element_size()})",
        )
    if sf_vec_size != BLOCK:
        raise unsupported(
            cls,
            f"sf_vec_size = {sf_vec_size}",
            f"sf_vec_size = {BLOCK} (one e4m3 scale per {BLOCK} elements)",
        )
    if nvfp4_4over6 is not None:
        raise unsupported(
            cls,
            "an NVFP4 4over6 scale-candidate search",
            "the plain NVFP4 recipe (nvfp4_4over6=None): a candidate search is another specification",
        )
    if expanded_idx_to_permuted_idx is not None:
        raise unsupported(
            cls,
            "expanded_idx_to_permuted_idx",
            "no row map (the scales are stored at the input's rows)",
        )
    if out_scale is not None:
        raise unsupported(
            cls,
            "out_scale",
            "the per-token scale as the class returns it (fold out_scale in the caller)",
        )
    if isinstance(a_global_sf, torch.Tensor):
        if a_global_sf.numel() != 1:
            raise unsupported(
                cls,
                f"a global scale of {a_global_sf.numel()} elements",
                "one scalar global scale",
            )
    else:
        try:
            float(a_global_sf)
        except (TypeError, ValueError):
            raise unsupported(
                cls,
                f"a global scale of type {type(a_global_sf).__name__}",
                "a float or a 1-element tensor",
            ) from None
    try:
        layout = layout_of(sfLayout)
    except ValueError:
        raise unsupported(
            cls,
            f"the scale-factor layout {sfLayout!r}",
            "SfLayout.layout_128x4, SfLayout.layout_8x4 or SfLayout.layout_linear",
        ) from None
    if do_shuffle and layout != "128x4":
        raise unsupported(
            cls,
            f"do_shuffle with the {layout} layout",
            "do_shuffle with SfLayout.layout_128x4 (as FlashInfer's backends)",
        )
    try:
        x2 = _rows(a, K, "input")
    except ValueError:
        x2 = a.reshape(
            -1, K
        )  # a copy: the input could not be viewed as rows (a placement, not numerics)
    if x2.data_ptr() % 16 or (
        x2.shape[0] > 1 and (x2.stride(0) * x2.element_size()) % 16
    ):
        # an under-aligned storage offset or a row stride that is not a multiple of 16 bytes: the 16-byte vector loads
        # need both; the copy is a placement, not numerics
        x2 = x2.clone(memory_format=torch.contiguous_format)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    store_layout = "linear" if do_shuffle else layout
    M = x2.shape[0]
    if do_shuffle and M % _SHUFFLE_ROWS:
        raise unsupported(
            cls,
            f"do_shuffle with {M} rows",
            f"do_shuffle with a multiple of {_SHUFFLE_ROWS} rows (FlashInfer's shuffled layout permutes rows in blocks of "
            f"{_SHUFFLE_ROWS}; its backends assert the same)",
        )
    pcols = padded_sf_cols(K, BLOCK, store_layout)
    lead = a.shape[:-1]
    if M == 0:  # no rows: nothing to launch; the outputs keep their documented shapes
        fp4 = torch.empty((*lead, K // 2), dtype=torch.uint8, device=a.device)
        sf = torch.empty(
            sf_numel(0, K, BLOCK, store_layout), dtype=torch.uint8, device=a.device
        ).view(-1, pcols)
        if per_token_activation:
            return fp4, sf, torch.empty(0, dtype=torch.float32, device=a.device)
        return fp4, sf
    res = nvfp4_quantize_rows(
        x2,
        a_global_sf,
        store_layout,
        per_token=per_token_activation,
        config=config,
        enable_pdl=bool(enable_pdl),
        sm_count=sm_count,
    )
    fp4, sf = res[0], res[1].view(-1, pcols)
    fp4 = fp4.view(*lead, K // 2)
    if do_shuffle:
        # FlashInfer's shuffled (TRT-LLM) weight layout: its own host permutations of the rows and of the linear-layout
        # scales, composed exactly as its backends compose them
        from ...quantization.fp4_quantization import (
            get_shuffle_matrix_a_row_indices,
            shuffle_matrix_a,
            shuffle_matrix_sf_a,
        )

        fp4_2d = fp4.view(M, K // 2)
        if per_token_activation:
            row_indices = get_shuffle_matrix_a_row_indices(fp4_2d, _SHUFFLE_TILE_M).to(
                fp4.device
            )
            fp4_2d = fp4_2d[row_indices]
            res = (res[0], res[1], res[2][row_indices])
        else:
            fp4_2d = shuffle_matrix_a(fp4_2d, _SHUFFLE_TILE_M)
        fp4 = fp4_2d.view(*lead, K // 2)
        sf = shuffle_matrix_sf_a(sf, _SHUFFLE_TILE_M).reshape(sf.shape)
    if per_token_activation:
        return fp4, sf, res[2]
    return fp4, sf


def fp4_quantize_batch_invariant(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    sf_vec_size: int,
    sf_use_ue8m0: bool,
    is_sf_swizzled_layout: bool,
    is_sf_8x4_layout: bool,
    is_global_scale_inversed: bool,
    enable_pdl: Optional[bool],
    nvfp4_4over6_config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``flashinfer.fp4_quantize(..., backend="auto", batch_invariant=True)``, the entry point behind
    :func:`flashinfer.nvfp4_quantize` and :func:`flashinfer.mxfp4_quantize`: the per-tensor NVFP4 class for
    ``sf_vec_size=16, sf_use_ue8m0=False`` and the MXFP4 class for ``sf_vec_size=32, sf_use_ue8m0=True``, the two recipes
    it quantizes, with its own argument forms and output contract: ``x_q`` uint8 ``(..., K / 2)``; ``sf`` as a
    ``(rows, columns)`` view (the padded columns of the swizzled layouts, ``K / sf_vec_size`` for the linear one); a
    column-major input is quantized as the transpose of its row-major view and the outputs are transposed back, as the
    entry point does; ``global_scale`` None is the scale 1.0; ``is_global_scale_inversed`` gives the class
    ``1 / global_scale`` by one correctly rounded fp32 division. Any other recipe, an fp8 input or a 4over6 recipe raises
    :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back."""
    from ...utils import device_support_pdl
    from .mx_api import CLASS_MXFP4, mxfp4_quantize_batch_invariant

    is_nvfp4 = sf_vec_size == BLOCK and not sf_use_ue8m0
    is_mxfp4 = sf_vec_size == 2 * BLOCK and sf_use_ue8m0
    if not (is_nvfp4 or is_mxfp4):
        raise unsupported(
            f"{CLASS_NVFP4} / {CLASS_MXFP4}",
            f"sf_vec_size = {sf_vec_size} with sf_use_ue8m0 = {sf_use_ue8m0}",
            "sf_vec_size = 16 with e4m3 scales (the NVFP4 class) or sf_vec_size = 32 with UE8M0 scales (the MXFP4 class)",
        )
    cls = CLASS_NVFP4 if is_nvfp4 else CLASS_MXFP4
    if nvfp4_4over6_config is not None:
        raise unsupported(
            cls,
            "an NVFP4 4over6 scale-candidate search",
            "the plain NVFP4 recipe (nvfp4_4over6=None): a candidate search is another specification",
        )
    if is_sf_swizzled_layout:
        layout = "8x4" if is_sf_8x4_layout else "128x4"
    else:
        layout = "linear"
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    # The entry point quantizes a column-major input as the transpose of its row-major view (a placement).
    column_major = input.dim() >= 2 and input.stride(-2) == 1 and input.stride(-1) != 1
    x = input.transpose(-2, -1) if column_major else input
    if is_mxfp4:
        codes, sf = mxfp4_quantize_batch_invariant(x, layout, bool(enable_pdl))
    else:
        if global_scale is None:
            gs: Union[float, torch.Tensor] = 1.0
        elif is_global_scale_inversed:
            gs = 1.0 / global_scale.to(device=x.device, dtype=torch.float32)
        else:
            gs = global_scale
        codes, sf = nvfp4_quantize_batch_invariant(
            x, gs, layout, enable_pdl=bool(enable_pdl)
        )
    K = int(x.shape[-1])
    cols = (
        -(-(K // sf_vec_size) // 4) * 4 if is_sf_swizzled_layout else K // sf_vec_size
    )
    sf = sf.reshape(-1, cols)
    if column_major:
        codes, sf = codes.transpose(-2, -1), sf.transpose(-2, -1)
    return codes, sf
