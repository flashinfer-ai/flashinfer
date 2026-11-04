"""Torch-facing entry points of the MX quantizer.

:func:`mxfp8_quantize_batch_invariant` is what :func:`flashinfer.mxfp8_quantize` runs under ``batch_invariant=True``
(``backend="auto"``): the class ``mxfp8.quant`` with FlashInfer's output contract exactly (``x_q`` a ``float8_e4m3fn``
tensor of shape ``(*input.shape[:-1], padded_K)``, ``sf`` the flat ``uint8`` UE8M0 scales of the chosen ``SfLayout``;
``alignment`` pads K to a multiple of ``alignment`` with zero columns, as FlashInfer's own backends do). Every
unsupported input raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`; nothing falls back.

:func:`mx_quantize` is the generic entry the tests drive (``config=`` to force a member
of the enumerated space, ``sm_count=`` to ask the dispatcher for another SKU's pick, ``out=`` for preallocated outputs).
Inputs bf16 / f16 / fp32 (one class per format: the 16-bit inputs are exact widenings).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..support import check_device, unsupported
from .layout import layout_name, padded_sf_cols, padded_sf_rows, sf_numel
from .mx_schedule import BLOCK, MXQuantConfig, default_config, num_blocks
from .mx_spec import DEFAULT_RULE, FORMATS, MXFP4, MXFP8, MXFormat, ScaleRule

CLASS_MXFP8 = "mxfp8.quant"
CLASS_MXFP4 = "mxfp4.quant"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
MXFP8_QUANT_ARCHS = ((10, 0), (10, 3), (10, 7))
#: The enumerated configuration space covers rows of up to this many input bytes (K <= 65536 for bf16 / f16, 32768 for fp32).
MAX_ROW_BYTES = 131072
_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_LAYOUT_NAMES = {
    "128x4": "128x4",
    "layout_128x4": "128x4",
    "swizzled": "128x4",
    "swizzled_128x4": "128x4",
    "8x4": "8x4",
    "layout_8x4": "8x4",
    "swizzled_8x4": "8x4",
    "linear": "linear",
    "layout_linear": "linear",
}
_sm_count_cache: dict = {}


def layout_of(layout) -> str:
    """The scale-factor layout name: ``"128x4"`` (``None`` / ``True`` / FlashInfer's ``SfLayout.layout_128x4``), ``"8x4"``
    (``SfLayout.layout_8x4``) or ``"linear"`` (``False`` / ``SfLayout.layout_linear``)."""
    if layout is None or layout is True:
        return "128x4"
    if layout is False:
        return "linear"
    if isinstance(layout, str):
        if layout in _LAYOUT_NAMES:
            return _LAYOUT_NAMES[layout]
        raise ValueError(
            f"unsupported scale-factor layout {layout!r} (128x4 / 8x4 swizzled or linear)"
        )
    name = getattr(layout, "name", None)  # flashinfer.SfLayout
    if name in _LAYOUT_NAMES:
        return _LAYOUT_NAMES[name]
    raise ValueError(
        f"unsupported scale-factor layout {layout!r} (128x4 / 8x4 swizzled or linear)"
    )


def device_sm_count(device) -> int:
    key = torch.device(device).index
    if key not in _sm_count_cache:
        _sm_count_cache[key] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return _sm_count_cache[key]


def _rows(t: torch.Tensor, K: int, name: str) -> torch.Tensor:
    try:
        t2 = t.view(-1, K)
    except RuntimeError:
        raise ValueError(
            f"{name} of shape {tuple(t.shape)} with strides {tuple(t.stride())} cannot be viewed as "
            f"(rows, {K}) without a copy; pass a contiguous (or row-strided 2-D) tensor"
        ) from None
    if t2.stride(-1) != 1:
        raise ValueError(f"{name} must have a unit stride along the last dimension")
    return t2


def mx_quantize(
    x: torch.Tensor,
    fmt: MXFormat,
    swizzled="128x4",
    *,
    rule: ScaleRule = DEFAULT_RULE,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    config: Optional[MXQuantConfig] = None,
    enable_pdl: bool = False,
    sm_count: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` (bf16 / f16 / fp32, last dim ``K`` a multiple of 32) to the MX format ``fmt``.

    Returns ``(codes, sf)``: ``codes`` uint8 of shape ``(..., K)`` (e4m3 bytes); ``sf`` the flat uint8 UE8M0 scale bytes in the layout ``swizzled``
    names: ``"128x4"`` (rows padded to 128, scale columns to 4, padding zero), ``"8x4"`` (rows padded to 8, columns to 4)
    or ``"linear"`` (``M x K/32``). Bitwise identical for every ``config`` in :func:`mx_schedule.all_valid_configs`, for
    every batch composition, every SM count and the three layouts."""
    from .mx_kernel import get_compiled_kernel

    if x.dtype not in _DTYPE_STR:
        raise ValueError(f"input dtype {x.dtype} not supported (bf16 / f16 / fp32)")
    swizzled = layout_name(swizzled)
    K = x.shape[-1]
    nb = num_blocks(K)
    x2 = _rows(x, K, "input")
    M = x2.shape[0]
    elem_bytes = x.element_size()
    padded_M = padded_sf_rows(M, swizzled)
    kbytes = K * fmt.elem_bits // 8
    nsf = sf_numel(M, K, BLOCK, swizzled)
    if out is None:
        codes = torch.empty(M, kbytes, dtype=torch.uint8, device=x.device)
        sf = torch.empty(nsf, dtype=torch.uint8, device=x.device)
    else:
        codes, sf = out
        codes = _rows(codes.view(torch.uint8), kbytes, "codes out")
        assert codes.shape == (M, kbytes), (codes.shape, M, kbytes)
        sf = sf.view(torch.uint8).view(-1)
        assert sf.is_contiguous() and sf.numel() >= nsf, (sf.numel(), nsf)
        sf = sf[:nsf]
    if config is None:
        config = default_config(
            M, K, elem_bytes, sm_count or device_sm_count(x.device), swizzled
        )
    else:
        config.validate(K, elem_bytes, swizzled)
    if x2.data_ptr() % 16:
        # an under-aligned storage offset: the 16-byte vector loads need the base aligned (a copy is a placement)
        x2 = x2.clone(memory_format=torch.contiguous_format)
    if codes.data_ptr() % 16 or sf.data_ptr() % 16:
        raise ValueError("the codes and sf output buffers must be 16-byte aligned")
    vec = 16 // elem_bytes
    assert x2.stride(0) % vec == 0 or M == 1, (
        "input row stride must be a multiple of 16 bytes"
    )
    assert codes.stride(0) % 16 == 0 or M == 1, (
        "codes row stride must be a multiple of 16 bytes"
    )
    assert nb >= 1
    kernel = get_compiled_kernel(
        _DTYPE_STR[x.dtype], K, config, fmt, rule, swizzled, bool(enable_pdl)
    )
    kernel(x2, codes, sf, M, K, padded_M)
    return codes.view(*x.shape[:-1], kbytes), sf


def mxfp8_quantize_batch_invariant(
    input: torch.Tensor,
    sf_swizzle_layout=None,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
    *,
    config: Optional[MXQuantConfig] = None,
    sm_count: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``flashinfer.mxfp8_quantize(input, ..., backend="auto", batch_invariant=True)``: the class ``mxfp8.quant``.

    FlashInfer's contract is reproduced exactly: ``x_q`` is a ``float8_e4m3fn`` tensor of shape
    ``(*input.shape[:-1], padded_K)`` with ``padded_K = ceil(K / alignment) * alignment`` (the padding columns are zero
    e4m3 bytes under a zero scale byte, as FlashInfer's backends produce them), ``sf`` the flat ``uint8`` UE8M0 scales
    of ``sf_swizzle_layout`` (``SfLayout.layout_128x4``: rows padded to 128 and scale columns to 4; ``layout_8x4``: rows
    to 8, columns to 4; ``layout_linear``: ``M x padded_K / 32``). ``enable_pdl=None`` follows the device (PDL on
    SM >= 9.0). ``config`` / ``sm_count`` force a configuration / another SKU's dispatcher pick (members of the enumerated
    space: the bits never depend on them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    K not a multiple of 32, layout, device, architecture); it never falls back."""
    from ...utils import device_support_pdl

    check_device(input.device, CLASS_MXFP8, MXFP8_QUANT_ARCHS)
    if input.dtype not in _DTYPE_STR:
        raise unsupported(
            CLASS_MXFP8, f"input dtype {input.dtype}", "bf16, f16 or fp32 inputs"
        )
    if input.dim() < 1:
        raise unsupported(
            CLASS_MXFP8, f"a {input.dim()}-D input", "inputs of shape (..., K)"
        )
    K = int(input.shape[-1])
    if K <= 0 or K % BLOCK:
        raise unsupported(CLASS_MXFP8, f"K = {K}", f"K a positive multiple of {BLOCK}")
    if alignment <= 0 or alignment % BLOCK:
        raise unsupported(
            CLASS_MXFP8, f"alignment = {alignment}", f"a positive multiple of {BLOCK}"
        )
    padded_k = -(-K // alignment) * alignment
    if padded_k * input.element_size() > MAX_ROW_BYTES:
        raise unsupported(
            CLASS_MXFP8,
            f"K = {K} (padded to {padded_k}) for {input.dtype}",
            f"rows of at most {MAX_ROW_BYTES} bytes (padded K <= {MAX_ROW_BYTES // input.element_size()})",
        )
    try:
        layout = layout_of(sf_swizzle_layout)
    except ValueError:
        raise unsupported(
            CLASS_MXFP8,
            f"the scale-factor layout {sf_swizzle_layout!r}",
            "SfLayout.layout_128x4, SfLayout.layout_8x4 or SfLayout.layout_linear",
        ) from None
    if padded_k == K:
        try:
            x2 = _rows(input, K, "input")
        except ValueError:
            x2 = input.reshape(
                -1, K
            )  # a copy: the input could not be viewed as rows (a placement, not numerics)
    else:
        # FlashInfer pads K to a multiple of `alignment` with zero columns; a zero block quantizes to zero codes
        # under the zero scale byte in both the class and FlashInfer's backends.
        rows = input.reshape(-1, K)
        x2 = torch.zeros(
            rows.shape[0], padded_k, dtype=input.dtype, device=input.device
        )
        x2[:, :K] = rows
    required_align = 16
    if x2.data_ptr() % required_align or (
        x2.shape[0] > 1 and (x2.stride(0) * x2.element_size()) % required_align
    ):
        # an under-aligned storage offset or a row stride that is not a multiple of 16 bytes (a view into a wider
        # tensor): the 16-byte vector loads need both; the copy is a placement, not numerics
        x2 = x2.clone(memory_format=torch.contiguous_format)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if (
        x2.shape[0] == 0
    ):  # no rows: nothing to launch; the outputs keep their documented shapes
        codes = torch.empty(
            (*input.shape[:-1], padded_k),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        sf = torch.empty(
            sf_numel(0, padded_k, BLOCK, layout), dtype=torch.uint8, device=input.device
        )
        return codes, sf
    codes, sf = mx_quantize(
        x2, MXFP8, layout, config=config, enable_pdl=bool(enable_pdl), sm_count=sm_count
    )
    return codes.view(torch.float8_e4m3fn).view(*input.shape[:-1], padded_k), sf


def mxfp4_quantize_batch_invariant(
    a: torch.Tensor,
    sfLayout=None,
    enable_pdl: Optional[bool] = None,
    *,
    config: Optional[MXQuantConfig] = None,
    sm_count: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``flashinfer.mxfp4_quantize(a, ..., backend="auto", batch_invariant=True)``: the class ``mxfp4.quant``.

    FlashInfer's contract is reproduced: ``a_fp4`` a ``uint8`` tensor of shape ``(*a.shape[:-1], K / 2)`` (two e2m1
    codes per byte, element ``2j`` in the low nibble), ``a_sf`` the ``uint8`` UE8M0 scales of ``sfLayout`` as the operator's
    ``(rows, columns)`` view (the padded columns of the swizzled layouts, ``K / 32`` for the linear one)
    (``SfLayout.layout_128x4``: rows padded to 128 and scale columns to 4; ``layout_8x4``: rows to 8, columns to 4;
    ``layout_linear``: ``M x K / 32``). ``enable_pdl=None`` follows the device. ``config`` / ``sm_count`` force a
    configuration / another part's dispatcher pick (members of the enumerated space: the bits never depend on them).

    Raises :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for any input outside the class (dtype,
    K not a multiple of 32, a row beyond the enumerated space, layout, device, architecture); it never falls back."""
    from ...utils import device_support_pdl

    check_device(a.device, CLASS_MXFP4, MXFP8_QUANT_ARCHS)
    if a.dtype not in _DTYPE_STR:
        raise unsupported(
            CLASS_MXFP4, f"input dtype {a.dtype}", "bf16, f16 or fp32 inputs"
        )
    if a.dim() < 1:
        raise unsupported(
            CLASS_MXFP4, f"a {a.dim()}-D input", "inputs of shape (..., K)"
        )
    K = int(a.shape[-1])
    if K <= 0 or K % BLOCK:
        raise unsupported(CLASS_MXFP4, f"K = {K}", f"K a positive multiple of {BLOCK}")
    if K * a.element_size() > MAX_ROW_BYTES:
        raise unsupported(
            CLASS_MXFP4,
            f"K = {K} for {a.dtype}",
            f"rows of at most {MAX_ROW_BYTES} bytes (K <= {MAX_ROW_BYTES // a.element_size()})",
        )
    try:
        layout = layout_of(sfLayout)
    except ValueError:
        raise unsupported(
            CLASS_MXFP4,
            f"the scale-factor layout {sfLayout!r}",
            "SfLayout.layout_128x4, SfLayout.layout_8x4 or SfLayout.layout_linear",
        ) from None
    try:
        x2 = _rows(a, K, "input")
    except ValueError:
        x2 = a.reshape(
            -1, K
        )  # a copy: the input could not be viewed as rows (a placement, not numerics)
    if x2.data_ptr() % 16 or (
        x2.shape[0] > 1 and (x2.stride(0) * x2.element_size()) % 16
    ):
        x2 = x2.clone(memory_format=torch.contiguous_format)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    if x2.shape[0] == 0:
        codes = torch.empty((*a.shape[:-1], K // 2), dtype=torch.uint8, device=a.device)
        sf = torch.empty(
            sf_numel(0, K, BLOCK, layout), dtype=torch.uint8, device=a.device
        ).view(-1, padded_sf_cols(K, BLOCK, layout))
        return codes, sf
    codes, sf = mx_quantize(
        x2, MXFP4, layout, config=config, enable_pdl=bool(enable_pdl), sm_count=sm_count
    )
    return codes.view(*a.shape[:-1], K // 2), sf.view(
        -1, padded_sf_cols(K, BLOCK, layout)
    )


def format_of(name: str) -> MXFormat:
    return FORMATS[name]
