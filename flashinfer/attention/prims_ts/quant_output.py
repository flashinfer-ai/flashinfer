# Copyright (c) 2026 by FlashInfer team.
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

"""Host side of block-scaled (quantized) attention outputs; no CuTe DSL imports.

A quantized output is selected by a pair of dtypes: ``out_dtype`` is the
element dtype of ``out`` and ``out_scale_dtype`` the dtype of its block scales.
``out`` keeps the usual output shape, and ``out_scale`` holds one scale per
``sf_vec_size`` channels of the ``[tokens, Hq * Dv]`` view of ``out``, in the
128x4 swizzled layout the block-scaled GEMMs consume
(``mxfp8_quantize(..., is_sf_swizzled_layout=True)``). The device recipes and
the layout are described in ``kernels/quant_output.py``.

======================  ========================  ======  ==============
``out_dtype``           ``out_scale_dtype``       Format  channels/scale
======================  ========================  ======  ==============
``torch.float8_e4m3fn`` ``torch.float8_e8m0fnu``  MXFP8   32
``torch.float8_e4m3fn`` ``torch.uint8``           MXFP8   32
======================  ========================  ======  ==============

``uint8`` scales hold the same UE8M0 bytes, as ``mxfp8_quantize`` returns
and ``mm_mxfp8`` takes them. A plan keeps the scale dtype it was given for
its ``out_scale`` buffers; kernels and compile keys use the format's
canonical ``sf_dtype``, so both spellings share compiled kernels.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class QuantOutputFormat:
    """One block-scaled output format, keyed by ``(data_dtype, sf_dtype)``."""

    # Short name used in messages and documentation.
    name: str
    # Element dtype of the ``out`` tensor.
    data_dtype: torch.dtype
    # Canonical scale dtype; buffers may also use a byte alias.
    sf_dtype: torch.dtype
    # Consecutive channels of one token row covered by one scale.
    sf_vec_size: int


_MXFP8 = QuantOutputFormat(
    name="mxfp8",
    data_dtype=torch.float8_e4m3fn,
    sf_dtype=torch.float8_e8m0fnu,
    sf_vec_size=32,
)

# Keyed by (out_dtype, out_scale_dtype); several scale dtypes may name one
# format.
QUANT_OUTPUT_FORMATS: dict[tuple[torch.dtype, torch.dtype], QuantOutputFormat] = {
    (torch.float8_e4m3fn, torch.float8_e8m0fnu): _MXFP8,
    (torch.float8_e4m3fn, torch.uint8): _MXFP8,
}


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _supported_pairs() -> str:
    return ", ".join(
        f"{fmt.name} (out_dtype={data}, out_scale_dtype={sf})"
        for (data, sf), fmt in QUANT_OUTPUT_FORMATS.items()
    )


def quant_output_format(
    out_dtype: torch.dtype, out_scale_dtype: Optional[torch.dtype]
) -> Optional[QuantOutputFormat]:
    """The quantized format of ``(out_dtype, out_scale_dtype)``.

    Returns None without a scale dtype and rejects unsupported pairs.
    """
    if out_scale_dtype is None:
        return None
    if not isinstance(out_scale_dtype, torch.dtype):
        raise TypeError("out_scale_dtype must be a torch.dtype or None")
    fmt = QUANT_OUTPUT_FORMATS.get((out_dtype, out_scale_dtype))
    if fmt is None:
        raise NotImplementedError(
            f"unsupported quantized output out_dtype={out_dtype}, "
            f"out_scale_dtype={out_scale_dtype}; supported: {_supported_pairs()}"
        )
    return fmt


def implied_quant_out_dtype(
    out_dtype: Optional[torch.dtype], out_scale: torch.Tensor
) -> torch.dtype:
    """Resolve a one-shot call's ``out_dtype`` from its ``out_scale`` buffer.

    ``out_scale.dtype`` is the scale dtype. A given ``out_dtype`` must pair
    with it; an omitted one is the data dtype of the only format with that
    scale dtype.
    """
    if not isinstance(out_scale, torch.Tensor):
        raise TypeError("out_scale must be a torch.Tensor")
    if out_dtype is not None:
        quant_output_format(out_dtype, out_scale.dtype)
        return out_dtype
    data_dtypes = [data for data, sf in QUANT_OUTPUT_FORMATS if sf == out_scale.dtype]
    if len(data_dtypes) != 1:
        raise ValueError(
            f"out_scale of dtype {out_scale.dtype} needs out_dtype to select a "
            f"quantized output; supported: {_supported_pairs()}"
        )
    return data_dtypes[0]


def out_scale_cols(fmt: QuantOutputFormat, num_qo_heads: int, head_dim_vo: int) -> int:
    """Scale columns of one token row, padded to the 4-column swizzle atom."""
    if head_dim_vo % fmt.sf_vec_size:
        raise ValueError(
            f"{fmt.name} output requires a value head dimension divisible by "
            f"{fmt.sf_vec_size}, got {head_dim_vo}"
        )
    return _round_up(num_qo_heads * head_dim_vo // fmt.sf_vec_size, 4)


def out_scale_numel(
    fmt: QuantOutputFormat, num_tokens: int, num_qo_heads: int, head_dim_vo: int
) -> int:
    """Scales in the swizzled buffer for ``num_tokens`` output rows."""
    return _round_up(num_tokens, 128) * out_scale_cols(fmt, num_qo_heads, head_dim_vo)


def prepare_out_scale(
    out_scale: Optional[torch.Tensor],
    *,
    out_dtype: torch.dtype,
    out_scale_dtype: Optional[torch.dtype],
    out_shape: tuple[int, ...],
    device: torch.device,
    validate: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(returned, kernel_view)`` for a quantized output's scale buffer.

    ``out_shape`` is the output data shape ``[..., Hq, Dv]``; its leading
    dimensions are the token rows. An omitted buffer is allocated flat in the
    planned ``out_scale_dtype``. A caller buffer is any contiguous tensor of
    that dtype with exactly ``out_scale_numel`` elements, such as the 2-D
    ``[round_up(tokens, 128), round_up(Hq * Dv / sf_vec_size, 4)]`` view; it
    is returned as given. The kernel receives a flat ``uint8`` view. Plans
    without a quantized output reject ``out_scale``.
    """
    fmt = quant_output_format(out_dtype, out_scale_dtype)
    if fmt is None:
        if out_scale is not None:
            raise ValueError(
                "out_scale belongs to quantized-output plans "
                "(plan(out_scale_dtype=...))"
            )
        return None, None
    num_tokens = math.prod(out_shape[:-2])
    numel = out_scale_numel(fmt, num_tokens, out_shape[-2], out_shape[-1])
    if out_scale is None:
        out_scale = torch.empty(numel, dtype=out_scale_dtype, device=device)
    elif validate:
        if not isinstance(out_scale, torch.Tensor):
            raise TypeError("out_scale must be a torch.Tensor")
        if out_scale.device != device:
            raise ValueError(f"out_scale must be on {device}, got {out_scale.device}")
        if out_scale.dtype != out_scale_dtype:
            raise ValueError(
                f"out_scale must have the planned scale dtype {out_scale_dtype}, "
                f"got {out_scale.dtype}"
            )
        if out_scale.numel() != numel or not out_scale.is_contiguous():
            raise ValueError(
                f"out_scale must be a contiguous buffer of {numel} swizzled "
                f"scales for {num_tokens} tokens, got shape "
                f"{tuple(out_scale.shape)}"
            )
        if out_scale.data_ptr() % 16:
            raise ValueError("out_scale data pointer must be 16-byte aligned")
    return out_scale, out_scale.view(torch.uint8).view(-1)
