"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.quantization import gen_quantization_module
from ..utils import (
    INT4Tensor,
    INT4_GROUP_SIZE,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_quantization_module():
    return gen_quantization_module().build_and_load()


@register_custom_op("flashinfer::packbits", mutates_args=())
def _packbits(x: torch.Tensor, bitorder: str) -> torch.Tensor:
    device = x.device
    x = x.to(torch.bool)
    y = torch.empty((x.size(0) + 7) // 8, dtype=torch.uint8, device=device)
    get_quantization_module().packbits(x, bitorder, y)
    return y


@register_fake_op("flashinfer::packbits")
def _fake_packbits(x: torch.Tensor, bitorder: str) -> torch.Tensor:
    return torch.empty((x.size(0) + 7) // 8, dtype=torch.uint8, device=x.device)


@flashinfer_api
def packbits(x: torch.Tensor, bitorder: str = "big") -> torch.Tensor:
    r"""Pack the elements of a binary-valued array into bits in a uint8 array.

    The semantics of this function is the same as `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_.

    Parameters
    ----------
    x: torch.Tensor
        The 1D binary-valued array to pack.
    bitorder: str
        The bit-order ("bit"/"little") of the output. Default is "big".

    Returns
    -------
    y: torch.Tensor
        An uint8 packed array, shape ``((x.size(0) + 7) / 8),)``.

    Examples
    --------

    >>> import torch
    >>> from flashinfer import packbits
    >>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool, device="cuda")
    >>> x_packed = packbits(x)
    >>> list(map(bin, x_packed.tolist()))
    ['0b10110011']

    See Also
    --------
    segment_packbits
    """
    return _packbits(x, bitorder)


@flashinfer_api
def segment_packbits(
    x: torch.Tensor, indptr: torch.Tensor, bitorder: str = "big"
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Pack a batch of binary-valued segments into bits in a uint8 array.

    For each segment, the semantics of this function is the same as `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_.

    Parameters
    ----------
    x: torch.Tensor
        The 1D binary-valued array to pack, shape ``(indptr[-1],)``.
    indptr: torch.Tensor
        The index pointer of each segment in :attr:`x`, shape ``(batch_size + 1,)``.
        The i-th segment in :attr:`x` is ``x[indptr[i]:indptr[i+1]]``.
    bitorder: str
        The bit-order ("bit"/"little") of the output. Default is "big".

    Returns
    -------
    y: torch.Tensor
        An uint8 packed array, shape: ``(new_indptr[-1],)``.
        The ``y[new_indptr[i]:new_indptr[i+1]]`` contains the packed bits ``x[indptr[i]:indptr[i+1]]``.
    new_indptr: torch.Tensor
        The new index pointer of each packed segment in :attr:`y`, shape ``(batch_size + 1,)``.
        It's guaranteed that ``new_indptr[i+1] - new_indptr[i] == (indptr[i+1] - indptr[i] + 7) // 8``.

    Examples
    --------

    >>> import torch
    >>> from flashinfer import segment_packbits
    >>> x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=torch.bool, device="cuda")
    >>> x_packed, new_indptr = segment_packbits(x, torch.tensor([0, 4, 7, 11], device="cuda"), bitorder="big")
    >>> list(map(bin, x_packed.tolist()))
    ['0b10110000', '0b100000', '0b11010000']
    >>> new_indptr
    tensor([0, 1, 2, 3], device='cuda:0')

    Note
    ----
    ``torch.compile`` is not supported for this function because it's data dependent.

    See Also
    --------
    packbits
    """
    seglen = indptr[1:] - indptr[:-1]
    packed_len = (seglen + 7) // 8
    indptr_new = torch.zeros(len(indptr), dtype=indptr.dtype, device=indptr.device)
    indptr_new[1:] = torch.cumsum(packed_len, 0)
    output_nnzs = indptr_new[-1].item()

    device = x.device
    indptr = indptr.to(torch.int32)
    indptr_new = indptr_new.to(torch.int32)
    y = torch.empty(output_nnzs, dtype=torch.uint8, device=device)
    get_quantization_module().segment_packbits(x, indptr, indptr_new, bitorder, y)
    return y, indptr_new


@flashinfer_api
def int4_quantize(
    x: torch.Tensor,
    group_size: int = INT4_GROUP_SIZE,
) -> INT4Tensor:
    r"""Quantize the input tensor into grouped packed int4 format."""

    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    hidden_dim = x.shape[-1]
    if hidden_dim % group_size != 0:
        raise ValueError(
            f"x.shape[-1] must be divisible by group_size, got {hidden_dim} and {group_size}"
        )

    x_fp32 = x.to(torch.float32)
    num_groups = hidden_dim // group_size
    x_grouped = x_fp32.reshape(*x.shape[:-1], num_groups, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = torch.round(x_grouped / scale).clamp_(-8, 7).to(torch.int8)
    q_unsigned = (q + 8).to(torch.uint8).reshape(*x.shape[:-1], hidden_dim)

    if hidden_dim % 2 != 0:
        pad = torch.zeros(
            (*q_unsigned.shape[:-1], 1),
            dtype=q_unsigned.dtype,
            device=q_unsigned.device,
        )
        q_unsigned = torch.cat([q_unsigned, pad], dim=-1)

    packed = q_unsigned[..., 0::2] | (q_unsigned[..., 1::2] << 4)
    return INT4Tensor(
        packed.contiguous(),
        scale.squeeze(-1).to(torch.float16).contiguous(),
        group_size=group_size,
        original_shape=tuple(x.shape),
    )


@flashinfer_api
def int4_dequantize(
    x: INT4Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    r"""Dequantize an :class:`INT4Tensor` back to a dense tensor."""

    if not isinstance(x, INT4Tensor):
        raise TypeError(f"x must be an INT4Tensor, got {type(x)}")

    hidden_dim = x.original_shape[-1]
    unpacked_dim = math.ceil(hidden_dim / 2) * 2
    unpacked = torch.empty(
        (*x.data.shape[:-1], unpacked_dim),
        dtype=torch.uint8,
        device=x.data.device,
    )
    unpacked[..., 0::2] = x.data & 0x0F
    unpacked[..., 1::2] = x.data >> 4
    unpacked = unpacked[..., :hidden_dim]

    num_groups = math.ceil(hidden_dim / x.group_size)
    padded_hidden_dim = num_groups * x.group_size
    if padded_hidden_dim != hidden_dim:
        pad = torch.full(
            (*unpacked.shape[:-1], padded_hidden_dim - hidden_dim),
            8,
            dtype=torch.uint8,
            device=x.data.device,
        )
        unpacked = torch.cat([unpacked, pad], dim=-1)

    q = unpacked.to(torch.int16) - 8
    q = q.reshape(*x.original_shape[:-1], num_groups, x.group_size).to(dtype)
    scale = x.scale.to(dtype).unsqueeze(-1)
    return (q * scale).reshape(*x.original_shape[:-1], padded_hidden_dim)[
        ..., :hidden_dim
    ]
