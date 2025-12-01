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
from typing import Tuple

import torch

from .api_logging import flashinfer_api
from .jit.quantization import gen_quantization_module
from .utils import register_custom_op, register_fake_op


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
