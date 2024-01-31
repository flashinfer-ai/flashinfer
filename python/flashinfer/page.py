"""
Copyright (c) 2023 by FlashInfer team.

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
import torch

try:
    import torch
    from . import _kernels
except ImportError as e:
    import os
    import logging
    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning(
            "Kernels are not loaded in documentation build mode."
        )
    else:
        raise e

def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    append_indptr: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
):
    r"""Append a batch of key-value pairs to a paged key-value cache.

    Parameters
    ----------
    append_key : torch.Tensor
        The key tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    append_value : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], num_kv_heads, head_dim]``.
    append_indptr : torch.Tensor
        The indptr tensor of the key-value pairs to append, shape: ``[batch_size + 1]``.
    kv_data : torch.Tensor
        The 5-D tensor of the paged key-value cache, shape:
        ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
        :attr:`kv_layout` is ``NHD``, or
        ``[max_num_pages, 2, num_kv_heads, page_size, num_kv_heads]`` if
        :attr:`kv_layout` is ``NHD``.
    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    kv_layout : str
        The layout of the paged kv-cache, either ``NHD`` or ``HND``.

    Notes
    -----
    The function assumes that the space for appended k/v have already been allocated,
    which means :attr:`kv_indices`, :attr:`kv_indptr`, :attr:`kv_last_page_len` has
    incorporated appended k/v.
    """
    check_kv_layout(kv_layout)
    _kernels.append_paged_kv_cache(
        append_key,
        append_value,
        append_indptr,
        kv_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        getattr(TensorLayout, kv_layout),
    )
