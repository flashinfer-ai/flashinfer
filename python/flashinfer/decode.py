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
import math
from typing import Optional, Union

import torch

try:
    from . import _kernels
except ImportError:
    _kernels = None

from .utils import (
    RotaryMode,
    TensorLayout,
    expand_5d,
    check_rotary_mode,
    check_kv_layout,
)

_cache_buf = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device):
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rotary_mode: str = "NONE",
    kv_layout: str = "NHD",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request decode with KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [num_qo_heads, head_dim]
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    kv_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    """
    check_rotary_mode(rotary_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_decode_with_kv_cache(
        q,
        k,
        v,
        tmp,
        getattr(RotaryMode, rotary_mode),
        getattr(TensorLayout, kv_layout),
        sm_scale,
        rope_scale,
        rope_theta,
    )


def batch_decode_with_padded_kv_cache(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    kv_layout: str = "NHD",
    rotary_mode: str = "NONE",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Batch decode with padded KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.batch_decode_with_padded_kv_cache(
        q,
        k_padded,
        v_padded,
        getattr(TensorLayout, kv_layout),
        getattr(RotaryMode, rotary_mode),
        sm_scale,
        rope_scale,
        rope_theta,
        False,
    )[0]


def batch_decode_with_padded_kv_cache_return_lse(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    kv_layout: str = "NHD",
    rotary_mode: str = "NONE",
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""
    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    v_padded : torch.Tensor
        Shape: [batch_size, padded_seq_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, padded_seq_len, head_dim] if HND
    kv_layout: str
        The layout of the input k_padded/v_padded tensors, could be either
        "NHD" or "HND"
    rotary_mode: str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    sm_scale: Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale: Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta: Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    S : torch.Tensor
        Shape: [batch_size, num_heads]
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.batch_decode_with_padded_kv_cache(
        q,
        k_padded,
        v_padded,
        getattr(TensorLayout, kv_layout),
        getattr(RotaryMode, rotary_mode),
        sm_scale,
        rope_scale,
        rope_theta,
        True,
    )


class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for batch_decode_with_paged_kv_cache kernel.

    To accelerate computation, FlashInfer's batch decode operators creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode calls (e.g. different Transformer layers). This wrapper class manages
    the lifecycle of these data structures.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchDecodeWithPagedKVCachePyTorchWrapper(
            getattr(TensorLayout, kv_layout)
        )
        self._paged_kv_indptr = None
        self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        rotary_mode: str = "NONE",
        data_type: Union[str, torch.dtype] = "float16",
    ):
        r"""The begin_forward method should be called before any batch decode calls,
        auxiliary data structures will be created during this call and cached for
        multiple forward calls.
        """
        self._paged_kv_indptr = indptr
        self._paged_kv_indices = indices
        self._paged_kv_last_page_len = last_page_len

        batch_size = len(indptr) - 1
        # NOTE(Zihao): the following tensor acts as placeholder to pass dtype info
        empty_data = torch.empty(
            0,
            dtype=getattr(torch, data_type)
            if isinstance(data_type, str)
            else data_type,
        )
        self._wrapper.begin_forward(
            self._workspace_buffer,
            indptr,
            last_page_len,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            getattr(RotaryMode, rotary_mode),
            empty_data,
        )

    def end_forward(self):
        r"""The end_forward method can clear the cached data structures."""
        self._paged_kv_indptr = None
        self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        rotary_mode: str = "NONE",
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_rotary_mode(rotary_mode)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        return self._wrapper.forward(
            q,
            paged_kv_data,
            self._paged_kv_indptr,
            self._paged_kv_indices,
            self._paged_kv_last_page_len,
            getattr(RotaryMode, rotary_mode),
            rope_scale,
            rope_theta,
            False,
        )[0]

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        rotary_mode: str = "NONE",
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_rotary_mode(rotary_mode)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        return self._wrapper.forward(
            q,
            paged_kv_data,
            self._paged_kv_indptr,
            self._paged_kv_indices,
            self._paged_kv_last_page_len,
            getattr(RotaryMode, rotary_mode),
            rope_scale,
            rope_theta,
            True,
        )
