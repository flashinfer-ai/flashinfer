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
from typing import Optional

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


def _get_cache_buf(name: str, bytes: int, device: torch.device):
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    rotary_mode: str = "NONE",
    kv_layout: str = "NHD",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request prefill with KV cache kernel.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    causal : bool
        Whether to apply causal mask to the attention matrix.
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    kv_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (could be significantly faster for GeForce cards, at
        the cost of precision loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    """
    check_rotary_mode(rotary_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_prefill_with_kv_cache(
        q,
        k,
        v,
        tmp,
        causal,
        getattr(TensorLayout, kv_layout),
        getattr(RotaryMode, rotary_mode),
        allow_fp16_qk_reduction,
        rope_scale,
        rope_theta,
        False,
    )[0]


def single_prefill_with_kv_cache_return_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    rotary_mode: str = "NONE",
    kv_layout: str = "NHD",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Single request prefill with KV cache kernel, return logsumexp value.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    k : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    v : torch.Tensor
        Shape: [kv_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, kv_len, head_dim] if HND
    causal : bool
        Whether to apply causal mask to the attention matrix.
    rotary_mode : str
        Whether to apply rotary embeddings inside attention kernels, could be
        "NONE" or "LLAMA".
    kv_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (could be significantly faster for GeForce cards, at
        the cost of precision loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        The attention output.
        Shape: [qo_len, num_qo_heads, head_dim] if NHD
               [num_qo_heads, qo_len, head_dim] if HND
    S : torch.Tensor
        The logsumexp value.
        Shape: [qo_len, num_qo_heads]
    """
    check_rotary_mode(rotary_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_return_lse_tmp", 8 * 1024 * 1024, q.device
    )
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    return _kernels.single_prefill_with_kv_cache(
        q,
        k,
        v,
        tmp,
        causal,
        getattr(TensorLayout, kv_layout),
        getattr(RotaryMode, rotary_mode),
        allow_fp16_qk_reduction,
        rope_scale,
        rope_theta,
        True,
    )


class BatchPrefillWithPagedKVCacheWrapper:
    r"""Wrapper class of batch_prefill_with_paged_kv_cache kernel."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithPagedKVCachePyTorchWrapper(
            getattr(TensorLayout, kv_layout)
        )
        self._qo_indptr = None
        self._paged_kv_indptr = None
        self._paged_kv_indices = None
        self._paged_kv_last_page_len = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
    ):
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._wrapper.begin_forward(
            self._workspace_buffer, qo_indptr, batch_size, num_qo_heads, num_kv_heads
        )

    def end_forward(self):
        self._qo_indptr = None
        self._paged_kv_indptr = None
        self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        causal: bool = True,
        rotary_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
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
            self._qo_indptr,
            paged_kv_data,
            self._paged_kv_indptr,
            self._paged_kv_indices,
            self._paged_kv_last_page_len,
            causal,
            getattr(RotaryMode, rotary_mode),
            allow_fp16_qk_reduction,
            rope_scale,
            rope_theta,
            False,
        )[0]

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        causal: bool = True,
        rotary_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
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
            self._qo_indptr,
            paged_kv_data,
            self._paged_kv_indptr,
            self._paged_kv_indices,
            self._paged_kv_last_page_len,
            causal,
            getattr(RotaryMode, rotary_mode),
            allow_fp16_qk_reduction,
            rope_scale,
            rope_theta,
            True,
        )


class BatchPrefillWithRaggedKVCacheWrapper:
    r"""Wrapper class of batch_prefill_with_ragged_kv_cache kernel."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithRaggedKVCachePyTorchWrapper(
            getattr(TensorLayout, kv_layout)
        )
        self._qo_indptr = None
        self._kv_indptr = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
    ):
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._kv_indptr = kv_indptr
        self._wrapper.begin_forward(
            self._workspace_buffer, qo_indptr, batch_size, num_qo_heads, num_kv_heads
        )

    def end_forward(self):
        self._qo_indptr = None
        self._kv_indptr = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        rotary_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_rotary_mode(rotary_mode)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        return self._wrapper.forward(
            q,
            self._qo_indptr,
            k,
            v,
            self._kv_indptr,
            causal,
            getattr(RotaryMode, rotary_mode),
            allow_fp16_qk_reduction,
            rope_scale,
            rope_theta,
            False,
        )[0]

    def forward_return_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        rotary_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_rotary_mode(rotary_mode)
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        return self._wrapper.forward(
            q,
            self._qo_indptr,
            k,
            v,
            self._kv_indptr,
            causal,
            getattr(RotaryMode, rotary_mode),
            allow_fp16_qk_reduction,
            rope_scale,
            rope_theta,
            True,
        )
