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

from .decode import (
    batch_decode_with_padded_kv_cache_return_lse,
    BatchDecodeWithPagedKVCacheWrapper,
)
from .prefill import (
    single_prefill_with_kv_cache_return_lse,
    BatchPrefillWithPagedKVCacheWrapper,
)
from .utils import (
    expand_5d,
    check_rotary_mode,
    check_kv_layout,
    RotaryMode,
    TensorLayout,
)


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    r"""Merge the attention output (V) and the logsumexp value (S) from the two KV-segments.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment A.
        Shape: [seq_len, num_heads, head_dim]
    s_a : torch.Tensor
        The logsumexp value from the KV segment A. Expected to be a float32 tensor.
        Shape: [seq_len, num_heads]
    v_b : torch.Tensor
        The attention output from the KV segment B.
        Shape: [seq_len, num_heads, head_dim]
    s_b : torch.Tensor
        The logsumexp value from the KV segment B. Expected to be a float32 tensor.
        Shape: [seq_len, num_heads]

    Returns
    -------
    V : torch.Tensor
        The merged attention output (equivalent to attention with merged KV-segment [A: B]).
        Shape: [batch_size, num_heads, head_dim]
    S : torch.Tensor
        The logsumexp value from the merged KV-segment [A: B].
        Shape: [batch_size, num_heads]
    """
    return _kernels.merge_state(v_a, s_a, v_b, s_b)


def merge_state_in_place(
    v: torch.Tensor, s: torch.Tensor, v_other: torch.Tensor, s_other: torch.Tensor
):
    r"""Merge the self-attention state (v, s) with another state (v_other, s_other) in-place.

    Parameters
    ----------
    v : torch.Tensor
        The partial v to be updated in-place.
        Shape: (seq_len, num_heads, head_dim)
    s : torch.Tensor
        The partial logsumexpr value to be updated in-place, expected to be a float32 tensor.
        Shape: (seq_len, num_heads)
    v_other : torch.Tensor
        The other v to be merged.
        Shape: (seq_len, num_heads, head_dim)
    s_other : torch.Tensor
        The other logsumexp value to be merged, expected to be a float32 tensor.
        Shape: (seq_len, num_heads)
    """
    _kernels.merge_state_in_place(v, s, v_other, s_other)


def merge_states(v: torch.Tensor, s: torch.Tensor):
    r"""Merge the attention output (V) and the logsumexp value (S) from multiple KV-segments.

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments.
        Shape: [seq_len, num_kv_segments, num_heads, head_dim]
    s : torch.Tensor
        The logsumexp value from the KV segments.
        Shape: [seq_len, num_kv_segments, num_heads]

    Returns
    -------
    V : torch.Tensor
        The merged attention output.
        Shape: [seq_len, num_heads, head_dim]
    S : torch.Tensor
        The logsumexp value from the merged KV-segments.
        Shape: [seq_len, num_heads]
    """
    return _kernels.merge_states(v, s)


def batch_decode_with_shared_prefix_padded_kv_cache(
    q: torch.Tensor,
    k_shared: torch.Tensor,
    v_shared: torch.Tensor,
    k_unique: torch.Tensor,
    v_unique: torch.Tensor,
    kv_layout: str = "NHD",
    allow_fp16_qk_reduction=False,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Batch decode with shared prefix padded KV cache.

    Parameters
    ----------
    q : torch.Tensor
        Shape: [batch_size, num_qo_heads, head_dim]
    k_shared : torch.Tensor
        Shape: [shared_prefix_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, shared_prefix_len, head_dim] if HND
    v_shared : torch.Tensor
        Shape: [shared_prefix_len, num_kv_heads, head_dim] if NHD
               [num_kv_heads, shared_prefix_len, head_dim] if HND
    k_unique : torch.Tensor
        Shape: [batch_size, unique_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, unique_len, head_dim] if HND
    v_unique : torch.Tensor
        Shape: [batch_size, unique_len, num_kv_heads, head_dim] if NHD
               [batch_size, num_kv_heads, unique_len, head_dim] if HND
    kv_layout : str
        The layout of the input k/v tensors, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (could be significantly faster for GeForce cards, at
        the cost of slight precision loss).
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to 1 / sqrt(head_dim)
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    V : torch.Tensor
        Shape: [batch_size, num_heads, head_dim]
    """
    check_kv_layout(kv_layout)
    V_shared, S_shared = single_prefill_with_kv_cache_return_lse(
        q,
        k_shared,
        v_shared,
        causal=False,
        rotary_mode="NONE",
        kv_layout=kv_layout,
        allow_fp16_qk_reduction=allow_fp16_qk_reduction,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    V_unique, S_unique = batch_decode_with_padded_kv_cache_return_lse(
        q,
        k_unique,
        v_unique,
        kv_layout=kv_layout,
        rotary_mode="NONE",
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )

    merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
    return V_shared


class BatchDecodeWithSharedPrefixPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        self._batch_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._batch_decode_wrapper.reset_workspace_buffer(new_workspace_buffer)

    def begin_forward(
        self,
        unique_kv_indptr: torch.Tensor,
        unique_kv_indices: torch.Tensor,
        unique_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        data_type: str = "float16",
    ):
        self._batch_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            rotary_mode="NONE",
            data_type=data_type,
        )

    def end_forward(self):
        self._batch_decode_wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_data: torch.Tensor,
        allow_fp16_qk_reduction=False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        V_shared, S_shared = single_prefill_with_kv_cache_return_lse(
            q,
            k_shared,
            v_shared,
            causal=False,
            rotary_mode="NONE",
            kv_layout=self._kv_layout,
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        V_unique, S_unique = self._batch_decode_wrapper.forward_return_lse(
            q,
            unique_kv_data,
            rotary_mode="NONE",
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared


class BatchPrefillWithSharedPrefixPagedKVCacheWrapper:
    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        self._batch_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        self._batch_prefill_wrapper.reset_workspace_buffer(new_workspace_buffer)

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
    ):
        self._batch_prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
        )

    def end_forward(self):
        self._batch_prefill_wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_data: torch.Tensor,
        causal: bool = True,
        allow_fp16_qk_reduction: bool = False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        V_shared, S_shared = single_prefill_with_kv_cache_return_lse(
            q,
            k_shared,
            v_shared,
            causal=False,
            rotary_mode="NONE",
            kv_layout=self._kv_layout,
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        V_unique, S_unique = self._batch_prefill_wrapper.forward_return_lse(
            q,
            unique_kv_data,
            causal=causal,
            rotary_mode="NONE",
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared
