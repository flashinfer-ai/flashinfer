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
    r"""Merge the attention output ``V`` and the logsumexp value ``S`` from the two
    KV-segments.
    Check :ref:`our tutorial <recursive-attention>` on the mathematical details.

    Parameters
    ----------
    v_a : torch.Tensor
        The attention output from the KV segment ``A``, shape:
        ``[seq_len, num_heads, head_dim]``.
    s_a : torch.Tensor
        The logsumexp value from the KV segment ``A``. expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``.
    v_b : torch.Tensor
        The attention output from the KV segment ``B``,
        shape: ``[seq_len, num_heads, head_dim]``.
    s_b : torch.Tensor
        The logsumexp value from the KV segment ``B``, expected to be a float32 tensor,
        shape: ``[seq_len, num_heads]``

    Returns
    -------
    V : torch.Tensor
        The merged attention output (equivalent to attention with merged KV-segment
        ``[A: B]``), shape: ``[batch_size, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segment ``[A: B]``, shape:
        ``[batch_size, num_heads]``.
    """
    return _kernels.merge_state(v_a, s_a, v_b, s_b)


def merge_state_in_place(
    v: torch.Tensor, s: torch.Tensor, v_other: torch.Tensor, s_other: torch.Tensor
):
    r"""Merge the self-attention state ``(v, s)`` with another state
    ``(v_other, s_other)`` in-place.

    Parameters
    ----------
    v : torch.Tensor
        The partial attention output to be updated in-place, shape:
        ``(seq_len, num_heads, head_dim)``.
    s : torch.Tensor
        The partial logsumexpr value to be updated in-place, expected to be a float32
        tensor, shape: ``(seq_len, num_heads)``.
    v_other : torch.Tensor
        The other attention output to be merged, shape:
        ``(seq_len, num_heads, head_dim)``.
    s_other : torch.Tensor
        The other logsumexp value to be merged, expected to be a float32 tensor,
        shape: ``(seq_len, num_heads)``.
    """
    _kernels.merge_state_in_place(v, s, v_other, s_other)


def merge_states(v: torch.Tensor, s: torch.Tensor):
    r"""Merge the attention output ``V`` and the logsumexp value ``S`` from multiple
    KV-segments.

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments, shape:
        ``[seq_len, num_kv_segments, num_heads, head_dim]``.
    s : torch.Tensor
        The logsumexp value from the KV segments, shape:
        ``[seq_len, num_kv_segments, num_heads]``, expected
        to be a float32 tensor.

    Returns
    -------
    V : torch.Tensor
        The merged attention output, shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segments, shape:
        ``[seq_len, num_heads]``.
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
    r"""Decode attention between queries and shared prefix kv-cache for batch of
    requests.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
    k_shared : torch.Tensor
        The shared prefix key tensor, shape:
        ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
        or ``[num_kv_heads, shared_prefix_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v_shared : torch.Tensor
        The shared prefix value tensor, shape:
        ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
        or ``[num_kv_heads, shared_prefix_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    k_unique : torch.Tensor
        The request-independent suffix key tensor, shape:
        ``[batch_size, unique_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
        ``NHD``, or ``[batch_size, num_kv_heads, unique_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    v_unique : torch.Tensor
        The request-independent suffix value tensor, shape:
        ``[batch_size, unique_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
        ``NHD``, or ``[batch_size, num_kv_heads, unique_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the kv-cache, could be either "NHD" or "HND".
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.

    Returns
    -------
    V : torch.Tensor
        The attention output, shape: ``[batch_size, num_heads, head_dim]``
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
    r"""Wrapper class for decode attention with shared-prefix paged kv-cache for batch
    of requests.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    Note
    ----
    To accelerate computation, FlashInfer's shared prefix batch decode attention creates
    some auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
     manages the lifecycle of these data structures.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        self._batch_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        r"""Reset the workspace buffer.

        Parameters
        ----------
        new_workspace_buffer : torch.Tensor
            The new workspace buffer, the device of the new workspace buffer should
            be the same as the device of the input tensors.
        """
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
        r"""Create auxiliary data structures for shared-prefix batch decode for multiple
        forward calls within the same decode step.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        rotary_mode : str
            Whether to apply RoPE on-the-fly inside attention kernels, could be
            ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls,
        auxiliary data structures will be created during this call and cached for
        multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention<https://arxiv.org/abs/2305.13245>`_.
        """
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
        r"""Clear auxiliary data structures created by :meth:`begin_forward`."""
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
        r"""Compute batch decode attention between queries and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared : torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_data : torch.Tensor
            A 5-D tensor of paged kv-cache data storing the request-independent suffix
            key and value tensors, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[batch_size, num_heads, head_dim]``
        """
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
    r"""Wrapper class for prefill/append attention with shared-prefix paged kv-cache for
    batch of requests.

    Check :ref:`our tutorial<page-layout>` for paged kv-cache layout.

    Note
    ----
    To accelerate computation, FlashInfer's shared-prefix batch prefill/append attention
    operators creates some auxiliary data structures, these data structures can be
    reused across multiple prefill/append attention calls (e.g. different Transformer
    layers). This wrapper class manages the lifecycle of these data structures.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        r"""Constructor of :class:`BatchDecodeWithSharedPrefixPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB, the device of the workspace buffer should be the
            same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        self._batch_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        r"""Reset the workspace buffer.

        Parameters
        ----------
        new_workspace_buffer : torch.Tensor
            The new workspace buffer, the device of the new workspace buffer should
            be the same as the device of the input tensors.
        """
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
        r"""Create auxiliary data structures for shared-prefix batch prefill/append
        attention for multiple forward calls within the same prefill/append step.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.

        Notes
        -----
        The :meth:`begin_forward` method should be called before any :meth:`forward`
        or :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """

        self._batch_prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
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
        r"""Compute batch prefill/append attention between query and shared-prefix paged
        kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        k_shared : torch.Tensor
            The shared prefix key tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        v_shared ; torch.Tensor
            The shared prefix value tensor, shape:
            ``[shared_prefix_len, num_kv_heads, head_dim]`` if :attr:`kv_layout` is
            ``NHD``, or ``[num_kv_heads, shared_prefix_len, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        unique_kv_data : torch.Tensor
            A 5-D tensor of paged kv-cache data storing the request-independent suffix
            key and value tensors, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        causal : bool
            Whether to apply causal mask on the attention matrix.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_heads, head_dim]``.
        """
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
