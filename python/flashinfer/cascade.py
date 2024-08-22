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
from typing import Optional, Tuple
import torch

# mypy: disable-error-code="attr-defined"
try:
    from . import _kernels
except ImportError as e:
    import os
    import logging

    if os.environ.get("BUILD_DOC", "0") == "1":
        _kernels = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e

from .decode import (
    BatchDecodeWithPagedKVCacheWrapper,
)
from .prefill import (
    single_prefill_with_kv_cache_return_lse,
    BatchPrefillWithPagedKVCacheWrapper,
)


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_state(va, sa, vb, sb)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    return _kernels.merge_state(v_a, s_a, v_b, s_b)


def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
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
    mask : Optional[torch.Tensor]
        The boolean mask tensor for whether to merge the state for a corresponding sequence
        or not. Useful for CUDA graphs. If not specified (default), will merge states for
        all sequences.
        shape: ``[seq_len]``

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> v = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_other = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    >>> s_other = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    >>> flashinfer.merge_state_in_place(v, s, v_other, s_other)
    """
    _kernels.merge_state_in_place(v, s, v_other, s_other, mask)


def merge_states(v: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Merge multiple attention states (v, s).

    Parameters
    ----------
    v : torch.Tensor
        The attention output from the KV segments, shape:
        ``[seq_len, num_states, num_heads, head_dim]``.
    s : torch.Tensor
        The logsumexp value from the KV segments, shape:
        ``[seq_len, num_states, num_heads]``, expected
        to be a float32 tensor.

    Returns
    -------
    V : torch.Tensor
        The merged attention output, shape: ``[seq_len, num_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value from the merged KV-segments, shape:
        ``[seq_len, num_heads]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> seq_len = 2048
    >>> num_heads = 32
    >>> head_dim = 128
    >>> num_states = 100
    >>> v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    >>> s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")
    >>> v_merged, s_merged = flashinfer.merge_states(v, s)
    >>> v_merged.shape
    torch.Size([2048, 32, 128])
    >>> s_merged.shape
    torch.Size([2048, 32])
    """
    return _kernels.merge_states(v, s)


class MultiLevelCascadeAttentionWrapper:
    r"""Attention wrapper for memory efficient multi-level cascade inference, this API assumes all
    levels KV-Cache are stored in a unified paged table.

    Check :ref:`our tutorial<page-layout>` for page table layout, and
    `Cascade Inference Query/Output Layout <cascade-qo-indptr-layout>` for query/output layout.

    The idea of cascade inference is introduced in our `blog post <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
    ...     2, workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_kv_num_pages = 512
    >>> unique_kv_num_pages = 128
    >>> total_num_pages = shared_kv_num_pages + unique_kv_num_pages
    >>> shared_kv_page_indices = torch.arange(shared_kv_num_pages).int().to("cuda:0")
    >>> shared_kv_page_indptr = torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0")
    >>> unique_kv_page_indices = torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> shared_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda:0")
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> qo_indptr_arr = [
    ...     torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),  # top-level for shared KV-Cache
    ...     torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")    # bottom-level for unique KV-Cache
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.begin_forward(
    ...     qo_indptr_arr,
    ...     [shared_kv_page_indptr, unique_kv_page_indptr],
    ...     [shared_kv_page_indices, unique_kv_page_indices],
    ...     [shared_kv_last_page_len, unique_kv_last_page_len],
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.forward(q, kv_cache_at_layer[i])
    ...     outputs.append(o)
    ...
    >>> # clear auxiliary data structures
    >>> wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([7, 64, 128])
    """

    def __init__(
        self, num_levels, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        r"""Constructor of :class:`MultiLevelCascadeAttentionWrapper`.

        Parameters
        ----------
        num_levels : int
            The number of levels in the cascade attention.
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        self._batch_prefill_wrappers = [
            BatchPrefillWithPagedKVCacheWrapper(float_workspace_buffer, kv_layout)
            for _ in range(num_levels)
        ]
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffers: list[torch.Tensor],
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        for wrapper, int_workspace_buffer in zip(
            self._batch_prefill_wrappers, int_workspace_buffers
        ):
            wrapper.reset_workspace_buffer(float_workspace_buffer, int_workspace_buffer)

    def begin_forward(
        self,
        qo_indptr_arr: list[torch.Tensor],
        paged_kv_indptr_arr: list[torch.Tensor],
        paged_kv_indices_arr: list[torch.Tensor],
        paged_kv_last_page_len: list[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
    ):
        r"""Create auxiliary data structures for multi-level cascade attention for multiple
        forward calls within the same decode step.

        Parameters
        ----------
        qo_indptr_arr : list[torch.Tensor]
            An array of qo indptr tensors for each level, the array length should be equal to
            the number of levels. Check
            `Cascade Inference Query/Output Layout <cascade-qo-indptr-layout>` for query/output layout.
            The last element of each tensor should be the total number of queries/outputs.
        paged_kv_indptr_arr : list[torch.Tensor]
            An array of paged kv-cache indptr tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_indices_arr : list[torch.Tensor]
            An array of paged kv-cache indices tensors for each level, the array length should be
            equal to the number of levels.
        paged_kv_last_page_len : list[torch.Tensor]
            An array of paged kv-cache last page length tensors for each level, the array length
            should be equal to the number of levels.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.
        """
        for (
            wrapper,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) in zip(
            self._batch_prefill_wrappers,
            qo_indptr_arr,
            paged_kv_indptr_arr,
            paged_kv_indices_arr,
            paged_kv_last_page_len,
        ):
            wrapper.begin_forward(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
            )

    def end_forward(self):
        r"""Clear auxiliary data structures created by :meth:`begin_forward`."""
        for wrapper in self._batch_prefill_wrappers:
            wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        **kwargs,
    ):
        r"""Compute multi-level cascade attention.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        """
        out, lse = self._batch_prefill_wrappers[-1].forward_return_lse(
            q, paged_kv_cache, **kwargs
        )
        # NOTE(Zihao): causal mask should be False for all levels except the last level
        kwargs["causal"] = False
        for wrapper in self._batch_prefill_wrappers[:-1]:
            out_i, lse_i = wrapper.forward_return_lse(q, paged_kv_cache, **kwargs)
            merge_state_in_place(out, lse, out_i, lse_i)

        return out


class BatchDecodeWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with shared-prefix paged kv-cache for batch
    of requests. The shared-prefix KV-Cache was stored in a standalone tensors, and the
    unique KV-Cache of each request was stored in a paged KV-Cache data stucture.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    It is recommended to use :class:`MultiLevelCascadeAttentionWrapper` instead for general
    multi-level cascade inference, where the KV-Cache of each level is stored in a unified
    page table. This API will be deprecated in the future.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> unique_kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> unique_kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> unique_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> unique_kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.begin_forward(
    ...     unique_kv_page_indptr,
    ...     unique_kv_page_indices,
    ...     unique_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     unique_kv_cache = unique_kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = wrapper.forward(q, k_shared, v_shared, unique_kv_cache)
    ...     outputs.append(o)
    ...
    >>> # clear auxiliary data structures
    >>> wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared prefix batch decode attention creates
    some auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        self._batch_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_decode_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

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
    ) -> None:
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
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.


        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        self._batch_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=data_type,
        )

    def end_forward(self) -> None:
        r"""Clear auxiliary data structures created by :meth:`begin_forward`."""
        self._batch_decode_wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
        allow_fp16_qk_reduction=False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
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
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
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
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        V_unique, S_unique = self._batch_decode_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            pos_encoding_mode="NONE",
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared


class BatchPrefillWithSharedPrefixPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with shared-prefix paged kv-cache for
    batch of requests.

    Check :ref:`our tutorial<page-layout>` for paged kv-cache layout.

    It is recommended to use :class:`MultiLevelCascadeAttentionWrapper` instead for general
    multi-level cascade inference, where the KV-Cache of each level is stored in a unified
    page table. This API will be deprecated in the future.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> shared_prefix_len = 8192
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len= torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_k_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> shared_v_data_at_layer = [
    ...     torch.randn(
    ...         shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     k_shared = shared_k_data_at_layer[i]
    ...     v_shared = shared_v_data_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.forward(
    ...         q, k_shared, v_shared, kv_cache, causal=True
    ...     )
    ...     outputs.append(o)
    ...
    s[0].shape>>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([100, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's shared-prefix batch prefill/append attention
    operators creates some auxiliary data structures, these data structures can be
    reused across multiple prefill/append attention calls (e.g. different Transformer
    layers). This wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithSharedPrefixPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        self._batch_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer, kv_layout
        )
        self._kv_layout = kv_layout

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._batch_prefill_wrapper.reset_workspace_buffer(
            float_workspace_buffer, int_workspace_buffer
        )

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
    ) -> None:
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
        head_dim : int
            The dimension of the heads.
        page_size : int
            The page size of the paged kv-cache.

        Note
        ----
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
            head_dim,
            page_size,
        )

    def end_forward(self) -> None:
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        self._batch_prefill_wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k_shared: torch.Tensor,
        v_shared: torch.Tensor,
        unique_kv_cache: torch.Tensor,
        causal: bool = True,
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
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
        unique_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The request-independent suffix paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.

        causal : bool
            Whether to apply causal mask on the attention matrix.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_heads, head_dim]``.

        See Also
        --------
        MultiLevelCascadeAttentionWrapper
        """
        V_shared, S_shared = single_prefill_with_kv_cache_return_lse(
            q,
            k_shared,
            v_shared,
            causal=False,
            pos_encoding_mode="NONE",
            kv_layout=self._kv_layout,
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        V_unique, S_unique = self._batch_prefill_wrapper.forward_return_lse(
            q,
            unique_kv_cache,
            causal=causal,
            pos_encoding_mode="NONE",
            allow_fp16_qk_reduction=allow_fp16_qk_reduction,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        merge_state_in_place(V_shared, S_shared, V_unique, S_unique)
        return V_shared
