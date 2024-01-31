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
    kv_layout: str = "NHD",
    rotary_mode: str = "NONE",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    causal : bool
        Whether to apply causal mask to the attention matrix.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    rotary_mode : str
        Whether to apply RoPE on-the-fly inside attention kernels, could be
        ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.

    Returns
    -------
    torch.Tensor
        The attention output, shape: ``[qo_len, num_qo_heads, head_dim]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>> q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,
            allow_fp16_qk_reduction=True)
    >>> o.shape
    torch.Size([128, 32, 128])

    Notes
    -----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
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
    kv_layout: str = "NHD",
    rotary_mode: str = "NONE",
    allow_fp16_qk_reduction: bool = False,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Prefill/Append attention with KV cache for single request, return attention
    output and logsumexp of attention scores.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    causal : bool
        Whether to apply causal mask to the attention matrix.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    rotary_mode : str
        Whether to apply RoPE on-the-fly inside attention kernels, could be
        ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.

    Returns
    -------
    V : torch.Tensor
        The attention output, shape: ``[qo_len, num_qo_heads, head_dim]``.
    S : torch.Tensor
        The logsumexp value, shape: ``[qo_len, num_qo_heads]``

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>> q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> V, S = flashinfer.single_prefill_with_kv_cache_return_lse(q, k, v, causal=True)
    >>> V.shape
    torch.Size([128, 32, 128])
    >>> S.shape
    torch.Size([128, 32])

    Notes
    -----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
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
    r"""Wrapper class for prefill/append attention with paged kv-cache for batch of
    requests.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB, the device of the workspace buffer should be the
            same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
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
        r"""Reset the workspace buffer.

        Parameters
        ----------
        new_workspace_buffer : torch.Tensor
            The new workspace buffer, the device of the new workspace buffer should
            be the same as the device of the input tensors.
        """
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
        r"""Create auxiliary data structures for batch prefill/append attention for
        multiple forward calls within the same prefill/append step.

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
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._wrapper.begin_forward(
            self._workspace_buffer, qo_indptr, batch_size, num_qo_heads, num_kv_heads
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
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
        r"""Compute batch prefill/append attention between query and paged kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        paged_kv_data : torch.Tensor
            A 5-D tensor of the reserved paged kv-cache data, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]``
            if :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]``
            if :attr:`kv_layout` is ``HND``.
        causal : bool
            Whether to apply causal mask to the attention matrix.
        rotary_mode : str
            Whether to apply RoPE on-the-fly inside attention kernels, could be
            ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
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
        torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        """
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
        r"""Compute batch prefill/append attention paged kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        paged_kv_data : torch.Tensor
            A 5-D tensor of the reserved paged kv-cache data, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]``
            if :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        causal : bool
            Whether to apply causal mask to the attention matrix.
        rotary_mode : str
            Whether to apply RoPE on-the-fly inside attention kernels, could be
            ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
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
            The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        S : torch.Tensor
            The logsumexp of attention output, shape:
            ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        """
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
    r"""Wrapper class for prefill/append attention with ragged (tensor) kv-cache for
    batch of requests.

    Check :ref:`our tutorial<ragged-layout>` for ragged kv-cache layout.

    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        r"""Constructor of :class:`BatchDecodeWithRaggedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB, the device of the workspace buffer should be the
            same as the device of the input tensors.
        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithRaggedKVCachePyTorchWrapper(
            getattr(TensorLayout, kv_layout)
        )
        self._qo_indptr = None
        self._kv_indptr = None

    def reset_workspace_buffer(self, new_workspace_buffer: torch.Tensor):
        r"""Reset the workspace buffer.

        Parameters
        ----------
        new_workspace_buffer : torch.Tensor
            The new workspace buffer, the device of the new workspace buffer should
            be the same as the device of the input tensors.
        """
        self._workspace_buffer = new_workspace_buffer

    def begin_forward(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
    ):
        r"""Create auxiliary data structures for batch prefill/append attention for
        multiple forward calls within the same prefill/append step.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr : torch.Tensor
            The indptr of the key/value tensor, shape: ``[batch_size + 1]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.

        Notes
        -----
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        batch_size = len(qo_indptr) - 1
        self._qo_indptr = qo_indptr
        self._kv_indptr = kv_indptr
        self._wrapper.begin_forward(
            self._workspace_buffer, qo_indptr, batch_size, num_qo_heads, num_kv_heads
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
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
        r"""Compute batch prefill/append attention between query and kv-cache stored in
        ragged tensor.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        causal : bool
            Whether to apply causal mask to the attention matrix.
        rotary_mode : str
            Whether to apply RoPE on-the-fly inside attention kernels, could be
            ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
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
        torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        """
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
        r"""Compute batch prefill/append attention between query and kv-cache stored in
        ragged tensor. Return attention output and logsumexp of attention scores.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        causal : bool
            Whether to apply causal mask to the attention matrix.
        rotary_mode : str
            Whether to apply RoPE on-the-fly inside attention kernels, could be
            ``NONE`` or ``LLAMA`` (LLAMA style rotary embedding).
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        V : torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        S : torch.Tensor
            The logsumexp of attention output, shape:
            ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        """
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
