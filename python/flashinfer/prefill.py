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
import logging

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

from .utils import (
    PosEncodingMode,
    TensorLayout,
    expand_5d,
    check_pos_encoding_mode,
    check_kv_layout,
    is_float8,
)


_cache_buf = {}


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
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    logits_cap: bool = False,
    allow_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
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
    custom_mask : Optional[torch.Tensor]
        The custom mask tensor, shape: ``[qo_len, kv_len]``.
        If provided, the custom mask will be added to the attention matrix before
        softmax and after scaling, and the :attr:`causal` parameter will be ignored.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    logits_cap : bool
        Whether to apply logits cap to pre-attention logits.
        If ``True``, the logits will be capped according to formula (proposed in
        Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
        Defaults to ``False``.
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.
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
    >>> mask = torch.triu(
    >>>     torch.full((qo_len, kv_len), -float("inf"), dtype=torch.float32, device="cuda:0"),
    >>>     diagonal=(kv_len - qo_len + 1),
    >>> )
    >>> mask
    tensor([[0., 0., 0.,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        ...,
        [0., 0., 0.,  ..., 0., -inf, -inf],
        [0., 0., 0.,  ..., 0., 0., -inf],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
    >>> o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_custom, rtol=1e-3, atol=1e-3)
    True

    Notes
    -----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    check_pos_encoding_mode(pos_encoding_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if custom_mask is not None:
        return _kernels.single_prefill_with_kv_cache_custom_mask(
            q,
            k,
            v,
            custom_mask,
            tmp,
            TensorLayout[kv_layout].value,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            False,
        )[0]
    else:
        return _kernels.single_prefill_with_kv_cache(
            q,
            k,
            v,
            tmp,
            causal,
            TensorLayout[kv_layout].value,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            False,
        )[0]


def single_prefill_with_kv_cache_return_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    logits_cap: bool = False,
    allow_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
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
    custom_mask : Optional[torch.Tensor]
        The custom_mask tensor, shape: ``[qo_len, kv_len]``.
        If provided, the custom mask will be added to the attention matrix before
        softmax and after scaling, and the :attr:`causal` parameter will be ignored.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    logits_cap : bool
        Whether to apply logits cap to pre-attention logits.
        If ``True``, the logits will be capped according to formula (proposed in
        Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
        Defaults to ``False``.
    allow_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.
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
    >>> mask = torch.triu(
    >>>     torch.full((qo_len, kv_len), -float("inf"), dtype=torch.float32, device="cuda:0"),
    >>>     diagonal=(kv_len - qo_len + 1),
    >>> )
    >>> mask
    tensor([[0., 0., 0.,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        ...,
        [0., 0., 0.,  ..., 0., -inf, -inf],
        [0., 0., 0.,  ..., 0., 0., -inf],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
    >>> V_custom, S_custom = flashinfer.single_prefill_with_kv_cache_return_lse(q, k, v, custom_mask=mask)
    >>> torch.allclose(V, V_custom, rtol=1e-3, atol=1e-3)
    True
    >>> torch.allclose(S, S_custom, rtol=1e-3, atol=1e-3)
    True

    Notes
    -----
    Please refer to the :ref:`tutorial <recursive-attention>` for a detailed
    explanation of the log-sum-exp function and attention states.

    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    check_pos_encoding_mode(pos_encoding_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_return_lse_tmp", 8 * 1024 * 1024, q.device
    )
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if is_float8(q):
        logging.warning(
            "Our current prefill kernel implementation needs f16 input, the f8 inputs "
            " are casted to f16, which could result in performance degradation."
        )
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    if custom_mask is not None:
        return _kernels.single_prefill_with_kv_cache_custom_mask(
            q,
            k,
            v,
            custom_mask,
            tmp,
            TensorLayout[kv_layout].value,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            True,
        )
    else:
        return _kernels.single_prefill_with_kv_cache(
            q,
            k,
            v,
            tmp,
            causal,
            TensorLayout[kv_layout].value,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            allow_fp16_qk_reduction,
            sm_scale,
            rope_scale,
            rope_theta,
            True,
        )


def _compute_page_qk_indptr(
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
):
    if len(qo_indptr) != len(paged_kv_indptr):
        raise ValueError(
            "The length of qo_indptr and paged_kv_indptr should be the same."
        )
    qk_indptr = torch.empty_like(qo_indptr)
    qk_indptr[0] = 0
    qk_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1])
        * (
            (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) * page_size
            + paged_kv_last_page_len
        ),
        0,
    )
    return qk_indptr


class BatchPrefillWithPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with paged kv-cache for batch of
    requests.

    Check :ref:`our tutorial<page-layout>` for page table layout.

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
    >>> # allocate 16MB workspace buffer
    >>> workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> kv_data_at_layer = torch.randn(
    ...     num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ... )
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
    ...     q = q_at_layer[i]
    ...     kv_data = kv_data_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.forward(
    ...         q, kv_data, causal=True
    ...     )
    ...     outputs.append(o)
    ...
    >>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.triu(
    ...         torch.full((qo_len[i], kv_len[i]), -float("inf"), dtype=torch.float32, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i] + 1),
    ...     )
    ...     mask_arr.append(mask_i)
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     mask
    ... )
    >>> outputs_custom_mask = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_data = kv_data_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.forward(
    ...         q, kv_data
    ...     )
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...
    >>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()


    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        enable_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        qk_indptr_buf: Optional[torch.Tensor] = None,
    ):
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        enable_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True``.

        paged_kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indptr`` array, the size of this
            buffer should be ``[batch_size + 1]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True``.

        paged_kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indices`` array, should be large
            enough to store the maximum possible size of the ``paged_kv_indices`` array during
            the lifetime of the wrapper. This argument is only effective when ``enable_cuda_graph``
            is ``True``.

        paged_kv_last_page_len_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_last_page_len`` array, the size of
            the buffer should be ``[batch_size]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved buffer to store the custom mask tensor, should be large enough to
            store the maximum possible size of the custom mask tensor during the lifetime of the
            wrapper. This argument is only effective when ``enable_cuda_graph`` is set to ``True``
            and the custom mask will be used in attention computation.

        qk_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qk_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True`` and the custom
            mask will be used in attention computation.
        """
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value,
            enable_cuda_graph,
        )
        if enable_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indptr_buf):
                raise ValueError(
                    "paged_kv_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buf):
                raise ValueError(
                    "paged_kv_indices_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buf):
                raise ValueError(
                    "paged_kv_last_page_len_buf should be a torch.Tensor in CUDA graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                )
            if len(paged_kv_last_page_len_buf) != self._fixed_batch_size:
                raise ValueError(
                    "The length of paged_kv_last_page_len_buf should be batch_size."
                )
            # NOTE(Zihao): do not check custom_mask_buf and qk_indptr_buf here, as they are optional
        else:
            self._fixed_batch_size = 0

        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._custom_mask_buf = custom_mask_buf
        self._qk_indptr_buf = qk_indptr_buf

    @property
    def is_cuda_graph_enabled(self):
        return self._wrapper.is_cuda_graph_enabled()

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
        head_dim: int,
        page_size: int,
        custom_mask: Optional[torch.Tensor] = None,
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
        head_dim : int
            The dimension of the heads.
        page_size : int
            The size of each page in the paged kv-cache.
        custom_mask : Optional[torch.Tensor]
            The flattened mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            If provided, the custom mask will be added to the attention matrix before softmax
            and after scaling. The mask tensor should be in the same device as the input tensors.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

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

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    "cuda graph mode, the runtime batch size {} mismatches the batch size {} "
                    " set during initialization.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(paged_kv_indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The length of paged_kv_indices exceeds the allocated buffer size."
                )

            self._qo_indptr_buf.copy_(qo_indptr)
            self._paged_kv_indptr_buf.copy_(paged_kv_indptr)
            self._paged_kv_indices_buf[: len(paged_kv_indices)] = paged_kv_indices
            self._paged_kv_last_page_len_buf.copy_(paged_kv_last_page_len)

            if custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._qk_indptr_buf):
                    raise ValueError(
                        "qk_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                self._custom_mask_buf[: len(custom_mask)] = custom_mask
                # NOTE(Zihao): qk_indptr has the same length as qo_indptr
                self._qk_indptr_buf.copy_(
                    _compute_page_qk_indptr(
                        qo_indptr,
                        paged_kv_indptr,
                        paged_kv_last_page_len,
                        page_size,
                    )
                )
        else:
            self._qo_indptr_buf = qo_indptr
            self._paged_kv_indptr_buf = paged_kv_indptr
            self._paged_kv_indices_buf = paged_kv_indices
            self._paged_kv_last_page_len_buf = paged_kv_last_page_len
            if custom_mask is not None:
                self._custom_mask = custom_mask
                self._qk_indptr = _compute_page_qk_indptr(
                    qo_indptr,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                    page_size,
                )
        self._wrapper.begin_forward(
            self._workspace_buffer,
            qo_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        if not self.is_cuda_graph_enabled:
            self._qo_indptr = None
            self._paged_kv_indptr = None
            self._paged_kv_indices = None
            self._paged_kv_last_page_len = None
            self._custom_mask = None
            self._qk_indptr = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        causal: bool = True,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
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
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`begin_forward`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        logits_cap : bool
            Whether to apply logits cap to pre-attention logits, 
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
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
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            paged_kv_data = paged_kv_data.to(torch.float16)

        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        if self._custom_mask_buf is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr_buf,
                paged_kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                causal,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                False,
            )[0]
        else:
            return self._wrapper.forward_custom_mask(
                q,
                self._qo_indptr_buf,
                paged_kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                self._custom_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                False,
            )[0]

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        causal: bool = True,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
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
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        logits_cap : bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
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
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            paged_kv_data = paged_kv_data.to(torch.float16)

        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        if self._custom_mask_buf is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr_buf,
                paged_kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                causal,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                True,
            )
        else:
            return self._wrapper.forward(
                q,
                self._qo_indptr_buf,
                paged_kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                self._custom_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                True,
            )


def _compute_qk_indptr(qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
    if len(qo_indptr) != len(kv_indptr):
        raise ValueError("The length of qo_indptr and kv_indptr should be the same.")
    qk_indptr = torch.empty_like(qo_indptr)
    qk_indptr[0] = 0
    qk_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1]) * (kv_indptr[1:] - kv_indptr[:-1]),
        0,
    )
    return qk_indptr


class BatchPrefillWithRaggedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with ragged (tensor) kv-cache for
    batch of requests.

    Check :ref:`our tutorial<ragged-layout>` for ragged kv-cache layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> # allocate 16MB workspace buffer
    >>> workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_kv = 100
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_indptr = qo_indptr.clone()
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.forward(
    ...         q, k, v, causal=True
    ...     )
    ...     outputs.append(o)
    ...
    >>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.triu(
    ...         torch.full((qo_len[i], kv_len[i]), -float("inf"), dtype=torch.float32, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i] + 1),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.begin_forward(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     mask
    ... )
    >>> outputs_custom_mask = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.forward(q, k, v)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...
    >>> # clear auxiliary data structures
    >>> prefill_wrapper.end_forward()


    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        enable_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        qk_indptr_buf: Optional[torch.Tensor] = None,
    ):
        r"""Constructor of :class:`BatchDecodeWithRaggedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        enable_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in the provided buffers.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True``.

        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the custom mask tensor, should be large
            enough to store the maximum possible size of the custom mask tensor during the
            lifetime of the wrapper. This argument is only effective when ``enable_cuda_graph``
            is ``True`` and custom mask will be used in attention computation.

        qk_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``qk_indptr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``enable_cuda_graph`` is ``True`` and custom mask
            will be used in attention computation.
        """
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithRaggedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value,
            enable_cuda_graph,
        )
        if enable_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            if not torch.is_tensor(kv_indptr_buf):
                raise ValueError(
                    "kv_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf)
            if len(kv_indptr_buf) != self._fixed_batch_size:
                raise ValueError(
                    "The length of kv_indptr_buf ({}) should be the same as qo_indptr_buf ({}).".format(
                        len(kv_indptr_buf), self._fixed_batch_size
                    )
                )
            # NOTE(Zihao): do not check custom_mask_buf and qk_indptr_buf here,
            # as they may not be used.

        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._custom_mask_buf = custom_mask_buf
        self._qk_indptr_buf = qk_indptr_buf

    @property
    def is_cuda_graph_enabled(self):
        return self._wrapper.is_cuda_graph_enabled()

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
        head_dim: int,
        custom_mask: Optional[torch.Tensor] = None,
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
        head_dim : int
            The dimension of the heads.
        custom_mask : Optional[torch.Tensor]
            The flattened mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            If provided, the custom mask will be added to the attention matrix before softmax
            and after scaling. The mask tensor should be in the same device as the input tensors.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

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
        if len(kv_indptr) != batch_size + 1:
            raise ValueError(
                "The kv_indptr length should be equal to qk_indptr length."
            )

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            self._qo_indptr_buf.copy_(qo_indptr)
            self._kv_indptr_buf.copy_(kv_indptr)
            if custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._qk_indptr_buf):
                    raise ValueError(
                        "qk_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in the attention computation."
                    )
                self._custom_mask_buf[: len(custom_mask)] = custom_mask
                self._qk_indptr_buf.copy_(_compute_qk_indptr(qo_indptr, kv_indptr))
        else:
            self._qo_indptr_buf = qo_indptr
            self._kv_indptr_buf = kv_indptr
            if custom_mask is not None:
                self._custom_mask_buf = custom_mask
                self._qk_indptr_buf = _compute_qk_indptr(qo_indptr, kv_indptr)
        self._wrapper.begin_forward(
            self._workspace_buffer,
            qo_indptr,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )

    def end_forward(self):
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        if not self.is_cuda_graph_enabled:
            self._qo_indptr = None
            self._kv_indptr = None
            self._custom_mask = None
            self._qk_indptr = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
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
            This argument is ignored if ``mask`` is provided in :meth:`begin_forward`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        logits_cap : bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
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
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        if self._custom_mask_buf is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr_buf,
                k,
                v,
                self._kv_indptr_buf,
                causal,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                False,
            )[0]
        else:
            return self._wrapper.forward_custom_mask(
                q,
                self._qo_indptr_buf,
                k,
                v,
                self._kv_indptr_buf,
                self._custom_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
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
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        allow_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
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
            This argument is ignored if ``mask`` is provided in :meth:`begin_forward`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        logits_cap : bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        allow_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
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
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        if self._custom_mask is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr_buf,
                k,
                v,
                self._kv_indptr_buf,
                causal,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                True,
            )
        else:
            return self._wrapper.forward_custom_mask(
                q,
                self._qo_indptr_buf,
                k,
                v,
                self._kv_indptr_buf,
                self._custom_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                logits_cap,
                allow_fp16_qk_reduction,
                sm_scale,
                rope_scale,
                rope_theta,
                True,
            )
