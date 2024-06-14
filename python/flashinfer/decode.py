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
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    logits_cap: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Decode attention with KV Cache for single request, return attention output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The value tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if
        :attr:`kv_layout` is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    logits_cap : bool
        Whether to apply logits cap to pre-attention logits.
        If ``True``, the logits will be capped according to formula (proposed in
        Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
        Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.

    Returns
    -------
    torch.Tensor
        The attention output, shape: ``[num_qo_heads, head_dim]``

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_decode_with_kv_cache(q, k, v)
    >>> o.shape
    torch.Size([32, 128])

    Notes
    -----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    check_pos_encoding_mode(pos_encoding_mode)
    check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 8 * 1024 * 1024, q.device)
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    out = _kernels.single_decode_with_kv_cache(
        q,
        k,
        v,
        tmp,
        PosEncodingMode[pos_encoding_mode].value,
        logits_cap,
        TensorLayout[kv_layout].value,
        sm_scale,
        rope_scale,
        rope_theta,
    )
    if v_scale is not None:
        out *= v_scale
    return out


def batch_decode_with_padded_kv_cache(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    logits_cap: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Decode attention with padded KV cache for batch of requests, return attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
    k_padded : torch.Tensor
        The padded key tensor, shape:
        ``[batch_size, padded_seq_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD`` or ``[batch_size, num_kv_heads, padded_seq_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    v_padded : torch.Tensor
        The padded value tensor, shape:
        ``[batch_size, padded_seq_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD`` or ``[batch_size, num_kv_heads, padded_seq_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    logits_cap : bool
        Whether to apply logits cap to pre-attention logits.
        If ``True``, the logits will be capped according to formula (proposed in
        Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
        Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.

    Returns
    -------
    torch.Tensor
        The attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> padded_kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> batch_size = 7
    >>> head_dim = 128
    >>> q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_padded = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_padded = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.batch_decode_with_padded_kv_cache(
    ...     q, k_padded, v_padded, "NHD", "LLAMA"
    ... )
    >>> o.shape
    torch.Size([7, 32, 128])

    Notes
    -----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    out = _kernels.batch_decode_with_padded_kv_cache(
        q,
        k_padded,
        v_padded,
        TensorLayout[kv_layout].value,
        PosEncodingMode[pos_encoding_mode].value,
        logits_cap,
        sm_scale,
        rope_scale,
        rope_theta,
        False,
    )[0]
    if v_scale is not None:
        out *= v_scale
    return out


def batch_decode_with_padded_kv_cache_return_lse(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    logits_cap: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    r"""Decode attention with padded KV cache for batch of requests, return attention
    output and logsumexp of attention scores, return attention output and logsumexp of
    attention scores.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``.
    k_padded : torch.Tensor
        The padded key tensor, shape:
        ``[batch_size, padded_seq_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD`` or ``[batch_size, num_kv_heads, padded_seq_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    v_padded : torch.Tensor
        The padded value tensor, shape:
        ``[batch_size, padded_seq_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD`` or ``[batch_size, num_kv_heads, padded_seq_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    logits_cap : bool
        Whether to apply logits cap to pre-attention logits.
        If ``True``, the logits will be capped according to formula (proposed in
        Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
        Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.

    Returns
    -------
    V : torch.Tensor
        The attention output, shape: [batch_size, num_qo_heads, head_dim]
    S : torch.Tensor
        The logsumexp of attention scores, Shape: [batch_size, num_qo_heads]

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> padded_kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> batch_size = 7
    >>> head_dim = 128
    >>> q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_padded = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_padded = torch.randn(batch_size, padded_kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v, s = flashinfer.batch_decode_with_padded_kv_cache_return_lse(
    ...     q, k_padded, v_padded, "NHD"
    ... )
    >>> v.shape
    torch.Size([7, 32, 128])
    >>> s.shape
    torch.Size([7, 32])

    Notes
    -----
    Please refer to the :ref:`tutorial <recursive-attention>` for a detailed
    explanation of the log-sum-exp function and attention states.

    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    V, s = _kernels.batch_decode_with_padded_kv_cache(
        q,
        k_padded,
        v_padded,
        TensorLayout[kv_layout].value,
        PosEncodingMode[pos_encoding_mode].value,
        logits_cap,
        sm_scale,
        rope_scale,
        rope_theta,
        True,
    )
    if v_scale is not None:
        V *= v_scale
    return V, s


class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with paged kv-cache (first proposed in
    `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 16MB workspace buffer
    >>> workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_data_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.begin_forward(
    ...     kv_page_indptr,
    ...     kv_page_indices,
    ...     kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     pos_encoding_mode="NONE",
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_data = kv_data_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = decode_wrapper.forward(q, kv_data)
    ...     outputs.append(o)
    ...
    >>> # clear auxiliary data structures
    >>> decode_wrapper.end_forward()
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's batch decode attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        enable_cuda_graph: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    ):
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 16MB (128MB if cudagraph enabled), the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        enable_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored in the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``enable_cuda_graph`` is ``True``.

        indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``enable_cuda_graph`` is ``True``.

        last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``enable_cuda_graph`` is ``True``.
        """
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer

        if enable_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer

        self._wrapper = _kernels.BatchDecodeWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value,
            enable_cuda_graph,
            self._fixed_batch_size,
        )

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
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        data_type: Union[str, torch.dtype] = "float16",
        q_data_type: Optional[Union[str, torch.dtype]] = None,
    ):
        r"""Create auxiliary data structures for batch decode for multiple forward calls
        within the same decode step.

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
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        logits_cap: bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache. Defaults to ``float16``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor. If None, will be set to
            ``data_type``. Defaults to ``None``.

        Note
        ----
        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        batch_size = len(last_page_len)

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr)
            self._paged_kv_indices_buf[: len(indices)] = indices
            self._paged_kv_last_page_len_buf.copy_(last_page_len)
        else:
            self._paged_kv_indptr_buf = indptr
            self._paged_kv_indices_buf = indices
            self._paged_kv_last_page_len_buf = last_page_len

        # NOTE(Zihao): the following tensors acts as placeholder to pass dtype info
        if not q_data_type:
            q_data_type = data_type
        empty_q_data = torch.empty(
            0,
            dtype=(
                getattr(torch, q_data_type)
                if isinstance(q_data_type, str)
                else q_data_type
            ),
        )
        empty_kv_data = torch.empty(
            0,
            dtype=(
                getattr(torch, data_type) if isinstance(data_type, str) else data_type
            ),
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
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            empty_q_data,
            empty_kv_data,
        )

    def end_forward(self):
        r"""Clear auxiliary data structures created by :meth:`begin_forward`."""
        if not self.is_cuda_graph_enabled:
            self._paged_kv_indptr_buf = None
            self._paged_kv_indices_buf = None
            self._paged_kv_last_page_len_buf = None
        self._wrapper.end_forward()

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_data : torch.Tensor
            A 5-D tensor of the reserved paged kv-cache data, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        logits_cap: bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        sm_scale : Optional[float]
            The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.

        Returns
        -------
        torch.Tensor
            The attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
        """
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        out = self._wrapper.forward(
            q,
            paged_kv_data,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            sm_scale,
            rope_scale,
            rope_theta,
            False,
        )[0]
        if v_scale is not None:
            out *= v_scale
        return out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_data: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        logits_cap: bool = False,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        r"""Compute batch decode attention with paged kv cache, return attention output
        and logsumexp of attention scores.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_data : torch.Tensor
            A 5-D tensor of the reserved paged kv-cache data, shape:
            ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
            :attr:`kv_layout` is ``NHD``, or
            ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
            :attr:`kv_layout` is ``HND``.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        logits_cap: bool
            Whether to apply logits cap to pre-attention logits.
            If ``True``, the logits will be capped according to formula (proposed in
            Grok-1): :math:`30 \times \mathrm{tanh}(x / 30)`, where :math:`x` is the input logits.
            Defaults to ``False``.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
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
            The attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
        S : torch.Tensor
            The logsumexp of attention scores, Shape: ``[batch_size, num_qo_heads]``.

        Notes
        -----
        Please refer to the :ref:`tutorial <recursive-attention>` for a detailed
        explanation of the log-sum-exp function and attention states.
        """
        check_pos_encoding_mode(pos_encoding_mode)
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        paged_kv_data = expand_5d(paged_kv_data, self._kv_layout)
        V, s = self._wrapper.forward(
            q,
            paged_kv_data,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            PosEncodingMode[pos_encoding_mode].value,
            logits_cap,
            sm_scale,
            rope_scale,
            rope_theta,
            True,
        )
        if v_scale is not None:
            V *= v_scale
        return V, s


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    r"""CUDAGraph-compatible Wrapper class for decode attention with paged kv-cache (first
    proposed in `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Note that this wrapper may not be as efficient as :class:`BatchDecodeWithPagedKVCacheWrapper`
    because we won't dispatch to different kernels for different batch sizes/sequence lengths/etc
    to accomodate the CUDAGraph requirement.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    Note
    ----
    The :meth:`begin_forward` method could not be captured by CUDAGraph.

    See Also
    --------
    :class:`BatchDecodeWithPagedKVCacheWrapper`
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        kv_layout: str = "NHD",
    ):
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer on GPU used to store auxiliary data structures,
            recommended size is 128MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        indptr_buffer : torch.Tensor
            The user reserved buffer on GPU to store the indptr of the paged kv cache, should
            be large enough to store the indptr of maximum batch size (``[max_batch_size + 1]``)
            during the lifecycle of this wrapper.

        indices_buffer : torch.Tensor
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.

        last_page_len_buffer : torch.Tensor
            The user reserved buffer on GPU to store the number of entries in the last page,
            should be large enough to store the maximum batch size (``[max_batch_size]``)
            during the lifecycle of this wrapper.
        """
        super().__init__(
            workspace_buffer,
            kv_layout,
            True,
            indptr_buffer,
            indices_buffer,
            last_page_len_buffer,
        )
