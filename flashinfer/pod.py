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

import functools
import math
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .jit import gen_pod_module, gen_batch_pod_module
from .page import get_seq_lens
from .prefill import get_batch_prefill_module
from .quantization import packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    device_support_pdl,
)


@functools.cache
def get_pod_module(*args):
    module = gen_pod_module(*args).build_and_load()
    return SimpleNamespace(run_tensor=module.pod_with_kv_cache_tensor)


@functools.cache
def get_batch_pod_module(*args):
    module = gen_batch_pod_module(*args).build_and_load()
    return SimpleNamespace(run_tensor=module.batch_pod_with_kv_cache_tensor)


class PODWithPagedKVCacheWrapper:
    r"""Wrapper class for POD-Attention with paged kv-cache (first proposed in
    `<https://arxiv.org/abs/2410.18038>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

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
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.PODWithPagedKVCacheWrapper(
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
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.plan(
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
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     # TODO_AK: DEMONSTRATE USAGE OF POD
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's POD-Attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`PODWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)
        """
        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0], gen_customize_batch_prefill_module("fa2", *jit_args)
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0], gen_customize_batch_decode_module(*jit_args)
                )
        else:
        """
        # Override options. Only tensor core version is performant.
        use_tensor_cores = True
        self._jit_module: SimpleNamespace = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        if use_cuda_graph:
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
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = use_cuda_graph

        if use_cuda_graph:
            # NOTE(Zihao): if once created, no need to update it in plan/run
            self._qo_indptr_buf = torch.arange(
                self._fixed_batch_size + 1,
                dtype=torch.int32,
                device=float_workspace_buffer.device,
            )

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

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
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
    ) -> None:
        r"""Plan POD's batch decode for given problem specification.

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
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        # Logits soft cap is not supported currently
        batch_size = len(last_page_len)
        logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
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
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            self._cached_module = get_batch_prefill_module(
                "fa2",
                q_data_type,
                kv_data_type,
                q_data_type,
                indptr.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                window_left != -1,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            indptr_host,
            kv_lens_arr_host,
            batch_size,  # total_num_rows
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,  # causal
            window_left,
            -1,  # fixed_split_size
            False,  # disable_split_kv
            0,  # num_colocated_ctas
        )

        self._indptr_type = indptr.dtype
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    @flashinfer_api
    def run(
        self,
        # Main params (prefill and decode)
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        v_p: torch.Tensor,
        q_d: torch.Tensor,
        paged_kv_cache_d: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        # Prefill options
        custom_mask_p: Optional[torch.Tensor] = None,
        packed_custom_mask_p: Optional[torch.Tensor] = None,
        causal_p: bool = False,
        kv_layout_p: str = "NHD",
        pos_encoding_mode_p: str = "NONE",
        sm_scale_p: Optional[float] = None,
        window_left_p: int = -1,
        rope_scale_p: Optional[float] = None,
        rope_theta_p: Optional[float] = None,
        return_lse_p: bool = False,
        # Decode options
        custom_mask_d: Optional[torch.Tensor] = None,
        packed_custom_mask_d: Optional[torch.Tensor] = None,
        causal_d: bool = False,
        kv_layout_d: str = "NHD",
        pos_encoding_mode_d: str = "NONE",
        sm_scale_d: Optional[float] = None,
        window_left_d: int = -1,
        rope_scale_d: Optional[float] = None,
        rope_theta_d: Optional[float] = None,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse_d: bool = False,
        use_fp16_qk_reduction: bool = False,
        enable_pdl: Optional[bool] = None,
        *args,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute POD-attention for a batch of requests."""
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q_p.device)

        # Currently unsupported
        logits_soft_cap_p = None
        logits_soft_cap_d = None
        # Prefill setup
        _check_pos_encoding_mode(pos_encoding_mode_p)
        _check_kv_layout(kv_layout_p)
        tmp_p = _get_cache_buf("pod_with_kv_cache_tmp", 32 * 1024 * 1024, q_p.device)
        if logits_soft_cap_p is None:
            logits_soft_cap_p = 0.0
        if sm_scale_p is None:
            sm_scale_p = 1.0 / math.sqrt(q_p.size(-1))
        if rope_scale_p is None:
            rope_scale_p = 1.0
        if rope_theta_p is None:
            rope_theta_p = 1e4
        if custom_mask_p is not None and packed_custom_mask_p is None:
            # create packed custom mask from custom mask
            packed_custom_mask_p = packbits(
                custom_mask_p.contiguous().view(-1), bitorder="little"
            )

        if packed_custom_mask_p is not None:
            mask_mode_p = MaskMode.CUSTOM.value
        else:
            if causal_p:
                mask_mode_p = MaskMode.CAUSAL.value
            else:
                mask_mode_p = MaskMode.NON_CAUSAL.value

        lse_p = None
        if return_lse_p:
            lse_p = torch.empty(
                (q_p.size(0), q_p.size(1)), dtype=torch.float32, device=q_p.device
            )

        out_p = torch.empty_like(q_p)

        # Decode setup
        k_cache_d, v_cache_d = _unpack_paged_kv_cache(paged_kv_cache_d, self._kv_layout)
        _check_cached_qkv_data_type(
            q_d, k_cache_d, self._cached_q_data_type, self._cached_kv_data_type
        )
        # TODO_AK: Where are these coming from?
        pos_encoding_mode_d = self._pos_encoding_mode
        window_left_d = self._window_left
        logits_soft_cap_d = self._logits_soft_cap
        sm_scale_d = self._sm_scale
        rope_scale_d = self._rope_scale
        rope_theta_d = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode_d)
        # What are the above for and what are the below?
        if logits_soft_cap_d is None:
            logits_soft_cap_d = 0.0
        if sm_scale_d is None:
            head_dim = q_d.shape[-1]
            sm_scale_d = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale_d *= q_scale
        if k_scale is not None:
            sm_scale_d *= k_scale
        if rope_scale_d is None:
            rope_scale_d = 1.0
        if rope_theta_d is None:
            rope_theta_d = 1e4

        lse_d = None
        if return_lse_d:
            lse_d = torch.empty(
                (q_d.size(0), q_d.size(1)), dtype=torch.float32, device=q_d.device
            )
        out_d = torch.empty_like(q_d)

        module_getter = get_pod_module(
            # Prefill params
            q_p.dtype,
            k_p.dtype,
            q_p.dtype,
            q_p.shape[-1],
            PosEncodingMode[pos_encoding_mode_p].value,
            window_left_p >= 0,  # use_sliding_window
            logits_soft_cap_p > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
            # Decode params
            # q_d.dtype,
            # self._cached_kv_data_type,
            # self._cached_q_data_type,
            self._indptr_type,
            # head_dim,  # head_dim_qk
            # head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode_d].value,
            window_left_d != -1,  # use_sliding_window
            logits_soft_cap_d > 0,  # use_logits_soft_cap
        )
        module_getter.run_tensor(
            # Prefill params
            q_p,
            k_p,
            v_p,
            tmp_p,
            out_p,
            lse_p,
            mask_mode_p,
            TensorLayout[kv_layout_p].value,
            window_left_p,
            packed_custom_mask_p,
            _get_cache_alibi_slopes_buf(q_p.shape[1], q_p.device),
            logits_soft_cap_p,
            sm_scale_p,
            1.0 / rope_scale_p,
            1.0 / rope_theta_p,
            # Decode params
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_d,
            k_cache_d,
            v_cache_d,
            self._qo_indptr_buf,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            out_d,
            lse_d,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[self._kv_layout].value,
            window_left_d,
            None,  # packed_custom_mask
            None,  # mask_indptr_buf
            _get_cache_alibi_slopes_buf(q_d.shape[1], q_d.device),
            logits_soft_cap_d,
            sm_scale_d,
            1.0 / rope_scale_d,
            1.0 / rope_theta_d,
            enable_pdl,
        )

        if v_scale is not None:
            out_d *= v_scale

        return (out_p, out_d)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


class BatchPODWithPagedKVCacheWrapper:
    r"""Wrapper class for POD-Attention with paged kv-cache (first proposed in
    `<https://arxiv.org/abs/2410.18038>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 8
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> device = 0
    >>> page_block_size = 1
    >>> causal = True
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> # Prefill and decode parameters
    >>> p_qo_lens = [2048] * 2
    >>> d_qo_lens = [1] * 128
    >>> p_kv_lens = [2048] * 2
    >>> d_kv_lens = [2048] * 128
    >>> # Prefill plan inputs
    >>> p_seq_lens_blocks = torch.ceil(
    ...     torch.tensor(p_kv_lens, dtype=torch.int32) / page_block_size
    ... ).int()
    >>> p_q_indptr = torch.cat(
    ...     [torch.tensor([0]), torch.cumsum(torch.tensor(p_qo_lens), 0)], dim=0
    ... ).int()
    >>> p_kv_indptr = torch.cat(
    ...     [torch.tensor([0]), torch.cumsum(p_seq_lens_blocks, 0)], dim=0
    ... ).int()
    >>> kv_indices_p = torch.arange(0, p_kv_indptr[-1], device=device, dtype=torch.int32)
    >>> last_page_len_p = (p_seq_lens_blocks - 1) % page_block_size + 1
    >>> # Decode plan inputs
    >>> d_seq_lens_blocks = torch.ceil(
    ...     torch.tensor(d_kv_lens, dtype=torch.int32) / page_block_size
    ... ).int()
    >>> d_q_indptr = torch.cat(
    ...     [torch.tensor([0]), torch.cumsum(torch.tensor(d_qo_lens), 0)], dim=0
    ... ).int()
    >>> d_kv_indptr = torch.cat(
    ...     [torch.tensor([0]), torch.cumsum(d_seq_lens_blocks, 0)], dim=0
    ... ).int()
    >>> kv_indices_d = torch.arange(0, d_kv_indptr[-1], device=device, dtype=torch.int32)
    >>> last_page_len_d = (d_seq_lens_blocks - 1) % page_block_size + 1
    >>> # create auxiliary data structures for batch decode attention
    >>> wrapper.plan(
    ...     # Prefill params
    ...     p_q_indptr.to(device),
    ...     p_kv_indptr.to(device),
    ...     kv_indices_p.to(device),
    ...     last_page_len_p,
    ...     # Decode params
    ...     d_q_indptr.to(device),
    ...     d_kv_indptr.to(device),
    ...     kv_indices_d.to(device),
    ...     last_page_len_d,
    ...     # Common params
    ...     num_qo_heads=num_qo_heads,
    ...     num_kv_heads=num_kv_heads,
    ...     head_dim=head_dim,
    ...     page_size=page_block_size,
    ...     q_data_type=torch.bfloat16,
    ...     kv_data_type=torch.bfloat16,
    ... )
    >>> # Prefill input tensors
    >>> q_p = torch.rand(p_q_indptr[-1].item(), num_qo_heads, head_dim).to(
    ...     device, dtype=torch.bfloat16
    ... )
    >>> kv_p = torch.randn(p_kv_indptr[-1], 2, page_block_size, num_kv_heads, head_dim).to(
    ...     device, dtype=torch.bfloat16
    ... ).unbind(1)
    >>> # Decode input tensors
    >>> q_d = torch.rand(d_q_indptr[-1].item(), num_qo_heads, head_dim).to(
    ...     device, dtype=torch.bfloat16
    ... )
    >>> kv_d = torch.randn(d_kv_indptr[-1], 2, page_block_size, num_kv_heads, head_dim).to(
    ...     device, dtype=torch.bfloat16
    ... ).unbind(1)
    >>> for i in range(num_layers):
    ...     o_p_batch, o_d_batch = wrapper.run(
    ...         q_p,
    ...         kv_p,
    ...         q_d,
    ...         kv_d,
    ...         causal_p=causal,
    ...     )
    >>> print(o_p_batch.shape, o_d_batch.shape)
    torch.Size([4096, 64, 128]) torch.Size([128, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's POD-Attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
    ) -> None:
        r"""Constructor of :class:`BatchPODWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        """
        _check_kv_layout(kv_layout)
        # Override options. Only tensor core version is performant.
        use_tensor_cores = True
        self._jit_module: SimpleNamespace = None

        self._kv_layout = kv_layout
        float_workspace_buffer_p, float_workspace_buffer_d = torch.chunk(
            float_workspace_buffer, 2, dim=0
        )
        self._float_workspace_buffer_p = float_workspace_buffer_p
        self._float_workspace_buffer_d = float_workspace_buffer_d
        self.device = float_workspace_buffer_p.device
        self._int_workspace_buffer_p = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._int_workspace_buffer_d = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer_p = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._pin_memory_int_workspace_buffer_d = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        # SM aware scheduling buffer, requires SMs count + 2 entries
        dev_prop = torch.cuda.get_device_properties(self.device)
        self._sm_aware_sched = torch.empty(
            (dev_prop.multi_processor_count + 2), dtype=torch.int, device=self.device
        )

        self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = None
        self._paged_kv_indices_buf = None
        self._paged_kv_last_page_len_buf = None
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = False

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    @flashinfer_api
    def plan(
        self,
        qo_indptr_p: torch.Tensor,
        kv_indptr_p: torch.Tensor,
        kv_indices_p: torch.Tensor,
        last_page_len_p: torch.Tensor,
        qo_indptr_d: torch.Tensor,
        kv_indptr_d: torch.Tensor,
        kv_indices_d: torch.Tensor,
        last_page_len_d: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
    ) -> None:
        r"""Plan POD's batch prefill and decode for given problem specification.

        Parameters
        ----------
        qo_indptr_p : torch.Tensor
            The prefill indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr_p : torch.Tensor
            The prefill indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices_p : torch.Tensor
            The prefill page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
        last_page_len_p : torch.Tensor
            The number of entries in the last page of each prefill request in the paged
            kv-cache, shape: ``[batch_size]``.
        qo_indptr_d : torch.Tensor
            The decode indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr_d : torch.Tensor
            The decode indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices_d : torch.Tensor
            The decode page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
        last_page_len_d : torch.Tensor
            The number of entries in the last page of each decode request in the paged
            kv-cache, shape: ``[batch_size]``.
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
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim_qk)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        # Logits soft cap is not supported currently
        logits_soft_cap = 0.0

        # Setup prefill params
        batch_size_p = len(last_page_len_p)
        qo_indptr_host_p = qo_indptr_p.to("cpu")
        total_num_rows_p = int(qo_indptr_host_p[-1])
        self._kv_indptr_buf_p = kv_indptr_p.to(self.device, non_blocking=non_blocking)
        self._kv_indices_buf_p = kv_indices_p.to(self.device, non_blocking=non_blocking)
        self._kv_last_page_len_buf_p = last_page_len_p.to(
            self.device, non_blocking=non_blocking
        )
        self._qo_indptr_buf_p = qo_indptr_host_p.to(
            self.device, non_blocking=non_blocking
        )
        kv_indptr_host_p = kv_indptr_p.to("cpu")
        last_page_len_host_p = last_page_len_p.to("cpu")
        kv_lens_arr_host_p = get_seq_lens(
            kv_indptr_host_p, last_page_len_host_p, page_size
        )

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            self._cached_module = get_batch_prefill_module(
                "fa2",
                q_data_type,
                kv_data_type,
                q_data_type,
                kv_indptr_p.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                window_left != -1,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )

        # Setup decode params
        batch_size_d = len(last_page_len_d)
        qo_indptr_host_d = qo_indptr_d.to("cpu")
        total_num_rows_d = int(qo_indptr_host_d[-1])
        self._kv_indptr_buf_d = kv_indptr_d.to(self.device, non_blocking=non_blocking)
        self._kv_indices_buf_d = kv_indices_d.to(self.device, non_blocking=non_blocking)
        self._kv_last_page_len_buf_d = last_page_len_d.to(
            self.device, non_blocking=non_blocking
        )
        self._qo_indptr_buf_d = qo_indptr_host_d.to(
            self.device, non_blocking=non_blocking
        )
        kv_indptr_host_d = kv_indptr_d.to("cpu")
        last_page_len_host_d = last_page_len_d.to("cpu")
        kv_lens_arr_host_d = get_seq_lens(
            kv_indptr_host_d, last_page_len_host_d, page_size
        )

        self._plan_info_d = self._cached_module.plan(
            self._float_workspace_buffer_d,
            self._int_workspace_buffer_d,
            self._pin_memory_int_workspace_buffer_d,
            qo_indptr_host_d,
            kv_indptr_host_d,
            kv_lens_arr_host_d,
            total_num_rows_d,  # total_num_rows
            batch_size_d,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,  # causal
            window_left,
            -1,  # fixed_split_size
            False,  # disable_split_kv
            0,  # num_colocated_ctas
        )

        num_colocated_ctas = self._plan_info_d[0]
        # Splitting small prefill causes unecessary bandwidth contention
        if total_num_rows_p > 1536:
            num_colocated_ctas = 0
        self._plan_info_p = self._cached_module.plan(
            self._float_workspace_buffer_p,
            self._int_workspace_buffer_p,
            self._pin_memory_int_workspace_buffer_p,
            qo_indptr_host_p,
            kv_indptr_host_p,
            kv_lens_arr_host_p,
            total_num_rows_p,  # total_num_rows
            batch_size_p,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,  # causal
            window_left,
            -1,  # fixed_split_size
            False,  # disable_split_kv
            num_colocated_ctas,
        )
        self._indptr_type = kv_indptr_p.dtype
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    @flashinfer_api
    def run(
        self,
        # Main params (prefill and decode)
        q_p: torch.Tensor,
        paged_kv_cache_p: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        q_d: torch.Tensor,
        paged_kv_cache_d: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        # Prefill options
        custom_mask_p: Optional[torch.Tensor] = None,
        packed_custom_mask_p: Optional[torch.Tensor] = None,
        causal_p: bool = False,
        # Decode options
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        # Common options
        return_lse: bool = False,
        use_fp16_qk_reduction: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ]:
        r"""Compute POD-attention for a batch of requests."""
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q_p.device)

        # Currently unsupported
        logits_soft_cap_p = None
        logits_soft_cap_d = None
        # Prefill setup
        k_cache_p, v_cache_p = _unpack_paged_kv_cache(paged_kv_cache_p, self._kv_layout)
        _check_cached_qkv_data_type(
            q_p, k_cache_p, self._cached_q_data_type, self._cached_kv_data_type
        )
        # Get params from plan
        pos_encoding_mode_p = self._pos_encoding_mode
        window_left_p = self._window_left
        logits_soft_cap_p = self._logits_soft_cap
        sm_scale_p = self._sm_scale
        rope_scale_p = self._rope_scale
        rope_theta_p = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode_p)
        if logits_soft_cap_p is None:
            logits_soft_cap_p = 0.0
        if sm_scale_p is None:
            head_dim = q_p.shape[-1]
            sm_scale_p = 1.0 / math.sqrt(head_dim)
        if rope_scale_p is None:
            rope_scale_p = 1.0
        if rope_theta_p is None:
            rope_theta_p = 1e4

        if custom_mask_p is not None and packed_custom_mask_p is None:
            # create packed custom mask from custom mask
            packed_custom_mask_p = packbits(
                custom_mask_p.contiguous().view(-1), bitorder="little"
            )

        if packed_custom_mask_p is not None:
            mask_mode_p = MaskMode.CUSTOM.value
        else:
            if causal_p:
                mask_mode_p = MaskMode.CAUSAL.value
            else:
                mask_mode_p = MaskMode.NON_CAUSAL.value

        lse_p = None
        if return_lse:
            lse_p = torch.empty(
                (q_p.size(0), q_p.size(1)), dtype=torch.float32, device=q_p.device
            )
        out_p = torch.empty_like(q_p)

        # Decode setup
        k_cache_d, v_cache_d = _unpack_paged_kv_cache(paged_kv_cache_d, self._kv_layout)
        _check_cached_qkv_data_type(
            q_d, k_cache_d, self._cached_q_data_type, self._cached_kv_data_type
        )
        # Get params from plan
        pos_encoding_mode_d = self._pos_encoding_mode
        window_left_d = self._window_left
        logits_soft_cap_d = self._logits_soft_cap
        sm_scale_d = self._sm_scale
        rope_scale_d = self._rope_scale
        rope_theta_d = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode_d)
        if logits_soft_cap_d is None:
            logits_soft_cap_d = 0.0
        if sm_scale_d is None:
            head_dim = q_d.shape[-1]
            sm_scale_d = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale_d *= q_scale
        if k_scale is not None:
            sm_scale_d *= k_scale
        if rope_scale_d is None:
            rope_scale_d = 1.0
        if rope_theta_d is None:
            rope_theta_d = 1e4

        lse_d = None
        if return_lse:
            lse_d = torch.empty(
                (q_d.size(0), q_d.size(1)), dtype=torch.float32, device=q_d.device
            )
        out_d = torch.empty_like(q_d)

        module_getter = get_batch_pod_module(
            # Prefill params
            q_p.dtype,
            k_cache_p.dtype,
            q_p.dtype,
            q_p.shape[-1],
            PosEncodingMode[pos_encoding_mode_p].value,
            window_left_p >= 0,  # use_sliding_window
            logits_soft_cap_p > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
            # Decode params
            self._indptr_type,
            PosEncodingMode[pos_encoding_mode_d].value,
            window_left_d != -1,  # use_sliding_window
            logits_soft_cap_d > 0,  # use_logits_soft_cap
        )
        module_getter.run_tensor(
            # Prefill params
            self._float_workspace_buffer_p,
            self._int_workspace_buffer_p,
            self._plan_info_p,
            q_p,
            k_cache_p,
            v_cache_p,
            self._qo_indptr_buf_p,
            self._kv_indptr_buf_p,
            self._kv_indices_buf_p,
            self._kv_last_page_len_buf_p,
            out_p,
            lse_p,
            mask_mode_p,
            TensorLayout[self._kv_layout].value,
            window_left_p,
            packed_custom_mask_p,  # packed_custom_mask
            None,  # mask_indptr_buf
            _get_cache_alibi_slopes_buf(q_p.shape[1], q_p.device),
            logits_soft_cap_p,
            sm_scale_p,
            1.0 / rope_scale_p,
            1.0 / rope_theta_p,
            # Decode params
            self._float_workspace_buffer_d,
            self._int_workspace_buffer_d,
            self._plan_info_d,
            q_d,
            k_cache_d,
            v_cache_d,
            self._qo_indptr_buf_d,
            self._kv_indptr_buf_d,
            self._kv_indices_buf_d,
            self._kv_last_page_len_buf_d,
            out_d,
            lse_d,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[self._kv_layout].value,
            window_left_d,
            None,  # packed_custom_mask
            None,  # mask_indptr_buf
            _get_cache_alibi_slopes_buf(q_d.shape[1], q_d.device),
            logits_soft_cap_d,
            sm_scale_d,
            1.0 / rope_scale_d,
            1.0 / rope_theta_d,
            enable_pdl,
            self._sm_aware_sched,
        )

        if v_scale is not None:
            out_d *= v_scale

        return ((out_p, out_d), (lse_p, lse_d)) if return_lse else (out_p, out_d)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass
