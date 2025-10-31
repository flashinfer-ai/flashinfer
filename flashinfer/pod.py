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

from .jit import gen_pod_module, get_pod_uri
from .quantization import packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _get_cache_alibi_slopes_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    device_support_pdl,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_pod_module(*args):
    """Get POD module with cached compilation."""
    # Use the proper JIT compilation system like batch prefill
    uri = get_pod_uri(*args)
    module = gen_pod_module(*args).build_and_load()
    plan_func = module.PODWithKVCachePlan
    run_tensor_func = module.PODWithKVCacheTensorRun

    # Register custom op for POD tensor run
    @register_custom_op(
        f"flashinfer::{uri}_pod_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def pod_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        layout: int,
        # Prefill params
        q_p: torch.Tensor,
        mask_mode_code_p: int,
        window_left_p: int,
        maybe_custom_mask_p: Optional[torch.Tensor],
        maybe_alibi_slopes_p: Optional[torch.Tensor],
        logits_soft_cap_p: float,
        sm_scale_p: float,
        rope_rcp_scale_p: float,
        rope_rcp_theta_p: float,
        # Decode params
        q_d: torch.Tensor,
        mask_mode_code_d: int,
        window_left_d: int,
        maybe_custom_mask_d: Optional[torch.Tensor],
        maybe_mask_indptr_d: Optional[torch.Tensor],
        maybe_alibi_slopes_d: Optional[torch.Tensor],
        logits_soft_cap_d: float,
        sm_scale_d: float,
        rope_rcp_scale_d: float,
        rope_rcp_theta_d: float,
        enable_pdl: bool,
    ) -> None:
        run_tensor_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            layout,
            q_p,
            mask_mode_code_p,
            window_left_p,
            maybe_custom_mask_p,
            maybe_alibi_slopes_p,
            logits_soft_cap_p,
            sm_scale_p,
            rope_rcp_scale_p,
            rope_rcp_theta_p,
            q_d,
            mask_mode_code_d,
            window_left_d,
            maybe_custom_mask_d,
            maybe_mask_indptr_d,
            maybe_alibi_slopes_d,
            logits_soft_cap_d,
            sm_scale_d,
            rope_rcp_scale_d,
            rope_rcp_theta_d,
            enable_pdl,
        )

    @register_fake_op(f"flashinfer::{uri}_pod_run")
    def _fake_pod_run(*args) -> None:
        pass

    # # Create a simple namespace that wraps the JIT module functions
    # class PODModule:
    #     def __init__(self):
    #         pass

    #     def plan(self, *args):
    #         """Call the POD plan function."""
    #         return plan_func(*args)

    #     def run_tensor(self, *args):
    #         """Call the POD tensor run function."""
    #         return pod_run(*args)

    # return PODModule()
    return SimpleNamespace(run_tensor=pod_run, plan=plan_func)


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
    >>> wrapper = flashinfer.PODWithPagedKVCacheWrapper(
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
    >>> wrapper.plan(
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

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        custom_mask_buf_p: Optional[torch.Tensor] = None,
        mask_indptr_buf_p: Optional[torch.Tensor] = None,
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

        qo_indptr_buffer: Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indptr_buffer: Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the prefill paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer: Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the prefill paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer: Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page for prefill, the
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
        assert custom_mask_buf_p is None and mask_indptr_buf_p is None, (
            "custom_mask_buf_p and mask_indptr_buf_p are not supported yet"
        )

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._qo_indptr_buf = qo_indptr_buffer
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
            if not torch.is_tensor(qo_indptr_buffer):
                raise ValueError(
                    "qo_indptr_buffer should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indptr_buffer) or not torch.is_tensor(
                paged_kv_indptr_buffer
            ):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer) or not torch.is_tensor(
                paged_kv_indices_buffer
            ):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(
                paged_kv_last_page_len_buffer
            ) or not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buffer_p should be batch_size + 1"
                )
            if len(paged_kv_last_page_len_buffer) != self._fixed_batch_size:
                raise ValueError(
                    "The length of paged_kv_last_page_len_buffer_p should be batch_size"
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

    def plan(
        self,
        qo_indptr_p: torch.Tensor,
        kv_indptr_p: torch.Tensor,
        kv_indices_p: torch.Tensor,
        last_page_len_p: torch.Tensor,
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
        logits_soft_cap: Optional[float] = 0.0,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
    ) -> None:
        r"""Plan POD's batch decode for given problem specification.

        Parameters
        ----------
        qo_indptr_p: torch.Tensor
            The indptr of the query/output tensor for prefill, shape: ``[batch_size + 1]``.
        kv_indptr_p: torch.Tensor
            The indptr of the paged kv cache for prefill, shape: ``[batch_size + 1]``.
        kv_indices_p: torch.Tensor
            The page indices of the paged kv cache for prefill, shape: ``[kv_indptr[-1]]``.
        last_page_len_p : torch.Tensor
            The number of entries in the last page of each request in the kv
            cache, shape: ``[batch_size]``
        kv_indptr_d : torch.Tensor
            The indptr of the paged kv cache for decode, shape: ``[batch_size + 1]``
        kv_indices_d : torch.Tensor
            The page indices of the paged kv cache for decode, shape: ``[kv_indptr[-1]]``
        last_page_len_d : torch.Tensor
            The number of entries in the last page of each request in the kv
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
        # Logits soft cap is not supported currently; keep a float for typing
        batch_size_p = len(last_page_len_p)
        batch_size_d = len(last_page_len_d)
        batch_size = batch_size_p + batch_size_d
        # keep logits_soft_cap as float consistently

        qo_indptr_host_p = qo_indptr_p.to("cpu", non_blocking=True)
        qo_indptr_host_d = _get_range_buf(batch_size_d + 1, "cpu")
        to_device = lambda x: x.to(self.device, non_blocking=non_blocking)
        qo_indptr_p = to_device(qo_indptr_p)
        qo_indptr = torch.cat(
            [qo_indptr_p, to_device(qo_indptr_host_d)[1:] + qo_indptr_p[-1]]
        )
        kv_indptr_p = to_device(kv_indptr_p)
        kv_indptr = torch.cat(
            [kv_indptr_p, to_device(kv_indptr_d)[1:] + kv_indptr_p[-1]]
        )
        kv_indices_p = to_device(kv_indices_p)
        kv_indices = torch.cat(
            [kv_indices_p, to_device(kv_indices_d)[1:] + kv_indices_p[-1]]
        )
        last_page_len = torch.cat(
            [to_device(last_page_len_p), to_device(last_page_len_d)]
        )
        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size_d, self._fixed_batch_size
                    )
                )
            if len(kv_indices_d) + len(kv_indices_p) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf[: batch_size + 1].copy_(
                kv_indptr, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf[: batch_size + 1].copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(kv_indices)].copy_(
                kv_indices, non_blocking=non_blocking
            )
        else:
            self._qo_indptr_buf = qo_indptr
            self._paged_kv_indptr_buf = kv_indptr
            self._paged_kv_indices_buf = kv_indices
            self._paged_kv_last_page_len_buf = last_page_len

        kv_indptr_host_p = kv_indptr_p.to("cpu")
        kv_indptr_host_d = kv_indptr_d.to("cpu")

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
        self._indptr_type = kv_indptr_d.dtype
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        assert (
            qo_indptr_p.dtype == self._indptr_type
            and kv_indptr_p.dtype == self._indptr_type
            and qo_indptr_host_d.dtype == self._indptr_type
            and kv_indptr_d.dtype == self._indptr_type
            and f"Indices dtype mismatch: {qo_indptr_p.dtype}, {kv_indptr_p.dtype}, {qo_indptr_host_d.dtype}, {kv_indptr_d.dtype}"
        )

        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            self._cached_module = get_pod_module(
                # Prefill params
                q_data_type,
                kv_data_type,
                q_data_type,
                head_dim,  # head_dim_qk
                PosEncodingMode[pos_encoding_mode].value,
                window_left != -1,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
                # Decode params
                self._indptr_type,
                PosEncodingMode[pos_encoding_mode].value,
                window_left != -1,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
            )
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host_p,
            kv_indptr_host_p,
            int(qo_indptr_host_p[-1]),  # total_num_rows_p
            batch_size_p,
            qo_indptr_host_d,
            kv_indptr_host_d,
            int(qo_indptr_host_d[-1]),  # total_num_rows_d
            batch_size_d,
            num_qo_heads,
            num_kv_heads,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            page_size,
            self.is_cuda_graph_enabled,
        )

    begin_forward = plan

    def run(
        self,
        # Main params (prefill and decode)
        q_p: torch.Tensor,
        q_d: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        # Prefill options
        custom_mask_p: Optional[torch.Tensor] = None,
        packed_custom_mask_p: Optional[torch.Tensor] = None,
        causal_p: bool = False,
        pos_encoding_mode_p: str = "NONE",
        sm_scale_p: Optional[float] = None,
        window_left_p: int = -1,
        rope_scale_p: Optional[float] = None,
        rope_theta_p: Optional[float] = None,
        # Decode options
        custom_mask_d: Optional[torch.Tensor] = None,
        packed_custom_mask_d: Optional[torch.Tensor] = None,
        causal_d: bool = False,
        pos_encoding_mode_d: str = "NONE",
        sm_scale_d: Optional[float] = None,
        window_left_d: int = -1,
        rope_scale_d: Optional[float] = None,
        rope_theta_d: Optional[float] = None,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse: bool = False,
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

        lse = None
        if return_lse:
            lse = torch.empty(
                (q_p.size(0) + q_d.size(0), q_p.size(1)),
                dtype=torch.float32,
                device=q_p.device,
            )
        qo_len_p, num_qo_heads, head_dim = q_p.shape
        qo_len_d, _, _ = q_d.shape
        out = torch.empty(
            qo_len_p + qo_len_d,
            num_qo_heads,
            head_dim,
            device=q_p.device,
            dtype=q_p.dtype,
        )

        # Decode setup
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        _check_cached_qkv_data_type(
            q_d, k_cache, self._cached_q_data_type, self._cached_kv_data_type
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

        module_getter = get_pod_module(
            # Prefill params
            q_p.dtype,
            k_cache.dtype,
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
            # Shared params
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            k_cache,
            v_cache,
            self._qo_indptr_buf,  # contains both prefill and decode indptr
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            out,
            lse,
            TensorLayout[self._kv_layout].value,
            # Prefill params
            q_p,
            mask_mode_p,
            window_left_p,
            packed_custom_mask_p,
            _get_cache_alibi_slopes_buf(q_p.shape[1], q_p.device),
            logits_soft_cap_p,
            sm_scale_p,
            1.0 / rope_scale_p,
            1.0 / rope_theta_p,
            # Decode params
            q_d,
            MaskMode.NON_CAUSAL.value,
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
            out *= v_scale

        return out[:qo_len_p], out[qo_len_p:]

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass
