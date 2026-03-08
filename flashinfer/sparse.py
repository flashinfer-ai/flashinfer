"""
Copyright (c) 2024 by FlashInfer team.

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
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .decode import get_batch_decode_module
from .prefill import _compute_page_mask_indptr, get_batch_prefill_module
from .quantization import segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    canonicalize_torch_dtype,
    determine_attention_backend,
    device_support_pdl,
    is_float8,
)


def convert_bsr_mask_layout(mask: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    r"""Convert mask from BSR data layout to flashinfer's flattened mask layout.

    Parameters
    ----------
    mask : torch.Tensor
        A boolean mask tensor with shape ``(nnz, R, C)``.
    indptr : torch.Tensor
        The indptr tensor in BSR format.

    Returns
    -------
    flattened_mask : torch.Tensor
        A flattenedd mask tensor with shape ``(nnz * R * C,)``.
    """
    nnz, R, C = mask.shape
    MB = len(indptr) - 1
    mask_flashinfer = torch.empty((nnz * R * C,), dtype=mask.dtype, device=mask.device)
    for i in range(MB):
        mask_flashinfer[indptr[i] * R * C : indptr[i + 1] * R * C] = (
            mask[indptr[i] : indptr[i + 1]].transpose(0, 1).reshape(-1)
        )
    return mask_flashinfer


class BlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    The definition of block sparse matrix can be found at
    `bsr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_
    in SciPy.

    This API supports any block size ``(R, C)``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 32
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
    >>> # sparse mask: [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    >>> M = 3
    >>> N = 3
    >>> indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")
    >>> indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")
    >>> bsr_wrapper.plan(
    ...     indptr,
    ...     indices,
    ...     M,
    ...     N,
    ...     1, # R(block_rows)=1
    ...     1, # C(block_columns)=1
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = bsr_wrapper.run(q, k, v)
    >>> # use dense implementation with attention mask for comparison
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")
    >>> o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_ref)
    True
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`BlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._packed_mask_buf: Optional[torch.Tensor] = None
        self._mask_indptr_buf: Optional[torch.Tensor] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
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
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        M: int,
        N: int,
        R: int,
        C: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mask: Optional[torch.Tensor] = None,
        packed_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Union[str, torch.dtype] = "float16",
        non_blocking: bool = True,
    ) -> None:
        r"""Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        indptr : torch.Tensor
            The block index pointer of the block-sparse matrix on row dimension, shape ``(MB + 1,)``,
            where ``MB`` is the number of blocks in the row dimension.
        indices: torch.Tensor
            The block indices of the block-sparse matrix on column dimension, shape ``(nnz,)``, where
            ``nnz`` is the number of non-zero blocks. The elements in ``indices`` array should be less then ``NB``:
            the number of blocks in the column dimension.
        M : int
            The number of rows of the block-sparse matrix, ``MB = ceil_div(M, R)``.
        N : int
            The number of columns of the block-sparse matrix, ``NB = N // C``, ``N`` should be divisible by ``C``.
        R : int
            The number of rows in each block.
        C : int
            The number of columns in each block.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor.
        head_dim : int
            The dimension of each head.
        mask : torch.Tensor, optional
            The mask tensor with shape ``(nnz, R, C,)``, where nnz is the number of non-zero blocks.
            If every block is full, then we don't need to provide the mask tensor.
        packed_mask : torch.Tensor, optional
            The 1D packed mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : str, optional
            The data type of the query tensor.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        o_data_type : str, optional
            The data type of the output tensor. Default is ``half``. As output dtype cannot
            be inferred by input dtype in quantization
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = canonicalize_torch_dtype(o_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        num_blocks_row = len(indptr) - 1
        qo_indptr_host = R * torch.arange(num_blocks_row + 1, dtype=torch.int32)
        qo_indptr_host[-1] = M
        qo_indptr = qo_indptr_host.to(indptr.device, non_blocking=non_blocking)
        if indices.max().item() * C > N:
            raise ValueError("indices out of bound")
        last_block_len = torch.full(
            (num_blocks_row,), C, dtype=torch.int32, device=indptr.device
        )

        if mask is not None or packed_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                indptr,  # paged_kv_indptr
                last_block_len,  # paged_kv_last_page_len
                C,  # page_size
            )
        if packed_mask is None and mask is not None:
            # first convert BSR mask to flashinfer layout
            mask = convert_bsr_mask_layout(mask, indptr)
            # create packed mask from mask
            packed_mask, mask_indptr = segment_packbits(
                mask.contiguous().view(-1), mask_indptr, bitorder="little"
            )

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking)
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, non_blocking=non_blocking
        )
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask.to(
                self.device, non_blocking=non_blocking
            )
            self._mask_indptr_buf = mask_indptr.to(
                self.device, non_blocking=non_blocking
            )
            mask_mode = MaskMode.CUSTOM.value
        else:
            self._packed_mask_buf = None
            self._mask_indptr_buf = None
            mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._mask_mode = mask_mode

        self.M = M
        self.N = N
        self.R = R
        self.C = C

        kv_indptr_host = indptr.to("cpu")

        # NOTE(Zihao): we haven't supported mask in cuda-core implementations but it should
        # be easy to add support for it if needed, leave it as a future work.
        # at this moment, when mask is provided, we use the tensor-core implementation
        if (
            R * (num_qo_heads // num_kv_heads) < 4
            and mask_mode != MaskMode.CUSTOM.value
            and q_data_type not in [torch.float8_e4m3fn, torch.float8_e5m2]
        ):
            # If the operation is not compute-bound, we use the cuda-core implementation
            self._use_tensor_cores = False
            self._cached_module = get_batch_decode_module(
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
            )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                kv_indptr_host,
                num_blocks_row,
                num_qo_heads,
                num_kv_heads,
                C,
                False,  # is_cuda_graph_enabled
                -1,  # window_left
                logits_soft_cap,  # logits_soft_cap
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )
        else:
            # if the operation is compute-bound, we use the tensor-core implementation
            self._use_tensor_cores = True

            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    mask_mode == MaskMode.CUSTOM.value,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(
                self._backend, *get_module_args
            )

            kv_lens_arr_host = (kv_indptr_host[1:] - kv_indptr_host[:-1]) * self.C
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host,
            )

            args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_lens_arr_host,
                M,  # total_num_rows
                num_blocks_row,  # batch_size
                num_qo_heads,
                num_kv_heads,
                self.C,  # page_size
                False,  # is_cuda_graph_enabled,
                head_dim,
                head_dim,
                causal,
                -1,  # window_left
            ]
            if self._backend == "fa2":
                args.append(-1)  # fixed_split_size
                args.append(False)  # disable_split_kv
                args.append(0)  # num_colocated_ctas
            self._plan_info = self._cached_module.plan(
                *args,
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v, scale_q, scale_k, scale_v)

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(M, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(N, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(N, num_kv_heads, head_dim)``.
        scale_q : Optional[torch.Tensor]
            The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_k : Optional[torch.Tensor]
            The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_v : Optional[torch.Tensor]
            The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        k = k.reshape(-1, self.C, *k.shape[-2:])
        v = v.reshape(-1, self.C, *v.shape[-2:])

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        if is_float8(q):
            assert q.dtype == k.dtype == v.dtype
            assert q.shape[-1] == k.shape[-1] == v.shape[-1]
            assert self._backend == "fa3" and self._use_tensor_cores

            if scale_q is None:
                scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
            if scale_k is None:
                scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
            if scale_v is None:
                scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

        if self._use_tensor_cores:
            self._cached_module.paged_run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._qo_indptr,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                self._mask_mode,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                enable_pdl,
                # ADDITIONAL_FUNC_PARAMS
                self._packed_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                None,  # maybe_prefix_len_ptr
                None,  # maybe_token_pos_in_items_ptr
                None,  # maybe_max_item_len_ptr
                logits_soft_cap,
                sm_scale,
                scale_q,
                scale_k,
                scale_v,
                rope_scale,
                rope_theta,
                0,  # token_pos_in_items_len
                self._workspace_size,  # workspace_size
            )
        else:
            self._cached_module.run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                enable_pdl,
                # ADDITIONAL_FUNC_PARAMS
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            )

        return (out, lse) if return_lse else out

    def end_forward(self) -> None:
        r"""Warning: This method is deprecated and has no effect."""
        pass


class VariableBlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    This API supports variable block sizes provided by ``block_row_sz`` and ``block_col_sz``.
    Besides, each ``kv_head_idx`` can specify its own sparse patterns without using the same mask.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 1
    >>> num_kv_heads = 1
    >>> head_dim = 128
    >>> seq_len = 6 # This corresponds to the `block_row_sz` and `block_col_sz`
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.VariableBlockSparseAttentionWrapper(workspace_buffer)
    >>> block_mask_map = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device="cuda:0")
    >>> block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda:0")
    >>> block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda:0")
    >>> wrapper.plan(
    ...     block_mask_map,
    ...     block_row_sz,
    ...     block_col_sz,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((num_qo_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = wrapper.run(q, k, v)
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`VariableBlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
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
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        r"""Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        block_mask_map : torch.Tensor
            The block mask map (boolean), shape ``(num_kv_heads, MB, NB)``, where ``MB`` is the number of blocks in the row dimension,
            ``NB`` is the number of blocks in the column dimension.
        block_row_sz : torch.Tensor
            The block row size, shape ``(num_kv_heads, MB,)``.
        block_col_sz : torch.Tensor
            The block column size, shape ``(num_kv_heads, NB,)``.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor. Note that a group of ``qo_heads`` shares the same sparse pattern of ``kv_heads``.
        head_dim : int
            The dimension of each head.
        causal : bool
            Whether to apply causal mask to the attention matrix.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = q_data_type

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        # num_blocks are constant across kv_heads
        num_blocks_row = block_row_sz.shape[-1]
        num_blocks_col = block_col_sz.shape[-1]

        # q layout: [seq_len, num_kv_heads, gqa_group_size, head_dim]
        # padded into: [seq_len * num_kv_heads, 1, gqa_group_size, head_dim]
        qo_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
                torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
            ],
            dim=0,
        )
        qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
        last_block_len = torch.full(
            (num_blocks_row * num_kv_heads,),
            1,
            dtype=torch.int32,
            device=block_mask_map.device,
        )  # We use page_size == 1 for variable length support

        # HND kv layout: [num_kv_heads, num_blocks, block_size, head_dim]
        # padded into: [num_kv_heads * num_blocks, block_size, 1, head_dim]
        # for customized attention mask for each kv_head
        # NOTE(Yilong): This could be perf bottleneck. Consider Triton implementation.
        def _block_mask_map_to_expanded_indices(
            block_mask_map: torch.Tensor,  # [H, R, C] bool / {0,1}
            block_col_sz: torch.Tensor,  # [H, C]     int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                block_mask_map:  bool/int  [num_kv_heads, num_blocks_row, num_blocks_col]
                block_col_sz:    int32/64  [num_kv_heads, num_blocks_col]
            Returns:
                kv_indptr:  [H*R + 1]  int32  —  CSR indptr
                kv_indices: [nnz]      int32  —  token indices per (head, row)
            """
            device = block_mask_map.device
            dtype_i = torch.int32

            # 1) Calculate the total length of each row (head, row)
            row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)  # [H,R]
            kv_indptr = torch.cat(
                [
                    torch.zeros(1, dtype=dtype_i, device=device),
                    torch.cumsum(row_lengths.flatten(), 0),
                ],
                dim=0,
            )

            # 2) Calculate the offset of each column block within its head
            col_offset = (
                torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz
            )  # [H,C]
            head_len = block_col_sz.sum(1, dtype=dtype_i)
            head_offset = torch.cumsum(head_len, 0) - head_len

            # 3) Find all selected (h,r,c)
            h_idx, r_idx, c_idx = block_mask_map.nonzero(as_tuple=True)
            lengths = block_col_sz[h_idx, c_idx].to(dtype_i)  # [N]
            base = head_offset[h_idx] + col_offset[h_idx, c_idx]  # [N]

            # 4) Expand variable-length column blocks into token-level indices
            cum = torch.cumsum(lengths, 0)
            starts = torch.repeat_interleave(cum - lengths, lengths)  # [total]
            offsets_within = torch.arange(cum[-1], device=device) - starts
            kv_indices = torch.repeat_interleave(base, lengths) + offsets_within

            return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
                dtype=dtype_i, device=device
            )

        kv_indptr, kv_indices = _block_mask_map_to_expanded_indices(
            block_mask_map, block_col_sz
        )
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
        kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = kv_indices.to(
            self.device, non_blocking=non_blocking
        )
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, non_blocking=non_blocking
        )
        torch.cuda.synchronize()  # for non-blocking copy
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

        # Sanity check
        assert num_qo_heads % num_kv_heads == 0, (
            "num_qo_heads must be a multiple of num_kv_heads"
        )
        assert num_blocks_row * num_kv_heads + 1 == kv_indptr_host.shape[0]
        assert kv_indptr_host[-1].item() == kv_indices_host.shape[0], (
            f"{kv_indptr_host[-1].item()} != {kv_indices_host.shape[0]}"
        )
        assert num_kv_heads == block_mask_map.shape[0]
        assert num_kv_heads == block_row_sz.shape[0]
        assert num_kv_heads == block_col_sz.shape[0]
        assert num_blocks_row == block_mask_map.shape[1]
        assert num_blocks_col == block_mask_map.shape[2]

        if self._backend == "auto":
            self._backend = determine_attention_backend(
                self.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,
                self._mask_mode == MaskMode.CUSTOM.value,  # use_custom_mask
                q_data_type,
                kv_data_type,
            )

        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            False,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
        )
        self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

        kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]  # page_size == 1
        self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
            kv_lens_arr_host,
        )

        args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_lens_arr_host,
            qo_indptr_host[-1].item(),  # total_num_rows
            num_blocks_row * num_kv_heads,  # batch_size
            num_qo_heads // num_kv_heads,  # num_qo_heads (gqa_group_size)
            1,  # num_kv_heads,
            1,  # page_size
            False,  # is_cuda_graph_enabled,
            head_dim,
            head_dim,
            causal,
            -1,  # window_left
        ]
        if self._backend == "fa2":
            args.append(-1)  # fixed_split_size
            args.append(False)  # disable_split_kv
            args.append(0)  # num_colocated_ctas
        self._plan_info = self._cached_module.plan(
            *args,
        )

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads
        self._gqa_group_size = num_qo_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(num_qo_heads, qo_len, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        # NOTE(Zihao): defer import of einops
        import einops

        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        # reshape to pad num_kv_heads into seq_len
        # input [num_qo_heads, qo_len, head_dim]
        # kernel layout is NHD -> [qo_len * num_kv_heads, gqa_group_size, head_dim]
        q = einops.rearrange(
            q,
            "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()
        # HND -> [kv_len * num_kv_heads (num_pages), 1 (page_size), 1 (new_num_kv_heads), head_dim]
        k = einops.rearrange(
            k,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()
        v = einops.rearrange(
            v,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        self._cached_module.paged_run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            -1,  # window_left
            enable_pdl,
            # ADDITIONAL_FUNC_PARAMS
            # Not supported yet
            None,  # packed_mask_buf
            None,  # mask_indptr_buf
            None,  # alibi_slopes_buf
            None,
            None,
            None,
            logits_soft_cap,
            sm_scale,
            None,  # scale_q
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
            0,  # token_pos_in_items_len
            self._workspace_size,
        )

        # [qo_len * num_kv_heads, gqa_group_size, head_dim] -> HND
        out = einops.rearrange(
            out,
            "(num_kv_heads qo_len) gqa_group_size head_dim -> (num_kv_heads gqa_group_size) qo_len head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()

        if return_lse:
            lse = einops.rearrange(
                lse,
                "(num_kv_heads qo_len) gqa_group_size -> (num_kv_heads gqa_group_size) qo_len",
                num_kv_heads=self._num_kv_heads,
            ).contiguous()

        return (out, lse) if return_lse else out
