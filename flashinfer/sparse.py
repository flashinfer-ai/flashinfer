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

from .decode import get_batch_decode_module
from .page import block_sparse_indices_to_vector_sparse_offsets
from .prefill import _compute_page_mask_indptr, get_batch_prefill_module
from .quantization import segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
    _get_cache_alibi_slopes_buf,
    canonicalize_torch_dtype,
    determine_attention_backend,
    get_cuda_stream,
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
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        if backend in ["fa3", "auto"]:
            # NOTE(Zihao): assume maximum accumulate kv length is 4M
            self._vector_sparse_indices_buffer = torch.empty(
                (4 * 1024 * 1024,), dtype=torch.int32, device=self.device
            )
            # NOTE(Zihao): assume maximum batch size is 32768
            self._vector_sparse_indptr_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
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
            pin_memory=True,
        )

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
        non_blocking: bool = False,
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
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``False``.
            If ``True``, user should synchronize before calling :meth:`run` or cuda graph replay.


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
        ):
            # If the operation is not compute-bound, we use the cuda-core implementation
            self._use_tensor_cores = False
            self._cached_module = get_batch_decode_module(
                q_data_type,
                kv_data_type,
                q_data_type,
                indptr.dtype,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
            )

            with self.device as device:
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
                    torch.empty(0, dtype=q_data_type),
                    torch.empty(0, dtype=kv_data_type),
                    get_cuda_stream(device),
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
                q_data_type,
                indptr.dtype,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(self._backend)(
                *get_module_args
            )

            kv_lens_arr_host = (kv_indptr_host[1:] - kv_indptr_host[:-1]) * self.C
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host, non_blocking=non_blocking
            )

            if self._backend == "fa3":
                if self.C != 1:
                    vector_sparse_indptr_host = torch.cat(
                        [
                            torch.tensor([0], dtype=torch.int32),
                            torch.cumsum(kv_lens_arr_host, dim=0, dtype=torch.int32),
                        ],
                        dim=0,
                    )
                    self._vector_sparse_indptr_buffer[
                        : len(vector_sparse_indptr_host)
                    ].copy_(vector_sparse_indptr_host, non_blocking=non_blocking)
                    kv_indptr_host = vector_sparse_indptr_host

            with self.device as device:
                self._plan_info = self._cached_module.plan(
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
                    causal,
                    get_cuda_stream(device),
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

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_lse: bool = False,
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
        return_lse : bool
            Whether to return the logsumexp of attention output

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
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

        stride_block = k.stride(0)
        stride_n = k.stride(1)

        lse = None
        if return_lse:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
            )

        out = torch.empty_like(q)
        if self._use_tensor_cores:
            if self._backend == "fa3":
                sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                    self._paged_kv_indices_buf,
                    self._paged_kv_indptr_buf,
                    self._vector_sparse_indices_buffer,  # output
                    self._vector_sparse_indptr_buffer,
                    self._kv_lens_buffer,
                    stride_block // stride_n,
                    1,  # stride_n // stride_n
                    self.C,  # block_size
                )
                sparse_indptr = self._vector_sparse_indptr_buffer
            else:
                sparse_indices = self._paged_kv_indices_buf
                sparse_indptr = self._paged_kv_indptr_buf

            self._cached_module.paged_run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._qo_indptr,
                sparse_indptr,
                sparse_indices,
                self._paged_kv_last_page_len,
                out,
                lse,
                self._mask_mode,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                self._packed_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
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
                lse,
                out,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
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
