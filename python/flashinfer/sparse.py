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
from typing import Optional
import logging
import torch
from .prefill import _compute_page_qk_indptr
from .quantization import segment_packbits
from .utils import (
    _check_pos_encoding_mode,
    PosEncodingMode,
    TensorLayout,
)

# mypy: disable-error-code="attr-defined"
try:
    from . import _prefill
except ImportError as e:
    import os

    if os.environ.get("BUILD_DOC", "0") == "1":
        _prefill = None
        logging.warning("Kernels are not loaded in documentation build mode.")
    else:
        raise e


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
    >>> bsr_wrapper.begin_forward(
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
    >>> o = bsr_wrapper.forward(q, k, v)
    >>> # use dense implementation with attention mask for comparison
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")
    >>> o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_ref)
    True
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
    ) -> None:
        r"""Constructs of :class:`BlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=float_workspace_buffer.device
        )
        self._wrapper = _prefill.BatchPrefillWithPagedKVCachePyTorchWrapper(
            TensorLayout["NHD"].value,
            False,  # use_cuda_graph
        )
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._packed_mask_buf: Optional[torch.Tensor] = None
        self._qk_indptr_buf: Optional[torch.Tensor] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None

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
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._wrapper.update_page_locked_buffer_size(
            int_workspace_buffer.numel() * int_workspace_buffer.element_size()
        )

    def begin_forward(
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
        q_data_type: str = "float16",
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
        q_data_type : str, optional
            The data type of the query tensor.

        The :meth:`begin_forward` method should be called before any :meth:`forward` or
        :meth:`forward_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple forward calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        num_blocks_row = len(indptr) - 1
        qo_indptr_host = R * torch.arange(num_blocks_row + 1, dtype=torch.int32)
        qo_indptr_host[-1] = M
        self._qo_indptr = qo_indptr_host.to(indptr.device)
        if indices.max().item() * C > N:
            raise ValueError("indices out of bound")
        last_block_len = torch.full(
            (num_blocks_row,), C, dtype=torch.int32, device=indptr.device
        )

        if mask is not None or packed_mask is not None:
            qk_indptr = _compute_page_qk_indptr(
                self._qo_indptr,
                indptr,  # paged_kv_indptr
                last_block_len,  # paged_kv_last_page_len
                C,  # page_size
            )
        if packed_mask is None and mask is not None:
            # first convert BSR mask to flashinfer layout
            mask = convert_bsr_mask_layout(mask, indptr)
            # create packed mask from mask
            packed_mask, qk_indptr = segment_packbits(
                mask.contiguous().view(-1), qk_indptr, bitorder="little"
            )

        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        self._paged_kv_last_page_len = last_block_len
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask
            self._qk_indptr_buf = qk_indptr
        else:
            self._packed_mask_buf = None

        empty_q_data = torch.empty(
            0,
            dtype=(
                getattr(torch, q_data_type)
                if isinstance(q_data_type, str)
                else q_data_type
            ),
        )

        self.M = M
        self.N = N
        self.R = R
        self.C = C

        self._wrapper.begin_forward(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._qo_indptr,
            self._paged_kv_indptr_buf,
            num_blocks_row,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            C,
            empty_q_data,
        )

    def end_forward(self) -> None:
        r"""Clear the auxiliary data structures created by :meth:`begin_forward`."""
        self._qo_indptr = None
        self._paged_kv_indptr_buf = None
        self._paged_kv_indices_buf = None
        self._paged_kv_last_page_len = None
        self._packed_mask_buf = None
        self._qk_indptr_buf = None
        self.M = None
        self.N = None
        self.R = None
        self.C = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(M, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(N, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(N, num_kv_heads, head_dim)``.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        allow_fp16_qk_reduction : bool
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

        Returns
        -------
        torch.Tensor
            The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
        """
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        k = k.reshape(-1, self.C, *k.shape[-2:]).contiguous()
        v = v.reshape(-1, self.C, *v.shape[-2:]).contiguous()
        if self._packed_mask_buf is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr,
                None,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                False,  # causal
                PosEncodingMode[pos_encoding_mode].value,
                allow_fp16_qk_reduction,
                -1,  # window_left
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                False,  # return LSE
            )[0]
        else:
            return self._wrapper.forward_custom_mask(
                q,
                self._qo_indptr,
                None,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                self._packed_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                allow_fp16_qk_reduction,
                -1,  # window_left
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                False,  # return LSE
            )[0]
