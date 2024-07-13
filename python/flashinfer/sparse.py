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
import torch
import logging
from .prefill import _compute_qk_indptr
from .quantization import segment_packbits
from .utils import check_pos_encoding_mode, check_kv_layout, is_float8, expand_5d, PosEncodingMode, TensorLayout

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


class BlockSparseFlashAttentionWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
    ):
        r"""Constructs of :class:`BlockSparseFlashAttentionWrapper`. 

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store auxiliary data structures,
            recommended size is 128MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._workspace_buffer = workspace_buffer
        self._wrapper = _kernels.BatchPrefillWithPagedKVCachePyTorchWrapper(
            TensorLayout[kv_layout].value,
            False, # use_cuda_graph
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
    ):
        num_rows = len(indptr) - 1
        qo_indptr_host = R * torch.arange(num_rows + 1, dtype=torch.int32)
        qo_indptr_host[-1] = M
        self._qo_indptr = qo_indptr_host.to(indptr.device)
        if mask is not None or packed_mask is not None:
            qk_indptr = _compute_qk_indptr(
                self._qo_indptr,
                indptr, # paged_kv_indptr
                indices, # paged_kv_last_page_len
                C # page_size
            )
        if packed_mask is None and mask is not None:
            # create packed mask from mask
            packed_mask, qk_indptr = segment_packbits(
                mask.contiguous().view(-1),
                qk_indptr,
                bitorder="little"
            )

        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        if indices.max().item() * C > N:
            raise ValueError("indices out of bound")
        
        row_empty = indptr[1:] == indptr[:1]
        last_block_pos = indices[torch.clamp(indptr[1:], min=1) - 1]
        last_block_pos.masked_fill_(row_empty, 0)

        self._paged_kv_last_page_len = torch.clamp(N - last_block_pos * C, max=C)
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask
            self._qk_indptr_buf = qk_indptr
        
        empty_q_data = torch.empty(
            0,
            dtype=(
                getattr(torch, q_data_type)
                if isinstance(q_data_type, str)
                else q_data_type
            ),
        )

        self._wrapper.begin_forward(
            self._workspace_buffer,
            self._qo_indptr,
            self._paged_kv_indptr_buf,
            num_rows,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            C,
            empty_q_data
        )

    def end_forward(self):
        self._qo_indptr = None
        self._paged_kv_indptr_buf = None
        self._paged_kv_indices_buf = None
        self._paged_kv_last_page_len = None
        self._packed_mask_buf = None
        self._qk_indptr_buf = None

    def forward(
        self,
        q: torch.Tensor,
        kv_data: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        allow_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ):
        check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
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
            kv_data = kv_data.to(torch.float16)

        kv_data = expand_5d(kv_data, self._kv_layout)

        if self._packed_mask_buf is None:
            return self._wrapper.forward(
                q,
                self._qo_indptr,
                kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                False, # causal
                PosEncodingMode[pos_encoding_mode].value,
                allow_fp16_qk_reduction,
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
                kv_data,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                self._packed_mask_buf,
                self._qk_indptr_buf,
                PosEncodingMode[pos_encoding_mode].value,
                allow_fp16_qk_reduction,
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                False,  # return LSE
            )
