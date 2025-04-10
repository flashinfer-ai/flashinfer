"""
Copyright (c) 2025 by FlashInfer team.

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
import logging
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import FLASHINFER_CSRC_DIR, load_cuda_ops
from .prefill import BatchPrefillWithPagedKVCacheWrapper
from .utils import MaskMode, _unpack_paged_kv_cache


def get_holistic_attention_module():
    return load_cuda_ops(
        "holistic_persistent_attention",
        [
            FLASHINFER_CSRC_DIR / "batch_persistent.cu",
            FLASHINFER_CSRC_DIR / "batch_persistent_pybind.cu",
        ],
    )


class BatchAttention:
    def __init__(
        self,
        kv_layout: str = "NHD",
        kv_indices_buf: torch.Tensor = None,
    ):
        self._kv_layout = kv_layout
        self.float_workspace_buffer = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cuda"),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cuda"),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        # Used for CUDA Graph
        # All other metadata is stored in workspace buffers
        if kv_indices_buf is None:
            kv_indices_buf = torch.empty(
                8*1024*1024,
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
        self._kv_indices_buf = kv_indices_buf
        self.module = get_holistic_attention_module()

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool = False,
        sm_scale: float = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        qo_indptr_host = qo_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_indptr_host = kv_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_len_arr_host = kv_len_arr.to(torch.device("cpu"), non_blocking=True)
        torch.cuda.synchronize()
        
        batch_size = kv_len_arr.shape[0]
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._sm_scale = sm_scale
        
        # copy kv_indices into kv_indices_buf
        assert kv_indices.shape[0] <= self._kv_indices_buf.shape[0]
        self._kv_indices_buf[: kv_indices.shape[0]] = kv_indices
        
        self._plan_info = self.module.plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            # lse shape: [batch_size, num_qo_heads]
            lse = torch.empty(q.shape[0], q.shape[1], device=q.device)

        head_dim_qk = q.shape[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)

        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices_buf,
            out,
            lse,
            self._mask_mode,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self._sm_scale,
        )
        
        return out, lse
