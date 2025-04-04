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

from .prefill import BatchPrefillWithPagedKVCacheWrapper


def get_holistic_attention_module():
    pass


class BatchAttention:
    def __init__(
        self,
        kv_layout: str = "NHD",
    ):
        self.float_workspace_buffer = torch.empty(
            256 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cuda"),
        )
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=False,
        )

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
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype = torch.float16,
    ) -> None:
        last_page_len = (kv_len_arr - 1) % page_size + 1
        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            head_dim_vo=head_dim_vo,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            non_blocking=True,
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.wrapper.run(
            q,
            kv_cache,
            out,
            lse,
        )
