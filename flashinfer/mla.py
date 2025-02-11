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
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import gen_batch_mla_module, get_batch_mla_uri
from .utils import MaskMode, get_cuda_stream, register_custom_op, register_fake_op

_batch_mla_modules = {}


def get_batch_mla_module(*args):
    global _batch_mla_modules
    if args not in _batch_mla_modules:
        _batch_mla_modules[args] = gen_batch_mla_module(*args)
    return _batch_mla_modules[args]


class BatchMLAPageAttentionWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "fa2",
    ) -> None:
        r"""Constructor for BatchMLAPageAttentionWrapper."""
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: torch.Tensor,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
    ) -> None:
        self._cached_module = get_batch_mla_module(
            q_data_type,
            kv_data_type,
            q_data_type,
            qo_indptr.dtype,
            head_dim_ckv,
            head_dim_kpe,
        )
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")

        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale

        with self.device as device:
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_len_arr_host,
                num_heads,
                head_dim_ckv,  # head_dim_o
                causal,
                get_cuda_stream(device),
            )

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor:
        ...
    
    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_heads = q_nope.shape[1]
        page_size = self._page_size
        sm_scale = self._sm_scale
        causal = self._causal
        mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        with self.device as device:
            o = torch.empty_like(q_nope)
            maybe_lse = (
                torch.empty(q_nope.shape[:2], dtype=torch.float32, device=device)
                if return_lse
                else None
            )
            self._cached_module.run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                self._kv_indices_buf,
                o,
                maybe_lse,
                mask_mode,
                num_heads,
                page_size,
                sm_scale,
                get_cuda_stream(device),
            )

        return (o, maybe_lse) if return_lse else o
