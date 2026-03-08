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
import math
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .jit import gen_batch_attention_module
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    determine_attention_backend,
)
from .prefill import BatchPrefillWithPagedKVCacheWrapper
from .jit.attention.variants import attention_sink_decl
from .jit.utils import filename_safe_dtype_map


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    @flashinfer_api
    def __init__(
        self,
        kv_layout: str = "NHD",
        device: str = "cuda",
    ):
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout

        self.float_workspace_buffer = torch.empty(
            384 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    @flashinfer_api
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
        logits_soft_cap: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        use_profiler: bool = False,
    ) -> None:
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap

        # get jit module
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            logits_soft_cap > 0.0,
            use_profiler,  # different compiler path
        )
        self.module = get_holistic_attention_module(*get_module_args)

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
        self._use_profiler = use_profiler

        # No addtional buf allocated for CUDA graph tensor
        # Allocate outside FlashInfer
        self._kv_indices = kv_indices
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

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        if logits_soft_cap > 0.0 and self._logits_soft_cap <= 0.0:
            raise ValueError(
                "logits_soft_cap used in kernel run but not provided in plan(). This will cause template deduction error."
            )

        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            # lse shape: [batch_size, num_qo_heads]
            lse = torch.empty(
                q.shape[0], q.shape[1], device=q.device, dtype=torch.float32
            )
        head_dim_qk = q.shape[2]
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim_qk)
        if k_scale is not None:
            sm_scale *= k_scale
        if v_scale is None:
            v_scale = 1.0
        # profiler_buffer is optional
        profiler_args = (profiler_buffer,) if self._use_profiler else ()

        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            v_scale,
            sm_scale,
            logits_soft_cap,
            # ADDITIONAL_FUNC_PARAMS
            # PROFILER_FUNC_PARAMS
            *profiler_args,
        )

        return out, lse


class BatchAttentionWithAttentionSinkWrapper(BatchPrefillWithPagedKVCacheWrapper):
    r"""
    Wrapper for prefill and decode attention with paged KV-cache that adds support for
    attention sinks. This class extends `BatchPrefillWithPagedKVCacheWrapper`, providing
    a convenient interface for using attention sinks during prefill or decode attention.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
        window_left: int = -1,
    ) -> None:
        # trtllm is separate code path
        assert backend in ["fa2", "fa3", "auto"]
        if backend == "auto":
            # dispatch backend before init jit module
            backend = determine_attention_backend(
                float_workspace_buffer.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,  # use_fp16_qk_reduction
                custom_mask_buf is not None,  # use_custom_mask
                q_data_type,
                kv_data_type,
            )

        jit_args = [
            f"batch_prefill_attention_sink_{filename_safe_dtype_map[q_data_type]}_swa_{window_left >= 0}_{backend}",  # uri
            q_data_type,  # dtype_q
            kv_data_type,  # dtype_kv
            q_data_type,  # dtype_o
            torch.int32,  # idtype
            head_dim_qk,  # hidden_dim_qk
            head_dim_vo,  # hidden_dim_vo
            ["sink"],  # additional_tensor_names
            ["float"],  # additional_tensor_dtypes
            ["sm_scale"],  # additional_scalar_names
            ["double"],  # additional_scalar_dtypes
            "AttentionSink",
            attention_sink_decl[backend],
        ]
        jit_kwargs = {
            "use_sliding_window": window_left >= 0,
            "use_fp16_qk_reduction": use_fp16_qk_reduction,
            "pos_encoding_mode": PosEncodingMode[pos_encoding_mode].value,
        }

        super().__init__(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=paged_kv_indptr_buf,
            paged_kv_indices_buf=paged_kv_indices_buf,
            paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
            custom_mask_buf=custom_mask_buf,
            mask_indptr_buf=mask_indptr_buf,
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
