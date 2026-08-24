"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from typing import ClassVar, Optional

import torch

from ._capabilities import MLAPlanCapabilities
from ._fa_common import (
    _BatchMLAPagedAttentionFaBackendBase,
    _validate_generated_fa_plan,
    get_batch_mla_module,
)


class _BatchMLAPagedAttentionFa2Backend(_BatchMLAPagedAttentionFaBackendBase):
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="fa2",
        lse_modes=frozenset({"none", "base2", "basee"}),
        kv_layouts=frozenset({"combined", "adjacent-split", "independent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "kv-per-tensor"}),
    )

    def __init__(
        self,
        *,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool,
        qo_indptr_buf: Optional[torch.Tensor],
        kv_indptr_buf: Optional[torch.Tensor],
        kv_indices_buf: Optional[torch.Tensor],
        kv_len_arr_buf: Optional[torch.Tensor],
        query_split_widths: tuple[int, int],
        kv_split_widths: tuple[int, int],
        int_workspace_buffer: Optional[torch.Tensor] = None,
        pin_memory_int_workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            backend="fa2",
            float_workspace_buffer=float_workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            kv_indptr_buf=kv_indptr_buf,
            kv_indices_buf=kv_indices_buf,
            kv_len_arr_buf=kv_len_arr_buf,
            query_split_widths=query_split_widths,
            kv_split_widths=kv_split_widths,
            int_workspace_buffer=int_workspace_buffer,
            pin_memory_int_workspace_buffer=pin_memory_int_workspace_buffer,
        )

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        output_dtype: torch.dtype,
        scale_mode: str,
        use_profiler: bool,
    ) -> None:
        _validate_generated_fa_plan(
            backend="fa2",
            device=self.device,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            output_dtype=output_dtype,
            scale_mode=scale_mode,
        )
        self._plan_generated_fa(
            module_loader=lambda: get_batch_mla_module(
                "fa2",
                q_data_type,
                kv_data_type,
                output_dtype,
                qo_indptr.dtype,
                head_dim_ckv,
                head_dim_kpe,
                use_profiler,
            ),
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            page_size=page_size,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            use_profiler=use_profiler,
        )


__all__ = ["_BatchMLAPagedAttentionFa2Backend", "get_batch_mla_module"]
