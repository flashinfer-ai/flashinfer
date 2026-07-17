"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

from typing import Callable, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError

from ._fa_common import (
    _BatchMLAGeneratedFaMechanics,
    _BatchMLAGeneratedFaWorkspace,
    get_batch_mla_module,
)


def _get_batch_mla_fa2_module(*args):
    return get_batch_mla_module("fa2", *args)


class _BatchMLAPagedAttentionFa2Backend(_BatchMLAGeneratedFaMechanics):
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        generated_fa_workspace: _BatchMLAGeneratedFaWorkspace,
        use_cuda_graph: bool,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: Optional[torch.Tensor],
        kv_indices: Optional[torch.Tensor],
        kv_len_arr: Optional[torch.Tensor],
        before_metadata_commit: Optional[Callable[[], None]],
    ) -> None:
        self._backend = "fa2"
        super().__init__(
            float_workspace_buffer,
            generated_fa_workspace,
            use_cuda_graph,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            before_metadata_commit,
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
        use_profiler: bool,
    ) -> None:
        if kv_data_type not in (torch.float16, torch.bfloat16):
            raise _BackendPlanUnsupportedError(
                f"MLA kv_data_type {kv_data_type} is not supported by the fa2 "
                "backend. Supported dtypes: "
                f"{[torch.float16, torch.bfloat16]}."
            )
        self._plan_generated_fa(
            module_loader=lambda: _get_batch_mla_fa2_module(
                q_data_type,
                kv_data_type,
                q_data_type,
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

    def run(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_run_input_dtypes(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
        )
        return self._run_generated_fa_after_input_validation(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=1.0,  # FP8 KV is unsupported by fa2.
            kpe_scale=1.0,  # FP8 KV is unsupported by fa2.
        )
