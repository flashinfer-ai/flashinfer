"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import math
from typing import Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.utils import get_compute_capability

from .._planning import (
    _audit_plan_from_wrapper_arguments,
    _MLAGeneratedFaWorkspace,
    _MLAPlanArguments,
    _MLAWrapperPlanResult,
)
from ._fa_common import (
    _BatchMLAGeneratedFaMechanics,
    get_batch_mla_module,
)


def _get_batch_mla_fa3_module(*args):
    return get_batch_mla_module("fa3", *args)


class _BatchMLAPagedAttentionFa3Backend(_BatchMLAGeneratedFaMechanics):
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        generated_fa_workspace: _MLAGeneratedFaWorkspace,
        use_cuda_graph: bool,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: Optional[torch.Tensor],
        kv_indices: Optional[torch.Tensor],
        kv_len_arr: Optional[torch.Tensor],
    ) -> None:
        self._backend = "fa3"
        super().__init__(
            float_workspace_buffer,
            generated_fa_workspace,
            use_cuda_graph,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
        )

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        args._generated_fa_workspace.raise_if_invalid()
        enable_pdl = args.enable_pdl
        is_var_seq = args.is_var_seq
        cute_dsl_impl = args.cute_dsl_impl
        use_sinks = args.use_sinks
        qk_nope_head_dim = args.qk_nope_head_dim
        if enable_pdl:
            raise ValueError("enable_pdl is not supported by the fa3 wrapper backend.")
        if is_var_seq is not None:
            raise ValueError("is_var_seq is not supported by the fa3 wrapper backend.")
        if cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the fa3 wrapper backend."
            )
        if use_sinks:
            raise ValueError("use_sinks is not supported by the fa3 wrapper backend.")
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is only supported with trtllm-gen backend."
            )
        csr = args.csr
        backend = cls(
            args._float_workspace_buffer,
            args._generated_fa_workspace,
            args._use_cuda_graph,
            args._qo_indptr_buf,
            args._kv_indptr_buf,
            args._kv_indices_buf,
            args._kv_len_arr_buf,
        )
        backend.plan(
            qo_indptr=csr.qo_indptr,
            kv_indptr=csr.kv_indptr,
            kv_indices=csr.kv_indices,
            kv_len_arr=csr.kv_len_arr,
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            page_size=args.page_size,
            causal=args.causal,
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            use_profiler=args.use_profiler,
        )
        return _MLAWrapperPlanResult(backend_impl=backend)

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
        supported_kv_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
        )
        if kv_data_type not in supported_kv_dtypes:
            raise _BackendPlanUnsupportedError(
                f"MLA kv_data_type {kv_data_type} is not supported. "
                f"Supported dtypes: {list(supported_kv_dtypes)}."
            )
        if kv_data_type == torch.float8_e4m3fn:
            major, minor = get_compute_capability(self.device)
            if major != 9:
                raise _BackendPlanUnsupportedError(
                    "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                    f"got SM{major}{minor}."
                )
            if q_data_type != torch.bfloat16:
                raise _BackendPlanUnsupportedError(
                    "FP8 kv_data_type for MLA currently only supports "
                    f"q_data_type=torch.bfloat16, got {q_data_type}."
                )
            if head_dim_ckv != 512 or head_dim_kpe != 64:
                raise _BackendPlanUnsupportedError(
                    "FP8 kv_data_type for MLA currently only supports "
                    "head_dim_ckv=512 and head_dim_kpe=64 (DeepSeek MLA), got "
                    f"head_dim_ckv={head_dim_ckv}, head_dim_kpe={head_dim_kpe}."
                )
        self._plan_generated_fa(
            module_loader=lambda: _get_batch_mla_fa3_module(
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
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_run_input_dtypes(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
        )
        if self._kv_data_type == torch.float8_e4m3fn:
            if ckv_scale is None or kpe_scale is None:
                raise ValueError(
                    "ckv_scale and kpe_scale are required when kv_data_type is FP8."
                )
            ckv_scale_f = float(ckv_scale)
            kpe_scale_f = float(kpe_scale)
            if not math.isfinite(ckv_scale_f) or ckv_scale_f <= 0.0:
                raise ValueError(
                    f"ckv_scale must be a finite positive value, got {ckv_scale}"
                )
            if not math.isfinite(kpe_scale_f) or kpe_scale_f <= 0.0:
                raise ValueError(
                    f"kpe_scale must be a finite positive value, got {kpe_scale}"
                )
        else:
            if ckv_scale is not None or kpe_scale is not None:
                raise ValueError(
                    "ckv_scale / kpe_scale are only valid when kv_data_type is FP8."
                )
            ckv_scale_f = 1.0
            kpe_scale_f = 1.0

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
            ckv_scale=ckv_scale_f,
            kpe_scale=kpe_scale_f,
        )

    def run_from_wrapper(
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
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._generated_fa_workspace.raise_if_invalid()
        if sinks is not None:
            raise ValueError("sinks are not supported by the fa3 wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "fa3 wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError("bmm1_scale is not supported by the fa3 wrapper backend.")
        if bmm2_scale is not None:
            raise ValueError("bmm2_scale is not supported by the fa3 wrapper backend.")
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
        return self.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=ckv_scale,
            kpe_scale=kpe_scale,
        )
