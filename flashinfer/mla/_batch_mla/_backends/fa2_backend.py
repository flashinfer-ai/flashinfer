"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

from typing import Any, Optional, Sequence, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError

from ._capabilities import MLAPlanCapabilities, validate_plan_capabilities
from .._planning import (
    _MLAGeneratedFaWorkspace,
    _MLAPlanArguments,
)
from .._contracts import (
    _FunctionalBackendUnsupportedError,
    _FunctionalMLARequest,
    MLAKVCache,
    MLAInputContract,
    MLAQuery,
    _split_mla_value_objects,
)
from ._fa_common import (
    _BatchMLAGeneratedFaMechanics,
    _GeneratedFaMlaRunner,
    get_batch_mla_module,
)


def _get_batch_mla_fa2_module(*args):
    return get_batch_mla_module("fa2", *args)


class _BatchMLAPagedAttentionFa2Backend(_BatchMLAGeneratedFaMechanics):
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="fa2",
        lse_modes=frozenset({"none", "base2", "basee"}),
        kv_layouts=frozenset({"combined", "adjacent-split", "independent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default"}),
    )

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        generated_fa_workspace: _MLAGeneratedFaWorkspace,
        use_cuda_graph: bool,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: Optional[torch.Tensor],
        kv_indices: Optional[torch.Tensor],
        kv_len_arr: Optional[torch.Tensor],
        input_contract: Optional[MLAInputContract] = None,
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
            input_contract,
        )

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionFa2Backend":
        validate_plan_capabilities(args, cls._plan_capabilities)
        output_dtype = args.output_dtype
        if output_dtype != args.q_data_type:
            raise _BackendPlanUnsupportedError(
                "fa2 backend does not support this output contract; it only "
                "supports q_data_type output without o_scale."
            )
        args._generated_fa_workspace.raise_if_invalid()
        csr = args.csr
        backend = cls(
            args._float_workspace_buffer,
            args._generated_fa_workspace,
            args._use_cuda_graph,
            args._qo_indptr_buf,
            args._kv_indptr_buf,
            args._kv_indices_buf,
            args._kv_len_arr_buf,
            args.input_contract,
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
        return backend

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
        if use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the fa2 backend."
            )
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

    def run_from_wrapper(
        self,
        *,
        query: MLAQuery,
        kv: MLAKVCache,
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
        q_nope, q_pe, ckv_cache, kpe_cache = _split_mla_value_objects(
            query, kv, getattr(self, "_input_contract", None)
        )
        if sinks is not None:
            raise ValueError("sinks are not supported by the fa2 wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "fa2 wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError("bmm1_scale is not supported by the fa2 wrapper backend.")
        if bmm2_scale is not None:
            raise ValueError("bmm2_scale is not supported by the fa2 wrapper backend.")
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
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
        )


class Fa2MlaRunner(_GeneratedFaMlaRunner):
    """Direct functional generated-FA2 MLA runner."""

    backend_name = "fa2"

    def _load_functional_module(self, module_args: Sequence[Any]) -> Any:
        return get_batch_mla_module("fa2", *module_args)

    def _validate_backend_capability(self, request: _FunctionalMLARequest) -> None:
        if request.kv_cache.dtype not in (torch.float16, torch.bfloat16):
            raise _FunctionalBackendUnsupportedError(
                f"MLA kv_data_type {request.kv_cache.dtype} is not supported by the fa2 "
                f"backend. Supported dtypes: {[torch.float16, torch.bfloat16]}."
            )
