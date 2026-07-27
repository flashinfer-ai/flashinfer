"""Monolithic CuTe DSL backend for planned MLA decode."""

from typing import Any, Optional, Tuple

import torch

from ._cute_dsl_common import _BatchMLAPagedAttentionCuteDslBackendBase


class _BatchMLAPagedAttentionCuteDslMonolithicBackend(
    _BatchMLAPagedAttentionCuteDslBackendBase
):
    """Concrete monolithic CuTe DSL MLA backend."""

    _backend_name = "cute-dsl-monolithic"
    _supports_lse = True

    def _compile_kernel(
        self,
        *,
        q_data_type: torch.dtype,
        out_dtype: torch.dtype,
        page_size: int,
        batch_size: int,
        num_heads: int,
        q_len: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        resolved_is_var_seq: bool,
        use_sinks: bool,
        enable_pdl: bool,
    ) -> Tuple[Any, Any, torch.Tensor, int, int]:
        if use_sinks:
            raise ValueError(
                "cute-dsl-monolithic does not support sinks; use cute-dsl-modular."
            )
        from flashinfer.cute_dsl.attention.monolithic import (
            mla_decode as implementation,
        )

        implementation._check_can_implement(
            torch_dtype=q_data_type,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            is_persistent=not resolved_is_var_seq,
            is_var_seq=resolved_is_var_seq,
            is_var_split_kv=False,
        )
        workspace_i8 = implementation._as_cute_dsl_workspace_i8(
            self._float_workspace_buffer
        )
        split_kv, workspace_size = implementation._get_split_kv_and_workspace_size(
            batch_size,
            q_len,
            num_heads,
            head_dim_ckv,
            implementation.get_num_sm(self.device),
        )
        compiled_kernel = implementation._get_compiled_mla_kernel(
            torch_dtype=q_data_type,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            num_heads=num_heads,
            seq_len_q=q_len,
            is_persistent=not resolved_is_var_seq,
            is_var_seq=resolved_is_var_seq,
            is_var_split_kv=False,
            is_workspace_size_zero=workspace_size == 0,
            enable_pdl=enable_pdl,
        )
        return (
            implementation,
            compiled_kernel,
            workspace_i8,
            workspace_size,
            split_kv,
        )

    def _launch_compiled_kernel(
        self, launch_args: Tuple[Any, ...], sinks: Optional[torch.Tensor]
    ) -> None:
        if sinks is not None:
            raise ValueError("cute-dsl-monolithic does not support sinks.")
        self._compiled_kernel(*launch_args)
