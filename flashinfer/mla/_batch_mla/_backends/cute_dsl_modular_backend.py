"""Functional, planned and tunable modular CuTe DSL MLA backend."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

import torch

from ..._utils import _round_to_seq_len_bucket
from ._capabilities import MLAPlanCapabilities
from ._cute_dsl_common import (
    _BatchMLAPagedAttentionCuteDslBackendBase,
    _CuteDslKernelUnsupportedError,
)


class _BatchMLAPagedAttentionCuteDslModularBackend(
    _BatchMLAPagedAttentionCuteDslBackendBase
):
    """Concrete modular CuTe DSL MLA backend."""

    _backend_name = "cute-dsl-modular"
    _reject_cuda_graph = True
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="cute-dsl-modular",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar"}),
        # The native modular reduction counts sink mass once per N warp
        # partition. Reject until the kernel implements a single sink mass.
        supports_sinks=False,
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    # Wrapper preparation and execution are inherited from the CuTe base.

    @classmethod
    def from_functional(
        cls,
        *,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        max_seq_len: int,
        softmax_scale: float,
        output_scale: float,
        out_dtype: torch.dtype,
        enable_pdl: bool,
        is_var_seq: bool,
        uses_shared_paged_kv_idx: bool,
        lse: Optional[torch.Tensor],
        return_lse: bool,
        return_lse_base: Optional[Literal["basee", "base2"]],
        sinks: Optional[torch.Tensor],
        cute_dsl_impl: str,
        enable_dcp: bool = False,
        cp_world: int = 1,
        cp_rank: int = 0,
    ):
        from flashinfer.cute_dsl.attention.wrappers import batch_mla as implementation

        if enable_dcp or cp_world != 1 or cp_rank != 0:
            raise ValueError("cute-dsl-modular does not support DCP.")
        backend = cls(workspace_buffer)
        backend._is_planned = False
        backend._functional_run = implementation.cute_dsl_mla_decode
        backend._workspace_sizer = implementation._get_split_kv_and_workspace_size
        backend.kv_cache = kv_cache
        backend.kv_lora_rank = kv_lora_rank
        backend.qk_nope_head_dim = qk_nope_head_dim
        backend.qk_rope_head_dim = qk_rope_head_dim
        backend.page_size = kv_cache.shape[-2]
        backend.max_seq_len = max_seq_len
        backend.softmax_scale = softmax_scale
        backend.output_scale = output_scale
        backend.out_dtype = out_dtype
        backend.enable_pdl = enable_pdl
        backend.is_var_seq = is_var_seq
        backend.uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        backend.lse = lse
        backend.return_lse = return_lse
        backend.sinks = sinks
        backend.cute_dsl_impl = cute_dsl_impl
        return backend

    # Planned kernel preparation and native launch.

    def _compile_kernel(
        self,
        *,
        workspace_buffer: torch.Tensor,
        device: torch.device,
        q_data_type: torch.dtype,
        out_dtype: torch.dtype,
        page_size: int,
        batch_size: int,
        num_heads: int,
        q_len: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        resolved_is_var_seq: bool,
        is_var_q: bool,
        total_q: int,
        max_seq_len: int,
        use_sinks: bool,
        enable_pdl: bool,
    ) -> tuple[Any, Any, torch.Tensor, int, int]:
        try:
            from flashinfer.cute_dsl.attention.fusion.variant import AttentionWithSink
            from flashinfer.cute_dsl.attention.wrappers import (
                batch_mla as implementation,
            )
        except ImportError as error:
            raise _CuteDslKernelUnsupportedError(str(error)) from error

        try:
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
        except (ImportError, ValueError) as error:
            raise _CuteDslKernelUnsupportedError(str(error)) from error
        workspace_i8 = implementation._as_cute_dsl_workspace_i8(workspace_buffer)
        split_kv, workspace_size = implementation._get_split_kv_and_workspace_size(
            batch_size,
            q_len,
            num_heads,
            head_dim_ckv,
            implementation.get_num_sm(device),
        )
        variant = None
        params_shape = None
        if use_sinks:
            placeholder = torch.empty((num_heads,), dtype=torch.float32, device=device)
            variant = AttentionWithSink(placeholder)
            params_shape = tuple(placeholder.shape)
        compiled_kernel = implementation._compile_mla_kernel(
            torch_dtype=q_data_type,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            is_persistent=not resolved_is_var_seq,
            is_var_seq=resolved_is_var_seq,
            is_var_split_kv=False,
            is_workspace_size_zero=workspace_size == 0,
            enable_pdl=enable_pdl,
            variant=variant,
            params_shape=params_shape,
        )
        return (
            implementation,
            compiled_kernel,
            workspace_i8,
            workspace_size,
            split_kv,
        )

    def _launch_compiled_kernel(
        self, launch_args: tuple[Any, ...], sinks: Optional[torch.Tensor]
    ) -> None:
        self._execution_state.compiled_kernel(*launch_args, sinks)

    # Autotuning support for functional profiles and the current wrapper plan.

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        if self._is_planned:
            return self._get_planned_valid_tactics(inputs)
        from flashinfer.cute_dsl.utils import get_num_sm

        query = inputs[0]
        batch_size, q_len, num_heads, _ = query.shape
        _, required = self._workspace_sizer(
            batch_size, q_len, num_heads, self.kv_lora_rank, get_num_sm(query.device)
        )
        workspace_bytes = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )
        return [-1] if required <= workspace_bytes else []

    def get_cache_key_extras(self, inputs):
        if self._is_planned:
            return ("planned", self._planned_tuning_key)
        query, _, _, out = inputs[:4]
        sinks_key = (
            None if self.sinks is None else (tuple(self.sinks.shape), self.sinks.dtype)
        )
        return (
            query.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            _round_to_seq_len_bucket(self.max_seq_len),
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size(),
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            sinks_key,
            self.cute_dsl_impl,
        )

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        run_options=None,
        **kwargs,
    ):
        if tactic != -1:
            raise ValueError(f"Unsupported modular CuTe MLA tactic: {tactic!r}.")
        if self._is_planned:
            query, kv_cache, out, lse, sinks = inputs[:5]
            return self.run_from_wrapper(
                query=query,
                kv_cache=kv_cache,
                out=out,
                lse=lse,
                sinks=sinks,
                profiler_buffer=None,
                kv_len=None,
                page_table=None,
                ckv_scale_arr=None,
                **(self._planned_run_options if run_options is None else run_options),
            )
        query, block_tables, seq_lens, out = inputs[:4]
        return self._functional_run(
            query=query,
            kv_cache=self.kv_cache,
            workspace_buffer=self._float_workspace_buffer,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=self.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            out=out,
            out_dtype=self.out_dtype,
            is_var_seq=self.is_var_seq,
            enable_pdl=self.enable_pdl,
            lse=self.lse,
            return_lse=self.return_lse,
            sinks=self.sinks,
        )
