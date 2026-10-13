"""Shared functional, planned and tunable monolithic CuTe DSL MLA backend."""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional

import torch

from .._planning import _MLAPlanArguments
from ._capabilities import MLAPlanCapabilities
from ._cute_dsl_common import (
    _BatchMLAPagedAttentionCuteDslBackendBase,
    _CuteDslKernelUnsupportedError,
)


class _BatchMLAPagedAttentionCuteDslMonolithicBackend(
    _BatchMLAPagedAttentionCuteDslBackendBase
):
    """Concrete monolithic CuTe DSL MLA backend."""

    _profile_lse: Optional[torch.Tensor]
    _lse_scale: float
    return_lse_base: Optional[Literal["basee", "base2"]]
    enable_dcp: bool
    cp_world: int
    cp_rank: int
    _backend_name = "cute-dsl-monolithic"
    _supports_lse = True
    _supports_variable_q = True
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="cute-dsl-monolithic",
        lse_modes=frozenset({"none", "basee", "base2"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar"}),
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    # Wrapper preparation and execution are shared with the CuTe adapter base.

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCuteDslMonolithicBackend":
        backend = super().plan_from_wrapper(args)
        # The public run contract fixes the LSE base at plan time.
        backend._lse_scale = math.log(2.0) if args.lse_mode == "basee" else 1.0
        return backend

    # Functional entrypoint. This never performs wrapper planning or host reads.

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
        from flashinfer.cute_dsl.attention.monolithic import (
            mla_decode as implementation,
        )

        backend = cls(workspace_buffer)
        backend._is_planned = False
        backend._functional_run = implementation.cute_dsl_mla_decode
        backend.kv_cache = kv_cache
        backend.kv_lora_rank = kv_lora_rank
        backend.qk_nope_head_dim = qk_nope_head_dim
        backend.qk_rope_head_dim = qk_rope_head_dim
        # kv_cache may be 3D [num_pages, page_size, D] or 4D
        # [num_pages, 1, page_size, D] after `_check_mla_functional_shape` at
        # dispatcher level — page_size is the second-to-last dim in both.
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
        backend.return_lse_base = return_lse_base
        backend.sinks = sinks
        backend.cute_dsl_impl = cute_dsl_impl
        backend.enable_dcp = enable_dcp
        backend.cp_world = cp_world
        backend.cp_rank = cp_rank
        backend._profile_lse = None
        backend._workspace_sizer = implementation._get_split_kv_and_workspace_size
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
        if use_sinks:
            raise _CuteDslKernelUnsupportedError(
                "cute-dsl-monolithic does not support sinks."
            )
        try:
            from flashinfer.cute_dsl.attention.monolithic import (
                mla_decode as implementation,
            )
        except ImportError as error:
            raise _CuteDslKernelUnsupportedError(str(error)) from error
        try:
            _, num_q_tiles, _ = implementation.compute_q_tile_layout(num_heads, q_len)
            implementation._validate_nonpersistent_grid_y(
                batch_size, num_q_tiles, not resolved_is_var_seq
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
            split_kv, workspace_size = implementation._get_split_kv_and_workspace_size(
                batch_size,
                q_len,
                num_heads,
                head_dim_ckv,
                implementation.get_num_sm(device),
                max_seq_len=max_seq_len,
                occupancy_q_tiles=(
                    max(1, min(total_q, batch_size * num_q_tiles)) if is_var_q else None
                ),
            )
            if workspace_size:
                # The planned reducer launches one grid-Y slot per query.
                try:
                    implementation._validate_nonpersistent_grid_y(1, q_len, False)
                except ValueError as error:
                    raise ValueError(f"split-KV reducer: {error}") from error
        except (ImportError, ValueError) as error:
            raise _CuteDslKernelUnsupportedError(str(error)) from error
        # Compilation errors must propagate rather than exclude the candidate.
        # Use one D tile and specialize reducer capacity to the planned split count.
        compiled_kernel = implementation._get_compiled_mla_kernel(
            arch=implementation.cute_dsl_compile_arch(
                *implementation.get_compute_capability(device)
            ),
            torch_dtype=q_data_type,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            num_heads=num_heads,
            seq_len_q=q_len,
            is_persistent=not resolved_is_var_seq,
            is_var_seq=resolved_is_var_seq,
            is_var_q=is_var_q,
            is_var_split_kv=False,
            reducer_max_splits=implementation._get_reducer_max_splits(split_kv),
            is_workspace_size_zero=workspace_size == 0,
            enable_pdl=enable_pdl,
        )
        return (
            implementation,
            compiled_kernel,
            implementation._as_cute_dsl_workspace_i8(workspace_buffer),
            workspace_size,
            split_kv,
        )

    def _launch_compiled_kernel(
        self, launch_args: tuple[Any, ...], sinks: Optional[torch.Tensor]
    ) -> None:
        if sinks is not None:
            raise ValueError("cute-dsl-monolithic does not support sinks.")
        state = self._execution_state
        state.compiled_kernel(
            *launch_args[:10],
            state.cum_seq_lens_q,
            None,  # no wrapper DCP contract
            state.Int32(0),
            *launch_args[10:],
            state.Float32(self._lse_scale),
        )

    # Autotuning support for functional profiles and the current wrapper plan.

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        if self._is_planned:
            return self._get_planned_valid_tactics(inputs)
        # Workspace-bound: cute-dsl's per-CTA split-K state grows with B.
        # If the caller's workspace can't fit batch=B for this profile, opt
        # out so the autotuner skips us (no JIT cost) and trtllm-gen wins by
        # default for that bucket.
        from flashinfer.cute_dsl.utils import get_num_sm

        q = inputs[0]
        B, q_len, num_heads, _ = q.shape
        _, ws = self._workspace_sizer(
            B,
            q_len,
            num_heads,
            self.kv_lora_rank,
            get_num_sm(q.device),
            self.max_seq_len,
        )
        workspace_bytes = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )
        if ws > workspace_bytes:
            return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        if self._is_planned:
            return ("planned", self._planned_tuning_key)
        q, _, _, out = inputs[:4]
        # Preserve the functional workload facts. K-tile count and workspace
        # capacity prevent reuse across different split-workspace decisions.
        sinks_key = (
            None if self.sinks is None else (tuple(self.sinks.shape), self.sinks.dtype)
        )
        workspace_bytes = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )
        seq_len_workspace_key = (self.max_seq_len + 127) // 128
        return (
            q.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            seq_len_workspace_key,
            workspace_bytes,
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            sinks_key,
            self.cute_dsl_impl,
            self.enable_dcp,
            self.cp_world,
        )

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        run_options=None,
        **kwargs,
    ):
        if tactic != -1:
            raise ValueError(f"Unsupported monolithic CuTe MLA tactic: {tactic!r}.")
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
        causal_seqlens_kv_global = inputs[4] if self.enable_dcp else None

        # LSE is not a tuning input because it does not influence tactic
        # selection. When a synthetic batch differs from the caller batch,
        # provide matching temporary storage while retaining the caller's LSE
        # for the final invocation.
        lse = self.lse
        lse_scale = 1.0 if self.return_lse_base == "base2" else math.log(2.0)
        if self.return_lse:
            expected_numel = query.numel() // query.shape[-1]
            if lse is None or lse.numel() != expected_numel:
                expected_shape = (
                    expected_numel // query.shape[-2],
                    query.shape[-2],
                )
                if (
                    self._profile_lse is None
                    or tuple(self._profile_lse.shape) != expected_shape
                ):
                    self._profile_lse = torch.empty(
                        expected_shape,
                        dtype=torch.float32,
                        device=query.device,
                    )
                lse = self._profile_lse
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
            lse=lse,
            return_lse=self.return_lse,
            lse_scale=lse_scale,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            enable_dcp=self.enable_dcp,
            cp_world=self.cp_world,
            cp_rank=self.cp_rank,
            causal_seqlens_kv_global=causal_seqlens_kv_global,
        )
