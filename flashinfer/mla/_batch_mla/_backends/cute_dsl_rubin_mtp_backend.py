"""Planned wrapper adapter for Rubin absorbed MLA MTP decode."""

from __future__ import annotations

from dataclasses import replace

import torch

from flashinfer.cute_dsl._mla_validation import _validate_mtp_scales
from flashinfer.utils import get_compute_capability

from .._contracts import _resolve_structural_mla_input
from ._capabilities import _BackendPlanUnsupportedError
from ._cute_dsl_common import (
    _CuteDslKernelUnsupportedError,
    _validate_scalar_bmm_scales,
)
from .cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)


class _BatchMLAPagedAttentionCuteDslRubinMtpBackend(
    _BatchMLAPagedAttentionCuteDslMonolithicBackend
):
    """Causal, uniform Q2/Q4 FP8 MTP with live per-request KV lengths.

    Positive tactics are static split budgets. They never depend on device
    length values, so a retained tactic remains usable as those values change.
    """

    _backend_name = "cute-dsl-rubin-mtp"
    _supports_variable_q = False
    _always_live_kv = True
    _split_tactic_schema = "rubin-mtp-live-split-v1"
    _plan_capabilities = replace(
        _BatchMLAPagedAttentionCuteDslMonolithicBackend._plan_capabilities,
        backend_name=_backend_name,
    )

    @staticmethod
    def _implementation():
        from flashinfer.cute_dsl.attention.rubin_mtp import mla_decode

        return mla_decode

    @classmethod
    def preflight_plan_from_wrapper(cls, args):
        if get_compute_capability(args._float_workspace_buffer.device) != (10, 7):
            raise _BackendPlanUnsupportedError("cute-dsl-rubin-mtp requires SM107.")
        if (
            args.q_data_type != torch.float8_e4m3fn
            or args.kv_data_type != torch.float8_e4m3fn
            or args.output_dtype != torch.float8_e4m3fn
            or args.num_heads != 128
            or args.head_dim_ckv != 512
            or args.head_dim_kpe != 64
            or args.page_size not in (64, 128)
        ):
            raise _BackendPlanUnsupportedError(
                "cute-dsl-rubin-mtp requires FP8 E4M3 query/KV/output, H=128, "
                "latent512+RoPE64 and page64/128."
            )
        super().preflight_plan_from_wrapper(args)
        try:
            from flashinfer.cute_dsl.availability import cute_dsl_compile_arch

            arch = cute_dsl_compile_arch(10, 7)
        except (ImportError, NotImplementedError) as error:
            raise _BackendPlanUnsupportedError(str(error)) from error
        if not arch.startswith("sm_107"):
            raise _BackendPlanUnsupportedError(
                "cute-dsl-rubin-mtp requires a compiler with native SM107 support."
            )
        try:
            _validate_mtp_scales(args.sm_scale)
        except ValueError as error:
            raise _BackendPlanUnsupportedError(str(error)) from error

    def _plan(self, **kwargs):
        try:
            implementation = self._implementation()
            for name in ("cum_seq_lens_q", "block_tables", "seq_lens"):
                implementation._check_tensor_indexing(kwargs[name], name)
        except (ImportError, ValueError) as error:
            raise _BackendPlanUnsupportedError(str(error)) from error
        super()._plan(**kwargs)

    def _compile_kernel(
        self,
        *,
        workspace_buffer,
        device,
        q_data_type,
        out_dtype,
        page_size,
        batch_size,
        num_heads,
        q_len,
        head_dim_ckv,
        head_dim_kpe,
        resolved_is_var_seq,
        is_var_q,
        total_q,
        max_seq_len,
        use_sinks,
        enable_pdl,
        num_kv_splits=None,
    ):
        if is_var_q or use_sinks or enable_pdl or q_len not in (2, 4):
            raise _CuteDslKernelUnsupportedError(
                "cute-dsl-rubin-mtp requires uniform Q2/Q4 without sinks or PDL."
            )
        try:
            implementation = self._implementation()
            implementation._check_can_implement(
                torch_dtype=q_data_type,
                torch_out_dtype=out_dtype,
                page_size=page_size,
                num_heads=num_heads,
                seq_len_q=q_len,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                is_persistent=False,
                is_var_seq=True,
                is_var_split_kv=False,
            )
            size_args = (
                batch_size,
                q_len,
                num_heads,
                head_dim_ckv,
                implementation.get_num_sm(device),
            )
            try:
                splits, size = implementation._get_split_kv_and_workspace_size(
                    *size_args, max_seq_len=max_seq_len, num_kv_splits=num_kv_splits
                )
            except ValueError:
                if num_kv_splits is not None:
                    raise
                splits, size = implementation._get_split_kv_and_workspace_size(
                    *size_args, max_seq_len=max_seq_len, num_kv_splits=1
                )
            capacity = workspace_buffer.numel() * workspace_buffer.element_size()
            if num_kv_splits is None and size > capacity:
                splits, size = implementation._get_split_kv_and_workspace_size(
                    *size_args, max_seq_len=max_seq_len, num_kv_splits=1
                )
            if size > capacity:
                raise ValueError(
                    "workspace_buffer too small for the requested split tactic"
                )
        except (ImportError, ValueError) as error:
            raise _CuteDslKernelUnsupportedError(str(error)) from error
        if num_kv_splits is None:
            self._compile_options = dict(
                workspace_buffer=workspace_buffer,
                device=device,
                q_data_type=q_data_type,
                out_dtype=out_dtype,
                page_size=page_size,
                batch_size=batch_size,
                num_heads=num_heads,
                q_len=q_len,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                resolved_is_var_seq=resolved_is_var_seq,
                is_var_q=is_var_q,
                total_q=total_q,
                max_seq_len=max_seq_len,
                use_sinks=use_sinks,
                enable_pdl=enable_pdl,
            )
        # Compile outside the known-support refusal boundary. Unexpected
        # compiler and runtime failures must reach explicit/auto callers alike.
        compiled = implementation.prepare_cute_dsl_mla_decode(
            device=device,
            torch_dtype=q_data_type,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            max_seq_len=max_seq_len,
            num_kv_splits=splits,
        )
        return (
            implementation,
            compiled,
            implementation._as_cute_dsl_workspace_i8(workspace_buffer),
            size,
            splits,
        )

    def get_valid_tactics(self, inputs, profile):
        options = getattr(self, "_planned_run_options", {})
        try:
            _validate_scalar_bmm_scales(
                options.get("bmm1_scale"), options.get("bmm2_scale")
            )
            _validate_mtp_scales(options.get("bmm1_scale"), options.get("bmm2_scale"))
        except ValueError:
            return []
        tactics = super().get_valid_tactics(inputs, profile)
        if not tactics:
            return []
        state = self._execution_state
        widths = (state.kv_lora_rank, state.qk_rope_head_dim)
        query = _resolve_structural_mla_input(
            inputs[0], desired="packed", widths=widths, name="query"
        )
        kv_cache = _resolve_structural_mla_input(
            inputs[1], desired="packed", widths=widths, name="KV cache"
        )
        out, lse = inputs[2:4]
        tables, lengths = state.block_tables, state.seq_lens
        try:
            implementation = self._implementation()
            implementation._check_kv_tensor_indexing(kv_cache, "kv_cache")
            for name, tensor in (
                ("query", query),
                ("out", out),
                ("lse", lse),
                ("block_tables", tables),
                ("seq_lens", lengths),
            ):
                if tensor is not None:
                    implementation._check_tensor_indexing(tensor, name)
        except ValueError:
            return []
        return tactics

    def _run(self, *, bmm1_scale=None, bmm2_scale=None, **kwargs):
        _validate_scalar_bmm_scales(bmm1_scale, bmm2_scale)
        _validate_mtp_scales(bmm1_scale, bmm2_scale)
        return super()._run(bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, **kwargs)
