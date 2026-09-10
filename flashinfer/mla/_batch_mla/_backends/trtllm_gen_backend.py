"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations

import functools
import math
from typing import ClassVar, Optional, Union

import torch

from flashinfer.jit import gen_trtllm_gen_fmha_module, setup_cubin_loader
from flashinfer.utils import (
    _check_block_tables_shape,
    _get_trtllm_gen_multi_ctas_kv_counter_buffer,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
)

from .._contracts import _resolve_structural_mla_input
from .._planning import _MLAPlanArguments, _audit_plan_from_wrapper_arguments
from ._capabilities import (
    _BackendPlanUnsupportedError,
    MLAPlanCapabilities,
    plan_capability_rejection_reason,
)


_SUPPORTED_MLA_DIMENSIONS = frozenset({(512, 64), (256, 64)})


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


def _get_q_layout(qo_indptr: torch.Tensor) -> tuple[int, int, int, bool, int]:
    if qo_indptr.ndim != 1:
        raise _BackendPlanUnsupportedError(
            f"trtllm-gen backend expects cum_seq_lens_q.ndim == 1, got {qo_indptr.ndim}."
        )
    if qo_indptr.numel() < 2:
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects cum_seq_lens_q to contain at least two entries."
        )

    qo_indptr_host = qo_indptr.to(device="cpu", dtype=torch.int64)
    if int(qo_indptr_host[0].item()) != 0:
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects cum_seq_lens_q to start at zero."
        )
    q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
    if torch.any(q_lens < 0).item():
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects nondecreasing cum_seq_lens_q."
        )
    max_q_len = int(q_lens.max().item())
    if max_q_len <= 0:
        raise _BackendPlanUnsupportedError(
            f"trtllm-gen backend expects positive query length, got {max_q_len}."
        )
    q_len = int(q_lens[0].item())
    is_uniform = not torch.any(q_lens != q_len).item()
    batch_size = qo_indptr.numel() - 1
    total_q = int(qo_indptr_host[-1].item())
    return batch_size, total_q, max_q_len, is_uniform, q_len


def _check_trtllm_gen_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    block_tables: torch.Tensor,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
) -> torch.Tensor:
    if query.ndim not in (3, 4):
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    if kv_cache.ndim == 3:
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    if (kv_lora_rank, qk_rope_head_dim) not in _SUPPORTED_MLA_DIMENSIONS:
        raise ValueError(
            "Unsupported MLA dimensions for trtllm-gen backend, got "
            f"kv_lora_rank={kv_lora_rank} and qk_rope_head_dim={qk_rope_head_dim}."
        )

    if query.ndim == 4:
        batch_size = query.shape[0]
    elif batch_size is None or max_q_len is None:
        raise ValueError("batch_size and max_q_len are required when query.ndim == 3")
    qk_head_dim = query.shape[-1]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or kv_cache.shape[3] != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, "
            f"got {qk_head_dim} and {kv_cache.shape[3]}."
        )

    _check_block_tables_shape(block_tables, True)
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "Expected block_tables.shape[0] to match query batch size, got "
            f"{block_tables.shape[0]} and {batch_size}."
        )
    if block_tables.shape[-1] == 0:
        raise ValueError("Expected block_tables to have positive width.")
    if page_size <= 0:
        raise ValueError(f"Expected positive page_size, got {page_size}.")

    return kv_cache


def _trtllm_gen_mla_incompatibility_reason(
    num_heads_q: int, block_size: int
) -> Optional[str]:
    if 64 < num_heads_q < 128:
        return (
            "trtllm-gen MLA decode does not support 64 < num_heads_q < 128; "
            f"got num_heads_q={num_heads_q}."
        )
    if block_size not in (32, 64):
        return f"trtllm-gen requires block_size in (32, 64), got {block_size}"
    return None


def _validate_trtllm_gen_mla_native_capability(
    *,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
    sm_scale: float,
) -> None:
    if q_data_type != kv_data_type or q_data_type not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend requires matching bfloat16 or float8_e4m3fn "
            f"query and KV tensors, got q_data_type={q_data_type} and "
            f"kv_data_type={kv_data_type}."
        )
    if (head_dim_ckv, head_dim_kpe) not in _SUPPORTED_MLA_DIMENSIONS:
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects supported MLA dimensions "
            "(head_dim_ckv, head_dim_kpe) in "
            f"{sorted(_SUPPORTED_MLA_DIMENSIONS)}, got "
            f"({head_dim_ckv}, {head_dim_kpe})."
        )
    if type(sm_scale) is not float or not math.isfinite(sm_scale):
        raise TypeError(
            "trtllm-gen backend expects sm_scale to be a finite Python float, "
            f"got {sm_scale!r}."
        )


def _validate_trtllm_gen_mla_device_capability(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise _BackendPlanUnsupportedError(
            "trtllm-gen MLA requires SM100/SM103, got compute capability "
            f"SM{major}{minor}."
        )


def _normalize_trtllm_gen_mla_metadata(
    *,
    device: torch.device,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        block_tables.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous(),
        seq_lens.to(device=device, dtype=torch.int32, non_blocking=True).contiguous(),
        cum_seq_lens_q.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous(),
    )


def _ensure_scalar_bmm_scales(
    bmm1_scale: Optional[Union[float, torch.Tensor]],
    bmm2_scale: Optional[Union[float, torch.Tensor]],
) -> None:
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        raise ValueError(
            "BMM tensor scales are not supported by this planned-wrapper version."
        )


class _BatchMLAPagedAttentionTrtllmGenBackend:
    _plan_capability_error_type = _BackendPlanUnsupportedError
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="trtllm-gen",
        lse_modes=frozenset({"none", "base2"}),
        kv_layouts=frozenset({"combined"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar"}),
        supports_skip_softmax=True,
        supports_skip_softmax_with_lse=False,
        supports_enable_pdl=True,
        supports_sinks=True,
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "trtllm-gen"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionTrtllmGenBackend":
        cls._validate_plan_arguments_before_metadata(args)
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise _BackendPlanUnsupportedError(reason)
        dense = args.native_dense()
        backend = cls(args._float_workspace_buffer)
        backend.plan(
            cum_seq_lens_q=dense.cum_seq_lens_q,
            block_tables=dense.block_tables,
            seq_lens=dense.seq_lens,
            max_q_len=dense.max_q_len,
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            page_size=args.page_size,
            causal=args.causal,
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            use_profiler=args.use_profiler,
            lse_mode=args.lse_mode,
            enable_pdl=args.enable_pdl,
            use_sinks=args.use_sinks,
        )
        return backend

    @classmethod
    def _validate_plan_arguments_before_metadata(cls, args: _MLAPlanArguments) -> None:
        if args.output_dtype != torch.bfloat16:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires a bfloat16 output contract without o_scale."
            )
        if (
            not isinstance(args.page_size, int)
            or isinstance(args.page_size, bool)
            or args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {args.page_size!r}."
            )
        if args.page_size > 128 or 128 % args.page_size != 0:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen dense metadata requires page_size to divide 128, "
                f"got {args.page_size}."
            )
        if args.use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the trtllm-gen backend."
            )
        _validate_trtllm_gen_mla_device_capability(args._float_workspace_buffer.device)
        if args.enable_pdl is not None and not isinstance(args.enable_pdl, bool):
            raise TypeError(
                "trtllm-gen backend expects enable_pdl to be bool or None, got "
                f"{args.enable_pdl!r}."
            )
        if not isinstance(args.use_sinks, bool):
            raise TypeError(
                f"trtllm-gen backend expects use_sinks to be bool, got {args.use_sinks!r}."
            )
        if args.causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the trtllm-gen backend."
            )
        if reason := _trtllm_gen_mla_incompatibility_reason(
            args.num_heads, args.page_size
        ):
            raise _BackendPlanUnsupportedError(reason)
        _validate_trtllm_gen_mla_native_capability(
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            sm_scale=args.sm_scale,
        )

    def plan(
        self,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool,
        lse_mode: str,
        enable_pdl: Optional[bool] = None,
        use_sinks: bool = False,
    ) -> None:
        for name, tensor in (
            ("cum_seq_lens_q", cum_seq_lens_q),
            ("block_tables", block_tables),
            ("seq_lens", seq_lens),
        ):
            if tensor.dtype != torch.int32:
                raise _BackendPlanUnsupportedError(
                    f"trtllm-gen backend expects {name} to have dtype "
                    f"torch.int32, got {tensor.dtype}."
                )
        batch_size, total_q, actual_max_q_len, is_uniform, q_len = _get_q_layout(
            cum_seq_lens_q
        )
        if not is_uniform and lse_mode != "none":
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend does not support LSE with compact variable-Q."
            )
        if max_q_len < actual_max_q_len:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects max_q_len to be at least the "
                f"maximum query length {actual_max_q_len}, got {max_q_len}."
            )
        if seq_lens.ndim != 1:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects seq_lens to be rank-1, "
                f"got rank {seq_lens.ndim}."
            )
        if seq_lens.numel() != batch_size:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects seq_lens.shape[0] == batch_size, "
                f"got {seq_lens.numel()} and {batch_size}."
            )
        try:
            _check_block_tables_shape(block_tables, True)
        except ValueError as error:
            raise _BackendPlanUnsupportedError(str(error)) from error
        if block_tables.shape[0] != batch_size:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects block_tables batch dimension "
                f"{batch_size}, got {tuple(block_tables.shape)}."
            )
        if block_tables.shape[-1] == 0:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects block_tables width to be positive."
            )
        (
            self._block_tables,
            self._seq_lens,
            self._cum_seq_lens_q,
        ) = _normalize_trtllm_gen_mla_metadata(
            device=self.device,
            block_tables=block_tables,
            seq_lens=seq_lens,
            cum_seq_lens_q=cum_seq_lens_q,
        )

        self._batch_size = batch_size
        self._q_len = q_len
        self._has_ragged_query = not is_uniform
        self._use_sinks = use_sinks
        # Without query offsets, the native launcher uses this as the exact
        # per-request length rather than an upper bound.
        self._max_q_len = q_len if is_uniform else max_q_len
        self._total_q = total_q
        self._num_heads = num_heads
        self._kv_lora_rank = head_dim_ckv
        self._qk_rope_head_dim = head_dim_kpe
        self._page_size = page_size
        self._max_seq_len = int(block_tables.shape[-1] * page_size)
        self._bmm1_scale = float(sm_scale)
        self._bmm2_scale = 1.0
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._enable_pdl = (
            device_support_pdl(self.device) if enable_pdl is None else enable_pdl
        )
        self._sm_count = get_device_sm_count(self.device)
        self._module = get_trtllm_gen_fmha_module()
        self._multi_ctas_kv_counter_buffer = (
            _get_trtllm_gen_multi_ctas_kv_counter_buffer(
                batch_size, num_heads, self._sm_count, self.device
            )
        )

    def run_from_wrapper(
        self,
        *,
        query: object,
        kv_cache: object,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        ckv_scale_arr: Optional[torch.Tensor],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if profiler_buffer is not None:
            raise ValueError(
                "profiler_buffer is not supported with trtllm-gen backend."
            )
        if kv_len is not None:
            raise ValueError(
                "kv_len is not supported with trtllm-gen backend; KV lengths "
                "are captured from seq_lens at plan time."
            )
        if page_table is not None:
            raise ValueError(
                "page_table is not supported with trtllm-gen backend; "
                "block_tables are captured at plan time."
            )
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with trtllm-gen backend."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with trtllm-gen backend.")
        if ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / ckv_scale_arr / kpe_scale are not supported with "
                "trtllm-gen planned backend."
            )
        packed_query = _resolve_structural_mla_input(
            query,
            desired="packed",
            widths=(self._kv_lora_rank, self._qk_rope_head_dim),
            name="query",
        )
        packed_kv_cache = _resolve_structural_mla_input(
            kv_cache,
            desired="packed",
            widths=(self._kv_lora_rank, self._qk_rope_head_dim),
            name="KV cache",
        )
        return self.run(
            query=packed_query,
            kv_cache=packed_kv_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionTrtllmGenBackend.run() called before plan()."
            )
        _ensure_scalar_bmm_scales(bmm1_scale, bmm2_scale)
        if out is None:
            raise ValueError(
                "out must be provided for trtllm-gen planned backend runs."
            )
        if return_lse and lse is None:
            raise ValueError(
                "lse must be provided when return_lse=True for trtllm-gen planned backend runs."
            )
        if (sinks is not None) != self._use_sinks:
            expected = "with" if self._use_sinks else "without"
            raise ValueError(
                f"trtllm-gen backend was planned {expected} use_sinks=True; "
                "run-time sinks must match the planned declaration."
            )
        if sinks is not None:
            check_shape_dtype_device(
                sinks,
                (self._num_heads,),
                torch.float32,
                self.device,
                "sinks",
            )
        if skip_softmax_threshold_scale_factor is not None and (
            return_lse or lse is not None
        ):
            raise ValueError(
                "trtllm-gen backend does not support LSE when skip-softmax is enabled."
            )

        resolved_bmm1_scale = self._bmm1_scale if bmm1_scale is None else bmm1_scale
        resolved_bmm2_scale = self._bmm2_scale if bmm2_scale is None else bmm2_scale

        check_shape_dtype_device(
            query,
            (
                self._total_q,
                self._num_heads,
                self._kv_lora_rank + self._qk_rope_head_dim,
            ),
            self._q_data_type,
            self.device,
            "query",
        )
        check_shape_dtype_device(
            kv_cache,
            (
                kv_cache.shape[0],
                self._page_size,
                self._kv_lora_rank + self._qk_rope_head_dim,
            ),
            self._kv_data_type,
            self.device,
            "kv_cache",
        )

        check_shape_dtype_device(
            out,
            query.shape[:-1] + (self._kv_lora_rank,),
            torch.bfloat16,
            self.device,
            "out",
        )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for trtllm-gen backend.")

        if not self._has_ragged_query:
            query_for_shape = query.reshape(
                self._batch_size,
                self._q_len,
                self._num_heads,
                self._kv_lora_rank + self._qk_rope_head_dim,
            )
        else:
            query_for_shape = query
        kv_cache = _check_trtllm_gen_mla_shape(
            query_for_shape,
            kv_cache,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            page_size=self._page_size,
            block_tables=self._block_tables,
            batch_size=self._batch_size,
            max_q_len=self._max_q_len,
        )
        query_flat = query_for_shape if self._has_ragged_query else query
        out_view = (
            out
            if self._has_ragged_query
            else out.reshape(
                self._batch_size, self._q_len, self._num_heads, self._kv_lora_rank
            )
        )

        if lse is not None:
            check_shape_dtype_device(
                lse,
                (self._total_q, self._num_heads),
                torch.float32,
                self.device,
                "lse",
            )
        lse_stride_tokens = 0 if lse is None else lse.stride(0)
        lse_stride_heads = 0 if lse is None else lse.stride(1)

        self._launch_native(
            out=out_view,
            query=query_flat,
            kv_cache=kv_cache,
            bmm1_scale=resolved_bmm1_scale,
            bmm2_scale=resolved_bmm2_scale,
            sinks=sinks,
            cum_seq_lens_q=(self._cum_seq_lens_q if self._has_ragged_query else None),
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            lse=lse,
            lse_stride_tokens=lse_stride_tokens,
            lse_stride_heads=lse_stride_heads,
        )
        return (out, lse) if return_lse else out

    def _launch_native(
        self,
        *,
        out: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: Union[float, torch.Tensor],
        sinks: Optional[torch.Tensor],
        cum_seq_lens_q: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        lse: Optional[torch.Tensor],
        lse_stride_tokens: int,
        lse_stride_heads: int,
    ) -> None:
        self._module.trtllm_paged_attention_decode(
            out,
            None,
            query,
            kv_cache,
            kv_cache,
            self._float_workspace_buffer,
            self._multi_ctas_kv_counter_buffer,
            self._block_tables,
            self._seq_lens,
            self._max_q_len,
            self._max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,
            -1,
            0,
            self._batch_size,
            -1,
            0,
            self._sm_count,
            self._enable_pdl,
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size(),
            sinks,
            cum_seq_lens_q,
            None,
            None,
            skip_softmax_threshold_scale_factor,
            True,
            lse,
            lse_stride_tokens,
            lse_stride_heads,
            False,
            None,
            0,
            None,
        )
