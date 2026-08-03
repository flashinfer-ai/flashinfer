"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import functools
import math
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import is_in_profile_measurement
from flashinfer.jit import gen_trtllm_gen_fmha_module, setup_cubin_loader
from flashinfer.utils import (
    _check_block_tables_shape,
    _get_trtllm_gen_multi_ctas_kv_counter_buffer,
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
    log2e,
    next_positive_power_of_2,
)

from ._capabilities import (
    BACKEND_OPERATIONAL_PLAN_FIELDS,
    MLAPlanCapabilities,
    validate_plan_capabilities,
)
from .._planning import (
    _MLAPlanArguments,
)
from .._contracts import (
    _FunctionalBackendUnsupportedError,
    _FunctionalMLARequest,
    _FunctionalMLARunner,
    MLAKVCache,
    MLAQuery,
)


_SUPPORTED_MLA_DIMENSIONS = (
    (512, 64, 128),
    (256, 64, 64),
    (512, 0, 256),
)

# Keep the trtllm-gen autotune sweep bounded; actual counter storage is sized
# dynamically per profiled batch.
_TRTLLM_GEN_MLA_MAX_BATCH = 8192


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


def _get_q_layout(qo_indptr: torch.Tensor) -> tuple[int, int, int, bool, int]:
    if qo_indptr.ndim != 1:
        raise _BackendPlanUnsupportedError(
            f"trtllm-gen backend expects qo_indptr.ndim == 1, got {qo_indptr.ndim}."
        )
    if qo_indptr.numel() < 2:
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects qo_indptr to contain at least two entries."
        )

    qo_indptr_host = qo_indptr.to(device="cpu", dtype=torch.int64)
    q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
    if torch.any(q_lens < 0).item():
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects nondecreasing qo_indptr."
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
    sparse_mla_top_k: int = 0,
    uses_shared_paged_kv_idx: bool = True,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
) -> torch.Tensor:
    if query.ndim not in (3, 4):
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    if kv_cache.ndim == 3:
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    if (kv_lora_rank, qk_rope_head_dim) not in {
        (dims[0], dims[1]) for dims in _SUPPORTED_MLA_DIMENSIONS
    }:
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

    if sparse_mla_top_k > 0:
        expected_shape = (
            (query.size(0), sparse_mla_top_k)
            if query.ndim == 3
            else (batch_size, query.size(1), sparse_mla_top_k)
        )
        if tuple(block_tables.shape) != expected_shape:
            raise ValueError(
                f"Expected block_tables.shape == {expected_shape}, got "
                f"{tuple(block_tables.shape)}"
            )
        return kv_cache

    _check_block_tables_shape(block_tables, uses_shared_paged_kv_idx)
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "Expected block_tables.shape[0] to match query batch size, got "
            f"{block_tables.shape[0]} and {batch_size}."
        )
    if block_tables.shape[-1] == 0:
        raise ValueError("Expected block_tables to have positive width.")

    return kv_cache


class _TrtllmGenMlaDecodeLauncher:
    """Own the dense TRTLLM-GEN launcher contract and counter lifecycle."""

    def __init__(self, run_func) -> None:
        self._run_func = run_func
        self._multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None
        self._counter_is_preplanned = False

    def reserve_counter_buffer(
        self,
        *,
        batch_size: int,
        num_qo_heads: int,
        sm_count: int,
        device: torch.device,
    ) -> None:
        """Ensure counter storage exists before a stateful execution phase."""
        required_counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_qo_heads, sm_count
        )
        counter_buffer = self._multi_ctas_kv_counter_buffer
        if counter_buffer is None or counter_buffer.numel() < required_counter_bytes:
            self._multi_ctas_kv_counter_buffer = (
                _get_trtllm_gen_multi_ctas_kv_counter_buffer(
                    batch_size, num_qo_heads, sm_count, device
                )
            )

    def run(
        self,
        *,
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_seq_len: int,
        bmm1_scale,
        bmm2_scale,
        batch_size: int,
        num_qo_heads: int,
        sparse_mla_top_k: int,
        sm_count: int,
        enable_pdl: Optional[bool],
        sinks=None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        uses_shared_paged_kv_idx: bool = True,
        lse: Optional[torch.Tensor] = None,
        lse_stride_tokens: int = 0,
        lse_stride_heads: int = 0,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
        sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    ) -> None:
        # Functional one-shot callers may not have a separate planning phase,
        # so retain lazy reservation as a compatibility fallback. Stateful
        # backends reserve this buffer during plan(), making their first run
        # safe for immediate CUDA graph capture.
        if multi_ctas_kv_counter_buffer is None and not self._counter_is_preplanned:
            self.reserve_counter_buffer(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                sm_count=sm_count,
                device=query.device,
            )
        counter_buffer = (
            multi_ctas_kv_counter_buffer
            if multi_ctas_kv_counter_buffer is not None
            else self._multi_ctas_kv_counter_buffer
        )
        assert counter_buffer is not None

        self._run_func(
            out,
            None,  # fp4 output (unsupported by MLA wrappers)
            query,
            key_cache,
            value_cache,
            workspace_buffer,
            counter_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            -1,  # window_left
            sparse_mla_top_k,
            sm_count,
            enable_pdl,
            workspace_buffer.numel() * workspace_buffer.element_size(),
            sinks,
            cum_seq_lens_q,
            None,  # key_block_scales
            None,  # value_block_scales
            skip_softmax_threshold_scale_factor,
            uses_shared_paged_kv_idx,
            lse,
            lse_stride_tokens,
            lse_stride_heads,
            False,  # enable_block_sparse_attention
            sparse_mla_top_k_lens,
        )


def _validate_trtllm_gen_scales(
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    device: torch.device,
) -> None:
    bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
    bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
    if bmm1_is_tensor != bmm2_is_tensor:
        raise ValueError(
            "bmm1_scale and bmm2_scale must be supplied together as a tensor pair."
        )
    for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
        if isinstance(scale, torch.Tensor):
            if scale.dtype != torch.float32:
                raise TypeError(f"{name} tensor must have dtype torch.float32")
            if scale.device != device:
                raise ValueError(
                    f"{name} tensor must be on device {device}, got {scale.device}."
                )
            if not scale.is_contiguous():
                raise ValueError(f"{name} tensor must be contiguous")
            if scale.numel() != 1:
                raise ValueError(f"{name} tensor must contain exactly one element")


def _trtllm_gen_mla_incompatibility_reason(kv_cache: torch.Tensor) -> Optional[str]:
    """Return the backend-local capability rejection used by auto selection."""
    block_size = kv_cache.size(-2)
    if block_size not in (32, 64):
        return f"trtllm-gen requires block_size in (32, 64), got {block_size}"
    return None


def _make_trtllm_gen_mla_launcher() -> _TrtllmGenMlaDecodeLauncher:
    """Create the native launcher without involving either public lifecycle."""
    return _TrtllmGenMlaDecodeLauncher(
        get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
    )


def _validate_trtllm_gen_mla_native_capability(
    *,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
    qk_nope_head_dim: int,
    sm_scale: float,
    allow_fp8: bool,
) -> None:
    """Validate constraints shared by planned and functional native launches."""
    supported_dtypes = (torch.bfloat16, torch.float8_e4m3fn)
    if (
        q_data_type != kv_data_type
        or q_data_type not in supported_dtypes
        or (not allow_fp8 and q_data_type != torch.bfloat16)
    ):
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend requires matching bfloat16 query and KV "
            "tensors (functional lowering additionally supports float8), got "
            f"{q_data_type=} and {kv_data_type=}."
        )
    supported = {
        (kv_lora_rank, qk_rope_head_dim, qk_nope)
        for kv_lora_rank, qk_rope_head_dim, qk_nope in _SUPPORTED_MLA_DIMENSIONS
    }
    if (head_dim_ckv, head_dim_kpe, qk_nope_head_dim) not in supported:
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects supported MLA dimensions "
            "(head_dim_ckv, head_dim_kpe, qk_nope_head_dim) in "
            f"{sorted(supported)}, got "
            f"({head_dim_ckv}, {head_dim_kpe}, {qk_nope_head_dim})."
        )
    expected_sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + head_dim_kpe)
    if not math.isclose(sm_scale, expected_sm_scale, rel_tol=1e-5, abs_tol=1e-8):
        raise _BackendPlanUnsupportedError(
            "trtllm-gen backend expects sm_scale to equal "
            f"1 / sqrt(qk_nope_head_dim + head_dim_kpe) = {expected_sm_scale}, "
            f"got {sm_scale}."
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
    """Copy native launch metadata to stable contiguous int32 storage."""
    return (
        block_tables.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous(),
        seq_lens.to(device=device, dtype=torch.int32, non_blocking=True).contiguous(),
        cum_seq_lens_q.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous(),
    )


class _BatchMLAPagedAttentionTrtllmGenBackend:
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="trtllm-gen",
        lse_modes=frozenset({"none", "base2"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar", "bmm-tensor"}),
        supports_skip_softmax=True,
        supports_enable_pdl=True,
        supports_is_var_seq=True,
        supports_sinks=True,
        supports_qk_nope_head_dim=True,
    )
    _backend_operational_plan_fields = BACKEND_OPERATIONAL_PLAN_FIELDS

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "trtllm-gen"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionTrtllmGenBackend":
        validate_plan_capabilities(args, cls._plan_capabilities)
        output_dtype = args.output_dtype
        if output_dtype != torch.bfloat16:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires a bfloat16 output contract without o_scale."
            )
        enable_pdl = args.enable_pdl
        is_var_seq = args.is_var_seq
        use_sinks = args.use_sinks
        qk_nope_head_dim = args.qk_nope_head_dim
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
        dense = args.native_dense
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
            qk_nope_head_dim=qk_nope_head_dim,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            use_sinks=use_sinks,
        )
        return backend

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
        qk_nope_head_dim: Optional[int],
        max_seq_len: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        use_sinks: bool = False,
        sparse_mla_top_k: int = 0,
        uses_shared_paged_kv_idx: bool = True,
        has_ragged_query: bool = False,
        allow_fp8: bool = False,
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
        if use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the trtllm-gen backend."
            )
        _validate_trtllm_gen_mla_device_capability(self.device)
        if enable_pdl is not None and not isinstance(enable_pdl, bool):
            raise _BackendPlanUnsupportedError(
                f"trtllm-gen backend expects enable_pdl to be bool or None, got {enable_pdl!r}."
            )
        if is_var_seq is not None and not isinstance(is_var_seq, bool):
            raise _BackendPlanUnsupportedError(
                f"trtllm-gen backend expects is_var_seq to be bool or None, got {is_var_seq!r}."
            )
        if not isinstance(use_sinks, bool):
            raise _BackendPlanUnsupportedError(
                f"trtllm-gen backend expects use_sinks to be bool, got {use_sinks!r}."
            )
        if causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the trtllm-gen backend."
            )
        if qk_nope_head_dim is None:
            raise _BackendPlanUnsupportedError(
                "qk_nope_head_dim is required with trtllm-gen backend."
            )
        _validate_trtllm_gen_mla_native_capability(
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            qk_nope_head_dim=qk_nope_head_dim,
            sm_scale=sm_scale,
            allow_fp8=allow_fp8,
        )

        batch_size, total_q, actual_max_q_len, is_uniform, q_len = _get_q_layout(
            cum_seq_lens_q
        )
        if max_q_len < actual_max_q_len:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects max_q_len to be at least the "
                f"maximum query length {actual_max_q_len}, got {max_q_len}."
            )
        if not is_uniform and not has_ragged_query:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen MLA wrapper requires uniform query lengths."
            )
        resolved_is_var_seq = not is_uniform if is_var_seq is None else is_var_seq
        if not resolved_is_var_seq and not is_uniform:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires is_var_seq=True for non-uniform "
                "cum_seq_lens_q metadata."
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
        if sparse_mla_top_k < 0:
            raise _BackendPlanUnsupportedError(
                f"trtllm-gen backend expects non-negative sparse_mla_top_k, got {sparse_mla_top_k}."
            )
        if sparse_mla_top_k > 0:
            expected_shape = (
                (total_q, sparse_mla_top_k)
                if has_ragged_query
                else (batch_size, q_len, sparse_mla_top_k)
            )
            if tuple(block_tables.shape) != expected_shape:
                raise _BackendPlanUnsupportedError(
                    "trtllm-gen backend expects sparse block_tables shape "
                    f"{expected_shape}, got {tuple(block_tables.shape)}."
                )
        else:
            try:
                _check_block_tables_shape(block_tables, uses_shared_paged_kv_idx)
            except ValueError as error:
                raise _BackendPlanUnsupportedError(str(error)) from error
            if block_tables.shape[0] != batch_size:
                raise _BackendPlanUnsupportedError(
                    "trtllm-gen backend expects block_tables batch dimension "
                    f"{batch_size}, got {tuple(block_tables.shape)}."
                )
        if page_size not in (32, 64):
            raise _BackendPlanUnsupportedError(
                f"trtllm-gen backend requires page_size in (32, 64), got {page_size}."
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
        self._is_var_seq = resolved_is_var_seq
        self._has_ragged_query = has_ragged_query
        self._use_sinks = use_sinks
        self._max_q_len = max_q_len
        self._total_q = total_q
        self._num_heads = num_heads
        self._kv_lora_rank = head_dim_ckv
        self._qk_rope_head_dim = head_dim_kpe
        self._page_size = page_size
        # seq_lens is graph-mutable launch metadata.  Keep the scalar launch
        # bound at the page-table capacity so in-place length growth remains
        # valid without changing captured arguments.
        self._max_seq_len = (
            int(block_tables.shape[-1] * page_size)
            if max_seq_len is None
            else max_seq_len
        )
        self._bmm1_scale = float(sm_scale)
        self._bmm2_scale = 1.0
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._sparse_mla_top_k = sparse_mla_top_k
        self._uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        self._enable_pdl = (
            device_support_pdl(self.device) if enable_pdl is None else enable_pdl
        )
        self._sm_count = get_device_sm_count(self.device)
        launcher = _make_trtllm_gen_mla_launcher()
        launcher.reserve_counter_buffer(
            batch_size=batch_size,
            num_qo_heads=num_heads,
            sm_count=self._sm_count,
            device=self.device,
        )
        launcher._counter_is_preplanned = True
        self._launcher = launcher

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
        packed_query = query.require_packed()
        kv_cache = kv.require_packed_view()
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
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
            )
        return self.run(
            query=packed_query,
            kv_cache=kv_cache,
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
        bmm1_scale: Optional[float | torch.Tensor] = None,
        bmm2_scale: Optional[float | torch.Tensor] = None,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_launcher"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionTrtllmGenBackend.run() called before plan()."
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

        bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
        bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
        if bmm1_is_tensor != bmm2_is_tensor:
            raise ValueError(
                "bmm1_scale and bmm2_scale must be supplied together as a tensor pair."
            )
        for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
            if isinstance(scale, torch.Tensor):
                if scale.dtype != torch.float32:
                    raise TypeError(f"{name} tensor must have dtype torch.float32")
                if scale.device != self.device:
                    raise ValueError(
                        f"{name} tensor must be on device {self.device}, got {scale.device}."
                    )
                if not scale.is_contiguous():
                    raise ValueError(f"{name} tensor must be contiguous")
                if scale.numel() != 1:
                    raise ValueError(f"{name} tensor must contain exactly one element")
        resolved_bmm1_scale = self._bmm1_scale if bmm1_scale is None else bmm1_scale
        resolved_bmm2_scale = self._bmm2_scale if bmm2_scale is None else bmm2_scale
        if isinstance(resolved_bmm1_scale, torch.Tensor):
            resolved_bmm1_scale = resolved_bmm1_scale * log2e

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

        if out is None:
            out = torch.empty(
                query.shape[:-1] + (self._kv_lora_rank,),
                dtype=torch.bfloat16,
                device=self.device,
            )
        else:
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
            query = query.reshape(
                self._batch_size,
                self._q_len,
                self._num_heads,
                self._kv_lora_rank + self._qk_rope_head_dim,
            )
        kv_cache = _check_trtllm_gen_mla_shape(
            query,
            kv_cache,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            page_size=self._page_size,
            block_tables=self._block_tables,
            sparse_mla_top_k=self._sparse_mla_top_k,
            uses_shared_paged_kv_idx=self._uses_shared_paged_kv_idx,
            batch_size=self._batch_size,
            max_q_len=self._max_q_len,
        )
        query_flat = query if self._has_ragged_query else query.flatten(0, 1)
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
        elif return_lse:
            lse = torch.empty(
                (self._total_q, self._num_heads),
                dtype=torch.float32,
                device=self.device,
            )
        lse_stride_tokens = 0 if lse is None else lse.stride(0)
        lse_stride_heads = 0 if lse is None else lse.stride(1)
        if multi_ctas_kv_counter_buffer is not None:
            multi_ctas_kv_counter_buffer = (
                _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                    multi_ctas_kv_counter_buffer,
                    self._batch_size,
                    self._num_heads,
                    self._sm_count,
                    self.device,
                )
            )

        self._launcher.run(
            out=out_view,
            query=query_flat,
            key_cache=kv_cache,
            value_cache=kv_cache,
            workspace_buffer=self._float_workspace_buffer,
            block_tables=self._block_tables,
            seq_lens=self._seq_lens,
            max_q_len=self._max_q_len,
            max_seq_len=self._max_seq_len,
            bmm1_scale=resolved_bmm1_scale,
            bmm2_scale=resolved_bmm2_scale,
            batch_size=self._batch_size,
            num_qo_heads=self._num_heads,
            sparse_mla_top_k=self._sparse_mla_top_k,
            sm_count=self._sm_count,
            enable_pdl=self._enable_pdl,
            sinks=sinks,
            cum_seq_lens_q=(self._cum_seq_lens_q if self._has_ragged_query else None),
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            lse=lse,
            lse_stride_tokens=lse_stride_tokens,
            lse_stride_heads=lse_stride_heads,
            uses_shared_paged_kv_idx=self._uses_shared_paged_kv_idx,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
        return (out, lse) if return_lse else out


@dataclass
class _TrtllmGenMlaFunctionalState:
    """Native one-shot launch state, independent of the planned wrapper."""

    launcher: _TrtllmGenMlaDecodeLauncher
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    cum_seq_lens_q: torch.Tensor
    max_q_len: int
    out: torch.Tensor
    lse: Optional[torch.Tensor]


class TrtllmGenMlaDecodeRunner(_FunctionalMLARunner):
    """Direct functional TRTLLM-GEN runner with one native tactic."""

    name = "trtllm-gen"

    def __init__(self, request: _FunctionalMLARequest) -> None:
        _FunctionalMLARunner.__init__(self, request)
        normalized, user_lse = self._normalize_request(request)
        if normalized.enable_pdl is None:
            normalized = replace(
                normalized,
                enable_pdl=device_support_pdl(normalized.query.device),
            )
        self.request = normalized
        self.kv_cache = normalized.kv_cache
        self.workspace_buffer = normalized.workspace_buffer
        self.qk_nope_head_dim = normalized.qk_nope_head_dim
        self.kv_lora_rank = normalized.kv_lora_rank
        self.qk_rope_head_dim = normalized.qk_rope_head_dim
        self.page_size = normalized.kv_cache.shape[-2]
        self.max_seq_len = normalized.max_seq_len
        self.sparse_mla_top_k = normalized.sparse_mla_top_k
        self.bmm1_scale = normalized.bmm1_scale
        self.bmm2_scale = normalized.bmm2_scale
        self.sinks = normalized.sinks
        self.skip_softmax_threshold_scale_factor = (
            normalized.skip_softmax_threshold_scale_factor
        )
        self.enable_pdl = normalized.enable_pdl
        self.is_var_seq = normalized.is_var_seq
        self.uses_shared_paged_kv_idx = normalized.uses_shared_paged_kv_idx
        self.return_lse = normalized.return_lse
        self.lse = normalized.lse
        self.cum_seq_lens_q = normalized.cum_seq_lens_q
        self.max_q_len = normalized.max_q_len
        self._user_lse = user_lse
        self._inputs = [
            normalized.query,
            normalized.block_tables,
            normalized.seq_lens,
            normalized.out,
        ]
        if normalized.sparse_mla_top_k_lens is not None:
            self._inputs.append(normalized.sparse_mla_top_k_lens)
        self._prepared_functional_state: Optional[_TrtllmGenMlaFunctionalState] = None
        self._dispatch_functional_state: Optional[_TrtllmGenMlaFunctionalState] = None
        self._dispatch_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._dispatch_request: Optional[_FunctionalMLARequest] = None
        self._ragged_cache_key_extra = None
        if normalized.cum_seq_lens_q is not None:
            offsets = tuple(
                int(value)
                for value in normalized.cum_seq_lens_q.detach().cpu().tolist()
            )
            query_lengths = tuple(
                end - start
                for start, end in zip(offsets[:-1], offsets[1:], strict=True)
            )
            self._ragged_cache_key_extra = (
                "ragged_query",
                query_lengths,
                None if normalized.max_q_len is None else int(normalized.max_q_len),
            )

    @staticmethod
    def _normalize_request(
        request: _FunctionalMLARequest,
    ) -> tuple[_FunctionalMLARequest, Optional[torch.Tensor]]:
        if (
            request.sparse_mla_top_k > 0
            and request.skip_softmax_threshold_scale_factor is not None
        ):
            raise ValueError("skip_softmax is not supported for sparse MLA")
        reason = _trtllm_gen_mla_incompatibility_reason(request.kv_cache)
        if reason is not None:
            raise ValueError(reason)
        _validate_trtllm_gen_scales(
            request.bmm1_scale, request.bmm2_scale, request.query.device
        )
        if request.seq_lens is None:
            raise ValueError("seq_lens is required for trtllm-gen MLA")

        query = request.query
        cum_seq_lens_q = request.cum_seq_lens_q
        max_q_len = request.max_q_len
        has_ragged_query = cum_seq_lens_q is not None
        if has_ragged_query:
            if request.return_lse or request.lse is not None:
                raise NotImplementedError(
                    "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
                )
            if query.ndim != 3:
                raise ValueError(
                    "query must have shape [total_q, num_heads, head_dim_qk] "
                    "when cum_seq_lens_q is provided"
                )
            check_shape_dtype_device(
                cum_seq_lens_q, None, torch.int32, query.device, "cum_seq_lens_q"
            )
            if cum_seq_lens_q.ndim != 1:
                raise ValueError(
                    f"Expected cum_seq_lens_q.ndim == 1, got {cum_seq_lens_q.ndim}"
                )
            if cum_seq_lens_q.size(0) < 2:
                raise ValueError("cum_seq_lens_q must contain at least two entries")
            batch_size = cum_seq_lens_q.size(0) - 1
            if batch_size != request.seq_lens.size(0):
                raise ValueError(
                    "Batch size mismatch: cum_seq_lens_q describes "
                    f"{batch_size} sequences, but seq_lens has "
                    f"{request.seq_lens.size(0)} entries"
                )
            if max_q_len is None:
                cum_seq_lens_q_host = cum_seq_lens_q.cpu()
                if cum_seq_lens_q_host[0].item() != 0:
                    raise ValueError("cum_seq_lens_q must start with 0")
                if cum_seq_lens_q_host[-1].item() != query.size(0):
                    raise ValueError(
                        "cum_seq_lens_q[-1] must match the flattened query length"
                    )
                q_lens = cum_seq_lens_q_host[1:] - cum_seq_lens_q_host[:-1]
                if torch.any(q_lens < 0).item():
                    raise ValueError(
                        "cum_seq_lens_q must be monotonically non-decreasing"
                    )
                max_q_len = int(q_lens.max().item())
                if max_q_len <= 0:
                    raise ValueError(
                        "cum_seq_lens_q must describe at least one query token"
                    )
            elif max_q_len <= 0:
                raise ValueError("max_q_len must be greater than 0")
            batch_size_for_counter = batch_size
        else:
            if query.ndim != 4:
                raise ValueError(
                    "query must have shape [batch_size, q_len, num_heads, head_dim_qk]"
                )
            batch_size_for_counter = query.size(0)

        kv_cache = _check_trtllm_gen_mla_shape(
            query,
            request.kv_cache,
            kv_lora_rank=request.kv_lora_rank,
            qk_rope_head_dim=request.qk_rope_head_dim,
            page_size=request.kv_cache.size(-2),
            block_tables=request.block_tables,
            sparse_mla_top_k=request.sparse_mla_top_k,
            uses_shared_paged_kv_idx=request.uses_shared_paged_kv_idx,
            batch_size=batch_size_for_counter,
            max_q_len=max_q_len,
        )
        expected_out_shape = query.shape[:-1] + (request.kv_lora_rank,)
        out = request.out
        if out is None:
            out = torch.empty(
                expected_out_shape, dtype=torch.bfloat16, device=query.device
            )
        else:
            check_shape_dtype_device(
                out, expected_out_shape, torch.bfloat16, query.device, "out"
            )

        user_lse = request.lse
        lse = request.lse
        if request.return_lse:
            flat_lse_shape = (
                query.size(0) * query.size(1),
                query.size(2),
            )
            nested_lse_shape = (query.size(0), query.size(1), query.size(2))
            if lse is None:
                lse = torch.empty(
                    flat_lse_shape, dtype=torch.float32, device=query.device
                )
                user_lse = lse
            elif tuple(lse.shape) == flat_lse_shape:
                check_shape_dtype_device(
                    lse, flat_lse_shape, torch.float32, query.device, "lse"
                )
            elif tuple(lse.shape) == nested_lse_shape:
                check_shape_dtype_device(
                    lse, nested_lse_shape, torch.float32, query.device, "lse"
                )
                lse = lse.view(flat_lse_shape)
            else:
                raise ValueError(
                    f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                    f"got {tuple(lse.shape)}"
                )

        return (
            replace(
                request,
                kv_cache=kv_cache,
                out=out,
                lse=lse,
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=max_q_len,
            ),
            user_lse,
        )

    @property
    def inputs(self) -> list[torch.Tensor]:
        return self._inputs

    def __hash__(self) -> int:
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        del inputs, profile
        return [-1]

    def get_cache_key_extras(self, inputs):
        q, _, _, out = inputs[:4]
        sinks_key = (
            None
            if self.sinks is None
            else tuple((tuple(t.shape), t.dtype) for t in self.sinks)
        )
        base_key = (
            q.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            next_positive_power_of_2(self.max_seq_len),
            self.sparse_mla_top_k,
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            "bmm1_tensor"
            if isinstance(self.bmm1_scale, torch.Tensor)
            else "bmm1_float",
            "bmm2_tensor"
            if isinstance(self.bmm2_scale, torch.Tensor)
            else "bmm2_float",
            sinks_key,
            self.skip_softmax_threshold_scale_factor,
            self.return_lse,
            len(inputs) == 5,
        )
        if self._ragged_cache_key_extra is None:
            return base_key
        return base_key + (self._ragged_cache_key_extra,)

    def _prepare_functional_state(
        self, request: _FunctionalMLARequest
    ) -> _TrtllmGenMlaFunctionalState:
        query = request.query
        seq_lens = request.seq_lens
        assert seq_lens is not None
        if request.cum_seq_lens_q is None:
            batch_size = query.size(0)
            max_q_len = query.size(1)
            cum_seq_lens_q = torch.arange(
                batch_size + 1, device=query.device, dtype=torch.int32
            ).mul_(max_q_len)
            has_ragged_query = False
        else:
            cum_seq_lens_q = request.cum_seq_lens_q
            assert request.max_q_len is not None
            max_q_len = request.max_q_len
            has_ragged_query = True

        for name, tensor in (
            ("cum_seq_lens_q", cum_seq_lens_q),
            ("block_tables", request.block_tables),
            ("seq_lens", seq_lens),
        ):
            if tensor.dtype != torch.int32:
                raise _BackendPlanUnsupportedError(
                    f"trtllm-gen backend expects {name} to have dtype "
                    f"torch.int32, got {tensor.dtype}."
                )
        try:
            _validate_trtllm_gen_mla_device_capability(query.device)
        except _BackendPlanUnsupportedError as error:
            raise _FunctionalBackendUnsupportedError(str(error)) from error
        if request.enable_pdl is not None and not isinstance(request.enable_pdl, bool):
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects enable_pdl to be bool or None, got "
                f"{request.enable_pdl!r}."
            )
        if not isinstance(request.is_var_seq, bool):
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects is_var_seq to be bool or None, got "
                f"{request.is_var_seq!r}."
            )
        sinks = request.sinks
        if isinstance(sinks, (list, tuple)):
            if len(sinks) != 1:
                raise ValueError("trtllm-gen MLA expects sinks to contain one tensor")
            sinks = sinks[0]
        try:
            _validate_trtllm_gen_mla_native_capability(
                q_data_type=query.dtype,
                kv_data_type=request.kv_cache.dtype,
                head_dim_ckv=request.kv_lora_rank,
                head_dim_kpe=request.qk_rope_head_dim,
                qk_nope_head_dim=request.qk_nope_head_dim,
                sm_scale=1.0
                / math.sqrt(request.qk_nope_head_dim + request.qk_rope_head_dim),
                allow_fp8=True,
            )
        except _BackendPlanUnsupportedError as error:
            raise _FunctionalBackendUnsupportedError(str(error)) from error
        batch_size, total_q, actual_max_q_len, is_uniform, q_len = _get_q_layout(
            cum_seq_lens_q
        )
        if max_q_len < actual_max_q_len:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects max_q_len to be at least the "
                f"maximum query length {actual_max_q_len}, got {max_q_len}."
            )
        if not is_uniform and not has_ragged_query:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen MLA wrapper requires uniform query lengths."
            )
        if not request.is_var_seq and not is_uniform:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires is_var_seq=True for non-uniform "
                "cum_seq_lens_q metadata."
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
        if request.sparse_mla_top_k < 0:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects non-negative sparse_mla_top_k, got "
                f"{request.sparse_mla_top_k}."
            )
        if request.sparse_mla_top_k > 0:
            expected_shape = (
                (total_q, request.sparse_mla_top_k)
                if has_ragged_query
                else (batch_size, q_len, request.sparse_mla_top_k)
            )
            if tuple(request.block_tables.shape) != expected_shape:
                raise _BackendPlanUnsupportedError(
                    "trtllm-gen backend expects sparse block_tables shape "
                    f"{expected_shape}, got {tuple(request.block_tables.shape)}."
                )
        else:
            try:
                _check_block_tables_shape(
                    request.block_tables, request.uses_shared_paged_kv_idx
                )
            except ValueError as error:
                raise _BackendPlanUnsupportedError(str(error)) from error
            if request.block_tables.shape[0] != batch_size:
                raise _BackendPlanUnsupportedError(
                    "trtllm-gen backend expects block_tables batch dimension "
                    f"{batch_size}, got {tuple(request.block_tables.shape)}."
                )
        if self.page_size not in (32, 64):
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires page_size in (32, 64), got "
                f"{self.page_size}."
            )
        if request.block_tables.shape[-1] == 0:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects block_tables width to be positive."
            )
        block_tables, seq_lens, cum_seq_lens_q = _normalize_trtllm_gen_mla_metadata(
            device=query.device,
            block_tables=request.block_tables,
            seq_lens=seq_lens,
            cum_seq_lens_q=cum_seq_lens_q,
        )
        lse = request.lse if request.return_lse else None
        if request.return_lse:
            lse_shape = (total_q, query.size(-2))
            if lse is None or tuple(lse.shape) != lse_shape:
                lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
        launcher = _make_trtllm_gen_mla_launcher()
        sm_count = get_device_sm_count(query.device)
        launcher.reserve_counter_buffer(
            batch_size=batch_size,
            num_qo_heads=query.size(-2),
            sm_count=sm_count,
            device=query.device,
        )
        launcher._counter_is_preplanned = True
        assert request.out is not None
        return _TrtllmGenMlaFunctionalState(
            launcher=launcher,
            block_tables=block_tables,
            seq_lens=seq_lens,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            out=request.out,
            lse=lse,
        )

    def _prepare_for_dispatch(self) -> None:
        """Prepare the caller shape before admitting this automatic candidate."""
        self._dispatch_functional_state = self._prepare_functional_state(self.request)
        self._dispatch_inputs = tuple(self._inputs)
        self._dispatch_request = self.request

    def _launch_functional_state(
        self,
        *,
        request: _FunctionalMLARequest,
        state: _TrtllmGenMlaFunctionalState,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = request.query
        has_ragged_query = request.cum_seq_lens_q is not None
        batch_size = state.cum_seq_lens_q.numel() - 1
        sinks = request.sinks
        if isinstance(sinks, (list, tuple)):
            sinks = sinks[0]
        if sinks is not None:
            check_shape_dtype_device(
                sinks, (query.size(-2),), torch.float32, query.device, "sinks"
            )
        if (
            request.skip_softmax_threshold_scale_factor is not None
            and state.lse is not None
        ):
            raise ValueError(
                "trtllm-gen backend does not support LSE when skip-softmax is enabled."
            )
        bmm1_scale = request.bmm1_scale
        if isinstance(bmm1_scale, torch.Tensor):
            bmm1_scale = bmm1_scale * log2e
        counter = multi_ctas_kv_counter_buffer
        if counter is not None:
            counter = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                counter,
                batch_size,
                query.size(-2),
                get_device_sm_count(query.device),
                query.device,
            )
        state.launcher.run(
            out=state.out,
            query=query if has_ragged_query else query.flatten(0, 1),
            key_cache=request.kv_cache,
            value_cache=request.kv_cache,
            workspace_buffer=request.workspace_buffer,
            block_tables=state.block_tables,
            seq_lens=state.seq_lens,
            max_q_len=state.max_q_len,
            max_seq_len=request.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=request.bmm2_scale,
            batch_size=batch_size,
            num_qo_heads=query.size(-2),
            sparse_mla_top_k=request.sparse_mla_top_k,
            sm_count=get_device_sm_count(query.device),
            enable_pdl=(
                device_support_pdl(query.device)
                if request.enable_pdl is None
                else request.enable_pdl
            ),
            sinks=sinks,
            cum_seq_lens_q=(state.cum_seq_lens_q if has_ragged_query else None),
            skip_softmax_threshold_scale_factor=(
                request.skip_softmax_threshold_scale_factor
            ),
            uses_shared_paged_kv_idx=request.uses_shared_paged_kv_idx,
            lse=state.lse,
            lse_stride_tokens=0 if state.lse is None else state.lse.stride(0),
            lse_stride_heads=0 if state.lse is None else state.lse.stride(1),
            multi_ctas_kv_counter_buffer=counter,
            sparse_mla_top_k_lens=request.sparse_mla_top_k_lens,
        )
        if not request.return_lse:
            return state.out
        public_lse = (
            self._user_lse
            if request.query is self.request.query and self._user_lse is not None
            else state.lse
        )
        assert public_lse is not None
        return state.out, public_lse

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del kwargs
        if tactic != -1:
            raise ValueError(f"trtllm-gen MLA only supports tactic -1, got {tactic!r}.")
        if len(inputs) not in (4, 5):
            raise ValueError(
                "trtllm-gen MLA runner expects four or five dynamic inputs."
            )
        query, block_tables, seq_lens, out = inputs[:4]
        sparse_mla_top_k_lens = inputs[4] if len(inputs) == 5 else None
        profile_measurement = is_in_profile_measurement()
        use_dispatch_state = (
            not profile_measurement
            and not do_preparation
            and self._dispatch_functional_state is not None
            and self._dispatch_inputs is not None
            and all(
                actual is prepared
                for actual, prepared in zip(inputs, self._dispatch_inputs, strict=True)
            )
        )
        if use_dispatch_state:
            assert self._dispatch_request is not None
            request = self._dispatch_request
            state = self._dispatch_functional_state
        else:
            dynamic_request = replace(
                self.request,
                query=query,
                block_tables=block_tables,
                seq_lens=seq_lens,
                out=out,
                sparse_mla_top_k_lens=sparse_mla_top_k_lens,
            )
        if not use_dispatch_state and profile_measurement and not do_preparation:
            if self._prepared_functional_state is None:
                raise RuntimeError(
                    "TRTLLM-GEN autotuner launch was not prepared before profiling."
                )
            state = self._prepared_functional_state
            request = replace(dynamic_request, lse=state.lse)
        elif not use_dispatch_state:
            initial_shape = self.request.query.shape
            dynamic_lse = self.request.lse if query.shape == initial_shape else None
            request, _ = self._normalize_request(
                replace(dynamic_request, lse=dynamic_lse)
            )
        if not use_dispatch_state and do_preparation:
            self._prepared_functional_state = self._prepare_functional_state(request)
            state = self._prepared_functional_state
        elif not use_dispatch_state and not profile_measurement:
            state = self._prepare_functional_state(request)
        if (
            multi_ctas_kv_counter_buffer is None
            and not do_preparation
            and not profile_measurement
        ):
            multi_ctas_kv_counter_buffer = request.multi_ctas_kv_counter_buffer
        return self._launch_functional_state(
            request=request,
            state=state,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
