"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations

import functools
import math
from typing import Callable, ClassVar, List, Literal, Optional, Union
import torch

from flashinfer.autotuner import TunableRunner
from flashinfer.jit import gen_trtllm_gen_fmha_module, setup_cubin_loader
from flashinfer.utils import (
    _check_block_tables_shape,
    _get_trtllm_gen_multi_ctas_kv_counter_buffer,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
    log2e,
)

from ..._utils import (
    _round_to_seq_len_bucket,
    _check_mla_query_kv_shape,
    _check_mla_dense_page_table_shape,
)
from .._contracts import _resolve_structural_mla_input
from .._planning import _MLAPlanArguments, _audit_plan_from_wrapper_arguments
from ._capabilities import (
    _BackendPlanUnsupportedError,
    MLAPlanCapabilities,
    plan_capability_rejection_reason,
)

_SUPPORTED_MLA_DIMENSIONS = frozenset({(512, 64), (256, 64)})


@functools.lru_cache(maxsize=1)
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    for library_path in mod.get_library_paths():
        setup_cubin_loader(library_path)
    return op


def _prepare_trtllm_gen_mla_output(
    out: Optional[torch.Tensor],
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Allocate functional output or validate caller-owned BF16 storage.

    Adapters retain their own contiguity and required-output contracts.
    """
    if out is None:
        return torch.empty(expected_shape, dtype=torch.bfloat16, device=device)
    check_shape_dtype_device(out, expected_shape, torch.bfloat16, device, "out")
    return out


def _lower_bmm1_scale(scale):
    # Scalar conversion is performed by the native launcher. Device values must
    # be converted in the execution graph, never read back or frozen at plan time.
    return scale * log2e


class _BatchMLAPagedAttentionTrtllmGenBackend(TunableRunner):
    """TRTLLM-GEN execution shared by planned and functional MLA APIs.

    ``plan_from_wrapper`` prepares persistent metadata and counters. Wrapper
    calls use ``run_from_wrapper``; planned autotuning passes
    ``[query, kv_cache, out, lse, sinks]`` to ``forward``, optionally followed by
    the current tensor BMM scales. It uses the prepared metadata and scalar run
    options supplied through ``configure_tuning`` while profiling, or the
    current call's ``run_options`` when executing a retained selection.

    ``from_functional`` binds the request's normalized KV cache and execution
    options without wrapper planning. Functional calls pass
    ``[query, block_tables, seq_lens, out]`` to ``forward``, optionally followed
    by sparse lengths. Ragged calls also pass query offsets and a validated
    maximum query length directly, bypassing the autotuner.

    Both factories initialize common workspace state and set the instance mode.
    Shared execution and cache-key methods branch on that mode, preserving the
    distinct input contracts and tuning identities.

    Before functional construction, the dispatcher normalizes KV cache through
    ``_check_mla_functional_shape`` and computes ``sm_count``. Both adapters use
    shared execution configuration and scale lowering without repeating wrapper
    metadata preparation.
    """

    # Capabilities and common state

    _is_planned: bool
    kv_cache: torch.Tensor
    _qk_nope_head_dim: int
    sinks: Optional[List[torch.Tensor]]
    skip_softmax_threshold_scale_factor: Optional[float]
    _is_var_seq: bool
    return_lse: bool
    lse: Optional[torch.Tensor]
    return_lse_base: Optional[Literal["basee", "base2"]]

    _plan_capability_error_type = _BackendPlanUnsupportedError
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="trtllm-gen",
        lse_modes=frozenset({"none", "base2", "basee"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar", "bmm-tensor"}),
        supports_skip_softmax=True,
        supports_skip_softmax_with_lse=False,
        supports_enable_pdl=True,
        supports_sinks=True,
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    def __init__(self, workspace_buffer: torch.Tensor) -> None:
        self._backend = "trtllm-gen"
        self._workspace_buffer = workspace_buffer
        self.device = workspace_buffer.device
        # Functional profiling allocates counters lazily; wrapper planning
        # prepares them up front. Both paths retain them for ordered launches.
        self._multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None

    def _initialize_execution(
        self,
        *,
        native_run: Callable,
        sm_count: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_seq_len: int,
        bmm1_scale,
        bmm2_scale,
        enable_pdl: bool,
        sparse_mla_top_k: int = 0,
        uses_shared_paged_kv_idx: bool = True,
        use_fp16_softmax: Optional[bool] = None,
    ) -> None:
        """Bind native configuration after adapter-specific lowering.

        This never inspects or copies query metadata. The wrapper retains its
        prepared tensors; functional profiling supplies metadata on each call.
        """
        self._native_run = native_run
        self._sm_count = sm_count
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._page_size = page_size
        self._max_seq_len = max_seq_len
        self._bmm1_scale = bmm1_scale
        self._bmm2_scale = bmm2_scale
        self._enable_pdl = enable_pdl
        self._sparse_mla_top_k = sparse_mla_top_k
        self._uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        self._use_fp16_softmax = use_fp16_softmax

    # Wrapper planning and execution

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionTrtllmGenBackend":
        cls._validate_plan_arguments_before_metadata(args)
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise _BackendPlanUnsupportedError(reason)
        args.require_cuda_graph_dense_metadata("trtllm-gen")
        dense = args.native_dense()
        backend = cls(args._float_workspace_buffer)
        backend._is_planned = True
        backend._plan(
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
            lse_mode=args.lse_mode,
            enable_pdl=args.enable_pdl,
            skip_softmax=args.skip_softmax,
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
        major, minor = get_compute_capability(args._float_workspace_buffer.device)
        if (major, minor) not in ((10, 0), (10, 3), (10, 7)):
            raise _BackendPlanUnsupportedError(
                "trtllm-gen MLA requires SM100/SM103/SM107, got compute capability "
                f"SM{major}{minor}."
            )
        if args.enable_pdl is not None and not isinstance(args.enable_pdl, bool):
            raise TypeError(
                "trtllm-gen backend expects enable_pdl to be bool or None, got "
                f"{args.enable_pdl!r}."
            )
        if not isinstance(args.use_sinks, bool):
            raise TypeError(
                f"trtllm-gen backend expects use_sinks to be bool, got {args.use_sinks!r}."
            )
        if reason := _trtllm_gen_mla_incompatibility_reason(
            args.num_heads, args.page_size
        ):
            raise _BackendPlanUnsupportedError(reason)
        if args.q_data_type != args.kv_data_type or args.q_data_type not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend requires matching bfloat16 or float8_e4m3fn "
                f"query and KV tensors, got q_data_type={args.q_data_type} and "
                f"kv_data_type={args.kv_data_type}."
            )
        if args.scale_mode == "bmm-tensor" and args.q_data_type != torch.float8_e4m3fn:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen wrapper tensor BMM scales require float8_e4m3fn query and KV tensors."
            )
        if (args.head_dim_ckv, args.head_dim_kpe) not in _SUPPORTED_MLA_DIMENSIONS:
            raise _BackendPlanUnsupportedError(
                "trtllm-gen backend expects supported MLA dimensions "
                "(head_dim_ckv, head_dim_kpe) in "
                f"{sorted(_SUPPORTED_MLA_DIMENSIONS)}, got "
                f"({args.head_dim_ckv}, {args.head_dim_kpe})."
            )
        if type(args.sm_scale) is not float or not math.isfinite(args.sm_scale):
            raise TypeError(
                "trtllm-gen backend expects sm_scale to be a finite Python float, "
                f"got {args.sm_scale!r}."
            )

    def _plan(
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
        lse_mode: str,
        enable_pdl: Optional[bool] = None,
        skip_softmax: bool = False,
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
        if reason := _trtllm_gen_mla_incompatibility_reason(
            num_heads,
            page_size,
            has_var_q=not is_uniform,
            lse_requested=lse_mode != "none",
            causal=causal,
            max_q_len=actual_max_q_len,
        ):
            raise _BackendPlanUnsupportedError(reason)
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
        # Match the initial native launch selection exactly. Uniform Q uses its
        # actual length; ragged Q and KV use the declared dense capacities.
        native_max_q_len = q_len if is_uniform else max_q_len
        native_max_kv_len = int(block_tables.shape[-1] * page_size)
        sm_count = get_device_sm_count(self.device)
        module = get_trtllm_gen_fmha_module()
        # A skip-softmax plan allows both zero and nonzero runtime thresholds,
        # which select different native metadata. Preserve its existing behavior
        # until that run-time variant is known. Limit the new planning check to
        # SM100; other architectures retain their existing planning behavior.
        if not skip_softmax and get_compute_capability(self.device) == (10, 0):
            heads_per_cta = module._mla_plan_head_divisor(
                self._workspace_buffer,
                q_data_type == torch.float8_e4m3fn,
                batch_size,
                native_max_q_len,
                native_max_kv_len,
                num_heads,
                head_dim_ckv + head_dim_kpe,
                head_dim_ckv,
                page_size,
                sm_count,
            )
            if num_heads % heads_per_cta:
                raise _BackendPlanUnsupportedError(
                    f"trtllm-gen requires query heads ({num_heads}) to be divisible "
                    f"by the initial native kernel's heads per CTA ({heads_per_cta})."
                )
        self._block_tables = block_tables.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        self._seq_lens = seq_lens.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        self._cum_seq_lens_q = cum_seq_lens_q.to(
            device=self.device, dtype=torch.int32, non_blocking=True
        ).contiguous()

        self._batch_size = batch_size
        self._q_len = q_len
        self._has_ragged_query = not is_uniform
        # Without query offsets, the native launcher uses this as the exact
        # per-request length rather than an upper bound.
        self._max_q_len = native_max_q_len
        self._total_q = total_q
        self._num_heads = num_heads
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._initialize_execution(
            native_run=module.trtllm_paged_attention_decode,
            sm_count=sm_count,
            kv_lora_rank=head_dim_ckv,
            qk_rope_head_dim=head_dim_kpe,
            page_size=page_size,
            max_seq_len=native_max_kv_len,
            bmm1_scale=float(sm_scale),
            bmm2_scale=1.0,
            enable_pdl=device_support_pdl(self.device)
            if enable_pdl is None
            else enable_pdl,
        )
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
        # Plan capabilities and the wrapper's run contract enforce scale modes,
        # sink presence, and LSE/skip-softmax compatibility before backend entry.
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
        return self._run(
            query=packed_query,
            kv_cache=packed_kv_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            return_lse_base_on_e=return_lse_base_on_e,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def _run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        return_lse_base_on_e: bool = False,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self, "_native_run"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionTrtllmGenBackend._run() called before _plan()."
            )
        if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
            for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
                if isinstance(scale, torch.Tensor):
                    if (
                        scale.dtype != torch.float32
                        or scale.device != self.device
                        or scale.numel() != 1
                    ):
                        raise ValueError(
                            f"{name} must be a single-element float32 tensor on {self.device}."
                        )
        if out is None:
            raise ValueError(
                "out must be provided for trtllm-gen planned backend runs."
            )
        if return_lse and lse is None:
            raise ValueError(
                "lse must be provided when return_lse=True for trtllm-gen planned backend runs."
            )
        if sinks is not None:
            check_shape_dtype_device(
                sinks,
                (self._num_heads,),
                torch.float32,
                self.device,
                "sinks",
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

        out = _prepare_trtllm_gen_mla_output(
            out, query.shape[:-1] + (self._kv_lora_rank,), self.device
        )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for trtllm-gen backend.")

        kv_cache = _check_trtllm_gen_mla_shape(
            query,
            kv_cache,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            page_size=self._page_size,
            block_tables=self._block_tables,
            batch_size=self._batch_size,
            max_q_len=self._max_q_len,
        )
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
        self._execute(
            query=query,
            kv_cache=kv_cache,
            out=out_view,
            block_tables=self._block_tables,
            seq_lens=self._seq_lens,
            batch_size=self._batch_size,
            max_q_len=self._max_q_len,
            counter_buffer=self._multi_ctas_kv_counter_buffer,
            bmm1_scale=(
                _lower_bmm1_scale(resolved_bmm1_scale)
                if isinstance(resolved_bmm1_scale, torch.Tensor)
                else resolved_bmm1_scale
            ),
            bmm2_scale=resolved_bmm2_scale,
            sinks=sinks,
            cum_seq_lens_q=self._cum_seq_lens_q if self._has_ragged_query else None,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            lse=lse,
            lse_base_e=return_lse_base_on_e,
        )
        return (out, lse) if return_lse else out

    # Functional construction

    @classmethod
    def from_functional(
        cls,
        *,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sm_count: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        max_seq_len: int,
        sparse_mla_top_k: int,
        bmm1_scale,
        bmm2_scale,
        sinks: Optional[List[torch.Tensor]],
        skip_softmax_threshold_scale_factor: Optional[float],
        enable_pdl: bool,
        is_var_seq: bool,
        uses_shared_paged_kv_idx: bool,
        return_lse: bool,
        lse: Optional[torch.Tensor],
        return_lse_base: Optional[Literal["basee", "base2"]] = None,
        use_fp16_softmax: Optional[bool] = None,
    ) -> "_BatchMLAPagedAttentionTrtllmGenBackend":
        """Bind call-local configuration without preparing a wrapper plan."""
        backend = cls(workspace_buffer)
        if isinstance(bmm1_scale, torch.Tensor):
            bmm1_scale = _lower_bmm1_scale(bmm1_scale)
        backend._is_planned = False
        backend._initialize_execution(
            native_run=get_trtllm_gen_fmha_module().trtllm_paged_attention_decode,
            sm_count=sm_count,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=kv_cache.shape[-2],
            max_seq_len=max_seq_len,
            sparse_mla_top_k=sparse_mla_top_k,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            enable_pdl=enable_pdl,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            use_fp16_softmax=use_fp16_softmax,
        )
        backend.kv_cache = kv_cache
        backend._qk_nope_head_dim = qk_nope_head_dim
        backend.sinks = sinks
        backend.skip_softmax_threshold_scale_factor = (
            skip_softmax_threshold_scale_factor
        )
        backend._is_var_seq = is_var_seq
        backend.return_lse = return_lse
        backend.lse = lse
        backend.return_lse_base = return_lse_base
        return backend

    # Shared callable execution
    # Used by the autotuner and directly by functional ragged calls via
    # TunableRunner.__call__, which bypass tuning and pass query offsets here.

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        run_options=None,
        **kwargs,
    ):
        if self._is_planned:
            if tactic != -1:
                raise ValueError(f"Unsupported trtllm-gen MLA tactic: {tactic!r}.")
            query, kv_cache, out, lse, sinks = inputs[:5]
            options = self._planned_run_options if run_options is None else run_options
            if len(inputs) == 7:
                options = dict(options, bmm1_scale=inputs[5], bmm2_scale=inputs[6])
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
                **options,
            )

        query, block_tables, seq_lens, out = inputs[:4]
        sparse_mla_top_k_lens = inputs[4] if len(inputs) == 5 else None
        if cum_seq_lens_q is None:
            batch_size = query.size(0)
            max_q_len = query.size(1)
            num_qo_heads = query.size(2)
            query_flat = query.flatten(0, 1)
        else:
            # Core validates offsets and resolves the maximum before dispatch.
            # Keep ragged metadata call-local; uniform tuning profiles must not
            # inherit offsets or lengths from a previous real request.
            if max_q_len is None:
                raise ValueError(
                    "max_q_len is required for ragged TRTLLM-GEN execution"
                )
            if self.return_lse:
                raise NotImplementedError(
                    "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
                )
            batch_size = cum_seq_lens_q.numel() - 1
            num_qo_heads = query.size(1)
            query_flat = query

        if self.return_lse:
            lse_shape = (batch_size * max_q_len, num_qo_heads)
            # Reuse caller's lse when its shape matches the current input
            # (final dispatcher call); otherwise allocate fresh for the
            # autotune profile loop (synthetic inputs at bucket batch dims).
            if self.lse is not None and tuple(self.lse.shape) == lse_shape:
                lse = self.lse
            else:
                lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
        else:
            lse = None

        counter_buffer = multi_ctas_kv_counter_buffer
        if counter_buffer is None:
            counter_buffer = self._multi_ctas_kv_counter_buffer
            required_counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch_size, num_qo_heads, self._sm_count
            )
            counter_buffer_bytes = (
                0
                if counter_buffer is None
                else counter_buffer.numel() * counter_buffer.element_size()
            )
            if counter_buffer is None or counter_buffer_bytes < required_counter_bytes:
                counter_buffer = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
                    batch_size,
                    num_qo_heads,
                    self._sm_count,
                    query.device,
                )
                self._multi_ctas_kv_counter_buffer = counter_buffer
        self._execute(
            query=query_flat,
            kv_cache=self.kv_cache,
            out=out,
            block_tables=block_tables,
            seq_lens=seq_lens,
            batch_size=batch_size,
            max_q_len=max_q_len,
            counter_buffer=counter_buffer,
            bmm1_scale=self._bmm1_scale,
            bmm2_scale=self._bmm2_scale,
            sinks=self.sinks,
            cum_seq_lens_q=cum_seq_lens_q,
            skip_softmax_threshold_scale_factor=self.skip_softmax_threshold_scale_factor,
            lse=lse,
            lse_base_e=self.return_lse_base == "basee",
            sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        )
        return out

    def _execute(
        self,
        *,
        query,
        kv_cache,
        out,
        block_tables,
        seq_lens,
        batch_size,
        max_q_len,
        counter_buffer,
        bmm1_scale,
        bmm2_scale,
        sinks,
        cum_seq_lens_q,
        skip_softmax_threshold_scale_factor,
        lse,
        lse_base_e,
        sparse_mla_top_k_lens=None,
    ) -> None:
        """Lower dynamic options and launch using the shared native configuration.

        Adapters lower device scales at their execution boundary: once per
        functional invocation or wrapper run. Profiling reuses the functional
        conversion; graph replay retains the device conversion operation.
        """
        self._native_run(
            out,
            None,  # out_scale_factor
            query,
            kv_cache,  # key_cache
            kv_cache,  # value_cache
            self._workspace_buffer,
            counter_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            self._max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            -1,  # window_left
            self._sparse_mla_top_k,
            self._sm_count,
            self._enable_pdl,
            self._workspace_buffer.numel() * self._workspace_buffer.element_size(),
            sinks,
            cum_seq_lens_q,
            None,  # key_block_scales
            None,  # value_block_scales
            skip_softmax_threshold_scale_factor,
            self._uses_shared_paged_kv_idx,
            lse,
            1.0 / log2e if lse_base_e else 1.0,
            0 if lse is None else lse.stride(0),
            0 if lse is None else lse.stride(1),
            False,  # enable_block_sparse_attention
            sparse_mla_top_k_lens,
            0,  # bf16q_fp8kv_transform_mode
            self._use_fp16_softmax,
        )

    # Autotuning support

    def configure_tuning(self, *, cache_key: tuple, run_options: dict) -> None:
        """Bind scalar options for profiling the already prepared workload.

        The selection policy owns workload identity and calls this outside the
        warmed execution path. Query, KV, output, LSE and sinks remain call-local
        inputs, so a cached selection never captures another request's tensors.
        """
        if any(isinstance(value, torch.Tensor) for value in run_options.values()):
            raise TypeError("Planned tuning options must not contain tensors.")
        self._planned_tuning_key = cache_key
        self._planned_run_options = dict(run_options)

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        if self._is_planned:
            out = _prepare_trtllm_gen_mla_output(
                inputs[2],
                (self._total_q, self._num_heads, self._kv_lora_rank),
                self.device,
            )
            if not out.is_contiguous():
                return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        if self._is_planned:
            return ("planned", self._planned_tuning_key)

        q, _, _, out = inputs[:4]
        sinks_key = (
            None
            if self.sinks is None
            else tuple((tuple(t.shape), t.dtype) for t in self.sinks)
        )
        return (
            q.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self._qk_nope_head_dim,
            self._kv_lora_rank,
            self._qk_rope_head_dim,
            self._page_size,
            _round_to_seq_len_bucket(self._max_seq_len),
            self._sparse_mla_top_k,
            self._is_var_seq,
            self._uses_shared_paged_kv_idx,
            self._enable_pdl,
            (
                "bmm1_tensor"
                if isinstance(self._bmm1_scale, torch.Tensor)
                else "bmm1_float"
            ),
            (
                "bmm2_tensor"
                if isinstance(self._bmm2_scale, torch.Tensor)
                else "bmm2_float"
            ),
            sinks_key,
            self.skip_softmax_threshold_scale_factor,
            self.return_lse,
            len(inputs) == 5,
        )

    def __hash__(self):
        # The default `TunableRunner.__hash__` walks `self.__dict__` and falls
        # back to `id(...)` for unhashable values; our kv_cache / workspace /
        # sinks attributes are tensors or lists whose `id()` differs per
        # dispatcher call, which would poison the autotune cache key. All
        # tactic-determining state is already captured by
        # `get_cache_key_extras`, so return a class-stable hash here.
        return hash(type(self))


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
    # The adapter retains its narrower planned dimensions and dense metadata
    # contract; structural checks are shared with functional MLA backends.
    if (kv_lora_rank, qk_rope_head_dim) not in _SUPPORTED_MLA_DIMENSIONS:
        raise ValueError(
            "Unsupported MLA dimensions for trtllm-gen backend, got "
            f"kv_lora_rank={kv_lora_rank} and qk_rope_head_dim={qk_rope_head_dim}."
        )
    kv_cache = _check_mla_query_kv_shape(
        query, kv_cache, kv_lora_rank, qk_rope_head_dim, batch_size, max_q_len
    )
    if query.ndim == 4:
        batch_size = query.shape[0]
    _check_mla_dense_page_table_shape(block_tables, batch_size, page_size, True, False)
    if block_tables.shape[-1] == 0:
        raise ValueError("Expected block_tables to have positive width.")
    if page_size <= 0:
        raise ValueError(f"Expected positive page_size, got {page_size}.")
    return kv_cache


def _trtllm_gen_mla_incompatibility_reason(
    num_heads_q: int,
    block_size: int,
    *,
    has_var_q: bool = False,
    lse_requested: bool = False,
    causal: bool = True,
    max_q_len: int = 1,
) -> Optional[str]:
    """Native restrictions shared by functional selection and wrapper planning.

    Uses host facts only: no executable acquisition or metadata preparation.
    Wrapper-only feature restrictions are checked by the wrapper adapter.
    """
    if not causal and max_q_len > 1:
        return "trtllm-gen supports multi-Q only with causal=True."
    if has_var_q and lse_requested:
        return "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
    if 64 < num_heads_q < 128:
        return (
            "trtllm-gen MLA decode does not support 64 < num_heads_q < 128; "
            f"got num_heads_q={num_heads_q}."
        )
    if block_size not in (32, 64):
        return f"trtllm-gen requires block_size in (32, 64), got {block_size}"
    return None
