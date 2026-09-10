"""Shared mechanics for concrete planned CuTe DSL MLA backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, ClassVar, Optional, Union

import torch

from flashinfer.utils import check_shape_dtype_device, get_compute_capability

from .._contracts import _resolve_structural_mla_input
from .._planning import _MLAPlanArguments, _audit_plan_from_wrapper_arguments
from ._capabilities import (
    _BackendPlanUnsupportedError,
    MLAPlanCapabilities,
    plan_capability_rejection_reason,
)


class _CuteDslKernelUnsupportedError(RuntimeError):
    """Known native CuTe capability refusal, safe to convert to typed unsupported."""


@dataclass
class _CuteDslMlaExecutionState:
    compiled_kernel: Any
    workspace_bytes: Optional[torch.Tensor]
    split_kv: int
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    batch_size: int
    q_len: int
    num_heads: int
    total_q: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype: torch.dtype
    out_dtype: torch.dtype
    lse_scratch: torch.Tensor
    bmm1_scale: float
    bmm2_scale: float
    use_sinks: bool
    device: torch.device
    Float32: Any
    Int32: Any


def _validate_cute_dsl_plan_args_before_metadata(args: _MLAPlanArguments) -> None:
    if args.output_dtype != torch.bfloat16:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires a bfloat16 output contract without o_scale."
        )
    if args.use_profiler:
        raise _BackendPlanUnsupportedError(
            "use_profiler is not supported by the cute-dsl backend."
        )
    if args.causal:
        raise _BackendPlanUnsupportedError(
            "causal=True is not supported by the cute-dsl backend."
        )
    if args.enable_pdl is True:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend does not support enable_pdl=True."
        )
    if args.enable_pdl is not None and not isinstance(args.enable_pdl, bool):
        raise TypeError(
            "cute-dsl backend expects enable_pdl to be bool or None, got "
            f"{args.enable_pdl!r}."
        )
    if not isinstance(args.use_sinks, bool):
        raise TypeError(
            f"cute-dsl backend expects use_sinks to be bool, got {args.use_sinks!r}."
        )
    if (
        not isinstance(args.page_size, int)
        or isinstance(args.page_size, bool)
        or args.page_size <= 0
    ):
        raise ValueError(f"page_size must be a positive int, got {args.page_size!r}.")
    if args.page_size > 128 or 128 % args.page_size != 0:
        raise _BackendPlanUnsupportedError(
            "cute-dsl dense metadata requires page_size to divide 128, "
            f"got {args.page_size}."
        )
    if args.q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend supports bfloat16 or float8_e4m3fn query tensors, "
            f"got {args.q_data_type}."
        )
    if args.kv_data_type != args.q_data_type:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires kv_data_type to match q_data_type, got "
            f"{args.kv_data_type} and {args.q_data_type}."
        )
    if type(args.sm_scale) is not float or not math.isfinite(args.sm_scale):
        raise TypeError(
            "cute-dsl backend expects sm_scale to be a finite Python float, "
            f"got {args.sm_scale!r}."
        )
    major, minor = get_compute_capability(args._float_workspace_buffer.device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires SM100/SM103, got compute capability "
            f"SM{major}{minor}."
        )
    from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(major, minor):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend: the installed CuTe DSL does not support "
            f"sm_{major}{minor}."
        )


def _q_layout(cum_seq_lens_q: torch.Tensor) -> tuple[int, int, int, bool, int]:
    if cum_seq_lens_q.ndim != 1:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects cum_seq_lens_q to be rank-1."
        )
    if cum_seq_lens_q.numel() < 2:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects cum_seq_lens_q to contain at least two entries."
        )
    q_offsets = cum_seq_lens_q.to(device="cpu", dtype=torch.int64)
    q_lengths = q_offsets[1:] - q_offsets[:-1]
    if bool(torch.any(q_lengths <= 0).item()):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires positive query lengths."
        )
    q_len = int(q_lengths[0].item())
    is_uniform = not bool(torch.any(q_lengths != q_len).item())
    if not is_uniform:
        raise _BackendPlanUnsupportedError(
            "cute-dsl planned backend does not support compact variable-Q metadata."
        )
    return (
        q_lengths.numel(),
        int(q_offsets[-1].item()),
        int(q_lengths.max().item()),
        is_uniform,
        q_len,
    )


def _prepare_cute_dsl_mla_execution_state(
    *,
    workspace_buffer: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    sm_scale: float,
    q_data_type: torch.dtype,
    use_cuda_graph: bool,
    use_sinks: bool,
    compile_kernel: Callable[..., tuple[Any, Any, torch.Tensor, int, int]],
) -> _CuteDslMlaExecutionState:
    batch_size, total_q, actual_max_q_len, _, q_len = _q_layout(cum_seq_lens_q)
    if max_q_len < actual_max_q_len:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects max_q_len to be at least the maximum "
            f"query length {actual_max_q_len}, got {max_q_len}."
        )
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects rank-2 block_tables with batch dimension "
            f"{batch_size}, got {tuple(block_tables.shape)}."
        )
    if block_tables.shape[1] == 0:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects block_tables width to be positive."
        )
    if seq_lens.ndim != 1 or seq_lens.numel() != batch_size:
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend expects rank-1 seq_lens of length {batch_size}."
        )
    seq_lens_host = seq_lens.to(device="cpu", dtype=torch.int64)
    if bool(torch.any(seq_lens_host <= 0).item()):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires positive seq_lens."
        )
    has_variable_kv_lengths = bool(torch.any(seq_lens_host != seq_lens_host[0]).item())
    resolved_is_var_seq = use_cuda_graph or has_variable_kv_lengths

    out_dtype = torch.bfloat16
    try:
        implementation, compiled_kernel, workspace_i8, workspace_size, split_kv = (
            compile_kernel(
                workspace_buffer=workspace_buffer,
                device=workspace_buffer.device,
                q_data_type=q_data_type,
                out_dtype=out_dtype,
                page_size=page_size,
                batch_size=batch_size,
                num_heads=num_heads,
                q_len=q_len,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                resolved_is_var_seq=resolved_is_var_seq,
                use_sinks=use_sinks,
                enable_pdl=False,
            )
        )
    except _CuteDslKernelUnsupportedError as error:
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend unsupported configuration: {error}"
        ) from error
    if workspace_i8.numel() < workspace_size:
        raise ValueError(
            "workspace_buffer too small for cute-dsl backend: "
            f"have {workspace_i8.numel()} bytes, need {workspace_size} bytes."
        )
    lse_scratch = torch.empty(
        (batch_size, q_len, num_heads),
        dtype=torch.float32,
        device=workspace_buffer.device,
    )
    return _CuteDslMlaExecutionState(
        compiled_kernel=compiled_kernel,
        workspace_bytes=None if workspace_size == 0 else workspace_i8[:workspace_size],
        split_kv=split_kv,
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=batch_size,
        q_len=q_len,
        num_heads=num_heads,
        total_q=total_q,
        kv_lora_rank=head_dim_ckv,
        qk_rope_head_dim=head_dim_kpe,
        page_size=page_size,
        q_dtype=q_data_type,
        out_dtype=out_dtype,
        lse_scratch=lse_scratch,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        use_sinks=use_sinks,
        device=workspace_buffer.device,
        Float32=implementation.Float32,
        Int32=implementation.Int32,
    )


def _validate_scalar_bmm_scales(
    bmm1_scale: Optional[Union[float, torch.Tensor]],
    bmm2_scale: Optional[Union[float, torch.Tensor]],
) -> None:
    for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
        if isinstance(scale, torch.Tensor):
            raise ValueError(
                f"cute-dsl backend accepts {name} as a float only; tensor scales "
                "are not supported."
            )
        if scale is not None and (
            not isinstance(scale, Real)
            or isinstance(scale, bool)
            or not math.isfinite(scale)
        ):
            raise ValueError(
                f"cute-dsl backend expects {name} to be a finite Python float, "
                f"got {scale!r}."
            )


def _run_cute_dsl_mla_execution_state(
    *,
    state: _CuteDslMlaExecutionState,
    launch_kernel: Callable[[tuple[Any, ...], Optional[torch.Tensor]], None],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
    bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    supports_lse: bool = False,
    backend_name: str = "cute-dsl",
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if (sinks is not None) != state.use_sinks:
        expected = "with" if state.use_sinks else "without"
        raise ValueError(
            f"cute-dsl backend was planned {expected} use_sinks=True; "
            "run-time sinks must match the planned declaration."
        )
    if sinks is not None:
        check_shape_dtype_device(
            sinks, (state.num_heads,), torch.float32, state.device, "sinks"
        )
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous for cute-dsl backend.")
    if (return_lse or lse is not None) and not supports_lse:
        raise ValueError(f"{backend_name} does not support LSE.")
    if return_lse and lse is None:
        raise ValueError(
            "caller-owned lse must be provided when return_lse=True for "
            "cute-dsl planned backend runs."
        )
    _validate_scalar_bmm_scales(bmm1_scale, bmm2_scale)

    check_shape_dtype_device(
        query,
        (
            state.total_q,
            state.num_heads,
            state.kv_lora_rank + state.qk_rope_head_dim,
        ),
        state.q_dtype,
        state.device,
        "query",
    )
    check_shape_dtype_device(
        kv_cache,
        (
            kv_cache.shape[0],
            state.page_size,
            state.kv_lora_rank + state.qk_rope_head_dim,
        ),
        state.q_dtype,
        state.device,
        "kv_cache",
    )
    check_shape_dtype_device(
        out,
        (state.total_q, state.num_heads, state.kv_lora_rank),
        state.out_dtype,
        state.device,
        "out",
    )
    if not query.is_contiguous():
        raise ValueError("query must be contiguous for cute-dsl backend.")
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous for cute-dsl backend.")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous for cute-dsl backend.")
    if lse is not None:
        check_shape_dtype_device(
            lse, (state.total_q, state.num_heads), torch.float32, state.device, "lse"
        )
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous for cute-dsl backend.")

    query_4d = query.view(
        state.batch_size,
        state.q_len,
        state.num_heads,
        state.kv_lora_rank + state.qk_rope_head_dim,
    )
    out_4d = out.view(
        state.batch_size,
        state.q_len,
        state.num_heads,
        state.kv_lora_rank,
    )
    lse_kernel = (
        state.lse_scratch
        if lse is None
        else lse.view(state.batch_size, state.q_len, state.num_heads)
    )
    launch_args: tuple[Any, ...] = (
        query_4d[..., : state.kv_lora_rank],
        query_4d[..., state.kv_lora_rank :],
        kv_cache[..., : state.kv_lora_rank],
        kv_cache[..., state.kv_lora_rank :],
        state.block_tables,
        out_4d,
        lse_kernel,
        state.workspace_bytes,
        state.Int32(state.split_kv),
        state.seq_lens,
        None,
        state.Float32(state.bmm1_scale if bmm1_scale is None else float(bmm1_scale)),
        state.Float32(state.bmm2_scale if bmm2_scale is None else float(bmm2_scale)),
    )
    launch_kernel(launch_args, sinks)
    return (out, lse) if return_lse else out


class _BatchMLAPagedAttentionCuteDslBackendBase:
    """Shared state and validation for planned CuTe DSL MLA backends."""

    _backend_name = "cute-dsl"
    _supports_lse = False
    _reject_cuda_graph = False
    _plan_capability_error_type = _BackendPlanUnsupportedError
    _plan_capabilities: ClassVar[MLAPlanCapabilities]

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = self._backend_name
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCuteDslBackendBase":
        cls.preflight_plan_from_wrapper(args)
        dense = args.native_device_dense()
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
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            use_cuda_graph=args._use_cuda_graph,
            use_sinks=args.use_sinks,
        )
        return backend

    @classmethod
    def preflight_plan_from_wrapper(cls, args: _MLAPlanArguments) -> None:
        _validate_cute_dsl_plan_args_before_metadata(args)
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise _BackendPlanUnsupportedError(reason)
        if cls._reject_cuda_graph and args._use_cuda_graph:
            raise _BackendPlanUnsupportedError(
                f"{cls._backend_name} backend does not support CUDA graph planning."
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
        sm_scale: float,
        q_data_type: torch.dtype,
        use_cuda_graph: bool,
        use_sinks: bool,
    ) -> None:
        self._execution_state = _prepare_cute_dsl_mla_execution_state(
            workspace_buffer=self._float_workspace_buffer,
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            use_cuda_graph=use_cuda_graph,
            use_sinks=use_sinks,
            compile_kernel=self._compile_kernel,
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
        if not hasattr(self, "_execution_state"):
            raise RuntimeError(f"{type(self).__name__}.run() called before plan().")
        state = self._execution_state
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cute-dsl backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with cute-dsl backend."
            )
        if return_lse_base_on_e and not self._supports_lse:
            raise ValueError(
                "return_lse_base_on_e is not supported with cute-dsl backend."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with cute-dsl backend.")
        if ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / ckv_scale_arr / kpe_scale are not supported with "
                "cute-dsl planned backend."
            )
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with cute-dsl backend."
            )
        if out is None:
            raise ValueError("out must be provided for cute-dsl planned backend runs.")
        packed_query = _resolve_structural_mla_input(
            query,
            desired="packed",
            widths=(state.kv_lora_rank, state.qk_rope_head_dim),
            name="query",
        )
        packed_kv_cache = _resolve_structural_mla_input(
            kv_cache,
            desired="packed",
            widths=(state.kv_lora_rank, state.qk_rope_head_dim),
            name="KV cache",
        )
        return self.run(
            query=packed_query,
            kv_cache=packed_kv_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            sinks=sinks,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: torch.Tensor,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        sinks: Optional[torch.Tensor] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self, "_execution_state"):
            raise RuntimeError(f"{type(self).__name__}.run() called before plan().")
        return _run_cute_dsl_mla_execution_state(
            state=self._execution_state,
            launch_kernel=self._launch_compiled_kernel,
            query=query,
            kv_cache=kv_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            sinks=sinks,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            supports_lse=self._supports_lse,
            backend_name=self._backend,
        )

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
        use_sinks: bool,
        enable_pdl: bool,
    ) -> tuple[Any, Any, torch.Tensor, int, int]:
        del (
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
            use_sinks,
            enable_pdl,
        )
        raise RuntimeError("CuTe DSL backend base cannot be planned directly.")

    def _launch_compiled_kernel(
        self, launch_args: tuple[Any, ...], sinks: Optional[torch.Tensor]
    ) -> None:
        del launch_args, sinks
        raise RuntimeError("CuTe DSL backend base cannot be launched directly.")
