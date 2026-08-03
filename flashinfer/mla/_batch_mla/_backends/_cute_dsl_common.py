"""Shared mechanics for the concrete CuTe DSL MLA backends."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.utils import (
    check_shape_dtype_device,
    get_compute_capability,
)

from ._capabilities import MLAPlanCapabilities, validate_plan_capabilities
from .._planning import (
    _MLAPlanArguments,
)
from .._contracts import (
    MLAKVCache,
    MLAQuery,
)


class _CuteDslKernelUnsupportedError(RuntimeError):
    """A known native CuTe capability refusal, safe for family fallback."""


@dataclass
class _CuteDslMlaExecutionState:
    """Native CuTe launch state with no planned-wrapper ownership."""

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
    bmm1_scale: float
    bmm2_scale: float
    use_sinks: bool
    device: torch.device
    Float32: Any
    Int32: Any


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
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    use_profiler: bool,
    is_var_seq: Optional[bool],
    use_sinks: bool,
    enable_pdl: Optional[bool],
    compile_kernel: Callable[..., Tuple[Any, Any, torch.Tensor, int, int]],
) -> _CuteDslMlaExecutionState:
    """Validate metadata and compile one concrete native CuTe launch."""
    device = workspace_buffer.device
    cc = get_compute_capability(device)
    if cc not in ((10, 0), (10, 3)):
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend requires SM100/SM103, got SM{cc[0]}{cc[1]}."
        )
    if causal:
        raise _BackendPlanUnsupportedError(
            "causal=True is not supported by the cute-dsl backend."
        )
    if use_profiler:
        raise _BackendPlanUnsupportedError(
            "use_profiler is not supported by the cute-dsl backend."
        )
    if enable_pdl is not None and not isinstance(enable_pdl, bool):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects enable_pdl to be bool or None, got "
            f"{enable_pdl!r}."
        )
    if not isinstance(use_sinks, bool):
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend expects use_sinks to be bool, got {use_sinks!r}."
        )
    if is_var_seq is not None and not isinstance(is_var_seq, bool):
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend expects is_var_seq to be bool or None, got {is_var_seq!r}."
        )
    if q_data_type not in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend supports float16, bfloat16, or float8_e4m3fn "
            f"query tensors, got {q_data_type}."
        )
    if kv_data_type != q_data_type:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires kv_data_type to match q_data_type, got "
            f"{kv_data_type} and {q_data_type}."
        )
    if not isinstance(sm_scale, (float, int)) or not math.isfinite(sm_scale):
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend expects a finite float sm_scale, got {sm_scale!r}."
        )
    for name, tensor in (
        ("cum_seq_lens_q", cum_seq_lens_q),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
    ):
        if tensor.dtype != torch.int32:
            raise _BackendPlanUnsupportedError(
                f"cute-dsl backend expects {name} to have dtype torch.int32, got {tensor.dtype}."
            )
    q_offsets = cum_seq_lens_q.to(device="cpu", dtype=torch.int64)
    q_lengths = q_offsets[1:] - q_offsets[:-1]
    batch_size = q_lengths.numel()
    if batch_size == 0 or torch.any(q_lengths <= 0).item():
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires a non-empty batch with positive query lengths."
        )
    q_len = int(q_lengths[0].item())
    if torch.any(q_lengths != q_len).item():
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires uniform query lengths."
        )
    if max_q_len < q_len:
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend requires max_q_len >= {q_len}, got {max_q_len}."
        )
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend expects rank-2 block_tables with batch dimension "
            f"{batch_size}, got {tuple(block_tables.shape)}."
        )
    if seq_lens.ndim != 1 or seq_lens.numel() != batch_size:
        raise _BackendPlanUnsupportedError(
            f"cute-dsl backend expects rank-1 seq_lens of length {batch_size}."
        )
    seq_lens_host = seq_lens.to(device="cpu", dtype=torch.int64)
    if torch.any(seq_lens_host <= 0).item():
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires positive seq_lens."
        )
    resolved_is_var_seq = (
        bool(torch.any(seq_lens_host != seq_lens_host[0]).item())
        if is_var_seq is None
        else is_var_seq
    )
    if not resolved_is_var_seq and torch.any(seq_lens_host != seq_lens_host[0]).item():
        raise _BackendPlanUnsupportedError(
            "cute-dsl backend requires is_var_seq=True for non-uniform seq_lens."
        )

    out_dtype = torch.bfloat16
    try:
        implementation, compiled_kernel, workspace_i8, workspace_size, split_kv = (
            compile_kernel(
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
                use_sinks=use_sinks,
                enable_pdl=bool(enable_pdl),
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
    return _CuteDslMlaExecutionState(
        compiled_kernel=compiled_kernel,
        workspace_bytes=(
            None if workspace_size == 0 else workspace_i8[:workspace_size]
        ),
        split_kv=split_kv,
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=batch_size,
        q_len=q_len,
        num_heads=num_heads,
        total_q=int(q_offsets[-1].item()),
        kv_lora_rank=head_dim_ckv,
        qk_rope_head_dim=head_dim_kpe,
        page_size=page_size,
        q_dtype=q_data_type,
        out_dtype=out_dtype,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        use_sinks=use_sinks,
        device=device,
        Float32=implementation.Float32,
        Int32=implementation.Int32,
    )


def _run_cute_dsl_mla_execution_state(
    *,
    state: _CuteDslMlaExecutionState,
    launch_kernel: Callable[
        [_CuteDslMlaExecutionState, Tuple[Any, ...], Optional[torch.Tensor]], None
    ],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    sinks: Optional[torch.Tensor] = None,
    bmm1_scale: Optional[float] = None,
    bmm2_scale: Optional[float] = None,
    supports_lse: bool = False,
    backend_name: str = "cute-dsl",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Validate dynamic tensors and launch a prepared native CuTe kernel."""
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
        raise ValueError(
            f"{backend_name} does not support LSE; plan "
            "backend='cute-dsl-monolithic' to request LSE."
        )
    for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
        if isinstance(scale, torch.Tensor):
            raise ValueError(
                f"cute-dsl backend accepts {name} as a float only; tensor scales are not supported."
            )
        if scale is not None and (type(scale) is not float or not math.isfinite(scale)):
            raise ValueError(
                f"cute-dsl backend expects {name} to be a finite Python float, got {scale!r}."
            )

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
    if out is None:
        out = torch.empty(
            (state.total_q, state.num_heads, state.kv_lora_rank),
            dtype=state.out_dtype,
            device=state.device,
        )
    else:
        check_shape_dtype_device(
            out,
            (state.total_q, state.num_heads, state.kv_lora_rank),
            state.out_dtype,
            state.device,
            "out",
        )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for cute-dsl backend.")
    if lse is not None:
        check_shape_dtype_device(
            lse,
            (state.total_q, state.num_heads),
            torch.float32,
            state.device,
            "lse",
        )
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous for cute-dsl backend.")
    elif return_lse:
        lse = torch.empty(
            (state.total_q, state.num_heads),
            dtype=torch.float32,
            device=state.device,
        )

    query = query.reshape(
        state.batch_size,
        state.q_len,
        state.num_heads,
        state.kv_lora_rank + state.qk_rope_head_dim,
    )
    q_latent = query[..., : state.kv_lora_rank]
    q_rope = query[..., state.kv_lora_rank :]
    c_latent = kv_cache[..., : state.kv_lora_rank]
    c_rope = kv_cache[..., state.kv_lora_rank :]
    lse_kernel = lse
    if lse_kernel is None:
        lse_kernel = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )
    elif lse_kernel.ndim == 2:
        lse_kernel = lse_kernel.view(state.batch_size, state.q_len, state.num_heads)
    launch_args: Tuple[Any, ...] = (
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        state.block_tables,
        out.view(
            state.batch_size,
            state.q_len,
            state.num_heads,
            state.kv_lora_rank,
        ),
        lse_kernel,
        state.workspace_bytes,
        state.Int32(state.split_kv),
        state.seq_lens,
        None,
        state.Float32(state.bmm1_scale if bmm1_scale is None else float(bmm1_scale)),
        state.Float32(state.bmm2_scale if bmm2_scale is None else float(bmm2_scale)),
    )
    launch_kernel(state, launch_args, sinks)
    if return_lse:
        assert lse is not None
        return out, lse
    return out


class _BatchMLAPagedAttentionCuteDslBackendBase:
    """Shared state and validation for concrete CuTe DSL MLA backends."""

    _backend_name = "cute-dsl"
    _supports_lse = False
    _plan_capabilities: MLAPlanCapabilities

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = self._backend_name
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCuteDslBackendBase":
        validate_plan_capabilities(args, cls._plan_capabilities)
        output_dtype = args.output_dtype
        if output_dtype != torch.bfloat16:
            raise _BackendPlanUnsupportedError(
                f"{cls._backend_name} backend requires a bfloat16 output contract "
                "without o_scale."
            )
        is_var_seq = args.is_var_seq
        use_sinks = args.use_sinks
        if args.use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the cute-dsl wrapper backend."
            )
        if args.causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the cute-dsl wrapper backend."
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
                "cute-dsl dense metadata requires page_size to divide 128, "
                f"got {args.page_size}."
            )
        dense = args.dense(
            table_width_alignment=128 // args.page_size,
        )
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
        is_var_seq: Optional[bool],
        use_sinks: bool,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        state = _prepare_cute_dsl_mla_execution_state(
            workspace_buffer=self._float_workspace_buffer,
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            use_profiler=use_profiler,
            is_var_seq=is_var_seq,
            use_sinks=use_sinks,
            enable_pdl=enable_pdl,
            compile_kernel=self._compile_kernel,
        )
        self._execution_state = state
        # These metadata tensors remain wrapper-owned so CUDA graph replay
        # retains the same references published by plan().
        self._cum_seq_lens_q = cum_seq_lens_q
        self._block_tables = block_tables
        self._seq_lens = seq_lens

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
            raise ValueError("profiler_buffer is not supported with cute-dsl backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with cute-dsl backend."
            )
        if return_lse_base_on_e and not self._supports_lse:
            raise ValueError(
                "return_lse_base_on_e is not supported with cute-dsl backend; "
                "CuTe DSL LSE is already returned in natural-log base."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with cute-dsl backend.")
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are not supported with cute-dsl backend."
            )
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with "
                "cute-dsl backend."
            )
        return self.run(
            query=packed_query,
            kv_cache=kv_cache,
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
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        sinks: Optional[torch.Tensor] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
    ):
        if not hasattr(self, "_execution_state"):
            raise RuntimeError(f"{type(self).__name__}.run() called before plan().")
        return _run_cute_dsl_mla_execution_state(
            state=self._execution_state,
            launch_kernel=lambda state, launch_args, launch_sinks: (
                self._launch_compiled_kernel(launch_args, launch_sinks)
            ),
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
    ) -> Tuple[Any, Any, torch.Tensor, int, int]:
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
        raise RuntimeError(
            "CuTe DSL backend base cannot be planned directly; "
            "select a concrete implementation."
        )

    def _launch_compiled_kernel(
        self, launch_args: Tuple[Any, ...], sinks: Optional[torch.Tensor]
    ) -> None:
        del launch_args, sinks
        raise RuntimeError(
            "CuTe DSL backend base cannot be launched directly; "
            "select a concrete implementation."
        )
