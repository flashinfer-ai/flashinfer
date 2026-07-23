"""CuTe DSL implementation compartment for functional and planned MLA decode."""

import math
from typing import Any, List, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import TunableRunner, is_in_profile_measurement
from flashinfer.utils import (
    check_shape_dtype_device,
    get_compute_capability,
    next_positive_power_of_2,
)

from .._planning import (
    _audit_plan_from_wrapper_arguments,
    _MLAPlanArguments,
    _MLAWrapperPlanResult,
)
from ._layout import _concat_adjacent_views_or_cat


def _cute_dsl_max_supported_batch(
    workspace_bytes: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    candidate_max: int,
) -> int:
    """Largest batch the caller's workspace can support for CuTe DSL MLA."""
    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        _get_split_kv_and_workspace_size,
    )

    lo, hi = 1, max(1, candidate_max)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, workspace_size = _get_split_kv_and_workspace_size(
            mid, q_len, num_heads, kv_lora_rank, max_active_blocks
        )
        if workspace_size <= workspace_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cute_dsl_incompatibility_reason(
    query: torch.Tensor,
    out_dtype: torch.dtype,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    sparse_mla_top_k: int,
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    page_size: int,
    is_var_seq: bool,
    return_lse: bool,
    lse: Optional[torch.Tensor],
    cute_dsl_impl: str = "auto",
) -> Optional[str]:
    """Return ``None`` when CuTe DSL can implement the functional request."""
    del return_lse, lse
    cc = get_compute_capability(query.device)
    if cc[0] < 10:
        return (
            "cute-dsl backend (MLA decode kernel) requires SM100+, "
            f"got SM{cc[0]}{cc[1]}"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor "
            "bmm1_scale, please pass a float value"
        )
    if isinstance(bmm2_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor "
            "bmm2_scale, please pass a float value"
        )
    if isinstance(sinks, (list, tuple)) and len(sinks) != 1:
        return (
            "cute-dsl backend (MLA decode kernel) expects sinks to be a "
            f"single tensor or a length-1 list/tuple; got len={len(sinks)}"
        )
    if sparse_mla_top_k > 0:
        return "cute-dsl backend (MLA decode kernel) does not support sparse_mla_top_k"
    if skip_softmax_threshold_scale_factor is not None:
        return (
            "cute-dsl backend (MLA decode kernel) does not support "
            "skip_softmax_threshold_scale_factor"
        )
    if not uses_shared_paged_kv_idx:
        return (
            "cute-dsl backend (MLA decode kernel) does not support separate KV "
            "page indices (uses_shared_paged_kv_idx=False)"
        )

    _, q_len, num_heads, _ = query.shape
    try:
        from flashinfer.cute_dsl.attention.mla_dispatch import _resolve_impl

        resolved_impl = _resolve_impl(requested=cute_dsl_impl, kwargs={"sinks": sinks})
    except (ValueError, ImportError) as error:
        return f"cute-dsl backend (MLA decode kernel): {error}"

    try:
        if resolved_impl == "monolithic":
            from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
                _check_can_implement,
            )
        else:
            from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
                _check_can_implement,
            )

        _check_can_implement(
            torch_dtype=query.dtype,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            is_persistent=not is_var_seq,
            is_var_seq=is_var_seq,
            is_var_split_kv=False,
        )
    except (ValueError, ImportError) as error:
        return (
            "cute-dsl backend (MLA decode kernel) cannot implement this "
            f"configuration: {error}"
        )
    return None


class _BatchMLAPagedAttentionCuteDslBackend:
    """Stateful explicit-only CuTe DSL backend for the public MLA wrapper."""

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "cute-dsl"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        enable_pdl = args.enable_pdl
        qk_nope_head_dim = args.qk_nope_head_dim
        if enable_pdl:
            raise ValueError(
                "enable_pdl is not supported by the cute-dsl wrapper backend."
            )
        if args.use_profiler:
            raise ValueError(
                "use_profiler is not supported by the cute-dsl wrapper backend."
            )
        if args.causal:
            raise ValueError(
                "causal=True is not supported by the cute-dsl wrapper backend."
            )
        if qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is not supported by the cute-dsl wrapper backend."
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
            raise ValueError(
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
            is_var_seq=args.is_var_seq,
            cute_dsl_impl=args.cute_dsl_impl,
            use_sinks=args.use_sinks,
        )
        return _MLAWrapperPlanResult(backend_impl=backend)

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
        cute_dsl_impl: str,
        use_sinks: bool,
        enable_pdl: Optional[bool] = None,
    ) -> None:
        cc = get_compute_capability(self.device)
        if cc[0] < 10:
            raise _BackendPlanUnsupportedError(
                f"cute-dsl backend requires SM100+, got SM{cc[0]}{cc[1]}."
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
        if (
            not resolved_is_var_seq
            and torch.any(seq_lens_host != seq_lens_host[0]).item()
        ):
            raise _BackendPlanUnsupportedError(
                "cute-dsl backend requires is_var_seq=True for non-uniform seq_lens."
            )

        try:
            from flashinfer.cute_dsl.attention.mla_dispatch import _resolve_impl

            resolved_impl = _resolve_impl(
                requested=cute_dsl_impl,
                kwargs={"sinks": object() if use_sinks else None},
            )
            # Match the existing functional MLA controller, whose public
            # CuTe DSL path always returns BF16 regardless of input dtype.
            out_dtype = torch.bfloat16
            if resolved_impl == "monolithic":
                from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
                    _check_can_implement,
                )
            else:
                from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
                    _check_can_implement,
                )
            _check_can_implement(
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
            if resolved_impl == "monolithic":
                from flashinfer.cute_dsl.attention.monolithic import (
                    mla_decode as implementation,
                )

                workspace_i8 = implementation._as_cute_dsl_workspace_i8(
                    self._float_workspace_buffer
                )
                split_kv, workspace_size = (
                    implementation._get_split_kv_and_workspace_size(
                        batch_size,
                        q_len,
                        num_heads,
                        head_dim_ckv,
                        implementation.get_num_sm(self.device),
                    )
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
                    enable_pdl=bool(enable_pdl),
                )
            else:
                from flashinfer.cute_dsl.attention.fusion.variant import (
                    AttentionWithSink,
                )
                from flashinfer.cute_dsl.attention.wrappers import (
                    batch_mla as implementation,
                )

                workspace_i8 = implementation._as_cute_dsl_workspace_i8(
                    self._float_workspace_buffer
                )
                split_kv, workspace_size = (
                    implementation._get_split_kv_and_workspace_size(
                        batch_size,
                        q_len,
                        num_heads,
                        head_dim_ckv,
                        implementation.get_num_sm(self.device),
                    )
                )
                variant = None
                params_shape = None
                if use_sinks:
                    placeholder = torch.empty(
                        (num_heads,), dtype=torch.float32, device=self.device
                    )
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
                    enable_pdl=bool(enable_pdl),
                    variant=variant,
                    params_shape=params_shape,
                )
            if workspace_i8.numel() < workspace_size:
                raise ValueError(
                    "workspace_buffer too small for cute-dsl backend: "
                    f"have {workspace_i8.numel()} bytes, need {workspace_size} bytes."
                )
        except (ImportError, ValueError) as error:
            raise _BackendPlanUnsupportedError(
                f"cute-dsl backend unsupported configuration: {error}"
            ) from error

        self._cum_seq_lens_q = cum_seq_lens_q
        self._block_tables = block_tables
        self._seq_lens = seq_lens
        self._batch_size = batch_size
        self._q_len = q_len
        self._total_q = int(q_offsets[-1].item())
        self._num_heads = num_heads
        self._kv_lora_rank = head_dim_ckv
        self._qk_rope_head_dim = head_dim_kpe
        self._page_size = page_size
        self._q_dtype = q_data_type
        self._out_dtype = out_dtype
        self._bmm1_scale = float(sm_scale)
        self._bmm2_scale = 1.0
        self._use_sinks = use_sinks
        self._cute_dsl_impl = resolved_impl
        self._supports_lse = resolved_impl == "monolithic"
        self._Float32 = implementation.Float32
        self._Int32 = implementation.Int32
        self._compiled_kernel = compiled_kernel
        self._workspace_bytes = (
            None if workspace_size == 0 else workspace_i8[:workspace_size]
        )
        self._split_kv = split_kv

    def run_from_wrapper(
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
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cute-dsl backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with cute-dsl backend."
            )
        if return_lse_base_on_e:
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
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
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
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        sinks: Optional[torch.Tensor] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
    ):
        if not hasattr(self, "_compiled_kernel"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionCuteDslBackend.run() called before plan()."
            )
        if (sinks is not None) != self._use_sinks:
            expected = "with" if self._use_sinks else "without"
            raise ValueError(
                f"cute-dsl backend was planned {expected} use_sinks=True; "
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
            if not sinks.is_contiguous():
                raise ValueError("sinks must be contiguous for cute-dsl backend.")
        if (return_lse or lse is not None) and not self._supports_lse:
            raise ValueError(
                "cute-dsl modular implementation does not support LSE; plan "
                "cute_dsl_impl='monolithic' without sinks to request LSE."
            )
        for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
            if isinstance(scale, torch.Tensor):
                raise ValueError(
                    f"cute-dsl backend accepts {name} as a float only; tensor scales are not supported."
                )
            if scale is not None and (
                type(scale) is not float or not math.isfinite(scale)
            ):
                raise ValueError(
                    f"cute-dsl backend expects {name} to be a finite Python float, got {scale!r}."
                )

        check_shape_dtype_device(
            q_nope,
            (self._total_q, self._num_heads, self._kv_lora_rank),
            self._q_dtype,
            self.device,
            "q_nope",
        )
        check_shape_dtype_device(
            q_pe,
            (self._total_q, self._num_heads, self._qk_rope_head_dim),
            self._q_dtype,
            self.device,
            "q_pe",
        )
        check_shape_dtype_device(
            ckv_cache,
            (ckv_cache.shape[0], self._page_size, self._kv_lora_rank),
            self._q_dtype,
            self.device,
            "ckv_cache",
        )
        check_shape_dtype_device(
            kpe_cache,
            (ckv_cache.shape[0], self._page_size, self._qk_rope_head_dim),
            self._q_dtype,
            self.device,
            "kpe_cache",
        )
        if out is None:
            out = torch.empty(
                (self._total_q, self._num_heads, self._kv_lora_rank),
                dtype=self._out_dtype,
                device=self.device,
            )
        else:
            check_shape_dtype_device(
                out,
                (self._total_q, self._num_heads, self._kv_lora_rank),
                self._out_dtype,
                self.device,
                "out",
            )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous for cute-dsl backend.")
        if lse is not None:
            check_shape_dtype_device(
                lse,
                (self._total_q, self._num_heads),
                torch.float32,
                self.device,
                "lse",
            )
            if not lse.is_contiguous():
                raise ValueError("lse must be contiguous for cute-dsl backend.")
        elif return_lse:
            lse = torch.empty(
                (self._total_q, self._num_heads),
                dtype=torch.float32,
                device=self.device,
            )

        query = _concat_adjacent_views_or_cat(q_nope, q_pe).reshape(
            self._batch_size,
            self._q_len,
            self._num_heads,
            self._kv_lora_rank + self._qk_rope_head_dim,
        )
        kv_cache = _concat_adjacent_views_or_cat(ckv_cache, kpe_cache)
        q_latent = query[..., : self._kv_lora_rank]
        q_rope = query[..., self._kv_lora_rank :]
        c_latent = kv_cache[..., : self._kv_lora_rank]
        c_rope = kv_cache[..., self._kv_lora_rank :]
        lse_kernel = lse
        if lse_kernel is None:
            lse_kernel = torch.empty(
                query.shape[:-1], dtype=torch.float32, device=query.device
            )
        elif lse_kernel.ndim == 2:
            lse_kernel = lse_kernel.view(self._batch_size, self._q_len, self._num_heads)
        launch_args: Tuple[Any, ...] = (
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            self._block_tables,
            out.view(
                self._batch_size,
                self._q_len,
                self._num_heads,
                self._kv_lora_rank,
            ),
            lse_kernel,
            self._workspace_bytes,
            self._Int32(self._split_kv),
            self._seq_lens,
            None,
            self._Float32(
                self._bmm1_scale if bmm1_scale is None else float(bmm1_scale)
            ),
            self._Float32(
                self._bmm2_scale if bmm2_scale is None else float(bmm2_scale)
            ),
        )
        if self._cute_dsl_impl == "modular":
            launch_args += (sinks,)
        self._compiled_kernel(*launch_args)
        if return_lse:
            return out, lse
        return out


class CuteDslMlaDecodeRunner(TunableRunner):
    """Functional autotuner adapter over the planned CuTe DSL backend."""

    def __init__(
        self,
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
        sinks: Optional[torch.Tensor],
        cute_dsl_impl: str,
    ):
        self.kv_cache = kv_cache
        self.workspace_buffer = workspace_buffer
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = kv_cache.shape[-2]
        self.max_seq_len = max_seq_len
        self.softmax_scale = softmax_scale
        self.output_scale = output_scale
        self.out_dtype = out_dtype
        self.enable_pdl = enable_pdl
        self.is_var_seq = is_var_seq
        self.uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        self.lse = lse
        self.return_lse = return_lse
        self.sinks = sinks
        self.cute_dsl_impl = cute_dsl_impl
        self._prepared_backend: Optional[_BatchMLAPagedAttentionCuteDslBackend] = None

    def _plan_backend(
        self,
        *,
        query: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> _BatchMLAPagedAttentionCuteDslBackend:
        batch_size, q_len, num_heads, _ = query.shape
        cum_seq_lens_q = torch.arange(
            batch_size + 1, device=query.device, dtype=torch.int32
        ).mul_(q_len)
        backend = _BatchMLAPagedAttentionCuteDslBackend(self.workspace_buffer)
        backend.plan(
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=q_len,
            num_heads=num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=False,
            sm_scale=self.softmax_scale,
            q_data_type=query.dtype,
            kv_data_type=self.kv_cache.dtype,
            use_profiler=False,
            is_var_seq=self.is_var_seq,
            cute_dsl_impl=self.cute_dsl_impl,
            use_sinks=self.sinks is not None,
            enable_pdl=self.enable_pdl,
        )
        return backend

    def __hash__(self):
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        del profile
        from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
            _get_split_kv_and_workspace_size,
        )
        from flashinfer.cute_dsl.utils import get_num_sm

        query = inputs[0]
        batch_size, q_len, num_heads, _ = query.shape
        _, workspace_size = _get_split_kv_and_workspace_size(
            batch_size,
            q_len,
            num_heads,
            self.kv_lora_rank,
            get_num_sm(query.device),
        )
        workspace_bytes = (
            self.workspace_buffer.numel() * self.workspace_buffer.element_size()
        )
        if workspace_size > workspace_bytes:
            return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        query, _, _, out = inputs
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
            next_positive_power_of_2(self.max_seq_len),
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
        **kwargs,
    ):
        del tactic, kwargs
        query, block_tables, seq_lens, out = inputs
        if do_preparation:
            # The profile's cold-L2 input batches are clones of these tensors.
            # They share shapes and metadata values, but not object identity.
            # Retain the plan so graph capture never performs planning work.
            self._prepared_backend = self._plan_backend(
                query=query,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )
            backend = self._prepared_backend
        elif is_in_profile_measurement():
            if self._prepared_backend is None:
                raise RuntimeError(
                    "CuTe DSL autotuner launch was not prepared before profiling."
                )
            backend = self._prepared_backend
        else:
            backend = self._plan_backend(
                query=query,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )
        return backend.run(
            q_nope=query[..., : self.kv_lora_rank].flatten(0, 1),
            q_pe=query[..., self.kv_lora_rank :].flatten(0, 1),
            ckv_cache=self.kv_cache.squeeze(1)[..., : self.kv_lora_rank],
            kpe_cache=self.kv_cache.squeeze(1)[..., self.kv_lora_rank :],
            out=out.flatten(0, 1),
            lse=self.lse,
            return_lse=self.return_lse,
            sinks=self.sinks,
            bmm1_scale=self.softmax_scale,
            bmm2_scale=self.output_scale,
        )
