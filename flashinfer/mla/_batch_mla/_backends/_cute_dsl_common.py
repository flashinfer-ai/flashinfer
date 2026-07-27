"""Shared mechanics for the concrete CuTe DSL MLA backends."""

import math
from typing import Any, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.utils import (
    check_shape_dtype_device,
    get_compute_capability,
)

from .._planning import (
    _audit_plan_from_wrapper_arguments,
    _MLAPlanArguments,
    _MLAWrapperPlanResult,
)
from ._layout import _concat_adjacent_views_or_cat


class _BatchMLAPagedAttentionCuteDslBackendBase:
    """Shared state and validation for concrete CuTe DSL MLA backends."""

    _backend_name = "cute-dsl"
    _supports_lse = False

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = self._backend_name
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
            # Match the existing functional MLA controller, whose public
            # CuTe DSL path always returns BF16 regardless of input dtype.
            out_dtype = torch.bfloat16
            (
                implementation,
                compiled_kernel,
                workspace_i8,
                workspace_size,
                split_kv,
            ) = self._compile_kernel(
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
        kv_cache: Optional[torch.Tensor],
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
        if kv_cache is None:
            raise ValueError(
                "CuTe DSL KV cache must be adjacent views or a combined kv_cache."
            )
        return self.run(
            q_nope=q_nope,
            q_pe=q_pe,
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
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        sinks: Optional[torch.Tensor] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
    ):
        if not hasattr(self, "_compiled_kernel"):
            raise RuntimeError(f"{type(self).__name__}.run() called before plan().")
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
                f"{self._backend} does not support LSE; plan "
                "backend='cute-dsl-monolithic' to request LSE."
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
            kv_cache,
            (
                kv_cache.shape[0],
                self._page_size,
                self._kv_lora_rank + self._qk_rope_head_dim,
            ),
            self._q_dtype,
            self.device,
            "kv_cache",
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
        self._launch_compiled_kernel(launch_args, sinks)
        if return_lse:
            return out, lse
        return out

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
        del (
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
