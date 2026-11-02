"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations

import functools
import math
from numbers import Real
from typing import ClassVar, Optional, Union

import torch

from flashinfer.autotuner import TunableRunner
from flashinfer.utils import (
    _check_block_tables_shape,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    is_sm12x_supported,
)

from ..._utils import _check_mla_dense_page_table_shape, _check_mla_query_kv_shape
from .._contracts import _resolve_structural_mla_input
from .._planning import _MLAPlanArguments, _audit_plan_from_wrapper_arguments
from ._capabilities import (
    _BackendPlanUnsupportedError,
    MLAPlanCapabilities,
    plan_capability_rejection_reason,
)


_SUPPORTED_MLA_DIMENSIONS = frozenset({(512, 64)})
_SUPPORTED_XQA_PAGE_SIZES = frozenset({16, 32, 64, 128})
_XQA_MIN_WORKSPACE_BYTES = 128 * 1024 * 1024
_XQA_SEMAPHORE_BYTES = 8 * 1024 * 1024


@functools.lru_cache(maxsize=128)
def get_xqa_module_mla(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool = False,
):
    from flashinfer.xqa import get_xqa_module_mla as _get_xqa_module_mla

    return _get_xqa_module_mla(
        input_dtype,
        kv_cache_dtype,
        page_size,
        head_dim,
        head_group_ratio,
        use_sliding_window,
    )


def _validate_xqa_device_capability(device: torch.device) -> None:
    if device.type != "cuda":
        raise _BackendPlanUnsupportedError(
            "XQA MLA wrapper requires a CUDA device with SM12x capability."
        )
    major, minor = get_compute_capability(device)
    if not is_sm12x_supported(device):
        raise _BackendPlanUnsupportedError(
            "XQA MLA wrapper requires SM120a (CUDA >= 12.8) or SM12x "
            "minor >= 1 (CUDA >= 12.9), got compute capability "
            f"SM{major}{minor} with CUDA {torch.version.cuda}."
        )


def _validate_xqa_plan_contract(args: _MLAPlanArguments) -> None:
    if args.lse_mode != "none":
        raise _BackendPlanUnsupportedError(
            "XQA backend does not support this LSE contract."
        )
    if args.output_dtype != torch.bfloat16:
        raise _BackendPlanUnsupportedError(
            "XQA backend requires a bfloat16 output contract without o_scale."
        )
    if args.output_scale != "none":
        raise _BackendPlanUnsupportedError(
            "XQA backend does not support this output contract."
        )
    if args.scale_mode not in ("default", "bmm-scalar"):
        raise _BackendPlanUnsupportedError(
            "XQA backend does not support this scale contract."
        )
    if args.skip_softmax:
        raise _BackendPlanUnsupportedError(
            "XQA backend does not support the skip-softmax contract."
        )
    if args.use_sinks:
        raise _BackendPlanUnsupportedError("XQA backend does not support sink inputs.")
    if args.use_profiler:
        raise _BackendPlanUnsupportedError(
            "use_profiler is not supported by the XQA backend."
        )
    if args.causal:
        raise _BackendPlanUnsupportedError(
            "causal=True is not supported by the XQA backend."
        )
    if args.query_kind == "independent-split":
        raise _BackendPlanUnsupportedError(
            "XQA backend requires a packed query view; representative query is "
            "independent split-only."
        )
    if args.kv_kind == "independent-split":
        raise _BackendPlanUnsupportedError(
            "XQA backend requires a packed KV-cache view; representative kv_cache "
            "is independent split-only."
        )
    if args.num_heads != 128:
        raise _BackendPlanUnsupportedError(
            f"XQA MLA only supports 128 query heads, got {args.num_heads}."
        )
    if (args.head_dim_ckv, args.head_dim_kpe) not in _SUPPORTED_MLA_DIMENSIONS:
        raise _BackendPlanUnsupportedError(
            "Unsupported MLA dimensions for XQA wrapper, got "
            f"head_dim_ckv={args.head_dim_ckv} and head_dim_kpe={args.head_dim_kpe}; "
            f"supported dimensions are {sorted(_SUPPORTED_MLA_DIMENSIONS)}."
        )
    if args.page_size not in _SUPPORTED_XQA_PAGE_SIZES:
        raise _BackendPlanUnsupportedError(
            "XQA MLA page_size must be one of "
            f"{sorted(_SUPPORTED_XQA_PAGE_SIZES)}, got {args.page_size}."
        )
    if args.q_data_type != args.kv_data_type:
        raise _BackendPlanUnsupportedError(
            "XQA MLA query and KV cache must use the same dtype, got "
            f"{args.q_data_type} and {args.kv_data_type}."
        )
    if args.q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
        raise _BackendPlanUnsupportedError(
            "XQA MLA wrapper supports BF16 or FP8 E4M3 inputs only, "
            f"got {args.q_data_type}."
        )
    if type(args.sm_scale) is not float or not math.isfinite(args.sm_scale):
        raise TypeError(
            "XQA MLA wrapper expects sm_scale to be a finite Python float, "
            f"got {args.sm_scale!r}."
        )
    if args.enable_pdl is not None and type(args.enable_pdl) is not bool:
        raise TypeError(
            "XQA MLA wrapper expects enable_pdl to be bool or None, got "
            f"{args.enable_pdl!r}."
        )


def _validate_scalar_bmm_scales(
    bmm1_scale: Optional[Union[float, torch.Tensor]],
    bmm2_scale: Optional[Union[float, torch.Tensor]],
) -> None:
    for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
        if isinstance(scale, torch.Tensor):
            raise ValueError(
                f"XQA MLA wrapper accepts {name} as a float only; tensor scales "
                "are not supported."
            )
        if scale is not None and (
            not isinstance(scale, Real)
            or isinstance(scale, bool)
            or not math.isfinite(scale)
        ):
            raise ValueError(
                f"XQA MLA wrapper expects {name} to be a finite Python float, "
                f"got {scale!r}."
            )


class _BatchMLAPagedAttentionXqaBackend(TunableRunner):
    """Functional, planned and tunable XQA MLA execution."""

    _plan_capability_error_type = _BackendPlanUnsupportedError
    _bmm1_scale: Union[float, torch.Tensor]
    _bmm2_scale: Union[float, torch.Tensor]
    _kv_cache: torch.Tensor
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="XQA",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar"}),
        supports_enable_pdl=True,
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "xqa"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._is_planned = True

    # Wrapper preparation and execution.

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionXqaBackend":
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            _validate_xqa_plan_contract(args)
            if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
                raise _BackendPlanUnsupportedError(reason)
            _validate_xqa_device_capability(args._float_workspace_buffer.device)
            args.require_cuda_graph_dense_metadata("xqa")
            dense = args.device_dense(table_width_alignment=128 // args.page_size)
            backend = cls(args._float_workspace_buffer)
            workspace = args._float_workspace_buffer
            if not workspace.is_contiguous():
                raise ValueError(
                    "workspace buffer must be contiguous for XQA MLA wrapper."
                )
            workspace_u8 = workspace.view(torch.uint8).flatten()
            if workspace_u8.numel() < _XQA_MIN_WORKSPACE_BYTES:
                raise _BackendPlanUnsupportedError(
                    "XQA MLA wrapper workspace must contain at least 128 MiB, got "
                    f"{workspace_u8.numel()} bytes."
                )
            # plan() zeroes only the semaphore; scratch is first written by run().
            semaphore = workspace_u8[:_XQA_SEMAPHORE_BYTES]
            snapshots.append((semaphore, semaphore.clone()))
            backend.plan(
                cum_seq_lens_q=dense.cum_seq_lens_q,
                block_tables=dense.block_tables,
                seq_lens=dense.seq_lens,
                num_heads=args.num_heads,
                head_dim_ckv=args.head_dim_ckv,
                head_dim_kpe=args.head_dim_kpe,
                page_size=args.page_size,
                sm_scale=args.sm_scale,
                q_data_type=args.q_data_type,
                kv_data_type=args.kv_data_type,
                enable_pdl=args.enable_pdl,
            )
            return backend
        except BaseException:
            for buffer, original in snapshots:
                buffer.copy_(original)
            raise

    def plan(
        self,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        enable_pdl: Optional[bool],
    ) -> None:
        for name, tensor in (
            ("cum_seq_lens_q", cum_seq_lens_q),
            ("block_tables", block_tables),
            ("seq_lens", seq_lens),
        ):
            check_shape_dtype_device(tensor, None, torch.int32, self.device, name)
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous for XQA MLA wrapper.")
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.numel() < 2:
            raise ValueError(
                "XQA MLA wrapper expects one-dimensional cum_seq_lens_q with "
                "at least two entries."
            )
        batch_size = cum_seq_lens_q.numel() - 1
        expected_cum_seq_lens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        if not torch.equal(cum_seq_lens_q, expected_cum_seq_lens_q):
            raise _BackendPlanUnsupportedError(
                "XQA MLA wrapper requires exactly one query token per request."
            )
        _check_block_tables_shape(block_tables, True)
        if block_tables.shape[0] != batch_size:
            raise ValueError(
                "XQA MLA block_tables batch dimension must match cum_seq_lens_q, "
                f"got {block_tables.shape[0]} and {batch_size}."
            )
        alignment = 128 // page_size
        if block_tables.shape[1] == 0 or block_tables.shape[1] % alignment != 0:
            raise ValueError(
                "XQA MLA block_tables width must be a positive multiple of "
                f"{alignment} for page_size={page_size}."
            )
        check_shape_dtype_device(
            seq_lens, (batch_size,), torch.int32, self.device, "seq_lens"
        )

        resolved_enable_pdl = (
            device_support_pdl(self.device) if enable_pdl is None else enable_pdl
        )
        if type(resolved_enable_pdl) is not bool:
            raise TypeError(
                "XQA MLA wrapper expects enable_pdl to be bool or None, got "
                f"{enable_pdl!r}."
            )
        if not self._float_workspace_buffer.is_contiguous():
            raise ValueError("workspace buffer must be contiguous for XQA MLA wrapper.")
        workspace_u8 = self._float_workspace_buffer.view(torch.uint8).flatten()
        if workspace_u8.numel() < _XQA_MIN_WORKSPACE_BYTES:
            raise _BackendPlanUnsupportedError(
                "XQA MLA wrapper workspace must contain at least 128 MiB, got "
                f"{workspace_u8.numel()} bytes."
            )

        module = get_xqa_module_mla(
            q_data_type,
            kv_data_type,
            page_size,
            head_dim_ckv + head_dim_kpe,
            num_heads,
            False,
        )
        semaphore = workspace_u8[:_XQA_SEMAPHORE_BYTES]
        scratch = workspace_u8[_XQA_SEMAPHORE_BYTES:]
        semaphore.zero_()

        self._module = module
        self._block_tables = block_tables
        self._seq_lens_2d = seq_lens.unsqueeze(1)
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._kv_lora_rank = head_dim_ckv
        self._qk_rope_head_dim = head_dim_kpe
        self._head_dim = head_dim_ckv + head_dim_kpe
        self._page_size = page_size
        self._q_dtype = q_data_type
        self._kv_dtype = kv_data_type
        self._bmm1_scale = sm_scale
        self._bmm2_scale = 1.0
        self._enable_pdl = resolved_enable_pdl
        self._sm_count = get_device_sm_count(self.device)
        self._max_seq_len = block_tables.shape[1] * page_size
        self._semaphore = semaphore
        self._scratch = scratch

    def prepare_for_tuning(self) -> None:
        # Other candidates may overwrite any part of the shared scratch buffer.
        # XQA semaphores must start at zero and stay private between launches.
        self._semaphore = torch.zeros_like(self._semaphore)

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
    ) -> torch.Tensor:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionXqaBackend.run() called before plan()."
            )
        if return_lse or lse is not None:
            raise ValueError("XQA MLA wrapper does not support LSE output.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with XQA backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with XQA backend."
            )
        if return_lse_base_on_e:
            raise ValueError("return_lse_base_on_e is not supported with XQA backend.")
        if o_scale is not None:
            raise ValueError("o_scale is not supported with XQA backend.")
        if ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / ckv_scale_arr / kpe_scale are not supported with XQA backend."
            )
        if sinks is not None:
            raise ValueError("sinks are not supported with XQA backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with XQA backend."
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
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionXqaBackend.run() called before plan()."
            )
        if out is None:
            raise ValueError("out must be provided for XQA planned backend runs.")
        _validate_scalar_bmm_scales(bmm1_scale, bmm2_scale)
        resolved_bmm1_scale = self._bmm1_scale if bmm1_scale is None else bmm1_scale
        resolved_bmm2_scale = self._bmm2_scale if bmm2_scale is None else bmm2_scale

        check_shape_dtype_device(
            query,
            (self._batch_size, self._num_heads, self._head_dim),
            self._q_dtype,
            self.device,
            "query",
        )
        if not query.is_contiguous():
            raise ValueError("query must be contiguous for XQA MLA wrapper.")
        check_shape_dtype_device(
            kv_cache,
            (kv_cache.shape[0], self._page_size, self._head_dim),
            self._kv_dtype,
            self.device,
            "kv_cache",
        )
        if not kv_cache.is_contiguous():
            raise ValueError("kv_cache must be contiguous for XQA MLA wrapper.")
        check_shape_dtype_device(
            out,
            (self._batch_size, self._num_heads, self._kv_lora_rank),
            torch.bfloat16,
            self.device,
            "out",
        )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for XQA MLA wrapper.")

        query_4d = query.view(self._batch_size, 1, self._num_heads, self._head_dim)
        kv_cache_4d = kv_cache.unsqueeze(2)
        return self._execute(
            query_4d,
            kv_cache_4d,
            self._block_tables,
            self._seq_lens_2d,
            out,
            float(resolved_bmm1_scale),
            float(resolved_bmm2_scale),
        )

    # Functional preparation. Metadata stays on the device; no wrapper plan or
    # device-to-host reads are needed for this invocation.

    @classmethod
    def from_functional(
        cls,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        block_tables: torch.Tensor,
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: Union[float, torch.Tensor],
        sinks: object,
        enable_pdl: Optional[bool],
    ) -> "_BatchMLAPagedAttentionXqaBackend":
        q_len_per_request = query.size(1)
        if q_len_per_request != 1:
            raise ValueError(
                f"XQA MLA only supports q_len_per_request == 1, got {q_len_per_request}"
            )
        if not is_sm12x_supported(query.device):
            raise ValueError(
                "XQA MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 12.9)"
            )
        if query.dtype != kv_cache.dtype or query.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
        ):
            raise ValueError(
                "XQA MLA supports (fp8, fp8) or (bfloat16, bfloat16) only, "
                f"got {query.dtype} and {kv_cache.dtype}"
            )
        if sinks is not None:
            raise ValueError("XQA MLA does not support sinks")
        page_size = kv_cache.shape[-2]
        kv_cache = _check_mla_query_kv_shape(
            query, kv_cache, kv_lora_rank, qk_rope_head_dim
        )
        _check_mla_dense_page_table_shape(
            block_tables, query.shape[0], page_size, True, True
        )
        num_heads = query.shape[-2]
        if num_heads != 128:
            raise ValueError(
                "XQA MLA only supports 128 query heads (head_group_ratio=128), "
                f"got {num_heads} query heads"
            )
        backend = cls(workspace_buffer)
        backend._is_planned = False
        backend._module = get_xqa_module_mla(
            query.dtype, kv_cache.dtype, page_size, query.shape[-1], num_heads, False
        )
        # Rebuild the singleton KV head axis to preserve the native strides.
        backend._kv_cache = kv_cache.squeeze(1).unsqueeze(2)
        backend._page_size = page_size
        backend._max_seq_len = block_tables.shape[-1] * page_size
        backend._bmm1_scale = bmm1_scale
        backend._bmm2_scale = bmm2_scale
        backend._enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        backend._sm_count = get_device_sm_count(query.device)
        workspace_u8 = workspace_buffer.view(torch.uint8)
        backend._semaphore = workspace_u8[:_XQA_SEMAPHORE_BYTES]
        backend._scratch = workspace_u8[_XQA_SEMAPHORE_BYTES:]
        return backend

    # Shared native execution. Tensor scales pass through unchanged, matching
    # the existing functional XQA convention; the wrapper accepts scalars only.

    def _execute(
        self, query, kv_cache, block_tables, seq_lens, out, bmm1_scale, bmm2_scale
    ) -> torch.Tensor:
        self._module.xqa_mla(
            self._sm_count,
            bmm1_scale,
            out,
            query,
            kv_cache,
            kv_cache,
            block_tables,
            self._max_seq_len,
            seq_lens,
            query.shape[0],
            bmm2_scale,
            self._semaphore,
            self._scratch,
            self._enable_pdl,
        )
        return out

    # Autotuning support for functional execution and the current wrapper plan.

    def __hash__(self):
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile):
        if self._is_planned:
            widths = (self._kv_lora_rank, self._qk_rope_head_dim)
            query = _resolve_structural_mla_input(
                inputs[0], desired="packed", widths=widths, name="query"
            )
            kv_cache = _resolve_structural_mla_input(
                inputs[1], desired="packed", widths=widths, name="KV cache"
            )
            out = inputs[2]
            # Invalid contracts must still fail instead of silently removing a
            # candidate. Only a valid but unsupported layout is an admission miss.
            check_shape_dtype_device(
                query,
                (self._batch_size, self._num_heads, self._head_dim),
                self._q_dtype,
                self.device,
                "query",
            )
            check_shape_dtype_device(
                kv_cache,
                (kv_cache.shape[0], self._page_size, self._head_dim),
                self._kv_dtype,
                self.device,
                "kv_cache",
            )
            check_shape_dtype_device(
                out,
                (self._batch_size, self._num_heads, self._kv_lora_rank),
                torch.bfloat16,
                self.device,
                "out",
            )
            if not all(tensor.is_contiguous() for tensor in (query, kv_cache, out)):
                return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        if self._is_planned:
            return ("planned", self._planned_tuning_key)
        query, _, _, out = inputs[:4]
        scale_keys = tuple(
            (tuple(scale.shape), tuple(scale.stride()), scale.dtype)
            if isinstance(scale, torch.Tensor)
            else scale
            for scale in (self._bmm1_scale, self._bmm2_scale)
        )
        return (
            "functional",
            query.dtype,
            tuple(query.stride()),
            self._kv_cache.dtype,
            tuple(self._kv_cache.shape),
            tuple(self._kv_cache.stride()),
            out.dtype,
            tuple(out.stride()),
            self._page_size,
            self._max_seq_len,
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size(),
            self._enable_pdl,
            scale_keys,
        )

    def configure_tuning(self, *, cache_key: tuple, run_options: dict) -> None:
        if any(isinstance(value, torch.Tensor) for value in run_options.values()):
            raise ValueError(
                "XQA planned tuning options must contain only host values."
            )
        self._planned_tuning_key = cache_key
        self._planned_run_options = dict(run_options)

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        run_options=None,
        **kwargs,
    ):
        if tactic != -1:
            raise ValueError(f"Unsupported XQA MLA tactic: {tactic!r}.")
        if self._is_planned:
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
        bmm1_scale, bmm2_scale = (
            inputs[4:6] if len(inputs) > 4 else (self._bmm1_scale, self._bmm2_scale)
        )
        return self._execute(
            query,
            self._kv_cache,
            block_tables,
            seq_lens.unsqueeze(1),
            out,
            bmm1_scale,
            bmm2_scale,
        )
