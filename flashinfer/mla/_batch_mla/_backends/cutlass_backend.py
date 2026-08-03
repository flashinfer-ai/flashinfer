"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import functools
import logging
import math
from dataclasses import replace
from typing import Any, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import TunableRunner
from flashinfer.jit.mla import gen_mla_module
from ._capabilities import (
    BACKEND_OPERATIONAL_PLAN_FIELDS,
    MLAPlanCapabilities,
    validate_plan_capabilities,
)
from .._planning import (
    _MLAPlanArguments,
    _validate_dense_metadata,
)
from .._contracts import (
    MLAKVCache,
    MLAQuery,
    _FunctionalMLARequest,
    _concat_adjacent_views_or_cat,
    _split_mla_value_objects,
)
from flashinfer.utils import check_shape_dtype_device, get_compute_capability


logger = logging.getLogger(__name__)


def _validate_cutlass_metadata(
    kv_len: torch.Tensor,
    page_table: torch.Tensor,
    *,
    batch_size: int,
    page_size: int,
    device: torch.device,
) -> None:
    if kv_len.ndim != 1:
        raise ValueError(f"kv_len must be rank 1, got rank {kv_len.ndim}.")
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be rank 2, got rank {page_table.ndim}.")
    if kv_len.shape[0] != batch_size or page_table.shape[0] != batch_size:
        raise ValueError(
            "kv_len and page_table batch dimension must match the planned "
            f"batch size {batch_size}, got {kv_len.shape[0]} and "
            f"{page_table.shape[0]}."
        )
    if kv_len.dtype != torch.int32:
        raise ValueError(f"kv_len must have dtype torch.int32, got {kv_len.dtype}.")
    if page_table.dtype != torch.int32:
        raise ValueError(
            f"page_table must have dtype torch.int32, got {page_table.dtype}."
        )
    if kv_len.device != device or page_table.device != device:
        raise ValueError(
            "kv_len and page_table must be on the same device as the CUTLASS "
            f"backend ({device}), got {kv_len.device} and {page_table.device}."
        )
    if not kv_len.is_contiguous():
        raise ValueError("kv_len must be contiguous for the CUTLASS launcher.")
    if not page_table.is_contiguous():
        raise ValueError("page_table must be contiguous for the CUTLASS launcher.")
    required_width_multiple = 128 // page_size
    if page_table.shape[1] == 0 or page_table.shape[1] % required_width_multiple != 0:
        raise ValueError(
            "page_table width must be a positive multiple of "
            f"{required_width_multiple} for page_size={page_size}, got "
            f"{page_table.shape[1]}."
        )
    kv_len_host = kv_len.to(device="cpu", dtype=torch.int64)
    if torch.any(kv_len_host < 0).item():
        raise ValueError("kv_len must be nonnegative.")
    live_pages = torch.div(
        kv_len_host + page_size - 1,
        page_size,
        rounding_mode="floor",
    )
    if live_pages.numel() and int(live_pages.max().item()) > page_table.shape[1]:
        raise ValueError(
            "page_table width is smaller than the live CUTLASS page count."
        )


def _is_same_tensor_view(actual: torch.Tensor, planned: torch.Tensor) -> bool:
    return (
        actual.shape == planned.shape
        and actual.dtype == planned.dtype
        and actual.device == planned.device
        and actual.stride() == planned.stride()
        and actual.storage_offset() == planned.storage_offset()
        and actual.data_ptr() == planned.data_ptr()
    )


def _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table):
    if q_nope_pe.ndim != 3:
        raise ValueError(f"Expected q_nope_pe.ndim == 3, got {q_nope_pe.ndim}")
    if ckv_kpe_cache.ndim != 3:
        raise ValueError(f"Expected ckv_kpe_cache.ndim == 3, got {ckv_kpe_cache.ndim}")
    if kv_len.ndim != 1:
        raise ValueError(f"Expected kv_len.ndim == 1, got {kv_len.ndim}")
    if page_table.ndim != 2:
        raise ValueError(f"Expected page_table.ndim == 2, got {page_table.ndim}")
    B_q, H, D_q = q_nope_pe.shape
    D_ckv = ckv_kpe_cache.shape[2]
    if H != 128:
        raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for q_nope_pe and ckv_kpe_cache, got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = page_table.shape
    block_size = ckv_kpe_cache.shape[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, got {B_q} and {B_block_table}"
        )
    if block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
        )


@functools.cache
def get_mla_module():
    return gen_mla_module().build_and_load()


class _BatchMLAPagedAttentionCutlassBackend:
    """CUTLASS MLA backend with plan-preferred launch metadata.

    The public query and cache inputs remain split into NoPE/PE tensors and are
    concatenated internally for the launcher. ``kv_len`` and ``page_table`` may
    be captured by :meth:`plan`; :meth:`run` then uses those planned tensors
    unless callers provide cheap-verified aliases of the same tensor views.
    """

    _plan_capabilities = MLAPlanCapabilities(
        backend_name="cutlass",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined", "adjacent-split", "independent-split"}),
        output_scales=frozenset({"none", "per-tensor"}),
        scale_modes=frozenset({"default"}),
    )
    _backend_operational_plan_fields = BACKEND_OPERATIONAL_PLAN_FIELDS

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "cutlass"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCutlassBackend":
        validate_plan_capabilities(args, cls._plan_capabilities)
        output_dtype = args.output_dtype
        output_scale = args.output_scale
        if output_scale == "none" and output_dtype != args.q_data_type:
            raise _BackendPlanUnsupportedError(
                "cutlass backend requires q_data_type output without o_scale."
            )
        if output_scale == "per-tensor" and output_dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            raise _BackendPlanUnsupportedError(
                "cutlass backend requires FP8 output for the per-tensor output contract."
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
                "cutlass dense metadata requires page_size to divide 128, "
                f"got {args.page_size}."
            )
        dense = args.dense(table_width_alignment=128 // args.page_size)
        batch_size = dense.cum_seq_lens_q.shape[0] - 1
        backend = cls(args._float_workspace_buffer)
        backend.plan(
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            page_size=args.page_size,
            causal=args.causal,
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            use_profiler=args.use_profiler,
            batch_size=batch_size,
            kv_len=dense.seq_lens,
            page_table=dense.block_tables,
        )
        return backend

    def plan(
        self,
        *,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool,
        batch_size: int,
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
    ) -> None:
        if use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the cutlass backend."
            )
        if causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the cutlass backend."
            )
        if num_heads != 128:
            raise _BackendPlanUnsupportedError(
                f"Expected 128 heads for cutlass backend, got {num_heads}."
            )
        if head_dim_ckv != 512 or head_dim_kpe != 64:
            raise _BackendPlanUnsupportedError(
                "cutlass backend expects head_dim_ckv=512 and head_dim_kpe=64, "
                f"got {head_dim_ckv=} and {head_dim_kpe=}."
            )
        if page_size <= 0 or page_size > 128 or 128 % page_size != 0:
            raise _BackendPlanUnsupportedError(
                "cutlass backend expects page_size to be a positive divisor of "
                f"128 no larger than 128, got {page_size}."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise _BackendPlanUnsupportedError(
                "cutlass backend expects q_data_type to be torch.float16 or "
                f"torch.bfloat16, got {q_data_type}."
            )
        if kv_data_type != q_data_type:
            raise _BackendPlanUnsupportedError(
                "cutlass backend expects kv_data_type to match q_data_type, "
                f"got {kv_data_type=} and {q_data_type=}."
            )
        expected_sm_scale = 1.0 / math.sqrt(128 + head_dim_kpe)
        if not math.isclose(sm_scale, expected_sm_scale, rel_tol=1e-5, abs_tol=1e-8):
            raise _BackendPlanUnsupportedError(
                "cutlass backend uses a fixed MLA softmax scale of "
                f"{expected_sm_scale}, got {sm_scale}."
            )
        if (kv_len is None) != (page_table is None):
            raise ValueError("kv_len and page_table must be provided together.")
        if kv_len is not None and page_table is not None:
            _validate_cutlass_metadata(
                kv_len,
                page_table,
                batch_size=batch_size,
                page_size=page_size,
                device=self.device,
            )
        try:
            major, minor = get_compute_capability(self.device)
        except ValueError as err:
            raise _BackendPlanUnsupportedError(
                "cutlass backend requires a CUDA device with compute capability "
                f"major version 10 or 11, got {self.device}."
            ) from err
        if major not in (10, 11):
            raise _BackendPlanUnsupportedError(
                "cutlass backend supports only compute capability major versions "
                f"10 and 11, got SM{major}{minor}."
            )
        self._batch_size = batch_size
        self._page_size = page_size
        self._head_dim_ckv = head_dim_ckv
        self._kv_len = kv_len
        self._page_table = page_table
        self._cached_module = get_mla_module()

    def _resolve_metadata(
        self,
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (kv_len is None) != (page_table is None):
            raise ValueError(
                "run-time kv_len and page_table must both be omitted or both be provided."
            )

        if self._kv_len is not None and self._page_table is not None:
            if kv_len is None and page_table is None:
                return self._kv_len, self._page_table
            if not _is_same_tensor_view(kv_len, self._kv_len):
                raise ValueError(
                    "run-time kv_len must be the same tensor view as planned kv_len."
                )
            if not _is_same_tensor_view(page_table, self._page_table):
                raise ValueError(
                    "run-time page_table must be the same tensor view as planned "
                    "page_table."
                )
            return self._kv_len, self._page_table

        if kv_len is None or page_table is None:
            raise ValueError(
                "kv_len and page_table are required at run time when they were "
                "not provided to plan()."
            )
        _validate_cutlass_metadata(
            kv_len,
            page_table,
            batch_size=self._batch_size,
            page_size=self._page_size,
            device=self.device,
        )
        logger.debug(
            "CUTLASS MLA compatibility path is using run-time-only kv_len and "
            "page_table metadata."
        )
        return kv_len, page_table

    @staticmethod
    def _validate_wrapper_run_options(
        *,
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> None:
        if sinks is not None:
            raise ValueError("sinks are not supported by the cutlass wrapper backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported by the "
                "cutlass wrapper backend."
            )
        if bmm1_scale is not None:
            raise ValueError(
                "bmm1_scale is not supported by the cutlass wrapper backend."
            )
        if bmm2_scale is not None:
            raise ValueError(
                "bmm2_scale is not supported by the cutlass wrapper backend."
            )
        if return_lse:
            raise ValueError("return_lse is not supported with cutlass backend.")
        if lse is not None:
            raise ValueError("lse is not supported with cutlass backend.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cutlass backend.")
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with cutlass backend."
            )
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are only supported with the fa3 backend "
                "and FP8 kv_data_type."
            )

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
        self._validate_wrapper_run_options(
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=ckv_scale,
            kpe_scale=kpe_scale,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        packed_query = query.require_packed()
        kv_cache = kv.packed_or_adjacent()
        if kv_cache is None:
            ckv_cache, kpe_cache = kv.split_views(None)
            kv_cache = _concat_adjacent_views_or_cat(ckv_cache, kpe_cache)
        return self.run(
            query=packed_query,
            kv_cache=kv_cache,
            out=out,
            kv_len=kv_len,
            page_table=page_table,
            o_scale=o_scale,
        )

    @classmethod
    def run_unplanned_from_wrapper(
        cls,
        float_workspace_buffer: torch.Tensor,
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
    ) -> torch.Tensor:
        """Run the deprecated explicit-CUTLASS compatibility path without plan()."""
        cls._validate_wrapper_run_options(
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=ckv_scale,
            kpe_scale=kpe_scale,
            sinks=sinks,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        q_nope, q_pe, ckv_cache, kpe_cache = _split_mla_value_objects(query, kv, None)
        packed_query = query.require_packed()
        kv_cache = kv.packed_or_adjacent()
        backend = cls(float_workspace_buffer)
        backend.plan(
            num_heads=q_nope.shape[1],
            head_dim_ckv=q_nope.shape[2],
            head_dim_kpe=q_pe.shape[2],
            page_size=ckv_cache.shape[1],
            causal=False,
            sm_scale=1.0 / math.sqrt(128 + q_pe.shape[2]),
            q_data_type=q_nope.dtype,
            kv_data_type=ckv_cache.dtype,
            use_profiler=False,
            batch_size=q_nope.shape[0],
            kv_len=None,
            page_table=None,
        )
        return backend.run(
            query=packed_query,
            kv_cache=(
                _concat_adjacent_views_or_cat(ckv_cache, kpe_cache)
                if kv_cache is None
                else kv_cache
            ),
            out=out,
            kv_len=kv_len,
            page_table=page_table,
            o_scale=o_scale,
        )

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        o_scale: Optional[float],
    ) -> torch.Tensor:
        """Launch using planned metadata, or compatible run-time metadata."""
        if not hasattr(self, "_cached_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionCutlassBackend.run() called before plan()."
            )

        kv_len, page_table = self._resolve_metadata(kv_len, page_table)
        out_shape = query.shape[:-1] + (self._head_dim_ckv,)

        output_scale = 1.0
        if o_scale is not None:
            output_scale = float(o_scale)
            if not math.isfinite(output_scale) or output_scale <= 0.0:
                raise ValueError(
                    f"o_scale must be a finite positive value, got {o_scale}"
                )
            if out is None:
                raise ValueError(
                    "out tensor must be provided when o_scale is used for FP8 output."
                )
            if out.dtype not in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                raise ValueError(
                    f"out must be an FP8 tensor when o_scale is provided, got {out.dtype}"
                )
            check_shape_dtype_device(
                out,
                out_shape,
                None,
                query.device,
                "out",
            )
        elif out is None:
            out = torch.empty(
                out_shape,
                dtype=query.dtype,
                device=query.device,
            )
        else:
            check_shape_dtype_device(
                out,
                out_shape,
                query.dtype,
                query.device,
                "out",
            )
        _check_cutlass_shape(query, kv_cache, kv_len, page_table)
        lse = torch.empty(0, dtype=torch.float32, device=self.device)
        self._cached_module.cutlass_mla_paged_attention(
            self._float_workspace_buffer,
            out,
            lse,
            query,
            kv_cache,
            kv_len,
            page_table,
            output_scale,
        )
        return out


class CutlassMlaRunner(TunableRunner):
    """Direct functional runner for the fixed-shape CUTLASS MLA kernel."""

    name = "cutlass"

    def __init__(self, request: _FunctionalMLARequest) -> None:
        self.request = request
        self._validate_functional_options(request)
        initial_out = request.out
        if initial_out is None:
            initial_out = torch.empty(
                request.query.shape[:-1] + (request.kv_lora_rank,),
                dtype=request.query.dtype,
                device=request.query.device,
            )
        self._normalize_request(replace(request, out=initial_out))
        self._inputs = [
            request.query,
            request.block_tables,
            request.seq_lens,
            initial_out,
        ]

    @staticmethod
    def _validate_functional_options(request: _FunctionalMLARequest) -> None:
        if request.sparse_mla_top_k:
            raise ValueError("cutlass MLA does not support sparse_mla_top_k.")
        if not request.uses_shared_paged_kv_idx:
            raise ValueError("cutlass MLA requires dense shared page-table metadata.")
        if request.cum_seq_lens_q is not None:
            raise ValueError("cutlass MLA does not support ragged queries.")
        if request.return_lse or request.lse is not None:
            raise ValueError("cutlass MLA does not support LSE output.")
        if isinstance(request.bmm1_scale, torch.Tensor) or isinstance(
            request.bmm2_scale, torch.Tensor
        ):
            raise ValueError("cutlass MLA does not support tensor scales.")
        try:
            bmm1_scale = float(request.bmm1_scale)
            bmm2_scale = float(request.bmm2_scale)
        except (TypeError, ValueError) as error:
            raise ValueError("cutlass MLA scales must be scalar floats.") from error
        if not math.isfinite(bmm1_scale) or not math.isfinite(bmm2_scale):
            raise ValueError("cutlass MLA scales must be finite.")
        if bmm2_scale != 1.0:
            raise ValueError("cutlass MLA requires bmm2_scale == 1.0.")
        if request.sinks is not None:
            raise ValueError("cutlass MLA does not support sinks.")
        if request.skip_softmax_threshold_scale_factor is not None:
            raise ValueError("cutlass MLA does not support skip-softmax.")
        if request.enable_pdl is not None:
            raise ValueError("cutlass MLA does not support enable_pdl.")
        if request.is_var_seq is not True:
            raise ValueError("cutlass MLA requires is_var_seq=True.")
        if request.multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by trtllm-gen."
            )

    @staticmethod
    def _normalize_request(
        request: _FunctionalMLARequest,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        query = request.query
        kv_cache = request.kv_cache
        if not isinstance(query, torch.Tensor) or query.ndim != 4:
            raise ValueError(
                "cutlass MLA requires a rank-4 dense query with q_len == 1."
            )
        if query.shape[1] != 1:
            raise ValueError("cutlass MLA requires q_len == 1.")
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 3:
            raise ValueError("cutlass MLA requires a rank-3 packed kv_cache.")
        if query.device != kv_cache.device:
            raise ValueError("query and kv_cache must be on the same device.")
        if query.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ) or kv_cache.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            raise ValueError("functional cutlass MLA does not support FP8 query or KV.")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("cutlass MLA requires float16 or bfloat16 query and KV.")
        if query.dtype != kv_cache.dtype:
            raise ValueError("query and kv_cache must have matching dtype.")
        if (
            not isinstance(request.kv_lora_rank, int)
            or isinstance(request.kv_lora_rank, bool)
            or not isinstance(request.qk_rope_head_dim, int)
            or isinstance(request.qk_rope_head_dim, bool)
            or request.kv_lora_rank <= 0
            or request.qk_rope_head_dim <= 0
        ):
            raise ValueError("kv_lora_rank and qk_rope_head_dim must be positive ints.")
        packed_head_dim = request.kv_lora_rank + request.qk_rope_head_dim
        if query.shape[-1] != packed_head_dim or kv_cache.shape[-1] != packed_head_dim:
            raise ValueError(
                "query and kv_cache must use the packed MLA head dimension."
            )
        if request.seq_lens is None:
            raise ValueError("seq_lens is required for cutlass MLA.")
        page_size = kv_cache.shape[1]
        if page_size <= 0:
            raise ValueError("kv_cache page_size must be positive.")
        if page_size > 128 or 128 % page_size != 0:
            raise ValueError("cutlass MLA requires page_size to divide 128.")
        if not isinstance(request.workspace_buffer, torch.Tensor):
            raise ValueError("workspace_buffer must be a torch.Tensor.")
        if request.workspace_buffer.device != query.device:
            raise ValueError("workspace_buffer must be on the query device.")
        cumulative_q = torch.arange(
            query.shape[0] + 1, dtype=torch.int32, device=query.device
        )
        dense = _validate_dense_metadata(
            cum_seq_lens_q=cumulative_q,
            block_tables=request.block_tables,
            seq_lens=request.seq_lens,
            max_q_len=1,
            page_size=page_size,
            device=query.device,
            table_width_alignment=128 // page_size,
        )
        _validate_cutlass_metadata(
            dense.seq_lens,
            dense.block_tables,
            batch_size=query.shape[0],
            page_size=page_size,
            device=query.device,
        )
        expected_out_shape = query.shape[:-1] + (request.kv_lora_rank,)
        out = request.out
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != expected_out_shape
            or out.dtype != query.dtype
            or out.device != query.device
        ):
            raise ValueError(
                "out must match the functional CUTLASS output shape, dtype, and device."
            )
        _check_cutlass_shape(query[:, 0], kv_cache, dense.seq_lens, dense.block_tables)
        return (
            query[:, 0],
            kv_cache,
            dense.seq_lens,
            dense.block_tables,
            float(request.bmm1_scale),
        )

    @property
    def inputs(self) -> list[torch.Tensor]:
        return self._inputs

    def get_valid_tactics(self, inputs, profile) -> list[int]:
        del inputs, profile
        return [-1]

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del do_preparation, kwargs
        if tactic != -1:
            raise ValueError(f"cutlass MLA only supports tactic -1, got {tactic!r}.")
        if len(inputs) != 4:
            raise ValueError("cutlass MLA runner expects four dynamic inputs.")
        query, block_tables, seq_lens, out = inputs
        request = replace(
            self.request,
            query=query,
            block_tables=block_tables,
            seq_lens=seq_lens,
            out=out,
        )
        self._validate_functional_options(request)
        packed_query, kv_cache, kv_len, page_table, sm_scale = self._normalize_request(
            request
        )
        lse = torch.empty(0, dtype=torch.float32, device=query.device)
        get_mla_module().cutlass_mla_paged_attention(
            request.workspace_buffer,
            out[:, 0],
            lse,
            packed_query,
            kv_cache,
            kv_len,
            page_table,
            sm_scale,
        )
        return out
