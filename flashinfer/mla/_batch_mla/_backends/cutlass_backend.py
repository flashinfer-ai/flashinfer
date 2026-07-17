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
from typing import Optional

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.jit.mla import gen_mla_module
from .._planning import (
    _CSRPlanMetadata,
    _DensePlanMetadata,
    _max_q_len,
)
from flashinfer.utils import check_shape_dtype_device, get_compute_capability

from ._layout import (
    _concat_adjacent_views_or_cat,
)


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


def _validate_cutlass_plan_metadata(
    kv_len: torch.Tensor,
    page_table: torch.Tensor,
    *,
    csr: _CSRPlanMetadata,
    page_size: int,
    device: torch.device,
) -> _DensePlanMetadata:
    """Validate legacy CUTLASS metadata and its logical equivalence to CSR."""
    batch_size = csr.qo_indptr.numel() - 1
    _validate_cutlass_metadata(
        kv_len,
        page_table,
        batch_size=batch_size,
        page_size=page_size,
        device=device,
    )
    if not torch.equal(kv_len, csr.kv_len_arr):
        raise ValueError("CUTLASS kv_len must equal CSR kv_len_arr.")

    kv_indptr_host = csr.kv_indptr.to(device="cpu", dtype=torch.int64)
    for row in range(batch_size):
        start = int(kv_indptr_host[row].item())
        end = int(kv_indptr_host[row + 1].item())
        if not torch.equal(
            page_table[row, : end - start], csr.kv_indices[start:end]
        ):
            raise ValueError(
                "CUTLASS live page_table entries must equal CSR page lists."
            )

    return _DensePlanMetadata(
        cum_seq_lens_q=csr.qo_indptr,
        block_tables=page_table,
        seq_lens=kv_len,
        max_q_len=_max_q_len(csr.qo_indptr),
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

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "cutlass"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

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

    def run(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
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
            check_shape_dtype_device(out, q_nope.shape, None, q_nope.device, "out")
        elif out is None:
            out = torch.empty_like(q_nope)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )
        q_nope_pe = _concat_adjacent_views_or_cat(q_nope, q_pe)
        ckv_kpe_cache = _concat_adjacent_views_or_cat(ckv_cache, kpe_cache)
        _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
        lse = torch.empty(0, dtype=torch.float32, device=self.device)
        self._cached_module.cutlass_mla_paged_attention(
            self._float_workspace_buffer,
            out,
            lse,
            q_nope_pe,
            ckv_kpe_cache,
            kv_len,
            page_table,
            output_scale,
        )
        return out
