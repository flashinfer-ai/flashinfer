"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import functools
import math
from typing import ClassVar, Optional, cast

import torch

from ....jit.mla import gen_mla_module
from ....utils import check_shape_dtype_device, get_compute_capability
from ._capabilities import MLAPlanCapabilities, plan_capability_rejection_reason
from .._planning import _MLAPlanArguments


def _get_compute_capability(device: torch.device):
    return get_compute_capability(device)


def _validate_cutlass_page_size(page_size: int) -> int:
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or page_size <= 0
        or page_size > 128
        or 128 % page_size != 0
    ):
        raise ValueError(
            "cutlass backend requires integer page_size with "
            f"0 < page_size <= 128 and 128 % page_size == 0, got {page_size!r}."
        )
    return 128 // page_size


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
            "Expected head dim 576 for q_nope_pe and ckv_kpe_cache, "
            f"got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = page_table.shape
    block_size = ckv_kpe_cache.shape[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, "
            f"got {B_q} and {B_block_table}"
        )
    if kv_len.shape[0] != B_q:
        raise ValueError(
            "kv_len must contain one entry per query/page-table row, "
            f"got {kv_len.shape[0]} entries for batch size {B_q}."
        )
    table_width_alignment = _validate_cutlass_page_size(block_size)
    if block_num % table_width_alignment != 0:
        raise ValueError(
            "Expected page_table width to align with cutlass page_size, got "
            f"{block_num=} and page_size={block_size}."
        )


@functools.cache
def get_mla_module():
    return gen_mla_module().build_and_load()


class _BatchMLAPagedAttentionCutlassBackend:
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="cutlass",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none", "per-tensor"}),
        scale_modes=frozenset({"default"}),
        requires_packed_query=True,
        requires_packed_kv_cache=True,
    )

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "cutlass"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCutlassBackend":
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise ValueError(reason)
        table_width_alignment = _validate_cutlass_page_size(args.page_size)
        dense = args.dense(table_width_alignment=table_width_alignment)
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
            output_dtype=args.output_dtype,
            output_scale=args.output_scale,
            use_profiler=args.use_profiler,
            batch_size=batch_size,
            kv_len=dense.seq_lens.to(args._float_workspace_buffer.device),
            page_table=dense.block_tables.to(args._float_workspace_buffer.device),
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
        output_dtype: torch.dtype,
        output_scale: str,
        use_profiler: bool,
        batch_size: int,
        kv_len: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        # ---------------------------------------------------------------------------
        # Validate the CUTLASS plan contract
        # ---------------------------------------------------------------------------
        if use_profiler:
            raise ValueError("use_profiler is not supported by the cutlass backend.")
        if causal:
            raise ValueError("causal=True is not supported by the cutlass backend.")
        if num_heads != 128:
            raise ValueError(
                f"Expected 128 heads for cutlass backend, got {num_heads}."
            )
        if head_dim_ckv != 512 or head_dim_kpe != 64:
            raise ValueError(
                "cutlass backend expects head_dim_ckv=512 and head_dim_kpe=64, "
                f"got {head_dim_ckv=} and {head_dim_kpe=}."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "cutlass backend expects q_data_type to be torch.float16 or "
                f"torch.bfloat16, got {q_data_type}."
            )
        if kv_data_type != q_data_type:
            raise ValueError(
                "cutlass backend expects kv_data_type to match q_data_type, "
                f"got {kv_data_type=} and {q_data_type=}."
            )
        if output_scale == "none":
            if output_dtype != q_data_type:
                raise ValueError(
                    "cutlass unscaled output_dtype must match q_data_type, got "
                    f"{output_dtype} and {q_data_type}."
                )
        elif output_scale == "per-tensor":
            if output_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(
                    "cutlass per-tensor output scaling requires an FP8 "
                    f"output_dtype, got {output_dtype}."
                )
        else:
            raise ValueError(f"unsupported cutlass output_scale {output_scale!r}.")
        expected_sm_scale = 1.0 / math.sqrt(128 + head_dim_kpe)
        if not math.isclose(sm_scale, expected_sm_scale, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(
                "cutlass backend uses a fixed MLA softmax scale of "
                f"{expected_sm_scale}, got {sm_scale}."
            )
        _validate_cutlass_page_size(page_size)
        major, minor = _get_compute_capability(self.device)
        if major not in (10, 11):
            raise ValueError(
                "cutlass backend supports only compute capability major versions "
                f"10 and 11, got SM{major}{minor}."
            )

        # ---------------------------------------------------------------------------
        # Publish the validated plan state
        # ---------------------------------------------------------------------------
        self._batch_size = batch_size
        self._page_size = page_size
        self._head_dim_ckv = head_dim_ckv
        self._head_dim_kpe = head_dim_kpe
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._output_dtype = output_dtype
        self._output_scale = output_scale
        self._kv_len = kv_len
        self._page_table = page_table
        self._cached_module = get_mla_module()
        self._empty_lse = torch.empty(0, dtype=torch.float32, device=self.device)

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
    ) -> torch.Tensor:
        # ---------------------------------------------------------------------------
        # Validate the run contract and resolve backend inputs
        # ---------------------------------------------------------------------------
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
        if ckv_scale is not None or kpe_scale is not None or ckv_scale_arr is not None:
            raise ValueError(
                "ckv_scale / kpe_scale / ckv_scale_arr are only supported with "
                "an fa2/fa3 backend and FP8 kv_data_type."
            )
        if (kv_len is None) != (page_table is None):
            raise ValueError(
                "run-time kv_len and page_table must both be omitted or both be provided."
            )
        kv_len = self._kv_len if kv_len is None else kv_len
        page_table = self._page_table if page_table is None else page_table
        q_nope_pe = cast(torch.Tensor, query)
        ckv_kpe_cache = cast(torch.Tensor, kv_cache)
        if q_nope_pe.dtype != self._q_data_type:
            raise ValueError(
                f"query dtype {q_nope_pe.dtype} does not match planned "
                f"q_data_type {self._q_data_type}."
            )
        if ckv_kpe_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"KV cache dtype {ckv_kpe_cache.dtype} does not match planned "
                f"kv_data_type {self._kv_data_type}."
            )
        for name, tensor in (("kv_len", kv_len), ("page_table", page_table)):
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must have dtype torch.int32.")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}.")
        _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
        if q_nope_pe.shape[0] != self._batch_size:
            raise ValueError(
                f"query batch size {q_nope_pe.shape[0]} does not match planned "
                f"batch size {self._batch_size}."
            )
        if ckv_kpe_cache.shape[1] != self._page_size:
            raise ValueError(
                f"KV cache page size {ckv_kpe_cache.shape[1]} does not match "
                f"planned page size {self._page_size}."
            )

        # ---------------------------------------------------------------------------
        # Prepare the output buffer and scale
        # ---------------------------------------------------------------------------
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
            if out.dtype != self._output_dtype:
                raise ValueError(
                    f"out dtype must match planned output_dtype "
                    f"{self._output_dtype}, got {out.dtype}"
                )
            check_shape_dtype_device(
                out,
                (*q_nope_pe.shape[:-1], self._head_dim_ckv),
                self._output_dtype,
                q_nope_pe.device,
                "out",
            )
        elif out is None:
            out = torch.empty(
                (*q_nope_pe.shape[:-1], self._head_dim_ckv),
                dtype=self._output_dtype,
                device=q_nope_pe.device,
            )
        else:
            check_shape_dtype_device(
                out,
                (*q_nope_pe.shape[:-1], self._head_dim_ckv),
                self._output_dtype,
                q_nope_pe.device,
                "out",
            )

        # ---------------------------------------------------------------------------
        # Launch the CUTLASS backend
        # ---------------------------------------------------------------------------
        self._cached_module.cutlass_mla_paged_attention(
            self._float_workspace_buffer,
            out,
            self._empty_lse,
            q_nope_pe,
            ckv_kpe_cache,
            kv_len,
            page_table,
            output_scale,
        )
        return out

    @classmethod
    def run_planless(
        cls,
        *,
        float_workspace_buffer: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        profiler_buffer: Optional[torch.Tensor],
        kv_len: torch.Tensor,
        page_table: torch.Tensor,
        o_scale: Optional[float],
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        num_heads: int,
        page_size: int,
    ) -> torch.Tensor:
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cutlass backend.")
        _validate_cutlass_page_size(page_size)
        if num_heads != 128:
            raise ValueError(
                f"Expected 128 heads for cutlass backend, got {num_heads}."
            )
        if query.dtype != q_data_type:
            raise ValueError(
                f"query dtype {query.dtype} does not match run q_data_type {q_data_type}."
            )
        if kv_cache.dtype != kv_data_type:
            raise ValueError(
                "KV cache dtype "
                f"{kv_cache.dtype} does not match run kv_data_type {kv_data_type}."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "cutlass backend expects q_data_type to be torch.float16 or "
                f"torch.bfloat16, got {q_data_type}."
            )
        if kv_data_type != q_data_type:
            raise ValueError(
                "cutlass backend expects kv_data_type to match q_data_type, "
                f"got {kv_data_type=} and {q_data_type=}."
            )
        major, minor = _get_compute_capability(float_workspace_buffer.device)
        if major not in (10, 11):
            raise ValueError(
                "cutlass backend supports only compute capability major versions "
                f"10 and 11, got SM{major}{minor}."
            )
        for name, tensor in (("kv_len", kv_len), ("page_table", page_table)):
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must have dtype torch.int32.")
            if tensor.device != float_workspace_buffer.device:
                raise ValueError(f"{name} must be on {float_workspace_buffer.device}.")
        _check_cutlass_shape(query, kv_cache, kv_len, page_table)
        if kv_cache.shape[1] != page_size:
            raise ValueError(
                f"KV cache page size {kv_cache.shape[1]} does not match run page size "
                f"{page_size}."
            )

        output_dtype = q_data_type
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
            if out.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(
                    "out must be an FP8 tensor when o_scale is provided, "
                    f"got {out.dtype}"
                )
            output_dtype = out.dtype
        elif out is not None:
            output_dtype = out.dtype
            if output_dtype != q_data_type:
                raise ValueError(
                    "cutlass unscaled output dtype must match q_data_type, got "
                    f"{output_dtype} and {q_data_type}."
                )

        output_shape = (*query.shape[:-1], 512)
        if out is None:
            out = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        else:
            check_shape_dtype_device(
                out, output_shape, output_dtype, query.device, "out"
            )

        empty_lse = torch.empty(
            0, dtype=torch.float32, device=float_workspace_buffer.device
        )
        get_mla_module().cutlass_mla_paged_attention(
            float_workspace_buffer,
            out,
            empty_lse,
            query,
            kv_cache,
            kv_len,
            page_table,
            output_scale,
        )
        return out


__all__ = [
    "_BatchMLAPagedAttentionCutlassBackend",
    "_check_cutlass_shape",
    "_validate_cutlass_page_size",
    "get_mla_module",
]
