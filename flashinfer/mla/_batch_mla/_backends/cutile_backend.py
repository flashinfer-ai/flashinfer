"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import functools
import math
from typing import ClassVar, Optional, cast

import torch

from ....attention.prims_ts._tensor_aliasing import (
    _validate_out_does_not_overlap_inputs,
)
from ....utils import get_compute_capability
from .._contracts import _are_adjacent_last_dim_views
from .._planning import _MLAPlanArguments
from ._capabilities import MLAPlanCapabilities, plan_capability_rejection_reason


_CUTILE_SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(10, 0), (10, 3), (12, 0), (12, 1)})


def _get_compute_capability(device: torch.device):
    if device.type != "cuda":
        raise ValueError(
            f"cutile backend requires a CUDA workspace device, got {device}."
        )
    return get_compute_capability(device)


@functools.cache
def get_cutile_mla_decode():
    """Load the cuda.tile kernel only after the cuTile plan is validated."""

    from ....attention.kernels.cutile.fmha_decode_bsr_cutile import (
        decode_mla_kv_paged_cutile,
    )

    return decode_mla_kv_paged_cutile


def _validate_cutile_page_size(page_size: int) -> None:
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or page_size < 2
        or page_size > 128
        or page_size & (page_size - 1)
    ):
        raise ValueError(
            "cutile backend requires a power-of-two integer page_size in "
            f"[2, 128], got {page_size!r}."
        )


def _validate_cutile_num_heads(num_heads: int) -> None:
    if (
        not isinstance(num_heads, int)
        or isinstance(num_heads, bool)
        or num_heads < 8
        or num_heads > 128
        or num_heads % 8 != 0
    ):
        raise ValueError(
            "cutile backend requires num_heads to be a multiple of 8 in [8, 128], "
            f"got {num_heads!r}."
        )


def _validate_compact_split_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    name: str,
) -> None:
    if left.is_contiguous() and right.is_contiguous():
        return
    if _are_adjacent_last_dim_views(left, right):
        return
    raise ValueError(
        f"cutile {name} tensors must be contiguous independent tensors or "
        "adjacent views of one contiguous packed tensor."
    )


def _unpack_split_pair(
    value: object,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if type(value) is not tuple:
        raise TypeError(f"cutile backend requires split {name} tensors.")
    if len(value) != 2:
        raise ValueError(f"cutile split {name} must contain exactly two tensors.")
    left, right = value
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError(f"cutile split {name} leaves must be torch.Tensor.")
    return left, right


def _validate_launch_tensor(
    tensor: object,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"cutile launch tensor {name} must be a torch.Tensor.")
    if tensor.layout != torch.strided:
        raise ValueError(f"cutile launch tensor {name} must have strided layout.")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"cutile launch tensor {name} must have shape {shape}, got "
            f"{tuple(tensor.shape)}."
        )
    if tensor.dtype != dtype:
        raise ValueError(
            f"cutile launch tensor {name} must have dtype {dtype}, got {tensor.dtype}."
        )
    if tensor.device != device:
        raise ValueError(
            f"cutile launch tensor {name} must be on workspace device {device}, "
            f"got {tensor.device}."
        )


def _validate_runtime_metadata(
    tensor: object,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout.")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32.")
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on workspace device {device}, got {tensor.device}."
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


class _BatchMLAPagedAttentionCutileBackend:
    _plan_capabilities: ClassVar[MLAPlanCapabilities] = MLAPlanCapabilities(
        backend_name="cutile",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined", "independent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default"}),
        requires_packed_query=False,
        requires_packed_kv_cache=False,
    )

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._backend = "cutile"
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionCutileBackend":
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise ValueError(reason)
        dense = args.native_dense()
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
            use_cuda_graph=args._use_cuda_graph,
            cum_seq_lens_q=dense.cum_seq_lens_q,
            max_q_len=dense.max_q_len,
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
        output_dtype: torch.dtype,
        output_scale: str,
        use_profiler: bool,
        use_cuda_graph: bool,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        kv_len: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        # -----------------------------------------------------------------------
        # Validate the cuTile plan contract before importing cuda.tile
        # -----------------------------------------------------------------------
        if use_profiler:
            raise ValueError("use_profiler is not supported by the cutile backend.")
        if causal:
            raise ValueError("causal=True is not supported by the cutile backend.")
        _validate_cutile_num_heads(num_heads)
        if head_dim_ckv != 512 or head_dim_kpe != 64:
            raise ValueError(
                "cutile backend expects head_dim_ckv=512 and head_dim_kpe=64, "
                f"got {head_dim_ckv=} and {head_dim_kpe=}."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "cutile backend expects q_data_type to be torch.float16 or "
                f"torch.bfloat16, got {q_data_type}."
            )
        if kv_data_type != q_data_type:
            raise ValueError(
                "cutile backend expects kv_data_type to match q_data_type, "
                f"got {kv_data_type=} and {q_data_type=}."
            )
        if output_dtype != q_data_type:
            raise ValueError(
                "cutile backend expects output_dtype to match q_data_type, "
                f"got {output_dtype=} and {q_data_type=}."
            )
        if output_scale != "none":
            raise ValueError(
                f"cutile backend does not support output_scale={output_scale!r}."
            )
        if not isinstance(sm_scale, (int, float)) or isinstance(sm_scale, bool):
            raise ValueError(
                f"sm_scale must be a finite positive number, got {sm_scale!r}."
            )
        resolved_sm_scale = float(sm_scale)
        if not math.isfinite(resolved_sm_scale) or resolved_sm_scale <= 0.0:
            raise ValueError(
                f"sm_scale must be a finite positive number, got {sm_scale!r}."
            )
        _validate_cutile_page_size(page_size)
        major, minor = _get_compute_capability(self.device)
        if (major, minor) not in _CUTILE_SUPPORTED_COMPUTE_CAPABILITIES:
            raise ValueError(
                "cutile backend supports only the validated Blackwell targets "
                f"SM100, SM103, SM120, and SM121, got SM{major}{minor}."
            )

        batch_size = cum_seq_lens_q.numel() - 1
        if batch_size <= 0:
            raise ValueError("cutile backend requires a positive batch size.")
        if max_q_len != 1:
            raise ValueError(
                "cutile backend supports decode plans with one query per request, "
                f"got max_q_len={max_q_len}."
            )
        expected_cum_seq_lens_q = torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device=cum_seq_lens_q.device,
        )
        if not torch.equal(cum_seq_lens_q, expected_cum_seq_lens_q):
            raise ValueError(
                "cutile backend requires cum_seq_lens_q=[0, 1, ..., batch_size]."
            )
        if tuple(kv_len.shape) != (batch_size,):
            raise ValueError(
                f"cutile kv_len must have shape {(batch_size,)}, got "
                f"{tuple(kv_len.shape)}."
            )
        if page_table.ndim != 2 or page_table.shape[0] != batch_size:
            raise ValueError(
                "cutile page_table must be rank 2 with one row per request, got "
                f"shape {tuple(page_table.shape)}."
            )
        if page_table.shape[1] <= 0:
            raise ValueError("cutile page_table must have positive width.")

        # Stage canonical metadata during planning. Device inputs retain their
        # identity, which lets callers mutate values in place before graph replay.
        planned_kv_len = kv_len.to(device=self.device, non_blocking=True)
        planned_page_table = page_table.to(device=self.device, non_blocking=True)
        decode_mla_kv_paged_cutile = get_cutile_mla_decode()

        # -----------------------------------------------------------------------
        # Publish only fully validated plan state
        # -----------------------------------------------------------------------
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._page_size = page_size
        self._page_table_width = page_table.shape[1]
        self._head_dim_ckv = head_dim_ckv
        self._head_dim_kpe = head_dim_kpe
        self._sm_scale = resolved_sm_scale
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._output_dtype = output_dtype
        self._use_cuda_graph = use_cuda_graph
        self._kv_len = planned_kv_len
        self._page_table = planned_page_table
        self._decode_mla_kv_paged_cutile = decode_mla_kv_paged_cutile

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
        # -----------------------------------------------------------------------
        # Validate the run contract and resolve planned metadata
        # -----------------------------------------------------------------------
        if return_lse:
            raise ValueError("return_lse is not supported with cutile backend.")
        if lse is not None:
            raise ValueError("lse is not supported with cutile backend.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with cutile backend.")
        if return_lse_base_on_e:
            raise ValueError(
                "return_lse_base_on_e is not supported with cutile backend."
            )
        if o_scale is not None:
            raise ValueError("o_scale is not supported with cutile backend.")
        if ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / ckv_scale_arr / kpe_scale are not supported with "
                "cutile backend."
            )
        if (kv_len is None) != (page_table is None):
            raise ValueError(
                "run-time kv_len and page_table must both be omitted or both be provided."
            )
        kv_len = self._kv_len if kv_len is None else kv_len
        page_table = self._page_table if page_table is None else page_table
        _validate_runtime_metadata(
            kv_len,
            name="kv_len",
            device=self.device,
            shape=(self._batch_size,),
        )
        _validate_runtime_metadata(
            page_table,
            name="page_table",
            device=self.device,
            shape=(self._batch_size, self._page_table_width),
        )

        q_nope, q_pe = _unpack_split_pair(query, name="query")
        ckv_cache, kpe_cache = _unpack_split_pair(kv_cache, name="KV cache")
        query_prefix = (self._batch_size, self._num_heads)
        _validate_launch_tensor(
            q_nope,
            name="q_nope",
            device=self.device,
            dtype=self._q_data_type,
            shape=(*query_prefix, self._head_dim_ckv),
        )
        _validate_launch_tensor(
            q_pe,
            name="q_pe",
            device=self.device,
            dtype=self._q_data_type,
            shape=(*query_prefix, self._head_dim_kpe),
        )
        if ckv_cache.ndim != 3:
            raise ValueError(
                f"cutile launch tensor ckv_cache must have rank 3, got "
                f"{ckv_cache.ndim}."
            )
        if ckv_cache.shape[0] <= 0:
            raise ValueError("cutile KV cache must contain at least one page.")
        kv_prefix = (ckv_cache.shape[0], self._page_size)
        _validate_launch_tensor(
            ckv_cache,
            name="ckv_cache",
            device=self.device,
            dtype=self._kv_data_type,
            shape=(*kv_prefix, self._head_dim_ckv),
        )
        _validate_launch_tensor(
            kpe_cache,
            name="kpe_cache",
            device=self.device,
            dtype=self._kv_data_type,
            shape=(*kv_prefix, self._head_dim_kpe),
        )
        _validate_compact_split_pair(q_nope, q_pe, name="query")
        _validate_compact_split_pair(ckv_cache, kpe_cache, name="KV-cache")

        # -----------------------------------------------------------------------
        # Prepare and validate the output without converting launch inputs
        # -----------------------------------------------------------------------
        output_shape = (*query_prefix, self._head_dim_ckv)
        if out is None:
            out = torch.empty(
                output_shape,
                dtype=self._output_dtype,
                device=self.device,
            )
        else:
            _validate_launch_tensor(
                out,
                name="out",
                device=self.device,
                dtype=self._output_dtype,
                shape=output_shape,
            )
            if not out.is_contiguous():
                raise ValueError("cutile launch tensor out must be contiguous.")
        _validate_out_does_not_overlap_inputs(
            out,
            ("q_nope", q_nope),
            ("q_pe", q_pe),
            ("ckv_cache", ckv_cache),
            ("kpe_cache", kpe_cache),
            ("kv_len", kv_len),
            ("page_table", page_table),
            ("float_workspace_buffer", self._float_workspace_buffer),
        )

        # -----------------------------------------------------------------------
        # Launch the lazily loaded cuTile backend
        # -----------------------------------------------------------------------
        launch_args = (
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            kv_len,
            page_table,
            self._sm_scale,
            1.0,
        )
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._decode_mla_kv_paged_cutile(
                    *launch_args,
                    max_seq_len=-1,
                    outputs=out,
                )
        else:
            # This path is reachable only by tests that replace the architecture
            # probe and kernel loader with CPU fakes.
            self._decode_mla_kv_paged_cutile(
                *launch_args,
                max_seq_len=-1,
                outputs=out,
            )
        return cast(torch.Tensor, out)


__all__ = [
    "_BatchMLAPagedAttentionCutileBackend",
    "_get_compute_capability",
    "_validate_cutile_num_heads",
    "_validate_cutile_page_size",
    "get_cutile_mla_decode",
]
