"""Compressed sparse MLA integration through the shared sparse-MLA core."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import torch

from .api import (
    _get_mla_output_view,
    _use_sm120_sparse_mla,
    _validate_tensor_storage_bounds,
)
from .compressed_config import (
    compressed_sparse_mla_split_chunks_for_contract,
)
from .compressed_reference import (
    COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
    COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_HEAD_DIM,
    compressed_sparse_mla_page_nbytes,
)


_LN2 = math.log(2.0)


def _should_use_sm121_single_pass_decode(
    *,
    rows: int,
    heads: int,
    swa_width: int,
    indexed_width: int,
    swa_page_size: int,
    indexed_page_size: int | None,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the 32-head final-output kernel for Spark's one-wave decode."""

    if compute_capability is None:
        if not torch.cuda.is_available():
            return False
        compute_capability = tuple(torch.cuda.get_device_capability())
    if compute_capability != (12, 1):
        return False
    if rows < 16 or heads != 32 or int(swa_page_size) != 64:
        return False
    if indexed_width and int(indexed_page_size or 0) != 64:
        return False
    chunks = (int(swa_width) + 63) // 64 + (int(indexed_width) + 63) // 64
    return chunks <= 10


def compressed_sparse_mla_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor | None = None,
    swa_topk_lengths: torch.Tensor | None = None,
    binding=None,
    sm_scale: float,
    swa_page_size: int = COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    indexed_k_cache: torch.Tensor | None = None,
    indexed_indices: torch.Tensor | None = None,
    indexed_topk_lengths: torch.Tensor | None = None,
    indexed_page_size: int | None = None,
    indexed_page_table: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    expected_num_q_heads: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    backend: str | None = None,
    out: torch.Tensor | None = None,
    cache_format: Literal["deepseek_v4", "deepseek_v41"] | None = None,
    mapper_launcher: object | None = None,
    prepared=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run compressed sparse MLA decode directly from compressed KV pages.

    ``out``, when given, is the final O destination: the kernel/merge writes it
    directly (no workspace-to-caller copy). Must be a contiguous BF16
    [rows, heads, 512] tensor on q's device; anything else raises.
    """

    if lse_scale not in ("base2", "natural"):
        raise ValueError(f"lse_scale must be 'base2' or 'natural', got {lse_scale!r}")

    if binding is None:
        raise TypeError("compressed_sparse_mla_decode_forward requires binding")
    extras = [
        name
        for name, value in (
            ("q_all", q_all),
            ("swa_indices", swa_indices),
            ("swa_topk_lengths", swa_topk_lengths),
            ("indexed_indices", indexed_indices),
            ("indexed_topk_lengths", indexed_topk_lengths),
            ("indexed_page_table", indexed_page_table),
        )
        if value is not None
    ]
    if extras:
        raise ValueError(
            "compressed MLA binding owns q and index tensors; "
            f"do not also pass {', '.join(extras)}"
        )
    scratch = getattr(binding, "scratch", None)
    if scratch is None:
        raise TypeError("compressed MLA binding is missing scratch")
    planned_cache_format = getattr(scratch, "cache_format", "deepseek_v4")
    if cache_format is not None and cache_format != planned_cache_format:
        raise ValueError("cache_format contradicts the bound compressed MLA recipe")
    cache_format = planned_cache_format
    q_all = getattr(binding, "q")
    swa_indices = getattr(binding, "swa_indices")
    swa_topk_lengths = getattr(binding, "swa_lengths")
    indexed_indices = getattr(binding, "indexed_indices", None)
    indexed_topk_lengths = getattr(binding, "indexed_lengths", None)
    indexed_page_table = getattr(binding, "indexed_page_table", None)

    q3 = _normalize_compressed_q(q_all)
    rows, heads, _ = q3.shape
    if expected_num_q_heads is not None and heads != int(expected_num_q_heads):
        raise ValueError(
            f"q_all local heads must match expected_num_q_heads={int(expected_num_q_heads)}, got {heads}"
        )
    if int(q3.shape[-1]) != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed_sparse_mla_decode_forward is DSV4 (q_head_dim="
            f"{COMPRESSED_SPARSE_MLA_HEAD_DIM}) by construction; got q_head_dim="
            f"{int(q3.shape[-1])}"
        )
    if not _use_sm120_sparse_mla(backend=backend, device=q3.device):
        raise RuntimeError(
            "compressed sparse MLA requires the active SM120 sparse MLA kernel path; "
            "legacy compressed sparse MLA kernels have been retired"
        )
    swa_indices_2d = _normalize_index_matrix(swa_indices, name="swa_indices")
    swa_k_cache = _compressed_sparse_mla_cache_byte_view(
        swa_k_cache, name="swa_k_cache"
    )
    _validate_compressed_cache_layout(
        swa_k_cache,
        page_size=swa_page_size,
        name="swa_k_cache",
        cache_format=cache_format,
        cache_kind="swa",
        allow_empty=swa_indices_2d.shape[1] == 0,
    )

    swa_indices_2d = _normalize_index_matrix(swa_indices, name="swa_indices")
    if swa_indices_2d.device != q3.device:
        raise ValueError("swa_indices must be on the same device as q_all")
    if swa_indices_2d.shape[0] != rows:
        raise ValueError("swa_indices row count must match q_all")
    _validate_lengths(
        swa_topk_lengths,
        rows=rows,
        name="swa_topk_lengths",
    )
    if swa_topk_lengths.device != q3.device:
        raise ValueError("swa_topk_lengths must be on the same device as q_all")

    has_indexed = (
        indexed_k_cache is not None
        or indexed_indices is not None
        or indexed_topk_lengths is not None
    )
    if has_indexed:
        if (
            indexed_k_cache is None
            or indexed_indices is None
            or indexed_topk_lengths is None
        ):
            raise ValueError(
                "indexed_k_cache, indexed_indices, and indexed_topk_lengths must be provided together"
            )
        if indexed_page_size is None:
            raise ValueError(
                "indexed_page_size is required when indexed_k_cache is provided"
            )
        indexed_k_cache = _compressed_sparse_mla_cache_byte_view(
            indexed_k_cache, name="indexed_k_cache"
        )
        indexed_indices_2d = _normalize_index_matrix(
            indexed_indices, name="indexed_indices"
        )
        _validate_compressed_cache_layout(
            indexed_k_cache,
            page_size=int(indexed_page_size),
            name="indexed_k_cache",
            cache_format=cache_format,
            cache_kind="indexed",
            allow_empty=indexed_indices_2d.shape[1] == 0,
        )
        indexed_indices_2d = _normalize_index_matrix(
            indexed_indices, name="indexed_indices"
        )
        if indexed_indices_2d.device != q3.device:
            raise ValueError("indexed_indices must be on the same device as q_all")
        if indexed_indices_2d.shape[0] != rows:
            raise ValueError("indexed_indices row count must match q_all")
        _validate_lengths(
            indexed_topk_lengths,
            rows=rows,
            name="indexed_topk_lengths",
        )
        if indexed_topk_lengths.device != q3.device:
            raise ValueError("indexed_topk_lengths must be on the same device as q_all")
        if indexed_page_table is not None:
            if cache_format != "deepseek_v41":
                raise ValueError(
                    "mapped indexed_page_table is only supported for deepseek_v41"
                )
            from ...compressed_sparse_mla._metadata import map_indexed_pages
            mapped = getattr(scratch, "mapped_indices", None)
            if mapped is None:
                raise ValueError(
                    "mapped indexed pages require component-planned scratch"
                )
            if mapper_launcher is None:
                raise RuntimeError(
                    "mapped indexed pages require a prepared native mapper"
                )
            indexed_indices_2d = map_indexed_pages(
                indexed_indices_2d,
                indexed_topk_lengths,
                indexed_page_table,
                mapped,
                page_size=int(indexed_page_size),
                num_pages=int(indexed_k_cache.shape[0]),
                launcher=mapper_launcher,
            )
    else:
        indexed_indices_2d = None
        if indexed_page_table is not None:
            raise ValueError(
                "indexed_page_table requires indexed_k_cache/indices/lengths"
            )

    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if attn_sink.shape != (heads,):
            raise ValueError(
                f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}"
            )
        if attn_sink.device != q3.device:
            raise ValueError(
                f"attn_sink device {attn_sink.device} does not match q_all device {q3.device}"
            )
        if attn_sink.dtype != torch.float32:
            raise TypeError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous")

    _validate_compressed_sparse_mla_scratch(
        scratch=scratch,
        rows=rows,
        heads=heads,
        width=swa_indices_2d.shape[1]
        + (indexed_indices_2d.shape[1] if has_indexed else 0),
    )

    if out is not None:
        _validate_compressed_sparse_mla_out(out, q3=q3)
    if rows == 0:
        output = out if out is not None else scratch.output_buffer[:0]
        return (output, scratch.final_lse[:0]) if return_lse else output

    # A prepared compressed execution owns one exact route.  In particular,
    # live row counts must not rerun the SM121 split-vs-single-pass heuristic:
    # the prefill tuple and unified decode launch are distinct executable owners.
    if scratch.mode in ("extend", "verify", "draft_extend") or isinstance(prepared, tuple):
        if prepared is None:
            raise RuntimeError("compressed MLA prefill requires prepared native launchers")
        return _run_sm120_compressed_prefill(
            q3=q3,
            swa_k_cache=swa_k_cache,
            swa_indices=swa_indices_2d,
            swa_topk_lengths=swa_topk_lengths,
            workspace=scratch,
            sm_scale=sm_scale,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_k_cache if has_indexed else None,
            indexed_indices=indexed_indices_2d,
            indexed_topk_lengths=indexed_topk_lengths if has_indexed else None,
            indexed_page_size=indexed_page_size if has_indexed else None,
            attn_sink=attn_sink,
            return_lse=return_lse,
            lse_scale=lse_scale,
            out=out,
            prepared=prepared,
        )

    if prepared is None:
        raise RuntimeError("compressed MLA decode requires a prepared native launch")

    from .kernel import run_prepared_unified_decode

    return run_prepared_unified_decode(
        prepared=prepared,
        q_all=q3,
        swa_k_cache=swa_k_cache,
        swa_indices=swa_indices_2d,
        swa_topk_lengths=swa_topk_lengths,
        workspace=scratch,
        sm_scale=sm_scale,
        swa_page_size=swa_page_size,
        indexed_k_cache=indexed_k_cache if has_indexed else None,
        indexed_indices=indexed_indices_2d,
        indexed_topk_lengths=indexed_topk_lengths if has_indexed else None,
        indexed_page_size=indexed_page_size if has_indexed else None,
        attn_sink=attn_sink,
        return_lse=return_lse,
        lse_scale=lse_scale,
        out=out,
        **(
            {
                "v41_compute_mode": scratch.execution_config.v41_compute_mode,
                "v41_heads_per_block": scratch.execution_config.v41_heads_per_block,
            }
            if cache_format == "deepseek_v41"
            else {}
        ),
    )


def _validate_compressed_sparse_mla_out(out: torch.Tensor, *, q3: torch.Tensor) -> None:
    rows, heads, _ = q3.shape
    expected = (int(rows), int(heads), COMPRESSED_SPARSE_MLA_HEAD_DIM)
    if tuple(out.shape) != expected:
        raise ValueError(
            f"compressed MLA out must have shape {expected}, got {tuple(out.shape)}"
        )
    if out.dtype != torch.bfloat16:
        raise TypeError(f"compressed MLA out must be bfloat16, got {out.dtype}")
    if out.device != q3.device:
        raise ValueError(
            f"compressed MLA out device {out.device} does not match q device {q3.device}"
        )
    if not out.is_contiguous():
        raise ValueError("compressed MLA out must be contiguous")


def _run_sm120_compressed_prefill(
    *,
    q3: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    workspace: object,
    sm_scale: float,
    swa_page_size: int,
    indexed_k_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_topk_lengths: torch.Tensor | None,
    indexed_page_size: int | None,
    attn_sink: torch.Tensor | None,
    return_lse: bool,
    lse_scale: Literal["base2", "natural"],
    out: torch.Tensor | None = None,
    prepared: object | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Route a DSV4 prefill-like (extend/verify/draft_extend) compressed call to the
    active SM120 single-pass prefill.

    Mirrors upstream's num_tokens>64 prefill orchestrator: ONE 384-thread CTA per
    (token, HPB head-group) over ALL topk tiles (FINAL_BF16 epilogue, no split-K
    merge). Main cache OR the DSV4 DUAL-CACHE union (extra tokens). Per-token
    ``swa_topk_lengths`` is the per-token main topk_length; attn_sink + return_lse
    are supported. An unsupported prefill shape RAISEs inside run_unified_prefill
    (error like upstream, NOT a legacy fallback).
    """
    from .kernel import run_unified_prefill
    from .traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

    execution_config = getattr(workspace, "execution_config", None)
    if execution_config is None:
        raise RuntimeError("compressed MLA scratch is missing execution_config")
    traits = (
        replace(
            make_unified_traits(
                ModelType.DSV41,
                ComputeMode.BF16,
                ScaleFormat.NVFP4_E4M3,
                fp8_rope=False,
            ),
            fp8_internal=execution_config.v41_compute_mode == "fp8",
        )
        if getattr(workspace, "cache_format", "deepseek_v4") == "deepseek_v41"
        else None
    )

    swa_indices_2d = _normalize_index_matrix(swa_indices, name="swa_indices")
    if out is not None:
        output = out
    else:
        output = _get_mla_output_view(
            workspace=workspace,
            q_all=q3,
            v_head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
        )

    extra_kwargs: dict = {}
    if indexed_k_cache is not None:
        extra_kwargs = dict(
            extra_kv_cache=indexed_k_cache,
            extra_indices=_normalize_index_matrix(
                indexed_indices, name="indexed_indices"
            ),
            extra_topk_length=indexed_topk_lengths,
            extra_page_block_size=int(indexed_page_size),
        )

    result = run_unified_prefill(
        q=q3,
        kv_cache=swa_k_cache,
        topk_indices=swa_indices_2d,
        sm_scale=float(sm_scale),
        page_block_size=int(swa_page_size),
        topk_length=swa_topk_lengths,
        attn_sink=attn_sink,
        output=output,
        lse_out=workspace.final_lse[: int(q3.shape[0])],
        traits_override=traits,
        prepared=prepared,
        # Compressed DSV4's single-pass route is always the selected MG route.
        # Runtime must not reread the legacy environment control.
        mg_enabled=True,
        **extra_kwargs,
    )
    if isinstance(result, tuple) and result and not isinstance(result[0], torch.Tensor):
        return result
    _, lse_base2 = result
    if not return_lse:
        return output
    lse = lse_base2 if lse_scale == "base2" else (lse_base2.mul_(_LN2))
    return output, lse


def _stage_fixed_compressed_sparse_mla_inputs(
    *,
    workspace: object,
    q_all: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    indexed_indices: torch.Tensor,
    indexed_lengths: torch.Tensor,
    indexed_page_table: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    q_stage = workspace.compressed_sparse_mla_q_stage
    swa_indices_stage = workspace.compressed_sparse_mla_swa_indices_stage
    swa_lengths_stage = workspace.compressed_sparse_mla_swa_lengths_stage
    indexed_indices_stage = workspace.compressed_sparse_mla_indexed_indices_stage
    indexed_lengths_stage = workspace.compressed_sparse_mla_indexed_lengths_stage
    indexed_page_table_stage = workspace.compressed_sparse_mla_indexed_page_table_stage
    if (
        q_stage is None
        or swa_indices_stage is None
        or swa_lengths_stage is None
        or indexed_indices_stage is None
        or indexed_lengths_stage is None
        or indexed_page_table_stage is None
    ):
        raise RuntimeError(
            "fixed compressed MLA scratch is missing capacity staging buffers; "
            "set reserve_compressed_sparse_mla_staging=True when planning scratch"
        )

    rows = int(q_all.shape[0])
    cap_rows = int(workspace.max_total_q)
    if rows > cap_rows:
        raise ValueError(
            f"q rows {rows} exceed fixed compressed MLA staging capacity {cap_rows}"
        )
    if q_stage.shape != (
        cap_rows,
        int(workspace.num_q_heads),
        COMPRESSED_SPARSE_MLA_HEAD_DIM,
    ):
        raise ValueError(
            "compressed MLA q staging buffer shape mismatch: "
            f"got {tuple(q_stage.shape)}, expected "
            f"({cap_rows}, {int(workspace.num_q_heads)}, {COMPRESSED_SPARSE_MLA_HEAD_DIM})"
        )
    for name, stage in (
        ("swa_lengths", swa_lengths_stage),
        ("indexed_lengths", indexed_lengths_stage),
    ):
        if stage.shape != (cap_rows,):
            raise ValueError(
                f"compressed MLA {name} staging buffer shape mismatch: "
                f"got {tuple(stage.shape)}, expected ({cap_rows},)"
            )
        if stage.dtype != torch.int32:
            raise TypeError(
                f"compressed MLA {name} staging buffer must be int32, got {stage.dtype}"
            )
        if stage.device != q_all.device:
            raise ValueError(
                f"compressed MLA {name} staging buffer must be on {q_all.device}"
            )
    q_stage[:rows].copy_(q_all.detach())

    swa_indices_view = _stage_fixed_int_matrix(
        swa_indices_stage,
        swa_indices,
        rows=rows,
        cap_rows=cap_rows,
        name="swa_indices",
    )
    indexed_indices_view = _stage_fixed_int_matrix(
        indexed_indices_stage,
        indexed_indices,
        rows=rows,
        cap_rows=cap_rows,
        name="indexed_indices",
    )
    indexed_page_table_view = _stage_fixed_int_matrix(
        indexed_page_table_stage,
        indexed_page_table,
        rows=rows,
        cap_rows=cap_rows,
        name="indexed_page_table",
    )
    swa_lengths_view = swa_lengths_stage[:cap_rows]
    indexed_lengths_view = indexed_lengths_stage[:cap_rows]
    swa_lengths_view[:rows].copy_(swa_lengths.detach())
    indexed_lengths_view[:rows].copy_(indexed_lengths.detach())
    if rows < cap_rows:
        swa_lengths_view[rows:].zero_()
        indexed_lengths_view[rows:].zero_()

    return (
        q_stage,
        swa_indices_view,
        swa_lengths_view,
        indexed_indices_view,
        indexed_lengths_view,
        indexed_page_table_view,
    )


def _stage_fixed_int_matrix(
    stage: torch.Tensor,
    source: torch.Tensor,
    *,
    rows: int,
    cap_rows: int,
    name: str,
) -> torch.Tensor:
    width = int(source.shape[1])
    if int(stage.shape[0]) < cap_rows or int(stage.shape[1]) < width:
        raise ValueError(
            f"{name} staging buffer is too small: stage={tuple(stage.shape)} "
            f"required=({cap_rows}, {width})"
        )
    view = stage.reshape(-1)[: cap_rows * width].view(cap_rows, width)
    if rows:
        view[:rows].copy_(source.detach())
    return view


def _normalize_compressed_q(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0]
    if q.ndim != 3 or q.shape[-1] != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"q_all must have shape [rows, heads, {COMPRESSED_SPARSE_MLA_HEAD_DIM}], got {tuple(q.shape)}"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q_all must have dtype torch.bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q_all must be contiguous for compressed MLA")
    return q.detach()


def _validate_compressed_cache_layout(
    cache: torch.Tensor,
    *,
    page_size: int,
    name: str,
    cache_format: str = "deepseek_v4",
    cache_kind: str = "swa",
    allow_empty: bool = False,
) -> None:
    if int(cache.shape[0]) < 1 and not allow_empty:
        raise ValueError(f"{name} must contain at least one physical page")
    if cache_format == "deepseek_v41" and (
        cache.data_ptr() % 16 or cache.stride(0) % 16
    ):
        raise ValueError(
            f"{name} V4.1 pages must have 16-byte aligned storage and strides"
        )
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError(f"{name} page_size must be positive, got {page_size}")
    from .kv_cache import page_nbytes as cache_page_nbytes

    padded_page_nbytes = cache_page_nbytes(
        page_size, cache_format=cache_format, cache_kind=cache_kind
    )
    payload_nbytes = (
        page_size * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
        if cache_format == "deepseek_v4"
        else padded_page_nbytes
    )
    page_nbytes = int(cache.shape[1])
    if page_nbytes not in (payload_nbytes, padded_page_nbytes):
        raise ValueError(
            f"{name} page byte width must be the contiguous payload "
            f"{payload_nbytes} or padded width {padded_page_nbytes} for "
            f"page_size {page_size}, got {page_nbytes}"
        )
def _compressed_sparse_mla_cache_byte_view(
    cache: torch.Tensor, *, name: str
) -> torch.Tensor:
    if cache.dtype == torch.uint8:
        byte_cache = cache
    elif cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        byte_cache = cache.detach().view(torch.uint8)
    else:
        raise TypeError(
            f"{name} must have dtype torch.uint8 or FP8 storage, got {cache.dtype}"
        )
    if byte_cache.ndim != 2:
        raise ValueError(
            f"{name} must have shape [pages, page_bytes], got {tuple(cache.shape)}"
        )
    if int(byte_cache.stride(1)) != 1:
        raise ValueError(
            f"{name} page payload must be contiguous in the last dimension, "
            f"got stride {tuple(byte_cache.stride())}"
        )
    return byte_cache


def _validate_compressed_launch_views(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    q_rows: int,
    heads: int,
    launch_num_chunks: int,
    direct_output: bool,
) -> None:
    if tmp_output is None or tmp_lse is None:
        raise RuntimeError("compressed MLA launch is missing scratch/output buffers")
    q_rows = int(q_rows)
    heads = int(heads)
    launch_num_chunks = int(launch_num_chunks)
    if tmp_output.dtype != torch.bfloat16:
        raise TypeError(
            f"compressed MLA tmp_output must be BF16, got {tmp_output.dtype}"
        )
    if tmp_lse.dtype != torch.float32:
        raise TypeError(f"compressed MLA tmp_lse must be FP32, got {tmp_lse.dtype}")
    if tmp_output.device != tmp_lse.device:
        raise ValueError(
            "compressed MLA tmp_output and tmp_lse must be on the same device"
        )
    _validate_tensor_storage_bounds(tmp_output, name="compressed MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="compressed MLA tmp_lse")
    if direct_output:
        if tmp_output.ndim != 3:
            raise ValueError(
                f"compressed MLA direct output must be rank-3, got {tuple(tmp_output.shape)}"
            )
        required = (q_rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM)
        if (
            int(tmp_output.shape[0]) < q_rows
            or int(tmp_output.shape[1]) < heads
            or int(tmp_output.shape[2]) < COMPRESSED_SPARSE_MLA_HEAD_DIM
        ):
            raise ValueError(
                "compressed MLA direct output is too small: "
                f"buffer={tuple(tmp_output.shape)} required>={required}"
            )
    else:
        if tmp_output.ndim != 4:
            raise ValueError(
                f"compressed MLA split output must be rank-4, got {tuple(tmp_output.shape)}"
            )
        required = (q_rows, heads, launch_num_chunks, COMPRESSED_SPARSE_MLA_HEAD_DIM)
        if (
            int(tmp_output.shape[0]) < q_rows
            or int(tmp_output.shape[1]) < heads
            or int(tmp_output.shape[2]) < launch_num_chunks
            or int(tmp_output.shape[3]) < COMPRESSED_SPARSE_MLA_HEAD_DIM
        ):
            raise ValueError(
                "compressed MLA split output is too small: "
                f"buffer={tuple(tmp_output.shape)} required>={required}"
            )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"compressed MLA tmp_lse must be rank-3, got {tuple(tmp_lse.shape)}"
        )
    required_lse = (q_rows, heads, max(1, launch_num_chunks))
    if (
        int(tmp_lse.shape[0]) < q_rows
        or int(tmp_lse.shape[1]) < heads
        or int(tmp_lse.shape[2]) < max(1, launch_num_chunks)
    ):
        raise ValueError(
            "compressed MLA tmp_lse is too small: "
            f"buffer={tuple(tmp_lse.shape)} required>={required_lse}"
        )


def _is_row_shared_index_matrix(indices: torch.Tensor) -> bool:
    return (
        indices.ndim == 2
        and int(indices.stride(0)) == 0
        and int(indices.stride(1)) == 1
    )


def _normalize_index_matrix(
    indices: torch.Tensor,
    *,
    name: str,
    allow_row_shared: bool = False,
) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(
            f"{name} must have shape [rows, width] or [rows, 1, width], got {tuple(indices.shape)}"
        )
    if indices.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {indices.dtype}")
    if not indices.is_contiguous() and not (
        allow_row_shared and _is_row_shared_index_matrix(indices)
    ):
        raise ValueError(f"{name} must be contiguous for compressed MLA")
    return indices


def _validate_lengths(
    lengths: torch.Tensor,
    *,
    rows: int,
    name: str,
) -> None:
    if lengths.shape != (rows,):
        raise ValueError(f"{name} must have shape [{rows}], got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {lengths.dtype}")
    if not lengths.is_contiguous():
        raise ValueError(f"{name} must be contiguous for compressed MLA")


def _validate_compressed_sparse_mla_scratch(
    scratch: object,
    *,
    rows: int,
    heads: int,
    width: int,
) -> None:
    if rows > scratch.max_total_q:
        raise ValueError(
            f"q rows {rows} exceed compressed MLA scratch max_total_q {scratch.max_total_q}"
        )
    if rows > scratch.max_batch and scratch.mode == "decode":
        raise ValueError(
            f"decode rows {rows} exceed compressed MLA scratch max_batch {scratch.max_batch}"
        )
    if heads != scratch.num_q_heads:
        raise ValueError(
            f"q_all num_heads {heads} does not match compressed MLA scratch num_q_heads {scratch.num_q_heads}"
        )
    if scratch.head_dim != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA scratch head_dim must be {COMPRESSED_SPARSE_MLA_HEAD_DIM}, got {scratch.head_dim}"
        )
    if scratch.v_head_dim != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"compressed MLA scratch v_head_dim must be {COMPRESSED_SPARSE_MLA_HEAD_DIM}, got {scratch.v_head_dim}"
        )
    if width > scratch.topk:
        raise ValueError(
            f"compressed MLA width {width} exceeds scratch topk {scratch.topk}"
        )
