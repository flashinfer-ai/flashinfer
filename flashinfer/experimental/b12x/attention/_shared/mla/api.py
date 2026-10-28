"""Sparse MLA API oriented around the NSA runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep the pure-PyTorch fallback usable outside vLLM images.
    triton = None
    tl = None

from b12x._lib.compile_plan import launch_triton

from .merge import clear_sparse_mla_merge_kernel_cache
from .reference import sparse_mla_reference
from .traits import ModelType, kv_fp8_rope_enabled, resolve_unplanned_traits

_MLA_STRATEGY_ENV = "B12X_MLA_PREFILL_STRATEGY"
_MLA_FORCE_SINGLE_PASS_ENV = "B12X_MLA_FORCE_SINGLE_PASS"
_MLA_FORCE_SPLIT_ENV = "B12X_MLA_FORCE_SPLIT"
_MLA_SM120_BACKEND = "sm120"
# GLM_NSA uncompressed decode contract (q_head_dim = d_nope+d_rope = 512+64).
_MLA_UNIFIED_GLM_Q_HEAD_DIM = 576
_MLA_UNIFIED_GLM_NEXT_Q_HEAD_DIM = 512
_MLA_UNIFIED_GLM_NEXT_RECORD_BYTES = 528
_MLA_UNIFIED_GLM_NEXT_NVFP4_RECORD_BYTES = 304
_MLA_SINGLE_PASS_TARGET_Q_ROWS = 2048
_MLA_SINGLE_PASS_TARGET_TOPK = 2048
_LN2 = math.log(2.0)


if triton is not None and tl is not None:

    @triton.jit
    def _split_decode_final_lse_kernel(
        tmp_lse_ptr,
        out_lse_ptr,
        tmp_lse_stride_b: tl.constexpr,
        tmp_lse_stride_h: tl.constexpr,
        tmp_lse_stride_c: tl.constexpr,
        out_lse_stride_b: tl.constexpr,
        out_lse_stride_h: tl.constexpr,
        chunk_count: tl.constexpr,
        block_c: tl.constexpr,
        natural_scale: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, block_c)
        valid = offs < chunk_count
        vals = tl.load(
            tmp_lse_ptr
            + row * tmp_lse_stride_b
            + head * tmp_lse_stride_h
            + offs * tmp_lse_stride_c,
            mask=valid,
            other=-float("inf"),
        )
        vals = tl.where(vals != vals, -float("inf"), vals)
        lse_max = tl.max(vals, axis=0)
        safe_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
        lse_sum = tl.sum(tl.exp2(vals - safe_max), axis=0)
        lse_base2 = safe_max + tl.log2(lse_sum)
        out = lse_base2
        if natural_scale:
            out = out * 0.69314718055994530942
        out = tl.where(lse_max == -float("inf"), -float("inf"), out)
        tl.store(out_lse_ptr + row * out_lse_stride_b + head * out_lse_stride_h, out)


@dataclass(frozen=True)
class MLASparseDecodeMetadata:
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    max_seq_len_k: int


@dataclass(frozen=True)
class MLASparseExtendMetadata:
    selected_token_offsets: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    nsa_cu_seqlens_q: torch.Tensor
    nsa_cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int
    mode: Literal["extend", "verify", "target_verify", "draft_extend"] = "extend"


def clear_mla_caches() -> None:
    """Clear any cached MLA runtime state."""
    clear_sparse_mla_merge_kernel_cache()
    from .kv_cache import clear_glm_next_mla_kv_cache_kernel_cache

    clear_glm_next_mla_kv_cache_kernel_cache()


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _validate_tensor_storage_bounds(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.numel() == 0:
        return
    min_offset = int(tensor.storage_offset())
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent = (int(size) - 1) * int(stride)
        if extent >= 0:
            max_offset += extent
        else:
            min_offset += extent
    storage_elems = tensor.untyped_storage().nbytes() // tensor.element_size()
    if min_offset < 0 or max_offset >= storage_elems:
        raise ValueError(
            f"{name} view is out of storage bounds: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())} storage_offset={int(tensor.storage_offset())} "
            f"storage_elems={storage_elems}"
        )


def _is_supported_packed_kv_cache_view(
    tensor: torch.Tensor,
    *,
    page_size: int,
) -> bool:
    if tensor.ndim != 3 or int(tensor.shape[1]) != int(page_size):
        return False
    if int(tensor.stride(2)) != 1:
        return False
    if int(tensor.stride(1)) != int(tensor.shape[2]):
        return False
    page_elems = int(tensor.shape[1]) * int(tensor.shape[2])
    return int(tensor.stride(0)) >= page_elems


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _resolve_kv_fp8_rope(value: bool | None) -> bool:
    # Cache layout is a process-lifetime ABI: only the literal operator gate
    # value "1" enables it; unset and every other value retain the 432-byte ABI.
    if value is not None:
        return bool(value)
    return kv_fp8_rope_enabled()


def _use_sm120_sparse_mla(*, backend: str | None, device: torch.device) -> bool:
    """Return whether the active SM120 sparse MLA kernel path is available."""
    if backend is not None and backend != _MLA_SM120_BACKEND:
        raise ValueError(
            f"backend must be None or {_MLA_SM120_BACKEND!r}; legacy sparse MLA "
            f"kernels have been retired, got {backend!r}"
        )
    if device.type != "cuda":
        return False
    from b12x._lib.intrinsics import get_sm_version

    return get_sm_version(device) >= 120


def _resolve_mla_prefill_strategy() -> Literal["auto", "single", "split"]:
    if _env_flag(_MLA_FORCE_SINGLE_PASS_ENV):
        return "single"
    if _env_flag(_MLA_FORCE_SPLIT_ENV):
        return "split"

    value = os.environ.get(_MLA_STRATEGY_ENV, "auto").strip().lower()
    if value in ("auto", ""):
        return "auto"
    if value in ("single", "single-pass", "nonsplit", "onepass"):
        return "single"
    if value == "split":
        return "split"
    raise ValueError(
        f"{_MLA_STRATEGY_ENV} must be auto, split, or single-pass/nonsplit; got {value!r}"
    )


def _apply_mla_prefill_strategy(
    *,
    split_cfg,
    workspace: object,
    active_token_counts: torch.Tensor,
    device: torch.device,
    q_rows: int,
    topk_width: int,
):
    if split_cfg is None:
        return None

    strategy = _resolve_mla_prefill_strategy()
    if strategy == "single":
        return None
    if strategy == "split":
        return split_cfg
    if (
        workspace.mode in ("extend", "verify", "draft_extend")
        and int(workspace.max_batch) == 1
        and int(q_rows) >= _MLA_SINGLE_PASS_TARGET_Q_ROWS
        and int(topk_width) >= _MLA_SINGLE_PASS_TARGET_TOPK
    ):
        return None

    return split_cfg


def _get_mla_output_view(
    *,
    workspace: object,
    q_all: torch.Tensor,
    v_head_dim: int,
) -> torch.Tensor:
    rows = int(q_all.shape[0])
    heads = int(q_all.shape[1])
    v_head_dim = int(v_head_dim)
    output_buffer = workspace.output_buffer
    if output_buffer is None:
        raise RuntimeError("workspace is missing MLA output buffer")
    if output_buffer.device != q_all.device:
        raise ValueError(
            f"workspace MLA output buffer is on {output_buffer.device}, expected {q_all.device}"
        )
    if output_buffer.dtype != q_all.dtype:
        raise TypeError(
            f"workspace MLA output buffer has dtype {output_buffer.dtype}, expected {q_all.dtype}"
        )
    if output_buffer.ndim != 3:
        raise ValueError(
            f"workspace MLA output buffer must be rank 3, got {output_buffer.ndim}"
        )
    if (
        int(output_buffer.shape[0]) < rows
        or int(output_buffer.shape[1]) < heads
        or int(output_buffer.shape[2]) < v_head_dim
    ):
        raise ValueError(
            "workspace MLA output buffer is too small: "
            f"buffer={tuple(output_buffer.shape)} required=({rows}, {heads}, {v_head_dim})"
        )
    _validate_tensor_storage_bounds(output_buffer, name="workspace MLA output buffer")
    return output_buffer[:rows, :heads, :v_head_dim]


def _get_mla_lse_view(
    *,
    workspace: object,
    q_all: torch.Tensor,
) -> torch.Tensor:
    rows = int(q_all.shape[0])
    heads = int(q_all.shape[1])
    final_lse = workspace.final_lse
    if final_lse is None:
        raise RuntimeError("workspace is missing MLA final LSE buffer")
    if final_lse.device != q_all.device:
        raise ValueError(
            f"workspace MLA final LSE buffer is on {final_lse.device}, "
            f"expected {q_all.device}"
        )
    if final_lse.dtype != torch.float32:
        raise TypeError(
            "workspace MLA final LSE buffer must have dtype torch.float32, "
            f"got {final_lse.dtype}"
        )
    if final_lse.ndim != 2:
        raise ValueError(
            f"workspace MLA final LSE buffer must be rank 2, got {final_lse.ndim}"
        )
    if int(final_lse.shape[0]) < rows or int(final_lse.shape[1]) < heads:
        raise ValueError(
            "workspace MLA final LSE buffer is too small: "
            f"buffer={tuple(final_lse.shape)} required=({rows}, {heads})"
        )
    _validate_tensor_storage_bounds(final_lse, name="workspace MLA final LSE buffer")
    return final_lse[:rows, :heads]


def _validate_split_control_tensors(
    *,
    workspace: object,
) -> None:
    for name, tensor in (
        ("kv_chunk_size_ptr", workspace.kv_chunk_size_ptr),
        ("num_chunks_ptr", workspace.num_chunks_ptr),
    ):
        if tensor is None:
            raise RuntimeError(f"workspace is missing {name}")
        if tensor.shape != (1,):
            raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tensor.device != workspace.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match workspace device {workspace.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _validate_split_workspace_views(
    *,
    workspace: object,
    q_rows: int,
    num_heads: int,
    v_head_dim: int,
    launch_num_chunks: int,
) -> None:
    _validate_split_control_tensors(workspace=workspace)
    if workspace.tmp_output is None or workspace.tmp_lse is None:
        raise RuntimeError("workspace is missing split MLA buffers")

    q_rows = int(q_rows)
    num_heads = int(num_heads)
    v_head_dim = int(v_head_dim)
    launch_num_chunks = int(launch_num_chunks)
    tmp_output = workspace.tmp_output
    tmp_lse = workspace.tmp_lse

    if tmp_output.device != workspace.device or tmp_lse.device != workspace.device:
        raise ValueError("split MLA scratch buffers must be on the workspace device")
    if tmp_output.dtype != workspace.dtype:
        raise TypeError(
            f"split MLA tmp_output dtype {tmp_output.dtype} does not match workspace dtype {workspace.dtype}"
        )
    if tmp_lse.dtype != torch.float32:
        raise TypeError(
            f"split MLA tmp_lse must have dtype torch.float32, got {tmp_lse.dtype}"
        )
    if tmp_output.ndim != 4:
        raise ValueError(
            f"split MLA tmp_output must be rank-4, got {tuple(tmp_output.shape)}"
        )
    if tmp_lse.ndim != 3:
        raise ValueError(
            f"split MLA tmp_lse must be rank-3, got {tuple(tmp_lse.shape)}"
        )
    _validate_tensor_storage_bounds(tmp_output, name="split MLA tmp_output")
    _validate_tensor_storage_bounds(tmp_lse, name="split MLA tmp_lse")
    required_output = (q_rows, num_heads, launch_num_chunks, v_head_dim)
    if (
        int(tmp_output.shape[0]) < q_rows
        or int(tmp_output.shape[1]) < num_heads
        or int(tmp_output.shape[2]) < launch_num_chunks
        or int(tmp_output.shape[3]) < v_head_dim
    ):
        raise ValueError(
            "split MLA tmp_output is too small: "
            f"buffer={tuple(tmp_output.shape)} required>={required_output}"
        )
    required_lse = (q_rows, num_heads, launch_num_chunks)
    if (
        int(tmp_lse.shape[0]) < q_rows
        or int(tmp_lse.shape[1]) < num_heads
        or int(tmp_lse.shape[2]) < launch_num_chunks
    ):
        raise ValueError(
            "split MLA tmp_lse is too small: "
            f"buffer={tuple(tmp_lse.shape)} required>={required_lse}"
        )


def _resolve_sparse_mla_binding(
    *,
    binding,
    q_all: torch.Tensor | None,
    selected_indices: torch.Tensor | None,
    cache_seqlens_int32: torch.Tensor | None,
    nsa_cache_seqlens_int32: torch.Tensor | None,
    selected_name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
    torch.Tensor | None,
]:
    if binding is None:
        raise TypeError("sparse MLA forward requires binding")

    extras = [
        name
        for name, value in (
            ("q_all", q_all),
            (selected_name, selected_indices),
            ("cache_seqlens_int32", cache_seqlens_int32),
            ("nsa_cache_seqlens_int32", nsa_cache_seqlens_int32),
        )
        if value is not None
    ]
    if extras:
        raise ValueError(
            "sparse MLA binding owns runtime tensors and scratch; "
            f"do not also pass {', '.join(extras)}"
        )
    return (
        binding.q,
        binding.selected_indices,
        binding.cache_seqlens_int32,
        binding.nsa_cache_seqlens_int32,
        binding.scratch,
        binding.kv_cache,
    )


def sparse_mla_decode_forward(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    page_table_1: torch.Tensor | None = None,
    cache_seqlens_int32: torch.Tensor | None = None,
    nsa_cache_seqlens_int32: torch.Tensor | None = None,
    binding=None,
    sm_scale: float,
    latent_scale: float = 1.0,
    v_head_dim: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    attn_sink: torch.Tensor | None = None,
    identity_page_table: bool = False,
    backend: str | None = None,
    forced_num_splits: int | None = None,
    scale_format: int | None = None,
    model_type: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool | None = None,
    prepared: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    (
        q_all,
        page_table_1,
        cache_seqlens_int32,
        nsa_cache_seqlens_int32,
        workspace,
        bound_kv_cache,
    ) = _resolve_sparse_mla_binding(
        binding=binding,
        q_all=q_all,
        selected_indices=page_table_1,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        selected_name="page_table_1",
    )
    if bound_kv_cache is not None:
        if kv_cache is not None and kv_cache is not bound_kv_cache:
            raise ValueError(
                "sparse MLA binding owns kv_cache; do not also pass kv_cache"
            )
        kv_cache = bound_kv_cache
    if kv_cache is None:
        raise TypeError("sparse MLA forward requires kv_cache in its binding")
    if v_head_dim is None:
        v_head_dim = workspace.v_head_dim
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_indices=page_table_1,
        cache_seqlens_int32=cache_seqlens_int32,
        active_token_counts=nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        latent_scale=latent_scale,
        v_head_dim=v_head_dim,
        return_lse=return_lse,
        lse_scale=lse_scale,
        attn_sink=attn_sink,
        identity_page_table=identity_page_table,
        backend=backend,
        forced_num_splits=forced_num_splits,
        scale_format=scale_format,
        model_type=model_type,
        fp8_rope=fp8_rope,
        latent_scale_per_token=latent_scale_per_token,
        prepared=prepared,
    )


def sparse_mla_extend_forward(
    *,
    q_all: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    selected_token_offsets: torch.Tensor | None = None,
    cache_seqlens_int32: torch.Tensor | None = None,
    nsa_cache_seqlens_int32: torch.Tensor | None = None,
    binding=None,
    sm_scale: float,
    latent_scale: float = 1.0,
    v_head_dim: int | None = None,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    identity_page_table: bool = False,
    scale_format: int | None = None,
    model_type: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool | None = None,
    prepared: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    (
        q_all,
        selected_token_offsets,
        cache_seqlens_int32,
        nsa_cache_seqlens_int32,
        workspace,
        bound_kv_cache,
    ) = _resolve_sparse_mla_binding(
        binding=binding,
        q_all=q_all,
        selected_indices=selected_token_offsets,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        selected_name="selected_token_offsets",
    )
    if bound_kv_cache is not None:
        if kv_cache is not None and kv_cache is not bound_kv_cache:
            raise ValueError(
                "sparse MLA binding owns kv_cache; do not also pass kv_cache"
            )
        kv_cache = bound_kv_cache
    if kv_cache is None:
        raise TypeError("sparse MLA forward requires kv_cache in its binding")
    if v_head_dim is None:
        v_head_dim = workspace.v_head_dim
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        selected_indices=selected_token_offsets,
        cache_seqlens_int32=cache_seqlens_int32,
        active_token_counts=nsa_cache_seqlens_int32,
        workspace=workspace,
        sm_scale=sm_scale,
        latent_scale=latent_scale,
        v_head_dim=v_head_dim,
        return_lse=return_lse,
        lse_scale=lse_scale,
        identity_page_table=identity_page_table,
        scale_format=scale_format,
        model_type=model_type,
        fp8_rope=fp8_rope,
        latent_scale_per_token=latent_scale_per_token,
        prepared=prepared,
    )


def _run_sparse_mla(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    active_token_counts: torch.Tensor,
    workspace: object,
    sm_scale: float,
    latent_scale: float = 1.0,
    v_head_dim: int,
    return_lse: bool = False,
    lse_scale: Literal["base2", "natural"] = "base2",
    attn_sink: torch.Tensor | None = None,
    identity_page_table: bool = False,
    backend: str | None = None,
    forced_num_splits: int | None = None,
    scale_format: int | None = None,
    model_type: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool | None = None,
    planned_cache_traits: object | None = None,
    prepared: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if q_all.ndim != 3:
        raise ValueError(f"q_all must be rank-3, got {tuple(q_all.shape)}")
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if selected_indices.ndim != 2:
        raise ValueError(
            f"selected indices must be rank-2, got {tuple(selected_indices.shape)}"
        )
    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, "
            f"got {tuple(cache_seqlens_int32.shape)}"
        )
    if active_token_counts.ndim != 1:
        raise ValueError(
            "nsa_cache_seqlens_int32 must be rank-1, "
            f"got {tuple(active_token_counts.shape)}"
        )
    if not q_all.is_contiguous():
        raise ValueError("q_all must be contiguous for sparse MLA")
    if not kv_cache.is_contiguous():
        if not _is_supported_packed_kv_cache_view(
            kv_cache,
            page_size=int(workspace.page_size),
        ):
            raise ValueError(
                "kv_cache must be contiguous or a packed page-strided view for sparse MLA; "
                f"got shape={tuple(kv_cache.shape)} stride={tuple(kv_cache.stride())}"
            )
        _validate_tensor_storage_bounds(kv_cache, name="sparse MLA kv_cache")
    if not selected_indices.is_contiguous():
        raise ValueError("selected indices must be contiguous for sparse MLA")
    if not cache_seqlens_int32.is_contiguous():
        raise ValueError("cache_seqlens_int32 must be contiguous for sparse MLA")
    if not active_token_counts.is_contiguous():
        raise ValueError("nsa_cache_seqlens_int32 must be contiguous for sparse MLA")
    if q_all.device != workspace.device:
        raise ValueError(
            f"q_all device {q_all.device} does not match workspace device {workspace.device}"
        )
    if kv_cache.device != workspace.device:
        raise ValueError(
            f"kv_cache device {kv_cache.device} does not match workspace device {workspace.device}"
        )
    if selected_indices.device != workspace.device:
        raise ValueError(
            "selected indices device "
            f"{selected_indices.device} does not match workspace device {workspace.device}"
        )
    if cache_seqlens_int32.device != workspace.device:
        raise ValueError(
            "cache_seqlens_int32 device "
            f"{cache_seqlens_int32.device} does not match workspace device {workspace.device}"
        )
    if active_token_counts.device != workspace.device:
        raise ValueError(
            "nsa_cache_seqlens_int32 device "
            f"{active_token_counts.device} does not match workspace device {workspace.device}"
        )
    if q_all.dtype != workspace.dtype:
        raise ValueError(
            f"q_all dtype {q_all.dtype} does not match workspace dtype {workspace.dtype}"
        )
    if kv_cache.dtype != workspace.kv_dtype:
        raise ValueError(
            f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {workspace.kv_dtype}"
        )
    if selected_indices.dtype != torch.int32:
        raise ValueError(
            f"selected indices must have dtype torch.int32, got {selected_indices.dtype}"
        )
    if cache_seqlens_int32.dtype != torch.int32:
        raise ValueError(
            "cache_seqlens_int32 must have dtype torch.int32, "
            f"got {cache_seqlens_int32.dtype}"
        )
    if active_token_counts.dtype != torch.int32:
        raise ValueError(
            "nsa_cache_seqlens_int32 must have dtype torch.int32, "
            f"got {active_token_counts.dtype}"
        )
    if int(v_head_dim) != workspace.v_head_dim:
        raise ValueError(
            f"v_head_dim {v_head_dim} does not match workspace v_head_dim {workspace.v_head_dim}"
        )
    _sm120_route = _use_sm120_sparse_mla(backend=backend, device=q_all.device)
    # A binding-owned cache was validated against these immutable traits at
    # bind time. The planned serving path consumes them without repeating ABI
    # checks for every launch.
    cache_traits = planned_cache_traits
    if cache_traits is not None:
        model_type_for_call = int(cache_traits.model_type)
        scale_format_for_call = int(cache_traits.scale_format)
        fp8_rope_for_call = bool(cache_traits.fp8_rope)
        latent_scale_per_token_for_call = bool(cache_traits.latent_scale_per_token)
    else:
        # The compatibility path accepts a cache supplied at run time. It must
        # resolve and validate the cache recipe because bind did not own it.
        cache_traits = getattr(workspace, "cache_traits", None)
        if cache_traits is None and _sm120_route:
            cache_traits = resolve_unplanned_traits(
                int(q_all.shape[-1]),
                kv_cache.dtype,
                int(kv_cache.shape[-1]),
                model_type=model_type,
                scale_format=scale_format,
                fp8_rope=fp8_rope,
                latent_scale_per_token=bool(latent_scale_per_token),
            )
        if cache_traits is not None:
            model_type_for_call = int(cache_traits.model_type)
            scale_format_for_call = int(cache_traits.scale_format)
            fp8_rope_for_call = bool(cache_traits.fp8_rope)
            latent_scale_per_token_for_call = bool(cache_traits.latent_scale_per_token)
            expected_record_bytes = int(cache_traits.kv_gmem_stride)
            for name, supplied, planned in (
                ("model_type", model_type, model_type_for_call),
                ("scale_format", scale_format, scale_format_for_call),
                ("fp8_rope", fp8_rope, fp8_rope_for_call),
                (
                    "latent_scale_per_token",
                    latent_scale_per_token,
                    latent_scale_per_token_for_call,
                ),
            ):
                if supplied is not None and int(supplied) != int(planned):
                    if name == "fp8_rope" and model_type_for_call == ModelType.GLM_NEXT:
                        raise ValueError(
                            "GLM_NEXT sparse MLA has no RoPE cache payload"
                        )
                    raise ValueError(
                        f"sparse MLA run {name} does not match its plan: "
                        f"run={supplied!r}, plan={planned!r}"
                    )
            if int(kv_cache.shape[-1]) != expected_record_bytes:
                raise ValueError(
                    "sparse MLA kv_cache record width does not match its plan: "
                    f"got {int(kv_cache.shape[-1])}, expected {expected_record_bytes}"
                )
            if model_type_for_call not in (ModelType.GLM_NSA, ModelType.GLM_NEXT):
                raise ValueError(
                    "attention.sparse_mla supports ModelType.GLM_NSA or "
                    f"ModelType.GLM_NEXT; got model_type={model_type_for_call}"
                )
        else:
            model_type_for_call = None
            scale_format_for_call = None if scale_format is None else int(scale_format)
            fp8_rope_for_call = None
            latent_scale_per_token_for_call = False
    if scale_format_for_call == 2 and not _sm120_route:
        raise NotImplementedError(
            "NVFP4 sparse MLA requires the active SM120 kernel path"
        )
    if attn_sink is not None:
        attn_sink = attn_sink.detach()
        if not _sm120_route:
            raise ValueError(
                "sparse MLA attn_sink requires the active SM120 kernel path"
            )
        if attn_sink.ndim != 1 or int(attn_sink.shape[0]) != int(q_all.shape[1]):
            raise ValueError(
                f"attn_sink must have shape ({int(q_all.shape[1])},), got {tuple(attn_sink.shape)}"
            )
        if attn_sink.device != workspace.device:
            raise ValueError(
                f"attn_sink device {attn_sink.device} does not match workspace device {workspace.device}"
            )
        if attn_sink.dtype != torch.float32:
            raise ValueError(
                f"attn_sink must have dtype torch.float32, got {attn_sink.dtype}"
            )
        if not attn_sink.is_contiguous():
            raise ValueError("attn_sink must be contiguous for fused sparse MLA")
    if q_all.shape[0] > workspace.max_total_q:
        raise ValueError(
            f"q_all rows {q_all.shape[0]} exceed workspace capacity {workspace.max_total_q}"
        )
    if q_all.shape[0] != selected_indices.shape[0]:
        raise ValueError(
            f"selected-index rows {selected_indices.shape[0]} do not match q_all rows {q_all.shape[0]}"
        )
    if selected_indices.shape[0] != active_token_counts.shape[0]:
        raise ValueError(
            "selected-index rows "
            f"{selected_indices.shape[0]} do not match nsa_cache_seqlens_int32 rows "
            f"{active_token_counts.shape[0]}"
        )
    if cache_seqlens_int32.shape[0] > workspace.max_batch:
        raise ValueError(
            "cache_seqlens_int32 batch "
            f"{cache_seqlens_int32.shape[0]} exceeds workspace capacity {workspace.max_batch}"
        )
    if selected_indices.shape[1] > workspace.topk:
        raise ValueError(
            f"selected-index width {selected_indices.shape[1]} exceeds workspace topk {workspace.topk}"
        )
    if q_all.shape[1] != workspace.num_q_heads:
        raise ValueError(
            f"q_all num_heads {q_all.shape[1]} does not match workspace num_q_heads {workspace.num_q_heads}"
        )
    if q_all.shape[-1] != workspace.head_dim:
        raise ValueError(
            f"q_all head_dim {q_all.shape[-1]} does not match workspace head_dim {workspace.head_dim}"
        )
    if _sm120_route:
        q_head_dim = int(q_all.shape[-1])
        expected_q_head_dim = (
            _MLA_UNIFIED_GLM_NEXT_Q_HEAD_DIM
            if model_type_for_call == ModelType.GLM_NEXT
            else _MLA_UNIFIED_GLM_Q_HEAD_DIM
        )
        if q_head_dim != expected_q_head_dim:
            raise ValueError(
                "SM120 sparse MLA requires the selected GLM contract "
                f"(q_head_dim={expected_q_head_dim}); got q_head_dim={q_head_dim}"
            )
        if workspace.mode in ("extend", "verify", "draft_extend"):
            return _run_sm120_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                selected_indices=selected_indices,
                active_token_counts=active_token_counts,
                workspace=workspace,
                sm_scale=float(sm_scale),
                latent_scale=float(latent_scale),
                v_head_dim=int(v_head_dim),
                attn_sink=attn_sink,
                return_lse=return_lse,
                lse_scale=lse_scale,
                scale_format=scale_format_for_call,
                model_type=model_type_for_call,
                fp8_rope=fp8_rope_for_call,
                latent_scale_per_token=latent_scale_per_token_for_call,
                cache_traits=cache_traits,
            )
        from .kernel import run_unified_decode

        return run_unified_decode(
            q_all=q_all,
            swa_k_cache=kv_cache,
            swa_indices=selected_indices,
            swa_topk_lengths=active_token_counts,
            workspace=workspace,
            sm_scale=float(sm_scale),
            latent_scale=float(latent_scale),
            swa_page_size=int(workspace.page_size),
            attn_sink=attn_sink,
            return_lse=return_lse,
            lse_scale=lse_scale,
            forced_num_splits=forced_num_splits,
            scale_format_override=scale_format_for_call,
            model_type_override=model_type_for_call,
            fp8_rope_override=fp8_rope_for_call,
            latent_scale_per_token=latent_scale_per_token_for_call,
            traits_override=cache_traits,
        )
    if prepared:
        raise RuntimeError(
            "prepared sparse MLA requires the native SM120 route; "
            "a prepared execution cannot fall back to the reference implementation"
        )
    if _is_cuda_graph_capture_active(q_all.device):
        raise RuntimeError(
            "b12x MLA would use the PyTorch reference during CUDA graph capture; "
            "the active SM120 sparse MLA kernel path is unavailable for this device"
        )
    if identity_page_table:
        raise RuntimeError(
            "identity page-table sparse MLA requires the active SM120 kernel path"
        )
    reference_kwargs = dict(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=selected_indices,
        active_token_counts=active_token_counts,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )
    if return_lse:
        reference_kwargs["return_lse"] = True
    output = sparse_mla_reference(**reference_kwargs)
    if return_lse:
        output, lse = output
        if lse_scale == "natural":
            lse = lse * _LN2
        return output, lse
    return output


def _run_sm120_prefill(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    active_token_counts: torch.Tensor,
    workspace: object,
    sm_scale: float,
    latent_scale: float,
    v_head_dim: int,
    attn_sink: torch.Tensor | None,
    return_lse: bool,
    lse_scale: Literal["base2", "natural"],
    scale_format: int | None = None,
    model_type: int | None = None,
    fp8_rope: bool | None = None,
    latent_scale_per_token: bool = False,
    cache_traits: object | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Route a prefill-like call to the active SM120 single-pass prefill."""
    from .kernel import run_unified_prefill

    output = _get_mla_output_view(
        workspace=workspace,
        q_all=q_all,
        v_head_dim=v_head_dim,
    )
    lse_out = _get_mla_lse_view(
        workspace=workspace,
        q_all=q_all,
    )
    _, lse_base2 = run_unified_prefill(
        q=q_all,
        kv_cache=kv_cache,
        topk_indices=selected_indices,
        sm_scale=float(sm_scale),
        latent_scale=float(latent_scale),
        page_block_size=int(workspace.page_size),
        topk_length=active_token_counts,
        attn_sink=attn_sink,
        output=output,
        lse_out=lse_out,
        scale_format=scale_format,
        model_type=model_type,
        fp8_rope=fp8_rope,
        latent_scale_per_token=latent_scale_per_token,
        traits_override=cache_traits,
    )
    if not return_lse:
        return output
    if lse_scale == "natural":
        lse_base2.mul_(_LN2)
    return output, lse_base2


def _final_lse_from_split_workspace(
    *,
    workspace: object,
    q_rows: int,
    num_heads: int,
    launch_num_chunks: int,
    scale: Literal["base2", "natural"] = "base2",
    _prepared: object | None = None,
    resolved_programs: list[object] | None = None,
) -> torch.Tensor:
    if workspace.tmp_lse is None:
        raise RuntimeError("workspace is missing split MLA LSE buffer")
    if workspace.final_lse is None:
        raise RuntimeError("workspace is missing final MLA LSE buffer")
    q_rows = int(q_rows)
    num_heads = int(num_heads)
    chunk_count = int(launch_num_chunks)
    if chunk_count <= 0:
        raise ValueError(f"launch_num_chunks must be positive, got {chunk_count}")
    if workspace.tmp_lse.ndim != 3:
        raise ValueError(
            f"workspace split MLA LSE buffer must be rank-3, got {tuple(workspace.tmp_lse.shape)}"
        )
    _validate_tensor_storage_bounds(
        workspace.tmp_lse, name="workspace split MLA LSE buffer"
    )
    if (
        int(workspace.tmp_lse.shape[0]) < q_rows
        or int(workspace.tmp_lse.shape[1]) < num_heads
        or int(workspace.tmp_lse.shape[2]) < chunk_count
    ):
        raise ValueError(
            "workspace split MLA LSE buffer is too small: "
            f"buffer={tuple(workspace.tmp_lse.shape)} required>=({q_rows}, {num_heads}, {chunk_count})"
        )
    if workspace.final_lse.ndim != 2:
        raise ValueError(
            f"workspace final MLA LSE buffer must be rank-2, got {tuple(workspace.final_lse.shape)}"
        )
    _validate_tensor_storage_bounds(
        workspace.final_lse, name="workspace final MLA LSE buffer"
    )
    if (
        int(workspace.final_lse.shape[0]) < q_rows
        or int(workspace.final_lse.shape[1]) < num_heads
    ):
        raise ValueError(
            "workspace final MLA LSE buffer is too small: "
            f"buffer={tuple(workspace.final_lse.shape)} required>=({q_rows}, {num_heads})"
        )
    final_lse = workspace.final_lse.as_strided(
        (q_rows, num_heads), workspace.final_lse.stride()
    )
    if final_lse.dtype != torch.float32:
        raise TypeError(
            f"workspace final MLA LSE buffer must be FP32, got {final_lse.dtype}"
        )
    chunk_lse = workspace.tmp_lse.as_strided(
        (q_rows, num_heads, chunk_count), workspace.tmp_lse.stride()
    )
    if chunk_lse.dtype != torch.float32:
        raise TypeError(
            f"workspace split MLA LSE buffer must be FP32, got {chunk_lse.dtype}"
        )
    if (
        triton is not None
        and chunk_lse.device.type == "cuda"
        and final_lse.device == chunk_lse.device
    ):
        block_c = triton.next_power_of_2(chunk_count)
        args = (
            chunk_lse,
            final_lse,
            chunk_lse.stride(0),
            chunk_lse.stride(1),
            chunk_lse.stride(2),
            final_lse.stride(0),
            final_lse.stride(1),
            chunk_count,
            block_c,
            scale == "natural",
        )
        grid = (q_rows, num_heads)
        if _prepared is not None:
            _prepared[(q_rows, num_heads, 1)](*args)
        else:
            launched = launch_triton(_split_decode_final_lse_kernel, grid, *args)
            if resolved_programs is not None:
                resolved_programs.append(launched)
        return final_lse
    chunk_lse.mul_(_LN2)
    torch.logsumexp(chunk_lse, dim=-1, out=final_lse)
    if scale == "natural":
        return final_lse
    final_lse.div_(_LN2)
    return final_lse


def _get_sm_scale_tensor(
    *,
    workspace: object,
    device: torch.device,
    sm_scale: float,
) -> torch.Tensor:
    sm_scale_tensor = workspace.sm_scale_tensor
    if (
        sm_scale_tensor is None
        or sm_scale_tensor.device != device
        or sm_scale_tensor.dtype != torch.float32
    ):
        sm_scale_tensor = torch.empty((1,), dtype=torch.float32, device=device)
        workspace.sm_scale_tensor = sm_scale_tensor
        workspace.sm_scale_value = None
    sm_scale_value = float(sm_scale)
    if workspace.sm_scale_value != sm_scale_value:
        sm_scale_tensor.fill_(sm_scale_value)
        workspace.sm_scale_value = sm_scale_value
    return sm_scale_tensor
