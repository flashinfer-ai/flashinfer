# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-scheduled paged MLA decode with a plan/run lifecycle."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import copy
import math
import struct
import functools
from typing import Any, Literal, NamedTuple, Optional, cast

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.attention import (
    prims_ts_decode_mla_one_shot_trace_dispatch,
    prims_ts_decode_mla_trace_dispatch,
    prims_ts_decode_mla_wrapper_trace_dispatch,
)

from .decode import (
    _WorkspaceSection,
    _align_up,
    _append_workspace_section,
    _dtype_key,
    _resolve_cuda_device,
    _validate_16byte_alignment,
    _validate_mask,
    _validate_page_size,
    _validate_positive_int,
    _validate_runtime_device,
    _validate_scale,
    _validate_workspace_buffer,
    _workspace_section_view,
)


from .kernels.mla_decode.sparse_policy import select_sparse_mla_profile

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_QUERY_DIM = _MLA_LATENT_DIM + _MLA_ROPE_DIM
_SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16,)
_INT32_MAX = 2**31 - 1
# The largest public 1CTA schedule can pad one K/V split group across 128
# splits, two 128-token K/V instructions apiece. Reserve that complete span so
# every padded tile boundary remains representable as signed Int32.
_MLA_MAX_KV_COORDINATE_SPAN = 128 * 2 * 128
_MLA_MAX_KV_LEN = _INT32_MAX - (_MLA_MAX_KV_COORDINATE_SPAN - 1)


@dataclass(frozen=True)
class _MLADecodeLaunchSpec:
    """Automatic MLA policy and scratch geometry for one plan."""

    kernel: Any
    policy: tuple[tuple[str, object], ...]
    kernel_workspace_bytes: int
    split_kv: int


@dataclass(frozen=True)
class _MLADecodeCompileSpec:
    """Batch-independent identity and implementation for one MLA compile."""

    device_index: int
    kernel_signature: tuple[object, ...]
    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype_key: str
    output_dtype_key: str
    max_seq_len_q: int
    packed_query: bool
    has_kernel_workspace: bool
    split_kv: int
    kernel: Any = field(compare=False, hash=False, repr=False)
    device_scales: bool = False


@dataclass(frozen=True)
class _MLAWorkspaceLayout:
    """Private MLA scratch layout; only ``total_bytes`` is public."""

    kernel_workspace: _WorkspaceSection
    lse: _WorkspaceSection
    total_bytes: int


@dataclass(frozen=True)
class _MLAWorkspaceViews:
    kernel_workspace: Optional[torch.Tensor]
    lse: torch.Tensor


@dataclass(frozen=True)
class _MLADecodePlanState:
    """Immutable compile and workspace state for one reusable MLA plan."""

    device: torch.device
    batch_size: int
    num_heads: int
    max_seq_len_q: int
    packed_query: bool
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: Literal["dense", "causal"]
    max_kv_len: int
    required_page_columns: int
    workspace_buffer: torch.Tensor
    workspace_layout: _MLAWorkspaceLayout
    workspace_views: _MLAWorkspaceViews
    compiled: Callable[..., object]
    policy: tuple[tuple[str, object], ...]
    split_kv: int


@dataclass(frozen=True)
class _MLARuntime:
    query: torch.Tensor
    normalized_cache: torch.Tensor
    out: torch.Tensor
    num_physical_pages: int
    bmm1_scale: float
    bmm2_scale: float
    extra_cache: Optional[torch.Tensor] = None
    scale_params: Optional[torch.Tensor] = None
    sparse_inputs: Optional[tuple] = None


def _make_mla_workspace_layout(
    kernel_workspace_bytes: int,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int = 1,
) -> _MLAWorkspaceLayout:
    kernel_workspace, byte_end = _append_workspace_section(
        0, (kernel_workspace_bytes,), torch.int8
    )
    lse, byte_end = _append_workspace_section(
        byte_end, (batch_size, max_seq_len_q, num_heads), torch.float32
    )
    return _MLAWorkspaceLayout(
        kernel_workspace=kernel_workspace,
        lse=lse,
        total_bytes=_align_up(byte_end),
    )


def _bind_mla_workspace(
    workspace_buffer: torch.Tensor, layout: _MLAWorkspaceLayout
) -> _MLAWorkspaceViews:
    kernel_workspace = None
    if layout.kernel_workspace.byte_size > 0:
        kernel_workspace = _workspace_section_view(
            workspace_buffer, layout.kernel_workspace
        )
    return _MLAWorkspaceViews(
        kernel_workspace=kernel_workspace,
        lse=_workspace_section_view(workspace_buffer, layout.lse),
    )


def _validate_mla_dims(kv_lora_rank: int, qk_rope_head_dim: int) -> None:
    kv_lora_rank = _validate_positive_int(kv_lora_rank, "kv_lora_rank")
    qk_rope_head_dim = _validate_positive_int(qk_rope_head_dim, "qk_rope_head_dim")
    if (kv_lora_rank, qk_rope_head_dim) != (_MLA_LATENT_DIM, _MLA_ROPE_DIM):
        raise NotImplementedError(
            "attention-ts MLA decode currently requires "
            f"kv_lora_rank={_MLA_LATENT_DIM} and "
            f"qk_rope_head_dim={_MLA_ROPE_DIM}; got "
            f"{kv_lora_rank} and {qk_rope_head_dim}"
        )


def _validate_mla_max_kv_len(value: int, name: str) -> int:
    """Reserve the largest padded split-KV coordinate span in signed Int32."""
    value = _validate_positive_int(value, name)
    if value > _MLA_MAX_KV_LEN:
        raise NotImplementedError(
            f"{name} must be <= {_MLA_MAX_KV_LEN} so padded MLA K/V "
            "coordinates fit in a signed int32"
        )
    return value


def _validate_mla_int32_extent(value: int, name: str) -> int:
    """Validate a flattened metadata/cache extent used by Int32 coordinates."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_mla_nonnegative_int32_extent(value: int, name: str) -> int:
    """Validate a possibly empty flattened extent used by Int32 coordinates."""
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_mla_query_head_extent(
    *,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    total_q: Optional[int] = None,
) -> None:
    """Keep fixed-capacity and packed query-head coordinates in signed Int32."""
    _validate_mla_int32_extent(
        batch_size * max_seq_len_q * num_heads,
        "batch_size * max_seq_len_q * num_heads",
    )
    if total_q is not None:
        _validate_mla_nonnegative_int32_extent(
            total_q * num_heads,
            "total_q * num_heads",
        )


def _validate_mla_policy_coordinate_span(
    policy: tuple[tuple[str, object], ...],
) -> None:
    """Keep the host K/V bound coupled to the automatically selected policy."""
    resolved = dict(policy)
    span = (
        int(cast(int, resolved["tile_size_kv"]))
        * int(cast(int, resolved["num_insts_kv"]))
        * max(int(cast(int, resolved["split_kv"])), 1)
    )
    if span > _MLA_MAX_KV_COORDINATE_SPAN:
        raise RuntimeError(
            "MLA Int32 extent safety assumes a padded K/V coordinate span no "
            f"larger than {_MLA_MAX_KV_COORDINATE_SPAN}, got {span}"
        )


def _validate_mla_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    _dtype_key(q_dtype)
    _dtype_key(kv_dtype)
    _dtype_key(output_dtype)
    if q_dtype != kv_dtype:
        raise NotImplementedError(
            "attention-ts MLA decode requires query and KV cache to use the "
            f"same dtype; got {q_dtype} and {kv_dtype}"
        )
    if q_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"attention-ts MLA decode supports BF16 and FP8-E4M3 input; got {q_dtype}"
        )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise NotImplementedError(
            "attention-ts MLA decode currently supports BF16 output only; "
            f"got {output_dtype}"
        )


def _validate_int32_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
    require_contiguous: bool = True,
    require_16byte_alignment: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got rank {tensor.ndim}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if require_16byte_alignment:
        _validate_16byte_alignment(tensor, name)


def _validate_mla_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.device, int, int]:
    _validate_int32_cuda_tensor(
        block_tables,
        "block_tables",
        ndim=2,
        require_contiguous=False,
        require_16byte_alignment=False,
    )
    _validate_int32_cuda_tensor(
        seq_lens,
        "seq_lens",
        ndim=1,
        require_16byte_alignment=False,
    )
    if block_tables.device != seq_lens.device:
        raise ValueError("block_tables and seq_lens must be on the same device")
    batch_size = int(seq_lens.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens must contain at least one request")
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables must have one row per request: expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    max_num_pages = int(block_tables.shape[1])
    if max_num_pages <= 0:
        raise ValueError("block_tables must contain at least one page column")
    if block_tables.stride(1) != 1:
        raise ValueError("block_tables must be contiguous within each row")
    if block_tables.stride(0) < max_num_pages:
        raise ValueError(
            "block_tables rows must not overlap: row stride must be at least "
            f"the column count ({max_num_pages}), got {block_tables.stride(0)}"
        )
    if block_tables.data_ptr() % 4 != 0:
        raise ValueError("block_tables must be 4-byte aligned")
    if seq_lens.data_ptr() % 4 != 0:
        raise ValueError("seq_lens must be 4-byte aligned")
    _validate_mla_int32_extent(batch_size, "batch_size")
    _validate_mla_int32_extent(int(block_tables.numel()), "block_tables elements")
    return seq_lens.device, batch_size, max_num_pages


def _validate_qo_indptr(
    qo_indptr: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> None:
    """Validate the public packed-query metadata without synchronizing."""

    _validate_int32_cuda_tensor(
        qo_indptr,
        "qo_indptr",
        ndim=1,
        require_16byte_alignment=False,
    )
    if qo_indptr.device != device:
        raise ValueError(f"qo_indptr must be on {device}, got {qo_indptr.device}")
    expected_offsets = batch_size + 1
    if qo_indptr.numel() != expected_offsets:
        raise ValueError(
            "qo_indptr must contain batch_size + 1 cumulative offsets: "
            f"expected {expected_offsets}, got {qo_indptr.numel()}"
        )


def _derive_max_seq_len_q(
    qo_indptr: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate cumulative offsets and derive their maximum nonnegative delta.

    This helper intentionally synchronizes and is therefore used only by
    explicit runtime validation and the one-shot convenience path.
    """

    offsets = [int(value) for value in qo_indptr.tolist()]
    if len(offsets) != batch_size + 1:
        raise ValueError("qo_indptr must contain batch_size + 1 offsets")
    if offsets[0] != 0:
        raise ValueError("qo_indptr must start at 0")
    q_lengths = tuple(
        end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(length < 0 for length in q_lengths):
        raise ValueError("qo_indptr must be nondecreasing")
    return max(q_lengths), offsets[-1], q_lengths


def _validate_mla_run_metadata(
    state: _MLADecodePlanState,
    runtime: _MLARuntime,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
) -> None:
    """Validate per-run request metadata against one static MLA plan.

    Value checks synchronize CUDA metadata with the host. ``run(validate=False)``
    is the synchronization-free path for compilation and graph capture.
    """

    device, batch_size, max_num_pages = _validate_mla_metadata(block_tables, seq_lens)
    if device != state.device:
        raise ValueError(
            f"MLA metadata must be on the planned device {state.device}, got {device}"
        )
    if batch_size != state.batch_size:
        raise ValueError(
            "MLA metadata batch size must match the plan "
            f"({state.batch_size}), got {batch_size}"
        )
    if max_num_pages < state.required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({state.required_page_columns}), got {max_num_pages}"
        )

    if state.packed_query:
        if qo_indptr is None:
            raise ValueError("packed-query MLA run requires qo_indptr")
        _validate_qo_indptr(
            qo_indptr,
            device=state.device,
            batch_size=state.batch_size,
        )
        runtime_max_seq_len_q, total_q, q_lengths = _derive_max_seq_len_q(
            qo_indptr,
            batch_size=state.batch_size,
        )
        if total_q != int(runtime.query.shape[0]):
            raise ValueError(
                "qo_indptr must end at the packed query row count "
                f"({runtime.query.shape[0]}), got {total_q}"
            )
        if runtime_max_seq_len_q > state.max_seq_len_q:
            raise ValueError(
                "qo_indptr contains a per-request Q length larger than "
                f"max_seq_len_q ({state.max_seq_len_q}): got "
                f"{runtime_max_seq_len_q}"
            )
    else:
        if qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        q_lengths = (state.max_seq_len_q,) * state.batch_size

    seq_lens_host = tuple(int(value) for value in seq_lens.tolist())
    if any(seq_len <= 0 for seq_len in seq_lens_host):
        raise ValueError("every runtime request must contain at least one KV token")
    runtime_max_kv_len = max(seq_lens_host)
    if runtime_max_kv_len > state.max_kv_len:
        raise ValueError(
            "runtime KV metadata contains a request longer than "
            f"max_kv_len ({state.max_kv_len}): got {runtime_max_kv_len}"
        )
    block_table_rows = block_tables.tolist()
    for request_idx, (row, seq_len) in enumerate(
        zip(block_table_rows, seq_lens_host, strict=True)
    ):
        required_pages = _ceil_div(seq_len, state.page_size)
        if any(
            int(page_id) < 0 or int(page_id) >= runtime.num_physical_pages
            for page_id in row[:required_pages]
        ):
            raise ValueError(
                "block_tables values for active pages must index the physical "
                f"K/V cache in [0, {runtime.num_physical_pages}); request "
                f"{request_idx} contains an invalid page ID"
            )
    if state.mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(q_lengths, seq_lens_host, strict=True)
        ):
            if q_len > kv_len:
                raise ValueError(
                    "causal MLA decode requires every per-request Q length "
                    "to be no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )


def _resolve_max_seq_len_q_alias(
    *,
    seq_len_q: Optional[int],
    max_seq_len_q: Optional[int],
    default: Optional[int],
) -> Optional[int]:
    """Resolve the legacy fixed-Q name and the explicit static-bound name."""

    legacy_bound = (
        _validate_positive_int(seq_len_q, "seq_len_q")
        if seq_len_q is not None
        else None
    )
    explicit_bound = (
        _validate_positive_int(max_seq_len_q, "max_seq_len_q")
        if max_seq_len_q is not None
        else None
    )
    if (
        legacy_bound is not None
        and explicit_bound is not None
        and legacy_bound != explicit_bound
    ):
        raise ValueError(
            "seq_len_q and max_seq_len_q must agree when both are provided: "
            f"got {legacy_bound} and {explicit_bound}"
        )
    if explicit_bound is not None:
        return explicit_bound
    if legacy_bound is not None:
        return legacy_bound
    return default


def _validate_query(
    query: torch.Tensor,
    *,
    packed_query: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_heads: Optional[int] = None,
    max_seq_len_q: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
    query_dim: int = _MLA_QUERY_DIM,
) -> None:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    expected_rank = 3 if packed_query else 4
    if query.ndim != expected_rank:
        expected_shape = (
            f"[total_q, H, {query_dim}]" if packed_query else f"[B, SQ, H, {query_dim}]"
        )
        raise ValueError(f"query must have shape {expected_shape}")
    if packed_query:
        if int(query.shape[1]) <= 0:
            raise ValueError("query head extent must be positive")
    elif any(int(extent) <= 0 for extent in query.shape[:-1]):
        raise ValueError("query row and head extents must be positive")
    if query.shape[-1] != query_dim:
        raise ValueError(
            f"query last dimension must be {query_dim}, got {query.shape[-1]}"
        )
    if query.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"unsupported attention-ts MLA query dtype {query.dtype}"
        )
    if query.device.type != "cuda":
        raise ValueError("query must be a CUDA tensor")
    if device is not None and query.device != device:
        raise ValueError(
            f"query must be on the planned device {device}, got {query.device}"
        )
    if not packed_query and batch_size is not None and query.shape[0] != batch_size:
        raise ValueError(
            f"query batch size must match the plan ({batch_size}), got {query.shape[0]}"
        )
    head_axis = 1 if packed_query else 2
    if num_heads is not None and query.shape[head_axis] != num_heads:
        raise ValueError(
            "query head count must match the plan "
            f"({num_heads}), got {query.shape[head_axis]}"
        )
    if max_seq_len_q is not None:
        if packed_query:
            if batch_size is None:
                raise ValueError("batch_size is required to validate packed query")
            total_q = int(query.shape[0])
            if total_q > batch_size * max_seq_len_q:
                raise ValueError(
                    "packed query total rows must be within "
                    f"[0, {batch_size * max_seq_len_q}], got {total_q}"
                )
        elif query.shape[1] != max_seq_len_q:
            raise ValueError(
                "fixed query length must equal the planned max_seq_len_q "
                f"({max_seq_len_q}), got {query.shape[1]}"
            )
        if batch_size is not None and num_heads is not None:
            _validate_mla_query_head_extent(
                batch_size=batch_size,
                num_heads=num_heads,
                max_seq_len_q=max_seq_len_q,
                total_q=int(query.shape[0]) if packed_query else None,
            )
    if q_dtype is not None and query.dtype != q_dtype:
        raise ValueError(
            f"query dtype must match the plan ({q_dtype}), got {query.dtype}"
        )
    if not query.is_contiguous():
        layout = (
            f"[total_q, H, {query_dim}]" if packed_query else f"[B, SQ, H, {query_dim}]"
        )
        raise ValueError(f"query must be compact in {layout} layout")
    _validate_16byte_alignment(query, "query")


def _normalize_mla_kv_cache(
    kv_cache: torch.Tensor,
    *,
    expected_device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    if not isinstance(kv_cache, torch.Tensor):
        raise TypeError("kv_cache must be a torch.Tensor")
    if kv_cache.ndim == 4:
        if kv_cache.shape[1] != 1:
            raise ValueError(
                "rank-4 kv_cache must have shape [num_pages, 1, page_size, 576]"
            )
        if not kv_cache.is_contiguous():
            raise ValueError("rank-4 kv_cache must be compact")
        normalized = kv_cache[:, 0]
    elif kv_cache.ndim == 3:
        if not kv_cache.is_contiguous():
            raise ValueError("rank-3 kv_cache must be compact")
        normalized = kv_cache
    else:
        raise ValueError(
            "kv_cache must have shape [num_pages, page_size, 576] or "
            "[num_pages, 1, page_size, 576]"
        )
    if normalized.device != expected_device:
        raise ValueError(
            f"kv_cache must be on the planned device {expected_device}, "
            f"got {normalized.device}"
        )
    if normalized.shape[0] <= 0 or normalized.shape[1] <= 0:
        raise ValueError("kv_cache page count and page size must be positive")
    _validate_mla_int32_extent(int(normalized.shape[0]), "kv_cache physical pages")
    if normalized.shape[2] != _MLA_QUERY_DIM:
        raise ValueError(
            f"kv_cache last dimension must be {_MLA_QUERY_DIM}, "
            f"got {normalized.shape[2]}"
        )
    _validate_16byte_alignment(normalized, "kv_cache")
    return normalized, int(normalized.shape[0]), int(normalized.shape[1])


def _validate_out(
    out: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    packed_query: bool,
    total_q: Optional[int] = None,
    output_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if packed_query:
        if total_q is None:
            raise ValueError("total_q is required to validate packed output")
        expected_shape: tuple[int, ...]
        expected_shape = (total_q, num_heads, _MLA_LATENT_DIM)
    else:
        expected_shape = (batch_size, max_seq_len_q, num_heads, _MLA_LATENT_DIM)
    if out.shape != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != output_dtype:
        raise ValueError(f"out must have dtype {output_dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        layout = "[total_q, H, 512]" if packed_query else "[B, SQ, H, 512]"
        raise ValueError(f"out must be compact in {layout} layout")
    _validate_16byte_alignment(out, "out")


def _kernel_dtype_name(dtype_key: str) -> str:
    names = {
        "bfloat16": "bf16",
        "float8_e4m3fn": "e4m3",
    }
    try:
        return names[dtype_key]
    except KeyError as error:
        raise NotImplementedError(
            f"unsupported attention-ts MLA dtype key {dtype_key!r}"
        ) from error


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _separate_reducer_provenance(
    kernel,
    *,
    split_kv: int,
    use_cluster_reduction: bool,
) -> tuple[str, Optional[int]]:
    """Describe the derived standalone reducer without exposing a knob."""

    if split_kv <= 1 or use_cluster_reduction:
        return "none", None
    if bool(getattr(kernel, "use_parallel_reduction", False)):
        topology = getattr(kernel, "parallel_reduction_topology", None)
        if topology is None:
            raise RuntimeError("parallel MLA reducer is missing its topology")
        return "parallel", int(topology.cluster_size)
    return "reference", 1


@functools.cache
def _resolve_mla_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    seq_len_q: int = 1,
):
    """Resolve and cache MLA policy/workspace without compiling."""

    max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")

    import cutlass
    import cutlass.utils as cutlass_utils
    from cuda.bindings import driver as cuda_drv

    from .kernels.mla_decode.kernel_policy import (
        resolve_mla_kernel_policy,
        select_mla_ts_kernel,
    )
    from .kernels.mla_decode.helpers.query import FlatQueryTileLayout
    from .kernels.mla_decode.throughput_2cta.config import (
        compute_split_kv,
        compute_workspace_size as compute_2cta_workspace_size,
    )
    from .kernels.mla_decode.throughput_2cta.kernel import MlaDecodeTs
    from .kernels.mla_decode.throughput_latency_1cta.config import (
        compute_workspace_size as compute_1cta_workspace_size,
        q_tile_work_count,
        resolve_auto_flat_query_launch_shape,
        resolve_runtime_cluster_reduction_mode,
        select_auto_split_kv,
    )
    from .kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )

    if q_dtype_key != kv_dtype_key:
        raise ValueError("the cached TS MLA compiler requires one QKV dtype")
    seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=seq_len_q,
    )
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    qkv_dtype_name = _kernel_dtype_name(q_dtype_key)
    output_dtype_name = _kernel_dtype_name(output_dtype_key)

    with torch.cuda.device(device_index):
        plan_stream = cuda_drv.CUstream(
            torch.cuda.current_stream(device_index).cuda_stream
        )
        hardware_info = cutlass_utils.HardwareInfo(device_index)
        max_active_one_cta_clusters = hardware_info.get_max_active_clusters(
            1, plan_stream
        )
        max_active_two_cta_clusters = hardware_info.get_max_active_clusters(
            2, plan_stream
        )
        one_cta_launch_shape = resolve_auto_flat_query_launch_shape(
            num_heads_q=num_heads,
            seq_len_q=seq_len_q,
        )
        one_cta_base_work = q_tile_work_count(
            batch_size,
            one_cta_launch_shape.num_heads_q,
            one_cta_launch_shape.seq_len_q,
            one_cta_launch_shape.tile_size_q,
        )
        one_cta_split_kv = select_auto_split_kv(
            seq_len_kv=max_kv_len,
            tile_size_q=one_cta_launch_shape.tile_size_q,
            base_work=one_cta_base_work,
            target_work=max_active_one_cta_clusters,
        )
        two_cta_launch_shape = FlatQueryTileLayout.for_tile(num_heads, seq_len_q, 128)
        two_cta_split_kv = compute_split_kv(
            batch_size=batch_size,
            num_q_tiles=two_cta_launch_shape.num_tiles,
            seq_len_kv=max_kv_len,
            mma_qk_tiler_mn=(128, 128),
            max_active_blocks=max_active_two_cta_clusters * 2,
        )
        requested_policy, policy_source = resolve_mla_kernel_policy(
            None,
            num_heads,
            seq_len_q,
            one_cta_split_kv=one_cta_split_kv,
            two_cta_split_kv=two_cta_split_kv,
        )
        use_throughput_latency = requested_policy == "throughput_latency_1cta"

        kernel: Any
        if use_throughput_latency:
            max_active_clusters = max_active_one_cta_clusters
            launch_shape = one_cta_launch_shape
            decision = select_mla_ts_kernel(
                requested_policy=requested_policy,
                batch_size=batch_size,
                num_heads=launch_shape.num_heads_q,
                seq_len_q=launch_shape.seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=launch_shape.tile_size_q,
                max_active_clusters=max_active_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if not decision.implementation_ready or decision.config is None:
                if policy_source != "auto":
                    raise NotImplementedError(decision.reason)
                requested_policy = "throughput_2cta"
                use_throughput_latency = False

        if use_throughput_latency:
            assert decision.config is not None
            reduction_mode = resolve_runtime_cluster_reduction_mode(
                decision.config,
                reduction_mode=None,
                hardware_info=hardware_info,
                stream=plan_stream,
            )
            kernel = ThroughputLatencyMlaDecodeTs(
                batch_size=batch_size,
                num_heads=launch_shape.num_heads_q,
                seq_len_q=launch_shape.seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                max_active_clusters=max_active_clusters,
                acc_dtype=cutlass.Float32,
                lse_dtype=cutlass.Float32,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                profile=decision.profile_name,
                reduction_mode=reduction_mode,
                logical_num_heads=num_heads,
                logical_seq_len_q=seq_len_q,
                tile_size_q=launch_shape.tile_size_q,
                explicit_split_kv=None,
                explicit_persistent=None,
                mask_type=mask_type,
            )
            final_cfg = kernel._make_config()
            split_kv = int(final_cfg.num_ctas_per_seq_kv)
            workspace_size = compute_1cta_workspace_size(
                cfg=final_cfg,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = _separate_reducer_provenance(
                kernel,
                split_kv=split_kv,
                use_cluster_reduction=bool(final_cfg.use_cluster_reduction),
            )
            policy = (
                ("kernel", decision.selected_kernel),
                ("source", policy_source),
                ("profile", decision.profile_name),
                ("tile_size_q", int(final_cfg.tile_size_q)),
                ("tile_size_kv", int(final_cfg.tile_size_kv)),
                ("num_insts_kv", int(final_cfg.num_insts_kv)),
                ("split_kv", split_kv),
                ("num_ctas_per_head_dim", int(final_cfg.num_ctas_per_head_dim)),
                ("head_dim_per_cta_v", int(final_cfg.head_dim_per_cta_v)),
                ("use_cluster_reduction", bool(final_cfg.use_cluster_reduction)),
                (
                    "use_persistent_scheduler",
                    bool(final_cfg.use_persistent_scheduler),
                ),
                (
                    "use_clc_dynamic_persistent_scheduler",
                    bool(final_cfg.use_clc_dynamic_persistent_scheduler),
                ),
                ("separate_reducer_impl", separate_reducer_impl),
                ("reducer_cluster_size", reducer_cluster_size),
            )
        else:
            max_active_clusters = max_active_two_cta_clusters
            launch_shape = two_cta_launch_shape
            decision = select_mla_ts_kernel(
                requested_policy=requested_policy,
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                seq_len_k=max_kv_len,
                latent_dim=kv_lora_rank,
                rope_dim=qk_rope_head_dim,
                page_size=page_size,
                dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                throughput_latency_profile=None,
                throughput_latency_tile_size_q=None,
                max_active_clusters=max_active_clusters,
                throughput_latency_split_kv=None,
                throughput_latency_persistent=None,
            )
            if not decision.implementation_ready:
                raise NotImplementedError(decision.reason)
            split_kv = two_cta_split_kv
            work_clusters = batch_size * launch_shape.num_tiles * max(split_kv, 1)
            # Dynamic cluster stealing only helps once logical work exceeds a
            # resident wave.  Within one wave every cluster already launches,
            # so the CLC producer/response pipeline is pure overhead.
            is_persistent = work_clusters > max_active_clusters
            kernel = MlaDecodeTs(
                acc_dtype=cutlass.Float32,
                lse_dtype=cutlass.Float32,
                mma_qk_tiler_mn=(128, 128),
                mma_pv_tiler_mn=(128, 256),
                max_active_clusters=max_active_clusters,
                page_size=page_size,
                is_persistent=is_persistent,
                is_var_seq=False,
                is_var_split_kv=False,
                static_split_kv=split_kv,
                static_seq_len_k=None,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                rope_dim=qk_rope_head_dim,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                batch_size=batch_size,
                mask_type=mask_type,
            )
            workspace_size = compute_2cta_workspace_size(
                tile_size_q=int(launch_shape.tile_size_q),
                num_q_tiles=int(launch_shape.num_tiles),
                latent_dim=kv_lora_rank,
                batch_size=batch_size,
                split_kv=split_kv,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = _separate_reducer_provenance(
                kernel,
                split_kv=split_kv,
                use_cluster_reduction=False,
            )
            policy = (
                ("kernel", decision.selected_kernel),
                ("source", policy_source),
                ("profile", None),
                ("tile_size_q", 128),
                ("tile_size_kv", 128),
                ("num_insts_kv", 1),
                ("split_kv", int(split_kv)),
                ("num_ctas_per_head_dim", 2),
                ("head_dim_per_cta_v", 256),
                ("use_cluster_reduction", False),
                ("use_persistent_scheduler", bool(is_persistent)),
                (
                    "use_clc_dynamic_persistent_scheduler",
                    bool(is_persistent and qkv_dtype_name == "bf16"),
                ),
                ("separate_reducer_impl", separate_reducer_impl),
                ("reducer_cluster_size", reducer_cluster_size),
            )
        _validate_mla_policy_coordinate_span(policy)

    return _MLADecodeLaunchSpec(
        kernel=kernel,
        policy=policy,
        kernel_workspace_bytes=int(workspace_size),
        split_kv=int(split_kv),
    )


def _mla_kernel_compile_signature(kernel: Any) -> tuple[object, ...]:
    """Return all static kernel state except the batch extent."""

    make_signature = getattr(kernel, "compile_signature", None)
    if not callable(make_signature):
        raise TypeError("MLA kernels must define compile_signature()")
    signature = (
        type(kernel).__module__,
        type(kernel).__qualname__,
        make_signature(),
    )
    try:
        hash(signature)
    except TypeError as error:
        raise TypeError("MLA kernel compile state must be hashable") from error
    return signature


def _make_mla_decode_compile_spec(
    launch_spec: _MLADecodeLaunchSpec,
    *,
    device_index: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    q_dtype_key: str,
    output_dtype_key: str,
    max_seq_len_q: int,
    packed_query: bool,
    device_scales: bool = False,
) -> _MLADecodeCompileSpec:
    """Keep policy resolution plan-specific and JIT identity batch-free."""

    return _MLADecodeCompileSpec(
        device_index=device_index,
        kernel_signature=_mla_kernel_compile_signature(launch_spec.kernel),
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        q_dtype_key=q_dtype_key,
        output_dtype_key=output_dtype_key,
        max_seq_len_q=max_seq_len_q,
        packed_query=packed_query,
        has_kernel_workspace=launch_spec.kernel_workspace_bytes > 0,
        split_kv=launch_spec.split_kv,
        kernel=launch_spec.kernel,
        device_scales=device_scales,
    )


@functools.cache
def _get_compiled_mla_decode(
    compile_spec: _MLADecodeCompileSpec,
):
    """Compile and cache one batch-dynamic TS MLA topology."""

    import cutlass
    import cutlass.cute as cute

    device_index = compile_spec.device_index
    num_heads = compile_spec.num_heads
    kv_lora_rank = compile_spec.kv_lora_rank
    qk_rope_head_dim = compile_spec.qk_rope_head_dim
    query_dim = kv_lora_rank + qk_rope_head_dim
    # The D512 path has no separate RoPE stage. Use an unused full-width
    # alias in the callable ABI rather than a zero-extent tensor map.
    rope_view_dim = qk_rope_head_dim or kv_lora_rank
    page_size = compile_spec.page_size
    max_seq_len_q = compile_spec.max_seq_len_q
    packed_query = compile_spec.packed_query
    kernel = compile_spec.kernel
    dtype_map = {
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    qkv_dtype = dtype_map[compile_spec.q_dtype_key]
    output_dtype = dtype_map[compile_spec.output_dtype_key]
    physical_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_total_q = cute.sym_int()

    # These fake tensors pin the public ABI while allowing runtime page counts,
    # table widths/row strides, and batch metadata pointers to vary.
    q_stride_h = query_dim
    q_stride_q = num_heads * query_dim
    q_latent_shape: tuple[int, ...]
    q_rope_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    if packed_query:
        q_latent_shape = (num_heads, kv_lora_rank, runtime_total_q)
        q_rope_shape = (num_heads, rope_view_dim, runtime_total_q)
        q_stride = (q_stride_h, 1, q_stride_q)
    else:
        q_stride_batch = max_seq_len_q * q_stride_q
        q_latent_shape = (num_heads, kv_lora_rank, max_seq_len_q, batch_size)
        q_rope_shape = (
            num_heads,
            rope_view_dim,
            max_seq_len_q,
            batch_size,
        )
        q_stride = (q_stride_h, 1, q_stride_q, q_stride_batch)
    q_latent_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_latent_shape, stride=q_stride, assumed_align=16
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_rope_shape, stride=q_stride, assumed_align=16
    )
    cache_token_stride = query_dim
    cache_page_stride = page_size * query_dim
    c_latent_fake = cute.runtime.make_fake_tensor(
        qkv_dtype,
        (page_size, kv_lora_rank, physical_pages),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        qkv_dtype,
        (
            page_size,
            rope_view_dim,
            cute.sym_int() if page_size == 1 else physical_pages,
        ),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    runtime_page_columns = cute.sym_int()
    runtime_page_row_stride = cute.sym_int64(divisibility=1)
    page_offsets_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (runtime_page_columns, batch_size),
        stride=(1, runtime_page_row_stride),
        assumed_align=4,
    )
    out_stride_row = num_heads * kv_lora_rank
    out_shape: tuple[int, ...]
    out_stride: tuple[int, ...]
    lse_shape: tuple[int, ...]
    lse_stride: tuple[int, ...]
    if packed_query:
        out_shape = (num_heads, kv_lora_rank, runtime_total_q)
        out_stride = (kv_lora_rank, 1, out_stride_row)
        lse_shape = (num_heads, runtime_total_q)
        lse_stride = (1, num_heads)
    else:
        out_stride_batch = max_seq_len_q * out_stride_row
        out_shape = (num_heads, kv_lora_rank, max_seq_len_q, batch_size)
        out_stride = (kv_lora_rank, 1, out_stride_row, out_stride_batch)
        lse_shape = (num_heads, max_seq_len_q, batch_size)
        lse_stride = (1, num_heads, max_seq_len_q * num_heads)
    out_fake = cute.runtime.make_fake_tensor(
        output_dtype, out_shape, stride=out_stride, assumed_align=16
    )
    lse_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, lse_shape, stride=lse_stride, assumed_align=16
    )
    workspace_fake = None
    if compile_spec.has_kernel_workspace:
        workspace_bytes = cute.sym_int()
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (workspace_bytes,),
            stride_order=(0,),
            assumed_align=32,
        )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (batch_size,),
        stride_order=(0,),
        assumed_align=4,
    )
    qo_indptr_fake = None
    if packed_query:
        runtime_num_q_offsets = cute.sym_int()
        qo_indptr_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (runtime_num_q_offsets,),
            stride_order=(0,),
            assumed_align=4,
        )
    scale_params_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        if compile_spec.device_scales
        else None
    )
    if getattr(kernel, "direct_sparse", False):

        def sparse_indices(capacity):
            return cute.runtime.make_fake_tensor(
                cutlass.Int32,
                (batch_size, max(1, capacity)),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            )

        scalar = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
        )
        sink = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_heads,), stride_order=(0,), assumed_align=4
        )
        scale_params_fake = (
            sparse_indices(kernel.direct_sparse_capacities[0]),
            sparse_indices(kernel.direct_sparse_capacities[1]),
            cache_seqs_fake,
            cache_seqs_fake,
            scalar,
            scalar,
            scalar,
            scalar,
            sink,
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    with torch.cuda.device(device_index):
        compiled = cute.compile[cute.FrontendNext](
            kernel,
            q_latent_fake,
            q_rope_fake,
            c_latent_fake,
            c_rope_fake,
            page_offsets_fake,
            out_fake,
            lse_fake,
            workspace_fake,
            cutlass.Int32(compile_spec.split_kv),
            cache_seqs_fake,
            qo_indptr_fake,
            scale_params_fake,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            stream_fake,
            options=_COMPILE_OPTIONS,
        )
    return compiled


def get_prims_ts_batch_mla_decode_workspace_size(
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_seq_len: int,
    *,
    seq_len_q: Optional[int] = None,
    max_seq_len_q: Optional[int] = None,
    q_dtype: torch.dtype = torch.bfloat16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    mask_type: Literal["dense", "causal"] = "causal",
    device=None,
) -> int:
    """Return caller-workspace bytes for one automatic MLA policy.

    The arguments define the static geometry used to resolve the same automatic
    policy and private scratch layout as
    :func:`prims_ts_batch_mla_decode_with_kv_cache`, without compiling a kernel.
    ``max_seq_len_q`` is the static per-request Q bound for both fixed and
    packed-query launches;
    ``seq_len_q`` remains a backward-compatible fixed-Q alias. If neither is
    supplied, the bound is one. The returned byte count includes both split-KV
    scratch and the internal FP32 LSE tensor. Allocate a contiguous
    ``torch.int8`` or ``torch.uint8`` CUDA buffer; MLA does not require its
    contents to be initialized before first use.
    """

    batch_size = _validate_positive_int(batch_size, "batch_size")
    num_heads = _validate_positive_int(num_heads, "num_heads")
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    page_size = _validate_page_size(page_size)
    max_seq_len = _validate_mla_max_kv_len(max_seq_len, "max_seq_len")
    max_seq_len_q = _resolve_max_seq_len_q_alias(
        seq_len_q=seq_len_q,
        max_seq_len_q=max_seq_len_q,
        default=1,
    )
    assert max_seq_len_q is not None
    _validate_mla_query_head_extent(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=max_seq_len_q,
    )
    _validate_mask(mask_type)
    if kv_dtype is None:
        kv_dtype = q_dtype
    _validate_mla_dtype_pair(q_dtype, kv_dtype, out_dtype)
    _, device_index = _resolve_cuda_device(device)

    spec = _resolve_mla_decode_launch_spec(
        device_index,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        _dtype_key(q_dtype),
        _dtype_key(kv_dtype),
        _dtype_key(out_dtype),
        mask_type,
        max_seq_len_q,
    )
    return _make_mla_workspace_layout(
        spec.kernel_workspace_bytes, batch_size, num_heads, max_seq_len_q
    ).total_bytes


def _prepare_mla_runtime(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_heads: int,
    max_seq_len_q: int,
    packed_query: bool,
    qo_indptr: Optional[torch.Tensor],
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    bmm1_scale: float,
    bmm2_scale: float,
    out: Optional[torch.Tensor],
    validate: bool,
) -> _MLARuntime:
    """Normalize one launch, optionally validating its public arguments."""

    if validate:
        if packed_query:
            if qo_indptr is None:
                raise ValueError("packed-query MLA run requires qo_indptr")
            _validate_qo_indptr(
                qo_indptr,
                device=device,
                batch_size=batch_size,
            )
        elif qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        _validate_query(
            query,
            packed_query=packed_query,
            device=device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            q_dtype=q_dtype,
        )
        (
            normalized_cache,
            num_physical_pages,
            runtime_page_size,
        ) = _normalize_mla_kv_cache(kv_cache, expected_device=device)
        if runtime_page_size != page_size:
            raise ValueError(
                "kv_cache page size does not match the launch: expected "
                f"{page_size}, got {runtime_page_size}"
            )
        if normalized_cache.dtype != kv_dtype:
            raise ValueError(
                f"kv_cache dtype must match the launch ({kv_dtype}), "
                f"got {normalized_cache.dtype}"
            )
        effective_bmm1_scale = _validate_scale(bmm1_scale, "bmm1_scale")
        effective_bmm2_scale = _validate_scale(bmm2_scale, "bmm2_scale")
    else:
        normalized_cache = kv_cache[:, 0] if kv_cache.ndim == 4 else kv_cache
        num_physical_pages = int(normalized_cache.shape[0])
        effective_bmm1_scale = bmm1_scale
        effective_bmm2_scale = bmm2_scale
    total_q = int(query.shape[0]) if packed_query else None
    if out is None:
        out_shape = (
            (total_q, num_heads, _MLA_LATENT_DIM)
            if packed_query
            else (batch_size, max_seq_len_q, num_heads, _MLA_LATENT_DIM)
        )
        out = torch.empty(out_shape, device=device, dtype=output_dtype)
    elif validate:
        _validate_out(
            out,
            device=device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            total_q=total_q,
            output_dtype=output_dtype,
        )
    return _MLARuntime(
        query=query,
        normalized_cache=normalized_cache,
        out=out,
        num_physical_pages=num_physical_pages,
        bmm1_scale=effective_bmm1_scale,
        bmm2_scale=effective_bmm2_scale,
    )


def _launch_mla_decode(
    runtime: _MLARuntime,
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    packed_query: bool,
    kv_lora_rank: int,
    split_kv: int,
    workspace: _MLAWorkspaceViews,
    compiled: Callable[..., object],
) -> torch.Tensor:
    """Form the dimension-first views and launch one compiled MLA kernel."""

    if packed_query and int(runtime.query.shape[0]) == 0:
        return runtime.out
    if packed_query:
        q_latent = runtime.query[..., :kv_lora_rank].permute(1, 2, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(1, 2, 0)
        out_kernel = runtime.out.permute(1, 2, 0)
        total_q = int(runtime.query.shape[0])
        lse_kernel = workspace.lse.view(-1, workspace.lse.shape[-1])[
            :total_q
        ].transpose(0, 1)
    else:
        q_latent = runtime.query[..., :kv_lora_rank].permute(2, 3, 1, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(2, 3, 1, 0)
        out_kernel = runtime.out.permute(2, 3, 1, 0)
        lse_kernel = workspace.lse.permute(2, 1, 0)
    c_latent = runtime.normalized_cache[..., :kv_lora_rank].permute(1, 2, 0)
    c_rope = runtime.normalized_cache[..., kv_lora_rank:].permute(1, 2, 0)
    if runtime.query.shape[-1] == kv_lora_rank:
        q_rope = q_latent
        c_rope = c_latent
        if runtime.extra_cache is not None:
            c_rope = runtime.extra_cache.permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)
    compiled(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        page_offsets,
        out_kernel,
        lse_kernel,
        workspace.kernel_workspace,
        split_kv,
        seq_lens,
        qo_indptr,
        runtime.sparse_inputs
        if runtime.sparse_inputs is not None
        else runtime.scale_params,
        runtime.bmm1_scale,
        runtime.bmm2_scale,
    )
    return runtime.out


@flashinfer_experimental_api(trace=prims_ts_decode_mla_trace_dispatch)
def prims_ts_batch_mla_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    mask_type: Literal["dense", "causal"] = "causal",
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch fixed or packed-query paged MLA decode with caller-owned scratch.

    With ``qo_indptr=None``, ``query`` has fixed shape ``[B, SQ, H, 576]``.
    Otherwise ``query`` has compact shape ``[total_q, H, 576]`` and
    ``qo_indptr`` contains the ``B + 1`` cumulative Q offsets. Runtime Q
    lengths are exclusively ``qo_indptr[b + 1] - qo_indptr[b]``;
    ``max_seq_len_q`` is only the static policy, JIT, and workspace bound and
    is required for compact launches. Individual packed requests may be empty,
    and an all-empty launch returns its empty output without dispatching a GPU
    kernel. The last query dimension concatenates the 512 latent and 64 RoPE
    dimensions. ``kv_cache`` accepts compact rank-3
    ``[pages, page_size, 576]`` or rank-4 ``[pages, 1, page_size, 576]``
    storage. ``block_tables`` and ``seq_lens`` follow FlashInfer's native dense
    paged-cache ABI. The table is contiguous within each row and may have
    padding between rows; ``max_seq_len`` is the exact static policy/JIT
    maximum.
    Causal masking is bottom-right aligned: query row ``i`` can attend through
    KV row ``seq_lens[b] - q_len[b] + i`` for request ``b``.

    The workspace is exclusive to one in-flight launch or captured graph and
    must not overlap query, K/V cache, metadata, or output storage. Output must
    also be disjoint from every input; storage overlap is not checked.
    Runtime K/V lengths must remain positive and no larger than ``max_seq_len``;
    this hot path deliberately performs no device-to-host metadata reads. For
    packed launches, callers must ensure that offsets start at zero, are
    nondecreasing, end at ``query.shape[0]``, and have every delta no
    larger than ``max_seq_len_q``. For causal masking, every fixed or packed
    per-request Q length must also be no greater than the corresponding live
    ``seq_lens`` value. Warm the planned topology before CUDA graph
    capture and provide ``out`` to avoid an output allocation. Captured graphs
    must retain stable ``block_tables``, ``seq_lens``, and, for packed Q,
    ``qo_indptr`` storage. Values may change only between completed replays
    while the runtime metadata contracts and captured query/output extents
    remain valid. No backend fallback or scheduling knob is exposed.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace for this planned layout.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    block_tables : torch.Tensor
        Dense physical-page table for each request. Rows must be inner
        contiguous and non-overlapping, but may have padding between them.
    seq_lens : torch.Tensor
        Live K/V sequence lengths.
    max_seq_len : int
        Static maximum K/V length used for policy selection and JIT caching.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Static packed-query length bound.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    bmm1_scale, bmm2_scale : float
        QK and value/output scaling factors.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    out_dtype : torch.dtype
        Output dtype.
    """

    packed_query = qo_indptr is not None
    _validate_query(query, packed_query=packed_query)
    metadata_device, batch_size, max_num_pages = _validate_mla_metadata(
        block_tables, seq_lens
    )
    if metadata_device != query.device:
        raise ValueError(
            f"MLA metadata must be on {query.device}, got {metadata_device}"
        )
    normalized_cache, _, page_size = _normalize_mla_kv_cache(
        kv_cache, expected_device=query.device
    )
    if packed_query:
        _validate_qo_indptr(
            qo_indptr,
            device=query.device,
            batch_size=batch_size,
        )
        if max_seq_len_q is None:
            raise ValueError(
                "max_seq_len_q is required when qo_indptr selects packed query"
            )
        max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
        num_heads = int(query.shape[1])
    else:
        fixed_seq_len_q = int(query.shape[1])
        if max_seq_len_q is None:
            max_seq_len_q = fixed_seq_len_q
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            if max_seq_len_q != fixed_seq_len_q:
                raise ValueError(
                    "fixed query length must equal max_seq_len_q: "
                    f"got SQ={fixed_seq_len_q} and max_seq_len_q={max_seq_len_q}"
                )
        num_heads = int(query.shape[2])
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    _validate_page_size(page_size)
    max_seq_len = _validate_mla_max_kv_len(max_seq_len, "max_seq_len")
    required_page_columns = _ceil_div(max_seq_len, page_size)
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_seq_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )
    _validate_mask(mask_type)
    _validate_mla_dtype_pair(query.dtype, normalized_cache.dtype, out_dtype)
    device_index = _validate_runtime_device(query.device)
    spec_key = (
        device_index,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        _dtype_key(query.dtype),
        _dtype_key(normalized_cache.dtype),
        _dtype_key(out_dtype),
        mask_type,
        max_seq_len_q,
    )
    spec = _resolve_mla_decode_launch_spec(*spec_key)
    layout = _make_mla_workspace_layout(
        spec.kernel_workspace_bytes, batch_size, num_heads, max_seq_len_q
    )
    _validate_workspace_buffer(
        workspace_buffer,
        device=query.device,
        required_bytes=layout.total_bytes,
    )
    runtime = _prepare_mla_runtime(
        query,
        normalized_cache,
        device=query.device,
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len_q=max_seq_len_q,
        packed_query=packed_query,
        qo_indptr=qo_indptr,
        page_size=page_size,
        q_dtype=query.dtype,
        kv_dtype=normalized_cache.dtype,
        output_dtype=out_dtype,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        validate=True,
    )
    compile_spec = _make_mla_decode_compile_spec(
        spec,
        device_index=device_index,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        q_dtype_key=_dtype_key(query.dtype),
        output_dtype_key=_dtype_key(out_dtype),
        max_seq_len_q=max_seq_len_q,
        packed_query=packed_query,
    )
    compiled = _get_compiled_mla_decode(compile_spec)
    workspace = _bind_mla_workspace(workspace_buffer, layout)
    return _launch_mla_decode(
        runtime,
        block_tables=block_tables,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        packed_query=packed_query,
        kv_lora_rank=kv_lora_rank,
        split_kv=spec.split_kv,
        workspace=workspace,
        compiled=compiled,
    )


class BatchMLADecodePagedTSWrapper:
    """Compile and reuse task-scheduled paged MLA decode launches."""

    @flashinfer_experimental_api
    def __init__(self) -> None:
        """Initialize an unplanned task-scheduled paged-MLA wrapper."""
        self._plan_state: Optional[_MLADecodePlanState] = None

    @flashinfer_experimental_api
    def plan(
        self,
        device: int | str | torch.device,
        batch_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_kv_len: int,
        *,
        max_seq_len_q: int,
        packed_query: bool,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        o_data_type: torch.dtype,
        mask_type: Literal["dense", "causal"] = "causal",
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        """Compile one static MLA shape and bind its reusable workspace.

        Planning consumes only compile-time geometry, capacity, dtype, and
        storage-mode inputs. Request metadata belongs exclusively to
        :meth:`run` and is never retained by the wrapper. A successful re-plan
        atomically replaces the previous immutable state; a failed re-plan
        leaves that state usable.

        ``packed_query=False`` selects fixed ``[B, SQ, H, 576]`` query storage,
        where ``SQ`` is exactly ``max_seq_len_q``. ``packed_query=True`` selects
        ``[total_q, H, 576]`` storage with per-run cumulative offsets supplied to
        every run. ``max_seq_len_q`` is then the per-request capacity.
        Individual packed requests may be empty; an all-empty run returns its
        empty output without dispatching a GPU kernel.

        If ``workspace_buffer`` is omitted, the plan allocates private scratch.
        A workspace is mutable and exclusive to one in-flight launch or graph
        replay. Warm the plan before graph capture, and call ``run`` with
        ``validate=False`` inside compiled or captured regions.

        Parameters
        ----------
        device : int, str, or torch.device
            CUDA device on which the plan will execute.
        batch_size : int
            Exact runtime request count.
        num_heads, kv_lora_rank, qk_rope_head_dim, page_size : int
            Static MLA head geometry and K/V page size.
        max_kv_len, max_seq_len_q : int
            Static per-request K/V and Q capacities.
        packed_query : bool
            Select packed rather than fixed query storage.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            Query, K/V, and output dtypes used to compile the plan.
        mask_type : {"dense", "causal"}
            Attention mask mode.
        workspace_buffer : torch.Tensor, optional
            Caller-owned contiguous int8 or uint8 scratch on ``device``. It
            must be 32-byte aligned and large enough for the selected plan.
            When omitted, planning allocates the buffer. The retained buffer
            is exclusive to one in-flight launch or graph replay.
        """

        if not isinstance(packed_query, bool):
            raise TypeError("packed_query must be a bool")
        _validate_mask(mask_type)
        batch_size = _validate_positive_int(batch_size, "batch_size")
        _validate_mla_int32_extent(batch_size, "batch_size")
        num_heads = _validate_positive_int(num_heads, "num_heads")
        _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
        page_size = _validate_page_size(page_size)
        max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")
        max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
        _validate_mla_query_head_extent(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
        )
        _validate_mla_dtype_pair(q_data_type, kv_data_type, o_data_type)
        device, device_index = _resolve_cuda_device(device)
        required_page_columns = _ceil_div(max_kv_len, page_size)

        spec_key = (
            device_index,
            batch_size,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_kv_len,
            _dtype_key(q_data_type),
            _dtype_key(kv_data_type),
            _dtype_key(o_data_type),
            mask_type,
            max_seq_len_q,
        )
        spec = _resolve_mla_decode_launch_spec(*spec_key)
        compile_spec = _make_mla_decode_compile_spec(
            spec,
            device_index=device_index,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype_key=_dtype_key(q_data_type),
            output_dtype_key=_dtype_key(o_data_type),
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
        )
        policy = spec.policy
        workspace_layout = _make_mla_workspace_layout(
            spec.kernel_workspace_bytes, batch_size, num_heads, max_seq_len_q
        )
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes, device=device, dtype=torch.int8
            )
        else:
            _validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
        workspace = _bind_mla_workspace(workspace_buffer, workspace_layout)
        compiled = _get_compiled_mla_decode(compile_spec)

        # Publish only after validation, compilation, allocation, and binding
        # succeed, so a failed re-plan leaves the previous plan usable.
        self._plan_state = _MLADecodePlanState(
            device=device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=page_size,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            output_dtype=o_data_type,
            mask_type=mask_type,
            max_kv_len=max_kv_len,
            required_page_columns=required_page_columns,
            workspace_buffer=workspace_buffer,
            workspace_layout=workspace_layout,
            workspace_views=workspace,
            compiled=compiled,
            policy=policy,
            split_kv=int(dict(policy)["split_kv"]),
        )

    @flashinfer_experimental_api(trace=prims_ts_decode_mla_wrapper_trace_dispatch)
    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        qo_indptr: Optional[torch.Tensor] = None,
        bmm1_scale: float = 1.0,
        bmm2_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor:
        """Launch the most recently planned MLA decode on the current stream.

        ``block_tables`` and ``seq_lens`` are required per-run bindings.
        ``qo_indptr`` is required by a packed-query plan and rejected by a
        fixed-query plan. With validation enabled, tensor structure, metadata
        values, scales, and every static capacity are checked before
        launch. These checks synchronize metadata to the host. Set
        ``validate=False`` only after validating representative inputs, and use
        it for ``torch.compile`` or CUDA graph capture.

        In either mode, output and workspace must be disjoint from each other
        and from all inputs. Storage overlap is not checked.

        Parameters
        ----------
        query : torch.Tensor
            Runtime fixed or packed query tensor matching the plan.
        kv_cache : torch.Tensor
            Runtime compact paged latent K/V cache.
        block_tables : torch.Tensor
            Runtime physical-page table with one inner-contiguous,
            non-overlapping row per request. Inter-row padding is accepted.
        seq_lens : torch.Tensor
            Runtime K/V lengths with one element per request.
        qo_indptr : torch.Tensor, optional
            Runtime cumulative query offsets for a packed-query plan.
        bmm1_scale, bmm2_scale : float
            QK and value/output scaling factors.
        out : torch.Tensor, optional
            Caller-owned output tensor. A new tensor is allocated when omitted.
        validate : bool
            Enable explicit runtime validation. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            The fixed or packed MLA attention output.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        runtime_qo_indptr = qo_indptr if state.packed_query else None
        runtime = _prepare_mla_runtime(
            query,
            kv_cache,
            device=state.device,
            batch_size=state.batch_size,
            num_heads=state.num_heads,
            max_seq_len_q=state.max_seq_len_q,
            packed_query=state.packed_query,
            qo_indptr=runtime_qo_indptr,
            page_size=state.page_size,
            q_dtype=state.q_dtype,
            kv_dtype=state.kv_dtype,
            output_dtype=state.output_dtype,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
            validate=validate,
        )
        if validate:
            _validate_mla_run_metadata(
                state,
                runtime,
                block_tables,
                seq_lens,
                qo_indptr,
            )
        return _launch_mla_decode(
            runtime,
            block_tables=block_tables,
            seq_lens=seq_lens,
            qo_indptr=runtime_qo_indptr,
            packed_query=state.packed_query,
            kv_lora_rank=state.kv_lora_rank,
            split_kv=state.split_kv,
            workspace=state.workspace_views,
            compiled=state.compiled,
        )


@flashinfer_experimental_api(trace=prims_ts_decode_mla_one_shot_trace_dispatch)
def batch_mla_decode_with_paged_kv_cache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    kv_lora_rank: int = _MLA_LATENT_DIM,
    qk_rope_head_dim: int = _MLA_ROPE_DIM,
    mask_type: Literal["dense", "causal"] = "causal",
    max_kv_len: Optional[int] = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One-shot convenience wrapper for fixed or packed-query MLA decode.

    This helper reads ``seq_lens`` and, for packed Q, ``qo_indptr`` on the host
    to derive plan bounds, then constructs a temporary wrapper. Invoke it
    outside CUDA Graph capture. Capture-sensitive callers should pre-plan
    :class:`BatchMLADecodePagedTSWrapper` and use ``run(validate=False)``.

    Parameters
    ----------
    query : torch.Tensor
        Fixed or packed query tensor with concatenated latent and RoPE heads.
    kv_cache : torch.Tensor
        Compact paged latent K/V cache.
    block_tables : torch.Tensor
        Dense physical-page table for each request. Rows must be inner
        contiguous and non-overlapping, but may have padding between them.
    seq_lens : torch.Tensor
        Live K/V sequence lengths.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_seq_len_q : int, optional
        Per-request packed-query length capacity. For packed Q it defaults to
        the maximum delta in ``qo_indptr``; an explicit value may be larger.
        For fixed Q it defaults to the query's sequence extent and, when
        provided, must equal that extent.
    kv_lora_rank, qk_rope_head_dim : int
        Latent and RoPE dimensions.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    max_kv_len : int, optional
        Static K/V length bound; defaults to the metadata maximum.
    bmm1_scale, bmm2_scale : float
        QK and value/output scaling factors.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        The fixed or packed MLA attention output.
    """

    packed_query = qo_indptr is not None
    _validate_query(query, packed_query=packed_query)
    metadata_device, batch_size, _ = _validate_mla_metadata(block_tables, seq_lens)
    if metadata_device != query.device:
        raise ValueError(
            f"MLA metadata must be on {query.device}, got {metadata_device}"
        )
    normalized_cache, _, page_size = _normalize_mla_kv_cache(
        kv_cache, expected_device=query.device
    )
    _validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    _validate_page_size(page_size)
    _validate_mla_dtype_pair(query.dtype, normalized_cache.dtype, out_dtype)
    if packed_query:
        _validate_qo_indptr(
            qo_indptr,
            device=query.device,
            batch_size=batch_size,
        )
        num_heads = int(query.shape[1])
        derived_max_seq_len_q, total_q, runtime_q_lengths = _derive_max_seq_len_q(
            qo_indptr,
            batch_size=batch_size,
        )
        if total_q != int(query.shape[0]):
            raise ValueError(
                "qo_indptr must end at the packed query row count "
                f"({query.shape[0]}), got {total_q}"
            )
        if max_seq_len_q is None:
            if derived_max_seq_len_q == 0:
                raise ValueError(
                    "max_seq_len_q is required for an all-empty packed query"
                )
            max_seq_len_q = derived_max_seq_len_q
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            if derived_max_seq_len_q > max_seq_len_q:
                raise ValueError(
                    "qo_indptr contains a per-request Q length larger than "
                    f"max_seq_len_q ({max_seq_len_q}): got "
                    f"{derived_max_seq_len_q}"
                )
        _validate_mla_query_head_extent(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            total_q=int(query.shape[0]),
        )
    else:
        num_heads = int(query.shape[2])
        fixed_seq_len_q = int(query.shape[1])
        if max_seq_len_q is None:
            max_seq_len_q = fixed_seq_len_q
        else:
            max_seq_len_q = _validate_positive_int(max_seq_len_q, "max_seq_len_q")
            if max_seq_len_q != fixed_seq_len_q:
                raise ValueError(
                    "fixed query length must equal max_seq_len_q: "
                    f"got SQ={fixed_seq_len_q} and "
                    f"max_seq_len_q={max_seq_len_q}"
                )
        _validate_mla_query_head_extent(
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
        )
        runtime_q_lengths = (max_seq_len_q,) * batch_size
    seq_lens_host = tuple(int(value) for value in seq_lens.tolist())
    if any(seq_len <= 0 for seq_len in seq_lens_host):
        raise ValueError("every runtime request must contain at least one KV token")
    metadata_max_kv_len = max(seq_lens_host)
    if max_kv_len is None:
        max_kv_len = metadata_max_kv_len
    else:
        max_kv_len = _validate_mla_max_kv_len(max_kv_len, "max_kv_len")
        if metadata_max_kv_len > max_kv_len:
            raise ValueError(
                "runtime KV metadata contains a request longer than "
                f"max_kv_len ({max_kv_len}): got {metadata_max_kv_len}"
            )
    if mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(runtime_q_lengths, seq_lens_host, strict=True)
        ):
            if q_len > kv_len:
                raise ValueError(
                    "causal MLA decode requires every per-request Q length "
                    "to be no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )
    assert max_seq_len_q is not None
    assert max_kv_len is not None
    if out is not None:
        _validate_out(
            out,
            device=query.device,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            total_q=int(query.shape[0]) if packed_query else None,
            output_dtype=out_dtype,
        )

    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        query.device,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_kv_len,
        max_seq_len_q=max_seq_len_q,
        packed_query=packed_query,
        q_data_type=query.dtype,
        kv_data_type=normalized_cache.dtype,
        o_data_type=out_dtype,
        mask_type=mask_type,
    )
    return wrapper.run(
        query,
        normalized_cache,
        block_tables,
        seq_lens,
        qo_indptr=qo_indptr,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
    )


__all__ = [
    "BatchMLADecodePagedTSWrapper",
    "batch_mla_decode_with_paged_kv_cache",
    "get_prims_ts_batch_mla_decode_workspace_size",
    "prims_ts_batch_mla_decode_with_kv_cache",
]


class SparseMLAPreparedMetadata(NamedTuple):
    """Caller-owned storage-row indices and prepared attention metadata.

    Indices are int32 [rows, capacity], with -1 invalid; lengths are int32
    [rows]. Primary and extra indices address their own flattened native pool.
    No block-table lookup or page-stride conversion occurs in run.

    Combined-route schedules also consume routes [passes, rows, route_capacity],
    execution_lengths/valid_counts [passes, rows], and FP32 scale_params
    [passes, 2 + heads + rows]. Route bit 31 selects the extra pool; bits 0:31
    encode the storage row, with 0x7fffffff invalid. Each source's execution
    span is rounded to 128. Empty rows execute one masked slot. Scale rows
    contain [QK scale, PV scale, sinks[heads], valid_counts[rows]]. Two passes
    are required when the sources use independent KV descales.

    Preparation must refresh all dependent fields when routes, lengths,
    scales or sinks change. Buffers must remain alive and unaliased with
    writable attention buffers through graph replay. Applications may supply
    these buffers from any preparer satisfying this contract.
    """

    indices: torch.Tensor
    lengths: torch.Tensor
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None
    routes: torch.Tensor | None = None
    execution_lengths: torch.Tensor | None = None
    valid_counts: torch.Tensor | None = None
    scale_params: torch.Tensor | None = None


def _fake_sparse_tensor(dtype, shape):
    """Compact, four-byte-aligned ABI shared by sparse finishing kernels."""
    import cutlass.cute as cute

    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=4,
    )


@functools.cache
def _compile_finish(device_index, heads, independent):
    import cutlass
    import cutlass.cute as cute
    from .kernels.mla_decode.sparse_reduce import FinishSparseMla

    rows = cute.sym_int()

    with torch.cuda.device(device_index):
        partial = _fake_sparse_tensor(cutlass.BFloat16, (rows, heads, 512))
        lse = _fake_sparse_tensor(cutlass.Float32, (rows, heads))
        lens = _fake_sparse_tensor(cutlass.Int32, (rows,))
        return cute.compile[cute.FrontendNext](
            FinishSparseMla(independent),
            partial,
            partial,
            lse,
            lse,
            lens,
            lens,
            _fake_sparse_tensor(cutlass.Float32, (heads,)),
            partial,
            lse,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=_COMPILE_OPTIONS,
        )


@functools.cache
def _compile_sparse_reduce(device_index, heads, storage_heads, splits, direct=False):
    import cutlass
    import cutlass.cute as cute
    from .kernels.mla_decode.sparse_reduce import FinishSparseMlaSplit

    rows = cute.sym_int()

    with torch.cuda.device(device_index):
        return cute.compile[cute.FrontendNext](
            FinishSparseMlaSplit(splits, direct),
            _fake_sparse_tensor(cutlass.BFloat16, (rows, storage_heads, splits, 512)),
            _fake_sparse_tensor(cutlass.Float32, (rows, storage_heads, splits)),
            _fake_sparse_tensor(cutlass.Int32, (rows,)),
            _fake_sparse_tensor(cutlass.Float32, (heads,)),
            _fake_sparse_tensor(cutlass.BFloat16, (rows, heads, 512)),
            _fake_sparse_tensor(cutlass.Float32, (rows, heads)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=_COMPILE_OPTIONS,
        )


def _resolve_sparse_mla_plan(
    device,
    batch_size,
    num_heads,
    *,
    max_topk,
    max_extra_topk=0,
    max_seq_len_q=1,
    packed_query=False,
    q_data_type=torch.bfloat16,
    kv_layout="NHD",
    has_sinks=False,
    return_lse=False,
    assume_valid_prefix=False,
):
    """Resolve sparse geometry and scratch sizing without allocation/compilation."""
    import cutlass
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
        compute_workspace_size,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
        compute_workspace_size as workspace_2cta,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
        MlaDecodeTs,
    )

    if not isinstance(assume_valid_prefix, bool):
        raise TypeError("assume_valid_prefix must be bool")
    device, _ = _resolve_cuda_device(device)
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("sparse TS MLA requires SM100 or SM103")
    for name, value in (
        ("batch_size", batch_size),
        ("num_heads", num_heads),
        ("max_seq_len_q", max_seq_len_q),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_heads > 128:
        raise ValueError("num_heads must be at most 128")
    for value in (max_topk, max_extra_topk):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("source capacities must be nonnegative integers")
    if max_topk == 0:
        raise ValueError("the primary selected-slot capacity must be positive")
    if q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("native BF16 or E4M3 is required")
    if kv_layout not in ("NHD", "HND"):
        raise ValueError("kv_layout must be NHD or HND")
    max_rows = batch_size * max_seq_len_q
    capacity = max(
        256,
        ((max_topk + 127) // 128 + (max_extra_topk + 127) // 128) * 128,
    )
    if max_rows * num_heads >= 2**31 or capacity >= 2**31 - 32768:
        raise ValueError("query/route extent exceeds the int32 kernel domain")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    profile = select_sparse_mla_profile(
        rows=max_rows,
        heads=num_heads,
        query_length=max_seq_len_q,
        capacity=capacity,
        dtype="bf16" if q_data_type == torch.bfloat16 else "fp8",
        sm_count=sm_count,
    )
    family, tile = profile.family, profile.tile_size_q
    splits, head_dim_ctas = profile.split_kv, profile.head_dim_ctas
    dtype_name = "bf16" if q_data_type == torch.bfloat16 else "e4m3"
    if family == "2cta":
        kernel = MlaDecodeTs(
            page_size=1,
            max_active_clusters=sm_count // 2,
            is_persistent=profile.scheduler != "nonpersistent",
            is_var_seq=False,
            is_var_split_kv=False,
            static_split_kv=splits,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            rope_dim=0,
            num_heads=num_heads,
            seq_len_q=1,
            batch_size=max_rows,
            mask_type="dense",
            device_scales=True,
        )
        kernel.sparse_gather_warps = profile.gather_issue_warps
        kernel.sparse_kv_stages = profile.kv_pipeline_stages
        kernel.sparse_uniform_pages = profile.uniform_offset_cache
        kernel.sparse_balanced_registers = profile.balanced_registers
        core_bytes = workspace_2cta(
            tile_size_q=128,
            num_q_tiles=1,
            latent_dim=512,
            batch_size=max_rows,
            split_kv=splits,
            partial_o_dtype=cutlass.BFloat16,
            lse_dtype=cutlass.Float32,
        )
    else:
        kernel_profile = MlaProfile(
            name="sparse",
            kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
            tile_size_q=tile,
            num_ctas_per_head_dim=head_dim_ctas,
            num_ctas_per_seq_kv=splits,
            use_multi_ctas_kv=int(splits > 1),
            use_cluster_reduction=int(profile.reduction == "cluster"),
            use_persistent_scheduler=int(profile.scheduler != "nonpersistent"),
            use_clc_dynamic_persistent_scheduler=int(profile.scheduler == "clc"),
        )
        kernel = ThroughputLatencyMlaDecodeTs(
            batch_size=max_rows,
            num_heads=tile,
            seq_len_q=(num_heads + tile - 1) // tile,
            seq_len_k=capacity,
            rope_dim=0,
            page_size=1,
            max_active_clusters=sm_count,
            qkv_dtype=dtype_name,
            out_dtype="bf16",
            profile=kernel_profile,
            reduction_mode=profile.reduction,
            logical_num_heads=num_heads,
            logical_seq_len_q=1,
            tile_size_q=tile,
            sparse_profile=profile,
            mask_type="dense",
            device_scales=True,
        )
        core_bytes = compute_workspace_size(
            cfg=kernel._make_config(),
            partial_o_dtype=cutlass.BFloat16,
            lse_dtype=cutlass.Float32,
        )
    spec = _make_mla_decode_compile_spec(
        _MLADecodeLaunchSpec(kernel, (), core_bytes, splits),
        device_index=device.index,
        num_heads=num_heads,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        page_size=1,
        q_dtype_key=str(q_data_type).removeprefix("torch."),
        output_dtype_key="bfloat16",
        max_seq_len_q=1,
        packed_query=False,
        device_scales=True,
    )
    sections = {}
    byte_end = 0
    for name, shape, dtype in (
        ("core", (core_bytes,), torch.int8),
        ("partial", (2, max_rows, 1, num_heads, 512), torch.bfloat16),
        # Both source views must satisfy the core's 16-byte base alignment,
        # including H6/H12 and odd maximum query counts.
        ("lse", (2, (max_rows + 3) // 4 * 4, 1, num_heads), torch.float32),
        ("public_lse", (max_rows, num_heads), torch.float32),
    ):
        sections[name], byte_end = _append_workspace_section(byte_end, shape, dtype)
    return device, profile, spec, sections, byte_end, capacity


class BatchSparseMLADecodePagedTSWrapper:
    """Plan native BF16/E4M3 attention over one or two pools, then bind metadata.

    Q is [B,Sq,H,512] or packed [total_q,H,512]. Caller-prepared metadata
    contains int32 [rows, capacity] storage-row indices and live lengths.
    Indices already include physical page strides; -1 is masked. Both source
    lists participate in one attention distribution. Packed KV is unsupported.

    Planning with ``assume_valid_prefix=True`` promises that every index before
    each live source length is valid (no -1 holes). Direct FP8 2CTA kernels
    then derive masks from lengths without reading indices or issuing ballots.
    Other kernels retain their generic mask path; no gain was established.
    ``run(validate=True)`` checks the promise; graph replay with validation
    disabled must preserve it. Tail entries beyond the lengths are ignored.

    A wrapper/workspace permits one in-flight run. Graphs require a completed
    warmup, stable addresses, preallocated outputs, and validate=False.
    Input/output/workspace storage must not overlap; this is an unchecked
    caller precondition. Separate wrappers/workspaces are needed per stream.
    """

    @flashinfer_experimental_api
    def __init__(self, workspace_buffer=None):
        self._workspace_buffer = workspace_buffer
        self._state = None
        self._scalar_cache = {}
        self.workspace_size_bytes = 0

    def plan(
        self,
        device,
        batch_size,
        num_heads,
        *,
        max_topk,
        max_extra_topk=0,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_layout="NHD",
        has_sinks=False,
        return_lse=False,
        assume_valid_prefix=False,
    ):
        """Plan D512 attention over one primary and an optional extra pool.

        ``max_topk`` describes arbitrary selected primary entries, not an SWA
        window. Prepared indices already include the physical page stride, so
        the core uses page size one regardless of the pools' external pages.
        Planning compiles no metadata-preparation kernel.
        """
        device, profile, spec, sections, byte_end, capacity = _resolve_sparse_mla_plan(
            device,
            batch_size,
            num_heads,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            q_data_type=q_data_type,
            kv_layout=kv_layout,
            has_sinks=has_sinks,
            return_lse=return_lse,
            assume_valid_prefix=assume_valid_prefix,
        )
        kernel = spec.kernel
        family, tile = profile.family, profile.tile_size_q
        splits = profile.split_kv
        max_rows = batch_size * max_seq_len_q
        # Policy resolves supported geometry before compilation. The fused
        # kernel handles one split or a cluster reduction; other splits use
        # the separate reducer. Independent source scales use the unfused path.
        use_cluster_epilogue = (
            family == "swap" and splits > 1 and profile.reduction == "cluster"
        )
        use_fused_epilogue = splits == 1 or use_cluster_epilogue
        use_fused_reduction = splits > 1 and profile.reduction == "gmem_separate"
        use_direct = profile.direct_inputs
        workspace = self._workspace_buffer
        if workspace is None:
            workspace = torch.empty(byte_end, device=device, dtype=torch.uint8)
        _validate_workspace_buffer(workspace, device=device, required_bytes=byte_end)
        buffers = {
            name: _workspace_section_view(workspace, section)
            for name, section in sections.items()
        }
        compiled = _get_compiled_mla_decode(spec)
        compiled_fused = None
        compiled_static = None
        compiled_reducer = None
        storage_heads = ((num_heads + tile - 1) // tile) * tile
        if use_fused_epilogue or use_fused_reduction:
            fused_kernel = copy.copy(kernel)
            if family == "2cta":
                fused_kernel.fuse_sparse_epilogue = use_fused_epilogue
                fused_kernel.external_sparse_reduction = use_fused_reduction
            else:
                fused_kernel.finalize_output = True
            fused_kernel.direct_sparse = use_direct
            if family == "2cta":
                fused_kernel.assume_valid_prefix = assume_valid_prefix
            fused_kernel.direct_sparse_capacities = (max_topk, max_extra_topk)
            fused_spec = replace(
                spec,
                kernel=fused_kernel,
                kernel_signature=_mla_kernel_compile_signature(fused_kernel),
            )
            compiled_fused = _get_compiled_mla_decode(fused_spec)
            if use_direct:
                static_kernel = copy.copy(fused_kernel)
                static_kernel.direct_static_scales = True
                compiled_static = _get_compiled_mla_decode(
                    replace(
                        fused_spec,
                        kernel=static_kernel,
                        kernel_signature=_mla_kernel_compile_signature(static_kernel),
                    )
                )
            if use_fused_reduction:
                compiled_reducer = _compile_sparse_reduce(
                    device.index, num_heads, storage_heads, splits, use_direct
                )
        finish = tuple(
            _compile_finish(device.index, num_heads, independent)
            for independent in (False, True)
        )
        self._state = dict(
            device=device,
            batch=batch_size,
            heads=num_heads,
            max_q=max_seq_len_q,
            max_rows=max_rows,
            packed=packed_query,
            dtype=q_data_type,
            ks=max_topk,
            kc=max_extra_topk,
            capacity=capacity,
            kv_layout=kv_layout,
            has_sinks=has_sinks,
            return_lse=return_lse,
            buffers=buffers,
            workspace=workspace,
            compiled=compiled,
            compiled_fused=compiled_fused,
            compiled_static=compiled_static,
            compiled_reducer=compiled_reducer,
            storage_heads=storage_heads,
            direct_inputs=use_direct,
            assume_valid_prefix=assume_valid_prefix,
            finish=finish,
            splits=splits,
            default_cl=torch.full(
                (max_rows,), max_extra_topk, device=device, dtype=torch.int32
            ),
            dummy_indices=torch.zeros((max_rows, 1), device=device, dtype=torch.int32),
            dummy_scales=torch.ones((1, 2), device=device, dtype=torch.float32),
            dummy_cache=torch.zeros((1, 1, 512), device=device, dtype=q_data_type),
            default_sinks=torch.full(
                (num_heads,), -torch.inf, device=device, dtype=torch.float32
            ),
        )

        self.workspace_size_bytes = byte_end
        self._scalar_cache.clear()

    def _scalar(self, value, name, validate):
        state = self._state
        if isinstance(value, torch.Tensor):
            if (
                value.dtype != torch.float32
                or value.device != state["device"]
                or value.numel() != 1
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} must be a scalar CUDA FP32 tensor")
            if validate and (
                not torch.isfinite(value).all().item() or (value <= 0).any().item()
            ):
                raise ValueError(f"{name} must be positive and finite")
            return value.reshape(1)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
        try:
            key = struct.unpack("f", struct.pack("f", float(value)))[0]
        except OverflowError as error:
            raise ValueError(f"{name} must be representable in float32") from error
        if not math.isfinite(key) or key <= 0:
            raise ValueError(f"{name} must be positive finite float32")
        if key not in self._scalar_cache:
            self._scalar_cache[key] = torch.full(
                (1,), key, device=state["device"], dtype=torch.float32
            )
        return self._scalar_cache[key]

    def _pool(self, cache, name):
        state = self._state
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if cache.ndim == 4:
            axis = 1 if state["kv_layout"] == "HND" else 2
            if cache.shape[axis] != 1:
                raise ValueError(f"{name} requires one KV head")
            cache = cache.squeeze(axis)
        if (
            cache.ndim != 3
            or cache.shape[1] <= 0
            or cache.shape[2] != 512
            or cache.shape[0] <= 0
        ):
            raise ValueError(
                f"{name} must have shape [pages,page_size,512] with nonempty pages"
            )
        page_size = cache.shape[1]
        if cache.device != state["device"] or cache.dtype != state["dtype"]:
            raise ValueError(f"{name} must match the planned device and dtype")
        if (
            cache.stride(-1) != 1
            or cache.stride(1) != 512
            or cache.stride(0) < page_size * 512
            or cache.stride(0) % 512
            or cache.data_ptr() % 16
        ):
            raise ValueError(
                f"{name} requires aligned compact rows and a page stride divisible by 512 elements"
            )
        row_stride = cache.stride(0) // 512
        rows = (cache.shape[0] - 1) * row_stride + page_size
        if rows >= 0x7FFFFFFF:
            raise ValueError(f"{name} exceeds signed gather coordinate range")
        return cache.as_strided((rows, 1, 512), (512, 512, 1))

    def _source_metadata(
        self, indices, lengths, capacity, tokens, name, *, rows, validate
    ):
        state = self._state
        if capacity == 0:
            if indices is not None or lengths is not None:
                raise ValueError(f"{name} metadata is not expected by the plan")
            return state["dummy_indices"][:rows], state["default_cl"][:rows]
        if not isinstance(indices, torch.Tensor) or (
            indices.shape != (rows, capacity)
            or indices.dtype != torch.int32
            or indices.device != state["device"]
            or indices.stride(-1) != 1
        ):
            raise ValueError(f"invalid {name} indices shape/dtype/device/stride")
        if not isinstance(lengths, torch.Tensor) or (
            lengths.shape != (rows,)
            or lengths.dtype != torch.int32
            or lengths.device != state["device"]
            or not lengths.is_contiguous()
        ):
            raise ValueError(f"invalid {name} lengths")
        if validate:
            if ((lengths < 0) | (lengths > capacity)).any().item():
                raise ValueError(f"{name} length exceeds capacity")
            active = (
                torch.arange(capacity, device=state["device"])[None, :]
                < lengths[:, None]
            )
            if (active & ((indices < -1) | (indices >= tokens))).any().item():
                raise ValueError(f"{name} active index outside pool")
            if state["assume_valid_prefix"] and (active & (indices < 0)).any().item():
                raise ValueError(
                    f"{name} active prefix contains a hole with assume_valid_prefix=True"
                )
        return indices, lengths

    def _bind_metadata(self, metadata, *, rows, passes, direct, primary_lengths):
        state = self._state
        buffers = state["buffers"]
        prepared_buffers = {}
        for key, value, dtype, shape in (
            (
                "routes",
                metadata.routes,
                torch.int32,
                (passes, rows, state["capacity"]),
            ),
            (
                "lengths",
                metadata.execution_lengths,
                torch.int32,
                (passes, rows),
            ),
            (
                "counts",
                metadata.valid_counts,
                torch.int32,
                (passes, rows),
            ),
            (
                "scales",
                metadata.scale_params,
                torch.float32,
                (passes, 2 + state["heads"] + rows),
            ),
        ):
            if value is None:
                if not direct:
                    raise ValueError(f"prepared {key} are required by this schedule")
                continue
            if (
                value.dtype != dtype
                or value.device != state["device"]
                or value.shape != shape
                or not value.is_contiguous()
            ):
                raise ValueError(f"prepared {key} must be contiguous {dtype}{shape}")
            prepared_buffers[key] = value
        buffers = (
            buffers
            | dict(
                routes=state["dummy_indices"][:rows].unsqueeze(0),
                lengths=primary_lengths.unsqueeze(0),
                counts=primary_lengths.unsqueeze(0),
                scales=state["dummy_scales"],
            )
            | prepared_buffers
        )
        return buffers

    @flashinfer_experimental_api
    def run(
        self,
        query,
        kv_cache,
        metadata: SparseMLAPreparedMetadata,
        extra_kv_cache=None,
        *,
        qo_indptr=None,
        softmax_scale=512**-0.5,
        q_scale=1.0,
        kv_scale=1.0,
        extra_kv_scale=None,
        output_scale=1.0,
        sinks=None,
        out=None,
        lse=None,
        validate=True,
    ):
        """Launch with caller-prepared metadata, without index conversion.

        Caller-owned metadata and plan workspace must outlive graph replay. Q and
        native KV are used directly; flattening padded pages only creates a
        tensor view. Required attention reductions/finishing remain included.
        Metadata is an unchecked consistency contract: its packed routes,
        counts and scales must agree with the supplied indices and scalars.
        """
        state = self._state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        if not isinstance(metadata, SparseMLAPreparedMetadata):
            raise TypeError("metadata must be SparseMLAPreparedMetadata")
        _validate_query(
            query,
            packed_query=state["packed"],
            device=state["device"],
            batch_size=state["batch"],
            num_heads=state["heads"],
            max_seq_len_q=state["max_q"],
            q_dtype=state["dtype"],
            query_dim=512,
        )
        prefix = tuple(query.shape[:-2])
        rows = math.prod(prefix)
        if state["packed"]:
            _validate_qo_indptr(
                qo_indptr, device=state["device"], batch_size=state["batch"]
            )
            if validate:
                maximum, total, _ = _derive_max_seq_len_q(
                    qo_indptr, batch_size=state["batch"]
                )
                if total != rows or maximum > state["max_q"]:
                    raise ValueError("invalid packed query offsets")
        elif qo_indptr is not None:
            raise ValueError("fixed queries do not accept qo_indptr")

        primary = self._pool(kv_cache, "kv_cache")
        if extra_kv_cache is None:
            if (
                state["kc"]
                or metadata.extra_indices is not None
                or metadata.extra_lengths is not None
            ):
                raise ValueError("extra pool and indices are required by the plan")
            extra = state["dummy_cache"]
        else:
            extra = self._pool(extra_kv_cache, "extra_kv_cache")

        si, sl = self._source_metadata(
            metadata.indices,
            metadata.lengths,
            state["ks"],
            primary.shape[0],
            "primary",
            rows=rows,
            validate=validate,
        )
        ci, cl = self._source_metadata(
            metadata.extra_indices,
            metadata.extra_lengths,
            state["kc"],
            extra.shape[0],
            "extra",
            rows=rows,
            validate=validate,
        )
        if state["has_sinks"] != (sinks is not None):
            raise ValueError("sinks presence must match the plan")
        if sinks is None:
            sinks = state["default_sinks"]
        if (
            sinks.shape != (state["heads"],)
            or sinks.dtype != torch.float32
            or sinks.device != state["device"]
            or not sinks.is_contiguous()
        ):
            raise ValueError("sinks must be contiguous CUDA FP32[H]")
        if validate and torch.isnan(sinks).any().item():
            raise ValueError("sinks must not contain NaN")
        if extra_kv_scale is None:
            extra_kv_scale = kv_scale
        shared_scale = extra_kv_scale is kv_scale or (
            not isinstance(kv_scale, torch.Tensor)
            and not isinstance(extra_kv_scale, torch.Tensor)
            and kv_scale == extra_kv_scale
        )
        independent = state["kc"] > 0 and not shared_scale
        scale_tensors = [
            self._scalar(v, n, validate)
            for v, n in (
                (softmax_scale, "softmax_scale"),
                (q_scale, "q_scale"),
                (kv_scale, "kv_scale"),
                (extra_kv_scale, "extra_kv_scale"),
                (output_scale, "output_scale"),
            )
        ]
        if validate and state["dtype"] == torch.bfloat16:
            if any(t.item() != 1 for t in scale_tensors[1:4]):
                raise ValueError("BF16 Q/KV descales must be one")
        if out is None:
            out = torch.empty(
                (*prefix, state["heads"], 512),
                device=state["device"],
                dtype=torch.bfloat16,
            )
        _validate_out(
            out,
            device=state["device"],
            batch_size=state["batch"],
            num_heads=state["heads"],
            max_seq_len_q=state["max_q"],
            packed_query=state["packed"],
            total_q=rows if state["packed"] else None,
            output_dtype=torch.bfloat16,
        )
        if lse is None:
            # A returned result must survive the next call on this wrapper.
            # Use plan-owned scratch only when the LSE is not exposed.
            lse = (
                torch.empty(
                    (*prefix, state["heads"]),
                    device=state["device"],
                    dtype=torch.float32,
                )
                if state["return_lse"]
                else state["buffers"]["public_lse"][:rows].view(*prefix, state["heads"])
            )
        if (
            lse.shape != (*prefix, state["heads"])
            or lse.dtype != torch.float32
            or lse.device != state["device"]
            or not lse.is_contiguous()
        ):
            raise ValueError(
                "lse must be contiguous FP32 with one value per query/head"
            )
        fused = (
            state["compiled_fused"] is not None
            and not independent
            and lse.data_ptr() % 16 == 0
        )
        fused_main = fused and state["compiled_reducer"] is None
        direct = fused and state["direct_inputs"]
        static_scales = direct and all(
            not isinstance(v, torch.Tensor)
            for v in (softmax_scale, q_scale, kv_scale, output_scale)
        )
        bmm1_scale, bmm2_scale = 1.0, 1.0
        if static_scales:

            def f32(value):
                return struct.unpack("f", struct.pack("f", float(value)))[0]

            bmm1_scale = f32(f32(f32(softmax_scale) * f32(q_scale)) * f32(kv_scale))
            bmm2_scale = f32(f32(output_scale) * f32(kv_scale))
        sparse_inputs = (
            (
                si,
                ci,
                sl,
                cl,
                scale_tensors[0],
                scale_tensors[1],
                scale_tensors[2],
                scale_tensors[4],
                sinks,
            )
            if direct
            else None
        )
        if rows:
            buffers = self._bind_metadata(
                metadata,
                rows=rows,
                passes=2 if independent else 1,
                direct=direct,
                primary_lengths=sl,
            )
            query_view = query.view(rows, 1, state["heads"], 512)
            for slot in range(2 if independent else 1):
                _launch_mla_decode(
                    _MLARuntime(
                        query_view,
                        primary,
                        (
                            out.view(rows, 1, state["heads"], 512)
                            if fused_main
                            else buffers["partial"][slot, :rows]
                        ),
                        primary.shape[0],
                        bmm1_scale,
                        bmm2_scale,
                        extra_cache=extra,
                        scale_params=buffers["scales"][slot],
                        sparse_inputs=sparse_inputs,
                    ),
                    block_tables=buffers["routes"][slot, :rows],
                    seq_lens=buffers["lengths"][slot, :rows],
                    qo_indptr=None,
                    packed_query=False,
                    kv_lora_rank=512,
                    split_kv=state["splits"],
                    workspace=_MLAWorkspaceViews(
                        buffers["core"] if buffers["core"].numel() else None,
                        (
                            lse.view(rows, 1, state["heads"])
                            if fused_main
                            else buffers["lse"][slot, :rows]
                        ),
                    ),
                    compiled=(
                        state["compiled_static"]
                        if static_scales
                        else state["compiled_fused"]
                        if fused
                        else state["compiled"]
                    ),
                )
            if fused and not fused_main:
                split_elements = rows * state["storage_heads"] * state["splits"]
                o_bytes = split_elements * 512 * 2
                partial = (
                    buffers["core"][:o_bytes]
                    .view(torch.bfloat16)
                    .view(rows, state["storage_heads"], state["splits"], 512)
                )
                partial_lse = (
                    buffers["core"][o_bytes : o_bytes + split_elements * 4]
                    .view(torch.float32)
                    .view(rows, state["storage_heads"], state["splits"])
                )
                state["compiled_reducer"](
                    partial,
                    partial_lse,
                    buffers["counts"][0, :rows],
                    sinks,
                    out.view(rows, state["heads"], 512),
                    lse.view(rows, state["heads"]),
                )
            if not fused:
                state["finish"][int(independent)](
                    buffers["partial"][0, :rows].view(rows, state["heads"], 512),
                    buffers["partial"][1, :rows].view(rows, state["heads"], 512),
                    buffers["lse"][0, :rows].view(rows, state["heads"]),
                    buffers["lse"][1, :rows].view(rows, state["heads"]),
                    buffers["counts"][0, :rows],
                    buffers["counts"][1 if independent else 0, :rows],
                    sinks,
                    out.view(rows, state["heads"], 512),
                    lse.view(rows, state["heads"]),
                )
        return (out, lse) if state["return_lse"] else out


def get_prims_ts_sparse_mla_decode_workspace_size(*plan_args, **plan_kwargs):
    """Return workspace bytes using the same arguments as wrapper.plan().

    Resolves the default kernel geometry without allocating GPU scratch or
    compiling a kernel. Bind the resulting byte buffer to the wrapper constructor
    and call plan() before graph capture; run() is the prepared standalone launch.
    """
    _, _, _, _, size_bytes, _ = _resolve_sparse_mla_plan(*plan_args, **plan_kwargs)
    return size_bytes


@flashinfer_experimental_api
def batch_sparse_mla_decode_with_paged_kv_cache(
    query,
    kv_cache,
    metadata: SparseMLAPreparedMetadata,
    extra_kv_cache=None,
    *,
    qo_indptr=None,
    max_seq_len_q=None,
    kv_layout="NHD",
    softmax_scale=512**-0.5,
    q_scale=1.0,
    kv_scale=1.0,
    extra_kv_scale=None,
    output_scale=1.0,
    sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    workspace_buffer=None,
    assume_valid_prefix=False,
):
    """Eager plan-and-run helper using caller-prepared metadata.

    Use a planned wrapper for CUDA Graph replay. Preparation remains external.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "plan BatchSparseMLADecodePagedTSWrapper before CUDA Graph capture"
        )
    if not isinstance(metadata, SparseMLAPreparedMetadata):
        raise TypeError("metadata must be SparseMLAPreparedMetadata")
    packed = query.ndim == 3
    if packed:
        if qo_indptr is None:
            raise ValueError("packed queries require qo_indptr")
        batch = qo_indptr.numel() - 1
        if max_seq_len_q is None:
            max_seq_len_q = max(1, int((qo_indptr[1:] - qo_indptr[:-1]).max().item()))
    else:
        batch = query.shape[0]
        max_seq_len_q = query.shape[1] if max_seq_len_q is None else max_seq_len_q
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace_buffer)
    wrapper.plan(
        query.device,
        batch,
        query.shape[-2],
        max_topk=metadata.indices.shape[-1],
        max_extra_topk=0
        if metadata.extra_indices is None
        else metadata.extra_indices.shape[-1],
        max_seq_len_q=max_seq_len_q,
        packed_query=packed,
        q_data_type=query.dtype,
        kv_layout=kv_layout,
        has_sinks=sinks is not None,
        return_lse=return_lse,
        assume_valid_prefix=assume_valid_prefix,
    )
    return wrapper.run(
        query,
        kv_cache,
        metadata,
        extra_kv_cache,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        extra_kv_scale=extra_kv_scale,
        output_scale=output_scale,
        sinks=sinks,
        out=out,
        lse=lse,
    )
