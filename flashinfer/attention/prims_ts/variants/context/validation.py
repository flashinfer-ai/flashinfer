# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Argument and runtime-input checks for the PrimTS context frontends."""

from __future__ import annotations

import itertools
import math
import torch
from ...common.paged_kv import validate_native_hnd_tma_strides
from ...common.scales import GlobalScale
from ...common.tensors import compact_strides, cuda_device_index
from ..sage.config import SageAttentionConfig, validate_sage_recipe
from ..vc_attention import VCAttentionConfig, vc_repair_kv_len
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ContextPlanState
    from .geometry import ContextGeometry, ContextPlanGeometry, PagedContextPlanGeometry
    from .kernel.config import FmhaConfig


SUPPORTED_HEAD_DIMS = (128, 192, 256)
DEFAULT_PAGED_KV_PAGE_SIZE = 32
MAX_CONTEXT_SOFTMAX_SCALE = torch.finfo(torch.float32).max / math.log2(math.e)
SUPPORTED_PAGED_KV_PAGE_SIZES = (4, 8, 16, 32, 64, 128)
SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
INT32_MAX = 2**31 - 1
CUDA_GRID_YZ_MAX = 65_535
CONTEXT_KV_TILE_N = 128
CONTEXT_TILE_SIZE_Q = 128
# Query-paired D128 represents two 128-row Q tiles in one work tile.  Kernel
# coordinates include the padded tail of that 256-row span and, for masking,
# its exclusive right boundary.  Reserve the maximum 255-row tail padding in
# every packed logical data extent so all such coordinates remain signed
# Int32 even when the final request ends at the plan's packed extent.
CONTEXT_MAX_Q_ROWS_PER_WORK_TILE = 256
CONTEXT_PADDED_EXTENT_MAX = INT32_MAX - (CONTEXT_MAX_Q_ROWS_PER_WORK_TILE - 1)


def context_dtype_key(dtype: torch.dtype) -> str:
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        # MXFP8 Q/K: float8_e4m3fn data with one E8M0 scale per 32 channels.
        torch.float8_e8m0fnu: "float8_e8m0fnu",
        torch.int8: "int8",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts context supports torch.float16, torch.bfloat16, "
            "torch.float8_e4m3fn, torch.float8_e8m0fnu (MXFP8), and torch.int8 "
            f"Q/K; got {dtype}"
        ) from error


def fixed_kv_rows(max_seq_len_k: int, vc: Optional[VCAttentionConfig]) -> int:
    """K/V rows of a fixed plan. V repair adds its tiles to the sequence."""
    if vc is not None and vc.repair_tiles:
        return vc_repair_kv_len(max_seq_len_k, vc.repair_tiles)
    return max_seq_len_k


def mxfp8_scale_atoms(
    scales: torch.Tensor, tensor: torch.Tensor, name: str, validate: bool
) -> torch.Tensor:
    """View the MXFP8 scales of ``tensor``, the ``[B, H, tiles, 128, 4]`` atoms
    ``mxfp8_quantize_bshd`` writes, as the kernel's ``[B, H, tiles, 512]`` bytes.
    The atom shape is required, so the linear scales of ``mxfp8_quantize``
    cannot pass as atoms by byte count alone."""
    batch_size, seq_len, num_heads, _ = tensor.shape
    tiles = (seq_len + 127) // 128
    if validate:
        validate_tensor(scales, name)
        if scales.device != tensor.device:
            raise ValueError(f"{name} must be on {tensor.device}, got {scales.device}")
        validate_alignment(scales, name, 16)
        if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(f"{name} must hold uint8 or float8_e8m0fnu bytes")
        atoms = (batch_size, num_heads, tiles, 128, 4)
        if tuple(scales.shape) != atoms or not scales.is_contiguous():
            raise ValueError(
                f"{name} must be the contiguous {atoms} scale atoms of "
                f"mxfp8_quantize_bshd; got {tuple(scales.shape)}"
            )
    return scales.view(torch.uint8).reshape(batch_size, num_heads, tiles, 512)


def read_indptr(
    indptr: torch.Tensor,
    name: str,
    *,
    expected_total: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read cumulative offsets and validate nonnegative row lengths."""

    values = tuple(int(value) for value in indptr.tolist())
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if values[-1] != expected_total:
        raise ValueError(
            f"the final {name} offset must equal the packed tensor extent; "
            f"expected {expected_total}, got {values[-1]}"
        )
    lengths = tuple(curr - prev for prev, curr in itertools.pairwise(values))
    if any(length < 0 for length in lengths):
        raise ValueError(f"{name} offsets must be nondecreasing")
    return values, lengths


def read_int32_values(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    """Read one metadata vector after validating its extent."""

    if tensor.numel() != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} elements, got {tensor.numel()}"
        )
    return tuple(int(value) for value in tensor.tolist())


def refresh_variable_window_cta_starts(
    starts: torch.Tensor, *, state: ContextPlanState
) -> torch.Tensor:
    """Refresh plan-owned CTA minima from the current variable-window bounds."""

    geometry = state.geometry
    tile_size_q = variable_window_q_tile_size(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    padded_rows = num_seq_tiles * tile_size_q
    starts_2d = starts.view(geometry.batch_size, geometry.max_q_len)
    padded = state.variable_window_padded_starts
    if padded is not None:
        padded.fill_(INT32_MAX)
        padded[:, : geometry.max_q_len].copy_(starts_2d)
        starts_2d = padded
    elif padded_rows != geometry.max_q_len:
        raise RuntimeError("variable-window padded-start scratch is missing")
    torch.amin(
        starts_2d.view(geometry.batch_size, num_seq_tiles, tile_size_q),
        dim=-1,
        out=state.variable_window_cta_starts.view(geometry.batch_size, num_seq_tiles),
    )
    return state.variable_window_cta_starts


def resolve_context_device(
    device: int | str | torch.device,
) -> tuple[torch.device, int]:
    """Resolve one explicit CUDA plan device and validate its architecture."""

    if isinstance(device, bool):
        raise TypeError("device must identify a CUDA device")
    try:
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TypeError("device must identify a CUDA device") from error
    if resolved.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {resolved}")
    device_index = validate_device(resolved)
    return torch.device("cuda", device_index), device_index


def v_dtype_follows_qk(q_dtype: torch.dtype, v_dtype: torch.dtype) -> bool:
    """Whether V may take ``v_dtype`` next to ``q_dtype`` Q/K.

    V matches Q/K except on the QK-BF16/PV-FP8 path and under a Sage recipe's
    Int8 Q/K or MXFP8 Q/K, which keep E4M3 V.
    """
    return (v_dtype == q_dtype and q_dtype != torch.float8_e8m0fnu) or (
        q_dtype in (torch.bfloat16, torch.int8, torch.float8_e8m0fnu)
        and v_dtype == torch.float8_e4m3fn
    )


def validate_alignment(tensor: torch.Tensor, name: str, alignment: int) -> None:
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")


def validate_base_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allow_mla: bool = False,
) -> None:
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        validate_tensor(tensor, name)
        validate_alignment(tensor, name, 16)
    if k.device != q.device or v.device != q.device:
        raise ValueError(
            "Q, K, and V must be on one CUDA device; "
            f"got {q.device}, {k.device}, and {v.device}"
        )
    validate_qkv_dtype(q, k, v)
    mla_shape = (
        allow_mla
        and k.ndim >= 1
        and tuple(v.shape) == (*k.shape[:-1], 128)
        and k.shape[-1] == 192
    )
    if tuple(v.shape) != tuple(k.shape) and not mla_shape:
        raise ValueError(
            f"v must have the same shape as k; got {tuple(v.shape)} and {tuple(k.shape)}"
        )


def validate_block_tables_tensor(
    block_tables: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    required_page_columns: int,
) -> None:
    """Validate a fixed page table while preserving its runtime row stride."""

    validate_tensor(block_tables, "block_tables")
    if block_tables.device != device:
        raise ValueError(f"block_tables must be on {device}, got {block_tables.device}")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must have dtype torch.int32")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be rank 2, got rank {block_tables.ndim}")
    if int(block_tables.shape[0]) != batch_size:
        raise ValueError(
            "block_tables must have one row per request; expected "
            f"{batch_size}, got {block_tables.shape[0]}"
        )
    page_columns = int(block_tables.shape[1])
    if page_columns < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {page_columns}"
        )
    row_stride = int(block_tables.stride(0))
    if block_tables.stride(1) != 1 or row_stride < page_columns:
        raise ValueError(
            "block_tables must have row-strided [B, C] storage with "
            f"stride(1) == 1 and stride(0) >= C; got shape "
            f"{tuple(block_tables.shape)} and strides {block_tables.stride()}"
        )
    validate_extent(row_stride, "block_tables row stride")
    row_span = (batch_size - 1) * row_stride + page_columns
    validate_extent(row_span, "block_tables addressable row span")
    validate_alignment(block_tables, "block_tables", 4)


def validate_compact(tensor: torch.Tensor, name: str, layout: str) -> None:
    expected = compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected:
        raise ValueError(
            f"{name} must have compact {layout} strides {expected}, "
            f"but has {tensor.stride()}"
        )


def validate_context_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a contiguous-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (k_dtype, "k_dtype"),
        (v_dtype, "v_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        context_dtype_key(dtype)
    validate_output_dtype(out_dtype)
    if k_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts context requires Q and K to use the same dtype; "
            f"got q_dtype={q_dtype} and k_dtype={k_dtype}"
        )
    if not v_dtype_follows_qk(q_dtype, v_dtype):
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype "
            "except for QK-BF16/PV-FP8 and Int8 Q/K with E4M3 V; "
            f"got q_dtype={q_dtype} and v_dtype={v_dtype}"
        )


def validate_context_head_dim(
    q_head_dim: int, kv_head_dim: int, v_head_dim: int | None = None
) -> int:
    if q_head_dim != kv_head_dim:
        raise ValueError(
            "Q and K head dimensions must match; "
            f"got Q {q_head_dim} and K {kv_head_dim}"
        )
    if (q_head_dim, v_head_dim) == (192, 128):
        return q_head_dim
    if v_head_dim is not None and v_head_dim != q_head_dim:
        raise NotImplementedError(
            "separate QK/V dimensions require contiguous QK/V=(192, 128); "
            f"got ({q_head_dim}, {v_head_dim})"
        )
    if q_head_dim not in SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "attention-ts context supports head_dim in "
            f"{SUPPORTED_HEAD_DIMS}, or contiguous QK/V=(192, 128); got {q_head_dim}"
        )
    return q_head_dim


def validate_context_head_geometry(num_qo_heads: int, num_kv_heads: int) -> int:
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "the Q head count must be divisible by the KV head count; "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    return num_qo_heads // num_kv_heads


def validate_context_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal", "variable_window"):
        raise ValueError(
            "mask_type must be exactly 'dense', 'causal', or "
            f"'variable_window', got {mask_type!r}"
        )


def validate_context_page_size(page_size: int) -> int:
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size not in SUPPORTED_PAGED_KV_PAGE_SIZES:
        raise NotImplementedError(
            "attention-ts paged context supports page_size in "
            f"{SUPPORTED_PAGED_KV_PAGE_SIZES}; got {page_size}"
        )
    return page_size


def validate_context_sage_config(
    geometry: ContextPlanGeometry, sage: SageAttentionConfig
) -> None:
    """Reject a Sage recipe the contiguous context kernel does not run.

    Beyond the recipe every wrapper checks (``validate_sage_recipe``), Sage
    attention covers the dense, fixed-length, query-paired D128 path.
    """

    validate_sage_recipe(
        sage,
        q_dtype=geometry.qk_dtype,
        k_dtype=geometry.qk_dtype,
        v_dtype=geometry.pv_dtype,
        output_dtype=geometry.out_dtype,
    )
    if geometry.mask_type != "dense" or geometry.packed_query or geometry.head_paired:
        raise NotImplementedError(
            "Sage attention requires a dense, fixed-length, query-paired context plan"
        )
    if geometry.head_dim != 128 or geometry.head_dim_vo not in (None, 128):
        raise NotImplementedError("Sage attention requires head_dim 128")


def validate_context_softmax_scale(scale: GlobalScale) -> None:
    """Check normalized host scales against the launcher's base-2 conversion limit."""
    if not isinstance(scale, torch.Tensor) and scale > MAX_CONTEXT_SOFTMAX_SCALE:
        raise ValueError(
            "softmax_scale is too large for the launcher's FP32 base-2 conversion"
        )


def validate_context_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left == 0:
        raise ValueError("window_left=0 is unsupported; use -1 to disable the window")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or positive")
    if window_left > INT32_MAX - 1:
        raise ValueError(f"window_left must be no larger than {INT32_MAX - 1}")
    if window_left > 0 and mask_type != "causal":
        raise ValueError("a positive window_left requires mask_type='causal'")
    return window_left


def validate_device(device: torch.device) -> int:
    device_index = cuda_device_index(device)
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts context requires an SM100a/B200, SM103a/B300 or "
            "SM107a/Rubin GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    # Rubin runs through the sm_100f family target; a CuTe DSL older than 4.8
    # cannot emit for it unless CUTE_DSL_ARCH=sm_100f is set before import.
    if capability == (10, 7):
        from .....cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device_index)
    return device_index


def validate_extent(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def validate_indptr_tensor(
    indptr: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    validate_tensor(indptr, name)
    if indptr.device != device:
        raise ValueError(f"{name} must be on {device}, got {indptr.device}")
    if indptr.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if indptr.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {indptr.ndim}")
    if indptr.numel() < 2:
        raise ValueError(f"{name} must contain at least start and end offsets")
    validate_compact(indptr, name, "[B+1]")
    validate_alignment(indptr, name, 4)


def validate_output_dtype(out_dtype: torch.dtype) -> None:
    if not isinstance(out_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    context_dtype_key(out_dtype)
    if out_dtype == torch.int8:
        raise NotImplementedError("attention-ts context does not write an int8 output")


def validate_padded_data_extent(value: int, name: str) -> int:
    """Validate an Int32 data extent while reserving work-tile tail padding."""
    validate_extent(value, name)
    if value > CONTEXT_PADDED_EXTENT_MAX:
        raise NotImplementedError(
            f"{name} must be <= {CONTEXT_PADDED_EXTENT_MAX} so padded "
            "context work-tile coordinates fit in a signed int32"
        )
    return value


def validate_paged_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    """Validate the explicit dtypes of a paged-context specialization."""

    for dtype, name in (
        (q_dtype, "q_dtype"),
        (k_dtype, "k_dtype"),
        (v_dtype, "v_dtype"),
        (out_dtype, "out_dtype"),
    ):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype")
        context_dtype_key(dtype)
        if dtype == torch.int8:
            raise NotImplementedError(
                "attention-ts paged context does not support int8 tensors"
            )
    if k_dtype != q_dtype:
        raise NotImplementedError(
            "attention-ts paged context requires Q and K to use the same "
            f"dtype; got q_dtype={q_dtype} and k_dtype={k_dtype}"
        )
    if not v_dtype_follows_qk(q_dtype, v_dtype):
        raise NotImplementedError(
            "attention-ts paged context requires Q, K, and V to use the same "
            "dtype except for QK-BF16/PV-FP8; "
            f"got q_dtype={q_dtype} and v_dtype={v_dtype}"
        )


def validate_paged_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    validate_tensor(tensor, name)
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {tensor.ndim}")
    validate_compact(tensor, name, "one-dimensional")
    validate_alignment(tensor, name, 4)


def validate_paged_runtime_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    geometry: PagedContextPlanGeometry,
) -> None:
    """Validate per-run data tensors without fixing packed/page extents."""

    validate_base_tensors(q, k_cache, v_cache)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.qk_dtype:
        raise ValueError(f"q must have dtype {geometry.qk_dtype}, got {q.dtype}")
    if k_cache.dtype != geometry.qk_dtype:
        raise ValueError(
            f"k_cache must have dtype {geometry.qk_dtype}, got {k_cache.dtype}"
        )
    if v_cache.dtype != geometry.pv_dtype:
        raise ValueError(
            f"v_cache must have dtype {geometry.pv_dtype}, got {v_cache.dtype}"
        )
    if q.ndim != 3 or tuple(q.shape[1:]) != (
        geometry.num_qo_heads,
        geometry.head_dim,
    ):
        raise ValueError(
            "q must have shape [total_q, Hq, D] with "
            f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got {tuple(q.shape)}"
        )
    expected_kv_tail = (
        geometry.num_kv_heads,
        geometry.page_size,
        geometry.head_dim,
    )
    if k_cache.ndim != 4 or tuple(k_cache.shape[1:]) != expected_kv_tail:
        raise ValueError(
            "k_cache must have shape [num_pages, Hkv, page_size, D] with "
            f"Hkv/page/D={expected_kv_tail}, got {tuple(k_cache.shape)}"
        )
    total_q = int(q.shape[0])
    validate_padded_data_extent(total_q, "total_q")
    if total_q > (geometry.batch_size * geometry.max_q_len):
        raise ValueError(
            "q must contain no more than batch_size * max_q_len "
            f"rows; got {total_q} rows for batch_size={geometry.batch_size} and "
            f"max_q_len={geometry.max_q_len}"
        )
    validate_extent(int(k_cache.shape[0]), "num_physical_pages")
    validate_compact(q, "q", "[total_q, Hq, D]")
    validate_native_hnd_tma_strides(k_cache, "k_cache")
    validate_native_hnd_tma_strides(v_cache, "v_cache")


def validate_paged_runtime_metadata(
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    geometry: PagedContextPlanGeometry,
    *,
    total_q: int,
    num_physical_pages: int,
) -> tuple[int, int]:
    """Validate per-run fixed-table storage and active metadata values."""

    validate_indptr_tensor(qo_indptr, "qo_indptr", device=geometry.device)
    if qo_indptr.numel() != geometry.batch_size + 1:
        raise ValueError(
            "qo_indptr must contain batch_size + 1 elements "
            f"({geometry.batch_size + 1}), got {qo_indptr.numel()}"
        )

    required_page_columns = (
        geometry.max_kv_len + geometry.page_size - 1
    ) // geometry.page_size
    validate_block_tables_tensor(
        block_tables,
        device=geometry.device,
        batch_size=geometry.batch_size,
        required_page_columns=required_page_columns,
    )

    validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=geometry.device)
    if seq_lens_kv.numel() != geometry.batch_size:
        raise ValueError(
            "seq_lens_kv must contain one value per request; expected "
            f"{geometry.batch_size}, got {seq_lens_kv.numel()}"
        )

    _, q_lengths = read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    seq_lens_kv = read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=geometry.batch_size,
    )
    block_table_rows = block_tables.tolist()
    for batch_idx, (q_length, kv_length, block_table_row) in enumerate(
        zip(q_lengths, seq_lens_kv, block_table_rows, strict=True)
    ):
        if q_length > geometry.max_q_len:
            raise ValueError(
                "qo_indptr deltas must not exceed max_q_len; "
                f"batch {batch_idx} has {q_length}, maximum is "
                f"{geometry.max_q_len}"
            )
        if geometry.uniform_packed_lengths and q_length != geometry.max_q_len:
            raise ValueError(
                "uniform_packed_lengths=True requires every qo_indptr delta "
                f"to equal max_q_len={geometry.max_q_len}; batch "
                f"{batch_idx} has {q_length}"
            )
        if kv_length < 0 or kv_length > geometry.max_kv_len:
            raise ValueError(
                "seq_lens_kv entries must be nonnegative and not exceed "
                f"max_kv_len={geometry.max_kv_len}; batch {batch_idx} has "
                f"{kv_length}"
            )
        if geometry.uniform_packed_lengths and kv_length != geometry.max_kv_len:
            raise ValueError(
                "uniform_packed_lengths=True requires every seq_lens_kv entry "
                f"to equal max_kv_len={geometry.max_kv_len}; batch "
                f"{batch_idx} has {kv_length}"
            )
        required_pages = (kv_length + geometry.page_size - 1) // geometry.page_size
        if geometry.mask_type == "causal" and kv_length > 0 and q_length > kv_length:
            raise ValueError(
                "bottom-right causal context requires Sq <= Sk for each "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        if (
            geometry.mask_type == "causal"
            and not geometry.has_q_offset
            and q_length != kv_length
        ):
            raise ValueError(
                "has_q_offset=False requires Sq == Sk for every causal "
                f"request; got batch {batch_idx}: Sq={q_length}, Sk={kv_length}"
            )
        for row_offset, page_idx in enumerate(block_table_row[:required_pages]):
            if page_idx < 0 or page_idx >= num_physical_pages:
                raise ValueError(
                    "active block_tables entries must index the physical "
                    f"page pool; batch {batch_idx}, row entry {row_offset} is "
                    f"{page_idx}, pool has {num_physical_pages} pages"
                )
    return max(q_lengths), max(seq_lens_kv)


def validate_qkv_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate run-time Q/K/V tensor dtypes.

    Q and K must match; V may differ only for the QK-BF16/PV-FP8 path and
    for Int8 Q/K with E4M3 V.
    """
    context_dtype_key(q.dtype)
    if q.dtype != k.dtype:
        raise NotImplementedError(
            "attention-ts context requires Q and K to use the same dtype; "
            f"got Q {q.dtype} and K {k.dtype}"
        )
    if not v_dtype_follows_qk(q.dtype, v.dtype):
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype "
            "except for QK-BF16/PV-FP8 and Int8 Q/K with E4M3 V; "
            f"got Q {q.dtype}, K {k.dtype}, and V {v.dtype}"
        )


def validate_query_work_tile_span(config: "FmhaConfig") -> None:
    """Keep the public Int32 padding guard coupled to generated topology."""
    q_rows = int(config.q_tile_m) * int(config.work_tile_q_seq_tiles)
    if q_rows > CONTEXT_MAX_Q_ROWS_PER_WORK_TILE:
        raise RuntimeError(
            "context Int32 extent safety assumes at most "
            f"{CONTEXT_MAX_Q_ROWS_PER_WORK_TILE} Q rows per work tile, got {q_rows}"
        )


def validate_runtime_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: ContextPlanGeometry,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
) -> tuple[int, int]:
    """Validate contiguous data and request metadata against one plan."""

    validate_base_tensors(q, k, v, allow_mla=True)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    # MXFP8 Q/K plans name the E8M0 scale dtype, but the data is actually E4M3.
    qk_dtype = (
        torch.float8_e4m3fn
        if geometry.qk_dtype == torch.float8_e8m0fnu
        else geometry.qk_dtype
    )
    if q.dtype != qk_dtype:
        raise ValueError(f"q must have dtype {qk_dtype}, got {q.dtype}")
    if k.dtype != qk_dtype:
        raise ValueError(f"k must have dtype {qk_dtype}, got {k.dtype}")
    if v.dtype != geometry.pv_dtype:
        raise ValueError(f"v must have dtype {geometry.pv_dtype}, got {v.dtype}")
    v_shape = (*k.shape[:-1], geometry.head_dim_vo or geometry.head_dim)
    if tuple(v.shape) != v_shape:
        raise ValueError(f"v must have shape {v_shape}, got {tuple(v.shape)}")

    if geometry.packed_query:
        if q.ndim != 3 or tuple(q.shape[1:]) != (
            geometry.num_qo_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "q must have packed shape [total_q, Hq, D] with "
                f"Hq/D=({geometry.num_qo_heads}, {geometry.head_dim}), got "
                f"{tuple(q.shape)}"
            )
        if k.ndim != 3 or tuple(k.shape[1:]) != (
            geometry.num_kv_heads,
            geometry.head_dim,
        ):
            raise ValueError(
                "k and v must have packed shape [total_k, Hkv, D] with "
                f"Hkv/D=({geometry.num_kv_heads}, {geometry.head_dim}), got "
                f"{tuple(k.shape)}"
            )
        total_q = int(q.shape[0])
        total_k = int(k.shape[0])
        validate_padded_data_extent(total_q, "total_q")
        validate_padded_data_extent(total_k, "total_k")
        if total_q > (geometry.batch_size * geometry.max_q_len):
            raise ValueError(
                "q must contain no more than "
                "batch_size * max_q_len rows; got "
                f"{total_q} for batch_size={geometry.batch_size} and "
                f"max_q_len={geometry.max_q_len}"
            )
        if total_k > (geometry.batch_size * geometry.max_seq_len_k):
            raise ValueError(
                "k and v must contain no more than "
                "batch_size * max_kv_len rows; got "
                f"{total_k} for batch_size={geometry.batch_size} and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("packed plans require qo_indptr and kv_indptr in run()")
        request_lengths: dict[str, tuple[int, ...]] = {}
        for indptr, name, expected_total in (
            (qo_indptr, "qo_indptr", total_q),
            (kv_indptr, "kv_indptr", total_k),
        ):
            validate_indptr_tensor(indptr, name, device=geometry.device)
            if indptr.numel() != geometry.batch_size + 1:
                raise ValueError(
                    f"{name} must contain batch_size + 1 elements "
                    f"({geometry.batch_size + 1}), got {indptr.numel()}"
                )
            _, lengths = read_indptr(
                indptr,
                name,
                expected_total=expected_total,
            )
            request_lengths[name] = lengths
            capacity = (
                geometry.max_q_len if name == "qo_indptr" else geometry.max_seq_len_k
            )
            for batch_idx, length in enumerate(lengths):
                if length > capacity:
                    raise ValueError(
                        f"{name} deltas must not exceed the planned capacity "
                        f"{capacity}; batch {batch_idx} has {length}"
                    )
        q_lengths = request_lengths["qo_indptr"]
        k_lengths = request_lengths["kv_indptr"]
        if geometry.mask_type == "causal":
            for batch_idx, (q_length, k_length) in enumerate(
                zip(q_lengths, k_lengths, strict=True)
            ):
                if k_length > 0 and q_length > k_length:
                    raise ValueError(
                        "bottom-right causal context requires Sq <= Sk for each "
                        f"request; got batch {batch_idx}: Sq={q_length}, "
                        f"Sk={k_length}"
                    )
        validate_compact(q, "q", "[total_q, Hq, D]")
        validate_compact(k, "k", "[total_k, Hkv, D]")
        validate_value_layout(v)
        return max(q_lengths), max(k_lengths)
    else:
        if qo_indptr is not None or kv_indptr is not None:
            raise ValueError(
                "qo_indptr and kv_indptr require packed_query=True in plan()"
            )
        q_shape = (
            geometry.batch_size,
            geometry.max_q_len,
            geometry.num_qo_heads,
            geometry.head_dim,
        )
        kv_shape = (
            geometry.batch_size,
            fixed_kv_rows(geometry.max_seq_len_k, geometry.vc),
            geometry.num_kv_heads,
            geometry.head_dim,
        )
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != kv_shape:
            raise ValueError(
                f"k and v must have shape {kv_shape}, got {tuple(k.shape)}"
            )
        validate_compact(q, "q", "[B, Sq, Hq, D]")
        validate_compact(k, "k", "[B, Sk, Hkv, D]")
        validate_value_layout(v)
        return int(q.shape[1]), geometry.max_seq_len_k


def validate_static_extent(value: object, name: str) -> int:
    """Validate one explicit positive plan bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return validate_extent(value, name)


def validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")


def validate_value_layout(v: torch.Tensor) -> None:
    """Validate compact V heads/features with aligned, nonoverlapping rows."""

    head_dim = int(v.shape[-1])
    row_elements = int(v.shape[-2]) * head_dim
    if v.stride()[-2:] != (head_dim, 1):
        raise ValueError("v must have compact head and feature strides (D, 1)")
    token_stride = v.stride(-3)
    if token_stride < row_elements:
        raise ValueError("v token stride must not overlap adjacent token rows")
    if token_stride * v.element_size() % 16:
        raise ValueError("v token stride must be 16-byte aligned")
    if v.ndim == 4:
        batch_span = (int(v.shape[1]) - 1) * token_stride + row_elements
        if v.stride(0) < batch_span:
            raise ValueError("v batch stride must not overlap adjacent batches")
        if v.stride(0) * v.element_size() % 16:
            raise ValueError("v batch stride must be 16-byte aligned")


def validate_variable_window_bounds(
    starts: Optional[torch.Tensor],
    ends: Optional[torch.Tensor],
    *,
    geometry: ContextGeometry | ContextPlanGeometry,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Validate inclusive per-query K bounds for contiguous attention."""
    if starts is None or ends is None:
        raise ValueError(
            "variable_window_token_starts and variable_window_token_ends are "
            "required for mask_type='variable_window'"
        )
    if geometry.packed_query:
        raise NotImplementedError(
            "variable_window currently requires fixed [B, S, H, D] tensors"
        )
    expected_shape = (geometry.batch_size, geometry.max_q_len)
    for tensor, name in (
        (starts, "variable_window_token_starts"),
        (ends, "variable_window_token_ends"),
    ):
        validate_tensor(tensor, name)
        if tensor.device != geometry.device:
            raise ValueError(
                f"{name} must be on {geometry.device}, got {tensor.device}"
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        validate_compact(tensor, name, "[B, Sq]")
        validate_alignment(tensor, name, 4)
    start_values = tuple(int(value) for value in starts.flatten().tolist())
    end_values = tuple(int(value) for value in ends.flatten().tolist())
    for offset, (start, end) in enumerate(zip(start_values, end_values, strict=True)):
        if start < 0 or start > end or end >= geometry.max_seq_len_k:
            raise ValueError(
                "variable-window bounds must satisfy "
                "0 <= start <= end < max_kv_len; flattened row "
                f"{offset} has start={start}, end={end}, and "
                f"max_kv_len={geometry.max_seq_len_k}"
            )
        if offset % geometry.max_q_len and end < end_values[offset - 1]:
            batch_idx, query_idx = divmod(offset, geometry.max_q_len)
            raise ValueError(
                "variable_window_token_ends must be nondecreasing within each "
                f"batch item; batch {batch_idx}, query {query_idx} has "
                f"end={end} after end={end_values[offset - 1]}"
            )
    return starts.flatten(), ends.flatten(), start_values


def validate_variable_window_cta_starts(
    cta_starts: torch.Tensor,
    *,
    token_start_values: tuple[int, ...],
    geometry: ContextPlanGeometry,
) -> torch.Tensor:
    """Validate caller-precomputed per-CTA minimum K-token starts."""

    validate_tensor(cta_starts, "variable_window_cta_starts")
    if cta_starts.device != geometry.device:
        raise ValueError(
            "variable_window_cta_starts must be on "
            f"{geometry.device}, got {cta_starts.device}"
        )
    if cta_starts.dtype != torch.int32:
        raise ValueError(
            "variable_window_cta_starts must have dtype torch.int32, "
            f"got {cta_starts.dtype}"
        )
    tile_size_q = variable_window_q_tile_size(geometry.head_dim)
    num_seq_tiles = (geometry.max_q_len + tile_size_q - 1) // tile_size_q
    expected_shape = (geometry.batch_size, num_seq_tiles)
    if tuple(cta_starts.shape) != expected_shape:
        raise ValueError(
            "variable_window_cta_starts must have shape "
            f"{expected_shape}, got {tuple(cta_starts.shape)}"
        )
    validate_compact(cta_starts, "variable_window_cta_starts", "[B, Q tiles]")
    validate_alignment(cta_starts, "variable_window_cta_starts", 4)

    actual_values = tuple(int(value) for value in cta_starts.flatten().tolist())
    for batch_idx in range(geometry.batch_size):
        row_offset = batch_idx * geometry.max_q_len
        for tile_idx in range(num_seq_tiles):
            tile_begin = row_offset + tile_idx * tile_size_q
            tile_end = min(
                tile_begin + tile_size_q,
                row_offset + geometry.max_q_len,
            )
            expected = min(token_start_values[tile_begin:tile_end])
            actual = actual_values[batch_idx * num_seq_tiles + tile_idx]
            if actual != expected:
                raise ValueError(
                    "variable_window_cta_starts must contain the exact minimum "
                    f"token start for each Q tile; entry [{batch_idx}, "
                    f"{tile_idx}] is {actual}, expected {expected}"
                )
    return cta_starts.flatten()


def variable_window_q_tile_size(head_dim: int) -> int:
    """Return the number of Q rows sharing one variable-window CTA bound."""

    return (
        CONTEXT_MAX_Q_ROWS_PER_WORK_TILE
        if head_dim == CONTEXT_TILE_SIZE_Q
        else CONTEXT_TILE_SIZE_Q
    )
