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

"""MLA decode launch specs, argument checks, workspace, compilation and launch.

Builds go through ``compile_cache.build_and_load_attention_kernel``; see its
module docstring for how concurrent builds share the cache.
"""

import copy
import functools
import torch
from collections.abc import Callable
from dataclasses import dataclass, field
from ...common.attention_args import validate_positive_int
from ...common.hardware import get_max_active_clusters
from ...common.scales import GlobalScale, fake_scale, normalize_scale
from ...common.tensors import attention_dtype_key, validate_16byte_alignment
from ...common.workspace import (
    WorkspaceSection,
    align_up,
    append_workspace_section,
    workspace_section_view,
)
from ...compile_cache import build_and_load_attention_kernel
from .kernel.device.kv_formats import cache_record_size
from flashinfer.utils import get_device_sm_count
from typing import Any, Optional, cast


COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 2"
MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64
MLA_QUERY_DIM = MLA_LATENT_DIM + MLA_ROPE_DIM
SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16,)
INT32_MAX = 2**31 - 1
# Retain the existing broad input guard; after policy resolution, validate
# the complete padded endpoint from its actual capacity (including >128).
# This legacy coordinate reserve is not a split-count ceiling.
MLA_LEGACY_KV_COORDINATE_RESERVE = 128 * 2 * 128
MLA_MAX_KV_LEN = INT32_MAX - (MLA_LEGACY_KV_COORDINATE_RESERVE - 1)


@dataclass(frozen=True)
class MLADecodeCompileSpec:
    """Batch-independent identity and implementation for one MLA compile."""

    device_index: int
    kernel_signature: tuple[object, ...]
    num_qo_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype_key: str
    out_dtype_key: str
    max_q_len: int | None
    packed_query: bool
    has_kernel_workspace: bool
    # Active split count is a runtime operand; kernel_signature carries capacity.
    split_kv: int = field(compare=False, hash=False)
    kernel: Any = field(compare=False, hash=False, repr=False)
    write_final_lse: bool = True
    lse_mode: str = "none"
    scale_types: tuple[bool, bool, bool] = (False, False, False)
    cp_world: int = 1
    cp_interleave_size: int = 1
    zero_empty_rows: bool = False


@dataclass(frozen=True)
class MLADecodeLaunchSpec:
    """Automatic MLA policy and scratch geometry for one plan."""

    kernel: Any
    policy: tuple[tuple[str, object], ...]
    kernel_workspace_bytes: int
    split_kv: int


@dataclass(frozen=True)
class MLARuntime:
    query: torch.Tensor
    normalized_cache: torch.Tensor
    out: torch.Tensor
    num_physical_pages: int
    softmax_scale: GlobalScale
    output_scale: GlobalScale
    extra_cache: Optional[torch.Tensor] = None
    sparse_inputs: Optional[tuple] = None
    output_quant_args: Optional[tuple] = None


@dataclass(frozen=True)
class MLAWorkspaceLayout:
    """Private MLA scratch layout; only ``total_bytes`` is public."""

    kernel_workspace: WorkspaceSection
    lse: WorkspaceSection
    total_bytes: int


@dataclass(frozen=True)
class MLAWorkspaceViews:
    kernel_workspace: Optional[torch.Tensor]
    lse: Optional[torch.Tensor]


def bind_mla_workspace(
    workspace_buffer: torch.Tensor, layout: MLAWorkspaceLayout
) -> MLAWorkspaceViews:
    kernel_workspace = None
    if layout.kernel_workspace.byte_size > 0:
        kernel_workspace = workspace_section_view(
            workspace_buffer, layout.kernel_workspace
        )
    return MLAWorkspaceViews(
        kernel_workspace=kernel_workspace,
        lse=workspace_section_view(workspace_buffer, layout.lse)
        if layout.lse.byte_size
        else None,
    )


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compile_mla_decode(compile_spec: MLADecodeCompileSpec):
    import cutlass
    import cutlass.cute as cute

    num_qo_heads = compile_spec.num_qo_heads
    kv_lora_rank = compile_spec.kv_lora_rank
    qk_rope_head_dim = compile_spec.qk_rope_head_dim
    query_dim = kv_lora_rank + qk_rope_head_dim
    # The D512 path has no separate RoPE stage. Use an unused full-width
    # alias in the callable ABI rather than a zero-extent tensor map.
    rope_view_dim = qk_rope_head_dim or kv_lora_rank
    page_size = compile_spec.page_size
    max_q_len = (
        cute.sym_int() if compile_spec.max_q_len is None else compile_spec.max_q_len
    )
    packed_query = compile_spec.packed_query
    # Keep the cached spec pristine if compiled-callable caches are cleared.
    kernel = copy.copy(compile_spec.kernel)
    sparse = getattr(kernel, "direct_sparse", False)
    nvfp4_kv = getattr(kernel, "has_nvfp4_kv", False)
    dtype_map = {
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    qkv_dtype = dtype_map[compile_spec.q_dtype_key]
    out_dtype = dtype_map[compile_spec.out_dtype_key]
    physical_pages = cute.sym_int()
    batch_size = cute.sym_int()
    runtime_total_q = cute.sym_int()

    # These fake tensors pin the public ABI while allowing runtime page counts,
    # table widths/row strides, and batch metadata pointers to vary.
    q_stride_h = query_dim
    q_stride_q = num_qo_heads * query_dim
    q_latent_shape: tuple[int, ...]
    q_rope_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    if packed_query:
        q_latent_shape = (num_qo_heads, kv_lora_rank, runtime_total_q)
        q_rope_shape = (num_qo_heads, rope_view_dim, runtime_total_q)
        q_stride = (q_stride_h, 1, q_stride_q)
    else:
        q_stride_batch = max_q_len * q_stride_q
        q_latent_shape = (num_qo_heads, kv_lora_rank, max_q_len, batch_size)
        q_rope_shape = (
            num_qo_heads,
            rope_view_dim,
            max_q_len,
            batch_size,
        )
        q_stride = (q_stride_h, 1, q_stride_q, q_stride_batch)
    q_latent_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_latent_shape, stride=q_stride, assumed_align=16
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        qkv_dtype, q_rope_shape, stride=q_stride, assumed_align=16
    )
    cache_dim = (
        cache_record_size(kernel.kv_cache_format, qk_rope_head_dim)
        if nvfp4_kv
        else query_dim
    )
    extra_cache_dim = (
        cache_record_size(kernel.extra_kv_cache_format, qk_rope_head_dim)
        if nvfp4_kv
        else query_dim
    )
    cache_dtype = cutlass.Uint8 if nvfp4_kv else qkv_dtype
    cache_token_stride = cache_dim
    cache_page_stride = page_size * cache_dim
    c_latent_fake = cute.runtime.make_fake_tensor(
        cache_dtype,
        (page_size, cache_dim if nvfp4_kv else kv_lora_rank, physical_pages),
        stride=(cache_token_stride, 1, cache_page_stride),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        cache_dtype,
        (
            page_size,
            extra_cache_dim if sparse else rope_view_dim,
            cute.sym_int() if page_size == 1 else physical_pages,
        ),
        stride=(extra_cache_dim, 1, page_size * extra_cache_dim)
        if sparse
        else (cache_token_stride, 1, cache_page_stride),
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
    out_stride_row = num_qo_heads * kv_lora_rank
    out_shape: tuple[int, ...]
    out_stride: tuple[int, ...]
    lse_shape: tuple[int, ...]
    lse_stride: tuple[int, ...]
    if packed_query:
        out_shape = (num_qo_heads, kv_lora_rank, runtime_total_q)
        out_stride = (kv_lora_rank, 1, out_stride_row)
        lse_shape = (num_qo_heads, runtime_total_q)
        lse_stride = (1, num_qo_heads)
    else:
        out_stride_batch = max_q_len * out_stride_row
        out_shape = (num_qo_heads, kv_lora_rank, max_q_len, batch_size)
        out_stride = (kv_lora_rank, 1, out_stride_row, out_stride_batch)
        lse_shape = (num_qo_heads, max_q_len, batch_size)
        lse_stride = (1, num_qo_heads, max_q_len * num_qo_heads)
    out_fake = cute.runtime.make_fake_tensor(
        out_dtype, out_shape, stride=out_stride, assumed_align=16
    )
    lse_fake = (
        cute.runtime.make_fake_tensor(
            cutlass.Float32, lse_shape, stride=lse_stride, assumed_align=4
        )
        if compile_spec.write_final_lse
        else None
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
    sparse_inputs_fake = None
    if getattr(kernel, "direct_sparse", False):
        # Describe runtime index-list extents; the plan capacity does not specialize
        # this view.
        def sparse_indices(capacity):
            return cute.runtime.make_fake_tensor(
                cutlass.Int32,
                (batch_size, cute.sym_int()),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            )

        sink = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_qo_heads,), stride_order=(0,), assumed_align=4
        )
        sparse_inputs_fake = (
            sparse_indices(kernel.direct_sparse_capacities[0]),
            sparse_indices(kernel.direct_sparse_capacities[1]),
            cache_seqs_fake,
            cache_seqs_fake,
            sink,
            fake_scale(compile_spec.scale_types[2]),
        )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    output_quant_args = fake_output_quant_args(getattr(kernel, "output_quant", None))

    # Task objects carry loop-local state through generated control flow, so
    # select the public staged frontend for this compilation.
    from .kernel.device.diagnostics import mla_build_diagnostics

    with mla_build_diagnostics():
        return cute.compile[cute.FrontendNext](
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
            sparse_inputs_fake,
            fake_scale(compile_spec.scale_types[0]),
            fake_scale(compile_spec.scale_types[1]),
            stream_fake,
            output_quant_args,
            cutlass.Int32(0),
            cutlass.Int32(0),
            options=COMPILE_OPTIONS,
        )


def derive_max_seq_len_q(
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


def fake_output_quant_args(quant):
    if quant is None:
        return None
    import cutlass
    import cutlass.cute as cute

    rows = cute.sym_int()
    padded_rows = cute.sym_int(divisibility=4)
    columns, scale_stride = output_quant_scale_layout(quant, padded_rows)
    return (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float8E4M3FN,
            (cute.sym_int64(),),
            stride_order=(0,),
            assumed_align=16,
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (rows,), stride_order=(0,), assumed_align=4
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), 64), stride_order=(1, 0), assumed_align=16
        ),
        cute.runtime.make_fake_tensor(
            cutlass.Int32 if quant.ue8m0 else cutlass.Float32,
            (rows, quant.num_groups, columns),
            stride=scale_stride,
            assumed_align=16,
        ),
    )


@functools.cache
def get_compiled_mla_decode(compile_spec: MLADecodeCompileSpec):
    """Compile and cache one batch-dynamic TS MLA topology."""
    with torch.cuda.device(compile_spec.device_index):
        return build_and_load_attention_kernel(
            "mla_decode",
            "main",
            compile_spec,
            COMPILE_OPTIONS,
            functools.partial(compile_mla_decode, compile_spec),
        )


def has_runtime_query_lengths(kernel) -> bool:
    """Whether the kernel takes its Q length at launch.

    Sparse MLA flattens Q rows into virtual batch items, so its physical Q length
    stays compile-time even when the KV lengths are runtime operands.
    """
    return not kernel.fixed_seqlen and getattr(kernel, "sparse_profile", None) is None


def kernel_dtype_name(dtype_key: str) -> str:
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


def launch_mla_decode(
    runtime: MLARuntime,
    *,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    packed_query: bool,
    kv_lora_rank: int,
    split_kv: int,
    workspace: MLAWorkspaceViews,
    compiled: Callable[..., object],
    lse: Optional[torch.Tensor] = None,
    max_q_len: int,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Form the dimension-first views and launch one compiled MLA kernel."""

    lse_storage = workspace.lse if lse is None else lse
    if packed_query and int(runtime.query.shape[0]) == 0:
        return runtime.out
    if packed_query:
        q_latent = runtime.query[..., :kv_lora_rank].permute(1, 2, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(1, 2, 0)
        out_kernel = runtime.out.permute(1, 2, 0)
        total_q = int(runtime.query.shape[0])
        lse_kernel = (
            lse_storage.view(-1, lse_storage.shape[-1])[:total_q].transpose(0, 1)
            if lse_storage is not None
            else None
        )
    else:
        q_latent = runtime.query[..., :kv_lora_rank].permute(2, 3, 1, 0)
        q_rope = runtime.query[..., kv_lora_rank:].permute(2, 3, 1, 0)
        out_kernel = runtime.out.permute(2, 3, 1, 0)
        lse_kernel = lse_storage.permute(2, 1, 0) if lse_storage is not None else None
    c_latent = (
        runtime.normalized_cache
        if runtime.normalized_cache.dtype == torch.uint8
        else runtime.normalized_cache[..., :kv_lora_rank]
    ).permute(1, 2, 0)
    c_rope = runtime.normalized_cache[..., kv_lora_rank:].permute(1, 2, 0)
    if runtime.query.shape[-1] == kv_lora_rank:
        q_rope = q_latent
        c_rope = c_latent
    if runtime.extra_cache is not None:
        c_rope = runtime.extra_cache.permute(1, 2, 0)
    page_offsets = block_tables.transpose(0, 1)
    # Persisted FFI callables require positional arguments, including None.
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
        seq_lens_kv,
        qo_indptr,
        runtime.sparse_inputs,
        runtime.softmax_scale,
        runtime.output_scale,
        runtime.output_quant_args,
        max_q_len,
        cp_rank,
    )
    return runtime.out


def make_mla_decode_compile_spec(
    launch_spec: MLADecodeLaunchSpec,
    *,
    device_index: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    q_dtype_key: str,
    out_dtype_key: str,
    max_q_len: int,
    packed_query: bool,
    lse_mode: str = "none",
    runtime_kv_lengths: bool = False,
    write_final_lse: bool = True,
    scale_types: tuple[bool, bool, bool] = (False, False, False),
    cp_world: int = 1,
    cp_interleave_size: int = 1,
    zero_empty_rows: bool = True,
) -> MLADecodeCompileSpec:
    """Keep policy resolution plan-specific and JIT identity batch-free."""

    kernel = copy.copy(launch_spec.kernel)
    kernel.lse_base_on_e = False
    kernel.runtime_kv_lengths = runtime_kv_lengths
    kernel.cp_world = cp_world
    kernel.cp_interleave_size = cp_interleave_size
    kernel.zero_empty_rows = cp_world > 1 and zero_empty_rows
    return MLADecodeCompileSpec(
        device_index=device_index,
        kernel_signature=mla_kernel_compile_signature(kernel),
        num_qo_heads=num_qo_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
        q_dtype_key=q_dtype_key,
        out_dtype_key=out_dtype_key,
        max_q_len=(
            None if has_runtime_query_lengths(launch_spec.kernel) else max_q_len
        ),
        packed_query=packed_query,
        has_kernel_workspace=launch_spec.kernel_workspace_bytes > 0,
        split_kv=launch_spec.split_kv,
        kernel=kernel,
        lse_mode=lse_mode,
        write_final_lse=write_final_lse,
        scale_types=scale_types,
        cp_world=cp_world,
        cp_interleave_size=cp_interleave_size,
        zero_empty_rows=kernel.zero_empty_rows,
    )


# Dense launch specs are cached; sparse plans construct transient kernel objects.
# Reuse dense signature construction without retaining each transient sparse plan.
cached_mla_decode_compile_spec = functools.cache(make_mla_decode_compile_spec)


# Pack kernel scratch bytes and [batch, max Q tokens, heads] FP32 LSE storage.
def make_mla_workspace_layout(
    kernel_workspace_bytes: int,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int = 1,
    *,
    include_lse: bool = True,
) -> MLAWorkspaceLayout:
    kernel_workspace, byte_end = append_workspace_section(
        0, (kernel_workspace_bytes,), torch.int8
    )
    lse, byte_end = append_workspace_section(
        byte_end,
        (batch_size, max_q_len, num_qo_heads) if include_lse else (0,),
        torch.float32,
    )
    return MLAWorkspaceLayout(
        kernel_workspace=kernel_workspace,
        lse=lse,
        total_bytes=align_up(byte_end),
    )


def mla_cp_local_length(
    length: int, cp_world: int, cp_interleave_size: int, cp_rank: int
) -> int:
    """Map a global prefix to this rank's contiguous local KV prefix."""
    if cp_world == 1:
        return length
    cycles, tail = divmod(max(length, 0), cp_world * cp_interleave_size)
    return cycles * cp_interleave_size + min(
        max(tail - cp_rank * cp_interleave_size, 0), cp_interleave_size
    )


def mla_kernel_compile_signature(kernel: Any) -> tuple[object, ...]:
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


def normalize_mla_kv_cache(
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
    validate_mla_int32_extent(int(normalized.shape[0]), "kv_cache physical pages")
    if normalized.shape[2] != MLA_QUERY_DIM:
        raise ValueError(
            f"kv_cache last dimension must be {MLA_QUERY_DIM}, "
            f"got {normalized.shape[2]}"
        )
    validate_16byte_alignment(normalized, "kv_cache")
    return normalized, int(normalized.shape[0]), int(normalized.shape[1])


def output_quant_scale_layout(quant, padded_rows):
    """Return the scale columns per group and the strides of the scale view.

    Scales are stored column-major as ``[G, columns, padded_rows]`` and exposed
    as a token-major ``[T, G, columns]`` view; UE8M0 packs four scales per INT32
    column. The runtime binder and the compiled signature both use this layout.
    """
    columns = quant.num_heads // quant.num_groups * 512 // quant.block_size
    if quant.ue8m0:
        columns //= 4
    return columns, (1, columns * padded_rows, padded_rows)


def prepare_mla_runtime(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    packed_query: bool,
    qo_indptr: Optional[torch.Tensor],
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    softmax_scale: GlobalScale,
    output_scale: GlobalScale,
    out: Optional[torch.Tensor],
    validate: bool,
) -> MLARuntime:
    """Normalize one launch, optionally validating its public arguments."""

    if validate:
        if packed_query:
            if qo_indptr is None:
                raise ValueError("packed-query MLA run requires qo_indptr")
            validate_mla_qo_indptr(
                qo_indptr,
                device=device,
                batch_size=batch_size,
            )
        elif qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        validate_mla_query(
            query,
            packed_query=packed_query,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            q_dtype=q_dtype,
        )
        (
            normalized_cache,
            num_physical_pages,
            runtime_page_size,
        ) = normalize_mla_kv_cache(kv_cache, expected_device=device)
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
        effective_softmax_scale = normalize_scale(
            softmax_scale, "softmax_scale", device
        )
        effective_output_scale = normalize_scale(output_scale, "output_scale", device)
    else:
        normalized_cache = kv_cache[:, 0] if kv_cache.ndim == 4 else kv_cache
        num_physical_pages = int(normalized_cache.shape[0])
        effective_softmax_scale = softmax_scale
        effective_output_scale = output_scale
    total_q = int(query.shape[0]) if packed_query else None
    if out is None:
        out_shape = (
            (total_q, num_qo_heads, MLA_LATENT_DIM)
            if packed_query
            else (batch_size, max_q_len, num_qo_heads, MLA_LATENT_DIM)
        )
        out = torch.empty(out_shape, device=device, dtype=out_dtype)
    elif validate:
        validate_mla_out(
            out,
            device=device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=max_q_len,
            packed_query=packed_query,
            total_q=total_q,
            out_dtype=out_dtype,
        )
    return MLARuntime(
        query=query,
        normalized_cache=normalized_cache,
        out=out,
        num_physical_pages=num_physical_pages,
        softmax_scale=effective_softmax_scale,
        output_scale=effective_output_scale,
    )


def resolve_max_seq_len_q_alias(
    *,
    seq_len_q: Optional[int],
    max_q_len: Optional[int],
    default: Optional[int],
    validate: bool = True,
) -> Optional[int]:
    """Resolve the legacy fixed-Q name and the explicit static-bound name."""

    legacy_bound = (
        validate_positive_int(seq_len_q, "seq_len_q")
        if validate and seq_len_q is not None
        else seq_len_q
    )
    explicit_bound = (
        validate_positive_int(max_q_len, "max_q_len")
        if validate and max_q_len is not None
        else max_q_len
    )
    if (
        legacy_bound is not None
        and explicit_bound is not None
        and legacy_bound != explicit_bound
    ):
        raise ValueError(
            "seq_len_q and max_q_len must agree when both are provided: "
            f"got {legacy_bound} and {explicit_bound}"
        )
    if explicit_bound is not None:
        return explicit_bound
    if legacy_bound is not None:
        return legacy_bound
    return default


# CPU policies are small and tensor-free. Cache them across layers and decode
# steps; native compilation remains cached independently by its static signature.
@functools.cache
def resolve_mla_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_qo_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    max_kv_len: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    out_dtype_key: str,
    mask_type: str,
    seq_len_q: int = 1,
    softmax_rescale_threshold_log2: float = 0.0,
    cp_world: int = 1,
    cp_interleave_size: int = 1,
):
    """Resolve and cache MLA policy/workspace without compiling."""

    policy: tuple[tuple[str, Any], ...]
    max_kv_len = validate_mla_max_kv_len(max_kv_len, "max_kv_len")

    import cutlass

    from .kernel.selection import resolve_mla_kernel_policy, select_mla_ts_kernel
    from .kernel.device.query import FlatQueryTileLayout
    from .kernel.throughput_2cta.config import (
        compute_split_kv,
        compute_workspace_size as compute_2cta_workspace_size,
    )
    from .kernel.throughput_2cta.kernel import MlaDecodeTs
    from .kernel.latency_1cta.config import (
        compute_workspace_size as compute_1cta_workspace_size,
        q_tile_work_count,
        resolve_auto_flat_query_launch_shape,
        resolve_runtime_cluster_reduction_mode,
        resolve_auto_cluster_reduction_config,
        select_auto_split_kv,
    )
    from .kernel.latency_1cta.kernel import ThroughputLatencyMlaDecodeTs

    if q_dtype_key != kv_dtype_key:
        raise ValueError("the cached TS MLA compiler requires one QKV dtype")
    seq_len_q = validate_positive_int(seq_len_q, "seq_len_q")
    validate_mla_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_q_len=seq_len_q,
    )
    validate_mla_dims(kv_lora_rank, qk_rope_head_dim)
    qkv_dtype_name = kernel_dtype_name(q_dtype_key)
    output_dtype_name = kernel_dtype_name(out_dtype_key)

    with torch.cuda.device(device_index):
        max_active_one_cta_clusters = get_max_active_clusters(1)
        max_active_two_cta_clusters = get_max_active_clusters(2)
        one_cta_launch_shape = resolve_auto_flat_query_launch_shape(
            num_heads_q=num_qo_heads,
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
        two_cta_launch_shape = FlatQueryTileLayout.for_tile(
            num_qo_heads, seq_len_q, 128
        )
        two_cta_split_kv = compute_split_kv(
            batch_size=batch_size,
            num_q_tiles=two_cta_launch_shape.num_tiles,
            seq_len_kv=max_kv_len,
            mma_qk_tiler_mn=(128, 128),
            max_active_blocks=max_active_two_cta_clusters * 2,
        )
        sm_count = get_device_sm_count(torch.device("cuda", device_index))
        one_cta_split_kv = min(one_cta_split_kv, sm_count)
        two_cta_split_kv = min(two_cta_split_kv, sm_count)
        requested_policy, policy_source = resolve_mla_kernel_policy(
            None,
            num_qo_heads,
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
            # Cluster fanout is physical topology, not a rounded scratch bound.
            prepared_cfg = resolve_auto_cluster_reduction_config(
                decision.config,
                reduction_mode=None,
            )
            reduction_mode = (
                "gmem_separate"
                if decision.config.use_cluster_reduction
                and not prepared_cfg.use_cluster_reduction
                else resolve_runtime_cluster_reduction_mode(
                    prepared_cfg, reduction_mode=None
                )
            )
            capacity = (
                one_cta_split_kv
                if prepared_cfg.use_cluster_reduction or one_cta_split_kv == 1
                else sm_count
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
                logical_num_heads=num_qo_heads,
                logical_seq_len_q=seq_len_q,
                tile_size_q=launch_shape.tile_size_q,
                explicit_split_kv=None,
                split_kv_capacity=capacity,
                reduction_policy_splits=one_cta_split_kv,
                explicit_persistent=None,
                mask_type=mask_type,
                cp_world=cp_world,
                cp_interleave_size=cp_interleave_size,
                softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
            )
            kernel.runtime_split_kv = True
            final_cfg = kernel.make_config()
            split_kv = one_cta_split_kv
            workspace_size = compute_1cta_workspace_size(
                cfg=final_cfg,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = separate_reducer_provenance(
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
                num_heads=num_qo_heads,
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
            capacity = 1 if split_kv == 1 else sm_count
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
                static_split_kv=1 if capacity == 1 else None,
                split_kv_capacity=capacity,
                static_seq_len_k=None,
                qkv_dtype=qkv_dtype_name,
                out_dtype=output_dtype_name,
                rope_dim=qk_rope_head_dim,
                num_heads=num_qo_heads,
                seq_len_q=seq_len_q,
                batch_size=batch_size,
                mask_type=mask_type,
                cp_world=cp_world,
                cp_interleave_size=cp_interleave_size,
                softmax_rescale_threshold_log2=softmax_rescale_threshold_log2,
            )
            kernel.configure_parallel_reduction_topology(split_kv)
            workspace_size = compute_2cta_workspace_size(
                tile_size_q=int(launch_shape.tile_size_q),
                num_q_tiles=int(launch_shape.num_tiles),
                latent_dim=kv_lora_rank,
                batch_size=batch_size,
                split_kv=capacity,
                partial_o_dtype=cutlass.BFloat16,
                lse_dtype=cutlass.Float32,
            )
            separate_reducer_impl, reducer_cluster_size = separate_reducer_provenance(
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
        policy = (
            *policy,
            (
                "split_capacity",
                capacity,
            ),
            ("max_split_kv", sm_count),
        )
        validate_mla_policy_coordinate_span(policy, max_kv_len=max_kv_len)

    return MLADecodeLaunchSpec(
        kernel=kernel,
        policy=policy,
        kernel_workspace_bytes=int(workspace_size),
        split_kv=int(split_kv),
    )


def separate_reducer_provenance(
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


def validate_int32_cuda_tensor(
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
        validate_16byte_alignment(tensor, name)


def validate_mla_cp(cp_world: int, cp_interleave_size: int) -> None:
    """Keep compile-time CP geometry safe even on the trusted launch path."""
    for name, value in (
        ("cp_world", cp_world),
        ("cp_interleave_size", cp_interleave_size),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
        if not 1 <= value <= INT32_MAX:
            raise ValueError(f"{name} must be in [1, {INT32_MAX}]")
    if cp_world * cp_interleave_size > INT32_MAX:
        raise ValueError("cp_world * cp_interleave_size must fit int32")


def validate_mla_cp_rank(cp_rank: int, cp_world: int) -> None:
    if type(cp_rank) is not int:
        raise TypeError("cp_rank must be an int")
    if not 0 <= cp_rank < cp_world:
        raise ValueError(f"cp_rank must be in [0, {cp_world})")


def validate_mla_dims(kv_lora_rank: int, qk_rope_head_dim: int) -> None:
    kv_lora_rank = validate_positive_int(kv_lora_rank, "kv_lora_rank")
    qk_rope_head_dim = validate_positive_int(qk_rope_head_dim, "qk_rope_head_dim")
    if (kv_lora_rank, qk_rope_head_dim) != (MLA_LATENT_DIM, MLA_ROPE_DIM):
        raise NotImplementedError(
            "attention-ts MLA decode currently requires "
            f"kv_lora_rank={MLA_LATENT_DIM} and "
            f"qk_rope_head_dim={MLA_ROPE_DIM}; got "
            f"{kv_lora_rank} and {qk_rope_head_dim}"
        )


def validate_mla_dtype_pair(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    attention_dtype_key(q_dtype)
    attention_dtype_key(kv_dtype)
    attention_dtype_key(out_dtype)
    if q_dtype != kv_dtype:
        raise NotImplementedError(
            "attention-ts MLA decode requires query and KV cache to use the "
            f"same dtype; got {q_dtype} and {kv_dtype}"
        )
    if q_dtype not in SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(
            f"attention-ts MLA decode supports BF16 and FP8-E4M3 input; got {q_dtype}"
        )
    if out_dtype not in SUPPORTED_OUTPUT_DTYPES:
        raise NotImplementedError(
            "attention-ts MLA decode currently supports BF16 output only; "
            f"got {out_dtype}"
        )


def validate_mla_int32_extent(value: int, name: str) -> int:
    """Validate a flattened metadata/cache extent used by Int32 coordinates."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def validate_mla_max_kv_len(value: int, name: str) -> int:
    """Reserve the largest padded split-KV coordinate span in signed Int32."""
    value = validate_positive_int(value, name)
    if value > MLA_MAX_KV_LEN:
        raise NotImplementedError(
            f"{name} must be <= {MLA_MAX_KV_LEN} so padded MLA K/V "
            "coordinates fit in a signed int32"
        )
    return value


def validate_mla_metadata(
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
) -> tuple[torch.device, int, int]:
    validate_int32_cuda_tensor(
        block_tables,
        "block_tables",
        ndim=2,
        require_contiguous=False,
        require_16byte_alignment=False,
    )
    validate_int32_cuda_tensor(
        seq_lens_kv,
        "seq_lens_kv",
        ndim=1,
        require_16byte_alignment=False,
    )
    if block_tables.device != seq_lens_kv.device:
        raise ValueError("block_tables and seq_lens_kv must be on the same device")
    batch_size = int(seq_lens_kv.numel())
    if batch_size <= 0:
        raise ValueError("seq_lens_kv must contain at least one request")
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
    if seq_lens_kv.data_ptr() % 4 != 0:
        raise ValueError("seq_lens_kv must be 4-byte aligned")
    validate_mla_int32_extent(batch_size, "batch_size")
    validate_mla_int32_extent(int(block_tables.numel()), "block_tables elements")
    return seq_lens_kv.device, batch_size, max_num_pages


def validate_mla_nonnegative_int32_extent(value: int, name: str) -> int:
    """Validate a possibly empty flattened extent used by Int32 coordinates."""
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if value > INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


# Validate caller-owned output shape, dtype, device and 16-byte alignment.
def validate_mla_out(
    out: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    packed_query: bool,
    total_q: Optional[int] = None,
    out_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if packed_query:
        if total_q is None:
            raise ValueError("total_q is required to validate packed output")
        expected_shape: tuple[int, ...]
        expected_shape = (total_q, num_qo_heads, MLA_LATENT_DIM)
    else:
        expected_shape = (batch_size, max_q_len, num_qo_heads, MLA_LATENT_DIM)
    if out.shape != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        layout = "[total_q, H, 512]" if packed_query else "[B, SQ, H, 512]"
        raise ValueError(f"out must be compact in {layout} layout")
    validate_16byte_alignment(out, "out")


def validate_mla_policy_coordinate_span(
    policy: tuple[tuple[str, object], ...],
    *,
    max_kv_len: int = 1,
) -> None:
    """Keep the host K/V bound coupled to the automatically selected policy."""
    resolved = dict(policy)
    span = (
        int(cast(int, resolved["tile_size_kv"]))
        * int(cast(int, resolved["num_insts_kv"]))
        * max(int(cast(int, resolved.get("split_capacity", resolved["split_kv"]))), 1)
    )
    if max_kv_len + span - 1 > INT32_MAX:
        raise NotImplementedError(
            f"MLA padded K/V extent must fit signed Int32: max_kv_len={max_kv_len}, span={span}"
        )


def validate_mla_qo_indptr(
    qo_indptr: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> None:
    """Validate the public packed-query metadata without synchronizing."""

    validate_int32_cuda_tensor(
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


# Validate compact fixed or packed CUDA Q storage against optional plan bounds.
def validate_mla_query(
    query: torch.Tensor,
    *,
    packed_query: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_qo_heads: Optional[int] = None,
    max_q_len: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
    query_dim: int = MLA_QUERY_DIM,
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
    if query.dtype not in SUPPORTED_INPUT_DTYPES:
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
    if num_qo_heads is not None and query.shape[head_axis] != num_qo_heads:
        raise ValueError(
            "query head count must match the plan "
            f"({num_qo_heads}), got {query.shape[head_axis]}"
        )
    if max_q_len is not None:
        if packed_query:
            if batch_size is None:
                raise ValueError("batch_size is required to validate packed query")
            total_q = int(query.shape[0])
            if total_q > batch_size * max_q_len:
                raise ValueError(
                    "packed query total rows must be within "
                    f"[0, {batch_size * max_q_len}], got {total_q}"
                )
        elif query.shape[1] != max_q_len:
            raise ValueError(
                "fixed query length must equal the planned max_q_len "
                f"({max_q_len}), got {query.shape[1]}"
            )
        if batch_size is not None and num_qo_heads is not None:
            validate_mla_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_q_len=max_q_len,
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
    validate_16byte_alignment(query, "query")


def validate_mla_query_head_extent(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
    total_q: Optional[int] = None,
) -> None:
    """Keep fixed-capacity and packed query-head coordinates in signed Int32."""
    validate_mla_int32_extent(
        batch_size * max_q_len * num_qo_heads,
        "batch_size * max_q_len * num_qo_heads",
    )
    if total_q is not None:
        validate_mla_nonnegative_int32_extent(
            total_q * num_qo_heads,
            "total_q * num_qo_heads",
        )


def validate_mla_run_metadata(
    runtime: MLARuntime,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    required_page_columns: int,
    packed_query: bool,
    max_q_len: int,
    max_kv_len: int,
    page_size: int,
    mask_type: str,
    cp_world: int = 1,
    cp_interleave_size: int = 1,
    cp_rank: int = 0,
) -> int:
    """Validate per-run request metadata against one static MLA plan.

    Value checks synchronize CUDA metadata with the host. ``run(validate=False)``
    is the synchronization-free path for compilation and graph capture.
    """

    metadata_device, metadata_batch_size, max_num_pages = validate_mla_metadata(
        block_tables, seq_lens_kv
    )
    if metadata_device != device:
        raise ValueError(
            f"MLA metadata must be on the planned device {device}, got {metadata_device}"
        )
    if metadata_batch_size != batch_size:
        raise ValueError(
            "MLA metadata batch size must match the plan "
            f"({batch_size}), got {metadata_batch_size}"
        )
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_tables must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )

    if packed_query:
        if qo_indptr is None:
            raise ValueError("packed-query MLA run requires qo_indptr")
        validate_mla_qo_indptr(
            qo_indptr,
            device=device,
            batch_size=batch_size,
        )
        runtime_max_seq_len_q, total_q, q_lengths = derive_max_seq_len_q(
            qo_indptr,
            batch_size=batch_size,
        )
        if total_q != int(runtime.query.shape[0]):
            raise ValueError(
                "qo_indptr must end at the packed query row count "
                f"({runtime.query.shape[0]}), got {total_q}"
            )
        if runtime_max_seq_len_q > max_q_len:
            raise ValueError(
                "qo_indptr contains a per-request Q length larger than "
                f"max_q_len ({max_q_len}): got "
                f"{runtime_max_seq_len_q}"
            )
    else:
        if qo_indptr is not None:
            raise ValueError("fixed-query MLA plan does not accept qo_indptr")
        runtime_max_seq_len_q = int(runtime.query.shape[1])
        q_lengths = (runtime_max_seq_len_q,) * batch_size

    seq_lens_host = tuple(int(value) for value in seq_lens_kv.tolist())
    if cp_world > 1 and any(length < 0 for length in seq_lens_host):
        raise ValueError("global runtime KV lengths must be nonnegative with CP")
    local_lengths = tuple(
        mla_cp_local_length(length, cp_world, cp_interleave_size, cp_rank)
        for length in seq_lens_host
    )
    runtime_max_kv_len = max(local_lengths)
    if runtime_max_kv_len > max_kv_len:
        raise ValueError(
            "runtime KV metadata contains a local request longer than "
            f"max_kv_len ({max_kv_len}): got {runtime_max_kv_len}"
        )
    block_table_rows = block_tables.tolist()
    for request_idx, (row, seq_len) in enumerate(
        zip(block_table_rows, local_lengths, strict=True)
    ):
        required_pages = ceil_div(max(seq_len, 0), page_size)
        if any(
            int(page_id) < 0 or int(page_id) >= runtime.num_physical_pages
            for page_id in row[:required_pages]
        ):
            raise ValueError(
                "block_tables values for active pages must index the physical "
                f"K/V cache in [0, {runtime.num_physical_pages}); request "
                f"{request_idx} contains an invalid page ID"
            )
    if mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(q_lengths, seq_lens_host, strict=True)
        ):
            if kv_len > 0 and q_len > kv_len:
                raise ValueError(
                    "causal MLA decode requires every per-request Q length "
                    "to be no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )
    return runtime_max_seq_len_q


def validate_softmax_rescale_threshold_log2(
    threshold: object, *, q_dtype: torch.dtype
) -> float:
    """Return the threshold as float32-rounded ``float``; raise if out of range.

    Values outside ``[0, bound]`` (E4M3: 8, BF16: 32) raise instead of being
    clamped, as trtllm-gen's ``checkSkipCorrThreshold`` does.
    """

    from .kernel.device.math import canonicalize_softmax_rescale_threshold_log2

    if threshold == 0.0 and type(threshold) in (int, float):
        # Zero selects the default kernel for every dtype, including ones
        # without a threshold bound.
        return 0.0
    return canonicalize_softmax_rescale_threshold_log2(
        threshold, qkv_dtype=kernel_dtype_name(attention_dtype_key(q_dtype))
    )
