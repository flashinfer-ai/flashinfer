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

"""Task-scheduled block-sparse attention with a plan/run lifecycle.

``plan()`` inspects caller-owned canonical BSR on the GPU, allocates static
compact or bounded-dynamic prepared-route capacity, chooses a Q8/Q16/Q32
SWAPAB or Q64/Q128 KeepsAB specialization, and atomically publishes an
immutable revision.
``run()`` validates compact BSHD Q/K/V tensors and enqueues route preparation
followed by attention in one compiled adapter on the caller's CUDA stream.
"""

from collections.abc import Callable
from dataclasses import dataclass
import functools
import threading
from typing import TYPE_CHECKING, Literal

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import prims_ts_block_sparse_trace
from flashinfer.utils import ceil_div

from ._block_sparse.common import (
    _PREPARED_KV_ROUTE_SIZE,
    _SIGNED_INT32_MAX,
    _canonical_block_sparse_q_tile_size,
    _validate_sparse_block_size,
)
from ._block_sparse.inspection import _inspect_block_sparse_bsr
from ._block_sparse.prepared import _PreparedBlockSparseLayout
from ._block_sparse.plan import (
    _BlockSparsePlanState,
    _allocate_dummy_kv_valid_bits,
    _record_block_sparse_plan_ready_event,
    _serialize_plan,
    _wait_and_record_block_sparse_plan,
)
from ._block_sparse.runtime import (
    launch_block_sparse as _launch_block_sparse,
    validate_block_sparse_run as _validate_block_sparse_run,
)
from .decode import (
    _dtype_key,
    _validate_exact_compact_strides,
    _validate_mask,
    _validate_positive_int,
    _validate_runtime_device,
)

if TYPE_CHECKING:
    from .kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
# Prepared metadata remains a KV128 ABI even when a future compute kernel
# consumes multiple records per MMA tile. The current attention core consumes
# one record per KV128 compute tile.
_BLOCK_SPARSE_COMPUTE_KV_TILE_SIZE = 128
# CLC task dequeue overhead is visible for short sparse causal launches. In a
# B200 FP16 Q128/KV64 probe, fixed-top-7 latency CLC/static is 1.522 at 2.6
# waves and 0.879 at 5.2, while short rows with at most three routes remain
# 1.107 at 5.2 waves. Causal CLC is therefore selected only when waves > 5 and
# the maximum retained row has at least four routes; correctness is independent
# of this launch-policy threshold.
_CAUSAL_CLC_WAVE_THRESHOLD = 5
_CAUSAL_CLC_MIN_MAX_ROW_ROUTES = 4
# Q8/B8 no-mask and Q8/B16 with either token policy remain CLC-qualified
# through the measured R128 range. Masked Q8/B8 is different: its CLC profile
# is 7.3% faster at R12, while static is 3.4% faster at R16. Beyond R128 and
# for other unmeasured Q8 cross-geometries, prefer the qualified static path
# rather than extrapolating a low-margin scheduler result.
_Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES = 8
_Q8_B8_MASKED_CLC_MAX_ROW_ROUTES = 12
_Q8_FINE_KV_CLC_MAX_QUALIFIED_ROW_ROUTES = 128
# A SWAPAB loop schedules two KV128 route records at a time, padding an odd
# capacity. Representative B200 Q8/B8 and Q16/B16 sweeps place the crossover
# at three pairs for masked B8 and four pairs for B16; unmasked B8 benefits
# immediately. Reuse those measurements as KV-side defaults for every Swaps Q
# tile so cross-geometries do not create additional codegen policy variants.
_B8_MASKED_MIN_PARALLEL_ROUTE_PAIRS = 3
_B16_MIN_PARALLEL_ROUTE_PAIRS = 4


_BlockSparseCompileKey = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    Literal["dense", "causal"],
    bool,
    bool,
    bool,
]


@dataclass(frozen=True)
class _BlockSparseLaunchSpec:
    """Resolved launch policy and compiler key for one sparse plan."""

    policy: tuple[tuple[str, object], ...]
    compile_key: _BlockSparseCompileKey


@dataclass(frozen=True)
class _BlockSparseExecutionPolicy:
    """Compile-time choices that jointly define one sparse execution path."""

    use_persistent_scheduler: bool
    use_parallel_sparse_kv_loads: bool


def _should_consider_clc(
    *,
    q_tile_size: int,
    kv_block_size: int,
    mask_type: Literal["dense", "causal"],
    max_row_route_capacity: int,
    use_kv_valid_bits: bool,
) -> bool:
    """Return whether the common selector may choose CLC for this task."""

    if q_tile_size != 8:
        return True
    if mask_type == "causal":
        return max_row_route_capacity <= _Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES
    if kv_block_size == 8:
        if use_kv_valid_bits:
            return (
                max_row_route_capacity <= _Q8_B8_MASKED_CLC_MAX_ROW_ROUTES
            )
        return (
            max_row_route_capacity <= _Q8_FINE_KV_CLC_MAX_QUALIFIED_ROW_ROUTES
        )
    if kv_block_size == 16:
        return (
            max_row_route_capacity <= _Q8_FINE_KV_CLC_MAX_QUALIFIED_ROW_ROUTES
        )
    return max_row_route_capacity <= _Q8_CONSERVATIVE_CLC_MAX_ROW_ROUTES


def _select_parallel_sparse_kv_loads(
    *,
    kv_block_size: int,
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
    use_persistent_scheduler: bool,
) -> bool:
    """Choose two K/V issuer warps from KV-side TMA issue pressure."""

    if kv_block_size not in (8, 16):
        return False
    if not use_persistent_scheduler:
        return True

    # Base the crossover on paired route capacity so R(2n - 1) and R(2n),
    # whose final pair differs only by one padded route, share one path.
    if kv_block_size == 8 and use_kv_valid_bits:
        min_route_pairs = _B8_MASKED_MIN_PARALLEL_ROUTE_PAIRS
    elif kv_block_size == 16:
        min_route_pairs = _B16_MIN_PARALLEL_ROUTE_PAIRS
    else:
        return True
    route_capacity_pairs = (max_row_route_capacity + 1) // 2
    return route_capacity_pairs >= min_route_pairs


def _resolve_block_sparse_execution_policy(
    *,
    kv_block_size: int,
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
    use_persistent_scheduler: bool,
) -> _BlockSparseExecutionPolicy:
    """Resolve dependent codegen choices for one scheduler candidate."""

    use_parallel_sparse_kv_loads = _select_parallel_sparse_kv_loads(
        kv_block_size=kv_block_size,
        use_kv_valid_bits=use_kv_valid_bits,
        max_row_route_capacity=max_row_route_capacity,
        use_persistent_scheduler=use_persistent_scheduler,
    )
    return _BlockSparseExecutionPolicy(
        use_persistent_scheduler=use_persistent_scheduler,
        use_parallel_sparse_kv_loads=use_parallel_sparse_kv_loads,
    )


def _validate_matching_dtypes(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> str:
    dtype_key = _dtype_key(q_dtype)
    _dtype_key(kv_dtype)
    _dtype_key(output_dtype)
    if not (q_dtype == kv_dtype == output_dtype):
        raise ValueError("block-sparse requires matching Q, K/V, and output dtypes")
    if q_dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "block-sparse supports only torch.float16 and torch.bfloat16"
        )
    return dtype_key


def _validate_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
    dtype: torch.dtype,
    expected_shape: tuple[int, ...] | None = None,
    expected_device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got rank {tensor.ndim}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if expected_device is not None and tensor.device != expected_device:
        raise ValueError(
            f"{name} must be on planned device {expected_device}, got {tensor.device}"
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.numel() > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name}.numel() must fit in signed int32")
    _validate_exact_compact_strides(tensor, name, f"rank-{ndim}")
    if tensor.data_ptr() % 4 != 0:
        raise ValueError(f"{name} data pointer must be 4-byte aligned")


def _validate_plan_metadata(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_kv_heads: int,
    q_block_size: int,
) -> tuple[torch.device, int]:
    num_q_blocks = ceil_div(seq_len_q, q_block_size)
    expected_indptr_shape = (batch_size, num_kv_heads, num_q_blocks + 1)
    _validate_metadata_tensor(
        block_indptr,
        "block_indptr",
        ndim=3,
        dtype=torch.int32,
        expected_shape=expected_indptr_shape,
    )
    device = block_indptr.device
    device_index = _validate_runtime_device(device)
    _validate_metadata_tensor(
        block_indices,
        "block_indices",
        ndim=1,
        dtype=torch.int32,
        expected_device=device,
    )
    if kv_valid_bits is not None:
        _validate_metadata_tensor(
            kv_valid_bits,
            "kv_valid_bits",
            ndim=2,
            dtype=torch.uint32,
            expected_shape=(batch_size, ceil_div(seq_len_kv, 32)),
            expected_device=device,
        )
    return device, device_index


def _allocate_route_storage(
    block_indptr: torch.Tensor,
    *,
    kv_block_size: int,
    route_layout: _PreparedBlockSparseLayout,
    uniform_row_route_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate plan-owned row offsets and mutable route scratch.

    Static plans use compact capacities computed from the initial indptr.
    Dynamic plans use one declared per-row capacity, so row boundaries may
    change between runs without changing any workspace address. Int64
    arithmetic prevents plan-side offset construction from overflowing before
    the validated result is copied into the device Int32 ABI.
    """

    if uniform_row_route_capacity is None:
        row_nnz = (
            block_indptr[..., 1:].to(torch.int64)
            - block_indptr[..., :-1].to(torch.int64)
        ).reshape(-1)
        if row_nnz.numel() != route_layout.num_rows:
            raise RuntimeError("route row count does not match block_indptr geometry")
        row_route_capacity = torch.div(
            row_nnz * kv_block_size + (route_layout.kv_route_size - 1),
            route_layout.kv_route_size,
            rounding_mode="floor",
        )
        offsets_i64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=block_indptr.device),
                torch.cumsum(row_route_capacity, dim=0),
            )
        )
    else:
        expected_capacity = route_layout.num_rows * uniform_row_route_capacity
        if expected_capacity != route_layout.route_metadata_capacity:
            raise RuntimeError("uniform row capacity does not match route layout")
        offsets_i64 = torch.arange(
            route_layout.num_rows + 1,
            dtype=torch.int64,
            device=block_indptr.device,
        ) * int(uniform_row_route_capacity)
    # Keep the prefix sum asynchronous. Static capacity was produced by the
    # same inspected row-length formula, so copying its tail back to the host
    # would add a redundant plan synchronization.
    row_route_offsets = offsets_i64.to(torch.int32)
    route_workspace = torch.empty(
        route_layout.workspace_size_words,
        dtype=torch.int32,
        device=block_indptr.device,
    )
    return row_route_offsets, route_workspace


def _validate_max_blocks_per_row(
    max_blocks_per_row: int | None,
    *,
    dynamic_metadata: bool,
    seq_len_kv: int,
    kv_block_size: int,
) -> int | None:
    """Validate an optional semantic BSR row-capacity declaration."""

    if max_blocks_per_row is None:
        return None
    if isinstance(max_blocks_per_row, bool) or not isinstance(
        max_blocks_per_row, int
    ):
        raise TypeError("max_blocks_per_row must be a Python integer")
    if not dynamic_metadata:
        raise ValueError("max_blocks_per_row requires dynamic_metadata=True")
    if max_blocks_per_row < 0:
        raise ValueError("max_blocks_per_row must be non-negative")
    num_kv_blocks = ceil_div(seq_len_kv, kv_block_size)
    if max_blocks_per_row > num_kv_blocks:
        raise ValueError(
            "max_blocks_per_row cannot exceed the number of semantic KV blocks "
            f"({num_kv_blocks})"
        )
    return max_blocks_per_row


def _make_block_sparse_config(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    use_persistent_scheduler: bool,
    use_parallel_sparse_kv_loads: bool,
) -> "FmhaDecodeConfig":
    import cutlass

    from .kernels.fmha_decode.fmha_decode_config import make_decode_config

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    dtype = dtype_map[dtype_key]
    q_tile_size = _canonical_block_sparse_q_tile_size(q_block_size)
    use_keeps_mma_ab = q_tile_size >= 64
    config_args: dict[str, object] = {
        "use_keeps_mma_ab": use_keeps_mma_ab,
        "tile_size_q": q_tile_size,
        "tile_size_kv": _BLOCK_SPARSE_COMPUTE_KV_TILE_SIZE,
        "groups_tokens_heads_q": True,
        "use_block_sparse": True,
        "q_block_size": q_block_size,
        "kv_block_size": kv_block_size,
        "use_kv_valid_bits": use_kv_valid_bits,
        "num_kv_valid_words": (ceil_div(seq_len_kv, 32) if use_kv_valid_bits else 0),
        "use_parallel_sparse_kv_loads": use_parallel_sparse_kv_loads,
    }
    if use_persistent_scheduler:
        config_args["use_persistent_scheduler"] = True
    return make_decode_config(
        headdim=head_dim,
        args=config_args,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_q=num_heads,
        num_heads_kv=num_heads,
        qkv_dtype=dtype,
        o_dtype=dtype,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type=mask_type,
        auto_tuner=False,
    )


@functools.cache
def _resolve_block_sparse_launch_spec(
    device_index: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    max_row_route_capacity: int,
) -> _BlockSparseLaunchSpec:
    """Resolve and cache one validated static or CLC launch.

    ``max_row_route_capacity`` is a conservative prepared-route bound. Live
    index values and physical-tail morphology never specialize this cache
    entry. If the selected persistent profile is unsupported, retain the valid
    static profile instead.
    """

    from .kernels.fmha_decode.fmha_decode_config import _select_auto_launch_mode

    q_tile_size = _canonical_block_sparse_q_tile_size(q_block_size)
    scheduler_kv_capacity_tokens = (
        max_row_route_capacity * _PREPARED_KV_ROUTE_SIZE
    )
    mode = "static"
    if _should_consider_clc(
        q_tile_size=q_tile_size,
        kv_block_size=kv_block_size,
        mask_type=mask_type,
        max_row_route_capacity=max_row_route_capacity,
        use_kv_valid_bits=use_kv_valid_bits,
    ):
        with torch.cuda.device(device_index):
            mode = _select_auto_launch_mode(
                batch_size=batch_size,
                num_heads_kv=num_heads,
                seq_len_kv=scheduler_kv_capacity_tokens,
                num_q_tiles=ceil_div(seq_len_q, q_tile_size),
                tile_size_kv=_BLOCK_SPARSE_COMPUTE_KV_TILE_SIZE,
                persistent_min_waves=(
                    _CAUSAL_CLC_WAVE_THRESHOLD if mask_type == "causal" else 1
                ),
                persistent_min_tiles_per_cta=(
                    _CAUSAL_CLC_MIN_MAX_ROW_ROUTES if mask_type == "causal" else 1
                ),
            )

    def resolve_execution_policy(
        use_persistent_scheduler: bool,
    ) -> _BlockSparseExecutionPolicy:
        return _resolve_block_sparse_execution_policy(
            kv_block_size=kv_block_size,
            use_kv_valid_bits=use_kv_valid_bits,
            max_row_route_capacity=max_row_route_capacity,
            use_persistent_scheduler=use_persistent_scheduler,
        )

    def validate_profile(execution_policy: _BlockSparseExecutionPolicy) -> None:
        _make_block_sparse_config(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            dtype_key=dtype_key,
            mask_type=mask_type,
            use_kv_valid_bits=use_kv_valid_bits,
            use_persistent_scheduler=execution_policy.use_persistent_scheduler,
            use_parallel_sparse_kv_loads=(
                execution_policy.use_parallel_sparse_kv_loads
            ),
        )

    execution_policy = resolve_execution_policy(mode == "persistent")
    try:
        validate_profile(execution_policy)
    except ValueError:
        if not execution_policy.use_persistent_scheduler:
            raise
        execution_policy = resolve_execution_policy(False)
        validate_profile(execution_policy)

    compile_key: _BlockSparseCompileKey = (
        device_index,
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q_block_size,
        kv_block_size,
        dtype_key,
        mask_type,
        use_kv_valid_bits,
        execution_policy.use_persistent_scheduler,
        execution_policy.use_parallel_sparse_kv_loads,
    )
    policy: tuple[tuple[str, object], ...] = (
        ("tile_size_q", q_tile_size),
        (
            "use_persistent_scheduler",
            execution_policy.use_persistent_scheduler,
        ),
        ("max_row_route_capacity", max_row_route_capacity),
        # Deprecated diagnostic alias retained for existing benchmark parsers.
        ("max_execution_tiles", max_row_route_capacity),
        # Preserve the public diagnostic key for existing benchmark schemas.
        # It reports conservative scheduler capacity, not current visibility.
        ("visible_kv_tokens", scheduler_kv_capacity_tokens),
        ("execution_path", "prepared_bsr_decode"),
        ("use_kv_valid_bits", use_kv_valid_bits),
        (
            "use_parallel_sparse_kv_loads",
            execution_policy.use_parallel_sparse_kv_loads,
        ),
    )
    return _BlockSparseLaunchSpec(
        policy,
        compile_key,
    )


@functools.cache
def _get_compiled_block_sparse(
    device_index: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    use_persistent_scheduler: bool,
    use_parallel_sparse_kv_loads: bool,
) -> Callable[..., object]:
    """Compile and cache one prepare-plus-attention TVM-FFI adapter."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from .kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig
    from .kernels.fmha_decode.block_sparse_prepare import (
        _PrepareBlockSparseRoutes,
    )
    from .kernels.fmha_decode.fmha_decode_kernel import (
        fmha_block_sparse_launch,
    )

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    dtype = dtype_map[dtype_key]
    config = _make_block_sparse_config(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        dtype_key=dtype_key,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        use_persistent_scheduler=use_persistent_scheduler,
        use_parallel_sparse_kv_loads=use_parallel_sparse_kv_loads,
    )
    prepare_routes = _PrepareBlockSparseRoutes(
        batch_size=batch_size,
        num_kv_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        kv_route_size=_PREPARED_KV_ROUTE_SIZE,
        has_token_bits=use_kv_valid_bits,
    )
    Int32 = cutlass.Int32
    Float32 = cutlass.Float32

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
        prepare_routes(
            block_indptr,
            block_indices,
            kv_valid_bits,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
            stream,
        )
        # Live per-row route counts occupy the first words of run scratch.
        row_route_counts = route_workspace.iterator
        route_metadata = route_workspace.iterator + Int32(
            prepare_routes.route_metadata_base_word_offset
        )
        fmha_block_sparse_launch(
            (
                Int32(static_batch_size),
                Int32(static_num_heads),
                Int32(static_num_heads),
                Int32(static_seq_len_kv),
                Int32(static_head_dim),
            ),
            q.iterator,
            k.iterator,
            v.iterator,
            out.iterator,
            row_route_offsets.iterator,
            row_route_counts,
            route_metadata,
            sm_scale,
            stream,
            static_config,
            static_seq_len_kv,
        )

    def fake_compact(
        dtype: object, shape: tuple[object, ...], alignment: int
    ) -> object:
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=alignment,
        )

    logical_nnz = cute.sym_int()
    logical_workspace_words = cute.sym_int()
    q_shape = (batch_size, seq_len_q, num_heads, head_dim)
    kv_shape = (batch_size, seq_len_kv, num_heads, head_dim)
    q_fake = fake_compact(dtype, q_shape, 16)
    k_fake = fake_compact(dtype, kv_shape, 16)
    v_fake = fake_compact(dtype, kv_shape, 16)
    out_fake = fake_compact(dtype, q_shape, 16)
    num_q_blocks = ceil_div(seq_len_q, q_block_size)
    indptr_fake = fake_compact(Int32, (batch_size, num_heads, num_q_blocks + 1), 4)
    indices_fake = fake_compact(Int32, (logical_nnz,), 4)
    valid_bits_fake = fake_compact(
        cutlass.Uint32, (batch_size, ceil_div(seq_len_kv, 32)), 4
    )
    row_route_offsets_fake = fake_compact(
        Int32,
        (batch_size * num_heads * num_q_blocks + 1,),
        4,
    )
    route_workspace_fake = fake_compact(
        Int32,
        (logical_workspace_words,),
        4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            indptr_fake,
            indices_fake,
            valid_bits_fake,
            row_route_offsets_fake,
            route_workspace_fake,
            Int32(-1),
            Float32(1.0),
            stream_fake,
            config,
            batch_size,
            seq_len_kv,
            num_heads,
            head_dim,
            options=_COMPILE_OPTIONS,
        )


class BlockSparseTSWrapper:
    """Plan and reuse compact-BSHD block-sparse attention launches.

    Q is ``[B, Sq, H, D]`` and K/V are ``[B, Skv, H, D]``. Sparse rows are
    grouped by batch, head, and query block. Eager metadata must remain
    immutable until every run consuming its plan revision has completed on
    the GPU. Publishing a replacement revision does not make metadata from
    the old revision immediately safe to modify. CUDA Graph capture pins old
    revisions until wrapper destruction; the wrapper and its caller-owned
    metadata must outlive captured graphs.

    Metadata is immutable by default. A plan created with
    ``dynamic_metadata=True`` instead permits the caller to update retained
    indptr, block-index, and token-mask values between ordered launches. Tensor
    identities, shapes, and dtypes remain fixed, and every row must stay within
    its planned capacity, strictly increasing, unique, and in range. This is
    the intended lifecycle for CUDA Graph replay: capture only :meth:`run`,
    update retained tensors in place, then replay. Callers must synchronize
    cross-stream updates and must not modify metadata while a consuming launch
    is in flight. One plan revision owns one mutable route workspace, so
    unordered concurrent runs require distinct wrappers.
    """

    def __init__(self) -> None:
        self._plan_state: _BlockSparsePlanState | None = None
        self._plan_lock = threading.Lock()
        self._capture_pin_lock = threading.Lock()
        self._captured_plan_states: dict[int, _BlockSparsePlanState] = {}

    def _published_state(self) -> _BlockSparsePlanState:
        state = self._plan_state
        if state is None:
            raise AttributeError("plan() has not published a state")
        return state

    @property
    def _policy(self) -> tuple[tuple[str, object], ...]:
        return self._published_state().policy

    @_serialize_plan
    def plan(
        self,
        block_indptr: torch.Tensor,
        block_indices: torch.Tensor,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_block_size: int,
        kv_block_size: int,
        *,
        kv_valid_bits: torch.Tensor | None = None,
        mask_type: Literal["dense", "causal"] = "dense",
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype | None = None,
        o_data_type: torch.dtype | None = None,
        dynamic_metadata: bool = False,
        max_blocks_per_row: int | None = None,
    ) -> None:
        """Validate metadata, choose a legal profile, and compile the launch.

        ``block_indptr`` has shape
        ``[B, Hkv, ceil(Sq / q_block_size) + 1]``. Its entries are absolute
        offsets into flat ``block_indices[nnz]``; both are compact CUDA
        ``torch.int32``, and each index selects one ``kv_block_size``-token
        block. Optional ``kv_valid_bits`` is a shared batch token mask with
        shape ``[B, ceil(Skv / 32)]`` and dtype ``torch.uint32``. Token ``t``
        uses bit ``t % 32`` of word ``t // 32`` (least-significant bit first);
        one means valid, and padding bits beyond ``Skv`` are ignored. The mask
        is not replicated per head or sparse row.

        The first implementation is MHA-only (``Hq == Hkv``) with ``D=128``.
        Q, K, V, and O use one matching ``torch.float16`` or
        ``torch.bfloat16`` dtype. Runtime tensor shapes are documented by
        :meth:`run`.

        Each block size may be 8, 16, 32, or a positive multiple of 64; Q and
        KV block sizes may differ. Fine Q blocks use the corresponding
        Q8/Q16/Q32 SWAPAB tile. Coarse Q blocks divisible by 128 use Q128
        KeepsAB, and the remaining coarse sizes use Q64 KeepsAB. Fine KV blocks
        currently require a fine SWAPAB Q tile. Every run prepares canonical
        BSR into compact KV128 route metadata, and the attention core consumes
        only that metadata. This remains true when every KV block is selected;
        callers that know a pattern is dense should choose the dense FMHA API
        explicitly.

        Planning validates canonical BSR offsets and strictly increasing,
        unique, in-range block values with one GPU inspection. The same pass
        reduces the indptr-only workspace capacity without reading token-mask
        contents. Invalid metadata fails before a new revision is published.
        Inspection performs one packed D2H and synchronizes the plan stream, so
        ``plan()`` is host-synchronizing and must run outside CUDA Graph
        capture.

        Supplying ``kv_valid_bits`` always selects the masked profile; callers
        with no token-level mask should pass ``None``. ``dynamic_metadata=True``
        permits in-place updates to that supplied mask, block indices, and
        indptr values between ordered runs. Tensor identities and extents stay
        fixed. Updated rows must remain strictly increasing, unique, in range,
        and within their planned capacities; they are not reinspected by
        :meth:`run`. Callers whose replay can use more indices than the initial
        pattern must allocate ``block_indices`` at its maximum extent before
        planning; entries outside the initial indptr ranges act as spare
        storage and are ignored by inspection. Device guards prevent invalid
        ranges, IDs, or capacities from causing out-of-bounds accesses, but
        contract-invalid metadata does not raise a host error and its output
        must not be consumed.

        Dynamic plans reserve one uniform row capacity.
        ``max_blocks_per_row`` declares it in semantic BSR blocks; when
        omitted, the largest initial row's conservative route capacity is the
        compatibility bound. Static plans retain compact per-row allocation.
        This keeps run free of host synchronization, prefix scans, and dynamic
        allocation while allowing row boundaries to move within a known
        envelope. Plans with a supplied token mask use the masked profile;
        Q64/Q128 derive their route-full fast path from current token words on
        every run.

        Concurrent plans are serialized; run keeps using the published state.
        One revision has one mutable route workspace, so its runs must be
        ordered on one stream or externally synchronized. Unordered concurrent
        runs require distinct wrappers.
        """

        batch_size = _validate_positive_int(batch_size, "batch_size")
        seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
        seq_len_kv = _validate_positive_int(seq_len_kv, "seq_len_kv")
        num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
        num_kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
        head_dim = _validate_positive_int(head_dim, "head_dim")
        if not isinstance(dynamic_metadata, bool):
            raise TypeError("dynamic_metadata must be a bool")
        q_tile_size = _canonical_block_sparse_q_tile_size(q_block_size)
        kv_block_size = _validate_sparse_block_size(
            kv_block_size,
            "kv_block_size",
        )
        max_blocks_per_row = _validate_max_blocks_per_row(
            max_blocks_per_row,
            dynamic_metadata=dynamic_metadata,
            seq_len_kv=seq_len_kv,
            kv_block_size=kv_block_size,
        )
        use_keeps_mma_ab = q_tile_size >= 64
        if kv_block_size < 64 and use_keeps_mma_ab:
            raise ValueError("fine KV blocks require a SwapsMmaAb Q tile")
        _validate_mask(mask_type)
        if head_dim != 128:
            raise ValueError("block-sparse requires head_dim=128")
        if num_qo_heads != num_kv_heads:
            raise ValueError("block-sparse requires MHA with Hq == Hkv")
        if mask_type == "causal" and seq_len_q > seq_len_kv:
            raise ValueError("causal block-sparse requires seq_len_q <= seq_len_kv")
        if kv_data_type is None:
            kv_data_type = q_data_type
        if o_data_type is None:
            o_data_type = q_data_type
        dtype_key = _validate_matching_dtypes(
            q_data_type,
            kv_data_type,
            o_data_type,
        )
        device, device_index = _validate_plan_metadata(
            block_indptr,
            block_indices,
            kv_valid_bits,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_kv_heads=num_kv_heads,
            q_block_size=q_block_size,
        )

        plan_stream = torch.cuda.current_stream(device)
        inspection = _inspect_block_sparse_bsr(
            block_indptr,
            block_indices,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_route_size=_PREPARED_KV_ROUTE_SIZE,
            stream=plan_stream,
        )
        # Mask values are run-time data. Full routes are detected from current
        # prepared words, never frozen as a plan-time morphology specialization.
        use_token_mask = kv_valid_bits is not None
        # Live indices and physical-tail shape never specialize the launch.
        # Dynamic plans use a uniform row envelope so indptr boundaries may
        # move without changing the route workspace ABI.
        max_row_route_capacity = inspection.max_row_route_capacity
        if max_blocks_per_row is not None:
            if max_blocks_per_row < inspection.max_row_block_count:
                raise ValueError(
                    "max_blocks_per_row is smaller than an initial BSR row"
                )
            declared_route_capacity = ceil_div(
                max_blocks_per_row * kv_block_size,
                _PREPARED_KV_ROUTE_SIZE,
            )
            max_row_route_capacity = declared_route_capacity
        num_rows = batch_size * num_kv_heads * ceil_div(seq_len_q, q_block_size)
        total_route_capacity = inspection.total_route_capacity
        if dynamic_metadata:
            total_route_capacity = num_rows * max_row_route_capacity
        route_layout = _PreparedBlockSparseLayout.create(
            kv_route_size=_PREPARED_KV_ROUTE_SIZE,
            kv_block_size=kv_block_size,
            has_token_bits=use_token_mask,
            route_metadata_capacity=total_route_capacity,
            num_rows=num_rows,
        )
        with torch.cuda.device(device_index), torch.cuda.stream(plan_stream):
            spec = _resolve_block_sparse_launch_spec(
                device_index,
                batch_size,
                seq_len_q,
                seq_len_kv,
                num_qo_heads,
                head_dim,
                q_block_size,
                kv_block_size,
                dtype_key,
                mask_type,
                use_token_mask,
                max_row_route_capacity,
            )
            policy = (
                *spec.policy,
                ("dynamic_metadata", dynamic_metadata),
                ("max_blocks_per_row", max_blocks_per_row),
            )
            compiled = _get_compiled_block_sparse(*spec.compile_key)
            effective_kv_valid_bits = (
                kv_valid_bits
                if use_token_mask
                else _allocate_dummy_kv_valid_bits(
                    batch_size=batch_size,
                    seq_len_kv=seq_len_kv,
                    device=device,
                )
            )
            (
                row_route_offsets,
                route_workspace,
            ) = _allocate_route_storage(
                block_indptr,
                kv_block_size=kv_block_size,
                route_layout=route_layout,
                uniform_row_route_capacity=(
                    max_row_route_capacity if dynamic_metadata else None
                ),
            )
            ready_event = _record_block_sparse_plan_ready_event(plan_stream)

        candidate = _BlockSparsePlanState(
            device=device,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_qo_heads,
            head_dim=head_dim,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            output_dtype=o_data_type,
            block_indptr=block_indptr,
            block_indices=block_indices,
            kv_valid_bits=effective_kv_valid_bits,
            route_layout=route_layout,
            row_route_offsets=row_route_offsets,
            route_workspace=route_workspace,
            max_blocks_per_row=max_blocks_per_row,
            policy=policy,
            compiled=compiled,
            ready_event=ready_event,
            ready_stream_handle=plan_stream.cuda_stream,
        )
        # This is the only wrapper mutation. Every failure above leaves the
        # previously published revision intact and runnable.
        self._plan_state = candidate

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch the current plan on the caller's current CUDA stream.

        ``q``, ``k``, and ``v`` use compact BSHD with the shapes and dtypes
        fixed by ``plan()``. If supplied, ``out`` must match Q's shape and the
        planned output dtype. The returned tensor is exactly ``out`` when one
        was supplied; otherwise it is a newly allocated compact BSHD tensor.
        Only O is returned; this PrimTS API does not return LSE. The launch is
        enqueued asynchronously on the caller's current CUDA stream.

        Keep this wrapper alive until every captured CUDA Graph is destroyed.
        """

        state = self._plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        run_stream = torch.cuda.current_stream(state.device)
        if torch.cuda.is_current_stream_capturing():
            with self._capture_pin_lock:
                self._captured_plan_states.setdefault(id(state), state)
        _wait_and_record_block_sparse_plan(state, run_stream)
        run_args = _validate_block_sparse_run(
            q,
            k,
            v,
            block_indptr=state.block_indptr,
            block_indices=state.block_indices,
            kv_valid_bits=state.kv_valid_bits,
            device=state.device,
            batch_size=state.batch_size,
            seq_len_q=state.seq_len_q,
            seq_len_kv=state.seq_len_kv,
            num_heads=state.num_heads,
            head_dim=state.head_dim,
            q_dtype=state.q_dtype,
            kv_dtype=state.kv_dtype,
            output_dtype=state.output_dtype,
            sm_scale=sm_scale,
            out=out,
        )
        return _launch_block_sparse(
            run_args,
            block_indptr=state.block_indptr,
            block_indices=state.block_indices,
            kv_valid_bits=state.kv_valid_bits,
            row_route_offsets=state.row_route_offsets,
            route_workspace=state.route_workspace,
            max_blocks_per_row=state.max_blocks_per_row,
            compiled=state.compiled,
        )


@flashinfer_api(trace=prims_ts_block_sparse_trace)
def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    *,
    kv_valid_bits: torch.Tensor | None = None,
    mask_type: Literal["dense", "causal"] = "dense",
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plan and run one compact-BSHD block-sparse attention launch.

    Metadata follows :meth:`BlockSparseTSWrapper.plan`; use the wrapper
    directly when reusing the same metadata and tensor geometry. This one-shot
    form performs inspection on every call and therefore cannot itself be
    invoked inside CUDA Graph capture; plan a wrapper outside capture and
    capture only ``run()`` instead.
    """

    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank 4 compact BSHD")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    batch_size, seq_len_q, num_qo_heads, head_dim = map(int, q.shape)
    kv_batch_size, seq_len_kv, num_kv_heads, kv_head_dim = map(int, k.shape)
    if kv_batch_size != batch_size or kv_head_dim != head_dim:
        raise ValueError("Q and K batch/head dimensions must agree")
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError("K and V must have identical shapes")

    wrapper = BlockSparseTSWrapper()
    wrapper.plan(
        block_indptr,
        block_indices,
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_block_size,
        kv_block_size,
        kv_valid_bits=kv_valid_bits,
        mask_type=mask_type,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=q.dtype if out is None else out.dtype,
    )
    return wrapper.run(q, k, v, sm_scale=sm_scale, out=out)


__all__ = [
    "BlockSparseTSWrapper",
    "block_sparse_attention",
]
