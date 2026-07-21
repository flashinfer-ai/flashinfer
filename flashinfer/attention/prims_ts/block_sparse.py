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

``plan()`` inspects caller-owned canonical BSR on the GPU, chooses a raw-BSR
Q64/Q128 specialization, compiles/caches that launch, and atomically publishes
an immutable revision. ``run()`` validates compact BSHD Q/K/V tensors and
launches the published revision without preparing or copying sparse routes.
"""

from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass
import functools
import threading
from typing import TYPE_CHECKING, Literal

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import prims_ts_block_sparse_trace_dispatch

from ._block_sparse.inspection import _inspect_block_sparse_bsr
from ._block_sparse.plan import (
    _BlockSparseCompileKey,
    _BlockSparsePlanState,
    _BlockSparseExecutionGeometry as _BlockSparseExecutionGeometry,
    _BlockSparseLaunchSpec,
    _allocate_dummy_kv_valid_bits,
    _record_block_sparse_plan_ready_event,
    _resolve_execution_geometry,
    _serialize_plan,
    _wait_and_record_block_sparse_plan,
)
from ._block_sparse.runtime import (
    launch_block_sparse as _launch_block_sparse,
    prepare_block_sparse_runtime as _prepare_block_sparse_runtime,
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
# Comparing a token word with all-ones has its own instruction/branch cost. The
# inspector therefore compiles this fast path only when at least half of the
# execution-weighted sites can skip 32 per-token predicates. The heuristic only
# selects generated code; either specialization has identical mask semantics.
_TOKEN_WORD_ALL_VALID_FAST_PATH_MIN_RATIO_NUMERATOR = 1
_TOKEN_WORD_ALL_VALID_FAST_PATH_MIN_RATIO_DENOMINATOR = 2
_Q128_ROUTE_ALL_VALID_FAST_PATH_MIN_RATIO_NUMERATOR = 1
_Q128_ROUTE_ALL_VALID_FAST_PATH_MIN_RATIO_DENOMINATOR = 2
# CLC task dequeue overhead is visible for short sparse causal launches. In a
# B200 FP16 Q128/KV64 probe, fixed-top-7 latency CLC/static is 1.522 at 2.6
# waves and 0.879 at 5.2, while short rows with at most three routes remain
# 1.107 at 5.2 waves. Causal CLC is therefore selected only when waves > 5 and
# the maximum retained row has at least four routes; correctness is independent
# of this launch-policy threshold.
_CAUSAL_CLC_WAVE_THRESHOLD = 5
_CAUSAL_CLC_MIN_MAX_ROW_ROUTES = 4


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_matching_dtypes(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    for dtype in (q_dtype, kv_dtype, output_dtype):
        _dtype_key(dtype)
    if not (q_dtype == kv_dtype == output_dtype):
        raise ValueError("block-sparse requires matching Q, K/V, and output dtypes")
    if q_dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "block-sparse supports only torch.float16 and torch.bfloat16"
        )


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
    num_q_blocks = _ceil_div(seq_len_q, q_block_size)
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
            expected_shape=(batch_size, _ceil_div(seq_len_kv, 32)),
            expected_device=device,
        )
    return device, device_index


def _make_block_sparse_config(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    q_tile_size: int,
    q_dtype_key: str,
    output_dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    use_persistent_scheduler: bool,
    use_token_word_full_guard: bool = False,
    use_q128_token_route_full_guard: bool = False,
) -> "FmhaDecodeConfig":
    import cutlass

    from .kernels.fmha_decode.fmha_decode_config import make_decode_config

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    geometry = _resolve_execution_geometry(
        q_block_size,
        kv_block_size,
        q_tile_size=q_tile_size,
    )
    config_args: dict[str, object] = {
        "use_keeps_mma_ab": True,
        "tile_size_q": geometry.q_tile_size,
        "tile_size_kv": geometry.kv_tile_size,
        "groups_tokens_heads_q": True,
        "use_block_sparse": True,
        "q_block_size": q_block_size,
        "kv_block_size": kv_block_size,
        "use_kv_valid_bits": use_kv_valid_bits,
        "num_kv_valid_words": (_ceil_div(seq_len_kv, 32) if use_kv_valid_bits else 0),
        "use_token_word_full_guard": use_token_word_full_guard,
        "use_q128_token_route_full_guard": use_q128_token_route_full_guard,
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
        qkv_dtype=dtype_map[q_dtype_key],
        o_dtype=dtype_map[output_dtype_key],
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type=mask_type,
        auto_tuner=False,
    )


@dataclass(frozen=True)
class _RawBlockSparseLaunchTraits:
    """Hashable compile-time traits shared by static and persistent profiles."""

    device_index: int
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    head_dim: int
    q_block_size: int
    kv_block_size: int
    q_tile_size: int
    q_dtype_key: str
    kv_dtype_key: str
    output_dtype_key: str
    mask_type: Literal["dense", "causal"]
    use_kv_valid_bits: bool
    use_token_word_full_guard: bool = False
    use_q128_token_route_full_guard: bool = False


@dataclass(frozen=True)
class _RawBlockSparseLaunchProfile:
    """One scheduler choice: resolved config plus its exact compiler key."""

    config: "FmhaDecodeConfig"
    use_persistent_scheduler: bool
    compile_key: _BlockSparseCompileKey


def _make_raw_block_sparse_launch_profile(
    traits: _RawBlockSparseLaunchTraits,
    *,
    use_persistent_scheduler: bool,
) -> _RawBlockSparseLaunchProfile:
    """Build one raw-BSR scheduler specialization."""

    if traits.q_dtype_key != traits.kv_dtype_key:
        raise ValueError("the cached sparse compiler requires one QKV dtype")
    config = _make_block_sparse_config(
        batch_size=traits.batch_size,
        seq_len_q=traits.seq_len_q,
        seq_len_kv=traits.seq_len_kv,
        num_heads=traits.num_heads,
        head_dim=traits.head_dim,
        q_block_size=traits.q_block_size,
        kv_block_size=traits.kv_block_size,
        q_tile_size=traits.q_tile_size,
        q_dtype_key=traits.q_dtype_key,
        output_dtype_key=traits.output_dtype_key,
        mask_type=traits.mask_type,
        use_kv_valid_bits=traits.use_kv_valid_bits,
        use_persistent_scheduler=use_persistent_scheduler,
        use_token_word_full_guard=traits.use_token_word_full_guard,
        use_q128_token_route_full_guard=(traits.use_q128_token_route_full_guard),
    )
    compile_key: _BlockSparseCompileKey = (
        traits.device_index,
        traits.batch_size,
        traits.seq_len_q,
        traits.seq_len_kv,
        traits.num_heads,
        traits.head_dim,
        traits.q_block_size,
        traits.kv_block_size,
        traits.q_tile_size,
        traits.q_dtype_key,
        traits.kv_dtype_key,
        traits.output_dtype_key,
        traits.mask_type,
        traits.use_kv_valid_bits,
        use_persistent_scheduler,
        traits.use_token_word_full_guard,
        traits.use_q128_token_route_full_guard,
    )
    return _RawBlockSparseLaunchProfile(
        config=config,
        use_persistent_scheduler=use_persistent_scheduler,
        compile_key=compile_key,
    )


@functools.cache
def _resolve_cached_raw_block_sparse_static_launch_profile(
    traits: _RawBlockSparseLaunchTraits,
) -> _RawBlockSparseLaunchProfile:
    return _make_raw_block_sparse_launch_profile(
        traits,
        use_persistent_scheduler=False,
    )


@functools.cache
def _resolve_cached_raw_block_sparse_persistent_launch_profile(
    traits: _RawBlockSparseLaunchTraits,
) -> _RawBlockSparseLaunchProfile | None:
    static_profile = _resolve_cached_raw_block_sparse_static_launch_profile(traits)
    persistent_probe = deepcopy(static_profile.config)
    persistent_probe.use_persistent_scheduler = True
    if not persistent_probe.supports_grouped_keeps:
        return None
    return _make_raw_block_sparse_launch_profile(
        traits,
        use_persistent_scheduler=True,
    )


def _resolve_raw_block_sparse_launch_spec(
    device_index: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_block_size: int,
    kv_block_size: int,
    q_tile_size: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    max_retained_routes: int,
    *,
    use_token_word_full_guard: bool = False,
    use_q128_token_route_full_guard: bool = False,
) -> _BlockSparseLaunchSpec:
    """Select static or CLC scheduling without preparing route payloads.

    ``max_retained_routes`` is measured in physical-tail-trimmed KV128 routes,
    which is the scheduler's work unit; it is not semantic BSR row NNZ.
    """

    from .kernels.fmha_decode.fmha_decode_config import _select_auto_launch_mode

    geometry = _resolve_execution_geometry(
        q_block_size,
        kv_block_size,
        q_tile_size=q_tile_size,
    )
    visible_kv_tokens = max_retained_routes * geometry.kv_tile_size
    with torch.cuda.device(device_index):
        mode = _select_auto_launch_mode(
            batch_size=batch_size,
            num_heads_kv=num_heads,
            seq_len_kv=visible_kv_tokens,
            num_q_tiles=_ceil_div(seq_len_q, geometry.q_tile_size),
            tile_size_kv=geometry.kv_tile_size,
            persistent_min_waves=(
                _CAUSAL_CLC_WAVE_THRESHOLD if mask_type == "causal" else 1
            ),
            persistent_min_tiles_per_cta=(
                _CAUSAL_CLC_MIN_MAX_ROW_ROUTES if mask_type == "causal" else 1
            ),
        )
    traits = _RawBlockSparseLaunchTraits(
        device_index=device_index,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_tile_size=q_tile_size,
        q_dtype_key=q_dtype_key,
        kv_dtype_key=kv_dtype_key,
        output_dtype_key=output_dtype_key,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        use_token_word_full_guard=use_token_word_full_guard,
        use_q128_token_route_full_guard=use_q128_token_route_full_guard,
    )
    profile = None
    if mode == "persistent":
        profile = _resolve_cached_raw_block_sparse_persistent_launch_profile(traits)
    if profile is None:
        profile = _resolve_cached_raw_block_sparse_static_launch_profile(traits)
    policy: tuple[tuple[str, object], ...] = (
        ("tile_size_q", geometry.q_tile_size),
        ("use_persistent_scheduler", profile.use_persistent_scheduler),
        ("max_execution_tiles", max_retained_routes),
        ("visible_kv_tokens", visible_kv_tokens),
        ("execution_path", "raw_bsr_decode"),
        ("use_kv_valid_bits", use_kv_valid_bits),
        ("use_token_word_full_guard", use_token_word_full_guard),
        (
            "use_q128_token_route_full_guard",
            use_q128_token_route_full_guard,
        ),
    )
    return _BlockSparseLaunchSpec(
        copy(profile.config),
        policy,
        profile.compile_key,
    )


def _clear_block_sparse_launch_profile_cache() -> None:
    """Clear raw scheduler-profile caches for isolated tests."""

    _resolve_cached_raw_block_sparse_persistent_launch_profile.cache_clear()
    _resolve_cached_raw_block_sparse_static_launch_profile.cache_clear()


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
    q_tile_size: int,
    q_dtype_key: str,
    kv_dtype_key: str,
    output_dtype_key: str,
    mask_type: Literal["dense", "causal"],
    use_kv_valid_bits: bool,
    use_persistent_scheduler: bool,
    use_token_word_full_guard: bool = False,
    use_q128_token_route_full_guard: bool = False,
) -> Callable[..., object]:
    """Compile and cache one canonical raw-BSR TVM-FFI adapter."""

    if q_dtype_key != kv_dtype_key:
        raise ValueError("the cached sparse compiler requires one QKV dtype")
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from .kernels.fmha_decode.fmha_decode_config import FmhaDecodeConfig
    from .kernels.fmha_decode.fmha_decode_kernel import (
        fmha_block_sparse_launch,
    )

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
    }
    qkv_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    config = _make_block_sparse_config(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_tile_size=q_tile_size,
        q_dtype_key=q_dtype_key,
        output_dtype_key=output_dtype_key,
        mask_type=mask_type,
        use_kv_valid_bits=use_kv_valid_bits,
        use_persistent_scheduler=use_persistent_scheduler,
        use_token_word_full_guard=use_token_word_full_guard,
        use_q128_token_route_full_guard=use_q128_token_route_full_guard,
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
        sm_scale: cutlass.Float32,
        stream: cuda_drv.CUstream,
        static_config: cutlass.Constexpr[FmhaDecodeConfig],
        static_batch_size: cutlass.Constexpr[int],
        static_seq_len_kv: cutlass.Constexpr[int],
        static_num_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
    ) -> None:
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
            block_indptr.iterator,
            block_indices.iterator,
            kv_valid_bits.iterator,
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
    q_shape = (batch_size, seq_len_q, num_heads, head_dim)
    kv_shape = (batch_size, seq_len_kv, num_heads, head_dim)
    q_fake = fake_compact(qkv_dtype, q_shape, 16)
    k_fake = fake_compact(qkv_dtype, kv_shape, 16)
    v_fake = fake_compact(qkv_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, q_shape, 16)
    num_q_blocks = _ceil_div(seq_len_q, q_block_size)
    indptr_fake = fake_compact(Int32, (batch_size, num_heads, num_q_blocks + 1), 4)
    indices_fake = fake_compact(Int32, (logical_nnz,), 4)
    valid_bits_fake = fake_compact(
        cutlass.Uint32, (batch_size, _ceil_div(seq_len_kv, 32)), 4
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
    metadata must outlive captured graphs and remain immutable across replans.
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
    def _planned(self) -> bool:
        return self._plan_state is not None

    @property
    def _policy(self) -> tuple[tuple[str, object], ...]:
        return self._published_state().policy

    @property
    def _block_indptr(self) -> torch.Tensor:
        return self._published_state().block_indptr

    @property
    def _block_indices(self) -> torch.Tensor:
        return self._published_state().block_indices

    @property
    def _kv_valid_bits(self) -> torch.Tensor | None:
        return self._published_state().kv_valid_bits

    @property
    def _compiled(self) -> Callable[..., object]:
        return self._published_state().compiled

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

        Both block sizes must each be a positive multiple of 64; they may differ.
        Query tiles follow the semantic row size: ``q_block_size`` values
        divisible by 128 use Q128, and the remaining supported values use Q64.
        The kernel consumes the canonical BSR directly and expands selected KV
        blocks into KV128 execution routes. This remains true when every KV
        block is selected; callers that know a pattern is dense should choose
        the dense FMHA API explicitly.

        Planning validates canonical BSR offsets and strictly increasing,
        unique, in-range block values with one GPU inspection. Invalid metadata
        fails before a new revision is published; plan does not allocate a
        route payload. Inspection performs one packed D2H and synchronizes the
        plan stream, so ``plan()`` is host-synchronizing and must run outside
        CUDA Graph capture.

        Concurrent plans are serialized; run keeps using the published state.
        """

        batch_size = _validate_positive_int(batch_size, "batch_size")
        seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
        seq_len_kv = _validate_positive_int(seq_len_kv, "seq_len_kv")
        num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
        num_kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
        head_dim = _validate_positive_int(head_dim, "head_dim")
        geometry = _resolve_execution_geometry(q_block_size, kv_block_size)
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
        _validate_matching_dtypes(q_data_type, kv_data_type, o_data_type)
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

        previous_state = self._plan_state
        revision = 0 if previous_state is None else previous_state.revision + 1
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
            kv_valid_bits=kv_valid_bits,
            stream=plan_stream,
        )
        max_row_nnz = inspection.max_row_nnz

        # Turn the packed inspection into compile-time policy:
        #   reachable holes       -> compile token masking;
        #   guard skips/checks    -> specialize the common full-word branch;
        #   Q128 mask-full/routes -> specialize the whole-route branch;
        #   max retained routes   -> size the static/CLC scheduler workload.
        # An all-one user mask therefore compiles like no mask because the
        # inspector reports holes only among execution-reachable real tokens.
        use_token_mask = kv_valid_bits is not None and inspection.token_mask_has_holes
        # check_count = sum(q_tiles_per_row * padded_route_slots *
        #                   (2 if Q64 else 4)).  skip_count counts those same
        # sites whose word pair/word is all valid.
        use_token_word_full_guard = (
            use_token_mask
            and inspection.runtime_token_guard_check_count > 0
            and inspection.runtime_token_guard_skip_count
            * _TOKEN_WORD_ALL_VALID_FAST_PATH_MIN_RATIO_DENOMINATOR
            >= inspection.runtime_token_guard_check_count
            * _TOKEN_WORD_ALL_VALID_FAST_PATH_MIN_RATIO_NUMERATOR
        )
        # Only Q128 evaluates four independent word guards per scheduled route.
        # The check count includes minimum-schedule and odd-pair dummy routes;
        # the exact mask-full count includes only real routes. Q64 evaluates two
        # paired-word guards, so this Q128-only diagnostic remains zero there.
        runtime_scheduled_route_count = 0
        if geometry.q_tile_size == 128:
            if inspection.runtime_token_guard_check_count % 4 != 0:
                raise RuntimeError(
                    "Q128 inspection token-guard check count must be divisible by four"
                )
            runtime_scheduled_route_count = (
                inspection.runtime_token_guard_check_count // 4
            )
        use_q128_token_route_full_guard = (
            use_token_word_full_guard
            and geometry.q_tile_size == 128
            and runtime_scheduled_route_count > 0
            and inspection.runtime_token_mask_full_route_count
            * _Q128_ROUTE_ALL_VALID_FAST_PATH_MIN_RATIO_DENOMINATOR
            >= runtime_scheduled_route_count
            * _Q128_ROUTE_ALL_VALID_FAST_PATH_MIN_RATIO_NUMERATOR
        )
        with torch.cuda.device(device_index), torch.cuda.stream(plan_stream):
            q_dtype_key = _dtype_key(q_data_type)
            kv_dtype_key = _dtype_key(kv_data_type)
            output_dtype_key = _dtype_key(o_data_type)
            spec = _resolve_raw_block_sparse_launch_spec(
                device_index,
                batch_size,
                seq_len_q,
                seq_len_kv,
                num_qo_heads,
                head_dim,
                q_block_size,
                kv_block_size,
                geometry.q_tile_size,
                q_dtype_key,
                kv_dtype_key,
                output_dtype_key,
                mask_type,
                use_token_mask,
                inspection.max_retained_routes,
                use_token_word_full_guard=use_token_word_full_guard,
                use_q128_token_route_full_guard=(use_q128_token_route_full_guard),
            )
            config = spec.config
            policy = (
                *spec.policy,
                ("max_row_nnz", max_row_nnz),
                (
                    "runtime_full_guard_slots",
                    inspection.runtime_token_guard_skip_count,
                ),
                (
                    "runtime_guard_slots",
                    inspection.runtime_token_guard_check_count,
                ),
                (
                    "runtime_full_routes",
                    inspection.runtime_token_mask_full_route_count,
                ),
                ("runtime_routes", runtime_scheduled_route_count),
            )
            compiled = _get_compiled_block_sparse(*spec.compile_key)
            runtime_kv_valid_bits = (
                kv_valid_bits
                if use_token_mask
                else _allocate_dummy_kv_valid_bits(
                    batch_size=batch_size,
                    seq_len_kv=seq_len_kv,
                    device=device,
                )
            )
            ready_event = _record_block_sparse_plan_ready_event(plan_stream)

        candidate = _BlockSparsePlanState(
            revision=revision,
            device=device,
            device_index=device_index,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_qo_heads,
            head_dim=head_dim,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            geometry=geometry,
            q_dtype=q_data_type,
            kv_dtype=kv_data_type,
            output_dtype=o_data_type,
            mask_type=mask_type,
            block_indptr=block_indptr,
            block_indices=block_indices,
            # Retain only a mask consumed by run. Once inspection proves an
            # optional mask is all-valid, the unmasked specialization and its
            # plan-owned dummy no longer need to extend the caller tensor's
            # lifetime.
            kv_valid_bits=kv_valid_bits if use_token_mask else None,
            runtime_kv_valid_bits=runtime_kv_valid_bits,
            max_row_nnz=max_row_nnz,
            config=config,
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
        runtime = _prepare_block_sparse_runtime(
            q,
            k,
            v,
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
            runtime,
            block_indptr=state.block_indptr,
            block_indices=state.block_indices,
            runtime_kv_valid_bits=state.runtime_kv_valid_bits,
            compiled=state.compiled,
        )


@flashinfer_api(trace=prims_ts_block_sparse_trace_dispatch)
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
