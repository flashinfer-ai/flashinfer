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
Q8/Q16/Q32 SWAPAB or Q64/Q128 KeepsAB specialization, compiles/caches that
launch, and atomically publishes an immutable revision. ``run()`` validates
compact BSHD Q/K/V tensors and launches the published revision without
preparing or copying sparse routes.
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
# CLC task dequeue overhead is visible for short sparse causal launches. In a
# B200 FP16 Q128/KV64 probe, fixed-top-7 latency CLC/static is 1.522 at 2.6
# waves and 0.879 at 5.2, while short rows with at most three routes remain
# 1.107 at 5.2 waves. Causal CLC is therefore selected only when waves > 5 and
# the maximum retained row has at least four routes; correctness is independent
# of this launch-policy threshold.
_CAUSAL_CLC_WAVE_THRESHOLD = 5
_CAUSAL_CLC_MIN_MAX_ROW_ROUTES = 4
# Raw-BSR Q8 spends enough of each task on live atom-route parsing that CLC
# work discovery does not amortize. Keep this as caller policy: dense and
# future route-specialized Q8 profiles can still select persistent scheduling.
_RAW_BSR_CLC_MIN_TILE_SIZE_Q = 16


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
    use_keeps_mma_ab = geometry.q_tile_size >= 64
    config_args: dict[str, object] = {
        "use_keeps_mma_ab": use_keeps_mma_ab,
        "tile_size_q": geometry.q_tile_size,
        "tile_size_kv": geometry.kv_tile_size,
        "groups_tokens_heads_q": True,
        "use_block_sparse": True,
        "q_block_size": q_block_size,
        "kv_block_size": kv_block_size,
        "use_kv_valid_bits": use_kv_valid_bits,
        "num_kv_valid_words": (_ceil_div(seq_len_kv, 32) if use_kv_valid_bits else 0),
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
    if static_profile.config.use_keeps_mma_ab:
        persistent_probe = deepcopy(static_profile.config)
        persistent_probe.use_persistent_scheduler = True
        if not persistent_probe.supports_grouped_keeps:
            return None
        return _make_raw_block_sparse_launch_profile(
            traits,
            use_persistent_scheduler=True,
        )

    # Grouped SWAPAB has no Keeps-style capability predicate. Reuse the full
    # profile validator and retain the valid static launch on rejection.
    try:
        return _make_raw_block_sparse_launch_profile(
            traits,
            use_persistent_scheduler=True,
        )
    except ValueError:
        return None


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
    max_execution_tiles: int,
) -> _BlockSparseLaunchSpec:
    """Select static or CLC scheduling without preparing route payloads.

    ``max_execution_tiles`` is the per-row KV128 scheduler capacity. Static
    plans use the largest physical-tail-trimmed route count; dynamic plans use
    the untrimmed capacity implied by the fixed indptr row length.
    """

    from .kernels.fmha_decode.fmha_decode_config import _select_auto_launch_mode

    geometry = _resolve_execution_geometry(
        q_block_size,
        kv_block_size,
        q_tile_size=q_tile_size,
    )
    scheduler_kv_capacity_tokens = max_execution_tiles * geometry.kv_tile_size
    with torch.cuda.device(device_index):
        mode = _select_auto_launch_mode(
            batch_size=batch_size,
            num_heads_kv=num_heads,
            seq_len_kv=scheduler_kv_capacity_tokens,
            num_q_tiles=_ceil_div(seq_len_q, geometry.q_tile_size),
            tile_size_q=geometry.q_tile_size,
            tile_size_kv=geometry.kv_tile_size,
            persistent_min_waves=(
                _CAUSAL_CLC_WAVE_THRESHOLD if mask_type == "causal" else 1
            ),
            persistent_min_tiles_per_cta=(
                _CAUSAL_CLC_MIN_MAX_ROW_ROUTES if mask_type == "causal" else 1
            ),
            persistent_min_tile_size_q=_RAW_BSR_CLC_MIN_TILE_SIZE_Q,
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
    )
    profile = None
    if mode == "persistent":
        profile = _resolve_cached_raw_block_sparse_persistent_launch_profile(traits)
    if profile is None:
        profile = _resolve_cached_raw_block_sparse_static_launch_profile(traits)
    policy: tuple[tuple[str, object], ...] = (
        ("tile_size_q", geometry.q_tile_size),
        ("use_persistent_scheduler", profile.use_persistent_scheduler),
        ("max_execution_tiles", max_execution_tiles),
        # Preserve the public diagnostic key for existing benchmark schemas.
        # Dynamic plans report scheduler capacity rather than current visibility.
        ("visible_kv_tokens", scheduler_kv_capacity_tokens),
        ("execution_path", "raw_bsr_decode"),
        ("use_kv_valid_bits", use_kv_valid_bits),
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
    metadata must outlive captured graphs.

    Metadata is immutable by default. A plan created with
    ``dynamic_metadata=True`` instead permits the caller to update retained
    block-index and token-mask values between ordered launches. Row offsets,
    tensor identities, shapes, and dtypes remain fixed. The initial and updated
    rows must remain strictly increasing, unique, and in range. This is the
    intended lifecycle for CUDA Graph replay: capture only :meth:`run`, update
    retained tensors in place, then replay. Callers must synchronize
    cross-stream updates and must not modify metadata while a consuming launch
    is in flight.
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
        dynamic_metadata: bool = False,
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
        currently require a fine SWAPAB Q tile. The kernel consumes canonical
        BSR directly and assembles selected KV blocks into KV128 execution
        routes. This remains true when every KV block is selected; callers that
        know a pattern is dense should choose the dense FMHA API explicitly.

        Planning validates canonical BSR offsets and strictly increasing,
        unique, in-range block values with one GPU inspection. Invalid metadata
        fails before a new revision is published; plan does not allocate a
        route payload. Inspection performs one packed D2H and synchronizes the
        plan stream, so ``plan()`` is host-synchronizing and must run outside
        CUDA Graph capture.

        ``dynamic_metadata=True`` keeps a supplied ``kv_valid_bits`` tensor in
        the runtime plan even when the inspected routes currently reach only
        valid tokens. This conservative specialization is required when later
        in-place route updates may reach token-mask holes. The row offsets stay
        fixed. Subsequent block-index rows must remain strictly increasing,
        unique, in range, and within the fixed tensor extent; they are not
        reinspected by :meth:`run`.

        A dynamic plan derives its execution-route capacity from the fixed
        row offsets (the untrimmed per-row bound), not from the inspected
        data, so any legal in-place index update fits the compiled schedule.
        Token-mask *values* may also change between launches. Dynamic plans
        therefore use the masked profile; Q64/Q128 derive their route-full
        fast path from the current token words on every run rather than
        specializing on plan-time morphology.

        Concurrent plans are serialized; run keeps using the published state.
        """

        batch_size = _validate_positive_int(batch_size, "batch_size")
        seq_len_q = _validate_positive_int(seq_len_q, "seq_len_q")
        seq_len_kv = _validate_positive_int(seq_len_kv, "seq_len_kv")
        num_qo_heads = _validate_positive_int(num_qo_heads, "num_qo_heads")
        num_kv_heads = _validate_positive_int(num_kv_heads, "num_kv_heads")
        head_dim = _validate_positive_int(head_dim, "head_dim")
        if not isinstance(dynamic_metadata, bool):
            raise TypeError("dynamic_metadata must be a bool")
        geometry = _resolve_execution_geometry(q_block_size, kv_block_size)
        use_keeps_mma_ab = geometry.q_tile_size >= 64
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
        # Dynamic plans must compile the masked path regardless of current
        # values, so scanning a mutable bitset here cannot affect policy.
        inspection_kv_valid_bits = None if dynamic_metadata else kv_valid_bits
        inspection = _inspect_block_sparse_bsr(
            block_indptr,
            block_indices,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_valid_bits=inspection_kv_valid_bits,
            stream=plan_stream,
        )
        max_row_nnz = inspection.max_row_nnz

        # Static plans may compile away an all-valid token mask. Dynamic plans
        # keep masking enabled because the caller can update the same bitset
        # between runs. Full KV128 routes are detected by the kernel from the
        # current words, never from plan-time morphology.
        use_token_mask = kv_valid_bits is not None and (
            dynamic_metadata or inspection.token_mask_has_holes
        )
        # Dynamic plans must also size execution routes for any legal
        # in-place index update, not for the current data: row offsets are
        # fixed, so the untrimmed bound ceil(max_row_nnz * kv_block_size /
        # route_size) holds for every replay, while the inspected value may
        # undercount when a later update avoids the physical tail.
        max_execution_tiles = (
            -(-(max_row_nnz * kv_block_size) // geometry.kv_tile_size)
            if dynamic_metadata
            else inspection.max_retained_routes
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
                max_execution_tiles,
            )
            config = spec.config
            policy = (
                *spec.policy,
                ("dynamic_metadata", dynamic_metadata),
                ("max_row_nnz", max_row_nnz),
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
