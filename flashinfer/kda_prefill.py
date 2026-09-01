"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Kimi Delta Attention Prefill - Backend Layer
============================================

This file provides workspace management, validation, and frozen-kernel launch
support for recurrent KDA prefill.  The stable public dispatcher remains in
``flashinfer.kda``.
"""

import functools
import heapq
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
)

import torch

from .utils import get_compute_capability

if TYPE_CHECKING:
    from .jit.cake_kda import CakeKDATarget, CakeKDAVariant
    from .jit.flash_kda import (
        FlashKDATarget,
        FlashKDAVariant,
        GeneratedFlashKDAModule,
    )

_FLASH_KDA_HEAD_DIM = 128
_FLASH_KDA_DEFAULT_SCALE = _FLASH_KDA_HEAD_DIM**-0.5
_FLASH_KDA_BETA_TMA_HEADS_PER_BOX = 8
_FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES = 6 * 128
_FLASH_KDA_SEVEN_DESCRIPTOR_STORAGE_BYTES = 7 * 128
_FLASH_KDA_PERSISTENT_MIN_BALANCED_CTAS = 128
_FLASH_KDA_LPT_MAX_IMBALANCE_NUMERATOR = 21
_FLASH_KDA_LPT_MAX_IMBALANCE_DENOMINATOR = 20
_FLASH_KDA_GB200_LPT_MAX_IMBALANCE_NUMERATOR = 263
_FLASH_KDA_GB200_LPT_MAX_IMBALANCE_DENOMINATOR = 250
_FLASH_KDA_SMALL_BH_GROUP_SIZE = 8
_FLASH_KDA_SMALL_BH_RING_STAGES = 35
_FLASH_KDA_SMALL_BH_PACKET_ROWS = 123
_FLASH_KDA_SMALL_BH_PACKET_ELEMENTS = 128
_FLASH_KDA_SMALL_BH_MAX_TASKS = 8
_FLASH_KDA_SMALL_BH_MIN_SEQUENCE_LENGTH = 2048
_FLASH_KDA_BT16_CHUNK = 16
_FLASH_KDA_BT16_VALUE_SPLITS = 2
_FLASH_KDA_BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA = 6
_FLASH_KDA_BT16_GENERAL_HIGH_WORK_CHUNKS_PER_PREP_CTA = 8
_FLASH_KDA_BT16_GENERAL_HIGH_WORK_MIN_CHUNK_HEADS = 16_384
_FLASH_KDA_BT16_H12_CHUNKS_PER_PREP_CTA = 4
_FLASH_KDA_BT16_H12_CPC1_MAX_TOTAL_CHUNKS = 128
_FLASH_KDA_BT16_PREP_WAVE_QUANT_MIN_WAVES = 8
_FLASH_KDA_BT16_PREP_WAVE_QUANT_MIN_RETAINED_PERCENT = 98
_FLASH_KDA_BT16_DENSE_PREP_WAVES = 5
_FLASH_KDA_BT16_DENSE_MIN_HEADS = 60
_FLASH_KDA_BT16_DENSE_MAX_HEADS = 64
_FLASH_KDA_BT16_DENSE_MIN_SEQUENCE_LENGTH = 4096
_FLASH_KDA_BT16_N16_ONE_CHAIN_WAVE_MIN_SEQUENCE_LENGTH = 512
_FLASH_KDA_BT16_N16_TWO_CHAIN_WAVE_MIN_SEQUENCE_LENGTH = 3072
_FLASH_KDA_BT16_N16_MULTI_WAVE_MIN_SEQUENCE_LENGTH = 512
_FLASH_KDA_BT16_N16_MAX_DIRECT_WAVES = 3
_FLASH_KDA_BT16_MID_MIN_SEQUENCE_LENGTH = 4096
_FLASH_KDA_BT16_LONG_MIN_SEQUENCE_LENGTH = 65_536
_FLASH_KDA_BT16_MID_MAX_TASKS = 32
_FLASH_KDA_H12_DIRECT_N32_MIN_SEQUENCE_LENGTH = 64
_FLASH_KDA_H12_DIRECT_N32_MAX_SEQUENCE_LENGTH = 256
_FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH = 128
_FLASH_KDA_N32_REGISTER_INVERSE_MIN_SEQUENCE_LENGTH = 256
_FLASH_KDA_N32_PREDICTION_FIRST_MIN_HEADS = 24
_FLASH_KDA_N32_PREDICTION_FIRST_MIXED_MIN_SEQUENCE_LENGTH = 512
_FLASH_KDA_INDEPENDENT_DVSPLIT_CTAS = 2
_FLASH_KDA_INDEPENDENT_DVSPLIT_MIN_SEQUENCE_LENGTH = 512
_FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS = 128
_FLASH_KDA_ROUTE_DIRECT_M128 = "direct_m128"
_FLASH_KDA_ROUTE_DIRECT_M128_N16 = "direct_m128_n16"
_FLASH_KDA_ROUTE_HEAD_GROUPED_M128 = "head_grouped_persistent_m128"
_FLASH_KDA_ROUTE_LPT_M128 = "lpt_persistent_m128"
_FLASH_KDA_ROUTE_SCALAR_CHUNK_LPT_M128 = "scalar_chunk_lpt_m128"
_FLASH_KDA_ROUTE_SOURCE_VTILE_M128 = "source599_vtile_m128"
_FLASH_KDA_ROUTE_PERSISTENT_M128 = "persistent_m128"
_FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128 = "piece_persistent_m128"
_FLASH_KDA_ROUTE_M64 = "independent_dvsplit_m64"
_FLASH_KDA_ROUTE_SMALL_BH_M128 = "small_bh_owner_helper_m128"
_FLASH_KDA_ROUTE_BT16_M64 = "bt16_prepare_chain_m64"
_FLASH_KDA_ROUTE_AFFINE_M128 = "affine_split_m128"
_CAKE_KDA_ROUTE_AFFINE_M128 = "cake_affine_split_m128"

_FLASH_KDA_GENERATED_ROUTE_ABI_FAMILY = {
    (_FLASH_KDA_ROUTE_DIRECT_M128, "main"): "direct_m128",
    (_FLASH_KDA_ROUTE_DIRECT_M128_N16, "main"): "direct_m128",
    (_FLASH_KDA_ROUTE_SOURCE_VTILE_M128, "main"): "vtile_m128",
    (_FLASH_KDA_ROUTE_BT16_M64, "bt16_prepare"): "bt16_prepare",
    (_FLASH_KDA_ROUTE_BT16_M64, "main"): "bt16_chain",
    (_FLASH_KDA_ROUTE_M64, "main"): "m64",
    (_FLASH_KDA_ROUTE_SCALAR_CHUNK_LPT_M128, "main"): "scalar_lpt_m128",
    (_FLASH_KDA_ROUTE_HEAD_GROUPED_M128, "main"): "taskized_persistent_m128",
    (_FLASH_KDA_ROUTE_LPT_M128, "main"): "taskized_persistent_m128",
    (_FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128, "main"): ("taskized_persistent_m128"),
    (_FLASH_KDA_ROUTE_SMALL_BH_M128, "main"): "small_bh_m128",
    (_FLASH_KDA_ROUTE_AFFINE_M128, "affine_main"): "direct_m128",
    (_FLASH_KDA_ROUTE_AFFINE_M128, "affine_map"): "direct_m128",
    (_FLASH_KDA_ROUTE_AFFINE_M128, "affine_scan"): "affine_scan",
    (_FLASH_KDA_ROUTE_AFFINE_M128, "affine_correction"): "direct_m128",
}
_FLASH_KDA_GENERATED_SPECIALIZATION_FIELDS = {
    "direct_m128": (
        "chunk",
        "serving_native_abi",
        "gate_kind",
        "checkpoint_tma",
        "pair_packed_beta",
        "scalar_beta",
        "early_n32_state_pack",
        "generic_register_inverse",
        "n32_prediction_first",
        "tensor_state_decay",
        "state_dtype_is_fp32",
        "n32_ft_slab",
        "pdl_wait_initial_state_f32",
        "pdl_publish_final_state",
        "affine_main_indexed_initial",
        "affine_main_indexed_initial_bf16",
    ),
    "vtile_m128": (
        "full_n32_chunks",
        "num_heads",
        "use_initial_state",
        "store_final_state",
        "scale",
        "lower_bound",
        "persistent_mode",
        "persistent_six_task_schedule",
        "persistent_stride_head_aligned",
        "state_dtype_is_fp32",
    ),
    "bt16_prepare": (),
    "bt16_chain": (
        "bt16_stage_count",
        "state_dtype_is_fp32",
        "serving_native_abi",
    ),
    "m64": (
        "full_n32_chunks",
        "num_heads",
        "use_initial_state",
        "store_final_state",
        "scale",
        "lower_bound",
        "state_dtype_is_fp32",
    ),
    "scalar_lpt_m128": (
        "num_heads",
        "use_initial_state",
        "store_final_state",
        "scale",
        "lower_bound",
        "persistent_schedule",
        "state_dtype_is_fp32",
    ),
    "taskized_persistent_m128": (
        "piece_tasks",
        "state_dtype_is_fp32",
    ),
    "small_bh_m128": (
        "serving_native_abi",
        "state_dtype_is_fp32",
    ),
    "affine_scan": ("use_pdl",),
}

# Physical contract for the frozen persistent-M128 schedule. The generated
# launch reserves an additional aligned control prefix; the roofline uses the
# schedule's data-pool footprint when resolving resident CTA count.
_FLASH_KDA_M128_CHUNK = 32
_FLASH_KDA_PERSISTENT_THREADS_PER_CTA = 1024
_FLASH_KDA_PERSISTENT_SMEM_POOL_BYTES_PER_CTA = 220_672
_FLASH_KDA_PERSISTENT_TMEM_COLS_PER_CTA = 256
_FLASH_KDA_BLACKWELL_MAX_THREADS_PER_SM = 2048
_FLASH_KDA_BLACKWELL_SMEM_BYTES_PER_SM = 228 * 1024
_FLASH_KDA_BLACKWELL_TMEM_COLS_PER_SM = 512
_FLASH_KDA_BLACKWELL_BF16_TFLOPS = 2250.0
_FLASH_KDA_BLACKWELL_HBM_GBPS = 8000.0
_FLASH_KDA_PERSISTENT_TENSOR_FLOPS_PER_CHUNK = (
    3 * 2 * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_M128_CHUNK * _FLASH_KDA_HEAD_DIM
    + 2 * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_M128_CHUNK * _FLASH_KDA_M128_CHUNK
)
_FLASH_KDA_PERSISTENT_STREAM_BYTES_PER_CHUNK = (
    5 * _FLASH_KDA_M128_CHUNK * _FLASH_KDA_HEAD_DIM * 2 + _FLASH_KDA_M128_CHUNK * 2
)
_FLASH_KDA_PERSISTENT_STATE_BYTES = _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM * 2
_FLASH_KDA_PERSISTENT_TASK_REFILL_CHUNKS = 2
_flash_kda_tensor_cache: dict[tuple, torch.Tensor] = {}
_flash_kda_tensor_cache_lock = threading.Lock()

_PackedMetadataSignature = tuple[int, int, int, int, bool]
_PersistentTaskPlan = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
_PackedTaskMetadata = tuple[
    tuple[int, ...],
    Optional[_PersistentTaskPlan],
    bool,
    tuple[int, ...],
    tuple[int, ...],
]


@dataclass(frozen=True)
class _PersistentM128Roofline:
    """Resolved occupancy and critical-path lower bounds in nanoseconds."""

    resident_ctas_per_sm: int
    worker_count: int
    handoff_count: int
    chunk_ns: float
    state_transfer_ns: float
    task_refill_ns: float
    direct_ns: float
    piece_ns: float


@dataclass(frozen=True)
class _GeneratedAffineCarriers:
    """Workspace-cached typed carriers shared by the affine module roles."""

    dummy_bf16: torch.Tensor
    dummy_i32: torch.Tensor
    dummy_i64: torch.Tensor
    dummy_f32: torch.Tensor
    dummy_u32: torch.Tensor
    empty_bf16: torch.Tensor
    empty_f32: torch.Tensor
    empty_i64: torch.Tensor
    empty_u8: torch.Tensor


@dataclass(frozen=True)
class _GeneratedAffineModule:
    """One exact selector-to-module resolution for an affine launch role."""

    role: str
    selector_key: dict[str, object]
    metadata: "GeneratedFlashKDAModule"
    module: "_GeneratedAffineRuntimeModule"


class _GeneratedAffineRuntimeModule(Protocol):
    def run(self, *args: object) -> object: ...


@dataclass(frozen=True)
class _GeneratedAffineModuleBundle:
    """Workspace-local immutable resolutions for one affine launch plan."""

    main: _GeneratedAffineModule
    map: _GeneratedAffineModule
    scan: _GeneratedAffineModule
    correction: _GeneratedAffineModule


_AffineBetaTMALayout = Literal["pair_packed", "direct", "padded"]


@dataclass(frozen=True)
class _GeneratedAffineLaunchPlanKey:
    """Tensor-free identity of one workspace-local affine launch plan."""

    target: str
    token_offsets: tuple[int, ...]
    num_heads: int
    state_dtype: torch.dtype
    beta_layouts: tuple[
        _AffineBetaTMALayout,
        _AffineBetaTMALayout,
        _AffineBetaTMALayout,
    ]


@dataclass(frozen=True)
class _CakeKDAAffinePlan:
    """Exact target and token partition for the sealed affine composite."""

    target: Literal["sm100a", "sm103a"]
    token_offsets: tuple[int, ...]

    @property
    def num_parts(self) -> int:
        return len(self.token_offsets) - 1


class _CakeKDAAffineModule(Protocol):
    def run(self, *args: object) -> object: ...


@dataclass(frozen=True)
class _CakeKDAAffineModuleBundle:
    main: _CakeKDAAffineModule
    map: _CakeKDAAffineModule
    scan: _CakeKDAAffineModule
    correction: _CakeKDAAffineModule


@dataclass(frozen=True)
class _CakeKDAAffineLaunchPlanKey:
    target: Literal["sm100a", "sm103a"]
    token_offsets: tuple[int, ...]
    num_heads: int


@dataclass(frozen=True)
class _CakeKDAAffineLaunchPlan:
    key: _CakeKDAAffineLaunchPlanKey
    num_parts: int
    tail_start: int
    main_lengths: tuple[int, ...]
    tail_lengths: tuple[int, ...]
    main_final: torch.Tensor
    map_identity: torch.Tensor
    map_state: torch.Tensor
    carry: torch.Tensor
    correction_final: torch.Tensor
    final_compact: torch.Tensor
    final_external: torch.Tensor
    zero_v: torch.Tensor
    map_out: torch.Tensor
    correction_out: torch.Tensor
    state_indices_i64: torch.Tensor
    split_cu_seqlens: torch.Tensor
    tail_cu_seqlens: torch.Tensor
    main_seq_order: torch.Tensor
    tail_seq_order: torch.Tensor
    main_descriptor_storage: torch.Tensor
    map_descriptor_storage: torch.Tensor
    correction_descriptor_storage: torch.Tensor
    empty_bf16: torch.Tensor
    empty_f32: torch.Tensor
    empty_i32: torch.Tensor
    modules: _CakeKDAAffineModuleBundle


@dataclass(frozen=True)
class _GeneratedAffineLaunchPlan:
    """Static scalars and workspace-owned views for one affine route shape."""

    key: _GeneratedAffineLaunchPlanKey
    num_parts: int
    tail_start: int
    total_tokens: int
    tail_tokens: int
    main_lengths: tuple[int, ...]
    tail_lengths: tuple[int, ...]
    main_final: torch.Tensor
    map_identity: torch.Tensor
    map_state: torch.Tensor
    carry: torch.Tensor
    correction_final: torch.Tensor
    final_compact: torch.Tensor
    zero_v: torch.Tensor
    map_out: torch.Tensor
    correction_out: torch.Tensor
    state_indices_i64: torch.Tensor
    final_external: Optional[torch.Tensor]
    main_beta_padded: Optional[torch.Tensor]
    map_beta_padded: Optional[torch.Tensor]
    correction_beta_padded: Optional[torch.Tensor]
    split_cu_seqlens: torch.Tensor
    tail_cu_seqlens: torch.Tensor
    main_seq_order: torch.Tensor
    tail_seq_order: torch.Tensor
    main_descriptor_storage: torch.Tensor
    map_descriptor_storage: torch.Tensor
    correction_descriptor_storage: torch.Tensor
    modules: _GeneratedAffineModuleBundle


_GeneratedAffineLaunchObserver = Callable[
    [str, dict[str, object], object, object], None
]
_generated_affine_launch_observer: ContextVar[
    Optional[_GeneratedAffineLaunchObserver]
] = ContextVar("generated_affine_launch_observer", default=None)


@contextmanager
def _observe_generated_affine_launches(
    observer: _GeneratedAffineLaunchObserver,
) -> Iterator[None]:
    """Observe exact affine physical launches within the current context."""

    token = _generated_affine_launch_observer.set(observer)
    try:
        yield
    finally:
        _generated_affine_launch_observer.reset(token)


def _generated_affine_module_for_launch(
    resolved: _GeneratedAffineModule,
    observer: Optional[_GeneratedAffineLaunchObserver],
) -> _GeneratedAffineRuntimeModule:
    """Record one exact physical launch immediately before returning its module."""

    if observer is not None:
        observer(
            resolved.role,
            resolved.selector_key,
            resolved.metadata,
            resolved.module,
        )
    return resolved.module


class _RecurrentKDAPrefillWorkspaceBase:
    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("RecurrentKDAPrefillWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._state_scratch: Optional[torch.Tensor] = None
        self._beta_padding: Optional[torch.Tensor] = None
        self._small_bh_packet_workspace: Optional[torch.Tensor] = None
        self._small_bh_packet_ready: Optional[torch.Tensor] = None
        self._small_bh_packet_consumed: Optional[torch.Tensor] = None
        self._small_bh_helper_done: Optional[torch.Tensor] = None
        self._piece_mid_state: Optional[torch.Tensor] = None
        self._piece_mid_state_ready: Optional[torch.Tensor] = None
        self._bt16_cu_chunks: Optional[torch.Tensor] = None
        self._bt16_chunk_to_seq: Optional[torch.Tensor] = None
        self._bt16_qd: Optional[torch.Tensor] = None
        self._bt16_kd: Optional[torch.Tensor] = None
        self._bt16_w: Optional[torch.Tensor] = None
        self._bt16_qk: Optional[torch.Tensor] = None
        self._bt16_diag: Optional[torch.Tensor] = None
        self._bt16_metadata_signature: Optional[tuple] = None
        self._cake_kda_affine_buffers: dict[str, torch.Tensor] = {}
        self._cake_kda_affine_map_identity_data_ptr: Optional[int] = None
        self._cake_kda_affine_launch_plan: Optional[_CakeKDAAffineLaunchPlan] = None
        self._cute_dsl_workspace: Optional[torch.Tensor] = None
        self._descriptor_storages = {
            variant: torch.empty(
                (
                    _FLASH_KDA_SEVEN_DESCRIPTOR_STORAGE_BYTES
                    if variant in ("m128_n16_checkpoint", "small_bh_m128")
                    or variant.startswith("bt16_")
                    else _FLASH_KDA_DESCRIPTOR_STORAGE_BYTES
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for variant in (
                "m64",
                "m128",
                "m128_tensor_state_decay",
                "m128_h12_short",
                "m128_h12_long",
                "m128_n16",
                "m128_n16_checkpoint",
                "m128_n16_short",
                "persistent_m128",
                "piece_persistent_m128",
                "small_bh_m128",
                "bt16_prepare",
                "bt16_prepare_beta_tma",
                "bt16_chain_m64_s7",
                "bt16_chain_m64_s8",
                "bt16_chain_m64_s9",
                "m128_unbounded_softplus",
                "m128_bt64_unbounded_softplus",
            )
        }
        # Receipt-selected modules own independent descriptor lifetimes.  The
        # dictionary is populated lazily because the physical selector (and
        # therefore the module id) is resolved from the runtime shape and
        # exact architecture, not from a coarse route name.
        self._generated_descriptor_storages: dict[str, torch.Tensor] = {}
        self._descriptor_signatures: dict[str, tuple] = {}
        self._generated_scalar_schedules: dict[
            tuple, tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        self._affine_buffers: dict[str, torch.Tensor] = {}
        self._affine_map_identity_data_ptr: Optional[int] = None
        self._generated_affine_carriers: Optional[_GeneratedAffineCarriers] = None
        self._generated_affine_module_bundle: Optional[_GeneratedAffineModuleBundle] = (
            None
        )
        self._generated_affine_launch_plan: Optional[_GeneratedAffineLaunchPlan] = None
        self._packed_metadata_lock = threading.Lock()
        self._packed_metadata_tensor: Optional[torch.Tensor] = None
        self._packed_metadata_signature: Optional[_PackedMetadataSignature] = None
        self._packed_metadata: Optional[_PackedTaskMetadata] = None
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False
        # The SM120 backend's per-workspace state, created on first use by
        # ``_sm120_prefill_resources``. A Cake-only caller carries this one
        # ``None`` field and never imports CuTe DSL; composing rather than
        # subclassing keeps ``RecurrentKDAPrefillWorkspace`` the single public
        # workspace type while letting each backend own the buffers only it
        # understands.
        self._sm120_state: Optional[object] = None
        #: Guards the creation of ``_sm120_state``.  Two threads reaching a
        #: workspace's first SM120 call would otherwise both read ``None`` and
        #: both construct: one assignment wins and the loser runs against an
        #: orphan with its own lock, its own scratch and its own capture flag,
        #: so nothing serializes the launch sequence, device memory doubles,
        #: and a capture is recorded where no one will look for it.
        self._sm120_state_lock = threading.Lock()


class RecurrentKDAPrefillWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Caller-owned storage required for recurrent-KDA CUDA graph capture.

    Construct one workspace per captured
    :func:`flashinfer.kda.recurrent_kda` invocation on the graph's CUDA
    device. Warm it by invoking that function eagerly with the exact tensors
    and capture stream, then synchronize that stream before capture. The
    workspace owns optional final-state scratch, backend metadata, TMA
    descriptors, and schedule-specific scratch for the lifetime of the graph.
    On SM100-family devices this includes beta padding, M64/M128-N32/M128-N16
    and BT64 descriptor storage, small-BH packet-ring storage, and BT16
    prepare/chain metadata, factors, and independent descriptor storage.
    Persistent M128 is an eager-only B200/GB200 route; explicit workspaces use
    non-persistent direct, M64, small-BH, or eligible BT16 schedules so graph
    capture never synchronizes sequence lengths to construct host task bins.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed to Python again, either eagerly or in another capture.
    Graph replay does not invoke Python and remains valid for the lifetime of
    the workspace.
    """


class _FlashKDAStreamWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Internal eager-only workspace for one CUDA stream."""


_flash_kda_stream_workspaces: dict[tuple[int, int], _FlashKDAStreamWorkspace] = {}
_flash_kda_stream_workspaces_lock = threading.Lock()


def _is_plain_multi_token_prefill(
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
) -> bool:
    if num_spec_tokens is not None or not isinstance(q, torch.Tensor) or q.ndim != 4:
        return False
    if cu_seqlens is None:
        return q.shape[1] > 1
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
        return False
    num_sequences = cu_seqlens.numel() - 1
    return num_sequences > 0 and q.shape[1] > num_sequences


#: TMA's base-address requirement, mirrored here so backend selection and
#: backend validation cannot drift apart.  The backend owns the real constant
#: (``runtime.GLOBAL_BASE_ALIGN``); a test asserts the two agree, because a
#: silent divergence turns a fallback into an exception.
_SM120_TMA_BASE_ALIGN = 16


def _is_contiguous_cuda_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
    )


def _is_token_row_strided_cuda_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.ndim >= 2
        and tensor.stride(-1) == 1
        and tensor.stride(-2) >= tensor.shape[-1]
    )


def _is_state_pool_tensor(
    tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
    num_heads: int,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype in (torch.bfloat16, torch.float32)
        and tensor.ndim == 4
        and tensor.shape[0] > 0
        and tensor.data_ptr() % 16 == 0
        and tuple(tensor.shape[1:])
        == (num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
        and tensor.stride(-1) == 1
        and tensor.stride(-2) == _FLASH_KDA_HEAD_DIM
        and tensor.stride(-3) == _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
        and tensor.stride(0) >= num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
        and tensor.stride(0) * tensor.element_size() % 16 == 0
    )


def _flash_kda_prefill_is_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Return whether the call exactly matches the frozen FlashKDA contract."""

    if not _is_plain_multi_token_prefill(q, cu_seqlens, num_spec_tokens):
        return False
    if (
        num_accepted_tokens is not None
        or initial_state_source is not None
        or initial_state_indices is not None
    ):
        return False
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and (
            lower_bound is None
            or (math.isfinite(float(lower_bound)) and float(lower_bound) < 0.0)
        )
    ):
        return False
    if (
        not q.is_cuda
        or get_compute_capability(q.device)
        not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
    ):
        return False
    if not _is_contiguous_cuda_tensor(q, dtype=torch.bfloat16, device=q.device):
        return False
    if q.ndim != 4:
        return False
    batch_size, total_or_fixed_tokens, num_heads, head_dim = q.shape
    if (
        batch_size <= 0
        or total_or_fixed_tokens <= 1
        or num_heads <= 0
        or head_dim != _FLASH_KDA_HEAD_DIM
    ):
        return False
    for tensor in (k, v, g):
        if (
            not _is_contiguous_cuda_tensor(
                tensor, dtype=torch.bfloat16, device=q.device
            )
            or tensor.shape != q.shape
        ):
            return False
    if not _is_token_row_strided_cuda_tensor(
        beta, dtype=torch.bfloat16, device=q.device
    ) or beta.shape != (batch_size, total_or_fixed_tokens, num_heads):
        return False
    if batch_size > 1 and beta.stride(0) != total_or_fixed_tokens * beta.stride(1):
        return False
    if not _is_contiguous_cuda_tensor(
        A_log, dtype=torch.float32, device=q.device
    ) or A_log.shape != (num_heads,):
        return False
    if not _is_contiguous_cuda_tensor(dt_bias, dtype=torch.float32, device=q.device):
        return False
    if dt_bias.numel() != num_heads * _FLASH_KDA_HEAD_DIM or dt_bias.ndim not in (1, 2):
        return False
    if dt_bias.ndim == 2 and dt_bias.shape != (num_heads, _FLASH_KDA_HEAD_DIM):
        return False

    if cu_seqlens is None:
        num_sequences = batch_size
    else:
        if (
            batch_size != 1
            or not cu_seqlens.is_cuda
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or total_or_fixed_tokens <= num_sequences:
            return False

    if ssm_state_indices is not None:
        if (
            initial_state is None
            or not _is_contiguous_cuda_tensor(
                ssm_state_indices, dtype=torch.int32, device=q.device
            )
            or ssm_state_indices.ndim != 1
            or ssm_state_indices.numel() != num_sequences
        ):
            return False
    if initial_state is not None:
        if not _is_state_pool_tensor(
            initial_state,
            device=q.device,
            num_heads=num_heads,
        ):
            return False
        # The exported FP32-state portfolio is the indexed state-pool API.
        # Compact FP32 state has no receipt-backed product route, so accepting
        # it here would let dispatch invent an unobserved physical contract.
        if initial_state.dtype == torch.float32 and ssm_state_indices is None:
            return False
        if ssm_state_indices is None and initial_state.shape[0] != num_sequences:
            return False
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens > torch.iinfo(torch.int32).max
        or checkpoint_every_n_tokens % 16 != 0
    ):
        return False
    if checkpoint_every_n_tokens:
        if initial_state is not None and initial_state.dtype == torch.float32:
            return False
        if (
            not _is_contiguous_cuda_tensor(
                state_checkpoints, dtype=torch.bfloat16, device=q.device
            )
            or state_checkpoints.ndim != 4
            or tuple(state_checkpoints.shape[1:])
            != (num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
            or not _is_contiguous_cuda_tensor(
                checkpoint_cu_starts, dtype=torch.int64, device=q.device
            )
            or checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.numel() != num_sequences + 1
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    if output is not None:
        if (
            not _is_contiguous_cuda_tensor(
                output, dtype=torch.bfloat16, device=q.device
            )
            or output.shape != q.shape
        ):
            return False
    return True


def _select_flash_kda_prefill_variant(
    *,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    needs_direct_m128: bool = False,
    use_persistent_m128: bool = False,
    use_small_bh_m128: bool = False,
    use_exact_n16: bool = False,
    unbounded_softplus: bool = False,
    use_bt64_unbounded_softplus: bool = False,
) -> "FlashKDAVariant | CakeKDAVariant":
    if unbounded_softplus:
        if use_bt64_unbounded_softplus:
            return "m128_bt64_unbounded_softplus"
        return "m128_unbounded_softplus"
    if num_heads == 12 or use_exact_n16:
        return "m128_n16"
    if (
        not needs_direct_m128
        and fixed_layout
        and num_sequences == 1
        and num_heads == 64
    ):
        return "m64"
    if use_small_bh_m128:
        return "small_bh_m128"
    if use_persistent_m128:
        return "persistent_m128"
    return "m128"


@functools.cache
def _flash_kda_device_sm_count(device: torch.device) -> int:
    """Resolve and cache the physical SM count for one CUDA device."""

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _cake_kda_affine_export_is_available() -> bool:
    """Query the sealed affine bundle without importing JIT eagerly."""

    from .jit.cake_kda import cake_kda_affine_is_available

    return cake_kda_affine_is_available()


@functools.cache
def _select_cake_kda_affine_plan(
    *,
    export_available: bool,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    batch_size: int,
    total_tokens: int,
    num_heads: int,
    head_dim: int,
    qkv_shapes_equal: bool,
    qkv_dtype: torch.dtype,
    beta_contiguous: bool,
    beta_dtype: torch.dtype,
    indexed_state: bool,
    initial_state_dtype: Optional[torch.dtype],
    has_checkpoints: bool,
    lower_bound: Optional[float],
) -> Optional[_CakeKDAAffinePlan]:
    """Select only the exact contract covered by the sealed affine export."""

    if (
        not export_available
        or compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        or not fixed_layout
        or batch_size != 1
        or total_tokens < 8192
        or total_tokens % _FLASH_KDA_M128_CHUNK
        or num_heads <= 0
        or num_heads > 32
        or head_dim != _FLASH_KDA_HEAD_DIM
        or not qkv_shapes_equal
        or qkv_dtype != torch.bfloat16
        or not beta_contiguous
        or beta_dtype != torch.bfloat16
        or not indexed_state
        or initial_state_dtype != torch.bfloat16
        or has_checkpoints
        or lower_bound is not None
        or sm_count <= 0
    ):
        return None

    chunks = total_tokens // _FLASH_KDA_M128_CHUNK
    candidate_parts = min(
        sm_count,
        max(2, sm_count // num_heads),
        max(2, chunks // 32),
    )
    if candidate_parts < 2:
        return None
    if candidate_parts < 8 and chunks < 2048:
        return None
    chunks_per_part = (chunks + candidate_parts - 1) // candidate_parts
    chunk_offsets = tuple(
        sorted(
            {min(part * chunks_per_part, chunks) for part in range(candidate_parts + 1)}
        )
    )
    if len(chunk_offsets) < 3:
        return None
    target: Literal["sm100a", "sm103a"] = (
        "sm100a" if compute_capability == (10, 0) else "sm103a"
    )
    return _CakeKDAAffinePlan(
        target=target,
        token_offsets=tuple(
            chunk_offset * _FLASH_KDA_M128_CHUNK for chunk_offset in chunk_offsets
        ),
    )


def _cake_kda_affine_workspace_buffer(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    name: str,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    zero_on_allocate: bool = False,
) -> torch.Tensor:
    """Return a stable, grow-only workspace view for one affine role."""

    if not name or not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(
            "Cake KDA affine workspace names and dimensions must be positive"
        )
    numel = math.prod(shape)
    buffer = workspace._cake_kda_affine_buffers.get(name)
    if (
        buffer is None
        or buffer.numel() < numel
        or buffer.dtype != dtype
        or buffer.device != device
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Cake KDA affine workspace is not warmed for "
                f"{name}; invoke the largest shape before capture"
            )
        factory = torch.zeros if zero_on_allocate else torch.empty
        buffer = factory(numel, dtype=dtype, device=device)
        workspace._cake_kda_affine_buffers[name] = buffer
    return buffer[:numel].view(shape)


@functools.cache
def _get_cake_kda_affine_module_bundle(
    target: Literal["sm100a", "sm103a"],
) -> _CakeKDAAffineModuleBundle:
    """Load the four sealed role modules for one exact Blackwell target."""

    from .jit.cake_kda import get_cake_kda_affine_module

    return _CakeKDAAffineModuleBundle(
        main=get_cake_kda_affine_module(target, "main"),
        map=get_cake_kda_affine_module(target, "map"),
        scan=get_cake_kda_affine_module(target, "scan"),
        correction=get_cake_kda_affine_module(target, "correction"),
    )


def _cake_kda_affine_launch_plan(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    affine_plan: _CakeKDAAffinePlan,
    device: torch.device,
    num_heads: int,
    capturing: bool,
) -> _CakeKDAAffineLaunchPlan:
    """Resolve stable metadata and scratch for the four-stage composite."""

    key = _CakeKDAAffineLaunchPlanKey(
        target=affine_plan.target,
        token_offsets=affine_plan.token_offsets,
        num_heads=num_heads,
    )
    cached = workspace._cake_kda_affine_launch_plan
    if cached is not None and cached.key == key:
        return cached
    if capturing:
        raise RuntimeError(
            "Cake KDA affine launch plan is not warmed for CUDA graph capture"
        )

    token_offsets = affine_plan.token_offsets
    num_parts = affine_plan.num_parts
    tail_start = token_offsets[1]
    total_tokens = token_offsets[-1]
    tail_tokens = total_tokens - tail_start
    main_lengths = tuple(
        right - left
        for left, right in zip(token_offsets[:-1], token_offsets[1:], strict=True)
    )
    tail_lengths = main_lengths[1:]
    tail_offsets = tuple(offset - tail_start for offset in token_offsets[1:])
    state_shape = (
        num_parts,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )
    tail_state_shape = (
        num_parts - 1,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )

    def buffer(
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        zero_on_allocate: bool = False,
    ) -> torch.Tensor:
        return _cake_kda_affine_workspace_buffer(
            workspace=workspace,
            name=name,
            device=device,
            shape=shape,
            dtype=dtype,
            zero_on_allocate=zero_on_allocate,
        )

    main_final = buffer("affine_main_final_f32", state_shape, torch.float32)
    map_identity = buffer(
        "affine_map_identity_bf16",
        tail_state_shape,
        torch.bfloat16,
        zero_on_allocate=True,
    )
    if workspace._cake_kda_affine_map_identity_data_ptr != map_identity.data_ptr():
        map_identity.diagonal(dim1=-2, dim2=-1).fill_(1)
        workspace._cake_kda_affine_map_identity_data_ptr = map_identity.data_ptr()
    map_state = buffer("affine_map_state_bf16", tail_state_shape, torch.bfloat16)
    carry = buffer("affine_carry_f32", tail_state_shape, torch.float32)
    correction_final = buffer(
        "affine_correction_final_f32", tail_state_shape, torch.float32
    )
    final_shape = (1, num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
    final_compact = buffer("affine_final_compact_f32", final_shape, torch.float32)
    final_external = buffer("affine_final_external_bf16", final_shape, torch.bfloat16)
    tail_value_shape = (1, tail_tokens, num_heads, _FLASH_KDA_HEAD_DIM)
    zero_v = buffer(
        "affine_zero_v_bf16",
        tail_value_shape,
        torch.bfloat16,
        zero_on_allocate=True,
    )
    map_out = buffer("affine_map_out_bf16", tail_value_shape, torch.bfloat16)
    correction_out = buffer(
        "affine_correction_out_bf16", tail_value_shape, torch.bfloat16
    )
    state_indices_i64 = buffer("affine_state_indices_i64", (1,), torch.int64)
    empty_bf16 = buffer("affine_empty_bf16", (1,), torch.bfloat16)[:0]
    empty_f32 = buffer("affine_empty_f32", (1,), torch.float32)[:0]
    empty_i32 = buffer("affine_empty_i32", (1,), torch.int32)[:0]

    split_cu_seqlens = _cached_tensor(
        ("cake_affine_split_cu", *_stream_cache_key(device), token_offsets),
        lambda: torch.tensor(token_offsets, dtype=torch.int64, device=device),
        capture_error=(
            "Cake KDA affine split offsets are not warmed for CUDA graph capture"
        ),
    )
    tail_cu_seqlens = _cached_tensor(
        ("cake_affine_tail_cu", *_stream_cache_key(device), tail_offsets),
        lambda: torch.tensor(tail_offsets, dtype=torch.int64, device=device),
        capture_error=(
            "Cake KDA affine tail offsets are not warmed for CUDA graph capture"
        ),
    )
    modules = _get_cake_kda_affine_module_bundle(affine_plan.target)
    resolved = _CakeKDAAffineLaunchPlan(
        key=key,
        num_parts=num_parts,
        tail_start=tail_start,
        main_lengths=main_lengths,
        tail_lengths=tail_lengths,
        main_final=main_final,
        map_identity=map_identity,
        map_state=map_state,
        carry=carry,
        correction_final=correction_final,
        final_compact=final_compact,
        final_external=final_external,
        zero_v=zero_v,
        map_out=map_out,
        correction_out=correction_out,
        state_indices_i64=state_indices_i64,
        split_cu_seqlens=split_cu_seqlens,
        tail_cu_seqlens=tail_cu_seqlens,
        main_seq_order=_identity_seq_order(device=device, num_sequences=num_parts),
        tail_seq_order=_identity_seq_order(device=device, num_sequences=num_parts - 1),
        main_descriptor_storage=buffer(
            "affine_main_descriptors",
            (_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,),
            torch.uint8,
        ),
        map_descriptor_storage=buffer(
            "affine_map_descriptors",
            (_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,),
            torch.uint8,
        ),
        correction_descriptor_storage=buffer(
            "affine_correction_descriptors",
            (_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,),
            torch.uint8,
        ),
        empty_bf16=empty_bf16,
        empty_f32=empty_f32,
        empty_i32=empty_i32,
        modules=modules,
    )
    workspace._cake_kda_affine_launch_plan = resolved
    return resolved


def _run_cake_kda_affine_direct_role(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    module: _CakeKDAAffineModule,
    role: Literal["main", "map", "correction"],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_tma: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_order: torch.Tensor,
    state_indices: torch.Tensor,
    initial_state_bf16: torch.Tensor,
    out: torch.Tensor,
    final_state_bf16: torch.Tensor,
    initial_state_f32: torch.Tensor,
    final_state_f32: torch.Tensor,
    descriptor_storage: torch.Tensor,
    num_heads: int,
    state_slot_stride: int,
    scale: float,
    grid_x: int,
    stream_ptr: int,
    capturing: bool,
) -> None:
    signature = _descriptor_signature(q=q, k=k, v=v, g=g, beta_tma=beta_tma, out=out)
    signature_key = f"cake_affine:{role}"
    warmed_signature = workspace._descriptor_signatures.get(signature_key)
    if capturing and warmed_signature != signature:
        raise RuntimeError(
            f"Cake KDA affine {role} descriptors are not warmed for capture"
        )
    prepare_descriptors = 0 if capturing else int(warmed_signature != signature)
    try:
        module.run(
            q,
            k,
            v,
            g,
            beta,
            beta_tma,
            A_log,
            dt_bias,
            cu_seqlens,
            seq_order,
            state_indices,
            initial_state_bf16,
            out,
            final_state_bf16,
            initial_state_f32,
            final_state_f32,
            descriptor_storage,
            prepare_descriptors,
            num_heads,
            beta.stride(-2),
            state_slot_stride,
            scale,
            0.0,
            grid_x,
            1,
            1,
            stream_ptr,
        )
    except Exception:
        if prepare_descriptors:
            workspace._descriptor_signatures.pop(signature_key, None)
        raise
    if prepare_descriptors:
        workspace._descriptor_signatures[signature_key] = signature


def _run_cake_kda_affine_route(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    affine_plan: _CakeKDAAffinePlan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    scale: float,
    stream_ptr: int,
    capturing: bool,
) -> None:
    """Launch main -> map -> scan -> correction and publish indexed state."""

    plan = _cake_kda_affine_launch_plan(
        workspace=workspace,
        affine_plan=affine_plan,
        device=q.device,
        num_heads=num_heads,
        capturing=capturing,
    )
    empty_bf16 = plan.empty_bf16
    empty_f32 = plan.empty_f32
    empty_i32 = plan.empty_i32
    total_tokens = q.numel() // (num_heads * _FLASH_KDA_HEAD_DIM)
    q_flat = q.reshape(total_tokens, num_heads, _FLASH_KDA_HEAD_DIM)
    k_flat = k.reshape_as(q_flat)
    g_flat = g.reshape_as(q_flat)
    out_flat = out.reshape_as(q_flat)
    beta_flat = beta.reshape(total_tokens, num_heads)
    tail_tokens = total_tokens - plan.tail_start
    q_tail = q_flat[plan.tail_start :].view(
        1, tail_tokens, num_heads, _FLASH_KDA_HEAD_DIM
    )
    k_tail = k_flat[plan.tail_start :].view_as(q_tail)
    g_tail = g_flat[plan.tail_start :].view_as(q_tail)
    beta_tail = beta_flat[plan.tail_start :].view(1, tail_tokens, num_heads)
    main_beta_tma = _beta_tma_source(beta, workspace, chunk_tokens=32)
    tail_beta_tma = _beta_tma_source(beta_tail, workspace, chunk_tokens=32)
    compact_stride = num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM

    _run_cake_kda_affine_direct_role(
        workspace=workspace,
        module=plan.modules.main,
        role="main",
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        beta_tma=main_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.split_cu_seqlens,
        seq_order=plan.main_seq_order,
        state_indices=state_indices,
        initial_state_bf16=initial_state,
        out=out,
        final_state_bf16=empty_bf16,
        initial_state_f32=empty_f32,
        final_state_f32=plan.main_final,
        descriptor_storage=plan.main_descriptor_storage,
        num_heads=num_heads,
        state_slot_stride=initial_state.stride(0),
        scale=scale,
        grid_x=plan.num_parts * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    _run_cake_kda_affine_direct_role(
        workspace=workspace,
        module=plan.modules.map,
        role="map",
        q=q_tail,
        k=k_tail,
        v=plan.zero_v,
        g=g_tail,
        beta=beta_tail,
        beta_tma=tail_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.tail_cu_seqlens,
        seq_order=plan.tail_seq_order,
        state_indices=empty_i32,
        initial_state_bf16=plan.map_identity,
        out=plan.map_out,
        final_state_bf16=plan.map_state,
        initial_state_f32=plan.main_final,
        final_state_f32=empty_f32,
        descriptor_storage=plan.map_descriptor_storage,
        num_heads=num_heads,
        state_slot_stride=compact_stride,
        scale=scale,
        grid_x=(plan.num_parts - 1) * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    plan.modules.scan.run(
        plan.main_final,
        plan.map_state,
        plan.carry,
        num_heads,
        plan.num_parts,
        32 * num_heads,
        1,
        1,
        stream_ptr,
    )
    _run_cake_kda_affine_direct_role(
        workspace=workspace,
        module=plan.modules.correction,
        role="correction",
        q=q_tail,
        k=k_tail,
        v=plan.zero_v,
        g=g_tail,
        beta=beta_tail,
        beta_tma=tail_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.tail_cu_seqlens,
        seq_order=plan.tail_seq_order,
        state_indices=empty_i32,
        initial_state_bf16=empty_bf16,
        out=plan.correction_out,
        final_state_bf16=empty_bf16,
        initial_state_f32=plan.carry,
        final_state_f32=plan.correction_final,
        descriptor_storage=plan.correction_descriptor_storage,
        num_heads=num_heads,
        state_slot_stride=compact_stride,
        scale=scale,
        grid_x=(plan.num_parts - 1) * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    out_flat[plan.tail_start :].add_(
        plan.correction_out.reshape_as(out_flat[plan.tail_start :])
    )
    torch.add(
        plan.main_final[-1:],
        plan.correction_final[-1:],
        out=plan.final_compact,
    )
    plan.final_external.copy_(plan.final_compact)
    plan.state_indices_i64.copy_(state_indices)
    final_state.index_copy_(0, plan.state_indices_i64, plan.final_external)


def _uses_measured_sm100_persistent_policy(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
) -> bool:
    return compute_capability == (10, 0) and sm_count in (148, 152)


def _should_use_independent_dvsplit(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    max_sequence_length: int,
) -> bool:
    """Select M64 when its doubled fixed-layout grid remains one resident wave."""

    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and fixed_layout
        and num_sequences == 1
        and max_sequence_length >= _FLASH_KDA_INDEPENDENT_DVSPLIT_MIN_SEQUENCE_LENGTH
        and _FLASH_KDA_INDEPENDENT_DVSPLIT_CTAS * num_heads <= sm_count
    )


def _should_use_source_vtile_direct(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the source one-wave M128 schedule for long dense H96 work."""

    return (
        compute_capability == (10, 3)
        and fixed_layout
        and uniform_sequences
        and num_heads == 96
        and num_sequences * num_heads <= sm_count
        and max_sequence_length >= 4096
    )


def _should_use_source_vtile_persistent(
    *,
    compute_capability: tuple[int, int],
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the source persistent M128 schedule by work-per-CTA bucket."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability == (10, 3)
        and not fixed_layout
        and uniform_sequences
        and num_heads in (64, 96)
        and total_tasks % _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS == 0
        and total_tasks // _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS in (4, 6)
        and max_sequence_length >= 512
    )


def _should_use_small_bh_owner_helper(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    sequence_length: int,
) -> bool:
    """Select the fixed small-BH region whose eight-CTA groups fully reside."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and 0 < total_tasks <= _FLASH_KDA_SMALL_BH_MAX_TASKS
        and num_heads <= _FLASH_KDA_SMALL_BH_MAX_TASKS
        and sequence_length >= _FLASH_KDA_SMALL_BH_MIN_SEQUENCE_LENGTH
        and _FLASH_KDA_SMALL_BH_GROUP_SIZE * total_tasks <= sm_count
    )


def _bt16_chunks_per_prepare_cta(*, num_heads: int, total_chunks: int) -> int:
    if num_heads == 12:
        if total_chunks <= _FLASH_KDA_BT16_H12_CPC1_MAX_TOTAL_CHUNKS:
            return 1
        return _FLASH_KDA_BT16_H12_CHUNKS_PER_PREP_CTA
    if num_heads * total_chunks >= _FLASH_KDA_BT16_GENERAL_HIGH_WORK_MIN_CHUNK_HEADS:
        return _FLASH_KDA_BT16_GENERAL_HIGH_WORK_CHUNKS_PER_PREP_CTA
    return _FLASH_KDA_BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA


def _wave_quantized_bt16_prepare_ctas(
    *, rectangular_ctas: int, num_heads: int, sm_count: int
) -> int:
    if rectangular_ctas < _FLASH_KDA_BT16_PREP_WAVE_QUANT_MIN_WAVES * sm_count:
        return rectangular_ctas
    full_wave_ctas = (rectangular_ctas // sm_count) * sm_count
    if (
        full_wave_ctas < num_heads
        or full_wave_ctas * 100
        < rectangular_ctas * _FLASH_KDA_BT16_PREP_WAVE_QUANT_MIN_RETAINED_PERCENT
    ):
        return rectangular_ctas
    return full_wave_ctas


def _should_use_bt16_dense_wavefront(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    max_sequence_length: int,
) -> bool:
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and fixed_layout
        and num_sequences == 1
        and _FLASH_KDA_BT16_DENSE_MIN_HEADS
        <= num_heads
        <= _FLASH_KDA_BT16_DENSE_MAX_HEADS
        and max_sequence_length >= _FLASH_KDA_BT16_DENSE_MIN_SEQUENCE_LENGTH
        and _FLASH_KDA_BT16_VALUE_SPLITS * num_heads <= sm_count
    )


def _should_use_bt16_prepare_chain(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    max_sequence_length: int,
    n16_alternative: bool = False,
) -> bool:
    total_tasks = num_sequences * num_heads
    if n16_alternative:
        chain_waves = (
            _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks + sm_count - 1
        ) // sm_count
        if chain_waves <= 1:
            min_sequence_length = _FLASH_KDA_BT16_N16_ONE_CHAIN_WAVE_MIN_SEQUENCE_LENGTH
        elif chain_waves == 2:
            min_sequence_length = _FLASH_KDA_BT16_N16_TWO_CHAIN_WAVE_MIN_SEQUENCE_LENGTH
        else:
            min_sequence_length = _FLASH_KDA_BT16_N16_MULTI_WAVE_MIN_SEQUENCE_LENGTH
        max_tasks = _FLASH_KDA_BT16_N16_MAX_DIRECT_WAVES * sm_count
    elif total_tasks <= _FLASH_KDA_SMALL_BH_MAX_TASKS:
        min_sequence_length = _FLASH_KDA_BT16_LONG_MIN_SEQUENCE_LENGTH
        max_tasks = _FLASH_KDA_SMALL_BH_MAX_TASKS
    else:
        min_sequence_length = _FLASH_KDA_BT16_MID_MIN_SEQUENCE_LENGTH
        max_tasks = _FLASH_KDA_BT16_MID_MAX_TASKS
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and 0 < total_tasks <= max_tasks
        and max_sequence_length >= min_sequence_length
        and (n16_alternative or _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks <= sm_count)
    )


def _direct_m128_route(*, num_heads: int, max_sequence_length: int = 0) -> str:
    return (
        _FLASH_KDA_ROUTE_DIRECT_M128_N16
        if num_heads == 12 or 0 < max_sequence_length <= _FLASH_KDA_BT16_CHUNK
        else _FLASH_KDA_ROUTE_DIRECT_M128
    )


def _should_use_h12_direct_n32(
    *,
    compute_capability: tuple[int, int],
    num_heads: int,
    max_sequence_length: int,
) -> bool:
    """Select the measured H12 range where two N16 chunks lose to one N32."""

    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and num_heads == 12
        and _FLASH_KDA_H12_DIRECT_N32_MIN_SEQUENCE_LENGTH
        <= max_sequence_length
        <= _FLASH_KDA_H12_DIRECT_N32_MAX_SEQUENCE_LENGTH
    )


def _should_use_n32_tensor_state_decay(
    *,
    compute_capability: tuple[int, int],
    route: str,
    uniform_sequences: bool,
    num_heads: int,
    total_tasks: int,
    max_sequence_length: int,
) -> bool:
    """Select the measured full-tile tensor-core state-decay schedule."""

    return (
        compute_capability == (10, 3)
        and route == _FLASH_KDA_ROUTE_DIRECT_M128
        and uniform_sequences
        and num_heads >= 64
        and total_tasks >= 96
        and max_sequence_length >= 256
        and max_sequence_length % 32 == 0
    )


def _select_flash_kda_bf16_route(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> str:
    """Mirror the measured Cake route policy for the frozen BF16 portfolio."""

    direct_route = _direct_m128_route(
        num_heads=num_heads, max_sequence_length=max_sequence_length
    )
    if _should_use_bt16_dense_wavefront(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_BT16_M64
    if direct_route == _FLASH_KDA_ROUTE_DIRECT_M128_N16:
        if (
            compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
            and _FLASH_KDA_H12_DIRECT_N32_MIN_SEQUENCE_LENGTH
            <= max_sequence_length
            <= _FLASH_KDA_H12_DIRECT_N32_MAX_SEQUENCE_LENGTH
        ):
            return _FLASH_KDA_ROUTE_DIRECT_M128
        total_tasks = num_sequences * num_heads
        direct_waves = (total_tasks + sm_count - 1) // sm_count
        chain_waves = (
            _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks + sm_count - 1
        ) // sm_count
        if (
            compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
            and uniform_sequences
            and max_sequence_length > _FLASH_KDA_H12_DIRECT_N32_MAX_SEQUENCE_LENGTH
            and chain_waves > direct_waves
        ):
            return _FLASH_KDA_ROUTE_DIRECT_M128
        if total_tasks > 2 * sm_count and max_sequence_length >= 512:
            return _FLASH_KDA_ROUTE_DIRECT_M128
        if _should_use_bt16_prepare_chain(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_sequences=num_sequences,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            n16_alternative=True,
        ):
            return _FLASH_KDA_ROUTE_BT16_M64
        return direct_route
    if _should_use_bt16_prepare_chain(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_BT16_M64
    if _should_use_small_bh_owner_helper(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_SMALL_BH_M128
    if _requires_exact_n16_recurrence(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
    ):
        return _FLASH_KDA_ROUTE_DIRECT_M128_N16
    if (
        fixed_layout
        and num_sequences == 1
        and num_heads == 64
        and max_sequence_length >= 512
        and 2 * num_heads <= sm_count
    ):
        return _FLASH_KDA_ROUTE_M64
    if _should_use_uniform_piece_persistent(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    ):
        return _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128
    return direct_route


def _select_bt16_physical_variants(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    max_sequence_length: int,
) -> tuple["FlashKDAVariant", "FlashKDAVariant", bool]:
    dense_wavefront = _should_use_bt16_dense_wavefront(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    )
    prepare_variant: Literal["bt16_prepare", "bt16_prepare_beta_tma"] = (
        "bt16_prepare_beta_tma"
        if dense_wavefront and num_heads % _FLASH_KDA_BETA_TMA_HEADS_PER_BOX == 0
        else "bt16_prepare"
    )
    total_tasks = num_sequences * num_heads
    chain_variant: Literal[
        "bt16_chain_m64_s7",
        "bt16_chain_m64_s8",
        "bt16_chain_m64_s9",
    ]
    if _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks > sm_count:
        chain_variant = "bt16_chain_m64_s7"
    elif total_tasks <= 8 or (
        prepare_variant == "bt16_prepare_beta_tma"
        and _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks <= sm_count
    ):
        chain_variant = "bt16_chain_m64_s9"
    else:
        chain_variant = "bt16_chain_m64_s8"
    return prepare_variant, chain_variant, dense_wavefront


def _requires_exact_n16_recurrence(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
) -> bool:
    """Retain the N16 accuracy fallback for the 148-SM H96/N128 row."""

    return (
        sm_count == 148
        and not fixed_layout
        and num_sequences == 128
        and num_heads == 96
        and uniform_sequences
    )


def _uniform_persistent_worker_count(total_tasks: int, *, sm_count: int) -> int:
    if total_tasks <= 0 or sm_count <= 0:
        raise ValueError("total_tasks and sm_count must be positive")
    if total_tasks <= sm_count:
        return total_tasks
    trips = (total_tasks + sm_count - 1) // sm_count
    if total_tasks % trips == 0:
        balanced_workers = total_tasks // trips
        if balanced_workers >= _FLASH_KDA_PERSISTENT_MIN_BALANCED_CTAS:
            return balanced_workers
    return sm_count


def _make_uniform_head_grouped_bins(
    *,
    num_sequences: int,
    num_heads: int,
    worker_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total_tasks = num_sequences * num_heads
    if num_sequences <= 0 or num_heads <= 0 or not 0 < worker_count <= total_tasks:
        raise ValueError("head-grouped bins require positive sequence/head/task counts")
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_idx in range(worker_count):
        begin = worker_idx * total_tasks // worker_count
        end = (worker_idx + 1) * total_tasks // worker_count
        for head_major_idx in range(begin, end):
            head_idx, ordered_seq_idx = divmod(head_major_idx, num_sequences)
            task_ids.append(ordered_seq_idx * num_heads + head_idx)
        task_offsets.append(len(task_ids))
    return tuple(task_ids), tuple(task_offsets)


def _make_lpt_task_bins(
    ordered_sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    sm_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    total_tasks = len(ordered_sequence_lengths) * num_heads
    if (
        not ordered_sequence_lengths
        or num_heads <= 0
        or not 0 < sm_count <= total_tasks
    ):
        raise ValueError("LPT bins require positive sequence/head/task counts")
    bins: list[list[int]] = [[] for _ in range(sm_count)]
    loads = [0] * sm_count
    for ordered_seq_idx, seq_len in enumerate(ordered_sequence_lengths):
        chunk_count = (seq_len + 31) // 32
        for head_idx in range(num_heads):
            worker_idx = min(range(sm_count), key=lambda index: (loads[index], index))
            bins[worker_idx].append(ordered_seq_idx * num_heads + head_idx)
            loads[worker_idx] += chunk_count
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_tasks in bins:
        task_ids.extend(worker_tasks)
        task_offsets.append(len(task_ids))
    return tuple(task_ids), tuple(task_offsets), tuple(loads)


def _lpt_bins_are_balanced(loads: tuple[int, ...]) -> bool:
    return bool(loads) and (
        max(loads) * _FLASH_KDA_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
        <= sum(loads) * _FLASH_KDA_LPT_MAX_IMBALANCE_NUMERATOR
    )


def _should_use_lpt_persistent(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_heads: int,
    loads: tuple[int, ...],
) -> bool:
    """Select the measured H96 LPT route on exact SM100 device classes."""

    if (
        not _uses_measured_sm100_persistent_policy(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        or num_heads != 96
    ):
        return False
    if sm_count == 152:
        return bool(loads) and (
            max(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
            <= sum(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_NUMERATOR
        )
    return _lpt_bins_are_balanced(loads)


def _should_use_scalar_chunk_lpt(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
) -> bool:
    """Select the complete-chain scalar LPT schedule on mixed dense work."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and not uniform_sequences
        and num_heads in (64, 96)
        and max_sequence_length > 0
        and 2 * sm_count <= total_tasks < 1024
        and (max_sequence_length + _FLASH_KDA_M128_CHUNK - 1) // _FLASH_KDA_M128_CHUNK
        < 256
    )


@functools.lru_cache(maxsize=64)
def _make_uniform_piece_task_bins(
    *,
    num_sequences: int,
    num_heads: int,
    sequence_length: int,
    worker_count: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    tuple[int, ...],
]:
    """Split quantization-bound uniform chains across persistent CTA bins."""

    total_tasks = num_sequences * num_heads
    if (
        num_sequences <= 0
        or num_heads <= 0
        or sequence_length <= 0
        or worker_count <= 0
        or worker_count > total_tasks
    ):
        raise ValueError("uniform piece bins require positive resolved work")

    chunk_count = (sequence_length + _FLASH_KDA_M128_CHUNK - 1) // _FLASH_KDA_M128_CHUNK
    bins: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for task_idx in range(total_tasks):
        worker_idx = min(
            range(worker_count),
            key=lambda index: (loads[index], index),
        )
        bins[worker_idx].append((task_idx, 0, sequence_length, -1, -1))
        loads[worker_idx] += chunk_count

    base_tasks, extra_tasks = divmod(total_tasks, worker_count)
    piece_count = (
        min(base_tasks, worker_count // extra_tasks, chunk_count) if extra_tasks else 1
    )
    if piece_count >= 2:
        peak_load = (base_tasks + 1) * chunk_count
        peak_slots = [
            worker_idx for worker_idx, load in enumerate(loads) if load == peak_load
        ]
        if len(peak_slots) != extra_tasks:
            raise RuntimeError("uniform LPT peak count did not match task remainder")
        overflow_tasks = []
        for worker_idx in peak_slots:
            task = bins[worker_idx].pop()
            loads[worker_idx] -= chunk_count
            overflow_tasks.append(task[0])

        handoff_count = 0
        chunk_base = chunk_count // piece_count
        chunk_remainder = chunk_count % piece_count
        chunk_cuts = [0]
        for piece_idx in range(piece_count):
            # Put longer pieces last so every dependency gets at least the
            # preceding whole-chain interval of scheduling slack.
            piece_chunks = chunk_base + int(piece_idx >= piece_count - chunk_remainder)
            chunk_cuts.append(chunk_cuts[-1] + piece_chunks)

        for overflow_idx, task_idx in enumerate(overflow_tasks):
            handoffs = tuple(range(handoff_count, handoff_count + piece_count - 1))
            handoff_count += piece_count - 1
            for piece_idx in range(piece_count):
                chunk_start = chunk_cuts[piece_idx]
                chunk_end = chunk_cuts[piece_idx + 1]
                token_start = chunk_start * _FLASH_KDA_M128_CHUNK
                token_end = min(sequence_length, chunk_end * _FLASH_KDA_M128_CHUNK)
                source = -1 if piece_idx == 0 else handoffs[piece_idx - 1]
                destination = (
                    -1 if piece_idx + 1 == piece_count else handoffs[piece_idx]
                )
                worker_idx = piece_idx * extra_tasks + overflow_idx
                insert_at = min(1 + piece_idx, len(bins[worker_idx]))
                bins[worker_idx].insert(
                    insert_at,
                    (
                        task_idx,
                        token_start,
                        token_end - token_start,
                        source,
                        destination,
                    ),
                )
                loads[worker_idx] += chunk_end - chunk_start
    else:
        handoff_count = 0

    task_ids: list[int] = []
    task_token_starts: list[int] = []
    task_token_counts: list[int] = []
    task_state_sources: list[int] = []
    task_state_destinations: list[int] = []
    task_offsets = [0]
    for worker_tasks in bins:
        for task_idx, token_start, token_count, source, destination in worker_tasks:
            task_ids.append(task_idx)
            task_token_starts.append(token_start)
            task_token_counts.append(token_count)
            task_state_sources.append(source)
            task_state_destinations.append(destination)
        task_offsets.append(len(task_ids))
    return (
        tuple(task_ids),
        tuple(task_offsets),
        tuple(task_token_starts),
        tuple(task_token_counts),
        tuple(task_state_sources),
        tuple(task_state_destinations),
        handoff_count,
        tuple(loads),
    )


@functools.lru_cache(maxsize=64)
def _persistent_m128_roofline(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    sequence_length: int,
    use_initial_state: bool,
    store_final_state: bool,
) -> Optional[_PersistentM128Roofline]:
    """Compare direct and recurrence-piece persistent critical paths."""

    if compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES:
        return None
    if num_sequences <= 0 or num_heads <= 0 or sequence_length <= 0:
        raise ValueError("persistent-M128 roofline requires resolved positive extents")

    resident_ctas_per_sm = min(
        _FLASH_KDA_BLACKWELL_MAX_THREADS_PER_SM
        // _FLASH_KDA_PERSISTENT_THREADS_PER_CTA,
        _FLASH_KDA_BLACKWELL_SMEM_BYTES_PER_SM
        // _FLASH_KDA_PERSISTENT_SMEM_POOL_BYTES_PER_CTA,
        _FLASH_KDA_BLACKWELL_TMEM_COLS_PER_SM
        // _FLASH_KDA_PERSISTENT_TMEM_COLS_PER_CTA,
    )
    if resident_ctas_per_sm <= 0:
        raise RuntimeError(
            "persistent-M128 schedule is not resident on the selected device"
        )
    worker_count = sm_count * resident_ctas_per_sm
    total_tasks = num_sequences * num_heads
    if total_tasks <= worker_count:
        return None

    (
        _task_ids,
        task_offsets,
        _token_starts,
        token_counts,
        state_sources,
        state_destinations,
        handoff_count,
        _loads,
    ) = _make_uniform_piece_task_bins(
        num_sequences=num_sequences,
        num_heads=num_heads,
        sequence_length=sequence_length,
        worker_count=worker_count,
    )
    if handoff_count == 0:
        return None

    worker_flops_per_ns = _FLASH_KDA_BLACKWELL_BF16_TFLOPS * 1_000.0 / worker_count
    worker_bytes_per_ns = _FLASH_KDA_BLACKWELL_HBM_GBPS / worker_count
    chunk_ns = max(
        _FLASH_KDA_PERSISTENT_TENSOR_FLOPS_PER_CHUNK / worker_flops_per_ns,
        _FLASH_KDA_PERSISTENT_STREAM_BYTES_PER_CHUNK / worker_bytes_per_ns,
    )
    state_transfer_ns = _FLASH_KDA_PERSISTENT_STATE_BYTES / worker_bytes_per_ns
    task_refill_ns = _FLASH_KDA_PERSISTENT_TASK_REFILL_CHUNKS * chunk_ns
    chunks_per_task = (
        sequence_length + _FLASH_KDA_M128_CHUNK - 1
    ) // _FLASH_KDA_M128_CHUNK
    direct_task_ns = chunks_per_task * chunk_ns
    if use_initial_state:
        direct_task_ns += state_transfer_ns
    if store_final_state:
        direct_task_ns += state_transfer_ns
    direct_ns = ((total_tasks + worker_count - 1) // worker_count) * direct_task_ns

    entry_count = len(token_counts)
    edges: list[set[int]] = [set() for _ in range(entry_count)]
    indegree = [0] * entry_count

    def add_edge(source: int, destination: int) -> None:
        if destination not in edges[source]:
            edges[source].add(destination)
            indegree[destination] += 1

    for worker_idx in range(worker_count):
        begin = task_offsets[worker_idx]
        end = task_offsets[worker_idx + 1]
        for entry_idx in range(begin + 1, end):
            add_edge(entry_idx - 1, entry_idx)
    handoff_producers = {
        destination: entry_idx
        for entry_idx, destination in enumerate(state_destinations)
        if destination >= 0
    }
    if len(handoff_producers) != handoff_count:
        raise RuntimeError("piece roofline did not resolve every handoff producer")
    for entry_idx, source in enumerate(state_sources):
        if source >= 0:
            try:
                producer = handoff_producers[source]
            except KeyError as exc:
                raise RuntimeError(
                    f"piece roofline did not resolve handoff source {source}"
                ) from exc
            add_edge(producer, entry_idx)

    ready = [entry_idx for entry_idx, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    worker_first_entries = frozenset(task_offsets[:-1])
    earliest_start = [0.0] * entry_count
    finish = [0.0] * entry_count
    visited = 0
    while ready:
        entry_idx = heapq.heappop(ready)
        duration = (
            (token_counts[entry_idx] + _FLASH_KDA_M128_CHUNK - 1)
            // _FLASH_KDA_M128_CHUNK
            * chunk_ns
        )
        if entry_idx not in worker_first_entries:
            duration += task_refill_ns
        if state_sources[entry_idx] >= 0 or use_initial_state:
            duration += state_transfer_ns
        if state_destinations[entry_idx] >= 0 or store_final_state:
            duration += state_transfer_ns
        finish[entry_idx] = earliest_start[entry_idx] + duration
        visited += 1
        for successor in edges[entry_idx]:
            earliest_start[successor] = max(
                earliest_start[successor], finish[entry_idx]
            )
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if visited != entry_count:
        raise RuntimeError("piece roofline dependency graph contains a cycle")

    return _PersistentM128Roofline(
        resident_ctas_per_sm=resident_ctas_per_sm,
        worker_count=worker_count,
        handoff_count=handoff_count,
        chunk_ns=chunk_ns,
        state_transfer_ns=state_transfer_ns,
        task_refill_ns=task_refill_ns,
        direct_ns=direct_ns,
        piece_ns=max(finish),
    )


def _should_use_uniform_piece_persistent(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    max_sequence_length: int,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> bool:
    """Select recurrence pieces when their occupancy-aware roofline wins."""

    if sm_count not in (148, 152) or not uniform_sequences or max_sequence_length <= 0:
        return False
    estimate = _persistent_m128_roofline(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        sequence_length=max_sequence_length,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )
    return estimate is not None and estimate.piece_ns < estimate.direct_ns


def _select_bf16_route(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
    lpt_loads: tuple[int, ...],
    max_sequence_length: int = 0,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> str:
    """Select one material BF16 schedule family from resolved host metadata."""

    direct_route = _direct_m128_route(
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    )
    if num_heads == 64 and _should_use_independent_dvsplit(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_M64
    if _should_use_source_vtile_direct(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_SOURCE_VTILE_M128
    if _should_use_source_vtile_persistent(
        compute_capability=compute_capability,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_SOURCE_VTILE_M128
    if _should_use_bt16_dense_wavefront(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_BT16_M64
    if direct_route == _FLASH_KDA_ROUTE_DIRECT_M128_N16:
        if _should_use_h12_direct_n32(
            compute_capability=compute_capability,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        ):
            return _FLASH_KDA_ROUTE_DIRECT_M128
        total_tasks = num_sequences * num_heads
        direct_waves = (total_tasks + sm_count - 1) // sm_count
        chain_waves = (
            _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks + sm_count - 1
        ) // sm_count
        if (
            compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
            and uniform_sequences
            and max_sequence_length > _FLASH_KDA_H12_DIRECT_N32_MAX_SEQUENCE_LENGTH
            and chain_waves > direct_waves
        ):
            return _FLASH_KDA_ROUTE_DIRECT_M128
        if total_tasks > 2 * sm_count and max_sequence_length >= 512:
            return _FLASH_KDA_ROUTE_DIRECT_M128
        if _should_use_bt16_prepare_chain(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_sequences=num_sequences,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            n16_alternative=True,
        ):
            return _FLASH_KDA_ROUTE_BT16_M64
        return direct_route
    if _should_use_bt16_prepare_chain(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_BT16_M64
    if _should_use_small_bh_owner_helper(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_SMALL_BH_M128
    if _requires_exact_n16_recurrence(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
    ):
        return _FLASH_KDA_ROUTE_DIRECT_M128_N16
    if _should_use_independent_dvsplit(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_M64
    if _should_use_scalar_chunk_lpt(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
    ):
        return _FLASH_KDA_ROUTE_SCALAR_CHUNK_LPT_M128
    total_tasks = num_sequences * num_heads
    if _should_use_uniform_piece_persistent(
        compute_capability=compute_capability,
        sm_count=sm_count,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    ):
        return _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128
    if (
        _uses_measured_sm100_persistent_policy(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        and num_heads in (64, 96)
        and uniform_sequences
        and total_tasks > sm_count
    ):
        return _FLASH_KDA_ROUTE_HEAD_GROUPED_M128
    if (
        not uniform_sequences
        and total_tasks > sm_count
        and _should_use_lpt_persistent(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_heads=num_heads,
            loads=lpt_loads,
        )
    ):
        return _FLASH_KDA_ROUTE_LPT_M128
    return direct_route


def select_bf16_schedule_route(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> str:
    """Select one physical BF16 schedule from runtime-resolved shape metadata.

    This is the canonical metadata adapter for callers that need to share the
    production dispatch policy without reproducing its shape guards. Shape
    values select among materially different schedule families; they do not
    create per-shape kernel identities.
    """

    if sm_count <= 0 or num_heads <= 0:
        raise ValueError("sm_count and num_heads must be positive")
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("sequence_lengths must contain positive lengths")
    num_sequences = len(sequence_lengths)
    uniform_sequences = len(set(sequence_lengths)) == 1
    total_tasks = num_sequences * num_heads
    lpt_loads: tuple[int, ...] = ()
    if (
        _uses_measured_sm100_persistent_policy(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        and not uniform_sequences
        and total_tasks > sm_count
    ):
        ordered_sequence_lengths = tuple(sorted(sequence_lengths, reverse=True))
        _task_ids, _task_offsets, lpt_loads = _make_lpt_task_bins(
            ordered_sequence_lengths,
            num_heads=num_heads,
            sm_count=sm_count,
        )
    return _select_bf16_route(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        lpt_loads=lpt_loads,
        max_sequence_length=max(sequence_lengths),
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )


def _persistent_task_plan(
    sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    sm_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Build Cake's measured SM100 task plan, or return direct-route evidence."""

    total_tasks = len(sequence_lengths) * num_heads
    if sm_count not in (148, 152) or num_heads == 12 or total_tasks <= sm_count:
        return None
    sequence_order = tuple(
        sorted(
            range(len(sequence_lengths)),
            key=lambda index: sequence_lengths[index],
            reverse=True,
        )
    )
    ordered_lengths = tuple(sequence_lengths[index] for index in sequence_order)
    if len(set(sequence_lengths)) == 1:
        if num_heads not in (64, 96):
            return None
        worker_count = _uniform_persistent_worker_count(
            total_tasks,
            sm_count=sm_count,
        )
        task_ids, task_offsets = _make_uniform_head_grouped_bins(
            num_sequences=len(sequence_lengths),
            num_heads=num_heads,
            worker_count=worker_count,
        )
        return sequence_order, task_ids, task_offsets
    if num_heads != 96:
        return None
    task_ids, task_offsets, loads = _make_lpt_task_bins(
        ordered_lengths,
        num_heads=num_heads,
        sm_count=sm_count,
    )
    if sm_count == 152:
        if not loads or (
            max(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
            > sum(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_NUMERATOR
        ):
            return None
    elif not _lpt_bins_are_balanced(loads):
        return None
    return sequence_order, task_ids, task_offsets


def _cached_tensor(
    key: tuple,
    factory,
    *,
    capture_error: str,
) -> torch.Tensor:
    with _flash_kda_tensor_cache_lock:
        tensor = _flash_kda_tensor_cache.get(key)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(capture_error)
            tensor = factory()
            _flash_kda_tensor_cache[key] = tensor
        return tensor


def _cached_int32_metadata(
    *,
    device: torch.device,
    kind: str,
    values: tuple[int, ...],
) -> torch.Tensor:
    key = (kind, *_stream_cache_key(device), values)
    return _cached_tensor(
        key,
        lambda: torch.tensor(values, dtype=torch.int32, device=device),
        capture_error=(
            f"recurrent_kda {kind} metadata is not warmed for CUDA graph "
            "capture; invoke the same shape once before capture"
        ),
    )


def _fixed_cu_seqlens(
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    key = ("fixed_cu", *_stream_cache_key(device), batch_size, seq_len)
    return _cached_tensor(
        key,
        lambda: torch.arange(
            0,
            batch_size * seq_len + 1,
            seq_len,
            dtype=torch.int64,
            device=device,
        ),
        capture_error=(
            "fixed-layout recurrent_kda prefill metadata is not warmed for "
            "CUDA graph capture; invoke the same shape once before capture"
        ),
    )


def _identity_seq_order(
    *,
    device: torch.device,
    num_sequences: int,
) -> torch.Tensor:
    key = ("seq_order", *_stream_cache_key(device), num_sequences)
    return _cached_tensor(
        key,
        lambda: torch.arange(num_sequences, dtype=torch.int32, device=device),
        capture_error=(
            "recurrent_kda prefill seq_order is not warmed for CUDA graph "
            "capture; pass a preallocated seq_order or warm the shape first"
        ),
    )


def _dummy_bf16(device: torch.device) -> torch.Tensor:
    key = ("dummy_bf16", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.bfloat16, device=device),
        capture_error=(
            "recurrent_kda prefill dummy state is not warmed for CUDA graph "
            "capture; invoke the same device once before capture"
        ),
    )


def _dummy_i32(device: torch.device) -> torch.Tensor:
    key = ("dummy_i32", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.int32, device=device),
        capture_error=(
            "recurrent_kda prefill dummy int32 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _dummy_i64(device: torch.device) -> torch.Tensor:
    key = ("dummy_i64", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.int64, device=device),
        capture_error=(
            "recurrent_kda prefill dummy int64 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _dummy_f32(device: torch.device) -> torch.Tensor:
    key = ("dummy_f32", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.float32, device=device),
        capture_error=(
            "recurrent_kda prefill dummy float32 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _dummy_u32(device: torch.device) -> torch.Tensor:
    key = ("dummy_u32", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.uint32, device=device),
        capture_error=(
            "recurrent_kda prefill dummy uint32 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _empty_cuda_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = ("empty", dtype, *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(0, dtype=dtype, device=device),
        capture_error=(
            "recurrent_kda prefill empty typed carrier is not warmed for CUDA "
            "graph capture; invoke the same device once before capture"
        ),
    )


def _generated_affine_carriers(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
) -> _GeneratedAffineCarriers:
    carriers = workspace._generated_affine_carriers
    if carriers is None:
        carriers = _GeneratedAffineCarriers(
            dummy_bf16=_dummy_bf16(device),
            dummy_i32=_dummy_i32(device),
            dummy_i64=_dummy_i64(device),
            dummy_f32=_dummy_f32(device),
            dummy_u32=_dummy_u32(device),
            empty_bf16=_empty_cuda_tensor(device, torch.bfloat16),
            empty_f32=_empty_cuda_tensor(device, torch.float32),
            empty_i64=_empty_cuda_tensor(device, torch.int64),
            empty_u8=_empty_cuda_tensor(device, torch.uint8),
        )
        workspace._generated_affine_carriers = carriers
    return carriers


def _stream_cache_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return device_index, int(stream.cuda_stream)


def _get_stream_workspace(device: torch.device) -> _FlashKDAStreamWorkspace:
    key = _stream_cache_key(device)
    with _flash_kda_stream_workspaces_lock:
        workspace = _flash_kda_stream_workspaces.get(key)
        if workspace is None:
            workspace = _FlashKDAStreamWorkspace(device)
            _flash_kda_stream_workspaces[key] = workspace
        return workspace


def _cached_packed_task_metadata(
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    num_heads: int,
    sm_count: int,
    build_persistent_plan: bool,
) -> _PackedTaskMetadata:
    """Cache host-built sequence order and optional persistent task bins."""

    signature = (
        int(cu_seqlens._version),
        total_tokens,
        num_heads,
        sm_count,
        build_persistent_plan,
    )
    with workspace._packed_metadata_lock:
        cached_metadata = workspace._packed_metadata
        if (
            workspace._packed_metadata_tensor is cu_seqlens
            and workspace._packed_metadata_signature == signature
            and cached_metadata is not None
        ):
            return cached_metadata
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "packed recurrent_kda prefill metadata is not warmed for "
                "CUDA graph capture; eagerly invoke the same offsets once "
                "with this RecurrentKDAPrefillWorkspace before capture"
            )
        offsets = tuple(int(value) for value in cu_seqlens.tolist())
        if (
            not offsets
            or offsets[0] != 0
            or offsets[-1] != total_tokens
            or any(
                right <= left for left, right in zip(offsets, offsets[1:], strict=False)
            )
        ):
            raise ValueError(
                "cu_seqlens must start at zero, be strictly increasing, "
                "and end at the packed token count"
            )
        sequence_lengths = tuple(
            right - left for left, right in zip(offsets, offsets[1:], strict=False)
        )
        sequence_order = tuple(
            sorted(
                range(len(sequence_lengths)),
                key=lambda index: sequence_lengths[index],
                reverse=True,
            )
        )
        persistent_plan = (
            _persistent_task_plan(
                sequence_lengths,
                num_heads=num_heads,
                sm_count=sm_count,
            )
            if build_persistent_plan
            else None
        )
        metadata = (
            sequence_order,
            persistent_plan,
            len(set(sequence_lengths)) == 1,
            offsets,
            sequence_lengths,
        )
        workspace._packed_metadata_tensor = cu_seqlens
        workspace._packed_metadata_signature = signature
        workspace._packed_metadata = metadata
        return metadata


def _workspace_buffer(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    attribute: str,
    device: torch.device,
    numel: int,
    capture_error: str,
    dtype: torch.dtype = torch.bfloat16,
    zero_on_allocate: bool = False,
) -> torch.Tensor:
    buffer = getattr(workspace, attribute)
    capturing = torch.cuda.is_current_stream_capturing()
    if buffer is None or buffer.numel() < numel:
        if capturing:
            raise RuntimeError(capture_error)
        factory = torch.zeros if zero_on_allocate else torch.empty
        buffer = factory(numel, dtype=dtype, device=device)
        setattr(workspace, attribute, buffer)
    elif buffer.dtype != dtype:
        raise RuntimeError(
            f"recurrent_kda workspace buffer {attribute} has dtype "
            f"{buffer.dtype}, expected {dtype}"
        )
    return buffer[:numel]


def _state_scratch(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    numel = math.prod(shape)
    return _workspace_buffer(
        workspace=workspace,
        attribute="_state_scratch",
        device=device,
        numel=numel,
        capture_error=(
            "recurrent_kda prefill final-state workspace is not large enough "
            "for CUDA graph capture; warm the largest shape on this stream "
            "before capture"
        ),
    ).view(shape)


def _beta_tma_source(
    beta: torch.Tensor,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    if beta.stride(-1) != 1:
        raise ValueError("beta must have unit head stride")
    if batch_size == 1:
        beta_flat = beta[0]
    else:
        if beta.stride(0) != seq_len * beta.stride(1):
            raise ValueError("beta batch/token dimensions must collapse without a copy")
        beta_flat = beta.as_strided(
            (total_tokens, num_heads),
            (beta.stride(1), beta.stride(2)),
        )
    if (
        total_tokens >= chunk_tokens
        and num_heads >= _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        and beta_flat.data_ptr() % 16 == 0
        and beta_flat.stride(0) * beta.element_size() % 16 == 0
    ):
        return beta_flat
    padded_tokens = max(total_tokens, chunk_tokens)
    padded_heads = (
        (num_heads + _FLASH_KDA_BETA_TMA_HEADS_PER_BOX - 1)
        // _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        * _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
    )
    shape = (padded_tokens, padded_heads)
    padded = _workspace_buffer(
        workspace=workspace,
        attribute="_beta_padding",
        device=beta.device,
        numel=math.prod(shape),
        capture_error=(
            "recurrent_kda prefill beta TMA workspace is not large enough for "
            "CUDA graph capture; warm the largest padded token/head shape on "
            "this stream before capture"
        ),
    ).view(shape)
    # The frozen binding refreshes head-padded storage from ``beta`` immediately
    # before launching the frozen kernel. Keeping pack + main-kernel submission
    # in one FFI call avoids two Python-dispatched activities and their host gap,
    # while retaining stable storage for the TMA descriptor and CUDA graphs.
    return padded


def _pair_packed_beta_tma_source(beta: torch.Tensor) -> Optional[torch.Tensor]:
    """Alias dense H12 beta rows as a TensorMap-legal two-token carrier."""

    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    if beta.stride(-1) != 1:
        raise ValueError("beta must have unit head stride")
    if batch_size == 1:
        beta_flat = beta[0]
    else:
        if beta.stride(0) != seq_len * beta.stride(1):
            raise ValueError("beta batch/token dimensions must collapse without a copy")
        beta_flat = beta.as_strided(
            (total_tokens, num_heads),
            (beta.stride(1), beta.stride(2)),
        )
    if (
        num_heads != 12
        or total_tokens % 2 != 0
        or not beta_flat.is_contiguous()
        or beta_flat.data_ptr() % 16 != 0
    ):
        return None
    return beta_flat.view(total_tokens // 2, 24)


def _small_bh_workspace(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    total_tasks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packet_slots = total_tasks * _FLASH_KDA_SMALL_BH_RING_STAGES
    packet_shape = (
        packet_slots * _FLASH_KDA_SMALL_BH_PACKET_ROWS,
        _FLASH_KDA_SMALL_BH_PACKET_ELEMENTS,
    )
    capture_error = (
        "recurrent_kda small-BH packet workspace is not large enough for "
        "CUDA graph capture; warm the largest small-BH shape on this stream "
        "before capture"
    )
    packet_workspace = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_workspace",
        device=device,
        numel=math.prod(packet_shape),
        capture_error=capture_error,
    ).view(packet_shape)
    packet_ready = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_ready",
        device=device,
        numel=packet_slots,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    packet_consumed = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_consumed",
        device=device,
        numel=packet_slots,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    helper_done = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_helper_done",
        device=device,
        numel=total_tasks,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    return packet_workspace, packet_ready, packet_consumed, helper_done


def _piece_persistent_workspace(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    handoff_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if handoff_count <= 0:
        raise ValueError("piece-persistent workspace requires a positive handoff count")
    capture_error = (
        "recurrent_kda piece-persistent workspace is not large enough for "
        "CUDA graph capture; warm the largest shape on this stream before capture"
    )
    mid_state = _workspace_buffer(
        workspace=workspace,
        attribute="_piece_mid_state",
        device=device,
        numel=handoff_count * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM,
        capture_error=capture_error,
    ).view(handoff_count, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
    mid_state_ready = _workspace_buffer(
        workspace=workspace,
        attribute="_piece_mid_state_ready",
        device=device,
        numel=handoff_count,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    return mid_state, mid_state_ready


def _bt16_workspace(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    offsets: tuple[int, ...],
    num_heads: int,
    sm_count: int,
    dense_wavefront: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    """Resolve stable BT16 metadata, factor buffers, and prepare grid size."""

    chunk_counts = tuple(
        (end - start + _FLASH_KDA_BT16_CHUNK - 1) // _FLASH_KDA_BT16_CHUNK
        for start, end in zip(offsets, offsets[1:], strict=False)
    )
    host_cu_chunks = [0]
    host_chunk_to_seq: list[int] = []
    for sequence_index, chunk_count in enumerate(chunk_counts):
        host_cu_chunks.append(host_cu_chunks[-1] + chunk_count)
        host_chunk_to_seq.extend([sequence_index] * chunk_count)
    total_chunks = host_cu_chunks[-1]
    metadata_signature = (offsets, num_heads)
    capturing = torch.cuda.is_current_stream_capturing()
    if workspace._bt16_metadata_signature != metadata_signature:
        if capturing:
            raise RuntimeError(
                "BT16 recurrent_kda metadata is not warmed for CUDA graph "
                "capture; eagerly invoke the same offsets once with this "
                "RecurrentKDAPrefillWorkspace before capture"
            )
        cu_chunks = _workspace_buffer(
            workspace=workspace,
            attribute="_bt16_cu_chunks",
            device=device,
            numel=len(host_cu_chunks),
            capture_error="BT16 cu_chunks workspace is not warmed for capture",
            dtype=torch.int32,
        )
        chunk_to_seq = _workspace_buffer(
            workspace=workspace,
            attribute="_bt16_chunk_to_seq",
            device=device,
            numel=len(host_chunk_to_seq),
            capture_error="BT16 chunk_to_seq workspace is not warmed for capture",
            dtype=torch.int32,
        )
        cu_chunks.copy_(torch.tensor(host_cu_chunks, dtype=torch.int32, device=device))
        chunk_to_seq.copy_(
            torch.tensor(host_chunk_to_seq, dtype=torch.int32, device=device)
        )
        workspace._bt16_metadata_signature = metadata_signature
    else:
        assert workspace._bt16_cu_chunks is not None
        assert workspace._bt16_chunk_to_seq is not None
        cu_chunks = workspace._bt16_cu_chunks[: len(host_cu_chunks)]
        chunk_to_seq = workspace._bt16_chunk_to_seq[: len(host_chunk_to_seq)]

    padded_tokens = total_chunks * _FLASH_KDA_BT16_CHUNK
    factor_numel = num_heads * padded_tokens * _FLASH_KDA_HEAD_DIM
    qd = _workspace_buffer(
        workspace=workspace,
        attribute="_bt16_qd",
        device=device,
        numel=factor_numel,
        capture_error="BT16 qd workspace is not large enough for capture",
    ).view(1, num_heads, padded_tokens, _FLASH_KDA_HEAD_DIM)
    kd = _workspace_buffer(
        workspace=workspace,
        attribute="_bt16_kd",
        device=device,
        numel=factor_numel,
        capture_error="BT16 kd workspace is not large enough for capture",
    ).view_as(qd)
    w = _workspace_buffer(
        workspace=workspace,
        attribute="_bt16_w",
        device=device,
        numel=factor_numel,
        capture_error="BT16 w workspace is not large enough for capture",
    ).view_as(qd)
    qk_numel = num_heads * total_chunks * _FLASH_KDA_BT16_CHUNK * _FLASH_KDA_BT16_CHUNK
    qk = _workspace_buffer(
        workspace=workspace,
        attribute="_bt16_qk",
        device=device,
        numel=qk_numel,
        capture_error="BT16 qk workspace is not large enough for capture",
    ).view(
        1,
        num_heads,
        total_chunks,
        _FLASH_KDA_BT16_CHUNK,
        _FLASH_KDA_BT16_CHUNK,
    )
    diag = _workspace_buffer(
        workspace=workspace,
        attribute="_bt16_diag",
        device=device,
        numel=num_heads * total_chunks * _FLASH_KDA_HEAD_DIM,
        capture_error="BT16 diagonal workspace is not large enough for capture",
        dtype=torch.float32,
    ).view(1, num_heads, total_chunks, _FLASH_KDA_HEAD_DIM)

    chunks_per_cta = _bt16_chunks_per_prepare_cta(
        num_heads=num_heads, total_chunks=total_chunks
    )
    prepare_ctas = ((total_chunks + chunks_per_cta - 1) // chunks_per_cta) * num_heads
    prepare_ctas = _wave_quantized_bt16_prepare_ctas(
        rectangular_ctas=prepare_ctas,
        num_heads=num_heads,
        sm_count=sm_count,
    )
    if dense_wavefront:
        prepare_ctas = min(
            num_heads * total_chunks,
            _FLASH_KDA_BT16_DENSE_PREP_WAVES * sm_count,
        )
    return (
        cu_chunks,
        chunk_to_seq,
        qd,
        kd,
        w,
        qk,
        diag,
        total_chunks,
        prepare_ctas,
    )


def _tensor_descriptor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _descriptor_signature(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_tma: torch.Tensor,
    out: torch.Tensor,
    packet_workspace: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
) -> tuple:
    signature = tuple(
        _tensor_descriptor_signature(tensor) for tensor in (q, k, v, g, beta_tma, out)
    )
    if packet_workspace is not None:
        signature += (_tensor_descriptor_signature(packet_workspace),)
    if state_checkpoints is not None:
        signature += (_tensor_descriptor_signature(state_checkpoints),)
    return signature


def _bind_workspace(
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
    explicit: bool,
) -> None:
    if workspace.device != device:
        raise ValueError(
            "RecurrentKDAPrefillWorkspace is bound to "
            f"{workspace.device}, but recurrent_kda inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace is bound to a different CUDA "
            "stream; warm and capture it on one stream"
        )
    if explicit and workspace._captured:
        reuse_kind = "captured by another CUDA graph" if capturing else "reused eagerly"
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace has participated in CUDA graph "
            f"capture and cannot be {reuse_kind} or mutated"
        )


def _storage_ranges_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False

    def storage_end(tensor: torch.Tensor) -> int:
        max_element_offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 0
        )
        return tensor.data_ptr() + (max_element_offset + 1) * tensor.element_size()

    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = storage_end(left)
    right_end = storage_end(right)
    return left_start < right_end and right_start < left_end


def _check_output_does_not_overlap_inputs(
    output: torch.Tensor,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> None:
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
    ):
        if tensor is not None and _storage_ranges_overlap(output, tensor):
            raise ValueError(
                f"output must not overlap {name} for frozen recurrent_kda prefill"
            )


def _validate_prefill_seq_order(
    seq_order: Optional[torch.Tensor],
    *,
    fixed_layout: bool,
    num_sequences: int,
    device: torch.device,
) -> torch.Tensor:
    if seq_order is None:
        return _identity_seq_order(device=device, num_sequences=num_sequences)
    if fixed_layout:
        raise ValueError("seq_order is only supported for packed recurrent_kda prefill")
    if not isinstance(seq_order, torch.Tensor):
        raise TypeError("seq_order must be a torch.Tensor")
    if (
        not seq_order.is_cuda
        or seq_order.device != device
        or seq_order.dtype != torch.int32
        or seq_order.ndim != 1
        or not seq_order.is_contiguous()
        or seq_order.numel() != num_sequences
    ):
        raise ValueError(
            "seq_order must be a contiguous CUDA int32 tensor with one "
            f"entry per sequence ({num_sequences})"
        )
    return seq_order


def _is_cuda_version_at_least(version: str) -> bool:
    # Keep JIT imports lazy so importing the public KDA facade does not
    # initialize the extension toolchain.
    from .jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


def _select_flash_kda_prefill_target(device: torch.device) -> "FlashKDATarget":
    compute_capability = get_compute_capability(device)
    if compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "frozen recurrent-KDA prefill requires compute capability 10.0 "
            "(SM100a; B200/GB200) or 10.3 (SM103a; B300/GB300); got "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
    if compute_capability == (10, 0):
        if not _is_cuda_version_at_least("12.8"):
            raise RuntimeError(
                "frozen recurrent-KDA prefill on compute capability 10.0 "
                "requires CUDA 12.8 or newer"
            )
        return "sm100a"
    if not _is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "frozen recurrent-KDA prefill on compute capability 10.3 requires "
            "CUDA 12.9 or newer for the sm_103a target"
        )
    return "sm103a"


def _select_cake_kda_prefill_target(device: torch.device) -> "CakeKDATarget":
    compute_capability = get_compute_capability(device)
    if compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "Cake recurrent-KDA prefill requires compute capability 10.0 "
            "(SM100a; B200/GB200) or 10.3 (SM103a; B300/GB300); got "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
    if compute_capability == (10, 0):
        if not _is_cuda_version_at_least("12.8"):
            raise RuntimeError(
                "Cake recurrent-KDA prefill on compute capability 10.0 "
                "requires CUDA 12.8 or newer"
            )
        return "sm100a"
    if not _is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "Cake recurrent-KDA prefill on compute capability 10.3 requires "
            "CUDA 12.9 or newer"
        )
    return "sm103a"


def _get_flash_kda_prefill_module(variant: "FlashKDAVariant", target: "FlashKDATarget"):
    from .jit.flash_kda import get_flash_kda_prefill_module

    return get_flash_kda_prefill_module(variant, target)


def _flash_kda_generated_serving_native_abi(
    *,
    use_state_indices: bool,
    checkpoint_every_n_tokens: int,
    beta_token_stride: int,
    num_heads: int,
    state_slot_stride: int,
) -> bool:
    """Resolve the serving ABI from public tensor metadata only."""

    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint_every_n_tokens must be nonnegative")
    if beta_token_stride <= 0 or num_heads <= 0 or state_slot_stride <= 0:
        raise ValueError("generated FlashKDA ABI strides and heads must be positive")
    return (
        use_state_indices
        or checkpoint_every_n_tokens != 0
        or beta_token_stride != num_heads
        or state_slot_stride != num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
    )


def _flash_kda_generated_bt16_stage_count(
    *,
    total_tasks: int,
    sm_count: int,
    use_beta_tma: bool,
) -> int:
    """Resolve the receipt-backed BT16 chain stage specialization."""

    if total_tasks <= 0 or sm_count <= 0:
        raise ValueError("BT16 stage selection requires positive tasks and SMs")
    split_tasks = _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks
    if split_tasks > sm_count:
        return 7
    if total_tasks <= 8 or (use_beta_tma and split_tasks <= sm_count):
        return 9
    return 8


def _flash_kda_generated_full_n32_chunks(
    sequence_lengths: tuple[int, ...],
) -> bool:
    """Return exact M64 full-chunk specialization from resolved lengths."""

    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise ValueError("sequence_lengths must contain positive lengths")
    return all(length % _FLASH_KDA_M128_CHUNK == 0 for length in sequence_lengths)


def _flash_kda_generated_direct_specialization(
    *,
    target: "FlashKDATarget",
    route: str,
    num_heads: int,
    num_sequences: int,
    uniform_sequences: bool,
    max_sequence_length: int,
    serving_native_abi: bool,
    gate_kind: str,
    checkpoint_every_n_tokens: int,
    pair_packed_beta: bool,
    state_dtype_is_fp32: bool,
    n32_ft_slab: bool = False,
    pdl_wait_initial_state_f32: bool = False,
    pdl_publish_final_state: bool = False,
    affine_main_indexed_initial: bool = False,
    affine_main_indexed_initial_bf16: bool = False,
) -> dict[str, object]:
    """Compute the direct-family specialization from runtime-resolved facts."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"generated FlashKDA has no exact architecture for {target!r}")
    if route not in (
        _FLASH_KDA_ROUTE_DIRECT_M128,
        _FLASH_KDA_ROUTE_DIRECT_M128_N16,
        _FLASH_KDA_ROUTE_AFFINE_M128,
    ):
        raise ValueError(f"route {route!r} is not a direct-M128 family")
    if num_heads <= 0 or num_sequences <= 0 or max_sequence_length <= 0:
        raise ValueError("direct specialization requires positive resolved extents")
    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint_every_n_tokens must be nonnegative")
    if gate_kind not in ("lower_bound", "unbounded_softplus"):
        raise ValueError(f"unsupported KDA gate kind {gate_kind!r}")
    if affine_main_indexed_initial and not state_dtype_is_fp32:
        raise ValueError("affine indexed initial state requires FP32 state I/O")
    if affine_main_indexed_initial_bf16 and not affine_main_indexed_initial:
        raise ValueError(
            "BF16 affine indexed initial state requires indexed initial state"
        )
    unbounded_softplus = gate_kind == "unbounded_softplus"
    direct_n16 = route == _FLASH_KDA_ROUTE_DIRECT_M128_N16
    chunk = _FLASH_KDA_BT16_CHUNK if direct_n16 else _FLASH_KDA_M128_CHUNK
    scalar_beta = (
        route
        in (
            _FLASH_KDA_ROUTE_DIRECT_M128,
            _FLASH_KDA_ROUTE_DIRECT_M128_N16,
            _FLASH_KDA_ROUTE_AFFINE_M128,
        )
        and num_heads == 12
    )
    early_n32_state_pack = (
        scalar_beta
        and not direct_n16
        and max_sequence_length
        <= _FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH
    )
    generic_register_inverse = (
        not direct_n16
        and not unbounded_softplus
        and (
            scalar_beta
            or max_sequence_length
            >= _FLASH_KDA_N32_REGISTER_INVERSE_MIN_SEQUENCE_LENGTH
        )
    )
    n32_prediction_first = (
        target == "sm103a"
        and route == _FLASH_KDA_ROUTE_DIRECT_M128
        and num_heads >= _FLASH_KDA_N32_PREDICTION_FIRST_MIN_HEADS
        and max_sequence_length >= _FLASH_KDA_N32_REGISTER_INVERSE_MIN_SEQUENCE_LENGTH
        and (
            uniform_sequences
            or max_sequence_length
            >= _FLASH_KDA_N32_PREDICTION_FIRST_MIXED_MIN_SEQUENCE_LENGTH
        )
        and generic_register_inverse
    )
    tensor_state_decay = (
        checkpoint_every_n_tokens == 0
        and _should_use_n32_tensor_state_decay(
            compute_capability=(10, 3) if target == "sm103a" else (10, 0),
            route=route,
            uniform_sequences=uniform_sequences,
            num_heads=num_heads,
            total_tasks=num_sequences * num_heads,
            max_sequence_length=max_sequence_length,
        )
        and n32_prediction_first
    )
    return {
        "chunk": chunk,
        "serving_native_abi": serving_native_abi,
        "gate_kind": gate_kind,
        "checkpoint_tma": bool(checkpoint_every_n_tokens and direct_n16),
        "pair_packed_beta": pair_packed_beta,
        "scalar_beta": scalar_beta,
        "early_n32_state_pack": early_n32_state_pack,
        "generic_register_inverse": generic_register_inverse,
        "n32_prediction_first": n32_prediction_first,
        "tensor_state_decay": tensor_state_decay,
        "state_dtype_is_fp32": state_dtype_is_fp32,
        "n32_ft_slab": n32_ft_slab and not direct_n16,
        "pdl_wait_initial_state_f32": pdl_wait_initial_state_f32,
        "pdl_publish_final_state": pdl_publish_final_state,
        "affine_main_indexed_initial": affine_main_indexed_initial,
        "affine_main_indexed_initial_bf16": affine_main_indexed_initial_bf16,
    }


def _flash_kda_generated_vtile_specialization(
    *,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    fixed_layout: bool,
    use_initial_state: bool,
    store_final_state: bool,
    scale: float,
    lower_bound: float,
    state_dtype_is_fp32: bool,
) -> dict[str, object]:
    """Compute the source-vtile specialization without source identities."""

    if num_heads <= 0:
        raise ValueError("vtile specialization requires a positive head count")
    if not math.isfinite(scale) or not math.isfinite(lower_bound):
        raise ValueError("vtile scale and lower_bound must be finite")
    full_n32_chunks = _flash_kda_generated_full_n32_chunks(sequence_lengths)
    total_tasks = len(sequence_lengths) * num_heads
    worker_count = (
        total_tasks if fixed_layout else _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS
    )
    if total_tasks % worker_count != 0:
        raise ValueError(
            "vtile task count must divide its runtime-resolved worker count"
        )
    persistent_tasks = total_tasks // worker_count
    return {
        "full_n32_chunks": full_n32_chunks,
        "num_heads": num_heads,
        "use_initial_state": use_initial_state,
        "store_final_state": store_final_state,
        "scale": float(scale),
        "lower_bound": float(lower_bound),
        "persistent_mode": persistent_tasks > 1,
        "persistent_six_task_schedule": persistent_tasks == 6,
        "persistent_stride_head_aligned": worker_count % num_heads == 0,
        "state_dtype_is_fp32": state_dtype_is_fp32,
    }


def _flash_kda_generated_bt16_prepare_specialization() -> dict[str, object]:
    return {}


def _flash_kda_generated_bt16_chain_specialization(
    *,
    total_tasks: int,
    sm_count: int,
    use_beta_tma: bool,
    state_dtype_is_fp32: bool,
    serving_native_abi: bool,
) -> dict[str, object]:
    return {
        "bt16_stage_count": _flash_kda_generated_bt16_stage_count(
            total_tasks=total_tasks,
            sm_count=sm_count,
            use_beta_tma=use_beta_tma,
        ),
        "state_dtype_is_fp32": state_dtype_is_fp32,
        "serving_native_abi": serving_native_abi,
    }


def _flash_kda_generated_m64_specialization(
    *,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    use_initial_state: bool,
    store_final_state: bool,
    scale: float,
    lower_bound: float,
    state_dtype_is_fp32: bool,
) -> dict[str, object]:
    if num_heads <= 0 or not math.isfinite(scale) or not math.isfinite(lower_bound):
        raise ValueError(
            "M64 specialization requires finite scalars and positive heads"
        )
    return {
        "full_n32_chunks": _flash_kda_generated_full_n32_chunks(sequence_lengths),
        "num_heads": num_heads,
        "use_initial_state": use_initial_state,
        "store_final_state": store_final_state,
        "scale": float(scale),
        "lower_bound": float(lower_bound),
        "state_dtype_is_fp32": state_dtype_is_fp32,
    }


def _flash_kda_generated_scalar_lpt_specialization(
    *,
    num_heads: int,
    use_initial_state: bool,
    store_final_state: bool,
    scale: float,
    lower_bound: float,
    state_dtype_is_fp32: bool,
) -> dict[str, object]:
    if num_heads <= 0 or not math.isfinite(scale) or not math.isfinite(lower_bound):
        raise ValueError(
            "scalar-LPT specialization requires finite scalars and positive heads"
        )
    return {
        "num_heads": num_heads,
        "use_initial_state": use_initial_state,
        "store_final_state": store_final_state,
        "scale": float(scale),
        "lower_bound": float(lower_bound),
        "persistent_schedule": True,
        "state_dtype_is_fp32": state_dtype_is_fp32,
    }


def _flash_kda_generated_taskized_persistent_specialization(
    *,
    piece_tasks: bool,
    state_dtype_is_fp32: bool,
) -> dict[str, object]:
    return {
        "piece_tasks": piece_tasks,
        "state_dtype_is_fp32": state_dtype_is_fp32,
    }


def _flash_kda_generated_small_bh_specialization(
    *,
    serving_native_abi: bool,
    state_dtype_is_fp32: bool,
) -> dict[str, object]:
    return {
        "serving_native_abi": serving_native_abi,
        "state_dtype_is_fp32": state_dtype_is_fp32,
    }


def _flash_kda_generated_affine_scan_specialization() -> dict[str, object]:
    return {"use_pdl": True}


@functools.cache
def _flash_kda_affine_token_offsets(
    *,
    total_tokens: int,
    num_heads: int,
    sm_count: int,
    state_dtype: torch.dtype,
) -> Optional[tuple[int, ...]]:
    """Resolve the receipt-backed affine split, or ``None`` when ineligible."""

    if (
        total_tokens <= 0
        or total_tokens % _FLASH_KDA_M128_CHUNK
        or num_heads <= 0
        or num_heads > 32
        or 2 * num_heads > sm_count
        or state_dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    chunks = total_tokens // _FLASH_KDA_M128_CHUNK
    minimum_chunks = max(256, num_heads * 32) if state_dtype == torch.float32 else 256
    if chunks < minimum_chunks:
        return None
    candidate_parts = min(
        sm_count,
        max(2, sm_count // num_heads),
        max(2, chunks // 32),
    )
    if candidate_parts < 8 and chunks < 2048:
        return None
    chunks_per_part = (chunks + candidate_parts - 1) // candidate_parts
    chunk_offsets = tuple(
        sorted(
            {
                min(index * chunks_per_part, chunks)
                for index in range(candidate_parts + 1)
            }
        )
    )
    if len(chunk_offsets) < 3:
        return None
    return tuple(offset * _FLASH_KDA_M128_CHUNK for offset in chunk_offsets)


def _affine_workspace_buffer(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    name: str,
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    zero_on_allocate: bool = False,
) -> torch.Tensor:
    numel = math.prod(shape)
    buffer = workspace._affine_buffers.get(name)
    if buffer is None or buffer.numel() < numel or buffer.dtype != dtype:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "affine FlashKDA workspace is not warmed for "
                f"{name}; invoke the largest shape before capture"
            )
        factory = torch.zeros if zero_on_allocate else torch.empty
        buffer = factory(numel, dtype=dtype, device=device)
        workspace._affine_buffers[name] = buffer
    return buffer[:numel].view(shape)


def _affine_descriptor_storage(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    role: str,
    device: torch.device,
) -> torch.Tensor:
    storage = _affine_workspace_buffer(
        workspace=workspace,
        name=f"descriptor_storage_{role}",
        device=device,
        shape=(_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,),
        dtype=torch.uint8,
    )
    if storage.data_ptr() % 64:
        raise RuntimeError(f"affine {role} descriptor storage is not 64-byte aligned")
    return storage


def _make_flash_kda_generated_selector_key(
    *,
    target: "FlashKDATarget",
    route: str,
    route_role: str,
    state_mode: str,
    family_specialization: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact public selector key consumed by the sealed registry."""

    arch_by_target = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
    try:
        arch = arch_by_target[target]
    except KeyError as error:
        raise ValueError(
            f"generated FlashKDA has no exact architecture for {target!r}"
        ) from error
    if not all(
        isinstance(value, str) and value for value in (route, route_role, state_mode)
    ):
        raise ValueError("generated FlashKDA selector identity fields must be nonempty")
    try:
        abi_family = _FLASH_KDA_GENERATED_ROUTE_ABI_FAMILY[(route, route_role)]
    except KeyError as error:
        raise ValueError(
            "generated FlashKDA has no receipt-backed ABI family for "
            f"route {route!r} role {route_role!r}"
        ) from error
    if state_mode not in (
        "bf16",
        "fp32",
        "bf16_f32_dependency",
        "none",
    ):
        raise ValueError(f"generated FlashKDA has no state mode {state_mode!r}")
    stateless_family = abi_family in ("bt16_prepare", "affine_scan")
    if (state_mode == "none") != stateless_family:
        raise ValueError(
            f"generated FlashKDA {abi_family} requires "
            "state_mode="
            f"{'none' if stateless_family else 'bf16, fp32, or bf16_f32_dependency'}"
        )
    expected_fields = _FLASH_KDA_GENERATED_SPECIALIZATION_FIELDS[abi_family]
    if not isinstance(family_specialization, Mapping) or set(
        family_specialization
    ) != set(expected_fields):
        missing = sorted(set(expected_fields) - set(family_specialization))
        unknown = sorted(set(family_specialization) - set(expected_fields))
        raise ValueError(
            f"generated FlashKDA {abi_family} specialization fields differ; "
            f"missing={missing}, unknown={unknown}"
        )
    specialization_vector: list[list[object]] = []
    for field in expected_fields:
        value = family_specialization[field]
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"generated FlashKDA specialization {field!r} must be scalar"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"generated FlashKDA specialization {field!r} must be finite"
            )
        specialization_vector.append([field, value])
    return {
        "arch": arch,
        "route": route,
        "route_role": route_role,
        "abi_family": abi_family,
        "state_mode": state_mode,
        "family_specialization_vector": specialization_vector,
    }


def _get_flash_kda_generated_module(selector_key: dict[str, object]):
    """Resolve metadata and load the exact receipt-selected physical module."""

    from .jit.flash_kda import (
        get_flash_kda_generated_module_for_selector,
        load_flash_kda_generated_module,
    )

    metadata = get_flash_kda_generated_module_for_selector(selector_key)
    return metadata, load_flash_kda_generated_module(metadata.variant_id)


def _generated_descriptor_storage(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    variant_id: str,
    device: torch.device,
    bytes_required: int = _FLASH_KDA_DESCRIPTOR_STORAGE_BYTES,
) -> torch.Tensor:
    if bytes_required <= 0:
        raise ValueError("generated descriptor storage size must be positive")
    storage = workspace._generated_descriptor_storages.get(variant_id)
    if storage is None or storage.numel() < bytes_required:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "generated FlashKDA descriptor storage is not warmed for "
                f"{variant_id}; invoke the same route before capture"
            )
        storage = torch.empty(bytes_required, dtype=torch.uint8, device=device)
        workspace._generated_descriptor_storages[variant_id] = storage
    if storage.data_ptr() % 64:
        raise RuntimeError(
            f"generated FlashKDA descriptor storage for {variant_id} is not 64-byte aligned"
        )
    return storage


def _generated_descriptor_prepare(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    variant_id: str,
    signature: tuple,
    capturing: bool,
) -> int:
    key = f"generated:{variant_id}"
    warmed_signature = workspace._descriptor_signatures.get(key)
    if capturing:
        if warmed_signature != signature:
            raise RuntimeError(
                "RecurrentKDAPrefillWorkspace is not warmed for the exact "
                f"generated module {variant_id} descriptor signature"
            )
        return 0
    return int(warmed_signature != signature)


def _record_generated_descriptor_signature(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    variant_id: str,
    signature: tuple,
    prepared: int,
) -> None:
    if prepared:
        workspace._descriptor_signatures[f"generated:{variant_id}"] = signature


def _clear_generated_descriptor_signature(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    variant_id: str,
    prepared: int,
) -> None:
    if prepared:
        workspace._descriptor_signatures.pop(f"generated:{variant_id}", None)


def _build_generated_scalar_schedule(
    sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    worker_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the exact one-wave scalar-chunk LPT schedule."""

    if worker_count <= 0 or num_heads <= 0:
        raise ValueError("scalar schedule requires positive workers and heads")
    bins: list[tuple[int, int, list[tuple[int, int]]]] = [
        (0, worker, []) for worker in range(worker_count)
    ]
    heapq.heapify(bins)
    ordered_sequences = sorted(
        range(len(sequence_lengths)),
        key=lambda sequence: (sequence_lengths[sequence] + 31) // 32,
        reverse=True,
    )
    for sequence in ordered_sequences:
        chunks = (sequence_lengths[sequence] + 31) // 32
        for head in range(num_heads):
            load, worker, tasks = heapq.heappop(bins)
            tasks.append((sequence * num_heads + head, chunks))
            heapq.heappush(bins, (load + chunks, worker, tasks))

    # Rebalance two bins exactly before encoding the scalar-chunk schedule.
    while True:
        light_index = min(
            range(worker_count), key=lambda index: (bins[index][0], bins[index][1])
        )
        heavy_index = max(
            range(worker_count), key=lambda index: (bins[index][0], -bins[index][1])
        )
        light_load, light_worker, light_tasks = bins[light_index]
        heavy_load, heavy_worker, heavy_tasks = bins[heavy_index]
        pair_tasks = light_tasks + heavy_tasks
        pair_load = light_load + heavy_load
        reachable = {0: 0}
        for task_index, (_task, chunks) in enumerate(pair_tasks):
            for load, mask in list(reachable.items()):
                reachable.setdefault(load + chunks, mask | (1 << task_index))
        split_load = min(
            reachable,
            key=lambda load: (
                max(load, pair_load - load),
                abs(pair_load - 2 * load),
            ),
        )
        if max(split_load, pair_load - split_load) >= heavy_load:
            break
        split_mask = reachable[split_load]
        bins[light_index] = (
            split_load,
            light_worker,
            [
                task
                for index, task in enumerate(pair_tasks)
                if split_mask & (1 << index)
            ],
        )
        bins[heavy_index] = (
            pair_load - split_load,
            heavy_worker,
            [
                task
                for index, task in enumerate(pair_tasks)
                if not split_mask & (1 << index)
            ],
        )

    while worker_count >= 3:
        ordered_bins = sorted(
            range(worker_count), key=lambda index: (bins[index][0], bins[index][1])
        )
        light_index = ordered_bins[0]
        heavy_index = ordered_bins[-1]
        average_load = sum(load for load, _worker, _tasks in bins) / worker_count
        middle_index = min(
            ordered_bins[1:-1],
            key=lambda index: (
                abs(bins[index][0] - average_load),
                -max(chunks for _task, chunks in bins[index][2]),
                bins[index][1],
            ),
        )
        selected_indices = (light_index, middle_index, heavy_index)
        selected_tasks = [task for index in selected_indices for task in bins[index][2]]
        selected_load = sum(chunks for _task, chunks in selected_tasks)
        heavy_load = bins[heavy_index][0]
        if selected_load > 1024:
            break
        reachable_pairs = {(0, 0): 0}
        processed_load = 0
        for task_index, (_task, chunks) in enumerate(selected_tasks):
            next_pairs = dict(reachable_pairs)
            for (first_load, second_load), assignment in reachable_pairs.items():
                third_load = processed_load - first_load - second_load
                if first_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load + chunks, second_load),
                        assignment | (1 << (2 * task_index)),
                    )
                if second_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load, second_load + chunks),
                        assignment | (2 << (2 * task_index)),
                    )
                if third_load + chunks >= heavy_load:
                    next_pairs.pop((first_load, second_load), None)
            reachable_pairs = next_pairs
            processed_load += chunks
        if not reachable_pairs:
            break
        (first_load, second_load), assignment = min(
            reachable_pairs.items(),
            key=lambda item: max(item[0][0], item[0][1], selected_load - sum(item[0])),
        )
        split_loads = (
            first_load,
            second_load,
            selected_load - first_load - second_load,
        )
        if max(split_loads) >= heavy_load:
            break
        split_tasks: list[list[tuple[int, int]]] = [[], [], []]
        for task_index, task in enumerate(selected_tasks):
            encoded_group = (assignment >> (2 * task_index)) & 3
            group = 0 if encoded_group == 1 else 1 if encoded_group == 2 else 2
            split_tasks[group].append(task)
        for index, load, tasks in zip(
            selected_indices, split_loads, split_tasks, strict=True
        ):
            _old_load, worker, _old_tasks = bins[index]
            bins[index] = (load, worker, tasks)

    bins.sort(key=lambda item: item[1])
    counts = [load for load, _worker, _tasks in bins]
    stride = max(counts)
    schedule = [0] * (worker_count * stride)
    for _load, worker, tasks in bins:
        slot = 0
        for encoded_task, chunks in tasks:
            for local_chunk in range(chunks):
                schedule[worker * stride + slot] = (
                    encoded_task | (local_chunk << 10) | (chunks << 18)
                )
                slot += 1
    return (
        torch.tensor(schedule, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int32, device=device),
        stride,
    )


def _generated_scalar_schedule(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    worker_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (sequence_lengths, num_heads, worker_count)
    schedule = workspace._generated_scalar_schedules.get(key)
    if schedule is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "generated scalar-LPT schedule is not warmed for CUDA graph capture"
            )
        schedule = _build_generated_scalar_schedule(
            sequence_lengths,
            num_heads=num_heads,
            worker_count=worker_count,
            device=device,
        )
        workspace._generated_scalar_schedules[key] = schedule
    return schedule


def _run_bt16_prepare_chain(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    target: "FlashKDATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_order: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    offsets: tuple[int, ...],
    num_heads: int,
    sm_count: int,
    compute_capability: tuple[int, int],
    fixed_layout: bool,
    max_sequence_length: int,
    use_initial_state: bool,
    store_final_state: bool,
    scale: float,
    lower_bound: float,
    stream_ptr: int,
    capturing: bool,
) -> None:
    total_tasks = (len(offsets) - 1) * num_heads
    prepare_variant, chain_variant, dense_wavefront = _select_bt16_physical_variants(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=len(offsets) - 1,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    )
    (
        cu_chunks,
        chunk_to_seq,
        qd,
        kd,
        w,
        qk,
        diag,
        total_chunks,
        prepare_ctas,
    ) = _bt16_workspace(
        workspace=workspace,
        device=q.device,
        offsets=offsets,
        num_heads=num_heads,
        sm_count=sm_count,
        dense_wavefront=dense_wavefront,
    )

    prepare_tensors = (
        q,
        k,
        g,
        beta,
        A_log,
        dt_bias,
        cu_seqlens,
        cu_chunks,
        chunk_to_seq,
        qd,
        kd,
        w,
        qk,
        diag,
    )
    prepare_signature = tuple(
        _tensor_descriptor_signature(tensor) for tensor in prepare_tensors
    )
    chain_signature = tuple(
        _tensor_descriptor_signature(tensor)
        for tensor in (
            qd,
            kd,
            w,
            qk,
            diag,
            v,
            cu_seqlens,
            cu_chunks,
            seq_order,
            out,
        )
    )
    signatures = {
        prepare_variant: prepare_signature,
        chain_variant: chain_signature,
    }
    prepare_flags: dict["FlashKDAVariant", int] = {}
    for variant, signature in signatures.items():
        warmed_signature = workspace._descriptor_signatures.get(variant)
        if capturing and warmed_signature != signature:
            raise RuntimeError(
                "RecurrentKDAPrefillWorkspace is not warmed for the exact "
                f"{variant} descriptor signature; eagerly invoke the same "
                "call on this stream before capture"
            )
        prepare_flags[variant] = 0 if capturing else int(warmed_signature != signature)

    combined_variant: Optional["FlashKDAVariant"] = None
    if prepare_variant == "bt16_prepare" and chain_variant == "bt16_chain_m64_s8":
        combined_variant = "bt16_prepare_chain_m64_s8"
    combined_module = (
        _get_flash_kda_prefill_module(combined_variant, target)
        if combined_variant is not None
        else None
    )
    prepare_module = (
        None
        if combined_module is not None
        else _get_flash_kda_prefill_module(prepare_variant, target)
    )
    chain_module = (
        None
        if combined_module is not None
        else _get_flash_kda_prefill_module(chain_variant, target)
    )
    try:
        if combined_module is not None:
            combined_module.run(
                q,
                k,
                g,
                beta,
                A_log,
                dt_bias,
                cu_seqlens,
                cu_chunks,
                chunk_to_seq,
                qd,
                kd,
                w,
                qk,
                diag,
                v,
                seq_order,
                initial_state,
                out,
                final_state,
                workspace._descriptor_storages[prepare_variant],
                workspace._descriptor_storages[chain_variant],
                prepare_flags[prepare_variant],
                prepare_flags[chain_variant],
                total_chunks,
                num_heads,
                lower_bound,
                prepare_ctas,
                int(use_initial_state),
                int(store_final_state),
                scale,
                _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks,
                stream_ptr,
            )
        else:
            assert prepare_module is not None
            assert chain_module is not None
            prepare_module.run(
                q,
                k,
                g,
                beta,
                A_log,
                dt_bias,
                cu_seqlens,
                cu_chunks,
                chunk_to_seq,
                qd,
                kd,
                w,
                qk,
                diag,
                workspace._descriptor_storages[prepare_variant],
                prepare_flags[prepare_variant],
                total_chunks,
                num_heads,
                lower_bound,
                prepare_ctas,
                stream_ptr,
            )
            chain_module.run(
                qd,
                kd,
                w,
                qk,
                diag,
                v,
                cu_seqlens,
                cu_chunks,
                seq_order,
                initial_state,
                out,
                final_state,
                workspace._descriptor_storages[chain_variant],
                prepare_flags[chain_variant],
                num_heads,
                int(use_initial_state),
                int(store_final_state),
                scale,
                _FLASH_KDA_BT16_VALUE_SPLITS * total_tasks,
                stream_ptr,
            )
    except Exception:
        for variant, flag in prepare_flags.items():
            if flag:
                workspace._descriptor_signatures.pop(variant, None)
        raise
    for variant, flag in prepare_flags.items():
        if flag:
            workspace._descriptor_signatures[variant] = signatures[variant]


def _run_generated_bt16_prepare_chain(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    target: "FlashKDATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_order: torch.Tensor,
    state_indices: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    offsets: tuple[int, ...],
    num_heads: int,
    sm_count: int,
    compute_capability: tuple[int, int],
    fixed_layout: bool,
    max_sequence_length: int,
    use_initial_state: bool,
    store_final_state: bool,
    use_state_indices: bool,
    state_slot_stride: int,
    state_dtype_is_fp32: bool,
    scale: float,
    lower_bound: float,
    stream_ptr: int,
    capturing: bool,
) -> None:
    total_tasks = (len(offsets) - 1) * num_heads
    _prepare_variant, _chain_variant, dense_wavefront = _select_bt16_physical_variants(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=len(offsets) - 1,
        num_heads=num_heads,
        max_sequence_length=max_sequence_length,
    )
    (
        cu_chunks,
        chunk_to_seq,
        qd,
        kd,
        w,
        qk,
        diag,
        total_chunks,
        prepare_ctas,
    ) = _bt16_workspace(
        workspace=workspace,
        device=q.device,
        offsets=offsets,
        num_heads=num_heads,
        sm_count=sm_count,
        dense_wavefront=dense_wavefront,
    )
    prepare_key = _make_flash_kda_generated_selector_key(
        target=target,
        route=_FLASH_KDA_ROUTE_BT16_M64,
        route_role="bt16_prepare",
        state_mode="none",
        family_specialization=_flash_kda_generated_bt16_prepare_specialization(),
    )
    use_beta_tma = (
        dense_wavefront and num_heads % _FLASH_KDA_BETA_TMA_HEADS_PER_BOX == 0
    )
    serving_native_abi = _flash_kda_generated_serving_native_abi(
        use_state_indices=use_state_indices,
        checkpoint_every_n_tokens=0,
        beta_token_stride=beta.stride(-2),
        num_heads=num_heads,
        state_slot_stride=state_slot_stride,
    )
    chain_key = _make_flash_kda_generated_selector_key(
        target=target,
        route=_FLASH_KDA_ROUTE_BT16_M64,
        route_role="main",
        state_mode="fp32" if state_dtype_is_fp32 else "bf16",
        family_specialization=_flash_kda_generated_bt16_chain_specialization(
            total_tasks=total_tasks,
            sm_count=sm_count,
            use_beta_tma=use_beta_tma,
            state_dtype_is_fp32=state_dtype_is_fp32,
            serving_native_abi=serving_native_abi,
        ),
    )
    prepare_metadata, prepare_module = _get_flash_kda_generated_module(prepare_key)
    chain_metadata, chain_module = _get_flash_kda_generated_module(chain_key)
    prepare_signature = tuple(
        _tensor_descriptor_signature(tensor)
        for tensor in (
            q,
            k,
            g,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            cu_chunks,
            chunk_to_seq,
            qd,
            kd,
            w,
            qk,
            diag,
        )
    )
    chain_signature = tuple(
        _tensor_descriptor_signature(tensor)
        for tensor in (
            qd,
            kd,
            w,
            qk,
            diag,
            v,
            cu_seqlens,
            cu_chunks,
            seq_order,
            out,
        )
    )
    prepare_flag = _generated_descriptor_prepare(
        workspace=workspace,
        variant_id=prepare_metadata.variant_id,
        signature=prepare_signature,
        capturing=capturing,
    )
    chain_flag = _generated_descriptor_prepare(
        workspace=workspace,
        variant_id=chain_metadata.variant_id,
        signature=chain_signature,
        capturing=capturing,
    )
    prepare_storage = _generated_descriptor_storage(
        workspace=workspace,
        variant_id=prepare_metadata.variant_id,
        device=q.device,
        bytes_required=_FLASH_KDA_SEVEN_DESCRIPTOR_STORAGE_BYTES,
    )
    chain_storage = _generated_descriptor_storage(
        workspace=workspace,
        variant_id=chain_metadata.variant_id,
        device=q.device,
        bytes_required=_FLASH_KDA_SEVEN_DESCRIPTOR_STORAGE_BYTES,
    )
    try:
        prepare_module.run(
            q,
            k,
            g,
            beta,
            A_log,
            dt_bias,
            cu_seqlens,
            cu_chunks,
            chunk_to_seq,
            qd,
            kd,
            w,
            qk,
            diag,
            prepare_storage,
            prepare_flag,
            total_chunks,
            num_heads,
            lower_bound,
            prepare_ctas,
            1,
            1,
            stream_ptr,
        )
        chain_module.run(
            qd,
            kd,
            w,
            qk,
            diag,
            v,
            cu_seqlens,
            cu_chunks,
            seq_order,
            state_indices,
            initial_state,
            out,
            final_state,
            chain_storage,
            chain_flag,
            num_heads,
            state_slot_stride,
            int(use_state_indices),
            int(use_initial_state),
            int(store_final_state),
            scale,
            2 * total_tasks,
            1,
            1,
            stream_ptr,
        )
    except Exception:
        _clear_generated_descriptor_signature(
            workspace=workspace,
            variant_id=prepare_metadata.variant_id,
            prepared=prepare_flag,
        )
        _clear_generated_descriptor_signature(
            workspace=workspace,
            variant_id=chain_metadata.variant_id,
            prepared=chain_flag,
        )
        raise
    _record_generated_descriptor_signature(
        workspace=workspace,
        variant_id=prepare_metadata.variant_id,
        signature=prepare_signature,
        prepared=prepare_flag,
    )
    _record_generated_descriptor_signature(
        workspace=workspace,
        variant_id=chain_metadata.variant_id,
        signature=chain_signature,
        prepared=chain_flag,
    )


def _run_generated_single_route(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    target: "FlashKDATarget",
    route: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_tma: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_order: torch.Tensor,
    state_indices: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    state_checkpoints: torch.Tensor,
    checkpoint_cu_starts: torch.Tensor,
    checkpoint_every_n_tokens: int,
    sequence_lengths: tuple[int, ...],
    fixed_layout: bool,
    uniform_sequences: bool,
    num_heads: int,
    sm_count: int,
    use_state_indices: bool,
    use_initial_state: bool,
    store_final_state: bool,
    state_slot_stride: int,
    state_dtype_is_fp32: bool,
    scale: float,
    lower_bound: float,
    persistent_task_ids: Optional[torch.Tensor],
    persistent_task_offsets: Optional[torch.Tensor],
    piece_task_token_starts: Optional[torch.Tensor],
    piece_task_token_counts: Optional[torch.Tensor],
    piece_task_state_sources: Optional[torch.Tensor],
    piece_task_state_destinations: Optional[torch.Tensor],
    piece_mid_state: Optional[torch.Tensor],
    piece_mid_state_ready: Optional[torch.Tensor],
    packet_workspace: Optional[torch.Tensor],
    packet_ready: Optional[torch.Tensor],
    packet_consumed: Optional[torch.Tensor],
    helper_done: Optional[torch.Tensor],
    stream_ptr: int,
    capturing: bool,
) -> bool:
    total_tasks = len(sequence_lengths) * num_heads
    state_mode = "fp32" if state_dtype_is_fp32 else "bf16"
    beta_token_stride = beta.stride(-2)
    serving_native_abi = _flash_kda_generated_serving_native_abi(
        use_state_indices=use_state_indices,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_token_stride=beta_token_stride,
        num_heads=num_heads,
        state_slot_stride=state_slot_stride,
    )
    route_role = "main"
    if route in (_FLASH_KDA_ROUTE_DIRECT_M128, _FLASH_KDA_ROUTE_DIRECT_M128_N16):
        pair_packed_beta = (
            route == _FLASH_KDA_ROUTE_DIRECT_M128
            and num_heads == 12
            and max(sequence_lengths)
            > _FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH
            and beta_tma.ndim == 2
            and beta_tma.shape[1] == 24
        )
        specialization = _flash_kda_generated_direct_specialization(
            target=target,
            route=route,
            num_heads=num_heads,
            num_sequences=len(sequence_lengths),
            uniform_sequences=uniform_sequences,
            max_sequence_length=max(sequence_lengths),
            serving_native_abi=serving_native_abi,
            gate_kind="lower_bound",
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            pair_packed_beta=pair_packed_beta,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    elif route == _FLASH_KDA_ROUTE_SOURCE_VTILE_M128:
        specialization = _flash_kda_generated_vtile_specialization(
            sequence_lengths=sequence_lengths,
            num_heads=num_heads,
            fixed_layout=fixed_layout,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            scale=scale,
            lower_bound=lower_bound,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    elif route == _FLASH_KDA_ROUTE_M64:
        specialization = _flash_kda_generated_m64_specialization(
            sequence_lengths=sequence_lengths,
            num_heads=num_heads,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            scale=scale,
            lower_bound=lower_bound,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    elif route == _FLASH_KDA_ROUTE_SCALAR_CHUNK_LPT_M128:
        specialization = _flash_kda_generated_scalar_lpt_specialization(
            num_heads=num_heads,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            scale=scale,
            lower_bound=lower_bound,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    elif route in (
        _FLASH_KDA_ROUTE_HEAD_GROUPED_M128,
        _FLASH_KDA_ROUTE_LPT_M128,
        _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128,
    ):
        specialization = _flash_kda_generated_taskized_persistent_specialization(
            piece_tasks=route == _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    elif route == _FLASH_KDA_ROUTE_SMALL_BH_M128:
        specialization = _flash_kda_generated_small_bh_specialization(
            serving_native_abi=serving_native_abi,
            state_dtype_is_fp32=state_dtype_is_fp32,
        )
    else:
        raise ValueError(f"route {route!r} has no generated single-launch adapter")

    selector_key = _make_flash_kda_generated_selector_key(
        target=target,
        route=route,
        route_role=route_role,
        state_mode=state_mode,
        family_specialization=specialization,
    )
    from .jit.flash_kda import _GeneratedFlashKDASelectorNotFoundError

    try:
        metadata, module = _get_flash_kda_generated_module(selector_key)
    except _GeneratedFlashKDASelectorNotFoundError:
        return False
    signature_tensors = [q, k, v, g, beta_tma, out]
    if route == _FLASH_KDA_ROUTE_SMALL_BH_M128:
        assert packet_workspace is not None
        signature_tensors.append(packet_workspace)
    if checkpoint_every_n_tokens:
        signature_tensors.append(state_checkpoints)
    signature = tuple(
        _tensor_descriptor_signature(tensor) for tensor in signature_tensors
    )
    prepare_descriptors = _generated_descriptor_prepare(
        workspace=workspace,
        variant_id=metadata.variant_id,
        signature=signature,
        capturing=capturing,
    )
    descriptor_storage = _generated_descriptor_storage(
        workspace=workspace,
        variant_id=metadata.variant_id,
        device=q.device,
        bytes_required=(
            _FLASH_KDA_SEVEN_DESCRIPTOR_STORAGE_BYTES
            if route == _FLASH_KDA_ROUTE_SMALL_BH_M128 or checkpoint_every_n_tokens
            else _FLASH_KDA_DESCRIPTOR_STORAGE_BYTES
        ),
    )
    dummy_bf16 = _dummy_bf16(q.device)
    dummy_i64 = _dummy_i64(q.device)
    dummy_u32 = _dummy_u32(q.device)
    empty_u8 = _empty_cuda_tensor(q.device, torch.uint8)
    descriptor_u32 = descriptor_storage.view(torch.uint32)
    try:
        if route in (
            _FLASH_KDA_ROUTE_DIRECT_M128,
            _FLASH_KDA_ROUTE_DIRECT_M128_N16,
        ):
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                state_indices,
                initial_state,
                out,
                final_state,
                state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_cu_starts if checkpoint_every_n_tokens else dummy_i64,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                descriptor_u32 if checkpoint_every_n_tokens else dummy_u32,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                A_log,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                A_log,
                state_checkpoints if checkpoint_every_n_tokens else dummy_bf16,
                A_log,
                descriptor_u32 if checkpoint_every_n_tokens else dummy_u32,
                empty_u8,
                descriptor_storage,
                prepare_descriptors,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                checkpoint_every_n_tokens,
                0,
                len(sequence_lengths),
                scale,
                lower_bound,
                total_tasks,
                1,
                1,
                stream_ptr,
            )
        elif route == _FLASH_KDA_ROUTE_SOURCE_VTILE_M128:
            worker_count = (
                total_tasks
                if fixed_layout
                else _FLASH_KDA_SOURCE_VTILE_PERSISTENT_WORKERS
            )
            persistent_tasks = total_tasks // worker_count
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                state_indices,
                initial_state,
                out,
                final_state,
                descriptor_storage,
                prepare_descriptors,
                max(sequence_lengths),
                persistent_tasks,
                worker_count,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                scale,
                lower_bound,
                worker_count,
                1,
                1,
                stream_ptr,
            )
        elif route == _FLASH_KDA_ROUTE_M64:
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                state_indices,
                initial_state,
                out,
                final_state,
                descriptor_storage,
                prepare_descriptors,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                scale,
                lower_bound,
                2 * total_tasks,
                1,
                1,
                stream_ptr,
            )
        elif route == _FLASH_KDA_ROUTE_SCALAR_CHUNK_LPT_M128:
            tile_schedule, tile_schedule_counts, schedule_stride = (
                _generated_scalar_schedule(
                    workspace=workspace,
                    sequence_lengths=sequence_lengths,
                    num_heads=num_heads,
                    worker_count=sm_count,
                    device=q.device,
                )
            )
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                tile_schedule,
                tile_schedule_counts,
                state_indices,
                initial_state,
                out,
                final_state,
                descriptor_storage,
                prepare_descriptors,
                schedule_stride,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                scale,
                lower_bound,
                sm_count,
                1,
                1,
                stream_ptr,
            )
        elif route in (
            _FLASH_KDA_ROUTE_HEAD_GROUPED_M128,
            _FLASH_KDA_ROUTE_LPT_M128,
            _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128,
        ):
            assert persistent_task_ids is not None
            assert persistent_task_offsets is not None
            entry_count = persistent_task_ids.numel()
            if route == _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128:
                assert piece_task_token_starts is not None
                assert piece_task_token_counts is not None
                assert piece_task_state_sources is not None
                assert piece_task_state_destinations is not None
                assert piece_mid_state is not None
                assert piece_mid_state_ready is not None
                token_starts = piece_task_token_starts
                token_counts = piece_task_token_counts
                state_sources = piece_task_state_sources
                state_destinations = piece_task_state_destinations
                mid_state = piece_mid_state
                mid_state_ready = piece_mid_state_ready
            else:
                zero_values = (0,) * entry_count
                minus_one_values = (-1,) * entry_count
                token_starts = _cached_int32_metadata(
                    device=q.device,
                    kind="generated_task_token_starts",
                    values=zero_values,
                )
                token_counts = _cached_int32_metadata(
                    device=q.device,
                    kind="generated_task_token_counts",
                    values=zero_values,
                )
                state_sources = _cached_int32_metadata(
                    device=q.device,
                    kind="generated_task_state_sources",
                    values=minus_one_values,
                )
                state_destinations = _cached_int32_metadata(
                    device=q.device,
                    kind="generated_task_state_destinations",
                    values=minus_one_values,
                )
                mid_state = _cached_tensor(
                    (
                        "generated_task_mid_state",
                        *_stream_cache_key(q.device),
                    ),
                    lambda: torch.empty(
                        (1, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM),
                        dtype=torch.bfloat16,
                        device=q.device,
                    ),
                    capture_error=(
                        "generated taskized-persistent dummy state is not warmed "
                        "for CUDA graph capture"
                    ),
                )
                mid_state_ready = dummy_u32
            grid_x = persistent_task_offsets.numel() - 1
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                persistent_task_ids,
                persistent_task_offsets,
                token_starts,
                token_counts,
                state_sources,
                state_destinations,
                mid_state,
                mid_state_ready,
                state_indices,
                initial_state,
                out,
                final_state,
                descriptor_storage,
                prepare_descriptors,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                scale,
                lower_bound,
                grid_x,
                1,
                1,
                stream_ptr,
            )
        else:
            assert route == _FLASH_KDA_ROUTE_SMALL_BH_M128
            assert packet_workspace is not None
            assert packet_ready is not None
            assert packet_consumed is not None
            assert helper_done is not None
            module.run(
                q,
                k,
                v,
                g,
                beta,
                beta_tma,
                A_log,
                dt_bias,
                cu_seqlens,
                seq_order,
                state_indices,
                initial_state,
                out,
                final_state,
                state_checkpoints,
                checkpoint_cu_starts,
                packet_workspace,
                packet_ready,
                packet_consumed,
                helper_done,
                descriptor_storage,
                prepare_descriptors,
                num_heads,
                beta_token_stride,
                state_slot_stride,
                int(use_state_indices),
                int(use_initial_state),
                int(store_final_state),
                checkpoint_every_n_tokens,
                scale,
                lower_bound,
                8 * total_tasks,
                1,
                1,
                stream_ptr,
            )
    except Exception:
        _clear_generated_descriptor_signature(
            workspace=workspace,
            variant_id=metadata.variant_id,
            prepared=prepare_descriptors,
        )
        raise
    _record_generated_descriptor_signature(
        workspace=workspace,
        variant_id=metadata.variant_id,
        signature=signature,
        prepared=prepare_descriptors,
    )
    return True


def _affine_beta_tma_layout(
    beta: torch.Tensor,
) -> tuple[_AffineBetaTMALayout, torch.Tensor]:
    rows = beta.numel() // beta.shape[-1]
    num_heads = beta.shape[-1]
    flat = beta.reshape(rows, num_heads)
    if (
        num_heads == 12
        and rows % 2 == 0
        and flat.is_contiguous()
        and flat.data_ptr() % 16 == 0
    ):
        return "pair_packed", flat
    if (
        rows >= _FLASH_KDA_M128_CHUNK
        and num_heads >= _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        and flat.data_ptr() % 16 == 0
        and flat.stride(0) * beta.element_size() % 16 == 0
    ):
        return "direct", flat
    return "padded", flat


def _affine_beta_tma_from_plan(
    *,
    layout: _AffineBetaTMALayout,
    flat: torch.Tensor,
    padded: Optional[torch.Tensor],
) -> torch.Tensor:
    if layout == "pair_packed":
        return flat.view(flat.shape[0] // 2, 24)
    if layout == "direct":
        return flat
    assert padded is not None
    return padded


def _affine_padded_beta_tma(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    name: str,
    device: torch.device,
    rows: int,
    num_heads: int,
) -> torch.Tensor:
    padded_heads = (
        (num_heads + _FLASH_KDA_BETA_TMA_HEADS_PER_BOX - 1)
        // _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        * _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
    )
    return _affine_workspace_buffer(
        workspace=workspace,
        name=name,
        device=device,
        shape=(max(rows, _FLASH_KDA_M128_CHUNK), padded_heads),
        dtype=torch.bfloat16,
    )


@functools.cache
def _flash_kda_generated_affine_direct_selector_key(
    *,
    target: "FlashKDATarget",
    role: str,
    num_heads: int,
    num_sequences: int,
    uniform_sequences: bool,
    max_sequence_length: int,
    pair_packed_beta: bool,
    external_state_is_fp32: bool,
) -> dict[str, object]:
    state_dtype_is_fp32 = role != "affine_map"
    specialization = _flash_kda_generated_direct_specialization(
        target=target,
        route=_FLASH_KDA_ROUTE_AFFINE_M128,
        num_heads=num_heads,
        num_sequences=num_sequences,
        uniform_sequences=uniform_sequences,
        max_sequence_length=max_sequence_length,
        serving_native_abi=False,
        gate_kind="lower_bound",
        checkpoint_every_n_tokens=0,
        pair_packed_beta=pair_packed_beta,
        state_dtype_is_fp32=state_dtype_is_fp32,
        n32_ft_slab=True,
        pdl_wait_initial_state_f32=role
        in (
            "affine_map",
            "affine_correction",
        ),
        pdl_publish_final_state=role in ("affine_main", "affine_map"),
        affine_main_indexed_initial=role == "affine_main",
        affine_main_indexed_initial_bf16=(
            role == "affine_main" and not external_state_is_fp32
        ),
    )
    return _make_flash_kda_generated_selector_key(
        target=target,
        route=_FLASH_KDA_ROUTE_AFFINE_M128,
        route_role=role,
        state_mode=(
            "bf16"
            if role == "affine_map"
            else (
                "bf16_f32_dependency"
                if role == "affine_main" and not external_state_is_fp32
                else "fp32"
            )
        ),
        family_specialization=specialization,
    )


@functools.cache
def _flash_kda_generated_affine_scan_selector_key(
    *, target: "FlashKDATarget"
) -> dict[str, object]:
    return _make_flash_kda_generated_selector_key(
        target=target,
        route=_FLASH_KDA_ROUTE_AFFINE_M128,
        route_role="affine_scan",
        state_mode="none",
        family_specialization=_flash_kda_generated_affine_scan_specialization(),
    )


def _generated_affine_module_bundle(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    main_selector_key: dict[str, object],
    map_selector_key: dict[str, object],
    scan_selector_key: dict[str, object],
    correction_selector_key: dict[str, object],
    capturing: bool,
) -> _GeneratedAffineModuleBundle:
    cached = workspace._generated_affine_module_bundle
    if (
        cached is not None
        and cached.main.selector_key is main_selector_key
        and cached.map.selector_key is map_selector_key
        and cached.scan.selector_key is scan_selector_key
        and cached.correction.selector_key is correction_selector_key
    ):
        return cached
    if capturing:
        raise RuntimeError(
            "generated affine modules are not warmed for CUDA graph capture"
        )

    def resolve(role: str, selector_key: dict[str, object]):
        metadata, module = _get_flash_kda_generated_module(selector_key)
        return _GeneratedAffineModule(role, selector_key, metadata, module)

    resolved = _GeneratedAffineModuleBundle(
        main=resolve("affine_main", main_selector_key),
        map=resolve("affine_map", map_selector_key),
        scan=resolve("affine_scan", scan_selector_key),
        correction=resolve("affine_correction", correction_selector_key),
    )
    workspace._generated_affine_module_bundle = resolved
    return resolved


def _generated_affine_launch_plan(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    target: "FlashKDATarget",
    device: torch.device,
    token_offsets: tuple[int, ...],
    num_heads: int,
    state_dtype: torch.dtype,
    beta_layouts: tuple[
        _AffineBetaTMALayout,
        _AffineBetaTMALayout,
        _AffineBetaTMALayout,
    ],
    capturing: bool,
) -> _GeneratedAffineLaunchPlan:
    key = _GeneratedAffineLaunchPlanKey(
        target=target,
        token_offsets=token_offsets,
        num_heads=num_heads,
        state_dtype=state_dtype,
        beta_layouts=beta_layouts,
    )
    cached = workspace._generated_affine_launch_plan
    if cached is not None and cached.key == key:
        return cached
    if capturing:
        raise RuntimeError(
            "generated affine launch plan is not warmed for CUDA graph capture"
        )

    num_parts = len(token_offsets) - 1
    tail_start = token_offsets[1]
    total_tokens = token_offsets[-1]
    tail_tokens = total_tokens - tail_start
    state_shape = (num_parts, num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
    tail_state_shape = (
        num_parts - 1,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )
    main_lengths = tuple(
        right - left
        for left, right in zip(token_offsets, token_offsets[1:], strict=False)
    )
    tail_lengths = main_lengths[1:]
    tail_offsets = tuple(offset - tail_start for offset in token_offsets[1:])

    def buffer(
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        zero_on_allocate: bool = False,
    ) -> torch.Tensor:
        return _affine_workspace_buffer(
            workspace=workspace,
            name=name,
            device=device,
            shape=shape,
            dtype=dtype,
            zero_on_allocate=zero_on_allocate,
        )

    main_final = buffer("main_final_fp32", state_shape, torch.float32)
    map_identity = buffer(
        "map_identity_bfloat16",
        tail_state_shape,
        torch.bfloat16,
        zero_on_allocate=True,
    )
    if workspace._affine_map_identity_data_ptr != map_identity.data_ptr():
        map_identity.diagonal(dim1=-2, dim2=-1).fill_(1)
        workspace._affine_map_identity_data_ptr = map_identity.data_ptr()
    map_state = buffer("map_state_bfloat16", tail_state_shape, torch.bfloat16)
    carry = buffer("carry_float32", tail_state_shape, torch.float32)
    correction_final = buffer(
        "correction_final_float32", tail_state_shape, torch.float32
    )
    final_compact_shape = (1, num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
    final_compact = buffer("final_compact_float32", final_compact_shape, torch.float32)
    zero_v_shape = (1, tail_tokens, num_heads, _FLASH_KDA_HEAD_DIM)
    zero_v = buffer("zero_v", zero_v_shape, torch.bfloat16, zero_on_allocate=True)
    map_out = buffer("map_out", zero_v_shape, torch.bfloat16)
    correction_out = buffer("correction_out", zero_v_shape, torch.bfloat16)
    state_indices_i64 = buffer("state_indices_i64", (1,), torch.int64)
    final_external = (
        buffer("final_external", final_compact_shape, torch.bfloat16)
        if state_dtype == torch.bfloat16
        else None
    )

    def padded_beta(
        layout: _AffineBetaTMALayout, name: str, rows: int
    ) -> Optional[torch.Tensor]:
        if layout != "padded":
            return None
        return _affine_padded_beta_tma(
            workspace=workspace,
            name=name,
            device=device,
            rows=rows,
            num_heads=num_heads,
        )

    main_beta_padded = padded_beta(beta_layouts[0], "beta_tma_main", total_tokens)
    map_beta_padded = padded_beta(beta_layouts[1], "beta_tma_map", tail_tokens)
    correction_beta_padded = padded_beta(
        beta_layouts[2], "beta_tma_correction", tail_tokens
    )
    split_cu_seqlens = _cached_tensor(
        ("affine_split_cu", *_stream_cache_key(device), token_offsets),
        lambda: torch.tensor(token_offsets, dtype=torch.int64, device=device),
        capture_error="affine split offsets are not warmed for CUDA graph capture",
    )
    tail_cu_seqlens = _cached_tensor(
        ("affine_tail_cu", *_stream_cache_key(device), tail_offsets),
        lambda: torch.tensor(tail_offsets, dtype=torch.int64, device=device),
        capture_error="affine tail offsets are not warmed for CUDA graph capture",
    )
    main_seq_order = _identity_seq_order(device=device, num_sequences=num_parts)
    tail_seq_order = _identity_seq_order(device=device, num_sequences=num_parts - 1)

    main_selector_key = _flash_kda_generated_affine_direct_selector_key(
        target=target,
        role="affine_main",
        num_heads=num_heads,
        num_sequences=len(main_lengths),
        uniform_sequences=len(set(main_lengths)) == 1,
        max_sequence_length=max(main_lengths),
        pair_packed_beta=beta_layouts[0] == "pair_packed",
        external_state_is_fp32=state_dtype == torch.float32,
    )
    map_selector_key = _flash_kda_generated_affine_direct_selector_key(
        target=target,
        role="affine_map",
        num_heads=num_heads,
        num_sequences=len(tail_lengths),
        uniform_sequences=len(set(tail_lengths)) == 1,
        max_sequence_length=max(tail_lengths),
        pair_packed_beta=beta_layouts[1] == "pair_packed",
        external_state_is_fp32=False,
    )
    scan_selector_key = _flash_kda_generated_affine_scan_selector_key(target=target)
    correction_selector_key = _flash_kda_generated_affine_direct_selector_key(
        target=target,
        role="affine_correction",
        num_heads=num_heads,
        num_sequences=len(tail_lengths),
        uniform_sequences=len(set(tail_lengths)) == 1,
        max_sequence_length=max(tail_lengths),
        pair_packed_beta=beta_layouts[2] == "pair_packed",
        external_state_is_fp32=True,
    )
    modules = _generated_affine_module_bundle(
        workspace=workspace,
        main_selector_key=main_selector_key,
        map_selector_key=map_selector_key,
        scan_selector_key=scan_selector_key,
        correction_selector_key=correction_selector_key,
        capturing=False,
    )
    resolved = _GeneratedAffineLaunchPlan(
        key=key,
        num_parts=num_parts,
        tail_start=tail_start,
        total_tokens=total_tokens,
        tail_tokens=tail_tokens,
        main_lengths=main_lengths,
        tail_lengths=tail_lengths,
        main_final=main_final,
        map_identity=map_identity,
        map_state=map_state,
        carry=carry,
        correction_final=correction_final,
        final_compact=final_compact,
        zero_v=zero_v,
        map_out=map_out,
        correction_out=correction_out,
        state_indices_i64=state_indices_i64,
        final_external=final_external,
        main_beta_padded=main_beta_padded,
        map_beta_padded=map_beta_padded,
        correction_beta_padded=correction_beta_padded,
        split_cu_seqlens=split_cu_seqlens,
        tail_cu_seqlens=tail_cu_seqlens,
        main_seq_order=main_seq_order,
        tail_seq_order=tail_seq_order,
        main_descriptor_storage=_affine_descriptor_storage(
            workspace=workspace, role="main", device=device
        ),
        map_descriptor_storage=_affine_descriptor_storage(
            workspace=workspace, role="map", device=device
        ),
        correction_descriptor_storage=_affine_descriptor_storage(
            workspace=workspace, role="correction", device=device
        ),
        modules=modules,
    )
    workspace._generated_affine_launch_plan = resolved
    return resolved


def _run_generated_affine_direct_role(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    carriers: _GeneratedAffineCarriers,
    resolved_module: _GeneratedAffineModule,
    descriptor_storage: torch.Tensor,
    launch_observer: Optional[_GeneratedAffineLaunchObserver],
    role: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    beta_tma: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_order: torch.Tensor,
    state_indices: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    initial_state_f32_dependency: torch.Tensor,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    use_state_indices: bool,
    state_slot_stride: int,
    scale: float,
    lower_bound: float,
    grid_x: int,
    stream_ptr: int,
    capturing: bool,
) -> None:
    num_sequences = cu_seqlens.numel() - 1
    if len(sequence_lengths) != num_sequences:
        raise ValueError("affine role lengths must match its sequence offsets")
    metadata = resolved_module.metadata
    module = resolved_module.module
    signature = tuple(
        _tensor_descriptor_signature(tensor) for tensor in (q, k, v, g, beta_tma, out)
    )
    signature_key = f"affine:{role}:{metadata.variant_id}"
    warmed_signature = workspace._descriptor_signatures.get(signature_key)
    if capturing and warmed_signature != signature:
        raise RuntimeError(
            f"affine {role} descriptors are not warmed for CUDA graph capture"
        )
    prepare_descriptors = 0 if capturing else int(warmed_signature != signature)
    empty_state = (
        carriers.empty_f32
        if initial_state.dtype == torch.float32
        else carriers.empty_bf16
    )
    try:
        module = _generated_affine_module_for_launch(resolved_module, launch_observer)
        module.run(
            q,
            k,
            v,
            g,
            beta,
            beta_tma,
            A_log,
            dt_bias,
            cu_seqlens,
            seq_order,
            state_indices,
            initial_state,
            out,
            final_state,
            empty_state,
            carriers.empty_i64,
            carriers.dummy_i64,
            carriers.dummy_bf16,
            carriers.dummy_u32,
            carriers.dummy_bf16,
            carriers.dummy_bf16,
            carriers.dummy_bf16,
            carriers.dummy_bf16,
            initial_state_f32_dependency,
            carriers.dummy_bf16,
            carriers.dummy_bf16,
            carriers.dummy_bf16,
            carriers.dummy_f32,
            carriers.dummy_bf16,
            carriers.dummy_f32,
            carriers.dummy_u32,
            carriers.empty_u8,
            descriptor_storage,
            prepare_descriptors,
            num_heads,
            beta.stride(-2),
            state_slot_stride,
            int(use_state_indices),
            1,
            1,
            0,
            0,
            num_sequences,
            scale,
            lower_bound,
            grid_x,
            1,
            1,
            stream_ptr,
        )
    except Exception:
        if prepare_descriptors:
            workspace._descriptor_signatures.pop(signature_key, None)
        raise
    if prepare_descriptors:
        workspace._descriptor_signatures[signature_key] = signature


def _run_generated_affine_route(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    target: "FlashKDATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    token_offsets: tuple[int, ...],
    num_heads: int,
    scale: float,
    lower_bound: float,
    stream_ptr: int,
    capturing: bool,
) -> None:
    device = q.device
    carriers = _generated_affine_carriers(workspace=workspace, device=device)
    total_tokens = q.numel() // (num_heads * _FLASH_KDA_HEAD_DIM)
    q_flat = q.reshape(total_tokens, num_heads, _FLASH_KDA_HEAD_DIM)
    k_flat = k.reshape_as(q_flat)
    g_flat = g.reshape_as(q_flat)
    main_beta_layout, beta_flat = _affine_beta_tma_layout(beta)
    out_flat = out.reshape_as(q_flat)
    tail_start = token_offsets[1]
    tail_tokens = total_tokens - tail_start
    q_tail = q_flat[tail_start:].view(1, tail_tokens, num_heads, _FLASH_KDA_HEAD_DIM)
    k_tail = k_flat[tail_start:].view_as(q_tail)
    g_tail = g_flat[tail_start:].view_as(q_tail)
    beta_tail = beta_flat[tail_start:].view(1, tail_tokens, num_heads)
    map_beta_layout, map_beta_flat = _affine_beta_tma_layout(beta_tail)
    correction_beta_layout = map_beta_layout
    plan = _generated_affine_launch_plan(
        workspace=workspace,
        target=target,
        device=device,
        token_offsets=token_offsets,
        num_heads=num_heads,
        state_dtype=initial_state.dtype,
        beta_layouts=(
            main_beta_layout,
            map_beta_layout,
            correction_beta_layout,
        ),
        capturing=capturing,
    )
    main_beta_tma = _affine_beta_tma_from_plan(
        layout=main_beta_layout,
        flat=beta_flat,
        padded=plan.main_beta_padded,
    )
    map_beta_tma = _affine_beta_tma_from_plan(
        layout=map_beta_layout,
        flat=map_beta_flat,
        padded=plan.map_beta_padded,
    )
    correction_beta_tma = _affine_beta_tma_from_plan(
        layout=correction_beta_layout,
        flat=map_beta_flat,
        padded=plan.correction_beta_padded,
    )
    main_initial_arg = initial_state
    main_use_indices = True
    main_state_stride = initial_state.stride(0)
    launch_observer = _generated_affine_launch_observer.get()

    _run_generated_affine_direct_role(
        workspace=workspace,
        carriers=carriers,
        resolved_module=plan.modules.main,
        descriptor_storage=plan.main_descriptor_storage,
        launch_observer=launch_observer,
        role="affine_main",
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        beta_tma=main_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.split_cu_seqlens,
        seq_order=plan.main_seq_order,
        state_indices=(state_indices if main_use_indices else carriers.dummy_i32),
        initial_state=main_initial_arg,
        out=out,
        final_state=plan.main_final,
        initial_state_f32_dependency=carriers.dummy_f32,
        sequence_lengths=plan.main_lengths,
        num_heads=num_heads,
        use_state_indices=main_use_indices,
        state_slot_stride=main_state_stride,
        scale=scale,
        lower_bound=lower_bound,
        grid_x=plan.num_parts * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    _run_generated_affine_direct_role(
        workspace=workspace,
        carriers=carriers,
        resolved_module=plan.modules.map,
        descriptor_storage=plan.map_descriptor_storage,
        launch_observer=launch_observer,
        role="affine_map",
        q=q_tail,
        k=k_tail,
        v=plan.zero_v,
        g=g_tail,
        beta=beta_tail,
        beta_tma=map_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.tail_cu_seqlens,
        seq_order=plan.tail_seq_order,
        state_indices=carriers.dummy_i32,
        initial_state=plan.map_identity,
        out=plan.map_out,
        final_state=plan.map_state,
        initial_state_f32_dependency=plan.main_final,
        sequence_lengths=plan.tail_lengths,
        num_heads=num_heads,
        use_state_indices=False,
        state_slot_stride=num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM,
        scale=scale,
        lower_bound=lower_bound,
        grid_x=(plan.num_parts - 1) * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    scan_module = _generated_affine_module_for_launch(
        plan.modules.scan, launch_observer
    )
    scan_module.run(
        plan.main_final,
        plan.map_state,
        plan.carry,
        num_heads,
        plan.num_parts,
        32 * num_heads,
        1,
        1,
        stream_ptr,
    )
    _run_generated_affine_direct_role(
        workspace=workspace,
        carriers=carriers,
        resolved_module=plan.modules.correction,
        descriptor_storage=plan.correction_descriptor_storage,
        launch_observer=launch_observer,
        role="affine_correction",
        q=q_tail,
        k=k_tail,
        v=plan.zero_v,
        g=g_tail,
        beta=beta_tail,
        beta_tma=correction_beta_tma,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=plan.tail_cu_seqlens,
        seq_order=plan.tail_seq_order,
        state_indices=carriers.dummy_i32,
        initial_state=plan.carry,
        out=plan.correction_out,
        final_state=plan.correction_final,
        initial_state_f32_dependency=plan.carry,
        sequence_lengths=plan.tail_lengths,
        num_heads=num_heads,
        use_state_indices=False,
        state_slot_stride=num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM,
        scale=scale,
        lower_bound=lower_bound,
        grid_x=(plan.num_parts - 1) * num_heads,
        stream_ptr=stream_ptr,
        capturing=capturing,
    )
    out_flat[tail_start:].add_(plan.correction_out.reshape_as(out_flat[tail_start:]))
    torch.add(
        plan.main_final[-1:],
        plan.correction_final[-1:],
        out=plan.final_compact,
    )
    if final_state.dtype == torch.bfloat16:
        assert plan.final_external is not None
        plan.final_external.copy_(plan.final_compact)
        plan.state_indices_i64.copy_(state_indices)
        final_state.index_copy_(0, plan.state_indices_i64, plan.final_external)
    else:
        plan.state_indices_i64.copy_(state_indices)
        final_state.index_copy_(0, plan.state_indices_i64, plan.final_compact)


def _get_cake_kda_prefill_module(variant: "CakeKDAVariant", target: "CakeKDATarget"):
    from .jit.cake_kda import get_cake_kda_prefill_module

    return get_cake_kda_prefill_module(variant, target)


def _run_flash_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
    state_indices: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    backend: Literal["cake"] = "cake",
    final_state: Optional[torch.Tensor] = None,
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
):
    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of recurrent_kda prefill requires an explicit "
            "RecurrentKDAPrefillWorkspace warmed with the exact tensors on "
            "the capture stream"
        )
    batch_size, seq_len, num_heads, _ = q.shape
    fixed_layout = cu_seqlens is None
    num_sequences = batch_size if fixed_layout else cu_seqlens.numel() - 1
    flash_target = (
        None if lower_bound is None else _select_flash_kda_prefill_target(q.device)
    )
    compute_capability = get_compute_capability(q.device)
    sm_count = _flash_kda_device_sm_count(q.device)
    stream_workspace = (
        _get_stream_workspace(q.device) if prefill_workspace is None else None
    )
    metadata_workspace: _RecurrentKDAPrefillWorkspaceBase
    if prefill_workspace is None:
        assert stream_workspace is not None
        metadata_workspace = stream_workspace
    else:
        metadata_workspace = prefill_workspace
    # Only contracts that materially require the direct ABI are forced here.
    # Indexed/strided and FP32 state are physical-selector fields supported by
    # several generated families and must remain visible to the source route
    # policy instead of being collapsed to one legacy module.
    needs_direct_m128 = checkpoint_every_n_tokens != 0 or not beta.is_contiguous()
    legacy_persistent_candidate = (
        _uses_measured_sm100_persistent_policy(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        and lower_bound is not None
        and not needs_direct_m128
        and prefill_workspace is None
        and seq_order is None
        and initial_state is not None
        and num_heads != 12
        and num_sequences * num_heads > sm_count
    )
    piece_persistent_candidate = (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and prefill_workspace is None
        and seq_order is None
        and initial_state is not None
        and num_heads != 12
        and num_sequences * num_heads > sm_count
    )
    automatic_sequence_order = None
    persistent_plan = None
    cu_seqlens_i64: Optional[torch.Tensor]
    if fixed_layout:
        sequence_lengths = (seq_len,) * num_sequences
        offsets = tuple(index * seq_len for index in range(num_sequences + 1))
        uniform_sequences = True
        cu_seqlens_i64 = None
    else:
        assert cu_seqlens is not None
        if cu_seqlens.dtype == torch.int32 and capturing:
            raise RuntimeError(
                "packed recurrent_kda prefill requires int64 cu_seqlens "
                "during CUDA graph capture; convert it before capture"
            )
        cu_seqlens_i64 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int64
            else cu_seqlens.to(torch.int64)
        )
        (
            automatic_sequence_order,
            persistent_plan,
            uniform_sequences,
            offsets,
            sequence_lengths,
        ) = _cached_packed_task_metadata(
            metadata_workspace,
            cu_seqlens_i64,
            total_tokens=batch_size * seq_len,
            num_heads=num_heads,
            sm_count=sm_count,
            build_persistent_plan=legacy_persistent_candidate,
        )
    if fixed_layout and legacy_persistent_candidate:
        persistent_plan = _persistent_task_plan(
            sequence_lengths,
            num_heads=num_heads,
            sm_count=sm_count,
        )
    max_sequence_length = max(sequence_lengths)
    affine_token_offsets = None
    if (
        lower_bound is not None
        and fixed_layout
        and batch_size == 1
        and state_indices is not None
        and initial_state is not None
        and beta.is_contiguous()
        and checkpoint_every_n_tokens == 0
        and state_checkpoints is None
        and checkpoint_cu_starts is None
    ):
        affine_token_offsets = _flash_kda_affine_token_offsets(
            total_tokens=batch_size * seq_len,
            num_heads=num_heads,
            sm_count=sm_count,
            state_dtype=initial_state.dtype,
        )
    affine_plan = None
    if lower_bound is None and seq_order is None:
        affine_plan = _select_cake_kda_affine_plan(
            export_available=_cake_kda_affine_export_is_available(),
            compute_capability=compute_capability,
            sm_count=sm_count,
            fixed_layout=fixed_layout,
            batch_size=batch_size,
            total_tokens=batch_size * seq_len,
            num_heads=num_heads,
            head_dim=q.shape[-1],
            qkv_shapes_equal=k.shape == q.shape == v.shape,
            qkv_dtype=q.dtype,
            beta_contiguous=beta.is_contiguous(),
            beta_dtype=beta.dtype,
            indexed_state=state_indices is not None,
            initial_state_dtype=(
                initial_state.dtype if initial_state is not None else None
            ),
            has_checkpoints=(
                checkpoint_every_n_tokens != 0
                or state_checkpoints is not None
                or checkpoint_cu_starts is not None
            ),
            lower_bound=lower_bound,
        )
    use_exact_n16 = (
        checkpoint_every_n_tokens != 0 and checkpoint_every_n_tokens % 32 != 0
    ) or _requires_exact_n16_recurrence(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
    )
    if affine_token_offsets is not None:
        route = _FLASH_KDA_ROUTE_AFFINE_M128
        persistent_plan = None
    elif affine_plan is not None:
        route = _CAKE_KDA_ROUTE_AFFINE_M128
        persistent_plan = None
    elif needs_direct_m128:
        route = (
            _FLASH_KDA_ROUTE_DIRECT_M128_N16
            if use_exact_n16
            else _direct_m128_route(
                num_heads=num_heads,
                max_sequence_length=max_sequence_length,
            )
        )
    else:
        route = select_bf16_schedule_route(
            compute_capability=compute_capability,
            sm_count=sm_count,
            fixed_layout=fixed_layout,
            sequence_lengths=sequence_lengths,
            num_heads=num_heads,
            use_initial_state=initial_state is not None,
            store_final_state=initial_state is not None or output_final_state,
        )
        # The receipt portfolio has no serving-native BF16 BT16-chain or
        # small-BH module.  Indexed BF16 calls therefore stay on the exact
        # direct family unless the affine split route is selected below.
        if (
            state_indices is not None
            and initial_state is not None
            and initial_state.dtype == torch.bfloat16
            and route in (_FLASH_KDA_ROUTE_BT16_M64, _FLASH_KDA_ROUTE_SMALL_BH_M128)
        ):
            route = _direct_m128_route(
                num_heads=num_heads,
                max_sequence_length=max_sequence_length,
            )
    if (
        route == _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128
        and not piece_persistent_candidate
    ):
        route = _direct_m128_route(
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )
    if (
        route
        in (
            _FLASH_KDA_ROUTE_HEAD_GROUPED_M128,
            _FLASH_KDA_ROUTE_LPT_M128,
        )
        and persistent_plan is None
    ):
        # Taskized routes require the exact host-resolved bin plan.  CUDA
        # graph workspaces intentionally avoid the host synchronization
        # needed to create it and retain a direct generated route.
        route = _direct_m128_route(
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )
    use_bt16 = route == _FLASH_KDA_ROUTE_BT16_M64
    use_tensor_state_decay = (
        state_indices is None
        and checkpoint_every_n_tokens == 0
        and _should_use_n32_tensor_state_decay(
            compute_capability=compute_capability,
            route=route,
            uniform_sequences=uniform_sequences,
            num_heads=num_heads,
            total_tasks=num_sequences * num_heads,
            max_sequence_length=max_sequence_length,
        )
    )
    if route in (
        _FLASH_KDA_ROUTE_DIRECT_M128_N16,
        _FLASH_KDA_ROUTE_BT16_M64,
        _FLASH_KDA_ROUTE_M64,
        _FLASH_KDA_ROUTE_SMALL_BH_M128,
    ):
        persistent_plan = None
    variant: Literal[
        "bt16",
        "m64",
        "m128",
        "m128_tensor_state_decay",
        "m128_h12_short",
        "m128_h12_long",
        "m128_n16",
        "m128_n16_checkpoint",
        "m128_n16_short",
        "persistent_m128",
        "piece_persistent_m128",
        "small_bh_m128",
        "cake_affine_m128",
        "m128_unbounded_softplus",
        "m128_bt64_unbounded_softplus",
    ]
    if affine_plan is not None:
        variant = "cake_affine_m128"
    elif use_bt16:
        variant = "bt16"
    elif route == _FLASH_KDA_ROUTE_M64:
        variant = "m64"
    elif route == _FLASH_KDA_ROUTE_SMALL_BH_M128:
        variant = "small_bh_m128"
    elif use_tensor_state_decay:
        variant = "m128_tensor_state_decay"
    elif route == _FLASH_KDA_ROUTE_PIECE_PERSISTENT_M128:
        variant = "piece_persistent_m128"
    elif persistent_plan is not None:
        variant = "persistent_m128"
    elif route == _FLASH_KDA_ROUTE_DIRECT_M128_N16:
        variant = (
            "m128_n16_short"
            if num_heads != 12
            and 0 < max_sequence_length <= _FLASH_KDA_BT16_CHUNK
            and checkpoint_every_n_tokens == 0
            else "m128_n16"
        )
    elif num_heads == 12:
        variant = (
            "m128_h12_short"
            if max_sequence_length
            <= _FLASH_KDA_H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQUENCE_LENGTH
            else "m128_h12_long"
        )
    else:
        variant = "m128"
    if lower_bound is None and affine_plan is None:
        variant = (
            "m128_bt64_unbounded_softplus"
            if num_heads == 4
            and checkpoint_every_n_tokens > 0
            and checkpoint_every_n_tokens % 64 == 0
            else "m128_unbounded_softplus"
        )
        persistent_plan = None
    if checkpoint_every_n_tokens and variant == "m128_n16":
        variant = "m128_n16_checkpoint"
    piece_plan = None
    if variant == "piece_persistent_m128":
        piece_roofline = _persistent_m128_roofline(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_sequences=num_sequences,
            num_heads=num_heads,
            sequence_length=max_sequence_length,
            use_initial_state=initial_state is not None,
            store_final_state=initial_state is not None or output_final_state,
        )
        if (
            piece_roofline is None
            or piece_roofline.piece_ns >= piece_roofline.direct_ns
        ):
            raise RuntimeError(
                "piece-persistent route selected without a resolved roofline advantage"
            )
        piece_plan = _make_uniform_piece_task_bins(
            num_sequences=num_sequences,
            num_heads=num_heads,
            sequence_length=max_sequence_length,
            worker_count=piece_roofline.worker_count,
        )
        persistent_plan = None
    persistent_task_ids = None
    persistent_task_offsets = None
    seq_order_i32: Optional[torch.Tensor] = None
    if route == _FLASH_KDA_ROUTE_AFFINE_M128:
        pass
    elif persistent_plan is None:
        if seq_order is not None or automatic_sequence_order is None:
            seq_order_i32 = _validate_prefill_seq_order(
                seq_order,
                fixed_layout=fixed_layout,
                num_sequences=num_sequences,
                device=q.device,
            )
        else:
            seq_order_i32 = _cached_int32_metadata(
                device=q.device,
                kind="automatic_seq_order",
                values=automatic_sequence_order,
            )
    else:
        sequence_order, task_ids, task_offsets = persistent_plan
        seq_order_i32 = _cached_int32_metadata(
            device=q.device,
            kind="persistent_seq_order",
            values=sequence_order,
        )
        persistent_task_ids = _cached_int32_metadata(
            device=q.device,
            kind="persistent_task_ids",
            values=task_ids,
        )
        persistent_task_offsets = _cached_int32_metadata(
            device=q.device,
            kind="persistent_task_offsets",
            values=task_offsets,
        )
    piece_task_token_starts = None
    piece_task_token_counts = None
    piece_task_state_sources = None
    piece_task_state_destinations = None
    piece_handoff_count = 0
    if piece_plan is not None:
        (
            task_ids,
            task_offsets,
            token_starts,
            token_counts,
            state_sources,
            state_destinations,
            piece_handoff_count,
            _piece_loads,
        ) = piece_plan
        persistent_task_ids = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_ids",
            values=task_ids,
        )
        persistent_task_offsets = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_offsets",
            values=task_offsets,
        )
        piece_task_token_starts = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_token_starts",
            values=token_starts,
        )
        piece_task_token_counts = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_token_counts",
            values=token_counts,
        )
        piece_task_state_sources = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_state_sources",
            values=state_sources,
        )
        piece_task_state_destinations = _cached_int32_metadata(
            device=q.device,
            kind="piece_persistent_task_state_destinations",
            values=state_destinations,
        )
    if route == _FLASH_KDA_ROUTE_AFFINE_M128:
        dummy_state = None
        dummy_i32 = None
        dummy_i64 = None
    else:
        dummy_state = _dummy_bf16(q.device)
        dummy_i32 = _dummy_i32(q.device) if variant != "m64" else None
        dummy_i64 = _dummy_i64(q.device) if variant != "m64" else None

    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor for "
                "recurrent_kda prefill"
            )
        out_buf = torch.empty_like(q)
    else:
        out_buf = output
    _check_output_does_not_overlap_inputs(
        out_buf,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )

    state_shape = (
        num_sequences,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )
    use_initial_state = initial_state is not None
    use_state_indices = state_indices is not None
    if use_state_indices and initial_state is None:
        raise ValueError("state_indices requires an initial_state pool")
    if final_state is not None and (initial_state is None or not output_final_state):
        raise ValueError(
            "a separate final_state requires initial_state and output_final_state=True"
        )
    if initial_state is not None:
        initial_state_arg = initial_state
        final_state_arg = initial_state if final_state is None else final_state
        store_final_state = True
        returned_state = final_state_arg
    elif output_final_state:
        assert dummy_state is not None
        initial_state_arg = dummy_state
        if prefill_workspace is None:
            final_state_arg = torch.empty(
                state_shape, dtype=torch.bfloat16, device=q.device
            )
            returned_state = final_state_arg
        else:
            # Assigned to caller-owned stable state scratch under its lock.
            returned_state = None
        store_final_state = True
    else:
        assert dummy_state is not None
        initial_state_arg = dummy_state
        final_state_arg = dummy_state
        store_final_state = False
        returned_state = None
    state_slot_stride = (
        initial_state.stride(0)
        if initial_state is not None and initial_state.ndim == 4
        else num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
    )

    scale_value = _FLASH_KDA_DEFAULT_SCALE if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
    explicit_workspace = prefill_workspace is not None
    workspace: _RecurrentKDAPrefillWorkspaceBase
    if prefill_workspace is None:
        assert stream_workspace is not None
        workspace = stream_workspace
    else:
        workspace = prefill_workspace
    # TVM FFI may release the GIL. Serialize the complete shared-workspace
    # enqueue sequence so two host threads cannot interleave preparation or
    # launch on the same CUDA stream.
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=capturing,
            explicit=explicit_workspace,
        )
        if route == _FLASH_KDA_ROUTE_AFFINE_M128:
            assert affine_token_offsets is not None
            assert flash_target is not None
            assert lower_bound is not None
            assert state_indices is not None
            _run_generated_affine_route(
                workspace=workspace,
                target=flash_target,
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=initial_state_arg,
                final_state=final_state_arg,
                state_indices=state_indices,
                out=out_buf,
                token_offsets=affine_token_offsets,
                num_heads=num_heads,
                scale=scale_value,
                lower_bound=float(lower_bound),
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
            if capturing and explicit_workspace:
                workspace._captured = True
            return (out_buf, returned_state if output_final_state else None)
        if route == _CAKE_KDA_ROUTE_AFFINE_M128:
            assert affine_plan is not None
            assert initial_state is not None
            assert state_indices is not None
            _run_cake_kda_affine_route(
                workspace=workspace,
                affine_plan=affine_plan,
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=initial_state_arg,
                final_state=final_state_arg,
                state_indices=state_indices,
                out=out_buf,
                num_heads=num_heads,
                scale=scale_value,
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
            if capturing and explicit_workspace:
                workspace._captured = True
            return (out_buf, returned_state if output_final_state else None)
        assert variant != "cake_affine_m128"
        if cu_seqlens_i64 is None:
            cu_seqlens_i64 = _fixed_cu_seqlens(
                device=q.device, batch_size=batch_size, seq_len=seq_len
            )
        assert seq_order_i32 is not None
        assert dummy_state is not None
        if variant == "m128_h12_long":
            pair_packed_beta_tma = _pair_packed_beta_tma_source(beta)
            if pair_packed_beta_tma is None:
                # Preserve the general public route for accepted layouts that
                # cannot expose the source runtime's zero-copy H12 carrier.
                variant = "m128"
                beta_tma = _beta_tma_source(beta, workspace, chunk_tokens=32)
            else:
                beta_tma = pair_packed_beta_tma
        else:
            beta_tma = _beta_tma_source(
                beta,
                workspace,
                chunk_tokens=(
                    16
                    if variant
                    in (
                        "m128_n16",
                        "m128_n16_checkpoint",
                        "m128_n16_short",
                    )
                    else 32
                ),
            )
        packet_workspace = None
        packet_ready = None
        packet_consumed = None
        helper_done = None
        piece_mid_state = None
        piece_mid_state_ready = None
        if variant == "small_bh_m128":
            (
                packet_workspace,
                packet_ready,
                packet_consumed,
                helper_done,
            ) = _small_bh_workspace(
                workspace=workspace,
                device=q.device,
                total_tasks=num_sequences * num_heads,
            )
        elif variant == "piece_persistent_m128":
            piece_mid_state, piece_mid_state_ready = _piece_persistent_workspace(
                workspace=workspace,
                device=q.device,
                handoff_count=piece_handoff_count,
            )
        if initial_state is None and output_final_state and explicit_workspace:
            final_state_arg = _state_scratch(
                workspace=workspace,
                device=q.device,
                shape=state_shape,
            )
            if initial_state is None:
                returned_state = final_state_arg
        if variant == "bt16":
            assert flash_target is not None
            assert lower_bound is not None
            _run_generated_bt16_prepare_chain(
                workspace=workspace,
                target=flash_target,
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens_i64,
                seq_order=seq_order_i32,
                state_indices=(
                    state_indices if state_indices is not None else _dummy_i32(q.device)
                ),
                initial_state=initial_state_arg,
                out=out_buf,
                final_state=final_state_arg,
                offsets=offsets,
                num_heads=num_heads,
                sm_count=sm_count,
                compute_capability=compute_capability,
                fixed_layout=fixed_layout,
                max_sequence_length=max_sequence_length,
                use_initial_state=use_initial_state,
                store_final_state=store_final_state,
                use_state_indices=use_state_indices,
                state_slot_stride=state_slot_stride,
                state_dtype_is_fp32=(
                    initial_state is not None and initial_state.dtype == torch.float32
                ),
                scale=scale_value,
                lower_bound=float(lower_bound),
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
            if capturing and explicit_workspace:
                workspace._captured = True
            return (out_buf, returned_state if output_final_state else None)
        generated_state_dtype = (
            initial_state.dtype if initial_state is not None else torch.bfloat16
        )
        generated_state_checkpoints = (
            state_checkpoints
            if state_checkpoints is not None
            else _empty_cuda_tensor(q.device, generated_state_dtype)
        )
        generated_checkpoint_cu_starts = (
            checkpoint_cu_starts
            if checkpoint_cu_starts is not None
            else _empty_cuda_tensor(q.device, torch.int64)
        )
        generated_route_launched = (
            _run_generated_single_route(
                workspace=workspace,
                target=flash_target,
                route=route,
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                beta_tma=beta_tma,
                A_log=A_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens_i64,
                seq_order=seq_order_i32,
                state_indices=(
                    state_indices if state_indices is not None else _dummy_i32(q.device)
                ),
                initial_state=initial_state_arg,
                out=out_buf,
                final_state=final_state_arg,
                state_checkpoints=generated_state_checkpoints,
                checkpoint_cu_starts=generated_checkpoint_cu_starts,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                sequence_lengths=sequence_lengths,
                fixed_layout=fixed_layout,
                uniform_sequences=uniform_sequences,
                num_heads=num_heads,
                sm_count=sm_count,
                use_state_indices=use_state_indices,
                use_initial_state=use_initial_state,
                store_final_state=store_final_state,
                state_slot_stride=state_slot_stride,
                state_dtype_is_fp32=(
                    initial_state is not None and initial_state.dtype == torch.float32
                ),
                scale=scale_value,
                lower_bound=float(lower_bound),
                persistent_task_ids=persistent_task_ids,
                persistent_task_offsets=persistent_task_offsets,
                piece_task_token_starts=piece_task_token_starts,
                piece_task_token_counts=piece_task_token_counts,
                piece_task_state_sources=piece_task_state_sources,
                piece_task_state_destinations=piece_task_state_destinations,
                piece_mid_state=piece_mid_state,
                piece_mid_state_ready=piece_mid_state_ready,
                packet_workspace=packet_workspace,
                packet_ready=packet_ready,
                packet_consumed=packet_consumed,
                helper_done=helper_done,
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
            if flash_target is not None
            else False
        )
        if generated_route_launched:
            if capturing and explicit_workspace:
                workspace._captured = True
            result = (out_buf, returned_state if output_final_state else None)
            if checkpoint_every_n_tokens:
                assert state_checkpoints is not None
                return (*result, state_checkpoints)
            return result

        if variant == "m64" and num_heads != 64:
            # The legacy M64 ABI is specialized for exactly 64 heads. If an
            # otherwise valid small-head M64 schedule is outside the sealed
            # generated selector portfolio, use the established generic ABI.
            variant = "m128"
            dummy_i32 = _dummy_i32(q.device)
            dummy_i64 = _dummy_i64(q.device)

        # Retain the established public kernel for valid runtime shapes that
        # are outside the exact selector portfolio. Receipt-backed selectors
        # always return above; malformed or conflicting receipts still fail.
        signature = _descriptor_signature(
            q=q,
            k=k,
            v=v,
            g=g,
            beta_tma=beta_tma,
            out=out_buf,
            packet_workspace=packet_workspace,
            state_checkpoints=(
                state_checkpoints if variant == "m128_n16_checkpoint" else None
            ),
        )
        warmed_signature = workspace._descriptor_signatures.get(variant)
        if capturing:
            if warmed_signature != signature:
                raise RuntimeError(
                    "RecurrentKDAPrefillWorkspace is not warmed for the exact "
                    f"{variant} descriptor signature; eagerly invoke the same "
                    "call on this stream before capture"
                )
            prepare_descriptors = 0
        else:
            prepare_descriptors = int(warmed_signature != signature)
        descriptor_storage = workspace._descriptor_storages[variant]
        if (
            variant == "m128_unbounded_softplus"
            or variant == "m128_bt64_unbounded_softplus"
        ):
            module = _get_cake_kda_prefill_module(
                variant, _select_cake_kda_prefill_target(q.device)
            )
        else:
            assert flash_target is not None
            module = _get_flash_kda_prefill_module(variant, flash_target)
        try:
            if variant == "m64":
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound if lower_bound is not None else 0.0),
                    stream_ptr,
                )
            elif variant == "small_bh_m128":
                assert packet_workspace is not None
                assert packet_ready is not None
                assert packet_consumed is not None
                assert helper_done is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    packet_workspace,
                    packet_ready,
                    packet_consumed,
                    helper_done,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            elif variant == "persistent_m128":
                assert persistent_task_ids is not None
                assert persistent_task_offsets is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    persistent_task_ids,
                    persistent_task_offsets,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            elif variant == "piece_persistent_m128":
                assert persistent_task_ids is not None
                assert persistent_task_offsets is not None
                assert piece_task_token_starts is not None
                assert piece_task_token_counts is not None
                assert piece_task_state_sources is not None
                assert piece_task_state_destinations is not None
                assert piece_mid_state is not None
                assert piece_mid_state_ready is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    persistent_task_ids,
                    persistent_task_offsets,
                    piece_task_token_starts,
                    piece_task_token_counts,
                    piece_task_state_sources,
                    piece_task_state_destinations,
                    piece_mid_state,
                    piece_mid_state_ready,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            else:
                assert dummy_i32 is not None
                assert dummy_i64 is not None
                direct_args = (
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    state_indices if state_indices is not None else dummy_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    (
                        state_checkpoints
                        if state_checkpoints is not None
                        else dummy_state
                    ),
                    (
                        checkpoint_cu_starts
                        if checkpoint_cu_starts is not None
                        else dummy_i64
                    ),
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    beta.stride(-2),
                    state_slot_stride,
                    int(use_state_indices),
                    int(use_initial_state),
                    int(store_final_state),
                    checkpoint_every_n_tokens,
                    scale_value,
                    float(lower_bound if lower_bound is not None else 0.0),
                    stream_ptr,
                )
                module.run(*direct_args)
        except Exception:
            if prepare_descriptors:
                workspace._descriptor_signatures.pop(variant, None)
            raise
        if prepare_descriptors:
            workspace._descriptor_signatures[variant] = signature
        if capturing and explicit_workspace:
            workspace._captured = True
    result = (out_buf, returned_state if output_final_state else None)
    if checkpoint_every_n_tokens:
        assert state_checkpoints is not None
        return (*result, state_checkpoints)
    return result


# ===========================================================================
# SM120a ordinary multi-token prefill.
#
# A second prefill backend beside the frozen SM100-family Cake one above, on a
# disjoint set of architectures: Cake takes compute capability 10.0 and 10.3,
# this takes 12.0, and no device is eligible for both.  So the two do not
# compete, and adding this one cannot change which kernel an SM100 call gets.
#
# The split of responsibility follows the Cake half exactly.  This file owns
# the public contract -- the structural checks, the output and state
# adaptation, the workspace shell -- and knows nothing about CuTe DSL.  The
# kernels, their caches and their variant choice live in
# ``flashinfer.kda_kernels.sm120_prefill`` and are reached only through the
# optional facade in ``flashinfer.kda_kernels``, so a CPU-only import, an
# SM100 box or a missing CuTe DSL leaves this a no-op rather than an error.
# ===========================================================================

_SM120_KDA_SUPPORTED_COMPUTE_CAPABILITIES = {(12, 0)}

#: The gate's supported range. The safe gate's worst-case chunk prefix is
#: ``16 * lower_bound * log2e``, which reaches the ``rcp.approx.ftz`` cliff at
#: ``lower_bound == -5.4585``; this keeps a real margin below that.  The public
#: API also excludes ``0.0`` because it is the degenerate zero gate; the direct
#: backend ABI accepts it for internal use, but public dispatch remains strict.
_SM120_KDA_LOWER_BOUND_MIN = -5.0


def _sm120_kda_prefill_rejection_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> Optional[str]:
    """Why this call cannot use the SM120 backend, or ``None`` if it can.

    Structural only: shapes, dtypes, devices, strides, alignment and the
    handful of flags whose values are host-side. It reads no tensor contents,
    launches nothing, allocates nothing and never synchronizes, so the
    dispatcher can afford to call it on every request and a graph capture can
    call it too.

    ``cu_seqlens``'s *values* -- starting at zero, non-decreasing, ending at
    the token count -- are deliberately not checked here. Reading them needs a
    device-to-host copy, which synchronizes and is illegal during capture; the
    backend validates them once during eager warmup and caches the result on
    the tensor's identity and version instead.

    A string rather than a bool because the same predicate serves two callers
    with opposite needs: the dispatcher wants "eligible or not" and discards
    the reason, while a caller who forced this backend wants to be told which
    of thirty conditions it missed.
    """
    if not _is_plain_multi_token_prefill(q, cu_seqlens, num_spec_tokens):
        return "not an ordinary multi-token prefill"
    if num_accepted_tokens is not None:
        return "num_accepted_tokens is a speculative-decode argument"
    if initial_state_source is not None or initial_state_indices is not None:
        return "initial_state_source/initial_state_indices are unsupported"
    if ssm_state_indices is not None:
        return "ssm_state_indices (state pooling) is unsupported"
    if seq_order is not None:
        return "seq_order is unsupported"
    if (
        checkpoint_every_n_tokens != 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        return "prefill state checkpoints are unsupported"
    if not use_qk_l2norm_in_kernel:
        return "use_qk_l2norm_in_kernel=False is unsupported"
    if not use_gate_in_kernel:
        return "use_gate_in_kernel=False is unsupported"
    if not beta_is_logit:
        return "beta_is_logit=False is unsupported"
    if lower_bound is None or not math.isfinite(float(lower_bound)):
        return "lower_bound must be a finite negative float"
    if not _SM120_KDA_LOWER_BOUND_MIN <= float(lower_bound) < 0.0:
        return (
            f"lower_bound must be in [{_SM120_KDA_LOWER_BOUND_MIN}, 0.0), got "
            f"{lower_bound}"
        )
    if scale is not None and not math.isfinite(float(scale)):
        return "scale must be finite"

    if not q.is_cuda:
        return "q must be a CUDA tensor"
    if (
        get_compute_capability(q.device)
        not in _SM120_KDA_SUPPORTED_COMPUTE_CAPABILITIES
    ):
        return "device compute capability is not 12.0"

    device = q.device
    if not _is_contiguous_cuda_tensor(q, dtype=torch.bfloat16, device=device):
        return "q must be a contiguous CUDA bfloat16 tensor"
    if q.ndim != 4:
        return "q must be rank 4"
    batch_size, tokens, num_heads, head_dim = q.shape
    if head_dim != _FLASH_KDA_HEAD_DIM:
        return f"the head dimension is fixed at {_FLASH_KDA_HEAD_DIM}"
    if num_heads <= 0 or batch_size <= 0:
        return "B and H must be positive"
    for name, tensor in (("k", k), ("v", v), ("g", g)):
        if not _is_contiguous_cuda_tensor(tensor, dtype=torch.bfloat16, device=device):
            return f"{name} must be a contiguous CUDA bfloat16 tensor"
        if tensor.shape != q.shape:
            return f"{name} must have q's shape; GQA and V != K are unsupported"
    if not _is_contiguous_cuda_tensor(
        beta, dtype=torch.bfloat16, device=device
    ) or beta.shape != (batch_size, tokens, num_heads):
        return "beta must be a contiguous CUDA bfloat16 [B, T, H] tensor"

    if not _is_contiguous_cuda_tensor(
        A_log, dtype=torch.float32, device=device
    ) or A_log.shape != (num_heads,):
        return "A_log must be a contiguous CUDA float32 [H] tensor"
    if not _is_contiguous_cuda_tensor(dt_bias, dtype=torch.float32, device=device):
        return "dt_bias must be a contiguous CUDA float32 tensor"
    if dt_bias.numel() != num_heads * _FLASH_KDA_HEAD_DIM or dt_bias.ndim not in (
        1,
        2,
    ):
        return f"dt_bias must hold H * {_FLASH_KDA_HEAD_DIM} float32 values"
    if dt_bias.ndim == 2 and dt_bias.shape != (num_heads, _FLASH_KDA_HEAD_DIM):
        return f"a rank-2 dt_bias must be [H, {_FLASH_KDA_HEAD_DIM}]"

    if cu_seqlens is None:
        num_sequences = batch_size
    else:
        if batch_size != 1:
            return "packed input needs a leading dimension of exactly 1"
        if (
            not cu_seqlens.is_cuda
            or cu_seqlens.device != device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            return "cu_seqlens must be a contiguous CUDA int32/int64 [N + 1] tensor"
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or tokens <= num_sequences:
            return "packed input needs more tokens than sequences"

    if initial_state is not None:
        if not _is_contiguous_cuda_tensor(
            initial_state, dtype=torch.bfloat16, device=device
        ):
            return "initial_state must be a contiguous CUDA bfloat16 tensor"
        if tuple(initial_state.shape) != (
            num_sequences,
            num_heads,
            _FLASH_KDA_HEAD_DIM,
            _FLASH_KDA_HEAD_DIM,
        ):
            return (
                f"initial_state must be [N, H, {_FLASH_KDA_HEAD_DIM}, "
                f"{_FLASH_KDA_HEAD_DIM}]; a state pool is unsupported"
            )

    if output is not None:
        if (
            not _is_contiguous_cuda_tensor(output, dtype=torch.bfloat16, device=device)
            or output.shape != v.shape
        ):
            return "output must be a contiguous CUDA bfloat16 tensor with v's shape"
        # The backend's own ABI would accept an exact ``out``/``v`` alias, and
        # the kernel's schedule proves it safe. The public contract does not:
        # ``recurrent_kda`` promises a caller that ``output`` and its inputs are
        # distinct buffers, and quietly honouring an alias here would make that
        # promise depend on which architecture the call landed on.
        if _check_sm120_output_overlaps(
            output, (q, k, v, g, beta, A_log, dt_bias, initial_state)
        ):
            return "output must not overlap any input in GMEM"

    # TMA describes q, k, v, g, output and the states, and a TensorMap requires
    # a 16-byte-aligned base.  The backend validates this too, but validation
    # raises where a rejection reason falls back -- and contiguity does not
    # imply alignment: `base[1:].view(...)` on a bfloat16 buffer is contiguous
    # and starts two bytes in.  Without this the dispatcher would select SM120
    # for such a tensor and the caller would get an exception from descriptor
    # construction instead of the SM100 path that would have run it.
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("output", output),
        ("initial_state", initial_state),
    ):
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.numel()
            and tensor.data_ptr() % _SM120_TMA_BASE_ALIGN
        ):
            return f"{name} must be {_SM120_TMA_BASE_ALIGN}-byte aligned for TMA"

    if prefill_workspace is not None and prefill_workspace.device != device:
        return "prefill_workspace was created for a different device"
    return None


def _check_sm120_output_overlaps(output: torch.Tensor, inputs) -> bool:
    """Does ``output`` share any bytes with an input?

    Byte ranges, not storage identity: two tensors can share a storage object
    and never overlap, and two tensors from different allocations can be views
    of one block.
    """
    for tensor in inputs:
        if tensor is None or tensor.numel() == 0:
            continue
        if _storage_ranges_overlap(output, tensor):
            return True
    return False


def _sm120_kda_prefill_is_eligible(**kwargs) -> bool:
    """Whether the SM120 backend should take this call.

    Two stages, cheapest and most self-contained first. The structural checks
    above use only torch, so they run without importing anything from the
    kernel package; only once they pass is the optional backend consulted at
    all, and that question -- can this process natively build ``sm_120a``? --
    is the one thing this file cannot answer for itself.

    Fail-closed throughout: a missing CuTe DSL, an unbuildable target or a
    facade that failed to import all read as "not eligible", and the call falls
    through to the existing backends.
    """
    if _sm120_kda_prefill_rejection_reason(**kwargs) is not None:
        return False
    from . import kda_kernels

    can_implement = kda_kernels.can_implement_kda_prefill_sm120
    return can_implement is not None and can_implement(**kwargs)


def _sm120_prefill_resources(
    workspace: Optional[RecurrentKDAPrefillWorkspace],
    device: torch.device,
):
    """The SM120 half of a caller's workspace, created on first use.

    A Cake-only caller never reaches this, so their workspace carries one
    ``None`` field and never imports CuTe DSL. Composition rather than a second
    public workspace class: ``RecurrentKDAPrefillWorkspace`` stays the only
    workspace a caller constructs, and which backend it ends up bound to is an
    implementation detail of the call it was warmed on.
    """
    if workspace is None:
        return None
    resources = workspace._sm120_state
    if resources is not None:
        # Double-checked: the warm path never takes the lock, and the object it
        # reads is only ever published once.
        return resources
    with workspace._sm120_state_lock:
        resources = workspace._sm120_state
        if resources is None:
            from .kda_kernels.sm120_prefill.runtime import SM120PrefillResources

            resources = SM120PrefillResources(device=device)
            workspace._sm120_state = resources
    return resources


def _sm120_final_state_scratch(sequences: int, heads: int, device, resources):
    """A bfloat16 final state for a call that supplied no initial state.

    Eagerly this is a fresh allocation. Under capture it cannot be -- allocating
    inside a graph is precisely what the workspace exists to avoid -- so the
    buffer comes from the workspace and keeps its address for as long as the
    workspace lives.
    """
    shape = (sequences, heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
    if resources is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    scratch = resources.state_scratch
    if scratch is not None and tuple(scratch.shape) == shape:
        return scratch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CUDA graph capture cannot allocate the final-state scratch; warm "
            "the workspace with one eager call at this shape first"
        )
    resources.state_scratch = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return resources.state_scratch


def _run_sm120_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the SM120 backend and adapt its result to the public contract.

    The state semantics are the public ones, not the backend's:

    * a supplied ``initial_state`` is updated in place whether or not
      ``output_final_state`` is set -- that is what the caller's buffer is for;
    * with ``output_final_state=False`` the second return value is ``None``
      even so, because returning a state the caller did not ask for would be a
      different contract than the one they called;
    * without an initial state but with ``output_final_state=True`` a bfloat16
      final state is allocated, or taken from the workspace when one is bound.
    """
    if output is None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "CUDA graph capture requires a preallocated output tensor for "
            "recurrent_kda prefill"
        )

    from . import kda_kernels

    run = kda_kernels.run_kda_prefill_sm120
    if run is None:
        raise RuntimeError(
            "the SM120 KDA CuTe DSL prefill backend is unavailable"
        ) from kda_kernels._kda_sm120_import_error

    # ``[H * 128]`` adapts with a no-copy view; the backend's ABI is ``[H, 128]``.
    if dt_bias.ndim == 1:
        dt_bias = dt_bias.view(q.shape[2], _FLASH_KDA_HEAD_DIM)

    if prefill_workspace is not None and prefill_workspace._captured:
        # Check the shared shell before creating its SM120 state: it may have
        # been spent by a captured Cake call that never needed these resources.
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace has participated in CUDA graph "
            "capture and cannot be reused or mutated; create another one"
        )

    resources = _sm120_prefill_resources(prefill_workspace, q.device)

    out = output if output is not None else torch.empty_like(v)
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1

    def _resolve_final_state():
        """The public state contract, applied here rather than in the backend.

        The backend's ABI is the kernels' own -- ``initial_state`` read,
        ``final_state`` written, the exact alias between them allowed -- and the
        public promise is a different one: a supplied initial state is updated
        in place, whether or not a final state was requested. Aliasing the two
        is how that promise is kept, and doing it here preserves the backend's
        direct validation ABI.

        Callable rather than a value because the workspace branch resolves it
        under ``resources.lock``: the middle case can *replace* workspace-owned
        scratch, and doing that outside the hold that guards the spent flag lets
        a second thread drop the buffer a first thread's live graph reads at its
        captured address.
        """
        if initial_state is not None:
            return initial_state
        if output_final_state:
            return _sm120_final_state_scratch(
                num_sequences, q.shape[2], q.device, resources
            )
        # Nothing to store: the kernels skip the state write entirely rather
        # than filling a buffer the caller will not read.
        return None

    def _launch(final_state):
        run(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            lower_bound=float(lower_bound),
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            output=out,
            resources=resources,
        )
        # ``output_final_state=False`` returns None even when a state was
        # updated in place: returning it because it happens to exist would be a
        # different contract than the one the caller asked for.
        return out, (final_state if output_final_state else None)

    if resources is None:
        # Refuse capture here, at the outermost adapter, rather than relying on
        # the guards further down.  Those sit on cache *misses* -- the
        # device-to-host read for cu_seqlens, the canonical-offsets allocation
        # -- so a warm cache walks straight past them and the capture succeeds.
        #
        # It must not.  Without an explicit workspace, the descriptors and
        # scratch a capture records belong to a bounded LRU, and the graph holds
        # their raw addresses.  Nothing pins them: the next distinct shape can
        # evict the entry, and clear_kda_prefill_sm120_caches() drops it
        # outright.  Replay then reads freed memory, arbitrarily far from the
        # call that captured it, and the symptom is wrong output rather than an
        # error.  An explicit workspace is what makes those addresses the
        # caller's to keep alive.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture of the SM120 KDA prefill backend requires "
                "an explicit workspace: without one the graph would record "
                "addresses owned by an evictable cache. Pass a plan's "
                "workspace, and warm it with one eager call on the capture "
                "stream before capturing."
            )
        return _launch(_resolve_final_state())

    # The workspace serializes its own launch sequence: the decomposed variant
    # enqueues two kernels that share one scratch arena, and two host threads
    # interleaving those pairs would have the second prepare overwrite factors
    # the first recurrence has not read yet.
    #
    # One hold covers the spent check, the scratch resolution and the launch.
    # Split, they race each other: a thread that read ``captured`` as False can
    # replace ``state_scratch`` after another thread's capture has recorded the
    # old buffer's address, and two threads wanting different state shapes can
    # each install their own, leaving the loser holding a ``final_state`` the
    # workspace no longer owns. The backend re-checks the flag, which orders the
    # launches, but it cannot undo a replacement that already happened.
    with resources.lock:
        if resources.captured:
            raise RuntimeError(
                "RecurrentKDAPrefillWorkspace has participated in CUDA graph "
                "capture and cannot be reused or mutated; create another one"
            )
        result = _launch(_resolve_final_state())
        if torch.cuda.is_current_stream_capturing():
            # A workspace that has been captured is spent: replay reads its
            # buffers at the addresses capture recorded, and handing it back to
            # Python -- eagerly or for a second capture -- would let those
            # addresses move under a live graph.
            resources.captured = True
            prefill_workspace._captured = True
    return result
