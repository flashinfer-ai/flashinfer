"""Tensor-parallel MoE entrypoints backed by fused CuTe DSL kernels."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS

import os
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from b12x._lib.compile_plan import launch_triton, planning
from b12x._lib.program_cache import register_program_cache
from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.intrinsics import (
    align_up,
    as_grouped_scale_view,
    swizzle_block_scale,
)
from b12x._lib.utils import (
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from cutlass.cutlass_dsl import Int32
from b12x.moe._shared.routing import (
    route_topk as triton_route_topk,
    _validate_score_options,
)
from b12x.moe._shared.kernels.relu2 import (
    MoEDynamicKernelRelu2,
    MoEMicroKernelRelu2,
)
from b12x.moe._shared.kernels.silu import (
    MoEDynamicKernelSilu,
    MoEDynamicKernelSwiGLUOAI,
    MoEMicroKernelSilu,
    MoEMicroKernelSwiGLUOAI,
)
from b12x.moe._shared.kernels.situ import (
    MoEDynamicKernelSitu,
    MoEMicroKernelSitu,
)
from b12x.moe._shared.kernels.activations import (
    SITU,
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    moe_activation_w1_rows,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from b12x.moe._shared.kernels.micro import (
    _BLOCK_DIM as _DIRECT_MICRO_BLOCK_DIM,
    MoEMicroKernelBackend,
    _direct_k_segments_for_k,
    _direct_k_segments_supported,
)
from b12x.moe._shared.execution import (
    MoEExecutionPlan,
    MoERegime,
    GemmEngine,
    MoESpec,
    MoEWeightPreparationPlan,
    OutputReduction,
    PreparedWeightLayout,
    WeightPreparationTransform,
    WorkScheduler,
    lower_moe_execution,
    make_moe_spec,
    plan_moe_weight_preparation,
)
from b12x.moe._shared.execution import (
    _SOURCE_FORMATS as _EXECUTION_SOURCE_FORMATS,
    _TRELLIS_SOURCE_FORMATS as _EXECUTION_TRELLIS_SOURCE_FORMATS,
)
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x.preparation import DeviceIdentity
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)

from ._route_pack_cache import route_pack_prewarm_key
from ._tuning import MoeDecodeConfig, MoeDecodeQuery

logger = logging.getLogger(__name__)
_B12X_TIMING = (
    os.getenv("B12X_TIMING", "0") == "1" or os.getenv("VLLM_B12X_TIMING", "0") == "1"
)
_B12X_TIMING_THRESHOLD_MS = float(
    os.getenv(
        "B12X_TIMING_THRESHOLD_MS",
        os.getenv("VLLM_B12X_TIMING_THRESHOLD_MS", "0"),
    )
)

_NVFP4_BLOCK_SIZE = 16
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
_DYNAMIC_SLICE_CHUNK = 1
_DYNAMIC_WORK_SOURCE_ENV = "B12X_DYNAMIC_WORK_SOURCE"
_DYNAMIC_WORK_SOURCE_DEFAULT = "materialized_queue"
_DYNAMIC_EXTERNAL_ROUTE_PLAN_ENV = "B12X_DYNAMIC_EXTERNAL_ROUTE_PLAN"
_DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS = 256
_DYNAMIC_WORK_SOURCES = {
    "persistent_grid",
    "materialized_queue",
    "ready_queue",
}
_W4A16_ROUTE_PACK_PREWARMED: set[tuple[object, ...]] = set()
# W4A8's unified dynamic specialization consumes an N256/K128 lane-major
# weight representation.  Preparation is independent from scheduling: the
# same representation serves both materialized queue and persistent-grid work.
_DYNAMIC_W4A8_REPACKED_ENV = "B12X_DYNAMIC_W4A8_REPACKED"
_DYNAMIC_W4A8_SHARE_INPUT_ENV = "B12X_DYNAMIC_W4A8_SHARE_INPUT"
_DYNAMIC_W4A8_MATERIALIZED_ENV = "B12X_DYNAMIC_W4A8_MATERIALIZED"
# Captured by plan construction as a constraint on NVFP4 materialization.
_DYNAMIC_NVFP4_MATERIALIZED_ENV = "B12X_NVFP4_DYNAMIC_MATERIALIZED"
_W4A8_CONVERT_SCRATCH_MB_ENV = "B12X_W4A8_CONVERT_SCRATCH_MB"
_W4A8_CONVERT_SCRATCH_MB_DEFAULT = 64
# The source-format vocabulary is owned by b12x.moe._shared.execution;
# these views keep the historical local names.
_FP4_SOURCE_FORMATS = {name: name for name in _EXECUTION_SOURCE_FORMATS}
_TRELLIS_SOURCE_FORMATS = frozenset(_EXECUTION_TRELLIS_SOURCE_FORMATS)
_PROJECTION_MIXED_TRELLIS_MAX_ROUTE_BLOCK_SIZE = 48
_W4A16_SCALE_FORMATS = {
    "iq2_xs": "iq2_xs",
    "iq2_xxs": "iq2_xxs",
    "q8_0": "q8_0",
    "e4m3_k16": "e4m3_k16",
    "e4m3_k16_csf": "e4m3_k16_csf",
    "e4m3_k32": "e4m3_k32",
    "e8m0_k32": "e8m0_k32",
}
_W13_LAYOUTS = {
    "w13": "w13",
    "w31": "w31",
    # Accept the physical FC1-half spellings as aliases; "up_gate" needs the
    # in-place W13 row rotation ("w13"), "gate_up" is already kernel-native.
    "up_gate": "w13",
    "gate_up": "w31",
    "trellis_t256_proj": "trellis_t256_proj",
}

_DEVICE_CAPABILITY_CACHE: dict[int, tuple[int, int]] = {}


def _current_compute_capability() -> tuple[int, int] | None:
    """Return the active CUDA device capability without repeated driver queries."""

    if not torch.cuda.is_available():
        return None
    device_index = torch.cuda.current_device()
    capability = _DEVICE_CAPABILITY_CACHE.get(device_index)
    if capability is None:
        capability = tuple(torch.cuda.get_device_capability(device_index))
        _DEVICE_CAPABILITY_CACHE[device_index] = capability
    return capability


@dataclass(kw_only=True)
class TPMoEWorkspace:
    """Reusable scratch buffers for one `b12x_moe_fp4` shape family."""

    implementation: str
    quant_mode: str
    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor
    packed_a_view: object = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None
    packed_a_storage_ptr: object = None
    route_workspace: "_TPRouteWorkspace | None" = None
    volatile_launch_state: bool = False

    def bind_fp4(self, **kwargs) -> "TPMoEFP4Binding":
        return build_tp_moe_fp4_binding(scratch=self, **kwargs)

    def bind_route(self, **kwargs) -> "TPMoERouteBinding":
        return build_tp_moe_route_binding(scratch=self, **kwargs)

    def bind_sparse_fp4(self, **kwargs) -> "TPMoESparseFP4Binding":
        return build_tp_moe_sparse_fp4_binding(scratch=self, **kwargs)


@dataclass(kw_only=True)
class TPMicroWorkspace(TPMoEWorkspace):
    routed_rows_capacity: int
    active_expert_count: torch.Tensor
    weight_expert_ids: torch.Tensor
    global_to_local_expert: torch.Tensor
    compact_topk_ids: torch.Tensor
    micro_intermediate: torch.Tensor


@dataclass(kw_only=True)
class TPDynamicWorkspace(TPMoEWorkspace):
    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    route_output: torch.Tensor
    materialized_intermediate: torch.Tensor
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    input_gs: torch.Tensor
    down_input_scale: torch.Tensor
    pair_head: torch.Tensor
    producers_done_count: torch.Tensor
    all_work_published: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_ready: torch.Tensor
    task_expert: torch.Tensor
    task_m_tile: torch.Tensor
    task_slice_begin: torch.Tensor
    task_slice_count: torch.Tensor
    task_valid_rows: torch.Tensor
    tile_write_count: torch.Tensor
    input_gs_src_ptr: int = 0
    down_input_scale_src_ptr: int = 0


@dataclass(kw_only=True)
class TPW4A16Workspace:
    implementation: str
    quant_mode: str
    activation: str
    state_E: int
    weight_E: int
    route_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    routed_rows_capacity: int
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    prefill_sum_accum: torch.Tensor | None
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor | None = None
    full_rotation_output: torch.Tensor | None = None
    rotation_a_gate: torch.Tensor | None = None
    rotation_a_up: torch.Tensor | None = None
    kernel_workspace: torch.Tensor | None = None
    full_rotation: bool = False
    trellis_bits: int = 3
    trellis_tile_config: tuple[int, int, int, int] | None = None
    trellis_pair_kinds: frozenset[str] | None = None
    trellis_codebook: str | None = None
    intermediate_hadamard: bool = False
    route_block_size_m: int | None = None
    planned_token_counts: frozenset[int] = field(default_factory=frozenset)
    planned_apply_router_weight_on_input: bool = False
    planned_swiglu_limit: float | None = None
    planned_swiglu_alpha: float = 1.0
    planned_swiglu_beta: float = 0.0
    planned_scale_format: str = "e4m3_k16"
    planned_collect_activation_amax: bool = False
    planned_prefill_fused_sum_fp32: bool = False
    planned_fused_moe_launches: dict[object, object] = field(default_factory=dict)
    planned_topk_sum_launches: dict[object, object] = field(default_factory=dict)
    # Mapped direct-route launches, keyed by exact live token count. These
    # consume global router ids plus the binding's global->local map inside the
    # fused kernel and therefore bypass expert route packing during decode.
    planned_mapped_direct_launches: dict[int, object] = field(default_factory=dict)
    # Unmapped direct-top-k launches for non-gated W4A16, keyed by exact live
    # token count. These retain the separate top-k reduction.
    planned_direct_topk_launches: dict[int, object] = field(default_factory=dict)
    # TC-decode fused-sum launches, keyed by exact token count (only the small-M
    # decode sizes in TC-decode's supported set, packed layout only).
    planned_tc_decode_launches: dict[int, object] = field(default_factory=dict)
    route_workspace: "_TPRouteWorkspace | None" = None
    volatile_launch_state: bool = False

    def bind_fp4(self, **kwargs) -> "TPMoEFP4Binding":
        return build_tp_moe_fp4_binding(scratch=self, **kwargs)

    def bind_route(self, **kwargs) -> "TPMoERouteBinding":
        return build_tp_moe_route_binding(scratch=self, **kwargs)

    def bind_sparse_fp4(self, **kwargs) -> "TPMoESparseFP4Binding":
        return build_tp_moe_sparse_fp4_binding(scratch=self, **kwargs)


@dataclass
class TPMoEWorkspacePool:
    """Caller-owned capacity-based workspace cache for one execution lane.

    A single explicit pool may be shared across layers in a lane. Independent
    overlapping lanes must use distinct pools; internal fork/join streams share
    the lane pool and therefore the same scratch arena.
    """

    workspaces: Dict[Tuple, object] = field(default_factory=dict)
    route_workspaces: Dict[Tuple, "_TPRouteWorkspace"] = field(default_factory=dict)
    core_arenas: Dict[Tuple, "_TPCoreArena"] = field(default_factory=dict)
    shared_arena: torch.Tensor | None = None
    shared_arena_nbytes: int = 0
    route_workspace_nbytes: int = 0
    core_arena_offset_bytes: int = 0
    core_arena_nbytes: int = 0
    frozen: bool = False

    def clear(self) -> None:
        self.workspaces.clear()
        self.route_workspaces.clear()
        self.core_arenas.clear()

    def bind_shared_arena(
        self,
        shared_arena: torch.Tensor,
        *,
        route_workspace_nbytes: int,
        core_workspace_nbytes: int,
        frozen: bool = True,
    ) -> None:
        if shared_arena.dtype != torch.uint8:
            raise TypeError(
                f"shared_arena must have dtype torch.uint8, got {shared_arena.dtype}"
            )
        route_workspace_nbytes = align_up(max(int(route_workspace_nbytes), 0), 16)
        core_workspace_nbytes = max(int(core_workspace_nbytes), 0)
        required = route_workspace_nbytes + core_workspace_nbytes
        if shared_arena.numel() < max(required, 1):
            raise ValueError(
                f"shared_arena has {shared_arena.numel()} bytes, but MoE workspace requires {required}"
            )
        self.clear()
        self.shared_arena = shared_arena
        self.shared_arena_nbytes = int(shared_arena.numel())
        self.route_workspace_nbytes = route_workspace_nbytes
        self.core_arena_offset_bytes = route_workspace_nbytes
        self.core_arena_nbytes = core_workspace_nbytes
        self.frozen = bool(frozen)

    def bind_fp4(self, **kwargs) -> "TPMoEFP4Binding":
        return build_tp_moe_fp4_binding(scratch=self, **kwargs)

    def bind_route(self, **kwargs) -> "TPMoERouteBinding":
        return build_tp_moe_route_binding(scratch=self, **kwargs)

    def bind_sparse_fp4(self, **kwargs) -> "TPMoESparseFP4Binding":
        return build_tp_moe_sparse_fp4_binding(scratch=self, **kwargs)


@dataclass(frozen=True, kw_only=True)
class _PreparedW4A8Weights:
    """Exact tiled W4A8 weights prepared for the unified dynamic kernel."""

    w13_rp: torch.Tensor
    w13_sfb: torch.Tensor
    w2_rp: torch.Tensor
    w2_sfb: torch.Tensor
    num_experts: int
    hidden_size: int
    intermediate_size: int
    params_dtype: torch.dtype
    source_format: str = "fp4_e8m0_k32"
    w13_layout: str = "w13"
    n64_repack: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_format",
            _normalize_fp4_source_format(self.source_format),
        )
        object.__setattr__(self, "w13_layout", _normalize_w13_layout(self.w13_layout))
        object.__setattr__(self, "num_experts", int(self.num_experts))
        object.__setattr__(self, "hidden_size", int(self.hidden_size))
        object.__setattr__(self, "intermediate_size", int(self.intermediate_size))


@dataclass(frozen=True, kw_only=True)
class _PreparedWeightRepresentation:
    """One materialized representation selected by the weight planner."""

    quant_mode: str
    layout: PreparedWeightLayout
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "quant_mode", str(self.quant_mode).lower())
        object.__setattr__(self, "layout", PreparedWeightLayout(self.layout))


@dataclass(frozen=True, kw_only=True)
class B12XFP4ExpertWeights:
    """The sole owner and complete runtime contract for FP4 MoE experts.

    The weight and scale fields are canonical storage handles.  A planner may
    reinterpret or repack those allocations in place, but a representation is
    invalid if its FC1/FC2 weights live in different storage.  Execution takes
    this object directly; there is no raw-weights-plus-prepared dual API.
    """

    plan: MoEWeightPreparationPlan
    a1_gscale: torch.Tensor  # reciprocal activation global scale for FC1 input
    w1_fp4: torch.Tensor
    w1_blockscale: torch.Tensor
    w1_alphas: torch.Tensor
    a2_gscale: torch.Tensor  # reciprocal activation global scale for FC2 input
    w2_fp4: torch.Tensor
    w2_blockscale: torch.Tensor
    w2_alphas: torch.Tensor
    representation: _PreparedWeightRepresentation | None = None
    immutable_input_scales: bool = False
    nvfp4_csf: object | None = None
    # Packed W4A16 representation over the expanded scale scratch, for calls
    # planned above the stage-scale token limit (nvfp4_csf expands into it).
    w4a16_expanded: object | None = None
    mxfp4_csf: object | None = None
    # Compact W4A8 experts prepared from MXFP4-CSF checkpoints whose kernels read
    # compressed scales inline: (w13, w2) Mxfp4CsfInlinePlane. The canonical scale
    # fields keep the caller's expansion scratch, which these launches never read.
    mxfp4_csf_inline: tuple | None = None
    _uniform_a1_scale: bool = field(default=False, init=False, repr=False)
    _a1_scale_version: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, MoEWeightPreparationPlan):
            raise TypeError("plan must be a MoEWeightPreparationPlan")
        for name, tensor in (
            ("a1_gscale", self.a1_gscale),
            ("w1_fp4", self.w1_fp4),
            ("w1_blockscale", self.w1_blockscale),
            ("w1_alphas", self.w1_alphas),
            ("a2_gscale", self.a2_gscale),
            ("w2_fp4", self.w2_fp4),
            ("w2_blockscale", self.w2_blockscale),
            ("w2_alphas", self.w2_alphas),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        representation = self.representation
        if representation is not None:
            if not isinstance(representation, _PreparedWeightRepresentation):
                raise TypeError(
                    "representation must be a planner-created prepared weight "
                    "representation"
                )
            key = (representation.quant_mode, representation.layout)
            if representation.quant_mode not in self.plan.quant_modes:
                raise ValueError(
                    f"prepared quant_mode={representation.quant_mode!r} is absent "
                    "from the weight-preparation plan"
                )
            required = self.plan.required_weight_layout(representation.quant_mode)
            if required is not representation.layout:
                raise ValueError(
                    f"prepared layout={representation.layout.value!r} does not "
                    "match planned layout="
                    f"{required.value if required is not None else None!r}"
                )
            value = representation.value
            prepared_w1 = getattr(value, "w13_rp", getattr(value, "w13", None))
            prepared_w2 = getattr(value, "w2_rp", getattr(value, "w2", None))
            prepared_s1 = getattr(value, "w13_sfb", getattr(value, "w13_scale", None))
            prepared_s2 = getattr(value, "w2_sfb", getattr(value, "w2_scale", None))
            if not isinstance(prepared_w1, torch.Tensor) or not isinstance(
                prepared_w2, torch.Tensor
            ):
                raise TypeError("prepared representation is missing FC1/FC2 tensors")
            if (
                prepared_w1.untyped_storage().data_ptr()
                != self.w1_fp4.untyped_storage().data_ptr()
                or prepared_w2.untyped_storage().data_ptr()
                != self.w2_fp4.untyped_storage().data_ptr()
            ):
                raise ValueError(
                    "prepared FC1/FC2 tensors must reuse the expert package's "
                    "canonical allocations; retaining source plus repack is invalid"
                )
            if self.plan.discards_source_parameters:
                if not isinstance(prepared_s1, torch.Tensor) or not isinstance(
                    prepared_s2, torch.Tensor
                ):
                    raise TypeError("prepared representation is missing FC1/FC2 scales")
                if (
                    prepared_s1.untyped_storage().data_ptr()
                    != self.w1_blockscale.untyped_storage().data_ptr()
                    or prepared_s2.untyped_storage().data_ptr()
                    != self.w2_blockscale.untyped_storage().data_ptr()
                ):
                    raise ValueError(
                        "a transferred representation must be the package's "
                        "canonical scale storage"
                    )
            for name, expected_value in (
                ("num_experts", self.num_experts),
                ("hidden_size", self.hidden_size),
                ("intermediate_size", self.intermediate_size),
            ):
                actual_value = getattr(value, name, None)
                if actual_value is not None and int(actual_value) != expected_value:
                    raise ValueError(
                        f"prepared {name}={int(actual_value)} does not match "
                        f"plan {name}={expected_value}"
                    )
            actual = {key}
        else:
            actual = set()
        expected = {
            (mode, required)
            for mode in self.plan.quant_modes
            if (required := self.plan.required_weight_layout(mode)) is not None
        }
        if actual != expected:
            raise ValueError(
                "materialized weight representation does not match the plan: "
                f"expected={sorted((m, l.value) for m, l in expected)}, "
                f"actual={sorted((m, l.value) for m, l in actual)}"
            )
        source_mode_selected = bool({"nvfp4", "w4a8_nvfp4"} & self.plan.quant_modes)
        if source_mode_selected or representation is None:
            if self.w1_fp4.ndim != 3 or self.w2_fp4.ndim != 3:
                raise ValueError("source-native FP4 expert weights must be rank-3")
            actual_w2 = (
                int(self.w2_fp4.shape[0]),
                int(self.w2_fp4.shape[1]),
                int(self.w2_fp4.shape[2]) * 2,
            )
            expected_w2 = (
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
            )
            if actual_w2 != expected_w2:
                raise ValueError(
                    "FC2 shape does not match the prepared plan: "
                    f"actual={actual_w2}, expected={expected_w2}"
                )

        if (
            self.immutable_input_scales
            and "nvfp4" in self.plan.quant_modes
            and not self.a1_gscale.is_meta
        ):
            # Weight preparation may inspect values; binding and graph replay
            # may not. Preserve the vector so caller-owned scratch bindings can
            # keep reading canonical expert scales without expansion copies.
            scales = self.a1_gscale.detach().reshape(-1)
            if scales.numel() > 1:
                uniform = bool(
                    (torch.isfinite(scales) & (scales > 0) & (scales == scales[0]))
                    .all()
                    .item()
                )
                object.__setattr__(self, "_uniform_a1_scale", uniform)
                if not torch.is_inference(self.a1_gscale):
                    object.__setattr__(
                        self, "_a1_scale_version", self.a1_gscale._version
                    )

    def can_share_input(self, *, input_scales_static: bool) -> bool:
        """Whether one NVFP4 input quantization can serve every routed expert.

        Vector sharing requires immutable_input_scales at weight preparation:
        values must not change while a binding or captured graph remains live.
        Reprepare weights and recapture graphs after changing those values.
        Versioned eager mutations invalidate the preparation-time proof;
        inference tensors rely on the explicitly supplied static contract.
        """
        if self.a1_gscale.numel() == 1:
            return True
        if (
            not self.immutable_input_scales
            or not input_scales_static
            or not self._uniform_a1_scale
        ):
            return False
        return (
            self._a1_scale_version is None
            or self.a1_gscale._version == self._a1_scale_version
        )

    @property
    def source_format(self) -> str:
        return self.plan.source_format

    @property
    def w13_layout(self) -> str:
        return self.plan.w13_layout

    @property
    def activation(self) -> str:
        return self.plan.activation

    @property
    def num_experts(self) -> int:
        return self.plan.num_experts

    @property
    def hidden_size(self) -> int:
        return self.plan.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.plan.intermediate_size

    def representation_for(self, quant_mode: str) -> Any | None:
        quant_mode = str(quant_mode).lower()
        required = self.plan.required_weight_layout(quant_mode)
        if required is None:
            return None
        representation = self.representation
        if (
            representation is not None
            and representation.quant_mode == quant_mode
            and representation.layout is required
        ):
            return representation.value
        raise RuntimeError(
            f"planned {quant_mode!r} representation {required.value!r} was not "
            "materialized"
        )


def _prepared_payload_for_runtime(
    experts: B12XFP4ExpertWeights,
    *,
    quant_mode: str,
    source_format: str,
    activation: str,
    w13_layout: str,
    dtype: torch.dtype,
    hidden_size: int,
) -> Any | None:
    if not isinstance(experts, B12XFP4ExpertWeights):
        raise TypeError(
            "runtime MoE launches require B12XFP4ExpertWeights from "
            "prepare_b12x_fp4_moe_weights"
        )
    expected = (
        str(quant_mode).lower(),
        _normalize_fp4_source_format(source_format),
        normalize_moe_activation(activation),
        _normalize_w13_layout_for_activation(activation, w13_layout),
        str(dtype).removeprefix("torch."),
        int(hidden_size),
    )
    actual = (
        str(quant_mode).lower(),
        experts.source_format,
        experts.plan.activation,
        experts.w13_layout,
        experts.plan.io_dtype,
        experts.hidden_size,
    )
    if expected != actual:
        raise ValueError(
            "prepared-weight contract does not match runtime: "
            f"runtime={expected}, experts={actual}"
        )
    return experts.representation_for(quant_mode)


def _select_prepared_quant_mode(
    experts: B12XFP4ExpertWeights,
    requested: str | None,
) -> str:
    modes = experts.plan.quant_modes
    if requested is None:
        if len(modes) != 1:
            raise ValueError(
                "quant_mode is required when a prepared-weight plan contains "
                f"multiple recipes: {sorted(modes)}"
            )
        return next(iter(modes))
    quant_mode = _normalize_quant_mode_requested(requested)
    if quant_mode not in modes:
        raise ValueError(
            f"quant_mode={quant_mode!r} is absent from the prepared-weight "
            f"plan {sorted(modes)}"
        )
    return quant_mode


@dataclass(frozen=True, kw_only=True)
class B12XTopKRouting:
    """Top-k routing selection for sparse-block MoE wrappers."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None = None
    flat_ids: torch.Tensor | None = None
    flat_weights: torch.Tensor | None = None


@dataclass(kw_only=True)
class _TPRouteWorkspace:
    router_logits: torch.Tensor
    topk_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class _TensorAllocSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    init: str = "empty"


@dataclass(frozen=True, kw_only=True)
class _TPCoreWorkspacePlan:
    implementation: str
    quant_mode: str
    activation: str
    swiglu_limit: float | None = None
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    state_E: int
    weight_E: int
    route_E: int
    routed_rows: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    deterministic_output: bool = False
    dynamic_physical_tiles: int | None = None
    dynamic_task_capacity: int | None = None
    dynamic_tile_m: int | None = None
    dynamic_tile_n: int | None = None
    dynamic_route_mode: str | None = None
    full_rotation: bool = False
    projection_mixed_trellis: bool = False
    trellis_bits: int = 3
    trellis_tile_config: tuple[int, int, int, int] | None = None
    trellis_pair_kinds: frozenset[str] | None = None
    trellis_codebook: str | None = None
    intermediate_hadamard: bool = False
    route_block_size_m: int | None = None
    prefill_fused_sum_fp32: bool = False
    tensor_specs: Tuple[_TensorAllocSpec, ...] = ()


@dataclass
class _TPCoreArena:
    plan: _TPCoreWorkspacePlan
    shared_arena: torch.Tensor
    tensors: Dict[str, torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class TPMoEArenaLayout:
    route_workspace_nbytes: int
    core_workspace_nbytes: int
    total_nbytes: int
    core_token_counts: tuple[int, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TPMoEPlan:
    """Logical launch plan plus precision-neutral MoE execution descriptors."""

    spec: MoESpec
    execution: MoEExecutionPlan
    implementation: str
    quant_mode: str
    activation: str
    decode_config: MoeDecodeConfig
    swiglu_limit: float | None = None
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    state_E: int
    weight_E: int
    routed_rows: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    max_tokens_per_launch: int
    dynamic_physical_tiles: int | None = None
    dynamic_task_capacity: int | None = None

    @property
    def deterministic_output(self) -> bool:
        return self.execution.reduction is OutputReduction.ROUTE_BUFFER_TOPK_SUM


@dataclass(frozen=True, kw_only=True)
class TPMoEScratchCaps:
    max_tokens: int
    num_topk: int
    device: torch.device | str
    weight_plan: MoEWeightPreparationPlan
    quant_mode: str
    decode_config: MoeDecodeConfig
    core_token_counts: tuple[int, ...] | None = None
    route_num_experts: int | None = None
    route_logits_dtype: torch.dtype | None = None
    apply_router_weight_on_input: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    collect_activation_amax: bool = False
    deterministic_output: bool | None = None
    w4a16_block_size_m: int | None = None
    w4a16_fast_math: bool = True
    w4a16_prefill_fused_sum: bool | None = None
    trellis_decode_table: str = "auto"
    w4a16_skip_empty_m_blocks: bool = True
    w4a16_small_m_occupancy: int = 1
    w4a8_csf_inline: bool = False
    frozen: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "num_topk", max(int(self.num_topk), 1))
        object.__setattr__(self, "device", torch.device(self.device))
        if not isinstance(self.decode_config, MoeDecodeConfig):
            raise TypeError("decode_config must be a concrete MoeDecodeConfig")
        if not isinstance(self.weight_plan, MoEWeightPreparationPlan):
            raise TypeError("weight_plan must be a MoEWeightPreparationPlan")
        quant_mode = _normalize_quant_mode_for_source(
            self.quant_mode,
            self.weight_plan.source_format,
        )
        if quant_mode not in self.weight_plan.quant_modes:
            raise ValueError(
                f"quant_mode={quant_mode!r} is absent from the weight plan"
            )
        object.__setattr__(self, "quant_mode", quant_mode)
        if self.core_token_counts is not None:
            object.__setattr__(
                self,
                "core_token_counts",
                tuple(max(int(count), 1) for count in self.core_token_counts),
            )
        if self.route_num_experts is not None:
            object.__setattr__(
                self,
                "route_num_experts",
                max(int(self.route_num_experts), 0),
            )
        limit, alpha, beta = _normalize_swiglu_params(
            self.activation,
            self.swiglu_limit,
            self.swiglu_alpha,
            self.swiglu_beta,
        )
        object.__setattr__(self, "swiglu_limit", limit)
        object.__setattr__(self, "swiglu_alpha", alpha)
        object.__setattr__(self, "swiglu_beta", beta)
        object.__setattr__(
            self, "collect_activation_amax", bool(self.collect_activation_amax)
        )
        if self.deterministic_output is not None:
            object.__setattr__(
                self, "deterministic_output", bool(self.deterministic_output)
            )
        if self.w4a16_block_size_m is not None:
            block_size_m = int(self.w4a16_block_size_m)
            if block_size_m not in (8, 16, 32, 48, 64):
                raise ValueError("w4a16_block_size_m must be one of 8, 16, 32, 48, 64")
            object.__setattr__(self, "w4a16_block_size_m", block_size_m)
        object.__setattr__(self, "w4a16_fast_math", bool(self.w4a16_fast_math))
        if self.trellis_decode_table not in {"auto", "compact", "full"}:
            raise ValueError("trellis decode table must be auto, compact, or full")
        if self.w4a16_prefill_fused_sum is None:
            from b12x.moe._shared.kernels.w4a16.host import prefill_fused_sum_enabled

            object.__setattr__(
                self, "w4a16_prefill_fused_sum", prefill_fused_sum_enabled()
            )
        object.__setattr__(self, "frozen", bool(self.frozen))

    @property
    def weight_E(self) -> int:
        return self.weight_plan.num_experts

    @property
    def k(self) -> int:
        return self.weight_plan.hidden_size

    @property
    def n(self) -> int:
        return self.weight_plan.intermediate_size

    @property
    def dtype(self) -> torch.dtype:
        try:
            return {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[self.weight_plan.io_dtype]
        except KeyError as exc:
            raise TypeError(
                f"unsupported MoE plan dtype {self.weight_plan.io_dtype!r}"
            ) from exc

    @property
    def activation(self) -> str:
        return self.weight_plan.activation

    @property
    def source_format(self) -> str:
        return self.weight_plan.source_format

    @property
    def w13_layout(self) -> str:
        return self.weight_plan.w13_layout

    @property
    def w4a16_weight_layout(self) -> str | None:
        return self.weight_plan.w4a16_weight_layout

    @property
    def w4a16_scale_format(self) -> str | None:
        return self.weight_plan.w4a16_scale_format


@dataclass(frozen=True)
class TPMoEScratchPlan:
    caps: TPMoEScratchCaps
    layout: TPMoEArenaLayout
    launch_plan: TPMoEPlan
    _core_workspace_plan: _TPCoreWorkspacePlan
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    _prewarmed_fused_launches: tuple[tuple[int, object], ...] = field(
        default=(), repr=False
    )
    _prewarmed_topk_sum_launches: tuple[tuple[torch.dtype, bool, object], ...] = field(
        default=(), repr=False
    )
    _mixed_trellis_launches: tuple[tuple[torch.dtype, bool, bool, object], ...] = field(
        default=(), repr=False
    )
    _prewarmed_route_pack_launches: object | None = field(default=None, repr=False)

    @property
    def full_rotation(self) -> bool:
        """Whether this plan owns the EXL3 full-rotation execution path."""
        return self._core_workspace_plan.full_rotation

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        a: torch.Tensor,
        experts: B12XFP4ExpertWeights,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor | None = None,
        input_scales_static: bool = False,
        fast_math: bool | None = None,
        unit_scale_contract: bool = False,
        activation_amax: torch.Tensor | None = None,
        layer_idx: int | None = None,
        route_expert_map: torch.Tensor | None = None,
        output_expert_map: torch.Tensor | None = None,
        scales_expanded: bool = False,
        _w4a16_launches: object | None = None,
    ) -> "TPMoEFP4Binding":
        """Bind live tensors to this scratch plan.

        ``scales_expanded`` states that ``expand_scales(experts)`` ran on this
        call's stream (or one it waits for) after the last other use of the
        shared NVFP4-CSF scratch; the call then skips its own expansion.
        """
        if not isinstance(experts, B12XFP4ExpertWeights):
            raise TypeError("experts must come from prepare_b12x_fp4_moe_weights")
        weight_plan = experts.plan
        if (
            weight_plan.w4a16_compressed_scales
            and not self.caps.weight_plan.w4a16_compressed_scales
        ):
            # Calls above the stage-scale token limit run on expanded scales.
            weight_plan = replace(weight_plan, w4a16_compressed_scales=False)
        if weight_plan != self.caps.weight_plan:
            raise ValueError(
                "experts do not match the plan used to size TP MoE scratch"
            )
        if int(a.shape[0]) > int(self.caps.max_tokens):
            raise ValueError(
                f"input tokens {int(a.shape[0])} exceed TP MoE scratch capacity "
                f"{int(self.caps.max_tokens)}"
            )
        if int(a.shape[1]) != int(self.caps.k):
            raise ValueError(
                f"input hidden size {int(a.shape[1])} does not match TP MoE "
                f"scratch K={int(self.caps.k)}"
            )
        if int(topk_ids.shape[1]) != int(self.caps.num_topk):
            raise ValueError(
                f"top-k {int(topk_ids.shape[1])} does not match TP MoE scratch "
                f"top-k={int(self.caps.num_topk)}"
            )
        scratch_storage = scratch_tensor(scratch, self._scratch_specs, owner="TP MoE")
        # Eager vLLM bind: MAP caller-owned scratch into per-spec kernel-arg views
        # and build the binding directly. NEVER construct a workspace/arena object
        # (no _materialize_core_arena / _TPCoreArena) and never init/allocate -- the
        # kernel prologue zeros counters/queues, weight_expert_ids is write-first,
        # and the launch wrapper re-zeros the barrier scalars in-place
        # (volatile_launch_state=True on the reconstructed workspaces below). This
        # keeps bind allocation-free / CUDA-graph-capturable, matching the
        # compressed-MLA views-container discipline.
        tensors = _map_core_workspace_views(
            self._core_workspace_plan,
            scratch_storage,
            offset_bytes=self.layout.route_workspace_nbytes,
            capacity_nbytes=self.layout.core_workspace_nbytes,
            do_init=False,
        )
        if self._core_workspace_plan.projection_mixed_trellis:
            return _bind_projection_mixed_trellis_from_views(
                scratch_plan=self,
                tensors=tensors,
                a=a,
                experts=experts,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
                input_scales_static=input_scales_static,
                fast_math=fast_math,
                activation_amax=activation_amax,
                layer_idx=layer_idx,
                route_expert_map=route_expert_map,
                output_expert_map=output_expert_map,
            )
        if fast_math is None and self.caps.quant_mode == "w4a16":
            fast_math = self.caps.w4a16_fast_math
        # Full-rotation Trellis materialization owns its resolved native
        # launchers.  Do not ask a module cache to rediscover them at bind/run
        # time: cache eviction is allowed after trial cleanup while this
        # prepared plan remains leased.  Route-id width and mapped routing
        # are ABI inputs. Prefer exact decode carriers and use the retained
        # prefill capacity for other live row counts.
        fused_launch = None
        topk_sum_launch = None
        route_pack_launches = None
        if (
            self.caps.quant_mode == "w4a16"
            and not self._core_workspace_plan.full_rotation
            and _w4a16_launches is not None
        ):
            select = getattr(_w4a16_launches, "select", None)
            if select is None:
                raise TypeError("prepared W4A16 launch owner has no selector")
            fused_launch, topk_sum_launch, route_pack_launches = select(
                tokens=int(a.shape[0]),
                route_ids_dtype=topk_ids.dtype,
                has_route_map=route_expert_map is not None,
                activation_amax=activation_amax,
            )
        elif (
            self.caps.quant_mode == "w4a16" and self._core_workspace_plan.full_rotation
        ):
            route_pack_launches = self._prewarmed_route_pack_launches
            tokens = int(a.shape[0])
            prepared = experts.representation_for("w4a16")
            broadcast_suh = prepared.gate_suh.numel() == self.caps.k
            broadcast_svh = prepared.down_svh.numel() == self.caps.k
            capacity = min(
                (
                    count
                    for count, _ in self._prewarmed_fused_launches
                    if count >= tokens
                ),
                default=None,
            )
            fused_launch = next(
                (
                    launch
                    for count, launch in self._prewarmed_fused_launches
                    if count == capacity and launch.broadcast_suh == broadcast_suh
                ),
                None,
            )
            topk_sum_launch = next(
                (
                    launch
                    for ids_dtype, mapped, launch in self._prewarmed_topk_sum_launches
                    if ids_dtype == topk_ids.dtype
                    and mapped
                    == (output_expert_map is not None or route_expert_map is not None)
                    and launch.broadcast_svh == broadcast_svh
                ),
                None,
            )
            if fused_launch is None:
                raise RuntimeError(
                    f"W4A16 token count {tokens} exceeds the retained launch capacity"
                )
            if topk_sum_launch is None:
                raise RuntimeError(
                    "W4A16 route-id/map specialization was not materialized"
                )
        binding = _build_tp_moe_fp4_binding_from_views(
            plan=self._core_workspace_plan,
            execution_plan=self.launch_plan,
            tensors=tensors,
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=self.caps.apply_router_weight_on_input,
            output=output,
            input_scales_static=input_scales_static,
            fast_math=fast_math,
            quant_mode=self.caps.quant_mode,
            decode_config=self.caps.decode_config,
            unit_scale_contract=unit_scale_contract,
            swiglu_limit=self.caps.swiglu_limit,
            swiglu_alpha=self.caps.swiglu_alpha,
            swiglu_beta=self.caps.swiglu_beta,
            activation_amax=activation_amax,
            layer_idx=layer_idx,
            route_expert_map=route_expert_map,
            output_expert_map=output_expert_map,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
            route_pack_launches=route_pack_launches,
        )
        return replace(
            binding,
            scales_expanded=bool(scales_expanded),
            w4a8_csf_inline=self.caps.w4a8_csf_inline,
        )


@dataclass(frozen=True, kw_only=True)
class TPMoEFP4Binding:
    a: torch.Tensor
    experts: B12XFP4ExpertWeights
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    implementation: str
    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    dtype: torch.dtype
    execution_plan: TPMoEPlan | None = None
    apply_router_weight_on_input: bool = False
    output: torch.Tensor | None = None
    output_cast_target: torch.Tensor | None = None
    input_scales_static: bool = False
    compact_launches: object | None = None
    fast_math: bool | None = None
    quant_mode: str | None = None
    decode_config: MoeDecodeConfig | None = None
    plan: object = None
    deterministic_output: bool = False
    unit_scale_contract: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    activation_amax: torch.Tensor | None = None
    layer_idx: int | None = None
    row_counts: torch.Tensor | None = None
    token_map: torch.Tensor | None = None
    token_weights: torch.Tensor | None = None
    packed_input: torch.Tensor | None = None
    packed_input_scale: torch.Tensor | None = None
    barrier_count: torch.Tensor | None = None
    barrier_epoch: torch.Tensor | None = None
    packed_a_view: object = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None
    packed_a_storage_ptr: object = None
    routed_rows_capacity: int | None = None
    active_expert_count: torch.Tensor | None = None
    weight_expert_ids: torch.Tensor | None = None
    global_to_local_expert: torch.Tensor | None = None
    compact_topk_ids: torch.Tensor | None = None
    micro_intermediate: torch.Tensor | None = None
    physical_tiles_capacity: int | None = None
    task_capacity: int | None = None
    expert_write_rows: torch.Tensor | None = None
    expert_tile_base: torch.Tensor | None = None
    input_gs: torch.Tensor | None = None
    down_input_scale: torch.Tensor | None = None
    pair_head: torch.Tensor | None = None
    producers_done_count: torch.Tensor | None = None
    all_work_published: torch.Tensor | None = None
    task_head: torch.Tensor | None = None
    task_tail: torch.Tensor | None = None
    task_ready: torch.Tensor | None = None
    task_expert: torch.Tensor | None = None
    task_m_tile: torch.Tensor | None = None
    task_slice_begin: torch.Tensor | None = None
    task_slice_count: torch.Tensor | None = None
    task_valid_rows: torch.Tensor | None = None
    tile_write_count: torch.Tensor | None = None
    route_output: torch.Tensor | None = None
    materialized_intermediate: torch.Tensor | None = None
    intermediate_cache13: torch.Tensor | None = None
    intermediate_cache2: torch.Tensor | None = None
    prefill_sum_accum: torch.Tensor | None = None
    fc1_c_tmp: torch.Tensor | None = None
    fc2_c_tmp: torch.Tensor | None = None
    packed_route_indices: torch.Tensor | None = None
    block_expert_ids: torch.Tensor | None = None
    packed_route_count: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    expert_counts: torch.Tensor | None = None
    rotation_a_gate: torch.Tensor | None = None
    rotation_a_up: torch.Tensor | None = None
    kernel_workspace: torch.Tensor | None = None
    # Preserve the route geometry selected by the caller-owned scratch plan.
    # Re-running the live-batch heuristic here can select a smaller block than
    # the arena was sized for (for example, planned 64 versus live 48).
    route_block_size_m: int | None = None
    route_expert_map: torch.Tensor | None = None
    output_expert_map: torch.Tensor | None = None
    fused_launch: object | None = None
    topk_sum_launch: object | None = None
    route_pack_launches: object | None = None
    # The caller expanded every expert's NVFP4-CSF scales with expand_scales().
    scales_expanded: bool = False
    w4a8_csf_inline: bool = False
    mixed_trellis_binding: object | None = None
    mixed_trellis_buffers: object | None = None

    def run(self) -> torch.Tensor:
        return b12x_moe_fp4(binding=self)


@dataclass(frozen=True, kw_only=True)
class TPMoERouteBinding:
    hidden_states: torch.Tensor
    top_k: int
    scratch: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool | None = None
    gate_weight: torch.Tensor | None = None
    gate_bias: torch.Tensor | None = None
    router_logits: torch.Tensor | None = None
    renormalize: bool = True
    score_func: str = "softmax"
    correction_bias: torch.Tensor | None = None
    image_correction_bias: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None
    routed_scaling_factor: float = 1.0
    plan: object | None = None

    def run(self) -> B12XTopKRouting:
        return b12x_route_experts_fast(binding=self)


@dataclass(frozen=True, kw_only=True)
class TPMoESparseFP4Binding:
    hidden_states: torch.Tensor
    experts: B12XFP4ExpertWeights
    scratch: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool
    routing: B12XTopKRouting | None = None
    top_k: int | None = None
    gate_weight: torch.Tensor | None = None
    gate_bias: torch.Tensor | None = None
    router_logits: torch.Tensor | None = None
    renormalize_topk: bool = True
    score_func: str = "softmax"
    correction_bias: torch.Tensor | None = None
    image_correction_bias: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None
    routed_scaling_factor: float = 1.0
    output: torch.Tensor | None = None
    return_routing: bool = False
    input_scales_static: bool = False
    fast_math: bool | None = None
    quant_mode: str | None = None
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    activation_amax: torch.Tensor | None = None
    layer_idx: int | None = None
    plan: object | None = None

    def run(self) -> torch.Tensor | tuple[torch.Tensor, B12XTopKRouting]:
        return b12x_sparse_moe_fp4(binding=self)


@dataclass(frozen=True, kw_only=True)
class _TPMoEWorkspacePolicy:
    can_chunk: bool


@dataclass
class _WeightViews:
    """Cached weight views for the concatenated expert-weight layout."""

    w13: torch.Tensor  # [2*n, k//2, E] uint8 (permuted view, no copy)
    down: torch.Tensor  # [k, n//2, E] uint8 (permuted view, no copy)
    w13_sf: torch.Tensor  # 6D MMA view for concatenated w13 scale factors
    down_sf: torch.Tensor  # [E, down_sf_rows, sf_cols] uint8 (view)
    w1_alpha: torch.Tensor  # [E] float32 contiguous tensor
    w2_alpha: torch.Tensor  # [E] float32 contiguous tensor
    w1_storage: torch.Tensor  # original [E, w1_n, k//2] tensor for direct micro
    w1_scale_storage: torch.Tensor
    w2_storage: torch.Tensor  # original [E, k, n//2] tensor for direct micro
    w2_scale_storage: torch.Tensor
    # Pre-computed fp4 views and CuTe pointers
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None
    # w4a8 recipe operands (None unless prepared): plain UE8M0 K/32 grids and
    # E4M3 per-K/16 residuals from the NVFP4 scale decomposition.
    sfb_w13_mx: torch.Tensor | None = None
    sfb_down_mx: torch.Tensor | None = None
    w13_residual: torch.Tensor | None = None
    down_residual: torch.Tensor | None = None
    # A gated NVFP4 FC1 whose half boundary lands inside a 32-row MMA scale
    # slice uses separately swizzled scale views for the zero-copy up/gate
    # weight descriptors. The original concatenated view remains authoritative
    # for micro and every naturally aligned dynamic geometry.
    w13_up_sf: torch.Tensor | None = None
    w13_gate_sf: torch.Tensor | None = None
    dynamic_down_fp4: torch.Tensor | None = None
    dynamic_down_sf: torch.Tensor | None = None


@dataclass(frozen=True)
class _PackedInputBindingViews:
    packed_a_view: object
    sfa_ptr: object
    packed_a_flat: torch.Tensor
    scale_flat: torch.Tensor
    packed_a_storage_ptr: object


@dataclass(frozen=True)
class _ActivationKernelSpec:
    activation: str
    is_gated: bool
    micro_kernel_cls: type
    dynamic_kernel_cls: type
    default_w13_layout: str = "w13"

    def w1_rows(self, n: int) -> int:
        return (2 if self.is_gated else 1) * n

    def make_micro_kernel(
        self,
        *,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        **kernel_kwargs,
    ):
        if self.is_gated:
            kernel_kwargs.update(
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
            )
        return self.micro_kernel_cls(**kernel_kwargs)

    def make_dynamic_kernel(
        self,
        *,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        **kernel_kwargs,
    ):
        if self.is_gated:
            kernel_kwargs.update(
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
            )
        return self.dynamic_kernel_cls(**kernel_kwargs)


_ACTIVATION_KERNEL_SPECS = {
    "silu": _ActivationKernelSpec(
        activation="silu",
        is_gated=True,
        micro_kernel_cls=MoEMicroKernelSilu,
        dynamic_kernel_cls=MoEDynamicKernelSilu,
    ),
    SITU: _ActivationKernelSpec(
        activation=SITU,
        is_gated=True,
        micro_kernel_cls=MoEMicroKernelSitu,
        dynamic_kernel_cls=MoEDynamicKernelSitu,
    ),
    SWIGLUOAI_UNINTERLEAVE: _ActivationKernelSpec(
        activation=SWIGLUOAI_UNINTERLEAVE,
        is_gated=True,
        micro_kernel_cls=MoEMicroKernelSwiGLUOAI,
        dynamic_kernel_cls=MoEDynamicKernelSwiGLUOAI,
        default_w13_layout="w31",
    ),
    "relu2": _ActivationKernelSpec(
        activation="relu2",
        is_gated=False,
        micro_kernel_cls=MoEMicroKernelRelu2,
        dynamic_kernel_cls=MoEDynamicKernelRelu2,
    ),
}


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("", "0", "false", "False")


def _dynamic_deterministic_output_enabled(
    *, quant_mode: str, device: torch.device
) -> bool:
    if torch.device(device).type != "cuda":
        return False
    if _normalize_quant_mode(quant_mode) == "w4a16":
        return False
    return _env_flag("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", default=False)


def _dynamic_work_source() -> str:
    """Select the compile-time dynamic-kernel work source.

    ``materialized_queue`` load-balances the fully published work domain and is
    the production default. ``persistent_grid`` provides arithmetic striding
    for controlled A/B measurements. ``ready_queue`` enables the experimental
    overlapped publisher.
    """

    source = (
        os.environ.get(_DYNAMIC_WORK_SOURCE_ENV, _DYNAMIC_WORK_SOURCE_DEFAULT)
        .strip()
        .lower()
    )
    aliases = {
        "grid": "persistent_grid",
        "queue": "materialized_queue",
        "streaming": "ready_queue",
    }
    source = aliases.get(source, source)
    if source not in _DYNAMIC_WORK_SOURCES:
        raise ValueError(
            f"unsupported {_DYNAMIC_WORK_SOURCE_ENV}={source!r}; "
            f"expected one of {sorted(_DYNAMIC_WORK_SOURCES)}"
        )
    return source


def default_moe_quant_mode() -> str:
    return "nvfp4"


_W4A8_QUANT_MODES = {"w4a8_mx", "w4a8_nvfp4"}
# W6A8 MX-FP6 (E2M3 weights x FP8 E4M3 activations on the mxf8f6f4 MMA) and
# W8A8 MXFP8 (E4M3 weights x E4M3 activations on the same m16n8k32 atom) are
# distinct recipe families: neither may satisfy _is_w4a8_quant_mode, whose
# predicates assume FP4 weight packing and the W4A8 repacked layouts.
_W6A8_QUANT_MODES = {"w6a8_mx"}
_W8A8_QUANT_MODES = {"w8a8_mx"}
# Both share the one-byte-per-element activation scratch and UE8M0 K/32 scale
# plane (the FP4 recipes pack two codes per byte and use E4M3 K/16 scales).
_MX_BYTE_CONTAINER_QUANT_MODES = _W6A8_QUANT_MODES | _W8A8_QUANT_MODES


def _is_w4a8_quant_mode(quant_mode: str) -> bool:
    return quant_mode in _W4A8_QUANT_MODES


def _dynamic_kernel_intermediate_size(n: int, quant_mode: str) -> int:
    """Return the intermediate extent consumed by the dynamic kernel.

    Prepared W4A8-MX weights use ceil-tiled N128 storage. The launch treats
    that zero-padded tail as part of the kernel extent, so every plan-time
    geometry and scratch calculation must use the same padded value.
    """

    n = int(n)
    if _normalize_quant_mode(quant_mode) == "w4a8_mx":
        return align_up(n, 128)
    return n


def _micro_scale_format_for_quant_mode(quant_mode: str) -> str:
    quant_mode = _normalize_quant_mode(quant_mode)
    return "e8m0_k32" if quant_mode == "w4a8_mx" else "e4m3_k16"


def _micro_e8m0_scale_layout_for_quant_mode(quant_mode: str) -> str:
    quant_mode = _normalize_quant_mode(quant_mode)
    return "logical" if quant_mode == "w4a8_mx" else "packed"


def _normalize_quant_mode_requested(quant_mode: str | None) -> str:
    """Lower-case a requested quant mode and reject unsupported recipes."""
    if quant_mode is None:
        return "nvfp4"
    normalized = str(quant_mode).lower()
    if (
        normalized
        not in {"nvfp4", "w4a16"} | _W4A8_QUANT_MODES | _MX_BYTE_CONTAINER_QUANT_MODES
    ):
        raise ValueError(f"unsupported quant_mode {quant_mode!r}")
    return normalized


def _normalize_quant_mode(quant_mode: str | None) -> str:
    return _normalize_quant_mode_requested(quant_mode)


def _normalize_fp4_source_format(source_format: str) -> str:
    if source_format.lower() == "mxfp4_native":
        raise ValueError(
            "source_format='mxfp4_native' has been removed; use "
            "source_format='fp4_e8m0_k32' for byte-preserved E8M0 K/32 "
            "scales, or add a real MXFP4 source contract"
        )
    try:
        return _FP4_SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of "
            f"{sorted(_FP4_SOURCE_FORMATS)}, got {source_format!r}"
        ) from exc


def _normalize_quant_mode_for_source(
    quant_mode: str | None,
    source_format: str,
) -> str:
    source_format = _normalize_fp4_source_format(source_format)
    normalized = _normalize_quant_mode_requested(quant_mode)
    _validate_fp4_source_format_for_quant_mode(
        source_format=source_format,
        quant_mode=normalized,
    )
    return normalized


def _normalize_w4a16_scale_format(scale_format: str) -> str:
    try:
        return _W4A16_SCALE_FORMATS[scale_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "scale_format must be one of 'e4m3_k16', 'e4m3_k32', or "
            "'e8m0_k32', "
            f"got {scale_format!r}"
        ) from exc


def _w4a16_scale_format_for_source(source_format: str) -> str:
    source_format = _normalize_fp4_source_format(source_format)
    if source_format in BLOCK_CODECS:
        return source_format
    if source_format in _TRELLIS_SOURCE_FORMATS:
        return "e4m3_k32"
    return "e8m0_k32" if source_format == "fp4_e8m0_k32" else "e4m3_k16"


def _w4a16_weight_layout_for_source(
    source_format: str,
    *,
    intermediate_size: int | None = None,
) -> str:
    """Return the default W4A16 layout selected by weight preparation."""
    source_format = _normalize_fp4_source_format(source_format)
    if source_format in BLOCK_CODECS:
        return source_format
    if source_format in _TRELLIS_SOURCE_FORMATS:
        return "trellis_t256"
    if (
        source_format == "fp4_e8m0_k32"
        and intermediate_size is not None
        and int(intermediate_size) % 128 != 0
    ):
        return "modelopt"
    return "packed"


_W4A16_WEIGHT_LAYOUTS = {
    "packed",
    "modelopt",
    "trellis_t256",
    "iq2_xs",
    "iq2_xxs",
    "q8_0",
}


def _normalize_w4a16_weight_layout(weight_layout: str) -> str:
    layout = str(weight_layout).lower()
    if layout not in _W4A16_WEIGHT_LAYOUTS:
        raise ValueError(
            "weight_layout must be one of "
            f"{sorted(_W4A16_WEIGHT_LAYOUTS)}, got {weight_layout!r}"
        )
    return layout


def _normalize_w13_layout(w13_layout: str) -> str:
    try:
        return _W13_LAYOUTS[w13_layout.lower()]
    except KeyError as exc:
        raise ValueError(
            "w13_layout must be one of 'w13'/'w31' (or the 'up_gate'/'gate_up' "
            f"aliases), got {w13_layout!r}"
        ) from exc


def _normalize_w13_layout_for_activation(activation: str, w13_layout: str) -> str:
    activation = normalize_moe_activation(activation)
    layout = _normalize_w13_layout(w13_layout)
    if activation == SWIGLUOAI_UNINTERLEAVE:
        return "w31"
    return layout


def _normalize_swiglu_params(
    activation: str,
    swiglu_limit: float | None,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
) -> tuple[float | None, float, float]:
    activation = normalize_moe_activation(activation)
    return (
        normalize_swiglu_limit_for_activation(activation, swiglu_limit),
        normalize_swiglu_alpha_for_activation(activation, swiglu_alpha),
        normalize_swiglu_beta_for_activation(activation, swiglu_beta),
    )


def _derive_trellis_codebook(
    source_format: str, trellis_codebook: str | None
) -> str | None:
    """Kernel decode codebook declared by the Trellis weight plan."""

    del source_format
    return trellis_codebook


def _fc_trellis_pair_kind(workspace) -> str | None:
    """Compile-time pair kind from the plan's declared rate summary.

    Compact P33/P43 descriptors serve {P33, P43}; PDYNAMIC mode tables
    serve {P33} and {P33, P24}.
    """

    kinds = workspace.trellis_pair_kinds
    if kinds:
        return "P33_P43" if "P43" in kinds else "PDYNAMIC"
    return None


def _validate_fp4_source_format_for_quant_mode(
    *, source_format: str, quant_mode: str
) -> None:
    """Validate a canonical checkpoint-format/numeric-recipe pair."""

    source_format = _normalize_fp4_source_format(source_format)
    quant_mode = _normalize_quant_mode_requested(quant_mode)
    # The packed MX-FP6 and raw MXFP8 sources are each exclusive to their own
    # recipe, and both are mutually exclusive with every FP4 pairing (including
    # w4a16, which keeps its FP4 trio).
    if source_format == "mxfp6_e2m3" or quant_mode == "w6a8_mx":
        if source_format == "mxfp6_e2m3" and quant_mode == "w6a8_mx":
            return
        raise ValueError(
            f"source_format={source_format!r} with quant_mode={quant_mode!r} "
            "is unsupported; quant_mode='w6a8_mx' requires "
            "source_format='mxfp6_e2m3' and vice versa"
        )
    if source_format == "mxfp8_e8m0_k32" or quant_mode == "w8a8_mx":
        if source_format == "mxfp8_e8m0_k32" and quant_mode == "w8a8_mx":
            return
        raise ValueError(
            f"source_format={source_format!r} with quant_mode={quant_mode!r} "
            "is unsupported; quant_mode='w8a8_mx' requires "
            "source_format='mxfp8_e8m0_k32' and vice versa"
        )
    if quant_mode == "w4a16":
        return
    if source_format == "modelopt_nvfp4":
        return
    if source_format == "fp4_e8m0_k32" and quant_mode == "w4a8_mx":
        return
    if source_format in _TRELLIS_SOURCE_FORMATS and quant_mode == "w4a8_mx":
        return
    raise ValueError(
        f"source_format={source_format!r} with quant_mode={quant_mode!r} is "
        "unsupported; use quant_mode='w4a16' for non-NVFP4 sources, "
        "quant_mode='w4a8_mx' for source_format='fp4_e8m0_k32', or "
        "source_format='modelopt_nvfp4' for NVFP4/W4A8-NVFP4 kernels"
    )


def _get_activation_kernel_spec(
    activation: str,
    *,
    quant_mode: str = "nvfp4",
) -> _ActivationKernelSpec:
    if _normalize_quant_mode(quant_mode) == "w4a16":
        raise ValueError(
            "W4A16 dispatch uses b12x.moe._shared.kernels.w4a16.kernel directly"
        )
    if (
        _is_w4a8_quant_mode(_normalize_quant_mode(quant_mode))
        and activation == SWIGLUOAI_UNINTERLEAVE
    ):
        raise NotImplementedError(
            "activation='swigluoai_uninterleave' is not supported for W4A8 MoE"
        )
    try:
        return _ACTIVATION_KERNEL_SPECS[normalize_moe_activation(activation)]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {activation!r}") from exc


def _activation_w1_rows(activation: str, n: int) -> int:
    return moe_activation_w1_rows(activation, n)


# Override for the dynamic-kernel MMA tile (tile_m, tile_n). Set via the env
# B12X_DYNAMIC_TILE_MN="64x128" or programmatically (tp_moe._DYNAMIC_TILE_MN_OVERRIDE).
# Used to specialize/benchmark dynamic across the tile set it should cover.
_DYNAMIC_TILE_MN_OVERRIDE: Tuple[int, int] | None = None


def _dynamic_tile_mn_override() -> Tuple[int, int] | None:
    env = os.environ.get("B12X_DYNAMIC_TILE_MN")
    if env:
        m, n = (int(x) for x in env.split("x"))
        return (m, n)
    return _DYNAMIC_TILE_MN_OVERRIDE


def _dynamic_tile_n(quant_mode: str = "nvfp4") -> int:
    _normalize_quant_mode(quant_mode)
    ovr = _dynamic_tile_mn_override()
    return ovr[1] if ovr is not None else _LEVEL_TILE_N


def _dynamic_tile_m(quant_mode: str = "nvfp4") -> int:
    _normalize_quant_mode(quant_mode)
    ovr = _dynamic_tile_mn_override()
    return ovr[0] if ovr is not None else _LEVEL_TILE_M


def _select_dynamic_tile_mn(
    routed_rows: int,
    n: int,
    quant_mode: str = "nvfp4",
    *,
    num_experts: int,
    activation: str = "silu",
    compute_capability: tuple[int, int] | None = None,
    planned_tile_m: int | None = None,
) -> Tuple[int, int]:
    """Tile planner for the dynamic kernel.

    Generally keyed on average routed rows per expert, with narrowly measured
    static-shape tactic entries for serving workloads whose route distribution
    is known.  The scratch plan and kernel build both call this with the same
    inputs, so they select the SAME tile -- a mismatch mis-sizes grouped
    task/scale scratch.

    tile_n is fixed at 128 (the verified column). A small tile_m cuts per-expert
    M-tile padding and dead rows in the grouped GEMM for small decode-band
    workloads; 128 amortizes best for large prefill. An explicit
    B12X_DYNAMIC_TILE_MN / programmatic override wins (for benchmarking).
    """
    quant_mode = _normalize_quant_mode(quant_mode)
    ovr = _dynamic_tile_mn_override()
    if ovr is not None:
        return ovr
    if planned_tile_m is not None:
        planned_tile_m = int(planned_tile_m)
        if planned_tile_m not in {16, 32, 64, 128}:
            raise ValueError("planned dynamic tile M must be one of 16, 32, 64, 128")
        if quant_mode in _MX_BYTE_CONTAINER_QUANT_MODES and planned_tile_m != 128:
            raise ValueError(
                "MX byte-container recipes support only the M128 dynamic tile"
            )
        return planned_tile_m, _LEVEL_TILE_N
    routed_rows = max(1, int(routed_rows))
    num_experts = max(1, int(num_experts))
    if quant_mode in _MX_BYTE_CONTAINER_QUANT_MODES:
        # The MX byte-container dynamic kernels build only the (128, 128) MMA
        # tile (their ctor rejects everything else), so the M16-M64 ladder does
        # not exist for w6a8_mx or w8a8_mx.  Every planner/scratch/kernel
        # caller therefore selects the same fixed tile.
        return (_LEVEL_TILE_M, _LEVEL_TILE_N)
    if _is_w4a8_quant_mode(quant_mode):
        if compute_capability is None:
            compute_capability = _current_compute_capability()
        # DSV4-Flash TP2 speculative-verify band.  Like the CUTLASS tactic
        # selector, this is keyed only by the static workload shape and routed
        # row count; live expert counts still come from the device histogram.
        # M32 reduces repeated expert-weight streaming for the measured
        # 64--384-token band without adding a second routing/scheduling path.
        if (
            quant_mode == "w4a8_mx"
            and activation == "silu"
            and num_experts == 256
            and int(n) == 1024
            and 384 <= routed_rows <= 2304
        ):
            return (32, _LEVEL_TILE_N)
        # GB10 has only 48 SMs and substantially less memory bandwidth than
        # desktop SM120 parts.  Its fused M32 specialization keeps FC1/FC2 in
        # one persistent kernel and wins through the measured DSV4 TP2 prefill
        # band; the M64/M128 tactics split routing, FC1, and FC2 into three
        # launches and lose 16--36% here.  Keep the SM120 crossover unchanged.
        if (
            compute_capability == (12, 1)
            and quant_mode == "w4a8_mx"
            and activation == "silu"
            and num_experts == 256
            and int(n) == 1024
            and 2304 < routed_rows <= 384 * num_experts
        ):
            return (32, _LEVEL_TILE_N)
        # W4A8 uses M16/M32 for sparse decode and the split M64 compute path
        # for dense prefill.  DSV4 TP2 graph-replay probes place M32->M64
        # between 34.5 and 36 routed rows/expert (M32 wins at 34.5; M64 wins
        # at 36).  Keep the conservative integer boundary at 36.  M128 remains
        # an explicit research override: its external compute tasks are also
        # M64, while its coarser routing domain adds expert-tail waste.
        if routed_rows >= 36 * num_experts:
            return (64, _LEVEL_TILE_N)
        if routed_rows > 16 * num_experts:
            return (32, _LEVEL_TILE_N)
        return (16, _LEVEL_TILE_N)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    activation = activation_spec.activation
    if activation == "relu2":
        return (_LEVEL_TILE_M, _LEVEL_TILE_N)
    # Gated NVFP4 supports the full M16/M32/M64/M128 ladder.  DSV4 TP2/TP4
    # boundary probes put M16->M32 between 14 and 15 routed rows/expert
    # (M16 wins at 14; M32 wins by 2.7-3.0% at 15), with later crossovers near
    # 48 and 96.  Integer products keep this deterministic at boundaries.
    if routed_rows < 15 * num_experts:
        tile_m = 16
    elif routed_rows < 48 * num_experts:
        tile_m = 32
    elif routed_rows < 96 * num_experts:
        tile_m = 64
    else:
        tile_m = _LEVEL_TILE_M
    if (
        quant_mode == "nvfp4"
        and tile_m == 64
        and activation_spec.is_gated
        and int(n) % 128 != 0
        and int(n) % 32 == 0
    ):
        # The swapped FC1 specialization uses a fixed M32 tile. The M64
        # tactic selects an M64 MMA atom, leaving zero FC1 M tiles and invalid
        # CuTe IR. M32 preserves the exact mid-atom gate-half addressing.
        tile_m = 32
    return (tile_m, _LEVEL_TILE_N)


def _w4a8_dynamic_dense_candidate(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    k: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    """Whether the proven token-major materialized W4A8 regime can run.

    This is intentionally a structural predicate, separate from the feature
    env flags.  All dispatch, preparation, scratch validation, and kernel
    specialization sites use the same predicate so they cannot disagree about
    which representation the launch consumes.
    """

    return bool(
        _normalize_quant_mode(quant_mode) == "w4a8_mx"
        and activation in {"silu", SITU}
        and k % 256 == 0
        # Half-aligned ceil-tiled rp storage keeps the up/gate halves on
        # 128-row tile boundaries, so the per-tile pairing works for any
        # 32-aligned shard (352 = 2048/TP6, 192 = 3072/TP16).
        and n % 32 == 0
        and _select_dynamic_tile_mn(
            routed_rows,
            n,
            quant_mode,
            num_experts=num_experts,
            activation=activation,
            planned_tile_m=planned_tile_m,
        )
        in {(32, 128), (64, 128), (128, 128)}
        and _dynamic_work_source() != "ready_queue"
    )


def _w4a8_dynamic_decode_candidate(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    """Whether prepared W4A8 should use its shared-input decode regime."""

    return bool(
        _normalize_quant_mode(quant_mode) == "w4a8_mx"
        and activation in {"silu", SITU}
        and 0 < routed_rows <= _W4A8_DECODE_MAX_ROUTED_ROWS
        and _select_dynamic_tile_mn(
            routed_rows,
            n,
            quant_mode,
            num_experts=num_experts,
            activation=activation,
            planned_tile_m=planned_tile_m,
        )
        == (16, 128)
        and _dynamic_work_source() != "ready_queue"
        and not deterministic_output
    )


def _w4a8_dynamic_direct_candidate(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    """Whether tiny prepared W4A8 can bypass grouped route compaction."""

    return bool(
        activation == "silu"
        and routed_rows <= _DIRECT_ROUTING_MAX_ROUTED_ROWS
        and _w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=num_experts,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )


def _nvfp4_dynamic_direct_candidate(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    """Whether tiny A4 execution can bypass grouped route compaction."""

    return bool(
        _normalize_quant_mode(quant_mode) == "nvfp4"
        and activation == "silu"
        and 0 < routed_rows <= _DIRECT_ROUTING_MAX_ROUTED_ROWS
        and _select_dynamic_tile_mn(
            routed_rows,
            n,
            quant_mode,
            num_experts=num_experts,
            activation=activation,
            planned_tile_m=planned_tile_m,
        )
        == (16, 128)
        and _dynamic_work_source() != "ready_queue"
        and not deterministic_output
    )


@triton.jit(do_not_specialize=["live_routes"])
def _dynamic_route_plan_kernel(
    topk_ids,
    row_counts,
    expert_tile_base,
    barrier_count,
    barrier_epoch,
    live_routes,
    NUM_EXPERTS: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    """Initialize grouped routing and its resident-grid barrier in one CTA."""
    tl.store(barrier_count, 0)
    tl.store(barrier_epoch, 0)
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    tl.store(row_counts + experts, 0, mask=expert_mask)
    tl.debug_barrier()

    routes = tl.arange(0, BLOCK_ROUTES)
    expert_ids = tl.load(
        topk_ids + routes,
        mask=routes < live_routes,
        other=-1,
    ).to(tl.int32)
    valid = (routes < live_routes) & (expert_ids >= 0) & (expert_ids < NUM_EXPERTS)
    tl.atomic_add(row_counts + expert_ids, 1, mask=valid, sem="relaxed")
    tl.debug_barrier()

    counts = tl.load(row_counts + experts, mask=expert_mask, other=0)
    tiles = (counts + TILE_M - 1) // TILE_M
    tiles = tl.where(expert_mask, tiles, 0)
    inclusive = tl.cumsum(tiles, axis=0)
    tl.store(expert_tile_base + experts, inclusive - tiles, mask=expert_mask)
    tl.store(expert_tile_base + NUM_EXPERTS, tl.sum(tiles, axis=0))


def _dynamic_direct_routing_candidate(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    return bool(
        _w4a8_dynamic_direct_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=num_experts,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
        or _nvfp4_dynamic_direct_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=num_experts,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )


def _dynamic_direct_routing_selected(
    *,
    route_mode: str,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    num_experts: int,
    n: int,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    route_mode = str(route_mode).lower()
    if route_mode not in {"direct", "grouped"}:
        raise ValueError("dynamic route mode must be 'direct' or 'grouped'")
    direct_supported = _dynamic_direct_routing_candidate(
        quant_mode=quant_mode,
        activation=activation,
        routed_rows=routed_rows,
        num_experts=num_experts,
        n=n,
        deterministic_output=deterministic_output,
        planned_tile_m=planned_tile_m,
    )
    if route_mode == "direct" and not direct_supported:
        raise RuntimeError(
            "planned dynamic direct routing is unsupported for this launch shape"
        )
    return route_mode == "direct"


def _dynamic_external_route_plan_supported(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    planned_tile_m: int,
    dynamic_route_mode: str,
    deterministic_output: bool,
    w4a8_n64_repacked: bool = False,
) -> bool:
    return bool(
        (
            _normalize_quant_mode(quant_mode) == "nvfp4"
            or (_normalize_quant_mode(quant_mode) == "w4a8_mx" and w4a8_n64_repacked)
        )
        and activation == "silu"
        and int(planned_tile_m) == 16
        and 0 < int(routed_rows) <= _DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS
        and dynamic_route_mode == "grouped"
        and not deterministic_output
        and _dynamic_work_source() in {_DYNAMIC_WORK_SOURCE_DEFAULT, "persistent_grid"}
    )


def _w4a8_dynamic_materialized_enabled(
    *,
    quant_mode: str,
    activation: str,
    num_tokens: int,
    routed_rows: int,
    num_experts: int,
    k: int,
    n: int,
    w4a8_repacked: bool,
    share_input_across_experts: bool,
    deterministic_output: bool,
    planned_tile_m: int | None = None,
) -> bool:
    """Resolve the complete unified-W4A8 specialization as one decision."""

    dense_candidate = _w4a8_dynamic_dense_candidate(
        quant_mode=quant_mode,
        activation=activation,
        routed_rows=routed_rows,
        num_experts=num_experts,
        k=k,
        n=n,
        deterministic_output=deterministic_output,
        planned_tile_m=planned_tile_m,
    )
    m1_candidate = bool(
        int(num_tokens) == 1
        and k % 256 == 0
        # Half-aligned ceil-tiled rp storage keeps the up/gate halves on
        # 128-row tile boundaries, so the per-tile pairing works for any
        # 32-aligned shard (352 = 2048/TP6, 192 = 3072/TP16).
        and n % 32 == 0
        and _w4a8_dynamic_direct_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=num_experts,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )
    candidate = dense_candidate or m1_candidate
    return bool(
        candidate
        and w4a8_repacked
        and share_input_across_experts
        and _env_flag(_DYNAMIC_W4A8_MATERIALIZED_ENV, default=candidate)
    )


def _dynamic_materialized_intermediate_bytes(
    *, quant_mode: str, rows_padded: int, n: int
) -> int:
    """Return the byte extent of the materialized payload and scale planes."""
    if quant_mode == "nvfp4":
        return rows_padded * (n // 128) * 72
    materialized_n = align_up(n, 128) if _is_w4a8_quant_mode(quant_mode) else n
    return rows_padded * (materialized_n + materialized_n // 32)


_WEIGHT_CACHE: Dict[Tuple[int, int, int], _WeightViews] = {}
# Compiled-kernel memos hold every program a process compiles; the
# preparation session evicts the ones no prepared plan retains, so timing
# trials do not keep their losing candidates resident.
_MICRO_KERNEL_CACHE: Dict[Tuple, object] = {}
_DYNAMIC_KERNEL_CACHE: Dict[Tuple, object] = {}
register_program_cache(_MICRO_KERNEL_CACHE)
register_program_cache(_DYNAMIC_KERNEL_CACHE)
_MAC_CACHE: Dict[Tuple[int, str], int] = {}  # (device_idx, impl) → max_active_clusters
# Micro owns the tiny tail below this routed-row cutover; dynamic owns the rest.
# The measured GLM crossover under CUDA graph replay is 64 routed rows.
_MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT = 64
# Micro keeps the m tokens' activations resident in registers; 8 is the budget
# ceiling. Micro is correct for any 1<=m<=8 (not just powers of two).
_MICRO_MAX_TOKENS = 8
# Prepared MXFP4 decode keeps the shared-input producer through M=8, while
# pair-direct routing wins through M=4 for the common top-k=8 regime.
_W4A8_DECODE_MAX_ROUTED_ROWS = 64
_DIRECT_ROUTING_MAX_ROUTED_ROWS = 32
_MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE: Dict[str, int] = {}
_DYNAMIC_MULTICTA_CACHE: bool | None = None
_DYNAMIC_DOWN_SCALE_CACHE: bool | None = None
_DYNAMIC_SWAP_AB_OVERRIDE = os.environ.get("B12X_DYNAMIC_SWAP_AB")
_LAST_WEIGHTS: Tuple = (None, None)  # (cache_key, views)
_MICRO_DIRECT_LAUNCH_CAP_CACHE: Dict[Tuple[int, int], bool] = {}
_CURRENT_DISPATCH_STAGE: str | None = None
_DIRECT_MICRO_SHAPE_ATTR = "_b12x_direct_micro_shape"


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return int(t._version)
    except RuntimeError:
        # Inference tensors intentionally do not track a version counter.
        # Model weights/scales are expected to be immutable during serving.
        return 0


@contextmanager
def b12x_moe_dispatch_context(stage: str | None):
    global _CURRENT_DISPATCH_STAGE
    previous_stage = _CURRENT_DISPATCH_STAGE
    _CURRENT_DISPATCH_STAGE = stage
    try:
        yield
    finally:
        _CURRENT_DISPATCH_STAGE = previous_stage


def clear_tp_moe_caches() -> None:
    """Clear runtime caches owned by `tp_moe`.

    Explicit workspaces and workspace pools are caller-owned and intentionally
    unaffected by this helper.
    """
    from b12x.moe._shared.kernels.w4a16.kernel import (
        clear_w4a16_kernel_cache,
    )
    from b12x.moe._shared.kernels.w4a8_compact_micro import (
        _compiled_direct_w4a8_compact,
    )

    global _LAST_WEIGHTS
    global _MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE
    global _DYNAMIC_MULTICTA_CACHE
    global _DYNAMIC_DOWN_SCALE_CACHE
    global _DYNAMIC_SWAP_AB_OVERRIDE
    _WEIGHT_CACHE.clear()
    clear_w4a16_kernel_cache()
    _compiled_direct_w4a8_compact.cache_clear()
    _MICRO_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()
    _MAC_CACHE.clear()
    _MICRO_DIRECT_LAUNCH_CAP_CACHE.clear()
    _MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE.clear()
    _DYNAMIC_MULTICTA_CACHE = None
    _DYNAMIC_DOWN_SCALE_CACHE = None
    _DYNAMIC_SWAP_AB_OVERRIDE = os.environ.get("B12X_DYNAMIC_SWAP_AB")
    _LAST_WEIGHTS = (None, None)


_FAST_MATH_DEFAULT = _env_flag("B12X_FAST_MATH", default=True)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _get_micro_dynamic_cutover_pairs(quant_mode: str = "nvfp4") -> int:
    quant_mode = _normalize_quant_mode(quant_mode)
    cached = _MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE.get(quant_mode)
    if cached is None:
        cutover = os.environ.get("B12X_MICRO_DYNAMIC_CUTOVER_PAIRS")
        if cutover is None:
            cached = _MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT
        else:
            cached = max(0, int(cutover))
        _MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE[quant_mode] = cached
    return cached


def _arena_core_token_counts(
    *,
    max_tokens: int,
    num_topk: int,
    core_token_counts: tuple[int, ...] | None,
    quant_mode: str,
    bucket_w4a16_tokens: bool = True,
) -> tuple[int, ...]:
    max_tokens = max(int(max_tokens), 1)
    num_topk = max(int(num_topk), 1)
    quant_mode = _normalize_quant_mode(quant_mode)
    if quant_mode == "w4a16" and bucket_w4a16_tokens:
        from b12x.moe._shared.kernels.w4a16.host import (
            route_pack_token_capacity,
        )

        max_tokens = route_pack_token_capacity(max_tokens, num_topk)
    if core_token_counts is None:
        normalized = (max_tokens,)
    else:
        normalized = tuple(
            max(int(token_count), 1) for token_count in core_token_counts
        )
        if quant_mode == "w4a16" and bucket_w4a16_tokens:
            normalized = tuple(
                route_pack_token_capacity(token_count, num_topk)
                for token_count in normalized
            )
        if max_tokens not in normalized:
            normalized = (max_tokens, *normalized)
    normalized = tuple(dict.fromkeys(normalized))
    micro_cutover_pairs = _get_micro_dynamic_cutover_pairs(quant_mode)
    max_micro_tokens = micro_cutover_pairs // num_topk
    if max_micro_tokens >= 1:
        micro_boundary_tokens = min(max_tokens, max_micro_tokens)
        if quant_mode == "w4a16" and bucket_w4a16_tokens:
            micro_boundary_tokens = route_pack_token_capacity(
                micro_boundary_tokens,
                num_topk,
            )
        if micro_boundary_tokens not in normalized:
            normalized = (*normalized, micro_boundary_tokens)
    return normalized


def _dynamic_multicta_enabled() -> bool:
    global _DYNAMIC_MULTICTA_CACHE
    if _DYNAMIC_MULTICTA_CACHE is None:
        multicta_env = _first_env(
            "B12X_DYNAMIC_ENABLE_MULTICTA",
            "B12X_LEVEL10_ENABLE_MULTICTA",
        )
        if multicta_env is None:
            multicta_env = "1"
        _DYNAMIC_MULTICTA_CACHE = multicta_env == "1"
    return _DYNAMIC_MULTICTA_CACHE


def _dynamic_down_scale_enabled() -> bool:
    global _DYNAMIC_DOWN_SCALE_CACHE
    if _DYNAMIC_DOWN_SCALE_CACHE is None:
        _DYNAMIC_DOWN_SCALE_CACHE = _env_flag(
            "B12X_ENABLE_DYNAMIC_DOWN_SCALE", default=False
        )
    return _DYNAMIC_DOWN_SCALE_CACHE


def _flatten_routing_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    with record_function("tp_moe.flatten_routing_ids"):
        flat_ids = topk_ids.view(-1)
        if flat_ids.dtype not in (torch.int32, torch.int64):
            with record_function("tp_moe.flatten_routing_ids.cast_int32"):
                return flat_ids.to(torch.int32)
        if not flat_ids.is_contiguous():
            with record_function("tp_moe.flatten_routing_ids.contiguous"):
                return flat_ids.contiguous()
        return flat_ids


def _flatten_routing_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    with record_function("tp_moe.flatten_routing_weights"):
        flat_weights = topk_weights.view(-1)
        if flat_weights.dtype != torch.float32:
            with record_function("tp_moe.flatten_routing_weights.cast_fp32"):
                return flat_weights.to(torch.float32)
        if not flat_weights.is_contiguous():
            with record_function("tp_moe.flatten_routing_weights.contiguous"):
                return flat_weights.contiguous()
        return flat_weights


def _prepare_expert_scale(scale: torch.Tensor, weight_E: int) -> torch.Tensor:
    with record_function("tp_moe.prepare_expert_scale"):
        if scale.numel() == 1:
            with record_function("tp_moe.prepare_expert_scale.expand_scalar"):
                return scale.reshape(()).expand(weight_E).to(torch.float32).contiguous()
        if scale.numel() != weight_E:
            raise ValueError(
                f"expected expert scale with {weight_E} elements, got {scale.numel()}"
            )
        # A contiguous [E] scale stays contiguous through reshape + to(float32),
        # so the trailing .contiguous() was redundant here (kept in the scalar
        # .expand() branch above, where expand yields a non-contiguous view).
        return scale.reshape(weight_E).to(torch.float32)


def _prepare_expert_scale_vector(
    scale: torch.Tensor,
    weight_E: int,
    *,
    name: str,
) -> torch.Tensor:
    with record_function("tp_moe.prepare_expert_scale_vector"):
        if scale.numel() == 1:
            with record_function("tp_moe.prepare_expert_scale_vector.expand_scalar"):
                return scale.reshape(()).expand(weight_E).to(torch.float32).contiguous()
        if scale.numel() != weight_E:
            raise ValueError(
                f"expected {name} with {weight_E} elements, got {scale.numel()}"
            )
        return scale.reshape(weight_E).to(torch.float32).contiguous()


def _safe_dynamic_max_rows_per_launch(
    E: int,
    k: int,
    _n: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Largest graph-safe routed-row budget for the compact dynamic workspace.

    Dynamic now stores routed activations in a compact physical-tile pool, so
    the dominant CuTe memref extents scale with `rows_padded` rather than
    `E * max_rows`. Graph-safe chunking still has to budget for the worst-case
    active-expert envelope, so it reserves `E - 1` extra 128-row tiles in that
    large-row regime.
    """
    tile_m = _dynamic_tile_m(quant_mode)
    rows_padded_limit = _dynamic_rows_padded_limit(k, quant_mode=quant_mode)
    extra_rows = max(0, E - 1) * tile_m
    safe_rows = rows_padded_limit - extra_rows
    if safe_rows <= 0:
        return tile_m
    return max(tile_m, safe_rows - (safe_rows % tile_m))


def _dynamic_rows_padded_limit(k: int, *, quant_mode: str = "nvfp4") -> int:
    """Return the largest padded row count whose packed input and scale planes fit
    the runtime memref limit.
    """
    tile_m = _dynamic_tile_m(quant_mode)
    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    _qm = _normalize_quant_mode(quant_mode)
    input_cols = (
        k
        if (
            _qm == "w4a16"
            or _is_w4a8_quant_mode(_qm)
            or _qm in _MX_BYTE_CONTAINER_QUANT_MODES
        )
        else k // 2
    )
    rows_padded_limit = min(
        _RUNTIME_MEMREF_LIMIT // max(1, input_cols),
        _RUNTIME_MEMREF_LIMIT // max(1, cols_pad_k),
    )
    return rows_padded_limit - (rows_padded_limit % tile_m)


def _safe_dynamic_token_chunk(
    E: int,
    k: int,
    n: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Largest token chunk that fits the compact dynamic launch ABI."""
    tile_m = _dynamic_tile_m(quant_mode)
    safe_rows = _safe_dynamic_max_rows_per_launch(E, k, n, quant_mode)
    max_tokens = max(1, safe_rows // max(1, num_topk))
    while max_tokens > 1 and align_up(max_tokens * num_topk, tile_m) > safe_rows:
        max_tokens -= 1
    return max_tokens


def _dynamic_token_chunk_limit(
    E: int,
    k: int,
    n: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> int:
    """Largest token chunk supported by the compact dynamic launch ABI."""
    return _safe_dynamic_token_chunk(E, k, n, num_topk, quant_mode)


def _workspace_policy(
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
) -> _TPMoEWorkspacePolicy:
    is_pool = isinstance(workspace, TPMoEWorkspacePool)
    return _TPMoEWorkspacePolicy(
        can_chunk=is_pool,
    )


def select_tp_moe_backend(
    *,
    num_tokens: int,
    num_topk: int,
    quant_mode: str = "nvfp4",
) -> str:
    """Pick the fused MoE backend from the intrinsic routed workload shape.

    Direct-micro owns the tiny-decode tail where it wins on GPU time; dynamic
    owns all larger routed workloads.
    """
    routed_rows = num_tokens * num_topk
    cutover = _get_micro_dynamic_cutover_pairs(quant_mode)
    if num_tokens <= _MICRO_MAX_TOKENS and routed_rows < cutover:
        return "micro"
    return "dynamic"


def _canonical_moe_policy_source_format(source_format: str) -> str:
    source_format = _normalize_fp4_source_format(source_format)
    if source_format in _TRELLIS_SOURCE_FORMATS:
        return "b12x_trellis"
    return source_format


def _w4a8_mx_micro_supported(
    *,
    num_tokens: int,
    k: int,
    n: int,
    num_topk: int,
    weight_E: int,
    activation: str,
    source_format: str,
) -> bool:
    source_format = _canonical_moe_policy_source_format(source_format)
    if source_format == "b12x_trellis":
        return bool(
            num_tokens <= _MICRO_MAX_TOKENS
            and n % 128 == 0
            and MoEMicroKernelBackend.is_supported(
                m=num_tokens,
                k=k,
                n=n,
                num_topk=num_topk,
                weight_E=weight_E,
            )
        )
    if source_format == "fp4_e8m0_k32" and n % 128 == 64:
        return bool(
            1 <= num_tokens <= _MICRO_MAX_TOKENS
            and k % 256 == 0
            and activation == "silu"
            and 0 < num_topk <= 32
            and weight_E > 0
        )
    return _tiny_decode_enabled() and _tiny_decode_supports(
        num_tokens=num_tokens,
        k=k,
        n=n,
        activation=activation,
    )


def _w4a8_mx_micro_preferred(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
) -> bool:
    if not _w4a8_mx_micro_supported(
        num_tokens=query.num_tokens,
        k=query.hidden_size,
        n=query.intermediate_size,
        num_topk=query.top_k,
        weight_E=query.num_experts,
        activation=query.activation,
        source_format=query.source_format,
    ):
        return False
    capability = (
        device.compute_capability
        if device is not None
        else _current_compute_capability()
    )
    return not (
        capability == (12, 1)
        and query.num_tokens >= 3
        and query.hidden_size == 6144
        and query.intermediate_size == 1024
    )


def _policy_micro_supported(query: MoeDecodeQuery) -> bool:
    if (
        query.quant_mode == "w4a8_nvfp4"
        and query.routed_rows > _DIRECT_ROUTING_MAX_ROUTED_ROWS
    ):
        return False
    if query.quant_mode == "w4a8_mx":
        return _w4a8_mx_micro_supported(
            num_tokens=query.num_tokens,
            k=query.hidden_size,
            n=query.intermediate_size,
            num_topk=query.top_k,
            weight_E=query.num_experts,
            activation=query.activation,
            source_format=query.source_format,
        )
    return _band_runs_direct_micro(
        num_tokens=query.num_tokens,
        k=query.hidden_size,
        n=query.intermediate_size,
        num_topk=query.top_k,
        weight_E=query.num_experts,
        activation=query.activation,
        quant_mode=query.quant_mode,
    )


def _heuristic_dynamic_route_mode(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
    *,
    planned_tile_m: int,
) -> str:
    dynamic_n = _dynamic_kernel_intermediate_size(
        query.intermediate_size,
        query.quant_mode,
    )
    direct = _dynamic_direct_routing_candidate(
        quant_mode=query.quant_mode,
        activation=query.activation,
        routed_rows=query.routed_rows,
        num_experts=query.num_experts,
        n=dynamic_n,
        deterministic_output=False,
        planned_tile_m=planned_tile_m,
    )
    if (
        direct
        and device is not None
        and device.compute_capability == (12, 1)
        and query.quant_mode == "w4a8_mx"
        and query.num_experts == 256
        and dynamic_n == 1024
    ):
        direct = False
    return "direct" if direct else "grouped"


def _w4a16_direct_routing_supported(query: MoeDecodeQuery) -> bool:
    weight_layout = (
        "modelopt"
        if query.quant_mode == "nvfp4_auto"
        else query.w4a16_weight_layout
        or _w4a16_weight_layout_for_source(
            query.source_format,
            intermediate_size=query.intermediate_size,
        )
    )
    if weight_layout in BLOCK_CODECS:
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _TC_DECODE_MAX_M,
        )

        capacity_supported = (
            query.activation in {"silu", "relu2"}
            and query.num_tokens <= _TC_DECODE_MAX_M
        )
        return bool(
            capacity_supported
            and query.io_dtype == "bfloat16"
            and not query.deterministic_output
            and not query.collect_activation_amax
            and not query.apply_router_weight_on_input
        )
    if weight_layout == "trellis_t256":
        return False
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _MAX_DIRECT_TOPK_ROUTE_M,
        _small_m_direct_supported,
    )

    if weight_layout == "modelopt":
        return not query.collect_activation_amax and _small_m_direct_supported(
            m=query.num_tokens,
            hidden_size=query.hidden_size,
            intermediate_size=query.intermediate_size,
            num_experts=query.num_experts,
            topk=query.top_k,
            activation=query.activation,
            apply_router_weight_on_input=query.apply_router_weight_on_input,
            swiglu_limit=query.swiglu_limit,
            swiglu_alpha=query.swiglu_alpha,
            swiglu_beta=query.swiglu_beta,
            element_dtype="bf16" if query.io_dtype == "bfloat16" else "fp16",
            weight_layout=weight_layout,
            w13_layout=query.w13_layout,
            scale_format=query.w4a16_scale_format
            or _w4a16_scale_format_for_source(query.source_format),
        )
    # Planned launches carry the unmapped direct top-k kernel only up to
    # _MAX_DIRECT_TOPK_ROUTE_M rows; the TC-decode variant that would serve
    # larger small-M calls is not a planned launch, so a declaration above
    # that limit cannot honor a direct route mode.
    return bool(query.num_tokens <= _MAX_DIRECT_TOPK_ROUTE_M)


def _heuristic_w4a16_route_mode(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
) -> str:
    if query.source_format in BLOCK_CODECS:
        return (
            "direct"
            if query.num_tokens == 1
            and query.w4a16_block_size_m in (None, 8)
            and _w4a16_direct_routing_supported(query)
            else "packed"
        )
    if not _w4a16_direct_routing_supported(query):
        return "packed"
    if (
        _w4a16_weight_layout_for_source(
            query.source_format,
            intermediate_size=query.intermediate_size,
        )
        == "modelopt"
    ):
        return "direct"
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _MAX_DIRECT_TOPK_ROUTE_M,
        _TC_DECODE_MAX_M,
        _w4a16_tc_decode_preferred,
    )

    if query.num_tokens <= _MAX_DIRECT_TOPK_ROUTE_M:
        return "direct"
    assert query.num_tokens <= _TC_DECODE_MAX_M
    sms = device.sm_count if device is not None else get_num_sm(torch.device("cuda"))
    return (
        "direct"
        if _w4a16_tc_decode_preferred(
            m=query.num_tokens,
            topk=query.top_k,
            num_experts=query.num_experts,
            sms=sms,
        )
        else "packed"
    )


def _heuristic_moe_decode_config(
    query: MoeDecodeQuery,
    device: DeviceIdentity | None,
) -> MoeDecodeConfig:
    """Choose the backend, tile, and route mode for a decode query without tuning
    data.
    """
    if query.quant_mode == "nvfp4_auto":
        if (
            device is not None
            and device.compute_capability in ((12, 0), (12, 1))
            and 1 <= query.num_tokens <= 8
            and _w4a16_direct_routing_supported(query)
        ):
            return MoeDecodeConfig(
                backend="w4a16",
                route_planner="internal",
                max_active_clusters=None,
                w4a16_route_mode="direct",
            )
        return _heuristic_moe_decode_config(replace(query, quant_mode="nvfp4"), device)
    if query.quant_mode == "w4a16":
        return MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=None,
            dynamic_route_mode=None,
            w4a16_route_mode=_heuristic_w4a16_route_mode(query, device),
        )
    if query.quant_mode in _MX_BYTE_CONTAINER_QUANT_MODES:
        # The MX byte-container recipes exist only in the unified dynamic
        # kernel; there is no micro specialization for them.
        backend = "dynamic"
    elif query.quant_mode == "w4a8_mx":
        backend = "micro" if _w4a8_mx_micro_preferred(query, device) else "dynamic"
    else:
        backend = select_tp_moe_backend(
            num_tokens=query.num_tokens,
            num_topk=query.top_k,
            quant_mode=query.quant_mode,
        )
        if backend == "micro" and not _policy_micro_supported(query):
            backend = "dynamic"
        if (
            backend == "micro"
            and query.quant_mode == "w4a8_nvfp4"
            and query.routed_rows > _DIRECT_ROUTING_MAX_ROUTED_ROWS
        ):
            backend = "dynamic"
    dynamic_tile_m = None
    dynamic_route_mode = None
    if backend == "dynamic":
        compact_n64_pipeline = bool(
            query.quant_mode == "w4a8_mx"
            and query.source_format == "fp4_e8m0_k32"
            and query.intermediate_size % 128 == 64
        )
        if compact_n64_pipeline:
            # Larger compact launches use the split-materialized M16 kernels.
            # The direct compact micro path owns one through eight live tokens.
            dynamic_tile_m = 16
            dynamic_route_mode = "grouped"
        else:
            dynamic_n = _dynamic_kernel_intermediate_size(
                query.intermediate_size,
                query.quant_mode,
            )
            dynamic_tile_m = _select_dynamic_tile_mn(
                query.routed_rows,
                dynamic_n,
                query.quant_mode,
                num_experts=query.num_experts,
                activation=query.activation,
                compute_capability=(
                    None if device is None else device.compute_capability
                ),
            )[0]
            dynamic_route_mode = _heuristic_dynamic_route_mode(
                query,
                device,
                planned_tile_m=dynamic_tile_m,
            )
    return MoeDecodeConfig(
        backend=backend,
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=dynamic_tile_m,
        dynamic_route_mode=dynamic_route_mode,
        w4a16_route_mode=None,
    )


def _dynamic_task_geometry(
    E: int,
    n: int,
    routed_rows: int,
    tile_m: int = _LEVEL_TILE_M,
    tile_n: int = _LEVEL_TILE_N,
) -> tuple[int, int, int]:
    routed_rows = max(1, routed_rows)
    base_m_tiles = align_up(routed_rows, tile_m) // tile_m
    # At most one new physical tile is introduced per active expert beyond the
    # first, and the routed workload cannot touch more experts than routed rows.
    active_expert_upper_bound = min(E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + tile_n - 1) // tile_n)
    slice_groups = max(
        1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def _refresh_dynamic_workspace_scales(
    workspace: TPDynamicWorkspace,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    input_scales_static: bool,
    force: bool = False,
) -> None:
    a1_src_ptr = a1_gscale.data_ptr()
    a2_src_ptr = a2_gscale.data_ptr()
    # Canonical per-expert vectors can be consumed directly. Their owner stays
    # alive through the binding, and live value updates need no staging copy.
    if (
        workspace.input_gs.data_ptr() == a1_src_ptr
        and workspace.down_input_scale.data_ptr() == a2_src_ptr
    ):
        workspace.input_gs_src_ptr = a1_src_ptr
        workspace.down_input_scale_src_ptr = a2_src_ptr
        return
    if (
        force
        or not input_scales_static
        or workspace.input_gs_src_ptr != a1_src_ptr
        or workspace.down_input_scale_src_ptr != a2_src_ptr
    ):
        workspace.input_gs.copy_(a1_gscale.expand(workspace.weight_E))
        workspace.down_input_scale.copy_(a2_gscale.expand(workspace.weight_E))
        workspace.input_gs_src_ptr = a1_src_ptr if input_scales_static else 0
        workspace.down_input_scale_src_ptr = a2_src_ptr if input_scales_static else 0


def _packed_input_binding_views(
    *,
    packed_input: torch.Tensor,
    packed_input_scale: torch.Tensor,
) -> _PackedInputBindingViews:
    sf_dtype = cutlass.Float8E4M3FN
    # Keep as uint8 — the float4 element type is conveyed to CUTLASS via
    # _gptr / compile-time dtype, and dlpack does not support float4.
    return _PackedInputBindingViews(
        packed_a_view=packed_input.permute(1, 2, 0),
        packed_a_flat=packed_input.view(-1),
        scale_flat=packed_input_scale.view(-1),
        sfa_ptr=make_ptr(
            sf_dtype,
            packed_input_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        packed_a_storage_ptr=make_ptr(
            cutlass.Uint8,
            packed_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
    )


def _finalize_workspace_views(workspace: TPMoEWorkspace) -> None:
    views = _packed_input_binding_views(
        packed_input=workspace.packed_input,
        packed_input_scale=workspace.packed_input_scale,
    )
    workspace.packed_a_view = views.packed_a_view
    workspace.packed_a_flat = views.packed_a_flat
    workspace.scale_flat = views.scale_flat
    workspace.sfa_ptr = views.sfa_ptr
    workspace.packed_a_storage_ptr = views.packed_a_storage_ptr


def _build_tp_moe_fp4_binding_from_views(
    *,
    plan: _TPCoreWorkspacePlan,
    execution_plan: TPMoEPlan,
    tensors: Dict[str, torch.Tensor],
    a: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    output: torch.Tensor | None = None,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    quant_mode: str | None = None,
    decode_config: MoeDecodeConfig | None = None,
    unit_scale_contract: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
    route_expert_map: torch.Tensor | None = None,
    output_expert_map: torch.Tensor | None = None,
    fused_launch: object | None = None,
    topk_sum_launch: object | None = None,
    route_pack_launches: object | None = None,
) -> TPMoEFP4Binding:
    if not isinstance(experts, B12XFP4ExpertWeights):
        raise TypeError("experts must come from prepare_b12x_fp4_moe_weights")
    if a.ndim != 2:
        raise ValueError(
            f"expected input activations with rank 2, got {tuple(a.shape)}"
        )
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("topk_weights and topk_ids must be rank-2 tensors")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids shape mismatch: "
            f"{tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
        )
    if topk_ids.shape[0] != a.shape[0]:
        raise ValueError(
            f"routing batch mismatch: expected {a.shape[0]}, got {topk_ids.shape[0]}"
        )

    source_format = experts.source_format
    quant_mode = _normalize_quant_mode_for_source(
        quant_mode if quant_mode is not None else plan.quant_mode,
        source_format,
    )
    unit_scale_contract = bool(unit_scale_contract and quant_mode == "w4a16")
    activation = experts.activation
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    w13_layout = experts.w13_layout
    _validate_fp4_source_format_for_quant_mode(
        source_format=source_format,
        quant_mode=quant_mode,
    )
    if quant_mode != plan.quant_mode:
        raise ValueError(
            f"scratch plan quant_mode={plan.quant_mode!r} cannot bind "
            f"quant_mode={quant_mode!r}"
        )
    plan_activation = activation
    if quant_mode != "w4a16":
        plan_activation = _get_activation_kernel_spec(
            activation,
            quant_mode=quant_mode,
        ).activation
    if plan_activation != plan.activation:
        raise ValueError(
            f"scratch plan activation={plan.activation!r} cannot bind "
            f"activation={activation!r}"
        )
    if (
        swiglu_limit != plan.swiglu_limit
        or swiglu_alpha != plan.swiglu_alpha
        or swiglu_beta != plan.swiglu_beta
    ):
        raise ValueError(
            "scratch plan activation params do not match binding params: "
            f"planned=(limit={plan.swiglu_limit}, alpha={plan.swiglu_alpha}, beta={plan.swiglu_beta}), "
            f"binding=(limit={swiglu_limit}, alpha={swiglu_alpha}, beta={swiglu_beta})"
        )

    m, k = map(int, a.shape)
    _prepared_payload_for_runtime(
        experts,
        quant_mode=quant_mode,
        source_format=source_format,
        activation=activation,
        w13_layout=w13_layout,
        dtype=a.dtype,
        hidden_size=k,
    )
    num_topk = int(topk_ids.shape[1])
    weight_E = experts.num_experts
    n = experts.intermediate_size
    if k != plan.k:
        raise ValueError(f"scratch plan K={plan.k} cannot bind input K={k}")
    if num_topk != plan.num_topk:
        raise ValueError(
            f"scratch plan top-k={plan.num_topk} cannot bind top-k={num_topk}"
        )
    if weight_E != plan.weight_E:
        raise ValueError(
            f"scratch plan weight_E={plan.weight_E} cannot bind weight_E={weight_E}"
        )
    if n != plan.n:
        raise ValueError(f"scratch plan N={plan.n} cannot bind N={n}")
    if m * num_topk > plan.routed_rows:
        raise ValueError(
            f"scratch plan routed-row capacity {plan.routed_rows} cannot bind "
            f"{m * num_topk} routed rows"
        )
    if output_expert_map is not None and not plan.full_rotation:
        raise ValueError("output_expert_map requires a full-rotation Trellis plan")
    if route_expert_map is not None and plan.implementation != "w4a16":
        raise ValueError("route_expert_map is only supported for W4A16 plans")
    for name, expert_map in (
        ("route_expert_map", route_expert_map),
        ("output_expert_map", output_expert_map),
    ):
        if expert_map is None:
            continue
        if (
            expert_map.dtype != torch.int32
            or expert_map.device != a.device
            or tuple(expert_map.shape) != (plan.route_E,)
            or not expert_map.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous int32[{plan.route_E}] on {a.device}"
            )
    output_cast_target = None
    if plan.full_rotation:
        if topk_weights.dtype != torch.float32:
            raise TypeError("full-rotation Trellis topk_weights must be float32")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("full-rotation Trellis topk_ids must be int32 or int64")
        if output is None:
            output = tensors["full_rotation_output"][:m]
        elif (
            output.device != a.device
            or output.ndim != 2
            or tuple(output.shape)
            not in {(m, k), tuple(tensors["full_rotation_output"].shape)}
            or not output.is_contiguous()
            or output.dtype not in {torch.float32, a.dtype}
        ):
            raise ValueError(
                "full-rotation Trellis output must be a contiguous live or "
                "capacity buffer in FP32 or the input dtype on the input device"
            )
        if output.dtype == a.dtype:
            output_cast_target = output[:m]
            output = tensors["full_rotation_output"][:m]
        else:
            output = output[:m]
        # The persistent fused kernel uses this scratch for grid-barrier state.
        # Unlike the ordinary W4A16 prepared object, this arena is caller-owned
        # and may contain arbitrary bytes on its first bind. Record one zero-fill
        # in the same stream as the launch; when bind occurs during CUDA graph
        # capture the initialization is replayed with the graph as required.
        tensors["kernel_workspace"].zero_()

    common_kwargs = dict(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        implementation=plan.implementation,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
        execution_plan=execution_plan,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        output=output,
        output_cast_target=output_cast_target,
        input_scales_static=bool(input_scales_static),
        fast_math=fast_math,
        quant_mode=quant_mode,
        decode_config=decode_config,
        deterministic_output=plan.deterministic_output,
        unit_scale_contract=unit_scale_contract,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
    )
    if plan.implementation == "w4a16":
        return TPMoEFP4Binding(
            **common_kwargs,
            routed_rows_capacity=plan.routed_rows,
            intermediate_cache13=tensors["intermediate_cache13"],
            intermediate_cache2=tensors["intermediate_cache2"],
            prefill_sum_accum=tensors.get("prefill_sum_accum"),
            fc1_c_tmp=tensors["fc1_c_tmp"],
            fc2_c_tmp=tensors["fc2_c_tmp"],
            packed_route_indices=tensors["packed_route_indices"],
            block_expert_ids=tensors["block_expert_ids"],
            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            expert_counts=tensors.get("expert_counts"),
            rotation_a_gate=tensors.get("rotation_a_gate"),
            rotation_a_up=(
                tensors.get("rotation_a_gate")
                if plan.intermediate_hadamard
                else tensors.get("rotation_a_up")
            ),
            kernel_workspace=tensors.get("kernel_workspace"),
            route_block_size_m=plan.route_block_size_m,
            route_expert_map=route_expert_map,
            output_expert_map=output_expert_map,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
            route_pack_launches=route_pack_launches,
        )

    if plan.implementation == "micro":
        view_kwargs = _packed_input_binding_views(
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
        )
        return TPMoEFP4Binding(
            **common_kwargs,
            row_counts=tensors["row_counts"],
            token_map=tensors["token_map"],
            token_weights=tensors["token_weights"],
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
            barrier_count=tensors["barrier_count"],
            barrier_epoch=tensors["barrier_epoch"],
            routed_rows_capacity=plan.routed_rows,
            packed_a_view=view_kwargs.packed_a_view,
            sfa_ptr=view_kwargs.sfa_ptr,
            packed_a_flat=view_kwargs.packed_a_flat,
            scale_flat=view_kwargs.scale_flat,
            packed_a_storage_ptr=view_kwargs.packed_a_storage_ptr,
            active_expert_count=tensors["active_expert_count"],
            weight_expert_ids=tensors["weight_expert_ids"],
            global_to_local_expert=tensors["global_to_local_expert"],
            compact_topk_ids=tensors["compact_topk_ids"],
            micro_intermediate=tensors["micro_intermediate"],
        )

    if plan.implementation == "dynamic":
        if plan.dynamic_physical_tiles is None or plan.dynamic_task_capacity is None:
            raise RuntimeError("dynamic TP MoE binding plan is missing capacities")
        direct_scales = all(
            scale.dtype == torch.float32
            and scale.device == a.device
            and scale.numel() == plan.weight_E
            and scale.is_contiguous()
            for scale in (experts.a1_gscale, experts.a2_gscale)
        )
        # Bind maps views only. Scalar/strided scales are expanded into the
        # caller's scratch by the run-time refresh before their consumer.
        input_gs = (
            experts.a1_gscale.view(plan.weight_E)
            if direct_scales
            else tensors["input_gs"]
        )
        down_input_scale = (
            experts.a2_gscale.view(plan.weight_E)
            if direct_scales
            else tensors["down_input_scale"]
        )
        view_kwargs = _packed_input_binding_views(
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
        )
        return TPMoEFP4Binding(
            **common_kwargs,
            row_counts=tensors["row_counts"],
            token_map=tensors["token_map"],
            token_weights=tensors["token_weights"],
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
            barrier_count=tensors["barrier_count"],
            barrier_epoch=tensors["barrier_epoch"],
            routed_rows_capacity=plan.routed_rows,
            packed_a_view=view_kwargs.packed_a_view,
            sfa_ptr=view_kwargs.sfa_ptr,
            packed_a_flat=view_kwargs.packed_a_flat,
            scale_flat=view_kwargs.scale_flat,
            packed_a_storage_ptr=view_kwargs.packed_a_storage_ptr,
            physical_tiles_capacity=plan.dynamic_physical_tiles,
            task_capacity=plan.dynamic_task_capacity,
            route_output=tensors["route_output"],
            materialized_intermediate=tensors["materialized_intermediate"],
            expert_write_rows=tensors["expert_write_rows"],
            expert_tile_base=tensors["expert_tile_base"],
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            pair_head=tensors["pair_head"],
            producers_done_count=tensors["producers_done_count"],
            all_work_published=tensors["all_work_published"],
            task_head=tensors["task_head"],
            task_tail=tensors["task_tail"],
            task_ready=tensors["task_ready"],
            task_expert=tensors["task_expert"],
            task_m_tile=tensors["task_m_tile"],
            task_slice_begin=tensors["task_slice_begin"],
            task_slice_count=tensors["task_slice_count"],
            task_valid_rows=tensors["task_valid_rows"],
            tile_write_count=tensors["tile_write_count"],
        )

    raise ValueError(
        f"unsupported TP MoE binding implementation {plan.implementation!r}"
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return dtype.itemsize


def _tensor_numel(shape: Tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _plan_core_workspace(
    implementation: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    routed_rows: int,
    max_rows: int,
    activation: str = "silu",
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
    dynamic_tile_m: int | None = None,
    dynamic_tile_n: int | None = None,
    dynamic_route_mode: str | None = None,
    nvfp4_materialize_intermediate: bool = False,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    w4a16_weight_layout: str | None = None,
    w4a16_scale_format: str | None = None,
    route_num_experts: int | None = None,
    w4a16_block_size_m: int | None = None,
    w4a16_prefill_fused_sum: bool | None = None,
    trellis_bits: int = 3,
    trellis_tile_config: tuple[int, int, int, int] | None = None,
    trellis_pair_kinds: frozenset[str] | None = None,
    trellis_codebook: str | None = None,
    intermediate_hadamard: bool = False,
    projection_mixed_trellis: bool = False,
    apply_router_weight_on_input: bool = False,
    collect_activation_amax: bool = False,
    deterministic_output: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> _TPCoreWorkspacePlan:
    """Size every scratch buffer the selected MoE implementation needs for this
    geometry.
    """
    source_format = _normalize_fp4_source_format(source_format)
    quant_mode = _normalize_quant_mode_for_source(quant_mode, source_format)
    activation = normalize_moe_activation(activation)
    deterministic_output = bool(deterministic_output and implementation == "dynamic")
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if implementation == "w4a16":
        from b12x.moe._shared.kernels.w4a16.host import (
            max_packed_route_slots,
            packed_gemm_scratch_elements,
            prefill_fused_sum_eligible,
            route_block_sizes_for_capacity,
            select_route_block_size_m,
        )
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _MAX_DIRECT_TOPK_ROUTE_M,
            _TC_DECODE_MAX_M,
            _small_m_direct_supported,
        )

        w13_layout = _normalize_w13_layout(w13_layout)
        scale_format = (
            _normalize_w4a16_scale_format(w4a16_scale_format)
            if w4a16_scale_format is not None
            else _w4a16_scale_format_for_source(source_format)
        )
        weight_layout = (
            _normalize_w4a16_weight_layout(w4a16_weight_layout)
            if w4a16_weight_layout is not None
            else _w4a16_weight_layout_for_source(
                source_format,
                intermediate_size=n,
            )
        )
        requested_route_E = int(route_num_experts or weight_E)
        full_rotation = weight_layout == "trellis_t256"
        # Route metadata must address every physical weight and every global
        # router ID accepted through route_expert_map.
        route_E = max(int(weight_E), requested_route_E)
        if full_rotation:
            if source_format not in _TRELLIS_SOURCE_FORMATS:
                raise ValueError(
                    "trellis_t256 workspace requires a trellis source format"
                )
            if int(k) % 128 != 0 or int(n) % 128 != 0:
                raise ValueError(
                    "full-rotation Trellis requires hidden and intermediate "
                    "dimensions divisible by 128"
                )
            if int(trellis_bits) not in (2, 3, 4, 5, 6):
                raise ValueError("trellis_bits must be one of 2, 3, 4, 5, 6")
            trellis_tile_config = trellis_tile_config or (64, 256, 64, 256)
            if trellis_pair_kinds and (int(trellis_bits) != 3 or int(n) != 256):
                raise ValueError(
                    "per-expert-pair exl3 extents require n=256 and trellis_bits=3"
                )
        if projection_mixed_trellis:
            if not full_rotation or trellis_codebook != "mcg":
                raise ValueError("projection-mixed Trellis requires native MCG weights")
            if activation != "silu":
                raise NotImplementedError(
                    "projection-mixed Trellis execution currently requires silu"
                )
            if apply_router_weight_on_input:
                raise NotImplementedError(
                    "projection-mixed Trellis does not apply router weights on input"
                )
            if intermediate_hadamard:
                raise ValueError(
                    "projection-mixed Trellis does not support an expert transform"
                )
            if route_E != int(weight_E):
                raise NotImplementedError(
                    "projection-mixed Trellis currently requires the route and "
                    "weight expert namespaces to match"
                )
            if int(weight_E) > 256:
                raise ValueError(
                    "projection-mixed Trellis supports at most 256 experts"
                )
            routed_capacity = max(int(routed_rows), 1)
            topk = max(int(num_topk), 1)
            token_capacity = (routed_capacity + topk - 1) // topk
            block_size_m = (
                int(w4a16_block_size_m)
                if w4a16_block_size_m is not None
                else select_route_block_size_m(token_capacity, topk, route_E)
            )
            # Projection-mixed MCG always includes a K5 tier. Its fixed
            # 128x128 tile exceeds the SM120/SM121 opt-in shared-memory limit
            # at a 64-row route block, while the 48-row specialization fits.
            block_size_m = min(
                block_size_m,
                _PROJECTION_MIXED_TRELLIS_MAX_ROUTE_BLOCK_SIZE,
            )
            route_slots_capacity = max_packed_route_slots(
                routed_capacity,
                block_size_m,
                route_E,
            )
            route_blocks_capacity = (
                route_slots_capacity + block_size_m - 1
            ) // block_size_m
            sms = max(1, int(get_num_sm(device)))
            fc1_cols = 2 * int(n)
            tensor_specs = (
                _TensorAllocSpec(
                    "intermediate_cache13",
                    (routed_capacity * fc1_cols,),
                    torch.float16,
                ),
                _TensorAllocSpec(
                    "intermediate_cache2",
                    (routed_capacity * int(n),),
                    torch.float16,
                ),
                _TensorAllocSpec(
                    "fc1_c_tmp",
                    (
                        packed_gemm_scratch_elements(
                            size_n=fc1_cols,
                            route_slots=route_slots_capacity,
                            moe_block_size=block_size_m,
                            sms=sms,
                        ),
                    ),
                    torch.float32,
                ),
                _TensorAllocSpec(
                    "fc2_c_tmp",
                    (
                        packed_gemm_scratch_elements(
                            size_n=int(k),
                            route_slots=route_slots_capacity,
                            moe_block_size=block_size_m,
                            sms=sms,
                        ),
                    ),
                    torch.float32,
                ),
                _TensorAllocSpec(
                    "packed_route_indices", (route_slots_capacity,), torch.int32
                ),
                _TensorAllocSpec(
                    "block_expert_ids", (route_blocks_capacity,), torch.int32
                ),
                _TensorAllocSpec("packed_route_count", (1,), torch.int32),
                _TensorAllocSpec("expert_offsets", (route_E + 1,), torch.int32),
                _TensorAllocSpec("expert_counts", (route_E,), torch.int32),
                _TensorAllocSpec(
                    "full_rotation_output",
                    (token_capacity, int(k)),
                    torch.float32,
                ),
                _TensorAllocSpec(
                    "rotation_a_gate",
                    (routed_capacity, int(k)),
                    torch.float16,
                ),
                _TensorAllocSpec(
                    "rotation_a_up",
                    (routed_capacity, int(k)),
                    torch.float16,
                ),
                _TensorAllocSpec(
                    "kernel_workspace",
                    (sms * 4 + 2,),
                    torch.int32,
                    init="zeros",
                ),
            )
            return _TPCoreWorkspacePlan(
                implementation="trellis_mixed3",
                quant_mode=quant_mode,
                activation=activation,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                state_E=state_E,
                weight_E=weight_E,
                route_E=route_E,
                routed_rows=routed_capacity,
                max_rows=max(max_rows, routed_capacity),
                k=k,
                n=n,
                num_topk=num_topk,
                device=device,
                dtype=dtype,
                deterministic_output=False,
                full_rotation=True,
                projection_mixed_trellis=True,
                trellis_bits=3,
                trellis_tile_config=(128, 128, 128, 128),
                trellis_codebook="mcg",
                intermediate_hadamard=False,
                route_block_size_m=block_size_m,
                tensor_specs=tensor_specs,
            )
        routed_capacity = max(int(routed_rows), 1)
        fc1_cols = _activation_w1_rows(activation, int(n))
        route_slots_capacity = 1
        route_blocks_capacity = 1
        fc1_c_tmp_elements = 1
        fc2_c_tmp_elements = 1
        sms = max(1, int(get_num_sm(device)))
        topk = max(int(num_topk), 1)
        token_capacity = (routed_capacity + topk - 1) // topk
        direct_route_slots_by_block: dict[int, int] = {}
        if weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}:
            direct_m_cap = _MAX_DIRECT_TOPK_ROUTE_M
            if dtype == torch.bfloat16 and (
                is_gated_moe_activation(activation)
                or (weight_layout in BLOCK_CODECS and activation == "relu2")
            ):
                direct_m_cap = _TC_DECODE_MAX_M
            for direct_m in range(1, min(token_capacity, direct_m_cap) + 1):
                direct_block_size = (
                    int(w4a16_block_size_m)
                    if w4a16_block_size_m is not None
                    else select_route_block_size_m(direct_m, topk, route_E)
                )
                direct_route_slots_by_block[direct_block_size] = max(
                    direct_route_slots_by_block.get(direct_block_size, 0),
                    direct_m * topk * direct_block_size,
                )
        block_sizes = (
            (int(w4a16_block_size_m),)
            if w4a16_block_size_m is not None
            else route_block_sizes_for_capacity(
                max_tokens=token_capacity,
                topk=topk,
                num_experts=route_E,
            )
        )
        for block_size in block_sizes:
            route_slots = max_packed_route_slots(
                routed_capacity,
                int(block_size),
                route_E,
            )
            scratch_route_slots = max(
                route_slots,
                direct_route_slots_by_block.get(int(block_size), 0),
            )
            route_blocks = (route_slots + int(block_size) - 1) // int(block_size)
            route_slots_capacity = max(route_slots_capacity, route_slots)
            route_blocks_capacity = max(route_blocks_capacity, route_blocks)
            fc1_c_tmp_elements = max(
                fc1_c_tmp_elements,
                packed_gemm_scratch_elements(
                    size_n=fc1_cols,
                    route_slots=scratch_route_slots,
                    moe_block_size=int(block_size),
                    sms=sms,
                    weight_layout=weight_layout,
                ),
            )
            fc2_c_tmp_elements = max(
                fc2_c_tmp_elements,
                packed_gemm_scratch_elements(
                    size_n=int(k),
                    route_slots=scratch_route_slots,
                    moe_block_size=int(block_size),
                    sms=sms,
                    weight_layout=weight_layout,
                ),
            )
        # Packed BF16 decode uses a direct top-k route for every supported
        # small-M shape. Unlike expert-packed routing, each routed row owns a
        # whole M block, so its GEMM scratch capacity is
        # ``m * topk * block_size``. This can exceed the expert-pack upper
        # bound when a compact tier has fewer experts than top-k (for example,
        # one kept expert with top-k=8). Frozen workspaces prewarm every
        # direct-decode shape up to their token capacity, so budget each exact
        # route geometry here as well.
        direct_decode_capacity = routed_capacity // max(int(num_topk), 1)
        if (
            weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
            and dtype == torch.bfloat16
            and is_gated_moe_activation(activation)
            and direct_decode_capacity >= 1
        ):
            for direct_m in range(
                1,
                min(direct_decode_capacity, _TC_DECODE_MAX_M) + 1,
            ):
                direct_block_size = (
                    int(w4a16_block_size_m)
                    if w4a16_block_size_m is not None
                    else select_route_block_size_m(
                        direct_m,
                        int(num_topk),
                        route_E,
                    )
                )
                direct_route_slots = direct_m * int(num_topk) * direct_block_size
                fc1_c_tmp_elements = max(
                    fc1_c_tmp_elements,
                    packed_gemm_scratch_elements(
                        size_n=fc1_cols,
                        route_slots=direct_route_slots,
                        moe_block_size=direct_block_size,
                        sms=sms,
                        weight_layout=weight_layout,
                    ),
                )
                fc2_c_tmp_elements = max(
                    fc2_c_tmp_elements,
                    packed_gemm_scratch_elements(
                        size_n=int(k),
                        route_slots=direct_route_slots,
                        moe_block_size=direct_block_size,
                        sms=sms,
                        weight_layout=weight_layout,
                    ),
                )
        intermediate_cache2_elements = routed_capacity * int(n)
        direct_m = routed_capacity // max(int(num_topk), 1)
        if (
            not full_rotation
            and routed_capacity == direct_m * int(num_topk)
            and _small_m_direct_supported(
                m=direct_m,
                hidden_size=int(k),
                intermediate_size=int(n),
                num_experts=route_E,
                topk=int(num_topk),
                activation=activation,
                apply_router_weight_on_input=bool(apply_router_weight_on_input),
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                element_dtype=_w4a16_element_dtype(dtype),
                weight_layout=weight_layout,
                w13_layout=w13_layout,
                scale_format=scale_format,
                expert_map=None,
            )
        ):
            fc2_n_chunks = ((int(n) // 2) + 127) // 128
            direct_cache2_u32 = direct_m * fc2_n_chunks * 128 * int(num_topk)
            direct_cache2_nbytes = direct_cache2_u32 * _dtype_nbytes(torch.uint32)
            intermediate_cache2_elements = max(
                intermediate_cache2_elements,
                align_up(direct_cache2_nbytes, _dtype_nbytes(dtype))
                // _dtype_nbytes(dtype),
            )
        cache_dtype = torch.float16 if full_rotation else dtype
        use_prefill_fused_sum = prefill_fused_sum_eligible(
            dtype=dtype,
            m=token_capacity,
            full_rotation=full_rotation,
            weight_layout=weight_layout,
            collect_activation_amax=collect_activation_amax,
            enabled=w4a16_prefill_fused_sum,
        )
        intermediate_cache13_elements = (
            max(
                routed_capacity * fc1_cols,
                token_capacity * int(k),
            )
            if use_prefill_fused_sum
            else routed_capacity * max(fc1_cols, int(k))
        )
        tensor_specs = [
            _TensorAllocSpec(
                "intermediate_cache13",
                (intermediate_cache13_elements,),
                cache_dtype,
            ),
            _TensorAllocSpec(
                "intermediate_cache2",
                (intermediate_cache2_elements,),
                cache_dtype,
            ),
            _TensorAllocSpec("fc1_c_tmp", (fc1_c_tmp_elements,), torch.float32),
            _TensorAllocSpec("fc2_c_tmp", (fc2_c_tmp_elements,), torch.float32),
            _TensorAllocSpec(
                "packed_route_indices", (route_slots_capacity,), torch.int32
            ),
            _TensorAllocSpec("block_expert_ids", (route_blocks_capacity,), torch.int32),
            _TensorAllocSpec("packed_route_count", (1,), torch.int32),
            _TensorAllocSpec("expert_offsets", (route_E + 1,), torch.int32),
            _TensorAllocSpec("expert_counts", (route_E,), torch.int32),
        ]
        if use_prefill_fused_sum:
            tensor_specs.append(
                _TensorAllocSpec(
                    "prefill_sum_accum",
                    (token_capacity * int(k),),
                    torch.float32,
                )
            )
        if full_rotation:
            max_tokens = max(routed_capacity // max(int(num_topk), 1), 1)
            tensor_specs.extend(
                (
                    _TensorAllocSpec(
                        "full_rotation_output",
                        (max_tokens, int(k)),
                        torch.float32,
                    ),
                    _TensorAllocSpec(
                        "rotation_a_gate",
                        (routed_capacity, int(k)),
                        torch.float16,
                    ),
                    _TensorAllocSpec("kernel_workspace", (sms * 4 + 2,), torch.int32),
                )
            )
            if not intermediate_hadamard:
                tensor_specs.append(
                    _TensorAllocSpec(
                        "rotation_a_up",
                        (routed_capacity, int(k)),
                        torch.float16,
                    )
                )
        return _TPCoreWorkspacePlan(
            implementation=implementation,
            quant_mode=quant_mode,
            activation=activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            state_E=state_E,
            weight_E=weight_E,
            route_E=route_E,
            routed_rows=routed_capacity,
            max_rows=max(max_rows, routed_capacity),
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=dtype,
            deterministic_output=False,
            full_rotation=full_rotation,
            trellis_bits=int(trellis_bits),
            trellis_tile_config=trellis_tile_config,
            trellis_pair_kinds=trellis_pair_kinds,
            trellis_codebook=trellis_codebook,
            intermediate_hadamard=bool(intermediate_hadamard),
            route_block_size_m=w4a16_block_size_m,
            prefill_fused_sum_fp32=use_prefill_fused_sum,
            tensor_specs=tuple(tensor_specs),
        )

    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)

    cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)
    direct_micro_tokens = max(1, routed_rows // max(1, num_topk))
    direct_micro_k_supported = (
        k > 0
        and k % _NVFP4_BLOCK_SIZE == 0
        and k % 128 == 0
        and _direct_k_segments_supported(_direct_k_segments_for_k(k))
    )
    direct_micro_token_supported = 1 <= direct_micro_tokens <= _MICRO_MAX_TOKENS
    direct_micro_candidate = (
        implementation == "micro"
        and n % _NVFP4_BLOCK_SIZE == 0
        and direct_micro_k_supported
        and 0 < num_topk <= 32
        and weight_E > 0
        and routed_rows == direct_micro_tokens * num_topk
        and direct_micro_token_supported
    )
    barrier_slots = max(1, routed_rows)
    if direct_micro_candidate:
        barrier_slots = max(barrier_slots, routed_rows + direct_micro_tokens * 16)
    common_specs = (
        _TensorAllocSpec("row_counts", (state_E,), torch.int32, init="zeros"),
        _TensorAllocSpec("barrier_count", (barrier_slots,), torch.int32, init="zeros"),
        _TensorAllocSpec("barrier_epoch", (barrier_slots,), torch.int32, init="zeros"),
    )
    if implementation == "micro":
        micro_rows_pad_k = align_up(max_rows, 128)
        packed_input_shape = (state_E, max_rows, k // 2)
        packed_input_dtype = torch.uint8
        micro_intermediate_elements = state_E * n
        if quant_mode == "w4a8_mx":
            # tiny_decode stages fp32 [routed_rows, 2n] gate/up sums here.
            micro_intermediate_elements = max(
                micro_intermediate_elements, max_rows * 2 * n
            )
            if n % 128 == 0:
                # Trellis-native micro decode: per-token f16-pair
                # intermediate words plus the FC1 K-split scratch tail
                # (2*i_chunk floats and one arrival counter per base task,
                # i_chunk = 128). The workspace is zero-initialized and the
                # kernel restores the tail to zero after each use.
                micro_intermediate_elements = max(
                    micro_intermediate_elements,
                    direct_micro_tokens * num_topk * (n // 2)
                    + direct_micro_tokens * num_topk * (n // 128) * 257,
                )
            if source_format == "fp4_e8m0_k32" and n % 128 == 64:
                from b12x.moe._shared.kernels.w4a8_compact_micro import (
                    micro_scratch_nbytes,
                )

                micro_intermediate_elements = max(
                    micro_intermediate_elements,
                    (
                        micro_scratch_nbytes(
                            direct_micro_tokens,
                            k,
                            n,
                            num_topk,
                        )
                        + 3
                    )
                    // 4,
                )
        if direct_micro_candidate:
            fc2_n_chunks = (n // 2 + 127) // 128
            micro_intermediate_elements = max(
                micro_intermediate_elements,
                direct_micro_tokens * num_topk * k
                + direct_micro_tokens * num_topk * fc2_n_chunks * 128,
            )
        return _TPCoreWorkspacePlan(
            implementation=implementation,
            quant_mode=quant_mode,
            activation=activation_spec.activation,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            state_E=state_E,
            weight_E=weight_E,
            route_E=weight_E,
            routed_rows=routed_rows,
            max_rows=max_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            dtype=dtype,
            deterministic_output=False,
            tensor_specs=common_specs
            + (
                _TensorAllocSpec(
                    "token_map", (state_E, max_rows), torch.int32, init="zeros"
                ),
                _TensorAllocSpec(
                    "token_weights", (state_E, max_rows), torch.float32, init="zeros"
                ),
                _TensorAllocSpec(
                    "packed_input", packed_input_shape, packed_input_dtype
                ),
                _TensorAllocSpec(
                    "packed_input_scale",
                    (state_E, micro_rows_pad_k, cols_pad_k),
                    torch.uint8,
                ),
                _TensorAllocSpec(
                    "active_expert_count", (1,), torch.int32, init="zeros"
                ),
                _TensorAllocSpec(
                    "weight_expert_ids", (state_E,), torch.int32, init="arange"
                ),
                _TensorAllocSpec("global_to_local_expert", (weight_E,), torch.int32),
                _TensorAllocSpec("compact_topk_ids", (routed_rows,), torch.int32),
                _TensorAllocSpec(
                    "micro_intermediate",
                    (micro_intermediate_elements,),
                    torch.float32,
                    init="zeros",
                ),
            ),
        )

    # Tile planner: must match the kernel's choice (keyed identically on the
    # workspace capacity) so the grouped task/scale scratch is sized correctly.
    dynamic_kernel_n = _dynamic_kernel_intermediate_size(n, quant_mode)
    dynamic_tile_m, selected_tile_n = _select_dynamic_tile_mn(
        routed_rows,
        dynamic_kernel_n,
        quant_mode,
        num_experts=state_E,
        activation=activation_spec.activation,
        planned_tile_m=dynamic_tile_m,
    )
    if dynamic_tile_n is not None and int(dynamic_tile_n) != selected_tile_n:
        raise ValueError(
            f"planned dynamic tile N={dynamic_tile_n} does not match "
            f"supported tile N={selected_tile_n}"
        )
    dynamic_tile_n = selected_tile_n
    if dynamic_physical_tiles is None or dynamic_task_capacity is None:
        dynamic_tiles, _, dynamic_max_tasks = _dynamic_task_geometry(
            state_E,
            dynamic_kernel_n,
            routed_rows,
            tile_m=dynamic_tile_m,
            tile_n=dynamic_tile_n,
        )
        if _dynamic_direct_routing_selected(
            route_mode=dynamic_route_mode or "grouped",
            quant_mode=quant_mode,
            activation=activation_spec.activation,
            routed_rows=routed_rows,
            num_experts=state_E,
            n=dynamic_kernel_n,
            deterministic_output=False,
            planned_tile_m=dynamic_tile_m,
        ):
            direct_groups = max(
                1,
                (
                    (dynamic_kernel_n + dynamic_tile_n - 1) // dynamic_tile_n
                    + _DYNAMIC_SLICE_CHUNK
                    - 1
                )
                // _DYNAMIC_SLICE_CHUNK,
            )
            dynamic_tiles = max(dynamic_tiles, routed_rows)
            dynamic_max_tasks = max(dynamic_max_tasks, routed_rows * direct_groups)
    else:
        dynamic_tiles = dynamic_physical_tiles
        dynamic_max_tasks = dynamic_task_capacity
    dynamic_rows_padded = dynamic_tiles * dynamic_tile_m
    # NVFP4 scale addressing uses 128-row hardware SF atoms even when the
    # compute tile is M16/M32/M64.  The final partial atom therefore needs
    # physical storage through its full 128-row extent.  W4A8 uses a plain
    # row-major scale plane, for which the extra tail is harmless.
    dynamic_scale_rows = align_up(dynamic_rows_padded, 128)
    # w4a8 and w6a8 both stage one E4M3 byte per activation element; nvfp4
    # packs two FP4 elements per byte.
    packed_input_cols = (
        k
        if (
            _is_w4a8_quant_mode(quant_mode)
            or quant_mode in _MX_BYTE_CONTAINER_QUANT_MODES
        )
        else k // 2
    )
    packed_input_shape = (1, dynamic_rows_padded, packed_input_cols)
    packed_input_dtype = torch.uint8
    # Atomic scatter writes directly into the caller-owned [M, K] output and
    # never addresses route_output.  Reserve only one ABI-compatible row for
    # that production-default policy; the opt-in deterministic reduction owns
    # a distinct fixed-capacity plan with one row per routed pair.
    route_output_rows = max(int(routed_rows), 1) if deterministic_output else 1
    # Materialized W4A8 phase 1 writes one E4M3 byte per activation plus one
    # UE8M0 byte per K32 block.  This storage must not alias route_output:
    # deterministic phase 2 writes one BF16 row per token-major route there
    # before the fixed-order top-k reduction.  Keep a small aligned sentinel
    # for unsupported recipes and geometry so every binding has an invariant ABI.
    # The selected plan exposes NVFP4 payload and scales through its scratch
    # specifications. The caller allocates that extent before binding; bind and
    # run only map/use the supplied storage.
    # Layout: [physical_row][n//128 * 16] payload u32, followed by
    # [n//128][rows_capacity][2] scale u32: rows_padded * (n//128) * 72 bytes.
    materialized_intermediate_bytes = 16
    if _is_w4a8_quant_mode(quant_mode):
        materialized_intermediate_bytes = max(
            16,
            _dynamic_materialized_intermediate_bytes(
                quant_mode=quant_mode,
                rows_padded=dynamic_rows_padded,
                n=dynamic_kernel_n,
            ),
        )
    elif quant_mode == "nvfp4" and nvfp4_materialize_intermediate:
        materialized_intermediate_bytes = max(
            16,
            _dynamic_materialized_intermediate_bytes(
                quant_mode=quant_mode,
                rows_padded=dynamic_rows_padded,
                n=int(n),
            ),
        )
    materialized_intermediate_rows = max(
        1,
        (materialized_intermediate_bytes + int(k) * int(dtype.itemsize) - 1)
        // (int(k) * int(dtype.itemsize)),
    )
    return _TPCoreWorkspacePlan(
        implementation=implementation,
        quant_mode=quant_mode,
        activation=activation_spec.activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        state_E=state_E,
        weight_E=weight_E,
        route_E=weight_E,
        routed_rows=routed_rows,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        deterministic_output=deterministic_output,
        dynamic_physical_tiles=dynamic_tiles,
        dynamic_task_capacity=dynamic_max_tasks,
        dynamic_tile_m=dynamic_tile_m,
        dynamic_tile_n=dynamic_tile_n,
        dynamic_route_mode=dynamic_route_mode,
        tensor_specs=common_specs
        + (
            _TensorAllocSpec(
                "token_map", (dynamic_rows_padded,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "token_weights", (dynamic_rows_padded,), torch.float32, init="zeros"
            ),
            _TensorAllocSpec("route_output", (route_output_rows, int(k)), dtype),
            _TensorAllocSpec(
                "materialized_intermediate",
                (materialized_intermediate_rows, int(k)),
                dtype,
            ),
            _TensorAllocSpec("packed_input", packed_input_shape, packed_input_dtype),
            _TensorAllocSpec(
                "packed_input_scale", (dynamic_scale_rows, cols_pad_k), torch.uint8
            ),
            _TensorAllocSpec(
                "expert_write_rows", (state_E,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "expert_tile_base", (state_E + 1,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec("input_gs", (weight_E,), torch.float32),
            _TensorAllocSpec("down_input_scale", (weight_E,), torch.float32),
            _TensorAllocSpec("pair_head", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("producers_done_count", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("all_work_published", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("task_head", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec("task_tail", (1,), torch.int32, init="zeros"),
            _TensorAllocSpec(
                "task_ready", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_expert", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_m_tile", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_slice_begin", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_slice_count", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "task_valid_rows", (dynamic_max_tasks,), torch.int32, init="zeros"
            ),
            _TensorAllocSpec(
                "tile_write_count", (dynamic_tiles,), torch.int32, init="zeros"
            ),
        ),
    )


def _allocate_arena_tensor(
    shared_arena: torch.Tensor,
    offset: int,
    spec: _TensorAllocSpec,
    *,
    do_init: bool = True,
) -> tuple[torch.Tensor, int]:
    alignment = max(16, _dtype_nbytes(spec.dtype))
    offset = align_up(offset, alignment)
    nbytes = _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    storage = shared_arena.narrow(0, offset, nbytes)
    if spec.dtype == torch.uint8:
        tensor = storage.view(spec.shape)
    else:
        tensor = storage.view(spec.dtype).view(spec.shape)
    # The vLLM eager-bind path passes do_init=False: a binding must only MAP
    # caller-owned scratch into views and must never write/allocate (any
    # allocation -- e.g. the init="arange" temp -- is illegal under CUDA graph
    # capture). Per-call state is instead owned by the kernel: the persistent MoE
    # kernel zeros its counters/queues in its Phase-0 prologue and writes
    # weight_expert_ids/token_map write-first, and the launch wrapper re-zeros the
    # read-before-write barrier scalars in-place (gated on volatile_launch_state).
    # The sglang workspace/pool path keeps do_init=True (one-time, pre-capture).
    if do_init:
        if spec.init == "zeros":
            tensor.zero_()
        elif spec.init == "arange":
            tensor.copy_(
                torch.arange(
                    tensor.numel(), dtype=tensor.dtype, device=tensor.device
                ).view(spec.shape)
            )
        elif spec.init != "empty":
            raise ValueError(f"unsupported tensor init mode {spec.init!r}")
    return tensor, offset + nbytes


def _core_workspace_nbytes(plan: _TPCoreWorkspacePlan) -> int:
    arena_nbytes = 0
    for spec in plan.tensor_specs:
        arena_nbytes = align_up(arena_nbytes, max(16, _dtype_nbytes(spec.dtype)))
        arena_nbytes += _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    return int(arena_nbytes)


def _emit_core_workspace_stats(
    plan: _TPCoreWorkspacePlan,
    *,
    storage: str,
    required_nbytes: int,
    capacity_nbytes: int | None = None,
) -> None:
    return


def _map_core_workspace_views(
    plan: _TPCoreWorkspacePlan,
    shared_arena: torch.Tensor,
    *,
    offset_bytes: int = 0,
    capacity_nbytes: int | None = None,
    do_init: bool = True,
) -> Dict[str, torch.Tensor]:
    """Map caller-owned scratch into the per-spec kernel-arg views (no arena/workspace
    object). With do_init=False this is the vLLM eager-bind primitive: pure
    narrow()+view() at computed offsets, zero allocation, zero init writes."""
    arena_nbytes = _core_workspace_nbytes(plan)
    offset_bytes = int(offset_bytes)
    if capacity_nbytes is None:
        capacity_nbytes = shared_arena.numel() - offset_bytes
    if capacity_nbytes < arena_nbytes:
        raise ValueError(
            f"MoE core arena requires {arena_nbytes} bytes, but only {capacity_nbytes} are available"
        )
    relative_offset = 0
    tensors: Dict[str, torch.Tensor] = {}
    for spec in plan.tensor_specs:
        tensor, absolute_next = _allocate_arena_tensor(
            shared_arena,
            offset_bytes + relative_offset,
            spec,
            do_init=do_init,
        )
        tensors[spec.name] = tensor
        relative_offset = absolute_next - offset_bytes
    return tensors


def _materialize_core_arena(
    plan: _TPCoreWorkspacePlan,
    shared_arena: torch.Tensor,
    *,
    offset_bytes: int = 0,
    capacity_nbytes: int | None = None,
) -> _TPCoreArena:
    # SGLANG workspace/arena materialization (one-time, pre-capture; init writes
    # allowed). The vLLM eager bind path must NEVER call this -- it maps views via
    # _map_core_workspace_views and builds the binding directly, never owning or
    # constructing a workspace/arena object.
    tensors = _map_core_workspace_views(
        plan,
        shared_arena,
        offset_bytes=offset_bytes,
        capacity_nbytes=capacity_nbytes,
        do_init=True,
    )
    return _TPCoreArena(plan=plan, shared_arena=shared_arena, tensors=tensors)


def _allocate_core_arena(plan: _TPCoreWorkspacePlan) -> _TPCoreArena:
    arena_nbytes = _core_workspace_nbytes(plan)
    shared_arena = torch.empty(
        arena_nbytes,
        dtype=torch.uint8,
        device=plan.device,
    )
    arena = _materialize_core_arena(plan, shared_arena)
    _emit_core_workspace_stats(
        plan,
        storage="standalone",
        required_nbytes=arena_nbytes,
    )
    return arena


def _materialize_workspace_from_core_arena(
    plan: _TPCoreWorkspacePlan,
    arena: _TPCoreArena,
    *,
    a1_gscale: torch.Tensor | None,
    a2_gscale: torch.Tensor | None,
    input_scales_static: bool,
    volatile_launch_state: bool = False,
) -> TPMoEWorkspace | TPW4A16Workspace:
    tensors = arena.tensors
    if plan.implementation == "w4a16":
        return TPW4A16Workspace(
            implementation=plan.implementation,
            quant_mode=plan.quant_mode,
            activation=plan.activation,
            planned_swiglu_limit=plan.swiglu_limit,
            planned_swiglu_alpha=plan.swiglu_alpha,
            planned_swiglu_beta=plan.swiglu_beta,
            state_E=plan.state_E,
            weight_E=plan.weight_E,
            route_E=plan.route_E,
            max_rows=plan.max_rows,
            k=plan.k,
            n=plan.n,
            num_topk=plan.num_topk,
            device=plan.device,
            dtype=plan.dtype,
            routed_rows_capacity=plan.routed_rows,
            intermediate_cache13=tensors["intermediate_cache13"],
            intermediate_cache2=tensors["intermediate_cache2"],
            prefill_sum_accum=tensors.get("prefill_sum_accum"),
            fc1_c_tmp=tensors["fc1_c_tmp"],
            fc2_c_tmp=tensors["fc2_c_tmp"],
            packed_route_indices=tensors["packed_route_indices"],
            block_expert_ids=tensors["block_expert_ids"],
            packed_route_count=tensors["packed_route_count"],
            expert_offsets=tensors["expert_offsets"],
            expert_counts=tensors.get("expert_counts"),
            full_rotation_output=tensors.get("full_rotation_output"),
            rotation_a_gate=tensors.get("rotation_a_gate"),
            rotation_a_up=(
                tensors.get("rotation_a_gate")
                if plan.intermediate_hadamard
                else tensors.get("rotation_a_up")
            ),
            kernel_workspace=tensors.get("kernel_workspace"),
            full_rotation=plan.full_rotation,
            trellis_bits=plan.trellis_bits,
            trellis_tile_config=plan.trellis_tile_config,
            trellis_pair_kinds=plan.trellis_pair_kinds,
            trellis_codebook=plan.trellis_codebook,
            intermediate_hadamard=plan.intermediate_hadamard,
            route_block_size_m=plan.route_block_size_m,
            planned_prefill_fused_sum_fp32=plan.prefill_fused_sum_fp32,
            volatile_launch_state=bool(volatile_launch_state),
        )
    if a1_gscale is None or a2_gscale is None:
        raise ValueError("NVFP4 workspace materialization requires input scale tensors")

    common_kwargs = dict(
        implementation=plan.implementation,
        quant_mode=plan.quant_mode,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
        row_counts=tensors["row_counts"],
        barrier_count=tensors["barrier_count"],
        barrier_epoch=tensors["barrier_epoch"],
        volatile_launch_state=bool(volatile_launch_state),
    )
    if plan.implementation == "micro":
        workspace = TPMicroWorkspace(
            **common_kwargs,
            routed_rows_capacity=plan.routed_rows,
            token_map=tensors["token_map"],
            token_weights=tensors["token_weights"],
            packed_input=tensors["packed_input"],
            packed_input_scale=tensors["packed_input_scale"],
            active_expert_count=tensors["active_expert_count"],
            weight_expert_ids=tensors["weight_expert_ids"],
            global_to_local_expert=tensors["global_to_local_expert"],
            compact_topk_ids=tensors["compact_topk_ids"],
            micro_intermediate=tensors["micro_intermediate"],
        )
        _finalize_workspace_views(workspace)
        return workspace

    assert plan.dynamic_physical_tiles is not None
    assert plan.dynamic_task_capacity is not None
    workspace = TPDynamicWorkspace(
        **common_kwargs,
        routed_rows_capacity=plan.routed_rows,
        physical_tiles_capacity=plan.dynamic_physical_tiles,
        task_capacity=plan.dynamic_task_capacity,
        route_output=tensors["route_output"],
        materialized_intermediate=tensors["materialized_intermediate"],
        token_map=tensors["token_map"],
        token_weights=tensors["token_weights"],
        packed_input=tensors["packed_input"],
        packed_input_scale=tensors["packed_input_scale"],
        expert_write_rows=tensors["expert_write_rows"],
        expert_tile_base=tensors["expert_tile_base"],
        input_gs=tensors["input_gs"],
        down_input_scale=tensors["down_input_scale"],
        pair_head=tensors["pair_head"],
        producers_done_count=tensors["producers_done_count"],
        all_work_published=tensors["all_work_published"],
        task_head=tensors["task_head"],
        task_tail=tensors["task_tail"],
        task_ready=tensors["task_ready"],
        task_expert=tensors["task_expert"],
        task_m_tile=tensors["task_m_tile"],
        task_slice_begin=tensors["task_slice_begin"],
        task_slice_count=tensors["task_slice_count"],
        task_valid_rows=tensors["task_valid_rows"],
        tile_write_count=tensors["tile_write_count"],
    )
    _refresh_dynamic_workspace_scales(
        workspace,
        a1_gscale,
        a2_gscale,
        input_scales_static=input_scales_static,
        force=volatile_launch_state,
    )
    _finalize_workspace_views(workspace)
    return workspace


def _alloc_workspace(
    implementation: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    routed_rows: int,
    max_rows: int,
    input_scales_static: bool,
    deterministic_output: bool = False,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    dynamic_physical_tiles: int | None = None,
    dynamic_task_capacity: int | None = None,
    dynamic_tile_m: int | None = None,
    dynamic_tile_n: int | None = None,
    dynamic_route_mode: str | None = None,
    nvfp4_materialize_intermediate: bool = False,
    pool: TPMoEWorkspacePool | None = None,
    storage_key: tuple | None = None,
) -> TPMoEWorkspace | TPW4A16Workspace:
    plan = _plan_core_workspace(
        implementation,
        quant_mode,
        state_E,
        weight_E,
        k,
        n,
        num_topk,
        device,
        dtype,
        routed_rows=routed_rows,
        max_rows=max_rows,
        deterministic_output=deterministic_output,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        dynamic_physical_tiles=dynamic_physical_tiles,
        dynamic_task_capacity=dynamic_task_capacity,
        dynamic_tile_m=dynamic_tile_m,
        dynamic_tile_n=dynamic_tile_n,
        dynamic_route_mode=dynamic_route_mode,
        nvfp4_materialize_intermediate=nvfp4_materialize_intermediate,
    )
    if pool is not None:
        if storage_key is None:
            raise ValueError(
                "storage_key is required when allocating from a workspace pool"
            )
        arena = pool.core_arenas.get(storage_key)
        if arena is None or arena.plan != plan:
            if pool.shared_arena is None:
                arena = _allocate_core_arena(plan)
            else:
                if pool.shared_arena.device != plan.device:
                    raise ValueError(
                        f"MoE pool arena device {pool.shared_arena.device} does not match plan device {plan.device}"
                    )
                arena = _materialize_core_arena(
                    plan,
                    pool.shared_arena,
                    offset_bytes=pool.core_arena_offset_bytes,
                    capacity_nbytes=pool.core_arena_nbytes,
                )
                _emit_core_workspace_stats(
                    plan,
                    storage="shared",
                    required_nbytes=_core_workspace_nbytes(plan),
                    capacity_nbytes=pool.core_arena_nbytes,
                )
            pool.core_arenas[storage_key] = arena
    else:
        arena = _allocate_core_arena(plan)
    return _materialize_workspace_from_core_arena(
        plan,
        arena,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        input_scales_static=input_scales_static,
        volatile_launch_state=bool(pool is not None and pool.shared_arena is not None),
    )


def _unswizzle_block_scales_batched(
    swizzled_u8: torch.Tensor, rows: int, cols_blocks: int
) -> torch.Tensor:
    """Invert the FlashInfer vec16 scale swizzle for a stack of experts.

    Batched equivalent of ``reference.unswizzle_block_scale``; returns f32
    ``[E, rows, cols_blocks]``.
    """
    E = swizzled_u8.shape[0]
    rows_pad = align_up(rows, 128)
    cols_pad = align_up(cols_blocks, 4)
    s = swizzled_u8.reshape(E, rows_pad // 128, cols_pad // 4, 32, 4, 4)
    s = s.permute(0, 1, 4, 3, 2, 5).reshape(E, rows_pad, cols_pad)
    return s[:, :rows, :cols_blocks].view(torch.float8_e4m3fn).to(torch.float32)


def _split_gated_nvfp4_scale_views(
    swizzled_u8: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build independently atom-aligned scale views for gated FC1 halves."""
    rows = 2 * n
    cols_blocks = k // _NVFP4_BLOCK_SIZE
    logical_u8 = _unswizzle_block_scale_bytes(
        swizzled_u8,
        rows,
        cols_blocks,
    )

    def make_view(row_begin: int) -> torch.Tensor:
        logical_half = logical_u8[
            :, row_begin : row_begin + n, :cols_blocks
        ].contiguous()
        storage = swizzle_block_scale(logical_half.view(torch.float8_e4m3fn)).view(
            torch.uint8
        )
        return as_grouped_scale_view(storage, n, k)

    return make_view(0), make_view(n)


def _unswizzle_block_scale_bytes(
    swizzled_u8: torch.Tensor,
    rows: int,
    cols_blocks: int,
) -> torch.Tensor:
    experts = swizzled_u8.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols_blocks, 4)
    logical_u8 = (
        swizzled_u8.view(torch.uint8)
        .reshape(experts, rows_padded // 128, cols_padded // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(experts, rows_padded, cols_padded)
    )
    return logical_u8[:, :rows, :cols_blocks].contiguous()


def _pad_nvfp4_down_for_tma(
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    k: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad FC2's packed row stride to TMA's 16-byte alignment."""
    n_padded = align_up(n, 32)
    padded_weight = w2_fp4.new_zeros(
        (w2_fp4.shape[0], k, n_padded // 2),
        dtype=torch.uint8,
    )
    padded_weight[:, :, : n // 2].copy_(w2_fp4.view(torch.uint8))

    logical_scale = _unswizzle_block_scale_bytes(
        w2_blockscale,
        k,
        n // _NVFP4_BLOCK_SIZE,
    )
    padded_scale = logical_scale.new_zeros(
        (logical_scale.shape[0], k, n_padded // _NVFP4_BLOCK_SIZE)
    )
    padded_scale[:, :, : n // _NVFP4_BLOCK_SIZE].copy_(logical_scale)
    scale_storage = swizzle_block_scale(padded_scale.view(torch.float8_e4m3fn)).view(
        torch.uint8
    )
    return (
        padded_weight.permute(1, 2, 0),
        as_grouped_scale_view(scale_storage, k, n_padded),
    )


def _derive_w4a8_weight_grids(
    blockscale_u8: torch.Tensor, rows: int, k_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 checkpoint scales -> (UE8M0 K/32 grid, E4M3 residual grid)."""
    from b12x.moe._shared.kernels.reference import (
        decompose_nvfp4_scales_to_mx_residual,
    )

    scales = _unswizzle_block_scales_batched(blockscale_u8, rows, k_dim // 16)
    ue8m0, residual = decompose_nvfp4_scales_to_mx_residual(scales)
    return ue8m0.contiguous(), residual.view(torch.uint8).contiguous()


def _pad_w4a8_grid_rows(
    grid: torch.Tensor,
    *,
    logical_n: int,
    is_gated: bool,
) -> torch.Tensor:
    padded_n = align_up(logical_n, _LEVEL_TILE_N)
    if padded_n == logical_n:
        return grid
    padded_rows = 2 * padded_n if is_gated else padded_n
    padded = grid.new_zeros((grid.shape[0], padded_rows, grid.shape[2]))
    if is_gated:
        padded[:, :logical_n].copy_(grid[:, :logical_n])
        padded[:, padded_n : padded_n + logical_n].copy_(
            grid[:, logical_n : 2 * logical_n]
        )
    else:
        padded[:, :logical_n].copy_(grid)
    return padded


def _pad_w4a8_grid_columns(
    grid: torch.Tensor,
    *,
    logical_n: int,
    elements_per_scale: int,
) -> torch.Tensor:
    padded_columns = align_up(logical_n, _LEVEL_TILE_N) // elements_per_scale
    if padded_columns == grid.shape[2]:
        return grid
    padded = grid.new_zeros((grid.shape[0], grid.shape[1], padded_columns))
    padded[:, :, : grid.shape[2]].copy_(grid)
    return padded


def _as_e8m0_k32_grid(
    scales: torch.Tensor, rows: int, k_dim: int, *, name: str
) -> torch.Tensor:
    """Validate a checkpoint-native per-K/32 E8M0 grid for w4a8_mx."""
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(
            f"{name} must be E8M0 bytes (uint8/float8_e8m0fnu) for w4a8_mx, "
            f"got {scales.dtype}"
        )
    grid = scales.view(torch.uint8)
    # Already-prepared ceil-tiled sfb views arrive with padded extents
    # (rows to 256-tiles, K to 128-tiles); logical grids arrive unpadded.
    rows_pad = -(-rows // 256) * 256
    cols = k_dim // 32
    cols_pad = (-(-k_dim // 128) * 128) // 32
    expected = (int(grid.shape[0]), rows, cols)
    expected_pad = (int(grid.shape[0]), rows_pad, cols_pad)
    if grid.dim() != 3 or tuple(grid.shape) not in (expected, expected_pad):
        raise ValueError(
            f"{name} must be [E, rows, K//32] = {expected} (or the ceil-tiled "
            f"{expected_pad}) for w4a8_mx, got {tuple(grid.shape)}"
        )
    return grid.contiguous()


def _w4a8_rp_shape(size_n: int, size_k: int) -> tuple[int, int, int, int, int, int]:
    size_n = int(size_n)
    size_k = int(size_k)
    if size_n % 8 != 0 or size_k % 32 != 0:
        raise ValueError(
            "W4A8 weight repack requires N multiple of 8 and K "
            f"multiple of 32; got N={size_n}, K={size_k}"
        )
    # Ceil-tiled: shards that 256/128 tiles don't divide (e.g. 2048/TP6 = 352)
    # store zero-filled tail tiles so the fixed 4096-word tile addressing
    # stays uniform; kernels bound their reads by the logical sizes.
    return (-(-size_n // 256), -(-size_k // 128), 4, 8, 32, 4)


def _w4a8_sfb_shape(size_n: int, size_k: int) -> tuple[int, int, int, int]:
    size_n = int(size_n)
    size_k = int(size_k)
    if size_n % 8 != 0 or size_k % 32 != 0:
        raise ValueError(
            "W4A8 scale repack requires N multiple of 8 and K "
            f"multiple of 32; got N={size_n}, K={size_k}"
        )
    return (-(-size_n // 256), -(-size_k // 128), 32, 8)


def _w4a16_packed_weight_shape(size_k: int, size_n: int) -> tuple[int, int]:
    size_k = int(size_k)
    size_n = int(size_n)
    if size_k % 16 != 0 or size_n % 64 != 0:
        raise ValueError(
            f"W4A16 packed weights require K,N multiples of 16,64; got "
            f"K={size_k}, N={size_n}"
        )
    return (size_k // 16, (size_n // 64) * 128)


@triton.jit
def _w4a16_packed_to_qweight_kernel(
    src,
    q,
    total_programs: tl.constexpr,
    k_tiles: tl.constexpr,
    n_tiles: tl.constexpr,
    q_cols: tl.constexpr,
    src_expert_stride: tl.constexpr,
    q_expert_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_programs:
        return
    tile = pid
    expert = tile // (k_tiles * n_tiles)
    rem = tile - expert * (k_tiles * n_tiles)
    kt = rem // n_tiles
    nt = rem - kt * n_tiles
    offs = tl.arange(0, 128)
    source_half = offs // 64
    source_row = offs - source_half * 64
    warp_id = source_row // 16
    col_in_warp = source_row - warp_id * 16
    col_group = col_in_warp // 8
    tc_col = col_in_warp - col_group * 8
    slot_base = col_group * 2 + source_half

    values = tl.zeros((128,), tl.int32)
    for row_pair in range(4):
        th_id = tc_col * 4 + row_pair
        source_lane = th_id * 4 + warp_id
        word = tl.load(
            src
            + expert * src_expert_stride
            + kt * (n_tiles * 128)
            + nt * 128
            + source_lane
        )
        low = (word >> (slot_base * 4)) & 0xF
        high = (word >> ((slot_base + 4) * 4)) & 0xF
        values |= low << (row_pair * 8)
        values |= high << (row_pair * 8 + 4)

    q_base = expert * q_expert_stride + (nt * 64 + source_row) * q_cols
    tl.store(q + q_base + kt * 2 + source_half, values)


@triton.jit
def _qweight_to_w4a8_rp_kernel(
    q,
    dst,
    total_programs: tl.constexpr,
    n_tiles_256: tl.constexpr,
    k_tiles_128: tl.constexpr,
    q_cols: tl.constexpr,
    rows: tl.constexpr,
    q_expert_stride: tl.constexpr,
    dst_expert_stride: tl.constexpr,
    row_rotation: tl.constexpr,
    half_rows: tl.constexpr,
    half_rows_pad: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_programs:
        return
    block_in_tile = pid % tl.cdiv(4096, BLOCK)
    tile = pid // tl.cdiv(4096, BLOCK)
    expert = tile // (n_tiles_256 * k_tiles_128)
    rem = tile - expert * (n_tiles_256 * k_tiles_128)
    nt = rem // k_tiles_128
    kt = rem - nt * k_tiles_128
    idx = block_in_tile * BLOCK + tl.arange(0, BLOCK)
    mask = idx < 4096

    n8i = idx & 3
    tmp = idx >> 2
    combined = tmp & 31
    cgrp = combined & 3
    r8 = combined >> 2
    tmp = tmp >> 5
    n8c = tmp & 7
    k32 = tmp >> 3

    row = n8c * 32 + n8i * 8 + r8
    dst_row = nt * 256 + row
    src_col = kt * 16 + k32 * 4 + cgrp
    if half_rows_pad > 0:
        # Gated W13 with a ceil-tiled half: dst rows [0, half_pad) hold the
        # up half, [half_pad, 2*half_pad) the gate half, each zero-padded to
        # a 128-row boundary so the dynamic kernel's per-128-tile up/gate
        # pairing stays aligned. `row_rotation` here is the SOURCE row of the
        # up half (n for w31 sources, 0 for w13); the other half starts at
        # (rotation + half) % (2*half).
        is_gate = dst_row >= half_rows_pad
        half_dst = dst_row - tl.where(is_gate, half_rows_pad, 0)
        half_src_base = row_rotation + tl.where(is_gate, half_rows, 0)
        half_src_base = tl.where(
            half_src_base >= 2 * half_rows,
            half_src_base - 2 * half_rows,
            half_src_base,
        )
        src_row = half_src_base + half_dst
        in_bounds = (
            (dst_row < 2 * half_rows_pad) & (half_dst < half_rows) & (src_col < q_cols)
        )
    else:
        src_row = dst_row + row_rotation
        src_row = tl.where(src_row >= rows, src_row - rows, src_row)
        # Tail tiles: positions past the logical N/K read as zero (FP4 zero
        # rows and columns), keeping the fixed tile addressing uniform.
        in_bounds = (dst_row < rows) & (src_col < q_cols)
    src_row = tl.where(in_bounds, src_row, 0)
    src_col = tl.where(in_bounds, src_col, 0)
    values = tl.load(
        q + expert * q_expert_stride + src_row * q_cols + src_col,
        mask=mask & in_bounds,
        other=0,
    )
    values = tl.where(in_bounds, values, 0)
    dst_base = expert * dst_expert_stride + (nt * k_tiles_128 + kt) * 4096
    tl.store(dst + dst_base + idx, values, mask=mask)


@triton.jit
def _e8m0_scale_to_grid_kernel(
    scale,
    grid,
    total_elements: tl.constexpr,
    rows: tl.constexpr,
    scale_cols: tl.constexpr,
    rows_pad: tl.constexpr,
    scale_expert_stride: tl.constexpr,
    grid_expert_stride: tl.constexpr,
    source_is_logical: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_elements
    expert = idx // (rows * scale_cols)
    rem = idx - expert * (rows * scale_cols)
    row = rem // scale_cols
    col = rem - row * scale_cols

    if source_is_logical:
        src_off = expert * scale_expert_stride + row * scale_cols + col
    else:
        perm = (row % 8) * 8 + (row // 8) % 8
        pre_row = (row // 64) * 64 + perm
        flat = col * rows_pad + pre_row
        group4 = flat // 4
        byte = flat - group4 * 4
        src_byte = tl.where(byte == 1, 2, tl.where(byte == 2, 1, byte))
        src_off = expert * scale_expert_stride + group4 * 4 + src_byte
    values = tl.load(scale + src_off, mask=mask, other=0)
    values = tl.minimum(values, 247)
    tl.store(
        grid + expert * grid_expert_stride + row * scale_cols + col,
        values,
        mask=mask,
    )


@triton.jit
def _qweight_to_w4a8_n64_kernel(
    q,
    dst,
    total_words: tl.constexpr,
    rows: tl.constexpr,
    group_rows: tl.constexpr,
    q_cols: tl.constexpr,
    row_rotation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_words
    expert_words = rows * q_cols
    expert = idx // expert_words
    local = idx - expert * expert_words
    group_words = group_rows * q_cols
    group = local // group_words
    group_local = local - group * group_words

    full_n_tiles = group_rows // 128
    full_n_words = full_n_tiles * 128 * q_cols
    if group_rows % 128:
        n_tail = group_local >= full_n_words
        nt = tl.where(n_tail, full_n_tiles, group_local // (128 * q_cols))
        n_rows = tl.where(n_tail, group_rows % 128, 128)
        within_n = tl.where(
            n_tail,
            group_local - full_n_words,
            group_local - nt * 128 * q_cols,
        )
    else:
        nt = group_local // (128 * q_cols)
        n_rows = 128
        within_n = group_local - nt * 128 * q_cols

    full_k_tiles = q_cols // 16
    full_k_words = full_k_tiles * n_rows * 16
    if q_cols % 16:
        k_tail = within_n >= full_k_words
        kt = tl.where(k_tail, full_k_tiles, within_n // (n_rows * 16))
        k_words = tl.where(k_tail, q_cols % 16, 16)
        within_kt = tl.where(
            k_tail,
            within_n - full_k_words,
            within_n - kt * n_rows * 16,
        )
    else:
        kt = within_n // (n_rows * 16)
        k_words = 16
        within_kt = within_n - kt * n_rows * 16

    n32_count = n_rows // 32
    k32 = within_kt // (n32_count * 128)
    within_k32 = within_kt - k32 * n32_count * 128
    n32 = within_k32 // 128
    inner = within_k32 - n32 * 128
    n8 = inner & 3
    combined = inner >> 2
    cgrp = combined & 3
    r8 = combined >> 2
    row = group * group_rows + nt * 128 + n32 * 32 + n8 * 8 + r8
    row = row + row_rotation
    row = tl.where(row >= rows, row - rows, row)
    col = kt * 16 + k32 * 4 + cgrp
    value = tl.load(
        q + expert * expert_words + row * q_cols + col,
        mask=mask,
        other=0,
    )
    tl.store(dst + idx, value, mask=mask)


@triton.jit
def _grid_to_w4a8_n64_sfb_kernel(
    grid,
    dst,
    total_bytes: tl.constexpr,
    rows: tl.constexpr,
    group_rows: tl.constexpr,
    scale_cols: tl.constexpr,
    row_rotation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_bytes
    expert_bytes = rows * scale_cols
    expert = idx // expert_bytes
    local = idx - expert * expert_bytes
    group_bytes = group_rows * scale_cols
    group = local // group_bytes
    group_local = local - group * group_bytes

    full_n_tiles = group_rows // 128
    full_n_bytes = full_n_tiles * 128 * scale_cols
    if group_rows % 128:
        n_tail = group_local >= full_n_bytes
        nt = tl.where(n_tail, full_n_tiles, group_local // (128 * scale_cols))
        n_rows = tl.where(n_tail, group_rows % 128, 128)
        within_n = tl.where(
            n_tail,
            group_local - full_n_bytes,
            group_local - nt * 128 * scale_cols,
        )
    else:
        nt = group_local // (128 * scale_cols)
        n_rows = 128
        within_n = group_local - nt * 128 * scale_cols

    full_k_tiles = scale_cols // 4
    full_k_bytes = full_k_tiles * n_rows * 4
    if scale_cols % 4:
        k_tail = within_n >= full_k_bytes
        kt = tl.where(k_tail, full_k_tiles, within_n // (n_rows * 4))
        tile_cols = tl.where(k_tail, scale_cols % 4, 4)
        within_kt = tl.where(
            k_tail,
            within_n - full_k_bytes,
            within_n - kt * n_rows * 4,
        )
    else:
        kt = within_n // (n_rows * 4)
        tile_cols = 4
        within_kt = within_n - kt * n_rows * 4
    row = group * group_rows + nt * 128 + within_kt // tile_cols
    row = row + row_rotation
    row = tl.where(row >= rows, row - rows, row)
    col = kt * 4 + within_kt % tile_cols
    value = tl.load(
        grid + expert * expert_bytes + row * scale_cols + col,
        mask=mask,
        other=0,
    )
    tl.store(dst + idx, value, mask=mask)


@triton.jit
def _grid_to_w4a8_sfb_kernel(
    grid,
    dst,
    total_elements: tl.constexpr,
    rows: tl.constexpr,
    scale_cols: tl.constexpr,
    n_tiles_256: tl.constexpr,
    k_tiles_128: tl.constexpr,
    grid_expert_stride: tl.constexpr,
    dst_expert_stride: tl.constexpr,
    row_rotation: tl.constexpr,
    half_rows: tl.constexpr,
    half_rows_pad: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_elements
    expert = idx // dst_expert_stride
    local = idx - expert * dst_expert_stride

    kb = local & 3
    tmp = local >> 2
    row8 = tmp & 7
    tmp = tmp >> 3
    n32 = tmp & 31
    tmp = tmp >> 5
    kt = tmp % k_tiles_128
    nt = tmp // k_tiles_128

    dst_row = nt * 256 + n32 * 8 + row8
    col = kt * 4 + kb
    if half_rows_pad > 0:
        # Same half-aligned mapping as the weight repack (see
        # _qweight_to_w4a8_rp_kernel).
        is_gate = dst_row >= half_rows_pad
        half_dst = dst_row - tl.where(is_gate, half_rows_pad, 0)
        half_src_base = row_rotation + tl.where(is_gate, half_rows, 0)
        half_src_base = tl.where(
            half_src_base >= 2 * half_rows,
            half_src_base - 2 * half_rows,
            half_src_base,
        )
        row = half_src_base + half_dst
        in_bounds = (
            (dst_row < 2 * half_rows_pad) & (half_dst < half_rows) & (col < scale_cols)
        )
    else:
        row = dst_row + row_rotation
        row = tl.where(row >= rows, row - rows, row)
        # Tail tiles: scale bytes past the logical grid are zero (2^-127
        # scale on already-zero FP4 weights).
        in_bounds = (dst_row < rows) & (col < scale_cols)
    row = tl.where(in_bounds, row, 0)
    col = tl.where(in_bounds, col, 0)
    values = tl.load(
        grid + expert * grid_expert_stride + row * scale_cols + col,
        mask=mask & in_bounds,
        other=0,
    )
    values = tl.where(in_bounds, values, 0)
    tl.store(dst + expert * dst_expert_stride + local, values, mask=mask)


def _w4a8_convert_scratch_nbytes() -> int:
    raw = os.environ.get(_W4A8_CONVERT_SCRATCH_MB_ENV)
    mb = _W4A8_CONVERT_SCRATCH_MB_DEFAULT
    if raw is not None and raw.strip():
        mb = max(1, int(raw))
    return int(mb) << 20


def _w4a8_convert_chunk_experts(
    *,
    weight_E: int,
    scratch_elements_per_expert: int,
    dtype: torch.dtype,
) -> int:
    per_expert = max(1, int(scratch_elements_per_expert)) * _dtype_nbytes(dtype)
    chunk = _w4a8_convert_scratch_nbytes() // per_expert
    return max(1, min(int(weight_E), int(chunk)))


def _decode_w4a16_packed_expert_to_qweight(
    packed_expert: torch.Tensor,
    q_scratch: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
) -> None:
    """Decode one W4A16-packed expert into a caller-owned qweight scratch."""
    packed = packed_expert.view(torch.int32)
    size_k = int(size_k)
    size_n = int(size_n)
    k_tiles = size_k // 16
    n_tiles = size_n // 64
    if tuple(packed.shape) != (k_tiles, n_tiles * 128):
        raise ValueError(
            f"packed expert must be {(k_tiles, n_tiles * 128)}, "
            f"got {tuple(packed.shape)}"
        )
    if q_scratch.dtype != torch.int32 or tuple(q_scratch.shape) != (
        size_n,
        size_k // 8,
    ):
        raise ValueError(
            f"q_scratch must be int32 {(size_n, size_k // 8)}, "
            f"got {q_scratch.dtype} {tuple(q_scratch.shape)}"
        )

    src = packed.reshape(k_tiles, n_tiles, 128)
    out_pos = torch.arange(128, device=packed.device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=packed.device, dtype=torch.long)
    pack_idx = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7],
        device=packed.device,
        dtype=torch.long,
    )
    elem = tc_row[:, None] + offsets[None, :]
    row = elem // 8
    pos = elem % 8
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat(
        [row * 64 + col1, row * 64 + col2],
        dim=1,
    )[:, pack_idx]
    source_shift = (torch.cat([pos, pos], dim=1)[:, pack_idx] * 4).to(torch.int32)
    source_half = source_index // 64
    source_col = source_index % 64

    for kt in range(k_tiles):
        cols = q_scratch[:, kt * 2 : (kt + 1) * 2]
        cols.zero_()
        cols_view = cols.view(n_tiles, 64, 2)
        src_tile = src[kt]
        for slot in range(8):
            values = ((src_tile >> (slot * 4)) & 0xF) << source_shift[:, slot].view(
                1, 128
            )
            values = values.to(torch.int32)
            for half in (0, 1):
                mask = source_half[:, slot] == half
                idx = (
                    source_col[:, slot][mask]
                    .view(1, -1)
                    .expand(
                        n_tiles,
                        -1,
                    )
                )
                cols_view[:, :, half].scatter_add_(1, idx, values[:, mask])


def _copy_qweight_to_w4a8_rp_inplace(
    dst: torch.Tensor,
    qweight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
) -> None:
    size_k = int(size_k)
    size_n = int(size_n)
    nt = size_n // 256
    kt = size_k // 128
    if dst.dtype != torch.int32 or tuple(dst.shape) != _w4a8_rp_shape(size_n, size_k):
        raise ValueError(
            f"dst must be W4A8 rp int32 {_w4a8_rp_shape(size_n, size_k)}, "
            f"got {dst.dtype} {tuple(dst.shape)}"
        )
    if qweight.dtype != torch.int32 or tuple(qweight.shape) != (size_n, size_k // 8):
        raise ValueError(
            f"qweight must be int32 {(size_n, size_k // 8)}, "
            f"got {qweight.dtype} {tuple(qweight.shape)}"
        )
    rotation = 0 if row_rotation is None else int(row_rotation)
    if rotation < 0 or rotation >= size_n:
        raise ValueError(f"row_rotation={rotation} is invalid for N={size_n}")

    dst_view = dst.view(nt, kt, 4, 8, 8, 4, 4)
    wrap_tmp: torch.Tensor | None = None
    for n_tile in range(nt):
        src_start = (n_tile * 256 + rotation) % size_n
        if src_start + 256 <= size_n:
            block = qweight[src_start : src_start + 256]
        else:
            if wrap_tmp is None:
                wrap_tmp = torch.empty(
                    (256, size_k // 8),
                    dtype=qweight.dtype,
                    device=qweight.device,
                )
            first = size_n - src_start
            wrap_tmp[:first].copy_(qweight[src_start:])
            wrap_tmp[first:].copy_(qweight[: 256 - first])
            block = wrap_tmp
        src = block.reshape(8, 4, 8, kt, 4, 4).permute(3, 4, 0, 2, 5, 1)
        dst_view[n_tile].copy_(src)


def _logical_weight_to_w4a8_rp_inplace(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    gated_half_rows: int | None = None,
) -> torch.Tensor:
    size_k = int(size_k)
    size_n = int(size_n)
    if weight.dtype != torch.uint8 or tuple(weight.shape[1:]) != (
        size_n,
        size_k // 2,
    ):
        raise ValueError(
            f"logical FP4 weight must be uint8 [E, {size_n}, {size_k // 2}], "
            f"got {weight.dtype} {tuple(weight.shape)}"
        )
    rp_shape = _w4a8_rp_shape(size_n, size_k)
    n_tiles = -(-size_n // 256)
    k_tiles = -(-size_k // 128)
    rp_words = _tensor_numel(rp_shape)
    has_tail = rp_words != size_n * (size_k // 8)
    # Gated W13 tails place each half's zero rows at a 128-row boundary so
    # the dynamic kernel's per-tile up/gate pairing stays aligned. The
    # interpretation of row_rotation shifts to "source row of the up half"
    # (numerically identical for the aligned case).
    half_rows = 0
    half_rows_pad = 0
    if has_tail and gated_half_rows is not None:
        half_rows = int(gated_half_rows)
        half_rows_pad = -(-half_rows // 128) * 128
        assert 2 * half_rows == size_n, (half_rows, size_n)
        assert 2 * half_rows_pad == n_tiles * 256, (half_rows_pad, n_tiles)
    if weight.is_cuda:
        weight_E = int(weight.shape[0])
        q_cols = size_k // 8
        chunk = _w4a8_convert_chunk_experts(
            weight_E=weight_E,
            scratch_elements_per_expert=size_n * q_cols,
            dtype=torch.int32,
        )
        block = 1024
        row_rot = 0 if row_rotation is None else int(row_rotation)
        weight_i32 = weight.view(torch.int32)
        # Ceil-tiled tails don't fit the source storage; write a fresh
        # destination and read the (never-aliased) source directly.
        out_i32 = (
            torch.empty((weight_E, rp_words), dtype=torch.int32, device=weight.device)
            if has_tail
            else None
        )
        dst_expert_stride = rp_words
        blocks_per_tile = triton.cdiv(4096, block)
        for e0 in range(0, weight_E, chunk):
            e1 = min(weight_E, e0 + chunk)
            if has_tail:
                q_chunk = weight_i32[e0:e1].reshape(e1 - e0, size_n, q_cols)
                dst_chunk = out_i32[e0:e1]
            else:
                q_chunk = torch.empty(
                    (e1 - e0, size_n, q_cols),
                    dtype=torch.int32,
                    device=weight.device,
                )
                q_chunk.copy_(weight_i32[e0:e1].reshape(e1 - e0, size_n, q_cols))
                dst_chunk = weight_i32[e0:e1]
            total_programs = (e1 - e0) * n_tiles * k_tiles
            total_programs *= blocks_per_tile
            _qweight_to_w4a8_rp_kernel[(total_programs,)](
                q_chunk,
                dst_chunk,
                total_programs,
                n_tiles,
                k_tiles,
                q_cols,
                size_n,
                size_n * q_cols,
                dst_expert_stride,
                row_rot,
                half_rows,
                half_rows_pad,
                BLOCK=block,
                num_warps=4,
            )
        base = out_i32 if has_tail else weight_i32
        return base.view(weight_E, *rp_shape)

    if has_tail:
        raise NotImplementedError(
            "ceil-tiled W4A8 repack tails are CUDA-only; move the weights to "
            f"a CUDA device (N={size_n}, K={size_k})"
        )
    q_scratch = torch.empty(
        (size_n, size_k // 8),
        dtype=torch.int32,
        device=weight.device,
    )
    weight_i32 = weight.view(torch.int32)
    for expert in range(int(weight.shape[0])):
        q_scratch.copy_(weight_i32[expert].reshape(size_n, size_k // 8))
        dst = weight_i32[expert].view(rp_shape)
        _copy_qweight_to_w4a8_rp_inplace(
            dst,
            q_scratch,
            size_k=size_k,
            size_n=size_n,
            row_rotation=row_rotation,
        )
    return weight_i32.view(int(weight.shape[0]), *rp_shape)


def _logical_weight_to_w4a8_n64_inplace(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    group_rows: int | None = None,
) -> torch.Tensor:
    """Repack exact N128 tiles with compact N64/K64 boundary tiles."""

    size_k = int(size_k)
    size_n = int(size_n)
    group_rows = size_n if group_rows is None else int(group_rows)
    if size_n % 64 or size_k % 64 or group_rows % 64 or size_n % group_rows:
        raise ValueError(
            "exact W4A8 repack requires N, K, and group rows divisible by 64 "
            f"with whole row groups; got N={size_n}, K={size_k}, "
            f"group_rows={group_rows}"
        )
    if weight.dtype != torch.uint8 or tuple(weight.shape[1:]) != (
        size_n,
        size_k // 2,
    ):
        raise ValueError(
            f"logical FP4 weight must be uint8 [E, {size_n}, {size_k // 2}], "
            f"got {weight.dtype} {tuple(weight.shape)}"
        )
    if not weight.is_cuda:
        raise NotImplementedError("exact W4A8 repack currently requires CUDA weights")

    weight_E = int(weight.shape[0])
    q_cols = size_k // 8
    expert_words = size_n * q_cols
    weight_i32 = weight.view(torch.int32)
    chunk = _w4a8_convert_chunk_experts(
        weight_E=weight_E,
        scratch_elements_per_expert=expert_words,
        dtype=torch.int32,
    )
    row_rot = 0 if row_rotation is None else int(row_rotation)
    block = 256
    for e0 in range(0, weight_E, chunk):
        e1 = min(weight_E, e0 + chunk)
        q_chunk = torch.empty(
            (e1 - e0, size_n, q_cols),
            dtype=torch.int32,
            device=weight.device,
        )
        q_chunk.copy_(weight_i32[e0:e1].reshape(e1 - e0, size_n, q_cols))
        dst_chunk = weight_i32[e0:e1].reshape(-1)
        total_words = (e1 - e0) * expert_words
        _qweight_to_w4a8_n64_kernel[(triton.cdiv(total_words, block),)](
            q_chunk,
            dst_chunk,
            total_words,
            size_n,
            group_rows,
            q_cols,
            row_rot,
            BLOCK=block,
            num_warps=4,
        )
    return weight_i32.view(weight_E, -1)


def _packed_w4a16_weight_to_w4a8_rp_inplace(
    packed_weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
) -> torch.Tensor:
    size_k = int(size_k)
    size_n = int(size_n)
    packed = packed_weight.view(torch.int32)
    packed_shape = _w4a16_packed_weight_shape(size_k, size_n)
    if packed.dim() != 3 or tuple(packed.shape[1:]) != packed_shape:
        raise ValueError(
            f"W4A16 packed weight must be [E, {packed_shape[0]}, "
            f"{packed_shape[1]}], got {tuple(packed.shape)}"
        )
    rp_shape = _w4a8_rp_shape(size_n, size_k)
    if packed.is_cuda:
        weight_E = int(packed.shape[0])
        q_cols = size_k // 8
        chunk = _w4a8_convert_chunk_experts(
            weight_E=weight_E,
            scratch_elements_per_expert=size_n * q_cols,
            dtype=torch.int32,
        )
        block = 1024
        row_rot = 0 if row_rotation is None else int(row_rotation)
        src_expert_stride = _tensor_numel(packed_shape)
        dst_expert_stride = _tensor_numel(rp_shape)
        blocks_per_tile = triton.cdiv(4096, block)
        for e0 in range(0, weight_E, chunk):
            e1 = min(weight_E, e0 + chunk)
            q_scratch = torch.empty(
                (e1 - e0, size_n, q_cols),
                dtype=torch.int32,
                device=packed.device,
            )
            total_decode = (e1 - e0) * (size_k // 16) * (size_n // 64)
            _w4a16_packed_to_qweight_kernel[(total_decode,)](
                packed[e0:e1],
                q_scratch,
                total_decode,
                size_k // 16,
                size_n // 64,
                q_cols,
                src_expert_stride,
                size_n * q_cols,
                num_warps=4,
            )
            total_pack = (e1 - e0) * (size_n // 256) * (size_k // 128)
            total_pack *= blocks_per_tile
            _qweight_to_w4a8_rp_kernel[(total_pack,)](
                q_scratch,
                packed[e0:e1],
                total_pack,
                size_n // 256,
                size_k // 128,
                q_cols,
                size_n,
                size_n * q_cols,
                dst_expert_stride,
                row_rot,
                BLOCK=block,
                num_warps=4,
            )
        return packed.view(weight_E, *rp_shape)

    q_scratch = torch.empty(
        (size_n, size_k // 8),
        dtype=torch.int32,
        device=packed.device,
    )
    for expert in range(int(packed.shape[0])):
        _decode_w4a16_packed_expert_to_qweight(
            packed[expert],
            q_scratch,
            size_k=size_k,
            size_n=size_n,
        )
        dst = packed[expert].view(rp_shape)
        _copy_qweight_to_w4a8_rp_inplace(
            dst,
            q_scratch,
            size_k=size_k,
            size_n=size_n,
            row_rotation=row_rotation,
        )
    return packed.view(int(packed.shape[0]), *rp_shape)


def _recover_w4a16_e8m0_scale_expert(
    scale: torch.Tensor,
    grid_scratch: torch.Tensor,
    *,
    rows: int,
    k_dim: int,
) -> None:
    rows = int(rows)
    k_dim = int(k_dim)
    scale_cols = k_dim // 32
    packed = scale.view(torch.uint8)
    if packed.dim() != 2 or int(packed.shape[0]) != scale_cols:
        raise ValueError(
            f"packed E8M0 scale expert must be [{scale_cols}, >= {rows}], "
            f"got {tuple(packed.shape)}"
        )
    if int(packed.shape[1]) < rows or int(packed.shape[1]) % 64 != 0:
        raise ValueError(
            f"packed E8M0 scale expert second dim must be a multiple of 64 "
            f"and >= {rows}, got {int(packed.shape[1])}"
        )
    if grid_scratch.dtype != torch.uint8 or tuple(grid_scratch.shape) != (
        rows,
        scale_cols,
    ):
        raise ValueError(
            f"grid_scratch must be uint8 {(rows, scale_cols)}, "
            f"got {grid_scratch.dtype} {tuple(grid_scratch.shape)}"
        )

    perm = torch.tensor(
        [i + 8 * j for i in range(8) for j in range(8)],
        dtype=torch.long,
        device=packed.device,
    )
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(64, dtype=torch.long, device=packed.device)

    stage = torch.empty_like(packed)
    stage.view(-1, 4).copy_(packed.view(-1, 4)[:, [0, 2, 1, 3]])
    stage.copy_(stage.reshape(-1, 64)[:, inv_perm].reshape_as(stage))
    grid_scratch.copy_(stage.transpose(0, 1)[:rows])


def _copy_scale_grid_to_w4a8_sfb_inplace(
    dst: torch.Tensor,
    grid: torch.Tensor,
    *,
    rows: int,
    k_dim: int,
    row_rotation: int | None = None,
) -> None:
    rows = int(rows)
    k_dim = int(k_dim)
    nt = rows // 256
    kt = k_dim // 128
    if grid.dtype != torch.uint8 or tuple(grid.shape) != (rows, k_dim // 32):
        raise ValueError(
            f"scale grid must be uint8 {(rows, k_dim // 32)}, "
            f"got {grid.dtype} {tuple(grid.shape)}"
        )
    dst_u8 = dst.view(torch.uint8).view(nt, kt, 32, 8, 4)
    rotation = 0 if row_rotation is None else int(row_rotation)
    if rotation < 0 or rotation >= rows:
        raise ValueError(f"row_rotation={rotation} is invalid for rows={rows}")

    wrap_tmp: torch.Tensor | None = None
    for n_tile in range(nt):
        src_start = (n_tile * 256 + rotation) % rows
        if src_start + 256 <= rows:
            block = grid[src_start : src_start + 256]
        else:
            if wrap_tmp is None:
                wrap_tmp = torch.empty(
                    (256, k_dim // 32),
                    dtype=grid.dtype,
                    device=grid.device,
                )
            first = rows - src_start
            wrap_tmp[:first].copy_(grid[src_start:])
            wrap_tmp[first:].copy_(grid[: 256 - first])
            block = wrap_tmp
        src = block.reshape(32, 8, kt, 4).permute(2, 0, 1, 3)
        dst_u8[n_tile].copy_(src)


def _e8m0_scale_to_w4a8_sfb_inplace(
    scale: torch.Tensor,
    *,
    weight_E: int,
    rows: int,
    k_dim: int,
    row_rotation: int | None = None,
    gated_half_rows: int | None = None,
) -> torch.Tensor:
    weight_E = int(weight_E)
    rows = int(rows)
    k_dim = int(k_dim)
    scale_cols = k_dim // 32
    scale_u8 = scale.view(torch.uint8)
    is_logical, is_packed = _validate_e8m0_scale_w4a8_convertible(
        scale_u8,
        weight_E=weight_E,
        rows=rows,
        k_dim=k_dim,
    )
    sfb_shape = _w4a8_sfb_shape(rows, k_dim)
    n_tiles = -(-rows // 256)
    k_tiles = -(-k_dim // 128)
    sfb_bytes = _tensor_numel(sfb_shape) * 4
    has_tail = sfb_bytes != rows * (k_dim // 32)
    half_rows = 0
    half_rows_pad = 0
    if has_tail and gated_half_rows is not None:
        half_rows = int(gated_half_rows)
        half_rows_pad = -(-half_rows // 128) * 128
        assert 2 * half_rows == rows, (half_rows, rows)
    if scale.is_cuda:
        scale_cols = k_dim // 32
        rows_pad = rows if is_logical else int(scale_u8.shape[2])
        if rows_pad != rows:
            raise ValueError(
                "W4A8 in-place scale conversion requires unpadded E8M0 "
                f"scale storage; got rows={rows}, rows_pad={rows_pad}"
            )
        chunk = _w4a8_convert_chunk_experts(
            weight_E=weight_E,
            scratch_elements_per_expert=rows * scale_cols,
            dtype=torch.uint8,
        )
        block = 256
        row_rot = 0 if row_rotation is None else int(row_rotation)
        total_per_expert = rows * scale_cols
        source_stride = rows * scale_cols if is_logical else scale_cols * rows_pad
        # Ceil-tiled tails don't fit the source storage in place.
        out_u8 = (
            torch.empty((weight_E, sfb_bytes), dtype=torch.uint8, device=scale.device)
            if has_tail
            else None
        )
        for e0 in range(0, weight_E, chunk):
            e1 = min(weight_E, e0 + chunk)
            chunk_experts = e1 - e0
            grid_scratch = torch.empty(
                (chunk_experts, rows, scale_cols),
                dtype=torch.uint8,
                device=scale.device,
            )
            total = chunk_experts * total_per_expert
            _e8m0_scale_to_grid_kernel[(triton.cdiv(total, block),)](
                scale_u8[e0:e1],
                grid_scratch,
                total,
                rows,
                scale_cols,
                rows_pad,
                source_stride,
                total_per_expert,
                is_logical,
                BLOCK=block,
                num_warps=4,
            )
            dst_chunk = out_u8[e0:e1] if has_tail else scale_u8[e0:e1]
            total_sfb = chunk_experts * sfb_bytes
            _grid_to_w4a8_sfb_kernel[(triton.cdiv(total_sfb, block),)](
                grid_scratch,
                dst_chunk,
                total_sfb,
                rows,
                scale_cols,
                n_tiles,
                k_tiles,
                total_per_expert,
                sfb_bytes,
                row_rot,
                half_rows,
                half_rows_pad,
                BLOCK=block,
                num_warps=4,
            )
        base = out_u8 if has_tail else scale_u8
        return base.view(torch.int32).view(weight_E, *sfb_shape)

    if has_tail:
        raise NotImplementedError(
            "ceil-tiled W4A8 scale tails are CUDA-only; move the scales to "
            f"a CUDA device (rows={rows}, K={k_dim})"
        )
    grid_scratch = torch.empty(
        (rows, scale_cols),
        dtype=torch.uint8,
        device=scale.device,
    )
    for expert in range(weight_E):
        if is_logical:
            grid_scratch.copy_(scale_u8[expert])
            grid_scratch.clamp_(max=247)
        else:
            _recover_w4a16_e8m0_scale_expert(
                scale_u8[expert],
                grid_scratch,
                rows=rows,
                k_dim=k_dim,
            )
        dst = scale_u8[expert].view(torch.int32).view(sfb_shape)
        _copy_scale_grid_to_w4a8_sfb_inplace(
            dst,
            grid_scratch,
            rows=rows,
            k_dim=k_dim,
            row_rotation=row_rotation,
        )
    return scale_u8.view(torch.int32).view(weight_E, *sfb_shape)


def _e8m0_scale_to_w4a8_n64_sfb_inplace(
    scale: torch.Tensor,
    *,
    weight_E: int,
    rows: int,
    k_dim: int,
    row_rotation: int | None = None,
    group_rows: int | None = None,
) -> torch.Tensor:
    """Repack E8M0 scales into exact N128 tiles with compact boundaries."""

    weight_E = int(weight_E)
    rows = int(rows)
    k_dim = int(k_dim)
    group_rows = rows if group_rows is None else int(group_rows)
    if rows % 64 or k_dim % 64 or group_rows % 64 or rows % group_rows:
        raise ValueError(
            "exact W4A8 scale repack requires N, K, and group rows divisible "
            f"by 64 with whole row groups; got N={rows}, K={k_dim}, "
            f"group_rows={group_rows}"
        )
    scale_cols = k_dim // 32
    scale_u8 = scale.view(torch.uint8)
    is_logical, _ = _validate_e8m0_scale_w4a8_convertible(
        scale_u8,
        weight_E=weight_E,
        rows=rows,
        k_dim=k_dim,
    )
    if not scale.is_cuda:
        raise NotImplementedError(
            "exact W4A8 scale repack currently requires CUDA scales"
        )
    rows_pad = rows if is_logical else int(scale_u8.shape[2])
    if rows_pad != rows:
        raise ValueError(
            "exact W4A8 in-place scale conversion requires unpadded E8M0 "
            f"storage; got rows={rows}, rows_pad={rows_pad}"
        )
    expert_bytes = rows * scale_cols
    chunk = _w4a8_convert_chunk_experts(
        weight_E=weight_E,
        scratch_elements_per_expert=expert_bytes,
        dtype=torch.uint8,
    )
    row_rot = 0 if row_rotation is None else int(row_rotation)
    source_stride = expert_bytes if is_logical else scale_cols * rows_pad
    block = 256
    for e0 in range(0, weight_E, chunk):
        e1 = min(weight_E, e0 + chunk)
        chunk_experts = e1 - e0
        grid_scratch = torch.empty(
            (chunk_experts, rows, scale_cols),
            dtype=torch.uint8,
            device=scale.device,
        )
        total_bytes = chunk_experts * expert_bytes
        _e8m0_scale_to_grid_kernel[(triton.cdiv(total_bytes, block),)](
            scale_u8[e0:e1],
            grid_scratch,
            total_bytes,
            rows,
            scale_cols,
            rows_pad,
            source_stride,
            expert_bytes,
            is_logical,
            BLOCK=block,
            num_warps=4,
        )
        _grid_to_w4a8_n64_sfb_kernel[(triton.cdiv(total_bytes, block),)](
            grid_scratch,
            scale_u8[e0:e1].reshape(-1),
            total_bytes,
            rows,
            group_rows,
            scale_cols,
            row_rot,
            BLOCK=block,
            num_warps=4,
        )
    return scale_u8.reshape(weight_E, -1).view(torch.int32)


def _validate_e8m0_scale_w4a8_convertible(
    scale_u8: torch.Tensor,
    *,
    weight_E: int,
    rows: int,
    k_dim: int,
) -> tuple[bool, bool]:
    weight_E = int(weight_E)
    rows = int(rows)
    k_dim = int(k_dim)
    scale_cols = k_dim // 32
    logical_shape = (weight_E, rows, scale_cols)
    packed_shape = (weight_E, scale_cols)
    is_logical = scale_u8.dim() == 3 and tuple(scale_u8.shape) == logical_shape
    is_packed = (
        scale_u8.dim() == 3
        and tuple(scale_u8.shape[:2]) == packed_shape
        and int(scale_u8.shape[2]) >= rows
        and int(scale_u8.shape[2]) % 64 == 0
    )
    if not is_logical and not is_packed:
        raise ValueError(
            "E8M0 scale must be logical [E, rows, K//32] or W4A16 packed "
            f"[E, K//32, rows_pad]; got {tuple(scale_u8.shape)}"
        )
    _w4a8_sfb_shape(rows, k_dim)
    return is_logical, is_packed


def _swap_w13_scale_halves_inplace(
    blockscale: torch.Tensor, *, rows: int, cols_blocks: int
) -> None:
    """Swap the FC1 half blocks of a FlashInfer vec16-swizzled scale stack."""
    u8 = blockscale.view(torch.uint8)
    E = u8.shape[0]
    rows_pad = align_up(rows, 128)
    cols_pad = align_up(cols_blocks, 4)
    grid = (
        u8.reshape(E, rows_pad // 128, cols_pad // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(E, rows_pad, cols_pad)
    )
    half = rows // 2
    swapped = torch.cat([grid[:, half:rows], grid[:, :half], grid[:, rows:]], dim=1)
    back = (
        swapped.reshape(E, rows_pad // 128, 4, 32, cols_pad // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(E, -1)
    )
    u8.reshape(E, -1).copy_(back)


# Storages already flipped to the kernel-native [up; gate] order, keyed by
# data_ptr. Values hold the tensors so the allocator cannot recycle a
# registered address (a recycled data_ptr would skip the flip for new weights);
# deliberately NOT cleared with the view caches — the storage stays normalized.
_W13_NORMALIZED_STORAGES: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _ensure_w13_kernel_order_inplace(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    *,
    n: int,
    k: int,
    quant_mode: str = "nvfp4",
) -> None:
    """One-time in-place flip of gate-first ("w31") FC1 storage to kernel order.

    The gated micro/dynamic kernels consume fused FC1 weights as [up; gate]
    ("w13"). vLLM fuses [gate; up], so flip the caller's weight and block-scale
    storage once at first use — the load-time mirror of the W4A16 prepare-path
    row rotation. Mutates caller storage in place.
    """
    reg_key = (w1_fp4.data_ptr(), w1_blockscale.data_ptr())
    if reg_key in _W13_NORMALIZED_STORAGES:
        return
    if not w1_fp4.is_contiguous() or not w1_blockscale.is_contiguous():
        raise ValueError("w31 FC1 normalization requires contiguous w1 storage")
    w1_u8 = w1_fp4.view(torch.uint8)
    if w1_u8.dim() != 3 or int(w1_u8.shape[1]) != 2 * n:
        raise ValueError(
            f"w31 FC1 weights must be [E, 2n, K bytes]; got {tuple(w1_u8.shape)} "
            f"for n={n}"
        )
    E = int(w1_u8.shape[0])
    # Swap halves a few experts at a time to bound the temporary.
    per_expert = int(w1_u8.shape[1]) * int(w1_u8.shape[2])
    chunk = max(1, min(E, (32 << 20) // max(1, per_expert)))
    for e0 in range(0, E, chunk):
        sl = w1_u8[e0 : e0 + chunk]
        tmp = sl[:, :n].clone()
        sl[:, :n] = sl[:, n:]
        sl[:, n:] = tmp
    if quant_mode in ("w4a8_mx", "w8a8_mx"):
        # MXFP4 and MXFP8 sources carry plain [E, 2n, K//32] grids — swap rows
        # directly (the byte width of the payload differs, the grid does not).
        grid = _as_e8m0_k32_grid(w1_blockscale, 2 * n, k, name="w1_blockscale")
        tmp = grid[:, :n].clone()
        grid[:, :n] = grid[:, n:]
        grid[:, n:] = tmp
    else:
        _swap_w13_scale_halves_inplace(w1_blockscale, rows=2 * n, cols_blocks=k // 16)
    _W13_NORMALIZED_STORAGES[reg_key] = (w1_fp4, w1_blockscale)


def _get_w6a8_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    *,
    activation_spec: _ActivationKernelSpec,
) -> _WeightViews:
    """Weight views for the prepared w6a8_mx (MX-FP6) expert layout.

    Mirrors ``_get_weight_views`` structurally, but the operands are the
    PreparedW6A8MXFP6Weights canonical tensors: packed FP6 code bytes
    ``[E, N, 3K//4]`` and MMA-swizzled UE8M0 scale atoms
    ``[E, pad128(N), pad4(K//32)]``.  The dynamic launch passes the permuted
    code views through Float8E4M3FN gmem pointers and the swizzled scale bytes
    through Float8E8M0FNU pointers (sf_vec_size 32).
    """
    global _LAST_WEIGHTS
    w1_n = activation_spec.w1_rows(n)
    w1_u8 = w1_fp4.view(torch.uint8)
    w2_u8 = w2_fp4.view(torch.uint8)
    if w1_u8.dim() != 3 or tuple(w1_u8.shape[1:]) != (w1_n, 3 * k // 4):
        raise ValueError(
            "w6a8_mx FC1 weights must be packed [E, w1_n, 3*k//4]; got "
            f"{tuple(w1_u8.shape)} for w1_n={w1_n}, k={k}"
        )
    if w2_u8.dim() != 3 or tuple(w2_u8.shape[1:]) != (k, 3 * n // 4):
        raise ValueError(
            "w6a8_mx FC2 weights must be packed [E, k, 3*n//4]; got "
            f"{tuple(w2_u8.shape)} for k={k}, n={n}"
        )
    if not w1_alphas.is_contiguous() or not w2_alphas.is_contiguous():
        raise ValueError("w1_alphas and w2_alphas must be contiguous")
    key = (
        w1_u8.data_ptr(),
        w1_blockscale.data_ptr(),
        w2_u8.data_ptr(),
        w2_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_alphas.data_ptr(),
        activation_spec.activation,
        "w6a8_mx",
    )
    last_wkey, last_wval = _LAST_WEIGHTS
    if last_wkey == key:
        return last_wval
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _LAST_WEIGHTS = (key, cached)
        return cached

    # Permute [E, rows, 3K//4] -> [rows, 3K//4, E] (view, no copy!) to match
    # the kernel's k-major [N, K_packed, E] weight tensors.
    w13 = w1_u8.permute(1, 2, 0)
    down = w2_u8.permute(1, 2, 0)
    w13_sf = w1_blockscale.view(torch.uint8)
    down_sf = w2_blockscale.view(torch.uint8)
    views = _WeightViews(
        w13=w13,
        down=down,
        w13_sf=w13_sf,
        down_sf=down_sf,
        w1_alpha=w1_alphas,
        w2_alpha=w2_alphas,
        w1_storage=w1_u8,
        w1_scale_storage=w13_sf,
        w2_storage=w2_u8,
        w2_scale_storage=down_sf,
    )
    # One FP6 code byte-triplet per four elements; the Float8E4M3FN element
    # type is conveyed via _gptr / compile-time fakes, so keep uint8 views.
    views.w13_fp4 = w13
    views.down_fp4 = down
    views.sfb_w13_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        w13_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    views.sfb_down_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        down_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    _WEIGHT_CACHE[key] = views
    _LAST_WEIGHTS = (key, views)
    return views


def _resolve_w8a8_weight_views(
    w1_fp8: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    *,
    activation_spec: _ActivationKernelSpec,
) -> _WeightViews:
    """Resolve the dynamic-launch views of prepared w8a8_mx (MXFP8) experts.

    Called once by weight preparation; the result is stored on the prepared
    container (``PreparedW8A8MXFP8Weights.launch_views``) and the launch path
    reads it without re-validating geometry.  The operands are the canonical
    tensors: byte-exact E4M3 codes ``[E, N, K]`` (gmem K extent == logical K,
    no 3:4 packing) and MMA-swizzled UE8M0 K/32 scale atoms
    ``[E, pad128(N), pad4(K//32)]``.  The dynamic launch passes the permuted
    code views through Float8E4M3FN gmem pointers and the swizzled scale bytes
    through Float8E8M0FNU pointers (sf_vec_size 32), so the native MmaMXF8Op
    consumes them directly.
    """
    w1_n = activation_spec.w1_rows(n)
    w1_u8 = w1_fp8.view(torch.uint8)
    w2_u8 = w2_fp8.view(torch.uint8)
    if w1_u8.dim() != 3 or tuple(w1_u8.shape[1:]) != (w1_n, k):
        raise ValueError(
            "w8a8_mx FC1 weights must be [E, w1_n, k] with one E4M3 byte per "
            f"element; got {tuple(w1_u8.shape)} for w1_n={w1_n}, k={k}"
        )
    if w2_u8.dim() != 3 or tuple(w2_u8.shape[1:]) != (k, n):
        raise ValueError(
            "w8a8_mx FC2 weights must be [E, k, n] with one E4M3 byte per "
            f"element; got {tuple(w2_u8.shape)} for k={k}, n={n}"
        )
    if not w1_alphas.is_contiguous() or not w2_alphas.is_contiguous():
        raise ValueError("w1_alphas and w2_alphas must be contiguous")

    # Permute [E, rows, K] -> [rows, K, E] (view, no copy!) to match the
    # kernel's k-major [N, K, E] weight tensors.
    w13 = w1_u8.permute(1, 2, 0)
    down = w2_u8.permute(1, 2, 0)
    down_sf = w2_blockscale.view(torch.uint8)
    w13_gate_sf = None
    if activation_spec.is_gated and n % _LEVEL_TILE_N != 0:
        # Non-128 intermediate: preparation swizzled the up and gate halves
        # independently ([2, E, pad128(n), pad4(k//32)]) for the kernel's
        # separate FC1 half descriptors.
        split_sf = w1_blockscale.view(torch.uint8)
        if split_sf.dim() != 4 or split_sf.shape[0] != 2:
            raise ValueError(
                "w8a8_mx with intermediate % 128 != 0 requires half-major "
                f"[2, E, rows, cols] FC1 scales; got {tuple(split_sf.shape)}"
            )
        w13_sf = split_sf[0]
        w13_gate_sf = split_sf[1]
    else:
        w13_sf = w1_blockscale.view(torch.uint8)
    views = _WeightViews(
        w13=w13,
        down=down,
        w13_sf=w13_sf,
        down_sf=down_sf,
        w1_alpha=w1_alphas,
        w2_alpha=w2_alphas,
        w1_storage=w1_u8,
        w1_scale_storage=w13_sf,
        w2_storage=w2_u8,
        w2_scale_storage=down_sf,
    )
    views.w13_gate_sf = w13_gate_sf
    # One E4M3 byte per element: the Float8E4M3FN element type is conveyed via
    # _gptr / compile-time fakes, so keep uint8 views here.
    views.w13_fp4 = w13
    views.down_fp4 = down
    views.sfb_w13_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        w13_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    views.sfb_down_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        down_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return views


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    *,
    activation_spec: _ActivationKernelSpec,
    quant_mode: str = "nvfp4",
    w13_layout: str = "w13",
) -> _WeightViews:
    """Create weight views from the expert-weight layout.

    For gated SwiGLU kernels, ``w1_fp4`` is `[E, 2*n, k//2]`; ``w13_layout``
    names the source half order ("w31"/gate-first sources are flipped to the
    kernel-native [up; gate] order in place, once per storage).
    For relu2 kernels, ``w1_fp4`` is `[E, n, k//2]`.
    """
    global _LAST_WEIGHTS
    quant_mode = _normalize_quant_mode(quant_mode)
    if quant_mode == "w6a8_mx":
        # PreparedW6A8MXFP6Weights canonical storage: w1_fp4/w2_fp4 carry the
        # 3:4-packed FP6 code bytes ([E, N, 3K//4] uint8, already [up; gate])
        # and the blockscales carry the MMA-swizzled UE8M0 K/32 atoms
        # ([E, pad128(N), pad4(K//32)] uint8).  The dynamic kernel consumes
        # the packed bytes as Float8E4M3FN views with gmem K extent 3K/4
        # (expanded to byte-containers in smem) and rebuilds SF geometry from
        # the LOGICAL K, so the views are plain [N, 3K//4, E] permutes; the
        # FP4 K//2 view math and the in-place w31 flip below never apply.
        return _get_w6a8_weight_views(
            w1_fp4,
            w1_blockscale,
            w2_fp4,
            w2_blockscale,
            w1_alphas,
            w2_alphas,
            n,
            k,
            activation_spec=activation_spec,
        )
    if quant_mode == "w4a8_mx" and w1_fp4.dim() != 3:
        # In-place N256/K128-repacked storage (tiny_decode band): view flat as the
        # source-native 3-D shape for pointer plumbing. The prep rotation
        # already normalized the half order -- never flip rp storage in place.
        e_dim = w1_fp4.shape[0]
        # Ceil-tiled storage views back as ceil-row/col 3-D shapes (equal to
        # the logical shapes when the tiles divide exactly); consumers use rp
        # tile addressing from the base pointer, never these extents.
        w1_rows_pad = -(-activation_spec.w1_rows(n) // 256) * 256
        n_pad = -(-n // 128) * 128
        w1_fp4 = w1_fp4.view(torch.uint8).reshape(e_dim, w1_rows_pad, k // 2)
        w2_fp4 = w2_fp4.view(torch.uint8).reshape(e_dim, k, n_pad // 2)
        w1_blockscale = w1_blockscale.view(torch.uint8).reshape(
            e_dim, w1_rows_pad, k // 32
        )
        w2_blockscale = w2_blockscale.view(torch.uint8).reshape(e_dim, k, n_pad // 32)
        w13_layout = "w13"
    w13_layout = _normalize_w13_layout_for_activation(
        activation_spec.activation,
        w13_layout,
    )
    if w13_layout == "w31" and activation_spec.is_gated:
        _ensure_w13_kernel_order_inplace(
            w1_fp4, w1_blockscale, n=n, k=k, quant_mode=quant_mode
        )
    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_alphas.data_ptr(),
        activation_spec.activation,
        quant_mode if _is_w4a8_quant_mode(quant_mode) else "nvfp4",
    )
    last_wkey, last_wval = _LAST_WEIGHTS
    if last_wkey == key:
        return last_wval
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        _LAST_WEIGHTS = (key, cached)
        return cached

    # Permute [E, w1_n, k//2] → [w1_n, k//2, E] (view, no copy!)
    w13 = w1_fp4.permute(1, 2, 0)  # [w1_n, k//2, E]
    down = w2_fp4.permute(1, 2, 0)  # [k, n//2, E]

    # Compact contiguous scale storage for the FC1 weights.
    w1_n = activation_spec.w1_rows(n)
    bs_u8 = w1_blockscale.view(torch.uint8)
    w1_scale_storage = w1_blockscale
    w2_scale_storage = w2_blockscale
    if quant_mode == "w4a8_mx":
        # MXFP4 sources carry checkpoint-native per-K/32 E8M0 grids
        # ([E, rows, K//32] bytes); there is no vec16 scale stack to view.
        # The w4a8 kernels never read the vec16 SF descriptors, so the
        # vec16 view slots are pointed at the grid bytes below purely to
        # plumb valid addresses through the launch.
        w13_sf = _as_e8m0_k32_grid(w1_blockscale, w1_n, k, name="w1_blockscale")
        down_sf = _as_e8m0_k32_grid(w2_blockscale, k, n, name="w2_blockscale")
        w1_scale_storage = w13_sf
        w2_scale_storage = down_sf
    else:
        w13_sf = as_grouped_scale_view(bs_u8, w1_n, k)
        down_sf = as_grouped_scale_view(w2_blockscale.view(torch.uint8), k, n)
    if not w1_alphas.is_contiguous() or not w2_alphas.is_contiguous():
        raise ValueError("w1_alphas and w2_alphas must be contiguous")

    sf_dtype = cutlass.Float8E4M3FN
    views = _WeightViews(
        w13=w13,
        down=down,
        w13_sf=w13_sf,
        down_sf=down_sf,
        w1_alpha=w1_alphas,
        w2_alpha=w2_alphas,
        w1_storage=w1_fp4,
        w1_scale_storage=w1_scale_storage,
        w2_storage=w2_fp4,
        w2_scale_storage=w2_scale_storage,
    )
    # Keep as uint8 for dlpack compatibility — torch float4 types are not
    # supported by dlpack, and sglang may load weights as native float4.
    # The CUTLASS kernel receives the element type via _gptr / compile-time
    # dtype, not from the torch tensor dtype.
    views.w13_fp4 = w13.view(torch.uint8)
    views.down_fp4 = down.view(torch.uint8)
    views.sfb_w13_ptr = make_ptr(
        sf_dtype, w13_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    views.sfb_down_ptr = make_ptr(
        sf_dtype, down_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    if _is_w4a8_quant_mode(quant_mode):
        if quant_mode == "w4a8_nvfp4":
            dynamic_w13_storage = _pad_w4a8_grid_rows(
                w1_fp4,
                logical_n=n,
                is_gated=activation_spec.is_gated,
            )
            views.w13_fp4 = dynamic_w13_storage.permute(1, 2, 0).view(torch.uint8)
            w13_mx, w13_residual = _derive_w4a8_weight_grids(bs_u8, w1_n, k)
            down_mx, down_residual = _derive_w4a8_weight_grids(
                w2_blockscale.view(torch.uint8), k, n
            )
            views.sfb_w13_mx = _pad_w4a8_grid_rows(
                w13_mx,
                logical_n=n,
                is_gated=activation_spec.is_gated,
            )
            views.w13_residual = _pad_w4a8_grid_rows(
                w13_residual,
                logical_n=n,
                is_gated=activation_spec.is_gated,
            )
            views.sfb_down_mx = _pad_w4a8_grid_columns(
                down_mx,
                logical_n=n,
                elements_per_scale=32,
            )
            views.down_residual = _pad_w4a8_grid_columns(
                down_residual,
                logical_n=n,
                elements_per_scale=16,
            )
        else:
            # w4a8_mx: the checkpoint E8M0 K/32 grids feed the kernel
            # directly. Residuals stay None — the w4a8_mx kernel neither
            # stages nor reads them (const_expr-gated) and the launch
            # substitutes dummy pointers.
            views.sfb_w13_mx = w13_sf
            views.sfb_down_mx = down_sf
    _WEIGHT_CACHE[key] = views
    _LAST_WEIGHTS = (key, views)
    return views


def _ensure_dynamic_nvfp4_tail_views(
    views: _WeightViews,
    n: int,
    k: int,
    *,
    activation_spec: _ActivationKernelSpec,
) -> None:
    if n % 32 != 16:
        return
    if activation_spec.is_gated and views.w13_up_sf is None:
        views.w13_up_sf, views.w13_gate_sf = _split_gated_nvfp4_scale_views(
            views.w1_scale_storage,
            n,
            k,
        )
    if views.dynamic_down_fp4 is None:
        views.dynamic_down_fp4, views.dynamic_down_sf = _pad_nvfp4_down_for_tma(
            views.w2_storage,
            views.w2_scale_storage,
            k,
            n,
        )


# --------------------------------------------------------------------------
# Prepared W4A8 weight representation
# --------------------------------------------------------------------------


def _w4a8_trellis_weight_views(
    prepared: object,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
) -> _WeightViews:
    """Micro-launch views for the trellis-native w4a8 representation.

    The trellis payloads are opaque byte streams to the launch ABI, and the
    micro trellis arm reads no weight block scales. The w13 scale slot
    therefore carries the per-expert boundary rotations (fp16, [E, 3*I]
    ordinary or [E, 6*I] with an intermediate Hadamard); the compact micro
    launch rebinds it as the rotation operand and fetches the shared value table from the
    per-device cache.
    """

    if not w1_alphas.is_contiguous() or not w2_alphas.is_contiguous():
        raise ValueError("w1_alphas and w2_alphas must be contiguous")
    rotations = prepared.intermediate_rotations
    if rotations.dtype != torch.float16:
        raise TypeError("trellis intermediate rotations must be fp16")
    rotations = rotations.contiguous()
    w13 = prepared.w13.contiguous().view(torch.uint8).reshape(-1)
    w2 = prepared.w2.contiguous().view(torch.uint8).reshape(-1)
    dummy = prepared.w2_scale
    return _WeightViews(
        w13=w13,
        down=w2,
        w13_sf=dummy,
        down_sf=dummy,
        w1_alpha=w1_alphas,
        w2_alpha=w2_alphas,
        w1_storage=w13,
        w1_scale_storage=rotations,
        w2_storage=w2,
        w2_scale_storage=dummy,
    )


def _w4a8_prepared_dict(prepared: object) -> dict[str, torch.Tensor]:
    """Normalize public prepared metadata to the dynamic launch ABI."""

    if isinstance(prepared, dict):
        values = prepared
    elif getattr(prepared, "weight_layout", None) == "trellis_t256":
        # Trellis-native representation: the rp slots carry the
        # projection-major payloads and the sfb slots carry the fp16
        # boundary rotations. The dynamic launch keys trellis mode off
        # that dtype; the kernel's SFB operands are compile-time dead
        # (identity UE8M0 word).
        values = {
            "w13_rp": prepared.w13,
            "w13_sfb": prepared.intermediate_rotations,
            "w2_rp": prepared.w2,
            "w2_sfb": prepared.intermediate_rotations,
        }
    else:
        values = {
            name: getattr(prepared, name)
            for name in ("w13_rp", "w13_sfb", "w2_rp", "w2_sfb")
        }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"prepared W4A8 {name} must be a tensor")
    return values


def _w4a8_prepared_weight_views(
    prepared: object,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
) -> _WeightViews:
    """Build pointer carriers when source-native storage was compacted.

    Every source-layout operand is compile-time dead in the repacked W4A8
    specialization.  Reuse the prepared tensors as valid pointer carriers so
    serving does not need to retain a second copy of the logical weights.
    """

    if not w1_alphas.is_contiguous() or not w2_alphas.is_contiguous():
        raise ValueError("w1_alphas and w2_alphas must be contiguous")
    views = _w4a8_prepared_dict(prepared)
    w13_rp = views["w13_rp"]
    w13_sfb = views["w13_sfb"]
    w2_rp = views["w2_rp"]
    w2_sfb = views["w2_sfb"]
    return _WeightViews(
        w13=w13_rp,
        down=w2_rp,
        w13_sf=w13_sfb,
        down_sf=w2_sfb,
        w1_alpha=w1_alphas,
        w2_alpha=w2_alphas,
        w1_storage=w13_rp,
        w1_scale_storage=w13_sfb,
        w2_storage=w2_rp,
        w2_scale_storage=w2_sfb,
        w13_fp4=w13_rp,
        down_fp4=w2_rp,
        sfb_w13_mx=w13_sfb,
        sfb_down_mx=w2_sfb,
    )


def _require_unit_weight_alphas(
    w1_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    context: str,
) -> None:
    unit_alphas = bool(
        torch.all(w1_global_scale == 1.0).item()
        and torch.all(w2_global_scale == 1.0).item()
    )
    if not unit_alphas:
        raise RuntimeError(f"{context} requires unit w1/w2 global scales")


def _prepare_w4a8_from_e8m0_source(
    *,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    activation: str,
    params_dtype: torch.dtype,
    source_format: str,
    w13_layout: str,
    n64_repack: bool = False,
) -> _PreparedW4A8Weights:
    source_format = _normalize_fp4_source_format(source_format)
    if source_format != "fp4_e8m0_k32":
        raise ValueError(
            "W4A8 preparation requires source_format='fp4_e8m0_k32', "
            f"got {source_format!r}"
        )
    activation = normalize_moe_activation(activation)
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    _require_unit_weight_alphas(
        w1_global_scale,
        w2_global_scale,
        context="W4A8 preparation",
    )

    if w1_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise TypeError(
            "W4A8 preparation requires logical uint8 FP4 weights; "
            f"got w1={w1_fp4.dtype}, w2={w2_fp4.dtype}"
        )
    if w1_fp4.dim() != 3 or w2_fp4.dim() != 3:
        raise ValueError(
            "W4A8 preparation requires logical rank-3 FP4 weights; "
            f"got w1={tuple(w1_fp4.shape)}, w2={tuple(w2_fp4.shape)}"
        )
    weight_E = int(w1_fp4.shape[0])
    if int(w2_fp4.shape[0]) != weight_E:
        raise ValueError(
            "W4A8 preparation expert count mismatch: "
            f"w1={int(w1_fp4.shape[0])}, w2={int(w2_fp4.shape[0])}"
        )
    k = int(w2_fp4.shape[1])
    n = int(w2_fp4.shape[2]) * 2
    if int(w1_fp4.shape[2]) * 2 != k:
        raise ValueError(
            "W4A8 preparation hidden size mismatch: "
            f"w1 K={int(w1_fp4.shape[2]) * 2}, w2 K={k}"
        )
    w1_rows = _activation_w1_rows(activation, n)
    expected_w1_shape = (weight_E, w1_rows, k // 2)
    expected_w2_shape = (weight_E, k, n // 2)
    if tuple(w1_fp4.shape) != expected_w1_shape:
        raise ValueError(
            "W4A8 preparation requires w1 logical shape "
            f"{expected_w1_shape}; got {tuple(w1_fp4.shape)}"
        )
    if tuple(w2_fp4.shape) != expected_w2_shape:
        raise ValueError(
            "W4A8 preparation requires w2 logical shape "
            f"{expected_w2_shape}; got {tuple(w2_fp4.shape)}"
        )
    _as_e8m0_k32_grid(w1_blockscale, w1_rows, k, name="w1_blockscale")
    _as_e8m0_k32_grid(w2_blockscale, k, n, name="w2_blockscale")

    is_gated = is_gated_moe_activation(activation)
    row_rotation = n if w13_layout == "w31" and is_gated else None
    if n64_repack:
        w13_rp = _logical_weight_to_w4a8_n64_inplace(
            w1_fp4,
            size_k=k,
            size_n=w1_rows,
            row_rotation=row_rotation,
            group_rows=n if is_gated else None,
        )
        w2_rp = _logical_weight_to_w4a8_n64_inplace(
            w2_fp4,
            size_k=n,
            size_n=k,
        )
        w13_sfb = _e8m0_scale_to_w4a8_n64_sfb_inplace(
            w1_blockscale,
            weight_E=weight_E,
            rows=w1_rows,
            k_dim=k,
            row_rotation=row_rotation,
            group_rows=n if is_gated else None,
        )
        w2_sfb = _e8m0_scale_to_w4a8_n64_sfb_inplace(
            w2_blockscale,
            weight_E=weight_E,
            rows=k,
            k_dim=n,
        )
    else:
        w13_rp = _logical_weight_to_w4a8_rp_inplace(
            w1_fp4,
            size_k=k,
            size_n=w1_rows,
            row_rotation=row_rotation,
            gated_half_rows=n if is_gated else None,
        )
        w2_rp = _logical_weight_to_w4a8_rp_inplace(
            w2_fp4,
            size_k=n,
            size_n=k,
        )
        w13_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w1_blockscale,
            weight_E=weight_E,
            rows=w1_rows,
            k_dim=k,
            row_rotation=row_rotation,
            gated_half_rows=n if is_gated else None,
        )
        w2_sfb = _e8m0_scale_to_w4a8_sfb_inplace(
            w2_blockscale,
            weight_E=weight_E,
            rows=k,
            k_dim=n,
        )
    return _PreparedW4A8Weights(
        w13_rp=w13_rp,
        w13_sfb=w13_sfb,
        w2_rp=w2_rp,
        w2_sfb=w2_sfb,
        num_experts=weight_E,
        hidden_size=k,
        intermediate_size=n,
        params_dtype=params_dtype,
        source_format=source_format,
        w13_layout=w13_layout,
        n64_repack=n64_repack,
    )


def _prepare_w6a8_from_mxfp6_source(
    *,
    w1_fp6: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2_fp6: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    activation: str,
    source_format: str,
    w13_layout: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    """Prepare packed MX-FP6 experts for the w6a8_mx dynamic-kernel port.

    Weight codes stay source-native ([E, N, 3K//4] uint8); UE8M0 grids are
    swizzled to the MMA layout and per-expert alphas come from the
    ``*_weight_scale_2`` globals divided by the reciprocal activation global
    scales.  Returns a
    :class:`b12x.moe._shared.kernels.w6a8.PreparedW6A8MXFP6Weights`.
    """
    from b12x.moe._shared.kernels.w6a8 import prepare_w6a8_mxfp6_weights

    source_format = _normalize_fp4_source_format(source_format)
    if source_format != "mxfp6_e2m3":
        raise ValueError(
            "W6A8 preparation requires source_format='mxfp6_e2m3', "
            f"got {source_format!r}"
        )
    activation = normalize_moe_activation(activation)
    if activation != "silu":
        raise ValueError(
            f"W6A8 MX-FP6 preparation currently requires silu, got {activation!r}"
        )
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    if w13_layout != "w13":
        raise NotImplementedError(
            "w6a8_mx: gate-first ('w31') FC1 row rotation of packed FP6 "
            "storage pending kernel port; fuse FC1 as [up; gate] ('w13')"
        )
    return prepare_w6a8_mxfp6_weights(
        w13_packed=w1_fp6,
        w13_scale=w1_blockscale,
        w13_scale_2=w1_global_scale,
        w2_packed=w2_fp6,
        w2_scale=w2_blockscale,
        w2_scale_2=w2_global_scale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def _prepare_w8a8_from_mxfp8_source(
    *,
    w1_fp8: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    activation: str,
    source_format: str,
    w13_layout: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
):
    """Prepare MXFP8 experts and resolve their dynamic-launch views.

    Weight codes stay source-native and byte-exact ([E, N, K] uint8 — no
    requantization and no re-packing); UE8M0 grids are validated and swizzled
    to the MMA layout by the canonical helpers, and per-expert alphas come
    from the ``*_weight_scale_2`` globals divided by the reciprocal activation
    global scales.  A gate-first ("w31") FC1 is rotated to kernel order here,
    while the payload and its scale grid are still both unswizzled, so the
    row flip and the MMA swizzle cannot disagree about which row is which.
    Returns a
    :class:`b12x.moe._shared.kernels.w8a8.PreparedW8A8MXFP8Weights` whose
    ``launch_views`` carry the validated kernel operand views and pointers.
    """
    from b12x.moe._shared.kernels.w8a8 import prepare_w8a8_mxfp8_weights

    source_format = _normalize_fp4_source_format(source_format)
    if source_format != "mxfp8_e8m0_k32":
        raise ValueError(
            "W8A8 preparation requires source_format='mxfp8_e8m0_k32', "
            f"got {source_format!r}"
        )
    activation = normalize_moe_activation(activation)
    if activation != "silu":
        raise ValueError(f"W8A8 MXFP8 preparation requires silu, got {activation!r}")
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    if w13_layout == "w31":
        _ensure_w13_kernel_order_inplace(
            w1_fp8,
            w1_blockscale,
            n=intermediate_size,
            k=hidden_size,
            quant_mode="w8a8_mx",
        )
    prepared = prepare_w8a8_mxfp8_weights(
        w13_values=w1_fp8,
        w13_scale=w1_blockscale,
        w13_scale_2=w1_global_scale,
        w2_values=w2_fp8,
        w2_scale=w2_blockscale,
        w2_scale_2=w2_global_scale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    # Geometry and pointer resolution happen here, once per prepared layer;
    # the launch path reads launch_views directly.
    launch_views = _resolve_w8a8_weight_views(
        prepared.w13_values,
        prepared.w13_sf_swizzled,
        prepared.w2_values,
        prepared.w2_sf_swizzled,
        prepared.w13_alpha,
        prepared.w2_alpha,
        intermediate_size,
        hidden_size,
        activation_spec=_get_activation_kernel_spec(activation, quant_mode="w8a8_mx"),
    )
    return replace(prepared, launch_views=launch_views)


def _prepare_modelopt_nvfp4_runtime_alphas(
    w1_global_scale: torch.Tensor,
    a1_gscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    a2_gscale: torch.Tensor,
    *,
    weight_E: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare W4A4 runtime alphas from raw ModelOpt NVFP4 scales.

    ModelOpt stores per-expert weight global scales separately from activation
    input scales. The existing NVFP4 W4A4 kernels consume
    ``weight_global_scale * input_scale`` while vLLM exposes reciprocal input
    scales in ``a*_gscale``. Keep that conversion in B12X preparation so
    integrations do not mutate checkpoint-owned scale tensors in-place.
    """
    if weight_E is None:
        weight_E = int(w1_global_scale.numel())
    w1_scale = _prepare_expert_scale_vector(
        w1_global_scale,
        weight_E,
        name="w1_global_scale",
    )
    w2_scale = _prepare_expert_scale_vector(
        w2_global_scale,
        weight_E,
        name="w2_global_scale",
    )
    a1_recip = _prepare_expert_scale_vector(a1_gscale, weight_E, name="a1_gscale")
    a2_recip = _prepare_expert_scale_vector(a2_gscale, weight_E, name="a2_gscale")
    w1_runtime = torch.empty_like(w1_scale)
    w2_runtime = torch.empty_like(w2_scale)
    torch.div(w1_scale, a1_recip, out=w1_runtime)
    torch.div(w2_scale, a2_recip, out=w2_runtime)
    return w1_runtime.contiguous(), w2_runtime.contiguous()


def plan_b12x_fp4_moe_weights(
    *,
    quant_modes: str | Sequence[str],
    source_format: str,
    activation: str,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    nvfp4_inline_scales: bool = False,
    w4a16_compressed_scales: bool = False,
    w4a8_csf_inline: bool = False,
    w13_layout: str = "w13",
    w4a16_layout: PreparedWeightLayout | str | None = None,
    trellis_bits: int | None = None,
    trellis_tile_config: tuple[int, int, int, int] | None = None,
    intermediate_hadamard: bool | None = None,
    trellis_codebook: str | None = None,
    trellis_rate_granularity: str | None = None,
    trellis_pair_kinds: Sequence[str] | frozenset[str] | None = None,
    intermediate_hadamard_blocks: tuple[int, int] | None = None,
) -> MoEWeightPreparationPlan:
    """Plan the one canonical weight allocation used by selected recipes."""

    source_format = _normalize_fp4_source_format(source_format)
    activation = normalize_moe_activation(activation)
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    modes = (quant_modes,) if isinstance(quant_modes, str) else tuple(quant_modes)
    if not modes:
        raise ValueError("quant_modes must contain at least one recipe")
    specs = tuple(
        make_moe_spec(
            quant_mode=_normalize_quant_mode_for_source(mode, source_format),
            source_format=source_format,
            activation=activation,
            io_dtype=str(params_dtype).removeprefix("torch."),
            w13_layout=w13_layout,
        )
        for mode in modes
    )
    result = plan_moe_weight_preparation(
        specs,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w4a16_layout=w4a16_layout,
        trellis_bits=trellis_bits,
        trellis_tile_config=trellis_tile_config,
        intermediate_hadamard=intermediate_hadamard,
        trellis_codebook=trellis_codebook,
        trellis_rate_granularity=trellis_rate_granularity,
        trellis_pair_kinds=trellis_pair_kinds,
        intermediate_hadamard_blocks=intermediate_hadamard_blocks,
    )

    return replace(
        result,
        nvfp4_inline_scales=bool(nvfp4_inline_scales),
        w4a16_compressed_scales=bool(w4a16_compressed_scales),
        w4a8_csf_inline=bool(w4a8_csf_inline),
    )


def prepare_b12x_fp4_moe_weights(
    *,
    plan: MoEWeightPreparationPlan,
    params_dtype: torch.dtype,
    w1_fp4: torch.Tensor | None = None,
    w2_fp4: torch.Tensor | None = None,
    w1_global_scale: torch.Tensor | None = None,
    w2_global_scale: torch.Tensor | None = None,
    w1_blockscale: torch.Tensor | None = None,
    w2_blockscale: torch.Tensor | None = None,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
    exl3_layer: object | None = None,
    exl3_device: torch.device | str | None = None,
    dummy_scale: torch.Tensor | None = None,
    immutable_input_scales: bool = False,
) -> B12XFP4ExpertWeights:
    """Transfer source tensors into the planner-selected runtime owner.

    immutable_input_scales promises that input scale values cannot change
    until this owner and its captured graphs are discarded. It permits a
    preparation-time equality proof without changing mutable-scale callers.
    """

    if not isinstance(plan, MoEWeightPreparationPlan):
        raise TypeError("plan must be a MoEWeightPreparationPlan")
    actual_dtype = str(params_dtype).removeprefix("torch.")
    if actual_dtype != plan.io_dtype:
        raise ValueError(
            f"params_dtype={actual_dtype!r} does not match plan dtype={plan.io_dtype!r}"
        )
    if plan.source_format == "exl3":
        from b12x.moe._shared.kernels.w4a16.exl3 import (
            Exl3Layer,
            prepare_exl3_moe_weights,
        )

        if not isinstance(exl3_layer, Exl3Layer):
            raise ValueError(
                "exl3 preparation requires exl3_layer, a Exl3Layer extent from "
                "read_exl3_layer"
            )
        if exl3_device is None:
            raise ValueError(
                "exl3 preparation requires exl3_device, the CUDA device that "
                "owns the prepared weights"
            )
        if len(plan.quant_modes) != 1 or not plan.quant_modes <= frozenset(
            {"w4a16", "w4a8_mx"}
        ):
            raise ValueError(
                "exl3 weights support exactly one of quant_mode='w4a16' or "
                "quant_mode='w4a8_mx'"
            )
        manifest = exl3_layer.manifest
        if manifest.codebook != plan.trellis_codebook:
            raise ValueError(
                f"exl3 manifest codebook {manifest.codebook!r} does not match "
                f"the plan's trellis_codebook {plan.trellis_codebook!r}"
            )
        if manifest.rates.bits is not None and manifest.rates.bits != plan.trellis_bits:
            raise ValueError(
                f"exl3 manifest declares bits={manifest.rates.bits}; the plan "
                f"declares trellis_bits={plan.trellis_bits}"
            )
        if manifest.rates.structure != (plan.trellis_rate_granularity or "uniform"):
            raise ValueError(
                "exl3 manifest rate structure "
                f"{manifest.rates.structure!r} does not match the plan's "
                f"{plan.trellis_rate_granularity!r}"
            )
        if (manifest.rates.pair_kinds or None) != plan.trellis_pair_kinds:
            raise ValueError(
                "exl3 manifest pair kinds do not match the plan's trellis_pair_kinds"
            )
        if manifest.hadamard.intermediate_hadamard != plan.intermediate_hadamard:
            raise ValueError(
                "exl3 manifest intermediate-Hadamard declaration does not match the plan"
            )
        if (
            manifest.geometry.num_experts != plan.num_experts
            or manifest.geometry.hidden_size != plan.hidden_size
        ):
            raise ValueError(
                "exl3 manifest geometry does not match the plan's expert "
                "count or hidden size"
            )
        if exl3_layer.local_intermediate_size != plan.intermediate_size:
            raise ValueError(
                f"exl3 extent covers {exl3_layer.local_intermediate_size} "
                "intermediate channels; the plan declares "
                f"{plan.intermediate_size}"
            )
        value = prepare_exl3_moe_weights(
            exl3_layer,
            activation=plan.activation,
            device=exl3_device,
            params_dtype=torch.float16,
            tile_config=plan.trellis_tile_config,
            dummy_scale=dummy_scale,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode=next(iter(plan.quant_modes)),
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=value,
        )
        input_scale = torch.ones((), dtype=torch.float32, device=value.w13.device)
        return B12XFP4ExpertWeights(
            plan=plan,
            a1_gscale=a1_gscale if a1_gscale is not None else input_scale,
            w1_fp4=value.w13,
            w1_blockscale=value.w13_scale,
            w1_alphas=value.w13_global_scale,
            a2_gscale=a2_gscale if a2_gscale is not None else input_scale,
            w2_fp4=value.w2,
            w2_blockscale=value.w2_scale,
            w2_alphas=value.w2_global_scale,
            representation=representation,
        )
    if w1_fp4 is None or w2_fp4 is None:
        raise ValueError("weight preparation requires both w1_fp4 and w2_fp4")

    required_tensors = {
        "w1_global_scale": w1_global_scale,
        "w2_global_scale": w2_global_scale,
        "w1_blockscale": w1_blockscale,
        "w2_blockscale": w2_blockscale,
        "a1_gscale": a1_gscale,
        "a2_gscale": a2_gscale,
    }
    missing = sorted(name for name, value in required_tensors.items() if value is None)
    if missing:
        raise ValueError(
            "non-Trellis weight preparation is missing: " + ", ".join(missing)
        )
    assert w1_global_scale is not None and w2_global_scale is not None
    assert w1_blockscale is not None and w2_blockscale is not None
    assert a1_gscale is not None and a2_gscale is not None

    if int(w1_fp4.shape[0]) != plan.num_experts:
        raise ValueError(
            f"w1 expert count {int(w1_fp4.shape[0])} does not match "
            f"plan expert count {plan.num_experts}"
        )

    w1_runtime_alphas = _prepare_expert_scale_vector(
        w1_global_scale,
        plan.num_experts,
        name="w1_global_scale",
    )
    w2_runtime_alphas = _prepare_expert_scale_vector(
        w2_global_scale,
        plan.num_experts,
        name="w2_global_scale",
    )
    if plan.prepares_runtime_alphas:
        w1_runtime_alphas, w2_runtime_alphas = _prepare_modelopt_nvfp4_runtime_alphas(
            w1_global_scale,
            a1_gscale,
            w2_global_scale,
            a2_gscale,
            weight_E=plan.num_experts,
        )

    representation: _PreparedWeightRepresentation | None = None
    if WeightPreparationTransform.W4A16_NATIVE in plan.transforms:
        if representation is not None:
            raise AssertionError("weight plan selected multiple prepared layouts")
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_w4a16_e8m0_native_weights,
            prepare_w4a16_modelopt_native_weights,
        )

        if plan.source_format == "modelopt_nvfp4":
            value = prepare_w4a16_modelopt_native_weights(
                w1_fp4,
                w1_blockscale,
                w1_global_scale,
                w2_fp4,
                w2_blockscale,
                w2_global_scale,
                activation=plan.activation,
                params_dtype=params_dtype,
                source_format=plan.source_format,
                w13_layout=plan.w13_layout,
            )
        elif plan.source_format == "fp4_e8m0_k32":
            value = prepare_w4a16_e8m0_native_weights(
                w1_fp4,
                w1_blockscale,
                w1_global_scale,
                w2_fp4,
                w2_blockscale,
                w2_global_scale,
                activation=plan.activation,
                params_dtype=params_dtype,
                w13_layout=plan.w13_layout,
            )
        else:
            raise AssertionError(
                f"planner selected native W4A16 for {plan.source_format!r}"
            )
        representation = _PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.SOURCE_NATIVE,
            value=value,
        )

    if WeightPreparationTransform.W4A16_PACKED in plan.transforms:
        if representation is not None:
            raise AssertionError("weight plan selected multiple prepared layouts")
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_w4a16_packed_weights,
        )

        value = prepare_w4a16_packed_weights(
            w1_fp4,
            w1_blockscale,
            w1_global_scale,
            w2_fp4,
            w2_blockscale,
            w2_global_scale,
            activation=plan.activation,
            params_dtype=params_dtype,
            source_format=plan.source_format,
            w13_layout=plan.w13_layout,
            reuse_input_storage=True,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.MMA_PACKED,
            value=value,
        )

    if WeightPreparationTransform.W4A8_QMMA in plan.transforms:
        if representation is not None:
            raise AssertionError("weight plan selected multiple prepared layouts")
        value = _prepare_w4a8_from_e8m0_source(
            w1_fp4=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_global_scale=w1_global_scale,
            w2_fp4=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_global_scale=w2_global_scale,
            activation=plan.activation,
            params_dtype=params_dtype,
            source_format=plan.source_format,
            w13_layout=plan.w13_layout,
            n64_repack=plan.intermediate_size % 128 == 64,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode="w4a8_mx",
            layout=PreparedWeightLayout.QMMA_REPACKED,
            value=value,
        )

    if WeightPreparationTransform.W6A8_MXFP6 in plan.transforms:
        if representation is not None:
            raise AssertionError("weight plan selected multiple prepared layouts")
        # For the mxfp6_e2m3 source, the w1_fp4/w2_fp4 arguments carry the
        # packed FP6 code bytes and the blockscales carry unswizzled UE8M0
        # K/32 grids.
        value = _prepare_w6a8_from_mxfp6_source(
            w1_fp6=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_global_scale=w1_global_scale,
            w2_fp6=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_global_scale=w2_global_scale,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            activation=plan.activation,
            source_format=plan.source_format,
            w13_layout=plan.w13_layout,
            num_experts=plan.num_experts,
            hidden_size=plan.hidden_size,
            intermediate_size=plan.intermediate_size,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode="w6a8_mx",
            layout=PreparedWeightLayout.SOURCE_NATIVE,
            value=value,
        )

    if WeightPreparationTransform.W8A8_MXFP8 in plan.transforms:
        if representation is not None:
            raise AssertionError("weight plan selected multiple prepared layouts")
        # For the mxfp8_e8m0_k32 source, the w1_fp4/w2_fp4 arguments carry the
        # raw E4M3 code bytes (one per element) and the blockscales carry
        # unswizzled UE8M0 K/32 grids.
        value = _prepare_w8a8_from_mxfp8_source(
            w1_fp8=w1_fp4,
            w1_blockscale=w1_blockscale,
            w1_global_scale=w1_global_scale,
            w2_fp8=w2_fp4,
            w2_blockscale=w2_blockscale,
            w2_global_scale=w2_global_scale,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            activation=plan.activation,
            source_format=plan.source_format,
            w13_layout=plan.w13_layout,
            num_experts=plan.num_experts,
            hidden_size=plan.hidden_size,
            intermediate_size=plan.intermediate_size,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode="w8a8_mx",
            layout=PreparedWeightLayout.SOURCE_NATIVE,
            value=value,
        )

    canonical_w1 = w1_fp4
    canonical_s1 = w1_blockscale
    canonical_a1 = w1_runtime_alphas
    canonical_w2 = w2_fp4
    canonical_s2 = w2_blockscale
    canonical_a2 = w2_runtime_alphas
    source_mode_selected = bool({"nvfp4", "w4a8_nvfp4"} & plan.quant_modes)
    if representation is not None and not source_mode_selected:
        value = representation.value
        canonical_w1 = getattr(value, "w13_rp", getattr(value, "w13", None))
        canonical_s1 = getattr(value, "w13_sfb", getattr(value, "w13_scale", None))
        canonical_w2 = getattr(value, "w2_rp", getattr(value, "w2", None))
        canonical_s2 = getattr(value, "w2_sfb", getattr(value, "w2_scale", None))
        canonical_a1 = getattr(value, "w13_global_scale", canonical_a1)
        canonical_a2 = getattr(value, "w2_global_scale", canonical_a2)
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (canonical_w1, canonical_s1, canonical_w2, canonical_s2)
        ):
            raise RuntimeError(
                "prepared representation is missing canonical runtime tensors"
            )

    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=a1_gscale,
        w1_fp4=canonical_w1,
        w1_blockscale=canonical_s1,
        w1_alphas=canonical_a1,
        a2_gscale=a2_gscale,
        w2_fp4=canonical_w2,
        w2_blockscale=canonical_s2,
        w2_alphas=canonical_a2,
        representation=representation,
        immutable_input_scales=immutable_input_scales,
    )


def prepare_b12x_x4t_weights(*, plan, weights) -> B12XFP4ExpertWeights:
    """Keep exact nibbles and compressed scale planes in the expert owner."""
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_x4t_weights

    unit = torch.ones(plan.num_experts, dtype=torch.float32, device=weights.w13.device)
    native = WeightPreparationTransform.W4A16_NATIVE in plan.transforms
    value = prepare_w4a16_x4t_weights(
        weights.w13,
        weights.w13_scales,
        unit,
        weights.w2,
        weights.w2_scales,
        unit,
        weights.w13_scale_scratch,
        weights.w2_scale_scratch,
        activation=plan.activation,
        params_dtype=getattr(torch, plan.io_dtype),
        w13_layout=plan.w13_layout,
        weight_layout="modelopt" if native else "packed",
    )
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit,
        a2_gscale=unit,
        w1_fp4=value.w13,
        w2_fp4=value.w2,
        w1_blockscale=value.w13_scale,
        w2_blockscale=value.w2_scale,
        w1_alphas=value.w13_global_scale,
        w2_alphas=value.w2_global_scale,
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.SOURCE_NATIVE
            if native
            else PreparedWeightLayout.MMA_PACKED,
            value=value,
        ),
    )


def prepare_b12x_iq2_xs_weights(*, plan, weights) -> B12XFP4ExpertWeights:
    """Own compact IQ2_XS planes through the canonical expert package."""
    from b12x.moe._shared.kernels.w4a16.iq2_xs import prepare_iq2_xs_moe_weights

    value = prepare_iq2_xs_moe_weights(
        weights.w13,
        weights.w2,
        hidden_size=plan.hidden_size,
        intermediate_size=plan.intermediate_size,
        num_experts=plan.num_experts,
        activation=plan.activation,
        w13_layout=plan.w13_layout,
        codec=weights.codec,
    )
    unit = torch.ones((), dtype=torch.float32, device=value.w13.device)
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit,
        a2_gscale=unit,
        w1_fp4=value.w13,
        w2_fp4=value.w2,
        w1_blockscale=value.w13_scale,
        w2_blockscale=value.w2_scale,
        w1_alphas=value.w13_global_scale,
        w2_alphas=value.w2_global_scale,
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout(weights.codec + "_compact"),
            value=value,
        ),
    )


def prepare_b12x_trellis_v2_weights(
    *,
    plan: MoEWeightPreparationPlan,
    source: object,
    weights: object,
    device: torch.device | str | None = None,
    staging: object | None = None,
    rotation_dtype: torch.dtype | None = None,
) -> B12XFP4ExpertWeights:
    """Prepare canonical v2 trellis tensors into the normal expert owner."""

    from b12x.moe.fused_moe.source import TrellisSource
    from b12x.moe.fused_moe.trellis import prepare_trellis_weights
    from b12x.moe.fused_moe.weights import TrellisWeights

    if not isinstance(source, TrellisSource):
        raise TypeError("source must be a TrellisSource")
    if not isinstance(weights, TrellisWeights):
        raise TypeError("trellis preparation requires TrellisWeights")
    params_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(plan.io_dtype)
    if params_dtype is None:
        raise TypeError(f"unsupported trellis activation dtype {plan.io_dtype!r}")
    value = prepare_trellis_weights(
        source,
        weights,
        activation=plan.activation,
        params_dtype=rotation_dtype or params_dtype,
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        intermediate_size=plan.intermediate_size,
        device=device,
        staging=staging,
        tile_config=plan.trellis_tile_config,
    )
    representation = _PreparedWeightRepresentation(
        quant_mode="w4a16",
        layout=PreparedWeightLayout.TRELLIS_NATIVE,
        value=value,
    )
    unit_input = torch.ones((), dtype=torch.float32, device=value.w13.device)
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit_input,
        w1_fp4=value.w13,
        w1_blockscale=value.w13_scale,
        w1_alphas=value.w13_global_scale,
        a2_gscale=unit_input,
        w2_fp4=value.w2,
        w2_blockscale=value.w2_scale,
        w2_alphas=value.w2_global_scale,
        representation=representation,
    )


def prepare_w4a16_fc2_e8m0(
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
):
    """Prepare a source-native MXFP4 down projection for FC2-only execution."""

    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_fc2_e8m0_weights,
    )

    return prepare_w4a16_fc2_e8m0_weights(
        w2_fp4,
        w2_e8m0_scale,
        params_dtype=torch.bfloat16,
    )


def prewarm_w4a16_fc2_e8m0(experts, *, route_ids_dtype: torch.dtype):
    """Resolve and return the native FC2 carrier before graph capture."""

    from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
    from b12x.moe._shared.kernels.w4a16.prepare import W4A16FC2Weights

    if isinstance(experts, W4A16FC2Weights):
        hidden_size = experts.hidden_size
        intermediate_size = experts.intermediate_size
        num_experts = experts.num_experts
        device = experts.w2.device
    elif isinstance(experts, B12XFP4ExpertWeights):
        if experts.plan.required_weight_layout("w4a16") is None:
            raise ValueError("prepared expert package has no W4A16 representation")
        hidden_size = experts.hidden_size
        intermediate_size = experts.intermediate_size
        num_experts = experts.num_experts
        device = experts.w2_fp4.device
    else:
        raise TypeError("experts must carry a prepared W4A16 FC2 representation")
    return w4a16_kernel._compile_w4a16_fc2_direct(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk_ids_dtype=route_ids_dtype,
        device=device,
    )


def run_w4a16_fc2_e8m0(
    intermediate: torch.Tensor,
    experts,
    route_expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    launcher=None,
) -> torch.Tensor:
    """Execute route-major BF16 activations through native MXFP4 FC2 weights."""

    # Importing the kernel module registers the custom launch op.
    from b12x.moe._shared.kernels.w4a16 import kernel as _w4a16_kernel  # noqa: F401
    from b12x.moe._shared.kernels.w4a16.prepare import W4A16FC2Weights

    if isinstance(experts, B12XFP4ExpertWeights):
        experts = experts.representation_for("w4a16")
    if not isinstance(experts, W4A16FC2Weights):
        raise TypeError("experts must carry a prepared W4A16 FC2 representation")
    if intermediate.ndim != 2 or intermediate.dtype != torch.bfloat16:
        raise TypeError("intermediate must be rank-2 torch.bfloat16")
    if not intermediate.is_cuda or not intermediate.is_contiguous():
        raise ValueError("intermediate must be contiguous CUDA storage")
    m, intermediate_size = map(int, intermediate.shape)
    if m <= 0:
        raise ValueError("FC2-only execution requires at least one route")
    if intermediate_size != experts.intermediate_size:
        raise ValueError(
            f"intermediate width {intermediate_size} does not match "
            f"prepared width {experts.intermediate_size}"
        )
    if route_expert_ids.ndim not in (1, 2) or route_expert_ids.numel() != m:
        raise ValueError("route_expert_ids must contain one ID per route")
    if route_expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("route_expert_ids must be int32 or int64")
    if route_weights.ndim not in (1, 2) or route_weights.numel() != m:
        raise ValueError("route_weights must contain one value per route")
    if route_weights.dtype != torch.float32:
        raise TypeError("route_weights must be torch.float32")
    route_expert_ids = route_expert_ids.reshape(-1)
    route_weights = route_weights.reshape(-1)
    for name, tensor in (
        ("route_expert_ids", route_expert_ids),
        ("route_weights", route_weights),
    ):
        if tensor.device != intermediate.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {intermediate.device}")
    if experts.w2.device != intermediate.device:
        raise ValueError("prepared FC2 weights and activations must share a device")
    if output is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FC2-only output must be preallocated during capture")
        output = torch.empty(
            (m, experts.hidden_size),
            dtype=torch.bfloat16,
            device=intermediate.device,
        )
    elif (
        tuple(output.shape) != (m, experts.hidden_size)
        or output.dtype != torch.bfloat16
        or output.device != intermediate.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous BF16 [routes, hidden_size] on the input device"
        )

    if launcher is None:
        torch.ops.b12x.w4a16_fc2_direct_launch(
            intermediate,
            experts.w2,
            experts.w2_scale,
            experts.w2_global_scale,
            route_expert_ids,
            route_weights,
            output,
            experts.workspace[-2:-1],
            experts.workspace[-1:],
            m,
            experts.hidden_size,
            experts.intermediate_size,
            experts.num_experts,
            int(current_cuda_stream()),
        )
    else:
        _w4a16_kernel._w4a16_fc2_direct_launch_flat(
            intermediate,
            experts.w2,
            experts.w2_scale,
            experts.w2_global_scale,
            route_expert_ids,
            route_weights,
            output,
            experts.workspace[-2:-1],
            experts.workspace[-1:],
            m,
            experts.hidden_size,
            experts.intermediate_size,
            experts.num_experts,
            int(current_cuda_stream()),
            launcher=launcher,
        )
    return output


def _band_runs_direct_micro(
    *,
    num_tokens: int,
    k: int,
    n: int,
    num_topk: int,
    weight_E: int,
    activation: str,
    quant_mode: str,
) -> bool:
    """Whether direct-micro can run this (tiny-decode band) shape.

    Direct-micro owns the tiny-decode band when it supports the shape. Shapes it
    rejects (for example, more than 12 direct K segments, ``k % 128 != 0``, or
    ``n % 16 != 0``) route to the dynamic grouped GEMM.
    """
    _qm = _normalize_quant_mode(quant_mode)
    if _qm != "nvfp4" and not _is_w4a8_quant_mode(_qm):
        return False
    micro_cls = _get_activation_kernel_spec(
        activation, quant_mode=quant_mode
    ).micro_kernel_cls
    return micro_cls.is_supported(
        m=num_tokens, k=k, n=n, num_topk=num_topk, weight_E=weight_E
    )


def _resolve_workspace_layout(
    *,
    num_tokens: int,
    weight_E: int,
    num_topk: int,
    k: int,
    n: int,
    activation: str,
    quant_mode: str = "nvfp4",
    source_format: str = "modelopt_nvfp4",
    weight_layout: str | None = None,
    decode_config: MoeDecodeConfig,
) -> tuple[str, int, int]:
    routed_rows = num_tokens * num_topk
    normalized_quant_mode = _normalize_quant_mode(quant_mode)
    if not isinstance(decode_config, MoeDecodeConfig):
        raise TypeError("decode_config must be a concrete MoeDecodeConfig")
    implementation = decode_config.backend
    if normalized_quant_mode == "w4a16":
        if implementation != "w4a16":
            raise RuntimeError("W4A16 policy did not resolve the W4A16 family")
        return implementation, weight_E, max(1, routed_rows)
    if implementation == "micro":
        return implementation, max(1, routed_rows), max(1, routed_rows)
    if implementation != "dynamic":
        raise RuntimeError(
            f"unsupported {normalized_quant_mode} MoE implementation {implementation!r}"
        )
    dynamic_n = _dynamic_kernel_intermediate_size(n, normalized_quant_mode)
    tile_m, _ = _select_dynamic_tile_mn(
        routed_rows,
        dynamic_n,
        normalized_quant_mode,
        num_experts=weight_E,
        activation=activation,
        planned_tile_m=decode_config.dynamic_tile_m,
    )
    return implementation, weight_E, align_up(routed_rows, tile_m)


def plan_tp_moe_execution(
    *,
    num_tokens: int,
    num_topk: int,
    device: torch.device | str,
    weight_plan: MoEWeightPreparationPlan,
    quant_mode: str,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    apply_router_weight_on_input: bool = False,
    deterministic_output: bool | None = None,
    decode_config: MoeDecodeConfig,
) -> TPMoEPlan:
    """Lower one recipe from an authoritative weight-preparation plan.

    ``implementation`` names the concrete kernel family. ``spec`` and
    ``execution`` carry the independent numeric and scheduling decisions.
    """
    if not isinstance(weight_plan, MoEWeightPreparationPlan):
        raise TypeError("weight_plan must be a MoEWeightPreparationPlan")
    quant_mode = _normalize_quant_mode_for_source(
        quant_mode,
        weight_plan.source_format,
    )
    if quant_mode not in weight_plan.quant_modes:
        raise ValueError(f"quant_mode={quant_mode!r} is absent from the weight plan")
    try:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[weight_plan.io_dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported MoE plan dtype {weight_plan.io_dtype!r}") from exc
    device = torch.device(device)
    if not isinstance(decode_config, MoeDecodeConfig):
        raise TypeError("decode_config must be a concrete MoeDecodeConfig")
    weight_E = weight_plan.num_experts
    k = weight_plan.hidden_size
    n = weight_plan.intermediate_size
    source_format = weight_plan.source_format
    activation = weight_plan.activation
    w13_layout = weight_plan.w13_layout
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if quant_mode == "w4a16":
        _activation_w1_rows(activation, 1)
    else:
        activation = _get_activation_kernel_spec(
            activation, quant_mode=quant_mode
        ).activation
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    routed_rows = num_tokens * num_topk
    implementation, state_E, max_rows = _resolve_workspace_layout(
        num_tokens=num_tokens,
        weight_E=weight_E,
        num_topk=num_topk,
        k=k,
        n=n,
        activation=activation,
        quant_mode=quant_mode,
        source_format=source_format,
        weight_layout=weight_plan.w4a8_weight_layout,
        decode_config=decode_config,
    )
    dynamic_physical_tiles = None
    dynamic_task_capacity = None
    dynamic_tile_m = None
    dynamic_tile_n = None
    max_tokens_per_launch = num_tokens
    if implementation == "dynamic":
        dynamic_kernel_n = _dynamic_kernel_intermediate_size(n, quant_mode)
        # Tile planner (same routed_rows/n as _plan_core_workspace and the kernel
        # build) -> the task/row geometry sized here matches what the kernel
        # indexes. Must NOT use the bare _dynamic_tile_m (would size for 128 while
        # the kernel runs the planner's tile -> scratch under-size).
        dynamic_tile_m, dynamic_tile_n = _select_dynamic_tile_mn(
            routed_rows,
            dynamic_kernel_n,
            quant_mode,
            num_experts=state_E,
            activation=activation,
            planned_tile_m=decode_config.dynamic_tile_m,
        )
        dynamic_physical_tiles, gate_tile_cnt, dynamic_task_capacity = (
            _dynamic_task_geometry(
                state_E,
                dynamic_kernel_n,
                routed_rows,
                tile_m=dynamic_tile_m,
                tile_n=dynamic_tile_n,
            )
        )
        if _dynamic_direct_routing_selected(
            route_mode=decode_config.dynamic_route_mode or "",
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=state_E,
            n=dynamic_kernel_n,
            deterministic_output=False,
            planned_tile_m=dynamic_tile_m,
        ):
            direct_groups = max(
                1,
                (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK,
            )
            dynamic_physical_tiles = max(dynamic_physical_tiles, routed_rows)
            dynamic_task_capacity = max(
                dynamic_task_capacity, routed_rows * direct_groups
            )
        max_tokens_per_launch = _dynamic_token_chunk_limit(
            weight_E,
            k,
            n,
            num_topk,
            quant_mode,
        )
    spec = make_moe_spec(
        quant_mode=quant_mode,
        source_format=source_format,
        activation=activation,
        io_dtype=str(dtype).removeprefix("torch."),
        w13_layout=w13_layout,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    execution_scheduler = None
    if implementation == "micro":
        regime = MoERegime.DIRECT
    elif implementation == "dynamic":
        dynamic_work_source = _dynamic_work_source()
        if dynamic_work_source == "ready_queue":
            regime = MoERegime.STREAMING_FUSED
        else:
            regime = MoERegime.MATERIALIZED_FUSED
            execution_scheduler = (
                WorkScheduler.ATOMIC_QUEUE
                if dynamic_work_source == "materialized_queue"
                else WorkScheduler.PERSISTENT_GRID
            )
    else:
        regime = MoERegime.MATERIALIZED_PERSISTENT
    if deterministic_output is None:
        deterministic_output = _dynamic_deterministic_output_enabled(
            quant_mode=quant_mode,
            device=device,
        )
    route_block_rows = None
    if implementation == "w4a16":
        from b12x.moe._shared.kernels.w4a16.host import (
            select_route_block_size_m,
        )

        route_block_rows = select_route_block_size_m(
            num_tokens,
            num_topk,
            weight_E,
        )
    execution = lower_moe_execution(
        spec,
        regime=regime,
        tile_m=dynamic_tile_m,
        tile_n=dynamic_tile_n,
        route_block_rows=route_block_rows,
        deterministic_output=bool(deterministic_output),
        scheduler=execution_scheduler,
        required_weight_layout=weight_plan.required_weight_layout(quant_mode),
    )
    compact_w4a8_micro = bool(
        implementation == "micro"
        and quant_mode == "w4a8_mx"
        and source_format == "fp4_e8m0_k32"
        and n % 128 == 64
    )
    if compact_w4a8_micro:
        if apply_router_weight_on_input:
            raise ValueError("compact W4A8 micro requires route weights after FC2")
        execution = replace(
            execution,
            gemm_engine=GemmEngine.MXFP8_QMMA,
            reduction=OutputReduction.ROUTE_BUFFER_TOPK_SUM,
            tile_m=16,
            tile_n=128,
        )
    if not weight_plan.supports(
        quant_mode=quant_mode,
        execution=execution,
    ):
        raise RuntimeError(
            f"weight plan does not support execution layout "
            f"{execution.weight_layout.value!r}"
        )
    return TPMoEPlan(
        spec=spec,
        execution=execution,
        implementation=implementation,
        quant_mode=quant_mode,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        state_E=state_E,
        weight_E=weight_E,
        routed_rows=routed_rows,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        dtype=dtype,
        max_tokens_per_launch=max_tokens_per_launch,
        dynamic_physical_tiles=dynamic_physical_tiles,
        dynamic_task_capacity=dynamic_task_capacity,
        decode_config=decode_config,
    )


def _validate_workspace(
    workspace: object,
    *,
    plan: TPMoEPlan,
) -> None:
    def _canonical_device(device: torch.device) -> torch.device:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    expected = (
        plan.implementation,
        plan.quant_mode,
        plan.weight_E,
        plan.k,
        plan.n,
        plan.num_topk,
        _canonical_device(plan.device),
        plan.dtype,
    )
    actual = (
        workspace.implementation,
        workspace.quant_mode,
        workspace.weight_E,
        workspace.k,
        workspace.n,
        workspace.num_topk,
        _canonical_device(workspace.device),
        workspace.dtype,
    )
    if actual != expected:
        raise ValueError(
            "workspace metadata mismatch: "
            f"expected {(plan.implementation, plan.quant_mode, plan.weight_E, plan.k, plan.n, plan.num_topk, plan.device, plan.dtype)}, "
            f"got {actual}"
        )
    if (
        plan.implementation == "micro"
        and plan.quant_mode == "w4a8_mx"
        and plan.n % 128 == 64
    ):
        from b12x.moe._shared.kernels.w4a8_compact_micro import (
            micro_scratch_nbytes,
        )

        required = micro_scratch_nbytes(
            plan.max_rows // plan.num_topk,
            plan.k,
            plan.n,
            plan.num_topk,
        )
        if workspace.micro_intermediate.nbytes < required:
            raise ValueError(
                "compact W4A8 micro workspace is smaller than its planned capacity"
            )
    if plan.implementation == "w4a16":
        if not isinstance(workspace, TPW4A16Workspace):
            raise TypeError("expected a TPW4A16Workspace for the W4A16 backend")
        if workspace.activation != plan.activation:
            raise ValueError("workspace activation mismatch")
        if workspace.state_E < plan.state_E:
            raise ValueError(
                "workspace expert capacity mismatch: "
                f"expected at least {plan.state_E}, got {workspace.state_E}"
            )
        if workspace.max_rows < plan.max_rows:
            raise ValueError(
                "workspace row capacity mismatch: "
                f"expected at least {plan.max_rows}, got {workspace.max_rows}"
            )
        if workspace.routed_rows_capacity < plan.routed_rows:
            raise ValueError(
                "workspace routed-row capacity mismatch: "
                f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
            )
        return
    if workspace.state_E < plan.state_E:
        raise ValueError(
            "workspace expert capacity mismatch: "
            f"expected at least {plan.state_E}, got {workspace.state_E}"
        )
    if workspace.max_rows < plan.max_rows:
        raise ValueError(
            "workspace row capacity mismatch: "
            f"expected at least {plan.max_rows}, got {workspace.max_rows}"
        )
    if plan.implementation == "micro" and not isinstance(workspace, TPMicroWorkspace):
        raise TypeError("expected a TPMicroWorkspace for the direct-micro backend")
    if plan.implementation == "dynamic" and not isinstance(
        workspace, TPDynamicWorkspace
    ):
        raise TypeError("expected a TPDynamicWorkspace for the dynamic backend")
    if (
        isinstance(workspace, TPMicroWorkspace)
        and workspace.routed_rows_capacity < plan.routed_rows
    ):
        raise ValueError(
            "workspace routed-row capacity mismatch: "
            f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and workspace.routed_rows_capacity < plan.routed_rows
    ):
        raise ValueError(
            "workspace routed-row capacity mismatch: "
            f"expected at least {plan.routed_rows}, got {workspace.routed_rows_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and plan.dynamic_physical_tiles is not None
        and workspace.physical_tiles_capacity < plan.dynamic_physical_tiles
    ):
        raise ValueError(
            "workspace physical-tile capacity mismatch: "
            f"expected at least {plan.dynamic_physical_tiles}, got {workspace.physical_tiles_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and plan.dynamic_task_capacity is not None
        and workspace.task_capacity < plan.dynamic_task_capacity
    ):
        raise ValueError(
            "workspace task capacity mismatch: "
            f"expected at least {plan.dynamic_task_capacity}, got {workspace.task_capacity}"
        )
    if (
        isinstance(workspace, TPDynamicWorkspace)
        and plan.deterministic_output
        and workspace.route_output.numel() < plan.routed_rows * plan.k
    ):
        raise ValueError(
            "workspace deterministic route-output capacity mismatch: "
            f"expected at least {plan.routed_rows * plan.k} elements, got "
            f"{workspace.route_output.numel()}"
        )


def _workspace_pool_key(
    implementation: str,
    *,
    quant_mode: str,
    activation: str,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    # Pool-backed workspaces are capacity-based. Avoid
    # exact-shape keys here or long-tail prompt lengths will accumulate one
    # retained workspace per distinct routed-row count.
    if implementation in ("micro", "dynamic", "w4a16"):
        state_E = -1
        max_rows = -1
    return (
        implementation,
        quant_mode,
        activation if implementation == "w4a16" else "",
        state_E,
        weight_E,
        max_rows,
        k,
        n,
        num_topk,
        device.index or 0,
        dtype,
    )


def _lookup_capture_micro_workspace(
    workspace: TPMoEWorkspacePool,
    *,
    plan: TPMoEPlan,
) -> TPMicroWorkspace | None:
    if plan.implementation != "micro":
        return None
    for candidate in workspace.workspaces.values():
        if not isinstance(candidate, TPMicroWorkspace):
            continue
        if (
            candidate.implementation != plan.implementation
            or candidate.quant_mode != plan.quant_mode
            or candidate.weight_E != plan.weight_E
            or candidate.k != plan.k
            or candidate.n != plan.n
            or candidate.num_topk != plan.num_topk
            or candidate.device != plan.device
            or candidate.dtype != plan.dtype
        ):
            continue
        if candidate.state_E < plan.state_E:
            continue
        if candidate.max_rows < plan.max_rows:
            continue
        if candidate.routed_rows_capacity < plan.routed_rows:
            continue
        return candidate
    return None


def _normalize_w4a16_swiglu_limit(value: float | None) -> float | None:
    return None if value is None else float(value)


def _validate_frozen_w4a16_launch(
    workspace: TPW4A16Workspace,
    *,
    plan: TPMoEPlan,
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    collect_activation_amax: bool = False,
) -> None:
    scale_format = _normalize_w4a16_scale_format(scale_format)
    token_count = int(plan.max_tokens_per_launch)
    planned_capacity = min(
        (
            planned
            for planned in workspace.planned_token_counts
            if planned >= token_count
        ),
        default=None,
    )
    if planned_capacity is None:
        raise RuntimeError(
            "frozen W4A16 MoE workspace was asked to launch an unplanned token "
            f"count: tokens={token_count}, planned={sorted(workspace.planned_token_counts)}"
        )
    if bool(apply_router_weight_on_input) != bool(
        workspace.planned_apply_router_weight_on_input
    ):
        raise RuntimeError(
            "frozen W4A16 MoE workspace apply_router_weight_on_input mismatch: "
            f"requested={bool(apply_router_weight_on_input)}, "
            f"planned={workspace.planned_apply_router_weight_on_input}"
        )
    requested_limit = _normalize_w4a16_swiglu_limit(swiglu_limit)
    if requested_limit != workspace.planned_swiglu_limit:
        raise RuntimeError(
            "frozen W4A16 MoE workspace swiglu_limit mismatch: "
            f"requested={requested_limit}, planned={workspace.planned_swiglu_limit}"
        )
    if float(swiglu_alpha) != float(workspace.planned_swiglu_alpha):
        raise RuntimeError(
            "frozen W4A16 MoE workspace swiglu_alpha mismatch: "
            f"requested={float(swiglu_alpha)}, planned={workspace.planned_swiglu_alpha}"
        )
    if float(swiglu_beta) != float(workspace.planned_swiglu_beta):
        raise RuntimeError(
            "frozen W4A16 MoE workspace swiglu_beta mismatch: "
            f"requested={float(swiglu_beta)}, planned={workspace.planned_swiglu_beta}"
        )
    planned_scale_format = _normalize_w4a16_scale_format(
        getattr(workspace, "planned_scale_format", "e4m3_k16")
    )
    if scale_format != planned_scale_format:
        raise RuntimeError(
            "frozen W4A16 MoE workspace scale_format mismatch: "
            f"requested={scale_format!r}, planned={planned_scale_format!r}"
        )
    fused = workspace.planned_fused_moe_launches.get(
        (weight_layout, scale_format, planned_capacity, bool(collect_activation_amax))
    )
    if fused is None:
        raise RuntimeError(
            "frozen W4A16 MoE workspace is missing its preplanned fused launch "
            f"for capacity={planned_capacity}, weight_layout={weight_layout!r}, "
            f"scale_format={scale_format!r}, "
            f"collect_activation_amax={bool(collect_activation_amax)}"
        )
    fused_reduces_routes = bool(
        getattr(fused, "tc_decode_fused_sum", False)
        or getattr(fused, "prefill_fused_sum_fp32", False)
    )
    has_topk_sum = planned_capacity in workspace.planned_topk_sum_launches
    if workspace.full_rotation:
        has_topk_sum = any(
            isinstance(key, tuple) and key[0] == planned_capacity
            for key in workspace.planned_topk_sum_launches
        )
    if not fused_reduces_routes and not has_topk_sum:
        raise RuntimeError(
            "frozen W4A16 MoE workspace is missing its preplanned top-k sum launch "
            f"for capacity={planned_capacity}"
        )


def _w4a16_preplanned_launches(
    workspace: TPW4A16Workspace,
    *,
    token_count: int,
    weight_layout: str,
    scale_format: str = "e4m3_k16",
    collect_activation_amax: bool = False,
    route_ids_dtype: torch.dtype = torch.int32,
    use_route_expert_map: bool = False,
    use_output_expert_map: bool = False,
    route_mode: str = "auto",
) -> tuple[object | None, object | None]:
    token_count = int(token_count)
    scale_format = _normalize_w4a16_scale_format(scale_format)
    collect_activation_amax = bool(collect_activation_amax)
    route_mode = str(route_mode).lower()
    if route_mode not in {"auto", "direct", "packed"}:
        raise ValueError("W4A16 route mode must be 'auto', 'direct', or 'packed'")
    if not workspace.planned_token_counts:
        return None, None
    # Prefer a preplanned TC-decode launch for an exact small-M decode size. It
    # was compiled at this exact token count (its FC2 atomically sums per-route
    # partials into the output), so it only applies when m matches exactly.
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _TC_DECODE_MAX_M,
        _w4a16_tc_decode_preferred,
    )

    sum_uses_expert_map = bool(use_output_expert_map or use_route_expert_map)
    if (
        route_mode != "packed"
        and use_route_expert_map
        and not collect_activation_amax
        and route_ids_dtype == torch.int32
    ):
        mapped_direct = workspace.planned_mapped_direct_launches.get(token_count)
        if mapped_direct is not None:
            if bool(getattr(mapped_direct, "tc_decode_fused_sum", False)):
                return mapped_direct, None
            return mapped_direct, workspace.planned_topk_sum_launches.get(
                (token_count, route_ids_dtype, sum_uses_expert_map)
            )

    prefer_tc_decode = route_mode == "direct" or (
        route_mode == "auto"
        and _w4a16_tc_decode_preferred(
            m=token_count,
            topk=workspace.num_topk,
            num_experts=workspace.weight_E,
            sms=get_num_sm(workspace.device),
        )
    )
    if (
        not collect_activation_amax
        and route_mode != "packed"
        and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
        and is_gated_moe_activation(workspace.activation)
        and token_count <= _TC_DECODE_MAX_M
        and prefer_tc_decode
    ):
        tc_decode = workspace.planned_tc_decode_launches.get(token_count)
        if tc_decode is not None:
            return tc_decode, workspace.planned_topk_sum_launches.get(token_count)
    if (
        not collect_activation_amax
        and route_mode == "direct"
        and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
    ):
        direct = workspace.planned_direct_topk_launches.get(token_count)
        if direct is not None:
            return direct, workspace.planned_topk_sum_launches.get(token_count)
    planned_capacity = min(
        (
            planned
            for planned in workspace.planned_token_counts
            if planned >= token_count
        ),
        default=None,
    )
    if planned_capacity is None:
        raise RuntimeError(
            "W4A16 MoE workspace was asked to launch an unplanned token count: "
            f"tokens={token_count}, planned={sorted(workspace.planned_token_counts)}"
        )
    fused = workspace.planned_fused_moe_launches.get(
        (weight_layout, scale_format, planned_capacity, collect_activation_amax)
    )
    if fused is not None and bool(
        getattr(fused, "tc_decode_fused_sum", False)
        or getattr(fused, "prefill_fused_sum_fp32", False)
    ):
        return fused, None
    topk_sum_key: object = planned_capacity
    if workspace.full_rotation:
        topk_sum_key = (
            planned_capacity,
            route_ids_dtype,
            sum_uses_expert_map,
        )
    topk_sum = workspace.planned_topk_sum_launches.get(topk_sum_key)
    if fused is None or topk_sum is None:
        raise RuntimeError(
            "W4A16 MoE workspace is missing preplanned launches for "
            f"capacity={planned_capacity}, weight_layout={weight_layout!r}, "
            f"scale_format={scale_format!r}, "
            f"collect_activation_amax={collect_activation_amax}"
        )
    return fused, topk_sum


def _resolve_workspace(
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
    *,
    plan: TPMoEPlan,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    input_scales_static: bool,
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    collect_activation_amax: bool = False,
) -> object:
    scale_format = _normalize_w4a16_scale_format(scale_format)
    if isinstance(workspace, (TPMoEWorkspace, TPW4A16Workspace)):
        _validate_workspace(workspace, plan=plan)
        if isinstance(workspace, TPDynamicWorkspace):
            _refresh_dynamic_workspace_scales(
                workspace,
                a1_gscale,
                a2_gscale,
                input_scales_static=input_scales_static,
            )
        return workspace

    if not isinstance(workspace, TPMoEWorkspacePool):
        raise TypeError(
            "workspace must be a TPMoEWorkspace, TPW4A16Workspace, or TPMoEWorkspacePool"
        )

    key = _workspace_pool_key(
        plan.implementation,
        state_E=plan.state_E,
        weight_E=plan.weight_E,
        max_rows=plan.max_rows,
        k=plan.k,
        n=plan.n,
        num_topk=plan.num_topk,
        device=plan.device,
        dtype=plan.dtype,
        quant_mode=plan.quant_mode,
        activation=plan.activation,
    )
    resolved = workspace.workspaces.get(key)
    if resolved is None and torch.cuda.is_current_stream_capturing():
        capture_micro = _lookup_capture_micro_workspace(workspace, plan=plan)
        if capture_micro is not None:
            # Capture may switch to a dedicated stream, but the micro
            # workspace is stream-agnostic scratch. Reuse the warmed eager
            # workspace instead of allocating a fresh one inside capture.
            workspace.workspaces[key] = capture_micro
            resolved = capture_micro
    if resolved is None:
        if workspace.frozen:
            raise RuntimeError(
                "frozen MoE workspace pool does not contain a preplanned workspace "
                f"for implementation={plan.implementation!r}, quant_mode={plan.quant_mode!r}, "
                f"tokens={plan.max_tokens_per_launch}, routed_rows={plan.routed_rows}"
            )
        if plan.implementation == "w4a16" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 workspace is not initialized for CUDA graph capture; "
                "run a warmup with the workspace pool or allocate a sufficient workspace before capture"
            )
        resolved = _alloc_workspace(
            plan.implementation,
            plan.quant_mode,
            plan.state_E,
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            a1_gscale,
            a2_gscale,
            routed_rows=plan.routed_rows,
            max_rows=plan.max_rows,
            input_scales_static=input_scales_static,
            deterministic_output=plan.deterministic_output,
            activation=plan.activation,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
            dynamic_tile_m=plan.execution.tile_m,
            dynamic_tile_n=plan.execution.tile_n,
            dynamic_route_mode=plan.decode_config.dynamic_route_mode,
            nvfp4_materialize_intermediate=plan.decode_config.nvfp4_materialize_intermediate,
            pool=workspace,
            storage_key=key,
        )
        workspace.workspaces[key] = resolved
        return resolved

    needs_growth = (
        resolved.state_E < plan.state_E
        or resolved.max_rows < plan.max_rows
        or (
            isinstance(resolved, (TPDynamicWorkspace, TPMicroWorkspace))
            and resolved.routed_rows_capacity < plan.routed_rows
        )
        or (
            isinstance(resolved, TPW4A16Workspace)
            and resolved.routed_rows_capacity < plan.routed_rows
        )
        or (
            isinstance(resolved, TPDynamicWorkspace)
            and plan.dynamic_physical_tiles is not None
            and resolved.physical_tiles_capacity < plan.dynamic_physical_tiles
        )
        or (
            isinstance(resolved, TPDynamicWorkspace)
            and plan.dynamic_task_capacity is not None
            and resolved.task_capacity < plan.dynamic_task_capacity
        )
        or (
            isinstance(resolved, TPDynamicWorkspace)
            and plan.deterministic_output
            and resolved.route_output.numel() < plan.routed_rows * plan.k
        )
    )
    if needs_growth:
        if workspace.frozen:
            raise RuntimeError(
                "frozen MoE workspace pool capacity is too small for a requested "
                f"launch: implementation={plan.implementation!r}, quant_mode={plan.quant_mode!r}, "
                f"tokens={plan.max_tokens_per_launch}, routed_rows={plan.routed_rows}"
            )
        if plan.implementation == "w4a16" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 workspace capacity is too small for CUDA graph capture; "
                "run an eager warmup with a larger routed-row budget before capture"
            )
        dynamic_tiles = plan.dynamic_physical_tiles
        dynamic_tasks = plan.dynamic_task_capacity
        if isinstance(resolved, TPDynamicWorkspace):
            dynamic_tiles = max(dynamic_tiles or 0, resolved.physical_tiles_capacity)
            dynamic_tasks = max(dynamic_tasks or 0, resolved.task_capacity)
        resolved = _alloc_workspace(
            plan.implementation,
            plan.quant_mode,
            max(plan.state_E, resolved.state_E),
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            a1_gscale,
            a2_gscale,
            routed_rows=max(
                plan.routed_rows, getattr(resolved, "routed_rows_capacity", 0)
            ),
            max_rows=max(plan.max_rows, resolved.max_rows),
            input_scales_static=input_scales_static,
            deterministic_output=plan.deterministic_output,
            activation=plan.activation,
            dynamic_physical_tiles=dynamic_tiles,
            dynamic_task_capacity=dynamic_tasks,
            dynamic_tile_m=plan.execution.tile_m,
            dynamic_tile_n=plan.execution.tile_n,
            dynamic_route_mode=plan.decode_config.dynamic_route_mode,
            nvfp4_materialize_intermediate=plan.decode_config.nvfp4_materialize_intermediate,
            pool=workspace,
            storage_key=key,
        )
        workspace.workspaces[key] = resolved
        return resolved

    if workspace.frozen and isinstance(resolved, TPW4A16Workspace):
        _validate_frozen_w4a16_launch(
            resolved,
            plan=plan,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            weight_layout=weight_layout,
            scale_format=scale_format,
            collect_activation_amax=collect_activation_amax,
        )

    if isinstance(resolved, TPDynamicWorkspace):
        _refresh_dynamic_workspace_scales(
            resolved,
            a1_gscale,
            a2_gscale,
            input_scales_static=input_scales_static,
            force=resolved.volatile_launch_state,
        )
    return resolved


def plan_tp_moe_arena_layout(
    *,
    max_tokens: int,
    num_topk: int,
    device: torch.device | str,
    weight_plan: MoEWeightPreparationPlan,
    quant_mode: str,
    core_token_counts: tuple[int, ...] | None = None,
    route_num_experts: int | None = None,
    route_logits_dtype: torch.dtype | None = None,
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    collect_activation_amax: bool = False,
    deterministic_output: bool | None = None,
    w4a16_block_size_m: int | None = None,
    w4a16_prefill_fused_sum: bool | None = None,
    decode_config: MoeDecodeConfig,
) -> TPMoEArenaLayout:
    """Compute the byte layout needed by one lane-owned MoE pool."""
    if not isinstance(weight_plan, MoEWeightPreparationPlan):
        raise TypeError("weight_plan must be a MoEWeightPreparationPlan")
    source_format = weight_plan.source_format
    quant_mode = _normalize_quant_mode_for_source(quant_mode, source_format)
    if quant_mode not in weight_plan.quant_modes:
        raise ValueError(f"quant_mode={quant_mode!r} is absent from the weight plan")
    activation = weight_plan.activation
    w13_layout = weight_plan.w13_layout
    w4a16_weight_layout = weight_plan.w4a16_weight_layout
    w4a16_scale_format = weight_plan.w4a16_scale_format
    weight_E = weight_plan.num_experts
    k = weight_plan.hidden_size
    n = weight_plan.intermediate_size
    try:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[weight_plan.io_dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported MoE plan dtype {weight_plan.io_dtype!r}") from exc
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    device = torch.device(device)
    if deterministic_output is None:
        deterministic_output = _dynamic_deterministic_output_enabled(
            quant_mode=quant_mode,
            device=device,
        )
    else:
        deterministic_output = bool(deterministic_output)
    max_tokens = max(int(max_tokens), 1)
    weight_E = max(int(weight_E), 1)
    k = max(int(k), 1)
    n = max(int(n), 1)
    num_topk = max(int(num_topk), 1)
    core_token_counts = _arena_core_token_counts(
        max_tokens=max_tokens,
        num_topk=num_topk,
        core_token_counts=core_token_counts,
        quant_mode=quant_mode,
        # Trellis plans one fixed caller-owned capacity and its route-pack
        # wrapper already falls back to that exact capacity when the supplied
        # buffers are smaller than the generic compile bucket. Rounding a
        # 3072-token EXL3 plan to 4096 therefore reserves roughly 320 MiB of
        # unreachable activation scratch without reducing launch variants.
        bucket_w4a16_tokens=source_format not in _TRELLIS_SOURCE_FORMATS,
    )
    route_num_experts = int(
        route_num_experts if route_num_experts is not None else weight_E
    )
    route_logits_dtype = route_logits_dtype or dtype
    core_nbytes = 0
    for token_count in core_token_counts:
        plan = plan_tp_moe_execution(
            num_tokens=token_count,
            num_topk=num_topk,
            device=device,
            weight_plan=weight_plan,
            quant_mode=quant_mode,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            apply_router_weight_on_input=apply_router_weight_on_input,
            deterministic_output=deterministic_output,
            decode_config=decode_config,
        )
        core_plan = _plan_core_workspace(
            plan.implementation,
            plan.quant_mode,
            plan.state_E,
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            routed_rows=plan.routed_rows,
            max_rows=plan.max_rows,
            activation=plan.activation,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
            dynamic_tile_m=plan.execution.tile_m,
            dynamic_tile_n=plan.execution.tile_n,
            dynamic_route_mode=plan.decode_config.dynamic_route_mode,
            nvfp4_materialize_intermediate=plan.decode_config.nvfp4_materialize_intermediate,
            source_format=source_format,
            w13_layout=w13_layout,
            w4a16_weight_layout=w4a16_weight_layout,
            w4a16_scale_format=w4a16_scale_format,
            route_num_experts=route_num_experts,
            w4a16_block_size_m=w4a16_block_size_m,
            w4a16_prefill_fused_sum=w4a16_prefill_fused_sum,
            trellis_bits=weight_plan.trellis_bits or 3,
            trellis_tile_config=weight_plan.trellis_tile_config,
            trellis_pair_kinds=weight_plan.trellis_pair_kinds,
            trellis_codebook=_derive_trellis_codebook(
                weight_plan.source_format, weight_plan.trellis_codebook
            ),
            intermediate_hadamard=weight_plan.intermediate_hadamard,
            projection_mixed_trellis=_uses_projection_mixed_trellis(weight_plan),
            apply_router_weight_on_input=apply_router_weight_on_input,
            collect_activation_amax=collect_activation_amax,
            deterministic_output=plan.deterministic_output,
            swiglu_limit=plan.swiglu_limit,
            swiglu_alpha=plan.swiglu_alpha,
            swiglu_beta=plan.swiglu_beta,
        )
        core_nbytes = max(core_nbytes, _core_workspace_nbytes(core_plan))
    if route_num_experts > 0:
        route_nbytes = _route_workspace_nbytes(
            num_tokens=max_tokens,
            num_experts=route_num_experts,
            top_k=num_topk,
            logits_dtype=route_logits_dtype,
        )
    else:
        route_nbytes = 0
    route_nbytes = align_up(route_nbytes, 16)
    return TPMoEArenaLayout(
        route_workspace_nbytes=route_nbytes,
        core_workspace_nbytes=core_nbytes,
        total_nbytes=max(route_nbytes + core_nbytes, 1),
        core_token_counts=core_token_counts,
    )


def _plan_full_rotation_w4a16_launches(
    *,
    caps: TPMoEScratchCaps,
    core_plan: _TPCoreWorkspacePlan,
    capacity_tokens: int,
) -> tuple[
    tuple[tuple[int, object], ...],
    tuple[tuple[torch.dtype, bool, object], ...],
]:
    """Compile the fixed Trellis launches before serving memory profiling.

    The immutable scratch plan owns compiled programs independently of module
    cache eviction. Runtime binding performs no kernel resolution or allocation.
    """
    if not core_plan.full_rotation or core_plan.projection_mixed_trellis:
        return (), ()
    if caps.quant_mode != "w4a16":
        raise RuntimeError("full-rotation TP MoE requires W4A16 execution")
    if core_plan.device.type != "cuda":
        return (), ()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available before planning Trellis MoE")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Trellis launch planning cannot run during capture")

    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _DEFAULT_MAX_SHARED_MEM,
        compile_w4a16_fused_moe,
        compile_w4a16_topk_sum,
    )

    capacity_tokens = max(int(capacity_tokens), 1)
    if not core_plan.route_block_size_m:
        raise RuntimeError(
            "Trellis launch planning requires the arena's route_block_size_m"
        )
    block_size_m = int(core_plan.route_block_size_m)
    weight_layout = _normalize_w4a16_weight_layout(
        caps.w4a16_weight_layout
        or _w4a16_weight_layout_for_source(
            caps.source_format,
            intermediate_size=caps.n,
        )
    )
    scale_format = _normalize_w4a16_scale_format(
        caps.w4a16_scale_format or _w4a16_scale_format_for_source(caps.source_format)
    )
    if weight_layout != "trellis_t256" or scale_format != "e4m3_k32":
        raise RuntimeError(
            "full-rotation Trellis launch planning requires "
            "weight_layout='trellis_t256' and scale_format='e4m3_k32'; "
            f"got weight_layout={weight_layout!r}, scale_format={scale_format!r}"
        )
    w13_layout = "trellis_t256_proj"
    _, _, capacity_m_blocks = route_pack_capacity(
        capacity_tokens * core_plan.num_topk,
        block_size_m,
        core_plan.route_E,
        topk=core_plan.num_topk,
        bucket_tokens=False,
    )
    rotation_input_dtype = _w4a16_element_dtype(core_plan.dtype)

    with torch.cuda.device(core_plan.device):
        props = torch.cuda.get_device_properties(core_plan.device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(
                props,
                "shared_memory_per_block_optin",
                _DEFAULT_MAX_SHARED_MEM,
            )
        )
        # Decode plans are deliberately exact-M: the unified API's measured
        # speedup over the retired Trellis wrapper comes from specializing the
        # small live shapes instead of always launching its M=32 capacity
        # kernel.  Large prefill plans retain one capacity launch.
        fused_token_counts = (
            tuple(range(1, capacity_tokens + 1))
            if capacity_tokens <= 32
            else (capacity_tokens,)
        )

        def compile_fused(token_count: int, broadcast_suh: bool) -> object:
            return compile_w4a16_fused_moe(
                size_m=token_count,
                hidden_size=core_plan.k,
                intermediate_size=core_plan.n,
                num_experts=core_plan.weight_E,
                top_k=core_plan.num_topk,
                activation=core_plan.activation,
                apply_router_weight_on_input=caps.apply_router_weight_on_input,
                zero_fc2_output=False,
                moe_block_size=block_size_m,
                # Match the lazy unified path: specialize the live M while
                # retaining the caller-owned route arena's full grid capacity.
                max_m_blocks=capacity_m_blocks,
                element_dtype="fp16",
                fast_math=caps.w4a16_fast_math,
                sms=sms,
                max_shared_mem=max_shared_mem,
                swiglu_limit=core_plan.swiglu_limit,
                swiglu_alpha=core_plan.swiglu_alpha,
                swiglu_beta=core_plan.swiglu_beta,
                weight_layout=weight_layout,
                scale_format=scale_format,
                w13_layout=w13_layout,
                trellis_bits=core_plan.trellis_bits,
                trellis_codebook=core_plan.trellis_codebook or LUT_E4M3,
                trellis_decode_table=caps.trellis_decode_table,
                skip_empty_m_blocks=caps.w4a16_skip_empty_m_blocks,
                small_m_occupancy=caps.w4a16_small_m_occupancy,
                force_tile_config=core_plan.trellis_tile_config,
                intermediate_rotation=True,
                full_rotation=True,
                intermediate_hadamard=core_plan.intermediate_hadamard,
                rotation_input_dtype=rotation_input_dtype,
                broadcast_suh=broadcast_suh,
            )

        fused_launches = tuple(
            (token_count, compile_fused(token_count, broadcast_suh))
            for token_count in fused_token_counts
            for broadcast_suh in (False, True)
        )
        topk_sum_launches = tuple(
            (
                ids_dtype,
                mapped,
                compile_w4a16_topk_sum(
                    m=capacity_tokens,
                    topk=core_plan.num_topk,
                    hidden_size=core_plan.k,
                    element_dtype="fp16",
                    full_rotation=True,
                    intermediate_hadamard=core_plan.intermediate_hadamard,
                    num_experts=core_plan.weight_E,
                    route_num_experts=(core_plan.route_E if mapped else 0),
                    route_ids_dtype=ids_dtype,
                    use_expert_map=mapped,
                    broadcast_svh=broadcast_svh,
                ),
            )
            for ids_dtype in (torch.int32, torch.int64)
            for mapped in (False, True)
            for broadcast_svh in (False, True)
        )
    return fused_launches, topk_sum_launches


def _plan_projection_mixed_trellis_launches(
    *,
    caps: TPMoEScratchCaps,
    core_plan: _TPCoreWorkspacePlan,
    capacity_tokens: int,
) -> tuple[tuple[torch.dtype, bool, bool, object], ...]:
    """Compile every graph-reachable MCG K3/K4/K5 launch specialization."""

    if not core_plan.projection_mixed_trellis:
        return ()
    if core_plan.device.type != "cuda":
        return ()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available before planning Trellis MoE")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Trellis launch planning cannot run during capture")
    if not core_plan.route_block_size_m:
        raise RuntimeError(
            "projection-mixed Trellis requires a planned route block size"
        )

    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.kernel import _DEFAULT_MAX_SHARED_MEM
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        MixedTrellisBuffers,
        compile_mixed_trellis3,
        warmup_mixed_trellis_route_pack,
    )

    capacity_tokens = max(int(capacity_tokens), 1)
    block_size_m = int(core_plan.route_block_size_m)
    capacity_rows = capacity_tokens * int(core_plan.num_topk)
    route_slots = max_packed_route_slots(
        capacity_rows,
        block_size_m,
        core_plan.route_E,
    )
    route_blocks = (route_slots + block_size_m - 1) // block_size_m
    rotation_input_dtype = _w4a16_element_dtype(core_plan.dtype)

    with torch.cuda.device(core_plan.device):
        props = torch.cuda.get_device_properties(core_plan.device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(
                props,
                "shared_memory_per_block_optin",
                _DEFAULT_MAX_SHARED_MEM,
            )
        )
        launches = tuple(
            (
                ids_dtype,
                broadcast_suh,
                broadcast_svh,
                compile_mixed_trellis3(
                    size_m=capacity_tokens,
                    hidden_size=core_plan.k,
                    intermediate_size=core_plan.n,
                    tier0_num_experts=core_plan.weight_E,
                    tier1_num_experts=core_plan.weight_E,
                    tier2_num_experts=core_plan.weight_E,
                    route_num_experts=core_plan.route_E,
                    top_k=core_plan.num_topk,
                    max_m_blocks=route_blocks,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    force_tile_config=core_plan.trellis_tile_config
                    or (128, 256, 64, 256),
                    trellis_codebook="mcg",
                    moe_block_size=block_size_m,
                    rotation_input_dtype=rotation_input_dtype,
                    route_ids_dtype=ids_dtype,
                    broadcast_suh=broadcast_suh,
                    broadcast_svh=broadcast_svh,
                ),
            )
            for ids_dtype in (torch.int32, torch.int64)
            for broadcast_suh in (False, True)
            for broadcast_svh in (False, True)
        )
        for _, _, _, launch in launches:
            if int(launch.blocks_per_sm) > 4:
                raise RuntimeError(
                    "projection-mixed Trellis exceeds the planned workspace "
                    f"capacity: blocks_per_sm={int(launch.blocks_per_sm)}"
                )

        # Route-pack compilation uses only these metadata buffers. Avoid
        # materializing temporary activation scratch while the serving layer
        # is still establishing persistent-memory capacity.
        dummy_f16 = torch.empty(1, dtype=torch.float16, device=core_plan.device)
        dummy_f32 = torch.empty(1, dtype=torch.float32, device=core_plan.device)
        dummy_i32 = torch.empty(1, dtype=torch.int32, device=core_plan.device)
        route_buffers = MixedTrellisBuffers(
            rotation_gate=dummy_f16,
            rotation_up=dummy_f16,
            fc1=dummy_f16,
            activated=dummy_f16,
            fc2=dummy_f16,
            output=dummy_f32,
            packed_route_indices=torch.empty(
                route_slots, dtype=torch.int32, device=core_plan.device
            ),
            block_expert_ids=torch.empty(
                route_blocks, dtype=torch.int32, device=core_plan.device
            ),
            packed_route_count=torch.empty(
                1, dtype=torch.int32, device=core_plan.device
            ),
            expert_offsets=torch.empty(
                core_plan.route_E + 1,
                dtype=torch.int32,
                device=core_plan.device,
            ),
            expert_counts=torch.empty(
                core_plan.route_E,
                dtype=torch.int32,
                device=core_plan.device,
            ),
            fc1_scratch=dummy_f32,
            fc2_scratch=dummy_f32,
            workspace=dummy_i32,
        )
        expert_map = torch.arange(
            core_plan.route_E,
            dtype=torch.int32,
            device=core_plan.device,
        )
        for ids_dtype in (torch.int32, torch.int64):
            launch = next(item[3] for item in launches if item[0] == ids_dtype)
            warmup_mixed_trellis_route_pack(
                launch,
                route_buffers,
                expert_map=expert_map,
            )
    return launches


def _uses_projection_mixed_trellis(
    weight_plan: MoEWeightPreparationPlan,
) -> bool:
    return (
        weight_plan.source_format in _TRELLIS_SOURCE_FORMATS
        and weight_plan.trellis_codebook == "mcg"
        and weight_plan.trellis_rate_granularity == "per_expert_projection"
    )


def _bind_projection_mixed_trellis_from_views(
    *,
    scratch_plan: TPMoEScratchPlan,
    tensors: Mapping[str, torch.Tensor],
    a: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor | None,
    input_scales_static: bool,
    fast_math: bool | None,
    activation_amax: torch.Tensor | None,
    layer_idx: int | None,
    route_expert_map: torch.Tensor | None,
    output_expert_map: torch.Tensor | None,
) -> TPMoEFP4Binding:
    """Bind canonical MCG weights to caller-owned mixed-tier scratch views."""

    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        MixedTrellisBuffers,
        bind_mixed_trellis3,
    )
    from b12x.moe.fused_moe.trellis import PreparedProjectionTrellisWeights

    core_plan = scratch_plan._core_workspace_plan
    if not core_plan.projection_mixed_trellis:
        raise TypeError("scratch plan is not projection-mixed Trellis")
    if route_expert_map is not None or output_expert_map is not None:
        raise NotImplementedError(
            "projection-mixed Trellis does not yet support external expert maps"
        )
    if activation_amax is not None:
        raise NotImplementedError(
            "projection-mixed Trellis does not collect activation amax"
        )
    effective_fast_math = (
        scratch_plan.caps.w4a16_fast_math if fast_math is None else bool(fast_math)
    )
    if not effective_fast_math:
        raise NotImplementedError(
            "projection-mixed Trellis currently has only the fast-math kernel"
        )
    if a.ndim != 2 or tuple(a.shape[1:]) != (core_plan.k,):
        raise ValueError(
            f"input must have shape [M,{core_plan.k}], got {tuple(a.shape)}"
        )
    m = int(a.shape[0])
    capacity_tokens = core_plan.routed_rows // core_plan.num_topk
    if m <= 0 or m > capacity_tokens:
        raise ValueError(f"input rows must be in [1,{capacity_tokens}], got {m}")
    expected_routes = (m, core_plan.num_topk)
    if tuple(topk_weights.shape) != expected_routes:
        raise ValueError(
            f"topk_weights must have shape {expected_routes}, "
            f"got {tuple(topk_weights.shape)}"
        )
    if tuple(topk_ids.shape) != expected_routes:
        raise ValueError(
            f"topk_ids must have shape {expected_routes}, got {tuple(topk_ids.shape)}"
        )
    for name, tensor, dtype in (
        ("input", a, core_plan.dtype),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.device != a.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {a.device}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be torch.int32 or torch.int64")
    if topk_ids.device != a.device or not topk_ids.is_contiguous():
        raise ValueError(f"topk_ids must be contiguous on {a.device}")
    if int(a.data_ptr()) % 16:
        raise ValueError("input must have at least 16-byte alignment")

    prepared = experts.representation_for("w4a16")
    if not isinstance(prepared, PreparedProjectionTrellisWeights):
        raise TypeError(
            "projection-mixed scratch requires prepared b12x_trellis MCG weights"
        )
    if prepared.w13.device != a.device:
        raise ValueError(
            f"prepared weights must be on {a.device}, got {prepared.w13.device}"
        )

    broadcast_suh = int(prepared.rotations.gate_suh.shape[0]) == 1
    broadcast_svh = int(prepared.rotations.down_svh.shape[0]) == 1
    launch = next(
        (
            candidate
            for ids_dtype, candidate_suh, candidate_svh, candidate in (
                scratch_plan._mixed_trellis_launches
            )
            if ids_dtype == topk_ids.dtype
            and candidate_suh == broadcast_suh
            and candidate_svh == broadcast_svh
        ),
        None,
    )
    if launch is None:
        raise RuntimeError(
            "projection-mixed Trellis launch specialization was not planned"
        )

    output_cast_target = None
    if output is None:
        output_tensor = tensors["full_rotation_output"][:m]
    elif output.dtype == a.dtype:
        if (
            output.device != a.device
            or output.ndim != 2
            or tuple(output.shape)
            not in {(m, core_plan.k), tuple(tensors["full_rotation_output"].shape)}
            or not output.is_contiguous()
            or int(output.data_ptr()) % 16
        ):
            raise ValueError(
                "projection-mixed output must be contiguous, 16-byte-aligned "
                "and have live or capacity shape on the input device"
            )
        output_cast_target = output[:m]
        output_tensor = tensors["full_rotation_output"][:m]
    else:
        output_tensor = output
    if (
        tuple(output_tensor.shape) != (m, core_plan.k)
        or output_tensor.dtype != torch.float32
        or output_tensor.device != a.device
        or not output_tensor.is_contiguous()
        or int(output_tensor.data_ptr()) % 16
    ):
        raise ValueError(
            "projection-mixed output must be contiguous, 16-byte-aligned "
            f"float32[{m},{core_plan.k}] on {a.device}"
        )

    # This caller-owned arena may contain arbitrary bytes. The cooperative
    # mixed grid reads its barrier count and sense before writing them.
    tensors["kernel_workspace"].zero_()
    buffers = MixedTrellisBuffers(
        rotation_gate=tensors["rotation_a_gate"],
        rotation_up=tensors["rotation_a_up"],
        fc1=tensors["intermediate_cache13"].view(
            core_plan.routed_rows, 2 * core_plan.n
        ),
        activated=tensors["intermediate_cache2"].view(
            core_plan.routed_rows, core_plan.n
        ),
        fc2=tensors["rotation_a_gate"],
        output=output_tensor,
        packed_route_indices=tensors["packed_route_indices"],
        block_expert_ids=tensors["block_expert_ids"],
        packed_route_count=tensors["packed_route_count"],
        expert_offsets=tensors["expert_offsets"],
        expert_counts=tensors["expert_counts"],
        fc1_scratch=tensors["fc1_c_tmp"],
        fc2_scratch=tensors["fc2_c_tmp"],
        workspace=tensors["kernel_workspace"],
    )
    mixed_binding = bind_mixed_trellis3(
        prepared.tiers[0],
        prepared.tiers[1],
        prepared.tiers[2],
        prepared.global_to_combined,
        prepared.descriptor_map,
        prepared.rotations,
        launch,
        gate_experts=prepared.gate_counts,
        up_experts=prepared.up_counts,
    )
    return TPMoEFP4Binding(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        implementation="trellis_mixed3",
        state_E=core_plan.state_E,
        weight_E=core_plan.weight_E,
        max_rows=core_plan.max_rows,
        k=core_plan.k,
        n=core_plan.n,
        num_topk=core_plan.num_topk,
        device=core_plan.device,
        dtype=core_plan.dtype,
        execution_plan=scratch_plan.launch_plan,
        output=output_tensor,
        output_cast_target=output_cast_target,
        input_scales_static=bool(input_scales_static),
        fast_math=True,
        quant_mode="w4a16",
        decode_config=scratch_plan.caps.decode_config,
        routed_rows_capacity=core_plan.routed_rows,
        layer_idx=layer_idx,
        route_block_size_m=core_plan.route_block_size_m,
        mixed_trellis_binding=mixed_binding,
        mixed_trellis_buffers=buffers,
    )


def _resolve_trellis_route_block_size(caps: TPMoEScratchCaps) -> int | None:
    """Resolve the one route geometry shared by trellis sizing and binding."""
    if caps.w4a16_block_size_m is not None:
        return int(caps.w4a16_block_size_m)
    if caps.source_format not in _TRELLIS_SOURCE_FORMATS:
        return None
    if _normalize_quant_mode(caps.quant_mode) != "w4a16":
        # Trellis sources under w4a8_mx run the micro/dynamic regimes,
        # which do not use the w4a16 route-block geometry.
        return None
    from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m

    token_counts = _arena_core_token_counts(
        max_tokens=caps.max_tokens,
        num_topk=caps.num_topk,
        core_token_counts=caps.core_token_counts,
        quant_mode=caps.quant_mode,
        bucket_w4a16_tokens=False,
    )
    route_experts = caps.route_num_experts or caps.weight_E
    preferred = select_route_block_size_m(
        max(token_counts),
        caps.num_topk,
        route_experts,
    )
    tile = caps.weight_plan.trellis_tile_config
    if tile is None or torch.device(caps.device).type != "cuda":
        return preferred
    # Native Trellis packing fixes both projection tiles. The route block
    # must fit those tiles' shared memory, not a different automatically
    # selected GEMM tile with the same logical weight dimensions.
    from b12x.moe._shared.kernels.w4a16.host import (
        route_block_sizes_for_capacity,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import _candidate_tile_fits

    props = torch.cuda.get_device_properties(caps.device)
    budget = int(props.shared_memory_per_block_optin) - 512
    candidates = route_block_sizes_for_capacity(
        max_tokens=max(token_counts),
        topk=caps.num_topk,
        num_experts=route_experts,
    )
    dimensions = (
        (_activation_w1_rows(caps.weight_plan.activation, caps.n), caps.k),
        (caps.k, caps.n),
    )
    for block_rows in reversed(candidates):
        if all(
            _candidate_tile_fits(
                problem_n=n,
                problem_k=k,
                cta_m_blocks=(block_rows + 15) // 16,
                tile_k=tk,
                tile_n=tn,
                cta_threads=tk * tn // 64,
                max_shared_mem=budget,
                scale_format="e4m3_k32",
                weight_layout="trellis_t256",
                weight_bits=max(4, caps.weight_plan.trellis_bits or 3),
            )
            for (n, k), (tk, tn) in zip(dimensions, (tile[:2], tile[2:]))
        ):
            return block_rows
    raise ValueError("Native Trellis projection tiles exceed shared-memory capacity")


def _plan_tp_moe_arena_layout_from_caps(
    caps: TPMoEScratchCaps,
    *,
    deterministic_output: bool | None = None,
) -> TPMoEArenaLayout:
    if not isinstance(caps, TPMoEScratchCaps):
        raise TypeError("caps must be a TPMoEScratchCaps")
    if deterministic_output is None:
        deterministic_output = caps.deterministic_output
    return plan_tp_moe_arena_layout(
        max_tokens=caps.max_tokens,
        num_topk=caps.num_topk,
        device=caps.device,
        weight_plan=caps.weight_plan,
        core_token_counts=caps.core_token_counts,
        route_num_experts=caps.route_num_experts,
        route_logits_dtype=caps.route_logits_dtype,
        quant_mode=caps.quant_mode,
        apply_router_weight_on_input=caps.apply_router_weight_on_input,
        swiglu_limit=caps.swiglu_limit,
        swiglu_alpha=caps.swiglu_alpha,
        swiglu_beta=caps.swiglu_beta,
        collect_activation_amax=caps.collect_activation_amax,
        deterministic_output=deterministic_output,
        w4a16_block_size_m=_resolve_trellis_route_block_size(caps),
        w4a16_prefill_fused_sum=caps.w4a16_prefill_fused_sum,
        decode_config=caps.decode_config,
    )


def tp_moe_required_nbytes(caps: TPMoEScratchCaps) -> int:
    """Return the planned arena bytes without compiling launches or retaining storage."""
    return _plan_tp_moe_arena_layout_from_caps(caps).total_nbytes


def plan_tp_moe_scratch(
    caps: TPMoEScratchCaps,
    *,
    prewarm_launches: bool = True,
) -> TPMoEScratchPlan:
    deterministic_output = caps.deterministic_output
    layout = _plan_tp_moe_arena_layout_from_caps(
        caps,
        deterministic_output=deterministic_output,
    )
    capacity_tokens = max(layout.core_token_counts or (int(caps.max_tokens),))
    launch_plan = plan_tp_moe_execution(
        num_tokens=capacity_tokens,
        num_topk=caps.num_topk,
        device=caps.device,
        weight_plan=caps.weight_plan,
        quant_mode=caps.quant_mode,
        swiglu_limit=caps.swiglu_limit,
        swiglu_alpha=caps.swiglu_alpha,
        swiglu_beta=caps.swiglu_beta,
        apply_router_weight_on_input=caps.apply_router_weight_on_input,
        deterministic_output=deterministic_output,
        decode_config=caps.decode_config,
    )
    resolved_block_size_m = _resolve_trellis_route_block_size(caps)
    core_workspace_plan = _plan_core_workspace(
        launch_plan.implementation,
        launch_plan.quant_mode,
        launch_plan.state_E,
        launch_plan.weight_E,
        launch_plan.k,
        launch_plan.n,
        launch_plan.num_topk,
        launch_plan.device,
        launch_plan.dtype,
        routed_rows=launch_plan.routed_rows,
        max_rows=launch_plan.max_rows,
        activation=launch_plan.activation,
        dynamic_physical_tiles=launch_plan.dynamic_physical_tiles,
        dynamic_task_capacity=launch_plan.dynamic_task_capacity,
        dynamic_tile_m=launch_plan.execution.tile_m,
        dynamic_tile_n=launch_plan.execution.tile_n,
        dynamic_route_mode=launch_plan.decode_config.dynamic_route_mode,
        nvfp4_materialize_intermediate=launch_plan.decode_config.nvfp4_materialize_intermediate,
        source_format=caps.source_format,
        w13_layout=caps.w13_layout,
        w4a16_weight_layout=caps.w4a16_weight_layout,
        w4a16_scale_format=caps.w4a16_scale_format,
        route_num_experts=caps.route_num_experts,
        w4a16_block_size_m=resolved_block_size_m,
        w4a16_prefill_fused_sum=caps.w4a16_prefill_fused_sum,
        trellis_bits=caps.weight_plan.trellis_bits or 3,
        trellis_tile_config=caps.weight_plan.trellis_tile_config,
        trellis_pair_kinds=caps.weight_plan.trellis_pair_kinds,
        trellis_codebook=_derive_trellis_codebook(
            caps.weight_plan.source_format, caps.weight_plan.trellis_codebook
        ),
        intermediate_hadamard=caps.weight_plan.intermediate_hadamard,
        projection_mixed_trellis=_uses_projection_mixed_trellis(caps.weight_plan),
        apply_router_weight_on_input=caps.apply_router_weight_on_input,
        collect_activation_amax=caps.collect_activation_amax,
        deterministic_output=launch_plan.deterministic_output,
        swiglu_limit=launch_plan.swiglu_limit,
        swiglu_alpha=launch_plan.swiglu_alpha,
        swiglu_beta=launch_plan.swiglu_beta,
    )
    if prewarm_launches:
        fused_launches, topk_sum_launches = _plan_full_rotation_w4a16_launches(
            caps=caps,
            core_plan=core_workspace_plan,
            capacity_tokens=capacity_tokens,
        )
        mixed_trellis_launches = _plan_projection_mixed_trellis_launches(
            caps=caps,
            core_plan=core_workspace_plan,
            capacity_tokens=capacity_tokens,
        )
    else:
        fused_launches = ()
        topk_sum_launches = ()
        mixed_trellis_launches = ()
    route_pack_launches = None
    if (
        prewarm_launches
        and core_workspace_plan.full_rotation
        and not core_workspace_plan.projection_mixed_trellis
        and core_workspace_plan.device.type == "cuda"
    ):
        from b12x.moe._shared.kernels.w4a16.route_pack import (
            compile_w4a16_route_pack_launches,
        )

        route_pack_launches = compile_w4a16_route_pack_launches(
            tokens=capacity_tokens,
            topk=core_workspace_plan.num_topk,
            block_size=resolved_block_size_m,
            num_experts=core_workspace_plan.route_E,
            ordinal=(
                core_workspace_plan.device.index
                if core_workspace_plan.device.index is not None
                else torch.cuda.current_device()
            ),
            bucket_tokens=False,
        )
    return TPMoEScratchPlan(
        caps=caps,
        layout=layout,
        launch_plan=launch_plan,
        _core_workspace_plan=core_workspace_plan,
        _scratch_specs=(
            scratch_buffer_spec(
                "tp_moe.scratch",
                nbytes=layout.total_nbytes,
                device=torch.device(caps.device),
            ),
        ),
        # Retain every route/GEMM/reduction program required after cache eviction.
        _prewarmed_fused_launches=fused_launches,
        _prewarmed_topk_sum_launches=topk_sum_launches,
        _mixed_trellis_launches=mixed_trellis_launches,
        _prewarmed_route_pack_launches=route_pack_launches,
    )


def _w4a16_element_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise TypeError(f"unsupported W4A16 activation dtype {dtype}")


def _prewarm_w4a16_planned_launches(
    workspace: TPW4A16Workspace,
    *,
    token_counts: tuple[int, ...],
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
    w13_layout: str = "w13",
    collect_activation_amax: bool = False,
    prefill_fused_sum: bool = False,
) -> None:
    """Resolve every W4A16 kernel shape owned by a frozen arena.

    ``weight_layout`` and ``w13_layout`` must match the prepared weights the
    binding will run with: E8M0 sources keep native ``modelopt`` weights (so the
    FC1 ``source_n_rotation`` depends on ``w13_layout``), while NVFP4 sources are
    ``packed`` (rotation already folded in, ``w13_layout`` irrelevant).
    """
    if workspace.device.type != "cuda":
        raise RuntimeError("W4A16 MoE launch planning requires a CUDA device")
    scale_format = _normalize_w4a16_scale_format(scale_format)
    weight_layout = _normalize_w4a16_weight_layout(weight_layout)
    full_rotation = bool(workspace.full_rotation)
    if full_rotation:
        w13_layout = "trellis_t256_proj"
    else:
        w13_layout = _normalize_w13_layout(w13_layout)
    collect_activation_amax = bool(collect_activation_amax)

    from b12x.moe._shared.kernels.w4a16.host import (
        prefill_fused_sum_eligible,
        route_pack_capacity,
        select_route_block_size_m,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _DEFAULT_MAX_SHARED_MEM,
        _MAX_DIRECT_TOPK_ROUTE_M,
        _rot_scales_dummy,
        _TC_DECODE_MAX_M,
        _W4A16_SMALL_M_DIRECT_MAX_M,
        compile_w4a16_fused_moe,
        compile_w4a16_topk_sum,
        pack_topk_routes_by_expert,
    )

    token_counts = tuple(
        sorted({max(int(token_count), 1) for token_count in token_counts})
    )
    if not token_counts:
        raise ValueError("W4A16 launch planning requires at least one token count")

    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    with torch.cuda.device(workspace.device):
        is_capturing = torch.cuda.is_current_stream_capturing()
        _rot_scales_dummy(workspace.device)
        props = torch.cuda.get_device_properties(workspace.device)
        sms = int(props.multi_processor_count)
        max_shared_mem = int(
            getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
        )
        input_element_dtype = _w4a16_element_dtype(workspace.dtype)
        element_dtype = "fp16" if full_rotation else input_element_dtype
        fused_launches: dict[object, object] = {}
        topk_sum_launches: dict[int, object] = {}
        mapped_direct_launches: dict[int, object] = {}
        direct_topk_launches: dict[int, object] = {}
        tc_decode_launches: dict[int, object] = {}
        # TC-decode is a packed-layout small-M decode path; always build its
        # fused-sum launch variant for the supported decode sizes so the binding
        # can dispatch to it instead of the general fused launch.
        build_tc_decode = bool(
            not collect_activation_amax
            and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
            and element_dtype == "bf16"
            and is_gated_moe_activation(workspace.activation)
        )
        for token_count in token_counts:
            t_token = time.perf_counter() if _B12X_TIMING else 0.0
            block_size_m = workspace.route_block_size_m or select_route_block_size_m(
                token_count, workspace.num_topk, workspace.route_E
            )
            routed_rows = int(token_count) * int(workspace.num_topk)
            _, _, max_m_blocks = route_pack_capacity(
                routed_rows,
                block_size_m,
                workspace.route_E,
                topk=workspace.num_topk,
                bucket_tokens=not full_rotation,
            )
            t_shape = time.perf_counter() if _B12X_TIMING else 0.0
            fused_key = (
                weight_layout,
                scale_format,
                token_count,
                collect_activation_amax,
            )
            build_prefill_fused_sum = prefill_fused_sum_eligible(
                dtype=element_dtype,
                m=token_count,
                full_rotation=full_rotation,
                weight_layout=weight_layout,
                collect_activation_amax=collect_activation_amax,
                enabled=prefill_fused_sum,
            )
            # Rotation-table row count belongs to the prepared artifact, which
            # is not bound until after the workspace is frozen. Resolve both
            # full-rotation specializations now so either artifact contract is
            # graph-safe and allocation-free at bind/run time.
            broadcast_suh_options = (False, True) if full_rotation else (False,)
            for broadcast_suh in broadcast_suh_options:
                resolved_fused = compile_w4a16_fused_moe(
                    size_m=token_count,
                    hidden_size=workspace.k,
                    intermediate_size=workspace.n,
                    num_experts=workspace.weight_E,
                    top_k=workspace.num_topk,
                    activation=workspace.activation,
                    apply_router_weight_on_input=bool(apply_router_weight_on_input),
                    zero_fc2_output=False,
                    moe_block_size=block_size_m,
                    max_m_blocks=max_m_blocks,
                    element_dtype=element_dtype,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    swiglu_limit=swiglu_limit,
                    swiglu_alpha=swiglu_alpha,
                    swiglu_beta=swiglu_beta,
                    weight_layout=weight_layout,
                    scale_format=scale_format,
                    w13_layout=w13_layout,
                    prefill_fused_sum_fp32=build_prefill_fused_sum,
                    collect_activation_amax=collect_activation_amax,
                    trellis_bits=workspace.trellis_bits,
                    trellis_codebook=workspace.trellis_codebook or LUT_E4M3,
                    fc1_trellis_pair_kind=_fc_trellis_pair_kind(workspace),
                    fc2_trellis_pair_kind=_fc_trellis_pair_kind(workspace),
                    force_tile_config=workspace.trellis_tile_config,
                    intermediate_rotation=full_rotation,
                    full_rotation=full_rotation,
                    intermediate_hadamard=workspace.intermediate_hadamard,
                    rotation_input_dtype=input_element_dtype,
                    broadcast_suh=broadcast_suh,
                )
                if not broadcast_suh:
                    fused_launches[fused_key] = resolved_fused
            t_fused = time.perf_counter() if _B12X_TIMING else 0.0
            if full_rotation:
                for ids_dtype in (torch.int32, torch.int64):
                    for mapped in (False, True):
                        for broadcast_svh in (False, True):
                            resolved_topk_sum = compile_w4a16_topk_sum(
                                m=token_count,
                                topk=workspace.num_topk,
                                hidden_size=workspace.k,
                                element_dtype=element_dtype,
                                full_rotation=True,
                                intermediate_hadamard=workspace.intermediate_hadamard,
                                num_experts=workspace.weight_E,
                                route_num_experts=(workspace.route_E if mapped else 0),
                                route_ids_dtype=ids_dtype,
                                use_expert_map=mapped,
                                broadcast_svh=broadcast_svh,
                            )
                            if not broadcast_svh:
                                topk_sum_launches[(token_count, ids_dtype, mapped)] = (
                                    resolved_topk_sum
                                )
            elif not build_prefill_fused_sum:
                topk_sum_launches[token_count] = compile_w4a16_topk_sum(
                    m=token_count,
                    topk=workspace.num_topk,
                    hidden_size=workspace.k,
                    element_dtype=element_dtype,
                )
            t_sum = time.perf_counter() if _B12X_TIMING else 0.0

            # The real route-pack launch happens in run_w4a16_moe. During CUDA
            # graph capture this prewarm-only launch would be recorded as
            # useless work in every captured MoE graph.
            if is_capturing:
                if _B12X_TIMING:
                    total_ms = (t_sum - t_token) * 1000.0
                    if total_ms >= _B12X_TIMING_THRESHOLD_MS:
                        logger.warning(
                            "b12x_w4a16_prewarm timing tokens=%d capturing=%s "
                            "shape=%.3fms compile_fused=%.3fms "
                            "compile_sum=%.3fms route_pack=skipped total=%.3fms",
                            int(token_count),
                            is_capturing,
                            (t_shape - t_token) * 1000.0,
                            (t_fused - t_shape) * 1000.0,
                            (t_sum - t_fused) * 1000.0,
                            total_ms,
                        )
                continue

            route_pack_key = route_pack_prewarm_key(
                workspace.device.type,
                int(torch.cuda.current_device()),
                torch.int32,
                token_count,
                workspace.num_topk,
                workspace.packed_route_indices.numel(),
                workspace.block_expert_ids.numel(),
                int(block_size_m),
                int(workspace.route_E),
                bool(False),
            )
            if route_pack_key in _W4A16_ROUTE_PACK_PREWARMED:
                if _B12X_TIMING:
                    total_ms = (t_sum - t_token) * 1000.0
                    if total_ms >= _B12X_TIMING_THRESHOLD_MS:
                        logger.warning(
                            "b12x_w4a16_prewarm timing tokens=%d capturing=%s "
                            "shape=%.3fms compile_fused=%.3fms "
                            "compile_sum=%.3fms route_pack=cached total=%.3fms",
                            int(token_count),
                            is_capturing,
                            (t_shape - t_token) * 1000.0,
                            (t_fused - t_shape) * 1000.0,
                            (t_sum - t_fused) * 1000.0,
                            total_ms,
                        )
                continue

            dummy_topk_ids = torch.empty(
                token_count,
                workspace.num_topk,
                dtype=torch.int32,
                device=workspace.device,
            )
            dummy_topk_ids.zero_()
            pack_topk_routes_by_expert(
                dummy_topk_ids,
                block_size_m,
                workspace.route_E,
                packed_route_indices=workspace.packed_route_indices,
                block_expert_ids=workspace.block_expert_ids,
                packed_route_count=workspace.packed_route_count,
                expert_offsets=workspace.expert_offsets,
            )
            _W4A16_ROUTE_PACK_PREWARMED.add(route_pack_key)
            if _B12X_TIMING:
                t_route = time.perf_counter()
                total_ms = (t_route - t_token) * 1000.0
                if total_ms >= _B12X_TIMING_THRESHOLD_MS:
                    logger.warning(
                        "b12x_w4a16_prewarm timing tokens=%d capturing=%s "
                        "shape=%.3fms compile_fused=%.3fms compile_sum=%.3fms "
                        "route_pack=%.3fms total=%.3fms",
                        int(token_count),
                        is_capturing,
                        (t_shape - t_token) * 1000.0,
                        (t_fused - t_shape) * 1000.0,
                        (t_sum - t_fused) * 1000.0,
                        (t_route - t_sum) * 1000.0,
                        total_ms,
                    )
        mapped_tc_decode = bool(
            not full_rotation
            and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
            and element_dtype == "bf16"
            and is_gated_moe_activation(workspace.activation)
        )
        if (full_rotation or mapped_tc_decode) and not collect_activation_amax:
            # Capture buckets are usually powers of two, while speculative
            # decode can produce any m in this range. Precompile every exact
            # mapped-direct geometry. Packed W4A16 folds the top-k reduction
            # into FC2; full-rotation Trellis retains its ordered FP32 reducer.
            for direct_m in range(1, _W4A16_SMALL_M_DIRECT_MAX_M + 1):
                direct_block_size_m = (
                    workspace.route_block_size_m
                    or select_route_block_size_m(
                        direct_m,
                        workspace.num_topk,
                        workspace.route_E,
                    )
                )
                for broadcast_suh in (False, True) if full_rotation else (False,):
                    resolved_mapped_direct = compile_w4a16_fused_moe(
                        size_m=direct_m,
                        hidden_size=workspace.k,
                        intermediate_size=workspace.n,
                        num_experts=workspace.weight_E,
                        top_k=workspace.num_topk,
                        activation=workspace.activation,
                        apply_router_weight_on_input=bool(apply_router_weight_on_input),
                        zero_fc2_output=False,
                        moe_block_size=direct_block_size_m,
                        max_m_blocks=direct_m * int(workspace.num_topk),
                        element_dtype=element_dtype,
                        sms=sms,
                        max_shared_mem=max_shared_mem,
                        swiglu_limit=swiglu_limit,
                        swiglu_alpha=swiglu_alpha,
                        swiglu_beta=swiglu_beta,
                        weight_layout=weight_layout,
                        scale_format=scale_format,
                        w13_layout=w13_layout,
                        collect_activation_amax=False,
                        trellis_bits=workspace.trellis_bits,
                        fc1_trellis_pair_kind=_fc_trellis_pair_kind(workspace),
                        fc2_trellis_pair_kind=_fc_trellis_pair_kind(workspace),
                        force_tile_config=workspace.trellis_tile_config,
                        direct_topk_routes=True,
                        use_expert_map=True,
                        tc_decode_fused_sum=mapped_tc_decode,
                        intermediate_rotation=full_rotation,
                        full_rotation=full_rotation,
                        intermediate_hadamard=workspace.intermediate_hadamard,
                        rotation_input_dtype=input_element_dtype,
                        broadcast_suh=broadcast_suh,
                    )
                    if not broadcast_suh:
                        mapped_direct_launches[direct_m] = resolved_mapped_direct
                if full_rotation:
                    for broadcast_svh in (False, True):
                        resolved_topk_sum = compile_w4a16_topk_sum(
                            m=direct_m,
                            topk=workspace.num_topk,
                            hidden_size=workspace.k,
                            element_dtype=element_dtype,
                            full_rotation=True,
                            intermediate_hadamard=workspace.intermediate_hadamard,
                            num_experts=workspace.weight_E,
                            route_num_experts=workspace.route_E,
                            route_ids_dtype=torch.int32,
                            use_expert_map=True,
                            broadcast_svh=broadcast_svh,
                        )
                        if not broadcast_svh:
                            topk_sum_launches[(direct_m, torch.int32, True)] = (
                                resolved_topk_sum
                            )

        if build_tc_decode:
            # The capture/route-pack token counts above are powers of two, but the
            # real decode shapes are seqs*(1+num_spec) (e.g. 3, 6 under MTP). The
            # binding looks up TC-decode launches by the *exact* runtime m, so build
            # one for every small-M size in the supported range, independent of the
            # capture buckets.
            for tc_m in range(1, _TC_DECODE_MAX_M + 1):
                tc_block_size_m = select_route_block_size_m(
                    tc_m, workspace.num_topk, workspace.weight_E
                )
                tc_decode_launches[tc_m] = compile_w4a16_fused_moe(
                    size_m=tc_m,
                    hidden_size=workspace.k,
                    intermediate_size=workspace.n,
                    num_experts=workspace.weight_E,
                    top_k=workspace.num_topk,
                    activation=workspace.activation,
                    apply_router_weight_on_input=bool(apply_router_weight_on_input),
                    zero_fc2_output=False,
                    moe_block_size=tc_block_size_m,
                    # Direct-topk routing uses one block per routed row.
                    max_m_blocks=tc_m * int(workspace.num_topk),
                    element_dtype=element_dtype,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    swiglu_limit=swiglu_limit,
                    swiglu_alpha=swiglu_alpha,
                    swiglu_beta=swiglu_beta,
                    weight_layout=weight_layout,
                    scale_format=scale_format,
                    w13_layout=w13_layout,
                    direct_topk_routes=True,
                    tc_decode_fused_sum=True,
                )

        build_direct_topk = bool(
            not collect_activation_amax
            and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
            and not is_gated_moe_activation(workspace.activation)
        )
        if build_direct_topk:
            for direct_m in range(1, _MAX_DIRECT_TOPK_ROUTE_M + 1):
                direct_block_size_m = select_route_block_size_m(
                    direct_m,
                    workspace.num_topk,
                    workspace.route_E,
                )
                direct_topk_launches[direct_m] = compile_w4a16_fused_moe(
                    size_m=direct_m,
                    hidden_size=workspace.k,
                    intermediate_size=workspace.n,
                    num_experts=workspace.weight_E,
                    top_k=workspace.num_topk,
                    activation=workspace.activation,
                    apply_router_weight_on_input=bool(apply_router_weight_on_input),
                    zero_fc2_output=False,
                    moe_block_size=direct_block_size_m,
                    max_m_blocks=direct_m * int(workspace.num_topk),
                    element_dtype=element_dtype,
                    sms=sms,
                    max_shared_mem=max_shared_mem,
                    swiglu_limit=swiglu_limit,
                    swiglu_alpha=swiglu_alpha,
                    swiglu_beta=swiglu_beta,
                    weight_layout=weight_layout,
                    scale_format=scale_format,
                    w13_layout=w13_layout,
                    direct_topk_routes=True,
                )
                topk_sum_launches[direct_m] = compile_w4a16_topk_sum(
                    m=direct_m,
                    topk=workspace.num_topk,
                    hidden_size=workspace.k,
                    element_dtype=element_dtype,
                )

        workspace.planned_fused_moe_launches = fused_launches
        workspace.planned_topk_sum_launches = topk_sum_launches
        workspace.planned_mapped_direct_launches = mapped_direct_launches
        workspace.planned_direct_topk_launches = direct_topk_launches
        workspace.planned_tc_decode_launches = tc_decode_launches
        workspace.planned_scale_format = scale_format
        workspace.planned_collect_activation_amax = collect_activation_amax
    if _B12X_TIMING:
        t_done = time.perf_counter()
        total_ms = (t_done - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_w4a16_prewarm_total timing counts=%s capturing=%s total=%.3fms",
                token_counts,
                is_capturing,
                total_ms,
            )


def materialize_tp_moe_arena_workspaces(
    pool: TPMoEWorkspacePool,
    *,
    caps: TPMoEScratchCaps,
) -> None:
    """Materialize graph-sensitive pool state from one authoritative plan."""
    if not isinstance(caps, TPMoEScratchCaps):
        raise TypeError("caps must be a TPMoEScratchCaps")
    t0 = time.perf_counter() if _B12X_TIMING else 0.0
    weight_plan = caps.weight_plan
    max_tokens = caps.max_tokens
    num_topk = caps.num_topk
    device = caps.device
    quant_mode = caps.quant_mode
    source_format = caps.source_format
    w13_layout = caps.w13_layout
    apply_router_weight_on_input = caps.apply_router_weight_on_input
    swiglu_limit = caps.swiglu_limit
    swiglu_alpha = caps.swiglu_alpha
    swiglu_beta = caps.swiglu_beta
    collect_activation_amax = caps.collect_activation_amax
    deterministic_output = caps.deterministic_output
    if deterministic_output is None:
        deterministic_output = _dynamic_deterministic_output_enabled(
            quant_mode=quant_mode,
            device=torch.device(device),
        )
    w4a16_scale_format = caps.w4a16_scale_format or _w4a16_scale_format_for_source(
        source_format
    )
    w4a16_weight_layout = caps.w4a16_weight_layout or _w4a16_weight_layout_for_source(
        source_format,
        intermediate_size=caps.n,
    )
    core_token_counts = _arena_core_token_counts(
        max_tokens=max_tokens,
        num_topk=num_topk,
        core_token_counts=caps.core_token_counts,
        quant_mode=quant_mode,
        bucket_w4a16_tokens=source_format not in _TRELLIS_SOURCE_FORMATS,
    )
    resolved_block_size_m = _resolve_trellis_route_block_size(caps)
    t_counts = time.perf_counter() if _B12X_TIMING else 0.0
    selected: dict[tuple, tuple[TPMoEPlan, _TPCoreWorkspacePlan, int]] = {}
    for token_count in core_token_counts:
        plan = plan_tp_moe_execution(
            num_tokens=token_count,
            num_topk=num_topk,
            device=device,
            weight_plan=weight_plan,
            quant_mode=quant_mode,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            apply_router_weight_on_input=apply_router_weight_on_input,
            deterministic_output=deterministic_output,
            decode_config=caps.decode_config,
        )
        core_plan = _plan_core_workspace(
            plan.implementation,
            plan.quant_mode,
            plan.state_E,
            plan.weight_E,
            plan.k,
            plan.n,
            plan.num_topk,
            plan.device,
            plan.dtype,
            routed_rows=plan.routed_rows,
            max_rows=plan.max_rows,
            activation=plan.activation,
            dynamic_physical_tiles=plan.dynamic_physical_tiles,
            dynamic_task_capacity=plan.dynamic_task_capacity,
            dynamic_tile_m=plan.execution.tile_m,
            dynamic_tile_n=plan.execution.tile_n,
            dynamic_route_mode=plan.decode_config.dynamic_route_mode,
            nvfp4_materialize_intermediate=plan.decode_config.nvfp4_materialize_intermediate,
            source_format=source_format,
            w13_layout=w13_layout,
            w4a16_weight_layout=w4a16_weight_layout,
            w4a16_scale_format=w4a16_scale_format,
            route_num_experts=caps.route_num_experts,
            w4a16_block_size_m=resolved_block_size_m,
            w4a16_prefill_fused_sum=caps.w4a16_prefill_fused_sum,
            trellis_bits=weight_plan.trellis_bits or 3,
            trellis_tile_config=weight_plan.trellis_tile_config,
            trellis_pair_kinds=weight_plan.trellis_pair_kinds,
            trellis_codebook=_derive_trellis_codebook(
                weight_plan.source_format, weight_plan.trellis_codebook
            ),
            intermediate_hadamard=weight_plan.intermediate_hadamard,
            projection_mixed_trellis=_uses_projection_mixed_trellis(weight_plan),
            apply_router_weight_on_input=apply_router_weight_on_input,
            collect_activation_amax=collect_activation_amax,
            deterministic_output=plan.deterministic_output,
            swiglu_limit=plan.swiglu_limit,
            swiglu_alpha=plan.swiglu_alpha,
            swiglu_beta=plan.swiglu_beta,
        )
        required_nbytes = _core_workspace_nbytes(core_plan)
        key = _workspace_pool_key(
            plan.implementation,
            quant_mode=plan.quant_mode,
            activation=plan.activation,
            state_E=plan.state_E,
            weight_E=plan.weight_E,
            max_rows=plan.max_rows,
            k=plan.k,
            n=plan.n,
            num_topk=plan.num_topk,
            device=plan.device,
            dtype=plan.dtype,
        )
        existing_selection = selected.get(key)
        if existing_selection is None or required_nbytes > existing_selection[2]:
            selected[key] = (plan, core_plan, required_nbytes)

    t_selected = time.perf_counter() if _B12X_TIMING else 0.0
    materialize_ms = 0.0
    prewarm_ms = 0.0
    for key, (plan, core_plan, required_nbytes) in selected.items():
        existing = pool.workspaces.get(key)
        if existing is not None:
            with suppress(TypeError, ValueError):
                _validate_workspace(existing, plan=plan)
                if isinstance(existing, TPW4A16Workspace) and (
                    not set(core_token_counts).issubset(existing.planned_token_counts)
                    or existing.intermediate_hadamard != core_plan.intermediate_hadamard
                    or existing.planned_apply_router_weight_on_input
                    != bool(apply_router_weight_on_input)
                    or existing.planned_swiglu_limit != plan.swiglu_limit
                    or existing.planned_swiglu_alpha != plan.swiglu_alpha
                    or existing.planned_swiglu_beta != plan.swiglu_beta
                    or _normalize_w4a16_scale_format(
                        getattr(existing, "planned_scale_format", "e4m3_k16")
                    )
                    != w4a16_scale_format
                    or bool(getattr(existing, "planned_collect_activation_amax", False))
                    != collect_activation_amax
                    or bool(
                        getattr(
                            existing,
                            "planned_prefill_fused_sum_fp32",
                            False,
                        )
                    )
                    != core_plan.prefill_fused_sum_fp32
                ):
                    pass
                else:
                    continue

        if pool.shared_arena is None:
            arena = _allocate_core_arena(core_plan)
        else:
            if pool.shared_arena.device != plan.device:
                raise ValueError(
                    f"MoE pool arena device {pool.shared_arena.device} does not match plan device {plan.device}"
                )
            arena = _materialize_core_arena(
                core_plan,
                pool.shared_arena,
                offset_bytes=pool.core_arena_offset_bytes,
                capacity_nbytes=pool.core_arena_nbytes,
            )
            _emit_core_workspace_stats(
                core_plan,
                storage="shared",
                required_nbytes=required_nbytes,
                capacity_nbytes=pool.core_arena_nbytes,
            )
        pool.core_arenas[key] = arena

        if quant_mode == "w4a16":
            a1_init = None
            a2_init = None
        else:
            a1_init = torch.ones((), dtype=torch.float32, device=plan.device)
            a2_init = torch.ones((), dtype=torch.float32, device=plan.device)
        t_materialize0 = time.perf_counter() if _B12X_TIMING else 0.0
        materialized = _materialize_workspace_from_core_arena(
            core_plan,
            arena,
            a1_gscale=a1_init,
            a2_gscale=a2_init,
            input_scales_static=True,
            volatile_launch_state=bool(pool.shared_arena is not None),
        )
        if _B12X_TIMING:
            materialize_ms += (time.perf_counter() - t_materialize0) * 1000.0
        if quant_mode == "w4a16":
            if not isinstance(materialized, TPW4A16Workspace):
                raise TypeError(
                    "expected W4A16 arena materialization to create TPW4A16Workspace"
                )
            materialized.planned_token_counts = frozenset(core_token_counts)
            materialized.planned_apply_router_weight_on_input = bool(
                apply_router_weight_on_input
            )
            materialized.planned_swiglu_limit = plan.swiglu_limit
            materialized.planned_swiglu_alpha = plan.swiglu_alpha
            materialized.planned_swiglu_beta = plan.swiglu_beta
            materialized.planned_collect_activation_amax = collect_activation_amax
            materialized.planned_prefill_fused_sum_fp32 = (
                core_plan.prefill_fused_sum_fp32
            )
            t_prewarm0 = time.perf_counter() if _B12X_TIMING else 0.0
            _prewarm_w4a16_planned_launches(
                materialized,
                token_counts=core_token_counts,
                apply_router_weight_on_input=bool(apply_router_weight_on_input),
                swiglu_limit=materialized.planned_swiglu_limit,
                swiglu_alpha=materialized.planned_swiglu_alpha,
                swiglu_beta=materialized.planned_swiglu_beta,
                scale_format=w4a16_scale_format,
                weight_layout=w4a16_weight_layout,
                w13_layout=w13_layout,
                collect_activation_amax=collect_activation_amax,
                prefill_fused_sum=core_plan.prefill_fused_sum_fp32,
            )
            if _B12X_TIMING:
                prewarm_ms += (time.perf_counter() - t_prewarm0) * 1000.0
        pool.workspaces[key] = materialized
    if _B12X_TIMING:
        t_done = time.perf_counter()
        total_ms = (t_done - t0) * 1000.0
        if total_ms >= _B12X_TIMING_THRESHOLD_MS:
            logger.warning(
                "b12x_tp_moe_materialize timing max_tokens=%d counts=%s "
                "selected=%d quant=%s counts=%.3fms select=%.3fms "
                "materialize=%.3fms prewarm=%.3fms total=%.3fms",
                max_tokens,
                core_token_counts,
                len(selected),
                quant_mode,
                (t_counts - t0) * 1000.0,
                (t_selected - t_counts) * 1000.0,
                materialize_ms,
                prewarm_ms,
                total_ms,
            )


def allocate_tp_moe_workspace_pool(
    *,
    shared_arena: torch.Tensor | None = None,
    route_workspace_nbytes: int = 0,
    core_workspace_nbytes: int = 0,
    frozen: bool = False,
) -> TPMoEWorkspacePool:
    """Allocate an explicit caller-owned workspace pool for one execution lane."""
    pool = TPMoEWorkspacePool()
    if shared_arena is not None:
        pool.bind_shared_arena(
            shared_arena,
            route_workspace_nbytes=route_workspace_nbytes,
            core_workspace_nbytes=core_workspace_nbytes,
            frozen=frozen,
        )
    return pool


def build_tp_moe_fp4_binding(
    *,
    scratch: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
    a: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    output: torch.Tensor | None = None,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    quant_mode: str | None = None,
    decode_config: MoeDecodeConfig,
    unit_scale_contract: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
    route_expert_map: torch.Tensor | None = None,
    output_expert_map: torch.Tensor | None = None,
) -> TPMoEFP4Binding:
    workspace = scratch
    if not isinstance(experts, B12XFP4ExpertWeights):
        raise TypeError("experts must come from prepare_b12x_fp4_moe_weights")
    if not isinstance(
        workspace, (TPMoEWorkspace, TPW4A16Workspace, TPMoEWorkspacePool)
    ):
        raise TypeError("scratch must be a TP MoE scratch object")
    if a.ndim != 2:
        raise ValueError(
            f"expected input activations with rank 2, got {tuple(a.shape)}"
        )
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("topk_weights and topk_ids must be rank-2 tensors")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights and topk_ids shape mismatch: "
            f"{tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
        )
    if topk_ids.shape[0] != a.shape[0]:
        raise ValueError(
            f"routing batch mismatch: expected {a.shape[0]}, got {topk_ids.shape[0]}"
        )
    source_format = experts.source_format
    workspace_quant_mode = getattr(workspace, "quant_mode", None)
    quant_mode = _select_prepared_quant_mode(
        experts,
        quant_mode if quant_mode is not None else workspace_quant_mode,
    )
    _validate_fp4_source_format_for_quant_mode(
        source_format=source_format,
        quant_mode=quant_mode,
    )
    collect_activation_amax = activation_amax is not None
    unit_scale_contract = bool(unit_scale_contract and quant_mode == "w4a16")
    activation = experts.activation
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    w13_layout = experts.w13_layout
    m, k = map(int, a.shape)
    prepared_payload = _prepared_payload_for_runtime(
        experts,
        quant_mode=quant_mode,
        source_format=source_format,
        activation=activation,
        w13_layout=w13_layout,
        dtype=a.dtype,
        hidden_size=k,
    )
    num_topk = int(topk_ids.shape[1])
    requested_deterministic_output = _dynamic_deterministic_output_enabled(
        quant_mode=quant_mode,
        device=a.device,
    )
    plan = plan_tp_moe_execution(
        num_tokens=m,
        num_topk=num_topk,
        device=a.device,
        weight_plan=experts.plan,
        quant_mode=quant_mode,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        apply_router_weight_on_input=apply_router_weight_on_input,
        deterministic_output=requested_deterministic_output,
        decode_config=decode_config,
    )
    deterministic_output = plan.deterministic_output
    if isinstance(workspace, TPMoEWorkspacePool):
        weight_layout = "packed"
        scale_format = "e4m3_k16"
        if quant_mode == "w4a16":
            if prepared_payload is not None:
                weight_layout = getattr(prepared_payload, "weight_layout", "packed")
                scale_format = _normalize_w4a16_scale_format(
                    getattr(
                        prepared_payload,
                        "scale_format",
                        _w4a16_scale_format_for_source(source_format),
                    )
                )
            else:
                weight_layout = _w4a16_weight_layout_for_source(
                    source_format,
                    intermediate_size=plan.n,
                )
                scale_format = _w4a16_scale_format_for_source(source_format)
        workspace = _resolve_workspace(
            workspace,
            plan=plan,
            a1_gscale=experts.a1_gscale,
            a2_gscale=experts.a2_gscale,
            input_scales_static=input_scales_static,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            weight_layout=weight_layout,
            scale_format=scale_format,
            collect_activation_amax=collect_activation_amax,
        )
    elif isinstance(workspace, TPDynamicWorkspace):
        deterministic_output = requested_deterministic_output
        if deterministic_output and workspace.route_output.numel() < m * num_topk * k:
            raise ValueError(
                "dynamic workspace was planned without enough deterministic "
                "route-output capacity: "
                f"expected at least {m * num_topk * k} elements, got "
                f"{workspace.route_output.numel()}"
            )
    common_kwargs = dict(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        output=output,
        input_scales_static=bool(input_scales_static),
        fast_math=fast_math,
        quant_mode=quant_mode,
        execution_plan=plan,
        deterministic_output=deterministic_output,
        unit_scale_contract=unit_scale_contract,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        route_expert_map=route_expert_map,
        output_expert_map=output_expert_map,
    )
    if isinstance(workspace, TPW4A16Workspace):
        if quant_mode != "w4a16":
            raise ValueError(
                f"TPW4A16Workspace cannot bind non-W4A16 quant_mode {quant_mode!r}"
            )
        weight_layout = "packed"
        scale_format = "e4m3_k16"
        if prepared_payload is not None:
            weight_layout = getattr(
                prepared_payload,
                "weight_layout",
                "packed",
            )
            scale_format = _normalize_w4a16_scale_format(
                getattr(
                    prepared_payload,
                    "scale_format",
                    _w4a16_scale_format_for_source(source_format),
                )
            )
        elif quant_mode == "w4a16":
            weight_layout = _w4a16_weight_layout_for_source(
                source_format,
                intermediate_size=plan.n,
            )
            scale_format = _w4a16_scale_format_for_source(source_format)
        if workspace.full_rotation:
            if workspace.full_rotation_output is None:
                raise RuntimeError("Trellis workspace is missing its FP32 output")
            if output is None:
                common_kwargs["output"] = workspace.full_rotation_output[:m]
            else:
                if (
                    output.dtype != torch.float32
                    or output.device != a.device
                    or output.ndim != 2
                    or tuple(output.shape)
                    not in {
                        (m, k),
                        tuple(workspace.full_rotation_output.shape),
                    }
                    or not output.is_contiguous()
                ):
                    raise ValueError(
                        "full-rotation Trellis output must be a contiguous FP32 "
                        "live or capacity buffer on the input device"
                    )
                common_kwargs["output"] = output[:m]
            if topk_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("full-rotation Trellis topk_ids must be int32 or int64")
            for name, expert_map in (
                ("route_expert_map", route_expert_map),
                ("output_expert_map", output_expert_map),
            ):
                if expert_map is None:
                    continue
                if (
                    expert_map.dtype != torch.int32
                    or expert_map.device != a.device
                    or tuple(expert_map.shape) != (workspace.route_E,)
                    or not expert_map.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous int32[{workspace.route_E}] "
                        f"on {a.device}"
                    )
        route_mode = plan.decode_config.w4a16_route_mode
        if route_mode is None:
            raise RuntimeError("W4A16 MoE configuration is missing its route mode")
        fused_launch, topk_sum_launch = _w4a16_preplanned_launches(
            workspace,
            token_count=m,
            weight_layout=weight_layout,
            scale_format=scale_format,
            collect_activation_amax=collect_activation_amax,
            route_ids_dtype=topk_ids.dtype,
            use_route_expert_map=route_expert_map is not None,
            use_output_expert_map=output_expert_map is not None,
            route_mode=route_mode,
        )
        return TPMoEFP4Binding(
            **common_kwargs,
            implementation=workspace.implementation,
            state_E=workspace.state_E,
            weight_E=workspace.weight_E,
            max_rows=workspace.max_rows,
            k=workspace.k,
            n=workspace.n,
            num_topk=workspace.num_topk,
            device=workspace.device,
            dtype=workspace.dtype,
            routed_rows_capacity=workspace.routed_rows_capacity,
            intermediate_cache13=workspace.intermediate_cache13,
            intermediate_cache2=workspace.intermediate_cache2,
            prefill_sum_accum=workspace.prefill_sum_accum,
            fc1_c_tmp=workspace.fc1_c_tmp,
            fc2_c_tmp=workspace.fc2_c_tmp,
            packed_route_indices=workspace.packed_route_indices,
            block_expert_ids=workspace.block_expert_ids,
            packed_route_count=workspace.packed_route_count,
            expert_offsets=workspace.expert_offsets,
            expert_counts=workspace.expert_counts,
            rotation_a_gate=workspace.rotation_a_gate,
            rotation_a_up=workspace.rotation_a_up,
            kernel_workspace=workspace.kernel_workspace,
            route_block_size_m=workspace.route_block_size_m,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
        )
    if isinstance(workspace, TPMicroWorkspace):
        return TPMoEFP4Binding(
            **common_kwargs,
            implementation=workspace.implementation,
            state_E=workspace.state_E,
            weight_E=workspace.weight_E,
            max_rows=workspace.max_rows,
            k=workspace.k,
            n=workspace.n,
            num_topk=workspace.num_topk,
            device=workspace.device,
            dtype=workspace.dtype,
            row_counts=workspace.row_counts,
            token_map=workspace.token_map,
            token_weights=workspace.token_weights,
            packed_input=workspace.packed_input,
            packed_input_scale=workspace.packed_input_scale,
            barrier_count=workspace.barrier_count,
            barrier_epoch=workspace.barrier_epoch,
            packed_a_view=workspace.packed_a_view,
            sfa_ptr=workspace.sfa_ptr,
            packed_a_flat=workspace.packed_a_flat,
            scale_flat=workspace.scale_flat,
            packed_a_storage_ptr=workspace.packed_a_storage_ptr,
            routed_rows_capacity=workspace.routed_rows_capacity,
            active_expert_count=workspace.active_expert_count,
            weight_expert_ids=workspace.weight_expert_ids,
            global_to_local_expert=workspace.global_to_local_expert,
            compact_topk_ids=workspace.compact_topk_ids,
            micro_intermediate=workspace.micro_intermediate,
        )
    if isinstance(workspace, TPDynamicWorkspace):
        return TPMoEFP4Binding(
            **common_kwargs,
            implementation=workspace.implementation,
            state_E=workspace.state_E,
            weight_E=workspace.weight_E,
            max_rows=workspace.max_rows,
            k=workspace.k,
            n=workspace.n,
            num_topk=workspace.num_topk,
            device=workspace.device,
            dtype=workspace.dtype,
            row_counts=workspace.row_counts,
            token_map=workspace.token_map,
            token_weights=workspace.token_weights,
            packed_input=workspace.packed_input,
            packed_input_scale=workspace.packed_input_scale,
            barrier_count=workspace.barrier_count,
            barrier_epoch=workspace.barrier_epoch,
            packed_a_view=workspace.packed_a_view,
            sfa_ptr=workspace.sfa_ptr,
            packed_a_flat=workspace.packed_a_flat,
            scale_flat=workspace.scale_flat,
            packed_a_storage_ptr=workspace.packed_a_storage_ptr,
            routed_rows_capacity=workspace.routed_rows_capacity,
            physical_tiles_capacity=workspace.physical_tiles_capacity,
            task_capacity=workspace.task_capacity,
            route_output=workspace.route_output,
            materialized_intermediate=workspace.materialized_intermediate,
            expert_write_rows=workspace.expert_write_rows,
            expert_tile_base=workspace.expert_tile_base,
            input_gs=workspace.input_gs,
            down_input_scale=workspace.down_input_scale,
            pair_head=workspace.pair_head,
            producers_done_count=workspace.producers_done_count,
            all_work_published=workspace.all_work_published,
            task_head=workspace.task_head,
            task_tail=workspace.task_tail,
            task_ready=workspace.task_ready,
            task_expert=workspace.task_expert,
            task_m_tile=workspace.task_m_tile,
            task_slice_begin=workspace.task_slice_begin,
            task_slice_count=workspace.task_slice_count,
            task_valid_rows=workspace.task_valid_rows,
            tile_write_count=workspace.tile_write_count,
        )
    if not isinstance(workspace, TPMoEWorkspace):
        raise TypeError("expected a materialized TP MoE workspace")
    return TPMoEFP4Binding(
        **common_kwargs,
        implementation=workspace.implementation,
        state_E=workspace.state_E,
        weight_E=workspace.weight_E,
        max_rows=workspace.max_rows,
        k=workspace.k,
        n=workspace.n,
        num_topk=workspace.num_topk,
        device=workspace.device,
        dtype=workspace.dtype,
        row_counts=workspace.row_counts,
        token_map=workspace.token_map,
        token_weights=workspace.token_weights,
        packed_input=workspace.packed_input,
        packed_input_scale=workspace.packed_input_scale,
        barrier_count=workspace.barrier_count,
        barrier_epoch=workspace.barrier_epoch,
        packed_a_view=workspace.packed_a_view,
        sfa_ptr=workspace.sfa_ptr,
        packed_a_flat=workspace.packed_a_flat,
        scale_flat=workspace.scale_flat,
        packed_a_storage_ptr=workspace.packed_a_storage_ptr,
    )


def build_tp_moe_route_binding(
    *,
    scratch: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool | None = None,
    hidden_states: torch.Tensor,
    top_k: int,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize: bool = True,
    score_func: str = "softmax",
    correction_bias: torch.Tensor | None = None,
    image_correction_bias: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> TPMoERouteBinding:
    """Bind prepared routing; score controls match :func:`route_topk`."""
    if scratch is not None and not isinstance(
        scratch,
        (TPMoEWorkspace, TPW4A16Workspace, TPMoEWorkspacePool),
    ):
        raise TypeError("scratch must be a TP MoE scratch object or None")
    if hidden_states.ndim != 2:
        raise ValueError(
            f"expected hidden_states with rank 2, got shape {tuple(hidden_states.shape)}"
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits is not None and gate_weight is not None:
        raise ValueError("pass either router_logits or gate_weight, not both")
    if router_logits is None and gate_weight is None:
        raise ValueError("expected router_logits or gate_weight")
    return TPMoERouteBinding(
        hidden_states=hidden_states,
        top_k=int(top_k),
        scratch=scratch,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        router_logits=router_logits,
        renormalize=bool(renormalize),
        score_func=score_func,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
        routed_scaling_factor=float(routed_scaling_factor),
    )


def build_tp_moe_sparse_fp4_binding(
    *,
    scratch: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool,
    hidden_states: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    routing: B12XTopKRouting | None = None,
    top_k: int | None = None,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize_topk: bool = True,
    score_func: str = "softmax",
    correction_bias: torch.Tensor | None = None,
    image_correction_bias: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    output: torch.Tensor | None = None,
    return_routing: bool = False,
    input_scales_static: bool = False,
    fast_math: bool | None = None,
    quant_mode: str | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> TPMoESparseFP4Binding:
    if not isinstance(scratch, (TPMoEWorkspace, TPW4A16Workspace, TPMoEWorkspacePool)):
        raise TypeError("scratch must be a TP MoE scratch object")
    if hidden_states.ndim != 2:
        raise ValueError(
            "expected hidden_states with rank 2, got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if not isinstance(experts, B12XFP4ExpertWeights):
        raise TypeError("experts must be a B12XFP4ExpertWeights")
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        experts.activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    _select_prepared_quant_mode(experts, quant_mode)
    if routing is not None:
        if (
            top_k is not None
            or gate_weight is not None
            or gate_bias is not None
            or router_logits is not None
            or score_func != "softmax"
            or correction_bias is not None
            or image_correction_bias is not None
            or image_mask is not None
        ):
            raise ValueError(
                "routing is mutually exclusive with gate, top-k, and score-selection arguments"
            )
    elif top_k is None:
        raise ValueError("top_k is required when routing is not provided")
    return TPMoESparseFP4Binding(
        hidden_states=hidden_states,
        experts=experts,
        scratch=scratch,
        routing=routing,
        top_k=None if top_k is None else int(top_k),
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        router_logits=router_logits,
        renormalize_topk=bool(renormalize_topk),
        score_func=score_func,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
        routed_scaling_factor=float(routed_scaling_factor),
        output=output,
        return_routing=bool(return_routing),
        input_scales_static=bool(input_scales_static),
        fast_math=fast_math,
        quant_mode=quant_mode,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
    )


def _get_kernel_cache(impl: str) -> Dict[Tuple, Tuple]:
    if impl == "micro":
        return _MICRO_KERNEL_CACHE
    if impl == "dynamic":
        return _DYNAMIC_KERNEL_CACHE
    raise ValueError(f"unsupported implementation {impl!r}")


def _get_impl_mac(impl: str, *, routed_rows: int | None = None) -> int:
    dev_idx = torch.cuda.current_device()
    key = (dev_idx, impl)
    mac = _MAC_CACHE.get(key)
    sm_count = get_num_sm(torch.device("cuda"))
    mac_limit = min(get_max_active_clusters(1), sm_count)
    override_name = f"B12X_{impl.upper()}_MAX_ACTIVE_CLUSTERS"
    if impl.startswith("dynamic"):
        mac_override = _first_env(
            override_name,
            "B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS",
            "B12X_LEVEL10_MAX_ACTIVE_CLUSTERS",
        )
    else:
        mac_override = _first_env(override_name)
    if mac is None:
        if mac_override is not None:
            mac = max(1, min(int(mac_override), mac_limit))
        else:
            mac = mac_limit
        _MAC_CACHE[key] = mac
    if mac_override is not None:
        return mac
    return mac


def _select_micro_mma_tiler_mn(
    max_rows: int,
    n: int,
    *,
    resident_clusters: int | None = None,
) -> tuple[int, int]:
    if os.environ.get("B12X_MOE_TILE_MN"):
        return tuple(int(x) for x in os.environ["B12X_MOE_TILE_MN"].split("x"))
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    if max_rows <= 32 and n <= 256:
        return (64, 128)
    if resident_clusters is not None and resident_clusters < sm_count:
        return coarse_tile
    coarse_tiles = ((max_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # Single-token decode often lands exactly on the "half the machine" boundary.
    # Keeping the coarse 128x128 tile there leaves the M dimension badly underfilled.
    if max_rows <= 64 or (max_rows <= 128 and coarse_tiles <= max(1, sm_count // 2)):
        return (64, 128)
    return (128, 128)


def _get_micro_kernel(
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    *,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    mac_override: int | None = None,
    activation: str = "silu",
    device: torch.device | None = None,
    quant_mode: str = "nvfp4",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    compile_time_phase: int = 0,
    weight_layout: str | None = None,
    trellis_bits: int = 0,
    trellis_intermediate_hadamard: bool = False,
):
    quant_mode = _normalize_quant_mode(quant_mode)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    mac = mac_override if mac_override is not None else _get_impl_mac("micro")
    is_w4a8 = _is_w4a8_quant_mode(quant_mode)
    scale_format = _micro_scale_format_for_quant_mode(quant_mode)
    e8m0_scale_layout = _micro_e8m0_scale_layout_for_quant_mode(quant_mode)
    dynamic_down_scale = _dynamic_down_scale_enabled() and not is_w4a8

    micro_kwargs = dict(
        sf_vec_size=16,
        mma_tiler_mn=(64, 128),
        output_tile_count_n=1,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts and not is_w4a8,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        dynamic_down_scale=dynamic_down_scale,
        a8_mx_mode=is_w4a8,
        scale_format=scale_format,
        e8m0_scale_layout=e8m0_scale_layout,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    micro_trellis = weight_layout == "trellis_t256"
    if micro_trellis:
        micro_kwargs["weight_layout"] = weight_layout
        micro_kwargs["trellis_bits"] = int(trellis_bits)
        micro_kwargs["trellis_intermediate_hadamard"] = bool(
            trellis_intermediate_hadamard
        )
    # The native NVFP4 split is currently a GLM SiLU decode specialization.
    # Keep existing activation wrappers untouched for the normal fused phase.
    if compile_time_phase:
        micro_kwargs["compile_time_phase"] = int(compile_time_phase)
    kernel = activation_spec.make_micro_kernel(**micro_kwargs)
    kernel.configure(m, k, n, num_topk, weight_E, max_active_ctas=mac, device=device)
    kernel_key = kernel.__cache_key__

    cache_key = (
        quant_mode,
        "micro_direct",
        kernel_key,
        topk_ids_dtype,
    )
    # Compile planning produces deferred programs that must never be memoized
    # or returned in place of a compiled kernel.
    reuse_compiled = (
        not planning() and os.environ.get("B12X_MICRO_REUSE_COMPILED", "1") != "0"
    )
    if reuse_compiled:
        cached = _MICRO_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, kernel.grid_x

    def dummy(dt):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compile_kwargs = {}
    compile_options = os.environ.get("B12X_DIRECT_CUTE_OPTIONS", "")
    if compile_options:
        compile_kwargs["options"] = compile_options
    compile_m = int(kernel.m_const) if int(kernel.m_const) != 0 else 8
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),  # x_ptr
        dummy(cutlass.Uint8),  # w1_ptr
        dummy(cutlass.Uint8),  # w1s_ptr
        dummy(cutlass.Float32),  # w1a_ptr
        dummy(cutlass.Float32),  # a1_ptr
        dummy(cutlass.Float32),  # a2_ptr
        dummy(cutlass.Uint32),  # inter_ptr
        dummy(cutlass.Uint8),  # w2_ptr
        dummy(cutlass.Uint8),  # w2s_ptr
        dummy(cutlass.Float32),  # w2a_ptr
        dummy(ids_dtype),  # tid_ptr
        dummy(cutlass.Float32),  # tw_ptr
        dummy(cutlass.BFloat16),  # out_ptr
        barrier_fake,  # barrier_count
        barrier_fake,  # barrier_epoch
        Int32(weight_E),  # route_expert_limit
        Int32(compile_m),  # m_val
        Int32(1),  # grid_x
        current_cuda_stream(),  # stream
        *(
            (
                dummy(cutlass.Uint8),  # trellis_lut_ptr
                dummy(cutlass.Float16),  # trellis_rot_ptr
            )
            if micro_trellis
            else ()
        ),
        compile_spec=KernelCompileSpec.from_key(
            "integration.tp_moe.micro_direct",
            1,
            cache_key,
        ),
        **compile_kwargs,
    )
    with suppress(Exception):
        setattr(
            compiled,
            _DIRECT_MICRO_SHAPE_ATTR,
            cache_key,
        )

    if reuse_compiled:
        _MICRO_KERNEL_CACHE[cache_key] = compiled
    return compiled, kernel.grid_x


def _direct_micro_shape_accepts_block_dim(compiled, block_dim: int) -> bool:
    return True


def _compiled_direct_micro_accepts_block_dim(compiled, block_dim: int) -> bool:
    """Return whether the compiled direct micro kernel can launch `block_dim` threads."""
    cache_key = (
        id(compiled),
        int(block_dim),
        getattr(compiled, _DIRECT_MICRO_SHAPE_ATTR, None),
    )
    cached = _MICRO_DIRECT_LAUNCH_CAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not _direct_micro_shape_accepts_block_dim(compiled, block_dim):
        _MICRO_DIRECT_LAUNCH_CAP_CACHE[cache_key] = False
        return False

    accepted = False
    try:
        from cuda.bindings import driver, runtime

        executor = compiled.to(None)
        kernel_info = getattr(compiled, "kernel_info", None) or {}
        kernel_name = next(iter(kernel_info.keys()), None)
        if kernel_name is None and hasattr(compiled, "_get_name"):
            kernel_name = compiled._get_name()
        if isinstance(kernel_name, str):
            kernel_name = kernel_name.encode()
        if kernel_name is None:
            raise RuntimeError("compiled micro kernel did not expose a kernel name")

        jit_module = getattr(executor, "jit_module", None)
        cuda_library = getattr(jit_module, "cuda_library", None)
        if isinstance(cuda_library, (list, tuple)):
            cuda_library = cuda_library[0] if cuda_library else None
        if cuda_library is None:
            cuda_library = getattr(executor, "kernel", None)
        if cuda_library is None:
            raise RuntimeError("compiled micro kernel did not expose a CUDA library")

        err, kernel = runtime.cudaLibraryGetKernel(cuda_library, kernel_name)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaLibraryGetKernel failed with {err}")
        cu_kernel = driver.CUkernel(int(kernel))
        err, max_threads = driver.cuKernelGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            cu_kernel,
            0,
        )
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelGetAttribute failed with {err}")
        accepted = int(max_threads) >= int(block_dim)
    except Exception:
        accepted = False

    _MICRO_DIRECT_LAUNCH_CAP_CACHE[cache_key] = accepted
    return accepted


class _DynamicMoELaunch:
    """Thin wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(self, kernel, k, n, w1_n, num_topk):
        self._kernel = kernel
        self._k = k
        self._n = n
        self._w1_n = w1_n
        self._half_k = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = align_up(k // _NVFP4_BLOCK_SIZE, 4)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        intermediate_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        sfb_w13_gate_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        scatter_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_phys_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Assemble the kernel's tensor views from flat pointers and invoke it."""
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((scatter_rows, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._half_k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        if cutlass.const_expr(self._kernel.nvfp4_split_materialized):
            intermediate_tiles = self._n // 128
            words_per_row = intermediate_tiles * 16
            intermediate_u32 = cute.make_tensor(
                intermediate_ptr,
                layout=cute.make_layout(
                    (
                        rows_padded * words_per_row
                        + intermediate_tiles * rows_padded * 2,
                    ),
                    stride=(1,),
                ),
            )
        else:
            intermediate_u32 = cute.make_tensor(
                intermediate_ptr, layout=cute.make_layout((1,), stride=(1,))
            )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
        )
        if cutlass.const_expr(self._kernel.separate_w13_halves):
            half_layout = cute.make_layout(
                (self._n, self._k, row_counts.shape[0]),
                stride=(self._k, 1, self._w1_n * self._k),
            )
            b_w13_up = cute.make_tensor(b_w13.iterator, half_layout)
            b_w13_bytes = cute.recast_ptr(
                b_w13.iterator.align(16),
                dtype=cutlass.Uint8,
            )
            # _half_k is the packed bytes per K row: k // 2 for FP4 codes,
            # k for the one-byte MXFP8 codes.
            gate_iterator = cute.recast_ptr(
                (b_w13_bytes + self._n * self._half_k).align(16),
                dtype=b_w13.element_type,
            )
            b_w13_gate = cute.make_tensor(
                gate_iterator,
                half_layout,
            )
        else:
            b_w13_up = b_w13
            b_w13_gate = b_w13
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            intermediate_u32,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            b_w13_up,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
            b_w13_gate=b_w13_gate,
            sfb_w13_gate_ptr=sfb_w13_gate_ptr,
        )


class _DynamicMoEByteContainerLaunch(_DynamicMoELaunch):
    """MX byte-container variant of _DynamicMoELaunch.

    Serves both ``w6a8_mx`` and ``w8a8_mx`` (same kernel-call ABI, no repack or
    residual operands).  Only the flat scratch extents differ from the FP4
    recipes: packed-A stages one E4M3 byte per element (k bytes/row instead of
    the FP4 k//2), and the activation-scale plane holds the swizzled UE8M0
    K/32 atoms (4 bytes per 128 K elements per row).  ``packed_a`` keeps the
    [rows_padded, k, 1] shape; its Float8E4M3FN element type comes from the
    caller's pointer.
    """

    def __init__(self, kernel, k, n, w1_n, num_topk):
        super().__init__(kernel, k, n, w1_n, num_topk)
        # One E4M3 byte per element (the FP4 recipes pack two per byte).
        self._half_k = k
        # Swizzled 128-row SF atoms: one UE8M0 byte per K/32 block, padded to
        # the 4-block atom column -> align_up(k // 32, 4) bytes per row
        # (the vec16 recipes use align_up(k // 16, 4)).
        self._cols_pad_k = align_up(k // 32, 4)


class _DynamicMoEW4A8Launch:
    """w4a8 variant of _DynamicMoELaunch: adds the plain UE8M0 weight-scale
    grids (and, for the NVFP4-source recipe, the E4M3 residual grids), and
    sizes the packed-A scratch for one E4M3 byte per element with plain
    per-32 activation scales."""

    def __init__(self, kernel, k, n, w1_n, num_topk):
        self._kernel = kernel
        self._k = k
        self._n = n
        self._w1_n = w1_n
        self._num_topk = num_topk

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        intermediate_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        sfb_w13_mx_ptr: cute.Pointer,
        sfb_down_mx_ptr: cute.Pointer,
        w13_residual_ptr: cute.Pointer,
        down_residual_ptr: cute.Pointer,
        w13_rp_ptr: cute.Pointer,
        w13_sfb_rp_ptr: cute.Pointer,
        down_rp_ptr: cute.Pointer,
        down_sfb_rp_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        scatter_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_phys_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        trellis_lut_ptr: cute.Pointer | None = None,
        trellis_rot_ptr: cute.Pointer | None = None,
    ):
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((scatter_rows, self._k), stride=(self._k, 1)),
        )
        # E4M3 storage is one byte per element; the fp4-typed view spans 2k
        # nibble positions so the byte footprint is k per row.
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, 2 * self._k, 1),
                stride=(2 * self._k, 1, rows_padded * 2 * self._k),
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * (self._k // 32),), stride=(1,)),
        )
        materialized_n = align_up(self._n, 128)
        intermediate_u32 = cute.make_tensor(
            intermediate_ptr,
            layout=cute.make_layout(
                (rows_padded * (materialized_n + materialized_n // 32) // 4,),
                stride=(1,),
            ),
        )
        w13_grid_rows = self._w1_n
        down_mx_columns = self._n // 32
        down_residual_columns = self._n // 16
        if cutlass.const_expr(self._kernel.w4a8_residual):
            padded_n = align_up(self._n, _LEVEL_TILE_N)
            w13_grid_rows = 2 * padded_n if self._kernel.is_gated else padded_n
            down_mx_columns = padded_n // 32
            down_residual_columns = padded_n // 16
        sfb_w13_mx = cute.make_tensor(
            sfb_w13_mx_ptr,
            layout=cute.make_layout(
                (w13_grid_rows, self._k // 32, row_counts.shape[0]),
                stride=(
                    self._k // 32,
                    1,
                    w13_grid_rows * (self._k // 32),
                ),
            ),
        )
        sfb_down_mx = cute.make_tensor(
            sfb_down_mx_ptr,
            layout=cute.make_layout(
                (self._k, down_mx_columns, row_counts.shape[0]),
                stride=(down_mx_columns, 1, self._k * down_mx_columns),
            ),
        )
        w13_residual = cute.make_tensor(
            w13_residual_ptr,
            layout=cute.make_layout(
                (w13_grid_rows, self._k // 16, row_counts.shape[0]),
                stride=(
                    self._k // 16,
                    1,
                    w13_grid_rows * (self._k // 16),
                ),
            ),
        )
        down_residual = cute.make_tensor(
            down_residual_ptr,
            layout=cute.make_layout(
                (self._k, down_residual_columns, row_counts.shape[0]),
                stride=(
                    down_residual_columns,
                    1,
                    self._k * down_residual_columns,
                ),
            ),
        )
        num_experts = row_counts.shape[0]
        if cutlass.const_expr(getattr(self._kernel, "w4a8_trellis", False)):
            # Projection-major trellis payloads: w13 [2][E][K16][N16]
            # window blocks of 8*bits u32 (the prepared trellis layout), down
            # [E][K16(I)][N16(K)]. SFB tensors are compile-time dead
            # (identity UE8M0 word).
            _tr_bits = int(self._kernel.trellis_bits)
            _tr_w13_u32 = 2 * (self._k // 16) * ((self._w1_n // 2) // 16) * 8 * _tr_bits
            _tr_down_u32 = (self._n // 16) * (self._k // 16) * 8 * _tr_bits
            w13_rp = cute.make_tensor(
                w13_rp_ptr,
                layout=cute.make_layout((num_experts * _tr_w13_u32,), stride=(1,)),
            )
            down_rp = cute.make_tensor(
                down_rp_ptr,
                layout=cute.make_layout((num_experts * _tr_down_u32,), stride=(1,)),
            )
            _tr_sentinel = cute.make_layout((1,), stride=(1,))
            w13_sfb_rp = cute.make_tensor(w13_sfb_rp_ptr, layout=_tr_sentinel)
            down_sfb_rp = cute.make_tensor(down_sfb_rp_ptr, layout=_tr_sentinel)
        elif cutlass.const_expr(self._kernel.w4a8_n64_repacked):
            w13_rp = cute.make_tensor(
                w13_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * self._w1_n * self._k // 8,),
                    stride=(1,),
                ),
            )
            w13_sfb_rp = cute.make_tensor(
                w13_sfb_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * self._w1_n * (self._k // 32) // 4,),
                    stride=(1,),
                ),
            )
            down_rp = cute.make_tensor(
                down_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * self._k * self._n // 8,),
                    stride=(1,),
                ),
            )
            down_sfb_rp = cute.make_tensor(
                down_sfb_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * self._k * (self._n // 32) // 4,),
                    stride=(1,),
                ),
            )
        elif cutlass.const_expr(self._kernel.w4a8_repacked):
            w13_rp = cute.make_tensor(
                w13_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * (self._w1_n // 256) * (self._k // 128) * 4096,),
                    stride=(1,),
                ),
            )
            w13_sfb_rp = cute.make_tensor(
                w13_sfb_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * (self._w1_n // 256) * (self._k // 128) * 256,),
                    stride=(1,),
                ),
            )
            down_rp = cute.make_tensor(
                down_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * (self._k // 256) * (self._n // 128) * 4096,),
                    stride=(1,),
                ),
            )
            down_sfb_rp = cute.make_tensor(
                down_sfb_rp_ptr,
                layout=cute.make_layout(
                    (num_experts * (self._k // 256) * (self._n // 128) * 256,),
                    stride=(1,),
                ),
            )
        else:
            # CUTLASS DSL 4.6 rejects zero-sized layouts.  Repacked tensors are
            # compile-time dead in this specialization, so retain ABI-stable
            # sentinel views instead of constructing e.g. w1_n // 256 == 0.
            sentinel_layout = cute.make_layout((1,), stride=(1,))
            w13_rp = cute.make_tensor(w13_rp_ptr, layout=sentinel_layout)
            w13_sfb_rp = cute.make_tensor(w13_sfb_rp_ptr, layout=sentinel_layout)
            down_rp = cute.make_tensor(down_rp_ptr, layout=sentinel_layout)
            down_sfb_rp = cute.make_tensor(down_sfb_rp_ptr, layout=sentinel_layout)
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
        )
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            intermediate_u32,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
            sfb_w13_mx=sfb_w13_mx,
            sfb_down_mx=sfb_down_mx,
            w13_residual=w13_residual,
            down_residual=down_residual,
            w13_rp=w13_rp,
            w13_sfb_rp=w13_sfb_rp,
            down_rp=down_rp,
            down_sfb_rp=down_sfb_rp,
            trellis_lut=(
                cute.make_tensor(
                    trellis_lut_ptr,
                    layout=cute.make_layout((4096,), stride=(1,)),
                )
                if cutlass.const_expr(trellis_lut_ptr is not None)
                else None
            ),
            trellis_rotations=(
                cute.make_tensor(
                    trellis_rot_ptr,
                    layout=cute.make_layout(
                        (
                            row_counts.shape[0]
                            * (
                                6
                                if getattr(
                                    self._kernel, "trellis_intermediate_hadamard", False
                                )
                                else 3
                            )
                            * self._n,
                        ),
                        stride=(1,),
                    ),
                )
                if cutlass.const_expr(trellis_rot_ptr is not None)
                else None
            ),
        )


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    mac_override: int | None = None,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    w4a8_repacked: bool = False,
    w4a8_n64_repacked: bool = False,
    nvfp4_materialize_intermediate: bool = False,
    nvfp4_inline_scales: bool = False,
    direct_routing: bool = False,
    external_route_plan: bool = False,
    share_input_across_experts: bool = False,
    deterministic_output: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    trellis_bits: int = 0,
    trellis_intermediate_hadamard: bool = False,
    planned_tile_m: int | None = None,
    planned_num_tokens: int | None = None,
    w4a8_csf_inline: bool = False,
):
    """Resolve, compile, and cache the dynamic MoE kernel specialization for this
    launch geometry.
    """
    quant_mode = _normalize_quant_mode(quant_mode)
    # Both MX byte-container recipes ride the nvfp4-shaped launch ABI (no
    # repack/residual operands) and the MXFP8 activation scratch with UE8M0
    # K/32 scales (sf_vec_size 32).  w6a8_mx passes Float8E4M3FN views of the
    # 3:4-packed FP6 bytes (gmem K extent 3K/4, inline mxf8f6f4 MMA); w8a8_mx
    # passes views of the raw E4M3 bytes (gmem K extent == logical K, native
    # MmaMXF8Op).  The ctor rejects swap_ab, direct_routing,
    # materialize_intermediate and share_input_across_experts for these
    # recipes; all resolve to False below via the quant-mode predicates.
    is_w6a8 = quant_mode in _W6A8_QUANT_MODES
    is_w8a8 = quant_mode in _W8A8_QUANT_MODES
    is_byte_container = is_w6a8 or is_w8a8
    share_input_across_experts = bool(
        share_input_across_experts
        and (quant_mode == "nvfp4" or (quant_mode == "w4a8_mx" and w4a8_repacked))
    )
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation_spec.activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    # The MX recipes quantize per MX K/32 block (the ctor requires sf_vec_size
    # 32); the FP4 recipes keep the NVFP4 vec16 scale granularity.
    sf_vec_size = 32 if is_byte_container else 16
    mac = mac_override if mac_override is not None else _get_impl_mac("dynamic")
    is_w4a8 = _is_w4a8_quant_mode(quant_mode)
    # w4a8 is self-ranging; the dynamic per-tile FC2 scale does not apply.
    dynamic_down_scale = _dynamic_down_scale_enabled() and not is_w4a8
    work_source = _dynamic_work_source()
    # Tile planner (same routed_rows as the scratch plan -> consistent choice).
    mma_tiler_mn = _select_dynamic_tile_mn(
        m * num_topk,
        n,
        quant_mode,
        num_experts=E,
        activation=activation_spec.activation,
        planned_tile_m=planned_tile_m,
    )
    # A capacity plan must retain its FC1/FC2 representation when the live
    # count shrinks to one; otherwise replay asks for an unprepared kernel.
    specialization_tokens = m if planned_num_tokens is None else planned_num_tokens
    materialize_intermediate = _w4a8_dynamic_materialized_enabled(
        quant_mode=quant_mode,
        activation=activation_spec.activation,
        num_tokens=specialization_tokens,
        routed_rows=specialization_tokens * num_topk,
        num_experts=E,
        k=k,
        n=n,
        w4a8_repacked=w4a8_repacked,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        planned_tile_m=planned_tile_m,
    ) or (quant_mode == "nvfp4" and nvfp4_materialize_intermediate)
    if w4a8_n64_repacked and not w4a8_repacked:
        raise ValueError("w4a8_n64_repacked requires repacked W4A8 weights")
    if w4a8_csf_inline and (not w4a8_n64_repacked or int(trellis_bits) > 0):
        raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
    if w4a8_n64_repacked and n % 64 != 0:
        raise ValueError("w4a8_n64_repacked requires N divisible by 64")
    if w4a8_n64_repacked:
        materialize_intermediate = True
        share_input_across_experts = True
    # Independent zero-copy up/gate descriptors: NVFP4 at the 16-mod-32
    # boundary, and MXFP8 whenever the gate half starts inside a 128-row
    # tile/scale atom (the halves' tail tiles are TMA zero-filled).
    separate_w13_halves = bool(
        activation_spec.is_gated
        and (
            (quant_mode == "nvfp4" and int(n) % 32 == 16)
            or (is_w8a8 and int(n) % _LEVEL_TILE_N != 0)
        )
    )
    # Gated FC1 swap_ab handles 32-aligned, non-128 shards. The 16-mod-32
    # boundary uses independent zero-copy up/gate descriptors instead.
    swap_ab = bool(
        quant_mode == "nvfp4"
        and activation_spec.is_gated
        and int(n) % 128 != 0
        and int(n) % 32 == 0
    )
    _swap_env = _DYNAMIC_SWAP_AB_OVERRIDE
    if _swap_env is not None:
        swap_ab = _swap_env != "0"
    if is_w4a8 or is_byte_container:
        # None of these recipes build the swapped FC1 (their ctors reject it).
        swap_ab = False
    if separate_w13_halves:
        swap_ab = False
    dynamic_down_n = align_up(n, 32) if quant_mode == "nvfp4" and n % 32 == 16 else n

    # The key is also the program's compile spec, so every element is spelled
    # canonically: values that compare equal must serialize identically.
    cache_key = (
        str(quant_mode),
        "dynamic",
        int(E),
        int(k),
        int(n),
        int(num_topk),
        tuple(int(value) for value in mma_tiler_mn),
        topk_ids_dtype,
        bool(fast_math),
        str(activation),
        None if swiglu_limit is None else float(swiglu_limit),
        None if swiglu_alpha is None else float(swiglu_alpha),
        None if swiglu_beta is None else float(swiglu_beta),
        bool(dynamic_down_scale),
        bool(share_input_across_experts),
        bool(swap_ab),
        "split_w13_v2" if separate_w13_halves else False,
        "down_stride32" if dynamic_down_n != n else False,
        bool(deterministic_output),
        str(work_source),
        bool(w4a8_repacked),
        bool(w4a8_n64_repacked),
        bool(direct_routing),
        bool(external_route_plan),
        bool(materialize_intermediate),
        int(trellis_bits),
        bool(trellis_intermediate_hadamard),
        *(("nvfp4_inline_scales_v2",) if nvfp4_inline_scales else ()),
        *(("w4a8_csf_inline",) if w4a8_csf_inline else ()),
    )
    reuse_compiled = _first_env(
        "B12X_DYNAMIC_REUSE_COMPILED",
        "B12X_LEVEL10_REUSE_COMPILED",
    )
    if reuse_compiled is None:
        reuse_compiled = "1"
    # Compile planning produces deferred programs that must never be memoized
    # or returned in place of a compiled kernel.
    reuse_compiled = reuse_compiled != "0" and not planning()
    if reuse_compiled:
        cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, mac

    if is_byte_container:
        # Both MX recipes carry one byte per operand element in Float8E4M3FN
        # views (FP6 codes for w6a8_mx, real E4M3 codes for w8a8_mx) and read
        # every block scale as UE8M0.
        weight_dtype = cutlass.Float8E4M3FN
        a_scratch_dtype = cutlass.Float8E4M3FN
        sf_dtype = cutlass.Float8E8M0FNU
    else:
        weight_dtype = cutlass.Float4E2M1FN
        a_scratch_dtype = weight_dtype
        sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel_kwargs = dict(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        fast_math=fast_math,
        dynamic_down_scale=dynamic_down_scale,
    )
    kernel_kwargs["share_input_across_experts"] = share_input_across_experts
    kernel_kwargs["deterministic_output"] = bool(deterministic_output)
    kernel_kwargs["num_topk"] = int(num_topk)
    kernel_kwargs["swap_ab"] = swap_ab
    kernel_kwargs["separate_w13_halves"] = separate_w13_halves
    kernel_kwargs["work_source"] = work_source
    kernel_kwargs["materialize_intermediate"] = materialize_intermediate
    kernel_kwargs["swiglu_limit"] = swiglu_limit
    kernel_kwargs["swiglu_alpha"] = swiglu_alpha
    kernel_kwargs["swiglu_beta"] = swiglu_beta
    kernel_kwargs["direct_routing"] = bool(direct_routing)
    if external_route_plan:
        kernel_kwargs["external_route_plan"] = True
    if is_w4a8 and int(trellis_bits) > 0:
        kernel_kwargs["quant_recipe"] = "w4a8_trellis"
        kernel_kwargs["w4a8_repacked"] = True
        kernel_kwargs["trellis_bits"] = int(trellis_bits)
        kernel_kwargs["trellis_intermediate_hadamard"] = bool(
            trellis_intermediate_hadamard
        )
    elif is_w4a8:
        kernel_kwargs["quant_recipe"] = quant_mode
        kernel_kwargs["w4a8_repacked"] = bool(w4a8_repacked)
        kernel_kwargs["w4a8_n64_repacked"] = w4a8_n64_repacked
        kernel_kwargs["w4a8_n64_tail"] = bool(w4a8_n64_repacked and n % 128)
    elif is_w6a8 or is_w8a8:
        # w6a8_mx keeps the ctor defaults ("e4m3" MXFP8 activations against
        # "e2m3" FP6 weights); w8a8_mx keeps mxfp6_fmt_a/b unset so the kernel
        # emits the native MmaMXF8Op instead of the inline FP6 MMA.
        kernel_kwargs["quant_recipe"] = quant_mode
    if nvfp4_inline_scales:
        kernel_kwargs["nvfp4_inline_scales"] = True
    if w4a8_csf_inline:
        kernel_kwargs["w4a8_csf_inline"] = True
    kernel = activation_spec.make_dynamic_kernel(**kernel_kwargs)
    if is_w4a8:
        launch = _DynamicMoEW4A8Launch(
            kernel,
            k=k,
            n=n,
            w1_n=activation_spec.w1_rows(n),
            num_topk=num_topk,
        )
    elif is_byte_container:
        launch = _DynamicMoEByteContainerLaunch(
            kernel,
            k=k,
            n=n,
            w1_n=activation_spec.w1_rows(n),
            num_topk=num_topk,
        )
    else:
        launch = _DynamicMoELaunch(
            kernel,
            k=k,
            n=n,
            w1_n=activation_spec.w1_rows(n),
            num_topk=num_topk,
        )

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # a_input, topk_ids, topk_weights, scatter_output are pointers — shapes
    # are constructed at runtime from num_tokens Int32.
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    packed_a_fake = make_ptr(
        a_scratch_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    intermediate_fake = make_ptr(
        cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    task_ready_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_m_tile_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_begin_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    tile_write_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    w1_n = activation_spec.w1_rows(n)
    dynamic_w1_n = w1_n
    if quant_mode == "w4a8_nvfp4":
        padded_n = align_up(n, _LEVEL_TILE_N)
        dynamic_w1_n = 2 * padded_n if activation_spec.is_gated else padded_n
    # The MX-FP6 weight tensors are the 3:4-packed FP6 bytes, so their gmem K
    # extent is 3K/4 one-byte elements (the kernel expands to full-K
    # byte-containers in smem).  MXFP8 weights keep gmem K extent == logical K.
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (dynamic_w1_n, 3 * k // 4 if is_w6a8 else k, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, 3 * n // 4 if is_w6a8 else dynamic_down_n, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E,),
        assumed_align=4,
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E,),
        assumed_align=4,
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (E + 1,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (E,),
        assumed_align=16,
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    dsl_compile_options = None
    if is_w4a8:
        # ptxas -O3's scheduler register-starves the scalar-heavy w4a8
        # mainloop (40-reg compile, accumulators spilled around every MMA);
        # -O2 allocates honestly (observed 255 regs, 320B stack).
        from cutlass.base_dsl.compiler import OptLevel

        dsl_compile_options = OptLevel(2)
    compiled = b12x_compile(
        launch,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        intermediate_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        pair_head_fake,
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake,
        sfb_w13_fake,
        *((sfb_w13_fake,) if not is_w4a8 else ()),
        b_down_fake,
        sfb_down_fake,
        *(
            (
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
            )
            if is_w4a8
            else ()
        ),
        row_counts_fake,
        expert_write_rows_fake,
        expert_tile_base_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        *(
            (
                make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(
                    cutlass.Float16,
                    16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
            )
            if int(trellis_bits) > 0
            else ()
        ),
        compile_spec=KernelCompileSpec.from_key(
            "integration.tp_moe.dynamic",
            9,
            cache_key,
        ),
        dsl_compile_options=dsl_compile_options,
    )

    if reuse_compiled:
        _DYNAMIC_KERNEL_CACHE[cache_key] = compiled
    return compiled, mac


def _launch_dynamic_flat(
    *,
    packed_a_view: torch.Tensor,
    packed_a_flat: torch.Tensor,
    scale_flat: torch.Tensor,
    materialized_intermediate: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    pair_head: torch.Tensor,
    producers_done_count: torch.Tensor,
    all_work_published: torch.Tensor,
    task_head: torch.Tensor,
    task_tail: torch.Tensor,
    task_ready: torch.Tensor,
    task_expert: torch.Tensor,
    task_m_tile: torch.Tensor,
    task_slice_begin: torch.Tensor,
    task_slice_count: torch.Tensor,
    task_valid_rows: torch.Tensor,
    tile_write_count: torch.Tensor,
    row_counts: torch.Tensor,
    expert_write_rows: torch.Tensor,
    expert_tile_base: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_sf: torch.Tensor,
    down_fp4: torch.Tensor,
    down_sf: torch.Tensor,
    sfb_w13_mx: torch.Tensor,
    sfb_down_mx: torch.Tensor,
    w13_residual: torch.Tensor,
    down_residual: torch.Tensor,
    w13_rp: torch.Tensor,
    w13_sfb_rp: torch.Tensor,
    down_rp: torch.Tensor,
    down_sfb_rp: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    max_rows: int,
    physical_tiles_capacity: int,
    task_capacity: int,
    topk_ids_are_i32: bool,
    fast_math: bool,
    activation: str,
    quant_mode: str,
    w4a8_repacked: bool,
    w4a8_n64_repacked: bool,
    nvfp4_materialize_intermediate: bool,
    nvfp4_inline_scales: bool,
    share_input_across_experts: bool,
    deterministic_output: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    external_route_plan_requested: bool,
    policy_max_active_clusters: int,
    volatile_launch_state: bool,
    planned_tile_m: int,
    planned_direct_routing: bool,
    planned_num_tokens: int,
    w4a8_csf_inline: bool = False,
) -> None:
    """Launch the compiled dynamic MoE kernel over bound workspace and weight views."""
    quant_mode = _normalize_quant_mode(quant_mode)
    # Output rows follow the reduction contract, never an independent caller
    # quantity: deterministic kernels write one row per route, others per token.
    scatter_rows = routed_rows if deterministic_output else m
    # A trellis-native w4a8 binding carries the fp16 boundary rotations in
    # the sfb slot (the QMMA repack stores int32 scale words there); the
    # trellis geometry is recovered from the operand extents.
    w4a8_trellis = _is_w4a8_quant_mode(quant_mode) and w13_sfb_rp.dtype == torch.float16
    trellis_bits = 0
    trellis_intermediate_hadamard = False
    trellis_lut_tensor = None
    if w4a8_trellis:
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _trellis256_execution_lut,
        )

        trellis_lut_tensor = _trellis256_execution_lut(a.device, LUT_E4M3)
        payload_u32 = w13_rp.numel() * w13_rp.element_size() // 4
        window_words = 2 * E * (k // 16) * (n // 16) * 8
        trellis_bits = payload_u32 // window_words
        if trellis_bits not in (2, 3, 4) or trellis_bits * window_words != payload_u32:
            raise RuntimeError(
                "w4a8 trellis payload extent does not match the launch "
                f"shape (E={E}, k={k}, n={n}, payload_u32={payload_u32})"
            )
        rot_elems = w13_sfb_rp.numel()
        if rot_elems == E * 6 * n:
            trellis_intermediate_hadamard = True
        elif rot_elems != E * 3 * n:
            raise RuntimeError(
                "w4a8 trellis rotation extent must be E*3n or E*6n fp16 "
                f"values, got {rot_elems} for E={E}, n={n}"
            )
    if w4a8_repacked and int(n) % 128 != 0 and not w4a8_n64_repacked:
        # The legacy N128 repack materializes zero-padded weight tails. Launch
        # that representation at its physical N extent. Exact N64 repacks keep
        # logical N and specialize the CuTe staging path instead.
        n = _dynamic_kernel_intermediate_size(n, quant_mode)
    decode_regime = bool(
        w4a8_repacked
        and _w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=planned_num_tokens * num_topk,
            num_experts=E,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )
    direct_routing = bool(
        (quant_mode != "w4a8_mx" or w4a8_repacked)
        and _dynamic_direct_routing_selected(
            route_mode="direct" if planned_direct_routing else "grouped",
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=E,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
    )
    mac_backend = "dynamic_w4a8_decode" if decode_regime else "dynamic"
    effective_mac = _get_impl_mac(mac_backend, routed_rows=routed_rows)
    multicta_enabled = _dynamic_multicta_enabled()
    selected_tile_m = _select_dynamic_tile_mn(
        m * num_topk,
        n,
        quant_mode,
        num_experts=E,
        activation=activation,
        planned_tile_m=planned_tile_m,
    )[0]
    materialize_intermediate = _w4a8_dynamic_materialized_enabled(
        quant_mode=quant_mode,
        activation=activation,
        num_tokens=planned_num_tokens,
        routed_rows=planned_num_tokens * num_topk,
        num_experts=E,
        k=k,
        n=n,
        w4a8_repacked=w4a8_repacked,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        planned_tile_m=planned_tile_m,
    ) or (quant_mode == "nvfp4" and nvfp4_materialize_intermediate)
    external_route_plan_supported = _dynamic_external_route_plan_supported(
        quant_mode=quant_mode,
        activation=activation,
        routed_rows=routed_rows,
        planned_tile_m=selected_tile_m,
        dynamic_route_mode=("direct" if direct_routing else "grouped"),
        deterministic_output=deterministic_output,
        w4a8_n64_repacked=w4a8_n64_repacked,
    )
    external_route_plan = bool(
        external_route_plan_supported
        and _env_flag(
            _DYNAMIC_EXTERNAL_ROUTE_PLAN_ENV,
            default=external_route_plan_requested,
        )
    )
    materialize_intermediate = materialize_intermediate or w4a8_n64_repacked
    if materialize_intermediate:
        required_intermediate_bytes = _dynamic_materialized_intermediate_bytes(
            quant_mode=quant_mode,
            rows_padded=physical_tiles_capacity * selected_tile_m,
            n=n,
        )
        available_intermediate_bytes = (
            materialized_intermediate.numel() * materialized_intermediate.element_size()
        )
        if available_intermediate_bytes < required_intermediate_bytes:
            raise ValueError(
                f"dynamic {quant_mode} materialized scratch exceeds preplanned "
                f"capacity: need {required_intermediate_bytes} bytes, have "
                f"{available_intermediate_bytes}; physical_tiles_capacity="
                f"{physical_tiles_capacity}, selected_tile_m={selected_tile_m}, "
                f"n={n}, routed_rows={routed_rows}, m={m}, num_topk={num_topk}"
            )
    if not multicta_enabled:
        effective_mac = 1
    elif w4a8_n64_repacked and selected_tile_m == 16:
        # The compact split-M16 front-end retains one cooperative participant
        # per SM. Its standalone phase kernels launch their own two-wave grids,
        # but those CTAs do not join the front-end's grid-wide barrier.
        effective_mac = min(effective_mac, get_num_sm(torch.device("cuda")))
    elif w4a8_repacked and selected_tile_m <= 32:
        # The compact repacked-W4A8 storage specialization is deliberately
        # sized for two resident CTAs/SM (49.15 KiB and a two-block register
        # limit).  The generic resident-grid cap is one CTA/SM because its
        # barrier must never over-launch.  This specialization has a proven-
        # resident second wave, so expose it to the same materialized work
        # queue.  Preserve routed-row MAC tuning proportionally and cap at the
        # physical two-wave grid.
        effective_mac = min(
            effective_mac * 2,
            get_num_sm(torch.device("cuda")) * 2,
        )
        if (
            _current_compute_capability() == (12, 1)
            and E == 256
            and k == 6144
            and n == 1024
            and 24 <= routed_rows <= 48
        ):
            # DSV4F TP2 decode saturates Spark's memory controllers with 24
            # resident CTAs; a larger grid only adds barrier participants.
            effective_mac = min(effective_mac, 24)
    if direct_routing:
        # One physical M tile per routed pair and one task per N128 slice.
        # Materialized M=1 has a second, wider route x N256 phase; size the
        # resident grid for whichever fixed phase exposes more parallel work.
        direct_task_count = routed_rows * max(1, (n + 127) // 128)
        if materialize_intermediate and m == 1:
            direct_task_count = max(
                direct_task_count,
                routed_rows * max(1, (k + 255) // 256),
            )
        effective_mac = min(effective_mac, direct_task_count)
    if (
        (
            external_route_plan
            or w4a8_n64_repacked
            or (w4a8_repacked and n % 128 == 0 and selected_tile_m <= 32)
        )
        and policy_max_active_clusters > 0
        and _first_env(
            f"B12X_{mac_backend.upper()}_MAX_ACTIVE_CLUSTERS",
            "B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS",
            "B12X_LEVEL10_MAX_ACTIVE_CLUSTERS",
        )
        is None
    ):
        effective_mac = min(
            effective_mac,
            policy_max_active_clusters,
        )
    # Do not enlarge this grid beyond the measured resident count: the fused
    # kernel has a resident-grid barrier, so over-launching would deadlock.
    compiled, mac = _get_dynamic_kernel(
        E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        topk_ids_dtype=torch.int32 if topk_ids_are_i32 else torch.int64,
        fast_math=fast_math,
        mac_override=effective_mac,
        activation=activation,
        quant_mode=quant_mode,
        w4a8_repacked=w4a8_repacked,
        w4a8_n64_repacked=w4a8_n64_repacked,
        nvfp4_materialize_intermediate=nvfp4_materialize_intermediate,
        nvfp4_inline_scales=nvfp4_inline_scales,
        direct_routing=direct_routing,
        external_route_plan=external_route_plan,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        trellis_bits=trellis_bits,
        trellis_intermediate_hadamard=trellis_intermediate_hadamard,
        planned_tile_m=planned_tile_m,
        planned_num_tokens=planned_num_tokens,
        w4a8_csf_inline=w4a8_csf_inline,
    )
    if volatile_launch_state and not external_route_plan:
        barrier_count.zero_()
        barrier_epoch.zero_()

    if external_route_plan:
        block_e = 1 << (int(E) - 1).bit_length()
        # A planned launch: compile-only planning warms and records this
        # specialization, so the race and the frozen serving path never
        # meet an uncached Triton program here.
        launch_triton(
            _dynamic_route_plan_kernel,
            (1,),
            flat_ids,
            row_counts,
            expert_tile_base,
            barrier_count,
            barrier_epoch,
            routed_rows,
            NUM_EXPERTS=E,
            TILE_M=selected_tile_m,
            BLOCK_E=block_e,
            BLOCK_ROUTES=_DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS,
            num_warps=8,
        )

    def _gptr(dtype, t, align=16):
        return make_ptr(
            dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    ids_cutlass_dtype = cutlass.Int32 if topk_ids_are_i32 else cutlass.Int64
    ids_align = 4 if topk_ids_are_i32 else 8
    # The MX byte-container recipes stage one E4M3 byte per activation element
    # in packed_a and UE8M0 K/32 block scales (weight SFBs included); the FP4
    # recipes keep the FP4x2 packed view and E4M3 vec16 scale pointers.
    is_byte_container = quant_mode in _MX_BYTE_CONTAINER_QUANT_MODES
    packed_a_cutlass_dtype = (
        cutlass.Float8E4M3FN if is_byte_container else cutlass.Float4E2M1FN
    )
    sf_cutlass_dtype = (
        cutlass.Float8E8M0FNU if is_byte_container else cutlass.Float8E4M3FN
    )
    compiled(
        _gptr(cutlass.BFloat16, a),
        _gptr(ids_cutlass_dtype, flat_ids, ids_align),
        _gptr(cutlass.Float32, flat_weights, 4),
        _gptr(packed_a_cutlass_dtype, packed_a_view),
        _gptr(sf_cutlass_dtype, scale_flat),
        _gptr(cutlass.Uint8, packed_a_flat),
        _gptr(cutlass.Uint8, scale_flat),
        _gptr(cutlass.Uint32, materialized_intermediate),
        barrier_count,
        barrier_epoch,
        pair_head,
        producers_done_count,
        all_work_published,
        task_head,
        task_tail,
        _gptr(cutlass.Int32, task_ready, 4),
        _gptr(cutlass.Int32, task_expert, 4),
        _gptr(cutlass.Int32, task_m_tile, 4),
        _gptr(cutlass.Int32, task_slice_begin, 4),
        _gptr(cutlass.Int32, task_slice_count, 4),
        _gptr(cutlass.Int32, task_valid_rows, 4),
        _gptr(cutlass.Int32, tile_write_count, 4),
        w13_fp4,
        _gptr(sf_cutlass_dtype, w13_sf),
        *(
            (_gptr(sf_cutlass_dtype, sfb_w13_mx),)
            if not _is_w4a8_quant_mode(quant_mode)
            else ()
        ),
        down_fp4,
        _gptr(sf_cutlass_dtype, down_sf),
        *(
            (
                _gptr(cutlass.Uint8, sfb_w13_mx),
                _gptr(cutlass.Uint8, sfb_down_mx),
                _gptr(cutlass.Uint8, w13_residual),
                _gptr(cutlass.Uint8, down_residual),
                _gptr(cutlass.Uint32, w13_rp),
                _gptr(cutlass.Uint32, w13_sfb_rp),
                _gptr(cutlass.Uint32, down_rp),
                _gptr(cutlass.Uint32, down_sfb_rp),
            )
            if _is_w4a8_quant_mode(quant_mode)
            else ()
        ),
        row_counts,
        expert_write_rows,
        expert_tile_base,
        input_gs,
        w1_alpha,
        w2_alpha,
        down_input_scale,
        _gptr(cutlass.BFloat16, scatter_output),
        _gptr(cutlass.Int32, token_map, 4),
        _gptr(cutlass.Float32, token_weights, 4),
        m,
        max_rows,
        scatter_rows,
        physical_tiles_capacity
        * _select_dynamic_tile_mn(
            m * num_topk,
            n,
            quant_mode,
            num_experts=E,
            activation=activation,
            planned_tile_m=planned_tile_m,
        )[0],
        task_capacity,
        physical_tiles_capacity,
        mac,
        current_cuda_stream(),
        *(
            (
                _gptr(cutlass.Uint8, trellis_lut_tensor),
                _gptr(cutlass.Float16, w13_sfb_rp),
            )
            if w4a8_trellis
            else ()
        ),
    )


_DYNAMIC_TILE_M_POLICY_CODES = {16: 0, 32: 1, 64: 2, 128: 3}
_DYNAMIC_POLICY_TILE_MS = tuple(_DYNAMIC_TILE_M_POLICY_CODES)


def _encode_dynamic_launch_policy(
    *,
    volatile_launch_state: bool,
    external_route_plan_requested: bool,
    policy_max_active_clusters: int,
    planned_tile_m: int,
    planned_direct_routing: bool,
    w4a8_n64_repacked: bool,
    nvfp4_materialize_intermediate: bool = False,
    nvfp4_inline_scales: bool = False,
    w4a8_csf_inline: bool = False,
) -> int:
    try:
        tile_code = _DYNAMIC_TILE_M_POLICY_CODES[int(planned_tile_m)]
    except KeyError as exc:
        raise ValueError(
            f"planned_tile_m must be one of {_DYNAMIC_POLICY_TILE_MS}"
        ) from exc
    if policy_max_active_clusters < -1:
        raise ValueError("policy_max_active_clusters must be -1 or nonnegative")
    return (
        int(bool(volatile_launch_state))
        | (int(bool(external_route_plan_requested)) << 1)
        | (tile_code << 2)
        | (int(bool(planned_direct_routing)) << 4)
        | (int(bool(w4a8_n64_repacked)) << 5)
        | (int(bool(nvfp4_materialize_intermediate)) << 6)
        | (int(bool(nvfp4_inline_scales)) << 7)
        | (int(bool(w4a8_csf_inline)) << 8)
        | ((int(policy_max_active_clusters) + 1) << 9)
    )


def _decode_dynamic_launch_policy(
    value: int,
) -> tuple[bool, bool, int, bool, bool, bool, bool, bool, int]:
    value = int(value)
    return (
        bool(value & 1),
        bool(value & 2),
        _DYNAMIC_POLICY_TILE_MS[(value >> 2) & 3],
        bool(value & 16),
        bool(value & 32),
        bool(value & 64),
        bool(value & 128),
        bool(value & 256),
        (value >> 9) - 1,
    )


@torch.library.custom_op(
    "b12x::tp_moe_dynamic_launch",
    # Only caller-owned execution buffers are writable. Marking every operand
    # mutable increments canonical weight/scale versions even though the GPU
    # only reads them, invalidating preparation-time scale metadata.
    mutates_args=(
        "packed_a_view",
        "packed_a_flat",
        "scale_flat",
        "materialized_intermediate",
        "barrier_count",
        "barrier_epoch",
        "pair_head",
        "producers_done_count",
        "all_work_published",
        "task_head",
        "task_tail",
        "task_ready",
        "task_expert",
        "task_m_tile",
        "task_slice_begin",
        "task_slice_count",
        "task_valid_rows",
        "tile_write_count",
        "row_counts",
        "expert_write_rows",
        "expert_tile_base",
        "token_map",
        "token_weights",
        "scatter_output",
    ),
)
def _tp_moe_dynamic_launch_op(
    packed_a_view: torch.Tensor,
    packed_a_flat: torch.Tensor,
    scale_flat: torch.Tensor,
    materialized_intermediate: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    pair_head: torch.Tensor,
    producers_done_count: torch.Tensor,
    all_work_published: torch.Tensor,
    task_head: torch.Tensor,
    task_tail: torch.Tensor,
    task_ready: torch.Tensor,
    task_expert: torch.Tensor,
    task_m_tile: torch.Tensor,
    task_slice_begin: torch.Tensor,
    task_slice_count: torch.Tensor,
    task_valid_rows: torch.Tensor,
    tile_write_count: torch.Tensor,
    row_counts: torch.Tensor,
    expert_write_rows: torch.Tensor,
    expert_tile_base: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_sf: torch.Tensor,
    down_fp4: torch.Tensor,
    down_sf: torch.Tensor,
    sfb_w13_mx: torch.Tensor,
    sfb_down_mx: torch.Tensor,
    w13_residual: torch.Tensor,
    down_residual: torch.Tensor,
    w13_rp: torch.Tensor,
    w13_sfb_rp: torch.Tensor,
    down_rp: torch.Tensor,
    down_sfb_rp: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    max_rows: int,
    physical_tiles_capacity: int,
    task_capacity: int,
    topk_ids_are_i32: bool,
    fast_math: bool,
    activation: str,
    quant_mode: str,
    w4a8_repacked: bool,
    share_input_across_experts: bool,
    deterministic_output: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    launch_policy: int,
    planned_num_tokens: int,
) -> None:
    (
        volatile_launch_state,
        external_route_plan_requested,
        planned_tile_m,
        planned_direct_routing,
        w4a8_n64_repacked,
        nvfp4_materialize_intermediate,
        nvfp4_inline_scales,
        w4a8_csf_inline,
        policy_max_active_clusters,
    ) = _decode_dynamic_launch_policy(launch_policy)
    _launch_dynamic_flat(
        packed_a_view=packed_a_view,
        packed_a_flat=packed_a_flat,
        scale_flat=scale_flat,
        materialized_intermediate=materialized_intermediate,
        sfb_w13_mx=sfb_w13_mx,
        sfb_down_mx=sfb_down_mx,
        w13_residual=w13_residual,
        down_residual=down_residual,
        w13_rp=w13_rp,
        w13_sfb_rp=w13_sfb_rp,
        down_rp=down_rp,
        down_sfb_rp=down_sfb_rp,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        pair_head=pair_head,
        producers_done_count=producers_done_count,
        all_work_published=all_work_published,
        task_head=task_head,
        task_tail=task_tail,
        task_ready=task_ready,
        task_expert=task_expert,
        task_m_tile=task_m_tile,
        task_slice_begin=task_slice_begin,
        task_slice_count=task_slice_count,
        task_valid_rows=task_valid_rows,
        tile_write_count=tile_write_count,
        row_counts=row_counts,
        expert_write_rows=expert_write_rows,
        expert_tile_base=expert_tile_base,
        input_gs=input_gs,
        down_input_scale=down_input_scale,
        token_map=token_map,
        token_weights=token_weights,
        w13_fp4=w13_fp4,
        w13_sf=w13_sf,
        down_fp4=down_fp4,
        down_sf=down_sf,
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        a=a,
        flat_ids=flat_ids,
        flat_weights=flat_weights,
        scatter_output=scatter_output,
        E=E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        routed_rows=routed_rows,
        max_rows=max_rows,
        physical_tiles_capacity=physical_tiles_capacity,
        task_capacity=task_capacity,
        topk_ids_are_i32=topk_ids_are_i32,
        fast_math=fast_math,
        activation=activation,
        quant_mode=quant_mode,
        w4a8_repacked=w4a8_repacked,
        w4a8_n64_repacked=w4a8_n64_repacked,
        nvfp4_materialize_intermediate=nvfp4_materialize_intermediate,
        nvfp4_inline_scales=nvfp4_inline_scales,
        share_input_across_experts=share_input_across_experts,
        deterministic_output=deterministic_output,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        external_route_plan_requested=external_route_plan_requested,
        policy_max_active_clusters=policy_max_active_clusters,
        volatile_launch_state=volatile_launch_state,
        planned_tile_m=planned_tile_m,
        planned_direct_routing=planned_direct_routing,
        planned_num_tokens=planned_num_tokens,
        w4a8_csf_inline=w4a8_csf_inline,
    )


@_tp_moe_dynamic_launch_op.register_fake
def _tp_moe_dynamic_launch_fake(
    packed_a_view: torch.Tensor,
    packed_a_flat: torch.Tensor,
    scale_flat: torch.Tensor,
    materialized_intermediate: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    pair_head: torch.Tensor,
    producers_done_count: torch.Tensor,
    all_work_published: torch.Tensor,
    task_head: torch.Tensor,
    task_tail: torch.Tensor,
    task_ready: torch.Tensor,
    task_expert: torch.Tensor,
    task_m_tile: torch.Tensor,
    task_slice_begin: torch.Tensor,
    task_slice_count: torch.Tensor,
    task_valid_rows: torch.Tensor,
    tile_write_count: torch.Tensor,
    row_counts: torch.Tensor,
    expert_write_rows: torch.Tensor,
    expert_tile_base: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    token_map: torch.Tensor,
    token_weights: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_sf: torch.Tensor,
    down_fp4: torch.Tensor,
    down_sf: torch.Tensor,
    sfb_w13_mx: torch.Tensor,
    sfb_down_mx: torch.Tensor,
    w13_residual: torch.Tensor,
    down_residual: torch.Tensor,
    w13_rp: torch.Tensor,
    w13_sfb_rp: torch.Tensor,
    down_rp: torch.Tensor,
    down_sfb_rp: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    max_rows: int,
    physical_tiles_capacity: int,
    task_capacity: int,
    topk_ids_are_i32: bool,
    fast_math: bool,
    activation: str,
    quant_mode: str,
    w4a8_repacked: bool,
    share_input_across_experts: bool,
    deterministic_output: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    launch_policy: int,
    planned_num_tokens: int,
) -> None:
    del launch_policy
    return None


def _launch_dynamic(
    *,
    workspace: TPDynamicWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    max_rows: int,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    stream,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    w4a8_prepared: dict | None = None,
    w4a8_n64_repacked: bool = False,
    nvfp4_materialize_intermediate: bool = False,
    nvfp4_inline_scales: tuple | None = None,
    share_input_across_experts: bool = False,
    deterministic_output: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    external_route_plan_requested: bool = False,
    policy_max_active_clusters: int = -1,
    planned_tile_m: int = 128,
    dynamic_route_mode: str = "grouped",
    planned_num_tokens: int | None = None,
    w4a8_csf_inline: tuple | None = None,
) -> None:
    del stream
    if dynamic_route_mode not in {"direct", "grouped"}:
        raise ValueError("dynamic_route_mode must be 'direct' or 'grouped'")
    launch_policy = _encode_dynamic_launch_policy(
        volatile_launch_state=workspace.volatile_launch_state,
        external_route_plan_requested=external_route_plan_requested,
        policy_max_active_clusters=policy_max_active_clusters,
        planned_tile_m=planned_tile_m,
        planned_direct_routing=dynamic_route_mode == "direct",
        w4a8_n64_repacked=w4a8_n64_repacked,
        nvfp4_materialize_intermediate=nvfp4_materialize_intermediate,
        nvfp4_inline_scales=nvfp4_inline_scales is not None,
        w4a8_csf_inline=w4a8_csf_inline is not None,
    )
    if deterministic_output and workspace.route_output.numel() < routed_rows * k:
        raise RuntimeError(
            "deterministic dynamic launch exceeds the planned route-output "
            f"capacity: need {routed_rows * k} elements, got "
            f"{workspace.route_output.numel()}"
        )
    kernel_output = workspace.route_output if deterministic_output else scatter_output
    w4a8_repacked = w4a8_prepared is not None
    w13_rp = w4a8_prepared["w13_rp"] if w4a8_repacked else workspace.row_counts
    w13_sfb_rp = w4a8_prepared["w13_sfb"] if w4a8_repacked else workspace.row_counts
    down_rp = w4a8_prepared["w2_rp"] if w4a8_repacked else workspace.row_counts
    down_sfb_rp = w4a8_prepared["w2_sfb"] if w4a8_repacked else workspace.row_counts
    if w4a8_csf_inline is not None:
        if not (w4a8_repacked and w4a8_n64_repacked):
            raise RuntimeError("inline MXFP4-CSF scales require compact W4A8 weights")
        # Every scale operand carries the inline storage: no launch argument
        # references the caller's expansion scratch.
        w13_sfb_rp, down_sfb_rp = (plane.storage for plane in w4a8_csf_inline)
    dynamic_w13_sf = (
        weights.w13_up_sf if weights.w13_up_sf is not None else weights.w13_sf
    )
    dynamic_w13_gate_sf = (
        weights.w13_gate_sf if weights.w13_gate_sf is not None else weights.sfb_w13_mx
    )
    dynamic_down_fp4 = (
        weights.dynamic_down_fp4
        if weights.dynamic_down_fp4 is not None
        else weights.down_fp4
    )
    dynamic_down_sf = (
        weights.dynamic_down_sf
        if weights.dynamic_down_sf is not None
        else weights.down_sf
    )
    if nvfp4_inline_scales is not None:
        dynamic_w13_sf, dynamic_down_sf = (p.storage for p in nvfp4_inline_scales)
    sfb_down_mx = weights.sfb_down_mx
    if w4a8_csf_inline is not None:
        dynamic_w13_sf = dynamic_w13_gate_sf = w13_sfb_rp
        dynamic_down_sf = sfb_down_mx = down_sfb_rp
    torch.ops.b12x.tp_moe_dynamic_launch(
        workspace.packed_a_view,
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.materialized_intermediate,
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_ready,
        workspace.task_expert,
        workspace.task_m_tile,
        workspace.task_slice_begin,
        workspace.task_slice_count,
        workspace.task_valid_rows,
        workspace.tile_write_count,
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        workspace.input_gs,
        workspace.down_input_scale,
        workspace.token_map,
        workspace.token_weights,
        weights.w13_fp4,
        dynamic_w13_sf,
        dynamic_down_fp4,
        dynamic_down_sf,
        dynamic_w13_gate_sf
        if dynamic_w13_gate_sf is not None
        else workspace.row_counts,
        sfb_down_mx if sfb_down_mx is not None else workspace.row_counts,
        weights.w13_residual
        if weights.w13_residual is not None
        else workspace.row_counts,
        weights.down_residual
        if weights.down_residual is not None
        else workspace.row_counts,
        w13_rp,
        w13_sfb_rp,
        down_rp,
        down_sfb_rp,
        weights.w1_alpha,
        weights.w2_alpha,
        a,
        flat_ids,
        flat_weights,
        kernel_output,
        E,
        m,
        k,
        n,
        num_topk,
        routed_rows,
        max_rows,
        workspace.physical_tiles_capacity,
        workspace.task_capacity,
        topk_ids_dtype == torch.int32,
        bool(fast_math),
        activation,
        quant_mode,
        w4a8_repacked,
        bool(share_input_across_experts),
        bool(deterministic_output),
        swiglu_limit,
        float(swiglu_alpha),
        float(swiglu_beta),
        launch_policy,
        m if planned_num_tokens is None else planned_num_tokens,
    )


def _launch_dynamic_topk_sum(
    *,
    route_output: torch.Tensor,
    output: torch.Tensor,
    m: int,
    num_topk: int,
    k: int,
    stream,
) -> None:
    if route_output.numel() < m * num_topk * k:
        raise RuntimeError(
            "top-k reduction exceeds the planned route-output capacity: "
            f"need {m * num_topk * k} elements, got {route_output.numel()}"
        )
    from b12x.moe._shared.kernels.w4a16.kernel import (
        compile_w4a16_topk_sum,
    )

    element_dtype = _w4a16_element_dtype(route_output.dtype)
    compile_w4a16_topk_sum(
        m=m,
        topk=num_topk,
        hidden_size=k,
        element_dtype=element_dtype,
        float32_output=output.dtype == torch.float32,
    )
    torch.ops.b12x.w4a16_topk_sum_launch(
        route_output,
        output,
        m,
        num_topk,
        k,
        element_dtype,
        int(stream),
    )


def _launch_compact_micro_flat(
    *,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    fast_math: bool,
    share_input_across_experts: bool,
    share_expert_scales: bool,
    activation: str,
    quant_mode: str,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    volatile_launch_state: bool,
) -> None:
    quant_mode = _normalize_quant_mode(quant_mode)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    micro_cls = activation_spec.micro_kernel_cls
    # The QMMA tiny-decode band never reaches the compact op, so a w4a8_mx
    # arrival here is the trellis-native micro path: w1_storage is the
    # projection-major payload, w1_scale_storage carries the per-expert
    # boundary rotations, and the trellis geometry is recovered from the
    # operand extents.
    w4a8_trellis = quant_mode == "w4a8_mx"
    trellis_bits = 0
    trellis_intermediate_hadamard = False
    trellis_lut = None
    trellis_rotations = None
    if w4a8_trellis:
        window_words = 2 * weight_E * (k // 16) * (n // 16) * 32
        trellis_bits = w1_storage.numel() // window_words
        if (
            trellis_bits not in (2, 3, 4)
            or trellis_bits * window_words != w1_storage.numel()
        ):
            raise RuntimeError(
                "w4a8 trellis payload extent does not match the launch "
                f"shape (E={weight_E}, k={k}, n={n}, "
                f"payload_bytes={w1_storage.numel()})"
            )
        rot_elems = w1_scale_storage.numel()
        if rot_elems == weight_E * 6 * n:
            trellis_intermediate_hadamard = True
        elif rot_elems != weight_E * 3 * n:
            raise RuntimeError(
                "w4a8 trellis rotation extent must be E*3n or E*6n fp16 "
                f"values, got {rot_elems} for E={weight_E}, n={n}"
            )
        from b12x.moe._shared.kernels.w4a16.kernel import (
            _trellis256_execution_lut,
        )

        trellis_lut = _trellis256_execution_lut(a.device, LUT_E4M3)
        trellis_rotations = w1_scale_storage
    # The split path is opt-in until it matches W4A8 residual-scale numerics.
    use_native_nvfp4_split = (
        quant_mode == "w4a8_nvfp4"
        and 1 <= m <= 8
        and activation == "silu"
        and os.environ.get("B12X_NVFP4_SPLIT_DECODE", "0") != "0"
    )
    if use_native_nvfp4_split:
        # Preserve the ModelOpt NVFP4 representation and scalar math exactly.
        # Stream order supplies the FC1->FC2 dependency, so the 188-CTA
        # resident-grid barrier in the fused decode body is unnecessary.
        for phase in (1, 2):
            compiled, grid_x = _get_micro_kernel(
                weight_E,
                m,
                k,
                n,
                num_topk,
                topk_ids_dtype=launch_ids.dtype,
                fast_math=fast_math,
                share_input_across_experts=share_input_across_experts,
                share_expert_scales=share_expert_scales,
                single_token=True,
                activation=activation,
                device=a.device,
                quant_mode=quant_mode,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                compile_time_phase=phase,
            )
            split_block_dim = (
                _DIRECT_MICRO_BLOCK_DIM // 2
                if phase == 2 and m == 1
                else _DIRECT_MICRO_BLOCK_DIM
            )
            if not _compiled_direct_micro_accepts_block_dim(compiled, split_block_dim):
                raise RuntimeError(
                    "compiled split NVFP4 micro MoE kernel cannot launch"
                )
            micro_cls.launch(
                compiled,
                x=a,
                w1_fp4=w1_storage,
                w1_blockscale=w1_scale_storage,
                w1_alphas=w1_alpha,
                a1_gscale=input_gs,
                a2_gscale=down_input_scale,
                inter_fp32=micro_intermediate,
                w2_fp4=w2_storage,
                w2_blockscale=w2_scale_storage,
                w2_alphas=w2_alpha,
                topk_ids=launch_ids.view(m, num_topk),
                topk_weights=flat_weights.view(m, num_topk),
                out=scatter_output,
                barrier_count=barrier_count,
                barrier_epoch=barrier_epoch,
                m=m,
                grid_x=grid_x,
            )
        return
    compiled, grid_x = _get_micro_kernel(
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype=launch_ids.dtype,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=(m == 1),
        activation=activation,
        device=a.device,
        quant_mode=quant_mode,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout="trellis_t256" if w4a8_trellis else None,
        trellis_bits=trellis_bits,
        trellis_intermediate_hadamard=trellis_intermediate_hadamard,
    )
    if not _compiled_direct_micro_accepts_block_dim(
        compiled,
        _DIRECT_MICRO_BLOCK_DIM,
    ):
        raise RuntimeError("compiled direct micro MoE kernel cannot launch")
    if volatile_launch_state:
        barrier_count.zero_()
        barrier_epoch.zero_()
    micro_cls.launch(
        compiled,
        x=a,
        w1_fp4=w1_storage,
        w1_blockscale=w1_scale_storage,
        w1_alphas=w1_alpha,
        a1_gscale=input_gs,
        a2_gscale=down_input_scale,
        inter_fp32=micro_intermediate,
        w2_fp4=w2_storage,
        w2_blockscale=w2_scale_storage,
        w2_alphas=w2_alpha,
        topk_ids=launch_ids.view(m, num_topk),
        topk_weights=flat_weights.view(m, num_topk),
        out=scatter_output,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        m=m,
        grid_x=grid_x,
        trellis_lut=trellis_lut,
        trellis_rotations=trellis_rotations,
    )


def _tiny_decode_enabled() -> bool:
    # Default on; B12X_W4A8_TINY_DECODE=0 is the kill switch.
    return os.environ.get("B12X_W4A8_TINY_DECODE", "1") != "0"


def _tiny_decode_supports(*, num_tokens: int, k: int, n: int, activation: str) -> bool:
    # The rp storage is ceil-tiled with zero-filled tails, and the tiny FC2
    # bounds its intermediate loads by the logical n, so any 32-aligned shard
    # works (e.g. 352 from GLM 2048/TP6, 192 from DS4-Pro 3072/TP16).
    return (
        1 <= num_tokens <= 4
        and activation == "silu"
        and k % 256 == 0
        and n % 32 == 0
        and (k // 128) % 4 == 0
    )


_TINY_DECODE_KERNEL_CACHE: dict = {}
register_program_cache(_TINY_DECODE_KERNEL_CACHE)


def _get_tiny_decode_kernel(
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    *,
    device: torch.device | None = None,
):
    from b12x.moe._shared.kernels.tiny_decode import (
        MoETinyDecodeKernelBackendPhase1,
        MoETinyDecodeKernelBackendPhase2,
    )

    compiled_phases = []
    for kernel_cls in (
        MoETinyDecodeKernelBackendPhase1,
        MoETinyDecodeKernelBackendPhase2,
    ):
        kernel = kernel_cls()
        kernel.configure(m, k, n, num_topk, weight_E, device=device)
        cache_key = ("tiny_decode",) + kernel.__cache_key__
        cached = None if planning() else _TINY_DECODE_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            compiled_phases.append(cached)
            continue

        def dummy(dt):
            return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

        raise_if_kernel_resolution_frozen(
            "cute.compile", target=kernel, cache_key=cache_key
        )
        compiled = b12x_compile(
            kernel,
            dummy(cutlass.BFloat16),
            dummy(cutlass.Uint8),
            dummy(cutlass.Uint8),
            dummy(cutlass.Float32),
            dummy(cutlass.Uint8),
            dummy(cutlass.Uint8),
            dummy(cutlass.Int32),
            dummy(cutlass.Float32),
            dummy(cutlass.BFloat16),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "integration.tp_moe.tiny_decode",
                1,
                cache_key,
            ),
        )
        if not planning():
            _TINY_DECODE_KERNEL_CACHE[cache_key] = compiled
        compiled_phases.append(compiled)
    return compiled_phases[0], compiled_phases[1]


def _launch_tiny_decode_flat(
    *,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
) -> None:
    from b12x.moe._shared.kernels.tiny_decode import (
        MoETinyDecodeKernelBackend,
    )

    del barrier_count, barrier_epoch
    compiled_fc1, compiled_fc2 = _get_tiny_decode_kernel(
        weight_E, m, k, n, num_topk, device=a.device
    )
    rt = m * num_topk
    inter = micro_intermediate.view(torch.float32).reshape(-1)[: rt * 2 * n]
    MoETinyDecodeKernelBackend.launch(
        compiled_fc1,
        compiled_fc2,
        x=a.reshape(-1),
        w13_rp=w1_storage,
        sfb13=w1_scale_storage,
        inter_fp32=inter,
        w2_rp=w2_storage,
        sfb2=w2_scale_storage,
        topk_ids=launch_ids,
        topk_weights=flat_weights,
        out=scatter_output.reshape(-1),
    )


@torch.library.custom_op(
    "b12x::tp_moe_tiny_decode_launch",
    mutates_args="unknown",
)
def _tp_moe_tiny_decode_launch_op(
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    volatile_launch_state: bool,
) -> None:
    del volatile_launch_state
    _launch_tiny_decode_flat(
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        micro_intermediate=micro_intermediate,
        w1_storage=w1_storage,
        w1_scale_storage=w1_scale_storage,
        w2_storage=w2_storage,
        w2_scale_storage=w2_scale_storage,
        a=a,
        launch_ids=launch_ids,
        flat_weights=flat_weights,
        scatter_output=scatter_output,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
    )


@_tp_moe_tiny_decode_launch_op.register_fake
def _tp_moe_tiny_decode_launch_fake(
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    volatile_launch_state: bool,
) -> None:
    return None


@torch.library.custom_op(
    "b12x::tp_moe_compact_micro_launch",
    # Decode must not invalidate immutable scale metadata used by prefill.
    mutates_args=(
        "barrier_count",
        "barrier_epoch",
        "micro_intermediate",
        "scatter_output",
    ),
)
def _tp_moe_compact_micro_launch_op(
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    fast_math: bool,
    share_input_across_experts: bool,
    share_expert_scales: bool,
    activation: str,
    quant_mode: str,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    volatile_launch_state: bool,
) -> None:
    _launch_compact_micro_flat(
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        micro_intermediate=micro_intermediate,
        w1_storage=w1_storage,
        w1_scale_storage=w1_scale_storage,
        w2_storage=w2_storage,
        w2_scale_storage=w2_scale_storage,
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        a=a,
        launch_ids=launch_ids,
        flat_weights=flat_weights,
        input_gs=input_gs,
        down_input_scale=down_input_scale,
        scatter_output=scatter_output,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        activation=activation,
        quant_mode=quant_mode,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        volatile_launch_state=volatile_launch_state,
    )


@_tp_moe_compact_micro_launch_op.register_fake
def _tp_moe_compact_micro_launch_fake(
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    micro_intermediate: torch.Tensor,
    w1_storage: torch.Tensor,
    w1_scale_storage: torch.Tensor,
    w2_storage: torch.Tensor,
    w2_scale_storage: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    a: torch.Tensor,
    launch_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    fast_math: bool,
    share_input_across_experts: bool,
    share_expert_scales: bool,
    activation: str,
    quant_mode: str,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    volatile_launch_state: bool,
) -> None:
    return None


def _launch_micro(
    *,
    workspace: TPMicroWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    routed_rows: int,
    topk_ids_dtype: torch.dtype,
    fast_math: bool,
    stream,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    activation: str = "silu",
    quant_mode: str = "nvfp4",
    swiglu_limit: float | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    unit_scale_contract: bool = False,
    w4a8_trellis: bool = False,
) -> None:
    del stream, unit_scale_contract
    quant_mode = _normalize_quant_mode(quant_mode)
    if quant_mode == "w4a8_mx" and not w4a8_trellis:
        # The w4a8_mx tiny band reaches the compact family only via the
        # tiny_decode resolve gate; its weights are N256/K128-repacked,
        # which direct-micro cannot read.
        if flat_ids.dtype == torch.int32 and flat_ids.is_contiguous():
            launch_ids = flat_ids
        else:
            launch_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            launch_ids.copy_(flat_ids.to(torch.int32))
        torch.ops.b12x.tp_moe_tiny_decode_launch(
            workspace.barrier_count,
            workspace.barrier_epoch,
            workspace.micro_intermediate,
            weights.w1_storage,
            weights.w1_scale_storage,
            weights.w2_storage,
            weights.w2_scale_storage,
            a,
            launch_ids,
            flat_weights,
            scatter_output,
            weight_E,
            m,
            k,
            n,
            num_topk,
            workspace.volatile_launch_state,
        )
        return
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    micro_cls = activation_spec.micro_kernel_cls
    del routed_rows, topk_ids_dtype
    # The trellis band is gated on the shared backend's shape predicate:
    # activation wrappers such as SiTU keep is_supported False to hold the
    # nvfp4-family compact path closed while serving the trellis arm.
    shape_supported = (
        MoEMicroKernelBackend.is_supported if w4a8_trellis else micro_cls.is_supported
    )
    use_micro_direct = (
        quant_mode == "nvfp4" or _is_w4a8_quant_mode(quant_mode)
    ) and shape_supported(
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        weight_E=weight_E,
    )
    # _resolve_workspace_layout routes unsupported shapes to dynamic, so a
    # micro workspace must always be able to run its bound shape.
    if not use_micro_direct:
        raise RuntimeError(
            "compact MoE workspace reached with a shape direct-micro cannot run "
            f"(m={m}, k={k}, n={n}, num_topk={num_topk}); such shapes must route "
            "to the dynamic backend (see _resolve_workspace_layout)."
        )
    if flat_ids.dtype == torch.int32 and flat_ids.is_contiguous():
        launch_ids = flat_ids
    else:
        launch_ids = workspace.compact_topk_ids[: flat_ids.numel()]
        launch_ids.copy_(flat_ids.to(torch.int32))
    torch.ops.b12x.tp_moe_compact_micro_launch(
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.micro_intermediate,
        weights.w1_storage,
        weights.w1_scale_storage,
        weights.w2_storage,
        weights.w2_scale_storage,
        weights.w1_alpha,
        weights.w2_alpha,
        a,
        launch_ids,
        flat_weights,
        input_gs,
        down_input_scale,
        scatter_output,
        weight_E,
        m,
        k,
        n,
        num_topk,
        bool(fast_math),
        bool(share_input_across_experts),
        bool(share_expert_scales),
        activation,
        quant_mode,
        swiglu_limit,
        float(swiglu_alpha),
        float(swiglu_beta),
        workspace.volatile_launch_state,
    )


def _require_binding_field(binding: TPMoEFP4Binding, field_name: str):
    value = getattr(binding, field_name)
    if value is None:
        raise RuntimeError(f"TP MoE FP4 binding is missing {field_name}")
    return value


def _finalize_trellis_output(
    binding: TPMoEFP4Binding, result: torch.Tensor
) -> torch.Tensor:
    target = binding.output_cast_target
    if target is None:
        return result
    target.copy_(result)
    return target


# Planned token capacity limit for W4A16 per-stage compressed-scale reads.
W4A16_CSF_STAGE_MAX_TOKENS = int(
    os.environ.get("B12X_W4A16_CSF_STAGE_MAX_TOKENS", "1536")
)


# Planned token capacity limit for compact W4A8 inline scale reads.
W4A8_CSF_INLINE_MAX_TOKENS = int(
    os.environ.get("B12X_W4A8_CSF_INLINE_MAX_TOKENS", "1536")
)


def _w4a16_reads_stage_scales(binding) -> bool:
    """Whether this W4A16 call reads compressed scales per stage (its planned launch's format)."""
    launch = getattr(binding, "fused_launch", None)
    if launch is not None:
        return getattr(launch, "scale_format", None) == "e4m3_k16_csf"
    raise RuntimeError("Compressed W4A16 scales require a prepared fused launch")


def b12x_moe_fp4(*, binding: TPMoEFP4Binding) -> torch.Tensor:
    """Execute one fully planned, prepared, and scratch-bound FP4 MoE launch."""
    if not isinstance(binding, TPMoEFP4Binding):
        raise TypeError("binding must be a TPMoEFP4Binding")

    a = binding.a
    experts = binding.experts
    if not isinstance(experts, B12XFP4ExpertWeights):
        raise TypeError("binding.experts must be a B12XFP4ExpertWeights")
    if binding.implementation == "trellis_mixed3":
        from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
            MixedTrellis3Binding,
            MixedTrellisBuffers,
            run_bound_mixed_trellis3,
        )

        mixed_binding = binding.mixed_trellis_binding
        mixed_buffers = binding.mixed_trellis_buffers
        if not isinstance(mixed_binding, MixedTrellis3Binding):
            raise TypeError(
                "projection-mixed Trellis binding is missing prepared artifacts"
            )
        if not isinstance(mixed_buffers, MixedTrellisBuffers):
            raise TypeError(
                "projection-mixed Trellis binding is missing scratch buffers"
            )
        result = run_bound_mixed_trellis3(
            binding.a,
            binding.topk_weights,
            binding.topk_ids,
            mixed_binding,
            mixed_buffers,
        )
        return _finalize_trellis_output(binding, result)
    plan = binding.execution_plan
    if plan is None:
        raise RuntimeError(
            "TP MoE binding is missing its execution plan; construct bindings "
            "with TPMoEScratchPlan.bind() or build_tp_moe_fp4_binding()"
        )
    a1_gscale = experts.a1_gscale
    w1_fp4 = experts.w1_fp4
    w1_blockscale = experts.w1_blockscale
    w1_alphas = experts.w1_alphas
    a2_gscale = experts.a2_gscale
    w2_fp4 = experts.w2_fp4
    w2_blockscale = experts.w2_blockscale
    w2_alphas = experts.w2_alphas
    topk_weights = binding.topk_weights
    topk_ids = binding.topk_ids
    # Inline MXFP4-CSF experts read compressed scales inside the W4A8 kernels;
    # other CSF experts expand their routed experts' scales into scratch first.
    csf_inline = experts.mxfp4_csf_inline
    if (
        csf_inline is not None
        and binding.implementation == "dynamic"
        and not binding.w4a8_csf_inline
    ):
        # Plans above the inline limit run the native kernels over every
        # expert's scales, expanded once from the inline storage.
        from b12x._lib.quant.mxfp4_csf_inline import expand_mxfp4_csf_inline

        for plane, scales in zip(
            csf_inline, (w1_blockscale, w2_blockscale), strict=True
        ):
            expand_mxfp4_csf_inline(
                plane, scales.view(torch.uint8).view(plane.num_experts, -1)
            )
        csf_inline = None
    if experts.mxfp4_csf is not None and csf_inline is None:
        experts.mxfp4_csf.decode(topk_ids, w1_blockscale, w2_blockscale)
    # The A4 prefill path (when present) reads expanded (W4A16-layout) scales.
    stage_scales = (
        experts.w4a16_expanded is not None
        and getattr(binding, "a4_prefill_launches", None) is None
        and _w4a16_reads_stage_scales(binding)
    )
    csf_reset_barriers = (
        experts.nvfp4_csf is not None
        and not stage_scales
        and binding.implementation == "micro"
        and topk_ids.numel() > 0
    )
    inline_scales = None
    if (
        experts.nvfp4_csf is not None
        and binding.implementation == "dynamic"
        and binding.quant_mode == "nvfp4"
        and plan.decode_config.nvfp4_inline_scales
    ):
        inline_scales = experts.nvfp4_csf.inline_scales
    if (
        experts.nvfp4_csf is not None
        and inline_scales is None
        and not stage_scales
        and not (binding.scales_expanded and not csf_reset_barriers)
    ):
        barriers = (
            (
                _require_binding_field(binding, "barrier_count"),
                _require_binding_field(binding, "barrier_epoch"),
            )
            if csf_reset_barriers
            else None
        )
        # W4A16 compressed storage expands into its shared scratch, not in place.
        expanded = experts.w4a16_expanded
        experts.nvfp4_csf.decode(
            topk_ids,
            w1_blockscale if expanded is None else expanded.w13_scale,
            w2_blockscale if expanded is None else expanded.w2_scale,
            barriers=barriers,
        )
    workspace = None
    apply_router_weight_on_input = binding.apply_router_weight_on_input
    output = binding.output
    input_scales_static = binding.input_scales_static
    fast_math = binding.fast_math
    activation = experts.activation
    quant_mode = binding.quant_mode
    swiglu_limit = binding.swiglu_limit
    swiglu_alpha = binding.swiglu_alpha
    swiglu_beta = binding.swiglu_beta
    activation_amax = binding.activation_amax
    layer_idx = binding.layer_idx
    unit_scale_contract = binding.unit_scale_contract
    source_format = experts.source_format
    w13_layout = experts.w13_layout
    quant_mode = _normalize_quant_mode_for_source(quant_mode, source_format)
    unit_scale_contract = bool(unit_scale_contract and quant_mode == "w4a16")

    if binding.implementation == "micro":
        workspace = TPMicroWorkspace(
            implementation=binding.implementation,
            # CSF expansion initializes these scalars in its preceding kernel.
            # Other weights retain the launch wrapper's scalar initialization.
            volatile_launch_state=not csf_reset_barriers,
            quant_mode=quant_mode,
            state_E=binding.state_E,
            weight_E=binding.weight_E,
            max_rows=binding.max_rows,
            k=binding.k,
            n=binding.n,
            num_topk=binding.num_topk,
            device=binding.device,
            dtype=binding.dtype,
            row_counts=_require_binding_field(binding, "row_counts"),
            token_map=_require_binding_field(binding, "token_map"),
            token_weights=_require_binding_field(binding, "token_weights"),
            packed_input=_require_binding_field(binding, "packed_input"),
            packed_input_scale=_require_binding_field(binding, "packed_input_scale"),
            barrier_count=_require_binding_field(binding, "barrier_count"),
            barrier_epoch=_require_binding_field(binding, "barrier_epoch"),
            packed_a_view=binding.packed_a_view,
            sfa_ptr=binding.sfa_ptr,
            packed_a_flat=binding.packed_a_flat,
            scale_flat=binding.scale_flat,
            packed_a_storage_ptr=binding.packed_a_storage_ptr,
            routed_rows_capacity=_require_binding_field(
                binding, "routed_rows_capacity"
            ),
            active_expert_count=_require_binding_field(binding, "active_expert_count"),
            weight_expert_ids=_require_binding_field(binding, "weight_expert_ids"),
            global_to_local_expert=_require_binding_field(
                binding, "global_to_local_expert"
            ),
            compact_topk_ids=_require_binding_field(binding, "compact_topk_ids"),
            micro_intermediate=_require_binding_field(binding, "micro_intermediate"),
        )
    elif binding.implementation == "dynamic":
        workspace = TPDynamicWorkspace(
            implementation=binding.implementation,
            # Eager-bind maps views only and no longer zeros the read-before-write
            # barrier scalars; the launch wrapper must re-zero them in-place each
            # call (gated on volatile_launch_state), like the W4A16 path.
            volatile_launch_state=True,
            quant_mode=quant_mode,
            state_E=binding.state_E,
            weight_E=binding.weight_E,
            max_rows=binding.max_rows,
            k=binding.k,
            n=binding.n,
            num_topk=binding.num_topk,
            device=binding.device,
            dtype=binding.dtype,
            row_counts=_require_binding_field(binding, "row_counts"),
            token_map=_require_binding_field(binding, "token_map"),
            token_weights=_require_binding_field(binding, "token_weights"),
            packed_input=_require_binding_field(binding, "packed_input"),
            packed_input_scale=_require_binding_field(binding, "packed_input_scale"),
            barrier_count=_require_binding_field(binding, "barrier_count"),
            barrier_epoch=_require_binding_field(binding, "barrier_epoch"),
            packed_a_view=binding.packed_a_view,
            sfa_ptr=binding.sfa_ptr,
            packed_a_flat=binding.packed_a_flat,
            scale_flat=binding.scale_flat,
            packed_a_storage_ptr=binding.packed_a_storage_ptr,
            routed_rows_capacity=_require_binding_field(
                binding, "routed_rows_capacity"
            ),
            physical_tiles_capacity=_require_binding_field(
                binding, "physical_tiles_capacity"
            ),
            task_capacity=_require_binding_field(binding, "task_capacity"),
            route_output=_require_binding_field(binding, "route_output"),
            materialized_intermediate=_require_binding_field(
                binding, "materialized_intermediate"
            ),
            expert_write_rows=_require_binding_field(binding, "expert_write_rows"),
            expert_tile_base=_require_binding_field(binding, "expert_tile_base"),
            input_gs=_require_binding_field(binding, "input_gs"),
            down_input_scale=_require_binding_field(binding, "down_input_scale"),
            pair_head=_require_binding_field(binding, "pair_head"),
            producers_done_count=_require_binding_field(
                binding, "producers_done_count"
            ),
            all_work_published=_require_binding_field(binding, "all_work_published"),
            task_head=_require_binding_field(binding, "task_head"),
            task_tail=_require_binding_field(binding, "task_tail"),
            task_ready=_require_binding_field(binding, "task_ready"),
            task_expert=_require_binding_field(binding, "task_expert"),
            task_m_tile=_require_binding_field(binding, "task_m_tile"),
            task_slice_begin=_require_binding_field(binding, "task_slice_begin"),
            task_slice_count=_require_binding_field(binding, "task_slice_count"),
            task_valid_rows=_require_binding_field(binding, "task_valid_rows"),
            tile_write_count=_require_binding_field(binding, "tile_write_count"),
        )
    elif binding.implementation != "w4a16":
        raise TypeError(
            f"unsupported TP MoE FP4 binding implementation {binding.implementation!r}"
        )

    quant_mode_arg = quant_mode
    source_format = _normalize_fp4_source_format(source_format)
    quant_mode = _normalize_quant_mode_for_source(quant_mode_arg, source_format)
    unit_scale_contract = bool(unit_scale_contract and quant_mode == "w4a16")
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    w13_layout = _normalize_w13_layout_for_activation(activation, w13_layout)
    if activation_amax is not None and quant_mode != "w4a16":
        raise NotImplementedError(
            "activation_amax calibration is only supported for W4A16"
        )
    num_topk = topk_ids.shape[1]
    m, k = a.shape
    prepared_payload = _prepared_payload_for_runtime(
        experts,
        quant_mode=quant_mode,
        source_format=source_format,
        activation=activation,
        w13_layout=w13_layout,
        dtype=a.dtype,
        hidden_size=k,
    )
    device = a.device
    prepared_hidden = experts.hidden_size
    if prepared_hidden != k:
        raise ValueError(
            f"prepared hidden_size mismatch: expected {k}, got {prepared_hidden}"
        )
    prepared_dtype = experts.plan.io_dtype
    if prepared_dtype != str(a.dtype).removeprefix("torch."):
        raise TypeError(
            f"prepared weights were built for {prepared_dtype}, but a has "
            f"dtype {a.dtype}"
        )
    weight_E = experts.num_experts
    n = experts.intermediate_size
    routed_rows = m * num_topk
    if apply_router_weight_on_input and quant_mode != "w4a16":
        raise NotImplementedError(
            "apply_router_weight_on_input is not implemented in b12x_moe_fp4"
        )
    if activation == SWIGLUOAI_UNINTERLEAVE and _is_w4a8_quant_mode(quant_mode):
        raise NotImplementedError(
            "activation='swigluoai_uninterleave' is not supported for W4A8 MoE"
        )
    if fast_math is None:
        fast_math = _FAST_MATH_DEFAULT
    # Shared scalar input scales are weight-side constants in the benchmarked
    # path, so treat them as static and avoid re-expanding them every launch.
    effective_input_scales_static = input_scales_static or (
        a1_gscale.numel() == 1 and a2_gscale.numel() == 1
    )
    if quant_mode == "w4a16":
        from b12x.moe._shared.kernels.w4a16.kernel import (
            run_w4a16_moe,
        )

        prepared = prepared_payload
        if experts.w4a16_expanded is not None and not stage_scales:
            prepared = experts.w4a16_expanded
        if prepared is None:
            raise RuntimeError(
                "the W4A16 weight plan did not materialize its required representation"
            )
        full_rotation = getattr(prepared, "weight_layout", "") == "trellis_t256"
        output_dtype = torch.float32 if full_rotation else a.dtype
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a caller-owned output buffer"
                )
            scatter_output = torch.empty(m, k, dtype=output_dtype, device=device)
        else:
            scatter_output = output
        if scatter_output.shape != (m, k):
            raise ValueError(
                f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}"
            )
        if scatter_output.dtype != output_dtype:
            raise ValueError(
                f"output must have dtype {output_dtype}, got {scatter_output.dtype}"
            )
        if scatter_output.device != device:
            raise ValueError(
                f"output must be on device {device}, got {scatter_output.device}"
            )
        if not scatter_output.is_contiguous():
            raise ValueError("output must be contiguous")

        if binding.implementation != "w4a16":
            raise TypeError("expected a W4A16 TP MoE binding")
        if (binding.routed_rows_capacity or 0) < routed_rows:
            raise RuntimeError(
                "W4A16 TP MoE binding capacity is too small: "
                f"capacity={binding.routed_rows_capacity}, requested={routed_rows}"
            )
        intermediate_cache13 = _require_binding_field(binding, "intermediate_cache13")
        intermediate_cache2 = _require_binding_field(binding, "intermediate_cache2")
        prefill_sum_accum = binding.prefill_sum_accum
        fc1_c_tmp = _require_binding_field(binding, "fc1_c_tmp")
        fc2_c_tmp = _require_binding_field(binding, "fc2_c_tmp")
        packed_route_indices = _require_binding_field(binding, "packed_route_indices")
        block_expert_ids = _require_binding_field(binding, "block_expert_ids")
        packed_route_count = _require_binding_field(binding, "packed_route_count")
        expert_offsets = _require_binding_field(binding, "expert_offsets")
        if full_rotation:
            # Persistent weights retain only a zero-copy placeholder. The live
            # barrier/counter storage is part of the caller-owned scratch plan.
            prepared = replace(
                prepared,
                workspace=_require_binding_field(binding, "kernel_workspace"),
            )
        fused_launch = binding.fused_launch
        topk_sum_launch = binding.topk_sum_launch
        if not topk_weights.is_contiguous():
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires contiguous W4A16 topk_weights"
                )
            topk_weights = topk_weights.contiguous()
        if not topk_ids.is_contiguous():
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires contiguous W4A16 topk_ids"
                )
            topk_ids = topk_ids.contiguous()
        result = run_w4a16_moe(
            a,
            prepared,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            fast_math=fast_math,
            intermediate_cache13=intermediate_cache13,
            intermediate_cache2=intermediate_cache2,
            prefill_sum_accum=prefill_sum_accum,
            output=scatter_output,
            fc1_c_tmp=fc1_c_tmp,
            fc2_c_tmp=fc2_c_tmp,
            packed_route_indices=packed_route_indices,
            block_expert_ids=block_expert_ids,
            packed_route_count=packed_route_count,
            expert_offsets=expert_offsets,
            expert_counts=_require_binding_field(binding, "expert_counts"),
            expert_map=binding.route_expert_map,
            output_expert_map=binding.output_expert_map,
            activation_amax=activation_amax,
            layer_idx=layer_idx,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            fused_launch=fused_launch,
            topk_sum_launch=topk_sum_launch,
            route_pack_launches=binding.route_pack_launches,
            route_block_size_m=binding.route_block_size_m,
            intermediate_rotation_scales=(
                prepared.intermediate_rotations if full_rotation else None
            ),
            full_rotation=full_rotation,
            suh_gate_table=prepared.gate_suh if full_rotation else None,
            suh_up_table=prepared.up_suh if full_rotation else None,
            svh_table=prepared.down_svh if full_rotation else None,
            rotation_a_gate=(
                _require_binding_field(binding, "rotation_a_gate")
                if full_rotation
                else None
            ),
            rotation_a_up=(
                _require_binding_field(binding, "rotation_a_up")
                if full_rotation
                else None
            ),
            route_mode=(
                "packed"
                if binding.route_pack_launches is not None
                else plan.decode_config.w4a16_route_mode or "auto"
            ),
        )
        return _finalize_trellis_output(binding, result)
    activation_spec = _get_activation_kernel_spec(activation, quant_mode=quant_mode)
    impl = plan.implementation
    max_rows = plan.max_rows
    if impl == "dynamic" and m > plan.max_tokens_per_launch:
        raise ValueError(
            "the bound MoE launch exceeds the dynamic kernel's per-launch "
            f"token limit ({m} > {plan.max_tokens_per_launch}); split the "
            "request while constructing bindings"
        )

    s = _resolve_workspace(
        workspace,
        plan=plan,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        input_scales_static=effective_input_scales_static,
    )

    # CUDA graph capture may run on a non-default stream, so the launch stream
    # must be fetched per-call rather than cached per-device.
    stream = current_cuda_stream()

    if impl == "micro":
        assert isinstance(s, TPMicroWorkspace)
        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

        micro_w4a8_trellis = (
            quant_mode == "w4a8_mx"
            and getattr(prepared_payload, "weight_layout", None) == "trellis_t256"
        )
        if micro_w4a8_trellis:
            wv = _w4a8_trellis_weight_views(
                prepared_payload,
                w1_alphas,
                w2_alphas,
            )
        elif prepared_payload is not None and quant_mode == "w4a8_mx":
            wv = _w4a8_prepared_weight_views(
                prepared_payload,
                w1_alphas,
                w2_alphas,
            )
        else:
            wv = _get_weight_views(
                w1_fp4,
                w1_blockscale,
                w2_fp4,
                w2_blockscale,
                w1_alphas,
                w2_alphas,
                n,
                k,
                activation_spec=activation_spec,
                quant_mode=quant_mode,
                w13_layout=w13_layout,
            )
        input_gs = _prepare_expert_scale(a1_gscale, weight_E)
        down_input_scale = _prepare_expert_scale(a2_gscale, weight_E)
    else:
        assert isinstance(s, TPDynamicWorkspace)
        if quant_mode == "w6a8_mx":
            # The prepared payload is a PreparedW6A8MXFP6Weights container
            # (source-native packed FP6 codes, MMA-swizzled UE8M0 scales,
            # per-expert alphas).  Build the dynamic-launch views from it
            # directly, mirroring the prepared-W4A8 branch below; the FP4
            # view builder must never touch packed FP6 bytes.
            if prepared_payload is None:
                raise RuntimeError(
                    "the w6a8_mx weight plan did not materialize its "
                    "prepared MX-FP6 representation"
                )
            wv = _get_w6a8_weight_views(
                prepared_payload.w13_packed,
                prepared_payload.w13_sf_swizzled,
                prepared_payload.w2_packed,
                prepared_payload.w2_sf_swizzled,
                w1_alphas,
                w2_alphas,
                n,
                k,
                activation_spec=activation_spec,
            )
        elif quant_mode == "w8a8_mx":
            # Weight preparation validated the MXFP8 geometry and resolved the
            # launch views once (_prepare_w8a8_from_mxfp8_source); the planner
            # guarantees the representation exists (representation_for).
            wv = prepared_payload.launch_views
        elif prepared_payload is not None and quant_mode == "w4a8_mx":
            wv = _w4a8_prepared_weight_views(
                prepared_payload,
                w1_alphas,
                w2_alphas,
            )
        else:
            wv = _get_weight_views(
                w1_fp4,
                w1_blockscale,
                w2_fp4,
                w2_blockscale,
                w1_alphas,
                w2_alphas,
                n,
                k,
                activation_spec=activation_spec,
                quant_mode=quant_mode,
                w13_layout=w13_layout,
            )
            if quant_mode == "nvfp4":
                _ensure_dynamic_nvfp4_tail_views(
                    wv,
                    n,
                    k,
                    activation_spec=activation_spec,
                )
        input_gs = s.input_gs
        down_input_scale = s.down_input_scale
        flat_ids = _flatten_routing_ids(topk_ids)
        flat_weights = _flatten_routing_weights(topk_weights)

    if output is None:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA graph capture requires a caller-owned output buffer")
        scatter_output = torch.zeros(m, k, dtype=a.dtype, device=device)
    else:
        scatter_output = output
    if scatter_output.shape != (m, k):
        raise ValueError(
            f"output must have shape {(m, k)}, got {tuple(scatter_output.shape)}"
        )
    if scatter_output.dtype != a.dtype:
        raise ValueError(
            f"output must have dtype {a.dtype}, got {scatter_output.dtype}"
        )
    if scatter_output.device != device:
        raise ValueError(
            f"output must be on device {device}, got {scatter_output.device}"
        )
    if not scatter_output.is_contiguous():
        raise ValueError("output must be contiguous")

    compact_w4a8_micro = bool(
        impl == "micro"
        and quant_mode == "w4a8_mx"
        and getattr(prepared_payload, "n64_repack", False)
    )
    if compact_w4a8_micro:
        compact = binding.compact_launches
        if compact is None:
            raise RuntimeError(
                "compact W4A8 execution requires its prepared native launchers"
            )
        from b12x.moe._shared.kernels.w4a8_compact_micro import (
            launch_w4a8_compact_micro,
        )

        if bool(compact.csf_inline) != (csf_inline is not None):
            raise RuntimeError(
                "compact W4A8 launchers and experts disagree on inline MXFP4-CSF scales"
            )
        route_output = launch_w4a8_compact_micro(
            scratch=s.micro_intermediate,
            _prepared_kernel=compact.kernels[topk_ids.dtype],
            _prepared_quantize=compact.quantize,
            max_active_clusters=plan.decode_config.max_active_clusters,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13=wv.w1_storage,
            w13_scales=(
                wv.w1_scale_storage if csf_inline is None else csf_inline[0].storage
            ),
            w2=wv.w2_storage,
            w2_scales=(
                wv.w2_scale_storage if csf_inline is None else csf_inline[1].storage
            ),
            alpha1=wv.w1_alpha,
            alpha2=wv.w2_alpha,
            input_scale=input_gs,
            down_scale=down_input_scale,
            max_tokens=plan.max_rows // num_topk,
            num_topk=num_topk,
            swiglu_limit=swiglu_limit,
            fast_math=fast_math,
        )
        from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat

        _w4a16_topk_sum_launch_flat(
            route_output,
            scatter_output,
            m,
            num_topk,
            k,
            "bf16",
            int(stream),
            launcher=compact.topk_sum,
        )
        return scatter_output

    if impl == "dynamic":
        if plan.decode_config.nvfp4_share_input and not experts.can_share_input(
            input_scales_static=True
        ):
            raise ValueError("input scales changed after shared-input preparation")
        deterministic_output = plan.deterministic_output
        decode_config = plan.decode_config
        planned_tile_m = plan.execution.tile_m
        if planned_tile_m is None:
            raise RuntimeError("dynamic MoE plan is missing its planned tile M")
        dense_w4a8_candidate = _w4a8_dynamic_dense_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=weight_E,
            k=k,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
        decode_w4a8_candidate = _w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=activation,
            routed_rows=routed_rows,
            num_experts=weight_E,
            n=n,
            deterministic_output=deterministic_output,
            planned_tile_m=planned_tile_m,
        )
        dynamic_w4a8_prepared = (
            _w4a8_prepared_dict(prepared_payload)
            if prepared_payload is not None and quant_mode == "w4a8_mx"
            else None
        )
        if (
            quant_mode == "w4a8_mx"
            and dynamic_w4a8_prepared is None
            and plan.execution.weight_layout is PreparedWeightLayout.QMMA_REPACKED
        ):
            raise RuntimeError(
                "the W4A8-MX weight plan did not materialize its required "
                "QMMA representation"
            )
        _launch_dynamic(
            workspace=s,
            weights=wv,
            a=a,
            flat_ids=flat_ids,
            flat_weights=flat_weights,
            scatter_output=scatter_output,
            E=weight_E,
            m=m,
            k=k,
            n=n,
            num_topk=num_topk,
            routed_rows=routed_rows,
            max_rows=max_rows,
            topk_ids_dtype=flat_ids.dtype,
            fast_math=fast_math,
            stream=stream,
            activation=activation,
            quant_mode=quant_mode,
            w4a8_prepared=dynamic_w4a8_prepared,
            nvfp4_inline_scales=inline_scales,
            w4a8_csf_inline=csf_inline if quant_mode == "w4a8_mx" else None,
            w4a8_n64_repacked=bool(getattr(prepared_payload, "n64_repack", False)),
            nvfp4_materialize_intermediate=decode_config.nvfp4_materialize_intermediate,
            deterministic_output=deterministic_output,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            external_route_plan_requested=decode_config.route_planner == "triton",
            policy_max_active_clusters=(
                -1
                if decode_config.max_active_clusters is None
                else decode_config.max_active_clusters
            ),
            planned_tile_m=planned_tile_m,
            planned_num_tokens=plan.routed_rows // plan.num_topk,
            dynamic_route_mode=decode_config.dynamic_route_mode or "",
            share_input_across_experts=(
                (quant_mode == "nvfp4" and decode_config.nvfp4_share_input)
                or (
                    dynamic_w4a8_prepared is not None
                    and _env_flag(
                        _DYNAMIC_W4A8_SHARE_INPUT_ENV,
                        default=(dense_w4a8_candidate or decode_w4a8_candidate),
                    )
                )
            ),
        )
        if deterministic_output:
            _launch_dynamic_topk_sum(
                route_output=s.route_output,
                output=scatter_output,
                m=m,
                num_topk=num_topk,
                k=k,
                stream=stream,
            )
    else:
        _launch_micro(
            workspace=s,
            weights=wv,
            a=a,
            flat_ids=flat_ids,
            flat_weights=flat_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            weight_E=weight_E,
            m=m,
            k=k,
            n=n,
            num_topk=num_topk,
            routed_rows=routed_rows,
            topk_ids_dtype=flat_ids.dtype,
            fast_math=fast_math,
            stream=stream,
            share_input_across_experts=(
                activation in ("relu2", "silu")
                and m == 1
                and a1_gscale.numel() == 1
                and os.environ.get("B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS", "1") != "0"
            ),
            share_expert_scales=(
                activation in ("relu2", "silu")
                and a1_gscale.numel() == 1
                and a2_gscale.numel() == 1
            ),
            activation=activation,
            quant_mode=quant_mode,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            unit_scale_contract=unit_scale_contract,
            w4a8_trellis=micro_w4a8_trellis,
        )
    return scatter_output


def _validate_sparse_routing(
    hidden_states: torch.Tensor, routing: B12XTopKRouting
) -> None:
    if routing.topk_ids.ndim != 2:
        raise ValueError(
            f"expected topk_ids with rank 2, got shape {tuple(routing.topk_ids.shape)}"
        )
    if routing.topk_weights.ndim != 2:
        raise ValueError(
            "expected topk_weights with rank 2, got shape "
            f"{tuple(routing.topk_weights.shape)}"
        )
    if routing.topk_ids.shape != routing.topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same shape, got "
            f"{tuple(routing.topk_ids.shape)} and {tuple(routing.topk_weights.shape)}"
        )
    if routing.topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "routing batch mismatch: expected "
            f"{hidden_states.shape[0]}, got {routing.topk_ids.shape[0]}"
        )
    if (
        routing.router_logits is not None
        and routing.router_logits.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError(
            "router_logits batch mismatch: expected "
            f"{hidden_states.shape[0]}, got {routing.router_logits.shape[0]}"
        )
    if (
        routing.flat_ids is not None
        and routing.flat_ids.numel() != routing.topk_ids.numel()
    ):
        raise ValueError(
            "flat_ids size mismatch: expected "
            f"{routing.topk_ids.numel()}, got {routing.flat_ids.numel()}"
        )
    if (
        routing.flat_weights is not None
        and routing.flat_weights.numel() != routing.topk_weights.numel()
    ):
        raise ValueError(
            "flat_weights size mismatch: expected "
            f"{routing.topk_weights.numel()}, got {routing.flat_weights.numel()}"
        )


def _alloc_route_workspace(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    logits_dtype: torch.dtype,
) -> _TPRouteWorkspace:
    required = _route_workspace_nbytes(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    )
    _emit_route_workspace_stats(
        storage="standalone",
        required_nbytes=required,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
        logits_dtype=logits_dtype,
    )
    return _TPRouteWorkspace(
        router_logits=torch.empty(
            num_tokens, num_experts, device=device, dtype=logits_dtype
        ),
        topk_logits=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
        topk_ids=torch.empty(num_tokens, top_k, device=device, dtype=torch.int32),
        topk_weights=torch.empty(num_tokens, top_k, device=device, dtype=torch.float32),
    )


def _route_workspace_specs(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> tuple[_TensorAllocSpec, ...]:
    return (
        _TensorAllocSpec("router_logits", (num_tokens, num_experts), logits_dtype),
        _TensorAllocSpec("topk_logits", (num_tokens, top_k), torch.float32),
        _TensorAllocSpec("topk_ids", (num_tokens, top_k), torch.int32),
        _TensorAllocSpec("topk_weights", (num_tokens, top_k), torch.float32),
    )


def _route_workspace_nbytes(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> int:
    offset = 0
    for spec in _route_workspace_specs(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    ):
        offset = align_up(offset, max(16, _dtype_nbytes(spec.dtype)))
        offset += _tensor_numel(spec.shape) * _dtype_nbytes(spec.dtype)
    return int(offset)


def _emit_route_workspace_stats(
    *,
    storage: str,
    required_nbytes: int,
    capacity_nbytes: int | None = None,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    logits_dtype: torch.dtype,
) -> None:
    return


def _materialize_route_workspace(
    shared_arena: torch.Tensor,
    *,
    offset_bytes: int,
    capacity_nbytes: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
) -> _TPRouteWorkspace:
    required = _route_workspace_nbytes(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    )
    if capacity_nbytes < required:
        raise ValueError(
            f"MoE route workspace requires {required} bytes, but only {capacity_nbytes} are available"
        )
    _emit_route_workspace_stats(
        storage="shared",
        required_nbytes=required,
        capacity_nbytes=capacity_nbytes,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        device=shared_arena.device,
        logits_dtype=logits_dtype,
    )
    offset = int(offset_bytes)
    tensors: Dict[str, torch.Tensor] = {}
    for spec in _route_workspace_specs(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
    ):
        tensors[spec.name], offset = _allocate_arena_tensor(shared_arena, offset, spec)
    return _TPRouteWorkspace(
        router_logits=tensors["router_logits"],
        topk_logits=tensors["topk_logits"],
        topk_ids=tensors["topk_ids"],
        topk_weights=tensors["topk_weights"],
    )


def _slice_route_workspace(
    route_workspace: _TPRouteWorkspace, num_tokens: int
) -> _TPRouteWorkspace:
    if route_workspace.router_logits.shape[0] == num_tokens:
        return route_workspace
    return _TPRouteWorkspace(
        router_logits=route_workspace.router_logits[:num_tokens],
        topk_logits=route_workspace.topk_logits[:num_tokens],
        topk_ids=route_workspace.topk_ids[:num_tokens],
        topk_weights=route_workspace.topk_weights[:num_tokens],
    )


def _get_route_workspace(
    hidden_states: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    logits_dtype: torch.dtype,
    workspace: TPMoEWorkspace | TPW4A16Workspace | TPMoEWorkspacePool | None,
) -> _TPRouteWorkspace | None:
    if workspace is None:
        return None

    m = hidden_states.shape[0]
    device = hidden_states.device

    if isinstance(workspace, TPMoEWorkspacePool):
        key = (
            device.index,
            num_experts,
            top_k,
            logits_dtype,
        )
        route_workspace = workspace.route_workspaces.get(key)
        needs_growth = (
            route_workspace is None
            or route_workspace.router_logits.shape[0] < m
            or route_workspace.router_logits.shape[1] != num_experts
            or route_workspace.topk_ids.shape[1] != top_k
            or route_workspace.router_logits.dtype != logits_dtype
            or route_workspace.router_logits.device != device
        )
        if needs_growth:
            if workspace.shared_arena is None:
                route_workspace = _alloc_route_workspace(
                    num_tokens=m,
                    num_experts=num_experts,
                    top_k=top_k,
                    device=device,
                    logits_dtype=logits_dtype,
                )
            else:
                if workspace.shared_arena.device != device:
                    raise ValueError(
                        f"MoE pool arena device {workspace.shared_arena.device} does not match hidden_states device {device}"
                    )
                route_workspace = _materialize_route_workspace(
                    workspace.shared_arena,
                    offset_bytes=0,
                    capacity_nbytes=workspace.route_workspace_nbytes,
                    num_tokens=m,
                    num_experts=num_experts,
                    top_k=top_k,
                    logits_dtype=logits_dtype,
                )
            workspace.route_workspaces[key] = route_workspace
        return _slice_route_workspace(route_workspace, m)

    route_workspace = workspace.route_workspace
    if (
        route_workspace is None
        or route_workspace.router_logits.shape != (m, num_experts)
        or route_workspace.topk_logits.shape != (m, top_k)
        or route_workspace.router_logits.dtype != logits_dtype
        or route_workspace.router_logits.device != device
    ):
        route_workspace = _alloc_route_workspace(
            num_tokens=m,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            logits_dtype=logits_dtype,
        )
        workspace.route_workspace = route_workspace
    return route_workspace


def _select_experts_reference(
    hidden_states: torch.Tensor,
    *,
    top_k: int,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    renormalize: bool = True,
    score_func: str = "softmax",
    correction_bias: torch.Tensor | None = None,
    image_correction_bias: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> B12XTopKRouting:
    """Reference routing selection for sparse-block MoE wrappers.

    Keep this path simple and obviously correct. Optimized routing should live
    in a separate public fast path rather than accreting special cases here.
    """

    if hidden_states.ndim != 2:
        raise ValueError(
            "expected hidden_states with rank 2, got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits is not None and gate_weight is not None:
        raise ValueError("pass either router_logits or gate_weight, not both")
    if router_logits is None and gate_weight is None:
        raise ValueError("expected router_logits or gate_weight")

    if router_logits is None:
        assert gate_weight is not None
        if gate_weight.ndim != 2:
            raise ValueError(
                f"expected gate_weight with rank 2, got shape {tuple(gate_weight.shape)}"
            )
        if gate_weight.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "gate_weight hidden-size mismatch: expected "
                f"{hidden_states.shape[1]}, got {gate_weight.shape[1]}"
            )
        if gate_bias is not None:
            if gate_bias.ndim != 1:
                raise ValueError(
                    f"expected gate_bias with rank 1, got shape {tuple(gate_bias.shape)}"
                )
            if gate_bias.shape[0] != gate_weight.shape[0]:
                raise ValueError(
                    "gate_bias expert mismatch: expected "
                    f"{gate_weight.shape[0]}, got {gate_bias.shape[0]}"
                )
        router_logits = F.linear(hidden_states, gate_weight, gate_bias)
    else:
        if router_logits.ndim != 2:
            raise ValueError(
                "expected router_logits with rank 2, got shape "
                f"{tuple(router_logits.shape)}"
            )
        if router_logits.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "router_logits batch mismatch: expected "
                f"{hidden_states.shape[0]}, got {router_logits.shape[0]}"
            )

    num_experts = router_logits.shape[1]
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    _validate_score_options(
        router_logits,
        score_func=score_func,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
    )
    logits_f32 = router_logits.float()
    if score_func == "sqrtsoftplus":
        scores = F.softplus(logits_f32).sqrt()
    elif correction_bias is not None or (
        image_correction_bias is not None and image_mask is not None
    ):
        scores = logits_f32.softmax(dim=-1)
    else:
        scores = logits_f32
    selection_scores = scores
    if correction_bias is not None:
        selection_scores = scores + correction_bias
    if image_correction_bias is not None and image_mask is not None:
        selection_scores = torch.where(
            image_mask[:, None], scores + image_correction_bias, selection_scores
        )
    # Stable descending sort on reversed experts matches the native larger-ID tie rule.
    reverse_ids = selection_scores.flip(-1).argsort(
        dim=-1, descending=True, stable=True
    )[:, :top_k]
    topk_ids = num_experts - 1 - reverse_ids
    if score_func == "sqrtsoftplus":
        topk_weights = scores.gather(1, topk_ids)
        if renormalize and top_k > 1:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
    else:
        topk_logits = logits_f32.gather(1, topk_ids)
        topk_weights = topk_logits.softmax(dim=-1) if renormalize else topk_logits
    topk_weights = topk_weights * routed_scaling_factor
    return B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )


def b12x_route_experts_fast(
    *, binding: TPMoERouteBinding, route_topk_launcher=None
) -> B12XTopKRouting:
    """Run top-k routing over the binding's caller-owned route buffers."""
    if not isinstance(binding, TPMoERouteBinding):
        raise TypeError("binding must be a TPMoERouteBinding")
    hidden_states = binding.hidden_states
    top_k = binding.top_k
    gate_weight = binding.gate_weight
    gate_bias = binding.gate_bias
    router_logits = binding.router_logits
    renormalize = binding.renormalize
    score_func = binding.score_func
    correction_bias = binding.correction_bias
    image_correction_bias = binding.image_correction_bias
    image_mask = binding.image_mask
    routed_scaling_factor = binding.routed_scaling_factor
    scratch = binding.scratch
    if hidden_states.ndim != 2:
        raise ValueError(
            "expected hidden_states with rank 2, got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if router_logits is not None and gate_weight is not None:
        raise ValueError("pass either router_logits or gate_weight, not both")
    if router_logits is None and gate_weight is None:
        raise ValueError("expected router_logits or gate_weight")

    if router_logits is None:
        assert gate_weight is not None
        if gate_weight.ndim != 2:
            raise ValueError(
                f"expected gate_weight with rank 2, got shape {tuple(gate_weight.shape)}"
            )
        if gate_weight.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "gate_weight hidden-size mismatch: expected "
                f"{hidden_states.shape[1]}, got {gate_weight.shape[1]}"
            )
        if gate_bias is not None:
            if gate_bias.ndim != 1:
                raise ValueError(
                    f"expected gate_bias with rank 1, got shape {tuple(gate_bias.shape)}"
                )
            if gate_bias.shape[0] != gate_weight.shape[0]:
                raise ValueError(
                    "gate_bias expert mismatch: expected "
                    f"{gate_weight.shape[0]}, got {gate_bias.shape[0]}"
                )
        num_experts = gate_weight.shape[0]
        logits_dtype = torch.result_type(hidden_states, gate_weight)
    else:
        if router_logits.ndim != 2:
            raise ValueError(
                "expected router_logits with rank 2, got shape "
                f"{tuple(router_logits.shape)}"
            )
        if router_logits.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "router_logits batch mismatch: expected "
                f"{hidden_states.shape[0]}, got {router_logits.shape[0]}"
            )
        num_experts = router_logits.shape[1]
        logits_dtype = router_logits.dtype

    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")

    if not hidden_states.is_cuda or num_experts > 1024:
        raise ValueError(
            "prepared native routing requires CUDA and at most 1024 experts"
        )

    route_workspace = _get_route_workspace(
        hidden_states,
        num_experts=num_experts,
        top_k=top_k,
        logits_dtype=logits_dtype,
        workspace=scratch,
    )
    if route_workspace is None:
        route_workspace = _alloc_route_workspace(
            num_tokens=hidden_states.shape[0],
            num_experts=num_experts,
            top_k=top_k,
            device=hidden_states.device,
            logits_dtype=logits_dtype,
        )

    if router_logits is None:
        assert gate_weight is not None
        torch.mm(hidden_states, gate_weight.t(), out=route_workspace.router_logits)
        if gate_bias is not None:
            route_workspace.router_logits.add_(
                gate_bias.to(route_workspace.router_logits.dtype)
            )
        router_logits = route_workspace.router_logits
    else:
        if not router_logits.is_contiguous():
            route_workspace.router_logits.copy_(router_logits)
            router_logits = route_workspace.router_logits

    triton_route_topk(
        router_logits,
        route_workspace.topk_logits,
        route_workspace.topk_ids,
        route_workspace.topk_weights,
        renormalize=renormalize,
        score_func=score_func,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
        routed_scaling_factor=routed_scaling_factor,
        _launcher=route_topk_launcher,
    )
    topk_ids = route_workspace.topk_ids
    topk_weights = route_workspace.topk_weights

    return B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
        flat_ids=topk_ids.view(-1),
        flat_weights=topk_weights.view(-1),
    )


def b12x_sparse_moe_fp4(
    *, binding: TPMoESparseFP4Binding, route_topk_launcher=None
) -> torch.Tensor | tuple[torch.Tensor, B12XTopKRouting]:
    """Execute a fully bound ``gate -> top-k -> routed experts`` block."""
    if not isinstance(binding, TPMoESparseFP4Binding):
        raise TypeError("binding must be a TPMoESparseFP4Binding")
    hidden_states = binding.hidden_states
    experts = binding.experts
    workspace = binding.scratch
    routing = binding.routing
    top_k = binding.top_k
    gate_weight = binding.gate_weight
    gate_bias = binding.gate_bias
    router_logits = binding.router_logits
    renormalize_topk = binding.renormalize_topk
    score_func = binding.score_func
    correction_bias = binding.correction_bias
    image_correction_bias = binding.image_correction_bias
    image_mask = binding.image_mask
    routed_scaling_factor = binding.routed_scaling_factor
    output = binding.output
    return_routing = binding.return_routing
    input_scales_static = binding.input_scales_static
    fast_math = binding.fast_math
    quant_mode = binding.quant_mode
    swiglu_limit = binding.swiglu_limit
    swiglu_alpha = binding.swiglu_alpha
    swiglu_beta = binding.swiglu_beta
    activation_amax = binding.activation_amax
    layer_idx = binding.layer_idx

    quant_mode_normalized = _select_prepared_quant_mode(
        experts,
        quant_mode,
    )

    if routing is not None:
        if (
            top_k is not None
            or gate_weight is not None
            or gate_bias is not None
            or router_logits is not None
            or score_func != "softmax"
            or correction_bias is not None
            or image_correction_bias is not None
            or image_mask is not None
        ):
            raise ValueError(
                "routing is mutually exclusive with gate, top-k, and score-selection arguments"
            )
        selected = routing
    else:
        if top_k is None:
            raise ValueError("top_k is required when routing is not provided")
        route_binding = build_tp_moe_route_binding(
            scratch=workspace,
            hidden_states=hidden_states,
            top_k=top_k,
            gate_weight=gate_weight,
            gate_bias=gate_bias,
            router_logits=router_logits,
            renormalize=renormalize_topk,
            score_func=score_func,
            correction_bias=correction_bias,
            image_correction_bias=image_correction_bias,
            image_mask=image_mask,
            routed_scaling_factor=routed_scaling_factor,
        )
        selected = b12x_route_experts_fast(
            binding=route_binding,
            route_topk_launcher=route_topk_launcher,
        )

    _validate_sparse_routing(hidden_states, selected)

    moe_binding = build_tp_moe_fp4_binding(
        scratch=workspace,
        a=hidden_states,
        experts=experts,
        topk_weights=selected.topk_weights,
        topk_ids=selected.topk_ids,
        output=output,
        input_scales_static=input_scales_static,
        fast_math=fast_math,
        quant_mode=quant_mode_normalized,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
    )
    routed_output = b12x_moe_fp4(binding=moe_binding)
    if routed_scaling_factor != 1.0:
        routed_output.mul_(routed_scaling_factor)
    if return_routing:
        return routed_output, selected
    return routed_output
