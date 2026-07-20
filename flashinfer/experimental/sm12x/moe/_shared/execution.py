# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/execution.py @ 7f72b398 (2026-07-07) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Canonical MoE semantics and execution-regime descriptors.

The public ``quant_mode`` names in :mod:`flashinfer.experimental.sm12x.moe.fused_moe._impl` historically
identify several independent choices at once: activation encoding, weight-scale
encoding, route materialization, work scheduling, graph fusion, and prepared
weight layout.  That makes genuinely orthogonal implementations look like
unrelated kernels.

This module names those axes independently.  It deliberately contains no CUDA,
Torch, or CuTe code so that planning, diagnostics, and tests can reason about a
MoE implementation without importing a kernel lowering.

The central scheduling distinction is ``WorkAvailability``:

* ``INLINE`` work is decoded directly from top-k routing by a tiny kernel.
* ``PRECOMPUTED`` work has been materialized before its compute phase (either
  by an earlier launch or by an earlier globally synchronized phase of the
  same kernel) and can therefore use either arithmetic grid assignment or a
  work-stealing queue without readiness checks.
* ``STREAMING`` work is published while the compute kernel is already resident;
  a readiness-aware source (currently the append-only ready queue) is required.

Thus readiness, load balancing, and arithmetic precision are separate axes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class OperandEncoding(_StringEnum):
    BF16 = "bf16"
    FP4_E2M1 = "fp4_e2m1"
    MXFP8_E4M3 = "mxfp8_e4m3"


class ScaleEncoding(_StringEnum):
    E4M3_K16 = "e4m3_k16"
    E8M0_K32 = "e8m0_k32"
    E8M0_K32_X_E4M3_K16_RESIDUAL = "e8m0_k32_x_e4m3_k16_residual"


class MoERegime(_StringEnum):
    """Graph-level implementation regime, intentionally precision-neutral."""

    DIRECT = "direct"
    STREAMING_FUSED = "streaming_fused"
    MATERIALIZED_FUSED = "materialized_fused"
    MATERIALIZED_PERSISTENT = "materialized_persistent"


class RouteLayout(_StringEnum):
    DIRECT_TOPK = "direct_topk"
    APPEND_ONLY_EXPERT = "append_only_expert"
    SORTED_PADDED_EXPERT = "sorted_padded_expert"


class WorkAvailability(_StringEnum):
    INLINE = "inline"
    PRECOMPUTED = "precomputed"
    STREAMING = "streaming"


class WorkScheduler(_StringEnum):
    DIRECT = "direct"
    PERSISTENT_GRID = "persistent_grid"
    ATOMIC_QUEUE = "atomic_queue"
    READY_QUEUE = "ready_queue"


class GraphPartition(_StringEnum):
    """Partitioning of FC1 -> activation/quantization -> FC2.

    Route materialization is represented independently by
    :class:`WorkAvailability`.
    """

    FUSED = "fused"


class GemmEngine(_StringEnum):
    DIRECT_DOT = "direct_dot"
    NVFP4_MMA = "nvfp4_mma"
    MXFP8_QMMA = "mxfp8_qmma"
    W4A16_MMA = "w4a16_mma"


class PreparedWeightLayout(_StringEnum):
    SOURCE_NATIVE = "source_native"
    MMA_VIEW = "mma_view"
    MMA_PACKED = "mma_packed"
    QMMA_REPACKED = "qmma_repacked"


class PreparedScaleLayout(_StringEnum):
    """Physical scale/metadata forms made available at model load time."""

    SOURCE_NATIVE = "source_native"
    RUNTIME_ALPHA = "runtime_alpha"
    MMA_PACKED = "mma_packed"
    QMMA_SFB = "qmma_sfb"


class WeightPreparationTransform(_StringEnum):
    """Concrete one-time transforms executed by the tensor integration layer."""

    RUNTIME_ALPHAS = "runtime_alphas"
    W4A16_NATIVE = "w4a16_native"
    W4A16_PACKED = "w4a16_packed"
    W4A8_QMMA = "w4a8_qmma"


class WeightStoragePolicy(_StringEnum):
    """Ownership policy for checkpoint storage after preparation.

    ``KEEP_SOURCE`` leaves source tensors as the canonical runtime storage.
    ``TRANSFER_SOURCE`` keeps their bytes unchanged but transfers ownership to
    a prepared representation, allowing checkpoint parameter handles to go
    away. ``REUSE_SOURCE`` transforms those bytes in-place and also transfers
    ownership. There is intentionally no policy that keeps a source allocation
    plus a second model-sized repack: incompatible representation requests are
    planning errors.
    """

    KEEP_SOURCE = "keep_source"
    TRANSFER_SOURCE = "transfer_source"
    REUSE_SOURCE = "reuse_source"


class OutputReduction(_StringEnum):
    FUSED_TOPK_SUM = "fused_topk_sum"
    ATOMIC_SCATTER = "atomic_scatter"
    ROUTE_BUFFER_TOPK_SUM = "route_buffer_topk_sum"
    SEPARATE_TOPK_SUM = "separate_topk_sum"


_QUANT_MODES = {"nvfp4", "w4a16", "w4a8_mx", "w4a8_nvfp4"}
_SOURCE_FORMATS = {
    "modelopt_nvfp4",
    "fp4_e8m0_k32",
    "compressed_tensors",
}
_SOURCES_BY_QUANT_MODE = {
    "nvfp4": frozenset({"modelopt_nvfp4"}),
    "w4a8_nvfp4": frozenset({"modelopt_nvfp4"}),
    "w4a8_mx": frozenset({"fp4_e8m0_k32"}),
    "w4a16": frozenset(_SOURCE_FORMATS),
}


def validate_moe_source_quant(*, source_format: str, quant_mode: str) -> None:
    """Validate the canonical checkpoint-format/numeric-recipe pairing."""

    quant_mode = str(quant_mode).lower()
    source_format = str(source_format).lower()
    if quant_mode not in _QUANT_MODES:
        raise ValueError(f"unsupported quant_mode {quant_mode!r}")
    if source_format not in _SOURCE_FORMATS:
        raise ValueError(f"unsupported source_format {source_format!r}")
    if source_format not in _SOURCES_BY_QUANT_MODE[quant_mode]:
        raise ValueError(
            f"source_format={source_format!r} is incompatible with "
            f"quant_mode={quant_mode!r}"
        )


@dataclass(frozen=True, kw_only=True)
class MoESpec:
    """Semantic/numeric contract before choosing a kernel regime.

    ``source_weight_scale`` describes checkpoint storage.  ``weight_scale``
    describes the representation consumed by the selected arithmetic recipe;
    they differ for W4A8-on-NVFP4, whose K/16 E4M3 scales are decomposed into
    a K/32 E8M0 grid and a K/16 residual.
    """

    quant_mode: str
    source_format: str
    activation: str
    io_dtype: str
    activation_encoding: OperandEncoding
    activation_scale: ScaleEncoding | None
    weight_encoding: OperandEncoding
    source_weight_scale: ScaleEncoding
    weight_scale: ScaleEncoding
    w13_layout: str
    apply_router_weight_on_input: bool = False
    accumulator_dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "quant_mode", str(self.quant_mode).lower())
        object.__setattr__(self, "source_format", str(self.source_format).lower())
        object.__setattr__(
            self,
            "activation_encoding",
            OperandEncoding(self.activation_encoding),
        )
        object.__setattr__(
            self,
            "activation_scale",
            None
            if self.activation_scale is None
            else ScaleEncoding(self.activation_scale),
        )
        object.__setattr__(
            self,
            "weight_encoding",
            OperandEncoding(self.weight_encoding),
        )
        object.__setattr__(
            self,
            "source_weight_scale",
            ScaleEncoding(self.source_weight_scale),
        )
        object.__setattr__(
            self,
            "weight_scale",
            ScaleEncoding(self.weight_scale),
        )
        if self.quant_mode not in _QUANT_MODES:
            raise ValueError(f"unsupported quant_mode {self.quant_mode!r}")
        if self.source_format not in _SOURCE_FORMATS:
            raise ValueError(f"unsupported source_format {self.source_format!r}")
        validate_moe_source_quant(
            source_format=self.source_format,
            quant_mode=self.quant_mode,
        )


@dataclass(frozen=True, kw_only=True)
class MoEWeightPreparationPlan:
    """Load-time representations required by a set of runtime MoE recipes.

    The plan is deliberately tensor-free.  Kernel planning owns the choice of
    representation; an integration layer merely executes ``transforms`` and
    applies ``storage_policy`` before CUDA graph capture.
    """

    specs: tuple[MoESpec, ...]
    num_experts: int
    hidden_size: int
    intermediate_size: int
    transforms: frozenset[WeightPreparationTransform]
    weight_layouts: frozenset[PreparedWeightLayout]
    scale_layouts: frozenset[PreparedScaleLayout]
    storage_policy: WeightStoragePolicy

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("weight preparation requires at least one MoE spec")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "num_experts", int(self.num_experts))
        object.__setattr__(self, "hidden_size", int(self.hidden_size))
        object.__setattr__(self, "intermediate_size", int(self.intermediate_size))
        object.__setattr__(
            self,
            "transforms",
            frozenset(WeightPreparationTransform(value) for value in self.transforms),
        )
        object.__setattr__(
            self,
            "weight_layouts",
            frozenset(PreparedWeightLayout(value) for value in self.weight_layouts),
        )
        object.__setattr__(
            self,
            "scale_layouts",
            frozenset(PreparedScaleLayout(value) for value in self.scale_layouts),
        )
        object.__setattr__(
            self, "storage_policy", WeightStoragePolicy(self.storage_policy)
        )
        for name in ("num_experts", "hidden_size", "intermediate_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        contract = (
            specs[0].source_format,
            specs[0].activation,
            specs[0].io_dtype,
            specs[0].w13_layout,
        )
        seen_modes: set[str] = set()
        for spec in specs:
            actual = (
                spec.source_format,
                spec.activation,
                spec.io_dtype,
                spec.w13_layout,
            )
            if actual != contract:
                raise ValueError(
                    "all weight-preparation specs must share source format, "
                    "activation, dtype, and W13 layout"
                )
            if spec.quant_mode in seen_modes:
                raise ValueError(
                    f"duplicate quant_mode in weight preparation: {spec.quant_mode!r}"
                )
            seen_modes.add(spec.quant_mode)
        if not self.transforms:
            raise ValueError("weight preparation plan has no transforms")
        if not self.weight_layouts:
            raise ValueError("weight preparation plan has no weight layouts")

    @property
    def source_format(self) -> str:
        return self.specs[0].source_format

    @property
    def activation(self) -> str:
        return self.specs[0].activation

    @property
    def io_dtype(self) -> str:
        return self.specs[0].io_dtype

    @property
    def w13_layout(self) -> str:
        return self.specs[0].w13_layout

    @property
    def quant_modes(self) -> frozenset[str]:
        return frozenset(spec.quant_mode for spec in self.specs)

    @property
    def prepares_runtime_alphas(self) -> bool:
        return WeightPreparationTransform.RUNTIME_ALPHAS in self.transforms

    @property
    def discards_source_parameters(self) -> bool:
        return self.storage_policy in {
            WeightStoragePolicy.TRANSFER_SOURCE,
            WeightStoragePolicy.REUSE_SOURCE,
        }

    @property
    def reuses_source_storage(self) -> bool:
        return self.storage_policy is WeightStoragePolicy.REUSE_SOURCE

    @property
    def w4a16_weight_layout(self) -> str | None:
        if WeightPreparationTransform.W4A16_NATIVE in self.transforms:
            return "modelopt"
        if WeightPreparationTransform.W4A16_PACKED in self.transforms:
            return "packed"
        return None

    @property
    def w4a16_scale_format(self) -> str | None:
        if self.w4a16_weight_layout is None:
            return None
        return self.specs[0].source_weight_scale.value

    def required_weight_layout(self, quant_mode: str) -> PreparedWeightLayout | None:
        """Return a fixed load-time layout for ``quant_mode``, if it has one.

        Source-native NVFP4 recipes deliberately return ``None`` because their
        direct and tensor-core regimes consume two zero-copy views of the same
        allocation.  The execution lowering chooses between those views.
        """

        quant_mode = str(quant_mode).lower()
        if quant_mode not in self.quant_modes:
            raise ValueError(
                f"quant_mode={quant_mode!r} is absent from this preparation plan"
            )
        if quant_mode == "w4a16":
            return (
                PreparedWeightLayout.SOURCE_NATIVE
                if WeightPreparationTransform.W4A16_NATIVE in self.transforms
                else PreparedWeightLayout.MMA_PACKED
            )
        if quant_mode == "w4a8_mx":
            return PreparedWeightLayout.QMMA_REPACKED
        return None

    def supports(
        self,
        *,
        quant_mode: str,
        execution: "MoEExecutionPlan",
    ) -> bool:
        quant_mode = str(quant_mode).lower()
        if quant_mode not in self.quant_modes:
            return False
        required = self.required_weight_layout(quant_mode)
        if required is not None:
            return execution.weight_layout is required
        return execution.weight_layout in {
            PreparedWeightLayout.SOURCE_NATIVE,
            PreparedWeightLayout.MMA_VIEW,
        }


@dataclass(frozen=True, kw_only=True)
class MoEExecutionPlan:
    """A lowering of :class:`MoESpec` onto one concrete execution regime."""

    regime: MoERegime
    route_layout: RouteLayout
    work_availability: WorkAvailability
    scheduler: WorkScheduler
    graph_partition: GraphPartition
    gemm_engine: GemmEngine
    weight_layout: PreparedWeightLayout
    reduction: OutputReduction
    tile_m: int | None = None
    tile_n: int | None = None
    route_block_rows: int | None = None

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("regime", MoERegime),
            ("route_layout", RouteLayout),
            ("work_availability", WorkAvailability),
            ("scheduler", WorkScheduler),
            ("graph_partition", GraphPartition),
            ("gemm_engine", GemmEngine),
            ("weight_layout", PreparedWeightLayout),
            ("reduction", OutputReduction),
        ):
            object.__setattr__(self, name, enum_type(getattr(self, name)))
        if self.work_availability is WorkAvailability.STREAMING:
            if self.scheduler is not WorkScheduler.READY_QUEUE:
                raise ValueError("streaming work requires a readiness-aware scheduler")
        elif self.scheduler is WorkScheduler.READY_QUEUE:
            raise ValueError("ready_queue is only valid for streaming work")
        if self.scheduler is WorkScheduler.ATOMIC_QUEUE:
            if self.work_availability is not WorkAvailability.PRECOMPUTED:
                raise ValueError("atomic_queue is only valid for precomputed work")
        if self.route_layout is RouteLayout.APPEND_ONLY_EXPERT:
            if self.work_availability not in {
                WorkAvailability.PRECOMPUTED,
                WorkAvailability.STREAMING,
            }:
                raise ValueError(
                    "append-only expert routing must materialize or stream work"
                )
        if self.route_layout is RouteLayout.SORTED_PADDED_EXPERT:
            if self.work_availability is not WorkAvailability.PRECOMPUTED:
                raise ValueError(
                    "sorted/padded routes must be materialized before compute"
                )
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("route_block_rows", self.route_block_rows),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def work_is_streamed(self) -> bool:
        return self.work_availability is WorkAvailability.STREAMING

    @property
    def grid_addressable(self) -> bool:
        """Whether all compute work is knowable without waiting in-kernel."""

        return self.work_availability in {
            WorkAvailability.INLINE,
            WorkAvailability.PRECOMPUTED,
        }

    @property
    def uses_explicit_task_queue(self) -> bool:
        return self.scheduler in {
            WorkScheduler.ATOMIC_QUEUE,
            WorkScheduler.READY_QUEUE,
        }


def make_moe_spec(
    *,
    quant_mode: str,
    source_format: str,
    activation: str,
    io_dtype: str,
    w13_layout: str,
    apply_router_weight_on_input: bool = False,
) -> MoESpec:
    """Build the canonical numeric contract for a normalized public request."""

    quant_mode = str(quant_mode).lower()
    source_format = str(source_format).lower()
    validate_moe_source_quant(
        source_format=source_format,
        quant_mode=quant_mode,
    )

    source_scale = (
        ScaleEncoding.E8M0_K32
        if source_format == "fp4_e8m0_k32"
        else ScaleEncoding.E4M3_K16
    )
    if quant_mode == "w4a16":
        activation_encoding = OperandEncoding.BF16
        activation_scale = None
        weight_scale = source_scale
    elif quant_mode == "nvfp4":
        activation_encoding = OperandEncoding.FP4_E2M1
        activation_scale = ScaleEncoding.E4M3_K16
        weight_scale = ScaleEncoding.E4M3_K16
    elif quant_mode == "w4a8_mx":
        activation_encoding = OperandEncoding.MXFP8_E4M3
        activation_scale = ScaleEncoding.E8M0_K32
        weight_scale = ScaleEncoding.E8M0_K32
    else:
        activation_encoding = OperandEncoding.MXFP8_E4M3
        activation_scale = ScaleEncoding.E8M0_K32
        weight_scale = ScaleEncoding.E8M0_K32_X_E4M3_K16_RESIDUAL

    return MoESpec(
        quant_mode=quant_mode,
        source_format=source_format,
        activation=str(activation),
        io_dtype=str(io_dtype),
        activation_encoding=activation_encoding,
        activation_scale=activation_scale,
        weight_encoding=OperandEncoding.FP4_E2M1,
        source_weight_scale=source_scale,
        weight_scale=weight_scale,
        w13_layout=str(w13_layout),
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
    )


def plan_moe_weight_preparation(
    specs: MoESpec | Iterable[MoESpec],
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    w4a16_layout: PreparedWeightLayout | str | None = None,
) -> MoEWeightPreparationPlan:
    """Choose the minimal representation set for the requested recipes.

    W4A16's automatic policy mirrors the production serving choices: native
    ModelOpt storage avoids a second full copy, compact E8M0 K tails stay native,
    and aligned E8M0/CompressedTensors sources use the packed MMA layout.  An
    explicit ``w4a16_layout`` is a development/deployment override, not an
    adapter-side reimplementation of the policy.
    """

    if isinstance(specs, MoESpec):
        normalized_specs = (specs,)
    else:
        normalized_specs = tuple(specs)
    if not normalized_specs:
        raise ValueError("weight preparation requires at least one MoE spec")
    num_experts = int(num_experts)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    for name, value in (
        ("num_experts", num_experts),
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    source_format = normalized_specs[0].source_format
    requested_w4a16_layout = (
        None if w4a16_layout is None else PreparedWeightLayout(w4a16_layout)
    )
    if requested_w4a16_layout not in {
        None,
        PreparedWeightLayout.SOURCE_NATIVE,
        PreparedWeightLayout.MMA_PACKED,
    }:
        raise ValueError(
            "W4A16 preparation requires source_native or mma_packed layout"
        )

    transforms: set[WeightPreparationTransform] = set()
    weight_layouts: set[PreparedWeightLayout] = set()
    scale_layouts: set[PreparedScaleLayout] = set()
    for spec in normalized_specs:
        if spec.source_format != source_format:
            raise ValueError("all preparation specs must share one source format")
        if spec.quant_mode in {"nvfp4", "w4a8_nvfp4"}:
            transforms.add(WeightPreparationTransform.RUNTIME_ALPHAS)
            # One source allocation supplies both direct-micro and dynamic MMA
            # views; these are execution contracts, not duplicate byte copies.
            weight_layouts.update(
                {PreparedWeightLayout.SOURCE_NATIVE, PreparedWeightLayout.MMA_VIEW}
            )
            scale_layouts.update(
                {
                    PreparedScaleLayout.SOURCE_NATIVE,
                    PreparedScaleLayout.RUNTIME_ALPHA,
                }
            )
            continue
        if spec.quant_mode == "w4a8_mx":
            if spec.activation != "silu":
                raise ValueError("W4A8-MX preparation currently requires silu")
            # The rp storage ceil-tiles partial 256/128 tiles (zero-filled),
            # so 32-aligned shards (352 = 2048/TP6, 192 = 3072/TP16) prepare
            # fine; consumers bound their reads by the logical sizes.
            if hidden_size % 256 != 0 or intermediate_size % 32 != 0:
                raise ValueError(
                    "W4A8-MX QMMA layout requires hidden_size % 256 == 0 and "
                    "intermediate_size % 32 == 0"
                )
            transforms.add(WeightPreparationTransform.W4A8_QMMA)
            weight_layouts.add(PreparedWeightLayout.QMMA_REPACKED)
            scale_layouts.add(PreparedScaleLayout.QMMA_SFB)
            continue
        if spec.quant_mode == "w4a16":
            layout = requested_w4a16_layout
            if layout is None:
                # The packed MMA kernels only have tile configs for
                # 128-aligned intermediate shards; the native path already
                # supports compact sub-32 tails (see
                # test_w4a16_e8m0_native_compact_tail_uses_ceil_scale_grid),
                # so route every non-128-aligned e8m0 shard through it
                # instead of failing at kernel-config time (e.g. 2048/TP6 =
                # 352, 3072/TP16 = 192).
                layout = (
                    PreparedWeightLayout.SOURCE_NATIVE
                    if source_format == "modelopt_nvfp4"
                    or (
                        source_format == "fp4_e8m0_k32" and intermediate_size % 128 != 0
                    )
                    else PreparedWeightLayout.MMA_PACKED
                )
            if (
                source_format == "compressed_tensors"
                and layout is not PreparedWeightLayout.MMA_PACKED
            ):
                raise ValueError(
                    "CompressedTensors W4A16 requires the packed MMA layout"
                )
            transforms.add(
                WeightPreparationTransform.W4A16_NATIVE
                if layout is PreparedWeightLayout.SOURCE_NATIVE
                else WeightPreparationTransform.W4A16_PACKED
            )
            weight_layouts.add(layout)
            scale_layouts.add(PreparedScaleLayout.MMA_PACKED)
            continue
        raise ValueError(f"unsupported quant_mode {spec.quant_mode!r}")

    mutating = transforms & {
        WeightPreparationTransform.W4A16_PACKED,
        WeightPreparationTransform.W4A8_QMMA,
    }
    source_recipe_selected = bool(
        {"nvfp4", "w4a8_nvfp4"} & {spec.quant_mode for spec in normalized_specs}
    )
    native_representation = WeightPreparationTransform.W4A16_NATIVE in transforms
    if len(mutating) > 1:
        raise ValueError(
            "requested MoE recipes require multiple incompatible model-sized "
            "weight repacks; keeping both is not a supported serving policy"
        )
    if mutating and (source_recipe_selected or native_representation):
        raise ValueError(
            "requested MoE recipes require both source-native weights and a "
            "model-sized repack; choose a shared/native regime instead"
        )
    if source_recipe_selected:
        storage_policy = WeightStoragePolicy.KEEP_SOURCE
    elif native_representation:
        storage_policy = WeightStoragePolicy.TRANSFER_SOURCE
    elif mutating:
        storage_policy = WeightStoragePolicy.REUSE_SOURCE
    else:
        storage_policy = WeightStoragePolicy.KEEP_SOURCE

    return MoEWeightPreparationPlan(
        specs=normalized_specs,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        transforms=frozenset(transforms),
        weight_layouts=frozenset(weight_layouts),
        scale_layouts=frozenset(scale_layouts),
        storage_policy=storage_policy,
    )


def _gemm_engine_for_spec(spec: MoESpec) -> GemmEngine:
    if spec.activation_encoding is OperandEncoding.BF16:
        return GemmEngine.W4A16_MMA
    if spec.activation_encoding is OperandEncoding.MXFP8_E4M3:
        return GemmEngine.MXFP8_QMMA
    return GemmEngine.NVFP4_MMA


def lower_moe_execution(
    spec: MoESpec,
    *,
    regime: MoERegime,
    tile_m: int | None = None,
    tile_n: int | None = None,
    route_block_rows: int | None = None,
    direct_routes: bool = False,
    deterministic_output: bool = False,
    scheduler: WorkScheduler | None = None,
    required_weight_layout: PreparedWeightLayout | str | None = None,
) -> MoEExecutionPlan:
    """Lower one semantic spec without coupling regime to precision.

    Activation precision does not select the scheduler: an MXFP8 activation
    spec can lower to either queue or persistent-grid ownership in the unified
    materialized kernel.
    """

    regime = MoERegime(regime)
    required_weight_layout = (
        None
        if required_weight_layout is None
        else PreparedWeightLayout(required_weight_layout)
    )
    if regime is MoERegime.DIRECT:
        return MoEExecutionPlan(
            regime=regime,
            route_layout=RouteLayout.DIRECT_TOPK,
            work_availability=WorkAvailability.INLINE,
            scheduler=WorkScheduler.DIRECT,
            graph_partition=GraphPartition.FUSED,
            gemm_engine=GemmEngine.DIRECT_DOT,
            weight_layout=(
                required_weight_layout or PreparedWeightLayout.SOURCE_NATIVE
            ),
            reduction=OutputReduction.FUSED_TOPK_SUM,
            tile_m=tile_m,
            tile_n=tile_n,
        )

    if regime is MoERegime.STREAMING_FUSED:
        engine = _gemm_engine_for_spec(spec)
        reduction = (
            OutputReduction.ROUTE_BUFFER_TOPK_SUM
            if deterministic_output
            else OutputReduction.ATOMIC_SCATTER
        )
        return MoEExecutionPlan(
            regime=regime,
            route_layout=RouteLayout.APPEND_ONLY_EXPERT,
            work_availability=WorkAvailability.STREAMING,
            scheduler=WorkScheduler.READY_QUEUE,
            graph_partition=GraphPartition.FUSED,
            gemm_engine=engine,
            weight_layout=required_weight_layout or PreparedWeightLayout.MMA_VIEW,
            reduction=reduction,
            tile_m=tile_m,
            tile_n=tile_n,
        )

    if regime is MoERegime.MATERIALIZED_FUSED:
        engine = _gemm_engine_for_spec(spec)
        resolved_scheduler = (
            WorkScheduler.ATOMIC_QUEUE
            if scheduler is None
            else WorkScheduler(scheduler)
        )
        if resolved_scheduler not in {
            WorkScheduler.PERSISTENT_GRID,
            WorkScheduler.ATOMIC_QUEUE,
        }:
            raise ValueError(
                "materialized_fused requires persistent_grid or atomic_queue"
            )
        reduction = (
            OutputReduction.ROUTE_BUFFER_TOPK_SUM
            if deterministic_output
            else OutputReduction.ATOMIC_SCATTER
        )
        weight_layout = required_weight_layout or (
            PreparedWeightLayout.QMMA_REPACKED
            if spec.quant_mode == "w4a8_mx"
            else PreparedWeightLayout.MMA_VIEW
        )
        return MoEExecutionPlan(
            regime=regime,
            route_layout=RouteLayout.APPEND_ONLY_EXPERT,
            work_availability=WorkAvailability.PRECOMPUTED,
            scheduler=resolved_scheduler,
            graph_partition=GraphPartition.FUSED,
            gemm_engine=engine,
            weight_layout=weight_layout,
            reduction=reduction,
            tile_m=tile_m,
            tile_n=tile_n,
        )

    if regime is MoERegime.MATERIALIZED_PERSISTENT:
        engine = _gemm_engine_for_spec(spec)
        return MoEExecutionPlan(
            regime=regime,
            route_layout=(
                RouteLayout.DIRECT_TOPK
                if direct_routes
                else RouteLayout.SORTED_PADDED_EXPERT
            ),
            work_availability=WorkAvailability.PRECOMPUTED,
            scheduler=WorkScheduler.PERSISTENT_GRID,
            graph_partition=GraphPartition.FUSED,
            gemm_engine=engine,
            weight_layout=required_weight_layout or PreparedWeightLayout.MMA_PACKED,
            reduction=OutputReduction.SEPARATE_TOPK_SUM,
            tile_m=tile_m,
            tile_n=tile_n,
            route_block_rows=route_block_rows,
        )

    raise ValueError(f"unsupported MoE execution regime {regime!s}")


__all__ = [
    "GemmEngine",
    "GraphPartition",
    "MoEExecutionPlan",
    "MoERegime",
    "MoESpec",
    "OperandEncoding",
    "OutputReduction",
    "PreparedScaleLayout",
    "PreparedWeightLayout",
    "RouteLayout",
    "ScaleEncoding",
    "WorkAvailability",
    "WeightPreparationTransform",
    "WeightStoragePolicy",
    "WorkScheduler",
    "lower_moe_execution",
    "make_moe_spec",
    "MoEWeightPreparationPlan",
    "plan_moe_weight_preparation",
    "validate_moe_source_quant",
]
