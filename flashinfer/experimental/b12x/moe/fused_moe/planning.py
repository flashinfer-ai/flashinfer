"""Canonical fused-MoE weight planning and preparation."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS

from dataclasses import dataclass
from enum import Enum

import torch

from .._shared.execution import MoEWeightPreparationPlan
from .._shared.kernels.activations import is_gated_moe_activation
from ._impl import (
    plan_b12x_fp4_moe_weights,
    prepare_b12x_fp4_moe_weights,
    prepare_b12x_trellis_v2_weights,
    prepare_b12x_iq2_xs_weights,
)
from .config import TrellisConfig
from .source import PackedSource, TrellisSource, WeightSource
from .trellis_layout import TrellisStaging
from .weights import (
    PackedWeights,
    IQ2XSWeights,
    PreparedExperts,
    PreparedWeightFormat,
    ScaleEncoding,
    TrellisWeights,
    WeightEncoding,
    WeightPacking,
)


class ActivationMode(str, Enum):
    """Numeric activation contract at the fused GEMM boundaries."""

    A16 = "a16"
    A8 = "a8"
    A4 = "a4"
    AUTO = "auto"


@dataclass(frozen=True, kw_only=True)
class ActivationSpec:
    """Activation precision, nonlinearity, and public I/O dtype.

    ``rotation_dtype`` selects the internal full-rotation trellis arithmetic,
    independently of BF16/FP16 public inputs. None retains the I/O dtype.
    Only the FP16 full-rotation implementation is supported explicitly.

    ``a16_max_tokens`` forces A16 up to an inclusive token capacity; zero
    disables the constraint. Larger calls retain ``mode``.
    """

    mode: ActivationMode
    nonlinearity: str
    io_dtype: torch.dtype
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    rotation_dtype: torch.dtype | None = None
    a16_max_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ActivationMode(self.mode))
        object.__setattr__(self, "nonlinearity", str(self.nonlinearity).lower())
        if self.io_dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError("io_dtype must be torch.bfloat16 or torch.float16")
        if self.rotation_dtype not in (None, torch.float16):
            raise TypeError("rotation_dtype must be torch.float16 or None")
        if type(self.a16_max_tokens) is not int or self.a16_max_tokens < 0:
            raise ValueError("a16_max_tokens must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class MoEGeometry:
    """Shape of one tensor-parallel expert shard."""

    num_experts: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, kw_only=True)
class WeightPlanConstraints:
    """Integration requirements that constrain B12X preparation policy."""

    required_packing: WeightPacking | None = None

    def __post_init__(self) -> None:
        if self.required_packing is not None:
            object.__setattr__(
                self,
                "required_packing",
                WeightPacking(self.required_packing),
            )


@dataclass(frozen=True, kw_only=True)
class WeightPlan:
    """Canonical load-time plan with no public execution-recipe names."""

    source: WeightSource
    activation: ActivationSpec
    geometry: MoEGeometry
    prepared_format: PreparedWeightFormat
    _impl: MoEWeightPreparationPlan


def _packed_recipe(source: PackedSource, mode: ActivationMode) -> str:
    source_format = source.format.value
    if mode is ActivationMode.A16:
        if source_format == "mxfp6_e2m3":
            raise ValueError("MXFP6 weights require A8 activations")
        return "w4a16"
    if mode is ActivationMode.A4:
        if source_format != "modelopt_nvfp4":
            raise ValueError("A4 activations require ModelOpt NVFP4 weights")
        return "nvfp4"
    try:
        return {
            "fp4_e8m0_k32": "w4a8_mx",
            "modelopt_nvfp4": "w4a8_nvfp4",
            "mxfp6_e2m3": "w6a8_mx",
        }[source_format]
    except KeyError as exc:
        raise ValueError(
            f"source format {source_format!r} does not support A8 activations"
        ) from exc


def _validate_trellis_runtime(source: TrellisConfig, *, explicit_uniform: bool = False) -> None:
    if source.codebook.value == "lut_fp16" and not explicit_uniform:
        raise NotImplementedError(
            "lut_fp16 preparation requires an explicit uniform_bits declaration"
        )
    if source.rate.group_size is not None:
        raise NotImplementedError(
            "grouped trellis rates are defined by the checkpoint schema but "
            "are not implemented by the fused MoE runtime"
        )
    if source.transform.projection.kind != "scaled_hadamard" or (
        source.transform.projection.block_size != 128
    ):
        raise NotImplementedError(
            "fused MoE trellis execution requires scaled_hadamard(128)"
        )
    expert = source.transform.expert
    if expert.kind == "intermediate_hadamard" and (
        source.codebook.value != "lut_e4m3"
        or source.rate.granularity.value != "uniform"
    ):
        raise NotImplementedError(
            "fused MoE intermediate_hadamard execution currently requires the "
            "lut_e4m3 codebook with uniform rates"
        )
    if expert.kind == "intermediate_hadamard" and (
        expert.pre_block_size,
        expert.post_block_size,
    ) != (512, 128):
        raise NotImplementedError(
            "fused MoE intermediate_hadamard requires block sizes (512, 128)"
        )


def _prepared_format(
    *,
    source: WeightSource,
    plan: MoEWeightPreparationPlan,
    recipe: str,
    constraints: WeightPlanConstraints,
) -> PreparedWeightFormat:
    available = frozenset(WeightPacking(layout.value) for layout in plan.weight_layouts)
    required = plan.required_weight_layout(recipe)
    default_packing = (
        WeightPacking.SOURCE_NATIVE
        if required is None
        else WeightPacking(required.value)
    )
    packing = constraints.required_packing or default_packing
    if packing not in available:
        raise ValueError(
            f"required packing {packing.value!r} is not available; "
            f"planner produced {sorted(value.value for value in available)}"
        )
    if isinstance(source, TrellisSource):
        weights = WeightEncoding.TRELLIS
        scales = ScaleEncoding.TRELLIS_SCALES
    else:
        weights = (
            WeightEncoding(source.format.value) if source.format.value in BLOCK_CODECS else
            WeightEncoding.FP6_E2M3
            if source.format.value == "mxfp6_e2m3"
            else WeightEncoding.FP4_E2M1
        )
        scales = ScaleEncoding(plan.specs[0].weight_scale.value)
    return PreparedWeightFormat(
        weights=weights,
        scales=scales,
        packing=packing,
        available_packings=available,
    )


def plan_weights(
    *,
    source: WeightSource,
    activation: ActivationSpec,
    geometry: MoEGeometry,
    constraints: WeightPlanConstraints | None = None,
) -> WeightPlan:
    """Plan preparation from independent source, activation, and shape axes."""

    if not isinstance(activation, ActivationSpec):
        raise TypeError("activation must be an ActivationSpec")
    if not isinstance(geometry, MoEGeometry):
        raise TypeError("geometry must be a MoEGeometry")
    constraints = constraints or WeightPlanConstraints()
    if not isinstance(constraints, WeightPlanConstraints):
        raise TypeError("constraints must be WeightPlanConstraints")
    if isinstance(source, TrellisConfig):
        source = TrellisSource(config=source)
    if activation.rotation_dtype is not None and not isinstance(source, TrellisSource):
        raise ValueError("rotation_dtype is only valid for trellis weights")

    if isinstance(source, PackedSource):
        automatic = activation.mode is ActivationMode.AUTO
        if activation.a16_max_tokens and (
            source.format.value != "modelopt_nvfp4" or activation.io_dtype is not torch.bfloat16
        ):
            raise ValueError("A16 token cutoff requires BF16 inputs and ModelOpt NVFP4 weights")
        if automatic and (
            source.format.value != "modelopt_nvfp4"
            or activation.io_dtype is not torch.bfloat16
            or activation.nonlinearity != "silu"
        ):
            raise ValueError("automatic MoE precision requires BF16 inputs, SiLU, and ModelOpt NVFP4 weights")
        if automatic and constraints.required_packing not in {None, WeightPacking.SOURCE_NATIVE}:
            raise ValueError("automatic MoE precision requires source-native weight storage")
        if automatic and source.w13_layout.value != "w13":
            raise ValueError("automatic MoE precision requires up/gate W13 row order")
        recipe = _packed_recipe(source, ActivationMode.A4 if automatic else activation.mode)
        shared_a16 = activation.a16_max_tokens > 0 and recipe != "w4a16"
        requested_layout = None
        if automatic:
            requested_layout = WeightPacking.SOURCE_NATIVE.value
        if recipe == "w4a16" and constraints.required_packing is not None:
            if constraints.required_packing not in {
                WeightPacking.SOURCE_NATIVE,
                WeightPacking.MMA_PACKED,
                WeightPacking.IQ2_XS_COMPACT,
                WeightPacking.IQ2_XXS_COMPACT,
                WeightPacking.Q8_0_COMPACT,
            }:
                raise ValueError(
                    "A16 preparation requires source_native or mma_packed packing"
                )
            requested_layout = constraints.required_packing.value
        raw_plan = plan_b12x_fp4_moe_weights(
            quant_modes=(
                ("nvfp4", "w4a16") if automatic else
                (recipe, "w4a16") if shared_a16 else recipe
            ),
            source_format=source.format.value,
            activation=activation.nonlinearity,
            params_dtype=activation.io_dtype,
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            w13_layout="w13" if shared_a16 else source.w13_layout.value,
            w4a16_layout=requested_layout,
        )
    elif isinstance(source, TrellisSource):
        config = source.config
        if activation.a16_max_tokens:
            raise ValueError("A16 token cutoff requires ModelOpt NVFP4 weights")
        if activation.mode is not ActivationMode.A16:
            raise ValueError("Trellis fused MoE requires A16 activations")
        if constraints.required_packing not in {None, WeightPacking.TRELLIS_NATIVE}:
            raise ValueError("Trellis weights require trellis_native packing")
        _validate_trellis_runtime(config, explicit_uniform=source.uniform_bits is not None)
        expert = config.transform.expert
        if source.extent is not None:
            if source.extent.intermediate_size != geometry.intermediate_size:
                raise ValueError("trellis extent width differs from the weight geometry")
            if expert.kind == "intermediate_hadamard" and (
                32 * source.extent.first_slot % expert.post_block_size
                or geometry.intermediate_size % expert.post_block_size
            ):
                raise ValueError("trellis extent must contain complete post-Hadamard blocks")
        if activation.rotation_dtype is not None and (
            config.codebook.value == "mcg" and source.uniform_bits is None
        ):
            raise ValueError("rotation_dtype requires a uniform full-rotation path")
        recipe = "w4a16"
        bits = source.uniform_bits or 3
        # Match the native K2 intermediate-Hadamard and projection-tiered
        # geometries; container identity does not select an execution policy.
        tile_config = (
            (128, 128, 128, 128)
            if expert.kind == "intermediate_hadamard" and bits == 2
            else (128, 256, 64, 256)
            if config.codebook.value == "mcg" and source.uniform_bits is None
            else (64, 256, 64, 256)
        )
        raw_plan = plan_b12x_fp4_moe_weights(
            quant_modes=recipe,
            source_format="b12x_trellis",
            activation=activation.nonlinearity,
            params_dtype=activation.io_dtype,
            num_experts=geometry.num_experts,
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            w13_layout="w31",
            w4a16_layout="trellis_native",
            trellis_bits=bits,
            trellis_tile_config=tile_config,
            intermediate_hadamard=expert.kind == "intermediate_hadamard",
            trellis_codebook=config.codebook.value,
            trellis_rate_granularity=config.rate.granularity.value,
            intermediate_hadamard_blocks=(
                None if expert.kind == "none"
                else (expert.pre_block_size, expert.post_block_size)
            ),
        )
    else:
        raise TypeError("source must be a PackedSource, TrellisConfig or TrellisSource")

    return WeightPlan(
        source=source,
        activation=activation,
        geometry=geometry,
        prepared_format=_prepared_format(
            source=source,
            plan=raw_plan,
            recipe=recipe,
            constraints=constraints,
        ),
        _impl=raw_plan,
    )


def prepare_weights(
    *,
    plan: WeightPlan,
    weights: PackedWeights | TrellisWeights | IQ2XSWeights,
    device: torch.device | str | None = None,
    staging: TrellisStaging | None = None,
) -> PreparedExperts:
    """Materialize the in-memory representation selected by ``plan_weights``."""

    if not isinstance(plan, WeightPlan):
        raise TypeError("plan must be a WeightPlan")
    if (device is not None or staging is not None) and not isinstance(plan.source, TrellisSource):
        raise ValueError("device and staging are only supported for trellis preparation")
    if isinstance(plan.source, PackedSource) and plan.source.format.value in BLOCK_CODECS:
        if not isinstance(weights, IQ2XSWeights) or weights.codec != plan.source.format.value:
            raise TypeError(f"{plan.source.format.value} preparation requires matching BlockQuantWeights")
        prepared = prepare_b12x_iq2_xs_weights(plan=plan._impl, weights=weights)
    elif isinstance(plan.source, TrellisSource):
        if not isinstance(weights, TrellisWeights):
            raise TypeError("Trellis preparation requires TrellisWeights")
        prepared = prepare_b12x_trellis_v2_weights(
            plan=plan._impl,
            source=plan.source,
            weights=weights,
            device=device,
            staging=staging,
            rotation_dtype=plan.activation.rotation_dtype,
        )
    else:
        if not isinstance(weights, PackedWeights):
            raise TypeError("packed preparation requires PackedWeights")
        if (plan.activation.a16_max_tokens and is_gated_moe_activation(plan._impl.activation)
                and plan.source.w13_layout.value == "w31" and plan._impl.w13_layout == "w13"):
            from ._impl import _ensure_w13_kernel_order_inplace

            # Both activation precisions must see the same physical FC1 halves.
            mode = _packed_recipe(plan.source, plan.activation.mode)
            _ensure_w13_kernel_order_inplace(
                weights.w13, weights.w13_block_scales,
                n=plan.geometry.intermediate_size, k=plan.geometry.hidden_size,
                quant_mode=mode,
            )
        input_scale = weights.input_scale
        intermediate_scale = weights.intermediate_scale
        if plan.activation.mode is ActivationMode.A16:
            input_scale = torch.ones(
                plan.geometry.num_experts,
                dtype=torch.float32,
                device=weights.w13.device,
            )
            intermediate_scale = input_scale
        elif (input_scale is None) != (intermediate_scale is None):
            raise ValueError(
                "input_scale and intermediate_scale must be supplied together"
            )
        elif input_scale is None:
            if plan.source.format.value == "modelopt_nvfp4":
                raise ValueError(
                    "ModelOpt NVFP4 A4/A8 preparation requires activation scales"
                )
            input_scale = torch.ones(
                plan.geometry.num_experts,
                dtype=torch.float32,
                device=weights.w13.device,
            )
            intermediate_scale = input_scale
        prepared = prepare_b12x_fp4_moe_weights(
            plan=plan._impl,
            params_dtype=plan.activation.io_dtype,
            w1_fp4=weights.w13,
            w2_fp4=weights.w2,
            w1_global_scale=weights.w13_global_scales,
            w2_global_scale=weights.w2_global_scales,
            w1_blockscale=weights.w13_block_scales,
            w2_blockscale=weights.w2_block_scales,
            immutable_input_scales=weights.immutable_input_scales,
            a1_gscale=input_scale,
            a2_gscale=intermediate_scale,
        )
    return PreparedExperts(plan=plan, _impl=prepared)


__all__ = [
    "ActivationMode",
    "ActivationSpec",
    "MoEGeometry",
    "WeightPlan",
    "WeightPlanConstraints",
    "plan_weights",
    "prepare_weights",
]
