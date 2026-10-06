"""Canonical fused-MoE weight planning and preparation."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS

import os
from dataclasses import dataclass, replace
from enum import Enum

import torch

from .._shared.execution import MoEWeightPreparationPlan
from .._shared.kernels.activations import is_gated_moe_activation
from ._impl import (
    plan_b12x_fp4_moe_weights,
    prepare_b12x_fp4_moe_weights,
    prepare_b12x_trellis_v2_weights,
    prepare_b12x_iq2_xs_weights,
    prepare_b12x_x4t_weights,
)
from .config import TrellisConfig
from .source import PackedSource, TrellisSource, WeightSource
from .trellis_layout import TrellisStaging
from .weights import (
    CsfScalePlanes,
    Nvfp4CsfWeights,
    PackedWeights,
    Mxfp4CsfWeights,
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


def _validate_trellis_runtime(
    source: TrellisConfig, *, explicit_uniform: bool = False
) -> None:
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
            WeightEncoding(source.format.value)
            if source.format.value in BLOCK_CODECS
            else WeightEncoding.FP6_E2M3
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
            source.format.value != "modelopt_nvfp4"
            or activation.io_dtype is not torch.bfloat16
        ):
            raise ValueError(
                "A16 token cutoff requires BF16 inputs and ModelOpt NVFP4 weights"
            )
        if automatic and (
            source.format.value != "modelopt_nvfp4"
            or activation.io_dtype is not torch.bfloat16
            or activation.nonlinearity != "silu"
        ):
            raise ValueError(
                "automatic MoE precision requires BF16 inputs, SiLU, and ModelOpt NVFP4 weights"
            )
        if automatic and constraints.required_packing not in {
            None,
            WeightPacking.SOURCE_NATIVE,
        }:
            raise ValueError(
                "automatic MoE precision requires source-native weight storage"
            )
        if automatic and source.w13_layout.value != "w13":
            raise ValueError("automatic MoE precision requires up/gate W13 row order")
        recipe = _packed_recipe(
            source, ActivationMode.A4 if automatic else activation.mode
        )
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
                ("nvfp4", "w4a16")
                if automatic
                else (recipe, "w4a16")
                if shared_a16
                else recipe
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
        _validate_trellis_runtime(
            config, explicit_uniform=source.uniform_bits is not None
        )
        expert = config.transform.expert
        if source.extent is not None:
            if source.extent.intermediate_size != geometry.intermediate_size:
                raise ValueError(
                    "trellis extent width differs from the weight geometry"
                )
            if expert.kind == "intermediate_hadamard" and (
                32 * source.extent.first_slot % expert.post_block_size
                or geometry.intermediate_size % expert.post_block_size
            ):
                raise ValueError(
                    "trellis extent must contain complete post-Hadamard blocks"
                )
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
                None
                if expert.kind == "none"
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
    weights: PackedWeights
    | TrellisWeights
    | IQ2XSWeights
    | Mxfp4CsfWeights
    | Nvfp4CsfWeights,
    device: torch.device | str | None = None,
    staging: TrellisStaging | None = None,
) -> PreparedExperts:
    """Materialize the in-memory representation selected by ``plan_weights``."""

    if not isinstance(plan, WeightPlan):
        raise TypeError("plan must be a WeightPlan")
    if (device is not None or staging is not None) and not isinstance(
        plan.source, TrellisSource
    ):
        raise ValueError(
            "device and staging are only supported for trellis preparation"
        )
    if isinstance(weights, Nvfp4CsfWeights):
        from b12x._lib.quant.nvfp4_csf import (
            Nvfp4CsfDecoder,
            make_nvfp4_csf_batch,
            repack_nvfp4_csf_batch,
        )

        if (
            not isinstance(plan.source, PackedSource)
            or plan.source.format.value != "modelopt_nvfp4"
            or plan.source.w13_layout.value != "w13"
            or plan.activation.mode not in (ActivationMode.A4, ActivationMode.A16)
            or (
                plan.activation.mode is ActivationMode.A16
                and plan.prepared_format.packing is not WeightPacking.MMA_PACKED
            )
            or plan.activation.a16_max_tokens
        ):
            raise ValueError(
                "NVFP4-CSF requires native ModelOpt NVFP4 A4/A16 in up/gate order"
            )
        geometry = plan.geometry
        planes = tuple(
            make_nvfp4_csf_batch(
                plane.fixed,
                plane.exceptions,
                rows=rows,
                columns=columns,
                device=weights.packed.w13.device,
            )
            if isinstance(plane, CsfScalePlanes)
            else plane
            for plane, rows, columns in (
                (
                    weights.w13_scales,
                    2 * geometry.intermediate_size,
                    geometry.hidden_size // 16,
                ),
                (
                    weights.w2_scales,
                    geometry.hidden_size,
                    geometry.intermediate_size // 16,
                ),
            )
        )
        if plan.activation.mode is ActivationMode.A16:
            from b12x.moe._shared.kernels.w4a16.prepare import (
                _nvfp4_compute_scale_factor,
                _process_nvfp4_packed_scales,
            )

            # Standard W4A16 preparation chooses one scale normalization per
            # projection and packs weights/scales into its MMA layout. Expand
            # one layer in shared scratch to retain that exact preparation law.
            source_scales = (
                weights.packed.w13_block_scales,
                weights.packed.w2_block_scales,
            )
            staging_decoder = Nvfp4CsfDecoder.prepare(*planes, *source_scales)
            ids = torch.arange(
                plan.geometry.num_experts,
                dtype=torch.int32,
                device=weights.packed.w13.device,
            )
            staging_decoder.decode(ids, *source_scales)
            factors = tuple(
                _nvfp4_compute_scale_factor(t, plan.activation.io_dtype)
                for t in source_scales
            )
        prepared = prepare_weights(plan=plan, weights=weights.packed)
        if plan.activation.mode is ActivationMode.A16:
            representation = prepared._impl.representation
            packed = representation.value
            outputs = (
                source_scales[0].view_as(packed.w13_scale),
                source_scales[1].view_as(packed.w2_scale),
            )
            tables = []
            for factor in factors:
                alphabet = torch.arange(
                    256, dtype=torch.uint8, device=weights.packed.w13.device
                ).view(torch.float8_e4m3fn)
                alphabet = (
                    alphabet[:, None]
                    .expand(256, 4)
                    .contiguous()
                    .to(plan.activation.io_dtype)
                )
                tables.append(
                    _process_nvfp4_packed_scales(alphabet, scale_factor=factor)
                    .view(torch.uint8)[:, 0]
                    .contiguous()
                )
            planes = tuple(
                repack_nvfp4_csf_batch(plane, row_rotation=rotation, value_lut=table)
                for plane, rotation, table in zip(
                    planes, (plan.geometry.intermediate_size, 0), tables, strict=True
                )
            )
            if _w4a16_stage_scales(plan.geometry):
                # Keep the compressed planes as stage-readable storage. Calls
                # planned up to the stage-scale token limit rebuild each
                # pipeline stage's scales in shared memory, with no expansion
                # pass before the layer. Larger calls expand their routed
                # experts into the shared scratch first, from the same storage.
                from b12x._lib.quant.nvfp4_csf_packed import (
                    PackedCsfPlane,
                    build_packed_csf_scales,
                )

                stored = tuple(build_packed_csf_scales(plane) for plane in planes)
                expander = Nvfp4CsfDecoder.prepare(
                    *(PackedCsfPlane.of(scales) for scales in stored), *outputs
                )
                expanded = replace(packed, w13_scale=outputs[0], w2_scale=outputs[1])
                packed = replace(
                    packed,
                    w13_scale=stored[0].storage,
                    w2_scale=stored[1].storage,
                    scale_format="e4m3_k16_csf",
                )
                plan = replace(
                    plan, _impl=replace(plan._impl, w4a16_compressed_scales=True)
                )
                return PreparedExperts(
                    plan=plan,
                    _impl=replace(
                        prepared._impl,
                        plan=plan._impl,
                        representation=replace(representation, value=packed),
                        w1_blockscale=stored[0].storage,
                        w2_blockscale=stored[1].storage,
                        nvfp4_csf=expander,
                        w4a16_expanded=expanded,
                    ),
                )
            packed = replace(packed, w13_scale=outputs[0], w2_scale=outputs[1])
            prepared = replace(
                prepared,
                _impl=replace(
                    prepared._impl,
                    representation=replace(representation, value=packed),
                    w1_blockscale=outputs[0],
                    w2_blockscale=outputs[1],
                ),
            )
        inline = None
        if (
            plan.activation.mode is ActivationMode.A4
            and plan.activation.nonlinearity == "silu"
            and plan.geometry.intermediate_size % 64 == 0
            and plan.geometry.hidden_size % 128 == 0
        ):
            from b12x._lib.quant.nvfp4_csf_inline import prepare_inline_scales

            inline = tuple(prepare_inline_scales(plane) for plane in planes)
            # Both execution paths own views of one fixed stream. Only the
            # exception index differs between whole-plane and tile reads.
            planes = tuple(
                replace(plane, fixed=value.storage[1024: 1024 + plane.fixed.numel()].view_as(plane.fixed))
                for plane, value in zip(planes, inline, strict=True)
            )
            plan = replace(plan, _impl=replace(plan._impl, nvfp4_inline_scales=True))
        decoder = Nvfp4CsfDecoder.prepare(
            *planes,
            prepared._impl.w1_blockscale,
            prepared._impl.w2_blockscale,
            inline_scales=inline,
        )
        return PreparedExperts(
            plan=plan,
            _impl=replace(
                prepared._impl, plan=plan._impl,
                nvfp4_csf=replace(decoder, inline_scales=inline),
            ),
        )
    if isinstance(weights, Mxfp4CsfWeights):
        if (
            not isinstance(plan.source, PackedSource)
            or plan.source.format.value != "fp4_e8m0_k32"
            or plan.activation.mode not in (ActivationMode.A16, ActivationMode.A8)
            or (
                plan.activation.mode is ActivationMode.A16
                and plan.prepared_format.packing
                not in {WeightPacking.MMA_PACKED, WeightPacking.SOURCE_NATIVE}
            )
        ):
            raise ValueError(
                "MXFP4-CSF requires native MXFP4 with uniform A16 or A8 activations"
            )
        from b12x._lib.quant.x4t_scales import make_x4t_scale_batch

        geometry = plan.geometry
        rotation = (
            geometry.intermediate_size if plan.source.w13_layout.value == "w13" else 0
        )
        planes = tuple(
            make_x4t_scale_batch(
                plane.fixed,
                plane.exceptions,
                rows=rows,
                columns=columns,
                device=weights.w13.device,
                exception_task_rows=64,
                exception_row_rotation=row_rotation,
            )
            if isinstance(plane, CsfScalePlanes)
            else plane
            for plane, rows, columns, row_rotation in (
                (
                    weights.w13_scales,
                    2 * geometry.intermediate_size,
                    geometry.hidden_size // 32,
                    rotation,
                ),
                (
                    weights.w2_scales,
                    geometry.hidden_size,
                    geometry.intermediate_size // 32,
                    0,
                ),
            )
        )
        weights = replace(weights, w13_scales=planes[0], w2_scales=planes[1])
        if plan.activation.mode is ActivationMode.A8:
            from b12x._lib.quant.mxfp4_csf import (
                Mxfp4CsfDecoder,
                repack_mxfp4_csf_batch,
            )
            from b12x._lib.quant.x4t_scales import decode_x4t_scales

            if (
                not isinstance(plan.source, PackedSource)
                or plan.source.format.value != "fp4_e8m0_k32"
                or plan.activation.a16_max_tokens
            ):
                raise ValueError("MXFP4-CSF A8 requires a uniform MXFP4/MXFP8 plan")
            e, h, n = (
                plan.geometry.num_experts,
                plan.geometry.hidden_size,
                plan.geometry.intermediate_size,
            )
            planes = (weights.w13_scales, weights.w2_scales)
            source_scales = (
                weights.w13_scale_scratch.view(e, 2 * n, h // 32),
                weights.w2_scale_scratch.view(e, h, n // 32),
            )
            ids = torch.arange(e, dtype=torch.int32, device=weights.w13.device)
            for plane, output in zip(planes, source_scales, strict=True):
                decode_x4t_scales(plane, ids, output)
            unit = torch.ones(e, dtype=torch.float32, device=weights.w13.device)
            packed = PackedWeights(
                w13=weights.w13,
                w2=weights.w2,
                w13_block_scales=source_scales[0],
                w2_block_scales=source_scales[1],
                w13_global_scales=unit,
                w2_global_scales=unit,
            )
            prepared = prepare_weights(plan=plan, weights=packed)
            representation = prepared._impl.representation
            native = representation.value
            outputs = []
            for buffer, scales in zip(
                (weights.w13_scale_scratch, weights.w2_scale_scratch),
                (native.w13_sfb, native.w2_sfb),
                strict=True,
            ):
                if (
                    buffer.numel() * buffer.element_size()
                    != scales.numel() * scales.element_size()
                ):
                    raise ValueError(
                        "MXFP4-CSF A8 scratch must match the prepared scale storage"
                    )
                output = buffer.view(-1).view(scales.dtype).view_as(scales)
                output.copy_(scales)
                outputs.append(output)
            native = replace(native, w13_sfb=outputs[0], w2_sfb=outputs[1])
            prepared = replace(
                prepared,
                _impl=replace(
                    prepared._impl,
                    representation=replace(representation, value=native),
                    w1_blockscale=outputs[0],
                    w2_blockscale=outputs[1],
                ),
            )
            if _w4a8_csf_inline(plan):
                # The scratch now holds every expert's native scales. Keep them
                # as inline storage that the compact W4A8 kernels read per
                # pipeline stage. Larger planned capacities expand this storage
                # into the shared scratch before execution.
                from b12x._lib.quant.mxfp4_csf_inline import build_mxfp4_csf_inline

                inline = tuple(
                    build_mxfp4_csf_inline(
                        output.view(torch.uint8).view(e, -1),
                        rows=rows,
                        columns=columns,
                        group_rows=group,
                    )
                    for output, rows, columns, group in (
                        (outputs[0], 2 * n, h // 32, n),
                        (outputs[1], h, n // 32, h),
                    )
                )
                plan = replace(plan, _impl=replace(plan._impl, w4a8_csf_inline=True))
                return PreparedExperts(
                    plan=plan,
                    _impl=replace(
                        prepared._impl, plan=plan._impl, mxfp4_csf_inline=inline
                    ),
                )
            rotation = n if plan.source.w13_layout.value == "w31" else 0
            native_planes = tuple(
                repack_mxfp4_csf_batch(
                    plane,
                    compact=n % 128 == 64,
                    group_rows=group,
                    row_rotation=rot,
                )
                for plane, group, rot in zip(planes, (n, h), (rotation, 0), strict=True)
            )
            decoder = Mxfp4CsfDecoder.prepare(
                *native_planes,
                prepared._impl.w1_blockscale,
                prepared._impl.w2_blockscale,
            )
            return PreparedExperts(
                plan=plan, _impl=replace(prepared._impl, mxfp4_csf=decoder)
            )
        prepared = prepare_b12x_x4t_weights(plan=plan._impl, weights=weights)
    elif (
        isinstance(plan.source, PackedSource)
        and plan.source.format.value in BLOCK_CODECS
    ):
        if (
            not isinstance(weights, IQ2XSWeights)
            or weights.codec != plan.source.format.value
        ):
            raise TypeError(
                f"{plan.source.format.value} preparation requires matching BlockQuantWeights"
            )
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
        if (
            plan.activation.a16_max_tokens
            and is_gated_moe_activation(plan._impl.activation)
            and plan.source.w13_layout.value == "w31"
            and plan._impl.w13_layout == "w13"
        ):
            from ._impl import _ensure_w13_kernel_order_inplace

            # Both activation precisions must see the same physical FC1 halves.
            mode = _packed_recipe(plan.source, plan.activation.mode)
            _ensure_w13_kernel_order_inplace(
                weights.w13,
                weights.w13_block_scales,
                n=plan.geometry.intermediate_size,
                k=plan.geometry.hidden_size,
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


def _w4a8_csf_inline(plan: WeightPlan) -> bool:
    """Whether compact W4A8 experts read MXFP4-CSF scales inline.

    Compact (N64) W4A8 execution runs the compact micro and split M16 kernels
    only; all of them read inline storage. B12X_W4A8_CSF_INLINE=0 keeps the
    per-call expansion of the routed experts' scales into scratch.
    """
    return (
        os.environ.get("B12X_W4A8_CSF_INLINE", "1") != "0"
        and plan.activation.nonlinearity == "silu"
        and plan.geometry.intermediate_size % 128 == 64
    )


def _w4a16_stage_scales(geometry) -> bool:
    """Whether W4A16 keeps NVFP4-CSF scales as stage-readable storage.

    Stages cover whole 128-row slabs of four-k-group atoms in both projections.
    B12X_W4A16_CSF_INLINE=0 keeps the per-layer expansion pass.
    """
    return (
        os.environ.get("B12X_W4A16_CSF_INLINE", "1") != "0"
        and geometry.hidden_size % 128 == 0
        and geometry.intermediate_size % 64 == 0
    )


__all__ = [
    "ActivationMode",
    "ActivationSpec",
    "MoEGeometry",
    "WeightPlan",
    "WeightPlanConstraints",
    "plan_weights",
    "prepare_weights",
]
