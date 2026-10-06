"""Typed configuration for the ``b12x_trellis`` checkpoint encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class TrellisCodebook(str, Enum):
    MCG = "mcg"
    LUT_E4M3 = "lut_e4m3"
    LUT_FP16 = "lut_fp16"


class RateGranularity(str, Enum):
    UNIFORM = "uniform"
    PER_LAYER = "per_layer"
    PER_EXPERT = "per_expert"
    PER_EXPERT_PROJECTION = "per_expert_projection"


class ScaleGranularity(str, Enum):
    NONE = "none"
    UNIFORM = "uniform"
    PER_LAYER = "per_layer"
    PER_EXPERT = "per_expert"


def _object(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")
    return value


def _enum(enum_type: type[Enum], value: object, *, name: str):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(f"{name} must be one of {choices}; got {value!r}") from exc


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class TrellisRateConfig:
    granularity: RateGranularity
    group_size: int | None = None

    @classmethod
    def from_dict(cls, value: object) -> "TrellisRateConfig":
        data = _object(
            value,
            name="b12x_trellis.rate",
            required=frozenset({"granularity"}),
            optional=frozenset({"group_size"}),
        )
        group_size = data.get("group_size")
        if group_size is not None:
            group_size = _positive_int(
                group_size, name="b12x_trellis.rate.group_size"
            )
            if group_size % 32:
                raise ValueError(
                    "b12x_trellis.rate.group_size must be a multiple of the "
                    "32-channel slot width"
                )
        return cls(
            granularity=_enum(
                RateGranularity,
                data["granularity"],
                name="b12x_trellis.rate.granularity",
            ),
            group_size=group_size,
        )

    def tensor_shape(
        self,
        *,
        num_layers: int,
        num_experts: int,
        intermediate_size: int,
    ) -> tuple[int, ...]:
        """Return the model-level uint8 rate-tensor shape."""

        num_layers = _positive_int(num_layers, name="num_layers")
        num_experts = _positive_int(num_experts, name="num_experts")
        intermediate_size = _positive_int(
            intermediate_size, name="intermediate_size"
        )
        if self.granularity is RateGranularity.UNIFORM:
            shape: tuple[int, ...] = (1,)
        elif self.granularity is RateGranularity.PER_LAYER:
            shape = (num_layers,)
        elif self.granularity is RateGranularity.PER_EXPERT:
            shape = (num_layers, num_experts)
        else:
            shape = (num_layers, num_experts, 3)
        if self.group_size is not None:
            if intermediate_size % self.group_size:
                raise ValueError(
                    "intermediate_size must be divisible by rate.group_size"
                )
            shape += (intermediate_size // self.group_size,)
        return shape

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"granularity": self.granularity.value}
        if self.group_size is not None:
            result["group_size"] = self.group_size
        return result


@dataclass(frozen=True)
class TrellisScaleFactorsConfig:
    vectors: ScaleGranularity
    gains: ScaleGranularity

    @classmethod
    def from_dict(
        cls, value: object, *, name: str
    ) -> "TrellisScaleFactorsConfig":
        data = _object(
            value,
            name=name,
            required=frozenset({"vectors", "gains"}),
        )
        vectors = _enum(
            ScaleGranularity, data["vectors"], name=f"{name}.vectors"
        )
        gains = _enum(ScaleGranularity, data["gains"], name=f"{name}.gains")
        if vectors is ScaleGranularity.NONE:
            raise ValueError(f"{name}.vectors cannot be 'none'")
        return cls(vectors=vectors, gains=gains)

    def to_dict(self) -> dict[str, str]:
        return {"vectors": self.vectors.value, "gains": self.gains.value}


@dataclass(frozen=True)
class TrellisScaleConfig:
    input_scales: TrellisScaleFactorsConfig
    intermediate_scales: TrellisScaleFactorsConfig
    output_scales: TrellisScaleFactorsConfig

    @classmethod
    def from_dict(cls, value: object) -> "TrellisScaleConfig":
        data = _object(
            value,
            name="b12x_trellis.scale",
            required=frozenset(
                {"input_scales", "intermediate_scales", "output_scales"}
            ),
        )
        return cls(
            input_scales=TrellisScaleFactorsConfig.from_dict(
                data["input_scales"],
                name="b12x_trellis.scale.input_scales",
            ),
            intermediate_scales=TrellisScaleFactorsConfig.from_dict(
                data["intermediate_scales"],
                name="b12x_trellis.scale.intermediate_scales",
            ),
            output_scales=TrellisScaleFactorsConfig.from_dict(
                data["output_scales"],
                name="b12x_trellis.scale.output_scales",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_scales": self.input_scales.to_dict(),
            "intermediate_scales": self.intermediate_scales.to_dict(),
            "output_scales": self.output_scales.to_dict(),
        }


@dataclass(frozen=True)
class TrellisProjectionTransform:
    kind: str
    block_size: int | None = None

    @classmethod
    def from_dict(cls, value: object) -> "TrellisProjectionTransform":
        data = _object(
            value,
            name="b12x_trellis.transform.projection",
            required=frozenset({"kind"}),
            optional=frozenset({"block_size"}),
        )
        kind = data["kind"]
        if kind == "none":
            if "block_size" in data:
                raise ValueError(
                    "projection block_size is invalid when transform kind is 'none'"
                )
            return cls(kind="none")
        if kind != "scaled_hadamard":
            raise ValueError(
                "b12x_trellis.transform.projection.kind must be 'none' or "
                f"'scaled_hadamard'; got {kind!r}"
            )
        if "block_size" not in data:
            raise ValueError(
                "scaled_hadamard projection transforms require block_size"
            )
        block_size = _positive_int(
            data["block_size"],
            name="b12x_trellis.transform.projection.block_size",
        )
        if block_size & (block_size - 1):
            raise ValueError("projection transform block_size must be a power of two")
        return cls(kind=kind, block_size=block_size)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind}
        if self.block_size is not None:
            result["block_size"] = self.block_size
        return result


@dataclass(frozen=True)
class TrellisExpertTransform:
    kind: str
    pre_block_size: int | None = None
    post_block_size: int | None = None
    sign_pattern_granularity: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> "TrellisExpertTransform":
        data = _object(
            value,
            name="b12x_trellis.transform.expert",
            required=frozenset({"kind"}),
            optional=frozenset(
                {"pre_block_size", "post_block_size", "sign_pattern_granularity"}
            ),
        )
        kind = data["kind"]
        extra = frozenset(data) - {"kind"}
        if kind == "none":
            if extra:
                raise ValueError(
                    "expert transform fields are invalid when kind is 'none'"
                )
            return cls(kind="none")
        if kind != "intermediate_hadamard":
            raise ValueError(
                "b12x_trellis.transform.expert.kind must be 'none' or "
                f"'intermediate_hadamard'; got {kind!r}"
            )
        required = {"pre_block_size", "post_block_size", "sign_pattern_granularity"}
        missing = sorted(required - frozenset(data))
        if missing:
            raise ValueError(
                "intermediate_hadamard expert transform is missing: "
                + ", ".join(missing)
            )
        pre = _positive_int(
            data["pre_block_size"],
            name="b12x_trellis.transform.expert.pre_block_size",
        )
        post = _positive_int(
            data["post_block_size"],
            name="b12x_trellis.transform.expert.post_block_size",
        )
        if pre & (pre - 1) or post & (post - 1):
            raise ValueError("expert transform block sizes must be powers of two")
        sign_pattern_granularity = data["sign_pattern_granularity"]
        if sign_pattern_granularity != "per_expert":
            raise ValueError(
                "intermediate_hadamard sign_pattern_granularity must be 'per_expert'"
            )
        return cls(
            kind=kind,
            pre_block_size=pre,
            post_block_size=post,
            sign_pattern_granularity=sign_pattern_granularity,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind}
        if self.kind == "intermediate_hadamard":
            result.update(
                pre_block_size=self.pre_block_size,
                post_block_size=self.post_block_size,
                sign_pattern_granularity=self.sign_pattern_granularity,
            )
        return result


@dataclass(frozen=True)
class TrellisTransformConfig:
    projection: TrellisProjectionTransform
    expert: TrellisExpertTransform

    @classmethod
    def from_dict(cls, value: object) -> "TrellisTransformConfig":
        data = _object(
            value,
            name="b12x_trellis.transform",
            required=frozenset({"projection", "expert"}),
        )
        return cls(
            projection=TrellisProjectionTransform.from_dict(data["projection"]),
            expert=TrellisExpertTransform.from_dict(data["expert"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "projection": self.projection.to_dict(),
            "expert": self.expert.to_dict(),
        }


@dataclass(frozen=True)
class TrellisConfig:
    """The complete model-level ``quantization_config.b12x_trellis`` value."""

    version: int
    codebook: TrellisCodebook
    rate: TrellisRateConfig
    scale: TrellisScaleConfig
    transform: TrellisTransformConfig

    @classmethod
    def from_dict(cls, value: object) -> "TrellisConfig":
        data = _object(
            value,
            name="quantization_config.b12x_trellis",
            required=frozenset(
                {"version", "codebook", "rate", "scale", "transform"}
            ),
        )
        version = data["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("b12x_trellis.version must be an integer")
        if version != 2:
            raise ValueError(
                f"unsupported b12x_trellis version {version!r}; expected 2"
            )
        return cls(
            version=version,
            codebook=_enum(
                TrellisCodebook, data["codebook"], name="b12x_trellis.codebook"
            ),
            rate=TrellisRateConfig.from_dict(data["rate"]),
            scale=TrellisScaleConfig.from_dict(data["scale"]),
            transform=TrellisTransformConfig.from_dict(data["transform"]),
        )

    @classmethod
    def from_quantization_config(cls, value: object) -> "TrellisConfig":
        data = _object(
            value,
            name="quantization_config",
            required=frozenset({"quant_method", "b12x_trellis"}),
        )
        if data["quant_method"] != "b12x_trellis":
            raise ValueError(
                "quantization_config.quant_method must be 'b12x_trellis'"
            )
        return cls.from_dict(data["b12x_trellis"])

    @property
    def projection_mixed(self) -> bool:
        return self.rate.granularity is RateGranularity.PER_EXPERT_PROJECTION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "codebook": self.codebook.value,
            "rate": self.rate.to_dict(),
            "scale": self.scale.to_dict(),
            "transform": self.transform.to_dict(),
        }


__all__ = [
    "RateGranularity",
    "ScaleGranularity",
    "TrellisCodebook",
    "TrellisConfig",
    "TrellisExpertTransform",
    "TrellisProjectionTransform",
    "TrellisRateConfig",
    "TrellisScaleConfig",
    "TrellisScaleFactorsConfig",
    "TrellisTransformConfig",
]
