"""Complete per-invocation HyperConnection defaults and legal launch choices."""

from __future__ import annotations

import math
from dataclasses import dataclass

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.tuning import Knob, ParameterBinding, TuningContract


@dataclass(frozen=True, kw_only=True)
class HyperConnectionQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    streams: int
    lowrank: int
    operation: str
    output_mode: str = "provided"
    left_dtype: str = "bfloat16"
    right_dtype: str = "bfloat16"
    output_dtype: str = "bfloat16"
    token_mask: bool = False
    eps: float = 1.0e-6
    limit: float = float("inf")
    round_silu: bool = False
    zero_centered: bool = True
    weight_dtype: str = "bfloat16"




@dataclass(frozen=True, kw_only=True)
class HyperConnectionConfig:
    backend: str
    reduction_block_h: int
    pointwise_block: int
    reduction_num_warps: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "HyperConnectionConfig":
        expected = {
            "backend",
            "reduction_block_h",
            "pointwise_block",
            "reduction_num_warps",
        }
        if set(payload) != expected:
            raise ValueError(f"HyperConnection config fields must be {expected}")
        return cls(
            backend=str(payload["backend"]),
            reduction_block_h=int(payload["reduction_block_h"]),
            pointwise_block=int(payload["pointwise_block"]),
            reduction_num_warps=int(payload["reduction_num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "reduction_block_h": self.reduction_block_h,
            "pointwise_block": self.pointwise_block,
            "reduction_num_warps": self.reduction_num_warps,
        }


def _encode(query: HyperConnectionQuery) -> dict[str, object]:
    return {
        name: ("+inf" if name == "limit" and value == float("inf") else value)
        for name in query.__dataclass_fields__
        for value in (getattr(query, name),)
    }


def _default_config(
    query: HyperConnectionQuery,
    _device: DeviceIdentity | None,
) -> HyperConnectionConfig:
    reduction_block_h = 1 << (query.hidden_size - 1).bit_length()
    return HyperConnectionConfig(
        backend="cutedsl",
        reduction_block_h=reduction_block_h,
        pointwise_block=256,
        reduction_num_warps=8 if reduction_block_h >= 2048 else 4,
    )


def _validate(
    query: HyperConnectionQuery,
    config: HyperConnectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported HyperConnection backend {config.backend!r}")
    if config.reduction_block_h < query.hidden_size:
        raise ValueError("reduction_block_h must cover hidden_size")
    for name, value in (
        ("reduction_block_h", config.reduction_block_h),
        ("pointwise_block", config.pointwise_block),
    ):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if config.reduction_num_warps not in (1, 2, 4, 8):
        raise ValueError("reduction_num_warps must be one of 1, 2, 4, or 8")


def _validate_query(query, device):
    if not isinstance(query, HyperConnectionQuery):
        raise TypeError("query must be HyperConnectionQuery")
    if query.operation not in (
        "grouped_rmsnorm", "scaled_silu", "gate_mean", "combine", "combine_norm",
        "engram_mix", "swiglu", "add", "sigmoid",
    ):
        raise ValueError("unknown HyperConnection operation")
    if any(type(value) is not int or value <= 0 for value in (
        query.max_tokens, query.hidden_size, query.streams, query.lowrank,
    )):
        raise ValueError("HyperConnection dimensions must be positive integers")
    if query.output_mode not in ("provided", "functional"):
        raise ValueError("unknown HyperConnection output form")
    expected_mode = "functional" if query.operation in ("combine", "combine_norm") else "provided"
    if query.output_mode != expected_mode:
        raise ValueError("output form is not supported by this operation")
    if query.dtype != "bfloat16":
        raise ValueError("HyperConnection state dtype must be bfloat16")
    if query.operation not in ("add", "sigmoid") and any(value != "bfloat16" for value in (
        query.left_dtype, query.right_dtype, query.output_dtype,
    )):
        raise ValueError("this HyperConnection operation requires BF16 operands")
    if any(value not in ("bfloat16", "float32") for value in (
        query.left_dtype, query.right_dtype, query.output_dtype,
    )):
        raise ValueError("pointwise operands must be BF16 or FP32")
    if type(query.zero_centered) is not bool:
        raise TypeError("normalization centering must be a boolean")
    if query.weight_dtype not in ("bfloat16", "float32"):
        raise ValueError("affine weights must be BF16 or FP32")
    if query.operation != "grouped_rmsnorm" and (
        not query.zero_centered or query.weight_dtype != "bfloat16"
    ):
        raise ValueError("affine normalization controls require grouped RMSNorm")
    if query.zero_centered and query.weight_dtype != "bfloat16":
        raise ValueError("zero-centered grouped RMSNorm requires BF16 weights")
    if query.operation == "sigmoid" and query.right_dtype != query.left_dtype:
        raise ValueError("sigmoid uses one input dtype")
    if not math.isfinite(query.eps) or query.eps <= 0:
        raise ValueError("normalization epsilon must be finite and positive")
    if math.isnan(query.limit) or query.limit <= 0:
        raise ValueError("activation limit must be positive or infinity")
    if type(query.token_mask) is not bool or type(query.round_silu) is not bool:
        raise TypeError("operand and rounding flags must be boolean")
    if query.operation != "swiglu" and (query.round_silu or query.limit != float("inf")):
        raise ValueError("activation controls apply only to SwiGLU")
    if query.operation != "engram_mix" and query.token_mask:
        raise ValueError("token masks apply only to Engram mixing")


def _tuning_parameters(query, device):
    operation = query.operation
    covering = 1 << (query.hidden_size - 1).bit_length()
    # The owned vector domain includes the covering reduction and one overtile;
    # the reduction keeps that overtile as an explicit tuning range, not the
    # language's arange cap. The gate-mean launch fixes four warps per block and
    # grids cdiv(hidden_size, pointwise_block) blocks, so a partition wider than
    # the covering block adds only masked lanes to a single block. The floor of
    # one warp is a measured bound, not a dominance argument: a narrower
    # partition trades idle lanes for more blocks, and in gate_mean timings
    # every partition below 32 lanes trailed the best in-range partition by 1.7x
    # or more, reaching 40x at a single lane.
    blocks = tuple(
        block for block in (1 << exponent for exponent in range(covering.bit_length()))
        if block >= min(32, covering)
    )
    return {
        "reduction_block_h": (covering, 2 * covering)
        if operation == "grouped_rmsnorm" and query.zero_centered else (covering,),
        "pointwise_block": blocks if operation == "gate_mean" else (256,),
        "reduction_num_warps": (1, 2, 4, 8)
        if operation == "grouped_rmsnorm" and query.zero_centered else (4,),
    }


def _materialize_tuning(
    query: HyperConnectionQuery,
    device: DeviceIdentity | None,
    choice: FrozenMapping,
) -> HyperConnectionConfig:
    return HyperConnectionConfig(
        backend="cutedsl",
        reduction_block_h=choice["reduction_block_h"],
        pointwise_block=choice["pointwise_block"],
        reduction_num_warps=choice["reduction_num_warps"],
    )


TUNING = TuningContract(
    component_id="norm.hyperconnection",
    query_schema_version=3,
    config_schema_version=1,
    query_fields=frozenset(HyperConnectionQuery.__dataclass_fields__),
    config_fields=frozenset(HyperConnectionConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=HyperConnectionConfig.to_dict,
    decode_config=HyperConnectionConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate,
    default_config=_default_config,
    candidate_contract_version=6,
    knobs=(
        Knob(name="reduction_block_h", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="pointwise_block", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="reduction_num_warps", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE),
    ),
    materialize=_materialize_tuning,
    parameters=_tuning_parameters,
)


__all__ = [
    "HyperConnectionConfig",
    "HyperConnectionQuery",
    "TUNING",
]
