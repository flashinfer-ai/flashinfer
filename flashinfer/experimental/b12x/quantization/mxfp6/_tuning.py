"""Fixed preparation contract for native MX-FP6 dense execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from b12x.preparation import FrozenMapping, Knob, TuningContract


@dataclass(frozen=True, kw_only=True)
class Mxfp6DenseQuery:
    max_tokens: int
    in_features: int
    out_features: int
    weight_format: str
    activation_format: str
    weight_storage: str
    global_scale_kind: str
    output_mode: str
    per_row_global_scale: bool


@dataclass(frozen=True, kw_only=True)
class Mxfp6DenseConfig:
    implementation: str = "native"


def _validate_query(query: Mxfp6DenseQuery, _device) -> None:
    if not isinstance(query, Mxfp6DenseQuery):
        raise TypeError("query must be Mxfp6DenseQuery")
    if any(type(value) is not int or value <= 0 for value in (
        query.max_tokens, query.in_features, query.out_features,
    )):
        raise ValueError("MX-FP6 geometry must contain positive integers")
    if query.in_features % 128:
        raise ValueError("MX-FP6 in_features must be a multiple of 128")
    if query.out_features % 128:
        raise ValueError("MX-FP6 out_features must be a multiple of 128")
    if query.weight_format not in ("e2m3", "e3m2"):
        raise ValueError("unsupported MX-FP6 weight format")
    if query.activation_format not in ("e2m3", "e3m2", "e4m3"):
        raise ValueError("unsupported MX-FP6 activation format")
    if query.weight_storage not in ("packed", "expanded"):
        raise ValueError("MX-FP6 weight storage must be packed or expanded")
    if query.global_scale_kind != "multiplier":
        raise ValueError("MX-FP6 dense weights use multiplier global scales")
    if query.output_mode not in ("functional", "provided"):
        raise ValueError("unsupported MX-FP6 output mode")
    if type(query.per_row_global_scale) is not bool:
        raise TypeError("per_row_global_scale must be boolean")


def _validate_config(query, config, device) -> None:
    del query, device
    if not isinstance(config, Mxfp6DenseConfig) or config.implementation != "native":
        raise ValueError("MX-FP6 only supports the native fixed implementation")


def _default(query, device):
    del query, device
    return Mxfp6DenseConfig()


def _decode(payload: FrozenMapping) -> Mxfp6DenseConfig:
    return Mxfp6DenseConfig(**payload.to_dict())


TUNING = TuningContract(
    component_id="quantization.mxfp6",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(Mxfp6DenseQuery.__dataclass_fields__),
    config_fields=frozenset(Mxfp6DenseConfig.__dataclass_fields__),
    encode_query=lambda query: asdict(query),
    encode_config=lambda config: asdict(config),
    decode_config=_decode,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default,
    knobs=(Knob(name="implementation", values=("native",)),),
    materialize=lambda _query, _device, assignment: Mxfp6DenseConfig(
        implementation=str(assignment["implementation"])
    ),
)
