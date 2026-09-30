"""Configuration contract for NVFP4 activation quantization."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class Nvfp4QuantizationQuery:
    dtype: str
    rows: int
    columns: int


@dataclass(frozen=True, kw_only=True)
class Nvfp4QuantizationConfig:
    backend: str
    liveness_strategy: str

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "Nvfp4QuantizationConfig":
        if set(payload) != {"backend", "liveness_strategy"}:
            raise ValueError("NVFP4 configs require backend and liveness_strategy")
        return cls(
            backend=str(payload["backend"]),
            liveness_strategy=str(payload["liveness_strategy"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "liveness_strategy": self.liveness_strategy,
        }


def _encode(query: Nvfp4QuantizationQuery) -> dict[str, object]:
    return {
        "dtype": query.dtype,
        "rows": query.rows,
        "columns": query.columns,
    }


def _default_config(
    query: Nvfp4QuantizationQuery,
    _device: DeviceIdentity | None,
) -> Nvfp4QuantizationConfig:
    return Nvfp4QuantizationConfig(
        backend="cutedsl",
        liveness_strategy="retain" if query.rows == 128 else "packed",
    )


def _validate_query(
    query: Nvfp4QuantizationQuery,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(query, Nvfp4QuantizationQuery):
        raise TypeError("query must be Nvfp4QuantizationQuery")
    if query.dtype != "bfloat16":
        raise ValueError("NVFP4 quantization requires bfloat16 input")
    if any(type(value) is not int or value <= 0 for value in (query.rows, query.columns)):
        raise ValueError("NVFP4 quantization dimensions must be positive")
    if query.rows % 128 or query.columns % 128:
        raise ValueError("NVFP4 quantization dimensions must be multiples of 128")


def _validate_config(
    query: Nvfp4QuantizationQuery,
    config: Nvfp4QuantizationConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, Nvfp4QuantizationConfig):
        raise TypeError("config must be Nvfp4QuantizationConfig")
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported NVFP4 backend {config.backend!r}")
    if config.liveness_strategy not in {"retain", "packed"}:
        raise ValueError("NVFP4 liveness_strategy must be retain or packed")


TUNING = TuningContract(
    component_id="quantization.nvfp4",
    query_schema_version=1,
    config_schema_version=2,
    query_fields=frozenset(Nvfp4QuantizationQuery.__dataclass_fields__),
    config_fields=frozenset(Nvfp4QuantizationConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=Nvfp4QuantizationConfig.to_dict,
    decode_config=Nvfp4QuantizationConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(name="liveness_strategy", values=("retain", "packed"), binding=ParameterBinding.COMPILE),
    ),
)


__all__ = ["Nvfp4QuantizationConfig", "Nvfp4QuantizationQuery", "TUNING"]
