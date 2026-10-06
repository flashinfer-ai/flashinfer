"""Launch contract for MTP feedback fusion."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)
from ._cute_prefill_config import QWEN_HIDDEN_SIZE, QWEN_STREAMS


@dataclass(frozen=True, kw_only=True)
class MtpFeedbackQuery:
    dtype: str
    max_tokens: int
    hidden_size: int
    streams: int
    input_alignments: tuple[int, ...] = (16, 16, 16, 16)

    def __post_init__(self):
        object.__setattr__(self, "input_alignments", tuple(self.input_alignments))


@dataclass(frozen=True, kw_only=True)
class MtpFeedbackConfig:
    backend: str
    norm_block_h: int
    norm_block_s: int
    norm_num_warps: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "MtpFeedbackConfig":
        expected = {"backend", "norm_block_h", "norm_block_s", "norm_num_warps"}
        if set(payload) != expected:
            raise ValueError(f"MTP feedback config fields must be {expected}")
        return cls(
            backend=str(payload["backend"]),
            norm_block_h=int(payload["norm_block_h"]),
            norm_block_s=int(payload["norm_block_s"]),
            norm_num_warps=int(payload["norm_num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "norm_block_h": self.norm_block_h,
            "norm_block_s": self.norm_block_s,
            "norm_num_warps": self.norm_num_warps,
        }


def _encode(query: MtpFeedbackQuery) -> dict[str, object]:
    return {name: getattr(query, name) for name in query.__dataclass_fields__}


def _default_config(query: MtpFeedbackQuery, _device: DeviceIdentity | None) -> MtpFeedbackConfig:
    norm_block_h = 1 << (query.hidden_size - 1).bit_length()
    norm_block_s = 1 << (query.streams - 1).bit_length()
    return MtpFeedbackConfig(backend="cutedsl", norm_block_h=norm_block_h, norm_block_s=norm_block_s, norm_num_warps=8 if norm_block_h >= 2048 else 4)


def _validate_query(query: MtpFeedbackQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, MtpFeedbackQuery):
        raise TypeError("query must be MtpFeedbackQuery")
    if query.dtype != "bfloat16" or query.hidden_size != QWEN_HIDDEN_SIZE or query.streams != QWEN_STREAMS:
        raise ValueError("MTP feedback requires the existing Qwen BF16 geometry")
    if type(query.max_tokens) is not int or query.max_tokens <= 0:
        raise ValueError("MTP token capacity must be a positive integer")
    if len(query.input_alignments) != 4 or any(
        type(value) is not int or value not in (2, 4, 8, 16) for value in query.input_alignments
    ):
        raise ValueError("MTP requires four BF16 input-alignment coordinates")


def _validate_config(query: MtpFeedbackQuery, config: MtpFeedbackConfig, _device: DeviceIdentity | None) -> None:
    if not isinstance(config, MtpFeedbackConfig):
        raise TypeError("config must be MtpFeedbackConfig")
    if config.backend != "cutedsl": raise ValueError(f"unsupported MTP feedback backend {config.backend!r}")
    if config.norm_block_h < query.hidden_size: raise ValueError("norm_block_h must cover hidden_size")
    if config.norm_block_s < query.streams: raise ValueError("norm_block_s must cover streams")
    for name, value in (("norm_block_h", config.norm_block_h), ("norm_block_s", config.norm_block_s)):
        if value <= 0 or value & (value - 1): raise ValueError(f"{name} must be a positive power of two")
    if config.norm_num_warps not in (1, 2, 4, 8): raise ValueError("norm_num_warps must be one of 1, 2, 4, or 8")


def _tuning_parameters(query: MtpFeedbackQuery, device):
    del device
    h = 1 << (query.hidden_size - 1).bit_length()
    s = 1 << (query.streams - 1).bit_length()
    return {"norm_block_h": (h, 2 * h), "norm_block_s": (s, 2 * s)}


TUNING = TuningContract(
    component_id="sequence.mtp_feedback", query_schema_version=2, config_schema_version=1,
    query_fields=frozenset(MtpFeedbackQuery.__dataclass_fields__),
    config_fields=frozenset(MtpFeedbackConfig.__dataclass_fields__),
    encode_query=_encode, encode_config=MtpFeedbackConfig.to_dict,
    decode_config=MtpFeedbackConfig.from_config, validate_query=_validate_query,
    validate_config=_validate_config, default_config=_default_config,
    knobs=(
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(name="norm_block_h", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="norm_block_s", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="norm_num_warps", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=3, parameters=_tuning_parameters,
)

__all__ = ["MtpFeedbackConfig", "MtpFeedbackQuery", "TUNING"]
