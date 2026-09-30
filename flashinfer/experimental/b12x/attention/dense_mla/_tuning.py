"""Configuration contract for dense MLA planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from b12x.preparation._efficiency import capture_exhaustive_search, powers_of_two
from typing import Mapping

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

from .planner import Budget, choose_num_splits


@dataclass(frozen=True, kw_only=True)
class DenseMlaQuery:
    mode: str
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    page_size: int
    query_rows: int
    max_batch: int
    cache_tokens: int
    physical_record_width: int
    window_size: int | None
    use_cuda_graph: bool
    max_page_table_width: int
    num_cache_pages: int
    abi: FrozenMapping
    exhaustive: bool = field(default_factory=capture_exhaustive_search)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "num_q_heads": self.num_q_heads,
            "qk_head_dim": self.qk_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "query_rows": self.query_rows,
            "max_batch": self.max_batch,
            "cache_tokens": self.cache_tokens,
            "physical_record_width": self.physical_record_width,
            "window_size": self.window_size,
            "use_cuda_graph": self.use_cuda_graph,
            "max_page_table_width": self.max_page_table_width,
            "num_cache_pages": self.num_cache_pages,
            "abi": self.abi.to_dict(),
            "exhaustive": self.exhaustive,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DenseMlaQuery":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("dense MLA queries require complete capacity metadata")
        values = dict(payload)
        values["abi"] = FrozenMapping(values["abi"])
        return cls(**values)


@dataclass(frozen=True, kw_only=True)
class DenseMlaConfig:
    max_splits: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "DenseMlaConfig":
        if set(payload) != {"max_splits"}:
            raise ValueError("dense MLA configs require exactly max_splits")
        value = payload["max_splits"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("dense MLA max_splits must be an integer")
        return cls(max_splits=value)


def _query_tile(query: DenseMlaQuery) -> int:
    if query.mode == "decode" or query.max_batch != 1 or query.window_size is not None:
        return 1
    if query.kv_dtype == "float8_e4m3fn" and query.query_rows >= 3:
        return 4
    return 2


def _default_config(
    query: DenseMlaQuery,
    device: DeviceIdentity | None,
) -> DenseMlaConfig:
    max_attended_tokens = query.cache_tokens
    if query.window_size is not None:
        max_attended_tokens = min(
            query.cache_tokens,
            query.window_size + query.page_size - 1,
        )
    splits = choose_num_splits(
        max_cache_tokens=max_attended_tokens,
        max_total_q=query.query_rows,
        num_q_heads=query.num_q_heads,
        query_tile=_query_tile(query),
        sm_count=1 if device is None else device.sm_count,
        budget=Budget(),
    )
    return DenseMlaConfig(max_splits=splits)


def _validate_query(query: DenseMlaQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, DenseMlaQuery):
        raise TypeError("query must be DenseMlaQuery")


def _validate_config(
    query: DenseMlaQuery,
    config: DenseMlaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, DenseMlaConfig):
        raise TypeError("config must be DenseMlaConfig")
    if config.max_splits <= 0:
        raise ValueError("dense MLA max_splits must be positive")
    max_chunks = max(1, (query.cache_tokens + 63) // 64)
    if config.max_splits > max_chunks:
        raise ValueError("dense MLA max_splits cannot exceed the cache chunk count")


def _tuning_parameters(query: DenseMlaQuery, device: DeviceIdentity | None):
    capacity = max(1, (query.cache_tokens + 63) // 64)
    splits = set(powers_of_two(capacity))
    divisor = 1
    while divisor <= capacity:
        quotient = (capacity + divisor - 1) // divisor
        splits.add(quotient)
        if quotient == 1:
            break
        divisor = (capacity - 1) // (quotient - 1) + 1
    splits.add(_default_config(query, device).max_splits)
    return ParameterSpace.create(
        TUNING.knobs, values={"max_splits": range(1, capacity + 1)},
        exhaustive=query.exhaustive,
        efficiency_predicates=(lambda p: p["max_splits"] in splits,),
    )


TUNING = TuningContract(
    component_id="attention.mla",
    query_schema_version=5,
    config_schema_version=1,
    query_fields=frozenset(
        {
            "mode",
            "q_dtype",
            "kv_dtype",
            "num_q_heads",
            "qk_head_dim",
            "v_head_dim",
            "page_size",
            "query_rows",
            "max_batch",
            "cache_tokens",
            "physical_record_width",
            "window_size",
            "use_cuda_graph",
            "max_page_table_width",
            "num_cache_pages",
            "abi",
            "exhaustive",
        }
    ),
    config_fields=frozenset({"max_splits"}),
    encode_query=DenseMlaQuery.to_dict,
    encode_config=asdict,
    decode_config=DenseMlaConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="max_splits", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=3,
    parameters=_tuning_parameters,
)


__all__ = ["DenseMlaConfig", "DenseMlaQuery", "TUNING"]
