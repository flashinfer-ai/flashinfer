"""Typed tuning contract for contiguous batched and varlen attention."""

from __future__ import annotations

from dataclasses import dataclass, field

from b12x.preparation._efficiency import capture_exhaustive_search

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionQuery:
    variant: str
    dtype: str
    causal: bool
    batch_size: int
    q_heads: int
    kv_heads: int
    q_head_dim: int
    v_head_dim: int
    query_rows: int
    kv_rows: int
    max_seqlen_q: int
    max_seqlen_k: int
    exhaustive: bool = field(default_factory=capture_exhaustive_search)


@dataclass(frozen=True, kw_only=True)
class VarlenAttentionConfig:
    tile_m: int
    tile_n: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "VarlenAttentionConfig":
        if set(payload) != {"tile_m", "tile_n"}:
            raise ValueError("varlen attention configs require tile_m and tile_n")
        return cls(tile_m=int(payload["tile_m"]), tile_n=int(payload["tile_n"]))

    def to_dict(self) -> dict[str, object]:
        return {"tile_m": self.tile_m, "tile_n": self.tile_n}


def _encode(query: VarlenAttentionQuery) -> dict[str, object]:
    return {
        name: getattr(query, name) for name in VarlenAttentionQuery.__dataclass_fields__
    }


def _default_config(
    query: VarlenAttentionQuery,
    _device: DeviceIdentity | None,
) -> VarlenAttentionConfig:
    if query.q_head_dim <= 64:
        return VarlenAttentionConfig(tile_m=128, tile_n=128)
    if query.q_head_dim <= 128 or (
        query.q_head_dim == 192 and query.v_head_dim <= 128
    ):
        return VarlenAttentionConfig(tile_m=128, tile_n=64)
    if query.q_head_dim == 256:
        return VarlenAttentionConfig(
            tile_m=64,
            tile_n=32 if query.causal else 48,
        )
    raise ValueError(f"unsupported contiguous head_dim={query.q_head_dim}")


def _validate_query(
    query: VarlenAttentionQuery,
    _device: DeviceIdentity | None,
) -> None:
    if query.variant not in ("batched", "varlen"):
        raise ValueError(f"unsupported attention variant {query.variant!r}")
    if query.dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported attention dtype {query.dtype!r}")
    if query.q_heads <= 0 or query.kv_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if query.q_heads % query.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    if query.q_head_dim <= 0 or query.v_head_dim <= 0:
        raise ValueError("attention head dimensions must be positive")


def _validate_config(
    _query: VarlenAttentionQuery,
    config: VarlenAttentionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.tile_m <= 0 or config.tile_n <= 0:
        raise ValueError("attention tile dimensions must be positive")
    if config.tile_m % 16 or config.tile_n % 16:
        raise ValueError("attention tile dimensions must be multiples of 16")


def _tuning_parameters(
    query: VarlenAttentionQuery,
    _device: DeviceIdentity | None,
):
    import cutlass.utils as utils

    # Production fixes four compute warps, one stage and 160 threads. These
    # independent bounds contain every tile that can fit shared storage;
    # materialization applies the exact joint can_implement predicate.
    if query.q_head_dim <= 0 or query.v_head_dim <= 0:
        raise ValueError("attention head dimensions must be positive")
    capacity = utils.get_smem_capacity_in_bytes("sm_120")
    q_width = (query.q_head_dim + 15) // 16 * 16
    v_width = (query.v_head_dim + 15) // 16 * 16
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "tile_m": range(64, capacity // (2 * q_width) + 1, 64),
            "tile_n": range(16, capacity // (2 * (q_width + v_width)) + 1, 16),
        },
        exhaustive=query.exhaustive,
        efficiency_predicates=(lambda p: p["tile_m"] <= 128,),
    )


def _materialize_tuning(
    query: VarlenAttentionQuery,
    _device: DeviceIdentity | None,
    choice: FrozenMapping,
) -> VarlenAttentionConfig:
    import cutlass

    from .._shared.contiguous.forward import ContiguousAttentionForwardKernel

    if any(type(choice[name]) is not int for name in ("tile_m", "tile_n")):
        raise ValueError("attention tuning tiles must be integers")
    config = VarlenAttentionConfig.from_config(choice)
    _validate_config(query, config, _device)
    dtype = cutlass.BFloat16 if query.dtype == "bfloat16" else cutlass.Float16
    if not ContiguousAttentionForwardKernel.can_implement(
        dtype,
        query.q_head_dim,
        query.v_head_dim,
        config.tile_m,
        config.tile_n,
        1,
        160,
        query.causal,
    ):
        raise ValueError(
            "attention tuning tile is unsupported by the production kernel"
        )
    return config


TUNING = TuningContract(
    component_id="attention.varlen",
    query_schema_version=2,
    config_schema_version=1,
    query_fields=frozenset(VarlenAttentionQuery.__dataclass_fields__),
    config_fields=frozenset(VarlenAttentionConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=VarlenAttentionConfig.to_dict,
    decode_config=VarlenAttentionConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="tile_m", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="tile_n", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=3,
    parameters=_tuning_parameters,
    materialize=_materialize_tuning,
)


__all__ = ["VarlenAttentionConfig", "VarlenAttentionQuery", "TUNING"]
