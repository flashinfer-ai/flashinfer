"""Startup selection for W_o projection launch geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from b12x.preparation import FrozenMapping, Knob, ParameterBinding, ParameterSpace, TuningContract


@dataclass(frozen=True, kw_only=True)
class WoProjectionQuery:
    dtype: str
    max_tokens: int
    dynamic_tokens: bool = False
    groups: int
    group_width: int
    rank: int
    hidden: int
    codegen: FrozenMapping = FrozenMapping()
    operation: str = "plain"
    heads_per_group: int | None = None
    nope_dim: int | None = None
    rope_dim: int | None = None
    return_3d: bool = False
    positions_dtype: str = "int64"
    cos_sin_dtype: str = "bfloat16"
    sfb_k_replicated: bool = False
    wo_a_tiled: bool = False
    wo_b_tiled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "codegen", FrozenMapping(self.codegen))

@dataclass(frozen=True, kw_only=True)
class WoProjectionConfig:
    backend: str = "mxfp8"
    decode_tile_n: int = 0


def _decode_domain(query):
    return (
        not query.dynamic_tokens and query.dtype == "bfloat16" and 1 <= query.max_tokens <= 8
        and (query.groups, query.group_width, query.rank, query.hidden)
        == (2, 4096, 1024, 5120)
    )


def _validate_query(query, device):
    if not isinstance(query, WoProjectionQuery):
        raise TypeError("WO query must be WoProjectionQuery")
    if query.dtype not in ("bfloat16", "float16"):
        raise TypeError("WO input must be BF16 or FP16")
    if any(type(value) is not int or value <= 0 for value in (
        query.max_tokens, query.groups, query.group_width, query.rank, query.hidden,
    )):
        raise ValueError("WO geometry must contain positive integer dimensions")
    if type(query.dynamic_tokens) is not bool:
        raise TypeError("WO dynamic_tokens must be a boolean")
    if any(type(value) is not bool for value in (query.sfb_k_replicated, query.wo_a_tiled, query.wo_b_tiled)):
        raise TypeError("WO packed-weight contracts must be boolean")
    if query.wo_a_tiled and (query.groups, query.rank, query.group_width) != (4, 1024, 4096):
        raise ValueError("tiled WO-A requires four packed 1024-by-4096 weight groups")
    if query.wo_b_tiled and (query.hidden, query.groups * query.rank) != (4096, 4096):
        raise ValueError("tiled WO-B requires the packed 4096-by-4096 weight layout")
    if query.operation not in ("plain", "inv_rope"):
        raise ValueError("WO operation must be plain or inv_rope")
    if query.operation == "inv_rope" and (
        any(type(value) is not int or value <= 0 for value in (query.heads_per_group, query.nope_dim, query.rope_dim))
        or query.heads_per_group * (query.nope_dim + query.rope_dim) != query.group_width
        or query.rope_dim % 2
        or query.positions_dtype not in ("int32", "int64")
        or query.cos_sin_dtype not in ("bfloat16", "float16", "float32")
    ):
        raise ValueError("inverse-RoPE WO geometry or pointer dtypes are invalid")


def _validate(query, config, device):
    if not isinstance(config, WoProjectionConfig) or config.backend != "mxfp8":
        raise ValueError("WO projection requires an MXFP8 configuration")
    if type(config.decode_tile_n) is not int or config.decode_tile_n not in (0, 64, 128):
        raise ValueError("WO decode_tile_n must be 0, 64 or 128")
    if config.decode_tile_n and not _decode_domain(query):
        raise ValueError("WO decode tile requires BF16 2x4096/1024/5120 and M1..8")
    override = query.codegen.get("wo_b_fused_tile", "")
    if override and config.decode_tile_n not in (0, int(override.split("x")[1])):
        raise ValueError("WO decode tile conflicts with the captured tile control")


def _default(query, device):
    if not _decode_domain(query):
        return WoProjectionConfig()
    override = query.codegen.get("wo_b_fused_tile", "")
    tile_n = int(override.split("x")[1]) if override else (64 if query.max_tokens == 1 else 128)
    return WoProjectionConfig(decode_tile_n=tile_n)


def _parameters(query, device):
    override = query.codegen.get("wo_b_fused_tile", "")
    tiles = (int(override.split("x")[1]),) if override else (64, 128)
    return ParameterSpace.create(TUNING.knobs, values={
        "decode_tile_n": tiles if _decode_domain(query) else (0,),
    })


TUNING = TuningContract(
    component_id="gemm.wo_projection",
    query_schema_version=6,
    config_schema_version=2,
    query_fields=frozenset(WoProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(WoProjectionConfig.__dataclass_fields__),
    encode_query=lambda query: {name: getattr(query, name) for name in query.__dataclass_fields__},
    encode_config=asdict,
    decode_config=lambda payload: WoProjectionConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=_validate,
    default_config=_default,
    knobs=(
        Knob(name="backend", values=("mxfp8",), binding=ParameterBinding.COMPILE),
        Knob(name="decode_tile_n", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=5,
    parameters=_parameters,
    materialize=lambda query, device, choice: WoProjectionConfig(**dict(choice)),
)


__all__ = ["TUNING", "WoProjectionConfig", "WoProjectionQuery"]
