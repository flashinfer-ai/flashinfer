"""Legal fused tiles and existing projection implementations for stream gating."""

from dataclasses import asdict, dataclass, fields

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract
from ..bf16_gemv._tuning import GemvConfig, GemvQuery, TUNING as GEMV_TUNING


@dataclass(frozen=True, kw_only=True)
class Query:
    max_rows: int
    hidden_size: int
    streams: int
    rank: int


@dataclass(frozen=True, kw_only=True)
class Config:
    implementation: str = "fused"
    tile_m: int = 32
    tile_h: int = 32
    tile_k: int = 64
    num_warps: int = 4
    num_stages: int = 3
    rows_per_tile: int = 8
    block_threads: int = 128
    gate_block: int = 256


def projection_query(query):
    return GemvQuery(
        source_dtype="bfloat16",
        weight_dtype="bfloat16",
        output_dtype="bfloat16",
        max_rows=query.max_rows,
        in_features=query.rank,
        out_features=query.streams * query.hidden_size,
        source_contiguous=True,
        source_aligned=True,
        weight_contiguous=True,
        weight_aligned=True,
        output_contiguous=True,
        output_aligned=True,
    )


def projection_config(config):
    return GemvConfig(
        backend=config.implementation,
        rows_per_tile=config.rows_per_tile,
        block_threads=config.block_threads,
    )


def validate_query(query, device):
    if not isinstance(query, Query):
        raise TypeError("hyperconnection gate query must be Query")
    if any(type(v) is not int or v <= 0 for v in asdict(query).values()):
        raise ValueError("hyperconnection gate dimensions must be positive integers")
    if query.streams not in (1, 2, 4, 8):
        raise ValueError("hyperconnection gate supports 1, 2, 4, or 8 streams")
    if query.rank % 16:
        raise ValueError("hyperconnection gate rank must be a multiple of 16")
    if query.hidden_size > 65536 or query.max_rows >= 2**31 or query.rank >= 2**31:
        raise ValueError(
            "hyperconnection gate hidden width, rank, or row capacity exceeds supported indexing"
        )


def _fused_supported(device):
    return device is None or device.compute_capability in ((12, 0), (12, 1))


def validate_config(query, config, device):
    if config.implementation == "fused" and not _fused_supported(device):
        raise ValueError("fused hyperconnection gate requires SM120 or SM121")
    if config.implementation != "fused":
        GEMV_TUNING.validate_config(
            projection_query(query), projection_config(config), device
        )
    if config.tile_m not in (16, 32, 64) or config.tile_h not in (16, 32, 64):
        raise ValueError("hyperconnection gate tile sizes must be 16, 32, or 64")
    if config.tile_k not in (32, 64) or config.num_warps != 4:
        raise ValueError("unsupported hyperconnection gate K tile or warp count")
    if config.num_stages not in (2, 3):
        raise ValueError("hyperconnection gate stages must be 2 or 3")
    if (
        config.implementation == "fused"
        and config.tile_m * config.tile_h * query.streams > 16384
    ):
        raise ValueError("fused tile exceeds the supported accumulator footprint")
    covering = 1 << (query.hidden_size - 1).bit_length()
    if config.gate_block not in {
        256,
        *(1 << e for e in range(covering.bit_length()) if 1 << e >= min(32, covering)),
    }:
        raise ValueError("gate block is outside the hidden partition range")


def parameters(query, device):
    projection_space = GEMV_TUNING.parameter_space(projection_query(query), device)
    backends = tuple(
        dict.fromkeys(c["backend"] for c in projection_space.configurations())
    )
    covering = 1 << (query.hidden_size - 1).bit_length()
    blocks = tuple(
        sorted(
            {
                256,
                *(
                    1 << e
                    for e in range(covering.bit_length())
                    if 1 << e >= min(32, covering)
                ),
            }
        )
    )
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "implementation": ("fused", *backends)
            if _fused_supported(device)
            else backends,
            "gate_block": blocks,
        },
        predicates=(
            lambda c: (
                c["implementation"] != "fused"
                or (
                    c["gate_block"] == 256
                    and c["tile_m"] * c["tile_h"] * query.streams <= 16384
                )
            ),
        ),
    )


def default_config(query, device):
    if _fused_supported(device):
        return Config()
    existing = GEMV_TUNING.default_config(projection_query(query), device)
    return Config(
        implementation=existing.backend,
        rows_per_tile=existing.rows_per_tile,
        block_threads=existing.block_threads,
    )


TUNING = TuningContract(
    component_id="gemm.hyperconnection_gate",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(Query)),
    config_fields=frozenset(f.name for f in fields(Config)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda p: Config(**dict(p)),
    validate_query=validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(
        Knob(name="implementation", values=None),
        *(
            Knob(
                name=name,
                values=values,
                when={"implementation": "fused"},
                otherwise=default,
            )
            for name, values, default in (
                ("tile_m", (16, 32, 64), 32),
                ("tile_h", (16, 32, 64), 32),
                ("tile_k", (32, 64), 64),
                ("num_warps", (4,), 4),
                ("num_stages", (2, 3), 3),
            )
        ),
        Knob(
            name="rows_per_tile",
            values=(1, 2, 4, 8),
            when={"implementation": "simt"},
            otherwise=8,
        ),
        Knob(
            name="block_threads",
            values=(128, 256),
            when={"implementation": "simt"},
            otherwise=128,
        ),
        Knob(name="gate_block", values=None, binding=ParameterBinding.COMPILE),
    ),
    parameters=parameters,
    materialize=lambda q, d, c: Config(**dict(c)),
)
