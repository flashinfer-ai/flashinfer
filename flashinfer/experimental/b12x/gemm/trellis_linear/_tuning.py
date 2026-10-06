"""Host launch contract for Trellis linear."""

from dataclasses import asdict, dataclass, fields

from b12x.preparation import Knob, ParameterBinding, TuningContract


@dataclass(frozen=True, kw_only=True)
class TrellisQuery:
    max_rows: int
    in_features: int
    out_features: int
    input_dtype: str
    compute_dtype: str
    output_dtype: str
    codebook: str
    bits: int
    weight_layout: str = "native"
    pair_kind: str | None = None
    rate_axis: str | None = None
    output_mode: str = "functional"
    c_tmp_mode: str = "internal"
    transform_mode: str = "extension"
    gemm_output_provided: bool = False
    input_f16_provided: bool = False
    rotated_f16_provided: bool = False
    rotated_compute_provided: bool = False
    gemm_output_f16_provided: bool = False
    output_f16_provided: bool = False

@dataclass(frozen=True, kw_only=True)
class TrellisConfig:
    backend: str
    block_rows: int
    tile_k: int
    tile_n: int

    @classmethod
    def from_config(cls, payload):
        return cls(**dict(payload))


def validate_query(query):
    if (
        query.max_rows <= 0
        or min(query.in_features, query.out_features) <= 0
        or query.in_features % 128
        or query.out_features % 128
        or query.input_dtype not in ("float16", "bfloat16")
        or query.compute_dtype not in ("float16", "bfloat16")
        or query.output_dtype != query.input_dtype
        or query.output_mode not in ("functional", "provided")
        or query.c_tmp_mode not in ("internal", "provided")
        or query.transform_mode not in ("extension", "supplied")
        or any(
            type(value) is not bool
            for value in (
                query.gemm_output_provided,
                query.input_f16_provided,
                query.rotated_f16_provided,
                query.rotated_compute_provided,
                query.gemm_output_f16_provided,
                query.output_f16_provided,
            )
        )
        or query.weight_layout not in ("native", "p24_k", "p24_n", "p33_k", "p33_n")
    ):
        raise ValueError(
            "Trellis plan requires positive BF16/FP16 rows, matching output dtype, "
            "128-aligned N/K, and supported output/workspace/transform semantics"
        )
    from b12x.moe._shared.trellis_codebooks import (
        normalize_codebook,
        validate_codebook_bits,
    )

    if normalize_codebook(query.codebook) != query.codebook or query.bits not in (
        2, 3, 4, 5, 6,
    ):
        raise ValueError(
            "Trellis queries require a canonical codebook and supported native rate"
        )
    validate_codebook_bits(query.codebook, query.bits)
    if query.weight_layout == "native":
        if query.pair_kind is not None or query.rate_axis is not None:
            raise ValueError("native Trellis layout cannot carry pair metadata")
        return
    pair_kind, rate_axis = query.weight_layout.split("_")
    if query.pair_kind != pair_kind.upper() or query.rate_axis != rate_axis:
        raise ValueError("Trellis pair layout and explicit pair/rate metadata differ")
    size = query.in_features if rate_axis == "k" else query.out_features
    if query.bits != 3 or size != 256 or query.codebook not in ("mcg", "lut_e4m3"):
        raise ValueError(
            "compact Trellis pairs require three stored bits per weight, a 256-channel "
            "rate axis, and the mcg or lut_e4m3 codebook"
        )


def validate_config(query, config, device):
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _candidate_tile_fits,
        _w4a16_num_regs,
    )

    if (
        not isinstance(config, TrellisConfig)
        or config.backend != "cutedsl"
        or config.block_rows not in (8, 16, 32, 48, 64)
        or config.tile_k not in (64, 128)
        or config.tile_n not in (64, 128, 256)
    ):
        raise ValueError("invalid native Trellis launch decision")
    if query.weight_layout.endswith("_n") and config.tile_n != 256:
        raise ValueError("N-axis Trellis pairs require a complete 256-column tile")
    threads = config.tile_n * config.tile_k // 64
    if threads not in (128, 256):
        raise ValueError("Trellis requires 128 or 256 CTA threads")
    if not _candidate_tile_fits(
        problem_n=query.out_features,
        problem_k=query.in_features,
        cta_m_blocks=(config.block_rows + 15) // 16,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        cta_threads=threads,
        max_shared_mem=101_376 - 512,
        scale_format="e4m3_k32",
        weight_layout="trellis_t256",
        weight_bits=max(4, query.bits),
    ):
        raise ValueError("native Trellis tile exceeds shape or shared-memory limits")
    _w4a16_num_regs(
        cta_threads=threads,
        cta_m_blocks=(config.block_rows + 15) // 16,
        cta_n_blocks=config.tile_n // 16,
        cta_k_blocks=config.tile_k // 16,
        uses_m_block_8=config.block_rows == 8,
        weight_layout="trellis_t256",
    )


def default_config(query, device):
    from b12x.moe._shared.kernels.w4a16.kernel import _trellis256_dense_launch_geometry

    if device is None:
        raise ValueError("Trellis launch selection requires the device SM count")
    block_rows, (tile_k, tile_n) = _trellis256_dense_launch_geometry(
        size_m=query.max_rows,
        size_k=query.in_features,
        size_n=query.out_features,
        sms=device.sm_count,
    )
    return TrellisConfig(
        backend="cutedsl", block_rows=block_rows, tile_k=tile_k, tile_n=tile_n
    )


def _validate_query(query, device) -> None:
    if not isinstance(query, TrellisQuery):
        raise TypeError("query must be TrellisQuery")
    validate_query(query)


def launch_options(query, config):
    return {
        "_moe_block_size": config.block_rows,
        "_force_tile_config": (config.tile_k, config.tile_n),
    }


def _tuning_parameters(query, device):
    # These axes are the native override domain; validate_config checks the
    # interdependent CTA-thread, register, shared-memory and packed-layout constraints.
    return {
        "tile_n": (256,) if query.weight_layout.endswith("_n") else (64, 128, 256),
    }


TUNING = TuningContract(
    component_id="gemm.trellis_linear",
    query_schema_version=4,
    config_schema_version=1,
    semantic_version=1,
    query_fields=frozenset(field.name for field in fields(TrellisQuery)),
    config_fields=frozenset(field.name for field in fields(TrellisConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=TrellisConfig.from_config,
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    candidate_contract_version=2,
    knobs=(
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(name="block_rows", values=(8, 16, 32, 48, 64), binding=ParameterBinding.COMPILE),
        Knob(name="tile_k", values=(64, 128), binding=ParameterBinding.COMPILE),
        Knob(name="tile_n", values=(64, 128, 256), binding=ParameterBinding.COMPILE),
    ),
    parameters=_tuning_parameters,
)
