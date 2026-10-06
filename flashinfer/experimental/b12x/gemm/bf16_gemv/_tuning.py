"""Startup selection for unquantized BF16 and FP32 projections."""

from dataclasses import asdict, dataclass, fields

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract


@dataclass(frozen=True, kw_only=True)
class GemvQuery:
    source_dtype: str
    weight_dtype: str
    max_rows: int
    in_features: int
    out_features: int
    source_contiguous: bool
    source_aligned: bool
    weight_contiguous: bool
    weight_aligned: bool
    output_dtype: str = "bfloat16"
    output_contiguous: bool = True
    output_aligned: bool = True
    bias_dtype: str | None = None


@dataclass(frozen=True, kw_only=True)
class GemvConfig:
    backend: str
    rows_per_tile: int = 8


def validate_query(query):
    if not isinstance(query, GemvQuery):
        raise TypeError("projection query must be GemvQuery")
    if any(dtype not in ("bfloat16", "float32") for dtype in (
        query.source_dtype, query.weight_dtype, query.output_dtype,
    )) or query.bias_dtype not in (None, "bfloat16", "float32"):
        raise ValueError("unquantized projection requires BF16 or FP32 operands")
    if not 0 < query.max_rows < 2**31 or min(query.in_features, query.out_features) <= 0:
        raise ValueError("projection geometry must be positive and row capacity fit Int32")
    for name in ("source_contiguous", "source_aligned", "weight_contiguous",
                 "weight_aligned", "output_contiguous", "output_aligned"):
        if type(getattr(query, name)) is not bool:
            raise TypeError(f"projection {name} must be a boolean")


def _mma_eligible(query):
    return (query.source_dtype == query.weight_dtype == "bfloat16"
            and query.out_features >= 256 and query.in_features >= 16)


def _torch_eligible(query):
    return (query.source_dtype == query.weight_dtype == query.output_dtype == "bfloat16"
            and query.bias_dtype in (None, "bfloat16"))


def _prefill_eligible(query):
    return (
        query.bias_dtype is None
        and query.source_dtype == query.weight_dtype == "bfloat16"
        and (query.out_features, query.in_features) in ((384, 5120), (512, 5120), (1024, 5120))
        and query.source_contiguous and query.source_aligned
        and query.weight_contiguous and query.weight_aligned
        and query.output_contiguous and query.output_aligned
    )


def default_config(query, device):
    prefill_min_rows = 128 if query.out_features == 1024 else 256
    if _prefill_eligible(query) and query.max_rows >= prefill_min_rows:
        return GemvConfig(backend="prefill")
    if _torch_eligible(query) and query.max_rows > 8:
        return GemvConfig(backend="torch")
    n_tiles = (query.out_features + 63) // 64
    minimum_mma_rows = 24 if n_tiles >= 64 else ((64 + n_tiles - 1) // n_tiles) * 32
    if _mma_eligible(query) and query.max_rows >= minimum_mma_rows:
        return GemvConfig(backend="mma")
    rows_per_tile = 8
    if (
        query.source_dtype == query.weight_dtype == "bfloat16"
        and query.in_features == 5120 and query.out_features in (32, 384, 512)
        and query.max_rows <= 8
        and query.source_contiguous and query.source_aligned
        and query.weight_contiguous and query.weight_aligned
    ):
        rows_per_tile = 2 if query.out_features == 32 else 4
    return GemvConfig(backend="simt", rows_per_tile=rows_per_tile)


def _validate_query(query, device):
    validate_query(query)


def validate_config(query, config, device):
    if not isinstance(config, GemvConfig):
        raise TypeError("projection config must be GemvConfig")
    if config.backend == "simt":
        if type(config.rows_per_tile) is not int or config.rows_per_tile not in (1, 2, 4, 8):
            raise ValueError("SIMT projection rows_per_tile must be 1, 2, 4 or 8")
        return
    if config.rows_per_tile != 8:
        raise ValueError("rows_per_tile only configures SIMT projection")
    if config.backend == "mma" and _mma_eligible(query):
        return
    if config.backend == "torch" and _torch_eligible(query):
        return
    if config.backend == "prefill" and _prefill_eligible(query):
        return
    raise ValueError(f"projection backend {config.backend!r} is ineligible for this query")


def _parameters(query, device):
    backends = ["simt"]
    if _mma_eligible(query):
        backends.append("mma")
    if _prefill_eligible(query):
        backends.append("prefill")
    if _torch_eligible(query):
        backends.append("torch")
    return ParameterSpace.create(
        TUNING.knobs, values={"backend": tuple(backends)}, predicates=(_eligible,),
    )


def _eligible(choice):
    return choice["backend"] == "simt" or choice["rows_per_tile"] == 8


TUNING = TuningContract(
    component_id="gemm.bf16_gemv",
    query_schema_version=4,
    config_schema_version=5,
    query_fields=frozenset(field.name for field in fields(GemvQuery)),
    config_fields=frozenset(field.name for field in fields(GemvConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: GemvConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(
        Knob(name="backend", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="rows_per_tile", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=3,
    parameters=_parameters,
    materialize=lambda query, device, choice: GemvConfig(**dict(choice)),
)
