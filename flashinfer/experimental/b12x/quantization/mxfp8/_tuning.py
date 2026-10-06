"""Knob-only metadata for the public MXFP8 row quantizer."""

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class Mxfp8Query:
    """Complete immutable row-quantization declaration metadata."""
    rows: int
    columns: int
    dtype: str
    value_order: str
    scale_rows_layout: str
    scale_rows_storage: str
    scale_mma_layout: str
    scale_mma_storage: str


Mxfp8Config = BackendConfig
TUNING = make_fixed_contract(
    component_id="quantization.mxfp8",
    query_type=Mxfp8Query,
    backend="cutedsl",
)
_validate_fixed_query = TUNING.validate_query


def _validate_query(query: Mxfp8Query, device) -> None:
    _validate_fixed_query(query, device)
    if type(query.rows) is not int or query.rows <= 0:
        raise ValueError("MXFP8 planned rows must be a positive integer")
    if type(query.columns) is not int or query.columns <= 0 or query.columns % 32:
        raise ValueError("MXFP8 CuTe quantizer requires K divisible by 32")
    if query.dtype not in ("bfloat16", "float16"):
        raise ValueError("MXFP8 CuTe quantizer requires BF16 or FP16 input")
    if query.value_order not in ("linear", "trellis_native_mma"):
        raise ValueError("unsupported MXFP8 value order")
    if query.scale_rows_layout not in ("row_major", "grouped"):
        raise ValueError("unsupported MXFP8 row-scale layout")
    if query.scale_rows_storage not in ("uint8", "float8_e8m0fnu"):
        raise ValueError("unsupported MXFP8 row-scale storage")
    if query.scale_mma_layout not in ("linear_storage", "dense_gemm_swizzled"):
        raise ValueError("unsupported MXFP8 MMA-scale layout")
    if query.scale_mma_storage not in ("uint8", "float8_e8m0fnu"):
        raise ValueError("unsupported MXFP8 MMA-scale storage")


# Lane layout and threads are derived once from declaration metadata; private
# compiler parameters remain unavailable as public overrides.
TUNING = replace(
    TUNING,
    query_schema_version=3,
    semantic_version=3,
    validate_query=_validate_query,
)

__all__ = ["Mxfp8Query", "Mxfp8Config", "TUNING"]
