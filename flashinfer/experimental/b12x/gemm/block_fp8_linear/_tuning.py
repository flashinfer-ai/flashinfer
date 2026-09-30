"""Serialized block-FP8 activations and weights use the dense MXFP8 contract."""
from __future__ import annotations

from dataclasses import dataclass, field

from b12x.preparation._efficiency import capture_exhaustive_search

from b12x.preparation import FrozenMapping, TuningContract
from b12x.gemm import _tuning as dense


@dataclass(frozen=True, kw_only=True)
class BlockFp8LinearQuery:
    max_tokens: int
    in_features: int
    out_features: int
    source_dtype: str
    output_dtype: str
    output_mode: str
    weight_block_size: int = 128
    exhaustive: bool = field(default_factory=capture_exhaustive_search)
    codegen: FrozenMapping | None = None

    def __post_init__(self):
        object.__setattr__(self, "codegen", dense._codegen_snapshot()
                           if self.codegen is None else FrozenMapping(self.codegen))


def dense_query(query):
    from .._shared.block_fp8 import _physical_mxfp8_k

    return dense.DenseGemmQuery(
        recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",
        output_dtype=query.output_dtype, batch=1, max_rows=query.max_tokens,
        in_features=_physical_mxfp8_k(query.in_features),
        out_features=query.out_features, output_mode="provided", alpha_mode="unit",
        expected_m=query.max_tokens, sfb_k_replicated=query.weight_block_size == 128,
        codegen=query.codegen, exhaustive=query.exhaustive,
    )


def _validate_query(query, device):
    if not isinstance(query, BlockFp8LinearQuery):
        raise TypeError("query must be BlockFp8LinearQuery")
    if type(query.in_features) is not int or query.in_features <= 0 or query.in_features % 32:
        raise ValueError("block-FP8 input features must be a positive multiple of 32")
    if query.source_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported source dtype {query.source_dtype!r}")
    if query.output_mode not in ("functional", "provided"):
        raise ValueError(f"unsupported block-FP8 output mode {query.output_mode!r}")
    if query.weight_block_size not in (32, 128):
        raise ValueError("block-FP8 weight block size must be 32 or 128")
    dense.validate_query(dense_query(query))


def _default_config(query, device):
    return dense.default_config(dense_query(query), device)


def _validate_config(query, config, device):
    dense.validate_config(dense_query(query), config, device)


def _parameters(query, device):
    return dense.TUNING.parameters(dense_query(query), device)


TUNING = TuningContract(
    component_id="gemm.block_fp8_linear",
    query_schema_version=6,
    config_schema_version=4,
    query_fields=frozenset(BlockFp8LinearQuery.__dataclass_fields__),
    config_fields=dense.TUNING.config_fields,
    encode_query=lambda query: {name: getattr(query, name) for name in query.__dataclass_fields__},
    encode_config=dense.TUNING.encode_config,
    decode_config=dense.TUNING.decode_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    candidate_contract_version=dense.TUNING.candidate_contract_version,
    parameters=_parameters,
    knobs=dense.TUNING.knobs,
)


__all__ = ["TUNING", "BlockFp8LinearQuery"]
