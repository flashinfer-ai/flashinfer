"""Fixed native hash-and-lookup preparation contract for PLE tables."""
from __future__ import annotations

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract
from b12x.sequence.ple_hash._tuning import PleHashQuery, TUNING as HASH_TUNING


@dataclass(frozen=True, kw_only=True)
class PleEmbeddingQuery(PleHashQuery):
    quant_mode: str
    table_memory: str
    output_dtype: str
    embedding_dim: int
    tp_size: int
    tp_rank: int
    lookup_alignments: tuple[int, ...] = (16, 16, 16, 16)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "lookup_alignments", tuple(self.lookup_alignments))

    @property
    def head_dim(self):
        return self.embedding_dim // self.head_count


def _validate_query(query, device):
    if not isinstance(query, PleEmbeddingQuery):
        raise TypeError("query must be PleEmbeddingQuery")
    HASH_TUNING.validate_query(query, device)
    if query.quant_mode not in {"bf16", "fp8_e4m3_per_tensor", "nvfp4_group16"}:
        raise ValueError("unsupported PLE table quantization")
    if query.table_memory not in {"device", "mapped_host", "io_uring"}:
        raise ValueError("unsupported PLE table storage")
    if query.output_dtype != "bfloat16":
        raise ValueError("PLE embedding output must be BF16")
    if type(query.embedding_dim) is not int or query.embedding_dim <= 0 or query.embedding_dim % query.head_count:
        raise ValueError("embedding_dim must be positive and divisible by head_count")
    if type(query.tp_size) is not int or query.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if type(query.tp_rank) is not int or not 0 <= query.tp_rank < query.tp_size:
        raise ValueError("invalid table shard rank")
    if query.geometry[3] % query.tp_size:
        raise ValueError("padded vocabulary must be divisible by tp_size")
    if query.geometry[3] // query.tp_size * query.head_dim > (1 << 63) - 1:
        raise ValueError("TP-local PLE weight extent must fit signed int64 indexing")
    if query.quant_mode == "nvfp4_group16" and query.head_dim % 16:
        raise ValueError("NVFP4 table head_dim must be divisible by 16")
    if len(query.lookup_alignments) != 4 or any(type(value) is not int or value not in (1, 2, 4, 8, 16) for value in query.lookup_alignments):
        raise ValueError("invalid PLE lookup pointer alignment")


PleEmbeddingConfig = BackendConfig
TUNING = replace(
    make_fixed_contract(component_id="sequence.ple_embedding", query_type=PleEmbeddingQuery, backend="triton"),
    query_schema_version=3,
    validate_query=_validate_query,
)

__all__ = ["PleEmbeddingConfig", "PleEmbeddingQuery", "TUNING"]
