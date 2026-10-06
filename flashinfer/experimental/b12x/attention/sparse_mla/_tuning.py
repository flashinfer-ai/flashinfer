"""Fixed configuration contract for sparse MLA planning."""

from __future__ import annotations

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    max_q_rows: int
    max_width: int
    page_size: int
    model_type: int | None
    head_major_output: bool
    scale_format: int
    cache_record_bytes: int
    fp8_rope: bool
    latent_scale_per_token: bool
    has_attention_sink: bool
    cache_layout: str
    operation: str
    slot_dtype: str | None
    prefill_mg_enabled: bool
    max_batch: int = 0
    max_kv_rows: int = 0
    max_page_table_width: int = 0
    max_chunks_per_row: int = 0
    max_q_chunks: int = 0
    physical_block_size: int = 0
    physical_record_width: int = 0
    num_cache_blocks: int = 0
    max_physical_records: int = 0
    tp_size: int = 0
    use_cuda_graph: bool = False
    budget_max_splits: int | None = None
    budget_max_partial_rows: int | None = None
    pool_size: int = 0
    pool_topk: int = 0

SparseMlaConfig = BackendConfig
TUNING = replace(
    make_fixed_contract(
        component_id="attention.sparse_mla",
        query_type=SparseMlaQuery,
        backend="native",
    ),
    query_schema_version=3,
    candidate_contract_version=3,
)


__all__ = ["SparseMlaConfig", "SparseMlaQuery", "TUNING"]
