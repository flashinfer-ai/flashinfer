"""Fixed hash backend and complete immutable PLE launch coordinates."""
from __future__ import annotations

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract


OPERANDS = (
    "token_ids", "query_start_loc", "committed_history", "num_seqs", "num_tokens",
    "multipliers", "prime_sizes", "table_offsets", "out", "request_ids",
)


@dataclass(frozen=True, kw_only=True)
class PleHashQuery:
    max_tokens: int
    max_seqs: int
    vocab_size: int
    max_order: int
    heads_per_order: int
    base_table_size: int
    eos_token_id: int = 0
    dense_layer_ordinal: int = 0
    table_alignment: int = 128
    geometry: tuple | None = None
    input_alignments: tuple[int, ...] = (16,) * len(OPERANDS)

    def __post_init__(self):
        if self.geometry is not None:
            sizes, offsets, factors, padded = self.geometry
            object.__setattr__(self, "geometry", (tuple(sizes), tuple(offsets), tuple(factors), padded))
        object.__setattr__(self, "input_alignments", tuple(self.input_alignments))

    @property
    def head_count(self):
        return (self.max_order - 1) * self.heads_per_order


def _validate_query(query, device):
    if not isinstance(query, PleHashQuery):
        raise TypeError("query must be PleHashQuery")
    if any(type(value) is not int or value <= 0 for value in (
        query.max_tokens, query.max_seqs, query.vocab_size, query.heads_per_order,
        query.base_table_size, query.table_alignment,
    )) or type(query.max_order) is not int or query.max_order < 2:
        raise ValueError("invalid PLE hash dimensions")
    if type(query.dense_layer_ordinal) is not int or query.dense_layer_ordinal < 0:
        raise ValueError("PLE dense layer ordinal must be nonnegative")
    if type(query.eos_token_id) is not int or not 0 <= query.eos_token_id < query.vocab_size:
        raise ValueError("PLE EOS token must fit the vocabulary")
    if query.geometry is None:
        raise ValueError("PLE hash queries require host geometry")
    from .geometry import compute_geometry
    geometry = compute_geometry(
        query, prime_sizes=query.geometry[0], table_offsets=query.geometry[1], multipliers=query.geometry[2],
    )
    if geometry.key() != query.geometry:
        raise ValueError("PLE hash geometry extent is inconsistent")
    if len(query.input_alignments) != len(OPERANDS) or any(
        type(value) is not int or value not in (1, 2, 4, 8, 16) for value in query.input_alignments
    ):
        raise ValueError("PLE hash requires alignment coordinates for every operand")


PleHashConfig = BackendConfig
TUNING = replace(
    make_fixed_contract(component_id="sequence.ple_hash", query_type=PleHashQuery, backend="triton"),
    query_schema_version=3, validate_query=_validate_query,
)

__all__ = ["PleHashConfig", "PleHashQuery", "TUNING"]
