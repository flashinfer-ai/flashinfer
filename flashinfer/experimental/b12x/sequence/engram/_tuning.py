"""Typed fixed preparation contract for native Engram hashing and lookup."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

from b12x.preparation import make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class EngramQuery:
    max_tokens: int
    max_seqs: int
    vocab_size: int
    layer_id: int
    tp_size: int
    tp_rank: int
    compressed_vocab_size: int
    table_rows: int
    operation: Literal["hash", "lookup"]
    compact_rows: bool
    pad_id: int
    resident_scales: bool = False
    max_requests: int = 1


def _validate_query(query, device):
    del device
    if not isinstance(query, EngramQuery):
        raise TypeError("query must be EngramQuery")
    if query.operation not in ("hash", "lookup"):
        raise ValueError("Engram operation must be 'hash' or 'lookup'")
    if type(query.compact_rows) is not bool:
        raise TypeError("Engram compact_rows must be boolean")
    if type(query.resident_scales) is not bool:
        raise TypeError("resident_scales must be boolean")
    if query.resident_scales and (query.operation != "lookup" or not query.compact_rows):
        raise ValueError("resident scales require compact disk lookup rows")
    if query.operation == "hash" and query.compact_rows:
        raise ValueError("hash Engram declarations cannot select compact lookup rows")
    if type(query.pad_id) is not int or not 0 <= query.pad_id < query.vocab_size:
        raise ValueError("Engram pad_id must be an actual vocabulary ID")


# The committed-history slot count sizes the caller's pool and is checked at
# bind time; no kernel depends on it, so it stays out of the selection key.
_KEY_FIELDS = frozenset(EngramQuery.__dataclass_fields__) - {"max_requests"}
TUNING = replace(
    make_fixed_contract(
        component_id="sequence.engram", query_type=EngramQuery, backend="triton"
    ),
    query_schema_version=4,
    validate_query=_validate_query,
    query_fields=_KEY_FIELDS,
    encode_query=lambda query: {name: getattr(query, name) for name in _KEY_FIELDS},
)

__all__ = ["EngramQuery", "TUNING"]
