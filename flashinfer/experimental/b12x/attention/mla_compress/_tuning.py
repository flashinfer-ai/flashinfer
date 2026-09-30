"""Typed fixed contract for CSA1/CSA2 MLA compression."""
from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import make_fixed_contract


@dataclass(frozen=True, kw_only=True)
class MlaCompressQuery:
    ratio: int
    max_tokens: int
    max_requests: int
    max_states: int
    head_dim: int = 512


TUNING = make_fixed_contract(
    component_id="attention.mla_compress", query_type=MlaCompressQuery, backend="cute"
)

__all__ = ["MlaCompressQuery", "TUNING"]
