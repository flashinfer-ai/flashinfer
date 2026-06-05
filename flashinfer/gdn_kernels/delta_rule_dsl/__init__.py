try:
    from .delta_rule_sm90 import (
        delta_rule_prefill_dsl_sm90 as chunk_gated_delta_rule_sm90,
    )
except ImportError:
    chunk_gated_delta_rule_sm90 = None  # type: ignore


__all__ = [
    "chunk_gated_delta_rule_sm90",
]
