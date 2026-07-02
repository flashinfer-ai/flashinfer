try:
    from .delta_rule_sm90 import (
        delta_rule_prefill_dsl_sm90 as chunk_gated_delta_rule_sm90,
    )
except ImportError:
    chunk_gated_delta_rule_sm90 = None  # type: ignore

try:
    from .delta_rule_sm120 import delta_rule_prefill_dsl as chunk_gated_delta_rule_sm120
except (ImportError, RuntimeError):
    chunk_gated_delta_rule_sm120 = None  # type: ignore

try:
    from .delta_rule_cp_sm90 import cp_delta_rule_dsl_sm90
except (ImportError, RuntimeError):
    cp_delta_rule_dsl_sm90 = None  # type: ignore

try:
    from .delta_rule_cp_sm120 import cp_delta_rule_dsl_sm120
except (ImportError, RuntimeError):
    cp_delta_rule_dsl_sm120 = None  # type: ignore

__all__ = [
    "chunk_gated_delta_rule_sm90",
    "chunk_gated_delta_rule_sm120",
    "cp_delta_rule_dsl_sm90",
    "cp_delta_rule_dsl_sm120",
]
