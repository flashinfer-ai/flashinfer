"""Fixed PLE residual pipeline configuration and complete launch metadata."""
from __future__ import annotations

from dataclasses import dataclass, replace

from b12x.preparation import BackendConfig, make_fixed_contract


OPERANDS = (
    "residual", "key", "value", "k_norm_weight", "q_norm_weight", "u_norm_weight",
    "conv_weight", "query_start_loc", "state_slot_ids", "state_is_fresh",
    "num_accepted_tokens", "request_is_prefill", "num_seqs", "num_tokens",
    "conv_state", "out", "normalized_u", "gathered_state", "request_ids",
)


@dataclass(frozen=True, kw_only=True)
class PleQuery:
    mode: str
    dtype: str
    max_tokens: int
    max_seqs: int
    max_speculative_tokens: int
    streams: int
    hidden_size: int
    kernel_size: int
    dilation: int
    max_state_slots: int = 1
    state_strides: tuple[int, int, int] | None = None
    input_alignments: tuple[int, ...] = (16,) * len(OPERANDS)

    def __post_init__(self):
        capacity = self.dilation * (self.kernel_size - 1) + self.max_speculative_tokens
        strides = (self.streams * self.hidden_size * capacity, capacity, 1) if self.state_strides is None else tuple(self.state_strides)
        object.__setattr__(self, "state_strides", strides)
        object.__setattr__(self, "input_alignments", tuple(self.input_alignments))


def _validate_query(query, device):
    if not isinstance(query, PleQuery):
        raise TypeError("query must be PleQuery")
    if query.mode not in ("decode", "prefill", "mixed") or query.dtype != "bfloat16":
        raise ValueError("unsupported PLE mode or dtype")
    if any(type(value) is not int or value <= 0 for value in (
        query.max_tokens, query.max_seqs, query.max_state_slots, query.streams,
        query.hidden_size, query.dilation,
    )):
        raise ValueError("PLE capacities and dimensions must be positive integers")
    if type(query.kernel_size) is not int or query.kernel_size < 2:
        raise ValueError("PLE kernel size must be at least two")
    if type(query.max_speculative_tokens) is not int or query.max_speculative_tokens < 0:
        raise ValueError("PLE speculative capacity must be nonnegative")
    if len(query.state_strides) != 3 or any(type(value) is not int or value <= 0 for value in query.state_strides):
        raise ValueError("PLE convolution state requires three positive strides")
    if len(query.input_alignments) != len(OPERANDS) or any(
        type(value) is not int or value not in (1, 2, 4, 8, 16) for value in query.input_alignments
    ):
        raise ValueError("PLE requires alignment metadata for every pipeline operand")


PleConfig = BackendConfig
# The state slot count sizes the caller's pool; it does not change which
# configuration is fastest, so it stays out of the selection key.
_KEY_FIELDS = frozenset(PleQuery.__dataclass_fields__) - {"max_state_slots"}
TUNING = replace(
    make_fixed_contract(component_id="sequence.ple", query_type=PleQuery, backend="triton"),
    query_schema_version=4, validate_query=_validate_query, query_fields=_KEY_FIELDS,
    encode_query=lambda query: {name: getattr(query, name) for name in _KEY_FIELDS},
)

__all__ = ["PleConfig", "PleQuery", "TUNING"]
