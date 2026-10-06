"""Configuration contract for chunked KDA prefill."""

from __future__ import annotations

from dataclasses import dataclass, field

from b12x.preparation._efficiency import capture_exhaustive_search, powers_of_two

from b12x.preparation import Knob, ParameterBinding, ParameterSpace, TuningContract
from .._shared.delta_prefill.config import PrefillConfig, validate_metadata_query
from .._shared.delta_prefill.workspace import (
    BACKEND,
    CHUNK_TOKENS,
    K_SPLIT_CHOICES,
    STAGE_CHOICES,
    V_SPLIT_CHOICES,
    WINDOW_BYTES_BUDGET,
    WORKSPACE_RECORD_BYTES,
    WorkspaceRecord,
    default_window_tiles,
    tiles_capacity,
    validate_shared_memory,
)


@dataclass(frozen=True, kw_only=True)
class KdaPrefillQuery:
    heads: int
    head_dim: int
    model_dtype: str
    state_dtype: str
    qk_l2norm: bool
    checkpoint_export: bool
    max_tokens: int
    max_seqs: int
    max_state_slots: int = 1
    null_state_index: int | None = None
    a_log_dtype: str = "float32"
    dt_bias_dtype: str = "float32"
    state_indices_dtype: str = "int32"
    exhaustive: bool = field(default_factory=capture_exhaustive_search)

    def to_dict(self) -> dict[str, object]:
        return {
            "heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "model_dtype": str(self.model_dtype),
            "state_dtype": str(self.state_dtype),
            "qk_l2norm": bool(self.qk_l2norm),
            "checkpoint_export": bool(self.checkpoint_export),
            "max_tokens": int(self.max_tokens),
            "max_seqs": int(self.max_seqs),
            "max_state_slots": self.max_state_slots,
            "null_state_index": self.null_state_index,
            "a_log_dtype": self.a_log_dtype,
            "dt_bias_dtype": self.dt_bias_dtype,
            "state_indices_dtype": self.state_indices_dtype,
            "exhaustive": self.exhaustive,
        }


class KdaPrefillConfig(PrefillConfig):
    """Launch configuration for lower-bounded KDA prefill."""


def _default_config(query: KdaPrefillQuery, device) -> KdaPrefillConfig:
    del device
    return KdaPrefillConfig(
        backend=BACKEND,
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=default_window_tiles(
            query.heads, query.max_tokens, query.max_seqs
        ),
    )


def _validate_query(query: KdaPrefillQuery, device) -> None:
    del device
    if not isinstance(query, KdaPrefillQuery):
        raise TypeError("query must be KdaPrefillQuery")
    validate_metadata_query(query)
    if query.head_dim != 128:
        raise ValueError(f"sequence.kda_prefill requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            "sequence.kda_prefill requires bfloat16 activations and float32 "
            f"state, got {query.model_dtype}/{query.state_dtype}"
        )


def _validate_config(query: KdaPrefillQuery, config: KdaPrefillConfig, device) -> None:
    del query, device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported sequence.kda_prefill backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(f"unsupported sequence.kda_prefill v_split {config.v_split!r}; expected one of {V_SPLIT_CHOICES}")
    if config.k_split not in K_SPLIT_CHOICES:
        raise ValueError(f"unsupported sequence.kda_prefill k_split {config.k_split!r}; expected one of {K_SPLIT_CHOICES}")
    if config.stages not in STAGE_CHOICES:
        raise ValueError(f"unsupported sequence.kda_prefill stages {config.stages!r}; expected one of {STAGE_CHOICES}")
    if 2 * config.v_split * config.k_split + 32 > 1024:
        raise ValueError("sequence.kda_prefill v_split x k_split exceeds the thread limit")
    if isinstance(config.window_tiles, bool) or int(config.window_tiles) < 1:
        raise ValueError("sequence.kda_prefill window_tiles must be a positive integer")
    validate_shared_memory(config.v_split, config.k_split, config.stages, config.window_tiles, reuse_value_buffer=False)


def _tuning_parameters(query: KdaPrefillQuery, device):
    del device
    capacity = tiles_capacity(query.max_tokens, query.max_seqs)
    windows = powers_of_two(capacity) | {
        capacity, default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    }
    return ParameterSpace.create(
        TUNING.knobs, values={"window_tiles": range(1, capacity + 1)},
        exhaustive=query.exhaustive,
        efficiency_predicates=(lambda p: p["window_tiles"] in windows,),
    )


# The state slot count sizes the caller's pool; it does not change which
# configuration is fastest, so it stays out of the selection key.
_KEY_FIELDS = frozenset(KdaPrefillQuery.__dataclass_fields__) - {"max_state_slots"}


def _encode_query(query: KdaPrefillQuery) -> dict[str, object]:
    return {name: value for name, value in query.to_dict().items() if name in _KEY_FIELDS}


TUNING = TuningContract(
    component_id="sequence.kda_prefill", query_schema_version=5, config_schema_version=1,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles"}),
    encode_query=_encode_query, encode_config=KdaPrefillConfig.to_dict,
    decode_config=KdaPrefillConfig.from_config, validate_query=_validate_query,
    validate_config=_validate_config, default_config=_default_config,
    knobs=(Knob(name="backend", values=(BACKEND,), binding=ParameterBinding.COMPILE), Knob(name="v_split", values=V_SPLIT_CHOICES, binding=ParameterBinding.COMPILE), Knob(name="k_split", values=K_SPLIT_CHOICES, binding=ParameterBinding.COMPILE), Knob(name="stages", values=STAGE_CHOICES, binding=ParameterBinding.COMPILE), Knob(name="window_tiles", values=None, binding=ParameterBinding.COMPILE)),
    candidate_contract_version=3, parameters=_tuning_parameters,
)

__all__ = ["BACKEND", "CHUNK_TOKENS", "K_SPLIT_CHOICES", "KdaPrefillConfig", "KdaPrefillQuery", "STAGE_CHOICES", "V_SPLIT_CHOICES", "WINDOW_BYTES_BUDGET", "WORKSPACE_RECORD_BYTES", "WorkspaceRecord", "default_window_tiles", "tiles_capacity", "TUNING"]
