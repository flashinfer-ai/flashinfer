"""Configuration contract for chunked GDN prefill."""
from __future__ import annotations

from dataclasses import dataclass, field

from b12x.preparation._efficiency import capture_exhaustive_search

from b12x.preparation import FrozenMapping, Knob, ParameterBinding, ParameterSpace, TuningContract
from .._shared.delta_prefill.config import PrefillConfig, validate_metadata_query
from .._shared.delta_prefill.workspace import (
    BACKEND, CHUNK_TOKENS, K_SPLIT_CHOICES, STAGE_CHOICES, V_SPLIT_CHOICES,
    WINDOW_BYTES_BUDGET, WORKSPACE_RECORD_BYTES, WorkspaceRecord,
    default_window_tiles, tiles_capacity, validate_shared_memory,
)

SEGMENT_CHOICES = (128, 256, 512, 1024)


@dataclass(frozen=True, kw_only=True)
class GdnPrefillQuery:
    key_heads: int
    value_heads: int
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

    @property
    def heads(self) -> int:
        return self.value_heads

    def to_dict(self) -> dict[str, object]:
        return {
            "key_heads": int(self.key_heads),
            "value_heads": int(self.value_heads),
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


@dataclass(frozen=True)
class GdnPrefillConfig(PrefillConfig):
    algorithm: str = "sequential"
    segment_tokens: int = 256

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "GdnPrefillConfig":
        base = PrefillConfig.from_config(FrozenMapping({
            key: value for key, value in payload.items()
            if key not in ("algorithm", "segment_tokens")
        }))
        algorithm = payload.get("algorithm", "sequential")
        segment_tokens = payload.get("segment_tokens", 256)
        if not isinstance(algorithm, str):
            raise TypeError("algorithm must be a string")
        if isinstance(segment_tokens, bool) or not isinstance(segment_tokens, int):
            raise TypeError("segment_tokens must be an integer")
        return cls(**base.to_dict(), algorithm=algorithm, segment_tokens=segment_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            **super().to_dict(),
            "algorithm": self.algorithm,
            "segment_tokens": self.segment_tokens,
        }


def _default_config(query: GdnPrefillQuery, device) -> GdnPrefillConfig:
    del device
    return GdnPrefillConfig(
        backend=BACKEND, v_split=64, k_split=1, stages=3,
        window_tiles=default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    )


def _validate_query(query: GdnPrefillQuery, device) -> None:
    del device
    if not isinstance(query, GdnPrefillQuery):
        raise TypeError("query must be GdnPrefillQuery")
    validate_metadata_query(query)
    if query.key_heads <= 0 or query.value_heads != 3 * query.key_heads:
        raise ValueError("GDN prefill requires three value heads per key head")
    if query.head_dim != 128:
        raise ValueError(f"sequence.gdn_prefill requires head_dim 128, got {query.head_dim}")
    if query.model_dtype != "bfloat16" or query.state_dtype != "float32":
        raise ValueError(
            "sequence.gdn_prefill requires bfloat16 activations and float32 state, "
            f"got {query.model_dtype}/{query.state_dtype}"
        )


def _validate_config(query: GdnPrefillQuery, config: GdnPrefillConfig, device) -> None:
    del device
    if config.backend != BACKEND:
        raise ValueError(f"unsupported sequence.gdn_prefill backend {config.backend!r}")
    if config.v_split not in V_SPLIT_CHOICES:
        raise ValueError(f"unsupported sequence.gdn_prefill v_split {config.v_split!r}; expected one of {V_SPLIT_CHOICES}")
    if config.k_split not in K_SPLIT_CHOICES:
        raise ValueError(f"unsupported sequence.gdn_prefill k_split {config.k_split!r}; expected one of {K_SPLIT_CHOICES}")
    if config.stages not in STAGE_CHOICES:
        raise ValueError(f"unsupported sequence.gdn_prefill stages {config.stages!r}; expected one of {STAGE_CHOICES}")
    if 2 * config.v_split * config.k_split + 32 > 1024:
        raise ValueError("sequence.gdn_prefill v_split x k_split exceeds the thread limit")
    if isinstance(config.window_tiles, bool) or int(config.window_tiles) < 1:
        raise ValueError("sequence.gdn_prefill window_tiles must be a positive integer")
    if config.algorithm not in ("sequential", "chunk_parallel"):
        raise ValueError(f"unsupported GDN prefill algorithm {config.algorithm!r}")
    if config.segment_tokens not in SEGMENT_CHOICES:
        raise ValueError(f"segment_tokens must be one of {SEGMENT_CHOICES}")
    if config.algorithm == "chunk_parallel":
        segments = (query.max_tokens + config.segment_tokens - 1) // config.segment_tokens + query.max_seqs
        if segments > 4096:
            raise ValueError("chunk-parallel GDN supports at most 4096 planned segments")
        window = tiles_capacity(query.max_tokens, segments)
        if config.window_tiles != window:
            raise ValueError(f"chunk-parallel window_tiles must equal planned tile capacity {window}")
    validate_shared_memory(
        config.v_split, config.k_split, config.stages, config.window_tiles,
        max_sequence_tiles=config.segment_tokens // 16 if config.algorithm == "chunk_parallel" else 0,
        summary_mode=5 if config.algorithm == "chunk_parallel" and config.k_split == 1 else 0,
    )


def _tuning_parameters(query: GdnPrefillQuery, device):
    del device
    capacity = tiles_capacity(query.max_tokens, query.max_seqs)
    # A window is the tile band the pipeline keeps prepared in L2; one, two and
    # four windows bracket the residency/overlap trade within a factor of two.
    # The widest window whose prepared tiles fit the L2 budget is the other end
    # of that trade and lies below the whole ladder once capacity exceeds it.
    windows = tuple(sorted({
        capacity, -(-capacity // 2), -(-capacity // 4),
        default_window_tiles(query.heads, query.max_tokens, query.max_seqs),
    }))
    # Segments at or above max_tokens all plan one segment per sequence: the
    # same segment count, window and work, over a wider shared-memory band.
    covering = tuple(value for value in SEGMENT_CHOICES if value >= query.max_tokens)
    segments = tuple(value for value in SEGMENT_CHOICES if value < query.max_tokens) + covering[:1]
    return ParameterSpace.create(
        TUNING.knobs,
        values={"window_tiles": range(1, capacity + 1), "segment_tokens": SEGMENT_CHOICES},
        exhaustive=query.exhaustive,
        efficiency_predicates=(
            lambda p: p["algorithm"] != "sequential" or p["window_tiles"] in windows,
            lambda p: p["algorithm"] != "chunk_parallel" or p["segment_tokens"] in segments,
            # One segment per sequence leaves chunk-parallel no chunk-level
            # parallelism: it runs the sequential schedule plus summary and combine.
            lambda parameters: (
                parameters["algorithm"] != "chunk_parallel"
                or query.max_tokens > SEGMENT_CHOICES[0]
            ),
        ),
    )


def _materialize_tuning(query: GdnPrefillQuery, device, choice: FrozenMapping):
    del device
    payload = choice.to_dict()
    if payload["algorithm"] == "chunk_parallel":
        segment = payload["segment_tokens"]
        segments = (query.max_tokens + segment - 1) // segment + query.max_seqs
        payload["window_tiles"] = tiles_capacity(query.max_tokens, segments)
    return GdnPrefillConfig.from_config(FrozenMapping(payload))


# The state slot count sizes the caller's pool; it does not change which
# configuration is fastest, so it stays out of the selection key.
_KEY_FIELDS = frozenset(GdnPrefillQuery.__dataclass_fields__) - {"max_state_slots"}


def _encode_query(query: GdnPrefillQuery) -> dict[str, object]:
    return {name: value for name, value in query.to_dict().items() if name in _KEY_FIELDS}


TUNING = TuningContract(
    component_id="sequence.gdn_prefill", query_schema_version=5, config_schema_version=2,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset({"backend", "v_split", "k_split", "stages", "window_tiles", "algorithm", "segment_tokens"}),
    encode_query=_encode_query, encode_config=GdnPrefillConfig.to_dict,
    decode_config=GdnPrefillConfig.from_config, validate_query=_validate_query,
    validate_config=_validate_config, default_config=_default_config,
    knobs=(
        Knob(name="backend", values=(BACKEND,), binding=ParameterBinding.COMPILE),
        Knob(name="algorithm", values=("sequential", "chunk_parallel"), binding=ParameterBinding.COMPILE),
        Knob(name="v_split", values=V_SPLIT_CHOICES, binding=ParameterBinding.COMPILE),
        Knob(name="k_split", values=K_SPLIT_CHOICES, binding=ParameterBinding.COMPILE),
        Knob(name="stages", values=STAGE_CHOICES, binding=ParameterBinding.COMPILE),
        Knob(name="segment_tokens", values=SEGMENT_CHOICES, binding=ParameterBinding.COMPILE,
             when=FrozenMapping({"algorithm": "chunk_parallel"}), otherwise=256),
        Knob(name="window_tiles", values=None, binding=ParameterBinding.COMPILE,
             when=FrozenMapping({"algorithm": "sequential"})),
    ),
    candidate_contract_version=4, parameters=_tuning_parameters, materialize=_materialize_tuning,
)

__all__ = [
    "BACKEND", "CHUNK_TOKENS", "K_SPLIT_CHOICES", "SEGMENT_CHOICES",
    "GdnPrefillConfig", "GdnPrefillQuery",
    "STAGE_CHOICES", "V_SPLIT_CHOICES", "WINDOW_BYTES_BUDGET", "WORKSPACE_RECORD_BYTES",
    "WorkspaceRecord", "default_window_tiles", "tiles_capacity", "TUNING",
]
