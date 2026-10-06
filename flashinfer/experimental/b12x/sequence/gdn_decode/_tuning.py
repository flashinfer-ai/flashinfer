"""Configuration contract for GDN decode planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)


_OPERANDS = (
    "mixed_qkv", "a", "b", "z", "A_log", "dt_bias", "norm_weight",
    "recurrent_state", "query_start_loc", "num_accepted_tokens", "state_indices",
    "num_seqs", "num_tokens", "output",
)
_QWEN_ALIGNED_OPERANDS = frozenset((
    "query_start_loc", "num_accepted_tokens", "num_seqs", "num_tokens", "norm_weight",
))
_KDA_ALIGNED_OPERANDS = frozenset(_OPERANDS) - {"z", "state_indices"}


def aligned_operands(kda):
    return _KDA_ALIGNED_OPERANDS if kda else _QWEN_ALIGNED_OPERANDS


def _default_kda_strides(key_heads, value_heads):
    return (
        (2 * key_heads + value_heads) * 128, value_heads * 128, 128,
        value_heads, 1, 128, value_heads * 128 * 128, 128 * 128, 128,
        value_heads * 128, 128,
    )


@dataclass(frozen=True, kw_only=True)
class GdnQuery:
    gate_activation: str
    qk_l2norm: bool
    state_dtype: str
    key_heads: int
    value_heads: int
    max_seqs: int
    max_tokens: int
    state_index_columns: int
    max_state_slots: int = 1
    null_state_index: int | None = None
    a_log_dtype: str = "float32"
    dt_bias_dtype: str | None = None
    norm_weight_dtype: str = "bfloat16"
    state_indices_dtype: str = "int32"
    kda_strides: tuple[int, ...] | None = None
    pointer_alignments: FrozenMapping | None = None
    recover_speculative_state: bool = False

    def __post_init__(self):
        kda = self.key_heads == self.value_heads
        if self.dt_bias_dtype is None:
            object.__setattr__(self, "dt_bias_dtype", "float32" if kda else "bfloat16")
        strides = self.kda_strides
        if kda:
            strides = _default_kda_strides(self.key_heads, self.value_heads) if strides is None else tuple(strides)
        object.__setattr__(self, "kda_strides", strides)
        alignments = {name: 16 for name in _OPERANDS}
        if self.pointer_alignments is not None:
            alignments.update(self.pointer_alignments)
        if any(type(value) is not int or value not in (1, 2, 4, 8, 16) for value in alignments.values()):
            raise ValueError("invalid GDN pointer alignment")
        # CuTe recurrence pointers and explicitly nonspecialized Triton pointers
        # retain their existing dynamic-alignment contract.
        relevant = aligned_operands(kda)
        for name in _OPERANDS:
            if name not in relevant:
                alignments[name] = 16
        object.__setattr__(self, "pointer_alignments", FrozenMapping(alignments))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, kw_only=True)
class GdnConfig:
    backend: str
    recurrent_block_v: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "GdnConfig":
        expected = {"backend", "recurrent_block_v"}
        if set(payload) != expected:
            raise ValueError(
                "GDN configs require exactly backend and recurrent_block_v"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("GDN backend must be a string")
        recurrent_block_v = payload["recurrent_block_v"]
        if not isinstance(recurrent_block_v, int) or isinstance(
            recurrent_block_v, bool
        ):
            raise TypeError("GDN recurrent_block_v must be an integer")
        return cls(
            backend=backend,
            recurrent_block_v=recurrent_block_v,
        )


def _default_config(
    query: GdnQuery,
    _device: DeviceIdentity | None,
) -> GdnConfig:
    backend = _backend(query)
    return GdnConfig(
        backend=backend,
        recurrent_block_v=32,
    )


def _validate_query(query: GdnQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, GdnQuery):
        raise TypeError("query must be GdnQuery")
    if any(type(value) is not int or value <= 0 for value in (
        query.key_heads, query.value_heads, query.max_seqs, query.max_tokens,
        query.state_index_columns, query.max_state_slots,
    )):
        raise ValueError("GDN dimensions must be positive integers")
    if query.state_index_columns > 8 or query.max_tokens > query.max_seqs * query.state_index_columns:
        raise ValueError("GDN token capacity must fit at most eight state columns per sequence")
    kda = query.key_heads == query.value_heads
    if type(query.recover_speculative_state) is not bool:
        raise TypeError("recover_speculative_state must be boolean")
    if query.recover_speculative_state and (
        not kda or query.state_dtype != "float32" or not query.qk_l2norm
    ):
        raise ValueError("KDA recovery requires equal heads, FP32 state and Q/K normalization")
    if not kda and query.value_heads != 3 * query.key_heads:
        raise ValueError("Qwen GDN requires three value heads per key head")
    if query.gate_activation not in ("silu", "sigmoid") or (kda and query.gate_activation != "sigmoid"):
        raise ValueError("KDA requires sigmoid gating; Qwen supports silu or sigmoid")
    if type(query.qk_l2norm) is not bool:
        raise TypeError("qk_l2norm must be boolean")
    if query.state_dtype not in ("bfloat16", "float32") or any(
        value not in ("bfloat16", "float32") for value in (
            query.a_log_dtype, query.dt_bias_dtype, query.norm_weight_dtype,
        )
    ):
        raise ValueError("unsupported GDN state or parameter dtype")
    if query.state_indices_dtype not in ("int32", "int64"):
        raise ValueError("GDN state indices require int32 or int64")
    if query.null_state_index is not None and (
        type(query.null_state_index) is not int or not -(1 << 63) <= query.null_state_index < (1 << 63)
    ):
        raise ValueError("null state index must fit int64")
    if kda:
        if len(query.kda_strides) != 11 or any(type(value) is not int or value < 0 for value in query.kda_strides):
            raise ValueError("KDA requires eleven nonnegative compile-time strides")
    elif query.kda_strides is not None:
        raise ValueError("Qwen recurrent strides remain dynamic")
    if set(query.pointer_alignments) != set(_OPERANDS) or any(
        type(value) is not int or value not in (1, 2, 4, 8, 16)
        for value in query.pointer_alignments.values()
    ):
        raise ValueError("GDN pointer metadata must describe every operand alignment")


def _validate_config(
    query: GdnQuery,
    config: GdnConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, GdnConfig):
        raise TypeError("config must be GdnConfig")
    expected_backend = _backend(query)
    if config.backend != expected_backend:
        raise ValueError(f"GDN recipe requires the {expected_backend} backend")
    if not isinstance(config.recurrent_block_v, int) or isinstance(
        config.recurrent_block_v, bool
    ):
        raise TypeError("GDN recurrent_block_v must be an integer")
    if config.recurrent_block_v not in {16, 32}:
        raise ValueError(
            f"GDN recurrent_block_v must be 16 or 32, got {config.recurrent_block_v}"
        )
    if query.key_heads != query.value_heads and config.recurrent_block_v != 32:
        raise ValueError("Qwen GDN requires recurrent_block_v=32")


def _backend(query: GdnQuery) -> str:
    return "triton" if query.key_heads == query.value_heads and not query.recover_speculative_state else "cutedsl"


def _tuning_parameters(query: GdnQuery, device):
    del device
    # Production dispatch is fixed by the equal-head KDA / grouped-head GDN recipe.
    return {"backend": (_backend(query),)}


# The state slot count sizes the caller's pool; it does not change which
# configuration is fastest, so it stays out of the selection key.
_KEY_FIELDS = frozenset(GdnQuery.__dataclass_fields__) - {"max_state_slots"}


def _encode_query(query: GdnQuery) -> dict[str, object]:
    return {name: value for name, value in query.to_dict().items() if name in _KEY_FIELDS}


TUNING = TuningContract(
    component_id="attention.gdn",
    query_schema_version=6,
    config_schema_version=4,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset({"backend", "recurrent_block_v"}),
    encode_query=_encode_query,
    encode_config=asdict,
    decode_config=GdnConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="backend", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="recurrent_block_v", values=(16, 32), binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=4,
    parameters=_tuning_parameters,
)


__all__ = ["GdnConfig", "GdnQuery", "TUNING"]
