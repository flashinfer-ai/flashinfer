"""Configuration contract for BF16 vocabulary projection preparation."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)

MAX_IN_FEATURES = 8_192
MIN_TRITON_OUT_FEATURES = 16_384
_TRITON_WARPS = frozenset((1, 2, 4, 8))
_LOOP_BLOCKS = frozenset((256, 512, 1_024))


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionQuery:
    dtype: str
    max_tokens: int
    in_features: int
    out_features: int


@dataclass(frozen=True, kw_only=True)
class Bf16VocabProjectionConfig:
    backend: str
    algorithm: str
    block_k: int
    num_warps: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "Bf16VocabProjectionConfig":
        expected = {"backend", "algorithm", "block_k", "num_warps"}
        if set(payload) != expected:
            raise ValueError(
                "BF16 vocabulary projection configs require backend, "
                "algorithm, block_k, and num_warps"
            )
        return cls(
            backend=str(payload["backend"]),
            algorithm=str(payload["algorithm"]),
            block_k=int(payload["block_k"]),
            num_warps=int(payload["num_warps"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "algorithm": self.algorithm,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
        }


def _encode(query: Bf16VocabProjectionQuery) -> dict[str, object]:
    return {
        name: getattr(query, name)
        for name in Bf16VocabProjectionQuery.__dataclass_fields__
    }


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _default_config(
    query: Bf16VocabProjectionQuery,
    device: DeviceIdentity | None,
) -> Bf16VocabProjectionConfig:
    supported_device = device is not None and device.compute_capability in {
        (12, 0),
        (12, 1),
    }
    if (
        supported_device
        and query.dtype == "bfloat16"
        and query.max_tokens == 1
        and 0 < query.in_features <= MAX_IN_FEATURES
        and query.out_features >= MIN_TRITON_OUT_FEATURES
    ):
        return Bf16VocabProjectionConfig(
            backend="triton",
            algorithm="row",
            block_k=_next_power_of_two(query.in_features),
            num_warps=8,
        )
    return Bf16VocabProjectionConfig(
        backend="torch",
        algorithm="torch",
        block_k=0,
        num_warps=0,
    )


def _validate_query(
    query: Bf16VocabProjectionQuery,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(query, Bf16VocabProjectionQuery):
        raise TypeError("query must be Bf16VocabProjectionQuery")
    if query.dtype != "bfloat16":
        raise ValueError(f"unsupported vocabulary projection dtype {query.dtype!r}")
    if query.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if query.in_features <= 0 or query.out_features <= 0:
        raise ValueError("projection dimensions must be positive")


def _validate_config(
    query: Bf16VocabProjectionQuery,
    config: Bf16VocabProjectionConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, Bf16VocabProjectionConfig):
        raise TypeError("config must be Bf16VocabProjectionConfig")
    if config.backend == "torch":
        if (config.algorithm, config.block_k, config.num_warps) != ("torch", 0, 0):
            raise ValueError("torch projection configs cannot carry Triton knobs")
        return
    if config.backend != "triton":
        raise ValueError(f"unsupported projection backend {config.backend!r}")
    if query.max_tokens != 1:
        raise ValueError("the Triton vocabulary GEMV requires max_tokens=1")
    if query.in_features > MAX_IN_FEATURES:
        raise ValueError(f"the Triton vocabulary GEMV supports K <= {MAX_IN_FEATURES}")
    if config.num_warps not in _TRITON_WARPS:
        raise ValueError(f"unsupported Triton warp count {config.num_warps}")
    if config.algorithm == "row":
        if (
            config.block_k < query.in_features
            or config.block_k > MAX_IN_FEATURES
            or config.block_k & (config.block_k - 1)
        ):
            raise ValueError("row block_k must be a covering power of two")
    elif config.algorithm == "loop":
        if config.block_k not in _LOOP_BLOCKS:
            raise ValueError(f"unsupported loop block_k {config.block_k}")
    else:
        raise ValueError(f"unsupported Triton algorithm {config.algorithm!r}")


def _tuning_parameters(query, device):
    row_blocks = tuple(
        1 << exponent
        for exponent in range(MAX_IN_FEATURES.bit_length())
        if 1 << exponent >= query.in_features
    )
    return {
        "backend": ("torch", "triton")
        if query.max_tokens == 1 and query.in_features <= MAX_IN_FEATURES
        else ("torch",),
        "block_k": tuple(sorted(set(row_blocks) | _LOOP_BLOCKS)),
    }


TUNING = TuningContract(
    component_id="gemm.bf16_vocab_projection",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(Bf16VocabProjectionQuery.__dataclass_fields__),
    config_fields=frozenset(Bf16VocabProjectionConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=Bf16VocabProjectionConfig.to_dict,
    decode_config=Bf16VocabProjectionConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    candidate_contract_version=2,
    knobs=(
        Knob(name="backend", values=("torch", "triton"), binding=ParameterBinding.COMPILE),
        Knob(name="algorithm", values=("row", "loop"), binding=ParameterBinding.COMPILE,
             when=FrozenMapping({"backend": "triton"}), otherwise="torch"),
        Knob(name="block_k", values=None, binding=ParameterBinding.COMPILE,
             when=FrozenMapping({"backend": "triton"}), otherwise=0),
        Knob(name="num_warps", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE,
             when=FrozenMapping({"backend": "triton"}), otherwise=0),
    ),
    parameters=_tuning_parameters,
)


__all__ = [
    "TUNING",
    "Bf16VocabProjectionConfig",
    "Bf16VocabProjectionQuery",
    "MAX_IN_FEATURES",
    "MIN_TRITON_OUT_FEATURES",
]
