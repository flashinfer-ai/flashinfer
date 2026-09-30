"""Typed tuning contract for Qwen grouped-selector sparse GQA."""

from __future__ import annotations

from dataclasses import dataclass

from ._sparse_gqa_cute_config import (
    BLOCK_N as _QWEN_CUTE_BLOCK_N,
    NUM_SPLITS as _QWEN_CUTE_NUM_SPLITS,
    is_qwen_geometry,
)
from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class QsaQuery:
    q_dtype: str
    kv_dtype: str
    q_heads: int
    kv_heads: int
    head_dim: int
    index_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_rotary_dim: int
    main_page_size: int
    max_batch: int
    max_q_rows: int
    max_seq_len: int
    max_speculative_tokens: int
    compress_ratio: int
    budget: int
    position_axes: int
    mrope_interleaved: bool
    max_raw_state_slots: int
    num_main_cache_pages: int
    num_compressed_cache_pages: int
    compressed_page_size: int
    mrope_sections: tuple[int, int, int] | None
    rms_norm_eps: float
    abi: FrozenMapping
    dcp_size: int = 1
    dcp_rank: int = 0
    cp_kv_cache_interleave_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "abi", FrozenMapping(self.abi))

    def to_dict(self) -> dict[str, object]:
        return {
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "index_heads": self.index_heads,
            "index_kv_heads": self.index_kv_heads,
            "index_head_dim": self.index_head_dim,
            "index_rotary_dim": self.index_rotary_dim,
            "main_page_size": self.main_page_size,
            "max_batch": self.max_batch,
            "max_q_rows": self.max_q_rows,
            "max_seq_len": self.max_seq_len,
            "max_speculative_tokens": self.max_speculative_tokens,
            "compress_ratio": self.compress_ratio,
            "budget": self.budget,
            "position_axes": self.position_axes,
            "mrope_interleaved": self.mrope_interleaved,
            "max_raw_state_slots": self.max_raw_state_slots,
            "num_main_cache_pages": self.num_main_cache_pages,
            "num_compressed_cache_pages": self.num_compressed_cache_pages,
            "compressed_page_size": self.compressed_page_size,
            "mrope_sections": self.mrope_sections,
            "rms_norm_eps": self.rms_norm_eps,
            "dcp_size": self.dcp_size,
            "dcp_rank": self.dcp_rank,
            "cp_kv_cache_interleave_size": self.cp_kv_cache_interleave_size,
            "abi": self.abi,
        }

@dataclass(frozen=True, kw_only=True)
class QsaConfig:
    backend: str
    sparse_gqa_direct_kv_warps: int = 2

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "QsaConfig":
        if set(payload) != {"backend", "sparse_gqa_direct_kv_warps"}:
            raise ValueError(
                "QSA configs require exactly backend and sparse_gqa_direct_kv_warps"
            )
        backend = payload["backend"]
        if not isinstance(backend, str):
            raise TypeError("QSA backend must be a string")
        direct_kv_warps = payload["sparse_gqa_direct_kv_warps"]
        if not isinstance(direct_kv_warps, int) or isinstance(direct_kv_warps, bool):
            raise TypeError("QSA sparse_gqa_direct_kv_warps must be an integer")
        return cls(
            backend=backend,
            sparse_gqa_direct_kv_warps=direct_kv_warps,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sparse_gqa_direct_kv_warps": self.sparse_gqa_direct_kv_warps,
        }


def _default_config(
    _query: QsaQuery,
    _device: DeviceIdentity | None,
) -> QsaConfig:
    return QsaConfig(
        backend="cutedsl",
        sparse_gqa_direct_kv_warps=2,
    )


def _validate_query(
    query: QsaQuery,
    _device: DeviceIdentity | None,
) -> None:
    if query.q_dtype != "bfloat16":
        raise ValueError("QSA requires BF16 queries")
    if query.kv_dtype not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("QSA requires BF16 or FP8 E4M3 KV storage")
    selection_width = int(query.budget) + int(query.compress_ratio) - 1
    if not is_qwen_geometry(
        q_heads=int(query.q_heads),
        kv_heads=int(query.kv_heads),
        head_dim=int(query.head_dim),
        selection_width=selection_width,
        block_n=_QWEN_CUTE_BLOCK_N,
        splits=_QWEN_CUTE_NUM_SPLITS,
    ):
        raise ValueError(
            "QSA requires the native CuTe sparse-GQA geometry: positive "
            "q_heads divisible by kv_heads, head_dim=256, and "
            "selection_width>=2051"
        )
    if query.head_dim != 256 or query.index_head_dim != 128:
        raise ValueError("unsupported QSA head dimensions")
    if query.index_heads <= 0 or query.index_kv_heads != 1:
        raise ValueError("unsupported QSA index-head layout")
    if query.index_rotary_dim <= 0 or query.index_rotary_dim > query.index_head_dim:
        raise ValueError("unsupported QSA index rotary dimension")
    if (
        min(
            query.main_page_size,
            query.max_batch,
            query.max_q_rows,
            query.max_seq_len,
            query.compress_ratio,
            query.max_raw_state_slots,
            query.num_main_cache_pages,
            query.num_compressed_cache_pages,
            query.compressed_page_size,
        )
        <= 0
    ):
        raise ValueError("QSA profile geometry must be positive")
    if query.max_q_rows < query.max_batch:
        raise ValueError("QSA max_q_rows must cover max_batch")
    if query.max_speculative_tokens < 0:
        raise ValueError("QSA speculative-token capacity must be nonnegative")
    from ._dcp import validate_geometry

    validate_geometry(
        size=query.dcp_size,
        rank=query.dcp_rank,
        token_interleave=query.cp_kv_cache_interleave_size,
        compress_ratio=query.compress_ratio,
    )
    if query.position_axes not in (1, 3):
        raise ValueError("QSA position_axes must be 1 or 3")
    if query.position_axes == 1 and query.mrope_interleaved:
        raise ValueError("scalar-position QSA cannot use interleaved M-RoPE")
    if query.mrope_sections is not None and (
        len(query.mrope_sections) != 3
        or any(type(section) is not int or section <= 0 for section in query.mrope_sections)
    ):
        raise ValueError("QSA M-RoPE sections must contain three positive integers")
    operands = query.abi.get("operands")
    expected_ranks = {
        "request_ids": 1, "rope_positions": 2, "index_query": 3,
        "raw_index_key": 2, "main_k_cache": 4, "main_v_cache": 4,
        "main_block_table": 2, "compressed_k_cache": 3,
        "compressed_block_table": 2,
        "raw_k_ring": 3, "raw_logical_positions": 2, "raw_rope_positions": 3,
        "raw_interval_start_positions": 1, "raw_state_slot_ids": 1,
        "index_q_norm_weight": 1, "index_k_norm_weight": 1,
        "rope_cos": 2, "rope_sin": 2,
    }
    if not isinstance(operands, FrozenMapping) or set(operands) != set(expected_ranks):
        raise ValueError("QSA query requires complete normalized ABI metadata")
    for name, rank in expected_ranks.items():
        descriptor = operands[name]
        if (
            not isinstance(descriptor, FrozenMapping)
            or set(descriptor) != {"dtype", "strides"}
            or not isinstance(descriptor["dtype"], str)
            or len(tuple(descriptor["strides"])) != rank
            or any(type(stride) is not int or stride <= 0 for stride in descriptor["strides"])
        ):
            raise ValueError(f"QSA {name} ABI metadata is invalid")
    expected_dtypes = {
        "request_ids": ("int32", "int64"),
        "rope_positions": ("int64",),
        "index_query": ("bfloat16",),
        "raw_index_key": ("bfloat16",),
        "main_k_cache": (query.kv_dtype,),
        "main_v_cache": (query.kv_dtype,),
        "main_block_table": ("int32",),
        "compressed_k_cache": (query.q_dtype,),
        "compressed_block_table": ("int32",),
        "raw_k_ring": ("bfloat16",),
        "raw_logical_positions": ("int64",),
        "raw_rope_positions": ("int64",),
        "raw_interval_start_positions": ("int64",),
        "raw_state_slot_ids": ("int32", "int64"),
        "index_q_norm_weight": ("bfloat16", "float32"),
        "index_k_norm_weight": ("bfloat16", "float32"),
        "rope_cos": ("bfloat16", "float32"),
        "rope_sin": ("bfloat16", "float32"),
    }
    for name, dtypes in expected_dtypes.items():
        if operands[name]["dtype"] not in dtypes:
            raise ValueError(f"QSA {name} ABI dtype is unsupported")
    if not isinstance(query.rms_norm_eps, float) or not query.rms_norm_eps > 0:
        raise ValueError("QSA RMS normalization epsilon must be positive")


def _validate_config(
    _query: QsaQuery,
    config: QsaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if config.backend != "cutedsl":
        raise ValueError(f"unsupported QSA backend {config.backend!r}")
    if (
        not isinstance(config.sparse_gqa_direct_kv_warps, int)
        or isinstance(config.sparse_gqa_direct_kv_warps, bool)
        or config.sparse_gqa_direct_kv_warps not in (1, 2, 4)
    ):
        raise ValueError("QSA sparse_gqa_direct_kv_warps must be 1, 2, or 4")


# The cache page counts size the caller's pools; they do not change which
# configuration is fastest, so they stay out of the selection key.
_KEY_FIELDS = frozenset(QsaQuery.__dataclass_fields__) - {
    "num_main_cache_pages", "num_compressed_cache_pages",
}


def _encode_query(query: QsaQuery) -> dict[str, object]:
    return {name: value for name, value in query.to_dict().items() if name in _KEY_FIELDS}


TUNING = TuningContract(
    component_id="attention.qsa",
    query_schema_version=7,
    config_schema_version=2,
    query_fields=_KEY_FIELDS,
    config_fields=frozenset(QsaConfig.__dataclass_fields__),
    encode_query=_encode_query,
    encode_config=QsaConfig.to_dict,
    decode_config=QsaConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="backend", values=("cutedsl",), binding=ParameterBinding.COMPILE),
        Knob(
            name="sparse_gqa_direct_kv_warps",
            values=(2,),
            binding=ParameterBinding.COMPILE,
        ),
    ),
    candidate_contract_version=4,
)


__all__ = ["QsaConfig", "QsaQuery", "TUNING"]
