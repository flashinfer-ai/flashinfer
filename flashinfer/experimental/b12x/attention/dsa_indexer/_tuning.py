"""Preparation tuning contract for native DSA indexer execution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._efficiency import capture_exhaustive_search
from b12x.preparation.tuning import Knob, ParameterBinding, ParameterSpace, TuningContract

FUSED_MERGE_AUTO = "auto"
FUSED_MERGE_COOPERATIVE = "cooperative"
FUSED_MERGE_SERIAL = "serial"
FUSED_MERGE_CHOICES = (FUSED_MERGE_AUTO, FUSED_MERGE_COOPERATIVE, FUSED_MERGE_SERIAL)
_BACKEND = "native"


@dataclass(frozen=True, kw_only=True)
class DsaIndexerQuery:
    source_layout: str
    mode: str
    dtype: str
    kv_dtype: str
    num_q_heads: int
    num_idx_heads: int
    max_q_rows: int
    max_k_rows: int
    top_k: int
    page_size: int
    score_mode: str
    shared_page_table: bool
    max_page_table_width: int
    route: str
    output_physical_slots: bool
    supertile_k: int
    prefill_block_k: int
    reserve_paged_logits: bool
    paged_logits_k_rows: int
    operands: FrozenMapping
    cache_format: str = "fp8"
    max_candidates: int = 0
    candidate_topk_blocks: int = 0
    exhaustive: bool = field(default_factory=capture_exhaustive_search)


@dataclass(frozen=True, kw_only=True)
class DsaIndexerConfig:
    backend: str
    fused_merge: str = FUSED_MERGE_AUTO
    mxfp4_score_kind: str | None = None

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "DsaIndexerConfig":
        if set(payload) != {"backend", "fused_merge", "mxfp4_score_kind"}:
            raise ValueError("DSA indexer config requires backend, fused_merge and mxfp4_score_kind")
        return cls(backend=payload["backend"], fused_merge=payload["fused_merge"], mxfp4_score_kind=payload["mxfp4_score_kind"])

    def to_dict(self) -> dict[str, object]:
        return {"backend": self.backend, "fused_merge": self.fused_merge, "mxfp4_score_kind": self.mxfp4_score_kind}


def _encode_query(query: DsaIndexerQuery) -> dict[str, object]:
    if not isinstance(query, DsaIndexerQuery):
        raise TypeError("query must be DsaIndexerQuery")
    return {name: getattr(query, name) for name in DsaIndexerQuery.__dataclass_fields__}


def _encode_config(config: DsaIndexerConfig) -> dict[str, object]:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    return config.to_dict()


def _validate_query(query: DsaIndexerQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, DsaIndexerQuery):
        raise TypeError("query must be DsaIndexerQuery")
    if query.source_layout not in ("paged", "contiguous"):
        raise ValueError("unsupported DSA source layout")
    if query.mode not in ("decode", "prefill"):
        raise ValueError("unsupported DSA mode")
    if query.route not in ("auto", "paged_fused", "paged_tiled", "packed_contiguous"):
        raise ValueError("unsupported DSA route")
    if query.page_size <= 0 or query.max_q_rows <= 0 or query.top_k <= 0:
        raise ValueError("DSA capacities must be positive")
    if not isinstance(query.operands, FrozenMapping):
        raise TypeError("DSA declarations require immutable operand metadata")
    if query.cache_format not in ("fp8", "mxfp4"):
        raise ValueError("unsupported DSA cache format")
    if query.cache_format == "mxfp4":
        if query.top_k != 512 or query.page_size <= 0:
            raise ValueError("MXFP4 requires logical top-k=512 and a positive page size")
        if query.num_q_heads > 32 or 32 % query.num_q_heads:
            raise ValueError("MXFP4 index heads must divide 32")
        if query.max_candidates and query.candidate_topk_blocks:
            raise ValueError("MXFP4 source and reindex candidates are exclusive")


def _validate_config(query, config: DsaIndexerConfig, _device) -> None:
    if not isinstance(config, DsaIndexerConfig):
        raise TypeError("config must be DsaIndexerConfig")
    if config.backend != _BACKEND:
        raise ValueError(f"unsupported attention.dsa_indexer backend {config.backend!r}")
    if config.fused_merge not in FUSED_MERGE_CHOICES:
        raise ValueError(f"unsupported fused_merge {config.fused_merge!r}")

    if query.cache_format == "mxfp4":
        if config.mxfp4_score_kind not in ("score", "score_tensorcore"):
            raise ValueError("MXFP4 requires scalar or tensor-core scoring")
        if config.fused_merge != FUSED_MERGE_AUTO:
            raise ValueError("MXFP4 does not use the FP8 fused merge")
    elif config.mxfp4_score_kind is not None:
        raise ValueError("MXFP4 score selection requires the MXFP4 cache recipe")


def _default(query, _device) -> DsaIndexerConfig:
    score_kind = None
    if query.cache_format == "mxfp4":
        score_kind = "score_tensorcore" if query.mode == "prefill" or query.num_q_heads == 32 else "score"
    return DsaIndexerConfig(backend=_BACKEND, fused_merge=FUSED_MERGE_AUTO, mxfp4_score_kind=score_kind)


def _tuning_ctas(query: DsaIndexerQuery, device: DeviceIdentity | None) -> int:
    from .fused_indexer import resolve_fused_indexer_path
    from .kernel import _num_q_head_tiles
    if query.source_layout != "paged" or query.route in ("paged_tiled", "packed_contiguous"):
        return 0
    if query.shared_page_table or query.mode == "prefill":
        if query.route == "paged_fused":
            raise ValueError("fused paged DSA is decode-only")
        return 0
    if device is None:
        raise ValueError("paged DSA tuning requires a device identity")
    supported = (os.getenv("B12X_FUSED_INDEXER", "1") != "0" and resolve_fused_indexer_path(
        topk=query.top_k, num_rows=query.max_q_rows,
        width=query.max_page_table_width * query.page_size,
        num_heads=query.num_q_heads, compute_capability=device.compute_capability,
    ) and _num_q_head_tiles(query.num_q_heads) in (1, 2, 4))
    if not supported:
        if query.route == "paged_fused":
            raise ValueError("fused paged DSA is unavailable for this query")
        return 0
    return max(1, min(query.max_page_table_width, device.sm_count // max(1, query.max_q_rows)))


def _equivalence(query, device, config):
    if query.cache_format == "mxfp4":
        return config.mxfp4_score_kind
    from .fused_indexer import resolve_fused_merge_threshold
    return resolve_fused_merge_threshold(config.fused_merge, ctas_per_group=_tuning_ctas(query, device), num_heads=query.num_q_heads, topk=query.top_k)


def _parameters(query, _device):
    tensorcore_prefill = (
        query.cache_format == "mxfp4" and query.mode == "prefill"
        and query.num_q_heads == 32 and query.max_q_rows >= 64
    )
    return ParameterSpace.create(
        TUNING.knobs,
        values={
            "backend": (_BACKEND,),
            "fused_merge": (FUSED_MERGE_AUTO,) if query.cache_format == "mxfp4" else FUSED_MERGE_CHOICES,
            "mxfp4_score_kind": ("score", "score_tensorcore") if query.cache_format == "mxfp4" else (None,),
        },
        exhaustive=query.exhaustive,
        efficiency_predicates=(lambda p: not tensorcore_prefill or p["mxfp4_score_kind"] != "score",),
    )


TUNING = TuningContract(
    component_id="attention.dsa_indexer", query_schema_version=4, config_schema_version=3,
    query_fields=frozenset(DsaIndexerQuery.__dataclass_fields__),
    config_fields=frozenset(DsaIndexerConfig.__dataclass_fields__),
    encode_query=_encode_query, encode_config=_encode_config, decode_config=DsaIndexerConfig.from_config,
    validate_query=_validate_query, validate_config=_validate_config, default_config=_default,
    knobs=(
        Knob(name="backend", values=(_BACKEND,), binding=ParameterBinding.COMPILE),
        Knob(name="fused_merge", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="mxfp4_score_kind", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=4, equivalence_key=_equivalence, parameters=_parameters,
    materialize=lambda query, device, choice: DsaIndexerConfig.from_config(choice),
)

__all__ = ["DsaIndexerConfig", "DsaIndexerQuery", "TUNING", "FUSED_MERGE_AUTO", "FUSED_MERGE_CHOICES", "FUSED_MERGE_COOPERATIVE", "FUSED_MERGE_SERIAL"]
