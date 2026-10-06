"""Immutable configuration contract for prepared compressed sparse MLA."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    TuningContract,
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaQuery:
    """All compile-identity metadata for the DSV4 compressed MLA path.

    Per-request sequence lengths, selected page IDs, and their contents remain
    dynamic inputs; capacity, storage layout, operand form, and numerical output
    contract do not.
    """

    cache_format: str
    layout: str
    mode: str
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    query_rows: int
    max_batch: int
    max_kv_rows: int
    max_page_table_width: int
    max_q_chunks: int | None
    decode_row_capacity: int | None
    use_cuda_graph: bool
    q_shape: tuple[int, ...]
    q_stride: tuple[int, ...]
    q_alignment: int
    swa_cache_shape: tuple[int, ...]
    swa_cache_stride: tuple[int, ...]
    swa_cache_alignment: int
    indexed_cache_present: bool
    indexed_cache_shape: tuple[int, ...] | None
    indexed_cache_stride: tuple[int, ...] | None
    indexed_cache_alignment: int | None
    attn_sink_present: bool
    return_lse: bool
    lse_scale: str
    output_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class SparseMlaConfig:
    """Prepared split/merge capacity selected for this immutable query."""

    max_chunks_per_row: int
    split_chunk_size: int
    single_pass: bool
    v41_compute_mode: str = "fp8"
    v41_heads_per_block: int = 16

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "SparseMlaConfig":
        expected = {
            "max_chunks_per_row",
            "split_chunk_size",
            "single_pass",
            "v41_compute_mode",
            "v41_heads_per_block",
        }
        if set(payload) != expected:
            raise ValueError("compressed MLA configs require exact split and V4.1 fields")
        chunks = payload["max_chunks_per_row"]
        chunk_size = payload["split_chunk_size"]
        single_pass = payload["single_pass"]
        compute_mode = payload["v41_compute_mode"]
        heads_per_block = payload["v41_heads_per_block"]
        if (
            type(chunks) is not int
            or type(chunk_size) is not int
            or type(single_pass) is not bool
            or type(compute_mode) is not str
            or type(heads_per_block) is not int
        ):
            raise TypeError("compressed MLA config fields have invalid types")
        return cls(
            max_chunks_per_row=chunks,
            split_chunk_size=chunk_size,
            single_pass=single_pass,
            v41_compute_mode=compute_mode,
            v41_heads_per_block=heads_per_block,
        )


def _single_pass(query: SparseMlaQuery, device: DeviceIdentity | None) -> bool:
    from b12x.attention._shared.mla.compressed_api import _should_use_sm121_single_pass_decode

    if query.mode in ("extend", "verify", "draft_extend"):
        return True
    return query.cache_format == "deepseek_v4" and _should_use_sm121_single_pass_decode(
        rows=query.query_rows,
        heads=query.num_q_heads,
        swa_width=query.swa_width,
        indexed_width=query.indexed_width,
        swa_page_size=query.swa_page_size,
        indexed_page_size=query.indexed_page_size if query.indexed_cache_present else None,
        compute_capability=(0, 0) if device is None else device.compute_capability,
    )


def _split_config(query: SparseMlaQuery, max_chunks: int):
    from b12x.attention._shared.mla.compressed_config import compressed_sparse_mla_split_config_for_contract

    return compressed_sparse_mla_split_config_for_contract(
        rows=query.query_rows,
        width=query.swa_width + query.indexed_width,
        max_chunks=max_chunks,
        decode_row_capacity=query.decode_row_capacity,
    )


def _default_config(query: SparseMlaQuery, device: DeviceIdentity | None) -> SparseMlaConfig:
    single_pass = _single_pass(query, device)
    split = (
        None
        if single_pass
        else _split_config(query, 256)
    )
    return SparseMlaConfig(
        max_chunks_per_row=1 if split is None else split.num_chunks,
        split_chunk_size=1 if split is None else split.chunk_size,
        single_pass=single_pass,
        v41_compute_mode="fp8",
        v41_heads_per_block=(8 if query.cache_format == "deepseek_v41" and not single_pass
                             and query.num_q_heads % 16 else 16),
    )


def _validate_query(query: SparseMlaQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, SparseMlaQuery):
        raise TypeError("query must be SparseMlaQuery")
    if query.layout != "compressed_dsv4":
        raise ValueError("compressed MLA requires the compressed_dsv4 layout")
    if query.cache_format not in ("deepseek_v4", "deepseek_v41"):
        raise ValueError("unsupported compressed MLA cache_format")
    if query.mode not in ("decode", "extend", "verify", "draft_extend"):
        raise ValueError("unsupported compressed MLA mode")
    expected_kv_dtype = "uint8" if query.cache_format == "deepseek_v41" else None
    if query.q_dtype != "bfloat16" or (
        query.kv_dtype not in ("uint8", "float8_e4m3fn")
        or (expected_kv_dtype is not None and query.kv_dtype != expected_kv_dtype)
    ):
        raise TypeError("compressed MLA requires BF16 q and its declared KV recipe")
    if query.num_q_heads <= 0 or query.query_rows <= 0 or query.max_batch <= 0:
        raise ValueError("compressed MLA capacities must be positive")
    if query.swa_width < 0 or query.indexed_width < 0:
        raise ValueError("compressed MLA widths must be nonnegative")
    if query.swa_page_size <= 0 or query.indexed_page_size <= 0:
        raise ValueError("compressed MLA page sizes must be positive")
    if not query.indexed_cache_present and (query.indexed_cache_shape is not None or query.indexed_cache_stride is not None):
        raise ValueError("absent indexed cache cannot carry storage metadata")
    if query.lse_scale not in ("base2", "natural") or query.output_mode not in ("internal", "provided"):
        raise ValueError("invalid compressed MLA numerical/output contract")
    if len(query.q_shape) not in (3, 4) or len(query.q_shape) != len(query.q_stride):
        raise ValueError("q metadata must retain its rank and strides")
    if query.q_alignment <= 0 or query.swa_cache_alignment <= 0:
        raise ValueError("compressed MLA tensor alignments must be positive")
    if query.q_alignment % 16:
        raise ValueError("compressed MLA Q requires 16-byte alignment")


def _validate_config(
    query: SparseMlaQuery, config: SparseMlaConfig, device: DeviceIdentity | None
) -> None:
    from b12x.attention._shared.mla.compressed_config import (
        _COMPRESSED_SPARSE_MLA_SPLIT_MAX_CHUNKS,
    )

    if not isinstance(config, SparseMlaConfig):
        raise TypeError("config must be SparseMlaConfig")
    if not 1 <= config.max_chunks_per_row <= _COMPRESSED_SPARSE_MLA_SPLIT_MAX_CHUNKS:
        raise ValueError("compressed MLA split capacity exceeds the production limit")
    if config.split_chunk_size <= 0:
        raise ValueError("compressed MLA split chunk size must be positive")
    if config.single_pass != _single_pass(query, device):
        raise ValueError("compressed MLA config path does not match the declared query")
    if config.v41_compute_mode not in ("bf16", "fp8"):
        raise ValueError("v41_compute_mode must be 'bf16' or 'fp8'")
    if config.v41_heads_per_block not in (8, 16):
        raise ValueError("v41_heads_per_block must be 8 or 16")
    fp8_decode = query.cache_format == "deepseek_v41" and not config.single_pass and config.v41_compute_mode == "fp8"
    if not fp8_decode and config.v41_heads_per_block != 16:
        raise ValueError("head grouping only configures V4.1 FP8 decode")
    if fp8_decode and config.v41_heads_per_block == 16 and query.num_q_heads % 16:
        raise ValueError("V4.1 H16 decode requires complete 16-head groups")
    if query.cache_format == "deepseek_v4" and config.v41_compute_mode != "fp8":
        raise ValueError("V4.1 precision controls require the V4.1 cache format")
    if config.single_pass:
        if config.max_chunks_per_row != 1 or config.split_chunk_size != 1:
            raise ValueError("single-pass compressed MLA has no split workspace")
        return
    split = _split_config(query, config.max_chunks_per_row)
    if split.num_chunks > config.max_chunks_per_row or split.chunk_size != config.split_chunk_size:
        raise ValueError("compressed MLA config is inconsistent with the declared split contract")


def _parameters(
    query: SparseMlaQuery, device: DeviceIdentity | None
) -> dict[str, tuple[object, ...]]:
    default = _default_config(query, device)
    if query.cache_format == "deepseek_v4":
        modes = (default.v41_compute_mode,)
        head_blocks = (default.v41_heads_per_block,)
    else:
        modes = ("fp8", "bf16")
        head_blocks = (8, 16)
    return {
        "max_chunks_per_row": (default.max_chunks_per_row,),
        "split_chunk_size": (default.split_chunk_size,),
        "single_pass": (default.single_pass,),
        "v41_compute_mode": modes,
        "v41_heads_per_block": head_blocks,
    }


def _materialize(
    query: SparseMlaQuery, device: DeviceIdentity | None, choice: FrozenMapping
) -> SparseMlaConfig:
    config = SparseMlaConfig.from_config(choice)
    _validate_config(query, config, device)
    return config


TUNING = TuningContract(
    component_id="attention.compressed_sparse_mla",
    query_schema_version=5,
    config_schema_version=4,
    query_fields=frozenset(SparseMlaQuery.__dataclass_fields__),
    config_fields=frozenset(SparseMlaConfig.__dataclass_fields__),
    encode_query=SparseMlaQuery.to_dict,
    encode_config=asdict,
    decode_config=SparseMlaConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=_default_config,
    knobs=(
        Knob(name="max_chunks_per_row", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="split_chunk_size", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="single_pass", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="v41_compute_mode", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="v41_heads_per_block", values=None, binding=ParameterBinding.COMPILE),
    ),
    candidate_contract_version=5,
    parameters=_parameters,
    materialize=_materialize,
    equivalence_key=lambda query, _device, config: (
        config.max_chunks_per_row,
        config.split_chunk_size,
        config.single_pass,
        (config.v41_compute_mode, config.v41_heads_per_block)
        if query.cache_format == "deepseek_v41"
        else None,
    ),
)


__all__ = ["SparseMlaConfig", "SparseMlaQuery", "TUNING"]
