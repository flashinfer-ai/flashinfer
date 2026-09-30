"""Reviewed serving-shape fixtures for attention correctness and benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .cases import WorkloadCase

COMMON_SEQUENCE_CAPACITIES = (*range(1, 17), 32, 64, 128, 256)
COMMON_PREFILL_TOKEN_CAPACITIES = (1_024, 2_048, 4_096, 8_192)
COMMON_BATCHES = COMMON_SEQUENCE_CAPACITIES
COMMON_CONTEXT_TOKENS = (128, 16_384, 32_768, 65_536, 131_072)
COMMON_PAGE_SIZES = (64, 128)
COMMON_KV_DTYPES = ("bfloat16", "float8_e4m3fn")
GDN_STATE_INDEX_COLUMNS = tuple(range(1, 9))
QSA_BATCHES = (4,)
QSA_CONTEXT_TOKENS = (2_048, 32_768, 262_144)
QSA_PAGE_SIZES = (16, 64, 1_504, 3_008)
QSA_SPECULATIVE_CONTEXT_TOKENS = (32_768, 262_144)
QSA_POSITION_LAYOUTS = ((3, True),)
QSA_DIRECT_PREFILL_ROWS = {
    2_048: (65, 128, 1_024, 2_048),
    32_768: (65, 128, 1_024, 4_096, 6_016, 8_192),
    262_144: (65, 128, 1_024, 4_096, 6_016, 8_192),
}


@dataclass(frozen=True, kw_only=True)
class AttentionBenchmarkPreset:
    preset_id: str
    component: str
    model_id: str
    source: str

    def __post_init__(self) -> None:
        if not self.preset_id or not self.component or not self.model_id:
            raise ValueError("attention benchmark preset fields must be non-empty")
        if not self.source:
            raise ValueError("attention benchmark presets require a source")


@dataclass(frozen=True, kw_only=True)
class GdnGeometry:
    model_id: str
    key_heads: int
    value_heads: int
    query_lengths: tuple[int, ...]
    source: str
    state_dtype: str = "float32"
    decay_recipe: str = "gdn"

    def __post_init__(self) -> None:
        if not self.model_id or not self.source or not self.query_lengths:
            raise ValueError("GDN geometry labels and query lengths are required")
        if self.key_heads <= 0 or self.value_heads <= 0:
            raise ValueError("GDN head counts must be positive")
        if any(length <= 0 for length in self.query_lengths):
            raise ValueError("GDN query lengths must be positive")
        if max(GDN_STATE_INDEX_COLUMNS) < max(self.query_lengths):
            raise ValueError("GDN column capacity must cover every query length")
        if max(COMMON_SEQUENCE_CAPACITIES) < len(self.query_lengths):
            raise ValueError("GDN sequence capacity must cover the live batch")
        if self.decay_recipe not in {"gdn", "kda"}:
            raise ValueError("GDN decay recipe must be 'gdn' or 'kda'")
        if self.decay_recipe == "gdn" and self.value_heads != 3 * self.key_heads:
            raise ValueError("GDN geometries require three value heads per key head")
        if self.decay_recipe == "kda" and self.value_heads != self.key_heads:
            raise ValueError("KDA geometries require equal key and value heads")


GDN_GEOMETRIES = (
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk16-v48-decode-bs1",
        query_lengths=(1,),
        key_heads=16,
        value_heads=48,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk16-v48-spec4-bs1",
        query_lengths=(4,),
        key_heads=16,
        value_heads=48,
        source="Qwen3.8 Flash Next TP1 MTP serving capacity",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-decode-bs1",
        query_lengths=(1,),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-decode-bs4",
        query_lengths=(1, 1, 1, 1),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec2-bs4",
        query_lengths=(2, 2, 2, 2),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-bs1",
        query_lengths=(4,),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-uneven",
        query_lengths=(4, 2, 1, 3),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk8-v24-spec4-bs4",
        query_lengths=(4, 4, 4, 4),
        key_heads=8,
        value_heads=24,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk4-v12-decode-bs1",
        query_lengths=(1,),
        key_heads=4,
        value_heads=12,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk4-v12-spec4-bs1",
        query_lengths=(4,),
        key_heads=4,
        value_heads=12,
        source="Qwen3.8 Flash Next TP4 MTP serving capacity",
    ),
    GdnGeometry(
        model_id="qwen3.8-flash-next-qk2-v6-decode-bs1",
        query_lengths=(1,),
        key_heads=2,
        value_heads=6,
        source="benchmark_gdn_decode.QWEN38_GDN_CASES",
    ),
    *(
        GdnGeometry(
            model_id=(
                f"glm-5.3-flash-kda-tp{tp_size}-"
                f"{'decode' if query_lengths == (1,) else 'spec6'}-bs1"
            ),
            query_lengths=query_lengths,
            key_heads=64 // tp_size,
            value_heads=64 // tp_size,
            source="GLM-5.3 Flash KDA serving geometry",
            decay_recipe="kda",
        )
        for tp_size in (1, 2, 4, 8, 16)
        for query_lengths in ((1,), (6,))
    ),
)


@dataclass(frozen=True, kw_only=True)
class GqaGeometry:
    model_id: str
    q_heads: int
    kv_heads: int
    head_dim: int
    source: str
    kv_dtypes: tuple[str, ...] = COMMON_KV_DTYPES
    page_sizes: tuple[int, ...] = COMMON_PAGE_SIZES
    batch_sizes: tuple[int, ...] = COMMON_BATCHES
    context_tokens: tuple[int, ...] = COMMON_CONTEXT_TOKENS
    cache_layouts: tuple[str, ...] = ("separate", "combined")

    def __post_init__(self) -> None:
        if not self.model_id or not self.source:
            raise ValueError("GQA geometry labels are required")
        if min(self.q_heads, self.kv_heads, self.head_dim) <= 0:
            raise ValueError("GQA geometry values must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("GQA query heads must be divisible by KV heads")
        if not all(
            (
                self.kv_dtypes,
                self.page_sizes,
                self.batch_sizes,
                self.context_tokens,
                self.cache_layouts,
            )
        ):
            raise ValueError("GQA sweep axes must be non-empty")


def _tp_sliced_gqa_geometry(
    *,
    model_id: str,
    q_heads: int,
    kv_heads: int,
    source: str,
) -> GqaGeometry:
    return GqaGeometry(
        model_id=model_id,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=256,
        source=source,
    )


GQA_GEOMETRIES = (
    GqaGeometry(
        model_id="qwen3.8-flash-next-180b",
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        source="Qwen3.8 Flash Next text_config",
    ),
    _tp_sliced_gqa_geometry(
        model_id="qwen3.8-flash-next-180b-tp2",
        q_heads=12,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
    _tp_sliced_gqa_geometry(
        model_id="qwen3.8-flash-next-180b-tp4",
        q_heads=6,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
    GqaGeometry(
        model_id="qwen3.8-27b",
        q_heads=24,
        kv_heads=4,
        head_dim=256,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    _tp_sliced_gqa_geometry(
        model_id="qwen3.8-27b-tp2",
        q_heads=12,
        kv_heads=2,
        source="Qwen3.8 27B tensor-parallel slicing",
    ),
    _tp_sliced_gqa_geometry(
        model_id="qwen3.8-27b-tp8",
        q_heads=3,
        kv_heads=1,
        source="Qwen3.8 27B tensor-parallel slicing",
    ),
    GqaGeometry(
        model_id="qwen-gqa",
        q_heads=8,
        kv_heads=1,
        head_dim=256,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    GqaGeometry(
        model_id="minimax-m2.7",
        q_heads=24,
        kv_heads=4,
        head_dim=128,
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    *(
        GqaGeometry(
            model_id=f"minimax-m2.7-tp{tp_size}",
            q_heads=48 // tp_size,
            kv_heads=max(1, 8 // tp_size),
            head_dim=128,
            source="MiniMax-M2.7 tensor-parallel slicing",
        )
        for tp_size in (1, 3, 4, 6, 8, 12, 16)
    ),
    GqaGeometry(
        model_id="minimax-m3",
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        source="MiniMax-M3 production paged-attention benchmark contract",
    ),
    GqaGeometry(
        model_id="minimax-m3-tp2",
        q_heads=32,
        kv_heads=2,
        head_dim=128,
        source="benchmark_moe MiniMax-M3 TP2 profile",
    ),
    GqaGeometry(
        model_id="minimax-m3-tp4",
        q_heads=16,
        kv_heads=1,
        head_dim=128,
        source="benchmark_moe MiniMax-M3 TP4 shape profile",
    ),
)


@dataclass(frozen=True, kw_only=True)
class QsaGeometry:
    model_id: str
    tensor_parallel_size: int
    q_heads: int
    kv_heads: int
    source: str


QSA_GEOMETRIES = (
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp1",
        tensor_parallel_size=1,
        q_heads=24,
        kv_heads=2,
        source="Qwen3.8 Flash Next text_config",
    ),
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp2",
        tensor_parallel_size=2,
        q_heads=12,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
    QsaGeometry(
        model_id="qwen3.8-flash-next-180b-tp4",
        tensor_parallel_size=4,
        q_heads=6,
        kv_heads=1,
        source="Qwen3.8 Flash Next tensor-parallel slicing",
    ),
)


@dataclass(frozen=True, kw_only=True)
class MlaGeometry:
    model_id: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    page_size: int
    source: str


MLA_GEOMETRIES = (
    MlaGeometry(
        model_id="kimi-k3-dense-mla",
        num_q_heads=8,
        qk_head_dim=576,
        v_head_dim=512,
        page_size=944,
        source="benchmark_dense_mla.py production native K3 defaults",
    ),
)


@dataclass(frozen=True, kw_only=True)
class SparseMlaGeometry:
    model_id: str
    layout: str
    cache_format: str
    num_q_heads: int
    qk_head_dim: int
    v_head_dim: int
    swa_width: int
    swa_page_size: int
    indexed_width: int
    indexed_page_size: int
    source: str


SPARSE_MLA_GEOMETRIES = tuple(
    SparseMlaGeometry(
        model_id=f"{model_name}-{contract}-h{heads}",
        layout="compressed_dsv4",
        cache_format=cache_format,
        num_q_heads=heads,
        qk_head_dim=512,
        v_head_dim=512,
        swa_width=128,
        swa_page_size=64,
        indexed_width=indexed_width,
        indexed_page_size=indexed_page_size,
        source="benchmark_compressed_sparse_mla.py TP-sliced contracts",
    )
    for model_name, cache_format in (
        ("deepseek-v4-flash", "deepseek_v4"),
        ("deepseek-v41", "deepseek_v41"),
    )
    for heads in (64, 32, 16, 8)
    for contract, indexed_width, indexed_page_size in (
        ("swa", 0, 64),
        ("swa-c4", 512, 64),
        ("swa-c128", 512, 2),
    )
)


ATTENTION_BENCHMARK_PRESETS = (
    AttentionBenchmarkPreset(
        preset_id="paged:qwen3.8-27b",
        component="attention.gqa",
        model_id="qwen3.8-27b",
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    AttentionBenchmarkPreset(
        preset_id="paged:qwen-gqa",
        component="attention.gqa",
        model_id="qwen-gqa",
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    AttentionBenchmarkPreset(
        preset_id="paged:minimax-m2.7",
        component="attention.gqa",
        model_id="minimax-m2.7",
        source="benchmark_paged_attention.BENCHMARK_PROFILES",
    ),
    AttentionBenchmarkPreset(
        preset_id="vllm-paged:qwen-gqa",
        component="attention.gqa",
        model_id="qwen-gqa",
        source="benchmark_vllm_triton_paged_attention.PROFILES",
    ),
    AttentionBenchmarkPreset(
        preset_id="vllm-paged:minimax-m2.7",
        component="attention.gqa",
        model_id="minimax-m2.7",
        source="benchmark_vllm_triton_paged_attention.PROFILES",
    ),
    AttentionBenchmarkPreset(
        preset_id="paged-msa:minimax-m3-default",
        component="attention.gqa",
        model_id="minimax-m3",
        source="benchmark_paged_msa.py defaults",
    ),
    *(
        AttentionBenchmarkPreset(
            preset_id=f"qsa:{profile}",
            component="attention.qsa",
            model_id="qwen3.8-flash-next-180b",
            source="benchmark_qsa.PROFILES",
        )
        for profile in ("tp1", "tp2", "tp4")
    ),
    *(
        AttentionBenchmarkPreset(
            preset_id=f"gdn:{case_name}",
            component="attention.gdn",
            model_id="qwen3.8-flash-next-180b",
            source="benchmark_gdn_decode.QWEN38_GDN_CASES",
        )
        for case_name in (
            "qk16-v48-decode-bs1",
            "qk8-v24-decode-bs1",
            "qk8-v24-decode-bs4",
            "qk8-v24-spec2-bs4",
            "qk8-v24-spec4-bs1",
            "qk8-v24-spec4-uneven",
            "qk8-v24-spec4-bs4",
            "qk4-v12-decode-bs1",
            "qk2-v6-decode-bs1",
        )
    ),
    AttentionBenchmarkPreset(
        preset_id="dense-mla:kimi-k3",
        component="attention.mla",
        model_id="kimi-k3",
        source="benchmark_dense_mla.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="dsa-indexer:glm-5.1-default",
        component="attention.dsa_indexer",
        model_id="glm-5.1",
        source="benchmark_dsa_indexer.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="msa-indexer:minimax-m3-default",
        component="attention.dsa_indexer",
        model_id="minimax-m3",
        source="benchmark_msa_indexer.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="paged-indexer:deepseek-v4-flash-default",
        component="attention.dsa_indexer",
        model_id="deepseek-v4-flash",
        source="benchmark_paged_indexer.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="mla:glm-5.2-default",
        component="attention.sparse_mla",
        model_id="glm-5.2",
        source="benchmark_mla.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="mla:target-prefill64k-bs1",
        component="attention.sparse_mla",
        model_id="glm-5.2",
        source="benchmark_mla.TARGET_PREFILL64K_BS1_PRESET",
    ),
    AttentionBenchmarkPreset(
        preset_id="mla:target-glm52-prefill4k-ctx16k",
        component="attention.sparse_mla",
        model_id="glm-5.2",
        source="benchmark_mla.TARGET_GLM52_PREFILL4K_CTX16K_PRESET",
    ),
    AttentionBenchmarkPreset(
        preset_id="mla:target-dsv4-trace",
        component="attention.compressed_sparse_mla",
        model_id="deepseek-v4-flash",
        source="benchmark_mla.TARGET_DSV4_TRACE_PRESET",
    ),
    AttentionBenchmarkPreset(
        preset_id="compressed-mla:vllm-dsv4-trace",
        component="attention.compressed_sparse_mla",
        model_id="deepseek-v4-flash",
        source="benchmark_compressed_sparse_mla.VLLM_DSV4_TRACE_PRESET",
    ),
    AttentionBenchmarkPreset(
        preset_id="compressed-mla:deepseek-v4-flash-default",
        component="attention.compressed_sparse_mla",
        model_id="deepseek-v4-flash",
        source="benchmark_compressed_sparse_mla.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="unified-mla:glm-5.1-decode",
        component="attention.sparse_mla",
        model_id="glm-5.1",
        source="benchmark_unified_mla_sm120.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="unified-mla:deepseek-v4-flash-decode",
        component="attention.compressed_sparse_mla",
        model_id="deepseek-v4-flash",
        source="benchmark_unified_mla_sm120.py defaults",
    ),
    AttentionBenchmarkPreset(
        preset_id="unified-mla:deepseek-v4-flash-prefill",
        component="attention.compressed_sparse_mla",
        model_id="deepseek-v4-flash",
        source="benchmark_unified_mla_sm120.py defaults",
    ),
)


def gdn_cases() -> tuple[WorkloadCase, ...]:
    cases = []
    exercised_queries = set()
    capacity_edges = {}
    for geometry in GDN_GEOMETRIES:
        lengths = geometry.query_lengths
        route_columns = 6 if geometry.decay_recipe == "kda" else 4
        for max_seqs in (1, 4):
            if max_seqs < len(lengths):
                continue
            query = {
                "gate_activation": "sigmoid",
                "qk_l2norm": True,
                "key_heads": geometry.key_heads,
                "value_heads": geometry.value_heads,
                "state_dtype": geometry.state_dtype,
                "max_seqs": max_seqs,
                "max_tokens": max_seqs * route_columns,
                "state_index_columns": route_columns,
            }
            query_key = tuple(sorted(query.items()))
            if max(lengths) == route_columns:
                exercised_queries.add(query_key)
            cases.append(
                WorkloadCase.create(
                    group_id=geometry.model_id,
                    query=query,
                    metadata={
                        "decay_recipe": geometry.decay_recipe,
                        "model_id": geometry.model_id,
                        "query_lengths": list(lengths),
                        "source": geometry.source,
                    },
                    label=f"{geometry.model_id}-capacity-bs{max_seqs}",
                )
            )

    for geometry in GDN_GEOMETRIES:
        for max_seqs in COMMON_SEQUENCE_CAPACITIES:
            for columns in GDN_STATE_INDEX_COLUMNS:
                query = {
                    "gate_activation": "sigmoid",
                    "qk_l2norm": True,
                    "key_heads": geometry.key_heads,
                    "value_heads": geometry.value_heads,
                    "state_dtype": geometry.state_dtype,
                    "max_seqs": max_seqs,
                    "max_tokens": max_seqs * columns,
                    "state_index_columns": columns,
                }
                query_key = tuple(sorted(query.items()))
                capacity_edges.setdefault(query_key, (geometry, query))

    for query_key, (geometry, query) in capacity_edges.items():
        if query_key in exercised_queries:
            continue
        columns = int(query["state_index_columns"])
        cases.append(
            WorkloadCase.create(
                group_id=geometry.model_id,
                query=query,
                scenario="capacity-edge",
                metadata={
                    "decay_recipe": geometry.decay_recipe,
                    "model_id": geometry.model_id,
                    "query_lengths": [columns],
                    "source": geometry.source,
                },
                label=f"{geometry.model_id}-columns{columns}-edge",
            )
        )
    return tuple(cases)


def gqa_cases() -> tuple[WorkloadCase, ...]:
    cases = []
    for geometry in GQA_GEOMETRIES:
        for kv_dtype in geometry.kv_dtypes:
            for page_size in geometry.page_sizes:
                for batch_size in geometry.batch_sizes:
                    for cache_tokens in geometry.context_tokens:
                        group_id = (
                            f"{geometry.model_id}-{kv_dtype}-page{page_size}"
                            f"-ctx{cache_tokens}"
                        )
                        for layout in geometry.cache_layouts:
                            query = {
                                "mode": "decode",
                                "q_dtype": "bfloat16",
                                "kv_dtype": kv_dtype,
                                "q_heads": geometry.q_heads,
                                "kv_heads": geometry.kv_heads,
                                "head_dim_qk": geometry.head_dim,
                                "head_dim_vo": geometry.head_dim,
                                "page_size": page_size,
                                "batch_size": batch_size,
                                "query_len": 1,
                                "cache_tokens": cache_tokens,
                                "window_left": -1,
                                "requested_graph_ctas_per_sm": None,
                                "force_split_kv": None,
                                "kv_cache_layout": layout,
                            }
                            cases.append(
                                WorkloadCase.create(
                                    group_id=group_id,
                                    query=query,
                                    metadata={
                                        "model_id": geometry.model_id,
                                        "source": geometry.source,
                                    },
                                    label=geometry.model_id,
                                )
                            )
    return tuple(cases)


def _qsa_query(
    *,
    geometry: QsaGeometry,
    kv_dtype: str,
    main_page_size: int,
    max_batch: int,
    max_q_rows: int,
    max_seq_len: int,
    max_speculative_tokens: int,
    position_axes: int,
    mrope_interleaved: bool,
) -> dict[str, object]:
    return {
        "q_dtype": "bfloat16",
        "kv_dtype": kv_dtype,
        "q_heads": geometry.q_heads,
        "kv_heads": geometry.kv_heads,
        "head_dim": 256,
        "index_heads": 4,
        "index_kv_heads": 1,
        "index_head_dim": 128,
        "index_rotary_dim": 64,
        "main_page_size": main_page_size,
        "max_batch": max_batch,
        "max_q_rows": max_q_rows,
        "max_seq_len": max_seq_len,
        "max_speculative_tokens": max_speculative_tokens,
        "compress_ratio": 4,
        "budget": 2_048,
        "position_axes": position_axes,
        "mrope_interleaved": mrope_interleaved,
    }


def _qsa_case(
    *,
    geometry: QsaGeometry,
    query: dict[str, object],
    label_suffix: str,
) -> WorkloadCase:
    group_id = (
        f"{geometry.model_id}-{query['kv_dtype']}-page{query['main_page_size']}"
        f"-ctx{query['max_seq_len']}"
    )
    return WorkloadCase.create(
        group_id=group_id,
        query=query,
        metadata={
            "model_id": geometry.model_id,
            "source": geometry.source,
            "tensor_parallel_size": geometry.tensor_parallel_size,
        },
        label=f"{geometry.model_id}-{label_suffix}",
    )


def qsa_cases() -> tuple[WorkloadCase, ...]:
    """Return the measured selected-prefill policy corpus.

    Every case executes the direct ``rows > 64`` route with the serving plan
    capacity (batch four, three speculative tokens, interleaved 3-axis M-RoPE).
    The reduced planner intentionally leaves every other page/capacity point
    uncovered so AUTO resolution uses the N32 heuristic there.
    """
    cases = []
    for geometry in QSA_GEOMETRIES:
        for kv_dtype in COMMON_KV_DTYPES:
            for main_page_size in QSA_PAGE_SIZES:
                for max_seq_len, row_counts in QSA_DIRECT_PREFILL_ROWS.items():
                    for max_q_rows in row_counts:
                        query = _qsa_query(
                            geometry=geometry,
                            kv_dtype=kv_dtype,
                            main_page_size=main_page_size,
                            max_batch=4,
                            max_q_rows=max_q_rows,
                            max_seq_len=max_seq_len,
                            max_speculative_tokens=3,
                            position_axes=3,
                            mrope_interleaved=True,
                        )
                        cases.append(
                            WorkloadCase.create(
                                group_id=(
                                    f"{geometry.model_id}-{kv_dtype}"
                                    f"-page{main_page_size}"
                                ),
                                query=query,
                                scenario="direct-prefill",
                                metadata={
                                    "model_id": geometry.model_id,
                                    "source": geometry.source,
                                    "tensor_parallel_size": (
                                        geometry.tensor_parallel_size
                                    ),
                                    "rows": max_q_rows,
                                    "context": max_seq_len,
                                    "main_page_size": main_page_size,
                                    "kv_dtype": kv_dtype,
                                    "kind": "prefill",
                                },
                                label=(
                                    f"{geometry.model_id}-direct-prefill"
                                    f"-r{max_q_rows}-c{max_seq_len}"
                                    f"-p{main_page_size}-{kv_dtype}"
                                ),
                            )
                        )
    return tuple(cases)


def mla_cases() -> tuple[WorkloadCase, ...]:
    cases = []
    decode_rows = COMMON_SEQUENCE_CAPACITIES
    extend_rows = (128, *COMMON_PREFILL_TOKEN_CAPACITIES, 16_384)
    cache_tokens = (1_024, 32_768, 65_536, 131_072)
    for geometry in MLA_GEOMETRIES:
        for kv_dtype in COMMON_KV_DTYPES:
            group_id = f"{geometry.model_id}-{kv_dtype}"
            for mode, rows_values in (
                ("decode", decode_rows),
                ("extend", extend_rows),
            ):
                for query_rows in rows_values:
                    for width in cache_tokens:
                        if width < query_rows:
                            continue
                        query = {
                            "mode": mode,
                            "q_dtype": kv_dtype,
                            "kv_dtype": kv_dtype,
                            "num_q_heads": geometry.num_q_heads,
                            "qk_head_dim": geometry.qk_head_dim,
                            "v_head_dim": geometry.v_head_dim,
                            "page_size": geometry.page_size,
                            "query_rows": query_rows,
                            "max_batch": (query_rows if mode == "decode" else 1),
                            "cache_tokens": width,
                            "physical_record_width": geometry.qk_head_dim,
                            "window_size": None,
                            "use_cuda_graph": True,
                        }
                        cases.append(
                            WorkloadCase.create(
                                group_id=group_id,
                                query=query,
                                metadata={
                                    "model_id": geometry.model_id,
                                    "source": geometry.source,
                                },
                                label=geometry.model_id,
                            )
                        )
    return tuple(cases)


def sparse_mla_cases() -> tuple[WorkloadCase, ...]:
    cases = []
    for geometry in SPARSE_MLA_GEOMETRIES:
        for rows in (
            *COMMON_SEQUENCE_CAPACITIES,
            *COMMON_PREFILL_TOKEN_CAPACITIES,
        ):
            query = {
                "layout": geometry.layout,
                "cache_format": geometry.cache_format,
                "mode": "decode" if rows <= 256 else "extend",
                "q_dtype": "bfloat16",
                "kv_dtype": "uint8" if geometry.cache_format == "deepseek_v41" else "float8_e4m3fn",
                "num_q_heads": geometry.num_q_heads,
                "qk_head_dim": geometry.qk_head_dim,
                "v_head_dim": geometry.v_head_dim,
                "query_rows": rows,
                "swa_width": geometry.swa_width,
                "swa_page_size": geometry.swa_page_size,
                "indexed_width": geometry.indexed_width,
                "indexed_page_size": geometry.indexed_page_size,
            }
            cases.append(
                WorkloadCase.create(
                    group_id=geometry.model_id,
                    query=query,
                    metadata={
                        "model_id": geometry.model_id,
                        "source": geometry.source,
                    },
                    label=geometry.model_id,
                )
            )
    return tuple(cases)


def _manifest_payload(component: str) -> dict[str, object]:
    shared = {
        "benchmark_presets": [asdict(item) for item in ATTENTION_BENCHMARK_PRESETS],
        "common_batches": list(COMMON_BATCHES),
        "common_context_tokens": list(COMMON_CONTEXT_TOKENS),
        "common_kv_dtypes": list(COMMON_KV_DTYPES),
        "common_page_sizes": list(COMMON_PAGE_SIZES),
        "common_prefill_token_capacities": list(COMMON_PREFILL_TOKEN_CAPACITIES),
        "common_sequence_capacities": list(COMMON_SEQUENCE_CAPACITIES),
    }
    if component == "gdn":
        shared["geometries"] = [asdict(item) for item in GDN_GEOMETRIES]
    elif component == "gqa":
        shared["geometries"] = [asdict(item) for item in GQA_GEOMETRIES]
    elif component == "qsa":
        shared["geometries"] = [asdict(item) for item in QSA_GEOMETRIES]
        shared["qsa_batches"] = list(QSA_BATCHES)
        shared["qsa_context_tokens"] = list(QSA_CONTEXT_TOKENS)
        shared["qsa_page_sizes"] = list(QSA_PAGE_SIZES)
        shared["qsa_position_layouts"] = [list(item) for item in QSA_POSITION_LAYOUTS]
        shared["qsa_speculative_context_tokens"] = list(QSA_SPECULATIVE_CONTEXT_TOKENS)
        shared["qsa_direct_prefill_rows"] = {
            str(context): list(rows)
            for context, rows in QSA_DIRECT_PREFILL_ROWS.items()
        }
    elif component == "mla":
        shared["geometries"] = [asdict(item) for item in MLA_GEOMETRIES]
    elif component == "sparse_mla":
        shared["geometries"] = [asdict(item) for item in SPARSE_MLA_GEOMETRIES]
    else:
        raise ValueError(f"unknown attention corpus {component!r}")
    return shared


def attention_corpus_manifest(component: str) -> dict[str, object]:
    payload = {"schema_version": 1, **_manifest_payload(component)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["corpus_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


__all__ = [
    "ATTENTION_BENCHMARK_PRESETS",
    "AttentionBenchmarkPreset",
    "COMMON_BATCHES",
    "COMMON_CONTEXT_TOKENS",
    "COMMON_KV_DTYPES",
    "COMMON_PAGE_SIZES",
    "COMMON_PREFILL_TOKEN_CAPACITIES",
    "COMMON_SEQUENCE_CAPACITIES",
    "GDN_GEOMETRIES",
    "GDN_STATE_INDEX_COLUMNS",
    "GQA_GEOMETRIES",
    "MLA_GEOMETRIES",
    "QSA_GEOMETRIES",
    "QSA_BATCHES",
    "QSA_CONTEXT_TOKENS",
    "QSA_DIRECT_PREFILL_ROWS",
    "QSA_PAGE_SIZES",
    "QSA_POSITION_LAYOUTS",
    "QSA_SPECULATIVE_CONTEXT_TOKENS",
    "SPARSE_MLA_GEOMETRIES",
    "attention_corpus_manifest",
    "gdn_cases",
    "gqa_cases",
    "mla_cases",
    "qsa_cases",
    "sparse_mla_cases",
]
