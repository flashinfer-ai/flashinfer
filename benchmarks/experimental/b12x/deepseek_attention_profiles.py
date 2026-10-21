"""Model-derived contracts for native DeepSeek indexer and sparse-MLA benchmarks.

The fields mirror the V4/V4.1 integrations, not interchangeable cache recipes.
Tensor contents are supplied by each benchmark; this module loads geometry only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


MODEL_REPOS = {
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-v4.1-flash": "deepseek-ai/DeepSeek-V4.1-Flash",
}


@dataclass(frozen=True)
class IndexerForm:
    name: str
    ratio: int
    page_size: int
    heads: int
    topk: int
    cache_format: str
    layers: tuple[int, ...]
    max_candidates: int = 0
    candidate_topk_blocks: int = 0


@dataclass(frozen=True)
class MLAForm:
    name: str
    ratio: int
    swa_width: int
    swa_page_size: int
    indexed_page_size: int
    indexed_width: int | None
    layers: tuple[int, ...]
    draft: bool = False


@dataclass(frozen=True)
class DeepseekAttentionProfile:
    name: str
    model_path: Path
    config: dict
    tp_size: int
    block_size: int
    speculative_tokens: int
    attention_heads: int
    index_heads: int
    cache_format: str
    indexer_forms: tuple[IndexerForm, ...]
    mla_forms: tuple[MLAForm, ...]


def _config_path(name: str, model_path: str | Path | None) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser()
        return path if path.is_file() else path / "config.json"
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(
            hf_hub_download(MODEL_REPOS[name], "config.json", local_files_only=True)
        )
    except LocalEntryNotFoundError:
        return Path(hf_hub_download(MODEL_REPOS[name], "config.json"))


def load_attention_profile(
    name: str,
    model_path: str | Path | None = None,
    *,
    tp_size: int = 4,
    block_size: int = 256,
    speculative_tokens: int = 7,
) -> DeepseekAttentionProfile:
    if name not in MODEL_REPOS:
        raise ValueError(f"unknown DeepSeek attention profile {name!r}")
    path = _config_path(name, model_path)
    raw = json.loads(path.read_text())
    v41 = name == "deepseek-v4.1-flash"
    expected_root = "deepseek_v41" if v41 else "deepseek_v4"
    if raw.get("model_type") != expected_root:
        raise ValueError(f"{name} requires root model_type={expected_root}")
    config = raw.get("text_config", raw)
    expected = "deepseek_v41_text" if v41 else "deepseek_v4"
    if config.get("model_type") != expected:
        raise ValueError(
            f"{name} requires model_type={expected}, not {config.get('model_type')!r}"
        )
    if tp_size < 1 or config["num_attention_heads"] % tp_size:
        raise ValueError("TP size must divide the model's attention heads")
    if block_size < 1 or speculative_tokens < 0:
        raise ValueError(
            "block size must be positive and speculative token count non-negative"
        )
    if config["head_dim"] != 512 or config["index_head_dim"] != 128:
        raise ValueError("unsupported DeepSeek attention/indexer head geometry")
    heads = config["num_attention_heads"] // tp_size
    # Both integrations use ReplicatedLinear for the index queries and weights.
    index_heads = config["index_n_heads"]
    topk, window = config["index_topk"], config["sliding_window"]
    count = config["num_hidden_layers"]
    ratios = config["compress_ratios"][:count]
    if len(ratios) != count:
        raise ValueError("compression ratios do not cover all target layers")
    indexer, attention = [], []
    if v41:
        if set(ratios) - {0, 1, 2}:
            raise ValueError("unsupported V4.1 compression ratio")
        if block_size % 2:
            raise ValueError("V4.1 base block size must support ratio-two pages")
        sources = tuple(config["index_source_layer_ids"])
        candidate_source = config["candidate_source_layer_id"]
        if (
            candidate_source not in sources
            or sources != tuple(sorted(set(sources)))
            or any(layer < 0 or layer >= count for layer in sources)
            or any(
                ratios[layer] != (2 if layer < candidate_source else 1)
                for layer in sources
            )
        ):
            raise ValueError(
                "V4.1 index sources do not match C2/source/reindex topology"
            )
        indexer.extend(
            (
                IndexerForm(
                    "c2-dense",
                    2,
                    block_size // 2,
                    index_heads,
                    topk,
                    "mxfp4",
                    tuple(layer for layer in sources if layer < candidate_source),
                ),
                IndexerForm(
                    "c1-source",
                    1,
                    block_size,
                    index_heads,
                    topk,
                    "mxfp4",
                    (candidate_source,),
                    candidate_topk_blocks=config["candidate_topk_blocks"],
                ),
                IndexerForm(
                    "c1-reindex",
                    1,
                    block_size,
                    index_heads,
                    topk,
                    "mxfp4",
                    tuple(layer for layer in sources if layer > candidate_source),
                    max_candidates=config["candidate_topk_blocks"]
                    * config["candidate_block_size"],
                ),
            )
        )
        for label, ratio in (("swa", 0), ("c2", 2), ("c1", 1)):
            attention.append(
                MLAForm(
                    label,
                    ratio,
                    window,
                    32,
                    block_size // max(ratio, 1),
                    topk if ratio else 0,
                    tuple(i for i, r in enumerate(ratios) if r == ratio),
                )
            )
        if speculative_tokens:
            draft_count = int(config.get("num_nextn_predict_layers", 0))
            if draft_count:
                draft_width = ((window + speculative_tokens + 63) // 64) * 64
                attention.append(
                    MLAForm(
                        "draft-swa",
                        0,
                        draft_width,
                        32,
                        block_size,
                        0,
                        tuple(range(count, count + draft_count)),
                        draft=True,
                    )
                )
        cache_format = "deepseek_v41"
    else:
        if set(ratios) - {0, 4, 128}:
            raise ValueError("unsupported V4.0 compression ratio")
        if block_size != 256:
            raise ValueError("V4.0 B12x C4 indexer requires base block size 256")
        c4_layers = tuple(i for i, r in enumerate(ratios) if r == 4)
        indexer.append(
            IndexerForm("c4-dense", 4, 64, index_heads, topk, "fp8", c4_layers)
        )
        for label, ratio, width in (
            ("swa", 0, 0),
            ("c4", 4, topk),
            ("c128", 128, None),
        ):
            attention.append(
                MLAForm(
                    label,
                    ratio,
                    window,
                    64,
                    block_size // max(ratio, 1),
                    width,
                    tuple(i for i, r in enumerate(ratios) if r == ratio),
                )
            )
        cache_format = "deepseek_v4"
    return DeepseekAttentionProfile(
        name,
        path.parent,
        config,
        tp_size,
        block_size,
        speculative_tokens,
        heads,
        index_heads,
        cache_format,
        tuple(indexer),
        tuple(attention),
    )


def integration_provenance(vllm_path: str | Path | None = None) -> dict:
    """Identify the integration whose prepared-tensor contracts are benchmarked."""
    from benchmarks.experimental.b12x.benchmark_v41_serving import repository_state

    root = (
        Path(vllm_path).expanduser()
        if vllm_path
        else Path(__file__).resolve().parents[2] / "vllm-hh-rebase"
    )
    sources = (
        "vllm/models/deepseek_v4_1/attention.py",
        "vllm/models/deepseek_v4_1/sparse_mla.py",
        "vllm/models/deepseek_v4/attention.py",
        "vllm/models/deepseek_v4/nvidia/b12x.py",
        "vllm/models/deepseek_v4/nvidia/b12x_indexer.py",
        "vllm/v1/attention/backends/mla/sparse_swa.py",
        "vllm/v1/attention/backends/mla/compressor_utils.py",
    )
    return {
        "repository": repository_state(root),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "scope": "Prepared-tensor contracts matched to this integration; not a full adapter/model invocation.",
    }
