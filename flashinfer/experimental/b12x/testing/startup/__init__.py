"""Model-derived benchmark resource builders; never imported by serving."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


MODEL_CHECKPOINTS = {
    "qwen3.8-flash-next-180b": Path(
        "/data/models/qwen3.8-flash-next-mixed/"
        "qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1"
    ),
    "glm-5.3-flash": Path("/data/models/GLM-5.3-Flash-4p67"),
}


def model_metadata(
    model: str,
    *,
    tp: int,
    checkpoint=None,
    max_seqs: int = 1,
    cache_tokens: int = 4096,
    spec_tokens: int = 3,
) -> dict:
    """Read actual checkpoint geometry, separate from every knob declaration."""
    if model not in MODEL_CHECKPOINTS:
        raise ValueError(f"unknown native startup model {model!r}")
    if tp != 2:
        raise ValueError("the initial native startup acceptance workload is TP=2")
    if max_seqs <= 0 or cache_tokens <= 0 or spec_tokens < 0:
        raise ValueError(
            "native startup capacities must be positive and speculation nonnegative"
        )
    path = MODEL_CHECKPOINTS[model] if checkpoint is None else Path(checkpoint)
    payload = json.loads((path / "config.json").read_text())
    metadata = dict(payload.get("text_config", payload))
    maximum = metadata.get("max_position_embeddings")
    if maximum is not None and cache_tokens > maximum:
        raise ValueError(
            "native startup cache capacity exceeds the model context limit"
        )
    metadata.update(
        _model_id=model,
        _tp=tp,
        _max_seqs=max_seqs,
        _cache_tokens=cache_tokens,
        _spec_tokens=spec_tokens,
        _checkpoint_path=str(path),
        _full_config=payload,
    )
    return metadata


def make_benchmark_requests(metadata, *, device, rows, groups=None):
    """Build benchmark requests for explicit workloads without sampling coordinates."""
    names = ("dense", "moe", "attention", "sequence", "norm", "ple")
    if groups is not None:
        if set(groups) - set(names):
            raise ValueError(f"unknown startup groups: {set(groups) - set(names)}")
        names = tuple(name for name in names if name in groups)
    requests, owners = [], {}
    for name in names:
        module = importlib.import_module(f"{__name__}.{name}")
        for request in module.make_benchmark_requests(metadata, device=device, rows=rows):
            if request.name in owners:
                raise ValueError(f"duplicate native startup request {request.name!r}")
            requests.append(request)
            owners[request.name] = module
    return requests, owners


__all__ = ["MODEL_CHECKPOINTS", "make_benchmark_requests", "model_metadata"]
