"""Native DeepSeek mHC checkpoint profiles for residual benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.experimental.b12x.checkpoint_loader import IndexedSafetensorLoader


@dataclass(frozen=True)
class MHCProfile:
    name: str
    hf_repo_id: str
    model_type: str
    lagged: bool
    default_layer_idx: int = 3


@dataclass(frozen=True)
class MHCProfileBundle:
    profile: MHCProfile
    model_path: Path
    config: dict[str, Any]
    tensors: dict[str, torch.Tensor]


MODEL_PROFILES = {
    "deepseek-v4-flash": MHCProfile(
        name="deepseek-v4-flash",
        hf_repo_id="deepseek-ai/DeepSeek-V4-Flash",
        model_type="deepseek_v4",
        lagged=False,
    ),
    "deepseek-v4.1-flash": MHCProfile(
        name="deepseek-v4.1-flash",
        hf_repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
        model_type="deepseek_v41_text",
        lagged=True,
    ),
}

_REQUIRED_CONFIG = (
    "hidden_size",
    "rms_norm_eps",
    "hc_mult",
    "hc_eps",
    "hc_sinkhorn_iters",
)


def _cached_snapshot(repo_id: str) -> Path | None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(snapshot_download(
            repo_id=repo_id, local_files_only=True,
            allow_patterns=("config.json", "model.safetensors.index.json"),
        ))
    except LocalEntryNotFoundError:
        return None
    if (path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file():
        return path
    return None


def _download_metadata(profile: MHCProfile) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=profile.hf_repo_id,
            allow_patterns=("config.json", "model.safetensors.index.json"),
        )
    )


def _ensure_required_shards(
    profile: MHCProfile,
    model_path: Path,
    required_keys: tuple[str, ...],
    *,
    download_missing: bool,
) -> None:
    weight_map = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    try:
        shards = {weight_map[key] for key in required_keys}
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing required mHC tensor {exc.args[0]!r}") from exc
    missing = sorted(shard for shard in shards if not (model_path / shard).is_file())
    if not missing:
        return
    if not download_missing:
        raise FileNotFoundError(f"required checkpoint shards unavailable: {missing}")

    from huggingface_hub import snapshot_download

    revision = model_path.name
    snapshot_download(
        repo_id=profile.hf_repo_id,
        revision=revision,
        allow_patterns=tuple(missing),
    )
    still_missing = [shard for shard in missing if not (model_path / shard).is_file()]
    if still_missing:
        raise FileNotFoundError(f"required checkpoint shards unavailable: {still_missing}")


def _resolve_model_path(profile: MHCProfile, model_path: str | Path | None) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"model path is not a directory: {path}")
        return path
    return _cached_snapshot(profile.hf_repo_id) or _download_metadata(profile)


def _text_config(profile: MHCProfile, raw_config: dict[str, Any]) -> dict[str, Any]:
    if profile.name == "deepseek-v4-flash":
        if raw_config.get("model_type") != profile.model_type:
            raise ValueError("checkpoint model_type does not match DeepSeek V4 Flash")
        config = raw_config
    else:
        text_config = raw_config.get("text_config")
        if (
            raw_config.get("model_type") != "deepseek_v41"
            or not isinstance(text_config, dict)
            or text_config.get("model_type") != profile.model_type
        ):
            raise ValueError("checkpoint model_type does not match DeepSeek V4.1 Flash")
        config = text_config
    missing = [key for key in _REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"checkpoint config is missing mHC fields: {missing}")
    if config["hc_mult"] != 4:
        raise ValueError("mHC benchmark requires hc_mult=4")
    return dict(config)


def _validate_tensor(name: str, tensor: torch.Tensor, *, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be loaded on CPU")
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have dtype {dtype} and shape {shape}; got {tensor.dtype} {tuple(tensor.shape)}"
        )


def load_mhc_profile(
    name: str, model_path: str | Path | None = None, layer_idx: int = 3
) -> MHCProfileBundle:
    """Load one layer's native mHC parameters without allocating GPU memory."""
    try:
        profile = MODEL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown mHC profile: {name}") from exc
    path = _resolve_model_path(profile, model_path)
    try:
        raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"checkpoint config is missing from {path}") from None
    config = _text_config(profile, raw_config)
    layers = config.get("num_hidden_layers")
    if not isinstance(layers, int) or not 0 <= layer_idx < layers:
        raise ValueError(f"layer_idx {layer_idx} is outside checkpoint layers")

    prefix = f"layers.{layer_idx}"
    source_keys = {
        "fn": f"{prefix}.hc_ffn_fn",
        "scale": f"{prefix}.hc_ffn_scale",
        "bias": f"{prefix}.hc_ffn_base",
        "prev_fn": f"{prefix}.hc_attn_fn",
        "prev_scale": f"{prefix}.hc_attn_scale",
        "prev_bias": f"{prefix}.hc_attn_base",
        "norm_weight": f"{prefix}.ffn_norm.weight",
    }
    _ensure_required_shards(
        profile,
        path,
        tuple(source_keys.values()),
        download_missing=model_path is None,
    )
    loader = IndexedSafetensorLoader(path)
    tensors = {name: loader.get_tensor(key) for name, key in source_keys.items()}

    hidden_size = config["hidden_size"]
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("checkpoint hidden_size must be a positive integer")
    for key in ("fn", "prev_fn"):
        _validate_tensor(key, tensors[key], shape=(24, 4 * hidden_size), dtype=torch.float32)
    for key in ("scale", "prev_scale"):
        _validate_tensor(key, tensors[key], shape=(3,), dtype=torch.float32)
    for key in ("bias", "prev_bias"):
        _validate_tensor(key, tensors[key], shape=(24,), dtype=torch.float32)
    _validate_tensor(
        "norm_weight", tensors["norm_weight"], shape=(hidden_size,), dtype=torch.bfloat16
    )
    return MHCProfileBundle(profile=profile, model_path=path, config=config, tensors=tensors)
