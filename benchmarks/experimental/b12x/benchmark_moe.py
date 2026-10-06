#!/usr/bin/env python3
"""Static MoE benchmark and shared weight-loading utilities.

By default this is a pre-routed benchmark: model loading, routing-logit
generation, top-k selection, compilation, and oracle/reference checks all
happen outside the timed region. Use ``--include-routing`` to include the
deterministic top-k + softmax routing step in the measured closure.
"""

from __future__ import annotations

from b12x._lib.quant.block_codec import IQ2_CODECS, block_codec

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from b12x.moe._shared.kernels.reference import (
    OracleMetrics,
    compare_to_reference,
    decompose_nvfp4_scales_to_mx_residual,
    moe_reference_f32,
    moe_reference_nvfp4,
    moe_reference_w4a8_mx,
    moe_reference_w4a16_fp4_e8m0_k32,
    moe_reference_w4a16_f32,
    unswizzle_block_scale,
)

from b12x.moe._shared.kernels.activations import (
    SUPPORTED_MOE_ACTIVATIONS,
    SWIGLUOAI_DEFAULT_ALPHA,
    SWIGLUOAI_DEFAULT_BETA,
    SWIGLUOAI_DEFAULT_LIMIT,
    SWIGLUOAI_UNINTERLEAVE,
    moe_activation_w1_rows,
    normalize_moe_activation,
)
from b12x._lib.intrinsics import as_grouped_scale_view, swizzle_block_scale
from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes, nvidia_smi_gpu_mode_snapshot
from b12x.testing.reference.w4a16_reference import moe_reference_w4a16

from b12x.moe import fused_moe
from b12x.preparation import PreparationSession
from benchmarks.experimental.b12x.moe_preparation import prepared_call, request_for_capacity, scratch_for

LEGACY_BATCH_SIZES = [1, 2, 4, 8]
NANO35_MTP_BATCH_SIZES = [1, 4, 9]
# Qwen3.8 Flash Next serves single-request decode at M=1 and verifies three
# MTP draft tokens with the sampled token at M=4. Keep the full micro/dynamic
# boundary and the 128-token serving cap in the profile so planner regressions
# are visible around those production points.
QWEN38_FLASH_NEXT_BATCH_SIZES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    16,
    32,
    64,
    128,
]
# Observed in the live single-request sglang probe:
# - prefill m=23 for the prompt itself
# - larger prefill chunk m=80 during the same request path
# - decode remains effectively m=1 for a single running request
RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES = [1, 23, 80]
# Representative eager-prefill forwards without CUDA graph replay.
EAGER_PREFILL_BATCH_SIZES = [16384, 32768]
# Representative total-token sizes for packed chunked-prefill forwards.
# The first point is one full server-side prefill chunk, then we scale to
# larger packed forwards up to four chunks' worth of tokens.
CHUNKED_PREFILL_BATCH_SIZES = [8192, 16384, 24576, 32768]
BATCH_SIZE_PROFILES = {
    "eager-prefill": EAGER_PREFILL_BATCH_SIZES,
    "micro": LEGACY_BATCH_SIZES,
    "nano35-mtp": NANO35_MTP_BATCH_SIZES,
    "qwen38-flash-next": QWEN38_FLASH_NEXT_BATCH_SIZES,
    "sglang-single-request": RECORDED_SGLANG_SINGLE_REQUEST_BATCH_SIZES,
    "chunked-prefill": CHUNKED_PREFILL_BATCH_SIZES,
}
TP_SIZE = 4
TP_RANK = 0
EP_SIZE = 1
EP_RANK = 0
BENCHMARK_ACTIVATION_CHOICES = sorted(SUPPORTED_MOE_ACTIVATIONS)
_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class ActivationParams:
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None

    def kwargs(self) -> dict[str, float | None]:
        return {
            "swiglu_limit": self.swiglu_limit,
            "swiglu_alpha": self.swiglu_alpha,
            "swiglu_beta": self.swiglu_beta,
        }


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def bench_events(fn, *, warmup, iters, l2_flush=None):
    from b12x.testing.benchmark import samples_ms
    return samples_ms(fn, warmup=warmup, iters=iters, l2_flush=l2_flush)


def fmt_us(times_ms: list[float]) -> str:
    median_us = statistics.median(times_ms) * 1000.0
    min_us = min(times_ms) * 1000.0
    return f"{median_us:8.1f} us (min {min_us:.1f})"


@dataclass(frozen=True)
class TimingStats:
    per_repeat_median_ms: list[float]
    per_repeat_min_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.per_repeat_median_ms)

    @property
    def min_ms(self) -> float:
        return min(self.per_repeat_min_ms)

    @property
    def median_us(self) -> float:
        return self.median_ms * 1000.0

    @property
    def min_us(self) -> float:
        return self.min_ms * 1000.0

    @property
    def repeat_count(self) -> int:
        return len(self.per_repeat_median_ms)

    @property
    def median_range_us(self) -> tuple[float, float]:
        return (
            min(self.per_repeat_median_ms) * 1000.0,
            max(self.per_repeat_median_ms) * 1000.0,
        )




@dataclass(frozen=True)
class RatioStats:
    per_repeat_ratio: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.per_repeat_ratio)

    @property
    def min(self) -> float:
        return min(self.per_repeat_ratio)

    @property
    def max(self) -> float:
        return max(self.per_repeat_ratio)

    @property
    def repeat_count(self) -> int:
        return len(self.per_repeat_ratio)


def fmt_ratio_stats(stats: RatioStats) -> str:
    if stats.repeat_count == 1:
        return f"{stats.median:.2f}x"
    return f"{stats.median:.2f}x (repeat range {stats.min:.2f}-{stats.max:.2f})"


@dataclass(frozen=True)
class BatchResult:
    backend_stats: TimingStats


def summarize_timing_runs(runs_ms: list[list[float]]) -> TimingStats:
    return TimingStats(
        per_repeat_median_ms=[statistics.median(run) for run in runs_ms],
        per_repeat_min_ms=[min(run) for run in runs_ms],
    )


def fmt_timing_stats(stats: TimingStats) -> str:
    if stats.repeat_count == 1:
        return f"{stats.median_us:8.1f} us (min {stats.min_us:.1f})"
    low_us, high_us = stats.median_range_us
    return (
        f"{stats.median_us:8.1f} us "
        f"(repeat medians {low_us:.1f}-{high_us:.1f}, sample min {stats.min_us:.1f})"
    )




@dataclass(frozen=True)
class ScaleContractParams:
    a1_gscale: torch.Tensor
    a2_gscale: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor


@dataclass
class ModelSpec:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tp_size: int
    tp_rank: int
    logical_intermediate_size: int | None = None

    @property
    def I_tp(self) -> int:
        if self.logical_intermediate_size is not None:
            return self.intermediate_size
        return self.intermediate_size // self.tp_size

    @property
    def logical_I_tp(self) -> int:
        return self.logical_intermediate_size or self.I_tp

    @property
    def global_intermediate_size(self) -> int:
        return self.logical_I_tp * self.tp_size


@dataclass(frozen=True)
class ShapeSpec:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


def logical_expert_bytes(
    spec: ModelSpec, *, activation: str, quant_mode: str, source_format: str,
) -> int:
    """Unique weight payload, excluding padding, activations, and rereads."""
    elements = spec.hidden_size * (
        moe_activation_w1_rows(activation, spec.I_tp) + spec.I_tp
    )
    if source_format in IQ2_CODECS:
        return elements // block_codec(source_format).block_weights * block_codec(source_format).block_bytes
    if quant_mode == "w4a8_nvfp4":
        # Native FP4 plus K32 exponents and K16 residual scales.
        return elements // 2 + elements // 32 + elements // 16
    block_size = 32 if source_format == "fp4_e8m0_k32" else 16
    return elements // 2 + elements // block_size


@dataclass(frozen=True)
class ModelProfile:
    label: str
    checkpoint_family: str
    default_layer_idx: int
    tp_size: int
    hf_repo_id: str | None
    default_model_path: pathlib.Path | None = None
    default_activation: str = "silu"
    default_quant_mode: str | None = None
    default_validate: str = "oracle"
    default_swiglu_limit: float | None = None
    default_swiglu_alpha: float | None = None
    default_swiglu_beta: float | None = None
    default_routing: str = "synthetic"
    default_scale_contract: str = "shared"
    shape: ShapeSpec | None = None


MODEL_PROFILES = {
    "puzzle3-iq2-xs": ModelProfile(
        label="Puzzle 3 IQ2_XS/NVFP4",
        checkpoint_family="puzzle3_iq2_xs",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id=None,
        default_activation="relu2",
        default_quant_mode="w4a16",
    ),
    "qwen36-35b-nvfp4": ModelProfile(
        label="Qwen3.6-35B-A3B NVFP4",
        checkpoint_family="qwen",
        default_layer_idx=0,
        tp_size=1,
        hf_repo_id="nvidia/Qwen3.6-35B-A3B-NVFP4",
        default_quant_mode="w4a16",
    ),
    "qwen36-35b-iq2-xs": ModelProfile(
        label="Qwen3.6-35B-A3B IQ2_XS/NVFP4",
        checkpoint_family="qwen_iq2_xs",
        default_layer_idx=0,
        tp_size=1,
        hf_repo_id=None,
        default_quant_mode="w4a16",
    ),
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/config.json
    "mimo26-flash-shape": ModelProfile(
        label="MiMo V2.6 Flash MXFP4 (shape)",
        checkpoint_family="mimo26_flash_shape",
        default_layer_idx=1,
        tp_size=2,
        hf_repo_id=None,
        default_quant_mode="w4a8_mx",
        shape=ShapeSpec(4096, 2048, 256, 8),
    ),
    "mimo26-flash-nvfp4-shape": ModelProfile(
        label="MiMo V2.6 Flash NVFP4 (synthetic comparison)",
        checkpoint_family="mimo26_flash_nvfp4_shape",
        default_layer_idx=1,
        tp_size=2,
        hf_repo_id=None,
        default_quant_mode="nvfp4",
        shape=ShapeSpec(4096, 2048, 256, 8),
    ),
    "qwen38-flash-next": ModelProfile(
        label="Qwen3.8 Flash Next",
        checkpoint_family="qwen",
        default_layer_idx=0,
        tp_size=1,
        hf_repo_id=None,
        default_model_path=pathlib.Path(
            "/data/models/qwen3.8-flash-next-mixed/"
            "qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1"
        ),
        default_quant_mode="nvfp4",
    ),
    "qwen38-flash-next-shape": ModelProfile(
        label="Qwen3.8 Flash Next (shape)",
        checkpoint_family="qwen38_flash_next_shape",
        default_layer_idx=0,
        tp_size=1,
        hf_repo_id=None,
        default_quant_mode="nvfp4",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=2560,
            intermediate_size=640,
            num_experts=512,
            top_k=10,
        ),
    ),
    "qwen397b": ModelProfile(
        label="Qwen3.5-397B",
        checkpoint_family="qwen",
        default_layer_idx=0,
        tp_size=TP_SIZE,
        hf_repo_id="nvidia/Qwen3.5-397B-A17B-NVFP4",
    ),
    "nemotron-backbone": ModelProfile(
        label="NVIDIA Nemotron Backbone",
        checkpoint_family="nemotron",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    ),
    "nano35-w4a16": ModelProfile(
        label="NVIDIA Nano3.5 BF16 NVFP4 W4A16",
        checkpoint_family="nano35_w4a16",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id="nvidia/Nano3.5-BF16-NVFP4-W4A16-LMHEAD-CT",
        default_activation="relu2",
        default_quant_mode="w4a16",
        default_validate="oracle",
    ),
    "nano35-w4a16-shape": ModelProfile(
        label="NVIDIA Nano3.5 BF16 NVFP4 W4A16 (shape)",
        checkpoint_family="nano35_w4a16_shape",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id=None,
        default_activation="relu2",
        default_quant_mode="w4a16",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=2688,
            intermediate_size=1856,
            num_experts=128,
            top_k=6,
        ),
    ),
    "dsv4f": ModelProfile(
        label="DSV4F W4A16 (shape)",
        checkpoint_family="dsv4f_shape",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_activation="silu",
        default_quant_mode="w4a16",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=256,
            top_k=8,
        ),
    ),
    "dsv4f-nvfp4": ModelProfile(
        label="DSV4F NVFP4 (shape)",
        checkpoint_family="dsv4f_nvfp4_shape",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_activation="silu",
        default_quant_mode="nvfp4",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=256,
            top_k=8,
        ),
    ),
    "minimax-m3-shape": ModelProfile(
        label="MiniMax-M3 (shape)",
        checkpoint_family="minimax_m3_shape",
        default_layer_idx=3,
        tp_size=4,
        hf_repo_id=None,
        default_activation=SWIGLUOAI_UNINTERLEAVE,
        default_quant_mode="nvfp4",
        default_validate="none",
        default_swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        default_swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        default_swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
        shape=ShapeSpec(
            hidden_size=6144,
            intermediate_size=3072,
            num_experts=128,
            top_k=4,
        ),
    ),
    "laguna-s21-shape": ModelProfile(
        label="Laguna S-2.1 (shape)",
        checkpoint_family="laguna_s21_shape",
        default_layer_idx=1,
        tp_size=1,
        hf_repo_id=None,
        default_activation="silu",
        default_quant_mode="nvfp4",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=3072,
            intermediate_size=1024,
            num_experts=256,
            top_k=10,
        ),
    ),
    "deepseek-v4-flash": ModelProfile(
        label="DeepSeek V4 Flash",
        checkpoint_family="deepseek_v4_flash",
        default_layer_idx=3,
        tp_size=4,
        hf_repo_id="deepseek-ai/DeepSeek-V4-Flash",
        default_activation="silu",
        default_quant_mode="w4a16",
        default_validate="oracle",
        default_swiglu_limit=10.0,
        default_routing="model",
    ),
    "deepseek-v4-flash-0731": ModelProfile(
        label="DeepSeek V4 Flash 0731",
        checkpoint_family="deepseek_v4_flash",
        default_layer_idx=3,
        tp_size=2,
        hf_repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        default_activation="silu",
        default_quant_mode="w4a8_mx",
        default_validate="oracle",
        default_swiglu_limit=10.0,
        default_routing="model",
        default_scale_contract="per-expert",
    ),
    "deepseek-v4.1-flash": ModelProfile(
        label="DeepSeek V4.1 Flash",
        checkpoint_family="deepseek_v41_flash",
        default_layer_idx=0,
        tp_size=4,
        hf_repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
        default_activation="silu",
        default_quant_mode="w4a8_mx",
        default_validate="oracle",
        default_swiglu_limit=10.0,
        default_routing="model",
    ),
    "glm51": ModelProfile(
        label="GLM-5.1",
        checkpoint_family="glm",
        default_layer_idx=3,
        tp_size=8,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/GLM-5.1-NVFP4"),
    ),
    "glm52": ModelProfile(
        label="GLM-5.2",
        checkpoint_family="glm",
        default_layer_idx=3,
        tp_size=8,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/GLM-5.2-trainer-minimal"),
        default_quant_mode="w4a8_nvfp4",
        default_validate="none",
    ),
    "glm53-flash": ModelProfile(
        label="GLM-5.3 Flash",
        checkpoint_family="glm53_flash",
        default_layer_idx=3,
        tp_size=1,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/GLM-5.3-Flash-4p67"),
        default_quant_mode="nvfp4",
        default_validate="oracle",
    ),
    "glm53-flash-shape": ModelProfile(
        label="GLM-5.3 Flash (shape)",
        checkpoint_family="glm53_flash_shape",
        default_layer_idx=3,
        tp_size=1,
        hf_repo_id=None,
        default_quant_mode="w4a16",
        default_validate="none",
        shape=ShapeSpec(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=288,
            top_k=8,
        ),
    ),
    "minimax-m27": ModelProfile(
        label="MiniMax-M2.7",
        checkpoint_family="minimax_m2",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/MiniMax-M2.7-NVFP4"),
    ),
    "minimax-m3": ModelProfile(
        label="MiniMax-M3",
        checkpoint_family="minimax_m3",
        default_layer_idx=0,
        tp_size=2,
        hf_repo_id=None,
        default_model_path=pathlib.Path("/data/models/MiniMax-M3-NVFP4"),
        default_activation=SWIGLUOAI_UNINTERLEAVE,
        default_swiglu_limit=SWIGLUOAI_DEFAULT_LIMIT,
        default_swiglu_alpha=SWIGLUOAI_DEFAULT_ALPHA,
        default_swiglu_beta=SWIGLUOAI_DEFAULT_BETA,
    ),
}

# Both GGML IQ2 codecs share checkpoint geometry and the same production route.
for _profile in ("puzzle3-iq2-xs", "qwen36-35b-iq2-xs"):
    MODEL_PROFILES[_profile.replace("-iq2-xs", "-iq2-xxs")] = replace(
        MODEL_PROFILES[_profile], label=MODEL_PROFILES[_profile].label.replace("IQ2_XS", "IQ2_XXS"),
    )


def _cached_snapshot_path(repo_id: str) -> pathlib.Path | None:
    cache_root = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
    main_ref = cache_root / "refs" / "main"
    if main_ref.is_file():
        candidate = snapshots_root / main_ref.read_text().strip()
        if candidate.is_dir() and (candidate / "model.safetensors.index.json").is_file():
            return candidate
    indexed_snapshots = [
        path for path in snapshots
        if (path / "model.safetensors.index.json").is_file()
    ]
    if indexed_snapshots:
        return indexed_snapshots[-1]
    if snapshots:
        return snapshots[-1]
    return None


def _default_model_path() -> pathlib.Path:
    local_qwen_path = pathlib.Path("/data/models/Qwen3.5-397B-A17B-NVFP4")
    if local_qwen_path.is_dir():
        return local_qwen_path
    cached_qwen_path = _cached_snapshot_path(MODEL_PROFILES["qwen397b"].hf_repo_id)
    if cached_qwen_path is not None:
        return cached_qwen_path
    return (
        pathlib.Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--nvidia--Qwen3.5-397B-A17B-NVFP4"
        / "snapshots"
        / "__missing__"
    )


def resolve_model_path(
    profile: ModelProfile,
    override: pathlib.Path | None,
) -> pathlib.Path:
    if override is not None:
        return override
    env_path = os.environ.get("B12X_MODEL_PATH")
    if env_path:
        return pathlib.Path(env_path)
    if profile.default_model_path is not None and profile.default_model_path.is_dir():
        return profile.default_model_path
    if profile.hf_repo_id is not None:
        cached_path = _cached_snapshot_path(profile.hf_repo_id)
        if cached_path is not None:
            return cached_path
        from huggingface_hub import snapshot_download

        return pathlib.Path(snapshot_download(repo_id=profile.hf_repo_id))
    if profile.shape is not None:
        return pathlib.Path("<shape-only>")

    raise FileNotFoundError(
        f"no default path found for {profile.label}; pass --model-path explicitly"
    )

MODEL_PATH = _default_model_path()


@dataclass
class ExpertWeights:
    layer_idx: int
    spec: ModelSpec
    w13_permuted: torch.Tensor | None
    w13_scale: torch.Tensor | None
    down_permuted: torch.Tensor | None
    down_scale: torch.Tensor | None
    w13_weight: torch.Tensor
    w13_blockscale_swizzled: torch.Tensor | None
    w2_weight: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor | None
    w13_input_scale: torch.Tensor
    w2_input_scale: torch.Tensor
    w13_input_scale_quant: torch.Tensor
    w2_input_scale_quant: torch.Tensor
    w13_input_scale_per_expert: torch.Tensor
    w2_input_scale_per_expert: torch.Tensor
    w13_input_scale_quant_per_expert: torch.Tensor
    w2_input_scale_quant_per_expert: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    g1_alphas_per_expert: torch.Tensor
    g2_alphas_per_expert: torch.Tensor
    source_format: str = "modelopt_nvfp4"
    w4a16_w13_global_scale: torch.Tensor | None = None
    w4a16_w2_global_scale: torch.Tensor | None = None
    gate_weight: torch.Tensor | None = None
    gate_bias: torch.Tensor | None = None
    gate_tid2eid: torch.Tensor | None = None
    gate_score_func: str = "softmax"
    gate_route_scale: float = 1.0
    gate_norm_topk_prob: bool = True
    oracle_w13_weight: torch.Tensor | None = None
    oracle_w13_scale: torch.Tensor | None = None
    oracle_w2_weight: torch.Tensor | None = None
    oracle_w2_scale: torch.Tensor | None = None
    w13_layout: str = "w31"




def _load_config(model_path: pathlib.Path) -> dict:
    raw_cfg = json.loads((model_path / "config.json").read_text())
    return raw_cfg.get("text_config", raw_cfg)


def _fp4_checkpoint_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected one-byte packed FP4 tensor, got {tensor.dtype}")
    return tensor.view(torch.uint8)

def _slice_v41_tp_shard(
    source: torch.Tensor,
    *,
    dimension: int,
    intermediate_size_per_partition: int,
    tp_rank: int,
    packing: int = 1,
) -> torch.Tensor:
    """Slice one exact, MX-block-aligned TP shard."""
    if (
        intermediate_size_per_partition <= 0
        or intermediate_size_per_partition % 32
        or intermediate_size_per_partition % packing
    ):
        raise ValueError("invalid V4.1 TP shard geometry")
    shard_width = intermediate_size_per_partition // packing
    offset = tp_rank * shard_width
    return source.narrow(dimension, offset, shard_width).contiguous()


def build_model_spec(model_path: pathlib.Path, profile: ModelProfile, *, tp_size_override: int | None = None, tp_rank: int = 0, layer_idx: int | None = None) -> ModelSpec:
    tp = tp_size_override if tp_size_override is not None else profile.tp_size
    if profile.shape is not None:
        return ModelSpec(
            hidden_size=profile.shape.hidden_size,
            intermediate_size=profile.shape.intermediate_size,
            num_experts=profile.shape.num_experts,
            top_k=profile.shape.top_k,
            tp_size=tp,
            tp_rank=tp_rank,
        )

    cfg = _load_config(model_path)
    if profile.checkpoint_family == "puzzle3_iq2_xs":
        selected = profile.default_layer_idx if layer_idx is None else layer_idx
        block = cfg["block_configs"][selected]
        if cfg.get("model_type") != "nemotron_h_puzzle" or block["block_type"] != "moe":
            raise ValueError("Puzzle 3 IQ2_XS requires a Nemotron Puzzle MoE layer")
        return ModelSpec(hidden_size=block["moe_latent_size"],
                         intermediate_size=block["moe_intermediate_size"],
                         num_experts=block["n_routed_experts"],
                         top_k=block["num_experts_per_tok"], tp_size=tp, tp_rank=tp_rank)
    if profile.checkpoint_family == "deepseek_v41_flash":
        if cfg.get("model_type") != "deepseek_v41_text":
            raise ValueError("DeepSeek V4.1 Flash requires its V4.1 text config, not V4.0")
        if tp <= 0 or not 0 <= tp_rank < tp:
            raise ValueError("V4.1 requires a valid TP rank")
        logical_I_tp, remainder = divmod(cfg["moe_intermediate_size"], tp)
        if remainder:
            raise ValueError("V4.1 requires an evenly divisible intermediate size")
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=logical_I_tp,
            logical_intermediate_size=logical_I_tp,
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family in {"qwen", "qwen_iq2_xs"}:
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family in {"glm", "glm53_flash"}:
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family in {"minimax_m2", "minimax_m3"}:
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_experts=cfg["num_local_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "nano35_w4a16":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "deepseek_v4_flash":
        return ModelSpec(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    if profile.checkpoint_family == "nemotron":
        if cfg["hidden_size"] % TP_SIZE != 0:
            raise ValueError(
                f"expected hidden_size {cfg['hidden_size']} to be divisible by {TP_SIZE} for Nemotron local shard"
            )
        # Nemotron expert tensors are stored for one hidden TP shard. The
        # benchmark --tp-size controls how the intermediate dimension is sliced,
        # not the hidden width loaded from each checkpoint shard.
        return ModelSpec(
            hidden_size=cfg["hidden_size"] // TP_SIZE,
            intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["n_routed_experts"],
            top_k=cfg["num_experts_per_tok"],
            tp_size=tp,
            tp_rank=tp_rank,
        )
    raise ValueError(f"unsupported checkpoint family {profile.checkpoint_family!r}")


def make_shape_only_expert_weights(
    spec: ModelSpec,
    *,
    layer_idx: int,
    activation: str,
    source_format: str = "modelopt_nvfp4",
) -> ExpertWeights:
    activation = normalize_moe_activation(activation)
    scale_group = 32 if source_format == "fp4_e8m0_k32" else 16
    if spec.hidden_size % scale_group != 0 or spec.I_tp % scale_group != 0:
        raise ValueError(
            f"shape-only profile requires K and I_tp divisible by {scale_group}, "
            f"got K={spec.hidden_size}, I_tp={spec.I_tp}"
        )

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    w13_rows = moe_activation_w1_rows(activation, I_tp)

    print(
        f"  Creating synthetic shape-only experts (E={E}, K={K}, I_tp={I_tp}, activation={activation})...",
        end="",
        flush=True,
    )
    if source_format == "fp4_e8m0_k32":
        from benchmarks.experimental.b12x.benchmark_ds4_moe import _make_quantized_stack

        gen = torch.Generator(device=device)
        gen.manual_seed(10_000 + layer_idx)
        w13_weight, w13_blockscale_swizzled = _make_quantized_stack(
            E,
            w13_rows,
            K,
            gen=gen,
            device=device,
        )
        w2_weight, w2_blockscale_swizzled = _make_quantized_stack(
            E,
            K,
            I_tp,
            gen=gen,
            device=device,
        )
        w13_layout = "w13"
    else:
        w13_weight = torch.empty(E, w13_rows, K // 2, dtype=torch.uint8, device=device)
        w2_weight = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)
        # Distinct signed weights/scales expose route, layout, and scale bugs
        # that identical experts with constant positive weights cannot catch.
        gen = torch.Generator(device=device).manual_seed(10_000 + layer_idx)
        w13_weight.random_(0, 256, generator=gen)
        w2_weight.random_(0, 256, generator=gen)
        w13_sf = torch.empty(E, w13_rows, K // 16, device=device).uniform_(
            0.01, 0.03, generator=gen,
        ).to(torch.float8_e4m3fn)
        down_sf = torch.empty(E, K, I_tp // 16, device=device).uniform_(
            0.01, 0.03, generator=gen,
        ).to(torch.float8_e4m3fn)
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)
        w13_layout = "w31" if activation == SWIGLUOAI_UNINTERLEAVE else "w13"

    if source_format == "fp4_e8m0_k32":
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            raise RuntimeError("shape-only E8M0/K32 scales require torch.float8_e8m0fnu")

    w13_permuted = w13_weight.permute(1, 2, 0)
    down_permuted = w2_weight.permute(1, 2, 0)
    if source_format == "fp4_e8m0_k32":
        w13_scale = w13_blockscale_swizzled
        down_scale = w2_blockscale_swizzled
    else:
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), w13_rows, K)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

    w13_input_scale_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    w2_input_scale_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    w13_input_scale = w13_input_scale_per_expert.max()
    w2_input_scale = w2_input_scale_per_expert.max()
    g1_alphas_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    g2_alphas_per_expert = torch.ones(E, dtype=torch.float32, device=device)
    if source_format == "modelopt_nvfp4":
        # Normalize the synthetic down projection's output units. FC1 and
        # activation quantization still see the full random dynamic range.
        g2_alphas_per_expert.mul_(1.0 / 64.0)
    g1_alphas = g1_alphas_per_expert
    g2_alphas = g2_alphas_per_expert
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)
    w13_input_scale_quant_per_expert = (1.0 / w13_input_scale_per_expert).to(torch.float32).contiguous()
    w2_input_scale_quant_per_expert = (1.0 / w2_input_scale_per_expert).to(torch.float32).contiguous()
    print(" done.")

    return ExpertWeights(
        layer_idx=layer_idx,
        spec=spec,
        w13_permuted=w13_permuted,
        w13_scale=w13_scale,
        down_permuted=down_permuted,
        down_scale=down_scale,
        w13_weight=w13_weight,
        w13_blockscale_swizzled=w13_blockscale_swizzled,
        w2_weight=w2_weight,
        w2_blockscale_swizzled=w2_blockscale_swizzled,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
        w13_input_scale_quant=w13_input_scale_quant,
        w2_input_scale_quant=w2_input_scale_quant,
        w13_input_scale_per_expert=w13_input_scale_per_expert,
        w2_input_scale_per_expert=w2_input_scale_per_expert,
        w13_input_scale_quant_per_expert=w13_input_scale_quant_per_expert,
        w2_input_scale_quant_per_expert=w2_input_scale_quant_per_expert,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
        source_format=source_format,
        w13_layout=w13_layout,
    )


def load_iq2_xs_expert_weights(model_path, spec, *, layer_idx, activation, device="cuda") -> ExpertWeights:
    """Keep CPU source blocks for the oracle and prepare compact device storage."""
    from benchmarks.experimental.b12x.iq2_xs_checkpoint import load_iq2_xs_layer

    layer = load_iq2_xs_layer(model_path, layer=layer_idx, tp_size=spec.tp_size, tp_rank=spec.tp_rank)
    if activation != layer.activation:
        raise ValueError(f"the IQ2_XS checkpoint requires {layer.activation}")
    if (layer.hidden_size, layer.intermediate_size, layer.route_num_experts, layer.top_k) != (spec.hidden_size, spec.I_tp, spec.num_experts, spec.top_k):
        raise ValueError("IQ2_XS checkpoint geometry differs from the benchmark specification")
    unit = torch.ones(spec.num_experts, device=device, dtype=torch.float32)
    return ExpertWeights(
        layer_idx=layer_idx, spec=spec,
        w13_permuted=None, w13_scale=None, down_permuted=None, down_scale=None,
        w13_weight=layer.weights.w13, w2_weight=layer.weights.w2,
        w13_blockscale_swizzled=None, w2_blockscale_swizzled=None,
        w13_input_scale=unit[:1], w2_input_scale=unit[:1],
        w13_input_scale_quant=unit[:1], w2_input_scale_quant=unit[:1],
        w13_input_scale_per_expert=unit, w2_input_scale_per_expert=unit,
        w13_input_scale_quant_per_expert=unit, w2_input_scale_quant_per_expert=unit,
        g1_alphas=unit, g2_alphas=unit, g1_alphas_per_expert=unit, g2_alphas_per_expert=unit,
        source_format=layer.weights.codec, w13_layout="w31",
    )


def load_expert_weights(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
) -> ExpertWeights:
    activation = normalize_moe_activation(activation)

    if checkpoint_family in {"qwen_iq2_xs", "puzzle3_iq2_xs"}:
        return load_iq2_xs_expert_weights(model_path, spec, layer_idx=layer_idx, activation=activation)

    device = torch.device("cuda")
    E = spec.num_experts
    K = spec.hidden_size
    I_tp = spec.I_tp
    source_format = "modelopt_nvfp4"
    w4a16_w13_global_scale = None
    w4a16_w2_global_scale = None
    gate_weight = None
    gate_bias = None
    gate_tid2eid = None
    gate_score_func = "softmax"
    gate_route_scale = 1.0
    gate_norm_topk_prob = True
    oracle_w13_weight = None
    oracle_w13_scale = None
    oracle_w2_weight = None
    oracle_w2_scale = None
    w13_layout = "w31"
    if checkpoint_family in {
        "nano35_w4a16_shape",
        "dsv4f_shape",
        "dsv4f_nvfp4_shape",
        "glm53_flash_shape",
        "laguna_s21_shape",
        "minimax_m3_shape",
        "qwen38_flash_next_shape",
        "mimo26_flash_shape",
        "mimo26_flash_nvfp4_shape",
    }:
        shape_source_format = (
            "fp4_e8m0_k32"
            if checkpoint_family in {"dsv4f_shape", "mimo26_flash_shape"}
            else "modelopt_nvfp4"
        )
        return make_shape_only_expert_weights(
            spec,
            layer_idx=layer_idx,
            activation=activation,
            source_format=shape_source_format,
        )

    from benchmarks.experimental.b12x.checkpoint_loader import IndexedSafetensorLoader

    cfg = _load_config(model_path)
    loader = IndexedSafetensorLoader(model_path)

    if checkpoint_family in {
        "qwen",
        "glm",
        "glm53_flash",
        "minimax_m2",
        "minimax_m3",
    }:
        if (
            checkpoint_family in {"glm", "glm53_flash", "minimax_m2"}
            and activation != "silu"
        ):
            raise ValueError(f"{checkpoint_family} FP4 benchmark only supports silu experts")
        if checkpoint_family == "minimax_m3" and activation != SWIGLUOAI_UNINTERLEAVE:
            raise ValueError("minimax_m3 FP4 benchmark expects swigluoai_uninterleave experts")
        if checkpoint_family == "qwen":
            cfg_num_experts = cfg["num_experts"]
            cfg_intermediate_size = cfg["moe_intermediate_size"]
            prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
            gate_proj = "gate_proj"
            up_proj = "up_proj"
            down_proj = "down_proj"
        elif checkpoint_family in {"minimax_m2", "minimax_m3"}:
            cfg_num_experts = cfg["num_local_experts"]
            cfg_intermediate_size = cfg["intermediate_size"]
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts"
            gate_proj = "w1"
            up_proj = "w3"
            down_proj = "w2"
        elif checkpoint_family == "glm":
            cfg_num_experts = cfg["n_routed_experts"]
            cfg_intermediate_size = cfg["moe_intermediate_size"]
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            gate_proj = "gate_proj"
            up_proj = "up_proj"
            down_proj = "down_proj"
        else:
            cfg_num_experts = cfg["n_routed_experts"]
            cfg_intermediate_size = cfg["moe_intermediate_size"]
            prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
            gate_proj = "gate_proj"
            up_proj = "up_proj"
            down_proj = "down_proj"
        assert cfg_num_experts == spec.num_experts
        assert cfg_intermediate_size == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        gate_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        gate_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

        gate_gs = torch.empty(E, dtype=torch.float32, device=device)
        down_gs = torch.empty(E, dtype=torch.float32, device=device)
        gate_is = torch.empty(E, dtype=torch.float32, device=device)
        down_is = torch.empty(E, dtype=torch.float32, device=device)

        print(f"  Loading {E} experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

            gate_w[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight").narrow(0, tp_off, I_tp).to(device)
            gate_sf[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight_scale").narrow(0, tp_off, I_tp).to(device)
            gate_gs[eid] = loader.get_tensor(f"{ep}.{gate_proj}.weight_scale_2").to(device)
            gate_is[eid] = loader.get_tensor(
                f"{ep}.{gate_proj}.input_scale"
            ).to(device)

            up_w[eid] = loader.get_tensor(f"{ep}.{up_proj}.weight").narrow(0, tp_off, I_tp).to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.{up_proj}.weight_scale").narrow(0, tp_off, I_tp).to(device)

            down_w[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            down_gs[eid] = loader.get_tensor(f"{ep}.{down_proj}.weight_scale_2").to(device)
            down_is[eid] = loader.get_tensor(
                f"{ep}.{down_proj}.input_scale"
            ).to(device)
        print(" done.")

        if checkpoint_family == "minimax_m3":
            w13_layout = "w31"
            w13_weight = torch.cat([gate_w, up_w], dim=1).contiguous()
            w13_sf = torch.cat([gate_sf, up_sf], dim=1).contiguous()
        else:
            w13_layout = "w13"
            w13_weight = torch.cat([up_w, gate_w], dim=1).contiguous()
            w13_sf = torch.cat([up_sf, gate_sf], dim=1).contiguous()
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), 2 * I_tp, K)
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

        w13_input_scale = gate_is.max()
        w2_input_scale = down_is.max()
        g1_alphas = (w13_input_scale * gate_gs).to(torch.float32)
        g2_alphas = (w2_input_scale * down_gs).to(torch.float32)
        w13_input_scale_per_expert = gate_is
        g1_alphas_per_expert = (gate_is * gate_gs).to(torch.float32)
    elif checkpoint_family == "nemotron":
        if activation != "relu2":
            raise ValueError("Nemotron backbone FP4 benchmark expects relu2 experts")
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] // TP_SIZE == spec.hidden_size

        prefix = f"backbone.layers.{layer_idx}.mixer.experts"
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

        up_gs = torch.empty(E, dtype=torch.float32, device=device)
        down_gs = torch.empty(E, dtype=torch.float32, device=device)
        up_is = torch.empty(E, dtype=torch.float32, device=device)
        down_is = torch.empty(E, dtype=torch.float32, device=device)

        print(f"  Loading {E} experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

            if spec.tp_size > 1:
                up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").narrow(0, tp_off, I_tp).to(device)
                up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
                down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").narrow(1, tp_off_packed, I_tp // 2).to(device)
                down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            else:
                up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight").to(device)
                up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").to(device)
                down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight").to(device)
                down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").to(device)
            up_gs[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale_2").to(device)
            up_is[eid] = loader.get_tensor(f"{ep}.up_proj.input_scale").to(device)
            down_gs[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale_2").to(device)
            down_is[eid] = loader.get_tensor(f"{ep}.down_proj.input_scale").to(device)
        print(" done.")

        w13_weight = up_w.contiguous()
        w13_sf = up_sf.contiguous()
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), I_tp, K)
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

        w13_input_scale = up_is.max()
        w2_input_scale = down_is.max()
        g1_alphas = (w13_input_scale * up_gs).to(torch.float32)
        g2_alphas = (w2_input_scale * down_gs).to(torch.float32)
        w13_input_scale_per_expert = up_is
        g1_alphas_per_expert = (up_is * up_gs).to(torch.float32)
    elif checkpoint_family == "nano35_w4a16":
        if activation != "relu2":
            raise ValueError("Nano3.5 W4A16 benchmark expects relu2 experts")
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == spec.intermediate_size
        assert cfg["hidden_size"] == spec.hidden_size

        prefix = f"backbone.layers.{layer_idx}.mixer.experts"
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        up_sf = torch.empty(E, I_tp, K // 16, dtype=torch.float8_e4m3fn, device=device)
        down_sf = torch.empty(E, K, I_tp // 16, dtype=torch.float8_e4m3fn, device=device)

        up_weight_global_scale = torch.empty(E, dtype=torch.float32, device=device)
        down_weight_global_scale = torch.empty(E, dtype=torch.float32, device=device)

        print(f"  Loading {E} CT W4A16 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            tp_off = spec.tp_rank * I_tp
            tp_off_packed = spec.tp_rank * (I_tp // 2)
            tp_sf_cols = I_tp // 16
            tp_sf_off = spec.tp_rank * tp_sf_cols

            up_w[eid] = loader.get_tensor(f"{ep}.up_proj.weight_packed").narrow(0, tp_off, I_tp).to(device)
            up_sf[eid] = loader.get_tensor(f"{ep}.up_proj.weight_scale").narrow(0, tp_off, I_tp).to(device)
            up_weight_global_scale[eid] = loader.get_tensor(f"{ep}.up_proj.weight_global_scale").to(device)

            down_w[eid] = loader.get_tensor(f"{ep}.down_proj.weight_packed").narrow(1, tp_off_packed, I_tp // 2).to(device)
            down_sf[eid] = loader.get_tensor(f"{ep}.down_proj.weight_scale").narrow(1, tp_sf_off, tp_sf_cols).to(device)
            down_weight_global_scale[eid] = loader.get_tensor(f"{ep}.down_proj.weight_global_scale").to(device)
        print(" done.")

        # The compressed-tensors W4A16 checkpoint stores per-tensor global
        # scales as the inverse convention of the B12X W4A16 path.  Fold
        # 1 / weight_global_scale into the FP4 block scales and launch with
        # unit alphas so the benchmark exercises the same model contract.
        up_sf = (
            up_sf.float() * (1.0 / up_weight_global_scale).view(E, 1, 1)
        ).to(torch.float8_e4m3fn).contiguous()
        down_sf = (
            down_sf.float() * (1.0 / down_weight_global_scale).view(E, 1, 1)
        ).to(torch.float8_e4m3fn).contiguous()

        w13_weight = up_w.contiguous()
        w13_sf = up_sf
        w13_blockscale_swizzled = swizzle_block_scale(w13_sf)
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = swizzle_block_scale(down_sf)

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = as_grouped_scale_view(w13_blockscale_swizzled.view(torch.uint8), I_tp, K)
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = as_grouped_scale_view(w2_blockscale_swizzled.view(torch.uint8), K, I_tp)

        ones = torch.ones(E, dtype=torch.float32, device=device)
        w13_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w2_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w13_input_scale_per_expert = ones
        down_is = ones
        down_gs = ones
        g1_alphas = ones
        g2_alphas = ones
        g1_alphas_per_expert = ones
    elif checkpoint_family in {"deepseek_v4_flash", "deepseek_v41_flash"}:
        uses_v41_checkpoint_layout = checkpoint_family == "deepseek_v41_flash"
        family_label = "DeepSeek V4.1 Flash" if uses_v41_checkpoint_layout else "DeepSeek V4 Flash"
        if uses_v41_checkpoint_layout:
            if cfg.get("model_type") != "deepseek_v41_text":
                raise ValueError("V4.1 requires a V4.1 checkpoint")
            if not 0 <= layer_idx < cfg["num_hidden_layers"]:
                raise ValueError("V4.1 Flash profile requires a target-model layer, not DSpark")
        if activation != "silu":
            raise ValueError(f"{family_label} FP4 benchmark expects silu experts")
        if spec.hidden_size % 32 != 0 or spec.I_tp % 32 != 0:
            raise ValueError(
                f"{family_label} requires K and local intermediate size divisible by 32, "
                f"got K={spec.hidden_size}, I_tp={spec.I_tp}"
            )
        assert cfg["n_routed_experts"] == spec.num_experts
        assert cfg["moe_intermediate_size"] == (
            spec.global_intermediate_size
            if uses_v41_checkpoint_layout
            else spec.intermediate_size
        )
        assert cfg["hidden_size"] == spec.hidden_size

        source_format = "fp4_e8m0_k32"
        w13_layout = "w31"
        prefix = f"layers.{layer_idx}.ffn.experts"
        gate_proj = "w1"
        up_proj = "w3"
        down_proj = "w2"
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            raise RuntimeError(f"{family_label} FP4 scales require torch.float8_e8m0fnu")

        gate_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        up_w = torch.empty(E, I_tp, K // 2, dtype=torch.uint8, device=device)
        down_w = torch.empty(E, K, I_tp // 2, dtype=torch.uint8, device=device)

        gate_sf = torch.empty(E, I_tp, K // 32, dtype=e8m0_dtype, device=device)
        up_sf = torch.empty(E, I_tp, K // 32, dtype=e8m0_dtype, device=device)
        down_sf = torch.empty(E, K, I_tp // 32, dtype=e8m0_dtype, device=device)
        if uses_v41_checkpoint_layout:
            def shard(
                tensor: torch.Tensor,
                dimension: int,
                packing: int = 1,
            ) -> torch.Tensor:
                return _slice_v41_tp_shard(
                    tensor,
                    dimension=dimension,
                    intermediate_size_per_partition=spec.logical_I_tp,
                    tp_rank=spec.tp_rank,
                    packing=packing,
                )


        print(f"  Loading {E} {family_label} FP4 experts...", end="", flush=True)
        for eid in range(E):
            ep = f"{prefix}.{eid}"
            if uses_v41_checkpoint_layout:
                gate_w[eid] = shard(
                    _fp4_checkpoint_bytes(loader.get_tensor(f"{ep}.{gate_proj}.weight")),
                    0,
                ).to(device)
                gate_sf[eid] = shard(
                    loader.get_tensor(f"{ep}.{gate_proj}.scale"), 0,
                ).to(device)
                up_w[eid] = shard(
                    _fp4_checkpoint_bytes(loader.get_tensor(f"{ep}.{up_proj}.weight")),
                    0,
                ).to(device)
                up_sf[eid] = shard(
                    loader.get_tensor(f"{ep}.{up_proj}.scale"), 0,
                ).to(device)
                down_w[eid] = shard(
                    _fp4_checkpoint_bytes(loader.get_tensor(f"{ep}.{down_proj}.weight")),
                    1,
                    packing=2,
                ).to(device)
                down_sf[eid] = shard(
                    loader.get_tensor(f"{ep}.{down_proj}.scale"),
                    1,
                    packing=32,
                ).to(device)
            else:
                tp_off = spec.tp_rank * I_tp
                tp_off_packed = tp_off // 2
                tp_sf_cols = I_tp // 32
                tp_sf_off = tp_off // 32
                gate_w[eid] = _fp4_checkpoint_bytes(
                    loader.get_tensor(f"{ep}.{gate_proj}.weight")
                ).narrow(0, tp_off, I_tp).to(device)
                gate_sf[eid] = loader.get_tensor(
                    f"{ep}.{gate_proj}.scale"
                ).narrow(0, tp_off, I_tp).to(device)
                up_w[eid] = _fp4_checkpoint_bytes(
                    loader.get_tensor(f"{ep}.{up_proj}.weight")
                ).narrow(0, tp_off, I_tp).to(device)
                up_sf[eid] = loader.get_tensor(
                    f"{ep}.{up_proj}.scale"
                ).narrow(0, tp_off, I_tp).to(device)
                down_w[eid] = _fp4_checkpoint_bytes(
                    loader.get_tensor(f"{ep}.{down_proj}.weight")
                ).narrow(1, tp_off_packed, I_tp // 2).to(device)
                down_sf[eid] = loader.get_tensor(
                    f"{ep}.{down_proj}.scale"
                ).narrow(1, tp_sf_off, tp_sf_cols).to(device)
        print(" done.")

        # Match vLLM FusedMoE loading for the B12X backend: native DeepSeek V4
        # FP4 W13 source is contiguous [w1/gate, w3/up], and B12X receives it
        # without an additional row swap.
        w13_weight = torch.cat([gate_w, up_w], dim=1).contiguous()
        w13_sf = torch.cat([gate_sf, up_sf], dim=1).contiguous()
        w13_sf.view(torch.uint8).clamp_(max=247)
        down_sf.view(torch.uint8).clamp_(max=247)
        w13_blockscale_swizzled = w13_sf
        w2_weight = down_w.contiguous()
        w2_blockscale_swizzled = down_sf.contiguous()

        w13_permuted = w13_weight.permute(1, 2, 0)
        w13_scale = w13_sf
        down_permuted = w2_weight.permute(1, 2, 0)
        down_scale = down_sf

        ones = torch.ones(E, dtype=torch.float32, device=device)
        w13_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w2_input_scale = torch.ones((), dtype=torch.float32, device=device)
        w13_input_scale_per_expert = ones
        down_is = ones
        down_gs = ones
        g1_alphas = ones
        g2_alphas = ones
        g1_alphas_per_expert = ones

        gate_prefix = f"layers.{layer_idx}.ffn.gate"
        gate_weight = loader.get_tensor(f"{gate_prefix}.weight").to(device=device).contiguous()
        bias_key = f"{gate_prefix}.bias"
        if bias_key in loader.weight_map:
            gate_bias = loader.get_tensor(bias_key).to(device=device).contiguous()
        tid2eid_key = f"{gate_prefix}.tid2eid"
        if tid2eid_key in loader.weight_map:
            gate_tid2eid = loader.get_tensor(tid2eid_key).to(device=device).contiguous()
        gate_score_func = str(cfg.get("scoring_func", "softmax")).lower()
        gate_route_scale = float(cfg.get("routed_scaling_factor", 1.0))
        gate_norm_topk_prob = bool(cfg.get("norm_topk_prob", True))
    else:
        raise ValueError(f"unsupported checkpoint family {checkpoint_family!r}")

    g2_alphas_per_expert = (down_is * down_gs).to(torch.float32)
    w13_input_scale_quant = (1.0 / w13_input_scale).to(torch.float32)
    w2_input_scale_quant = (1.0 / w2_input_scale).to(torch.float32)
    w13_input_scale_quant_per_expert = (1.0 / w13_input_scale_per_expert).to(torch.float32).contiguous()
    w2_input_scale_quant_per_expert = (1.0 / down_is).to(torch.float32).contiguous()

    return ExpertWeights(
        layer_idx=layer_idx,
        spec=spec,
        w13_permuted=w13_permuted,
        w13_scale=w13_scale,
        down_permuted=down_permuted,
        down_scale=down_scale,
        w13_weight=w13_weight,
        w13_blockscale_swizzled=w13_blockscale_swizzled,
        w2_weight=w2_weight,
        w2_blockscale_swizzled=w2_blockscale_swizzled,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
        w13_input_scale_quant=w13_input_scale_quant,
        w2_input_scale_quant=w2_input_scale_quant,
        w13_input_scale_per_expert=w13_input_scale_per_expert,
        w2_input_scale_per_expert=down_is,
        w13_input_scale_quant_per_expert=w13_input_scale_quant_per_expert,
        w2_input_scale_quant_per_expert=w2_input_scale_quant_per_expert,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        g1_alphas_per_expert=g1_alphas_per_expert,
        g2_alphas_per_expert=g2_alphas_per_expert,
        source_format=source_format,
        w4a16_w13_global_scale=w4a16_w13_global_scale,
        w4a16_w2_global_scale=w4a16_w2_global_scale,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
        gate_tid2eid=gate_tid2eid,
        gate_score_func=gate_score_func,
        gate_route_scale=gate_route_scale,
        gate_norm_topk_prob=gate_norm_topk_prob,
        oracle_w13_weight=oracle_w13_weight,
        oracle_w13_scale=oracle_w13_scale,
        oracle_w2_weight=oracle_w2_weight,
        oracle_w2_scale=oracle_w2_scale,

        w13_layout=w13_layout,
    )


def load_expert_weight_stack(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_start: int,
    num_layers: int,
    activation: str = "silu",
    checkpoint_family: str = "qwen",
) -> list[ExpertWeights]:
    return [
        load_expert_weights(
            model_path,
            spec,
            layer_idx=layer_start + layer_offset,
            activation=activation,
            checkpoint_family=checkpoint_family,

        )
        for layer_offset in range(num_layers)
    ]


def load_gate_weight(
    model_path: pathlib.Path,
    spec: ModelSpec,
    *,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Load the replicated sparse-gate projection for a Qwen-style MoE block."""
    from benchmarks.experimental.b12x.checkpoint_loader import IndexedSafetensorLoader

    cfg = _load_config(model_path)
    assert cfg["num_experts"] == spec.num_experts
    assert cfg["hidden_size"] == spec.hidden_size

    gate_weight = IndexedSafetensorLoader(model_path).get_tensor(
        f"model.language_model.layers.{layer_idx}.mlp.gate.weight"
    )
    expected_shape = (spec.num_experts, spec.hidden_size)
    if tuple(gate_weight.shape) != expected_shape:
        raise ValueError(
            f"expected gate.weight shape {expected_shape}, got {tuple(gate_weight.shape)}"
        )
    return gate_weight.to(device=torch.device("cuda")).contiguous()


def make_input_activations(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(m, spec.hidden_size, generator=generator, dtype=torch.float32)
    return x.to(device=device, dtype=torch.bfloat16)


def normalize_kernel_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the graph-safe routing tensor contract used by serving."""
    return (
        topk_ids.to(dtype=torch.int32).contiguous(),
        topk_weights.to(dtype=torch.float32).contiguous(),
    )


def make_routed_inputs(
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = make_input_activations(spec, m, seed, device)
    routing_generator = torch.Generator(device="cpu")
    routing_generator.manual_seed(seed + 1)
    routing_logits = torch.randn(
        m,
        spec.num_experts,
        generator=routing_generator,
        dtype=torch.float32,
    ).to(device=device)
    topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids, topk_weights = normalize_kernel_routing(topk_ids, topk_weights)
    return x, topk_ids, topk_weights


def compute_model_gate_routing(
    weights: ExpertWeights,
    x: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.gate_weight is None:
        raise ValueError("model gate routing requires gate_weight")
    scores = F.linear(x.float(), weights.gate_weight.float())
    score_func = weights.gate_score_func
    if score_func == "softmax":
        original_scores = torch.softmax(scores, dim=-1)
    elif score_func == "sigmoid":
        original_scores = torch.sigmoid(scores)
    elif score_func == "sqrtsoftplus":
        original_scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"unsupported model gate score function {score_func!r}")

    if weights.gate_tid2eid is not None:
        routing_generator = torch.Generator(device="cpu")
        routing_generator.manual_seed(seed + 17)
        input_ids = torch.randint(
            0,
            weights.gate_tid2eid.shape[0],
            (x.shape[0],),
            generator=routing_generator,
            dtype=torch.int64,
        ).to(device=x.device)
        topk_ids = weights.gate_tid2eid[input_ids].to(device=x.device)
    else:
        selection_scores = original_scores
        if weights.gate_bias is not None:
            selection_scores = selection_scores + weights.gate_bias.to(device=x.device)
        _topk_scores, topk_ids = torch.topk(selection_scores, weights.spec.top_k, dim=-1)

    topk_weights = original_scores.gather(1, topk_ids)
    if score_func != "softmax" and weights.gate_norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
    topk_weights = topk_weights * weights.gate_route_scale
    return normalize_kernel_routing(topk_ids, topk_weights)


def make_profile_routed_inputs(
    profile: ModelProfile,
    weights: ExpertWeights,
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = make_input_activations(spec, m, seed, device)
    if profile.default_routing == "model":
        topk_ids, topk_weights = compute_model_gate_routing(weights, x, seed=seed + 1)
        return x, topk_ids, topk_weights
    if profile.default_routing != "synthetic":
        raise ValueError(f"unsupported routing source {profile.default_routing!r}")
    _x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed, device)
    return x, topk_ids, topk_weights


def make_benchmark_case(
    profile: ModelProfile,
    weights: ExpertWeights,
    spec: ModelSpec,
    m: int,
    seed: int,
    device: torch.device,
    *,
    routing_workload: str = "default",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Build a reproducible validation/timing case without retaining it."""

    x = make_input_activations(spec, m, seed, device)
    if routing_workload != "default":
        from b12x.moe.fused_moe.workloads import make_routing_ids

        topk_ids = make_routing_ids(
            m, spec.top_k, spec.num_experts,
            workload=routing_workload, seed=seed + 1, device=device,
        )
        topk_weights = torch.softmax(
            torch.arange(spec.top_k, dtype=torch.float32, device=device) * .125,
            dim=-1,
        ).expand(m, -1).contiguous()
        topk_weights.mul_(getattr(weights, "gate_route_scale", 1.0))
        return x, topk_ids, topk_weights, None
    if profile.default_routing == "model":
        topk_ids, topk_weights = compute_model_gate_routing(
            weights,
            x,
            seed=seed + 1,
        )
        return x, topk_ids, topk_weights, None
    if profile.default_routing != "synthetic":
        raise ValueError(f"unsupported routing source {profile.default_routing!r}")
    routing_generator = torch.Generator(device="cpu")
    routing_generator.manual_seed(seed + 1)
    routing_logits = torch.randn(
        m,
        spec.num_experts,
        generator=routing_generator,
        dtype=torch.float32,
    ).to(device=device)
    topk_logits, topk_ids = torch.topk(routing_logits, spec.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids, topk_weights = normalize_kernel_routing(topk_ids, topk_weights)
    return x, topk_ids, topk_weights, routing_logits


def repeat_routing_pattern(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    period: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat the first ``period`` token routes across the full batch.

    This models concurrent requests following the same speculative-verify
    trajectory: token M grows with concurrency while the active-expert set
    stays close to the one-request route set.
    """

    if period <= 0:
        return topk_ids, topk_weights
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk ids and weights must have the same shape, got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if topk_ids.dim() != 2:
        raise ValueError(f"topk tensors must be rank 2, got rank {topk_ids.dim()}")
    tokens = int(topk_ids.shape[0])
    if period > tokens:
        raise ValueError(
            f"routing repeat period {period} exceeds token count {tokens}"
        )
    repeats = (tokens + period - 1) // period
    return (
        topk_ids[:period].repeat((repeats, 1))[:tokens].contiguous(),
        topk_weights[:period].repeat((repeats, 1))[:tokens].contiguous(),
    )


def _make_structured_routing_ids(
    spec: ModelSpec,
    m: int,
    *,
    layer_idx: int,
    pattern: str,
    seed: int,
) -> torch.Tensor:
    if pattern not in {"disjoint", "overlap", "random"}:
        raise ValueError(f"unsupported routing pattern {pattern!r}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_idx * 101)
    top_k = spec.top_k
    expert_count = spec.num_experts

    if pattern == "random":
        ids = torch.empty(m, top_k, dtype=torch.int64)
        for token_idx in range(m):
            ids[token_idx] = torch.randperm(expert_count, generator=generator)[:top_k]
        return ids

    pool_size = min(expert_count, max(top_k, m * top_k))
    layer_stride = pool_size if pattern == "disjoint" else max(top_k, pool_size // 2)
    pool_start = (seed * 17 + layer_idx * layer_stride) % expert_count
    pool = (torch.arange(pool_size, dtype=torch.int64) + pool_start) % expert_count
    pool = pool[torch.randperm(pool_size, generator=generator)]

    ids = torch.empty(m, top_k, dtype=torch.int64)
    cursor = 0
    for token_idx in range(m):
        if cursor + top_k > pool_size:
            pool = pool[torch.randperm(pool_size, generator=generator)]
            cursor = 0
        ids[token_idx] = pool[cursor:cursor + top_k]
        cursor += top_k
    return ids


def make_multilayer_routing_case(
    spec: ModelSpec,
    m: int,
    num_layers: int,
    device: torch.device,
    *,
    pattern: str,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    routing_case: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(num_layers):
        topk_ids = _make_structured_routing_ids(
            spec,
            m,
            layer_idx=layer_idx,
            pattern=pattern,
            seed=seed,
        ).to(device=device, dtype=torch.int32)
        weight_generator = torch.Generator(device="cpu")
        weight_generator.manual_seed(seed + layer_idx * 1009 + 7)
        topk_logits = torch.randn(m, spec.top_k, generator=weight_generator, dtype=torch.float32)
        topk_weights = torch.softmax(topk_logits, dim=-1).to(device=device)
        routing_case.append((topk_ids, topk_weights))
    return routing_case


def get_scale_contract_params(weights: ExpertWeights, scale_contract: str) -> ScaleContractParams:
    if scale_contract == "per-expert":
        return ScaleContractParams(
            a1_gscale=weights.w13_input_scale_quant_per_expert,
            a2_gscale=weights.w2_input_scale_quant_per_expert,
            g1_alphas=weights.g1_alphas_per_expert,
            g2_alphas=weights.g2_alphas_per_expert,
        )
    if scale_contract == "shared":
        return ScaleContractParams(
            a1_gscale=weights.w13_input_scale_quant,
            a2_gscale=weights.w2_input_scale_quant,
            g1_alphas=weights.g1_alphas,
            g2_alphas=weights.g2_alphas,
        )
    raise ValueError(f"Unsupported scale contract: {scale_contract}")


def get_quant_mode_params(
    weights: ExpertWeights,
    scale_contract: str,
    quant_mode: str,
) -> ScaleContractParams:
    params = get_scale_contract_params(weights, scale_contract)
    quant_mode = quant_mode.lower()
    if quant_mode == "nvfp4":
        return params
    if quant_mode in {"w4a8_mx", "w4a8_nvfp4"}:
        return params
    if quant_mode == "w4a16":
        # W4A16 keeps activations in BF16, so remove the activation input
        # scale factor from the fused alpha and leave only the weight global
        # scale component.
        return ScaleContractParams(
            a1_gscale=params.a1_gscale,
            a2_gscale=params.a2_gscale,
            g1_alphas=(params.g1_alphas * params.a1_gscale).to(torch.float32).contiguous(),
            g2_alphas=(params.g2_alphas * params.a2_gscale).to(torch.float32).contiguous(),
        )
    raise ValueError(f"Unsupported quant mode: {quant_mode}")


def get_w4a16_prepare_scales(
    weights: ExpertWeights,
    params: ScaleContractParams,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if weights.source_format == "compressed_tensors":
        if weights.w4a16_w13_global_scale is None or weights.w4a16_w2_global_scale is None:
            raise ValueError("compressed_tensors W4A16 weights require raw global scales")
        return (
            weights.w4a16_w13_global_scale,
            weights.w4a16_w2_global_scale,
            weights.source_format,
        )
    return params.g1_alphas, params.g2_alphas, weights.source_format


def plan_b12x_benchmark_weights(
    weights: ExpertWeights,
    *,
    quant_mode: str,
    activation: str,
    w4a16_native: bool = False,
    activation_params: ActivationParams | None = None,
):
    """Declare the benchmark's canonical checkpoint representation."""
    activation_params = activation_params or ActivationParams()
    modes = {
        "nvfp4": fused_moe.ActivationMode.A4,
        "w4a16": fused_moe.ActivationMode.A16,
        "w4a8_mx": fused_moe.ActivationMode.A8,
        "w4a8_nvfp4": fused_moe.ActivationMode.A8,
    }
    try:
        mode = modes[quant_mode.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark quant mode {quant_mode!r}") from exc
    return fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=weights.source_format, w13_layout=weights.w13_layout,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=weights.spec.num_experts,
            hidden_size=weights.spec.hidden_size,
            intermediate_size=weights.spec.I_tp,
        ),
        activation=fused_moe.ActivationSpec(
            mode=mode,
            nonlinearity=activation,
            io_dtype=torch.bfloat16,
            **activation_params.kwargs(),
        ),
        constraints=fused_moe.WeightPlanConstraints(
            required_packing=(
                fused_moe.WeightPacking.SOURCE_NATIVE
                if quant_mode.lower() == "w4a16" and w4a16_native
                else None
            ),
        ),
    )

def prepare_b12x_benchmark_weights(
    weights: ExpertWeights,
    params: ScaleContractParams,
    *,
    quant_mode: str,
    activation: str,
    w4a16_native: bool = False,
    plan=None,
    activation_params: ActivationParams | None = None,
):
    """Prepare canonical weights for a declaration, never an executable plan."""
    quant_mode = quant_mode.lower()
    if plan is None:
        plan = plan_b12x_benchmark_weights(
            weights,
            quant_mode=quant_mode,
            activation=activation,
            w4a16_native=w4a16_native,
            activation_params=activation_params,
        )
    if weights.source_format in IQ2_CODECS:
        if quant_mode != "w4a16":
            raise ValueError("IQ2_XS benchmark requires W4A16")
        experts = fused_moe.prepare_weights(
            plan=plan,
            weights=fused_moe.IQ2XSWeights(
                weights.w13_weight.to(params.g1_alphas.device),
                weights.w2_weight.to(params.g2_alphas.device),
                codec=weights.source_format,
            ),
        )
        return experts, params
    if quant_mode == "w4a16":
        w1_global_scale, w2_global_scale, _ = get_w4a16_prepare_scales(weights, params)
    elif quant_mode == "w4a8_mx":
        w1_global_scale = torch.ones_like(params.g1_alphas)
        w2_global_scale = torch.ones_like(params.g2_alphas)
    else:
        w1_global_scale = params.g1_alphas * params.a1_gscale
        w2_global_scale = params.g2_alphas * params.a2_gscale
    experts = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=weights.w13_weight,
            w2=weights.w2_weight,
            w13_block_scales=weights.w13_blockscale_swizzled,
            w2_block_scales=weights.w2_blockscale_swizzled,
            w13_global_scales=w1_global_scale,
            w2_global_scales=w2_global_scale,
            input_scale=params.a1_gscale,
            intermediate_scale=params.a2_gscale,
        ),
    )
    if experts._impl.plan.prepares_runtime_alphas:
        params = ScaleContractParams(
            a1_gscale=experts._impl.a1_gscale,
            a2_gscale=experts._impl.a2_gscale,
            g1_alphas=experts._impl.w1_alphas,
            g2_alphas=experts._impl.w2_alphas,
        )
    return experts, params


def get_w4a16_oracle_params(
    weights: ExpertWeights,
    params: ScaleContractParams,
) -> ScaleContractParams:
    if weights.source_format != "compressed_tensors":
        return params
    if weights.w4a16_w13_global_scale is None or weights.w4a16_w2_global_scale is None:
        raise ValueError("compressed_tensors W4A16 weights require raw global scales")
    return ScaleContractParams(
        a1_gscale=params.a1_gscale,
        a2_gscale=params.a2_gscale,
        g1_alphas=(1.0 / weights.w4a16_w13_global_scale).to(torch.float32).contiguous(),
        g2_alphas=(1.0 / weights.w4a16_w2_global_scale).to(torch.float32).contiguous(),
    )




def _dequant_mxfp4_expert(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    fp4_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize one source-layout E2M1/UE8M0 expert."""
    if tuple(packed.shape) != (rows, cols // 2):
        raise ValueError(
            f"expected packed MXFP4 expert {(rows, cols // 2)}, got {tuple(packed.shape)}"
        )
    scale_bytes = scales.view(torch.uint8).reshape(rows, -1)
    if scale_bytes.shape[1] < cols // 32:
        raise ValueError(
            f"expected at least {cols // 32} MXFP4 scales per row, "
            f"got {scale_bytes.shape[1]}"
        )

    packed_u8 = packed.view(torch.uint8)
    if fp4_lut is None:
        fp4_lut = torch.tensor(
            _FP4_E2M1_VALUES,
            dtype=torch.float32,
            device=packed.device,
        )
    lo = fp4_lut[(packed_u8 & 0x0F).to(torch.int64)]
    hi = fp4_lut[((packed_u8 >> 4) & 0x0F).to(torch.int64)]
    raw = torch.stack((lo, hi), dim=-1).reshape(rows, cols)
    block_scales = torch.exp2(
        scale_bytes[:, : cols // 32].to(torch.float32) - 127.0
    )
    return (
        raw.view(rows, cols // 32, 32) * block_scales.unsqueeze(-1)
    ).reshape(rows, cols).to(torch.bfloat16)


def _dequant_nvfp4_expert(
    packed: torch.Tensor,
    swizzled_scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    global_scale: torch.Tensor | float,
    fp4_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand one ModelOpt E2M1/E4M3-K16 expert into logical BF16."""
    if tuple(packed.shape) != (rows, cols // 2):
        raise ValueError(
            f"expected packed NVFP4 expert {(rows, cols // 2)}, got {tuple(packed.shape)}"
        )
    if fp4_lut is None:
        fp4_lut = torch.tensor(
            _FP4_E2M1_VALUES,
            dtype=torch.float32,
            device=packed.device,
        )
    packed_u8 = packed.view(torch.uint8)
    lo = fp4_lut[(packed_u8 & 0x0F).to(torch.int64)]
    hi = fp4_lut[((packed_u8 >> 4) & 0x0F).to(torch.int64)]
    raw = torch.stack((lo, hi), dim=-1).reshape(rows, cols)
    block_scales = unswizzle_block_scale(
        swizzled_scales.view(torch.uint8),
        rows,
        cols // 16,
    )
    logical = (
        raw.view(rows, cols // 16, 16) * block_scales.unsqueeze(-1)
    ).reshape(rows, cols)
    return (logical * torch.as_tensor(global_scale, device=packed.device)).to(
        torch.bfloat16
    )




def force_convert_nvfp4_weights_to_mxfp4(
    weights: ExpertWeights,
    params: ScaleContractParams,
    *,
    activation: str,
) -> ExpertWeights:
    """Re-quantize a ModelOpt NVFP4 MoE source to native MXFP4 E8M0/K32.

    This is intentionally an offline preparation step, before the b12x weight
    planner runs.  Each expert is reconstructed with the checkpoint's complete
    input/weight scale contract and then quantized onto a fresh K/32 power-of-
    two scale grid.  The result is a real ``fp4_e8m0_k32`` source for the
    native W4A8-MX QMMA path, not a runtime adapter over ModelOpt storage.
    """
    if weights.source_format != "modelopt_nvfp4":
        raise ValueError(
            "--force-mxfp4 requires ModelOpt NVFP4 source weights, got "
            f"{weights.source_format!r}"
        )
    spec = weights.spec
    if spec.hidden_size % 32 or spec.I_tp % 32:
        raise ValueError(
            "--force-mxfp4 requires K and I_tp divisible by 32, got "
            f"K={spec.hidden_size}, I_tp={spec.I_tp}"
        )

    from benchmarks.experimental.b12x.benchmark_ds4_moe import quantize_mxfp4_batched

    def requantize_stack(
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        rows: int,
        cols: int,
        global_scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        converted = torch.empty_like(packed, dtype=torch.uint8)
        converted_scales = torch.empty(
            spec.num_experts,
            rows,
            cols // 32,
            dtype=torch.uint8,
            device=packed.device,
        )
        for expert_id in range(spec.num_experts):
            logical = _dequant_nvfp4_expert(
                packed[expert_id],
                scales[expert_id],
                rows=rows,
                cols=cols,
                global_scale=global_scales[expert_id],
            )
            quantized, quantized_scales = quantize_mxfp4_batched(logical.unsqueeze(0))
            converted[expert_id].copy_(quantized[0])
            converted_scales[expert_id].copy_(quantized_scales[0])
        return converted.contiguous(), converted_scales.contiguous()

    # The ModelOpt runtime applies input-scale reciprocal x weight-global-scale
    # as its FC alpha.  Fold that complete factor into the offline logical
    # weights, leaving the MXFP4 source with unit alphas.
    fc1_scales = (params.g1_alphas * params.a1_gscale).to(torch.float32)
    fc2_scales = (params.g2_alphas * params.a2_gscale).to(torch.float32)
    print("  Forcing ModelOpt NVFP4 -> MXFP4 E8M0/K32...", end="", flush=True)
    w13_weight, w13_mx = requantize_stack(
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        rows=moe_activation_w1_rows(activation, spec.I_tp),
        cols=spec.hidden_size,
        global_scales=fc1_scales,
    )
    w2_weight, w2_mx = requantize_stack(
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        rows=spec.hidden_size,
        cols=spec.I_tp,
        global_scales=fc2_scales,
    )
    torch.cuda.synchronize()
    print(" done.")

    ones = torch.ones(spec.num_experts, dtype=torch.float32, device=w13_weight.device)
    one = torch.ones((), dtype=torch.float32, device=w13_weight.device)
    return replace(
        weights,
        w13_permuted=w13_weight.permute(1, 2, 0),
        w13_scale=w13_mx,
        down_permuted=w2_weight.permute(1, 2, 0),
        down_scale=w2_mx,
        w13_weight=w13_weight,
        w13_blockscale_swizzled=w13_mx,
        w2_weight=w2_weight,
        w2_blockscale_swizzled=w2_mx,
        w13_input_scale=one,
        w2_input_scale=one,
        w13_input_scale_quant=one,
        w2_input_scale_quant=one,
        w13_input_scale_per_expert=ones,
        w2_input_scale_per_expert=ones,
        w13_input_scale_quant_per_expert=ones,
        w2_input_scale_quant_per_expert=ones,
        g1_alphas=ones,
        g2_alphas=ones,
        g1_alphas_per_expert=ones,
        g2_alphas_per_expert=ones,
        source_format="fp4_e8m0_k32",
        oracle_w13_weight=None,
        oracle_w13_scale=None,
        oracle_w2_weight=None,
        oracle_w2_scale=None,

    )








def make_oracle_reference(
    oracle_mode: str,
    quant_mode: str,
    x: torch.Tensor,
    weights: ExpertWeights,
    params: ScaleContractParams,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation: str,
    activation_params: ActivationParams | None = None,
    quant_scale_math: str = "direct_division",
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    activation_params = activation_params or ActivationParams()
    spec = weights.spec
    quant_mode = quant_mode.lower()
    if weights.source_format in IQ2_CODECS:
        from b12x.testing.iq2_xs_reference import moe_reference_iq2_xs

        if quant_mode != "w4a16" or oracle_mode != "w4a16":
            raise ValueError("IQ2_XS requires the independent W4A16 block oracle")
        if activation_params.swiglu_alpha is not None or activation_params.swiglu_beta is not None:
            raise ValueError("IQ2_XS oracle supports SiLU without alpha/beta overrides")
        return moe_reference_iq2_xs(
            x, weights.w13_weight, weights.w2_weight, topk_ids, topk_weights,
            activation=activation, w13_layout=weights.w13_layout,
            swiglu_limit=activation_params.swiglu_limit,
        )
    if quant_mode == "w4a16":
        if oracle_mode == "nvfp4":
            raise ValueError("--oracle-mode nvfp4 is not valid with --quant-mode w4a16")
        params = get_w4a16_oracle_params(weights, params)
        if weights.source_format == "fp4_e8m0_k32":
            if oracle_mode not in {"f32", "w4a16"}:
                raise ValueError(f"unsupported W4A16 oracle mode {oracle_mode!r}")
            return moe_reference_w4a16_fp4_e8m0_k32(
                x,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                params.g1_alphas,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                params.g2_alphas,
                topk_ids,
                topk_weights,
                spec.num_experts,
                spec.hidden_size,
                spec.I_tp,
                activation=activation,
                w13_layout=weights.w13_layout,
                **activation_params.kwargs(),
            )
        if oracle_mode == "f32":
            return moe_reference_w4a16_f32(
                x,
                weights.w13_weight,
                weights.w13_blockscale_swizzled,
                params.g1_alphas,
                weights.w2_weight,
                weights.w2_blockscale_swizzled,
                params.g2_alphas,
                topk_ids,
                topk_weights,
                spec.num_experts,
                spec.hidden_size,
                spec.I_tp,
                activation=activation,
                **activation_params.kwargs(),
            )
        if oracle_mode != "w4a16":
            raise ValueError(f"unsupported W4A16 oracle mode {oracle_mode!r}")
        if activation == SWIGLUOAI_UNINTERLEAVE:
            raise ValueError(
                "--oracle-mode w4a16 does not support swigluoai_uninterleave; "
                "use --oracle-mode f32"
            )
        return moe_reference_w4a16(
            x,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            params.g1_alphas,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            params.g2_alphas,
            topk_ids,
            topk_weights,
            spec.num_experts,
            spec.hidden_size,
            spec.I_tp,
            activation=activation,
        )
    if quant_mode in {"w4a8_mx", "w4a8_nvfp4"}:
        if oracle_mode != quant_mode:
            raise ValueError(
                f"--oracle-mode {quant_mode} is required with --quant-mode {quant_mode}"
            )
        if quant_mode == "w4a8_mx":
            if weights.source_format != "fp4_e8m0_k32":
                raise ValueError(
                    "--quant-mode w4a8_mx requires source_format='fp4_e8m0_k32'"
                )
            w13_mx = weights.w13_blockscale_swizzled.view(torch.uint8)
            w2_mx = weights.w2_blockscale_swizzled.view(torch.uint8)
            w13_residual = None
            w2_residual = None
            alpha1 = params.g1_alphas
            alpha2 = params.g2_alphas
        else:
            if weights.source_format != "modelopt_nvfp4":
                raise ValueError(
                    "--quant-mode w4a8_nvfp4 requires source_format='modelopt_nvfp4'"
                )
            w13_scales = torch.stack(
                [
                    unswizzle_block_scale(
                        weights.w13_blockscale_swizzled[eid].view(torch.uint8),
                        2 * spec.I_tp,
                        spec.hidden_size // 16,
                    )
                    for eid in range(spec.num_experts)
                ]
            )
            w2_scales = torch.stack(
                [
                    unswizzle_block_scale(
                        weights.w2_blockscale_swizzled[eid].view(torch.uint8),
                        spec.hidden_size,
                        spec.I_tp // 16,
                    )
                    for eid in range(spec.num_experts)
                ]
            )
            w13_mx, w13_residual = decompose_nvfp4_scales_to_mx_residual(
                w13_scales
            )
            w2_mx, w2_residual = decompose_nvfp4_scales_to_mx_residual(
                w2_scales
            )
            alpha1 = params.g1_alphas * params.a1_gscale
            alpha2 = params.g2_alphas * params.a2_gscale
        return moe_reference_w4a8_mx(
            x,
            weights.w13_weight,
            w13_mx,
            w13_residual,
            alpha1,
            weights.w2_weight,
            w2_mx,
            w2_residual,
            alpha2,
            topk_ids,
            topk_weights,
            spec.num_experts,
            spec.hidden_size,
            spec.I_tp,
            activation=activation,
            w13_layout=weights.w13_layout,
            **activation_params.kwargs(),
        )
    if oracle_mode == "w4a16":
        raise ValueError("--oracle-mode w4a16 requires --quant-mode w4a16")
    oracle_fn = moe_reference_nvfp4 if oracle_mode == "nvfp4" else moe_reference_f32
    oracle_kwargs = (
        {"quant_scale_math": quant_scale_math} if oracle_mode == "nvfp4" else {}
    )
    return oracle_fn(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        params.g1_alphas,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        params.g2_alphas,
        params.a1_gscale,
        params.a2_gscale,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
        activation=activation,
        **oracle_kwargs,
        **activation_params.kwargs(),
    )


ORACLE_TOLERANCES = {
    "silu": {
        # Default fast_math keeps the fast FP4 dot path.  It is intentionally
        # not bit-exact against the f32-accum oracle, but should remain well
        # within the numerical tolerance of the MiniMax-M2.7 micro profile.
        "max_abs": 0.05,
        "rmse": 0.0075,
        "mean_abs": 0.005,
        "cos_min": 0.9999,
    },
    # relu2 outputs are ~1000x larger in magnitude than silu's, and the
    # activation's squaring step quadratically amplifies per-element noise.
    # Absolute thresholds don't transfer; cos is the correctness signal.
    "relu2": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9915,
    },
    SWIGLUOAI_UNINTERLEAVE: {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9975,
    },
}

W4A16_ORACLE_TOLERANCES = {
    "silu": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9975,
    },
    "relu2": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9900,
    },
    SWIGLUOAI_UNINTERLEAVE: {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9975,
    },
}


W4A8_ORACLE_TOLERANCES = {
    "silu": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9980,
    },
    "relu2": {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9915,
    },
    SWIGLUOAI_UNINTERLEAVE: {
        "max_abs": None,
        "rmse": None,
        "mean_abs": None,
        "cos_min": 0.9975,
    },
}


def format_oracle_metrics(name: str, metrics: OracleMetrics) -> str:
    return (
        f"{name}: max_abs={metrics.max_abs:.5f} "
        f"rmse={metrics.rmse:.5f} "
        f"mean_abs={metrics.mean_abs:.5f} "
        f"cos={metrics.cos:.6f}"
    )


def check_oracle_metrics(
    label: str,
    metrics: OracleMetrics,
    batch_size: int,
    *,
    activation: str = "silu",
    oracle_mode: str = "nvfp4",
    min_cosine: float | None = None,
) -> list[str]:
    failures = []
    metric_values = {
        "max_abs": metrics.max_abs,
        "rmse": metrics.rmse,
        "mean_abs": metrics.mean_abs,
        "cos": metrics.cos,
    }
    nonfinite = [name for name, value in metric_values.items() if not math.isfinite(value)]
    if nonfinite:
        failures.append(
            f"  bs={batch_size} {label}: non-finite metrics "
            f"{', '.join(nonfinite)}"
        )
        return failures
    if oracle_mode in {"w4a16"}:
        tol = W4A16_ORACLE_TOLERANCES[activation]
    elif oracle_mode in {"w4a8_mx", "w4a8_nvfp4"}:
        tol = W4A8_ORACLE_TOLERANCES[activation]
    else:
        tol = ORACLE_TOLERANCES[activation]
    if tol["max_abs"] is not None and metrics.max_abs > tol["max_abs"]:
        failures.append(f"  bs={batch_size} {label}: max_abs={metrics.max_abs:.5f} > {tol['max_abs']}")
    if tol["rmse"] is not None and metrics.rmse > tol["rmse"]:
        failures.append(f"  bs={batch_size} {label}: rmse={metrics.rmse:.5f} > {tol['rmse']}")
    if tol["mean_abs"] is not None and metrics.mean_abs > tol["mean_abs"]:
        failures.append(f"  bs={batch_size} {label}: mean_abs={metrics.mean_abs:.5f} > {tol['mean_abs']}")
    cos_min = tol["cos_min"] if min_cosine is None else float(min_cosine)
    if metrics.cos < cos_min:
        failures.append(
            f"  bs={batch_size} {label}: cos={metrics.cos:.6f} < {cos_min}"
        )
    return failures


def _clear_b12x_caches() -> None:
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches

    clear_tp_moe_caches()
    try:
        from b12x.moe._shared.kernels.w4a16.kernel import clear_w4a16_kernel_cache
    except ImportError:
        return
    clear_w4a16_kernel_cache()




GRAPH_REPLAY_TOLERANCES = {
    "max_abs": 5e-4,
    "rmse": 1e-4,
    "mean_abs": 1e-4,
    "cos_min": 0.9999,
}


def bench_repeated(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    l2_flush: Callable[[], None] | None = None,
) -> TimingStats:
    return summarize_timing_runs(
        [bench_events(fn, warmup=warmup, iters=iters, l2_flush=l2_flush) for _ in range(repeats)]
    )


def compare_graph_replay_outputs(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> OracleMetrics:
    metrics = compare_to_reference(actual, reference)
    actual_norm = actual.float().norm().item()
    reference_norm = reference.float().norm().item()
    if max(actual_norm, reference_norm) <= 1e-8:
        return OracleMetrics(
            max_abs=metrics.max_abs,
            rmse=metrics.rmse,
            mean_abs=metrics.mean_abs,
            cos=1.0,
        )
    return metrics


def prepare_moe_execution(
    *,
    session: PreparationSession,
    name: str,
    experts: object,
    top_k: int,
    inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    outputs: dict[int, torch.Tensor],
    route_mode: str = "auto",
    override: fused_moe.MoeDecodeConfig | None = None,
) -> object:
    """Prepare each exact benchmark M before binding its real caller buffers."""
    capacity = fused_moe.ExecutionCapacity(
        max_tokens=max(inputs),
        top_k=top_k,
        warmup_token_counts=tuple(inputs),
        route_num_experts=0,
    )
    if route_mode != "auto":
        if override is not None:
            raise ValueError("route_mode and explicit MoE config cannot both be set")
        override = fused_moe.MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode=route_mode,
        )
    declaration = fused_moe.plan_execution(
        experts=experts, capacity=capacity, override=override
    )
    calls = {
        m: prepared_call(
            output=outputs[m],
            bind=lambda state, scratch, m=m: state.bind(
                scratch=scratch,
                a=inputs[m][0],
                experts=experts,
                topk_ids=inputs[m][1],
                topk_weights=inputs[m][2],
                output=outputs[m],
                input_scales_static=True,
            ),
        )
        for m in getattr(declaration, "token_counts", (capacity.max_tokens,))
    }
    def race_call(state, *, tokens):
        call = calls[tokens](state)
        source = inputs[tokens][0].clone()
        return replace(
            call, produce=lambda: inputs[tokens][0].copy_(source),
            owners=(*call.owners, source),
        )

    benchmark_calls = {
        m: (lambda state, m=m: race_call(state, tokens=m)) for m in calls
    }
    session.prepare((
        request_for_capacity(
            declaration,
            name=name,
            calls=calls,
            benchmark_calls=benchmark_calls,
        ),
    ))
    execution = declaration
    selection = execution.selection
    if selection is None:
        variants = execution.variants
        print(
            f"  preparation {name}: "
            + ", ".join(
                f"M={m} {variant.selection.source}"
                for m, variant in variants.items()
                if variant.selection is not None
            )
        )
    else:
        print(f"  preparation {name}: {selection.source} {selection.config}")
    return execution


def bind_prepared_moe(
    execution: object,
    *,
    a: torch.Tensor,
    experts: object,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
):
    """Bind a ready exact-M execution; no raw scratch plan or JIT route exists."""
    exact_execution = getattr(execution, "variants", {}).get(int(a.shape[0]), execution)
    return fused_moe.bind(
        exact_execution,
        scratch=scratch_for(exact_execution),
        a=a,
        experts=experts,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        input_scales_static=True,
    )


def run_moe_layer_chain(
    executions: Sequence[object],
    experts_stack: Sequence[object],
    x: torch.Tensor,
    topk_ids_per_layer: Sequence[torch.Tensor],
    topk_weights_per_layer: Sequence[torch.Tensor],
    *,
    output_buffers: Sequence[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    if not (
        len(executions) == len(experts_stack) == len(topk_ids_per_layer)
        == len(topk_weights_per_layer)
    ):
        raise ValueError("layer-chain inputs must all have the same length")
    if output_buffers is not None and len(output_buffers) != len(experts_stack):
        raise ValueError("output_buffers must match the number of layers")

    layer_outputs: list[torch.Tensor] = []
    current = x
    for layer_idx, (execution, experts, topk_ids, topk_weights) in enumerate(
        zip(executions, experts_stack, topk_ids_per_layer, topk_weights_per_layer, strict=True)
    ):
        output = None if output_buffers is None else output_buffers[layer_idx]
        if output is None:
            output = torch.empty_like(current)
        binding = bind_prepared_moe(
            execution,
            a=current,
            experts=experts,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
        )
        current = fused_moe.run(binding=binding)
        layer_outputs.append(current)
    return layer_outputs


def capture_moe_layer_chain(
    executions: Sequence[object],
    experts_stack: Sequence[object],
    x: torch.Tensor,
    topk_ids_per_layer: Sequence[torch.Tensor],
    topk_weights_per_layer: Sequence[torch.Tensor],
    *,
    output_buffers: Sequence[torch.Tensor],
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_moe_layer_chain(
            executions,
            experts_stack,
            x,
            topk_ids_per_layer,
            topk_weights_per_layer,
            output_buffers=output_buffers,
        )
    return graph


def _check_graph_replay_metrics(
    label: str,
    metrics: OracleMetrics,
) -> list[str]:
    failures = []
    tol = GRAPH_REPLAY_TOLERANCES
    if metrics.max_abs > tol["max_abs"]:
        failures.append(f"{label}: max_abs={metrics.max_abs:.6f} > {tol['max_abs']}")
    if metrics.rmse > tol["rmse"]:
        failures.append(f"{label}: rmse={metrics.rmse:.6f} > {tol['rmse']}")
    if metrics.mean_abs > tol["mean_abs"]:
        failures.append(f"{label}: mean_abs={metrics.mean_abs:.6f} > {tol['mean_abs']}")
    if metrics.cos < tol["cos_min"]:
        failures.append(f"{label}: cos={metrics.cos:.6f} < {tol['cos_min']}")
    return failures


def bench_multilayer_graph_mode(
    args,
    model_path: pathlib.Path,
    profile: ModelProfile,
    spec: ModelSpec,
    batch_sizes: Sequence[int],
    device: torch.device,
) -> None:
    graph_num_layers = args.graph_num_layers
    activation_params = args.activation_params
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    if graph_num_layers < 2:
        raise ValueError("--graph-num-layers must be at least 2 in multi-layer graph mode")

    layer_start = args.graph_layer_start
    if profile.shape is None:
        cfg = _load_config(model_path)
        total_layers = cfg["num_hidden_layers"]
    else:
        total_layers = layer_start + graph_num_layers
    if layer_start < 0 or layer_start + graph_num_layers > total_layers:
        raise ValueError(
            f"requested layers [{layer_start}, {layer_start + graph_num_layers}) exceed model depth {total_layers}"
        )

    if args.validate != "none":
        print(
            "Note: multi-layer graph mode skips numerical oracle checks and validates graph replay "
            "against an eager layer chain."
        )
    print("Multi-layer graph mode")
    print("Backend: b12x")
    print(f"Quant mode: {args.quant_mode}")
    print(f"Layers: {layer_start}..{layer_start + graph_num_layers - 1}")
    print("Patterns: disjoint, overlap, random")
    print()

    _clear_b12x_caches()
    weights_stack = load_expert_weight_stack(
        model_path,
        spec,
        layer_start=layer_start,
        num_layers=graph_num_layers,
        activation=args.activation,
        checkpoint_family=profile.checkpoint_family,

    )
    experts_stack: list[object] = []
    for weights in weights_stack:
        params = get_quant_mode_params(
            weights,
            args.scale_contract,
            args.quant_mode,
        )
        experts, _ = prepare_b12x_benchmark_weights(
            weights,
            params,
            quant_mode=args.quant_mode,
            activation=args.activation,
            w4a16_native=args.w4a16_native,
            activation_params=activation_params,
        )
        experts_stack.append(experts)

    scenario_specs = [
        ("disjoint", "disjoint", 1100),
        ("overlap", "overlap", 2200),
        ("random-a", "random", 3300),
        ("random-b", "random", 4400),
    ]
    validation_failures: list[str] = []

    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(
            f"  batch_size={batch_size}  "
            f"(layers={graph_num_layers}, tokens*top_k={batch_size * spec.top_k})"
        )
        print(f"{'=' * 70}")

        x_buf = make_input_activations(spec, batch_size, 10_000 + batch_size, device)
        initial_case = make_multilayer_routing_case(
            spec,
            batch_size,
            graph_num_layers,
            device,
            pattern="disjoint",
            seed=20_000 + batch_size,
        )
        topk_ids_bufs = [topk_ids.clone() for topk_ids, _ in initial_case]
        topk_weights_bufs = [topk_weights.clone() for _, topk_weights in initial_case]
        graph_output_bufs = [torch.empty_like(x_buf) for _ in range(graph_num_layers)]
        eager_output_bufs = [torch.empty_like(x_buf) for _ in range(graph_num_layers)]
        session = PreparationSession(device=device)
        layer_executions = [
            prepare_moe_execution(
                session=session,
                name=f"layer-chain-{batch_size}-{layer_idx}",
                experts=experts,
                top_k=spec.top_k,
                inputs={batch_size: (x_buf, topk_ids_bufs[layer_idx], topk_weights_bufs[layer_idx])},
                outputs={batch_size: graph_output_bufs[layer_idx]},
            )
            for layer_idx, experts in enumerate(experts_stack)
        ]

        run_moe_layer_chain(
            layer_executions,
            experts_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            output_buffers=graph_output_bufs,
        )
        torch.cuda.synchronize()
        graph = capture_moe_layer_chain(
            layer_executions,
            experts_stack,
            x_buf,
            topk_ids_bufs,
            topk_weights_bufs,
            output_buffers=graph_output_bufs,
        )

        def eager_chain() -> None:
            run_moe_layer_chain(
                layer_executions,
                experts_stack,
                x_buf,
                topk_ids_bufs,
                topk_weights_bufs,
                output_buffers=eager_output_bufs,
            )

        for scenario_name, pattern, seed in scenario_specs:
            x_case = make_input_activations(
                spec,
                batch_size,
                30_000 + batch_size + seed,
                device,
            )
            routing_case = make_multilayer_routing_case(
                spec,
                batch_size,
                graph_num_layers,
                device,
                pattern=pattern,
                seed=40_000 + batch_size + seed,
            )

            x_buf.copy_(x_case)
            for layer_idx, (topk_ids, topk_weights) in enumerate(routing_case):
                topk_ids_bufs[layer_idx].copy_(topk_ids)
                topk_weights_bufs[layer_idx].copy_(topk_weights)

            graph.replay()
            torch.cuda.synchronize()
            graph_outputs = [buf.clone() for buf in graph_output_bufs]

            eager_chain()
            torch.cuda.synchronize()
            eager_outputs = [buf.clone() for buf in eager_output_bufs]

            final_metrics = compare_graph_replay_outputs(graph_outputs[-1], eager_outputs[-1])
            layer_metrics = [
                compare_graph_replay_outputs(graph_out, eager_out)
                for graph_out, eager_out in zip(graph_outputs, eager_outputs, strict=True)
            ]
            graph_stats = bench_repeated(
                lambda: run_moe_layer_chain(
                    layer_executions, experts_stack, x_buf, topk_ids_bufs,
                    topk_weights_bufs, output_buffers=graph_output_bufs,
                ),
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                l2_flush=l2_flush,
            )
            eager_stats = bench_repeated(
                eager_chain,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                l2_flush=l2_flush,
            )

            ratio_stats = RatioStats([graph_stats.median_ms / eager_stats.median_ms])
            print(
                f"  {scenario_name}: "
                f"graph {fmt_timing_stats(graph_stats)} | "
                f"eager {fmt_timing_stats(eager_stats)} | "
                f"ratio {fmt_ratio_stats(ratio_stats)}"
            )
            print(
                "    final:",
                format_oracle_metrics("graph vs eager", final_metrics),
            )
            for layer_idx, metrics in enumerate(layer_metrics):
                print(
                    f"    layer {layer_idx + layer_start}: "
                    f"max_abs={metrics.max_abs:.6f} rmse={metrics.rmse:.6f} cos={metrics.cos:.6f}"
                )
                validation_failures.extend(
                    _check_graph_replay_metrics(
                        f"bs={batch_size} {scenario_name} layer={layer_idx + layer_start}",
                        metrics,
                    )
                )
            validation_failures.extend(
                _check_graph_replay_metrics(
                    f"bs={batch_size} {scenario_name} final",
                    final_metrics,
                )
            )

        del graph
        torch.cuda.synchronize()
        session.close()

    if validation_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  MULTI-LAYER GRAPH VALIDATION FAILED")
        print(f"{'=' * 70}")
        for failure in validation_failures:
            print(f"  {failure}")
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)


def bench_e2e() -> None:
    from b12x.moe.fused_moe._impl import default_moe_quant_mode
    from b12x.moe.fused_moe.workloads import ROUTING_WORKLOADS

    quant_mode_default = default_moe_quant_mode()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Assigned CUDA ordinal within the existing visibility mask")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--timing-backend", choices=("stream-gated",), default="stream-gated",
        help="CUPTI sums GPU kernel durations; events include graph gaps.",
    )
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument(
        "--autotune", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--moe-config", type=json.loads, default=None,
        help="JSON MoeDecodeConfig override for controlled single-op experiments.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeat the timed measurement this many times per batch size and aggregate the results.",
    )
    parser.add_argument("--batch-size-profile", choices=sorted(BATCH_SIZE_PROFILES), default="micro")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--raw-samples-jsonl", type=pathlib.Path)
    parser.add_argument(
        "--torch-trace-dir", type=pathlib.Path,
        help="Save one untimed graph replay per case for launch/resource comparison.",
    )
    parser.add_argument(
        "--routing-workload", choices=("default", *ROUTING_WORKLOADS), default="default",
        help="shared_N reuses about N%% of token/expert assignments across the batch.",
    )
    parser.add_argument(
        "--routing-repeat-period",
        type=int,
        default=0,
        help=(
            "Repeat the first N token routes across each batch, modeling "
            "concurrent requests with the same speculative-verify trajectory; "
            "0 keeps the generated routing unchanged."
        ),
    )
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="qwen397b")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k for a synthetic shape-only model profile.",
    )
    parser.add_argument("--tp-size", type=int, default=None, help="Override TP size from model profile")
    parser.add_argument("--tp-rank", type=int, default=0, help="Rank slice to benchmark (default: 0)")
    parser.add_argument(
        "--intermediate-size", type=int, default=None,
        help="Override the global expert width of a shape-only profile (before TP).",
    )
    parser.add_argument("--tp-parallel", action="store_true", help="Load all TP rank slices and replay per-rank CUDA graphs in parallel streams")
    parser.add_argument("--model-path", type=pathlib.Path, default=None)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--activation", choices=BENCHMARK_ACTIVATION_CHOICES, default=None)
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=None,
        help="Clamp gated SwiGLU inputs before activation; defaults to the model profile value.",
    )
    parser.add_argument(
        "--swiglu-alpha",
        type=float,
        default=None,
        help="SwiGLU-OAI sigmoid multiplier; defaults to the model profile value.",
    )
    parser.add_argument(
        "--swiglu-beta",
        type=float,
        default=None,
        help="SwiGLU-OAI up-shift term; defaults to the model profile value.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=["nvfp4", "w4a8_mx", "w4a8_nvfp4", "w4a16"],
        default=None,
        help=(
            "Backend math mode. w4a16 keeps activations BF16 and dequantizes "
            "FP4 weights inline; w4a8_mx dynamically quantizes activations to "
            "MXFP8 and consumes E8M0/K32 FP4 weights; w4a8_nvfp4 uses the "
            "same MXFP8 activation path from ModelOpt NVFP4 K16 weights. If "
            "omitted, the model profile default is used."
        ),
    )
    parser.add_argument(
        "--force-mxfp4",
        action="store_true",
        help=(
            "Offline-requantize ModelOpt NVFP4 checkpoint weights to native "
            "MXFP4 E8M0/K32 before preparing --quant-mode w4a8_mx."
        ),
    )
    parser.add_argument("--graph-mode", choices=["single-op", "multi-layer"], default="single-op")
    parser.add_argument("--graph-num-layers", type=int, default=4)
    parser.add_argument("--graph-layer-start", type=int, default=0)
    parser.add_argument("--scale-contract", choices=["shared", "per-expert"], default=None)
    parser.add_argument(
        "--w4a16-native",
        action="store_true",
        help=(
            "Build the native (modelopt) W4A16 representation so small-M shapes "
            "route to the micro decode kernel. For w31 sources (e.g. DeepSeek V4 "
            "Flash) the W13 halves are reordered to w13 at prep time."
        ),
    )
    parser.add_argument(
        "--w4a16-route-policy",
        choices=["auto", "direct", "packed"],
        default="auto",
        help=(
            "Override direct TC-decode versus expert-packed routing for planner "
            "sweeps."
        ),
    )
    parser.add_argument(
        "--w4a16-tile-config",
        type=int,
        nargs=4,
        metavar=("FC1_K", "FC1_N", "FC2_K", "FC2_N"),
        default=None,
        help="Override the fused W4A16 FC1/FC2 tiles for planner sweeps.",
    )
    parser.add_argument("--validate", choices=["none", "oracle"], default=None)
    parser.add_argument(
        "--oracle-quant-scale-math",
        choices=("direct_division", "micro", "dynamic_fast", "dynamic_precise"),
        default="direct_division",
        help="NVFP4 oracle activation-quantization evaluation order.",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=[
            "nvfp4",
            "w4a8_mx",
            "w4a8_nvfp4",
            "w4a16",
            "f32",
        ],
        default=None,
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=None,
        help="Override only the oracle cosine gate; other tolerances are unchanged.",
    )
    parser.add_argument("--include-routing", action="store_true")
    parser.set_defaults(cuda_graph=True)
    parser.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        help="Qualify CUDA graph replay before stream-gated timing (default: enabled).",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable graph qualification and time eager prepared calls.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Measure only the graph-qualified workload path with stream-gated events.",
    )
    parser.add_argument(
        "--profile-once",
        choices=[
            "none",
            "backend",
        ],
        default="none",
    )
    parser.add_argument(
        "--profile-graphs", action="store_true",
        help="Expose one cold-L2 CUDA graph replay per case to CUDA profiling.",
    )
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evict GPU L2 before each measured invocation (default: enabled).",
    )
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="Bytes to touch when evicting L2; 0 uses 2x the reported L2 size.",
    )
    args = parser.parse_args()

    if args.torch_trace_dir is not None and (not args.graph_only or not args.cuda_graph or args.graph_mode != "single-op" or args.tp_parallel):
        parser.error("--torch-trace-dir requires single-op, single-rank --graph-only")
    if args.profile_graphs and (
        not args.graph_only or not args.cuda_graph or args.graph_mode != "single-op"
        or args.profile_once != "none" or not args.flush_l2
    ):
        parser.error("--profile-graphs requires cold-L2 single-op --graph-only execution")
    if (args.moe_config is not None or args.output_json is not None) and (
        args.graph_mode != "single-op" or args.tp_parallel
    ):
        parser.error("--moe-config/--output-json require single-op and one rank")
    config_override = (
        fused_moe.MoeDecodeConfig(**args.moe_config)
        if args.moe_config is not None else None
    )
    model_profile = MODEL_PROFILES[args.model_profile]
    if args.activation is None:
        args.activation = model_profile.default_activation
    args.activation = normalize_moe_activation(args.activation)
    if args.quant_mode is None:
        args.quant_mode = model_profile.default_quant_mode or quant_mode_default
    if args.scale_contract is None:
        args.scale_contract = model_profile.default_scale_contract
    use_w4a16 = args.quant_mode == "w4a16"
    use_w4a8 = args.quant_mode in {"w4a8_mx", "w4a8_nvfp4"}
    swiglu_limit = args.swiglu_limit if args.swiglu_limit is not None else model_profile.default_swiglu_limit
    swiglu_alpha = args.swiglu_alpha if args.swiglu_alpha is not None else model_profile.default_swiglu_alpha
    swiglu_beta = args.swiglu_beta if args.swiglu_beta is not None else model_profile.default_swiglu_beta
    activation_params = ActivationParams(
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    args.activation_params = activation_params
    if args.validate is None:
        args.validate = model_profile.default_validate
    if args.oracle_mode is None:
        args.oracle_mode = (
            "f32"
            if use_w4a16 and args.activation == SWIGLUOAI_UNINTERLEAVE
            else args.quant_mode
        )
    batch_sizes = (
        args.batch_sizes
        if args.batch_sizes is not None
        else BATCH_SIZE_PROFILES[args.batch_size_profile]
    )
    model_path = resolve_model_path(model_profile, args.model_path)
    layer_idx = model_profile.default_layer_idx if args.layer_idx is None else args.layer_idx

    if model_profile.checkpoint_family in {"qwen_iq2_xs", "puzzle3_iq2_xs"}:
        if args.quant_mode != "w4a16" or args.oracle_mode != "w4a16":
            raise ValueError("IQ2_XS requires W4A16 with its block oracle and no FP4 external reference")
        if args.w4a16_native or args.force_mxfp4:
            raise ValueError("IQ2_XS uses compact descriptor preparation")

    if args.force_mxfp4 and args.quant_mode != "w4a8_mx":
        raise ValueError("--force-mxfp4 requires --quant-mode w4a8_mx")
    if args.w4a16_route_policy != "auto" and not use_w4a16:
        raise ValueError("--w4a16-route-policy requires --quant-mode w4a16")
    if args.w4a16_tile_config is not None and not use_w4a16:
        raise ValueError("--w4a16-tile-config requires --quant-mode w4a16")
    if (
        model_profile.checkpoint_family == "deepseek_v4_flash"
        and args.quant_mode not in {"w4a8_mx", "w4a16"}
    ):
        raise ValueError(
            "DeepSeek V4 Flash FP4 checkpoint profile requires "
            "--quant-mode w4a16 or w4a8_mx"
        )
    if use_w4a16 and args.graph_mode != "single-op":
        raise ValueError("--quant-mode w4a16 currently supports --graph-mode single-op")
    if use_w4a16 and args.tp_parallel:
        raise ValueError("--quant-mode w4a16 currently does not support --tp-parallel")
    if args.graph_only and not args.cuda_graph:
        raise ValueError("--graph-only requires --cuda-graph")
    if args.routing_workload != "default" and (
        args.include_routing or args.routing_repeat_period or args.tp_parallel
        or args.graph_mode != "single-op"
    ):
        raise ValueError(
            "--routing-workload requires a pre-routed single-op, single-rank "
            "benchmark without --routing-repeat-period"
        )
    if args.routing_repeat_period < 0:
        raise ValueError("--routing-repeat-period must be non-negative")
    if args.routing_repeat_period and args.graph_mode != "single-op":
        raise ValueError("--routing-repeat-period requires --graph-mode single-op")
    if args.routing_repeat_period and any(
        args.routing_repeat_period > batch_size for batch_size in batch_sizes
    ):
        raise ValueError(
            "--routing-repeat-period cannot exceed any requested batch size"
        )
    if args.device is not None:
        torch.cuda.set_device(args.device)
    require_sm120()
    torch.empty(1, device="cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    if (
        args.w4a16_route_policy != "auto"
        or args.w4a16_tile_config is not None
    ):
        from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel

        if args.w4a16_route_policy != "auto":
            force_direct = args.w4a16_route_policy == "direct"

            def forced_tc_decode_preferred(**_: int) -> bool:
                return force_direct

            w4a16_kernel._w4a16_tc_decode_preferred = forced_tc_decode_preferred

        if args.w4a16_tile_config is not None:
            real_compile = w4a16_kernel.compile_w4a16_fused_moe
            forced_tiles = tuple(args.w4a16_tile_config)

            def compile_with_forced_tiles(*compile_args, **compile_kwargs):
                compile_kwargs["force_tile_config"] = forced_tiles
                return real_compile(*compile_args, **compile_kwargs)

            w4a16_kernel.compile_w4a16_fused_moe = compile_with_forced_tiles
    l2_flush = make_l2_flush_fn(enabled=args.flush_l2, bytes_hint=args.l2_flush_bytes)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0

    spec = build_model_spec(
        model_path, model_profile, tp_size_override=args.tp_size, tp_rank=args.tp_rank, layer_idx=layer_idx,
    )
    if args.intermediate_size is not None:
        if model_profile.shape is None:
            parser.error("--intermediate-size requires a shape-only profile")
        if args.intermediate_size <= 0 or args.intermediate_size % spec.tp_size:
            parser.error("--intermediate-size must be positive and divisible by TP")
        spec = replace(spec, intermediate_size=args.intermediate_size)
    if args.top_k is not None:
        if model_profile.shape is None:
            raise ValueError("--top-k is limited to synthetic shape-only profiles")
        if args.top_k < 1 or args.top_k > spec.num_experts:
            raise ValueError(
                f"--top-k must be in [1, {spec.num_experts}], got {args.top_k}"
            )
        spec = replace(spec, top_k=args.top_k)

    benchmark_scope = "Routing + MoE kernel" if args.include_routing else "Pre-routed MoE kernel only"
    print(f"MoE benchmark ({benchmark_scope})")
    print(
        f"{model_profile.label}  TP={spec.tp_size}, K={spec.hidden_size}, I_tp={spec.I_tp}, "
        f"E={spec.num_experts}, top_k={spec.top_k}"
    )
    if model_profile.checkpoint_family == "deepseek_v41_flash":
        print(
            f"Intermediate TP shard: rank {spec.tp_rank}, all {spec.num_experts} experts, "
            f"logical I_tp={spec.logical_I_tp}, physical I_tp={spec.I_tp}. "
            "Excludes the TP all-reduce and shared expert."
        )
    print(f"Model path: {model_path}")
    if model_profile.shape is not None:
        print("Weights: synthetic shape-only")
    print(f"Layer: {layer_idx}")
    print(f"Activation: {args.activation}")
    print(f"Quant mode: {args.quant_mode}")
    routing_source = (
        model_profile.default_routing
        if args.routing_workload == "default" else "synthetic"
    )
    print(f"Routing source: {routing_source}")
    print(f"Routing workload: {args.routing_workload}")
    if args.routing_repeat_period:
        print(f"Routing repeat period: {args.routing_repeat_period} tokens")
    print(f"Batch-size profile: {args.batch_size_profile} -> {batch_sizes}")
    backend_label = "b12x"
    print(f"Backend: {backend_label}")
    print(f"Scale contract: {args.scale_contract}")
    print(f"Validation: {args.validate}")
    print(f"Fast math: {'on' if args.fast_math else 'off'}")
    if use_w4a16:
        print("W4A16 kernel: fused W4A16 FC1+FC2")
        print(f"W4A16 route policy: {args.w4a16_route_policy}")
        print(f"W4A16 tile config: {args.w4a16_tile_config or 'auto'}")
    if swiglu_limit is not None:
        print(f"SwiGLU limit: {swiglu_limit:g}")
    if swiglu_alpha is not None:
        print(f"SwiGLU alpha: {swiglu_alpha:g}")
    if swiglu_beta is not None:
        print(f"SwiGLU beta: {swiglu_beta:g}")
    if args.flush_l2:
        print(f"L2 flush: on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)")
    else:
        print("L2 flush: off")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Graph only: {'yes' if args.graph_only else 'no'}")
    print(f"Timing passes per batch size: {args.repeats} x {args.iters} iterations")
    print(
        "Timed region: "
        + (
            "model gate routing + backend launch"
            if args.include_routing and model_profile.default_routing == "model"
            else "top-k + softmax routing + backend launch"
            if args.include_routing
            else "backend launch only"
        )
    )
    if args.validate == "oracle":
        print(f"Oracle mode: {args.oracle_mode}")
    print()

    if args.graph_mode == "multi-layer":
        if args.raw_samples_jsonl is not None:
            raise ValueError("raw sample evidence currently requires single-op graphs")
        bench_multilayer_graph_mode(args, model_path, model_profile, spec, batch_sizes, device)
        return

    weights = load_expert_weights(
        model_path,
        spec,
        layer_idx=layer_idx,
        activation=args.activation,
        checkpoint_family=model_profile.checkpoint_family,

    )
    if args.raw_samples_jsonl is not None:
        from b12x._lib.compiler import b12x_package_fingerprint

        args.raw_samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        payload_hashes = {}
        if weights.source_format in IQ2_CODECS:
            payload_hashes = {
                name: hashlib.sha256(memoryview(tensor.numpy())).hexdigest()
                for name, tensor in (("w13", weights.w13_weight), ("w2", weights.w2_weight))
            }
        with args.raw_samples_jsonl.open("x") as handle:
            handle.write(json.dumps(dict(
                kind="manifest", command=sys.argv, model=str(model_path), layer=layer_idx,
                spec=vars(spec), options=vars(args), torch=torch.__version__,
                source_fingerprint=b12x_package_fingerprint(), payload_sha256=payload_hashes,
                benchmark_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                device=nvidia_smi_gpu_mode_snapshot(),
                metric="cold-L2 graph replay microseconds; lower is better",
            ), default=str) + "\n")
    if args.force_mxfp4:
        source_params = get_quant_mode_params(weights, args.scale_contract, "w4a8_nvfp4")
        weights = force_convert_nvfp4_weights_to_mxfp4(
            weights,
            source_params,
            activation=args.activation,
        )
    print(f"Source format: {weights.source_format}")
    params = get_quant_mode_params(weights, args.scale_contract, args.quant_mode)
    weight_plan = plan_b12x_benchmark_weights(
        weights,
        quant_mode=args.quant_mode,
        activation=args.activation,
        activation_params=activation_params,
        w4a16_native=args.w4a16_native,
    )
    precomputed_oracles: dict[int, torch.Tensor] = {}
    if args.validate == "oracle":
        print(
            "  Precomputing oracle outputs before destructive weight "
            "preparation...",
            end="",
            flush=True,
        )
        for batch_size in batch_sizes:
            oracle_x, oracle_ids, oracle_topk, _ = make_benchmark_case(
                model_profile,
                weights,
                spec,
                batch_size,
                42 + batch_size,
                device,
                routing_workload=args.routing_workload,
            )
            oracle_ids, oracle_topk = repeat_routing_pattern(
                oracle_ids,
                oracle_topk,
                args.routing_repeat_period,
            )
            oracle_output = make_oracle_reference(
                args.oracle_mode,
                args.quant_mode,
                oracle_x,
                weights,
                params,
                oracle_ids,
                oracle_topk,
                activation=args.activation,
                activation_params=activation_params,
                quant_scale_math=args.oracle_quant_scale_math,
            )
            # Outputs are the only state retained across the ownership
            # transfer.  Keep them off-device; never retain a second model.
            precomputed_oracles[batch_size] = oracle_output.detach().cpu()
            del oracle_x, oracle_ids, oracle_topk, oracle_output
        torch.cuda.synchronize()
        # representation.  It must not survive into B12X preparation/runtime.
        weights.oracle_w13_weight = None
        weights.oracle_w13_scale = None
        weights.oracle_w2_weight = None
        weights.oracle_w2_scale = None
        torch.cuda.empty_cache()
        print(" done.")
    experts, params = prepare_b12x_benchmark_weights(
        weights,
        params,
        quant_mode=args.quant_mode,
        activation=args.activation,
        w4a16_native=args.w4a16_native,
        activation_params=activation_params,
        plan=weight_plan,
    )
    _clear_b12x_caches()
    b12x_session = PreparationSession(device=device, autotune=args.autotune)
    print("  Preparing b12x declarations...", end="", flush=True)
    x_warm, topk_ids_w, topk_weights_w = make_profile_routed_inputs(
        model_profile, weights, spec, 1, 42, device,
    )
    warmup_output = torch.empty_like(x_warm)
    warmup_execution = prepare_moe_execution(
        session=b12x_session,
        name="benchmark-moe-warmup",
        experts=experts,
        top_k=spec.top_k,
        inputs={1: (x_warm, topk_ids_w, topk_weights_w)},
        outputs={1: warmup_output},
        route_mode=args.w4a16_route_policy,
        override=config_override,
    )
    warmup_binding = bind_prepared_moe(
        warmup_execution,
        a=x_warm,
        experts=experts,
        topk_ids=topk_ids_w,
        topk_weights=topk_weights_w,
        output=warmup_output,
    )
    fused_moe.run(binding=warmup_binding)
    torch.cuda.synchronize()
    print(" done.")

    # ---- TP-parallel setup ----
    tp_parallel_ranks: list[tuple[ModelSpec, object, object]] = []
    if args.tp_parallel and spec.tp_size > 1:
        print("  Loading TP-parallel ranks...", end="", flush=True)
        for r in range(spec.tp_size):
            rspec = build_model_spec(model_path, model_profile, tp_size_override=args.tp_size, tp_rank=r, layer_idx=layer_idx)
            rw = load_expert_weights(
                model_path, rspec, layer_idx=layer_idx,
                activation=args.activation, checkpoint_family=model_profile.checkpoint_family,
            )
            rp = get_quant_mode_params(rw, args.scale_contract, args.quant_mode)
            rexperts, _ = prepare_b12x_benchmark_weights(
                rw,
                rp,
                quant_mode=args.quant_mode,
                activation=args.activation,
                activation_params=activation_params,
            )
            x_r = torch.randn(1, rspec.hidden_size, dtype=torch.bfloat16, device=device)
            rk_warm = torch.randn(1, rspec.num_experts, dtype=torch.float32, device=device)
            rk_logits, rk_ids = torch.topk(rk_warm, rspec.top_k, dim=-1)
            rk_weights = torch.softmax(rk_logits, dim=-1)
            rk_ids, rk_weights = normalize_kernel_routing(rk_ids, rk_weights)
            output_r = torch.empty_like(x_r)
            execution_r = prepare_moe_execution(
                session=b12x_session,
                name=f"benchmark-moe-tp-{r}",
                experts=rexperts,
                top_k=rspec.top_k,
                inputs={1: (x_r, rk_ids, rk_weights)},
                outputs={1: output_r},
            )
            binding_r = bind_prepared_moe(
                execution_r, a=x_r, experts=rexperts, topk_ids=rk_ids,
                topk_weights=rk_weights, output=output_r,
            )
            fused_moe.run(binding=binding_r)
            tp_parallel_ranks.append((rspec, rexperts, execution_r))
        torch.cuda.synchronize()
        print(f" {spec.tp_size} ranks done.")

    batch_results: dict[int, BatchResult] = {}
    accuracy_failures: list[str] = []
    evidence: list[dict] = []
    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"  batch_size={batch_size}  (tokens*top_k = {batch_size * spec.top_k} expert calls)")
        print(f"{'=' * 70}")

        x, topk_ids, topk_weights, routing_logits = make_benchmark_case(
            model_profile,
            weights,
            spec,
            batch_size,
            42 + batch_size,
            device,
            routing_workload=args.routing_workload,
        )
        topk_ids, topk_weights = repeat_routing_pattern(
            topk_ids,
            topk_weights,
            args.routing_repeat_period,
        )
        local_routes = topk_ids >= 0
        active_experts = int(torch.unique(topk_ids[local_routes]).numel())
        packed_expert_bytes = (
            sum(t[0].numel() * t.element_size() for t in (weights.w13_weight, weights.w2_weight))
            if weights.source_format in IQ2_CODECS else None
        )
        active_density = int(local_routes.sum().item()) / max(active_experts, 1)
        print(
            f"  routing: {active_experts} active experts, "
            f"{active_density:.1f} routed rows/active expert"
        )

        def compute_timed_routing() -> tuple[torch.Tensor, torch.Tensor]:
            if model_profile.default_routing == "model":
                timed_topk_ids, timed_topk_weights = compute_model_gate_routing(
                    weights, x, seed=43 + batch_size
                )
            else:
                assert routing_logits is not None
                timed_topk_logits, timed_topk_ids = torch.topk(
                    routing_logits, spec.top_k, dim=-1
                )
                timed_topk_weights = torch.softmax(timed_topk_logits, dim=-1)
                timed_topk_ids, timed_topk_weights = normalize_kernel_routing(
                    timed_topk_ids, timed_topk_weights
                )
            return repeat_routing_pattern(
                timed_topk_ids, timed_topk_weights, args.routing_repeat_period,
            )

        backend_output = torch.empty_like(x)
        backend_execution = prepare_moe_execution(
            session=b12x_session,
            name=f"benchmark-moe-m{batch_size}",
            experts=experts,
            top_k=spec.top_k,
            inputs={batch_size: (x, topk_ids, topk_weights)},
            outputs={batch_size: backend_output},
            route_mode=args.w4a16_route_policy,
            override=config_override,
        )
        backend_binding = bind_prepared_moe(
            backend_execution,
            a=x,
            experts=experts,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=backend_output,
        )

        def make_backend_e2e() -> Callable[[], torch.Tensor]:
            def impl_launch(
                topk_ids_local: torch.Tensor, topk_weights_local: torch.Tensor,
            ) -> torch.Tensor:
                if topk_ids_local is not backend_binding.topk_ids:
                    backend_binding.topk_ids.copy_(topk_ids_local)
                if topk_weights_local is not backend_binding.topk_weights:
                    backend_binding.topk_weights.copy_(topk_weights_local)
                return fused_moe.run(binding=backend_binding)

            def impl_e2e() -> torch.Tensor:
                if args.include_routing:
                    timed_topk_ids, timed_topk_weights = compute_timed_routing()
                    return impl_launch(timed_topk_ids, timed_topk_weights)
                return impl_launch(topk_ids, topk_weights)

            return impl_e2e

        backend_e2e = make_backend_e2e()

        oracle_ref = None
        if args.validate == "oracle":
            oracle_ref = precomputed_oracles[batch_size].to(device=device)
            print(
                "  oracle:".ljust(28),
                f"norm={oracle_ref.float().norm().item():.5f}",
                f"max={oracle_ref.float().abs().max().item():.5f}",
            )


        backend_out = backend_e2e().clone()
        torch.cuda.synchronize()
        if not torch.isfinite(backend_out).all() or (
            active_experts and not torch.count_nonzero(backend_out)
        ):
            raise RuntimeError(f"M={batch_size}: nonfinite or all-zero MoE output")
        backend_metrics = None


        if oracle_ref is not None:
            backend_metrics = compare_to_reference(backend_out, oracle_ref)
            print(f"  {format_oracle_metrics(f'{backend_label} vs oracle', backend_metrics)}")
            accuracy_failures.extend(
                check_oracle_metrics(
                    f"{backend_label} vs oracle", backend_metrics, batch_size,
                    activation=args.activation,
                    oracle_mode=args.oracle_mode,
                    min_cosine=args.min_cosine,
                )
            )
            if weights.source_format in IQ2_CODECS:
                reference_norm = oracle_ref.float().norm().item()
                relative_l2 = (backend_out.float() - oracle_ref.float()).norm().item() / max(reference_norm, 1e-30)
                if not torch.isfinite(backend_out).all() or not torch.count_nonzero(backend_out) or reference_norm == 0 or backend_metrics.cos < 0.999 or relative_l2 > 0.01:
                    accuracy_failures.append(f"  bs={batch_size} IQ2_XS: finite/nonzero/cosine/relative-L2 gate failed ({relative_l2=:.6f})")

        if accuracy_failures:
            raise RuntimeError("Oracle failed before timing: " + "; ".join(accuracy_failures))
        exact = getattr(backend_execution, "variants", {}).get(batch_size, backend_execution)
        record = {
            "tokens": batch_size,
            "active_experts": active_experts,
            "expert_row_counts": sorted(
                topk_ids[local_routes].unique(return_counts=True)[1].cpu().tolist()
            ),
            "logical_weight_bytes": active_experts * logical_expert_bytes(
                spec, activation=args.activation, quant_mode=args.quant_mode,
                source_format=weights.source_format,
            ),
            "selection": {
                "source": exact.selection.source,
                "config": asdict(exact.selection.config),
                "query": exact.selection.query.to_dict(),
                "invocation": exact.invocation.to_dict(),
            },
            "oracle": asdict(backend_metrics) if backend_metrics is not None else None,
            "graph_samples_ms": {},
        }
        evidence.append(record)

        if args.profile_once != "none":
            profile_fn = backend_e2e
            profile_name = backend_label
            print(f"  profiling once: {profile_name}")
            torch.cuda.synchronize()
            cudart = torch.cuda.cudart()
            cudart.cudaProfilerStart()
            profile_fn()
            torch.cuda.synchronize()
            cudart.cudaProfilerStop()
            print("  profiler range complete")
            return

        backend_stats = None
        if not args.graph_only:
            backend_runs_ms: list[list[float]] = []
            for _ in range(args.repeats):
                backend_run = bench_events(
                    backend_e2e,
                    warmup=args.warmup,
                    iters=args.iters,
                    l2_flush=l2_flush,
                )
                backend_runs_ms.append(backend_run)

            backend_stats = summarize_timing_runs(backend_runs_ms)


            print(f"  {backend_label} (no graph):".ljust(28), end="", flush=True)
            print(f" {fmt_timing_stats(backend_stats)}")

        if args.cuda_graph:
            graph_stats_by_name: dict[str, TimingStats] = {}
            graph_launches = [(backend_label, backend_e2e)]

            for name, fn in graph_launches:
                print(f"  {name} (CUDA graph):".ljust(28), end="", flush=True)
                try:
                    # Warm eager launch state before capture so compile/cache work
                    # does not leak into the replay measurement.
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()

                    def replay(g: torch.cuda.CUDAGraph = graph) -> None:
                        g.replay()

                    if name == backend_label and oracle_ref is not None:
                        backend_output.fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize()
                        if not torch.isfinite(backend_output).all() or (
                            active_experts and not torch.count_nonzero(backend_output)
                        ):
                            raise RuntimeError("Graph replay produced invalid MoE output")
                        graph_metrics = compare_to_reference(backend_output, oracle_ref)
                        failures = check_oracle_metrics(
                            f"{backend_label} graph vs oracle", graph_metrics, batch_size,
                            activation=args.activation, oracle_mode=args.oracle_mode,
                            min_cosine=args.min_cosine,
                        )
                        if failures:
                            raise RuntimeError("Graph oracle failed: " + "; ".join(failures))
                        record["graph_oracle"] = asdict(graph_metrics)

                    # Time the prepared invocation after graph correctness qualification.
                    device_before = nvidia_smi_gpu_mode_snapshot() if args.raw_samples_jsonl is not None else None
                    graph_runs = [
                        bench_events(fn, warmup=args.warmup, iters=args.iters,
                                     l2_flush=l2_flush)
                        for _ in range(args.repeats)
                    ]
                    record["graph_samples_ms"][name] = graph_runs
                    stats = summarize_timing_runs(graph_runs)
                    if args.raw_samples_jsonl is not None:
                        with args.raw_samples_jsonl.open("a") as handle:
                            handle.write(json.dumps(dict(
                                kind="case", rows=batch_size, backend=name,
                                active_experts=active_experts,
                                topk_ids=topk_ids.cpu().tolist(),
                                packed_expert_bytes=packed_expert_bytes,
                                unique_packed_weight_bytes=(
                                    active_experts * packed_expert_bytes
                                    if packed_expert_bytes is not None else None
                                ),
                                samples_us=[[sample * 1000 for sample in run] for run in graph_runs],
                                device_before=device_before, device_after=nvidia_smi_gpu_mode_snapshot(),
                                validation=args.validate, accuracy_failures=list(accuracy_failures),
                            )) + "\n")
                    graph_stats_by_name[name] = stats
                    print(f" {fmt_timing_stats(stats)}")
                    if args.torch_trace_dir is not None:
                        args.torch_trace_dir.mkdir(parents=True, exist_ok=True)
                        if l2_flush is not None:
                            l2_flush()
                        torch.cuda.synchronize()
                        with torch.profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]) as profiler:
                            graph.replay()
                            torch.cuda.synchronize()
                        trace_path = args.torch_trace_dir / f"m{batch_size}-{name}.trace.json"
                        profiler.export_chrome_trace(str(trace_path))
                    if args.profile_graphs:
                        if accuracy_failures:
                            raise RuntimeError("cannot profile a failed oracle check")
                        l2_flush()
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()
                        try:
                            graph.replay()
                            torch.cuda.synchronize()
                        finally:
                            torch.cuda.profiler.stop()
                    if name == backend_label:
                        bandwidth = record["logical_weight_bytes"] / stats.median_us / 1000
                        record["useful_weight_GBps"] = bandwidth
                        record["median_us"] = stats.median_us
                        print(f"    useful weight bandwidth: {bandwidth:.1f} GB/s")
                except Exception as exc:
                    print(f" FAILED ({type(exc).__name__}: {exc})")
                    raise

            batch_results[batch_size] = BatchResult(
                backend_stats=graph_stats_by_name[backend_label],
            )
        elif backend_stats is not None:
            batch_results[batch_size] = BatchResult(backend_stats=backend_stats)

        # ---- TP-parallel graph replay ----
        if tp_parallel_ranks:
            tp_n = len(tp_parallel_ranks)
            label = f"b12x TP={tp_n} parallel"
            print(f"  {label} (CUDA graph):".ljust(28), end="", flush=True)
            try:
                # Per-rank inputs, outputs, workspaces
                tp_x = [
                    torch.randn(batch_size, rs.hidden_size, dtype=torch.bfloat16, device=device)
                    for rs, _, _ in tp_parallel_ranks
                ]
                tp_routing = [
                    torch.randn(batch_size, rs.num_experts, dtype=torch.float32, device=device)
                    for rs, _, _ in tp_parallel_ranks
                ]
                tp_topk_ids: list[torch.Tensor] = []
                tp_topk_weights: list[torch.Tensor] = []
                for r_routing, (rspec, _, _) in zip(tp_routing, tp_parallel_ranks, strict=True):
                    r_logits, r_ids = torch.topk(r_routing, rspec.top_k, dim=-1)
                    tp_topk_ids.append(r_ids)
                    tp_topk_weights.append(torch.softmax(r_logits, dim=-1))
                tp_outputs = [torch.empty_like(tp_x[r]) for r in range(tp_n)]
                tp_streams = [torch.cuda.Stream() for _ in range(tp_n)]
                tp_bindings = []
                for r, (rspec, rexperts, _) in enumerate(tp_parallel_ranks):
                    execution = prepare_moe_execution(
                        session=b12x_session,
                        name=f"benchmark-moe-tp-{r}-m{batch_size}",
                        experts=rexperts,
                        top_k=rspec.top_k,
                        inputs={batch_size: (tp_x[r], tp_topk_ids[r], tp_topk_weights[r])},
                        outputs={batch_size: tp_outputs[r]},
                    )
                    tp_bindings.append(bind_prepared_moe(
                        execution,
                        a=tp_x[r],
                        experts=rexperts,
                        topk_ids=tp_topk_ids[r],
                        topk_weights=tp_topk_weights[r],
                        output=tp_outputs[r],
                    ))

                def launch_tp_rank(r: int) -> None:
                    fused_moe.run(binding=tp_bindings[r])

                # Warm eager launches
                for r, stream in enumerate(tp_streams):
                    with torch.cuda.stream(stream):
                        launch_tp_rank(r)
                torch.cuda.synchronize()

                # Capture each rank on its own stream. A single graph context only
                # captures work issued to its active capture stream.
                tp_graphs: list[torch.cuda.CUDAGraph] = []
                for r, stream in enumerate(tp_streams):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode="relaxed"):
                        launch_tp_rank(r)
                    tp_graphs.append(graph)
                torch.cuda.synchronize()

                def tp_replay(
                    graphs: list[torch.cuda.CUDAGraph] = tp_graphs,
                    streams: list[torch.cuda.Stream] = tp_streams,
                ) -> None:
                    timing_stream = torch.cuda.current_stream()
                    for stream in streams:
                        stream.wait_stream(timing_stream)
                    for graph, stream in zip(graphs, streams, strict=True):
                        with torch.cuda.stream(stream):
                            graph.replay()
                    for stream in streams:
                        timing_stream.wait_stream(stream)

                tp_replay()
                torch.cuda.synchronize()

                def tp_launch():
                    timing_stream = torch.cuda.current_stream()
                    for stream in tp_streams:
                        stream.wait_stream(timing_stream)
                    for rank, stream in enumerate(tp_streams):
                        with torch.cuda.stream(stream):
                            launch_tp_rank(rank)
                    for stream in tp_streams:
                        timing_stream.wait_stream(stream)

                tp_times = bench_events(
                    tp_launch, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush,
                )
                print(f" {fmt_us(tp_times)}")
            except Exception as exc:
                print(f" FAILED ({type(exc).__name__}: {exc})")

    # Captured graphs hold references into the prepared plans they replay.
    # Drop the final diagnostic graph before closing this benchmark's session.
    if "tp_graphs" in locals():
        del tp_graphs
    if "graph" in locals():
        del graph
    torch.cuda.synchronize()
    b12x_session.close()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({
            "command": sys.argv,
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "model_profile": args.model_profile,
            "geometry": asdict(spec),
            "quant_mode": args.quant_mode,
            "source_format": weights.source_format,
            "weight_storage": "pytorch",
            "scale_contract": args.scale_contract,
            "routing_workload": args.routing_workload,
            "timing_backend": args.timing_backend,
            "cold_l2": args.flush_l2,
            "validation": args.validate,
            "oracle_quant_scale_math": args.oracle_quant_scale_math,
            "cases": evidence,
        }, indent=2, default=str) + "\n")

    for batch_size, result in sorted(batch_results.items()):
        print(f"M={batch_size}: {backend_label} {result.backend_stats.median_us:.1f} us")

    if accuracy_failures:
        print(f"\n\033[1;31m{'=' * 70}")
        print("  ACCURACY CHECK FAILED")
        print(f"{'=' * 70}")
        for f in accuracy_failures:
            print(f)
        print(f"{'=' * 70}\033[0m")
        sys.exit(1)


def main() -> None:
    bench_e2e()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
