#!/usr/bin/env python3
"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Accuracy and performance checks for the MagiAttention FFA integration.

This script is intentionally independent from pytest so it can be used directly
after integrating MagiAttention into FlashInfer:

    python benchmarks/bench_ffa_integration.py --mode accuracy --profile quick
    python benchmarks/bench_ffa_integration.py --mode perf --profile quick
    python benchmarks/bench_ffa_integration.py --mode all --output-json ffa.json

It compares:

* MagiAttention's native ``flex_flash_attn_func``
* ``flashinfer.ffa_kernels`` direct wrappers
* ``flashinfer.single_prefill_with_kv_cache(..., backend="ffa")``
* ``flashinfer.BatchPrefillWithRaggedKVCacheWrapper(..., backend="ffa")``
* FlashInfer ``auto`` backend for comparable dense/ragged cases
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class FFAIntegrationCase:
    name: str
    kind: str
    dtype_name: str
    q_len: int
    k_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool = True
    doc_lens: tuple[int, ...] = ()
    window_size: int = 0
    block_size: int = 0


@dataclass
class CaseData:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    kv_indptr: torch.Tensor | None = None
    flops: float | None = None


@dataclass
class AccuracyResult:
    case: str
    impl: str
    status: str
    max_abs: float | None = None
    max_rel: float | None = None
    message: str = ""


@dataclass
class PerfResult:
    case: str
    impl: str
    status: str
    ms: float | None = None
    ms_p20: float | None = None
    ms_p80: float | None = None
    tflops: float | None = None
    tflops_p20: float | None = None
    tflops_p80: float | None = None
    overhead_vs_magi: float | None = None
    speedup_vs_auto: float | None = None
    message: str = ""


@dataclass(frozen=True)
class FFASparseIntegrationCase:
    name: str
    kind: str
    dtype_name: str
    seqlen: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    sparsity_ratio: float
    q_block_size: int = 0
    k_block_size: int = 0
    var_block_size: int = 0


@dataclass
class SparseCaseData:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor
    plan_num_qo_heads: int
    plan_num_kv_heads: int
    plan_kwargs: dict[str, Any]
    flops: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FlashInfer FFA integration accuracy/performance checks."
    )
    parser.add_argument(
        "--mode",
        choices=("accuracy", "perf", "all"),
        default="all",
        help="Which checks to run.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full", "official-dense", "official-sparse"),
        default="quick",
        help=(
            "Case set size. full adds larger perf cases and more accuracy cases. "
            "official-dense follows MagiAttention's dense single-GPU benchmark masks. "
            "official-sparse follows MagiAttention's block-sparse forward benchmarks."
        ),
    )
    parser.add_argument(
        "--bench-method",
        choices=("cuda-loop", "magi-official"),
        default=None,
        help=(
            "Timing method for performance mode. Defaults to magi-official for "
            "official profiles and cuda-loop otherwise. magi-official uses "
            "magi_attention.benchmarking.do_bench_flops with quantiles=[0.5,0.2,0.8]."
        ),
    )
    parser.add_argument(
        "--seqlens",
        default=None,
        help=(
            "Optional comma-separated sequence lengths for official profiles. "
            "Accepts integers or k-suffix values such as 1k,2k,4096."
        ),
    )
    parser.add_argument(
        "--sparse-kind",
        choices=("all", "block", "var-block"),
        default="all",
        help="Sparse benchmark family to run for profile=official-sparse.",
    )
    parser.add_argument(
        "--sparsity-ratios",
        default=None,
        help="Optional comma-separated sparsity ratios for profile=official-sparse.",
    )
    parser.add_argument(
        "--block-pairs",
        default=None,
        help=(
            "Optional comma-separated qxk block pairs for block-sparse, "
            "for example 128x128,128x32."
        ),
    )
    parser.add_argument(
        "--var-block-sizes",
        default=None,
        help="Optional comma-separated average block sizes for variable block-sparse.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="Input dtype for generated Q/K/V tensors.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for performance mode.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measured iterations for performance mode.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats; median is reported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible generated inputs.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for accuracy checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for accuracy checks.",
    )
    parser.add_argument(
        "--include-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include FlashInfer auto backend when a comparable API exists.",
    )
    parser.add_argument(
        "--fail-on-perf-regression",
        action="store_true",
        help="Exit non-zero if backend=ffa is slower than auto by more than max-regression.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Allowed relative slowdown when --fail-on-perf-regression is set.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write machine-readable results.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


OFFICIAL_DENSE_SEQLENS = tuple(k * 1024 for k in (1, 2, 4, 8, 16, 24, 32, 64))
OFFICIAL_DENSE_MASKS = (
    "full",
    "causal",
    "varlen_full",
    "varlen_causal",
    "sliding_window_causal",
    "varlen_block_causal",
)
OFFICIAL_BLOCK_SPARSE_SEQLENS = (8192, 16384, 32768, 65536)
OFFICIAL_BLOCK_SPARSE_RATIOS = (0.05, 0.1, 0.2, 0.5, 1.0)
OFFICIAL_BLOCK_SPARSE_BLOCK_PAIRS = (
    (128, 128),
    (128, 64),
    (128, 32),
    (128, 16),
    (128, 8),
    (128, 1),
)
OFFICIAL_VAR_BLOCK_SPARSE_SEQLENS = (49152, 16384)
OFFICIAL_VAR_BLOCK_SPARSE_RATIOS = (0.1, 0.2, 0.5, 0.8, 0.9)
OFFICIAL_VAR_BLOCK_SIZES = (128, 256, 512, 1024)
OFFICIAL_VARLEN_DISTRIBUTION = (
    (0, 2 * 1024, 0.16),
    (2 * 1024, 4 * 1024, 0.05),
    (4 * 1024, 8 * 1024, 0.04),
    (8 * 1024, 16 * 1024, 0.06),
    (16 * 1024, 32 * 1024, 0.08),
    (32 * 1024, 64 * 1024, 0.21),
    (64 * 1024, 128 * 1024, 0.4),
    (128 * 1024, 256 * 1024, 0.2),
    (256 * 1024, 512 * 1024, 0.05),
    (512 * 1024, 1024 * 1024, 0.04),
    (1024 * 1024, 2048 * 1024, 0.01),
    (2048 * 1024, 4096 * 1024, 0.01),
)


def parse_seqlens(value: str | None) -> tuple[int, ...]:
    if value is None:
        return OFFICIAL_DENSE_SEQLENS
    seqlens = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item.endswith("k"):
            seqlens.append(int(item[:-1]) * 1024)
        else:
            seqlens.append(int(item))
    if not seqlens:
        raise ValueError("--seqlens must contain at least one sequence length")
    return tuple(seqlens)


def parse_float_list(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return tuple(values)


def parse_int_list(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    values = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item.endswith("k"):
            values.append(int(item[:-1]) * 1024)
        else:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(values)


def parse_block_pairs(
    value: str | None,
    default: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if value is None:
        return default
    pairs = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid block pair '{raw_item}', expected QxK")
        q_block, k_block = item.split("x", 1)
        pairs.append((int(q_block), int(k_block)))
    if not pairs:
        raise ValueError("Expected at least one block pair")
    return tuple(pairs)


def official_varlen_doc_lens(total_seqlen: int, *, seed: int) -> tuple[int, ...]:
    """Use MagiAttention's benchmark varlen length sampler with a local seed."""
    import random

    rng = random.Random(seed)
    total_weight = sum(item[2] for item in OFFICIAL_VARLEN_DISTRIBUTION)
    current_total = 0
    doc_lens = []
    while current_total < total_seqlen:
        remaining = total_seqlen - current_total
        available = [
            (low, high, probability / total_weight)
            for low, high, probability in OFFICIAL_VARLEN_DISTRIBUTION
            if low < high and low <= remaining
        ]
        if not available:
            raise ValueError(f"No valid varlen interval for remaining={remaining}")

        intervals = [(low, high) for low, high, _ in available]
        weights = [weight for _, _, weight in available]
        low, high = rng.choices(intervals, weights=weights, k=1)[0]
        max_val = min(high - 1, remaining)
        length = rng.randint(low, max_val)
        doc_lens.append(length)
        current_total += length

    return tuple(length for length in doc_lens if length > 0)


def require_runtime(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for FFA integration checks.")

    try:
        import flashinfer
        import flashinfer.ffa_kernels as ffa
    except ImportError as exc:
        raise SystemExit(f"Failed to import FlashInfer FFA integration: {exc}") from exc

    try:
        from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
    except ImportError as exc:
        raise SystemExit(
            "magi_attention is required. Install it before running this script."
        ) from exc

    return flashinfer, ffa, flex_flash_attn_func


def build_accuracy_cases(dtype_name: str, profile: str) -> list[FFAIntegrationCase]:
    cases = [
        FFAIntegrationCase(
            "single_causal_mha_s128",
            "single",
            dtype_name,
            q_len=128,
            k_len=128,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim=64,
            causal=True,
        ),
        FFAIntegrationCase(
            "single_full_gqa_s192",
            "single",
            dtype_name,
            q_len=192,
            k_len=192,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
        ),
        FFAIntegrationCase(
            "single_causal_gqa_q64_k256",
            "single",
            dtype_name,
            q_len=64,
            k_len=256,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
        ),
        FFAIntegrationCase(
            "varlen_causal_gqa_3docs",
            "varlen",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
            doc_lens=(64, 128, 32),
        ),
        FFAIntegrationCase(
            "varlen_full_gqa_3docs",
            "varlen",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
            doc_lens=(64, 128, 32),
        ),
        FFAIntegrationCase(
            "sliding_window_causal_gqa_s256_w64",
            "sliding_window_causal",
            dtype_name,
            q_len=256,
            k_len=256,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
            window_size=64,
        ),
        FFAIntegrationCase(
            "varlen_block_causal_gqa_3docs_b64",
            "varlen_block_causal",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
            doc_lens=(96, 128, 80),
            block_size=64,
        ),
        FFAIntegrationCase(
            "mixed_native_ranges",
            "mixed",
            dtype_name,
            q_len=256,
            k_len=256,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim=64,
            causal=True,
            doc_lens=(128, 128),
        ),
    ]
    if profile == "full":
        cases.extend(
            [
                FFAIntegrationCase(
                    "single_causal_gqa_s512",
                    "single",
                    dtype_name,
                    q_len=512,
                    k_len=512,
                    num_qo_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    causal=True,
                ),
                FFAIntegrationCase(
                    "varlen_causal_mha_4docs",
                    "varlen",
                    dtype_name,
                    q_len=0,
                    k_len=0,
                    num_qo_heads=16,
                    num_kv_heads=16,
                    head_dim=128,
                    causal=True,
                    doc_lens=(128, 64, 256, 32),
                ),
            ]
        )
    return cases


def build_official_dense_cases(
    dtype_name: str,
    seqlens: tuple[int, ...],
) -> list[FFAIntegrationCase]:
    cases: list[FFAIntegrationCase] = []
    for seqlen in seqlens:
        official_dims = dict(
            dtype_name=dtype_name,
            q_len=seqlen,
            k_len=seqlen,
            num_qo_heads=48,
            num_kv_heads=8,
            head_dim=128,
        )
        varlen_docs = official_varlen_doc_lens(seqlen, seed=seqlen)
        for mask_type in OFFICIAL_DENSE_MASKS:
            if mask_type == "full":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_full_s{seqlen}",
                        "single",
                        causal=False,
                        **official_dims,
                    )
                )
            elif mask_type == "causal":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_causal_s{seqlen}",
                        "single",
                        causal=True,
                        **official_dims,
                    )
                )
            elif mask_type == "varlen_full":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_varlen_full_s{seqlen}",
                        "varlen",
                        causal=False,
                        doc_lens=varlen_docs,
                        **official_dims,
                    )
                )
            elif mask_type == "varlen_causal":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_varlen_causal_s{seqlen}",
                        "varlen",
                        causal=True,
                        doc_lens=varlen_docs,
                        **official_dims,
                    )
                )
            elif mask_type == "sliding_window_causal":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_sliding_window_causal_s{seqlen}",
                        "sliding_window_causal",
                        causal=True,
                        window_size=min(1024, seqlen),
                        **official_dims,
                    )
                )
            elif mask_type == "varlen_block_causal":
                cases.append(
                    FFAIntegrationCase(
                        f"official_dense_varlen_block_causal_s{seqlen}",
                        "varlen_block_causal",
                        causal=False,
                        doc_lens=varlen_docs,
                        block_size=2048,
                        **official_dims,
                    )
                )
            else:
                raise ValueError(f"Unknown official dense mask type: {mask_type}")
    return cases


def build_perf_cases(
    dtype_name: str,
    profile: str,
    *,
    seqlens: tuple[int, ...] | None = None,
) -> list[FFAIntegrationCase]:
    if profile == "official-dense":
        return build_official_dense_cases(
            dtype_name,
            OFFICIAL_DENSE_SEQLENS if seqlens is None else seqlens,
        )

    cases = [
        FFAIntegrationCase(
            "single_causal_gqa_s512",
            "single",
            dtype_name,
            q_len=512,
            k_len=512,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
        ),
        FFAIntegrationCase(
            "single_full_gqa_s512",
            "single",
            dtype_name,
            q_len=512,
            k_len=512,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
        ),
        FFAIntegrationCase(
            "single_causal_gqa_q512_k2048",
            "single",
            dtype_name,
            q_len=512,
            k_len=2048,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
        ),
        FFAIntegrationCase(
            "varlen_causal_gqa_4docs",
            "varlen",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
            doc_lens=(256, 512, 128, 768),
        ),
        FFAIntegrationCase(
            "varlen_full_gqa_4docs",
            "varlen",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
            doc_lens=(256, 512, 128, 768),
        ),
        FFAIntegrationCase(
            "sliding_window_causal_gqa_s1024_w256",
            "sliding_window_causal",
            dtype_name,
            q_len=1024,
            k_len=1024,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=True,
            window_size=256,
        ),
        FFAIntegrationCase(
            "varlen_block_causal_gqa_4docs_b256",
            "varlen_block_causal",
            dtype_name,
            q_len=0,
            k_len=0,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            causal=False,
            doc_lens=(256, 512, 128, 768),
            block_size=256,
        ),
        FFAIntegrationCase(
            "mixed_native_ranges_4docs",
            "mixed",
            dtype_name,
            q_len=1280,
            k_len=1280,
            num_qo_heads=16,
            num_kv_heads=16,
            head_dim=128,
            causal=True,
            doc_lens=(256, 256, 384, 384),
        ),
    ]
    if profile == "full":
        cases.extend(
            [
                FFAIntegrationCase(
                    "single_causal_gqa_s2048",
                    "single",
                    dtype_name,
                    q_len=2048,
                    k_len=2048,
                    num_qo_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    causal=True,
                ),
                FFAIntegrationCase(
                    "single_full_gqa_s2048",
                    "single",
                    dtype_name,
                    q_len=2048,
                    k_len=2048,
                    num_qo_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    causal=False,
                ),
                FFAIntegrationCase(
                    "single_causal_gqa_q2048_k4096",
                    "single",
                    dtype_name,
                    q_len=2048,
                    k_len=4096,
                    num_qo_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    causal=True,
                ),
                FFAIntegrationCase(
                    "varlen_causal_gqa_6docs",
                    "varlen",
                    dtype_name,
                    q_len=0,
                    k_len=0,
                    num_qo_heads=32,
                    num_kv_heads=8,
                    head_dim=128,
                    causal=True,
                    doc_lens=(512, 1024, 256, 768, 384, 1536),
                ),
            ]
        )
    return cases


def build_official_sparse_cases(
    dtype_name: str,
    *,
    sparse_kind: str,
    seqlens: tuple[int, ...] | None,
    block_sparsity_ratios: tuple[float, ...],
    var_block_sparsity_ratios: tuple[float, ...],
    block_pairs: tuple[tuple[int, int], ...],
    var_block_sizes: tuple[int, ...],
) -> list[FFASparseIntegrationCase]:
    cases: list[FFASparseIntegrationCase] = []
    if sparse_kind in ("all", "block"):
        block_seqlens = OFFICIAL_BLOCK_SPARSE_SEQLENS if seqlens is None else seqlens
        for seqlen in block_seqlens:
            for q_block_size, k_block_size in block_pairs:
                if seqlen % q_block_size != 0 or seqlen % k_block_size != 0:
                    continue
                for sparsity_ratio in block_sparsity_ratios:
                    cases.append(
                        FFASparseIntegrationCase(
                            name=(
                                "official_block_sparse"
                                f"_s{seqlen}_sp{sparsity_ratio:g}"
                                f"_qb{q_block_size}_kb{k_block_size}"
                            ),
                            kind="block_sparse",
                            dtype_name=dtype_name,
                            seqlen=seqlen,
                            num_qo_heads=16,
                            num_kv_heads=4,
                            head_dim=128,
                            sparsity_ratio=sparsity_ratio,
                            q_block_size=q_block_size,
                            k_block_size=k_block_size,
                        )
                    )

    if sparse_kind in ("all", "var-block"):
        var_seqlens = OFFICIAL_VAR_BLOCK_SPARSE_SEQLENS if seqlens is None else seqlens
        for seqlen in var_seqlens:
            for block_size in var_block_sizes:
                if seqlen % block_size != 0:
                    continue
                for sparsity_ratio in var_block_sparsity_ratios:
                    cases.append(
                        FFASparseIntegrationCase(
                            name=(
                                "official_var_block_sparse"
                                f"_s{seqlen}_sp{sparsity_ratio:g}"
                                f"_b{block_size}"
                            ),
                            kind="var_block_sparse",
                            dtype_name=dtype_name,
                            seqlen=seqlen,
                            num_qo_heads=4,
                            num_kv_heads=4,
                            head_dim=128,
                            sparsity_ratio=sparsity_ratio,
                            var_block_size=block_size,
                        )
                    )
    return cases


def make_case_data(case: FFAIntegrationCase, device: torch.device, seed: int) -> CaseData:
    dtype = dtype_from_name(case.dtype_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if case.kind in ("varlen", "varlen_block_causal"):
        total = sum(case.doc_lens)
        q_len = total
        k_len = total
    else:
        q_len = case.q_len
        k_len = case.k_len

    q = torch.randn(
        q_len,
        case.num_qo_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        k_len,
        case.num_kv_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        k_len,
        case.num_kv_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )

    if case.kind == "single":
        mask = 1 if case.causal else 0
        q_ranges = torch.tensor([[0, q_len]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, k_len]], dtype=torch.int32, device=device)
        attn_type_map = torch.tensor([mask], dtype=torch.int32, device=device)
        return CaseData(q, k, v, q_ranges, k_ranges, attn_type_map)

    if case.kind == "varlen":
        cu = torch.tensor(
            [0] + torch.tensor(case.doc_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=device,
        )
        ranges = [
            [int(cu[i].item()), int(cu[i + 1].item())]
            for i in range(len(case.doc_lens))
        ]
        q_ranges = torch.tensor(ranges, dtype=torch.int32, device=device)
        k_ranges = torch.tensor(ranges, dtype=torch.int32, device=device)
        attn_type_map = torch.full(
            (len(case.doc_lens),),
            1 if case.causal else 0,
            dtype=torch.int32,
            device=device,
        )
        return CaseData(q, k, v, q_ranges, k_ranges, attn_type_map, cu, cu)

    if case.kind == "sliding_window_causal":
        window_size = case.window_size
        if window_size <= 0:
            raise ValueError("sliding_window_causal cases require window_size > 0")
        if q_len != k_len:
            raise ValueError("sliding_window_causal cases require q_len == k_len")
        first_end = min(window_size, q_len)
        q_ranges = torch.tensor(
            [[0, first_end], [first_end, q_len]],
            dtype=torch.int32,
            device=device,
        )
        k_ranges = torch.tensor(
            [[0, first_end], [0, k_len]],
            dtype=torch.int32,
            device=device,
        )
        attn_type_map = torch.tensor(
            [1, 3], dtype=torch.int32, device=device
        )
        official_area = first_end * (first_end + 1) // 2
        official_area += max(0, q_len - first_end) * (window_size + 1)
        official_flops = float(
            4 * official_area * case.num_qo_heads * case.head_dim
        )
        return CaseData(
            q,
            k,
            v,
            q_ranges,
            k_ranges,
            attn_type_map,
            flops=official_flops,
        )

    if case.kind == "varlen_block_causal":
        if case.block_size <= 0:
            raise ValueError("varlen_block_causal cases require block_size > 0")
        cu = [0] + torch.tensor(case.doc_lens).cumsum(0).tolist()
        q_ranges_list: list[list[int]] = []
        k_ranges_list: list[list[int]] = []
        for doc_len, start_offset in zip(case.doc_lens, cu[:-1]):
            num_blocks = (doc_len + case.block_size - 1) // case.block_size
            for block_idx in range(num_blocks):
                start = block_idx * case.block_size
                end = min((block_idx + 1) * case.block_size, doc_len)
                q_ranges_list.append(
                    [start + start_offset, end + start_offset]
                )
                k_ranges_list.append([start_offset, end + start_offset])
        q_ranges = torch.tensor(q_ranges_list, dtype=torch.int32, device=device)
        k_ranges = torch.tensor(k_ranges_list, dtype=torch.int32, device=device)
        attn_type_map = torch.zeros(
            len(q_ranges_list), dtype=torch.int32, device=device
        )
        return CaseData(q, k, v, q_ranges, k_ranges, attn_type_map)

    if case.kind == "mixed":
        if sum(case.doc_lens) != q_len or q_len != k_len:
            raise ValueError("mixed cases expect doc_lens to sum to q_len == k_len")
        starts = [0] + list(torch.tensor(case.doc_lens).cumsum(0).tolist())
        ranges = [[starts[i], starts[i + 1]] for i in range(len(case.doc_lens))]
        q_ranges = torch.tensor(ranges, dtype=torch.int32, device=device)
        k_ranges = torch.tensor(ranges, dtype=torch.int32, device=device)
        mask_values = [1 if i % 2 == 0 else 0 for i in range(len(ranges))]
        attn_type_map = torch.tensor(mask_values, dtype=torch.int32, device=device)
        return CaseData(q, k, v, q_ranges, k_ranges, attn_type_map)

    raise ValueError(f"Unknown case kind: {case.kind}")


def make_sparse_case_data(
    case: FFASparseIntegrationCase,
    device: torch.device,
    seed: int,
) -> SparseCaseData:
    from magi_attention.utils.sparse_utils import (
        choose_ref_block,
        flatten_block_mask,
        generate_block_sparse_pattern,
        generate_ranges_from_block_mask_triton,
        generate_ranges_from_var_block_mask,
        generate_variable_block_sparse_pattern,
    )

    dtype = dtype_from_name(case.dtype_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    q_base = torch.randn(
        1,
        case.seqlen,
        case.num_qo_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    k_base = torch.randn(
        1,
        case.seqlen,
        case.num_kv_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    v_base = torch.randn(
        1,
        case.seqlen,
        case.num_kv_heads,
        case.head_dim,
        dtype=dtype,
        device=device,
    )

    if case.kind == "block_sparse":
        num_q_blocks = case.seqlen // case.q_block_size
        num_kv_blocks = case.seqlen // case.k_block_size
        block_mask, _ = generate_block_sparse_pattern(
            num_q_heads=case.num_qo_heads,
            num_kv_heads=case.num_kv_heads,
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            sparsity=case.sparsity_ratio,
            sparse_format="block_mask",
            device=str(device),
        )
        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
            block_mask,
            case.q_block_size,
            case.k_block_size,
        )
        attn_type_map = torch.zeros(
            len(q_ranges), dtype=torch.int32, device=device
        )

        qhead_per_khead = case.num_qo_heads // case.num_kv_heads
        q = (
            q_base.view(
                1,
                case.seqlen,
                case.num_kv_heads,
                qhead_per_khead,
                case.head_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(case.num_kv_heads * case.seqlen, qhead_per_khead, case.head_dim)
            .contiguous()
        )
        k = (
            k_base.permute(0, 2, 1, 3)
            .reshape(case.num_kv_heads * case.seqlen, 1, case.head_dim)
            .contiguous()
        )
        v = (
            v_base.permute(0, 2, 1, 3)
            .reshape(case.num_kv_heads * case.seqlen, 1, case.head_dim)
            .contiguous()
        )

        plan_kwargs = choose_ref_block(
            (case.q_block_size, case.k_block_size),
            qhead_per_khead=qhead_per_khead,
        )
        plan_kwargs["auto_range_merge"] = True
        plan_kwargs["disable_fwd_atomic_reduction"] = True

    elif case.kind == "var_block_sparse":
        block_size = case.var_block_size
        num_q_blocks = case.seqlen // block_size
        num_kv_blocks = case.seqlen // block_size
        block_mask, block_row_sz, block_col_sz = generate_variable_block_sparse_pattern(
            case.num_qo_heads,
            case.num_kv_heads,
            case.seqlen,
            case.seqlen,
            num_q_blocks,
            num_kv_blocks,
            min_q_block_size=128,
            min_kv_block_size=128,
            sparsity=case.sparsity_ratio,
            device=str(device),
        )
        flat_block_sparse_mask = flatten_block_mask(
            block_mask,
            case.num_qo_heads,
            case.num_kv_heads,
        )
        q_ranges, k_ranges = generate_ranges_from_var_block_mask(
            flat_block_sparse_mask,
            block_row_sz,
            block_col_sz,
            case.num_qo_heads,
            case.num_kv_heads,
        )
        attn_type_map = torch.zeros(
            len(q_ranges), dtype=torch.int32, device=device
        )

        q = (
            q_base.permute(0, 2, 1, 3)
            .reshape(case.num_qo_heads * case.seqlen, 1, case.head_dim)
            .contiguous()
        )
        k = (
            k_base.permute(0, 2, 1, 3)
            .reshape(case.num_kv_heads * case.seqlen, 1, case.head_dim)
            .contiguous()
        )
        v = (
            v_base.permute(0, 2, 1, 3)
            .reshape(case.num_kv_heads * case.seqlen, 1, case.head_dim)
            .contiguous()
        )
        plan_kwargs = {"auto_range_merge": True}

    else:
        raise ValueError(f"Unknown sparse case kind: {case.kind}")

    flops = (
        4
        * case.seqlen
        * case.seqlen
        * case.num_qo_heads
        * case.head_dim
        * case.sparsity_ratio
    )
    return SparseCaseData(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges.to(device=device, dtype=torch.int32),
        k_ranges=k_ranges.to(device=device, dtype=torch.int32),
        attn_type_map=attn_type_map,
        plan_num_qo_heads=q.shape[1],
        plan_num_kv_heads=k.shape[1],
        plan_kwargs=plan_kwargs,
        flops=float(flops),
    )


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask_type: int,
    sm_scale: float,
) -> torch.Tensor:
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    group_size = num_qo_heads // num_kv_heads
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    if group_size > 1:
        k_f = k_f.repeat_interleave(group_size, dim=1)
        v_f = v_f.repeat_interleave(group_size, dim=1)

    q_len = q.shape[0]
    k_len = k.shape[0]
    attn = torch.einsum("qhd,khd->hqk", q_f, k_f) * sm_scale
    row_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    col_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
    if mask_type == 0:
        mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
    elif mask_type == 1:
        mask = col_idx <= row_idx + (k_len - q_len)
    elif mask_type == 2:
        mask = col_idx >= row_idx
    elif mask_type == 3:
        mask = (col_idx <= row_idx + (k_len - q_len)) & (col_idx >= row_idx)
    else:
        raise ValueError(f"Unknown FFA mask type: {mask_type}")
    attn = attn.masked_fill(~mask.unsqueeze(0), float("-inf"))

    prob = torch.softmax(attn, dim=-1)
    prob = torch.where(torch.isfinite(prob), prob, torch.zeros_like(prob))
    out = torch.einsum("hqk,khd->qhd", prob, v_f)
    return out.to(q.dtype)


def reference_case(
    case: FFAIntegrationCase,
    data: CaseData,
    *,
    sm_scale: float,
) -> torch.Tensor:
    if case.kind == "single":
        return reference_attention(
            data.q,
            data.k,
            data.v,
            mask_type=1 if case.causal else 0,
            sm_scale=sm_scale,
        )

    if case.kind == "varlen":
        refs = []
        assert data.qo_indptr is not None and data.kv_indptr is not None
        for i in range(len(case.doc_lens)):
            qs = int(data.qo_indptr[i].item())
            qe = int(data.qo_indptr[i + 1].item())
            ks = int(data.kv_indptr[i].item())
            ke = int(data.kv_indptr[i + 1].item())
            refs.append(
                reference_attention(
                    data.q[qs:qe],
                    data.k[ks:ke],
                    data.v[ks:ke],
                    mask_type=1 if case.causal else 0,
                    sm_scale=sm_scale,
                )
            )
        return torch.cat(refs, dim=0)

    if case.kind in ("mixed", "sliding_window_causal", "varlen_block_causal"):
        out = torch.empty_like(data.q)
        q_ranges_cpu = data.q_ranges.cpu()
        k_ranges_cpu = data.k_ranges.cpu()
        attn_type_cpu = data.attn_type_map.cpu()
        for i in range(data.q_ranges.shape[0]):
            qs, qe = [int(x) for x in q_ranges_cpu[i]]
            ks, ke = [int(x) for x in k_ranges_cpu[i]]
            mask_type = int(attn_type_cpu[i].item())
            out[qs:qe] = reference_attention(
                data.q[qs:qe],
                data.k[ks:ke],
                data.v[ks:ke],
                mask_type=mask_type,
                sm_scale=sm_scale,
            )
        return out

    raise ValueError(f"Unknown case kind: {case.kind}")


def magi_flex_call(
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    data: CaseData,
    *,
    sm_scale: float,
) -> torch.Tensor:
    out, _ = flex_flash_attn_func(
        data.q,
        data.k,
        data.v,
        q_ranges=data.q_ranges,
        k_ranges=data.k_ranges,
        attn_type_map=data.attn_type_map,
        softmax_scale=sm_scale,
        softcap=0.0,
        deterministic=False,
        pack_gqa=False,
        disable_fwd_atomic_reduction=False,
    )
    return out


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float, str]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected_f.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    try:
        torch.testing.assert_close(actual_f, expected_f, rtol=rtol, atol=atol)
        return True, max_abs, max_rel, ""
    except AssertionError as exc:
        return False, max_abs, max_rel, str(exc).splitlines()[0]


def build_accuracy_impls(
    case: FFAIntegrationCase,
    data: CaseData,
    flashinfer: Any,
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    workspace: torch.Tensor,
    *,
    include_auto: bool,
    include_flex_prefill: bool = True,
    sm_scale: float,
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    impls: list[tuple[str, Callable[[], torch.Tensor]]] = [
        (
            "magi.flex_flash_attn_func",
            lambda: magi_flex_call(
                flex_flash_attn_func,
                data,
                sm_scale=sm_scale,
            ),
        ),
    ]
    if include_flex_prefill:
        impls.append(
            (
                "flashinfer.ffa_kernels.flex_prefill",
                lambda: ffa.flex_prefill(
                    data.q,
                    data.k,
                    data.v,
                    data.q_ranges,
                    data.k_ranges,
                    data.attn_type_map,
                    sm_scale=sm_scale,
                ),
            )
        )

    if case.kind in (
        "single",
        "varlen",
        "mixed",
        "sliding_window_causal",
        "varlen_block_causal",
    ):
        impls.append(
            (
                "flashinfer.ffa_kernels.BatchPrefillFFAWrapper/native-ranges",
                make_native_ffa_wrapper_runner(
                    ffa,
                    workspace,
                    case=case,
                    data=data,
                    sm_scale=sm_scale,
                ),
            )
        )

    if case.kind == "single":
        if case.causal:
            impls.append(
                (
                    "flashinfer.ffa_kernels.causal_prefill",
                    lambda: ffa.causal_prefill(
                        data.q,
                        data.k,
                        data.v,
                        sm_scale=sm_scale,
                    ),
                )
            )
        impls.append(
            (
                "flashinfer.single_prefill/backend=ffa",
                lambda: flashinfer.single_prefill_with_kv_cache(
                    data.q,
                    data.k,
                    data.v,
                    causal=case.causal,
                    sm_scale=sm_scale,
                    backend="ffa",
                ),
            )
        )
        if include_auto:
            impls.append(
                (
                    "flashinfer.single_prefill/auto",
                    lambda: flashinfer.single_prefill_with_kv_cache(
                        data.q,
                        data.k,
                        data.v,
                        causal=case.causal,
                        sm_scale=sm_scale,
                        backend="auto",
                    ),
                )
            )

    if case.kind == "varlen":
        assert data.qo_indptr is not None and data.kv_indptr is not None
        if case.causal:
            impls.append(
                (
                    "flashinfer.ffa_kernels.varlen_causal_prefill",
                    lambda: ffa.varlen_causal_prefill(
                        data.q,
                        data.k,
                        data.v,
                        data.qo_indptr,
                        data.kv_indptr,
                        sm_scale=sm_scale,
                    ),
                )
            )

        impls.append(
            (
                "flashinfer.BatchPrefillRagged/backend=ffa",
                make_ragged_wrapper_runner(
                    flashinfer,
                    workspace,
                    backend="ffa",
                    case=case,
                    data=data,
                    sm_scale=sm_scale,
                ),
            )
        )

        if include_auto:
            impls.append(
                (
                    "flashinfer.BatchPrefillRagged/auto",
                    make_ragged_wrapper_runner(
                        flashinfer,
                        workspace,
                        backend="auto",
                        case=case,
                        data=data,
                        sm_scale=sm_scale,
                    ),
                )
            )

    return impls


def make_ragged_wrapper_runner(
    flashinfer: Any,
    workspace: torch.Tensor,
    *,
    backend: str,
    case: FFAIntegrationCase,
    data: CaseData,
    sm_scale: float,
) -> Callable[[], torch.Tensor]:
    cache: dict[str, Any] = {}

    def run() -> torch.Tensor:
        wrapper = cache.get("wrapper")
        if wrapper is None:
            assert data.qo_indptr is not None and data.kv_indptr is not None
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                backend=backend,
            )
            wrapper.plan(
                data.qo_indptr,
                data.kv_indptr,
                case.num_qo_heads,
                case.num_kv_heads,
                case.head_dim,
                causal=case.causal,
                sm_scale=sm_scale,
                q_data_type=dtype_from_name(case.dtype_name),
                kv_data_type=dtype_from_name(case.dtype_name),
            )
            cache["wrapper"] = wrapper
        return wrapper.run(data.q, data.k, data.v)

    return run


def make_native_ffa_wrapper_runner(
    ffa: Any,
    workspace: torch.Tensor,
    *,
    case: FFAIntegrationCase,
    data: CaseData,
    sm_scale: float,
) -> Callable[[], torch.Tensor]:
    cache: dict[str, Any] = {}

    def run() -> torch.Tensor:
        wrapper = cache.get("wrapper")
        if wrapper is None:
            wrapper = ffa.BatchPrefillFFAWrapper(workspace, kv_layout="NHD")
            wrapper.plan(
                q_ranges=data.q_ranges,
                k_ranges=data.k_ranges,
                attn_type_map=data.attn_type_map,
                num_qo_heads=case.num_qo_heads,
                num_kv_heads=case.num_kv_heads,
                head_dim=case.head_dim,
                sm_scale=sm_scale,
            )
            cache["wrapper"] = wrapper
        return wrapper.run(data.q, data.k, data.v)

    return run


def magi_sparse_call(
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    data: SparseCaseData,
    *,
    sm_scale: float,
) -> torch.Tensor:
    out, _ = flex_flash_attn_func(
        data.q,
        data.k,
        data.v,
        q_ranges=data.q_ranges,
        k_ranges=data.k_ranges,
        attn_type_map=data.attn_type_map,
        softmax_scale=sm_scale,
        **data.plan_kwargs,
    )
    return out


def make_sparse_ffa_wrapper_runner(
    ffa: Any,
    workspace: torch.Tensor,
    *,
    case: FFASparseIntegrationCase,
    data: SparseCaseData,
    sm_scale: float,
) -> Callable[[], torch.Tensor]:
    cache: dict[str, Any] = {}

    def run() -> torch.Tensor:
        wrapper = cache.get("wrapper")
        if wrapper is None:
            wrapper = ffa.BatchPrefillFFAWrapper(workspace, kv_layout="NHD")
            wrapper.plan(
                q_ranges=data.q_ranges,
                k_ranges=data.k_ranges,
                attn_type_map=data.attn_type_map,
                num_qo_heads=data.plan_num_qo_heads,
                num_kv_heads=data.plan_num_kv_heads,
                head_dim=case.head_dim,
                sm_scale=sm_scale,
                **data.plan_kwargs,
            )
            cache["wrapper"] = wrapper
        return wrapper.run(data.q, data.k, data.v)

    return run


def build_sparse_impls(
    case: FFASparseIntegrationCase,
    data: SparseCaseData,
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    workspace: torch.Tensor,
    *,
    sm_scale: float,
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    return [
        (
            "magi.flex_flash_attn_func",
            lambda: magi_sparse_call(
                flex_flash_attn_func,
                data,
                sm_scale=sm_scale,
            ),
        ),
        (
            "flashinfer.ffa_kernels.BatchPrefillFFAWrapper/native-ranges",
            make_sparse_ffa_wrapper_runner(
                ffa,
                workspace,
                case=case,
                data=data,
                sm_scale=sm_scale,
            ),
        ),
    ]


def run_accuracy(
    cases: list[FFAIntegrationCase],
    flashinfer: Any,
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    args: argparse.Namespace,
    *,
    use_magi_reference: bool = False,
) -> list[AccuracyResult]:
    device = torch.device(args.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    results: list[AccuracyResult] = []

    for index, case in enumerate(cases):
        data = make_case_data(case, device, seed=args.seed + index)
        sm_scale = 1.0 / math.sqrt(case.head_dim)
        if use_magi_reference:
            try:
                expected = magi_flex_call(
                    flex_flash_attn_func,
                    data,
                    sm_scale=sm_scale,
                )
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001 - keep script diagnostic.
                results.append(
                    AccuracyResult(
                        case=case.name,
                        impl="magi.flex_flash_attn_func",
                        status="ERROR",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
        else:
            expected = reference_case(case, data, sm_scale=sm_scale)
        impls = build_accuracy_impls(
            case,
            data,
            flashinfer,
            ffa,
            flex_flash_attn_func,
            workspace,
            include_auto=args.include_auto,
            sm_scale=sm_scale,
        )

        for impl_name, fn in impls:
            if use_magi_reference and impl_name == "magi.flex_flash_attn_func":
                continue
            try:
                actual = fn()
                torch.cuda.synchronize(device)
                ok, max_abs, max_rel, message = compare_tensors(
                    actual,
                    expected,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                results.append(
                    AccuracyResult(
                        case=case.name,
                        impl=impl_name,
                        status="PASS" if ok else "FAIL",
                        max_abs=max_abs,
                        max_rel=max_rel,
                        message=message,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep script diagnostic.
                results.append(
                    AccuracyResult(
                        case=case.name,
                        impl=impl_name,
                        status="ERROR",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

    return results


def run_sparse_accuracy(
    cases: list[FFASparseIntegrationCase],
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    args: argparse.Namespace,
) -> list[AccuracyResult]:
    device = torch.device(args.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    results: list[AccuracyResult] = []

    for index, case in enumerate(cases):
        data = make_sparse_case_data(case, device, seed=args.seed + index)
        sm_scale = 1.0 / math.sqrt(case.head_dim)
        try:
            expected = magi_sparse_call(
                flex_flash_attn_func,
                data,
                sm_scale=sm_scale,
            )
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 - keep script diagnostic.
            results.append(
                AccuracyResult(
                    case=case.name,
                    impl="magi.flex_flash_attn_func",
                    status="ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        impls = build_sparse_impls(
            case,
            data,
            ffa,
            flex_flash_attn_func,
            workspace,
            sm_scale=sm_scale,
        )

        for impl_name, fn in impls:
            if impl_name == "magi.flex_flash_attn_func":
                continue
            try:
                actual = fn()
                torch.cuda.synchronize(device)
                ok, max_abs, max_rel, message = compare_tensors(
                    actual,
                    expected,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                results.append(
                    AccuracyResult(
                        case=case.name,
                        impl=impl_name,
                        status="PASS" if ok else "FAIL",
                        max_abs=max_abs,
                        max_rel=max_rel,
                        message=message,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep script diagnostic.
                results.append(
                    AccuracyResult(
                        case=case.name,
                        impl=impl_name,
                        status="ERROR",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

    return results


def mask_area(q_len: int, k_len: int, mask_type: int) -> int:
    if q_len <= 0 or k_len <= 0:
        return 0
    if mask_type == 0:
        return q_len * k_len

    area = 0
    for row in range(q_len):
        if mask_type == 1:
            area += max(0, min(k_len, row + (k_len - q_len) + 1))
        elif mask_type == 2:
            area += max(0, k_len - row)
        elif mask_type == 3:
            lower = row
            upper = row + (k_len - q_len)
            area += max(0, min(k_len - 1, upper) - max(0, lower) + 1)
        else:
            raise ValueError(f"Unknown FFA mask type: {mask_type}")
    return area


def attention_flops(case: FFAIntegrationCase, data: CaseData) -> float:
    if data.flops is not None:
        return data.flops
    total_area = 0
    q_ranges_cpu = data.q_ranges.cpu()
    k_ranges_cpu = data.k_ranges.cpu()
    attn_type_cpu = data.attn_type_map.cpu()
    for i in range(data.q_ranges.shape[0]):
        qs, qe = [int(x) for x in q_ranges_cpu[i]]
        ks, ke = [int(x) for x in k_ranges_cpu[i]]
        total_area += mask_area(qe - qs, ke - ks, int(attn_type_cpu[i].item()))
    return float(4 * total_area * case.num_qo_heads * case.head_dim)


def bench_cuda_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    repeats: int,
) -> float:
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize(device)

    measurements = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        measurements.append(start.elapsed_time(end) / iters)
    return float(statistics.median(measurements))


def bench_magi_official_ms(
    fn: Callable[[], torch.Tensor],
) -> tuple[float, float, float]:
    from magi_attention.benchmarking import do_bench_flops

    perf_dict = do_bench_flops(
        fn,
        quantiles=[0.5, 0.2, 0.8],
        mem_record_mode="peak",
    )
    values = perf_dict["flops"]
    if not isinstance(values, list):
        values = [values]
    if len(values) != 3:
        raise RuntimeError(f"Expected three official benchmark quantiles, got {values}")
    p50, p20, p80 = [float(value) for value in values]
    return p50, p20, p80


def build_perf_impls(
    case: FFAIntegrationCase,
    data: CaseData,
    flashinfer: Any,
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    workspace: torch.Tensor,
    *,
    include_auto: bool,
    sm_scale: float,
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    impls = build_accuracy_impls(
        case,
        data,
        flashinfer,
        ffa,
        flex_flash_attn_func,
        workspace,
        include_auto=False,
        include_flex_prefill=False,
        sm_scale=sm_scale,
    )
    if include_auto and case.kind == "single":
        impls.insert(
            0,
            (
                "flashinfer.single_prefill/auto",
                lambda: flashinfer.single_prefill_with_kv_cache(
                    data.q,
                    data.k,
                    data.v,
                    causal=case.causal,
                    sm_scale=sm_scale,
                    backend="auto",
                ),
            ),
        )
    if include_auto and case.kind == "varlen":
        assert data.qo_indptr is not None and data.kv_indptr is not None
        impls.insert(
            0,
            (
                "flashinfer.BatchPrefillRagged/auto",
                make_ragged_wrapper_runner(
                    flashinfer,
                    workspace,
                    backend="auto",
                    case=case,
                    data=data,
                    sm_scale=sm_scale,
                ),
            ),
        )
    return impls


def bench_perf_impls(
    case_name: str,
    impls: list[tuple[str, Callable[[], torch.Tensor]]],
    *,
    flops: float,
    device: torch.device,
    args: argparse.Namespace,
) -> list[PerfResult]:
    results: list[PerfResult] = []
    for impl_name, fn in impls:
        try:
            if args.bench_method == "magi-official":
                ms, ms_p20, ms_p80 = bench_magi_official_ms(fn)
            else:
                ms = bench_cuda_ms(
                    fn,
                    device=device,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                ms_p20 = None
                ms_p80 = None
            results.append(
                PerfResult(
                    case=case_name,
                    impl=impl_name,
                    status="PASS",
                    ms=ms,
                    ms_p20=ms_p20,
                    ms_p80=ms_p80,
                    tflops=flops / (ms * 1e-3) / 1e12,
                    tflops_p20=(
                        flops / (ms_p20 * 1e-3) / 1e12
                        if ms_p20 is not None
                        else None
                    ),
                    tflops_p80=(
                        flops / (ms_p80 * 1e-3) / 1e12
                        if ms_p80 is not None
                        else None
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep script diagnostic.
            results.append(
                PerfResult(
                    case=case_name,
                    impl=impl_name,
                    status="ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    auto_ms = next(
        (
            result.ms
            for result in results
            if result.status == "PASS" and result.impl.endswith("/auto")
        ),
        None,
    )
    if auto_ms is not None:
        for result in results:
            if result.status == "PASS" and result.ms is not None:
                result.speedup_vs_auto = auto_ms / result.ms

    magi_ms = next(
        (
            result.ms
            for result in results
            if result.status == "PASS" and result.impl == "magi.flex_flash_attn_func"
        ),
        None,
    )
    if magi_ms is not None:
        for result in results:
            if result.status == "PASS" and result.ms is not None:
                result.overhead_vs_magi = result.ms / magi_ms - 1.0
    return results


def run_perf(
    cases: list[FFAIntegrationCase],
    flashinfer: Any,
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    args: argparse.Namespace,
) -> list[PerfResult]:
    device = torch.device(args.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    results: list[PerfResult] = []

    for index, case in enumerate(cases):
        data = make_case_data(case, device, seed=args.seed + 1000 + index)
        sm_scale = 1.0 / math.sqrt(case.head_dim)
        impls = build_perf_impls(
            case,
            data,
            flashinfer,
            ffa,
            flex_flash_attn_func,
            workspace,
            include_auto=args.include_auto,
            sm_scale=sm_scale,
        )
        flops = attention_flops(case, data)
        results.extend(
            bench_perf_impls(
                case.name,
                impls,
                flops=flops,
                device=device,
                args=args,
            )
        )

    return results


def run_sparse_perf(
    cases: list[FFASparseIntegrationCase],
    ffa: Any,
    flex_flash_attn_func: Callable[..., tuple[torch.Tensor, Any]],
    args: argparse.Namespace,
) -> list[PerfResult]:
    device = torch.device(args.device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    results: list[PerfResult] = []

    for index, case in enumerate(cases):
        data = make_sparse_case_data(case, device, seed=args.seed + 2000 + index)
        sm_scale = 1.0 / math.sqrt(case.head_dim)
        impls = build_sparse_impls(
            case,
            data,
            ffa,
            flex_flash_attn_func,
            workspace,
            sm_scale=sm_scale,
        )
        results.extend(
            bench_perf_impls(
                case.name,
                impls,
                flops=data.flops,
                device=device,
                args=args,
            )
        )

    return results


def print_accuracy(results: list[AccuracyResult]) -> None:
    print("\nAccuracy")
    rows = [
        [
            result.status,
            result.case,
            result.impl,
            format_float(result.max_abs),
            format_float(result.max_rel),
            result.message,
        ]
        for result in results
    ]
    print_table(["status", "case", "impl", "max_abs", "max_rel", "message"], rows)


def print_perf(results: list[PerfResult]) -> None:
    print("\nPerformance")
    rows = [
        [
            result.status,
            result.case,
            result.impl,
            format_float(result.ms),
            format_float(result.ms_p20),
            format_float(result.ms_p80),
            format_float(result.tflops),
            format_float(result.tflops_p20),
            format_float(result.tflops_p80),
            format_percent(result.overhead_vs_magi),
            format_float(result.speedup_vs_auto),
            result.message,
        ]
        for result in results
    ]
    print_table(
        [
            "status",
            "case",
            "impl",
            "ms",
            "p20",
            "p80",
            "tflops",
            "tflops_p20",
            "tflops_p80",
            "overhead_vs_magi",
            "speedup_vs_auto",
            "message",
        ],
        rows,
    )


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.3%}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep_line = "  ".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def write_json(
    path: Path,
    args: argparse.Namespace,
    accuracy: list[AccuracyResult],
    perf: list[PerfResult],
) -> None:
    payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "accuracy": [asdict(result) for result in accuracy],
        "performance": [asdict(result) for result in perf],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_perf_regression(
    perf: list[PerfResult],
    *,
    max_regression: float,
) -> list[str]:
    failures = []
    by_case: dict[str, list[PerfResult]] = {}
    for result in perf:
        by_case.setdefault(result.case, []).append(result)

    for case_name, case_results in by_case.items():
        auto_ms = next(
            (
                result.ms
                for result in case_results
                if result.status == "PASS" and result.impl.endswith("/auto")
            ),
            None,
        )
        if auto_ms is None:
            continue
        ffa_ms = next(
            (
                result.ms
                for result in case_results
                if result.status == "PASS"
                and result.impl.endswith("single_prefill/backend=ffa")
            ),
            None,
        )
        if ffa_ms is None:
            ffa_ms = next(
                (
                    result.ms
                    for result in case_results
                    if result.status == "PASS"
                    and result.impl.endswith("BatchPrefillRagged/backend=ffa")
                ),
                None,
            )
        if ffa_ms is None:
            continue
        slowdown = ffa_ms / auto_ms - 1.0
        if slowdown > max_regression:
            failures.append(
                f"{case_name}: backend=ffa is {slowdown:.2%} slower than auto"
            )
    return failures


def main() -> int:
    args = parse_args()
    if args.bench_method is None:
        args.bench_method = (
            "magi-official"
            if args.profile in ("official-dense", "official-sparse")
            else "cuda-loop"
        )
    device = torch.device(args.device)
    flashinfer, ffa, flex_flash_attn_func = require_runtime(device)
    official_dense_seqlens = parse_seqlens(args.seqlens)
    sparse_seqlens = parse_seqlens(args.seqlens) if args.seqlens is not None else None
    sparse_ratios = args.sparsity_ratios
    block_sparsity_ratios = parse_float_list(
        sparse_ratios,
        OFFICIAL_BLOCK_SPARSE_RATIOS,
    )
    var_block_sparsity_ratios = parse_float_list(
        sparse_ratios,
        OFFICIAL_VAR_BLOCK_SPARSE_RATIOS,
    )
    block_pairs = parse_block_pairs(
        args.block_pairs,
        OFFICIAL_BLOCK_SPARSE_BLOCK_PAIRS,
    )
    var_block_sizes = parse_int_list(
        args.var_block_sizes,
        OFFICIAL_VAR_BLOCK_SIZES,
    )
    sparse_cases = build_official_sparse_cases(
        args.dtype,
        sparse_kind=args.sparse_kind,
        seqlens=sparse_seqlens,
        block_sparsity_ratios=block_sparsity_ratios,
        var_block_sparsity_ratios=var_block_sparsity_ratios,
        block_pairs=block_pairs,
        var_block_sizes=var_block_sizes,
    )

    torch.cuda.set_device(device.index or 0)
    torch.set_grad_enabled(False)

    accuracy_results: list[AccuracyResult] = []
    perf_results: list[PerfResult] = []

    if args.mode in ("accuracy", "all"):
        if args.profile == "official-sparse":
            accuracy_results = run_sparse_accuracy(
                sparse_cases,
                ffa,
                flex_flash_attn_func,
                args,
            )
        else:
            if args.profile == "official-dense":
                accuracy_cases = build_official_dense_cases(
                    args.dtype,
                    official_dense_seqlens,
                )
                use_magi_reference = True
            else:
                accuracy_cases = build_accuracy_cases(args.dtype, args.profile)
                use_magi_reference = False
            accuracy_results = run_accuracy(
                accuracy_cases,
                flashinfer,
                ffa,
                flex_flash_attn_func,
                args,
                use_magi_reference=use_magi_reference,
            )
        print_accuracy(accuracy_results)

    if args.mode in ("perf", "all"):
        if args.profile == "official-sparse":
            perf_results = run_sparse_perf(
                sparse_cases,
                ffa,
                flex_flash_attn_func,
                args,
            )
        else:
            perf_cases = build_perf_cases(
                args.dtype,
                args.profile,
                seqlens=official_dense_seqlens
                if args.profile == "official-dense"
                else None,
            )
            perf_results = run_perf(
                perf_cases,
                flashinfer,
                ffa,
                flex_flash_attn_func,
                args,
            )
        print_perf(perf_results)

    if args.output_json is not None:
        write_json(args.output_json, args, accuracy_results, perf_results)
        print(f"\nWrote JSON results to {args.output_json}")

    failed_accuracy = [
        result
        for result in accuracy_results
        if result.status in ("FAIL", "ERROR")
        and not (
            result.impl.endswith("/auto")
            and result.status == "ERROR"
            and args.include_auto
        )
    ]
    if failed_accuracy:
        print("\nAccuracy failures detected.")
        return 1

    perf_errors = [
        result
        for result in perf_results
        if result.status == "ERROR"
        and not (
            result.impl.endswith("/auto")
            and result.status == "ERROR"
            and args.include_auto
        )
    ]
    if perf_errors:
        print("\nPerformance implementation errors detected.")
        return 1

    if args.fail_on_perf_regression:
        regressions = check_perf_regression(
            perf_results,
            max_regression=args.max_regression,
        )
        if regressions:
            print("\nPerformance regressions detected:")
            for regression in regressions:
                print(f"  {regression}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
