"""Shared real-API fixtures for production DA MoE acceptance tests."""

from __future__ import annotations

import math

import pytest
import torch

from benchmarks.bench_trtllm_moe_da import BenchmarkShape, _benchmark_precision
from flashinfer.utils import get_compute_capability


PRODUCTION_PRECISIONS = (
    "nvfp4",
    "mxfp4",
    "w4a16",
    "bf16",
    "fp8_per_tensor",
    "fp8_block",
    "mxfp8",
    "mxint4",
)


def require_sm100() -> None:
    """Skip a production DA test unless an SM100-family CUDA device is active."""
    if not torch.cuda.is_available():
        pytest.skip("production DA MoE requires CUDA")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("production DA MoE requires an SM100-family GPU")


def compact_shape(*, num_tokens: int = 32) -> BenchmarkShape:
    """Return a small supported shape that still exercises real TRTLLM kernels."""
    return BenchmarkShape(
        num_tokens=num_tokens,
        num_experts=32,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=4,
        hidden_size=256,
        intermediate_size=256,
        n_group=4,
        topk_group=2,
        tune_max_num_tokens=num_tokens,
    )


def deepseek_l0_shape() -> BenchmarkShape:
    """Return the binding DeepSeek L0 geometry used for policy decisions."""
    return BenchmarkShape(
        num_tokens=1024,
        num_experts=256,
        local_num_experts=32,
        local_expert_offset=0,
        top_k=8,
        hidden_size=7168,
        intermediate_size=2048,
        n_group=8,
        topk_group=4,
        tune_max_num_tokens=1024,
    )


def run_matched_public_graphs(
    precision: str,
    *,
    distributions: tuple[str, ...] = ("uniform", "ddist:4"),
    cache: str | None = None,
    tune: bool = True,
    num_tokens: int = 32,
    shape: BenchmarkShape | None = None,
) -> list[dict[str, object]]:
    """Run matched ordinary and DA graphs through one exact public routed ABI."""
    # Reuse the benchmark's public preparation/capture path with a compact correctness shape.
    rows = _benchmark_precision(
        precision,
        compact_shape(num_tokens=num_tokens) if shape is None else shape,
        distributions,
        cache,
        tune,
        0,
        1,
    )
    # Validate numerical, timing, policy, and topology fields exposed to benchmark users.
    assert len(rows) == len(distributions)
    for row in rows:
        assert row["status"] == "pass"
        assert row["finite"] is True
        assert math.isfinite(float(row["noda_ms"]))
        assert math.isfinite(float(row["da_ms"]))
        assert math.isfinite(float(row["noda_autotune_ms"]))
        assert math.isfinite(float(row["da_autotune_ms"]))
        assert float(row["noda_autotune_ms"]) >= 0.0
        assert float(row["da_autotune_ms"]) >= 0.0
        assert float(row["max_abs_difference"]) <= 3e-2
        assert row["policy"] in {
            "da_switch",
            "da_single_body",
            "da_fallback",
        }
        if row["capture_policy"] == "da_switch":
            assert row["conditional_nodes"] == 1
            assert row["is_selector_preamble_parallelizable"] is True
        else:
            assert row["conditional_nodes"] in (None, 0)
    return rows
