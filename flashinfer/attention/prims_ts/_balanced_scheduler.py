# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrated cost models for balanced PrimsTS MLA decode.

Descriptor placement is implemented only by the exact and optimized CUDA
schedulers. This module remains CUDA-independent so calibration metadata and
workload-bucket selection can be inspected by build and tuning tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence


_INT32_MAX = 2**31 - 1


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@dataclass(frozen=True)
class BalancedCostModel:
    """Integer-nanosecond model used to rank candidate schedules."""

    cost_per_k_tile: int
    fixed_piece_cost: int
    split_fixed_cost: int = 0
    split_piece_cost: int = 0
    reducer_fixed_cost: int = 0
    reducer_piece_cost: int = 0


# Initial B200/BF16 seed copied from the measured full-V 2CTA balanced MLA
# family.  Re-fit this against PrimsTS forced-piece curves after the first
# correct end-to-end implementation; task scheduling and the stock reducer can
# change both the intercept and the split penalty.
B200_BF16_2CTA_COST = BalancedCostModel(
    cost_per_k_tile=3046,
    fixed_piece_cost=7300,
    split_fixed_cost=0,
    split_piece_cost=6000,
    reducer_fixed_cost=2800,
    reducer_piece_cost=170,
)


# Calibration buckets intentionally depend only on replay-time scheduling
# metadata.  They can therefore change across graph replays without changing
# the captured kernel or any device allocation.
_DENSE_AVERAGE_TILES_PER_PARTITION = 32
_SPARSE_SMALL_ACTIVE_REQUESTS = 8
_DENSE_SMALL_ACTIVE_REQUESTS = 32


def balanced_cost_bucket(
    seq_lens: Sequence[int],
    *,
    num_partitions: int,
    k_tile_tokens: int = 128,
) -> str:
    """Classify a workload for selection of a calibrated scheduler model."""

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if k_tile_tokens <= 0:
        raise ValueError("k_tile_tokens must be positive")
    normalized = tuple(int(value) for value in seq_lens)
    if any(value < 0 for value in normalized):
        raise ValueError("seq_lens must be non-negative")
    active_requests = sum(value > 0 for value in normalized)
    if active_requests <= 1:
        return "single"
    total_tiles = sum(
        _ceil_div(value, k_tile_tokens) for value in normalized if value > 0
    )
    average_tiles = _ceil_div(total_tiles, num_partitions)
    if average_tiles < _DENSE_AVERAGE_TILES_PER_PARTITION:
        # High-batch, very sparse launches were not part of the calibrated
        # curve and can over-split their one moderately long request.  Retain
        # the broadly swept legacy model in that corner.
        if active_requests > 64 and average_tiles < 16:
            return "legacy"
        average_active_tiles = _ceil_div(total_tiles, active_requests)
        max_tiles = max(
            _ceil_div(value, k_tile_tokens) for value in normalized if value > 0
        )
        # The sparse-small calibration is specifically for a concentrated
        # long request beside a few short requests.  Small uniform batches
        # have a different optimum and must retain the general sparse model.
        if (
            active_requests <= _SPARSE_SMALL_ACTIVE_REQUESTS
            and max_tiles >= 4 * average_active_tiles
        ):
            return "sparse_small"
        if active_requests <= _SPARSE_SMALL_ACTIVE_REQUESTS:
            # Moderate skew was not represented by either calibrated
            # endpoint. Preserve the previously swept scheduler there until
            # the tuner has a dedicated curve instead of extrapolating.
            if max_tiles >= 2 * average_active_tiles:
                return "legacy"
            return "sparse_uniform_small"
        return "sparse"
    if active_requests <= _DENSE_SMALL_ACTIVE_REQUESTS:
        return "dense_small"
    return "dense_large"


# B200 calibration from ``bench_prims_ts_balanced_mla_cost_model.py`` using
# artifact ``primsts_balanced_cost_model_b200_20260914_schedule_v3.jsonl``.
# Keep family and dtype names independent of CuTe types so calibration metadata
# remains importable in CPU-only build and tuning tools.
B200_BALANCED_COST_MODELS = {
    ("1cta", "bf16", "single"): BalancedCostModel(1000, 0, 0, 0, 0, 46),
    ("1cta", "bf16", "sparse_small"): BalancedCostModel(1000, 0),
    ("1cta", "bf16", "sparse_uniform_small"): BalancedCostModel(1000, 0),
    ("1cta", "bf16", "sparse"): BalancedCostModel(1000, 0),
    ("1cta", "bf16", "dense_small"): BalancedCostModel(1000, 0, 17839, 32860, 0, 15),
    ("1cta", "bf16", "dense_large"): BalancedCostModel(1000, 0, 103595, 60618, 0, 2),
    ("1cta", "fp8", "single"): BalancedCostModel(1000, 0, 0, 0, 0, 46),
    ("1cta", "fp8", "sparse_small"): BalancedCostModel(1000, 0),
    ("1cta", "fp8", "sparse_uniform_small"): BalancedCostModel(1000, 0),
    ("1cta", "fp8", "sparse"): BalancedCostModel(1000, 0),
    ("1cta", "fp8", "dense_small"): BalancedCostModel(1000, 0, 0, 42469),
    ("1cta", "fp8", "dense_large"): BalancedCostModel(1000, 0),
    ("2cta", "bf16", "single"): BalancedCostModel(1000, 0, 0, 0, 0, 112),
    ("2cta", "bf16", "sparse_small"): BalancedCostModel(1000, 0),
    ("2cta", "bf16", "sparse_uniform_small"): BalancedCostModel(1000, 0),
    ("2cta", "bf16", "sparse"): BalancedCostModel(1000, 0),
    ("2cta", "bf16", "dense_small"): BalancedCostModel(1000, 0, 35244),
    ("2cta", "bf16", "dense_large"): BalancedCostModel(1000, 64000),
    ("2cta", "fp8", "single"): BalancedCostModel(1000, 0, 0, 0, 0, 202),
    ("2cta", "fp8", "sparse_small"): BalancedCostModel(1000, 0),
    ("2cta", "fp8", "sparse_uniform_small"): BalancedCostModel(1000, 0),
    ("2cta", "fp8", "sparse"): BalancedCostModel(1000, 0),
    ("2cta", "fp8", "dense_small"): BalancedCostModel(1000, 0, 0, 0, 0, 410),
    ("2cta", "fp8", "dense_large"): BalancedCostModel(1000, 1000),
}


@dataclass(frozen=True)
class BalancedCostModelCalibration:
    """Checked-in cost-model identity for one exact GPU configuration."""

    model_id: str
    device_name: str
    compute_capability: tuple[int, int]
    multi_processor_count: int


B200_BALANCED_COST_MODEL_ID = "b200-sm100-148sm-balanced-mla-v2"

# Calibration is intentionally matched by exact product name, compute
# capability, and SM count. Architecture-family or name-substring fallbacks can
# silently apply a model to a device whose optimal split policy was never
# measured. Add a row only after running the calibration benchmark on that
# device and checking its emitted models into this module.
BALANCED_COST_MODEL_CALIBRATIONS: Mapping[
    tuple[str, tuple[int, int], int], BalancedCostModelCalibration
] = MappingProxyType(
    {
        ("NVIDIA B200", (10, 0), 148): BalancedCostModelCalibration(
            model_id=B200_BALANCED_COST_MODEL_ID,
            device_name="NVIDIA B200",
            compute_capability=(10, 0),
            multi_processor_count=148,
        ),
    }
)


def require_balanced_cost_model_calibration(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
    multi_processor_count: int,
    device_index: int | None = None,
) -> BalancedCostModelCalibration:
    """Return an exact device calibration or reject balanced execution."""

    identity: tuple[str, tuple[int, int], int] = (
        str(device_name),
        (int(compute_capability[0]), int(compute_capability[1])),
        int(multi_processor_count),
    )
    try:
        return BALANCED_COST_MODEL_CALIBRATIONS[identity]
    except KeyError as error:
        device_label = "the selected CUDA device"
        device_argument = "cuda:<index>"
        if device_index is not None:
            device_label = f"cuda:{device_index}"
            device_argument = f"cuda:{device_index}"
        raise NotImplementedError(
            "balanced PrimsTS MLA has no calibrated cost model for "
            f"{device_label}: name={identity[0]!r}, "
            f"compute_capability={identity[1]}, "
            f"multi_processor_count={identity[2]}. Run "
            "`python benchmarks/bench_prims_ts_balanced_mla_cost_model.py "
            f"--device {device_argument}` on this device, then add the measured "
            "cost models and exact hardware identity to "
            "flashinfer/attention/prims_ts/_balanced_scheduler.py. Balanced "
            "execution will not use an uncalibrated fallback."
        ) from error


def select_b200_balanced_cost_model(
    seq_lens: Sequence[int],
    *,
    kernel_family: str,
    dtype_name: str,
    num_partitions: int,
    k_tile_tokens: int = 128,
) -> tuple[str, BalancedCostModel]:
    """Return the calibrated B200 bucket and scheduler model."""

    family_aliases = {
        "1cta": "1cta",
        "2cta": "2cta",
        "throughput_latency_1cta": "1cta",
        "throughput_2cta": "2cta",
    }
    dtype_aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp8": "fp8",
        "e4m3": "fp8",
        "float8_e4m3fn": "fp8",
    }
    try:
        family = family_aliases[kernel_family]
    except KeyError as error:
        raise ValueError(
            f"unsupported balanced kernel family {kernel_family!r}"
        ) from error
    try:
        dtype = dtype_aliases[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported balanced input dtype {dtype_name!r}") from error
    bucket = balanced_cost_bucket(
        seq_lens,
        num_partitions=num_partitions,
        k_tile_tokens=k_tile_tokens,
    )
    if bucket == "legacy":
        return bucket, B200_BF16_2CTA_COST
    return bucket, B200_BALANCED_COST_MODELS[(family, dtype, bucket)]


def select_calibrated_balanced_cost_model(
    calibration: BalancedCostModelCalibration,
    seq_lens: Sequence[int],
    *,
    kernel_family: str,
    dtype_name: str,
    num_partitions: int,
    k_tile_tokens: int = 128,
) -> tuple[str, BalancedCostModel]:
    """Select a workload bucket from an explicitly identified calibration."""

    if calibration.model_id == B200_BALANCED_COST_MODEL_ID:
        return select_b200_balanced_cost_model(
            seq_lens,
            kernel_family=kernel_family,
            dtype_name=dtype_name,
            num_partitions=num_partitions,
            k_tile_tokens=k_tile_tokens,
        )
    raise RuntimeError(
        f"balanced cost-model calibration {calibration.model_id!r} has no selector"
    )
