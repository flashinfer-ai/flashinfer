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

"""Reference work planner and calibrated models for balanced PrimsTS MLA decode.

The Python planner deliberately has no CUDA or CUTLASS dependency. It mirrors
the native production planner for parity tests, offline analysis, and cost-model
fitting; production replay planning emits descriptors through the native
implementation.

The descriptor ABI uses a compact per-request split prefix so its partial
workspace is bounded independently of the producer descriptor capacity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import heapq
from types import MappingProxyType
from typing import Sequence


_INT32_MAX = 2**31 - 1


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
# Keep family and dtype names independent of CuTe types so the host planner
# remains importable in CPU-only tests and tools.
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


@dataclass(frozen=True)
class BalancedWorkDescriptor:
    """One compact main-kernel work item, expressed in K tiles."""

    request_idx: int
    local_split_idx: int
    k_tile_start: int
    k_tile_count: int


@dataclass(frozen=True)
class BalancedSchedule:
    """Immutable host result ready to stage into a replay-stable plan."""

    descriptors: tuple[BalancedWorkDescriptor, ...]
    partition_offsets: tuple[int, ...]
    split_counts: tuple[int, ...]
    split_begins: tuple[int, ...]
    partition_costs: tuple[int, ...]
    target_piece_tiles: int

    @property
    def num_partitions(self) -> int:
        return len(self.partition_offsets) - 1


@dataclass
class _Chunk:
    request_idx: int
    k_tile_start: int
    k_tile_end: int
    local_split_idx: int = 0
    num_pieces: int = 1


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _piece_cost(
    k_tiles: int,
    *,
    is_split: bool,
    num_pieces: int,
    cost: BalancedCostModel,
) -> int:
    value = cost.fixed_piece_cost + k_tiles * cost.cost_per_k_tile
    if is_split:
        value += cost.split_piece_cost
        value += cost.split_fixed_cost // max(num_pieces, 1)
    return value


def _reducer_cost(max_pieces: int, cost: BalancedCostModel) -> int:
    if max_pieces < 2:
        return 0
    return cost.reducer_fixed_cost + cost.reducer_piece_cost * max_pieces


def _compute_target_piece_tiles(
    *,
    total_tiles: int,
    num_partitions: int,
    max_tiles: int,
    cost: BalancedCostModel,
) -> tuple[int, int, int, int]:
    average = _ceil_div(total_tiles, num_partitions)
    if cost.cost_per_k_tile > 0:
        break_even = (
            _ceil_div(
                cost.split_fixed_cost + 2 * cost.split_piece_cost,
                cost.cost_per_k_tile,
            )
            + 3
        )
    else:
        break_even = 1_000_000_000
    positive_tile_cost = max(cost.cost_per_k_tile, 1)
    efficiency_floor = (
        _ceil_div(
            cost.fixed_piece_cost + cost.split_piece_cost + cost.split_fixed_cost // 2,
            positive_tile_cost,
        )
        + 1
    )
    # Keep every request at <= 127 pieces and, in practice, <= 126 pieces at
    # the chosen target.  The latter preserves the old N + P capacity bound.
    piece_count_floor = _ceil_div(max_tiles, 126)
    target = max(
        average + break_even,
        efficiency_floor,
        piece_count_floor,
        1,
    )
    if target > _INT32_MAX:
        raise OverflowError(
            "balanced scheduler target piece tiles exceeds int32 capacity"
        )
    return target, break_even, efficiency_floor, piece_count_floor


def _cost_search_target(
    k_tiles: Sequence[int],
    *,
    num_partitions: int,
    base_target: int,
    efficiency_floor: int,
    piece_count_floor: int,
    cost: BalancedCostModel,
) -> int:
    active_count = sum(value > 0 for value in k_tiles)
    max_tiles = max(k_tiles, default=0)
    candidates = {base_target}
    for pieces in range(1, min(num_partitions, 127) + 1):
        floor = 1 if pieces * active_count <= num_partitions else efficiency_floor
        candidates.add(
            max(
                _ceil_div(max_tiles, pieces),
                floor,
                piece_count_floor,
                1,
            )
        )

    best_cost: int | None = None
    best_target = base_target
    descriptor_capacity = active_count + num_partitions
    uses_additive_reducer = cost.reducer_fixed_cost != 0 or cost.reducer_piece_cost != 0
    previous_piece_counts: list[int] | None = None
    for target in sorted(candidates, reverse=True):
        piece_counts = [
            min(_ceil_div(tile_count, target), 127, num_partitions)
            if tile_count > target
            else int(tile_count > 0)
            for tile_count in k_tiles
        ]
        if sum(piece_counts) > descriptor_capacity:
            break
        if piece_counts == previous_piece_counts:
            continue
        previous_piece_counts = piece_counts
        piece_costs = []
        for tile_count, num_pieces in zip(k_tiles, piece_counts, strict=True):
            if num_pieces == 1:
                piece_costs.append(
                    _piece_cost(
                        tile_count,
                        is_split=False,
                        num_pieces=1,
                        cost=cost,
                    )
                )
            elif num_pieces > 1:
                piece_base, piece_remainder = divmod(tile_count, num_pieces)
                piece_costs.extend(
                    _piece_cost(
                        piece_base + int(piece_idx < piece_remainder),
                        is_split=not uses_additive_reducer,
                        num_pieces=num_pieces,
                        cost=cost,
                    )
                    for piece_idx in range(num_pieces)
                )
        optimistic = max(
            _ceil_div(sum(piece_costs), num_partitions),
            max(piece_costs, default=0),
        )
        if uses_additive_reducer:
            optimistic += _reducer_cost(max(piece_counts, default=1), cost)
        if best_cost is not None and optimistic >= best_cost:
            continue
        partition_costs = [0] * num_partitions
        _place_for_target(
            None,
            partition_costs,
            k_tiles,
            target,
            cost,
            charge_split_penalty=not uses_additive_reducer,
        )
        max_pieces = max(piece_counts, default=1)
        predicted = max(partition_costs)
        if uses_additive_reducer:
            predicted += _reducer_cost(max_pieces, cost)
        if (
            best_cost is None
            or predicted < best_cost
            or (predicted == best_cost and target > best_target)
        ):
            best_cost = predicted
            best_target = target
    return best_target


def _lpt_assign(
    partitions: list[list[_Chunk]] | None,
    partition_costs: list[int],
    request_indices: Sequence[int],
    k_tiles: Sequence[int],
    cost: BalancedCostModel,
) -> None:
    request_indices = sorted(
        request_indices,
        key=lambda request_idx: (
            -_piece_cost(
                k_tiles[request_idx],
                is_split=False,
                num_pieces=1,
                cost=cost,
            ),
            request_idx,
        ),
    )
    heap = [
        (value, partition_idx) for partition_idx, value in enumerate(partition_costs)
    ]
    heapq.heapify(heap)
    for request_idx in request_indices:
        running, partition_idx = heapq.heappop(heap)
        tile_count = k_tiles[request_idx]
        running += _piece_cost(
            tile_count,
            is_split=False,
            num_pieces=1,
            cost=cost,
        )
        partition_costs[partition_idx] = running
        if partitions is not None:
            partitions[partition_idx].append(_Chunk(request_idx, 0, tile_count))
        heapq.heappush(heap, (running, partition_idx))


def _equal_split_assign(
    partitions: list[list[_Chunk]] | None,
    partition_costs: list[int],
    request_indices: Sequence[int],
    k_tiles: Sequence[int],
    target_piece_tiles: int,
    cost: BalancedCostModel,
    *,
    charge_split_penalty: bool,
) -> None:
    request_indices = sorted(
        request_indices,
        key=lambda request_idx: (-k_tiles[request_idx], request_idx),
    )
    for request_idx in request_indices:
        tile_count = k_tiles[request_idx]
        num_pieces = min(
            _ceil_div(tile_count, target_piece_tiles),
            127,
            len(partition_costs),
        )
        selected = sorted(
            range(len(partition_costs)),
            key=lambda partition_idx: (
                partition_costs[partition_idx],
                partition_idx,
            ),
        )[:num_pieces]
        # Match the shared C++ emitter: select by load, then distribute the
        # remainder over the first physical partitions for deterministic,
        # truly equal piece sizes.
        selected.sort()
        piece_base, piece_remainder = divmod(tile_count, num_pieces)
        tile_start = 0
        for local_split_idx, partition_idx in enumerate(selected):
            piece_size = piece_base + int(local_split_idx < piece_remainder)
            partition_costs[partition_idx] += _piece_cost(
                piece_size,
                is_split=charge_split_penalty and num_pieces > 1,
                num_pieces=num_pieces,
                cost=cost,
            )
            if partitions is not None:
                partitions[partition_idx].append(
                    _Chunk(
                        request_idx,
                        tile_start,
                        tile_start + piece_size,
                        local_split_idx=local_split_idx,
                        num_pieces=num_pieces,
                    )
                )
            tile_start += piece_size


def _place_for_target(
    partitions: list[list[_Chunk]] | None,
    partition_costs: list[int],
    k_tiles: Sequence[int],
    target_piece_tiles: int,
    cost: BalancedCostModel,
    *,
    charge_split_penalty: bool,
) -> None:
    short_requests = [
        request_idx
        for request_idx, tile_count in enumerate(k_tiles)
        if 0 < tile_count <= target_piece_tiles
    ]
    long_requests = [
        request_idx
        for request_idx, tile_count in enumerate(k_tiles)
        if tile_count > target_piece_tiles
    ]
    _lpt_assign(
        partitions,
        partition_costs,
        short_requests,
        k_tiles,
        cost,
    )
    _equal_split_assign(
        partitions,
        partition_costs,
        long_requests,
        k_tiles,
        target_piece_tiles,
        cost,
        charge_split_penalty=charge_split_penalty,
    )


def build_balanced_schedule(
    seq_lens: Sequence[int],
    *,
    num_partitions: int,
    k_tile_tokens: int = 128,
    cost: BalancedCostModel = B200_BF16_2CTA_COST,
) -> BalancedSchedule:
    """Build a compact, partition-grouped balanced MLA schedule.

    Args:
        seq_lens: Runtime KV lengths, in tokens, one per request.
        num_partitions: Fixed physical cluster partitions in the launch grid.
        k_tile_tokens: Number of KV tokens consumed by one kernel K tile.
        cost: Calibrated scheduling cost model.

    Returns:
        An immutable schedule.  Its descriptors are grouped by partition, and
        ``partition_offsets[p:p+2]`` selects partition ``p``'s work.
    """

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if k_tile_tokens <= 0:
        raise ValueError("k_tile_tokens must be positive")
    if any(int(seq_len) < 0 for seq_len in seq_lens):
        raise ValueError("seq_lens must be non-negative")

    k_tiles = [
        _ceil_div(int(seq_len), k_tile_tokens) if int(seq_len) > 0 else 0
        for seq_len in seq_lens
    ]
    active = [idx for idx, value in enumerate(k_tiles) if value > 0]
    if not active:
        return BalancedSchedule(
            descriptors=(),
            partition_offsets=(0,) * (num_partitions + 1),
            split_counts=(0,) * len(k_tiles),
            split_begins=(0,) * len(k_tiles),
            partition_costs=(0,) * num_partitions,
            target_piece_tiles=0,
        )

    total_tiles = sum(k_tiles)
    max_tiles = max(k_tiles)
    min_active_tiles = min(k_tiles[idx] for idx in active)
    target, break_even, efficiency_floor, piece_count_floor = (
        _compute_target_piece_tiles(
            total_tiles=total_tiles,
            num_partitions=num_partitions,
            max_tiles=max_tiles,
            cost=cost,
        )
    )
    uses_cost_search = len(active) <= 2 * num_partitions
    if uses_cost_search:
        target = _cost_search_target(
            k_tiles,
            num_partitions=num_partitions,
            base_target=target,
            efficiency_floor=efficiency_floor,
            piece_count_floor=piece_count_floor,
            cost=cost,
        )

    partitions: list[list[_Chunk]] = [[] for _ in range(num_partitions)]
    partition_costs = [0] * num_partitions
    spread_is_uniform = max_tiles - min_active_tiles <= break_even
    enough_requests = 2 * len(active) >= num_partitions
    if spread_is_uniform and enough_requests and not uses_cost_search:
        _lpt_assign(partitions, partition_costs, active, k_tiles, cost)
    else:
        _place_for_target(
            partitions,
            partition_costs,
            k_tiles,
            target,
            cost,
            charge_split_penalty=not (
                cost.reducer_fixed_cost != 0 or cost.reducer_piece_cost != 0
            ),
        )

    split_counts = [0] * len(k_tiles)
    for partition in partitions:
        for chunk in partition:
            split_counts[chunk.request_idx] += 1
    split_begins = [0] * len(k_tiles)
    split_cursor = 0
    for request_idx, split_count in enumerate(split_counts):
        split_begins[request_idx] = split_cursor
        if split_count > 1:
            split_cursor += split_count

    descriptors: list[BalancedWorkDescriptor] = []
    partition_offsets: list[int] = []
    for partition in partitions:
        partition_offsets.append(len(descriptors))
        partition.sort(key=lambda chunk: (chunk.request_idx, chunk.k_tile_start))
        descriptors.extend(
            BalancedWorkDescriptor(
                request_idx=chunk.request_idx,
                local_split_idx=chunk.local_split_idx,
                k_tile_start=chunk.k_tile_start,
                k_tile_count=chunk.k_tile_end - chunk.k_tile_start,
            )
            for chunk in partition
        )
    partition_offsets.append(len(descriptors))

    capacity = len(active) + num_partitions
    if len(descriptors) > capacity:
        raise RuntimeError(
            f"balanced schedule needs {len(descriptors)} descriptors, "
            f"but the active + P limit is {capacity}"
        )
    partial_capacity = min(len(k_tiles) + num_partitions, 2 * num_partitions)
    if split_cursor > partial_capacity:
        raise RuntimeError(
            f"balanced schedule needs {split_cursor} split partials, "
            f"but the compact partial capacity is {partial_capacity}"
        )
    combine_count = sum(split_count > 1 for split_count in split_counts)
    reducer_capacity = min(len(k_tiles), num_partitions)
    if combine_count > reducer_capacity:
        raise RuntimeError(
            f"balanced schedule needs {combine_count} combine descriptors, "
            f"but the compact reducer capacity is {reducer_capacity}"
        )
    return BalancedSchedule(
        descriptors=tuple(descriptors),
        partition_offsets=tuple(partition_offsets),
        split_counts=tuple(split_counts),
        split_begins=tuple(split_begins),
        partition_costs=tuple(partition_costs),
        target_piece_tiles=target,
    )
