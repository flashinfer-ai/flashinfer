# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analytical route selection for recurrent KDA training on Blackwell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


_HEAD_DIM = 128
_C16_CHUNK = 16
_C32_CHUNK = 32

_TemplateName = Literal[
    "checkpoint_recurrent_c16",
    "tensor_tape_c32",
    "row_warp_checkpoint",
]
_RouteTag = Literal[
    "grouped_hybrid_c16_c32",
    "grouped_c16",
    "grouped_c32",
    "grouped_row_split",
    "c16",
    "c32",
    "row_split",
]
_RouteFamily = Literal["c16", "c32", "row_split"]


@dataclass(frozen=True)
class _TrainingRouteSpec:
    """Resolved strict execution route and its analytical template choice."""

    tag: _RouteTag
    selected_template: _TemplateName
    split_work_items: bool = False

    @property
    def family(self) -> _RouteFamily:
        if self.tag in ("c16", "grouped_c16", "grouped_hybrid_c16_c32"):
            return "c16"
        if self.tag in ("c32", "grouped_c32"):
            return "c32"
        return "row_split"

    @property
    def grouped(self) -> bool:
        return self.tag.startswith("grouped_")

    @property
    def uses_parameter_context(self) -> bool:
        return self.tag == "grouped_hybrid_c16_c32"


@dataclass(frozen=True)
class _Problem:
    seq_lens: tuple[int, ...]
    num_qk_heads: int
    num_v_heads: int
    resident_sms: int

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def sequence_heads(self) -> int:
        return len(self.seq_lens) * self.num_v_heads

    @property
    def grouped(self) -> bool:
        return self.num_qk_heads != self.num_v_heads


@dataclass(frozen=True)
class _WorkGroup:
    items: int
    chunks_per_item: int


def _persistent_grid_cost(
    groups: tuple[_WorkGroup, ...],
    *,
    resident_ctas: int,
    item_setup: float,
    chunk_compute: float,
    chunk_memory: float,
) -> float:
    service = max(chunk_compute, chunk_memory)
    total_items = 0
    total_work = 0.0
    largest_item = 0.0
    for group in groups:
        item_work = item_setup + group.chunks_per_item * service
        total_items += group.items
        total_work += group.items * item_work
        largest_item = max(largest_item, item_work)
    grid_waves = total_items / resident_ctas
    tail_fraction = math.ceil(grid_waves) - grid_waves
    return total_work / resident_ctas + tail_fraction * largest_item


def _unsplit_groups(problem: _Problem, *, chunk_tokens: int) -> tuple[_WorkGroup, ...]:
    return tuple(
        _WorkGroup(
            items=problem.num_v_heads,
            chunks_per_item=(length + chunk_tokens - 1) // chunk_tokens,
        )
        for length in problem.seq_lens
    )


def _c16_split_groups(problem: _Problem) -> tuple[tuple[_WorkGroup, ...], int]:
    chunk_counts = tuple(length // _C16_CHUNK for length in problem.seq_lens)
    total_chunk_heads = sum(chunk_counts) * problem.num_v_heads
    target_chunks = max(1, math.ceil(total_chunk_heads / problem.resident_sms))
    groups: list[_WorkGroup] = []
    boundaries = 0
    for chunks in chunk_counts:
        pieces = min(chunks, max(1, math.ceil(chunks / target_chunks)))
        span = math.ceil(chunks / pieces)
        groups.append(
            _WorkGroup(
                items=problem.num_v_heads * pieces,
                chunks_per_item=span,
            )
        )
        boundaries += problem.num_v_heads * (pieces - 1)
    return tuple(groups), boundaries


def _grouped_adapter_cost(problem: _Problem, *, expands_qk: bool) -> float:
    if not problem.grouped:
        return 0.0
    token_heads_per_sm = (
        problem.total_tokens * problem.num_v_heads / problem.resident_sms
    )
    return token_heads_per_sm * (0.050 if expands_qk else 0.020)


def _estimate_c16(problem: _Problem) -> tuple[float, bool]:
    unsplit = _persistent_grid_cost(
        _unsplit_groups(problem, chunk_tokens=_C16_CHUNK),
        resident_ctas=problem.resident_sms,
        item_setup=16.0,
        chunk_compute=1.0,
        chunk_memory=0.75,
    )
    split_groups, boundaries = _c16_split_groups(problem)
    split = _persistent_grid_cost(
        split_groups,
        resident_ctas=problem.resident_sms,
        item_setup=16.0,
        chunk_compute=1.0,
        chunk_memory=0.75,
    )
    split += boundaries * 8.0 / problem.resident_sms
    use_split = split < unsplit
    max_chunks = max(length // _C16_CHUNK for length in problem.seq_lens)
    dag_fill = 17.0 + min(12.0, 3.0 * max_chunks / 32.0)
    return (
        (split if use_split else unsplit)
        + dag_fill
        + _grouped_adapter_cost(problem, expands_qk=False),
        use_split,
    )


def _c32_groups(problem: _Problem) -> tuple[tuple[_WorkGroup, ...], int]:
    chunk_counts = tuple(math.ceil(length / _C32_CHUNK) for length in problem.seq_lens)
    sequence_heads = problem.sequence_heads
    split_multiplier = problem.resident_sms // sequence_heads
    forward_two_wave_fill = (
        problem.resident_sms // 2 < sequence_heads < problem.resident_sms
        and max(chunk_counts) >= 96
    )
    split = (split_multiplier >= 2 and max(chunk_counts) >= 64) or forward_two_wave_fill
    groups: list[_WorkGroup] = []
    boundaries = 0
    for chunks in chunk_counts:
        if not split:
            pieces = 1
        elif forward_two_wave_fill:
            pieces = min(
                (2 * problem.resident_sms) // sequence_heads,
                max(1, chunks // 32),
            )
        else:
            pieces = min(split_multiplier, max(1, chunks // 32))
        groups.append(
            _WorkGroup(
                items=problem.num_v_heads * pieces,
                chunks_per_item=math.ceil(chunks / pieces),
            )
        )
        if not forward_two_wave_fill:
            boundaries += problem.num_v_heads * (pieces - 1)
    return tuple(groups), boundaries


def _estimate_c32(problem: _Problem) -> float:
    groups, boundaries = _c32_groups(problem)
    item_setup = 90.0 if problem.num_v_heads <= 64 else 22.0
    recurrence = _persistent_grid_cost(
        groups,
        resident_ctas=problem.resident_sms,
        item_setup=item_setup,
        chunk_compute=1.75,
        chunk_memory=2.0,
    )
    recurrence += boundaries * 18.0 * 2.0 / problem.resident_sms
    return 8.0 + recurrence + _grouped_adapter_cost(problem, expands_qk=True)


def _estimate_row(problem: _Problem) -> float:
    recurrence = (
        problem.total_tokens * problem.num_v_heads * 3.4 / (4 * problem.resident_sms)
    )
    return 4.0 + recurrence + _grouped_adapter_cost(problem, expands_qk=True)


@lru_cache(maxsize=256)
def _select_training_route_cached(problem: _Problem) -> _TrainingRouteSpec:
    candidates: list[tuple[float, _TemplateName, bool]] = []
    if all(length % _C16_CHUNK == 0 for length in problem.seq_lens):
        c16_cost, split = _estimate_c16(problem)
        candidates.append((c16_cost, "checkpoint_recurrent_c16", split))
    candidates.append((_estimate_c32(problem), "tensor_tape_c32", False))
    candidates.append((_estimate_row(problem), "row_warp_checkpoint", False))
    _, template, split = min(candidates, key=lambda candidate: candidate[0])

    grouped = problem.grouped
    if template == "checkpoint_recurrent_c16":
        # The low-head C16 schedule is strict for token/state gradients, while
        # C32 supplies its two strict gate-parameter gradients.  Materialize
        # both forward contexts before returning from the paired public API.
        if problem.num_v_heads <= 8:
            if grouped:
                return _TrainingRouteSpec("grouped_hybrid_c16_c32", template, split)
            return _TrainingRouteSpec("c32", template)
        return _TrainingRouteSpec("grouped_c16" if grouped else "c16", template, split)
    if template == "tensor_tape_c32":
        return _TrainingRouteSpec("grouped_c32" if grouped else "c32", template)
    return _TrainingRouteSpec("grouped_row_split" if grouped else "row_split", template)


def _select_training_route(
    seq_lens: tuple[int, ...],
    num_qk_heads: int,
    num_v_heads: int,
    *,
    resident_sms: int = 152,
) -> _TrainingRouteSpec:
    """Select the strict public route from runtime shape and device capacity."""

    if (
        not seq_lens
        or min(seq_lens) <= 0
        or num_qk_heads <= 0
        or num_v_heads <= 0
        or num_v_heads % num_qk_heads
        or resident_sms <= 0
    ):
        raise ValueError("invalid recurrent KDA training problem")
    return _select_training_route_cached(
        _Problem(seq_lens, num_qk_heads, num_v_heads, resident_sms)
    )


__all__ = ["_TrainingRouteSpec", "_select_training_route"]
