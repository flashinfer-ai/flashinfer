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
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import cast

import torch

from .jit.flash_kda_evolution import (
    FLASH_KDA_EVOLUTION_VARIANTS,
    FlashKDAEvolutionVariant,
    load_flash_kda_evolution_module,
)
from .kda_prefill import _select_flash_kda_prefill_target

_HEAD_DIM = 128
_PERSISTENT_SCALAR_CTAS = 152
_DESCRIPTOR_STORAGE_BYTES = 6 * 128


def _build_persistent_scalar_schedule(
    sequence_lengths: tuple[int, ...], num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    bins: list[tuple[int, int, list[tuple[int, int]]]] = [
        (0, cta, []) for cta in range(_PERSISTENT_SCALAR_CTAS)
    ]
    heapq.heapify(bins)
    ordered_sequences = sorted(
        range(len(sequence_lengths)),
        key=lambda sequence: (sequence_lengths[sequence] + 31) // 32,
        reverse=True,
    )
    for sequence in ordered_sequences:
        chunks = (sequence_lengths[sequence] + 31) // 32
        for head in range(num_heads):
            load, cta, tasks = heapq.heappop(bins)
            tasks.append((sequence * num_heads + head, chunks))
            heapq.heappush(bins, (load + chunks, cta, tasks))

    bins.sort(key=lambda item: item[1])
    counts = [load for load, _cta, _tasks in bins]
    stride = max(counts)
    schedule = [0] * (_PERSISTENT_SCALAR_CTAS * stride)
    for _load, cta, tasks in bins:
        slot = 0
        for task, chunks in tasks:
            for local_chunk in range(chunks):
                schedule[cta * stride + slot] = (
                    task | (local_chunk << 10) | (chunks << 18)
                )
                slot += 1
    return (
        torch.tensor(schedule, dtype=torch.int32, device="cuda"),
        torch.tensor(counts, dtype=torch.int32, device="cuda"),
        stride,
    )


@dataclass(frozen=True)
class _Route:
    variant: FlashKDAEvolutionVariant
    grid_x: int
    tile_schedule: torch.Tensor
    tile_schedule_counts: torch.Tensor


def _route(
    sequence_lengths: tuple[int, ...], num_heads: int, fixed_layout: bool
) -> _Route:
    num_sequences = len(sequence_lengths)
    full_chunks = all(length % 32 == 0 for length in sequence_lengths)
    use_m64 = fixed_layout and num_sequences == 1 and num_heads == 64
    use_vtile = not use_m64 and (fixed_layout or len(set(sequence_lengths)) == 1)
    dummy = torch.empty((1,), dtype=torch.int32, device="cuda")

    if use_m64:
        variant = f"m64_f{int(full_chunks)}_t{sequence_lengths[0]}_h{num_heads}"
        grid_x = 2 * num_sequences * num_heads
        tile_schedule = dummy
        tile_schedule_counts = dummy
    elif use_vtile:
        persistent_tasks = 1
        persistent_stride = num_sequences * num_heads
        if (
            full_chunks
            and num_sequences == 8
            and sequence_lengths[0] == 1024
            and num_heads in (64, 96)
        ):
            persistent_stride = 128
            persistent_tasks = num_sequences * num_heads // persistent_stride
        variant = (
            f"vtile_f{int(full_chunks)}_t{sequence_lengths[0]}_h{num_heads}"
            f"_p{persistent_tasks}_s{persistent_stride}"
        )
        grid_x = persistent_stride
        tile_schedule = dummy
        tile_schedule_counts = dummy
    else:
        use_persistent_scalar = (
            num_heads == 64
            and num_sequences * num_heads >= 2 * _PERSISTENT_SCALAR_CTAS
            and num_sequences * num_heads < 1024
            and max((length + 31) // 32 for length in sequence_lengths) < 256
        )
        if use_persistent_scalar:
            tile_schedule, tile_schedule_counts, stride = (
                _build_persistent_scalar_schedule(sequence_lengths, num_heads)
            )
            grid_x = _PERSISTENT_SCALAR_CTAS
        else:
            stride = 1
            grid_x = num_sequences * num_heads
            tile_schedule = dummy
            tile_schedule_counts = dummy
        variant = f"m128_h{num_heads}_p{int(use_persistent_scalar)}_s{stride}"

    if variant not in FLASH_KDA_EVOLUTION_VARIANTS:
        raise ValueError(
            "no generated Blackwell specialization for "
            f"heads={num_heads}, sequence_lengths={sequence_lengths}, "
            f"fixed_layout={fixed_layout}"
        )
    return _Route(
        variant=cast(FlashKDAEvolutionVariant, variant),
        grid_x=grid_x,
        tile_schedule=tile_schedule,
        tile_schedule_counts=tile_schedule_counts,
    )


class PreparedFlashKDAEvolution:
    """Prepared launch of one generated Blackwell recurrent-KDA specialization."""

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        output: torch.Tensor,
        final_state: torch.Tensor,
        *,
        scale: float,
        lower_bound: float,
        cu_seqlens: torch.Tensor | None = None,
    ) -> None:
        if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
            raise ValueError("q, k, v, and g must have identical shapes")
        if q.shape != output.shape or q.ndim != 4 or q.shape[-1] != _HEAD_DIM:
            raise ValueError("q and output must have shape [B, T, H, 128]")
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, g, beta)):
            raise TypeError("q, k, v, g, and beta must use torch.bfloat16")
        batch, tokens_per_batch, num_heads, _head_dim = q.shape
        total_tokens = batch * tokens_per_batch
        fixed_layout = cu_seqlens is None
        if fixed_layout:
            offsets = tuple(range(0, total_tokens + 1, tokens_per_batch))
            cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=q.device)
        else:
            offsets = tuple(int(value) for value in cu_seqlens.tolist())
        sequence_lengths = tuple(
            end - start for start, end in zip(offsets, offsets[1:], strict=False)
        )
        route = _route(sequence_lengths, num_heads, fixed_layout)
        seq_order = torch.tensor(
            sorted(
                range(len(sequence_lengths)),
                key=sequence_lengths.__getitem__,
                reverse=True,
            ),
            dtype=torch.int32,
            device=q.device,
        )
        beta_flat = beta.reshape(total_tokens, num_heads)
        padded_heads = ((num_heads + 7) // 8) * 8
        if padded_heads == num_heads and total_tokens >= 32:
            beta_tma = beta_flat
        else:
            beta_tma = torch.empty(
                (max(total_tokens, 32), padded_heads),
                dtype=beta.dtype,
                device=beta.device,
            )

        target = _select_flash_kda_prefill_target(q.device)
        self.module = load_flash_kda_evolution_module(route.variant, target)
        self.variant = route.variant
        self.target = target
        self.grid_x = route.grid_x
        self._prepare_descriptors = True
        self._args = (
            q.reshape(total_tokens, num_heads, _HEAD_DIM),
            k.reshape(total_tokens, num_heads, _HEAD_DIM),
            v.reshape(total_tokens, num_heads, _HEAD_DIM),
            g.reshape(total_tokens, num_heads, _HEAD_DIM),
            beta_flat,
            beta_tma,
            A_log,
            dt_bias,
            cu_seqlens,
            seq_order,
            route.tile_schedule,
            route.tile_schedule_counts,
            initial_state,
            output.reshape(total_tokens, num_heads, _HEAD_DIM),
            final_state,
            torch.empty(
                (_DESCRIPTOR_STORAGE_BYTES,),
                dtype=torch.uint8,
                device=q.device,
            ),
        )
        self._launch_scalars = (
            route.grid_x,
            num_heads,
            1,
            1,
            float(scale),
            float(lower_bound),
        )

    def launch(self) -> None:
        stream_ptr = int(torch.cuda.current_stream().cuda_stream)
        self.module.run(
            *self._args,
            int(self._prepare_descriptors),
            *self._launch_scalars,
            stream_ptr,
        )
        self._prepare_descriptors = False


def prepare_flash_kda_evolution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    *,
    scale: float,
    lower_bound: float = -5.0,
    cu_seqlens: torch.Tensor | None = None,
) -> PreparedFlashKDAEvolution:
    """Prepare one supported generated specialization for repeated launch."""
    return PreparedFlashKDAEvolution(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        output,
        final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["PreparedFlashKDAEvolution", "prepare_flash_kda_evolution"]
