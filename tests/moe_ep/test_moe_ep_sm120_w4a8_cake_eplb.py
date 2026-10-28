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

"""EPLB (redundant experts) with the Cake SM120 hybrid MegaMoE (multi-rank).

With expert-parallel load balancing the router maps every logical expert choice to one
of that expert's physical replicas; the kernel only sees physical expert ids and a
physical expert count (logical experts plus redundant replicas, divisible by the EP
world size), and each rank holds the weights of its physical experts.  A replica's
weights are a byte copy of its logical expert's, and the kernel computes every
(token, slot) partial independently and combines them in slot order, so an EPLB run
must reproduce the run on the logical routing bit for bit.

The placement here is deterministic: every rank keeps its block of logical experts
and receives ``redundant // world_size`` replicas of experts homed on the next rank;
tokens alternate between an expert's copies by ``(row + slot)``.  Run in its own
torchrun session (it creates two layers in turn):

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_w4a8_cake_eplb.py \
        -m "gpu_4 and arch_sm120" -v
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.moe_ep.test_moe_ep_sm120_w4a8_cake_mega import (
    V4FLASH,
    Oracle,
    Shape,
    check_close,
    init_distributed,
    local_weights,
    make_inputs,
    make_layer,
    table_spec,
)

REDUNDANT = int(os.environ.get("CAKE_SM120_MEGAMOE_EPLB_REDUNDANT", "32"))
ROWS = (40, 512, 2048)


def placement(world_size: int, logical: int, redundant: int):
    """``(physical_to_logical, logical_to_physical_copies)`` of the test placement."""

    if logical % world_size or redundant % world_size:
        raise ValueError("logical and redundant experts must divide the world size")
    home_local = logical // world_size
    per_rank = redundant // world_size
    physical_to_logical: list[int] = []
    for rank in range(world_size):
        physical_to_logical += range(rank * home_local, (rank + 1) * home_local)
        neighbor = (rank + 1) % world_size
        physical_to_logical += [
            neighbor * home_local + (j % home_local) for j in range(per_rank)
        ]
    copies: list[list[int]] = [[] for _ in range(logical)]
    for physical, expert in enumerate(physical_to_logical):
        copies[expert].append(physical)
    return physical_to_logical, copies


def to_physical(topk_ids: torch.Tensor, copies: list[list[int]]) -> torch.Tensor:
    """Route each (row, slot) to copy ``(row + slot) % n_copies`` of its logical expert."""

    width = max(len(c) for c in copies)
    table = torch.full((len(copies), width), -1, dtype=torch.int64)
    counts = torch.tensor([len(c) for c in copies], dtype=torch.int64)
    for expert, c in enumerate(copies):
        table[expert, : len(c)] = torch.tensor(c)
    table, counts = table.to(topk_ids.device), counts.to(topk_ids.device)
    rows, top_k = topk_ids.shape
    choice = (
        torch.arange(rows, device=topk_ids.device).unsqueeze(1)
        + torch.arange(top_k, device=topk_ids.device).unsqueeze(0)
    ) % counts[topk_ids.clamp_min(0)]
    physical = table[topk_ids.clamp_min(0), choice]
    return torch.where(topk_ids >= 0, physical, topk_ids)


def _run(rank, world_size, weights, variant, shape, cases):
    import torch.distributed as dist

    layer = make_layer(
        rank, world_size, weights, variant=variant, table=table_spec(), shape=shape
    )
    layer.warmup(cases[0])
    outputs = [layer(t).clone() for t in cases]
    torch.cuda.synchronize()
    assert layer._kernel.protocol_error(layer._workspace) == 0
    layer.destroy()
    dist.barrier()
    return outputs


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "variant",
    [
        v.strip()
        for v in os.environ.get(
            "CAKE_SM120_MEGAMOE_VARIANTS", "noshared,fp8shared"
        ).split(",")
        if v.strip()
    ],
)
def test_eplb_physical_experts_match_logical_bitwise(variant) -> None:
    from flashinfer.moe_ep import MoEEpTensors

    rank, world_size = init_distributed()
    if REDUNDANT % world_size:
        pytest.skip(f"{REDUNDANT} redundant experts on {world_size} ranks")
    shared = variant == "fp8shared"
    logical_shape = V4FLASH
    physical_shape = Shape(
        "v4flash_eplb",
        logical_shape.hidden,
        logical_shape.intermediate,
        logical_shape.experts + REDUNDANT,
        logical_shape.top_k,
    )
    physical_to_logical, copies = placement(
        world_size, logical_shape.experts, REDUNDANT
    )
    logical_cases = [
        make_inputs(rows, seed=7000 + 31 * index + rank, shape=logical_shape)
        for index, rows in enumerate(ROWS)
    ]
    physical_cases = [
        MoEEpTensors(
            hidden_states=t.hidden_states,
            topk_ids=to_physical(t.topk_ids, copies),
            topk_weights=t.topk_weights,
        )
        for t in logical_cases
    ]
    # The replica slots (the last ``REDUNDANT // world_size`` physical experts of every
    # rank) actually receive routes.
    local = physical_shape.experts // world_size
    home = logical_shape.experts // world_size
    used = torch.unique(torch.cat([t.topk_ids.reshape(-1) for t in physical_cases]))
    used = used[used >= 0]
    assert REDUNDANT == 0 or bool(((used % local) >= home).any())

    eplb_weights = local_weights(
        rank,
        world_size,
        seed=20261008,
        shared=shared,
        shape=physical_shape,
        physical_to_logical=physical_to_logical,
    )
    eplb_outputs = _run(
        rank, world_size, eplb_weights, variant, physical_shape, physical_cases
    )
    oracle = Oracle(eplb_weights, shared)
    for tensors, out in zip(physical_cases, eplb_outputs, strict=True):
        check_close(out, oracle(tensors))
    del oracle, eplb_weights

    logical_weights = local_weights(
        rank, world_size, seed=20261008, shared=shared, shape=logical_shape
    )
    logical_outputs = _run(
        rank, world_size, logical_weights, variant, logical_shape, logical_cases
    )
    for index, (a, b) in enumerate(zip(eplb_outputs, logical_outputs, strict=True)):
        assert torch.equal(a, b), f"EPLB output differs from logical at case {index}"
