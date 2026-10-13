# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Distributed public-API guards for source lengths; run with torchrun EP4."""

import dataclasses
import os

import pytest

pytest_plugins = ["tests.experimental._mok_bf16_test_utils"]
import torch
import torch.distributed as dist


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 16, 64),
    reason="Requires a 4-, 16- or 64-rank symmetric-memory group",
)
def test_source_capacity_and_context_ownership(mok_distributed_group):
    from flashinfer.experimental.cake_mok_bf16.workspace import MoKConfig
    from flashinfer.mok import prepare_mok_bf16

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    rank, ep = dist.get_rank(), dist.get_world_size()
    counts = [0, 1, 255, 256] * (ep // 4)
    n, h = counts[rank], 256
    k, le = (2, 4) if ep == 4 else (8, 256 // ep)
    f = prepare_mok_bf16(ep_size=ep, local_experts=le, topk=k)
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=2.0,
    )
    args = dict(device=device, num_local_tokens=n, hidden_size=h, topk=k)
    with pytest.raises(ValueError, match="nonnegative"):
        f.create_workspace(
            config,
            dist.group.WORLD,
            **dict(args, num_local_tokens=-1 if rank == 0 else n),
        )
    with pytest.raises(ValueError, match="agree"):
        f.create_workspace(
            config,
            dist.group.WORLD,
            **args,
            source_capacity=512 if rank == 0 else 1024,
        )
    with pytest.raises(ValueError, match="covering"):
        f.create_workspace(config, dist.group.WORLD, **args, source_capacity=200)
    workspace = f.create_workspace(
        config, dist.group.WORLD, **args, source_capacity=513
    )
    assert (
        workspace.source_capacity == 768
        and workspace.initial_source_counts == tuple(counts)
    )
    with pytest.raises(ValueError, match="exceeds source capacity"):
        f.build_schedule(
            workspace,
            config,
            torch.empty(769, k, dtype=torch.int64),
            num_local_experts=le,
        )
    with pytest.raises(ValueError, match="Expert IDs"):
        f.build_schedule(
            workspace,
            config,
            torch.empty(n, k, device=device, dtype=torch.float32),
            num_local_experts=le,
        )
    row = torch.arange(n, device=device)
    ids = torch.stack([(row + slot) % (ep * le) for slot in range(k)], -1)
    x = torch.zeros(n, h, device=device, dtype=torch.bfloat16)
    dy, scores = torch.zeros_like(x), torch.ones(n, k, device=device)
    shared = [
        torch.ones(h, h, device=device, dtype=torch.bfloat16) * 0.01 for _ in range(3)
    ]
    routed = [
        torch.ones(le, h, h, device=device, dtype=torch.bfloat16) * 0.01
        for _ in range(3)
    ]
    schedule = f.build_schedule(workspace, config, ids, num_local_experts=le)
    with pytest.raises(ValueError, match="belong"):
        f.forward(
            config,
            dataclasses.replace(workspace),
            schedule,
            x,
            scores,
            *shared,
            *routed,
        )
    with pytest.raises(ValueError, match="shape"):
        f.forward(
            config,
            workspace,
            schedule,
            torch.empty(n + 1, h, device=device, dtype=torch.bfloat16),
            scores,
            *shared,
            *routed,
        )
    y, context = f.forward(config, workspace, schedule, x, scores, *shared, *routed)
    with pytest.raises(ValueError, match="matching"):
        f.backward(
            config,
            workspace,
            dataclasses.replace(schedule),
            context,
            dy,
            x,
            scores,
            *shared,
            *routed,
        )
    gradients = f.backward(
        config, workspace, schedule, context, dy, x, scores, *shared, *routed
    )
    assert y.shape == gradients[0].shape == (n, h) and gradients[1].shape == (n, k)
    assert all(torch.count_nonzero(v).item() == 0 for v in (y, *gradients))
    torch.cuda.synchronize()
    dist.barrier()
