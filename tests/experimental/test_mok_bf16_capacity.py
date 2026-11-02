# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Distributed MoK adapter guards for source lengths; run with torchrun EP4.

Backend suite: the package is imported only through
:func:`tests.test_helpers.cake_package.cake_package`.
"""

import dataclasses
import os

import pytest

pytest_plugins = ["tests.experimental._mok_bf16_test_utils"]
import torch
import torch.distributed as dist

from tests.test_helpers.cake_package import cake_package

kernels = cake_package("cake_mok_bf16", "_kernels")
native = cake_package("cake_mok_bf16", "workspace")
MoKConfig = native.MoKConfig
prepare_mok_bf16 = cake_package("cake_mok_bf16", "api").prepare_mok_bf16


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 16, 64),
    reason="Requires a 4-, 16- or 64-rank symmetric-memory group",
)
def test_source_capacity_and_context_ownership(mok_distributed_group):
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
    with pytest.raises(TypeError, match="comm_split_log"):
        f.build_schedule(
            workspace,
            dataclasses.replace(config, comm_split_log=1),
            ids,
            num_local_experts=le,
        )
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
    # The schedule carries its expert layout, derived once; it equals a fresh
    # derivation on the written prefix (the rest is never read).
    capacity = workspace.storage.schedule_capacity
    assert schedule.layout is not None and schedule.layout.dtype == torch.int32
    assert schedule.layout.numel() == 3 * le + capacity // 256
    written = 3 * le + int(schedule.num_tokens.item()) // 256
    derived = kernels.expert_layout(
        f.scheduler.layout_module,
        schedule.tokens_per_expert,
        capacity,
        schedule.peer_rank,
    )
    assert torch.equal(schedule.layout[:written], derived[:written])
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
    with pytest.raises(ValueError, match="router_weights shape"):
        narrow = scores[:, :1].contiguous()
        f.forward(config, workspace, schedule, x, narrow, *shared, *routed)
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


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 16, 64),
    reason="Requires a 4-, 16- or 64-rank symmetric-memory group",
)
def test_backward_releases_launcher_scratch_before_epilogue(mok_distributed_group):
    """``backward`` keeps only ``dx_shared`` and the six weight gradients of the
    fused launch alive past the launch: the kernel's routed and shared scratch
    is released before the output epilogue runs, ``dx_shared`` before the
    score-gradient copy, and the returned gradients are the launcher's weight
    gradients in the documented order."""
    import weakref

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    ep = dist.get_world_size()
    n, h = 256, 256
    k, le = (2, 4) if ep == 4 else (8, 256 // ep)
    f = prepare_mok_bf16(ep_size=ep, local_experts=le, topk=k)
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=2.0,
    )
    workspace = f.create_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=n,
        hidden_size=h,
        topk=k,
    )
    generator = torch.Generator(device=device).manual_seed(2026 + dist.get_rank())
    options = dict(device=device, dtype=torch.bfloat16)
    x = torch.randn(n, h, generator=generator, **options) * 0.125
    dy = torch.randn(n, h, generator=generator, **options) * 0.125
    scores = torch.rand(n, k, device=device, generator=generator) + 0.1
    shared = [
        torch.randn(h, h, generator=generator, **options) / h**0.5 for _ in range(3)
    ]
    routed = [
        torch.randn(le, h, h, generator=generator, **options) / h**0.5 for _ in range(3)
    ]
    row = torch.arange(n, device=device)
    ids = torch.stack([(row + slot) % (ep * le) for slot in range(k)], -1)
    schedule = f.build_schedule(workspace, config, ids, num_local_experts=le)
    _, context = f.forward(config, workspace, schedule, x, scores, *shared, *routed)

    key = (False, False)  # (MXFP8, clamped SwiGLU) of the BF16 plain variant
    launcher = f._backward_kernels[key]
    scratch, dx_shared_ref, weight_grads = [], [], []

    class Recording:
        def __call__(self, *args, **kwargs):
            values = launcher(*args, **kwargs)
            dx_shared_ref.append(weakref.ref(values[0]))
            scratch.extend(weakref.ref(value) for value in values[1:9])
            weight_grads.extend(values[9:])
            return values

    epilogue = f.epilogues.backward

    def checking(shared_grad, routed_grad):
        # Entry of the epilogue: every scratch output of the launch is dead,
        # dx_shared is the tensor the launcher returned.
        assert len(scratch) == 8 and all(ref() is None for ref in scratch)
        assert dx_shared_ref[0]() is shared_grad
        return epilogue(shared_grad, routed_grad)

    f._backward_kernels[key] = Recording()
    f.epilogues.backward = checking
    gradients = f.backward(
        config, workspace, schedule, context, dy, x, scores, *shared, *routed
    )
    assert len(scratch) == 8 and dx_shared_ref[0]() is None
    assert gradients[0].shape == (n, h) and gradients[1].shape == (n, k)
    # (dx, dscores, routed gate / up / down, shared gate / up / down) = the
    # launcher's (shared gate, routed gate, shared up, routed up, shared down,
    # routed down) weight gradients, reordered.
    assert len(weight_grads) == 6
    order = (1, 3, 5, 0, 2, 4)
    assert all(gradients[2 + i] is weight_grads[j] for i, j in enumerate(order))
    torch.cuda.synchronize()
    dist.barrier()
