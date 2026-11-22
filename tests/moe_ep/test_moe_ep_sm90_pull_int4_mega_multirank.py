# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public SM90 INT4 x FP8 MegaMoE correctness on 2, 4, or 8 Hopper ranks.

Run with torchrun::

    torchrun --standalone --nproc_per_node=2 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_int4_mega_multirank.py -v
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.gpu_2, pytest.mark.arch_hopper]
LOCAL_EXPERTS = 4
CAPACITY = 129
CLAMP = 4.0


def _gather(tensor):
    import torch
    import torch.distributed as dist

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.stack(gathered)


def _raw_weights(rank, hidden, intermediate):
    import torch
    from flashinfer.moe_ep import MoEWeightPack

    generator = torch.Generator().manual_seed(8431 + rank)

    def leg(rows, k):
        payload = torch.randint(
            256, (LOCAL_EXPERTS, rows, k // 2), dtype=torch.uint8, generator=generator
        )
        scales = (
            torch.rand(LOCAL_EXPERTS, rows, k // 128, generator=generator) * 0.025
            + 0.005
        ).bfloat16()
        return payload.cuda(), scales.cuda()

    w13, s13 = leg(2 * intermediate, hidden)
    w2, s2 = leg(hidden, intermediate)
    return MoEWeightPack(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)


def _inputs(rank, world, hidden, tokens, pattern, launch):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    generator = torch.Generator().manual_seed(2909 + 97 * launch + rank)
    x = torch.randn(tokens, hidden, generator=generator).bfloat16().cuda()
    token = torch.arange(tokens, device="cuda").view(-1, 1)
    slot = torch.arange(world, device="cuda").view(1, -1)
    owner = (rank + slot + launch) % world
    expert = (token + slot + launch) % LOCAL_EXPERTS
    if pattern == "sparse_owner":
        owner = (slot // LOCAL_EXPERTS + launch) % world
        expert = token * 0 + slot % LOCAL_EXPERTS
    ids = owner * LOCAL_EXPERTS + expert
    if pattern == "all_masked" or (pattern == "zero_source" and rank == launch % world):
        ids.fill_(-1)
    weights = ((2 * slot + 1).float() / (world * world)).expand(tokens, world)
    return MoEEpTensors(
        hidden_states=x,
        topk_ids=ids.contiguous(),
        topk_weights=weights.contiguous(),
    )


def _reference(raw, hidden, ids, weights, intermediate):
    """Independent raw-weight and BF16-input oracle for every source rank."""
    import torch

    from tests.moe_ep.test_sm90_pull_int4_kernel_vs_reference import (
        _decode,
        _reference as reference,
    )

    world, tokens, width = hidden.shape
    inputs = hidden.flatten(0, 1).float()
    scale = (inputs.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
    operands = SimpleNamespace(
        x=(inputs / scale).to(torch.float8_e4m3fn),
        x_sf=scale,
        topk_idx=ids.flatten(0, 1),
        topk_weights=weights.flatten(0, 1),
        hidden=width,
        num_topk=world,
    )
    legs = (
        _decode(raw[0].flatten(0, 1), raw[2].flatten(0, 1)),
        _decode(raw[1].flatten(0, 1), raw[3].flatten(0, 1)),
    )
    return reference(operands, legs, world * tokens, intermediate, CLAMP).reshape(
        world, tokens, width
    )


@pytest.mark.parametrize(
    "hidden,intermediate,tile_k,prepared",
    [(384, 384, 128, False), (768, 512, 256, True)],
    ids=["k128-raw", "k256-prepared-graph"],
)
def test_int4_public_multirank_reference_reuse_and_graph(
    hidden, intermediate, tile_k, prepared
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
        preprocess_sm90_pull_int4_mega_weights,
    )
    from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_mega_multirank import _fused_tactic

    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world in (2, 4, 8), "launch with torchrun --nproc_per_node=2, 4, or 8"
    rank = int(os.environ["RANK"])
    bootstrap = BootstrapConfig(
        world_size=world,
        rank=rank,
        device=int(os.environ.get("LOCAL_RANK", str(rank))),
    )
    ensure_moe_ep_cuda_device(bootstrap)
    raw = _raw_weights(rank, hidden, intermediate)
    transformed = (
        preprocess_sm90_pull_int4_mega_weights(
            raw, hidden_size=hidden, intermediate_size=intermediate
        )
        if prepared
        else None
    )
    # Odd FC2 M tiles exercise a partial tile or a padded two-CTA cluster.
    tactic = dict(
        _fused_tactic(),
        mma_tiler_mnk=(256, 64, tile_k),
        cluster_shape_mnk=(2, 1, 1) if prepared else (1, 1, 1),
        group_hint=264,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=world * LOCAL_EXPERTS,
            max_tokens_per_rank=CAPACITY,
            token_hidden_size=hidden,
        ),
        weights=None if prepared else raw,
        backend=MegaConfig(
            megakernel=Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=world,
                gate_up_clamp=CLAMP,
                knobs=tactic,
            ),
            quantize_input=True,
            preprocess_weights=not prepared,
            transformed_weights=transformed,
        ),
    )
    graph = None
    captured = None
    try:
        global_raw = tuple(
            _gather(getattr(raw, name))
            for name in ("w13", "w2", "w13_scale", "w2_scale")
        )
        patterns = (
            "cross_rank",
            "sparse_owner",
            "zero_source",
            "all_masked",
            "cross_rank",
        )
        workspace = None
        for launch, pattern in enumerate(patterns):
            tokens = CAPACITY if launch == 0 else 13
            batch = _inputs(rank, world, hidden, tokens, pattern, launch)
            gathered = tuple(
                _gather(t)
                for t in (batch.hidden_states, batch.topk_ids, batch.topk_weights)
            )
            owners = gathered[1] // LOCAL_EXPERTS
            if pattern == "cross_rank":
                for source in range(world):
                    assert torch.equal(
                        owners[source].unique().sort().values,
                        torch.arange(world, device="cuda"),
                    )
                    assert (owners[source] != source).any()
            elif pattern == "sparse_owner":
                assert owners.unique().numel() < world
            elif pattern == "zero_source":
                assert (gathered[1][launch % world] == -1).all()
            else:
                assert (gathered[1] == -1).all()

            actual = layer.forward(batch).clone()
            if workspace is None:
                workspace = layer._workspace
            assert layer._workspace is workspace
            expected = _reference(global_raw, *gathered, intermediate)
            global_actual = _gather(actual)
            assert torch.isfinite(global_actual).all()
            torch.testing.assert_close(global_actual, expected, atol=0.02, rtol=0.02)
            error = (global_actual.float() - expected.float()).norm()
            assert error / expected.float().norm().clamp_min(1e-6) < 0.025
            if pattern == "all_masked":
                assert torch.count_nonzero(global_actual) == 0
            elif pattern == "cross_rank":
                assert not torch.equal(expected[0], expected[1])

        repeated = layer.forward(batch).clone()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        if prepared:
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer.forward(batch)
            dist.barrier()
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        dist.barrier()
    finally:
        if graph is not None:
            torch.cuda.synchronize()
        captured = None
        graph = None
        layer.destroy()
