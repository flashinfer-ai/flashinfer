# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in EP16/32 lifecycle and caller-managed expert-placement checks.

Launch one torchrun worker per GPU across a single direct-access NVLink domain.
Other world sizes skip these tests. The regular smaller-rank suites are unchanged.
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    PrequantizedMoEWeights,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as checks
from tests.moe_ep.sm107_test_utils import (
    assert_reference,
    canonical_reference,
    quantize,
    weight_pack,
)

pytestmark = [pytest.mark.gpu_2, pytest.mark.arch_rubin]


def _assert_collective_reference(actual, expected, kind):
    failure = None
    try:
        assert_reference(actual, expected, kind)
    except AssertionError as error:
        failure = str(error)
    failed = torch.tensor(int(failure is not None), device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    assert not failed.item(), failure or "reference failed on another rank"


@pytest.fixture(autouse=True)
def wide_geometry(monkeypatch):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (16, 32) or os.environ.get("MEGA_NO_DIST") == "1":
        pytest.skip("requires a dedicated EP16/32 torchrun invocation")
    for name, value in dict(
        HIDDEN=512,
        INTERMEDIATE=256,
        NUM_EXPERTS=64,
        TOP_K=4,
        NUM_TOKENS=16,
        MAX_TOKENS=32,
    ).items():
        monkeypatch.setattr(checks, name, value)


def test_wide_ep_live_counts_and_empty_ranks():
    checks.test_moe_ep_sm107_block_scaled_mega_multirank_torch_oracle(
        "mxfp8_e4m3", False, False
    )


def test_wide_ep_pooled_graph():
    checks.test_prequantized_situ_pooled_graph_with_scaling("mxfp4_mxfp8")


@pytest.mark.parametrize("family", ["inference", "genphase"])
def test_wide_ep_masked_graph(family):
    checks.test_masked_nonfinite_scores_pooled_graph(
        "mxfp4_mxfp8", family, "bf16", False, False
    )


@pytest.mark.parametrize("family", ["inference", "genphase"])
def test_wide_ep_transformed_expert_updates_in_captured_graph(family):
    """Replicate experts into physical slots and update their captured weights."""
    rank, world = checks._launcher_ranks()
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    kind = "mxfp4_mxfp8"
    cfg = checks._megakernel_config(
        kind,
        kernel_variant=family,
        apply_routing_weights_before_fc2=False,
        **(
            dict(cluster_shape_mn=(4, 1), fc2_use_bulk=True)
            if family == "genphase"
            else {}
        ),
    )
    kernel = create_mega_kernel(cfg)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layer = None
    graph = None
    try:
        w13, w2, x, _, scores = checks._global_problem(world)
        logical_experts = checks.NUM_EXPERTS // 2
        ids = (
            torch.arange(
                world * checks.NUM_TOKENS * checks.TOP_K, device="cuda"
            ).reshape(world, checks.NUM_TOKENS, checks.TOP_K)
            % logical_experts
        )
        count = checks.NUM_EXPERTS // world
        local = slice(rank * count, (rank + 1) * count)
        fleet = FleetParams(
            num_experts=checks.NUM_EXPERTS,
            max_tokens_per_rank=checks.MAX_TOKENS,
            token_hidden_size=checks.HIDDEN,
        )

        def prepare(a, b):
            pack, _, _ = weight_pack(a, b, kind)
            local_pack = PrequantizedMoEWeights(
                *(
                    getattr(pack, name)[local]
                    for name in ("w13", "w2", "w13_scale", "w2_scale")
                )
            )
            return pack, kernel.preprocess_weights(local_pack, fleet)

        original_pack, transformed = prepare(w13, w2)
        replica_w13 = torch.cat((w13[:logical_experts], w13[:logical_experts]))
        replica_w2 = torch.cat((w2[:logical_experts], w2[:logical_experts]))
        replica_pack, replica_weights = prepare(replica_w13, replica_w2)
        changed_w2 = replica_w2.clone()
        changed_w2[logical_experts:] *= 0.5
        changed_pack, changed_weights = prepare(replica_w13, changed_w2)
        pointers = tuple(t.data_ptr() for pair in transformed for t in pair)
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                rank=rank, world_size=world, auto_bootstrap=False
            ),
            fleet_params=fleet,
            weights=None,
            backend=MegaConfig(megakernel=cfg, transformed_weights=transformed),
        )
        transformed = layer.get_transformed_weights()
        t = MoEEpTensors(x[rank].clone(), ids[rank].clone(), scores[rank].clone())
        layer.warmup(t)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            output = layer.forward(t)

        def replay(pack):
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            xq, sf = quantize(t.hidden_states, kind)
            reference = canonical_reference(
                xq, sf, t.topk_ids, t.topk_weights, pack, kind, early=False
            )
            _assert_collective_reference(output, reference, kind)

        replay(original_pack)
        baseline = output.clone()
        t.topk_ids[::2].add_(logical_experts)
        for pack, replacement in (
            (replica_pack, replica_weights),
            (changed_pack, changed_weights),
        ):
            for dst_pair, src_pair in zip(transformed, replacement, strict=True):
                for dst, src in zip(dst_pair, src_pair, strict=True):
                    dst.view(torch.uint8).copy_(src.view(torch.uint8))
            assert tuple(v.data_ptr() for pair in transformed for v in pair) == pointers
            replay(pack)
            _assert_collective_reference(output[1::2], baseline[1::2], kind)
            if pack is replica_pack:
                _assert_collective_reference(output, baseline, kind)
    finally:
        if graph is not None:
            graph.reset()
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
