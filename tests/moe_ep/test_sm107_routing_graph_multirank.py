# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check every capacity row after sparse-to-dense EP CUDA-graph transitions."""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep.config import BootstrapConfig
from flashinfer.moe_ep.core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
    sm107_block_scaled_runtime_requirements,
)
from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg

pytestmark = [pytest.mark.gpu_2, pytest.mark.arch_rubin]


@pytest.fixture(scope="module")
def ep_runtime():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8) or os.environ.get("MEGA_NO_DIST") == "1":
        pytest.skip("requires torchrun with 2, 4, or 8 ranks and MEGA_NO_DIST unset")
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, sm107_block_scaled_runtime_requirements(bootstrap)
    )
    try:
        yield rank, world
    finally:
        finalize_moe_ep_runtime(runtime)


def _barrier():
    dist.barrier()
    torch.cuda.synchronize()


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize(
    "family,kind,cluster,capacity",
    [
        ("inference", "mxfp8_e4m3", 4, 1024),
        ("inference", "mxfp8_e4m3", 4, 2048),
        ("inference", "mxfp4_mxfp8", 4, 1024),
        ("genphase", "mxfp8_e4m3", 4, 1024),
        ("genphase", "mxfp8_e4m3", 2, 1024),
        ("genphase", "mxfp4_mxfp8", 4, 1024),
    ],
)
def test_sparse_to_dense_graph_multirank(
    ep_runtime, family, kind, cluster, capacity, early
):
    _check_sparse_to_dense_graph_multirank(
        ep_runtime, family, kind, cluster, capacity, early
    )


def _check_sparse_to_dense_graph_multirank(
    ep_runtime, family, kind, cluster, capacity, early
):
    rank, world = ep_runtime
    hidden, intermediate, experts = 512, 256, 16
    # DATA CTA count depends on capacity * topk, not token capacity alone.
    # Exceed 16384 routes so the 2048-capacity case exercises two DATA CTAs.
    topk = 10 if capacity == 2048 else 4
    torch.manual_seed(16447)
    w1 = (
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") / hidden**0.5
    ).bfloat16()
    w2 = (
        torch.randn(experts, hidden, intermediate, device="cuda") / intermediate**0.5
    ).bfloat16()
    local = slice(rank * (experts // world), (rank + 1) * (experts // world))
    weights = pkg.preprocess_block_scaled_weights(
        w1[local], w2[local], quant_kind=kind, intermediate_size=intermediate
    )
    knobs = dict(
        kernel_variant=family,
        cluster_shape_mn=(cluster, 1),
        fc2_use_bulk=True,
        apply_routing_weights_before_fc2=early,
        reduce_topk_in_kernel=False,
    )
    if family == "genphase":
        knobs.update(
            mma_tiler_mnk=(256, 64, 128),
            fc2_tma_stages=1,
            epi_flag_batches=(1, 4),
            work_id_mode="atomic_counter",
        )
    ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        rank,
        world,
        quant_kind=kind,
        **knobs,
    )
    graph = torch.cuda.CUDAGraph()
    failures = []
    try:
        if family == "inference":
            reset_bytes = sum(
                ws.kernel._device_workspace.tail_reset_bytes(space)
                for space in ("local", "shared")
            )
            assert ws.kernel.token_comm.tail_reset_single_cta is (
                reset_bytes <= 16384 and os.environ.get("MEGA_USE_NCU", "0") != "1"
            )
            assert ws.kernel.token_comm._router.router_data_cta_count == (
                2 if capacity == 2048 else 1
            )
        else:
            reset_bytes = sum(
                ws.kernel._device_workspace.tail_reset_bytes(space)
                for space in ("local", "shared")
            )
            assert ws.kernel.tail_reset_single_cta is (
                reset_bytes <= 16384 and os.environ.get("MEGA_USE_NCU", "0") != "1"
            )
        torch.manual_seed(17000 + rank)
        x = torch.randn(capacity, hidden, device="cuda").bfloat16()
        qx, sx = pkg.quantize_mxfp8_block32(x, torch.float8_e4m3fn)
        ws.x.copy_(qx)
        ws.x_sf.zero_()
        ws.x_sf[:, : hidden // 32].copy_(sx)
        ws.topk_weights.fill_(1 / topk)
        ws.note_staged_tokens(capacity)
        base_ids = (
            torch.arange(capacity * topk, device="cuda", dtype=torch.int32)
            .view(capacity, topk)
            .add(rank * 3)
            .remainder(experts)
        )
        ws.topk_idx.copy_(base_ids)
        _barrier()
        for _ in range(3):
            ws.launch(*weights)
        _barrier()
        with torch.cuda.graph(graph):
            ws.launch(*weights)
        _barrier()

        states = ("dense129", "empty", "tail", "holes", "asymmetric", "restore")
        for step, state in enumerate(states):
            ids = (base_ids + 5 * step) % experts
            if state == "dense129":
                ids[129:] = -1
            elif state == "empty":
                ids.fill_(-1)
            elif state == "tail":
                ids.fill_(-1)
                ids[-1, -1] = ((rank + 1) % world) * (experts // world)
            elif state == "holes":
                ids.fill_(-1)
                for pos in (0, 16, 63, 64, 127, 128, 255, 256, 1023):
                    ids[pos, pos % topk] = (pos + rank) % experts
            elif state == "asymmetric":
                valid_counts = (0, capacity) if world == 2 else (0, 1, 129, capacity)
                ids[valid_counts[rank % len(valid_counts)] :] = -1
                ids[::3, 0] = -1
            # Keep pointers, descriptors and graph fixed; change their contents.
            ws.topk_idx.copy_(ids)
            qx, sx = pkg.quantize_mxfp8_block32(x * (1 + step / 8), torch.float8_e4m3fn)
            ws.x.copy_(qx)
            ws.x_sf[:, : hidden // 32].copy_(sx)
            ws.topk_weights.fill_((1 + step / 16) / topk)
            _barrier()
            rows, expected = pkg.sampled_reference(
                ws, *weights, capacity, sample_size=capacity
            )
            assert rows.numel() == capacity
            _barrier()
            ws._pre_reduced_activation.view(torch.bfloat16).fill_(float("nan"))
            ws.output_activation.fill_(float("nan"))
            # Peers write the poisoned buffers. Finish poisoning on EVERY rank
            # before any peer starts, or the diagnostic itself introduces a race.
            _barrier()
            graph.replay()
            torch.cuda.synchronize()
            first = ws.output_activation.clone()
            for _ in range(64):
                graph.replay()
            torch.cuda.synchronize()
            inactive = ~((ids >= 0) & (ids < experts)).any(dim=1)
            stats = []
            for actual in (first, ws.output_activation):
                error = pkg.output_error(actual, rows, expected)
                stats.extend(
                    (
                        error if math.isfinite(error) else float("inf"),
                        float(not torch.isfinite(actual).all().item()),
                        float(torch.count_nonzero(actual[inactive]).item() != 0),
                    )
                )
            worst = torch.tensor(stats, device="cuda", dtype=torch.float64)
            dist.all_reduce(worst, op=dist.ReduceOp.MAX)
            values = worst.tolist()
            print(f"{family}/{kind}/cga{cluster}/{early=} {state}: {values}")
            if any(values[i] >= 0.02 for i in (0, 3)) or any(
                values[i] != 0 for i in (1, 2, 4, 5)
            ):
                failures.append((state, values))
        # All ranks assert the same aggregate result after completing collectives.
        assert not failures, (
            f"first/final replay L2, nonfinite, nonzero-invalid: {failures}"
        )
    finally:
        graph.reset()
        _barrier()
        ws.destroy()
        _barrier()


@pytest.mark.parametrize("early", [False, True])
def test_sparse_to_dense_genphase_quiescence(ep_runtime, monkeypatch, early):
    """Stress empty-work reset with slower payload progress per pusher warp."""
    pkg.require_sm107_dsl()
    pkg.bootstrap_paths()
    from sources import RubinInferenceGenphaseMegaMoE

    # This private geometry changes relative producer/consumer progress without
    # changing the math. Empty work must drain both the router and all main
    # roles before clearing readiness flags; epilogue completion is insufficient.
    monkeypatch.setattr(RubinInferenceGenphaseMegaMoE, "pusher_cta_count", 7)
    _check_sparse_to_dense_graph_multirank(
        ep_runtime, "genphase", "mxfp8_e4m3", 4, 1024, early
    )
