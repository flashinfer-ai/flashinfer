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
from flashinfer.moe_ep.shims.sm107 import cutedsl_megamoe as pkg

pytestmark = [pytest.mark.gpu_2, pytest.mark.arch_rubin]


@pytest.fixture(scope="module")
def ep_runtime():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8, 16, 32) or os.environ.get("MEGA_NO_DIST") == "1":
        pytest.skip(
            "requires torchrun with 2, 4, 8, 16, or 32 ranks and MEGA_NO_DIST unset"
        )
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
@pytest.mark.parametrize("caller_output", [False, True], ids=["workspace", "caller"])
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
    ep_runtime, family, kind, cluster, capacity, early, caller_output
):
    """Exercise routing changes with workspace and caller outputs across ranks."""
    _check_sparse_to_dense_graph_multirank(
        ep_runtime, family, kind, cluster, capacity, early, caller_output
    )


def _check_sparse_to_dense_graph_multirank(
    ep_runtime,
    family,
    kind,
    cluster,
    capacity,
    early,
    caller_output=False,
    *,
    topk_override=None,
    remote_experts_only=False,
    initial_epoch=None,
    graph_replays=65,
):
    """Check graph replay against the oracle after poisoning combine and output.

    Empty, sparse, and rank-asymmetric routes must clear inactive rows and
    preserve numerical accuracy through repeated replays with fixed bindings.
    """
    rank, world = ep_runtime
    hidden, intermediate, experts = 512, 256, max(16, 2 * world)
    # DATA CTA count depends on capacity * topk, not token capacity alone.
    # Exceed 16384 routes so the 2048-capacity case exercises two DATA CTAs.
    topk = 10 if capacity == 2048 else 4
    if topk_override is not None:
        topk = topk_override
    if remote_experts_only:
        experts = 128
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
    output = (
        torch.empty_like(ws.output_activation)
        if caller_output
        else ws.output_activation
    )

    def launch():
        """Launch with the output binding selected for this multirank case."""
        if caller_output:
            pkg.sm107_block_scaled_mega_moe(output, *weights, ws, num_tokens=capacity)
        else:
            ws.launch(*weights)

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
            assert ws.kernel.tail.world_size == world
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
        if remote_experts_only:
            base_ids = (
                torch.arange(topk, device="cuda", dtype=torch.int32)
                .add(experts // 2)
                .expand(capacity, topk)
                .contiguous()
            )
        ws.topk_idx.copy_(base_ids)
        _barrier()
        for _ in range(3):
            launch()
        _barrier()
        with torch.cuda.graph(graph):
            launch()
        _barrier()

        if initial_epoch is not None:
            workspace = ws.kernel._device_workspace
            tail = ws.kernel.tail
            offset = workspace.offset(tail.epoch_region)
            ws.local_workspace[offset : offset + 4].view(torch.int32).fill_(
                initial_epoch
            )
            offset = workspace.offset(ws.kernel.token_comm.call_sync_region)
            calls = ws.local_workspace[offset : offset + 8].view(torch.int32)
            calls[0] = initial_epoch
            calls[1] = 0
            phase = initial_epoch & 3
            offset = workspace.offset(tail.signal_region)
            signals = ws.shared_workspace[offset : offset + 8].view(torch.int32)
            signals[0] = world if phase in (1, 2) else 0
            signals[1] = world if phase in (2, 3) else 0
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
                    if pos >= capacity:
                        continue
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
            combine = ws._pre_reduced_activation.view(torch.bfloat16)
            combine.fill_(float("nan"))
            if family == "genphase" and ws.kernel.epilogue.combine_ll_encoded:
                # Zero means pending arrival in the encoded plane. Poison only
                # inactive slots so active cells cannot falsely signal ready.
                active = (ids >= 0) & (ids < experts)
                combine.masked_fill_(active.unsqueeze(-1), 0)
            output.fill_(float("nan"))
            # Peers write the poisoned buffers. Finish poisoning on EVERY rank
            # before any peer starts, or the diagnostic itself introduces a race.
            _barrier()
            if initial_epoch is not None:
                region = ws.kernel.token_comm.count_ll_inbox_region
                offset = workspace.offset(region)
                counts = (
                    ws.shared_workspace[offset : offset + workspace.nbytes(region)]
                    .view(torch.int64)
                    .view(2, world, -1)
                )
                parity = (initial_epoch + step) & 1
                previous_half = counts[1 - parity].clone()
            graph.replay()
            torch.cuda.synchronize()
            first = output.clone()
            if initial_epoch is not None:
                tag = ((initial_epoch + step + 1 + (1 << 31)) % (1 << 32)) - (1 << 31)
                tag = -(1 << 31) if tag == 0 else tag
                valid = torch.tensor(
                    int(torch.all((counts[parity] >> 32) == tag).item())
                    & int(torch.equal(counts[1 - parity], previous_half)),
                    device="cuda",
                )
                dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                assert valid.item(), (
                    "LL publication changed the wrong half or used a stale tag"
                )
            for _ in range(graph_replays - 1):
                graph.replay()
            torch.cuda.synchronize()
            inactive = ~((ids >= 0) & (ids < experts)).any(dim=1)
            stats = []
            for actual in (first, output):
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
    from flashinfer.moe_ep.kernels.cutedsl_megamoe import (
        RubinInferenceGenphaseMegaMoE,
    )
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.kernel_src.rubin.inference.mega.token_comm_gen_specialized import (
        GenphaseTokenComm,
    )

    # This private geometry changes relative producer/consumer progress without
    # changing the math. Empty work must drain both the router and all main
    # roles before clearing readiness flags; epilogue completion is insufficient.
    monkeypatch.setattr(
        RubinInferenceGenphaseMegaMoE,
        "pusher_cta_count",
        GenphaseTokenComm.sf_pusher_cta_count + 7,
    )
    _check_sparse_to_dense_graph_multirank(
        ep_runtime, "genphase", "mxfp8_e4m3", 4, 1024, early
    )


@pytest.mark.parametrize("topk", [31, 32, 33])
@pytest.mark.parametrize("early", [False, True])
def test_genphase_large_topk_remote_graph(ep_runtime, topk, early):
    # Ranks without expert work must wait for the remote FC2 stores before
    # static reduction; the polled path waits on its encoded cells instead.
    _check_sparse_to_dense_graph_multirank(
        ep_runtime,
        "genphase",
        "mxfp8_e4m3",
        4,
        512,
        early,
        topk_override=topk,
        remote_experts_only=True,
    )


@pytest.mark.parametrize("initial_epoch", [-2, (1 << 31) - 2])
@pytest.mark.parametrize("early", [False, True])
def test_genphase_graph_crosses_epoch_boundaries(ep_runtime, initial_epoch, early):
    _check_sparse_to_dense_graph_multirank(
        ep_runtime,
        "genphase",
        "mxfp8_e4m3",
        4,
        129,
        early,
        initial_epoch=initial_epoch,
        graph_replays=1,
    )
