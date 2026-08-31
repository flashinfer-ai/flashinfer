"""GPU acceptance for split MXFP4 collective autotune winner replay.

Run in a fresh process because the SM90 and SM100 CuTeDSL trees expose
colliding top-level module names::

    torchrun --standalone --nproc_per_node=4 -m pytest \
      tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_split_autotune_multirank.py \
      -v -m arch_hopper

The first forward collectively sweeps the compact manifest union.  The
backend must then own a newly committed, captured winner session that can be
replayed repeatedly without changing fixed pointers or poisoning the graph.
"""

from __future__ import annotations

import json
import os

import pytest


pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")


HIDDEN = 128
INTERMEDIATE = 128
LOCAL_EXPERTS = 4
TOP_K = 4
MAX_TOKENS = 64
REPLAY_ITERS = int(os.environ.get("SM90_MXFP4_AUTOTUNE_REPLAY_ITERS", "16"))


def _launcher_ranks() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


@pytest.mark.arch_hopper
def test_split_auto_commits_winner_and_replays_without_deadlock() -> None:
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        MegaMoEHopperMxfp4SplitSymmBuffer,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4_split import (
        _SPLIT_TUNING_IDENTITY,
    )
    from tests.moe_ep._sm90_mxfp4_split_e2e import (
        all_gather_stack,
        assert_output_matches,
        global_route_reference,
        output_digest,
    )
    from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_mega_multirank import (
        _gather_raw_weights,
        _make_raw_weights,
    )
    from tests.moe_ep.test_moe_ep_sm90_pull_mxfp4_split_multirank import (
        _make_case,
    )

    assert torch.cuda.is_available()
    rank, world_size, local_rank = _launcher_ranks()
    assert world_size in (1, 2, 4)
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        execution_mode="split",
        knobs="auto",
    )
    kernel = create_mega_kernel(config)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )
    layer = None
    try:
        raw = _make_raw_weights(rank)
        raw_global = _gather_raw_weights(raw)
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            fleet_params=FleetParams(
                num_experts=world_size * LOCAL_EXPERTS,
                max_tokens_per_rank=MAX_TOKENS,
                token_hidden_size=HIDDEN,
            ),
            weights=raw,
            backend=MegaConfig(
                megakernel=config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        hidden, ids, weights = _make_case(rank, world_size, "balanced", 8, seed=9817)
        tensors = MoEEpTensors(
            hidden_states=hidden,
            topk_ids=ids,
            topk_weights=weights,
        )
        expected = global_route_reference(
            all_gather_stack(hidden),
            all_gather_stack(ids),
            all_gather_stack(weights),
            raw_global,
        )

        # First forward tunes, commits a fresh winner session, captures it,
        # and performs the caller-visible launch.
        baseline = layer.forward(tensors).clone()
        torch.cuda.synchronize()
        assert_output_matches(
            all_gather_stack(baseline), expected, label="split_auto_first"
        )
        workspace = layer._workspace
        assert isinstance(workspace, MegaMoEHopperMxfp4SplitSymmBuffer)
        session = workspace.session
        assert session.captured and not session.poisoned and not session.destroyed
        pointer_key = session.fixed_pointer_key
        assert pointer_key is not None
        digest = output_digest(baseline)

        result = None
        for _ in range(REPLAY_ITERS):
            result = layer.forward(tensors)
        assert result is not None
        result = result.clone()
        torch.cuda.synchronize()
        assert output_digest(result) == digest
        assert workspace.session is session
        assert session.fixed_pointer_key == pointer_key
        assert not session.poisoned

        dist.barrier()
        cache_path = os.environ.get("FLASHINFER_MOE_EP_KNOB_CACHE")
        if rank == 0 and cache_path:
            with open(cache_path) as file:
                entries = json.load(file)["entries"]
            assert len(entries) == 1
            entry = entries[0]
            assert entry["dtype"] == _SPLIT_TUNING_IDENTITY
            assert entry["fp8_scale_mode"] == "mxfp4_hybrid"
            assert entry["world_size"] == world_size
            assert entry["max_tokens"] == MAX_TOKENS
            assert set(entry["knobs"]) == {
                "counter_epoch_banks",
                "enable_iket",
                "graph_variant",
                "k1_cluster_shape_mnk",
                "k1_group_hint",
                "k1_mma_tiler_mnk",
                "k1_num_sched_stages",
                "k1_sm_count",
                "k2_cluster_shape_mnk",
                "k2_group_hint",
                "k2_mma_tiler_mnk",
                "k2_num_sched_stages",
                "k2_sm_count",
            }
        dist.barrier()
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
