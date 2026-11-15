"""Run each MXFP4 default band and a partial override through the public API."""

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep
from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

pytestmark = pytest.mark.node_exclusive


@pytest.mark.arch_blackwell
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "capacity,knobs,tile_n,schedule",
    [
        (128, None, 64, "grouped"),
        (768, None, 128, "grouped"),
        (1536, None, 128, "phase_interleave"),
        (1536, {"load_balance_mode": "static"}, 128, "grouped"),
        (1536, {"schedule_mode": "phase_interleave"}, 128, "phase_interleave"),
        (
            1536,
            {"schedule_policy": ("phase_interleave", None)},
            128,
            "phase_interleave",
        ),
    ],
)
def test_mxfp4_default_bands(capacity, knobs, tile_n, schedule, monkeypatch):
    """Check selected tactics, reference accuracy, and fresh graph outputs."""
    if not torch.cuda.is_available():
        pytest.skip("requires Blackwell")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    if world == 1:
        monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, intermediate, local_experts, topk = 256, 4096, 16, 2
    rng = torch.Generator(device="cuda").manual_seed(7908 + rank)
    weights = moe_ep.MoEWeightPack(
        torch.randn(
            local_experts, 2 * intermediate, hidden, device="cuda", generator=rng
        )
        .div_(hidden**0.5)
        .bfloat16(),
        torch.randn(local_experts, hidden, intermediate, device="cuda", generator=rng)
        .div_(intermediate**0.5)
        .bfloat16(),
    )
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=world, rank=rank),
        moe_ep.FleetParams(
            num_experts=local_experts * world,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        weights,
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=intermediate,
                top_k=topk,
                apply_routing_weights_before_fc2=False,
                knobs=knobs,
            )
        ),
    )
    tensors = moe_ep.MoEEpTensors(
        hidden_states=torch.randn(
            capacity, hidden, device="cuda", generator=rng
        ).bfloat16(),
        topk_ids=torch.rand(
            capacity, local_experts * world, device="cuda", generator=rng
        )
        .topk(topk, dim=-1)
        .indices.int(),
        topk_weights=torch.full((capacity, topk), 1 / topk, device="cuda"),
    )
    graph = None
    try:
        layer.warmup(tensors)
        ws = layer._ensure_workspace()
        assert ws.config.mma_tiler_mnk == (256, tile_n, 128)
        assert ws.config.schedule_policy[0] == schedule
        output = layer.forward(tensors, return_workspace_view=True)
        indices, expected = pkg.sampled_reference(
            ws, *layer._transformed, capacity, sample_size=32
        )
        error = torch.tensor(pkg.output_error(output, indices, expected), device="cuda")
        if world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error.item() < 0.03
        assert torch.isfinite(output).all()
        eager = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer.forward(tensors, return_workspace_view=True)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.gpu_4
@pytest.mark.parametrize("failure", ["mismatch", "config", "occupancy"])
def test_mxfp4_rank_agreement_before_pool_reuse(failure, monkeypatch):
    """Reject divergent settings and rank-local failures before pooled reuse."""
    if not torch.cuda.is_available():
        pytest.skip("requires Blackwell")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("requires multiple EP ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    weights = moe_ep.MoEWeightPack(
        torch.zeros(2, 512, 256, device="cuda", dtype=torch.bfloat16),
        torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16),
    )
    layers = []
    try:
        for knobs in (None, None, {"flag_batch": 2 if rank == 0 else 1}):
            layer = moe_ep.MoEEpLayer(
                moe_ep.BootstrapConfig(world_size=world, rank=rank),
                moe_ep.FleetParams(
                    num_experts=2 * world,
                    max_tokens_per_rank=32,
                    token_hidden_size=256,
                ),
                weights,
                backend=moe_ep.MegaConfig(
                    megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                        intermediate_size=256, top_k=2, knobs=knobs
                    )
                ),
            )
            layers.append(layer)
            if len(layers) == 3:
                # Inject failures after construction to test workspace preparation.
                if rank == 0 and failure == "config":
                    monkeypatch.setattr(
                        layer._kernel._config, "knobs", {"schedule_mode": "invalid"}
                    )
                if rank == 0 and failure == "occupancy":

                    def failed_probe(cluster_size):
                        """Model a recoverable driver occupancy-query failure."""
                        raise RuntimeError("occupancy probe failed")

                    monkeypatch.setattr(layer._kernel, "_active_clusters", failed_probe)
                # Only rank 0 would miss the pool without the agreement check.
                message = {
                    "mismatch": "differs across EP ranks",
                    "config": "rank 0: MoEEpConfigError: Unsupported schedule_mode",
                    "occupancy": "rank 0: RuntimeError: occupancy probe failed",
                }[failure]
                with pytest.raises(moe_ep.MoEEpConfigError, match=message):
                    layer._ensure_workspace()
            elif len(layers) == 2:
                assert layer._ensure_workspace() is layers[0]._ensure_workspace()
            else:
                layer._ensure_workspace()
        dist.barrier()
    finally:
        for layer in reversed(layers):
            layer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.gpu_4
@pytest.mark.parametrize("failure", ["config", "occupancy"])
def test_mxfp4_create_workspace_validation_failure(failure, monkeypatch):
    """Fail collectively during public handle validation, before preparation."""
    if not torch.cuda.is_available():
        pytest.skip("requires Blackwell")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("requires multiple EP ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    layer = moe_ep.MoEEpLayer(
        moe_ep.BootstrapConfig(world_size=world, rank=rank),
        moe_ep.FleetParams(
            num_experts=2 * world, max_tokens_per_rank=32, token_hidden_size=256
        ),
        moe_ep.MoEWeightPack(
            torch.zeros(2, 512, 256, device="cuda", dtype=torch.bfloat16),
            torch.zeros(2, 256, 256, device="cuda", dtype=torch.bfloat16),
        ),
        backend=moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=256, top_k=2
            )
        ),
    )
    try:
        with (
            monkeypatch.context() as patch,
            mock.patch.object(type(layer._kernel), "prepare_workspace") as prepare,
        ):
            knobs = None
            if failure == "config":
                # Explicit knobs create a new backend before validation.
                knobs = {"schedule_mode": "invalid" if rank == 0 else "grouped"}
                message = "rank 0: MoEEpConfigError: Unsupported schedule_mode"
            else:
                if rank == 0:
                    probe = mock.Mock(
                        side_effect=RuntimeError("occupancy probe failed")
                    )
                    patch.setattr(layer._kernel, "_active_clusters", probe)
                message = "rank 0: RuntimeError: occupancy probe failed"
            # This capacity triggers the occupancy probe in validate_init().
            with pytest.raises(moe_ep.MoEEpConfigError, match=message):
                layer.create_workspace(256, knobs=knobs)
            prepare.assert_not_called()
            if failure == "occupancy" and rank == 0:
                probe.assert_called_once()
        dist.barrier()
        # Failure must not register a handle or prevent a later valid request.
        handle = layer.create_workspace(256)
        assert handle.max_tokens_per_rank == 256
        handle.destroy()
    finally:
        layer.destroy()
