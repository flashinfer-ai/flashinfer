"""Fault-tolerance state machine on 4+ GPUs, end to end.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_fault_tolerance_multirank.py -v \\
        -m "nvep and gpu_4" --backend=nccl_ep      # or nixl_ep

Injection method: a STALLED rank, not a killed one. NIXL's own elastic test
kills a worker with SIGTERM, but under ``torchrun -m pytest`` that tears down
the whole job and takes the surviving ranks' pytest session with it. Instead
the victim skips one iteration by sleeping past the FT timeout while the
survivors dispatch, which is what the transports actually detect: a peer whose
data never arrives. Every process stays alive, so the torch process group
stays usable for the test's own barriers, and the whole detect -> reconcile ->
degrade -> re-admit -> healthy cycle runs in one job in ~10s.

The hard-kill counterpart lives in smoke_ft_ep.py, which cannot be a pytest
test for exactly the reason above.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)

# Deliberately short so a stalled rank is detected in seconds rather than the
# ~100s (nccl) / 30s (nixl) default. Production should leave this at 0: a low
# timeout marks merely-slow ranks dead.
_FT_TIMEOUT_MS = 2000
_STALL_S = 4.0  # > _FT_TIMEOUT_MS, with margin
_VICTIM = 2  # a MIDDLE rank: not rank 0 (the reconcile coordinator), not last


def pytest_generate_tests(metafunc):
    """Generate `comm_backend` param values from --backend CLI."""
    if "comm_backend" not in metafunc.fixturenames:
        return
    cli = metafunc.config.getoption("--backend", default=None)
    if cli == "both" or cli is None:
        metafunc.parametrize("comm_backend", ["nccl_ep", "nixl_ep"])
    else:
        metafunc.parametrize("comm_backend", [cli])


def _init_dist():
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    return rank, world


def _build_layer(comm_backend, rank, world_size, num_tokens, num_experts, hidden):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        FleetAlgoKnobFaultTolerance,
        FleetAlgoKnobTopologyCapacity,
        FleetParams,
        IdentityConfig,
        MoEEpLayer,
        NCCLEPConfig,
        NvepConfig,
        SplitConfig,
        dummy_moe_weights,
    )

    tcp_store = None
    if comm_backend == "nixl_ep":
        tcp_store = dist.TCPStore(
            host_name=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            port=int(os.environ.get("MASTER_PORT", "29500")) + 1,
            world_size=world_size,
            is_master=(rank == 0),
        )

    knobs = [FleetAlgoKnobFaultTolerance(timeout_ms=_FT_TIMEOUT_MS)]
    if comm_backend == "nixl_ep":
        knobs.append(FleetAlgoKnobTopologyCapacity(n=world_size))

    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
            tcp_store=tcp_store,
        ),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_tokens,
            token_hidden_size=hidden,
            dtype_bytes=2,
            algorithm=EpAlgorithm.LOW_LATENCY,
        ),
        weights=dummy_moe_weights(
            num_local_experts=num_experts // world_size, hidden=hidden
        ),
        fleet_knobs=knobs,
        backend=SplitConfig(
            comm=NvepConfig() if comm_backend == "nixl_ep" else NCCLEPConfig(),
            kernel=IdentityConfig(),
        ),
    )


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_fault_tolerance_detect_reconcile_degrade_readmit(comm_backend):
    """Walk the whole FT state machine with one stalled rank."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpTensors, supports_fault_tolerance

    if not supports_fault_tolerance(comm_backend):
        pytest.skip(f"{comm_backend} cannot serve the FT API on this host")

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    num_tokens, num_experts, hidden, topk = 64, 8, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(42 + rank)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", generator=g), dim=-1
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)

    layer = _build_layer(
        comm_backend, rank, world_size, num_tokens, num_experts, hidden
    )
    survivors = [r for r in range(world_size) if r != _VICTIM]
    expected = [1 if r in survivors else 0 for r in range(world_size)]

    # --- HEALTHY ---------------------------------------------------------
    y = layer.forward(t)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x, atol=5e-2, rtol=5e-2)
    dist.barrier()

    fleet = layer._ensure_fleet()
    assert fleet.supports_fault_tolerance is True
    assert fleet.query_active_mask().cpu().tolist() == [1] * world_size
    assert fleet.query_fault() is False

    # --- FAULT: the victim stalls past the timeout ------------------------
    if rank == _VICTIM:
        time.sleep(_STALL_S)
    else:
        layer.forward(t)  # survivors' kernels time out waiting on the victim
        torch.cuda.synchronize()
    dist.barrier()  # the victim is alive, so the torch PG is still healthy

    if rank in survivors:
        assert fleet.query_fault() is True, "survivors must observe the fault"
        mask = fleet.query_active_mask().cpu().tolist()
        assert mask[_VICTIM] == 0, f"rank {_VICTIM} should be masked, got {mask}"

    # --- RECONCILE -------------------------------------------------------
    # Every rank reconciles in the same iteration slot. The victim learns it
    # was evicted (it stalled long enough for the survivors to give up).
    from flashinfer.moe_ep.errors import MoEEpRankEvictedError

    if rank == _VICTIM:
        with pytest.raises(MoEEpRankEvictedError):
            fleet.reconcile_active_mask()
        print(f"rank {rank}: correctly evicted")
        layer.destroy()
        dist.barrier()
        return

    agreed = fleet.reconcile_active_mask().cpu().tolist()
    assert agreed == expected, f"rank {rank} agreed {agreed}, expected {expected}"
    fleet.clear_faults(readmit=False)
    assert fleet.query_fault() is False, "clear_faults must re-arm detection"

    # --- DEGRADED --------------------------------------------------------
    # The dead rank's experts are gone and combine does NOT renormalize, so
    # each token comes out scaled by the surviving weight fraction rather
    # than 1. Assert exactly that, not "close to x".
    y_deg = layer.forward(t)
    torch.cuda.synchronize()
    assert torch.isfinite(y_deg.float()).all(), "degraded output must not be NaN/Inf"

    experts_per_rank = num_experts // world_size
    alive = torch.tensor(
        [expected[e // experts_per_rank] for e in range(num_experts)],
        device="cuda",
        dtype=torch.bool,
    )
    surviving_w = (topk_weights * alive[topk_ids]).sum(-1)  # [num_tokens]
    torch.testing.assert_close(
        y_deg.float(), x.float() * surviving_w.unsqueeze(-1), atol=8e-2, rtol=8e-2
    )
    print(f"rank {rank}: degraded forward matches surviving-weight scaling")
    dist.barrier()

    layer.destroy()
    dist.barrier()


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_readmit_restores_full_strength(comm_backend):
    """clear_faults(readmit=True) puts a merely-delayed rank back in service.

    Split from the test above because re-admission is collective over the
    SURVIVORS, so the victim must not participate in it.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpTensors, supports_fault_tolerance
    from flashinfer.moe_ep.config import EpLayout

    if not supports_fault_tolerance(comm_backend):
        pytest.skip(f"{comm_backend} cannot serve the FT API on this host")

    rank, world_size = _init_dist()
    num_tokens, num_experts, hidden, topk = 64, 8, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", generator=g), dim=-1
    )
    t = MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)

    layer = _build_layer(
        comm_backend, rank, world_size, num_tokens, num_experts, hidden
    )
    layer.forward(t)  # allocates the LL staging buffer MaskClean asserts on
    torch.cuda.synchronize()
    dist.barrier()

    fleet = layer._ensure_fleet()

    # Mask a rank by hand (no stall needed — we are testing re-admission, and
    # every rank applies the same vector so the fleet stays coherent).
    mask = [1] * world_size
    mask[_VICTIM] = 0
    if rank != _VICTIM:
        fleet.set_active_mask(mask)
        assert fleet.query_active_mask().cpu().tolist() == mask
    dist.barrier()

    # Re-admit. Collective over survivors; the victim was never masked
    # locally, so it simply does not participate.
    if rank != _VICTIM:
        import warnings

        with warnings.catch_warnings():
            # EXPERT_MAJOR + MaskClean warns by design (nccl only).
            warnings.simplefilter("ignore", RuntimeWarning)
            fleet.clear_faults(readmit=True)
        assert fleet.query_active_mask().cpu().tolist() == [1] * world_size
        if fleet.params.layout is EpLayout.EXPERT_MAJOR:
            print(f"rank {rank}: re-admitted under EXPERT_MAJOR (see MaskClean note)")
    dist.barrier()

    # --- HEALTHY again: full-strength forward matches the baseline --------
    y = layer.forward(t)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x, atol=5e-2, rtol=5e-2)
    print(f"rank {rank}: full-strength forward restored after re-admission")
    layer.destroy()
    dist.barrier()
