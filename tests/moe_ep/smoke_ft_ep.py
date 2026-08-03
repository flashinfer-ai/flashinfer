"""Fault-tolerance smoke: HARD-kill a rank and keep serving, degraded.

Usage:
    torchrun --nproc_per_node=4 --max-restarts=0 tests/moe_ep/smoke_ft_ep.py \\
        --backend nccl_ep      # or nixl_ep

This is the hard-kill counterpart to
``test_moe_ep_fault_tolerance_multirank.py``, which stalls a rank instead.
It is a standalone script and NOT a pytest test on purpose: the victim really
does SIGTERM itself, torchrun reports its non-zero exit, and that would fail
the surviving ranks' pytest session even though the survivors behaved
correctly. Judge success by counting ``SMOKE_RESULT:`` lines instead --
``nproc - 1`` of them is a pass (``run_tests.sh ft`` does this).

Modeled on NIXL's own examples/device/ep/tests/elastic/elastic.py: the victim
installs a SIGTERM handler that tears its Fleet down, then re-raises with
SIG_DFL so the process actually dies.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from datetime import timedelta

_PG_TIMEOUT = timedelta(minutes=60)
_FT_TIMEOUT_MS = 2000
_VICTIM = 2
_KILL_AFTER_S = 1.0
_DEGRADED_ITERS = 10

# See smoke_nccl_ep.py: drop this script's dir from sys.path so the installed
# `nccl_ep` / `nixl_ep` modules aren't shadowed by the test subpackages here.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


def _self_kill():
    """Die the way a real crashed worker does."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> int:
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
        MoEEpTensors,
        NCCLEPConfig,
        NvepConfig,
        SplitConfig,
        dummy_moe_weights,
        supports_fault_tolerance,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nccl_ep", choices=["nccl_ep", "nixl_ep"])
    args = ap.parse_args()
    backend = args.backend

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo", timeout=_PG_TIMEOUT
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if not supports_fault_tolerance(backend):
        print(f"rank {rank}: {backend} cannot serve the FT API here; skipping")
        return 0
    if world_size < 4:
        print(f"rank {rank}: needs >=4 ranks, got {world_size}")
        return 1

    num_tokens, num_experts, hidden, topk = 64, 8, 4096, 4
    g = torch.Generator(device="cuda").manual_seed(11 + rank)
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

    tcp_store = None
    if backend == "nixl_ep":
        tcp_store = dist.TCPStore(
            host_name=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            port=int(os.environ.get("MASTER_PORT", "29500")) + 1,
            world_size=world_size,
            is_master=(rank == 0),
        )

    knobs = [FleetAlgoKnobFaultTolerance(timeout_ms=_FT_TIMEOUT_MS)]
    if backend == "nixl_ep":
        knobs.append(FleetAlgoKnobTopologyCapacity(n=world_size))

    layer = MoEEpLayer(
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
            comm=NvepConfig() if backend == "nixl_ep" else NCCLEPConfig(),
            kernel=IdentityConfig(),
        ),
    )

    layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()
    fleet = layer._ensure_fleet()

    if rank == _VICTIM:

        def _handler(_signum, _frame):
            try:
                layer.destroy()
            finally:
                _self_kill()

        signal.signal(signal.SIGTERM, _handler)
        threading.Timer(
            _KILL_AFTER_S, lambda: os.kill(os.getpid(), signal.SIGTERM)
        ).start()
        # Keep dispatching until the timer fires mid-flight.
        while True:
            layer.forward(t)
            torch.cuda.synchronize()

    # Survivors: the victim vanishes mid-dispatch. Their kernels time out on
    # it, mask it, and complete instead of trapping.
    for _ in range(3):
        layer.forward(t)
        torch.cuda.synchronize()
        if fleet.query_fault():
            break

    if not fleet.query_fault():
        print(f"rank {rank}: FAILED - never observed the fault")
        return 1

    mask = fleet.query_active_mask().cpu().tolist()
    if mask[_VICTIM] != 0:
        print(f"rank {rank}: FAILED - victim not masked, mask={mask}")
        return 1

    # NOTE: no dist.barrier() from here on. The victim is really dead, so any
    # collective over the full process group would hang -- which is exactly
    # why reconciliation goes through the store rather than an allreduce.
    agreed = fleet.reconcile_active_mask().cpu().tolist()
    expected = [0 if r == _VICTIM else 1 for r in range(world_size)]
    if agreed != expected:
        print(f"rank {rank}: FAILED - agreed {agreed}, expected {expected}")
        return 1
    fleet.clear_faults(readmit=False)

    for i in range(_DEGRADED_ITERS):
        y = layer.forward(t)
        torch.cuda.synchronize()
        if not torch.isfinite(y.float()).all():
            print(f"rank {rank}: FAILED - non-finite output on degraded iter {i}")
            return 1

    print(f"SMOKE_RESULT: {backend} FT OK (survivor rank {rank}, mask={agreed})")
    layer.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
