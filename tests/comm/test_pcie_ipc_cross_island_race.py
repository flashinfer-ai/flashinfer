"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Deterministic regression for the TP8 cross-island scratch race.

The block scratch is double buffered so that a fast island cannot overwrite a
cross slot the slow island has not read yet. That property cannot be tested by
running the collective harder: the window only opens when one island stalls
*between* the owner-pair rendezvous and its cross read, which a host-side delay
cannot produce because the pair barrier releases both islands together.

So the kernel carries a debug-only stall, and this file pins the behaviour from
both sides:

  * with the stall and the double buffer  -> correct
  * with the stall and the double buffer disabled -> wrong

The negative control is the point. Without it, a passing test says nothing
about whether the fix is load-bearing.

Opt-in: needs 8 GPUs and builds two extra JIT modules (~1 min each), so it is
skipped unless FLASHINFER_TEST_PCIE_IPC_RACE=1.
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASHINFER_TEST_PCIE_IPC_RACE") != "1",
    reason="opt-in: needs 8 GPUs and builds two extra JIT modules",
)

_HIDDEN = 6144
# 100 us reproduced at batch 16 (one collective is ~63 us there); 200 us leaves
# margin on a busier machine without making the test slow.
_STALL_NS = 200_000
_ITERS = 40


def _mismatches(world_size: int, rank: int, stall_ns: int, no_block_epoch: int) -> int:
    """Run the race sequence and return the group-wide mismatch count."""
    import flashinfer.comm as comm
    from flashinfer.comm import pcie_ipc_ar
    from flashinfer.jit.comm import gen_pcie_ipc_comm_debug_module

    # The workspace resolves its module through this module-global, so swapping
    # it here is enough to put the instrumented build under the same wrapper
    # the shipping path uses.
    pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()
    original = pcie_ipc_ar.gen_pcie_ipc_comm_module
    pcie_ipc_ar.gen_pcie_ipc_comm_module = lambda: gen_pcie_ipc_comm_debug_module(
        stall_ns, 0, no_block_epoch
    )
    try:
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        batches = [16, 16, 96, 16]  # exercises ring blocks 1 -> 1 -> 2 -> 1
        # Pin the profile. The batches above are chosen against the rootcplx
        # table, where they all select the ring kernel -- the one the stall is
        # instrumented in. On a switch-paired machine the same batches select
        # the block kernel instead, and the test would still pass while
        # exercising something other than what it claims to.
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_HIDDEN * max(batches),
            dtype=torch.bfloat16,
            profile="rootcplx",
        )
        try:
            # Distinct payloads per call are essential. Reusing one input makes
            # every call produce the same island partial sums, so the racing
            # write stores a bit-identical value over the one it clobbers and
            # nothing is observable however wide the window is.
            inputs = [
                torch.randint(0, 16, (b, _HIDDEN), dtype=torch.int32, device=device).to(
                    torch.bfloat16
                )
                + i
                for i, b in enumerate(batches)
            ]
            refs = []
            for x in inputs:
                r = x.clone()
                dist.all_reduce(r, group=group)
                refs.append(r)
            dist.barrier(group=group)

            bad = 0
            for _ in range(_ITERS):
                for x, ref in zip(inputs, refs, strict=True):
                    bad += int((ws.all_reduce(x) != ref).sum().item())
        finally:
            ws.destroy()

        total = torch.tensor([bad], device=device)
        dist.all_reduce(total, group=group)
        return int(total.item())
    finally:
        pcie_ipc_ar.gen_pcie_ipc_comm_module = original
        pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()


def _fixed_worker(world_size: int, rank: int, port: int) -> None:
    try:
        _init_process_group(world_size, rank, port)
        bad = _mismatches(world_size, rank, _STALL_NS, no_block_epoch=0)
        assert bad == 0, (
            f"{bad} mismatched elements with the double buffer enabled: the "
            "cross-island scratch is being reused too early"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _negative_control_worker(world_size: int, rank: int, port: int) -> None:
    try:
        _init_process_group(world_size, rank, port)
        bad = _mismatches(world_size, rank, _STALL_NS, no_block_epoch=1)
        assert bad > 0, (
            "no mismatch with the double buffer disabled -- this test is not "
            "exercising the race, so its passing sibling proves nothing. "
            "Check that the stall is still between the pair rendezvous and the "
            "cross read, and that the payloads differ between calls."
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _grid_change_worker(world_size: int, rank: int, port: int) -> None:
    """The shipping sequence from test_pcie_ipc_grid_change, on the old scheme.

    Built with per-block epoch parity, so growing the grid puts a
    first-appearance block on the half the previous call used. Here that is not
    a corrupted number but a deadlock: the mis-addressed write lands on the
    address the victim polls next, so the victim waits for a sentinel nobody
    will clear.
    """
    from tests.comm.test_pcie_ipc_all_reduce import (
        _GRID_CHANGE_BATCHES,
        _grid_change_worker as shipping_worker,
    )

    from flashinfer.comm import pcie_ipc_ar
    from flashinfer.jit.comm import gen_pcie_ipc_comm_debug_module

    assert _GRID_CHANGE_BATCHES  # the sequence under test is the shipping one
    pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()
    original = pcie_ipc_ar.gen_pcie_ipc_comm_module
    pcie_ipc_ar.gen_pcie_ipc_comm_module = lambda: gen_pcie_ipc_comm_debug_module(
        0, 0, 0, 1
    )
    try:
        shipping_worker(world_size, rank, port)
    finally:
        pcie_ipc_ar.gen_pcie_ipc_comm_module = original
        pcie_ipc_ar.get_pcie_ipc_comm_module.cache_clear()


def _requires_8_gpus() -> None:
    if torch.cuda.device_count() < 8:
        pytest.skip("the cross-island race needs both 4-GPU islands")


def test_cross_island_scratch_is_double_buffered() -> None:
    _requires_8_gpus()
    multi_process_parallel(8, _fixed_worker)


def test_cross_island_race_reproduces_without_double_buffer() -> None:
    """Negative control: the fix must be load-bearing."""
    _requires_8_gpus()
    multi_process_parallel(8, _negative_control_worker)


def test_grid_change_deadlocks_with_per_block_epoch() -> None:
    """Negative control for the call-level epoch: the old scheme must deadlock.

    Asserting on the deadlock rather than on a mismatch count is not a
    compromise, it is the actual signal. In these kernels a mis-addressed write
    lands on the address the victim polls next, so corruption and deadlock are
    the same event -- there is no interleaving that yields a wrong number
    without also wedging the spin.

    The timeout is short because the hang is immediate: it happens on the third
    call, within a second of the workspace being built.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("the grid-change sequence is tuned against the 4-rank table")
    with pytest.raises(AssertionError, match="did not finish"):
        multi_process_parallel(4, _grid_change_worker, timeout_s=90)
