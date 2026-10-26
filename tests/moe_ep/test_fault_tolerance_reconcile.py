"""Active-mask reconciliation tests.

``reconcile_masks_via_store`` is a pure function of a torch.distributed Store,
so the whole protocol is exercised here in-process with a ``HashStore`` and
threads: no GPU, no CUDA, no transport, no torchrun.
"""

from __future__ import annotations

import threading

import pytest
import torch.distributed as dist

from flashinfer.moe_ep.core.comm.fault_tolerance import (
    ACTIVE,
    MASKED,
    reconcile_masks_via_store,
)


def _run_ranks(store, world_size, local_masks, participants=None, **kwargs):
    """Run reconcile concurrently for ``participants`` (default: all ranks).

    Returns {rank: agreed_mask_or_exception}.
    """
    if participants is None:
        participants = list(range(world_size))
    results: dict[int, object] = {}
    lock = threading.Lock()

    def work(r):
        try:
            out = reconcile_masks_via_store(
                store,
                rank=r,
                world_size=world_size,
                local_mask=local_masks[r],
                epoch=kwargs.get("epoch", 0),
                timeout_s=kwargs.get("timeout_s", 2.0),
                takeover_s=kwargs.get("takeover_s", 2.0),
            )
        except Exception as e:  # noqa: BLE001 - surfaced to the assertion below
            out = e
        with lock:
            results[r] = out

    threads = [threading.Thread(target=work, args=(r,)) for r in participants]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for t in threads:
        assert not t.is_alive(), "reconcile thread hung"
    return results


def _assert_all_agree(results, expected):
    for rank, got in results.items():
        assert not isinstance(got, Exception), f"rank {rank} raised: {got!r}"
        assert got == expected, f"rank {rank} got {got}, expected {expected}"


class TestHappyPath:
    def test_all_alive(self):
        world = 4
        store = dist.HashStore()
        masks = {r: [ACTIVE] * world for r in range(world)}
        results = _run_ranks(store, world, masks)
        _assert_all_agree(results, [ACTIVE] * world)

    def test_single_rank_world(self):
        store = dist.HashStore()
        results = _run_ranks(store, 1, {0: [ACTIVE]})
        _assert_all_agree(results, [ACTIVE])


class TestDeadRanks:
    def test_rank_never_reports(self):
        """Rank 2 never calls reconcile; survivors must agree it is gone."""
        world = 4
        store = dist.HashStore()
        # Survivors' local views still show 2 as active — only the missing
        # store key reveals the death.
        masks = {r: [ACTIVE] * world for r in range(world)}
        results = _run_ranks(
            store, world, masks, participants=[0, 1, 3], timeout_s=0.5, takeover_s=2.0
        )
        expected = [ACTIVE, ACTIVE, MASKED, ACTIVE]
        _assert_all_agree(results, expected)

    def test_disagreeing_views_are_anded(self):
        """Rank 1 saw 3 time out; rank 2 did not. AND => 3 masked everywhere.

        Rank 3 is alive and participating, but one peer's kernel already gave
        up on it, so it cannot stay in the group: leaving it active anywhere
        while rank 1 refuses to send to it is exactly the inconsistency that
        deadlocks the next dispatch. It therefore gets evicted.
        """
        from flashinfer.moe_ep.errors import MoEEpRankEvictedError

        world = 4
        store = dist.HashStore()
        masks = {
            0: [ACTIVE, ACTIVE, ACTIVE, ACTIVE],
            1: [ACTIVE, ACTIVE, ACTIVE, MASKED],  # rank 1 timed out on 3
            2: [ACTIVE, ACTIVE, ACTIVE, ACTIVE],
            3: [ACTIVE, ACTIVE, ACTIVE, ACTIVE],
        }
        results = _run_ranks(store, world, masks)
        expected = [ACTIVE, ACTIVE, ACTIVE, MASKED]
        for r in (0, 1, 2):
            assert results[r] == expected, f"rank {r} got {results[r]}"
        assert isinstance(results[3], MoEEpRankEvictedError)

    def test_rank_never_masks_itself(self):
        """A caller-supplied mask that masks the local rank is corrected."""
        world = 2
        store = dist.HashStore()
        masks = {0: [MASKED, ACTIVE], 1: [ACTIVE, ACTIVE]}
        results = _run_ranks(store, world, masks)
        # Rank 0 forces its own bit ACTIVE, so nothing is masked.
        _assert_all_agree(results, [ACTIVE, ACTIVE])

    def test_coordinator_is_the_dead_rank(self):
        """Rank 0 (lowest, the natural coordinator) is the one that died."""
        world = 4
        store = dist.HashStore()
        masks = {r: [ACTIVE] * world for r in range(world)}
        results = _run_ranks(
            store, world, masks, participants=[1, 2, 3], timeout_s=0.3, takeover_s=0.5
        )
        # Takeover elects rank 1; every survivor must adopt the same vector.
        _assert_all_agree(results, [MASKED, ACTIVE, ACTIVE, ACTIVE])

    def test_two_dead_ranks(self):
        world = 5
        store = dist.HashStore()
        masks = {r: [ACTIVE] * world for r in range(world)}
        results = _run_ranks(
            store, world, masks, participants=[0, 2, 4], timeout_s=0.3, takeover_s=0.5
        )
        _assert_all_agree(results, [ACTIVE, MASKED, ACTIVE, MASKED, ACTIVE])


class TestSplitBrain:
    def test_late_key_does_not_split_the_decision(self):
        """The regression test for the naive "everyone ANDs locally" design.

        Survivor A polls before the straggler's key lands, survivor B after.
        A naive implementation would have A mask the straggler and B keep it —
        two different masks, and a deadlock on the next dispatch. Because the
        decision is published through a single compare_set, both must return
        the coordinator's vector.
        """
        world = 3
        store = dist.HashStore()
        masks = {r: [ACTIVE] * world for r in range(world)}

        results: dict[int, object] = {}
        lock = threading.Lock()

        def run(r, timeout_s):
            try:
                out = reconcile_masks_via_store(
                    store,
                    rank=r,
                    world_size=world,
                    local_mask=masks[r],
                    epoch=0,
                    timeout_s=timeout_s,
                    takeover_s=5.0,
                )
            except Exception as e:  # noqa: BLE001 - asserted on below
                out = e
            with lock:
                results[r] = out

        def straggler():
            # Lands after rank 0's gather window but inside rank 1's.
            import time

            time.sleep(0.4)
            run(2, 5.0)

        threads = [
            threading.Thread(target=run, args=(0, 0.15)),  # gives up on rank 2
            threading.Thread(target=run, args=(1, 3.0)),  # waits and sees rank 2
            threading.Thread(target=straggler),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        for t in threads:
            assert not t.is_alive()

        assert len(results) == 3
        # The two survivors must return the SAME vector even though they
        # observed rank 2 differently. (Rank 2 itself either survives or is
        # evicted, depending on which side of rank 0's window it landed —
        # what must never happen is 0 and 1 disagreeing.)
        survivors = {
            r: v
            for r, v in results.items()
            if r in (0, 1) and not isinstance(v, Exception)
        }
        assert len(survivors) == 2, f"a survivor failed: {results}"
        distinct = {tuple(v) for v in survivors.values()}  # type: ignore[arg-type]
        assert len(distinct) == 1, f"split brain: ranks disagreed: {results}"

    def test_decision_is_write_once(self):
        """A later reconcile at the same epoch adopts the published decision."""
        world = 3
        store = dist.HashStore()
        # Pre-seed the peers' views so rank 0's gather succeeds and the
        # decision keeps everyone alive.
        store.set("ft/gen7/local/1", bytes([ACTIVE] * world))
        store.set("ft/gen7/local/2", bytes([ACTIVE] * world))
        first = reconcile_masks_via_store(
            store,
            rank=0,
            world_size=world,
            local_mask=[ACTIVE] * world,
            epoch=7,
            timeout_s=1.0,
            takeover_s=0.2,
        )
        assert first == [ACTIVE] * world
        # Rank 2 arrives later with a *different* local view but the same
        # epoch: it must adopt the published decision, not invent its own.
        second = reconcile_masks_via_store(
            store,
            rank=2,
            world_size=world,
            local_mask=[ACTIVE, MASKED, ACTIVE],
            epoch=7,
            timeout_s=0.1,
            takeover_s=0.2,
        )
        assert second == first

    def test_separate_epochs_are_independent(self):
        """The epoch namespaces the keys, so rounds do not leak into each other."""
        world = 2
        store = dist.HashStore()
        # Epoch 0: peer never reports -> presumed dead.
        a = reconcile_masks_via_store(
            store,
            rank=0,
            world_size=world,
            local_mask=[ACTIVE, ACTIVE],
            epoch=0,
            timeout_s=0.1,
            takeover_s=0.2,
        )
        assert a == [ACTIVE, MASKED]
        # Epoch 1: peer reports in -> alive again, unaffected by epoch 0.
        store.set("ft/gen1/local/1", bytes([ACTIVE, ACTIVE]))
        b = reconcile_masks_via_store(
            store,
            rank=0,
            world_size=world,
            local_mask=[ACTIVE, ACTIVE],
            epoch=1,
            timeout_s=1.0,
            takeover_s=0.2,
        )
        assert b == [ACTIVE, ACTIVE]


class TestEviction:
    def test_rank_masked_by_peers_raises(self):
        """A rank the survivors gave up on must not silently keep serving."""
        from flashinfer.moe_ep.errors import MoEEpRankEvictedError

        world = 3
        store = dist.HashStore()
        # Survivors already agreed rank 2 is dead.
        store.set("ft/gen0/decision", bytes([ACTIVE, ACTIVE, MASKED]))
        with pytest.raises(MoEEpRankEvictedError, match="rank 2 was masked out"):
            reconcile_masks_via_store(
                store,
                rank=2,
                world_size=world,
                local_mask=[ACTIVE] * world,
                epoch=0,
                timeout_s=0.1,
                takeover_s=0.5,
            )

    def test_survivors_are_unaffected(self):
        """The same decision is adopted normally by a rank it keeps alive."""
        world = 3
        store = dist.HashStore()
        store.set("ft/gen0/decision", bytes([ACTIVE, ACTIVE, MASKED]))
        got = reconcile_masks_via_store(
            store,
            rank=1,
            world_size=world,
            local_mask=[ACTIVE] * world,
            epoch=0,
            timeout_s=0.1,
            takeover_s=0.5,
        )
        assert got == [ACTIVE, ACTIVE, MASKED]


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rank": 5, "world_size": 4, "local_mask": [1] * 4}, "out of range"),
            ({"rank": 0, "world_size": 0, "local_mask": []}, "world_size"),
            ({"rank": 0, "world_size": 4, "local_mask": [1, 1]}, "local_mask"),
        ],
    )
    def test_rejects_bad_shapes(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            reconcile_masks_via_store(
                dist.HashStore(), epoch=0, timeout_s=0.1, takeover_s=0.1, **kwargs
            )

    def test_peer_from_a_different_world_is_ignored(self):
        """A stale key of the wrong length must not corrupt the AND."""
        world = 3
        store = dist.HashStore()
        store.set("ft/gen0/local/1", bytes([1, 1]))  # wrong length
        store.set("ft/gen0/local/2", bytes([1, 1, 1]))
        got = reconcile_masks_via_store(
            store,
            rank=0,
            world_size=world,
            local_mask=[ACTIVE] * world,
            epoch=0,
            timeout_s=0.1,
            takeover_s=0.2,
        )
        assert got == [ACTIVE, ACTIVE, ACTIVE]
