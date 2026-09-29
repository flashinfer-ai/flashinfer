"""Backend-agnostic active-mask reconciliation.

When a peer stops responding, each surviving rank's transport masks it
*locally* — NCCL-EP's own header calls mask consistency "a framework-level
concern". Two survivors can therefore end up with different views: rank A's
kernel timed out on the straggler, rank B's did not. Running the next
dispatch with disagreeing masks deadlocks, so the views must be reconciled
before serving continues.

This module owns that agreement protocol. It is deliberately transport-free —
a pure function of a :class:`torch.distributed.Store` — so it can be unit
tested with an in-process ``HashStore`` and threads, with no GPUs and no
transport at all.

Why a store and not a collective: a ``torch.distributed`` allreduce over the
EP group would hang on exactly the rank we are trying to mask out. A store
round-trip, by contrast, lets a missing key *be* the signal that a rank is
gone.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    import torch

ACTIVE = 1
MASKED = 0

# How often the gather/decision polls re-check the store. Small enough that a
# healthy reconcile finishes in ~one interval, large enough not to hammer a
# TCPStore server. Polling with check() rather than blocking in wait() is what
# lets us report WHICH ranks failed to report, which wait() cannot.
_POLL_INTERVAL_S = 0.02


# Why each FT entry point refuses CUDA-graph capture. The reasons genuinely
# differ per call, and the distinction matters: the transports do NOT compact
# the surviving ranks' data layout, so dispatch/combine inside a captured graph
# keep honouring the live mask on every replay. Masking is not the problem —
# baking a *decision* about it into a graph is.
_CAPTURE_REASONS = {
    "query_fault": (
        "it is a host-side read, not stream work, so it cannot be captured at "
        "all: it would return the capture-time answer and the branch taken on "
        "it would be frozen into the graph's structure -- a graph that ignores "
        "faults because none had happened when it was recorded"
    ),
    "query_active_mask": (
        "its result can only be consumed on the host, which needs a sync that "
        "capture forbids; a captured copy would replay, but the decision made "
        "from it would not"
    ),
    "set_active_mask": (
        "the rank and value are baked into the captured operation, so every "
        "replay would re-apply that one mask regardless of who is alive"
    ),
    "clear_faults": (
        "re-admission resets transport state exactly once; replaying that on "
        "every graph launch is never what you want"
    ),
}


def reject_graph_capture(op: str) -> None:
    """Raise if called during CUDA-graph capture, explaining why for ``op``."""
    import torch

    if not torch.cuda.is_current_stream_capturing():
        return
    why = _CAPTURE_REASONS.get(op, "capturing it would freeze fault state")
    raise RuntimeError(
        f"Fleet.{op}() must not be called during CUDA-graph capture: {why}. "
        "Call it from the host between graph replays. Note that dispatch and "
        "combine themselves ARE safe to capture -- they re-read the mask on "
        "every replay."
    )


def _local_key(epoch: int, rank: int) -> str:
    return f"ft/gen{epoch}/local/{rank}"


def _decision_key(epoch: int) -> str:
    return f"ft/gen{epoch}/decision"


def _wait_for_key(store, key: str, deadline: float, clock: Callable[[], float]):
    """Poll for ``key`` until ``deadline``; return its bytes or None."""
    while True:
        if store.check([key]):
            return store.get(key)
        if clock() >= deadline:
            return None
        time.sleep(_POLL_INTERVAL_S)


def reconcile_masks_via_store(
    store,
    *,
    rank: int,
    world_size: int,
    local_mask: Sequence[int],
    epoch: int,
    timeout_s: float,
    takeover_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> list[int]:
    """Agree one active mask across the survivors of an EP group.

    Args:
        store: any ``torch.distributed.Store`` (TCPStore / PrefixStore /
            HashStore). Must be shared by the whole EP group and namespaced to
            it — see ``resolve_rendezvous_store``.
        rank, world_size: this rank's position in the EP group.
        local_mask: this rank's view, ``1 = active``. ``local_mask[rank]`` is
            forced to ACTIVE; a rank never masks itself.
        epoch: round identifier. **Every survivor must pass the same value.**
            That is guaranteed by the ordering rule that all survivors
            reconcile in the same iteration slot (see the runbook); a rank
            that joins a round more than ``takeover_s`` late is treated as
            dead by its peers and may compute a decision of its own.
        timeout_s: budget for the gather phase. A rank whose key has not
            appeared by then is presumed dead.
        takeover_s: budget for waiting on the coordinator's decision before
            presuming the coordinator itself is dead.
        clock: injectable monotonic clock (tests).

    Returns:
        The agreed mask, ``1 = active``, length ``world_size``. Identical on
        every rank that returns.

    The protocol, and why each step is there:

    1. Publish our local view at ``ft/gen{E}/local/{rank}``.
    2. Wait only on the ranks we still *believe* are alive — never on one we
       already know is gone, which would burn the whole timeout every round.
    3. Elementwise-AND the views that arrived; mask any believed-alive rank
       whose view never showed up.
    4. **Single-source the decision.** The lowest-numbered survivor publishes
       the agreed vector with an atomic compare-and-set, and every other rank
       adopts whatever that key holds.

    Step 4 is the crux. With a naive "everyone ANDs locally and applies the
    result", a straggler's key landing at T+9.9s is seen by survivor B but not
    by survivor A polling at T+9.8s, and the two apply *different* masks — a
    split brain that deadlocks the next dispatch. Publishing through one
    compare-and-set makes the outcome identical by construction for everyone
    who reads it, regardless of timing. ``compare_set(key, b"", v)`` is
    set-if-absent-else-return-current on both TCPStore and HashStore, so even
    a coordinator takeover race resolves to a single value.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    if len(local_mask) != world_size:
        raise ValueError(
            f"local_mask has {len(local_mask)} entries, expected {world_size}"
        )

    view = [ACTIVE if int(v) == ACTIVE else MASKED for v in local_mask]
    view[rank] = ACTIVE  # a rank never masks itself

    # 1. publish our view
    store.set(_local_key(epoch, rank), bytes(view))

    # 2/3. gather from the ranks we believe are alive, then AND
    believed = [r for r in range(world_size) if view[r] == ACTIVE]
    pending = {r: _local_key(epoch, r) for r in believed if r != rank}
    deadline = clock() + timeout_s
    arrived: dict[int, list[int]] = {rank: list(view)}
    while pending:
        for r, key in list(pending.items()):
            if store.check([key]):
                arrived[r] = list(store.get(key))
                del pending[r]
        if not pending:
            break
        if clock() >= deadline:
            break
        time.sleep(_POLL_INTERVAL_S)

    agreed = list(view)
    for peer_view in arrived.values():
        if len(peer_view) != world_size:
            # A peer from a different world size cannot be reconciled with;
            # ignoring it is safer than truncating to a wrong length.
            continue
        agreed = [
            ACTIVE if (a == ACTIVE and int(b) == ACTIVE) else MASKED
            for a, b in zip(agreed, peer_view, strict=True)
        ]
    for r in pending:  # believed alive, never reported -> gone
        agreed[r] = MASKED
    agreed[rank] = ACTIVE

    # 4. single-source the decision through the lowest-numbered survivor
    decision = _decision_key(epoch)
    for _ in range(world_size + 1):
        coord = min(r for r in range(world_size) if agreed[r] == ACTIVE)
        if coord == rank:
            # compare_set is set-if-absent: if a takeover peer already
            # published, we adopt theirs instead of overwriting it.
            winner = store.compare_set(decision, b"", bytes(agreed))
            return _adopt(winner, world_size, agreed, rank, epoch)
        published = _wait_for_key(store, decision, clock() + takeover_s, clock)
        if published is not None:
            return _adopt(published, world_size, agreed, rank, epoch)
        # The coordinator never published: presume it died mid-round, drop it
        # and re-elect. Bounded by world_size iterations.
        agreed[coord] = MASKED

    raise RuntimeError(
        f"active-mask reconciliation for epoch {epoch} exhausted "
        f"{world_size + 1} coordinator elections without a decision; the "
        "rendezvous store may be unreachable."
    )


def _adopt(
    raw, world_size: int, fallback: list[int], rank: int, epoch: int
) -> list[int]:
    """Decode a published decision, refusing one that masks us.

    A decision that masks the local rank means the survivors gave up on us
    while we were still running — we stalled long enough for their kernels to
    time out. We cannot apply it (both transports reject masking the local
    rank) and we must not ignore it either, since our peers have stopped
    sending us tokens. Surface it so the framework can decide.
    """
    out = [int(v) for v in bytes(raw)]
    if len(out) != world_size:
        return fallback
    decided = [ACTIVE if v == ACTIVE else MASKED for v in out]
    if decided[rank] != ACTIVE:
        from ...errors import MoEEpRankEvictedError

        raise MoEEpRankEvictedError(
            f"rank {rank} was masked out by its peers during mask "
            f"reconciliation (epoch {epoch}); the agreed mask is {decided}. "
            "This rank stalled long enough for the survivors' kernels to time "
            "out on it and is no longer receiving tokens. Tear this worker "
            "down, or rejoin via a fresh Fleet after the survivors call "
            "clear_faults(readmit=True)."
        )
    return decided


class FaultToleranceMixin:
    """Default ``reconcile_active_mask`` for Fleets that can mask ranks.

    Backends mix this in and supply ``_ft_store()``, ``_ft_knob()``, plus the
    ``query_active_mask`` / ``set_active_mask`` pair; the reconciliation
    protocol itself stays transport-free above.
    """

    def reconcile_active_mask(
        self, *, timeout_s: float | None = None
    ) -> "torch.Tensor":
        import torch

        knob = self._ft_knob()  # type: ignore[attr-defined]
        store = self._ft_store()  # type: ignore[attr-defined]
        rank = self._bootstrap.rank  # type: ignore[attr-defined]
        world = self._bootstrap.world_size  # type: ignore[attr-defined]

        local = [int(v) for v in self.query_active_mask().cpu().tolist()]  # type: ignore[attr-defined]
        agreed = reconcile_masks_via_store(
            store,
            rank=rank,
            world_size=world,
            local_mask=local,
            epoch=self.active_mask_epoch,  # type: ignore[attr-defined]
            timeout_s=timeout_s if timeout_s is not None else knob.reconcile_timeout_s,
            takeover_s=knob.coordinator_takeover_s,
        )
        self.set_active_mask(agreed)  # type: ignore[attr-defined]
        return torch.tensor(agreed, dtype=torch.int32, device="cuda")


__all__ = [
    "ACTIVE",
    "MASKED",
    "FaultToleranceMixin",
    "reconcile_masks_via_store",
    "reject_graph_capture",
]
