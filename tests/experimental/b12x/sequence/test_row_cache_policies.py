"""Per-request contract tests for the shipped S3-FIFO row-cache index.

The PR ships ONE implementation: the C extension
``flashinfer/experimental/b12x/loader/_row_cache.c``, reached through the
thin Python facade ``b12x.sequence._shared.row_cache.S3FifoRowCache``. The
behavioral oracle is the pure-Python class that mirrored libCacheSim's
``S3FIFO.c`` and shipped in ``row_cache.py`` until this cutover; it is kept
here, verbatim, as ``_PyS3FifoReference`` — its old home of shipped code is
gone, and these tests are its only remaining copy. The libCacheSim
``cachesim`` replay gate (``probe/s3fifo_parity.py``) stays parent-owned;
this file's gate is C-vs-oracle, request by request: identical hit/miss
counters, identical ``RowPlan`` fields, identical resident/ghost/queue state,
and identical staged and pooled bytes after every served batch.

Everything runs without a GPU and without torch: the oracle is stdlib and the
C module binds plain numpy buffers, so the parity arms exercise the exact
byte-moving code the host tier uses on the device.
"""

from __future__ import annotations

import operator
from collections import deque
import random

import numpy as np
import pytest

from b12x.sequence._shared.row_cache import RowPlan

# ---------------------------------------------------------------------------
# The oracle: moved verbatim from flashinfer/experimental/b12x/sequence/
# _shared/row_cache.py (git history of feat/b12x-sieve-row-cache) for the
# single-policy PR. Do not "clean up" the references to its deleted siblings
# in the docstrings below: they quote the shipped code as it was validated
# against ``cachesim`` (see docs/DESIGN.md §1 "Policy selection").
# ---------------------------------------------------------------------------


class _PyS3FifoReference:
    """Fixed-capacity S3-FIFO over the same slot pool.

    A drop-in sibling of :class:`SieveIndex`: the constructor and the
    ``get``/``insert``/``plan``/``is_ready``/``reserve``/``confirm``/``cancel``
    contract match, so ``DiskRowCache`` may run either replacement policy over
    one payload pool. Slots are pool indices ``0..capacity-1`` and the queues
    are sequences of slots, so a slot handed out by ``insert`` keeps addressing
    the row its key owns.

    The bookkeeping mirrors libCacheSim's ``S3FIFO.c`` (the post-2024 variant,
    which is what the ``cachesim S3FIFO-0.1000-2`` reference numbers are), so
    parity with the C is the contract, not just the paper's:

    * small ``S`` = 10% of capacity, main ``M`` = the rest, ghost ``G`` (keys
      only) = 90% of capacity, each a FIFO.
    * A hit raises the 2-bit counter in whichever queue holds the row; a miss
      removes the key from ``G`` and remembers ``hit_on_ghost``. A resident
      hit never touches ``G``.
    * Admission: a ghost miss joins ``M`` directly; otherwise ``S`` — except
      that before the cache has ever evicted, an already-full ``S`` admits to
      ``M`` instead (the C's ``has_evicted`` fast path).
    * Recycling ``S`` promotes to ``M`` when the counter reaches
      ``move-to-main-threshold`` (2), *keeping* the counter; anything else
      demotes to ``G`` and frees its slot. Recycling ``M`` is a 2-bit clock:
      front rows with credit are reinserted with ``min(freq, 3) - 1`` until
      one without credit leaves.
    * The eviction source is ``M`` when ``M`` is over budget or ``S`` is
      empty, else ``S``; a promotion that overfills ``M`` is paid back from
      ``M`` at once, because the next eviction would pick ``M`` anyway.

    Loading (in-flight prefetch) slots are separate state, exactly as in
    :class:`SieveIndex`: a reserved key is invisible to ``get``, ``plan`` and
    ``is_ready``, its slot joins no queue, and no queue pop can reclaim one.
    A confirmed prefetch is admitted as a fresh row -- into ``M`` when its key
    is a ghost, else ``S`` (or ``M`` under the pre-first-eviction fast path).
    """

    _FREE = 0
    _LOADING = 1
    _READY = 2

    _UNQUEUED = 0
    _SMALL = 1
    _MAIN = 2

    # 2-bit saturation, and the capacity percentages of S and of the ghost
    # list, matching the C's DEFAULT_CACHE_PARAMS
    # ("small-size-ratio=0.10,ghost-size-ratio=0.90,move-to-main-threshold=2").
    _MAX_FREQ = 3
    _SMALL_SHARE = 10
    _GHOST_SHARE = 90
    _MOVE_TO_MAIN_THRESHOLD = 2

    def __init__(self, capacity: int) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        self.capacity = capacity
        # A 10% share rounds up to at least one row, so even a tiny cache has
        # a small queue to filter one-hit rows; M takes what is left. A
        # capacity of one leaves M with no budget at all, and the code below
        # treats an empty budget as "no promotions, no ghost fast-path"
        # rather than as a queue that may transiently hold a row nobody owns.
        self._small_cap = max(1, capacity * self._SMALL_SHARE // 100)
        self._main_cap = capacity - self._small_cap
        self._ghost_cap = capacity * self._GHOST_SHARE // 100
        self._slot: dict[int, int] = {}
        self._key = [0] * capacity
        self._freq = bytearray(capacity)
        # Per-slot state: _FREE / _LOADING / _READY.
        self._state = bytearray(capacity)
        # Per-slot queue tag. A loading slot is _UNQUEUED, which is what keeps
        # an in-flight prefetch out of reach of every pop.
        self._queue = bytearray(capacity)
        # key -> slot for rows a prefetch reserved but has not committed. They
        # are deliberately kept out of ``_slot`` so ``get`` and ``plan`` miss.
        self._loading: dict[int, int] = {}
        # Slots freed by a queue pop or cancel(); handed out before ``_used``
        # grows, so recycled rows and dropped prefetches do not leak capacity.
        self._free: list[int] = []
        self._small: deque[int] = deque()
        self._main: deque[int] = deque()
        # Ghost keys in demotion order, as a dict for O(1) membership: every
        # miss asks whether its key is a ghost.
        self._ghost: dict[int, None] = {}
        # Set once the cache has evicted its first row; gates the C's
        # "S full before first eviction admits to M" fast path.
        self._has_evicted = False
        self._used = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        # Occupied slots, not the high-water mark: a cancelled loading slot
        # sits in the free pool and must not count as resident.
        return len(self._slot) + len(self._loading)

    def __contains__(self, key: int) -> bool:
        # Resident means queued: S or M. A ghost is not resident, and neither
        # is a row whose prefetch is still in flight.
        return key in self._slot

    def _bind(self, slot: int, key: int, *, main: bool) -> None:
        """Make ``slot`` the ready home of ``key``, at the tail of one queue."""
        self._slot[key] = slot
        self._key[slot] = key
        self._freq[slot] = 0
        self._state[slot] = self._READY
        self._queue[slot] = self._MAIN if main else self._SMALL
        (self._main if main else self._small).append(slot)

    def _prepare_admission(self, *, main: bool) -> None:
        """Evict until the pool can take one more committed row.

        ``main`` is the row's *planned* destination: M's overfill test is
        evaluated as if the newcomer had already been appended, reproducing
        the C's single ``S3FIFO_evict`` call (the parent checks
        ``occupied_main > size_main`` with the insert pending). Loading slots
        are excluded -- they are neither queued nor counted by the C's
        sub-cache occupancy -- and a release that leaves a free slot ends the
        loop, exactly as the C's ``while (space_after_add > cache_size)``.
        """
        while len(self._slot) >= self.capacity:
            self._evict_one(main=main)
            if self._free:
                return

    def _settle_main_budget(self) -> None:
        """Charge M's overfill the pop C defers to its next eviction.

        The C allows M to sit one over budget between an M admission and the
        next ``S3FIFO_evict``; settling right after the bind keeps the queue
        budgets a hard invariant with the same victim -- the newcomer is at
        the tail, so it is never the popped row.
        """
        while len(self._main) > self._main_cap:
            self._pop_main()

    def _release(self, slot: int) -> None:
        """Drop ``slot``'s row and hand the pool index back."""
        del self._slot[self._key[slot]]
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._FREE
        self._free.append(slot)

    def _demote_ghost(self, key: int) -> None:
        self._ghost[key] = None
        while len(self._ghost) > self._ghost_cap:
            self._ghost.pop(next(iter(self._ghost)))

    def _pop_small(self) -> None:
        """Recycle S's front: threshold credit promotes into M keeping its
        counter, anything else demotes to a ghost and frees its slot.

        Promotions loop (the C's ``while !has_evicted``): only a demotion --
        or an exhausted S -- frees a pool slot for the caller.
        """
        while self._small:
            slot = self._small.popleft()
            if self._freq[slot] >= self._MOVE_TO_MAIN_THRESHOLD and self._main_cap:
                self._queue[slot] = self._MAIN
                self._main.append(slot)
                # Overfilling M is paid back now: the C's next eviction would
                # pick M anyway (``occupied > size``), so settling at the
                # budget is equivalent and keeps the queues testable.
                while len(self._main) > self._main_cap:
                    self._pop_main()
                continue
            key = self._key[slot]
            self._release(slot)
            self._demote_ghost(key)
            return

    def _pop_main(self) -> None:
        """2-bit clock over M: front rows with credit are reinserted with
        ``min(freq, 3) - 1`` until one without credit leaves."""
        while self._main:
            slot = self._main.popleft()
            freq = self._freq[slot]
            if freq >= 1:
                self._freq[slot] = min(freq, self._MAX_FREQ) - 1
                self._main.append(slot)
                continue
            self._release(slot)
            return

    def _evict_one(self, *, main: bool = False) -> None:
        """One eviction, choosing the source exactly as S3FIFO_evict does.

        ``main`` is the newcomer's planned destination: the C evaluates
        ``occupied_main > size_main`` with the pending insert counted, so an
        M admission at full M budget picks M, not S.
        """
        self._has_evicted = True
        if len(self._main) + main > self._main_cap or not self._small:
            self._pop_main()
        else:
            self._pop_small()

    def get(self, key: int) -> int | None:
        # A pure lookup: never consumes the ghost entry. In the C, the ghost
        # is removed by the miss path's find() only to raise hit_on_ghost, and
        # admission (insert) is where it is consumed; a lookup must not erase
        # the ghost fast path.
        slot = self._slot.get(key)
        if slot is None:
            return None
        if self._freq[slot] < self._MAX_FREQ:
            # One counter table serves both queues: S reads it to decide
            # promotion, M reads it to decide eviction or reinsertion.
            self._freq[slot] += 1
        return slot

    def is_ready(self, key: int) -> bool:
        """Whether ``key`` holds a committed (READY) row."""
        return key in self._slot

    def reserve(self, key: int) -> int | None:
        """Claim a loading slot for ``key``; ``None`` if it cannot be taken.

        A row already committed or already loading is a no-op (``None``), as is
        one that cannot be evicted for without stripping the last ready row.
        The slot joins neither queue, so no pop can reclaim an in-flight
        prefetch and no lookup can read a half-filled row.
        """
        if key in self._slot or key in self._loading:
            return None
        slot = self._claim_for_loading()
        if slot is None:
            return None
        self._key[slot] = key
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._LOADING
        self._loading[key] = slot
        return slot

    def _claim_for_loading(self) -> int | None:
        if self._free:
            return self._free.pop()
        if self._used < self.capacity:
            slot = self._used
            self._used += 1
            return slot
        if len(self._slot) <= 1:
            # Keep at least one committed row evictable so the synchronous
            # read path can never starve behind a queue of in-flight prefetches.
            return None
        self._evict_one()
        return self._free.pop() if self._free else None

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        key = self._key[slot]
        if self._state[slot] != self._LOADING:
            return slot
        self._loading.pop(key, None)
        # Room is made against committed rows only -- this slot is already
        # owned, and it sits in no queue. The ghost decision uses the C's
        # find()-time rule (a member of G admits to M), consumed at admission
        # exactly as the C's hit_on_ghost flag.
        main, ghost = self._admit(key)
        self._bind_loaded(slot, key, main=main)
        if ghost:
            self._ghost.pop(key, None)
        self._settle_main_budget()
        return slot

    def _bind_loaded(self, slot: int, key: int, *, main: bool) -> None:
        """Attach an already-owned loading slot as a ready row in one queue."""
        self._slot[key] = slot
        self._key[slot] = key
        self._freq[slot] = 0
        self._state[slot] = self._READY
        self._queue[slot] = self._MAIN if main else self._SMALL
        (self._main if main else self._small).append(slot)

    def _admit(
        self, key: int, *, hit_on_ghost: bool | None = None
    ) -> tuple[bool, bool]:
        """Plan one admission: evict for space, then resolve the queue.

        Returns ``(main, ghost_hit)``. Mirrors S3FIFO.c ordering: the ghost
        fast path is the parent's capacity loop evaluated with the pending
        insert counted into M (``occupied_main > size_main`` picks M); the
        plain path evicts for an S admission and only then reads the
        ``has_evicted`` gate, as insert() runs after the loop.
        """
        if hit_on_ghost is None:
            hit_on_ghost = key in self._ghost
        if hit_on_ghost and self._main_cap > 0:
            self._prepare_admission(main=True)
            return True, True
        self._prepare_admission(main=False)
        return self._ghost_admits_main(key) and self._main_cap > 0, hit_on_ghost

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        if self._state[slot] != self._LOADING:
            return
        self._loading.pop(self._key[slot], None)
        self._freq[slot] = 0
        self._queue[slot] = self._UNQUEUED
        self._state[slot] = self._FREE
        self._free.append(slot)

    def insert(self, key: int, *, hit_on_ghost: bool | None = None) -> int | None:
        """Admit ``key`` as a fresh row; ``None`` when no slot can be taken.

        ``hit_on_ghost`` is the C's latched flag: ``None`` (the direct API)
        reads the ghost set here, while :meth:`plan` passes the value observed
        at lookup time, before any eviction may have re-demoted this key.
        """
        if key in self._loading:
            # The synchronous path is finishing a row a prefetch is also
            # fetching: adopt the reserved slot so the two never own separate
            # copies of the same id.
            return self.confirm(self._loading[key])
        if key in self._slot:
            raise KeyError("row is already cached")
        main, ghost = self._admit(key, hit_on_ghost=hit_on_ghost)
        if ghost:
            self._ghost.pop(key, None)
        slot = self._free.pop() if self._free else None
        if slot is None:
            if self._used < self.capacity:
                slot = self._used
                self._used += 1
            else:
                return None
        self._bind(slot, key, main=main)
        self._settle_main_budget()
        return slot

    def _ghost_admits_main(self, key: int) -> bool:
        """The C's non-ghost rule: S already full before the first eviction."""
        return not self._has_evicted and len(self._small) >= self._small_cap

    def plan(self, ids) -> RowPlan:
        """Resolve one batch; duplicates in the batch fetch and store once."""
        hit_pos: list[int] = []
        hit_slot: list[int] = []
        miss_ids: list[int] = []
        miss_pos: list[int] = []
        miss_row: list[int] = []
        fetched: dict[int, int] = {}
        hit_on_ghost: dict[int, bool] = {}
        for position, key in enumerate(ids):
            slot = self.get(key)
            if slot is not None:
                hit_pos.append(position)
                hit_slot.append(slot)
                continue
            row = fetched.get(key)
            if row is None:
                row = fetched[key] = len(miss_ids)
                miss_ids.append(key)
                # Latch the C's find() flag at lookup time: an eviction during
                # this batch's insertions may otherwise re-demote the key and
                # grant it the ghost fast path it did not earn.
                hit_on_ghost[key] = key in self._ghost
            miss_pos.append(position)
            miss_row.append(row)
        self.hits += len(hit_pos)
        self.misses += len(miss_pos)
        # Insert after every lookup, so this batch's hits raise their counters
        # before any of its misses recycles a queue. A slot reassigned within
        # the batch keeps only its final owner. A miss that cannot obtain a
        # slot (every slot still loading) is simply not cached this time; its
        # staging is unaffected, since misses are staged from the fetched
        # planes rather than from the pool.
        stored: dict[int, int] = {}
        for row, key in enumerate(miss_ids):
            slot = self.insert(key, hit_on_ghost=hit_on_ghost[key])
            if slot is not None:
                stored[slot] = row
        return RowPlan(
            hit_pos,
            hit_slot,
            miss_ids,
            miss_pos,
            miss_row,
            list(stored),
            list(stored.values()),
        )


# ---------------------------------------------------------------------------
# C-side fixtures
# ---------------------------------------------------------------------------

_LCS_ALL = "~/.cache/vllm-multiarch-oci/probe/lcs_all.txt"
_PLAN_FIELDS = (
    "hit_pos",
    "hit_slot",
    "miss_ids",
    "miss_pos",
    "miss_row",
    "store_slot",
    "store_row",
)


def _c_module():
    """Load the row-cache extension; skip only when no C99 compiler exists."""
    import shutil

    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("no C99 compiler (CC/cc/gcc) on this host")
    from b12x.loader import _row_cache_native

    try:
        return _row_cache_native.load()
    except RuntimeError as exc:  # missing headers etc. on exotic hosts
        pytest.skip(f"row-cache C extension unavailable: {exc}")


def _payload(key: int, width: int) -> bytes:
    # Any deterministic per-key byte pattern: a slot quietly reassigned within
    # a batch, or a store landing on the wrong row, changes the staged bytes.
    return bytes(((key >> (8 * (i % 8))) ^ (i * 31 + 7)) & 0xFF for i in range(width))


class _Pair:
    """One cache driven as oracle (side 0) and as C (side 1) in lockstep.

    ``serve`` mirrors ``DiskRowCache._read_cached``: plan/read, fill the miss
    planes from the "source" (the payload pattern), then stage and store. A
    fake prefetch worker mirrors ``_run_prefetch_job``: pool bytes for a
    reserved slot, then confirm (or cancel), under the same single-threaded
    lock discipline the real cache serialises with.
    """

    def __init__(self, capacity: int, max_lookups: int, weight: int, scale: int):
        native = _c_module()
        self.native = native
        self.capacity = capacity
        self.max_lookups = max_lookups
        self.widths = [weight] + ([scale] if scale else [])
        self.ref = _PyS3FifoReference(capacity)
        alloc = lambda rows, width: np.zeros((rows, width), np.uint8)  # noqa: E731
        self.ids = [np.zeros(max_lookups, np.int64) for _ in range(2)]
        self.miss_ids = [np.zeros(max_lookups, np.int64) for _ in range(2)]
        self.miss = [[alloc(max_lookups, w) for w in self.widths] for _ in range(2)]
        self.stage = [[alloc(max_lookups, w) for w in self.widths] for _ in range(2)]
        self.pool = [[alloc(capacity, w) for w in self.widths] for _ in range(2)]
        self.rc = native.row_cache_create(
            capacity,
            max_lookups,
            memoryview(self.ids[1]),
            memoryview(self.miss_ids[1]),
            tuple(memoryview(plane) for plane in self.miss[1]),
            tuple(memoryview(plane) for plane in self.stage[1]),
            tuple(memoryview(plane) for plane in self.pool[1]),
        )

    def _fill_miss(self, side: int, keys: list[int]) -> None:
        for row, key in enumerate(keys):
            for plane, width in zip(self.miss[side], self.widths):
                plane[row][:] = np.frombuffer(_payload(key, width), np.uint8)

    def serve(self, batch: list[int]) -> RowPlan:
        """One batch through both implementations; returns the oracle plan."""
        plan = self.ref.plan(list(batch))
        self._fill_miss(0, plan.miss_ids)
        for pos, slot in zip(plan.hit_pos, plan.hit_slot):
            for stage, pool in zip(self.stage[0], self.pool[0]):
                stage[pos] = pool[slot]
        for pos, row in zip(plan.miss_pos, plan.miss_row):
            for stage, fetched in zip(self.stage[0], self.miss[0]):
                stage[pos] = fetched[row]
        for slot, row in zip(plan.store_slot, plan.store_row):
            for pool, fetched in zip(self.pool[0], self.miss[0]):
                pool[slot] = fetched[row]

        n = len(batch)
        if n:
            self.ids[1][:n] = np.asarray(batch, np.int64)
        self.native.row_cache_read(self.rc, n)
        self._fill_miss(1, plan.miss_ids)
        self.native.row_cache_store(self.rc)
        arrays = self.native.row_cache_plan_arrays(self.rc)
        for field in _PLAN_FIELDS:
            assert getattr(plan, field) == arrays[field], (field, plan, arrays)
        self.compare(step=str(batch[:4]))
        return plan

    def reserve(self, key: int) -> tuple[int | None, int | None]:
        ref_slot = self.ref.reserve(key)
        c_slot = self.native.row_cache_reserve(self.rc, key)
        assert ref_slot == c_slot, ("reserve", key, ref_slot, c_slot)
        return ref_slot, c_slot

    def cancel(self, slot: int) -> None:
        self.ref.cancel(slot)
        self.native.row_cache_cancel(self.rc, slot)

    def confirm(self, key: int, slot: int) -> None:
        # Mirror _run_prefetch_job: the worker writes the fetched bytes into
        # the reserved slot (both pools identically) before confirming.
        for side in (0, 1):
            for pool, width in zip(self.pool[side], self.widths):
                pool[slot][:] = np.frombuffer(_payload(key, width), np.uint8)
        assert self.ref.confirm(slot) == self.native.row_cache_confirm(self.rc, slot)

    def compare(self, step: object) -> None:
        """Full internal-state equality: counters, queues, ghosts, bytes."""
        st = self.native.row_cache_stats(self.rc)
        d = self.native.row_cache_debug_state(self.rc)
        assert (self.ref.hits, self.ref.misses) == (st["hits"], st["misses"]), (
            step,
            "counters",
        )
        assert len(self.ref) == st["resident"], (step, "resident")
        assert (len(self.ref._small), len(self.ref._main), len(self.ref._ghost)) == (
            st["s"],
            st["m"],
            st["g"],
        ), (step, "queue sizes")
        assert [
            (self.ref._key[s], int(self.ref._freq[s])) for s in self.ref._small
        ] == [tuple(x) for x in d["small"]], (step, "small queue")
        assert [(self.ref._key[s], int(self.ref._freq[s])) for s in self.ref._main] == [
            tuple(x) for x in d["main"]
        ], (step, "main queue")
        assert list(self.ref._ghost) == d["ghost"], (step, "ghost order")
        assert sorted(self.ref._slot) == sorted(d["ready"]), (step, "resident keys")
        assert self.ref._loading == d["loading"], (step, "loading map")
        assert self.ref._used == d["used"], (step, "used")
        assert self.ref._free == d["free"], (step, "free stack order")
        assert self.ref._has_evicted == bool(d["has_evicted"]), (step, "has_evicted")
        for side_planes, key in ((self.stage, "stage"), (self.pool, "pool")):
            for p in range(len(self.widths)):
                assert np.array_equal(side_planes[0][p], side_planes[1][p]), (
                    step,
                    key,
                    p,
                )


# ---------------------------------------------------------------------------
# the build itself
# ---------------------------------------------------------------------------


def test_c_extension_builds():
    """The shipped implementation is the C extension: no compiler-free bypass."""
    import shutil

    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("no C99 compiler (CC/cc/gcc) on this host")
    from b12x.loader import _row_cache_native

    path = _row_cache_native._build()
    assert path.is_file() and path.name.startswith("_b12x_row_cache")


def test_build_failure_raises_not_falls_back(monkeypatch):
    """Construction raises when the extension cannot be built (no silent path)."""
    from b12x.loader import _row_cache_native
    from b12x.sequence._shared.row_cache import S3FifoRowCache

    def boom():
        raise RuntimeError("b12x row cache native build failed: simulated")

    monkeypatch.setattr(_row_cache_native, "load", boom)
    alloc = lambda rows, width: np.zeros((rows, width), np.uint8)  # noqa: E731
    with pytest.raises(RuntimeError, match="simulated"):
        S3FifoRowCache(
            8,
            4,
            memoryview(np.zeros(4, np.int64)),
            memoryview(np.zeros(4, np.int64)),
            [memoryview(alloc(4, 3))],
            [memoryview(alloc(4, 3))],
            [memoryview(alloc(8, 3))],
        )


def test_facade_rejects_bad_capacity():
    from b12x.sequence._shared.row_cache import S3FifoRowCache

    alloc = lambda rows, width: np.zeros((rows, width), np.uint8)  # noqa: E731
    args = (
        memoryview(np.zeros(4, np.int64)),
        memoryview(np.zeros(4, np.int64)),
        [memoryview(alloc(4, 3))],
        [memoryview(alloc(4, 3))],
        [memoryview(alloc(4, 3))],
    )
    for bad in (0, -1):
        with pytest.raises(ValueError, match="capacity must be positive"):
            S3FifoRowCache(bad, 4, *args)


# ---------------------------------------------------------------------------
# per-request parity: hand sequences documented against the C reference
# ---------------------------------------------------------------------------


def _pair(capacity, max_lookups=40, weight=4, scale=2):
    return _Pair(capacity, max_lookups, weight, scale)


def test_hand_micro_batches_match_reference():
    """The sequence ``test_s3fifo_small_queue_promotes_hits_and_ghosts_re_admit``
    checked by hand against the libCacheSim C: replayed here as one-id plan
    batches through both implementations, with the documented states
    re-asserted on the C directly (serve() also diffs every step)."""
    pair = _pair(10)
    pair.serve([1])
    pair.serve([1])  # 1 earns one hit in S (freq 1, below the threshold)
    pair.serve([2])
    pair.serve([2])
    for key in range(3, 11):
        pair.serve([key])
    st = pair.native.row_cache_stats(pair.rc)
    assert (st["s"], st["m"], st["g"]) == (1, 9, 0)
    d = pair.native.row_cache_debug_state(pair.rc)
    assert [k for k, _ in d["small"]] == [1]
    assert [k for k, _ in d["main"]] == list(range(2, 11))
    assert d["ghost"] == [] and not d["has_evicted"]
    # 11 evicts S's front (one hit, below threshold): it demotes to a ghost
    # and 11 takes S — the reference's first-eviction step.
    pair.serve([11])
    d = pair.native.row_cache_debug_state(pair.rc)
    assert d["has_evicted"]
    assert [k for k, _ in d["small"]] == [11]
    assert d["ghost"] == [1]
    pair.serve([12])
    d = pair.native.row_cache_debug_state(pair.rc)
    assert [k for k, _ in d["small"]] == [12]
    assert d["ghost"] == [1, 11]
    # A ghost miss is admitted straight to M; the pending insert is counted
    # into M's overfill test, so M — not S — pays: the clock spends 2's
    # credit on one chance and 3 (no credit) leaves. 12, S's front, stays.
    pair.serve([1])
    d = pair.native.row_cache_debug_state(pair.rc)
    assert [k for k, _ in d["small"]] == [12]
    assert [k for k, _ in d["main"]] == [4, 5, 6, 7, 8, 9, 10, 2, 1]
    assert dict(d["main"])[2] == 0
    assert 3 not in d["ready"] and 1 not in d["ghost"]
    assert st["resident"] == 10


def test_threshold_promotion_keeps_counter():
    """A row read twice in S promotes to M at full strength (the reference
    class's second worked example)."""
    pair = _pair(10)
    for _ in range(3):
        pair.serve([1])  # insert, then two hits: freq reaches the threshold
    pair.serve([2])
    pair.serve([2])
    for key in range(3, 12):
        pair.serve([key])
    d = pair.native.row_cache_debug_state(pair.rc)
    assert dict(d["main"])[1] == 2
    assert [k for k, _ in d["main"]] == [4, 5, 6, 7, 8, 9, 10, 1, 2]
    assert [k for k, _ in d["small"]] == [11]


def test_duplicate_ids_fetch_and_store_once():
    """Duplicates in the batch fetch and store once, and hit the same slot."""
    pair = _pair(64)
    plan = pair.serve([7, 7, 8, 7, 8])
    assert plan.miss_ids == [7, 8]
    assert plan.miss_pos == [0, 1, 2, 3, 4]
    assert plan.miss_row == [0, 0, 1, 0, 1]
    assert sorted(set(plan.store_slot)) == sorted(plan.store_slot)
    pair.serve([7, 7])  # now resident: all hits, no stores
    assert pair.ref.hits == pair.native.row_cache_stats(pair.rc)["hits"]


def test_micro_sequence_replay_matches_reference():
    """The 210-request micro trace (the WTinyLFU file's MICRO_SEQ shape) both
    as batch=1 and as one giant batch through both arms. serve() has already
    diffed C-vs-reference per step within each pair; here the two batching
    modes must simply complete identically between their own arms and leave
    the caches in a sane end state. (Batching modes differ from EACH OTHER —
    an in-batch repeat is a deduped miss, not a hit — and that difference is
    itself part of the mirrored plan() contract.)"""
    micro_seq = list(range(1, 202)) + [199, 202, 203, 204, 205, 1, 2, 1]
    pair = _pair(200, max_lookups=len(micro_seq), weight=9, scale=2)
    for key in micro_seq:
        pair.serve([key])
    pair2 = _pair(200, max_lookups=len(micro_seq), weight=9, scale=2)
    pair2.serve(micro_seq)
    st = pair.native.row_cache_stats(pair.rc)
    assert st["hits"] > 0 and st["resident"] == 200
    assert pair.ref._used == pair2.ref._used


# ---------------------------------------------------------------------------
# per-request parity: prefetch lifecycle interleaved with batches
# ---------------------------------------------------------------------------


def test_reserve_confirm_admits_via_ghost_rule():
    """A confirmed prefetch is admitted as a fresh row: into M when its key is
    a ghost, else S — the same rule plan() applies, and the slot number the
    reference returns must match bit-for-bit (free-list LIFO included)."""
    pair = _pair(10)
    for key in list(range(1, 11)):
        pair.serve([key])
    pair.serve([11])  # evicts S's front to a ghost
    d = pair.native.row_cache_debug_state(pair.rc)
    ghost_key = d["ghost"][0]
    slot_ref, slot_c = pair.reserve(ghost_key)
    assert slot_ref is not None and slot_ref == slot_c
    assert not pair.ref.is_ready(ghost_key)
    assert not pair.native.row_cache_is_ready(pair.rc, ghost_key)
    pair.confirm(ghost_key, slot_c)
    assert pair.ref.is_ready(ghost_key)
    d = pair.native.row_cache_debug_state(pair.rc)
    assert ghost_key in d["ready"] and ghost_key not in d["ghost"]
    # Cancel path: a dropped reservation returns its slot to the free pool and
    # leaks no capacity.
    slot_ref, slot_c = pair.reserve(5555)
    pair.cancel(slot_c)
    assert pair.native.row_cache_debug_state(pair.rc)["loading"] == {}
    again_ref, again_c = pair.reserve(5555)
    assert again_ref == again_c == slot_c  # LIFO reuse, like _free.pop()


def test_random_churn_with_prefetch_lifecycle():
    """200 batches of random access with reserve/cancel/confirm interleaved
    (the invariant suite the Python class shipped with, now driving C):
    budgets hold, queues stay disjoint, one owner per slot, and every
    per-request comparison above holds after EVERY step."""
    for capacity, seed, scale in ((64, 1234, 2), (200, 99, 0), (7, 5, 3)):
        pair = _pair(capacity, max_lookups=40, weight=4, scale=scale)
        rng = random.Random(seed)
        hot = [7, 8, 150, 151, 289, 6, 290, 2**40]
        for batch_index in range(200):
            ids = [
                rng.choice(hot) if rng.random() < 0.6 else rng.randrange(300)
                for _ in range(rng.choice([1, 5, 17, 40]))
            ]
            pair.serve(ids)
            if batch_index % 7 == 3:
                key = 1000 + batch_index
                ref_slot, c_slot = pair.reserve(key)
                if c_slot is not None:
                    assert pair.native.row_cache_get(pair.rc, key) is None
                    assert not pair.native.row_cache_is_ready(pair.rc, key)
                    if batch_index % 3 == 0:
                        pair.cancel(c_slot)
                    else:
                        pair.confirm(key, c_slot)
                        assert pair.native.row_cache_is_ready(pair.rc, key)
        # Queue invariants the reference class shipped with: budgets hold and
        # the two arms finish in identical state (compare() checks the rest).
        st = pair.native.row_cache_stats(pair.rc)
        assert st["s"] <= st["small_cap"]
        assert st["m"] <= st["main_cap"]
        assert st["g"] <= st["ghost_cap"]
        pair.compare(("churn", capacity, seed))


# ---------------------------------------------------------------------------
# per-request parity: the production trace
# ---------------------------------------------------------------------------


def _lcs_all_prefix(count: int) -> list[int] | None:
    from pathlib import Path

    path = Path(_LCS_ALL).expanduser()
    if not path.is_file():
        return None
    ids: list[int] = []
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            # strtol semantics: the trace's negative obj-id rows are KEPT
            # (dropping them desynchronised an earlier replay series; see
            # docs/DESIGN.md §1 "Policy selection").
            ids.append(int(text))
            if len(ids) >= count:
                break
    return ids


def test_lcs_all_prefix_matches_reference():
    """The first 200k ids of the production PLE row-id trace, batched at the
    serving width, replayed through both arms: identical plan arrays per
    batch, full internal state compared every 997 batches and at the end."""
    ids = _lcs_all_prefix(200_000)
    if ids is None:
        pytest.skip(f"production trace corpus not present ({_LCS_ALL})")
    for capacity, weight, scale in ((1024, 9, 2), (10_000, 9, 2)):
        pair = _pair(capacity, max_lookups=40, weight=weight, scale=scale)
        for index in range(0, len(ids), 40):
            pair.serve(ids[index : index + 40])
            if index % (40 * 997) == 0:
                pair.compare(("lcs", capacity, index))
        st = pair.native.row_cache_stats(pair.rc)
        assert st["hits"] + st["misses"] == len(ids)


def test_stats_surface():
    pair = _pair(16)
    pair.serve([1, 2, 3, 1])  # the repeated 1 is a deduped miss, not a hit
    st = pair.native.row_cache_stats(pair.rc)
    assert set(st) >= {"hits", "misses", "resident", "s", "m", "g"}
    assert st["hits"] == 0 and st["misses"] == 4
    pair.serve([1])  # now resident: one hit, no stores
    assert pair.native.row_cache_stats(pair.rc)["hits"] == 1


# ---------------------------------------------------------------------------
# environment knobs (pure functions; DiskRowCache wiring tests live in
# test_disk_row_cache.py because construction requires CUDA)
# ---------------------------------------------------------------------------


def test_resolve_row_cache_size_precedence():
    from b12x.sequence._shared.disk_table import _resolve_row_cache_size

    # Nothing set: off, exactly the historical default.
    assert _resolve_row_cache_size(None, None, None, 90) == (0, False)
    assert _resolve_row_cache_size(None, "", None, 90) == (0, False)
    # Explicit ROWS keeps the off-by-default when zero.
    assert _resolve_row_cache_size(None, "0", None, 90) == (0, False)
    assert _resolve_row_cache_size(None, "16", None, 90) == (16, False)
    # An empty B12X_ROW_CACHE_BYTES is unset (like TIER/ROWS): never a parse
    # error, and never a collision with a set ROWS.
    assert _resolve_row_cache_size("", None, None, 90) == (0, False)
    assert _resolve_row_cache_size("", "16", None, 90) == (16, False)
    # BYTES wins as a size; both set at once is the same error as before.
    assert _resolve_row_cache_size("1GiB", None, None, 90) == (11_930_464, False)
    with pytest.raises(ValueError, match="only one of"):
        _resolve_row_cache_size("1GiB", "16", None, 90)
    # Enabled without a size: 1 GiB, rounded down to whole rows, defaulted
    # flag set (which is what selects the C path and the s3fifo default).
    assert _resolve_row_cache_size(None, None, "host", 90) == (11_930_464, True)
    assert _resolve_row_cache_size(None, "", "host", 90) == (11_930_464, True)
    # A 1 GiB budget below one row is an error, never a silent off.
    with pytest.raises(ValueError, match="rounds to zero rows"):
        _resolve_row_cache_size(None, None, "host", (1 << 30) + 1)


def test_validate_row_cache_tier():
    from b12x.sequence._shared.disk_table import _validate_row_cache_tier

    assert _validate_row_cache_tier(None) == "host"
    assert _validate_row_cache_tier("") == "host"
    assert _validate_row_cache_tier(" Host ") == "host"
    assert _validate_row_cache_tier("   ") == "host"
    # Single host tier: any other value is invalid.
    for bad in ("unified", "ram", "gpu"):
        with pytest.raises(ValueError, match="B12X_ROW_CACHE_TIER"):
            _validate_row_cache_tier(bad)
