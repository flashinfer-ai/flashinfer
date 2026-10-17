"""Replacement policy for immutable embedding-table rows: one C S3-FIFO.

Hashed n-gram tables (PLE, Engram) are read row-by-row with a Zipfian access
pattern: on llm-inference-bench traces of Qwen3.8-Flash-Next PLE, ~65% of row
reads repeat and a 10^4-10^5 row working set captures most of the reuse.
S3-FIFO (Yang et al., SOSP'23) keeps a small frequency-filtered FIFO, a main
FIFO that reinserts rows still holding counter credit, and a ghost list of
recently demoted keys; at every operating capacity it led the measured
comparison (see ``docs/DESIGN.md`` §1), so it is the single shipped
policy and its hot path is the C extension
``flashinfer/experimental/b12x/loader/_row_cache.c``.

``S3FifoRowCache`` is the thin Python facade around that extension: it owns
no bookkeeping itself, only the capsule and the bound host buffers, so
``DiskRowCache`` imports exactly this class and never sees the PyCapsule. The
SIEVE and W-TinyLFU Python indices that used to live in this module were
experiments; the rationale is in ``docs/DESIGN.md`` §1 "Policy selection", and
the Python class that mirrored
libCacheSim's ``S3FIFO.c`` — the behavioural oracle the C is validated
against per request — survives test-private as ``_PyS3FifoReference`` in
``tests/experimental/b12x/sequence/test_row_cache_policies.py``.

The facade owns only the id->slot mapping. Payload placement and the miss
source (disk or a host-resident table) belong to ``DiskRowCache``.

Every slot lives in one of three states. ``FREE`` slots are unclaimed. A
speculative prefetch ``reserve``\\ s a ``LOADING`` slot for a row it is still
fetching: a loading slot consumes capacity, is never an eviction victim, and
is never returned by ``get`` so an in-flight prefetch is invisible to the
read path. ``confirm`` promotes a loading slot to ``READY``, the only state the
synchronous read path observes. ``cancel`` drops a loading reservation
without touching the queue. ``DiskRowCache.prefetch`` drives that lifecycle;
ordinary reads only ever see ``READY`` slots.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import operator


@dataclass
class RowPlan:
    """One batch resolved against the index.

    ``hit_pos[i]`` reads cached slot ``hit_slot[i]``. ``miss_ids`` are the
    distinct uncached rows to fetch, in fetch order; position ``miss_pos[i]``
    receives fetched row ``miss_row[i]``. ``store_slot[i]`` receives fetched
    row ``store_row[i]`` after every hit has been read, so an eviction made
    for this batch never clobbers a row the batch still reads.
    """

    hit_pos: list[int]
    hit_slot: list[int]
    miss_ids: list[int]
    miss_pos: list[int]
    miss_row: list[int]
    store_slot: list[int]
    store_row: list[int]


class S3FifoRowCache:
    """Fixed-capacity S3-FIFO over host pool slots ``0..capacity-1``, in C.

    All state lives in the ``_b12x_row_cache`` extension, which mirrors
    libCacheSim's ``S3FIFO.c`` (the post-2024 variant the ``cachesim
    S3FIFO-0.1000-2`` reference numbers are): small ``S`` = 10% of capacity,
    main ``M`` = the rest, ghost ``G`` (keys only) = 90% of capacity; 2-bit
    frequency counters, ``move-to-main-threshold`` 2, the ``has_evicted``
    fast path, and the main-budget settle after every bind. Parity with the
    Python oracle is the contract and is gated per request in
    ``test_row_cache_policies.py``.

    The batched read path is split around the source fetch so there is no
    Python per-id work: ``read(count)`` resolves the rows already staged in
    the bound ids buffer (lookups, counter/ghost/queue mutation, and the hit
    copies from the bound pool into the bound staging planes) and writes the
    distinct miss ids into the bound miss-ids plane; the caller then fills
    the bound miss planes from the source and calls ``store()`` for the miss
    staging and the pool stores. ``plan`` keeps the old single-call surface
    (it stages ``ids`` into the ids buffer, runs ``read``, and returns the
    ``RowPlan``) for tests and diagnostics; the read path must use
    read/store.

    Like the oracle, this class is not thread-safe by itself: DiskRowCache
    serialises every mutator (the transaction on the read path, ``_lock``
    around the prefetch worker's reserve/confirm/cancel sections), the same
    discipline the Python class relied on.

    Every constructor buffer is bound for the object's lifetime (the C keeps
    the buffer references); the caller must keep the underlying allocations
    alive and untouched except through these methods. ``close`` drops the
    capsule and with it the extension's internal arrays.
    """

    # Numerically identical to the oracle's _FREE/_LOADING/_READY.
    _FREE = 0
    _LOADING = 1
    _READY = 2

    def __init__(
        self,
        capacity: int,
        max_lookups: int,
        ids,
        miss_ids,
        miss_planes,
        stage_planes,
        pool_planes,
    ) -> None:
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("row cache capacity must be positive")
        max_lookups = operator.index(max_lookups)
        if max_lookups <= 0:
            raise ValueError("row cache batch capacity must be positive")
        from b12x.loader import _row_cache_native

        native = _row_cache_native.load()  # raises on build failure; no fallback
        self._native = native
        self._ids_view = ids
        self._rc = native.row_cache_create(
            capacity,
            max_lookups,
            ids,
            miss_ids,
            tuple(miss_planes),
            tuple(stage_planes),
            tuple(pool_planes),
        )
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def _capsule(self):
        if self._rc is None:
            raise RuntimeError("row cache is closed")
        return self._rc

    def read(self, count: int) -> tuple[int, int]:
        """Resolve the ids already staged in the bound buffer; copy hits.

        Returns ``(hit_positions, distinct_miss_ids)``; the distinct miss ids
        have been written to the bound miss-ids buffer for the caller's
        fetch, and the batch's store phase is completed by ``store()``.
        """
        capsule = self._capsule()
        return self._native.row_cache_read(capsule, operator.index(count))

    def store(self) -> None:
        """Stage the batch's misses and write its stores into the pool."""
        capsule = self._capsule()
        self._native.row_cache_store(capsule)

    def plan(self, ids) -> RowPlan:
        """Resolve one batch through the C and return its ``RowPlan``.

        Diagnostic and test surface: it copies ``ids`` into the bound ids
        buffer (per-id Python work) and then behaves exactly like ``read``;
        the caller must still fill the miss planes from the source and call
        ``store()`` to complete the batch.
        """
        capsule = self._capsule()
        values = [operator.index(value) for value in ids]
        count = len(values)
        if count:
            # memoryview slice assignment requires a same-format bytes-like
            # rvalue; numpy int64 views carry the platform signed-long format
            # ('l' on LP64), so echo the bound view's own format.
            self._ids_view[:count] = array(self._ids_view.format or "l", values)
        self._native.row_cache_read(capsule, count)
        return RowPlan(**self._native.row_cache_plan_arrays(capsule))

    def get(self, key: int) -> int | None:
        """Resident (READY) slot for ``key``, raising its counter; else None."""
        capsule = self._capsule()
        return self._native.row_cache_get(capsule, operator.index(key))

    def is_ready(self, key: int) -> bool:
        """Whether ``key`` holds a committed (READY) row."""
        capsule = self._capsule()
        return self._native.row_cache_is_ready(capsule, operator.index(key))

    def is_loading(self, slot: int) -> bool:
        """Whether ``slot`` is an in-flight prefetch (state LOADING)."""
        capsule = self._capsule()
        return (
            self._native.row_cache_state(capsule, operator.index(slot)) == self._LOADING
        )

    def reserve(self, key: int) -> int | None:
        """Claim a loading slot for ``key``; ``None`` if it cannot be taken."""
        capsule = self._capsule()
        return self._native.row_cache_reserve(capsule, operator.index(key))

    def confirm(self, slot: int) -> int:
        """Promote a loading slot to ready; idempotent if already committed."""
        capsule = self._capsule()
        return self._native.row_cache_confirm(capsule, operator.index(slot))

    def cancel(self, slot: int) -> None:
        """Drop a loading reservation, returning the slot to the free pool."""
        capsule = self._capsule()
        self._native.row_cache_cancel(capsule, operator.index(slot))

    def stats(self) -> dict:
        """hits/misses/resident counters plus the S/M/G queue sizes."""
        capsule = self._capsule()
        return self._native.row_cache_stats(capsule)

    def plan_arrays(self) -> dict:
        """The last read's RowPlan fields, as lists (tests and diagnostics)."""
        capsule = self._capsule()
        return self._native.row_cache_plan_arrays(capsule)

    def debug_state(self) -> dict:
        """Queue contents, ghost order, and resident/loading keys.

        ``(key, freq)`` pairs in queue order for ``small``/``main``, ghost
        keys in demotion order for ``ghost``, resident keys in ``ready``, and
        ``loading`` as a key->slot map — the surfaces the Python oracle
        exposed through privates, for the per-request parity tests.
        """
        capsule = self._capsule()
        return self._native.row_cache_debug_state(capsule)

    @property
    def hits(self) -> int:
        return self.stats()["hits"]

    @property
    def misses(self) -> int:
        return self.stats()["misses"]

    def __len__(self) -> int:
        # Occupied slots: a cancelled loading slot must not count as resident.
        return self.stats()["resident"]

    def __contains__(self, key: int) -> bool:
        return self.is_ready(key)

    def close(self) -> None:
        self._rc = None
        self._native = None
        self._ids_view = None
