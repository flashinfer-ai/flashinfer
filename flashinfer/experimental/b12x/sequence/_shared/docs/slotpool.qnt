/**
 * S3-FIFO row-cache reclamation: deferred-free / use-after-free safety.
 *
 * The row cache has two distinct planes, each with its own lifecycle, and this
 * model proves the reclamation safety of both. It deliberately does NOT claim
 * the C guards pool eviction against in-use slots — it does not (there is no
 * staged-from set consulted by rc_pop_small / rc_pop_main). What makes the
 * design safe is two structural facts plus a batch fence, all modelled here:
 *
 * POOL PLANE (the persistent host rows, rc_entry packed state|queue):
 *   A slot with an in-flight fetch (LOADING, set by rc_reserve /
 *   rc_claim_for_loading) joins no ring, so the eviction pops — which walk only
 *   the SMALL/MAIN rings of READY slots — can never recycle it: "no pop can
 *   reach it" (_row_cache.c:30-31). And claim only takes FREE slots, so a
 *   LOADING slot is never re-handed to a second fetch. This is enforced
 *   structurally by the state machine, not by an eviction-time check, so it is
 *   not the DEFER_RECYCLE lever.
 *
 * STAGING PLANE (the batch-sized buffer the decode gathers rows into):
 *   A hit row is copied from its pool slot into the staging buffer (rc_copy_row,
 *   _row_cache.c:1089-1092) and the GPU decode reads the STAGING bytes, not the
 *   pool slot. The pool slot may therefore be evicted and rebound later in the
 *   same batch without harming the decode, because store() rewrites pool bytes
 *   only AFTER read() has copied every hit out (_row_cache.c:33-35,1106-1108)
 *   and DiskRowCache._lock serialises the prefetch worker's pool writes with the
 *   transaction. The genuine use-after-free is thus on the SHARED STAGING
 *   BUFFER: the next batch must not overwrite a staging slot while an
 *   outstanding, not-yet-retired decode from the previous transaction is still
 *   reading it. disk_table.py transaction() / _cache_done (the next
 *   transaction's stream.wait_event) fences that reuse.
 *
 * DEFER_RECYCLE is the code-level lever on that fence: when true, an overwrite
 * of a staging slot still pinned by an outstanding decode is refused; when
 * false (the broken arm) it happens and the decode reads bytes swapped under it.
 *
 * There is no per-reader inflight reference count anywhere in the shipped code;
 * the outstanding set below is batch-scoped, matching _cache_done.
 *
 * Run:
 *   quint verify slotpool.qnt --main verified \
 *     --invariant=NoUseAfterFree --inductive-invariant=IndInv
 *   quint verify slotpool.qnt --main broken \
 *     --invariant=NoUseAfterFree --max-steps=15
 */
module slotpool {
  type Slot = int
  type Key = str

  // Minimal universes that still exhibit both hazards.
  pure val Slots: Set[Slot] = 0.to(2)
  pure val Keys: Set[Key] = Set("a", "b", "c")
  pure val KeyU: Set[Key] = Keys.union(Set(""))
  pure val Statuses: Set[str] = Set("free", "loading", "ready")
  pure val Queues: Set[str] = Set("unqueued", "small", "main")

  const DEFER_RECYCLE: bool

  // --- pool plane (mirrors rc_entry packed state|queue) ---
  var status: Slot -> str
  var queue: Slot -> str
  // Slots with an in-flight fetch: set by reserve, cleared by confirm/cancel.
  // Their pool bytes are the destination of a pending store, so they must never
  // be recycled; the rings never hold them, so no pop can reach them.
  var loading_out: Set[Slot]

  // --- staging plane (the batch buffer the decode gathers from) ---
  // Current bytes in each staging slot, and the bytes an outstanding decode was
  // handed; they must agree while that decode is live.
  var stg_key: Slot -> Key
  var given_key: Slot -> Key
  // Staging slots being read by an unretired batch (fenced by _cache_done).
  var outstanding: Set[Slot]

  // Frame: Apalache requires each var constrained before it is read.
  def frame = and {
    status.in(setOfMaps(Slots, Statuses)),
    queue.in(setOfMaps(Slots, Queues)),
    loading_out.in(Slots.powerset()),
    stg_key.in(setOfMaps(Slots, KeyU)),
    given_key.in(setOfMaps(Slots, KeyU)),
    outstanding.in(Slots.powerset()),
  }

  action init = all {
    status' = Slots.mapBy(_ => "free"),
    queue' = Slots.mapBy(_ => "unqueued"),
    loading_out' = Set(),
    stg_key' = Slots.mapBy(_ => ""),
    given_key' = Slots.mapBy(_ => ""),
    outstanding' = Set(),
  }

  action step = {
    nondet s = Slots.oneOf()
    nondet k = Keys.oneOf()
    nondet q = Set("small", "main").oneOf()
    all {
      frame,
      any {
        // POOL RESERVE (rc_reserve / rc_claim_for_loading): claim a FREE slot
        // for an in-flight fetch. Joins no ring; not re-claimable (free-only).
        all {
          status.get(s) == "free",
          status' = status.set(s, "loading"),
          queue' = queue.set(s, "unqueued"),
          loading_out' = loading_out.union(Set(s)),
          stg_key' = stg_key,
          given_key' = given_key,
          outstanding' = outstanding,
        },
        // POOL CONFIRM (rc_confirm): the fetch landed; bind READY, join a ring.
        all {
          loading_out.contains(s),
          status.get(s) == "loading",
          status' = status.set(s, "ready"),
          queue' = queue.set(s, q),
          loading_out' = loading_out.exclude(Set(s)),
          stg_key' = stg_key,
          given_key' = given_key,
          outstanding' = outstanding,
        },
        // POOL CANCEL (rc_cancel): the fetch aborted; return the slot to free.
        all {
          loading_out.contains(s),
          status.get(s) == "loading",
          status' = status.set(s, "free"),
          queue' = queue.set(s, "unqueued"),
          loading_out' = loading_out.exclude(Set(s)),
          stg_key' = stg_key,
          given_key' = given_key,
          outstanding' = outstanding,
        },
        // POOL EVICT (rc_pop_small / rc_pop_main): recycle a ring victim. The
        // C walks only READY, queued slots and does NOT consult any staged set —
        // evicting a slot whose row was staged earlier is safe (store rewrites
        // pool bytes only after every hit is copied out). So a loading slot is
        // unreachable here by construction; no DEFER_RECYCLE guard applies.
        all {
          status.get(s) == "ready" and queue.get(s) != "unqueued",
          status' = status.set(s, "free"),
          queue' = queue.set(s, "unqueued"),
          loading_out' = loading_out,
          stg_key' = stg_key,
          given_key' = given_key,
          outstanding' = outstanding,
        },
        // STAGE WRITE (rc_copy_row pool->stage): a batch fills staging slot s
        // with row k. THE RULE UNDER TEST: the next batch must not overwrite a
        // slot an outstanding decode is still reading.
        all {
          not(DEFER_RECYCLE) or not(outstanding.contains(s)),
          stg_key' = stg_key.set(s, k),
          status' = status,
          queue' = queue,
          loading_out' = loading_out,
          given_key' = given_key,
          outstanding' = outstanding,
        },
        // DECODE BEGIN: an outstanding batch starts reading staging slot s; latch
        // the bytes it was handed.
        all {
          given_key' = given_key.set(s, stg_key.get(s)),
          outstanding' = outstanding.union(Set(s)),
          status' = status,
          queue' = queue,
          loading_out' = loading_out,
          stg_key' = stg_key,
        },
        // FENCE (_cache_done): the outstanding decode retires; staging slot freed
        // for reuse by the next transaction.
        all {
          outstanding.contains(s),
          outstanding' = outstanding.exclude(Set(s)),
          status' = status,
          queue' = queue,
          loading_out' = loading_out,
          stg_key' = stg_key,
          given_key' = given_key,
        },
        // tick (stutter).
        all {
          status' = status,
          queue' = queue,
          loading_out' = loading_out,
          stg_key' = stg_key,
          given_key' = given_key,
          outstanding' = outstanding,
        },
      }
    }
  }

  /// (1) A pending-fetch slot is still LOADING and joins no ring, so no pop
  /// walk (which only selects READY, queued slots) can recycle it, and claim
  /// (free-only) can never re-hand it out.
  val LoadingNeverVictim = loading_out.forall(s =>
    status.get(s) == "loading" and queue.get(s) == "unqueued")

  /// (2) SAFETY: while an outstanding decode reads staging slot s, the bytes in
  /// that slot are the row it was handed — no later batch swapped them under it.
  val StagingNotCorrupted = outstanding.forall(s =>
    stg_key.get(s) == given_key.get(s))

  val NoUseAfterFree = and { LoadingNeverVictim, StagingNotCorrupted }

  val IndInv = and { frame, NoUseAfterFree }
}

module verified { import slotpool(DEFER_RECYCLE = true).* }
module broken { import slotpool(DEFER_RECYCLE = false).* }
