// S3-FIFO host-tier index: a per-request mirror of the libCacheSim
// S3FIFO.c mirror (the S3FIFO-0.1000-2 variant) the repository shipped as
// S3FifoIndex in sequence/_shared/row_cache.py. That class is the
// behavioural oracle, now test-private as ``_PyS3FifoReference`` in
// tests/experimental/b12x/sequence/test_row_cache_policies.py; the
// request-facing contract here is byte-identical plan/store/reserve/
// confirm/cancel semantics, verified per-request against that oracle.
//
// Hot path: row_cache_read(count) resolves ids[:count] against the bound
// id plane, applies every hit/freq/ghost/queue mutation of plan(), and copies
// hit rows from the bound pool plane into the bound stage plane; then the
// caller fetches the miss wave into the bound miss planes and calls
// row_cache_store() for the miss staging and pool stores. No Python per-id
// work happens between create() and the counters. Every allocation is done at
// row_cache_create; read/store/reuse their persistent arrays and the stamped
// dedup tables, so the steady state is allocation-free.
//
// Lookup layout: the key->slot map and the ghost's membership map are both
// open-addressed linear-probing tables of 16 B records (key, id, packed
// freq|state|queue, tag), sized pow2 >= 2*entries and hashed with rc_mix64. A
// hit is one record line; a release, cancel, confirm or evict is one tombstone
// store through the slot/id -> position back-map, never a chain walk. The
// layout carries no decisions: which row hits, which slot becomes the victim,
// and what the ghost holds are exactly what the chained maps produced, which is
// what the per-request oracle parity test and the cachesim gate prove. The
// tombstone sweep that keeps probe runs short runs only from the insert path --
// the single point where no caller holds a table position and no eviction is in
// flight -- and allocates nothing, so the steady state stays allocation-free.
//
// Reclamation follows docs/slotpool.qnt: a LOADING slot joins no queue and
// is never an eviction victim (no pop can reach it), so an outstanding decode
// that staged from a READY slot keeps its bytes until the transaction fence
// (disk_table.py transaction() / _cache_done) retires it; the C port changes
// no release ordering: releases happen only from the same pop paths the
// Python class uses, and stores land after all hit copies of their batch.
//
// Threading: all mutation happens under DiskRowCache._lock (the transaction
// holds it across the read; the prefetch worker takes it around its
// reserve/confirm/cancel sections), and every entry point runs with the GIL
// held, so the structure needs no lock of its own — the same discipline the
// Python class relies on.

// posix_memalign/madvise back the big lookup arrays with 2 MiB-aligned,
// huge-page-eligible memory; the header comes in after Python.h because
// pyconfig.h pins the feature-test macros and including anything earlier
// makes glibc's features.h disagree with it under -Werror.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// MADV_HUGEPAGE is an enum constant in glibc's bits/mman-linux.h, not a macro,
// so #ifdef alone cannot see it and the advice would silently never be issued.
// The value is the Linux generic-ABI advice number on every arch this builds on
// (x86-64 and AArch64 alike); a libc that spells it as a macro keeps its own,
// and a non-Linux host is left with no advice at all.
#ifdef __linux__
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif
// glibc also keeps the madvise prototype behind _GNU_SOURCE, which an
// extension cannot set before Python.h without fighting pyconfig.h's own
// feature-test macros under -Werror. The signature is the Linux one, and a
// libc that does expose it (musl) declares it identically.
extern int madvise(void* addr, size_t len, int advice);
#endif

typedef char require_64_bit_pointers[(sizeof(size_t) == 8) ? 1 : -1];

#define RC_ROW_CACHE_CAPSULE "b12x.row_cache"
#define RC_ROW_CACHE_MAX_CAPACITY (INT32_MAX - 16)
#define RC_MAX_PLANES 2
#define RC_MAX_FREQ 3u
#define RC_SMALL_SHARE 10
#define RC_GHOST_SHARE 90
#define RC_MOVE_TO_MAIN_THRESHOLD 2u

// Slot states and queue tags, numerically identical to S3FifoIndex._FREE /
// _LOADING / _READY and _UNQUEUED / _SMALL / _MAIN.
enum { RC_FREE = 0, RC_LOADING = 1, RC_READY = 2 };
enum { RC_UNQUEUED = 0, RC_SMALL = 1, RC_MAIN = 2 };

// Table record (cross-lane contract, hash_lookup section 5): the key sits next
// to the id next to the packed policy byte, 16 B per record, four per cache
// line, so a resident hit costs one random line instead of the
// bucket/chain/key triple. `tag` is the probe filter: bit7 set means the
// position still belongs to a probe run (occupied or tombstone), bit6 marks the
// tombstone, and the low six bits carry the h2 fingerprint compared before the
// 8 B key. The resident map's id is the pool slot; the ghost map's id is the
// ghost entry index.
typedef struct {
  int64_t key;
  int32_t slot;
  uint8_t flags;  // freq2 | state2 | queue2 (P1); unused (0) in the ghost map
  uint8_t tag;    // RC_TAG_*
  uint8_t pad[2];
} rc_entry_t;

// Per-batch stamped tables: the three parallel arrays of the old design
// collapse into one 16 B entry so a dedup/store-probe step touches one line.
typedef struct {
  int64_t key;
  int32_t val;
  uint32_t stamp;
} rc_dedup_t;

typedef struct {
  int32_t slot;
  int32_t idx;
  uint32_t stamp;
  uint32_t pad;
} rc_store_probe_t;

#define RC_TAG_PRESENT 0x80u  // position stays part of a probe run
#define RC_TAG_TOMB 0xC0u     // tombstone: probes continue, inserts may reuse
#define RC_TAG_H2 0x3Fu       // h2 fingerprint bits
// h2 comes from a bit window disjoint from the home index, so records that
// share the head of a probe run are still told apart before the key compare.
#define RC_TAG_OF(h) (uint8_t)(RC_TAG_PRESENT | (((h) >> 40) & RC_TAG_H2))
#define RC_NO_POS UINT32_MAX  // back-map sentinel: this id holds no record

enum { RC_FREQ_SHIFT = 0, RC_STATE_SHIFT = 2, RC_QUEUE_SHIFT = 4 };
#define RC_FREQ_MASK 0x03u
#define RC_STATE_MASK 0x0Cu
#define RC_QUEUE_MASK 0x30u

static uint8_t rc_pack_flags(uint8_t freq, uint8_t state, uint8_t queue) {
  return (uint8_t)((freq & RC_FREQ_MASK) | ((state << RC_STATE_SHIFT) & RC_STATE_MASK) |
                   ((queue << RC_QUEUE_SHIFT) & RC_QUEUE_MASK));
}

// freq saturates at RC_MAX_FREQ (3) and state/queue at 2, so the two-bit
// fields cannot bleed into their neighbours when one is updated in place.
static uint8_t rc_freq_of(uint8_t flags) { return (uint8_t)(flags & RC_FREQ_MASK); }
static uint8_t rc_state_of(uint8_t flags) {
  return (uint8_t)((flags & RC_STATE_MASK) >> RC_STATE_SHIFT);
}
static uint8_t rc_queue_of(uint8_t flags) {
  return (uint8_t)((flags & RC_QUEUE_MASK) >> RC_QUEUE_SHIFT);
}

typedef struct {
  int nplanes;
  size_t row_bytes[RC_MAX_PLANES];
  int64_t capacity, small_cap, main_cap, ghost_cap, max_lookups;
  // Ghost entry universe: _demote_ghost inserts *before* trimming (the
  // Python dict momentarily holds cap+1 keys), so entries are sized cap + 2.
  int64_t ghost_entries;
  // Bound host buffers; the Py_buffer refs keep them alive (contract with
  // DiskRowCache._init_row_cache: ids/miss planes are pinned, pool/stage are
  // the host-tier numpy/mapped-host planes the Python path addresses).
  Py_buffer ids, miss_ids;
  Py_buffer miss[RC_MAX_PLANES], stage[RC_MAX_PLANES], pool[RC_MAX_PLANES];
  int64_t* ids_buf;
  int64_t* miss_ids_buf;
  uint8_t* miss_plane[RC_MAX_PLANES];
  uint8_t* stage_plane[RC_MAX_PLANES];
  uint8_t* pool_plane[RC_MAX_PLANES];
  // key -> slot map for READY and LOADING rows: open-addressed linear probing
  // over rc_entry_t records at load factor <= 0.5 (table = pow2 >= 2*capacity),
  // hashed with rc_mix64 exactly as the chained map it replaces was. A key is
  // never READY and LOADING at once (reserve rejects either, confirm is
  // state-guarded), so one table answers both ``key in _slot`` and
  // ``key in _loading``. freq/state/queue live in the record's flags byte
  // (P1); slotrec[] is the slot -> position back-map that makes delete-by-slot
  // one store (E1). An unbound slot has no position and reads back
  // FREE/UNQUEUED/0, which is what _release used to write into the arrays.
  rc_entry_t* table;
  uint32_t* slotrec;
  int64_t table_size, table_mask, n_used, n_tomb;
  // FIFO rings over slot ids (S3FifoIndex._small/_main deques) and the LIFO
  // free list (_free: list.pop() takes the most recently released slot).
  int32_t* small_ring;
  int32_t* main_ring;
  int64_t small_head, small_count, main_head, main_count;
  int32_t* free_stack;
  int64_t free_top;
  int64_t used, ready_count, loading_count;
  int has_evicted;
  // Ghost dict (_ghost): insertion-ordered key set with delete-any. The
  // doubly-linked order links stay exactly as they are (pop-oldest trim is
  // policy-semantic); only membership moved onto the same open-addressed
  // records as the resident map, so the record carries the key plus the entry
  // index and ghostrec[] is the entry -> position back-map. rc_ghost_contains
  // is one probe line and rc_ghost_del/unlink is one tombstone store instead of
  // a pointer chase through ghost_buckets/ghost_hash_next/ghost_key.
  rc_entry_t* ghost_table;
  uint32_t* ghostrec;
  int32_t* ghost_prev;
  int32_t* ghost_next;
  int64_t ghost_table_size, ghost_table_mask, ghost_n_used, ghost_n_tomb;
  int32_t* ghost_free_stack;
  int64_t ghost_free_top, ghost_count, ghost_head, ghost_tail;
  // Persistent per-batch arrays sized max_lookups (RowPlan fields) plus the
  // stamped dedup tables that replace the plan() ``fetched``/``hit_on_ghost``
  // and ``stored`` dicts without clearing between steps.
  int32_t *hit_pos, *hit_slot, *miss_pos, *miss_row, *store_slot, *store_row;
  uint8_t* ghost_latch;
  int32_t n_hit, n_miss_ids, n_miss_pos, n_store;
  rc_dedup_t* dedup;
  uint32_t dedup_gen;
  rc_store_probe_t* store_probe;
  uint32_t store_gen;
  // Hint only once the per-batch dedup table has outgrown an L2 (set at
  // create): below that size its probe lines are already resident and the
  // prefetch is pure pressure. Measured on x86-64: at batch 16384 (512 KiB
  // table) the hint costs ~3% ns/id, at the 96304-id prefill (4 MiB) it wins
  // ~12% against no hint at all.
  int dedup_pf;
  int64_t probe_mask;
  int64_t hits, misses;
} row_cache_t;

static uint64_t rc_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static int64_t rc_pow2_ceil(int64_t value) {
  int64_t n = 1;
  while (n < value) n <<= 1;
  return n;
}

// Blocks of at least one huge page are 2 MiB aligned and madvised MADV_HUGEPAGE:
// at the 12 M pool the lookup tables are random access, where a 4 KiB TLB turn
// costs a page walk per probe. THP is strictly best effort -- a host without
// MADV_HUGEPAGE, with THP set to never, or that rejects the advice (EINVAL or
// EAGAIN) behaves exactly as before -- and an alignment failure falls back to
// the plain allocator. Both are released with free().
#define RC_HUGE_PAGE (2u * 1024u * 1024u)

static void* rc_block(int64_t count, size_t elt, bool zero) {
  size_t bytes = (size_t)count * elt;
  if (bytes >= RC_HUGE_PAGE) {
    void* p = NULL;
    if (posix_memalign(&p, RC_HUGE_PAGE, bytes) == 0 && p) {
#ifdef MADV_HUGEPAGE
      (void)madvise(p, bytes, MADV_HUGEPAGE);
#endif
      if (zero) memset(p, 0, bytes);  // after the advice: faults get huge pages
      return p;
    }
  }
  return zero ? calloc(1, bytes ? bytes : elt) : malloc(bytes ? bytes : 1);
}

// ---------------------------------------------------------------------------
// Open-addressed core, shared by the resident key->slot map and the ghost's
// key->entry membership map (hash_lookup C1): linear probing over 16 B
// records, one store to delete, and an in-place sweep. back[] is the id ->
// position back-map (slotrec / ghostrec) that makes delete-by-id O(1).
// ---------------------------------------------------------------------------

// Position of `key`'s record given its mixed hash, or -1. Probes stop only at a
// never-populated position; tombstones are stepped over.
static int64_t rc_oa_probe(const rc_entry_t* table, int64_t mask, int64_t key, uint64_t h) {
  int64_t pos = (int64_t)(h & (uint64_t)mask);
  const uint8_t want = RC_TAG_OF(h);
  for (;;) {
    uint8_t t = table[pos].tag;
    if (t == 0) return -1;
    if (t == want && table[pos].key == key) return pos;
    pos = (pos + 1) & mask;
  }
}

static void rc_oa_put(rc_entry_t* table, uint32_t* back, int64_t pos, int64_t key, uint64_t h,
                      int32_t id, uint8_t flags) {
  table[pos].key = key;
  table[pos].slot = id;
  table[pos].flags = flags;
  table[pos].tag = RC_TAG_OF(h);
  back[id] = (uint32_t)pos;
}

// In-place tombstone sweep (hash_lookup R1): textbook backward-shift delete
// (Knuth 6.4 ex. 21, the re-insert-the-cluster formulation used in
// Sedgewick-Wayne 3.4) applied to every tombstone in position order. Dropping a
// tombstone is not enough -- records behind it probed through that position, so
// they would sit behind a hole and become unreachable -- hence each record in
// the rest of the cluster is lifted and re-probed from its own home, landing on
// the first never-populated position (tombstones count as occupied here and are
// handled by their own turn in the outer scan). No record can move past the
// cluster's terminating hole, so the scan is stable, and the outer pass leaves
// the table with zero tombstones: the caller's tombstone count goes to exactly
// zero, which is what keeps the trigger from under-reporting occupancy and
// letting a later insert probe a quietly full table forever.
static void rc_oa_sweep(rc_entry_t* table, int64_t size, int64_t mask, uint32_t* back) {
  for (int64_t t = 0; t < size; t++) {
    if (table[t].tag != RC_TAG_TOMB) continue;
    table[t].tag = 0;  // the hole this tombstone leaves
    for (int64_t j = (t + 1) & mask; table[j].tag != 0; j = (j + 1) & mask) {
      if (table[j].tag == RC_TAG_TOMB) continue;  // another hole: its own turn
      rc_entry_t moved = table[j];
      uint64_t mh = rc_mix64((uint64_t)moved.key);
      table[j].tag = 0;  // lift it out and re-probe from home
      int64_t p = (int64_t)(mh & (uint64_t)mask);
      while (table[p].tag != 0) p = (p + 1) & mask;
      table[p] = moved;
      table[p].tag = RC_TAG_OF(mh);
      back[moved.slot] = (uint32_t)p;
    }
  }
}

// Insert a key known to be absent, reusing the first tombstone of the run. The
// sweep runs only here, on the insert path: the single safe point where no
// caller holds a table position and the eviction sequence has completed, so
// table[]/back[] stay mutually consistent for every reader.
static void rc_oa_add(rc_entry_t* table, int64_t size, int64_t mask, uint32_t* back,
                      int64_t* n_used, int64_t* n_tomb, int64_t key, uint64_t h, int32_t id,
                      uint8_t flags) {
  if ((*n_used + *n_tomb) * 10 > size * 6) {  // 0.6 of the table: tombstone pile-up
    rc_oa_sweep(table, size, mask, back);
    *n_tomb = 0;
  }
  int64_t pos = (int64_t)(h & (uint64_t)mask);
  int64_t first_tomb = -1;
  for (;;) {
    uint8_t t = table[pos].tag;
    if (t == 0) {
      if (first_tomb >= 0) {  // the run stays the same length: tombstone -> occupied
        pos = first_tomb;
        *n_tomb -= 1;
      }
      *n_used += 1;
      rc_oa_put(table, back, pos, key, h, id, flags);
      return;
    }
    if (t == RC_TAG_TOMB && first_tomb < 0) first_tomb = pos;
    pos = (pos + 1) & mask;
  }
}

// E1: one store through the back-map, no hash re-walk. When the successor is a
// never-populated position the run ends here, so the record is cleared outright
// rather than left as a tombstone -- no live probe path crosses this position --
// which is what keeps the sweep trigger from firing as often (measured ~7 %
// faster per id at the evicting 100k pool).
static void rc_oa_kill(rc_entry_t* table, int64_t mask, uint32_t* back, int64_t* n_used,
                       int64_t* n_tomb, int64_t pos) {
  int32_t id = table[pos].slot;
  back[id] = RC_NO_POS;
  *n_used -= 1;
  if (table[(pos + 1) & mask].tag == 0) {
    table[pos].tag = 0;
    return;
  }
  table[pos].tag = RC_TAG_TOMB;
  *n_tomb += 1;
}

// ---------------------------------------------------------------------------
// key -> slot map (mirrors S3FifoIndex._slot plus _loading)
// ---------------------------------------------------------------------------

static int64_t rc_tbl_probe(const row_cache_t* rc, int64_t key, uint64_t h) {
  return rc_oa_probe(rc->table, rc->table_mask, key, h);
}

static void rc_tbl_add_h(row_cache_t* rc, int64_t key, uint64_t h, int32_t slot, uint8_t flags) {
  rc_oa_add(rc->table, rc->table_size, rc->table_mask, rc->slotrec, &rc->n_used, &rc->n_tomb, key,
            h, slot, flags);
}

static void rc_tbl_kill(row_cache_t* rc, int64_t pos) {
  rc_oa_kill(rc->table, rc->table_mask, rc->slotrec, &rc->n_used, &rc->n_tomb, pos);
}

static void rc_tbl_del_by_slot(row_cache_t* rc, int32_t slot) {
  uint32_t pos = rc->slotrec[slot];
  if (pos != RC_NO_POS) rc_tbl_kill(rc, (int64_t)pos);
}

// slot -> record through the back-map; NULL when the slot holds no row, whose
// state/queue/freq then read FREE/UNQUEUED/0 exactly as the cleared per-slot
// arrays did.
static const rc_entry_t* rc_slot_entry(const row_cache_t* rc, int32_t slot) {
  uint32_t pos = rc->slotrec[slot];
  return pos == RC_NO_POS ? NULL : &rc->table[pos];
}

static int rc_slot_state(const row_cache_t* rc, int32_t slot) {
  const rc_entry_t* e = rc_slot_entry(rc, slot);
  return e ? (int)rc_state_of(e->flags) : RC_FREE;
}

// ---------------------------------------------------------------------------
// FIFO rings (S3FifoIndex._small / _main append + popleft)
// ---------------------------------------------------------------------------

static void rc_ring_push(int32_t* ring, int64_t size, int64_t* head, int64_t* count, int32_t slot) {
  int64_t tail = (*head + *count) % size;
  ring[tail] = slot;
  *count += 1;
}

static int32_t rc_ring_pop(int32_t* ring, int64_t size, int64_t* head, int64_t* count) {
  int32_t slot = ring[*head];
  *head = (*head + 1) % size;
  *count -= 1;
  return slot;
}

// ---------------------------------------------------------------------------
// ghost dict (S3FifoIndex._ghost: dict-of-None, insertion order)
// ---------------------------------------------------------------------------

// Membership is the shared open-addressed probe; the entry index comes back in
// the record, so the chain walk is gone. `*_h` takes the caller's already
// mixed hash (the read loop hashes every id once for all three tables).
static bool rc_ghost_contains_h(row_cache_t* rc, int64_t key, uint64_t h) {
  if (rc->ghost_cap == 0) return false;
  return rc_oa_probe(rc->ghost_table, rc->ghost_table_mask, key, h) >= 0;
}

static bool rc_ghost_contains(row_cache_t* rc, int64_t key) {
  return rc_ghost_contains_h(rc, key, rc_mix64((uint64_t)key));
}

// Key of a live ghost entry, for the diagnostics that mirror the dict.
static int64_t rc_ghost_key_of(const row_cache_t* rc, int32_t entry) {
  uint32_t pos = rc->ghostrec[entry];
  return pos == RC_NO_POS ? 0 : rc->ghost_table[pos].key;
}

// E1 for the ghost: the entry -> position back-map makes unlink one store
// instead of re-walking a chain to find the entry.
static void rc_ghost_unlink_hash(row_cache_t* rc, int32_t entry) {
  uint32_t pos = rc->ghostrec[entry];
  if (pos == RC_NO_POS) return;
  rc_oa_kill(rc->ghost_table, rc->ghost_table_mask, rc->ghostrec, &rc->ghost_n_used,
             &rc->ghost_n_tomb, (int64_t)pos);
}

// self._ghost[key] = None: an existing key keeps its position (dict
// overwrite), a new one appends at the tail.
static void rc_ghost_push(row_cache_t* rc, int64_t key) {
  if (rc->ghost_cap == 0) return;  // trimmed to zero immediately anyway
  uint64_t h = rc_mix64((uint64_t)key);
  if (rc_oa_probe(rc->ghost_table, rc->ghost_table_mask, key, h) >= 0) return;
  int32_t e = rc->ghost_free_stack[--rc->ghost_free_top];
  rc->ghost_prev[e] = rc->ghost_tail;
  rc->ghost_next[e] = -1;
  if (rc->ghost_tail >= 0) {
    rc->ghost_next[rc->ghost_tail] = e;
  } else {
    rc->ghost_head = e;
  }
  rc->ghost_tail = e;
  rc_oa_add(rc->ghost_table, rc->ghost_table_size, rc->ghost_table_mask, rc->ghostrec,
            &rc->ghost_n_used, &rc->ghost_n_tomb, key, h, e, 0);
  rc->ghost_count += 1;
}

// pop(next(iter(self._ghost))): the oldest entry leaves.
static void rc_ghost_pop_first(row_cache_t* rc) {
  int32_t e = rc->ghost_head;
  rc->ghost_head = rc->ghost_next[e];
  if (rc->ghost_head < 0) {
    rc->ghost_tail = -1;
  } else {
    rc->ghost_prev[rc->ghost_head] = -1;
  }
  rc_ghost_unlink_hash(rc, e);
  rc->ghost_free_stack[rc->ghost_free_top++] = e;
  rc->ghost_count -= 1;
}

static void rc_ghost_del(row_cache_t* rc, int64_t key) {
  if (rc->ghost_cap == 0) return;
  int64_t pos = rc_oa_probe(rc->ghost_table, rc->ghost_table_mask, key, rc_mix64((uint64_t)key));
  if (pos < 0) return;
  int32_t e = rc->ghost_table[pos].slot;
  if (rc->ghost_prev[e] >= 0) {
    rc->ghost_next[rc->ghost_prev[e]] = rc->ghost_next[e];
  } else {
    rc->ghost_head = rc->ghost_next[e];
  }
  if (rc->ghost_next[e] >= 0) {
    rc->ghost_prev[rc->ghost_next[e]] = rc->ghost_prev[e];
  } else {
    rc->ghost_tail = rc->ghost_prev[e];
  }
  rc_ghost_unlink_hash(rc, e);
  rc->ghost_free_stack[rc->ghost_free_top++] = e;
  rc->ghost_count -= 1;
}

// S3FifoIndex._demote_ghost.
static void rc_demote_ghost(row_cache_t* rc, int64_t key) {
  rc_ghost_push(rc, key);
  while (rc->ghost_count > rc->ghost_cap) rc_ghost_pop_first(rc);
}

// ---------------------------------------------------------------------------
// pool slot lifecycle (S3FifoIndex._release / _bind / _bind_loaded)
// ---------------------------------------------------------------------------

static void rc_release(row_cache_t* rc, int32_t slot) {
  rc_tbl_del_by_slot(rc, slot);  // E1: one tombstone store, no hash re-walk
  rc->free_stack[rc->free_top++] = slot;
  rc->ready_count -= 1;
}

static void rc_bind(row_cache_t* rc, int32_t slot, int64_t key, uint64_t h, bool main) {
  rc_tbl_add_h(rc, key, h, slot, rc_pack_flags(0, RC_READY, (uint8_t)(main ? RC_MAIN : RC_SMALL)));
  if (main) {
    rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
  } else {
    rc_ring_push(rc->small_ring, rc->capacity, &rc->small_head, &rc->small_count, slot);
  }
  rc->ready_count += 1;
}

// S3FifoIndex._pop_main: the 2-bit clock over M.
static void rc_pop_main(row_cache_t* rc) {
  while (rc->main_count) {
    int32_t slot = rc_ring_pop(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count);
    // A queued slot is always bound, so the back-map addresses its record
    // directly: the clock reads the victim's freq from the same line the
    // release path needs for its tombstone store.
    rc_entry_t* e = &rc->table[rc->slotrec[slot]];
    uint8_t f = rc_freq_of(e->flags);
    if (f >= 1) {
      e->flags = (uint8_t)((e->flags & (uint8_t)~RC_FREQ_MASK) |
                           (uint8_t)((f < RC_MAX_FREQ ? f : RC_MAX_FREQ) - 1));
      rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
      continue;
    }
    rc_release(rc, slot);
    return;
  }
}

// S3FifoIndex._pop_small: recycle S's front; threshold credit promotes into M
// keeping its counter (overfill paid back at once), anything else demotes to
// a ghost and frees its slot. Promotions loop; only a demotion returns.
static void rc_pop_small(row_cache_t* rc) {
  while (rc->small_count) {
    int32_t slot = rc_ring_pop(rc->small_ring, rc->capacity, &rc->small_head, &rc->small_count);
    rc_entry_t* e = &rc->table[rc->slotrec[slot]];
    if (rc_freq_of(e->flags) >= RC_MOVE_TO_MAIN_THRESHOLD && rc->main_cap > 0) {
      e->flags = (uint8_t)((e->flags & (uint8_t)~RC_QUEUE_MASK) |
                           (uint8_t)((RC_MAIN << RC_QUEUE_SHIFT) & RC_QUEUE_MASK));
      rc_ring_push(rc->main_ring, rc->capacity, &rc->main_head, &rc->main_count, slot);
      while (rc->main_count > rc->main_cap) rc_pop_main(rc);
      continue;
    }
    int64_t key = e->key;
    rc_release(rc, slot);
    rc_demote_ghost(rc, key);
    return;
  }
}

// S3FifoIndex._evict_one: the eviction source rule, with the newcomer's
// planned destination counted into M's overfill test.
static void rc_evict_one(row_cache_t* rc, bool main) {
  rc->has_evicted = 1;
  if (rc->main_count + (main ? 1 : 0) > rc->main_cap || rc->small_count == 0) {
    rc_pop_main(rc);
  } else {
    rc_pop_small(rc);
  }
}

// S3FifoIndex._prepare_admission.
static void rc_prepare_admission(row_cache_t* rc, bool main) {
  while (rc->ready_count >= rc->capacity) {
    rc_evict_one(rc, main);
    if (rc->free_top) return;
  }
}

// S3FifoIndex._settle_main_budget.
static void rc_settle_main_budget(row_cache_t* rc) {
  while (rc->main_count > rc->main_cap) rc_pop_main(rc);
}

// S3FifoIndex._admit. latch == -1 mirrors hit_on_ghost=None (read the ghost
// set now); otherwise it is the value plan() latched at lookup time.
static void rc_admit(row_cache_t* rc, int64_t key, int latch, bool* out_main, bool* out_ghost) {
  bool hit_on_ghost = latch >= 0 ? (bool)latch : rc_ghost_contains(rc, key);
  if (hit_on_ghost && rc->main_cap > 0) {
    rc_prepare_admission(rc, true);
    *out_main = true;
    *out_ghost = true;
    return;
  }
  rc_prepare_admission(rc, false);
  // S3FifoIndex._ghost_admits_main: the C's has_evicted fast path.
  *out_main = !rc->has_evicted && rc->small_count >= rc->small_cap && rc->main_cap > 0;
  *out_ghost = hit_on_ghost;
}

static void rc_copy_row(row_cache_t* rc, uint8_t* const* dst_planes,
                        const uint8_t* const* src_planes, int64_t dst_row, int64_t src_row) {
  for (int p = 0; p < rc->nplanes; p++) {
    memcpy(dst_planes[p] + (size_t)dst_row * rc->row_bytes[p],
           src_planes[p] + (size_t)src_row * rc->row_bytes[p], rc->row_bytes[p]);
  }
}

// S3FifoIndex.confirm. Idempotent off a non-loading slot, exactly like the
// Python method.
static int32_t rc_confirm(row_cache_t* rc, int32_t slot) {
  const rc_entry_t* e = rc_slot_entry(rc, slot);
  if (e == NULL || rc_state_of(e->flags) != RC_LOADING) return slot;
  int64_t key = e->key;
  rc_tbl_del_by_slot(rc, slot);  // E1: release/cancel/confirm never re-walk the hash
  rc->loading_count -= 1;
  bool main, ghost;
  rc_admit(rc, key, -1, &main, &ghost);
  // _bind_loaded: same shape as _bind for an already-owned, unqueued slot.
  rc_bind(rc, slot, key, rc_mix64((uint64_t)key), main);
  if (ghost) rc_ghost_del(rc, key);
  rc_settle_main_budget(rc);
  return slot;
}

// S3FifoIndex.insert. Returns the slot, -1 when no slot can be taken, or -2
// for the "already cached" condition the Python method raises KeyError on
// (unreachable from plan because miss ids are distinct, still surfaced).
// `h` is the caller's mixed hash of `key`, reused for the presence probe and
// the insert so the read loop hashes each miss id once.
static int32_t rc_insert_h(row_cache_t* rc, int64_t key, uint64_t h, int latch) {
  int64_t pos = rc_tbl_probe(rc, key, h);
  if (pos >= 0) {
    rc_entry_t* e = &rc->table[pos];
    if (rc_state_of(e->flags) == RC_LOADING) return rc_confirm(rc, e->slot);
    return -2;
  }
  bool main, ghost;
  rc_admit(rc, key, latch, &main, &ghost);
  if (ghost) rc_ghost_del(rc, key);
  int32_t slot;
  if (rc->free_top) {
    slot = rc->free_stack[--rc->free_top];
  } else if (rc->used < rc->capacity) {
    slot = (int32_t)rc->used++;
  } else {
    return -1;
  }
  rc_bind(rc, slot, key, h, main);
  rc_settle_main_budget(rc);
  return slot;
}

// S3FifoIndex._claim_for_loading.
static int32_t rc_claim_for_loading(row_cache_t* rc) {
  if (rc->free_top) return rc->free_stack[--rc->free_top];
  if (rc->used < rc->capacity) return (int32_t)rc->used++;
  if (rc->ready_count <= 1) {
    // Keep at least one committed row evictable so the synchronous read path
    // can never starve behind a queue of in-flight prefetches.
    return -1;
  }
  rc_evict_one(rc, false);
  return rc->free_top ? rc->free_stack[--rc->free_top] : -1;
}

// S3FifoIndex.reserve.
static int32_t rc_reserve(row_cache_t* rc, int64_t key) {
  uint64_t h = rc_mix64((uint64_t)key);
  if (rc_tbl_probe(rc, key, h) >= 0) return -1;
  int32_t slot = rc_claim_for_loading(rc);
  if (slot < 0) return -1;
  rc_tbl_add_h(rc, key, h, slot, rc_pack_flags(0, RC_LOADING, RC_UNQUEUED));
  rc->loading_count += 1;
  return slot;
}

// S3FifoIndex.cancel.
static void rc_cancel(row_cache_t* rc, int32_t slot) {
  const rc_entry_t* e = rc_slot_entry(rc, slot);
  if (e == NULL || rc_state_of(e->flags) != RC_LOADING) return;
  rc_tbl_del_by_slot(rc, slot);
  rc->loading_count -= 1;
  rc->free_stack[rc->free_top++] = slot;
}

// S3FifoIndex.get: pure lookup, raises the 2-bit counter on a resident hit
// and never consumes the ghost entry.
// One record line: the probe lands on the entry that holds the slot, the
// READY test and the frequency bump are in that same byte (P1).
static int32_t rc_get_h(row_cache_t* rc, int64_t key, uint64_t h) {
  int64_t pos = rc_tbl_probe(rc, key, h);
  if (pos < 0) return -1;
  rc_entry_t* e = &rc->table[pos];
  if (rc_state_of(e->flags) != RC_READY) return -1;
  if (rc_freq_of(e->flags) < RC_MAX_FREQ) e->flags = (uint8_t)(e->flags + 1);
  return e->slot;
}

static int32_t rc_get(row_cache_t* rc, int64_t key) {
  return rc_get_h(rc, key, rc_mix64((uint64_t)key));
}

// ---------------------------------------------------------------------------
// stamped dedup tables (plan's ``fetched`` / latched ``hit_on_ghost`` and the
// ``stored`` slot dict: first-insert order kept, value updated)
// ---------------------------------------------------------------------------

#define RC_NOT_FOUND INT32_MIN

static int32_t rc_dedup_find_h(row_cache_t* rc, int64_t key, uint64_t h) {
  int64_t p = (int64_t)(h & (uint64_t)rc->probe_mask);
  while (rc->dedup[p].stamp == rc->dedup_gen) {
    if (rc->dedup[p].key == key) return rc->dedup[p].val;
    p = (p + 1) & rc->probe_mask;
  }
  return RC_NOT_FOUND;
}

static void rc_dedup_add_h(row_cache_t* rc, int64_t key, uint64_t h, int32_t val) {
  int64_t p = (int64_t)(h & (uint64_t)rc->probe_mask);
  while (rc->dedup[p].stamp == rc->dedup_gen) p = (p + 1) & rc->probe_mask;
  rc->dedup[p].stamp = rc->dedup_gen;
  rc->dedup[p].key = key;
  rc->dedup[p].val = val;
}

static int32_t rc_store_find(row_cache_t* rc, int32_t slot) {
  int64_t p = (int64_t)(rc_mix64((uint64_t)(uint32_t)slot) & (uint64_t)rc->probe_mask);
  while (rc->store_probe[p].stamp == rc->store_gen) {
    if (rc->store_probe[p].slot == slot) return rc->store_probe[p].idx;
    p = (p + 1) & rc->probe_mask;
  }
  return RC_NOT_FOUND;
}

static void rc_store_add(row_cache_t* rc, int32_t slot, int32_t idx) {
  int64_t p = (int64_t)(rc_mix64((uint64_t)(uint32_t)slot) & (uint64_t)rc->probe_mask);
  while (rc->store_probe[p].stamp == rc->store_gen) p = (p + 1) & rc->probe_mask;
  rc->store_probe[p].stamp = rc->store_gen;
  rc->store_probe[p].slot = slot;
  rc->store_probe[p].idx = idx;
}

// ---------------------------------------------------------------------------
// module glue
// ---------------------------------------------------------------------------

static void row_cache_release(row_cache_t* rc) {
  if (!rc) return;
  PyBuffer_Release(&rc->ids);
  PyBuffer_Release(&rc->miss_ids);
  for (int p = 0; p < rc->nplanes; p++) {
    PyBuffer_Release(&rc->miss[p]);
    PyBuffer_Release(&rc->stage[p]);
    PyBuffer_Release(&rc->pool[p]);
  }
  free(rc->table);
  free(rc->slotrec);
  free(rc->small_ring);
  free(rc->main_ring);
  free(rc->free_stack);
  free(rc->ghost_table);
  free(rc->ghostrec);
  free(rc->ghost_prev);
  free(rc->ghost_next);
  free(rc->ghost_free_stack);
  free(rc->hit_pos);
  free(rc->hit_slot);
  free(rc->miss_pos);
  free(rc->miss_row);
  free(rc->store_slot);
  free(rc->store_row);
  free(rc->ghost_latch);
  free(rc->dedup);
  free(rc->store_probe);
  free(rc);
}

static void rc_delete(PyObject* capsule) {
  row_cache_t* rc = PyCapsule_GetPointer(capsule, RC_ROW_CACHE_CAPSULE);
  if (rc) row_cache_release(rc);
}

static int rc_check_1d_int64(const Py_buffer* b, const char* name) {
  if (b->ndim != 1 || b->itemsize != 8 || !PyBuffer_IsContiguous(b, 'C') ||
      (b->format && strcmp(b->format, "l") != 0 && strcmp(b->format, "q") != 0) ||
      (b->strides && b->strides[0] != b->itemsize)) {
    PyErr_Format(PyExc_ValueError, "row cache %s must be a C-contiguous signed int64 1-D buffer",
                 name);
    return -1;
  }
  return 0;
}

static int rc_check_plane(const Py_buffer* b, int64_t rows, const char* name) {
  bool row_major =
      b->ndim == 2 &&
      (!b->strides || (b->strides[1] == b->itemsize && b->strides[0] == b->shape[1] * b->itemsize));
  if (!row_major || b->itemsize != 1 || !PyBuffer_IsContiguous(b, 'C') || b->shape[0] != rows) {
    PyErr_Format(PyExc_ValueError,
                 "row cache %s must be a C-contiguous uint8 plane of shape (%lld, width)", name,
                 (long long)rows);
    return -1;
  }
  return 0;
}

static int rc_bind_plane(PyObject* tuple, int plane, Py_buffer* out, const char* name) {
  (void)name;  // diagnostics live in rc_check_plane's caller-side validation
  PyObject* item = PyTuple_GET_ITEM(tuple, plane);
  // PyBUF_STRIDED so rc_check_plane sees the exporter's real strides: a
  // strided view of a larger tensor must fail, not be assumed row-major.
  if (PyObject_GetBuffer(item, out, PyBUF_STRIDED) < 0) return -1;
  return 0;
}

static PyObject* py_row_cache_create(PyObject* self, PyObject* args) {
  (void)self;
  long long capacity, max_lookups;
  PyObject *ids_object, *miss_ids_object, *miss_tuple, *stage_tuple, *pool_tuple;
  if (!PyArg_ParseTuple(args, "LLOOOOO", &capacity, &max_lookups, &ids_object, &miss_ids_object,
                        &miss_tuple, &stage_tuple, &pool_tuple))
    return NULL;
  if (capacity <= 0) {
    PyErr_SetString(PyExc_ValueError, "row cache capacity must be positive");
    return NULL;
  }
  if (capacity > RC_ROW_CACHE_MAX_CAPACITY) {
    PyErr_SetString(PyExc_ValueError, "row cache capacity exceeds the 32-bit slot limit");
    return NULL;
  }
  if (max_lookups <= 0 || max_lookups > RC_ROW_CACHE_MAX_CAPACITY) {
    PyErr_SetString(PyExc_ValueError,
                    "row cache batch capacity must be positive and fit a "
                    "32-bit slot index");
    return NULL;
  }
  if (!PyTuple_Check(miss_tuple) || !PyTuple_Check(stage_tuple) || !PyTuple_Check(pool_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) != PyTuple_GET_SIZE(stage_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) != PyTuple_GET_SIZE(pool_tuple) ||
      PyTuple_GET_SIZE(miss_tuple) < 1 || PyTuple_GET_SIZE(miss_tuple) > RC_MAX_PLANES) {
    PyErr_SetString(PyExc_ValueError,
                    "row cache planes must be matching 1- or 2-tuples of uint8 buffers");
    return NULL;
  }
  row_cache_t* rc = calloc(1, sizeof(row_cache_t));
  if (!rc) return PyErr_NoMemory();
  rc->capacity = capacity;
  rc->max_lookups = max_lookups;
  rc->small_cap = capacity * RC_SMALL_SHARE / 100;
  if (rc->small_cap < 1) rc->small_cap = 1;
  rc->main_cap = capacity - rc->small_cap;
  rc->ghost_cap = capacity * RC_GHOST_SHARE / 100;
  rc->nplanes = (int)PyTuple_GET_SIZE(miss_tuple);
  rc->table = rc->ghost_table = NULL;
  rc->slotrec = rc->ghostrec = NULL;
  rc->small_ring = rc->main_ring = rc->free_stack = NULL;
  rc->ghost_prev = rc->ghost_next = rc->ghost_free_stack = NULL;
  rc->hit_pos = rc->hit_slot = rc->miss_pos = rc->miss_row = rc->store_slot = rc->store_row = NULL;
  rc->ghost_latch = NULL;
  rc->dedup = NULL;
  rc->store_probe = NULL;
  rc->ghost_head = rc->ghost_tail = -1;
  rc->small_head = rc->main_head = 0;
  // Open addressing needs a spare hole to terminate every probe run: pow2 >=
  // 2*ids keeps the load factor at or below 0.5 for the resident table (whose
  // live records never pass capacity) and for the ghost (whose entries never
  // pass ghost_cap + 1, the moment the Python dict holds cap+1 keys).
  int64_t table_size = rc_pow2_ceil(capacity * 2 < 4 ? 4 : capacity * 2);
  rc->ghost_entries = rc->ghost_cap + 2;
  int64_t ghost_table_size = rc_pow2_ceil(rc->ghost_entries * 2 < 4 ? 4 : rc->ghost_entries * 2);
  int64_t probes = rc_pow2_ceil(max_lookups * 2 < 8 ? 8 : max_lookups * 2);
  rc->table_size = table_size;
  rc->table_mask = table_size - 1;
  rc->ghost_table_size = ghost_table_size;
  rc->ghost_table_mask = ghost_table_size - 1;
  rc->probe_mask = probes - 1;
  rc->dedup_pf = probes * (int64_t)sizeof(rc_dedup_t) >= RC_HUGE_PAGE;

  bool ok = true;
  ok &= PyObject_GetBuffer(ids_object, &rc->ids, PyBUF_STRIDED_RO | PyBUF_FORMAT) == 0;
  if (ok && rc_check_1d_int64(&rc->ids, "ids") != 0) ok = false;
  if (ok && rc->ids.len < max_lookups * 8) {
    PyErr_SetString(PyExc_ValueError, "row cache ids buffer must cover max_lookups entries");
    ok = false;
  }
  if (ok && PyObject_GetBuffer(miss_ids_object, &rc->miss_ids, PyBUF_STRIDED | PyBUF_FORMAT) == 0) {
    if (rc_check_1d_int64(&rc->miss_ids, "miss ids") != 0) ok = false;
  } else {
    ok = false;
  }
  if (ok && rc->miss_ids.len < max_lookups * 8) {
    PyErr_SetString(PyExc_ValueError, "row cache miss ids buffer must cover max_lookups entries");
    ok = false;
  }
  for (int p = 0; ok && p < rc->nplanes; p++) {
    ok &= rc_bind_plane(miss_tuple, p, &rc->miss[p], "miss") == 0;
    if (ok) ok &= rc_check_plane(&rc->miss[p], max_lookups, "miss") == 0;
    if (ok) {
      ok &= rc_bind_plane(stage_tuple, p, &rc->stage[p], "stage") == 0;
      if (ok) ok &= rc_check_plane(&rc->stage[p], max_lookups, "stage") == 0;
    }
    if (ok) {
      ok &= rc_bind_plane(pool_tuple, p, &rc->pool[p], "pool") == 0;
      if (ok) ok &= rc_check_plane(&rc->pool[p], capacity, "pool") == 0;
    }
    if (ok) {
      // Planes are independent row geometries (weight 257 B, scale 9 B in
      // the shipped PLE layout); the invariant is that plane p of the three
      // tuples describes the same rows and width.
      ok &= rc->stage[p].shape[1] == rc->miss[p].shape[1] &&
            rc->pool[p].shape[1] == rc->miss[p].shape[1];
      if (!ok)
        PyErr_Format(PyExc_ValueError, "row cache plane %d widths differ across miss/stage/pool",
                     p);
    }
    if (ok) rc->row_bytes[p] = (size_t)rc->pool[p].shape[1];
  }

#define RC_ALLOC(type, field, count, zero)               \
  if (ok) {                                              \
    rc->field = rc_block((count), sizeof(type), (zero)); \
    if (!rc->field) {                                    \
      PyErr_NoMemory();                                  \
      ok = false;                                        \
    }                                                    \
  }
  RC_ALLOC(rc_entry_t, table, table_size, true)
  RC_ALLOC(uint32_t, slotrec, capacity, false)
  RC_ALLOC(int32_t, small_ring, capacity, false)
  RC_ALLOC(int32_t, main_ring, capacity, false)
  RC_ALLOC(int32_t, free_stack, capacity, false)
  RC_ALLOC(rc_entry_t, ghost_table, ghost_table_size, true)
  RC_ALLOC(uint32_t, ghostrec, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_prev, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_next, rc->ghost_entries, false)
  RC_ALLOC(int32_t, ghost_free_stack, rc->ghost_entries, false)
  RC_ALLOC(int32_t, hit_pos, max_lookups, false)
  RC_ALLOC(int32_t, hit_slot, max_lookups, false)
  RC_ALLOC(int32_t, miss_pos, max_lookups, false)
  RC_ALLOC(int32_t, miss_row, max_lookups, false)
  RC_ALLOC(int32_t, store_slot, max_lookups, false)
  RC_ALLOC(int32_t, store_row, max_lookups, false)
  RC_ALLOC(uint8_t, ghost_latch, max_lookups, false)
  RC_ALLOC(rc_dedup_t, dedup, probes, true)
  RC_ALLOC(rc_store_probe_t, store_probe, probes, true)
#undef RC_ALLOC

  if (ok) {
    // tag 0 (never populated) is the empty state, so the tables need nothing
    // beyond their zeroed allocation; the back-maps say "no record" until a
    // bind puts the id in a record.
    for (int64_t i = 0; i < capacity; i++) rc->slotrec[i] = RC_NO_POS;
    for (int64_t i = 0; i < rc->ghost_entries; i++) rc->ghostrec[i] = RC_NO_POS;
    for (int64_t i = 0; i < rc->ghost_entries; i++) rc->ghost_free_stack[i] = (int32_t)i;
    rc->ghost_free_top = rc->ghost_entries;
    rc->ids_buf = (int64_t*)rc->ids.buf;
    rc->miss_ids_buf = (int64_t*)rc->miss_ids.buf;
    for (int p = 0; p < rc->nplanes; p++) {
      rc->miss_plane[p] = (uint8_t*)rc->miss[p].buf;
      rc->stage_plane[p] = (uint8_t*)rc->stage[p].buf;
      rc->pool_plane[p] = (uint8_t*)rc->pool[p].buf;
    }
  }
  if (!ok) {
    row_cache_release(rc);
    return NULL;
  }
  PyObject* capsule = PyCapsule_New(rc, RC_ROW_CACHE_CAPSULE, rc_delete);
  if (!capsule) row_cache_release(rc);
  return capsule;
}

static row_cache_t* rc_from(PyObject* capsule) {
  return PyCapsule_GetPointer(capsule, RC_ROW_CACHE_CAPSULE);
}

// Group size for the read path's two-phase prefetch: 16 ids x one line each is
// well inside the outstanding-miss capacity of the cores this runs on, and one
// pass over the group's hashes costs no memory traffic at all.
#define RC_PF_GROUP 16

// _PyS3FifoReference.plan: resolve ids[:count]; latches the ghost flag at
// lookup time, counts every miss position, then admits each distinct miss
// after all lookups so hits raise their counters before any recycling. Hit
// copies run here (after all inserts, so this batch's eviction never clobbers
// a row the batch still reads -- the same ordering as _read_cached); miss
// staging and pool stores run in row_cache_store after the caller's fetch
// filled the miss planes.
static PyObject* py_row_cache_read(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long count;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &count)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (count < 0 || count > rc->max_lookups) {
    PyErr_SetString(PyExc_ValueError, "row cache batch count exceeds batch capacity");
    return NULL;
  }
  rc->n_hit = rc->n_miss_ids = rc->n_miss_pos = rc->n_store = 0;
  if (++rc->dedup_gen == 0) {  // wrap: zeroed/stale stamps would read as live
    memset(rc->dedup, 0, (size_t)(rc->probe_mask + 1) * sizeof(rc_dedup_t));
    rc->dedup_gen = 1;
  }
  if (++rc->store_gen == 0) {
    memset(rc->store_probe, 0, (size_t)(rc->probe_mask + 1) * sizeof(rc_store_probe_t));
    rc->store_gen = 1;
  }
  const int64_t* ids = rc->ids_buf;
  // Two-phase group prefetch (hash_lookup pick 3): hash a whole group first and
  // issue a prefetch on every member's first probe line -- the resident record,
  // plus the dedup entry once that table has outgrown an L2 -- before resolving
  // any of them, so the batch's independent random loads are in flight together
  // instead of one dependent walk per id. __builtin_prefetch is the GCC/Clang
  // builtin on both supported arches (PREFETCHT0-class on x86-64, PRFM on
  // AArch64, a no-op where unsupported); every address is inside an allocation,
  // so the hint can never fault or change results. Resolution order is untouched,
  // so the plan, the freq/ghost/queue mutations and the admission order are
  // exactly the ungrouped loop's. The ghost record is deliberately not hinted:
  // it is consulted only for ids that miss, and hinting it measured 13-15 %
  // slower per id at the 1 GiB pool (hit-heavy, so the line is pure waste) while
  // the miss-side win it buys at the small pool is covered by the admission
  // loop's own record hint.
  uint64_t hg[RC_PF_GROUP];
  for (int64_t begin = 0; begin < count; begin += RC_PF_GROUP) {
    int64_t end = begin + RC_PF_GROUP;
    if (end > count) end = count;
    for (int64_t i = begin; i < end; i++) {
      uint64_t h = hg[i - begin] = rc_mix64((uint64_t)ids[i]);
      __builtin_prefetch(&rc->table[(int64_t)(h & (uint64_t)rc->table_mask)], 0, 3);
      if (rc->dedup_pf)
        __builtin_prefetch(&rc->dedup[(int64_t)(h & (uint64_t)rc->probe_mask)], 0, 3);
    }
    for (int64_t position = begin; position < end; position++) {
      int64_t key = ids[position];
      uint64_t h = hg[position - begin];
      int32_t slot = rc_get_h(rc, key, h);
      if (slot >= 0) {
        rc->hit_pos[rc->n_hit] = (int32_t)position;
        rc->hit_slot[rc->n_hit] = slot;
        rc->n_hit += 1;
        continue;
      }
      int32_t row = rc_dedup_find_h(rc, key, h);
      if (row == RC_NOT_FOUND) {
        row = rc->n_miss_ids;
        rc->miss_ids_buf[row] = key;
        // Latch the C's find() flag at lookup time (plan's hit_on_ghost dict).
        rc->ghost_latch[row] = rc_ghost_contains_h(rc, key, h) ? 1 : 0;
        rc_dedup_add_h(rc, key, h, row);
        rc->n_miss_ids += 1;
      }
      rc->miss_pos[rc->n_miss_pos] = (int32_t)position;
      rc->miss_row[rc->n_miss_pos] = row;
      rc->n_miss_pos += 1;
    }
  }
  rc->hits += rc->n_hit;
  rc->misses += rc->n_miss_pos;
  // Admission side, same treatment: each distinct miss id's record line is
  // prefetched for the group before its rows are inserted, in trace order as
  // before. The ring and free-stack heads are deliberately not hinted -- they
  // are walked sequentially from a moving head, so the hardware prefetcher is
  // ahead of us there, and hinting them measured slower.
  for (int32_t begin = 0; begin < rc->n_miss_ids; begin += RC_PF_GROUP) {
    int32_t end = begin + RC_PF_GROUP;
    if (end > rc->n_miss_ids) end = rc->n_miss_ids;
    for (int32_t row = begin; row < end; row++) {
      uint64_t h = hg[row - begin] = rc_mix64((uint64_t)rc->miss_ids_buf[row]);
      __builtin_prefetch(&rc->table[(int64_t)(h & (uint64_t)rc->table_mask)], 0, 3);
    }
    for (int32_t row = begin; row < end; row++) {
      int32_t slot = rc_insert_h(rc, rc->miss_ids_buf[row], hg[row - begin], rc->ghost_latch[row]);
      if (slot == -2) {
        PyErr_SetString(PyExc_KeyError, "row is already cached");
        return NULL;
      }
      if (slot < 0) continue;  // no slot could be taken: not cached this time
      int32_t existing = rc_store_find(rc, slot);
      if (existing != RC_NOT_FOUND) {
        rc->store_row[existing] = row;  // dict update keeps first-insert order
      } else {
        rc_store_add(rc, slot, rc->n_store);
        rc->store_slot[rc->n_store] = slot;
        rc->store_row[rc->n_store] = row;
        rc->n_store += 1;
      }
    }
  }
  for (int32_t i = 0; i < rc->n_hit; i++) {
    rc_copy_row(rc, rc->stage_plane, (const uint8_t* const*)rc->pool_plane, rc->hit_pos[i],
                rc->hit_slot[i]);
  }
  return Py_BuildValue("ii", rc->n_hit, rc->n_miss_ids);
}

// The store half of the batch: misses stage from the fetched planes and the
// pool absorbs the rows plan() reserved store slots for.
static PyObject* py_row_cache_store(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  for (int32_t i = 0; i < rc->n_miss_pos; i++) {
    rc_copy_row(rc, rc->stage_plane, (const uint8_t* const*)rc->miss_plane, rc->miss_pos[i],
                rc->miss_row[i]);
  }
  for (int32_t i = 0; i < rc->n_store; i++) {
    rc_copy_row(rc, rc->pool_plane, (const uint8_t* const*)rc->miss_plane, rc->store_slot[i],
                rc->store_row[i]);
  }
  Py_RETURN_NONE;
}

static PyObject* py_row_cache_reserve(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int32_t slot = rc_reserve(rc, (int64_t)key);
  if (slot < 0) Py_RETURN_NONE;
  return PyLong_FromLong(slot);
}

static PyObject* py_row_cache_confirm(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  return PyLong_FromLong(rc_confirm(rc, slot));
}

static PyObject* py_row_cache_cancel(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  rc_cancel(rc, slot);
  Py_RETURN_NONE;
}

static PyObject* py_row_cache_get(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int32_t slot = rc_get(rc, (int64_t)key);
  if (slot < 0) Py_RETURN_NONE;
  return PyLong_FromLong(slot);
}

static PyObject* py_row_cache_is_ready(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  long long key;
  if (!PyArg_ParseTuple(args, "OL", &capsule, &key)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  int64_t pos = rc_tbl_probe(rc, (int64_t)key, rc_mix64((uint64_t)key));
  return PyBool_FromLong(pos >= 0 && rc_state_of(rc->table[pos].flags) == RC_READY);
}

static PyObject* py_row_cache_state(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* capsule;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &capsule, &slot)) return NULL;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  if (slot < 0 || (int64_t)slot >= rc->capacity) {
    PyErr_SetString(PyExc_ValueError, "row cache slot index out of range");
    return NULL;
  }
  return PyLong_FromLong(rc_slot_state(rc, slot));
}

static PyObject* int_list(const int32_t* values, int64_t count) {
  PyObject* list = PyList_New(count);
  if (!list) return NULL;
  for (int64_t i = 0; i < count; i++) {
    PyObject* item = PyLong_FromLong(values[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

static PyObject* key_list(const int64_t* values, int64_t count) {
  PyObject* list = PyList_New(count);
  if (!list) return NULL;
  for (int64_t i = 0; i < count; i++) {
    PyObject* item = PyLong_FromLongLong(values[i]);
    if (!item) {
      Py_DECREF(list);
      return NULL;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

// The RowPlan the last row_cache_read produced, as lists: the parity tests
// compare these against _PyS3FifoReference.plan's fields. Diagnostic only:
// the hot path never calls it.
static PyObject* py_row_cache_plan_arrays(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* values[7];
  values[0] = int_list(rc->hit_pos, rc->n_hit);
  values[1] = int_list(rc->hit_slot, rc->n_hit);
  values[2] = key_list(rc->miss_ids_buf, rc->n_miss_ids);
  values[3] = int_list(rc->miss_pos, rc->n_miss_pos);
  values[4] = int_list(rc->miss_row, rc->n_miss_pos);
  values[5] = int_list(rc->store_slot, rc->n_store);
  values[6] = int_list(rc->store_row, rc->n_store);
  // (miss_ids is values[2]; no duplicate.)
  static const char* names[7] = {"hit_pos",  "hit_slot",   "miss_ids", "miss_pos",
                                 "miss_row", "store_slot", "store_row"};
  for (int i = 0; i < 7; i++) {
    if (!values[i] || PyDict_SetItemString(result, names[i], values[i]) != 0) {
      Py_XDECREF(values[i]);
      Py_DECREF(result);
      return NULL;
    }
    Py_DECREF(values[i]);
  }
  return result;
}

// _PyS3FifoReference queue/ghost state for per-request parity: (key, freq)
// pairs in queue order, ghost keys in demotion order, resident/loading keys.
static PyObject* py_row_cache_debug_state(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* queues[2];
  const int32_t* rings[2] = {rc->small_ring, rc->main_ring};
  const int32_t heads[2] = {(int32_t)rc->small_head, (int32_t)rc->main_head};
  const int64_t counts[2] = {rc->small_count, rc->main_count};
  for (int q = 0; q < 2; q++) {
    PyObject* list = PyList_New(counts[q]);
    if (!list) {
      Py_DECREF(result);
      return NULL;
    }
    for (int64_t i = 0; i < counts[q]; i++) {
      int32_t slot = rings[q][(heads[q] + i) % rc->capacity];
      const rc_entry_t* e = rc_slot_entry(rc, slot);  // a queued slot is always bound
      PyObject* pair = Py_BuildValue("Li", e ? e->key : 0, e ? (int)rc_freq_of(e->flags) : 0);
      if (!pair) {
        Py_DECREF(list);
        Py_DECREF(result);
        return NULL;
      }
      PyList_SET_ITEM(list, i, pair);
    }
    queues[q] = list;
  }
  if (PyDict_SetItemString(result, "small", queues[0]) != 0 ||
      PyDict_SetItemString(result, "main", queues[1]) != 0) {
    Py_DECREF(queues[0]);
    Py_DECREF(queues[1]);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(queues[0]);
  Py_DECREF(queues[1]);
  PyObject* ghost = PyList_New(rc->ghost_count);
  if (!ghost) {
    Py_DECREF(result);
    return NULL;
  }
  int64_t i = 0;
  for (int32_t e = rc->ghost_head; e >= 0; e = rc->ghost_next[e]) {
    PyObject* item = PyLong_FromLongLong(rc_ghost_key_of(rc, e));
    if (!item) {
      Py_DECREF(ghost);
      Py_DECREF(result);
      return NULL;
    }
    PyList_SET_ITEM(ghost, i++, item);
  }
  if (PyDict_SetItemString(result, "ghost", ghost) != 0) {
    Py_DECREF(ghost);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(ghost);
  PyObject* ready = PyList_New(rc->ready_count);
  PyObject* loading = PyDict_New();  // key -> slot, mirroring _loading
  if (!ready || !loading) {
    Py_XDECREF(ready);
    Py_XDECREF(loading);
    Py_DECREF(result);
    return NULL;
  }
  int64_t n_ready = 0;
  for (int64_t slot = 0; slot < rc->used; slot++) {
    const rc_entry_t* e = rc_slot_entry(rc, (int32_t)slot);
    if (e == NULL) continue;  // unbound slot: FREE, which the old walk skipped as well
    uint8_t st = rc_state_of(e->flags);
    if (st == RC_READY) {
      PyObject* item = PyLong_FromLongLong(e->key);
      if (!item) {
        Py_DECREF(ready);
        Py_DECREF(loading);
        Py_DECREF(result);
        return NULL;
      }
      PyList_SET_ITEM(ready, n_ready++, item);
    } else if (st == RC_LOADING) {
      PyObject* item = PyLong_FromLongLong(e->key);
      PyObject* value = PyLong_FromLong((long)slot);
      int added = !item || !value || PyDict_SetItem(loading, item, value) != 0;
      Py_XDECREF(item);
      Py_XDECREF(value);
      if (added) {
        Py_DECREF(ready);
        Py_DECREF(loading);
        Py_DECREF(result);
        return NULL;
      }
    }
  }
  if (PyDict_SetItemString(result, "ready", ready) != 0 ||
      PyDict_SetItemString(result, "loading", loading) != 0) {
    Py_DECREF(ready);
    Py_DECREF(loading);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(ready);
  Py_DECREF(loading);
  // Slot-allocation state: the parity tests compare the free list (order
  // matters: Python _free.pop() takes the most recently released slot), the
  // high-water mark, and the eviction gate against the oracle's privates.
  PyObject* free = PyList_New(rc->free_top);
  if (!free) {
    Py_DECREF(result);
    return NULL;
  }
  for (int64_t i = 0; i < rc->free_top; i++) {
    PyObject* item = PyLong_FromLong((long)rc->free_stack[i]);
    if (!item) {
      Py_DECREF(free);
      Py_DECREF(result);
      return NULL;
    }
    PyList_SET_ITEM(free, i, item);
  }
  if (PyDict_SetItemString(result, "free", free) != 0) {
    Py_DECREF(free);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(free);
  PyObject* scalars =
      Py_BuildValue("{s:L,s:L}", "used", rc->used, "has_evicted", (long long)rc->has_evicted);
  if (!scalars || PyDict_Update(result, scalars) != 0) {
    Py_XDECREF(scalars);
    Py_DECREF(result);
    return NULL;
  }
  Py_DECREF(scalars);
  return result;
}

// Diagnostic: raw arrays so the parity harness can diff C internals against
// the oracle's privates when the derived debug_state disagrees (never used by
// the hot path or DiskRowCache).
static PyObject* py_row_cache_raw_dump(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  PyObject* result = PyDict_New();
  if (!result) return NULL;
  PyObject* items[10];
  items[0] = NULL;  // unused; result is returned directly (no stray ref to leak)
  int64_t n = rc->used < rc->capacity ? rc->used : rc->capacity;
  items[1] = PyList_New(n);              // state
  items[2] = PyList_New(n);              // queue
  items[3] = PyList_New(n);              // freq
  items[4] = PyList_New(n);              // key
  items[5] = PyList_New(rc->capacity);   // small ring raw
  items[6] = PyList_New(rc->capacity);   // main ring raw
  items[7] = PyList_New(rc->free_top);   // free stack raw
  items[8] = PyList_New(rc->ghost_cap);  // ghost key raw (order via next links)
  items[9] = PyList_New(rc->ghost_cap);  // ghost next raw
  for (int i = 1; i < 10; i++) {
    if (!items[i]) {
      for (int k = 1; k < 10; k++) Py_XDECREF(items[k]);
      Py_DECREF(result);
      return NULL;
    }
  }
  for (int64_t i = 0; i < n; i++) {
    const rc_entry_t* e = rc_slot_entry(rc, (int32_t)i);
    uint8_t flags = e ? e->flags : rc_pack_flags(0, RC_FREE, RC_UNQUEUED);
    PyList_SET_ITEM(items[1], i, PyLong_FromLong(rc_state_of(flags)));
    PyList_SET_ITEM(items[2], i, PyLong_FromLong(rc_queue_of(flags)));
    PyList_SET_ITEM(items[3], i, PyLong_FromLong(rc_freq_of(flags)));
    PyList_SET_ITEM(items[4], i, PyLong_FromLongLong(e ? e->key : 0));
  }
  for (int64_t i = 0; i < rc->capacity; i++) {
    PyList_SET_ITEM(items[5], i, PyLong_FromLong(rc->small_ring[i]));
    PyList_SET_ITEM(items[6], i, PyLong_FromLong(rc->main_ring[i]));
  }
  for (int64_t i = 0; i < rc->free_top; i++)
    PyList_SET_ITEM(items[7], i, PyLong_FromLong(rc->free_stack[i]));
  for (int64_t i = 0; i < rc->ghost_cap; i++) {
    PyList_SET_ITEM(items[8], i, PyLong_FromLongLong(rc_ghost_key_of(rc, (int32_t)i)));
    PyList_SET_ITEM(items[9], i, PyLong_FromLong(rc->ghost_next[i]));
  }
  int rc_set = 0;
  rc_set |= PyDict_SetItemString(result, "state", items[1]);
  rc_set |= PyDict_SetItemString(result, "queue", items[2]);
  rc_set |= PyDict_SetItemString(result, "freq", items[3]);
  rc_set |= PyDict_SetItemString(result, "keys", items[4]);
  rc_set |= PyDict_SetItemString(result, "small_ring", items[5]);
  rc_set |= PyDict_SetItemString(result, "main_ring", items[6]);
  rc_set |= PyDict_SetItemString(result, "free_raw", items[7]);
  rc_set |= PyDict_SetItemString(result, "ghost_keys", items[8]);
  rc_set |= PyDict_SetItemString(result, "ghost_next", items[9]);
  // Tombstone occupancy, so the probe-tail soak can watch the sweep trigger
  // (0.6 of the table) without instrumenting the hot path.
  PyObject* tu = PyLong_FromLongLong(rc->n_used);
  PyObject* tt = PyLong_FromLongLong(rc->n_tomb);
  PyObject* gt = PyLong_FromLongLong(rc->ghost_n_tomb);
  if (!tu || !tt || !gt) {
    rc_set = -1;
  } else {
    rc_set |= PyDict_SetItemString(result, "table_used", tu);
    rc_set |= PyDict_SetItemString(result, "table_tombstones", tt);
    rc_set |= PyDict_SetItemString(result, "ghost_tombstones", gt);
  }
  Py_XDECREF(tu);
  Py_XDECREF(tt);
  Py_XDECREF(gt);
  PyObject* hs = PyLong_FromLongLong(rc->small_head);
  PyObject* hm = PyLong_FromLongLong(rc->main_head);
  PyObject* hg = PyLong_FromLong(rc->ghost_head);
  PyObject* tg = PyLong_FromLong(rc->ghost_tail);
  if (!hs || !hm || !hg || !tg) {
    rc_set = -1;
  } else {
    rc_set |= PyDict_SetItemString(result, "head_small", hs);
    rc_set |= PyDict_SetItemString(result, "head_main", hm);
    rc_set |= PyDict_SetItemString(result, "head_ghost", hg);
    rc_set |= PyDict_SetItemString(result, "tail_ghost", tg);
  }
  Py_XDECREF(hs);
  Py_XDECREF(hm);
  Py_XDECREF(hg);
  Py_XDECREF(tg);
  for (int i = 1; i < 10; i++) Py_DECREF(items[i]);
  if (rc_set != 0) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}

// Hit/miss counters and queue sizes for DiskRowCache.stats; the s/m/g fields
// keep the probe surface the Python class exposed through privates.
static PyObject* py_row_cache_stats(PyObject* self, PyObject* capsule) {
  (void)self;
  row_cache_t* rc = rc_from(capsule);
  if (!rc) return NULL;
  return Py_BuildValue("{s:L,s:L,s:L,s:l,s:l,s:l,s:L,s:L,s:L,s:L}",  //
                       "hits", rc->hits, "misses", rc->misses, "resident",
                       rc->ready_count + rc->loading_count, "s", rc->small_count, "m",
                       rc->main_count, "g", rc->ghost_count, "capacity", rc->capacity, "small_cap",
                       rc->small_cap, "main_cap", rc->main_cap, "ghost_cap", rc->ghost_cap);
}

static PyMethodDef methods[] = {
    {"row_cache_create", py_row_cache_create, METH_VARARGS, NULL},
    {"row_cache_read", py_row_cache_read, METH_VARARGS, NULL},
    {"row_cache_store", py_row_cache_store, METH_O, NULL},
    {"row_cache_reserve", py_row_cache_reserve, METH_VARARGS, NULL},
    {"row_cache_confirm", py_row_cache_confirm, METH_VARARGS, NULL},
    {"row_cache_cancel", py_row_cache_cancel, METH_VARARGS, NULL},
    {"row_cache_get", py_row_cache_get, METH_VARARGS, NULL},
    {"row_cache_is_ready", py_row_cache_is_ready, METH_VARARGS, NULL},
    {"row_cache_state", py_row_cache_state, METH_VARARGS, NULL},
    {"row_cache_plan_arrays", py_row_cache_plan_arrays, METH_O, NULL},
    {"row_cache_debug_state", py_row_cache_debug_state, METH_O, NULL},
    {"row_cache_stats", py_row_cache_stats, METH_O, NULL},
    {"row_cache_raw_dump", py_row_cache_raw_dump, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_b12x_row_cache",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__b12x_row_cache(void) {
  PyObject* result = PyModule_Create(&module);
  if (result && PyModule_AddIntConstant(result, "ABI_VERSION", 1) < 0) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}
