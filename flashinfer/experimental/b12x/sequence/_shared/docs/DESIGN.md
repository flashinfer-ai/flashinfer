# Row cache for disk-resident PLE / Engram tables

Design document and formal-model home for `DiskRowCache` (`../disk_table.py`)
and the row-cache index (`../row_cache.py` → the C extension
`b12x/loader/_row_cache.c`); the Quint model `slotpool.qnt` and its
counterexample trace sit beside this file.

**Decision: ship the C host-tier row cache, enabled with the 1 GiB pool
default, and do not build the zero-copy reclamation path.** The cache is a
single S3-FIFO index (§1 "Policy selection"); the SIEVE and W-TinyLFU Python
indices and the `B12X_ROW_CACHE_POLICY` knob that selected them were
branch-local experiments and are gone, with the per-request parity oracle
`_PyS3FifoReference` surviving in `tests/.../test_row_cache_policies.py`. The
cache is active only when `B12X_ROW_CACHE_BYTES` / `_ROWS` / `_TIER` is set.
§1 gives the problem and the policy decision; §2 the zero-copy decision and its
gate; §3–§5 the correctness invariant, its formal proof, and the gates that
must pass before this ships on; §6 the end-to-end measurement that fixes the
default.

## 1. Problem

A PLE (n-gram embedding) table is **320,001,536 rows × (80 B packed-NVFP4 weight +
10 B E4M3 group-64 scale) = 26.9 GiB**, in 128 shard files of 2,500,012 rows each
(`split_ngram_parts: 128`; measured from the CSF QAD checkpoint headers), far too
large for VRAM. `DiskRowCache` reads the rows a step needs over io_uring
(O_DIRECT, fixed buffers/files, 4 KiB block coalescing) into a batch-sized
mapped-host staging buffer; a Triton decode kernel then gathers rows from it.
Every step re-reads its rows — the cache stages exactly one batch and retains
nothing. Access is Zipfian: measured over 3.03 M row reads (1.05 M distinct)
from a real `llm-inference-bench` run, 65 % of reads are repeats, the top 1 %
of rows carry 42 %, and ~72 % of distinct rows are read exactly once. A small
persistent row cache can skip most of the re-reads.

The cost model that motivates the C index: a Python id→slot resolution plus a
`pool[slot] → staging[i]` copy on hit costs **8.4 µs + 181 ns/id**. At decode
batch sizes (n≈24–64) that is ≤20 µs/step — negligible. At prefill chunk sizes
(n = tokens × head_count, up to 96304) it is ~17 ms/step of Python, which is
where added latency would land. The shipped C index removes the per-id Python
term from the hot path.

### Policy selection

The cache ships one replacement policy, **S3-FIFO** (Yang et al., SOSP'23): a
frequency-filtered small queue `S` (10 % of capacity) feeds a main queue `M`
that re-inserts rows still holding 2-bit counter credit, over a ghost list `G`
(90 % of capacity); a row promotes `S→M` at threshold 2. It led the measured
comparison against SIEVE and W-TinyLFU at every operating capacity tested, so
those two Python indices and the `B12X_ROW_CACHE_POLICY` knob that selected
them were retired as branch-local experiments and never reached `main`. The
evidence is carried by the tests, not a separate ledger: the C index is checked
per-request against `_PyS3FifoReference` (the oracle mirroring libCacheSim's
`S3FIFO.c`) and against `cachesim` on the production PLE row-id trace prefix
(`test_lcs_all_prefix_matches_reference`) — negative obj-id rows in that prefix
are KEPT under `strtol` semantics, because dropping them desynchronised an
earlier replay series. §6 reports the end-to-end measurement that fixes the
1 GiB default.

## 2. Zero-copy path — decision: not built

The full zero-copy design would make the pool the single store and let the
decode kernel gather directly from it, so hits move no bytes and misses DMA
straight into their final slot:

- **Pool = registered DMA target.** Allocate the pool once (fixed base
  address); register its pages with `IORING_REGISTER_BUFFERS`. A miss issues
  `io_uring_prep_read_fixed` writing the row *into its pool slot* — no staging
  buffer, no bounce copy on the miss path.
- **Per-position slot index.** A persistent `slot[max_lookups]` int buffer
  (fixed address, so it is CUDA-graph-capturable) maps each of a step's
  `count` gather positions to a pool slot; the decode kernel's row-index
  operand becomes `slot[i]` against the pool base. This unifies the disk path
  with the existing resident gather, which already indexes the table by id.
- **C owns the plan.** Hit/miss resolution runs in C.

**The unbuilt half is the decision, not an oversight.** The prefill-exposure
gate found host work in the cached read path is **overlapped by GPU compute,
not exposed**: injecting up to 4 ms/call of CPU on the step thread moved
throughput inside the arm-to-arm noise; only an 8 ms/call injection (20 % of
the ~40.6 ms chunk cycle) cost 4.6 %. With the C port the host step already
fits inside ~79 % of the GPU chunk cycle, so removing the residual Python/copy
could return at most ~0.5–1 % on this host — inside noise. The gate to revisit
is a GPU ≥1.26× faster than the server21 300 W Max-Q board, or a Vera-class
host constant (~40 ns/id), **measured on the actual hardware**, before the
zero-copy prize justifies the deferred-reclamation risk in §3.

What the shipped C port does implement: hit/miss resolution and replacement
policy as the separate extension `_row_cache.c` (`S3FifoRowCache`) — the
open-addressed map, queues, and miss-id list. The io_uring reader
(`_ple_reader.c`, `ple_reader_run`) and its submit path are unchanged; the miss
wave still stages through the bounce buffer.

### Placements (the shipped host tier)

| Hardware | source | tier | hit cost | miss cost |
|---|---|---|---|---|
| GB10 / DGX Spark (unified) | disk | host pool | kernel reads host-mapped slot | DMA → pool slot |
| Discrete GPU (SM120/100) | disk | host-RAM pool | host copy to device slot | DMA → host slot, 1× H2D |

`row_cache.py` / `disk_table.py` implement a single host tier: rows stage
through mapped, page-locked host memory and the persistent pool lives in host
RAM. The `host_rows` source (a shard already resident as CPU tensors) shares
this layout.

## 3. The one dangerous invariant — and its proof

Everything is bookkeeping except reclamation of shared bytes: **data an
outstanding decode still reads must not be swapped under it.** The shipped C has
two distinct planes, each safe for a different reason; `slotpool.qnt` models
both, and importantly does *not* claim the C guards pool eviction against
in-use slots — it has no such check.

- **Pool plane — an in-flight fetch (`LOADING`) is never a victim (structural).**
  `rc_reserve` claims a `FREE` slot and sets `LOADING`; the slot joins no ring.
  The eviction pops (`rc_pop_small`, `rc_pop_main`) walk only the `SMALL`/`MAIN`
  rings of `READY` slots, so *"no pop can reach"* a `LOADING` slot
  (`_row_cache.c:30-31`); and `claim_for_loading` only takes `FREE`/fresh slots,
  so it is never re-handed to a second fetch until `rc_confirm`/`rc_cancel` ends
  the fetch. This follows from the state machine, not an eviction-time test.
- **Staging plane — the real reuse hazard.** The hot path is not zero-copy: a hit
  is `memcpy`'d from its pool slot into the batch's staging plane (`rc_copy_row`,
  `_row_cache.c:1089-1092`) and the decode gathers from **staging** bytes, not the
  pool slot. A pool slot may therefore be evicted and rebound later in the same
  batch harmlessly, because `store()` rewrites pool bytes only *after* `read()` has
  copied every hit out (`_row_cache.c:33-35, 1106-1108`) and `DiskRowCache._lock`
  serialises the prefetch worker's pool writes with the transaction. The genuine
  use-after-free is on the **shared batch staging buffer**: the next transaction
  must not overwrite a staging slot while an unretired decode from the previous one
  is still reading it. `disk_table.py transaction()` / `_cache_done` (the next
  transaction's `stream.wait_event`) fences that reuse.

The model carries these as the pool `loading_out` set plus a separate staging plane
(`stg_key`, `given_key`, `outstanding`), with actions mirroring the C:
`reserve/confirm/cancel/evict` on the pool (no staged check on `evict`), and
*stage write / decode begin / fence* on the staging buffer. `DEFER_RECYCLE` is the
code-level lever on the staging fence only — when true, a write to a staging slot
still pinned by an outstanding decode is refused; there is no pool-eviction guard
to toggle.

Proven in Quint + Apalache (`slotpool.qnt`):

- **`verified` (`DEFER_RECYCLE = true`):** `NoUseAfterFree` — every `loading_out`
  slot is `LOADING` and unqueued, and every `outstanding` staging slot still holds
  the bytes it was handed (`stg_key == given_key`) — holds by **inductive
  invariant** (all three Apalache obligations report `NoError`).
- **`broken` (`DEFER_RECYCLE = false`):** Apalache returns a concrete
  **counterexample** (`broken-use-after-free.itf.json`): a staging slot is
  overwritten while an unretired decode reads it, so its bytes no longer match what
  the decode was given. This keeps the `verified` gate non-vacuous.

## 4. What the formal model does NOT cover

- S3-FIFO eviction *quality* (hit-rate optimality) — that is the paper's claim,
  not verified here; only structural correctness of the lifecycle is proven.
- The Triton gather kernel's byte-level correctness — covered by the
  byte-identity unit tests in `test_disk_row_cache.py` (and, for the index
  itself, the per-request C-vs-oracle parity in `test_row_cache_policies.py`).
- Real CUDA event ordering / driver semantics — the model abstracts the
  outstanding-fetch set (`loading_out`) and the staging buffer's `outstanding`
  pins; the
  implementation must honour the same protocol with per-batch `Event`s.
- DMA failure / short-read paths (`_ple_reader.c` already guards these).

## 5. Correctness gates (must pass before this ships on)

### CI gates — run on flashinfer's PR-CI runners (cpu, sm75, sm86, h100)

Each of these produces a real pass/fail on those runners; none silently skips
on the hardware it names.

1. `quint verify slotpool.qnt --main verified --invariant=NoUseAfterFree
   --inductive-invariant=IndInv` → `[ok]` (unbounded). CPU-only (Apalache/Z3).
2. `quint verify slotpool.qnt --main broken --invariant=NoUseAfterFree` →
   `[violation]` (negative test; proves the gate is non-vacuous).
3. `tests/experimental/b12x/sequence/test_row_cache_policies.py` green — the
   GPU-free correctness suite (imports neither torch nor CUDA): the build
   checks (`test_c_extension_builds`,
   `test_build_failure_raises_not_falls_back`, `test_facade_rejects_bad_capacity`),
   the C-vs-`_PyS3FifoReference` per-request parity
   (`test_hand_micro_batches_match_reference`,
   `test_micro_sequence_replay_matches_reference`,
   `test_reserve_confirm_admits_via_ghost_rule`,
   `test_random_churn_with_prefetch_lifecycle`), the policy invariants
   (`test_threshold_promotion_keeps_counter`,
   `test_duplicate_ids_fetch_and_store_once`), `test_stats_surface`, and the
   pure knob validation (`test_resolve_row_cache_size_precedence`,
   `test_validate_row_cache_tier` — the single-host-tier contract, so the
   "`B12X_ROW_CACHE_TIER` rejects a non-`host` value" rule is gated here, not on
   a board). Its only skip is "no C99 compiler on this host," which
   `test_c_extension_builds` turns into a hard check wherever a compiler exists.

### Hardware gate — consumer-Blackwell SM120/SM121 only (not PR CI)

4. `tests/experimental/b12x/sequence/test_disk_row_cache.py` byte-identity
   across the host tier and both sources (disk + host_rows). `DiskRowCache`
   construction requires CUDA, and every device test in the file is gated by
   `require_b12x()` (`conftest.py`), which **skips** unless
   `torch.cuda.get_device_capability()` is `(12, 0)` or `(12, 1)`. No flashinfer
   PR-CI runner is SM12x (the matrix is sm75/sm86/h100/cpu), so this suite
   always skips there — it is a developer-board / SM12x self-hosted (or nightly)
   check (server21 in this work), not a PR gate. The construction-time raise it
   also exercises is the same `B12X_ROW_CACHE_TIER` contract, already covered
   GPU-free by gate 3, so no correctness claim is gated *only* on hardware.

## 6. End-to-end result — the measurement behind the decision

The optimized C port (inline key+slot open-addressed S3-FIFO + batched/512 B
io_uring reader) measured on the cloned production spec on server21 (native
SM120, RTX PRO 6000 96 GiB, LMCache recurrent connector, 1 MiB context,
`--gpu-memory-utilization 0.985`, MTP k=3): full lil-bench sweep
`--concurrency 1,2,4,8,16` × `--contexts 0,16k,64k,128k`, 45 s/cell, GSM8K c=1
profile per arm, **2 rounds reversed arm order**, pooled per cell, each arm
preceded by a `CACHE-STATE-VERIFIED` assertion that the host tier is genuinely
enabled. Pool sizes: the default arm is 11 930 464 rows × 90 B = exactly
1.0000 GiB; the small arm is 100 000 rows (~8.6 MiB). Both against the
disk-only `none` baseline.

| optimized C arm | decode pooled Δ vs none | t | c≥8 | gsm8k (pooled) |
|---|---|---|---|---|
| **1 GiB default** | **+6.30 %** | +5.9 | **+11.91 %** | **+4.00 %** |
| 100 k rows | +4.01 % | +4.9 | +6.80 % | +3.56 % |

**Decision: keep the 1 GiB pool as the default.** It is the stronger arm —
+1.86 % pooled and +4.87 % at c≥8 over 100 k (c=1 is a wash), consistent with
the C-index CPU replay showing the larger pool is cheaper per lookup and hits
more often. The win is concentrated in the concurrent regime; single-stream sits
inside the round-to-round spread (2.8 % mean-abs per cell overall, 1.25 % at
c≥8 — the high-variance cells are all single-stream long-context, the regime
§2 shows is host-overlapped and low-signal). Acceptance length is stable across
all arms (3.0–3.1): a throughput effect, not a numerics effect.
