# VibeCUDA Kimi-K3 AttnRes — Memory-Ordering and Synchronization Safety Audit

Scope: the two production payloads compiled by this package

* `csrc/vibecuda_kimi_k3_attn_res/vibecuda_kimi_k3_attn_res_kernels_sm100.cu`
  (SM100 / GB200, sha256 `39e050413be8deb931084c1748f6a92e0bd48e73daf6a65d037b2d592ca471ac`),
* `csrc/vibecuda_kimi_k3_attn_res/vibecuda_kimi_k3_attn_res_kernels_sm103.cu`
  (SM103 / B300, sha256 `99fd3f0d7d87de7b72d20da254aeab1fa60ecfbe8d69655f8a28c9d29703cde3`),

plus the TVM-FFI binding / host dispatch entry points in the same files and the
module generator `vibecuda_jit.py`. The two payloads differ in 16 diff hunks
only: arch string and provenance-hash comments, the persistent-scheduler grid
constant `NSMK` (152 -> 148, matching B300's 148 SMs), and comment trims.
Every synchronization construct is line-identical across the payloads, so a
single audit covers both architectures; line numbers below cite the SM100
payload. Publication cleanup changes only comments in the generated kernel
pair. The complete non-comment C++ token streams, including preprocessor
directives and string literals, were compared before and after cleanup and
are identical. This audit traces the actually-dispatched device code; it is
not a replacement for numerical, graph, memory or performance qualification.

PTX ISA references below are the authoritative named sections (section
numbers are for ISA 9.4): Memory Consistency Model / Proxies (8.6),
`cp.async.bulk.tensor` (bulk-asynchronous copy, 9.7.15.4 vicinity),
Asynchronous Copy Completion Mechanisms (9.7.10.28.1 / .2, incl.
`cp.async.mbarrier.arrive` and `mbarrier::complete_tx`), mbarrier state /
tracking / wait (9.7.15.16 vicinity, `mbarrier.try_wait.parity`), the Async
Proxy (8.6 proxy ordering, `fence.proxy.async`), `membar`/`fence`
(9.7.18.6.5 vicinity, `fence.mbarrier_init.release.cluster`), and the
`tcgen05` TMEM lifecycle (`tcgen05.alloc/relinquish_alloc_permit/st/wait::st/
ld/wait::ld/dealloc`).

## Platform synchronization surface

* One kernel launch per `launch()`, on the caller's current stream. No side
  streams, no host work in the hot path, no allocation in the hot path
  (verified by `test_launch_makes_no_allocation_and_replays_in_a_graph`).
* One-time host work lives in `prepare()`: JIT load (JitSpecNvcc file lock),
  `module.warmup()` (route-table init, `cudaFuncSetAttribute` for the
  production SMEM sizes, first-launch driver priming), then route-id lookup.
  All of it is ordered before the first launch through the calling thread and
  the stream.
* PDL: when `enable_pdl` is true the host sets
  `cudaLaunchAttributeProgrammaticStreamSerialization`; otherwise a plain
  launch is used (the `K3_PLAIN` env knob forces this). Every kernel begins
  with `griddepcontrol.wait` (`pdl_wait()` at the head, l.3308 and peers) and
  ends with `griddepcontrol.launch_dependents`
  (`pdl_launch_dependents...`, l.3302-3306, `"memory"` clobber). The
  early-trigger variants (`pdlr_kernel`, `epdl_kernel`) fire
  `launch_dependents` immediately after their final global-memory *reads*
  with the memory clobber; the soundness argument is reproduced in the source
  comments (l.3280-3306, l.3522-3548): every pre-trigger CTA has issued all
  of its global loads and none of its stores, so a dependent grid only
  overlaps with read-only tail execution.

## Per-construct audit (production dispatch ladder)

Route table: impl85 family (chp persistence), impl87/chtm, impl97 leanF,
impl108 (wpr8), impl112 (pdlr), impl126/chtg, impl131/chtgd, impl133,
impl134 (chtmd), impl136 chpg, impl138 (epdl), impl139 (chtclc), impl140
(chtcw, cluster multicast, env-opt-in only). Each construct below is
identified with its producer side, consumer side, accounting, and tails.

### 1. mbarrier initialization and publication

All kernels that use mbarriers initialize them under a single guarded thread
(e.g. `chp_kernel` l.2346-2354), then call `fence_mbar_init()` =
`fence.mbarrier_init.release.cluster` (l.2226-2228) followed by
`__syncthreads()` before the barrier's first use. Generic-proxy init ->
async-proxy use requires exactly this release fence (PTX 8.6; the hazard is
documented in the source at l.2223-2225). Cluster-scope consumers are covered
because the fence releases at cluster scope. Contributions and waits below
take the freshly published phase-0 barriers.

### 2. TMA bulk row-staging rings (`cp.async.bulk.tensor.shared::cta.global.mbarrier::complete_tx::bytes`)

chp/chpg (impl85/133/136), chtm family (87/134/140), chtg/chtgd (126/131),
chtclc (139), epdl (138). Pattern per stage slot pair `bar_f[slot]` /
`bar_e[slot]`, `slot = gc % NS`, `phase = (gc / NS) & 1`:

* init: `bar_f[*]` arrival count 1, `bar_e[*]` arrival count 256 (or the
  documented variant counts). Induction `gc >= NS` gate applies identically
  on producer and consumer, so slot and phase are mirrored.
* producer (warp 8, lane 0 elected): for reuse (`gc >= NS`)
  `mbar_wait(&bar_e[slot], phase ^ 1)` — the slot is free — then
  `mbar_arrive_expect_tx(&bar_f[slot], an * ROWB)` and one 1-D
  `cp.async.bulk` copy of `ROWB` bytes per live row; chunk tails predicate
  the copies with `r < an`, and `expect_tx` is set from the same `an`, so
  transaction bytes always match issued bytes exactly.
* consumers (256 threads, warps 0..7): `mbar_wait(&bar_f[slot], phase)`
  (phase-parity completion check per 9.7.10.28.x) before any LDS of the
  stage; the observed `complete_tx` completion supplies the async-proxy
  read ordering for generic-proxy consumption (PTX 8.6 + 9.7.10.28.2).
* release: after each thread's final read of the slot data, all 256 consumer
  threads `mbar_arrive(&bar_e[slot])` (unguarded — tail chunks still have the
  full CTA participate), so the producer's `phase ^ 1` wait on reuse pairs
  with exactly 256 arrivals. No arrival can be skipped by predication or by
  token-count tails because the arrive sits outside the `r < an` predicate.
* `chpg_kernel` uses the identical ring with a static tile count and
  `static_assert(NCH <= 2 * NS)` (l.2666).

### 3. TMEM lifecycle (tcgen05, chtm/chtclc/chtg/chtgd families)

* `tcgen05.alloc(cta_group::1)` + `relinquish_alloc_permit` by the designated
  warpgroup (warp 8 band ownership, l.2237-2290); the 32-bit TMEM address is
  published through SMEM (`st8`) and made visible with a `__syncthreads()`
  before any consumer reads it — no cross-warp hazard on `taddr`.
* writes: `tcgen05.st` followed by `tcgen05.wait::st` before any subsequent
  `tcgen05.ld` from the same region (required write->read ordering; the
  helper pair `tm05_wait_st` / `tm05_wait_ld` wraps the PTX exactly).
* reads: `tcgen05.ld` followed by `tcgen05.wait::ld` before the registers are
  consumed.
* dealloc: DEAL=0 routes dealloc on warp 0 after a `__syncthreads()` that
  guarantees every consumer's `wait::ld` retired. DEAL=1 routes (impl131
  chtgd, impl134 chtmd) hand the dealloc to the allocator warpgroup through
  named barrier ID 9 (l.2207-2215): the 256 consumer threads perform a
  non-blocking `bar.arrive(9)` after their last `tcgen05.wait::ld`
  (l.4107-4109) and the 32-thread owner warp `bar.sync(9)` (288 = 256 + 32)
  before `tcgen05.dealloc` (l.4138-4141). CTAs on a persistent loop whose
  final token count is zero execute the fallback arrive (l.4126-4130), so
  the barrier completes exactly once per CTA with a CTA-uniform predicate —
  no divergence-based under/over-arrival, and the deallower cannot retire
  storage while a consumer still reads it (documented hazard at l.2207-2215).

### 4. Named (hardware) barriers

* IDs 1 and 2, count 256: the per-chunk score-freeze and out-norm cross-warp
  rendezvous (`bar_cons_sync`, l.2204). Distinct IDs per logical rendezvous
  with the documented rationale (l.2198-2203): re-using an ID would let a
  fast warp's *next-chunk* arrival count toward a previous rendezvous.
* Parity-double-buffered cross-warp partials (s_partial / t_partial pairs
  indexed by chunk parity) remove the WAR hazard on the single rendezvous —
  writers to chunk g+2's partials cannot reach that smem until the
  barrier's completion of chunk g's read, and the parity alternation gives
  one extra chunk of slack; the invariant is spelled out in the rx5 comment
  block (l.5386-5389 region).
* ID 9, count 288: the DEAL handoff in section 3. It is used by no other
  rendezvous in the payload (disjoint-ID invariant documented at
  l.2207-2212).

### 5. cp.async (16B) rings — the lean CTA-per-token family

* `fusedW_kernel` / `leanF_kernel` (impl97, small token counts): all
  data-movement overlap is per-thread register prefetch and L1 reuse;
  `__syncthreads()` is the sole block barrier (l.477/499/503, 571, 750-782,
  1143-1148, 1399-1400), executed uniformly with no early exits inside the
  loops, so the generic-proxy ordering follows from the barrier alone.
* `leanws` family (bar[0..3]-style mbarrier pairs): one mbarrier pair per
  (warp, tile) with arrival count 32; producers publish with
  `cp.async.mbarrier.arrive.noinc` (l.1756) so the completion arrive does
  *not* increment the expected-arrival count — the preset 32 corresponds
  exactly to the warp's lanes (accounting rule documented l.1731-1756, the
  "issue/completion counts cancel otherwise" trap). Phases cycle by tile
  `it & 1` and every barrier's phases are consumed strictly in order.
* `leanring` / `leanpf`: `cp.async.commit_group` / `wait_group` staging with
  a fixed group depth (9.7.10.28.1); group indices mirror between producer
  issue and consumer wait on the same induction variable.
* `leansmt`: a two-slot TMA group pipeline with the same parity discipline
  as section 2.

### 6. Publisher/consumer overlap with named barriers + TMEM (chtclc, impl139)

The persistent tile scheduler's CLC ring (l.4157+): two-slot slot rings

* `clc_tok_bar[2]` (arrival count 1, expect_tx 16B) and
  `clc_done_bar[2]` (arrival count 256), init in section 1's pattern.
* producer: for `sc >= 2`, `mbar_wait(&clc_done_bar[cbuf], cph ^ 1)` —
  the response slot is known free — then
  `fence_proxy_async_cta()` (`fence.proxy.async.shared::cta`, l.4236) to
  order the consumers' generic-proxy *reads* of the prior response against
  the async unit's overwrite, then `clusterlaunchcontrol.try_cancel` with
  `mbar_arrive_expect_tx(&clc_tok_bar[cbuf], 16)` and its completion wait
  (l.4237-4239). `sc`/`cbuf`/`cph` are derived from one monotonically
  increasing hardware-response induction on both producer and consumers
  (`cbuf = sc & 1`, `cph = (sc >> 1) & 1`), so slot/phase are mirrored.
* consumers: `mbar_wait(&clc_tok_bar[cbuf], cph)` before reading the 16-byte
  response (visibility via observed complete_tx completion, 8.6), then after
  their reads all 256 threads `mbar_arrive(&clc_done_bar[cbuf])`
  (l.4355). Zero-token CTAs exit the loop without a stale pending arrival
  because the done-barrier waits are guarded by `sc >= 2` on the producer
  side and the loop-exit state is CTA-uniform (CLC responses are broadcast
  to all threads' barriers identically).

### 7. Cluster multicast (`chtcw`, impl140 — opt-in, `cmcw_enabled`)

* Host launches with a 2-CTA cluster attribute and an even grid when the
  route is enabled.
* CTAs `cluster_sync()` before the multicast (l.3830) so the peers'
  `bar_v` mbarriers are initialized (init is published cluster-wide by the
  section-1 `fence.mbarrier_init.release.cluster`).
* rank 0 issues two `cp.async.bulk` multicast copies (cta mask `0x3`,
  l.3833-3835), one `ROWB` each; *every* peer pre-registers
  `mbar_arrive_expect_tx(bar_v, 2 * ROWB)` on its own `bar_v` (l.3832), so
  each rank's barrier completes exactly when both copies land locally.
* consumers `mbar_wait(bar_v, 0)` (single-phase use per kernel instance,
  l.3908) before reading `s_cw`/`s_ow`.

### 8. epdl single-shot staging (impl138)

`epdl_kernel` (l.3550+): one mbarrier (count 1) init + section-1 publication;
one elected thread `mbar_arrive_expect_tx((R + 2) * ROWB)` plus exactly
`R + 2` bulk copies; the early PDL trigger fires after the last global read
(l.3594, section "PDL" above); all threads `mbar_wait(s_bar, 0)` (l.3595)
before consuming. Exact byte accounting, no reuse.

### 9. Non-staged kernels

`wpr8_kernel` (impl108, l.3038+) uses only `__syncthreads()` (three sites, no
early exits between them); `pdlr_kernel` (impl112) has no interplay beyond
the PDL frame in the platform section. No async constructs, no hazards
beyond plain smem barriers.

## Balance summary (arrival/byte accounting)

| construct | expected arrivals | arrivals issued | expected tx bytes | tx bytes issued |
|---|---|---|---|---|
| chp/chpg/chtm/chtg/chtgd bar_f | 1 | 1 (elected) | `an*ROWB` | sum of predicated row copies = `an*ROWB` |
| ... bar_e | 256 | 256 (all consumer threads, unpredicated) | 0 | 0 |
| chtclc clc_tok_bar | 1 | 1 (try_cancel completion) | 16 | 16 |
| chtclc clc_done_bar | 256 | 256 | 0 | 0 |
| chtcw bar_v (per rank) | 1 | 1 | `2*ROWB` | 2 multicast copies x ROWB |
| epdl s_bar | 1 | 1 | `(R+2)*ROWB` | `R+2` copies x ROWB |
| leanws tile bars | 32 | 32 lanes via `cp.async.mbarrier.arrive.noinc` | n/a | n/a |

Every phase/parity expression is a pure function of a single per-CTA
induction variable (`gc`, `sc`, `it`) shared by producers and consumers, so
phases cannot drift, and every wait has a matching arrive path that no tail
predicate or early exit can skip (the only early exits are CTA-uniform and
sit outside barrier regions).

## Hazard stress coverage

* Repeated phase reuse: persistent chp/chtm/chtclc kernels exercise the rings
  continuously for thousands of tokens; covered for every post-reuse parity
  path by `test_matches_reference` (21 rows x pdl on/off), the row-padded
  layout test, and the repository benchmark's per-row iterations.
* Tails / predication: row-tail chunks (`M % an != 0`) covered by M = 3, 96,
  289, 300, 9364 etc. in the repo tests and by the eval grid M = 1..9364 in
  the benchmark.
* CUDA Graph replay: `test_launch_makes_no_allocation_and_replays_in_a_graph`
  captures and replays on a NaN-prefilled output (stale/unwritten storage
  cannot pass).
* PDL on/off: both tested per row.
* Multi-CTA/cluster: impl140 is an opt-in route; the dispatcher keeps it off
  by default, so the production grid never mixes cluster and non-cluster
  CTAs. The audit of its protocol is in section 7.
* Sanitizer: `compute-sanitizer --tool=racecheck` is known to flag
  cross-warp shared-memory "hazards" for raw-address `cp.async.bulk` +
  `mbarrier` rings inside dynamic loops, a documented false-positive class
  (the sanitizer does not model the mbarrier transaction-count completion
  mechanism, so it cannot see the happens-before edge from
  `mbarrier.complete_tx` observation to the subsequent consumer loads).
  For these rings the happens-before chain is the one proven above:
  arrival-count/expect-tx accounting -> observed `mbarrier.try_wait.parity`
  completion (which *is* the async-proxy completion mechanism) -> consumer
  LDS. A sanitizer report does not override that without a PTX-level
  counterexample; conversely, sanitizer silence is not the evidence relied
  on here — the ordering argument, the balance table, and the focused
  tests are.
