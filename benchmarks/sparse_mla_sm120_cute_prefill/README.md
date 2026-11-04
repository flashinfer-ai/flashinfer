# CuTe C++ sparse prefill experiment

Explicit prototype for V41 inline FP8 (512 E4M3 bytes + 16 UE8M0 scale bytes per
KV row). It does not participate in public dispatch. CuTeDSL and decode are
unchanged. Each implementation has its own header and launch translation unit;
there are no constexpr version branches. `CUTE_PREFILL_VERSION` selects the file:

| Version | Change |
|---|---|
| `v1_whole` | Two WG, full KV stages |
| `v2_micro` | Three WG, independent K/V micro-pipelines |
| `v3_vector` | K LDSM.x2, packed scales, vector score merge, async zero-fill |
| `v4_sf_tile` | Loader transposes scales into 4-candidate × 4-group tiles |
| `v5_ldsm4` | K LDSM.x4 reads two adjacent candidate fragments |
| `v6_sf_cache` (default) | Cache K scales across both micro-steps; V scales in pairs |
| `v7_ring3` | Independent 3-slot K and V rings spanning candidate tiles |
| `v8_ring2x5` | Alternative 2-slot K / 5-slot V split at the same SMEM budget |
| `v9_ring2` | Two-slot control using the same mbarriers as v7 |
| `v10_pipeline3` | Initial CUTLASS completion-driven pipeline; unresolved racecheck reports |
| `v11_pipeline_copy` | Rolled unified copy loop; slower |
| `v12_pipeline_rolled` | Rolled CUTLASS pipeline; numerical tests pass, full racecheck unresolved |
| `v13_pipeline_commit` | Copy-group-boundary diagnostic |
| `v14_pipeline_noinc` | Fixed-count deferred-arrival diagnostic |

NCU optimization versions continue as separate files:

| Version | Change |
|---|---|
| `v15_packed_p` | Four P FP8 values packed per register |
| `v16_score_swizzle` | First score-buffer swizzle |
| `v17_packed_score` | Packed P plus score swizzle |
| `v18_cached_indices` | Cache row IDs per loader warp across feature slices |
| `v19_role_pipeline` | Rejected lifetime experiment: more spills and changed rounding |
| `v20_scale_bits` | Direct exponent-bit extraction; no measurable gain |
| `v21_score_banks` | Row permutation plus swizzle for QK/PV vector accesses |
| `v22_vector_q` | Vector Q loads, register RoPE partners, packed Q conversion |

Headers are under `include/flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/`.
`common.cuh` contains the shared argument ABI and small arithmetic helpers only.

## Three-WG schedule

Each CTA handles one query token, 32 heads, and loops over 64-candidate tiles.
Warp IDs below are local to each 128-thread WG.

- **WG0, QK:** warps 0/1 own features 0:256; warps 2/3 own 256:512.
  Even/odd warps own heads 0:16/16:32. Q RoPE and group-32 quantization run once;
  Q persists in registers. Each feature half consumes two 128-feature K stages.
  The lower feature half stores partial scores, and the upper half adds its
  partial scores before publishing the completed S tile.
- **WG1, PV:** warp `w` owns eight heads `[8*w, 8*(w+1))`. Scores load directly
  into MMA B ownership, making softmax warp-local (four lanes/head). P stays in
  registers. For each of four 128-feature V stages, process four 32-feature
  scale groups: multiply P by candidate-specific V scales, requantize P, then
  compute `O^T = V^T @ P^T`. The full output accumulator remains in registers.
- **WG2, loaders:** warp 0 loads the first K feature half; warp 1 loads the second.
  Warps 2/3 jointly load V. K and V have independent ready/free handshakes and
  advance independently, without a full-loader-WG barrier. Each uses `cp.async`
  for vector gathers, waits for completion, and publishes its stage.

PV normalizes with the optional sink, rounds context to BF16, applies inverse
RoPE, and casts/stores FP8 output and group-32 UE8M0 scales in the same launch.
The raw byte-transpose LDSM copy atom maps V into MMA A without the stock SM100
copy atom's STSM-oriented register repacking.

## CuTe layouts and buffering

Logical K: `(64, ((128,2),2))` = candidate × ((micro-feature, micro-step), feature-half).
Logical V: `(64, (128,4))` = candidate × (micro-feature, feature-tile).

Each physical feature stage uses:

```cpp
composition(Swizzle<3,4,3>{}, Layout<Shape<_64,_128>, Stride<_128,_1>>{})
```

The swizzle distributes 16-byte vectors across shared-memory banks without
padding the 128-byte rows. PV obtains transposed subtiles with `local_tile`;
loaders and consumers use the same swizzle. A plain 128-byte pitch was correct
but materially slower in the canonical test.

| Buffer | Bytes |
|---|---:|
| K: 2 stages × 2 feature halves × 64 × 128 | 32,768 |
| V: 2 stages × 64 × 128 | 16,384 |
| K scales: 2 candidate tiles × 64 × 16 | 2,048 |
| V scales: 2 candidate tiles × 64 × 16 | 2,048 |
| S: 2 candidate tiles × 32 × 64 × FP32 | 16,384 |
| **Total** | **69,632 (68 KiB)** |

K slots release after QK consumes their micro-step, S slots release after PV
loads scores, and V slots release after PV consumes their feature tile.
Scale slots alternate by candidate tile rather than feature micro-step. The
bounded two-feature-stage pipeline prevents reuse of a scale slot until its
previous consumer is finished; SF completion is published with the first
feature stage. No global temporary is introduced.

K and V address the same original MLA cache vector, but this design loads the
payload independently for each role. The two-WG variant instead shares a full
KV buffer. Extra cache traffic is the tradeoff for independent progress and
earlier buffer release.

Named barriers: 1=QK-local, 2/3=S-ready, 4/5=S-free, 6/7=K-ready, 8/9=K-free,
10/11=V-ready, 12/13=V-free. K/V handoffs have 64 producer + 128 consumer threads.
Register-allocation instructions stay inside disjoint role branches: loader=40,
QK=208, PV=256 registers/thread in v6. Ptxas reports an initial 168 registers/thread
for the 384-thread kernel, with an 8-byte stack frame, 8 bytes of spill stores
and 16 bytes of spill loads. The v2 baseline has zero spills. Moving allocation hints before
the role branches caused heavy compiler spilling and was rejected.

## Build, check, benchmark

Run from the FlashInfer repository with development dependencies installed:

```bash
export CUTE_PREFILL_ARTIFACT_DIR=/tmp/flashinfer-cute-prefill
CUTE_PREFILL_VERSION=v6_sf_cache python benchmarks/sparse_mla_sm120_cute_prefill/build.py
pytest -q tests/attention/test_sparse_mla_sm120_cute_cpp_prefill.py
MODE=bench python -m benchmarks.sparse_mla_sm120_cute_prefill.evaluate
```

The build targets SM120a and includes line information and ptxas diagnostics.
Set `CUTE_PREFILL_REFERENCE_MODULE=/path/to/two-wg/kernel.so` for bitwise
comparisons in the regression tests. Set `CUTE_PREFILL_BASELINE_MODULE` to an
archived original FlashInfer module for the unfused timing baseline; its
execution-plan ABI must match. Without it, the unfused baseline uses the current
native implementation. The `ldsm_mega` timing label assumes the current native
implementation is the LDSM mega baseline.

Canonical benchmark: B=1, local Q=4096, CP=4 rank=0, global primary KV=16384,
compressed KV=8192, H=64, D=512, no prefix, SWA<=128 + compressed top-k<=512.
It alternates three-launch CUDA graphs for 12 samples/implementation. Numerical
checks sample 16 query rows against independently decoded cache data in FP64.
Thresholds are unchanged: relative L2 < 0.07, max absolute LSE error < 0.06.

Tests cover B=2, CP ranks 0/3, H=8/16/32/64, lengths around 32/64, empty rows,
index holes and entirely invalid lists, mutable graph inputs, repeatability,
missing sinks, and +/-infinite sinks.

## Scope

Experimental loader, not a validated public API. Assumes contiguous Q/indices,
two inline FP8 caches with 528-byte rows, and valid indices/positions within
allocations. Implements RoPE but no additional Q normalization. Mixed FP4 and
V4 hybrid caches are outside this prototype's scope.

QK uses a CuTe `m16n8k32` FP8 atom with explicit FP32 scale application, not native
block-scaled QK MMA. The micro-pipeline preserves the two-WG prototype's arithmetic
and is checked bitwise against it. Both differ in reduction order and 64-entry P
quantization from the original 32-entry native schedule, so comparison against
that original uses the unchanged independent numerical gates.

## Earlier baseline result

On RTX PRO 6000 Blackwell Server Edition, the canonical interleaved medians were:

| Implementation | ms |
|---|---:|
| Three-WG micro-pipeline | 2.513 |
| Two-WG CuTe | 2.787 |
| Native LDSM mega | 4.223 |
| Archived original unfused pipeline | 4.780 |

The micro-pipeline is 9.8% lower latency than two WG. All nine tests and all three
Compute Sanitizer tools pass, with bitwise agreement to the two-WG kernel. NCU
confirms zero spill requests and 69,632 bytes dynamic shared memory.

## Vectorized data movement

K uses CuTe `SM75_U32x4_LDSM_N` with a `(16,16,32)` copy tile to load two
adjacent N=8 MMA fragments at once. QK caches eight scale bytes per candidate
using aligned 64-bit reads and reuses them across both 128-feature micro-steps.
The partial-score merge uses aligned vector copies.

The V loader transposes each four-candidate × four-group byte tile in registers.
Its physical SF layout is `(group_quad, candidate_quad, group_in_quad,
candidate_in_quad)`, with the last mode contiguous. PV reads two adjacent groups
as a 64-bit word, then extracts candidate scale bytes from registers. This
avoids LDS.U8 without retaining all four groups live at once. The loader still
uses 32-bit STS for transposed scales; widening these would require more exchange
between lanes. K/V payload gathers use 16-byte cp.async with zero-fill for invalid
candidates instead of explicit shared-memory zero stores.

NCU issued-instruction totals include predicated-off `@!PT LDS RZ, [RZ]` markers
around cp.async. These must be separated from actual shared-memory reads when
interpreting scalar LDS counts. Experiment records and source-correlated reports
are in workspace `results/direct/prefill_cute_vector/`.

Latest v6 canonical median: **2.398 ms**, versus v2 **2.513 ms**, native LDSM mega
**4.222 ms**, and archived original pipeline **4.782 ms** in one interleaved run.
Nine tests and all three sanitizers pass. Actual scalar LDS and LDS.U8 reads are
eliminated; scalar STS decreases 87.5%. NCU shows 98,304 local spill requests.
See workspace `results/direct/prefill_cute_vector/README.md` for counts, timings,
validation details and source snapshots.

## Independent physical rings

The feature iterator is not the shared-memory stage coordinate. In v7:

```text
logical K = (candidate=64, ((micro_feature=128, feature_iter=2), feature_half=2))
physical K = (candidate=64, micro_feature=128, feature_half=2, ring_slot=3)
logical V = (candidate=64, (micro_feature=128, feature_iter=4))
physical V = (candidate=64, micro_feature=128, ring_slot=3)

K iteration = candidate_tile * 2 + feature_iter
V iteration = candidate_tile * 4 + feature_iter
ring_slot = iteration % 3
ready_phase = (iteration // 3) % 2
free_phase = (iteration // 3 - 1) % 2  # producer waits only after filling the ring
```

Each slot is an independent 128-feature slice; a ring may contain slices from
adjacent candidate tiles. Producers still complete each cp.async batch before
publishing ready. Consumers release the slot after its last MMA use. K and V
progress independently; the logical iterator still determines cache addresses.

v7 uses K=48KiB, V=24KiB, S=16KiB, SF=4KiB, plus 128 bytes including barrier
storage/alignment: 94,336 bytes. Another 8KiB payload slot exceeds the SM120
101,376-byte limit. v8 allocates the same capacity as K=32KiB and V=40KiB.
The deeper rings use per-slot mbarriers, avoiding the named-barrier ID limit.
The two SF slots remain safe: before writing tile t+2's SF, the producer must
observe release of tile t's last feature slice. This bound must be revisited
before increasing capacities beyond these versions.

Measured ring variants: v6=2.405ms, v7=2.346ms, v8=2.423ms, v9=2.352ms.
The extra depth has little isolated benefit over the same-barrier control;
the loader still waits for copy completion before publishing each slot.
v7 passes all nine numerical/repeatability tests and all three sanitizers.
It changes a few FP8 output values due to compiler FMA contraction order, with
identical scales/LSE and unchanged numerical gates. Detailed evidence is in
workspace `results/direct/prefill_cute_ring/README.md`. v6 remains the default.

## CUTLASS completion-driven pipeline

`v10_pipeline3` retains v7's three K / three V slots and 94,336-byte allocation.
Both streams use `cutlass::PipelineAsync<3>`, with 64 producer arrivals and
128 consumer arrivals. Each role maintains its own `PipelineState`; producer
states start at the empty phase using `make_producer_start_state`.

```python
# Producer: loader warps, independently for K and V
for micro_tile in stream:
    pipeline.producer_acquire(write_state)       # wait for FREE
    prepare_scales_if_first_feature_slice()
    gather_payload_with_cp_async(write_state.index(), micro_tile)
    pipeline.producer_commit(write_state, cpasync_barrier_arrive)
    advance(write_state)
pipeline.producer_tail(write_state)               # wait for all slots FREE

# Consumer: QK or PV
for micro_tile in stream:
    pipeline.consumer_wait(read_state)            # wait for READY
    compute_from_shared_slot(read_state.index())
    pipeline.consumer_release(read_state)         # make FREE
    advance(read_state)
```

In the source, scales are prepared before the payload copies. K scales use
cp.async; V scales are transposed and stored by the loader. The commit callback
registers deferred hardware arrivals for preceding cp.async operations, then
CUTLASS performs the normal release arrival for producer participation and
ordinary shared stores. There is no producer wait for copy completion and no
cp.async wait_group/commit_group sequence. This is LDGSTS arrival tracking,
not TMA's expected-byte-count protocol. The empty-slot bound limits lookahead,
protecting both payload and the independently double-buffered scale storage.

The CUTLASS path is experimental and has **not** replaced the validated v6
default. v12 measures about2.280ms versus v7 2.330ms and v6 2.399ms; nine tests
pass bitwise against v7, and memcheck/synccheck pass. Full racecheck still reports
async-copy hazards despite clean focused cases. Neither explicit copy-group
boundaries nor fixed noinc arrival counts clears the affected-CTA report.
Do not interpret the focused probe as full racecheck validation. Reproducers,
traces and investigation records are under workspace
`results/direct/prefill_cute_pipeline/`. Performance numbers are provisional
until the remaining racecheck reports are resolved.

## NCU-guided vectorization

v22 builds on v18's cached sparse indices and packed P. Each loader warp caches
64 candidate IDs as two registers/lane, then selects each vector's row ID with
a warp shuffle. The same IDs serve every feature micro-slice and SF gather.
Q loads four contiguous BF16 features at once with an aligned64-bit load; RoPE
partners are already in those registers. Both Q and P convert four floats to
one E4M3 register using paired native conversions.

The final score layout has a hierarchical row mode:

```cpp
composition(Swizzle<2,3,3>{}, make_layout(
    Layout<Shape<_2,_2,_4>, Stride<_128,_64,_256>>{}, Layout<_64>{}))
```

That is the16-head QK slice. PV uses `_2` instead of `_4` in the last row mode
for8 heads. The row layout swaps the low two logical row bits; the swizzle then
XORs logical row bit0 into column bit4 and row bit1 into column bit3. This is
chosen for QK's2-float vectors and PV's4-float vectors. Buffer sizes, pipeline
phases, loop bounds and invalid-index handling remain unchanged.

Measured records and source-correlated profiles are under workspace
`results/direct/prefill_cute_ncu_opt/`. The CUTLASS path's unresolved racecheck
reports remain an acceptance issue; these experiments do not replace v6 by default.

Latest NCU-guided result: v22 **1.831 ms** versus v12 **2.282 ms** in an
interleaved canonical comparison (19.8% lower latency). NCU confirms zero spills,
90% fewer excessive shared wavefronts, and removal of the scalar-Q load
inefficiency. Nine bitwise tests, memcheck and synccheck pass. The existing
CUTLASS async-copy racecheck issue remains, including in v22, so the default
remains v6. See workspace `results/direct/prefill_cute_ncu_opt/README.md` for
per-change measurements, rejected experiments, final profile and validation.
