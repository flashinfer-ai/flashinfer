.. _apicake_sampling:

flashinfer.cake_sampling
========================

``cake_sampling`` is a thread-block-cluster implementation of top-k-then-top-p sampling from
probabilities for Hopper and newer GPUs (compute capability 9.x, 10.x, 11.x and 12.x).  It is
measured on H100 (9.0), B200 (10.0), B300 / GB300 (10.3) and Rubin R200 (10.7); 11.x and 12.x are
compile targets that have not been run on hardware.  It fuses the three stages of
:func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` into at most two kernels per call:

1. a thread-block-cluster radix select that writes the exact per-row top-k slab: one 11-bit
   cluster pass reduces the 2048-bucket histograms through distributed shared memory, the
   entries of the selected bucket (at most 2048 cluster-wide) are gathered into every CTA as
   64-bit ``(key, ~index)`` composites and the remaining key bits and boundary ties are resolved
   with local passes (rows whose bucket overflows that capacity take the exact three-pass
   cluster path); and
2. a programmatic-dependent-launch sparse top-p kernel that sorts the slab prefix with one
   composite-key bitonic network (registers, warp shuffles and a cross-warp shared-memory
   exchange), keeps the shortest prefix whose exclusive mass is below ``top_p`` times the top-k
   mass, renormalizes, and draws one token per row by inverse CDF from
   ``curand_init(seed, row, offset)``.  For ``top_k_max <= 64`` (``fused_tail_kcap`` in the
   manifest) this stage runs inside the stage-1 kernel on one warp of the cluster's first CTA,
   so the call is a single launch; the outputs are bitwise identical to the two-launch form.
   Every variant the dispatcher can pick for such a top-k carries the tail (manifest
   ``fused_tail``); the ``(8, 48)`` resident, a large-k pick, is built without it.  For
   ``64 < top_k_max <= 1024`` (``fused_block_tail_kcap``) the same stage can run on the whole first
   CTA of the cluster (512 threads, two slab entries each; launch flag bit 3) on a separate build of
   the variant (manifest entries with ``fused_block_tail``, symbol suffix ``_bt``).  Round 7 ships
   that build for every variant built with the one-warp tail (19 twins): the round-5 twin had run
   the tail twice through a duplicate block in the stream template (removed in round 7).  The host
   selects it only for a cluster-8 streaming variant with ``top_k_max > 768`` on a one-wave grid, on
   B200 and GB300 (compute capabilities 10.0 / 10.3).  Measured under CUDA-graph replay (the eager
   span of the two-launch chain includes the host gap between its launches, which made the twin
   look 10-50 % faster everywhere): the in-CTA tail loses to the chain on every resident (B200
   1.20-1.54x, GB300 1.40-1.84x of the chain's time) and on every stream at k <= 750 (1.0-1.26x at
   k = 200, 1.04-1.15x at k = 500, 0.94-1.05x at k = 750); the cluster-8 streams at k = 1000 win on
   both (B200 0.89-0.94, GB300 0.89-0.97).  H100 and R200 keep the chain (no graph-replay A/B
   recorded for them in round 7).
   Compiling the tail into the default build had cost every ``top_k <= 64`` launch 4-19 % on H100 /
   GB300 / R200, so it stays a separate build.  The whole-CTA form is bitwise identical to the
   two-launch form; rows of such a launch whose k is at most 64 take the one-warp tail.  A
   multi-wave grid keeps the two launches (its first CTA would run one tail per wave).  In the two-launch form the dispatcher also decides where the dependent kernel lands: stage 1 signals
   ``griddepcontrol.launch_dependents`` before its first pass only when the batch fits on the
   SMs its last wave leaves free (one stage-1 CTA per SM), so the stage-2/3 CTAs are never
   packed onto the few SMs free mid-flight; larger batches let the dependent launch as stage 1
   exits.  A streaming variant signals at that early point only on Blackwell and Rubin
   (compute capability 10.x); on Hopper it signals after its filter pass, once the whole row
   has been read, where the round-3 kernels did.  All three decisions travel in the stage-1
   ``launch_flags`` argument (bit 0 one-warp tail, bit 1 early trigger, bit 2 stream pre-pass
   point, bit 3 whole-CTA tail, bit 4 coarse sample, bit 5 row-span filter arm, bit 6 speculative
   sample) and none changes any output.  Bit 4 selects a
   streaming variant's coarse-sample build (manifest entries with ``coarse_sample``, symbol suffix
   ``_cs``): its sampled first pass reads one 64-byte block per 512 bytes of the row (1/8) instead of
   one per 256 (1/4), with the lower-bucket margin, the sampled-mass cap and the filter-arm density
   switch scaled to the rate, so only the sample's DRAM traffic changes and every output is
   bit-identical to the default build.  The host sets it for launches whose largest top-k is at most
   ``fused_tail_kcap`` (round 6, lever S4: the V = 262144 streams at B >= 64 run 7-9 % faster on
   B200 / GB300 / R200); a k ~ 1000 row's candidate list sits at the gather capacity, where the
   coarser estimate sends 3-12 % of the rows down the slower path, so larger top-k launches keep the
   default build, which is byte-identical to round 5.  Bit 5 makes a streaming variant's filter pass
   choose its float-threshold arm from the expected candidate density of the whole row (the
   cluster-wide sampled mass against the cluster's span, seven 16-entry groups per candidate)
   instead of one CTA's span, which the round-5 test compared the cluster-wide mass with (a
   cluster-8 row at k ~ 1000 never took the arm); both arms build identical candidate segments.  The
   host sets it for cluster >= 8 streams whose largest top-k exceeds ``fused_tail_kcap`` on Hopper,
   B200 and GB300 (round 6, lever FD5: the V = 262144 cluster-8 k = 1000 cells run 2-5 % faster);
   a cluster-1 row that takes the arm can lose 5 % and Rubin measures neutral, so nothing else.  Bit 6 selects a streaming variant's speculative-sample build (manifest entries with
   ``spec_sample``, symbol suffix ``_sp``): there is no separate sampled read; every warp loads its
   first register chunk (and the second where the row has one), histograms a strided subset of those
   registers as the sample (1/8 of the row for a largest top-k at most ``fused_tail_kcap``, 1/2
   above), keeps the chunk for the filter pass, which continues from the next chunk, and scales the
   lower-bucket margin and the sampled-mass cap to the realised rate, so again every output is
   bit-identical to the default build.  The host sets it for every ept-16 stream and for ept-32
   streams whose rows are one register chunk per CTA, two chunks on a cluster of 8, or at least 16
   chunks (round 7, lever SP: ept-16 streams 2-13 % faster at 1-16 chunks, the cluster-8 ept-32
   stream 13-18 % at one chunk and 0-6 % at two, the cluster-1 ept-32 stream at 16 chunks 2-15 %);
   ept-32 streams at 2-10 chunks per CTA on clusters 1-4 measured 0-6 % slower with it and keep
   their round-6 build, on B200 so does a two-chunk row whose second chunk is at least half full (the
   cluster-8 stream at V = 262144: 0.2-2.4 % slower with it at k <= 64, while V = 151936 keeps the twin
   and GB300 keeps it on both), and a cluster-8 stream above ``fused_tail_kcap`` (the two-launch chain, 1/2-rate
   sample) keeps it on every architecture (GB300 V = 262144 measured 11-17 % slower eager and 3-7 % slower
   under graph replay with the speculative build).  That is the B200 / GB300 rule; on Hopper and Rubin the same chunk rule
   applies only on a cluster of at most 4 CTAs whose grid has at least 64 CTAs, for a largest top-k
   at most ``fused_tail_kcap`` only to rows of at least 5 chunks per CTA, and on the two-launch
   chain with at least 128 CTAs for rows of 16 or more chunks: the round-7 H100 / R200 matrices
   measure the cluster-8 streams 1-8 % slower with the build, the 4-chunk rows at k <= 64 1-2 %
   slower than their coarse-sample build, and those wide streams 2-20 % faster.  Bits 4 and 6 are
   exclusive.

The stage-2/3 kernel exists in three static forms that differ only in instruction selection,
never in output: the base form (f64 top-p cut / sample tests, max-min bitonic exchange), a form
whose bitonic exchange is one 64-bit compare and one select (``one_cmp_select``, dispatched on
compute capability 9.0 and 10.0: 16-19 % of the k = 1000 kernel), and a form that adds integer
top-p cut / sample tests derived from one per-row threshold (``int_tests``, dispatched with the
former on 10.3, where the FP64 conversions are slow: together 0.78-0.84 of the round-4 kernel).
Every other device, including Rubin 10.7 whose toolchain already emits the short exchange, runs
the base form.  The manifest lists each slab once per form with its ``features`` and
``variant_flags``; :func:`stage23_variant_flags` returns the device's form and the binding selects
the kernel by ``(threads, items, variant_flags)``.

Semantics (support, tie-breaking toward the lower vocabulary index, Philox stream advancement
through the generator) follow the ``top_k_first`` route with two extra guarantees:

* **Strict determinism.** The support is exactly the first ``k`` entries of
  ``lexsort(-prob, index)``; every top-p and sampling decision is made on exact 64-bit
  fixed-point prefix sums of the sorted slab, and no atomic decides an output.  Identical inputs
  give bitwise identical samples, ``renorm_out`` and slab for every kernel variant, stream
  launch or CUDA-graph replay.  ``renorm_out`` and the workspace slab are emitted in sorted
  order (descending probability, ascending index).
* **NaN / Inf.** NaN, negative and ``-0.0`` probabilities are treated as ``+0`` and never
  sampled.  A row containing ``+inf`` keeps exactly its ``+inf`` entries, samples uniformly among
  them and renormalizes them to ``1/m``.  A row whose top-k mass is zero returns its smallest
  slab index with an all-zero ``renorm_out``.  Every output is finite.  Per-row ``top_p`` is
  clamped to ``(0, 1]`` and per-row ``top_k`` to ``[1, min(vocab, 1024)]``.

Requests the frozen kernels cannot serve are dispatched to the ``top_k_first`` route
(``deterministic=True``): top-k disabled or ``k >= vocab``, ``k > 1024``, non-``float32`` or
non-contiguous rows, or a device outside the build targets.  The build targets are FlashInfer's
``CompilationContext`` targets (``FLASHINFER_CUDA_ARCH_LIST`` when set, as in AOT builds on
hosts without a GPU; otherwise the capabilities of the visible devices) restricted to the
supported majors 9-12; a device whose architecture is not among them takes the ``top_k_first``
route instead of failing at launch.  Large
``batch * vocab`` launches run on the streaming stage-1 variants (a cluster of 1-8 CTAs walks the
row in register chunks, samples one 11-bit pass to bound the candidate range, filters the row into
a per-CTA candidate list and finishes on a gathered copy of that list with local passes; the
exact fallback for a list that overflowed or came up short streams the row again through a compact
runtime radix loop, kept small so the kernel's code stays resident in the SM instruction cache next
to the stage-2/3 kernel), so there is no size-based fallback.  :func:`cake_sampling_route` reports the decision without launching.

The checked-in source product lives in ``csrc/cake_sampling/generated/`` as one translation
unit (``cake_sampling_kernels.cu``, which includes the ``cake_sampling_kernels_part<N>.cuh``
body files in order; every file stays under the repository's 5 MiB limit) plus a manifest that
records every frozen variant's launch resources and every source file's size and SHA-256;
FlashInfer verifies each file and the include list and compiles the unit once into a single
fatbin with one ``-gencode`` per build target
(the kernels use only clusters, distributed shared memory, programmatic dependent launch and
``redux.sync``, so no per-architecture source exists).  Stage-1 variants whose dynamic shared
memory exceeds the device's opt-in limit are not dispatch candidates: on 12.x (99 KB) the
streaming variants drop out, so vocabularies above the register-resident capacity (196608)
take the ``top_k_first`` route there.

The stage-1 variant is chosen per call by a cost model whose single-wave CTA capacity table and
cost constants are keyed by the device's SM count (148 for B200 / B300, 132 for H100, 212 for
Rubin R200; other devices use the nearest measured table).  The streaming variants exist with 16
and, since round 5, 32 entries per thread (a 512 x 32-entry chunk halves the chunk count and keeps
twice the loads in flight; on B200 / B300 / R200 the ept-32 streams take every ``B >= 16`` row and
the small-batch rows of ``V >= 128K`` at ``k <= 64``, 4-14 % faster than the ept-16 streams, while
H100 keeps the ept-16 streams where those are faster).  The constants -- resident base / per-entry,
streaming wave base, per-chunk cost of each chunk width, a fixed per-cluster term (cluster barrier
and DSM exchange latency), a per-CTA cluster term, a launch-size term, a large-k term and an
H100-only fixed term for the ept-32 form -- were fitted per table on per-variant sweeps of all four
architectures (25 ``(V, B)`` cells x ``k = 50`` / ``1000``, every frozen variant) so that every
measured cell picks a variant within 3 % of its fastest frozen variant (148: worst 1.7 %, 212:
2.5 %, 132: 3.1 % on one cell).  One more term is measured on the chain rather than on the
kernel alone: an ept-32 stream on a row of at most three 512 x 32 chunks (``V < 65536``) that is
followed by the stage-2/3 launch (``k > 64``) runs 3-5 us longer than the same kernel launched
last, on all four GPUs, so the two-launch path charges that per-launch cost to those variants and
keeps the ept-16 stream there.

A launch whose largest top-k is above 64 has two regimes and the same table ranks the variants in
both: the *fused* regime (B200 / GB300 in round 7: a one-wave cluster-8 stream above k = 768 finishes
in one kernel with the whole-CTA tail, every other candidate pays the stage-2/3 kernel) and the
*two-launch* regime (H100 / R200, or ``two_launch=True``).  ``choose_stage1(..., two_launch=)`` selects it (None
= the device's policy).  The k > 64 constants were re-fitted in round 5 on same-run per-variant
sweeps of the chain and of the fused form (32 ``(V, B)`` cells per architecture, k = 1000): the
per-bucket list path of the streams costs per chunk rather than per wave (``stream_large_k_chunk_us``
/ ``stream_large_k_chunk32_us``), a streaming variant's in-kernel tail costs more than a resident's
(``fused_tail_stream_us``), the list path's cluster pass costs per CTA (``stream_large_k_cluster_cta_us``)
and the ept-32 form is not amortised below four 512 x 32 chunks per CTA
(``stream_wide_short_row_large_k_us``); none of them applies below k = 64, so the k <= 64 picks are
unchanged.  Regret against the measured-best frozen variant of every k = 1000 sweep cell: 148 fused regime
2.8 % worst (no cell over 3 %), two-launch regime (GB300, CUDA-graph rows) 4.5 % on one cell; 132 fused
8.0 % on two cells (``V = 32768 B = 64``: the (2,32) resident beats the (1,16) stream pick by 1.6 µs;
``V = 262144 B = 32``: the (2,32) stream beats the (2,16) pick by 1.5 µs), two-launch 3.0 % on one; 212
fused 5.4 % on five cells (``V = 262144 B <= 32``: the (4,32) stream beats the (8,16) / (4,16) picks by
0.6-1.2 µs), two-launch 6.8 % on one.  The residual cells are a batch- and vocabulary-dependent cost of the
ept-32 form that no chunk- or cluster-linear term of this model expresses without moving cells that are
right today (design doc, round 5, "Dispatch regimes"); they remain 1.6-3.6x faster than ``top_k_first``.

Round 6 re-fitted the k <= 64 stream constants of the 148 and 212 tables on policy-aware sweeps (every frozen
variant timed as the host launches it: the coarse-sample twin at k <= 64, launch flag bit 5 on the cluster >= 8
streams above it; B200, GB300 and R200, k = 50, 25 cells per table).  The grid search was constrained so that no
k > 64 pick changes (those stay chain-fitted) and no cell's pick gets slower: 148 ``stream_chunk_us`` 0.2 -> 0.25
with ``stream_large_k_chunk_us`` 0.4 -> 0.35 (the large-k per-chunk sum is unchanged) moves V = 151936 B <= 8 from
the (8,16) to the (8,32) stream (2.5-4.3 % faster), worst regret 4.5 -> 2.8 %; 212 ``stream_chunk_us`` 0.15,
``stream_chunk32_us`` 0.15, ``stream_cluster_us`` 0.25, ``stream_cluster_cta_us`` 0.05,
``stream_large_k_chunk32_us`` 0.05 move V = 128256 / 151936 B <= 16 to the (4,32) stream (2.3-3.7 % faster), worst
4.1 -> 1.7 %; 132 is unchanged.  The kernels and the frozen bundle are untouched by this change.

Correctness contract
--------------------

Since round 7 the route is held to a tolerance-level contract instead of bit-identity with the previous bundle
(the kernels may change bits; the precision level and the selection exactness may not):

* **Top-k set**: equal to the float64 sorted reference for every row and shape, including the tie rule (ties on the
  key are broken by the lower index: the 64-bit ``(key, ~index)`` composite is a total order, so every exact
  selection yields the same set and the same slab order).
* **Top-p cut**: equal to the float64 reference, except when the exact cumulative mass lies within ``eps = 1e-6``
  of ``top_p`` times the top-k mass, where the cut may differ by one element (``tests/utils/test_cake_sampling.py``
  places rows on that boundary and checks both sides).
* **Sample**: always inside the exact support (the kept prefix); run-to-run deterministic for identical inputs,
  ``philox_seed`` and ``philox_offset`` (CUDA-graph replay, concurrent streams and repeated invocation included);
  multi-seed next-token histograms within the 99 % binomial band of the exact distribution per row class; sglang
  GSM8K accuracy within ``top_k_first``'s seed spread with zero out-of-support tokens.
* **Renormalized slab** (``renorm_out``): fp32 with ``rtol 1e-6``, ``atol 1e-7`` against the float64 reference.
  Keys stay exact fp32 bit patterns; every accumulation is at least fp32 (no bf16 / fp16 anywhere); no tolerance is
  loosened to admit a kernel change.
* A stage-1 build that does not intend to change numerics (every round-7 twin: the speculative-sample ``_sp``, the
  coarse-sample ``_cs`` and the whole-CTA-tail ``_bt`` builds) is additionally gated on bit-identity with the default
  build on every tested row.  A change that moves bits must document exactly which rows can differ (ties, the eps
  boundary) and why; no round-7 kernel does.
* The exact fallbacks stay kernel-side: a candidate list that overflows the gather capacity takes the three-pass
  cluster path, a row with fewer than k candidates is kept whole, and vocabularies above 2^21 (and compute
  capabilities 12.x) take the reference ``top_k_first`` route.  The sm_120 / sm_121 route semantics and the host API
  are unchanged.

Measured performance
--------------------

``python benchmarks/bench_cake_sampling.py --cupti --skip-joint --batches 1,2,4,8,16,32,64,128
--vocabs 32768,128256,151936,262144`` with ``--top-k 10`` / ``50`` / ``1000`` and with ``--cuda-graph``
(``flashinfer.testing.bench_gpu_time``, CUPTI kernel time, p = 0.9, median µs), round-6 bundle
(the round-5 kernels plus the coarse-sample stage-1 twins for k <= 64, the row-span filter arm on the
cluster-8 two-launch streams and the re-fitted small-k dispatch) on the four supported architectures.
The summary compares every cell with FlashInfer's default ``top_k_first`` route and with the previous
frozen bundle (#5684, the round-5 bundle) measured in the same run; the per-cell tables below list the
round-6 medians.

.. list-table:: Round-6 bundle, 192 cells per architecture (8 batches × 4 vocabularies × 3 k × eager/graph)
   :header-rows: 1
   :widths: 18 14 20 24 24

   * - GPU
     - cells slower than the round-5 bundle (#5684) by > 2 %
     - speedup vs top_k_first (min / median / max)
     - gain vs #5684, B ≤ 16 (min / median / max)
     - gain vs #5684, B ≥ 32 (min / median)
   * - H100 SXM (sm_90a, 132 SMs)
     - 0 of 192
     - 1.67x / 3.47x / 8.88x
     - 0.98x / 1.00x / 1.09x
     - 1.00x / 1.04x
   * - B200 (sm_100a, 148 SMs)
     - 1 of 192
     - 1.63x / 3.97x / 8.62x
     - 0.97x / 1.01x / 1.05x
     - 0.99x / 1.01x
   * - GB300 (sm_103a, 152 SMs)
     - 4 of 192
     - 1.67x / 4.17x / 14.56x
     - 0.95x / 1.01x / 1.07x
     - 0.99x / 1.02x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 4 of 192
     - 1.58x / 3.69x / 7.26x
     - 0.97x / 1.01x / 1.05x
     - 0.98x / 1.02x

The "slower by > 2 %" column counts single forward-order matrix rows; every such cell (all k = 1000, most of them
CUDA-graph rows, where untouched cells carry a ±1.5 % run-to-run band) was re-measured in the reversed tree order and
in independent processes.  One cell stays behind: GB300 V = 128256, B = 64, k = 1000, eager, +0.9 % (28.8 → 29.1 µs)
with a byte- and time-identical stage 1 (the difference sits in the stage-2/3 launch queued behind the (2,32)
stream; the CUDA-graph form of the same cell is +0.2 %).  Every other re-measured cell is neutral or faster.
On R200 the k = 1000 CUDA-graph rows flagged in both orders (V = 262144, B = 1 / 2) are neutral in independent
processes (27.97 / 28.06 vs 28.00 / 28.10 µs and 27.42 / 27.49 vs 27.39 / 27.42 µs).

.. list-table:: H100 SXM (sm_90a, 132 SMs), CUPTI median µs, top_k_first → cake_sampling
   :header-rows: 1
   :widths: 8 5 22 22 22 22

   * - V
     - B
     - k = 50
     - k = 1000
     - k = 50, CUDA graph
     - k = 1000, CUDA graph
   * - 32768
     - 1
     - 46.3 → 10.2 (4.54x)
     - 46.2 → 14.7 (3.14x)
     - 45.5 → 20.9 (2.18x)
     - 47.6 → 26.3 (1.81x)
   * - 32768
     - 8
     - 50.1 → 10.8 (4.64x)
     - 49.8 → 15.1 (3.30x)
     - 49.4 → 21.8 (2.27x)
     - 56.6 → 27.1 (2.09x)
   * - 32768
     - 16
     - 52.7 → 11.5 (4.58x)
     - 53.1 → 15.8 (3.36x)
     - 52.4 → 22.3 (2.35x)
     - 54.5 → 27.6 (1.97x)
   * - 32768
     - 32
     - 54.5 → 11.7 (4.66x)
     - 55.0 → 17.7 (3.11x)
     - 55.2 → 22.5 (2.45x)
     - 55.9 → 29.7 (1.88x)
   * - 32768
     - 64
     - 55.7 → 13.0 (4.28x)
     - 56.3 → 19.3 (2.92x)
     - 58.7 → 23.6 (2.49x)
     - 60.6 → 30.9 (1.96x)
   * - 32768
     - 128
     - 58.0 → 16.5 (3.52x)
     - 58.7 → 23.6 (2.49x)
     - 66.5 → 27.1 (2.45x)
     - 71.6 → 35.6 (2.01x)
   * - 128256
     - 1
     - 112.9 → 12.7 (8.89x)
     - 65.9 → 19.9 (3.31x)
     - 82.2 → 23.7 (3.47x)
     - 80.9 → 31.8 (2.54x)
   * - 128256
     - 8
     - 113.5 → 14.5 (7.83x)
     - 81.0 → 20.5 (3.95x)
     - 86.5 → 25.6 (3.38x)
     - 93.2 → 33.1 (2.82x)
   * - 128256
     - 16
     - 113.2 → 15.9 (7.12x)
     - 92.7 → 22.1 (4.19x)
     - 90.1 → 27.1 (3.32x)
     - 101.2 → 35.4 (2.86x)
   * - 128256
     - 32
     - 116.8 → 18.7 (6.25x)
     - 106.8 → 29.7 (3.60x)
     - 92.6 → 29.9 (3.10x)
     - 127.6 → 41.6 (3.07x)
   * - 128256
     - 64
     - 117.5 → 25.7 (4.57x)
     - 168.6 → 34.9 (4.83x)
     - 101.3 → 37.1 (2.73x)
     - 185.1 → 47.9 (3.86x)
   * - 128256
     - 128
     - 117.3 → 42.0 (2.79x)
     - 236.3 → 46.7 (5.06x)
     - 123.2 → 53.1 (2.32x)
     - 261.8 → 59.0 (4.44x)
   * - 151936
     - 1
     - 113.6 → 13.3 (8.54x)
     - 74.1 → 21.0 (3.53x)
     - 86.6 → 24.6 (3.52x)
     - 90.8 → 33.4 (2.72x)
   * - 151936
     - 8
     - 113.7 → 14.9 (7.63x)
     - 91.7 → 22.0 (4.17x)
     - 91.7 → 26.3 (3.49x)
     - 128.6 → 34.6 (3.72x)
   * - 151936
     - 16
     - 114.1 → 16.5 (6.92x)
     - 104.6 → 23.1 (4.53x)
     - 94.7 → 27.9 (3.39x)
     - 105.9 → 36.7 (2.89x)
   * - 151936
     - 32
     - 114.9 → 21.0 (5.47x)
     - 123.3 → 27.8 (4.44x)
     - 99.6 → 32.2 (3.09x)
     - 141.4 → 40.5 (3.49x)
   * - 151936
     - 64
     - 118.4 → 28.9 (4.10x)
     - 195.6 → 34.9 (5.60x)
     - 113.6 → 40.6 (2.80x)
     - 212.8 → 48.9 (4.35x)
   * - 151936
     - 128
     - 155.2 → 42.8 (3.63x)
     - 278.7 → 52.3 (5.33x)
     - 174.2 → 54.3 (3.21x)
     - 292.9 → 66.0 (4.44x)
   * - 262144
     - 1
     - 113.9 → 13.9 (8.19x)
     - 92.4 → 22.5 (4.11x)
     - 88.3 → 25.3 (3.49x)
     - 111.0 → 34.0 (3.26x)
   * - 262144
     - 8
     - 115.3 → 16.5 (6.99x)
     - 121.4 → 23.3 (5.21x)
     - 92.9 → 28.2 (3.29x)
     - 132.9 → 37.1 (3.58x)
   * - 262144
     - 16
     - 114.7 → 19.8 (5.79x)
     - 149.6 → 26.6 (5.62x)
     - 98.4 → 31.5 (3.12x)
     - 142.5 → 40.8 (3.49x)
   * - 262144
     - 32
     - 126.0 → 28.0 (4.50x)
     - 231.5 → 39.6 (5.85x)
     - 146.3 → 40.4 (3.62x)
     - 237.7 → 53.0 (4.48x)
   * - 262144
     - 64
     - 133.6 → 44.2 (3.02x)
     - 315.9 → 51.9 (6.09x)
     - 149.5 → 56.4 (2.65x)
     - 332.7 → 65.5 (5.08x)
   * - 262144
     - 128
     - 156.7 → 64.5 (2.43x)
     - 471.2 → 71.5 (6.59x)
     - 175.2 → 75.8 (2.31x)
     - 510.1 → 85.2 (5.99x)

.. list-table:: B200 (sm_100a, 148 SMs), CUPTI median µs, top_k_first → cake_sampling
   :header-rows: 1
   :widths: 8 5 22 22 22 22

   * - V
     - B
     - k = 50
     - k = 1000
     - k = 50, CUDA graph
     - k = 1000, CUDA graph
   * - 32768
     - 1
     - 41.9 → 10.2 (4.11x)
     - 42.1 → 14.4 (2.92x)
     - 43.3 → 19.4 (2.23x)
     - 45.3 → 25.0 (1.81x)
   * - 32768
     - 8
     - 46.8 → 10.3 (4.54x)
     - 46.6 → 14.6 (3.19x)
     - 47.1 → 20.0 (2.35x)
     - 53.9 → 26.3 (2.05x)
   * - 32768
     - 16
     - 49.4 → 10.4 (4.75x)
     - 49.1 → 15.1 (3.25x)
     - 53.8 → 19.8 (2.72x)
     - 46.6 → 26.1 (1.79x)
   * - 32768
     - 32
     - 50.8 → 10.7 (4.75x)
     - 51.4 → 15.8 (3.25x)
     - 54.2 → 20.2 (2.68x)
     - 52.8 → 27.3 (1.93x)
   * - 32768
     - 64
     - 52.2 → 11.2 (4.66x)
     - 52.4 → 17.2 (3.05x)
     - 54.5 → 21.3 (2.56x)
     - 54.5 → 28.7 (1.90x)
   * - 32768
     - 128
     - 53.6 → 12.4 (4.32x)
     - 53.4 → 18.5 (2.89x)
     - 55.9 → 22.0 (2.54x)
     - 63.0 → 30.6 (2.06x)
   * - 128256
     - 1
     - 101.1 → 12.0 (8.42x)
     - 63.5 → 17.2 (3.69x)
     - 82.8 → 21.7 (3.82x)
     - 80.4 → 28.1 (2.86x)
   * - 128256
     - 8
     - 101.9 → 12.5 (8.15x)
     - 82.4 → 18.0 (4.58x)
     - 83.8 → 22.7 (3.69x)
     - 115.9 → 29.8 (3.89x)
   * - 128256
     - 16
     - 101.5 → 13.5 (7.52x)
     - 90.8 → 20.7 (4.39x)
     - 82.3 → 23.7 (3.47x)
     - 114.1 → 31.4 (3.63x)
   * - 128256
     - 32
     - 104.0 → 15.0 (6.93x)
     - 99.6 → 21.5 (4.63x)
     - 87.0 → 25.4 (3.43x)
     - 132.1 → 33.0 (4.00x)
   * - 128256
     - 64
     - 105.1 → 17.8 (5.90x)
     - 144.4 → 29.2 (4.95x)
     - 85.9 → 28.1 (3.06x)
     - 158.9 → 41.5 (3.83x)
   * - 128256
     - 128
     - 105.8 → 22.8 (4.64x)
     - 193.5 → 36.2 (5.35x)
     - 93.5 → 33.2 (2.82x)
     - 202.5 → 48.2 (4.20x)
   * - 151936
     - 1
     - 102.1 → 13.4 (7.62x)
     - 69.7 → 18.7 (3.73x)
     - 84.1 → 23.0 (3.66x)
     - 90.0 → 28.9 (3.11x)
   * - 151936
     - 8
     - 103.3 → 13.8 (7.49x)
     - 93.7 → 19.5 (4.81x)
     - 91.5 → 24.1 (3.80x)
     - 102.0 → 30.4 (3.36x)
   * - 151936
     - 16
     - 102.9 → 14.7 (7.00x)
     - 104.2 → 21.7 (4.80x)
     - 88.8 → 25.4 (3.50x)
     - 105.6 → 33.4 (3.16x)
   * - 151936
     - 32
     - 103.1 → 16.4 (6.29x)
     - 112.8 → 22.6 (4.99x)
     - 92.3 → 27.3 (3.38x)
     - 135.9 → 34.7 (3.92x)
   * - 151936
     - 64
     - 106.3 → 19.7 (5.40x)
     - 164.4 → 28.2 (5.83x)
     - 96.7 → 30.7 (3.15x)
     - 184.4 → 41.1 (4.49x)
   * - 151936
     - 128
     - 107.1 → 25.8 (4.15x)
     - 223.5 → 42.2 (5.30x)
     - 105.8 → 36.3 (2.91x)
     - 236.5 → 57.0 (4.15x)
   * - 262144
     - 1
     - 103.4 → 13.7 (7.55x)
     - 87.6 → 20.8 (4.21x)
     - 89.5 → 23.6 (3.79x)
     - 116.2 → 30.7 (3.79x)
   * - 262144
     - 8
     - 104.6 → 14.2 (7.37x)
     - 121.3 → 21.1 (5.75x)
     - 92.8 → 24.9 (3.73x)
     - 169.6 → 31.7 (5.35x)
   * - 262144
     - 16
     - 104.4 → 15.9 (6.57x)
     - 144.6 → 23.4 (6.18x)
     - 91.2 → 26.8 (3.40x)
     - 138.2 → 35.4 (3.90x)
   * - 262144
     - 32
     - 115.2 → 18.6 (6.19x)
     - 201.0 → 25.4 (7.91x)
     - 132.7 → 29.4 (4.51x)
     - 181.3 → 38.1 (4.76x)
   * - 262144
     - 64
     - 107.7 → 24.9 (4.33x)
     - 264.1 → 34.3 (7.70x)
     - 123.1 → 36.4 (3.38x)
     - 247.4 → 46.9 (5.28x)
   * - 262144
     - 128
     - 123.5 → 40.0 (3.09x)
     - 374.8 → 50.9 (7.36x)
     - 145.9 → 49.7 (2.94x)
     - 382.6 → 63.3 (6.04x)

.. list-table:: GB300 (sm_103a, 152 SMs), CUPTI median µs, top_k_first → cake_sampling
   :header-rows: 1
   :widths: 8 5 22 22 22 22

   * - V
     - B
     - k = 50
     - k = 1000
     - k = 50, CUDA graph
     - k = 1000, CUDA graph
   * - 32768
     - 1
     - 68.6 → 10.4 (6.60x)
     - 69.5 → 19.9 (3.49x)
     - 42.1 → 22.0 (1.91x)
     - 41.8 → 27.3 (1.53x)
   * - 32768
     - 8
     - 72.0 → 10.4 (6.92x)
     - 73.3 → 19.4 (3.78x)
     - 46.5 → 21.4 (2.17x)
     - 51.6 → 27.2 (1.90x)
   * - 32768
     - 16
     - 75.5 → 10.6 (7.12x)
     - 74.8 → 19.8 (3.78x)
     - 59.4 → 22.0 (2.70x)
     - 55.5 → 27.3 (2.03x)
   * - 32768
     - 32
     - 75.5 → 10.8 (6.99x)
     - 76.1 → 19.1 (3.98x)
     - 47.5 → 22.3 (2.13x)
     - 54.5 → 29.3 (1.86x)
   * - 32768
     - 64
     - 78.3 → 11.5 (6.81x)
     - 78.5 → 19.6 (4.01x)
     - 58.3 → 22.9 (2.55x)
     - 52.2 → 30.8 (1.69x)
   * - 32768
     - 128
     - 79.9 → 12.5 (6.39x)
     - 79.5 → 19.2 (4.14x)
     - 56.3 → 24.3 (2.32x)
     - 56.7 → 30.8 (1.84x)
   * - 128256
     - 1
     - 178.8 → 11.6 (15.41x)
     - 84.2 → 19.1 (4.41x)
     - 76.7 → 23.6 (3.25x)
     - 64.5 → 29.9 (2.16x)
   * - 128256
     - 8
     - 182.4 → 12.5 (14.59x)
     - 104.2 → 21.4 (4.87x)
     - 85.7 → 24.7 (3.47x)
     - 87.0 → 30.5 (2.85x)
   * - 128256
     - 16
     - 179.0 → 13.9 (12.88x)
     - 109.8 → 20.9 (5.25x)
     - 85.5 → 26.5 (3.23x)
     - 91.3 → 34.3 (2.66x)
   * - 128256
     - 32
     - 182.0 → 15.2 (11.97x)
     - 114.7 → 21.1 (5.44x)
     - 89.4 → 27.6 (3.24x)
     - 97.8 → 35.6 (2.75x)
   * - 128256
     - 64
     - 183.6 → 17.6 (10.43x)
     - 137.7 → 29.1 (4.73x)
     - 89.8 → 29.7 (3.02x)
     - 133.0 → 43.5 (3.06x)
   * - 128256
     - 128
     - 186.4 → 22.5 (8.28x)
     - 184.4 → 35.4 (5.21x)
     - 93.9 → 34.2 (2.75x)
     - 196.5 → 49.3 (3.99x)
   * - 151936
     - 1
     - 184.7 → 12.5 (14.78x)
     - 88.5 → 20.2 (4.38x)
     - 79.9 → 24.9 (3.21x)
     - 82.9 → 32.2 (2.57x)
   * - 151936
     - 8
     - 182.9 → 13.6 (13.45x)
     - 108.8 → 21.0 (5.18x)
     - 90.3 → 26.1 (3.46x)
     - 102.7 → 33.1 (3.10x)
   * - 151936
     - 16
     - 179.6 → 14.8 (12.14x)
     - 117.2 → 22.2 (5.28x)
     - 91.7 → 27.5 (3.33x)
     - 120.8 → 35.3 (3.42x)
   * - 151936
     - 32
     - 179.8 → 16.4 (10.96x)
     - 122.8 → 21.9 (5.61x)
     - 93.5 → 28.8 (3.25x)
     - 112.0 → 36.7 (3.05x)
   * - 151936
     - 64
     - 184.3 → 19.6 (9.40x)
     - 157.0 → 27.0 (5.81x)
     - 94.4 → 31.6 (2.99x)
     - 184.0 → 40.9 (4.50x)
   * - 151936
     - 128
     - 186.0 → 25.8 (7.21x)
     - 213.8 → 37.0 (5.78x)
     - 103.6 → 38.2 (2.71x)
     - 222.0 → 50.8 (4.37x)
   * - 262144
     - 1
     - 181.0 → 12.7 (14.25x)
     - 105.1 → 21.4 (4.91x)
     - 88.0 → 25.4 (3.46x)
     - 78.0 → 36.0 (2.17x)
   * - 262144
     - 8
     - 183.6 → 14.2 (12.93x)
     - 138.3 → 21.8 (6.34x)
     - 89.5 → 27.1 (3.30x)
     - 123.6 → 35.9 (3.44x)
   * - 262144
     - 16
     - 181.7 → 15.9 (11.43x)
     - 151.8 → 22.9 (6.63x)
     - 92.0 → 28.8 (3.19x)
     - 161.1 → 37.1 (4.34x)
   * - 262144
     - 32
     - 182.6 → 18.3 (9.98x)
     - 189.8 → 24.4 (7.78x)
     - 135.3 → 31.2 (4.34x)
     - 223.4 → 38.8 (5.76x)
   * - 262144
     - 64
     - 187.4 → 25.2 (7.44x)
     - 255.2 → 33.1 (7.71x)
     - 122.6 → 37.8 (3.24x)
     - 258.1 → 48.1 (5.37x)
   * - 262144
     - 128
     - 186.3 → 36.3 (5.13x)
     - 365.6 → 48.0 (7.62x)
     - 146.0 → 49.1 (2.97x)
     - 369.1 → 63.2 (5.84x)

.. list-table:: VR200 R200 (sm_107a, 212 SMs), CUPTI median µs, top_k_first → cake_sampling
   :header-rows: 1
   :widths: 8 5 22 22 22 22

   * - V
     - B
     - k = 50
     - k = 1000
     - k = 50, CUDA graph
     - k = 1000, CUDA graph
   * - 32768
     - 1
     - 29.8 → 9.1 (3.27x)
     - 30.1 → 12.5 (2.41x)
     - 38.2 → 18.3 (2.09x)
     - 36.6 → 22.7 (1.61x)
   * - 32768
     - 8
     - 33.1 → 10.0 (3.31x)
     - 33.6 → 13.7 (2.45x)
     - 39.6 → 19.3 (2.05x)
     - 44.0 → 24.5 (1.80x)
   * - 32768
     - 16
     - 35.3 → 10.6 (3.33x)
     - 35.2 → 13.8 (2.55x)
     - 42.8 → 19.7 (2.17x)
     - 40.6 → 24.4 (1.66x)
   * - 32768
     - 32
     - 36.3 → 11.0 (3.30x)
     - 36.9 → 13.9 (2.65x)
     - 42.5 → 20.0 (2.12x)
     - 44.7 → 24.4 (1.83x)
   * - 32768
     - 64
     - 37.7 → 10.9 (3.46x)
     - 38.3 → 15.9 (2.41x)
     - 46.3 → 19.8 (2.34x)
     - 43.8 → 26.1 (1.68x)
   * - 32768
     - 128
     - 38.9 → 11.4 (3.41x)
     - 39.2 → 16.3 (2.40x)
     - 45.4 → 20.0 (2.27x)
     - 50.8 → 27.7 (1.83x)
   * - 128256
     - 1
     - 78.9 → 11.2 (7.04x)
     - 55.4 → 14.7 (3.77x)
     - 72.9 → 20.5 (3.56x)
     - 64.9 → 25.6 (2.54x)
   * - 128256
     - 8
     - 81.6 → 11.9 (6.86x)
     - 67.6 → 16.2 (4.17x)
     - 77.2 → 21.2 (3.64x)
     - 73.5 → 26.4 (2.78x)
   * - 128256
     - 16
     - 80.0 → 13.0 (6.15x)
     - 75.9 → 16.9 (4.49x)
     - 73.8 → 22.3 (3.31x)
     - 92.8 → 27.4 (3.39x)
   * - 128256
     - 32
     - 84.8 → 13.5 (6.28x)
     - 83.3 → 18.5 (4.50x)
     - 79.8 → 22.7 (3.52x)
     - 85.2 → 29.3 (2.91x)
   * - 128256
     - 64
     - 84.3 → 15.5 (5.44x)
     - 93.0 → 24.4 (3.81x)
     - 81.6 → 24.6 (3.32x)
     - 96.1 → 35.6 (2.70x)
   * - 128256
     - 128
     - 84.5 → 19.2 (4.40x)
     - 139.9 → 30.0 (4.66x)
     - 81.8 → 28.1 (2.91x)
     - 148.8 → 42.3 (3.52x)
   * - 151936
     - 1
     - 81.4 → 12.1 (6.73x)
     - 63.0 → 17.1 (3.68x)
     - 72.9 → 21.1 (3.45x)
     - 72.4 → 27.3 (2.65x)
   * - 151936
     - 8
     - 80.4 → 13.0 (6.18x)
     - 76.9 → 18.0 (4.27x)
     - 80.1 → 22.1 (3.62x)
     - 83.1 → 28.2 (2.95x)
   * - 151936
     - 16
     - 82.1 → 14.1 (5.82x)
     - 86.2 → 18.4 (4.68x)
     - 79.5 → 23.0 (3.46x)
     - 87.7 → 29.2 (3.00x)
   * - 151936
     - 32
     - 82.9 → 14.6 (5.68x)
     - 96.3 → 19.4 (4.96x)
     - 84.8 → 24.0 (3.53x)
     - 111.5 → 29.8 (3.74x)
   * - 151936
     - 64
     - 84.8 → 17.0 (4.99x)
     - 107.0 → 22.5 (4.76x)
     - 84.5 → 26.1 (3.24x)
     - 132.2 → 32.8 (4.03x)
   * - 151936
     - 128
     - 86.0 → 22.2 (3.87x)
     - 162.5 → 36.2 (4.49x)
     - 92.5 → 30.9 (2.99x)
     - 168.4 → 48.0 (3.51x)
   * - 262144
     - 1
     - 81.1 → 12.1 (6.70x)
     - 78.0 → 18.6 (4.19x)
     - 77.9 → 21.3 (3.66x)
     - 87.9 → 29.8 (2.95x)
   * - 262144
     - 8
     - 80.4 → 13.3 (6.05x)
     - 102.6 → 19.4 (5.29x)
     - 81.5 → 22.4 (3.64x)
     - 136.2 → 30.2 (4.51x)
   * - 262144
     - 16
     - 82.3 → 14.1 (5.84x)
     - 118.0 → 20.3 (5.81x)
     - 79.3 → 23.0 (3.45x)
     - 157.6 → 30.8 (5.12x)
   * - 262144
     - 32
     - 80.2 → 15.8 (5.08x)
     - 133.6 → 21.1 (6.33x)
     - 84.3 → 25.3 (3.33x)
     - 150.2 → 32.3 (4.65x)
   * - 262144
     - 64
     - 88.6 → 20.6 (4.30x)
     - 196.8 → 28.1 (7.00x)
     - 113.3 → 29.5 (3.84x)
     - 209.5 → 38.8 (5.40x)
   * - 262144
     - 128
     - 107.6 → 27.5 (3.91x)
     - 296.2 → 38.0 (7.79x)
     - 127.4 → 36.5 (3.49x)
     - 300.6 → 47.9 (6.28x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
