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
   their round-6 build.  Bits 4 and 6 are exclusive.

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
     - 45.5 → 10.2 (4.46x)
     - 45.8 → 14.8 (3.09x)
     - 43.7 → 21.0 (2.08x)
     - 45.4 → 26.0 (1.75x)
   * - 32768
     - 8
     - 50.4 → 10.8 (4.67x)
     - 50.2 → 15.4 (3.26x)
     - 49.7 → 21.7 (2.29x)
     - 56.6 → 26.8 (2.11x)
   * - 32768
     - 16
     - 52.3 → 11.5 (4.55x)
     - 53.0 → 15.9 (3.33x)
     - 50.4 → 22.4 (2.25x)
     - 58.1 → 27.2 (2.14x)
   * - 32768
     - 32
     - 53.6 → 11.8 (4.54x)
     - 54.9 → 17.7 (3.10x)
     - 52.8 → 22.5 (2.35x)
     - 54.0 → 29.3 (1.84x)
   * - 32768
     - 64
     - 55.2 → 13.0 (4.25x)
     - 55.7 → 21.3 (2.62x)
     - 57.7 → 23.6 (2.44x)
     - 56.3 → 33.8 (1.67x)
   * - 32768
     - 128
     - 57.1 → 16.5 (3.46x)
     - 58.0 → 24.4 (2.38x)
     - 72.5 → 27.1 (2.68x)
     - 69.1 → 36.8 (1.88x)
   * - 128256
     - 1
     - 112.1 → 12.7 (8.83x)
     - 66.6 → 19.1 (3.49x)
     - 81.4 → 23.7 (3.43x)
     - 76.4 → 32.7 (2.34x)
   * - 128256
     - 8
     - 113.1 → 14.5 (7.80x)
     - 81.1 → 20.5 (3.96x)
     - 86.8 → 25.6 (3.39x)
     - 91.2 → 32.9 (2.77x)
   * - 128256
     - 16
     - 113.6 → 16.0 (7.10x)
     - 92.2 → 25.0 (3.69x)
     - 87.8 → 27.1 (3.24x)
     - 106.0 → 37.0 (2.86x)
   * - 128256
     - 32
     - 115.5 → 19.7 (5.86x)
     - 106.1 → 30.6 (3.47x)
     - 92.4 → 31.1 (2.97x)
     - 121.8 → 42.2 (2.89x)
   * - 128256
     - 64
     - 115.9 → 26.8 (4.32x)
     - 167.3 → 36.9 (4.53x)
     - 104.0 → 38.3 (2.72x)
     - 171.4 → 49.5 (3.46x)
   * - 128256
     - 128
     - 116.4 → 42.0 (2.77x)
     - 236.2 → 53.4 (4.42x)
     - 121.2 → 53.1 (2.28x)
     - 241.9 → 65.3 (3.70x)
   * - 151936
     - 1
     - 113.1 → 13.3 (8.50x)
     - 75.4 → 19.8 (3.81x)
     - 87.3 → 24.7 (3.53x)
     - 84.9 → 33.0 (2.57x)
   * - 151936
     - 8
     - 114.1 → 14.9 (7.66x)
     - 91.6 → 21.6 (4.24x)
     - 92.2 → 26.4 (3.49x)
     - 101.0 → 34.3 (2.94x)
   * - 151936
     - 16
     - 114.0 → 17.1 (6.67x)
     - 106.5 → 28.7 (3.71x)
     - 92.7 → 28.7 (3.23x)
     - 120.6 → 41.5 (2.91x)
   * - 151936
     - 32
     - 113.9 → 22.5 (5.06x)
     - 123.7 → 34.8 (3.55x)
     - 96.9 → 34.0 (2.85x)
     - 139.7 → 47.5 (2.94x)
   * - 151936
     - 64
     - 116.7 → 30.9 (3.78x)
     - 195.1 → 42.2 (4.62x)
     - 116.7 → 42.4 (2.75x)
     - 221.0 → 56.2 (3.93x)
   * - 151936
     - 128
     - 156.1 → 46.9 (3.33x)
     - 279.0 → 59.5 (4.69x)
     - 172.5 → 58.3 (2.96x)
     - 309.3 → 72.2 (4.28x)
   * - 262144
     - 1
     - 113.3 → 13.9 (8.15x)
     - 91.6 → 20.9 (4.38x)
     - 88.0 → 25.5 (3.45x)
     - 101.0 → 33.9 (2.98x)
   * - 262144
     - 8
     - 114.0 → 16.5 (6.91x)
     - 121.7 → 23.0 (5.29x)
     - 93.3 → 28.4 (3.29x)
     - 188.3 → 36.0 (5.23x)
   * - 262144
     - 16
     - 114.4 → 20.8 (5.50x)
     - 147.1 → 27.7 (5.31x)
     - 96.4 → 32.6 (2.96x)
     - 136.0 → 40.2 (3.38x)
   * - 262144
     - 32
     - 127.6 → 28.2 (4.52x)
     - 233.5 → 39.6 (5.90x)
     - 144.1 → 40.8 (3.53x)
     - 240.0 → 52.3 (4.59x)
   * - 262144
     - 64
     - 134.8 → 44.3 (3.04x)
     - 318.7 → 54.2 (5.88x)
     - 153.3 → 56.4 (2.72x)
     - 313.1 → 67.4 (4.65x)
   * - 262144
     - 128
     - 157.4 → 69.2 (2.27x)
     - 470.9 → 84.1 (5.60x)
     - 173.5 → 81.0 (2.14x)
     - 475.0 → 96.7 (4.91x)

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
     - 43.2 → 10.7 (4.04x)
     - 42.1 → 14.5 (2.90x)
     - 43.2 → 20.3 (2.13x)
     - 40.7 → 25.0 (1.63x)
   * - 32768
     - 8
     - 47.6 → 10.9 (4.37x)
     - 46.2 → 14.6 (3.16x)
     - 55.9 → 21.1 (2.65x)
     - 62.2 → 26.3 (2.37x)
   * - 32768
     - 16
     - 50.5 → 10.4 (4.86x)
     - 49.2 → 15.1 (3.26x)
     - 53.7 → 20.0 (2.69x)
     - 47.2 → 26.4 (1.79x)
   * - 32768
     - 32
     - 51.8 → 10.8 (4.80x)
     - 51.4 → 15.8 (3.25x)
     - 51.8 → 20.2 (2.56x)
     - 50.5 → 27.3 (1.85x)
   * - 32768
     - 64
     - 53.3 → 11.2 (4.76x)
     - 52.7 → 18.2 (2.90x)
     - 52.3 → 21.2 (2.47x)
     - 57.3 → 29.9 (1.92x)
   * - 32768
     - 128
     - 54.5 → 12.6 (4.33x)
     - 53.3 → 20.3 (2.63x)
     - 56.5 → 22.1 (2.56x)
     - 55.6 → 32.9 (1.69x)
   * - 128256
     - 1
     - 103.7 → 12.2 (8.50x)
     - 62.3 → 17.2 (3.62x)
     - 83.2 → 21.9 (3.80x)
     - 59.4 → 28.1 (2.11x)
   * - 128256
     - 8
     - 103.7 → 13.0 (7.98x)
     - 81.3 → 18.0 (4.52x)
     - 81.9 → 23.3 (3.52x)
     - 85.9 → 29.8 (2.88x)
   * - 128256
     - 16
     - 103.6 → 13.5 (7.67x)
     - 90.6 → 22.7 (3.99x)
     - 84.6 → 24.0 (3.52x)
     - 107.5 → 34.0 (3.16x)
   * - 128256
     - 32
     - 107.6 → 15.0 (7.17x)
     - 99.6 → 24.1 (4.13x)
     - 86.7 → 25.2 (3.44x)
     - 128.9 → 35.9 (3.59x)
   * - 128256
     - 64
     - 107.5 → 17.8 (6.04x)
     - 143.7 → 29.1 (4.94x)
     - 90.6 → 28.3 (3.20x)
     - 185.9 → 41.5 (4.48x)
   * - 128256
     - 128
     - 108.0 → 22.8 (4.74x)
     - 193.4 → 36.0 (5.37x)
     - 97.3 → 32.9 (2.96x)
     - 201.8 → 47.4 (4.26x)
   * - 151936
     - 1
     - 104.1 → 12.8 (8.13x)
     - 69.6 → 19.8 (3.52x)
     - 83.7 → 23.6 (3.55x)
     - 64.5 → 30.0 (2.15x)
   * - 151936
     - 8
     - 105.2 → 13.7 (7.68x)
     - 91.4 → 20.7 (4.42x)
     - 89.2 → 24.3 (3.67x)
     - 133.9 → 32.1 (4.17x)
   * - 151936
     - 16
     - 105.2 → 14.7 (7.16x)
     - 101.8 → 22.5 (4.52x)
     - 90.9 → 25.4 (3.58x)
     - 104.3 → 34.5 (3.02x)
   * - 151936
     - 32
     - 105.6 → 16.3 (6.48x)
     - 113.3 → 23.6 (4.80x)
     - 92.2 → 27.2 (3.39x)
     - 141.2 → 35.7 (3.96x)
   * - 151936
     - 64
     - 109.0 → 19.8 (5.51x)
     - 164.3 → 34.2 (4.80x)
     - 99.1 → 30.7 (3.23x)
     - 182.8 → 46.3 (3.95x)
   * - 151936
     - 128
     - 109.0 → 25.8 (4.22x)
     - 222.5 → 42.2 (5.27x)
     - 106.8 → 36.3 (2.94x)
     - 271.1 → 56.3 (4.82x)
   * - 262144
     - 1
     - 105.0 → 12.8 (8.20x)
     - 86.7 → 21.2 (4.09x)
     - 89.6 → 23.6 (3.80x)
     - 72.6 → 31.9 (2.28x)
   * - 262144
     - 8
     - 106.8 → 14.3 (7.47x)
     - 123.0 → 21.9 (5.62x)
     - 90.6 → 25.2 (3.60x)
     - 184.8 → 33.7 (5.48x)
   * - 262144
     - 16
     - 106.6 → 16.0 (6.66x)
     - 140.8 → 23.4 (6.02x)
     - 93.1 → 27.0 (3.45x)
     - 177.2 → 35.4 (5.01x)
   * - 262144
     - 32
     - 114.6 → 18.6 (6.16x)
     - 200.3 → 25.3 (7.92x)
     - 132.6 → 29.6 (4.48x)
     - 204.4 → 38.9 (5.25x)
   * - 262144
     - 64
     - 109.9 → 24.8 (4.43x)
     - 262.8 → 34.3 (7.66x)
     - 125.7 → 36.0 (3.49x)
     - 290.8 → 47.7 (6.10x)
   * - 262144
     - 128
     - 123.6 → 41.3 (2.99x)
     - 375.7 → 52.6 (7.14x)
     - 146.3 → 50.8 (2.88x)
     - 361.2 → 64.7 (5.58x)

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
     - 68.1 → 10.7 (6.36x)
     - 67.6 → 20.5 (3.30x)
     - 42.2 → 23.1 (1.83x)
     - 47.4 → 27.8 (1.71x)
   * - 32768
     - 8
     - 70.5 → 11.1 (6.35x)
     - 71.5 → 19.9 (3.59x)
     - 50.6 → 22.6 (2.24x)
     - 56.4 → 26.7 (2.11x)
   * - 32768
     - 16
     - 72.7 → 10.6 (6.86x)
     - 73.6 → 20.0 (3.68x)
     - 54.4 → 22.5 (2.42x)
     - 50.7 → 27.4 (1.85x)
   * - 32768
     - 32
     - 74.1 → 10.8 (6.86x)
     - 75.4 → 19.9 (3.79x)
     - 51.8 → 22.4 (2.31x)
     - 55.5 → 29.2 (1.90x)
   * - 32768
     - 64
     - 76.3 → 11.5 (6.63x)
     - 77.0 → 19.9 (3.87x)
     - 55.2 → 23.0 (2.40x)
     - 56.7 → 31.9 (1.78x)
   * - 32768
     - 128
     - 77.6 → 12.5 (6.21x)
     - 78.1 → 20.0 (3.90x)
     - 56.7 → 24.4 (2.32x)
     - 62.2 → 33.2 (1.87x)
   * - 128256
     - 1
     - 175.2 → 12.2 (14.36x)
     - 82.2 → 19.7 (4.17x)
     - 76.7 → 24.2 (3.17x)
     - 79.2 → 30.5 (2.60x)
   * - 128256
     - 8
     - 178.2 → 13.2 (13.50x)
     - 98.7 → 20.5 (4.81x)
     - 81.0 → 25.4 (3.19x)
     - 87.9 → 30.5 (2.88x)
   * - 128256
     - 16
     - 174.4 → 13.9 (12.55x)
     - 105.1 → 23.0 (4.57x)
     - 84.5 → 26.5 (3.19x)
     - 86.7 → 36.6 (2.37x)
   * - 128256
     - 32
     - 179.2 → 15.2 (11.79x)
     - 113.4 → 23.7 (4.78x)
     - 86.8 → 27.6 (3.14x)
     - 105.8 → 38.1 (2.78x)
   * - 128256
     - 64
     - 181.0 → 17.6 (10.28x)
     - 136.7 → 29.1 (4.70x)
     - 89.2 → 29.9 (2.98x)
     - 167.0 → 44.3 (3.77x)
   * - 128256
     - 128
     - 184.4 → 22.5 (8.20x)
     - 184.7 → 35.5 (5.20x)
     - 93.9 → 34.5 (2.72x)
     - 197.4 → 49.1 (4.02x)
   * - 151936
     - 1
     - 180.1 → 12.8 (14.07x)
     - 87.1 → 20.8 (4.19x)
     - 79.9 → 25.7 (3.11x)
     - 87.7 → 33.5 (2.62x)
   * - 151936
     - 8
     - 178.8 → 13.9 (12.86x)
     - 105.2 → 20.7 (5.08x)
     - 84.9 → 26.5 (3.20x)
     - 116.0 → 34.3 (3.38x)
   * - 151936
     - 16
     - 179.6 → 14.9 (12.05x)
     - 115.5 → 22.3 (5.18x)
     - 90.0 → 27.6 (3.26x)
     - 123.3 → 35.6 (3.46x)
   * - 151936
     - 32
     - 178.1 → 16.3 (10.93x)
     - 123.4 → 22.9 (5.39x)
     - 90.8 → 29.0 (3.13x)
     - 129.2 → 37.1 (3.48x)
   * - 151936
     - 64
     - 181.3 → 19.6 (9.25x)
     - 157.0 → 27.0 (5.81x)
     - 92.2 → 31.8 (2.90x)
     - 173.9 → 41.3 (4.21x)
   * - 151936
     - 128
     - 185.0 → 25.8 (7.17x)
     - 213.2 → 37.2 (5.73x)
     - 102.2 → 38.6 (2.65x)
     - 217.5 → 50.7 (4.29x)
   * - 262144
     - 1
     - 177.6 → 13.1 (13.56x)
     - 103.4 → 20.4 (5.07x)
     - 83.2 → 26.1 (3.19x)
     - 112.1 → 34.6 (3.24x)
   * - 262144
     - 8
     - 178.2 → 14.4 (12.37x)
     - 135.1 → 21.1 (6.40x)
     - 83.9 → 27.1 (3.10x)
     - 159.6 → 34.4 (4.64x)
   * - 262144
     - 16
     - 180.3 → 16.1 (11.20x)
     - 152.6 → 22.9 (6.66x)
     - 90.6 → 28.9 (3.13x)
     - 164.5 → 36.8 (4.47x)
   * - 262144
     - 32
     - 174.5 → 18.5 (9.43x)
     - 188.9 → 24.7 (7.65x)
     - 131.6 → 31.1 (4.23x)
     - 211.1 → 38.8 (5.44x)
   * - 262144
     - 64
     - 182.9 → 25.2 (7.26x)
     - 254.7 → 33.3 (7.65x)
     - 120.8 → 37.8 (3.20x)
     - 266.9 → 46.9 (5.69x)
   * - 262144
     - 128
     - 180.5 → 38.3 (4.71x)
     - 360.7 → 56.9 (6.34x)
     - 144.6 → 51.4 (2.81x)
     - 373.6 → 70.3 (5.31x)

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
     - 29.9 → 9.1 (3.29x)
     - 30.1 → 12.5 (2.41x)
     - 39.0 → 16.6 (2.35x)
     - 33.7 → 21.3 (1.58x)
   * - 32768
     - 8
     - 33.7 → 9.9 (3.40x)
     - 33.4 → 13.5 (2.47x)
     - 37.8 → 17.5 (2.16x)
     - 37.7 → 22.6 (1.67x)
   * - 32768
     - 16
     - 35.6 → 10.5 (3.39x)
     - 35.6 → 13.6 (2.62x)
     - 43.7 → 18.1 (2.41x)
     - 38.9 → 22.7 (1.71x)
   * - 32768
     - 32
     - 37.3 → 11.0 (3.39x)
     - 37.3 → 13.8 (2.70x)
     - 42.2 → 18.3 (2.31x)
     - 43.3 → 22.7 (1.91x)
   * - 32768
     - 64
     - 38.5 → 10.8 (3.56x)
     - 38.6 → 15.8 (2.44x)
     - 46.3 → 18.3 (2.53x)
     - 44.2 → 24.4 (1.81x)
   * - 32768
     - 128
     - 39.3 → 11.4 (3.45x)
     - 39.8 → 17.2 (2.31x)
     - 52.2 → 18.4 (2.84x)
     - 52.4 → 26.8 (1.96x)
   * - 128256
     - 1
     - 70.6 → 11.1 (6.36x)
     - 56.1 → 14.6 (3.84x)
     - 70.0 → 18.6 (3.76x)
     - 58.6 → 24.1 (2.43x)
   * - 128256
     - 8
     - 71.6 → 11.8 (6.07x)
     - 68.1 → 16.1 (4.23x)
     - 74.4 → 19.6 (3.80x)
     - 74.6 → 24.9 (3.00x)
   * - 128256
     - 16
     - 71.3 → 12.9 (5.53x)
     - 75.3 → 16.6 (4.54x)
     - 72.7 → 20.5 (3.55x)
     - 75.2 → 26.0 (2.89x)
   * - 128256
     - 32
     - 73.4 → 13.5 (5.44x)
     - 84.6 → 20.7 (4.09x)
     - 76.6 → 21.0 (3.65x)
     - 89.6 → 30.2 (2.97x)
   * - 128256
     - 64
     - 73.3 → 15.5 (4.73x)
     - 92.3 → 24.9 (3.71x)
     - 78.9 → 23.1 (3.42x)
     - 91.0 → 34.4 (2.65x)
   * - 128256
     - 128
     - 73.2 → 19.2 (3.81x)
     - 138.0 → 30.7 (4.50x)
     - 79.0 → 26.6 (2.97x)
     - 144.6 → 41.4 (3.49x)
   * - 151936
     - 1
     - 70.7 → 12.0 (5.89x)
     - 62.7 → 16.9 (3.71x)
     - 69.8 → 19.4 (3.60x)
     - 84.5 → 26.2 (3.23x)
   * - 151936
     - 8
     - 70.4 → 12.8 (5.50x)
     - 77.1 → 17.8 (4.33x)
     - 77.2 → 20.4 (3.78x)
     - 100.2 → 26.9 (3.72x)
   * - 151936
     - 16
     - 70.8 → 13.9 (5.09x)
     - 87.0 → 18.2 (4.78x)
     - 78.6 → 21.3 (3.69x)
     - 119.2 → 27.7 (4.30x)
   * - 151936
     - 32
     - 70.8 → 14.6 (4.85x)
     - 97.0 → 20.3 (4.78x)
     - 81.7 → 22.4 (3.65x)
     - 105.5 → 29.3 (3.60x)
   * - 151936
     - 64
     - 73.1 → 17.1 (4.27x)
     - 106.5 → 27.4 (3.89x)
     - 81.5 → 24.7 (3.30x)
     - 113.5 → 37.5 (3.03x)
   * - 151936
     - 128
     - 74.0 → 22.4 (3.30x)
     - 159.8 → 35.4 (4.51x)
     - 88.8 → 29.7 (2.99x)
     - 174.7 → 46.5 (3.76x)
   * - 262144
     - 1
     - 70.4 → 12.1 (5.82x)
     - 77.5 → 18.0 (4.31x)
     - 74.8 → 19.7 (3.80x)
     - 74.9 → 28.0 (2.68x)
   * - 262144
     - 8
     - 70.8 → 13.2 (5.36x)
     - 103.9 → 19.1 (5.44x)
     - 78.5 → 20.6 (3.81x)
     - 109.9 → 28.8 (3.82x)
   * - 262144
     - 16
     - 70.8 → 14.1 (5.02x)
     - 117.9 → 20.2 (5.84x)
     - 78.2 → 21.3 (3.67x)
     - 163.2 → 29.5 (5.53x)
   * - 262144
     - 32
     - 71.0 → 15.9 (4.47x)
     - 133.4 → 23.8 (5.61x)
     - 81.7 → 23.5 (3.48x)
     - 122.5 → 33.5 (3.66x)
   * - 262144
     - 64
     - 87.2 → 20.9 (4.17x)
     - 194.1 → 28.0 (6.93x)
     - 109.6 → 28.6 (3.83x)
     - 176.1 → 37.4 (4.71x)
   * - 262144
     - 128
     - 106.0 → 30.5 (3.48x)
     - 295.3 → 40.7 (7.26x)
     - 123.1 → 37.8 (3.26x)
     - 301.3 → 49.9 (6.04x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
