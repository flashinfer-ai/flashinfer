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
   the variant (manifest entries with ``fused_block_tail``, symbol suffix ``_bt``).  The frozen
   bundle ships no such build and no compute capability selects the form: on the shipped kernels it
   lost to the two-launch chain on every architecture except four H100 cells (round 5, unit 6),
   while compiling it into the default build had cost every ``top_k <= 64`` launch 4-19 % on
   H100 / GB300 / R200.  The mechanism stays for a build that wins.  When built, the
   whole-CTA form is again bitwise identical to the
   two-launch form; rows of such a launch whose k is at most 64 take the one-warp tail.  A
   multi-wave grid keeps the two launches (its first CTA would run one tail per wave), and
   every architecture keeps them in round 5 (the register-resident cells of B300 / GB300 run 6-9 %
   slower with the in-kernel tail, and the integrated build lost 5-32 % to the chain on R200, B200
   and most H100 cells).  In the two-launch form the dispatcher also decides where the dependent kernel lands: stage 1 signals
   ``griddepcontrol.launch_dependents`` before its first pass only when the batch fits on the
   SMs its last wave leaves free (one stage-1 CTA per SM), so the stage-2/3 CTAs are never
   packed onto the few SMs free mid-flight; larger batches let the dependent launch as stage 1
   exits.  A streaming variant signals at that early point only on Blackwell and Rubin
   (compute capability 10.x); on Hopper it signals after its filter pass, once the whole row
   has been read, where the round-3 kernels did.  All three decisions travel in the stage-1
   ``launch_flags`` argument (bit 0 one-warp tail, bit 1 early trigger, bit 2 stream pre-pass
   point, bit 3 whole-CTA tail, bit 4 coarse sample, bit 5 row-span filter arm) and none changes
   any output.  Bit 4 selects a
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
   a cluster-1 row that takes the arm can lose 5 % and Rubin measures neutral, so nothing else.

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
both: the *fused* regime (H100 / B200 / R200: a one-wave candidate finishes in one kernel with the
whole-CTA tail, a multi-wave or tail-less candidate pays the stage-2/3 kernel) and the *two-launch*
regime (B300 / GB300, or ``two_launch=True``).  ``choose_stage1(..., two_launch=)`` selects it (None
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
(``flashinfer.testing.bench_gpu_time``, CUPTI kernel time, p = 0.9, median µs), round-5 bundle
(software-pipelined sampled pass and ballot-gated filter, two-level cluster select, two-warp fused
tail, ept-32 streams, static per-architecture stage-2/3 forms, re-fitted k-aware dispatch) on the four
supported architectures.  The summary compares every cell with FlashInfer's default ``top_k_first``
route and with the previous frozen bundle (#5636) measured in the same run.

.. list-table:: Round-5 bundle, 192 cells per architecture (8 batches × 4 vocabularies × 3 k × eager/graph)
   :header-rows: 1
   :widths: 18 14 20 24 24

   * - GPU
     - cells slower than #5636 by > 2 %
     - speedup vs top_k_first (min / median / max)
     - gain vs #5636, B ≤ 16 (min / median / max)
     - gain vs #5636, B ≥ 32 (min / median)
   * - H100 SXM (sm_90a, 132 SMs)
     - 0 of 192
     - 1.73x / 3.44x / 9.09x
     - 1.02x / 1.15x / 1.75x
     - 1.02x / 1.09x
   * - B200 (sm_100a, 148 SMs)
     - 0 of 192
     - 1.74x / 3.82x / 8.23x
     - 1.00x / 1.14x / 1.50x
     - 1.02x / 1.22x
   * - GB300 (sm_103a, 152 SMs)
     - 0 of 192
     - 1.58x / 3.98x / 14.37x
     - 1.00x / 1.16x / 1.48x
     - 1.06x / 1.22x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 0 of 192
     - 1.50x / 3.59x / 7.24x
     - 1.00x / 1.08x / 1.29x
     - 1.00x / 1.18x

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
     - 45.7 → 10.3 (4.44x)
     - 45.8 → 14.9 (3.07x)
     - 48.0 → 20.7 (2.32x)
     - 46.9 → 25.9 (1.81x)
   * - 32768
     - 8
     - 50.4 → 10.8 (4.67x)
     - 50.2 → 15.1 (3.32x)
     - 48.1 → 21.4 (2.25x)
     - 49.1 → 26.8 (1.83x)
   * - 32768
     - 16
     - 53.6 → 11.5 (4.66x)
     - 52.8 → 15.9 (3.32x)
     - 63.0 → 22.0 (2.86x)
     - 58.0 → 27.3 (2.12x)
   * - 32768
     - 32
     - 54.7 → 11.9 (4.60x)
     - 54.5 → 17.7 (3.08x)
     - 57.5 → 22.2 (2.59x)
     - 56.0 → 29.4 (1.90x)
   * - 32768
     - 64
     - 56.4 → 13.2 (4.27x)
     - 56.3 → 21.1 (2.67x)
     - 57.4 → 23.6 (2.43x)
     - 59.0 → 33.8 (1.75x)
   * - 32768
     - 128
     - 58.7 → 16.9 (3.47x)
     - 58.3 → 24.5 (2.38x)
     - 66.0 → 27.2 (2.43x)
     - 66.7 → 37.0 (1.80x)
   * - 128256
     - 1
     - 117.2 → 12.9 (9.09x)
     - 66.0 → 19.7 (3.35x)
     - 81.3 → 23.7 (3.43x)
     - 81.7 → 31.9 (2.56x)
   * - 128256
     - 8
     - 117.6 → 14.5 (8.11x)
     - 81.5 → 20.4 (4.00x)
     - 83.6 → 25.2 (3.32x)
     - 90.0 → 33.0 (2.73x)
   * - 128256
     - 16
     - 117.1 → 16.2 (7.23x)
     - 93.9 → 25.0 (3.76x)
     - 85.1 → 27.0 (3.15x)
     - 125.0 → 37.2 (3.36x)
   * - 128256
     - 32
     - 120.5 → 20.2 (5.97x)
     - 107.9 → 30.6 (3.53x)
     - 92.3 → 31.2 (2.96x)
     - 112.8 → 42.7 (2.64x)
   * - 128256
     - 64
     - 120.7 → 28.5 (4.24x)
     - 167.9 → 37.0 (4.54x)
     - 101.3 → 39.5 (2.56x)
     - 173.2 → 50.0 (3.46x)
   * - 128256
     - 128
     - 121.6 → 45.3 (2.68x)
     - 236.9 → 53.5 (4.43x)
     - 120.9 → 56.1 (2.16x)
     - 239.8 → 65.8 (3.64x)
   * - 151936
     - 1
     - 117.3 → 13.4 (8.75x)
     - 74.0 → 20.0 (3.70x)
     - 86.8 → 24.5 (3.54x)
     - 91.4 → 34.3 (2.66x)
   * - 151936
     - 8
     - 118.8 → 15.0 (7.92x)
     - 92.5 → 21.3 (4.34x)
     - 89.3 → 26.3 (3.40x)
     - 100.9 → 34.6 (2.92x)
   * - 151936
     - 16
     - 118.2 → 17.3 (6.83x)
     - 105.1 → 29.1 (3.61x)
     - 90.0 → 28.7 (3.14x)
     - 139.1 → 41.6 (3.34x)
   * - 151936
     - 32
     - 119.1 → 22.9 (5.20x)
     - 123.9 → 34.6 (3.58x)
     - 96.7 → 34.3 (2.82x)
     - 146.4 → 47.9 (3.06x)
   * - 151936
     - 64
     - 122.8 → 32.9 (3.73x)
     - 195.4 → 42.6 (4.59x)
     - 114.0 → 44.3 (2.57x)
     - 224.4 → 57.2 (3.92x)
   * - 151936
     - 128
     - 156.9 → 50.5 (3.11x)
     - 279.2 → 59.7 (4.68x)
     - 174.8 → 61.9 (2.82x)
     - 302.3 → 73.0 (4.14x)
   * - 262144
     - 1
     - 118.0 → 14.1 (8.37x)
     - 91.6 → 21.4 (4.28x)
     - 88.1 → 25.4 (3.47x)
     - 112.4 → 35.4 (3.18x)
   * - 262144
     - 8
     - 119.3 → 16.5 (7.23x)
     - 121.2 → 23.8 (5.09x)
     - 90.1 → 28.1 (3.21x)
     - 164.7 → 37.1 (4.44x)
   * - 262144
     - 16
     - 118.9 → 21.2 (5.61x)
     - 144.7 → 27.7 (5.22x)
     - 93.8 → 32.8 (2.86x)
     - 199.6 → 41.2 (4.84x)
   * - 262144
     - 32
     - 127.7 → 30.0 (4.26x)
     - 230.5 → 39.5 (5.84x)
     - 143.5 → 42.3 (3.39x)
     - 215.9 → 52.8 (4.09x)
   * - 262144
     - 64
     - 135.1 → 47.4 (2.85x)
     - 317.0 → 54.4 (5.83x)
     - 149.7 → 59.3 (2.52x)
     - 370.2 → 68.3 (5.42x)
   * - 262144
     - 128
     - 158.1 → 75.7 (2.09x)
     - 472.7 → 84.0 (5.63x)
     - 172.8 → 87.1 (1.98x)
     - 481.9 → 97.4 (4.95x)

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
     - 41.6 → 10.4 (4.00x)
     - 42.5 → 14.2 (2.99x)
     - 43.6 → 20.3 (2.15x)
     - 45.6 → 24.9 (1.83x)
   * - 32768
     - 8
     - 45.7 → 10.9 (4.19x)
     - 46.7 → 14.7 (3.18x)
     - 53.3 → 21.0 (2.54x)
     - 47.0 → 25.6 (1.84x)
   * - 32768
     - 16
     - 48.7 → 10.3 (4.73x)
     - 49.6 → 14.8 (3.35x)
     - 55.1 → 20.2 (2.73x)
     - 52.0 → 26.3 (1.98x)
   * - 32768
     - 32
     - 50.6 → 10.8 (4.69x)
     - 51.4 → 15.7 (3.27x)
     - 52.6 → 20.3 (2.59x)
     - 53.4 → 27.2 (1.96x)
   * - 32768
     - 64
     - 52.3 → 11.4 (4.59x)
     - 52.5 → 18.2 (2.88x)
     - 49.0 → 21.3 (2.30x)
     - 55.7 → 29.7 (1.88x)
   * - 32768
     - 128
     - 52.9 → 12.6 (4.20x)
     - 53.9 → 20.1 (2.68x)
     - 56.7 → 22.2 (2.55x)
     - 56.5 → 32.5 (1.74x)
   * - 128256
     - 1
     - 101.7 → 12.7 (8.01x)
     - 65.0 → 17.0 (3.82x)
     - 76.9 → 22.6 (3.40x)
     - 87.0 → 28.0 (3.11x)
   * - 128256
     - 8
     - 101.5 → 13.2 (7.69x)
     - 80.6 → 18.1 (4.45x)
     - 81.5 → 23.4 (3.48x)
     - 92.4 → 29.2 (3.16x)
   * - 128256
     - 16
     - 101.4 → 13.8 (7.35x)
     - 91.1 → 22.5 (4.05x)
     - 78.8 → 24.2 (3.26x)
     - 89.3 → 34.2 (2.61x)
   * - 128256
     - 32
     - 105.0 → 15.4 (6.82x)
     - 100.0 → 24.0 (4.17x)
     - 83.5 → 25.5 (3.27x)
     - 105.2 → 35.8 (2.94x)
   * - 128256
     - 64
     - 105.0 → 18.4 (5.71x)
     - 143.1 → 28.9 (4.95x)
     - 86.7 → 29.0 (2.99x)
     - 150.6 → 41.0 (3.67x)
   * - 128256
     - 128
     - 106.2 → 24.3 (4.37x)
     - 191.9 → 35.9 (5.35x)
     - 96.6 → 34.8 (2.78x)
     - 200.7 → 47.4 (4.23x)
   * - 151936
     - 1
     - 102.4 → 13.4 (7.64x)
     - 73.8 → 19.3 (3.82x)
     - 81.2 → 24.0 (3.38x)
     - 96.5 → 30.2 (3.20x)
   * - 151936
     - 8
     - 113.0 → 14.2 (7.96x)
     - 91.4 → 20.0 (4.57x)
     - 86.0 → 24.9 (3.45x)
     - 104.0 → 31.5 (3.30x)
   * - 151936
     - 16
     - 102.8 → 15.3 (6.72x)
     - 102.9 → 22.2 (4.64x)
     - 84.9 → 26.0 (3.27x)
     - 140.4 → 34.4 (4.08x)
   * - 151936
     - 32
     - 103.3 → 16.9 (6.11x)
     - 112.2 → 23.6 (4.75x)
     - 89.1 → 27.9 (3.19x)
     - 116.6 → 35.5 (3.28x)
   * - 151936
     - 64
     - 106.4 → 20.8 (5.12x)
     - 163.3 → 34.1 (4.79x)
     - 98.1 → 31.7 (3.09x)
     - 174.3 → 47.0 (3.71x)
   * - 151936
     - 128
     - 106.6 → 28.4 (3.75x)
     - 221.5 → 42.4 (5.22x)
     - 106.0 → 38.7 (2.74x)
     - 271.3 → 55.8 (4.86x)
   * - 262144
     - 1
     - 103.2 → 13.3 (7.76x)
     - 88.9 → 20.6 (4.32x)
     - 83.4 → 24.3 (3.43x)
     - 123.7 → 32.3 (3.83x)
   * - 262144
     - 8
     - 104.0 → 14.4 (7.22x)
     - 123.2 → 21.8 (5.65x)
     - 87.1 → 25.2 (3.46x)
     - 141.8 → 33.5 (4.23x)
   * - 262144
     - 16
     - 104.7 → 16.4 (6.38x)
     - 140.7 → 23.1 (6.09x)
     - 90.4 → 27.4 (3.30x)
     - 164.9 → 35.4 (4.66x)
   * - 262144
     - 32
     - 113.3 → 19.2 (5.90x)
     - 200.0 → 25.3 (7.91x)
     - 133.0 → 30.0 (4.43x)
     - 218.0 → 38.7 (5.63x)
   * - 262144
     - 64
     - 107.5 → 26.7 (4.03x)
     - 266.5 → 34.4 (7.75x)
     - 125.2 → 37.8 (3.31x)
     - 267.1 → 47.5 (5.62x)
   * - 262144
     - 128
     - 123.3 → 44.9 (2.75x)
     - 379.4 → 52.6 (7.21x)
     - 142.8 → 55.0 (2.60x)
     - 409.9 → 64.5 (6.36x)

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
     - 71.5 → 10.8 (6.62x)
     - 71.0 → 20.5 (3.46x)
     - 48.7 → 22.5 (2.16x)
     - 43.6 → 27.6 (1.58x)
   * - 32768
     - 8
     - 75.1 → 11.1 (6.77x)
     - 72.3 → 19.8 (3.65x)
     - 53.1 → 22.8 (2.33x)
     - 47.0 → 26.7 (1.76x)
   * - 32768
     - 16
     - 76.9 → 10.7 (7.19x)
     - 73.7 → 20.5 (3.60x)
     - 52.7 → 22.0 (2.40x)
     - 56.2 → 27.1 (2.07x)
   * - 32768
     - 32
     - 77.6 → 11.1 (6.99x)
     - 74.9 → 20.1 (3.73x)
     - 52.2 → 22.5 (2.32x)
     - 53.0 → 29.2 (1.82x)
   * - 32768
     - 64
     - 80.6 → 11.8 (6.83x)
     - 77.6 → 20.3 (3.82x)
     - 52.7 → 23.1 (2.28x)
     - 52.8 → 31.7 (1.67x)
   * - 32768
     - 128
     - 80.7 → 12.7 (6.35x)
     - 80.0 → 20.0 (4.00x)
     - 57.8 → 24.2 (2.39x)
     - 57.3 → 33.2 (1.73x)
   * - 128256
     - 1
     - 181.0 → 12.6 (14.37x)
     - 84.2 → 20.3 (4.15x)
     - 77.7 → 24.4 (3.18x)
     - 93.0 → 30.1 (3.09x)
   * - 128256
     - 8
     - 179.7 → 13.3 (13.51x)
     - 99.9 → 20.2 (4.95x)
     - 84.2 → 25.5 (3.30x)
     - 91.5 → 30.3 (3.02x)
   * - 128256
     - 16
     - 178.6 → 14.3 (12.49x)
     - 105.8 → 23.1 (4.58x)
     - 83.2 → 26.5 (3.14x)
     - 99.4 → 36.3 (2.74x)
   * - 128256
     - 32
     - 178.3 → 15.4 (11.58x)
     - 112.9 → 23.7 (4.76x)
     - 86.9 → 27.6 (3.15x)
     - 90.5 → 38.1 (2.38x)
   * - 128256
     - 64
     - 180.1 → 18.1 (9.95x)
     - 137.4 → 28.7 (4.79x)
     - 90.5 → 30.1 (3.01x)
     - 154.9 → 43.9 (3.53x)
   * - 128256
     - 128
     - 181.6 → 24.0 (7.57x)
     - 184.3 → 35.5 (5.19x)
     - 95.0 → 35.6 (2.67x)
     - 194.6 → 49.2 (3.96x)
   * - 151936
     - 1
     - 175.7 → 13.3 (13.21x)
     - 87.7 → 20.8 (4.22x)
     - 80.0 → 25.5 (3.14x)
     - 74.6 → 33.6 (2.22x)
   * - 151936
     - 8
     - 180.5 → 14.4 (12.53x)
     - 108.0 → 20.5 (5.27x)
     - 87.4 → 26.8 (3.26x)
     - 101.9 → 34.0 (3.00x)
   * - 151936
     - 16
     - 177.7 → 15.4 (11.54x)
     - 116.1 → 22.2 (5.23x)
     - 87.8 → 27.7 (3.17x)
     - 98.7 → 35.9 (2.75x)
   * - 151936
     - 32
     - 183.7 → 16.6 (11.07x)
     - 123.4 → 22.8 (5.41x)
     - 91.2 → 28.9 (3.16x)
     - 128.0 → 37.4 (3.42x)
   * - 151936
     - 64
     - 174.4 → 20.6 (8.47x)
     - 157.4 → 27.0 (5.83x)
     - 94.5 → 32.8 (2.88x)
     - 176.7 → 41.8 (4.23x)
   * - 151936
     - 128
     - 183.2 → 27.7 (6.61x)
     - 213.7 → 37.2 (5.74x)
     - 104.1 → 40.1 (2.60x)
     - 229.5 → 51.2 (4.48x)
   * - 262144
     - 1
     - 179.6 → 13.2 (13.61x)
     - 104.1 → 20.9 (4.98x)
     - 83.3 → 25.7 (3.24x)
     - 86.7 → 35.3 (2.46x)
   * - 262144
     - 8
     - 179.9 → 14.7 (12.24x)
     - 138.4 → 21.3 (6.50x)
     - 88.9 → 27.5 (3.23x)
     - 137.1 → 35.8 (3.83x)
   * - 262144
     - 16
     - 178.7 → 16.5 (10.83x)
     - 146.8 → 22.9 (6.41x)
     - 89.1 → 29.2 (3.05x)
     - 134.7 → 37.6 (3.58x)
   * - 262144
     - 32
     - 176.5 → 18.8 (9.39x)
     - 190.2 → 24.6 (7.73x)
     - 132.9 → 31.5 (4.22x)
     - 327.1 → 39.2 (8.34x)
   * - 262144
     - 64
     - 178.2 → 26.5 (6.72x)
     - 252.8 → 33.3 (7.59x)
     - 123.8 → 39.2 (3.16x)
     - 258.9 → 47.3 (5.47x)
   * - 262144
     - 128
     - 176.8 → 42.2 (4.19x)
     - 363.1 → 56.5 (6.43x)
     - 146.9 → 54.7 (2.69x)
     - 398.8 → 70.3 (5.67x)

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
     - 30.1 → 12.4 (2.43x)
     - 32.3 → 16.5 (1.96x)
     - 33.3 → 21.4 (1.56x)
   * - 32768
     - 8
     - 33.2 → 9.8 (3.39x)
     - 33.3 → 13.5 (2.47x)
     - 40.2 → 17.5 (2.30x)
     - 37.7 → 22.7 (1.66x)
   * - 32768
     - 16
     - 35.8 → 10.4 (3.44x)
     - 35.6 → 13.6 (2.62x)
     - 40.9 → 17.9 (2.28x)
     - 38.7 → 22.9 (1.69x)
   * - 32768
     - 32
     - 36.7 → 10.9 (3.37x)
     - 37.2 → 13.8 (2.70x)
     - 43.1 → 18.3 (2.36x)
     - 45.5 → 22.6 (2.01x)
   * - 32768
     - 64
     - 38.0 → 10.6 (3.58x)
     - 38.5 → 15.7 (2.45x)
     - 49.2 → 18.1 (2.72x)
     - 45.0 → 24.4 (1.84x)
   * - 32768
     - 128
     - 39.1 → 11.4 (3.43x)
     - 39.9 → 17.1 (2.33x)
     - 43.1 → 18.7 (2.30x)
     - 47.1 → 26.6 (1.77x)
   * - 128256
     - 1
     - 70.0 → 11.7 (5.98x)
     - 55.8 → 14.6 (3.82x)
     - 69.4 → 19.4 (3.58x)
     - 58.8 → 24.2 (2.43x)
   * - 128256
     - 8
     - 70.6 → 12.2 (5.79x)
     - 67.8 → 16.0 (4.24x)
     - 73.1 → 20.0 (3.65x)
     - 79.6 → 24.6 (3.24x)
   * - 128256
     - 16
     - 70.5 → 12.9 (5.47x)
     - 75.3 → 16.5 (4.56x)
     - 72.1 → 20.6 (3.50x)
     - 76.6 → 26.0 (2.95x)
   * - 128256
     - 32
     - 72.0 → 14.0 (5.14x)
     - 83.9 → 20.3 (4.13x)
     - 74.3 → 21.4 (3.47x)
     - 101.3 → 29.7 (3.41x)
   * - 128256
     - 64
     - 73.0 → 16.5 (4.42x)
     - 91.3 → 24.5 (3.73x)
     - 77.9 → 23.9 (3.26x)
     - 101.1 → 33.8 (2.99x)
   * - 128256
     - 128
     - 73.3 → 20.5 (3.58x)
     - 137.2 → 30.7 (4.47x)
     - 78.8 → 27.9 (2.82x)
     - 141.1 → 40.8 (3.46x)
   * - 151936
     - 1
     - 70.5 → 12.4 (5.69x)
     - 62.8 → 16.8 (3.74x)
     - 69.2 → 19.9 (3.48x)
     - 64.5 → 25.5 (2.53x)
   * - 151936
     - 8
     - 71.2 → 12.9 (5.52x)
     - 78.0 → 17.7 (4.41x)
     - 76.3 → 20.6 (3.70x)
     - 86.2 → 26.5 (3.25x)
   * - 151936
     - 16
     - 70.6 → 13.7 (5.15x)
     - 86.2 → 18.1 (4.76x)
     - 77.8 → 21.2 (3.67x)
     - 95.3 → 27.5 (3.47x)
   * - 151936
     - 32
     - 70.8 → 15.2 (4.66x)
     - 96.9 → 20.2 (4.80x)
     - 79.7 → 22.9 (3.48x)
     - 113.2 → 29.2 (3.88x)
   * - 151936
     - 64
     - 72.9 → 18.3 (3.98x)
     - 106.8 → 27.1 (3.94x)
     - 80.8 → 25.9 (3.12x)
     - 110.9 → 36.5 (3.04x)
   * - 151936
     - 128
     - 73.3 → 23.9 (3.07x)
     - 160.8 → 35.4 (4.54x)
     - 87.7 → 31.3 (2.80x)
     - 170.5 → 45.3 (3.76x)
   * - 262144
     - 1
     - 70.2 → 12.5 (5.62x)
     - 79.9 → 18.1 (4.41x)
     - 74.2 → 20.2 (3.67x)
     - 75.9 → 27.4 (2.77x)
   * - 262144
     - 8
     - 70.8 → 13.4 (5.28x)
     - 103.4 → 19.0 (5.44x)
     - 77.5 → 20.9 (3.71x)
     - 112.5 → 28.5 (3.95x)
   * - 262144
     - 16
     - 70.5 → 14.6 (4.83x)
     - 117.4 → 20.1 (5.84x)
     - 76.2 → 21.8 (3.50x)
     - 117.6 → 29.1 (4.04x)
   * - 262144
     - 32
     - 70.9 → 16.9 (4.20x)
     - 135.2 → 23.7 (5.70x)
     - 79.8 → 24.6 (3.24x)
     - 149.8 → 33.3 (4.50x)
   * - 262144
     - 64
     - 88.7 → 22.6 (3.92x)
     - 197.0 → 28.0 (7.04x)
     - 110.6 → 30.0 (3.69x)
     - 201.9 → 37.4 (5.40x)
   * - 262144
     - 128
     - 106.1 → 33.4 (3.18x)
     - 294.8 → 40.7 (7.24x)
     - 122.6 → 40.7 (3.01x)
     - 328.7 → 49.8 (6.60x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
