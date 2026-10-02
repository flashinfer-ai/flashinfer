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
   H100, B200 and GB300 (compute capabilities 9.0 / 10.0 / 10.3).  Measured under CUDA-graph replay (the eager
   span of the two-launch chain includes the host gap between its launches, which made the twin
   look 10-50 % faster everywhere): the in-CTA tail loses to the chain on every resident (B200
   1.20-1.54x, GB300 1.40-1.84x of the chain's time) and on every stream at k <= 750 (1.0-1.26x at
   k = 200, 1.04-1.15x at k = 500, 0.94-1.05x at k = 750); the cluster-8 streams at k = 1000 win on
   both (B200 0.89-0.94, GB300 0.89-0.97).  Round 8 measured H100 the same way: the twin runs
   0.93-0.97 of the chain at V = 151936 and 0.96-0.99 at V = 262144 (B = 1-8, k = 800 / 1000), so
   H100 joins; R200 keeps the chain (no graph-replay A/B recorded for it).
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
   sample, bit 7 slab tail, bit 8 pushed coarse sums) and none changes any output.  Bit 4 selects a
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
   exclusive.  Bit 7 (round 8, lever L-B) selects, together with bit 0 and bit 4 or 6, the slab-tail
   form of a streaming variant's coarse- or speculative-sample build (manifest entries with
   ``slab_tail``, symbol suffix ``_cs_lb`` / ``_sp_lb``): the selected (key, index) pairs of a fused
   launch are pushed into rank 0's shared-memory slab with distributed-shared-memory stores and the
   two-warp tail reads them there instead of re-reading the global slab row, so the slab row,
   samples, renorm and count are bit-identical.  The default and whole-CTA-tail builds never carry
   it, so the k > 64 chains run binaries without the slab code (compiled in, it moved them by
   1-14 %).  The host sets bit 7 on compute capability 10.0 / 10.3 for rows of at most 5 register
   chunks per CTA, where the fused cells run 1-7 % faster with it in paired perturbed-process A/Bs
   (medians over four fresh processes) and in both orders of the round-8 matrices; Hopper measured
   its single-row cluster-8 cells 2-5 % slower and keeps the plain sample builds, and the long
   cluster-1 / cluster-2 rows (8-16 chunks per CTA: B = 64-128 at V >= 128256) measured 0-2 % slower
   with it on B200 / GB300 and keep them too.  Bit 8
   (round 8, lever L-G(d)) selects, on a two-launch chain without bits 0, 3, 4, 6 and 7, the
   pushed-coarse-sums form of a streaming variant's default build (manifest entries with
   ``coarse_push``, symbol suffix ``_lg``; the variants whose cluster runs the two-level select):
   each CTA stores its 128 coarse histogram sums into every CTA's shared memory as it builds them,
   so the cluster-wide lower-bucket select reads its peers' sums locally instead of over
   distributed shared memory inside the select loop; every output is bit-identical.  The host sets
   bit 8 on compute capability 9.0 / 10.3 for the ept-32 build, where the two-chunk cluster-8
   k = 1000 chains run 0.6-1.5 % faster with it (medians over four perturbed processes); B200
   measured them 0.9-1.7 % slower and keeps the plain default build, which also serves every fused
   launch, and the four-chunk ept-16 cluster-8 chains (V = 262144, B <= 8) measured 1.4-5.6 % slower
   with it on GB300 and keep it as well.

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
  coarse-sample ``_cs`` and the whole-CTA-tail ``_bt`` builds; the round-8 slab-tail twins ``_cs_lb`` and
  ``_sp_lb``) is additionally gated on bit-identity with the default build on every tested row.  A change that
  moves bits must document exactly which rows can differ (ties, the eps boundary) and why; no round-7 or round-8
  kernel does.
* The exact fallbacks stay kernel-side: a candidate list that overflows the gather capacity takes the three-pass
  cluster path, a row with fewer than k candidates is kept whole, and vocabularies above 2^21 (and compute
  capabilities 12.x) take the reference ``top_k_first`` route.  The sm_120 / sm_121 route semantics and the host API
  are unchanged.

Measured performance
--------------------

``python benchmarks/bench_cake_sampling.py --cupti --skip-joint --batches 1,2,4,8,16,32,64,128
--vocabs 32768,128256,151936,262144`` with ``--top-k 10`` / ``50`` / ``1000`` and with ``--cuda-graph``
(``flashinfer.testing.bench_gpu_time``, CUPTI, p = 0.9, median µs), round-8 bundle (the round-7 kernels plus the
slab-tail twins of the sample builds, the pushed-coarse-sums twin of the cluster-8 default build and the Hopper
whole-CTA tail policy) on the four supported architectures, measured against the previous frozen bundle (#5847, the
round-7 bundle) in the same run on the same node in both tree orders.  The eager value of a single-kernel cell
(B <= 8, k 10 / 50) is the median over ten fresh processes per tree with perturbed allocation histories (the eager
span of such a cell is a deterministic function of the process's allocation history, so one process is one sample);
every row above +0.5 % in both orders was re-measured with an interleaved same-node A/B of the round-7 and round-8
host policies in one process (bitwise-equal outputs), and only a row that A/B confirms counts as a regression.  The
per-cell tables list the round-8 medians (eager small cells: the perturbed-process medians).

.. list-table:: Round-8 bundle, 192 cells per architecture (8 batches × 4 vocabularies × 3 k × eager/graph), both tree orders
   :header-rows: 1
   :widths: 18 16 20 22 22

   * - GPU
     - rows > +0.5 % vs the round-7 bundle (#5847) in both tree orders / confirmed by the same-node interleaved A/B
     - speedup vs top_k_first (min / median / max)
     - round-7 time / round-8 time, B ≤ 16 (min / median / max)
     - round-7 time / round-8 time, B ≥ 32 (min / median / max)
   * - H100 SXM (sm_90a, 132 SMs)
     - 0 of 192 / 0
     - 1.72x / 3.57x / 8.65x
     - 1.00x / 1.00x / 1.15x
     - 1.00x / 1.00x / 1.01x
   * - B200 (sm_100a, 148 SMs)
     - 0 of 192 / 0
     - 1.71x / 3.99x / 9.49x
     - 1.00x / 1.02x / 1.13x
     - 0.91x / 1.02x / 1.04x
   * - GB300 (sm_103a, 152 SMs)
     - 22 of 192 / 0
     - 1.63x / 4.14x / 16.98x
     - 0.87x / 1.01x / 1.06x
     - 0.97x / 1.00x / 1.08x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 5 of 192 / 0
     - 1.48x / 3.62x / 7.74x
     - 0.99x / 1.00x / 1.08x
     - 0.99x / 1.00x / 1.02x

.. list-table:: H100 SXM (sm_90a, 132 SMs), CUPTI median µs, top_k_first → cake_sampling (round-8 bundle)
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
     - 45.3 → 10.3 (4.40x)
     - 46.1 → 14.7 (3.14x)
     - 48.1 → 20.5 (2.35x)
     - 45.0 → 26.1 (1.72x)
   * - 32768
     - 8
     - 48.9 → 10.8 (4.53x)
     - 50.0 → 15.1 (3.31x)
     - 52.3 → 21.2 (2.46x)
     - 52.9 → 26.9 (1.96x)
   * - 32768
     - 16
     - 52.5 → 11.5 (4.57x)
     - 52.9 → 15.8 (3.35x)
     - 51.9 → 21.9 (2.37x)
     - 50.4 → 27.4 (1.84x)
   * - 32768
     - 32
     - 54.0 → 11.7 (4.62x)
     - 55.0 → 17.8 (3.09x)
     - 55.0 → 22.1 (2.49x)
     - 55.8 → 29.6 (1.89x)
   * - 32768
     - 64
     - 55.2 → 13.0 (4.25x)
     - 55.8 → 19.4 (2.88x)
     - 56.2 → 23.1 (2.43x)
     - 59.8 → 30.8 (1.94x)
   * - 32768
     - 128
     - 57.8 → 16.4 (3.52x)
     - 58.5 → 23.7 (2.47x)
     - 63.7 → 26.6 (2.39x)
     - 66.0 → 35.5 (1.86x)
   * - 128256
     - 1
     - 110.4 → 12.8 (8.62x)
     - 66.8 → 19.8 (3.37x)
     - 82.9 → 23.3 (3.56x)
     - 103.1 → 31.6 (3.26x)
   * - 128256
     - 8
     - 111.1 → 14.5 (7.66x)
     - 81.3 → 20.5 (3.97x)
     - 86.0 → 25.1 (3.43x)
     - 91.3 → 32.8 (2.78x)
   * - 128256
     - 16
     - 110.8 → 16.0 (6.92x)
     - 93.5 → 22.2 (4.20x)
     - 87.1 → 26.6 (3.27x)
     - 93.3 → 35.1 (2.66x)
   * - 128256
     - 32
     - 114.3 → 18.7 (6.11x)
     - 107.2 → 29.7 (3.61x)
     - 94.1 → 29.4 (3.20x)
     - 115.3 → 41.3 (2.79x)
   * - 128256
     - 64
     - 114.4 → 25.4 (4.50x)
     - 168.4 → 35.0 (4.81x)
     - 103.4 → 36.4 (2.84x)
     - 176.8 → 47.6 (3.71x)
   * - 128256
     - 128
     - 114.8 → 41.8 (2.75x)
     - 237.7 → 46.8 (5.08x)
     - 120.3 → 52.4 (2.30x)
     - 245.8 → 59.1 (4.16x)
   * - 151936
     - 1
     - 111.6 → 13.4 (8.33x)
     - 75.1 → 18.6 (4.04x)
     - 87.9 → 24.1 (3.65x)
     - 79.6 → 29.7 (2.68x)
   * - 151936
     - 8
     - 112.0 → 15.0 (7.47x)
     - 92.3 → 20.2 (4.57x)
     - 91.6 → 25.9 (3.54x)
     - 101.6 → 31.3 (3.25x)
   * - 151936
     - 16
     - 112.1 → 16.6 (6.75x)
     - 106.2 → 23.1 (4.59x)
     - 91.9 → 27.5 (3.34x)
     - 138.2 → 36.5 (3.79x)
   * - 151936
     - 32
     - 110.9 → 21.1 (5.24x)
     - 124.4 → 27.8 (4.47x)
     - 98.3 → 31.9 (3.08x)
     - 118.1 → 39.8 (2.97x)
   * - 151936
     - 64
     - 115.3 → 28.7 (4.02x)
     - 196.3 → 34.9 (5.62x)
     - 115.8 → 39.9 (2.90x)
     - 216.1 → 48.2 (4.48x)
   * - 151936
     - 128
     - 157.1 → 42.7 (3.68x)
     - 279.9 → 52.3 (5.35x)
     - 171.4 → 53.6 (3.20x)
     - 286.5 → 65.5 (4.37x)
   * - 262144
     - 1
     - 111.9 → 14.0 (7.99x)
     - 92.5 → 20.0 (4.62x)
     - 89.6 → 24.9 (3.61x)
     - 92.6 → 31.5 (2.94x)
   * - 262144
     - 8
     - 112.8 → 16.5 (6.84x)
     - 121.3 → 22.3 (5.44x)
     - 92.5 → 27.7 (3.34x)
     - 128.8 → 34.0 (3.78x)
   * - 262144
     - 16
     - 112.4 → 19.9 (5.65x)
     - 149.4 → 26.7 (5.60x)
     - 95.9 → 31.0 (3.09x)
     - 146.1 → 39.8 (3.68x)
   * - 262144
     - 32
     - 128.1 → 28.1 (4.56x)
     - 231.5 → 39.6 (5.85x)
     - 145.2 → 39.9 (3.64x)
     - 245.2 → 52.3 (4.68x)
   * - 262144
     - 64
     - 135.4 → 43.9 (3.08x)
     - 316.4 → 51.9 (6.10x)
     - 151.9 → 55.7 (2.73x)
     - 333.3 → 64.8 (5.14x)
   * - 262144
     - 128
     - 158.3 → 64.0 (2.47x)
     - 471.6 → 71.5 (6.59x)
     - 172.9 → 74.9 (2.31x)
     - 488.6 → 84.6 (5.78x)

.. list-table:: B200 (sm_100a, 148 SMs), CUPTI median µs, top_k_first → cake_sampling (round-8 bundle)
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
     - 44.2 → 9.7 (4.56x)
     - 42.7 → 14.5 (2.94x)
     - 45.5 → 19.1 (2.38x)
     - 52.5 → 25.1 (2.10x)
   * - 32768
     - 8
     - 47.6 → 9.8 (4.86x)
     - 46.3 → 14.6 (3.17x)
     - 46.0 → 19.6 (2.35x)
     - 53.1 → 26.4 (2.01x)
   * - 32768
     - 16
     - 50.1 → 10.1 (4.96x)
     - 49.5 → 15.1 (3.28x)
     - 53.3 → 19.5 (2.73x)
     - 47.1 → 26.3 (1.79x)
   * - 32768
     - 32
     - 51.7 → 10.4 (4.95x)
     - 51.3 → 15.8 (3.25x)
     - 54.0 → 19.8 (2.73x)
     - 56.6 → 27.4 (2.07x)
   * - 32768
     - 64
     - 53.2 → 10.9 (4.88x)
     - 52.5 → 17.1 (3.07x)
     - 55.1 → 20.9 (2.64x)
     - 53.4 → 28.8 (1.86x)
   * - 32768
     - 128
     - 54.4 → 12.1 (4.50x)
     - 53.8 → 18.6 (2.89x)
     - 57.6 → 24.1 (2.39x)
     - 58.2 → 30.8 (1.89x)
   * - 128256
     - 1
     - 103.4 → 11.1 (9.32x)
     - 62.0 → 17.3 (3.58x)
     - 83.1 → 20.9 (3.99x)
     - 80.4 → 28.2 (2.85x)
   * - 128256
     - 8
     - 104.1 → 11.8 (8.82x)
     - 80.7 → 18.0 (4.48x)
     - 82.1 → 21.9 (3.76x)
     - 105.0 → 29.9 (3.51x)
   * - 128256
     - 16
     - 104.7 → 13.2 (7.93x)
     - 90.6 → 20.7 (4.38x)
     - 81.7 → 23.2 (3.52x)
     - 106.4 → 31.2 (3.40x)
   * - 128256
     - 32
     - 106.5 → 14.5 (7.34x)
     - 100.4 → 21.4 (4.68x)
     - 84.3 → 24.8 (3.41x)
     - 118.0 → 33.1 (3.56x)
   * - 128256
     - 64
     - 107.2 → 17.2 (6.23x)
     - 144.0 → 29.2 (4.93x)
     - 88.5 → 27.6 (3.20x)
     - 141.9 → 41.8 (3.40x)
   * - 128256
     - 128
     - 108.2 → 22.8 (4.75x)
     - 195.5 → 36.1 (5.42x)
     - 97.7 → 33.1 (2.95x)
     - 205.8 → 47.9 (4.30x)
   * - 151936
     - 1
     - 104.4 → 12.4 (8.42x)
     - 71.4 → 18.7 (3.82x)
     - 87.4 → 22.4 (3.90x)
     - 88.7 → 28.9 (3.07x)
   * - 151936
     - 8
     - 104.6 → 13.2 (7.92x)
     - 91.6 → 19.6 (4.67x)
     - 89.9 → 23.4 (3.83x)
     - 97.7 → 30.4 (3.22x)
   * - 151936
     - 16
     - 105.9 → 14.6 (7.25x)
     - 100.8 → 21.7 (4.65x)
     - 90.9 → 25.0 (3.64x)
     - 115.1 → 33.4 (3.45x)
   * - 151936
     - 32
     - 106.1 → 16.1 (6.59x)
     - 113.3 → 22.5 (5.04x)
     - 93.1 → 26.9 (3.46x)
     - 138.9 → 34.6 (4.01x)
   * - 151936
     - 64
     - 123.1 → 19.4 (6.35x)
     - 164.8 → 28.1 (5.86x)
     - 97.1 → 30.4 (3.19x)
     - 172.9 → 41.1 (4.21x)
   * - 151936
     - 128
     - 115.1 → 25.9 (4.45x)
     - 223.8 → 42.1 (5.32x)
     - 103.9 → 36.0 (2.88x)
     - 227.5 → 57.0 (3.99x)
   * - 262144
     - 1
     - 117.7 → 12.4 (9.49x)
     - 89.2 → 20.8 (4.30x)
     - 85.6 → 22.8 (3.76x)
     - 114.2 → 30.8 (3.71x)
   * - 262144
     - 8
     - 106.9 → 13.6 (7.86x)
     - 121.7 → 21.2 (5.74x)
     - 91.0 → 24.4 (3.73x)
     - 123.7 → 31.9 (3.88x)
   * - 262144
     - 16
     - 107.1 → 15.6 (6.87x)
     - 141.0 → 23.4 (6.03x)
     - 93.6 → 26.4 (3.55x)
     - 192.1 → 35.2 (5.46x)
   * - 262144
     - 32
     - 114.7 → 18.1 (6.34x)
     - 197.5 → 25.3 (7.81x)
     - 133.0 → 29.0 (4.59x)
     - 179.2 → 38.2 (4.69x)
   * - 262144
     - 64
     - 110.5 → 24.9 (4.45x)
     - 265.9 → 34.2 (7.77x)
     - 124.0 → 36.0 (3.44x)
     - 258.7 → 46.9 (5.52x)
   * - 262144
     - 128
     - 124.6 → 39.8 (3.13x)
     - 378.4 → 50.8 (7.46x)
     - 147.4 → 49.5 (2.97x)
     - 380.1 → 63.5 (5.99x)

.. list-table:: GB300 (sm_103a, 152 SMs), CUPTI median µs, top_k_first → cake_sampling (round-8 bundle)
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
     - 68.7 → 10.0 (6.87x)
     - 70.0 → 20.2 (3.47x)
     - 46.2 → 22.1 (2.09x)
     - 44.0 → 26.9 (1.63x)
   * - 32768
     - 8
     - 72.4 → 10.0 (7.24x)
     - 72.7 → 20.0 (3.63x)
     - 46.5 → 21.1 (2.21x)
     - 54.2 → 27.4 (1.98x)
   * - 32768
     - 16
     - 75.5 → 10.2 (7.40x)
     - 74.1 → 19.5 (3.80x)
     - 47.8 → 21.6 (2.21x)
     - 51.2 → 27.3 (1.88x)
   * - 32768
     - 32
     - 77.3 → 10.5 (7.36x)
     - 78.9 → 19.0 (4.15x)
     - 54.4 → 22.6 (2.40x)
     - 59.9 → 29.4 (2.04x)
   * - 32768
     - 64
     - 78.1 → 11.1 (7.04x)
     - 78.1 → 19.6 (3.99x)
     - 54.0 → 22.8 (2.37x)
     - 56.5 → 30.8 (1.83x)
   * - 32768
     - 128
     - 78.7 → 12.1 (6.50x)
     - 79.2 → 19.0 (4.18x)
     - 56.9 → 24.1 (2.36x)
     - 58.0 → 31.1 (1.87x)
   * - 128256
     - 1
     - 180.0 → 10.9 (16.51x)
     - 83.9 → 19.3 (4.35x)
     - 75.6 → 23.8 (3.18x)
     - 74.4 → 30.0 (2.48x)
   * - 128256
     - 8
     - 189.1 → 11.8 (16.03x)
     - 107.7 → 20.4 (5.28x)
     - 82.0 → 23.4 (3.50x)
     - 89.0 → 30.6 (2.91x)
   * - 128256
     - 16
     - 179.0 → 13.3 (13.46x)
     - 114.3 → 21.4 (5.33x)
     - 81.5 → 25.4 (3.20x)
     - 91.1 → 34.8 (2.62x)
   * - 128256
     - 32
     - 179.0 → 14.5 (12.34x)
     - 115.6 → 21.1 (5.48x)
     - 82.1 → 27.0 (3.04x)
     - 88.1 → 36.0 (2.44x)
   * - 128256
     - 64
     - 184.9 → 17.1 (10.81x)
     - 137.6 → 28.9 (4.75x)
     - 90.5 → 29.2 (3.10x)
     - 153.2 → 44.0 (3.49x)
   * - 128256
     - 128
     - 177.5 → 22.5 (7.89x)
     - 184.3 → 35.3 (5.21x)
     - 94.5 → 34.8 (2.72x)
     - 190.4 → 49.8 (3.83x)
   * - 151936
     - 1
     - 182.8 → 12.1 (15.11x)
     - 88.5 → 20.2 (4.38x)
     - 82.0 → 25.4 (3.23x)
     - 82.5 → 32.8 (2.52x)
   * - 151936
     - 8
     - 189.5 → 13.1 (14.47x)
     - 117.0 → 21.0 (5.57x)
     - 86.9 → 25.8 (3.37x)
     - 98.9 → 33.2 (2.97x)
   * - 151936
     - 16
     - 184.4 → 14.5 (12.72x)
     - 116.8 → 22.5 (5.20x)
     - 92.1 → 27.0 (3.41x)
     - 120.6 → 35.5 (3.40x)
   * - 151936
     - 32
     - 185.8 → 15.8 (11.76x)
     - 126.8 → 22.0 (5.76x)
     - 90.1 → 28.2 (3.20x)
     - 111.9 → 38.0 (2.95x)
   * - 151936
     - 64
     - 180.8 → 19.2 (9.42x)
     - 156.9 → 26.8 (5.85x)
     - 93.7 → 31.4 (2.98x)
     - 167.5 → 40.4 (4.15x)
   * - 151936
     - 128
     - 191.0 → 25.8 (7.40x)
     - 212.4 → 37.0 (5.74x)
     - 103.9 → 38.1 (2.73x)
     - 224.8 → 50.7 (4.43x)
   * - 262144
     - 1
     - 194.8 → 12.3 (15.84x)
     - 108.0 → 21.9 (4.94x)
     - 82.8 → 25.1 (3.29x)
     - 96.9 → 36.2 (2.68x)
   * - 262144
     - 8
     - 182.4 → 13.7 (13.31x)
     - 143.4 → 25.0 (5.74x)
     - 85.7 → 26.3 (3.26x)
     - 142.6 → 35.6 (4.01x)
   * - 262144
     - 16
     - 177.6 → 15.6 (11.38x)
     - 152.9 → 22.9 (6.68x)
     - 92.0 → 27.9 (3.30x)
     - 140.7 → 37.2 (3.78x)
   * - 262144
     - 32
     - 178.7 → 17.8 (10.04x)
     - 189.2 → 24.4 (7.77x)
     - 130.5 → 30.4 (4.30x)
     - 190.3 → 38.6 (4.92x)
   * - 262144
     - 64
     - 182.2 → 25.2 (7.23x)
     - 252.0 → 33.1 (7.61x)
     - 122.5 → 37.5 (3.27x)
     - 273.9 → 48.3 (5.67x)
   * - 262144
     - 128
     - 187.2 → 36.4 (5.14x)
     - 358.8 → 47.9 (7.49x)
     - 144.7 → 49.2 (2.94x)
     - 383.9 → 63.8 (6.01x)

.. list-table:: VR200 R200 (sm_107a, 212 SMs), CUPTI median µs, top_k_first → cake_sampling (round-8 bundle)
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
     - 29.7 → 9.1 (3.26x)
     - 29.8 → 12.6 (2.37x)
     - 38.6 → 18.1 (2.13x)
     - 36.7 → 23.9 (1.54x)
   * - 32768
     - 8
     - 33.2 → 9.9 (3.35x)
     - 32.6 → 13.7 (2.38x)
     - 42.2 → 19.4 (2.18x)
     - 39.9 → 24.4 (1.63x)
   * - 32768
     - 16
     - 35.3 → 10.6 (3.33x)
     - 35.3 → 13.8 (2.56x)
     - 44.1 → 20.0 (2.21x)
     - 40.7 → 24.6 (1.65x)
   * - 32768
     - 32
     - 36.6 → 11.0 (3.33x)
     - 36.8 → 14.0 (2.63x)
     - 47.6 → 20.1 (2.37x)
     - 45.2 → 24.4 (1.85x)
   * - 32768
     - 64
     - 37.6 → 10.9 (3.45x)
     - 37.9 → 15.9 (2.38x)
     - 44.4 → 19.9 (2.23x)
     - 44.7 → 26.2 (1.71x)
   * - 32768
     - 128
     - 39.2 → 11.3 (3.47x)
     - 39.1 → 16.3 (2.40x)
     - 48.4 → 20.1 (2.41x)
     - 55.7 → 27.8 (2.00x)
   * - 128256
     - 1
     - 79.2 → 11.2 (7.07x)
     - 55.6 → 14.8 (3.76x)
     - 74.0 → 20.4 (3.63x)
     - 65.6 → 25.8 (2.54x)
   * - 128256
     - 8
     - 81.1 → 11.9 (6.82x)
     - 67.9 → 16.2 (4.19x)
     - 77.1 → 21.2 (3.63x)
     - 77.9 → 26.4 (2.95x)
   * - 128256
     - 16
     - 80.4 → 12.9 (6.23x)
     - 76.5 → 16.8 (4.55x)
     - 73.6 → 22.1 (3.32x)
     - 86.8 → 27.7 (3.13x)
   * - 128256
     - 32
     - 87.4 → 13.5 (6.47x)
     - 84.1 → 18.5 (4.55x)
     - 80.0 → 22.6 (3.53x)
     - 100.0 → 29.5 (3.39x)
   * - 128256
     - 64
     - 84.7 → 15.6 (5.43x)
     - 92.6 → 24.2 (3.82x)
     - 80.0 → 24.9 (3.22x)
     - 93.8 → 36.0 (2.61x)
   * - 128256
     - 128
     - 85.0 → 19.4 (4.39x)
     - 139.1 → 30.6 (4.54x)
     - 82.0 → 28.4 (2.89x)
     - 144.5 → 42.6 (3.39x)
   * - 151936
     - 1
     - 81.4 → 12.1 (6.73x)
     - 64.4 → 17.1 (3.77x)
     - 73.4 → 21.1 (3.47x)
     - 72.1 → 28.4 (2.54x)
   * - 151936
     - 8
     - 81.2 → 13.0 (6.25x)
     - 77.4 → 17.9 (4.31x)
     - 80.1 → 22.1 (3.62x)
     - 87.0 → 28.4 (3.06x)
   * - 151936
     - 16
     - 80.9 → 14.0 (5.78x)
     - 87.3 → 18.4 (4.74x)
     - 80.1 → 22.9 (3.49x)
     - 87.3 → 29.4 (2.97x)
   * - 151936
     - 32
     - 81.9 → 14.6 (5.61x)
     - 95.8 → 19.4 (4.93x)
     - 85.0 → 24.1 (3.53x)
     - 92.2 → 29.9 (3.08x)
   * - 151936
     - 64
     - 85.2 → 17.2 (4.95x)
     - 106.7 → 22.6 (4.72x)
     - 82.8 → 26.5 (3.12x)
     - 118.2 → 33.0 (3.58x)
   * - 151936
     - 128
     - 86.0 → 22.5 (3.82x)
     - 161.5 → 35.8 (4.50x)
     - 93.9 → 31.4 (2.99x)
     - 182.0 → 48.3 (3.76x)
   * - 262144
     - 1
     - 80.2 → 12.3 (6.52x)
     - 80.3 → 18.7 (4.29x)
     - 78.4 → 21.4 (3.66x)
     - 88.6 → 30.0 (2.95x)
   * - 262144
     - 8
     - 80.6 → 13.2 (6.11x)
     - 103.0 → 19.2 (5.35x)
     - 81.7 → 22.3 (3.66x)
     - 158.9 → 30.4 (5.24x)
   * - 262144
     - 16
     - 80.0 → 14.1 (5.67x)
     - 120.1 → 20.3 (5.92x)
     - 79.7 → 23.1 (3.44x)
     - 115.9 → 31.0 (3.74x)
   * - 262144
     - 32
     - 79.9 → 15.9 (5.03x)
     - 134.9 → 21.2 (6.36x)
     - 84.9 → 25.3 (3.36x)
     - 207.9 → 32.4 (6.42x)
   * - 262144
     - 64
     - 87.9 → 21.1 (4.17x)
     - 196.0 → 28.3 (6.93x)
     - 111.1 → 30.3 (3.67x)
     - 193.5 → 39.3 (4.92x)
   * - 262144
     - 128
     - 107.4 → 28.5 (3.77x)
     - 294.1 → 38.0 (7.74x)
     - 127.0 → 37.5 (3.38x)
     - 307.4 → 49.1 (6.26x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
