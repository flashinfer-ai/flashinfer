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
   ``fused_tail``); the ``(8, 48)`` resident, a large-k pick, is built without it.  In the
   two-launch form the dispatcher also decides where the dependent kernel lands: stage 1 signals
   ``griddepcontrol.launch_dependents`` before its first pass only when the batch fits on the
   SMs its last wave leaves free (one stage-1 CTA per SM), so the stage-2/3 CTAs are never
   packed onto the few SMs free mid-flight; larger batches let the dependent launch as stage 1
   exits.  A streaming variant signals at that early point only on Blackwell and Rubin
   (compute capability 10.x); on Hopper it signals after its filter pass, once the whole row
   has been read, where the round-3 kernels did.  All three decisions travel in the stage-1
   ``launch_flags`` argument (bit 0 fused tail, bit 1 early trigger, bit 2 stream pre-pass
   point) and none changes any output.

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
a per-CTA candidate list and finishes on a gathered copy of that list with local passes), so
there is no size-based fallback.  :func:`cake_sampling_route` reports the decision without launching.

The checked-in source product lives in ``csrc/cake_sampling/generated/`` as one translation
unit plus a manifest that records every frozen variant's launch resources; FlashInfer verifies
the source hash and compiles it once into a single fatbin with one ``-gencode`` per build target
(the kernels use only clusters, distributed shared memory, programmatic dependent launch and
``redux.sync``, so no per-architecture source exists).  Stage-1 variants whose dynamic shared
memory exceeds the device's opt-in limit are not dispatch candidates: on 12.x (99 KB) the
streaming variants drop out, so vocabularies above the register-resident capacity (196608)
take the ``top_k_first`` route there.

The stage-1 variant is chosen per call by a cost model whose single-wave CTA capacity table and
cost constants are keyed by the device's SM count (148 for B200 / B300, 132 for H100, 212 for
Rubin R200; other devices use the nearest measured table); the constants were fitted per table on
per-variant sweeps of all four architectures so that every measured cell picks its fastest frozen
variant.

Measured performance
--------------------

``python benchmarks/bench_cake_sampling.py --cupti --skip-joint --batches 1,2,4,8,16,32,64,128
--vocabs 32768,128256,151936,262144`` with ``--top-k 10`` / ``50`` / ``1000`` and with ``--cuda-graph``
(``flashinfer.testing.bench_gpu_time``, CUPTI kernel time, p = 0.9, median µs), round-4 bundle
(streaming stage 1 with a gathered candidate list, fused stage 2/3 for ``top_k_max <= 64``, per-table
k-aware dispatch) on the four supported architectures.  The summary compares every cell with
FlashInfer's default ``top_k_first`` route and with the previous frozen bundle (#5607) measured in
the same run.

.. list-table:: Round-4 bundle, 192 cells per architecture (8 batches × 4 vocabularies × 3 k × eager/graph)
   :header-rows: 1
   :widths: 18 14 20 24 24

   * - GPU
     - cells slower than #5607 by > 2 %
     - speedup vs top_k_first (min / median / max)
     - gain vs #5607, B ≤ 16 (min / median / max)
     - gain vs #5607, B ≥ 32 (min / median)
   * - H100 SXM (sm_90a, 132 SMs)
     - 4 of 192
     - 1.61x / 2.97x / 7.78x
     - 0.96x / 1.09x / 1.52x
     - 0.97x / 1.14x
   * - B200 (sm_100a, 148 SMs)
     - 1 of 192
     - 1.62x / 3.26x / 7.07x
     - 0.98x / 1.10x / 1.53x
     - 1.01x / 1.12x
   * - GB300 (sm_103a, 152 SMs)
     - 3 of 192
     - 1.45x / 3.32x / 12.16x
     - 0.95x / 1.13x / 2.04x
     - 1.02x / 1.15x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 0 of 192
     - 1.48x / 3.19x / 5.45x
     - 0.99x / 1.08x / 1.45x
     - 0.98x / 1.08x

Cells that gain less than 15 % over #5585 are the ones that run the unchanged streaming stage-1
template (``V = 262144`` at every batch, and ``V = 128256 / 151936`` at ``B = 16`` on H100 and GB300)
and the ``k = 1000`` rows at ``V = 151936, B ≤ 8``, where the 1024-entry slab sort already dominated
before this round; every other B ≤ 16 cell gains 15-75 %.

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
     - 45.9 → 11.0 (4.17x)
     - 46.0 → 16.4 (2.80x)
     - 43.1 → 21.7 (1.99x)
     - 45.0 → 28.0 (1.61x)
   * - 32768
     - 8
     - 50.3 → 11.5 (4.37x)
     - 50.1 → 16.7 (3.00x)
     - 54.3 → 22.5 (2.41x)
     - 53.3 → 28.9 (1.84x)
   * - 32768
     - 16
     - 53.3 → 12.3 (4.33x)
     - 53.1 → 17.4 (3.05x)
     - 50.1 → 23.2 (2.16x)
     - 50.7 → 29.2 (1.74x)
   * - 32768
     - 32
     - 54.7 → 14.8 (3.70x)
     - 55.2 → 20.8 (2.65x)
     - 58.2 → 25.8 (2.26x)
     - 58.8 → 33.3 (1.77x)
   * - 32768
     - 64
     - 56.0 → 15.1 (3.71x)
     - 56.1 → 22.8 (2.46x)
     - 59.8 → 25.8 (2.32x)
     - 59.3 → 36.1 (1.64x)
   * - 32768
     - 128
     - 58.4 → 17.7 (3.30x)
     - 58.8 → 27.5 (2.14x)
     - 68.4 → 28.8 (2.38x)
     - 66.4 → 40.1 (1.66x)
   * - 128256
     - 1
     - 112.7 → 15.3 (7.37x)
     - 65.7 → 24.7 (2.66x)
     - 81.8 → 26.3 (3.11x)
     - 71.9 → 36.6 (1.96x)
   * - 128256
     - 8
     - 113.5 → 16.8 (6.76x)
     - 81.5 → 25.2 (3.23x)
     - 86.2 → 28.0 (3.08x)
     - 97.3 → 37.7 (2.58x)
   * - 128256
     - 16
     - 112.7 → 19.6 (5.75x)
     - 94.4 → 29.1 (3.24x)
     - 85.3 → 30.9 (2.76x)
     - 117.3 → 42.0 (2.79x)
   * - 128256
     - 32
     - 116.5 → 26.1 (4.46x)
     - 106.9 → 34.7 (3.08x)
     - 90.5 → 37.5 (2.41x)
     - 120.8 → 47.5 (2.54x)
   * - 128256
     - 64
     - 116.4 → 31.7 (3.67x)
     - 168.6 → 40.4 (4.17x)
     - 104.2 → 43.4 (2.40x)
     - 176.7 → 53.7 (3.29x)
   * - 128256
     - 128
     - 116.5 → 49.3 (2.36x)
     - 236.5 → 58.7 (4.03x)
     - 126.3 → 60.7 (2.08x)
     - 244.7 → 71.3 (3.43x)
   * - 151936
     - 1
     - 113.2 → 17.9 (6.32x)
     - 74.3 → 33.8 (2.20x)
     - 87.1 → 29.5 (2.95x)
     - 80.3 → 47.1 (1.70x)
   * - 151936
     - 8
     - 113.7 → 19.1 (5.95x)
     - 92.0 → 33.0 (2.79x)
     - 92.0 → 30.8 (2.99x)
     - 99.6 → 46.1 (2.16x)
   * - 151936
     - 16
     - 113.9 → 21.2 (5.37x)
     - 106.1 → 32.9 (3.22x)
     - 90.4 → 33.1 (2.73x)
     - 109.8 → 47.2 (2.33x)
   * - 151936
     - 32
     - 113.9 → 29.8 (3.82x)
     - 124.2 → 40.6 (3.06x)
     - 95.1 → 41.7 (2.28x)
     - 130.2 → 54.3 (2.40x)
   * - 151936
     - 64
     - 116.4 → 36.7 (3.17x)
     - 195.4 → 47.1 (4.15x)
     - 116.8 → 48.5 (2.41x)
     - 198.3 → 61.2 (3.24x)
   * - 151936
     - 128
     - 156.0 → 56.6 (2.76x)
     - 279.6 → 65.7 (4.26x)
     - 178.1 → 68.2 (2.61x)
     - 292.8 → 78.5 (3.73x)
   * - 262144
     - 1
     - 113.3 → 19.6 (5.78x)
     - 93.6 → 26.4 (3.55x)
     - 88.3 → 31.2 (2.83x)
     - 94.9 → 41.6 (2.28x)
   * - 262144
     - 8
     - 114.0 → 21.4 (5.33x)
     - 120.3 → 27.6 (4.36x)
     - 93.1 → 33.5 (2.78x)
     - 128.1 → 43.3 (2.96x)
   * - 262144
     - 16
     - 113.6 → 27.2 (4.18x)
     - 148.8 → 33.1 (4.50x)
     - 94.6 → 39.4 (2.40x)
     - 148.6 → 48.1 (3.09x)
   * - 262144
     - 32
     - 126.8 → 42.2 (3.00x)
     - 234.3 → 47.6 (4.92x)
     - 141.7 → 54.1 (2.62x)
     - 233.8 → 61.5 (3.80x)
   * - 262144
     - 64
     - 134.2 → 52.9 (2.54x)
     - 316.8 → 58.7 (5.40x)
     - 152.5 → 65.9 (2.31x)
     - 310.1 → 73.3 (4.23x)
   * - 262144
     - 128
     - 156.9 → 83.9 (1.87x)
     - 471.3 → 91.6 (5.15x)
     - 178.7 → 95.6 (1.87x)
     - 482.2 → 105.3 (4.58x)

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
     - 42.7 → 10.6 (4.03x)
     - 43.1 → 15.8 (2.73x)
     - 39.6 → 20.2 (1.96x)
     - 45.8 → 26.5 (1.73x)
   * - 32768
     - 8
     - 46.7 → 11.2 (4.17x)
     - 48.0 → 16.3 (2.94x)
     - 45.9 → 21.2 (2.17x)
     - 47.2 → 27.3 (1.73x)
   * - 32768
     - 16
     - 50.5 → 11.5 (4.39x)
     - 51.0 → 16.4 (3.11x)
     - 50.7 → 21.2 (2.39x)
     - 47.8 → 27.9 (1.71x)
   * - 32768
     - 32
     - 51.6 → 12.3 (4.20x)
     - 52.7 → 17.4 (3.03x)
     - 46.6 → 21.9 (2.13x)
     - 56.1 → 28.9 (1.94x)
   * - 32768
     - 64
     - 52.9 → 13.8 (3.83x)
     - 53.8 → 19.9 (2.70x)
     - 49.4 → 23.7 (2.08x)
     - 54.8 → 31.6 (1.73x)
   * - 32768
     - 128
     - 53.9 → 14.7 (3.67x)
     - 55.2 → 22.5 (2.45x)
     - 55.2 → 24.2 (2.28x)
     - 59.1 → 34.8 (1.70x)
   * - 128256
     - 1
     - 103.2 → 14.8 (6.97x)
     - 64.0 → 19.7 (3.25x)
     - 77.0 → 24.9 (3.09x)
     - 87.3 → 31.1 (2.81x)
   * - 128256
     - 8
     - 103.3 → 15.3 (6.75x)
     - 80.5 → 20.9 (3.85x)
     - 81.3 → 25.5 (3.19x)
     - 85.2 → 32.4 (2.63x)
   * - 128256
     - 16
     - 102.8 → 17.4 (5.91x)
     - 90.8 → 25.0 (3.63x)
     - 83.7 → 27.4 (3.05x)
     - 118.0 → 36.3 (3.25x)
   * - 128256
     - 32
     - 106.1 → 18.6 (5.70x)
     - 99.5 → 26.3 (3.78x)
     - 84.0 → 28.9 (2.91x)
     - 99.1 → 38.9 (2.55x)
   * - 128256
     - 64
     - 106.3 → 25.0 (4.25x)
     - 143.9 → 32.7 (4.40x)
     - 89.9 → 35.3 (2.55x)
     - 159.1 → 44.4 (3.58x)
   * - 128256
     - 128
     - 107.2 → 37.1 (2.89x)
     - 192.7 → 45.0 (4.28x)
     - 94.2 → 47.1 (2.00x)
     - 197.0 → 56.8 (3.47x)
   * - 151936
     - 1
     - 103.0 → 16.3 (6.32x)
     - 72.7 → 22.4 (3.25x)
     - 80.6 → 27.2 (2.96x)
     - 96.9 → 34.0 (2.85x)
   * - 151936
     - 8
     - 104.7 → 17.0 (6.16x)
     - 90.9 → 23.7 (3.84x)
     - 89.2 → 27.3 (3.27x)
     - 115.5 → 36.1 (3.20x)
   * - 151936
     - 16
     - 104.3 → 19.0 (5.49x)
     - 101.5 → 24.8 (4.09x)
     - 90.0 → 29.4 (3.06x)
     - 129.8 → 37.5 (3.46x)
   * - 151936
     - 32
     - 105.1 → 20.3 (5.18x)
     - 112.6 → 26.1 (4.31x)
     - 86.4 → 30.7 (2.81x)
     - 143.2 → 38.2 (3.75x)
   * - 151936
     - 64
     - 107.3 → 28.7 (3.74x)
     - 163.7 → 35.5 (4.61x)
     - 94.0 → 39.6 (2.37x)
     - 187.5 → 48.0 (3.91x)
   * - 151936
     - 128
     - 107.6 → 43.4 (2.48x)
     - 222.4 → 50.0 (4.45x)
     - 103.8 → 53.4 (1.94x)
     - 228.7 → 62.8 (3.64x)
   * - 262144
     - 1
     - 104.4 → 17.9 (5.83x)
     - 89.1 → 25.6 (3.48x)
     - 83.2 → 28.7 (2.90x)
     - 124.1 → 38.1 (3.26x)
   * - 262144
     - 8
     - 105.9 → 18.8 (5.63x)
     - 123.5 → 25.4 (4.86x)
     - 90.6 → 29.6 (3.06x)
     - 162.5 → 38.1 (4.27x)
   * - 262144
     - 16
     - 106.3 → 24.0 (4.43x)
     - 141.3 → 30.2 (4.68x)
     - 92.4 → 34.6 (2.67x)
     - 244.5 → 42.1 (5.81x)
   * - 262144
     - 32
     - 113.8 → 25.6 (4.45x)
     - 200.6 → 31.4 (6.39x)
     - 131.0 → 36.2 (3.62x)
     - 200.3 → 44.6 (4.49x)
   * - 262144
     - 64
     - 109.3 → 39.4 (2.77x)
     - 265.9 → 44.7 (5.95x)
     - 125.4 → 50.3 (2.49x)
     - 247.6 → 58.3 (4.25x)
   * - 262144
     - 128
     - 124.5 → 65.4 (1.90x)
     - 376.8 → 72.1 (5.23x)
     - 140.7 → 75.7 (1.86x)
     - 394.4 → 84.6 (4.66x)

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
     - 67.8 → 10.9 (6.22x)
     - 69.1 → 24.2 (2.86x)
     - 45.7 → 22.9 (2.00x)
     - 43.8 → 30.2 (1.45x)
   * - 32768
     - 8
     - 70.9 → 11.6 (6.11x)
     - 71.0 → 23.1 (3.07x)
     - 46.2 → 22.8 (2.03x)
     - 46.3 → 29.9 (1.55x)
   * - 32768
     - 16
     - 72.8 → 11.7 (6.22x)
     - 73.0 → 23.3 (3.13x)
     - 52.7 → 23.5 (2.24x)
     - 52.1 → 30.6 (1.70x)
   * - 32768
     - 32
     - 75.0 → 12.4 (6.05x)
     - 84.2 → 22.0 (3.83x)
     - 50.7 → 24.1 (2.10x)
     - 54.9 → 32.1 (1.71x)
   * - 32768
     - 64
     - 76.8 → 14.2 (5.41x)
     - 75.3 → 22.0 (3.42x)
     - 53.1 → 25.7 (2.07x)
     - 55.1 → 35.2 (1.57x)
   * - 32768
     - 128
     - 77.2 → 14.8 (5.22x)
     - 77.1 → 23.6 (3.27x)
     - 60.7 → 26.1 (2.33x)
     - 60.1 → 36.4 (1.65x)
   * - 128256
     - 1
     - 175.8 → 14.9 (11.80x)
     - 81.3 → 21.8 (3.73x)
     - 75.9 → 26.6 (2.85x)
     - 93.4 → 34.1 (2.74x)
   * - 128256
     - 8
     - 171.3 → 15.5 (11.05x)
     - 97.0 → 22.3 (4.35x)
     - 81.9 → 27.1 (3.02x)
     - 97.4 → 34.3 (2.84x)
   * - 128256
     - 16
     - 171.5 → 17.6 (9.74x)
     - 103.0 → 26.6 (3.87x)
     - 83.7 → 29.3 (2.86x)
     - 105.8 → 39.7 (2.66x)
   * - 128256
     - 32
     - 174.9 → 18.5 (9.45x)
     - 109.6 → 27.3 (4.01x)
     - 85.4 → 30.1 (2.84x)
     - 106.3 → 41.2 (2.58x)
   * - 128256
     - 64
     - 176.3 → 24.5 (7.20x)
     - 137.7 → 33.2 (4.15x)
     - 87.3 → 36.1 (2.42x)
     - 138.6 → 47.1 (2.94x)
   * - 128256
     - 128
     - 178.1 → 36.0 (4.95x)
     - 184.9 → 45.0 (4.11x)
     - 93.9 → 47.3 (1.99x)
     - 195.6 → 57.9 (3.38x)
   * - 151936
     - 1
     - 179.7 → 16.4 (10.96x)
     - 87.9 → 24.0 (3.66x)
     - 78.0 → 28.1 (2.78x)
     - 76.3 → 38.8 (1.97x)
   * - 151936
     - 8
     - 173.9 → 17.0 (10.23x)
     - 102.3 → 24.8 (4.12x)
     - 86.2 → 28.9 (2.98x)
     - 104.3 → 38.4 (2.72x)
   * - 151936
     - 16
     - 178.4 → 19.1 (9.34x)
     - 110.7 → 26.4 (4.19x)
     - 88.3 → 31.4 (2.81x)
     - 115.5 → 39.7 (2.91x)
   * - 151936
     - 32
     - 180.2 → 20.0 (9.01x)
     - 120.7 → 26.8 (4.50x)
     - 89.9 → 32.0 (2.81x)
     - 98.6 → 41.2 (2.39x)
   * - 151936
     - 64
     - 180.4 → 28.2 (6.40x)
     - 157.0 → 34.7 (4.52x)
     - 91.0 → 40.1 (2.27x)
     - 173.9 → 49.0 (3.55x)
   * - 151936
     - 128
     - 174.8 → 41.8 (4.18x)
     - 213.4 → 49.6 (4.30x)
     - 103.2 → 53.8 (1.92x)
     - 225.3 → 63.0 (3.58x)
   * - 262144
     - 1
     - 173.2 → 17.9 (9.68x)
     - 100.9 → 27.3 (3.70x)
     - 82.4 → 30.3 (2.72x)
     - 91.7 → 42.1 (2.18x)
   * - 262144
     - 8
     - 172.9 → 18.9 (9.15x)
     - 131.7 → 27.3 (4.82x)
     - 87.9 → 31.2 (2.82x)
     - 131.5 → 41.2 (3.19x)
   * - 262144
     - 16
     - 172.4 → 23.8 (7.24x)
     - 145.7 → 31.1 (4.68x)
     - 89.6 → 36.4 (2.46x)
     - 128.9 → 45.5 (2.83x)
   * - 262144
     - 32
     - 172.1 → 25.1 (6.86x)
     - 188.7 → 32.0 (5.90x)
     - 131.0 → 37.4 (3.50x)
     - 223.0 → 45.9 (4.86x)
   * - 262144
     - 64
     - 174.2 → 38.3 (4.55x)
     - 254.4 → 45.2 (5.63x)
     - 120.5 → 50.8 (2.37x)
     - 252.3 → 59.0 (4.28x)
   * - 262144
     - 128
     - 177.8 → 63.3 (2.81x)
     - 361.2 → 71.2 (5.07x)
     - 145.7 → 75.4 (1.93x)
     - 366.0 → 84.9 (4.31x)

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
     - 30.4 → 9.5 (3.20x)
     - 30.1 → 13.0 (2.32x)
     - 32.2 → 16.8 (1.92x)
     - 36.0 → 21.6 (1.67x)
   * - 32768
     - 8
     - 34.0 → 10.2 (3.33x)
     - 33.1 → 13.8 (2.40x)
     - 37.3 → 17.9 (2.08x)
     - 37.8 → 23.0 (1.64x)
   * - 32768
     - 16
     - 35.5 → 10.8 (3.29x)
     - 35.6 → 14.0 (2.54x)
     - 38.0 → 18.4 (2.07x)
     - 38.4 → 23.2 (1.66x)
   * - 32768
     - 32
     - 36.9 → 11.3 (3.27x)
     - 36.9 → 14.3 (2.58x)
     - 41.2 → 18.6 (2.22x)
     - 41.6 → 23.0 (1.81x)
   * - 32768
     - 64
     - 37.8 → 12.7 (2.98x)
     - 38.1 → 16.0 (2.38x)
     - 43.0 → 20.3 (2.12x)
     - 44.3 → 24.5 (1.81x)
   * - 32768
     - 128
     - 39.2 → 14.0 (2.80x)
     - 39.3 → 18.9 (2.08x)
     - 45.9 → 21.0 (2.19x)
     - 46.6 → 28.3 (1.65x)
   * - 128256
     - 1
     - 70.3 → 13.0 (5.41x)
     - 56.7 → 16.1 (3.52x)
     - 71.2 → 20.6 (3.46x)
     - 67.9 → 25.8 (2.63x)
   * - 128256
     - 8
     - 70.3 → 13.9 (5.06x)
     - 67.6 → 17.4 (3.89x)
     - 74.7 → 21.6 (3.46x)
     - 84.8 → 25.9 (3.27x)
   * - 128256
     - 16
     - 70.3 → 14.9 (4.72x)
     - 75.0 → 18.0 (4.17x)
     - 73.1 → 22.5 (3.25x)
     - 83.3 → 27.3 (3.05x)
   * - 128256
     - 32
     - 72.7 → 17.1 (4.25x)
     - 84.2 → 22.2 (3.79x)
     - 75.9 → 24.4 (3.11x)
     - 93.3 → 31.6 (2.95x)
   * - 128256
     - 64
     - 72.4 → 22.7 (3.19x)
     - 92.5 → 27.5 (3.36x)
     - 77.0 → 30.2 (2.55x)
     - 103.2 → 36.5 (2.83x)
   * - 128256
     - 128
     - 72.5 → 33.8 (2.14x)
     - 137.8 → 38.1 (3.62x)
     - 80.7 → 41.2 (1.96x)
     - 137.7 → 47.4 (2.91x)
   * - 151936
     - 1
     - 69.8 → 15.0 (4.65x)
     - 64.5 → 18.2 (3.54x)
     - 71.0 → 22.5 (3.16x)
     - 75.7 → 27.2 (2.78x)
   * - 151936
     - 8
     - 69.9 → 15.8 (4.42x)
     - 77.2 → 19.3 (4.00x)
     - 78.3 → 23.2 (3.38x)
     - 84.0 → 28.0 (3.00x)
   * - 151936
     - 16
     - 70.0 → 16.6 (4.22x)
     - 86.9 → 20.2 (4.30x)
     - 78.9 → 24.0 (3.29x)
     - 88.1 → 29.6 (2.98x)
   * - 151936
     - 32
     - 70.2 → 18.6 (3.77x)
     - 96.2 → 22.1 (4.35x)
     - 80.8 → 26.2 (3.08x)
     - 87.6 → 30.6 (2.86x)
   * - 151936
     - 64
     - 72.5 → 25.8 (2.81x)
     - 106.4 → 30.6 (3.48x)
     - 79.7 → 33.4 (2.39x)
     - 115.8 → 39.4 (2.94x)
   * - 151936
     - 128
     - 73.7 → 38.9 (1.89x)
     - 161.5 → 42.8 (3.77x)
     - 91.0 → 45.9 (1.98x)
     - 177.3 → 51.9 (3.42x)
   * - 262144
     - 1
     - 69.8 → 16.6 (4.20x)
     - 79.3 → 20.7 (3.83x)
     - 76.2 → 24.3 (3.14x)
     - 96.0 → 30.8 (3.12x)
   * - 262144
     - 8
     - 70.0 → 17.6 (3.98x)
     - 103.1 → 21.6 (4.77x)
     - 79.2 → 24.9 (3.18x)
     - 127.6 → 31.0 (4.12x)
   * - 262144
     - 16
     - 70.0 → 18.7 (3.74x)
     - 119.5 → 22.4 (5.33x)
     - 79.1 → 25.8 (3.07x)
     - 121.7 → 31.7 (3.84x)
   * - 262144
     - 32
     - 70.3 → 23.1 (3.04x)
     - 133.9 → 26.5 (5.05x)
     - 80.5 → 30.6 (2.63x)
     - 137.8 → 35.9 (3.84x)
   * - 262144
     - 64
     - 87.7 → 34.8 (2.52x)
     - 197.7 → 37.9 (5.22x)
     - 106.4 → 42.0 (2.53x)
     - 213.2 → 46.8 (4.56x)
   * - 262144
     - 128
     - 107.5 → 57.8 (1.86x)
     - 294.9 → 60.8 (4.85x)
     - 125.4 → 65.2 (1.92x)
     - 283.6 → 70.1 (4.05x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
