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
     - 0 of 192
     - 1.24x / 2.66x / 7.31x
     - 1.03x / 1.26x / 1.75x
     - 1.05x / 1.16x
   * - B200 (sm_100a, 148 SMs)
     - 0 of 192
     - 1.36x / 2.98x / 9.56x
     - 1.07x / 1.32x / 1.71x
     - 1.05x / 1.21x
   * - GB300 (sm_103a, 152 SMs)
     - 0 of 192
     - 1.17x / 3.02x / 9.86x
     - 1.07x / 1.25x / 1.42x
     - 1.06x / 1.17x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 0 of 192
     - 1.54x / 2.89x / 5.42x
     - 1.12x / 1.38x / 1.71x
     - 1.07x / 1.18x

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
     - 45.8 → 12.6 (3.63x)
     - 45.9 → 17.2 (2.67x)
     - 48.4 → 23.0 (2.10x)
     - 43.5 → 27.9 (1.56x)
   * - 32768
     - 8
     - 49.8 → 12.3 (4.05x)
     - 51.6 → 17.0 (3.04x)
     - 49.4 → 23.9 (2.07x)
     - 49.0 → 29.0 (1.69x)
   * - 32768
     - 16
     - 53.0 → 12.6 (4.21x)
     - 53.4 → 17.5 (3.05x)
     - 56.6 → 24.3 (2.33x)
     - 54.8 → 29.2 (1.88x)
   * - 32768
     - 32
     - 54.5 → 14.8 (3.68x)
     - 55.3 → 20.9 (2.65x)
     - 57.1 → 26.6 (2.15x)
     - 55.1 → 33.0 (1.67x)
   * - 32768
     - 64
     - 56.1 → 17.9 (3.13x)
     - 57.3 → 35.7 (1.61x)
     - 57.6 → 29.8 (1.93x)
     - 59.5 → 48.0 (1.24x)
   * - 32768
     - 128
     - 58.2 → 22.8 (2.55x)
     - 58.5 → 37.9 (1.54x)
     - 72.4 → 34.5 (2.10x)
     - 67.0 → 50.0 (1.34x)
   * - 128256
     - 1
     - 112.5 → 15.7 (7.17x)
     - 67.0 → 24.7 (2.71x)
     - 81.2 → 27.7 (2.93x)
     - 64.2 → 37.4 (1.72x)
   * - 128256
     - 8
     - 113.2 → 17.1 (6.62x)
     - 81.4 → 24.3 (3.35x)
     - 86.5 → 29.5 (2.93x)
     - 115.6 → 37.2 (3.11x)
   * - 128256
     - 16
     - 113.0 → 25.8 (4.38x)
     - 94.1 → 30.9 (3.05x)
     - 86.8 → 38.0 (2.28x)
     - 107.8 → 43.3 (2.49x)
   * - 128256
     - 32
     - 116.5 → 30.8 (3.78x)
     - 106.6 → 36.3 (2.94x)
     - 93.3 → 43.1 (2.16x)
     - 116.8 → 48.7 (2.40x)
   * - 128256
     - 64
     - 116.3 → 37.8 (3.08x)
     - 168.6 → 54.7 (3.08x)
     - 101.5 → 50.5 (2.01x)
     - 167.5 → 67.3 (2.49x)
   * - 128256
     - 128
     - 116.9 → 54.3 (2.15x)
     - 236.3 → 69.7 (3.39x)
     - 121.9 → 66.7 (1.83x)
     - 246.5 → 82.7 (2.98x)
   * - 151936
     - 1
     - 113.4 → 22.4 (5.06x)
     - 73.5 → 34.9 (2.11x)
     - 86.1 → 35.3 (2.44x)
     - 70.8 → 46.8 (1.51x)
   * - 151936
     - 8
     - 114.5 → 22.7 (5.04x)
     - 92.0 → 34.1 (2.70x)
     - 91.7 → 36.4 (2.52x)
     - 108.9 → 45.8 (2.38x)
   * - 151936
     - 16
     - 114.2 → 27.3 (4.18x)
     - 105.3 → 32.4 (3.25x)
     - 93.7 → 40.1 (2.34x)
     - 118.8 → 45.4 (2.62x)
   * - 151936
     - 32
     - 114.8 → 34.4 (3.34x)
     - 124.0 → 39.7 (3.12x)
     - 98.2 → 47.0 (2.09x)
     - 116.4 → 53.2 (2.19x)
   * - 151936
     - 64
     - 118.0 → 42.7 (2.76x)
     - 196.4 → 59.5 (3.30x)
     - 114.9 → 55.9 (2.06x)
     - 198.5 → 72.6 (2.73x)
   * - 151936
     - 128
     - 156.6 → 61.4 (2.55x)
     - 280.6 → 76.6 (3.66x)
     - 172.6 → 74.2 (2.33x)
     - 297.4 → 90.3 (3.29x)
   * - 262144
     - 1
     - 114.3 → 29.8 (3.84x)
     - 92.7 → 35.5 (2.61x)
     - 87.9 → 42.5 (2.07x)
     - 78.0 → 49.1 (1.59x)
   * - 262144
     - 8
     - 114.9 → 31.0 (3.71x)
     - 120.8 → 36.9 (3.27x)
     - 93.7 → 44.8 (2.09x)
     - 126.2 → 51.2 (2.46x)
   * - 262144
     - 16
     - 114.7 → 33.0 (3.48x)
     - 147.2 → 38.1 (3.86x)
     - 96.8 → 46.2 (2.10x)
     - 140.4 → 52.1 (2.69x)
   * - 262144
     - 32
     - 127.9 → 46.3 (2.76x)
     - 233.1 → 51.7 (4.51x)
     - 144.4 → 59.6 (2.42x)
     - 213.7 → 66.0 (3.24x)
   * - 262144
     - 64
     - 135.0 → 58.9 (2.29x)
     - 315.7 → 75.7 (4.17x)
     - 150.7 → 73.2 (2.06x)
     - 328.2 → 90.4 (3.63x)
   * - 262144
     - 128
     - 157.9 → 88.7 (1.78x)
     - 470.4 → 104.4 (4.51x)
     - 173.9 → 102.3 (1.70x)
     - 465.6 → 117.9 (3.95x)

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
     - 48.2 → 17.3 (2.79x)
     - 42.4 → 16.8 (2.52x)
     - 44.2 → 22.1 (2.00x)
     - 43.9 → 26.8 (1.64x)
   * - 32768
     - 8
     - 50.8 → 14.8 (3.43x)
     - 46.8 → 16.6 (2.82x)
     - 46.3 → 23.6 (1.96x)
     - 48.8 → 27.8 (1.76x)
   * - 32768
     - 16
     - 53.8 → 14.8 (3.64x)
     - 49.6 → 17.0 (2.92x)
     - 52.7 → 23.6 (2.23x)
     - 46.7 → 28.2 (1.66x)
   * - 32768
     - 32
     - 51.6 → 16.4 (3.15x)
     - 51.5 → 17.4 (2.96x)
     - 54.3 → 23.7 (2.29x)
     - 56.5 → 32.4 (1.74x)
   * - 32768
     - 64
     - 63.6 → 14.7 (4.33x)
     - 57.7 → 28.0 (2.06x)
     - 52.2 → 26.7 (1.96x)
     - 55.9 → 38.8 (1.44x)
   * - 32768
     - 128
     - 57.0 → 17.1 (3.33x)
     - 54.3 → 30.6 (1.77x)
     - 56.7 → 28.5 (1.99x)
     - 57.6 → 42.2 (1.36x)
   * - 128256
     - 1
     - 105.8 → 15.5 (6.83x)
     - 63.0 → 19.9 (3.17x)
     - 84.1 → 27.4 (3.07x)
     - 71.3 → 31.7 (2.25x)
   * - 128256
     - 8
     - 154.9 → 16.2 (9.56x)
     - 80.9 → 20.7 (3.91x)
     - 85.6 → 29.5 (2.90x)
     - 87.4 → 33.0 (2.65x)
   * - 128256
     - 16
     - 106.6 → 22.6 (4.72x)
     - 90.4 → 27.5 (3.29x)
     - 87.6 → 34.4 (2.55x)
     - 110.1 → 39.0 (2.82x)
   * - 128256
     - 32
     - 150.9 → 23.9 (6.31x)
     - 101.1 → 32.8 (3.08x)
     - 89.6 → 35.7 (2.51x)
     - 101.7 → 44.6 (2.28x)
   * - 128256
     - 64
     - 108.7 → 27.9 (3.90x)
     - 143.9 → 41.5 (3.47x)
     - 90.9 → 40.2 (2.26x)
     - 152.4 → 53.0 (2.88x)
   * - 128256
     - 128
     - 128.4 → 38.5 (3.34x)
     - 193.8 → 52.3 (3.71x)
     - 96.6 → 50.0 (1.93x)
     - 224.2 → 63.8 (3.51x)
   * - 151936
     - 1
     - 114.5 → 18.0 (6.36x)
     - 70.5 → 22.4 (3.15x)
     - 86.4 → 29.8 (2.90x)
     - 78.1 → 34.0 (2.30x)
   * - 151936
     - 8
     - 111.6 → 18.8 (5.94x)
     - 92.1 → 23.5 (3.92x)
     - 86.4 → 31.8 (2.72x)
     - 98.1 → 36.4 (2.70x)
   * - 151936
     - 16
     - 127.0 → 24.1 (5.27x)
     - 104.0 → 29.1 (3.57x)
     - 89.9 → 36.7 (2.45x)
     - 121.7 → 41.4 (2.94x)
   * - 151936
     - 32
     - 106.5 → 25.4 (4.19x)
     - 114.4 → 34.3 (3.34x)
     - 89.4 → 37.7 (2.37x)
     - 129.2 → 46.0 (2.81x)
   * - 151936
     - 64
     - 161.2 → 31.6 (5.10x)
     - 163.9 → 44.8 (3.66x)
     - 92.6 → 43.5 (2.13x)
     - 166.9 → 57.2 (2.92x)
   * - 151936
     - 128
     - 120.7 → 44.5 (2.71x)
     - 223.7 → 58.0 (3.86x)
     - 104.7 → 55.9 (1.87x)
     - 236.9 → 70.9 (3.34x)
   * - 262144
     - 1
     - 108.4 → 27.3 (3.97x)
     - 86.9 → 33.6 (2.59x)
     - 88.1 → 39.6 (2.22x)
     - 95.2 → 44.7 (2.13x)
   * - 262144
     - 8
     - 107.8 → 28.2 (3.82x)
     - 122.5 → 33.4 (3.67x)
     - 89.3 → 40.4 (2.21x)
     - 123.1 → 46.3 (2.66x)
   * - 262144
     - 16
     - 116.9 → 29.1 (4.02x)
     - 141.3 → 33.9 (4.17x)
     - 91.1 → 41.3 (2.21x)
     - 126.7 → 45.9 (2.76x)
   * - 262144
     - 32
     - 127.0 → 30.5 (4.16x)
     - 199.4 → 39.4 (5.06x)
     - 133.3 → 42.8 (3.11x)
     - 216.8 → 51.9 (4.18x)
   * - 262144
     - 64
     - 112.4 → 41.7 (2.70x)
     - 266.0 → 54.9 (4.85x)
     - 120.1 → 54.5 (2.20x)
     - 276.8 → 67.6 (4.09x)
   * - 262144
     - 128
     - 124.0 → 65.8 (1.88x)
     - 377.8 → 79.4 (4.76x)
     - 144.0 → 78.2 (1.84x)
     - 379.5 → 91.6 (4.14x)

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
     - 73.2 → 23.2 (3.16x)
     - 71.1 → 25.9 (2.75x)
     - 45.7 → 25.1 (1.82x)
     - 42.9 → 31.1 (1.38x)
   * - 32768
     - 8
     - 75.0 → 21.6 (3.47x)
     - 74.9 → 25.0 (3.00x)
     - 46.3 → 24.4 (1.90x)
     - 50.4 → 30.3 (1.66x)
   * - 32768
     - 16
     - 78.6 → 21.8 (3.61x)
     - 76.7 → 25.2 (3.04x)
     - 49.8 → 25.2 (1.98x)
     - 47.2 → 30.9 (1.53x)
   * - 32768
     - 32
     - 79.9 → 23.4 (3.41x)
     - 78.5 → 25.0 (3.14x)
     - 55.1 → 26.3 (2.10x)
     - 53.7 → 35.9 (1.50x)
   * - 32768
     - 64
     - 78.4 → 20.1 (3.90x)
     - 83.6 → 25.7 (3.25x)
     - 53.0 → 28.4 (1.87x)
     - 52.4 → 42.3 (1.24x)
   * - 32768
     - 128
     - 81.1 → 20.7 (3.92x)
     - 82.2 → 25.8 (3.19x)
     - 56.5 → 29.8 (1.90x)
     - 55.9 → 47.7 (1.17x)
   * - 128256
     - 1
     - 188.4 → 19.7 (9.56x)
     - 85.5 → 24.9 (3.43x)
     - 77.7 → 28.3 (2.75x)
     - 64.1 → 34.7 (1.85x)
   * - 128256
     - 8
     - 186.0 → 21.9 (8.49x)
     - 106.2 → 25.4 (4.18x)
     - 81.9 → 29.1 (2.81x)
     - 99.7 → 35.0 (2.85x)
   * - 128256
     - 16
     - 193.9 → 22.6 (8.58x)
     - 108.9 → 28.5 (3.82x)
     - 83.8 → 35.7 (2.35x)
     - 99.0 → 42.2 (2.35x)
   * - 128256
     - 32
     - 188.7 → 23.6 (8.00x)
     - 117.3 → 33.2 (3.53x)
     - 84.3 → 36.4 (2.32x)
     - 113.5 → 47.1 (2.41x)
   * - 128256
     - 64
     - 188.6 → 27.6 (6.83x)
     - 137.2 → 41.6 (3.30x)
     - 86.3 → 40.9 (2.11x)
     - 149.1 → 55.1 (2.71x)
   * - 128256
     - 128
     - 188.8 → 37.8 (4.99x)
     - 184.1 → 55.8 (3.30x)
     - 88.6 → 51.0 (1.74x)
     - 186.1 → 68.6 (2.71x)
   * - 151936
     - 1
     - 183.1 → 21.6 (8.48x)
     - 94.2 → 25.5 (3.69x)
     - 80.3 → 31.4 (2.56x)
     - 70.8 → 37.2 (1.90x)
   * - 151936
     - 8
     - 189.3 → 20.2 (9.37x)
     - 111.0 → 26.0 (4.27x)
     - 85.1 → 31.9 (2.67x)
     - 95.5 → 38.1 (2.51x)
   * - 151936
     - 16
     - 186.3 → 24.0 (7.76x)
     - 116.7 → 29.9 (3.90x)
     - 85.5 → 37.3 (2.29x)
     - 123.6 → 43.2 (2.86x)
   * - 151936
     - 32
     - 188.7 → 25.1 (7.52x)
     - 124.8 → 34.8 (3.59x)
     - 87.6 → 38.4 (2.28x)
     - 125.3 → 48.4 (2.59x)
   * - 151936
     - 64
     - 188.4 → 30.9 (6.10x)
     - 156.6 → 45.0 (3.48x)
     - 90.0 → 44.1 (2.04x)
     - 146.9 → 58.4 (2.52x)
   * - 151936
     - 128
     - 194.8 → 43.5 (4.48x)
     - 212.9 → 61.1 (3.48x)
     - 97.0 → 57.1 (1.70x)
     - 238.4 → 74.4 (3.20x)
   * - 262144
     - 1
     - 187.7 → 26.8 (7.00x)
     - 110.4 → 33.2 (3.33x)
     - 80.2 → 40.6 (1.98x)
     - 79.6 → 47.6 (1.67x)
   * - 262144
     - 8
     - 204.6 → 28.1 (7.28x)
     - 138.8 → 34.2 (4.06x)
     - 85.7 → 42.0 (2.04x)
     - 121.1 → 47.6 (2.54x)
   * - 262144
     - 16
     - 192.7 → 28.5 (6.76x)
     - 152.4 → 34.5 (4.42x)
     - 87.0 → 41.9 (2.08x)
     - 155.3 → 48.4 (3.21x)
   * - 262144
     - 32
     - 195.2 → 30.0 (6.51x)
     - 188.1 → 39.6 (4.75x)
     - 127.8 → 43.9 (2.91x)
     - 214.3 → 53.9 (3.98x)
   * - 262144
     - 64
     - 194.9 → 40.7 (4.79x)
     - 254.1 → 54.7 (4.65x)
     - 118.0 → 54.7 (2.16x)
     - 262.3 → 70.0 (3.75x)
   * - 262144
     - 128
     - 194.4 → 63.8 (3.05x)
     - 361.1 → 81.5 (4.43x)
     - 138.4 → 77.5 (1.79x)
     - 365.3 → 95.6 (3.82x)

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
     - 30.0 → 9.9 (3.03x)
     - 30.8 → 12.7 (2.43x)
     - 32.5 → 18.5 (1.76x)
     - 35.1 → 21.5 (1.63x)
   * - 32768
     - 8
     - 33.5 → 10.9 (3.07x)
     - 33.6 → 13.7 (2.45x)
     - 40.2 → 20.1 (2.00x)
     - 40.6 → 23.1 (1.76x)
   * - 32768
     - 16
     - 35.6 → 11.0 (3.24x)
     - 35.7 → 13.9 (2.57x)
     - 41.2 → 20.3 (2.03x)
     - 44.9 → 23.1 (1.94x)
   * - 32768
     - 32
     - 36.9 → 11.3 (3.27x)
     - 37.2 → 14.1 (2.64x)
     - 45.0 → 20.2 (2.23x)
     - 44.9 → 23.0 (1.95x)
   * - 32768
     - 64
     - 38.1 → 12.8 (2.98x)
     - 38.5 → 16.0 (2.41x)
     - 44.5 → 21.5 (2.07x)
     - 44.4 → 24.8 (1.79x)
   * - 32768
     - 128
     - 39.3 → 15.4 (2.55x)
     - 39.6 → 21.8 (1.82x)
     - 50.5 → 24.5 (2.06x)
     - 47.7 → 30.9 (1.54x)
   * - 128256
     - 1
     - 69.9 → 13.2 (5.30x)
     - 57.1 → 15.8 (3.61x)
     - 69.6 → 22.2 (3.14x)
     - 66.4 → 25.4 (2.61x)
   * - 128256
     - 8
     - 71.7 → 14.7 (4.88x)
     - 67.2 → 17.3 (3.88x)
     - 73.0 → 23.4 (3.12x)
     - 84.4 → 26.1 (3.23x)
   * - 128256
     - 16
     - 70.8 → 15.1 (4.69x)
     - 76.2 → 17.8 (4.28x)
     - 73.7 → 23.9 (3.08x)
     - 80.7 → 27.2 (2.97x)
   * - 128256
     - 32
     - 73.0 → 21.0 (3.48x)
     - 83.1 → 23.7 (3.51x)
     - 73.0 → 30.5 (2.39x)
     - 109.6 → 33.3 (3.29x)
   * - 128256
     - 64
     - 72.3 → 25.2 (2.87x)
     - 92.6 → 28.5 (3.25x)
     - 74.8 → 34.0 (2.20x)
     - 102.0 → 37.5 (2.72x)
   * - 128256
     - 128
     - 72.3 → 33.5 (2.16x)
     - 137.3 → 40.4 (3.40x)
     - 82.5 → 42.2 (1.95x)
     - 143.4 → 49.6 (2.89x)
   * - 151936
     - 1
     - 70.0 → 15.2 (4.61x)
     - 63.2 → 18.0 (3.51x)
     - 71.5 → 24.7 (2.89x)
     - 73.3 → 27.5 (2.67x)
   * - 151936
     - 8
     - 70.4 → 16.7 (4.22x)
     - 76.9 → 19.3 (3.98x)
     - 75.3 → 25.6 (2.94x)
     - 99.2 → 27.9 (3.56x)
   * - 151936
     - 16
     - 71.0 → 17.5 (4.06x)
     - 87.0 → 20.1 (4.33x)
     - 75.3 → 26.5 (2.84x)
     - 83.5 → 29.6 (2.82x)
   * - 151936
     - 32
     - 70.8 → 22.3 (3.17x)
     - 96.2 → 25.2 (3.82x)
     - 75.9 → 31.7 (2.39x)
     - 105.5 → 34.0 (3.10x)
   * - 151936
     - 64
     - 73.2 → 27.9 (2.62x)
     - 106.1 → 31.3 (3.39x)
     - 76.9 → 36.9 (2.08x)
     - 113.5 → 40.0 (2.84x)
   * - 151936
     - 128
     - 73.3 → 38.4 (1.91x)
     - 160.3 → 45.0 (3.56x)
     - 87.6 → 47.5 (1.84x)
     - 169.3 → 53.6 (3.16x)
   * - 262144
     - 1
     - 70.6 → 23.5 (3.00x)
     - 78.5 → 26.4 (2.97x)
     - 71.8 → 32.9 (2.18x)
     - 91.2 → 36.1 (2.53x)
   * - 262144
     - 8
     - 70.3 → 24.9 (2.82x)
     - 102.7 → 27.7 (3.71x)
     - 76.3 → 33.7 (2.26x)
     - 155.9 → 37.1 (4.20x)
   * - 262144
     - 16
     - 70.4 → 25.6 (2.75x)
     - 119.3 → 28.6 (4.17x)
     - 76.0 → 34.9 (2.18x)
     - 152.1 → 37.9 (4.01x)
   * - 262144
     - 32
     - 70.3 → 26.7 (2.63x)
     - 134.7 → 29.5 (4.57x)
     - 77.2 → 36.1 (2.14x)
     - 153.2 → 39.2 (3.91x)
   * - 262144
     - 64
     - 87.0 → 36.7 (2.37x)
     - 195.2 → 39.9 (4.89x)
     - 104.5 → 45.9 (2.28x)
     - 193.5 → 49.4 (3.92x)
   * - 262144
     - 128
     - 106.4 → 56.3 (1.89x)
     - 293.5 → 62.8 (4.67x)
     - 121.1 → 65.7 (1.84x)
     - 313.6 → 72.0 (4.36x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
