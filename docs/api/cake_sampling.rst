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
a per-CTA candidate list and finishes on a gathered copy of that list with local passes; the
exact fallback for a list that overflowed or came up short streams the row again through a compact
runtime radix loop, kept small so the kernel's code stays resident in the SM instruction cache next
to the stage-2/3 kernel), so there is no size-based fallback.  :func:`cake_sampling_route` reports the decision without launching.

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
     - 2 of 192
     - 1.65x / 3.04x / 7.50x
     - 0.98x / 1.11x / 1.56x
     - 0.98x / 1.15x
   * - B200 (sm_100a, 148 SMs)
     - 0 of 192
     - 1.56x / 3.33x / 7.01x
     - 0.98x / 1.10x / 1.58x
     - 1.00x / 1.11x
   * - GB300 (sm_103a, 152 SMs)
     - 2 of 192
     - 1.39x / 3.32x / 11.89x
     - 0.97x / 1.12x / 2.01x
     - 1.00x / 1.13x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 0 of 192
     - 1.58x / 3.19x / 5.55x
     - 0.99x / 1.09x / 1.50x
     - 1.00x / 1.09x

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
     - 45.7 → 11.0 (4.15x)
     - 45.7 → 16.4 (2.79x)
     - 47.8 → 21.8 (2.19x)
     - 50.7 → 28.0 (1.81x)
   * - 32768
     - 8
     - 49.3 → 11.5 (4.29x)
     - 50.1 → 16.7 (3.00x)
     - 54.2 → 22.7 (2.39x)
     - 54.0 → 28.8 (1.88x)
   * - 32768
     - 16
     - 52.9 → 12.3 (4.30x)
     - 53.1 → 17.4 (3.05x)
     - 54.1 → 23.2 (2.33x)
     - 54.0 → 29.2 (1.85x)
   * - 32768
     - 32
     - 54.7 → 14.7 (3.72x)
     - 54.9 → 20.8 (2.64x)
     - 55.7 → 25.7 (2.17x)
     - 57.3 → 33.2 (1.73x)
   * - 32768
     - 64
     - 55.9 → 14.8 (3.78x)
     - 56.4 → 22.9 (2.46x)
     - 67.3 → 25.6 (2.63x)
     - 61.4 → 36.0 (1.71x)
   * - 32768
     - 128
     - 58.0 → 17.6 (3.30x)
     - 58.7 → 25.6 (2.29x)
     - 65.3 → 28.6 (2.28x)
     - 66.9 → 38.3 (1.75x)
   * - 128256
     - 1
     - 111.1 → 15.3 (7.26x)
     - 65.8 → 25.3 (2.60x)
     - 82.6 → 26.3 (3.14x)
     - 87.1 → 36.4 (2.39x)
   * - 128256
     - 8
     - 112.5 → 16.9 (6.66x)
     - 81.5 → 25.0 (3.26x)
     - 84.1 → 28.2 (2.98x)
     - 110.7 → 37.1 (2.98x)
   * - 128256
     - 16
     - 111.7 → 19.0 (5.88x)
     - 94.6 → 27.3 (3.47x)
     - 88.1 → 30.3 (2.91x)
     - 106.5 → 40.0 (2.66x)
   * - 128256
     - 32
     - 114.4 → 25.3 (4.52x)
     - 106.1 → 33.6 (3.16x)
     - 92.8 → 36.8 (2.52x)
     - 123.3 → 46.3 (2.66x)
   * - 128256
     - 64
     - 114.2 → 31.4 (3.64x)
     - 168.0 → 39.9 (4.21x)
     - 102.0 → 43.0 (2.37x)
     - 181.4 → 53.2 (3.41x)
   * - 128256
     - 128
     - 114.5 → 48.8 (2.35x)
     - 236.7 → 57.5 (4.12x)
     - 121.2 → 60.2 (2.01x)
     - 247.6 → 70.6 (3.51x)
   * - 151936
     - 1
     - 111.9 → 17.4 (6.43x)
     - 75.0 → 34.0 (2.21x)
     - 87.7 → 29.0 (3.02x)
     - 97.7 → 46.8 (2.09x)
   * - 151936
     - 8
     - 115.2 → 18.6 (6.19x)
     - 91.7 → 32.9 (2.79x)
     - 89.9 → 30.5 (2.95x)
     - 109.8 → 45.6 (2.41x)
   * - 151936
     - 16
     - 113.5 → 20.5 (5.54x)
     - 105.5 → 30.9 (3.41x)
     - 93.0 → 32.5 (2.86x)
     - 137.2 → 43.8 (3.13x)
   * - 151936
     - 32
     - 114.3 → 28.8 (3.97x)
     - 123.2 → 37.7 (3.27x)
     - 97.7 → 40.9 (2.39x)
     - 115.2 → 50.8 (2.27x)
   * - 151936
     - 64
     - 116.9 → 36.2 (3.23x)
     - 195.6 → 45.0 (4.35x)
     - 114.8 → 48.2 (2.38x)
     - 201.3 → 58.8 (3.42x)
   * - 151936
     - 128
     - 156.9 → 55.9 (2.81x)
     - 278.8 → 63.1 (4.42x)
     - 172.9 → 67.7 (2.55x)
     - 300.8 → 77.3 (3.89x)
   * - 262144
     - 1
     - 112.7 → 19.1 (5.90x)
     - 92.7 → 26.7 (3.47x)
     - 89.2 → 31.0 (2.88x)
     - 124.9 → 40.2 (3.11x)
   * - 262144
     - 8
     - 114.2 → 20.9 (5.46x)
     - 120.2 → 27.4 (4.39x)
     - 91.3 → 33.0 (2.77x)
     - 150.5 → 42.3 (3.56x)
   * - 262144
     - 16
     - 114.0 → 26.1 (4.37x)
     - 148.5 → 32.1 (4.63x)
     - 97.3 → 38.5 (2.53x)
     - 163.5 → 46.8 (3.49x)
   * - 262144
     - 32
     - 128.1 → 40.6 (3.16x)
     - 236.2 → 46.1 (5.12x)
     - 145.3 → 53.2 (2.73x)
     - 224.2 → 60.2 (3.72x)
   * - 262144
     - 64
     - 135.7 → 52.5 (2.58x)
     - 316.7 → 58.2 (5.44x)
     - 151.0 → 65.9 (2.29x)
     - 335.1 → 73.6 (4.55x)
   * - 262144
     - 128
     - 158.4 → 83.2 (1.90x)
     - 470.3 → 90.8 (5.18x)
     - 174.9 → 95.6 (1.83x)
     - 478.0 → 104.7 (4.57x)

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
     - 42.2 → 10.8 (3.91x)
     - 42.8 → 16.0 (2.67x)
     - 45.5 → 20.4 (2.23x)
     - 41.6 → 26.7 (1.56x)
   * - 32768
     - 8
     - 46.4 → 11.3 (4.11x)
     - 47.3 → 16.4 (2.88x)
     - 51.4 → 21.2 (2.42x)
     - 47.4 → 27.9 (1.70x)
   * - 32768
     - 16
     - 49.0 → 11.5 (4.26x)
     - 49.5 → 16.7 (2.96x)
     - 46.3 → 21.2 (2.18x)
     - 51.5 → 28.1 (1.83x)
   * - 32768
     - 32
     - 50.5 → 12.2 (4.14x)
     - 51.3 → 17.4 (2.95x)
     - 52.2 → 21.9 (2.38x)
     - 52.2 → 29.1 (1.79x)
   * - 32768
     - 64
     - 52.2 → 13.7 (3.81x)
     - 53.3 → 19.9 (2.68x)
     - 53.7 → 23.7 (2.27x)
     - 56.6 → 31.3 (1.81x)
   * - 32768
     - 128
     - 53.2 → 14.8 (3.59x)
     - 54.5 → 22.6 (2.41x)
     - 56.4 → 24.2 (2.33x)
     - 54.8 → 34.7 (1.58x)
   * - 128256
     - 1
     - 101.2 → 14.7 (6.88x)
     - 65.2 → 20.0 (3.26x)
     - 82.8 → 24.6 (3.37x)
     - 68.6 → 31.2 (2.20x)
   * - 128256
     - 8
     - 101.1 → 15.4 (6.56x)
     - 81.0 → 21.1 (3.84x)
     - 83.9 → 25.6 (3.28x)
     - 87.4 → 32.8 (2.66x)
   * - 128256
     - 16
     - 101.2 → 17.6 (5.75x)
     - 92.9 → 25.5 (3.64x)
     - 82.2 → 27.6 (2.98x)
     - 92.7 → 36.5 (2.54x)
   * - 128256
     - 32
     - 104.5 → 18.8 (5.56x)
     - 100.3 → 26.6 (3.77x)
     - 87.2 → 29.1 (3.00x)
     - 107.1 → 38.5 (2.78x)
   * - 128256
     - 64
     - 104.4 → 25.1 (4.16x)
     - 144.4 → 32.8 (4.40x)
     - 88.6 → 35.5 (2.50x)
     - 150.9 → 44.8 (3.37x)
   * - 128256
     - 128
     - 105.3 → 37.0 (2.85x)
     - 192.8 → 45.0 (4.28x)
     - 94.9 → 46.9 (2.02x)
     - 213.3 → 57.4 (3.72x)
   * - 151936
     - 1
     - 101.7 → 16.2 (6.28x)
     - 73.4 → 22.4 (3.28x)
     - 83.9 → 26.4 (3.18x)
     - 74.7 → 33.6 (2.22x)
   * - 151936
     - 8
     - 102.6 → 17.0 (6.04x)
     - 92.3 → 23.8 (3.88x)
     - 91.7 → 27.3 (3.36x)
     - 98.2 → 36.0 (2.73x)
   * - 151936
     - 16
     - 102.2 → 19.3 (5.30x)
     - 101.8 → 25.3 (4.02x)
     - 88.1 → 29.5 (2.99x)
     - 106.0 → 37.4 (2.83x)
   * - 151936
     - 32
     - 102.7 → 20.5 (5.01x)
     - 113.8 → 26.2 (4.34x)
     - 89.7 → 30.9 (2.90x)
     - 134.0 → 38.6 (3.47x)
   * - 151936
     - 64
     - 105.7 → 28.9 (3.66x)
     - 164.7 → 35.9 (4.59x)
     - 92.8 → 39.8 (2.33x)
     - 170.6 → 48.0 (3.55x)
   * - 151936
     - 128
     - 105.6 → 43.2 (2.44x)
     - 221.6 → 50.0 (4.43x)
     - 104.4 → 53.4 (1.96x)
     - 242.2 → 62.2 (3.89x)
   * - 262144
     - 1
     - 101.9 → 17.9 (5.69x)
     - 92.0 → 26.4 (3.48x)
     - 89.1 → 28.3 (3.15x)
     - 86.4 → 37.6 (2.30x)
   * - 262144
     - 8
     - 103.3 → 18.9 (5.47x)
     - 118.4 → 25.7 (4.61x)
     - 93.2 → 29.7 (3.14x)
     - 204.6 → 38.6 (5.30x)
   * - 262144
     - 16
     - 104.0 → 24.3 (4.28x)
     - 140.5 → 30.5 (4.61x)
     - 91.5 → 35.0 (2.61x)
     - 193.2 → 42.3 (4.57x)
   * - 262144
     - 32
     - 114.7 → 25.8 (4.45x)
     - 200.5 → 31.5 (6.37x)
     - 133.0 → 36.5 (3.64x)
     - 218.5 → 45.0 (4.86x)
   * - 262144
     - 64
     - 107.3 → 39.7 (2.70x)
     - 267.2 → 45.2 (5.91x)
     - 123.8 → 50.7 (2.44x)
     - 276.3 → 58.9 (4.69x)
   * - 262144
     - 128
     - 123.5 → 65.3 (1.89x)
     - 376.9 → 71.9 (5.24x)
     - 143.8 → 75.7 (1.90x)
     - 397.8 → 85.1 (4.67x)

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
     - 66.2 → 11.1 (5.96x)
     - 74.6 → 23.9 (3.12x)
     - 42.3 → 22.7 (1.86x)
     - 42.4 → 30.5 (1.39x)
   * - 32768
     - 8
     - 72.3 → 11.5 (6.29x)
     - 70.8 → 22.2 (3.19x)
     - 46.0 → 22.7 (2.03x)
     - 48.2 → 30.6 (1.58x)
   * - 32768
     - 16
     - 71.6 → 11.7 (6.12x)
     - 73.2 → 22.2 (3.30x)
     - 46.7 → 23.2 (2.01x)
     - 54.6 → 30.7 (1.78x)
   * - 32768
     - 32
     - 74.5 → 12.4 (6.01x)
     - 75.1 → 22.2 (3.38x)
     - 54.5 → 24.1 (2.26x)
     - 52.2 → 32.4 (1.61x)
   * - 32768
     - 64
     - 76.8 → 14.1 (5.45x)
     - 76.9 → 22.1 (3.48x)
     - 57.9 → 25.6 (2.26x)
     - 55.7 → 35.6 (1.56x)
   * - 32768
     - 128
     - 76.9 → 14.8 (5.20x)
     - 79.1 → 23.7 (3.34x)
     - 57.1 → 26.3 (2.17x)
     - 54.1 → 36.7 (1.47x)
   * - 128256
     - 1
     - 179.2 → 15.2 (11.79x)
     - 82.2 → 22.2 (3.70x)
     - 77.6 → 27.1 (2.86x)
     - 70.0 → 34.6 (2.02x)
   * - 128256
     - 8
     - 173.4 → 15.6 (11.12x)
     - 101.5 → 22.6 (4.49x)
     - 83.7 → 27.4 (3.05x)
     - 100.1 → 35.2 (2.84x)
   * - 128256
     - 16
     - 172.9 → 17.8 (9.71x)
     - 107.4 → 26.8 (4.01x)
     - 84.1 → 29.8 (2.82x)
     - 89.9 → 40.3 (2.23x)
   * - 128256
     - 32
     - 179.6 → 18.6 (9.66x)
     - 114.0 → 27.6 (4.13x)
     - 85.5 → 30.6 (2.79x)
     - 110.4 → 41.6 (2.65x)
   * - 128256
     - 64
     - 178.2 → 24.7 (7.21x)
     - 139.1 → 33.6 (4.14x)
     - 88.4 → 36.4 (2.43x)
     - 151.6 → 47.4 (3.20x)
   * - 128256
     - 128
     - 181.0 → 36.0 (5.03x)
     - 184.5 → 45.1 (4.09x)
     - 92.0 → 47.4 (1.94x)
     - 192.6 → 58.3 (3.30x)
   * - 151936
     - 1
     - 176.5 → 16.6 (10.63x)
     - 86.3 → 24.1 (3.58x)
     - 85.9 → 28.7 (2.99x)
     - 103.2 → 38.6 (2.67x)
   * - 151936
     - 8
     - 175.0 → 17.2 (10.17x)
     - 106.6 → 25.1 (4.25x)
     - 89.3 → 29.4 (3.04x)
     - 96.6 → 38.4 (2.52x)
   * - 151936
     - 16
     - 175.1 → 19.3 (9.07x)
     - 114.6 → 26.6 (4.31x)
     - 89.7 → 31.9 (2.81x)
     - 118.9 → 39.9 (2.98x)
   * - 151936
     - 32
     - 176.6 → 20.2 (8.74x)
     - 122.5 → 27.1 (4.52x)
     - 90.2 → 32.5 (2.78x)
     - 102.7 → 41.8 (2.46x)
   * - 151936
     - 64
     - 179.7 → 28.3 (6.35x)
     - 157.1 → 35.4 (4.44x)
     - 91.9 → 40.5 (2.27x)
     - 162.1 → 49.3 (3.29x)
   * - 151936
     - 128
     - 179.3 → 41.8 (4.29x)
     - 212.9 → 49.4 (4.31x)
     - 101.0 → 53.6 (1.88x)
     - 230.5 → 63.2 (3.65x)
   * - 262144
     - 1
     - 175.3 → 18.2 (9.63x)
     - 106.7 → 28.2 (3.78x)
     - 82.7 → 30.4 (2.72x)
     - 90.5 → 42.8 (2.11x)
   * - 262144
     - 8
     - 175.2 → 19.1 (9.17x)
     - 137.1 → 27.2 (5.04x)
     - 90.2 → 31.6 (2.85x)
     - 123.1 → 41.6 (2.96x)
   * - 262144
     - 16
     - 174.4 → 24.2 (7.21x)
     - 150.1 → 31.4 (4.78x)
     - 90.2 → 36.9 (2.44x)
     - 169.0 → 45.5 (3.71x)
   * - 262144
     - 32
     - 173.7 → 25.4 (6.84x)
     - 190.7 → 32.3 (5.90x)
     - 131.9 → 38.1 (3.46x)
     - 207.7 → 46.6 (4.46x)
   * - 262144
     - 64
     - 177.3 → 38.6 (4.59x)
     - 254.3 → 45.4 (5.60x)
     - 120.4 → 51.2 (2.35x)
     - 271.4 → 59.4 (4.57x)
   * - 262144
     - 128
     - 178.6 → 63.4 (2.82x)
     - 361.2 → 71.3 (5.07x)
     - 143.2 → 75.9 (1.89x)
     - 375.9 → 84.9 (4.43x)

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
     - 30.0 → 9.6 (3.12x)
     - 30.9 → 12.9 (2.40x)
     - 32.6 → 16.9 (1.93x)
     - 34.4 → 21.7 (1.59x)
   * - 32768
     - 8
     - 33.4 → 10.3 (3.24x)
     - 33.5 → 13.8 (2.43x)
     - 41.5 → 18.0 (2.31x)
     - 42.0 → 23.0 (1.83x)
   * - 32768
     - 16
     - 35.3 → 10.8 (3.27x)
     - 35.5 → 14.0 (2.54x)
     - 42.2 → 18.5 (2.28x)
     - 43.5 → 23.2 (1.88x)
   * - 32768
     - 32
     - 37.1 → 11.3 (3.28x)
     - 37.1 → 14.2 (2.61x)
     - 44.4 → 18.5 (2.40x)
     - 41.5 → 22.9 (1.81x)
   * - 32768
     - 64
     - 38.2 → 12.6 (3.03x)
     - 38.4 → 15.9 (2.42x)
     - 44.0 → 20.1 (2.19x)
     - 43.7 → 24.5 (1.78x)
   * - 32768
     - 128
     - 39.3 → 13.4 (2.93x)
     - 39.5 → 18.5 (2.14x)
     - 45.0 → 20.7 (2.17x)
     - 51.4 → 28.5 (1.80x)
   * - 128256
     - 1
     - 70.2 → 13.1 (5.36x)
     - 56.6 → 16.0 (3.54x)
     - 70.5 → 20.6 (3.42x)
     - 62.4 → 25.5 (2.45x)
   * - 128256
     - 8
     - 70.1 → 13.9 (5.04x)
     - 67.9 → 17.4 (3.90x)
     - 74.0 → 21.6 (3.43x)
     - 81.1 → 26.0 (3.12x)
   * - 128256
     - 16
     - 70.0 → 14.8 (4.73x)
     - 76.4 → 18.0 (4.24x)
     - 73.0 → 22.4 (3.26x)
     - 85.3 → 27.5 (3.10x)
   * - 128256
     - 32
     - 72.4 → 16.5 (4.39x)
     - 84.4 → 21.5 (3.93x)
     - 75.2 → 23.9 (3.15x)
     - 97.7 → 31.0 (3.15x)
   * - 128256
     - 64
     - 72.5 → 22.5 (3.22x)
     - 92.9 → 27.4 (3.39x)
     - 76.9 → 30.0 (2.56x)
     - 115.1 → 36.7 (3.14x)
   * - 128256
     - 128
     - 72.3 → 32.7 (2.21x)
     - 137.1 → 37.1 (3.70x)
     - 78.5 → 40.0 (1.96x)
     - 145.3 → 46.4 (3.13x)
   * - 151936
     - 1
     - 70.0 → 14.5 (4.83x)
     - 64.2 → 18.1 (3.55x)
     - 70.2 → 22.0 (3.19x)
     - 68.8 → 27.3 (2.52x)
   * - 151936
     - 8
     - 70.6 → 15.2 (4.64x)
     - 77.1 → 19.3 (3.99x)
     - 77.4 → 22.8 (3.39x)
     - 104.3 → 28.0 (3.73x)
   * - 151936
     - 16
     - 70.4 → 15.9 (4.43x)
     - 87.3 → 20.1 (4.34x)
     - 78.6 → 23.3 (3.37x)
     - 87.8 → 29.5 (2.98x)
   * - 151936
     - 32
     - 70.3 → 18.0 (3.91x)
     - 96.4 → 21.3 (4.53x)
     - 80.4 → 25.7 (3.13x)
     - 111.5 → 30.1 (3.70x)
   * - 151936
     - 64
     - 72.5 → 25.5 (2.84x)
     - 105.9 → 30.5 (3.47x)
     - 79.4 → 33.1 (2.40x)
     - 110.4 → 39.3 (2.81x)
   * - 151936
     - 128
     - 73.5 → 37.6 (1.95x)
     - 161.0 → 41.6 (3.87x)
     - 89.2 → 44.9 (1.99x)
     - 175.6 → 50.3 (3.49x)
   * - 262144
     - 1
     - 70.0 → 16.1 (4.35x)
     - 79.1 → 20.1 (3.94x)
     - 75.1 → 23.7 (3.17x)
     - 83.5 → 29.8 (2.80x)
   * - 262144
     - 8
     - 70.3 → 17.0 (4.14x)
     - 102.8 → 20.9 (4.92x)
     - 77.2 → 24.4 (3.16x)
     - 121.2 → 30.3 (4.00x)
   * - 262144
     - 16
     - 70.2 → 18.0 (3.90x)
     - 119.4 → 21.5 (5.55x)
     - 78.3 → 25.2 (3.11x)
     - 123.9 → 30.8 (4.02x)
   * - 262144
     - 32
     - 70.5 → 22.4 (3.15x)
     - 134.3 → 25.6 (5.25x)
     - 80.2 → 30.1 (2.66x)
     - 136.1 → 35.4 (3.84x)
   * - 262144
     - 64
     - 87.1 → 34.5 (2.52x)
     - 195.6 → 37.6 (5.20x)
     - 108.0 → 41.8 (2.58x)
     - 209.8 → 46.9 (4.47x)
   * - 262144
     - 128
     - 106.7 → 55.5 (1.92x)
     - 293.9 → 58.7 (5.01x)
     - 123.4 → 63.0 (1.96x)
     - 333.6 → 67.9 (4.91x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
