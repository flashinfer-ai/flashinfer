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
     - 1.56x / 3.03x / 7.55x
     - 0.98x / 1.10x / 1.55x
     - 0.98x / 1.14x
   * - B200 (sm_100a, 148 SMs)
     - 1 of 192
     - 1.64x / 3.23x / 7.06x
     - 0.98x / 1.10x / 1.51x
     - 0.96x / 1.11x
   * - GB300 (sm_103a, 152 SMs)
     - 1 of 192
     - 1.40x / 3.33x / 11.90x
     - 0.97x / 1.13x / 2.06x
     - 1.00x / 1.14x
   * - VR200 R200 (sm_107a, 212 SMs)
     - 4 of 192
     - 1.61x / 3.19x / 5.40x
     - 0.99x / 1.08x / 1.46x
     - 0.97x / 1.08x

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
     - 45.6 → 11.0 (4.15x)
     - 45.8 → 16.4 (2.79x)
     - 56.8 → 21.8 (2.61x)
     - 43.6 → 27.9 (1.56x)
   * - 32768
     - 8
     - 50.9 → 11.5 (4.43x)
     - 49.8 → 16.7 (2.98x)
     - 49.4 → 22.5 (2.20x)
     - 58.7 → 28.8 (2.04x)
   * - 32768
     - 16
     - 52.5 → 12.3 (4.27x)
     - 52.8 → 17.5 (3.02x)
     - 50.2 → 23.2 (2.16x)
     - 57.7 → 29.2 (1.98x)
   * - 32768
     - 32
     - 54.8 → 14.8 (3.70x)
     - 55.3 → 20.8 (2.66x)
     - 50.8 → 25.9 (1.96x)
     - 56.4 → 33.2 (1.70x)
   * - 32768
     - 64
     - 55.9 → 15.2 (3.68x)
     - 56.1 → 22.3 (2.52x)
     - 58.1 → 25.9 (2.24x)
     - 61.2 → 35.4 (1.73x)
   * - 32768
     - 128
     - 57.9 → 17.9 (3.23x)
     - 58.4 → 25.7 (2.27x)
     - 62.9 → 28.9 (2.18x)
     - 66.8 → 38.4 (1.74x)
   * - 128256
     - 1
     - 111.1 → 15.2 (7.31x)
     - 66.5 → 23.9 (2.78x)
     - 84.0 → 26.4 (3.18x)
     - 66.7 → 37.1 (1.80x)
   * - 128256
     - 8
     - 112.3 → 17.2 (6.53x)
     - 81.8 → 24.4 (3.35x)
     - 86.3 → 28.3 (3.05x)
     - 109.2 → 37.2 (2.94x)
   * - 128256
     - 16
     - 111.5 → 19.5 (5.72x)
     - 92.7 → 27.7 (3.35x)
     - 87.8 → 30.7 (2.86x)
     - 125.2 → 40.3 (3.11x)
   * - 128256
     - 32
     - 114.1 → 25.9 (4.41x)
     - 106.4 → 34.2 (3.11x)
     - 93.0 → 37.3 (2.49x)
     - 127.6 → 47.1 (2.71x)
   * - 128256
     - 64
     - 114.7 → 31.7 (3.62x)
     - 167.0 → 40.4 (4.13x)
     - 104.6 → 43.3 (2.42x)
     - 171.7 → 53.4 (3.22x)
   * - 128256
     - 128
     - 114.8 → 49.3 (2.33x)
     - 236.3 → 57.9 (4.08x)
     - 121.3 → 60.7 (2.00x)
     - 269.3 → 70.8 (3.80x)
   * - 151936
     - 1
     - 111.4 → 17.4 (6.40x)
     - 74.3 → 33.9 (2.19x)
     - 89.5 → 28.9 (3.10x)
     - 73.2 → 46.7 (1.57x)
   * - 151936
     - 8
     - 112.8 → 18.8 (6.00x)
     - 92.3 → 33.0 (2.80x)
     - 92.4 → 30.7 (3.01x)
     - 140.0 → 46.2 (3.03x)
   * - 151936
     - 16
     - 112.5 → 21.1 (5.33x)
     - 105.5 → 30.3 (3.48x)
     - 93.2 → 33.2 (2.81x)
     - 105.3 → 43.3 (2.43x)
   * - 151936
     - 32
     - 113.5 → 29.5 (3.85x)
     - 124.7 → 38.0 (3.28x)
     - 97.7 → 41.9 (2.33x)
     - 131.6 → 51.6 (2.55x)
   * - 151936
     - 64
     - 115.7 → 36.7 (3.15x)
     - 195.6 → 44.8 (4.37x)
     - 117.3 → 48.7 (2.41x)
     - 217.4 → 59.3 (3.67x)
   * - 151936
     - 128
     - 157.0 → 56.5 (2.78x)
     - 280.1 → 63.7 (4.40x)
     - 173.5 → 68.3 (2.54x)
     - 319.9 → 77.8 (4.11x)
   * - 262144
     - 1
     - 113.1 → 19.2 (5.89x)
     - 91.1 → 27.0 (3.37x)
     - 91.6 → 31.8 (2.88x)
     - 82.8 → 40.8 (2.03x)
   * - 262144
     - 8
     - 114.9 → 21.2 (5.42x)
     - 120.0 → 28.0 (4.29x)
     - 94.3 → 34.0 (2.77x)
     - 128.8 → 42.5 (3.03x)
   * - 262144
     - 16
     - 114.1 → 27.1 (4.21x)
     - 147.9 → 33.5 (4.41x)
     - 97.6 → 40.1 (2.43x)
     - 186.2 → 47.4 (3.93x)
   * - 262144
     - 32
     - 127.7 → 41.8 (3.06x)
     - 233.3 → 47.3 (4.93x)
     - 145.2 → 54.8 (2.65x)
     - 237.0 → 61.2 (3.87x)
   * - 262144
     - 64
     - 135.5 → 52.9 (2.56x)
     - 319.1 → 58.6 (5.45x)
     - 153.8 → 66.5 (2.31x)
     - 327.5 → 73.5 (4.46x)
   * - 262144
     - 128
     - 158.2 → 83.9 (1.89x)
     - 471.9 → 91.2 (5.17x)
     - 174.9 → 96.4 (1.81x)
     - 484.5 → 104.9 (4.62x)

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
     - 41.7 → 10.5 (3.97x)
     - 41.7 → 15.7 (2.66x)
     - 40.4 → 20.4 (1.98x)
     - 43.5 → 26.5 (1.64x)
   * - 32768
     - 8
     - 46.4 → 11.2 (4.14x)
     - 47.1 → 16.3 (2.89x)
     - 48.0 → 21.2 (2.26x)
     - 47.4 → 27.5 (1.72x)
   * - 32768
     - 16
     - 49.5 → 11.5 (4.30x)
     - 49.7 → 16.4 (3.03x)
     - 49.4 → 21.2 (2.33x)
     - 46.8 → 27.8 (1.68x)
   * - 32768
     - 32
     - 50.9 → 12.2 (4.17x)
     - 51.2 → 17.4 (2.94x)
     - 54.5 → 22.0 (2.48x)
     - 54.5 → 28.9 (1.89x)
   * - 32768
     - 64
     - 52.4 → 13.7 (3.82x)
     - 56.5 → 19.9 (2.84x)
     - 55.7 → 23.9 (2.33x)
     - 54.4 → 31.5 (1.73x)
   * - 32768
     - 128
     - 53.4 → 15.0 (3.56x)
     - 53.5 → 22.8 (2.35x)
     - 62.8 → 24.4 (2.57x)
     - 61.3 → 34.8 (1.76x)
   * - 128256
     - 1
     - 102.5 → 14.8 (6.93x)
     - 65.7 → 19.8 (3.32x)
     - 76.6 → 25.1 (3.05x)
     - 75.6 → 31.0 (2.44x)
   * - 128256
     - 8
     - 101.9 → 15.4 (6.62x)
     - 81.2 → 21.0 (3.87x)
     - 81.6 → 25.6 (3.19x)
     - 111.6 → 32.1 (3.48x)
   * - 128256
     - 16
     - 101.7 → 17.6 (5.78x)
     - 91.5 → 25.2 (3.63x)
     - 84.1 → 27.7 (3.04x)
     - 94.0 → 36.3 (2.59x)
   * - 128256
     - 32
     - 104.4 → 18.8 (5.55x)
     - 99.3 → 26.5 (3.75x)
     - 86.3 → 29.0 (2.98x)
     - 113.3 → 38.4 (2.95x)
   * - 128256
     - 64
     - 104.2 → 25.0 (4.17x)
     - 144.0 → 32.7 (4.40x)
     - 84.4 → 35.3 (2.39x)
     - 152.1 → 44.5 (3.42x)
   * - 128256
     - 128
     - 105.7 → 37.6 (2.81x)
     - 192.8 → 45.7 (4.22x)
     - 90.6 → 47.6 (1.90x)
     - 206.2 → 57.1 (3.61x)
   * - 151936
     - 1
     - 102.9 → 16.3 (6.31x)
     - 72.6 → 22.4 (3.24x)
     - 77.2 → 26.9 (2.87x)
     - 82.9 → 33.8 (2.45x)
   * - 151936
     - 8
     - 103.7 → 17.0 (6.10x)
     - 92.4 → 23.5 (3.93x)
     - 89.3 → 27.4 (3.26x)
     - 127.8 → 35.3 (3.62x)
   * - 151936
     - 16
     - 102.6 → 19.2 (5.34x)
     - 103.0 → 25.1 (4.10x)
     - 90.2 → 29.7 (3.04x)
     - 119.0 → 37.0 (3.22x)
   * - 151936
     - 32
     - 103.2 → 20.6 (5.01x)
     - 113.3 → 26.2 (4.32x)
     - 91.8 → 31.2 (2.94x)
     - 118.7 → 38.1 (3.12x)
   * - 151936
     - 64
     - 106.4 → 28.9 (3.68x)
     - 164.5 → 35.8 (4.59x)
     - 95.3 → 39.8 (2.39x)
     - 186.1 → 47.5 (3.92x)
   * - 151936
     - 128
     - 107.2 → 44.1 (2.43x)
     - 221.8 → 50.9 (4.36x)
     - 103.1 → 54.3 (1.90x)
     - 226.2 → 63.6 (3.56x)
   * - 262144
     - 1
     - 103.4 → 18.0 (5.74x)
     - 88.8 → 25.5 (3.48x)
     - 83.1 → 28.8 (2.89x)
     - 101.8 → 38.1 (2.67x)
   * - 262144
     - 8
     - 104.9 → 18.9 (5.55x)
     - 121.8 → 26.0 (4.68x)
     - 90.6 → 29.6 (3.06x)
     - 128.8 → 38.1 (3.38x)
   * - 262144
     - 16
     - 104.6 → 24.3 (4.30x)
     - 141.4 → 30.2 (4.68x)
     - 93.0 → 35.0 (2.66x)
     - 137.6 → 42.4 (3.25x)
   * - 262144
     - 32
     - 113.4 → 25.9 (4.38x)
     - 198.9 → 31.6 (6.29x)
     - 133.6 → 36.7 (3.64x)
     - 204.7 → 44.1 (4.64x)
   * - 262144
     - 64
     - 108.3 → 39.6 (2.73x)
     - 267.4 → 45.2 (5.92x)
     - 122.3 → 55.7 (2.20x)
     - 296.2 → 58.0 (5.11x)
   * - 262144
     - 128
     - 123.3 → 66.7 (1.85x)
     - 376.4 → 73.4 (5.13x)
     - 163.6 → 77.1 (2.12x)
     - 386.4 → 86.0 (4.49x)

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
     - 63.6 → 10.9 (5.83x)
     - 64.4 → 24.3 (2.65x)
     - 41.8 → 22.3 (1.87x)
     - 42.5 → 30.3 (1.40x)
   * - 32768
     - 8
     - 69.5 → 11.5 (6.04x)
     - 68.0 → 22.0 (3.09x)
     - 45.5 → 22.5 (2.02x)
     - 45.6 → 30.0 (1.52x)
   * - 32768
     - 16
     - 70.7 → 11.7 (6.04x)
     - 72.3 → 22.1 (3.27x)
     - 50.5 → 23.2 (2.18x)
     - 54.4 → 30.3 (1.80x)
   * - 32768
     - 32
     - 70.8 → 12.4 (5.71x)
     - 70.7 → 22.2 (3.18x)
     - 49.5 → 23.9 (2.07x)
     - 51.6 → 32.2 (1.60x)
   * - 32768
     - 64
     - 73.8 → 14.1 (5.23x)
     - 74.0 → 22.2 (3.33x)
     - 56.6 → 25.4 (2.23x)
     - 53.4 → 35.0 (1.53x)
   * - 32768
     - 128
     - 74.8 → 14.9 (5.02x)
     - 76.0 → 23.7 (3.21x)
     - 56.5 → 26.5 (2.13x)
     - 55.0 → 36.6 (1.50x)
   * - 128256
     - 1
     - 170.6 → 14.8 (11.53x)
     - 80.4 → 22.0 (3.65x)
     - 75.6 → 26.8 (2.82x)
     - 69.2 → 34.5 (2.01x)
   * - 128256
     - 8
     - 166.9 → 15.4 (10.84x)
     - 95.4 → 22.3 (4.28x)
     - 80.1 → 27.3 (2.93x)
     - 82.6 → 34.3 (2.41x)
   * - 128256
     - 16
     - 173.1 → 17.7 (9.78x)
     - 103.3 → 26.6 (3.88x)
     - 81.7 → 29.7 (2.75x)
     - 106.9 → 39.4 (2.71x)
   * - 128256
     - 32
     - 173.2 → 18.6 (9.31x)
     - 111.1 → 27.4 (4.05x)
     - 84.6 → 30.8 (2.75x)
     - 102.8 → 40.9 (2.51x)
   * - 128256
     - 64
     - 174.8 → 24.6 (7.11x)
     - 136.4 → 33.3 (4.10x)
     - 88.0 → 36.3 (2.42x)
     - 155.4 → 46.6 (3.33x)
   * - 128256
     - 128
     - 174.2 → 36.3 (4.80x)
     - 184.4 → 45.4 (4.06x)
     - 94.5 → 47.8 (1.98x)
     - 205.1 → 58.1 (3.53x)
   * - 151936
     - 1
     - 172.3 → 16.2 (10.64x)
     - 84.1 → 23.5 (3.58x)
     - 79.4 → 28.4 (2.80x)
     - 75.5 → 36.7 (2.06x)
   * - 151936
     - 8
     - 172.1 → 17.0 (10.12x)
     - 101.7 → 24.7 (4.12x)
     - 84.1 → 28.8 (2.92x)
     - 92.2 → 36.8 (2.51x)
   * - 151936
     - 16
     - 172.6 → 19.3 (8.94x)
     - 112.8 → 26.4 (4.27x)
     - 86.2 → 31.5 (2.74x)
     - 106.1 → 39.6 (2.68x)
   * - 151936
     - 32
     - 167.3 → 20.2 (8.28x)
     - 117.1 → 27.1 (4.32x)
     - 90.2 → 32.6 (2.77x)
     - 102.1 → 41.2 (2.48x)
   * - 151936
     - 64
     - 174.7 → 28.2 (6.20x)
     - 155.6 → 35.1 (4.43x)
     - 89.7 → 40.0 (2.24x)
     - 169.8 → 48.4 (3.51x)
   * - 151936
     - 128
     - 176.2 → 42.2 (4.18x)
     - 213.1 → 49.8 (4.28x)
     - 102.5 → 54.3 (1.89x)
     - 227.8 → 63.5 (3.59x)
   * - 262144
     - 1
     - 170.9 → 17.8 (9.60x)
     - 100.8 → 26.7 (3.78x)
     - 81.9 → 30.1 (2.72x)
     - 91.1 → 42.8 (2.13x)
   * - 262144
     - 8
     - 169.2 → 18.9 (8.95x)
     - 131.2 → 27.7 (4.74x)
     - 85.6 → 31.1 (2.75x)
     - 155.4 → 41.3 (3.76x)
   * - 262144
     - 16
     - 167.1 → 24.0 (6.96x)
     - 145.2 → 31.3 (4.64x)
     - 87.9 → 36.0 (2.44x)
     - 125.0 → 45.8 (2.73x)
   * - 262144
     - 32
     - 171.1 → 25.2 (6.79x)
     - 189.9 → 32.3 (5.88x)
     - 131.4 → 37.3 (3.52x)
     - 185.6 → 46.0 (4.03x)
   * - 262144
     - 64
     - 176.2 → 38.3 (4.60x)
     - 254.3 → 45.1 (5.64x)
     - 119.4 → 50.7 (2.36x)
     - 277.8 → 58.6 (4.74x)
   * - 262144
     - 128
     - 178.4 → 64.0 (2.79x)
     - 361.1 → 72.0 (5.02x)
     - 143.7 → 76.0 (1.89x)
     - 364.5 → 85.2 (4.28x)

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
     - 30.1 → 12.9 (2.33x)
     - 31.8 → 16.9 (1.88x)
     - 35.0 → 21.8 (1.61x)
   * - 32768
     - 8
     - 33.4 → 10.3 (3.24x)
     - 33.7 → 13.8 (2.44x)
     - 37.4 → 17.9 (2.09x)
     - 37.5 → 22.9 (1.64x)
   * - 32768
     - 16
     - 35.6 → 10.8 (3.30x)
     - 35.7 → 14.0 (2.55x)
     - 42.2 → 18.4 (2.29x)
     - 44.3 → 23.3 (1.90x)
   * - 32768
     - 32
     - 36.7 → 11.3 (3.25x)
     - 37.2 → 14.1 (2.64x)
     - 44.9 → 18.5 (2.43x)
     - 45.2 → 22.9 (1.97x)
   * - 32768
     - 64
     - 38.1 → 12.6 (3.02x)
     - 38.4 → 15.9 (2.42x)
     - 44.6 → 20.1 (2.22x)
     - 45.4 → 24.5 (1.85x)
   * - 32768
     - 128
     - 39.1 → 13.9 (2.81x)
     - 39.2 → 19.0 (2.06x)
     - 46.8 → 21.1 (2.22x)
     - 47.6 → 28.5 (1.67x)
   * - 128256
     - 1
     - 69.4 → 13.0 (5.34x)
     - 55.7 → 16.0 (3.48x)
     - 70.6 → 20.6 (3.43x)
     - 65.9 → 25.6 (2.57x)
   * - 128256
     - 8
     - 70.8 → 13.9 (5.09x)
     - 67.9 → 17.4 (3.90x)
     - 73.8 → 21.6 (3.42x)
     - 74.3 → 26.1 (2.85x)
   * - 128256
     - 16
     - 70.4 → 14.8 (4.76x)
     - 76.7 → 18.0 (4.26x)
     - 73.3 → 22.4 (3.27x)
     - 79.5 → 27.4 (2.90x)
   * - 128256
     - 32
     - 73.0 → 16.9 (4.32x)
     - 84.2 → 21.9 (3.84x)
     - 75.4 → 24.3 (3.10x)
     - 103.0 → 31.3 (3.29x)
   * - 128256
     - 64
     - 72.9 → 22.8 (3.20x)
     - 92.5 → 27.6 (3.35x)
     - 78.6 → 30.3 (2.59x)
     - 96.4 → 36.9 (2.61x)
   * - 128256
     - 128
     - 72.6 → 34.0 (2.14x)
     - 137.8 → 38.5 (3.58x)
     - 80.5 → 41.3 (1.95x)
     - 142.7 → 47.7 (2.99x)
   * - 151936
     - 1
     - 70.3 → 14.9 (4.72x)
     - 63.3 → 18.1 (3.50x)
     - 70.1 → 22.2 (3.16x)
     - 73.0 → 27.1 (2.69x)
   * - 151936
     - 8
     - 70.1 → 15.6 (4.49x)
     - 77.3 → 19.3 (4.01x)
     - 77.2 → 23.3 (3.31x)
     - 83.5 → 27.8 (3.00x)
   * - 151936
     - 16
     - 69.8 → 16.3 (4.28x)
     - 88.0 → 20.1 (4.38x)
     - 78.9 → 23.8 (3.32x)
     - 97.7 → 29.4 (3.32x)
   * - 151936
     - 32
     - 70.0 → 18.4 (3.80x)
     - 97.2 → 21.8 (4.46x)
     - 80.5 → 26.0 (3.10x)
     - 110.9 → 30.4 (3.65x)
   * - 151936
     - 64
     - 72.2 → 25.9 (2.79x)
     - 106.0 → 30.6 (3.46x)
     - 81.2 → 33.5 (2.42x)
     - 118.4 → 39.3 (3.01x)
   * - 151936
     - 128
     - 73.1 → 39.3 (1.86x)
     - 160.4 → 43.2 (3.71x)
     - 90.7 → 46.6 (1.95x)
     - 165.1 → 52.3 (3.16x)
   * - 262144
     - 1
     - 69.6 → 16.6 (4.19x)
     - 79.1 → 20.7 (3.82x)
     - 75.6 → 24.1 (3.14x)
     - 90.1 → 29.7 (3.03x)
   * - 262144
     - 8
     - 70.2 → 17.4 (4.03x)
     - 102.8 → 21.3 (4.83x)
     - 78.1 → 24.8 (3.15x)
     - 109.4 → 30.9 (3.54x)
   * - 262144
     - 16
     - 70.7 → 18.3 (3.86x)
     - 118.3 → 22.0 (5.38x)
     - 78.7 → 25.6 (3.07x)
     - 147.4 → 31.1 (4.74x)
   * - 262144
     - 32
     - 70.0 → 23.1 (3.03x)
     - 135.9 → 26.3 (5.17x)
     - 80.3 → 30.9 (2.60x)
     - 192.8 → 35.7 (5.40x)
   * - 262144
     - 64
     - 87.3 → 35.0 (2.49x)
     - 195.7 → 38.1 (5.14x)
     - 107.7 → 42.5 (2.53x)
     - 199.3 → 47.5 (4.20x)
   * - 262144
     - 128
     - 106.7 → 58.2 (1.83x)
     - 294.9 → 61.3 (4.81x)
     - 125.2 → 65.7 (1.91x)
     - 287.8 → 70.8 (4.06x)

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
