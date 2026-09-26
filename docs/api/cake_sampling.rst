.. _apicake_sampling:

flashinfer.cake_sampling
========================

``cake_sampling`` is a thread-block-cluster implementation of top-k-then-top-p sampling from
probabilities for Hopper and newer GPUs (compute capability 9.x, 10.x, 11.x and 12.x).  It is
measured on H100 (9.0), B200 (10.0), B300 / GB300 (10.3) and Rubin R200 (10.7); 11.x and 12.x are
compile targets that have not been run on hardware.  It fuses the three stages of
:func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` into two kernels per call:

1. a thread-block-cluster radix select that writes the exact per-row top-k slab, reducing
   the 2048-bucket histograms across the cluster through distributed shared memory; and
2. a programmatic-dependent-launch sparse top-p kernel that sorts the slab prefix, keeps the
   shortest prefix whose exclusive mass is below ``top_p`` times the top-k mass, renormalizes,
   and draws one token per row by inverse CDF from ``curand_init(seed, row, offset)``.

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
``batch * vocab`` launches run on the streaming stage-1 variants (a cluster of 1-4 CTAs walks the
row in register chunks), so there is no size-based fallback.  :func:`cake_sampling_route` reports the decision without launching.

The checked-in source product lives in ``csrc/cake_sampling/generated/`` as one translation
unit plus a manifest that records every frozen variant's launch resources; FlashInfer verifies
the source hash and compiles it once into a single fatbin with one ``-gencode`` per build target
(the kernels use only clusters, distributed shared memory, programmatic dependent launch and
``redux.sync``, so no per-architecture source exists).  Stage-1 variants whose dynamic shared
memory exceeds the device's opt-in limit are not dispatch candidates: on 12.x (99 KB) the
streaming variants drop out, so vocabularies above the register-resident capacity (196608)
take the ``top_k_first`` route there.

The stage-1 variant is chosen per call by a cost model whose single-wave CTA capacity table is
keyed by the device's SM count (148 for B200 / B300, 132 for H100; other devices use the nearest
measured table); the cost constants were fitted on B200.

Measured performance
--------------------

``python benchmarks/bench_cake_sampling.py --cupti --batches 1,8,16,32,64,128
--vocabs 32768,128256,151936,262144`` with ``--top-k 50`` / ``--top-k 1000`` and with
``--cuda-graph`` (``flashinfer.testing.bench_gpu_time``, CUPTI kernel time, p = 0.9, median µs).
B200 and B300 numbers are in the pull request that introduced the module (#5439).

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
     - 45.4 → 19.3 (2.35x)
     - 45.8 → 25.5 (1.80x)
     - 45.7 → 30.6 (1.49x)
     - 45.4 → 36.7 (1.24x)
   * - 32768
     - 8
     - 49.5 → 19.7 (2.52x)
     - 49.8 → 26.0 (1.91x)
     - 48.8 → 31.6 (1.55x)
     - 55.7 → 38.0 (1.47x)
   * - 32768
     - 16
     - 52.8 → 20.5 (2.57x)
     - 53.0 → 26.9 (1.97x)
     - 54.0 → 31.8 (1.70x)
     - 55.7 → 38.0 (1.47x)
   * - 32768
     - 32
     - 54.4 → 22.5 (2.41x)
     - 54.5 → 29.2 (1.87x)
     - 55.2 → 34.0 (1.63x)
     - 57.9 → 40.6 (1.43x)
   * - 32768
     - 64
     - 55.6 → 28.2 (1.97x)
     - 56.0 → 56.1 (1.00x)
     - 59.1 → 40.0 (1.48x)
     - 60.4 → 67.9 (0.89x)
   * - 32768
     - 128
     - 57.8 → 28.4 (2.04x)
     - 58.6 → 56.7 (1.03x)
     - 64.1 → 39.6 (1.62x)
     - 66.6 → 67.9 (0.98x)
   * - 128256
     - 1
     - 111.0 → 27.1 (4.10x)
     - 66.9 → 32.8 (2.04x)
     - 80.9 → 38.6 (2.09x)
     - 74.3 → 44.4 (1.67x)
   * - 128256
     - 8
     - 111.9 → 28.8 (3.89x)
     - 82.0 → 34.9 (2.35x)
     - 86.1 → 40.7 (2.12x)
     - 92.3 → 46.5 (1.98x)
   * - 128256
     - 16
     - 111.6 → 30.9 (3.61x)
     - 92.8 → 37.1 (2.50x)
     - 86.9 → 42.6 (2.04x)
     - 108.6 → 48.7 (2.23x)
   * - 128256
     - 32
     - 114.6 → 35.8 (3.20x)
     - 106.6 → 42.4 (2.52x)
     - 92.4 → 47.8 (1.93x)
     - 119.9 → 54.2 (2.21x)
   * - 128256
     - 64
     - 114.6 → 45.4 (2.52x)
     - 167.9 → 73.3 (2.29x)
     - 101.8 → 57.4 (1.77x)
     - 185.3 → 85.1 (2.18x)
   * - 128256
     - 128
     - 115.7 → 60.2 (1.92x)
     - 236.7 → 88.9 (2.66x)
     - 121.7 → 72.0 (1.69x)
     - 267.3 → 100.3 (2.66x)
   * - 151936
     - 1
     - 112.4 → 30.1 (3.73x)
     - 75.5 → 36.0 (2.10x)
     - 86.9 → 42.3 (2.05x)
     - 97.0 → 48.0 (2.02x)
   * - 151936
     - 8
     - 112.8 → 31.2 (3.61x)
     - 92.1 → 37.4 (2.46x)
     - 92.3 → 43.9 (2.10x)
     - 163.6 → 49.7 (3.29x)
   * - 151936
     - 16
     - 112.7 → 32.4 (3.48x)
     - 105.3 → 38.7 (2.72x)
     - 93.1 → 44.6 (2.09x)
     - 119.6 → 50.4 (2.37x)
   * - 151936
     - 32
     - 113.5 → 39.4 (2.88x)
     - 122.0 → 45.9 (2.66x)
     - 97.9 → 52.0 (1.88x)
     - 122.5 → 58.1 (2.11x)
   * - 151936
     - 64
     - 115.8 → 50.2 (2.30x)
     - 195.8 → 78.1 (2.51x)
     - 115.5 → 63.1 (1.83x)
     - 211.4 → 90.4 (2.34x)
   * - 151936
     - 128
     - 159.6 → 66.9 (2.39x)
     - 279.9 → 95.3 (2.94x)
     - 174.0 → 79.3 (2.20x)
     - 293.9 → 107.4 (2.74x)
   * - 262144
     - 1
     - 111.5 → 34.8 (3.20x)
     - 91.2 → 40.8 (2.23x)
     - 88.1 → 47.6 (1.85x)
     - 97.7 → 53.2 (1.84x)
   * - 262144
     - 8
     - 112.6 → 36.4 (3.10x)
     - 122.4 → 42.5 (2.88x)
     - 93.9 → 49.8 (1.88x)
     - 171.8 → 55.2 (3.11x)
   * - 262144
     - 16
     - 113.4 → 38.1 (2.98x)
     - 144.9 → 44.4 (3.26x)
     - 96.6 → 51.2 (1.89x)
     - 142.7 → 56.6 (2.52x)
   * - 262144
     - 32
     - 128.2 → 51.4 (2.49x)
     - 231.4 → 57.8 (4.00x)
     - 143.4 → 65.1 (2.20x)
     - 253.7 → 70.8 (3.58x)
   * - 262144
     - 64
     - 134.9 → 66.6 (2.03x)
     - 317.7 → 94.5 (3.36x)
     - 150.6 → 80.2 (1.88x)
     - 356.1 → 107.5 (3.31x)
   * - 262144
     - 128
     - 158.2 → 94.3 (1.68x)
     - 471.0 → 123.3 (3.82x)
     - 174.5 → 107.4 (1.63x)
     - 479.6 → 135.5 (3.54x)

Every H100 cell is faster than ``top_k_first`` except four at ``V = 32768, k = 1000``:
``B = 64`` and ``B = 128`` are ties under stream launch (1.00x / 1.03x) and slower under
CUDA-graph replay (0.89x / 0.98x).  There the stage-2/3 sort of a 1024-entry slab per row
dominates while ``top_k_first`` is cheap at that vocabulary; the ``joint`` rejection sampler is
faster than both at ``B = 1`` (as on B200).  Stage 1 on H100 has one residual dispatch gap: at
``V = 32768, B = 32-64`` the register-resident 2-CTA variant is 12-17 % slower than the
streaming 1-CTA variant, which the B200-fitted cost constants cannot express without regressing
``V = 128256, B <= 8``.

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
     - 29.8 → 15.9 (1.88x)
     - 30.6 → 21.1 (1.45x)
     - 32.3 → 24.8 (1.30x)
     - 37.4 → 30.0 (1.25x)
   * - 32768
     - 8
     - 33.2 → 17.4 (1.91x)
     - 33.5 → 22.2 (1.51x)
     - 42.1 → 25.6 (1.64x)
     - 39.1 → 30.8 (1.27x)
   * - 32768
     - 16
     - 35.5 → 17.2 (2.06x)
     - 35.2 → 22.5 (1.56x)
     - 37.8 → 26.5 (1.42x)
     - 42.6 → 31.6 (1.35x)
   * - 32768
     - 32
     - 36.8 → 18.0 (2.05x)
     - 37.2 → 23.1 (1.61x)
     - 45.3 → 26.7 (1.70x)
     - 43.1 → 31.5 (1.37x)
   * - 32768
     - 64
     - 38.4 → 19.4 (1.98x)
     - 38.3 → 24.6 (1.56x)
     - 45.6 → 28.2 (1.62x)
     - 45.0 → 33.3 (1.35x)
   * - 32768
     - 128
     - 39.0 → 20.4 (1.91x)
     - 39.3 → 32.0 (1.23x)
     - 45.8 → 28.9 (1.58x)
     - 46.6 → 40.4 (1.15x)
   * - 128256
     - 1
     - 70.4 → 22.3 (3.16x)
     - 56.1 → 27.3 (2.06x)
     - 69.3 → 31.7 (2.19x)
     - 74.3 → 36.3 (2.04x)
   * - 128256
     - 8
     - 70.1 → 23.9 (2.94x)
     - 68.1 → 28.6 (2.38x)
     - 73.8 → 32.0 (2.31x)
     - 74.9 → 37.8 (1.98x)
   * - 128256
     - 16
     - 70.1 → 25.1 (2.79x)
     - 76.2 → 30.2 (2.53x)
     - 76.7 → 33.9 (2.26x)
     - 100.0 → 39.4 (2.54x)
   * - 128256
     - 32
     - 72.0 → 26.1 (2.76x)
     - 84.2 → 31.1 (2.70x)
     - 76.0 → 34.8 (2.19x)
     - 84.3 → 39.7 (2.13x)
   * - 128256
     - 64
     - 72.2 → 30.1 (2.39x)
     - 91.6 → 35.5 (2.58x)
     - 76.4 → 38.9 (1.96x)
     - 99.2 → 44.3 (2.24x)
   * - 128256
     - 128
     - 72.2 → 38.4 (1.88x)
     - 138.9 → 50.5 (2.75x)
     - 77.5 → 47.4 (1.64x)
     - 149.2 → 59.0 (2.53x)
   * - 151936
     - 1
     - 69.8 → 24.5 (2.85x)
     - 64.2 → 29.4 (2.18x)
     - 71.2 → 34.4 (2.07x)
     - 83.7 → 39.4 (2.13x)
   * - 151936
     - 8
     - 70.3 → 26.3 (2.67x)
     - 77.3 → 30.9 (2.50x)
     - 77.8 → 34.8 (2.24x)
     - 86.0 → 39.4 (2.18x)
   * - 151936
     - 16
     - 70.0 → 26.5 (2.64x)
     - 87.6 → 31.6 (2.77x)
     - 80.5 → 35.9 (2.24x)
     - 106.8 → 40.2 (2.66x)
   * - 151936
     - 32
     - 69.9 → 27.4 (2.55x)
     - 95.9 → 32.4 (2.95x)
     - 78.8 → 36.9 (2.14x)
     - 108.9 → 41.7 (2.61x)
   * - 151936
     - 64
     - 72.1 → 33.0 (2.19x)
     - 107.2 → 38.4 (2.79x)
     - 80.3 → 42.3 (1.90x)
     - 118.9 → 47.1 (2.52x)
   * - 151936
     - 128
     - 73.5 → 43.3 (1.70x)
     - 162.1 → 55.1 (2.94x)
     - 89.9 → 52.2 (1.72x)
     - 171.2 → 63.2 (2.71x)
   * - 262144
     - 1
     - 70.1 → 28.4 (2.47x)
     - 80.1 → 33.6 (2.39x)
     - 76.4 → 38.0 (2.01x)
     - 129.4 → 43.4 (2.98x)
   * - 262144
     - 8
     - 70.1 → 30.1 (2.33x)
     - 103.0 → 34.8 (2.96x)
     - 78.3 → 39.0 (2.01x)
     - 109.7 → 43.3 (2.53x)
   * - 262144
     - 16
     - 70.1 → 30.5 (2.30x)
     - 119.5 → 35.6 (3.36x)
     - 81.0 → 39.9 (2.03x)
     - 114.2 → 44.1 (2.59x)
   * - 262144
     - 32
     - 70.2 → 31.8 (2.21x)
     - 134.3 → 36.8 (3.65x)
     - 78.8 → 41.2 (1.91x)
     - 153.1 → 45.6 (3.36x)
   * - 262144
     - 64
     - 87.8 → 41.7 (2.10x)
     - 196.4 → 47.1 (4.17x)
     - 100.2 → 51.0 (1.96x)
     - 218.8 → 56.3 (3.88x)
   * - 262144
     - 128
     - 107.2 → 61.4 (1.75x)
     - 295.6 → 73.2 (4.04x)
     - 119.7 → 70.4 (1.70x)
     - 307.4 → 81.8 (3.76x)

Rubin R200 was measured with an internal CUDA 13.5 toolkit that lists ``compute_107`` (driver
620.43); every cell is faster than ``top_k_first`` (1.15x-4.17x).  The wave table used there is
the B200 one (nearest SM count); no Rubin-specific re-fit was needed.

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
