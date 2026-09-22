.. _apicake_sampling:

flashinfer.cake_sampling
========================

``cake_sampling`` is a Blackwell (B200/GB200 SM100a, B300/GB300 SM103a) implementation of
top-k-then-top-p sampling from probabilities.  It fuses the three stages of
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
non-contiguous rows, ``batch * vocab >= 2**24`` (where the persistent FlashInfer radix top-k is
faster), or a non-Blackwell device.  :func:`cake_sampling_route` reports the decision without
launching.

The checked-in source product lives in ``csrc/cake_sampling/<arch>/`` with one manifest per
architecture that records every frozen variant's launch resources; FlashInfer verifies the
source hash before JIT compilation.

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
