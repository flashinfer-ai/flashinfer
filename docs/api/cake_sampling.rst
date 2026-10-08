.. _apicake_sampling:

flashinfer.cake_sampling
========================

``cake_sampling`` is a thread-block-cluster implementation of top-k-then-top-p sampling from
probabilities for Hopper and newer GPUs (compute capability 9.x, 10.x, 11.x and 12.x).  It is
measured on H100 (9.0), B200 (10.0), B300 / GB300 (10.3) and Rubin R200 (10.7); 11.x and 12.x are
compile targets that have not been run on hardware.  It fuses the three stages of
:func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` into at most two kernels per call:

1. a thread-block-cluster radix select that writes the exact per-row top-k slab (``[batch, 1024]``
   values and indices plus a per-row count): one 11-bit cluster pass reduces the 2048-bucket
   histograms through distributed shared memory, the entries of the selected bucket are gathered
   into every CTA as 64-bit ``(key, ~index)`` composites and the remaining key bits and boundary
   ties are resolved with local passes (rows whose bucket overflows the gather capacity take the
   exact three-pass cluster path); and
2. a programmatic-dependent-launch sparse top-p kernel that sorts the slab prefix with one
   composite-key bitonic network, keeps the shortest prefix whose exclusive mass is below
   ``top_p`` times the top-k mass, renormalizes, and draws one token per row by inverse CDF from
   ``curand_init(seed, row, offset)``.

Launch forms
------------

Every form below produces bitwise identical outputs; the choice is timing only and the host makes
it once per ``(device, batch, vocab, top_k_max)`` (the decision is memoised, so the route check and
the launch share it).

* **Fused two-warp tail** (``top_k_max <= 64``, ``fused_tail_kcap`` in the manifest): stage 2/3
  runs on two warps of the cluster's first CTA inside the stage-1 kernel; one launch.  Every
  variant the dispatcher can pick for such a top-k carries the tail (manifest ``fused_tail``);
  the ``(8, 48)`` resident, a large-k pick, is built without it.
* **Whole-CTA tail** (``64 < top_k_max <= 1024``, ``fused_block_tail_kcap``; launch flag bit 3):
  stage 2/3 runs on the whole first CTA (512 threads, two slab entries each) in a separate build
  of the variant (manifest ``fused_block_tail``, symbol suffix ``_bt``).  Taken for a cluster-8
  streaming variant with ``top_k_max > 768`` on a one-wave grid on compute capability 9.0, 10.0
  and 10.3 (10.3 up to ``vocab = 196608``); every other large-k launch takes the chain.  Rows of
  such a launch whose k is at most 64 take the two-warp tail.
* **Two-launch chain** (everything else): the stage-2/3 kernel is PDL-chained behind stage 1.
  Stage 1 signals ``griddepcontrol.launch_dependents`` early (bit 1) only when the batch fits on
  the SMs its last wave leaves free; a streaming variant signals before its first pass on
  compute capability 10.x (bit 2) and after its filter pass on Hopper.
* **Build selection** (host side; the kernel never sees these bits): bit 4 selects a streaming
  variant's coarse-sample build (``_cs``: the sampled first pass reads 1/8 of the row instead of
  1/4), bit 6 its speculative-sample build (``_sp``: the first register chunk doubles as the
  sample), bit 7 the slab-tail form of either sample build (``_cs_lb`` / ``_sp_lb``: a fused
  launch's selected pairs are pushed into the first CTA's shared memory for the tail), bit 8 the
  pushed-coarse-sums form of the default build (``_lg``: coarse histogram sums are stored into
  every CTA's shared memory for the two-level select, two-launch chains only), bit 9 the
  leader-push exchange form of the default or sample build of a multi-CTA stream (``_lp`` /
  ``_cs_lp`` / ``_sp_lp``: every CTA stores its compacted candidate list straight into the first
  CTA's receive buffer and its length into every CTA before the single exchange barrier, so the
  pull form's DSM read rounds and exit rendezvous disappear; it displaces bits 7 and 8), bit 10
  the CTA-local select form of a leader-push sample build (``_cs_lp_l1`` / ``_sp_lp_l1``: each CTA
  picks its filter bucket from its own sample, so the cluster-wide coarse-histogram round
  disappears and the push is the only cluster round; fused launches with the smallest top-k, up to
  32 on compute capability 10.3, 20 on 10.0 and 10 on 9.0; on 10.0 and 10.3 it also carries the
  leader push onto the two-chunk ept-32 rows the bit-9 chunk rule excludes -- at any batch on 10.0,
  on 10.3 when the second chunk is full or the batch has at least four rows), bit 11 the integer-tested form of the whole-CTA
  tail build (``_bt_tia``: the tail's f64 target and sample tests run as the stage-2/3 integer
  emulation; compute capability 10.3, whose FP64 pipe is slow) and bit 5
  the row-span filter arm of a cluster-8 stream above the two-warp tail.  Each build is taken only on
  the capabilities, cluster sizes and row lengths where it measured faster; the policy constants
  live in :mod:`flashinfer.cake_sampling`.
* **Stage-2/3 static forms**: the stage-2/3 kernel exists in three forms that differ only in
  instruction selection -- the base form (f64 top-p tests, max / min bitonic exchange), a form
  whose bitonic exchange is one 64-bit compare and select (``one_cmp_select``, compute capability
  9.0 and 10.0) and a form that adds integer top-p tests (``int_tests`` with the former, 10.3).
  Every other device runs the base form.  The manifest lists each slab once per form with its
  ``features``, ``variant_flags`` and the ``capabilities`` that dispatch it,
  :func:`stage23_variant_flags` returns the device's form and the binding selects the kernel by
  ``(threads, items, variant_flags)``.

Dispatch and fallback
---------------------

Requests the frozen kernels cannot serve are dispatched to the ``top_k_first`` route of
:mod:`flashinfer.sampling` (same semantics, same generator advancement):
``probs`` not a CUDA ``float32`` tensor with contiguous rows (``stride(0) == vocab``), no
``top_k``, ``top_k`` outside ``[1, vocab)`` or above the slab (1024), a device outside the build
targets, or a vocabulary no frozen variant covers.  The build targets are FlashInfer's
``CompilationContext`` targets (``FLASHINFER_CUDA_ARCH_LIST`` when set, otherwise the capabilities
of the visible devices) restricted to the supported majors 9-12; a device whose architecture is
not among them takes the ``top_k_first`` route instead of failing at launch.  Streaming variants
serve any vocabulary up to 2^21; frozen variants whose dynamic shared memory exceeds the device's
opt-in limit are not dispatch candidates, so on 12.x (99 KB) the streaming variants drop out and
vocabularies above the register-resident capacity (196608) take the ``top_k_first`` route there.
:func:`cake_sampling_route` reports the decision without launching.  When ``top_k`` is a tensor
and ``top_k_max`` is not given, the host reads ``top_k.max()`` (one device synchronization).

The stage-1 variant is chosen per call by a cost model whose single-wave CTA capacity table and
constants are keyed by the device's SM count (148 for B200 / B300, 132 for H100, 212 for Rubin
R200; other devices use the nearest measured table).  On the 212-SM table a small-k (``top_k_max``
at most 64 or unknown) ept-32 streaming pick whose grid and the cluster-8 grid both fit one wave is
re-picked to the cluster-8 ept-16 stream (measured 8-15 % faster on R200 at batch <= 16; multi-wave
grids, large k and the other tables keep the ranked pick).  ``renorm_out`` and ``workspace`` expose the
sorted slab of a call.

Source product
--------------

The frozen source lives in ``csrc/cake_sampling/generated/``: one translation unit
(``cake_sampling_kernels.cu``, which includes the ``cake_sampling_kernels_part<N>.cuh`` body
files in order; every file stays under the repository's 5 MiB limit) plus ``manifest.json``,
which records every frozen variant's launch resources and build flags, the capabilities each
stage-2/3 form is dispatched on, the slab geometry and the file names.
:mod:`flashinfer.jit.cake_sampling` renders the manifest into the binding's variant tables and
compiles the unit once into a single fatbin with one ``-gencode`` per build target.  The module
name is content-derived (``cake_sampling_`` + 20 hex digits sealed over the frozen source parts,
the manifest, the binding header and the rendered binding, computed once per process), so a JIT
build or an installed AOT artifact of another bundle revision is never loaded for this one.

Correctness contract
--------------------

The route is held to a tolerance-level contract (the kernels may change bits between releases;
the precision level and the selection exactness may not):

* **Top-k set**: equal to the float64 sorted reference for every row and shape, including the tie rule (ties on the
  key are broken by the lower index: the 64-bit ``(key, ~index)`` composite is a total order, so every exact
  selection yields the same set and the same slab order).
* **Top-p cut**: equal to the float64 reference, except when the exact cumulative mass lies within ``eps = 1e-6``
  of ``top_p`` times the top-k mass, where the cut may differ by one element (``tests/utils/test_cake_sampling.py``
  places rows on that boundary and checks both sides).
* **Sample**: always inside the exact support (the kept prefix); run-to-run deterministic for identical inputs,
  ``philox_seed`` and ``philox_offset`` (CUDA-graph replay, concurrent streams and repeated invocation included);
  multi-seed next-token histograms within the 99 % binomial band of the exact distribution per row class.
  A call captured into a CUDA graph without explicit Philox parameters never reads the generator inside the
  capture (PyTorch would register it with the graph and replay two ``FillFunctor`` kernels per launch, +46 us per
  replay on GB300); it uses the generator's initial seed with a host-side offset that differs between captures.
* **Renormalized slab** (``renorm_out``): fp32 with ``rtol 1e-6``, ``atol 1e-7`` against the float64 reference.
  Keys stay exact fp32 bit patterns; every accumulation is at least fp32 (no bf16 / fp16 anywhere); no tolerance is
  loosened to admit a kernel change.
* Every stage-1 build of a variant (``_bt``, ``_cs``, ``_sp``, ``_cs_lb``, ``_sp_lb``, ``_lg``, ``_lp``, ``_cs_lp``,
  ``_sp_lp``) and every stage-2/3
  form is additionally gated on bit-identity with the default build on every tested row.  A change that moves bits
  must document exactly which rows can differ (ties, the eps boundary) and why.
* The exact fallbacks stay kernel-side: a candidate list that overflows the gather capacity takes the three-pass
  cluster path and a row with fewer than k candidates is kept whole; vocabularies above 2^21 take the reference
  ``top_k_first`` route.

Benchmark
---------

``python benchmarks/bench_cake_sampling.py --cupti --skip-joint --batches 1,2,4,8,16,32,64,128
--vocabs 32768,128256,151936,262144 --top-k {10,50,1000} [--cuda-graph]`` times the ``top_k_first``
route and this route with ``flashinfer.testing.bench_gpu_time`` (CUPTI, cold L2).  The measured
tables of each frozen bundle are part of the pull request that landed it.

.. currentmodule:: flashinfer.cake_sampling

.. autosummary::
    :toctree: ../generated

    top_k_top_p_sampling_from_probs
    top_k_probs_to_slab
    cake_sampling_route
    choose_stage1
    choose_stage23
    stage23_variant_flags
