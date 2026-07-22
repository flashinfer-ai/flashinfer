# Engine integration sketches for the unified paged-prefill API

Two reviewable diffs showing what vLLM and sglang delete and gain by moving
their paged-prefill path to `flashinfer.attention.unified` (draft PR #4015).

**Verification status — read this first.**  These diffs have NOT been run
against the live engines.  What IS machine-checked, on real GPUs, is the
data flow they rely on: `tests/attention/test_unified_prefill_engine_shapes.py`
replicates each engine's exact metadata pipeline (vLLM: pinned CPU mirrors +
async-H2D twins, dense block table, rebased mixed-batch prefill slice,
preallocated out; sglang: page_size=1 token-CSR with the +256 over-allocated
index tail, preallocated `(max_bs+1,)` indptr buffers, NHD pool,
`seq_lens_cpu` mirrors) and drives the unified API with it under a
zero-sync guard, against an independent fp32 oracle.

Friction found while writing these (also logged in the test docstring):
- fa2/fa3 paged `(192,128)` head dims require `k_page_stride ==
  v_page_stride` — separately-allocated K/V pools violate it, so the
  capability matrix does not declare it for the fa family (cudnn covers it).
- Everything else mapped 1:1; neither engine needs to build any metadata it
  does not already own.
