# SM90 CuTeDSL MegaMoE source provenance

The current implementation combines the FP8 tree merged by #4688
(`9a791d724f382673eec291e018e7eca2f845832e`, final PR head
`c969e2c19f4404a507d9c153cd217d16fe475008`) with the local fused Humming
MXFP4 overlay, on main including the FP8 update from #5338
(`28fae4f2e0ce6b7d9230f828291ca96aa0473ec9`). Both PRs are already in main;
the older pinned base below is historical provenance, not an outstanding
dependency.

## Current local overlay

- Packed Humming E2M1/K32 weights and hybrid FP8 activations share the FP8
  dispatch, scheduling and reduction implementations.
- Guarded paired BF16 FC2 stores, row-address reuse, K256 offset bulk copies
  and zero-count elision preserve the documented completion/reset protocol.
  Optional tail-N8 and segmented-ready strategies have separate narrow domains.
- Main's local 16-byte FC2 store is shared by compatible MXFP4 return modes.
  Its tail-pair scheduler is reused with packed-weight and offset adaptation;
  MXFP4 requires cluster `(1, 2, 1)` and whole-tile readiness. The normal
  MXFP4 tuner checks its bounded tail-pair catalog against each model's
  geometry and protocol requirements. H7168/I3072/E384/EP4 retains its 38
  candidates and their order; default buckets remain unchanged.
- Tiny positive activation scales retain a finite quantization multiplier
  independently of the rounded dequant scale. Normal/zero arithmetic is kept.
- Shared communication capabilities serve both precisions; FP8's new pair/skip
  defaults stay off. MXFP4's established domain remains, with compact pull and
  training output off. Main's FP8 N8, generate_c and compact-pull behavior stays.
- The current MXFP4 runtime uses fused execution and validates tactics and
  cache identities against the current supported domain. Ordinary library
  CUDA Graph replay is covered separately from direct benchmark launches.
- MXFP4 K128/K256 compute retains its qualified scale-promotion and WGMMA
  wait order; unreachable experimental alternatives and selectors are removed.
- The shared benchmark uses direct CUDA-event launches and the existing
  `--cga M,N` cluster override for both FP8 and MXFP4.

The defaults, weight ABI, numerical contract and current commands are in
`TUNING.md`. Tests cover independent references, full outputs, workspace reuse,
ordinary Graph replay, strategy guards and cache isolation; the PR separately
records which GPU cases and performance comparisons completed. The original source revisions above identify
the FP8 base; the MXFP4 changes are maintained in this PR.

## Updating this tree

The PR diff records the MXFP4 changes on top of main. For a new upstream
kernel drop, update the four packages under `src/` together (`common`, `src`,
`moe_nvfp4_swapab`, `moe_hopper_fp8`), then reapply the local changes described
above. Record the new upstream revision and test FP8/MXFP4 outputs, workspace
reuse, ordinary Graph replay and tuning/cache behavior before updating the
results in `TUNING.md`.
