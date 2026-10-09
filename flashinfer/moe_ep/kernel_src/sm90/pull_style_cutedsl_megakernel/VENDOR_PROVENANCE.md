# SM90 CuTeDSL MegaMoE source provenance

The current implementation combines the FP8 tree merged by #4688
(`9a791d724f382673eec291e018e7eca2f845832e`, final PR head
`c969e2c19f4404a507d9c153cd217d16fe475008`) with the local fused Humming
MXFP4 overlay, on main including the FP8 update from #5338
(`28fae4f2e0ce6b7d9230f828291ca96aa0473ec9`). Both PRs are already in main.

## Local MXFP4 changes

The Humming MXFP4 weight path reuses the shared FP8 dispatch, scheduling
and reduction code. The PR diff records the source changes;
[TUNING.md](TUNING.md) documents supported configurations, tuning,
measurements and reproduction commands.

## Updating this tree

The PR diff records the MXFP4 changes on top of main. For a new upstream
kernel drop, update the four packages under `src/` together (`common`, `src`,
`moe_nvfp4_swapab`, `moe_hopper_fp8`), then reapply the local changes described
above. Record the new upstream revision and test FP8/MXFP4 outputs, workspace
reuse, ordinary Graph replay and tuning/cache behavior before updating the
results in `TUNING.md`.
