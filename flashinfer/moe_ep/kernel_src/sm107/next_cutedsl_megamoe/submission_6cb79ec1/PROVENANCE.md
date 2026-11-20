# SM107 NVFP4 serving snapshot

Kernel source: `cutedsl_megamoe` at `6cb79ec1e9b68966a2b558d371e5cc7c23d0900c`.
Exported with that revision's `export_src.py --kernels RubinInferenceMegaMoE RubinInferenceGenphaseMegaMoE`.
The packaged-weight runtime under `../shim/submission_*.py` preserves the
submission launch policy, fused staging, and static tactics. The existing
FlashInfer generic launcher retains its independently patched vendor export.
