# Tools (run on the C3 devbox unless noted)

All paths are under `/scratch/ziangli/flashinfer-w4a16-tp-low-concurrency` on the devbox (`../remote_exec.sh` from the
project root); `../sync_source.sh` copies the candidate tree, `../sync_source.sh control` the pristine upstream tree.

| Tool | Purpose |
|---|---|
| `fc2_unit_sweep.py` | Kernel-level GEMM2 (FC2) harness. Real NVFP4 weights + `moe_sort` routing, runs FC1 once, then times `_run_grouped_gemm` for FC2 with `m_tiles_per_unit` forced per arm (`1`, `auto`, integers) via a monkeypatch of `_select_m_tiles_per_unit`. Every arm's sentinel-prefilled whole output buffer must equal arm `1` bit for bit. CUPTI, cold L2, CUDA graph. |
| `run_unit_sweep.sh`, `run_unit_sweep_r2.sh` | One sweep per GPU over the PR 7290 TP shapes (+ GLM-5); outputs `results/sweep/r1`, `r2`. |
| `summarize_sweep.py` | Per (shape, route tile, cluster) tables and geomeans (tokens >= 4); fails loudly on any bitwise mismatch. |
| `codegen_identity.py` | Compiles representative specializations in one tree with `CUTE_DSL_KEEP=ptx,cubin`, records normalized PTX/SASS hashes (`.loc`/`.file` dropped); `--compare control/receipt.json candidate/receipt.json` requires identity for one-tile-unit kernels. |
| `run_tests.sh` | Candidate pytest groups (one process per GPU) + MegaMoE W4A16 + cross-tree differential; outputs `results/tests/<run>`. |
| `diff_vs_control.py` | Whole-MoE outputs at model dims (6 presets x 10 token counts x 6 tactics, PDL on) saved per tree; `--compare` requires bit equality; `--oracle` checks samples against the independent W4A16 oracle. |
| `bench_w4a16_only.py` | Runs the unchanged `benchmarks/bench_moe_deepseek.py` for CuteDSL W4A16 only (PR 7290 observer trick: `profile_backend="cute-dsl-w4a16"` as a selection filter), recording raw CUPTI samples and the tactic per token count. |
| `caac.py` | PR 7290-style campaign: `prepare` (immutable tree snapshots + hashes), `warm` (serial JIT/artifact warm-up), `job` (tune control and candidate, then C1 / A-frozen / A-native / A-native / A-frozen / C2 on one GPU, fresh process per phase, private cache copies, replay must not rewrite caches), `summary` (sqrt(C1*C2) vs sqrt(A1*A2), geomeans by band). |
| `restructure_epilogue.py` | One-shot source transformation used during development (wrapping the non-gated epilogue in the unit loop). Kept for provenance. |

Environment notes: artifacts are read from the shared SHA-verified cache `/root/.cache/flashinfer/cubins`
(`FLASHINFER_CUBIN_DIR`, `FLASHINFER_NO_DOWNLOAD=1`) because the public artifact server returned 502s during the campaign.
CUPTI Python 13.5.0 / `nvidia-cuda-cupti` 13.5.25 attach correctly on driver 595.91.07 (no event fallback).
