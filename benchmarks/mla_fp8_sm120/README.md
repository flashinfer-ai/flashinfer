# FlashInfer MLA on SM120: GLM-4.7-Flash study

GLM-4.7-Flash MLA kernel study and experimental FP8×FP8 CUDA implementation,
tested on an NVIDIA RTX PRO 5000 72GB Blackwell (SM120).

- [新增 FlashInfer 派生 CUDA FP8：decode 与 prefill](native_fp8/README.zh.md)
- [原 BatchMLAPagedAttentionKernel、GLM head 配置与精度支持](README.zh.md)
- [前期 Triton FP8 KV 实验](fp8_kv_performance.zh.md)
- [FlashMLA 与 FlashInfer 的区别及本机兼容性](flashmla_vs_flashinfer.zh.md)

The native FP8 implementation reuses FlashInfer's paged work metadata, MMA and
memory-layout helpers, and stable split-KV merge algebra. Its scheduler and
CUDA specialization live in [the experimental backend](../../flashinfer/experimental/mla_fp8_sm120/); the installed FlashInfer package is
unchanged. This is an experimental implementation, not an official FlashInfer
backend or a deployed SGLang integration.

Both decode and causal/noncausal prefill are supported for absorbed MLA inputs
with Q/K dimension 576 and latent output dimension 512. FP8 E4M3 QK/PV use FP32
accumulation and softmax, with BF16 outputs. Measurements use synthetic inputs
and report operator latency, not whole-model throughput or task accuracy.

Tested environment:

| Component | Version |
|---|---|
| GPU | RTX PRO 5000 72GB, SM120 |
| CUDA toolkit | 12.9 |
| PyTorch | 2.11.0+cu129 |
| FlashInfer | 0.6.15.post1 |
| Triton | 3.6.0 |
| Matplotlib | 3.11.1 |
| nvidia-ml-py | 13.610.43 |

From this directory, using an environment with these dependencies and `nvcc`:

```bash
python native_fp8/wrapper.py  # Compile only; no attention is run.
python native_fp8/check.py
python native_fp8/bench.py --samples 30
python native_fp8/summarize.py
```

The native test and benchmark scripts wait for GPU idle windows and preserve
existing processes. Compilation products are cached under `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/experimental/mla_fp8_sm120` (default base: the user home). Benchmarks
resume completed cases from `/tmp/mla_fp8_sm120_results.json`; use `--output NAME.json`
for a fresh run. Low-level `NativeMLA` calls leave resource scheduling to the
caller.

FlashInfer-derived source is covered by the included
[Apache-2.0 license](native_fp8/LICENSE). The earlier study's links to installed
FlashInfer/SGLang source refer to the original test machine; reproducible source
hashes are retained in the provenance JSON. The original source snapshots can be obtained from the FlashInfer 0.6.15.post1 wheel.

The latency tables are the original 0.6.15.post1 measurements, **not measurements
of the current 0.7.2 branch**. By default the submitted backend compiles against
the headers in this checkout. Set `FLASHINFER_MLA_FP8_INCLUDE_DIR` to the 0.6.15.post1
wheel's `flashinfer/data/include` directory to reproduce the earlier header
selection. Final checkout verification is recorded separately in
[native_fp8/checkout_validation.json](native_fp8/checkout_validation.json).

This is a source-only development branch. The standalone loader imports just the
backend so it can coexist with an installed baseline without reinstalling
FlashInfer or changing a serving environment. There is no stable wrapper dispatch,
AOT registration, or automatic backend selection. See the backend's
[development tracking note](../../flashinfer/experimental/mla_fp8_sm120/TRACKING.md).

The native result JSON keeps the measurements and selected configurations;
`tuning_trials/*.json` holds every tuning candidate, referenced by `trials_file`.
This split preserves all accepted raw data while keeping each file reviewable.
