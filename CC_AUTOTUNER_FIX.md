# CC-safe autotuner timing (`%globaltimer`)

Single-purpose patch: make FlashInfer's `AutoTuner` time candidate tactics correctly under NVIDIA Confidential Computing (CC). Pure code change in `flashinfer/autotuner.py`; no behavior change off-CC.

## Problem

`AutoTuner.choose_one` ranks candidate tactics purely by `min(measured_time)`, and `pure_profile` measures that time with `cudaEvent` / `cudaEventElapsedTime`. Under CC (memory-encryption / bounce-buffer path) `cudaEventElapsedTime` is **unreliable — it can return negative values**. The min-time ranking then degenerates to a **near-random tactic pick per rank**, which gets baked into the on-disk tuning cache and used for the entire serving run. For Qwen3.5-397B TP4 this produced inconsistent, slow MoE-GEMM tactics and a measurable throughput/TPOT loss vs CC-off.

## Fix

Time the candidate run with the GPU's **`%globaltimer`** register instead of CUDA events. A tiny stamp kernel (PTX `mov.u64 %0, %%globaltimer`) is JIT-built once per process; `pure_profile` brackets the timed `graph.replay()` / kernel run with two stamps and returns the elapsed ns → ms. `%globaltimer` is monotonic and CC-safe, so timings are clean and the tactic pick is stable across ranks and shape buckets. The return value and signature are unchanged, so `choose_one` and the cache format are untouched.

Mirrors the TRT-LLM autotuner fix — **TensorRT-LLM PR #11657** ("time autotuner tactics with `%globaltimer` under CC") — and sglang's CC detection.

## Control

`FLASHINFER_AUTOTUNE_TIMER`:
- `auto` (default) — use `%globaltimer` only when CC is detected, else the legacy `cudaEvent` timer. **Off-CC runs are byte-for-byte unchanged.**
- `globaltimer` — force `%globaltimer`.
- `cudaevent` — force the legacy timer (revert).

CC detection mirrors sglang's `is_confidential_compute()` (NVML `nvmlSystemGetConfComputeState`, `ccFeature != 0`). Override with `FLASHINFER_CONFIDENTIAL_COMPUTE=1/0` (useful for CI or boxes without `pynvml`). If the stamp kernel fails to build, it logs a warning and falls back to `cudaEvent`.

When active, the autotuner logs once: `[Autotuner]: using %globaltimer timing (CC-safe).`

## Scope

- One file: `flashinfer/autotuner.py` (+103 lines: detection, lazy stamp-kernel build, and the timing branch in `pure_profile`).
- No new dependencies beyond what FlashInfer already uses (`torch`, optional `pynvml` for detection).
- Probes, call-tree docs, and smoke tests used to develop this live on separate branches, intentionally excluded here.
