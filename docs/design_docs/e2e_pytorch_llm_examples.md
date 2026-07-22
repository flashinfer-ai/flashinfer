# End-to-End PyTorch LLM Examples

Tracking doc for the `examples/pytorch/llm/` effort.

## Motivation

FlashInfer's test suite is kernel-centric: each op is validated in isolation
against a reference. What we lack is a *self-contained* end-to-end exercise of
the library — a real model forward pass that strings together attention
(paged KV cache, prefill + decode), RoPE, RMSNorm, activation, GEMM, and
sampling the way a serving stack would. Integration-level regressions are
invisible to unit tests but very visible to users, e.g.:

- JIT/cubin cache not being hit → kernels silently recompiling on every
  process start (or worse, every call);
- backend dispatch quietly falling back to a slow/generic path on some
  SM version;
- plan/run wrapper API drift that only shows up when prefill, append, and
  decode are used together across steps;
- dtype/layout mismatches between ops that each pass their own unit tests.

Serving frameworks that embed FlashInfer do catch some of this, but through
their own abstraction layers and on their own release cadence. These examples
let us close the loop *within this repo*: a plain-PyTorch reference usage of
the public API that we can run on any machine with weights from Hugging Face
and no external inference framework in the dependency chain. They are
**reference/verification code, not a serving engine** — for production
inference, use a real serving stack. The point is that when something breaks
end to end, we can reproduce and bisect it here with nothing but
`flashinfer + torch + transformers(tokenizer/config only) + safetensors`.

This extends the existing `examples/pytorch/` direction (the WAN diffusion
example) to the LLM serving path, which is FlashInfer's core use case.

## Non-goals

- Production-grade throughput (no continuous batching scheduler, no
  overlapping, no speculative decoding). Batching exists only to exercise the
  batched kernel paths.
- Competing with or benchmarking against inference frameworks.
- Covering every model architecture. We pick a small set that maximizes
  kernel-path coverage per line of example code.

## What fits on a B200 node

Working numbers: one B200 has 192 GB HBM3e (~180 GB usable). A node gives us
up to 4 GPUs ≈ 768 GB nominal. Weights-only footprint below; add ~10–20 % for
KV cache + activations at modest batch/context.

### Single GPU (1× B200, 192 GB)

| Model | Params | Weights (BF16) | Notes |
|---|---|---|---|
| Qwen3-0.6B / 1.7B / 4B | 0.6–4 B | 1.2–8 GB | fast smoke-test tier; ungated |
| Llama-3.2-1B / 3B, Llama-3.1-8B | 1–8 B | 2–16 GB | gated on HF (license click-through) |
| Qwen3-8B / 14B / 32B (dense) | 8–32 B | 16–64 GB | ungated; good single-GPU "real model" tier |
| Qwen3-30B-A3B (MoE) | 30 B total / 3 B active | ~60 GB | exercises fused MoE path on one GPU |
| Mixtral-8x7B | 47 B | ~94 GB | classic MoE |
| GPT-OSS-20B / 120B | 20 B / 117 B | ~12 / ~60 GB (native MXFP4) | exercises MXFP4 weight path |
| Llama-3.3-70B | 70 B | ~140 GB BF16 (tight) / ~70 GB FP8 | FP8 comfortable on one GPU |

### 2–4 GPUs (TP)

| Model | Weights | Fits? |
|---|---|---|
| Qwen3-235B-A22B (MoE) | ~235 GB FP8 / ~470 GB BF16 | FP8 on 2 GPUs, BF16 on 4 |
| Llama-3.1-405B | ~405 GB FP8 | 4 GPUs, tight but viable at small batch |
| DeepSeek-V3 / R1 (671B) | ~671 GB native FP8 | **no** at 4 GPUs once KV + activations are counted; needs a full 8-GPU node |
| Kimi-K2 (1T) | > 1 TB | no |

Conclusion: everything we need for phase 1 (dense GQA + MoE + FP8/FP4 weight
paths) runs on **one** B200; 4 GPUs only become interesting when we add
tensor-parallel examples (phase 3).

## Example design

Directory: `examples/pytorch/llm/`

| File | Purpose |
|---|---|
| `modeling.py` | Llama/Qwen-family decoder built from FlashInfer ops: `BatchPrefillWithPagedKVCacheWrapper`, `BatchDecodeWithPagedKVCacheWrapper`, `append_paged_kv_cache`, `apply_rope_pos_ids`, `rmsnorm`/`fused_add_rmsnorm`, `silu_and_mul`. Weights loaded straight from HF safetensors (no `AutoModel`). |
| `generate.py` | CLI: prompt(s) → prefill → decode loop → text, with FlashInfer sampling ops; `--max-tokens`, `--batch`, greedy or top-k/top-p. |
| `smoke_test.py` | The verification harness (see below). |
| `README.md` | Usage + backend/env-var knobs, mirroring the style of `examples/pytorch/README.md`. |

Model support keyed off HF `config.json` `model_type` ∈ {`llama`, `qwen2`,
`qwen3`} — same skeleton (RMSNorm → GQA attn with RoPE → SwiGLU FFN), small
per-family deltas (Qwen3 q/k per-head norm, Qwen2 QKV bias, Llama-3.1 RoPE
scaling).

### The verification harness (`smoke_test.py`)

This is the actual point of the exercise. Checks, each cheap and scriptable:

1. **Correctness sanity** — greedy decode of a fixed prompt must be
   deterministic across runs and produce a plausible continuation. Optionally
   compare prefill logits against a `transformers` eager forward (tolerance
   check) when `--reference` is passed.
2. **JIT/cache hygiene** — run generation in a fresh subprocess twice; the
   second run must not invoke `nvcc`/`cicc` (asserted via
   `FLASHINFER_LOGGING_LEVEL=DEBUG` JIT logs) and its module-load phase must
   be dramatically faster. This is the "kernel cache silently broken" canary.
3. **No recompiles across steps** — within one process, decode steps after
   warmup must not trigger new JIT builds (plan/run reuse working).
4. **Backend dispatch** — the selected attention/GEMM backends actually run
   (no silent fallback), asserted via `FLASHINFER_LOGLEVEL` API logs.

## Status log

- **2026-07-21** — doc created; upstream main synced (`b7cd951d`). Phase 1
  in progress: dense Qwen3/Llama single-GPU example + smoke harness, to be
  validated on a computelab B200.
- **2026-07-21** — first B200 deployment (Qwen3-0.6B) surfaced two findings
  on day one, validating the approach:
  1. *Environment class:* a source checkout without the (new) `3rdparty/cccl`
     submodule fails every attention JIT compile — the JIT include path puts
     the vendored CCCL first and `fastdiv.cuh` now needs
     `cuda::fast_mod_div`, so an empty `3rdparty/cccl` silently falls back to
     the CUDA toolkit's older libcudacxx and nvcc errors out. Anyone
     updating an existing checkout across that upstream change hits this.
  2. *Harness calibration:* `JitSpecNvcc.try_load()` returns `None` for
     JIT-path modules *by design* (artifact freshness is delegated to
     ninja's dependency scan), so `build()` runs in every fresh process and
     ninja no-ops on a warm cache. "Recompile" must therefore be measured as
     "built artifact changed across `build()`", which is what the harness
     now does; raw `build()` invocations are reported separately as the
     warm-start overhead metric (`jit_build_calls`).
- **2026-07-21** — Phase 1 validated on B200 (computelab, CI cu130 image):
  - Cold cache: Qwen3-0.6B smoke run 1 compiled 4 modules (batch_prefill,
    rope, page, silu_and_mul; 57.8 s), run 2 compiled **0** (6.4 s) —
    cross-process cache reuse confirmed; greedy outputs identical; zero
    steady-state builds. `smoke_test.py` exit 0.
  - Qwen3-8B (36 layers): coherent batch-3 completions, ~316 tok/s decode
    (uninstrumented loop, not a benchmark), zero compiles — same kernel URIs
    as 0.6B (head_dim 128, BF16).
  - Note: decode with `use_tensor_cores=True` shares the prefill module, so
    the dense BF16 path needs only 4 JIT modules end to end.

## Roadmap

- **Phase 1 (this change):** dense single-GPU example (Qwen3 0.6B–32B,
  Llama-family), paged-KV prefill/decode, sampling, smoke harness; validate
  on B200.
- **Phase 2:** MoE variant (Qwen3-30B-A3B) exercising `fused_moe`; FP8/FP4
  weight-quantized linear via the shared `flashinfer_modules.py` backends.
- **Phase 3:** tensor-parallel (2–4 GPU) variant exercising
  `flashinfer.comm` all-reduce; unlocks the 70B–235B tier.
- **Phase 4:** wire the smoke harness into CI (it's designed to be a
  single-command pass/fail).
