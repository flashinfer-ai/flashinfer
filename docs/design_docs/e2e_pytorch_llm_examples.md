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
- **2026-07-21** — `--chat` mode validated on B200 (Qwen3-8B, greedy,
  96 tokens): correct answers to factual questions, EOS stop confirmed,
  ~404 tok/s batch-3 decode. One portability fix: transformers v5 changed
  `apply_chat_template`'s tokenized return type, so the example now renders
  the template to text and tokenizes explicitly.
- **2026-07-22** — Phase 2 (MoE) validated on B200. `reference_check.py`
  results vs transformers eager (bf16, tolerance 0.05 rel-L2):
  dense 0.008–0.010, MoE 0.007–0.039, top-1 agreement at every length;
  tiny-MoE decode loop deterministic. The check caught a real bug on its
  first run: transformers v5 moved rope config into nested
  `rope_parameters` (and renamed the expert count to `num_local_experts`),
  so the old flat-schema read silently used rope_theta=1e4 for v5-saved
  checkpoints — a with-length-growing logits divergence (0.16→0.64) that
  the HF-bf16-vs-fp32 baseline (~0.007, flat) proved was ours. Older
  checkpoints (all real HF models tested) were unaffected. Also measured:
  the `fused_moe_100` module is by far the heaviest JIT unit (267 objects,
  ~1 h cold compile at 30-way parallelism on the shared node).
- **2026-07-22** — Phase 3 TEP validated on 2× B200 (umbriel-b200-040):
  `torchrun --nproc_per_node=2 reference_check.py` matches the single-GPU
  transformers reference at identical quality (dense rel-L2 0.0077–0.0089,
  MoE 0.0080–0.0383, top-1 agreement everywhere) — TP attention sharding,
  EP expert dispatch (`ep_size/ep_rank` + global routing), and both
  allreduce points are numerically correct. `health_check.py` passes in
  both modes with `jit_builds_cold=0` — the NFS-mounted JIT cache
  (ops note below) served a fresh container on a different node with zero
  recompiles. Tiny-model TEP throughput is (expectedly) below single-GPU
  (decode 1601 vs 1847 tok/s) since allreduce latency dominates at toy
  sizes; real-model TEP numbers are a phase-4 measurement.

## Roadmap

- **Phase 1 (done):** dense single-GPU example (Qwen3 0.6B–32B,
  Llama-family), paged-KV prefill/decode, sampling, smoke harness; validate
  on B200.
- **Phase 2 (in progress):** `qwen3_moe` support through
  `fused_moe.cutlass_fused_moe` (BF16 SwiGLU experts, external
  softmax→top-k→renorm routing mirroring the HF reference), plus
  `reference_check.py`: tiny random-weight dense/MoE checkpoints built with
  `transformers`, our last-token prefill logits asserted against
  transformers eager within a relative-L2 tolerance. The tiny MoE exercises
  the identical kernel path as Qwen3-30B-A3B, so no large download is needed
  for functional verification. Remaining: FP8/FP4 weight-quantized linear
  backends.
- **Phase 3 (in progress):** TEP (TP=EP) multi-GPU — attention
  tensor-parallel, MoE expert-parallel over the same ranks, two NCCL
  allreduces per layer, launched with `torchrun`; `reference_check.py`
  validates the sharded result against the single-GPU transformers
  reference. Follow-ups: swap NCCL allreduce for `flashinfer.comm`
  (trtllm/vllm allreduce) as a selectable backend; 4-GPU runs for the
  70B–235B tier.
- **Phase 4 (started):** health instrumentation + performance gates.
  `health_check.py` measures the serving lifecycle (import, load, cold
  prefill incl. JIT builds, warm prefill/decode throughput, steady-state
  recompiles, optional autotune delta) and emits JSON — the gate inputs.
  Planned next: CUDA-graph capture of the decode step
  (`use_cuda_graph=True` wrapper mode with preallocated page tables),
  threshold files + trend comparison, then CI wiring of
  smoke/reference/health checks. Multi-node is explicitly out of scope for
  now.

### Ops note: persistent JIT cache for GPU-node runs

Node containers are ephemeral; a container-local `~/.cache/flashinfer`
loses the ~1 h `fused_moe_100` compile on every teardown. Runs now mount an
NFS-backed cache (`flashinfer2/var/jit_cache` →
`/home/aleyang/.cache/flashinfer`), so any node/container reuses compiled
artifacts.
