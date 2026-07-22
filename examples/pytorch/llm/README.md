# FlashInfer LLM Example (Llama / Qwen dense)

A self-contained, plain-PyTorch LLM decoder built from FlashInfer's
serving-path ops.

**Scope.** Serving models independently is explicitly *not* a goal of this
example — production inference belongs to the serving frameworks that build
on FlashInfer. What *is* the goal, and what we consider paramount for the
library's robustness, is being able to **close the loop independently**: a
functional end-to-end verification path that exercises FlashInfer's public
APIs the way a serving stack composes them — paged KV cache, batched
prefill/decode attention, RoPE, RMSNorm, SwiGLU activation, and sampling —
with nothing but `flashinfer`, `torch`, and a Hugging Face checkpoint.
Integration-level regressions (a broken JIT cache, per-step recompiles,
silent backend fallbacks) show up here even when every kernel unit test
passes, and can be reproduced and bisected inside this repo without any
external framework in the dependency chain. Background and roadmap:
[`docs/design_docs/e2e_pytorch_llm_examples.md`](../../../docs/design_docs/e2e_pytorch_llm_examples.md).

## FlashInfer APIs exercised

| API | Where |
|-----|-------|
| `BatchPrefillWithPagedKVCacheWrapper` (plan/run, causal, ragged batch) | prompt prefill |
| `BatchDecodeWithPagedKVCacheWrapper` (plan/run, tensor cores) | token-by-token decode |
| `append_paged_kv_cache`, `get_batch_indices_positions` | paged KV maintenance every step |
| `apply_rope_pos_ids_inplace` / `apply_llama31_rope_pos_ids_inplace` | rotary embedding (incl. Llama-3.1 scaling) |
| `rmsnorm`, `fused_add_rmsnorm` | pre/post-attention norms, Qwen3 per-head q/k norm |
| `silu_and_mul` | SwiGLU FFN |
| `sampling.top_k_top_p_sampling_from_logits` | non-greedy sampling |

## Supported models

Any dense Hugging Face checkpoint with `model_type` ∈ {`llama`, `qwen2`,
`qwen3`}. Weights are read straight from safetensors (BF16 compute). Sizing
guide (single GPU): Qwen3-0.6B…32B and Llama 1B…8B fit comfortably on one
modern data-center GPU; see the design doc for a full B200 fit table.

| Tier | Model | Notes |
|------|-------|-------|
| Smoke test (fast) | `Qwen/Qwen3-0.6B` | default; ungated download |
| Real model | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B` | ungated |
| Llama family | `meta-llama/Llama-3.1-8B-Instruct` | gated (HF license); exercises llama3 RoPE scaling |

## Usage

```bash
# Batch greedy completion with the built-in prompts
python generate.py --model-id Qwen/Qwen3-0.6B --max-tokens 32

# Chat template + sampling
python generate.py --model-id Qwen/Qwen3-8B --chat \
  --prompt "Explain KV cache paging in two sentences." \
  --temperature 0.7 --top-k 50 --top-p 0.9 --max-tokens 128

# Steer the attention backends (forwarded to the wrapper constructors)
python generate.py --prefill-backend fa2 --decode-backend fa2
```

`generate.py` prints the completions, coarse prefill/decode timing (this is a
correctness harness — the timing includes JIT warmup and per-step host work;
do not read it as a benchmark), and machine-readable `[smoke] key=value`
lines used by the smoke test.

## Smoke test

```bash
python smoke_test.py --model-id Qwen/Qwen3-0.6B --max-tokens 16
```

Runs generation in two fresh processes and asserts:

1. the second process compiles **zero** JIT modules (on-disk kernel cache is
   actually reused — catches "cache silently broken, everyone recompiles");
2. no JIT build ever happens during steady-state decode steps;
3. greedy outputs are identical across the two runs;
4. every request produced tokens.

Exit code 0/1, suitable for CI. To force a fully cold first run:
`rm -rf ~/.cache/flashinfer` first.

## Files

| File | Purpose |
|------|---------|
| `modeling.py` | config/weight loading, paged KV cache, decoder forward pass |
| `generate.py` | CLI generation loop (prefill → decode → sample) + JIT-build counter |
| `smoke_test.py` | two-process verification harness (see above) |

## Requirements

`flashinfer`, `torch`, and — for checkpoint/tokenizer handling only —
`transformers`, `huggingface_hub`, `safetensors`. No inference framework is
imported; model weights download to the standard HF cache (`HF_HOME`).
