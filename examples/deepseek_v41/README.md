# Experimental DeepSeek V4.1 index scoring

`flashinfer.deepseek_v41.deepseek_v41_index_scores_fp32` consumes packed MXFP4
Q and a paged index cache. It computes a score for each token:

```
score[b,j] = sum_h weights[b,h] * relu(dot(dequant(Q[b,h]), dequant(K[b,j])))
```

H=32, D=128, MXFP4 scale groups of 32. Dot products and head reduction use
FP32; weights and output are BF16. With `candidates[B,C]`, output column
`8*c+i` corresponds to token `8*candidates[b,c]+i`. Candidate order and
duplicates are preserved; invalid IDs/pages and invisible tokens return
negative infinity. C ranges from 1 to 2048; pages contain 32, 64 or 128 tokens.
Initialize referenced pages, including the unused slots in a partial block.
See the API docstring for exact byte layout, alignment and workspace sizing.

- `backend="triton"` (default): native MXFP4 Tensor Cores, adaptive candidate
  tiles of 128/256/512 tokens; supports both full-context and candidate scoring.
- `backend="cute_dsl"`: large-batch candidate scoring, CUTLASS DSL 4.7 or newer.
  Reuses Dhiraj Reddy's native FP4 scorer from FlashInfer [#4365](https://github.com/flashinfer-ai/flashinfer/pull/4365) and
  [#4737](https://github.com/flashinfer-ai/flashinfer/pull/4737), adapted
  from TensorRT-LLM and DeepGEMM. This extension adds H32/block8 gather, packed
  validity, cooperative cp.async gathers and fixed-width scheduling. H32 uses
  one math warpgroup to drain both accumulators, reusing its weight vector.
  It retains the native MMA, scale layouts and FP32 head-reduction recipe.
  Existing copyright and Apache 2.0 attribution remain.

Call the API explicitly to opt in. Warm it before graph capture; supply `out`
and an independent `workspace` per concurrent CuTe call for allocation-free
replay. The backend caches compiled code, not input-derived GPU metadata.
The optional scratch is rebuilt from the current candidates, lengths and page
table on every invocation. Each scorer gathers scales from its own KV cache.
No AOT or automatic-backend registration.

For layers that explicitly share candidates, page mapping and visibility,
`prepare_deepseek_v41_candidate_metadata` publishes an owned snapshot once.
Pass it to `deepseek_v41_candidate_scores_fp32` with each layer's Q, cache and
weights. It uses the same CuTe scoring kernel. Raw input changes do not alter
an existing snapshot: call preparation again with `out=metadata` to republish
without allocation. The example below exercises both interfaces.

Order consumers after preparation on the same stream or with events. A
completed snapshot can serve concurrent readers with distinct outputs;
republish only after all readers finish. The fixed batch, candidate count,
page size, physical page count, context and device must match. Layer cache
addresses and padded strides may differ. There is no implicit metadata cache
or assumption that arbitrary model layers share the same candidates.

This PR supplies scoring only. It needs already-packed inputs and does not
provide quantization, cache update, paging allocation, RoPE or TopK selection.
The existing cache/provider and selector can be used independently. Run the
constant-input example with `python examples/deepseek_v41/indexer.py`.

## Reproducible performance and correctness

```
python benchmarks/bench_deepseek_v41_indexer.py --output results.json
# Optional CUDA 13+ CUPTI verification of the same captured graph:
python benchmarks/bench_deepseek_v41_indexer.py --cupti --output cupti-results.json
# Separately labeled prepared consumer; publication is outside its timing:
python benchmarks/bench_deepseek_v41_indexer.py --backends cute_dsl,cute_prepared --cupti --output prepared-results.json
pytest tests/experimental/test_deepseek_v41_indexer_fp32.py
```

The `triton_fixed` benchmark arm reproduces the previous draft tile policy
with the same native MXFP4 math; it is not another shipped backend.
The benchmark uses model-shaped synthetic inputs from the pinned
[DeepSeek V4.1 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277),
not model weights or serving traces. It checks an independent FP64 oracle,
changed-input graph replay, exact invalid masks and output guards before
recording timing. JSON includes raw paired samples and source hashes. Timing
includes CuTe metadata preparation, masking and scoring, with allocation and
JIT excluded. These are scorer measurements, not serving E2E or TopK timings.
Prepared-consumer measurements must separately state whether metadata
publication is included and how many consumers reuse it. The optional
`cute_prepared` benchmark arm uses the public snapshot API, explicitly
republishes changed inputs before correctness checks, and times only its
consumer. Both arms use one layer's cache per case and ten calls per graph;
this comparison isolates per-call preparation overhead, not a serving model.

Support is currently SM100/SM103. This change was locally executed on an
SM100 NVIDIA Graphics Device with 148 SMs; SM103 was not independently tested.
Tests on unsupported devices skip and do not establish target-GPU correctness.
Owner/tracking/graduation plan: @YangXu1990uiuc, #5115. The API is experimental
and may change as serving integration and target-hardware CI are completed.
