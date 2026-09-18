# Experimental DeepSeek V4.1 kernels

## Frost decode

The mixed-cache and window-only decode APIs use the **Frost** implementation
on SM100, written in CuTe DSL with `cutlass.experimental.primitives` and
adapted from Mengyu Guo's CuTe DSL HCA ([#3943](https://github.com/flashinfer-ai/flashinfer/pull/3943),
[#4368](https://github.com/flashinfer-ai/flashinfer/pull/4368)). Original attribution
and licenses are retained. MMA, TMEM and cache memory operations use
`cutlass.experimental.primitives`. FlashMLA informed the single-CTA resource
budget, deferred softmax rescaling, WS QK/PV layouts with partial-score
exchange, fused softmax/correction and shared-memory/TMA output. The original
HCA foundation and these FlashMLA design contributions are acknowledged in
the source.

Run `python examples/deepseek_v41/decode.py` on SM100. The validated environment
uses CUTLASS DSL 4.7, PyTorch 2.13+cu130 and CUDA toolkit 13.2. Decode is JIT-only
and does not require the external `flash_mla` package. The included optional
cache writer makes the example self-contained. CuTe uses its bundled
compiler (CUDA 13.3 in this validation), independently of `CUDA_HOME`. With
bundled CUDA >= 13.2, compressed-cache decoding uses native FP4-to-BF16 and
packed BF16 multiplication. Older bundled compilers retain the existing
conversion sequence inside the same kernel. Packed conversion uses two small
`prims.inline_ptx` helpers (CUTLASS 4.7 lacks a typed FP4-to-BF16 wrapper);
WS MMA also uses public `prims.inline_ptx` to work around CUTLASS 4.7's typed
WS wrapper; other MMA/TMEM operations use the typed primitives.

The optional Triton cache writer was validated with **Triton 3.7.1** and its
bundled Blackwell `ptxas` **CUDA 13.1.80**, separate from the CuTe compiler and
system toolkit above. Its FP4 path emits `cvt.rn.satfinite.e2m1x2.f32` and is
limited to SM100. These are tested toolchain versions, not a package-wide
CUDA version requirement.

Frost keeps the measured split schedule for small batches and selects
WS QK/PV with fused softmax/correction for K512/B128+. When that batch exceeds
the device's SM count, CTAs process multiple requests through the same pipeline.
Dequant warps prefetch the next request's indices and compute cache
offsets in registers while the current request finishes. An existing boundary
barrier protects publication of this metadata into shared memory, where it
avoids repeated address calculation during gathers.
When both complete cache pools fit below 32 GiB, metadata uses 32-bit offsets
in 16-byte units; larger pools retain 64-bit byte offsets. This preserves support
for cache addresses above 2 GiB while reducing shared metadata traffic.
This is internal scheduling in one implementation and requires no API selector.

Decode currently accepts contiguous BF16 queries `[B,1,64,512]`, 64-token pages,
128 MXFP8 window slots and up to 512 FP4 compressed slots, padded to multiples
of 64. Window-only decode uses the same kernel with no compressed stream.
Physical slot `-1` and out-of-capacity slots are masked without reading cache
storage. Every valid cache slot must be initialized and causally visible.

Q/K/V and probabilities use BF16; accumulation and softmax use FP32.
The numerical gate is independent FP64 relative-L2 < 0.005, max-scaled error < 0.01,
and absolute LSE error < 1e-5. Checkpoint accuracy has not been established.

`deepseek_v41_decode` returns `(out, lse, plan)`. Output is BF16 with Q's shape;
LSE is FP32 `[B,64,1]`, natural-log and **sink-exclusive**. A finite per-head
sink logit or `-inf` is accepted. The sink contributes only to output's
normalization; an empty row returns zero output and `-inf` LSE. Applying the
sink at final normalization preserves LSE for very large sink logits.

Prepare once eagerly and reuse the plan for allocation-free CUDA Graph replay.
Plan-owned output/LSE/workspace are overwritten on every call. Ordinary plan
calls may use new input addresses if shape, strides, dtype and device match.
A captured graph retains the pointers recorded during capture: keep input
addresses stable and update values in place during replay. Changing addresses
requires graph recapture or an appropriate graph update; keep captured buffers
alive for the graph's lifetime. A plan must not run concurrently on multiple
streams, and its buffers must not alias
inputs. Training, speculative multi-token queries, SM103 and full-model E2E
validation are outside this initial scope.

`deepseek_v41_quantize_cache` is an optional Triton helper for producing these
two cache formats from contiguous BF16/FP32 `[N,512]` rows. It fuses quantization
and writing into page64 storage; `out` plus int32 `slots[N]` supports incremental
updates without allocation. Negative and out-of-capacity slots skip publication.
Valid slots must be unique: this is a caller precondition, not checked by the
helper. Unwritten slots remain unchanged. Apply
model-required RoPE before quantization. This helper is provided for integration
and reproducible examples, with no competitive-performance claim. Decode does
not call it and accepts compatible cache bytes from any producer.

Within each 64-token page, all data rows precede all scale rows. The opaque
`[pages,64,1,width]` view does not mean data and scales are interleaved per token:

| Cache | Data bytes/token | Scale bytes/token | Scale format |
|---|---:|---:|---|
| Main FP4, width 288 | 256 E2M1 | 32 | E4M3, group16, no global scale |
| Window MXFP8, width 528 | 512 E4M3 | 16 | E8M0, group32 |

The caller supplies physical slot IDs. For a logical token ID `t` in request
`b`, the usual mapping is `block_table[b, t // 64] * 64 + t % 64`, with invalid
entries mapped to `-1`. Slot mapping, RoPE, indexing and GEMM quantization remain
outside this decode API.

The reproducible GPU benchmark checks both outputs against FP64 before timing:

```bash
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --output frost.json
# Optional comparison requires an externally built FlashMLA installation:
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --flashmla --output paired.json
```

`--flashmla` requires a FlashMLA build with DS4.1 MXFP8/FP4 mixed-cache support
and the `FlashMLASchedMeta` API. `get_mla_metadata()` returns an empty scheduler
object and a `None` placeholder; the first eager call initializes it from the
fixture before graph capture. The benchmark creates fresh metadata per shape.

It reports warm CUDA Graph/event decode+merge GPU latency. Quantization, JIT,
allocation, graph capture and CPU overhead are excluded. These component
measurements are not model token latency or throughput.

## MXFP4 index scoring

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

The index-scoring APIs need already-packed inputs and do not
provide quantization, cache update, paging allocation, RoPE or TopK selection.
The existing cache/provider and selector can be used independently. Run the
constant-input example with `python examples/deepseek_v41/indexer.py`.

### Reproducible performance and correctness

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
