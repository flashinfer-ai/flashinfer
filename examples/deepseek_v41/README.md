# Experimental DeepSeek V4.1 components

The mixed-cache and window-only decode APIs use one SM100 CuTe DSL backend,
adapted from Mengyu Guo's CuTe DSL HCA ([#3943](https://github.com/flashinfer-ai/flashinfer/pull/3943),
[#4368](https://github.com/flashinfer-ai/flashinfer/pull/4368)). Original attribution
and licenses are retained. MMA, TMEM and cache memory operations use
`cutlass.experimental.primitives`. FlashMLA informed the single-CTA resource
budget and deferred softmax rescaling; no FlashMLA code was copied.

Run `python examples/deepseek_v41/decode.py` on SM100. The validated environment
uses CUTLASS DSL 4.7, PyTorch 2.13+cu130 and CUDA toolkit 13.2. Decode is JIT-only
and does not require the external `flash_mla` package. The cache examples use the
separate experimental cache/quantization component. CuTe uses its bundled
compiler (CUDA 13.3 in this validation), independently of `CUDA_HOME`. With
bundled CUDA >= 13.2, compressed-cache decoding uses native FP4-to-BF16 and
packed BF16 multiplication. Older bundled compilers retain the existing
conversion sequence inside the same kernel. Packed conversion uses two small
`prims.inline_ptx` helpers (CUTLASS 4.7 lacks a typed FP4-to-BF16 wrapper);
the MMA/TMEM pipeline remains expressed through primitives.

Decode currently accepts contiguous BF16 queries `[B,1,64,512]`, 64-token pages,
128 MXFP8 window slots and up to 512 FP4 compressed slots, padded to multiples
of 64. Window-only decode uses the same kernel with no compressed stream.
Physical slot `-1` and out-of-capacity slots are masked without reading cache
storage. Every valid cache slot must be initialized and causally visible.

Q/K/V and probabilities use BF16; accumulation and softmax use FP32. This is an
explicit change from the earlier draft's FP32-probability Triton recipes:
`deepseek_v41_decode_fp32`, `deepseek_v41_decode_bf16x3`, and the decode
`backend`/`arithmetic` arguments have been removed. The numerical gate for this
BF16-P path is independent FP64 relative-L2 < 0.005, max-scaled error < 0.01,
and absolute LSE error < 1e-5. Checkpoint accuracy has not been established.

`deepseek_v41_decode` returns `(out, lse, plan)`. Output is BF16 with Q's shape;
LSE is FP32 `[B,64,1]`, natural-log and **sink-exclusive**. A finite per-head
sink logit or `-inf` is accepted. The sink contributes only to output's
normalization; an empty row returns zero output and `-inf` LSE. Applying the
sink at final normalization preserves LSE for very large sink logits.

Prepare once eagerly and reuse the plan for allocation-free CUDA Graph replay.
Plan-owned output/LSE/workspace are overwritten on every call. Tensor values
and addresses may change if shape, strides, dtype and device match. A plan
must not run concurrently on multiple streams, and its buffers must not alias
inputs. Training, speculative multi-token queries, SM103 and full-model E2E
validation are outside this initial scope.

The reproducible GPU benchmark checks both outputs against FP64 before timing:

```bash
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --output cute.json
# Optional comparison requires an externally built FlashMLA installation:
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --flashmla --output paired.json
```

It reports warm CUDA Graph/event decode+merge GPU latency. Quantization, JIT,
allocation, graph capture and CPU overhead are excluded. These component
measurements are not model token latency or throughput.
