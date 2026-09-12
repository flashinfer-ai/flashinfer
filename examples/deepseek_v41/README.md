# Experimental Frost DeepSeek V4.1 decode

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
