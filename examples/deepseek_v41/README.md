# Experimental DeepSeek V4.1 decode

The default `backend="cute-dsl"` retains the original Frost API below.
Explicit `backend="prims-ts"` selects the separate native sparse MLA API described
in [Prims-TS native sparse MLA](#prims-ts-native-sparse-mla). There is no automatic
selector. Calling the experimental API is the opt-in; both paths are JIT-only.

## Existing Frost API

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
system toolkit above. Its FP4 path emits `cvt.rn.satfinite.e2m1x2.f32`.
The writer accepts SM100, SM103 and SM107; Frost attention remains SM100-only.
These are tested toolchain versions, not a package-wide CUDA version requirement.
SM107 also needs Triton's selected `ptxas` to recognize `sm_107a`; CuTe's bundled
compiler does not satisfy this separate requirement. With Triton 3.8.0, set
`TRITON_PTXAS_BLACKWELL_PATH` to a compatible installed assembler before importing
Triton if its default assembler is too old.

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

## Prims-TS native sparse MLA

The new explicit backend has a different argument and ownership contract from
the existing Frost API; arguments are forwarded unchanged to the selected
implementation. No BF16-Q conversion adapter or fallback is provided. Experimental
imports remain function-local, with no AOT or trace registration. This adapter
does not replace main's generic prepared-list sparse MLA implementation.

### Scope and public API

Native scope is **SM100 / SM103 / SM107, H64 / Sq1 / D512**, batch size
1–65535, page size64, 128 SWA positions and main width `K=0,64,...,512`.
It requires CuTe-DSL >=4.8 with its bundled CUDA13 compiler and support for the
device's compile target, independently of `CUDA_HOME`.

- Q is contiguous `torch.float8_e4m3fn[B,1,64,512]`.
- SWA is typed E4M3, width512, with no scale plane.
- Optional main storage is packed uint8 FP4, width288, with the existing
  separate payload/scale page planes.
- Native decode defaults to **HND** `[pages,1,64,width]`; explicit NHD uses
  `[pages,64,1,width]`. Singleton-head normalization is zero-copy.
- Decode returns only BF16 output with Q's shape; supplied `out` is returned
  by identity. No native public plan or LSE is returned.

The optional writer adds `format="swa_fp8"`: clamp to[-448,448] in cache units,
then nearest-even E4M3 rounding with signed zero/subnormal preservation. The
writer retains its original **NHD** default. Request `kv_layout="HND"` explicitly
for both SWA and main writers to match native decode's default, or consistently
select NHD for decode. Apply model-required RoPE before quantization.

```python
import torch
from flashinfer.deepseek_v41 import (
    deepseek_v41_decode,
    deepseek_v41_decode_workspace_size,
)

# Q/caches and metadata are already prepared on the same CUDA device.
workspace = torch.empty(
    deepseek_v41_decode_workspace_size(B, K),
    dtype=torch.uint8,
    device=query.device,
)
out = torch.empty(query.shape, dtype=torch.bfloat16, device=query.device)
result = deepseek_v41_decode(
    query,
    swa_kv_cache,
    workspace,
    sparse_indices=sparse_indices,
    compressed_kv_cache=compressed_kv_cache,
    sparse_topk_lens=sparse_topk_lens,
    seq_lens=seq_lens,
    out=out,
    bmm1_scale=512**-0.5,
    bmm2_scale=1.0,
    sinks=sinks,
    kv_layout="HND",
    backend="prims-ts",
)
assert result is out
```

The workspace-size helper is pure arithmetic and accepts only Prims-TS, which
is also its default. Allocate contiguous int8/uint8 storage with a **32-byte-aligned**
base and sufficient capacity; insufficient capacity is an error, not a hidden
allocation fallback. Other public buffers must be contiguous and16-byte aligned.
Eager calls may omit `out`; capture may not. The native window helper takes
`(query, swa_kv_cache, workspace_buffer, sparse_indices=None,
sparse_topk_lens=None, seq_lens=None, out=None, bmm1_scale=1.0, bmm2_scale=1.0,
sinks=None, kv_layout="HND", backend="prims-ts")` and uses K0.

### Live metadata and common units

All three metadata tensors are required contiguous CUDA int32 on Q's device:

- `sparse_indices[B,128+K]`: first128 physical SWA slots, then K physical main
  slots. K>0 requires main storage. Holes and duplicate multiplicity are preserved.
- `sparse_topk_lens[B]`: includes the fixed128 SWA positions. Main visibility
  clamps `topk-128` to `[0,K]`, not `topk-valid_swa_count`.
- `seq_lens[B]`: SWA visibility is `[0,clamp(seq_lens,0,128))`, as explicit
  V4.1 policy, not an equivalence claim about other model-specific kernels.

Invalid, INT32_MAX sentinel and out-of-capacity IDs mask safely. Live metadata
refreshes on every call/replay without CPU synchronization. Every valid cache
slot must be initialized and causally visible; callers own slot mapping and RoPE.

Both caches must already use the same cache units. BMM1 is a positive finite
FP32 Python scalar, BMM2 a finite FP32 Python scalar. For Q descale `q_s`, shared
KV descale `s>0` and attention factor `alpha`, pass `bmm1_scale=alpha*q_s*s`
and `bmm2_scale=s`. Defaults are1.0/1.0; the example above explicitly selects
`512**-0.5` with unit descales. Main staging rounds/saturates exact
`E2M1 * original_group16_E4M3_scale` directly to E4M3, including intentional
+/-448 saturation. There is no request amax, selected scale tensor, division
by BMM2 or repeated scale restoration. Inconsistent cache units are not repaired;
this convention does not establish equivalence to official token/block64 quantization.

### Kernel, capture and validation boundary

One joint finalizing attention uses Keep-AB/M64/BK128/retained-KV8/split1/
nonpersistent scheduling, inline metadata and optional FP32 per-head sinks.
There is no prepare kernel, per-source state or merge. Workspace contains
private natural-log, sink-exclusive FP32 LSE and the K0 dummy main storage.
Sinks contain finite logits or -inf and affect output normalization exactly once;
empty rows produce exact zero output and -inf LSE. Effective Q/K/V must be finite.

Capture requires caller output/workspace and eager warmup for the same declaration.
Keep addresses alive/fixed and update values in place. Changing Python scales
requires recapture, not recompilation. Concurrent calls need separate writable
buffers or synchronization. Output/workspace cannot overlap inputs or each other;
read-only inputs may alias. Cached factories retain kernels and host facts, not
caller buffers. Training, speculative/multi-token queries, architectures outside
SM100/SM103/SM107, and V4 varlen/PDL/RopeQuant features are out of scope.

Native tests retain the independent represented-input FP64 oracle, componentwise
FP8 output bound, LSE `rtol=0, atol=1e-5`, exact-empty/nonfinite checks and bitwise
layout/replay parity. Host contracts cover sparse views, workspace ownership,
defaults and task control; static/CLC graph coverage remains in
`tests/attention/test_attention_ts_sparse_mla.py`.
The native finalizer retains ordinary-range arithmetic and uses two factors for
extreme BMM2/sink combinations, avoiding premature underflow before PV scaling.
Regression tests cover strong sinks with large signed BMM2, signed zero,
BF16-subnormal output, and live metadata/layout/replay parity. The independent
oracle and numerical gates above are unchanged; true BF16 overflow is not removed.
Historical native-only frozen-source verification on **B200 (SM100) passed
197 host-contract and 100 GPU cases** without failures or skips, using PyTorch
2.11.0+cu128, CuTe-DSL 4.8.0 and bundled CUDA 13.4. Those counts predate the
rebase, host-test cleanup and architecture-guard changes.

After the architecture-guard changes, **B200 (SM100) passed 330 CPU contract
and 180 GPU cases** without failures or skips. The GPU run covered the complete
DS4.1 decode/cache test files, including Frost and Prims-TS. A fresh process
passed 16 native cache-reload cases, and the native mixed-cache/window-only
example passed. This run used PyTorch 2.14.0+cu130, CuTe-DSL 4.8.0, bundled CUDA
13.4 and Triton 3.8.0. These B200 results do not qualify SM103 or SM107.

On **R100 (SM107)** with the same package versions, the complete DS4.1 decode/cache
test files passed **133 cases**, with **47 expected SM100-only Frost skips** and
no failures. The native mixed-cache/window-only example passed. A fresh process
passed **16 native cache-reload cases** without skips; all **15 persisted CuTe
objects** retained identical paths, sizes, mtimes and SHA256 hashes. CuTe selected
`sm_107a` without an architecture override.

This R100 validation used an existing CUDA **13.4.59 `ptxas`** for Triton and a
matching **13.4.59 NVRTC/builtins pair** for the independent Torch references.
The runtime's older assembler rejected `sm_107a`, and its older NVRTC could not
compile the reference's `torch.exp2`. Process-scoped toolchain overrides fixed
those toolchain failures without changing kernels, references or numerical gates.
NVRTC symbol providers were verified in both global and Torch-bridge scopes,
along with architecture 107 support and an exact CUDA `torch.exp2` smoke test.
No packages were installed or images downloaded. CuTe, Triton's assembler and
Torch's NVRTC are separate compiler routes.

On **GB300 (SM103, aarch64)** with the same package versions, the complete DS4.1
decode/cache test files passed **133 cases**, with **47 expected SM100-only Frost
skips** and no failures. The native mixed-cache/window-only example passed. A
fresh process passed **16 native cache-reload cases** without skips; all **15
persisted CuTe objects** retained identical paths, sizes, mtimes and SHA256 hashes.
CuTe selected `sm_103a` without an architecture override; the SM107-specific
assembler/NVRTC overrides were not used.

Cold compilation was slow in this ARM container. The successful run restored
five CuTe objects from an earlier partial run and completed the GPU test phase
in about **2 hours 20 minutes**, within a four-hour allocation. This is validation
wall time, including JIT compilation, not GPU kernel latency. No kernels,
references or numerical gates were changed to make this target pass.

The generic sparse MLA suite, complete integration, checkpoint/full-model
equivalence, rare-path cost, performance and backend crossovers remain unqualified
by these runs.

Native example and benchmark usage (the benchmark was not run):

```bash
python examples/deepseek_v41/decode.py --backend prims-ts
python benchmarks/bench_deepseek_v41_decode.py \
  --backends prims-ts --batches 1,4,16,32,64,128,256 \
  --context 32768 --output prims_ts.json
```

The native example owns Q quantization, common KV units, explicit HND caches,
workspace and output. The benchmark uses explicit NHD and checks each arm against
its own represented-input FP64 reference before graph/event timing. Native-only
runs do not require importing the other decoder for provenance. JSON records
source/toolchain fingerprints, representations, numerical contracts and samples.
Timing covers public GPU replay, excluding cache writing, allocation, JIT, capture
and host overhead; these are not model-token latencies or measured speedups.
