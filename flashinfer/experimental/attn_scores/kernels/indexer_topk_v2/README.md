# Indexer-throwaway v2

This private CuTe DSL package implements sampling, candidate scoring, exact
selection and recovery. Use `flashinfer.FP8MQATopKWrapper(backend="v2")` with
contiguous K and separate scales. The public paged wrapper owns one continuous
wrapper and gathers K/scales before delegating Full, selective v1 or selective
v2. Its preallocated gather is capacity-bound: every replay copies all planned
storage, even when the live context is shorter. The gather is part of the paged
layer time; it is absent from the continuous API. Prepare every pool outside
capture, and retain the wrapper and graph inputs until caller graphs are discarded.

Pipeline ordering and packed-arithmetic compatibility adapters are private to
this package. Importing the backend preserves shared CuTe pipeline and arithmetic
methods, so later v1 and Full kernel compilation keeps its existing behavior.

The backend supports one prefill request with H4/H32/H64, logical D128,
K512/K1024/K2048, page sizes 32/64/128, compression ratios 1/4 and at most
1,048,576 keys after pooling. H8 and batched decode are unsupported. Row capacity must be
a multiple of 64 (H4), 8 (H32) or 4 (H64). Smaller live steps can reuse the capacity;
inactive and short rows receive -1 padding. Trace export is unsupported.

Continuous FP8 K is `[N,128]` with FP32 scales `[N]`. MXFP4 K is packed
uint8 `[N,64]` with four UE8M0 scale bytes per key; Q has 64 bytes per head.
Pass `q_scale` as uint8 `[R,H,4]` or packed int32 `[R,H]`. The paged adapter
accepts fused pages with all page key bytes followed by all page scale bytes,
then writes the separate continuous buffers. `logits_dtype` selects FP16,
BF16 or FP32 rounding; FP32 is the default. `math_mode` is immutable across
layers and captures, and NaN scores rank with negative infinity.

Omitting `model_config` selects the generic implementation. An explicit model
selects its numerical/template contract. The native model profiles are:

| Model | Heads | TopK | Native compression ratio |
| --- | ---: | ---: | ---: |
| DS-V3.2 | 64 | 2048 | 1 |
| DS-V4-Flash | 64 | 512 | 4 |
| DS-V4-Pro | 64 | 1024 | 4 |
| GLM-5.3-Flash | 32 | 512 | 4 |
| GLM-5.3 | 32 | 2048 | 1 |
| Qwen3.8-Flash-Next | 4 | 512 | 4 |

All six profiles support FP16, BF16 and FP32 logits; the profiles do not assign
a unique logits dtype per model. Compression ratios used as controls should be
labelled separately from the native profiles.

The Qwen template requires weights equal to FP32(1/sqrt(128)) and, for FP8,
unit key scales. The adapter checks the actual layer payload on every replay.
A violation returns -1 and accumulates bit 16 in `repair_error` until `plan`
resets it. Other templates support arbitrary per-head weights in strict mode.
Fast mode requires finite active weights that are zero or have magnitude at
least 2**-125; MXFP4 additionally requires active Q/K scale bytes at most 159.
These bounds are checked on every layer/replay and use the same bit-16 failure
signal. Unused row and key capacity does not participate in the checks. A model's
compression ratio describes already-compressed storage; the backend does not
compress inputs.

`VENDOR_MANIFEST.json` authenticates all current Python modules. Its
pre-relocation hashes describe the original snapshot, before the import-prefix
rewrite and ordered local patches; they do not hash the current patched files.
The import rewrite and each patch's before/after hashes retain that lineage.
Normalizing current imports alone does not recover the original snapshot:
reverse the recorded local patches first, then normalize the package prefix.
 The four specialized FP8 filters retire an empty producer/MMA tile
before loading query weights, using a wait/release without reading its scores
or scales. Live tiles keep the original scoring bounds and loop body; partial
physical tiles retain clamped scale loads. This preserves padding safety
without extending the live scoring loop to process a dummy tile. The generic
`fp8_safe_filter` retires the dummy at its existing KV-start location after
Q/weight setup; moving retirement earlier regressed its loop.
The H4/H32 branches of `fp4_safe_filter` and `fp4_glm_fulltile_filter` use
the same retirement at their original KV-start location; their H64 branches
are unchanged. Dispatch and scoring policies are unchanged. Vendor source
formatting is preserved; ordinary lint and type checking cover the public adapters.
Focused device tests cover supported public paths, exact score-multiset checks,
input changes under capture, template guards and constant-range sample proposals.
The original copyright notices are preserved; the snapshot is distributed under
the [BSD-3-Clause license](LICENSE).

BF16 shares the FP16 kernel families and eligible packed candidate layouts.
The dtype-specific filtering helper computes the exact FP32 preimage of BF16
round-to-nearest-even thresholds. Candidate publication uses BF16 conversions;
selection proves equal-score boundary bins before emitting ties, refines mixed
bins exactly, and skips constant low bits in BF16 streaming radix keys.
The strict/fast math contract, NaN-low ordering and private CuTe
compatibility helpers remain FlashInfer-specific.

Candidate payloads retain NaNs until the existing NaN-low ordering helpers;
fused publishers already use this representation. Removing the duplicate
encode-time conversion avoids extra K2 register pressure without changing
selection semantics. Crossing selectors use the source's single interval guard:
callers keep -1 for invalid spans, and a finite positive span guarantees a finite
lower bound and a positive reciprocal. These local changes are recorded after
the BF16 source patch in the manifest; the K2 filter bodies are unchanged.
