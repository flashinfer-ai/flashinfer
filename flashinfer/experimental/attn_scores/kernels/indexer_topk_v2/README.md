# Selective v2 prefill

This private CuTe DSL package implements sampling, candidate scoring, exact
selection and recovery. Use `flashinfer.FP8MQATopKWrapper(backend="v2")` with
contiguous K and separate scales. The public paged wrapper owns one continuous
wrapper and gathers K/scales before delegating Full, selective v1 or selective
v2. Its preallocated gather is capacity-bound: every replay copies all planned
storage, even when the live context is shorter. The gather is part of the paged
layer time; it is absent from the continuous API. Prepare every pool outside
capture, and retain the wrapper and graph inputs until caller graphs are discarded.

V2 supports single-request prefill with H4/H32/H64, D128, TopK 512/1024/2048,
compression ratios 1/4, and at most 1,048,576 physical keys. Query capacity must
be a multiple of 64 (H4), 8 (H32), or 4 (H64). Smaller live steps reuse this
capacity; missing output entries are padded with -1. Decode and trace export
are unsupported.

Policy selection uses `(num_heads, head_dim, top_k, compression_ratio)`.
An optional `model_config` must agree with those fields; it does not select a
separate implementation. These tuples have model-specific contracts:

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

FP8 uses E4M3FN Q/K and FP32 per-key scales. MXFP8 uses E4M3FN Q/K with
UE8M0 scales per 32 elements. MXFP4 packs two values per uint8 byte and uses
the same scale granularity. Block-scaled K storage must have a multiple of
256 rows; the plan describes the live extent. Pass block-scaled `q_scale`
as uint8[R,H,4] or its int32[R,H] view, and `k_scale` as uint8[N,4].

`logits_dtype` selects FP32, FP16, or BF16 score rounding before selection.
`weights_dtype` independently selects FP32 or BF16 input storage; BF16 weights
may have padded rows with contiguous heads. `math_mode` is fixed for the
wrapper's lifetime. Strict/fast modes use FP32 head accumulation; explicit
`native_bf16` and `native_fp16` use matching head arithmetic and logits dtype.
See the public wrapper docstring for the reduction and FP8/FP16 scaling rules.
K1, K2, and K4 share the same score rule. NaNs rank with negative infinity;
equal-score ties need not produce identical indices.

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

`VENDOR_MANIFEST.json` records SHA-256 digests of all Python modules. Device
tests cover score-multiset correctness, ties, captured input changes, numerical
contracts, and recovery. Copyright notices and the [BSD-3-Clause license](LICENSE)
are retained with the source.
