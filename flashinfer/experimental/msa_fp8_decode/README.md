# Experimental packed-FP8 MSA sparse decode (SM100/SM103)

Tracking issue: https://github.com/flashinfer-ai/flashinfer/issues/6020
(owner: @Analysis196884).

Both the API and this backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection; `flashinfer.msa_ops.msa_sparse_decode_attention` is
unchanged and keeps serving its existing routes.

`flashinfer.msa_ops.msa_packed_fp8_sparse_decode(...)` serves the MiniMax-M3
speculative sparse decode step over a packed FP8 E4M3 paged KV cache
`(num_pages, Hkv, 128, 256)`, consumed in place as the strided views
`k=cache[..., :128]`, `v=cache[..., 128:]`, with BF16 Q, device float32 scalar
K/V scales and a caller-owned `MSASparseAttentionWorkspace`:

- Each work item is one `(query token, KV head)` pair with its sixteen selected
  pages; speculative queries are not folded together. A small preparation
  kernel (`csrc/msa_decode_metadata.cu`) maps the logical top-k rows through
  the block table and folds the K scale into the softmax scale; the attention
  itself is the existing TRT-LLM block-sparse decode kernel
  (`enable_block_sparse_attention=True`), so K/V are never repacked.
- `seqlen_q` in `[1, 8]`, batch size in `[1, 256]`, head layouts 64/4 and 16/1,
  contexts up to 262144 tokens. Causal, right-aligned decode tokens only; no
  LSE output, no global scales, no explicit `q_offset`.
- The workspace owns the output and temporary buffers so their addresses stay
  stable for CUDA graph replay; warm it eagerly with the exact tensors,
  options, and capture stream before capture. Metadata, page tables and the
  scalar scale values may change in place between replays.
- The TRT-LLM workspace handed to the kernel is sized from the multi-CTA-KV
  scratch contract (one float2 partial-softmax slot plus one BF16 partial-O
  row per launched CTA, launch capped at the SM count), not the 128 MiB
  wrapper convention; it is about 1.2 MiB on B200.

## Graduation

Graduation (or removal) is targeted within four weeks of landing; see the
tracking issue for the plan. Graduation adds AOT registration and drops the
experimental marker without an import-path change.
