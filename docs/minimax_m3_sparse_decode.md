# MiniMax-M3 speculative sparse decode

`flashinfer.msa_ops.minimax_m3_sparse_attn_decode` implements the per-token
sparse decode contract from issue #4567 on SM100/SM103. It consumes the original
indexer's `topk_idx[Hkv, total_q, 16]`; it does not replace or run the indexer.

```python
import torch

from flashinfer.msa_ops import (
    MiniMaxM3SparseDecodeWorkspace,
    minimax_m3_sparse_attn_decode,
)

# Allocate outside the serving loop / CUDA graph. TP1: 64/4; TP4 rank: 16/1.
workspace = MiniMaxM3SparseDecodeWorkspace(
    batch_size=16, num_qo_heads=64, num_kv_heads=4, decode_query_len=4,
    device=q.device,
)
out = torch.empty_like(q)

# Warm the exact shape before graph capture. All tensor arguments stay on GPU.
minimax_m3_sparse_attn_decode(
    q, packed_kv, topk_idx, block_table, seq_lens,
    k_scale=k_scale, v_scale=v_scale, out=out, workspace=workspace,
)
```

The input/output contract is:

- BF16 Q/O `[B * Q, Hq, 128]`, with uniform query length `Q` in 1–8.
- Packed E4M3 cache `[physical_pages, Hkv, 128, 256]`, K followed by V within
  each token. The token stride is 256, not 128. The cache remains in place.
- CUDA int32 request page tables `[B, max_pages]`, lengths `[B]`, and independent
  sparse selections `[Hkv, B * Q, 16]`. Physical pages may be permuted/shared.
  Strided metadata is read directly, including the actual indexer's
  `out[:, :total_q, :]` view into a larger CUDA-graph output buffer.
- Explicit CUDA float32 scalar K/V dequantization scales (one element each).
  Tracing preserves 0D/1D scale ranks, including mixed forms, in distinct
  schemas. Higher-rank singleton views remain valid for decode, but tracing
  those forms is unsupported rather than emitting an incorrect schema.
- At most 262144 context tokens, 256 requests, and GQA 64/4 or 16/1.

For local query index `i`, its causal length is
`max(seq_lens[b] - Q + i + 1, 0)`. Only the first
`min(16, ceil(causal_length / 128))` indices are used. They must be distinct valid
logical page IDs. Their order can be arbitrary; unused slots are ignored, not
interpreted as padding sentinels. The tail page is not implicitly added.

The GPU metadata kernel maps and orders only those 16 selected IDs. It computes
the surviving token count and moves any partial logical page to the final slot.
The existing native TRT-LLM block-sparse kernel then runs with **one query per
independent row**, not one speculative request per row. This preserves distinct
token selections and causal masks without taking a union or sharing rows.

All output, metadata, scratch, and reduction-counter storage is allocated before
the steady-state call. Metadata is rebuilt on GPU on every invocation; there is
no CPU plan tied to length/index contents, no host sync/D2H, no KV gather, and no
query quantization. CUDA graph replay may change lengths, page tables, selections,
and scale values in place. Do not use the same workspace concurrently on multiple
streams or overlapping graph replays.

`sm_scale` is a finite positive host scalar. Tensor-valued, non-positive, and
non-finite values are rejected before any launch; K/V scale tensors remain
device-resident inputs and are not read back by this validation.

This route currently supports scalar KV scales, not the reference wrapper's
additional per-token/head scale arrays. Input validation checks tensor descriptors
without synchronizing to inspect device contents. Caller-provided lengths and
active indices must be valid. Numerical comparisons should use BF16 tolerances:
the native kernel folds scalar dequantization into attention scales, whereas the
Triton reference rounds scaled K/V to BF16 on load.

The tests use the existing FlashInfer MSA BF16 tolerance
(``atol=rtol=0.02``) and additionally require relative RMS error below 1.5%
for every individual query/head. A strict ``atol=0.01`` guarantee is not made:
on rare near-zero coordinates, the different scale-rounding positions can
exceed that threshold even though the native result is closer to an FP32
dequantization oracle. Neither bitwise equality nor identical intermediate
rounding is part of this route's contract.

For reproduction, run `tests/msa_ops/test_minimax_m3.py` and the dedicated
`benchmarks/bench_minimax_m3_sparse_decode.py` benchmark with the pinned vLLM
`sparse_attn.py` reference. Compare complete calls,
including the metadata kernel and QK-scale preparation, rather than attention
kernel time alone. No model weights or inference server are required.
