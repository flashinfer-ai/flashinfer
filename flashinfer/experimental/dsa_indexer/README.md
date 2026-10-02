# Experimental DSA indexer top-k (LiteTopK)

`flashinfer.dsa_indexer.dsa_indexer_topk` is an experimental API: it may change
or be removed without deprecation. Calling it is the opt-in and emits
FlashInfer's experimental API warning once. There is no automatic backend
selection.

DSA models such as GLM-5 select, for every query, the top-k KV
positions of an fp8 MQA "indexer" score. The dense path materializes the
`[num_q, seq_kv]` float32 logits (32 GiB for 8192 queries at 1M context) and
then runs a top-k over them. This kernel scores KV tiles with tcgen05 UMMA,
keeps only positions above a per-row threshold in a fixed-size candidate buffer
(`6 * num_q * cand_cap` bytes, 2.25 GiB for 8192 queries), and selects the
exact top-k from it. Spare warps tighten the threshold while the scan runs.

The threshold is seeded from the logits of a prefix `kv[:P]` that the caller
computes with any dense kernel. The prefix positions are candidates too, and
the scan covers `[P, cu_end[i])` (query `i` sees `kv[0:cu_end[i]]`). For chunked
prefill, place the previous chunk's selected positions in the prefix and map the
result back. The reordering must keep every query's visible set: hot positions
below `cu_end.min()` first, the remaining positions after them in order.

```python
# order: hot positions (< cu_end.min()) first, then the others in order
kv_p, scales_p = kv[order], kv_scales[order]
prefix = deep_gemm.fp8_mqa_logits(q, (kv_p[:P], scales_p[:P]), weights, ks, full_P,
                                  clean_logits=False)
idx, status = dsa_indexer_topk(q, kv_p, scales_p, weights, prefix, cu_end)
idx = order[idx.long()]  # rows with status != 0 are invalid
```

## Limitations

- SM100 (compute capability 10.0) only. The kernels compile against the SM100
  headers of the `deep_gemm` package (the layout of DeepGEMM 891d57b4, the commit
  the FlashInfer EP image installs), which must be installed.
- 32 heads, head dim 128, fp8 e4m3 `q`/`kv`, `top_k == 2048` (GLM-5).
- `seq_kv <= 2**20`, `kv_scales` padded to a multiple of 4 elements;
  `2048 <= P <= 8192` or `P == 12288`; every query must see the whole prefix
  (`P <= cu_end.min()`, not checked).
- A nonzero `status` marks an invalid row, e.g. when more than `cand_cap`
  positions pass a threshold seeded from an unrepresentative prefix.
- Indices are unordered. Among positions whose scores tie at the selection
  boundary, the kept ones can differ between runs.

## Validation and performance

`tests/experimental/test_dsa_indexer.py` checks recall against a dense reference.
`benchmarks/bench_dsa_indexer_topk.py` is a runnable end-to-end example and
compares against `deep_gemm.fp8_mqa_logits` + `flashinfer.top_k`.

B200, GLM-5.2 layer-0 indexer tensors, 8192 queries, top-2048, prefix of 12288
positions taken from the previous chunk's selection, end to end (dense: fp8 MQA
logits + `flashinfer.top_k`; fused: prefix logits + `dsa_indexer_topk`):

| seq_kv | dense ms | fused ms | speedup | dense peak GiB | fused peak GiB |
|-------:|---------:|---------:|--------:|---------------:|---------------:|
|   128K |     6.25 |     5.87 |   1.07x |           20.2 |            2.7 |
|   256K |    12.32 |    10.59 |   1.16x |           40.2 |            2.7 |
|   512K |    23.71 |    19.10 |   1.24x |           80.2 |            2.7 |
|   768K |    37.13 |    28.81 |   1.29x |          120.2 |            2.7 |
|     1M |    50.98 |    38.43 |   1.33x |          160.2 |            2.7 |

The dense peak includes the `flashinfer.top_k` workspace; the logits alone take
`4 * num_q * seq_kv` bytes (32 GiB at 1M). Recall against the dense top-2048 is
0.9998-1.0000.
