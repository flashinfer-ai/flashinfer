# SM120 DSA indexer top-k

Experimental `backend="sm120"` of `flashinfer.dsa_indexer.dsa_indexer_topk`.
It computes the DeepSeek Sparse Attention lightning indexer's scores and their
exact top-k on SM12x GPUs, in two forms: ragged (prefill) and paged (decode),
each in FP8 or MXFP4. Naming the backend is the opt-in; `backend="auto"` never
selects it. The backend is JIT-only. Tracking issue: flashinfer-ai/flashinfer#7084.

## Contract

- **Scores** are bit-identical to DeepGEMM's SM120 MQA logits:
  `fp8_mqa_logits` and `fp8_fp4_mqa_logits` for the ragged form,
  `fp8_fp4_paged_mqa_logits` for the paged form.
- **Score definition:** `s[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h] . k[j])`,
  where `w` carries every query, softmax and head scale.
- **Supported shapes:** 32 or 64 heads, head dimension 128, `top_k <= 4096`.
- **Selection:** each row gets its exact top `min(top_k, visible)` keys, ties
  going to the larger key id.
  - Ragged rows return ascending ids.
  - Paged rows return the selected set in no particular order.
  - Unused slots hold `-1` (index) and `-inf` (score).
- **Graph capture:** no call synchronizes with the host, and all launches are
  fixed by shapes. A CUDA graph therefore replays for any lengths up to the
  bound it was captured with: `max_seqlen_k`/`Tkv` for ragged calls,
  `block_table.shape[1] * page_rows` for paged calls.
- **SM count:** persistent grids are sized from the SMs of the launch
  stream's context, so green contexts and partitions are respected.

## Design

### Prefill

Rows of at most 16K, 32K or 48K keys take a *resident* route: one launch per
length class.
- Each CTA scores a query block into an L2 scratch row.
- Selector warps meanwhile select the previous block's rows, using a histogram
  cut, a list of the keys in the cut bin, and a bitmap write.

Longer rows take a sampled-threshold scan:
- Exact scores of evenly spread key chunks give each row a threshold aimed at
  about `2 * top_k` candidates.
- The scan appends the keys above the threshold.
- An exact selection runs over the candidates.
- Rows whose candidate count misses the bounds get an exact rescan.

### Decode

Decode uses two kernels:
1. **Logits scan.** It writes the logits to an L2-resident buffer, where large
   ones stay (evict_last) until the selection discards them. For rows longer
   than 16K keys it also builds the fp16-binned histogram the selection needs.
2. **Selection.** Each row of up to 16K keys gets one CTA, with the row held
   in registers (after SGLang's DeepSeek-V4 register top-k). Longer rows get
   several CTAs, which use the scan's histogram and an exact tie-break by id.

The long rows' selection state is double-buffered by epoch, so graph replays
need no reset.

## Performance

Measured on an RTX PRO 6000 Blackwell Server Edition (188 SMs) with CUPTI
kernel spans and a flushed L2. Each cell is compared against the fastest of
these pipelines:
- DeepGEMM logits followed by the top-k of vLLM, SGLang or FlashInfer;
- DeepGEMM logits followed by GVR;
- b12x's fused indexer.

| Form | Grid | Geomean speedup | Slower than 0.99x |
|---|---|---|---|
| Prefill, indices only | 152 cells: FP8 and MXFP4; H32 top-k 2048 and H64 top-k 512; fresh, append and mixed segments | 1.11x | none (worst 0.992x) |
| Decode | 126 cells: B 1-32, `next_n` 1/2/6, 1K-256K | 1.05x | H32 top-k 512, B32, `next_n` 2, 64K: 0.977x |

On the slower decode cell, the two-row scan is about 1.5% behind DeepGEMM and
the selection is about 2 us behind GVR's.
Four more cells read 0.94-0.98x inside the grid's single process. Measured
in a fresh process, they are at parity or faster:
- H64 B32 `next_n` 6 at 64K;
- H32 top-k 512 B1 at 256K;
- H64 top-k 2048 B1 at 1K and at 8K.

On an RTX PRO 5500 configuration (170 SMs, 112 MB L2) the tests pass, and
spot checks keep the same picture.

## Limitations

- Paged calls on concurrent streams of one device must not overlap: they share
  the per-device logits and long-row buffers. A call that needs more replaces
  a buffer with one at least twice its size; the replaced one stays alive for
  the CUDA graphs captured with it.
  - The logits take `rows * block_table.shape[1] * page_rows * 4` bytes.
  - The long-row state takes about 113 KB per row.
- Run a paged call outside CUDA graph capture before capturing it, so the
  long-row state is allocated and zeroed eagerly.
- `max_seqlen_k`, when given, must bound every segment's keys. Rows longer
  than the bound are left unwritten.
- Tested on SM120 (188 and 170 SMs) and SM121 (GB10, 48 SMs). Routing
  thresholds are tuned for SM120's 128 MB L2. On GB10's 24 MB L2, two cases
  fall behind the DeepGEMM-logits pipelines:
  - MXFP4 H32 prefill on rows of 32K-40K keys, down to 0.68x;
  - decode with `next_n` 6, 0.967x.

## Tests

`tests/experimental/test_sm120_dsa_indexer.py` checks ids and score bits
against an exact FP64 reference. It covers:
- both formats, both head counts, and top_k from 1 to 4096;
- causal offsets, compression ratios, crowded and tied scores;
- indices-only output;
- CUDA graph replay across lengths;
- paged rows from 0 to 140K keys, including more long rows than SMs.
