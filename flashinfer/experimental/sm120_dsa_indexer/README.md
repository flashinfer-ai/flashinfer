# SM120 DSA indexer top-k

Experimental `backend="sm120"` of `flashinfer.dsa_indexer.dsa_indexer_det_topk`.
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

### Decode candidate rows

DeepSeek-V4.1's consumer layers rank each row only within the key blocks a
source layer picked for it (`candidate_blocks`, at most 2048 blocks of 8 keys).
The paged form takes those lists directly instead of scanning the whole row
and masking it:
1. A small kernel turns each row's list into its distinct visible blocks in
   ascending order, and the length of the *virtual row* they form.
2. The scan treats the listed blocks as one row of at most `C * block_size`
   keys (16K for DeepSeek-V4.1). The producer issues one TMA box per block:
   a block is consecutive keys inside one page.
3. Each such row fits one selection CTA. Its epilogue maps the selected
   positions back to key ids. Because the blocks ascend, positions and key
   ids sort the same way, so ties still go to the larger key id.

A row's work is thus bounded by its candidates, not by its context.

The source layer publishes those lists (`candidate_out`). Its rows are scanned
and selected as usual, for their own top-k. In addition:
- The scan's epilogue writes each block's score: the largest score among its
  visible keys, with the row's newest block and any block with a NaN score
  pinned to `+inf`.
  - A warp's 16 keys of a tile lie on the MMA fragment's 8 row groups, so a
    block of 8 keys is three shuffles away and a block of 16 is the pair.
  - Blocks of 8 or 16 keys are therefore supported.
- The selection kernel then ranks these block rows like key rows, with
  `top_k = C`, and its ids are block ids.
- Block rows longer than one selection CTA (more than 16K blocks) get their
  histogram from the scan, beside the long key rows'. They keep their own
  long-row state.

## Performance

Measured on an RTX PRO 6000 Blackwell Server Edition (188 SMs) with CUPTI
kernel spans and a flushed L2, with every JIT-built kernel compiled by CUDA
13.4. Each cell is compared against the fastest of these pipelines:
- DeepGEMM logits (vLLM's pin, `a7a9277`) followed by the top-k of vLLM,
  SGLang or FlashInfer;
- DeepGEMM logits followed by GVR;
- b12x's fused indexer.

| Form | Grid | Geomean speedup | Slower than 0.99x |
|---|---|---|---|
| Prefill, indices only | 152 cells: FP8 and MXFP4; H32 and H64; top-k 512 and 2048; fresh, append and mixed segments | 1.16x | none (worst 1.003x) |
| Decode | 148 cells: B 1-32, `next_n` 1/2/6, 1K-256K, FP8 and MXFP4 | 1.05x | H32 top-k 512, B32, `next_n` 2, 64K: 0.979x |

On the slower decode cell, the two-row scan is about 1.5% behind DeepGEMM and
the selection is about 2 us behind GVR's. Cells that read below 0.99x inside
the grid's single process were re-measured in fresh processes; all others are
at parity or faster there.

The MXFP4 prefill scan is sensitive to the CUDA toolchain's instruction
scheduling: on the H64 resident path, a CUDA 13.0 build runs about 1% faster
than a 13.4 build, a 12.9 build about 8% slower and a 13.2 build about 12%
slower.

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
- Candidate rows and source rows are decode-only (the paged form).
- Rows whose candidates cover all their blocks still take the per-block loads,
  so up to 16K keys they run slower than the same rows without candidates.
- A source row costs one launch more than a plain row: the block selection.
  `candidate_out` takes blocks of 8 or 16 keys.
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
