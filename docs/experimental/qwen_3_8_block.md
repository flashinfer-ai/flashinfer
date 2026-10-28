# Qwen 3.8 Flash-Next gated attention block (experimental)

**Status: experimental API and experimental backend.** Everything here -- the names, the signatures, the decode form and the
vendored kernels -- may change or be removed without deprecation. Calling an entry point of `flashinfer.qwen_3_8_block` is the
opt-in (it emits the standard `ExperimentalWarning` once); no environment variable is read, nothing is reachable from a stable
API's `backend="auto"`, and nothing is registered for ahead-of-time compilation (JIT only, never in `flashinfer-jit-cache` /
`flashinfer-cubin`).

A SIBLING of the [Qwen 3.5 gated attention block](../qwen_3_5_block/README.md): the same vendored cuDNN Frontend tree
(`../qwen_3_5_block/_cudnn_frost/`, ONE drop, ONE manifest, ONE re-vendor command), the same cache roots, the same capture rules; the
geometry, the weights and the plan carry the sparse-attention indexer appended. The 3.5 package's dense block does not change.

## What it computes

The whole Qwen 3.8 Flash-Next sparse-attention (QSA) sub-layer at DECODE, from the post-layernorm hidden states to the attention
output, in bf16 on SM107 (Rubin, compute capability 10.7), in ONE call over the serving stack's own caches:

```
hidden_states [T, d_model]                      T = batch x q_len (q_len = 1 + drafts)
  -> (1) QKV + gate + indexer projection         W_qkvg [n_qkvg, d_model], bands Q | GATE | K | V | INDEX (the indexer's 4 x 128 query heads + its 128-wide raw key head)
  -> (2+3+4w) one "prepare" launch               QK-RMSNorm + partial RoPE in place; the post-RoPE K / V rows into the pools at slot_mapping; the indexer queries
                                                 RMSNormed + rotated; the raw indexer key into the RING at (ring block, pos % cap); the 4-token block a committed row
                                                 COMPLETES mean-pooled, RMSNormed and rotated at the block START into the paged COMPRESSED-KEY CACHE through the MAIN
                                                 block table (positions from the position tables)
  -> (4i) the paged scorer + the per-row top-k    logits[t, j] = scale x sum_h relu(q_i[t, h] . kbar[b, j]) over every complete block the row can see; the top_k best per row
  -> (4) block-sparse GQA attention, head dim 256 per row over its selected 4-token blocks plus the open tail of its own sequence (bottom-right causal), from the pools
  -> (5) O *= sigmoid(GATE)                       in the sparse tile's epilogue (unsplit) or in the split combine
  -> (6) out projection                           W_o [d_model, H_q * 256]
  -> out [T, d_model]
```

Eight kernel launches at a split (the slab GEMM, the prepare, the scorer, the top-k, the sparse tile, the combine, the out projection
and its split-K reducer), six at 128 rows; no memset, no memcpy, no host read; the first `run` after a plan captures with no warm-up.

| form | entry point | step | what it reads and writes |
|---|---|---|---|
| decode | `qwen_3_8_block_bf16_decode` | a uniform `[batch, q_len]` step, `q_len = 1 + drafts` (up to 16 at four indexer heads) | reads the pools, the ring and the compressed-key cache through the stack's block tables; writes the step's K / V rows, ring rows and completed cache rows in place |
| prefill, extend | -- | -- | a later cuDNN Frontend PR vendors the indexer's prefill forms; `prepare_qwen_3_8_block_bf16("prefill" / "extend", ...)` is a typed `NotImplementedError` naming it |

`prepare_qwen_3_8_block_bf16("decode", ...)` returns the plan behind the form (`Qwen38BlockPlan`: `run`, `workspace_size`,
`declaration`, `supports`); `Qwen38BlockWrapper(workspace_buffer, kv_layout, geometry=...)` is the same thing in FlashInfer's
`plan()` / `run()` idiom over a caller-owned workspace -- the serving idiom: ONE workspace sized at the maximum of
`qwen_3_8_block_workspace_size` over every `(batch, q_len)` bucket the stack plans serves every bucket and every layer run in turn on
one stream (under `"auto"` the largest batch is not the largest workspace: admissible batch 1 and 2 buckets with `top_k >= 8` can take the lean body, whose
split-KV slab is wider than the core's at batch 4). The functional entry point keeps its own small LRU (32 entries, one
per shape bucket, each owning a zeroed workspace; the Qwen 3.5 rules of retention and release, one LRU per package) --
`flashinfer.experimental.qwen_3_8_block.api.clear_plan_cache()` drops them.

**The decode step runs the cuDNN Frontend MEASURED defaults** (none of them passed by this block): the prepare fusion ON on cc 10.7
(`fuse_prepare=None` -> True: one launch for the norm / RoPE / write-through / ring / compress, bitwise the four-launch twin;
`fuse_prepare=False` is the twin), programmatic dependent launch OFF on the sparse chain (`use_pdl=None` -> False; `True` is a typed
decline until the vendored tree serves it there), the sparse core's BODY by the tree's placement rule (`sparse_tile_form="auto"`: the
LEAN decode tile -- `mma.sync` + `cp.async`, one CTA of `128 x N_MT` threads per (sequence, KV head, chunk), no TMEM / TMA / cluster -- on
cc 10.7 at `batch x H_kv <= 4` CTAs per chunk (the bound's unit: the step's rows ride inside the CTA, so `q_len` is not in it) at every
served step width the body admits -- under the in-block indexer the list is per row, which the lean body serves at one row only, i.e.
`batch <= 2` at `q_len_per_req 1` on the 24 / 2 geometry --, the tcgen05 core everywhere else; `"lean"` / `"tcgen05"` pin a body;
the same function under either body within the tree's budget, never bitwise ACROSS the bodies), the split-KV of the sparse tile by its
rule (the lean split, 65 at `top_k 512`, on the lean body; the fill rule on the core). Every one reads back in `plan.declaration`
(`fuse_prepare`, `fuse_prepare_requested`, `use_pdl`, `use_pdl_requested`, `use_pdl_effective`, `pdl_stages`, `sparse_tile_form_requested`,
`sparse_tile_form`, `sparse_tile` -- the body's `kernel_template`, `"sparse_lean_d256_f16"` / `"sparse_d256_f16"`, pinned against the compiled
kernel --, `sparse_sub_tiles_per_cta`, `decode_split_kv`, `indexer_launches`, `launches_per_execute`); `plan.sparse_tile` / `plan.sparse_tile_form`
carry the body too.

## A step, vLLM-shaped

```python
import torch
from flashinfer.qwen_3_8_block import (
    Qwen38BlockGeometry, Qwen38BlockWrapper, qwen_3_8_block_support, qwen_3_8_block_pack_weights,
    qwen_3_8_block_rope_position_tables,
)

# the geometry at THIS rank: the Qwen 3.5 fields first, the indexer appended (Qwen3.8-Flash-Next at TP 1; 12 / 1 at TP 2, 6 / 1 at TP 4)
geometry = Qwen38BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2,
                               index_heads=4, index_kv_heads=1, index_head_dim=128, block_size=4, top_k=512)
sup = qwen_3_8_block_support(torch.device("cuda"), geometry)                      # the init-time dispatch predicate (never raises)
assert sup.decode.served, sup.reason

# load time: the merged weight with the indexer band appended (vLLM packs the replicated indexer projection into the same GEMM), the
# four zero-centred norms -> the block's bands and (1 + w) norms; the position tables ONCE, padded to the block table's reach
weights = qwen_3_8_block_pack_weights(layer.qkv_proj.weight, layer.q_norm.weight, layer.k_norm.weight, geometry,
                                      index_q_norm_weight=layer.indexer.q_layernorm.weight, index_k_norm_weight=layer.indexer.k_layernorm.weight)
w_o = layer.o_proj.weight                                                          # [d_model, H_q * 256], used as is
cos_pos, sin_pos = qwen_3_8_block_rope_position_tables(layer.rotary_emb.cos_sin_cache, num_positions=max_num_blocks_per_req * page_size, dtype=torch.bfloat16)

# init: one workspace for every (batch, q_len) bucket; one plan per bucket, planned where the stack warms its kernels (outside capture)
workspace = torch.empty(sup_workspace_bytes, dtype=torch.uint8, device="cuda")     # >= every bucket's plan.workspace_size
w = Qwen38BlockWrapper(workspace, "UNIFIED", geometry=geometry)
w.plan("decode", weights=weights, w_o=w_o, kv_cache=layer.kv_cache, page_size=page_size, batch_size=B, q_len_per_req=1 + num_spec,
       block_table=block_table, index_kv_cache=compressed_key_cache, index_state_cache=raw_key_ring,
       index_state_block_table=ring_block_table[:B], position_tables=(cos_pos, sin_pos), ring_slots=True)

# per step (capturable from the first run): the stack's tensors renamed, nothing copied, nothing computed on the host
out = w.run(hidden_states, None, None, weights, w_o,
            kv_cache=layer.kv_cache, slot_mapping=slot_mapping, block_table=block_table[:B], seq_lens=seq_lens[:B],
            index_kv_cache=compressed_key_cache, index_state_cache=raw_key_ring, index_state_block_table=ring_block_table[:B],
            index_state_slot_mapping=ring_slot_mapping)
```

## Arguments and layouts

- **`kv_cache`**: FlashInfer's 5-D `[num_pages, 2, page_size, H_kv, D]` (`kv_layout="NHD"`) or `[num_pages, 2, H_kv, page_size, D]`
  (`"HND"`) buffer, a `(k_cache, v_cache)` pair in those layouts, or the serving stacks' UNIFIED row `[num_pages, page_size, H_kv, 2 x D]`
  (`kv_layout="UNIFIED"`: K at `[..., :D]`, V at `[..., D:]`, any strides the stack's layout planner chose); bf16. The block binds its
  `[num_pages, H_kv, page_size, D]` views by strides (no copy, no repack); `D` contiguous, every other stride a 16-byte multiple, the
  bases 16-byte aligned. `page_size` is 16, 32, 64 or 128 (the pages the paged scorer addresses: a power-of-two count of compressed
  rows) and is read from the cache's shape by the functional entry point. The pools' page count, layout and strides are compiled into a
  plan: another cache is another plan.
- **`block_table`**: the dense `[batch, max_pages]` int32 table as vLLM holds it (row `0` = the NULL block of a padded entry); the
  block-sparse core binds the WHOLE table, so a plan is made with the table at its full width (a leading-row slice of a wider stack
  table binds; a column slice does not). The compressed-key cache is addressed through this SAME table (one page of the main cache =
  `page_size / 4` compressed rows).
- **`seq_lens`**: `[batch]` int32, the LOGICAL KV length INCLUDING the step's new tokens (`0` = a padded entry). Every row's position is
  `seq_lens[b] - q_len + j`, derived on device.
- **`slot_mapping`**: `[T]` int32 / int64 flat slots `page * page_size + offset` of the step's tokens; `-1` writes nothing.
- **`index_kv_cache`**: the paged compressed-key cache `[num_pages, page_size / 4, 1, 128]` bf16 -- one row per complete 4-token block,
  the row of block `j` at `(block_table[b, j // (page_size / 4)], j % (page_size / 4))`; read by the scorer, the block a step completes
  written by the prepare launch. Its strides are the stack's (the kernels bind them symbolically); the page axis is the main table's.
- **`index_state_cache`**: the raw-key ring `[num_ring_blocks, cap, 1, 128]` bf16, position `p` of sequence `b` at
  `(index_state_block_table[b, 0], p % cap)`; `cap >= q_len + 3` (a completing block pools the three positions below the step while the
  step writes `q_len` rows; a stack's ring of whole 4-token groups covering `1 + num_speculative_tokens` rows satisfies it); block `0`
  the null block. The raw keys as the projection emits them (a bit copy, never quantized).
- **`index_state_block_table`**: `[batch, 1]` int32, the ring block per sequence (`0` for a padded entry; a wider stack table is handed
  as its leading-row slice).
- **`index_state_slot_mapping`** (the ring-slots arm, `ring_slots=True` at the plan): `[T]` of `slot_mapping`'s dtype, the ring WRITE
  rows `block x cap + pos % cap`, `-1` = no write -- a stack's own circular-buffer slots; `ring_slots=False` derives them in the kernel.
  Bitwise; a compile key.
- **`position_tables`**: `(cos_pos, sin_pos)` `[1, >= max_pages x page_size, rope_dim]` bf16 from
  `qwen_3_8_block_rope_position_tables(cos_sin_cache, num_positions=...)`, formed ONCE at load: every rotation of the
  step (the Q / K heads, the indexer queries, the completing block's key at its block START) reads them at the row's device-derived
  position. The block refuses a table shorter than the block table's reach by form; `num_positions` pads past the model's maximum
  position (rows no token reaches). With `rope_position_tables_only=False` the per-token `rope_cos` / `rope_sin` are read for the step's
  own rows instead (the unfused arm's form; `qwen_3_8_block_rope_tables` builds them; bitwise the table arm). The functional entry
  binds the pair of EACH call (required; checked against the bucket's plan -- a cache hit never rotates with the pair another call
  compiled in); an explicit plan binds the pair it was prepared with unless its `run` is handed one.
- **`hidden_states` / `out`**: contiguous `[T, d_model]` bf16 (or `[batch, q_len, d_model]`); `out` caller-owned for captured runs. Pad
  rows carry finite values (the GEMM reads them; a padded entry's `out` rows are exactly 0 regardless).
- **`block_ids_out` / `block_lens_out`** (optional): `[T, top_k]` / `[T]` int32 buffers that receive the block's own per-row selection
  (the ids strictly descending, `-1` past the valid prefix; the prefix length `min(top_k, floor((pos + 1) / 4))`) -- for tests, A/B and a
  stack that reuses the list; nothing on the step reads them.
- **`decode_split_kv`**: `None` = the sparse tile's rule (the fill rule on the tcgen05 core; the lean split, 65 at `top_k 512`, on the lean
  tile -- which ALWAYS splits), an int the explicit split (the combine as a second launch under a split, the gate riding it; on the lean
  body one of the lean splits, elsewhere a typed decline naming them); a performance knob -- the same function under every value.
- **`sparse_tile_form`**: `"auto"` (default) / `"lean"` / `"tcgen05"` -- WHICH BODY of the index-list sparse core serves the step (the
  paragraph above): `"auto"` follows the vendored placement rule (the lean tile on cc 10.7 at `batch x H_kv <= 4` CTAs per chunk -- the
  step width is not in the bound -- where the body admits the step: under the in-block indexer at `q_len_per_req 1`, the per-row list past
  one row being the core's --, the tcgen05 core everywhere else), an explicit body is honoured where it admits the step and a typed
  `NotImplementedError` naming the arm where it
  does not; any other spelling is a `ValueError` before any vendored code. A performance knob (the same function under either body within
  the tree's fixed-order-sum budget; bitwise within a body, never asserted bitwise across them) and a plan-cache key; the compiled body reads
  back in `plan.sparse_tile`. The lean tile's `sparse_sub_tiles_per_cta` pin is not exposed here (the vendored default k = 1 is the measured
  rendering). `qwen_3_8_block_workspace_size` takes the same knob: the carve is the body's (the lean split's partial slab is wider).
- **`kv_cache_dtype`**: `None` = bf16 pools. `torch.float8_e4m3fn` (vLLM's `--kv-cache-dtype fp8`) is a typed decline until the sparse
  core's in-kernel dequant arm lands; `k_scale` / `v_scale` are then taken at `run`.
- Every VALUE in the index tensors is device data (never read on the host); only the FORM is checked, host-side and typed, before any
  launch.

## Capture and replay

1. `prepare_*` / `plan` run outside CUDA-graph capture (they JIT-compile the kernels and size the workspace); under capture they raise a
   typed `RuntimeError`.
2. The FIRST `run` after a plan may be captured with no eager warm-up: no plan constant, no allocation, no sort, no host read on the
   execute path; the captured execute launches only kernels and replays bitwise an eager run (tested, over a new step too).
3. A captured graph replays over NEW values of `seq_lens`, `block_table`, `slot_mapping`, `index_state_slot_mapping`, the ring table,
   `hidden_states` and the caches' CONTENTS written in place into the SAME buffers (addresses are baked). The pad regime is the Qwen 3.5
   one: a padded entry is `seq_lens 0`, table row `0`, slot `-1`, ring block `0` (ring slot `-1`) -> `out` rows exactly 0, NO write to
   any pool, ring or cache row.
4. The pools must be FINITE where the kernels read past `seq_lens` inside the last live page; the compressed-key cache rows a step reads
   are the complete blocks below the row's position (written by a prefill or an earlier step); a `-1` table entry under a visible block
   reads as a zero key.
5. One plan / wrapper is a single-writer resource: do not re-plan while a captured or in-flight `run` may read its buffers.
6. The functional entry point under capture runs the bucket's CACHED plan (a cold bucket is a typed `RuntimeError`; warm it with one eager
   call); a plan a graph was captured over is RETAINED past its eviction and revived on the bucket's next call;
   `clear_plan_cache()` is the ONLY release (refused under capture).

## Two writers, one cache

A serving stack's PREFILL (and its mixed / chunked batches) writes the pools, the ring and the compressed-key cache with its own
kernels in the position-indexed form this block reads; the block writes the rows its decode steps complete. Both are exact w.r.t. their
own operands, but they round the pooled key differently (this block keeps the fp32 mean through the RMSNorm; a prefill writer that
rounds the mean to bf16 first differs by one bf16 ulp on a third of the elements), so a cache holding rows of both conventions is a
model-output difference of the second-GEMM-rounding class, not a correctness defect; a declared pooled-mean rounding form is a cuDNN
Frontend follow-up for a stack that wants the two writers bitwise. The FlashInfer tests prime the caches from a torch reference prefill
through the vendored compress kernel (one convention).

## Errors

| condition | exception |
|---|---|
| a tensor-parallel shard whose fused slab or out projection is off the capture-safe small-M projection route on the device (the TP 2 / TP 4 slabs, N 7296 / 4224: 256 does not divide N and its 64-row weight tiles fit one wave; the vendored route's own arithmetic decides, at the part's SM count); the probe lists it as `proj_slab_route` -- a FlashInfer-side row, not an envelope row | `NotImplementedError` naming the projection, N, K and the SM count |
| not cc 10.7; `nvidia-cutlass-dsl` absent, below 4.8.0 or without the Rubin target; f16 / fp8 activations on the bf16 entry; `head_dim != 256`, `d_model % 256`, `rope_dim` not 64 / 128; the indexer geometry off `index_head_dim 128` / `index_kv_heads 1` / `block_size 4` / `top_k` a multiple of 4 in [4, 512]; `index_list_form="shared"`; `index_cache_dtype=torch.float8_e4m3fn`; `kv_cache_dtype=torch.float8_e4m3fn`; `q_len_per_req x next_pow2(index_heads) > 64` (16 rows at four heads); a `page_size` the paged scorer does not address (not 16 / 32 / 64 / 128); `use_pdl=True` on the sparse chain; `fuse_prepare=False` with `rope_position_tables_only=True`; `"prefill"` / `"extend"` on this entry | `NotImplementedError` naming the fact (the vendored decline text where one exists; `qwen_3_8_block_support` reports it without raising, and its `decode.declines` table carries every vendored row by key: `indexer_f16`, `indexer_head_dim`, `indexer_thd`, `indexer_query_rows`, `indexer_page_size`, `indexer_e4m3_pool`, `indexer_shared_list`, `shared_list_rows`, `fuse_prepare_rope_tables_unfused`, plus the two FlashInfer-side rows `indexer_e4m3_cache` (the tree serves the e4m3 compressed-key cache; this entry's surface for it is a follow-up) and `proj_slab_route`) |
| a malformed operand (dtype, rank, shape, strides, alignment, device); a ring with `cap < q_len + 3` (the bound named); position tables shorter than `max_pages x page_size`; `index_state_slot_mapping` given / missing against the plan's `ring_slots`, or of another dtype than `slot_mapping`; `rope_cos` / `rope_sin` given under the table arm or missing under the per-token arm; `query_start_loc` / `num_actual_tokens` on the decode form; `k_scale` / `v_scale` on bf16 pools; `max_pages` narrower than the table; the CSR triple | `ValueError` naming the operand and the expected form |
| `prepare` / `plan` under capture; a cold functional bucket under capture; `clear_plan_cache()` under capture | `RuntimeError` naming the remedy |

No silent fallback: an unserved case never runs a slower path.

The probe and the plans share ONE geometry gate (`Qwen38BlockGeometry.unsupported_reason`: the Qwen 3.5 rows, then the indexer rows),
so a geometry `qwen_3_8_block_support` advertises is one the decode declaration admits, and a geometry it declines is the same typed
`NotImplementedError` from `prepare_*` / `plan` before any vendored code runs. The indexer rows are MIRRORED from the vendored tree
(`index_head_dims`, `index_kv_heads`, `index_block_sizes`, `index_top_k_range`, `index_query_rows_max`, `index_state_rows_extra`,
`index_page_sizes`, the capture-safe row bound); the first probe on a served device and the first plan re-read every one of them off
the vendored `DecodeEnvelope` and refuse a drift loudly (`check_vendored_rules`), and `tests/experimental/qwen_3_8_block/test_admission.py`
plants a drift per constant. The probe's appended fields (`index_sources`, `index_list_forms`, `index_cache_dtypes`, `kv_cache_dtypes`,
`fuse_prepare_default`, `pdl_default`, `sparse_tiles` -- the sparse core's bodies this part serves by `kernel_template`: the tcgen05 core
everywhere, the lean decode tile too on cc 10.7 --, `sparse_tile_lean_default` -- whether `"auto"` consults the lean placement rule on this
part --, ...) are READ off that envelope, never re-spelled; the per-cell reading (which body a `[batch, q_len]` step takes under a knob
value) is `flashinfer.experimental.qwen_3_8_block.support.sparse_tile(...)`, the envelope's own rule, and the mirrored vocabulary
(`SPARSE_TILE_FORMS`, the two `kernel_template` names, the AUTO bound `SPARSE_TILE_LEAN_MAX_UNITS = 4` and its unit -- the placement
rule's reading over `(batch, q_len)` cells, `SPARSE_TILE_LEAN_RULE_CELLS`: CTAs per chunk, the step width outside the bound) is re-checked
by the same tripwire.

## Requirements and limits

- SM107 (Rubin, cc 10.7) only; `nvidia-cutlass-dsl >= 4.8.0` (the kernels gate at runtime and decline by version); torch with CUDA 13;
  `cuda-python`.
- bf16 activations, weights, pools, ring and compressed-key cache; `head_dim = 256`; `d_model % 256 == 0`; `rope_dim` 64 or 128; the
  indexer `4 x 128` over one raw key head (the geometry fields are the model's: `indexer_n_heads`, `indexer_head_dim`,
  `indexer_kv_heads`, `indexer_compress_ratio` = `block_size`, `indexer_budget / compress_ratio` = `top_k`), `block_size` 4, `top_k` a
  multiple of 4 up to 512; text-only RoPE (the position tables are 1-D positions; mRoPE is the caller's to decline);
  `q_len_per_req <= 16` at four indexer heads (the paged scorer stages `q_len x next_pow2(index_heads) <= 64` query rows per
  sequence); `batch x q_len <= 128` for a CAPTURED step (the band slab GEMM's capture-safe 128-row tile; a wider declaration runs the
  graph route eagerly and cannot live in a CUDA graph -- `plan.declaration["capture_safe_rows"]` says so); TP 1 (24 / 2) is the
  served and measured geometry; the TP 2 (12 / 1) and TP 4 (6 / 1) shards are declined by name (`proj_slab_route`: their slabs,
  N 7296 / 4224, are off the capture-safe small-M projection route on every served part -- see Errors); no LSE output, no sliding window, no soft-cap,
  no attention sink.
- The decode indexer selects PER ROW (every row at its own position with its own visible count: the serving stacks' function); the
  step-0 shared list (`index_list_form="shared"`) is declined by name. Every row of a step commits (a rejected draft's ring row and
  completed block are rewritten by the next step at the same positions: the ring is position-indexed, a block recompute lands on its own
  row); the two-phase commit is a cuDNN Frontend follow-up.
- The first plan of a form pays the JIT (tens of seconds); later plans of the same kernels are served from the in-process memo and,
  across processes, from the compiled-plan cache below.

## Caches and environment

The same two on-disk caches as the Qwen 3.5 package, under the SAME root (`FLASHINFER_CACHE_DIR/qwen_3_5_block/{compiled_plans,
gemm_sources}`: one vendored tree, one cache; both packages' plans share it), the same variables honoured
(`CUDNN_FRONTEND_COMPILED_CACHE`, `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE`, `CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES`,
`CUDNN_FRONTEND_GEMM_KERNEL_CACHE`); `plan.declaration["caches"]` reports the locations in use. See the Qwen 3.5 README, "Caches and
environment", for the full table.

## Where the kernels come from

The backend is the Qwen 3.5 package's vendored cuDNN Frontend subset (`../qwen_3_5_block/_cudnn_frost/`, 118 files at the pinned commit
`977757763` of the cuDNN Frontend `develop` branch -- the block's forward with the in-block indexer's decode step, the block-sparse d256
core with its two bodies (the tcgen05 core and the lean decode tile, plus the visible-set helper they share) and its paged indexer kernels
(the fused prepare, the compress step, the paged scorer, the decode top-k) and their DSL-free hosts (the `QSA` selection package), beside
the dense forms' launches --, every file with its NVIDIA SPDX header, MIT or Apache-2.0; the import roots rewritten and the declared patches
of `../qwen_3_5_block/_cudnn_frost/VENDOR.md`; per-file digests in `VENDOR_MANIFEST.json`). The pinned commit carries the in-block
indexer's decode step (`QsaSpec(index_source="indexer")` over the pools: the paged compressed-key cache on the main block table, the
raw-key ring, the paged scorer, the decode top-k, the measured `fuse_prepare` default), the block-sparse decode schedule's split rule, the
chain's programmatic-dependent-launch sites (served on the caller-list form; under the in-block indexer `use_pdl=True` stays a typed
decline), the e4m3 compressed-key cache arms (served by the tree; this entry declines `index_cache_dtype=torch.float8_e4m3fn` by name until
its own e4m3 cache surface lands) and the LEAN decode tile of the sparse core behind the `sparse_tile_form` knob with its measured AUTO
default on cc 10.7 -- since this pin the bound counts the lean grid's own unit, `batch x H_kv <= 4` CTAs per chunk at every served step
width (its DSL-free rules module `sdpa/fwd/sparse_tile_rules.py` is the one file this pin adds to the closure); for this entry the batch 1
and 2 decode steps at `q_len_per_req 1`, the in-block indexer's per-row list being served by the lean body at one row only. `qwen_3_8_block_support(...).source`
reports the pin at runtime (the same record as the Qwen 3.5 package's). The deployed process never imports the `cudnn` package nor the
cuDNN library. Re-vendoring: `../qwen_3_5_block/_vendoring/` (its README is the recipe; the tracer's `decode_qsa` forms declare this route,
the batch-1 cells among them: the lean kernel module is loaded by path on that route only, so the closure needs their trace).

## Validation

`tests/experimental/qwen_3_8_block/`: the decode form against the TWO-STAGE oracle (a torch reference prefill primes the pools, the ring
and the paged compressed-key cache through the vendored compress kernel; then (1) the block's selection == the fp64 top-k of its own
operands up to ties at the k-th boundary, the counts rule, the canonical list, (2) `out` within the block suite's budget of the fp32
oracle fed THAT list; the written pool / ring / completed cache rows vs the reference) on the served 24 / 2 geometry at `q_len` 1 and 4
(the 12 / 1 and 6 / 1 shards are tested for their typed `proj_slab_route` decline only, at the probe and at `prepare`), a padded entry, a
sequence past the identity bound and one whose step completes a block, page 16 / 32 / 64 / 128, the UNIFIED
cache spelling bitwise the NHD one, the ring-slots arm bitwise the derived one, the fused prepare bitwise the unfused twin and the
position-table arm bitwise the per-token arm, the launch census by kernel count (8 at a split) and by kernel name (the lean decode tile at
batch 1 under the AUTO knob -- one `cudnn_sparse_lean_kernel`, no tcgen05 core --, the core under the explicit pin and at batch 4), the two
bodies of the sparse core on one primed batch-1 step (the list and every written row bitwise, `out` within the budget of each other, each
against the oracle, the workspace carve of each body), three carried steps with a rejected draft
re-stepped (the cache rows bitwise a from-scratch compress of the accepted stream), CUDA-graph capture of the FIRST run under
`set_sync_debug_mode("error")` replayed bitwise and over a new step, the functional decode captured through the plan cache and replayed
after its eviction, a cached step under the same mode, the typed declines at plan time and at run time, the compiled-plan cache's reload,
the functional entry binding each call's own position tables on a cache hit (another pair, a missing pair, a short pair; CPU),
the weight packer on both spellings bitwise the cuDNN Frontend loader, the position tables bitwise the per-token rows, the plan cache's
key and ownership contract on CPU (the body knob a key), the admission grid (the probe's verdict == the vendored declaration's on
geometries x step widths x list forms x cache dtypes x page sizes; a planted drift of every mirrored constant fires the tripwire, the
sparse-tile vocabulary included), the body knob's device-free facts (the envelope's per-cell rule at cc 10.7: the lean tile at batch 1 / 2,
the core past the bound and at `q_len 2`, the explicit bodies, no body over e4m3 pools; the host declaration's readbacks) and the
import-isolation tests (the `cudnn` package blocked; the closure modules the route added import alone, the lean kernel module included).

Validated on an SM107 (cc 10.7) GPU with `nvidia-cutlass-dsl` 4.8.0 and torch 2.14 in a Python environment without the cudnn-frontend
package; the public and internal CI matrices have no SM107 runner, so the GPU tests skip there and the CPU tests run everywhere.

No performance number is claimed in this draft beyond the cuDNN Frontend side's own per-layer measurement of the same kernels; the
FlashInfer entry adds host work only (invisible under CUDA-graph replay), to be confirmed through this entry on a 212-SM-class part
before the PR leaves draft.

Owner: @RomanAnders90. Tracking issue: #6805 (the Qwen 3.5 block's; the 3.8 sibling is its promised next step). Graduation intent:
re-vendor at the cuDNN Frontend `develop` commit, add the prefill / extend forms of the indexer when their cuDNN Frontend PR lands,
finalise the signatures with the vLLM integration, then graduate with the Qwen 3.5 block to a stable attention namespace.
