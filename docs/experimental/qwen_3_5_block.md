# Qwen 3.5 gated attention block (experimental)

**Status: experimental API and experimental backend.** Everything here -- the names, the signatures, the three forms and
the vendored kernels -- may change or be removed without deprecation. Calling an entry point of
`flashinfer.qwen_3_5_block` is the opt-in (it emits the standard `ExperimentalWarning` once); no environment variable is
read, nothing is reachable from a stable API's `backend="auto"`, and nothing is registered for ahead-of-time compilation
(JIT only, never in `flashinfer-jit-cache` / `flashinfer-cubin`).

## What it computes

The whole full-attention sub-layer of a Qwen3.5-style model, from the post-layernorm hidden states to the attention
output, in bf16 on SM107 (Rubin, compute capability 10.7):

```
hidden_states [T, d_model]
  -> (1) QKV + gate projection            W_qkvg [n_qkvg, d_model], bands Q | GATE | K | V
  -> (2)+(3) QK-RMSNorm + partial RoPE     fused into (1)'s epilogue (zero-centred norm weights as (1 + w); rope_dim leading dims rotate)
  -> (4w) paged-KV write-through           post-RoPE K / V rows into the serving stack's cache at slot_mapping (folded into (1))
  -> (4) GQA attention, head dim 256       over the step's own tokens (prefill) or the cache (extend / decode), causal, bottom-right diagonal
  -> (5) O *= sigmoid(GATE)                in (4)'s epilogue (prefill / extend) or in the decode split's combine
  -> (6) out projection                    W_o [d_model, H_q * 256]
  -> out [T, d_model]
```

Three serving forms over the stack's own paged KV cache, each its own compiled plan:

| form | entry point | step | what attention reads |
|---|---|---|---|
| prefill | `qwen_3_5_block_bf16_prefill` | a uniform `[batch, seq]` step, no cached prefix | the step's own tokens |
| extend | `qwen_3_5_block_bf16_extend` | a packed step of new tokens behind cached prefixes (chunked prefill, prefix-cache hit, speculative verify; an empty prefix is the prefill case), bounded by a token capacity | each sequence's whole logical range from the cache |
| decode | `qwen_3_5_block_bf16_decode` | a uniform `[batch, q_len]` step, `q_len = 1 + drafts` | the cache (the Rubin d256 decode tile, split-KV; `decode_split_kv=1` runs it unsplit with the gate as its own launch) |

`prepare_qwen_3_5_block_bf16(form, ...)` returns the plan behind each form (`Qwen35BlockPlan`: `run`, `workspace_size`,
`declaration`, `supports`); `Qwen35BlockWrapper(workspace_buffer, kv_layout, geometry=...)` is the same thing in
FlashInfer's `plan()` / `run()` idiom over a caller-owned workspace -- the serving idiom. The functional entry points are the
convenience form: they keep a small LRU (32 entries, one per shape bucket, each owning a zeroed workspace) keyed on exactly what
a plan compiles in, so a serving stack's shape buckets need no bookkeeping of their own;
`flashinfer.experimental.qwen_3_5_block.api.clear_plan_cache()` drops them.

## A step, vLLM-shaped

```python
import torch
from flashinfer.qwen_3_5_block import (
    Qwen35BlockGeometry, qwen_3_5_block_support, qwen_3_5_block_pack_weights,
    qwen_3_5_block_rope_tables, qwen_3_5_block_bf16_decode, qwen_3_5_block_bf16_extend,
)

geometry = Qwen35BlockGeometry(d_model=4096, num_qo_heads=32, num_kv_heads=2)   # Qwen3.5-397B at TP 1; 16 / 1 at TP 2, 8 / 1 at TP 4, ...
assert qwen_3_5_block_support(torch.device("cuda"), geometry).served               # the init-time dispatch predicate (never raises)

# load time: the merged qkv_proj weight (rows [q_proj (per-head [q_h | gate_h]) | k_proj | v_proj]) -> the block's bands
weights = qwen_3_5_block_pack_weights(layer.qkv_proj.weight, layer.q_norm.weight, layer.k_norm.weight, geometry)
w_o = layer.o_proj.weight                                                           # [d_model, H_q * 256], used as is

# per step: the cache the stack allocated (FlashInfer NHD 5-D [num_pages, 2, page_size, H_kv, 256]; HND or a (k, v) pair also bind)
cos, sin = qwen_3_5_block_rope_tables(positions, rotary_emb.cos_sin_cache)        # or the tables the stack already holds
out = qwen_3_5_block_bf16_decode(hidden_states, cos, sin, weights, w_o, geometry=geometry,
                                 kv_cache=kv_cache, slot_mapping=slot_mapping,
                                 block_table=block_table[:num_reqs], seq_lens=seq_lens[:num_reqs], q_len_per_req=1)

out = qwen_3_5_block_bf16_extend(hidden_states, cos, sin, weights, w_o, geometry=geometry,
                                 query_start_loc=query_start_loc, kv_cache=kv_cache, slot_mapping=slot_mapping,
                                 block_table=block_table, seq_lens=seq_lens,
                                 token_capacity=max_num_batched_tokens, num_actual_tokens=num_tokens)
```

## Arguments and layouts

- **`kv_cache`**: FlashInfer's 5-D `[num_pages, 2, page_size, H_kv, D]` (`kv_layout="NHD"`) or `[num_pages, 2, H_kv, page_size, D]`
  (`"HND"`) buffer, or a `(k_cache, v_cache)` pair of 4-D tensors in those layouts; bf16. The block binds its
  `[num_pages, H_kv, page_size, D]` views by strides (no copy, no repack); `D` must be contiguous, every other stride a
  16-byte multiple, the base 16-byte aligned. `page_size` is a positive multiple of 16 (16 / 32 / 64 / 128 validated) and is
  read from the cache's shape by the functional entry points. The pools' page count, layout and strides are compiled into a
  plan: another cache is another plan.
- **`block_table`**: the dense `[batch, max_pages]` int32 table as vLLM holds it (row `0` = the NULL block of a padded
  entry); any row stride (a row slice of a wider table binds as a view) -- the row stride is compiled in, so plan with the
  table the steps will bind. The CSR triple (`kv_indptr`, `kv_indices`, `kv_last_page_len`) is not accepted.
- **`seq_lens`**: `[batch]` int32, the LOGICAL KV length INCLUDING the step's new tokens (`0` = a padded entry).
- **`query_start_loc`** (extend): `[batch + 1]` int32 prefix sums of the new tokens per sequence (zero-length pads).
- **`slot_mapping`**: `[rows]` int32 / int64 flat slots `page * page_size + offset`; a negative slot writes nothing.
- **`rope_cos` / `rope_sin`**: `[rows, rope_dim]` tables with duplicated halves (what HF `apply_rotary_pos_emb` holds);
  `qwen_3_5_block_rope_tables(positions, cos_sin_cache, out_cos=, out_sin=)` builds them from vLLM's rotary cache. mRoPE / YaRN /
  NTK differ only in how the table is built.
- **`hidden_states` / `out`**: contiguous `[rows, d_model]` bf16 (prefill also takes `[batch, seq, d_model]`); `out` is
  caller-owned for captured runs (allocated when `None`).
- **`num_actual_tokens`** (extend): the live token count, a host int; on a plan of at most 128 tokens it must be a multiple of
  `plan.live_rows_multiple` (32 on a 64- or 128-token bf16 plan; see "Requirements and limits"), else a typed `ValueError` names
  the remedy.
- **`decode_split_kv`** (decode): `None` = the decode split model, `>= 2` the split tile with the sigmoid gate in its combine, `1`
  the unsplit tile with the gate as its own launch (`plan.declaration["stages"]` lists `_SigmoidGate`); a performance knob --
  the same function under every value.
- Every VALUE in the index tensors is device data (never read on the host); only the FORM is checked, host-side and
  typed, before any launch.

## Capture and replay

1. `prepare_*` / `plan` run outside CUDA-graph capture (they JIT-compile the kernels and allocate the workspace); under
   capture they raise a typed `RuntimeError`.
2. The FIRST `run` after a plan may be captured with no eager warm-up: the route creates nothing lazily, the captured
   execute launches only kernels and replays bitwise an eager run (tested).
3. A captured graph replays over NEW values of `seq_lens`, `block_table`, `slot_mapping`, `query_start_loc`, `hidden_states`
   and the tables written in place into the SAME buffers (addresses are baked). `num_actual_tokens` is a host int baked per
   capture: one capture per live-count bucket, shorter steps padded (finite pad rows, slot `-1`, `seq_lens == 0` for padded
   sequences) -- vLLM's capture-size buckets.
4. The pools must be FINITE where the kernels read past `seq_lens` inside the last live page (a zero-initialised cache is).
5. One plan / wrapper is a single-writer resource: do not re-plan while a captured or in-flight `run` may read its buffers.

## Errors

| condition | exception |
|---|---|
| not cc 10.7; `nvidia-cutlass-dsl` absent, below 4.8.0 or without the Rubin target; f16 / fp8 on the bf16 entry; `head_dim != 256`, `d_model % 256`, more than 32 decode rows | `NotImplementedError` naming the device / version / feature (and `qwen_3_5_block_support` reports it without raising) |
| a malformed tensor (dtype, rank, shape, strides, alignment, device), a page size off the contract, a call outside the plan's capacities, a token count off the small-capacity plans' multiple | `ValueError` naming the operand and the expected form |
| `prepare` / `plan` under capture | `RuntimeError` naming the remedy |

No silent fallback: an unserved case never runs a slower path.

## Requirements and limits

- SM107 (Rubin, cc 10.7) only; `nvidia-cutlass-dsl >= 4.8.0` (the first wheel with the `sm_107a` target; the repository-wide
  floor is unchanged -- the kernels gate at runtime and decline by version); torch with CUDA 13; `cuda-python`.
- bf16 activations, weights and pools; `head_dim = 256`; `d_model % 256 == 0`; GQA groups as the Qwen3.5 shards have them
  (32 / 2, 24 / 2, 16 / 1, 8 / 1, 4 / 1 ...); the decode envelope `q_len_per_req <= 4` claimed (the packed decode tile), up to 32
  routed on the unpacked tile and declared, a typed decline past it; no LSE output, no sliding window, no soft-cap, no fp8
  KV cache in this draft (the dtype suffix plan: `_fp8` / `_mxfp8` entry points later; the KV-cache dtype becomes an argument).
- EXTEND plans of at most 128 tokens (`token_capacity <= 128`) run the out projection on the small-capacity GEMM tile, whose
  token rows are the kernel's epilogue-vector axis: every `num_actual_tokens` must be a multiple of `plan.live_rows_multiple`
  (the widest vector that divides the capacity -- 32 on a 64- or 128-token plan, 16 on a 48-token plan, in bf16;
  `plan.declaration["out_proj_tile"]` names the tile). `run` refuses a live count off the multiple, typed, before any launch;
  hand `num_actual_tokens=None` with the token tensors at the capacity (pad rows finite, slot `-1`), round the live count up
  with pad rows, or plan above 128 tokens, where the forced 256-wide tile takes any count. vLLM's buckets (`max_num_batched_tokens`, thousands) are above the route; small verify batches
  on a small-capacity plan are what the rule guards. The cuDNN Frontend-side fix (declaring the live rows without the vector
  divisibility on that route) lifts the rule at a later re-vendor.
- The first plan of a form pays the JIT (tens of seconds); later plans of the same kernels are served from the in-process memo
  and, across processes, from the compiled-plan cache below.

## Caches and environment

The vendored tree keeps two on-disk caches. The adapter points both under FlashInfer's cache root
(`FLASHINFER_CACHE_DIR` = `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, `~/.cache/flashinfer` by default) at the first plan,
so a FlashInfer process never writes into the cuDNN Frontend package's own roots (`~/.cache/cudnn_frontend`,
`~/.cache/cudnn_gemm`); `plan.declaration["caches"]` reports the locations in use.

| cache | default location | variables honoured |
|---|---|---|
| compiled plans (the exported kernel objects of `compile_cached`, reloaded in milliseconds at the next start-up instead of a full JIT) | `FLASHINFER_CACHE_DIR/qwen_3_5_block/compiled_plans` | `CUDNN_FRONTEND_COMPILED_CACHE=<dir>` (a location of your own; it wins over the default), `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` (no persistence: every process JITs), `CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES` (the prune cap, 4 GiB; `0` = never prune), `CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0` (every compile reaches the files; a test knob) |
| rendered GEMM sources (the generated kernel modules the JIT imports; content-addressed, small) | `FLASHINFER_CACHE_DIR/qwen_3_5_block/gemm_sources` | `CUDNN_FRONTEND_GEMM_KERNEL_CACHE=<dir>` (set it yourself to choose; the adapter sets it only when unset, since the tree reads this variable live) |

The CuTe DSL's own MLIR cache (`CUTE_DSL_CACHE_DIR`) is the DSL's business, as for every CuTe DSL kernel in FlashInfer. Other
variables the vendored tree reads, all tuning or debugging knobs with sound defaults, none required: `CUDNN_FRONTEND_GEMM_ARCH_FAMILY`
(pins the GEMM template family, `sm100` on SM107), `CUDNN_FROST_DISABLE_MIXED_CGA` (disables the mixed-cluster fallback of the GEMM
JIT), `FROST_THD_CLUSTERS` / `FROST_THD_CTAS` / `FROST_THD_CTAS_PER_SM` (debug overrides of the packed-attention grid),
`XDG_CACHE_HOME` (only where a cache was not redirected). The tree's own experimental banner (`<class> is an experimental API`)
is filtered: FlashInfer's `ExperimentalWarning` is this API's single opt-in signal.

## Where the kernels come from

The backend is a subset of the cuDNN Frontend Python package vendored byte-faithfully under `_cudnn_frost/`
(95 files: the block's forward and its serving launches, the FROST SDPA adapter and its SM107 prefill / decode tiles and SM100
split combine, the FROST GEMM compiler and its kernel templates, the `tile_dsl` helpers; every file keeps its NVIDIA SPDX
header, MIT or Apache-2.0), with only the import roots rewritten and four declared, exact-text patches (P0-P2 replace the
nanobind extension's seams, P3 rewords eleven comment lines that named internal hosts or file paths; all listed in
`_cudnn_frost/VENDOR.md`; per-file digests in `_cudnn_frost/VENDOR_MANIFEST.json`). Pinned revision: cuDNN Frontend
`fe8b50da0` (1.32.0); `qwen_3_5_block_support(...).source` reports it at runtime. The deployed process never imports the
`cudnn` package nor the cuDNN library: the one native piece of that route, the SDPA launch binder, is served in Python
(`_sdpa_binder.py`, built on cuDNN Frontend's own retired Python binder implementation vendored from its test tree). The
kernels keep living in cuDNN Frontend and are re-vendored deliberately: the exporter, the closure list and the tracer live in
`_vendoring/` next to the tree (its README is the recipe; `VENDOR.md`, "How to re-vendor").

## Validation

`tests/experimental/qwen_3_5_block/`: the three forms against an fp32 torch oracle over the full sequences (Qwen3.5 24 / 2 at
`d_model` 2560 and 32 / 2 at 4096, page 16 / 32 / 64 / 128, both cache layouts and the `(k, v)` pair spelling, short / long /
padded sequences, decode `q_len` 1 and 4 on the packed tile and 8 / 32 routed on the unpacked one, the split model, an explicit
split and the unsplit tile with the gate as its own launch, the separate cache-write launch, extend behind prefixes and as two
chunks, the bounded token count and the small-capacity plans' live-row rule), CUDA-graph capture of the FIRST run of every form
under `set_sync_debug_mode("error")` replayed bitwise (decode also over a new step), cached steps under the same mode, the typed
declines at plan time and at run time, the compiled-plan cache's location and reload path, the weight packer at TP 1 / 2 / 4 / 8
(bitwise the cuDNN Frontend loader), the RoPE table helper, and the import-isolation test (the `cudnn` package blocked).
Validated on an SM107 (cc 10.7) GPU with `nvidia-cutlass-dsl` 4.8.0 and torch 2.14 in a Python environment without the
cudnn-frontend package; the public and internal CI matrices have no SM107 runner, so the GPU tests skip there and the CPU tests
(support, weights, RoPE, isolation) run everywhere, including under the CI images' `nvidia-cutlass-dsl` 4.7.0, where the
envelope declines by target.

No performance number is claimed in this draft; the decode levers and a fused decode step are in flight on the cuDNN
Frontend side and will be re-vendored before the final review.

Owner: @RomanAnders90. Tracking issue: TBD (opened with the PR). Graduation intent: re-vendor at each cuDNN Frontend
release while the decode levers land, finalise the signatures with the vLLM integration, then graduate to a stable
attention namespace with `docs/api/` pages.
