# Experimental MiniMax-M3 ICP indexer kernels (Cake backend)

This API and its Cake backend are experimental and may change without
backward compatibility. Calling it emits FlashInfer's experimental API
warning; no stable API selects it automatically.

The `flashinfer.msa_ops.msa_icp_*` entry points and the
`prepare_msa_icp_indexer` facade are the per-rank kernels of the
index-context-parallel (ICP) indexer of MiniMax-M3 sparse attention on compute
capability 10.0, 10.3 and 10.7. They follow the MSA sharded-indexer contract,
so a serving engine can switch providers without changing its cache layout or
buffers. The API reference is `docs/api/sparse.rst` ("MiniMax-M3 ICP
indexer").

| Step | Prefill / mixed | Decode |
|---|---|---|
| Plan | `msa_icp_index_score_sharded_plan_device`, once per query window and step | -- |
| Score | `msa_icp_index_score_sharded` per query chunk into caller score scratch | `msa_icp_decode_index_score_sharded` into caller score rows |
| Local Top16 | `msa_icp_topk_select_prefill` into the caller's carrier `[H, T, 33]` | `msa_icp_topk_select_publish` into the owner ranks' receive rows |
| Fused option | `msa_icp_prefill_score_select`: score and local Top16 in one chain, into the same carrier | `msa_icp_decode_score_publish` |
| Exchange | the caller's one all-to-all per layer | (in the publish: peer stores) |
| Merge | `msa_icp_merge_candidates(prefill=True)` | `msa_icp_merge_candidates` (legacy or tagged rows) |

The prefill fused option replaces the score and local Top16 steps of one
query window with `msa_icp_prefill_score_select`: a list scorer keeps the
Top16 list of each plan work item in a caller-owned uint8 `pool`
(`msa_icp_prefill_pool_bytes`, 32 KiB per list piece) and a list merger writes
the carrier rows of `msa_icp_topk_select_prefill`, byte for byte, without the
FP32 score scratch.  The facade runs it when `prefill_local(..., pool=...)` is
given (`score_scratch` may then be `None`), except that a window of at most
`prefill_short_tokens` tokens (`cake_backend.SHORT_PREFILL_MAX_TOKENS`, per
layout) runs the score rows and the Top16 select in a scratch the facade
allocates once (`[prefill_short_tokens, H, pitch]` float32 at the
`max_model_len` pitch): for short windows the list merge costs more than the
score rows it saves.  `prefill_pool_shape()` sizes the
pool for windows of `prefill_window_tokens()` = `min(8192,
max_num_batched_tokens)` tokens.  `MSAIcpIndexerPlan.fused_prefill` is the
serving default per layout (`cake_backend.FUSED_PREFILL`) and is `False` on
every layout in this version.

The decode column is the default chain.  Its score plane cuts each request
into one wave of the scorer (SMs x resident CTAs per SM / batch, at most 128
splits); on compute capability 10.3 at P128 TP4 the plane runs a
four-CTA-per-SM program and, from 14 requests on, at most 32 splits
(`cake_backend.plane_splits`).  The scores do not depend on the split count.
An optional fused chain replaces the
score and local Top16 steps with `msa_icp_decode_score_publish`: the scorer
keeps per-split Top16 lists in a uint8 `pool` (`msa_icp_decode_pool_bytes`)
and a list publisher merges them, giving the same receive rows.  It cuts each
request into at most 48 splits (`cake_jit.TOPK16_SPLIT_CAP`, per
architecture).  The facade runs it when `decode(..., pool=...)` is given, and
without a pool on compute capability 10.7 at P128 (TP2 and TP4) from 8 requests on, in
a pool it allocates once (`cake_backend.fused_decode_min_batch`;
`MSAIcpIndexerPlan.fused_decode_min_batch` is `None` on every other layout).
There the measured end-to-end decode step is faster with the fused chain;
below 8 requests and at P256 TP4 the score plane is faster.

*The two chains of one sparse layer.*

With programmatic dependent launch (`enable_pdl`), every kernel waits on its
predecessor before it reads data an earlier kernel of the step wrote, with one
exception: on compute capability 10.0 and 10.3 at 64 or 128 local K rows per
page (P128 TP2, P256 TP2 and TP4) the score plane, and on 10.7 at 32 or 64
(P128 TP4 and TP2, P256 TP4) both the score plane and the fused chain's list
scorer, read `seq_lens` and `block_table` and start loading the history
index-K pages (those ending at or before `seq_len - decode_query_len`) before
their wait.  The kernel launched just before the scorer must not write them,
and must either be launched without programmatic dependent launch or wait on
its own predecessor before it triggers its dependents.  The decode merge of
`MSAIcpIndexerPlan.decode` with tagged rows at world size 4 (vLLM's
decode-only steps) reads
`num_valid_pages`, its receive rows and the merge counters before its wait and
waits before it exits: no kernel of the step may write `num_valid_pages`, and a
kernel launched without programmatic dependent launch must run between two
decodes on one slot set (vLLM's decode K writer).  `enable_pdl=False` removes
both requirements.

The caller owns every buffer and every collective step: it allocates the
receive storage (`msa_icp_comm_layout`) in symmetric memory, exchanges the
peer addresses into `recv_bases`, zeroes the storage and the generation
counters once, rotates the three slot sets, and runs the prefill all-to-all.
The kernels never allocate, never read a device value on the host and run on
the current stream, so every call can be captured in a CUDA Graph once its
programs are loaded (`MSAIcpIndexerPlan.warmup()`).

Run `warmup()` on every rank before the first decode or prefill exchange and
before CUDA Graph capture.  It builds or loads every program of the device's
architecture and loads each kernel into the device's context.  Under CUDA lazy
module loading, a kernel loaded at its first launch can stall behind a peer's
merge that already spins on its rows, until the spin limit stops it.

*Step prepare.*  Within one step, every sparse layer makes the same launches
with the same arguments except the query and the layer's K shard.
`MSAIcpIndexerPlan.prepare_prefill_step(windows, score_scratch,
num_valid_pages, send, received, output)` and `prepare_decode_step(...)` (the
arguments of `prefill_local` / `prefill_merge` / `decode`) return step objects
whose `local(index_q, index_k)`, `merge()` and `run(index_q, index_k, slot=)`
make exactly the facade's launches.  The first layer runs the facade's host
checks and records the launches; later layers check only the query and K
metadata, swap those two tensors and launch.  The host time per layer drops;
the device work is unchanged.  `debug=True` runs the full per-call checks on
every layer and raises if a launch differs from the prepared one.

*Layer graphs.*  `MSAIcpIndexerPlan.layer_graphs(capture_after=1, max_keys=4)`
captures each layer of a step in its own CUDA Graph and replays it in later
steps whose launches are the same.  Per step,
`begin_step(prefill=, decode=, exchange=, exchange_key=, decode_tokens=)`
takes the step objects and the caller's exchange (one call per layer, from the
prefill `send` into `received`); `run(layer, index_q, index_k, slot=)` runs
one layer (decode rows first in `index_q`).  The key of a step is every launch
argument except the query and K (tensor address and metadata, scalar value),
the K shard of each layer, the query metadata and `exchange_key`.  A key's
first `capture_after` steps run eagerly, the next one captures and replays,
later ones only copy the query into a staging buffer and replay.  The
exchange is captured inside each layer's graph, once per layer.  Device data
(plans, block tables, lengths, K pages) may change between steps.  To keep
one key over steps of different lengths, build every window plan with
`max_seq_len` the model limit and size the score scratch with
`prefill_scratch_shape()`; with the fused path the pool needs no change.

*Device token count.*  The backend's `topk_select_prefill`,
`topk_select_publish`, `decode_score_publish` (its publisher) and
`merge_candidates` take an optional `token_count`, an int32 `[1]` device
tensor.  With it the call launches the `_dc` record of its role over the
grid of the passed tensors and only the rows below `token_count[0]` are
selected, published or merged; the rows past it are not written.  One
captured launch then serves every count up to its grid.

The package is a vendorable Cake package: `build.json` lists every
`<program>_<arch>` library with its sources and flags, `cake_build.py` builds
or loads them, and `cake_jit.py` is the registry that routes each call by
role, local K rows per page and the device's exact architecture. The exporter
writes `csrc/`, `build.json`, `cake_build.py` and the derived fields of
`cake_jit.py`.
