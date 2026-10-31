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
| Exchange | the caller's one all-to-all per layer | (in the publish: peer stores) |
| Merge | `msa_icp_merge_candidates(prefill=True)` | `msa_icp_merge_candidates` (legacy or tagged rows) |

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
architecture).  The facade runs it when `decode(..., pool=...)` is given.

*The two chains of one sparse layer.*

With programmatic dependent launch (`enable_pdl`), every kernel waits on its
predecessor before it reads data an earlier kernel of the step wrote, with one
exception: on compute capability 10.0 and 10.3 at 64 or 128 local K rows per
page (P128 TP2, P256 TP2 and TP4), the score plane reads `seq_lens` and
`block_table` and starts loading the history index-K pages (those ending at or
before `seq_len - decode_query_len`) before its wait.  The kernel launched just
before the score plane must not write them, and must either be launched
without programmatic dependent launch or wait on its own predecessor before it
triggers its dependents.  `enable_pdl=False` removes the requirement; the fused
chain has no such read.

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

The package is a vendorable Cake package: `build.json` lists every
`<program>_<arch>` library with its sources and flags, `cake_build.py` builds
or loads them, and `cake_jit.py` is the registry that routes each call by
role, local K rows per page and the device's exact architecture. The exporter
writes `csrc/`, `build.json`, `cake_build.py` and the derived fields of
`cake_jit.py`.
