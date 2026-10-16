# Experimental MiniMax-M3 ICP decode indexer chain (Cake backend)

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection and no stable API changes its behaviour.

`flashinfer.msa_ops.prepare_msa_icp_indexer(...)` serves one decode step of
the index-context-parallel (ICP) indexer of MiniMax-M3 sparse attention on
compute capability 10.0 and 10.3. The index keys of a sequence are sharded by
rows across the `world_size` (2 or 4) ranks of a process group: every
128-token page (or 256-token page for two ranks) holds `rows_per_rank` FP8
key rows per rank inside a compound page of configurable pitch. For every
decode token and index head the chain

1. **scores** the token's FP8 query against this rank's rows of every visible
   page (FP8 products, FP32 accumulation, `-inf` for rows the token may not
   see) and keeps the exact top-16 pages of the row under the canonical key
   (score descending, global block id ascending, `+0.0 == -0.0`) -- the
   **fused scorer** program, one CTA per (request, split) streaming the page
   fragments with TMA into tensor-core MMAs, the top-16 kept in the epilogue
   as compact candidate lists in a per-row pool;
2. **merges** the row's lists into the rank-local top-16, **pushes** the 16
   words into the owner rank's symmetric pool and, on the rows whose head this
   rank owns, **polls** the peers' words and writes the merged global block
   ids ascending with a `-1` tail, the forced (newest) block injected once and
   `n_ordinary` ordinary slots -- the **exchange merge** program, one CTA per
   (token, head) row.

## Usage

```python
import torch.distributed as dist
from flashinfer.msa_ops import (
    create_msa_icp_indexer_workspace,
    destroy_msa_icp_indexer_workspace,
    prepare_msa_icp_indexer,
)

# Collective, once per decode engine (not inside CUDA Graph capture).
workspace = create_msa_icp_indexer_workspace(
    world_size=dist.get_world_size(), rank=dist.get_rank(), tokens_cap=max_tokens, device=device
)

runner = prepare_msa_icp_indexer(
    index_q, index_k, block_table, query_start_loc, seq_lens, positions, active_rows,
    forced_column, forced_block, n_ordinary,
    workspace=workspace, query_len=4, out=out, candidates=candidates,
)
out = runner()                 # no allocation, no host synchronization, capturable
torch.cuda.synchronize()
assert workspace.status_value() == 0   # outside any capture; non-zero is fatal for the step

destroy_msa_icp_indexer_workspace(workspace)   # collective, after the last launch completed
```

Tensor contract (all on one CUDA device, read on device only):

| tensor | dtype / shape | meaning |
| --- | --- | --- |
| `index_q` | fp8_e4m3fn `[T, 4, 128]` contiguous | queries of this rank's `T = request_count * query_len` decode tokens |
| `index_k` | fp8_e4m3fn `[num_pages, rows_per_rank, 128]` view, `stride(0)` = page pitch (16-byte multiple), dense fragment | this rank's rows of every physical page |
| `block_table` | int32 `[request_count, columns]` | physical page ids, live prefix first; `columns` bounds every request |
| `query_start_loc` | int32 `[request_count + 1]` | token offsets (`[0] == 0`, `[-1] == T`, at most `query_len` tokens per request) |
| `seq_lens` | int32 `[request_count]` | visible history length including the decode tokens |
| `positions` | int64 `[T]` | causal position of each token |
| `active_rows` | uint8 `[T]` | inactive tokens score nothing and produce pad lists |
| `forced_column` | int32 `[T]` | column of the forced block inside this rank's readable window, else `-1` |
| `forced_block` | int32 `[T]` | global id of the forced (newest) block, `-1` for none |
| `n_ordinary` | int32 `[T]` | ordinary slots (at most 15 whenever a forced block exists) |
| `out` | int32 `[T, 4 // world_size, 16]` | merged ids of the heads this rank owns, ascending, `-1` tail |
| `candidates` | fp32 `[T, 4, 16, 2]` | rank-local merged lists (score, int32 id bits), best first, `(-inf, -1)` padded |

Preparation validates the tensors from their metadata, decides the **split
count** (`cake_backend.default_split_count`: the largest split that keeps
every scorer CTA resident at once on the device's SMs) and the **pool / merge
geometry** (`cake_backend.pool_geometry`: the merge CTA's warps and entries
per thread covering every request the block table admits), binds both
generated programs and returns a runner. One launch covers all `T` tokens of
`index_q` (every per-token tensor is exactly `[T]` and `T <= tokens_cap` of
the workspace); there is no launch-window parameter. The programs for the
index geometry (`query_len`, `rows_per_rank`, page size) and the resulting
merge geometry must be registered in this checkout (`cake_jit.PROGRAMS`);
otherwise preparation raises `NotImplementedError` instead of entering
another kernel.

## Workspace, exchange and status

The workspace (`cake_workspace.MSAIcpIndexerWorkspace`) owns the symmetric
transport pool -- two parity halves of `tokens_cap * (4 // world_size)` rows
times `world_size` source regions of 16 words, allocated through PyTorch
symmetric memory so every peer's half is mapped into this rank's address
space -- plus the per-row generation words, the status word and the per-CTA
timestamps of the merge launch. Creation and destruction are collectives over
the group and are not allowed inside CUDA Graph capture. The per-peer base
pointers form the peer table the host binding passes to the merge by value.

Every rank of the group launches the chain for the same step; the generation
words advance per launch and keep the ranks in step, so a prepared runner (or
a CUDA Graph capturing it) is replayed step after step without host work.
After a step completes, read `workspace.status_value()` outside any capture.
Non-zero is fatal for the step: 1 = NaN among the ordinary candidate scores,
2 = a row's list layout exceeded the pool stride, 4 = the merge's refine loop
made no progress, 8 = a poll passed `timeout_us` (default 50 ms; the missing
peer words merged as absent), 16 = `n_ordinary == 16` together with a forced
block. Call `workspace.reset()` before reusing the workspace after a non-zero
status.

Single-process tests emulate the ranks: `MSAIcpIndexerWorkspace.emulated(...)`
builds the ranks' pools as plain device tensors, and the merge is driven as
its serialised halves (`runner.launch_merge(PHASE_PUSH_ONLY)` on every rank,
then `PHASE_POLL_ONLY`). The kernels see device pointers either way.

## Numerics and validation

The scorer multiplies FP8 E4M3 values exactly and accumulates in FP32; the
canonical key ranks the exact FP32 scores, so the local lists and the merged
output are bit-exact against the CPU oracle in
`tests/test_helpers/cake_msa_icp_inputs.py` (full-page FP32 scores, canonical
top-16, merge with forced-block injection). The tests
(`tests/experimental/test_cake_msa_icp_indexer.py`) check the registry and
the planning helpers on the CPU, the emulated chain on one device (oracle,
replay, CUDA Graph, no allocation) and the public API over NCCL on two or
four devices. `benchmarks/bench_cake_msa_icp_indexer.py` times the chain
with CUPTI under `torchrun`.

## Files

- `cake_jit.py` -- program registry (`PROGRAMS`, `ARG_PLANS`, `POLICY`, written
  by the generator) and the per-architecture JIT specs.
- `cake_backend.py` -- validation, planning, binding, the runner.
- `cake_workspace.py` -- the symmetric-memory workspace.
- `csrc/cake_msa_icp_indexer/` (repository root) -- the generated device and
  host-binding sources plus their shared headers; receipt-bound, do not edit.
