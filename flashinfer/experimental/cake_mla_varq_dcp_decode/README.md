# Experimental Cake MLA variable-query decode with decode context parallelism

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4658 (tracker #4254).

`flashinfer.mla.cake_mla_varq_dcp_decode(...)` and
`flashinfer.mla.prepare_cake_mla_varq_dcp_decode(...)` serve the compact
variable-query DeepSeek MLA decode of one decode-context-parallel (DCP) rank
with the semantics of
`cute_dsl_mla_decode(..., is_var_seq=True, return_lse=True, cum_seq_lens_q=...,
max_q_len=..., enable_dcp=True, cp_world=..., cp_rank=...,
causal_seqlens_kv_global=...)`:

* keys are `576 = 512 latent + 64 rope` wide, values are the 512 latent dims;
  query and KV cache share one dtype, BF16 or FP8 (e4m3, dense 576-wide rows);
* the rank-local paged cache has 32-, 64- or 128-token pages;
* request `b` owns the compact query rows
  `cum_seq_lens_q[b] .. cum_seq_lens_q[b + 1]` (empty requests are legal) and
  `max_q_len` is the static per-request capacity; `num_heads <= 128`;
* rank `cp_rank` of `cp_world` holds the global positions
  `cp_world * k + cp_rank`; local key `k` of request `b` is visible to query
  token `q` iff `k < seq_lens[b]` and
  `cp_world * k + cp_rank <= causal_seqlens_kv_global[b] - q_len_b + q`;
  with `enable_dcp=False` the rank-local lengths are the causal bounds;
* the outputs are compact BF16 `out [total_q, num_heads, 512]` and natural-log
  FP32 `lse [total_q, num_heads]`; a row without any visible key (including a
  whole empty rank) writes `out = 0` and `lse = -inf`, so the LSE-weighted
  cross-rank merge of the upstream tests applies unchanged.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `query` | `[total_q, num_heads, 576]` | bfloat16 or float8_e4m3fn |
| `kv_cache` | `[num_pages, page_size, 576]` (or `[num_pages, 1, page_size, 576]`) | same as `query` |
| `block_tables` / `page_table` | `[batch_size, max_pages]` | int32 |
| `seq_lens` | `[batch_size]` rank-local key counts | int32 |
| `cum_seq_lens_q` | `[batch_size + 1]` | int32 |
| `causal_seqlens_kv_global` | `[batch_size]` global causal bounds (DCP) | int32 |
| `workspace_buffer` | `[>= workspace bytes]` | uint8 |
| `out` | `[total_q, num_heads, 512]` | bfloat16 |
| `lse` | `[total_q, num_heads]` | float32 (natural log) |

`max_seq_len` (the largest rank-local length) and `max_q_len` are host
integers exactly as in `cute_dsl_mla_decode`; the host never reads a CUDA
tensor's contents, so preparation and launch are CUDA-Graph capturable.

```python
import math
import torch
from flashinfer.mla import cake_mla_varq_dcp_decode
from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
    cake_mla_varq_dcp_decode_workspace_size,
)

num_heads, cp_world, cp_rank = 128, 8, 0
q_lens, local_lens = [1, 3, 0, 2], [4096, 40, 1, 512]   # rank-local key counts
global_lens = [cp_world * n - (cp_world - 1 - cp_rank) for n in local_lens]
query = torch.randn(sum(q_lens), num_heads, 576, device="cuda", dtype=torch.bfloat16)
pages = [max(1, -(-n // 64)) for n in local_lens]
kv_cache = torch.randn(sum(pages), 64, 576, device="cuda", dtype=torch.bfloat16)
page_table = torch.zeros(len(q_lens), max(pages), dtype=torch.int32, device="cuda")
first = 0
for b, n in enumerate(pages):
    page_table[b, :n] = torch.arange(first, first + n, dtype=torch.int32)
    first += n
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
workspace = torch.empty(
    cake_mla_varq_dcp_decode_workspace_size(
        batch_size=len(q_lens), max_q_len=max(q_lens), num_heads=num_heads,
        max_seq_len=max(local_lens), num_sms=num_sms,
    ),
    dtype=torch.uint8, device="cuda",
)
out, lse = cake_mla_varq_dcp_decode(
    query, kv_cache, workspace, page_table,
    torch.tensor(local_lens, dtype=torch.int32, device="cuda"),
    max(local_lens), 1.0 / math.sqrt(512),
    cum_seq_lens_q=torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
                                dtype=torch.int32, device="cuda"),
    max_q_len=max(q_lens), enable_dcp=True, cp_world=cp_world, cp_rank=cp_rank,
    causal_seqlens_kv_global=torch.tensor(global_lens, dtype=torch.int32, device="cuda"),
)
```

## Program structure

The host plan (`cake_backend.plan_varq_dcp_decode`, a port of the Cake
production plan) fixes a rectangular launch over
`batch_size * ceil(max_q_len * num_heads / 128)` items and picks one of the
traced main-kernel variants from host-known scalars only: KV dtype, page size,
scheduler capacity (`item_groups` 4 or 16: 128 or 512 items) and the
`partition_mode`:

* `partition_mode = 0` (ticket scheduler): a scheduler warp per two-CTA
  cluster claims units of KV tiles from a self-resetting device ticket
  counter; items that split across units write BF16 partials to the workspace
  and the split-KV merge kernel (`mla_varq_dcp_merge`), launched
  programmatically behind the main kernel, merges them while the last
  clusters still run. The merge launch happens only when the plan can split
  (`launches_merge(plan)`).
* `partition_mode = 1` (balanced static partition, chosen for few long items
  or few short whole items): every cluster derives its unit once and the
  item's clusters merge in-kernel; no second launch.

`cake_jit.MODULES` registers every generated physical module, `ROUTES` maps
`"<dtype>_p<page_size>_g<item_groups>_pm<partition_mode>__<arch>"` to its main
module and `MERGE_MODULES` maps an architecture to its merge kernel; all three
are populated by the generated-program export. A plan that selects a variant
this checkout does not register raises `NotImplementedError` naming the
variant (there is no fallback to another backend).

The `workspace_buffer` holds the BF16 split partials, the scheduler counters,
the partial-slot flags, the split table and the merge control words
(`workspace_layout`). The scheduler state is live across launches (it
self-resets by launch parity), so one workspace serves one prepared runner at
a time; `prepare_*` zeroes it. `cake_mla_varq_dcp_decode_workspace_size`
sizes it for one plan, `max_cake_mla_varq_dcp_decode_workspace_size` bounds it
for any `max_seq_len`.

## Limits of the current route

BF16 or FP8 e4m3 query/KV of one dtype, 576-wide keys / 512-wide values,
page size 32, 64 or 128, `num_heads <= 128`, `batch_size * ceil(max_q_len *
num_heads / 128) <= 512` items, at least one compact query row, SM100 (B200)
and SM103 (B300) only. The generated programs registered by the export cover
the variants selected by the validated shape set (see the tests); other
`(dtype, page_size, item_groups, partition_mode)` variants raise until they
are exported.

See `tests/experimental/test_cake_mla_varq_dcp_decode.py` for the torch
reference (the upstream DCP test semantics), the validated shape set and the
cross-rank merge check, and `benchmarks/bench_cake_mla_varq_dcp_decode.py`
for the comparison against `cute_dsl_mla_decode` on the 24 performance rows.
