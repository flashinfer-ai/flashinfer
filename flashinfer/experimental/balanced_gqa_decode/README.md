# Experimental balanced paged GQA decode (Cake backend)

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4832 (the "kernel-owned
length-aware work scheduler" for ragged paged decode).

`flashinfer.decode.prepare_balanced_batch_decode_with_kv_cache(...)` serves
BF16 paged GQA decode with a **fixed ratio of eight query heads per KV head**,
head dimension 128 and 16-token pages, for ragged batches whose KV lengths
differ widely between requests. One persistent launch (one CTA per SM) serves
the whole batch:

- A scheduler warp reads the device `seq_lens` buffer and derives, in two
  short passes, a chunk length `L` (in 256-token KV block pairs) and four
  length buckets. Long requests are split into `L`-pair chunks; short requests
  and remainders are packed after the full chunks, longest first.
- Work items `(query row tile, kv head, KV block range)` are claimed from a
  self-resetting global ticket counter and decoded online (no prefix arrays,
  no plan tensor, no second launch). Split tiles publish FP32 partials to a
  shape-independent workspace; the last CTA to finish a tile merges them.
- Uniform batches take the same path: when splitting cannot shorten the
  launch, every tile is one item, like a plain persistent decode kernel.
- Because nothing about the plan is decided on the host, a runner captured
  once into a CUDA Graph replays correctly for any KV-length distribution
  written into `seq_lens` afterwards (validated in the tests).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `query` | `[batch * q_len_per_req, num_q_heads, 128]` | bfloat16 |
| `k_cache`, `v_cache` | `[num_pages, num_kv_heads, 16, 128]` each (`kv_cache=(k_cache, v_cache)`, `HND`) | bfloat16 |
| `block_tables` | `[batch, max_pages]` | int32 |
| `seq_lens` | `[batch]` (KV length including the new tokens) | int32 |
| `workspace_buffer` | `>= balanced_gqa_decode_workspace_size(device)` bytes | uint8 |
| `out` | `[batch * q_len_per_req, num_q_heads, 128]` | bfloat16 |

Query head `h` attends to KV head `h // 8`. With `q_len_per_req > 1`
(speculative / MTP verify) row `j` of request `b` attends causally to the
first `seq_lens[b] - (q_len_per_req - 1 - j)` KV positions, the same contract
as `trtllm_batch_decode_with_kv_cache(..., q_len_per_req=...)`.

```python
import torch
from flashinfer.decode import prepare_balanced_batch_decode_with_kv_cache
from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
    balanced_gqa_decode_workspace_size,
)

batch, num_kv_heads, max_pages = 16, 1, 16384  # up to 256K tokens per request
num_q_heads = 8 * num_kv_heads
q = torch.randn(batch, num_q_heads, 128, dtype=torch.bfloat16, device="cuda")
k_cache = torch.randn(batch * max_pages, num_kv_heads, 16, 128, dtype=torch.bfloat16, device="cuda")
v_cache = torch.randn_like(k_cache)
block_tables = torch.arange(batch * max_pages, dtype=torch.int32, device="cuda").view(batch, max_pages)
seq_lens = torch.randint(1, max_pages * 16 + 1, (batch,), dtype=torch.int32, device="cuda")
workspace = torch.empty(balanced_gqa_decode_workspace_size(q.device), dtype=torch.uint8, device="cuda")

decode = prepare_balanced_batch_decode_with_kv_cache(
    q, (k_cache, v_cache), block_tables, seq_lens, workspace
)
out = decode()                      # launches on the current stream, returns out
seq_lens.copy_(torch.randint(1, max_pages * 16 + 1, (batch,), dtype=torch.int32, device="cuda"))
out = decode()                      # same runner (or a captured graph), new lengths
```

Preparation validates the inputs, zeroes the kernel's counters once inside
`workspace_buffer` and binds the tensors to the generated program; it makes no
host copy of `seq_lens`. `balanced_gqa_decode_workspace_size(device)` bounds
the workspace for any batch on that device (about 10.7 MB on a 160-SM GPU);
the counters are reset by the kernel itself at the end of every launch, so the
workspace region must not be reused by other work between launches of the
same runner. Block tables whose width is not a multiple of eight pages are
copied into a padded table inside the workspace at preparation (the runner
reports this as `block_tables_padded=True`); pass a width that is a multiple
of eight to read the caller's table in place at every launch.

`plan.py` is an exact host mirror of the device planner. The kernel never
calls it; tests use it to check the device-published plan (`runner.device_plan()`)
and the workspace bounds, and it documents the scheduling rule.

Limits of the current route: BF16 Q/K/V/O only, head dimension 128, page
size 16, exactly eight query heads per KV head, at most 1024 requests per
launch, SM100 (B200) and SM103 (B300/GB300) only (SM120/SM121 lack the
tcgen05/TMEM path). No attention sinks, sliding window or LSE output.

See `tests/experimental/test_cake_balanced_gqa_decode.py` for the torch
reference and validated shape set (uniform, ragged, the AgentX-derived
pattern of #4832, MTP `q_len_per_req=7`, CUDA Graph replay) and
`benchmarks/bench_cake_balanced_gqa_decode.py` for the CUPTI benchmark against
`trtllm_batch_decode_with_kv_cache(backend="trtllm-gen")` on the same tensors.
