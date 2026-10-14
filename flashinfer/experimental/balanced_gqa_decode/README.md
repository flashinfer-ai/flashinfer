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
- Speculative / MTP verification (`q_len_per_req` 3 to 8) runs a packed-row
  program: the eight query heads of every draft token of a request are packed
  into one `N = 8 * q_len_per_req` MMA tile (32 or 64 rows), so each KV block is
  streamed once per request instead of once per draft row. The scheduler picks
  the chunk length with the smallest modelled launch time among the balanced
  lengths, whole tiles, a few coarser lengths and the single-wave fit; tiles of
  more than two chunks are reduced by one to eight distributed reduce tickets
  (chosen from the longest split tile), two-chunk tiles are folded in place.
  `q_len_per_req` 1 uses the single-row program; `q_len_per_req` 2 and above 8
  are rejected at preparation. The program is selected at preparation from
  `q_len_per_req`, so a captured runner replays for any KV lengths but keeps
  its `q_len_per_req`.

## Kernels

The kernels are the `csrc/cake_fmha` balanced components of the Cake FMHA
product (registry `csrc/cake_fmha/registry.json`, route
`decode_balanced_bf16_v1`), the same programs the production
`trtllm_batch_decode_with_kv_cache(..., backend="cake")` route runs for long
ragged BF16 GQA-8 decode:

| `q_len_per_req` | Component | Threads | Dynamic SMEM |
| --- | --- | --- | --- |
| 1 | `decode_balanced_bf16` (row-tile kernel) | 512 | 215040 B |
| 3, 4 | `decode_balanced_bf16_mtp_n32` (packed 32-row tile) | 384 | 222208 B |
| 5 to 8 | `decode_balanced_bf16_mtp_n64` (packed 64-row tile) | 384 | 222208 B |

They are built per `q_len_per_req` by
`flashinfer.jit.cake_fmha.load_cake_fmha_decode_balanced_bf16_module(target, q_len)`
with the shared adapter `csrc/cake_fmha/jit/cake_fmha_decode_balanced_jit_binding.cu`,
whose `cake_paged_attention_decode` entry the runner calls. This package holds
no kernel sources and no JIT specifications of its own.

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
k_cache = torch.randn(
    batch * max_pages, num_kv_heads, 16, 128, dtype=torch.bfloat16, device="cuda"
)
v_cache = torch.randn_like(k_cache)
block_tables = torch.arange(batch * max_pages, dtype=torch.int32, device="cuda").view(
    batch, max_pages
)
seq_lens = torch.randint(
    1, max_pages * 16 + 1, (batch,), dtype=torch.int32, device="cuda"
)
workspace = torch.empty(
    balanced_gqa_decode_workspace_size(q.device), dtype=torch.uint8, device="cuda"
)

decode = prepare_balanced_batch_decode_with_kv_cache(
    q, (k_cache, v_cache), block_tables, seq_lens, workspace
)
out = decode()  # launches on the current stream, returns out
seq_lens.copy_(
    torch.randint(1, max_pages * 16 + 1, (batch,), dtype=torch.int32, device="cuda")
)
out = decode()  # same runner (or a captured graph), new lengths
```

Preparation validates the inputs, loads the component for `q_len_per_req`,
carves `workspace_buffer` into the adapter's two regions and zeroes the
counters once; it makes no host copy of `seq_lens` and no copy of the block
table (the kernels clamp page indices to each request's last page, so any
table width is read in place at every launch). A launch is the adapter's host
work only: argument checks, the three TMA descriptor encodes (`Q`, `K`, `V`,
passed to the kernel by value) and the kernel launch; no allocation and no
host synchronization, so a runner can be captured into a CUDA Graph.

`balanced_gqa_decode_workspace_size(device)` bounds the workspace for any
batch and `q_len_per_req` on that device (about 85.2 MB on a 160-SM GPU):
the FP32 partial slots and statistics sized by
`flashinfer.cake_fmha.cake_fmha_balanced_workspace_bytes` for the packed-row
programs' 64-row tiles, then the self-resetting counters sized by
`cake_fmha_balanced_counter_bytes`. The counters are reset by the kernel
itself at the end of every launch, so the workspace region must not be reused
by other work between launches of the same runner.

`runner.device_plan()` returns the `(chunk pairs, total items)` the device
planner published for the last launch (CTA 0 writes them into the reserved
statistics slot; reading them synchronizes). `cake_bounds.py` holds the
planner constants and the shape-independent workspace and ticket-loop bounds
the adapter applies; nothing on the host reads KV lengths. The exact host
mirror of both device planners is test-only
(`tests/test_helpers/cake_balanced_gqa_plan.py`): the tests use it to check
the published plan and it documents the scheduling rules.

Limits of the current route: BF16 Q/K/V/O only, head dimension 128, page
size 16, exactly eight query heads per KV head, `q_len_per_req` 1 or 3 to 8,
at most 1024 requests per launch, SM100 (B200) and SM103 (B300/GB300) only
(SM120/SM121 lack the tcgen05/TMEM path). No attention sinks, sliding window
or LSE output.

See `tests/experimental/test_cake_balanced_gqa_decode.py` for the torch
reference and validated shape set (uniform, ragged, the AgentX-derived
pattern of #4832, packed-row MTP `q_len_per_req` 3/4/7/8, CUDA Graph replay)
and `benchmarks/bench_cake_balanced_gqa_decode.py` for the CUPTI benchmark
against `trtllm_batch_decode_with_kv_cache(backend="trtllm-gen")` on the same
tensors.
