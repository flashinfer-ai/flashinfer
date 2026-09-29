# Experimental NVFP4 DeepSeek-V4 decode attention

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#5403.

`flashinfer.mla.prepare_nvfp4_batch_decode_with_kv_cache_mla(...)` serves the
DeepSeek-V4 main-attention decode geometry: paged MQA over one 512-wide NVFP4
latent row per token that is both K and V, 64-token pages, `q_len` query
tokens per request (derived from the query rows; six for DeepSeek-V4) with a
causal mask inside the block against the request's last tokens, an optional per-head attention sink, BF16 output and natural-log FP32
LSE. Query and KV are E2M1 values packed two per byte (256 bytes per row)
with UE4M3 block-16 scales (32 bytes per row); `quantize_nvfp4` in
`cake_backend.py` produces both from a float tensor.

| Tensor | Shape | dtype |
| --- | --- | --- |
| `query` | `[batch * q_len, num_heads, 256]` | uint8 (packed E2M1) |
| `query_scale` | `[batch * q_len, num_heads, 32]` | uint8 (UE4M3) |
| `kv_cache` | `[num_pages, 64, 256]` | uint8 (packed E2M1) |
| `kv_scale` | `[num_pages, 64, 32]` | uint8 (UE4M3) |
| `block_tables` | `[batch, max_pages]` | int32 |
| `seq_lens` | `[batch]` | int32 |
| `sinks` (optional) | `[num_heads]` | float32 |
| `out` | `[batch * q_len, num_heads, 512]` | bfloat16 |
| `lse` | `[batch * q_len, num_heads]` | float32 (natural log) |

Preparation validates the inputs, builds the host work plan from the
sequence lengths (one device-to-host copy unless `seq_lens_cpu` is passed),
carves the BF16 split partials and plan tables out of `workspace_buffer`, and
returns an `NVFP4MLADecodeRunner`. The decode kernel launches one cluster of
two CTAs per work unit; the pair forms one tensor-core MMA over the 128 query
rows of a tile (64 rows x 512 output dims per CTA) and shares a multicast
page stream. Calling the runner launches the persistent decode kernel and,
when any request is split across CTAs, the split-KV combine kernel (rows whose
request was not split are written by the decode kernel and left alone by the
combine), with no CUDA allocation or host synchronization:

```python
import torch
from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla
from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
    nvfp4_mla_decode_workspace_size,
    quantize_nvfp4,
)

batch, heads, kv_len = 32, 64, 8192
pages = batch * kv_len // 64
q, q_scale = quantize_nvfp4(torch.randn(batch * 6, heads, 512, device="cuda"))
kv, kv_scale = quantize_nvfp4(torch.randn(pages, 64, 512, device="cuda"))
block_tables = torch.arange(pages, dtype=torch.int32, device="cuda").view(batch, -1)
seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device="cuda")
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
workspace = torch.empty(
    nvfp4_mla_decode_workspace_size([kv_len] * batch, heads, num_sms=num_sms),
    dtype=torch.uint8,
    device="cuda",
)

decode = prepare_nvfp4_batch_decode_with_kv_cache_mla(
    q, q_scale, kv, kv_scale, block_tables, seq_lens, workspace,
    sm_scale=512**-0.5, return_lse=True,
)
out, lse = decode()  # Execute decode and write out / lse.
```

Prepare a new runner when sequence lengths, bindings or input values change.
CUDA Graph capture belongs to the caller. `max_nvfp4_mla_decode_workspace_size`
bounds the workspace for any plan; the runner's `main_kwargs` /
`reduce_kwargs` expose the bound tensors and host launch metadata and `out` /
`lse` are the exact tensors passed by the caller.

Limits of the current route: head dimension 512 with shared K/V rows (MQA),
page size 64, the same number of query tokens per request across the batch,
causal masking inside the
query block, optional sinks, at least 128 packed query rows per batch
(`batch * q_len * num_heads >= 128`, the kernel's query tile), SM100 (B200)
and SM103 (B300) only (SM120/SM121 lack the block-scaled tensor-core path).
At most 64 KV splits per request.

See `tests/experimental/test_cake_nvfp4_mla_decode.py` for the torch
reference and validated shape set and `benchmarks/bench_cake_nvfp4_mla_decode.py`
for the timing harness.
