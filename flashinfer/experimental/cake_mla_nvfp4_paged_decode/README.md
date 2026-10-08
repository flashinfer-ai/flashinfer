# Experimental Cake dense NVFP4 MLA decode (SM100 / SM103)

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection. Tracking: flashinfer-ai/flashinfer#4644 (Cake tracker #4254).

`flashinfer.mla.cake_mla_nvfp4_paged_decode(...)` serves absorbed multi-head
latent attention decode (Kimi-K3 / DeepSeek-V3 geometry: 512 latent + 64 rope
channels per key, 512-wide value) over the NVFP4 paged latent cache written by
flashinfer-ai/flashinfer#4676: dense decode (`q_len = 1`), packed variable-length
/ MTP queries (`cum_seq_lens_q` / `max_q_len`, bottom-right causal), 6..128
heads, page sizes that are powers of two from 32, optional natural-log LSE and
decode context parallelism (`cp_world` / `cp_rank` / `kv_len_global`). The query
is NVFP4 too; `quantize_mla_nvfp4_query` produces it from a BF16 `[.., 576]`
query (bit-identical to the Cake reference quantizer).

| Tensor | Shape | dtype |
| --- | --- | --- |
| `q_nope` | `[batch, q_len, num_heads, 256]` or `[total_q, num_heads, 256]` | uint8 (packed E2M1) |
| `q_sf` | `[.., 32]` | float8_e4m3fn (block-16 scales) |
| `q_rope` | `[.., 64]` | float8_e4m3fn |
| `q_scale` | `[..]` | float32 (per-row scale) |
| `ckv_cache` | `[num_pages, page_size, 256]` | uint8 (packed E2M1) |
| `ckv_sf_cache` | `[num_pages, page_size, 32]` | float8_e4m3fn (linear block-16 scales) |
| `kpe_cache` | `[num_pages, page_size, 64]` | float8_e4m3fn (rope at `kpe_scale`) |
| `block_tables` | `[batch, max_pages_per_seq]` | int32 |
| `seq_lens` | `[batch]` | int32 (local KV lengths) |
| `out` | `q.shape[:-1] + (512,)` | bfloat16 |
| `lse` (optional) | `q.shape[:-1]` | float32 (natural log) |

A decoded key is `e2m1 * sf * ckv_scale | fp8 * kpe_scale`; the logits are
`sm_scale * ckv_scale * q_scale * (QN . KN + QR . KR)`. The three cache tensors
may be strided views of one allocation (for example the 352-byte-per-token
`[.., 256 | 64 | 32]` buffer); token / page strides and the base must be 16-byte
multiples. `workspace_buffer` is a caller-owned uint8 buffer of at least
`cake_backend.workspace_bytes(rows, num_split)` bytes
(`max_workspace_bytes(rows)` covers every plan of a row count).

**Toolkit requirement: CUDA 13.4 or newer.** The attention programs convert V
with the Blackwell `QMUL4` instruction spelled in PTX ISA 9.4
(`mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite`); older ptxas reject it, and the
loader refuses the attention programs below the registered `min_cuda_version`
rather than falling back to a slower emulation.

The attention program is a swapped-AB tcgen05 schedule: one CTA per (request,
row tile of 16 / 32 / 48 packed (token, head) rows, KV split) computes
S^T[128 tokens, rows] with two `kind::mxf4nvf4` block-scaled MMAs (cache block
scales as the A scale factors, query block scales as the B scale factors) plus
one `kind::f8f6f4` rope MMA, converts V on chip to E4M3 with a per-token
power-of-two shift compensated in P, and accumulates O^T on `kind::f8f6f4`. The
host plan (row tile, split count, grids, reducer) depends only on the batch
shape, the longest KV and the device's SM count; a split-KV merge kernel runs
behind the attention kernel (programmatic dependent launch) when the plan has
more than one split. `prepare_cake_mla_nvfp4_paged_decode(...)` returns a
launch-only runner (`launch` allocates nothing; CUDA Graph ownership stays
with the caller).

```python
import torch
from flashinfer.mla import (
    cake_mla_nvfp4_paged_decode,
    quantize_mla_nvfp4_query,
)
from flashinfer.experimental.cake_mla_nvfp4_paged_decode.cake_backend import (
    max_workspace_bytes,
)

batch, heads = 8, 12
q = torch.randn(batch, 1, heads, 576, device="cuda", dtype=torch.bfloat16)
q_nope, q_sf, q_rope, q_scale = quantize_mla_nvfp4_query(q, ckv_scale, kpe_scale)
workspace = torch.empty(max_workspace_bytes(batch * heads), dtype=torch.uint8, device="cuda")
out = cake_mla_nvfp4_paged_decode(
    q_nope, q_sf, q_rope, q_scale,
    ckv_cache, ckv_sf_cache, kpe_cache, block_tables, seq_lens, workspace,
    sm_scale=1.0 / 192 ** 0.5, ckv_scale=ckv_scale,
)
```

The generated sources under `csrc/cake_mla_nvfp4_paged_decode/` and the
registry in `cake_jit.py` are written by the Cake generated-program export
(`exports/mla_nvfp4_paged_decode/` in the Cake repository); regenerate them,
do not edit them by hand. Tests: `tests/experimental/test_cake_mla_nvfp4_paged_decode.py`.
