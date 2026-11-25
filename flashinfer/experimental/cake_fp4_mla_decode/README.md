# Experimental MLA decode over the TensorRT-LLM `fp4_mla` KV cache

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling `flashinfer.mla.prepare_fp4_mla_decode`
explicitly opts into the experimental feature and emits FlashInfer's
experimental API warning. There is no automatic backend selection.

The decode reads TensorRT-LLM's NVFP4 MLA cache in place: 128-token pages of
320 bytes per token (E2M1 codes of the 512-channel latent, the 64-channel RoPE
base and the 64-channel RoPE residual) with 40 E4M3 block scales per token in
the 128x4 swizzle, and the step's prequantized query in the same format. QK
and PV run as block-scaled FP4 MMAs; the attention probabilities are quantized
to NVFP4 per 16 keys. The output is BF16 `[B * q_len * H, 512]`.

```python
from flashinfer.mla import prepare_fp4_mla_decode

decode = prepare_fp4_mla_decode(
    batch=B, q_len=q_len, num_heads=H, page_table_width=W, device="cuda",
    backend="cake",
)
decode(
    kv_cache, kv_scale_cache, v_scale_cache, q_fp4, q_scale, page_table,
    kv_lens, q_global_scale, kv_global_scale, out, sm_scale=sm_scale,
)
```

`prepare_fp4_mla_decode` fixes the bucket (batch, query tokens per request, heads
up to 128, page-table width) on one device, chooses the KV split count from the
bucket and the device's SM count, and allocates the split partials once. The
optional `max_kv` bounds every `kv_lens` value of every call (for example the
deployment's maximum sequence length). When neither it nor the page-table width
allows 32,768 keys, the runner uses the decode build without the long-row
branch of the P quantization, which gives the same result on shorter rows. A
call of the runner allocates no CUDA memory and reads no device value on the
host, so one runner serves every CUDA Graph replay of its bucket while
`kv_lens` changes. CUDA Graph capture belongs to the caller.

The runner takes TensorRT-LLM's per-layer views as they are: `kv_cache`
(`[pages, ...]`, 40,960 bytes per page), `kv_scale_cache` (`[pages, ...]`,
5,120 bytes per page), `v_scale_cache` (the MLA V-scale pool, `[pages, ...]`,
4,096 bytes per page: per latent dim, the K scale of the first token of each
16-token block), the prequantized query `q_fp4` (`[rows, 320]`, rows
`token * H + head`) and its swizzled scales `q_scale`, the int32 page table
`[B, W]`, the int32 KV lengths (each including the step's query tokens; query
token `t` of `q_len` sees `kv_len - (q_len - 1 - t)` keys) and the FP32
global scales on the device. The scale pools have no padding between pages
(page strides of 5,120 and 4,096 bytes) and use the page ids of `kv_cache`;
the runner refuses a pool in any other layout with a `ValueError`. The decode
reads the V scales from the V-scale pool on every call (it does not derive
them from `kv_scale_cache`), so the pool must hold the V scales TensorRT-LLM
writes with the cache. The split-KV reducer
launches with programmatic dependent launch: its blocks start while the
decode finishes and wait for the decode's results.

Supported GPUs: SM100 (`sm_100a`) and SM103 (`sm_103a`).

## Package layout

The backend is a self-contained package. `cake_jit.py` is the registry of the
generated programs: one `MODULES` record per program, the two decode builds
(with and without long rows) and the three builds of the split-KV reducer of
their partials (one, two or four warps per output row), carrying the host plan
(the architectures it
is delivered for, its stage, its launch record, the argument plan of its `run`
entry, the grid rule a launch follows, its workspace and the kernel-parameter
to logical-argument map) and the loader glue (`record_for`, `launch_grid`,
`launch_args`, `load`). What to compile is recorded once in `build.json`: the
generated CUDA sources under `csrc/`, their digests and nvcc flags, one entry
per (program, architecture). `cake_build.py` loads a prebuilt library from
`aot/` when a builder compiled one and otherwise JIT-builds the manifest entry
through FlashInfer from the same sources and flags.
`scripts/check_cake_build_manifests.py` checks the manifest against the
registry and the sources without importing the package.
`tests/experimental/test_cake_fp4_mla_decode.py` checks the decode against a
float64 reference of the same cache bytes.
