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
`kv_lens` changes. CUDA Graph capture belongs to the caller: run one eager
call of the runner before the capture (the first call loads the libraries,
JIT-building them when no prebuilt library exists), and keep the runner (it
owns the split partials) and every tensor a captured call reads or writes
alive until the last replay. The split count is fixed when the runner is
prepared (`runner.plan.num_split`); `kv_lens` never changes it. A runner
with more than one split writes and reads the same split partials on every
call, and on every replay of a graph that captured it, so its calls must
run in order: on one stream, or with each stream waiting for the other's
call (for example through a CUDA event). A runner with one split writes
`out` directly and does not touch the partials. Concurrent decode on
several streams is always safe with one runner per stream.

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
writes with the cache. `out` must be 32-byte aligned and `q_fp4` and the
scale pools 16-byte aligned, `sm_scale` finite and positive, and `kv_cache`
at most 1,677,721 pages; the runner refuses anything else with a
`ValueError` before a launch. A request whose KV length is 0 gets zero
output rows, but the decode still reads column 0 of its page-table row and
the scales of that page, so that entry must hold a page id of `kv_cache`
or -1 (the decode reads page 0 for -1). The split-KV reducer
launches with programmatic dependent launch: its blocks start while the
decode finishes and wait for the decode's results.

Supported GPUs: SM100 (`sm_100a`) and SM103 (`sm_103a`).

## Package layout

The backend is a self-contained package. `cake_jit.py` is the registry of the
generated programs: one `MODULES` record per program, the decode builds and
the three builds of the split-KV reducer of their partials (one, two or four
warps per output row). The decode builds are with and without long rows, each
per item (a cluster per work item) and persistent (at most a cluster per SM
pair, looping over the items), plus two forms of the persistent build without
long rows: a late form on SM103 and a form that runs each item's epilogue in
the V transpose warps on SM100, and a form of the per-item build without long
rows that stores the output through TMA on both. Every build gives the same
result; the runner
picks one from the bucket and the device. A record carries the host plan
(the architectures it is delivered for, its stage, its launch record, the argument plan of its `run`
entry, the grid rule a launch follows, its workspace and the kernel-parameter
to logical-argument map) and the loader glue (`record_for`, `launch_grid`,
`launch_args`, `load`). What to compile is recorded once in `build.json`: the
generated CUDA sources under `csrc/`, their digests and nvcc flags, one entry
per (program, architecture). The decode entries carry an `nvrtc` record:
`cake_nvrtc.py` compiles their device source with NVRTC to a cubin, nvcc
compiles the binding, and the library embeds the cubin (NVRTC and CUDA 12.9
or newer). The reducer entries, and the prebuilt library of any entry, load
through `cake_build.py`, which loads a
prebuilt library from `aot/` when a builder compiled one and otherwise
JIT-builds the manifest entry through FlashInfer from the same sources and
flags.
`scripts/check_cake_build_manifests.py` checks the manifest against the
registry and the sources without importing the package.
`tests/experimental/test_cake_fp4_mla_decode.py` checks the decode against a
float64 reference of the same cache bytes.
