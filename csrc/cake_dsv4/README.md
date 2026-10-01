# Cake DeepSeek V4 sparse MLA

Current support covers SM100 and SM103. The complete current 94-case tables, comparisons, runtime and hardware evidence are in the [current SM100](https://github.com/flashinfer-ai/flashinfer/pull/4573#current-sm100-validation) and [current SM103](https://github.com/flashinfer-ai/flashinfer/pull/4573#current-sm103-validation) PR sections.

## Historical SM103 implementation and validation

The original implementation details and acceptance values below remain historical evidence; the [unchanged historical 94-case table](https://github.com/flashinfer-ai/flashinfer/pull/4573#historical-sm103-94-case-performance) remains in the PR body.

Select the Cake backend through the existing public API:

```python
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

output = trtllm_batch_decode_sparse_mla_dsv4(
    query=query,
    swa_kv_cache=swa_kv_cache,
    workspace_buffer=workspace_buffer,
    sparse_indices=sparse_indices,
    compressed_kv_cache=compressed_kv_cache,
    sparse_topk_lens=sparse_topk_lens,
    seq_lens=seq_lens,
    backend="cake",
)
```

This backend targets SM103 GPUs and uses head dimension 512. The validated
94-case matrix covers BF16 and FP8 E4M3 inputs, NHD and HND cache layouts,
fixed and ragged queries, optional attention sinks, decode, and prefill.
Outputs are BF16. Cake is selected explicitly; `backend="auto"` retains the
existing architecture-based selection. Programmatic dependent launch and the
TRTLLM-GEN RopeQuant epilogue are unsupported by this backend.

The directory contains 24 generated device kernels and 24 corresponding
TVM-FFI bindings, plus four compiled dispatch programs that preserve the
original multi-stage call boundaries. `cake_dsv4_modules.json` records each variant's source files,
compilation flags, target architecture, and argument contract. FlashInfer's
JIT compiles the device and binding as separate translation units on first
use. Program host libraries link those kernels with their individual
optimization flags retained. Rebuilding requires a CUDA toolkit that supports `sm_103a`.

Python prepares descriptor storage before calling the generated binding.
Descriptor addresses remain immutable for the process lifetime, including
across caller-workspace replacement, while query and KV-cache tensors retain
their normal lifetimes. Generated bindings perform no device allocation.

Validation on GB300 covers all 94 reference cases, using
`atol=rtol=0.01` for BF16 and `atol=rtol=0.1` for FP8. Rebuilt correctness
results: 94/94 correct with zero fallback. All 24 kernel libraries and four program host libraries built
successfully.

This delivery uses a **2% per-row latency tolerance** for all three
comparisons below. Each rebuilt result must satisfy
`export latency <= 1.02 × baseline latency` across all 94 cases; aggregate
speedup alone does not establish acceptance. Correctness, zero fallback,
and source/export activity parity remain required. Original measurements
and strict results are preserved. Qualification under this delivery-specific
policy: **94/94 passed in all three comparisons**. Source/export activity parity passed all 94 cases, with zero fallback. The preserved raw strict verdict passes **54/94** cases jointly (73/94 for source/export, 70/94 for previous/new export, and 94/94 for TRTLLM-GEN/Cake).

| Comparison, all 94 cases | Baseline sum (ms) | Export sum (ms) | Aggregate speedup | Minimum row speedup | Maximum row latency change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original implementation / rebuilt export | 1.5146230 | 1.5112325 | 1.002243533× | 0.982532751× | +1.777778% |
| Previous Cake export / rebuilt export | 1.5151350 | 1.5146385 | 1.000327801× | 0.988235294× | +1.190476% |
| TRTLLM-GEN / Cake, retained public API | 1.8525280 | 1.5187200 | 1.219795617× | 1.006036217× | -0.600000% |

Performance uses CUPTI GPU kernel active-union time with cold L2 and includes
each comparison's stated API boundary. Launch gaps are excluded and
overlapping kernel intervals are counted once. Each comparison uses three
independent groups with equal ABBA and BAAB ordering within every group;
results use pooled active-union medians. The
[94-case comparison table in PR #4573](https://github.com/flashinfer-ai/flashinfer/pull/4573)
contains all 14 recorded columns for every canonical case, including shape
fields and the baseline latency, candidate latency, and speedup for each of
the three comparisons.

Selected sealed-row benchmark runtime sums to **19,937.948 s** across the 94 disjoint rows; this is harness runtime, not GPU active time. The corrected measurement campaign's physical turnaround was **4,221.966 s** (2026-09-12T13:51:04.703Z to 2026-09-12T15:01:26.669Z), including unsuccessful measurement attempts and retries. This elapsed interval covers the corrected campaign, before final qualification and publication; concurrent step durations are not added to obtain elapsed time.

Separate sanitizer results:
synccheck passed all 94 cases with zero errors; racecheck passed all 94 cases
with zero hazards, errors, or warnings. The public routing and descriptor
suite passed 101 tests.

Public routing, binding, workspace and metadata tests (CPU) plus the GPU
hardening suite:

```bash
pytest tests/mla/test_cake_dsv4.py -q
pytest tests/mla/test_cake_dsv4_hardening.py -q   # needs an SM100/SM103 GPU
```

## Hardened host contract (flashinfer#4671)

The Python host in `flashinfer/mla/cake_dsv4.py` follows the shared
sparse-metadata ABI of the regenerated kernels and makes no device allocation
on the call path.

### Metadata

Every variant receives the same eight kernel parameters, bound by name through
the registration `arg_plan` (`flashinfer/jit/cake_dsv4.py`): `swa_indices`,
`compressed_indices`, `sparse_topk_lens`, `swa_index_stride`,
`compressed_index_stride`, `sparse_topk_lens_offset`, `sparse_topk`,
`num_query_tokens`. Combined column `c` of row `t` is
`swa_indices[t * swa_index_stride + c]` for `c < 128` and
`compressed_indices[t * compressed_index_stride + c - 128]` otherwise; the
active length is `clamp(sparse_topk_lens[t] + sparse_topk_lens_offset, 0,
sparse_topk)`. The kernels never read a metadata slot outside
`t < num_query_tokens`, `c < sparse_topk`, and every unread staged slot is `-1`.

`trtllm_batch_decode_sparse_mla_dsv4(..., backend="cake")` accepts either
form without copying (int32 tables with unit column stride; row strides are
free):

| Form | Arguments | Host resolution |
| --- | --- | --- |
| Combined | `sparse_indices [T, sparse_topk]`, `sparse_topk_lens [T]` (counts the 128 SWA slots) | compressed view = column offset 128 of the same storage, both strides `= sparse_topk` |
| Separate | `sparse_indices [T, 128]` (SWA), `extra_sparse_indices [T, topk_c]`, `extra_sparse_topk_lens [T]` (compressed slots only) | offset `+= 128`; each table keeps its own row stride |
| Separate, combined lengths | as above with `sparse_topk_lens` instead of `extra_sparse_topk_lens` | offset unchanged |

`sparse_topk_lens_offset: int = 0` is added on top in every form.
`num_query_tokens = T` is the metadata row count; `query` / `out` may carry
more rows (dense `[B, Q, H, 512]` with `T <= B * Q`, ragged `[sum_q, H, 512]`
with `T <= sum_q`). Rows `>= T` are neither read nor written. Every grid and
workspace view derives from `T`. Batch-derived routes (`*_source_exact`,
`bf16_h64_guard_q_tma_batch_r25`, `fp8_h128_prefill_source_persistent`,
`fp8_h128_prefill_source_persistent_uniform`, `fp8_h64_prefill_source_persistent_m64`)
walk `cum_seq_lens_q`, so the metadata must cover every token those prefix
sums address.

### Workspace

One caller-owned `workspace_buffer` (contiguous, 128-byte aligned) is carved
deterministically:

```
[0,      1024)             TMA descriptor slab — the generated bindings encode fresh
                           descriptors and upload them here on every launch
[1024,   1024 + 256 KiB)   split-merge counters, uint32[65536]
[P,      P + O_bytes)      partial_O  BF16 [T * H * S * 512]      P = 1024 + 256 KiB
[P + O_bytes, ...)         partial_lse FP32 [T * H * S]
```

```python
from flashinfer.mla import get_cake_dsv4_workspace_bytes, cake_dsv4_workspace_reset

num_bytes = get_cake_dsv4_workspace_bytes(num_query_tokens, num_heads, sparse_topk, dtype)
# S = num_splits if given else max(ceil(sparse_topk / 128), 5)   (upper bound over routes)
# bytes = 1024 + 262144 + align128(T * H * S * 512 * 2) + align128(T * H * S * 4)
workspace = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
cake_dsv4_workspace_reset(workspace)   # zero the counter region once
```

Counters must be zero at first use and the kernels leave them zero after every
launch, so there is no per-call host state and CUDA graph replays are
self-contained. The first eager call with a workspace tensor also zeroes the
counters; if that first use happens during graph capture the host raises and
asks for `cake_dsv4_workspace_reset` or an eager warm-up instead. The only
allocation on the call path is the output when `out=None`; with `out` provided,
eager calls and graph replays leave `torch.cuda.memory_allocated()` unchanged.
The per-call padded index copy and the host-side split-merge generation counter
(`completion_base`) of the previous host are gone.

### Bindings

`_launch_variant` builds a name -> value map and walks the variant's `arg_plan`
(`tma_buffer` / `buffer` -> tensors, `parameter` -> ints, `workspace` -> the
descriptor slab, `grid` -> launch grid). TMA sources alias onto `Q`,
`SWA_cache`, `compressed_KV_cache`; `num_q_heads` / `num_split` alias onto
`num_heads` / `num_splits`. Compiled programs bind their `tensor_keys`,
`workspace_keys`, `scalar_names` the same way. Bindings that still declare the
pre-hardening combined `sparse_indices` accept only a combined table with
offset 0; `completion_base` is rejected. The CPU tests check that every
registered name is bindable, so a regenerated registration with a new name
fails fast on the host side.
