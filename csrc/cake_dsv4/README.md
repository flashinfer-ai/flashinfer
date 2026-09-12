# Cake DeepSeek V4 sparse MLA

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

Public routing and descriptor-storage tests:

```bash
pytest tests/mla/test_cake_dsv4.py -q
```
