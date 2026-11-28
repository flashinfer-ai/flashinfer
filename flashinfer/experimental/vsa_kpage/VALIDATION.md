# KPage integration validation

## Integrated functionality

`BlockSparseAttentionWrapper.plan(kpage=True)` selects the extracted cuDNN
CuTe DSL KPage implementation on `vsa_sm100_blk64` and `vsa_sm100_blk128`.
`plan()` builds the layout and `run()` reuses it. The default remains False.

| Functionality | blk64 | blk128 |
| --- | --- | --- |
| BF16, head dimension 128, SM100/SM103 | Yes | Yes |
| Forward | Yes | Yes |
| MHA | Yes | Yes |
| GQA with patterns per query head | No | Yes |
| Independent Q/KV sequence lengths | Aligned to 64 | Including tails |
| Per-KV-block valid prefix lengths | Yes | Yes |
| Per-query-block valid prefix lengths | No | Yes |
| Native indices, BSR, boolean masks | Yes | Yes |
| FI destination buffers and LSE | Yes | Yes |
| Fixed-plan CUDA graph replay after warmup | Tested | Tested |

This integration exposes forward only. Cake and automatic backend selection
do not select KPage. Metadata edits, including in-place changes, require replan.
The README records the complete support restrictions and source provenance.

## Test environment

- GPU: NVIDIA B200, SM100.
- PyTorch: `2.11.0a0+a6c236b9fd.nv26.03.46836102`.
- CUDA runtime: 13.2.
- CuTe DSL: 4.8.0.
- Ordinary FI blk128 regression dependency: `quack-kernels==0.6.4`.
- Extra validation dependencies were isolated from the system installation.

## KPage test results

File: `tests/experimental/test_vsa_kpage.py`.

| Test category | Cases | Result |
| --- | ---: | --- |
| Independent reference, layout reuse, destination buffers, scales | 12 | Passed |
| BSR/mask conversion and switching back to ordinary VSA | 4 | Passed |
| Invalid metadata, failed replan, invalid output alias | 2 | Passed |
| Unsupported backend combinations | 3 | Passed |
| Incompatible scheduler/masking options | 1 | Passed |
| Native FI shapes and graph replay with changed Q/K/V | 2 | Passed |
| Active blk128 query tails, including M64/M128 routing boundaries | 4 | Passed |
| GQA with Hq=4 and Hkv=2, different patterns and values | 1 | Passed |
| Exact empty-attention and query-padding semantics | 4 | Passed |
| Q-head masks, KV-head masks and BSR vs independent reference | 6 | Passed |
| Default counts, lengths and softmax scale | 6 | Passed |
| Replanning after changes to connections, lengths and shapes | 2 | Passed |
| **Total** | **47** | **Passed** |

The independent reference uses FP64 arithmetic. Output comparisons use
`atol=rtol=2e-2`; finite LSE comparisons use `atol=rtol=1e-3`. These are
acceptance tolerances, not measured maximum errors. Padding output/LSE are
required to be exactly zero. Empty attention rows with valid query tokens
require exactly zero output and LSE=-inf. NaN-filled destination buffers check
that these rows are actually written.

Additional regression targets are `tests/attention/test_cake_vsa_planning.py`
and `test_vsa_sm_scale`, `test_vsa_per_head_mask_correctness`, and
`test_vsa_blk64_accuracy` in `tests/attention/test_vsa_block_sparse.py`.
See `VALIDATION_RESULTS.json` for the combined run's exact counts and status.

Reproduce in a supported FI development environment:

```bash
pytest -q tests/experimental/test_vsa_kpage.py \
  tests/attention/test_cake_vsa_planning.py \
  tests/attention/test_vsa_block_sparse.py::test_vsa_sm_scale \
  tests/attention/test_vsa_block_sparse.py::test_vsa_per_head_mask_correctness \
  tests/attention/test_vsa_block_sparse.py::test_vsa_blk64_accuracy
```

## Source equivalence and independence checks

Manual comparisons against cuDNN Frontend commit
`862a438c851b77305391916f7db61b14d8fdeb4e` produced bit-identical output and
LSE for blk64 MHA, blk128 MHA, blk128 GQA and blk128 GQA with truncated sequence
tensors. These comparisons are separate from the FP64 active-query-tail tests.
They are migration checks, not tests that require cuDNN in normal FI CI.

Both block sizes also ran with imports of `cudnn` and `cudnn.*` explicitly
blocked. The migrated runtime does not depend on the cuDNN Python package.

## Performance regression data

Warm, reused-layout forward; M=N=8192, Hq=Hkv=8, D=128, 16 selected KV blocks
per query block, 40 valid tokens per KV block, all query rows valid:

| Block size | cuDNN source forward | FI forward | FI warm plan |
| --- | ---: | ---: | ---: |
| 64 | 0.088192 ms | 0.088096 ms | 0.659 ms |
| 128 | 0.065472 ms | 0.065408 ms | 0.666 ms |

Forward times include output allocation and exclude layout construction and
first JIT. `bench_gpu_time(enable_cupti=True)` used CUDA-event fallback because
CUPTI was unavailable. Plan times are synchronized wall-clock medians of five
warm calls. GPU clocks were unlocked. These small differences indicate similar
performance in the measured configurations, not a significant speedup.

The standalone `benchmarks/bench_vsa_kpage.py` separately measures warm plan and
reused execution with preallocated output buffers, so its timing boundary
differs from the source API comparison above.

## Remaining validation limits

- SM103 has not been exercised during this migration.
- A broad performance comparison against Cake and ordinary CuTe VSA is pending.
- Results do not establish benefits for dynamic layouts rebuilt every call.
- Graph results cover a warmed, fixed plan; replanning does not update an
  existing captured graph.
- Full-suite CI, model end-to-end performance and multi-GPU concurrency were
  not validated.
