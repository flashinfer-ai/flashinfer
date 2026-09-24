# B200 validation results

Environment: NVIDIA B200 (SM100), CUDA toolkit 13.3, PyTorch 2.11.0+cu130,
FlashInfer 0.7.0 at upstream commit
`975f90583d9ac8896db14cf0f26e99a853c2f136`.

Correctness:

```text
pytest -q tests/experimental/test_adaptive_sparse_block_mask.py -x -vv
13 passed
```

CUPTI cold-L2 medians, 10 warm-up and 30 measured iterations per
implementation. Measurements are split into two equal batches with reversed
execution order (`adaptive -> PyTorch`, then `PyTorch -> adaptive`) to reduce
clock and thermal-state bias:

| Shape | Adaptive CUDA | PyTorch topk pipeline | Speedup |
| --- | ---: | ---: | ---: |
| B1 H8 Qb64 Kb192, k=49 | 0.020928 ms | 0.032575 ms | 1.557x |
| B1 H8 Qb64 Kb1024, k=132 | 0.018256 ms | 0.058671 ms | 3.214x |
| B1 H8 Qb64 Kb4096, k=439 | 0.021855 ms | 0.098575 ms | 4.510x |
| B4 H8 Qb64 Kb1024, k=132 | 0.021488 ms | 0.118878 ms | 5.532x |

Reproducer:

```bash
python benchmarks/bench_adaptive_sparse_block_mask.py \
  --batch-size 1 --num-heads 8 --q-blocks 64 --k-blocks 1024
```

The baseline uses preallocated `torch.topk` outputs, threshold comparison, and
a precomputed mandatory prefix/window mask. `alpha=1` makes the row budget
constant so the two implementations have identical semantics, including
selecting all ties at the threshold. These are kernel-pipeline measurements,
not yet end-to-end sparse-attention results.
