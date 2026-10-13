# Experimental SM100 Union-GQA Prefill

Union-GQA exploits similarity between neighboring tokens' selected KV block
indices to improve Tensor Core utilization in GQA block-sparse attention.
In the target workload, each KV head serves only 16 query heads. Packing the
heads of one query token provides 16 query rows, leaving many rows unused in
64- or 128-row Blackwell attention tiles. Combining several tokens can fill
the tile, but their independently selected KV blocks require per-token sparse
masking rather than the shared key selection of a dense attention tile.
We compare against [Experimental Task-Scheduled Attention](../../attention/prims_ts/README.md),
referred to as PrimsTS below (from its source directory, `prims_ts`).
**In the SM100 benchmark below, Union-GQA is 1.04x to 4.01x faster than
PrimsTS across all tested query lengths when adjacent-query top-k KV-block
similarity (the selected-block sharing ratio) is strictly above 50%
(56.25% and higher in the tested grid).**

We observed that adjacent queries share some selected KV-block indices,
motivating Union-GQA's design. High KV-index sharing between adjacent queries
is our target scenario. Union-GQA forms a union for each group of eight
queries sharing a KV head and packs their heads into 128 rows, reusing staged
K/V while preserving each query's sparse and causal masks. Each CTA handles
two groups with separate unions; lower sharing increases the union size and
masked computation, reducing the benefit.

This package provides an explicit opt-in sparse-prefill path for SM100
(compute capability 10.0). Both the public `union_gqa_prefill` API and its
backend are experimental. Calling the API is the opt-in and emits an
`ExperimentalWarning`; the backend does not participate in automatic routing,
autotuning, trace-apply, or AOT packaging.

The current specialization supports only:

- contiguous, 16-byte-aligned BF16 Q, K, V, and output tensors with head
  dimension 128;
- causal flat self-attention;
- a fixed 16:1 query-to-KV head ratio;
- one to 16 selected 64-token KV blocks per query and KV head;
- CUDA 12.8 or newer on exact compute capability 10.0.

Each nonnegative entry in `block_indices` is local to its sequence and selects
the interval `[64 * block, 64 * (block + 1))`. Entries for a query and KV head
must be unique; negative entries are padding. Blocks beyond the causal boundary
are ignored. `cu_seqlens_q` must start at zero, end at `q.shape[0]`, and be
nondecreasing.

`max_seqlen_q` is the launch-grid capacity. Supplying it avoids a device-to-host
synchronization and is required during CUDA graph capture. The caller must set
it to at least the longest sequence represented by `cu_seqlens_q`; a smaller
value is invalid. CPU offsets are checked before launch. Invalid CUDA offsets
or an underestimated bound raise an asynchronous device error.

```python
import torch
from flashinfer import union_gqa_prefill

seq_len = 1024
q = torch.randn(seq_len, 16, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(seq_len, 1, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn_like(k)
block_indices = torch.arange(8, dtype=torch.int32, device="cuda")
block_indices = block_indices.view(1, 1, 8).expand(seq_len, 1, 8).contiguous()
cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

out = union_gqa_prefill(
    q,
    k,
    v,
    block_indices,
    cu_seqlens_q,
    max_seqlen_q=seq_len,
)
```

Run `examples/experimental/union_gqa_prefill.py` for a complete example and
`benchmarks/bench_union_gqa.py` for the reproducible ratio-aware testbench.

## Benchmark Workload and Measurement

PrimsTS uses `BlockSparseTSWrapper` with token-level block sparse row (BSR)
routes (`q_block_size=1`, `kv_block_size=64`), a 16:1 GQA head ratio, BF16,
and causal masking. It processes each query's selected blocks without
exploiting KV-block sharing between adjacent queries.

The strict-overlap grid has five measured query lengths, doubling from 2048
to 32768, and 17 sharing ratios from 0% to 100% in 6.25-percentage-point
steps: 85 cases per cache mode. Inputs are deterministic BF16 with 16 query
heads, one KV head, head dimension 128, topk=16, and seed=42.

Every measured query follows an 8192-token KV history. `seq_len` means the
number of measured queries; full Q/K/V storage has `8192 + seq_len` tokens.
The selected blocks are all in the fixed 128-block history, so every query
attends exactly 1024 fully visible keys. The generator advances a sorted
16-block circular window by `stride_blocks = 16 * (1 - sharing_ratio)` per
query. Ratios not exactly representable at this top-k are rejected.
Assertions check each adjacent intersection, including stage boundaries,
uniqueness, full causal visibility, and every eight-query union. Actual
sharing equals requested sharing at all lengths. Each stage has exactly
`16 + 7 * stride_blocks` union blocks; ratio=0 means zero adjacent overlap
and 128 union blocks even at the shortest measured length.

Only the query suffix is processed and timed; the 8192-token history supplies
K/V context. Union-GQA skips history queries through a benchmark-only JIT
specialization, while PrimsTS receives the query suffix with full K/V and a
causal offset of 8192. Regression tests verify correctness against the
production kernel and FP32, and confirm that history outputs remain untouched.

This fixed-history benchmark holds the selected KV working set and per-stage
union geometry constant across query lengths; it does not measure end-to-end
self-prefill.

The cold-L2 comparison runs Union-GQA and PrimsTS sequentially in separate
processes using the same CUDA Event timer. Every iteration flushes L2 before
the start event; flush work is outside the reported latency. Each point
preheats for 1 second, then uses 10 warmups and three rounds of 100 iterations.
The reported latency is the median of the three round means. Compilation,
input and static metadata construction, allocations, and correctness checks
are outside timing.
Union-GQA's in-kernel union construction and PrimsTS's `run()` route preparation
remain timed.

The latest comparison issues no GPU clock configuration commands. GPU
frequencies are not fixed or validated during this run.

The previous history-window table was superseded: short causal histories and
modulo wrapping made actual overlap differ from the row labels, especially
at ratio=0. The subsequent default-clock warm/cold results are retained as
historical artifacts. The latest table below measures both backends with cold L2.

## Reproducing the Measurements

Run from this checkout on an otherwise idle SM100 GPU, with CUDA 13.0,
a compatible CUDA-enabled PyTorch build, and CUTLASS DSL 4.8.0:

```bash
python -m pip install "nvidia-cutlass-dsl[cu13]==4.8.0" "quack-kernels==0.6.5"
```

Reproduce the table by timing PrimsTS and standalone Union-GQA sequentially
in separate processes on an idle GPU. These commands use **unlocked GPU clocks** and
**cold L2**; neither command configures GPU frequencies:

```bash
PYTHONPATH=. python benchmarks/bench_union_gqa_primsts_compare.py \
    --cold-l2-cache --backend primsts \
    --warmup 10 --iters 100 --rounds 3 --preheat-ms 1000 \
    > /tmp/union_gqa_primsts_comparison_cold.csv
PYTHONPATH=. python benchmarks/bench_union_gqa.py \
    --cold-l2-cache --csv \
    --dry-run-iters 10 --repeat-iters 100 --rounds 3 --preheat-ms 1000 \
    > /tmp/union_gqa_standalone_cold.csv
```

Each command uses the full 85-case grid, a 1-second preheat, 10 warmups,
and three rounds of 100 iterations per case. The table uses `primsts_us`
from the comparison CSV and `latency_us` from the standalone CSV. Match rows
by `seq_len` and sharing ratio: the comparison CSV calls this field
`requested_sharing`, while the standalone CSV calls it `sharing_ratio`.
Both equal the actual sharing ratio labeled "Sharing" in the table below.
PrimsTS timing includes route preparation. With `--backend primsts`, the
comparison script runs both kernels for full-output comparisons and FP32
sample checks, but times only PrimsTS. A short run can use
`--seq-lens 2048 --sharing-ratios 0,0.5,1`.

For an optional Union-GQA timing cross-check, run the comparison command
with `--backend both` and compare its `union_us` with `latency_us` from the
standalone CSV, matching rows as above.

For an optional locked-clock control, use
`benchmarks/bench_union_gqa_cold_locked.py`.

## Union-GQA versus PrimsTS

The Union-GQA rows report latency at each sharing ratio. The PrimsTS row
reports the arithmetic mean over all 17 measured ratios for each length,
since its latency varies little with sharing.

<!-- COLD_LOCKED_RESULTS_START -->

Measured on NVIDIA SM100. Standalone
and PrimsTS each completed the full 85-case grid in separate, sequential
processes. All 85 cases passed full-output Union/PrimsTS comparisons and
independent FP32 sample checks (`atol=0.002`, `rtol=0.02`). The maximum
absolute output difference was `0.000244140625`.

Each cell shows Union-GQA cold-L2 latency in microseconds
(speedup over PrimsTS). Speedup is PrimsTS / Union-GQA latency at the same
length and ratio; values above 1 mean Union-GQA is faster.

| Sharing ratio / Sequence length | 2,048                  | 4,096                   | 8,192                   | 16,384                  | 32,768                  |
| ------------------------: | ---------------------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: |
| 0%                        | 181.28&nbsp;(0.56x)    | 310.87&nbsp;(0.59x)     | 567.26&nbsp;(0.61x)     | 1034.77&nbsp;(0.65x)    | 2066.78&nbsp;(0.63x)    |
| 6.25%                     | 169.56&nbsp;(0.59x)    | 291.16&nbsp;(0.63x)     | 530.79&nbsp;(0.65x)     | 976.72&nbsp;(0.69x)     | 1934.79&nbsp;(0.68x)    |
| 12.5%                     | 155.51&nbsp;(0.65x)    | 266.24&nbsp;(0.69x)     | 487.68&nbsp;(0.71x)     | 900.18&nbsp;(0.74x)     | 1793.68&nbsp;(0.73x)    |
| 18.75%                    | 151.24&nbsp;(0.67x)    | 255.89&nbsp;(0.72x)     | 462.01&nbsp;(0.75x)     | 861.86&nbsp;(0.78x)     | 1707.65&nbsp;(0.77x)    |
| 25%                       | 135.19&nbsp;(0.75x) | 228.62&nbsp;(0.80x) | 417.74&nbsp;(0.83x) | 768.86&nbsp;(0.87x) | 1526.24&nbsp;(0.86x) |
| 31.25%                    | 133.07&nbsp;(0.76x)    | 221.60&nbsp;(0.83x)     | 403.82&nbsp;(0.86x)     | 757.44&nbsp;(0.88x)     | 1505.08&nbsp;(0.87x)    |
| 37.5%                     | 121.16&nbsp;(0.83x)    | 212.93&nbsp;(0.86x)     | 373.82&nbsp;(0.93x)     | 685.77&nbsp;(0.98x)     | 1359.74&nbsp;(0.96x)    |
| 43.75%                    | 115.82&nbsp;(0.87x)    | 196.21&nbsp;(0.93x)     | 349.18&nbsp;(0.99x)     | 647.37&nbsp;(1.03x)     | 1281.80&nbsp;(1.02x)    |
| 50%                       | 105.52&nbsp;(0.96x)    | 176.51&nbsp;(1.04x)     | 316.39&nbsp;(1.10x)     | 575.78&nbsp;(1.17x)     | 1156.47&nbsp;(1.13x)    |
| 56.25%                    | 97.02&nbsp;(1.04x)     | 168.01&nbsp;(1.09x)     | 303.17&nbsp;(1.14x)     | 548.70&nbsp;(1.22x)     | 1098.13&nbsp;(1.19x)    |
| 62.5%                     | 88.38&nbsp;(1.14x)     | 152.88&nbsp;(1.20x)     | 279.27&nbsp;(1.24x)     | 501.00&nbsp;(1.34x)     | 1010.76&nbsp;(1.29x)    |
| 68.75%                    | 79.99&nbsp;(1.26x)     | 136.59&nbsp;(1.34x)     | 242.85&nbsp;(1.42x)     | 443.06&nbsp;(1.51x)     | 889.82&nbsp;(1.47x)     |
| 75%                   | 70.82&nbsp;(1.43x) | 123.23&nbsp;(1.49x) | 219.60&nbsp;(1.58x) | 400.04&nbsp;(1.67x) | 806.27&nbsp;(1.62x) |
| 81.25%                    | 61.58&nbsp;(1.63x)     | 105.78&nbsp;(1.73x)     | 191.13&nbsp;(1.81x)     | 346.56&nbsp;(1.93x)     | 682.03&nbsp;(1.92x)     |
| 87.5%                     | 53.00&nbsp;(1.89x)     | 89.07&nbsp;(2.05x)      | 161.30&nbsp;(2.14x)     | 283.92&nbsp;(2.35x)     | 563.91&nbsp;(2.32x)     |
| 93.75%                    | 43.83&nbsp;(2.28x)     | 72.04&nbsp;(2.53x)      | 130.22&nbsp;(2.65x)     | 226.91&nbsp;(2.94x)     | 452.77&nbsp;(2.88x)     |
| 100%                  | 34.77&nbsp;(2.92x) | 55.08&nbsp;(3.33x)  | 98.94&nbsp;(3.50x)  | 174.75&nbsp;(3.82x) | 325.23&nbsp;(4.01x) |
| PrimsTS                   | 100.90                 | 183.24                  | 345.96                  | 669.73                  | 1307.97                 |

The sharing ratio measures the similarity of adjacent queries' selected
KV blocks for the same KV head:

$$
\mathrm{sharing\_ratio} =
\frac{\left|\mathrm{selected\_blocks}(q_i) \cap \mathrm{selected\_blocks}(q_{i+1})\right|}{\mathrm{topk}}.
$$

With topk=16, 50% sharing means eight blocks in common; 100% means
identical selections. High sharing is the target scenario: Union-GQA reuses
staged K/V across queries while filling Tensor Core rows. At low sharing,
larger unions require more block loads and masked computation, which can
outweigh the reuse benefit and make Union-GQA slower than PrimsTS.

The measured performance trends can be interpreted as follows:

1. When the sharing ratio exceeds 50%, Union-GQA outperforms PrimsTS at
   every tested sequence length.
2. For sequence lengths below 32768, the speedup generally increases with
   sequence length. Union-GQA computes the union of the KV indices for each
   CTA's query stages in the prologue. Its persistent schedule overlaps the
   prologue for the next work item with the epilogue of the current one.
   Longer sequences provide more work items per CTA, amortizing the initial
   prologue and final epilogue and hiding more of the union-construction cost.
3. At sharing ratios below 100%, the speedup decreases when the sequence
   length grows from 16384 to 32768. A possible explanation is increased L2
   cache pressure and a shift toward memory-bound execution: more queries
   require more KV-block accesses, and lower sharing produces larger unions
   with less reuse per staged block. The selected KV working set remains
   fixed in this benchmark, so this refers to total block accesses rather
   than an increase in distinct selected blocks. At 100% sharing, the
   speedup continues to increase; greater KV reuse reduces memory traffic
   and cache pressure, potentially delaying the memory bottleneck. Lower
   sharing and longer sequences make that bottleneck more likely.
4. At low sharing ratios, Union-GQA is slower than PrimsTS because each
   KV block contributes to fewer query tokens. Although the physical MMA
   tile retains its fixed M dimension, fewer query rows perform useful work
   for that block, reducing the effective M dimension and Tensor Core
   utilization. Lower sharing also increases the union size, requiring more
   KV blocks to be loaded and processed. Together, the lower useful-work
   fraction and greater block-processing cost can outweigh the benefit of
   packing queries.

<!-- COLD_LOCKED_RESULTS_END -->

## Acknowledgments

This experimental SM100 union-GQA sparse-prefill kernel was developed
for Xiaomi's recently released model architecture. The kernel uses a
two-warpgroup query schedule, a pipelined QK/PV mainloop, and a mixed
MUFU/polynomial softmax exponential path. Its launcher, metadata contract, and
sequence-tail handling are integrated through FlashInfer's framework-neutral
TVM-FFI JIT path.

**Collaborators:** [@bingyuel](https://github.com/bingyuel)(NVIDIA), [@uchihatmtkinu](https://github.com/uchihatmtkinu)(NVIDIA), and [@Ezio-csm](https://github.com/Ezio-csm)(XIAOMI).
