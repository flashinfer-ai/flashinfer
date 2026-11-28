# KPage execution for CuTe DSL VSA

This experimental, JIT-only implementation is extracted from NVIDIA cuDNN
Frontend commit `862a438c851b77305391916f7db61b14d8fdeb4e`, using the
`b300_pr1996_retest_20261111/source` snapshot. `SOURCE_MANIFEST.json` records
the original file SHA-256 digests. Original copyright and license notices are
retained. The source kernels gather valid 16-token KV pages into MMA tiles;
they preserve the supplied sparse connections, rather than selecting top-k.

Enable explicitly with `BlockSparseAttentionWrapper.plan(kpage=True)` on
`backend="vsa_sm100_blk64"` or `backend="vsa_sm100_blk128"`. The default is
False; Cake, auto, and other backends reject True. No auto-dispatch or AOT
registration is added. See `examples/vsa_kpage.py` for a runnable example.

| Contract | blk64 | blk128 |
| --- | --- | --- |
| Architecture | SM100/SM103 | SM100/SM103 |
| Q/K/V/output | BF16, D=128 | BF16, D=128 |
| Heads | MHA | MHA/GQA, sparse pattern per Q head |
| Sequence lengths | Multiples of 64 | Independent Q/KV lengths, including tails |
| Query valid prefixes | All rows valid | Optional `q_block_lens` |
| KV valid prefixes | Optional `kv_block_lens` | Optional `kv_block_lens` |

Requires CuTe DSL >=4.6.2 (tested with 4.8.0). Other FI modules may require a
newer DSL release. Forward only is exposed by FI; this does not add a backward
API. The extracted builder retains upstream backward-table helpers, but FI
does not build or retain deferred backward adjacency.

Use one sparse input form: BSR `indptr/indices`, boolean `block_mask`, or
native `q2k_indices`/`q2k_num`. Native indices have shape
`[1, Hq, ceil(M/R), capacity]`; counts have shape `[1, Hq, ceil(M/R)]`.
FI native indices/counts may also omit the leading singleton batch dimension,
matching the existing direct-index interface. Native indices, counts, and
lengths are int32 on the workspace CUDA device.
Omitting counts activates every capacity slot. Unused slots are ignored;
active indices must be in range and unique. BSR patterns are shared across
heads. Masks may be per Q head or per KV head (repeated over its GQA group).

`kv_block_lens` and `q_block_lens` are one-dimensional, shared across heads,
and describe valid **prefixes** in each sparse block. They cannot express holes.
Defaults are full blocks, clipped at the actual sequence end. Empty attention
rows produce O=0 and LSE=-inf. Padding query rows produce O=0 and LSE=0,
matching the source implementation. LSE uses the natural logarithm and FI's
`[M, Hq]` layout. Q/K/V and supplied output buffers must be compact NHD;
out/LSE may not overlap inputs or each other.

Only non-causal block sparsity is supported. Element masks, position encoding,
soft caps, FP8 scales, explicit CLC, split-KV other than None/1, and explicit
PDL are rejected. Finite positive, zero, and negative softmax scales are allowed.

`plan()` validates metadata and builds the page tables, including host
readbacks; call it outside graph capture. `run()` reuses them as Q/K/V values
change. Changing indices, counts, lengths, device, or shapes requires replan,
including in-place metadata edits. In-place edits are not detected. Failed
replanning invalidates the wrapper, requiring a successful plan before run.
Concurrent replanning/running on one wrapper is unsupported. Warm up execution
before capturing a fixed plan; graph replay must retain the wrapper and buffers.

The FI adaptations are import relocation, separation of the forward launcher
from backward imports, direct destination buffers, LSE stride support, and
FI-specific validation/lifecycle. Kernel math and page ordering are retained.
The source process-local compilation caches are retained; generated files are
not written into the package. Disk-cache integration and automatic selection
are outside this migration. Measure layout construction and reused execution
separately: KPage is not necessarily faster than ordinary VSA, especially blk64.
Run `python benchmarks/bench_vsa_kpage.py --block-size 64` (or 128) to measure
warm layout construction and reused execution with destination buffers.

## Migration validation

See [VALIDATION.md](VALIDATION.md) for the integrated feature matrix, testcase
results, precision criteria, and performance data, and
[VALIDATION_RESULTS.json](VALIDATION_RESULTS.json) for machine-readable results.

Validated on B200 (SM100), PyTorch 2.11.0a0, CUDA runtime 13.2 and CuTe DSL
4.8.0. The integration tests cover both block sizes, GQA, independent Q/KV
lengths, sequence tails, empty rows, query padding, nonpositive scales, native
indices/BSR/masks, direct outputs, failed replanning, switching back to ordinary
VSA, default metadata/scale behavior, replanning with changed metadata and
shapes, and fixed-plan CUDA graph replay with changed Q/K/V values. There are
47 passing KPage integration cases, including active query tails and multi-KV-head
GQA against an independent FP64 reference. Existing Cake
planning tests also pass. SM103 has not been tested during this migration.

Direct comparisons with the pinned cuDNN source produced bit-identical O/LSE
for blk64 MHA, blk128 MHA, blk128 GQA, and blk128 GQA with sequence tails.
A single warm timing check used M=N=8192, Hq=Hkv=8, D=128, 16 selected blocks
per query block, and 40 valid tokens per KV block, with all Q rows valid:

| Block size | Source reused forward | FI reused forward | FI warm plan |
| --- | --- | --- | --- |
| 64 | 0.088192 ms | 0.088096 ms | 0.659 ms |
| 128 | 0.065472 ms | 0.065408 ms | 0.666 ms |

These are migration regression checks, not general speedup claims. Both
forward measurements allocate outputs and exclude layout construction and
first JIT. `bench_gpu_time(enable_cupti=True)` fell back to CUDA events because
CUPTI was unavailable. Plan times are synchronized wall-clock medians of five
warm calls. GPU clocks were not locked. The runnable benchmark above uses
preallocated outputs, so its timing boundary differs from this source/API
comparison. Original source-file digests precede import relocation and FI
formatting; copied license text has whitespace normalized by repository hooks.

Before upstreaming, maintainers must assign an owner and tracking issue and
set a graduation target under the experimental policy.
