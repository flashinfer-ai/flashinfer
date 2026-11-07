# Multi-configuration module benchmark protocol

This stage measures one attention operator, not model forward or serving.
Use the existing FlashInfer `bench_gpu_time` helper and the runnable
`examples/xqa_decode_buckets.py`; no separate benchmark framework is required.

## Reference and adaptations

Length distributions come from
[Tencent HPC-Ops decode benchmark](https://github.com/Tencent/hpc-ops/blob/2a2e26562433a8ba4b504858f1c938eb7612c901/benchmark/attention_decode/bench_attention_decode_bf16.py).
Its default measures a single BF16 attention operator with Q8/KV1, dimension 128,
page64 and q1. This contribution retains BF16 Q/output, NVFP4 KV, dimension 256
and the declared configuration below. This is reuse of workload distributions,
not execution of Tencent's BF16 kernel on SM120.

The three 64-request patterns are reduced to 32 requests, retaining their length
proportions. The other six original patterns are unchanged:

| Pattern | Lengths in this stage |
| --- | --- |
| uniform512 | 32 x 512 |
| uniform4096 | 32 x 4096 |
| mixed | 16 x 128 + 16 x 4096 |
| extreme | 15 x 64 + 16384 |
| one64k7 | 65536 + 7 x 4096 |
| one64k15 | 65536 + 15 x 4096 |
| one64k31 | 65536 + 31 x 4096 |
| one128k31 | 131072 + 31 x 4096 |
| two32k30 | 2 x 32768 + 30 x 4096 |

## Representative configuration matrix

| Page | KV heads | Q/KV ratio | Query tokens/request | Patterns |
| ---: | ---: | ---: | ---: | --- |
| 32 | 1 | 6 | 1 | All nine |
| 16 | 2 | 4 | 2 | uniform512, uniform4096, extreme, one64k7 |
| 32 | 4 | 6 | 3 | uniform512, uniform4096, extreme, one64k7 |
| 64 | 8 | 8 | 5 | uniform512, uniform4096, extreme, one64k7 |
| 32 | 4 | 4 | 4 | one64k7 |

These 22 cases are representative performance coverage, not a Cartesian sweep of
all supported configurations. Correctness coverage is separate. Per-case lengths
include the causal query suffix. Both native and candidate receive the same mask,
cache, query and page-table geometry. Keep native's default dispatch/tile choice;
do not force it to match the candidate's 16-row tile.

## Timing and reporting

- One physical GPU and identical CPU affinity for paired controls/candidate.
- One layer per window; preserve previous two-layer measurements separately.
- Record native-before, candidate-compute-only, candidate-plan-plus-compute and
  native-after medians and samples. Direct plans have no Assign launch.
- Reuse existing FlashInfer hot CUDA Graph timing: five warmups, twenty samples,
  CUPTI off, cold-L2 handling off. Allocation and JIT are outside measurement.
  Tencent's original helper uses fifty event-timed replay samples by default;
  do not present our helper or sampling count as identical to theirs.
- GPU planning is included in the complete candidate window. Host completed-wall
  results are a separate measurement and cannot be inferred from GPU times.
- Check outputs and reset counters outside timing. Record resource usage and
  retain failures/partial runs. Report regressions, without retuning thresholds
  on each measured case or claiming universal non-regression.
- Persist one result per completed case so interrupted work can resume without
  rerunning completed cases. The existing bounded experiment launcher controls
  timeout, heartbeat, memory floor and ownership.

Status: completed once in `flashinfer-buckets-suite-20261030T170455Z`, with all 22
case output/counter checks passing and no retries or post-result kernel tuning.
Seventeen full-window medians are faster, five regress; the largest relative RMS
error is 0.004230 and maximum native bracket drift is 0.84%. Full results and
limitations follow below.

## Focused query-tile follow-up

`--suite query_tiles4` compares native, M16 and explicit M32 over the same
operand bank, with independent candidate plans/workspaces. It selects T3/R6/H4/P32
uniform512, uniform4096 and one64k7, plus T5/R8/H8/P64 one64k7. Native brackets
both candidates; each candidate retains separate compute-only and complete
planning-inclusive windows. The same helper, precision and sampling settings
above apply. This diagnoses the M16 row-group split; it does not replace the
22-case historical matrix or constitute a new full performance qualification.

`--suite query_modes4` subsequently compares existing persistent/aligned M32
ownership on those same four workloads. Both arms share Q/KV/scales/page tables
and the constant causal mask, with independent outputs/plans/workspaces. Actual
device descriptors and owner/partial counts are copied outside timing. Generic
SM120 native grids are explicitly marked source-derived, not profiler-observed.
One native bracket surrounds both modes. Compare within the matched physical
GPU/NUMA run; historical timings on another GPU are context, not direct controls.

## Compile and shared-plan integration anchor

The bounded compile follow-up uses the public bucket API with default Inductor
and `fullgraph=True`, then CUDA Graph replay with changed inputs. A separate
Dynamo explanation checks operator counts; it does not replace actual Inductor
execution. Plan construction, JIT, workspace allocation and Host descriptor
uploads stay outside compilation/capture. The GPU planner may be captured once
before multiple compatible attention consumers.

The timing anchor is two independent attention layers sharing request metadata:
B32, uniform4096, P32, H4, R6, T3, M32, aligned mode. Compare native before/after,
eager per-layer preparation, eager once-per-window preparation and compiled
once-per-window preparation on identical operands and one GPU/CPU placement.
Each layer has its own output and workspace. Use the existing hot CUDA Graph
helper and record all samples; count actual plan/compute operations and inspect
Inductor output for additional copies or kernels. This measures a two-attention
module window, not a whole transformer block or vLLM serving latency. It does
not claim that `torch.compile` itself fuses the opaque attention kernels.

## Results and source provenance

All results below were measured on RTX 5090 (SM120), PyTorch 2.14.0+cu130 and
CUDA 13, using BF16 Q/output, NVFP4 KV and E4M3 block scales. Each invocation
ran once with seed 0 and static metadata. Native-before and native-after use
identical placement and operands to their candidate. Separate invocations use
their own native controls; absolute times across invocations are not paired
comparisons. Results are module windows, not model or serving latency.

The 22-case M16 suite used the implementation at
`ec05ef44bb2683d3e6e28d0531f085db177150ad`. M32 and ownership follow-ups used
the explicit M32 implementation at `753cc23be1005650678c220c72d29a0421c3ebe3`;
subsequent ownership example/report changes were published at
`3aa1974545599c6b6619f19628cb2c25232109ca`. The real dispatcher/compile interface
was published at `5b497c46dd51bd3b3a949c5ce712b7e9b14b6ea3`. These revisions
record provenance; no runtime commit or file-hash requirement is imposed.

### Representative M16 suite

One attention layer, persistent mode, GPU planner, physical GPU0/NUMA0.
The hot Graph helper, five warmups and twenty samples described above apply.
The full window includes one Assign except for the two direct uniform512 cases.
Speedup is the mean native median divided by the full candidate median; values
below one are regressions. P/H/R/T denotes page size, KV heads, Q/KV ratio and
query length.

| P/H/R/T | Pattern | Native before / after (us) | Compute only (us) | Assign + compute (us) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| P32/H1/R6/T1 | uniform512 | 12.048 / 12.048 | 12.253 | 12.253 | 0.98x |
| P32/H1/R6/T1 | uniform4096 | 32.323 / 32.323 | 31.914 | 38.467 | 0.84x |
| P32/H1/R6/T1 | mixed | 31.299 / 31.299 | 20.240 | 28.125 | 1.11x |
| P32/H1/R6/T1 | extreme | 48.912 / 49.014 | 21.878 | 26.179 | 1.87x |
| P32/H1/R6/T1 | one64k7 | 86.813 / 86.771 | 37.648 | 40.717 | 2.13x |
| P32/H1/R6/T1 | one64k15 | 163.035 / 163.008 | 38.058 | 42.478 | 3.84x |
| P32/H1/R6/T1 | one64k31 | 315.254 / 315.509 | 45.067 | 51.984 | 6.07x |
| P32/H1/R6/T1 | one128k31 | 618.355 / 618.563 | 57.306 | 64.474 | 9.59x |
| P32/H1/R6/T1 | two32k30 | 162.626 / 163.088 | 44.557 | 52.106 | 3.13x |
| P16/H2/R4/T2 | uniform512 | 19.011 / 19.011 | 15.120 | 15.120 | 1.26x |
| P16/H2/R4/T2 | uniform4096 | 81.680 / 81.885 | 78.506 | 80.451 | 1.02x |
| P16/H2/R4/T2 | extreme | 123.510 / 122.894 | 26.384 | 31.504 | 3.91x |
| P16/H2/R4/T2 | one64k7 | 236.150 / 236.611 | 63.690 | 63.614 | 3.72x |
| P32/H4/R6/T3 | uniform512 | 24.336 / 24.541 | 36.829 | 43.997 | 0.56x |
| P32/H4/R6/T3 | uniform4096 | 161.056 / 161.091 | 231.283 | 239.171 | 0.67x |
| P32/H4/R6/T3 | extreme | 278.030 / 278.338 | 38.813 | 43.178 | 6.44x |
| P32/H4/R6/T3 | one64k7 | 480.835 / 480.682 | 208.042 | 213.674 | 2.25x |
| P64/H8/R8/T5 | uniform512 | 89.770 / 89.870 | 86.186 | 92.925 | 0.97x |
| P64/H8/R8/T5 | uniform4096 | 610.627 / 611.699 | 589.840 | 605.043 | 1.01x |
| P64/H8/R8/T5 | extreme | 557.021 / 556.507 | 105.898 | 110.301 | 5.05x |
| P64/H8/R8/T5 | one64k7 | 1131.075 / 1131.280 | 623.274 | 625.117 | 1.81x |
| P32/H4/R4/T4 | one64k7 | 480.237 / 481.446 | 84.494 | 89.768 | 5.36x |

All 22 output/counter checks passed the existing 3% relative RMS tolerance before
and after timing; maximum measured relative RMS was 0.004230. Seventeen complete
windows improved and five regressed. Maximum native bracket drift was 0.84%.
The balanced T3 cases already regress in compute, where M16 uses two row groups
for eighteen logical rows. Extra KV reads and row-group overhead are plausible
contributors from source inspection, not an isolated profiling attribution.
For q1 uniform4096, planning erases the small compute-only improvement.
Independently measured components are not assumed additive; small inversions
between compute and full times are retained.

### Explicit M32 comparison

One attention layer, persistent mode, physical GPU0/NUMA0, matched operands for
M16, M32 and native. Every full candidate window includes one GPU Assign.
The same hot Graph protocol applies. Units are microseconds.

| P/H/R/T | Length pattern | Native before / after | M16 compute / full | M32 compute / full | M32 vs M16 | M32 vs native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32/4/6/3 | 32 x 512 | 24.336 / 24.336 | 36.893 / 43.382 | 24.950 / 31.914 | 1.36x | 0.76x |
| 32/4/6/3 | 32 x 4096 | 160.835 / 161.142 | 231.286 / 239.259 | 224.323 / 232.259 | 1.03x | 0.69x |
| 32/4/6/3 | 65536 + 7 x 4096 | 482.032 / 482.574 | 208.144 / 213.059 | 107.997 / 113.219 | 1.88x | 4.26x |
| 64/8/8/5 | 65536 + 7 x 4096 | 1131.331 / 1133.482 | 625.834 / 627.370 | 601.870 / 605.590 | 1.04x | 1.87x |

Maximum relative RMS was 0.0041943 and native bracket drift 0.191%. M32 improves
these four cases over M16; both balanced T3 full windows still regress against
native. M32 uses 86,656 shared bytes versus M16's 70,272. Qualified M32 variants
use 219–235 registers, 64/112/128 stack bytes and one resident CTA/SM, with zero
compiler spill loads/stores. M16 remains the default.

### M32 ownership comparison

One attention layer, physical GPU7/NUMA1. Persistent and aligned share the same
Q/KV/scales/page tables and causal mask, with independent plans/workspaces.
Every full window includes one GPU Assign; units are microseconds.

| P/H/R/T | Length pattern | Native before / after | Persistent compute / full | Aligned compute / full | Aligned vs persistent full | Aligned vs native full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32/4/6/3 | 32 x 512 | 24.131 / 24.131 | 24.746 / 31.094 | 24.541 / 30.376 | 1.02x | 0.79x |
| 32/4/6/3 | 32 x 4096 | 158.790 / 158.787 | 221.099 / 229.034 | 161.245 / 168.824 | 1.36x | 0.94x |
| 32/4/6/3 | 65536 + 7 x 4096 | 474.640 / 476.074 | 106.768 / 111.582 | 167.187 / 173.432 | 0.64x | 2.74x |
| 64/8/8/5 | 65536 + 7 x 4096 | 1118.224 / 1118.226 | 596.293 / 599.416 | 577.194 / 578.730 | 1.04x | 1.93x |

Maximum relative RMS was 0.0041943 and native bracket drift 0.302%. For balanced 4K,
actual descriptors show persistent's 64 tasks, 43 active owners and two partials
per request in each of four groups. Aligned has 32 whole-request tasks per group
and no inter-CTA request merge: 128 active CTAs in a 172-CTA grid. Generic native
SM120 source derives 128 CTAs with no inter-CTA merge; native counts are
source-derived, while bucket counts come from copied descriptors outside timing.
These are not profiler launch measurements. Ownership, restarts, barriers and
compiler output vary together, so the comparison does not isolate merge cost.
The T3 skew counterexample favors persistent even though long-request partial
counts match; partial counts alone do not explain performance.

R6/P32 persistent/aligned use 219/228 registers and 112 stack bytes;
R8/P64 use 232/223 registers and 64 stack bytes. Shared memory is 86,656 bytes,
with one resident CTA/SM. Unchanged kernels reuse their zero-spill qualification.

### Compile and shared-plan results

PyTorch 2.14.0+cu130 default Inductor `fullgraph=True` passed warmed fixed-shape
public calls for direct q1, prepared M32 persistent, prepared M32 aligned,
host-prepared compiled compute, and two independent layer consumers sharing
one GPU plan. Q/K/V and lengths changed during CUDA Graph replay, including
zero/restore. Cold JIT and the first experimental warning stay outside compile.

Plan `opcheck` schema/fake-tensor checks passed. Compute fake-tensor checks
passed; schema checks encountered PyTorch's unsupported Float8 `mul_cuda`
inside `allclose`. Read-only inputs were independently checked unchanged through
byte views. Real Inductor numerical replay passed the same output tolerance.
No autograd, dynamic-shape or dynamic-AOT opcheck claim is made. Eager storage
checks reject misalignment and output aliasing; the compiler can normalize
misaligned query/output storage with conditional `copy_if_misaligned`, and those
accepted compiled cases passed numerical checks.

Two-layer B32 × 4096, P32/H4/R6/T3/M32 aligned, physical GPU7/NUMA1, independent
layer operands/outputs/workspaces. Hot Graph timing uses five warmups and twenty
samples; units are microseconds. Each shared window contains one preparation
and two attention computes.

| Window | GPU Assign calls | Median (us) |
| --- | ---: | ---: |
| Native before | 0 | 317.920 |
| Eager prepare before each layer | 2 | 337.066 |
| Eager shared prepare | 1 | 330.102 |
| Inductor shared prepare | 1 | 331.997 |
| Native after | 0 | 322.730 |

Native paired mean is 320.325 us, with 1.50% bracket drift. Shared eager preparation
reduces the observed per-layer-preparation window by 2.07%, while shared eager
and compiled remain 3.05% and 3.64% slower than native. This does not establish
a compile speedup. Actual Inductor output has one plan and two compute calls,
no unconditional capacity-sized clones/copies or additional generated GPU
kernels in the aligned-input window. Conditional alignment helpers remain.
Compute resources are 228 registers, 86,656 shared bytes, 112 stack bytes and one
resident CTA/SM.

Reproduce this boundary with explicit dimensions:

```bash
python examples/xqa_decode_buckets.py --compare-compile --layers 2 --mode aligned \
    --lengths 4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096 \
    --num-kv-heads 4 --head-group-ratio 6 --q-len-per-req 3 --query-tile 32 \
    --output-json compile.json
```

### Host-planner completed-wall comparison

This earlier q1/H1/R6/P32 comparison used two attention layers, persistent mode,
physical GPU0/NUMA0. CPU/device lengths were identical and static, so both arms
exclude length uploads. Completed-wall samples include stream completion;
GPU captured wall averages medians bracketing the host arm. GPU external wall
prepares outside its compute-only Graph, matching the host integration boundary.
Units are microseconds; component timings were measured independently.

| Lengths | GPU captured complete wall | GPU external complete wall | Host fill + H2D + Graph complete wall | CPU C++/FFI fill | H2D complete wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 4096 | 76.040 | 82.297 | 93.134 | 2.660 | 8.782 |
| 65536 + 7 x 4096 | 83.835 | 90.809 | 102.408 | 1.678 | 8.113 |
| 131072 + 31 x 4096 | 150.187 | 156.988 | 166.353 | 2.553 | 8.056 |
| 512, direct | 30.341 | 30.446 | 30.648 | no-op | no copy |

Host preparation is slower for all three prepared patterns, including against
external GPU preparation. GPU remains the default. Compute-only Graph medians
for GPU/host descriptors are 63.248/63.248, 74.922/74.717 and 137.027/136.874 us for
the prepared rows; they exclude host fill/H2D/dependencies. New multi-query
completed-wall host-planner timing is not inferred from this q1 result.

### Earlier direct and resource anchors

After the capacity-fitting direct change, two-layer q1/H1/R6/P32 windows on
physical GPU0/NUMA0 used no Assign for any row below. The same hot Graph protocol
applies. These small-work measurements precede the final matrix and remain
separate observations.

| Live lengths | Table capacity per request | Native before / after (us) | Complete candidate (us) |
| --- | ---: | ---: | ---: |
| 256 | 256 | 21.261 / 21.261 | 21.469 |
| 512 | 512 | 23.691 / 23.566 | 23.822 |
| 512 | 131072 | 70.822 / 70.822 | 23.310 |
| 4 x 512 | 512 | 23.718 / 23.581 | 23.874 |

The tight-capacity rows remain 0.8–1.0% slower than their paired native means;
the wide-table B1 improvement depends on capacity exposed by the caller. It
cannot be assumed for every serving configuration.

Earlier page-specialization qualification measured the following q1/H1/R6
resources; every row had 70,272 shared bytes, one resident CTA/SM and zero
compiler spill loads/stores. Later head-group changes have their own measured
register counts, so these are historical specialization anchors.

| Page size | Persistent registers | Aligned registers | Direct registers | Stack bytes |
| --- | ---: | ---: | ---: | ---: |
| 16 | 208 | 196 | 194 | 128 |
| 32 | 201 | 186 | 186 | 112 |
| 64 | 213 | 194 | 193 | 64 |

After shared-head expansion, page32 q1 resource reports were as follows, with
70,272 shared bytes, 112 stack bytes and one resident CTA/SM. Nineteen compiled
specializations reported zero spill loads/stores.

| Ratio | Persistent registers | Aligned registers | Direct registers |
| --- | ---: | ---: | ---: |
| 4 | 206 | 191 | 180 |
| 6 | 204 | 183 | 183 |
| 8 | 206 | 191 | 191 |

Historical setup failures and superseded intermediate timing are retained in
Git/evidence history. No intermediate result is substituted for the matched
controls above. Framework/model integration has a separate timing boundary.
