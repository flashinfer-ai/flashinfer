# SM120 BF16 VDN window attention

These measurements cover the window softmax operator, including its input
movement and output scatter. They do not cover VDN's linear recurrence,
learned gates, an entire model, or video generation. Inputs are synthetic BF16
activations with representative projection strides.

## Implementation and baseline

`VDNWindowAttentionWrapper` groups queries with equal KV unions and schedules
longer KV groups first. It uses FlashInfer's existing FA2 CUDA paged-prefill
kernel with a shared KV pool and one token per page. New Triton kernels fuse
Q permutation with strided V materialization and scatter the result. The
attention math is BF16 Q/K/V with FP32 accumulation; no FP8 or skip-softmax
approximation is involved. No new CuTe DSL attention kernel is introduced.

The benchmark baseline follows the decomposition in
[OpenVDN/vdn-minimax-h3 at 2f740c9](https://github.com/OpenVDN/vdn-minimax-h3/tree/2f740c9291431d89d4f2330743b093fac4390d09):
PyTorch SDPA for global/anchor query rows and PyTorch varlen attention for
gathered windows. Its implementation and attribution are in
[`bench_vdn_window.py`](../../bench_vdn_window.py).

## Environment and method

- Hardware reports `NVIDIA Graphics Device`, compute capability 12.0,
  PCI device ID `0x2BB310DE`, 73415 MiB; driver 580.95.05. The driver does not
  expose a commercial model name, so none is inferred.
- PyTorch `2.13.0+cu130`, CUDA 13.0, Python 3.12; Compute Sanitizer 2025.3.1.0.
- FlashInfer source base: `c0aca1fa944cd5efd0e7bf5c003dd77e35d341d1`.
  The source checkout reports version `0.0.0+unknown`; this is not a released
  package version. Each result includes the two production source SHA-256s.
- Five warmups, 21 samples per method, randomized interleaved order (seed 71),
  input seed 20260927. Every sample includes copies/gathers, allocation,
  attention, and output scatter. Planning is excluded and reported separately.
- The table uses host wall time through CUDA synchronization. Raw CUDA-event
  spans are also provided; neither timer is a sum of individual kernel times.
- `warm/` reuses inputs without an explicit cache flush. `cold/` zeros a
  256 MiB buffer and synchronizes before each timed call, outside the timer.
  Both modes use warmed JIT compilation and reused plans.
- `peak_run_extra_bytes` measures transient allocations and the returned
  output above the existing allocation baseline. It excludes persistent
  plans and FlashInfer's caller-owned 128 MiB workspace.

## Results

All times are median milliseconds. Speedup is PyTorch / FlashInfer.
Every case has 128 channels per head. Each JSON includes full geometry,
strides, error statistics, plan time, memory data and all timing samples.

| Case | Warm PyTorch | Warm FI | Speedup | Cold PyTorch | Cold FI | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 107f-7h-514 | 18.1074 | 16.0277 | 1.130x | 18.0645 | 16.0205 | 1.128x |
| 37f-56h-514 | 47.5468 | 43.0807 | 1.104x | 47.6027 | 43.1076 | 1.104x |
| 37f-7h-386 | 6.3094 | 5.5333 | 1.140x | 6.2862 | 5.5253 | 1.138x |
| 37f-7h-514 | 6.1533 | 5.4283 | 1.134x | 6.1444 | 5.4470 | 1.128x |
| columns-21f-h7 | 0.2490 | 0.2220 | 1.121x | 0.2521 | 0.2277 | 1.107x |
| dense-128-h1 | 0.0564 | 0.0658 | 0.857x | 0.0572 | 0.0676 | 0.846x |
| dense-21f-h7 | 0.3079 | 0.2432 | 1.266x | 0.3215 | 0.2481 | 1.296x |
| local-13f-h7 | 0.0850 | 0.0721 | 1.180x | 0.0882 | 0.0766 | 1.152x |
| rows-21f-h7 | 0.2385 | 0.2088 | 1.142x | 0.2480 | 0.2173 | 1.141x |
| tail-127-h7 | 0.1272 | 0.0763 | 1.667x | 0.1295 | 0.0767 | 1.689x |
| tail-129-h56 | 0.1382 | 0.0781 | 1.770x | 0.1374 | 0.0823 | 1.669x |
| tail-257-h7 | 0.1372 | 0.0859 | 1.597x | 0.1392 | 0.0861 | 1.617x |

The four long-sequence cases have 3623 global prefix tokens, 510 tokens per
frame, five-frame chunks, a one-chunk window radius and both anchor modes.
Thus 37 frames have 22493 tokens and 107 frames have 58193 tokens. The suffix
386/514 denotes the original projection width, not the attention head size.
The 386 case uses strided Q/K/V; the 514 cases use contiguous Q/K and strided V.

FlashInfer wins 11/12 cases in each cache mode. The 128-token dense case
regresses by 9.4 microseconds warm and 10.4 microseconds cold. There is no
automatic baseline fallback. The largest relative L2 difference against the
PyTorch BF16 baseline is 0.003118; all 24 outputs pass the benchmark's
relative L2 < 0.005 and maximum absolute error < 0.02 checks.

## Correctness coverage

`pytest tests/attention/test_vdn_window.py --full`: **241 passed, 0 skipped**
with two visible SM120 GPUs. The independent reference explicitly enumerates
token-pair visibility and uses FP32 math with TF32 disabled.

| Area | Coverage |
|---|---|
| Mask semantics | Global prefix/suffix, no globals, 0/1/2/7/13 frames, frame/chunk windows, all four anchor modes, overlapping anchors counted once, clipped and irregular bounds, nonconsecutive equal KV unions |
| Tile tails | Sequence lengths 127/128/129, 255/256/257, 1023/1024/1025; head counts 1/7/56; dense, local and irregular masks |
| Numerics | Random BF16 input, uniform scores, constant V, sharply peaked scores, masked large-value sentinel, multiple positive scales |
| Long sequences | 22493 tokens x 56 heads and 58193 x 7 heads, projection widths 386/514; 12 query rows per shape checked against independent FP32 attention over all visible keys |
| Layout and output | Contiguous, 386/514 projection views, channel stride, transposed/broadcast views, misalignment, output aliases and surrounding output guards |
| Lifecycle | Reuse, geometry-changing replan, invalid replan, injected device-plan failure, workspace and metadata released, nondefault stream, wrong stream and actual CUDA graph rejection |
| Devices and contract | Noncurrent CUDA device, cross-device rejection, SM120 guard, BF16/D128/equal-head contract, non-tensors, invalid ranks, workspace alignment/dtype/size/alias, autograd rejection |
| Indexing | Metadata overflow rejection and actual pack/scatter past 2^31 elements (16 GiB allocation; requires 18 GiB free) |

The four long-sequence FP32 sampled comparisons have relative L2 errors
0.002290–0.002323 and maximum absolute error at most 0.000330. These are
sampled-row checks, not a full S-by-S FP32 comparison.

Trace signature/golden and VDN template consistency checks pass (4 tests),
as does the trace registry suite (4 tests). The new example also executed on
an SM120 GPU and generated the checked-in golden definition. The trace
records tensor shapes; replay still needs the original host `plan()` arguments.

Coverage was compared with the validation approaches in
[skip-softmax #4859](https://github.com/flashinfer-ai/flashinfer/pull/4859),
[BF16 VSA #5569](https://github.com/flashinfer-ai/flashinfer/pull/5569), and
[Sage #5442](https://github.com/flashinfer-ai/flashinfer/pull/5442).
GQA, LSE, causal/dropout options, batching, FP8 and quantization are not exposed
by this API and are not claimed as supported. Validation here is SM120 only;
the complete FlashInfer repository test suite was not run.

## Reproduction

From a built FlashInfer source checkout, use a PyTorch version providing
`torch.nn.attention.varlen`, Triton, and two SM120 devices for the full tests:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLASHINFER_CUDA_ARCH_LIST=12.0a MAX_JOBS=4
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/attention/test_vdn_window.py --full -q
python -m pytest tests/trace/test_vdn_window_trace.py \
    tests/trace/test_fi_trace_template_consistency.py -k vdn -q
python -m pytest tests/trace/test_template_registry.py -q
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_vdn_window.py \
    --frames 107 --heads 7 --output /tmp/vdn-warm.json
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_vdn_window.py \
    --frames 107 --heads 7 --cold-l2 --output /tmp/vdn-cold.json
```

To repeat all 24 parameter sets without overwriting the recorded data, run
this Python snippet from the repository root with one idle SM120 GPU visible:

```python
import json
from pathlib import Path
import subprocess
import sys

root = Path("benchmarks/results/vdn_sm120_bf16")
for record in sorted(root.glob("*/*.json")):
    parameters = json.loads(record.read_text())["parameters"]
    command = [sys.executable, "benchmarks/bench_vdn_window.py"]
    for name, value in parameters.items():
        if name == "output":
            value = str(Path("/tmp/vdn-reproduction") / record.relative_to(root))
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend([option, str(value)])
    subprocess.run(command, check=True)
```

## Compute Sanitizer

The initial memcheck run reported 34 `CUDA_ERROR_INVALID_VALUE` errors at
`cuGetProcAddress_v2`. A `pytest --collect-only` control reproduced exactly
the same 34 errors without running any test or VDN kernel. After that control,
device checks were run with `--report-api-errors no`. This disables CUDA API
error reporting; the device memory/synchronization/race checks remain enabled.
The limitation is part of these results, not silently suppressed.
Racecheck additionally filters kernel names to ``flashinfer``,
``_pack_query_value`` and ``_scatter_output`` so the independent PyTorch FP32
oracle is not instrumented. An initial unfiltered racecheck was interrupted
after 48 completed cases because of its instrumentation cost; it is not
counted as a passing run. Memcheck and synccheck do not use a kernel filter.

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
    python -m pytest tests/attention/test_vdn_window.py --collect-only -q

for tool in memcheck synccheck racecheck; do
    extra=()
    if [ "$tool" = racecheck ]; then
        extra=(--kernel-name 'regex=(_pack_query_value|_scatter_output|flashinfer)')
    fi
    compute-sanitizer --tool "$tool" --report-api-errors no \
        --error-exitcode 99 --target-processes all "${extra[@]}" \
        python -m pytest tests/attention/test_vdn_window.py --full -q \
        -k 'test_strides_and_reuse or test_out or test_replan or test_failed_device_replan or test_tile_boundaries or test_masked_value_sentinel'
done
```

The collection control is expected to exit 99 on the recorded toolchain.
Sanitizer results and test/source identities are recorded in `validation.json`.
The sanitizer subset excludes the multi-GPU and very large allocation tests;
those passed in the ordinary full test run.
