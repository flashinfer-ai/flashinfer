# PrimsTS balanced scheduler

This package contains the replay-stable CUDA scheduler, its admission gate and
cost models, and the tools used to calibrate those models. Production balanced
MLA execution requires a checked-in calibration for the exact GPU product,
compute capability, and SM count. It deliberately fails instead of using a
model measured on another device.

## Supported modes and ordinary fallback

Balanced execution retains its qualified dense D576 feature set: latent width
512, RoPE width 64, page sizes 16/32/64/128, matching BF16 or E4M3 Q/KV, BF16
output, and host scalar scales. Both 1CTA/2CTA and fixed/packed Q are supported.
For fixed Q the runtime query extent must equal the planned extent; packed
request lengths may vary within their planned bound.

Main's context parallelism, public LSE, attention tuning (including skip
correction), device scalar scales, D512-only geometry, smaller pages, and sparse
or quantized-cache APIs are not extensions of the balanced path. Sparse and
quantized caches continue to use their separate upstream APIs.
`plan_auto()` uses ordinary kernels for unsupported modes without consulting
the balanced admission gate or calibration registry. `plan_balanced()` rejects
them before policy compilation or allocation. A supported but uncalibrated
auto candidate still warns and falls back as described below. Ordinary planning
and execution keep main's interfaces and behavior, including their CP/LSE and
inactive-request contracts.

`plan_balanced()` and `plan_auto()` retain their existing
`max_seq_len_q`/`q_data_type`/`kv_data_type`/`o_data_type` arguments; ordinary
`plan()` uses main's `max_q_len`/`q_dtype`/`kv_dtype`/`out_dtype` names.
Runtime scale keywords for all routes follow main: `softmax_scale` and
`output_scale`. The calibration tool uses the corresponding internal planner
and runtime bindings.

The rebase preserves main's ordinary page-window loader and reducer; balanced
combine reduction is a separate entry point. The existing B200/B300 coefficients
remain registered, but the rebased kernels require fresh GPU correctness,
graph-replay, and performance validation before production use.

Checked-in calibrations currently cover these exact CUDA device identities:

| Product name | Compute capability | SM count |
| --- | --- | --- |
| NVIDIA B200 | 10.0 | 148 |
| NVIDIA B300 SXM6 AC | 10.3 | 148 |

An 80-SM device or a different product name is not covered by the B300
calibration. `plan_auto()` warns and uses ordinary kernels if the admission
gate requests balanced execution on an uncalibrated device; `plan_balanced()`
requires an exact registered calibration.

## Calibrating a device

Run the tuner from the repository root in an environment with FlashInfer's
development dependencies and the target GPU:

```bash
python -m flashinfer.attention.prims_ts.balanced_scheduler.tune_cost_model \
  --device cuda:0 \
  --output balanced_mla_cost_model.jsonl
```

The output is append-only. An interrupted run can be continued with
`--resume`. If only the fitting implementation changes, reuse the immutable
measurement cohort and append a new fit with:

```bash
python -m flashinfer.attention.prims_ts.balanced_scheduler.tune_cost_model \
  --device cuda:0 \
  --output balanced_mla_cost_model.jsonl \
  --resume \
  --refit-generation 1
```

`--quick` is only a harness smoke test. It does not produce a production
calibration. A production run should retain the default families, dtypes,
warmups, iterations, and trials unless the resulting change documents why a
different timing protocol is representative.

The artifact records hardware, software, and source identities. Inspect its
paired validation records before copying the fitted models and exact hardware
identity into `cost_model.py`. Then run the balanced scheduler and MLA decode
tests and the full fixed-seed performance sweep. Do not add an
architecture-wide or device-name-substring fallback.

The default workload cohort fits the six specialized buckets. Registering a
new device also requires measurements and paired validation for its `legacy`
bucket (high-batch sparse launches and moderately skewed small batches). Do
not copy B200's legacy row into a new device calibration without measurement.
The B300 calibration includes separate forced-target curves for B128 with
512 tokens per request and B8 with seven 2048-token requests plus one
8192-token request, alongside the default ragged-B128 cases. Both families and
dtypes have device-specific legacy rows. Split-target calibration does not
recalibrate the admission gate or the 1CTA/2CTA family boundary.

## Measuring scheduler latency

The scheduler-only and scheduler-plus-attention CUDA Graph timings are exposed
through:

```bash
python -m flashinfer.attention.prims_ts.balanced_scheduler.benchmark_scheduler \
  --device cuda:0 \
  --dtype bf16 \
  --distributions rl prod \
  --samples 100
```

The RL and production samples are deterministic for a given distribution,
batch size, and sample index. Use the same sample range when comparing code
revisions.

## Calibrating the 1CTA/2CTA family boundary

Family comparison is separate from split-target calibration. Construct one
`BalancedMLADecodePlan` per candidate family with an
`evaluation_cost=BalancedCostModel(...)` expressed in physical latency
units, then call `schedule_device(..., estimate_cost=True)`. The CUDA
scheduler first emits the schedule using its ordinary calibrated workload
bucket and only then scores that fixed placement with the evaluation model.
Thus experimental family coefficients cannot perturb either candidate's split
target or descriptor placement. `last_predicted_cost` contains the
critical-partition cost plus any modeled combine cost after synchronization.

Always validate a family rule with paired kernel timings on identical sequence
samples. On B200 H64, a device-score regression was less accurate than the
directly measured dtype/batch boundary, so production uses BF16 2CTA below B16,
BF16 1CTA from B16 onward, and FP8 1CTA from B4 onward. The score path remains opt-in
calibration telemetry and adds no placement atomics or cost evaluation to
normal graph replay.
