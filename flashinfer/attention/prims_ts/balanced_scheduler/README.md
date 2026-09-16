# PrimsTS balanced scheduler

This package contains the replay-stable CUDA scheduler, its admission gate and
cost models, and the tools used to calibrate those models. Production balanced
MLA execution requires a checked-in calibration for the exact GPU product,
compute capability, and SM count. It deliberately fails instead of using a
model measured on another device.

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
