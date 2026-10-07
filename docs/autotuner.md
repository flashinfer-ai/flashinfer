# FlashInfer Autotuner

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASHINFER_AUTOTUNE_INDEPENDENT` | `"0"` | Set to `"1"` to enable independent per-rank profiling: each rank profiles with its own timings and handles its own OOM detection, skipping all collectives during the tuning loop. Use on homogeneous TP deployments where cold-start autotuning exceeds the distributed timeout. Leave at `"0"` when rank agreement on tactic choice is required (e.g., NCCL symmetric memory). See [#5898](https://github.com/flashinfer-ai/flashinfer/issues/5898). |
