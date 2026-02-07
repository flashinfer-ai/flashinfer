# HOW TO OPTIMIZE CUDA WITH NCU FOR ANY ARBITRARY KERNEL

This document is a practical workflow for optimizing CUDA kernels with Nsight Compute (`ncu`).
It is written to be reusable for arbitrary kernels, not tied to a specific project or operator.

## 1) Goal and Mindset

`ncu` is not just a profiler. It is a hypothesis testing tool.

Use this loop:

1. Make performance reproducible.
2. Profile one kernel (or one NVTX range) at a time.
3. Identify the dominant bottleneck class.
4. Apply one meaningful code change.
5. Re-profile and confirm metric movement.
6. Keep only changes that improve end-to-end time.

Do not optimize by trying random launch shapes or blind parameter sweeps. Use metrics to decide what to change.

## 2) Prerequisites

1. Install Nsight Compute (`ncu`) that matches your driver/CUDA stack.
2. Ensure kernel names or NVTX ranges are visible.
3. Build with line info for source correlation if possible.
4. Ensure input tensors and runtime environment are deterministic enough for comparison.

Recommended environment setup:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # if cublas determinism matters
```

If your project supports it, enable line info in JIT/compile flags (example):

```bash
export FLASHINFER_JIT_LINEINFO=1
```

## 3) FlashInfer NCU Integration (Codebase-Aligned)

This repository already has profiling hooks and conventions. Use them so results
are directly comparable with existing benchmarks and reports.

### 3.1 JIT + cache behavior in FlashInfer

FlashInfer JIT code is cached under:

1. Default: `~/.cache/flashinfer/...`
2. Overridden base: `FLASHINFER_WORKSPACE_BASE`

For clean A/B profiling after kernel rewrites, clear JIT cache:

```bash
rm -rf ~/.cache/flashinfer
```

Or programmatically:

```bash
python - <<'PY'
from flashinfer.jit import clear_cache_dir
clear_cache_dir()
PY
```

Enable line correlation in generated kernels:

```bash
export FLASHINFER_JIT_LINEINFO=1
```

### 3.2 Use benchmark `--ncu-single` mode

In this repo, many benchmark scripts provide an `--ncu-single` path that:

1. Runs warmup outside the profile range.
2. Calls `cudaProfilerStart/Stop`.
3. Wraps one invocation in an NVTX range.

Example (RMSNorm + MXFP8):

```bash
ncu \
  --set full \
  --target-processes all \
  --profile-from-start off \
  --nvtx --nvtx-include "rmsnorm_mxfp8]" \
  -o ncu_rmsnorm_mxfp8 \
  python benchmarks/bench_cute_dsl_rmsnorm_mxfp8_quantize.py \
    --ncu-single --batch-size 4096 --hidden-size 4096
```

### 3.3 FlashInfer-specific A/B knobs

For policy and dispatch comparisons, use env overrides already supported by kernels.

Example:

```bash
# Default dispatch
unset FLASHINFER_RMSNORM_MXFP8_CLUSTER_N
ncu --set full --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "rmsnorm_mxfp8]" \
  -o ncu_auto_b192_h4096 \
  python benchmarks/bench_cute_dsl_rmsnorm_mxfp8_quantize.py \
    --ncu-single --batch-size 192 --hidden-size 4096

# Forced cluster split
FLASHINFER_RMSNORM_MXFP8_CLUSTER_N=8 \
ncu --set full --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "rmsnorm_mxfp8]" \
  -o ncu_c8_b192_h4096 \
  python benchmarks/bench_cute_dsl_rmsnorm_mxfp8_quantize.py \
    --ncu-single --batch-size 192 --hidden-size 4096
```

### 3.4 How to interpret FlashInfer NVTX ranges

Inside one NVTX range, you may see:

1. A single fused kernel.
2. Multiple kernels for fallback paths.
3. Helper kernels (for example explicit fill/padding kernels).

Always optimize range-level duration first. If a helper kernel contributes
non-trivial time, consider folding that work into the main kernel when safe.

## 4) Build a Reproducible Benchmark Harness

Before `ncu`, prepare a benchmark entrypoint that:

1. Runs warmup iterations.
2. Measures median kernel latency (or range latency).
3. Can run one shape/config at a time.
4. Optionally wraps target execution in NVTX push/pop.

For `ncu`, the most reliable pattern is a single invocation inside NVTX:

```python
import torch.cuda.nvtx as nvtx
torch.cuda.cudart().cudaProfilerStart()
nvtx.range_push("target_kernel")
run_target_once()
torch.cuda.synchronize()
nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
```

## 5) First Profile: Fast Triage

Start with a lighter set to classify bottlenecks quickly:

```bash
ncu \
  --set basic \
  --target-processes all \
  --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_triage \
  python your_benchmark.py --ncu-single --shape ...
```

Then inspect:

```bash
ncu --import ncu_triage.ncu-rep --print-summary per-kernel
```

Look at:

1. `Duration`
2. `Compute (SM) Throughput`
3. `DRAM Throughput`
4. Scheduler stats (`Eligible Warps`, `Issued Warp`)

For a more detailed triage that includes memory analysis and roofline but stays smaller than `--set full`:

```bash
ncu \
  --set detailed \
  --target-processes all \
  --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_detailed \
  python your_benchmark.py --ncu-single --shape ...
```

## 6) Deep Profile: Full Diagnostic Pass

After triage, run full detail only on relevant shapes:

```bash
ncu \
  --set full \
  --target-processes all \
  --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_full \
  python your_benchmark.py --ncu-single --shape ...
```

### 6.1 Source & SASS Correlation

For code-level bottleneck identification, collect with source info and view:

```bash
# Collect with source correlation (build must include -lineinfo)
ncu --set full --import-source=on \
  --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_src \
  python your_benchmark.py --ncu-single --shape ...

# View source-correlated hotspots
ncu --import ncu_src.ncu-rep --page source

# Omit source info to shrink report size when not needed
ncu --set full --no-source-info ...
```

### 6.2 Export and Filtering

```bash
# Export full CSV (large output — grep selectively)
ncu --import ncu_full.ncu-rep --page raw --csv > ncu_full.csv

# Export per-kernel summary as CSV
ncu --import ncu_full.ncu-rep --print-summary per-kernel --csv > ncu_summary.csv

# View session metadata (device, driver, CUDA version)
ncu --import ncu_full.ncu-rep --page session
```

## 7) Bottleneck Classification Cheat Sheet

Use these metrics together. One metric alone is often misleading.

### A) Memory-bandwidth bound

Typical signs:

1. `dram__throughput.*pct_of_peak*` is high.
2. SM throughput is moderate/low relative to DRAM.
3. Scheduler has eligible warps but progress is memory-limited.
4. Poor coalescing: `dram__sectors.avg.pct_of_peak_sustained_active` low relative to bytes moved.
5. Low cache hit rates: `l1tex__t_hit_rate.pct` or `lts__t_hit_rate.pct` are low.

Likely actions:

1. Improve global memory coalescing.
2. Vectorize loads/stores.
3. Reduce bytes moved (fusion, data reuse, narrower types).
4. Improve cache locality or shared-memory staging.

### B) Latency/occupancy/scheduling bound

Typical signs:

1. DRAM throughput is low.
2. SM throughput is low.
3. `smsp__warps_eligible.avg.per_cycle_active` is low.
4. `No Eligible` stall fraction is high.
5. Dominant stall: `smsp__warps_issue_stalled_long_scoreboard` (waiting on memory) or `smsp__warps_issue_stalled_no_instructions` (instruction cache / branch overhead).

Likely actions:

1. Increase parallel work per launch (more CTAs or better mapping).
2. Reduce sync/barrier overhead.
3. Reduce serial sections and long dependency chains.
4. Use moderate cluster/tiling choices that improve eligibility.

### C) Compute/instruction bound

Typical signs:

1. SM throughput is high.
2. DRAM is not saturated.
3. Issue slots are busy; math pipes dominate.

Likely actions:

1. Use more efficient math instructions.
2. Reduce unnecessary conversions/precision churn.
3. Improve instruction-level parallelism and unrolling balance.

### D) Register pressure / spilling bound

Typical signs:

1. Nonzero local memory spilling requests.
2. Occupancy constrained by registers.
3. Increased local memory traffic.

Likely actions:

1. Reduce live range and temporary count.
2. Split kernel phases or simplify inner loops.
3. Tune unroll factors.

## 8) Metric-to-Action Mapping (Quick Reference)

1. Low `smsp__warps_eligible`, low DRAM, low SM:
   action: improve scheduling/parallelism, reduce launch overhead, reduce barriers.
2. High DRAM throughput and high memory stalls:
   action: reduce bytes and improve coalescing/cache reuse.
3. High local spill requests:
   action: reduce register pressure and simplify kernel body.
4. Extra tiny kernels in NVTX range:
   action: fold side work into main kernel if safe (for example padding writes).
5. Good kernel metric but poor end-to-end time:
   action: optimize the whole range, not just a single kernel.
6. High `smsp__warps_issue_stalled_long_scoreboard`:
   action: memory latency dominates — improve prefetching, coalescing, or cache reuse.
7. High `smsp__warps_issue_stalled_no_instructions`:
   action: instruction cache misses or excessive branching — simplify control flow, reduce code size.
8. Low `l1tex__t_hit_rate.pct` with high L1 traffic:
   action: improve spatial locality, use shared memory, or restructure access patterns.

## 9) NCU Quick Commands & Views Cheat Sheet

### 9.1 Available `--set` Options

List all sets: `ncu --list-sets`

| Set | Sections | Use Case |
|-----|----------|----------|
| `basic` | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | Fast first triage (~213 metrics) |
| `detailed` | Adds MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, SourceCounters, Roofline | Good middle ground (~906 metrics) |
| `full` | All sections including WarpStateStats, SchedulerStats, InstructionStats, MemoryTables | Complete diagnostic (~7794 metrics) |
| `roofline` | SpeedOfLight + all Roofline charts + WorkloadDistribution | Quick bound identification (~6649 metrics) |
| `pmsampling` | PmSampling, PmSampling_WarpStates | Warp state sampling |
| `nvlink` | Nvlink, Nvlink_Tables, Nvlink_Topology | Multi-GPU link analysis |

### 9.2 Available `--page` Options

| Page | Description |
|------|-------------|
| `details` | Sections and rules (default) |
| `raw` | All collected metrics (use with `--csv`) |
| `source` | Source code correlation (needs `-lineinfo` build) |
| `session` | Session and device attributes |

### 9.3 Section Identifiers (for `--section` Filtering)

List all sections: `ncu --list-sections`

Filter report output to specific sections using `--section` (can be repeated):

```bash
# Memory + scheduler + throughput overview
ncu --import ncu_full.ncu-rep --page details \
  --section MemoryWorkloadAnalysis --section SchedulerStats --section SpeedOfLight

# Warp stall analysis (very useful when latency bound)
ncu --import ncu_full.ncu-rep --page details --section WarpStateStats

# Memory access tables (coalescing, sector utilization, L1/L2 details)
ncu --import ncu_full.ncu-rep --page details --section MemoryWorkloadAnalysis_Tables

# Instruction mix breakdown
ncu --import ncu_full.ncu-rep --page details --section InstructionStats

# Launch config + occupancy
ncu --import ncu_full.ncu-rep --page details --section LaunchStats --section Occupancy

# Source-level counters (needs --set detailed or full)
ncu --import ncu_full.ncu-rep --page details --section SourceCounters
```

Common section identifiers:

| Identifier | What It Shows |
|------------|---------------|
| `SpeedOfLight` | SM and memory throughput % of peak |
| `SpeedOfLight_RooflineChart` | Roofline chart |
| `MemoryWorkloadAnalysis` | Memory throughput breakdown |
| `MemoryWorkloadAnalysis_Tables` | Coalescing, sector utilization, L1/L2 detail tables |
| `ComputeWorkloadAnalysis` | Compute pipe utilization |
| `SchedulerStats` | Warp eligibility, issue rate, active warps |
| `WarpStateStats` | Warp stall reasons breakdown |
| `InstructionStats` | Instruction mix and count |
| `LaunchStats` | Grid/block size, registers, shared memory |
| `Occupancy` | Theoretical vs achieved occupancy |
| `SourceCounters` | Source-line-level hotspot counters |

### 9.4 Single-Metric and Regex Queries (Fast Feedback)

Query specific metrics from an existing report without reading the full output:

```bash
# Single metrics
ncu --import ncu_full.ncu-rep --metrics dram__throughput.avg.pct_of_peak_sustained_active
ncu --import ncu_full.ncu-rep --metrics sm__throughput.avg.pct_of_peak_sustained_active
ncu --import ncu_full.ncu-rep --metrics smsp__warps_eligible.avg.per_cycle_active

# Regex groups — match multiple metrics at once
ncu --import ncu_full.ncu-rep --metrics "regex:.*throughput.*"
ncu --import ncu_full.ncu-rep --metrics "regex:smsp__warps_issue_stalled.*"
ncu --import ncu_full.ncu-rep --metrics "regex:.*hit_rate.*"
```

### 9.5 Key Metrics Reference

**Throughput & utilization:**

| Metric | Meaning |
|--------|---------|
| `dram__throughput.avg.pct_of_peak_sustained_active` | DRAM bandwidth utilization |
| `sm__throughput.avg.pct_of_peak_sustained_active` | SM compute utilization |
| `l1tex__t_sectors.avg.pct_of_peak_sustained_active` | L1 cache throughput |
| `lts__t_sectors.avg.pct_of_peak_sustained_active` | L2 cache throughput |

**Cache hit rates:**

| Metric | Meaning |
|--------|---------|
| `l1tex__t_hit_rate.pct` | L1 cache hit rate |
| `lts__t_hit_rate.pct` | L2 cache hit rate |

**Warp scheduling:**

| Metric | Meaning |
|--------|---------|
| `smsp__warps_eligible.avg.per_cycle_active` | Eligible warps per cycle (higher = better) |
| `smsp__warps_issue_stalled_long_scoreboard.avg.pct` | % stalled waiting on memory |
| `smsp__warps_issue_stalled_no_instructions.avg.pct` | % stalled on instruction fetch |
| `smsp__warps_issue_stalled_wait.avg.pct` | % stalled on fixed latency |
| `smsp__warps_issue_stalled_barrier.avg.pct` | % stalled on barriers/sync |

**Memory coalescing (from `--set full`):**

| Metric | Meaning |
|--------|---------|
| `dram__sectors.avg.pct_of_peak_sustained_active` | DRAM sector utilization (low = poor coalescing) |
| `lts__t_sectors.avg.pct_of_peak_sustained_active` | L2 sector utilization |

## 10) Advanced NVTX & Kernel Filtering

### 10.1 NVTX Patterns

```bash
# Multiple NVTX ranges (OR match)
--nvtx --nvtx-include "rmsnorm_mxfp8|attention|mlp"

# Exclude noise ranges
--nvtx --nvtx-exclude "cudaMalloc|cudaMemcpy|warmup|memset"

# Combine include + exclude
--nvtx --nvtx-include "critical_path" --nvtx-exclude "warmup"
```

### 10.2 Kernel Name Filtering

```bash
# Filter by demangled kernel name (regex)
--kernel-name-base demangled --kernel-name "regex:rmsnorm.*mxfp8"

# Only profile one invocation per kernel (reduces replay overhead)
--kernel-id :::1
```

### 10.3 Report Size Control

Full reports can grow to gigabytes. Useful flags to keep size manageable:

```bash
# Use detailed instead of full (much smaller, still very useful)
--set detailed

# Skip source info collection (saves significant space)
--no-source-info

# Only collect for a specific kernel
--kernel-name-base demangled --kernel-name "my_kernel"

# Force metrics even on very short kernels
--force-metrics-collection

# Multi-process output with unique filenames
--target-processes all -o report.%p
```

## 11) Comparing Two Runs (A/B Diff)

Very useful for validating optimizations:

```bash
# CLI-based diff between two reports
ncu --import baseline.ncu-rep --import new.ncu-rep --page details --diff

# Or open both in GUI for interactive comparison
ncu-ui baseline.ncu-rep new.ncu-rep
```

Pair this with the A/B workflow in Section 3.3 — collect baseline and variant reports,
then diff to see exactly which metrics moved and by how much.

## 12) Practical Optimization Workflow

For each candidate kernel rewrite:

1. Record baseline latency and key metrics.
2. Implement one structural change.
3. Re-run correctness checks.
4. Re-run benchmark (same shapes, same iteration counts).
5. Re-run `ncu` on representative shapes.
6. Compare delta in:
   - range duration
   - kernel duration
   - SM throughput
   - eligible warps
   - DRAM throughput
   - spill metrics
   - warp stall reasons
7. Use `ncu --diff` to compare reports directly (Section 11).
8. Keep or revert.

## 13) Shape Selection Strategy

Do not profile every shape initially.

Pick:

1. One known regression shape.
2. One median production shape.
3. One large throughput-oriented shape.
4. One edge shape (small batch, large hidden, or irregular dimensions).

After fixing regressions, expand the sweep.

## 14) NVTX and Multi-Kernel Ranges

If your operation emits multiple kernels:

1. Profile the full NVTX range.
2. Sum kernel durations in the range.
3. Identify helper kernels (memset, transpose, format conversion).
4. Remove/merge helper kernels when possible.

This frequently yields larger gains than micro-optimizing one already-fast kernel.

## 15) Guardrails for Reliable Conclusions

1. Use median over multiple repeats.
2. Keep clocks and power mode consistent if possible.
3. Avoid comparing runs with different warmup behavior.
4. Compare with and without autotuner cache effects explicitly.
5. Validate correctness at each major rewrite.

## 16) Example Command Set (Reusable Template)

Replace placeholders with your script and NVTX range.

```bash
# 1) Baseline benchmark sweep
python your_benchmark.py --dry-run-iters 10 --repeat-iters 100

# 2) Fast triage (basic set — smallest, fastest)
ncu --set basic --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_triage \
  python your_benchmark.py --ncu-single --shape ...

# 3) Medium triage (detailed set — adds memory analysis + roofline)
ncu --set detailed --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_detailed \
  python your_benchmark.py --ncu-single --shape ...

# 4) Full NCU (all sections — use for deep investigation)
ncu --set full --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_full \
  python your_benchmark.py --ncu-single --shape ...

# 5) Roofline-focused (quick bound identification)
ncu --set roofline --target-processes all --profile-from-start off \
  --nvtx --nvtx-include "target_kernel]" \
  -o ncu_roofline \
  python your_benchmark.py --ncu-single --shape ...

# 6) Report inspection
ncu --import ncu_full.ncu-rep --print-summary per-kernel
ncu --import ncu_full.ncu-rep --print-summary per-kernel --csv > summary.csv

# 7) Targeted section views
ncu --import ncu_full.ncu-rep --page details --section SchedulerStats --section WarpStateStats
ncu --import ncu_full.ncu-rep --page details --section MemoryWorkloadAnalysis_Tables

# 8) Quick single-metric checks
ncu --import ncu_full.ncu-rep --metrics "regex:.*throughput.*"
ncu --import ncu_full.ncu-rep --metrics "regex:smsp__warps_issue_stalled.*"

# 9) A/B comparison
ncu --import ncu_baseline.ncu-rep --import ncu_new.ncu-rep --page details --diff

# 10) Full CSV export (large — grep selectively)
ncu --import ncu_full.ncu-rep --page raw --csv > ncu_full.csv
```

## 17) Common Mistakes

1. Optimizing kernel metrics while end-to-end time gets worse.
2. Changing too many things at once and losing causality.
3. Interpreting high occupancy as guaranteed performance.
4. Ignoring helper kernels in the same range.
5. Overfitting one shape and regressing production distributions.
6. Using `--set full` for every run (wastes time and disk — start with `basic` or `detailed`).
7. Not using `--diff` for A/B comparisons and relying on eyeballing instead.

## 18) Definition of Done

An optimization is done only when all are true:

1. Correctness is preserved.
2. Target regression shapes are fixed.
3. Geomean (or weighted production mix) improves.
4. NCU metrics support the observed latency gain.
5. The change is maintainable and not just parameter spam.

---

If you follow this document literally, you will have a defensible and repeatable path from "kernel is slow" to "kernel is measurably better and we know why."
