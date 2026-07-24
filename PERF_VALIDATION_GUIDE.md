# Autotuner v2 — perf validation guide

**Pinned version**: tag `perf-val-0710` (= `a19a84c4`) on
`github.com/YangXu1990uiuc/flashinfer`, branch `yanxu/autotune-cache-v2-mvp`
(PR flashinfer-ai/flashinfer#3861, design RFC #3920).
CI is green on this head (H100/A10G/T4 JIT suites, AOT cu128/129/130).

## Setup

```bash
git clone -b perf-val-0710 https://github.com/YangXu1990uiuc/flashinfer.git
cd flashinfer && git submodule update --init --recursive
pip install --no-build-isolation -e . -v
pip install -U cupti-python          # for the cupti measurement policy
```

Use an **all-release stack** (pip torch + the cuDNN/cuBLAS wheels it pulls;
no local debug builds, no `FLASHINFER_JIT_DEBUG`). Verify what actually
loaded via `python -m flashinfer.collect_env`.

Cache location: `FLASHINFER_AUTOTUNE_CACHE_DIR=<dir>` (defaults under
`~/.cache/flashinfer/autotune`). Wipe it between runs that must tune fresh.

## What to validate (in order of value)

### 1. Selection quality (the headline claim)

```bash
python benchmarks/bench_autotuner_accuracy.py \
    --m-list 1 2 4 8 16 32 64 1024 --seeds 3 \
    --out accuracy_<gpu>.json
```

~2 min/op on B200. It sweeps every candidate under two deployment oracles
(interleaved rounds, SM clock recorded per round) and simulates three tuning
policies. **Reference numbers (production B200 183GB/1000W, bmm_fp8
N=K=8192)** — median regret across shapes, worst in parens:

| tuned with \ deployed as | cuda-graph-like | eager |
|---|---|---|
| v1 events+delay | 0.0% (9.8%) | 15.0% (1910%) |
| v2 cupti | 0.0% (0.8%) | 12.7% (1967%) |
| v2 eager | 15.9% (41.8%) | 0.0% (5.2%) |

Expected findings: diagonal ≈ 0; v2-cupti's worst-case beats v1's on the
span oracle; SM clocks flat on production boards. Deviations worth reporting:
any v2 regret > ~2% on its own deployment, clock instability, oracle
disagreement on shapes where both policies pick the same backend.

Multiple ops: `--ops bmm_fp8 mm_fp4 cutlass_moe_fp8` (composite MoE calls
sweep each internal GEMM). `--max-candidates N` subsamples large candidate
sets (MoE has ~324) — coverage is then PARTIAL and logged as such.

**CRITICAL — check the clock before trusting any number.** Each row records
`sm_clocks_mhz` per oracle round. Heavy ops (MoE, large-M GEMM) can drive a
throttle-limited board's SM clock down mid-sweep (an engineering-sample
SM100 collapsed 1965→120 MHz on the 64-expert MoE), and the oracle then
measures across a moving clock — manufacturing tens-of-percent phantom
regret that looks like a tuner defect. If `min(clocks)/max(clocks) < ~0.9`
for a row, discard that row: the board is throttling, not the tuner
mis-selecting. On a production B200 the same MoE holds 1762–1965 MHz and
regret is ~0%. Heavy ops on throttle-limited boards are not benchmarkable.

### 2. API behavior / overhead

```python
import flashinfer
from flashinfer import autotune_v2, MeasurementPolicy, autotune_v2_reload

with autotune_v2():                      # tune + publish (per-entry, atomic)
    run_dummy_workload()
# serving needs NO context; entries persist and reload in later processes
```

Check: warm-restart startup delta (second process with a populated store
must skip profiling — look for `source=managed cache` INFO logs and zero
`Autotuning process starts`); hot-path overhead (bare-call latency with
store attached vs plain v1 — expect none: lookups are memoized in memory);
concurrent multi-process tuning into one store (safe by design — validated
4-way; report any `.tmp` litter or corrupt entries).

### 3. e2e A/B (optional this round)

vLLM/SGLang integration patches are drafts (PR thread, "framework
patches" comment) — e2e framework A/B can wait for those to land. If you
do wire it manually, call `autotune_v2_reload()` after a post-tuning
barrier in multi-rank setups.

## Known boundaries (don't burn time rediscovering these)

- **MoE ops**: probe-realism issues (#3622/#3537 — routing distribution,
  EP-sharded M) are op-level and NOT fixed by this PR; sibling PRs coming.
  Validate GEMM-family ops first; MoE autotune regressions at high
  concurrency are a known, separately-tracked class.
- **`measure=` default is `"auto"`** (= today's v1 measurement). The
  cupti/eager policies are the new capability — test them explicitly.
  `execution_mode="cuda_graph"` forces capture: only for ops whose runners
  are capture-safe (bmm_fp8 backends are).
- Harness coverage is currently one op (`bmm_fp8`); per-op extension is a
  separate track — extra ops you wire up are welcome data.
- Tactic values changed representation upstream (#3707 structured tuples);
  cross-version cache reuse is intentionally impossible (env-hash miss).

## Reporting

Attach the harness JSON (it embeds GPU, clocks, per-shape detail) + the
`collect_env` dump. Anomalies → PR #3861 thread or directly to Yang Xu.
