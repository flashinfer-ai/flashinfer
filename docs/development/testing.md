# Testing and resource scheduling

Use the targeted commands in [AGENTS.md](../../AGENTS.md) for routine changes.
This reference covers:

- [CPU-only PR lane](#cpu-only-pr-lane)
- [Runner scheduling](#sharded-test-runner)
- [Resource telemetry](#resource-telemetry)
- [Collection and completion reports](#collection-and-completion-reports)
- [MXFP8 startup benchmark](#mxfp8-startup-benchmark)
- [Architecture guards](#architecture-guards)

## CPU-only PR lane

Every public run that executes tests also runs all of `tests/cpu` in the
"CPU Unittest" lane on a CPU runner, regardless of selected targets. These tests
must not require a GPU; GPU lanes exclude them. Docs-only PRs run no tests, and
`workflow_dispatch` with `skip_gpu` also drops the CPU lane.

## Sharded test runner

Run `pytest --confcutdir=tests/test_sharding tests/test_sharding/` for the
CPU-only coordinator suite.

The default is one worker per visible GPU. H100 PR CI uses
`--workers-per-gpu 2` (eight workers on four GPUs); other lanes keep the default.
Locally, use `scripts/task_run_unit_tests.sh --workers-per-gpu 2` for sharing or
`--workers N` for a total count. Workers share the JIT compilation budget.
The script prepares a CI container: it installs the Python requirements and the
local package, and runs `apt-get` when `jq` is missing. Run it in a disposable
container or environment, not in a development environment you want unchanged.

### Scheduling and memory

Scheduling markers apply to the whole source file:

- `node_exclusive`: run alone across the allocation after parallel work, with
  all GPUs visible. Use for multi-GPU requirements or host-memory OOM risk.
- `gpu_exclusive`: run alone on the assigned GPU; other GPUs can keep working.
- `host_memory_heavy`: prioritize this file and serialize its units with other
  host-heavy work, while lighter files can overlap.

Host/GPU reservations are acquired together and cover setup through cleanup.
Blocked units stay queued; workers can run other work or steal from another
queue. GPU-exclusive work can drain one GPU without holding the host-heavy slot
or blocking other GPUs. `RESOURCE WAIT` reports idle periods of at least one
second when queued work cannot use that worker's resources.

Markers control overlap, not memory limits. Most H100 annotations reflect
peaks above 128 GiB host RSS or 32 GiB GPU memory. Attention sink uses both
narrower markers: 89.2 GiB host RSS and 4.5 GiB GPU on a single-GPU H100 run,
allowing other GPUs to work. Cold JIT and other GPUs can change these peaks.
For concurrency changes, compare completion rate, OOM failures, and repeated
wall times with matching scope and cache conditions.

Capacity reports include the host-heavy/node-exclusive serial floor, but use
worker counts without topology or measured memory capacity. Treat their GPU
contention estimates as optimistic.

## Resource telemetry

Allocation telemetry defaults to a 5-second interval (`RESOURCE_MONITOR_INTERVAL`);
per-batch sampling defaults to 2 seconds (`MEMORY_MONITOR_INTERVAL`).
`resource_monitor.py` separates sampling, CSV persistence, and aggregation;
allocation/batch monitors share `scripts/test_sharding/host_cpu.py` readers.
Unavailable per-process GPU memory stays unknown and marks source summaries
partial; `memory_summary.csv` leaves maxima blank without valid observations.
Host rows flush before GPU queries, whose costs appear in `sampler.csv`.
Nested runners reuse samplers only while the exported owner PID lives.
Attribute namespace-relative cgroup paths only with membership evidence.

## Collection and completion reports

Public CI uploads collection stdout/stderr and best-effort JIT prewarm warnings
from `junit/collection-*/*.log`. Separate directories preserve concurrent and
previous attempts' diagnostics; collection metadata remains temporary.

Set `UNIT_TEST_RUN_STATUS_DIR` to a fresh directory unique to each
communication-test wrapper and container/rank to enable completion reports.
`test_run_status.sh` records pytest invocations; `test_run_status.py` atomically
writes `script-result.json` with schema version 1, `complete`, `exit_code`,
`tests_executed`, and `infrastructure_errors`. A pytest plugin records execution,
collection, and internal errors inside each MPI/torchrun worker. Launcher exit
1 alone does not prove an assertion failure.

Cache publication requires every local worker rank to finish and at least one
test to execute. Completed assertion failures remain eligible; crashes, timeouts,
and unfinished invocations do not. Unsetting the variable disables reports,
while missing JUnit output and file timeouts still fail the wrapper. The CPU
coordinator suite covers this contract without GPUs.

## MXFP8 startup benchmark

Run `python benchmarks/bench_mxfp8_startup.py --output-dir /path/to/new/results` on
B200/B300 to compare persistence disabled, serial/parallel disk-cache population,
fresh-process disk hits, and saved tuning winners. The benchmark isolates caches
and checks results after CUDA graph replay. Default shapes exercise the
SM100-family persistent/split-K runner; see
[cache scope and layers](../design_docs/cute_dsl_kernel_cache.md).

MXFP8 precompile workers use a dedicated interpreter entry point. They do not
rerun application scripts or require callers to add a `__main__` guard.

## Architecture guards

FlashInfer supports NVIDIA SM75, SM80, SM86, SM89, SM90, SM100, SM103, SM110,
SM120, and SM121. Support varies by operation and backend.

Use `flashinfer.utils.get_compute_capability(device)` for `(major, minor)`
checks and the feature helpers for architecture-specific kernels:
`is_sm90a_supported`, `is_sm100a_supported`, `is_sm100f_supported`,
`is_sm110a_supported`, `is_sm120a_supported`, `is_sm120f_supported`, and
`is_sm121a_supported`.

Common architecture checks for specific kernel paths:

| Kernel path | Architecture requirement | Guard |
| --- | --- | --- |
| FlashAttention-3 | SM90a, CUDA 12.3+ | `is_sm90a_supported(device)` |
| SM100a MLA attention | SM100a, CUDA 12.8+ | `is_sm100a_supported(device)` |
| FP8 GEMM hardware support | SM89+ | `get_compute_capability(device) >= (8, 9)` |

These checks do not replace an API's backend-support checks or its dtype and
shape requirements.

For example, guard a Hopper-specific test with:

```python
import pytest
import torch

from flashinfer.utils import is_sm90a_supported


def test_hopper_attention():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")
    # Test the operation here.
```

APIs decorated with `@backend_requirement` expose
`is_compute_capability_supported(cc)` and `is_backend_supported(backend, cc)`.
`cc` is an integer, `major * 10 + minor` (e.g. `90`), not the tuple returned by
`get_compute_capability`. Without `cc`, `is_backend_supported` only reports
whether the backend name is registered, not whether it runs on the current GPU;
pass `cc` in architecture guards. Use these checks rather than assuming all
backends support an architecture. Self-validating modes that
replace backend selection, such as `batch_invariant=True` in `mm_bf16`,
`bmm_fp8`, `mm_mxfp8`, and `mm_fp4`, use the decorator's `bypass=` predicate;
see the kernel tutorial.

Choose problem sizes that avoid OOM; the OOM handling in `tests/conftest.py`
is a fallback, not a sizing policy.
