"""Run the unchanged benchmarks/bench_moe_deepseek.py for CuteDSL W4A16 only.

Mirrors PR 7290's observer: `_benchmark_single` receives profile_backend="cute-dsl-w4a16"
(selection only; no CUDA profiler capture), so only the W4A16 backend is tuned and timed with
the benchmark's own CUPTI/cold-L2/CUDA-graph method. Records every raw sample list and the
W4A16 tactic(s) that launch_w4a16_moe received per token count into $BENCH_RECORD (JSON).

Usage (from the source tree root, PYTHONPATH=$PWD):
  BENCH_RECORD=out.json python bench_w4a16_only.py <bench_moe_deepseek.py args>
"""

import json
import os
import sys

repo = os.getcwd()
sys.path.insert(0, os.path.join(repo, "benchmarks"))
sys.argv = ["bench_moe_deepseek.py"] + sys.argv[1:]

import bench_moe_deepseek as bm  # noqa: E402

import flashinfer  # noqa: E402
import flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 as w4a16  # noqa: E402
import flashinfer.fused_moe.cute_dsl.tuner as tuner  # noqa: E402
import flashinfer.testing.utils as testing_utils  # noqa: E402

record = {
    "argv": sys.argv,
    "flashinfer_file": flashinfer.__file__,
    "samples_ms": {},
    "tactics": {},
}
_current = {"n": None}

_orig_single = bm._benchmark_single


def _single(n, *args, **kwargs):
    if not kwargs.get("profile_cuda"):
        kwargs["profile_backend"] = "cute-dsl-w4a16"
    _current["n"] = n
    return _orig_single(n, *args, **kwargs)


bm._benchmark_single = _single

# The benchmark imports bench_gpu_time from flashinfer.testing.utils inside each call.
_orig_bench = testing_utils.bench_gpu_time


def _bench(*args, **kwargs):
    times = _orig_bench(*args, **kwargs)
    record["samples_ms"].setdefault(str(_current["n"]), []).append([float(t) for t in times])
    return times


testing_utils.bench_gpu_time = _bench


def _wrap_launch(module):
    original = module.launch_w4a16_moe

    def launch(*args, **kwargs):
        tactics = record["tactics"].setdefault(str(_current["n"]), [])
        tactic = repr(kwargs.get("tactic"))
        if tactic not in tactics:
            tactics.append(tactic)
        return original(*args, **kwargs)

    module.launch_w4a16_moe = launch


_wrap_launch(w4a16)
if hasattr(tuner, "launch_w4a16_moe"):
    _wrap_launch(tuner)

try:
    bm.main()
finally:
    with open(os.environ["BENCH_RECORD"], "w") as f:
        json.dump(record, f)
