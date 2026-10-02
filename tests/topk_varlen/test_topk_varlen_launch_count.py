"""Guard rail: one top_k_varlen call must cost exactly the device launches its
backend documents -- no wrapper-side tensor ops that become extra CUDA-graph
nodes.  Motivation (2026-09-29): ``backend="sglang"`` issued two small torch
ops per call (``seq_lens // cr``, ``clamp``) that cost ~3.4 us as graph nodes,
doubling the 3.3 us kernel; ``radix_cutlass`` pays three such ops today.

Eager launches map one-to-one onto graph nodes, so the count is taken with
torch.profiler over one eager call after two warm-ups (JIT, cached buffers).
The counting runs in fresh interpreters, one per backend: torch.profiler
records no CUDA activities in a process where an earlier CPU-only profiler
session already ran (test_radix_filter.py has one), and a process that opens
many CUDA profiler sessions in a row can lose the later ones' events (a B200
CI runner recorded the probe and five backends, then nothing for the last two
in the same process).  Two sessions per process -- the probe and the backend
-- keep both away.  An empty record can never be a code path (every backend
launches at least its kernel), so it is reported as a profiler artifact and
skipped; more launches than documented remain a failure.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

import flashinfer

# backend -> device activities (kernels + memsets/memcpys) per call
EXPECTED = {
    "sglang": 1,
    "radix_primitives": 1,
    "walkfirst_primitives": 1,
    "cutlass_primitives": 1,
    "radix": 1,
    "radix_filter": 1,
    "gvr_2": 1,
}

# Runs in the child interpreter: one eager call per backend after two
# warm-ups, printed as {backend: [device activity names]}; a dict entry
# {"error": <exception type name>, "msg": <text>} means the call itself
# failed (the parent decides whether that is a skip or a failure).
_COUNT_SCRIPT = r"""
import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import flashinfer


def device_launches(fn):
    fn()
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


rows, n, k = 16, 16384, 512
torch.manual_seed(1)
logits = torch.randn(rows, n, device="cuda")
seq_lens = torch.randint(600, 2100, (rows,), dtype=torch.int32, device="cuda")
out = torch.empty(rows, k, dtype=torch.int32, device="cuda")
pre_idx = torch.arange(k, dtype=torch.int32, device="cuda").repeat(rows, 1)
# the profiler must see a plain kernel in this process before any count counts
result = {"__probe__": device_launches(lambda: torch.add(logits, 1.0))}
for backend in sys.argv[1:]:
    kw = {"backend": backend, "out_indices": out}
    if backend == "gvr_2":
        kw["pre_idx"] = pre_idx
    try:
        result[backend] = device_launches(
            lambda: flashinfer.top_k_varlen(logits, seq_lens, k, **kw)
        )
    except Exception as e:  # noqa: BLE001
        result[backend] = {"error": type(e).__name__, "msg": str(e)[:400]}
print("LAUNCHES " + json.dumps(result))
"""

# the parent's own markers for records that say nothing about launch counts
_PROFILER_UNAVAILABLE = "ProfilerUnavailable"


def _supported_backends():
    from flashinfer.utils import get_compute_capability

    major, minor = get_compute_capability(torch.device("cuda"))
    cc = major * 10 + minor
    supported = [
        b
        for b in sorted(EXPECTED)
        if flashinfer.top_k_varlen.is_backend_supported(b, cc)
    ]
    if "radix_filter" in supported:
        from flashinfer.topk_varlen.topk_varlen import _radix_filter_kernel_dsl_ok

        if not _radix_filter_kernel_dsl_ok():
            supported.remove("radix_filter")
    return supported


def _count_in_fresh_interpreter(backend, env):
    """{backend: [names] | {"error", "msg"}} measured in a child that imports
    this same flashinfer tree and opens exactly two profiler sessions."""
    proc = subprocess.run(
        [sys.executable, "-c", _COUNT_SCRIPT, backend],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("LAUNCHES ")]
    assert proc.returncode == 0 and lines, (
        f"launch-count child for {backend} failed (rc={proc.returncode}):\n"
        f"{proc.stderr[-2000:]}"
    )
    result = json.loads(lines[-1][len("LAUNCHES ") :])
    if not result["__probe__"]:
        # The probe is a plain torch.add: when even that records nothing,
        # CUPTI activity tracing is unavailable on this runner (profiling
        # restricted to administrators, or no CUPTI for this toolkit).  Seen
        # on a GB300 CI node with the cu129 toolkit while its cu130/cu134
        # siblings and every B200 job recorded normally.
        return {
            "error": _PROFILER_UNAVAILABLE,
            "msg": "torch.profiler recorded no CUDA activity for a plain kernel "
            "in a fresh interpreter (CUPTI unavailable or profiling restricted)",
        }
    return result[backend]


@pytest.fixture(scope="module")
def launches():
    """{backend: [activity names] or {"error", "msg"}}, one fresh interpreter
    per supported backend."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    env = dict(os.environ)
    tree = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    env["PYTHONPATH"] = tree + os.pathsep + env.get("PYTHONPATH", "")
    return {b: _count_in_fresh_interpreter(b, env) for b in _supported_backends()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_one_call_one_launch(backend, launches):
    if backend not in launches:
        pytest.skip(f"{backend} unsupported on this device")
    found = launches[backend]
    if isinstance(found, dict):
        # _supported_backends() already dropped what the device cannot run, so
        # the only legitimate refusals left are the DSL missing at call time
        # (BackendSupportedError), the checker's problem-size ValueError and a
        # runner without a working profiler; anything else (a sticky CUDA
        # error, a binding ICHECK) is a failure of this guard rail, not a skip.
        if found["error"] in ("BackendSupportedError", _PROFILER_UNAVAILABLE) or (
            "Problem size is not supported" in found["msg"]
        ):
            pytest.skip(f"{backend}: {found['error']}: {found['msg']}")
        pytest.fail(f"{backend}: {found['error']}: {found['msg']}")
    if not found:
        # every backend launches at least its kernel, so an empty record means
        # the profiler dropped this session's events, not that the call
        # launched nothing
        pytest.skip(
            f"{backend}: the profiler recorded no device activity for the call "
            "(CUPTI dropped the session's events); zero launches is not a code path"
        )
    assert len(found) == EXPECTED[backend], (
        f"{backend}: {len(found)} device launches per call, expected "
        f"{EXPECTED[backend]}: {found}"
    )
