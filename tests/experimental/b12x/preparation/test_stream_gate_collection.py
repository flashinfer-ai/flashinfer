"""A garbage collection inside a gated sample must not unload a CUDA library.

Unloading waits for queued device work, and the gate holds that work until the
sampling thread releases it. The race therefore runs in a child process: a
deadlock ends at the timeout instead of hanging the test session.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_RACE = textwrap.dedent(
    '''
    import gc
    from types import SimpleNamespace

    import torch
    from cuda.bindings import driver

    from b12x.preparation import _measurement

    PTX = b""".version 7.0
    .target sm_50
    .address_size 64
    .visible .entry noop() { ret; }
    \\0"""


    def check(result):
        assert result[0] == driver.CUresult.CUDA_SUCCESS, result[0]
        return result[1] if len(result) > 1 else None


    class Module:
        """A module that its kernel cache evicts right after a launch."""

        def __init__(self):
            self.library = check(driver.cuLibraryLoadData(PTX, None, None, 0, None, None, 0))
            kernel = check(driver.cuLibraryGetKernel(self.library, b"noop"))
            self.function = check(driver.cuKernelGetFunction(kernel))
            self.cycle = self

        def launch(self, stream):
            check(driver.cuLaunchKernel(self.function, 1, 1, 1, 1, 1, 1, 0, stream, 0, 0))

        def __del__(self):
            # Like a collected CuTe module: waits for its queued kernel.
            driver.cuLibraryUnload(self.library)


    loaded = []


    def produce():
        # Modules load before the gate, as primed candidates do.
        loaded.append(Module())


    def invoke():
        loaded.pop().launch(torch.cuda.current_stream().cuda_stream)
        # Tracked allocations start an automatic collection of the evicted module.
        gc.set_threshold(1)
        [[] for _ in range(64)]


    torch.zeros(1, device="cuda")
    call = SimpleNamespace(benchmark_producers=(), produce=produce, reset=None, invoke=invoke)
    gate = _measurement._StreamGate()
    timer = _measurement._TimedCall(call, lambda: None, 1, gate)
    timer.replay()
    torch.cuda.synchronize()
    gate.close()
    gc.collect()
    print("released", flush=True)
    '''
)


def test_collection_in_a_gated_sample_cannot_deadlock_library_unload():
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RACE], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        pytest.fail("a library unload inside the stream gate deadlocked the sample")
    assert result.returncode == 0 and "released" in result.stdout, result.stderr[-2000:]
