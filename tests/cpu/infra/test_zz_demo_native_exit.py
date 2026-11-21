"""DO NOT MERGE: demonstrates the missing-JUnit diagnostics from #9468 in CI.

The middle test stands in for a native library that calls exit() mid-test:
it prints an error and ends the process with os._exit(1), so pytest never
writes JUnit XML. The runner should report the batch as an infrastructure
error naming the exit code and this test, and print the batch log tail.
"""

import os
import sys


def test_before_native_exit():
    assert True


def test_native_library_calls_exit():
    print("CUDA error: an illegal memory access was encountered (demo)", flush=True)
    print(
        "  at deep_gemm/jit_utils.cuh:42 in CHECK_CUDA(cuModuleLoad(...)) (demo)",
        file=sys.stderr,
        flush=True,
    )
    os._exit(1)


def test_after_native_exit():
    assert True
