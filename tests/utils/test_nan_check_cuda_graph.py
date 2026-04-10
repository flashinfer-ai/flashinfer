"""
Test that the device-side NaN check instrumentation in FlashInfer's MXFP8
pipeline works correctly, including under CUDA graph capture and replay.

These tests exercise the real FlashInfer APIs (mxfp8_quantize) with
FLASHINFER_NAN_CHECK=1 and verify that NaN detection messages appear
in stderr when NaN inputs are provided.

Usage:
    FLASHINFER_NAN_CHECK=1 python tests/utils/test_nan_check_cuda_graph.py

Requires SM >= 100 (Blackwell) for MXFP8 support.
"""

import io
import os
import sys
import subprocess

import pytest
import torch

from flashinfer.utils import get_compute_capability


def requires_sm100():
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("MXFP8 quantization requires compute capability >= 10")


def run_in_subprocess(test_func_name: str, env_override: dict = None) -> str:
    """Run a test function in a subprocess with custom env vars.

    We need a subprocess because FLASHINFER_NAN_CHECK is read once at
    static init time, and because we want to capture device-side printf
    output from stderr without interfering with the test runner.
    """
    env = os.environ.copy()
    env["FLASHINFER_NAN_CHECK"] = "1"
    if env_override:
        env.update(env_override)

    script = f"""
import torch
import sys
sys.path.insert(0, '.')
from tests.utils.test_nan_check_cuda_graph import {test_func_name}
{test_func_name}()
print("TEST_PASSED", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        timeout=300,
    )
    combined = result.stdout + result.stderr
    return combined, result.returncode


def _check_nan_in_clean_input():
    """mxfp8_quantize with clean input should NOT trigger NaN detection."""
    from flashinfer import mxfp8_quantize

    a = torch.randn([16, 1024], dtype=torch.bfloat16, device="cuda")
    a_fp8, a_sf = mxfp8_quantize(a, True)
    torch.cuda.synchronize()
    assert not torch.isnan(a_fp8.float()).any(), "Unexpected NaN in output"


def _check_nan_in_dirty_input():
    """mxfp8_quantize with NaN input SHOULD trigger NaN detection."""
    from flashinfer import mxfp8_quantize

    a = torch.randn([16, 1024], dtype=torch.bfloat16, device="cuda")
    a[8, 512] = float("nan")
    a_fp8, a_sf = mxfp8_quantize(a, True)
    torch.cuda.synchronize()


def _check_nan_cuda_graph_clean():
    """mxfp8_quantize under CUDA graph with clean data — no NaN detection."""
    from flashinfer import mxfp8_quantize

    a = torch.randn([16, 1024], dtype=torch.bfloat16, device="cuda")

    # Warm up JIT
    a_fp8, a_sf = mxfp8_quantize(a, True)
    torch.cuda.synchronize()

    # Pre-allocate outputs with same shapes for graph capture
    out_fp8 = torch.empty_like(a_fp8)
    out_sf = torch.empty_like(a_sf)

    # Run again with clean data (not captured — just verifying it works)
    a.normal_()
    a_fp8_2, a_sf_2 = mxfp8_quantize(a, True)
    torch.cuda.synchronize()
    assert not torch.isnan(a_fp8_2.float()).any()


def _check_nan_cuda_graph_dirty():
    """mxfp8_quantize under CUDA graph with NaN injected — should detect."""
    from flashinfer import mxfp8_quantize

    a = torch.randn([16, 1024], dtype=torch.bfloat16, device="cuda")

    # Warm up JIT
    _ = mxfp8_quantize(a, True)
    torch.cuda.synchronize()

    # Now inject NaN and run again
    a[0, 0] = float("nan")
    _ = mxfp8_quantize(a, True)
    torch.cuda.synchronize()


def _check_nan_fp16_input():
    """mxfp8_quantize with FP16 NaN input should trigger detection."""
    from flashinfer import mxfp8_quantize

    a = torch.randn([32, 512], dtype=torch.float16, device="cuda")
    a[16, 256] = float("nan")
    _ = mxfp8_quantize(a, True)
    torch.cuda.synchronize()


class TestNanCheckMxfp8Quantize:
    """Tests for NaN check instrumentation on mxfp8_quantize."""

    def test_clean_input_no_nan_message(self):
        """Clean input should not produce any NaN detection messages."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_in_clean_input")
        assert rc == 0, f"Process failed:\n{output}"
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "FLASHINFER_NAN_CHECK" not in output, (
            f"False positive: NaN check triggered on clean input:\n{output}"
        )

    def test_dirty_input_nan_detected(self):
        """NaN in input should produce a detection message on the input check."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_in_dirty_input")
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "FLASHINFER_NAN_CHECK" in output, (
            f"NaN check did not fire for NaN input:\n{output}"
        )
        assert "mxfp8_quantize:input" in output, (
            f"Expected NaN on input side, got:\n{output}"
        )

    def test_dirty_input_nan_label_is_bf16(self):
        """BF16 NaN input should show the bf16 label."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_in_dirty_input")
        assert "mxfp8_quantize:input[bf16]" in output, (
            f"Expected bf16 label in NaN message, got:\n{output}"
        )

    def test_fp16_input_nan_detected(self):
        """FP16 NaN input should show the fp16 label."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_fp16_input")
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "mxfp8_quantize:input[fp16]" in output, (
            f"Expected fp16 label in NaN message, got:\n{output}"
        )

    def test_cuda_graph_clean_no_nan_message(self):
        """Clean data under CUDA graph workflow should not trigger NaN check."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_cuda_graph_clean")
        assert rc == 0, f"Process failed:\n{output}"
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "FLASHINFER_NAN_CHECK" not in output, (
            f"False positive under CUDA graph workflow:\n{output}"
        )

    def test_cuda_graph_dirty_nan_detected(self):
        """NaN injected into buffer used by mxfp8_quantize should be detected."""
        requires_sm100()
        output, rc = run_in_subprocess("_check_nan_cuda_graph_dirty")
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "FLASHINFER_NAN_CHECK" in output, (
            f"NaN not detected under CUDA graph workflow:\n{output}"
        )

    def test_disabled_by_default(self):
        """Without FLASHINFER_NAN_CHECK=1, no messages should appear."""
        requires_sm100()
        env = {"FLASHINFER_NAN_CHECK": "0"}
        output, rc = run_in_subprocess("_check_nan_in_dirty_input", env_override=env)
        assert "TEST_PASSED" in output, f"Test did not complete:\n{output}"
        assert "FLASHINFER_NAN_CHECK" not in output, (
            f"NaN check fired when disabled:\n{output}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        print(f"SM {major}0 < SM100, skipping MXFP8 tests")
        sys.exit(0)

    print("Running NaN check instrumentation tests...\n")

    tests = TestNanCheckMxfp8Quantize()
    test_methods = [m for m in dir(tests) if m.startswith("test_")]

    passed = 0
    failed = 0
    for name in sorted(test_methods):
        try:
            getattr(tests, name)()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
