"""Regression test: importing flashinfer must not require a CUDA device.

gdn_decode_bf16_state.py previously called torch.cuda.get_device_properties()/
get_device_capability() at module scope, so simply importing flashinfer crashed
on any machine without a visible CUDA device (regression of #3262/#3293,
reintroduced by #3502). Run as a subprocess with CUDA_VISIBLE_DEVICES unset so
this is meaningful even on a GPU-equipped CI runner (same technique #3293 used
to demonstrate the original fix).

    python -m pytest tests/test_gdn_decode_cpu_import.py -v --noconftest
"""

import os
import subprocess
import sys


def _run_import_with_no_cuda_devices(module: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_gdn_decode_without_cuda_device():
    result = _run_import_with_no_cuda_devices("flashinfer.gdn_decode")
    assert result.returncode == 0, result.stderr


def test_import_flashinfer_without_cuda_device():
    result = _run_import_with_no_cuda_devices("flashinfer")
    assert result.returncode == 0, result.stderr
