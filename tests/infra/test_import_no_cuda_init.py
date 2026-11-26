"""Importing FlashInfer must not initialize the CUDA driver (#4889).

A process that has called cuInit cannot use CUDA in a fork()ed child, so frameworks
that fork workers after importing FlashInfer (e.g. vLLM) would fail.
"""

import subprocess
import sys

import pytest
import torch

_PROBE = r"""
import os
import torch
import flashinfer, flashinfer.comm, flashinfer.fused_moe, flashinfer.gemm  # noqa: F401
assert not torch.cuda.is_initialized(), "torch initialized CUDA during import"
pid = os.fork()
if pid == 0:
    try:
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:
        os._exit(1)
    os._exit(0)
_, status = os.waitpid(pid, 0)
raise SystemExit(0 if status == 0 else 1)
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_forked_child_can_use_cuda_after_import():
    # NVML-based check so torch.cuda.is_available() in third-party imports stays fork-safe.
    env = {"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    import os

    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
