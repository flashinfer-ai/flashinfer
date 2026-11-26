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


class _FakeNvml:
    """Two GPUs, NVML order: index 0 = GPU-aaaa..., index 1 = GPU-b152...; MIG on 1."""

    class NVMLError(Exception):
        pass

    _uuids = ["GPU-aaaa1111-0000", "GPU-b152ded7-1111"]
    _caps = [(8, 0), (9, 0)]

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return self._uuids[h].encode()

    def nvmlDeviceGetHandleByUUID(self, uuid):
        if uuid == "MIG-0001":
            return 1
        raise self.NVMLError("not found")

    def nvmlDeviceGetCudaComputeCapability(self, h):
        return self._caps[h]


@pytest.mark.parametrize(
    "visible, expected",
    [
        (None, [(8, 0), (9, 0)]),
        ("1,0", [(9, 0), (8, 0)]),
        ("GPU-b152ded7", [(9, 0)]),  # unique UUID prefix, as CUDA accepts
        ("GPU-", []),  # ambiguous prefix: CUDA stops at this entry
        ("0,7,1", [(8, 0)]),  # invalid entry hides it and everything after
        ("MIG-0001", [(9, 0)]),
    ],
)
def test_visible_device_capabilities_follow_cuda_visible_devices(
    monkeypatch, visible, expected
):
    import flashinfer.compilation_context as cc

    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml())
    monkeypatch.setattr(cc.torch.cuda, "is_initialized", lambda: False)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert cc.get_visible_device_capabilities() == expected


def test_visible_device_capabilities_fall_back_without_nvml(monkeypatch):
    import flashinfer.compilation_context as cc

    fake = _FakeNvml()

    def broken_init():
        raise fake.NVMLError("libnvidia-ml.so not found")

    fake.nvmlInit = broken_init
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(cc.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(cc, "_torch_capabilities", lambda: [(12, 1)])
    assert cc.get_visible_device_capabilities() == [(12, 1)]
