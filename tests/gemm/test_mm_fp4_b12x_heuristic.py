"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Host-side contract tests for the ``mm_fp4`` auto-heuristic SM12x dispatch
(#3170, action item 3).

The heuristic must prefer ``b12x`` on every SM12x device (``major == 12``),
SM120 and SM121 alike: the b12x kernel is built on dense block-scaled MMA
atoms, a family-specific (120f) feature shared by the whole 12.x spec. No GPU
is required: compute capability and CUDA version are patched at the
``gemm_base`` module boundary.
"""

import pytest
import packaging.version
import torch

from flashinfer.gemm import gemm_base

_SUITABLE = ["b12x", "cutlass", "cudnn"]


class _FakeTensor:
    """The heuristic only reads ``a.device``."""

    def __init__(self):
        self.device = torch.device("cuda", 0)


@pytest.fixture
def patched_env(monkeypatch):
    def apply(major, minor, cuda_version):
        monkeypatch.setattr(
            gemm_base, "get_compute_capability", lambda device: (major, minor)
        )
        monkeypatch.setattr(
            gemm_base,
            "get_cuda_version",
            lambda: packaging.version.Version(cuda_version),
        )

    return apply


def _heuristic(use_nvfp4=True):
    return gemm_base._heuristic_func_mm_fp4(
        _SUITABLE,
        _FakeTensor(),
        _FakeTensor(),
        _FakeTensor(),
        _FakeTensor(),
        use_nvfp4=use_nvfp4,
    )


@pytest.mark.parametrize("major,minor", [(12, 0), (12, 1)])
def test_sm12x_auto_prefers_b12x(patched_env, major, minor):
    """Both SM120 and SM121 get the b12x-first auto order (CUDA >= 12.9, NVFP4)."""
    patched_env(major, minor, "13.0")
    assert _heuristic() == ["b12x", "cutlass", "cudnn"]


def test_sm121_auto_prefers_b12x_cuda_129(patched_env):
    """The NVFP4 b12x floor is CUDA 12.9; SM121 selects it from that version."""
    patched_env(12, 1, "12.9")
    assert _heuristic() == ["b12x", "cutlass", "cudnn"]


def test_sm121_mxpf4_needs_cuda_13(patched_env):
    """MXFP4 b12x requires CUDA 13, so older CUDA falls back on SM121 too."""
    patched_env(12, 1, "12.8")
    assert "b12x" not in _heuristic(use_nvfp4=False)
    patched_env(12, 1, "13.0")
    assert _heuristic(use_nvfp4=False) == ["b12x", "cutlass", "cudnn"]


@pytest.mark.parametrize("major,minor", [(10, 0), (10, 3), (10, 7)])
def test_sm100_family_keeps_cudnn_cutlass_order(patched_env, major, minor):
    """SM100-family auto order is untouched by the SM12x branch."""
    patched_env(major, minor, "13.0")
    backends = _heuristic()
    assert backends, "heuristic returned no backend"
    assert "b12x" not in backends
    assert set(backends) <= {"cutlass", "cudnn", "cute-dsl", "cutedsl_low_latency"}


def test_b12x_dropped_when_unsuitable(patched_env):
    """A backend absent from ``suitable_backends`` is never auto-selected."""
    patched_env(12, 1, "13.0")
    backends = gemm_base._heuristic_func_mm_fp4(
        ["cutlass", "cudnn"],
        _FakeTensor(),
        _FakeTensor(),
        _FakeTensor(),
        _FakeTensor(),
        use_nvfp4=True,
    )
    assert backends == ["cutlass", "cudnn"]
