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

"""CPU-reachable tests for the admission gate of the "specialized" SM120
fp8-blockwise GEMM backend (``is_specialized_supported``).

The gate orders its GPU-free checks (kill switch, geometry, allowlist,
dtypes, scale layout, contiguity, alignment, CUDA toolkit version) before
any device query, so every refusal tested here must be decidable - and must
record its fallback reason - without a GPU. In particular this pins the
toolkit floor: a toolkit below CUDA 12.9 is refused by the gate (recorded,
never raised), instead of failing later inside the JIT build.
"""

import contextlib
import json
from pathlib import Path

import pytest
import torch

import flashinfer.gemm.specialized_fp8_blockwise_sm120 as spec


def _valid_cpu_contract(m=4, n=64, k=2048):
    """An allowlisted, contract-satisfying call built on CPU tensors."""
    a = (torch.randn(m, k) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k) * 0.5).to(torch.float8_e4m3fn)
    a_scale = torch.rand(m, k // 128).float().contiguous()
    b_scale = torch.rand((n + 127) // 128, k // 128).float().contiguous()
    return a, b, a_scale, b_scale


def _gate(a, b, a_scale, b_scale):
    return spec.is_specialized_supported(
        a, b, a_scale, b_scale, "K", (1, 128, 128), out=None, out_dtype=torch.bfloat16
    )


def _last_reason():
    assert spec.fallback_reasons, "gate refused without recording a reason"
    backend, reason = spec.fallback_reasons[-1]
    assert backend == "specialized"
    return reason


@pytest.mark.cpu_only
def test_gate_refuses_toolkit_below_12_9(monkeypatch):
    """The 12.9 toolkit floor is enforced by the admission gate itself, on
    the CPU-reachable path, so an older install never reaches the JIT build
    (which is where it used to surface as a raise from ``backend="auto"``)."""
    monkeypatch.setattr(spec, "is_cuda_version_at_least", lambda _v: False)
    assert _gate(*_valid_cpu_contract()) is False
    assert _last_reason() == "toolkit:cuda_below_12.9"


@pytest.mark.cpu_only
def test_gate_refuses_undetectable_toolkit(monkeypatch):
    """No detectable toolkit version means no JIT build: refuse, don't raise."""

    def _boom(_v):
        raise RuntimeError("no toolkit")

    monkeypatch.setattr(spec, "is_cuda_version_at_least", _boom)
    assert _gate(*_valid_cpu_contract()) is False
    assert _last_reason() == "toolkit:version_unknown"


@pytest.mark.cpu_only
def test_gate_kill_switch(monkeypatch):
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    assert _gate(*_valid_cpu_contract()) is False
    assert _last_reason() == "kill_switch"


@pytest.mark.cpu_only
def test_gate_refuses_off_allowlist(monkeypatch):
    # Keep the toolkit check inert so the refusal is attributable to the
    # allowlist alone, whatever toolkit the CI host carries.
    monkeypatch.setattr(spec, "is_cuda_version_at_least", lambda _v: True)
    # M above the class's admitted cap
    n, k = next(iter(spec._allowlist()))
    m_max = spec._allowlist()[(n, k)]
    assert _gate(*_valid_cpu_contract(m=m_max + 1, n=n, k=k)) is False
    assert _last_reason().startswith("allowlist:")
    # (N, K) class not in the allowlist at all
    assert _gate(*_valid_cpu_contract(m=1, n=256, k=2048)) is False
    assert _last_reason().startswith("allowlist:")


@pytest.mark.cpu_only
def test_gate_reads_the_shipped_allowlist():
    """The gate's allowlist is exactly the shipped JSON, parsed as documented
    ("NxK" -> max admitted M), so tests parametrized from ``_allowlist()``
    cover precisely what auto-routing admits."""
    shipped = (
        Path(spec.__file__).parent / "specialized_fp8_blockwise_sm120_allowlist.json"
    )
    with open(shipped) as f:
        raw = json.load(f)["auto_route"]["nk_to_m_max"]
    expected = {tuple(int(x) for x in key.split("x")): int(v) for key, v in raw.items()}
    assert spec._allowlist() == expected
    assert expected, "shipped allowlist is empty"
    for (n, k), m_max in expected.items():
        assert n >= 1 and k % 128 == 0 and m_max >= 1


@pytest.mark.cpu_only
def test_gate_refuses_out_overlapping_an_input():
    """An ``out`` sharing storage bytes with an input must be refused: the
    kernel writes Y through raw pointers while it reads the operands, so an
    overlapping call belongs on the composable fallback (which reads its
    inputs before writing out)."""
    a, b, a_scale, b_scale = _valid_cpu_contract()
    m, n = a.shape[0], b.shape[0]
    out = b.flatten()[: m * n * 2].view(torch.bfloat16).view(m, n)
    assert not spec.is_specialized_supported(
        a, b, a_scale, b_scale, "K", (1, 128, 128), out=out, out_dtype=None
    )
    assert _last_reason() == "layout:out_overlaps_input"

    disjoint = torch.empty(m, n, dtype=torch.bfloat16)
    spec.fallback_reasons.clear()
    # A disjoint out passes every GPU-free check; on a CPU-only torch the
    # gate then reaches the device guards, which cannot run here.
    with contextlib.suppress(Exception):
        spec.is_specialized_supported(
            a, b, a_scale, b_scale, "K", (1, 128, 128), out=disjoint, out_dtype=None
        )
    assert ("specialized", "layout:out_overlaps_input") not in spec.fallback_reasons
