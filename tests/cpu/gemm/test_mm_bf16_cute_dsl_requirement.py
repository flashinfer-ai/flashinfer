# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape eligibility of the mm_bf16 cute-dsl backend; no GPU execution."""

import pytest
import torch

import flashinfer
from flashinfer.gemm.gemm_base import _cute_dsl_mm_bf16_requirement


@pytest.fixture
def sm100_with_cute_dsl(monkeypatch):
    """Report an SM100 device with the CuTe DSL installed."""
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (10, 0))
    monkeypatch.setattr("flashinfer.gemm.gemm_base.is_sm100a_supported", lambda _: True)
    monkeypatch.setattr(
        "flashinfer.cute_dsl.availability.is_cute_dsl_available", lambda: True
    )


def _operands(m, k, n=10240):
    a = torch.empty((m, k), dtype=torch.bfloat16)
    b = torch.empty((n, k), dtype=torch.bfloat16)
    return a, b.T


@pytest.mark.cpu_only
@pytest.mark.parametrize("m,k", [(1, 320), (32, 320), (8, 0), (8, 192)])
def test_low_m_rejects_k_not_divisible_by_128(sm100_with_cute_dsl, m, k):
    """M <= 32 rejects K % 128 != 0 with a ValueError that names K."""
    a, b = _operands(m, k)
    with pytest.raises(
        ValueError,
        match=rf"requires K to be a positive multiple of 128 for M <= 32, got K={k}\.",
    ):
        flashinfer.mm_bf16(a, b, backend="cute-dsl")


@pytest.mark.cpu_only
def test_m_above_32_accepts_k_not_divisible_by_128(sm100_with_cute_dsl):
    """K % 128 != 0 stays accepted for M > 32, which the low-M kernels do not serve."""
    a, b = _operands(33, 320)
    assert _cute_dsl_mm_bf16_requirement(a, b)
