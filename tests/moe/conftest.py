# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest configuration shared by MoE tests."""

import os
import sys

import pytest


# The DA acceptance tests reuse the public benchmark implementation. Pytest's
# importlib mode does not reliably add the checkout root for plain ``pytest``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Known Rubin (SM107) capability gaps in the CuTe DSL gather/finalize grouped
# GEMM kernels. Each is an explicit, deliberate NotImplementedError in the
# product code -- the Rubin kernel genuinely lacks the feature, it is not a
# regression -- so the affected parameterizations are expected failures rather
# than errors. Drop the corresponding entry here when the kernel gains the
# feature.
_RUBIN_SM107_MOE_GAPS = (
    # The Rubin wrapper has no a_per_token_scale_ptr parameter.
    "use_a_per_token_scale (per-token activation scale) is not supported",
    # The Rubin gather grouped GEMM fuses SwiGLU only.
    "is not supported by the Rubin (SM107) gather grouped GEMM kernel yet",
    # The Rubin finalize kernel always does the fused scatter-add.
    "use_fused_finalize=False is not supported",
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Report the deliberate Rubin SM107 MoE feature gaps as xfail."""
    try:
        yield
    except NotImplementedError as e:
        message = str(e)
        if any(gap in message for gap in _RUBIN_SM107_MOE_GAPS):
            pytest.xfail(f"known Rubin (SM107) MoE kernel gap: {message}")
        raise
