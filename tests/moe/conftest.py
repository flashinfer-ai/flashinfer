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


# The CuTe DSL gather/finalize grouped GEMM kernels raise a deliberate
# NotImplementedError on Rubin for configurations the SM107 kernel does not
# implement -- SwiGLU-only activation fusion, no per-token activation scale, no
# unfused finalize, Float32 router scales only -- and for a CuTe DSL older than
# the 4.8 those kernels require. Every one of these names the kernel in its
# message. They are documented product limitations, not regressions, so the
# affected parameterizations are reported as skips rather than errors.
#
# Matching on the pair of tokens rather than on whole sentences is deliberate:
# the guards are worded inconsistently (one says "is not supported", another
# "are not supported"), so sentence fragments silently stop matching when a
# message is reworded.
def _is_rubin_sm107_capability_gap(exc: NotImplementedError) -> bool:
    message = str(exc)
    return "Rubin" in message and "SM107" in message


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Skip configurations the Rubin (SM107) MoE kernels do not implement."""
    try:
        yield
    except NotImplementedError as e:
        if _is_rubin_sm107_capability_gap(e):
            pytest.skip(f"unsupported on Rubin (SM107): {e}")
        raise
