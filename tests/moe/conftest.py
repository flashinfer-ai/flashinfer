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

"""Shared MoE test fixtures and import-path setup."""

import os
import sys

import pytest

from tests.moe.trtllm_gen_fused_moe_utils import MoeGemmBackend

# The DA acceptance tests reuse the public benchmark implementation. Pytest's
# importlib mode does not reliably add the checkout root for plain ``pytest``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(autouse=True)
def _disable_prims_ts_exhaustive_checker_in_moe_integration_tests(monkeypatch):
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")


@pytest.fixture
def moe_gemm_backend():
    """Use TRT-LLM unless a focused test explicitly requests both backends."""
    return MoeGemmBackend.TRTLLM
