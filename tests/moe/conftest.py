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


def pytest_make_parametrize_id(config, val, argname):
    """Give QuantConfig / ActivationConfig parametrize values stable node ids.

    Without this, pytest labels them ``quant0`` / ``activation0``. Keep an
    explicit ``ids=`` only when a value is a hook-unknown object or tuple, or
    when several numeric columns need axis labels.
    """
    from tests.moe.utils import parametrize_id

    return parametrize_id(val)


@pytest.fixture(autouse=True)
def _disable_prims_ts_exhaustive_checker_in_moe_integration_tests(monkeypatch):
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")


@pytest.fixture
def moe_gemm_backend():
    """Use TRT-LLM unless a focused test explicitly requests both backends."""
    return MoeGemmBackend.TRTLLM


@pytest.fixture
def frost_metadata_only(monkeypatch):
    """Isolate logical catalog tests from CUDA binding and device cache state."""
    from flashinfer.fused_moe.backends.cudnn_frost import device, runtime
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import fc2
    from flashinfer.fused_moe.backends.cudnn_frost.bf16 import runtime as bf16
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import runtime as mxfp8
    from flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 import runtime as mixed
    from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import runtime as nvfp4

    runtimes = (bf16, mxfp8, mixed, nvfp4)
    device_facts = device.device_facts
    device_facts.cache_clear()
    for module in runtimes:
        module.clear_artifact_cache()
    # BF16 imports the binding function directly; block-scaled runtimes use
    # the shared module. Neither route should specialize a logical test pool.
    for module in (runtime, bf16, fc2):
        monkeypatch.setattr(module, "discovery_binding", lambda kernel: kernel)
    facts = device.DeviceFacts("sm_107a", 334848, 117440512, (84,) * 16)
    monkeypatch.setattr(device, "device_facts", lambda _: facts)
    try:
        yield
    finally:
        for module in runtimes:
            module.clear_artifact_cache()
        device_facts.cache_clear()
