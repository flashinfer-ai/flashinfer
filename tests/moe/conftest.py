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


@pytest.fixture(scope="module")
def observe_swap_ab_mma_plans():
    # K96 accumulation depends on the selected FC1/FC2 pipeline stages, including
    # auto k_steps. Observe compilation and cache hits without changing tactics.
    import torch
    from flashinfer.cute_dsl import is_cute_dsl_available

    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or not is_cute_dsl_available()
    ):
        yield
        return
    import cutlass.cute as cute
    from flashinfer.fused_moe.cute_dsl.blackwell import moe_swap_ab
    from tests.moe.test_helpers import (
        cute_dsl_moe_swap_ab_reference as reference_module,
    )

    configs, selected = {}, {}
    compile_kernel = cute.compile
    reference = reference_module.cute_dsl_moe_reference

    def record(compiled):
        phase, tile = configs[id(compiled)]
        selected[phase] = tile

    def compile_observed(function, *args, **kwargs):
        compiled = compile_kernel(function, *args, **kwargs)
        kernel = getattr(function, "__self__", None)
        if isinstance(kernel, moe_swap_ab.Sm100BlockScaledGroupedGemmKernel):
            configs[id(compiled)] = (
                kernel.scatter_output,
                kernel.tile_k if kernel.native_mma_k == 96 else None,
            )
            record(compiled)
        return compiled

    class ObservedCache(dict):
        def get(self, key, default=None):
            compiled = super().get(key, default)
            if compiled is not None:
                record(compiled)
            return compiled

    def reference_observed(**kwargs):
        if kwargs.get("sf_vec_size", 16) == 16:
            plans = []
            for phase, weight in ((False, "w1_weight"), (True, "w2_weight")):
                tile = selected[phase]
                plan = None
                if tile is not None:
                    k = kwargs[weight].shape[-1] * 2
                    plan = ()
                    for start in range(0, k, tile):
                        extent = min(tile, k - start)
                        plan += (96, 96) * (extent // 192)
                        plan += (64,) * ((extent % 192) // 64)
                plans.append(plan)
            kwargs["mma_k_plans"] = tuple(plans)
        return reference(**kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cute, "compile", compile_observed)
        patch.setattr(moe_swap_ab, "_blockscaled_kernel_cache", ObservedCache())
        patch.setattr(reference_module, "cute_dsl_moe_reference", reference_observed)
        yield


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
