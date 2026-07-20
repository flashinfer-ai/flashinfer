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

Tests for the CuTe-DSL kernel cache naming contract.

The kernel-name string is the sole per-kernel cache key: the module-level
meta.json guards arch / DSL version / source hash, but NOT per-kernel codegen
parameters. If a name function omits a codegen argument, two different
kernels collide on one artifact and the cache silently serves the wrong
binary. These tests enforce, for each cache adopter, that the name is a
function of every codegen argument:

1. Signature coverage: every parameter of each @functools.cache'd kernel
   getter appears in the name function's signature (catches a new parameter
   that was never threaded into the name).
2. Per-argument perturbation: changing any single name argument changes the
   returned name (catches a parameter that is accepted but ignored).

New cache adopters (other CuTe-DSL op families) should replicate this
pattern for their own name functions.
"""

import inspect
import re

import pytest

pytest.importorskip("cutlass")

from flashinfer.quantization.kernels.nvfp4_quantize import (  # noqa: E402
    SF_LAYOUT_8x4,
    SF_LAYOUT_128x4,
    _get_compiled_kernel_nvfp4,
    _get_compiled_kernel_nvfp4_per_token,
    _get_compiled_kernel_nvfp4_tma,
    _nvfp4_kernel_name,
)
from flashinfer.quantization.nvfp4_quantization_utils import (  # noqa: E402
    NVFP44Over6Config,
)

NVFP4_KERNEL_GETTERS = [
    _get_compiled_kernel_nvfp4,
    _get_compiled_kernel_nvfp4_per_token,
    _get_compiled_kernel_nvfp4_tma,
]

# Getter parameters that deliberately do NOT participate in the cache key.
# Empty today; add here (with justification) if a non-codegen parameter is
# ever introduced.
NVFP4_NON_CODEGEN_PARAMS: set = set()

# A baseline argument set and, for each argument, a distinct alternative.
NVFP4_NAME_BASELINE = {
    "variant": "swizzled",
    "dtype_key": "bfloat16",
    "K": 4096,
    "sf_layout": SF_LAYOUT_128x4,
    "enable_pdl": True,
    "disable_fp4_quant_fast_math": False,
    "nvfp4_4over6_config": None,
}
NVFP4_NAME_PERTURBED = {
    "variant": "linear",
    "dtype_key": "float16",
    "K": 2048,
    "sf_layout": SF_LAYOUT_8x4,
    "enable_pdl": False,
    "disable_fp4_quant_fast_math": True,
    "nvfp4_4over6_config": NVFP44Over6Config(),
}


@pytest.mark.parametrize("getter", NVFP4_KERNEL_GETTERS)
def test_nvfp4_kernel_name_signature_covers_codegen_params(getter):
    """Every kernel-getter parameter must be expressible in the cache key.

    Fails the moment someone adds a parameter to a kernel getter without
    threading it through the name function.
    """
    getter_params = set(inspect.signature(getter).parameters)
    name_params = set(inspect.signature(_nvfp4_kernel_name).parameters)
    missing = getter_params - name_params - NVFP4_NON_CODEGEN_PARAMS
    assert not missing, (
        f"{getter.__name__} has codegen parameter(s) {sorted(missing)} that "
        "_nvfp4_kernel_name cannot encode. Add them to the name function "
        "(or, if provably non-codegen, to NVFP4_NON_CODEGEN_PARAMS with a "
        "justification)."
    )


@pytest.mark.parametrize("param", sorted(NVFP4_NAME_BASELINE))
def test_nvfp4_kernel_name_varies_with_every_argument(param):
    """Changing any single argument must change the kernel name.

    Catches arguments that the name function accepts but ignores.
    """
    baseline_name = _nvfp4_kernel_name(**NVFP4_NAME_BASELINE)
    kwargs = dict(NVFP4_NAME_BASELINE)
    kwargs[param] = NVFP4_NAME_PERTURBED[param]
    assert _nvfp4_kernel_name(**kwargs) != baseline_name, (
        f"_nvfp4_kernel_name ignores argument {param!r}: two different "
        "kernel specializations would collide on one cache artifact."
    )


@pytest.mark.parametrize(
    "config",
    [
        NVFP44Over6Config(),
        NVFP44Over6Config(e4m3_max=256),
        NVFP44Over6Config(err_mode="MSE"),
        NVFP44Over6Config(err_use_fast_math=True),
    ],
)
def test_nvfp4_kernel_name_distinguishes_4over6_configs(config):
    """Each field of NVFP44Over6Config must be reflected in the name."""
    base = _nvfp4_kernel_name(**{**NVFP4_NAME_BASELINE, "nvfp4_4over6_config": None})
    with_cfg = _nvfp4_kernel_name(
        **{**NVFP4_NAME_BASELINE, "nvfp4_4over6_config": config}
    )
    assert with_cfg != base
    # And distinct configs must not collide with each other.
    others = [
        NVFP44Over6Config(),
        NVFP44Over6Config(e4m3_max=256),
        NVFP44Over6Config(err_mode="MSE"),
        NVFP44Over6Config(err_use_fast_math=True),
    ]
    names = {
        _nvfp4_kernel_name(**{**NVFP4_NAME_BASELINE, "nvfp4_4over6_config": c})
        for c in others
    }
    assert len(names) == len(others)


def test_nvfp4_kernel_name_is_symbol_safe():
    """Names must already be valid symbol/filename components.

    JitSpecCuteDsl sanitizes names before use; a name relying on that
    sanitization could collide with a different name that sanitizes to the
    same string, so the raw name must not need it.
    """
    for cfg in (None, NVFP44Over6Config(err_mode="MSE")):
        name = _nvfp4_kernel_name(**{**NVFP4_NAME_BASELINE, "nvfp4_4over6_config": cfg})
        assert re.fullmatch(r"[0-9A-Za-z_]+", name), name
