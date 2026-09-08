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

from flashinfer.gemm.gemm_svdquant import (  # noqa: E402
    _sm120_nvfp4_svdquant_runner,
    _sm120_nvfp4_svdquant_unfused_runner,
    _sm120_svdquant_kernel_name,
    _svdquant_kernel_source_files,
)

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

# Baseline argument sets shared by the individual one-at-a-time perturbation tests.
NVFP4_NAME_BASELINE = {
    "variant": "swizzled",
    "dtype_key": "bfloat16",
    "K": 4096,
    "sf_layout": SF_LAYOUT_128x4,
    "enable_pdl": True,
    "disable_fp4_quant_fast_math": False,
    "silu_and_mul": False,
    "nvfp4_4over6_config": None,
    "global_scale_is_tensor": True,
    "smooth_quant": False,
}

SVDQUANT_NAME_BASELINE = {
    "rank": 32,
    "with_bias": False,
    "mma_tiler_mn": (64, 64),
    "tile_k": 128,
    "swap_ab": False,
    "max_active_clusters": 1,
    "enable_pdl": False,
    "enable_iket": False,
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


@pytest.mark.parametrize(
    "param,alternate",
    [
        pytest.param("variant", "linear", id="variant"),
        pytest.param("dtype_key", "float16", id="dtype_key"),
        pytest.param("K", 2048, id="K"),
        pytest.param("sf_layout", SF_LAYOUT_8x4, id="sf_layout"),
        pytest.param("enable_pdl", False, id="enable_pdl"),
        pytest.param(
            "disable_fp4_quant_fast_math",
            True,
            id="disable_fp4_quant_fast_math",
        ),
        pytest.param("silu_and_mul", True, id="silu_and_mul"),
        pytest.param(
            "nvfp4_4over6_config",
            NVFP44Over6Config(),
            id="nvfp4_4over6_config",
        ),
        pytest.param("global_scale_is_tensor", False, id="global_scale_is_tensor"),
        pytest.param("smooth_quant", True, id="smooth_quant"),
    ],
)
def test_nvfp4_kernel_name_varies_with_every_argument(param, alternate):
    """Changing any single argument must change the kernel name.

    Catches arguments that the name function accepts but ignores.
    """
    baseline_name = _nvfp4_kernel_name(**NVFP4_NAME_BASELINE)
    kwargs = dict(NVFP4_NAME_BASELINE)
    kwargs[param] = alternate
    assert _nvfp4_kernel_name(**kwargs) != baseline_name, (
        f"_nvfp4_kernel_name ignores argument {param!r}: two different "
        "kernel specializations would collide on one cache artifact."
    )


@pytest.mark.parametrize(
    "param,alternate",
    [
        pytest.param("rank", 64, id="rank"),
        pytest.param("with_bias", True, id="with_bias"),
        pytest.param("mma_tiler_mn", (128, 64), id="mma_tiler_mn"),
        pytest.param("tile_k", 256, id="tile_k"),
        pytest.param("swap_ab", True, id="swap_ab"),
        pytest.param("max_active_clusters", 2, id="max_active_clusters"),
        pytest.param("enable_pdl", True, id="enable_pdl"),
        pytest.param("enable_iket", True, id="enable_iket"),
    ],
)
def test_sm120_svdquant_kernel_name_varies_with_every_argument(param, alternate):
    baseline_name = _sm120_svdquant_kernel_name(**SVDQUANT_NAME_BASELINE)
    kwargs = dict(SVDQUANT_NAME_BASELINE)
    kwargs[param] = alternate
    assert _sm120_svdquant_kernel_name(**kwargs) != baseline_name


def test_sm120_svdquant_kernel_name_is_symbol_safe():
    name = _sm120_svdquant_kernel_name(
        **{**SVDQUANT_NAME_BASELINE, "mma_tiler_mn": (128, 64), "enable_iket": True}
    )
    assert re.fullmatch(r"[A-Za-z0-9_]+", name), name


def test_sm120_svdquant_source_fingerprint_covers_layout_helpers():
    from flashinfer.cute_dsl import utils as cute_dsl_utils
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x

    sources = _svdquant_kernel_source_files()
    assert cute_dsl_utils.__file__ in sources
    assert dense_blockscaled_gemm_sm120_b12x.__file__ in sources


@pytest.mark.parametrize(
    "runner_factory",
    [
        _sm120_nvfp4_svdquant_runner,
        _sm120_nvfp4_svdquant_unfused_runner,
    ],
)
def test_svdquant_autotune_cache_distinguishes_pdl(runner_factory):
    disabled = runner_factory(False)
    enabled = runner_factory(True)
    assert disabled.get_cache_key_extras([]) == (False,)
    assert enabled.get_cache_key_extras([]) == (True,)


def test_sm100_svdquant_autotune_cache_distinguishes_pdl(monkeypatch):
    from flashinfer.gemm import gemm_svdquant

    monkeypatch.setattr(gemm_svdquant, "get_nvfp4_svdquant_module", object)
    disabled = gemm_svdquant._nvfp4_svdquant_gemm_runner(False)
    enabled = gemm_svdquant._nvfp4_svdquant_gemm_runner(True)
    assert disabled.get_cache_key_extras([]) == (False,)
    assert enabled.get_cache_key_extras([]) == (True,)


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


# ---------------------------------------------------------------------------
# mm_fp4 (flashinfer/gemm/gemm_mm_fp4_cute_dsl.py) cache adopter
# ---------------------------------------------------------------------------

import torch  # noqa: E402

from flashinfer.gemm.gemm_mm_fp4_cute_dsl import (  # noqa: E402
    _blockscaled_mxfp8_gemm_cache_key_files,
    _blockscaled_kernel_disk_name,
    _compile_block_scaled_gemm,
    _mxfp8_blockscaled_kernel_disk_name,
    _mm_fp4_cache_key,
    _mm_mxfp8_cache_key,
)

# A baseline argument set and, for each argument, a distinct alternative.
MM_FP4_NAME_BASELINE = {
    "sf_vec_size": 16,
    "mma_tiler_mn": (256, 128),
    "cluster_shape_mn": (2, 1),
    "swap_ab": False,
    "use_prefetch": False,
    "kernel_type": "sm100",
    "use_tma_store": None,
    "enable_pdl": False,
    "out_dtype": torch.bfloat16,
    "batch_size": 1,
    "max_active_clusters": 74,
}
MM_FP4_NAME_PERTURBED = {
    "sf_vec_size": 32,
    "mma_tiler_mn": (128, 256),  # transpose of baseline: catches mixed-up axes
    "cluster_shape_mn": (1, 2),
    "swap_ab": True,
    "use_prefetch": True,
    "kernel_type": "sm103",
    "use_tma_store": True,
    "enable_pdl": True,
    "out_dtype": torch.float16,
    "batch_size": 2,
    "max_active_clusters": 148,
}


def _mm_fp4_name(**kwargs):
    tactic = (
        kwargs["mma_tiler_mn"],
        kwargs["cluster_shape_mn"],
        kwargs["swap_ab"],
        kwargs["use_prefetch"],
        kwargs["kernel_type"],
        kwargs["use_tma_store"],
    )
    cache_key = _mm_fp4_cache_key(
        kwargs["sf_vec_size"], tactic, kwargs["enable_pdl"], kwargs["out_dtype"]
    )
    return _blockscaled_kernel_disk_name(
        cache_key, kwargs["batch_size"], kwargs["max_active_clusters"]
    )


@pytest.mark.parametrize("param", sorted(MM_FP4_NAME_BASELINE))
def test_mm_fp4_kernel_name_varies_with_every_argument(param):
    """Changing any single codegen argument must change the kernel name, and
    names must be symbol-safe as produced (see the nvfp4 twins above)."""
    baseline_name = _mm_fp4_name(**MM_FP4_NAME_BASELINE)
    kwargs = dict(MM_FP4_NAME_BASELINE)
    kwargs[param] = MM_FP4_NAME_PERTURBED[param]
    perturbed_name = _mm_fp4_name(**kwargs)
    assert perturbed_name != baseline_name, (
        f"_blockscaled_kernel_disk_name ignores argument {param!r}: two "
        "different kernel specializations would collide on one cache artifact."
    )
    for name in (baseline_name, perturbed_name):
        assert re.fullmatch(r"[0-9A-Za-z_]+", name), name


# ---------------------------------------------------------------------------
# mm_mxfp8 SM100 cache adopter
# ---------------------------------------------------------------------------

MM_MXFP8_NAME_BASELINE = {
    "sf_vec_size": 32,
    "mma_tiler_mn": (128, 128),
    "cluster_shape_mn": (2, 1),
    "swap_ab": False,
    "use_prefetch": False,
    "enable_pdl": False,
    "out_dtype": torch.bfloat16,
    "split_k_slices": 1,
    "batch_size": 1,
    "max_active_clusters": 74,
}
MM_MXFP8_NAME_PERTURBED = {
    "sf_vec_size": 16,
    "mma_tiler_mn": (256, 64),
    "cluster_shape_mn": (1, 2),
    "swap_ab": True,
    "use_prefetch": True,
    "enable_pdl": True,
    "out_dtype": torch.float16,
    "split_k_slices": 4,
    "batch_size": 2,
    "max_active_clusters": 148,
}


def _mm_mxfp8_name(**kwargs):
    cache_key = _mm_mxfp8_cache_key(
        kwargs["sf_vec_size"],
        kwargs["mma_tiler_mn"],
        kwargs["cluster_shape_mn"],
        kwargs["swap_ab"],
        kwargs["use_prefetch"],
        kwargs["enable_pdl"],
        kwargs["out_dtype"],
        kwargs["split_k_slices"],
    )
    return _mxfp8_blockscaled_kernel_disk_name(
        cache_key, kwargs["batch_size"], kwargs["max_active_clusters"]
    )


def test_mm_mxfp8_cache_key_schema_is_fully_exercised():
    name_only_params = {"batch_size", "max_active_clusters"}
    assert set(inspect.signature(_mm_mxfp8_cache_key).parameters) == (
        set(MM_MXFP8_NAME_BASELINE) - name_only_params
    )


@pytest.mark.parametrize("param", sorted(MM_MXFP8_NAME_BASELINE))
def test_mm_mxfp8_kernel_name_varies_with_every_argument(param):
    """Every SM100 MXFP8 specialization must own a distinct artifact name."""
    baseline_name = _mm_mxfp8_name(**MM_MXFP8_NAME_BASELINE)
    kwargs = dict(MM_MXFP8_NAME_BASELINE)
    kwargs[param] = MM_MXFP8_NAME_PERTURBED[param]
    perturbed_name = _mm_mxfp8_name(**kwargs)
    assert perturbed_name != baseline_name, (
        f"_mxfp8_blockscaled_kernel_disk_name ignores argument {param!r}: "
        "two different kernel specializations would collide on one cache artifact."
    )
    for name in (baseline_name, perturbed_name):
        assert re.fullmatch(r"[0-9A-Za-z_]+", name), name


def test_mm_mxfp8_source_fingerprint_covers_splitk_kernel():
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm100_splitk

    assert (
        dense_blockscaled_gemm_sm100_splitk.__file__
        in _blockscaled_mxfp8_gemm_cache_key_files()
    )


def test_blockscaled_compile_accepts_operation_specific_disk_cache(monkeypatch):
    """MXFP8 can reuse the FP4 harness without reusing its cache schema."""
    from flashinfer.cute_dsl import utils as cute_dsl_utils
    from flashinfer.gemm import gemm_mm_fp4_cute_dsl
    from flashinfer.jit import cute_dsl_core

    cache_key = _mm_mxfp8_cache_key(
        *(
            MM_MXFP8_NAME_BASELINE[key]
            for key in (
                "sf_vec_size",
                "mma_tiler_mn",
                "cluster_shape_mn",
                "swap_ab",
                "use_prefetch",
                "enable_pdl",
                "out_dtype",
                "split_k_slices",
            )
        )
    )
    compile_kernel = object()
    extra_key_files = ("mxfp8_kernel.py",)
    calls = []

    monkeypatch.setattr(cute_dsl_utils, "get_max_active_clusters", lambda _: 17)
    monkeypatch.setattr(
        gemm_mm_fp4_cute_dsl,
        "_make_blockscaled_gemm_compile_fn",
        lambda *args, **kwargs: compile_kernel,
    )

    def fake_build(module_name, kernel_name, compile_fn, *, extra_key_files):
        calls.append((module_name, kernel_name, compile_fn, extra_key_files))
        return "compiled"

    monkeypatch.setattr(cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build)

    result = _compile_block_scaled_gemm(
        {},
        cache_key,
        object,
        ab_cutlass_dtype=object(),
        sf_dtype=object(),
        c_cutlass_dtype=object(),
        ab_assumed_align=16,
        cluster_shape_mn=(2, 1),
        swap_ab=False,
        sf_m=1,
        sf_n=1,
        sf_k=1,
        batch_size=2,
        cache_module_name="mm_mxfp8",
        device_index=0,
        disk_kernel_name_fn=lambda key, batch, clusters: (
            f"mxfp8_{key[-1]}_{batch}_{clusters}"
        ),
        cache_key_files_fn=lambda: extra_key_files,
    )

    assert result == ("compiled", 17)
    assert calls == [("mm_mxfp8", "mxfp8_1_2_17", compile_kernel, extra_key_files)]


@pytest.mark.parametrize(
    "m,n,k,tactic",
    [
        pytest.param(
            64,
            128,
            256,
            ((128, 128), (1, 1), False, False, 1),
            id="persistent",
        ),
        pytest.param(
            8,
            128,
            256,
            ((128, 8), (1, 1), True, False, 2),
            id="split-k",
        ),
    ],
)
def test_mm_mxfp8_runner_routes_to_disk_cache(monkeypatch, m, n, k, tactic):
    from flashinfer.gemm import gemm_base

    compile_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return (lambda *launch_args: None), 1

    monkeypatch.setattr(gemm_base, "_compile_block_scaled_gemm", fake_compile)
    monkeypatch.setattr(gemm_base, "_prepare_alpha_for_launch", lambda *args: None)
    monkeypatch.setattr(gemm_base, "get_device_index", lambda device: 0)

    runner = gemm_base._cute_dsl_gemm_mxfp8_runner(10, 0, False, torch.bfloat16)
    a = torch.empty((m, k))
    b = torch.empty((k, n))
    out = torch.empty((m, n))
    scale = torch.empty(1)
    runner.forward([a, b, scale, scale, torch.bfloat16, out, None], tactic=tactic)

    assert len(compile_calls) == 1
    args, kwargs = compile_calls[0]
    split_k_slices = tactic[-1]
    assert args[1][-1] == split_k_slices
    assert kwargs["cluster_shape_k"] == split_k_slices
    assert kwargs["cache_module_name"] == "mm_mxfp8"
    assert kwargs["disk_kernel_name_fn"] is _mxfp8_blockscaled_kernel_disk_name
    assert kwargs["cache_key_files_fn"] is _blockscaled_mxfp8_gemm_cache_key_files
