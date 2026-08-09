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

Naming-contract tests for MoE CuTe-DSL disk cache adopters.

Replicates the contract enforced for the earlier adopters in
``tests/jit/test_cute_dsl_cache.py``, as the design doc's rollout note asks
of new adopters. The kernel-name string is the sole per-kernel cache key --
the module ``meta.json`` guards only arch / DSL version / source hashes -- so
a name that ignores a codegen parameter makes two different kernels collide
on one artifact and the cache silently serves the wrong binary.

1. Signature coverage: every parameter of each kernel getter is expressible
   in the corresponding cache-key function.
2. Per-argument perturbation: changing any single argument changes the name.
3. Symbol safety: names are valid filename/symbol components as produced.
"""

import inspect
import re

import pytest

pytest.importorskip("cutlass")

import torch  # noqa: E402

from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import (  # noqa: E402
    _disk_kernel_name as _w4a16_disk_kernel_name,
    _get_compiled_kernel as _get_compiled_w4a16_kernel,
    _w4a16_kernel_cache_key,
)
from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501, E402
    _disk_kernel_name as _gather_disk_kernel_name,
    _gather_kernel_cache_key,
    _get_compiled_gather_kernel,
)
from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501, E402
    _disk_kernel_name as _finalize_disk_kernel_name,
    _finalize_kernel_cache_key,
    _get_compiled_finalize_kernel,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: E402
    _disk_kernel_name as _b12x_disk_kernel_name,
    _dynamic_kernel_cache_key,
    _get_dynamic_kernel,
    _get_micro_kernel,
    _get_static_kernel,
    _micro_kernel_cache_key,
    _static_kernel_cache_key,
)
from flashinfer.tllm_enums import ActivationType  # noqa: E402

STATIC_BASELINE = {
    "activation_precision": "fp4",
    "state_E": 32,
    "weight_E": 32,
    "m": 64,
    "k": 2048,
    "n": 1024,
    "num_topk": 4,
    "max_rows": 256,
    "mac": 48,
    "mma_tiler_mn": (128, 128),
    "topk_ids_dtype": torch.int32,
    "input_scales_are_reciprocal": False,
    "fast_math": True,
    "activation": "silu",
    "swiglu_alpha": 1.702,
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
}
STATIC_PERTURBED = {
    "activation_precision": "w4a16",
    "state_E": 16,
    "weight_E": 16,
    "m": 128,
    "k": 4096,
    "n": 2048,
    "num_topk": 2,
    "max_rows": 512,
    "mac": 96,
    "mma_tiler_mn": (128, 64),  # not a transpose of the baseline square tile
    "topk_ids_dtype": torch.int64,
    "input_scales_are_reciprocal": True,
    "fast_math": False,
    "activation": "gelu",
    "swiglu_alpha": -1.702,  # sign flip: sanitized text alone would collide
    "swiglu_beta": 2.0,
    "swiglu_limit": 7.0,
}

MICRO_BASELINE = {
    k: v for k, v in STATIC_BASELINE.items() if k != "activation_precision"
}
MICRO_BASELINE.update(
    share_input_across_experts=False, share_expert_scales=False, single_token=False
)
MICRO_PERTURBED = {
    k: v for k, v in STATIC_PERTURBED.items() if k != "activation_precision"
}
MICRO_PERTURBED.update(
    share_input_across_experts=True, share_expert_scales=True, single_token=True
)

DYNAMIC_BASELINE = {
    "activation_precision": "fp4",
    "E": 32,
    "k": 2048,
    "n": 1024,
    "num_topk": 4,
    "mac": 48,
    "mma_tiler_mn": (128, 128),
    "topk_ids_dtype": torch.int32,
    "input_scales_are_reciprocal": False,
    "fast_math": True,
    "activation": "silu",
    "swiglu_alpha": 1.702,
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
    "share_input_across_experts": False,
}
DYNAMIC_PERTURBED = {
    "activation_precision": "w4a16",
    "E": 16,
    "k": 4096,
    "n": 2048,
    "num_topk": 2,
    "mac": 96,
    "mma_tiler_mn": (128, 64),
    "topk_ids_dtype": torch.int64,
    "input_scales_are_reciprocal": True,
    "fast_math": False,
    "activation": "gelu",
    "swiglu_alpha": -1.702,
    "swiglu_beta": 2.0,
    "swiglu_limit": 7.0,
    "share_input_across_experts": True,
}

W4A16_BASELINE = {
    "num_local_experts": 32,
    "activation_type": ActivationType.Swiglu,
    "swiglu_alpha": 1.0,
    "swiglu_beta": 0.0,
    "swiglu_limit": 7.0,
    "use_fused_finalize": False,
    "enable_pdl": True,
    "use_clc_scheduler": False,
    "mma_tiler_mnk": (128, 64, 256),
    "cluster_shape_mn": (1, 1),
    "raster_along_m": True,
    "transform_fragment_size": 128,
    "max_active_clusters": 48,
}
W4A16_PERTURBED = {
    "num_local_experts": 16,
    "activation_type": ActivationType.Relu2,
    "swiglu_alpha": -1.0,  # sign flip catches sanitize-only naming schemes
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
    "use_fused_finalize": True,
    "enable_pdl": False,
    "use_clc_scheduler": True,
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mn": (2, 1),
    "raster_along_m": False,
    "transform_fragment_size": 32,
    "max_active_clusters": 96,
}

W4A4_GATHER_BASELINE = {
    "ab_dtype": "float4_e2m1fn",
    "sf_dtype": "float8_e4m3fn",
    "c_dtype": "bfloat16",
    "sf_vec_size": 16,
    "tile_size": 128,
    "topk": 4,
    "mma_tiler_mn": (128, 128),
    "cluster_shape_mn": (1, 1),
    "vectorized_f32": True,
    "raster_along_m": False,
    "enable_pdl": True,
    "activation_type": ActivationType.Swiglu.value,
    "swiglu_alpha": 1.0,
    "swiglu_beta": 0.0,
    "swiglu_limit": 7.0,
    "situ_beta": None,
    "situ_linear_beta": None,
    "gated": True,
    "use_a_per_token_scale": False,
    "has_c_sf": False,
    "has_norm_const": False,
    "has_a_per_token_scale": False,
    "max_active_clusters": 48,
}
W4A4_GATHER_PERTURBED = {
    "ab_dtype": "bfloat16",
    "sf_dtype": "float32",
    "c_dtype": "float4_e2m1fn",
    "sf_vec_size": 32,
    "tile_size": 256,
    "topk": 8,
    "mma_tiler_mn": (256, 256),
    "cluster_shape_mn": (2, 1),
    "vectorized_f32": False,
    "raster_along_m": True,
    "enable_pdl": False,
    "activation_type": ActivationType.Relu2.value,
    "swiglu_alpha": -1.0,  # sign flip catches sanitize-only naming schemes
    "swiglu_beta": 1.0,
    "swiglu_limit": None,
    "situ_beta": 1.0,
    "situ_linear_beta": 0.5,
    "gated": False,
    "use_a_per_token_scale": True,
    "has_c_sf": True,
    "has_norm_const": True,
    "has_a_per_token_scale": True,
    "max_active_clusters": 96,
}

W4A4_FINALIZE_BASELINE = {
    "ab_dtype": "float4_e2m1fn",
    "sf_dtype": "float8_e4m3fn",
    "out_dtype": "bfloat16",
    "token_scales_dtype": "float32",
    "sf_vec_size": 16,
    "tile_size": 128,
    "mma_tiler_mn": (128, 128),
    "cluster_shape_mn": (1, 1),
    "raster_along_m": False,
    "enable_pdl": True,
    "use_a_per_token_scale": False,
    "use_fused_finalize": True,
    "has_a_per_token_scale": False,
    "max_active_clusters": 48,
}
W4A4_FINALIZE_PERTURBED = {
    "ab_dtype": "bfloat16",
    "sf_dtype": "float32",
    "out_dtype": "float16",
    "token_scales_dtype": "float16",
    "sf_vec_size": 32,
    "tile_size": 256,
    "mma_tiler_mn": (256, 256),
    "cluster_shape_mn": (2, 1),
    "raster_along_m": True,
    "enable_pdl": False,
    "use_a_per_token_scale": True,
    "use_fused_finalize": False,
    "has_a_per_token_scale": True,
    "max_active_clusters": 96,
}

ADOPTERS = [
    (
        "b12x_static",
        _get_static_kernel,
        _static_kernel_cache_key,
        _b12x_disk_kernel_name,
        STATIC_BASELINE,
        STATIC_PERTURBED,
    ),
    (
        "b12x_micro",
        _get_micro_kernel,
        _micro_kernel_cache_key,
        _b12x_disk_kernel_name,
        MICRO_BASELINE,
        MICRO_PERTURBED,
    ),
    (
        "b12x_dynamic",
        _get_dynamic_kernel,
        _dynamic_kernel_cache_key,
        _b12x_disk_kernel_name,
        DYNAMIC_BASELINE,
        DYNAMIC_PERTURBED,
    ),
    (
        "sm100_w4a16",
        _get_compiled_w4a16_kernel,
        _w4a16_kernel_cache_key,
        _w4a16_disk_kernel_name,
        W4A16_BASELINE,
        W4A16_PERTURBED,
    ),
    (
        "sm100_w4a4_gather",
        _get_compiled_gather_kernel,
        _gather_kernel_cache_key,
        _gather_disk_kernel_name,
        W4A4_GATHER_BASELINE,
        W4A4_GATHER_PERTURBED,
    ),
    (
        "sm100_w4a4_finalize",
        _get_compiled_finalize_kernel,
        _finalize_kernel_cache_key,
        _finalize_disk_kernel_name,
        W4A4_FINALIZE_BASELINE,
        W4A4_FINALIZE_PERTURBED,
    ),
]

# Getter parameters absent from a key function because those kernels genuinely
# do not specialize on them. The B12x dynamic kernel takes runtime-shaped
# operands as pointers, while the SM100 W4A4 kernels export runtime dimensions
# and ordinary pointers through TVM-FFI. Optional-pointer presence and pointer
# dtypes remain keyed separately because they do affect the exported ABI.
KEY_OMISSIONS = {
    # ``mac_override`` only selects the keyed ``mac`` value. Dynamic ``tile_m``
    # likewise becomes the keyed ``mma_tiler_mn[0]``.
    "b12x_static": {"mac_override"},
    "b12x_micro": {"mac_override"},
    "b12x_dynamic": {"m", "max_rows", "tile_m"},
    "sm100_w4a4_gather": {
        "orig_m",
        "permuted_m",
        "n",
        "k",
        "num_experts",
        "a_ptr",
        "b_ptr",
        "a_sf_ptr",
        "b_sf_ptr",
        "c_ptr",
        "c_sf_ptr",
        "alpha_ptr",
        "tile_idx_ptr",
        "mn_limit_ptr",
        "token_id_ptr",
        "num_tiles_ptr",
        "norm_const_ptr",
        "a_per_token_scale_ptr",
    },
    "sm100_w4a4_finalize": {
        "seq_len",
        "permuted_m",
        "n",
        "k",
        "num_experts",
        "topk",
        "a_ptr",
        "b_ptr",
        "a_sf_ptr",
        "b_sf_ptr",
        "c_ptr",
        "alpha_ptr",
        "tile_idx_ptr",
        "mn_limit_ptr",
        "permuted_idx_ptr",
        "num_tiles_ptr",
        "token_scales_ptr",
        "a_per_token_scale_ptr",
    },
}


@pytest.mark.parametrize("label,getter,key_fn", [(a[0], a[1], a[2]) for a in ADOPTERS])
def test_key_signature_covers_getter_params(label, getter, key_fn):
    """Every kernel-getter parameter must be expressible in the cache key.

    Fails the moment a parameter is added to a getter without threading it
    into the key (and therefore into the on-disk artifact name).
    """
    getter_params = set(inspect.signature(getter).parameters)
    key_params = set(inspect.signature(key_fn).parameters)
    missing = getter_params - key_params - KEY_OMISSIONS.get(label, set())
    assert not missing, (
        f"{getter.__name__} has codegen parameter(s) {sorted(missing)} that "
        f"{key_fn.__name__} cannot encode. Add them to the key function (or, "
        "if provably non-codegen, to KEY_OMISSIONS with a justification)."
    )


@pytest.mark.parametrize(
    "key_fn",
    [
        _gather_kernel_cache_key,
        _finalize_kernel_cache_key,
        _w4a16_kernel_cache_key,
    ],
)
def test_sm100_keys_include_max_active_clusters(key_fn):
    """SM100 wrappers specialize ``max_active_clusters`` as ``Constexpr``."""
    assert "max_active_clusters" in inspect.signature(key_fn).parameters


@pytest.mark.parametrize(
    "key_fn,abi_params",
    [
        (
            _gather_kernel_cache_key,
            {
                "ab_dtype",
                "sf_dtype",
                "c_dtype",
                "has_c_sf",
                "has_norm_const",
                "has_a_per_token_scale",
                "use_a_per_token_scale",
            },
        ),
        (
            _finalize_kernel_cache_key,
            {
                "ab_dtype",
                "sf_dtype",
                "out_dtype",
                "token_scales_dtype",
                "has_a_per_token_scale",
                "use_a_per_token_scale",
            },
        ),
    ],
)
def test_sm100_w4a4_keys_cover_pointer_abi(key_fn, abi_params):
    """Pointer dtypes and optional-pointer presence define the TVM-FFI ABI."""
    assert abi_params <= set(inspect.signature(key_fn).parameters)


@pytest.mark.parametrize(
    "label,key_fn,name_fn,baseline,perturbed,param",
    [(a[0], a[2], a[3], a[4], a[5], p) for a in ADOPTERS for p in sorted(a[4])],
)
def test_disk_name_varies_with_every_argument(
    label, key_fn, name_fn, baseline, perturbed, param
):
    """Changing any single codegen argument must change the on-disk name."""
    baseline_name = name_fn(label, key_fn(**baseline))
    kwargs = dict(baseline)
    kwargs[param] = perturbed[param]
    perturbed_name = name_fn(label, key_fn(**kwargs))
    assert perturbed_name != baseline_name, (
        f"the {label} kernel's on-disk name ignores argument {param!r}: two "
        "different kernel specializations would collide on one cache artifact."
    )


@pytest.mark.parametrize(
    "label,key_fn,name_fn,baseline", [(a[0], a[2], a[3], a[4]) for a in ADOPTERS]
)
def test_disk_name_is_symbol_safe(label, key_fn, name_fn, baseline):
    """Names must already be valid symbol/filename components.

    ``JitSpecCuteDsl`` sanitizes names before use; a name relying on that
    sanitization could collide with a different name that sanitizes to the
    same string, so the raw name must not need it.
    """
    name = name_fn(f"{label}_m64_k2048", key_fn(**baseline))
    assert re.fullmatch(r"[0-9A-Za-z_]+", name), name


@pytest.mark.parametrize(
    "label,key_fn,name_fn,baseline", [(a[0], a[2], a[3], a[4]) for a in ADOPTERS]
)
def test_disk_name_is_stable_for_equal_keys(label, key_fn, name_fn, baseline):
    """The same key must map to the same artifact name within a process.

    Guards against a name derived from anything unstable (object identity,
    iteration order); without this the cache would never hit.
    """
    first = name_fn(label, key_fn(**baseline))
    second = name_fn(label, key_fn(**dict(baseline)))
    assert first == second


def test_kernel_types_do_not_collide():
    """Distinct kernel families must never share an artifact name."""
    names = {
        name_fn("moe", key_fn(**baseline))
        for _, _, key_fn, name_fn, baseline, _ in ADOPTERS
    }
    assert len(names) == len(ADOPTERS)
