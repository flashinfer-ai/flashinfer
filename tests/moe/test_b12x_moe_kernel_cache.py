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

Naming-contract tests for the b12x MoE CuTe-DSL disk cache adopter.

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

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (  # noqa: E402
    _disk_kernel_name,
    _dynamic_kernel_cache_key,
    _get_dynamic_kernel,
    _get_micro_kernel,
    _get_static_kernel,
    _micro_kernel_cache_key,
    _static_kernel_cache_key,
)

# Getter parameters that deliberately do NOT participate in the cache key.
#
# Each of these only *selects* a value that is itself keyed, so the artifact is
# keyed on what reaches codegen rather than on how it was chosen:
#
# * ``mac_override`` selects the max-active-clusters value, and the resulting
#   ``mac`` is part of every key;
# * ``tile_m`` (dynamic kernel) becomes ``mma_tiler_mn[0]``, and
#   ``mma_tiler_mn`` is part of every key.
NON_CODEGEN_PARAMS = {"mac_override", "tile_m"}

STATIC_BASELINE = {
    "activation_precision": "fp4",
    "quant_mode": "nvfp4",
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
    "quant_mode": "mxfp4",
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
    "quant_mode": "nvfp4",
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
    "quant_mode": "mxfp4",
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

ADOPTERS = [
    (
        "static",
        _get_static_kernel,
        _static_kernel_cache_key,
        STATIC_BASELINE,
        STATIC_PERTURBED,
    ),
    (
        "micro",
        _get_micro_kernel,
        _micro_kernel_cache_key,
        MICRO_BASELINE,
        MICRO_PERTURBED,
    ),
    (
        "dynamic",
        _get_dynamic_kernel,
        _dynamic_kernel_cache_key,
        DYNAMIC_BASELINE,
        DYNAMIC_PERTURBED,
    ),
]

# Getter parameters absent from a key function because that kernel genuinely
# does not specialize on them: the dynamic kernel takes its runtime-shaped
# operands as pointers, so one artifact serves every m / max_rows.
KEY_OMISSIONS = {"dynamic": {"m", "max_rows"}}


@pytest.mark.parametrize("label,getter,key_fn", [(a[0], a[1], a[2]) for a in ADOPTERS])
def test_key_signature_covers_getter_params(label, getter, key_fn):
    """Every kernel-getter parameter must be expressible in the cache key.

    Fails the moment a parameter is added to a getter without threading it
    into the key (and therefore into the on-disk artifact name).
    """
    getter_params = set(inspect.signature(getter).parameters)
    key_params = set(inspect.signature(key_fn).parameters)
    missing = (
        getter_params
        - key_params
        - NON_CODEGEN_PARAMS
        - KEY_OMISSIONS.get(label, set())
    )
    assert not missing, (
        f"{getter.__name__} has codegen parameter(s) {sorted(missing)} that "
        f"{key_fn.__name__} cannot encode. Add them to the key function (or, "
        "if provably non-codegen, to NON_CODEGEN_PARAMS / KEY_OMISSIONS with a "
        "justification)."
    )


@pytest.mark.parametrize(
    "label,key_fn,baseline,perturbed,param",
    [(a[0], a[2], a[3], a[4], p) for a in ADOPTERS for p in sorted(a[3])],
)
def test_disk_name_varies_with_every_argument(
    label, key_fn, baseline, perturbed, param
):
    """Changing any single codegen argument must change the on-disk name."""
    baseline_name = _disk_kernel_name(label, key_fn(**baseline))
    kwargs = dict(baseline)
    kwargs[param] = perturbed[param]
    perturbed_name = _disk_kernel_name(label, key_fn(**kwargs))
    assert perturbed_name != baseline_name, (
        f"the {label} kernel's on-disk name ignores argument {param!r}: two "
        "different kernel specializations would collide on one cache artifact."
    )


@pytest.mark.parametrize(
    "label,key_fn,baseline", [(a[0], a[2], a[3]) for a in ADOPTERS]
)
def test_disk_name_is_symbol_safe(label, key_fn, baseline):
    """Names must already be valid symbol/filename components.

    ``JitSpecCuteDsl`` sanitizes names before use; a name relying on that
    sanitization could collide with a different name that sanitizes to the
    same string, so the raw name must not need it.
    """
    name = _disk_kernel_name(f"{label}_m64_k2048", key_fn(**baseline))
    assert re.fullmatch(r"[0-9A-Za-z_]+", name), name


@pytest.mark.parametrize(
    "label,key_fn,baseline", [(a[0], a[2], a[3]) for a in ADOPTERS]
)
def test_disk_name_is_stable_for_equal_keys(label, key_fn, baseline):
    """The same key must map to the same artifact name within a process.

    Guards against a name derived from anything unstable (object identity,
    iteration order); without this the cache would never hit.
    """
    first = _disk_kernel_name(label, key_fn(**baseline))
    second = _disk_kernel_name(label, key_fn(**dict(baseline)))
    assert first == second


def test_kernel_types_do_not_collide():
    """The three kernel families must never share an artifact name."""
    names = {
        _disk_kernel_name(label, key_fn(**baseline))
        for label, _, key_fn, baseline, _ in ADOPTERS
    }
    assert len(names) == len(ADOPTERS)
