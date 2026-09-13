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
    _direct_micro_kernel_cache_key,
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


# ---------------------------------------------------------------------------
# Direct micro kernel
#
# Deliberately not an ADOPTERS row. The three MMA kernels key on their getter's
# arguments, so a key function can be introspected against the getter's
# signature. The direct micro kernel keys on what ``configure()`` *derived*
# from those arguments (``_ShapeConfig``, ``m_const``, ``m1_fc2_onepass``),
# so the contract is checked by building real kernels and perturbing inputs.
#
# ``max_active_ctas`` is passed explicitly throughout: it is the only part of
# ``configure()`` that would otherwise query the current device, so these stay
# CPU-only tests of the naming contract.
# ---------------------------------------------------------------------------

DIRECT_MICRO_BASELINE = {
    "weight_E": 64,
    "m": 4,
    "k": 512,
    "n": 256,
    "num_topk": 2,
    "activation": "silu",
    "share_input_across_experts": False,
    "share_expert_scales": False,
    "single_token": False,
    "max_active_ctas": 148,
}

DIRECT_MICRO_PERTURBED = {
    "weight_E": 32,
    "m": 1,
    "k": 1024,
    "n": 512,
    "num_topk": 4,
    "activation": "relu2",
    "share_input_across_experts": True,
    "share_expert_scales": True,
    "single_token": True,
    "max_active_ctas": 84,
}


def _direct_micro_name(**overrides):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_direct_micro_kernel import (
        build_direct_micro_kernel,
    )

    kwargs = dict(DIRECT_MICRO_BASELINE)
    kwargs.update(overrides)
    topk_ids_dtype = kwargs.pop("topk_ids_dtype", torch.int32)
    kernel = build_direct_micro_kernel(**kwargs)
    return _disk_kernel_name(
        "direct_micro", _direct_micro_kernel_cache_key(kernel, topk_ids_dtype)
    )


# ``max_active_ctas`` only reaches codegen through ``m1_fc2_onepass``, which
# ``configure()`` gates on ``m == 1``. At the baseline's m=4 it is therefore
# genuinely non-codegen (grid_x itself is a runtime argument), so it gets the
# dedicated m=1 test below instead of a slot in the perturbation sweep.
DIRECT_MICRO_SWEEP_PARAMS = sorted(set(DIRECT_MICRO_BASELINE) - {"max_active_ctas"})


@pytest.mark.parametrize("param", DIRECT_MICRO_SWEEP_PARAMS)
def test_direct_micro_name_varies_with_every_argument(param):
    """Changing any single codegen argument must change the on-disk name."""
    baseline = _direct_micro_name()
    perturbed = _direct_micro_name(**{param: DIRECT_MICRO_PERTURBED[param]})
    assert perturbed != baseline, (
        f"the direct micro kernel's on-disk name ignores argument {param!r}: "
        "two different kernel specializations would collide on one artifact."
    )


def test_direct_micro_name_varies_with_max_active_ctas_at_m1():
    """At m=1 the cluster budget reaches codegen and must be keyed.

    ``m1_fc2_onepass = m == 1 and grid_x >= fc2_tasks`` is a compile-time
    constant, and grid_x is capped by the cluster budget. With k=512 the FC2
    task count is 512 // (2 * _K_PER_CTA) == 16, so a budget above and below
    that flips the flag.

    This is the device-dependent part of the key: two SM120 parts with
    different SM counts resolve different budgets while sharing the ``sm120a``
    module directory, so a name ignoring it would serve one part's binary to
    the other.
    """
    onepass = _direct_micro_name(m=1, single_token=True, max_active_ctas=148)
    multipass = _direct_micro_name(m=1, single_token=True, max_active_ctas=8)
    assert onepass != multipass


def test_direct_micro_name_varies_with_topk_ids_dtype():
    """topk_ids dtype is a compile-time pointer type, not a runtime value."""
    assert _direct_micro_name(topk_ids_dtype=torch.int64) != _direct_micro_name()


def test_direct_micro_name_ignores_fast_math():
    """``fast_math`` is accept-and-ignore for this kernel.

    ``MoEDirectMicroKernel.__init__`` does a literal ``del fast_math``, so it
    reaches no codegen decision. Pinning that here means the day it starts
    mattering, this test fails instead of the cache silently serving the wrong
    binary.
    """
    assert _direct_micro_name(fast_math=True) == _direct_micro_name(fast_math=False)


def test_direct_micro_name_is_symbol_safe():
    assert re.fullmatch(r"[0-9A-Za-z_]+", _direct_micro_name())


def test_direct_micro_name_is_stable_for_equal_keys():
    """Two identically configured kernels must map to one artifact name.

    ``__cache_key__`` holds a ``_ShapeConfig`` dataclass; this fails if it ever
    grows a field whose repr is address-based, which would make every process
    miss the cache it just populated.
    """
    assert _direct_micro_name() == _direct_micro_name()


def test_direct_micro_does_not_collide_with_mma_kernels():
    """The direct micro artifact must not share a name with the MMA kernels."""
    mma_names = {
        _disk_kernel_name(label, key_fn(**baseline))
        for label, _, key_fn, baseline, _ in ADOPTERS
    }
    assert _direct_micro_name() not in mma_names


def test_direct_micro_uses_a_separate_disk_module():
    """Direct micro must not share moe_dispatch's ``b12x_moe`` module.

    The module ``meta.json`` records one ``source_sha256`` for every kernel in
    its directory and a mismatch wipes the directory. Two adopters sharing a
    module while hashing different source lists would invalidate each other on
    every call, so this guards the separation rather than the name itself.
    """
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_direct_micro_kernel as dm,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    assert dm._CUTE_DSL_MODULE != moe_dispatch._CUTE_DSL_MODULE
    assert dm.__file__ in dm._kernel_source_files()


def test_direct_micro_probe_roundtrip(tmp_path, monkeypatch):
    """The persisted probe result must survive a process boundary.

    The launchable-block-dim probe reads ``cute.compile`` internals that a
    TVM-FFI ``.o`` reload does not expose, so a warm start reads this sidecar
    instead of re-probing. If it did not round-trip, every process after the
    first would fall back to the MMA micro kernel -- a silent perf regression
    that the naming tests above cannot see.
    """
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_direct_micro_kernel as dm,
    )

    monkeypatch.setattr(dm, "_probe_path", lambda name: tmp_path / f"{name}.json")

    assert dm._read_probe("kernel_a") is None  # absent reads as unknown
    dm._write_probe("kernel_a", True)
    assert dm._read_probe("kernel_a") is True
    dm._write_probe("kernel_b", False)
    assert dm._read_probe("kernel_b") is False
