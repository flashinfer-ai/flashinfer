"""Autotune cache-key coverage for the unified MoE runners.

The autotuner keys a profiling result on
``(custom_op, runner_class_name, runner_hash, nearest_profile, extras)``.
For the unified MoE adapters:

* ``custom_op`` is ``f"moe_{backend_key}"`` — constant per backend;
* ``nearest_profile`` is derived from the *profiled tensor list*, which carries
  activations and routing but not the weights (those travel in
  ``_static_kwargs``), so the expert geometry is invisible to it;
* ``ProfilingCacheKey.file_key`` deliberately omits ``runner_hash``, so the
  on-disk cache is keyed by ``extras`` alone.

Together that means a tactic-relevant dimension must appear in
``get_cache_key_extras()`` or two different configurations silently share one
tuned tactic — in memory via ``runner_hash`` and on disk via ``file_key``.
These tests pin that contract for every dimension that reaches
``trtllm_get_valid_moe_configs``.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.autotuner.autotuner import ProfilingCacheKey
from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import (
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
)
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability


def _skip_unless_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, minor = get_compute_capability(torch.device("cuda"))
    if (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip("trtllm-gen MoE runners target SM100/SM103")


# (runner class, backend config, quant variant, alternate quant variant or None)
_RUNNERS = [
    pytest.param(
        TrtllmFp8BlockRunner,
        TrtllmFp8BlockConfig(),
        QuantVariant.DeepSeekFp8,
        QuantVariant.MxFp8,
        id="fp8_block",
    ),
    pytest.param(
        TrtllmFp8PerTensorRunner,
        TrtllmFp8PerTensorConfig(),
        QuantVariant.FP8PerTensor,
        None,
        id="fp8_per_tensor",
    ),
    pytest.param(
        TrtllmBf16RoutedRunner,
        TrtllmBf16Config(),
        QuantVariant.BF16,
        None,
        id="bf16",
    ),
    pytest.param(
        TrtllmFp4RoutedRunner,
        TrtllmFp4Config(),
        QuantVariant.NVFP4,
        QuantVariant.MXFP4,
        id="fp4",
    ),
]


def _config(backend_cfg, variant, **overrides):
    fields = dict(
        num_experts=64,
        top_k=2,
        intermediate_size=256,
        local_expert_offset=0,
        local_num_experts=None,
        activation=ActivationConfig.swiglu,
    )
    fields.update(overrides)
    return MoEConfig(
        routing=RoutingConfig(
            num_experts=fields["num_experts"],
            top_k=fields["top_k"],
            method=RoutingMethodType.DeepSeekV3,
            n_group=8,
            topk_group=4,
            routed_scaling_factor=2.5,
        ),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=fields["intermediate_size"],
            local_expert_offset=fields["local_expert_offset"],
            local_num_experts=fields["local_num_experts"],
        ),
        activation=fields["activation"],
        backend=BackendOptions(candidates=(backend_cfg,)),
    )


# Dimensions that change which tactics are legal but are NOT recoverable from
# the profiled tensor shapes. Each must move both cache keys.
_DIMENSIONS = [
    ("intermediate_size", dict(intermediate_size=512)),
    ("top_k", dict(top_k=4)),
    ("num_experts", dict(num_experts=128)),
    ("local_num_experts", dict(local_num_experts=32)),
    ("local_expert_offset", dict(local_expert_offset=8)),
    ("activation", dict(activation=ActivationConfig.geglu)),
]


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
@pytest.mark.parametrize(
    "dimension,override", _DIMENSIONS, ids=[d[0] for d in _DIMENSIONS]
)
def test_tactic_dimension_changes_both_cache_keys(
    runner_cls, backend_cfg, variant, alt_variant, dimension, override
):
    """A tactic-relevant dimension must separate memory *and* file cache keys."""
    _skip_unless_sm100()
    device = torch.device("cuda")
    base = runner_cls(_config(backend_cfg, variant), device=device)
    other = runner_cls(_config(backend_cfg, variant, **override), device=device)

    assert hash(base) != hash(other), (
        f"{runner_cls.__name__}: {dimension} does not change runner_hash, so two "
        "configurations share one in-memory tuned tactic."
    )
    # file_key serializes extras with str(); compare the same way the autotuner
    # does rather than comparing tuples, so a non-stable repr is caught here.
    assert str(base.get_cache_key_extras([])) != str(other.get_cache_key_extras([])), (
        f"{runner_cls.__name__}: {dimension} does not change get_cache_key_extras(), "
        "so the on-disk cache (which drops runner_hash) collides."
    )


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_quant_variant_changes_both_cache_keys(
    runner_cls, backend_cfg, variant, alt_variant
):
    _skip_unless_sm100()
    if alt_variant is None:
        pytest.skip(f"{runner_cls.__name__} supports a single quant variant")
    device = torch.device("cuda")
    base = runner_cls(_config(backend_cfg, variant), device=device)
    other = runner_cls(_config(backend_cfg, alt_variant), device=device)
    assert hash(base) != hash(other)
    assert str(base.get_cache_key_extras([])) != str(other.get_cache_key_extras([]))


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_identical_config_shares_cache_key(
    runner_cls, backend_cfg, variant, alt_variant
):
    """Two runners built from the same configuration must still share a key.

    Guards the opposite failure: over-keying (e.g. folding in object identity)
    would make every layer re-tune from scratch.
    """
    _skip_unless_sm100()
    device = torch.device("cuda")
    a = runner_cls(_config(backend_cfg, variant), device=device)
    b = runner_cls(_config(backend_cfg, variant), device=device)
    assert hash(a) == hash(b)
    assert str(a.get_cache_key_extras([])) == str(b.get_cache_key_extras([]))


@pytest.mark.parametrize("runner_cls,backend_cfg,variant,alt_variant", _RUNNERS)
def test_cache_key_extras_are_str_stable(runner_cls, backend_cfg, variant, alt_variant):
    """Every element must survive the str() round trip the file cache performs.

    ``ProfilingCacheKey.file_key`` is ``str((custom_op, class, profile, extras))``
    and is written to disk, so an element whose repr embeds an address (or
    otherwise varies per process) would never match on reload.
    """
    _skip_unless_sm100()
    device = torch.device("cuda")
    runner = runner_cls(_config(backend_cfg, variant), device=device)
    extras = runner.get_cache_key_extras([])
    assert isinstance(extras, tuple)
    rendered = str(extras)
    assert "0x" not in rendered, (
        f"{runner_cls.__name__}: cache-key extras contain what looks like an "
        f"object address and will not match across processes: {rendered}"
    )
    for element in extras:
        assert isinstance(element, (int, float, str, bool, tuple)), (
            f"{runner_cls.__name__}: cache-key element {element!r} of type "
            f"{type(element).__name__} has no guaranteed stable repr."
        )


def test_every_registered_runner_defines_stable_extras():
    """No unified runner may fall back to the empty default.

    A backend that inherits ``get_cache_key_extras() -> ()`` from
    ``TunableRunner`` keys its on-disk cache on the profile alone, which cannot
    see the expert geometry.
    """
    assert _BACKEND_RUNNERS, "no unified runners registered"
    for runner_cls in set(_BACKEND_RUNNERS.values()):
        assert "_cache_key_extras" in dir(runner_cls), (
            f"{runner_cls.__name__} does not inherit the unified cache-key "
            "contract from MoERunner."
        )


def test_profiling_cache_key_file_key_separates_configs():
    """The on-disk key must separate configs that share a profile.

    Builds ``ProfilingCacheKey`` directly with an identical
    ``nearest_profile`` — the situation the adapters actually hit, since the
    weights are not in the profiled tensor list — and asserts ``file_key``
    still differs. ``file_key`` drops ``runner_hash``, so this is the check
    that the in-memory assertions above cannot make.
    """
    _skip_unless_sm100()
    device = torch.device("cuda")
    base = TrtllmFp8BlockRunner(
        _config(TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8), device=device
    )
    other = TrtllmFp8BlockRunner(
        _config(
            TrtllmFp8BlockConfig(), QuantVariant.DeepSeekFp8, intermediate_size=512
        ),
        device=device,
    )
    shared_profile = ((64, 256), (64, 2))

    def key_for(runner):
        return ProfilingCacheKey(
            custom_op="moe_trtllm_fp8_block",
            runner_class_name=type(runner).__name__,
            runner_hash=hash(runner),
            nearest_profile=shared_profile,
            extras=runner.get_cache_key_extras([]),
        )

    key_a, key_b = key_for(base), key_for(other)
    assert key_a != key_b
    assert key_a.file_key != key_b.file_key, (
        "identical profiles with different intermediate_size produced the same "
        "file_key; the persisted tactic would be reused across both."
    )
