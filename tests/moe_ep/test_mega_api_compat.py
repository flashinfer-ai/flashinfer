"""Host regressions for the MegaMoE argument renames."""

import dataclasses
import importlib
import inspect
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.config import FinalizeMode


def _config(arch, dtype):
    module = importlib.import_module(
        f"flashinfer.moe_ep.backends.mega.kernel.{arch}.{dtype}_cutedsl.config"
    )
    name = f"{arch.capitalize()}_{dtype.title()}_Cutedsl_MegaMoeConfig"
    return getattr(module, name)


@pytest.mark.parametrize(
    "arch,dtype",
    [
        ("sm100", "bf16_nvfp4_bf16"),
        ("sm100", "nvfp4_nvfp4_bf16"),
        ("sm107", "mxfp8_mxfp4_bf16"),
        ("sm107", "mxfp8_mxfp8_bf16"),
        ("sm107", "nvfp4_nvfp4_bf16"),
    ],
)
def test_routing_aliases_remain_live(arch, dtype):
    cls = _config(arch, dtype)
    with pytest.warns(DeprecationWarning, match="apply_topk_in_fc1"):
        config = cls(1024, 8, apply_topk_in_fc1=False)
    assert config.apply_topk_in_fc1 is False
    assert config.apply_routing_weights_before_fc2 is False
    config.apply_topk_in_fc1 = True
    assert config.apply_routing_weights_before_fc2 is True
    config.apply_routing_weights_before_fc2 = False
    assert config.apply_topk_in_fc1 is False
    assert dataclasses.replace(
        config, apply_routing_weights_before_fc2=True
    ).apply_topk_in_fc1
    with pytest.warns(DeprecationWarning):
        assert dataclasses.replace(
            config, apply_topk_in_fc1=True
        ).apply_routing_weights_before_fc2
    assert "apply_topk_in_fc1" not in dataclasses.asdict(config)


@pytest.mark.parametrize(
    "dtype", ["bf16_bf16_bf16", "bf16_nvfp4_bf16", "nvfp4_nvfp4_bf16"]
)
def test_finalize_alias_and_old_positional_slot(dtype):
    cls = _config("sm100", dtype)
    assert cls(1024, 8).do_finalize is True
    with pytest.warns(DeprecationWarning, match="do_finalize"):
        config = cls(1024, 8, do_finalize=False)
    assert config.finalize_mode is FinalizeMode.NONE
    assert config.do_finalize is False
    config.do_finalize = True
    assert config.finalize_mode is FinalizeMode.AUTO
    assert (
        dataclasses.replace(config, finalize_mode=FinalizeMode.NONE).do_finalize
        is False
    )
    with pytest.warns(DeprecationWarning):
        assert (
            dataclasses.replace(config, do_finalize=False).finalize_mode
            is FinalizeMode.NONE
        )

    # These are the public positional signatures before the rename.
    if dtype == "bf16_bf16_bf16":
        args = [1024, 8, cls.kernel_name, None, None, True, False, None, False]
    elif dtype == "bf16_nvfp4_bf16":
        args = [
            1024,
            8,
            cls.kernel_name,
            None,
            False,
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            "swiglu",
            None,
            None,
            None,
            False,
        ]
    else:
        args = [
            1024,
            8,
            cls.kernel_name,
            None,
            None,
            "swiglu",
            None,
            None,
            True,
            True,
            False,
            "bf16",
            2.5,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        ]
    with pytest.warns(DeprecationWarning, match="positional do_finalize"):
        config = cls(*args)
    assert config.finalize_mode is FinalizeMode.NONE
    if dtype == "nvfp4_nvfp4_bf16":
        assert config.input_norm_const == 2.5
        assert config.num_valid_tokens_tensor is None


def test_mxfp8_positional_knobs_are_not_finalize_mode():
    cls = _config("sm100", "mxfp8_mxfp8_bf16")
    config = cls(
        1024,
        8,
        cls.kernel_name,
        "mxfp8_e4m3",
        None,
        None,
        True,
        False,
        {},
        "genphase",
        (256, 128, 128),
        (2, 1),
        ("grouped", None),
    )
    assert config.knobs == {}
    assert config.kernel_variant == "genphase"
    assert config.schedule_policy == ("grouped", None)
    assert config.finalize_mode is FinalizeMode.AUTO
    assert config.num_valid_tokens_tensor is None


def test_new_finalize_validation_is_preserved():
    cls = _config("sm100", "nvfp4_nvfp4_bf16")
    with pytest.raises(TypeError, match="FinalizeMode"):
        cls(1024, 8, finalize_mode=False)
    with pytest.raises(ValueError, match="num_valid_tokens_tensor"):
        cls(1024, 8, finalize_mode=FinalizeMode.PERSISTENT)
    assert cls(1024, 8, finalize_mode=FinalizeMode.CAKE).do_finalize is True


_SHIM = "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim"


def test_frozen_shim_config_aliases_and_replace():
    module = importlib.import_module(f"{_SHIM}.nvfp4")
    cls = module.MegaMoENvfp4Config
    with pytest.warns(DeprecationWarning):
        config = cls(
            0,
            1,
            128,
            8,
            32,
            2048,
            1024,
            apply_topk_in_fc1=False,
            defer_topk_reduce=True,
        )
    assert config.apply_topk_in_fc1 is False
    assert config.defer_topk_reduce is True
    assert config.use_custom_finalize is True
    changed = dataclasses.replace(config, use_custom_finalize=False)
    assert changed.defer_topk_reduce is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.apply_topk_in_fc1 = True
    assert (
        module.MegaMoENvfp4Frontend.deferred_topk_reduce_workspace
        is module.MegaMoENvfp4Frontend.custom_finalize_workspace
    )


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("nvfp4", "MegaMoENvfp4Inputs"),
        ("mxfp8", "MegaMoEMxfp8Inputs"),
        ("bf16_mxfp8", "MegaMoEBf16Mxfp8Inputs"),
    ],
)
def test_input_live_token_count_is_optional(module_name, class_name):
    cls = getattr(importlib.import_module(f"{_SHIM}.{module_name}"), class_name)
    inputs = {
        f.name: object()
        for f in dataclasses.fields(cls)
        if f.name != "num_valid_tokens"
    }
    assert cls(**inputs).num_valid_tokens is None


def test_cache_accepts_old_and_new_keywords(tmp_path, monkeypatch):
    cache = importlib.import_module(f"{_SHIM}.knob_cache")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    geometry = dict(
        dtype="nvfp4",
        world_size=1,
        hidden=2048,
        intermediate=1024,
        num_experts=32,
        topk=8,
        max_tokens=128,
        device="test-device",
    )
    knobs = {"in_kernel_fc2_reduce": False}
    with pytest.warns(DeprecationWarning):
        cache.record_knobs(
            knobs, **geometry, apply_topk_in_fc1=True, defer_topk_reduce=True
        )
    assert (
        cache.lookup_knobs(
            **geometry, apply_routing_weights_before_fc2=True, use_custom_finalize=True
        )
        == knobs
    )
    with pytest.warns(DeprecationWarning):
        assert (
            cache.lookup_knobs(
                **geometry, apply_topk_in_fc1=True, defer_topk_reduce=True
            )
            == knobs
        )
    assert cache.lookup_knobs(**geometry, use_custom_finalize=True) is None


@pytest.mark.parametrize(
    "name",
    [
        "compute_megamoe_reference_bf16",
        "compute_megamoe_reference_bf16_mxfp8",
        "compute_megamoe_reference_mxfp8",
    ],
)
def test_reference_accepts_legacy_keyword_without_duplicate_argument(name, monkeypatch):
    helpers = importlib.import_module(f"{_SHIM}.kernel_helpers")
    function = getattr(helpers, name)
    monkeypatch.setattr(
        helpers.importlib,
        "import_module",
        lambda *_: SimpleNamespace(
            **{
                name: lambda **kwargs: kwargs,
                "compute_megamoe_reference": lambda **kwargs: kwargs,
                "compute_megamoe_reference_mxfp8_bf16": lambda **kwargs: kwargs,
            }
        ),
    )
    with pytest.warns(DeprecationWarning):
        assert function(apply_topk_in_fc1=True) == {"apply_topk_in_fc1": True}
    assert function(apply_routing_weights_before_fc2=False) == {
        "apply_topk_in_fc1": False
    }
    assert "apply_routing_weights_before_fc2" in inspect.signature(function).parameters
