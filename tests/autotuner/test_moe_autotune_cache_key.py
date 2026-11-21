"""CPU-only tests for TRTLLM MoE persistent autotune cache identity."""

from unittest.mock import MagicMock

import pytest

from flashinfer.autotuner import AutoTuner, TuningConfig
from flashinfer.fused_moe.runners import (
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
)
from flashinfer.tllm_enums import DtypeTrtllmGen, Fp8QuantizationType


_CUSTOM_OP = "flashinfer::trtllm_moe"
_PROFILE_SHAPES = ((128, 4096), (128, 8))
_TUNING_CONFIG = TuningConfig()


@pytest.fixture
def moe_runner_cls(monkeypatch):
    import flashinfer.fused_moe.core as core_mod

    get_module = core_mod.get_trtllm_moe_sm100_module
    get_module.cache_clear()

    mock_module = MagicMock()
    mock_module.get_library_path.return_value = "/tmp/fake.so"
    monkeypatch.setattr(
        core_mod,
        "gen_trtllm_gen_fused_moe_sm100_module",
        lambda: mock_module,
    )
    monkeypatch.setattr(core_mod, "setup_cubin_loader", lambda _: None)

    try:
        yield get_module().MoERunner
    finally:
        get_module.cache_clear()


@pytest.fixture
def tuner(monkeypatch):
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    AutoTuner._instance = None
    instance = AutoTuner.get()
    try:
        yield instance
    finally:
        AutoTuner._instance = None


def _make_runner(
    runner_cls,
    *,
    num_experts: int = 256,
    num_local_experts: int = 32,
    top_k: int = 8,
    num_fused_shared_experts: int = 0,
):
    return runner_cls(
        top_k=top_k,
        num_local_experts=num_local_experts,
        dtype_act=DtypeTrtllmGen.Bfloat16,
        dtype_weights=DtypeTrtllmGen.Bfloat16,
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=4096,
        intermediate_size=14336,
        num_experts=num_experts,
        num_fused_shared_experts=num_fused_shared_experts,
    )


@pytest.mark.parametrize(
    "changed_geometry",
    [
        {"num_experts": 128},
        {"num_local_experts": 64},
        {"top_k": 4},
        {"num_fused_shared_experts": 1},
    ],
    ids=["global-experts", "local-experts", "top-k", "fused-shared-experts"],
)
def test_moe_geometry_separates_persistent_cache_entries(
    moe_runner_cls, tuner, changed_geometry
):
    runner_a = _make_runner(moe_runner_cls)
    runner_b = _make_runner(moe_runner_cls, **changed_geometry)
    inputs = []

    key_a = AutoTuner._get_cache_key(
        _CUSTOM_OP,
        runner_a,
        _PROFILE_SHAPES,
        _TUNING_CONFIG,
        runner_a.get_cache_key_extras(inputs),
    )
    key_b = AutoTuner._get_cache_key(
        _CUSTOM_OP,
        runner_b,
        _PROFILE_SHAPES,
        _TUNING_CONFIG,
        runner_b.get_cache_key_extras(inputs),
    )

    assert key_a.custom_op == key_b.custom_op
    assert key_a.runner_class_name == key_b.runner_class_name
    assert key_a.nearest_profile == key_b.nearest_profile
    assert key_a.extras != key_b.extras
    assert key_a.file_key != key_b.file_key

    tuner._file_configs[key_a.file_key] = (runner_a.__class__.__name__, 42)
    hit_b, _, tactic_b, _ = tuner.search_cache(
        _CUSTOM_OP,
        [runner_b],
        _PROFILE_SHAPES,
        _TUNING_CONFIG,
        inputs=inputs,
    )
    assert not hit_b
    assert tactic_b == -1

    tuner._file_configs[key_b.file_key] = (runner_b.__class__.__name__, 17)
    assert len(tuner._file_configs) == 2

    hit_a, _, tactic_a, _ = tuner.search_cache(
        _CUSTOM_OP,
        [runner_a],
        _PROFILE_SHAPES,
        _TUNING_CONFIG,
        inputs=inputs,
    )
    hit_b, _, tactic_b, _ = tuner.search_cache(
        _CUSTOM_OP,
        [runner_b],
        _PROFILE_SHAPES,
        _TUNING_CONFIG,
        inputs=inputs,
    )
    assert (hit_a, tactic_a) == (True, 42)
    assert (hit_b, tactic_b) == (True, 17)


@pytest.mark.parametrize("runner_cls", [TrtllmFp4RoutedRunner, TrtllmBf16RoutedRunner])
def test_unified_trtllm_runner_delegates_cache_key_extras(runner_cls):
    runner = object.__new__(runner_cls)
    runner._inner = MagicMock()
    runner._inner.get_cache_key_extras.return_value = (256, 32, 8, 0)
    inputs = []

    assert runner.get_cache_key_extras(inputs) == (256, 32, 8, 0)
    runner._inner.get_cache_key_extras.assert_called_once_with(inputs)
