"""Explicit opt-in for experimental distribution-aware MoE configuration."""

from flashinfer.fused_moe.da_config import (
    DaMoeConfig,
    get_enabled_da_moe_config,
    is_trtllm_da_enabled,
)


def test_da_backend_defaults_when_environment_is_unset(monkeypatch):
    """DA remains disabled for both backends unless explicitly requested."""
    monkeypatch.delenv("FLASHINFER_DIST_AWARE_AUTOTUNE", raising=False)

    assert get_enabled_da_moe_config() is None
    assert not is_trtllm_da_enabled()
    assert not DaMoeConfig.from_environment().enabled


def test_da_environment_override_applies_to_every_backend(monkeypatch):
    """An explicit environment value overrides either backend's default policy."""
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
    assert get_enabled_da_moe_config() is None

    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
    assert get_enabled_da_moe_config() is not None
    assert is_trtllm_da_enabled()


def test_prims_ts_dispatch_uses_ordinary_tactic_without_opt_in(monkeypatch):
    from flashinfer.prims_ts.moe.da_runtime import run_prims_ts_da

    monkeypatch.delenv("FLASHINFER_DIST_AWARE_AUTOTUNE", raising=False)
    # Disabled DA must not even parse an experimental distribution catalog.
    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "invalid")
    baseline = (16, 3)
    result = object()

    def ordinary(tactic):
        assert tactic == baseline
        return result

    assert (
        run_prims_ts_da(
            custom_op="test",
            runner=None,
            tuning_config=None,
            inputs=[],
            runner_kwargs={},
            baseline_tactic=baseline,
            routing_input_mode=0,
            num_experts=4,
            local_expert_offset=0,
            num_local_experts=4,
            top_k=2,
            routing_method_type=0,
            routed_scaling_factor=None,
            run_fixed_tactic=ordinary,
            finish_switch=lambda: None,
        )
        is result
    )


def test_baseline_guard_is_opt_in_and_part_of_cache_identity(monkeypatch):
    monkeypatch.delenv("FLASHINFER_DA_BASELINE_GUARD", raising=False)
    unguarded = DaMoeConfig.from_environment()
    assert not unguarded.baseline_guard_enabled
    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "1")
    guarded = DaMoeConfig.from_environment()
    assert guarded.baseline_guard_enabled
    assert guarded.cache_identity() != unguarded.cache_identity()


def test_da_cache_identity_records_switch_minimum_improvement():
    # Plans admitted before near-tie pruning must not restore under the new policy.
    assert (
        DaMoeConfig.from_environment().cache_identity()["switch_minimum_improvement"]
        == 0.01
    )


def test_da_cache_identity_records_guard_profile_order(monkeypatch):
    """Records tuned with a different guard order must not restore as equivalent."""
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")

    config = get_enabled_da_moe_config()

    assert config is not None
    assert config.cache_identity()["guard_profile_order"] == "abba"
