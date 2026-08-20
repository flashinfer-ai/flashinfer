"""Backend-specific defaults for distribution-aware MoE configuration."""

from flashinfer.fused_moe.da_config import (
    get_enabled_da_moe_config,
    is_trtllm_da_enabled,
)


def test_da_backend_defaults_when_environment_is_unset(monkeypatch):
    """PrimsTS opts in by default while the TRTLLM compatibility path does not."""
    monkeypatch.delenv("FLASHINFER_DIST_AWARE_AUTOTUNE", raising=False)

    assert get_enabled_da_moe_config(default_enabled=True) is not None
    assert get_enabled_da_moe_config() is None
    assert not is_trtllm_da_enabled()


def test_da_environment_override_applies_to_every_backend(monkeypatch):
    """An explicit environment value overrides either backend's default policy."""
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
    assert get_enabled_da_moe_config(default_enabled=True) is None
    assert get_enabled_da_moe_config() is None

    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
    assert get_enabled_da_moe_config(default_enabled=True) is not None
    assert get_enabled_da_moe_config() is not None
    assert is_trtllm_da_enabled()
