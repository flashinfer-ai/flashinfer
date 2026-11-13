"""SM90 AOT inventory must cover device-selected MoE modules on any builder."""

from types import SimpleNamespace

import pytest
from packaging.version import Version


@pytest.mark.cpu_only
@pytest.mark.parametrize("runtime_prebuilt_d", [False, True])
def test_sm90_aot_registers_both_moe_variants(monkeypatch, runtime_prebuilt_d):
    from flashinfer import aot
    from flashinfer.jit import comm, fused_moe

    # Skip unrelated source generation while retaining the real SM90 generator
    # and AOT registration path. The builder's runtime choice must not matter.
    sm90_generator = fused_moe.gen_cutlass_fused_moe_sm90_module
    for module in (aot, comm):
        for attr in dir(module):
            if (
                attr.startswith("gen_")
                and "_module" in attr
                and attr != "gen_all_modules"
            ):
                spec = SimpleNamespace(name=attr)
                result = [spec] if attr.endswith("_modules") else spec
                monkeypatch.setattr(
                    module, attr, lambda *args, _result=result, **kwargs: _result
                )
    monkeypatch.setattr(aot, "gen_cutlass_fused_moe_sm90_module", sm90_generator)
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(fused_moe, "is_cuda_version_at_least", lambda version: True)
    monkeypatch.setattr(
        fused_moe,
        "_should_enable_sm90_moe_prebuilt_d_descriptor",
        lambda: runtime_prebuilt_d,
    )

    def capture_spec(flags, arch, fast_build, module_name=None):
        return SimpleNamespace(name=module_name or f"fused_moe_{arch}", flags=flags)

    monkeypatch.setattr(fused_moe, "gen_cutlass_fused_moe_module", capture_spec)
    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [False],
        [],
        {"sm90": True},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    moe = {spec.name: spec for spec in specs if spec.name.startswith("fused_moe_90")}
    assert set(moe) == {"fused_moe_90", "fused_moe_90_h20_prebuilt_d"}
    define = "-DFLASHINFER_SM90_MOE_PREBUILT_D_DESCRIPTOR=1"
    assert define not in moe["fused_moe_90"].flags
    assert define in moe["fused_moe_90_h20_prebuilt_d"].flags


@pytest.mark.cpu_only
@pytest.mark.parametrize("runtime_prebuilt_d", [False, True])
def test_sm90_runtime_keeps_device_selection(monkeypatch, runtime_prebuilt_d):
    from flashinfer.jit import fused_moe

    monkeypatch.setattr(fused_moe, "is_cuda_version_at_least", lambda version: True)
    monkeypatch.setattr(
        fused_moe,
        "_should_enable_sm90_moe_prebuilt_d_descriptor",
        lambda: runtime_prebuilt_d,
    )
    monkeypatch.setattr(
        fused_moe,
        "gen_cutlass_fused_moe_module",
        lambda flags, arch, fast_build, module_name=None: module_name
        or f"fused_moe_{arch}",
    )
    expected = "fused_moe_90_h20_prebuilt_d" if runtime_prebuilt_d else "fused_moe_90"
    assert fused_moe.gen_cutlass_fused_moe_sm90_module() == expected
