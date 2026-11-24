"""Host admission for the retained NVFP4 multicast barrier."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification import comm

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("enabled,no_dist", [(False, False), (True, True)])
def test_disabled_mapping_never_initializes_nvls(monkeypatch, enabled, no_dist):
    monkeypatch.setenv("MEGA_MULTIMEM_BARRIER", str(int(enabled)))
    monkeypatch.setenv("MEGA_NO_DIST", str(int(no_dist)))
    monkeypatch.setitem(sys.modules, "nvshmem.core", None)
    assert comm._multicast_base(object()) == 0


@pytest.mark.parametrize("mapping_fails", [False, True])
def test_mapping_success_and_fallback(monkeypatch, mapping_fails):
    monkeypatch.setenv("MEGA_MULTIMEM_BARRIER", "1")
    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    tensor, team, pointer = object(), object(), 0x123456789ABC
    calls = []

    def map_tensor(actual_team, actual_tensor):
        calls.append((actual_team, actual_tensor))
        if mapping_fails:
            raise RuntimeError("mapping unavailable")
        return SimpleNamespace(data_ptr=lambda: pointer)

    package = ModuleType("nvshmem")
    core = ModuleType("nvshmem.core")
    core.Teams = SimpleNamespace(TEAM_WORLD=team)
    core.get_multicast_tensor = map_tensor
    package.core = core
    monkeypatch.setitem(sys.modules, "nvshmem", package)
    monkeypatch.setitem(sys.modules, "nvshmem.core", core)
    if mapping_fails:
        with pytest.warns(RuntimeWarning, match="falling back"):
            assert comm._multicast_base(tensor) == 0
    else:
        assert comm._multicast_base(tensor) == pointer
    assert calls == [(team, tensor)]
