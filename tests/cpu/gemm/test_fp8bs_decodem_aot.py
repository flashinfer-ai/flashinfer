# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from flashinfer import aot


@pytest.mark.parametrize(
    "sm90,cuda,add_moe,want",
    [
        (True, "12.8", True, True),
        (True, "12.9", True, True),
        (True, "12.7", True, False),
        (False, "12.9", True, False),
        (True, "12.9", False, False),
    ],
)
def test_decodem_collection(monkeypatch, sm90, cuda, add_moe, want):
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    for name in tuple(vars(aot)):
        if name.startswith("gen_") and name.endswith("_module"):
            monkeypatch.setattr(
                aot, name, Mock(return_value=SimpleNamespace(name=name))
            )
    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm90": sm90},
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=add_moe,
        add_act=False,
        add_misc=False,
        add_xqa=False,
    )
    target = "gen_fp8bs_gemm_decodem_sm90_module"
    assert getattr(aot, target).call_count == int(want)
    assert sum(spec.name == target for spec in specs) == int(want)
    assert aot.gen_fp8_blockscale_gemm_sm90_module.call_count == int(sm90 and add_moe)


@pytest.mark.parametrize("cached", [False, True])
def test_capture_loads_available_artifact_without_building(monkeypatch, cached):
    import flashinfer.jit.gemm as jit_gemm
    from flashinfer.gemm import fp8bs_decodem_sm90 as mod

    module = object()
    spec = Mock()
    spec.try_load.return_value = module if cached else None
    monkeypatch.setattr(jit_gemm, "gen_fp8bs_gemm_decodem_sm90_module", lambda: spec)
    monkeypatch.setattr(mod.torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(mod, "_stats", {"jit_build": "not attempted"})
    mod._get_module.cache_clear()
    try:
        if cached:
            assert mod._get_module() is module
        else:
            with pytest.raises(mod.GraphCaptureJitError):
                mod._get_module()
        spec.try_load.assert_called_once_with()
        spec.build_and_load.assert_not_called()
    finally:
        mod._get_module.cache_clear()
