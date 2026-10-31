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
"""

from types import SimpleNamespace

import pytest
from packaging.version import Version

SSU_SM100 = "gen_selective_state_update_sm100_module"
# 3 dtype combos x 1 dim x 1 dstate x 4 ntokens x 2 cu_seqlens x 2 num_accepted
SSU_SPECS_PER_ARCH = 48
BLACKWELL_ARCHS = [
    "sm100",
    "sm100f",
    "sm103",
    "sm107",
    "sm110",
    "sm120",
    "sm120f",
    "sm121",
]


def _run_gen_all_modules(monkeypatch, sm_capabilities, overrides=None):
    """Run gen_all_modules with every module generator stubbed to its own name."""
    from flashinfer import aot
    from flashinfer.jit import comm as jit_comm

    for module in (aot, jit_comm):
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
    for attr, stub in (overrides or {}).items():
        monkeypatch.setattr(aot, attr, stub)
    monkeypatch.setattr(aot, "_gen_blackwell_bf16_bmm_aot_specs", lambda caps: [])
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [False],
        [],
        sm_capabilities,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
    )
    return {spec.name for spec in specs}


def _counting_stub(calls):
    def gen(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(name=f"{SSU_SM100}_{len(calls)}")

    return gen


@pytest.mark.parametrize("arch", BLACKWELL_ARCHS)
def test_blackwell_build_registers_ssu_sm100(monkeypatch, arch):
    # The runtime dispatches SM major >= 10 to the _sm100 SSU module, so a
    # jit-cache built for any of these archs must ship it.
    calls = []
    _run_gen_all_modules(monkeypatch, {arch: True}, {SSU_SM100: _counting_stub(calls)})

    assert len(calls) == SSU_SPECS_PER_ARCH


@pytest.mark.parametrize("arch", ["sm80", "sm90"])
def test_pre_blackwell_build_skips_ssu_sm100(monkeypatch, arch):
    registered = _run_gen_all_modules(monkeypatch, {arch: True})

    assert SSU_SM100 not in registered


def test_combined_blackwell_build_generates_ssu_sm100_once(monkeypatch):
    # One loop covers all Blackwell archs; a multi-arch build must not
    # register the 48 specs more than once.
    calls = []
    caps = {arch: True for arch in BLACKWELL_ARCHS}
    _run_gen_all_modules(monkeypatch, caps, {SSU_SM100: _counting_stub(calls)})

    assert len(calls) == SSU_SPECS_PER_ARCH
