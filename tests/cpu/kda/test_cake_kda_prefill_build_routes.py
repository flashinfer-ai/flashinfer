"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The BF16 KDA prefill builds the route tables name, decision by decision.

CPU-only: the loader's ``_factory`` resolves each request of a module through
the route tables; the device architecture is stubbed and the prepared kernel
is recorded instead of compiled.

- Work-binned launches of the N32 and N16 work-binned modules take their
  next-state prefetch builds (``_RENDERED_WORK_BIN_ROUTES``); the same
  requests without work bins keep the module's routed program.
- Owner/helper launches of the direct module take its per-warp checkpoint
  build (``_RENDERED_OWNER_HELPER_ROUTES``); the pooled owner/helper launches
  keep the pooled program.
"""

from types import SimpleNamespace

import pytest

import flashinfer.jit.cake_kda_tf32 as registry

pytestmark = pytest.mark.cpu_only

ARCHES = ["sm_100a", "sm_103a"]
N32 = "cake_kda_bf16_02c6dae5d13691ae05c3"
N16 = "cake_kda_bf16_51e813006f17147a02a5"
DIR = "cake_kda_bf16_893740628d53688ac269"
# The next-state prefetch builds of the work-binned programs.
PREFETCH = {
    N32: "cake_kda_prefill_0cfd846d62326f938704",
    N16: "cake_kda_prefill_cb5c38847fbc141ab226",
}
# The per-warp checkpoint build of the owner/helper program, and the pooled
# owner/helper program.
CHECKPOINT_SLICES = {DIR: "cake_kda_prefill_a1718dac35b1811fc4f0"}
POOLED = {DIR: "cake_kda_prefill_2056bae01a94fb39ddf2"}


@pytest.fixture
def factory(monkeypatch):
    def use(arch):
        monkeypatch.setattr(registry, "device_arch", lambda device=None: arch)
        monkeypatch.setattr(
            registry,
            "PreparedKernel",
            lambda program, arch: SimpleNamespace(program=program, arch=arch),
        )
        return registry._factory

    return use


def _requests(arch, module):
    """Every factory request (family, args, kwargs) that resolves to module."""
    found = [
        (family, args, dict(kwargs))
        for family, variants in registry.FACTORIES[arch].items()
        for (args, kwargs), name in variants.items()
        if name == module
    ]
    assert found, (arch, module)
    return found


def _registered(program, arch):
    record = registry._RENDERED_MODULES[program]
    assert arch in record["arches"], (program, arch)
    for path in record["sources"]:
        assert program in path, (program, path)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("module", [N32, N16], ids=["n32", "n16"])
def test_work_binned_launches_take_the_next_state_prefetch_build(factory, arch, module):
    program = PREFETCH[module]
    assert registry._RENDERED_WORK_BIN_ROUTES[arch][module] == program
    _registered(program, arch)
    build = factory(arch)
    routed = registry._RENDERED_ROUTES[arch].get(module, module)
    assert routed != program
    for family, args, kwargs in _requests(arch, module):
        assert build(family, *args, route_work_bins=True, **kwargs).program == program
        assert build(family, *args, **kwargs).program == routed


@pytest.mark.parametrize("arch", ARCHES)
def test_owner_helper_launches_take_the_per_warp_checkpoint_build(factory, arch):
    build = factory(arch)
    for module, program in CHECKPOINT_SLICES.items():
        assert registry._RENDERED_OWNER_HELPER_ROUTES[arch][module] == program
        pool = registry._RENDERED_OWNER_HELPER_POOL_ROUTES[arch]
        assert pool[module] == POOLED[module]
        _registered(program, arch)
        for family, args, kwargs in _requests(arch, module):
            helper = build(family, *args, owner_helper=True, **kwargs)
            assert helper.program == program and helper.owner_helper is True
            pooled = build(
                family, *args, owner_helper=True, owner_helper_pool=True, **kwargs
            )
            assert pooled.program == POOLED[module]
            assert pooled.owner_helper_pool is True
