# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

from flashinfer.prims_ts import utils as prims_ts_utils
from flashinfer.prims_ts.moe import support as moe_support


@pytest.mark.parametrize("compute_capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("native_target_available", [False, True])
def test_prims_ts_arch_availability_uses_native_target_probe(
    monkeypatch, compute_capability, native_target_available
):
    calls = []

    def fake_arch_supported(major, minor, native_only=False):
        calls.append((major, minor, native_only))
        return native_target_available

    monkeypatch.setattr(
        "flashinfer.cute_dsl.utils.is_cute_dsl_arch_supported",
        fake_arch_supported,
    )

    assert (
        prims_ts_utils.is_prims_ts_arch_available(*compute_capability)
        is native_target_available
    )
    assert calls == [(*compute_capability, True)]


@pytest.mark.parametrize("compute_capability", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("native_target_available", [False, True])
def test_moe_device_support_requires_native_target(
    monkeypatch, compute_capability, native_target_available
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        moe_support, "get_compute_capability", lambda device: compute_capability
    )
    monkeypatch.setattr(
        moe_support,
        "is_prims_ts_arch_available",
        lambda major, minor: native_target_available,
    )

    assert (
        moe_support._device_supports_prims_ts(torch.device("cuda"))
        is native_target_available
    )


def test_prims_ts_available_rejects_missing_dsl(monkeypatch):
    monkeypatch.setattr(prims_ts_utils.importlib.util, "find_spec", lambda module: None)

    assert not prims_ts_utils.is_prims_ts_available()
