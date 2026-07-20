# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Serving freeze guards (ported semantics from b12x runtime_control)."""

from __future__ import annotations

import importlib

import pytest

rc = importlib.import_module("flashinfer.experimental.sm12x._lib.runtime_control")


def test_freeze_blocks_kernel_resolution_with_context():
    assert not rc.kernel_resolution_frozen()
    rc.freeze_kernel_resolution("unit-test")
    try:
        assert rc.kernel_resolution_frozen()
        with pytest.raises(rc.KernelResolutionFrozenError) as excinfo:
            rc.raise_if_kernel_resolution_frozen(
                "compile",
                target=test_freeze_blocks_kernel_resolution_with_context,
                cache_key=("shape", 128),
            )
        message = str(excinfo.value)
        assert "unit-test" in message
        assert "compile" in message
        assert "freeze_kernel_resolution" in message
    finally:
        rc.unfreeze_kernel_resolution()
    assert not rc.kernel_resolution_frozen()
    rc.raise_if_kernel_resolution_frozen("compile")  # no-op when unfrozen


def test_compilation_aliases_are_the_same_functions():
    assert rc.freeze_compilation is rc.freeze_kernel_resolution
    assert rc.unfreeze_compilation is rc.unfreeze_kernel_resolution
    assert rc.compilation_frozen is rc.kernel_resolution_frozen


def test_namespace_root_reexports():
    sm12x = importlib.import_module("flashinfer.experimental.sm12x")
    assert sm12x.freeze_kernel_resolution is rc.freeze_kernel_resolution
    assert sm12x.KernelResolutionFrozenError is rc.KernelResolutionFrozenError
