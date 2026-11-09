# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture routing of ``grouped_gemm_nt_masked``.

The wrapper used to dispatch on ``major >= 10``, which is wider than the kernel
it hands the tensors to: the Blackwell kernel builds ``sm_version`` as
``f"sm_{major}{minor}"`` and asserts it is one of sm_100/sm_103/sm_107, so
SM120/SM121 got past the wrapper and tripped that assert from inside the kernel
class constructor. These tests pin the wrapper to the same triple the suite
already skips on, and pin that SM100/SM103 still reach the kernel.

Both device reads are stubbed, so this needs no GPU.
"""

import sys
import types

import pytest
import torch

from flashinfer.gemm.kernels import grouped_gemm_masked_wrapper as wrapper

BLACKWELL_MODULE = "flashinfer.gemm.kernels.grouped_gemm_masked_blackwell"


def _call(monkeypatch, capability, sm100_kernel=None):
    """Run the dispatch with the device probes replaced.

    The tensors are only read by the kernel the branch selects, so CPU zeros are
    enough to reach -- or refuse -- it.
    """
    if sm100_kernel is not None:
        stub = types.ModuleType(BLACKWELL_MODULE)
        stub._grouped_gemm_nt_masked_sm100 = sm100_kernel
        monkeypatch.setitem(sys.modules, BLACKWELL_MODULE, stub)
    monkeypatch.setattr(wrapper, "get_compute_capability", lambda device: capability)
    monkeypatch.setattr(wrapper, "get_num_sm", lambda device: 1)

    t = torch.zeros(1)
    return wrapper.grouped_gemm_nt_masked(
        (t, t),
        (t, t),
        t,
        t,
        ab_dtype="mxfp8",
        sf_dtype="e8m0",
        c_dtype="bf16",
        sf_vec_size=32,
    )


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (11, 1), (9, 0)])
def test_unsupported_architectures_are_refused_at_the_entry(monkeypatch, capability):
    with pytest.raises(ValueError, match=r"only supported on SM100, SM103 and SM107"):
        _call(monkeypatch, capability)


def test_sm110_keeps_its_own_earlier_refusal(monkeypatch):
    # The message that predates this change must stay the one SM110 gets.
    with pytest.raises(ValueError, match="SM110 is not supported for cute-dsl"):
        _call(monkeypatch, (11, 0))


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_blackwell_still_reaches_the_kernel(monkeypatch, capability):
    seen = {}

    def sm100_kernel(**kwargs):
        seen.update(kwargs)
        return "reached"

    assert _call(monkeypatch, capability, sm100_kernel=sm100_kernel) == "reached"
    assert seen["sm_count"] == 1, "the branch must pass the resolved SM count on"


def test_rubin_still_reaches_the_rubin_kernel(monkeypatch):
    from flashinfer.gemm.kernels import grouped_gemm_masked_rubin as rubin

    calls = {}

    def refuse(**kwargs):
        calls.update(kwargs)
        raise RuntimeError("stub")

    monkeypatch.setattr(rubin, "_grouped_gemm_nt_masked_sm107", refuse)
    with pytest.raises(RuntimeError, match="stub"):
        _call(monkeypatch, (10, 7))
    assert calls, "SM107 must still be routed, not refused"
