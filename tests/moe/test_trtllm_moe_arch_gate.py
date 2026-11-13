# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device gate on the trtllm-gen MoE path.

``get_trtllm_moe_sm100_module`` is the one getter every ``trtllm_*_moe`` entry point
goes through, and on main it asked for the module whatever the process could build.
Whether a module can be built is decided by ``CompilationContext.TARGET_CUDA_ARCHS``
-- one entry per visible device, or the contents of ``FLASHINFER_CUDA_ARCH_LIST`` --
filtered by ``supported_major_versions=[10, 12]`` in ``jit/fused_moe.py``, so the gate
reads that same set rather than the current device. Without an SM100-family target the
failure was terminal either way: an unreachable manifest died in ``cubin_loader``, a
reachable one died in ``CompilationContext`` with ``No supported CUDA architectures
found for major versions [10, 12]``, and ``is_trtllm_moe_supported`` had already
answered ``False`` for ``arch[0] < 10`` with nothing in the tree acting on it.

The builder is stubbed and the architecture set is injected, so nothing is compiled, no
cubin is fetched, and no GPU is needed.
"""

import pytest
import torch

from flashinfer.fused_moe import core
from flashinfer.jit.core import current_compilation_context


def _give_process(monkeypatch, archs, capability=(9, 0)):
    """Make exactly ``archs`` buildable and pin which device is current."""
    monkeypatch.setattr(current_compilation_context, "TARGET_CUDA_ARCHS", set(archs))
    monkeypatch.setattr(core, "get_compute_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_the_gate_reads_the_context_the_build_filters():
    # A second CompilationContext, or a copy, would let the gate and the builder
    # disagree -- which is the only thing this predicate is for.
    assert core.current_compilation_context is current_compilation_context


@pytest.mark.parametrize(
    "archs",
    [
        {(9, "0")},  # Hopper only
        {(8, "0")},  # Ampere only
        {(8, "9")},  # Ada only
        set(),  # no CUDA device visible
    ],
)
def test_a_process_without_an_sm100_family_target_is_refused_before_any_build(
    monkeypatch, archs
):
    _give_process(monkeypatch, archs)

    def must_not_build(enable_rubin):
        raise AssertionError(
            "the JIT module was requested with no SM100-family target to build it for"
        )

    monkeypatch.setattr(core, "_get_trtllm_moe_sm100_module_impl", must_not_build)
    with pytest.raises(
        ValueError, match=r"trtllm-gen MoE needs an SM100-family CUDA target"
    ) as excinfo:
        core.get_trtllm_moe_sm100_module()
    # What the process could build is the diagnosis; keep it in the message.
    assert repr(sorted(archs)) in str(excinfo.value)


@pytest.mark.parametrize(
    "archs,capability,enable_rubin_expected",
    [
        ({(10, "0a")}, (10, 0), False),
        ({(10, "3a")}, (10, 3), False),
        ({(10, "7a")}, (10, 7), True),
        # SM120/SM121 are deliberately left alone: they do build here, and the
        # reason they fail is the cubin manifest, which is a different gap.
        ({(12, "0f")}, (12, 0), False),
        # Mixed pool -- the current device is SM90 but an SM100 target is
        # visible, so main builds the module and the call works when the tensors
        # sit on the SM100. The gate must not refuse it: the tensors decide
        # where the kernel runs, not the current device.
        ({(9, "0"), (10, "0a")}, (9, 0), False),
    ],
)
def test_targets_the_build_can_satisfy_still_reach_the_builder(
    monkeypatch, archs, capability, enable_rubin_expected
):
    _give_process(monkeypatch, archs, capability)
    seen = {}

    def builder(enable_rubin):
        seen["enable_rubin"] = enable_rubin
        return "module"

    monkeypatch.setattr(core, "_get_trtllm_moe_sm100_module_impl", builder)
    assert core.get_trtllm_moe_sm100_module() == "module"
    assert seen["enable_rubin"] is enable_rubin_expected
