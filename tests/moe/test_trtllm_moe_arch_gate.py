# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device gate on the trtllm-gen MoE path.

``get_trtllm_moe_sm100_module`` is the one getter every ``trtllm_*_moe`` entry
point goes through, and it used to ask for a module on any device. The module's
JIT flags are built from ``supported_major_versions=[10, 12]``, so on Hopper
nothing survives the filter and the failure surfaces as
``No supported CUDA architectures found for major versions [10, 12]`` from
``CompilationContext`` -- naming neither this module nor trtllm-gen.
``is_trtllm_moe_supported`` has answered ``False`` for ``arch[0] < 10`` all
along; this is that answer acted on.

The builder itself is stubbed, so nothing is compiled and no cubin is fetched.
"""

import pytest
import torch

from flashinfer.fused_moe import core

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the getter reads the current CUDA device before it dispatches",
)


@pytest.mark.parametrize("capability", [(9, 0), (8, 9), (8, 6)])
def test_pre_blackwell_is_refused_before_any_build(monkeypatch, capability):
    monkeypatch.setattr(core, "get_compute_capability", lambda device: capability)

    def must_not_build(enable_rubin):
        raise AssertionError("the JIT module was requested on a pre-Blackwell device")

    monkeypatch.setattr(core, "_get_trtllm_moe_sm100_module_impl", must_not_build)
    with pytest.raises(
        ValueError, match=r"trtllm-gen MoE requires an SM100-family GPU"
    ):
        core.get_trtllm_moe_sm100_module()


@pytest.mark.parametrize(
    "capability,enable_rubin_expected",
    [((10, 0), False), ((10, 3), False), ((10, 7), True), ((12, 0), False)],
)
def test_datacenter_and_consumer_blackwell_still_reach_the_builder(
    monkeypatch, capability, enable_rubin_expected
):
    # SM120/SM121 are deliberately left alone here: they do compile, and the
    # reason they fail is the cubin manifest, which is a different gap.
    seen = {}

    def builder(enable_rubin):
        seen["enable_rubin"] = enable_rubin
        return "module"

    monkeypatch.setattr(core, "get_compute_capability", lambda device: capability)
    monkeypatch.setattr(core, "_get_trtllm_moe_sm100_module_impl", builder)
    assert core.get_trtllm_moe_sm100_module() == "module"
    assert seen["enable_rubin"] is enable_rubin_expected
