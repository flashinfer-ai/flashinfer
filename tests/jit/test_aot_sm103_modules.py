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

# SM100-named modules that SM103 loads at runtime, so a jit-cache built only for
# 10.3a (the sm103a provider wheel) must ship them.
SM103_SHARED_MODULES = {
    "gen_fmha_cutlass_sm100a_module",
    "gen_trtllm_gen_fmha_module",
    "gen_mla_module",
    "gen_xqa_module",
    "gen_gemm_sm100_module",
    "gen_gemm_sm100_module_cutlass_nvfp4_svdquant",
    "gen_gemm_sm100_module_cutlass_fp8",
    "gen_gemm_sm100_module_cutlass_mxfp8",
    "gen_mxfp8_quantization_sm100_module",
    "gen_trtllm_gen_gemm_module",
    "gen_trtllm_low_latency_gemm_module",
    "gen_trtllm_gen_fused_moe_sm100_module",
    "gen_trtllm_gen_routing_module",
    "gen_tgv_gemm_sm10x_module",
    "gen_moe_utils_module",
    "gen_trtllm_mnnvl_comm_module",
    "gen_dcp_alltoall_module",
    "gen_rmsnorm_silu_module",
    "gen_trtllm_utils_module",
}

# SM103 loads its own variants of these instead.
SM100_ONLY_MODULES = {
    "gen_fp4_quantization_sm100_module",
    "gen_cutlass_fused_moe_sm100_module",
    "gen_gemm_sm100_module_cutlass_fp4",
}

SM103_ONLY_MODULES = {
    "gen_fp4_quantization_sm103_module",
    "gen_cutlass_fused_moe_sm103_module",
    "gen_gemm_sm103_module_cutlass_fp4",
}


def _registered_generators(monkeypatch, sm_capabilities):
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
                # Plural generators return a list of specs.
                result = [spec] if attr.endswith("_modules") else spec
                monkeypatch.setattr(
                    module, attr, lambda *args, _result=result, **kwargs: _result
                )
    monkeypatch.setattr(aot, "_gen_blackwell_bf16_bmm_aot_specs", lambda caps: [])
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))

    config = aot.get_default_config()
    specs = aot.gen_all_modules(
        config["f16_dtype"],
        config["f8_dtype"],
        config["fa2_head_dim"],
        config["fa3_head_dim"],
        config["use_sliding_window"],
        config["use_logits_soft_cap"],
        sm_capabilities,
        config["add_comm"],
        config["add_gemma"],
        config["add_oai_oss"],
        config["add_moe"],
        config["add_act"],
        config["add_misc"],
        config["add_xqa"],
    )
    return {spec.name for spec in specs}


def test_sm103_only_build_registers_shared_sm100_modules(monkeypatch):
    registered = _registered_generators(
        monkeypatch, {"sm103": True, "sm103a_exact": True}
    )

    assert registered >= SM103_SHARED_MODULES | SM103_ONLY_MODULES
    assert not SM100_ONLY_MODULES & registered


def test_sm100_only_build_registers_shared_and_sm100_modules(monkeypatch):
    registered = _registered_generators(
        monkeypatch, {"sm100": True, "sm100a_exact": True, "sm100f": True}
    )

    assert registered >= SM103_SHARED_MODULES | SM100_ONLY_MODULES
    assert not SM103_ONLY_MODULES & registered


def test_combined_build_registers_both_variants(monkeypatch):
    registered = _registered_generators(
        monkeypatch,
        {
            "sm100": True,
            "sm100a_exact": True,
            "sm100f": True,
            "sm103": True,
            "sm103a_exact": True,
        },
    )

    assert registered >= SM103_SHARED_MODULES | SM100_ONLY_MODULES | SM103_ONLY_MODULES


@pytest.mark.parametrize(
    ("arch_list", "expected", "unexpected"),
    [
        ("10.0a", "compute_100a", "compute_103a"),
        ("10.3a", "compute_103a", "compute_100a"),
    ],
)
def test_trtllm_gen_gemm_targets_the_built_sm10x_arch(
    monkeypatch, arch_list, expected, unexpected
):
    # Provider validation rejects an sm_100a image in the sm103a wheel.
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit.gemm import core as gemm_core

    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", arch_list)
    monkeypatch.setattr(gemm_core, "current_compilation_context", CompilationContext())

    flags = " ".join(gemm_core._trtllm_gen_gemm_nvcc_flags(enable_rubin=False))

    assert expected in flags
    assert unexpected not in flags
