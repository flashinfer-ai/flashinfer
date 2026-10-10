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

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

# SM100-named modules that SM110 loads at runtime and whose generators build for
# 11.0a, so a jit-cache built only for 11.0a (the sm110a provider wheel) must
# ship them.
SM110_SHARED_MODULES = {
    "gen_fmha_cutlass_sm100a_module",
    "gen_mla_module",
    "gen_cutlass_fused_moe_sm100_module",
    "gen_gemm_sm100_module_cutlass_fp4",
    "gen_gemm_sm100_module_cutlass_fp8",
    "gen_gemm_sm100_module_cutlass_mxfp8",
    "gen_mxfp8_quantization_sm100_module",
    "gen_dcp_alltoall_module",
}

SM110_ONLY_MODULES = {"gen_fp4_quantization_sm110_module"}

# Gated on SM100/SM103 next to the modules above, but unusable on SM110: their
# generators exclude major 11, pin an sm_100a/sm_100f gencode, or load prebuilt
# SM10x cubins.
NOT_SM110_MODULES = {
    "gen_trtllm_gen_fmha_module",
    "gen_fp4_quantization_sm100_module",
    "gen_gemm_sm100_module",
    "gen_gemm_sm100_module_cutlass_nvfp4_svdquant",
    "gen_trtllm_gen_gemm_module",
    "gen_trtllm_low_latency_gemm_module",
    "gen_trtllm_gen_fused_moe_sm100_module",
    "gen_trtllm_gen_routing_module",
    "gen_tgv_gemm_sm10x_module",
    "gen_moe_utils_module",
    "gen_trtllm_mnnvl_comm_module",
}


def _registered_generators(monkeypatch, sm_capabilities, overrides=None):
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
    for attr, stub in (overrides or {}).items():
        monkeypatch.setattr(aot, attr, stub)
    monkeypatch.setattr(aot, "_gen_blackwell_bf16_bmm_aot_specs", lambda caps: [])
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version("13.0"))

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [False],  # XQA iterates over the sliding-window options.
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


SM100 = {"sm100": True, "sm100a_exact": True, "sm100f": True}
SM110 = {"sm110": True}


def test_sm110_only_build_registers_shared_sm100_modules(monkeypatch):
    registered = _registered_generators(monkeypatch, SM110)

    assert registered >= SM110_SHARED_MODULES | SM110_ONLY_MODULES
    assert not NOT_SM110_MODULES & registered


def test_sm100_only_build_is_unchanged(monkeypatch):
    registered = _registered_generators(monkeypatch, SM100)

    assert registered >= SM110_SHARED_MODULES | NOT_SM110_MODULES
    assert not SM110_ONLY_MODULES & registered


def test_combined_build_registers_both(monkeypatch):
    registered = _registered_generators(monkeypatch, {**SM100, **SM110})

    assert registered >= SM110_SHARED_MODULES | SM110_ONLY_MODULES | NOT_SM110_MODULES


def test_build_without_sm10x_or_sm110_registers_none(monkeypatch):
    registered = _registered_generators(monkeypatch, {"sm90": True})

    assert not SM110_SHARED_MODULES & registered


@pytest.mark.parametrize(
    "generator",
    ["gen_cutlass_fused_moe_sm100_module", "gen_gemm_sm100_module_cutlass_fp4"],
)
@pytest.mark.parametrize(
    "sm_capabilities", [SM110, {**SM100, **SM110}], ids=["sm110a", "sm100a+sm110a"]
)
def test_sm100_only_block_modules_are_generated_once(
    monkeypatch, generator, sm_capabilities
):
    # gen_all_modules deduplicates its returned specs by name, so also check the
    # generator is not invoked twice for a build that targets SM100 and SM110.
    calls = []

    def gen(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(name=generator)

    _registered_generators(monkeypatch, sm_capabilities, {generator: gen})

    assert len(calls) == 1


def _target(monkeypatch, arch_list):
    """Point every loaded flashinfer.jit module at a fresh context for arch_list,
    so the result does not depend on the GPU the test runs on."""
    from flashinfer.compilation_context import CompilationContext

    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", arch_list)
    ctx = CompilationContext()
    for name, module in list(sys.modules.items()):
        if name.startswith("flashinfer") and hasattr(
            module, "current_compilation_context"
        ):
            monkeypatch.setattr(module, "current_compilation_context", ctx)


SHARED_GENERATORS = [
    ("flashinfer.jit.attention.modules", "gen_fmha_cutlass_sm100a_module"),
    ("flashinfer.jit.mla", "gen_mla_module"),
    ("flashinfer.jit.fused_moe", "gen_cutlass_fused_moe_sm100_module"),
    ("flashinfer.jit.gemm.core", "gen_gemm_sm100_module_cutlass_fp4"),
    ("flashinfer.jit.gemm.core", "gen_gemm_sm100_module_cutlass_fp8"),
    ("flashinfer.jit.gemm.core", "gen_gemm_sm100_module_cutlass_mxfp8"),
    ("flashinfer.jit.fp8_quantization", "gen_mxfp8_quantization_sm100_module"),
    ("flashinfer.jit.comm", "gen_dcp_alltoall_module"),
]

FMHA_KWARGS = dict(
    dtype_q=torch.bfloat16,
    dtype_kv=torch.bfloat16,
    dtype_o=torch.bfloat16,
    dtype_idx=torch.int32,
    head_dim_qk=128,
    head_dim_vo=128,
    pos_encoding_mode=0,
    use_sliding_window=False,
    use_logits_soft_cap=False,
)


@pytest.mark.parametrize(
    ("module", "generator"),
    [pytest.param(m, g, id=g) for m, g in SHARED_GENERATORS],
)
def test_shared_generators_target_sm110(monkeypatch, module, generator):
    mod = importlib.import_module(module)
    _target(monkeypatch, "11.0a")
    kwargs = FMHA_KWARGS if generator == "gen_fmha_cutlass_sm100a_module" else {}

    flags = getattr(mod, generator)(**kwargs).extra_cuda_cflags

    assert "-gencode=arch=compute_110a,code=sm_110a" in flags


@pytest.mark.parametrize(
    ("module", "generator"),
    [
        ("flashinfer.jit.gemm.core", "gen_gemm_sm100_module"),
        ("flashinfer.jit.gemm.core", "gen_gemm_sm100_module_cutlass_nvfp4_svdquant"),
        ("flashinfer.jit.moe_utils", "gen_moe_utils_module"),
        ("flashinfer.jit.comm", "gen_trtllm_mnnvl_comm_module"),
    ],
)
def test_sm10x_only_generators_reject_sm110(monkeypatch, module, generator):
    # Keeps these out of the SM110 provider: an 11.0a-only build has no
    # architecture they accept.
    mod = importlib.import_module(module)
    _target(monkeypatch, "11.0a")

    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        getattr(mod, generator)()
