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


def _registered_generators(
    monkeypatch, sm_capabilities, overrides=None, add_moe=True, cuda="13.0"
):
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
    monkeypatch.setattr(aot, "tinygemm2_sm100_has_targets", lambda: True)
    for attr, stub in (overrides or {}).items():
        monkeypatch.setattr(aot, attr, stub)
    monkeypatch.setattr(aot, "_gen_blackwell_bf16_bmm_aot_specs", lambda caps: [])
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda))
    # The generated tinygemm_bf16 programs, one name per (operator, target).
    for operator in ("cake_tinygemm2_wgmma", "cake_tinygemm2_tcgen05"):
        monkeypatch.setattr(
            aot,
            operator,
            SimpleNamespace(
                aot_specs=lambda target, _op=operator: [
                    SimpleNamespace(name=f"{_op}:{target}")
                ]
            ),
            raising=False,
        )

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
        add_moe,
        False,
        True,
        True,
    )
    return {spec.name for spec in specs}


@pytest.mark.cpu_only
def test_sm103_only_build_registers_shared_sm100_modules(monkeypatch):
    registered = _registered_generators(
        monkeypatch, {"sm103": True, "sm103a_exact": True}
    )

    assert registered >= SM103_SHARED_MODULES | SM103_ONLY_MODULES
    assert not SM100_ONLY_MODULES & registered


@pytest.mark.cpu_only
def test_sm100_only_build_registers_shared_and_sm100_modules(monkeypatch):
    registered = _registered_generators(
        monkeypatch, {"sm100": True, "sm100a_exact": True, "sm100f": True}
    )

    assert registered >= SM103_SHARED_MODULES | SM100_ONLY_MODULES
    assert not SM103_ONLY_MODULES & registered


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("sm_capabilities", "expected_use_sm_100f"),
    [
        ({"sm100": True, "sm100a_exact": True, "sm100f": True}, False),
        ({"sm103": True, "sm103a_exact": True}, True),
        (
            {
                "sm100": True,
                "sm100a_exact": True,
                "sm100f": True,
                "sm103": True,
                "sm103a_exact": True,
            },
            True,
        ),
    ],
    ids=["sm100a", "sm103a", "sm100a+sm103a"],
)
def test_registered_tgv_gemm_runs_on_every_targeted_arch(
    monkeypatch, sm_capabilities, expected_use_sm_100f
):
    # Both TGV variants share a module name and the first one registered wins,
    # so it must be the SM100f build whenever SM103 is targeted.
    calls = []

    def gen_tgv(dtype, use_sm_100f=False):
        calls.append(use_sm_100f)
        return SimpleNamespace(name=f"tgv_gemm_{dtype}")

    _registered_generators(
        monkeypatch, sm_capabilities, {"gen_tgv_gemm_sm10x_module": gen_tgv}
    )

    assert calls[0] is expected_use_sm_100f


BF16_GEMM_MODULES = {
    "gen_mm_bf16_cublaslt_module",
    "gen_gemm_sm100_module_cutlass_bf16",
    "gen_tgv_gemm_sm10x_module",
    "gen_tinygemm2_module",
    "gen_tinygemm2_sm100_module",
}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm_capabilities",
    [
        {"sm100": True, "sm100a_exact": True, "sm100f": True},
        {"sm103": True, "sm103a_exact": True},
        {"sm107": True, "sm107a_exact": True},
    ],
    ids=["sm100a", "sm103a", "sm107a"],
)
def test_bf16_gemm_modules_are_prebuilt_for_every_blackwell_target(
    monkeypatch, sm_capabilities
):
    registered = _registered_generators(monkeypatch, sm_capabilities)

    assert registered >= BF16_GEMM_MODULES


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arch_list", "expected"),
    [
        ("10.0a", "-gencode=arch=compute_100a,code=sm_100a"),
        ("10.3a", "-gencode=arch=compute_103a,code=sm_103a"),
    ],
)
@pytest.mark.parametrize(
    "generator", ["gen_trtllm_gen_gemm_module", "gen_trtllm_low_latency_gemm_module"]
)
def test_trtllm_gen_gemm_runners_target_the_built_sm10x_arch(
    monkeypatch, generator, arch_list, expected
):
    # Provider validation rejects an sm_100a image in the sm103a wheel.
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit.gemm import core as gemm_core

    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", arch_list)
    monkeypatch.setattr(gemm_core, "current_compilation_context", CompilationContext())
    monkeypatch.setattr(gemm_core, "get_artifact", lambda *args: "artifact")
    monkeypatch.setattr(gemm_core, "get_meta_hash", lambda *args: "hash")
    monkeypatch.setattr(gemm_core, "ensure_symlink", lambda *args: None)
    monkeypatch.setattr(gemm_core, "verify_symlinked_headers", lambda *args: None)

    flags = getattr(gemm_core, generator)().extra_cuda_cflags

    assert [flag for flag in flags if flag.startswith("-gencode=")] == [expected]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch", ["sm90", "sm100", "sm103", "sm107", "sm110", "sm120", "sm121"]
)
def test_tinygemm2_is_prebuilt_for_every_arch_that_loads_it(monkeypatch, arch):
    # mm_bf16(backend="tinygemm") loads tinygemm2 on SM90+; when the jit-cache
    # lacks it, the first call runs nvcc.
    registered = _registered_generators(monkeypatch, {arch: True})

    assert "gen_tinygemm2_module" in registered


@pytest.mark.cpu_only
def test_tinygemm2_is_prebuilt_without_moe(monkeypatch):
    registered = _registered_generators(monkeypatch, {"sm100": True}, add_moe=False)

    assert {"gen_tinygemm2_module", "gen_tinygemm2_sm100_module"} <= registered


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm100", "sm103", "sm107"])
def test_bf16_cutlass_and_cublaslt_are_prebuilt_without_moe(monkeypatch, arch):
    registered = _registered_generators(monkeypatch, {arch: True}, add_moe=False)

    assert {
        "gen_mm_bf16_cublaslt_module",
        "gen_gemm_sm100_module_cutlass_bf16",
    } <= registered


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("sm90", False),
        ("sm100", True),
        ("sm103", True),
        ("sm107", True),
        ("sm110", False),
        ("sm120", False),
        ("sm121", False),
    ],
)
def test_tinygemm2_sm100_variant_is_prebuilt_only_for_sm10x(
    monkeypatch, arch, expected
):
    registered = _registered_generators(monkeypatch, {arch: True})

    assert ("gen_tinygemm2_sm100_module" in registered) == expected


@pytest.mark.cpu_only
def test_tinygemm2_sm100_is_skipped_when_no_target_survives_the_cuda_gate(
    monkeypatch,
):
    from flashinfer.jit import tinygemm2

    monkeypatch.setattr(
        tinygemm2, "is_cuda_version_at_least", lambda v: Version("13.0") >= Version(v)
    )
    monkeypatch.setattr(
        tinygemm2.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, "7a")}
    )

    registered = _registered_generators(
        monkeypatch,
        {"sm107": True},
        overrides={
            "tinygemm2_sm100_has_targets": tinygemm2.tinygemm2_sm100_has_targets
        },
    )

    assert "gen_tinygemm2_sm100_module" not in registered
    assert {
        "gen_mm_bf16_cublaslt_module",
        "gen_gemm_sm100_module_cutlass_bf16",
    } <= registered


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("cuda", "targets", "expected"),
    [
        ("13.4", {(10, "0a")}, ["compute_100a"]),
        ("13.4", {(10, "3a")}, ["compute_103a"]),
        ("13.4", {(10, "7a")}, ["compute_107a"]),
        (
            "13.4",
            {(10, "0a"), (10, "3a"), (10, "7a")},
            ["compute_100a", "compute_103a", "compute_107a"],
        ),
        (
            "13.0",
            {(10, "0a"), (10, "3a"), (10, "7a")},
            ["compute_100a", "compute_103a"],
        ),
        ("12.8", {(10, "0a"), (10, "3a"), (10, "7a")}, ["compute_100a"]),
        ("13.4", {(9, "0a"), (10, "0a")}, ["compute_100a"]),
    ],
)
def test_tinygemm2_sm100_targets_follow_the_build_targets(
    monkeypatch, cuda, targets, expected
):
    from flashinfer.jit import tinygemm2

    monkeypatch.setattr(
        tinygemm2, "is_cuda_version_at_least", lambda v: Version(cuda) >= Version(v)
    )
    monkeypatch.setattr(
        tinygemm2.current_compilation_context, "TARGET_CUDA_ARCHS", targets
    )
    flags = tinygemm2.gen_tinygemm2_sm100_module().extra_cuda_cflags

    assert [
        flag.split("arch=")[1].split(",")[0]
        for flag in flags
        if flag.startswith("-gencode=")
    ] == expected


def _cake_tinygemm2(registered):
    return {name for name in registered if name.startswith("cake_tinygemm2_")}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("sm_capabilities", "expected"),
    [
        ({"sm90": True, "sm90a_exact": True}, {"cake_tinygemm2_wgmma:sm_90a"}),
        # compute_90 without an exact sm_90a target: the wgmma programs need it.
        ({"sm90": True}, set()),
        (
            {"sm100": True, "sm100a_exact": True, "sm100f": True},
            {"cake_tinygemm2_tcgen05:sm_100a"},
        ),
        ({"sm103": True, "sm103a_exact": True}, {"cake_tinygemm2_tcgen05:sm_103a"}),
        (
            {
                "sm100": True,
                "sm100a_exact": True,
                "sm103": True,
                "sm103a_exact": True,
                "sm107": True,
                "sm107a_exact": True,
            },
            {
                "cake_tinygemm2_tcgen05:sm_100a",
                "cake_tinygemm2_tcgen05:sm_103a",
                "cake_tinygemm2_tcgen05:sm_107a",
            },
        ),
        ({"sm110": True}, set()),
        ({"sm120": True, "sm121": True}, set()),
    ],
    ids=["sm90a", "sm90-not-exact", "sm100a", "sm103a", "sm10x", "sm110", "sm12x"],
)
def test_cake_tinygemm2_programs_are_prebuilt_per_exact_target(
    monkeypatch, sm_capabilities, expected
):
    # tinygemm_bf16 tries these before the tinygemm2 kernels; they are
    # registered for the exact build targets, also without MoE.
    for add_moe in (True, False):
        registered = _registered_generators(
            monkeypatch, sm_capabilities, add_moe=add_moe, cuda="13.4"
        )
        assert _cake_tinygemm2(registered) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(("cuda", "registered"), [("13.0", False), ("13.4", True)])
def test_cake_tinygemm2_tcgen05_sm107a_needs_cuda_13_4(monkeypatch, cuda, registered):
    names = _registered_generators(
        monkeypatch, {"sm107": True, "sm107a_exact": True}, cuda=cuda
    )

    assert ("cake_tinygemm2_tcgen05:sm_107a" in names) == registered


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("module_name", "target", "count"),
    [
        ("cake_tinygemm2_wgmma", "sm_90a", 84),
        ("cake_tinygemm2_tcgen05", "sm_100a", 36),
        ("cake_tinygemm2_tcgen05", "sm_103a", 36),
        ("cake_tinygemm2_tcgen05", "sm_107a", 36),
    ],
)
def test_cake_tinygemm2_aot_specs_are_the_runtime_specs(
    monkeypatch, module_name, target, count
):
    # The jit-cache must ship the module names and flags the runtime loads.
    import importlib

    from flashinfer.jit import core

    # gen_jit_spec checks the build targets; without a GPU and without
    # FLASHINFER_CUDA_ARCH_LIST there are none, so build for this one.
    arch = {
        "sm_90a": (9, "0a"),
        "sm_100a": (10, "0a"),
        "sm_103a": (10, "3a"),
        "sm_107a": (10, "7a"),
    }[target]
    monkeypatch.setattr(core.current_compilation_context, "TARGET_CUDA_ARCHS", {arch})
    programs = importlib.import_module(f"flashinfer.jit.{module_name}")
    specs = programs.aot_specs(target)

    assert len(specs) == count
    assert [spec.name for spec in specs] == [
        program.module_name(target) for program in programs.PROGRAMS
    ]
    assert all(
        spec is program.spec(target)
        for spec, program in zip(specs, programs.PROGRAMS, strict=True)
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch", ["sm80", "sm89", "sm90", "sm100", "sm103", "sm107", "sm110", "sm120"]
)
def test_bf16_cublaslt_is_prebuilt_for_every_arch_mm_bf16_supports(monkeypatch, arch):
    # mm_bf16(backend="cublaslt") supports SM80+; CUTLASS BF16 only SM10x.
    registered = _registered_generators(monkeypatch, {arch: True}, add_moe=False)

    assert "gen_mm_bf16_cublaslt_module" in registered
    assert ("gen_gemm_sm100_module_cutlass_bf16" in registered) == (
        arch in ("sm100", "sm103", "sm107")
    )


def _sm103_with_cuda_12_8(monkeypatch, tmp_path):
    """An SM103 GPU and a local CUDA 12.8 toolkit, which cannot build sm_103a."""
    from flashinfer.jit import core, tinygemm2

    monkeypatch.setattr(
        tinygemm2, "is_cuda_version_at_least", lambda v: Version("12.8") >= Version(v)
    )
    monkeypatch.setattr(
        tinygemm2.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, "3a")}
    )
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    def no_compilation(*args, **kwargs):
        raise AssertionError("tinygemm2_sm100 must not be compiled")

    monkeypatch.setattr(core.JitSpecNvcc, "write_ninja", no_compilation)
    monkeypatch.setattr(core, "run_ninja", no_compilation)
    return core, tinygemm2


@pytest.mark.cpu_only
def test_tinygemm2_sm100_loads_a_prebuilt_sm103_module_with_cuda_12_8(
    monkeypatch, tmp_path
):
    core, tinygemm2 = _sm103_with_cuda_12_8(monkeypatch, tmp_path)
    artifact = core.jit_env.AOTArtifact(
        "flashinfer-jit-cache", tmp_path / "tinygemm2_sm100.so", None
    )
    monkeypatch.setattr(core.jit_env, "get_aot_artifacts", lambda name: (artifact,))
    loaded = object()
    monkeypatch.setattr(core.JitSpecNvcc, "load", lambda self, so_path=None: loaded)

    spec = tinygemm2.gen_tinygemm2_sm100_module()  # must not raise

    assert spec.name == "tinygemm2_sm100"
    assert spec.build_and_load() is loaded


@pytest.mark.cpu_only
def test_tinygemm2_sm100_aot_miss_with_cuda_12_8_does_not_compile(
    monkeypatch, tmp_path
):
    core, tinygemm2 = _sm103_with_cuda_12_8(monkeypatch, tmp_path)
    monkeypatch.setattr(core.jit_env, "get_aot_artifacts", lambda name: ())

    spec = tinygemm2.gen_tinygemm2_sm100_module()

    # No local target: a build would fall back to the global -gencode flags
    # (sm_103a here), which this toolkit cannot compile, so it is refused.
    assert not any(flag.startswith("-gencode=") for flag in spec.extra_cuda_cflags)
    with pytest.raises(RuntimeError, match="cannot build it"):
        spec.build_and_load()
