import importlib.util
import re
import sys
import types
from pathlib import Path


def _load_generate_kernels(monkeypatch, cuda_at_least_128=True):
    repo_root = Path(__file__).resolve().parents[1]
    cutlass_dir = repo_root / "flashinfer" / "jit" / "gemm" / "cutlass"

    for name in (
        "flashinfer",
        "flashinfer.jit",
        "flashinfer.jit.gemm",
        "flashinfer.jit.gemm.cutlass",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    cpp_ext = types.ModuleType("flashinfer.jit.cpp_ext")
    cpp_ext.is_cuda_version_at_least = lambda _: cuda_at_least_128
    monkeypatch.setitem(sys.modules, "flashinfer.jit.cpp_ext", cpp_ext)

    cutlass_library_spec = importlib.util.spec_from_file_location(
        "flashinfer.jit.gemm.cutlass.cutlass_library",
        cutlass_dir / "cutlass_library.py",
    )
    cutlass_library = importlib.util.module_from_spec(cutlass_library_spec)
    monkeypatch.setitem(sys.modules, cutlass_library_spec.name, cutlass_library)
    cutlass_library_spec.loader.exec_module(cutlass_library)

    generate_kernels_spec = importlib.util.spec_from_file_location(
        "flashinfer.jit.gemm.cutlass.generate_kernels",
        cutlass_dir / "generate_kernels.py",
    )
    generate_kernels = importlib.util.module_from_spec(generate_kernels_spec)
    monkeypatch.setitem(sys.modules, generate_kernels_spec.name, generate_kernels)
    generate_kernels_spec.loader.exec_module(generate_kernels)
    return generate_kernels


def _generate_sources(monkeypatch, tmp_path, architectures, cuda_at_least_128=True):
    generate_kernels = _load_generate_kernels(monkeypatch, cuda_at_least_128)
    generate_kernels.generate_gemm_operations(tmp_path, architectures)
    return "\n".join(path.read_text() for path in tmp_path.rglob("*.generated.cu"))


def test_sm90_mixed_moe_generator_includes_fp8_mxfp4_launcher(
    monkeypatch, tmp_path
):
    generated_sources = _generate_sources(monkeypatch, tmp_path, "90;90-real")

    assert re.search(
        r"sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, "
        r"__nv_fp4_e2m1, __nv_bfloat16,.*"
        r"cute::Shape<cute::Int<128>, cute::Int<64>, cute::Int<256>>, "
        r"cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, "
        r"cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        generated_sources,
        flags=re.DOTALL,
    )
    assert (
        'moe_gemm_tma_ws_mixed_input_launcher.inl"'
    ) in generated_sources
    assert (
        "INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_fp8_e4m3, "
        "__nv_fp4_e2m1"
    ) not in generated_sources

    assert re.search(
        r"sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, "
        r"__nv_fp4_e2m1, half,.*"
        r"cute::Shape<cute::Int<128>, cute::Int<64>, cute::Int<256>>, "
        r"cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, "
        r"cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        generated_sources,
        flags=re.DOTALL,
    )


def test_sm90_mixed_moe_generator_excludes_fp8_mxfp4_before_cuda_128(
    monkeypatch, tmp_path
):
    generated_sources = _generate_sources(
        monkeypatch, tmp_path, "90;90-real", cuda_at_least_128=False
    )

    assert "__nv_fp4_e2m1" not in generated_sources
    assert 'moe_gemm_tma_ws_mixed_input_launcher.inl"' in generated_sources


def test_sm100_and_sm120_grouped_moe_generators_do_not_use_sm90_mixed_input_launcher(
    monkeypatch, tmp_path
):
    generated_sources = _generate_sources(monkeypatch, tmp_path, "100;120")

    assert 'moe_gemm_tma_ws_launcher.inl"' in generated_sources
    assert 'moe_gemm_tma_ws_mixed_input_launcher.inl"' not in generated_sources
    assert "INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm100" in generated_sources
    assert "INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120" in generated_sources


def test_wfp4afp8_dispatch_keeps_blackwell_on_regular_tma_path():
    repo_root = Path(__file__).resolve().parents[1]
    dispatch_source = (
        repo_root
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels"
        / "moe_gemm"
        / "moe_gemm_template_dispatch.h"
    ).read_text()

    assert (
        "isValidTmaWarpSpecializedMOESpecialisation<T, WeightType, EpilogueTag>() &&\n"
        "                  !use_w4_groupwise"
    ) in dispatch_source
    assert (
        "if (inputs.gemm_config.sm_version >= 90 &&\n"
        "          !(use_wfp4afp8 && inputs.gemm_config.sm_version == 90))"
    ) in dispatch_source
    assert (
        'TLLM_CHECK_WITH_INFO(inputs.gemm_config.sm_version == 90,\n'
        '                           "wfp4afp8 mixed-input dispatch is only supported for SM90");'
    ) in dispatch_source

    regular_tma_return = dispatch_source.index(
        "selected_func(hopper_inputs, inputs.num_experts, inputs.gemm_config, "
        "multi_processor_count_,\n                      inputs.stream, inputs.occupancy, "
        "nullptr);\n        return;"
    )
    wfp4afp8_mixed_branch = dispatch_source.index("if constexpr (use_wfp4afp8)")
    assert regular_tma_return < wfp4afp8_mixed_branch


def test_mxfp8_mxfp4_profiler_uses_distinct_quant_params_layout():
    repo_root = Path(__file__).resolve().parents[1]
    binding_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "flashinfer_cutlass_fused_moe_binding.cu"
    ).read_text()
    kernels_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "cutlass_fused_moe_kernels.cuh"
    ).read_text()
    kernels_header = (
        repo_root
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels"
        / "include"
        / "moe_kernels.h"
    ).read_text()

    assert "bool mUseMxfp8ActScaling{};" in kernels_header
    assert "mProfiler->mUseMxfp8ActScaling = mUseMxfp8ActScaling;" in binding_source
    assert (
        "bool is_mxfp8_mxfp4_quant = is_fp8_act_quant && is_fp4_w_quant && "
        "mUseMxfp8ActScaling;"
    ) in kernels_source

    mxfp8_layout = re.search(
        r"if \(is_mxfp8_mxfp4_quant\) \{.*?"
        r"quant_1_size = getOffsetWeightSF\(num_experts_per_node, inter_size, "
        r"hidden_size, mScalingType\).*?"
        r"quant_2_size = num_experts_per_node \* sizeof\(float\);.*?"
        r"quant_3_size = getOffsetWeightSF\(num_experts_per_node, hidden_size, "
        r"inter_size, mScalingType\).*?"
        r"quant_4_size = num_experts_per_node \* sizeof\(float\);.*?"
        r"\} else if \(is_fp4_w_quant\)",
        kernels_source,
        flags=re.DOTALL,
    )
    assert mxfp8_layout

    mxfp8_branch = kernels_source.index("mQuantParams = QuantParams::MXFP8MXFP4(")
    fp8_branch = kernels_source.index("mQuantParams = QuantParams::FP8MXFP4(")
    assert mxfp8_branch < fp8_branch


def test_mxfp8_activation_uses_per_expert_global_scale():
    repo_root = Path(__file__).resolve().parents[1]
    kernels_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "cutlass_fused_moe_kernels.cuh"
    ).read_text()

    assert "size_t act_scale_idx = (use_per_expert_act_scale || IsMXFP8) ? expert : 0;" in (
        kernels_source
    )
    assert "#ifdef ENABLE_FP8\n  constexpr bool IsMXFP8" in kernels_source
    assert "#ifdef ENABLE_FP8\n    auto MXFPX" in kernels_source
    assert "#ifdef ENABLE_FP8\n    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)" in (
        kernels_source
    )
    assert "#if defined(ENABLE_FP4) && defined(ENABLE_FP8)\n  constexpr bool IsMXFP8" not in (
        kernels_source
    )
    assert (
        "fc2_act_global_scale = quant_params.mxfp8_mxfp4.fc2.global_scale;"
        in kernels_source
    )
    assert (
        "fc2_act_global_scale = quant_params.mxfp8_mxfp8.fc2.global_scale;"
        in kernels_source
    )
    assert "quant_params.fp4.fc2.act_global_scale, use_per_expert_act_scale" not in (
        kernels_source
    )


def test_wfp4afp8_binding_accepts_sm90_interleaved_scale_layout():
    repo_root = Path(__file__).resolve().parents[1]
    binding_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "flashinfer_cutlass_fused_moe_binding.cu"
    ).read_text()

    assert "bool const sm90_interleaved_layout =" in binding_source
    assert "mCurrentSM = getCurrentDeviceSM();" in binding_source
    assert "bool const require_sm90_interleaved_weight_scales = mCurrentSM == 90;" in (
        binding_source
    )
    assert (
        "weight_block.size(1) * FP8_PER_INT32 *\n"
        "                    TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize =="
    ) in binding_source
    assert "weight_block.size(2) == expected_n;" in binding_source
    assert (
        "sm90_interleaved_layout ||\n"
        "                (!require_sm90_interleaved_weight_scales && natural_layout)"
    ) in binding_source


def test_wfp4afp8_binding_ignores_unvalidated_autotune_profile_ids():
    repo_root = Path(__file__).resolve().parents[1]
    binding_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "flashinfer_cutlass_fused_moe_binding.cu"
    ).read_text()

    assert "profile successfully but produce invalid outputs" in binding_source
    assert "default-tactic-only" in binding_source
    assert "bool const default_only_sm90_wfp4afp8 =" in binding_source
    assert "isWMxfp4AFp8Quant() && mCurrentSM == 90;" in binding_source
    assert (
        "bool const use_profile_ids = profile_ids.has_value() && !default_only_sm90_wfp4afp8;"
        in binding_source
    )
    assert "if (use_profile_ids) {" in binding_source


def test_run_gemm_profile_fallback_uses_gemm2_default_profile():
    repo_root = Path(__file__).resolve().parents[1]
    binding_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "flashinfer_cutlass_fused_moe_binding.cu"
    ).read_text()

    assert "Fallback tactic (-1) uses each GEMM stage's default profile." in binding_source
    assert "if (profile_id != -1) {" in binding_source
    assert "if (gemm_idx == 2 && mGemm2TacticCount > 0 &&" in binding_source
    assert "return mAllProfiles.at(mGemm1TacticCount);" in binding_source


def test_wfp4afp8_binding_clears_reused_workspace():
    repo_root = Path(__file__).resolve().parents[1]
    binding_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "flashinfer_cutlass_fused_moe_binding.cu"
    ).read_text()

    assert "clearWorkspaceForWfp4AFp8(workspace_info, stream);" in binding_source
    assert "void clearWorkspaceForWfp4AFp8" in binding_source
    assert "if (!isWMxfp4AFp8Quant())" in binding_source
    assert "cudaMemsetAsync(workspace_info.workspace.data_ptr(), 0" in binding_source
    assert "Failed to clear WFP4AFP8 workspace" in binding_source
    assert "clearOutputForWfp4AFp8(output, stream);" in binding_source
    assert "void clearOutputForWfp4AFp8" in binding_source
    assert "output.numel() * get_element_size(output)" in binding_source
    assert "Failed to clear WFP4AFP8 output" in binding_source


def test_wfp4afp8_python_autotune_exposes_default_only_hopper_profiles():
    repo_root = Path(__file__).resolve().parents[1]
    core_source = (repo_root / "flashinfer" / "fused_moe" / "core.py").read_text()

    assert "def _uses_default_only_sm90_wfp4afp8_tactics" in core_source
    assert 'backend == "90"' in core_source
    assert "self.x_dtype == torch.float8_e4m3fn" in core_source
    assert "self.weight_dtype == torch.int64" in core_source
    assert "not self.use_mxfp8_act_scaling" in core_source
    assert "if self._uses_default_only_sm90_wfp4afp8_tactics():" in core_source
    assert "return [-1]" in core_source
    assert "skip_wfp4afp8_autotune" not in core_source


def test_wfp4afp8_public_wrapper_scopes_fresh_output_to_ep():
    repo_root = Path(__file__).resolve().parents[1]
    core_source = (repo_root / "flashinfer" / "fused_moe" / "core.py").read_text()

    assert "use_wfp4afp8_sm90 = (" in core_source
    assert "user_output = output" in core_source
    assert (
        "use_ep_kernel_output = use_wfp4afp8_sm90 and ep_size > 1 and user_output is not None"
        in core_source
    )
    assert "torch.cuda.is_current_stream_capturing()" in core_source
    assert "not CUDA graph capture safe" in core_source
    assert "kernel_output = (" in core_source
    assert (
        "torch.empty_like(output) if use_ep_kernel_output else output"
        in core_source
    )
    assert "output.copy_(kernel_output)" in core_source
    assert "result[0] = output" in core_source


def test_wfp4afp8_benchmark_allows_cuda_graph_after_capture_probe():
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_source = (
        repo_root / "benchmarks" / "routines" / "moe.py"
    ).read_text()

    assert 'choices=["base", "fp8", "nvfp4", "mxfp4_fp8"]' in benchmark_source
    assert (
        "FP8 x MXFP4 benchmark path has not been validated for capture"
        not in benchmark_source
    )
    assert 'if variant == "mxfp4_fp8" and ep_size > 1 and is_cuda_graph_compatible:' in (
        benchmark_source
    )
    assert "EP public wrapper path is not capture safe" in benchmark_source
    assert (
        'bandwidth_input_format = "fp8" if variant == "mxfp4_fp8" else variant'
        in benchmark_source
    )
    assert (
        'bandwidth_weight_format = "mxfp4" if variant == "mxfp4_fp8" else variant'
        in benchmark_source
    )
    assert "metric_active_token_expert_pairs = int(local_expert_mask.sum().item())" in (
        benchmark_source
    )
    assert "active_token_expert_pairs=metric_active_token_expert_pairs" in (
        benchmark_source
    )
    assert "routing_logits_dtype=None" in benchmark_source
    assert "def _quantize_mxfp4_no_global_batched" in benchmark_source
    assert "w31_mxfp4, w31_mxfp4_scale = _quantize_mxfp4_no_global_batched" in (
        benchmark_source
    )
    assert 'cur_res["weight_dtype"] = "mxfp4"' in benchmark_source
    assert 'cur_res["fp4_mode"] = "mxfp4_fp8" if variant == "mxfp4_fp8" else ""' in (
        benchmark_source
    )


def test_cutlass_moe_runner_cache_key_includes_parallelism_state():
    repo_root = Path(__file__).resolve().parents[1]
    core_source = (repo_root / "flashinfer" / "fused_moe" / "core.py").read_text()

    start = core_source.index("instance_key = (")
    end = core_source.index(")", start)
    key_source = core_source[start:end]
    for field in (
        "fc1_weight_shape",
        "fc2_weight_shape",
        "top_k",
        "tp_size",
        "tp_rank",
        "ep_size",
        "ep_rank",
        "cluster_size",
        "cluster_rank",
        "enable_alltoall",
        "min_latency_mode",
        "enable_pdl",
        "activation_type",
    ):
        assert field in key_source


def test_sm90_mixed_weight_interleave_zero_initializes_output_padding():
    repo_root = Path(__file__).resolve().parents[1]
    core_source = (repo_root / "flashinfer" / "fused_moe" / "core.py").read_text()

    assert "out = torch.zeros_like(weight)" in core_source
    assert "module.interleave_moe_weights_for_sm90_mixed_gemm(" in core_source


def test_wfp4afp8_tma_path_initializes_alpha_pointer_arrays():
    repo_root = Path(__file__).resolve().parents[1]
    kernels_source = (
        repo_root
        / "csrc"
        / "fused_moe"
        / "cutlass_backend"
        / "cutlass_fused_moe_kernels.cuh"
    ).read_text()

    assert "if constexpr (use_wfp4afp8) {" in kernels_source
    assert "quant_params.fp8_mxfp4.fc1.global_scale" in kernels_source
    assert "quant_params.mxfp8_mxfp4.fc1.global_scale" in kernels_source
    assert "quant_params.fp8_mxfp4.fc2.global_scale" in kernels_source
    assert "quant_params.mxfp8_mxfp4.fc2.global_scale" in kernels_source
    assert "int const effective_group_size =" in kernels_source


def test_sm90_fp8fp4_mixed_autotune_exposes_only_single_cta_cluster():
    repo_root = Path(__file__).resolve().parents[1]
    heuristic_source = (
        repo_root
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels"
        / "cutlass_heuristic.cpp"
    ).read_text()

    assert "bool const has_fp8fp4_mixed = config & CutlassGemmConfig::FP8FP4_MIXED;" in (
        heuristic_source
    )
    assert "if (has_fp8fp4_mixed) continue;" in heuristic_source


def test_wfp4afp8_hopper_path_does_not_add_ampere_fallback_configs():
    repo_root = Path(__file__).resolve().parents[1]
    dispatch_source = (
        repo_root
        / "csrc"
        / "nv_internal"
        / "tensorrt_llm"
        / "kernels"
        / "cutlass_kernels"
        / "moe_gemm"
        / "moe_gemm_template_dispatch.h"
    ).read_text()

    assert "(use_w4afp8 && sm != 89) || use_wfp4a16 || use_wfp4afp8" in (
        dispatch_source
    )
