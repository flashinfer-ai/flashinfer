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

import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe.nvfp4_checkpoint import NVFP4Checkpoint
from flashinfer.fused_moe.sm90_nvfp4_repack import (
    build_w4a8_v4_views,
    repack_nvfp4_sm90_v3,
)
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_w4a8_gemm import (
    STATIC_VARIANT_COUNT,
    SUPPORTED_BLOCK_M,
    SUPPORTED_BLOCK_N,
    SUPPORTED_GROUP_SIZES,
    SUPPORTED_RESIDUAL_SCHEMES,
    create_sm90_push_nvfp4_w4a8_gemm,
    get_sm90_push_nvfp4_w4a8_gemm_uri,
    load_sm90_push_nvfp4_w4a8_gemm_module,
)
from flashinfer.utils import is_sm90a_supported
from tests.moe._nvfp4_w4a8_oracle import simulate_w4a8_operand_bytes


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cuda_version_at_least("12.0")
    or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90 and CUDA Toolkit 12.0+",
)


def _checkpoint(
    device: torch.device,
    *,
    experts: int = 1,
    rows: int = 128,
    columns: int = 128,
) -> NVFP4Checkpoint:
    payload = torch.randint(
        0,
        256,
        (experts, rows, columns // 2),
        dtype=torch.uint8,
        device=device,
    )
    scales = (
        torch.rand(
            experts,
            rows,
            columns // 16,
            dtype=torch.float32,
            device=device,
        )
        * 1.5
        + 0.25
    ).to(torch.float8_e4m3fn)
    alpha = torch.rand(experts, dtype=torch.float32, device=device) + 0.5
    return NVFP4Checkpoint(
        payload,
        scales,
        alpha,
        (experts, rows, columns),
        tuple(range(experts)),
        "flashinfer.sm90_push.nvfp4.w4a8.test",
    )


def _linear_operand_bytes(view) -> torch.Tensor:
    tiled = simulate_w4a8_operand_bytes(
        view.packed_e2m1,
        view.promotion_residual,
        residual_scheme=view.manifest.residual_scheme,
    )
    experts, k_tiles, n_tiles, tile_n, tile_k = tiled.shape
    return (
        tiled.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, n_tiles * tile_n, k_tiles * tile_k)
    )


def _grouped_reference(
    activation: torch.Tensor,
    activation_scales: torch.Tensor,
    view,
    offsets: torch.Tensor,
) -> torch.Tensor:
    operand = _linear_operand_bytes(view).view(torch.float8_e4m3fn).float()
    experts, logical_n, _ = view.manifest.logical_shape
    _, padded_n, padded_k = view.manifest.padded_shape
    group_scales = (
        view.promotion_group_scale.permute(0, 2, 3, 1)
        .contiguous()
        .view(experts, padded_n, -1)
    )
    alpha = (
        view.global_alpha.expand(experts)
        if view.global_alpha.ndim == 0
        else view.global_alpha
    )
    group_size = view.manifest.group_size
    result = torch.zeros(
        activation.shape[0], logical_n, dtype=torch.float32, device=activation.device
    )
    for local_expert, source_expert in enumerate(view.manifest.expert_mapping):
        begin = int(offsets[source_expert].item())
        end = int(offsets[source_expert + 1].item())
        padded_begin = (begin + source_expert * 31) // 32 * 32
        for group in range(padded_k // group_size):
            k_begin = group * group_size
            k_end = k_begin + group_size
            activation_group = activation[begin:end, k_begin:k_end].float()
            weight_group = operand[local_expert, :logical_n, k_begin:k_end]
            partial = (activation_group[:, None, :] * weight_group[None, :, :]).sum(
                dim=-1
            )
            row_scale = activation_scales[
                k_begin // 128,
                padded_begin : padded_begin + end - begin,
                None,
            ]
            weight_scale = group_scales[local_expert, :logical_n, group][None, :]
            combined_scale = row_scale * weight_scale
            result[begin:end] += partial * combined_scale
        result[begin:end] *= alpha[local_expert]
    return result


def _normalized_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().square().sum().sqrt().clamp_min(1e-12)
    return float(
        (actual.float() - expected.float()).square().sum().sqrt() / denominator
    )


def _nonuniform_activation_scales(
    k_stages: int, padded_stride: int, device: torch.device
) -> torch.Tensor:
    stage_scale = torch.linspace(0.5, 1.5, k_stages, device=device)
    row_scale = torch.linspace(0.75, 1.25, padded_stride, device=device)
    return stage_scale[:, None] * row_scale[None, :]


def test_sm90_w4a8_static_variant_matrix_and_uri():
    assert STATIC_VARIANT_COUNT == 48
    assert SUPPORTED_BLOCK_M == (64, 128)
    assert SUPPORTED_BLOCK_N == (64, 128)
    assert SUPPORTED_GROUP_SIZES == (32, 64, 128)
    assert SUPPORTED_RESIDUAL_SCHEMES == ("generic", "pow2")
    uri = get_sm90_push_nvfp4_w4a8_gemm_uri(
        decode_vector=True,
        generic_decode_lut=True,
        overlap=False,
    )
    assert uri.startswith("sm90_push_nvfp4_w4a8_gemm_v3_dv1_dl1_ov0_pv4_")


def test_sm90_w4a8_optimization_flags_are_content_addressed():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    axis_contract = (
        ("decode_vector", "W4A8_DECODE_VECTOR", 1),
        ("generic_decode_lut", "W4A8_GENERIC_DECODE_LUT", 1),
        ("overlap", "W4A8_OVERLAP", 1),
    )
    runtime_fields = (
        "payload_layout",
        "prefer_n64_main",
        "m64_stages",
        "m128_stages",
        "tma_cache_capacity",
    )
    assert (
        tuple(field.name for field in fields(w4a8_gemm._OptimizationKnobs))
        == tuple(name for name, _macro, _default in axis_contract) + runtime_fields
    )
    assert (
        tuple(
            (name, tag, macro, default)
            for (name, macro, default), tag in zip(
                axis_contract,
                ("dv", "dl", "ov"),
                strict=True,
            )
        )
        == w4a8_gemm._KNOB_SPECS
    )
    defaults = w4a8_gemm._optimization_knobs()
    assert defaults == w4a8_gemm._OptimizationKnobs(
        decode_vector=1,
        generic_decode_lut=1,
        overlap=1,
        payload_layout=4,
        prefer_n64_main=0,
        m64_stages=3,
        m128_stages=4,
        tma_cache_capacity=128,
    )
    flags = w4a8_gemm._cuda_flags(defaults)
    for name, macro, default in axis_contract:
        assert getattr(defaults, name) == default
        assert f"-D{macro}={default}" in flags

    baseline_digest = w4a8_gemm._source_digest(knobs=defaults)
    for name, macro, default in axis_contract:
        overrides = {name: 1 - default}
        if name == "decode_vector":
            overrides["generic_decode_lut"] = 0
        variant = replace(defaults, **overrides)
        variant_flags = w4a8_gemm._cuda_flags(variant)
        assert f"-D{macro}={1 - default}" in variant_flags
        assert w4a8_gemm._source_digest(knobs=variant) != baseline_digest
        assert w4a8_gemm._optimization_knobs(**overrides) == variant

    with pytest.raises(ValueError, match=r"generic_decode_lut.*decode_vector"):
        w4a8_gemm._optimization_knobs(
            decode_vector=False,
            generic_decode_lut=True,
        )


def test_sm90_w4a8_loaded_module_cache_tracks_optimization_knobs(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    built = []

    class _Spec:
        def __init__(self, knobs):
            self.knobs = knobs

        def build_and_load(self):
            module = object()
            built.append((self.knobs, module))
            return module

    monkeypatch.setattr(
        w4a8_gemm,
        "_make_jit_spec",
        lambda knobs, *_args: _Spec(knobs),
    )
    w4a8_gemm._load_sm90_push_nvfp4_w4a8_gemm_module_cached.cache_clear()
    default_kwargs = {
        "decode_vector": True,
        "generic_decode_lut": True,
        "overlap": True,
    }
    variants = [default_kwargs]
    for name in default_kwargs:
        variant = dict(default_kwargs)
        variant[name] = not variant[name]
        if name == "decode_vector":
            variant["generic_decode_lut"] = False
        variants.append(variant)

    modules = []
    for kwargs in variants:
        module = w4a8_gemm.load_sm90_push_nvfp4_w4a8_gemm_module(**kwargs)
        assert w4a8_gemm.load_sm90_push_nvfp4_w4a8_gemm_module(**kwargs) is module
        modules.append(module)

    assert len({id(module) for module in modules}) == len(modules)
    assert [knobs for knobs, _module in built] == [
        w4a8_gemm._optimization_knobs(**kwargs) for kwargs in variants
    ]
    w4a8_gemm._load_sm90_push_nvfp4_w4a8_gemm_module_cached.cache_clear()


def test_sm90_w4a8_untrusted_schedule_validation_stays_on_device():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    sources = dict(w4a8_gemm._capture_source_snapshot().sources)
    scheduler = sources["scheduler.cuh"].decode("utf-8")
    binding = sources["binding.cu"].decode("utf-8")
    assert "sm90_w4a8_gemm: invalid offsets or expert mapping" in scheduler
    assert 'asm volatile("trap;")' in scheduler
    assert "source_expert < 0 || source_expert >= total_experts" in scheduler
    assert "total_experts_" in binding
    assert "cudaMemcpyDeviceToHost" not in binding
    assert "cudaStreamSynchronize" not in binding


def test_sm90_w4a8_decoded_stage_publication_uses_writer_arrivals():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    kernel_source = dict(w4a8_gemm._capture_source_snapshot().sources)[
        "kernel.cuh"
    ].decode("utf-8")
    kernel = "".join(kernel_source.split())
    initialization = (
        "reinterpret_cast<Barrier*>(&storage.decoded_ready[stage])"
        "->init(decoded_writer_threads<BlockN>());"
    )
    assert initialization in kernel
    assert "template<intBlockN>__host__" in kernel
    publication = (
        "if(wrote){fence_decoded_writer();}"
        "cutlass::arch::NamedBarrier(kProducerThreads,kProducerNamedBarrier).sync();"
        "if(wrote){decoded_ready->arrive();}"
    )
    assert publication in kernel
    assert "W4A8_SINGLE_READY" not in kernel


def test_sm90_w4a8_generic_decode_lut_keeps_scalar_and_pow2_fallbacks():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    decode_source = dict(w4a8_gemm._capture_source_snapshot().sources)[
        "decode.cuh"
    ].decode("utf-8")
    decode = "".join(decode_source.split())
    assert "#defineW4A8_GENERIC_DECODE_LUT1" in decode
    assert "#ifW4A8_GENERIC_DECODE_LUT&&!W4A8_DECODE_VECTOR" in decode
    assert (
        "structGenericDecodeLut{uint32_tlow;uint32_thigh;uint32_tresidual_sign;}"
        in decode
    )
    assert 'asmvolatile("prmt.b32%0,%1,%2,%3;\\n"' in decode
    assert "decode_generic_lut_pair" in decode
    assert "ifconstexpr(Scheme==ResidualScheme::kGeneric)" in decode
    assert "decode_two_packed_bytes<ResidualScheme::kGeneric>" in decode
    assert "structResidualDecoder<ResidualScheme::kPow2>" in decode
    assert "run_scalar_task" in decode


def test_sm90_w4a8_residual_tma_and_global_scale_contract():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    sources = {
        name: content.decode("utf-8")
        for name, content in w4a8_gemm._capture_source_snapshot().sources
    }
    kernel = "".join(sources["kernel.cuh"].split())
    instantiation = "".join(sources["kernel_instantiation.cuh"].split())
    launchers = "".join(sources["kernel_launchers.cuh"].split())
    binding = "".join(sources["binding.cu"].split())

    assert (
        "#ifW4A8_PAYLOAD_V4"
        "alignas(1024)ResidualStorageresidual"
        "[kStages][BlockN][kBlockK/kV3ResidualBlockK];"
        "#elsealignas(1024)ResidualStorageresidual"
        "[kStages][kBlockK/kV3PayloadTileK][BlockN][kV3ResidualsPerPayloadTile];"
        "#endif" in kernel
    )
    assert "W4A8_GROUP_SCALE_TMA" not in kernel
    assert "W4A8_GROUP_SCALE_STAGING" not in kernel
    assert (
        "constexprintkResidualStageBytes="
        "BlockN*(kBlockK/kV3ResidualBlockK)*sizeof(typenameStorage::ResidualStorage);"
        in kernel
    )
    assert (
        "kExpectedBytes=kActivationStageBytes+kRawStageBytes+"
        "kResidualStageBytes" in kernel
    )
    assert "arrive_and_expect_tx(kExpectedBytes)" in kernel
    assert "producer_decode_global_stage" not in kernel
    assert "producer_decode_stage<BlockN,Scheme>" in kernel
    assert "weight_scale0=__ldg(group_scales+column0);" in kernel
    assert "weight_scale1=__ldg(group_scales+column1);" in kernel

    assert "params.residual_map" in instantiation
    assert "params.group_scales" in instantiation
    assert "constfloat*group_scales;" in launchers
    assert "CUtensorMapresidual_map;" in launchers
    assert "constvoid*residual" not in launchers
    assert (
        "resolved.group_scales=static_cast<constfloat*>(group_scales.data_ptr());"
        in binding
    )
    for source in (kernel, instantiation, launchers, binding):
        assert "W4A8_RESIDUAL_TMA" not in source


def test_sm90_w4a8_retirement_and_tail_policy_are_fixed():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    sources = {
        name: content.decode("utf-8")
        for name, content in w4a8_gemm._capture_source_snapshot().sources
    }
    kernel = "".join(sources["kernel.cuh"].split())
    binding = "".join(sources["binding.cu"].split())

    assert "floatpartial[WGMMA::kNumAccum];" in kernel
    assert "floatpartial[2][WGMMA::kNumAccum];" not in kernel
    assert "W4A8_CROSS_STAGE_RETIRE" not in kernel
    assert "W4A8_SINGLE_PARTIAL" not in kernel
    assert "W4A8_EMPTY_FAMILY_EARLY_EXIT" not in kernel
    assert "W4A8_EXPERT_N_MAJOR" not in sources["scheduler.cuh"]
    assert "storage.last_group" not in kernel
    assert "deep_gemm::warpgroup_wait<0>();" in kernel
    assert "deep_gemm::warpgroup_wait<1>();" not in kernel
    assert (
        "final_accum,partial,activation_scale0,activation_scale1,"
        "current_group_scales,global_group,(k_stage+1)*kGroupsPerStage-1,"
        "&storage.empty[stage],lane" in kernel
    )
    assert (
        "constint64_ttiles_m64=m64_tile_count(rows);"
        "constint64_ttiles_m128=m128_tile_count(rows);" in binding
    )
    assert binding.count("launch_tile_family<DebugFp32,64,") == 3
    assert "W4A8_SPLIT_M64_TAIL" not in binding


def test_sm90_w4a8_split_tus_flags_and_launch_bounds_are_source_visible():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    sources = {
        name: content.decode("utf-8")
        for name, content in w4a8_gemm._capture_source_snapshot().sources
    }
    expected_instantiations = {
        "kernel_inst_m64_n64.cu": (64, 64),
        "kernel_inst_m64_n128.cu": (64, 128),
        "kernel_inst_m128_n64.cu": (128, 64),
        "kernel_inst_m128_n128.cu": (128, 128),
    }
    assert (
        "binding.cu",
        *expected_instantiations,
    ) == w4a8_gemm._TRANSLATION_UNIT_NAMES
    assert [name for name in sources if name.endswith(".cu")] == [
        "binding.cu",
        *expected_instantiations,
    ]
    assert '#include "kernel_launchers.cuh"' in sources["binding.cu"]
    for name, (block_m, block_n) in expected_instantiations.items():
        source = sources[name]
        assert source.count('#include "kernel_instantiation.cuh"') == 1
        assert source.count("FLASHINFER_SM90_W4A8_DEFINE_MN_VARIANTS") == 1
        assert (
            f"FLASHINFER_SM90_W4A8_DEFINE_MN_VARIANTS({block_m}, {block_n})" in source
        )

    instantiation = sources["kernel_instantiation.cuh"].split(
        "#define FLASHINFER_SM90_W4A8_DEFINE_MN_VARIANTS", maxsplit=1
    )[1]
    assert instantiation.count("detail::make_w4a8_kernel_variant<") == 6

    scheduler = "".join(sources["scheduler.cuh"].split())
    assert "structW4A8LaunchTraits" in scheduler
    assert "kProducerThreads=128" in scheduler
    assert "kConsumerThreadsFor=(BlockM/64)*128" in scheduler
    assert "kThreads=kThreadsFor<BlockM>" in scheduler
    assert "BlockM==64&&BlockN==64?2:1" in scheduler
    assert "kPipelineStages=BlockM==64?3:4" in scheduler
    assert "kDebugMinBlocksPerSm=1" in scheduler
    assert "Traits::kPipelineStages" in sources["kernel_instantiation.cuh"]
    instantiation_source = "".join(sources["kernel_instantiation.cuh"].split())
    assert "cudaFuncGetAttributes(&attributes,kernel)" in instantiation_source
    assert "resources->num_regs=attributes.numRegs" in instantiation_source
    assert (
        "resources->local_memory_bytes=attributes.localSizeBytes"
        in instantiation_source
    )
    binding = "".join(sources["binding.cu"].split())
    assert 'if(name=="kernel_resource_usage")' in binding
    assert "resource.local_memory_bytes" in binding
    assert 'if(name=="kernel_resources")' in binding
    assert "variant.consumer_register_cap" in binding
    assert "fp32_resources->blocks_per_sm,1" in binding
    shim = "".join(
        (Path(w4a8_gemm.__file__).resolve()).read_text(encoding="utf-8").split()
    )
    assert "def_require_usable_resources(self)->None:" in shim
    assert "self._require_usable_resources()" in shim
    kernel_source = sources["kernel.cuh"]
    bf16_declaration = "".join(
        kernel_source.split("grouped_w4a8_bf16_kernel", 1)[0]
        .rsplit("template <int BlockM", 1)[-1]
        .split()
    )
    fp32_declaration = "".join(
        kernel_source.split("grouped_w4a8_fp32_debug_kernel", 1)[0]
        .rsplit("template <int BlockM", 1)[-1]
        .split()
    )
    kernel = "".join(kernel_source.split())
    compile_switches = "".join((sources["decode.cuh"] + sources["kernel.cuh"]).split())
    assert "#defineW4A8_DECODE_VECTOR1" in compile_switches
    assert "#defineW4A8_GENERIC_DECODE_LUT1" in compile_switches
    assert "#defineW4A8_OVERLAP0" in compile_switches
    assert "-DW4A8_OVERLAP=1" in w4a8_gemm._cuda_flags(w4a8_gemm._optimization_knobs())
    assert "W4A8LaunchTraits<BlockM,BlockN,PipelineStages>::kThreads" in kernel
    assert (
        "ifconstexpr(!std::is_same_v<Output,float>){"
        "if(is_consumer){cutlass::arch::warpgroup_reg_alloc<kConsumerRegisters>();}"
        "else{cutlass::arch::warpgroup_reg_dealloc<"
        "Traits::kProducerRegisters>();}}" in kernel
    )
    assert (
        "W4A8LaunchTraits<BlockM,BlockN,PipelineStages>::kMinBlocksPerSm"
        in bf16_declaration
    )
    assert "kDebugMinBlocksPerSm" not in bf16_declaration
    assert (
        "W4A8LaunchTraits<BlockM,BlockN,PipelineStages>::kDebugMinBlocksPerSm"
        in fp32_declaration
    )
    assert "kMinBlocksPerSm" not in fp32_declaration.replace("kDebugMinBlocksPerSm", "")


def test_sm90_w4a8_jit_spec_consumes_the_hashed_source_snapshot(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_w4a8_gemm as w4a8_gemm,
    )

    source_root = tmp_path / "live" / "csrc"
    source_directory = source_root / "fused_moe" / "sm90_w4a8_gemm"
    source_directory.mkdir(parents=True)
    binding = source_directory / "binding.cu"
    kernel = source_directory / "kernel.cuh"
    binding.write_bytes(b"binding-v1\r\n")
    kernel.write_bytes(b"kernel-v1")
    deep_gemm = source_root / "nv_internal" / "tensorrt_llm" / "deep_gemm"
    deep_gemm.mkdir(parents=True)
    (deep_gemm / "utils.cuh").write_bytes(b"utils-v1")
    (deep_gemm / "nvrtc_cutlass.cuh").write_bytes(b"cutlass-v1")
    (source_root / "tvm_ffi_utils.h").write_bytes(b"ffi-v1")
    layout = source_root.parent / "include" / "flashinfer" / "layout.cuh"
    layout.parent.mkdir(parents=True)
    layout.write_bytes(b"layout-v1")
    monkeypatch.setattr(w4a8_gemm, "_SOURCE_NAMES", ("kernel.cuh", "binding.cu"))
    monkeypatch.setattr(w4a8_gemm, "_TRANSLATION_UNIT_NAMES", ("binding.cu",))
    monkeypatch.setattr(
        w4a8_gemm,
        "_DEEP_GEMM_DEPENDENCIES",
        ("utils.cuh", "nvrtc_cutlass.cuh"),
    )
    monkeypatch.setattr(w4a8_gemm, "_source_directory", lambda: source_directory)
    monkeypatch.setattr(w4a8_gemm, "_csrc_directory", lambda: source_root)
    monkeypatch.setattr(
        w4a8_gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated"
    )
    monkeypatch.setattr(w4a8_gemm, "is_cuda_version_at_least", lambda _: True)

    class _Spec:
        def __init__(self, name, sources, options):
            self.name = name
            self.sources = sources
            self.options = options

    monkeypatch.setattr(
        w4a8_gemm,
        "gen_jit_spec",
        lambda name, sources, **options: _Spec(name, sources, options),
    )
    expected_snapshot = w4a8_gemm._capture_source_snapshot()
    expected_digest = w4a8_gemm._source_digest(expected_snapshot)
    binding.write_bytes(b"binding-v1\n")
    assert w4a8_gemm._source_digest() == expected_digest
    binding.write_bytes(b"binding-v1\r\n")
    materialize = w4a8_gemm._materialize_source_snapshot

    def mutate_then_materialize(uri, snapshot):
        kernel.write_bytes(b"kernel-v2")
        return materialize(uri, snapshot)

    monkeypatch.setattr(
        w4a8_gemm, "_materialize_source_snapshot", mutate_then_materialize
    )
    spec = w4a8_gemm.gen_sm90_push_nvfp4_w4a8_gemm_module()

    snapshotted_binding = spec.sources[0]
    snapshotted_root = snapshotted_binding.parent.parent
    assert spec.name.endswith(f"_{expected_digest}")
    assert snapshotted_binding != binding
    assert snapshotted_binding.read_bytes() == b"binding-v1\n"
    assert (snapshotted_binding.parent / "kernel.cuh").read_bytes() == b"kernel-v1"
    assert (
        snapshotted_root / "nv_internal/tensorrt_llm/deep_gemm/utils.cuh"
    ).read_bytes() == b"utils-v1"
    assert (
        snapshotted_root / "nv_internal/tensorrt_llm/deep_gemm/nvrtc_cutlass.cuh"
    ).read_bytes() == b"cutlass-v1"
    assert (snapshotted_root / "flashinfer/layout.cuh").read_bytes() == b"layout-v1"
    assert spec.options["extra_include_paths"] == [
        snapshotted_root,
        snapshotted_binding.parent,
    ]
    assert kernel.read_bytes() == b"kernel-v2"


@requires_sm90
@pytest.mark.parametrize("decode_vector", (False, True), ids=("scalar", "vector"))
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("residual_scheme", SUPPORTED_RESIDUAL_SCHEMES)
def test_sm90_w4a8_operand_byte_gate(decode_vector, group_size, residual_scheme):
    device = torch.device("cuda")
    checkpoint = _checkpoint(device)
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        1,
        view,
        decode_vector=decode_vector,
        generic_decode_lut=decode_vector,
        overlap=True,
        payload_layout=3,
        allow_legacy_layout=True,
    )
    actual = runner.debug_decode()
    expected = simulate_w4a8_operand_bytes(
        view.packed_e2m1,
        view.promotion_residual,
        residual_scheme=residual_scheme,
    )
    assert torch.equal(actual, expected)


@requires_sm90
@pytest.mark.parametrize("decode_vector", (False, True), ids=("scalar", "vector"))
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("residual_scheme", SUPPORTED_RESIDUAL_SCHEMES)
def test_sm90_w4a8_v4_operand_byte_gate(decode_vector, group_size, residual_scheme):
    device = torch.device("cuda")
    v3 = repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=192, columns=256),
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    v4 = build_w4a8_v4_views(v3)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        1,
        v4,
        decode_vector=decode_vector,
        generic_decode_lut=decode_vector,
        overlap=True,
        payload_layout=4,
    )

    actual = runner.debug_decode()
    expected = simulate_w4a8_operand_bytes(
        v3.packed_e2m1,
        v3.promotion_residual,
        residual_scheme=residual_scheme,
    )
    assert torch.equal(actual, expected)


@requires_sm90
@pytest.mark.parametrize("residual_scheme", SUPPORTED_RESIDUAL_SCHEMES)
def test_sm90_w4a8_v4_vector_residual_index_keeps_each_n64_block(
    residual_scheme,
):
    device = torch.device("cuda")
    v3 = repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=192, columns=256),
        group_size=128,
        residual_scheme=residual_scheme,
    )
    for n64 in range(v3.promotion_residual.shape[2]):
        if residual_scheme == "generic":
            v3.promotion_residual[:, :, n64].fill_(0.5 + n64)
        else:
            v3.promotion_residual[:, :, n64].fill_(n64 - 1)
    v4 = build_w4a8_v4_views(v3, verify_checksums=False)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        1,
        v4,
        decode_vector=True,
        generic_decode_lut=True,
        overlap=True,
        payload_layout=4,
    )

    actual = runner.debug_decode()
    expected = simulate_w4a8_operand_bytes(
        v3.packed_e2m1,
        v3.promotion_residual,
        residual_scheme=residual_scheme,
    )
    assert torch.equal(actual, expected)


@requires_sm90
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("residual_scheme", SUPPORTED_RESIDUAL_SCHEMES)
def test_sm90_w4a8_v4_matches_v3_grouped_gemm(group_size, residual_scheme):
    torch.manual_seed(17)
    device = torch.device("cuda")
    rows = 129
    columns = 256
    v3 = repack_nvfp4_sm90_v3(
        _checkpoint(device, experts=2, rows=192, columns=columns),
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    v4 = build_w4a8_v4_views(v3)
    activation = torch.randn(rows, columns, device=device).to(torch.float8_e4m3fn)
    activation_scales = _nonuniform_activation_scales(
        columns // 128,
        (rows + 2 * 31) // 32 * 32,
        device,
    )
    offsets = torch.tensor([0, 65, rows], dtype=torch.int64, device=device)
    v3_runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows, v3, payload_layout=3, allow_legacy_layout=True
    )
    v4_runner = create_sm90_push_nvfp4_w4a8_gemm(rows, v4, payload_layout=4)

    assert torch.equal(
        v4_runner.run_debug_fp32(activation, activation_scales, offsets),
        v3_runner.run_debug_fp32(activation, activation_scales, offsets),
    )
    assert torch.equal(
        v4_runner.run(activation, activation_scales, offsets),
        v3_runner.run(activation, activation_scales, offsets),
    )


@requires_sm90
def test_sm90_w4a8_payload_layout_mismatch_fails_before_launch():
    device = torch.device("cuda")
    v3 = repack_nvfp4_sm90_v3(
        _checkpoint(device), group_size=128, residual_scheme="generic"
    )
    v4 = build_w4a8_v4_views(v3)

    with pytest.raises(ValueError, match="allow_legacy_layout"):
        create_sm90_push_nvfp4_w4a8_gemm(1, v3, payload_layout=3)
    with pytest.raises(ValueError, match="layout"):
        create_sm90_push_nvfp4_w4a8_gemm(1, v3, payload_layout=4)
    with pytest.raises(ValueError, match="layout"):
        create_sm90_push_nvfp4_w4a8_gemm(
            1, v4, payload_layout=3, allow_legacy_layout=True
        )


@requires_sm90
def test_sm90_w4a8_generic_lut_matches_scalar_for_all_bf16_residual_bits():
    device = torch.device("cuda")
    bits = torch.arange(1 << 16, dtype=torch.int32, device=device)
    signed_bits = torch.where(bits < (1 << 15), bits, bits - (1 << 16)).to(torch.int16)
    residual_values = signed_bits.view(torch.bfloat16)
    residual = residual_values.view(128, 4, 1, 64, 2).contiguous()

    codes = torch.arange(16, dtype=torch.uint8, device=device)
    packed_codes = codes[0::2] | (codes[1::2] << 4)
    packed_row = torch.cat((packed_codes, packed_codes))
    payload = packed_row.view(1, 1, 1, 1, 16).expand(128, 4, 1, 64, 16).contiguous()
    output_shape = (128, 4, 1, 64, 32)

    def decode(*, decode_vector, generic_decode_lut):
        module = load_sm90_push_nvfp4_w4a8_gemm_module(
            decode_vector=decode_vector,
            generic_decode_lut=generic_decode_lut,
            payload_layout=3,
        )
        ffi_runner = module.init()
        workspace_size = int(
            ffi_runner.get_workspace_size(1, 64, 64, 128, 128, 128, 128, "generic")
        )
        workspace = torch.empty(
            (max(workspace_size, 1),), dtype=torch.uint8, device=device
        )
        ffi_runner.configure_workspace(workspace)
        output = torch.empty(output_shape, dtype=torch.uint8, device=device)
        ffi_runner.debug_decode(output, payload, residual)
        return output

    scalar = decode(decode_vector=False, generic_decode_lut=False)
    vector_reference = decode(decode_vector=True, generic_decode_lut=False)
    lut = decode(decode_vector=True, generic_decode_lut=True)
    assert torch.equal(vector_reference, scalar)
    assert torch.equal(lut, vector_reference)


@requires_sm90
@pytest.mark.parametrize("decode_vector", (False, True), ids=("scalar", "vector"))
def test_sm90_w4a8_pow2_operand_gate_exhausts_int8_exponents(decode_vector):
    device = torch.device("cuda")
    codes = torch.arange(16, dtype=torch.uint8, device=device).repeat(2)
    payload = codes.repeat(1, 4, 1, 64, 1)
    payload = (payload[..., 0::2] | (payload[..., 1::2] << 4)).contiguous()
    exponents = torch.arange(-128, 128, dtype=torch.int16, device=device).to(torch.int8)
    residual = exponents.repeat(2).reshape(1, 4, 1, 64, 2).contiguous()
    expected = simulate_w4a8_operand_bytes(
        payload,
        residual,
        residual_scheme="pow2",
    )

    module = load_sm90_push_nvfp4_w4a8_gemm_module(
        decode_vector=decode_vector,
        generic_decode_lut=decode_vector,
        overlap=True,
        payload_layout=3,
    )
    ffi_runner = module.init()
    workspace_size = int(
        ffi_runner.get_workspace_size(1, 64, 64, 128, 1, 1, 32, "pow2")
    )
    workspace = torch.empty((max(workspace_size, 1),), dtype=torch.uint8, device=device)
    ffi_runner.configure_workspace(workspace)
    actual = torch.empty_like(expected)
    ffi_runner.debug_decode(actual, payload, residual)

    assert torch.equal(actual, expected)


@requires_sm90
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("residual_scheme", SUPPORTED_RESIDUAL_SCHEMES)
@pytest.mark.parametrize("logical_n", (64, 128, 192))
def test_sm90_w4a8_output_gates(group_size, residual_scheme, logical_n):
    torch.manual_seed(7)
    device = torch.device("cuda")
    rows = 129
    columns = 256
    checkpoint = _checkpoint(device, rows=logical_n, columns=columns)
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    activation = torch.randn(rows, columns, device=device).to(torch.float8_e4m3fn)
    padded_stride = max((rows + 31) // 32 * 32, 1)
    activation_scales = _nonuniform_activation_scales(
        columns // 128, padded_stride, device
    )
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows, view, payload_layout=3, allow_legacy_layout=True
    )
    expected = _grouped_reference(activation, activation_scales, view, offsets)

    debug_output = runner.run_debug_fp32(activation, activation_scales, offsets)
    assert _normalized_l2(debug_output, expected) <= 1e-3

    output = runner.run(activation, activation_scales, offsets).float()
    assert _normalized_l2(output, expected) <= 5e-3
    cosine = float(F.cosine_similarity(output.flatten(), expected.flatten(), dim=0))
    assert cosine >= 0.999
    absolute_scale = max(float(expected.abs().amax()), 1.0)
    assert float((output - expected).abs().amax()) <= 2e-2 * absolute_scale
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@requires_sm90
@pytest.mark.parametrize("rows", (0, 1, 31, 32, 63, 64, 65, 127, 128, 129))
def test_sm90_w4a8_m_boundaries(rows):
    torch.manual_seed(7)
    device = torch.device("cuda")
    view = repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=64),
        group_size=128,
        residual_scheme="generic",
    )
    activation = torch.randn(rows, 128, device=device).to(torch.float8_e4m3fn)
    padded_stride = max((rows + 31) // 32 * 32, 1)
    activation_scales = _nonuniform_activation_scales(1, padded_stride, device)
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows,
        view,
        payload_layout=3,
        allow_legacy_layout=True,
    )

    debug_output = runner.run_debug_fp32(activation, activation_scales, offsets)
    output = runner.run(activation, activation_scales, offsets).float()
    if rows == 0:
        assert debug_output.shape == output.shape == (0, 64)
        return

    expected = _grouped_reference(activation, activation_scales, view, offsets)
    assert _normalized_l2(debug_output, expected) <= 1e-3
    assert _normalized_l2(output, expected) <= 5e-3
    assert (
        float(F.cosine_similarity(output.flatten(), expected.flatten(), dim=0)) >= 0.999
    )


@requires_sm90
def test_sm90_w4a8_m_tail_policy_covers_multiple_experts():
    torch.manual_seed(7)
    device = torch.device("cuda")
    expert_rows = (1, 64, 65)
    offsets_host = [0]
    for rows in expert_rows:
        offsets_host.append(offsets_host[-1] + rows)
    total_rows = offsets_host[-1]
    view = repack_nvfp4_sm90_v3(
        _checkpoint(device, experts=len(expert_rows), rows=64),
        group_size=128,
        residual_scheme="generic",
    )
    activation = torch.randn(total_rows, 128, device=device).to(torch.float8_e4m3fn)
    padded_stride = max(
        (total_rows + len(expert_rows) * 31) // 32 * 32,
        1,
    )
    activation_scales = _nonuniform_activation_scales(1, padded_stride, device)
    offsets = torch.tensor(offsets_host, dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        total_rows,
        view,
        payload_layout=3,
        allow_legacy_layout=True,
    )

    expected = _grouped_reference(activation, activation_scales, view, offsets)
    debug_output = runner.run_debug_fp32(activation, activation_scales, offsets)
    output = runner.run(activation, activation_scales, offsets).float()
    assert _normalized_l2(debug_output, expected) <= 1e-3
    assert _normalized_l2(output, expected) <= 5e-3


@requires_sm90
@pytest.mark.parametrize(
    "optimization",
    (
        {"decode_vector": False, "generic_decode_lut": False},
        {"overlap": False},
    ),
    ids=(
        "scalar_decode",
        "no_overlap",
    ),
)
def test_sm90_w4a8_optimization_switch_matrix(optimization):
    torch.manual_seed(13)
    device = torch.device("cuda")
    rows = 65
    columns = 256
    view = repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=128, columns=columns),
        group_size=64,
        residual_scheme="pow2",
    )
    activation = torch.randn(rows, columns, device=device).to(torch.float8_e4m3fn)
    padded_stride = (rows + 31) // 32 * 32
    activation_scales = _nonuniform_activation_scales(
        columns // 128, padded_stride, device
    )
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows,
        view,
        payload_layout=3,
        allow_legacy_layout=True,
        **optimization,
    )

    expected = _grouped_reference(activation, activation_scales, view, offsets)
    debug_output = runner.run_debug_fp32(activation, activation_scales, offsets)
    output = runner.run(activation, activation_scales, offsets).float()
    assert _normalized_l2(debug_output, expected) <= 1e-3
    assert _normalized_l2(output, expected) <= 5e-3


def _assert_sm90_w4a8_delayed_retirement_case(
    columns, rows, logical_n, group_size, decode_vector
):
    torch.manual_seed(17)
    device = torch.device("cuda")
    view = repack_nvfp4_sm90_v3(
        _checkpoint(device, rows=logical_n, columns=columns),
        group_size=group_size,
        residual_scheme="generic",
    )
    activation = torch.randn(rows, columns, device=device).to(torch.float8_e4m3fn)
    padded_stride = (rows + 31) // 32 * 32
    activation_scales = _nonuniform_activation_scales(
        columns // 128, padded_stride, device
    )
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows,
        view,
        decode_vector=decode_vector,
        generic_decode_lut=decode_vector,
        payload_layout=3,
        allow_legacy_layout=True,
    )

    # Delayed retirement must prevent stage reuse until that stage's final WGMMA group retires.
    expected = _grouped_reference(activation, activation_scales, view, offsets)
    debug_output = runner.run_debug_fp32(activation, activation_scales, offsets)
    output = runner.run(activation, activation_scales, offsets).float()
    assert _normalized_l2(debug_output, expected) <= 1e-3
    assert _normalized_l2(output, expected) <= 5e-3


@requires_sm90
@pytest.mark.parametrize("group_size", SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("rows,logical_n", ((64, 64), (65, 128)))
@pytest.mark.parametrize("columns", (128, 256, 640, 1152))
def test_sm90_w4a8_delayed_retirement_k_hazards(columns, rows, logical_n, group_size):
    _assert_sm90_w4a8_delayed_retirement_case(
        columns, rows, logical_n, group_size, True
    )


@requires_sm90
def test_sm90_w4a8_scalar_decode_reuses_pipeline_stages():
    _assert_sm90_w4a8_delayed_retirement_case(1152, 65, 128, 32, False)


@requires_sm90
def test_sm90_w4a8_sparse_expert_mapping_uses_source_offsets_and_scale_rows():
    torch.manual_seed(11)
    device = torch.device("cuda")
    base = _checkpoint(device, rows=64)
    checkpoint = NVFP4Checkpoint(
        base.packed_e2m1,
        base.scale_e4m3_per16,
        base.global_alpha,
        base.logical_shape,
        (1,),
        base.source_format_version,
    )
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=64,
        residual_scheme="generic",
    )
    offsets = torch.tensor([0, 32, 97], dtype=torch.int64, device=device)
    activation = torch.randn(97, 128, device=device).to(torch.float8_e4m3fn)
    padded_stride = (97 + 2 * 31) // 32 * 32
    activation_scales = _nonuniform_activation_scales(1, padded_stride, device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        97,
        view,
        total_experts=2,
        payload_layout=3,
        allow_legacy_layout=True,
    )
    expected = _grouped_reference(activation, activation_scales, view, offsets)
    output = torch.zeros(97, 64, dtype=torch.float32, device=device)

    runner.run_debug_fp32(
        activation,
        activation_scales,
        offsets,
        out=output,
    )

    assert torch.count_nonzero(output[:32]) == 0
    assert _normalized_l2(output[32:], expected[32:]) <= 1e-3


@requires_sm90
def test_sm90_w4a8_rejects_invalid_untrusted_offsets():
    code = r"""
import torch
from flashinfer.fused_moe.nvfp4_checkpoint import NVFP4Checkpoint
from flashinfer.fused_moe.sm90_nvfp4_repack import repack_nvfp4_sm90_v3
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
    create_sm90_push_nvfp4_w4a8_gemm,
)

rows = 32
payload = torch.randint(0, 256, (1, 64, 64), dtype=torch.uint8, device="cuda")
scales = torch.ones(1, 64, 8, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
checkpoint = NVFP4Checkpoint(
    payload, scales, torch.ones(1, device="cuda"), (1, 64, 128), (0,), "test"
)
view = repack_nvfp4_sm90_v3(checkpoint, group_size=128, residual_scheme="generic")
runner = create_sm90_push_nvfp4_w4a8_gemm(
    rows, view, payload_layout=3, allow_legacy_layout=True
)
activation = torch.randn(rows, 128, device="cuda").to(torch.float8_e4m3fn)
activation_scales = torch.ones(1, 32, dtype=torch.float32, device="cuda")
offsets = torch.tensor([0, rows + 1], dtype=torch.int64, device="cuda")
runner.run(activation, activation_scales, offsets)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-1500:]
    assert "UNEXPECTED-SURVIVAL" not in result.stdout, combined[-1500:]
    assert "sm90_w4a8_gemm: invalid offsets or expert mapping" in combined, combined[
        -1500:
    ]
    assert "ImportError" not in combined, combined[-1500:]
    assert "ModuleNotFoundError" not in combined, combined[-1500:]


@requires_sm90
def test_sm90_w4a8_large_k_soak_uses_independent_operand_reference():
    torch.manual_seed(7)
    device = torch.device("cuda")
    rows, columns = 257, 2048
    checkpoint = _checkpoint(device, rows=128, columns=columns)
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=128,
        residual_scheme="generic",
    )
    padded_stride = (rows + 31) // 32 * 32
    activation_scales = _nonuniform_activation_scales(
        columns // 128, padded_stride, device
    )
    offsets = torch.tensor([0, rows], dtype=torch.int64, device=device)
    runner = create_sm90_push_nvfp4_w4a8_gemm(
        rows, view, payload_layout=3, allow_legacy_layout=True
    )

    for shot in range(50):
        generator = torch.Generator(device=device).manual_seed(1000 + shot)
        activation = torch.randn(rows, columns, generator=generator, device=device).to(
            torch.float8_e4m3fn
        )
        expected = _grouped_reference(activation, activation_scales, view, offsets)
        actual = runner.run_debug_fp32(activation, activation_scales, offsets)
        assert _normalized_l2(actual, expected) <= 1e-3
