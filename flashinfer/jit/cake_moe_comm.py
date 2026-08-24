"""Source-built Cake MoE communication kernels for SM100."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock
from tvm_ffi import cpp

from . import env as jit_env


_KERNEL_SYMBOLS = (
    "kernel_cake_trtllm_moe_reduction_float16_ws2_o0110",
    "kernel_cake_trtllm_moe_reduction_float16_ws2_o1110",
    "kernel_cake_trtllm_moe_reduction_float16_ws4_o0110",
    "kernel_cake_trtllm_moe_reduction_float16_ws4_o1110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o1110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o1110",
    "kernel_cake_trtllm_moe_finalize_float16_ws2_o110",
    "kernel_cake_trtllm_moe_finalize_float16_ws4_o110",
    "kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110",
    "kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110",
)


_HOST_SOURCE = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>

TVM_FFI_EMBED_CUBIN(cake_moe_comm);

namespace cake_moe_comm {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

void CheckCudaTensor(TensorView tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
}

void CheckSameDevice(TensorView tensor, TensorView reference, const char* name) {
  CheckCudaTensor(tensor, name);
  TVM_FFI_CHECK(tensor.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as the input";
}

void* OptionalPtr(Optional<TensorView> tensor) {
  return tensor.has_value() ? tensor.value().data_ptr() : nullptr;
}

tvm::ffi::cuda_api::LaunchConfig MakeLaunchConfig(TensorView input, int32_t tokens,
                                                   bool launch_with_pdl) {
  CUdevice cuda_device;
  TVM_FFI_CHECK(cuDeviceGet(&cuda_device, input.device().device_id) == CUDA_SUCCESS,
                RuntimeError)
      << "failed to resolve the CUDA device";
  int32_t sm_count = 0;
  TVM_FFI_CHECK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                     cuda_device) == CUDA_SUCCESS,
                RuntimeError)
      << "failed to query the SM count";
  int32_t grid_x = std::min(sm_count, tokens * 4);
  grid_x = (grid_x / 4) * 4;
  TVM_FFI_CHECK(grid_x >= 4, ValueError) << "Cake MoE launch grid must contain one cluster";

  DLDevice device = input.device();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  tvm::ffi::cuda_api::LaunchConfig config;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  static thread_local CUlaunchAttribute attrs[2];
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[0].value.clusterDim.x = 4;
  attrs[0].value.clusterDim.y = 1;
  attrs[0].value.clusterDim.z = 1;
  int32_t num_attrs = 1;
  if (launch_with_pdl) {
    attrs[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attrs[1].value.programmaticStreamSerializationAllowed = 1;
    num_attrs = 2;
  }
  config.gridDimX = static_cast<uint32_t>(grid_x);
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 224;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 256;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;
#else
  static thread_local cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {4, 1, 1};
  int32_t num_attrs = 1;
  if (launch_with_pdl) {
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    num_attrs = 2;
  }
  config.gridDim = {static_cast<uint32_t>(grid_x), 1, 1};
  config.blockDim = {224, 1, 1};
  config.dynamicSmemBytes = 256;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;
#endif
  return config;
}

template <typename Kernel>
void Launch(Kernel& kernel, void** args, tvm::ffi::cuda_api::LaunchConfig config) {
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(args, config));
}

#define CAKE_MOE_LAUNCH_CASE(Index, Symbol)                                      \
  case Index: {                                                                  \
    static auto kernel =                                                         \
        TVM_FFI_EMBED_CUBIN_GET_KERNEL(cake_moe_comm, #Symbol);                  \
    Launch(kernel, args, config);                                                 \
    return;                                                                       \
  }

void RunReduction(int64_t world_size, int64_t world_rank, int64_t token_num,
                  int64_t hidden_dim, TensorView workspace_ptrs, bool launch_with_pdl,
                  TensorView residual_in, TensorView rms_gamma, double rms_eps,
                  double scale_factor, int64_t active_experts, TensorView expert_scales,
                  TensorView active_expert_tokens, TensorView token_input,
                  Optional<TensorView> moe_allreduce_out, TensorView residual_out,
                  TensorView norm_out, Optional<double> weight_bias) {
  CheckCudaTensor(active_expert_tokens, "active_expert_tokens");
  CheckSameDevice(expert_scales, active_expert_tokens, "expert_scales");
  CheckSameDevice(token_input, active_expert_tokens, "token_input");
  CheckSameDevice(residual_in, active_expert_tokens, "residual_in");
  CheckSameDevice(rms_gamma, active_expert_tokens, "rms_gamma");
  CheckSameDevice(workspace_ptrs, active_expert_tokens, "workspace_ptrs");
  CheckSameDevice(residual_out, active_expert_tokens, "residual_out");
  CheckSameDevice(norm_out, active_expert_tokens, "norm_out");

  int32_t dtype_index = active_expert_tokens.dtype().code == kDLBfloat ? 1 : 0;
  int32_t world_index = world_size == 4 ? 1 : 0;
  int32_t output_index = moe_allreduce_out.has_value() ? 1 : 0;
  int32_t kernel_index = dtype_index * 4 + world_index * 2 + output_index;
  int32_t rank32 = static_cast<int32_t>(world_rank);
  int32_t tokens32 = static_cast<int32_t>(token_num);
  int32_t experts32 = static_cast<int32_t>(active_experts);
  float eps32 = static_cast<float>(rms_eps);
  float weight_bias32 = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
  float scale_factor32 = static_cast<float>(scale_factor);
  int64_t unused_comm_stride = 0;
  int32_t unused_layout = 0;

  void* p_active = active_expert_tokens.data_ptr();
  void* p_scales = expert_scales.data_ptr();
  void* p_token = token_input.data_ptr();
  void* p_residual = residual_in.data_ptr();
  void* p_gamma = rms_gamma.data_ptr();
  void* p_moe_out = OptionalPtr(moe_allreduce_out);
  void* p_residual_out = residual_out.data_ptr();
  void* p_norm_out = norm_out.data_ptr();
  void* p_quant_out = nullptr;
  void* p_scale_out = nullptr;
  void* p_workspace = workspace_ptrs.data_ptr();
  void* p_unused_lamport = nullptr;
  void* p_unused_completion = nullptr;
  void* args[] = {&p_active, &p_scales, &p_token, &p_residual, &p_gamma,
                  &p_moe_out, &p_residual_out, &p_norm_out, &p_quant_out,
                  &p_scale_out, &p_workspace, &rank32, &p_unused_lamport,
                  &p_unused_completion, &tokens32, &experts32, &eps32,
                  &weight_bias32, &scale_factor32, &unused_comm_stride,
                  &unused_layout};
  auto config = MakeLaunchConfig(active_expert_tokens, tokens32, launch_with_pdl);
  switch (kernel_index) {
    CAKE_MOE_LAUNCH_CASE(0, kernel_cake_trtllm_moe_reduction_float16_ws2_o0110)
    CAKE_MOE_LAUNCH_CASE(1, kernel_cake_trtllm_moe_reduction_float16_ws2_o1110)
    CAKE_MOE_LAUNCH_CASE(2, kernel_cake_trtllm_moe_reduction_float16_ws4_o0110)
    CAKE_MOE_LAUNCH_CASE(3, kernel_cake_trtllm_moe_reduction_float16_ws4_o1110)
    CAKE_MOE_LAUNCH_CASE(4, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110)
    CAKE_MOE_LAUNCH_CASE(5, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o1110)
    CAKE_MOE_LAUNCH_CASE(6, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110)
    CAKE_MOE_LAUNCH_CASE(7, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o1110)
  }
  TVM_FFI_LOG_AND_THROW(ValueError) << "invalid Cake MoE reduction kernel selection";
}

void RunFinalize(TensorView allreduce_in, TensorView residual_in, TensorView norm_weight,
                 TensorView inverse_indices, TensorView norm_out, TensorView residual_out,
                 TensorView workspace_ptrs, bool launch_with_pdl, int64_t world_rank,
                 int64_t world_size, double eps, Optional<TensorView> shared_expert_output,
                 TensorView expert_scales, double routed_scaling_factor,
                 Optional<double> weight_bias) {
  CheckCudaTensor(allreduce_in, "allreduce_in");
  CheckSameDevice(residual_in, allreduce_in, "residual_in");
  CheckSameDevice(norm_weight, allreduce_in, "norm_weight");
  CheckSameDevice(inverse_indices, allreduce_in, "inverse_indices");
  CheckSameDevice(expert_scales, allreduce_in, "expert_scales");
  CheckSameDevice(workspace_ptrs, allreduce_in, "workspace_ptrs");
  CheckSameDevice(residual_out, allreduce_in, "residual_out");
  CheckSameDevice(norm_out, allreduce_in, "norm_out");

  int32_t dtype_index = allreduce_in.dtype().code == kDLBfloat ? 1 : 0;
  int32_t world_index = world_size == 4 ? 1 : 0;
  int32_t kernel_index = dtype_index * 2 + world_index;
  int32_t rank32 = static_cast<int32_t>(world_rank);
  int32_t tokens32 = static_cast<int32_t>(residual_in.size(0));
  int32_t top_k32 = static_cast<int32_t>(inverse_indices.size(-1));
  int32_t has_shared32 = shared_expert_output.has_value() ? 1 : 0;
  float routed32 = static_cast<float>(routed_scaling_factor);
  float eps32 = static_cast<float>(eps);
  float weight_bias32 = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
  float unused_scale_factor = 1.0f;
  int64_t unused_comm_stride = 0;

  void* p_allreduce = allreduce_in.data_ptr();
  void* p_indices = inverse_indices.data_ptr();
  void* p_scales = expert_scales.data_ptr();
  void* p_shared = OptionalPtr(shared_expert_output);
  void* p_residual = residual_in.data_ptr();
  void* p_gamma = norm_weight.data_ptr();
  void* p_residual_out = residual_out.data_ptr();
  void* p_norm_out = norm_out.data_ptr();
  void* p_quant_out = nullptr;
  void* p_scale_out = nullptr;
  void* p_workspace = workspace_ptrs.data_ptr();
  void* p_unused_lamport = nullptr;
  void* p_unused_completion = nullptr;
  void* args[] = {&p_allreduce, &p_indices, &p_scales, &p_shared, &p_residual,
                  &p_gamma, &p_residual_out, &p_norm_out, &p_quant_out,
                  &p_scale_out, &p_workspace, &rank32, &p_unused_lamport,
                  &p_unused_completion, &tokens32, &top_k32, &has_shared32,
                  &routed32, &eps32, &weight_bias32, &unused_scale_factor,
                  &unused_comm_stride};
  auto config = MakeLaunchConfig(allreduce_in, tokens32, launch_with_pdl);
  switch (kernel_index) {
    CAKE_MOE_LAUNCH_CASE(0, kernel_cake_trtllm_moe_finalize_float16_ws2_o110)
    CAKE_MOE_LAUNCH_CASE(1, kernel_cake_trtllm_moe_finalize_float16_ws4_o110)
    CAKE_MOE_LAUNCH_CASE(2, kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110)
    CAKE_MOE_LAUNCH_CASE(3, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110)
  }
  TVM_FFI_LOG_AND_THROW(ValueError) << "invalid Cake MoE finalize kernel selection";
}

#undef CAKE_MOE_LAUNCH_CASE

}  // namespace cake_moe_comm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_reduction, cake_moe_comm::RunReduction);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_finalize, cake_moe_comm::RunFinalize);
"""


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_moe_allreduce_fusion"
    if installed.is_dir():
        return installed
    return Path(__file__).resolve().parents[2] / "csrc" / "cake_moe_allreduce_fusion"


def _reject_duplicate_manifest_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise RuntimeError(
                f"Cake MoE communication manifest contains duplicate key {key!r}"
            )
        decoded[key] = value
    return decoded


def _load_source_bundle() -> tuple[Path, bytes]:
    source_dir = _source_dir()
    source = source_dir / "cake_moe_allreduce_fusion_kernels.cu"
    manifest_path = source_dir / "manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "Cake MoE communication source bundle is not installed; expected "
            f"{source} and {manifest_path}"
        )
    source_bytes = source.read_bytes()
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_manifest_keys,
    )
    if not isinstance(manifest, dict):
        raise RuntimeError("Cake MoE communication manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "arch": "sm_100a",
        "compile_flags": ["--use_fast_math"],
        "launch": {
            "block_threads": 224,
            "cluster_dim": [4, 1, 1],
            "dynamic_smem_bytes": 256,
        },
        "constraints": {
            "dtypes": ["float16", "bfloat16"],
            "hidden_dim": 7168,
            "max_tokens": 2048,
            "quantization": False,
            "world_sizes": [2, 4],
        },
        "kernel_symbols": list(_KERNEL_SYMBOLS),
    }
    expected_keys = set(expected) | {"source_sha256"}
    if set(manifest) != expected_keys:
        missing = sorted(expected_keys - set(manifest))
        unexpected = sorted(set(manifest) - expected_keys)
        raise RuntimeError(
            "Cake MoE communication manifest top-level keys mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Cake MoE communication manifest mismatch for {key}")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if manifest.get("source_sha256") != source_sha256:
        raise RuntimeError("Cake MoE communication source checksum mismatch")
    return source, source_bytes


def _target_arch(device_index: int) -> str:
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (10, 0):
        raise ValueError(
            "Cake MoE communication requires SM100, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return "sm_100a"


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the Cake MoE communication backend")
    return Path(candidate).resolve()


@functools.cache
def load(device_index: int) -> Any:
    source, source_bytes = _load_source_bundle()
    arch = _target_arch(device_index)
    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update(_HOST_SOURCE.encode())
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    module_name = f"cake_moe_comm_{arch}_{digest.hexdigest()[:16]}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / "cake_moe_comm.cubin"
    with FileLock(build_dir / "cake_moe_comm.lock", thread_local=False):
        if not cubin_path.is_file():
            temporary = build_dir / f"cake_moe_comm.{os.getpid()}.tmp.cubin"
            command = [
                str(nvcc),
                "-cubin",
                f"-arch={arch}",
                "--std=c++17",
                "-O3",
                "--use_fast_math",
                str(source),
                "-o",
                str(temporary),
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            if process.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "Cake MoE communication nvcc compilation failed:\n"
                    f"{process.stderr}"
                )
            os.replace(temporary, cubin_path)

        return cpp.load_inline(
            module_name,
            cpp_sources=_HOST_SOURCE,
            embed_cubin={"cake_moe_comm": cubin_path.read_bytes()},
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )


__all__ = ["load"]
