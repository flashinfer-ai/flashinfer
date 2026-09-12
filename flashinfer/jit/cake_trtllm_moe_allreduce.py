"""Source-built TRT-LLM MoE all-reduce kernels for SM100 and SM103."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
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
    "kernel_cake_trtllm_moe_reduction_float16_ws8_o0110",
    "kernel_cake_trtllm_moe_reduction_float16_ws8_o1110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o1110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o1110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110",
    "kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o1110",
)

_SM103_T1_KERNEL_SYMBOLS = tuple(f"{symbol}_sm103_t1" for symbol in _KERNEL_SYMBOLS)
_SM100_WS8_MID_KERNEL_SYMBOLS = tuple(
    f"{_KERNEL_SYMBOLS[index]}_sm100_ws8_mid" for index in (4, 5, 10, 11)
)

_TARGET_ARCH_BY_CAPABILITY = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

_WORKSPACE_PROTOCOL = {
    "flag_region_bytes_by_world_size": {"2": 2048, "4": 4288, "8": 8576},
    "tp4_ack_tail": {
        "offset_bytes": 4096,
        "bytes": 192,
        "generations": 3,
        "consumers": 4,
        "slot_bytes": 16,
        "generation_stride_bytes": 64,
        "empty_sentinel_u16": 32768,
        "ready_u16": 0,
    },
    "tp8_ack_tail": {
        "offset_bytes": 8192,
        "bytes": 384,
        "generations": 3,
        "consumers": 8,
        "slot_bytes": 16,
        "generation_stride_bytes": 128,
        "empty_sentinel_u16": 32768,
        "ready_u16": 0,
    },
}


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

#ifndef CAKE_MOE_AR_ENABLE_SM103_T1
#error "CAKE_MOE_AR_ENABLE_SM103_T1 must be defined by the loader"
#endif
#ifndef CAKE_MOE_AR_ENABLE_SM100_WS8_MID
#error "CAKE_MOE_AR_ENABLE_SM100_WS8_MID must be defined by the loader"
#endif

TVM_FFI_EMBED_CUBIN(cake_trtllm_moe_allreduce);

namespace cake_trtllm_moe_allreduce {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int64_t kHiddenDim = 7168;
constexpr bool kEnableSm103T1 = CAKE_MOE_AR_ENABLE_SM103_T1 != 0;
constexpr bool kEnableSm100Ws8Mid = CAKE_MOE_AR_ENABLE_SM100_WS8_MID != 0;

void CheckCudaTensor(TensorView tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
}

void CheckSameDevice(TensorView tensor, TensorView reference, const char* name) {
  CheckCudaTensor(tensor, name);
  TVM_FFI_CHECK(tensor.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as the input";
}

int32_t DTypeIndex(TensorView tensor, const char* name) {
  auto dtype = tensor.dtype();
  TVM_FFI_CHECK(dtype.lanes == 1 && dtype.bits == 16 &&
                    (dtype.code == kDLFloat || dtype.code == kDLBfloat),
                ValueError)
      << name << " must have dtype float16 or bfloat16";
  return dtype.code == kDLBfloat ? 1 : 0;
}

void CheckHiddenDim(int64_t hidden_dim) {
  TVM_FFI_CHECK(hidden_dim == kHiddenDim, ValueError)
      << "Cake MoE hidden dimension must be 7168";
}

int32_t WorldIndex(int64_t world_size) {
  if (world_size == 2) return 0;
  if (world_size == 4) return 1;
  if (world_size == 8) return 2;
  TVM_FFI_LOG_AND_THROW(ValueError) << "Cake MoE world size must be 2, 4, or 8";
  return -1;
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
  TVM_FFI_CHECK(grid_x >= 4, ValueError)
      << "Cake MoE launch grid must contain one cluster";

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

#define CAKE_MOE_AR_LAUNCH_CASE(Index, Symbol)                                  \
  case Index: {                                                                  \
    static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(                         \
        cake_trtllm_moe_allreduce, #Symbol);                                     \
    Launch(kernel, args, config);                                                 \
    return;                                                                       \
  }

void RunReduction(int64_t world_size, int64_t world_rank, int64_t token_num,
                  int64_t hidden_dim, TensorView workspace_ptrs,
                  bool launch_with_pdl, TensorView residual_in,
                  TensorView rms_gamma, double rms_eps, double scale_factor,
                  int64_t active_experts, TensorView expert_scales,
                  TensorView active_expert_tokens, TensorView token_input,
                  Optional<TensorView> moe_allreduce_out,
                  TensorView residual_out, TensorView norm_out,
                  Optional<double> weight_bias) {
  CheckCudaTensor(active_expert_tokens, "active_expert_tokens");
  CheckSameDevice(expert_scales, active_expert_tokens, "expert_scales");
  CheckSameDevice(token_input, active_expert_tokens, "token_input");
  CheckSameDevice(residual_in, active_expert_tokens, "residual_in");
  CheckSameDevice(rms_gamma, active_expert_tokens, "rms_gamma");
  CheckSameDevice(workspace_ptrs, active_expert_tokens, "workspace_ptrs");
  CheckSameDevice(residual_out, active_expert_tokens, "residual_out");
  CheckSameDevice(norm_out, active_expert_tokens, "norm_out");

  int32_t dtype_index = DTypeIndex(active_expert_tokens, "active_expert_tokens");
  CheckHiddenDim(hidden_dim);
  int32_t world_index = WorldIndex(world_size);
  int32_t output_index = moe_allreduce_out.has_value() ? 1 : 0;
  int32_t kernel_index = dtype_index * 6 + world_index * 2 + output_index;
  if (kEnableSm103T1 && token_num == 1) {
    kernel_index += 12;
  } else if (kEnableSm100Ws8Mid && world_size == 8 &&
             (token_num == 64 || token_num == 128)) {
    kernel_index = 24 + dtype_index * 2 + output_index;
  }
  int32_t rank32 = static_cast<int32_t>(world_rank);
  int32_t tokens32 = static_cast<int32_t>(token_num);
  int32_t experts32 = static_cast<int32_t>(active_experts);
  float eps32 = static_cast<float>(rms_eps);
  float weight_bias32 =
      weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
  float scale_factor32 = static_cast<float>(scale_factor);
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
  void* args[] = {&p_active, &p_scales, &p_token, &p_residual, &p_gamma,
                  &p_moe_out, &p_residual_out, &p_norm_out, &p_quant_out,
                  &p_scale_out, &p_workspace, &rank32, &tokens32, &experts32,
                  &eps32, &weight_bias32, &scale_factor32, &unused_layout};
  auto config = MakeLaunchConfig(active_expert_tokens, tokens32, launch_with_pdl);
  switch (kernel_index) {
    CAKE_MOE_AR_LAUNCH_CASE(0, kernel_cake_trtllm_moe_reduction_float16_ws2_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(1, kernel_cake_trtllm_moe_reduction_float16_ws2_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(2, kernel_cake_trtllm_moe_reduction_float16_ws4_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(3, kernel_cake_trtllm_moe_reduction_float16_ws4_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(4, kernel_cake_trtllm_moe_reduction_float16_ws8_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(5, kernel_cake_trtllm_moe_reduction_float16_ws8_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(6, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(7, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(8, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(9, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(10, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110)
    CAKE_MOE_AR_LAUNCH_CASE(11, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o1110)
    CAKE_MOE_AR_LAUNCH_CASE(12, kernel_cake_trtllm_moe_reduction_float16_ws2_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(13, kernel_cake_trtllm_moe_reduction_float16_ws2_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(14, kernel_cake_trtllm_moe_reduction_float16_ws4_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(15, kernel_cake_trtllm_moe_reduction_float16_ws4_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(16, kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(17, kernel_cake_trtllm_moe_reduction_float16_ws8_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(18, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(19, kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(20, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(21, kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(22, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(23, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o1110_sm103_t1)
    CAKE_MOE_AR_LAUNCH_CASE(24, kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm100_ws8_mid)
    CAKE_MOE_AR_LAUNCH_CASE(25, kernel_cake_trtllm_moe_reduction_float16_ws8_o1110_sm100_ws8_mid)
    CAKE_MOE_AR_LAUNCH_CASE(26, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm100_ws8_mid)
    CAKE_MOE_AR_LAUNCH_CASE(27, kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o1110_sm100_ws8_mid)
  }
  TVM_FFI_LOG_AND_THROW(ValueError)
      << "invalid Cake MoE all-reduce kernel selection";
}

#undef CAKE_MOE_AR_LAUNCH_CASE

}  // namespace cake_trtllm_moe_allreduce

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_reduction,
                              cake_trtllm_moe_allreduce::RunReduction);
"""


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_trtllm_moe_allreduce_fusion"
    if installed.is_dir():
        return installed
    return (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_trtllm_moe_allreduce_fusion"
    )


def _reject_duplicate_manifest_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise RuntimeError(
                f"Cake TRT-LLM MoE all-reduce manifest contains duplicate key {key!r}"
            )
        decoded[key] = value
    return decoded


def _source_kernel_symbols(source_bytes: bytes) -> tuple[str, ...]:
    return tuple(
        match.decode()
        for match in re.findall(
            rb"(?m)^kernel_cake_trtllm_moe_reduction_[A-Za-z0-9_]+(?=\()",
            source_bytes,
        )
    )


def _load_source_bundle_details() -> tuple[Path, bytes, bool, bool]:
    source_dir = _source_dir()
    source = source_dir / "cake_trtllm_moe_allreduce_fusion_kernels.cu"
    manifest_path = source_dir / "manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            "Cake TRT-LLM MoE all-reduce source bundle is not installed; expected "
            f"{source} and {manifest_path}"
        )
    source_bytes = source.read_bytes()
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_manifest_keys,
    )
    if not isinstance(manifest, dict):
        raise RuntimeError("Cake TRT-LLM MoE all-reduce manifest identity is invalid")
    raw_sm103_t1_symbols = manifest.get("sm103_t1_kernel_symbols")
    has_sm103_t1_symbols = raw_sm103_t1_symbols is not None
    raw_sm100_ws8_mid_symbols = manifest.get("sm100_ws8_mid_kernel_symbols")
    has_sm100_ws8_mid_symbols = raw_sm100_ws8_mid_symbols is not None
    expected = {
        "schema_version": 1,
        "architectures": list(_TARGET_ARCH_BY_CAPABILITY.values()),
        "compile_flags": ["--use_fast_math"],
        "launch": {
            "block_threads": 224,
            "cluster_dim": [4, 1, 1],
            "dynamic_smem_bytes": 256,
        },
        "constraints": {
            "dtypes": ["float16", "bfloat16"],
            "hidden_dim": 7168,
            "max_lamport_comm_size_bytes": 2145386496,
            "quantization": False,
            "world_sizes": [2, 4, 8],
        },
        "kernel_symbols": list(_KERNEL_SYMBOLS),
        "workspace_protocol": _WORKSPACE_PROTOCOL,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    if has_sm103_t1_symbols:
        expected["sm103_t1_kernel_symbols"] = list(_SM103_T1_KERNEL_SYMBOLS)
    if has_sm100_ws8_mid_symbols:
        expected["sm100_ws8_mid_kernel_symbols"] = list(_SM100_WS8_MID_KERNEL_SYMBOLS)
    if manifest != expected:
        raise RuntimeError("Cake TRT-LLM MoE all-reduce manifest identity is invalid")
    expected_source_symbols: tuple[str, ...] = _KERNEL_SYMBOLS
    if has_sm103_t1_symbols:
        expected_source_symbols += _SM103_T1_KERNEL_SYMBOLS
    if has_sm100_ws8_mid_symbols:
        expected_source_symbols += _SM100_WS8_MID_KERNEL_SYMBOLS
    if _source_kernel_symbols(source_bytes) != expected_source_symbols:
        raise RuntimeError(
            "Cake TRT-LLM MoE all-reduce source symbol inventory is invalid"
        )
    return (
        source,
        source_bytes,
        has_sm103_t1_symbols,
        has_sm100_ws8_mid_symbols,
    )


def _load_source_bundle() -> tuple[Path, bytes]:
    source, source_bytes, _, _ = _load_source_bundle_details()
    return source, source_bytes


def _target_arch(device_index: int) -> str:
    capability = torch.cuda.get_device_capability(device_index)
    arch = _TARGET_ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            "Cake TRT-LLM MoE all-reduce requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError(
            "nvcc is required to build the Cake TRT-LLM MoE all-reduce backend"
        )
    return Path(candidate).resolve()


def _enable_sm103_t1(arch: str, has_sm103_t1_symbols: bool) -> bool:
    return arch == "sm_103a" and has_sm103_t1_symbols


def _enable_sm100_ws8_mid(arch: str, has_sm100_ws8_mid_symbols: bool) -> bool:
    return arch == "sm_100a" and has_sm100_ws8_mid_symbols


def _host_compile_flags(enable_sm103_t1: bool, enable_sm100_ws8_mid: bool) -> list[str]:
    return [
        "-O3",
        f"-DCAKE_MOE_AR_ENABLE_SM103_T1={int(enable_sm103_t1)}",
        f"-DCAKE_MOE_AR_ENABLE_SM100_WS8_MID={int(enable_sm100_ws8_mid)}",
    ]


def _module_name(
    source_bytes: bytes,
    arch: str,
    nvcc: Path,
    *,
    enable_sm103_t1: bool = False,
    enable_sm100_ws8_mid: bool = False,
) -> str:
    version = subprocess.run(
        [str(nvcc), "--version"],
        text=True,
        capture_output=True,
    )
    if version.returncode != 0 or not version.stdout.strip():
        detail = version.stderr.strip() or "nvcc --version returned no version text"
        raise RuntimeError(
            "failed to identify nvcc for the Cake TRT-LLM MoE all-reduce "
            f"cache key:\n{detail}"
        )

    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update(_HOST_SOURCE.encode())
    digest.update(arch.encode())
    digest.update(str(int(enable_sm103_t1)).encode())
    digest.update(str(int(enable_sm100_ws8_mid)).encode())
    digest.update(str(nvcc).encode())
    digest.update(version.stdout.encode())
    return f"cake_trtllm_moe_allreduce_{arch}_{digest.hexdigest()[:16]}"


@functools.cache
def load(device_index: int) -> Any:
    (
        source,
        source_bytes,
        has_sm103_t1_symbols,
        has_sm100_ws8_mid_symbols,
    ) = _load_source_bundle_details()
    arch = _target_arch(device_index)
    enable_sm103_t1 = _enable_sm103_t1(arch, has_sm103_t1_symbols)
    enable_sm100_ws8_mid = _enable_sm100_ws8_mid(arch, has_sm100_ws8_mid_symbols)
    nvcc = _nvcc()
    module_name = _module_name(
        source_bytes,
        arch,
        nvcc,
        enable_sm103_t1=enable_sm103_t1,
        enable_sm100_ws8_mid=enable_sm100_ws8_mid,
    )
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / "cake_trtllm_moe_allreduce.cubin"
    with FileLock(build_dir / "cake_trtllm_moe_allreduce.lock", thread_local=False):
        if not cubin_path.is_file():
            temporary = build_dir / f"cake_trtllm_moe_allreduce.{os.getpid()}.tmp.cubin"
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
                    "Cake TRT-LLM MoE all-reduce nvcc compilation failed:\n"
                    f"{process.stderr}"
                )
            os.replace(temporary, cubin_path)

        return cpp.load_inline(
            module_name,
            cpp_sources=_HOST_SOURCE,
            embed_cubin={"cake_trtllm_moe_allreduce": cubin_path.read_bytes()},
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=_host_compile_flags(enable_sm103_t1, enable_sm100_ws8_mid),
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )


__all__ = ["load"]
