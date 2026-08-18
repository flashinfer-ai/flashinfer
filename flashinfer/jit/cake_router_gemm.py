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

# Source-built Cake Router GEMM programs for datacenter Blackwell.

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch
from tvm_ffi import cpp

from . import env as jit_env


_HOST_SOURCE = r"""
// Cake Router GEMM source-level cubin launcher.
#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <cstdint>

TVM_FFI_EMBED_CUBIN(CAKE_MODULE_IDENT);

namespace cake_router_gemm {

using tvm::ffi::TensorView;

void CheckCudaTensor(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
}

void Run(TensorView mat_a, TensorView mat_b, TensorView out,
         bool launch_with_pdl, int64_t out_is_bf16) {
  CheckCudaTensor(mat_a, "mat_a");
  CheckCudaTensor(mat_b, "mat_b");
  CheckCudaTensor(out, "out");
  TVM_FFI_CHECK(mat_a.device().device_id == mat_b.device().device_id &&
                    mat_a.device().device_id == out.device().device_id,
                ValueError)
      << "mat_a, mat_b, and out must be on the same CUDA device";
  TVM_FFI_CHECK(out_is_bf16 == 0 || out_is_bf16 == 1, ValueError)
      << "out_is_bf16 must be 0 or 1";

  DLDevice device = mat_a.device();
  cudaStream_t stream =
      (cudaStream_t)TVMFFIEnvGetStream(device.device_type, device.device_id);
  void* p_mat_a = mat_a.data_ptr();
  void* p_mat_b = mat_b.data_ptr();
  void* p_out_f32 = out.data_ptr();
  void* p_out_bf16 = out.data_ptr();
  int32_t num_experts = (int32_t)mat_b.size(1);
  int32_t output_selector = (int32_t)out_is_bf16;
  void* args[] = {&p_mat_a, &p_mat_b, &p_out_f32, &p_out_bf16,
                  &num_experts, &output_selector};

  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT, "CAKE_KERNEL_SYMBOL");
  tvm::ffi::cuda_api::LaunchConfig config;
  int num_attrs = 0;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attrs[1];
  if (launch_with_pdl) {
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attrs[0].value.programmaticStreamSerializationAllowed = 1;
    num_attrs = 1;
  }
  config.gridDimX = (uint32_t)num_experts;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = CAKE_SMEM_BYTES;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;
#else
  cudaLaunchAttribute attrs[1];
  if (launch_with_pdl) {
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    num_attrs = 1;
  }
  config.gridDim = {(uint32_t)num_experts, 1, 1};
  config.blockDim = {128, 1, 1};
  config.dynamicSmemBytes = CAKE_SMEM_BYTES;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = num_attrs;
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(args, config));
}

}  // namespace cake_router_gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_router_gemm::Run);
"""


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_router_gemm"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "csrc" / "cake_router_gemm"


def _target_arch() -> str:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 0):
        return "sm_100a"
    if capability == (10, 3):
        return "sm_103a"
    raise ValueError(
        "Cake Router GEMM requires SM100 or SM103, got "
        f"SM{capability[0]}{capability[1]}"
    )


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the Cake Router GEMM backend")
    return Path(candidate).resolve()


def _program_source(num_tokens: int, hidden_dim: int) -> Path:
    if not 1 <= num_tokens <= 16:
        raise ValueError(f"num_tokens must be in [1, 16], got {num_tokens}")
    if hidden_dim not in (6144, 7168):
        raise ValueError(f"hidden_dim must be 6144 or 7168, got {hidden_dim}")
    source = (
        _source_dir()
        / f"cake_router_gemm_m{num_tokens}_k{hidden_dim}_device.cu"
    )
    if not source.is_file():
        raise RuntimeError(
            f"Cake Router GEMM source package is incomplete for M={num_tokens}, "
            f"K={hidden_dim}"
        )
    return source


@functools.cache
def _load_program(num_tokens: int, hidden_dim: int, arch: str):
    source = _program_source(num_tokens, hidden_dim)
    nvcc = _nvcc()
    module_ident = f"cake_router_gemm_m{num_tokens}_k{hidden_dim}"
    kernel_symbol = f"kernel_cake_blackwell_router_gemm_m{num_tokens}_k{hidden_dim}"
    host_source = (
        _HOST_SOURCE.replace("CAKE_MODULE_IDENT", module_ident)
        .replace("CAKE_KERNEL_SYMBOL", kernel_symbol)
        .replace("CAKE_SMEM_BYTES", str(num_tokens * 16))
    )

    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(host_source.encode())
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    key = digest.hexdigest()[:16]
    build_dir = jit_env.FLASHINFER_JIT_DIR / f"{module_ident}_{arch}_{key}"
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / f"{module_ident}.cubin"
    if not cubin_path.is_file():
        temporary_cubin = build_dir / f"{module_ident}.{os.getpid()}.tmp.cubin"
        command = [
            str(nvcc),
            "-cubin",
            f"-arch={arch}",
            "--std=c++17",
            "-O3",
            "--use_fast_math",
            "-I",
            str(nvcc.parent.parent / "include"),
            str(source),
            "-o",
            str(temporary_cubin),
        ]
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            temporary_cubin.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cake Router GEMM nvcc failed for M={num_tokens}, "
                f"K={hidden_dim}, arch={arch}:\n{process.stderr}"
            )
        os.replace(temporary_cubin, cubin_path)

    return cpp.load_inline(
        f"{module_ident}_{arch}_{key}",
        cpp_sources=host_source,
        embed_cubin={module_ident: cubin_path.read_bytes()},
        extra_include_paths=[str(nvcc.parent.parent / "include")],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


def run(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool,
) -> None:
    module = _load_program(int(mat_a.shape[0]), int(mat_a.shape[1]), _target_arch())
    module.run(mat_a, mat_b, out, launch_with_pdl, int(out.dtype == torch.bfloat16))


__all__ = ["run"]
