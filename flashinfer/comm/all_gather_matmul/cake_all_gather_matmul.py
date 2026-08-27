# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-built Blackwell all-gather matmul backend."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tvm_ffi import cpp

from flashinfer.jit import env as jit_env


_BLOCK_M = 128
_CHUNK_ROWS = 19 * _BLOCK_M
_K = 8192
_N = 2048
_PACKED_QKV_N = 2560
_TENSOR_MAP_BYTES = 128
_DESCRIPTOR_COUNT = 3
_DESCRIPTOR_CACHE_MAX_ENTRIES = 256
_MAIN_KERNEL_COUNT = 4
_SUPPORTED_WORLD_SIZES = frozenset((2, 4))
_SUPPORTED_DTYPES = frozenset((torch.bfloat16, torch.float16))
_KERNEL_SYMBOLS = (
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p0",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p1",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p0",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p1",
    "kernel_cake_blackwell_all_gather_matmul_float16_ws2",
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws2",
    "kernel_cake_blackwell_all_gather_matmul_float16_ws4",
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4",
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "arch",
        "compile_flags",
        "tma_abi",
        "kernel_count",
        "launch",
        "constraints",
        "kernel_symbols",
        "route_coverage",
        "source_sha256",
    }
)
_SMEM_TOTAL_PATTERN = re.compile(rb"^#define SMEM_TOTAL ([1-9][0-9]*)$", re.MULTILINE)
_CONSTRAINTS = {
    "dtypes": ["float16", "bfloat16"],
    "k": 8192,
    "m_multiple": 128,
    "n": 2048,
    "world_sizes": [2, 4],
}
_ROUTE_COVERAGE = {
    "ws2": {
        "barrier": list(_KERNEL_SYMBOLS[:2]),
        "main": {
            "bfloat16": _KERNEL_SYMBOLS[5],
            "float16": _KERNEL_SYMBOLS[4],
        },
    },
    "ws4": {
        "barrier": list(_KERNEL_SYMBOLS[2:4]),
        "main": {
            "bfloat16": _KERNEL_SYMBOLS[7],
            "float16": _KERNEL_SYMBOLS[6],
        },
    },
}

_HOST_SOURCE = r"""
// Source-level cubin launcher for Blackwell all-gather matmul.
#include <cuda.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cstdint>
#include <cstring>

TVM_FFI_EMBED_CUBIN(CAKE_MODULE_IDENT);

namespace flashinfer_cake_all_gather_matmul {

using tvm::ffi::TensorView;

constexpr int64_t kTensorMapBytes = sizeof(CUtensorMap);
constexpr int64_t kDescriptorCount = 3;
constexpr int32_t kMainThreads = 192;
constexpr int32_t kMainSmemBytes = CAKE_MAIN_SMEM_BYTES;
constexpr bool kPackedQkvExperimentSupported =
    CAKE_PACKED_QKV_EXPERIMENT_SUPPORTED;

static_assert(sizeof(CUtensorMap) == 128);

inline void CheckCudaTensor(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
}

inline void CheckCpuTensor(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCPU, ValueError)
      << name << " must be a CPU tensor";
}

inline void CheckContiguous(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError)
      << name << " must be contiguous";
}

inline void CheckSameDevice(const TensorView& tensor, const TensorView& reference,
                            const char* name) {
  TVM_FFI_CHECK(tensor.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as inp";
}

inline CUtensorMapDataType TensorMapDtype(const TensorView& tensor) {
  const DLDataType dtype = tensor.dtype();
  if (dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  }
  if (dtype.code == kDLFloat && dtype.bits == 16 && dtype.lanes == 1) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  }
  TVM_FFI_THROW(TypeError) << "inp must have bfloat16 or float16 dtype";
}

inline void CheckDescriptorInputs(const TensorView& inp,
                                  const TensorView& scratch,
                                  const TensorView& weight,
                                  int64_t world_size, int64_t rows) {
  CheckCudaTensor(inp, "inp");
  CheckCudaTensor(scratch, "scratch");
  CheckCudaTensor(weight, "weight");
  CheckContiguous(inp, "inp");
  CheckContiguous(scratch, "scratch");
  CheckContiguous(weight, "weight");
  CheckSameDevice(scratch, inp, "scratch");
  CheckSameDevice(weight, inp, "weight");
  TVM_FFI_CHECK(world_size == 2 || world_size == 4, ValueError)
      << "world_size must be 2 or 4";
  TVM_FFI_CHECK(rows > 0 && rows % 128 == 0, ValueError)
      << "rows must be a positive multiple of 128";
  TVM_FFI_CHECK(inp.ndim() == 2 && inp.size(0) == rows && inp.size(1) == 8192,
                ValueError)
      << "inp must have shape [rows, 8192]";
  TVM_FFI_CHECK(scratch.ndim() == 3 && scratch.size(0) == world_size &&
                    scratch.size(1) == rows && scratch.size(2) == 8192,
                ValueError)
      << "scratch must have shape [world_size, rows, 8192]";
  TVM_FFI_CHECK(weight.ndim() == 2 && weight.size(0) == 8192 &&
                    (weight.size(1) == 2048 || weight.size(1) == 2560),
                ValueError)
      << "weight must have shape [8192, 2048] or [8192, 2560]";
  const DLDataType dtype = inp.dtype();
  for (const auto* tensor : {&scratch, &weight}) {
    const DLDataType other = tensor->dtype();
    TVM_FFI_CHECK(other.code == dtype.code && other.bits == dtype.bits &&
                      other.lanes == dtype.lanes,
                  TypeError)
        << "inp, scratch, and weight must have the same dtype";
  }
  (void)TensorMapDtype(inp);
}

inline void CheckCommonInputs(const TensorView& inp, const TensorView& scratch,
                              const TensorView& weight, const TensorView& out,
                              int64_t world_size, int64_t rows) {
  CheckDescriptorInputs(inp, scratch, weight, world_size, rows);
  CheckCudaTensor(out, "out");
  CheckContiguous(out, "out");
  CheckSameDevice(out, inp, "out");
  TVM_FFI_CHECK(out.ndim() == 2 && out.size(0) == world_size * rows &&
                    out.size(1) == weight.size(1),
                ValueError)
      << "out must have shape [world_size * rows, weight.size(1)]";
  const DLDataType dtype = inp.dtype();
  const DLDataType out_dtype = out.dtype();
  TVM_FFI_CHECK(out_dtype.code == dtype.code && out_dtype.bits == dtype.bits &&
                    out_dtype.lanes == dtype.lanes,
                TypeError)
      << "inp and out must have the same dtype";
}

inline CUtensorMap EncodeActivationMap(const TensorView& tensor, int64_t rows,
                                       const char* name) {
  uint64_t global_dim[3] = {64, static_cast<uint64_t>(rows), 128};
  uint64_t global_strides[2] = {8192 * 2, 64 * 2};
  uint32_t box_dim[3] = {64, 128, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, TensorMapDtype(tensor), 3, tensor.data_ptr(), global_dim,
      global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled failed for " << name
      << " with CUresult=" << static_cast<int>(result);
  return map;
}

inline CUtensorMap EncodeWeightMap(const TensorView& weight) {
  const uint64_t n = static_cast<uint64_t>(weight.size(1));
  uint64_t global_dim[3] = {n, 8192, 1};
  uint64_t global_strides[2] = {n * 2, 8192ULL * n * 2ULL};
  uint32_t box_dim[3] = {64, 64, 1};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, TensorMapDtype(weight), 3, weight.data_ptr(), global_dim,
      global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled failed for weight with CUresult="
      << static_cast<int>(result);
  return map;
}

void PrepareDescriptors(TensorView inp, TensorView scratch, TensorView weight,
                        TensorView host_descriptor_storage,
                        int64_t world_size, int64_t rows) {
  CheckDescriptorInputs(inp, scratch, weight, world_size, rows);
  CheckCpuTensor(host_descriptor_storage, "host_descriptor_storage");
  CheckContiguous(host_descriptor_storage, "host_descriptor_storage");
  const DLDataType storage_dtype = host_descriptor_storage.dtype();
  TVM_FFI_CHECK(storage_dtype.code == kDLUInt && storage_dtype.bits == 8 &&
                    storage_dtype.lanes == 1,
                TypeError)
      << "host_descriptor_storage must have uint8 dtype";
  TVM_FFI_CHECK(host_descriptor_storage.numel() >=
                    kDescriptorCount * kTensorMapBytes,
                ValueError)
      << "host_descriptor_storage is too small";

  const std::array<CUtensorMap, kDescriptorCount> maps = {
      EncodeActivationMap(inp, rows, "inp"),
      EncodeActivationMap(scratch, world_size * rows, "scratch"),
      EncodeWeightMap(weight),
  };
  std::memcpy(host_descriptor_storage.data_ptr(), maps.data(), sizeof(maps));
}

inline tvm::ffi::CubinKernel& BarrierKernel(int64_t world_size, int64_t phase) {
  TVM_FFI_CHECK(world_size == 2 || world_size == 4, ValueError)
      << "world_size must be 2 or 4";
  TVM_FFI_CHECK(phase == 0 || phase == 1, ValueError)
      << "phase must be 0 or 1";
  static auto ws2_p0 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p0");
  static auto ws2_p1 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p1");
  static auto ws4_p0 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p0");
  static auto ws4_p1 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p1");
  if (world_size == 2) {
    return phase == 0 ? ws2_p0 : ws2_p1;
  }
  return phase == 0 ? ws4_p0 : ws4_p1;
}

inline tvm::ffi::CubinKernel& MainKernel(int64_t world_size, int64_t dtype_code,
                                         int64_t n) {
  TVM_FFI_CHECK(world_size == 2 || world_size == 4, ValueError)
      << "world_size must be 2 or 4";
  TVM_FFI_CHECK(dtype_code == 0 || dtype_code == 1, ValueError)
      << "dtype_code must be 0 (bfloat16) or 1 (float16)";
  TVM_FFI_CHECK(n == 2048 || n == 2560, ValueError)
      << "n must be 2048 or 2560";
  if (n == 2560) {
    TVM_FFI_CHECK(kPackedQkvExperimentSupported && world_size == 4 &&
                      dtype_code == 0,
                  ValueError)
        << "the packed-QKV experiment requires SM103, world_size=4, and bfloat16";
  }
  static auto bf16_ws2 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws2");
  static auto f16_ws2 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_float16_ws2");
  static auto bf16_ws4 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4");
  static auto f16_ws4 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_float16_ws4");
  if (world_size == 2) {
    return dtype_code == 0 ? bf16_ws2 : f16_ws2;
  }
  return dtype_code == 0 ? bf16_ws4 : f16_ws4;
}

void RunBarrier(TensorView flag_peers, int64_t world_size, int64_t rank,
                int64_t phase, int64_t cuda_stream) {
  CheckCudaTensor(flag_peers, "flag_peers");
  CheckContiguous(flag_peers, "flag_peers");
  const DLDataType dtype = flag_peers.dtype();
  TVM_FFI_CHECK(dtype.code == kDLInt && dtype.bits == 64 && dtype.lanes == 1,
                TypeError)
      << "flag_peers must have int64 dtype";
  TVM_FFI_CHECK(flag_peers.ndim() == 1 && flag_peers.numel() == world_size,
                ValueError)
      << "flag_peers must contain one pointer per rank";
  TVM_FFI_CHECK(rank >= 0 && rank < world_size, ValueError)
      << "rank is outside the process group";

  CUstream stream = reinterpret_cast<CUstream>(
      static_cast<uintptr_t>(cuda_stream));
  int32_t world = static_cast<int32_t>(world_size);
  int32_t local_rank = static_cast<int32_t>(rank);
  void* flags = flag_peers.data_ptr();
  void* args[] = {&world, &local_rank, &flags};
  auto& kernel = BarrierKernel(world_size, phase);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(args, tvm::ffi::dim3(1, 1, 1), tvm::ffi::dim3(32, 1, 1),
                    stream, 0));
}

void RunMain(TensorView inp, TensorView scratch, TensorView weight,
             TensorView out, TensorView descriptor_storage, TensorView ready,
             int64_t world_size, int64_t rank, int64_t rows, int64_t dtype_code,
             int64_t cuda_stream) {
  CheckCommonInputs(inp, scratch, weight, out, world_size, rows);
  CheckCudaTensor(descriptor_storage, "descriptor_storage");
  CheckCudaTensor(ready, "ready");
  CheckSameDevice(descriptor_storage, inp, "descriptor_storage");
  CheckSameDevice(ready, inp, "ready");
  CheckContiguous(descriptor_storage, "descriptor_storage");
  CheckContiguous(ready, "ready");
  TVM_FFI_CHECK(rank >= 0 && rank < world_size, ValueError)
      << "rank is outside the process group";
  TVM_FFI_CHECK(descriptor_storage.numel() >= kDescriptorCount * kTensorMapBytes,
                ValueError)
      << "descriptor_storage is too small";
  const DLDataType ready_dtype = ready.dtype();
  TVM_FFI_CHECK(ready_dtype.code == kDLUInt && ready_dtype.bits == 32 &&
                    ready_dtype.lanes == 1,
                TypeError)
      << "ready must have uint32 dtype";

  auto* descriptors = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  void* inp_map = descriptors + 0 * kTensorMapBytes;
  void* scratch_map = descriptors + 1 * kTensorMapBytes;
  void* weight_map = descriptors + 2 * kTensorMapBytes;
  void* output_ptr = out.data_ptr();
  void* scratch_ptr = scratch.data_ptr();
  void* ready_ptr = ready.data_ptr();
  int32_t local_rank = static_cast<int32_t>(rank);
  int32_t local_rows = static_cast<int32_t>(rows);
  void* args[] = {&inp_map, &scratch_map, &weight_map, &output_ptr,
                  &scratch_ptr, &ready_ptr, &local_rank, &local_rows};

  const int64_t n = weight.size(1);
  auto& kernel = MainKernel(world_size, dtype_code, n);
  namespace cuda_api = tvm::ffi::cuda_api;
  static signed char smem_configured[4][64] = {};
  TVM_FFI_CHECK(inp.device().device_id >= 0 && inp.device().device_id < 64,
                RuntimeError)
      << "CUDA device id exceeds the dynamic-smem cache";
  const int route = (world_size == 4 ? 2 : 0) + (dtype_code == 1 ? 1 : 0);
  if (smem_configured[route][inp.device().device_id] == 0) {
    auto device = cuda_api::GetDeviceHandle(inp.device().device_id);
    const auto result = cuda_api::SetKernelMaxDynamicSharedMem(
        kernel.GetHandle(), kMainSmemBytes, device);
    TVM_FFI_CHECK(result == cuda_api::kSuccess, RuntimeError)
        << "setting max dynamic shared memory failed";
    smem_configured[route][inp.device().device_id] = 1;
  }

  const int64_t chunk_rows = rows < 2432 ? rows : 2432;
  const uint32_t grid_x =
      static_cast<uint32_t>((chunk_rows / 128) * (n / 256));
  CUstream stream = reinterpret_cast<CUstream>(
      static_cast<uintptr_t>(cuda_stream));
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(args, tvm::ffi::dim3(grid_x, 1, 1),
                    tvm::ffi::dim3(kMainThreads, 1, 1), stream,
                    kMainSmemBytes));
}

}  // namespace flashinfer_cake_all_gather_matmul

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    prepare_descriptors,
    flashinfer_cake_all_gather_matmul::PrepareDescriptors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_barrier, flashinfer_cake_all_gather_matmul::RunBarrier);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_main, flashinfer_cake_all_gather_matmul::RunMain);
"""


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_all_gather_matmul"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[3] / "csrc" / "cake_all_gather_matmul"


def _target_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm_100a"
    if capability == (10, 3):
        return "sm_103a"
    raise ValueError(
        "The Cake all-gather matmul backend requires SM100 or SM103, got "
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
        raise RuntimeError(
            "nvcc is required to build the Cake all-gather matmul backend"
        )
    return Path(candidate).resolve()


def _reject_duplicate_manifest_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(
                f"Cake all-gather matmul manifest has duplicate key {key!r}"
            )
        document[key] = value
    return document


def _resolved_main_smem_bytes(source: bytes) -> int:
    values = [int(match) for match in _SMEM_TOTAL_PATTERN.findall(source)]
    if len(values) != _MAIN_KERNEL_COUNT or len(set(values)) != 1:
        raise RuntimeError(
            "Cake all-gather matmul source must expose one uniform SMEM_TOTAL "
            "for each main kernel"
        )
    return values[0]


def _launch_contract(source: bytes) -> dict[str, Any]:
    return {
        "barrier": {
            "block_threads": 32,
            "dynamic_smem_bytes": 0,
            "grid": [1, 1, 1],
        },
        "main": {
            "block_threads": 192,
            "dynamic_smem_bytes": _resolved_main_smem_bytes(source),
            "grid_x": "(min(M, 2432) / 128) * 8",
        },
    }


def _render_host_source(module_ident: str, manifest: dict[str, Any]) -> str:
    main_smem_bytes = int(manifest["launch"]["main"]["dynamic_smem_bytes"])
    if main_smem_bytes <= 0:
        raise RuntimeError(
            "Cake all-gather matmul main dynamic shared memory is invalid"
        )
    rendered = (
        _HOST_SOURCE.replace("CAKE_MODULE_IDENT", module_ident)
        .replace("CAKE_MAIN_SMEM_BYTES", str(main_smem_bytes))
        .replace(
            "CAKE_PACKED_QKV_EXPERIMENT_SUPPORTED",
            "true" if manifest["arch"] == "sm_103a" else "false",
        )
    )
    if any(
        placeholder in rendered
        for placeholder in (
            "CAKE_MODULE_IDENT",
            "CAKE_MAIN_SMEM_BYTES",
            "CAKE_PACKED_QKV_EXPERIMENT_SUPPORTED",
        )
    ):
        raise RuntimeError("Cake all-gather matmul host launch template is incomplete")
    return rendered


def _program_source(arch: str) -> tuple[Path, dict[str, Any]]:
    directory = _source_dir() / arch.replace("sm_", "sm")
    source = directory / "cake_all_gather_matmul_kernels.cu"
    manifest_path = directory / "manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Cake all-gather matmul source package is incomplete for {arch}"
        )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_manifest_keys,
    )
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    expected = {
        "schema_version": 1,
        "arch": arch,
        "compile_flags": ["--use_fast_math"],
        "tma_abi": "pointer",
        "kernel_count": 8,
        "launch": _launch_contract(source_bytes),
        "constraints": _CONSTRAINTS,
        "kernel_symbols": list(_KERNEL_SYMBOLS),
        "route_coverage": _ROUTE_COVERAGE,
        "source_sha256": source_sha256,
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_KEYS
        or manifest != expected
    ):
        raise RuntimeError(
            f"Cake all-gather matmul manifest identity is invalid for {arch}"
        )
    return source, manifest


@functools.cache
def _load_program(arch: str):
    source, manifest = _program_source(arch)
    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True).encode())
    digest.update(_HOST_SOURCE.encode())
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    key = digest.hexdigest()[:16]
    module_ident = f"cake_all_gather_matmul_{arch}_{key}"
    host_source = _render_host_source(module_ident, manifest)
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_ident
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin = build_dir / f"{module_ident}.cubin"
    if not cubin.is_file():
        temporary = build_dir / f"{module_ident}.{os.getpid()}.tmp.cubin"
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
            str(temporary),
        ]
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cake all-gather matmul nvcc failed for {arch}:\n{process.stderr}"
            )
        os.replace(temporary, cubin)
    return cpp.load_inline(
        module_ident,
        cpp_sources=host_source,
        embed_cubin={module_ident: cubin.read_bytes()},
        extra_include_paths=[str(nvcc.parent.parent / "include")],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


@dataclass
class _LaunchState:
    lock: Any = field(default_factory=RLock, repr=False)
    next_phase: int = 0
    tail_event: Any = None
    tail_stream: int | None = None
    flags: Any = None
    flag_handle: Any = None
    flag_peers: Any = None
    poisoned: bool = False


@dataclass
class _Workspace:
    scratch: torch.Tensor
    scratch_handle: Any
    comm_stream: torch.cuda.Stream
    descriptor_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = field(
        default_factory=OrderedDict, repr=False
    )


_CACHE_LOCK = RLock()
_LAUNCH_STATES: dict[tuple[int, int, str], _LaunchState] = {}
_WORKSPACES: dict[tuple[int, int, str, torch.dtype, int, int], _Workspace] = {}


def _tensor_fingerprint(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.index,
    )


def _group_name(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    if name is None:
        raise ValueError("group must expose a stable group_name for symmetric memory")
    return str(name)


def _validate_inputs(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    packed_qkv_experiment: bool = False,
) -> tuple[int, int, int, str]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("an initialized NCCL process group is required")
    if str(dist.get_backend(group)).lower() != "nccl":
        raise ValueError("group must use the NCCL backend")
    if inp.device.type != "cuda" or w.device.type != "cuda":
        raise ValueError("inp and w must be CUDA tensors")
    device_index = inp.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if w.device != inp.device:
        raise ValueError("inp and w must be on the same CUDA device")
    if not inp.is_contiguous() or not w.is_contiguous():
        raise ValueError("inp and w must be contiguous")
    if inp.dtype not in _SUPPORTED_DTYPES or w.dtype != inp.dtype:
        raise ValueError("inp and w must have the same bfloat16 or float16 dtype")
    if inp.ndim != 2 or w.ndim != 2:
        raise ValueError("inp and w must both be two-dimensional")
    rows, k = (int(dim) for dim in inp.shape)
    if rows <= 0 or rows % _BLOCK_M:
        raise ValueError("inp.shape[0] must be a positive multiple of 128")
    if packed_qkv_experiment:
        if k != _K or tuple(w.shape) != (_K, _PACKED_QKV_N):
            raise ValueError(
                "the packed-QKV experiment requires exact K=8192 and N=2560"
            )
    elif k != _K or tuple(w.shape) != (_K, _N):
        raise ValueError("the Cake backend requires exact K=8192 and N=2048")
    world_size = int(dist.get_world_size(group))
    rank = int(dist.get_rank(group))
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError("the Cake backend requires process-group world size 2 or 4")
    if not 0 <= rank < world_size:
        raise RuntimeError("process-group rank is outside its world size")
    if str(symm_mem.get_backend(inp.device)).upper() != "NVSHMEM":
        raise ValueError(
            "the Cake backend requires the NVSHMEM symmetric-memory backend"
        )
    arch = _target_arch(inp.device)
    if packed_qkv_experiment and (
        arch != "sm_103a" or inp.dtype != torch.bfloat16 or world_size != 4
    ):
        raise ValueError(
            "the packed-QKV experiment requires SM103, bfloat16, and world size 4"
        )
    return device_index, rank, world_size, _group_name(group)


def _ensure_launch_state(
    state: _LaunchState,
    *,
    device_index: int,
    rank: int,
    world_size: int,
    group_name: str,
) -> None:
    if state.flags is not None:
        return
    flags = symm_mem.empty(2, dtype=torch.uint32, device=device_index)
    handle = symm_mem.rendezvous(flags, group=group_name)
    if int(handle.rank) != rank or int(handle.world_size) != world_size:
        raise RuntimeError("barrier symmetric-memory topology does not match group")
    flags.zero_()
    peer_ptrs = [
        int(handle.get_buffer(peer, (2,), torch.uint32, 0).data_ptr())
        for peer in range(world_size)
    ]
    state.flags = flags
    state.flag_handle = handle
    state.flag_peers = torch.tensor(peer_ptrs, dtype=torch.int64, device=device_index)


def _workspace(
    *,
    device_index: int,
    group: dist.ProcessGroup,
    group_name: str,
    dtype: torch.dtype,
    world_size: int,
    rows: int,
) -> _Workspace:
    key = (device_index, id(group), group_name, dtype, world_size, rows)
    workspace = _WORKSPACES.get(key)
    if workspace is not None:
        return workspace
    scratch = symm_mem.empty(world_size, rows, _K, dtype=dtype, device=device_index)
    scratch_handle = symm_mem.rendezvous(scratch, group=group_name)
    workspace = _Workspace(
        scratch=scratch,
        scratch_handle=scratch_handle,
        comm_stream=torch.cuda.Stream(device=device_index),
    )
    _WORKSPACES[key] = workspace
    return workspace


def _descriptor_storage(
    workspace: _Workspace,
    module: Any,
    inp: torch.Tensor,
    w: torch.Tensor,
    *,
    device_index: int,
    main_stream: torch.cuda.Stream,
    world_size: int,
    rows: int,
) -> torch.Tensor:
    fingerprint = (
        _tensor_fingerprint(inp),
        _tensor_fingerprint(workspace.scratch),
        _tensor_fingerprint(w),
    )
    with _CACHE_LOCK:
        descriptors = workspace.descriptor_cache.get(fingerprint)
        if descriptors is not None:
            workspace.descriptor_cache.move_to_end(fingerprint)
            return descriptors
    host_descriptors = torch.empty(
        _DESCRIPTOR_COUNT * _TENSOR_MAP_BYTES,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    descriptors = torch.empty(
        _DESCRIPTOR_COUNT * _TENSOR_MAP_BYTES,
        dtype=torch.uint8,
        device=device_index,
    )
    module.prepare_descriptors(
        inp,
        workspace.scratch,
        w,
        host_descriptors,
        world_size,
        rows,
    )
    with torch.cuda.stream(main_stream):
        descriptors.copy_(host_descriptors, non_blocking=True)
    with _CACHE_LOCK:
        cached = workspace.descriptor_cache.get(fingerprint)
        if cached is not None:
            workspace.descriptor_cache.move_to_end(fingerprint)
            return cached
        workspace.descriptor_cache[fingerprint] = descriptors
        while len(workspace.descriptor_cache) > _DESCRIPTOR_CACHE_MAX_ENTRIES:
            workspace.descriptor_cache.popitem(last=False)
    return descriptors


def _run_cake_validated(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool,
    packed_qkv_experiment: bool,
) -> torch.Tensor:
    if packed_qkv_experiment:
        device_index, rank, world_size, group_name = _validate_inputs(
            inp, w, group, packed_qkv_experiment=True
        )
        output_n = _PACKED_QKV_N
    else:
        device_index, rank, world_size, group_name = _validate_inputs(inp, w, group)
        output_n = _N
    rows = int(inp.shape[0])
    arch = _target_arch(inp.device)
    module = _load_program(arch)
    state_key = (device_index, id(group), group_name)
    with _CACHE_LOCK:
        state = _LAUNCH_STATES.setdefault(state_key, _LaunchState())

    with state.lock:
        if state.poisoned:
            raise RuntimeError(
                "Cake all-gather matmul state is poisoned by a prior failed collective"
            )
        try:
            workspace_key = (
                device_index,
                id(group),
                group_name,
                inp.dtype,
                world_size,
                rows,
            )
            with _CACHE_LOCK:
                workspace = _WORKSPACES.get(workspace_key)
            main_stream = torch.cuda.current_stream(device_index)
            if state.flags is None or workspace is None:
                # Symmetric allocation and rendezvous are host operations, so
                # a stream dependency cannot order them after queued caller work.
                # Wait only on a cache miss; steady-state launches stay async.
                main_stream.synchronize()
                if state.tail_event is not None:
                    state.tail_event.synchronize()

            _ensure_launch_state(
                state,
                device_index=device_index,
                rank=rank,
                world_size=world_size,
                group_name=group_name,
            )
            if workspace is None:
                with _CACHE_LOCK:
                    workspace = _workspace(
                        device_index=device_index,
                        group=group,
                        group_name=group_name,
                        dtype=inp.dtype,
                        world_size=world_size,
                        rows=rows,
                    )

            descriptors = _descriptor_storage(
                workspace,
                module,
                inp,
                w,
                device_index=device_index,
                main_stream=main_stream,
                world_size=world_size,
                rows=rows,
            )
            output = torch.empty(
                world_size * rows, output_n, dtype=inp.dtype, device=device_index
            )

            chunk_size = min(rows, _CHUNK_ROWS)
            num_chunks = (rows + chunk_size - 1) // chunk_size
            signal_pad = workspace.scratch_handle.get_signal_pad(
                rank, (world_size, num_chunks), torch.uint32, 0
            )
            inp.record_stream(main_stream)
            inp.record_stream(workspace.comm_stream)
            w.record_stream(main_stream)
            descriptors.record_stream(main_stream)
            main_stream_id = int(main_stream.cuda_stream)
            if state.tail_event is not None and state.tail_stream != main_stream_id:
                main_stream.wait_event(state.tail_event)

            phase = state.next_phase
            module.run_barrier(
                state.flag_peers, world_size, rank, phase, main_stream_id
            )
            workspace.comm_stream.wait_stream(main_stream)
            with torch.cuda.stream(workspace.comm_stream):
                for shift in range(1, world_size):
                    peer = (rank + shift) % world_size
                    peer_scratch = workspace.scratch_handle.get_remote_tensor(
                        peer, workspace.scratch.shape, workspace.scratch.dtype
                    )[rank]
                    peer_signal = workspace.scratch_handle.get_signal_pad(
                        peer, (world_size, num_chunks), torch.uint32, 0
                    )
                    for chunk_idx in range(num_chunks):
                        begin = chunk_idx * chunk_size
                        end = min(begin + chunk_size, rows)
                        peer_scratch[begin:end].copy_(inp[begin:end], non_blocking=True)
                        torch.ops.symm_mem.stream_write_value32_(
                            peer_signal[rank], chunk_idx, 1
                        )

            module.run_main(
                inp,
                workspace.scratch,
                w,
                output,
                descriptors,
                signal_pad,
                world_size,
                rank,
                rows,
                int(inp.dtype == torch.float16),
                main_stream_id,
            )
            signal_pad.zero_()
            main_stream.wait_stream(workspace.comm_stream)
            if state.tail_event is None:
                state.tail_event = torch.cuda.Event(enable_timing=False)
            state.tail_event.record(main_stream)
            state.next_phase = 1 - phase
            state.tail_stream = main_stream_id
            if verbose and rank == 0:
                print(
                    "Cake all-gather matmul: "
                    f"arch={arch}, world_size={world_size}, M={rows}, "
                    f"K={_K}, N={output_n}"
                )
            return output
        except Exception:
            state.poisoned = True
            raise


def all_gather_matmul_cake(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str,
    verbose: bool = False,
) -> torch.Tensor:
    """Run the exact Blackwell source-built backend for one NCCL group.

    The input is local-only and need not be remotely addressable. NVSHMEM is
    still required for the backend's internal symmetric scratch and flags.
    """

    if backend != "cake":
        raise ValueError("backend must be exactly 'cake'")
    return _run_cake_validated(
        inp,
        w,
        group,
        verbose=verbose,
        packed_qkv_experiment=False,
    )


def _all_gather_matmul_cake_packed_qkv_sm103_tp4(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Run the private SM103/BF16/TP4 packed-QKV arithmetic experiment."""

    return _run_cake_validated(
        inp,
        w,
        group,
        verbose=verbose,
        packed_qkv_experiment=True,
    )


__all__ = ["all_gather_matmul_cake"]
