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
_TP8_PACKED_QKV_N = 1280
_PACKED_QKV_N = 2560
_PREPARED_PACKED_QKV_N_BY_WORLD_SIZE = {
    4: _PACKED_QKV_N,
    8: _TP8_PACKED_QKV_N,
}
_TENSOR_MAP_BYTES = 128
_DESCRIPTOR_COUNT = 3
_DESCRIPTOR_CACHE_MAX_ENTRIES = 256
_MAIN_KERNEL_COUNT = 6
_SUPPORTED_WORLD_SIZES = frozenset((2, 4, 8))
_SUPPORTED_DTYPES = frozenset((torch.bfloat16, torch.float16))
_KERNEL_SYMBOLS = (
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p0",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws2_p1",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p0",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws4_p1",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws8_p0",
    "kernel_cake_blackwell_all_gather_matmul_barrier_ws8_p1",
    "kernel_cake_blackwell_all_gather_matmul_float16_ws2",
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws2",
    "kernel_cake_blackwell_all_gather_matmul_float16_ws4",
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws4",
    "kernel_cake_blackwell_all_gather_matmul_float16_ws8",
    "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws8",
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
_COMMON_CONSTRAINTS: dict[str, Any] = {
    "dtypes": ["float16", "bfloat16"],
    "k": 8192,
    "m_multiple": 128,
    "n_by_world_size": {
        "2": [2048],
        "4": [2048],
        "8": [2048],
    },
    "world_sizes": [2, 4, 8],
}
_ROUTE_COVERAGE = {
    "ws2": {
        "barrier": list(_KERNEL_SYMBOLS[:2]),
        "main": {
            "bfloat16": _KERNEL_SYMBOLS[7],
            "float16": _KERNEL_SYMBOLS[6],
        },
    },
    "ws4": {
        "barrier": list(_KERNEL_SYMBOLS[2:4]),
        "main": {
            "bfloat16": _KERNEL_SYMBOLS[9],
            "float16": _KERNEL_SYMBOLS[8],
        },
    },
    "ws8": {
        "barrier": list(_KERNEL_SYMBOLS[4:6]),
        "main": {
            "bfloat16": _KERNEL_SYMBOLS[11],
            "float16": _KERNEL_SYMBOLS[10],
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
  TVM_FFI_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
                ValueError)
      << "world_size must be 2, 4, or 8";
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
                    (weight.size(1) == 1280 || weight.size(1) == 2048 ||
                     weight.size(1) == 2560),
                ValueError)
      << "weight must have shape [8192, 1280], [8192, 2048], or [8192, 2560]";
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
  TVM_FFI_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
                ValueError)
      << "world_size must be 2, 4, or 8";
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
  static auto ws8_p0 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws8_p0");
  static auto ws8_p1 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_barrier_ws8_p1");
  if (world_size == 2) {
    return phase == 0 ? ws2_p0 : ws2_p1;
  }
  if (world_size == 4) {
    return phase == 0 ? ws4_p0 : ws4_p1;
  }
  return phase == 0 ? ws8_p0 : ws8_p1;
}

inline tvm::ffi::CubinKernel& MainKernel(int64_t world_size, int64_t dtype_code,
                                         int64_t n) {
  TVM_FFI_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
                ValueError)
      << "world_size must be 2, 4, or 8";
  TVM_FFI_CHECK(dtype_code == 0 || dtype_code == 1, ValueError)
      << "dtype_code must be 0 (bfloat16) or 1 (float16)";
  TVM_FFI_CHECK(n == 1280 || n == 2048 || n == 2560, ValueError)
      << "n must be 1280, 2048, or 2560";
  if (n == 2560) {
    TVM_FFI_CHECK(kPackedQkvExperimentSupported && world_size == 4 &&
                      dtype_code == 0,
                  ValueError)
        << "the packed-QKV experiment requires SM103, world_size=4, and bfloat16";
  }
  if (n == 1280) {
    TVM_FFI_CHECK(kPackedQkvExperimentSupported && world_size == 8 &&
                      dtype_code == 0,
                  ValueError)
        << "N=1280 requires the SM103, world_size=8, bfloat16 packed-QKV route";
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
  static auto bf16_ws8 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_bfloat16_ws8");
  static auto f16_ws8 = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
      CAKE_MODULE_IDENT,
      "kernel_cake_blackwell_all_gather_matmul_float16_ws8");
  if (world_size == 2) {
    return dtype_code == 0 ? bf16_ws2 : f16_ws2;
  }
  if (world_size == 4) {
    return dtype_code == 0 ? bf16_ws4 : f16_ws4;
  }
  return dtype_code == 0 ? bf16_ws8 : f16_ws8;
}

inline tvm::ffi::CubinKernel& ConfiguredMainKernel(
    int64_t world_size, int64_t dtype_code, int64_t n, int64_t device_id) {
  auto& kernel = MainKernel(world_size, dtype_code, n);
  namespace cuda_api = tvm::ffi::cuda_api;
  static signed char smem_configured[6][64] = {};
  TVM_FFI_CHECK(device_id >= 0 && device_id < 64, RuntimeError)
      << "CUDA device id exceeds the dynamic-smem cache";
  const int topology = world_size == 2 ? 0 : (world_size == 4 ? 1 : 2);
  const int route = topology * 2 + (dtype_code == 1 ? 1 : 0);
  if (smem_configured[route][device_id] == 0) {
    auto device = cuda_api::GetDeviceHandle(device_id);
    const auto result = cuda_api::SetKernelMaxDynamicSharedMem(
        kernel.GetHandle(), kMainSmemBytes, device);
    TVM_FFI_CHECK(result == cuda_api::kSuccess, RuntimeError)
        << "setting max dynamic shared memory failed";
    smem_configured[route][device_id] = 1;
  }
  return kernel;
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
             int64_t ready_target, int64_t world_size, int64_t rank,
             int64_t rows, int64_t dtype_code, int64_t cuda_stream) {
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
  const DLDataType descriptor_dtype = descriptor_storage.dtype();
  TVM_FFI_CHECK(descriptor_dtype.code == kDLUInt &&
                    descriptor_dtype.bits == 8 &&
                    descriptor_dtype.lanes == 1,
                TypeError)
      << "descriptor_storage must have uint8 dtype";
  const DLDataType ready_dtype = ready.dtype();
  TVM_FFI_CHECK(ready_dtype.code == kDLUInt && ready_dtype.bits == 32 &&
                    ready_dtype.lanes == 1,
                TypeError)
      << "ready must have uint32 dtype";
  const int64_t chunk_rows = rows < 2432 ? rows : 2432;
  const int64_t num_chunks = (rows + chunk_rows - 1) / chunk_rows;
  TVM_FFI_CHECK(ready.ndim() == 2 && ready.size(0) == world_size &&
                    ready.size(1) == num_chunks,
                ValueError)
      << "ready must have shape [world_size, num_chunks]";
  TVM_FFI_CHECK(ready_target > 0 && ready_target <= UINT32_MAX, ValueError)
      << "ready_target must be in [1, UINT32_MAX]";

  auto* descriptors = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  void* inp_map = descriptors + 0 * kTensorMapBytes;
  void* scratch_map = descriptors + 1 * kTensorMapBytes;
  void* weight_map = descriptors + 2 * kTensorMapBytes;
  void* output_ptr = out.data_ptr();
  void* scratch_ptr = scratch.data_ptr();
  void* ready_ptr = ready.data_ptr();
  uint32_t ready_value = static_cast<uint32_t>(ready_target);
  int32_t local_rank = static_cast<int32_t>(rank);
  int32_t local_rows = static_cast<int32_t>(rows);
  void* args[] = {&inp_map, &scratch_map, &weight_map, &output_ptr,
                  &scratch_ptr, &ready_ptr, &ready_value, &local_rank,
                  &local_rows};

  const int64_t n = weight.size(1);
  auto& kernel = ConfiguredMainKernel(world_size, dtype_code, n,
                                      inp.device().device_id);

  const uint32_t grid_x =
      static_cast<uint32_t>((chunk_rows / 128) * (n / 256));
  CUstream stream = reinterpret_cast<CUstream>(
      static_cast<uintptr_t>(cuda_stream));
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      kernel.Launch(args, tvm::ffi::dim3(grid_x, 1, 1),
                    tvm::ffi::dim3(kMainThreads, 1, 1), stream,
                    kMainSmemBytes));
}

void RunPreparedPackedQkv(
    TensorView inp, TensorView scratch, TensorView weight, TensorView out,
    TensorView descriptor_storage, TensorView ready, TensorView flag_peers,
    TensorView peer_scratch_0, TensorView peer_signal_0,
    TensorView peer_scratch_1, TensorView peer_signal_1,
    TensorView peer_scratch_2, TensorView peer_signal_2,
    TensorView peer_scratch_3, TensorView peer_signal_3,
    TensorView peer_scratch_4, TensorView peer_signal_4,
    TensorView peer_scratch_5, TensorView peer_signal_5,
    TensorView peer_scratch_6, TensorView peer_signal_6,
    int64_t world_size, int64_t rank, int64_t rows, int64_t phase,
    int64_t ready_target, int64_t main_cuda_stream, int64_t comm_cuda_stream,
    int64_t bridge_cuda_event, int64_t expected_scratch_ptr,
    int64_t expected_ready_ptr, int64_t expected_peer_scratch_0,
    int64_t expected_peer_signal_0, int64_t expected_peer_scratch_1,
    int64_t expected_peer_signal_1, int64_t expected_peer_scratch_2,
    int64_t expected_peer_signal_2, int64_t expected_peer_scratch_3,
    int64_t expected_peer_signal_3, int64_t expected_peer_scratch_4,
    int64_t expected_peer_signal_4, int64_t expected_peer_scratch_5,
    int64_t expected_peer_signal_5, int64_t expected_peer_scratch_6,
    int64_t expected_peer_signal_6) {
  CheckCommonInputs(inp, scratch, weight, out, world_size, rows);
  CheckCudaTensor(descriptor_storage, "descriptor_storage");
  CheckCudaTensor(ready, "ready");
  CheckCudaTensor(flag_peers, "flag_peers");
  CheckSameDevice(descriptor_storage, inp, "descriptor_storage");
  CheckSameDevice(ready, inp, "ready");
  CheckSameDevice(flag_peers, inp, "flag_peers");
  CheckContiguous(descriptor_storage, "descriptor_storage");
  CheckContiguous(ready, "ready");
  CheckContiguous(flag_peers, "flag_peers");
  TVM_FFI_CHECK(descriptor_storage.numel() >=
                    kDescriptorCount * kTensorMapBytes,
                ValueError)
      << "descriptor_storage is too small";
  const DLDataType descriptor_dtype = descriptor_storage.dtype();
  TVM_FFI_CHECK(descriptor_dtype.code == kDLUInt &&
                    descriptor_dtype.bits == 8 &&
                    descriptor_dtype.lanes == 1,
                TypeError)
      << "descriptor_storage must have uint8 dtype";
  const DLDataType ready_dtype = ready.dtype();
  TVM_FFI_CHECK(ready_dtype.code == kDLUInt && ready_dtype.bits == 32 &&
                    ready_dtype.lanes == 1,
                TypeError)
      << "ready must have uint32 dtype";
  const DLDataType flag_dtype = flag_peers.dtype();
  TVM_FFI_CHECK(flag_dtype.code == kDLInt && flag_dtype.bits == 64 &&
                    flag_dtype.lanes == 1,
                TypeError)
      << "flag_peers must have int64 dtype";
  TVM_FFI_CHECK(
      kPackedQkvExperimentSupported &&
          ((world_size == 4 && weight.size(1) == 2560) ||
           (world_size == 8 && weight.size(1) == 1280)),
      ValueError)
      << "prepared packed-QKV launch requires SM103 and exact profile "
         "world_size=4,N=2560 or world_size=8,N=1280";
  TVM_FFI_CHECK(rank >= 0 && rank < world_size, ValueError)
      << "rank is outside the process group";
  TVM_FFI_CHECK(flag_peers.ndim() == 1 &&
                    flag_peers.numel() == world_size,
                ValueError)
      << "flag_peers must contain one pointer per rank";
  const DLDataType dtype = inp.dtype();
  TVM_FFI_CHECK(dtype.code == kDLBfloat && dtype.bits == 16 &&
                    dtype.lanes == 1,
                TypeError)
      << "prepared packed-QKV launch requires bfloat16";
  const int64_t chunk_rows = rows < 2432 ? rows : 2432;
  const int64_t num_chunks = (rows + chunk_rows - 1) / chunk_rows;
  TVM_FFI_CHECK(ready.ndim() == 2 && ready.size(0) == world_size &&
                    ready.size(1) == num_chunks,
                ValueError)
      << "ready must have shape [world_size, num_chunks]";
  TVM_FFI_CHECK(ready_target > 0 && ready_target <= UINT32_MAX, ValueError)
      << "ready_target must be in [1, UINT32_MAX]";
  TVM_FFI_CHECK(comm_cuda_stream != 0 && bridge_cuda_event != 0,
                ValueError)
      << "prepared packed-QKV communication CUDA handles must be nonzero";

  const std::array<const TensorView*, 7> peer_scratch = {
      &peer_scratch_0, &peer_scratch_1, &peer_scratch_2, &peer_scratch_3,
      &peer_scratch_4, &peer_scratch_5, &peer_scratch_6};
  const std::array<const TensorView*, 7> peer_signal = {
      &peer_signal_0, &peer_signal_1, &peer_signal_2, &peer_signal_3,
      &peer_signal_4, &peer_signal_5, &peer_signal_6};
  const std::array<int64_t, 7> expected_peer_scratch = {
      expected_peer_scratch_0, expected_peer_scratch_1,
      expected_peer_scratch_2, expected_peer_scratch_3,
      expected_peer_scratch_4, expected_peer_scratch_5,
      expected_peer_scratch_6};
  const std::array<int64_t, 7> expected_peer_signal = {
      expected_peer_signal_0, expected_peer_signal_1, expected_peer_signal_2,
      expected_peer_signal_3, expected_peer_signal_4, expected_peer_signal_5,
      expected_peer_signal_6};
  TVM_FFI_CHECK(expected_scratch_ptr != 0 && expected_ready_ptr != 0 &&
                    reinterpret_cast<uintptr_t>(scratch.data_ptr()) ==
                        static_cast<uintptr_t>(expected_scratch_ptr) &&
                    reinterpret_cast<uintptr_t>(ready.data_ptr()) ==
                        static_cast<uintptr_t>(expected_ready_ptr),
                ValueError)
      << "prepared workspace storage changed after binding";
  for (int64_t peer_index = 0; peer_index < world_size - 1; ++peer_index) {
    const auto& peer_scratch_tensor = *peer_scratch[peer_index];
    const auto& peer_signal_tensor = *peer_signal[peer_index];
    CheckCudaTensor(peer_scratch_tensor, "prepared peer scratch");
    CheckCudaTensor(peer_signal_tensor, "prepared peer signal");
    CheckSameDevice(peer_scratch_tensor, inp, "prepared peer scratch");
    CheckSameDevice(peer_signal_tensor, inp, "prepared peer signal");
    CheckContiguous(peer_scratch_tensor, "prepared peer scratch");
    CheckContiguous(peer_signal_tensor, "prepared peer signal");
    const DLDataType peer_scratch_dtype = peer_scratch_tensor.dtype();
    const DLDataType peer_signal_dtype = peer_signal_tensor.dtype();
    TVM_FFI_CHECK(peer_scratch_dtype.code == dtype.code &&
                      peer_scratch_dtype.bits == dtype.bits &&
                      peer_scratch_dtype.lanes == dtype.lanes &&
                      peer_scratch_tensor.ndim() == 2 &&
                      peer_scratch_tensor.size(0) == rows &&
                      peer_scratch_tensor.size(1) == 8192,
                  ValueError)
        << "prepared peer scratch binding changed";
    TVM_FFI_CHECK(peer_signal_dtype.code == kDLUInt &&
                      peer_signal_dtype.bits == 32 &&
                      peer_signal_dtype.lanes == 1 &&
                      peer_signal_tensor.ndim() == 1 &&
                      peer_signal_tensor.size(0) == num_chunks,
                  ValueError)
        << "prepared peer signal binding changed";
    TVM_FFI_CHECK(expected_peer_scratch[peer_index] != 0 &&
                      expected_peer_signal[peer_index] != 0 &&
                      reinterpret_cast<uintptr_t>(
                          peer_scratch_tensor.data_ptr()) ==
                          static_cast<uintptr_t>(
                              expected_peer_scratch[peer_index]) &&
                      reinterpret_cast<uintptr_t>(
                          peer_signal_tensor.data_ptr()) ==
                          static_cast<uintptr_t>(
                              expected_peer_signal[peer_index]),
                  ValueError)
        << "prepared peer storage changed after binding";
  }

  // Resolve every kernel/capability check before this rank enters the collective.
  (void)BarrierKernel(world_size, phase);
  (void)ConfiguredMainKernel(world_size, 0, weight.size(1),
                             inp.device().device_id);

  CUstream main_stream = reinterpret_cast<CUstream>(
      static_cast<uintptr_t>(main_cuda_stream));
  CUstream comm_stream = reinterpret_cast<CUstream>(
      static_cast<uintptr_t>(comm_cuda_stream));
  CUevent bridge_event = reinterpret_cast<CUevent>(
      static_cast<uintptr_t>(bridge_cuda_event));

  RunBarrier(flag_peers, world_size, rank, phase, main_cuda_stream);
  TVM_FFI_CHECK(cuEventRecord(bridge_event, main_stream) == CUDA_SUCCESS,
                RuntimeError)
      << "recording the prepared main-to-comm event failed";
  TVM_FFI_CHECK(cuStreamWaitEvent(comm_stream, bridge_event, 0) == CUDA_SUCCESS,
                RuntimeError)
      << "waiting for the prepared barrier on the comm stream failed";

  const size_t row_bytes = 8192 * sizeof(uint16_t);
  const CUdeviceptr input_base = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(inp.data_ptr()));
  for (int64_t peer_index = 0; peer_index < world_size - 1; ++peer_index) {
    const CUdeviceptr peer_scratch_base =
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(
            peer_scratch[peer_index]->data_ptr()));
    const CUdeviceptr peer_signal_base =
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(
            peer_signal[peer_index]->data_ptr()));
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      const int64_t begin = chunk * chunk_rows;
      const int64_t end = (begin + chunk_rows) < rows
                              ? (begin + chunk_rows)
                              : rows;
      const size_t offset = static_cast<size_t>(begin) * row_bytes;
      const size_t bytes = static_cast<size_t>(end - begin) * row_bytes;
      TVM_FFI_CHECK(cuMemcpyDtoDAsync(peer_scratch_base + offset,
                                     input_base + offset, bytes,
                                     comm_stream) == CUDA_SUCCESS,
                    RuntimeError)
          << "prepared peer copy submission failed";
      TVM_FFI_CHECK(
          cuStreamWriteValue32(
              comm_stream,
              peer_signal_base + static_cast<size_t>(chunk) * sizeof(uint32_t),
              static_cast<uint32_t>(ready_target),
              CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS,
          RuntimeError)
          << "prepared peer signal submission failed";
    }
  }

  RunMain(inp, scratch, weight, out, descriptor_storage, ready, ready_target,
          world_size, rank, rows, 0, main_cuda_stream);
  TVM_FFI_CHECK(cuEventRecord(bridge_event, comm_stream) == CUDA_SUCCESS,
                RuntimeError)
      << "recording the prepared comm-to-main event failed";
  TVM_FFI_CHECK(cuStreamWaitEvent(main_stream, bridge_event, 0) == CUDA_SUCCESS,
                RuntimeError)
      << "waiting for prepared peer copies on the main stream failed";
}

}  // namespace flashinfer_cake_all_gather_matmul

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    prepare_descriptors,
    flashinfer_cake_all_gather_matmul::PrepareDescriptors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_barrier, flashinfer_cake_all_gather_matmul::RunBarrier);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_main, flashinfer_cake_all_gather_matmul::RunMain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_prepared_packed_qkv,
    flashinfer_cake_all_gather_matmul::RunPreparedPackedQkv);
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
            "grid_x": "(min(M, 2432) / 128) * (N / 256)",
        },
    }


def _constraints_for_arch(arch: str) -> dict[str, Any]:
    constraints = {
        "dtypes": list(_COMMON_CONSTRAINTS["dtypes"]),
        "k": _COMMON_CONSTRAINTS["k"],
        "m_multiple": _COMMON_CONSTRAINTS["m_multiple"],
        "n_by_world_size": {
            key: list(values)
            for key, values in _COMMON_CONSTRAINTS["n_by_world_size"].items()
        },
        "world_sizes": list(_COMMON_CONSTRAINTS["world_sizes"]),
    }
    if arch == "sm_103a":
        constraints["prepared_packed_qkv"] = {
            "dtypes": ["bfloat16"],
            "n_by_world_size": {"4": [2560], "8": [1280]},
        }
    return constraints


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
        "kernel_count": 12,
        "launch": _launch_contract(source_bytes),
        "constraints": _constraints_for_arch(arch),
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
    ready_epoch: int = 0
    tail_event: Any = None
    tail_stream: int | None = None
    flags: Any = None
    flag_handle: Any = None
    flag_peers: Any = None
    initialization_event: Any = None
    initialization_stream: int | None = None
    poisoned: bool = False


@dataclass
class _PreparedDescriptorEntry:
    descriptors: torch.Tensor
    ready_event: torch.cuda.Event
    ready_stream: int


@dataclass
class _Workspace:
    scratch: torch.Tensor
    scratch_handle: Any
    comm_stream: torch.cuda.Stream
    bridge_event: torch.cuda.Event
    descriptor_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = field(
        default_factory=OrderedDict, repr=False
    )
    prepared_descriptor_cache: OrderedDict[
        tuple[Any, ...], _PreparedDescriptorEntry
    ] = field(default_factory=OrderedDict, repr=False)


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
    world_size = int(dist.get_world_size(group))
    rank = int(dist.get_rank(group))
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "the Cake backend requires process-group world size 2, 4, or 8"
        )
    if not 0 <= rank < world_size:
        raise RuntimeError("process-group rank is outside its world size")
    if packed_qkv_experiment:
        expected_n = _PREPARED_PACKED_QKV_N_BY_WORLD_SIZE.get(world_size)
        if expected_n is None or k != _K or tuple(w.shape) != (_K, expected_n):
            raise ValueError(
                "the packed-QKV experiment requires exact K=8192 and profile "
                "world_size=4,N=2560 or world_size=8,N=1280"
            )
    else:
        allowed_n = {
            2: frozenset((_N,)),
            4: frozenset((_N,)),
            8: frozenset((_N,)),
        }[world_size]
        if k != _K or int(w.shape[0]) != _K or int(w.shape[1]) not in allowed_n:
            raise ValueError(
                "the Cake backend requires exact K=8192 and an N supported "
                f"by world_size={world_size}: {sorted(allowed_n)}"
            )
    if str(symm_mem.get_backend(inp.device)).upper() != "NVSHMEM":
        raise ValueError(
            "the Cake backend requires the NVSHMEM symmetric-memory backend"
        )
    arch = _target_arch(inp.device)
    if packed_qkv_experiment and (arch != "sm_103a" or inp.dtype != torch.bfloat16):
        raise ValueError("the packed-QKV experiment requires SM103 and bfloat16")
    return device_index, rank, world_size, _group_name(group)


def _ensure_launch_state(
    state: _LaunchState,
    *,
    device_index: int,
    rank: int,
    world_size: int,
    group_name: str,
    main_stream: torch.cuda.Stream,
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
    state.initialization_event = torch.cuda.Event(enable_timing=False)
    state.initialization_event.record(main_stream)
    state.initialization_stream = int(main_stream.cuda_stream)


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
    comm_stream = torch.cuda.Stream(device=device_index)
    bridge_event = torch.cuda.Event(enable_timing=False)
    bridge_event.record(torch.cuda.current_stream(device_index))
    workspace = _Workspace(
        scratch=scratch,
        scratch_handle=scratch_handle,
        comm_stream=comm_stream,
        bridge_event=bridge_event,
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


def _prepared_descriptor_storage(
    workspace: _Workspace,
    module: Any,
    inp: torch.Tensor,
    w: torch.Tensor,
    *,
    device_index: int,
    main_stream: torch.cuda.Stream,
    world_size: int,
    rows: int,
    scratch_fingerprint: tuple[Any, ...],
    weight_fingerprint: tuple[Any, ...],
) -> _PreparedDescriptorEntry:
    """Resolve only the current input descriptor for a prepared launcher."""

    fingerprint = (
        _tensor_fingerprint(inp),
        scratch_fingerprint,
        weight_fingerprint,
    )
    with _CACHE_LOCK:
        entry = workspace.prepared_descriptor_cache.get(fingerprint)
        if entry is not None:
            workspace.prepared_descriptor_cache.move_to_end(fingerprint)
            return entry
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
    ready_event = torch.cuda.Event(enable_timing=False)
    ready_event.record(main_stream)
    entry = _PreparedDescriptorEntry(
        descriptors=descriptors,
        ready_event=ready_event,
        ready_stream=int(main_stream.cuda_stream),
    )
    with _CACHE_LOCK:
        cached = workspace.prepared_descriptor_cache.get(fingerprint)
        if cached is not None:
            workspace.prepared_descriptor_cache.move_to_end(fingerprint)
            return cached
        workspace.prepared_descriptor_cache[fingerprint] = entry
        while len(workspace.prepared_descriptor_cache) > _DESCRIPTOR_CACHE_MAX_ENTRIES:
            workspace.prepared_descriptor_cache.popitem(last=False)
    return entry


def _validate_prepared_view(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise RuntimeError(f"prepared packed-QKV {name} shape does not match binding")
    if tensor.dtype != dtype:
        raise RuntimeError(f"prepared packed-QKV {name} dtype does not match binding")
    if tensor.device != device:
        raise RuntimeError(f"prepared packed-QKV {name} device does not match binding")
    if not tensor.is_contiguous():
        raise RuntimeError(f"prepared packed-QKV {name} must be contiguous")


@dataclass(frozen=True)
class _PreparedPackedQkvSm103Launcher:
    group: dist.ProcessGroup = field(repr=False)
    group_id: int
    group_name: str
    rank: int
    world_size: int
    device_index: int
    device: torch.device
    arch: str
    dtype: torch.dtype
    rows: int
    output_n: int
    module: Any = field(repr=False)
    state: _LaunchState = field(repr=False)
    workspace: _Workspace = field(repr=False)
    weight: torch.Tensor = field(repr=False)
    weight_fingerprint: tuple[Any, ...] = field(repr=False)
    scratch_fingerprint: tuple[Any, ...] = field(repr=False)
    chunk_size: int
    num_chunks: int
    chunk_plan: tuple[tuple[int, int], ...]
    signal_pad: torch.Tensor = field(repr=False)
    signal_pad_ptr: int
    peer_routes: tuple[tuple[torch.Tensor, torch.Tensor], ...] = field(repr=False)
    peer_scratch_ptrs: tuple[int, ...]
    peer_signal_ptrs: tuple[int, ...]
    native_peer_args: tuple[Any, ...] = field(repr=False)
    native_expected_peer_args: tuple[int, ...]
    verbose: bool = False

    def _validate_hot_input(self, inp: torch.Tensor) -> None:
        if id(self.group) != self.group_id:
            raise RuntimeError("prepared packed-QKV group identity changed")
        if inp.device != self.device:
            raise ValueError("prepared packed-QKV inp device changed")
        if inp.dtype != self.dtype:
            raise ValueError("prepared packed-QKV inp dtype changed")
        if inp.ndim != 2 or tuple(inp.shape) != (self.rows, _K):
            raise ValueError(
                f"prepared packed-QKV inp must have shape [{self.rows}, {_K}]"
            )
        if not inp.is_contiguous():
            raise ValueError("prepared packed-QKV inp must be contiguous")
        if _tensor_fingerprint(self.weight) != self.weight_fingerprint:
            raise RuntimeError("prepared packed-QKV bound weight contract changed")

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        self._validate_hot_input(inp)
        state = self.state
        workspace = self.workspace
        w = self.weight
        with state.lock:
            if state.poisoned:
                raise RuntimeError(
                    "Cake all-gather matmul state is poisoned by a prior failed collective"
                )
            try:
                main_stream = torch.cuda.current_stream(self.device_index)
                main_stream_id = int(main_stream.cuda_stream)
                if (
                    state.initialization_event is not None
                    and state.initialization_stream != main_stream_id
                ):
                    main_stream.wait_event(state.initialization_event)
                descriptor_entry = _prepared_descriptor_storage(
                    workspace,
                    self.module,
                    inp,
                    w,
                    device_index=self.device_index,
                    main_stream=main_stream,
                    world_size=self.world_size,
                    rows=self.rows,
                    scratch_fingerprint=self.scratch_fingerprint,
                    weight_fingerprint=self.weight_fingerprint,
                )
                descriptors = descriptor_entry.descriptors
                output = torch.empty(
                    self.world_size * self.rows,
                    self.output_n,
                    dtype=self.dtype,
                    device=self.device_index,
                )

                inp.record_stream(main_stream)
                inp.record_stream(workspace.comm_stream)
                w.record_stream(main_stream)
                descriptors.record_stream(main_stream)
                if descriptor_entry.ready_stream != main_stream_id:
                    main_stream.wait_event(descriptor_entry.ready_event)
                if state.tail_event is not None and state.tail_stream != main_stream_id:
                    main_stream.wait_event(state.tail_event)

                phase = state.next_phase
                if state.ready_epoch >= 2**32 - 1:
                    raise RuntimeError(
                        "Cake all-gather matmul ready epoch exhausted uint32 range"
                    )
                state.ready_epoch += 1
                ready_target = state.ready_epoch
                self.module.run_prepared_packed_qkv(
                    inp,
                    workspace.scratch,
                    w,
                    output,
                    descriptors,
                    self.signal_pad,
                    state.flag_peers,
                    *self.native_peer_args,
                    self.world_size,
                    self.rank,
                    self.rows,
                    phase,
                    ready_target,
                    main_stream_id,
                    int(workspace.comm_stream.cuda_stream),
                    int(workspace.bridge_event.cuda_event),
                    int(self.scratch_fingerprint[0]),
                    self.signal_pad_ptr,
                    *self.native_expected_peer_args,
                )
                if state.tail_event is None:
                    state.tail_event = torch.cuda.Event(enable_timing=False)
                state.tail_event.record(main_stream)
                state.next_phase = 1 - phase
                state.tail_stream = main_stream_id
                if self.verbose and self.rank == 0:
                    print(
                        "Cake all-gather matmul prepared packed-QKV: "
                        f"arch={self.arch}, world_size={self.world_size}, "
                        f"M={self.rows}, K={_K}, N={self.output_n}"
                    )
                return output
            except Exception:
                state.poisoned = True
                raise


def _prepare_all_gather_matmul_cake_packed_qkv_sm103(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
) -> _PreparedPackedQkvSm103Launcher:
    """Bind immutable host state for an exact SM103/BF16 packed-QKV route."""

    device_index, rank, world_size, group_name = _validate_inputs(
        inp, w, group, packed_qkv_experiment=True
    )
    rows = int(inp.shape[0])
    device = torch.device("cuda", device_index)
    arch = _target_arch(device)
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
                main_stream.synchronize()
                if state.tail_event is not None:
                    state.tail_event.synchronize()
            _ensure_launch_state(
                state,
                device_index=device_index,
                rank=rank,
                world_size=world_size,
                group_name=group_name,
                main_stream=main_stream,
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

            expected_scratch_shape = (world_size, rows, _K)
            if tuple(workspace.scratch.shape) != expected_scratch_shape:
                raise RuntimeError(
                    "prepared packed-QKV workspace scratch shape does not match binding"
                )
            if workspace.scratch.dtype != inp.dtype:
                raise RuntimeError(
                    "prepared packed-QKV workspace scratch dtype does not match binding"
                )
            if workspace.scratch.device != device:
                raise RuntimeError(
                    "prepared packed-QKV workspace scratch device does not match binding"
                )
            if (
                int(workspace.scratch_handle.rank) != rank
                or int(workspace.scratch_handle.world_size) != world_size
            ):
                raise RuntimeError(
                    "prepared packed-QKV workspace topology does not match group"
                )

            weight_fingerprint = _tensor_fingerprint(w)
            scratch_fingerprint = _tensor_fingerprint(workspace.scratch)
            chunk_size = min(rows, _CHUNK_ROWS)
            num_chunks = (rows + chunk_size - 1) // chunk_size
            chunk_plan = tuple(
                (
                    chunk_idx * chunk_size,
                    min((chunk_idx + 1) * chunk_size, rows),
                )
                for chunk_idx in range(num_chunks)
            )
            signal_pad = workspace.scratch_handle.get_signal_pad(
                rank, (world_size, num_chunks), torch.uint32, 0
            )
            _validate_prepared_view(
                signal_pad,
                name="local signal_pad",
                shape=(world_size, num_chunks),
                dtype=torch.uint32,
                device=device,
            )
            peer_routes = []
            for shift in range(1, world_size):
                peer = (rank + shift) % world_size
                peer_scratch = workspace.scratch_handle.get_remote_tensor(
                    peer, workspace.scratch.shape, workspace.scratch.dtype
                )[rank]
                _validate_prepared_view(
                    peer_scratch,
                    name=f"peer {peer} scratch rank slice",
                    shape=(rows, _K),
                    dtype=inp.dtype,
                    device=device,
                )
                peer_signal_row = workspace.scratch_handle.get_signal_pad(
                    peer, (world_size, num_chunks), torch.uint32, 0
                )[rank]
                _validate_prepared_view(
                    peer_signal_row,
                    name=f"peer {peer} signal rank row",
                    shape=(num_chunks,),
                    dtype=torch.uint32,
                    device=device,
                )
                peer_routes.append((peer_scratch, peer_signal_row))
            peer_scratch_ptrs = tuple(
                int(peer_scratch.data_ptr()) for peer_scratch, _ in peer_routes
            )
            peer_signal_ptrs = tuple(
                int(peer_signal.data_ptr()) for _, peer_signal in peer_routes
            )
            padding = 7 - len(peer_routes)
            native_peer_routes = tuple(peer_routes) + (peer_routes[-1],) * padding
            native_peer_scratch_ptrs = (
                peer_scratch_ptrs + (peer_scratch_ptrs[-1],) * padding
            )
            native_peer_signal_ptrs = (
                peer_signal_ptrs + (peer_signal_ptrs[-1],) * padding
            )
            native_peer_args = tuple(
                tensor for route in native_peer_routes for tensor in route
            )
            native_expected_peer_args = tuple(
                pointer
                for pair in zip(
                    native_peer_scratch_ptrs,
                    native_peer_signal_ptrs,
                    strict=True,
                )
                for pointer in pair
            )
            _prepared_descriptor_storage(
                workspace,
                module,
                inp,
                w,
                device_index=device_index,
                main_stream=main_stream,
                world_size=world_size,
                rows=rows,
                scratch_fingerprint=scratch_fingerprint,
                weight_fingerprint=weight_fingerprint,
            )
            launcher = _PreparedPackedQkvSm103Launcher(
                group=group,
                group_id=id(group),
                group_name=group_name,
                rank=rank,
                world_size=world_size,
                device_index=device_index,
                device=device,
                arch=arch,
                dtype=inp.dtype,
                rows=rows,
                output_n=int(w.shape[1]),
                module=module,
                state=state,
                workspace=workspace,
                weight=w,
                weight_fingerprint=weight_fingerprint,
                scratch_fingerprint=scratch_fingerprint,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                chunk_plan=chunk_plan,
                signal_pad=signal_pad,
                signal_pad_ptr=int(signal_pad.data_ptr()),
                peer_routes=tuple(peer_routes),
                peer_scratch_ptrs=peer_scratch_ptrs,
                peer_signal_ptrs=peer_signal_ptrs,
                native_peer_args=native_peer_args,
                native_expected_peer_args=native_expected_peer_args,
                verbose=verbose,
            )
            return launcher
        except Exception:
            state.poisoned = True
            raise


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
        output_n = int(w.shape[1])
    else:
        device_index, rank, world_size, group_name = _validate_inputs(inp, w, group)
        output_n = int(w.shape[1])
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
                main_stream=main_stream,
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

            main_stream_id = int(main_stream.cuda_stream)
            if (
                state.initialization_event is not None
                and state.initialization_stream != main_stream_id
            ):
                main_stream.wait_event(state.initialization_event)

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
            if state.tail_event is not None and state.tail_stream != main_stream_id:
                main_stream.wait_event(state.tail_event)

            phase = state.next_phase
            if state.ready_epoch >= 2**32 - 1:
                raise RuntimeError(
                    "Cake all-gather matmul ready epoch exhausted uint32 range"
                )
            state.ready_epoch += 1
            ready_target = state.ready_epoch
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
                            peer_signal[rank], chunk_idx, ready_target
                        )

            module.run_main(
                inp,
                workspace.scratch,
                w,
                output,
                descriptors,
                signal_pad,
                ready_target,
                world_size,
                rank,
                rows,
                int(inp.dtype == torch.float16),
                main_stream_id,
            )
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
