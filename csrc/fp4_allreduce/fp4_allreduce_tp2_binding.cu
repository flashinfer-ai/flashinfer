/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host launcher of the TP2 NVFP4-transport all-reduce (oneshot push over Lamport receive windows).
// One JIT module per architecture links the generated kernels of every (input format, phase-B variant, rank)
// specialised for 16384 rows; this file selects the caller's program, validates the call against the kernel
// contract and launches it, optionally as a programmatic dependent launch.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <set>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

extern "C" __global__ void kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r0_cap16384(__nv_bfloat16* x, __nv_bfloat16* out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r1_cap16384(__nv_bfloat16* x, __nv_bfloat16* out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r0_cap16384(uint8_t* xq, float* gs, __nv_bfloat16* out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r1_cap16384(uint8_t* xq, float* gs, __nv_bfloat16* out, unsigned int* ctrl, int rows, const unsigned long long* workspace);

namespace {

constexpr int kWorldSize = 2;
constexpr int kVariants = 1;
constexpr int64_t kHidden = 6144;
constexpr int64_t kRecordBytes = 3456;  // NVFP4 input record: 3072 E2M1 value bytes + 384 E4M3 scale bytes
constexpr int64_t kCapacity = 16384;  // rows the delivered programs are specialised for
constexpr int64_t kMaxGrid = 4096;
constexpr int64_t kTableEntries = 4;
constexpr int64_t kCtrlWords = 16384;
constexpr int64_t kRowMultiple = 1;
constexpr int64_t kTeamMax = 0;
constexpr int64_t kByQuadMin = 0;
constexpr unsigned kThreads = 352;
constexpr unsigned kDynamicSmemBytes = 226176;

// [input format: 0 = bfloat16, 1 = nvfp4 records][phase-B variant][rank]
const void* const kKernels[2][kVariants][kWorldSize] = {
    {{reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_oneshot_bf16_fused_r1_cap16384)}},
    {{reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_oneshot_nvfp4_fused_r1_cap16384)}}
};

// Phase-B variant of one call from the rows each CTA owns; every rank evaluates it on the same grid.
int phase_b_variant(int64_t rows, int64_t grid) {
  (void)rows;
  (void)grid;
  return 0;
}

// Opt a kernel into its dynamic shared memory size once per process.
void enable_dynamic_smem(const void* kernel) {
  static std::mutex mutex;
  static std::set<const void*> enabled;
  std::lock_guard<std::mutex> guard(mutex);
  if (enabled.count(kernel)) return;
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kDynamicSmemBytes));
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: " << cudaGetErrorString(status);
  enabled.insert(kernel);
}

}  // namespace

// Launch this rank's program.  ``input`` is [rows, 6144] bfloat16 (input_format 0) or [rows, 3456] uint8
// NVFP4 records (input_format 1; ``scales`` then holds the static per-rank scale of every rank, float32
// [world size]).  ``out`` is [rows, 6144] bfloat16, ``ctrl`` the rank's int32 call-state words, ``workspace``
// the int64 peer window table in kernel slot order and ``grid`` the CTA count every rank agreed on.
void fp4_allreduce(TensorView input, Optional<TensorView> scales, TensorView out, TensorView ctrl,
                   TensorView workspace, int64_t rank, int64_t input_format, int64_t grid,
                   bool launch_with_pdl) {
  TVM_FFI_CHECK(rank >= 0 && rank < kWorldSize, ValueError)
      << "rank must be in [0, " << kWorldSize << "), got " << rank;
  TVM_FFI_CHECK(input_format == 0 || input_format == 1, ValueError)
      << "input_format must be 0 (bfloat16) or 1 (nvfp4 records), got " << input_format;
  TVM_FFI_CHECK(grid > 0 && grid <= kMaxGrid, ValueError)
      << "grid must be in (0, " << kMaxGrid << "], got " << grid;
  CHECK_INPUT(input);
  CHECK_DIM(2, input);
  const int64_t rows = input.size(0);
  TVM_FFI_CHECK(rows > 0 && rows <= kCapacity, ValueError)
      << "rows must be in (0, " << kCapacity << "], got " << rows;
  TVM_FFI_CHECK(rows % kRowMultiple == 0, ValueError)
      << "rows must be a multiple of " << kRowMultiple << ", got " << rows;
  if (input_format == 0) {
    CHECK_INPUT_TYPE(input, dl_bfloat16);
    TVM_FFI_CHECK(input.size(1) == kHidden, ValueError)
        << "input must be [rows, " << kHidden << "] bfloat16, got " << rows << " x " << input.size(1);
    TVM_FFI_CHECK(!scales.has_value(), ValueError) << "scales belong to NVFP4 input records only";
  } else {
    CHECK_INPUT_TYPE(input, dl_uint8);
    TVM_FFI_CHECK(input.size(1) == kRecordBytes, ValueError)
        << "NVFP4 input records must be [rows, " << kRecordBytes << "] uint8, got " << rows << " x "
        << input.size(1);
    TVM_FFI_CHECK(scales.has_value(), ValueError) << "NVFP4 input records require the per-rank scales";
    CHECK_INPUT_AND_TYPE(scales.value(), dl_float32);
    CHECK_DEVICE(scales.value(), input);
    TVM_FFI_CHECK(scales.value().numel() >= kWorldSize, ValueError)
        << "scales must hold one float32 per rank (" << kWorldSize << ")";
  }
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_DEVICE(out, input);
  TVM_FFI_CHECK(out.ndim() == 2 && out.size(0) == rows && out.size(1) == kHidden, ValueError)
      << "out must be [" << rows << ", " << kHidden << "] bfloat16";
  CHECK_INPUT_AND_TYPE(ctrl, dl_int32);
  CHECK_DEVICE(ctrl, input);
  TVM_FFI_CHECK(ctrl.numel() == kCtrlWords, ValueError)
      << "ctrl must hold exactly " << kCtrlWords << " int32 words, got " << ctrl.numel();
  CHECK_INPUT_AND_TYPE(workspace, dl_int64);
  CHECK_DEVICE(workspace, input);
  TVM_FFI_CHECK(workspace.ndim() == 1 && workspace.size(0) == kTableEntries, ValueError)
      << "workspace must hold exactly " << kTableEntries << " peer window pointers";

  const DLDevice device = input.device();
  ffi::CUDADeviceGuard device_guard(device.device_id);
  const int variant = phase_b_variant(rows, grid);
  const void* kernel = kKernels[input_format][variant][rank];
  enable_dynamic_smem(kernel);

  void* p_input = input.data_ptr();
  void* p_scales = scales.has_value() ? scales.value().data_ptr() : nullptr;
  void* p_out = out.data_ptr();
  void* p_ctrl = ctrl.data_ptr();
  void* p_workspace = workspace.data_ptr();
  int32_t v_rows = static_cast<int32_t>(rows);
  void* args_bf16[] = {&p_input, &p_out, &p_ctrl, &v_rows, &p_workspace};
  void* args_nvfp4[] = {&p_input, &p_scales, &p_out, &p_ctrl, &v_rows, &p_workspace};
  void** args = input_format == 0 ? args_bf16 : args_nvfp4;

  cudaLaunchAttribute attrs[1]{};
  unsigned num_attrs = 0;
  if (launch_with_pdl) {
    attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = get_stream(device);
  config.attrs = attrs;
  config.numAttrs = num_attrs;
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "fp4_allreduce launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_allreduce, fp4_allreduce);
