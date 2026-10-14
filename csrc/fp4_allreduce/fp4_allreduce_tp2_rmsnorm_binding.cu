/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host launcher of the TP2 NVFP4-transport all-reduce fused with the residual add and the Gemma RMSNorm
// (one-shot push over Lamport receive windows, FP32 normalisation input, BF16 / NVFP4 / MXFP8 output).  One JIT
// module per architecture links the generated kernels of every (input format, output format, rank) specialised for
// 16384 rows, eps 1e-6 and the Gemma weight bias 1; this file selects the caller's program, validates the call
// against the kernel contract and launches it, optionally as a programmatic dependent launch.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <set>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r0_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, __nv_bfloat16* out, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r1_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, __nv_bfloat16* out, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r0_cap16384(uint8_t* xq, float* gs, __nv_bfloat16* residual_in, __nv_bfloat16* weight, __nv_bfloat16* out, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r1_cap16384(uint8_t* xq, float* gs, __nv_bfloat16* residual_in, __nv_bfloat16* weight, __nv_bfloat16* out, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r0_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, float* gs_out, uint8_t* out_q, uint8_t* out_sf, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r1_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, float* gs_out, uint8_t* out_q, uint8_t* out_sf, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r0_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, uint8_t* out_q, uint8_t* out_sf, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);
extern "C" __global__ void kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r1_cap16384(__nv_bfloat16* x, __nv_bfloat16* residual_in, __nv_bfloat16* weight, uint8_t* out_q, uint8_t* out_sf, __nv_bfloat16* residual_out, unsigned int* ctrl, int rows, const unsigned long long* workspace);

namespace {

constexpr int kWorldSize = 2;
constexpr int kInputFormats = 2;   // 0 = bfloat16 activations, 1 = NVFP4 records
constexpr int kOutputFormats = 3;  // 0 = bfloat16 y, 1 = NVFP4 values + E4M3 scales, 2 = MXFP8 values + UE8M0 scales
constexpr int64_t kHidden = 6144;
constexpr int64_t kRecordBytes = 3456;  // NVFP4 input record: 3072 E2M1 value bytes + 384 E4M3 scale bytes
constexpr int64_t kCapacity = 16384;  // rows the delivered programs are specialised for
constexpr int64_t kMaxGrid = 65536;
constexpr int64_t kTableEntries = 2;
constexpr int64_t kCtrlWords = 4;
constexpr int64_t kRowMultiple = 4;
constexpr int64_t kScaleRowTile = 128;  // the swizzled scale tensors cover rows padded to a multiple of 128
constexpr int64_t kNvfp4ValueBytes = 3072;
constexpr int64_t kNvfp4ScalesPerRow = 384;
constexpr int64_t kMxfp8ValueBytes = 6144;
constexpr int64_t kMxfp8ScalesPerRow = 192;
// Threads and dynamic shared memory depend on the input format (six worker warps for bfloat16, five for records).
constexpr unsigned kThreads[kInputFormats] = {224, 192};
constexpr unsigned kDynamicSmemBytes[kInputFormats] = {214784, 198656};

// [input format][output format][rank]; nullptr marks a pair the family does not deliver (quantised output is
// specified for bfloat16 input only).
const void* const kKernels[kInputFormats][kOutputFormats][kWorldSize] = {
    {{reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_bf16_fused_r1_cap16384)},
     {reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_nvfp4_fused_r1_cap16384)},
     {reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_bf16_mxfp8_fused_r1_cap16384)}},
    {{reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r0_cap16384), reinterpret_cast<const void*>(kernel_fp4_allreduce_tp2_lnorm_nvfp4_bf16_fused_r1_cap16384)},
     {nullptr, nullptr},
     {nullptr, nullptr}}
};

// Opt a kernel into its dynamic shared memory size once per process.
void enable_dynamic_smem(const void* kernel, unsigned bytes) {
  static std::mutex mutex;
  static std::set<const void*> enabled;
  std::lock_guard<std::mutex> guard(mutex);
  if (enabled.count(kernel)) return;
  cudaError_t status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: " << cudaGetErrorString(status);
  enabled.insert(kernel);
}

int64_t padded_scale_rows(int64_t rows) { return (rows + kScaleRowTile - 1) / kScaleRowTile * kScaleRowTile; }

}  // namespace

// Launch this rank's fused program.  ``input`` is [rows, 6144] bfloat16 (input_format 0) or [rows, 3456] uint8
// NVFP4 records (input_format 1; ``scales`` then holds the static per-rank scale of every rank, float32 [world
// size]).  ``residual_in`` and ``residual_out`` are [rows, 6144] bfloat16 (distinct buffers; residual_out receives
// bf16(sum + residual)), ``weight`` the bfloat16 [6144] norm weight (y = z * rsqrt(mean(z^2) + eps) * (1 + weight)).
// output_format 0 writes ``out`` ([rows, 6144] bfloat16); 1 writes ``values`` ([rows, 3072] uint8 E2M1 pairs) and
// ``scales_out`` (uint8, at least 128-row-padded rows x 384 swizzled E4M3 scales) with ``global_scale`` (float32 [1]
// on the device, the quantiser's input global scale); 2 writes ``values`` ([rows, 6144] E4M3 bytes) and
// ``scales_out`` (uint8, at least padded rows x 192 swizzled UE8M0 scales).  ``ctrl`` is the rank's int32 call-state
// words, ``workspace`` the int64 peer window table in rank order and ``grid`` the CTA count every rank agreed on.
void fp4_allreduce_rmsnorm(TensorView input, Optional<TensorView> scales, TensorView residual_in,
                           TensorView weight, Optional<TensorView> out, Optional<TensorView> values,
                           Optional<TensorView> scales_out, Optional<TensorView> global_scale,
                           TensorView residual_out, TensorView ctrl, TensorView workspace, int64_t rank,
                           int64_t input_format, int64_t output_format, int64_t grid, bool launch_with_pdl) {
  TVM_FFI_CHECK(rank >= 0 && rank < kWorldSize, ValueError)
      << "rank must be in [0, " << kWorldSize << "), got " << rank;
  TVM_FFI_CHECK(input_format == 0 || input_format == 1, ValueError)
      << "input_format must be 0 (bfloat16) or 1 (nvfp4 records), got " << input_format;
  TVM_FFI_CHECK(output_format >= 0 && output_format < kOutputFormats, ValueError)
      << "output_format must be 0 (bfloat16), 1 (nvfp4) or 2 (mxfp8), got " << output_format;
  TVM_FFI_CHECK(grid > 0 && grid <= kMaxGrid, ValueError)
      << "grid must be in (0, " << kMaxGrid << "], got " << grid;
  const void* kernel = kKernels[input_format][output_format][rank];
  TVM_FFI_CHECK(kernel != nullptr, ValueError)
      << "the fused-norm programs deliver quantised (nvfp4 / mxfp8) output for bfloat16 input only";
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
  CHECK_INPUT_AND_TYPE(residual_in, dl_bfloat16);
  CHECK_DEVICE(residual_in, input);
  TVM_FFI_CHECK(residual_in.ndim() == 2 && residual_in.size(0) == rows && residual_in.size(1) == kHidden, ValueError)
      << "residual_in must be [" << rows << ", " << kHidden << "] bfloat16";
  CHECK_INPUT_AND_TYPE(residual_out, dl_bfloat16);
  CHECK_DEVICE(residual_out, input);
  TVM_FFI_CHECK(residual_out.ndim() == 2 && residual_out.size(0) == rows && residual_out.size(1) == kHidden,
                ValueError)
      << "residual_out must be [" << rows << ", " << kHidden << "] bfloat16";
  TVM_FFI_CHECK(residual_out.data_ptr() != residual_in.data_ptr(), ValueError)
      << "residual_out must not alias residual_in";
  CHECK_INPUT_AND_TYPE(weight, dl_bfloat16);
  CHECK_DEVICE(weight, input);
  TVM_FFI_CHECK(weight.numel() == kHidden, ValueError) << "weight must hold " << kHidden << " bfloat16 values";
  void* p_out = nullptr;
  void* p_values = nullptr;
  void* p_scales_out = nullptr;
  void* p_global_scale = nullptr;
  if (output_format == 0) {
    TVM_FFI_CHECK(out.has_value(), ValueError) << "bfloat16 output needs the out tensor";
    TVM_FFI_CHECK(!values.has_value() && !scales_out.has_value() && !global_scale.has_value(), ValueError)
        << "values / scales_out / global_scale belong to the quantised outputs";
    CHECK_INPUT_AND_TYPE(out.value(), dl_bfloat16);
    CHECK_DEVICE(out.value(), input);
    TVM_FFI_CHECK(out.value().ndim() == 2 && out.value().size(0) == rows && out.value().size(1) == kHidden,
                  ValueError)
        << "out must be [" << rows << ", " << kHidden << "] bfloat16";
    p_out = out.value().data_ptr();
  } else {
    TVM_FFI_CHECK(!out.has_value(), ValueError) << "out belongs to the bfloat16 output";
    TVM_FFI_CHECK(values.has_value() && scales_out.has_value(), ValueError)
        << "quantised output needs the values and scales_out tensors";
    const int64_t value_bytes = output_format == 1 ? kNvfp4ValueBytes : kMxfp8ValueBytes;
    const int64_t scales_per_row = output_format == 1 ? kNvfp4ScalesPerRow : kMxfp8ScalesPerRow;
    CHECK_INPUT(values.value());
    CHECK_DEVICE(values.value(), input);
    TVM_FFI_CHECK(values.value().dtype() == dl_uint8 || values.value().dtype() == dl_float8_e4m3fn, ValueError)
        << "values must be uint8 (or float8_e4m3fn for the mxfp8 output)";
    TVM_FFI_CHECK(values.value().ndim() == 2 && values.value().size(0) == rows &&
                      values.value().size(1) * static_cast<int64_t>(get_element_size(values.value())) == value_bytes,
                  ValueError)
        << "values must be [" << rows << ", " << value_bytes << "] bytes for output_format " << output_format;
    CHECK_INPUT_AND_TYPE(scales_out.value(), dl_uint8);
    CHECK_DEVICE(scales_out.value(), input);
    TVM_FFI_CHECK(scales_out.value().numel() >= padded_scale_rows(rows) * scales_per_row, ValueError)
        << "scales_out must hold at least " << padded_scale_rows(rows) * scales_per_row
        << " bytes (rows padded to 128 x " << scales_per_row << ")";
    p_values = values.value().data_ptr();
    p_scales_out = scales_out.value().data_ptr();
    if (output_format == 1) {
      TVM_FFI_CHECK(global_scale.has_value(), ValueError) << "nvfp4 output needs the float32 global_scale tensor";
      CHECK_INPUT_AND_TYPE(global_scale.value(), dl_float32);
      CHECK_DEVICE(global_scale.value(), input);
      TVM_FFI_CHECK(global_scale.value().numel() >= 1, ValueError) << "global_scale must hold one float32";
      p_global_scale = global_scale.value().data_ptr();
    } else {
      TVM_FFI_CHECK(!global_scale.has_value(), ValueError) << "global_scale belongs to the nvfp4 output";
    }
  }
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
  enable_dynamic_smem(kernel, kDynamicSmemBytes[input_format]);

  void* p_input = input.data_ptr();
  void* p_scales = scales.has_value() ? scales.value().data_ptr() : nullptr;
  void* p_residual = residual_in.data_ptr();
  void* p_weight = weight.data_ptr();
  void* p_residual_out = residual_out.data_ptr();
  void* p_ctrl = ctrl.data_ptr();
  void* p_workspace = workspace.data_ptr();
  int32_t v_rows = static_cast<int32_t>(rows);
  void* args_bf16_bf16[] = {&p_input, &p_residual, &p_weight, &p_out, &p_residual_out, &p_ctrl, &v_rows, &p_workspace};
  void* args_nvfp4_bf16[] = {&p_input, &p_scales, &p_residual, &p_weight, &p_out, &p_residual_out, &p_ctrl, &v_rows, &p_workspace};
  void* args_bf16_nvfp4[] = {&p_input, &p_residual, &p_weight, &p_global_scale, &p_values, &p_scales_out, &p_residual_out, &p_ctrl, &v_rows, &p_workspace};
  void* args_bf16_mxfp8[] = {&p_input, &p_residual, &p_weight, &p_values, &p_scales_out, &p_residual_out, &p_ctrl, &v_rows, &p_workspace};
  void** args = input_format == 1 ? args_nvfp4_bf16
                : output_format == 0 ? args_bf16_bf16
                : output_format == 1 ? args_bf16_nvfp4
                                     : args_bf16_mxfp8;

  cudaLaunchAttribute attrs[1]{};
  unsigned num_attrs = 0;
  if (launch_with_pdl) {
    attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  config.blockDim = dim3(kThreads[input_format], 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes[input_format];
  config.stream = get_stream(device);
  config.attrs = attrs;
  config.numAttrs = num_attrs;
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "fp4_allreduce_rmsnorm launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_allreduce_rmsnorm, fp4_allreduce_rmsnorm);
