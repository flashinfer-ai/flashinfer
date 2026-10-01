/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host launcher for the Cake MoE finalize + all-reduce fusion kernels in
// sm_100a/*_device.cu. One module per architecture links the twelve generated
// kernels (dtype x world size x optional NVFP4 epilogue); the launch
// configuration is shared and PDL is a launch attribute chosen at run time.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#define CAKE_MOE_FINALIZE_KERNEL(T, NAME)                                                   \
  extern "C" __global__ void NAME(                                                          \
      T* __restrict__ allreduce_in, int* __restrict__ inverse_indices,                      \
      T* __restrict__ expert_scales, T* __restrict__ shared_expert_output,                  \
      T* __restrict__ residual, T* __restrict__ norm_weight, T* __restrict__ residual_out,  \
      T* __restrict__ norm_out, T* __restrict__ quant_out, T* __restrict__ scale_out,       \
      long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k,      \
      int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, \
      float scale_factor);

CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws2_o110)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws2_o111)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o110)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o111)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o110)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o111)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111)

#undef CAKE_MOE_FINALIZE_KERNEL

namespace {

constexpr int64_t kHiddenDim = 7168;
constexpr unsigned kThreads = 224;
constexpr unsigned kClusterSize = 4;
constexpr unsigned kDynamicSmemBytes = 256;
constexpr int kMaxDevices = 64;

// [dtype][world size index][quant epilogue]
const void* const kKernels[2][3][2] = {
    {{reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws2_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws2_o111)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o111)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o111)}},
    {{reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111)}},
};

int sm_count(int device_id) {
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxDevices, ValueError)
      << "unsupported CUDA device index " << device_id;
  static int cache[kMaxDevices] = {};
  if (cache[device_id] == 0) {
    int count = 0;
    cudaError_t status = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id);
    TVM_FFI_CHECK(status == cudaSuccess && count > 0, RuntimeError)
        << "cudaDeviceGetAttribute(multiProcessorCount) failed for cuda:" << device_id << ": "
        << cudaGetErrorString(status);
    cache[device_id] = count;
  }
  return cache[device_id];
}

void check_activation(const TensorView& t, const TensorView& ref, const char* name, int64_t rows) {
  CHECK_INPUT(t);
  CHECK_DEVICE(t, ref);
  CHECK_SAME_DTYPE(t, ref);
  TVM_FFI_CHECK(t.ndim() == 2 && t.size(0) == rows && t.size(1) == kHiddenDim, ValueError)
      << name << " must have shape [" << rows << ", " << kHiddenDim << "]";
}

int64_t byte_size(const TensorView& t) { return t.numel() * get_element_size(t); }

}  // namespace

// expanded_idx_to_permuted_idx is [tokens, top_k]; allreduce_in is the permuted expert
// output [num_permuted_rows, 7168]. quant_out / scale_out carry packed FP4 data and E4M3
// scales (SWIZZLED_128x4); they are checked by byte size and accept any element type.
void cake_moe_finalize_allreduce_fusion(
    TensorView allreduce_in, TensorView residual_in, TensorView norm_weight,
    TensorView expanded_idx_to_permuted_idx, TensorView expert_scale_factor,
    Optional<TensorView> shared_expert_output, TensorView residual_out, TensorView norm_out,
    Optional<TensorView> quant_out, Optional<TensorView> scale_out, TensorView workspace_ptrs,
    int64_t world_rank, int64_t world_size, double eps, double routed_scaling_factor,
    double weight_bias, bool launch_with_pdl) {
  TVM_FFI_CHECK(world_size == 2 || world_size == 4 || world_size == 8, ValueError)
      << "Cake MoE finalize supports world_size 2, 4, or 8, got " << world_size;
  TVM_FFI_CHECK(world_rank >= 0 && world_rank < world_size, ValueError)
      << "world_rank must be in [0, " << world_size << "), got " << world_rank;
  TVM_FFI_CHECK(eps > 0.0, ValueError) << "eps must be positive, got " << eps;
  TVM_FFI_CHECK(weight_bias == 0.0 || weight_bias == 1.0, ValueError)
      << "Cake MoE finalize weight_bias must be 0.0 or 1.0, got " << weight_bias;

  CHECK_INPUT(allreduce_in);
  TVM_FFI_CHECK(
      allreduce_in.ndim() == 2 && allreduce_in.size(1) == kHiddenDim && allreduce_in.size(0) > 0,
      ValueError)
      << "allreduce_in must have shape [num_permuted_rows, " << kHiddenDim << "]";
  int dtype_index = 0;
  switch (encode_dlpack_dtype(allreduce_in.dtype())) {
    case float16_code:
      dtype_index = 0;
      break;
    case bfloat16_code:
      dtype_index = 1;
      break;
    default:
      TVM_FFI_LOG_AND_THROW(ValueError) << "Cake MoE finalize supports float16 and bfloat16 inputs";
  }

  CHECK_INPUT(residual_in);
  CHECK_DEVICE(residual_in, allreduce_in);
  CHECK_SAME_DTYPE(residual_in, allreduce_in);
  TVM_FFI_CHECK(
      residual_in.ndim() == 2 && residual_in.size(1) == kHiddenDim && residual_in.size(0) > 0,
      ValueError)
      << "residual_in must have shape [token_num, " << kHiddenDim << "] with token_num > 0";
  const int64_t tokens = residual_in.size(0);

  CHECK_INPUT(norm_weight);
  CHECK_DEVICE(norm_weight, allreduce_in);
  CHECK_SAME_DTYPE(norm_weight, allreduce_in);
  TVM_FFI_CHECK(norm_weight.ndim() == 1 && norm_weight.size(0) == kHiddenDim, ValueError)
      << "norm_weight must have shape [" << kHiddenDim << "]";

  CHECK_INPUT_AND_TYPE(expanded_idx_to_permuted_idx, dl_int32);
  CHECK_DEVICE(expanded_idx_to_permuted_idx, allreduce_in);
  TVM_FFI_CHECK(
      expanded_idx_to_permuted_idx.ndim() == 2 && expanded_idx_to_permuted_idx.size(0) == tokens &&
          (expanded_idx_to_permuted_idx.size(1) == 4 || expanded_idx_to_permuted_idx.size(1) == 8),
      ValueError)
      << "expanded_idx_to_permuted_idx must have shape [token_num, top_k] with top_k 4 or 8";
  const int64_t top_k = expanded_idx_to_permuted_idx.size(1);

  CHECK_INPUT(expert_scale_factor);
  CHECK_DEVICE(expert_scale_factor, allreduce_in);
  CHECK_SAME_DTYPE(expert_scale_factor, allreduce_in);
  TVM_FFI_CHECK(expert_scale_factor.ndim() == 2 && expert_scale_factor.size(0) == tokens &&
                    expert_scale_factor.size(1) == top_k,
                ValueError)
      << "expert_scale_factor must have shape [token_num, top_k]";

  if (shared_expert_output.has_value()) {
    check_activation(shared_expert_output.value(), allreduce_in, "shared_expert_output", tokens);
  }
  check_activation(residual_out, allreduce_in, "residual_out", tokens);
  check_activation(norm_out, allreduce_in, "norm_out", tokens);

  TVM_FFI_CHECK(quant_out.has_value() == scale_out.has_value(), ValueError)
      << "quant_out and scale_out must be provided together";
  const bool quantize = quant_out.has_value();
  if (quantize) {
    CHECK_INPUT(quant_out.value());
    CHECK_DEVICE(quant_out.value(), allreduce_in);
    CHECK_INPUT(scale_out.value());
    CHECK_DEVICE(scale_out.value(), allreduce_in);
    const int64_t quant_bytes = tokens * kHiddenDim / 2;
    TVM_FFI_CHECK(byte_size(quant_out.value()) >= quant_bytes, ValueError)
        << "quant_out needs at least " << quant_bytes << " bytes for " << tokens << " tokens";
    const int64_t padded_rows = (tokens + 127) / 128 * 128;
    const int64_t padded_columns = (kHiddenDim / 16 + 3) / 4 * 4;
    const int64_t scale_bytes = padded_rows * padded_columns;
    TVM_FFI_CHECK(byte_size(scale_out.value()) >= scale_bytes, ValueError)
        << "scale_out needs at least " << scale_bytes
        << " bytes for the padded SWIZZLED_128x4 layout";
  }

  CHECK_INPUT_AND_TYPE(workspace_ptrs, dl_int64);
  CHECK_DEVICE(workspace_ptrs, allreduce_in);
  TVM_FFI_CHECK(workspace_ptrs.ndim() == 1 && workspace_ptrs.size(0) == 3 * world_size + 1,
                ValueError)
      << "workspace_ptrs must contain exactly " << 3 * world_size + 1 << " pointers";

  const DLDevice device = allreduce_in.device();
  ffi::CUDADeviceGuard device_guard(device.device_id);
  const int64_t clusters =
      std::min<int64_t>(sm_count(device.device_id), tokens * kClusterSize) / kClusterSize;
  TVM_FFI_CHECK(clusters > 0, RuntimeError)
      << "Cake MoE finalize requires one complete four-CTA cluster";

  void* p_allreduce_in = allreduce_in.data_ptr();
  void* p_inverse_indices = expanded_idx_to_permuted_idx.data_ptr();
  void* p_expert_scales = expert_scale_factor.data_ptr();
  void* p_shared_expert_output =
      shared_expert_output.has_value() ? shared_expert_output.value().data_ptr() : nullptr;
  void* p_residual = residual_in.data_ptr();
  void* p_norm_weight = norm_weight.data_ptr();
  void* p_residual_out = residual_out.data_ptr();
  void* p_norm_out = norm_out.data_ptr();
  void* p_quant_out = quantize ? quant_out.value().data_ptr() : nullptr;
  void* p_scale_out = quantize ? scale_out.value().data_ptr() : nullptr;
  void* p_workspace = workspace_ptrs.data_ptr();
  int32_t v_world_rank = static_cast<int32_t>(world_rank);
  int32_t v_tokens = static_cast<int32_t>(tokens);
  int32_t v_top_k = static_cast<int32_t>(top_k);
  int32_t v_has_shared_expert = shared_expert_output.has_value() ? 1 : 0;
  float v_routed_scaling_factor = static_cast<float>(routed_scaling_factor);
  float v_epsilon = static_cast<float>(eps);
  float v_weight_bias = static_cast<float>(weight_bias);
  float v_scale_factor = 1.0f;
  void* args[] = {
      &p_allreduce_in, &p_inverse_indices, &p_expert_scales,     &p_shared_expert_output,
      &p_residual,     &p_norm_weight,     &p_residual_out,      &p_norm_out,
      &p_quant_out,    &p_scale_out,       &p_workspace,         &v_world_rank,
      &v_tokens,       &v_top_k,           &v_has_shared_expert, &v_routed_scaling_factor,
      &v_epsilon,      &v_weight_bias,     &v_scale_factor};

  cudaLaunchAttribute attrs[2]{};
  unsigned num_attrs = 0;
  attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
  attrs[num_attrs].val.clusterDim.x = kClusterSize;
  attrs[num_attrs].val.clusterDim.y = 1;
  attrs[num_attrs].val.clusterDim.z = 1;
  ++num_attrs;
  if (launch_with_pdl) {
    attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(clusters * kClusterSize), 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = get_stream(device);
  config.attrs = attrs;
  config.numAttrs = num_attrs;
  const int ws_index = world_size == 2 ? 0 : (world_size == 4 ? 1 : 2);
  cudaError_t status =
      cudaLaunchKernelExC(&config, kKernels[dtype_index][ws_index][quantize ? 1 : 0], args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "Cake MoE finalize launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_moe_finalize_allreduce_fusion,
                              cake_moe_finalize_allreduce_fusion);
