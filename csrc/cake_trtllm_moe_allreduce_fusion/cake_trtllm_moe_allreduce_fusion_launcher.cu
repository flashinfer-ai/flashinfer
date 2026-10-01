/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host launcher for the Cake TRT-LLM MoE all-reduce fusion kernels
// (cake_trtllm_moe_allreduce_fusion_kernels.cu). The bundle serves
// ``trtllm_moe_allreduce_fusion(backend="cake")`` when no all-reduce output is
// requested; calls with ``moe_allreduce_out`` go to the union export.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>

#include "tvm_ffi_utils.h"

#ifndef CAKE_MOE_AR_SM103_T1
#error "CAKE_MOE_AR_SM103_T1 must be defined (1 for sm_103a, 0 otherwise)"
#endif
#ifndef CAKE_MOE_AR_SM100_WS8_MID
#error "CAKE_MOE_AR_SM100_WS8_MID must be defined (1 for sm_100a, 0 otherwise)"
#endif

// Every kernel of the bundle has the same 18-parameter signature up to the
// 16-bit element type.
#define CAKE_MOE_AR_DECLARE_KERNEL(Symbol, T)                                                    \
  extern "C" __global__ void Symbol(                                                             \
      T* __restrict__ active_expert_tokens, float* __restrict__ expert_scales,                   \
      T* __restrict__ token_input, T* __restrict__ residual, T* __restrict__ gamma,              \
      T* __restrict__ moe_allreduce_out, T* __restrict__ residual_out, T* __restrict__ norm_out, \
      T* __restrict__ quant_out, T* __restrict__ scale_out,                                      \
      long long* __restrict__ workspace_tensor, int world_rank, int tokens, int active_experts,  \
      float epsilon, float weight_bias, float scale_factor, int layout_code);

CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws2_o0110, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws4_o0110, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110, __nv_bfloat16)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110, __nv_bfloat16)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110, __nv_bfloat16)
#if CAKE_MOE_AR_SM103_T1
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws2_o0110_sm103_t1, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws4_o0110_sm103_t1, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm103_t1, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110_sm103_t1,
                           __nv_bfloat16)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110_sm103_t1,
                           __nv_bfloat16)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm103_t1,
                           __nv_bfloat16)
#endif
#if CAKE_MOE_AR_SM100_WS8_MID
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm100_ws8_mid, __half)
CAKE_MOE_AR_DECLARE_KERNEL(kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm100_ws8_mid,
                           __nv_bfloat16)
#endif

#undef CAKE_MOE_AR_DECLARE_KERNEL

namespace cake_trtllm_moe_allreduce_fusion {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int64_t kHiddenDim = 7168;
constexpr uint32_t kBlockThreads = 224;
constexpr uint32_t kClusterCtas = 4;
constexpr uint32_t kDynamicSmemBytes = 256;

// Kernels indexed by [dtype][world size]: dtype 0 = float16, 1 = bfloat16;
// world index 0 = 2 ranks, 1 = 4 ranks, 2 = 8 ranks.
const void* const kGenericKernels[2][3] = {
    {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws2_o0110),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws4_o0110),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110)},
    {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110)},
};

#if CAKE_MOE_AR_SM103_T1
// SM103, one token: specialised schedule.
const void* const kSm103T1Kernels[2][3] = {
    {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws2_o0110_sm103_t1),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws4_o0110_sm103_t1),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm103_t1)},
    {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws2_o0110_sm103_t1),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws4_o0110_sm103_t1),
     reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm103_t1)},
};
#endif

#if CAKE_MOE_AR_SM100_WS8_MID
// SM100, eight ranks, 64 or 128 tokens: specialised schedule.
const void* const kSm100Ws8MidKernels[2] = {
    reinterpret_cast<const void*>(kernel_cake_trtllm_moe_reduction_float16_ws8_o0110_sm100_ws8_mid),
    reinterpret_cast<const void*>(
        kernel_cake_trtllm_moe_reduction_bfloat16_ws8_o0110_sm100_ws8_mid),
};
#endif

const void* SelectKernel(int32_t dtype_index, int32_t world_index, int64_t world_size,
                         int64_t token_num) {
#if CAKE_MOE_AR_SM103_T1
  if (token_num == 1) {
    return kSm103T1Kernels[dtype_index][world_index];
  }
#endif
#if CAKE_MOE_AR_SM100_WS8_MID
  if (world_size == 8 && (token_num == 64 || token_num == 128)) {
    return kSm100Ws8MidKernels[dtype_index];
  }
#endif
  return kGenericKernels[dtype_index][world_index];
}

int32_t DTypeIndex(const TensorView& tensor, const char* name) {
  DLDataType dtype = tensor.dtype();
  TVM_FFI_CHECK(
      dtype.lanes == 1 && dtype.bits == 16 && (dtype.code == kDLFloat || dtype.code == kDLBfloat),
      ValueError)
      << name << " must have dtype float16 or bfloat16";
  return dtype.code == kDLBfloat ? 1 : 0;
}

int32_t WorldIndex(int64_t world_size) {
  if (world_size == 2) return 0;
  if (world_size == 4) return 1;
  if (world_size == 8) return 2;
  TVM_FFI_LOG_AND_THROW(ValueError) << "Cake MoE world size must be 2, 4, or 8";
  return -1;
}

// The SM count is immutable per device; query it once and keep it.
int32_t MultiprocessorCount(int device_id) {
  constexpr int kMaxCachedCudaDevices = 64;
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxCachedCudaDevices, RuntimeError)
      << "SM-count cache does not cover cuda:" << device_id;
  static std::atomic<int> count_by_device[kMaxCachedCudaDevices]{};
  int cached = count_by_device[device_id].load(std::memory_order_acquire);
  if (cached > 0) return cached;
  int count = 0;
  cudaError_t error = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id);
  TVM_FFI_CHECK(error == cudaSuccess && count > 0, RuntimeError)
      << "querying multiProcessorCount failed for cuda:" << device_id
      << ": cudaError=" << static_cast<int>(error);
  count_by_device[device_id].store(count, std::memory_order_release);
  return count;
}

void RunReduction(int64_t world_size, int64_t world_rank, int64_t token_num, int64_t hidden_dim,
                  TensorView workspace_ptrs, bool launch_with_pdl, TensorView residual_in,
                  TensorView rms_gamma, double rms_eps, double scale_factor, int64_t active_experts,
                  TensorView expert_scales, TensorView active_expert_tokens, TensorView token_input,
                  Optional<TensorView> moe_allreduce_out, TensorView residual_out,
                  TensorView norm_out, Optional<double> weight_bias) {
  check_cuda_tensor(active_expert_tokens, "active_expert_tokens");
  check_same_device(expert_scales, active_expert_tokens, "expert_scales", "active_expert_tokens");
  check_same_device(token_input, active_expert_tokens, "token_input", "active_expert_tokens");
  check_same_device(residual_in, active_expert_tokens, "residual_in", "active_expert_tokens");
  check_same_device(rms_gamma, active_expert_tokens, "rms_gamma", "active_expert_tokens");
  check_same_device(workspace_ptrs, active_expert_tokens, "workspace_ptrs", "active_expert_tokens");
  check_same_device(residual_out, active_expert_tokens, "residual_out", "active_expert_tokens");
  check_same_device(norm_out, active_expert_tokens, "norm_out", "active_expert_tokens");
  TVM_FFI_CHECK(!moe_allreduce_out.has_value(), ValueError)
      << "the Cake MoE all-reduce fusion bundle has no all-reduce output kernels; "
         "calls with moe_allreduce_out are served by the union export";

  int32_t dtype_index = DTypeIndex(active_expert_tokens, "active_expert_tokens");
  TVM_FFI_CHECK(hidden_dim == kHiddenDim, ValueError)
      << "Cake MoE hidden dimension must be " << kHiddenDim;
  int32_t world_index = WorldIndex(world_size);
  TVM_FFI_CHECK(token_num > 0 && token_num <= INT32_MAX, ValueError)
      << "token_num must be a positive 32-bit integer";

  DLDevice device = active_expert_tokens.device();
  int32_t sm_count = MultiprocessorCount(device.device_id);
  int32_t tokens32 = static_cast<int32_t>(token_num);
  int32_t grid_x = std::min(sm_count, tokens32 * static_cast<int32_t>(kClusterCtas));
  grid_x = (grid_x / kClusterCtas) * kClusterCtas;
  TVM_FFI_CHECK(grid_x >= static_cast<int32_t>(kClusterCtas), ValueError)
      << "Cake MoE launch grid must contain one cluster";

  int32_t rank32 = static_cast<int32_t>(world_rank);
  int32_t experts32 = static_cast<int32_t>(active_experts);
  float eps32 = static_cast<float>(rms_eps);
  float weight_bias32 = weight_bias.has_value() ? static_cast<float>(weight_bias.value()) : 0.0f;
  float scale_factor32 = static_cast<float>(scale_factor);
  int32_t unused_layout = 0;

  void* p_active = active_expert_tokens.data_ptr();
  void* p_scales = expert_scales.data_ptr();
  void* p_token = token_input.data_ptr();
  void* p_residual = residual_in.data_ptr();
  void* p_gamma = rms_gamma.data_ptr();
  void* p_moe_out = nullptr;
  void* p_residual_out = residual_out.data_ptr();
  void* p_norm_out = norm_out.data_ptr();
  void* p_quant_out = nullptr;
  void* p_scale_out = nullptr;
  void* p_workspace = workspace_ptrs.data_ptr();
  void* args[] = {&p_active,      &p_scales,       &p_token,      &p_residual,  &p_gamma,
                  &p_moe_out,     &p_residual_out, &p_norm_out,   &p_quant_out, &p_scale_out,
                  &p_workspace,   &rank32,         &tokens32,     &experts32,   &eps32,
                  &weight_bias32, &scale_factor32, &unused_layout};

  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kClusterCtas;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  unsigned int num_attrs = 1;
  if (launch_with_pdl) {
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    num_attrs = 2;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = get_stream(device);
  config.attrs = attrs;
  config.numAttrs = num_attrs;

  const void* kernel = SelectKernel(dtype_index, world_index, world_size, token_num);
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "Cake MoE all-reduce fusion kernel launch failed: " << cudaGetErrorString(status);
}

}  // namespace cake_trtllm_moe_allreduce_fusion

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_reduction, cake_trtllm_moe_allreduce_fusion::RunReduction);
