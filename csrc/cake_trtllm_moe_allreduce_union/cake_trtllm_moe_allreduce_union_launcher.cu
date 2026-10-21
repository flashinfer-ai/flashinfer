/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// One tvm-ffi launcher for every kernel of the Cake MoE all-reduce union.
//
// The JIT compiles this translation unit once per kernel together with the
// kernel's own translation unit (``kernels/<name>.cu``) and selects the kernel
// through four preprocessor definitions:
//
//   CAKE_UNION_KERNEL      the extern "C" kernel symbol
//   CAKE_UNION_DTYPE       __half or __nv_bfloat16
//   CAKE_UNION_WORLD_SIZE  2, 4 or 8 (ranks of the collective)
//   CAKE_UNION_BLOCK       threads per CTA (224 or 896)
//   CAKE_UNION_CLUSTER     CTAs per cluster (4 or 1)
//
// PDL and the cooperative launch are runtime flags.  The kernels read every
// workspace address from the device pointer table (``workspace_tensor``); the
// typed control / per-peer payload origins are host-only resource arguments
// of the Cake source program and are not part of the generated signature.  The
// union builds emit no quant output, so the generated signature carries
// neither the quant pointers nor their scalars.  Every union kernel stores the
// all-reduce output; a caller without ``moe_allreduce_out`` is given a
// loader-owned scratch tensor by ``run_cake_moe_allreduce_union``.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "tvm_ffi_utils.h"

#ifndef CAKE_UNION_KERNEL
#error "CAKE_UNION_KERNEL must name the kernel symbol"
#endif
#ifndef CAKE_UNION_DTYPE
#error "CAKE_UNION_DTYPE must be __half or __nv_bfloat16"
#endif
#if CAKE_UNION_WORLD_SIZE != 2 && CAKE_UNION_WORLD_SIZE != 4 && CAKE_UNION_WORLD_SIZE != 8
#error "CAKE_UNION_WORLD_SIZE must be 2, 4 or 8"
#endif
#if CAKE_UNION_BLOCK * CAKE_UNION_CLUSTER != 896
#error "CAKE_UNION_BLOCK x CAKE_UNION_CLUSTER must cover one 7168-wide token (896 threads)"
#endif

namespace {

using T = CAKE_UNION_DTYPE;

}  // namespace

extern "C" __global__ void CAKE_UNION_KERNEL(
    T* active_expert_tokens, float* expert_scales, T* token_input, T* residual, T* gamma,
    T* moe_allreduce_out, T* residual_out, T* norm_out, long long* workspace_tensor,
    int world_rank, int tokens, int active_experts, float epsilon, float weight_bias);

namespace cake_trtllm_moe_allreduce_union {

using tvm::ffi::TensorView;

constexpr DLDataType kDType = std::is_same<T, __half>::value ? dl_float16 : dl_bfloat16;

inline void check_activation(const TensorView& t, const TensorView& reference, const char* name) {
  check_cuda_tensor(t, name);
  check_dtype(t, kDType, name);
  check_contiguous(t, name);
  check_same_device(t, reference, name, "active_expert_tokens");
}

inline void check_i32(int64_t value, const char* name) {
  TVM_FFI_CHECK(value >= INT32_MIN && value <= INT32_MAX, ValueError)
      << "scalar '" << name << "' value " << value << " is outside the int32 range";
}

void Run(TensorView active_expert_tokens, TensorView expert_scales, TensorView token_input,
         TensorView residual, TensorView gamma, TensorView moe_allreduce_out,
         TensorView residual_out, TensorView norm_out, TensorView workspace_tensor,
         int64_t world_rank, int64_t tokens, int64_t active_experts, double epsilon,
         double weight_bias, int64_t grid_x, bool use_pdl, bool cooperative) {
  check_cuda_tensor(active_expert_tokens, "active_expert_tokens");
  check_dtype(active_expert_tokens, kDType, "active_expert_tokens");
  check_contiguous(active_expert_tokens, "active_expert_tokens");
  check_cuda_tensor(expert_scales, "expert_scales");
  check_dtype(expert_scales, dl_float32, "expert_scales");
  check_contiguous(expert_scales, "expert_scales");
  check_same_device(expert_scales, active_expert_tokens, "expert_scales", "active_expert_tokens");
  check_activation(token_input, active_expert_tokens, "token_input");
  check_activation(residual, active_expert_tokens, "residual");
  check_activation(gamma, active_expert_tokens, "gamma");
  check_activation(moe_allreduce_out, active_expert_tokens, "moe_allreduce_out");
  check_activation(residual_out, active_expert_tokens, "residual_out");
  check_activation(norm_out, active_expert_tokens, "norm_out");
  check_cuda_tensor(workspace_tensor, "workspace_tensor");
  check_dtype(workspace_tensor, dl_int64, "workspace_tensor");
  check_contiguous(workspace_tensor, "workspace_tensor");
  check_same_device(workspace_tensor, active_expert_tokens, "workspace_tensor",
                    "active_expert_tokens");
  check_i32(world_rank, "world_rank");
  check_i32(tokens, "tokens");
  check_i32(active_experts, "active_experts");
  TVM_FFI_CHECK(world_rank >= 0 && world_rank < CAKE_UNION_WORLD_SIZE, ValueError)
      << "world_rank " << world_rank << " is outside [0, " << CAKE_UNION_WORLD_SIZE << ")";
  TVM_FFI_CHECK(grid_x > 0, ValueError) << "launch grid must be positive, got " << grid_x;
  TVM_FFI_CHECK(grid_x % CAKE_UNION_CLUSTER == 0, ValueError)
      << "launch grid " << grid_x << " must be divisible by the cluster width "
      << CAKE_UNION_CLUSTER;

  const DLDevice dev = active_expert_tokens.device();
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(current_device == dev.device_id, ValueError)
      << "active_expert_tokens lives on cuda:" << dev.device_id
      << " but the current device is cuda:" << current_device;
  cudaStream_t stream = get_stream(dev);

  T* p_active_expert_tokens = static_cast<T*>(active_expert_tokens.data_ptr());
  float* p_expert_scales = static_cast<float*>(expert_scales.data_ptr());
  T* p_token_input = static_cast<T*>(token_input.data_ptr());
  T* p_residual = static_cast<T*>(residual.data_ptr());
  T* p_gamma = static_cast<T*>(gamma.data_ptr());
  T* p_residual_out = static_cast<T*>(residual_out.data_ptr());
  T* p_norm_out = static_cast<T*>(norm_out.data_ptr());
  T* p_moe_allreduce_out = static_cast<T*>(moe_allreduce_out.data_ptr());
  long long* p_workspace_tensor = static_cast<long long*>(workspace_tensor.data_ptr());
  int32_t v_world_rank = static_cast<int32_t>(world_rank);
  int32_t v_tokens = static_cast<int32_t>(tokens);
  int32_t v_active_experts = static_cast<int32_t>(active_experts);
  float v_epsilon = static_cast<float>(epsilon);
  float v_weight_bias = static_cast<float>(weight_bias);

  void* kargs[] = {
      &p_active_expert_tokens,
      &p_expert_scales,
      &p_token_input,
      &p_residual,
      &p_gamma,
      &p_moe_allreduce_out,
      &p_residual_out,
      &p_norm_out,
      &p_workspace_tensor,
      &v_world_rank,
      &v_tokens,
      &v_active_experts,
      &v_epsilon,
      &v_weight_bias,
  };

  cudaLaunchAttribute attrs[4]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = CAKE_UNION_CLUSTER;
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
  attrs[n].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[n].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
  ++n;
  if (cooperative) {
    attrs[n].id = cudaLaunchAttributeCooperative;
    attrs[n].val.cooperative = 1;
    ++n;
  }
  if (use_pdl) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x), 1u, 1u);
  config.blockDim = dim3(CAKE_UNION_BLOCK, 1u, 1u);
  config.dynamicSmemBytes = 256u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
  status = cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(CAKE_UNION_KERNEL), kargs);
  if (status != cudaSuccess) {
    // A rejected launch (for example a cooperative grid larger than the co-resident
    // capacity) is not sticky; clear it so the caller's next CUDA call does not
    // report this launch's error.
    (void)cudaGetLastError();
  }
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "Cake MoE all-reduce union launch failed: " << cudaGetErrorString(status);
}

}  // namespace cake_trtllm_moe_allreduce_union

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_trtllm_moe_allreduce_union::Run);
