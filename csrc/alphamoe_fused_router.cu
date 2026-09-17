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

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030
#error "AlphaMoE fused router is supported only on SM100a and SM103a"
#endif

#include <cuda_fp8.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#if defined(FLASHINFER_ALPHAMOE_SM_100A)
TVM_FFI_EMBED_CUBIN(alphamoe_large_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_large_routed_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_medium_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_medium_routed_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_small_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_small_routed_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_tiny_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_tiny_routed_sm_100a);
TVM_FFI_EMBED_CUBIN(alphamoe_large_tail_sm_100a);
#endif
#if defined(FLASHINFER_ALPHAMOE_SM_103A)
TVM_FFI_EMBED_CUBIN(alphamoe_large_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_large_routed_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_medium_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_medium_routed_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_small_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_small_routed_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_tiny_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_tiny_routed_sm_103a);
TVM_FFI_EMBED_CUBIN(alphamoe_large_tail_sm_103a);
#endif
namespace alphamoe_router_large_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 4224;
}  // namespace alphamoe_router_large_generated
namespace alphamoe_router_large_routed_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 4224;
}  // namespace alphamoe_router_large_routed_generated
namespace alphamoe_router_medium_generated {
constexpr int kThreads = 128;
constexpr int kSmemTotal = 4224;
}  // namespace alphamoe_router_medium_generated
namespace alphamoe_router_medium_routed_generated {
constexpr int kThreads = 128;
constexpr int kSmemTotal = 4224;
}  // namespace alphamoe_router_medium_routed_generated
namespace alphamoe_router_small_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 20608;
}  // namespace alphamoe_router_small_generated
namespace alphamoe_router_small_routed_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 20608;
}  // namespace alphamoe_router_small_routed_generated
namespace alphamoe_router_tiny_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 20608;
}  // namespace alphamoe_router_tiny_generated
namespace alphamoe_router_tiny_routed_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 20608;
}  // namespace alphamoe_router_tiny_routed_generated
namespace alphamoe_router_large_tail_generated {
constexpr int kThreads = 256;
constexpr int kSmemTotal = 4224;
}  // namespace alphamoe_router_large_tail_generated

namespace flashinfer {
namespace alphamoe_fused_router {

constexpr int64_t kMaxExperts = 512;
constexpr int64_t kMaxTopK = 16;
constexpr int64_t kMaxBlockM = 16;
inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

using RouterKernels = std::array<tvm::ffi::CubinKernel*, 9>;

inline RouterKernels GetRouterKernels(int minor) {
#if defined(FLASHINFER_ALPHAMOE_SM_100A)
  if (minor == 0) {
    static auto large =
        TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_large_sm_100a, "kernel_alpha_moe_fused_router");
    static auto large_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_large_routed_sm_100a,
                                                              "kernel_alpha_moe_fused_router");
    static auto medium = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_medium_sm_100a,
                                                        "kernel_alpha_moe_fused_router_medium");
    static auto medium_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_medium_routed_sm_100a, "kernel_alpha_moe_fused_router_medium");
    static auto small = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_small_sm_100a,
                                                       "kernel_alpha_moe_fused_router_small");
    static auto small_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_small_routed_sm_100a, "kernel_alpha_moe_fused_router_small");
    static auto tiny = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_tiny_sm_100a,
                                                      "kernel_alpha_moe_fused_router_small");
    static auto tiny_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_tiny_routed_sm_100a,
                                                             "kernel_alpha_moe_fused_router_small");
    static auto large_tail = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_large_tail_sm_100a, "kernel_alpha_moe_fused_router_large_tail");
    return {&large,        &large_routed, &medium,      &medium_routed, &small,
            &small_routed, &tiny,         &tiny_routed, &large_tail};
  }
#endif
#if defined(FLASHINFER_ALPHAMOE_SM_103A)
  if (minor == 3) {
    static auto large =
        TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_large_sm_103a, "kernel_alpha_moe_fused_router");
    static auto large_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_large_routed_sm_103a,
                                                              "kernel_alpha_moe_fused_router");
    static auto medium = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_medium_sm_103a,
                                                        "kernel_alpha_moe_fused_router_medium");
    static auto medium_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_medium_routed_sm_103a, "kernel_alpha_moe_fused_router_medium");
    static auto small = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_small_sm_103a,
                                                       "kernel_alpha_moe_fused_router_small");
    static auto small_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_small_routed_sm_103a, "kernel_alpha_moe_fused_router_small");
    static auto tiny = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_tiny_sm_103a,
                                                      "kernel_alpha_moe_fused_router_small");
    static auto tiny_routed = TVM_FFI_EMBED_CUBIN_GET_KERNEL(alphamoe_tiny_routed_sm_103a,
                                                             "kernel_alpha_moe_fused_router_small");
    static auto large_tail = TVM_FFI_EMBED_CUBIN_GET_KERNEL(
        alphamoe_large_tail_sm_103a, "kernel_alpha_moe_fused_router_large_tail");
    return {&large,        &large_routed, &medium,      &medium_routed, &small,
            &small_routed, &tiny,         &tiny_routed, &large_tail};
  }
#endif
  TVM_FFI_THROW(RuntimeError) << "AlphaMoE router was not built for this CUDA device";
  return {};
}

struct RouterLaunchConfig {
  int sm_count;
  int active_blocks_per_sm;
  int medium_active_blocks_per_sm;
  int small_active_blocks_per_sm;
  RouterKernels kernels;
};

inline void CheckCudaDriver(CUresult status) {
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << "CUDA driver error " << static_cast<int>(status);
}

inline RouterLaunchConfig GetRouterLaunchConfig(int32_t device_id) {
  static std::mutex mutex;
  static std::unordered_map<int32_t, RouterLaunchConfig> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = cache.find(device_id);
  if (cached != cache.end()) return cached->second;
  int major = 0, minor = 0, sm_count = 0, cooperative_launch = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "AlphaMoE fused router requires compute capability 10.0 or 10.3, got " << major << "."
      << minor;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiprocessor count)");
  CheckCuda(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device_id),
            "cudaDeviceGetAttribute(cooperative launch)");
  TVM_FFI_ICHECK(cooperative_launch != 0)
      << "AlphaMoE fused router requires cooperative-launch support";
  const RouterKernels kernels = GetRouterKernels(minor);
  CUfunction large_function = nullptr;
  CheckCudaDriver(
      cuKernelGetFunction(&large_function, reinterpret_cast<CUkernel>(kernels[0]->GetHandle())));
  int large_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &large_active, large_function, alphamoe_router_large_generated::kThreads,
      alphamoe_router_large_generated::kSmemTotal));
  TVM_FFI_ICHECK(large_active > 0) << "AlphaMoE large has zero occupancy";
  CUfunction large_routed_function = nullptr;
  CheckCudaDriver(cuKernelGetFunction(&large_routed_function,
                                      reinterpret_cast<CUkernel>(kernels[1]->GetHandle())));
  int large_routed_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &large_routed_active, large_routed_function, alphamoe_router_large_routed_generated::kThreads,
      alphamoe_router_large_routed_generated::kSmemTotal));
  TVM_FFI_ICHECK(large_routed_active > 0) << "AlphaMoE large_routed has zero occupancy";
  CUfunction medium_function = nullptr;
  CheckCudaDriver(
      cuKernelGetFunction(&medium_function, reinterpret_cast<CUkernel>(kernels[2]->GetHandle())));
  int medium_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &medium_active, medium_function, alphamoe_router_medium_generated::kThreads,
      alphamoe_router_medium_generated::kSmemTotal));
  TVM_FFI_ICHECK(medium_active > 0) << "AlphaMoE medium has zero occupancy";
  CUfunction medium_routed_function = nullptr;
  CheckCudaDriver(cuKernelGetFunction(&medium_routed_function,
                                      reinterpret_cast<CUkernel>(kernels[3]->GetHandle())));
  int medium_routed_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &medium_routed_active, medium_routed_function,
      alphamoe_router_medium_routed_generated::kThreads,
      alphamoe_router_medium_routed_generated::kSmemTotal));
  TVM_FFI_ICHECK(medium_routed_active > 0) << "AlphaMoE medium_routed has zero occupancy";
  CUfunction small_function = nullptr;
  CheckCudaDriver(
      cuKernelGetFunction(&small_function, reinterpret_cast<CUkernel>(kernels[4]->GetHandle())));
  int small_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &small_active, small_function, alphamoe_router_small_generated::kThreads,
      alphamoe_router_small_generated::kSmemTotal));
  TVM_FFI_ICHECK(small_active > 0) << "AlphaMoE small has zero occupancy";
  CUfunction small_routed_function = nullptr;
  CheckCudaDriver(cuKernelGetFunction(&small_routed_function,
                                      reinterpret_cast<CUkernel>(kernels[5]->GetHandle())));
  int small_routed_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &small_routed_active, small_routed_function, alphamoe_router_small_routed_generated::kThreads,
      alphamoe_router_small_routed_generated::kSmemTotal));
  TVM_FFI_ICHECK(small_routed_active > 0) << "AlphaMoE small_routed has zero occupancy";
  CUfunction tiny_function = nullptr;
  CheckCudaDriver(
      cuKernelGetFunction(&tiny_function, reinterpret_cast<CUkernel>(kernels[6]->GetHandle())));
  int tiny_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &tiny_active, tiny_function, alphamoe_router_tiny_generated::kThreads,
      alphamoe_router_tiny_generated::kSmemTotal));
  TVM_FFI_ICHECK(tiny_active > 0) << "AlphaMoE tiny has zero occupancy";
  CUfunction tiny_routed_function = nullptr;
  CheckCudaDriver(cuKernelGetFunction(&tiny_routed_function,
                                      reinterpret_cast<CUkernel>(kernels[7]->GetHandle())));
  int tiny_routed_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &tiny_routed_active, tiny_routed_function, alphamoe_router_tiny_routed_generated::kThreads,
      alphamoe_router_tiny_routed_generated::kSmemTotal));
  TVM_FFI_ICHECK(tiny_routed_active > 0) << "AlphaMoE tiny_routed has zero occupancy";
  CUfunction large_tail_function = nullptr;
  CheckCudaDriver(cuKernelGetFunction(&large_tail_function,
                                      reinterpret_cast<CUkernel>(kernels[8]->GetHandle())));
  int large_tail_active = 0;
  CheckCudaDriver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &large_tail_active, large_tail_function, alphamoe_router_large_tail_generated::kThreads,
      alphamoe_router_large_tail_generated::kSmemTotal));
  TVM_FFI_ICHECK(large_tail_active > 0) << "AlphaMoE large_tail has zero occupancy";
  const RouterLaunchConfig config{
      sm_count, std::min(large_active, large_routed_active),
      std::min(medium_active, medium_routed_active),
      std::min({small_active, small_routed_active, tiny_active, tiny_routed_active}), kernels};
  cache.emplace(device_id, config);
  return config;
}

inline void LaunchRouterKernel(tvm::ffi::CubinKernel& kernel, unsigned int grid_x,
                               unsigned int threads, unsigned int smem, bool cooperative,
                               cudaStream_t stream, void** args) {
  if (cooperative) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    cudaLaunchConfig_t launch{};
    launch.gridDim = {grid_x, 1, 1};
    launch.blockDim = {threads, 1, 1};
    launch.dynamicSmemBytes = smem;
    launch.stream = stream;
    launch.attrs = &attribute;
    launch.numAttrs = 1;
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(args, launch));
  } else {
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(
        args, tvm::ffi::dim3(grid_x, 1, 1), tvm::ffi::dim3(threads, 1, 1), stream, smem));
  }
}

inline void CheckTensor(const TensorView& tensor, const char* name, DLDataType dtype, int64_t ndim,
                        int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(tensor.dtype() == dtype) << name << " has an unsupported dtype";
  TVM_FFI_ICHECK(tensor.ndim() == ndim)
      << name << " must be rank " << ndim << ", got rank " << tensor.ndim();
}

struct TensorRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorRange GetTensorRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " must have a byte-addressable dtype";
  const uint64_t bytes_per_element = bits / 8;
  TVM_FFI_ICHECK(static_cast<uint64_t>(tensor.numel()) <=
                 std::numeric_limits<uint64_t>::max() / bytes_per_element)
      << name << " byte count overflows uint64_t";
  const uint64_t bytes = static_cast<uint64_t>(tensor.numel()) * bytes_per_element;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoAlias(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                         const char* rhs_name) {
  const TensorRange lhs_range = GetTensorRange(lhs, lhs_name);
  const TensorRange rhs_range = GetTensorRange(rhs, rhs_name);
  TVM_FFI_ICHECK(!(lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end))
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

inline int64_t MaxRouteBlocks(int64_t m, int64_t top_k, int64_t num_experts, int64_t block_m) {
  const int64_t pairs = m * top_k;
  const int64_t nonempty = std::min(num_experts, pairs);
  return nonempty + (pairs - nonempty) / block_m;
}

void Run(TensorView logits, TensorView topk_weights, TensorView topk_ids,
         TensorView sorted_token_ids, TensorView expert_ids, TensorView num_tokens_post_padded,
         TensorView expert_counts, TensorView expert_offsets, TensorView expert_scatter_offsets,
         int64_t top_k, int64_t block_m, bool has_shared_expert) {
  TVM_FFI_ICHECK(logits.device().device_type == kDLCUDA) << "logits must be a CUDA tensor";
  const int32_t device_id = logits.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);

  CheckTensor(logits, "logits", dl_float32, 2, device_id);
  CheckTensor(topk_weights, "topk_weights", dl_float32, 2, device_id);
  CheckTensor(topk_ids, "topk_ids", dl_int32, 2, device_id);
  CheckTensor(sorted_token_ids, "sorted_token_ids", dl_int32, 1, device_id);
  CheckTensor(expert_ids, "expert_ids", dl_int32, 1, device_id);
  CheckTensor(num_tokens_post_padded, "num_tokens_post_padded", dl_int32, 1, device_id);
  CheckTensor(expert_counts, "expert_counts", dl_int32, 1, device_id);
  CheckTensor(expert_offsets, "expert_offsets", dl_int32, 1, device_id);
  CheckTensor(expert_scatter_offsets, "expert_scatter_offsets", dl_int32, 1, device_id);

  const int64_t m = logits.size(0);
  const int64_t num_experts = logits.size(1);
  TVM_FFI_ICHECK(m > 0) << "logits must contain at least one token";
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= kMaxExperts)
      << "num_experts must be in [1, " << kMaxExperts << "], got " << num_experts;
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= std::min(num_experts, kMaxTopK))
      << "top_k must be in [1, min(num_experts, " << kMaxTopK << ")], got " << top_k;
  TVM_FFI_ICHECK(block_m >= 1 && block_m <= kMaxBlockM)
      << "block_m must be in [1, " << kMaxBlockM << "], got " << block_m;
  TVM_FFI_ICHECK(!has_shared_expert || top_k >= 2) << "a forced shared expert requires top_k >= 2";
  TVM_FFI_ICHECK(m <= std::numeric_limits<int>::max()) << "num_tokens must fit in int32";
  TVM_FFI_ICHECK(m * top_k <= std::numeric_limits<int>::max())
      << "num_tokens * top_k must fit in int32";

  const int64_t max_route_blocks = MaxRouteBlocks(m, top_k, num_experts, block_m);
  TVM_FFI_ICHECK(max_route_blocks <= std::numeric_limits<int>::max() / block_m)
      << "maximum padded route count exceeds int32";
  const int64_t max_padded_pairs = max_route_blocks * block_m;

  TVM_FFI_ICHECK(topk_weights.size(0) == m && topk_weights.size(1) == top_k)
      << "topk_weights must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(topk_ids.size(0) == m && topk_ids.size(1) == top_k)
      << "topk_ids must have shape (" << m << ", " << top_k << ")";
  TVM_FFI_ICHECK(sorted_token_ids.numel() >= max_padded_pairs)
      << "sorted_token_ids capacity must be at least " << max_padded_pairs;
  TVM_FFI_ICHECK(expert_ids.numel() >= max_route_blocks)
      << "expert_ids capacity must be at least " << max_route_blocks;
  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must contain exactly one int32 element";
  TVM_FFI_ICHECK(expert_counts.numel() == num_experts)
      << "expert_counts must have shape (" << num_experts << ",)";
  TVM_FFI_ICHECK(expert_offsets.numel() == num_experts + 1)
      << "expert_offsets must have shape (" << num_experts + 1 << ",)";
  TVM_FFI_ICHECK(expert_scatter_offsets.numel() == num_experts)
      << "expert_scatter_offsets must have shape (" << num_experts << ",)";

  const std::array<const TensorView*, 9> tensors = {
      &logits,           &topk_weights,   &topk_ids,
      &sorted_token_ids, &expert_ids,     &num_tokens_post_padded,
      &expert_counts,    &expert_offsets, &expert_scatter_offsets,
  };
  const std::array<const char*, 9> names = {
      "logits",           "topk_weights",   "topk_ids",
      "sorted_token_ids", "expert_ids",     "num_tokens_post_padded",
      "expert_counts",    "expert_offsets", "expert_scatter_offsets",
  };
  for (size_t i = 0; i < tensors.size(); ++i) {
    for (size_t j = i + 1; j < tensors.size(); ++j) {
      CheckNoAlias(*tensors[i], names[i], *tensors[j], names[j]);
    }
  }

  const RouterLaunchConfig config = GetRouterLaunchConfig(device_id);
  const bool use_tiny = m <= 8;
  const bool use_small = m <= 128;
  const bool use_medium = m <= 512;
  int64_t grid_x =
      use_tiny ? 1 : std::max<int64_t>(1, std::min<int64_t>((m + 7) / 8, config.sm_count));
  int active_blocks = config.small_active_blocks_per_sm;
  if (!use_small && use_medium) {
    active_blocks = config.medium_active_blocks_per_sm;
    grid_x = std::max<int64_t>(
        1, std::min<int64_t>((m + 3) / 4, static_cast<int64_t>(config.sm_count) * active_blocks));
  } else if (!use_medium) {
    active_blocks = config.active_blocks_per_sm;
    const int64_t full_wave_grid =
        (((m + 3) / 4 + config.sm_count - 1) / config.sm_count) * config.sm_count;
    grid_x = std::max<int64_t>(
        1, std::min<int64_t>(full_wave_grid,
                             static_cast<int64_t>(config.sm_count) * std::min(5, active_blocks)));
  }
  const int64_t cooperative_capacity = static_cast<int64_t>(active_blocks) * config.sm_count;
  TVM_FFI_ICHECK(grid_x >= 1 && grid_x <= cooperative_capacity)
      << "AlphaMoE fused router grid " << grid_x << " exceeds cooperative residency capacity "
      << cooperative_capacity;

  float* logits_ptr = static_cast<float*>(logits.data_ptr());
  float* topk_weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  int* topk_ids_ptr = static_cast<int*>(topk_ids.data_ptr());
  int* sorted_token_ids_ptr = static_cast<int*>(sorted_token_ids.data_ptr());
  int* expert_ids_ptr = static_cast<int*>(expert_ids.data_ptr());
  int* num_tokens_post_padded_ptr = static_cast<int*>(num_tokens_post_padded.data_ptr());
  int* expert_counts_ptr = static_cast<int*>(expert_counts.data_ptr());
  int* expert_offsets_ptr = static_cast<int*>(expert_offsets.data_ptr());
  int* expert_scatter_offsets_ptr = static_cast<int*>(expert_scatter_offsets.data_ptr());
  int m_arg = static_cast<int>(m);
  int num_experts_arg = static_cast<int>(num_experts);
  int top_k_arg = static_cast<int>(top_k);
  int block_m_arg = static_cast<int>(block_m);
  int has_shared_expert_arg = static_cast<int>(has_shared_expert);
  void* arguments[] = {
      &logits_ptr,
      &topk_weights_ptr,
      &topk_ids_ptr,
      &sorted_token_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_post_padded_ptr,
      &expert_counts_ptr,
      &expert_offsets_ptr,
      &expert_scatter_offsets_ptr,
      &m_arg,
      &num_experts_arg,
      &top_k_arg,
      &block_m_arg,
      &has_shared_expert_arg,
  };

  const cudaStream_t stream = get_stream(logits.device());
  if (use_tiny) {
    LaunchRouterKernel(*config.kernels[has_shared_expert ? 6 : 7],
                       static_cast<unsigned int>(grid_x), alphamoe_router_tiny_generated::kThreads,
                       alphamoe_router_tiny_generated::kSmemTotal, false, stream, arguments);
  } else if (use_small) {
    LaunchRouterKernel(*config.kernels[has_shared_expert ? 4 : 5],
                       static_cast<unsigned int>(grid_x), alphamoe_router_small_generated::kThreads,
                       alphamoe_router_small_generated::kSmemTotal, true, stream, arguments);
  } else if (use_medium) {
    LaunchRouterKernel(*config.kernels[has_shared_expert ? 2 : 3],
                       static_cast<unsigned int>(grid_x),
                       alphamoe_router_medium_generated::kThreads,
                       alphamoe_router_medium_generated::kSmemTotal, true, stream, arguments);
  } else {
    LaunchRouterKernel(*config.kernels[has_shared_expert ? 0 : 1],
                       static_cast<unsigned int>(grid_x), alphamoe_router_large_generated::kThreads,
                       alphamoe_router_large_generated::kSmemTotal, true, stream, arguments);
    LaunchRouterKernel(*config.kernels[8], static_cast<unsigned int>(grid_x),
                       alphamoe_router_large_tail_generated::kThreads,
                       alphamoe_router_large_tail_generated::kSmemTotal, false, stream, arguments);
  }
}

}  // namespace alphamoe_fused_router
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_router_op, flashinfer::alphamoe_fused_router::Run);
