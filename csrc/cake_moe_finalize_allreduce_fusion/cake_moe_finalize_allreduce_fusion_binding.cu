/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host launcher for the Cake MoE finalize + all-reduce fusion kernels. One module per
// architecture links the programs of each dtype x world size x optional NVFP4 epilogue:
// a small-token program and a persistent program, and at world sizes 4 and 8 a
// large-token program. The caller passes the largest token count that runs the
// small-token program; above it the persistent program runs, or the large-token program
// once every cluster of the persistent program would hold at least three tokens. The
// launch configuration is shared and PDL is a launch attribute chosen at run time.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

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
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws2_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws2_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110_small)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111_small)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o110_lt)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws4_o111_lt)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o110_lt)
CAKE_MOE_FINALIZE_KERNEL(__half, kernel_cake_trtllm_moe_finalize_float16_ws8_o111_lt)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110_lt)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111_lt)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110_lt)
CAKE_MOE_FINALIZE_KERNEL(__nv_bfloat16, kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111_lt)

#undef CAKE_MOE_FINALIZE_KERNEL

// Launch shape shared by every program: threads per CTA, CTAs per cluster, dynamic shared
// memory bytes.
#define CAKE_MOE_FINALIZE_THREADS 224
#define CAKE_MOE_FINALIZE_CLUSTER_CTAS 4
#define CAKE_MOE_FINALIZE_DYNAMIC_SMEM_BYTES 256

namespace {

constexpr int64_t kHiddenDim = 7168;
constexpr int kMaxDevices = 64;

// Persistent programs, [dtype][world size index][quant epilogue]
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

// Small-token programs, same order as kKernels
const void* const kSmallKernels[2][3][2] = {
    {{reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws2_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws2_o111_small)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o111_small)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o111_small)}},
    {{reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111_small)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111_small)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110_small),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111_small)}},
};

// Large-token programs, same order as kKernels; world size 2 has none.
const void* const kLargeTokenKernels[2][3][2] = {
    {{nullptr, nullptr},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o110_lt),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws4_o111_lt)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o110_lt),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_float16_ws8_o111_lt)}},
    {{nullptr, nullptr},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110_lt),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111_lt)},
     {reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110_lt),
      reinterpret_cast<const void*>(kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111_lt)}},
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

constexpr int kMaxCtasPerSm = 5;

// Resident CTAs per SM for one kernel (register/shared-memory occupancy), capped at kMaxCtasPerSm.
// The persistent programs are bound by the bytes each SM keeps in flight: 5 CTAs per SM (the
// 56-register programs) is faster than 4, and 3 is slower.  Cached per (device, kernel).
int64_t ctas_per_sm(int device_id, const void* kernel) {
  static std::mutex mutex;
  static std::map<std::pair<int, const void*>, int64_t> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find({device_id, kernel});
  if (it != cache.end()) return it->second;
  int blocks = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel, CAKE_MOE_FINALIZE_THREADS, CAKE_MOE_FINALIZE_DYNAMIC_SMEM_BYTES);
  TVM_FFI_CHECK(status == cudaSuccess && blocks > 0, RuntimeError)
      << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed for cuda:" << device_id << ": "
      << cudaGetErrorString(status);
  const int64_t ctas = std::min<int64_t>(blocks, kMaxCtasPerSm);
  cache[{device_id, kernel}] = ctas;
  return ctas;
}

// Four-CTA clusters of one kernel that the device holds resident at once (cluster placement can
// leave SMs unused, so this can be below ctas_per_sm * sm_count / 4).  Every cluster of a launch
// must be resident together: a cluster waits for its peer ranks' clusters, and a cluster left for
// a second wave would wait on a rank whose resident clusters wait on it.  Cached per device and
// kernel.
int64_t max_active_clusters(int device_id, const void* kernel, cudaLaunchConfig_t config) {
  static std::mutex mutex;
  static std::map<std::pair<int, const void*>, int64_t> cache;
  std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find({device_id, kernel});
  if (it != cache.end()) return it->second;
  config.gridDim = dim3(CAKE_MOE_FINALIZE_CLUSTER_CTAS, 1, 1);
  int clusters = 0;
  cudaError_t status = cudaOccupancyMaxActiveClusters(&clusters, kernel, &config);
  TVM_FFI_CHECK(status == cudaSuccess && clusters > 0, RuntimeError)
      << "cudaOccupancyMaxActiveClusters failed for cuda:" << device_id << ": "
      << cudaGetErrorString(status);
  cache[{device_id, kernel}] = clusters;
  return clusters;
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
// Batches of at most small_token_max tokens run the small-token program; larger batches
// run the persistent program, or at world sizes 4 and 8 the large-token program when the
// batch exceeds twice the persistent program's launched clusters.
void cake_moe_finalize_allreduce_fusion(
    TensorView allreduce_in, TensorView residual_in, TensorView norm_weight,
    TensorView expanded_idx_to_permuted_idx, TensorView expert_scale_factor,
    Optional<TensorView> shared_expert_output, TensorView residual_out, TensorView norm_out,
    Optional<TensorView> quant_out, Optional<TensorView> scale_out, TensorView workspace_ptrs,
    int64_t world_rank, int64_t world_size, int64_t small_token_max, double eps,
    double routed_scaling_factor, double weight_bias, bool enable_pdl) {
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
  const int ws_index = world_size == 2 ? 0 : (world_size == 4 ? 1 : 2);
  const int quant_index = quantize ? 1 : 0;

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

  // clusterDim.x is CAKE_MOE_FINALIZE_CLUSTER_CTAS.
  cudaLaunchAttribute attrs[2]{};
  unsigned n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = 4;
  attrs[n].val.clusterDim.y = 1;
  attrs[n].val.clusterDim.z = 1;
  ++n;
  if (enable_pdl) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cudaLaunchConfig_t config{};
  config.blockDim = dim3(CAKE_MOE_FINALIZE_THREADS, 1, 1);
  config.dynamicSmemBytes = CAKE_MOE_FINALIZE_DYNAMIC_SMEM_BYTES;
  config.stream = get_stream(device);
  config.attrs = attrs;
  config.numAttrs = n;

  // Persistent grid: enough four-CTA clusters to keep every SM at its resident CTA count
  // (occupancy-derived, capped at kMaxCtasPerSm), never more clusters than tokens.  One CTA per SM
  // leaves the HBM/NVLink paths 3-5x under-subscribed for tokens >= 128 (measured on B200: 1.9x at
  // 128 tokens, 2.8x at 2048 with four).  The occupancy is the program's own, and the launch
  // further caps the cluster count at the device's active-cluster capacity for the program
  // (max_active_clusters).
  auto launched_clusters = [&](const void* program) {
    const int64_t wanted_clusters =
        std::min<int64_t>(ctas_per_sm(device.device_id, program) * sm_count(device.device_id),
                          tokens * CAKE_MOE_FINALIZE_CLUSTER_CTAS) /
        CAKE_MOE_FINALIZE_CLUSTER_CTAS;
    return std::min<int64_t>(wanted_clusters,
                             max_active_clusters(device.device_id, program, config));
  };
  const void* kernel = nullptr;
  int64_t clusters = 0;
  if (tokens <= small_token_max) {
    kernel = kSmallKernels[dtype_index][ws_index][quant_index];
    clusters = launched_clusters(kernel);
  } else {
    kernel = kKernels[dtype_index][ws_index][quant_index];
    clusters = launched_clusters(kernel);
    // The large-token program is faster only when every cluster of the persistent program would
    // hold at least three tokens.  Large-token / persistent speed measured on B200 and B300:
    // world size 8, 284 tokens 0.990 / 0.996, 352 tokens 1.047 / 1.038; world size 4, 288 tokens
    // 0.92, 352 tokens 1.013 / 1.016.
    const void* large_token = kLargeTokenKernels[dtype_index][ws_index][quant_index];
    if (large_token != nullptr && tokens > 2 * clusters) {
      kernel = large_token;
      clusters = launched_clusters(kernel);
    }
  }
  TVM_FFI_CHECK(clusters > 0, RuntimeError)
      << "Cake MoE finalize requires one complete four-CTA cluster";
  config.gridDim = dim3(static_cast<unsigned>(clusters * CAKE_MOE_FINALIZE_CLUSTER_CTAS), 1, 1);
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, args);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "Cake MoE finalize launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_moe_finalize_allreduce_fusion,
                              cake_moe_finalize_allreduce_fusion);
