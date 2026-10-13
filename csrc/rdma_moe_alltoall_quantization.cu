/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#include "flashinfer/comm/rdma_moe_alltoall/kernels/launch.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/precision.cuh"

namespace deep_ep {
namespace extensions {

// Query the current device: one process may own GPUs with different SM counts.
int get_sm_count() {
  int device = 0, sms = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return sms;
}

template <bool UniformScale>
__global__ __launch_bounds__(1024) void quantize_bf16_to_nvfp4_kernel(
    const void* input, const float* global_scale_per_token, float uniform_global_scale,
    void* nvfp4_packed_output, void* fp8_scales, int token_num, int hidden_access_num) {
  for (int token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
    const float global_scale_val =
        UniformScale ? uniform_global_scale : __ldg(global_scale_per_token + token_idx);
    const size_t row = size_t(token_idx) * hidden_access_num;
    for (int access_idx = threadIdx.x; access_idx < hidden_access_num; access_idx += blockDim.x) {
      const auto input_buffer = reinterpret_cast<const int4*>(input) + row + access_idx;
      auto packed_output_buffer =
          reinterpret_cast<uint32_t*>(nvfp4_packed_output) + row + access_idx;
      auto scales_buffer = reinterpret_cast<uint8_t*>(fp8_scales) + row / 2 + access_idx / 2;
      auto [e2m1_vec, fp8_scale_val] =
          quantize_bf16_to_nvfp4(__ldg(input_buffer), global_scale_val);
      packed_output_buffer[0] = e2m1_vec;
      if (threadIdx.x % 2 == 0) {
        scales_buffer[0] = fp8_scale_val;
      }
    }
  }
}

void quantize_bf16_to_nvfp4(const void* input, const float* global_scale_per_token,
                            void* nvfp4_packed_output, void* fp8_scales, int token_num,
                            int hidden_dim, cudaStream_t stream) {
  constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
  EP_HOST_ASSERT(hidden_dim % kElemNumPerAccess == 0);
  int hidden_access_num = hidden_dim / kElemNumPerAccess;
  int grid_size = std::min(token_num, get_sm_count());
  int block_size = std::min(1024, ceil_div(hidden_access_num, 32));
  EP_HOST_ASSERT(grid_size > 0 and block_size > 0);
  SETUP_LAUNCH_CONFIG(grid_size, block_size, stream);
  LAUNCH_KERNEL(&cfg, quantize_bf16_to_nvfp4_kernel<false>, input, global_scale_per_token, 0.f,
                nvfp4_packed_output, fp8_scales, token_num, hidden_access_num);
}

void quantize_bf16_to_nvfp4_uniform(const void* input, float global_scale,
                                    void* nvfp4_packed_output, void* fp8_scales, int token_num,
                                    int hidden_dim, cudaStream_t stream) {
  constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
  EP_HOST_ASSERT(token_num >= 0 && hidden_dim > 0 && hidden_dim % kElemNumPerAccess == 0);
  if (token_num == 0) return;
  const int hidden_access_num = hidden_dim / kElemNumPerAccess;
  // Full warps preserve the adjacent-lane 16-value scale reduction while
  // giving each token enough threads to process its BF16 row in parallel.
  SETUP_LAUNCH_CONFIG(std::min(token_num, get_sm_count()), 256, stream);
  LAUNCH_KERNEL(&cfg, quantize_bf16_to_nvfp4_kernel<true>, input, nullptr, global_scale,
                nvfp4_packed_output, fp8_scales, token_num, hidden_access_num);
}

__global__ __launch_bounds__(1024) void dequantize_nvfp4_to_bf16_kernel(
    const void* nvfp4_packed_input, const float* global_scale_per_token, const void* fp8_scales,
    void* output, int token_num, int hidden_access_num) {
  constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
  for (int token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
    float global_scale_val = __ldg(global_scale_per_token + token_idx);
    const size_t row = size_t(token_idx) * hidden_access_num;
    for (int access_idx = threadIdx.x; access_idx < hidden_access_num; access_idx += blockDim.x) {
      const auto packed_input_buffer =
          reinterpret_cast<const uint32_t*>(nvfp4_packed_input) + row + access_idx;
      const auto scales_buffer =
          reinterpret_cast<const uint8_t*>(fp8_scales) + row / 2 + access_idx / 2;
      auto output_buffer = reinterpret_cast<int4*>(output) + row + access_idx;
      auto e2m1_vec = __ldg(packed_input_buffer);
      auto scale_val = __ldg(scales_buffer);
      float output_fp32[kElemNumPerAccess];
      dequantize_nvfp4_to_bf16<false>(e2m1_vec, global_scale_val, scale_val, output_fp32);
      int4 output_vec;
      for (int i = 0; i < kElemNumPerAccess; ++i) {
        reinterpret_cast<nv_bfloat16*>(&output_vec)[i] = static_cast<nv_bfloat16>(output_fp32[i]);
      }
      *output_buffer = output_vec;
    }
  }
}

void dequantize_nvfp4_to_bf16(const void* nvfp4_packed_input, const float* global_scale_per_token,
                              const void* fp8_scales, void* output, int token_num, int hidden_dim,
                              cudaStream_t stream) {
  constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
  EP_HOST_ASSERT(hidden_dim % kElemNumPerAccess == 0);
  int hidden_access_num = hidden_dim / kElemNumPerAccess;
  int grid_size = std::min(token_num, get_sm_count());
  int block_size = std::min(1024, ceil_div(hidden_access_num, 32));
  EP_HOST_ASSERT(grid_size > 0 and block_size > 0);
  SETUP_LAUNCH_CONFIG(grid_size, block_size, stream);
  LAUNCH_KERNEL(&cfg, dequantize_nvfp4_to_bf16_kernel, nvfp4_packed_input, global_scale_per_token,
                fp8_scales, output, token_num, hidden_access_num);
}

__global__ __launch_bounds__(256) void quantize_bf16_to_fp8_kernel(const void* input,
                                                                   void* fp8_output,
                                                                   void* fp32_scales, int token_num,
                                                                   int hidden_access_num) {
  for (int token = blockIdx.x; token < token_num; token += gridDim.x) {
    for (int i = threadIdx.x; i < hidden_access_num; i += blockDim.x) {
      const size_t row = size_t(token) * hidden_access_num;
      auto [value, scale] = quantize_bf16_to_fp8(__ldg(static_cast<const int4*>(input) + row + i));
      static_cast<int2*>(fp8_output)[row + i] = value;
      if (i % 16 == 0) static_cast<float*>(fp32_scales)[row / 16 + i / 16] = scale;
    }
  }
}

void quantize_bf16_to_fp8(const void* input, void* fp8_output, void* fp32_scales, int token_num,
                          int hidden_dim, cudaStream_t stream) {
  EP_HOST_ASSERT(token_num >= 0 && hidden_dim > 0 && hidden_dim % 128 == 0);
  if (token_num == 0) return;
  SETUP_LAUNCH_CONFIG(std::min(token_num, get_sm_count()), 256, stream);
  LAUNCH_KERNEL(&cfg, quantize_bf16_to_fp8_kernel, input, fp8_output, fp32_scales, token_num,
                hidden_dim / 8);
}

__global__ __launch_bounds__(256) void dequantize_fp8_to_bf16_kernel(const void* fp8_input,
                                                                     const void* fp32_scales,
                                                                     void* output, int token_num,
                                                                     int hidden_access_num) {
  for (int token = blockIdx.x; token < token_num; token += gridDim.x) {
    for (int i = threadIdx.x; i < hidden_access_num; i += blockDim.x) {
      const size_t row = size_t(token) * hidden_access_num;
      const auto value = __ldg(static_cast<const int2*>(fp8_input) + row + i);
      const auto scale = __ldg(static_cast<const float*>(fp32_scales) + row / 16 + i / 16);
      float values[8];
      dequantize_fp8_to_bf16<false>(value, scale, values);
      int4 result;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        reinterpret_cast<nv_bfloat16*>(&result)[j] = static_cast<nv_bfloat16>(values[j]);
      static_cast<int4*>(output)[row + i] = result;
    }
  }
}

void dequantize_fp8_to_bf16(const void* fp8_input, const void* fp32_scales, void* output,
                            int token_num, int hidden_dim, cudaStream_t stream) {
  EP_HOST_ASSERT(token_num >= 0 && hidden_dim > 0 && hidden_dim % 128 == 0);
  if (token_num == 0) return;
  SETUP_LAUNCH_CONFIG(std::min(token_num, get_sm_count()), 256, stream);
  LAUNCH_KERNEL(&cfg, dequantize_fp8_to_bf16_kernel, fp8_input, fp32_scales, output, token_num,
                hidden_dim / 8);
}

}  // namespace extensions
}  // namespace deep_ep
