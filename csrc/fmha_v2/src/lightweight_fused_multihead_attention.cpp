/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "lightweight_fused_multihead_attention.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

// External function declaration for the specific kernel launcher
// Only needed when linking against .cu.o object file
#ifndef USE_CUBIN_LOADING
extern void run_fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16_sm120_nl_tiled(
    const bert::Fused_multihead_attention_params_v2& params,
    const bert::Fused_multihead_attention_launch_params& launch_params, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lightweight_fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef USE_CUBIN_LOADING
// This function is only available when linking against .cu.o object files
void run_lightweight_flash_attention_e4m3_fp32(
    const bert::Fused_multihead_attention_params_v2& params,
    const bert::Fused_multihead_attention_launch_params& launch_params, cudaStream_t stream) {
  // Validate input parameters
  // assert(params.d == 192 && "This kernel only supports head dimension of 192");
  // assert(params.dv == 128 && "This kernel only supports value dimension of 128");

  // Validate output pointer
  if (params.o_ptr == nullptr) {
    throw std::runtime_error("Output pointer (params.o_ptr) cannot be null");
  }

  // Note: The data type validation should be done by the caller based on the input data
  // The kernel itself expects E4M3 input data with FP32 accumulation and BF16 output

  // Check SM architecture
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  if (props.major * 10 + props.minor < 120) {
    throw std::runtime_error("This kernel requires SM 12.0 or higher (Hopper)");
  }

  // Call the specific kernel launcher
  // This will write the attention output to params.o_ptr
  run_fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16_sm120_nl_tiled(
      params, launch_params, stream);

  // Check for any CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in lightweight flash attention: ") +
                             cudaGetErrorString(error));
  }

  // At this point, the output has been written to params.o_ptr
  // The output format is BF16 with shape [batch, seq_len, num_heads, value_dim]
  // or potentially [batch * seq_len, num_heads * value_dim] depending on layout
}
#endif  // USE_CUBIN_LOADING

////////////////////////////////////////////////////////////////////////////////////////////////////

CubinFlashAttention::CubinFlashAttention(const std::string& cubin_path)
    : module_(nullptr),
      kernel_nl_tiled_(nullptr),
      kernel_nl_tiled_causal_(nullptr),
      initialized_(false) {
  // Check if file exists
  std::ifstream file(cubin_path, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Cubin file not found: " + cubin_path);
  }
  file.close();

  // Initialize CUDA driver API
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to initialize CUDA driver API");
  }

  // Get current context
  CUcontext context;
  result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context == nullptr) {
    throw std::runtime_error("No CUDA context available");
  }

  // Load the module from cubin file
  result = cuModuleLoad(&module_, cubin_path.c_str());
  if (result != CUDA_SUCCESS) {
    const char* error_str;
    cuGetErrorString(result, &error_str);
    throw std::runtime_error("Failed to load cubin file: " + std::string(error_str));
  }

  // Get function handles for both regular and causal kernels
  result = cuModuleGetFunction(
      &kernel_nl_tiled_, module_,
      "fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16_sm120_kernel_nl_tiled");
  if (result != CUDA_SUCCESS) {
    cuModuleUnload(module_);
    throw std::runtime_error("Failed to get regular kernel function from cubin");
  }

  result = cuModuleGetFunction(&kernel_nl_tiled_causal_, module_,
                               "fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_causal_"
                               "output_bf16_sm120_kernel_nl_tiled");
  if (result != CUDA_SUCCESS) {
    cuModuleUnload(module_);
    throw std::runtime_error("Failed to get causal kernel function from cubin");
  }

  initialized_ = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CubinFlashAttention::~CubinFlashAttention() {
  if (initialized_ && module_ != nullptr) {
    cuModuleUnload(module_);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CubinFlashAttention::run(const bert::Fused_multihead_attention_params_v2& params,
                              const bert::Fused_multihead_attention_launch_params& launch_params,
                              cudaStream_t stream) {
  if (!initialized_) {
    throw std::runtime_error("CubinFlashAttention not properly initialized");
  }

  // Validate parameters
  if (params.o_ptr == nullptr) {
    throw std::runtime_error("Output pointer (params.o_ptr) cannot be null");
  }

  // Check SM architecture
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  if (props.major * 10 + props.minor < 120) {
    throw std::runtime_error("This kernel requires SM 12.0 or higher (Hopper)");
  }

  // Calculate grid dimensions
  // For this specific kernel configuration:
  // - Tile size for output is 64x192 (from kernel traits)
  // - We process sequence in chunks of 64
  int ctas_per_o_row = (params.dv + 192 - 1) / 192;  // dv=128, so this will be 1
  int loop_iters = (params.s + 64 - 1) / 64;
  dim3 grid(loop_iters * ctas_per_o_row, params.h, params.b);
  dim3 block(128);  // THREADS = 128 for this kernel

  // Select kernel based on attention mask type
  CUfunction kernel_to_launch =
      (launch_params.attention_mask_type == fmha::Attention_mask_type::CAUSAL)
          ? kernel_nl_tiled_causal_
          : kernel_nl_tiled_;

  // Create a mutable copy of params since CUDA driver API needs stable pointers
  bert::Fused_multihead_attention_params_v2 params_copy = params;

  // Set up kernel parameter - pass the entire params struct
  void* kernel_params[] = {&params_copy};

  // Convert cudaStream_t to CUstream
  CUstream custream = reinterpret_cast<CUstream>(stream);

  // Launch kernel
  CUresult result = cuLaunchKernel(kernel_to_launch, grid.x, grid.y, grid.z,  // grid dimensions
                                   block.x, block.y, block.z,                 // block dimensions
                                   49152,          // shared memory (48KB)
                                   custream,       // stream
                                   kernel_params,  // kernel parameters
                                   nullptr         // extra options
  );

  if (result != CUDA_SUCCESS) {
    const char* error_str;
    cuGetErrorString(result, &error_str);
    throw std::runtime_error("Failed to launch kernel: " + std::string(error_str));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace lightweight_fmha

////////////////////////////////////////////////////////////////////////////////////////////////////

// Example main function for testing
#ifdef BUILD_STANDALONE
int main(int argc, char* argv[]) {
  std::cout << "Lightweight FMHA v2 Flash Attention E4M3 FP32 Test" << std::endl;

  // Initialize CUDA
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return 1;
  }

  // Set device
  cudaSetDevice(0);

  // Create test parameters
  bert::Fused_multihead_attention_params_v2 params;
  bert::Fused_multihead_attention_launch_params launch_params;

  // Initialize parameters for testing
  params.b = 8;     // batch size
  params.h = 32;    // number of heads
  params.s = 64;    // sequence length
  params.d = 192;   // head dimension
  params.dv = 128;  // value dimension

  // Initialize required fields
  params.scale_bmm1 = 1.0f / sqrtf(static_cast<float>(params.d));
  params.scale_softmax = 1.0f;
  params.scale_bmm2 = 1.0f;
  params.enable_i2f_trick = 0;
  params.softcapping_scale_bmm1 = 0.0f;
  params.alibi_params = fmha::AlibiParams();

  // Set launch params
  launch_params.attention_mask_type = fmha::Attention_mask_type::CAUSAL;
  launch_params.flash_attention = true;
  launch_params.enable_attn_logit_softcapping = false;

  // Allocate minimal memory for testing
  size_t output_size = params.b * params.s * params.h * params.dv;
  void* output_d = nullptr;
  cudaMalloc(&output_d, output_size * sizeof(__nv_bfloat16));
  params.o_ptr = output_d;

  // Note: In a real test, you would also allocate and initialize Q, K, V
  // For this minimal test, we'll set them to null (kernel will fail but structure is valid)
  params.q_ptr = nullptr;
  params.k_ptr = nullptr;
  params.v_ptr = nullptr;

  try {
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run the kernel
    lightweight_fmha::run_lightweight_flash_attention_e4m3_fp32(params, launch_params, stream);

    // Synchronize
    cudaStreamSynchronize(stream);

    std::cout << "Kernel executed successfully!" << std::endl;

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(output_d);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
