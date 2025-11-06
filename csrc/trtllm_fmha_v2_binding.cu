/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "tvm_ffi_utils.h"

// Include TRTLLM headers from the copied src directory
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_utils.h>

using tvm::ffi::Optional;

// Declare the generated kernel function
extern void run_fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16_sm120_nl_tiled(
    const bert::Fused_multihead_attention_params_v2 &params,
    const bert::Fused_multihead_attention_launch_params &launch_params,
    cudaStream_t stream);


/**
 * @brief TVM FFI binding for TRTLLM FMHA V2 kernel for MLA attention
 * 
 * This function calls a specific TRTLLM kernel variant with:
 * - Input type: E4M3 (8-bit floating point)
 * - Accumulator type: FP32
 * - Warp configuration: 64x64
 * - Layout: Separate Q, K, V tensors
 * - Q/K dimension: 192, V dimension: 128 (MLA-specific)
 * - Output type: BF16
 * - Target architecture: SM120 (Blackwell)
 * 
 * @param q Query tensor [batch, q_seqlen, num_heads, 192] in E4M3
 * @param k Key tensor [batch, kv_seqlen, num_kv_heads, 192] in E4M3
 * @param v Value tensor [batch, kv_seqlen, num_kv_heads, 128] in E4M3
 * @param o Output tensor [batch, q_seqlen, num_heads, 128] in BF16
 * @param maybe_lse Optional log-sum-exp tensor for softmax statistics
 * @param sm_scale Attention scale factor (typically 1/sqrt(head_dim))
 * @param num_heads Number of query heads
 * @param head_dim Head dimension (must be 192 for this kernel)
 * @param seq_len Sequence length (not used, extracted from tensor shapes)
 */
void TRTLLMFMHAv2Run(TensorView q, TensorView k, TensorView v, TensorView o,
                     Optional<TensorView> maybe_lse, double sm_scale,
                     int64_t num_heads, int64_t head_dim, int64_t seq_len) {
    // Get CUDA stream from TVM runtime
    cudaStream_t stream = static_cast<cudaStream_t>(get_stream(q.device()));
    
    // Extract tensor metadata
    const int batch_size = q.ndim() >= 3 ? q.shape()[0] : 1;
    const int q_seqlen = q.shape()[q.ndim() - 3];
    const int kv_seqlen = k.shape()[k.ndim() - 3];
    const int num_kv_heads = k.shape()[k.ndim() - 2];
    const int d = 192;  // Q/K dimension (from kernel name)
    const int dv = 128; // V dimension (from kernel name)
    
    // Validate dimensions match kernel expectations
    assert(head_dim == d && "Q/K dimension must be 192 for this kernel");
    assert(v.shape()[v.ndim() - 1] == dv && "V dimension must be 128 for this kernel");
    assert(num_heads % num_kv_heads == 0 && "num_heads must be divisible by num_kv_heads");
    
    // Initialize params
    bert::Fused_multihead_attention_params_v2 params;
    memset(&params, 0, sizeof(params));
    
    // Set basic dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_kv = num_kv_heads;
    params.h_q_per_kv = num_heads / num_kv_heads;
    params.s = q_seqlen;
    params.d = d;
    params.dv = dv;
    
    // Set pointers for separate Q, K, V layout
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = o.data_ptr();
    
    // Set strides (assuming contiguous last dimension)
    // E4M3 is 1 byte per element, BF16 is 2 bytes per element
    params.q_stride_in_bytes = num_heads * d * 1; // E4M3
    params.k_stride_in_bytes = num_kv_heads * d * 1; // E4M3
    params.v_stride_in_bytes = num_kv_heads * dv * 1; // E4M3
    params.o_stride_in_bytes = num_heads * dv * 2; // BF16
    
    // Set scale factors
    set_alpha(params.scale_bmm1, static_cast<float>(sm_scale), Data_type::DATA_TYPE_FP32); // FP32
    set_alpha(params.scale_softmax, 1.0f, Data_type::DATA_TYPE_FP32); // FP32
    set_alpha(params.scale_bmm2, 1.0f, Data_type::DATA_TYPE_BF16); // BF16
    
    // Set softmax stats pointer if provided
    if (maybe_lse.has_value()) {
        params.softmax_stats_ptr = maybe_lse.value().data_ptr();
        params.softmax_stats_stride_in_bytes = num_heads * 2 * sizeof(float);
    }
    
    // Initialize other required fields
    params.alibi_params = fmha::AlibiParams();
    params.enable_i2f_trick = 0;
    params.softcapping_scale_bmm1 = 0.0f;
    params.is_s_padded = false;
    params.use_int8_scale_max = false;
    
    // Allocate and set scale_bmm2_d (device pointer for scale)
    uint32_t* scale_bmm2_d;
    cudaMalloc(&scale_bmm2_d, sizeof(uint32_t));
    cudaMemcpyAsync(scale_bmm2_d, &params.scale_bmm2, sizeof(uint32_t), 
                    cudaMemcpyHostToDevice, stream);
    params.scale_bmm2_d = scale_bmm2_d;
    
    // Initialize launch params
    bert::Fused_multihead_attention_launch_params launch_params;
    memset(&launch_params, 0, sizeof(launch_params));
    
    // Set launch configuration
    launch_params.attention_mask_type = fmha::Attention_mask_type::CAUSAL;
    launch_params.attention_input_layout = fmha::Attention_input_layout::SEPARATE_Q_K_V;
    launch_params.flash_attention = true;
    launch_params.warp_specialization = false;  // nl_tiled kernels don't use warp specialization
    launch_params.use_tma = false;  // nl_tiled kernels don't use TMA
    launch_params.use_granular_tiling = true;  // nl_tiled kernels use granular tiling
    launch_params.force_fp32_acc = true;  // E4M3 kernels require FP32 accumulation
    launch_params.total_q_seqlen = q_seqlen * batch_size;
    launch_params.total_kv_seqlen = kv_seqlen * batch_size;
    
    // Call the kernel
    run_fmha_v2_flash_attention_e4m3_fp32_64_64_S_q_k_v_192x128_output_bf16_sm120_nl_tiled(
        params, launch_params, stream);
    
    // Clean up temporary allocation
    cudaFreeAsync(scale_bmm2_d, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, TRTLLMFMHAv2Run);