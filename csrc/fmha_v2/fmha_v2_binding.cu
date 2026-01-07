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

/*
 * FMHAv2 JIT Binding (Static)
 *
 * This is the static TVM FFI binding for FMHAv2 kernels.
 * It includes the JIT-generated dispatcher header which provides
 * the fixed-name wrapper function `fmha_v2_run_kernel()`.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_utils.h>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>

#include "tvm_ffi_utils.h"

// Include the JIT-generated dispatcher header
// This defines fmha_v2_run_kernel() which wraps the actual launcher
#include "fmha_v2_dispatcher.h"

using tvm::ffi::Optional;

using Params = bert::Fused_multihead_attention_params_v2;
using Launch_params = bert::Fused_multihead_attention_launch_params;
using Attention_mask_type = fmha::Attention_mask_type;
using Attention_input_layout = fmha::Attention_input_layout;
using Kv_block_array = fmha::Kv_block_array;

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace flashinfer {

static inline void set_params(Params& params, const Launch_params launch_params,
                              // types
                              Data_type data_type, Data_type acc_type, Data_type output_dtype,
                              // attention input layout
                              Attention_input_layout input_layout,
                              // sizes
                              const size_t b, const size_t s_q, const size_t s_kv, const size_t h,
                              const size_t h_kv, const size_t d, const size_t dv,
                              const size_t total, const size_t num_grouped_heads,
                              const size_t sliding_window_size, const size_t chunked_attention_size,
                              // paged kv cache block size.
                              const size_t tokens_per_block,
                              // device pointers
                              void* qkv_packed_d,
                              // contiguous q.
                              void* q_d,
                              // separate k.
                              void* k_d,
                              // separate v.
                              void* v_d,
                              // contiguous kv.
                              void* kv_d,
                              // start address of the paged kv pool.
                              void* paged_kv_pool_ptr,
                              // offsets for different blocks in terms of the start address.
                              int32_t* paged_block_offsets,
                              // mask input.
                              void* packed_mask_d, void* cu_mask_rows_d,
                              // attention sinks.
                              void* attention_sinks_d, void* cu_kv_seqlens_d, void* cu_q_seqlens_d,
                              void* o_packed_d, void* p_d, void* s_d, void* softmax_stats_d,
                              void* scale_bmm2_d,
                              // scale factors
                              float const scale_bmm1, float const scale_softmax,
                              float const scale_bmm2, float const softcapping_scale_bmm1,
                              // flags
                              bool const use_int8_scale_max, bool const interleaved,
                              bool const is_s_padded, bool const has_alibi) {
  memset(&params, 0, sizeof(params));

  params.o_ptr = o_packed_d;
  params.o_stride_in_bytes = get_size_in_bytes(h * dv, output_dtype);

  if (interleaved) {
    params.q_stride_in_bytes = total;
    params.o_stride_in_bytes = total;
  }

  if (input_layout == Attention_input_layout::PACKED_QKV) {
    params.qkv_ptr = qkv_packed_d;
    params.q_stride_in_bytes = params.k_stride_in_bytes = params.v_stride_in_bytes =
        get_size_in_bytes(h * d + h_kv * d + h_kv * dv, data_type);
  } else {
    params.q_ptr = q_d;
    params.q_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    if (input_layout == Attention_input_layout::CONTIGUOUS_Q_KV) {
      params.kv_ptr = kv_d;
      params.k_stride_in_bytes = params.v_stride_in_bytes =
          get_size_in_bytes(h_kv * (d + dv), data_type);
    } else if (input_layout == Attention_input_layout::Q_PAGED_KV) {
      int max_blocks_per_sequence = (s_kv + tokens_per_block - 1) / tokens_per_block;
      params.paged_kv_cache =
          Kv_block_array(b, max_blocks_per_sequence, tokens_per_block,
                         get_size_in_bytes(tokens_per_block * h_kv * std::gcd(d, dv), data_type),
                         paged_kv_pool_ptr);
      params.paged_kv_cache.mBlockOffsets = paged_block_offsets;
      params.k_stride_in_bytes = get_size_in_bytes(tokens_per_block * d, data_type);
      params.v_stride_in_bytes = get_size_in_bytes(tokens_per_block * dv, data_type);
    } else if (input_layout == Attention_input_layout::SEPARATE_Q_K_V) {
      params.k_ptr = k_d;
      params.v_ptr = v_d;
      params.k_stride_in_bytes = get_size_in_bytes(h_kv * d, data_type);
      params.v_stride_in_bytes = get_size_in_bytes(h_kv * dv, data_type);
    }
  }

  // Packed mask.
  params.packed_mask_ptr = packed_mask_d;
  params.packed_mask_stride_in_bytes =
      (align_to(int64_t(s_kv), int64_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT))) / 8;

  // Attention sinks.
  params.attention_sinks = reinterpret_cast<float*>(attention_sinks_d);

#if defined(STORE_P)
  params.p_ptr = p_d;
  params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif

#if defined(STORE_S)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif

  params.softmax_stats_ptr = softmax_stats_d;
  params.softmax_stats_stride_in_bytes = get_size_in_bytes(h * 2, DATA_TYPE_FP32);

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.s = s_q;
  params.d = d;
  params.dv = dv;
  params.num_grouped_heads = num_grouped_heads;
  params.sliding_window_size = sliding_window_size;
  assert((chunked_attention_size == 0 ||
          (chunked_attention_size & (chunked_attention_size - 1)) == 0) &&
         "chunked_attention_size has to be a power of 2");
  params.log2_chunked_attention_size =
      chunked_attention_size > 0 ? std::log2(chunked_attention_size) : 0;

  // cumulative q or kv sequence lengths.
  params.cu_q_seqlens = static_cast<int*>(cu_q_seqlens_d);
  params.cu_kv_seqlens = static_cast<int*>(cu_kv_seqlens_d);
  // cumulative mask sequence lengths.
  params.cu_mask_rows = static_cast<int*>(cu_mask_rows_d);

  // Set the different scale values.
  Data_type scale_type1 =
      (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? acc_type : DATA_TYPE_FP32;
  Data_type scale_softmax_type = scale_type1;
  Data_type scale_type2 =
      (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? data_type : DATA_TYPE_FP32;
  if (data_type == DATA_TYPE_E4M3) {
    scale_type1 = acc_type;
    scale_type2 = acc_type;
  }

  // Fuse 1.0f / softcapping_scale into scale_bmm1.
  bool const enable_attn_logit_softcapping = softcapping_scale_bmm1 != 0.f;
  float fused_scale_bmm1 =
      enable_attn_logit_softcapping ? scale_bmm1 / softcapping_scale_bmm1 : scale_bmm1;

  // use specialized hopper kernels without alibi support.
  if (launch_params.warp_specialization && !has_alibi && !enable_attn_logit_softcapping) {
    set_alpha(params.scale_bmm1, fused_scale_bmm1 * float(M_LOG2E), DATA_TYPE_FP32);
  } else {
    set_alpha(params.scale_bmm1, fused_scale_bmm1, scale_type1);
  }
  set_alpha(params.scale_softmax, scale_softmax, scale_softmax_type);
  set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);
  params.scale_bmm2_d = reinterpret_cast<uint32_t*>(scale_bmm2_d);
  params.softcapping_scale_bmm1 = softcapping_scale_bmm1;

  FMHA_CHECK_CUDA(cudaMemcpy(params.scale_bmm2_d, &params.scale_bmm2, sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

  // attention type, h_kv < h if MQA or GQA
  params.h_kv = h_kv;
  assert(h % h_kv == 0 && "MQA/GQA needs h to be divisible by h_kv!");
  params.h_q_per_kv = h / h_kv;
  params.has_alibi = has_alibi;
  params.alibi_params = fmha::AlibiParams(h);

  // Set flags
  params.is_s_padded = is_s_padded;
  params.use_int8_scale_max = use_int8_scale_max;

  // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
  if (data_type == DATA_TYPE_INT8) {
    params.enable_i2f_trick = -double(1 << 22) * double(scale_bmm2) <= -128.f &&
                              double(1 << 22) * double(scale_bmm2) >= 127.f;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void determine_launch_params(
    Launch_params& launch_params, Data_type data_type, int sm, const size_t s, const size_t d,
    const Attention_mask_type attention_mask_type, const Attention_input_layout input_layout,
    bool const interleaved, bool const ignore_b1opt, bool const force_unroll, bool const use_tma,
    bool const force_non_flash_attention, bool const force_non_warp_specialization,
    bool const force_non_granular_tiling, bool const force_fp32_acc,
    // device props
    const cudaDeviceProp props) {
  // Set launch params to choose kernels
  launch_params.ignore_b1opt = ignore_b1opt;
  launch_params.force_unroll = force_unroll;
  launch_params.force_fp32_acc = force_fp32_acc;
  launch_params.interleaved = interleaved;
  launch_params.attention_mask_type = attention_mask_type;
  launch_params.attention_input_layout = input_layout;

  // Set SM count and L2 cache size
  launch_params.multi_processor_count = props.multiProcessorCount;
  launch_params.device_l2_cache_size = props.l2CacheSize;

  // threshold for adopting flash attention or warp_specialized kernels.
  launch_params.flash_attention =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      (s >= 16 && d >= 16) && !force_non_flash_attention;

  // enable warp_specialized kernels on hopper
  launch_params.warp_specialization =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      sm == 90 && launch_params.flash_attention && !force_non_warp_specialization;
  launch_params.use_tma = use_tma || launch_params.warp_specialization;

  // use granular tiling on Ampere-style flash attention
  launch_params.use_granular_tiling = !force_non_granular_tiling && launch_params.flash_attention &&
                                      !launch_params.warp_specialization && sm >= 80;

  if (launch_params.use_granular_tiling && (data_type == DATA_TYPE_E4M3 && sm == 80)) {
    printf(
        "Fallback to non-granular-tiling kernels as tiled e4m3 kernels"
        "are not supported on Ada currently.\n");
    launch_params.use_granular_tiling = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief TVM FFI binding for FMHAv2 JIT-compiled kernel
 *
 * This function provides a Python-callable interface for running attention.
 * The actual kernel is determined at JIT compile time and is called via
 * the fmha_v2_run_kernel() wrapper defined in the generated dispatcher.
 *
 * @param q Query tensor [batch, q_seqlen, num_heads, head_dim]
 * @param k Key tensor [batch, kv_seqlen, num_kv_heads, head_dim]
 * @param v Value tensor [batch, kv_seqlen, num_kv_heads, head_dim_v]
 * @param o Output tensor [batch, q_seqlen, num_heads, head_dim_v]
 * @param maybe_lse Optional log-sum-exp tensor for softmax statistics
 * @param num_heads Number of query heads
 * @param head_dim Head dimension for Q and K
 * @param seq_len Sequence length (unused, derived from tensors)
 * @param scale_softmax Softmax scale factor
 * @param scale_bmm1 Scale factor for the first GEMM (Q @ K^T)
 * @param scale_bmm2 Scale factor for the second GEMM (softmax @ V)
 * @param attention_mask_type_code Mask type: 0=PADDING, 1=CAUSAL, 2=SLIDING, 3=CUSTOM
 * @param input_layout_code Input layout: 0=PACKED_QKV, 1=CONTIGUOUS_Q_KV, 2=Q_PAGED_KV, 3=SEPARATE
 * @param data_type_code Data type: 0=FP16, 1=BF16, 2=E4M3
 * @param use_warp_specialization Whether to use warp-specialized kernels (SM90+)
 * @param use_alibi Enable ALiBi position encoding
 * @param softcapping_scale Softcapping scale (0 to disable)
 */
void FMHAv2Run(TensorView q, TensorView k, TensorView v, TensorView o,
               Optional<TensorView> maybe_lse, int64_t num_heads, int64_t head_dim, int64_t seq_len,
               const float scale_softmax, const float scale_bmm1, const float scale_bmm2,
               int64_t attention_mask_type_code, int64_t input_layout_code, int64_t data_type_code,
               bool use_warp_specialization, bool use_alibi, float softcapping_scale) {
  const int batch_size = q.shape()[0];
  const int q_seqlen = q.shape()[1];
  const int kv_seqlen = k.shape()[1];
  assert(num_heads == q.shape()[2] &&
         "num_heads must be equal to the number of heads in the query tensor");
  const int num_kv_heads = k.shape()[2];
  const int head_dim_v = v.shape()[3];

  // Map data type code to Data_type enum
  Data_type data_type;
  Data_type output_dtype;
  switch (data_type_code) {
    case 0:
      data_type = DATA_TYPE_FP16;
      output_dtype = DATA_TYPE_FP16;
      break;
    case 1:
      data_type = DATA_TYPE_BF16;
      output_dtype = DATA_TYPE_BF16;
      break;
    case 2:
      data_type = DATA_TYPE_E4M3;
      output_dtype = DATA_TYPE_BF16;  // FP8 typically outputs BF16
      break;
    default:
      data_type = DATA_TYPE_FP16;
      output_dtype = DATA_TYPE_FP16;
  }
  Data_type acc_type = DATA_TYPE_FP32;

  Attention_mask_type attention_mask_type =
      static_cast<Attention_mask_type>(attention_mask_type_code);
  Attention_input_layout input_layout = static_cast<Attention_input_layout>(input_layout_code);

  CudaDevice device;
  int sm = device.sm;
  cudaDeviceProp props = device.props;

  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(q.device()));

  Launch_params launch_params;
  determine_launch_params(launch_params, data_type, sm, q_seqlen, head_dim, attention_mask_type,
                          input_layout,
                          false,                     // interleaved
                          false,                     // ignore_b1opt
                          false,                     // force_unroll
                          use_warp_specialization,   // use_tma
                          false,                     // force_non_flash_attention
                          !use_warp_specialization,  // force_non_warp_specialization
                          false,                     // force_non_granular_tiling
                          true,                      // force_fp32_acc
                          props);

  launch_params.total_q_seqlen = q_seqlen;
  launch_params.total_kv_seqlen = kv_seqlen;
  launch_params.enable_attn_logit_softcapping = (softcapping_scale != 0.0f);

  // device memory for scale_bmm2
  void* scale_bmm2_d;
  FMHA_CHECK_CUDA(cudaMalloc(&scale_bmm2_d, sizeof(uint32_t)));

  // Cumulative sequence lengths
  std::vector<uint32_t> cu_seqlens(batch_size + 1);
  for (int i = 0; i <= batch_size; i++) {
    cu_seqlens[i] = i * q_seqlen;
  }
  void* cu_seqlens_d;
  FMHA_CHECK_CUDA(cudaMalloc(&cu_seqlens_d, sizeof(uint32_t) * cu_seqlens.size()));
  FMHA_CHECK_CUDA(cudaMemcpy(cu_seqlens_d, cu_seqlens.data(), sizeof(uint32_t) * cu_seqlens.size(),
                             cudaMemcpyHostToDevice));

  if (maybe_lse.has_value()) {
    FMHA_CHECK_CUDA(cudaMemset(maybe_lse.value().data_ptr(), 0,
                               sizeof(float) * batch_size * q_seqlen * num_heads * 2));
  }

  Params params;

  // Determine pointers based on input layout
  void* qkv_packed_ptr = nullptr;
  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* kv_ptr = nullptr;

  if (input_layout == Attention_input_layout::PACKED_QKV) {
    qkv_packed_ptr = q.data_ptr();
  } else if (input_layout == Attention_input_layout::SEPARATE_Q_K_V) {
    q_ptr = q.data_ptr();
    k_ptr = k.data_ptr();
    v_ptr = v.data_ptr();
  } else if (input_layout == Attention_input_layout::CONTIGUOUS_Q_KV) {
    q_ptr = q.data_ptr();
    kv_ptr = k.data_ptr();
  }

  set_params(params, launch_params, data_type, acc_type, output_dtype, input_layout,
             batch_size,         // b
             q_seqlen,           // s_q
             kv_seqlen,          // s_kv
             num_heads,          // h
             num_kv_heads,       // h_kv
             head_dim,           // d
             head_dim_v,         // dv
             cu_seqlens.back(),  // total tokens
             1,                  // num_grouped_heads
             INT_MAX,            // sliding_window_size (disabled)
             0,                  // chunked_attention_size (disabled)
             64,                 // tokens_per_block
             qkv_packed_ptr,     // qkv_packed_d
             q_ptr,              // q_d
             k_ptr,              // k_d
             v_ptr,              // v_d
             kv_ptr,             // kv_d
             nullptr,            // paged_kv_pool_ptr
             nullptr,            // paged_block_offsets
             nullptr,            // packed_mask_d
             nullptr,            // cu_mask_rows_d
             nullptr,            // attention_sinks_d
             cu_seqlens_d,       // cu_kv_seqlens_d
             cu_seqlens_d,       // cu_q_seqlens_d
             o.data_ptr(),       // o_packed_d
             nullptr,            // p_d
             nullptr,            // s_d
             maybe_lse.has_value() ? maybe_lse.value().data_ptr() : nullptr, scale_bmm2_d,
             scale_bmm1,         // scale_bmm1
             scale_softmax,      // scale_softmax
             scale_bmm2,         // scale_bmm2
             softcapping_scale,  // softcapping_scale_bmm1
             false,              // use_int8_scale_max
             false,              // interleaved
             false,              // is_s_padded
             use_alibi);         // has_alibi

  // Call the JIT-generated kernel through the fixed-name wrapper
  fmha_v2_run_kernel(params, launch_params, stream);

  FMHA_CHECK_CUDA(cudaFree(scale_bmm2_d));
  FMHA_CHECK_CUDA(cudaFree(cu_seqlens_d));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::FMHAv2Run);

}  // namespace flashinfer
