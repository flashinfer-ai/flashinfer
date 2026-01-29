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

#include <flashinfer/allocator.h>
#include <fused_multihead_attention.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <numeric>
#include <tuple>
#include <vector>

#include "fmha_v2_api.h"

// #include "fmha_v2_dispatcher.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
namespace ffi = tvm::ffi;

using Launch_params = bert::Fused_multihead_attention_launch_params;
using Attention_mask_type = fmha::Attention_mask_type;
using Attention_input_layout = fmha::Attention_input_layout;
using Kv_block_array = fmha::Kv_block_array;
using AlignedAllocator = flashinfer::AlignedAllocator;

////////////////////////////////////////////////////////////////////////////////////////////////////
// set_params - copied exactly from fused_multihead_attention.cpp
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_v2& params,
                              const Launch_params launch_params,
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
    // For grouped- or multi-query attention (h denotes num_q_heads; h' denotes h_kv):
    //   qkv_layout = [b, s, [q_hd, k_h'd, v_h'd]]
    //   qkv_stride = (h+2*h')d * bytes_per_elt
    // Otherwise:
    //   qkv_layout = [b, s, 3, h, d] or [b, s, h, 3, d]
    //   qkv_stride = 3hd * bytes_per_elt
    params.qkv_ptr = qkv_packed_d;
    params.q_stride_in_bytes = params.k_stride_in_bytes = params.v_stride_in_bytes =
        get_size_in_bytes(h * d + h_kv * d + h_kv * dv, data_type);
  } else {
    // Layout [B, S, H, D].
    params.q_ptr = q_d;
    params.q_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    if (input_layout == Attention_input_layout::CONTIGUOUS_Q_KV) {
      // Layout [B, S, 2, H, D].
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
      // Layout [B, S, H_kv, D].
      params.k_ptr = k_d;
      // Layout [B, S, H_kv, Dv].
      params.v_ptr = v_d;
      params.k_stride_in_bytes = get_size_in_bytes(h_kv * d, data_type);
      params.v_stride_in_bytes = get_size_in_bytes(h_kv * dv, data_type);
    }
  }

  // Packed mask.
  params.packed_mask_ptr = packed_mask_d;
  // The N dimension has to be aligned.
  params.packed_mask_stride_in_bytes =
      (align_to(int64_t(s_kv), int64_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT))) / 8;

  // Attention sinks.
  params.attention_sinks = reinterpret_cast<float*>(attention_sinks_d);

#if defined(STORE_P)
  params.p_ptr = p_d;
  params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif  // defined(STORE_P)

#if defined(STORE_S)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif  // defined(STORE_S)

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
  // alibi or softcapping_scale cannot utilize the exp2f with fused_scale optimization.
  if (launch_params.warp_specialization && !has_alibi && !enable_attn_logit_softcapping) {
    set_alpha(params.scale_bmm1, fused_scale_bmm1 * float(M_LOG2E), DATA_TYPE_FP32);
  } else {
    set_alpha(params.scale_bmm1, fused_scale_bmm1, scale_type1);
  }
  set_alpha(params.scale_softmax, scale_softmax, scale_softmax_type);
  set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);
  // NOTE: scale_bmm2_d is now pre-populated from Python to avoid cudaMemcpy synchronization.
  // The Python side calls create_scale_bmm2_d_tensor() which replicates set_alpha logic.
  params.scale_bmm2_d = reinterpret_cast<uint32_t*>(scale_bmm2_d);
  params.softcapping_scale_bmm1 = softcapping_scale_bmm1;

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
// determine_launch_params - copied exactly from fused_multihead_attention.cpp
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

  // Set SM count and L2 cache size (used to determine launch blocks/grids to maximum performance)
  launch_params.multi_processor_count = props.multiProcessorCount;
  launch_params.device_l2_cache_size = props.l2CacheSize;

  // threshold for adopting flash attention or warp_specialized kernels.
  launch_params.flash_attention =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      (s >= 16 && d >= 16) && !force_non_flash_attention;

  // enable warp_speialized kernels when s >= 512 on hopper
  // note that warp_speialized kernels need flash attention + tma
  launch_params.warp_specialization =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      sm == 90 && launch_params.flash_attention && !force_non_warp_specialization;
  // warp specialization kernels on hopper need tma
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
// Helper function to convert DLDataType to Data_type enum
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline Data_type dltype_to_data_type(DLDataType dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 16) {
    return DATA_TYPE_FP16;
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    return DATA_TYPE_BF16;
  } else if (dtype.code == kDLFloat8_e4m3fn && dtype.bits == 8) {
    return DATA_TYPE_E4M3;
  } else if (dtype.code == kDLFloat && dtype.bits == 32) {
    return DATA_TYPE_FP32;
  } else if (dtype.code == kDLInt && dtype.bits == 8) {
    return DATA_TYPE_INT8;
  }
  assert(false && "Unsupported data type");
  return DATA_TYPE_FP16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main run function - called by fmha_v2_jit_binding.cu
////////////////////////////////////////////////////////////////////////////////////////////////////

void fmha_v2_paged_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView o,
                       ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
                       ffi::TensorView block_tables, int page_size, ffi::TensorView seq_lens,
                       ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
                       Attention_input_layout input_layout, Data_type output_dtype, int max_q_len,
                       int max_kv_len, int batch_size, int64_t mask_mode_code, float scale_softmax,
                       float scale_bmm1, float scale_bmm2, int window_left,
                       int chunked_attention_size, bool has_alibi, float softcapping_scale,
                       Optional<ffi::TensorView> softmax_stats, Optional<ffi::TensorView> sinks) {
  // TODO: Implement paged run
}

void fmha_v2_ragged_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView o,
                        ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
                        ffi::TensorView block_tables, int page_size, ffi::TensorView seq_lens,
                        ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
                        Attention_input_layout input_layout, Data_type output_dtype, int max_q_len,
                        int max_kv_len, int batch_size, int64_t mask_mode_code, float scale_softmax,
                        float scale_bmm1, float scale_bmm2, int window_left,
                        int chunked_attention_size, bool has_alibi, float softcapping_scale,
                        Optional<ffi::TensorView> softmax_stats, Optional<ffi::TensorView> sinks) {
  fmha_v2_paged_run(q, k, v, o, workspace_buffer, workspace_buffer_size_in_bytes, block_tables,
                    page_size, seq_lens, cum_seq_lens_q, cum_seq_lens_kv, input_layout,
                    output_dtype, max_q_len, max_kv_len, batch_size, mask_mode_code, scale_softmax,
                    scale_bmm1, scale_bmm2, window_left, chunked_attention_size, has_alibi,
                    softcapping_scale, softmax_stats, sinks);
}

void fmha_v2_run(
    ffi::TensorView q,  // [batch, s_q, num_heads, head_dim]
    ffi::TensorView k,  // [batch, s_kv, num_kv_heads, head_dim]
    ffi::TensorView v,  // [batch, s_kv, num_kv_heads, head_dim_v]
    ffi::TensorView o,  // [batch, s_q, num_heads, head_dim_v]
    ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
    Optional<ffi::TensorView> maybe_block_tables,  // [batch, num_pages]
    int page_size,
    ffi::TensorView seq_lens,         // [batch]
    ffi::TensorView cum_seq_lens_q,   // [batch + 1]
    ffi::TensorView cum_seq_lens_kv,  // [batch + 1]
    int input_layout_int,             // Cast from int for TVM FFI compatibility
    int output_dtype_int,             // Cast from int for TVM FFI compatibility
    int max_q_len, int max_kv_len, int batch_size, int total_q_tokens,
    int total_kv_tokens,     // Totals from cum_seq_lens (computed in Python)
    int64_t mask_mode_code,  // 0=PADDING, 1=CAUSAL, 2=SLIDING_OR_CHUNKED_CAUSAL, 3=CUSTOM_MASK
    float scale_softmax, float scale_bmm1, float scale_bmm2, int window_left,
    int chunked_attention_size, bool has_alibi, float softcapping_scale,
    ffi::TensorView scale_bmm2_d,             // Pre-populated scale_bmm2 on device [1] int32
    Optional<ffi::TensorView> softmax_stats,  // Optional [batch, s_q, num_heads, 2] for (max, sum)
    Optional<ffi::TensorView> sinks) {
  // Cast int parameters to enum types
  Attention_input_layout input_layout = static_cast<Attention_input_layout>(input_layout_int);
  Data_type output_dtype = static_cast<Data_type>(output_dtype_int);
  // Get device properties
  CudaDevice device;
  int sm = device.sm;
  cudaDeviceProp props = device.props;

  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(q.device()));

  // Extract dimensions based on input_layout:
  // - PACKED_QKV: q is 4D [total_tokens, 3, num_heads, head_dim], k/v are same as q
  // - Q_PAGED_KV: q is 3D [total_tokens, num_heads, head_dim], k/v are 4D paged
  // - SEPARATE_Q_K_V: q/k/v are all 3D [total_tokens, num_heads, head_dim]
  // - CONTIGUOUS_Q_KV: q is 3D, kv packed 3D
  const size_t b = batch_size;
  size_t h, h_kv, d, dv;
  if (input_layout == Attention_input_layout::PACKED_QKV) {
    // q is 4D: [total_tokens, 3, H, D]
    h = q.shape()[2];     // num_heads
    h_kv = q.shape()[2];  // same as h for packed QKV (MHA)
    d = q.shape()[3];     // head_dim_qk
    dv = q.shape()[3];    // head_dim_v (same as d for standard attention)
  } else if (input_layout == Attention_input_layout::Q_PAGED_KV) {
    // q is 3D: [total_tokens, H, D], k/v are 4D paged: [num_pages, H_kv, page_size, D]
    h = q.shape()[1];
    h_kv = k.shape()[1];
    d = q.shape()[2];
    dv = v.shape()[3];
  } else {
    // SEPARATE_Q_K_V or CONTIGUOUS_Q_KV: all 3D ragged [total_tokens, H, D]
    h = q.shape()[1];
    h_kv = k.shape()[1];
    d = q.shape()[2];
    dv = v.shape()[2];
  }

  const size_t s_q = max_q_len;
  const size_t s_kv = max_kv_len;
  const size_t s = s_kv;  // For compatibility with existing code

  // Determine data types from input tensors
  Data_type data_type = dltype_to_data_type(q.dtype());
  Data_type acc_type =
      (data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_FP16) ? DATA_TYPE_FP32 : data_type;

  int tokens_per_block = page_size;
  float softcapping_scale_bmm1 = softcapping_scale;

  // BF16 requires FP32 accumulation, but FP16 kernels use FP16 accumulation.
  // The generated kernel dispatch expects:
  // - FP16 kernels: !force_fp32_acc (force_fp32_acc = false)
  // - BF16 kernels: force_fp32_acc (force_fp32_acc = true)
  bool force_fp32_acc = (data_type == DATA_TYPE_BF16);

  // Determine attention mask type from mask_mode_code
  Attention_mask_type attention_mask_type = static_cast<Attention_mask_type>(mask_mode_code);

  // Sliding window attention parameters
  if (window_left > 0 && window_left < static_cast<int>(s)) {
    assert(chunked_attention_size == 0 &&
           "chunked_attention_size should not be used when sliding_window_size is set");
    attention_mask_type = Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL;
  }
  // Chunked attention.
  if (chunked_attention_size > 0) {
    assert((chunked_attention_size & (chunked_attention_size - 1)) == 0 &&
           "chunked_attention_size has to be a power of 2");
    attention_mask_type = Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL;
  }
  size_t sliding_window_size = size_t(INT_MAX);
  if (attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL) {
    if (window_left != -1) {
      sliding_window_size = size_t(window_left);
    }
  }

  // Total tokens passed from Python (computed as cum_seq_lens[-1].item())
  uint32_t total =
      static_cast<uint32_t>(total_q_tokens);  // Used for stride calculations in interleaved mode

  AlignedAllocator allocator(workspace_buffer.data_ptr(), workspace_buffer_size_in_bytes);

  // Validation for softmax save with MLA
  if (softmax_stats.has_value()) {
    bool is_MLA = (d == 192 && dv == 128);
    if (((!is_MLA) && input_layout != Attention_input_layout::CONTIGUOUS_Q_KV) ||
        (is_MLA && input_layout != Attention_input_layout::SEPARATE_Q_K_V)) {
      fprintf(stderr,
              "For normal attention, only CONTIGUOUS_Q_KV layout supports saving softmax stats. "
              "For MLA only SEPARATE_Q_K_V layout supports saving softmax stats.\n");
      exit(1);
    }
  }

  // Validate different q and kv lengths
  if (s_q != s_kv) {
    assert(input_layout != Attention_input_layout::PACKED_QKV &&
           "Packed QKV input layout is not supported with different q and kv lengths.");
    assert(s_kv >= s_q && "q seqlen has to be smaller than or equal to the kv seqlen!");
  }

  // Set the attention scale (default: 1/sqrt(d))
  if (scale_bmm1 == 0.f) {
    scale_bmm1 = 1.f / sqrtf(static_cast<float>(d));
  }

  // Adjust softmax scale for different data types
  if (data_type == DATA_TYPE_FP16 && scale_softmax == 0.f) {
    scale_softmax = 1.f;
  } else if (data_type == DATA_TYPE_INT8 && scale_softmax == 0.f) {
    scale_softmax = std::max(512.f, static_cast<float>(s));
  } else if (data_type == DATA_TYPE_E4M3 && scale_softmax == 0.f) {
    scale_softmax = 1.f;
  }

  // Enable causal mask if using alibi
  if (has_alibi && attention_mask_type == Attention_mask_type::PADDING) {
    attention_mask_type = Attention_mask_type::CAUSAL;
  }

  // BF16 only supports FP32 accumulation
  if (data_type == DATA_TYPE_BF16 && acc_type != DATA_TYPE_FP32) {
    fprintf(stderr, "Only FP32 accumulation is supported for BF16 I/O\n");
    exit(1);
  }

  // Determine the launch params to select kernels
  Launch_params launch_params;
  determine_launch_params(launch_params, data_type, sm, s, d, attention_mask_type, input_layout,
                          false, false, false, false, false, false, false, force_fp32_acc, props);

  // The decomposition of threads and warps for BMM1.
  size_t warps_m, warps_n, warps_k;
  std::tie(warps_m, warps_n, warps_k) = get_warps(launch_params, sm, data_type, s, b, d, 2);

  // For multi-CTA cases, determine the size of the CTA wave.
  int heads_per_wave, ctas_per_head;
  get_grid_size(heads_per_wave, ctas_per_head, sm, data_type, b, s, h, d,
                false,  // disable multi-cta kernels by default
                2);

  // The number of threads per CTA.
  const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
  // The number of mmas in the M dimension. We use one uint32_t per MMA in the M dimension.
  size_t mmas_m = (s + 16 * warps_m - 1) / (16 * warps_m);
  // The number of mmas in the N dimension.
  size_t mmas_n = (s + 16 * warps_n - 1) / (16 * warps_n);
  // The packed mask for dropout (in the fused kernel). Layout is B * MMAS_M * THREADS_PER_CTA.
  size_t packed_mask_size = b * mmas_m * threads_per_cta;

  // Flash attention on Ampere and Hopper, which supports multiple mmas_n
  if (attention_mask_type == Attention_mask_type::CUSTOM_MASK) {
    // We need to align q and k sequence lengths.
    size_t rounded_q_s = align_to(s, size_t(fmha::FLASH_ATTEN_MASK_M_ALIGNMENT));
    size_t rounded_k_s = align_to(s, size_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT));
    // The number of mmas in the M dimension (MMA_M = 64).
    mmas_m = rounded_q_s / fmha::FLASH_ATTEN_MASK_MMA_M;
    // The number of mmas in the N dimension (MMA_N = 64).
    mmas_n = rounded_k_s / fmha::FLASH_ATTEN_MASK_MMA_N;
    // Each thread holds 32 bit (2 rows, 16 cols -> 8 core MMAs) in one MMA here.
    packed_mask_size = b * mmas_m * mmas_n * threads_per_cta;
  }
  // The size in bytes.
  const size_t packed_mask_size_in_bytes = packed_mask_size * sizeof(uint32_t);

  // Packed mask (allocated conditionally for CUSTOM_MASK)
  void* packed_mask_d =
      (attention_mask_type == Attention_mask_type::CUSTOM_MASK)
          ? allocator.aligned_alloc<void>(packed_mask_size_in_bytes, 128, "packed_mask_d")
          : nullptr;

  // NOTE: scale_bmm2_d is now passed as a pre-populated tensor from Python
  // to avoid cudaMemcpy synchronization in set_params().

  // Softmax stats: stores (max, sum) per token, 2 floats per (b, s_q, h)
  const size_t softmax_stats_size = 2 * sizeof(float) * b * s_q * h;
  void* softmax_stats_d = allocator.aligned_alloc<void>(softmax_stats_size, 128, "softmax_stats_d");
  void* softmax_stats_ptr = softmax_stats.has_value() ? softmax_stats_d : nullptr;
  void* attention_sinks_d = nullptr;

  // Initialize pointers for different input layouts
  void* qkv_packed_d = nullptr;
  void* q_d = nullptr;
  void* k_d = nullptr;
  void* v_d = nullptr;
  void* contiguous_kv_d = nullptr;
  void* kv_cache_pool_ptr = nullptr;
  int32_t* kv_cache_block_offsets_d = nullptr;

  // For Q_PAGED_KV layout, block_tables is pre-expanded on the Python side from [B, M] to [B, 2, M]
  // where [:, 0, :] contains K offsets and [:, 1, :] contains V offsets.
  int block_table_max_blocks = 0;

  switch (input_layout) {
    case Attention_input_layout::PACKED_QKV:
      qkv_packed_d = q.data_ptr();
      break;
    case Attention_input_layout::CONTIGUOUS_Q_KV:
      q_d = q.data_ptr();
      contiguous_kv_d = k.data_ptr();
      break;
    case Attention_input_layout::SEPARATE_Q_K_V:
      q_d = q.data_ptr();
      k_d = k.data_ptr();
      v_d = v.data_ptr();
      break;
    case Attention_input_layout::Q_PAGED_KV: {
      q_d = q.data_ptr();
      kv_cache_pool_ptr = k.data_ptr();

      if (maybe_block_tables.has_value()) {
        // block_tables is pre-expanded on Python side with shape [B, 2, M]
        // where M is max_blocks_per_sequence
        ffi::TensorView block_tables = maybe_block_tables.value();
        block_table_max_blocks = block_tables.shape()[2];  // shape is [B, 2, M]
        kv_cache_block_offsets_d = static_cast<int32_t*>(block_tables.data_ptr());
      }
    } break;
    default:
      assert(false && "Invalid input layout");
      break;
  }

  // TODO: need to add/derive the following variables for set_params:
  // - cu_mask_rows_d           (void*) cumulative mask rows
  // - attention_sinks_d        (void*) attention sinks
  // - cu_seqlens_d             (void*) cumulative kv sequence lengths
  //
  // Also undeclared but used elsewhere:
  // - mqa_qkv_d, qkv_bsh3d_d, mqa_qkv_packed_d, qkv_packed_d (for qkv_d_view)
  // - o_packed_d, o_packed_size (for o_d_view)
  // - save_softmax (bool for softmax_stats_ptr)
  // - q_seqlens, seqlens (for launch_params)

  bert::Fused_multihead_attention_params_v2 params_v2;
  // Print all param set values before calling set_params
  bool debug = true;
  if (debug) {
    printf("=== set_params() arguments ===\n");
    printf("launch_params: ...\n");  // For struct, maybe print pointer or describe
    printf("data_type: %d\n", int(data_type));
    printf("acc_type: %d\n", int(acc_type));
    printf("output_dtype: %d\n", int(output_dtype));
    printf("input_layout: %d\n", int(input_layout));
    printf("b: %zu\n", size_t(b));
    printf("s_q: %zu\n", size_t(s_q));
    printf("s: %zu\n", size_t(s));
    printf("h: %zu\n", size_t(h));
    printf("h_kv: %zu\n", size_t(h_kv));
    printf("d: %zu\n", size_t(d));
    printf("dv: %zu\n", size_t(dv));
    printf("total: %zu\n", size_t(total));
    printf("sliding_window_size: %zu\n", size_t(sliding_window_size));
    printf("chunked_attention_size: %zu\n", size_t(chunked_attention_size));
    printf("tokens_per_block: %zu\n", size_t(tokens_per_block));
    printf("qkv_packed_d: %p\n", qkv_packed_d);
    printf("q_d: %p\n", q_d);
    printf("k_d: %p\n", k_d);
    printf("v_d: %p\n", v_d);
    printf("contiguous_kv_d: %p\n", contiguous_kv_d);
    printf("kv_cache_pool_ptr: %p\n", kv_cache_pool_ptr);
    printf("kv_cache_block_offsets_d: %p\n", kv_cache_block_offsets_d);
    printf("packed_mask_d: %p\n", packed_mask_d);
    printf("attention_sinks_d: %p\n", attention_sinks_d);
    printf("cum_seq_lens_kv: %p\n", cum_seq_lens_kv.data_ptr());
    printf("cum_seq_lens_q: %p\n", cum_seq_lens_q.data_ptr());
    printf("total_q_tokens: %d\n", total_q_tokens);
    printf("total_kv_tokens: %d\n", total_kv_tokens);
    printf("o: %p\n", o.data_ptr());
    printf("softmax_stats_ptr: %p\n", softmax_stats_ptr);
    printf("scale_bmm2_d: %p\n", scale_bmm2_d.data_ptr());
    printf("scale_bmm1: %f\n", scale_bmm1);
    printf("scale_softmax: %f\n", scale_softmax);
    printf("scale_bmm2: %f\n", scale_bmm2);
    printf("softcapping_scale_bmm1: %f\n", softcapping_scale_bmm1);
    printf("has_alibi: %d\n", int(has_alibi));
    printf("=============================\n");
  }

  set_params(params_v2, launch_params, data_type, acc_type, output_dtype, input_layout, b, s_q, s,
             h, h_kv, d, dv, total, 1, sliding_window_size, chunked_attention_size,
             // Paged kv cache.
             tokens_per_block, qkv_packed_d, q_d, k_d, v_d, contiguous_kv_d, kv_cache_pool_ptr,
             kv_cache_block_offsets_d, packed_mask_d, nullptr, attention_sinks_d,
             static_cast<void*>(cum_seq_lens_kv.data_ptr()),
             static_cast<void*>(cum_seq_lens_q.data_ptr()), o.data_ptr(), nullptr, nullptr,
             softmax_stats_ptr, scale_bmm2_d.data_ptr(), scale_bmm1, scale_softmax, scale_bmm2,
             softcapping_scale_bmm1, false, false, false, has_alibi);

  // For Q_PAGED_KV layout, override mMaxBlocksPerSeq to match the actual block_tables stride
  // that we used when expanding the block offsets from [B, M] to [B, 2, M]
  if (input_layout == Attention_input_layout::Q_PAGED_KV && block_table_max_blocks > 0) {
    params_v2.paged_kv_cache.mMaxBlocksPerSeq = block_table_max_blocks;
  }

  // Total number of tokens is needed to set TMA desc on the host.
  launch_params.total_q_seqlen = static_cast<uint32_t>(total_q_tokens);
  launch_params.total_kv_seqlen = static_cast<uint32_t>(total_kv_tokens);
  // set enable_attn_logit_softcapping to select the right kernel.
  launch_params.enable_attn_logit_softcapping = softcapping_scale_bmm1 != 0.f;

  // Compute sizes for conditional allocations
  size_t counters_sz = (ctas_per_head > 1) ? heads_per_wave * sizeof(int) : 0;
  size_t softmax_scratch_sz =
      (ctas_per_head > 1) ? heads_per_wave * ctas_per_head * threads_per_cta * sizeof(float) : 0;
  size_t o_scratch_sz = (ctas_per_head > 1 && data_type != DATA_TYPE_FP16)
                            ? heads_per_wave * threads_per_cta * MAX_STGS_PER_LOOP * sizeof(uint4)
                            : 0;

  // Allocate barriers and locks
  void* counters_d = (counters_sz > 0)
                         ? allocator.aligned_alloc<void>(3 * counters_sz, 16, "counters_d")
                         : nullptr;
  // Allocate scratch storage for softmax
  void* max_scratch_d = (softmax_scratch_sz > 0) ? allocator.aligned_alloc<void>(
                                                       softmax_scratch_sz, 128, "max_scratch_d")
                                                 : nullptr;
  void* sum_scratch_d = (softmax_scratch_sz > 0) ? allocator.aligned_alloc<void>(
                                                       softmax_scratch_sz, 128, "sum_scratch_d")
                                                 : nullptr;
  // Allocate temporary storage for the parallel reduction
  void* o_scratch_d = (o_scratch_sz > 0)
                          ? allocator.aligned_alloc<void>(o_scratch_sz, 128, "o_scratch_d")
                          : nullptr;
  // Allocate tile id for dynamic scheduling
  void* tile_id_counter_d =
      allocator.aligned_alloc<void>(sizeof(uint32_t), 16, "tile_id_counter_d");

  // The number of heads computed per wave.
  params_v2.heads_per_wave = heads_per_wave;

  // Barriers for the global sync in the multi-CTA kernel(s).
  params_v2.counters = (int*)counters_d + 0 * heads_per_wave;
  params_v2.max_barriers = (int*)counters_d + 0 * heads_per_wave;
  params_v2.sum_barriers = (int*)counters_d + 1 * heads_per_wave;
  params_v2.locks = (int*)counters_d + 2 * heads_per_wave;

  // Scratch storage for softmax.
  params_v2.max_scratch_ptr = (float*)max_scratch_d;
  params_v2.sum_scratch_ptr = (float*)sum_scratch_d;

  // Scratch storage for output.
  params_v2.o_scratch_ptr = (int*)o_scratch_d;

  // Tile id counter for dynamic scheduling
  params_v2.tile_id_counter_ptr = (uint32_t*)tile_id_counter_d;

  // V2 Custom Mask Packing (only if using CUSTOM_MASK)
  // Note: You need to populate packed_mask_d with your custom mask data here
  // using pack_flash_attention_mask() or provide pre-packed mask

  // Run the V2 kernel with runtime dispatch based on dtype and head dimensions
  run_fmha_v2(params_v2, launch_params, data_type, output_dtype, sm, stream);
}
