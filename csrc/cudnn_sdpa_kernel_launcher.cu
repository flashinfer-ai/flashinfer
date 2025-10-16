/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>

#include "cudnn_sdpa_utils.h"
#include "tvm_ffi_utils.h"

#ifdef CUDNN_SDPA_CUBIN_PATH
static const std::string cudnn_sdpa_cubin_path = std::string(CUDNN_SDPA_CUBIN_PATH);
#else
static_assert(false, "CUDNN_SDPA_CUBIN_PATH macro is not defined when compiling");
#endif

namespace flashinfer {

namespace cudnn_sdpa_kernel_launcher {

#include <flashinfer/cubin_loader.h>

using tvm::ffi::Optional;

inline __host__ int clz(int x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x) {
      return 31 - i;
    }
  }
  return 32;
}

inline __host__ int find_log_2(int x, bool round_up = false) {
  int a = 31 - clz(x);
  if (round_up) {
    a += (x & (x - 1)) ? 1 : 0;
  }
  return a;
}

inline __host__ void setFastDivisor(cudnn_sdpa::FastDivisor_t& d, uint32_t val) {
  uint32_t p = 31 + find_log_2(2 * val, true);
  uint32_t m = (uint32_t)(((1ull << p) + (uint32_t)(2 * val) - 1) / (uint32_t)(2 * val));

  d.val = val;
  d.mul = m;
  d.shr = p - 32;
}

static std::once_flag init_cudnn_cubin_flag;

constexpr size_t DIMS_QKV = 4;
constexpr int32_t BYTES_PER_ELEMENT = 2;

enum KernelType { PREFILL, PREFILL_DEEPSEEK, DECODE };

enum PrefillType {
  KERNEL_PREFILL,
  KERNEL_PREFILL_DEEPSEEK,
  KERNEL_PREFILL_CAUSAL,
  KERNEL_PREFILL_DEEPSEEK_CAUSAL,
  KERNEL_NUM_PREFILL_TYPES
};

void init_cudnn_cubin(std::map<KernelType, std::string>& cubin_map) {
  cubin_map[PREFILL] =
      getCubin(cudnn_sdpa_cubin_path + "/" + "cudnn_sm100_fprop_sdpa_prefill_d128_bf16.cubin",
               "ff14e8dcfc04d9b3a912dd44056be37d9aa8a85976e0070494ca0cce0524f2a1");

  cubin_map[DECODE] =
      getCubin(cudnn_sdpa_cubin_path + "/" + "cudnn_sm100_fprop_sdpa_decode_d128_bf16.cubin",
               "e7ce0408b4c3a36c42616498228534ee64cab785ef570af5741deaf9dd1b475c");

  cubin_map[PREFILL_DEEPSEEK] =
      getCubin(cudnn_sdpa_cubin_path + "/" + "cudnn_sm100_fprop_sdpa_prefill_d192_bf16.cubin",
               "2190967b8733e193cdcecc054eeb7c2907080a158a33fe7ba2004523a4aff6f9");
}

auto get_cudnn_cubin(KernelType kernel_type) -> std::string {
  static std::map<KernelType, std::string> cubin_map;
  std::call_once(init_cudnn_cubin_flag, init_cudnn_cubin, std::ref(cubin_map));
  return cubin_map[kernel_type];
}

__global__ static void __launch_bounds__(128)
    qkv_tma_setup_decode(const unsigned int b, const unsigned int h_qo, const unsigned int h_kv,
                         const unsigned int d, const unsigned int total_num_pages,
                         const unsigned int page_size, const unsigned int split_factor,
                         const unsigned int tile_m_1, const unsigned int tile_n_1,
                         const unsigned int kv_strides_2, const unsigned int kv_strides_1,
                         const unsigned int kv_strides_0, void* q_ptr, const void* k_ptr,
                         const void* v_ptr, void* o_ptr, void* partial_o_ptr,
                         tma::cudaTmaDesc* tma_desc_q_array, tma::cudaTmaDesc* tma_desc_k,
                         tma::cudaTmaDesc* tma_desc_v, tma::cudaTmaDesc* tma_desc_o_array,
                         tma::cudaTmaDesc* tma_desc_partial_o_array, int64_t* batch_strides_dev) {
  const int tid = threadIdx.x;

  constexpr unsigned int DIMS_QKV = 4;
  constexpr unsigned int BYTES_PER_ELEMENT = 2;

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_qo = {64, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_kv = {64, std::min(tile_n_1, page_size), 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_partial_o = {32, 1, 1, 1};

  std::array<uint32_t, DIMS_QKV> tensor_size_qo = {d, 1 /* s_qo */, h_qo, b};
  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};

  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                         d * BYTES_PER_ELEMENT, 0};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides_2 * (BYTES_PER_ELEMENT),
                                                         kv_strides_1 * (BYTES_PER_ELEMENT),
                                                         kv_strides_0 * (BYTES_PER_ELEMENT)};

  std::array<uint32_t, DIMS_QKV> tensor_size_partial_o = {d, split_factor, h_qo, b};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_partial_o = {
      h_qo * d * b * sizeof(float), d * b * sizeof(float), d * h_qo * sizeof(float)};

  tma::cudaSetTmaTileDescriptor(
      reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_k), k_ptr, DIMS_QKV, tensor_size_kv.data(),
      tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_kv.data(),
      tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  tma::cudaSetTmaTileDescriptor(
      reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_v), v_ptr, DIMS_QKV, tensor_size_kv.data(),
      tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_kv.data(),
      tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  int64_t batch_offset_qo = 0;
  int64_t batch_offset_partial_o = 0;
#pragma unroll 1
  for (int i = 0; i < b; ++i) {
    batch_strides_dev[i] = batch_offset_qo;
    uint16_t* per_batch_q_ptr =
        reinterpret_cast<uint16_t*>(static_cast<std::byte*>(q_ptr) + batch_offset_qo);
    uint16_t* per_batch_out_ptr =
        reinterpret_cast<uint16_t*>(static_cast<std::byte*>(o_ptr) + batch_offset_qo);
    // The two below comes from half
    float* per_batch_partial_o_ptr =
        reinterpret_cast<float*>(static_cast<std::byte*>(partial_o_ptr) + (batch_offset_partial_o));

    tma::cudaTmaDesc desc_q;
    tma::cudaTmaDesc desc_o;
    tma::cudaTmaDesc desc_partial_o;

    tma::cudaSetTmaTileDescriptor(&desc_q, (void*)per_batch_q_ptr, DIMS_QKV, tensor_size_qo.data(),
                                  tensor_stride_qo.data(), tensor_traversal_stride_qkv.data(),
                                  tensor_box_size_qo.data(), tma::cudaTmaDescFormat::BF16_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &desc_o, (void*)per_batch_out_ptr, DIMS_QKV, tensor_size_qo.data(), tensor_stride_qo.data(),
        tensor_traversal_stride_qkv.data(), tensor_box_size_qo.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(&desc_partial_o, (void*)per_batch_partial_o_ptr, DIMS_QKV,
                                  tensor_size_partial_o.data(), tensor_stride_partial_o.data(),
                                  tensor_traversal_stride_qkv.data(),
                                  tensor_box_size_partial_o.data(), tma::cudaTmaDescFormat::F32_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_q_array)[i] = desc_q;
    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_o_array)[i] = desc_o;
    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_partial_o_array)[i] = desc_partial_o;

    batch_offset_qo += d * h_qo * BYTES_PER_ELEMENT;
    batch_offset_partial_o += d * h_qo * sizeof(float);
  }
}

__global__ static void __launch_bounds__(128)
    qkv_tma_setup_prefill(const unsigned int b, const unsigned int h_qo, const unsigned int h_kv,
                          const unsigned int d_qk, const unsigned int d_vo, const bool is_kv_ragged,
                          const unsigned int page_size, const unsigned int total_num_pages,

                          const int64_t k_strides_2, const int64_t k_strides_1,
                          const int64_t k_strides_0, const int64_t v_strides_2,
                          const int64_t v_strides_1, const int64_t v_strides_0,

                          int32_t* actual_seq_lens_q_data, int32_t* actual_seq_lens_kv_data,

                          void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr,

                          tma::cudaTmaDesc* tma_desc_q_array, tma::cudaTmaDesc* tma_desc_k,
                          tma::cudaTmaDesc* tma_desc_v, tma::cudaTmaDesc* tma_desc_o_array
                          /* const int64_t *batch_offset_array */) {
  const int tid = threadIdx.x;

  constexpr unsigned int DIMS_QKV = 4;
  constexpr unsigned int TILE_M_1 = 128;
  constexpr unsigned int TILE_N_1 = 128;
  constexpr unsigned int BYTES_PER_ELEMENT = 2;
  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};

  if (is_kv_ragged) {
    int64_t batch_offset_k = 0;
    int64_t batch_offset_v = 0;
    std::array<uint32_t, DIMS_QKV> tensor_box_size_kv = {64, TILE_N_1, 1, 1};

#pragma unroll 1
    for (int i = 0; i < b; ++i) {
      const uint32_t actual_s_kv = static_cast<uint32_t>(actual_seq_lens_kv_data[i]);
      std::array<uint32_t, DIMS_QKV> packed_tensor_size_k = {d_qk, actual_s_kv, h_kv, 1};
      std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_k = {h_kv * d_qk * BYTES_PER_ELEMENT,
                                                                   d_qk * BYTES_PER_ELEMENT, 0};
      std::array<uint32_t, DIMS_QKV> packed_tensor_size_v = {d_vo, actual_s_kv, h_kv, 1};
      std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_v = {h_kv * d_vo * BYTES_PER_ELEMENT,
                                                                   d_vo * BYTES_PER_ELEMENT, 0};

      uint16_t* k_batch_ptr =
          reinterpret_cast<uint16_t*>(reinterpret_cast<std::byte*>(k_ptr) + batch_offset_k);
      uint16_t* v_batch_ptr =
          reinterpret_cast<uint16_t*>(reinterpret_cast<std::byte*>(v_ptr) + batch_offset_v);

      tma::cudaSetTmaTileDescriptor(&tma_desc_k[i], (void*)k_batch_ptr, DIMS_QKV,
                                    packed_tensor_size_k.data(), packed_tensor_stride_k.data(),
                                    tensor_traversal_stride_qkv.data(), tensor_box_size_kv.data(),
                                    tma::cudaTmaDescFormat::BF16_RN,
                                    tma::cudaTmaDescSwizzle::SWIZZLE_128B);

      tma::cudaSetTmaTileDescriptor(&tma_desc_v[i], (void*)v_batch_ptr, DIMS_QKV,
                                    packed_tensor_size_v.data(), packed_tensor_stride_v.data(),
                                    tensor_traversal_stride_qkv.data(), tensor_box_size_kv.data(),
                                    tma::cudaTmaDescFormat::BF16_RN,
                                    tma::cudaTmaDescSwizzle::SWIZZLE_128B);

      batch_offset_k += static_cast<int64_t>(actual_s_kv) * d_qk * h_kv *
                        BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
      batch_offset_v += static_cast<int64_t>(actual_s_kv) * d_vo * h_kv *
                        BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
    }
  } else {
    bool kv_cache_enabled = d_qk == 192 ? false : true;

    std::array<uint32_t, DIMS_QKV> tensor_size_k = {d_qk, page_size, h_kv, total_num_pages};
    std::array<uint64_t, DIMS_QKV - 1> tensor_stride_k = {k_strides_2 * (BYTES_PER_ELEMENT),
                                                          k_strides_1 * (BYTES_PER_ELEMENT),
                                                          k_strides_0 * (BYTES_PER_ELEMENT)};
    std::array<uint32_t, DIMS_QKV> tensor_size_v = {d_vo, page_size, h_kv, total_num_pages};
    std::array<uint64_t, DIMS_QKV - 1> tensor_stride_v = {v_strides_2 * (BYTES_PER_ELEMENT),
                                                          v_strides_1 * (BYTES_PER_ELEMENT),
                                                          v_strides_0 * (BYTES_PER_ELEMENT)};
    std::array<uint32_t, DIMS_QKV> tensor_box_size_k = {
        64, kv_cache_enabled ? std::min(TILE_N_1, page_size) : TILE_N_1, 1, 1};

    tma::cudaSetTmaTileDescriptor(
        reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_k), k_ptr, DIMS_QKV, tensor_size_k.data(),
        tensor_stride_k.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_v), v_ptr, DIMS_QKV, tensor_size_v.data(),
        tensor_stride_v.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
  }

  int64_t batch_offset_q = 0;
  int64_t batch_offset_k = 0;
  int64_t batch_offset_v = 0;
  int64_t batch_offset_o = 0;
  std::array<uint32_t, DIMS_QKV> tensor_box_size_q = {64, TILE_M_1, 1, 1};

#pragma unroll 1
  for (int i = 0; i < b; ++i) {
    const uint32_t actual_s_q = static_cast<uint32_t>(actual_seq_lens_q_data[i]);

    // batch_offset_qo = batch_offset_array ? batch_offset_array[i] : batch_offset_qo;
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_q = {d_qk, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_q = {h_qo * d_qk * BYTES_PER_ELEMENT,
                                                                 d_qk * BYTES_PER_ELEMENT, 0};
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_o = {d_vo, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_o = {h_qo * d_vo * BYTES_PER_ELEMENT,
                                                                 d_vo * BYTES_PER_ELEMENT, 0};

    uint16_t* per_batch_q_ptr =
        reinterpret_cast<uint16_t*>(reinterpret_cast<std::byte*>(q_ptr) + batch_offset_q);
    uint16_t* per_batch_out_ptr =
        reinterpret_cast<uint16_t*>(reinterpret_cast<std::byte*>(o_ptr) + batch_offset_o);

    tma::cudaTmaDesc desc_q;
    tma::cudaTmaDesc desc_o;

    tma::cudaSetTmaTileDescriptor(
        &desc_q, (void*)per_batch_q_ptr, DIMS_QKV, packed_tensor_size_q.data(),
        packed_tensor_stride_q.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &desc_o, (void*)per_batch_out_ptr, DIMS_QKV, packed_tensor_size_o.data(),
        packed_tensor_stride_o.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_q_array)[i] = desc_q;
    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_o_array)[i] = desc_o;

    batch_offset_q += static_cast<int64_t>(actual_s_q) * d_qk * h_qo *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
    batch_offset_o += static_cast<int64_t>(actual_s_q) * d_vo * h_qo *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
  }
}

static void create_packed_tma_desc_kv_prefill(int b, int32_t* actual_seq_lens_kv_data, int64_t d_qk,
                                              int64_t d_vo, int64_t h_kv,
                                              uint32_t* tensor_traversal_stride_qkv,
                                              uint32_t* tensor_box_size_kv,
                                              tma::cudaTmaDesc* packed_tma_desc_k,
                                              tma::cudaTmaDesc* packed_tma_desc_v, TensorView k,
                                              TensorView v) {
  int64_t batch_offset_k = 0;
  int64_t batch_offset_v = 0;
  // tma descriptors for packed q and o
  for (int i = 0; i < b; ++i) {
    const uint32_t actual_s_kv = static_cast<uint32_t>(actual_seq_lens_kv_data[i]);
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_k = {d_qk, actual_s_kv, h_kv, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_k = {h_kv * d_qk * BYTES_PER_ELEMENT,
                                                                 d_qk * BYTES_PER_ELEMENT, 0};
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_v = {d_vo, actual_s_kv, h_kv, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_v = {h_kv * d_vo * BYTES_PER_ELEMENT,
                                                                 d_vo * BYTES_PER_ELEMENT, 0};

    uint16_t* k_ptr = reinterpret_cast<uint16_t*>(k.data_ptr() + batch_offset_k);
    uint16_t* v_ptr = reinterpret_cast<uint16_t*>(v.data_ptr() + batch_offset_v);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_k[i], (void*)k_ptr, DIMS_QKV, packed_tensor_size_k.data(),
        packed_tensor_stride_k.data(), tensor_traversal_stride_qkv, tensor_box_size_kv,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_v[i], (void*)v_ptr, DIMS_QKV, packed_tensor_size_v.data(),
        packed_tensor_stride_v.data(), tensor_traversal_stride_qkv, tensor_box_size_kv,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    batch_offset_k += static_cast<int64_t>(actual_s_kv) * d_qk * h_kv *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
    batch_offset_v += static_cast<int64_t>(actual_s_kv) * d_vo * h_kv *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
  }
}

static void create_packed_tma_desc_qo_prefill(int b, int32_t* actual_seq_lens_q_data, int64_t d_qk,
                                              int64_t d_vo, int64_t h_qo,
                                              uint32_t* tensor_traversal_stride_qkv,
                                              uint32_t* tensor_box_size_q,
                                              tma::cudaTmaDesc* packed_tma_desc_q,
                                              tma::cudaTmaDesc* packed_tma_desc_o, TensorView q,
                                              TensorView out, int64_t* batch_offset_array) {
  int64_t batch_offset_q = 0;
  int64_t batch_offset_o = 0;
  // tma descriptors for packed q and o
  for (int i = 0; i < b; ++i) {
    const uint32_t actual_s_q = static_cast<uint32_t>(actual_seq_lens_q_data[i]);

    batch_offset_q = batch_offset_array ? batch_offset_array[i] : batch_offset_q;
    batch_offset_o = batch_offset_array ? batch_offset_array[i] : batch_offset_o;
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_q = {d_qk, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_q = {h_qo * d_qk * BYTES_PER_ELEMENT,
                                                                 d_qk * BYTES_PER_ELEMENT, 0};
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_o = {d_vo, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_o = {h_qo * d_vo * BYTES_PER_ELEMENT,
                                                                 d_vo * BYTES_PER_ELEMENT, 0};

    uint16_t* q_ptr = reinterpret_cast<uint16_t*>(q.data_ptr() + batch_offset_q);
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr() + batch_offset_o);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_q[i], (void*)q_ptr, DIMS_QKV, packed_tensor_size_q.data(),
        packed_tensor_stride_q.data(), tensor_traversal_stride_qkv, tensor_box_size_q,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_o[i], (void*)out_ptr, DIMS_QKV, packed_tensor_size_o.data(),
        packed_tensor_stride_o.data(), tensor_traversal_stride_qkv, tensor_box_size_q,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    batch_offset_q += static_cast<int64_t>(actual_s_q) * d_qk * h_qo *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
    batch_offset_o += static_cast<int64_t>(actual_s_q) * d_vo * h_qo *
                      BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
  }
}

void setup_prefill(CUfunction* prefill_func) {
  // Use cu++filt to get the kernel name
  std::string kernel_name_deepseek_causal =
      "_Z47cudnn_sm100_fprop_sdpa_prefill_bf16_"
      "128x128x192ILb1ELb0EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_fPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE";

  std::string kernel_name_causal =
      "_Z47cudnn_sm100_fprop_sdpa_prefill_bf16_"
      "128x128x128ILb1ELb1EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_fPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE";

  std::string kernel_name_deepseek =
      "_Z47cudnn_sm100_fprop_sdpa_prefill_bf16_"
      "128x128x192ILb0ELb0EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_fPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE";

  std::string kernel_name =
      "_Z47cudnn_sm100_fprop_sdpa_prefill_bf16_"
      "128x128x128ILb0ELb1EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_fPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE";

  std::string cubin = get_cudnn_cubin(PREFILL);
  std::string cubin_deepseek = get_cudnn_cubin(PREFILL_DEEPSEEK);

  if (cubin.empty()) {
    throw std::runtime_error("Failed to load cubin for prefill");
  }
  if (cubin_deepseek.empty()) {
    throw std::runtime_error("Failed to load cubin for prefill_deepseek");
  }

  CUmodule hmod{0};
  CUmodule hmod_deepseek{0};
  if (cuModuleLoadData(&hmod_deepseek, cubin_deepseek.data()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleLoadData for prefill_deepseek");
  }

  if (cuModuleLoadData(&hmod, cubin.data()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleLoadData for prefill");
  }

  if (cuModuleGetFunction(&prefill_func[KERNEL_PREFILL], hmod, kernel_name.c_str()) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for prefill");
  }

  if (cuModuleGetFunction(&prefill_func[KERNEL_PREFILL_DEEPSEEK], hmod_deepseek,
                          kernel_name_deepseek.c_str()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for prefill_deepseek");
  }

  if (cuModuleGetFunction(&prefill_func[KERNEL_PREFILL_CAUSAL], hmod, kernel_name_causal.c_str()) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for prefill");
  }

  if (cuModuleGetFunction(&prefill_func[KERNEL_PREFILL_DEEPSEEK_CAUSAL], hmod_deepseek,
                          kernel_name_deepseek_causal.c_str()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for prefill_deepseek");
  }
};

void setup_decode(CUfunction* hfunc_decode, CUfunction* lean_attn_reduction) {
  constexpr int NUM_DECODE_KERNELS = 5;

  std::string decode_kernel_name[NUM_DECODE_KERNELS] = {
      "_Z44cudnn_sm100_fprop_sdpa_decode_bf16_"
      "Mx128x128ILb1ELi1EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_ifPfNS0_7stridesES5_"
      "S5_PKjS9_S9_jjNS0_11FastDivisorE",
      "_Z44cudnn_sm100_fprop_sdpa_decode_bf16_"
      "Mx128x128ILb1ELi8EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_ifPfNS0_7stridesES5_"
      "S5_PKjS9_S9_jjNS0_11FastDivisorE",
      "_Z44cudnn_sm100_fprop_sdpa_decode_bf16_"
      "Mx128x128ILb1ELi16EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_ifPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE",
      "_Z44cudnn_sm100_fprop_sdpa_decode_bf16_"
      "Mx128x128ILb1ELi32EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_ifPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE",
      "_Z44cudnn_sm100_fprop_sdpa_decode_bf16_"
      "Mx128x128ILb1ELi64EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_ifPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE",
  };

  std::string lean_attn_reduction_kernel_name =
      "_Z19lean_attn_reductionN4fmha19AttentionDescriptorEiP13__nv_bfloat16PfS3_S3_NS_7stridesES4_"
      "S4_S4_Pl";

  std::string cubin = get_cudnn_cubin(DECODE);
  if (cubin.empty()) {
    throw std::runtime_error("Failed to load cubin for decode");
  }

  CUmodule hmod{0};
  if (cuModuleLoadData(&hmod, cubin.data()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleLoadData for decode");
  }

  for (int i = 0; i < NUM_DECODE_KERNELS; i++) {
    if (cuModuleGetFunction(&hfunc_decode[i], hmod, decode_kernel_name[i].c_str()) !=
        CUDA_SUCCESS) {
      throw std::runtime_error("Failed to cuModuleGetFunction for decode at location " +
                               std::to_string(i) + " " + decode_kernel_name[i]);
    }
  }
  if (cuModuleGetFunction(lean_attn_reduction, hmod, lean_attn_reduction_kernel_name.c_str()) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for lean_attn_reduction decode");
  }
};

void prefill(int64_t b, int64_t s_qo, int64_t max_s_kv, TensorView q, TensorView k_cache,
             TensorView v_cache, double scale, TensorView workspace_buffer,
             TensorView actual_seq_lens_q, TensorView actual_seq_lens_kv,
             TensorView actual_seq_lens_q_gpu, TensorView actual_seq_lens_kv_gpu,
             TensorView block_tables, bool causal, bool return_lse, TensorView out, TensorView lse,
             Optional<TensorView> batch_offset_q_array, Optional<TensorView> batch_offset_o_array,
             Optional<TensorView> batch_offset_k_array, Optional<TensorView> batch_offset_v_array,
             bool is_cuda_graph_compatible) {
  constexpr size_t SMEM_SIZE = 227 * 1024;  // All smem
  constexpr int64_t TILE_M_1 = 128;
  constexpr int64_t TILE_N_1 = 128;

  constexpr int32_t NUM_THREADS = 512;

  const CUstream stream = get_stream(q.device());

  int64_t* batch_offset_q_array_data = nullptr;
  int64_t* batch_offset_o_array_data = nullptr;
  int64_t* batch_offset_k_array_data = nullptr;
  int64_t* batch_offset_v_array_data = nullptr;
  int64_t* batch_offset_array_data = nullptr;
  if (batch_offset_q_array.has_value()) {
    batch_offset_array_data = static_cast<int64_t*>(
        batch_offset_q_array.value().data_ptr());  // Fix this to make it operational later
  }

  // Step 1: Setup the kernel pointer

  static CUfunction prefill_func[KERNEL_NUM_PREFILL_TYPES] = {nullptr, nullptr, nullptr, nullptr};

  int64_t d_qk = q.size(2);

  int64_t d_vo = v_cache.ndim() == 3 ? v_cache.size(2) : v_cache.size(3);

  if (prefill_func[0] == nullptr) {
    setup_prefill(prefill_func);

    for (int i = 0; i < KERNEL_NUM_PREFILL_TYPES; i++) {
      if (prefill_func[i] != nullptr) {
        cuErrCheck(cuFuncSetAttribute(prefill_func[i],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_SIZE));
        cuErrCheck(cuFuncSetAttribute(prefill_func[i],
                                      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
        cuErrCheck(cuFuncSetAttribute(prefill_func[i],
                                      CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
      }
    }
  }

  // Step 2: Extract attention descriptor

  int64_t h_qo = q.size(1);

  int64_t h_kv = k_cache.size(1);

  int64_t page_size = k_cache.ndim() == 4 ? k_cache.size(2) : 1;

  int64_t s_kv = max_s_kv;

  int64_t num_pages_per_seq = static_cast<int64_t>(std::ceil(1.0 * s_kv / page_size));

  int64_t total_num_pages = k_cache.ndim() == 4 ? k_cache.size(0) : 1;

  bool kv_cache_enabled = d_qk == 192 ? false : true;

  // Step 3: Setup the launch configuration

  CUlaunchConfig config;

  constexpr int NUM_ATTRS = 1;
  CUlaunchAttribute attrs[NUM_ATTRS];
  config.numAttrs = NUM_ATTRS;
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[0].value.clusterDim.x = 1;
  attrs[0].value.clusterDim.y = 1;
  attrs[0].value.clusterDim.z = 1;
  config.attrs = attrs;

  config.sharedMemBytes = SMEM_SIZE;
  config.hStream = stream;

  if (is_cuda_graph_compatible == false) {
    CHECK_CPU(actual_seq_lens_q);
    CHECK_CPU(actual_seq_lens_kv);
    auto actual_seq_lens_q_data = static_cast<int32_t*>(actual_seq_lens_q.data_ptr());
    auto actual_seq_lens_kv_data = static_cast<int32_t*>(actual_seq_lens_kv.data_ptr());

    uint32_t actual_num_tiles_per_head = std::transform_reduce(
        actual_seq_lens_q_data, actual_seq_lens_q_data + b, 0U, std::plus<>(), [](int32_t seq_len) {
          return static_cast<uint32_t>(std::ceil(seq_len / (TILE_M_1 * 2.0f)));
        });
    config.gridDimX = actual_num_tiles_per_head;

  } else {
    config.gridDimX = static_cast<int>(std::ceil(s_qo / (TILE_M_1 * 2.0f))) * b;
  }

  config.gridDimY = h_qo;
  config.gridDimZ = 1;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  // Step 4: Set up the launch arguments

  auto k_strides = k_cache.strides();
  auto v_strides = v_cache.strides();

  bool is_kv_ragged = k_cache.ndim() == 3;

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_size_k = {d_qk, page_size, h_kv, total_num_pages};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_k = {k_strides[2] * (BYTES_PER_ELEMENT),
                                                        k_strides[1] * (BYTES_PER_ELEMENT),
                                                        k_strides[0] * (BYTES_PER_ELEMENT)};
  std::array<uint32_t, DIMS_QKV> tensor_size_v = {d_vo, page_size, h_kv, total_num_pages};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_v = {v_strides[2] * (BYTES_PER_ELEMENT),
                                                        v_strides[1] * (BYTES_PER_ELEMENT),
                                                        v_strides[0] * (BYTES_PER_ELEMENT)};

  std::array<uint32_t, DIMS_QKV> tensor_box_size_q = {64, TILE_M_1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_k = {
      64, kv_cache_enabled ? std::min(TILE_N_1, page_size) : TILE_N_1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_v = {
      64, kv_cache_enabled ? std::min(TILE_N_1, page_size) : TILE_N_1, 1, 1};

  uint64_t batch_offset_qo = 0;
  int8_t* workspace_start = static_cast<int8_t*>(workspace_buffer.data_ptr());

  // These tensors are allocated in the workspace buffer
  // Using 2 * b for q and o
  std::unique_ptr<tma::cudaTmaDesc[]> packed_tma_desc(new tma::cudaTmaDesc[(4 * b)]);
  auto packed_tma_desc_q = packed_tma_desc.get();
  auto packed_tma_desc_o = packed_tma_desc.get() + b;
  auto tma_desc_k_host = packed_tma_desc.get() + (2 * b);
  auto tma_desc_v_host = packed_tma_desc.get() + (3 * b);

  tma::cudaTmaDesc* packed_tma_desc_q_dev = reinterpret_cast<tma::cudaTmaDesc*>(workspace_start);
  tma::cudaTmaDesc* packed_tma_desc_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * b);

  // These TMA descriptors are allocated in the host and passed by value
  tma::cudaTmaDesc* tma_desc_k =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * (2 * b));
  tma::cudaTmaDesc* tma_desc_v =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * (3 * b));

  if (is_cuda_graph_compatible == false) {
    if (is_kv_ragged) {
      auto actual_seq_lens_kv_data = static_cast<int32_t*>(actual_seq_lens_kv.data_ptr());
      create_packed_tma_desc_kv_prefill(
          b, actual_seq_lens_kv_data, d_qk, d_vo, h_kv, tensor_traversal_stride_qkv.data(),
          tensor_box_size_k.data(), tma_desc_k_host, tma_desc_v_host, k_cache, v_cache);
    } else {
      // tma descriptors for k and v
      tma::cudaSetTmaTileDescriptor(
          tma_desc_k_host, k_cache.data_ptr(), DIMS_QKV, tensor_size_k.data(),
          tensor_stride_k.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
          tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

      tma::cudaSetTmaTileDescriptor(
          tma_desc_v_host, v_cache.data_ptr(), DIMS_QKV, tensor_size_v.data(),
          tensor_stride_v.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_v.data(),
          tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    }
    auto actual_seq_lens_q_data = static_cast<int32_t*>(actual_seq_lens_q.data_ptr());
    create_packed_tma_desc_qo_prefill(b, actual_seq_lens_q_data, d_qk, d_vo, h_qo,
                                      tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
                                      packed_tma_desc_q, packed_tma_desc_o, q, out,
                                      batch_offset_array_data);

    cudaMemcpyAsync(workspace_start, packed_tma_desc.get(), sizeof(tma::cudaTmaDesc) * (4 * b),
                    cudaMemcpyHostToDevice, stream);
  } else {
    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);

    cudaStream_t raw_stream = get_stream(q.device());

    cudaError_t err = cudaStreamQuery(raw_stream);
    if (!(err == cudaSuccess || err == cudaErrorNotReady)) {
      throw std::runtime_error("CUDA cudnn stream error" + std::string(cudaGetErrorString(err)));
    }

    qkv_tma_setup_prefill<<<grid, block, 0, raw_stream>>>(
        b, h_qo, h_kv, d_qk, d_vo, is_kv_ragged, page_size, total_num_pages, k_cache.stride(2),
        k_cache.stride(1), k_cache.stride(0), v_cache.stride(2), v_cache.stride(1),
        v_cache.stride(0), static_cast<int32_t*>(actual_seq_lens_q_gpu.data_ptr()),
        static_cast<int32_t*>(actual_seq_lens_kv_gpu.data_ptr()), q.data_ptr(), k_cache.data_ptr(),
        v_cache.data_ptr(), out.data_ptr(), packed_tma_desc_q_dev, tma_desc_k, tma_desc_v,
        packed_tma_desc_o_dev);
  }

  cudnn_sdpa::AttentionDescriptor_t attn_desc{
      static_cast<uint32_t>(b),    static_cast<uint32_t>(h_qo),        static_cast<uint32_t>(h_kv),
      static_cast<uint32_t>(h_kv), static_cast<uint32_t>(s_qo),        static_cast<uint32_t>(s_kv),
      static_cast<uint32_t>(d_qk), static_cast<uint32_t>(h_qo / h_kv), is_kv_ragged};

  float attn_scale = scale;

  cudnn_sdpa::strides_t lse_strides = {h_qo * s_qo, 1, h_qo, 1};

  cudnn_sdpa::FastDivisor_t page_size_div;
  setFastDivisor(page_size_div, page_size);

  uint32_t page_size32 = static_cast<uint32_t>(page_size);
  uint32_t num_pages_per_seq32 = static_cast<uint32_t>(num_pages_per_seq);

  void* lse_tensor_pointer = return_lse ? lse.data_ptr() : NULL;

  void* actual_seq_lens_q_gpu_pointer = static_cast<int32_t*>(actual_seq_lens_q_gpu.data_ptr());
  void* actual_seq_lens_kv_gpu_pointer = static_cast<int32_t*>(actual_seq_lens_kv_gpu.data_ptr());
  void* block_tables_pointer = d_qk == 192 ? NULL : static_cast<int32_t*>(block_tables.data_ptr());

  auto print_cudaTmaDescTiled = [](tma::cudaTmaDescTiled* desc) {
    printf("addr %p", desc->tensor_common0);
    printf(" common1 %x", desc->tensor_common1);
    printf(" stride %x", (desc->tensor_stride_lower[0] << 4));
    printf(" stride %x", (desc->tensor_stride_lower[1] << 4));
    printf(" stride %x", (desc->tensor_stride_lower[2] << 4));
    printf(" stride %x", (desc->tensor_stride_lower[3] << 4));
    printf(" stride %x", desc->tensor_stride_upper);
    printf(" size0 %x", desc->tensor_size[0]);
    printf(" size1 %x", desc->tensor_size[1]);
    printf(" size2 %x", desc->tensor_size[2]);
    printf(" size3 %x", desc->tensor_size[3]);
    printf(" size4 %x", desc->tensor_size[4]);
    printf(" stride %x", desc->traversal_stride_box_0);
    printf(" box_size_end %d", desc->box_size_end);
    printf("\n");
  };

  // for (int i = 0; i < b; i++) {
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_q[i]));
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_o[i]));
  // }
  // print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(tma_desc_v_host));

  void* args[14];
  args[0] = (void*)&attn_desc;
  args[1] = (void*)&packed_tma_desc_q_dev;
  args[2] = (void*)&tma_desc_k;
  args[3] = (void*)&attn_scale;
  args[4] = &lse_tensor_pointer;
  args[5] = (void*)&lse_strides;
  args[6] = (void*)&tma_desc_v;
  args[7] = (void*)&packed_tma_desc_o_dev;
  args[8] = &actual_seq_lens_q_gpu_pointer;
  args[9] = &actual_seq_lens_kv_gpu_pointer;
  args[10] = &block_tables_pointer;
  args[11] = &page_size32;
  args[12] = &num_pages_per_seq32;
  args[13] = &page_size_div;

  auto err_launch = CUDA_SUCCESS;

  auto choice = KERNEL_PREFILL;
  if (causal) {
    choice = d_qk == 192 ? KERNEL_PREFILL_DEEPSEEK_CAUSAL : KERNEL_PREFILL_CAUSAL;
  } else {
    choice = d_qk == 192 ? KERNEL_PREFILL_DEEPSEEK : KERNEL_PREFILL;
  }

  err_launch = cuLaunchKernelEx(&config, prefill_func[choice], (void**)args, nullptr);

  if (err_launch != CUDA_SUCCESS) {
    const char* errstr = NULL;
    cuGetErrorString(err_launch, &errstr);
    throw std::runtime_error("Failed to cuLaunchKernelEx for prefill");
  }
}

static int32_t compute_split_factor(int32_t b, int32_t h_kv, int32_t h_qo, int32_t s_kv,
                                    uint32_t sm_count) {
  uint32_t split_factor = 1;
  if ((b * h_kv <= (sm_count / 2))) {
    split_factor = std::ceil(1.f * sm_count / (b * h_kv));
    int i = 2;
    for (; i < 128; i *= 2) {
      if (split_factor <= (i + (i / 2) + (i / 4))) {
        split_factor = i;
        break;
      }
    }
    if (i == 128) {
      split_factor = 64;
    }
    if ((h_qo / h_kv) <= 8) {
      while (std::ceil(1.f * s_kv / split_factor) < (h_qo / h_kv)) {
        split_factor /= 2;
      }
      if (s_kv <= 512) {
        split_factor = 1;
      }
    } else {
      if (s_kv <= 1024) {
        split_factor = 1;
      }
    }
    if (split_factor == 0) {
      split_factor = 1;
    }
  }
  return split_factor;
}

int32_t get_kernel_id(int32_t q_heads_per_kv) {
  auto kernel_id = 0;
  if (q_heads_per_kv == 1) {
    kernel_id = 0;
  } else if (q_heads_per_kv <= 8) {
    kernel_id = 1;
  } else if (q_heads_per_kv <= 16) {
    kernel_id = 2;
  } else if (q_heads_per_kv <= 32) {
    kernel_id = 3;
  } else {
    kernel_id = 4;
  }
  return kernel_id;
}

void setup_tma_desc_decode(int64_t b, int64_t s_kv, int64_t h_qo, int64_t h_kv, int64_t d,
                           int64_t total_num_pages, TensorView q, TensorView out,
                           TensorView k_cache, TensorView v_cache, int32_t split_factor,
                           int64_t page_size, int8_t* partial_o_dev, tma::cudaTmaDesc* tma_desc_q,
                           tma::cudaTmaDesc* tma_desc_o, tma::cudaTmaDesc* tma_desc_partial_o,
                           tma::cudaTmaDesc* tma_desc_k, tma::cudaTmaDesc* tma_desc_v) {
  auto kid = get_kernel_id(h_qo / h_kv);
  int64_t TILE_M_1 = 1;
  int64_t TILE_N_1 = 128;
  switch (kid) {
    case 0:
      TILE_M_1 = 1;
      break;
    case 1:
      TILE_M_1 = 8;
      break;
    case 2:
      TILE_M_1 = 16;
      break;
    case 3:
      TILE_M_1 = 32;
      break;
    case 4:
      TILE_M_1 = 64;
      break;
  }

  constexpr int64_t DIMS_QKV = 4;

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_qo = {64, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_kv = {64, std::min(TILE_N_1, page_size), 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_partial_o = {32, 1, 1, 1};

  std::array<uint32_t, DIMS_QKV> tensor_size_qo = {d, 1 /* s_qo */, h_qo, b};
  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};

  auto kv_strides = k_cache.strides();

  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                         d * BYTES_PER_ELEMENT, 0};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides[2] * (BYTES_PER_ELEMENT),
                                                         kv_strides[1] * (BYTES_PER_ELEMENT),
                                                         kv_strides[0] * (BYTES_PER_ELEMENT)};

  std::array<uint32_t, DIMS_QKV> tensor_size_partial_o = {d, split_factor, h_qo, b};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_partial_o = {
      h_qo * d * b * sizeof(float), d * b * sizeof(float), d * h_qo * sizeof(float)};
  uint16_t* q_ptr = reinterpret_cast<uint16_t*>(q.data_ptr());
  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr());
  float* partial_o_ptr = reinterpret_cast<float*>(partial_o_dev);

  int64_t batch_offset_qo = 0;

  for (int64_t i = 0; i < b; i++) {
    tma::cudaSetTmaTileDescriptor(
        &tma_desc_q[i], q_ptr + batch_offset_qo, DIMS_QKV, tensor_size_qo.data(),
        tensor_stride_qo.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_qo.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    tma::cudaSetTmaTileDescriptor(
        &tma_desc_o[i], out_ptr + batch_offset_qo, DIMS_QKV, tensor_size_qo.data(),
        tensor_stride_qo.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_qo.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    tma::cudaSetTmaTileDescriptor(&tma_desc_partial_o[i], partial_o_ptr + batch_offset_qo, DIMS_QKV,
                                  tensor_size_partial_o.data(), tensor_stride_partial_o.data(),
                                  tensor_traversal_stride_qkv.data(),
                                  tensor_box_size_partial_o.data(), tma::cudaTmaDescFormat::F32_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    batch_offset_qo += h_qo * d;
  }

  tma::cudaSetTmaTileDescriptor(tma_desc_k, k_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
                                tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(),
                                tensor_box_size_kv.data(), tma::cudaTmaDescFormat::BF16_RN,
                                tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  tma::cudaSetTmaTileDescriptor(tma_desc_v, v_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
                                tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(),
                                tensor_box_size_kv.data(), tma::cudaTmaDescFormat::BF16_RN,
                                tma::cudaTmaDescSwizzle::SWIZZLE_128B);
}

void decode(int64_t max_s_kv, TensorView q, TensorView k_cache, TensorView v_cache, double scale,
            TensorView workspace_buffer, TensorView actual_seq_lens_kv,
            TensorView actual_seq_lens_kv_gpu, TensorView block_tables, TensorView out,
            Optional<TensorView> batch_offset_q_array, Optional<TensorView> batch_offset_o_array,
            bool is_cuda_graph_compatible) {
  constexpr size_t SMEM_SIZE = 227 * 1024;  // All smem
  constexpr size_t REDUCTION_MEM_SIZE = 128 * 1024;
  constexpr int64_t TILE_N_1 = 128;

  constexpr int32_t NUM_THREADS = 384;

  int64_t* batch_offset_q_array_data = nullptr;
  if (batch_offset_q_array.has_value()) {
    batch_offset_q_array_data = static_cast<int64_t*>(batch_offset_q_array.value().data_ptr());
  }

  const CUstream stream = get_stream(q.device());

  constexpr int NUM_DECODE_KERNELS = 5;
  static CUfunction hfunc_decode[NUM_DECODE_KERNELS] = {nullptr, nullptr, nullptr, nullptr,
                                                        nullptr};
  static CUfunction lean_attn_reduction{nullptr};

  static int sm_count = 0;

  // Setup decode kernels
  if (hfunc_decode[0] == nullptr) {
    setup_decode(hfunc_decode, &lean_attn_reduction);

    for (int i = 0; i < NUM_DECODE_KERNELS; i++) {
      if (hfunc_decode[i] != nullptr) {
        cuErrCheck(cuFuncSetAttribute(hfunc_decode[i],
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_SIZE));
        cuErrCheck(cuFuncSetAttribute(hfunc_decode[i],
                                      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
        cuErrCheck(cuFuncSetAttribute(hfunc_decode[i],
                                      CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
      }
    }
    if (lean_attn_reduction != nullptr) {
      cuErrCheck(cuFuncSetAttribute(lean_attn_reduction,
                                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    REDUCTION_MEM_SIZE));
      cuErrCheck(cuFuncSetAttribute(lean_attn_reduction,
                                    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
      cuErrCheck(cuFuncSetAttribute(lean_attn_reduction,
                                    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }

    // Get number of SMs per GPU
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
  }

  int64_t b = q.size(0);
  int64_t h_qo = q.size(1);
  int64_t d = q.size(2);

  int64_t h_kv = k_cache.size(1);

  int64_t page_size = k_cache.ndim() == 4 ? k_cache.size(2) : 1;

  int64_t total_num_pages = k_cache.ndim() == 4 ? k_cache.size(0) : 1;

  int64_t s_kv = max_s_kv;

  int64_t s_qo = 1;

  int32_t split_factor = compute_split_factor(b, h_kv, h_qo, s_kv, sm_count);

  split_factor = 1;  // Fix split factor. Setting it to 1 for now

  // Set up TMA descriptors for Q, K, V, O
  auto qo_strides = q.strides();
  auto kv_strides = v_cache.strides();

  // Launch config for main kernel
  CUlaunchConfig config;
  CUlaunchAttribute attrs[1];
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[0].value.clusterDim.x = 1;
  attrs[0].value.clusterDim.y = 1;
  attrs[0].value.clusterDim.z = 1;

  const unsigned int CTAs_y = h_kv * std::ceil(1.f * (h_qo / h_kv) / 64);

  config.gridDimX = split_factor;  // Number of CTAs per row
  config.gridDimY = CTAs_y;
  config.gridDimZ = b;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  config.attrs = attrs;
  config.sharedMemBytes = SMEM_SIZE;

  config.hStream = stream;
  config.numAttrs = 1;

  int8_t* workspace_start = static_cast<int8_t*>(workspace_buffer.data_ptr());
  int8_t* partial_o_dev = workspace_start;
  int8_t* tma_descriptor_start =
      partial_o_dev + (b * s_qo * h_qo * d * sizeof(float) * split_factor);

  int8_t* batch_strides_dev = tma_descriptor_start + ((5 * b) * sizeof(tma::cudaTmaDesc));

  tma::cudaTmaDesc* packed_tma_desc_q_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start);
  tma::cudaTmaDesc* packed_tma_desc_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc));
  tma::cudaTmaDesc* packed_tma_desc_partial_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 2);
  tma::cudaTmaDesc* tma_desc_k_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 3);
  tma::cudaTmaDesc* tma_desc_v_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 4);

  int8_t* lse_dev = batch_strides_dev + (b * sizeof(int64_t));

  if (is_cuda_graph_compatible) {
    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);
    auto kid = get_kernel_id(h_qo / h_kv);
    int64_t TILE_M_1 = 1;
    switch (kid) {
      case 0:
        TILE_M_1 = 1;
        break;
      case 1:
        TILE_M_1 = 8;
        break;
      case 2:
        TILE_M_1 = 16;
        break;
      case 3:
        TILE_M_1 = 32;
        break;
      case 4:
        TILE_M_1 = 64;
        break;
    }

    qkv_tma_setup_decode<<<grid, block, 0, stream>>>(
        b, h_qo, h_kv, d, total_num_pages, page_size, split_factor, TILE_M_1, TILE_N_1,
        kv_strides[2], kv_strides[1], kv_strides[0], q.data_ptr(), k_cache.data_ptr(),
        v_cache.data_ptr(), out.data_ptr(), partial_o_dev, packed_tma_desc_q_dev, tma_desc_k_dev,
        tma_desc_v_dev, packed_tma_desc_o_dev, packed_tma_desc_partial_o_dev,
        reinterpret_cast<int64_t*>(batch_strides_dev));
  } else {
    std::unique_ptr<tma::cudaTmaDesc[]> tma_desc_host(new tma::cudaTmaDesc[5 * b]);

    tma::cudaTmaDesc* tma_desc_q = tma_desc_host.get();
    tma::cudaTmaDesc* tma_desc_o = tma_desc_host.get() + b;
    tma::cudaTmaDesc* tma_desc_partial_o = tma_desc_host.get() + b * 2;
    tma::cudaTmaDesc* tma_desc_k = tma_desc_host.get() + b * 3;
    tma::cudaTmaDesc* tma_desc_v = tma_desc_host.get() + b * 4;

    setup_tma_desc_decode(b, max_s_kv, h_qo, h_kv, d, total_num_pages, q, out, k_cache, v_cache,
                          split_factor, page_size, partial_o_dev, tma_desc_q, tma_desc_o,
                          tma_desc_partial_o, tma_desc_k, tma_desc_v);

    std::unique_ptr<int64_t[]> batch_strides(new int64_t[b]);
    for (int i = 0; i < b; i++) {
      batch_strides[i] = (i)*d * h_qo;
    }
    cudaMemcpyAsync(batch_strides_dev, batch_strides.get(), sizeof(int64_t) * b,
                    cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(tma_descriptor_start, tma_desc_host.get(), sizeof(tma::cudaTmaDesc) * (5 * b),
                    cudaMemcpyHostToDevice, stream);
  }

  cudnn_sdpa::AttentionDescriptor_t attnDesc{b,        h_qo, h_kv,        h_kv, s_qo,
                                             max_s_kv, d,    h_qo / h_kv, 0};

  cudnn_sdpa::FastDivisor_t page_size_div;
  setFastDivisor(page_size_div, page_size);

  uint32_t page_size32 = static_cast<uint32_t>(page_size);
  uint32_t num_pages_per_seq32 = static_cast<uint32_t>(max_s_kv / page_size);

  void* args[15];

  float attn_scale = scale;
  void* actual_seq_lens_q_gpu_pointer = nullptr;
  void* actual_seq_lens_kv_gpu_pointer = static_cast<int32_t*>(actual_seq_lens_kv_gpu.data_ptr());
  void* block_tables_pointer = static_cast<int32_t*>(block_tables.data_ptr());

  cudnn_sdpa::strides_t lse_strides = {h_qo, 1, h_qo, 1};
  cudnn_sdpa::strides_t partial_lse_strides = {h_qo, 1, h_qo * b, 1};

  cudnn_sdpa::strides_t partial_o_strides = {split_factor * h_qo * d, h_qo * d, d, 1};

  args[0] = (void*)&attnDesc;
  args[1] = (void*)&packed_tma_desc_q_dev;
  args[2] = (void*)&tma_desc_k_dev;
  args[3] = (void*)&split_factor;
  args[4] = (void*)&attn_scale;
  args[5] = (void*)&lse_dev;
  args[6] = split_factor == 1 ? (void*)&lse_strides : (void*)&partial_lse_strides;
  args[7] = (void*)&tma_desc_v_dev;
  args[8] =
      split_factor == 1 ? (void*)&packed_tma_desc_o_dev : (void*)&packed_tma_desc_partial_o_dev;
  args[9] = (void*)&actual_seq_lens_q_gpu_pointer;
  args[10] = (void*)&actual_seq_lens_kv_gpu_pointer;
  args[11] = (void*)&block_tables_pointer;
  args[12] = (void*)&page_size32;
  args[13] = (void*)&num_pages_per_seq32;
  args[14] = (void*)&page_size_div;

  auto kernel_id = get_kernel_id(attnDesc.q_heads_per_kv);

  auto err_launch = cuLaunchKernelEx(&config, hfunc_decode[kernel_id], (void**)args, nullptr);
  if (err_launch != CUDA_SUCCESS) {
    std::cerr << "cuLaunchKernelEx failed with error code " << err_launch << std::endl;
    throw std::runtime_error("cuLaunchKernelEx failed for decode");
  }

  // Now setting up the reduction kernel
  if (split_factor > 1) {
    // TODO: Add support for split_factor > 1
    void* args_lean_attn_reduction[11];
    void* o_dev = out.data_ptr();

    void* lse_final_dev = nullptr;

    cudnn_sdpa::strides_t o_strides = {h_qo * d, d, 1};

    args_lean_attn_reduction[0] = (void*)&attnDesc;
    args_lean_attn_reduction[1] = (void*)&split_factor;
    args_lean_attn_reduction[2] = (void*)&o_dev;
    args_lean_attn_reduction[3] = (void*)&partial_o_dev;
    args_lean_attn_reduction[4] = (void*)&lse_final_dev;
    args_lean_attn_reduction[5] = (void*)&lse_dev;
    args_lean_attn_reduction[6] = (void*)&o_strides;
    args_lean_attn_reduction[7] = (void*)&partial_o_strides;
    args_lean_attn_reduction[8] = (void*)&lse_strides;
    args_lean_attn_reduction[9] = (void*)&partial_lse_strides;
    args_lean_attn_reduction[10] = (void*)&batch_strides_dev;

    // Launch config for reduction kernel

    CUlaunchConfig reduction_config;

    reduction_config.gridDimX = h_qo;
    reduction_config.gridDimY = b;  // Same as CTAs_z of main kernel
    reduction_config.gridDimZ = 1;

    reduction_config.blockDimX = 128;  // 128 threads per block
    reduction_config.blockDimY = 1;
    reduction_config.blockDimZ = 1;

    reduction_config.sharedMemBytes = REDUCTION_MEM_SIZE;

    CUlaunchAttribute reduction_attrs[1];
    reduction_attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    reduction_attrs[0].value.clusterDim.x = 1;
    reduction_attrs[0].value.clusterDim.y = 1;
    reduction_attrs[0].value.clusterDim.z = 1;

    reduction_config.hStream = stream;
    reduction_config.numAttrs = 1;
    reduction_config.attrs = reduction_attrs;

    auto err_launch = cuLaunchKernelEx(&reduction_config, lean_attn_reduction,
                                       (void**)args_lean_attn_reduction, nullptr);
    if (err_launch != CUDA_SUCCESS) {
      std::cerr << "cuLaunchKernelEx failed with error code " << err_launch << std::endl;
      throw std::runtime_error("cuLaunchKernelEx failed for decode");
    }
  }
}

}  // namespace cudnn_sdpa_kernel_launcher

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill, cudnn_sdpa_kernel_launcher::prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode, cudnn_sdpa_kernel_launcher::decode);

}  // namespace flashinfer
