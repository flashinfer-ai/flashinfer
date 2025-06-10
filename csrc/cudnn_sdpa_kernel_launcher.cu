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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "cudnn_sdpa_utils.h"
#include "pytorch_extension_utils.h"

// TODO: Make sure this has a name space. Ask in the PR
std::string getCubin(const std::string& kernelName, const std::string& sha256);

namespace flashinfer {

namespace cudnn_sdpa_kernel_launcher {

#include <flashinfer/cubin_loader.h>

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

enum KernelType { PREFILL, DECODE };

void init_cudnn_cubin(std::map<KernelType, std::string>& cubin_map) {
  cubin_map[PREFILL] = getCubin("fmha/sm100/cudnn_sm100_fprop_sdpa_prefill_d128_bf16",
                                "942eaf580377478005bf0fb99f96dd8414d80b53ef85c083804b220bb4fd30a9");

  cubin_map[DECODE] = getCubin("fmha/sm100/cudnn_sm100_fprop_sdpa_decode_d128_bf16",
                               "09944136e9b9dda3a5943284b47482c3be1f63e7ecd69b8b166929b8ce563bfe");
}

auto get_cudnn_cubin(KernelType kernel_type) -> std::string {
  static std::map<KernelType, std::string> cubin_map;
  std::call_once(init_cudnn_cubin_flag, init_cudnn_cubin, std::ref(cubin_map));
  return cubin_map[kernel_type];
}

__global__ static void __launch_bounds__(128)
    qkv_tma_setup_prefill(const unsigned int b, const unsigned int h_qo, const unsigned int h_kv,
                          const unsigned int d, const unsigned int page_size,
                          const unsigned int total_num_pages,

                          const int64_t kv_strides_2, const int64_t kv_strides_1,
                          const int64_t kv_strides_0,

                          int32_t* actual_seq_lens_q_data,

                          void* q_ptr, const void* k_ptr, const void* v_ptr, void* o_ptr,

                          tma::cudaTmaDesc* tma_desc_q_array, void* tma_desc_k, void* tma_desc_v,
                          tma::cudaTmaDesc* tma_desc_o_array
                          /* const int64_t *batch_offset_array */) {
  const int tid = threadIdx.x;

  constexpr unsigned int DIMS_QKV = 4;
  constexpr unsigned int TILE_M_1 = 128;
  constexpr unsigned int TILE_N_1 = 128;
  constexpr unsigned int BYTES_PER_ELEMENT = 2;

  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides_2 * (BYTES_PER_ELEMENT),
                                                         kv_strides_1 * (BYTES_PER_ELEMENT),
                                                         kv_strides_0 * (BYTES_PER_ELEMENT)};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_k = {64, std::min(TILE_N_1, page_size), 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_q = {64, TILE_M_1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};

  tma::cudaSetTmaTileDescriptor(
      reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_k), k_ptr, DIMS_QKV, tensor_size_kv.data(),
      tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
      tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  tma::cudaSetTmaTileDescriptor(
      reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_v), v_ptr, DIMS_QKV, tensor_size_kv.data(),
      tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
      tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  int64_t batch_offset_qo = 0;
#pragma unroll 1
  for (int i = 0; i < b; ++i) {
    const uint32_t actual_s_q = static_cast<uint32_t>(actual_seq_lens_q_data[i]);

    // batch_offset_qo = batch_offset_array ? batch_offset_array[i] : batch_offset_qo;
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_qo = {d, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                                  d * BYTES_PER_ELEMENT, 0};

    uint16_t* per_batch_q_ptr = reinterpret_cast<uint16_t*>(q_ptr + batch_offset_qo);
    uint16_t* per_batch_out_ptr = reinterpret_cast<uint16_t*>(o_ptr + batch_offset_qo);

    tma::cudaTmaDesc desc_q;
    tma::cudaTmaDesc desc_o;

    tma::cudaSetTmaTileDescriptor(&desc_q, (void*)per_batch_q_ptr, DIMS_QKV,
                                  packed_tensor_size_qo.data(), packed_tensor_stride_qo.data(),
                                  tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
                                  tma::cudaTmaDescFormat::BF16_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(&desc_o, (void*)per_batch_out_ptr, DIMS_QKV,
                                  packed_tensor_size_qo.data(), packed_tensor_stride_qo.data(),
                                  tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
                                  tma::cudaTmaDescFormat::BF16_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_q_array)[i] = desc_q;
    reinterpret_cast<tma::cudaTmaDesc*>(tma_desc_o_array)[i] = desc_o;

    batch_offset_qo += static_cast<int64_t>(actual_s_q) * d * h_qo *
                       BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
  }
}

static void create_packed_tma_desc_prefill(int b, int32_t* actual_seq_lens_q_data, int64_t d,
                                           int64_t h_qo, uint32_t* tensor_traversal_stride_qkv,
                                           uint32_t* tensor_box_size_q,
                                           tma::cudaTmaDesc* packed_tma_desc_q,
                                           tma::cudaTmaDesc* packed_tma_desc_o, at::Tensor q,
                                           at::Tensor out, int64_t* batch_offset_array) {
  int64_t batch_offset_qo = 0;
  // tma descriptors for packed q and o
  for (int i = 0; i < b; ++i) {
    const uint32_t actual_s_q = static_cast<uint32_t>(actual_seq_lens_q_data[i]);

    batch_offset_qo = batch_offset_array ? batch_offset_array[i] : batch_offset_qo;
    std::array<uint32_t, DIMS_QKV> packed_tensor_size_qo = {d, actual_s_q, h_qo, 1};
    std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                                  d * BYTES_PER_ELEMENT, 0};

    uint16_t* q_ptr = reinterpret_cast<uint16_t*>(q.data_ptr() + batch_offset_qo);
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr() + batch_offset_qo);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_q[i], (void*)q_ptr, DIMS_QKV, packed_tensor_size_qo.data(),
        packed_tensor_stride_qo.data(), tensor_traversal_stride_qkv, tensor_box_size_q,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_o[i], (void*)out_ptr, DIMS_QKV, packed_tensor_size_qo.data(),
        packed_tensor_stride_qo.data(), tensor_traversal_stride_qkv, tensor_box_size_q,
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    batch_offset_qo += static_cast<int64_t>(actual_s_q) * d * h_qo *
                       BYTES_PER_ELEMENT;  // Becomes a no-op if batch_offset_array is provided
  }
}

static void create_packed_tma_desc_decode(int b, int64_t d, int64_t h_qo,
                                          uint32_t* tensor_traversal_stride_qkv,
                                          tma::cudaTmaDesc* packed_tma_desc_q,
                                          tma::cudaTmaDesc* packed_tma_desc_o,
                                          tma::cudaTmaDesc* packed_tma_desc_partial_o, at::Tensor q,
                                          at::Tensor out,
                                          // int64_t* batch_offset_array,
                                          uint32_t split_factor, int8_t* partial_o_dev) {
  int64_t batch_offset_qo = 0;
  const uint32_t actual_s_q = 1;

  constexpr unsigned int TILE_M_1 = 128;

  std::array<uint32_t, DIMS_QKV> tensor_box_size_q = {64, TILE_M_1, 1, 1};
  std::array<uint32_t, DIMS_QKV> packed_tensor_size_qo = {d, actual_s_q, h_qo, 1};
  std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                                d * BYTES_PER_ELEMENT, 0};

  // std::array<uint32_t, DIMS_QKV>     packed_tensor_size_partial_o   = {d, split_factor, h_qo, 1};
  // std::array<uint64_t, DIMS_QKV - 1> packed_tensor_stride_partial_o = {h_qo * d * sizeof(float),
  //   h_qo * d *sizeof(float), d * sizeof(float) };
  // std::array<uint32_t, DIMS_QKV> packed_tensor_box_size_partial_o   = {32, 1, 1, 1};

  for (int i = 0; i < b; ++i) {
    // batch_offset_qo = batch_offset_array ? batch_offset_array[i] : batch_offset_qo;
    uint16_t* q_ptr = reinterpret_cast<uint16_t*>(q.data_ptr() + batch_offset_qo);
    uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr() + batch_offset_qo);
    float* partial_out_ptr =
        reinterpret_cast<float*>(partial_o_dev + (batch_offset_qo * sizeof(float)));

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_q[i], q_ptr, DIMS_QKV, packed_tensor_size_qo.data(),
        packed_tensor_stride_qo.data(), tensor_traversal_stride_qkv, tensor_box_size_q.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        &packed_tma_desc_o[i], out_ptr, DIMS_QKV, packed_tensor_size_qo.data(),
        packed_tensor_stride_qo.data(), tensor_traversal_stride_qkv, tensor_box_size_q.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    // tma::cudaSetTmaTileDescriptor(
    //     &packed_tma_desc_partial_o[i], partial_out_ptr, DIMS_QKV,
    //     packed_tensor_size_partial_o.data(), packed_tensor_stride_partial_o.data(),
    //     tensor_traversal_stride_qkv, packed_tensor_box_size_partial_o.data(),
    //     tma::cudaTmaDescFormat::F32_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    batch_offset_qo += static_cast<uint64_t>(actual_s_q) * d * h_qo;
  }
}

void setup_prefill(CUfunction* hfunc) {
  // Use cu++filt to get the kernel name
  std::string kernel_name =
      "_Z47cudnn_sm100_fprop_sdpa_prefill_bf16_"
      "128x128x128ILb1ELb1EEvN4fmha19AttentionDescriptorEPKN3tma11cudaTmaDescES5_fPfNS0_"
      "7stridesES5_S5_PKjS9_S9_jjNS0_11FastDivisorE";

  std::string cubin = get_cudnn_cubin(PREFILL);
  if (cubin.empty()) {
    throw std::runtime_error("Failed to load cubin for prefill");
  }

  CUmodule hmod{0};
  if (cuModuleLoadData(&hmod, cubin.data()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleLoadData for prefill");
  }

  if (cuModuleGetFunction(hfunc, hmod, kernel_name.c_str()) != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to cuModuleGetFunction for prefill");
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

void prefill(int64_t b, int64_t s_qo, at::Tensor q, at::Tensor k_cache, at::Tensor v_cache,
             double scale, at::Tensor workspace_buffer, at::Tensor actual_seq_lens_q,
             at::Tensor actual_seq_lens_kv, at::Tensor actual_seq_lens_q_gpu,
             at::Tensor actual_seq_lens_kv_gpu, at::Tensor block_tables, bool causal,
             bool return_lse, at::Tensor out, at::Tensor lse,
             std::optional<at::Tensor> batch_offset_array, bool use_cuda_graph) {
  constexpr size_t SMEM_SIZE = 227 * 1024;  // All smem
  constexpr int64_t TILE_M_1 = 128;
  constexpr int64_t TILE_N_1 = 128;

  constexpr int32_t NUM_THREADS = 512;

  auto device = q.device();
  const CUstream stream = at::cuda::getCurrentCUDAStream(device.index());

  int64_t* batch_offset_array_data = nullptr;
  if (batch_offset_array.has_value()) {
    batch_offset_array_data = batch_offset_array.value().data_ptr<int64_t>();
  }

  // Step 1: Setup the kernel pointer

  static CUfunction hfunc{nullptr};

  if (hfunc == nullptr) {
    setup_prefill(&hfunc);

    if (hfunc != nullptr) {
      cuErrCheck(
          cuFuncSetAttribute(hfunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_SIZE));
      cuErrCheck(
          cuFuncSetAttribute(hfunc, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
      cuErrCheck(cuFuncSetAttribute(hfunc, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }
  }

  // Step 2: Extract attention descriptor

  TORCH_CHECK(k_cache.dim() >= 3, "Input tensor k_cache must have at least 3 dimensions");

  int64_t h_qo = q.size(1);
  int64_t d = q.size(2);

  int64_t h_kv = k_cache.size(1);
  int64_t page_size = k_cache.size(2);
  int64_t s_kv = (k_cache.size(0) / b) * page_size;

  int64_t num_pages_per_seq = std::ceil(s_kv / page_size);

  int64_t total_num_pages = k_cache.size(0);

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

  if (use_cuda_graph == false) {
    TORCH_CHECK(actual_seq_lens_q.is_cuda() == false,
                "actual_seq_lens_q must be on the same device as q");
    TORCH_CHECK(actual_seq_lens_kv.is_cuda() == false,
                "actual_seq_lens_kv must be on the same device as q");
    auto actual_seq_lens_q_data = actual_seq_lens_q.data_ptr<int32_t>();
    auto actual_seq_lens_kv_data = actual_seq_lens_kv.data_ptr<int32_t>();

    uint32_t actual_num_tiles_per_head = std::transform_reduce(
        actual_seq_lens_q_data, actual_seq_lens_q_data + b, 0U, std::plus<>(), [](int32_t seq_len) {
          return static_cast<uint32_t>(std::ceil(seq_len / (TILE_M_1 * 2.0f)));
        });
    config.gridDimX = actual_num_tiles_per_head;

  } else {
    config.gridDimX = static_cast<int>(std::ceil(q.size(0) / (TILE_M_1 * 2.0f))) * b;
  }

  config.gridDimY = h_qo;
  config.gridDimZ = 1;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  // Step 4: Set up the launch arguments

  auto kv_strides = v_cache.strides();

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides[2] * (BYTES_PER_ELEMENT),
                                                         kv_strides[1] * (BYTES_PER_ELEMENT),
                                                         kv_strides[0] * (BYTES_PER_ELEMENT)};

  std::array<uint32_t, DIMS_QKV> tensor_box_size_q = {64, TILE_M_1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_k = {64, std::min(TILE_N_1, page_size), 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_v = {64, std::min(TILE_N_1, page_size), 1, 1};

  uint64_t batch_offset_qo = 0;
  int8_t* workspace_start = workspace_buffer.data_ptr<int8_t>();

  // These tensors are allocated in the workspace buffer
  // Using 2 * b for q and o
  std::unique_ptr<tma::cudaTmaDesc[]> packed_tma_desc(new tma::cudaTmaDesc[(2 * b) + 2]);
  auto packed_tma_desc_q = packed_tma_desc.get();
  auto packed_tma_desc_o = packed_tma_desc.get() + b;
  auto tma_desc_k_host = packed_tma_desc.get() + (2 * b);
  auto tma_desc_v_host = packed_tma_desc.get() + (2 * b + 1);

  tma::cudaTmaDesc* packed_tma_desc_q_dev = reinterpret_cast<tma::cudaTmaDesc*>(workspace_start);
  tma::cudaTmaDesc* packed_tma_desc_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * b);

  // These TMA descriptors are allocated in the host and passed by value
  tma::cudaTmaDesc* tma_desc_k =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * (2 * b));
  tma::cudaTmaDesc* tma_desc_v =
      reinterpret_cast<tma::cudaTmaDesc*>(workspace_start + sizeof(tma::cudaTmaDesc) * (2 * b + 1));

  if (use_cuda_graph == false) {
    // tma descriptors for k and v
    tma::cudaSetTmaTileDescriptor(
        tma_desc_k_host, k_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
        tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_k.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    tma::cudaSetTmaTileDescriptor(
        tma_desc_v_host, v_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
        tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_v.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);

    auto actual_seq_lens_q_data = actual_seq_lens_q.data_ptr<int32_t>();
    create_packed_tma_desc_prefill(b, actual_seq_lens_q_data, d, h_qo,
                                   tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
                                   packed_tma_desc_q, packed_tma_desc_o, q, out,
                                   batch_offset_array_data);

    cudaMemcpyAsync(workspace_start, packed_tma_desc.get(), sizeof(tma::cudaTmaDesc) * (2 * b + 2),
                    cudaMemcpyHostToDevice, stream);
  } else {
    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);

    qkv_tma_setup_prefill<<<grid, block, 0, stream>>>(
        b, h_qo, h_kv, d, page_size, total_num_pages, v_cache.strides().data()[2],
        v_cache.strides().data()[1], v_cache.strides().data()[0],
        actual_seq_lens_q_gpu.data_ptr<int32_t>(), q.data_ptr(), k_cache.data_ptr(),
        v_cache.data_ptr(), out.data_ptr(), packed_tma_desc_q_dev, tma_desc_k, tma_desc_v,
        packed_tma_desc_o_dev);

    // auto actual_seq_lens_q_data = actual_seq_lens_q.data_ptr<int32_t>();
    // create_packed_tma_desc_prefill(b, actual_seq_lens_q_data, d, h_qo,
    //                                tensor_traversal_stride_qkv.data(), tensor_box_size_q.data(),
    //                                packed_tma_desc_q, packed_tma_desc_o, q, out,
    //                                batch_offset_array_data);

    // cudaMemcpyAsync(workspace_start, packed_tma_desc.get(), sizeof(tma::cudaTmaDesc) * (2 * b),
    //                 cudaMemcpyHostToDevice, stream);
  }

  cudnn_sdpa::AttentionDescriptor_t attn_desc{
      static_cast<uint32_t>(b),    static_cast<uint32_t>(h_qo),       static_cast<uint32_t>(h_kv),
      static_cast<uint32_t>(h_kv), static_cast<uint32_t>(s_qo),       static_cast<uint32_t>(s_kv),
      static_cast<uint32_t>(d),    static_cast<uint32_t>(h_qo / h_kv)};

  float attn_scale = scale;

  cudnn_sdpa::strides_t lse_strides = {h_qo * s_qo, 1, h_qo, 1};

  cudnn_sdpa::FastDivisor_t page_size_div;
  setFastDivisor(page_size_div, page_size);

  uint32_t page_size32 = static_cast<uint32_t>(page_size);
  uint32_t num_pages_per_seq32 = static_cast<uint32_t>(num_pages_per_seq);

  void* lse_tensor_pointer = return_lse ? lse.data_ptr() : NULL;

  void* actual_seq_lens_q_gpu_pointer = actual_seq_lens_q_gpu.data_ptr<int32_t>();
  void* actual_seq_lens_kv_gpu_pointer = actual_seq_lens_kv_gpu.data_ptr<int32_t>();
  void* block_tables_pointer = block_tables.data_ptr<int32_t>();

  auto print_cudaTmaDescTiled = [](tma::cudaTmaDescTiled* desc) {
    std::cout << std::hex;
    printf("addr %p", desc->tensor_common0);
    printf(" common1 %x", desc->tensor_common1);
    printf(" stride %x", (desc->tensor_stride_lower[0] << 4));
    printf(" stride %x", (desc->tensor_stride_lower[1] << 4));
    std::cout << " stride " << (desc->tensor_stride_lower[2] << 4);
    std::cout << " stride " << (desc->tensor_stride_lower[3] << 4);
    std::cout << " stride " << desc->tensor_stride_upper;
    std::cout << " size " << desc->tensor_size[0];
    std::cout << " size " << desc->tensor_size[1];
    std::cout << " size " << desc->tensor_size[2];
    std::cout << " size " << desc->tensor_size[3];
    std::cout << " size " << desc->tensor_size[4];
    std::cout << " stride " << desc->traversal_stride_box_0;
    std::cout << " box_size_end " << desc->box_size_end;
    std::cout << std::dec;
    std::cout << std::endl;
  };

  // for (int i = 0; i < b; i++) {
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_q[i]));
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_o[i]));
  // }
  print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(tma_desc_v_host));

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

  auto err_launch = cuLaunchKernelEx(&config, hfunc, (void**)args, nullptr);
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

void decode_2(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, double scale,
              at::Tensor workspace_buffer, at::Tensor actual_seq_lens_kv,
              at::Tensor actual_seq_lens_kv_gpu, at::Tensor block_tables, int64_t num_pages_per_seq,
              at::Tensor out, std::optional<at::Tensor> batch_offset_array, bool use_cuda_graph) {
  constexpr size_t SMEM_SIZE = 227 * 1024;  // All smem
  constexpr size_t REDUCTION_MEM_SIZE = 128 * 1024;
  constexpr int64_t TILE_N_1 = 128;

  constexpr int32_t NUM_THREADS = 384;

  int64_t* batch_offset_array_data = nullptr;
  if (batch_offset_array.has_value()) {
    batch_offset_array_data = batch_offset_array.value().data_ptr<int64_t>();
  }

  auto device = q.device();

  const CUstream stream = at::cuda::getCurrentCUDAStream(device.index());

  // Step 1: Setup the kernel pointer

  constexpr int NUM_DECODE_KERNELS = 5;
  static CUfunction hfunc_decode[NUM_DECODE_KERNELS] = {nullptr, nullptr, nullptr, nullptr,
                                                        nullptr};
  static CUfunction lean_attn_reduction{nullptr};

  static uint32_t sm_count = 0;

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

    // Get number of SMs perf GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm_count = prop.multiProcessorCount;
  }

  // Step 2: Extract attention descriptor

  TORCH_CHECK(k_cache.dim() >= 3, "Input tensor k_cache must have at least 3 dimensions");

  int64_t b = q.size(0);
  int64_t h_qo = q.size(1);
  int64_t d = q.size(2);
  int64_t s_q = 1;

  int64_t h_kv = k_cache.size(1);
  int64_t page_size = k_cache.size(2);
  int64_t s_kv = (k_cache.size(0) / b) * page_size;

  int64_t total_num_pages = k_cache.size(0);

  // Step 3: Compute split factor
  int32_t split_factor = compute_split_factor(b, h_kv, h_qo, s_kv, sm_count);

  std::cout << "split_factor: " << split_factor << " total_num_pages: " << total_num_pages
            << std::endl;

  // Partial O dims [B, H, split_factor, D]

  cudnn_sdpa::strides_t lse_strides = {h_qo * s_q, 1, h_qo, 1};

  cudnn_sdpa::strides_t partial_o_strides = {split_factor * h_qo * d, h_qo * d, d, 1};
  cudnn_sdpa::strides_t partial_lse_strides = {h_qo, 1, h_qo * split_factor, 1};

  // Set up TMA descriptors for Q, K, V, O
  auto qo_strides = q.strides();
  auto kv_strides = k_cache.strides();

  std::cout << "qo_strides: " << qo_strides[0] << ", " << qo_strides[1] << ", " << qo_strides[2]
            << std::endl;
  std::cout << "kv_strides: " << kv_strides[0] << ", " << kv_strides[1] << ", " << kv_strides[2]
            << std::endl;

  // // Setup TMA desc for partial tensor O for Lean Attention
  // std::array<uint32_t, DIMS_QKV> tensor_size_partial_o = {d, split_factor, h_qo, b};
  // std::array<uint64_t, DIMS_QKV - 1> tensor_stride_partial_o = {
  //     partial_o_strides[2] * sizeof(float), partial_o_strides[1] * sizeof(float),
  //     partial_o_strides[0] * sizeof(float)};
  // std::array<uint32_t, DIMS_QKV> tensor_box_size_partial_o = {32, 1, 1, 1};

  // Launch config for main kernel
  CUlaunchConfig config;

  CUlaunchAttribute attrs[1];
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[0].value.clusterDim.x = 1;
  attrs[0].value.clusterDim.y = 1;
  attrs[0].value.clusterDim.z = 1;

  const unsigned int CTAs_x = split_factor;  // Number of CTAs per row
  const unsigned int CTAs_y = h_kv * std::ceil(1.f * (h_qo / h_kv) / 64);
  const unsigned int CTAs_z = b;

  config.gridDimX = CTAs_x;
  config.gridDimY = CTAs_y;
  config.gridDimZ = CTAs_z;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  config.attrs = attrs;
  config.sharedMemBytes = SMEM_SIZE;

  config.hStream = stream;
  config.numAttrs = 1;
  config.attrs = attrs;

  int8_t* workspace_start = workspace_buffer.data_ptr<int8_t>();
  int8_t* partial_o_dev = workspace_start;

  int8_t* tma_descriptor_start =
      partial_o_dev + (b * s_q * h_qo * d * sizeof(float) * split_factor);

  int8_t* batch_strides_dev = tma_descriptor_start + (3 * b * sizeof(tma::cudaTmaDesc));

  int8_t* lse_dev = batch_strides_dev + (b * sizeof(int64_t));

  // These TMA descriptors are allocated in the host and passed by value
  tma::cudaTmaDesc tma_desc_k, tma_desc_v;

  /* KV cache TMA descriptors */

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides[2] * (BYTES_PER_ELEMENT),
                                                         kv_strides[1] * (BYTES_PER_ELEMENT),
                                                         kv_strides[0] * (BYTES_PER_ELEMENT)};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_kv = {64, std::min(TILE_N_1, page_size), 1, 1};

  tma::cudaSetTmaTileDescriptor(&tma_desc_k, k_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
                                tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(),
                                tensor_box_size_kv.data(), tma::cudaTmaDescFormat::BF16_RN,
                                tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  tma::cudaSetTmaTileDescriptor(&tma_desc_v, v_cache.data_ptr(), DIMS_QKV, tensor_size_kv.data(),
                                tensor_stride_kv.data(), tensor_traversal_stride_qkv.data(),
                                tensor_box_size_kv.data(), tma::cudaTmaDescFormat::BF16_RN,
                                tma::cudaTmaDescSwizzle::SWIZZLE_128B);

  // These tensors are allocated in the workspace buffer
  // Using 3 * b for q, o and partial_o
  std::unique_ptr<tma::cudaTmaDesc[]> packed_tma_desc(new tma::cudaTmaDesc[3 * b]);
  auto packed_tma_desc_q = packed_tma_desc.get();
  auto packed_tma_desc_o = packed_tma_desc.get() + b;
  auto packed_tma_desc_partial_o = packed_tma_desc.get() + 2 * b;

  /* Query and output TMA descriptors */
  create_packed_tma_desc_decode(b, d, h_qo, tensor_traversal_stride_qkv.data(), packed_tma_desc_q,
                                packed_tma_desc_o, packed_tma_desc_partial_o, q, out, split_factor,
                                partial_o_dev);

  cudaMemcpyAsync(tma_descriptor_start, packed_tma_desc.get(), sizeof(tma::cudaTmaDesc) * 3 * b,
                  cudaMemcpyHostToDevice, stream);

  std::unique_ptr<int64_t[]> batch_strides(new int64_t[b]);

  for (int i = 0; i < b; i++) {
    batch_strides[i] = (batch_offset_array_data ? batch_offset_array_data[i] : i) * d * h_qo;
  }

  cudaMemcpyAsync(batch_strides_dev, batch_strides.get(), sizeof(int64_t) * b,
                  cudaMemcpyHostToDevice, stream);

  auto print_cudaTmaDescTiled = [](tma::cudaTmaDescTiled* desc) {
    std::cout << std::hex;
    printf("addr %p", desc->tensor_common0);
    printf(" common1 %x", desc->tensor_common1);
    printf(" stride %x", (desc->tensor_stride_lower[0] << 4));
    printf(" stride %x", (desc->tensor_stride_lower[1] << 4));
    std::cout << " stride " << (desc->tensor_stride_lower[2] << 4);
    std::cout << " stride " << (desc->tensor_stride_lower[3] << 4);
    std::cout << " stride " << desc->tensor_stride_upper;
    std::cout << " size " << desc->tensor_size[0];
    std::cout << " size " << desc->tensor_size[1];
    std::cout << " size " << desc->tensor_size[2];
    std::cout << " size " << desc->tensor_size[3];
    std::cout << " size " << desc->tensor_size[4];
    std::cout << " stride " << desc->traversal_stride_box_0;
    std::cout << " box_size_end " << desc->box_size_end;
    std::cout << std::dec;
    std::cout << std::endl;
  };

  // for (int i = 0; i < b; i++) {
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_q[i]));
  //   print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&packed_tma_desc_o[i]));
  // }
  print_cudaTmaDescTiled(reinterpret_cast<tma::cudaTmaDescTiled*>(&tma_desc_v));

  // Prepare launch arguments
  cudnn_sdpa::AttentionDescriptor_t attnDesc{b, h_qo, h_kv, h_kv, s_q, s_kv, d, h_qo / h_kv};

  cudnn_sdpa::FastDivisor_t page_size_div;
  setFastDivisor(page_size_div, page_size);

  // void* lse_tensor_pointer = return_lse ? lse.data_ptr() : NULL;

  void* actual_seq_lens_q_gpu_pointer = nullptr;
  void* actual_seq_lens_kv_gpu_pointer = actual_seq_lens_kv_gpu.data_ptr<int32_t>();
  void* block_tables_pointer = block_tables.data_ptr<int32_t>();

  uint32_t page_size32 = static_cast<uint32_t>(page_size);
  uint32_t num_pages_per_seq32 = static_cast<uint32_t>(num_pages_per_seq);

  float attn_scale = scale;

  void* args[15];

  tma::cudaTmaDesc* packed_tma_desc_q_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start);
  tma::cudaTmaDesc* packed_tma_desc_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc));
  tma::cudaTmaDesc* packed_tma_desc_partial_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 2);

  args[0] = (void*)&attnDesc;
  args[1] = (void*)&packed_tma_desc_q_dev;
  args[2] = (void*)&tma_desc_k;
  args[3] = (void*)&split_factor;
  args[4] = (void*)&attn_scale;
  args[5] = (void*)&lse_dev;
  args[6] = split_factor == 1 ? (void*)&lse_strides : (void*)&partial_lse_strides;
  args[7] = (void*)&tma_desc_v;
  args[8] =
      split_factor == 1 ? (void*)&packed_tma_desc_o_dev : (void*)&packed_tma_desc_partial_o_dev;
  args[9] = (void*)&actual_seq_lens_q_gpu_pointer;
  args[10] = (void*)&actual_seq_lens_kv_gpu_pointer;
  args[11] = (void*)&block_tables_pointer;
  args[12] = (void*)&page_size32;
  args[13] = (void*)&num_pages_per_seq32;
  args[14] = (void*)&page_size_div;

  auto kernel_id = get_kernel_id(attnDesc.q_heads_per_kv);

  printf("kernel_id %d\n", kernel_id);

  auto err_launch = cuLaunchKernelEx(&config, hfunc_decode[kernel_id], (void**)args, nullptr);
  if (err_launch != CUDA_SUCCESS) {
    std::cerr << "cuLaunchKernelEx failed with error code " << err_launch << std::endl;
    throw std::runtime_error("cuLaunchKernelEx failed for decode");
  }

  // Now setting up the reduction kernel

  if (split_factor > 1) {
    // Reduction kernel arg setup
    void* args_lean_attn_reduction[11];

    void* o_dev = out.data_ptr();

    void* lse_final_dev = nullptr;

    args_lean_attn_reduction[0] = (void*)&attnDesc;
    args_lean_attn_reduction[1] = (void*)&split_factor;
    args_lean_attn_reduction[2] = (void*)&o_dev;
    args_lean_attn_reduction[3] = (void*)&partial_o_dev;
    args_lean_attn_reduction[4] =
        (void*)&lse_final_dev;  // Where final op is written not needed for decode
    args_lean_attn_reduction[5] = (void*)&lse_dev;
    // args_lean_attn_reduction[6] = (void*)&o_strides;
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

void setup_tma_desc_decode(int64_t b, int64_t s_kv, int64_t h_qo, int64_t h_kv, int64_t d,
                           int64_t total_num_pages, at::Tensor q, at::Tensor out,
                           at::Tensor k_cache, at::Tensor v_cache, int32_t split_factor,
                           int64_t page_size, int8_t* partial_o_dev, tma::cudaTmaDesc* tma_desc_q,
                           tma::cudaTmaDesc* tma_desc_o, tma::cudaTmaDesc* tma_desc_partial_o,
                           tma::cudaTmaDesc* tma_desc_k, tma::cudaTmaDesc* tma_desc_v) {
  int64_t TILE_M_1 = 128;
  int64_t TILE_N_1 = 128;

  constexpr int64_t DIMS_QKV = 4;

  std::array<uint32_t, DIMS_QKV> tensor_traversal_stride_qkv = {1, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_qo = {64, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_box_size_kv = {64, std::min(TILE_N_1, page_size), 1, 1};

  std::array<uint32_t, DIMS_QKV> tensor_size_qo = {d, 1 /* s_qo */, h_qo, b};
  std::array<uint32_t, DIMS_QKV> tensor_size_kv = {d, page_size, h_kv, total_num_pages};

  auto kv_strides = k_cache.strides();

  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_qo = {h_qo * d * BYTES_PER_ELEMENT,
                                                         d * BYTES_PER_ELEMENT, 0};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_kv = {kv_strides[2] * (BYTES_PER_ELEMENT),
                                                         kv_strides[1] * (BYTES_PER_ELEMENT),
                                                         kv_strides[0] * (BYTES_PER_ELEMENT)};

  std::array<uint32_t, DIMS_QKV> tensor_box_size_partial_o = {32, 1, 1, 1};
  std::array<uint32_t, DIMS_QKV> tensor_size_partial_o = {d, split_factor, h_qo, b};
  std::array<uint64_t, DIMS_QKV - 1> tensor_stride_partial_o = {h_qo * d * sizeof(float),
                                                                d * sizeof(float), 1};
  uint16_t* q_ptr = reinterpret_cast<uint16_t*>(q.data_ptr());
  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out.data_ptr());
  float* partial_o_ptr = reinterpret_cast<float*>(partial_o_dev);

  int64_t batch_offset_qo = 0;

  for (int64_t i = 0; i < b; i++) {
    std::cout << "batch_offset_qo " << batch_offset_qo << " h_qo " << h_qo << " d " << d << " b "
              << b << std::endl;

    tma::cudaSetTmaTileDescriptor(
        &tma_desc_q[i], q_ptr + batch_offset_qo, DIMS_QKV, tensor_size_qo.data(),
        tensor_stride_qo.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_qo.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    std::cout << "tma_desc_q[i] " << std::endl;
    tma::cudaSetTmaTileDescriptor(
        &tma_desc_o[i], out_ptr + batch_offset_qo, DIMS_QKV, tensor_size_qo.data(),
        tensor_stride_qo.data(), tensor_traversal_stride_qkv.data(), tensor_box_size_qo.data(),
        tma::cudaTmaDescFormat::BF16_RN, tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    std::cout << "tma_desc_o[i] " << std::endl;
    tma::cudaSetTmaTileDescriptor(&tma_desc_partial_o[i], partial_o_ptr + batch_offset_qo, DIMS_QKV,
                                  tensor_size_partial_o.data(), tensor_stride_partial_o.data(),
                                  tensor_traversal_stride_qkv.data(),
                                  tensor_box_size_partial_o.data(), tma::cudaTmaDescFormat::F32_RN,
                                  tma::cudaTmaDescSwizzle::SWIZZLE_128B);
    std::cout << "tma_desc_partial_o[i] " << std::endl;
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

void decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, double scale,
            at::Tensor workspace_buffer, at::Tensor actual_seq_lens_kv,
            at::Tensor actual_seq_lens_kv_gpu, at::Tensor block_tables, at::Tensor out,
            std::optional<at::Tensor> batch_offset_array, bool use_cuda_graph) {
  constexpr size_t SMEM_SIZE = 227 * 1024;  // All smem
  constexpr size_t REDUCTION_MEM_SIZE = 128 * 1024;
  constexpr int64_t TILE_N_1 = 128;

  constexpr int32_t NUM_THREADS = 384;

  int64_t* batch_offset_array_data = nullptr;
  if (batch_offset_array.has_value()) {
    batch_offset_array_data = batch_offset_array.value().data_ptr<int64_t>();
  }

  auto device = q.device();

  const CUstream stream = at::cuda::getCurrentCUDAStream(device.index());

  constexpr int NUM_DECODE_KERNELS = 5;
  static CUfunction hfunc_decode[NUM_DECODE_KERNELS] = {nullptr, nullptr, nullptr, nullptr,
                                                        nullptr};
  static CUfunction lean_attn_reduction{nullptr};

  static uint32_t sm_count = 0;

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

    // Get number of SMs perf GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm_count = prop.multiProcessorCount;
  }

  int64_t b = q.size(0);
  int64_t h_qo = q.size(1);
  int64_t h_kv = k_cache.size(1);
  int64_t page_size = k_cache.size(3);
  int64_t d = q.size(2);
  int64_t total_num_pages = k_cache.size(0);

  int64_t max_skv = (k_cache.size(0) / b) * page_size;

  std::cout << "b " << b << " h_qo " << h_qo << " h_kv " << h_kv << " page_size " << page_size
            << " d " << d << " total_num_pages " << total_num_pages << " max_skv " << max_skv
            << std::endl;

  int64_t s_qo = 1;

  // TODO: Add support for split_factor > 1
  int32_t split_factor = 1;

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

  const unsigned int CTAs_x = split_factor;  // Number of CTAs per row
  const unsigned int CTAs_y = h_kv * std::ceil(1.f * (h_qo / h_kv) / 64);
  const unsigned int CTAs_z = b;

  config.gridDimX = CTAs_x;
  config.gridDimY = CTAs_y;
  config.gridDimZ = CTAs_z;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  config.blockDimX = NUM_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;

  config.attrs = attrs;
  config.sharedMemBytes = SMEM_SIZE;

  config.hStream = stream;
  config.numAttrs = 1;
  config.attrs = attrs;

  std::cout << "config.gridDimX " << config.gridDimX << " config.gridDimY " << config.gridDimY
            << " config.gridDimZ " << config.gridDimZ << std::endl;
  std::cout << "config.blockDimX " << config.blockDimX << " config.blockDimY " << config.blockDimY
            << " config.blockDimZ " << config.blockDimZ << std::endl;

  int8_t* workspace_start = workspace_buffer.data_ptr<int8_t>();
  int8_t* partial_o_dev = workspace_start;
  int8_t* tma_descriptor_start =
      partial_o_dev + (b * s_qo * h_qo * d * sizeof(float) * split_factor);

  int8_t* batch_strides_dev = tma_descriptor_start + ((3 * b + 2) * sizeof(tma::cudaTmaDesc));

  int8_t* lse_dev = batch_strides_dev + (b * sizeof(int64_t));

  std::unique_ptr<tma::cudaTmaDesc[]> tma_desc_host(new tma::cudaTmaDesc[(3 * b) + 2]);

  tma::cudaTmaDesc* tma_desc_q = tma_desc_host.get();
  tma::cudaTmaDesc* tma_desc_o = tma_desc_host.get() + b;
  tma::cudaTmaDesc* tma_desc_partial_o = tma_desc_host.get() + b * 2;
  tma::cudaTmaDesc* tma_desc_k = tma_desc_host.get() + b * 3;
  tma::cudaTmaDesc* tma_desc_v = tma_desc_host.get() + (b * 3) + 1;

  setup_tma_desc_decode(b, max_skv, h_qo, h_kv, d, total_num_pages, q, out, k_cache, v_cache,
                        split_factor, page_size, partial_o_dev, tma_desc_q, tma_desc_o,
                        tma_desc_partial_o, tma_desc_k, tma_desc_v);

  std::unique_ptr<int64_t[]> batch_strides(new int64_t[b]);
  for (int i = 0; i < b; i++) {
    batch_strides[i] = (i)*d * h_qo;
  }
  cudaMemcpyAsync(batch_strides_dev, batch_strides.get(), sizeof(int64_t) * b,
                  cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(tma_descriptor_start, tma_desc_host.get(), sizeof(tma::cudaTmaDesc) * (3 * b + 2),
                  cudaMemcpyHostToDevice, stream);

  cudnn_sdpa::AttentionDescriptor_t attnDesc{b, h_qo, h_kv, h_kv, s_qo, max_skv, d, h_qo / h_kv};

  cudnn_sdpa::FastDivisor_t page_size_div;
  setFastDivisor(page_size_div, page_size);

  uint32_t page_size32 = static_cast<uint32_t>(page_size);
  uint32_t num_pages_per_seq32 = static_cast<uint32_t>(max_skv / page_size);

  void* args[15];
  tma::cudaTmaDesc* packed_tma_desc_q_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start);
  tma::cudaTmaDesc* packed_tma_desc_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc));
  tma::cudaTmaDesc* packed_tma_desc_partial_o_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 2);
  tma::cudaTmaDesc* tma_desc_k_dev =
      reinterpret_cast<tma::cudaTmaDesc*>(tma_descriptor_start + b * sizeof(tma::cudaTmaDesc) * 3);
  tma::cudaTmaDesc* tma_desc_v_dev = reinterpret_cast<tma::cudaTmaDesc*>(
      tma_descriptor_start + (sizeof(tma::cudaTmaDesc) * ((b * 3) + 1)));
  float attn_scale = scale;
  void* actual_seq_lens_q_gpu_pointer = nullptr;
  void* actual_seq_lens_kv_gpu_pointer = actual_seq_lens_kv_gpu.data_ptr<int32_t>();
  void* block_tables_pointer = block_tables.data_ptr<int32_t>();

  cudnn_sdpa::strides_t lse_strides = {h_qo * s_qo, 1, h_qo, 1};

  args[0] = (void*)&attnDesc;
  args[1] = (void*)&packed_tma_desc_q_dev;
  args[2] = (void*)&tma_desc_k_dev;
  args[3] = (void*)&split_factor;
  args[4] = (void*)&attn_scale;
  args[5] = (void*)&lse_dev;
  args[6] = (void*)&lse_strides;
  args[7] = (void*)&tma_desc_v_dev;
  args[8] = &packed_tma_desc_o_dev;
  args[9] = (void*)&actual_seq_lens_q_gpu_pointer;
  args[10] = (void*)&actual_seq_lens_kv_gpu_pointer;
  args[11] = (void*)&block_tables_pointer;
  args[12] = (void*)&page_size32;
  args[13] = (void*)&num_pages_per_seq32;
  args[14] = (void*)&page_size_div;

  auto err_launch = cuLaunchKernelEx(&config, hfunc_decode[0], (void**)args, nullptr);
  if (err_launch != CUDA_SUCCESS) {
    std::cerr << "cuLaunchKernelEx failed with error code " << err_launch << std::endl;
    throw std::runtime_error("cuLaunchKernelEx failed for decode");
  }
}

}  // namespace cudnn_sdpa_kernel_launcher

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", cudnn_sdpa_kernel_launcher::prefill);
  m.def("decode", cudnn_sdpa_kernel_launcher::decode);
}

}  // namespace flashinfer
