/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_SCHEDULER_CUH_

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../allocator.h"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "heap.h"

namespace flashinfer {

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__
                                                  typename AttentionVariant::ParamsT params);

auto PrefillBinarySearchKVChunkSize(const uint32_t max_batch_size_if_split,
                                    const std::vector<int64_t>& packed_qo_len_arr,
                                    const std::vector<int64_t>& kv_len_arr,
                                    const uint32_t qo_chunk_size,
                                    const uint32_t min_kv_chunk_size = 1) {
  int64_t low = min_kv_chunk_size, high = 0;
  int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }
  high = max_kv_len;
  int64_t new_batch_size;
  while (low < high) {
    int64_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size +=
          ceil_div(packed_qo_len_arr[i], qo_chunk_size) * ceil_div(kv_len_arr[i], mid);
    }
    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                      ceil_div(std::max(int(kv_len_arr[i]), 1), low);
  }
  return std::make_tuple(low < max_kv_len, low, new_batch_size);
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param split_kv Whether to split the KV cache into multiple chunks
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param pos_encoding_mode The positional encoding mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant>
cudaError_t BatchDecodeWithPagedKVCacheNumCTAs(uint32_t& num_ctas) {
  using DTypeKV = typename AttentionVariant::DTypeKV;
  using IdType = typename AttentionVariant::IdType;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, AttentionVariant>;
    int num_ctas_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_ctas = num_ctas_per_sm * num_sm;
  });
  return cudaSuccess;
}

struct DecodePlanInfo {
  int64_t work_indptr_offset;
  int64_t request_indices_offset;
  int64_t kv_indptr_start_offset;
  int64_t kv_indptr_end_offset;
  int64_t kv_head_idx_offset;
  int64_t merge_indptr_offset;
  int64_t merge_indices_offset;
  int64_t v_offset;
  int64_t s_offset;
  int num_ctas;

  DecodePlanInfo()
      : work_indptr_offset(0),
        request_indices_offset(0),
        kv_indptr_start_offset(0),
        kv_indptr_end_offset(0),
        kv_head_idx_offset(0),
        merge_indptr_offset(0),
        merge_indices_offset(0),
        v_offset(0),
        s_offset(0),
        num_ctas(0) {}

  // convert DecodePlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {work_indptr_offset,
            request_indices_offset,
            kv_indptr_start_offset,
            kv_indptr_end_offset,
            kv_head_idx_offset,
            merge_indptr_offset,
            merge_indices_offset,
            v_offset,
            s_offset,
            num_ctas};
  }

  // From std::vector<int64_t> to DecodePlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 10) {
      std::ostringstream err_msg;
      err_msg << "DecodePlanInfo::FromVector: vec.size() should be 10, but got " << vec.size();
      throw std::invalid_argument(err_msg.str());
    }
    work_indptr_offset = vec[0];
    request_indices_offset = vec[1];
    kv_indptr_start_offset = vec[2];
    kv_indptr_end_offset = vec[3];
    kv_head_idx_offset = vec[4];
    merge_indptr_offset = vec[5];
    merge_indices_offset = vec[6];
    v_offset = vec[7];
    s_offset = vec[8];
    num_ctas = vec[9];
  }
};

float cost_function(int kv_len, int page_size, int group_size) {
  return float(group_size) / float(page_size) + kv_len;
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec, int size_after_flatten) {
  std::vector<T> result;
  result.reserve(size_after_flatten);
  for (const auto& inner_vec : vec) {
    result.insert(result.end(), inner_vec.begin(), inner_vec.end());
  }
  return std::move(result);
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant>
cudaError_t DecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                       void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
                       DecodePlanInfo& plan_info, typename AttentionVariant::IdType* indptr_h,
                       uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                       uint32_t page_size, bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeO = typename AttentionVariant::DTypeO;
  using IdType = typename AttentionVariant::IdType;
  uint32_t num_ctas;
  uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  DISPATCH_GQA_GROUP_SIZE(gqa_group_size, GROUP_SIZE, {
    auto get_num_ctas_func =
        BatchDecodeWithPagedKVCacheNumCTAs<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE,
                                           AttentionVariant>;
    FLASHINFER_CUDA_CALL(get_num_ctas_func(num_ctas));
  });
  plan_info.num_ctas = num_ctas;

  MinHeap heap(num_ctas);

  std::vector<int64_t> kv_len_vec(batch_size);
  float total_cost = 0.f;
  for (uint32_t i = 0; i < batch_size; ++i) {
    kv_len_vec[i] = indptr_h[i + 1] - indptr_h[i];
    if (kv_len_vec[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "indptr[" << i + 1 + batch_size << "]" << indptr_h[i + 1 + batch_size]
              << " - indptr[" << i + batch_size << "]" << indptr_h[i + batch_size]
              << " should be non-negative";
      throw std::invalid_argument(err_msg.str());
    }
    total_cost += cost_function(kv_len_vec[i], page_size, gqa_group_size);
  }
  total_cost = total_cost * float(num_kv_heads);
  float bucket_cost_limit = total_cost / float(num_ctas);

  std::vector<std::vector<IdType>> cta_request_indices(num_ctas, std::vector<IdType>()),
      cta_kv_indptr_start(num_ctas, std::vector<IdType>()),
      cta_kv_indptr_end(num_ctas, std::vector<IdType>()),
      cta_kv_head_idx(num_ctas, std::vector<IdType>());

  for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      int64_t remaining_len = kv_len_vec[i];
      while (remaining_len > 0) {
        auto [cta_idx, accum_cost] = heap.pop();
        int64_t actual_len =
            std::min(remaining_len, int64_t(std::ceil(bucket_cost_limit - accum_cost)));
        heap.insert({cta_idx, accum_cost + cost_function(actual_len, page_size, gqa_group_size)});
        cta_request_indices[cta_idx].push_back(i);
        cta_kv_indptr_start[cta_idx].push_back(kv_len_vec[i] - remaining_len);
        cta_kv_indptr_end[cta_idx].push_back(kv_len_vec[i] - remaining_len + actual_len);
        cta_kv_head_idx[cta_idx].push_back(kv_head_idx);
        remaining_len -= actual_len;
      }
    }
  }

  std::vector<IdType> work_indptr_vec(num_ctas + 1, 0);
  for (uint32_t i = 0; i < num_ctas; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cta_request_indices[i].size();
  }
  IdType total_num_works = work_indptr_vec[num_ctas];
  auto request_indices_vec = flatten(cta_request_indices, total_num_works);
  auto kv_indptr_start_vec = flatten(cta_kv_indptr_start, total_num_works);
  auto kv_indptr_end_vec = flatten(cta_kv_indptr_end, total_num_works);
  auto kv_head_idx_vec = flatten(cta_kv_head_idx, total_num_works);
  std::vector<std::vector<IdType>> local_merge_indices(batch_size * num_qo_heads,
                                                       std::vector<IdType>());
  std::vector<IdType> merge_indptr_vec(batch_size * num_qo_heads + 1, 0);
  for (uint32_t work_iter = 0; work_iter < total_num_works; ++work_iter) {
    int kv_head_idx = kv_head_idx_vec[work_iter];
    int batch_idx = request_indices_vec[work_iter];
    for (uint32_t i = 0; i < gqa_group_size; ++i) {
      int qo_head_idx = kv_head_idx * gqa_group_size + i;
      local_merge_indices[batch_idx * num_qo_heads + qo_head_idx].push_back(
          work_iter * gqa_group_size + i);
    }
  }
  for (uint32_t i = 0; i < batch_size * num_qo_heads + 1; ++i) {
    merge_indptr_vec[i + 1] = merge_indptr_vec[i] + local_merge_indices[i].size();
  }
  auto merge_indices_vec = flatten(local_merge_indices, merge_indptr_vec.back());

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);

  plan_info.work_indptr_offset =
      int_allocator.aligned_alloc_offset(16384 * sizeof(IdType), 16, "batch_decode_work_indptr");
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      131072 * sizeof(IdType), 16, "batch_decode_request_indices");
  plan_info.kv_indptr_start_offset = int_allocator.aligned_alloc_offset(
      131072 * sizeof(IdType), 16, "batch_decode_kv_indptr_start");
  plan_info.kv_indptr_end_offset =
      int_allocator.aligned_alloc_offset(131072 * sizeof(IdType), 16, "batch_decode_kv_indptr_end");
  plan_info.kv_head_idx_offset =
      int_allocator.aligned_alloc_offset(131072 * sizeof(IdType), 16, "batch_decode_kv_head_idx");
  plan_info.merge_indptr_offset =
      int_allocator.aligned_alloc_offset(131072 * sizeof(IdType), 16, "batch_decode_merge_indptr");
  plan_info.merge_indices_offset =
      int_allocator.aligned_alloc_offset(131072 * sizeof(IdType), 16, "batch_decode_merge_indices");

  IdType* work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);
  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* kv_indptr_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_start_offset);
  IdType* kv_indptr_end_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_end_offset);
  IdType* kv_head_idx_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_head_idx_offset);
  IdType* merge_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
  IdType* merge_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indices_offset);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), work_indptr_h);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(kv_indptr_start_vec.begin(), kv_indptr_start_vec.end(), kv_indptr_start_h);
  std::copy(kv_indptr_end_vec.begin(), kv_indptr_end_vec.end(), kv_indptr_end_h);
  std::copy(kv_head_idx_vec.begin(), kv_head_idx_vec.end(), kv_head_idx_h);
  std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
  std::copy(merge_indices_vec.begin(), merge_indices_vec.end(), merge_indices_h);

  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.v_offset = float_allocator.aligned_alloc_offset(
      num_qo_heads * 8192 * HEAD_DIM * sizeof(DTypeO), 16, "batch_decode_tmp_v");
  plan_info.s_offset = float_allocator.aligned_alloc_offset(num_qo_heads * 8192 * sizeof(float), 16,
                                                            "batch_decode_tmp_s");

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

template <typename IdType>
auto PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t batch_size,
                            uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                            uint32_t page_size, uint32_t max_batch_size_if_split,
                            bool enable_cuda_graph) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  uint32_t total_num_rows = qo_indptr_h[batch_size];

  // step 1: compute qo_chunk_size
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  int64_t sum_packed_qo_len = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    if (packed_qo_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      throw std::invalid_argument(err_msg.str());
    }
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    if (kv_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_indptr[" << i + 1 << "]" << kv_indptr_h[i + 1] << " - kv_indptr[" << i << "]"
              << kv_indptr_h[i] << " should be non-negative";
      throw std::invalid_argument(err_msg.str());
    }
    sum_packed_qo_len += packed_qo_len_arr[i];
  }
  int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
  uint32_t cta_tile_q;
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    cta_tile_q = 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        // avg_packed_qo_len <= 64
        cta_tile_q = 64;
      } else {
        // avg_packed_qo_len <= 16
        cta_tile_q = 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
      cta_tile_q = 64;
    }
  }

  uint32_t total_num_tiles_q = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    total_num_tiles_q += ceil_div(packed_qo_len_arr[request_idx], cta_tile_q);
  }

  // step 2: determine kv_chunk_size
  auto [split_kv, kv_chunk_size, new_batch_size] = PrefillBinarySearchKVChunkSize(
      max_batch_size_if_split, packed_qo_len_arr, kv_len_arr, cta_tile_q,
      /*min_kv_chunk_size=*/std::max((128 / page_size), 1U));

  if (enable_cuda_graph) {
    split_kv = total_num_tiles_q < max_batch_size_if_split;
  }

  // step 3: split qo_indptr and kv_indptr
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int64_t packed_qo_len = packed_qo_len_arr[request_idx],
            kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q),
            num_tiles_kv = ceil_div(kv_len, kv_chunk_size);
    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return std::make_tuple(split_kv, total_num_tiles_q, new_batch_size, cta_tile_q, kv_chunk_size,
                         total_num_rows, std::move(request_indices), std::move(qo_tile_indices),
                         std::move(kv_tile_indices), std::move(merge_indptr), std::move(o_indptr));
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 14) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 14, but got " << vec.size();
      throw std::invalid_argument(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    cta_tile_q = vec[2];
    request_indices_offset = vec[3];
    qo_tile_indices_offset = vec[4];
    kv_tile_indices_offset = vec[5];
    merge_indptr_offset = vec[6];
    o_indptr_offset = vec[7];
    kv_chunk_size_ptr_offset = vec[8];
    v_offset = vec[9];
    s_offset = vec[10];
    block_valid_mask_offset = vec[11];
    enable_cuda_graph = vec[12];
    split_kv = vec[13];
  }
};

template <typename IdType>
cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                        void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
                        PrefillPlanInfo& plan_info, IdType* qo_indptr_h, IdType* kv_indptr_h,
                        uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                        uint32_t head_dim, uint32_t page_size, bool enable_cuda_graph,
                        uint32_t sizeof_dtype_o, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, total_num_tiles_q, new_batch_size, cta_tile_q, kv_chunk_size, total_num_rows,
        request_indices_vec, qo_tile_indices_vec, kv_tile_indices_vec, merge_indptr_vec,
        o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, batch_size, num_qo_heads, num_kv_heads,
                             head_dim, page_size, max_batch_size_if_split, enable_cuda_graph);
  plan_info.cta_tile_q = cta_tile_q;
  plan_info.total_num_rows = total_num_rows;

  plan_info.enable_cuda_graph = enable_cuda_graph;
  size_t padded_batch_size =
      enable_cuda_graph ? std::max(max_batch_size_if_split, total_num_tiles_q) : new_batch_size;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.split_kv = split_kv;

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_request_indices");
  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_qo_tile_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * (batch_size + 1),
                                                                 16, "batch_prefill_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");

  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim * sizeof_dtype_o, 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float), 16, "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");
    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_SCHEDULER_CUH_
