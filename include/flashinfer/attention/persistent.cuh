/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_PERSISTENT_CUH_
#define FLASHINFER_PERSISTENT_CUH_

#include "prefill.cuh"

namespace flashinfer {

template <typename Params>
__device__ __forceinline__ auto get_block_coord(const Params& params, const uint32_t work_idx) {
  return std::tuple(params.q_indptr[work_idx], params.kv_indptr[work_idx],
                    params.partial_indptr[work_idx], params.q_len[work_idx],
                    params.kv_len[work_idx], params.q_start[work_idx], params.kv_start[work_idx],
                    params.kv_end[work_idx]);
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BlockBatchPagedAttentionPersistent(
    const Params& params, typename KTraits::SharedStorage* smem_storage) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;
  [[maybe_unused]] constexpr uint32_t NUM_STAGES = KTraits::NUM_STAGES;

  DTypeQ* q = params.q;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  IdType* kv_indices = params.kv_indices;
  IdType* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[NUM_MMA_KV][8];
  alignas(16) float o_frag[NUM_MMA_D_VO][8];
  float m[2];
  float d[2];

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_page = params.k_stride_page;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t v_stride_page = params.v_stride_page;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    const auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start,
                kv_end] = get_block_coord(params, work_idx);

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q;
    const uint32_t qo_upperbound = min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, num_heads));

    init_states<KTraits>(o_frag, m, d);

    __syncthreads();
    // load_q

    int kv_tile_idx =
        ceil_div(
            (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                    : kv_end),
            CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    // last kv tile
    __syncthreads();
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;

    // load_kv
    cp_async::commit_group();
#pragma unroll
    for (int stage_idx = 1; stage_idx < NUM_STAGES; ++stage_idx) {
      if (kv_tile_idx - stage_idx >= 0) {
        // load_kv
        cp_async::commit_group();
      }
    }
  }
}

template <typename KTraits1, typename KTraits2, typename Params>
__global__ __launch_bounds__(128) void BatchPagedAttentionPersistentHolisticKernel(
    const __grid_constant__ Params params_1, const __grid_constant__ Params params_2) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage_1 = reinterpret_cast<typename KTraits1::SharedStorage&>(smem);
  BlockBatchPagedAttentionPersistent<KTraits1>(params_1, &smem_storage_1);
  auto& smem_storage_2 = reinterpret_cast<typename KTraits2::SharedStorage&>(smem);
  BlockBatchPagedAttentionPersistent<KTraits2>(params_2, &smem_storage_2);
}

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
MaskMode MASK_MODE,
typename AttentionVariant,
typename Params>
cudaError_t BatchPagedAttentionPersistentHolistic(
    const Params params_1, const Params params_2, const uint32_t num_blks_x,
    const uint32_t num_blks_y, const cudaStream_t stream) {

}

};  // namespace flashinfer

#endif  // FLASHINFER_PERSISTENT_CUH_
