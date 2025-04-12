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
                    params.kv_end[work_idx], params.kv_head_idx_arr[work_idx]);
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BlockBatchPagedAttentionPersistent(
    const Params& params, typename KTraits::SharedStorage* smem_storage) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
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
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::MASK_MODE == MaskMode::kCausal;
  [[maybe_unused]] constexpr uint32_t NUM_STAGES = KTraits::NUM_STAGES;

  DTypeQ* q = params.q;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
  float m[NUM_MMA_Q][2];
  float d[NUM_MMA_Q][2];

  const uint_fastdiv& gqa_group_size = params.gqa_group_size;
  const uint_fastdiv& block_size = params.page_size;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_page = params.k_stride_page;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t v_stride_page = params.v_stride_page;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;
  smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage->q_smem);

  AttentionVariant variant(params, /*batch_idx=*/0, nullptr);

  const uint32_t lane_idx = threadIdx.x % 32;
  const uint32_t warp_idx = threadIdx.x / 32;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    const auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start,
                kv_end, kv_head_idx] = get_block_coord(params, work_idx);

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q;
    const uint32_t qo_upperbound =
        min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));

    init_states<KTraits>(variant, o_frag, m, d);

    DTypeQ* q_ptr_base = q + q_indptr * q_stride_n + (kv_head_idx * gqa_group_size) * q_stride_h;
    // load_q
    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upperbound, q_ptr_base, q_stride_n,
                                q_stride_h, gqa_group_size, &qo_smem, threadIdx);

    int kv_tile_idx =
        ceil_div((CAUSAL ? min(kv_end,
                               kv_len - q_len + (packed_qo_start + cluster_tile_q) / gqa_group_size)
                         : kv_end),
                 CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / gqa_group_size) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    // last kv tile
    __syncthreads();
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;

    cp_async::commit_group();
#pragma unroll
    for (int stage_idx = 1; stage_idx < NUM_STAGES; ++stage_idx) {
      if (kv_tile_idx - stage_idx >= 0) {
        // load_kv<false, KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n,
        // ckv_stride_page,
        //                         kpe_stride_n, kpe_stride_page, packed_kv_bound,
        //                         block_iter_base + (kv_tile_idx - stage_idx) * CTA_TILE_KV,
        //                         block_size, (kv_tile_idx - stage_idx) % NUM_STAGES);
        cp_async::commit_group();
      }
    }

    // loop with mask
    LOOP_SPLIT_MASK(kv_tile_idx, kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0,
                    kv_tile_idx + 1 > NUM_STAGES,
                    {

                    });

#pragma unroll
    for (; kv_tile_idx >= 0; --kv_tile_idx) {
    }

    __syncthreads();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    // threadblock_allreduce<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, threadIdx);

    // normalize d
    normalize_d<KTraits>(o_frag, m, d);

    // write back to global memory
  }
}

template <typename KTraits1, typename KTraits2, typename Params>
__global__ __launch_bounds__(std::max(
    KTraits1::NUM_THREADS,
    KTraits2::NUM_THREADS)) void BatchPagedAttentionPersistentHolisticKernel(const __grid_constant__
                                                                                 Params params_1,
                                                                             const __grid_constant__
                                                                                 Params params_2) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage_1 = reinterpret_cast<typename KTraits1::SharedStorage&>(smem);
  BlockBatchPagedAttentionPersistent<KTraits1>(params_1, &smem_storage_1);
  auto& smem_storage_2 = reinterpret_cast<typename KTraits2::SharedStorage&>(smem);
  BlockBatchPagedAttentionPersistent<KTraits2>(params_2, &smem_storage_2);
}

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t BatchPagedAttentionPersistentHolistic(const Params params_1, const Params params_2,
                                                  const uint32_t num_blks_x,
                                                  const uint32_t num_blks_y,
                                                  const cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  constexpr uint32_t NUM_WARPS_Q_1 = get_num_warps_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_WARPS_KV_1 = get_num_warps_kv(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_Q_1 = get_num_mma_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_KV_1 = 2;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using KTraits1 = KernelTraits<MASK_MODE, CTA_TILE_Q_1, NUM_MMA_Q_1, NUM_MMA_KV_1, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_1, NUM_WARPS_KV_1, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;
  constexpr uint32_t NUM_WARPS_Q_2 = get_num_warps_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_WARPS_KV_2 = get_num_warps_kv(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_Q_2 = get_num_mma_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_KV_2 = 8;
  using KTraits2 = KernelTraits<MASK_MODE, CTA_TILE_Q_2, NUM_MMA_Q_2, NUM_MMA_KV_2, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_2, NUM_WARPS_KV_2, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;

  size_t smem_size =
      max(sizeof(typename KTraits1::SharedStorage), sizeof(typename KTraits2::SharedStorage));
  auto kernel = BatchPagedAttentionPersistentHolisticKernel<KTraits1, KTraits2, Params>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(max(KTraits1::NUM_THREADS, KTraits2::NUM_THREADS));
  void* args[] = {(void*)&params_1, (void*)&params_2};

  FLASHINFER_CUDA_CALL(
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

  return cudaSuccess;
}

template <typename Params>
cudaError_t BatchAttentionScoreReductionPersisitent(const Params params, const uint32_t num_blks_x,
                                                    const uint32_t num_blks_y,
                                                    const cudaStream_t stream) {
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  constexpr uint32_t NUM_THREADS = 128;
  auto kernel = BatchPagedAttentionPersistentHolisticKernel<Params, NUM_THREADS>;

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(NUM_THREADS);
  void* args[] = {(void*)&params};

  size_t smem_size = 16 * 1024;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  FLASHINFER_CUDA_CALL(
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

template <typename Params, uint32_t NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void BatchPagedAttentionPersistentHolisticKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  using DTypeIn = Params::DTypeO;
  using DTypeOut = Params::DTypeO;
  using DTypeAccum = float;
  using IdType = typename Params::IdType;

  // [max_batch_size, num_kv_heads, max_seqlen]
  // gqa_group_size is packed into first dim
  DTypeIn* qk_in_ptr = params.qk_ptr;
  const uint32_t qk_stride = params.qk_max_seqlen;

  // [max_batch_size, num_kv_heads, max_seqlen]
  DTypeOut* reduced_o_ptr = params.reduced_o_ptr;
  const uint32_t reduced_o_stride = params.qk_stride;

  // [nnz, num_qo_heads]
  float* lse = params.lse;
  IdType* work_indptr = params.work_indptr;

  const uint_fastdiv& gqa_group_size = params.gqa_group_size;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t num_qo_heads = num_kv_heads * gqa_group_size;
  const uint32_t lane_idx = threadIdx.x;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    const auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start,
                kv_end, kv_head_idx] = get_block_coord(params, work_idx);
    if (packed_qo_start != 0) {
      // the first block takes care of all qo_len
      // for minimal code changes
      continue;
    }
    // else packed_qo_start == 0
    const auto o_indptr = params.o_indptr[work_idx];
    const auto qk_indptr = params.qk_indptr[work_idx];  // packed
    DTypeIn* qk_ptr_base = qk_in_ptr + ((qk_indptr * num_kv_heads) + kv_head_idx) * qk_stride;
    DTypeOut* o_ptr_base =
        reduced_o_ptr + ((o_indptr * num_kv_heads) + kv_head_idx) * reduced_o_stride;
    float* lse_base = lse + q_indptr * num_qo_heads;

    // reduction kernel:
    // each threadblock read qk_ptr_base[0:q_len*gqa_group_size, kv_head_idx, kv_start:kv_end]
    // do a reduction into a tensor with shape (1,1,kv_start:kv_end)
    // and write it back to o_ptr_base[0, kv_head_idx, kv_start:kv_end]
    {
      const uint32_t total_rows = q_len * gqa_group_size;
      for (uint32_t kv_idx = kv_start + lane_idx; kv_idx < kv_end; kv_idx += NUM_THREADS) {
        DTypeAccum sum = 0.0f;
        for (uint32_t i = 0; i < total_rows; ++i) {
          DTypeAccum cur_lse = static_cast<DTypeAccum>(lse_base[i]);
          sum += cur_lse * static_cast<DTypeAccum>(
                               qk_ptr_base[(i * num_kv_heads + kv_head_idx) * qk_stride + kv_idx]);
        }
        o_ptr_base[kv_idx] = static_cast<DTypeOut>(sum);
      }
    }
  }
}
};  // namespace flashinfer

#endif  // FLASHINFER_PERSISTENT_CUH_
