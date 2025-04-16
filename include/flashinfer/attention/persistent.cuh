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

#include "persistent_template.cuh"
#include "prefill.cuh"

namespace flashinfer {

template <typename Params>
__device__ __forceinline__ auto get_block_coord(const Params& params, const uint32_t work_idx) {
  return std::tuple(params.batch_indices[work_idx], params.q_indptr[work_idx],
                    params.kv_indptr[work_idx], params.partial_indptr[work_idx],
                    params.q_len[work_idx], params.kv_len[work_idx], params.q_start[work_idx],
                    params.kv_start[work_idx], params.kv_end[work_idx],
                    params.kv_head_idx_arr[work_idx]);
}

template <typename KTraits>
__device__ __forceinline__ void prefetch_offest(
    const uint32_t packed_block_iter_base, const uint32_t packed_kv_bound,
    const uint32_t kv_head_idx, const uint32_t kv_stride_page, const uint32_t kv_stride_h,
    const uint32_t kv_stride_n, const uint_fastdiv& block_size, typename KTraits::IdType* indices,
    size_t* kv_offset) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
  constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  const uint32_t lane_idx = threadIdx.x % 32, warp_idx = threadIdx.x / 32;

#pragma unroll
  for (uint32_t i = 0;
       i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
    uint32_t page_iter, entry_idx;
    uint32_t packed_block_iter = packed_block_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                 lane_idx / KV_THR_LAYOUT_COL +
                                 KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i;
    block_size.divmod(packed_block_iter, page_iter, entry_idx);
    kv_offset[i] = (packed_block_iter < packed_kv_bound ? indices[page_iter] : 0) * kv_stride_page +
                   entry_idx * kv_stride_n + kv_head_idx * kv_stride_h +
                   (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
  }
}

template <typename KTraits_, typename Params_>
struct BlockBatchPagedAttentionPersistent {
  using KTraits = KTraits_;
  using Params = Params_;

  static __device__ __forceinline__ void Run(const Params& params,
                                             typename KTraits::SharedStorage* smem_storage) {
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
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t v_stride_page = params.v_stride_page;
    const uint32_t v_stride_h = params.v_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t o_stride_n = params.o_stride_n;
    const uint32_t o_stride_h = params.o_stride_h;
    const uint32_t cluster_tile_q = gridDim.x * CTA_TILE_Q;
    smem_t<SWIZZLE_MODE_Q> q_smem(smem_storage->q_smem);

    AttentionVariant variant(params, /*batch_idx=*/0, nullptr);

    const uint32_t lane_idx = threadIdx.x % 32;
    const uint32_t warp_idx = threadIdx.x / 32;

    uint32_t q_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_Q, UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(warp_idx) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);
    uint32_t k_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(warp_idx) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(warp_idx) * NUM_MMA_KV * 16 + lane_idx % 16,
                 lane_idx / 16);
    uint32_t k_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_K>(
                 warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
                 lane_idx % KTraits::KV_THR_LAYOUT_COL),
             v_smem_offset_w = get_permuted_offset<SWIZZLE_MODE_KV, UPCAST_STRIDE_V>(
                 warp_idx * KTraits::KV_THR_LAYOUT_ROW + lane_idx / KTraits::KV_THR_LAYOUT_COL,
                 lane_idx % KTraits::KV_THR_LAYOUT_COL);
    size_t thr_local_kv_offset[NUM_MMA_KV * KTraits::KV_THR_LAYOUT_COL / 2 / KTraits::NUM_WARPS_Q];

#pragma unroll 1
    for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
         ++work_idx) {
      const auto [batch_idx, q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start,
                  kv_start, kv_end, kv_head_idx] = get_block_coord(params, work_idx);

      const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * CTA_TILE_Q +
                                          get_warp_idx_q<KTraits>(warp_idx) * NUM_MMA_Q * 16;
      const uint32_t qo_upperbound =
          min(q_len, ceil_div(qo_packed_idx_base + CTA_TILE_Q, gqa_group_size));

      init_states<KTraits>(variant, o_frag, m, d);

      DTypeQ* q_ptr_base = q + q_indptr * q_stride_n + (kv_head_idx * gqa_group_size) * q_stride_h;

      // load_q
      load_q_global_smem<KTraits>(smem_storage, qo_packed_idx_base, qo_upperbound, q_ptr_base,
                                  q_stride_n, q_stride_h, gqa_group_size);

      smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage->k_smem), v_smem(smem_storage->v_smem);
      int kv_tile_idx =
          ceil_div((CAUSAL ? min(kv_end, kv_len - q_len +
                                             (packed_qo_start + cluster_tile_q) / gqa_group_size)
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

      prefetch_offest<KTraits>(block_iter_base + kv_tile_idx * CTA_TILE_KV, packed_kv_bound,
                               kv_head_idx, k_stride_page, k_stride_h, k_stride_n, block_size,
                               kv_indices, thr_local_kv_offset);
      page_load_kv<false, KTraits>(smem_storage, &k_smem_offset_w, k,
                                   kv_start + kv_tile_idx * CTA_TILE_KV, thr_local_kv_offset,
                                   kv_end);
      cp_async::commit_group();
      page_load_kv<true, KTraits>(smem_storage, &v_smem_offset_w, v,
                                  kv_start + kv_tile_idx * CTA_TILE_KV, thr_local_kv_offset,
                                  kv_end);
      cp_async::commit_group();

      // loop with mask
      LOOP_SPLIT_MASK(
          kv_tile_idx, kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0,
          kv_tile_idx + 1 > NUM_STAGES, {
            prefetch_offest<KTraits>(block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV,
                                     packed_kv_bound, kv_head_idx, k_stride_page, k_stride_h,
                                     k_stride_n, block_size, kv_indices, thr_local_kv_offset);
            cp_async::wait_group<1>();
            __syncthreads();

            gemm_qk<KTraits>(&q_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
            if constexpr (WITH_MASK) {
              logits_mask<KTraits>(
                  params, variant, batch_idx, qo_packed_idx_base,
                  kv_start + (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(warp_idx)) *
                                 NUM_MMA_KV * 16,
                  q_len, kv_len, kv_end, gqa_group_size, s_frag, kv_head_idx);
            }
            update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

            __syncthreads();
            page_load_kv<false, KTraits>(smem_storage, &k_smem_offset_w, k,
                                         kv_start + (kv_tile_idx - 1) * CTA_TILE_KV,
                                         thr_local_kv_offset, kv_end);
            cp_async::commit_group();
            cp_async::wait_group<1>();

            __syncthreads();
            gemm_pv<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
            __syncthreads();

            page_load_kv<true, KTraits>(smem_storage, &v_smem_offset_w, v,
                                        kv_start + (kv_tile_idx - 1) * CTA_TILE_KV,
                                        thr_local_kv_offset, kv_end);
            cp_async::commit_group();
          });
      cp_async::wait_group<0>();
      __syncthreads();

#pragma unroll
      for (; kv_tile_idx >= 0; --kv_tile_idx) {
        gemm_qk<KTraits>(&q_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
        logits_mask<KTraits>(
            params, variant, batch_idx, qo_packed_idx_base,
            kv_start +
                (kv_tile_idx * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(warp_idx)) * NUM_MMA_KV * 16,
            q_len, kv_len, kv_end, gqa_group_size, s_frag, kv_head_idx);
        update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);
        gemm_pv<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
      }

      __syncthreads();

      finalize_m<KTraits>(variant, m);

      // threadblock synchronization
      threadblock_allreduce<KTraits>(o_frag, smem_storage, m, d);

      // normalize d
      normalize_d<KTraits>(o_frag, m, d);

      // write back to global memory
      // NOTE(Zihao): use new write back
      DTypeO* o_ptr_base =
          final_o + q_indptr * o_stride_n + (kv_head_idx * gqa_group_size) * o_stride_h;
      write_o_reg_gmem<KTraits>(o_frag, &q_smem, o_ptr_base, qo_packed_idx_base, qo_upperbound,
                                o_stride_n, o_stride_h, gqa_group_size);
    }
  }
};

template <uint32_t CTA_TILE_Q_1, uint32_t CTA_TILE_Q_2, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t BatchPagedAttentionPersistent(const Params params_1, const Params params_2,
                                          const uint32_t num_blks_x, const uint32_t num_blks_y,
                                          const cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  constexpr uint32_t NUM_WARPS_Q_1 = get_num_warps_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_WARPS_KV_1 = get_num_warps_kv(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_Q_1 = get_num_mma_q(CTA_TILE_Q_1);
  constexpr uint32_t NUM_MMA_KV_1 = 4;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using KTraits1 = KernelTraits<MASK_MODE, CTA_TILE_Q_1, NUM_MMA_Q_1, NUM_MMA_KV_1, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_1, NUM_WARPS_KV_1, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;
  constexpr uint32_t NUM_WARPS_Q_2 = get_num_warps_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_WARPS_KV_2 = get_num_warps_kv(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_Q_2 = get_num_mma_q(CTA_TILE_Q_2);
  constexpr uint32_t NUM_MMA_KV_2 = 2;
  using KTraits2 = KernelTraits<MASK_MODE, CTA_TILE_Q_2, NUM_MMA_Q_2, NUM_MMA_KV_2, NUM_MMA_D_QK,
                                NUM_MMA_D_VO, NUM_WARPS_Q_2, NUM_WARPS_KV_2, PosEncodingMode::kNone,
                                DTypeQ, DTypeKV, DTypeO, float, IdType, AttentionVariant>;

  size_t smem_size =
      max(sizeof(typename KTraits1::SharedStorage), sizeof(typename KTraits2::SharedStorage));
  auto kernel = PersistentKernelTemplate<BlockBatchPagedAttentionPersistent<KTraits1, Params>,
                                         BlockBatchPagedAttentionPersistent<KTraits2, Params>>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(max(KTraits1::NUM_THREADS, KTraits2::NUM_THREADS));
  void* args[] = {(void*)&params_1, (void*)&params_2};

  FLASHINFER_CUDA_CALL(
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

  return cudaSuccess;
}

template <typename Params, uint32_t NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void BatchAttentionScoreReductionPersisitentKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  using DTypeIn = typename Params::DTypeO;
  using DTypeOut = typename Params::DTypeO;
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

template <typename Params>
cudaError_t BatchAttentionScoreReductionPersisitent(const Params params, const uint32_t num_blks_x,
                                                    const uint32_t num_blks_y,
                                                    const cudaStream_t stream) {
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  constexpr uint32_t NUM_THREADS = 128;
  auto kernel = BatchAttentionScoreReductionPersisitentKernel<Params, NUM_THREADS>;

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

};  // namespace flashinfer

#endif  // FLASHINFER_PERSISTENT_CUH_
