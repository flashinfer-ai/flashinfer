/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
#define FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "variants.cuh"

namespace flashinfer {

template <typename Ktraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, bool MULTIITEMSCORING,
          typename WarpScheduler, typename AttentionVariant, typename Params,
          typename MainloopPipeline, typename PipelineState, typename SharedStorage,
          typename FrgTensorO, typename AttentionUpdater>
CUTLASS_DEVICE void mma_f16(
    const Params& mainloop_params, AttentionVariant& variant, MainloopPipeline pipeline_k,
    MainloopPipeline pipeline_v, PipelineState& smem_pipe_read_k, PipelineState& smem_pipe_read_v,
    FrgTensorO& tOrO, AttentionUpdater& attention_updater, int kv_tile_idx_count,
    int swa_begin_kv_tile_idx, int swa_end_kv_tile_idx, int thread_idx, int work_idx,
    int q_tile_idx, SharedStorage& shared_storage, const int32_t qo_len, const int32_t kv_len,
    const int32_t qo_head_idx, const int32_t kv_head_idx, const uint32_t prefix_len,
    uint16_t* token_pos_in_items, const int num_kv_tiles_outside_items_window = 0,
    const int num_kv_tiles_prefix = 0) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  typename Ktraits::TiledMmaPV tiled_mma_pv;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  auto threadMmaPV = tiled_mma_pv.get_thread_slice(thread_idx);

  Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
  Tensor tSrK = threadMmaQK.partition_fragment_B(sK);
  Tensor tOrV = threadMmaPV.partition_fragment_B(sVt);

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
  int kv_tile_idx = kv_tile_idx_count - 1;

  cutlass::ConsumerToken barrier_token =
      static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
  if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
    shared_storage.barrier_Q.wait(work_idx % 2);
  }

  Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
  consumer_wait(pipeline_k, smem_pipe_read_k);

  WarpScheduler::barrier_sync();
  gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()),
                                      tSrS);
  WarpScheduler::barrier_arrive();

  if (work_idx != 0) {
    int lane_predicate = cute::elect_one_sync();
    if (cutlass::canonical_warp_idx_sync() == Ktraits::NUM_WARPS - 1 && lane_predicate) {
#pragma unroll
      for (uint32_t cta_id = 0; cta_id < 1; ++cta_id) {
        shared_storage.barrier_O.arrive(cta_id, lane_predicate);
      }
    }
  }
  warpgroup_wait<0>();
  pipeline_k.consumer_release(smem_pipe_read_k);
  ++smem_pipe_read_k;

  auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
  auto col_limit_left = [&](int qo_idx) {
    return qo_idx + kv_len - qo_len - mainloop_params.window_left;
  };
  auto mask_multi_item_scoring = [&](decltype(tSrS)& tSrS, int i, int qo_idx, int kv_idx) {
    const uint32_t idx_in_original_seq = qo_idx + kv_len - qo_len;
    const bool out_of_boundary =
        kv_idx > idx_in_original_seq || (kv_idx >= std::min(kv_len, col_limit_right(qo_idx)));
    const bool is_prefix = idx_in_original_seq < prefix_len;
    uint16_t token_pos_in_items_regs = 0;
    // Only access idx_in_original_seq >= prefix_len && idx_in_original_seq < kv_len to avoid
    // out-of-bounds memory access
    if (idx_in_original_seq >= prefix_len & idx_in_original_seq < kv_len) {
      token_pos_in_items_regs = __ldca(token_pos_in_items + idx_in_original_seq - prefix_len);
    }
    if (out_of_boundary || is_prefix) {
      tSrS(i) = out_of_boundary ? (AttentionUpdater::fill_value) : tSrS(i);
    } else {
      tSrS(i) = (kv_idx < prefix_len | (idx_in_original_seq < kv_idx + token_pos_in_items_regs))
                    ? tSrS(i)
                    : (AttentionUpdater::fill_value);
    }
  };
  auto mask_multi_item_scoring_assume_in_bound = [&](decltype(tSrS)& tSrS, int i, int qo_idx,
                                                     int kv_idx) {
    const uint32_t idx_in_original_seq = qo_idx + kv_len - qo_len;
    const bool is_prefix = idx_in_original_seq < prefix_len;
    if (is_prefix) {
      tSrS(i) = AttentionUpdater::fill_value;
    } else {
      uint16_t token_pos_in_items_regs = 0;
      // Only access idx_in_original_seq >= prefix_len && idx_in_original_seq < kv_len to avoid
      // out-of-bounds memory access
      if (idx_in_original_seq >= prefix_len & idx_in_original_seq < kv_len) {
        token_pos_in_items_regs = __ldca(token_pos_in_items + idx_in_original_seq - prefix_len);
      }

      tSrS(i) = (kv_idx < prefix_len | (idx_in_original_seq < kv_idx + token_pos_in_items_regs))
                    ? tSrS(i)
                    : (AttentionUpdater::fill_value);
    }
  };
  auto kv_tile_idx_decrement = [&](int kv_tile_idx) {
    int result = kv_tile_idx - 1;
    if constexpr (MULTIITEMSCORING) {
      if ((kv_tile_idx == num_kv_tiles_outside_items_window) &
          (kv_tile_idx >= num_kv_tiles_prefix)) {
        result = num_kv_tiles_prefix - 1;
      }
    }
    return result;
  };
  {
    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
    Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
      int kv_idx = get<1>(tScS(i)) + kv_tile_idx * CTA_KV;
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/0, qo_idx, kv_idx,
                                        qo_head_idx, kv_head_idx);
      if constexpr (MULTIITEMSCORING) {
        mask_multi_item_scoring(tSrS, i, qo_idx, kv_idx);
      } else if constexpr (!CAUSAL) {  // Just masking based on col
        if (kv_idx >= kv_len) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      } else {
        if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
      if constexpr (LEFT_SLIDING_WINDOW) {
        if (kv_idx < col_limit_left(qo_idx)) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
    }
  }

  attention_updater.update</*init=*/true>(tSrS);
  Tensor tOrP = make_tensor(convert_type<DTypeKV>(tSrS).data(),
                            convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout()));

  constexpr int n_masking_steps = MULTIITEMSCORING ? (cute::ceil_div(CTA_Q, CTA_KV) + 1)
                                                   : (CAUSAL ? cute::ceil_div(CTA_Q, CTA_KV) : 0);
  // masking loops
  // ziangl@nvidia.com: for multi item scoring, we use this loop only to mask along the diagonal
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps && kv_tile_idx > swa_begin_kv_tile_idx;
       ++masking_step, kv_tile_idx = kv_tile_idx_decrement(kv_tile_idx)) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    WarpScheduler::barrier_sync();
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()),
                                        tSrS);
    if (masking_step > 0) {
      attention_updater.rescale_o(tOrO);
    }
    consumer_wait(pipeline_v, smem_pipe_read_v);
    gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP,
                                         tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
    WarpScheduler::barrier_arrive();
    warpgroup_wait<1>();
    pipeline_k.consumer_release(smem_pipe_read_k);  // release K
    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
    Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
      int kv_idx = get<1>(tScS(i)) + kv_tile_idx_decrement(kv_tile_idx) * CTA_KV;
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/0, qo_idx, kv_idx,
                                        qo_head_idx, kv_head_idx);
      if (MULTIITEMSCORING) {
        mask_multi_item_scoring(tSrS, i, qo_idx, kv_idx);
      } else {
        if (kv_idx >= col_limit_right(qo_idx)) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
      if constexpr (LEFT_SLIDING_WINDOW) {
        if (kv_idx < col_limit_left(qo_idx)) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
    }
    attention_updater.update</*init=*/false>(tSrS);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);  // release V
    ++smem_pipe_read_k;
    ++smem_pipe_read_v;
    cute::copy(make_tensor(convert_type<DTypeKV>(tSrS).data(),
                           convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout())),
               tOrP);
  }

#pragma unroll 1
  for (; kv_tile_idx > swa_end_kv_tile_idx + 1; kv_tile_idx = kv_tile_idx_decrement(kv_tile_idx)) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    WarpScheduler::barrier_sync();
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()),
                                        tSrS);
    attention_updater.rescale_o(tOrO);
    consumer_wait(pipeline_v, smem_pipe_read_v);
    gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP,
                                         tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
    WarpScheduler::barrier_arrive();
    warpgroup_wait<1>();
    pipeline_k.consumer_release(smem_pipe_read_k);  // release K
                                                    // #pragma unroll
    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
    Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
      int kv_idx = get<1>(tScS(i)) + kv_tile_idx_decrement(kv_tile_idx) * CTA_KV;
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/0, qo_idx, kv_idx,
                                        qo_head_idx, kv_head_idx);
    }
    if constexpr (MULTIITEMSCORING) {
      // auto nums_tiles_outside_causal_diagonal = kv_tile_idx_count - cute::ceil_div(CTA_Q,
      // CTA_KV);
      if (kv_tile_idx >= num_kv_tiles_prefix - 1) {
#pragma unroll
        for (int i = 0; i < size(tSrS); ++i) {
          int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
          int kv_idx = get<1>(tScS(i)) + kv_tile_idx_decrement(kv_tile_idx) * CTA_KV;
          mask_multi_item_scoring_assume_in_bound(tSrS, i, qo_idx, kv_idx);
        }
      }
    }
    attention_updater.update</*init=*/false>(tSrS);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);  // release V
    ++smem_pipe_read_k;
    ++smem_pipe_read_v;
    cute::copy(make_tensor(convert_type<DTypeKV>(tSrS).data(),
                           convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout())),
               tOrP);
  }

  if constexpr (LEFT_SLIDING_WINDOW) {
#pragma unroll 1
    for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
      Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
      consumer_wait(pipeline_k, smem_pipe_read_k);
      WarpScheduler::barrier_sync();
      gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ,
                                          tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      attention_updater.rescale_o(tOrO);
      consumer_wait(pipeline_v, smem_pipe_read_v);
      gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP,
                                           tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      WarpScheduler::barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read_k);  // release K
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
      Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
        int kv_idx = get<1>(tScS(i)) + (kv_tile_idx - 1) * CTA_KV;
        tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/0, qo_idx, kv_idx,
                                          qo_head_idx, kv_head_idx);
        if (kv_idx < col_limit_left(qo_idx)) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
      attention_updater.update</*init=*/false>(tSrS);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v);  // release V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
      cute::copy(make_tensor(convert_type<DTypeKV>(tSrS).data(),
                             convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout())),
                 tOrP);
    }
  }

  // Tell warp 0 that smem_q is ready
  cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kQueryEmpty));
  attention_updater.rescale_o(tOrO);
  consumer_wait(pipeline_v, smem_pipe_read_v);
  gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()),
                                       tOrO);
  attention_updater.finalize(tSrS, get_variant_scale_pv(variant));
  warpgroup_wait<0>();
  pipeline_v.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
  ++smem_pipe_read_v;

  attention_updater.rescale_o(tOrO);
  return;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
