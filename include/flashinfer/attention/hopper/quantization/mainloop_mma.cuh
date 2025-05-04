/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_MAINLOOP_MMA_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_MAINLOOP_MMA_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flashinfer {

template <typename Ktraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename WarpScheduler,
          typename AttentionVariant, typename Params, typename MainloopPipeline,
          typename MainloopPipelineVt, typename PipelineState, typename SharedStorage,
          typename FrgTensorO, typename AttentionUpdater>
CUTLASS_DEVICE void mma_fp8(const Params& mainloop_params, AttentionVariant& variant,
                            MainloopPipeline pipeline_k, MainloopPipelineVt pipeline_vt,
                            PipelineState& smem_pipe_read_k, PipelineState& smem_pipe_read_v,
                            FrgTensorO& tOrO, AttentionUpdater& attention_updater,
                            int kv_tile_idx_count, int swa_begin_kv_tile_idx,
                            int swa_end_kv_tile_idx, int thread_idx, int work_idx, int q_tile_idx,
                            SharedStorage& shared_storage, const int32_t qo_len,
                            const int32_t kv_len, const int32_t qo_head_idx,
                            const int32_t kv_head_idx, const int32_t batch_idx) {
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
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVt{});

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
  {
    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
    Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      int qo_idx = get<0>(tScS(i)) + q_tile_idx * CTA_Q;
      int kv_idx = get<1>(tScS(i)) + kv_tile_idx * CTA_KV;
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/batch_idx, qo_idx,
                                        kv_idx, qo_head_idx, kv_head_idx);
      if constexpr (!CAUSAL) {  // Just masking based on col
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
  // Re-quantize P after softmax
  variant.PQuantize(tSrS);

  // Cast back to FP8
  Tensor tOrP =
      make_tensor(convert_type<DTypeKV>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
  permute_regs_A_to_C(tOrP);

  constexpr int n_masking_steps = CAUSAL ? cute::ceil_div(CTA_Q, CTA_KV) : 0;
  // masking loops
#pragma unroll
  for (int masking_step = 0; masking_step < n_masking_steps && kv_tile_idx > swa_begin_kv_tile_idx;
       ++masking_step, --kv_tile_idx) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    WarpScheduler::barrier_sync();
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()),
                                        tSrS);
    if (masking_step > 0) {
      attention_updater.rescale_o(tOrO);
    }
    consumer_wait(pipeline_vt, smem_pipe_read_v);
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
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/batch_idx, qo_idx,
                                        kv_idx, qo_head_idx, kv_head_idx);
      if (kv_idx >= col_limit_right(qo_idx)) {
        tSrS(i) = AttentionUpdater::fill_value;
      }
      if constexpr (LEFT_SLIDING_WINDOW) {
        if (kv_idx < col_limit_left(qo_idx)) {
          tSrS(i) = AttentionUpdater::fill_value;
        }
      }
    }
    attention_updater.update</*init=*/false>(tSrS);
    // Re-quantize P after softmax
    variant.PQuantize(tSrS);

    warpgroup_wait<0>();
    pipeline_vt.consumer_release(smem_pipe_read_v);  // release V
    ++smem_pipe_read_k;
    ++smem_pipe_read_v;
    cute::copy(make_tensor(convert_type<DTypeKV>(tSrS).data(),
                           convert_layout_acc_Aregs_fp8(tSrS.layout())),
               tOrP);
    permute_regs_A_to_C(tOrP);
  }

#pragma unroll 1
  for (; kv_tile_idx > swa_end_kv_tile_idx + 1; --kv_tile_idx) {
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    WarpScheduler::barrier_sync();
    gemm</*init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()),
                                        tSrS);
    attention_updater.rescale_o(tOrO);
    consumer_wait(pipeline_vt, smem_pipe_read_v);
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
      int kv_idx = get<1>(tScS(i)) + (kv_tile_idx - 1) * CTA_KV;
      tSrS(i) = variant.LogitsTransform(mainloop_params, tSrS(i), /*batch_idx=*/batch_idx, qo_idx,
                                        kv_idx, qo_head_idx, kv_head_idx);
    }

    attention_updater.update</*init=*/false>(tSrS);
    // Re-quantize P after softmax
    variant.PQuantize(tSrS);

    warpgroup_wait<0>();
    pipeline_vt.consumer_release(smem_pipe_read_v);  // release V
    ++smem_pipe_read_k;
    ++smem_pipe_read_v;
    cute::copy(make_tensor(convert_type<DTypeKV>(tSrS).data(),
                           convert_layout_acc_Aregs_fp8(tSrS.layout())),
               tOrP);
    permute_regs_A_to_C(tOrP);
  }

  // Tell warp 0 that smem_q is ready
  cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kQueryEmpty));
  attention_updater.rescale_o(tOrO);
  consumer_wait(pipeline_vt, smem_pipe_read_v);
  gemm</*init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()),
                                       tOrO);
  attention_updater.finalize(tSrS, variant.scale_pv);
  warpgroup_wait<0>();
  pipeline_vt.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
  ++smem_pipe_read_v;

  attention_updater.rescale_o(tOrO);
  // Dequantize output o with P/V scale
  variant.ODequantize(mainloop_params, tOrO, qo_head_idx, kv_head_idx);
  return;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_MAINLOOP_MMA_CUH_
