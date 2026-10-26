#pragma once

template <int Tile, typename T, typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorScore>
__forceinline__ __device__ void issue_qk_tile_ops(TiledMMA& qk_mma, TensorA const& qk_a,
                                                  TensorB const& qk_b, TensorScore& score) {
  static_assert(Tile < T::NUM_K_TILES);
  static_assert(T::TILE_K == 64);
  CUTLASS_PRAGMA_UNROLL
  for (int k16 = 0; k16 < T::TILE_K / 16; ++k16) {
    cute::gemm(qk_mma, qk_a(_, _, Tile * (T::TILE_K / 16) + k16),
               qk_b(_, _, Tile * (T::TILE_K / 16) + k16), score);
    qk_mma.accumulate_ = GMMA::ScaleOut::One;
  }
}

template <int Tile0, int Tile1, bool ZeroInit, typename T, typename TiledMMA, typename TensorA,
          typename TensorB, typename TensorScore, typename Plan>
__forceinline__ __device__ void issue_qk_ready_pair(TiledMMA& qk_mma, TensorA const& qk_a,
                                                    TensorB const& qk_b, TensorScore& score,
                                                    Plan& plan, bool kv_phase) {
  plan.kv_barriers[Tile0].wait(kv_phase);
  plan.kv_barriers[Tile1].wait(kv_phase);
  warpgroup_arrive();
  qk_mma.accumulate_ = ZeroInit ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  issue_qk_tile_ops<Tile0, T>(qk_mma, qk_a, qk_b, score);
  issue_qk_tile_ops<Tile1, T>(qk_mma, qk_a, qk_b, score);
  warpgroup_commit_batch();
  warpgroup_wait<1>();
}

template <int Tile, bool ZeroInit, typename T, typename TiledMMA, typename TensorA,
          typename TensorB, typename TensorScore, typename Plan>
__forceinline__ __device__ void issue_qk_ready_single(TiledMMA& qk_mma, TensorA const& qk_a,
                                                      TensorB const& qk_b, TensorScore& score,
                                                      Plan& plan, bool kv_phase) {
  plan.kv_barriers[Tile].wait(kv_phase);
  warpgroup_arrive();
  qk_mma.accumulate_ = ZeroInit ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  issue_qk_tile_ops<Tile, T>(qk_mma, qk_a, qk_b, score);
  warpgroup_commit_batch();
  warpgroup_wait<1>();
}

template <typename T, bool WaitForCompletion = true, typename TensorOutput, typename TensorSP,
          typename Plan>
__forceinline__ __device__ void issue_aux_pv_dynamic(TensorOutput& output_fragment,
                                                     TensorSP const& sP, Plan& plan, int slot,
                                                     int idx_in_warpgroup) {
  Tensor sAuxV =
      make_tensor(make_smem_ptr(plan.aux_kv[slot].data()), typename T::SmemLayoutVSlice{});
  typename T::TiledMMAPV mma;
  auto thread_mma = mma.get_slice(idx_in_warpgroup);
  Tensor a = thread_mma.partition_fragment_A(sAuxV);
  Tensor b = thread_mma.partition_fragment_B(sP);
  issue_wgmma<false, WaitForCompletion>(mma, a, b, output_fragment);
}

template <int Tile, int Slot, bool ZeroInit, typename T, typename TiledMMA, typename TensorA,
          typename TensorB, typename TensorScore, typename Plan>
__forceinline__ __device__ void issue_qk_ready_lookahead_single(
    TiledMMA& qk_mma, TensorA const& qk_a, TensorB const& qk_b, TensorScore& score, Plan& plan,
    int idx_in_warpgroup, bool use_aux, bool stage_phase) {
  static_assert(T::LOOKAHEAD_K > 0);
  static_assert(Tile == T::NUM_V_TILES - T::LOOKAHEAD_K + Slot);
  if (!use_aux) {
    issue_qk_ready_single<Tile, ZeroInit, T>(qk_mma, qk_a, qk_b, score, plan, stage_phase);
    return;
  }
  plan.aux_kv_barriers[Slot + T::AUX_GUARD_SLICES].wait(stage_phase);
  Tensor sAux = make_tensor(make_smem_ptr(plan.aux_kv[Slot + T::AUX_GUARD_SLICES].data()),
                            typename T::SmemLayoutKVSlice{});
  auto qk_thread = qk_mma.get_slice(idx_in_warpgroup);
  Tensor aux_a = qk_thread.partition_fragment_A(sAux);
  warpgroup_arrive();
  qk_mma.accumulate_ = ZeroInit ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k16 = 0; k16 < T::TILE_K / 16; ++k16) {
    cute::gemm(qk_mma, aux_a(_, _, k16), qk_b(_, _, Tile * (T::TILE_K / 16) + k16), score);
    qk_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  warpgroup_wait<1>();
}

template <int Slot, typename T, bool ZeroInitFirst = true, typename TiledMMA, typename TensorA,
          typename TensorB, typename TensorScore, typename Plan>
__forceinline__ __device__ void issue_qk_lookahead_descending(TiledMMA& qk_mma, TensorA const& qk_a,
                                                              TensorB const& qk_b,
                                                              TensorScore& score, Plan& plan,
                                                              int idx_in_warpgroup, bool use_aux,
                                                              bool stage_phase) {
  static_assert(Slot >= 0 && Slot < T::LOOKAHEAD_K);
  constexpr int Tile = T::NUM_V_TILES - T::LOOKAHEAD_K + Slot;
  issue_qk_ready_lookahead_single<Tile, Slot, ZeroInitFirst && Slot == T::LOOKAHEAD_K - 1, T>(
      qk_mma, qk_a, qk_b, score, plan, idx_in_warpgroup, use_aux, stage_phase);
  if constexpr (Slot > 0) {
    issue_qk_lookahead_descending<Slot - 1, T, ZeroInitFirst>(
        qk_mma, qk_a, qk_b, score, plan, idx_in_warpgroup, use_aux, stage_phase);
  }
}

template <int Tile, typename T, typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorScore, typename Plan>
__forceinline__ __device__ void issue_qk_nonlookahead_ascending(TiledMMA& qk_mma,
                                                                TensorA const& qk_a,
                                                                TensorB const& qk_b,
                                                                TensorScore& score, Plan& plan,
                                                                bool kv_phase) {
  static_assert(Tile < T::NUM_V_TILES - T::LOOKAHEAD_K);
  issue_qk_ready_single<Tile, false, T>(qk_mma, qk_a, qk_b, score, plan, kv_phase);
  if constexpr (Tile + 1 < T::NUM_V_TILES - T::LOOKAHEAD_K) {
    issue_qk_nonlookahead_ascending<Tile + 1, T>(qk_mma, qk_a, qk_b, score, plan, kv_phase);
  }
}

template <typename T, typename TiledMMA, typename TensorA, typename TensorB, typename TensorScore,
          typename Plan>
__forceinline__ __device__ void issue_qk_ready_order_lookahead(
    TiledMMA& qk_mma, TensorA const& qk_a, TensorB const& qk_b, TensorScore& score, Plan& plan,
    int idx_in_warpgroup, bool use_aux, bool selected_stage_phase, bool kv_phase) {
  static_assert(T::LOOKAHEAD_K > 0);
  Tensor aux_score =
      partition_fragment_C(typename T::TiledMMAQK{}, Shape<Int<T::BLOCK_N>, Int<T::BLOCK_M>>{});
  warpgroup_fence_operand(aux_score);
  issue_qk_lookahead_descending<T::LOOKAHEAD_K - 1, T>(
      qk_mma, qk_a, qk_b, aux_score, plan, idx_in_warpgroup, use_aux, selected_stage_phase);
  warpgroup_wait<0>();
  warpgroup_fence_operand(aux_score);
  warpgroup_fence_operand(score);
  issue_qk_ready_single<T::NUM_V_TILES, true, T>(qk_mma, qk_a, qk_b, score, plan, kv_phase);
  issue_qk_nonlookahead_ascending<0, T>(qk_mma, qk_a, qk_b, score, plan, kv_phase);
  warpgroup_wait<0>();
  warpgroup_fence_operand(score);
#pragma unroll
  for (int value = 0; value < size(aux_score); ++value) {
    score(value) += aux_score(value);
  }
}
