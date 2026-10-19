#pragma once

template <int Start, int End, typename TMAK, typename TensorG, typename TensorS>
__forceinline__ __device__ void launch_kv_tma(TensorG const& gkv, TensorS& skv, TMAK const& tma_k,
                                              TMABarrier* barriers, int thread_idx) {
  if (thread_idx == 0) {
    auto thr_tma = tma_k.get_slice(_0{});
    Tensor src = thr_tma.partition_S(gkv)(_, _0{}, Int<Start>{});
    Tensor dst = thr_tma.partition_D(skv)(_, _0{}, Int<Start>{});
    cute::copy(tma_k.with(reinterpret_cast<typename TMABarrier::ValueType&>(barriers[Start]), 0,
                          cute::TMA::CacheHintSm90::EVICT_FIRST),
               src, dst);
    if constexpr (Start + 1 < End) {
      launch_kv_tma<Start + 1, End>(gkv, skv, tma_k, barriers, thread_idx);
    }
  }
}

template <int Tile, bool WaitForFree, typename T, typename TmaParamsT, typename TensorSKV,
          typename Plan>
__forceinline__ __device__ void issue_kv_tma_async(TensorSKV& sKV, Plan& plan, int physical_page,
                                                   int lane, bool free_phase,
                                                   TmaParamsT const& tma_params) {
  static_assert(Tile < T::NUM_K_TILES);
  if (lane != 0) {
    return;
  }
  if constexpr (WaitForFree) {
    static_assert(Tile < T::NUM_V_TILES);
    plan.kv_free_barriers[Tile].wait(free_phase);
  }
  if (physical_page < 0) {
    return;
  }
  Tensor gkv = tma_params.tma_k.get_tma_tensor(tma_params.shape_k)(_, _, physical_page);
  launch_kv_tma<Tile, Tile + 1>(gkv, sKV, tma_params.tma_k, plan.kv_barriers, 0);
  plan.kv_barriers[Tile].arrive_and_expect_tx(T::BLOCK_N * T::TILE_K * sizeof(typename T::InputT));
}

template <int Tile, int End, typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_kv_tma_stream(TensorSKV& sKV, Plan& plan, int physical_page,
                                                    int lane, bool free_phase,
                                                    TmaParamsT const& tma_params) {
  static_assert(Tile < End && End <= T::NUM_V_TILES);
  issue_kv_tma_async<Tile, true, T>(sKV, plan, physical_page, lane, free_phase, tma_params);
  if constexpr (Tile + 1 < End) {
    issue_kv_tma_stream<Tile + 1, End, T>(sKV, plan, physical_page, lane, free_phase, tma_params);
  }
}

template <typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_base_kv_tma_dynamic(TensorSKV& sKV, Plan& plan,
                                                          int physical_page, int tile, int lane,
                                                          TmaParamsT const& tma_params) {
  if (lane != 0 || physical_page < 0) {
    return;
  }
  Tensor gkv = tma_params.tma_k.get_tma_tensor(tma_params.shape_k)(_, _, physical_page);
  auto thr_tma = tma_params.tma_k.get_slice(_0{});
  Tensor src = thr_tma.partition_S(gkv)(_, _0{}, tile);
  Tensor dst = thr_tma.partition_D(sKV)(_, _0{}, tile);
  cute::copy(tma_params.tma_k.with(
                 reinterpret_cast<typename TMABarrier::ValueType&>(plan.kv_barriers[tile]), 0,
                 cute::TMA::CacheHintSm90::EVICT_FIRST),
             src, dst);
  plan.kv_barriers[tile].arrive_and_expect_tx(T::BLOCK_N * T::TILE_K * sizeof(typename T::InputT));
}

template <typename T, typename TmaParamsT, typename Plan>
__forceinline__ __device__ void issue_aux_kv_tma_dynamic(Plan& plan, int physical_page, int tile,
                                                         int slot, int lane,
                                                         TmaParamsT const& tma_params) {
  if (lane != 0 || physical_page < 0) {
    return;
  }
  Tensor gkv = tma_params.tma_k.get_tma_tensor(tma_params.shape_k)(_, _, physical_page);
  Tensor sAux =
      make_tensor(make_smem_ptr(plan.aux_kv[slot].data()), typename T::SmemLayoutKVSlice{});
  auto thr_tma = tma_params.tma_k.get_slice(_0{});
  Tensor src = thr_tma.partition_S(gkv)(_, _0{}, tile);
  Tensor dst = thr_tma.partition_D(sAux)(_, _0{}, _0{});
  cute::copy(tma_params.tma_k.with(
                 reinterpret_cast<typename TMABarrier::ValueType&>(plan.aux_kv_barriers[slot]), 0,
                 cute::TMA::CacheHintSm90::EVICT_FIRST),
             src, dst);
  plan.aux_kv_barriers[slot].arrive_and_expect_tx(T::BLOCK_N * T::TILE_K *
                                                  sizeof(typename T::InputT));
}

template <typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_owner_tma_for_slice(int global_slice, TensorSKV& sKV,
                                                          Plan& plan, int next_physical_page,
                                                          int page_plus2_physical, int local_page,
                                                          int lane, TmaParamsT const& tma_params) {
  static_assert(T::NUM_V_TILES == 16);
  if (global_slice >= T::NUM_V_TILES - T::LOOKAHEAD_K) {
    const int slot = global_slice - (T::NUM_V_TILES - T::LOOKAHEAD_K);
    if ((local_page & 1) == 0) {
      issue_base_kv_tma_dynamic<T>(sKV, plan, page_plus2_physical, global_slice, lane, tma_params);
    } else {
      issue_aux_kv_tma_dynamic<T>(plan, page_plus2_physical, global_slice, slot, lane, tma_params);
    }
  } else {
    issue_base_kv_tma_dynamic<T>(sKV, plan, next_physical_page, global_slice, lane, tma_params);
  }
}

template <int Tile, int End, typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_base_kv_range_async(TensorSKV& sKV, Plan& plan,
                                                          int physical_page, int lane,
                                                          TmaParamsT const& tma_params) {
  static_assert(Tile < End && End <= T::NUM_K_TILES);
  issue_kv_tma_async<Tile, false, T>(sKV, plan, physical_page, lane, false, tma_params);
  if constexpr (Tile + 1 < End) {
    issue_base_kv_range_async<Tile + 1, End, T>(sKV, plan, physical_page, lane, tma_params);
  }
}

template <int Slot, typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_prologue_selected_base_descending(
    TensorSKV& sKV, Plan& plan, int physical_page, int lane, TmaParamsT const& tma_params) {
  constexpr int Tile = T::NUM_V_TILES - T::LOOKAHEAD_K + Slot;
  issue_kv_tma_async<Tile, false, T>(sKV, plan, physical_page, lane, false, tma_params);
  if constexpr (Slot > 0) {
    issue_prologue_selected_base_descending<Slot - 1, T>(sKV, plan, physical_page, lane,
                                                         tma_params);
  }
}

template <int Tile, int Slot, typename T, typename TmaParamsT, typename Plan>
__forceinline__ __device__ void issue_aux_kv_tma_async(Plan& plan, int physical_page, int lane,
                                                       TmaParamsT const& tma_params) {
  static_assert(T::LOOKAHEAD_K > 0);
  static_assert(Tile >= T::NUM_V_TILES - T::LOOKAHEAD_K);
  static_assert(Tile < T::NUM_V_TILES);
  static_assert(Slot == Tile - (T::NUM_V_TILES - T::LOOKAHEAD_K));
  if (lane != 0 || physical_page < 0) {
    return;
  }
  Tensor gkv = tma_params.tma_k.get_tma_tensor(tma_params.shape_k)(_, _, physical_page);
  Tensor sAux = make_tensor(make_smem_ptr(plan.aux_kv[Slot + T::AUX_GUARD_SLICES].data()),
                            typename T::SmemLayoutKVSlice{});
  auto thr_tma = tma_params.tma_k.get_slice(_0{});
  Tensor src = thr_tma.partition_S(gkv)(_, _0{}, Int<Tile>{});
  Tensor dst = thr_tma.partition_D(sAux)(_, _0{}, _0{});
  cute::copy(tma_params.tma_k.with(reinterpret_cast<typename TMABarrier::ValueType&>(
                                       plan.aux_kv_barriers[Slot + T::AUX_GUARD_SLICES]),
                                   0, cute::TMA::CacheHintSm90::EVICT_FIRST),
             src, dst);
  plan.aux_kv_barriers[Slot + T::AUX_GUARD_SLICES].arrive_and_expect_tx(T::BLOCK_N * T::TILE_K *
                                                                        sizeof(typename T::InputT));
}

template <int Slot, typename T, typename TmaParamsT, typename Plan>
__forceinline__ __device__ void issue_prologue_selected_aux_descending(
    Plan& plan, int physical_page, int lane, TmaParamsT const& tma_params) {
  constexpr int Tile = T::NUM_V_TILES - T::LOOKAHEAD_K + Slot;
  issue_aux_kv_tma_async<Tile, Slot, T>(plan, physical_page, lane, tma_params);
  if constexpr (Slot > 0) {
    issue_prologue_selected_aux_descending<Slot - 1, T>(plan, physical_page, lane, tma_params);
  }
}

template <typename T, typename TmaParamsT, typename TensorSKV, typename Plan>
__forceinline__ __device__ void issue_lookahead_prologue(TensorSKV& sKV, Plan& plan,
                                                         int first_physical_page,
                                                         int second_physical_page, int lane,
                                                         TmaParamsT const& tma_params) {
  static_assert(T::LOOKAHEAD_K > 0);
  issue_prologue_selected_base_descending<T::LOOKAHEAD_K - 1, T>(sKV, plan, first_physical_page,
                                                                 lane, tma_params);
  issue_kv_tma_async<T::NUM_V_TILES, false, T>(sKV, plan, first_physical_page, lane, false,
                                               tma_params);
  issue_base_kv_range_async<0, T::NUM_V_TILES - T::LOOKAHEAD_K, T>(sKV, plan, first_physical_page,
                                                                   lane, tma_params);
  issue_prologue_selected_aux_descending<T::LOOKAHEAD_K - 1, T>(plan, second_physical_page, lane,
                                                                tma_params);
}
