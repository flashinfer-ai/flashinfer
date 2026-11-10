#pragma once

template <int LocalSlice, int GlobalSlice, typename T, typename TmaParamsT, typename TensorOutput,
          typename TensorSV, typename TensorSP, typename TensorSKV, typename Plan>
__forceinline__ __device__ void pv_owner_tma_slice_static(TensorOutput& output_fragment_all,
                                                          TensorSV const& sV, TensorSP const& sP,
                                                          TensorSKV& sKV, Plan& plan,
                                                          int idx_in_warpgroup, int lane,
                                                          int next_physical_page, bool has_next,
                                                          TmaParamsT const& tma_params) {
  static_assert(LocalSlice >= 0 && LocalSlice < 6);
  static_assert(GlobalSlice >= 0 && GlobalSlice < 16);
  Tensor output_fragment_flat = output_fragment_all(_, Int<LocalSlice>{}, _0{});
  Tensor output_fragment =
      make_tensor(output_fragment_flat.data(), typename T::OutputSliceFragmentLayout{});
  rescale_output_fragment<T>(output_fragment, plan, lane);
  Tensor v_tiles = flat_divide(sV, Shape<Int<T::TILE_K>, Int<T::BLOCK_N>>{});
  Tensor v_tile = v_tiles(_, _, Int<GlobalSlice>{}, _0{});
  issue_pv_gemm<T>(output_fragment, v_tile, sP, idx_in_warpgroup);
  if (has_next && idx_in_warpgroup == 0) {
    plan.kv_free_barriers[GlobalSlice].arrive();
    issue_kv_tma_async<GlobalSlice, false, T>(sKV, plan, next_physical_page, lane, false,
                                              tma_params);
  }
}

template <typename T, typename TmaParamsT, typename TensorOutput, typename TensorSV,
          typename TensorSP, typename TensorSKV, typename Plan>
__forceinline__ __device__ void pv_step_owner_tma_static(TensorOutput& output_fragment_all,
                                                         int slice_begin, TensorSV const& sV,
                                                         TensorSP const& sP, TensorSKV& sKV,
                                                         Plan& plan, int idx_in_warpgroup, int lane,
                                                         int next_physical_page, bool has_next,
                                                         TmaParamsT const& tma_params) {
  static_assert(T::BLOCK_M == 32);
  static_assert(T::NUM_V_TILES == 16);
  static_assert(T::NUM_PV_WARPGROUPS == 3);
  if (slice_begin == 0) {
    pv_owner_tma_slice_static<0, 0, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<1, 1, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<2, 2, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<3, 3, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<4, 4, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
  } else if (slice_begin == 5) {
    pv_owner_tma_slice_static<0, 5, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<1, 6, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<2, 7, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<3, 8, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<4, 9, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params);
  } else {
    pv_owner_tma_slice_static<0, 10, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<1, 11, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<2, 12, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<3, 13, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<4, 14, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
    pv_owner_tma_slice_static<5, 15, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params);
  }
}
