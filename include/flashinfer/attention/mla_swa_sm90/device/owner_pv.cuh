#pragma once

struct OwnerRowScaleCache {
  float values[8];
  __forceinline__ __device__ float operator[](int row) const {
    return values[(row / 8) * 2 + (row & 1)];
  }
};
struct OwnerScalePlan {
  OwnerRowScaleCache row_scale;
};

template <int Local, int Begin, int Count, class T, class Output, class V, class P, class KV,
          class Plan, class TMA>
__forceinline__ __device__ void m32_lookahead_owner_slices(Output& output, V const& sV, P const& sP,
                                                           KV& sKV, Plan& plan, int idx, int lane,
                                                           int local_page, int next_physical,
                                                           int page_plus2, bool has_next,
                                                           TMA const& tma,
                                                           OwnerScalePlan const& scales) {
  constexpr int Slice = Begin + Local;
  Tensor flat = output(_, Int<Local>{}, _0{});
  Tensor acc = make_tensor(flat.data(), typename T::OutputSliceFragmentLayout{});
  if constexpr (T::Config::S == 1 || T::Config::S == 2)
    rescale_output_fragment<T>(acc, scales, lane);
  else
    rescale_output_fragment<T>(acc, plan, lane);
  if constexpr (Slice >= 16 - T::LOOKAHEAD_K) {
    if ((local_page & 1) != 0) {
      issue_aux_pv_dynamic<T>(acc, sP, plan, Slice - (16 - T::LOOKAHEAD_K), idx);
    } else {
      Tensor tiles = flat_divide(sV, Shape<Int<64>, Int<64>>{});
      Tensor v = tiles(_, _, Int<Slice>{}, _0{});
      issue_pv_gemm<T>(acc, v, sP, idx);
    }
  } else {
    Tensor tiles = flat_divide(sV, Shape<Int<64>, Int<64>>{});
    Tensor v = tiles(_, _, Int<Slice>{}, _0{});
    issue_pv_gemm<T>(acc, v, sP, idx);
  }
  // This slice's wait0 has completed before its current bank is reused.
  if (has_next && idx == 0)
    issue_owner_tma_for_slice<T>(Slice, sKV, plan, next_physical, page_plus2, local_page, lane,
                                 tma);
  if constexpr (Local + 1 < Count)
    m32_lookahead_owner_slices<Local + 1, Begin, Count, T>(output, sV, sP, sKV, plan, idx, lane,
                                                           local_page, next_physical, page_plus2,
                                                           has_next, tma, scales);
}

template <class T, class Output, class V, class P, class KV, class Plan, class TMA>
__forceinline__ __device__ void pv_step_owner_lookahead_m32(Output& output, int begin, V const& sV,
                                                            P const& sP, KV& sKV, Plan& plan,
                                                            int idx, int lane, int local_page,
                                                            int next_physical, int page_plus2,
                                                            bool has_next, TMA const& tma) {
  static_assert(T::BLOCK_M == 32 && T::LOOKAHEAD_K == 2 && T::NUM_PV_WARPGROUPS == 3);
  OwnerScalePlan scales;
  if constexpr (T::Config::S == 1 || T::Config::S == 2) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int row = group * 8 + (lane % 4) * 2;
      scales.row_scale.values[group * 2] = plan.row_scale[row];
      scales.row_scale.values[group * 2 + 1] = plan.row_scale[row + 1];
    }
  }
  if (begin == 0)
    m32_lookahead_owner_slices<0, 0, 5, T>(output, sV, sP, sKV, plan, idx, lane, local_page,
                                           next_physical, page_plus2, has_next, tma, scales);
  else if (begin == 5)
    m32_lookahead_owner_slices<0, 5, 5, T>(output, sV, sP, sKV, plan, idx, lane, local_page,
                                           next_physical, page_plus2, has_next, tma, scales);
  else
    m32_lookahead_owner_slices<0, 10, 6, T>(output, sV, sP, sKV, plan, idx, lane, local_page,
                                            next_physical, page_plus2, has_next, tma, scales);
}

template <int LocalSlice, int GlobalSlice, typename T, typename TmaParamsT, typename TensorOutput,
          typename TensorSV, typename TensorSP, typename TensorSKV, typename Plan>
__forceinline__ __device__ void pv_owner_tma_slice_static(
    TensorOutput& output_fragment_all, TensorSV const& sV, TensorSP const& sP, TensorSKV& sKV,
    Plan& plan, int idx_in_warpgroup, int lane, int next_physical_page, bool has_next,
    TmaParamsT const& tma_params, OwnerScalePlan const& scale_plan) {
  static_assert(LocalSlice >= 0 && LocalSlice < 6);
  static_assert(GlobalSlice >= 0 && GlobalSlice < 16);
  Tensor output_fragment_flat = output_fragment_all(_, Int<LocalSlice>{}, _0{});
  Tensor output_fragment =
      make_tensor(output_fragment_flat.data(), typename T::OutputSliceFragmentLayout{});
  if constexpr (T::Config::S == 1 || T::Config::S == 2)
    rescale_output_fragment<T>(output_fragment, scale_plan, lane);
  else
    rescale_output_fragment<T>(output_fragment, plan, lane);
  Tensor v_tiles = flat_divide(sV, Shape<Int<T::TILE_K>, Int<T::BLOCK_N>>{});
  Tensor v_tile = v_tiles(_, _, Int<GlobalSlice>{}, _0{});
  issue_pv_gemm<T>(output_fragment, v_tile, sP, idx_in_warpgroup);
  if (has_next && idx_in_warpgroup == 0) {
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
  // Cache the page's row scales across this WG's PV slices.
  OwnerScalePlan scale_plan;
  if constexpr (T::Config::S == 1 || T::Config::S == 2) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int row = group * 8 + (lane % 4) * 2;
      scale_plan.row_scale.values[group * 2] = plan.row_scale[row];
      scale_plan.row_scale.values[group * 2 + 1] = plan.row_scale[row + 1];
    }
  }
  if (slice_begin == 0) {
    pv_owner_tma_slice_static<0, 0, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<1, 1, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<2, 2, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<3, 3, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<4, 4, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
  } else if (slice_begin == 5) {
    pv_owner_tma_slice_static<0, 5, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<1, 6, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<2, 7, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<3, 8, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<4, 9, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                       lane, next_physical_page, has_next, tma_params, scale_plan);
  } else {
    pv_owner_tma_slice_static<0, 10, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<1, 11, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<2, 12, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<3, 13, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<4, 14, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
    pv_owner_tma_slice_static<5, 15, T>(output_fragment_all, sV, sP, sKV, plan, idx_in_warpgroup,
                                        lane, next_physical_page, has_next, tma_params, scale_plan);
  }
}
