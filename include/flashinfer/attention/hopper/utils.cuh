/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "../../utils.cuh"

#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()
#include <cute/tensor.hpp>


namespace flashinfer {

using namespace cute;

CUTLASS_HOST_DEVICE auto get_gmem_layout(int nnz, int num_heads, int head_dim, int64_t n_stride,
                                         int64_t h_stride) {
  return make_layout(make_shape(nnz, head_dim, num_heads),
                     make_stride(n_stride, cute::_1{}, h_stride));
}

CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(int nnz, int num_heads) {
  return make_layout(make_shape(num_heads, nnz), make_stride(int64_t(nnz), cute::_1()));
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                          int offset, int seq_len, int head_idx) {
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(1, get<1>(tile_shape)),
                             make_coord(offset, _0{}));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(seq_len, get<1>(tile_shape)), g_offset.stride()));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_lse_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                              int offset, int seq_len, int head_idx) {
  auto g_offset = local_tile(m_tensor(head_idx, _), cute::make_shape(_1{}), make_coord(offset));
  auto g_sequence =
      make_tensor(g_offset.data(), make_layout(cute::make_shape(seq_len), cute::make_shape(_1{})));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const& x, float const& y) { return max(x, y); }
};

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = acc_layout;
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
};

// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16,
// MMA_N))
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <bool zero_init = false, int wg_wait = 0, bool arrive = true, bool commit = true,
          typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB,
                                     Tensor2& tCrC) {
  constexpr bool Is_RS =
      !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (arrive) {
    warpgroup_arrive();
  }
  if constexpr (zero_init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false,
          bool Clear_OOB_K = true, typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3,
          typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                                     Tensor<Engine1, Layout1>& D,
                                     Tensor<Engine2, Layout2> const& identity_MN,
                                     Tensor<Engine3, Layout3> const& predicate_K,
                                     const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

template <int NUM_TMA_THREADS, typename ElemO, typename TiledCopyO, typename LayoutO,
          typename TileShapeO, typename SMemO>
__forceinline__ __device__ void write_tiled(ElemO* O, const TiledCopyO& tiled_copy_O,
                                            const LayoutO& layout_O, const TileShapeO& tile_shape_O,
                                            const SMemO& sO, int q_tile_idx, int head_idx,
                                            int64_t qo_len) {
  Tensor mO = make_tensor(make_gmem_ptr(O), layout_O);
  Tensor gO = get_local_tile_tensor(mO, tile_shape_O, head_idx, /*offset=*/0, qo_len)(
      _, _, q_tile_idx);  // (M, K)

  ThrCopy thr_copy_O = tiled_copy_O.get_slice(threadIdx.x - NUM_TMA_THREADS);
  Tensor tOgO = thr_copy_O.partition_D(gO);  // (CPY,CPY_M,CPY_K,k)
  Tensor tOsO = thr_copy_O.partition_S(sO);  // (CPY,CPY_M,CPY_K)

  // Prepare for TiledCopy.
  // Grouping is needed because cute::copy_if() does group_modes<1, R> for src and dst.
  // After grouping, the first dim is number of elements to read together.
  Tensor tOsOFlatten = cute::flatten(tOsO);
  Tensor tOsOGroup = cute::group_modes<1, rank(tOsOFlatten)>(tOsOFlatten);
  Tensor tOgOFlatten = cute::flatten(tOgO);
  Tensor tOgOGroup = cute::group_modes<1, rank(tOgOFlatten)>(tOgOFlatten);

  // Get thread coords to global index mapping.
  Tensor gOCounting = cute::make_identity_tensor(gO.shape());
  Tensor tSgOCounting = thr_copy_O.partition_D(gOCounting);
  Tensor tSgOCountingFlatten = cute::flatten(tSgOCounting);
  Tensor tSgOCountingGrouped = cute::group_modes<1, rank(tSgOCountingFlatten)>(tSgOCountingFlatten);

  // Write out to GMEM.
  const int kNumMsPerTile = get<0>(tile_shape_O);
  int cta_m = std::min<int>(qo_len - q_tile_idx * kNumMsPerTile, kNumMsPerTile);
  if (cta_m == kNumMsPerTile) {
    copy(tiled_copy_O, tOsOGroup, tOgOGroup);
  } else {
    auto predicate_fn = [&](auto coords) {
      auto s_coords = tSgOCountingGrouped(_0{}, coords);
      return elem_less(get<0>(s_coords), cta_m);
    };
    copy_if(tiled_copy_O, predicate_fn, tOsOGroup, tOgOGroup);
  }
}

template <int NUM_TMA_THREADS, typename ElemO, typename TMACopyO, typename TiledCopyO,
          typename LayoutO, typename TileShapeO, typename SMemO>
__forceinline__ __device__ void write_O(ElemO* O, const TMACopyO& tma_copy_O,
                                        const TiledCopyO& tiled_copy_O, const LayoutO& layout_O,
                                        const TileShapeO& tile_shape_O, const SMemO& sO,
                                        int q_tile_idx, int head_idx, int qo_len,
                                        int write_warp_idx) {
  write_tiled<NUM_TMA_THREADS>(O, tiled_copy_O, layout_O, tile_shape_O, sO, q_tile_idx, head_idx,
                              qo_len);
}

}  // namespace flashinfer
