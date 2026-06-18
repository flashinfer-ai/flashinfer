/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "utils.cuh"

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled {

using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

namespace utils {

template <int kPerWarpN, typename Element>
CUTE_HOST_DEVICE constexpr auto SelectSmemCopyAtomB() {
  if constexpr (kPerWarpN >= 16)
    return Copy_Atom<SM75_U32x4_LDSM_N, Element>{};
  else if constexpr (kPerWarpN >= 8)
    return Copy_Atom<SM75_U32x2_LDSM_N, Element>{};
  else
    return Copy_Atom<SM75_U32x1_LDSM_N, Element>{};
}

template <int kTileN>
CUTE_HOST_DEVICE constexpr auto SelectSmemCopyAtomR2S() {
  if constexpr (kTileN >= 32)
    return SM90_U16x8_STSM_T{};
  else if constexpr (kTileN >= 16)
    return SM90_U16x4_STSM_T{};
  else
    return SM90_U16x2_STSM_T{};
}

template <bool kFlat, bool kSwapAB, int kTileN, bool kPerBatchAB>
CUTE_HOST_DEVICE constexpr bool EnableTmaStore() {
  if constexpr (!kFlat) {
    return false;
  } else {
    if constexpr (kSwapAB && !kPerBatchAB && kTileN > 32) {
      return false;
    } else {
      return true;
    }
  }
}

template <int kTileK, typename Element>
CUTE_HOST_DEVICE constexpr auto SelectSmemLayoutAtomK() {
  if constexpr (kTileK == 128)
    return GMMA::Layout_K_SW128_Atom<Element>{};
  else if constexpr (kTileK == 64)
    return GMMA::Layout_K_SW64_Atom<Element>{};
  else
    return GMMA::Layout_K_SW32_Atom<Element>{};
}

}  // namespace utils

template <int TileM_, int TileN_, int TileK_, int Stages_, int GranK_, typename ElementSFLoad_,
          typename ElementSFCompute_, GemmType kGemmType_ = GemmType::Normal, bool kSwapAB_ = false>
struct Sm120BlockScaledSFConfig {
  using ElementSFLoad = ElementSFLoad_;
  using ElementSFCompute = ElementSFCompute_;

  static constexpr GemmType kGemmType = kGemmType_;
  static constexpr bool kSwapAB = kSwapAB_;
  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = TileK_;
  static constexpr int AB_Stages = Stages_;
  static constexpr int kGranK = GranK_;

  static constexpr int SFVecSize = 32;
  static constexpr int MMA_K = SFVecSize;
  static constexpr int MMA_NSF = 1;
  static constexpr int kTileSF = 1;
  static constexpr int PACK_NSF = sizeof(ElementSFLoad) / sizeof(ElementSFCompute);
  static constexpr int PACK_NK = kGranK * PACK_NSF;
  static constexpr int kNumMMAPerSF = kGranK / MMA_K;
  static constexpr int kAtomsPerSfPerTileK = (kGranK < kTileK ? kGranK : kTileK) / MMA_K;
  static constexpr int kNumSFPerTileK = (kTileK >= kGranK) ? (kTileK / kGranK) : 1;
  static constexpr int kNumTileKPerSF = (kGranK > kTileK) ? (kGranK / kTileK) : 1;
  static constexpr int kNumTileKPerPackSF = PACK_NK / kTileK;
  static constexpr int kNumTileKGroupsPerPackSF = kNumTileKPerPackSF / kNumTileKPerSF;
  static constexpr int kNumStagePerSF = kNumTileKPerPackSF / AB_Stages;
  static constexpr int SF_Stages =
      AB_Stages > kNumTileKPerPackSF ? AB_Stages / kNumTileKPerPackSF : 1;

  static_assert(kTileK % MMA_K == 0, "kTileK must be a multiple of MMA_K");
  static_assert(kGranK % MMA_K == 0, "kGranK must be a multiple of MMA_K");
  static_assert(PACK_NK % kTileK == 0, "PACK_NK must be a multiple of kTileK");
  static_assert((kTileK % kGranK == 0) || (kGranK % kTileK == 0),
                "kTileK and kGranK must be related by integer multiple");
  static_assert((kNumTileKPerPackSF >= AB_Stages && kNumTileKPerPackSF % AB_Stages == 0) ||
                    (AB_Stages > kNumTileKPerPackSF && AB_Stages % kNumTileKPerPackSF == 0),
                "AB_Stages and kNumTileKPerPackSF must be divisible (in either direction)");
  static_assert((AB_Stages & (AB_Stages - 1)) == 0,
                "AB_Stages must be power of 2 (kernel uses bit-mask `& (AB_Stages-1)` as modulo)");

  using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
  using ProblemShape = Shape<int, int, int, int>;

  CUTE_HOST_DEVICE
  static auto get_tma_aligned_size(const int& x) {
    constexpr int kNumTMAAlignmentBytes = 16;
    CUTE_STATIC_ASSERT(kNumTMAAlignmentBytes % sizeof(ElementSFLoad) == 0,
                       "element_size must be a multiple of 16");
    auto alignment = kNumTMAAlignmentBytes / sizeof(ElementSFLoad);
    return math::align(x, alignment);
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfa_layout(ProblemShape const& problem_shape, int num_experts_for_sf = 0) {
    auto [M, N, K, L] = problem_shape;
    constexpr int pack_nk = PACK_NK;
    int64_t scale_m;
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithZeroPadding && !kSwapAB) {
      scale_m = static_cast<int64_t>(get_tma_aligned_size(
          math::compute_padded_offset(int64_t(M), int64_t(num_experts_for_sf))));
    } else {
      scale_m = static_cast<int64_t>(get_tma_aligned_size(M));
    }
    int64_t scale_k = static_cast<int64_t>(math::ceil_div(K, pack_nk));
    return make_layout(make_shape(scale_m, scale_k, L),
                       make_stride(Int<1>{}, scale_m, scale_m * scale_k));
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfb_layout(ProblemShape const& problem_shape, int num_experts_for_sf = 0) {
    auto [M, N, K, L] = problem_shape;
    constexpr int pack_nk = PACK_NK;
    int64_t scale_n;
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithZeroPadding && kSwapAB) {
      scale_n = static_cast<int64_t>(get_tma_aligned_size(
          math::compute_padded_offset(int64_t(N), int64_t(num_experts_for_sf))));
    } else {
      scale_n = static_cast<int64_t>(get_tma_aligned_size(N));
    }
    int64_t scale_k = static_cast<int64_t>(math::ceil_div(K, pack_nk));
    return make_layout(make_shape(scale_n, scale_k, L),
                       make_stride(Int<1>{}, scale_n, scale_n * scale_k));
  }

  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                                    TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
    auto tiled_sfa = logical_divide(sfatensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
    auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile);
    using AtomLayoutSFA_TV = Layout<Shape<Shape<_2, _2, _8>, _1>, Stride<Stride<_8, _0, _1>, _16>>;
    auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfa, thr_tile);
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(SFATensor&& sfatensor,
                                                                ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                  thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    auto frg_SFA = make_fragment_like<ElementSFLoad>(partition_SFA);
    return frg_SFA;
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE static constexpr auto get_layoutSFA_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFA(ref_A, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto atile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));
    auto tv_sfa = thr_tensor.compose(atile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfa.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                                    TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
    auto tiled_sfb = logical_divide(sfbtensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
    auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile);
    using AtomLayoutSFB_TV = Layout<Shape<Shape<_4, _8>, _1>, Stride<Stride<_0, _1>, _8>>;
    auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfb, thr_tile);
    return thr_tensor;
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor,
                                                                ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                  thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    auto frg_SFB = make_fragment_like<ElementSFLoad>(partition_SFB);
    return frg_SFB;
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE static constexpr auto get_layoutSFB_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFB(ref_B, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto btile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<0>{}, Int<1>{})),
                     _));
    auto tv_sfb = thr_tensor.compose(btile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfb.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  template <class Tensor>
  CUTE_HOST_DEVICE static constexpr auto transform_fragment_for_qmma(Tensor&& tensor) {
    CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<2>(shape(tensor)) == Int<1>{});
    auto old_ptr = tensor.data();
    auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
    auto old_layout = tensor.layout();
    auto num_mn = size<1>(shape(old_layout));
    auto atoms_per_sf_per_tilek = Int<kAtomsPerSfPerTileK>{};
    auto num_sf_per_tilek = Int<kNumSFPerTileK>{};
    auto num_tilek_per_sf = Int<kNumTileKPerSF>{};
    auto num_tilek_groups = Int<kNumTileKGroupsPerPackSF>{};
    auto new_layout = make_layout(
        make_shape(Int<SFVecSize>{}, num_mn, make_shape(atoms_per_sf_per_tilek, num_sf_per_tilek),
                   make_shape(num_tilek_per_sf, num_tilek_groups)),
        make_stride(_0{}, Int<PACK_NSF>{}, make_stride(_0{}, _1{}),
                    make_stride(_0{}, num_sf_per_tilek)));
    auto new_tensor = make_tensor(new_ptr, new_layout);
    return new_tensor;
  }

  using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;

  using SmemLayoutAtomSFA =
      decltype(make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  using SmemLayoutAtomSFB =
      decltype(make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  static constexpr int kSfStageStrideElems = 128 / int(sizeof(ElementSFLoad));
  static constexpr int kSfaStageElems =
      (int(cute::cosize_v<SmemLayoutAtomSFA>) < kSfStageStrideElems)
          ? kSfStageStrideElems
          : int(cute::cosize_v<SmemLayoutAtomSFA>);
  static constexpr int kSfbStageElems =
      (int(cute::cosize_v<SmemLayoutAtomSFB>) < kSfStageStrideElems)
          ? kSfStageStrideElems
          : int(cute::cosize_v<SmemLayoutAtomSFB>);

  using SmemLayoutSFA = Layout<Shape<decltype(shape<0>(ScaleTileShape{})),
                                     decltype(shape<2>(ScaleTileShape{})), Int<SF_Stages>>,
                               Stride<_1, _0, Int<kSfaStageElems>>>;

  using SmemLayoutSFB = Layout<Shape<decltype(shape<1>(ScaleTileShape{})),
                                     decltype(shape<2>(ScaleTileShape{})), Int<SF_Stages>>,
                               Stride<_1, _0, Int<kSfbStageElems>>>;

  using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
  using StrideSFB = Stride<Int<1>, int64_t, int64_t>;

  using TMA_SFA = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFA{}, int64_t(0)),
                  StrideSFA{}),
      SmemLayoutSFA{}(_, _, cute::Int<0>{}),
      make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));

  using TMA_SFB = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFB{}, int64_t(0)),
                  StrideSFB{}),
      SmemLayoutSFB{}(_, _, cute::Int<0>{}),
      make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));

  static constexpr uint32_t TmaTransactionBytesSFA = static_cast<uint32_t>(cutlass::bits_to_bytes(
      cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSFLoad>));
  static constexpr uint32_t TmaTransactionBytesSFB = static_cast<uint32_t>(cutlass::bits_to_bytes(
      cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSFLoad>));
  static constexpr uint32_t TmaSFTransactionBytes = TmaTransactionBytesSFA + TmaTransactionBytesSFB;
};

template <int kTileM, int kTileN, bool kUseTmaStore, bool kSwapAB = false>
struct Sm120BlockScaledMMAConfig {
  using MMA_Atom =
      cute::MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<cute::float_e4m3_t, cute::float_e4m3_t,
                                                             float, cute::float_ue8m0_t, 32>>;

  static_assert(kTileN >= 8 && (kTileN % 8) == 0,
                "kTileN must be >= 8 and multiple of MMA atom_N=8");
  static constexpr int kNumMathWarpN = (kTileN >= 32) ? 4 : (kTileN / 8);
  static constexpr int kNumMathWarpM = 8 / kNumMathWarpN;
  static constexpr int kNumMathWarps = kNumMathWarpM * kNumMathWarpN;
  static constexpr int kNumMathThreads = kNumMathWarps * 32;
  static constexpr int kNumMathWG = kNumMathThreads / 128;
  static_assert(kNumMathWarps == 8, "Total math warps must be 8 (256 threads = 2 WG)");
  static_assert(kTileM >= kNumMathWarpM * 16,
                "kTileM must be >= kNumMathWarpM * 16 (ThrLayout M-direction lower bound).");

  using PermMmaTileM = Int<(kUseTmaStore && !kSwapAB) ? ((kTileM < 32) ? kTileM : 32) : kTileM>;
  using PermMmaTileN = Int<kTileN>;

  using TiledMma = TiledMMA<
      MMA_Atom,
      Layout<Shape<Int<kNumMathWarpM>, Int<kNumMathWarpN>, _1>, Stride<_1, Int<kNumMathWarpM>, _0>>,
      Tile<PermMmaTileM, PermMmaTileN, Underscore>>;

  static_assert(size<2>(typename MMA_Atom::Shape_MNK{}) == 32,
                "MMA atom K-dim must be 32 to match SFVecSize");
};

template <int kTileM, int kTileN, int kTileK, int kStages, typename ElementA, typename ElementB>
struct Sm120BlockScaledABLoadConfig {
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;

  static constexpr int kNumMathWarpN = (kTileN >= 32) ? 4 : (kTileN / 8);
  static constexpr int kPerWarpN = kTileN / kNumMathWarpN;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB = decltype(utils::SelectSmemCopyAtomB<kPerWarpN, ElementB>());

  using SmemLayoutAtomA = decltype(utils::SelectSmemLayoutAtomK<kTileK, ElementA>());
  using SmemLayoutAtomB = decltype(utils::SelectSmemLayoutAtomK<kTileK, ElementB>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;

  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementA>(nullptr), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      SmemLayoutA{}(_, _, Int<0>{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
      _1{}));

  using TMA_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      SmemLayoutB{}(_, _, Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
      _1{}));

  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>));
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
  static constexpr uint32_t TmaABTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
};

template <int kTileM, int kTileN, typename ElementD, bool kEnabled = true>
struct Sm120BlockScaledTmaStoreConfig {
  using StrideD = Stride<int64_t, Int<1>, int64_t>;

  static constexpr int kEpiTileM = 32;
  static constexpr int kEpiTileN = kTileN;
  static constexpr int StagesD = kTileM / kEpiTileM;
  using EpilogueTile_MN = Shape<Int<kEpiTileM>, Int<kEpiTileN>>;
  static constexpr int kNumEpiM = kTileM / kEpiTileM;
  static constexpr int kNumEpiN = kTileN / kEpiTileN;

  using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using SmemLayoutAtomD = GMMA::Layout_K_SW128_Atom<ElementD>;
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{}, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}, Int<StagesD>{}),
      Step<_1, _2, _3>{}));
  using CopyOpR2S = SM90_U32x4_STSM_N;
  using CopyOpS2G = SM90_TMA_STORE;
  using TMA_D =
      decltype(make_tma_copy_C_sm90(CopyOpS2G{},
                                    make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)),
                                                repeat_like(StrideD{}, int64_t(0)), StrideD{}),
                                    take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

  struct SharedStorageTmaStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledTmaStoreConfig<kTileM, kTileN, ElementD, /*kEnabled=*/false> {
  struct TMA_D {};
  using StrideD = Stride<int64_t, Int<1>, int64_t>;
  static constexpr int StagesD = 1;
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledR2GStoreConfig {
  static constexpr int kSmemON = kTileN;

  using SmemAtomLayoutO = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kSmemON>>{}));
  using SmemCopyAtomR2S = Copy_Atom<AutoVectorizingCopy, ElementD>;
  using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;
  using TiledCopyS2R = decltype(make_tiled_copy(
      SmemCopyAtomS2R{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));

  struct SharedStorageR2G : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
  };
};

template <int kTileM, int kTileN, typename ElementD, bool kEnabled = true>
struct Sm120BlockScaledSwapABTmaStoreConfig {
  static_assert(kTileM == 128, "SwapAB TmaStoreConfig hardcoded for kTileM=128");

  using StrideD = Stride<Int<1>, int64_t, int64_t>;

  static constexpr int kEpiTileM = kTileM;
  static constexpr int kEpiTileN = kTileN;
  static constexpr int StagesD = 1;
  using EpilogueTile_MN = Shape<Int<kEpiTileM>, Int<kEpiTileN>>;
  static constexpr int kNumEpiM = kTileM / kEpiTileM;
  static constexpr int kNumEpiN = kTileN / kEpiTileN;

  using CopyOpR2S = decltype(utils::SelectSmemCopyAtomR2S<kTileN>());
  using CopyAtomC = Copy_Atom<CopyOpR2S, ElementD>;

  using SmemLayoutAtomD = GMMA::Layout_MN_SW128_Atom<ElementD>;
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{}, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}, Int<StagesD>{}),
      Step<_1, _2, _3>{}));

  using CopyOpS2G = SM90_TMA_STORE;
  using TMA_D =
      decltype(make_tma_copy_C_sm90(CopyOpS2G{},
                                    make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)),
                                                repeat_like(StrideD{}, int64_t(0)), StrideD{}),
                                    take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

  struct SharedStorageTmaStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledSwapABTmaStoreConfig<kTileM, kTileN, ElementD, /*kEnabled=*/false> {
  struct TMA_D {};
  using StrideD = Stride<Int<1>, int64_t, int64_t>;
  static constexpr int StagesD = 1;
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledSwapABR2GStoreConfig {
  static_assert(kTileM == 128, "SwapAB R2GStoreConfig hardcoded for kTileM=128");

  using StrideD = Stride<Int<1>, int64_t, int64_t>;

  using SmemAtomLayoutO = GMMA::Layout_MN_SW128_Atom<ElementD>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kTileN>>{}));

  using SmemCopyAtomR2S = Copy_Atom<decltype(utils::SelectSmemCopyAtomR2S<kTileN>()), ElementD>;
  using SmemCopyAtomS2R = Copy_Atom<AutoVectorizingCopy, ElementD>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;

  using TiledCopyS2R =
      decltype(make_tiled_copy(SmemCopyAtomS2R{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{},
                               Layout<Shape<_4, Int<kTileN / 8>>>{}));

  struct SharedStorageR2G : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
  };
};

template <int TileM_ = 32, int TileN_ = 128, int TileK_ = 128, int Stages_ = 4, int GranK_ = 128,
          GemmType GemmType_ = GemmType::MGroupedContiguousWithPsumLayout, bool SwapAB_ = false>
struct SM120BlockScaledBuilder {
  using ElementA = cute::float_e4m3_t;
  using ElementB = cute::float_e4m3_t;
  using ElementAccum = float;
  using ElementD = cute::bfloat16_t;

  static constexpr GemmType kGemmType = GemmType_;
  static constexpr bool kFlat = is_flat_gemm(GemmType_);
  static constexpr bool kSwapAB = SwapAB_;
  static constexpr bool kPerBatchAB =
      (GemmType_ == GemmType::Batched || GemmType_ == GemmType::MGroupedMasked);
  static constexpr bool kUseTmaStore = utils::EnableTmaStore<kFlat, kSwapAB, TileN_, kPerBatchAB>();
  static constexpr bool kUnionSmem = !kUseTmaStore;
  static constexpr int AB_Stages = Stages_;

  static constexpr int kGranK = GranK_;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = TileK_;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using ProblemShape = Shape<int, int, int, int>;

  using SFConfig = Sm120BlockScaledSFConfig<kTileM, kTileN, TileK_, Stages_, GranK_, int32_t,
                                            cute::float_ue8m0_t, kGemmType, kSwapAB>;
  using MMAConfig = Sm120BlockScaledMMAConfig<kTileM, kTileN, kUseTmaStore, kSwapAB>;
  using ABLoadConfig =
      Sm120BlockScaledABLoadConfig<kTileM, kTileN, kTileK, AB_Stages, ElementA, ElementB>;
  using TmaStoreConfig = std::conditional_t<
      kSwapAB, Sm120BlockScaledSwapABTmaStoreConfig<kTileM, kTileN, ElementD, kUseTmaStore>,
      Sm120BlockScaledTmaStoreConfig<kTileM, kTileN, ElementD, kUseTmaStore>>;
  using R2GStoreConfig =
      std::conditional_t<kSwapAB, Sm120BlockScaledSwapABR2GStoreConfig<kTileM, kTileN, ElementD>,
                         Sm120BlockScaledR2GStoreConfig<kTileM, kTileN, ElementD>>;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024)
        cute::ArrayEngine<ElementB, cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB;
  };

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;

  static constexpr int kNumStoreMbar = kUseTmaStore ? TmaStoreConfig::StagesD : 1;
  struct BarrierStorage {
    FullBarrier ab_full_mbar[AB_Stages];
    EmptyBarrier ab_empty_mbar[AB_Stages];
    FullBarrier sf_full_mbar[SFConfig::SF_Stages];
    EmptyBarrier sf_empty_mbar[SFConfig::SF_Stages];
    EmptyBarrier store_full_mbar[kNumStoreMbar];
    EmptyBarrier store_empty_mbar[kNumStoreMbar];
  };

  struct TensorStorageSplit {
    SharedStorageLoad load;
    typename TmaStoreConfig::SharedStorageTmaStore store;
  };

  union TensorStorageUnion {
    SharedStorageLoad load;
    typename R2GStoreConfig::SharedStorageR2G store;
  };
};

}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled
