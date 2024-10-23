/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"
#include "../../math.cuh"

namespace flashinfer {

using namespace cute;

// template <int HEAD_DIM_, int CTA_Q_, int CTA_KV_, int kNWarps_, typename Element_>
template <typename Ktraits>
struct CollectiveEpilogue {
  using Element = typename Ktraits::OutputType;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;
  static constexpr int HEAD_DIM = Ktraits::HEAD_DIM;
  using TileShape_MNK = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM>>;

  static constexpr int kNWarps = Ktraits::kNWarps;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr bool Is_WS = kNWarps >= 12;

  static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumMmaThreads = kNThreads - NumCopyThreads;

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>>;

  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<int64_t, _1>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using TMA_O = decltype(make_tma_copy(
      GmemTiledCopyOTMA{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeT{}, StrideT{}),
      SmemLayoutO{}, select<0, 2>(TileShape_MNK{}), _1{}));  // no mcast for O

  // These are for storing the output tensor without TMA (e.g., for setting output to zero and
  // var-seq-len)
  static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<Element>);
  static_assert(HEAD_DIM % kNumVecElem == 0);
  static constexpr int kNumThreadsPerRow = HEAD_DIM / kNumVecElem;
  static_assert(NumMmaThreads % kNumThreadsPerRow == 0);
  static constexpr int kNumRows = NumMmaThreads / kNumThreadsPerRow;
  using TiledCopyOAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, Element>;
  using TiledCopyOThrLayout = decltype(cute::make_layout(
      cute::make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}), LayoutRight{}));
  using TiledCopyOValLayout =
      decltype(cute::make_layout(cute::make_shape(_1{}, Int<kNumVecElem>{}), LayoutRight{}));
  using TiledCopyO =
      decltype(make_tiled_copy(TiledCopyOAtom{}, TiledCopyOThrLayout{},  // Thr layout
                               TiledCopyOValLayout{}                     // Val layout
                               ));

  // used for rmem -> smem O copy in fp8 kernel to undo column permutation
  using ThreadLayoutrO = Layout<Shape<_8, Int<CTA_Q / 16>, _4, _1>, Stride<_4, _32, _1, _0>>;
  using ValueLayoutrO =
      Layout<Shape<_1, _2, Shape<_2, _2>, Int<HEAD_DIM / 16>>, Stride<_0, _2, Stride<_4, _1>, _8>>;
  using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, Element>{},
                                               ThreadLayoutrO{}, ValueLayoutrO{}));
  using TiledCopyShaperO = Shape<_8, Int<CTA_Q / 8>, _16, Int<HEAD_DIM / 16>>;
  using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

  // Host side kernel arguments
  struct Arguments {
    Element* ptr_O;
    LayoutT const layout_O;
    float* ptr_LSE;
    LayoutLseT const layout_LSE;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_O;
    LayoutT const layout_O;
    float* ptr_LSE;
    LayoutLseT const layout_LSE;
    TMA_O tma_store_O;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.layout_O);
    TMA_O tma_store_O = make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{},
                                      select<0, 2>(TileShape_MNK{}), _1{});  // no mcast for O
    return {args.ptr_O, args.layout_O, args.ptr_LSE, args.layout_LSE, tma_store_O};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& epilogue_params) {}

  template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE void store(Params const& epilogue_params, FrgTensorO const& tOrO,
                            FrgTensorLSE const& lse, SharedStorage& shared_storage,
                            TiledMma tiled_mma, int thread_idx,
                            cute::tuple<int32_t, int32_t> const& block_coord,
                            const int32_t qo_len) {
    auto [q_tile_idx, head_idx] = block_coord;
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOrO_out = convert_type<Element>(tOrO);
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);  // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading V
    cutlass::arch::NamedBarrier::sync(NumMmaThreads,
                                      static_cast<int>(NamedBarriers::kValueEmpty) /*id*/);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), epilogue_params.layout_LSE);
    Tensor gLSE = get_lse_local_tile_tensor(mLSE, Shape<Int<CTA_Q>>{}, /*offset=*/0, qo_len,
                                            head_idx)(_, q_tile_idx);
    Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);  // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    // taccOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
    Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));  // MMA_M
    if (get<1>(taccOcO_row(_0{})) == 0) {
#pragma unroll
      for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < qo_len - q_tile_idx * CTA_Q) {
          gLSE(row) = lse(mi);
        }
      }
    }

    int write_warp_idx = kNWarps - 1;
    if (cutlass::canonical_warp_idx_sync() == write_warp_idx) {
      cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }
    TiledCopyO gmem_tiled_copy_O;
    write_O<NumCopyThreads>(epilogue_params.ptr_O, epilogue_params.tma_store_O, gmem_tiled_copy_O,
                            epilogue_params.layout_O, select<0, 2>(TileShape_MNK{}), sO, q_tile_idx,
                            head_idx, qo_len, write_warp_idx);
  }

  CUTLASS_DEVICE void store_tail() { tma_store_wait<0>(); }

  // Write 0 to output and -inf to LSE
  template <typename SharedStorage>
  CUTLASS_DEVICE void store_zero(Params const& epilogue_params, SharedStorage& shared_storage,
                                 int thread_idx, cute::tuple<int32_t, int32_t> const& block_coord,
                                 const int32_t qo_len) {
    auto [q_tile_idx, head_idx] = block_coord;
    Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.layout_O);
    Tensor gO = get_local_tile_tensor(mO, select<0, 2>(TileShape_MNK{}), head_idx, /*offset=*/0,
                                      qo_len)(_, _, q_tile_idx);  // (M, K)
    Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), epilogue_params.layout_LSE);
    Tensor gLSE = get_lse_local_tile_tensor(mLSE, Shape<Int<CTA_Q>>{}, head_idxm /*offset=*/0,
                                            qo_len)(_, q_tile_idx);

    TiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_fragment_like(tOgO);
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = cute::make_identity_tensor(
        select<0, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.layout_O.shape());
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false,
         /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                qo_len - q_tile_idx * CTA_Q);
    static_assert(CTA_Q <= NumMmaThreads);
    if (thread_idx < qo_len - q_tile_idx * CTA_Q) {
      gLSE(thread_idx) = -math::inf;
    }
  }
};

}  // namespace flashinfer
