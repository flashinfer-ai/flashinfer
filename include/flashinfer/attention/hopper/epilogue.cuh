/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_EPILOGUE_CUH_
#define FLASHINFER_ATTENTION_HOPPER_EPILOGUE_CUH_

#include <cutlass/cutlass.h>

#include "../../math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <int NUM_COPY_THREADS, typename DTypeO, typename TiledCopyO, typename LayoutO,
          typename TileShapeO, typename SMemO>
__forceinline__ __device__ void write_tiled(DTypeO* O, const TiledCopyO& tiled_copy_O,
                                            const LayoutO& layout_O, const TileShapeO& tile_shape_O,
                                            const SMemO& sO, int thread_idx, int qo_tile_idx,
                                            int qo_head_idx, int qo_indptr, int64_t qo_len) {
  Tensor mO = make_tensor(make_gmem_ptr(O + qo_indptr * stride<0>(layout_O)), layout_O);
  Tensor gO =
      get_local_tile_tensor(mO, tile_shape_O, qo_head_idx, 0, qo_len)(_, _, qo_tile_idx);  // (O, D)
  Tensor cO = cute::make_identity_tensor(gO.shape());  // (O, D) -> (o_idx, d_idx)

  ThrCopy thr_copy_O = tiled_copy_O.get_slice(thread_idx);
  Tensor tOgO = thr_copy_O.partition_D(gO);  // (CPY, CPY_O, CPY_D)
  Tensor tOsO = thr_copy_O.partition_S(sO);  // (CPY, CPY_O, CPY_D)
  Tensor tOcO = thr_copy_O.partition_D(cO);  // (CPY, CPY_O, CPY_D)
  Tensor tOsOGroup = flatten_1(tOsO);        // (CPY, (CPY_O, CPY_D))
  Tensor tOgOGroup = flatten_1(tOgO);        // (CPY, (CPY_O, CPY_D))
  Tensor tOcOGroup = flatten_1(tOcO);        // (CPY, (CPY_O, CPY_D))

  const int qo_tile_size = get<0>(tile_shape_O);
  int valid_qo_tile_size = std::min<int>(qo_len - qo_tile_idx * qo_tile_size, qo_tile_size);
  if (valid_qo_tile_size == qo_tile_size) {
    copy(tiled_copy_O, tOsOGroup, tOgOGroup);
  } else {
    // copy if not out of bound
    auto predicate_fn = [&](auto coords) {
      auto s_coords = tOcOGroup(_0{}, coords);
      return elem_less(get<0>(s_coords), valid_qo_tile_size);
    };
    copy_if(tiled_copy_O, predicate_fn, tOsOGroup, tOgOGroup);
  }
}

template <int NUM_COPY_THREADS, typename ElemO, typename TiledCopyO, typename LayoutO,
          typename TileShapeO, typename SMemO>
__forceinline__ __device__ void write_O(ElemO* O, const TiledCopyO& tiled_copy_O,
                                        const LayoutO& layout_O, const TileShapeO& tile_shape_O,
                                        const SMemO& sO, int thread_idx, int qo_tile_idx,
                                        int qo_head_idx, int qo_indptr, int qo_len,
                                        int write_warp_idx) {
  write_tiled<NUM_COPY_THREADS>(O, tiled_copy_O, layout_O, tile_shape_O, sO, thread_idx,
                                qo_tile_idx, qo_head_idx, qo_indptr, qo_len);
}

template <typename Ktraits>
struct CollectiveEpilogue {
  using DTypeO = typename Ktraits::DTypeO;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  static constexpr int NUM_WARPS = Ktraits::NUM_WARPS;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;

  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int NUM_MMA_THREADS = NUM_THREADS - NUM_COPY_THREADS;

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));

  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, DTypeO>;
  using SharedStorage = cute::array_aligned<DTypeO, cute::cosize_v<SmemLayoutO>>;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<_1, int64_t>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;
  using TMA_O = decltype(make_tma_copy(
      GmemTiledCopyOTMA{},
      make_tensor(make_gmem_ptr(static_cast<DTypeO*>(nullptr)), ShapeT{}, StrideT{}), SmemLayoutO{},
      select<0, 1>(TileShape_PDV{}), _1{}));  // no mcast for O

  static constexpr int VEC_SIZE = cute::ceil_div(128, sizeof_bits_v<DTypeO>);
  static_assert(HEAD_DIM_VO % VEC_SIZE == 0);
  static constexpr int NUM_THREADS_PER_ROW = HEAD_DIM_VO / VEC_SIZE;
  static_assert(NUM_MMA_THREADS % NUM_THREADS_PER_ROW == 0);
  static constexpr int NUM_ROWS = NUM_MMA_THREADS / NUM_THREADS_PER_ROW;
  using TiledCopyOAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, DTypeO>;
  using TiledCopyOThrLayout = decltype(cute::make_layout(
      cute::make_shape(Int<NUM_ROWS>{}, Int<NUM_THREADS_PER_ROW>{}), LayoutRight{}));
  using TiledCopyOValLayout =
      decltype(cute::make_layout(cute::make_shape(_1{}, Int<VEC_SIZE>{}), LayoutRight{}));
  using TiledCopyO =
      decltype(make_tiled_copy(TiledCopyOAtom{}, TiledCopyOThrLayout{},  // Thr layout
                               TiledCopyOValLayout{}                     // Val layout
                               ));

  // used for rmem -> smem O copy in fp8 kernel to undo column permutation
  using ThreadLayoutrO = Layout<Shape<_8, Int<CTA_Q / 16>, _4, _1>, Stride<_4, _32, _1, _0>>;
  using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<HEAD_DIM_VO / 16>>,
                               Stride<_0, _2, Stride<_4, _1>, _8>>;
  using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, DTypeO>{},
                                               ThreadLayoutrO{}, ValueLayoutrO{}));
  using TiledCopyShaperO = Shape<_8, Int<CTA_Q / 8>, _16, Int<HEAD_DIM_VO / 16>>;
  using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

  // Host side kernel arguments
  struct Arguments {
    DTypeO* O_ptr;
    LayoutT const layout_O;
    float* lse_ptr;
    LayoutLseT const layout_LSE;
  };

  // Device side kernel params
  struct Params {
    DTypeO* O_ptr;
    LayoutT const layout_O;
    float* lse_ptr;
    LayoutLseT const layout_LSE;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.O_ptr), args.layout_O);
    return {args.O_ptr, args.layout_O, args.lse_ptr, args.layout_LSE};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& epilogue_params) {}

  template <typename BlockCoord, typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE,
            typename TiledMma>
  CUTLASS_DEVICE void store(Params const& epilogue_params, FrgTensorO const& tOrO,
                            FrgTensorLSE const& lse, SharedStorage& shared_storage,
                            TiledMma tiled_mma, int thread_idx, BlockCoord const& block_coord) {
    auto [qo_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOrO_out = convert_type<DTypeO>(tOrO);
    Tensor tOrO_retile = smem_thr_copy_O.retile_S(tOrO_out);  // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tOsO = smem_thr_copy_O.partition_D(sO);            // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading V
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kValueEmpty));
    cute::copy(smem_tiled_copy_O, tOrO_retile, tOsO);
    cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA
    cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.lse_ptr), epilogue_params.layout_LSE);
    Tensor gLSE = get_lse_local_tile_tensor(mLSE, Shape<Int<CTA_Q>>{}, qo_head_idx, qo_indptr,
                                            qo_len)(_, qo_tile_idx);
    Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_PDV{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tOcO = thread_mma.partition_C(cO);  // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0, 0>(tOcO))::value == 2);
    static_assert(decltype(size<0, 1>(tOcO))::value == 2);
    // tOcO has shape ((2, 2, V), MMA_M, MMA_K), we only take only the row indices.
    Tensor tOcO_row = tOcO(make_coord(_0{}, _, _0{}), _, _0{});
    CUTE_STATIC_ASSERT_V(size(lse) == size(tOcO_row));  // MMA_M
    if (epilogue_params.lse_ptr) {                      // don't write to LSE if it's nullptr
      if (get<1>(tOcO_row(_0{})) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
          const int row = get<0>(tOcO_row(mi));
          if (row < qo_len - qo_tile_idx * CTA_Q) {
            gLSE(row) = lse(mi);
          }
        }
      }
    }

    int write_warp_idx = NUM_WARPS - 1;
    // Make sure all MMA WGs finish STSM O
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS,
                                      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    TiledCopyO gmem_tiled_copy_O;
    write_O<NUM_COPY_THREADS>(epilogue_params.O_ptr, gmem_tiled_copy_O, epilogue_params.layout_O,
                              select<0, 1>(TileShape_PDV{}), sO, thread_idx, qo_tile_idx,
                              qo_head_idx, qo_indptr, qo_len, write_warp_idx);
  }

  CUTLASS_DEVICE void store_tail() {
    // tma_store_wait<0>();
  }

  // Write 0 to output and -inf to LSE
  template <typename BlockCoord, typename SharedStorage>
  CUTLASS_DEVICE void store_zero(Params const& epilogue_params, SharedStorage& shared_storage,
                                 int thread_idx, BlockCoord const& block_coord) {
    auto [qo_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.O_ptr), epilogue_params.layout_O);
    Tensor gO = get_local_tile_tensor(mO, select<0, 1>(TileShape_PDV{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, qo_tile_idx);  // (O, D)
    Tensor cO = cute::make_identity_tensor(gO.shape());            // (O, D) -> (o_idx, d_idx)
    Tensor mLSE = make_tensor(make_gmem_ptr(epilogue_params.lse_ptr), epilogue_params.layout_LSE);
    Tensor gLSE = get_lse_local_tile_tensor(mLSE, Shape<Int<CTA_Q>>{}, qo_head_idx, qo_indptr,
                                            qo_len)(_, qo_tile_idx);

    TiledCopyO tiled_copy_O;
    auto thr_copy_O = tiled_copy_O.get_thread_slice(thread_idx);
    Tensor tOgO = thr_copy_O.partition_D(gO);  // (CPY, CPY_O, CPY_D)
    Tensor tOrO = make_fragment_like(tOgO);    // (CPY, CPY_O, CPY_D)
    clear(tOrO);
    Tensor tOcO = thr_copy_O.partition_D(cO);  // (CPY, CPY_O, CPY_D)
    Tensor tOgOGroup = flatten_1(tOgO);        // (CPY, (CPY_O, CPY_D))
    Tensor tOrOGroup = flatten_1(tOrO);        // (CPY, (CPY_O, CPY_D))
    Tensor tOcOGroup = flatten_1(tOcO);        // (CPY, (CPY_O, CPY_D))

    const int qo_tile_size = get<0>(TileShape_PDV{});
    int valid_qo_tile_size = std::min<int>(qo_len - qo_tile_idx * qo_tile_size, qo_tile_size);
    if (valid_qo_tile_size == qo_tile_size) {
      copy(tiled_copy_O, tOrOGroup, tOgOGroup);
    } else {
      auto predicate_fn = [&](auto coords) {
        auto s_coords = tOcOGroup(_0{}, coords);
        return elem_less(get<0>(s_coords), valid_qo_tile_size);
      };
      copy_if(tiled_copy_O, predicate_fn, tOrOGroup, tOgOGroup);
    }

    static_assert(CTA_Q <= NUM_MMA_THREADS);
    if (epilogue_params.lse_ptr) {  // don't write to LSE if it's nullptr
      if (thread_idx < qo_len - qo_tile_idx * CTA_Q) {
        gLSE(thread_idx) = -math::inf;
      }
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_EPILOGUE_CUH_
