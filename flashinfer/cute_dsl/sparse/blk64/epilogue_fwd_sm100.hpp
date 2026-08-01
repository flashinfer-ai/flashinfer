/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// CollectiveEpilogueFwd — TMA store for Step 2 (2-acc_O)
// Correction logic has moved to CollectiveMainloopFwd (mainloop_fwd.hpp).
// This file retains TensorStorage (o_exchange, o_staging, sO) and TMA store.
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "pipeline.hpp"
#include "utils.h"

namespace flash {

namespace cute = ::cute;

struct CollectiveEpilogueFwd {
  // ---- Element types ----
  using ElementA = cutlass::bfloat16_t;

  // ---- Tile sizes ----
  static constexpr int kRows = 64;
  static constexpr int kOutputCols = 128;

  // ---- SMEM layout for O ----
  using SmemLayoutO = decltype(cute::coalesce(
      cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<ElementA>{},
                          cute::Shape<cute::Int<kRows>, cute::Int<kOutputCols>>{},
                          cute::Step<cute::_1, cute::_2>{}),
      cute::Shape<cute::_1, cute::_1>{}));
  static constexpr int kOBytes = kRows * kOutputCols * sizeof(ElementA);

  // ---- TMA type aliases ----
  using ShapeTensor = cute::Shape<int, int, int>;
  using StrideO = cute::Stride<cute::Int<kOutputCols>, cute::_1, int>;

  using TMA_O = decltype(cute::make_tma_copy(
      cute::SM90_TMA_STORE{},
      cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementA*>(nullptr)),
                        cute::make_layout(ShapeTensor{}, StrideO{})),
      SmemLayoutO{}));

  // ---- TensorStorage ----
  // o_exchange/o_staging are used by correction (in mainloop) via SharedStorage union.
  // sO is the SMEM staging buffer for TMA store.
  static constexpr int kCSpan = 128;
  static constexpr int kExchangePerWarp = kCSpan * 32;  // 128 cols × 32 lanes = 4096 floats
  struct TensorStorage {
    alignas(16) float o_exchange[4][kExchangePerWarp];  // 4 * 16KB = 64KB
    alignas(16) float o_staging[4][64];                 // 4 * 256B = 1KB (stats only)
    alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutO>> sO;  // 16KB
  };

  // ---- Arguments (host-side) ----
  struct Arguments {
    ElementA* ptr_O;
    float* ptr_LSE = nullptr;  // (batch * heads * num_row_tiles * kRows,) or nullptr
  };

  // ---- Params (device-side) ----
  struct Params {
    TMA_O tma_store_O;
    ShapeTensor shape_O;
    int num_heads = 1;
    int num_row_tiles = 1;
    float* ptr_LSE = nullptr;
  };

  // ---- Convert Arguments -> Params ----
  static Params to_underlying_arguments(Arguments const& args, int rows_padded, int heads,
                                        int batch = 1) {
    using namespace cute;
    int const num_row_tiles = rows_padded / kRows;

    auto shape_o = make_shape(kRows, kOutputCols, batch * heads * num_row_tiles);
    auto stride_o = make_stride(Int<kOutputCols>{}, _1{}, kRows * kOutputCols);
    auto tma_o = make_tma_copy(
        SM90_TMA_STORE{}, make_tensor(make_gmem_ptr(args.ptr_O), make_layout(shape_o, stride_o)),
        SmemLayoutO{});

    return {tma_o, shape_o, heads, num_row_tiles, args.ptr_LSE};
  }

  // ===========================================================================
  // TMA store sO -> GMEM (Epilogue warp 13)
  // ===========================================================================

  struct EpiState {
    int phase = 0;
  };

  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE EpiState tma_store(Params const& params, PipelineOEpi& pipeline_o_epi,
                                    SharedStorage& shared_storage, int head, int row_tile,
                                    int batch, int num_row_tiles, EpiState state) {
    using namespace cute;
    auto& el = shared_storage.tensors.epilogue;

    if (elect_one_sync()) {
      int phase = state.phase;
      pipeline_o_epi.consumer_wait(phase);

      int o_tile_idx = (batch * params.num_heads + head) * params.num_row_tiles + row_tile;
      auto thr_tma_o = params.tma_store_O.get_slice(Int<0>{});
      auto sO = make_tensor(make_smem_ptr(el.sO.begin()), SmemLayoutO{});
      Tensor gO_full = params.tma_store_O.get_tma_tensor(params.shape_O);
      Tensor gO_tile = gO_full(_, _, o_tile_idx);
      cute::copy(params.tma_store_O, thr_tma_o.partition_S(sO), thr_tma_o.partition_D(gO_tile));
      tma_store_arrive();
      tma_store_wait<0>();

      pipeline_o_epi.consumer_release();
      phase ^= 1;
      state.phase = phase;
    }
    return state;
  }
};

}  // namespace flash
