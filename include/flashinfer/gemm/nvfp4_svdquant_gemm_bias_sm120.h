/*
 * SM120 SVDQuant fused GEMM: smem-staged per-row bias broadcast for the swapped layout.
 *
 * On the swap_ab path the bias vector is a column broadcast of the swapped problem, and
 * the upstream visitor (Sm90ColBroadcast) loads it with a per-thread predicated gmem
 * copy: every consumer thread privately issues scalar LDG.E.U16 for its accumulator
 * fragment rows, so a 128-row tile issues dozens of uncoalesced loads per thread and the
 * first visit() stalls on the long-scoreboard dependency. The per-column path does not
 * have this problem because upstream Sm90RowBroadcast already stages the vector through
 * shared memory with a cooperative coalesced gmem-to-smem copy.
 *
 * Sm120StagedColBroadcast below is Sm90ColBroadcast (pinned CUTLASS,
 * 3rdparty/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp)
 * with exactly that staging pattern transposed to columns:
 *   - begin() issues a cooperative coalesced gmem-to-smem copy of the tile's CTA_M bias
 *     elements (threads laid out along M), with the same residue predication upstream
 *     applies to its gmem loads; out-of-residue slots are filled with null_default so the
 *     smem-to-register copy needs no predication.
 *   - begin_sync_needed() requests the epilogue's named barrier so the smem fill is
 *     visible before the store loop reads it.
 *   - the first begin_loop() fills the SAME register fragment upstream fills
 *     (make_tensor_like of the partitioned static column tensor, zero-stride N modes
 *     preserved) through the SAME NumericArrayConverter, so every value visit() returns
 *     is bit-identical to the upstream visitor's. visit() is verbatim upstream.
 *   - the nullptr-bias fallback is verbatim upstream (constructor fill + early return);
 *     it touches neither smem nor the barrier.
 *
 * Like the epilogue-overlap collective copy, the class is written against the upstream
 * namespace (a NEW name inside cutlass::epilogue::fusion - no collision), because the
 * visitor/callbacks machinery it composes with lives there and the 3rdparty tree is
 * frozen. The FusionCallbacks specialization is keyed on Sm90TmaWarpSpecialized exactly
 * like the upstream per-row-bias one: the generic Sm120TmaWarpSpecialized forwarder in
 * sm120_callbacks_tma_warpspecialized.hpp routes the SM120 builder here. Its Arguments
 * struct is field-for-field the upstream LinCombPerRowBias Arguments, so the launcher
 * epilogue-argument code is unchanged. Only instantiated by the
 * SVDQ_SM120_BIAS_COALESCED build.
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace epilogue {
namespace fusion {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Column vector broadcast staged through shared memory.
template <int Stages, class CtaTileShapeMNK, class ElementInput_,
          class ElementCompute = cute::remove_pointer_t<ElementInput_>,
          class StrideMNL_ = Stride<_1, _0, _0>,
          int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
          bool EnableNullptr = true  // Fallback scalar broadcast for nullptr params
          >
struct Sm120StagedColBroadcast {
  using StrideMNL = StrideMNL_;
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  static constexpr bool IsDynamicBroadcast =
      is_same_v<remove_cvref_t<decltype(get<0>(StrideMNL{}))>, bool>;
  // The staging copy below is written for the plain static column-vector case only;
  // grouped/scalar-dynamic variants keep the upstream visitor.
  static_assert(!IsArrayOfPointers, "staged column broadcast supports a single bias pointer only");
  static_assert(!IsDynamicBroadcast,
                "staged column broadcast supports a static column-vector stride only");
  using PtrColType = ElementInput const*;

  static_assert(Stages == 0, "Column broadcast doesn't support smem pipelining");
  static_assert(is_static_v<decltype(take<0, 2>(StrideMNL{}))>);
  static_assert(take<0, 2>(StrideMNL{}) == Stride<_1, _0>{});

  // One tile row's worth of bias elements, filled cooperatively in begin().
  struct SharedStorage {
    array_aligned<ElementInput, size<0>(CtaTileShapeMNK{})> smem;
  };

  struct Arguments {
    PtrColType ptr_col = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dCol = {};
  };

  struct Params {
    PtrColType ptr_col = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dCol = {};
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                  Arguments const& args, void* workspace) {
    return {args.ptr_col, ElementCompute(args.null_default), args.dCol};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }

  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  CUTLASS_DEVICE bool is_zero() const { return is_zero_; }

  CUTLASS_HOST_DEVICE
  Sm120StagedColBroadcast() {}

  CUTLASS_HOST_DEVICE
  Sm120StagedColBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params),
        is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    if (EnableNullptr && params.ptr_col == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput* smem = nullptr;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class GS_GTensor, class GS_STensor, class GS_CTensor, class SR_STensor, class RTensor,
            class Residue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GS_GTensor tGS_gCol_, GS_STensor tGS_sCol_, GS_CTensor tGS_cCol_,
                           SR_STensor tSR_sCol_, RTensor tCrCol_, Residue residue_cCol_,
                           Params const& params_)
        : tGS_gCol(tGS_gCol_),
          tGS_sCol(tGS_sCol_),
          tGS_cCol(tGS_cCol_),
          tSR_sCol(tSR_sCol_),
          tCrCol(tCrCol_),
          residue_cCol(residue_cCol_),
          params(params_) {
      if (EnableNullptr && params.ptr_col == nullptr) {
        fill(tCrCol, params.null_default);
      }
    }

    GS_GTensor tGS_gCol;  // (CPY,CPY_M,CPY_N)
    GS_STensor tGS_sCol;  // (CPY,CPY_M,CPY_N)
    GS_CTensor tGS_cCol;  // (CPY,CPY_M,CPY_N)

    SR_STensor tSR_sCol;  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensor tCrCol;       // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    Residue residue_cCol;  // (m, n)
    Params const& params;
    bool filled = false;

    CUTLASS_DEVICE void begin() {
      if (EnableNullptr && params.ptr_col == nullptr) {
        return;
      }

      Tensor tGS_gCol_flt = filter_zeros(tGS_gCol);
      Tensor tGS_sCol_flt = filter_zeros(tGS_sCol);
      Tensor tGS_cCol_flt = filter_zeros(tGS_cCol, tGS_gCol.stride());

      for (int i = 0; i < size(tGS_gCol_flt); ++i) {
        if (get<0>(tGS_cCol_flt(i)) >= size<0>(CtaTileShapeMNK{})) {
          continue;  // OOB of SMEM
        }
        if (elem_less(tGS_cCol_flt(i), residue_cCol)) {
          tGS_sCol_flt(i) = tGS_gCol_flt(i);  // issue async gmem to smem load
        } else {
          // fill OOB values so smem to RF load can issue without predication; the input-typed
          // cast is required (Params keeps null_default compute-typed for the nullptr register
          // fill) and only lands on residue lanes the store never writes
          tGS_sCol_flt(i) = ElementInput(params.null_default);
        }
      }
    }

    CUTLASS_DEVICE bool begin_sync_needed() const {
      // Ensure visibility of the cooperative gmem-to-smem fill; the nullptr fallback
      // never touches smem, so it does not pay the barrier.
      return !(EnableNullptr && params.ptr_col == nullptr);
    }

    CUTLASS_DEVICE void begin_loop(int epi_m, int epi_n) {
      // The column fragment covers every epilogue subtile at once (upstream fills it in
      // begin(); here the fill must wait for the smem barrier). Keyed on a flag rather
      // than (epi_m, epi_n) == (0, 0) because the store loop may visit a filtered
      // subtile range that does not start at the first subtile.
      if (filled || (EnableNullptr && params.ptr_col == nullptr)) {
        return;
      }
      filled = true;

      Tensor tSR_sCol_flt = filter_zeros(tSR_sCol);
      Tensor tCrCol_flt = make_tensor_like<ElementInput>(filter_zeros(tCrCol));
      copy_aligned(tSR_sCol_flt, tCrCol_flt);

      constexpr int FrgSize = size(tCrCol_flt);
      using FrgInput = Array<ElementInput, FrgSize>;
      using FrgCompute = Array<ElementCompute, FrgSize>;
      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FrgSize>;

      Tensor tCrCol_input_frg = recast<FrgInput>(coalesce(tCrCol_flt));
      Tensor tCrCol_compute_frg = recast<FrgCompute>(filter(tCrCol));
      ConvertInput convert_input{};

      tCrCol_compute_frg(_0{}) = convert_input(tCrCol_input_frg(_0{}));
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize> visit(
        Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_col;
      Tensor tCrCol_mn = tCrCol(_, _, _, epi_m, epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_col[i] = tCrCol_mn(epi_v * FragmentSize + i);
      }

      return frg_col;
    }
  };

  template <bool ReferenceSrc,  // do register tensors reference the src or dst layout of the tiled
                                // copy
            class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using ThreadCount = decltype(size(args.tiled_copy));

    auto layout_M = make_layout(M);
    auto layout_N = make_layout(N, repeat_like(N, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dCol));
    Tensor mCol =
        make_tensor(make_gmem_ptr(params.ptr_col), make_layout(layout_M, layout_N, layout_L));
    Tensor gCol = local_tile(mCol(_, _, l), take<0, 2>(args.tile_shape_mnk),
                             make_coord(m, n));  // (CTA_M, CTA_N)
    Tensor sCol = make_tensor(make_smem_ptr(smem),
                              make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})),
                              make_shape(_1{}, _0{}));  // (CTA_M, CTA_N)

    //// G2S: Gmem to Smem, threads along M so the loads coalesce
    auto tiled_g2s =
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementInput>{},
                        Layout<Shape<ThreadCount, _1>, Stride<_1, _0>>{}, Layout<_1>{});
    auto thr_g2s = tiled_g2s.get_slice(args.thread_idx);
    Tensor tGS_gCol = thr_g2s.partition_S(gCol);
    Tensor tGS_sCol = thr_g2s.partition_D(sCol);

    //// G2S: Coord
    Tensor tGS_cCol = thr_g2s.partition_S(args.cD);

    //// S2R: Smem to Reg
    Tensor tSR_sCol = detail::sm90_partition_for_epilogue<ReferenceSrc>(
        sCol, args.epi_tile, args.tiled_copy, args.thread_idx);  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    // Register fragment identical to the upstream direct-gmem visitor: partition the
    // (M,N,L) column tensor the same way, so make_tensor_like preserves the zero-stride
    // N modes and one register holds each unique column element.
    Tensor tCgCol =
        detail::sm90_partition_for_epilogue<ReferenceSrc>(  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
            mCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy,
            args.thread_idx);
    Tensor tCrCol = make_tensor_like<ElementCompute>(tCgCol);  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    return ConsumerStoreCallbacks(tGS_gCol, tGS_sCol, tGS_cCol, tSR_sCol, tCrCol, args.residue_cD,
                                  params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per-row bias, bias staged through shared memory.
// Same semantics (and Arguments) as LinCombPerRowBias; a distinct type so FusionCallbacks
// can key the smem-staged EVT without touching the upstream operation.
template <class ElementOutput, class ElementCompute, class ElementBias = ElementOutput,
          class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
          int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
          FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
struct LinCombPerRowBiasStaged
    : LinCombPerRowBias<ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar,
                        AlignmentBias, RoundStyle> {};

// Upstream Sm90LinCombPerRowBias with the bias leaf swapped for the staged visitor.
template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute,
          class ElementBias = ElementOutput, class ElementSource = ElementOutput,
          class ElementScalar = ElementCompute,
          int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
          FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
using Sm120LinCombPerRowBiasStaged =
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute,
                        RoundStyle>,  // beta * C + (alpha * acc + bias)
            Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,  // beta
            Sm90SrcFetch<ElementSource>,                                  // C
            Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute,
                                RoundStyle>,  // alpha * acc + bias
                    Sm90ScalarBroadcast<ElementScalar, Stride<_0, _0, int64_t>>,  // alpha
                    Sm90AccFetch,                                                 // acc
                    Sm120StagedColBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute,
                                            Stride<_1, _0, int64_t>, AlignmentBias>  // bias
                    >>;

// Keyed on the Sm90 policy exactly like the upstream per-row-bias callbacks; the generic
// Sm120TmaWarpSpecialized-to-Sm90TmaWarpSpecialized forwarder routes the SM120 builder
// here. Arguments is field-for-field the upstream LinCombPerRowBias Arguments.
template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class ElementSource,
          class ElementScalar, int AlignmentBias, FloatRoundStyle RoundStyle, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasStaged<ElementOutput, ElementCompute, ElementBias, ElementSource,
                                    ElementScalar, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Sm120LinCombPerRowBiasStaged<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias,
                                   ElementSource, ElementScalar, AlignmentBias, RoundStyle> {
  using Impl =
      Sm120LinCombPerRowBiasStaged<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias,
                                   ElementSource, ElementScalar, AlignmentBias, RoundStyle>;
  using Operation =
      fusion::LinCombPerRowBiasStaged<ElementOutput, ElementCompute, ElementBias, ElementSource,
                                      ElementScalar, AlignmentBias, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0, _0, int64_t>;
    using StrideBeta = Stride<_0, _0, int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta dBeta = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1, _0, int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return {
          // ternary op : beta * C + (alpha * acc + bias)
          {{beta}, {beta_ptr}, {dBeta}},  // leaf args : beta
          {},                             // leaf args : C
          {
              // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}},   // leaf args : alpha
              {},                                 // leaf args : acc
              {bias_ptr, ElementBias(0), dBias},  // leaf args : bias
              {}                                  // ternary args : multiply_add
          },                                      // end ternary op
          {}                                      // ternary args : multiply_add
      };  // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fusion
}  // namespace epilogue
}  // namespace cutlass
