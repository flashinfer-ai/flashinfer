/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"
#include "cutlass_extensions/gemm/threadblock/dq_mma_base.h"
#include "cutlass_extensions/gemm/warp/mma_tensorop_dequantizer.h"
#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Iterators over scales in global memory
    typename IteratorScale_,
    /// Iterators over scales in shared memory
    typename SmemIteratorScale_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Layout of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Converter for B matrix applied immediately after the LDG (before STS)
    typename TransformBAfterLDG_,
    /// Converter for B matrix applited immediately after the LDS
    typename TransformBAfterLDS_,
    /// The quantization operator being used
    WeightOnlyQuantOp QuantOp_>
class DqMmaPipelined<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, IteratorScale_,
                     SmemIteratorScale_, ElementC_, LayoutC_, Policy_, TransformBAfterLDG_,
                     TransformBAfterLDS_, QuantOp_, std::enable_if_t<isFinegrained(QuantOp_)>>
    : public DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_> {
 public:
  ///< Base class
  using Base = DqMmaBase<Shape_, Policy_, typename SmemIteratorScale_::Element, 2, QuantOp_>;

  using Shape = Shape_;          ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;  ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;  ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;    ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;      ///< Layout of accumulator matrix
  using Policy = Policy_;        ///< Policy describing tuning details

  using IteratorScale = IteratorScale_;
  using ElementScale = typename IteratorScale::Element;
  using LayoutScale = typename IteratorScale::Layout;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorScale = SmemIteratorScale_;

  using TransformBAfterLDG = TransformBAfterLDG_;
  using TransformBAfterLDS = TransformBAfterLDS_;

  static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of operand Scale loaded from global memory;
  using FragmentScale = typename IteratorScale::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  using Dequantizer =
      warp::MmaTensorOpDequantizer<Operator, typename Base::WarpGemm, Operand::kB,
                                   typename SmemIteratorScale::Element, LayoutScale, 32, QuantOp>;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for DqMmaPipelined is two (Double-buffered pipeline)
  static_assert((Base::kStages == 2), "DqMmaPipelined requires kStages set to value 2");

  static_assert(Base::SharedStorage::ShapeScale::kRow == Base::kStages, "");
  static_assert(Base::SharedStorage::ShapeScale::kColumn == Shape::kN, "");

 private:
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;
  Dequantizer warp_dequantizer_;

  using WarpFragmentScale = typename Dequantizer::FragmentScale;
  using WarpFragmentZero = typename Dequantizer::FragmentZero;

  using ElementA = typename IteratorA::Element;
  using ElementB = typename IteratorB::Element;
  using LayoutDetailsForB = kernel::LayoutDetailsB<ElementA, ElementB, ArchTag>;

  static constexpr bool RequiresTileInterleave =
      layout::IsColumnMajorTileInterleave<typename LayoutDetailsForB::Layout>::value;
  static_assert(!RequiresTileInterleave ||
                    (RequiresTileInterleave && (Shape::kK == LayoutDetailsForB::ThreadblockK)),
                "Layout K must match threadblockK");

 protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Iterator to write threadblock-scoped tile of scale and zero operand to shared memory
  SmemIteratorScale smem_iterator_scale_;

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  DqMmaPipelined(
      typename Base::SharedStorage&
          shared_storage,    ///< Shared storage needed for internal use by threadblock-scoped GEMM
      int const group_size,  ///< The group size for quantization
      int thread_idx,        ///< ID within the threadblock
      int warp_idx,          ///< ID of warp
      int lane_idx           ///< ID of each thread within a warp
      )
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        warp_dequantizer_(
            {shared_storage.operand_scale.data(), LayoutScale(Shape::kN)},
            {shared_storage.operand_zero.data(), LayoutScale(Shape::kN)},
            (warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN)) / Base::WarpCount::kM,
            lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_iterator_scale_(LayoutScale(Shape::kN), shared_storage.operand_scale.data(),
                             shared_storage.operand_zero.data(), {Base::kStages, Shape::kN},
                             thread_idx, group_size) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterationsForB * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void copy_scales_and_advance(IteratorScale& iterator_scale) {
    using TransformScale =
        NumericArrayConverter<typename SmemIteratorScale::Element, typename FragmentScale::Element,
                              FragmentScale::kElements>;

    FragmentScale tb_frag_scales;
    FragmentScale tb_frag_zeros;
    tb_frag_scales.clear();
    tb_frag_zeros.clear();

    TransformScale transformScale;

    using FragmentElement = typename FragmentScale::Element;

    auto gmem_scale_ptr = iterator_scale.get_scale();
    auto gmem_zero_ptr = iterator_scale.get_zero();

    arch::global_load<FragmentScale, sizeof(FragmentScale)>(tb_frag_scales, gmem_scale_ptr,
                                                            iterator_scale.valid());

    if (gmem_zero_ptr != nullptr) {
      arch::global_load<FragmentScale, sizeof(FragmentScale)>(tb_frag_zeros, gmem_zero_ptr,
                                                              iterator_scale.valid());
    }

    typename TransformScale::result_type tb_frag_scales_fp16 = transformScale(tb_frag_scales);
    typename TransformScale::result_type tb_frag_zeros_fp16;
    if (gmem_zero_ptr != nullptr) tb_frag_zeros_fp16 = transformScale(tb_frag_zeros);

    auto frag_scale_ptr_fp16 =
        reinterpret_cast<typename SmemIteratorScale::Element*>(&tb_frag_scales_fp16);
    auto frag_zero_ptr_fp16 =
        reinterpret_cast<typename SmemIteratorScale::Element*>(&tb_frag_zeros_fp16);
    auto smem_scale_ptr = this->smem_iterator_scale_.get_scale();
    auto smem_zero_ptr = this->smem_iterator_scale_.get_zero();

    if (iterator_scale.valid()) {
      auto smem_offset = cast_smem_ptr_to_uint(smem_scale_ptr);
      arch::shared_store<sizeof(FragmentScale)>(smem_offset, frag_scale_ptr_fp16);

      if (gmem_zero_ptr != nullptr) {
        smem_offset = cast_smem_ptr_to_uint(smem_zero_ptr);
        arch::shared_store<sizeof(FragmentScale)>(smem_offset, frag_zero_ptr_fp16);
      }
    }

    if (iterator_scale.group_size_ == 64) {
      iterator_scale.add_tile_offset({1, 0});
    } else if (iterator_scale.group_size_ == 128) {
      if constexpr (Shape::kK == 128) {
        iterator_scale.add_tile_offset({1, 0});
      } else if constexpr (Shape::kK == 64) {
        if (iterator_scale.row_groupsize64_ & 0x1) {
          iterator_scale.add_tile_offset({1, 0});
        }
      } else {
        static_assert(Shape::kK == 0, "Unsupported k tile shape, can only be 64 or 128");
      }
    }

    iterator_scale.row_groupsize64_++;

    this->smem_iterator_scale_.add_tile_offset({1, 0});
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations,         ///< number of iterations of the mainloop
                  FragmentC& accum,              ///< destination accumulator tile
                  IteratorA iterator_A,          ///< iterator over A operand in global memory
                  IteratorB iterator_B,          ///< iterator over B operand in global memory
                  IteratorScale iterator_scale,  ///< iterator over scale operand in global memory
                  FragmentC const& src_accum) {  ///< source accumulator tile

    //
    // Prologue
    //
    TransformBAfterLDG ldg_converter;
    TransformBAfterLDS lds_converter;

    using TransformA = NumericArrayConverter<typename WarpFragmentA::Element,
                                             typename FragmentA::Element, FragmentA::kElements>;

    // These transforms are mainly to handle when we have bfloat activations and weights in GMEM and
    // want to issue HMMA on architectures older than Ampere. We will convert to FP16 before STS.
    TransformA transformA;

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    this->smem_iterator_A_.store(transformA(tb_frag_A));
    this->smem_iterator_B_.store(ldg_converter(tb_frag_B));

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    copy_scales_and_advance(iterator_scale);

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    WarpFragmentScale warp_frag_scales;
    WarpFragmentZero warp_frag_zero;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    warp_dequantizer_.load(warp_frag_scales, warp_frag_zero);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    warp_dequantizer_.add_pointer_offset(Shape::kN);

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);
    iterator_scale.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing
    // shared memory loads (which have the tighest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          // Write fragments to shared memory
          this->smem_iterator_A_.store(transformA(tb_frag_A));

          this->smem_iterator_B_.store(ldg_converter(tb_frag_B));

          __syncthreads();

          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in
          // shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
            this->smem_iterator_scale_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterationsForB, 0});
            warp_dequantizer_.add_pointer_offset(-Base::kStages * Shape::kN);
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_A_;

        int const warp_tileB_k_compute_offset = warp_mma_k % Base::kNumKIterationsPerWarpBLoad;
        int const warp_tileB_k_load_offset = warp_mma_k / Base::kNumKIterationsPerWarpBLoad;
        // We are just about to finish computing on a fragment of B, so initiate the load for the
        // next fragment.
        if (warp_tileB_k_compute_offset == Base::kNumKIterationsPerWarpBLoad - 1) {
          this->warp_tile_iterator_B_.set_kgroup_index((warp_tileB_k_load_offset + 1) %
                                                       Base::kWarpGemmIterationsForB);
          this->warp_tile_iterator_B_.load(warp_frag_B[(warp_tileB_k_load_offset + 1) % 2]);
          ++this->warp_tile_iterator_B_;
        }

        if (warp_mma_k == 0) {
          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);

          ++iterator_A;
          ++iterator_B;

          copy_scales_and_advance(iterator_scale);

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
          iterator_scale.clear_mask(gemm_k_iterations <= 2);
        }

        typename TransformBAfterLDS::result_type converted_frag_B =
            lds_converter(warp_frag_B[warp_tileB_k_load_offset % 2]);
        warp_dequantizer_.dequantize(converted_frag_B, warp_frag_scales, warp_frag_zero);
        warp_mma(accum, warp_frag_A[warp_mma_k % 2], converted_frag_B, accum,
                 warp_tileB_k_compute_offset);
      }

      // Load the scales needed for the next tile iteration
      warp_dequantizer_.load(warp_frag_scales, warp_frag_zero);
      // Update internal pointer to the set of scales in shared memory
      warp_dequantizer_.add_pointer_offset(Shape::kN);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
