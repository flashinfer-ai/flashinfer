/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "../common/gemm_with_interleave.h"
#include "../primitives/barrier.cuh"
#include "../quantization/fp4_convert.cuh"
#include "../utils/layout.cuh"
#include "../utils/math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
namespace qk_mxfp8_pv_nvfp4_attention {

using namespace cute;

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {
  using Element = typename Ktraits::Element;
  using ElementSF = typename Ktraits::ElementSF;
  using ElementPV = typename Ktraits::ElementPV;
  using ElementSFPV = typename Ktraits::ElementSFPV;
  using ElementDS = typename Ktraits::ElementDS;
  // using TMAElement = Element;
  // using TMAElementSF = typename Ktraits::ElementSF;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using SFQTileShape_MNK = typename Ktraits::SFQTileShape_MNK;
  using SFKTileShape_MNK = typename Ktraits::SFKTileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kHeadDim = Ktraits::kHeadDim;
  static constexpr int BlockMean = Ktraits::BlockMean;
  using GmemTiledCopy = typename Ktraits::GmemTiledCopy;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutDS = typename Ktraits::SmemLayoutDS;
  using SmemLayoutAtomDS = typename Ktraits::SmemLayoutAtomDS;
  using LayoutDS = decltype(blocked_product(
      SmemLayoutAtomDS{}, make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                                      make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeSF =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d // 16, head, batch)
  using LayoutSF = typename Ktraits::LayoutSF;
  using LayoutP = typename Ktraits::LayoutP;
  using LayoutSFP = typename Ktraits::LayoutSFP;
  using SfAtom = typename Ktraits::SfAtom;
  using TMA_Q =
      decltype(make_tma_copy(GmemTiledCopy{},
                             make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                         repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
                             SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));

  using TMA_KV =
      decltype(make_tma_copy(GmemTiledCopy{},
                             make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                         repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
                             take<0, 2>(SmemLayoutK{}), select<1, 2>(TileShape_MNK{}), _1{}));

  using TMA_Vt = decltype(make_tma_copy(
      GmemTiledCopy{},
      make_tensor(make_gmem_ptr(static_cast<ElementPV const*>(nullptr)),
                  repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      take<0, 2>(SmemLayoutVt{}), make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
      _1{}));

  using TMA_DS = decltype(make_tma_copy(
      GmemTiledCopy{},
      make_tensor(make_gmem_ptr(static_cast<ElementDS const*>(nullptr)), LayoutDS{}),
      take<0, 2>(SmemLayoutDS{}), make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
      _1{}));

  using BlkScaledConfig = typename Ktraits::BlkScaledConfig;
  using GmemTiledCopySF = typename Ktraits::GmemTiledCopySF;
  using SmemLayoutSFQ = typename Ktraits::SmemLayoutSFQ;
  using SmemLayoutSFK = typename Ktraits::SmemLayoutSFK;
  using SmemLayoutSFV = typename Ktraits::SmemLayoutSFV;
  using SmemLayoutSFVt = typename Ktraits::SmemLayoutSFVt;

  using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
      SmemLayoutSFQ{}, make_shape(shape<0>(SFQTileShape_MNK{}), shape<2>(SFQTileShape_MNK{})),
      _1{}));  // No programmatic multicast

  using TMA_SFKV = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
      SmemLayoutSFK{}(_, _, cute::Int<0>{}),
      make_shape(shape<1>(SFKTileShape_MNK{}), shape<2>(SFKTileShape_MNK{})), _1{}));

  // SFVt TMA: V^T scale factors (UE4M3, FP4 PV path)
  using LayoutSFPV = typename Ktraits::LayoutSFPV;
  using TMA_SFVt = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{}, make_tensor(static_cast<ElementSFPV const*>(nullptr), LayoutSFPV{}),
      SmemLayoutSFVt{}(_, _, cute::Int<0>{}),
      make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{}));

  using SmemCopyAtomQ = typename Ktraits::SmemCopyAtomQ;
  using SmemCopyAtomKV = typename Ktraits::SmemCopyAtomKV;
  using SmemCopyAtomV = typename Ktraits::SmemCopyAtomV;
  using SmemCopyAtomSF = typename Ktraits::SmemCopyAtomSF;
  using SmemCopyAtomSFPV = typename Ktraits::SmemCopyAtomSFPV;
  using TiledMmaQK = typename Ktraits::TiledMmaQK;
  using TiledMmaPV = typename Ktraits::TiledMmaPV;
  static constexpr int NumMmaThreads = size(TiledMmaQK{});
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
  using PipelineParamsQ = typename Ktraits::PipelineParamsQ;
  using PipelineStateQ = typename Ktraits::PipelineStateQ;
  using EpilogueBarrier = typename Ktraits::EpilogueBarrier;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize((SmemLayoutSFQ{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size((SmemLayoutQ{})) * sizeof_bits<Element>::value));

  static constexpr uint32_t TmaTransactionBytesDS = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutDS{})) * cute::sizeof_bits_v<ElementDS>));
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits<Element>::value));

  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFVt{})) *
                             cute::sizeof_bits_v<ElementSFPV>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutVt{})) * sizeof_bits<ElementPV>::value));

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    ShapeQKV const shape_Q;
    ShapeQKV const unpadded_shape_Q;
    StrideQKV const stride_Q;
    Element const* ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    ShapeQKV const unpadded_shape_K;
    ElementPV const* ptr_Vt;
    ShapeQKV const shape_Vt;
    StrideQKV const stride_Vt;
    ElementSF const* ptr_SFQ{nullptr};
    ShapeSF const shape_SFQ{};
    ElementSF const* ptr_SFK{nullptr};
    ShapeSF const shape_SFK{};
    ElementSFPV const* ptr_SFVt{nullptr};  // V SF: UE4M3 (PV stays FP4)
    ShapeSF const shape_SFVt{};
    cutlass::FastDivmod const group_size_fastdiv;
    float const softmax_scale_log2;
  };

  // Device side kernel params
  struct Params {
    ShapeQKV const shape_Q;
    ShapeQKV const unpadded_shape_Q;
    LayoutSF const layout_SFQ;
    ShapeQKV const shape_K;
    ShapeQKV const unpadded_shape_K;
    LayoutSF const layout_SFK;
    ShapeQKV const shape_Vt;
    LayoutSFPV const layout_SFVt;  // V SF: FP4 PV path layout
    TMA_Q tma_load_Q;
    TMA_SFQ tma_load_SFQ;
    TMA_KV tma_load_K;
    TMA_SFKV tma_load_SFK;
    TMA_Vt tma_load_Vt;
    TMA_SFVt tma_load_SFVt;
    cutlass::FastDivmod const group_size_fastdiv;
    float const softmax_scale_log2;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopy{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_MNK{}), _1{});  // no mcast for Q
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_KV tma_load_K =
        make_tma_copy(GmemTiledCopy{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}),
                      _1{});  // mcast along M mode for this N load, if any
    Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, args.stride_Vt);
    TMA_Vt tma_load_Vt =
        make_tma_copy(GmemTiledCopy{}, mVt, SmemLayoutVt{}(_, _, _0{}),
                      make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                      _1{});  // mcast along M mode for this N load, if any
    LayoutSF layout_sfq = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFQ);
    Tensor mSFQ = make_tensor(make_gmem_ptr(args.ptr_SFQ), layout_sfq);
    TMA_SFQ tma_load_sfq = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFQ, SmemLayoutSFQ{},
        make_shape(shape<0>(SFQTileShape_MNK{}), shape<2>(SFQTileShape_MNK{})), _1{});
    LayoutSF layout_sfk = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFK);
    Tensor mSFK = make_tensor(make_gmem_ptr(args.ptr_SFK), layout_sfk);
    TMA_SFKV tma_load_sfk = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFK, SmemLayoutSFK{}(_, _, _0{}),
        make_shape(shape<1>(SFKTileShape_MNK{}), shape<2>(SFKTileShape_MNK{})), _1{});
    // SFVt: loaded via TMA as ue4m3 (same format as K SF)
    using BlkScaledConfigPV = typename Ktraits::BlkScaledConfigPV;
    LayoutSFPV layout_sfvt = BlkScaledConfigPV::tile_atom_to_shape_SFVt(args.shape_SFVt);
    Tensor mSFVt = make_tensor(make_gmem_ptr(args.ptr_SFVt), layout_sfvt);
    TMA_SFVt tma_load_sfvt = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFVt, SmemLayoutSFVt{}(_, _, _0{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{});
    return {args.shape_Q,
            args.unpadded_shape_Q,
            layout_sfq,
            args.shape_K,
            args.unpadded_shape_K,
            layout_sfk,
            args.shape_Vt,
            layout_sfvt,
            tma_load_Q,
            tma_load_sfq,
            tma_load_K,
            tma_load_sfk,
            tma_load_Vt,
            tma_load_sfvt,
            args.group_size_fastdiv,
            args.softmax_scale_log2};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFQ.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFK.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFVt.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_n_block_max(Params const& mainloop_params, int m_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const seqlen_q = get<0>(mainloop_params.unpadded_shape_Q);
    int const seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
    int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
    if constexpr (Is_causal) {
      n_block_max = std::min(
          n_block_max, cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
    }
    return n_block_max;
  }

  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                             TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);  // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile =
        make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);  // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor =
        zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                             TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);  // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile =
        make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);  // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor =
        zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto partition_fragment_SFA(SFATensor&& sfatensor,
                                                         ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                  thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor,
                                                         ThrMma& thread_mma) {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                  thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE constexpr auto get_layoutSFA_TV(TiledMma& mma) {
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMma& mma) {
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<0>{}, Int<1>{})),
                     _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  template <typename SchedulerParams, typename SharedStorage, typename WorkTileInfo>
  CUTLASS_DEVICE void load(Params const& mainloop_params, SchedulerParams const& scheduler_params,
                           MainloopPipelineQ pipeline_q, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineStateQ& smem_pipe_write_q,
                           PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v,
                           SharedStorage& shared_storage, WorkTileInfo work_tile_info,
                           int& work_idx, int& tile_count_semaphore) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
    int const kv_head = mainloop_params.group_size_fastdiv.divide(bidh);

    int n_block_max = get_n_block_max(mainloop_params, m_block);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q_storage().begin()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
    Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
    Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
    Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
    Tensor mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
    Tensor mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(shape(mainloop_params.layout_SFQ));
    Tensor mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(shape(mainloop_params.layout_SFK));
    Tensor mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(shape(mainloop_params.layout_SFVt));
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _0{}));  // (M, K)
    Tensor gK = local_tile(mK(_, _, kv_head, bidb), select<1, 2>(TileShape_MNK{}),
                           make_coord(_, _0{}));  // (N, K, _)
    Tensor gVt = local_tile(mVt(_, _, kv_head, bidb),
                            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                            make_coord(_0{}, _));  // (N, K, _)
    Tensor gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(SFQTileShape_MNK{}),
                             make_coord(m_block / Ktraits::kSFQTilesPerScaleTile, _0{}));
    Tensor gSFK = local_tile(mSFK(_, _, kv_head, bidb), select<1, 2>(SFKTileShape_MNK{}),
                             make_coord(_, _0{}));
    Tensor gSFVt = local_tile(mSFVt(_, _, kv_head, bidb),
                              make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                              make_coord(_0{}, _));
    auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = block_tma_q.partition_S(gQ);
    Tensor tQsQ = block_tma_q.partition_D(sQ);
    auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
    Tensor tQgSFQ = block_tma_sfq.partition_S(gSFQ);
    Tensor tQsSFQ = block_tma_sfq.partition_D(sSFQ);
    // Rank-adaptive group: flatten all prefix modes except the last one.
    // Handles both kStages=3 (rank 4+) and kStages=2 (rank 3+) TMA partitions.
    auto gp = [](auto t) {
      constexpr int R = decltype(rank(t))::value;
      if constexpr (R <= 2) {
        return t;
      } else {
        return group_modes<0, R - 1>(t);
      }
    };
    auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
    Tensor tKgK = gp(block_tma_k.partition_S(gK));
    Tensor tKsK = gp(block_tma_k.partition_D(sK));
    auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
    Tensor tKgSFK = gp(block_tma_sfk.partition_S(gSFK));
    Tensor tKsSFK = gp(block_tma_sfk.partition_D(sSFK));
    auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
    Tensor tVgVt = gp(block_tma_vt.partition_S(gVt));
    Tensor tVsVt = gp(block_tma_vt.partition_D(sVt));
    auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
    Tensor tVgSFVt = gp(block_tma_sfvt.partition_S(gSFVt));
    Tensor tVsSFVt = gp(block_tma_sfvt.partition_D(sSFVt));
    uint16_t mcast_mask_kv = 0;

    int n_block = n_block_max - 1;
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline_q.producer_acquire(smem_pipe_write_q);
      copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
           tQgQ, tQsQ);
      copy(
          mainloop_params.tma_load_SFQ.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
          tQgSFQ, tQsSFQ);
      ++smem_pipe_write_q;
      pipeline_k.producer_acquire(smem_pipe_write_k);
      copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                           mcast_mask_kv),
           tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
      copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             mcast_mask_kv),
           tKgSFK(_, n_block / Ktraits::kSFKTilesPerScaleTile),
           tKsSFK(_, smem_pipe_write_k.index()));
      ++smem_pipe_write_k;
      pipeline_v.producer_acquire(smem_pipe_write_v);
      copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                            mcast_mask_kv),
           tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
      copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                              mcast_mask_kv),
           tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
      ++smem_pipe_write_v;
    }

    n_block--;
    if (lane_predicate) {
// CUTLASS_PRAGMA_NO_UNROLL
#pragma unroll 2
      for (; n_block >= 0; --n_block) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             mcast_mask_kv),
             tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                               mcast_mask_kv),
             tKgSFK(_, n_block / Ktraits::kSFKTilesPerScaleTile),
             tKsSFK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                              mcast_mask_kv),
             tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
        copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                                mcast_mask_kv),
             tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    }
    ++work_idx;
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipelineQ pipeline_q, MainloopPipeline pipeline_k,
                                MainloopPipeline pipeline_v, PipelineStateQ& smem_pipe_write_q,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    // Issue the epilogue waits
    if (lane_predicate) {
      pipeline_q.producer_tail(smem_pipe_write_q);
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  // Default no-op refill for warp-specialized kernels, whose producer runs
  // independently from the consumer warp groups.
  struct NoOpRefill {
    CUTLASS_DEVICE void refill_k(int) {}
    CUTLASS_DEVICE void refill_v(int) {}
  };

  template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
            typename TmaRefill = NoOpRefill>
  CUTLASS_DEVICE void mma(Params const& mainloop_params, MainloopPipelineQ pipeline_q,
                          MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                          PipelineStateQ& smem_pipe_read_q, PipelineState& smem_pipe_read_k,
                          PipelineState& smem_pipe_read_v, FrgTensorO& tOrO_store,
                          SoftmaxFused& softmax_fused,
                          int n_block_count,  // total N-blocks
                          int thread_idx,     // 0-127 (per-WG MMA thread)
                          int work_idx, int m_block,
                          int wg_id,  // 0 or 1
                          SharedStorage& shared_storage, bool defer_q_release = false,
                          TmaRefill tma_refill = {}) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockK = get<2>(TileShape_MNK{});
    static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;  // 64 (WS) or 128 (Non-WS)
    static constexpr int kSFQTilesPerScaleTile = Ktraits::kSFQTilesPerScaleTile;

    // ============ Smem tensors ============
    Tensor sQ_full =
        make_tensor(make_smem_ptr(shared_storage.smem_q_storage().begin()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
    Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
    Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
    Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

    // ============ Per-WG Q slice (64 M-rows) ============
    auto sQ =
        local_tile(sQ_full, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(wg_id, 0));

    // ============ MMA setup ============
    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    // 8-atom MMA (256 threads) for SFQ partitioning only.
    // consumer_thread_idx: WG1=0-127 → atoms 0-3 (rows 0-63),
    //                      WG2=128-255 → atoms 4-7 (rows 64-127).
    using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
    TiledMmaQK_Full tiled_mma_qk_full;
    int consumer_thread_idx_full = thread_idx + wg_id * NumMmaThreads;
    auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

    // Fragment A from WG's Q half (64 M-rows)
    Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
    Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_, _, Int<0>{}));
    Tensor tOrP = make_tensor_like<ElementPV>(LayoutP{});
    auto select_sfq_tile = [&](int m_block_for_sfq) {
      if constexpr (kSFQTilesPerScaleTile == 1) {
        return sSFQ_full;
      } else {
        return local_tile(sSFQ_full, make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                          make_coord(m_block_for_sfq % kSFQTilesPerScaleTile, _0{}));
      }
    };
    auto sSFQ_active = select_sfq_tile(m_block);
    // SFQ uses the full compute-M MMA layout after selecting the active
    // 64-row half of the hardware's 128-row scale-factor tile.
    Tensor tSrSFQ = partition_fragment_SFA(sSFQ_active, thread_mma_qk_full);
    auto tSrSFK_full = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
    Tensor tSrSFK = tSrSFK_full;
    // SFV is a PV operand scale factor, so its register fragment must match
    // the PV MMA scale-factor layout.
    Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
    Tensor tOrSFP = make_tensor<ElementSFPV>(LayoutSFP{});
    Tensor tOrSFP_flt = filter_zeros(tOrSFP);

    // ============ Smem copy setup ============
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
    Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomV{}, tiled_mma_pv);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sVt));
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    auto v_register_block = [&](auto block_id) { return tOrVt(_, _, block_id); };
    auto v_copy_register_block = [&](auto block_id) { return tOrVt_copy_view(_, _, block_id); };

    auto tile_shape_mnk = tile_shape(tiled_mma_qk);
    // SFQ smem copy: use 8-atom MMA layout + consumer_thread_idx_full
    auto tile_shape_mnk_full = tile_shape(tiled_mma_qk_full);
    auto smem_tiled_copy_SFQ = make_tiled_copy_impl(
        SmemCopyAtomSF{}, get_layoutSFA_TV(tiled_mma_qk_full),
        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
    auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
    Tensor tSsSFQ =
        smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_active));
    Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

    auto smem_tiled_copy_SFK =
        make_tiled_copy_impl(SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_qk),
                             make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
    Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
    Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

    // SFV: PV-path SF format (UE4M3), derive layout from PV MMA
    auto smem_tiled_copy_SFV =
        make_tiled_copy_impl(SmemCopyAtomSFPV{}, get_layoutSFB_TV(tiled_mma_pv),
                             make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
    Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
    Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

    // ============ Helpers ============
    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    int const seqlen_q = get<0>(mainloop_params.unpadded_shape_Q);
    int const seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
    int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
    int const wg_m_offset = wg_id * kBlockMPerWG;  // 0 or 64

    auto copy_k_block = [&](auto block_id) {
      auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
      auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
      copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
      copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
    };

    // Copy one N32 group for all FP8 QK K32 repeats. This is used after
    // an N64 score slot has been retired, so only the K registers needed
    // to rebuild that half of the following score tile are touched.
    auto copy_k_group = [&](auto group_id) {
      auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
      auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
        copy(smem_tiled_copy_K, tSsK_stage(_, group_id, k_block),
             tSrK_copy_view(_, group_id, k_block));
        copy(smem_tiled_copy_SFK, tSsSFK_stage(_, group_id, k_block),
             tSrSFK_copy_view(_, group_id, k_block));
      }
    };

    auto copy_v_block = [&](auto block_id) {
      auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
      auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());
      copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), v_copy_register_block(block_id));
      copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
    };

    auto add_delta_s = [&](auto& acc) { cute::clear(acc); };

    // S accumulator for 64 M-rows × kBlockN (declared here so lambdas can capture)
    Tensor tSrS =
        partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
    Tensor tSrS_converion_view = make_tensor(
        tSrS.data(), qk_mxfp8_pv_nvfp4_attention::convert_to_conversion_layout(tSrS.layout()));
    Tensor AbsMaxP = make_tensor_like<float>(make_layout(
        shape(group<1, 4>(flatten(tSrS_converion_view.layout()(make_coord(_0{}, _), _, _))))));

    auto add_delta_s_slot = [&](auto& acc, auto score_slot) {
      constexpr int ScoreSlot = decltype(score_slot)::value;
      constexpr int MmaNPerScoreSlot = 2;
      constexpr int FirstMmaN = ScoreSlot * MmaNPerScoreSlot;
      static_assert(ScoreSlot == 0 || ScoreSlot == 1, "N64 score slot must be 0 or 1");
      cute::clear(acc(_, _, Int<FirstMmaN>{}));
      cute::clear(acc(_, _, Int<FirstMmaN + 1>{}));
    };

    static_assert(kBlockN == 128, "N64 score-slot reuse requires an N128 attention tile");
    static_assert(decltype(size<2>(tSrS))::value == 4,
                  "The N128 score tile must contain four N32 MMA repeats");
    static_assert(decltype(size<2>(tSrS_converion_view))::value == 2,
                  "The score allocation must expose two N64 conversion slots");
    static_assert(decltype(size<2>(tOrP))::value == 2,
                  "Each N64 score slot must map to one PV block");

    // Causal mask boundary: row is local (0-63), add wg_m_offset for global position
    auto col_limit_causal = [&](int row, int n_block) {
      return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
    };

    // Masking lambda (applies seqlen + causal mask)
    auto apply_mask = [&](auto& tSrS_local, int n_block_local) {
      if constexpr (!Is_causal) {
        if (int(unpadded_seqlen_k - n_block_local * kBlockN) >= int(kBlockN)) {
          return;
        }
      }
      Tensor cS = cute::make_identity_tensor(make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
      Tensor tScS = thread_mma_qk.partition_C(cS);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS_local); ++i) {
        int const col = qk_mxfp8_pv_nvfp4_attention::qk_acc_col_to_k_col(int(get<1>(tScS(i))));
        if constexpr (!Is_causal) {
          if (col >= int(unpadded_seqlen_k - n_block_local * kBlockN)) {
            tSrS_local(i) = -INFINITY;
          }
        } else {
          if (col >= std::min(seqlen_k - n_block_local * kBlockN,
                              col_limit_causal(int(get<0>(tScS(i))), n_block_local))) {
            tSrS_local(i) = -INFINITY;
          }
        }
      }
    };

    auto apply_tail_mask_noncausal = [&](auto& tSrS_local, int tail_valid_cols) {
      Tensor cS = cute::make_identity_tensor(make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
      Tensor tScS = thread_mma_qk.partition_C(cS);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS_local); ++i) {
        if (qk_mxfp8_pv_nvfp4_attention::qk_acc_col_to_k_col(int(get<1>(tScS(i)))) >=
            tail_valid_cols) {
          tSrS_local(i) = -INFINITY;
        }
      }
    };

    // Quantize P to NVFP4 so the second GEMM uses the NVFP4 P/V path.
    int const quant_quad_id = threadIdx.x & 3;
    uint32_t const quant_sfp_mask = uint32_t(0xFF00FF) << ((quant_quad_id & 1) * 8);
    bool const quant_quad_even = (quant_quad_id & 1) == 0;

    auto quantize = [&](auto mma_k, auto acc_conversion_view) {
      Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
      Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
      Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
      Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));
      Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
      Tensor SFP_uint32_view = recast<uint32_t>(SFP);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
        uint32_t& tmp = SFP_uint32_view(i / 4);
        qk_mxfp8_pv_nvfp4_attention::packed_float_to_ue4m3(AbsMaxP_stagek(i), AbsMaxP_stagek(i + 1),
                                                           AbsMaxP_stagek(i + 2),
                                                           AbsMaxP_stagek(i + 3), tmp);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          qk_mxfp8_pv_nvfp4_attention::packed_float_to_e2m1(
              acc_conversion_stagek(make_coord(_0{}, i), mma_m),
              acc_conversion_stagek(make_coord(_1{}, i), mma_m),
              acc_conversion_stagek(make_coord(_2{}, i), mma_m),
              acc_conversion_stagek(make_coord(_3{}, i), mma_m),
              acc_conversion_stagek(make_coord(_4{}, i), mma_m),
              acc_conversion_stagek(make_coord(_5{}, i), mma_m),
              acc_conversion_stagek(make_coord(_6{}, i), mma_m),
              acc_conversion_stagek(make_coord(_7{}, i), mma_m), tOrP_uint32_view(i, mma_m));
        }
        uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
        uint32_t peer_sfp = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
        if (quant_quad_even) {
          tOrSFP_uint32_view(_0{}, mma_m) =
              (local_sfp & quant_sfp_mask) | ((peer_sfp & quant_sfp_mask) << 8);
        } else {
          tOrSFP_uint32_view(_0{}, mma_m) =
              (peer_sfp & quant_sfp_mask) | ((local_sfp & quant_sfp_mask) >> 8);
        }
      }
    };

    // ============ Load Q (both WGs, each reads its 64-row half) ============
    consumer_wait(pipeline_q, smem_pipe_read_q);
    copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
    if (!defer_q_release) {
      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;
    }

    if constexpr (!Is_causal) {
      // Symmetric N64/N64 score-slot reuse. Build the first N128 score tile,
      // then retire each N64 half after quantization and immediately reuse
      // those registers to construct the corresponding half of the next QK
      // tile while packed P remains live in its dedicated PV fragment.
      auto refill_score_slot = [&](auto score_slot, int refill_tile_idx) {
        constexpr int ScoreSlot = decltype(score_slot)::value;
        constexpr int FirstMmaN = ScoreSlot * 2;
        static_assert(ScoreSlot == 0 || ScoreSlot == 1, "N64 score slot must be 0 or 1");

        copy_k_group(Int<FirstMmaN>{});
        copy_k_group(Int<FirstMmaN + 1>{});
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
          if constexpr (ScoreSlot == 0) {
            cute::gemm(tiled_mma_qk,
                       make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                       make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, _0{}),
                       tSrS(_, _0{}, _0{}));
            cute::gemm(tiled_mma_qk,
                       make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                       make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, _1{}),
                       tSrS(_, _0{}, _1{}));
          } else {
            cute::gemm(tiled_mma_qk,
                       make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                       make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, _2{}),
                       tSrS(_, _0{}, _2{}));
            cute::gemm(tiled_mma_qk,
                       make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                       make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, _3{}),
                       tSrS(_, _0{}, _3{}));
          }
        }
      };

      auto process_score_slot = [&](auto score_slot, auto has_next_tile,
                                    int refill_tile_idx) __attribute__((always_inline)) {
        constexpr int ScoreSlot = decltype(score_slot)::value;
        constexpr bool HasNextTile = decltype(has_next_tile)::value;
        static_assert(ScoreSlot == 0 || ScoreSlot == 1, "N64 score slot must be 0 or 1");

        softmax_fused.template softmax_quantize_n64<ScoreSlot, Is_causal>(
            tSrS, AbsMaxP, mainloop_params.softmax_scale_log2, tOrP);
        if constexpr (ScoreSlot == 0) {
          // Preserve the compiler-visible convergence point while the
          // first score half changes from softmax input to packed P and
          // then back into a QK accumulator.
          __syncwarp();
        }
        quantize(score_slot, tSrS_converion_view);

        if constexpr (HasNextTile) {
          if constexpr (ScoreSlot == 0) {
            consumer_wait(pipeline_k, smem_pipe_read_k);
          }
          add_delta_s_slot(tSrS, score_slot);
          refill_score_slot(score_slot, refill_tile_idx);
          if constexpr (ScoreSlot == 1) {
            pipeline_k.consumer_release(smem_pipe_read_k);
            ++smem_pipe_read_k;
            tma_refill.refill_k(refill_tile_idx);
          }
        }

        if constexpr (ScoreSlot == 0) {
          consumer_wait(pipeline_v, smem_pipe_read_v);
        }
        copy_v_block(score_slot);
        cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, score_slot), tOrSFP(_, _, score_slot)),
                   make_zip_tensor(v_register_block(score_slot), tOrSFVt(_, _, score_slot)),
                   tOrO_store);
      };

      static_assert(decltype(size<2>(tSrQ))::value == 4,
                    "The slot-interleaved path requires four QK K32 repeats");

      auto softmax_score_slot = [&](auto score_slot) __attribute__((always_inline)) {
        constexpr int ScoreSlot = decltype(score_slot)::value;
        softmax_fused.template softmax_quantize_n64<ScoreSlot, Is_causal>(
            tSrS, AbsMaxP, mainloop_params.softmax_scale_log2, tOrP);
        if constexpr (ScoreSlot == 0) {
          __syncwarp();
        }
      };

      auto pack_score_slot = [&](auto score_slot) __attribute__((always_inline)) {
        quantize(score_slot, tSrS_converion_view);
      };

      auto prepare_score_slot = [&](auto score_slot) __attribute__((always_inline)) {
        softmax_score_slot(score_slot);
        pack_score_slot(score_slot);
      };

      auto gemm_score_chunk_k = [&](auto mma_n, auto k_block) __attribute__((always_inline)) {
        constexpr int MmaN = decltype(mma_n)::value;
        cute::gemm(tiled_mma_qk,
                   make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block))(_, _0{}),
                   make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block))(_, Int<MmaN>{}),
                   tSrS(_, _0{}, Int<MmaN>{}));
      };

      auto gemm_score_slot_k = [&](auto score_slot, auto k_block) __attribute__((always_inline)) {
        constexpr int ScoreSlot = decltype(score_slot)::value;
        constexpr int FirstMmaN = ScoreSlot * 2;
        gemm_score_chunk_k(Int<FirstMmaN>{}, k_block);
        gemm_score_chunk_k(Int<FirstMmaN + 1>{}, k_block);
      };

      auto consume_pv_slot = [&](auto score_slot) __attribute__((always_inline)) {
        constexpr int ScoreSlot = decltype(score_slot)::value;
        if constexpr (ScoreSlot == 0) {
          consumer_wait(pipeline_v, smem_pipe_read_v);
        }
        copy_v_block(score_slot);
        cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, score_slot), tOrSFP(_, _, score_slot)),
                   make_zip_tensor(v_register_block(score_slot), tOrSFVt(_, _, score_slot)),
                   tOrO_store);
      };

      auto apply_noncausal_tail_mask = [&](auto& scores, int n_block_local) {
        int const valid_cols = int(unpadded_seqlen_k - n_block_local * kBlockN);
        if (valid_cols >= kBlockN) {
          return;
        }

        // Derive logical coordinates directly from the FP8 QK accumulator
        // layout. This avoids keeping a 64x128 identity tensor live solely
        // for the (at most one) partial K tile.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(scores); ++i) {
          int const lane = thread_idx & 31;
          int const col = ((i >> 4) << 5) + ((lane & 3) << 3) + (i & 7);
          if (col >= valid_cols) {
            scores(i) = -INFINITY;
          }
        }
      };

      clear(tOrO_store);

      // Prologue: construct a complete N128 score tile.
      consumer_wait(pipeline_k, smem_pipe_read_k);
      add_delta_s_slot(tSrS, _0{});
      add_delta_s_slot(tSrS, _1{});
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
        copy_k_block(k_block);
        cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                   make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS);
      }
      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;
      tma_refill.refill_k(0);

      int const first_n_block = n_block_count - 1;
      apply_noncausal_tail_mask(tSrS, first_n_block);
      softmax_fused.template prepare_online_softmax_n128<true, Is_causal>(
          tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);

// Branch-free steady state: consume P0/P1 while rebuilding QK0/QK1.
#pragma unroll 1
      for (int tile_idx = 0; tile_idx < n_block_count - 1; ++tile_idx) {
        // Fine-grained one-shot phase probes let the two consumer warp
        // groups enter the steady-state loop at complementary scalar/MMA
        // positions.  Point 9 sits between slot-0 softmax and packing;
        // the older point 1 is after both operations.
        prepare_score_slot(_0{});

        consumer_wait(pipeline_k, smem_pipe_read_k);
        add_delta_s_slot(tSrS, _0{});
        copy_k_group(_0{});
        copy_k_group(_1{});

        // The first independent QK group creates tensor-scoreboard room
        // for the other N64 half's scalar softmax and FP4 conversion.
        gemm_score_slot_k(_0{}, _0{});
        prepare_score_slot(_1{});
        gemm_score_slot_k(_0{}, _1{});
        gemm_score_slot_k(_0{}, _2{});
        gemm_score_slot_k(_0{}, _3{});
        consume_pv_slot(_0{});

        add_delta_s_slot(tSrS, _1{});
        copy_k_group(_2{});
        copy_k_group(_3{});
        gemm_score_slot_k(_1{}, _0{});
        gemm_score_slot_k(_1{}, _1{});
        gemm_score_slot_k(_1{}, _2{});
        gemm_score_slot_k(_1{}, _3{});
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;
        tma_refill.refill_k(tile_idx + 1);
        consume_pv_slot(_1{});

        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
        tma_refill.refill_v(tile_idx);

        softmax_fused.template prepare_online_softmax_n128<false, Is_causal>(
            tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);
        softmax_fused.rescale_o_inplace(tOrO_store);
      }

      // Drain the final tile without a has-next predicate in the hot loop.
      process_score_slot(_0{}, cute::false_type{}, n_block_count);
      process_score_slot(_1{}, cute::false_type{}, n_block_count);

      pipeline_v.consumer_release(smem_pipe_read_v);
      ++smem_pipe_read_v;
      tma_refill.refill_v(n_block_count - 1);

      softmax_fused.finalize(tOrO_store);
      return;
    }

    bool is_first_compute = true;

    // Causal tiles use the conventional online-softmax path. Both
    // consumer warp groups traverse every K/V tile for their own M64 rows.

#pragma unroll 1
    for (int tile_idx = 0; tile_idx < n_block_count; ++tile_idx) {
      int n_block = n_block_count - 1 - tile_idx;

      // --- K: both WGs wait for data ready ---
      consumer_wait(pipeline_k, smem_pipe_read_k);

      // --- QK GEMM ---
      Tensor tSrS_local =
          partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
      Tensor tSrS_local_cv = make_tensor(
          tSrS_local.data(),
          qk_mxfp8_pv_nvfp4_attention::convert_to_conversion_layout(tSrS_local.layout()));

      // Default path: streaming K copy + GEMM (no early release)
      copy_k_block(_0{});
      add_delta_s(tSrS_local);
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
        cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                   make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS_local);
        if (k_block < size<2>(tSrQ) - 1) {
          copy_k_block(k_block + 1);
        }
      }
      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;
      // Non-WS refill: thread 0 issues TMA for the next K tile
      tma_refill.refill_k(tile_idx);

      // Apply the logical mask before the online-softmax update.
      if constexpr (Is_causal) {
        apply_mask(tSrS_local, n_block);
      } else if (tile_idx == 0) {
        apply_tail_mask_noncausal(tSrS_local, int(unpadded_seqlen_k - n_block * kBlockN));
      }

      if (is_first_compute) {
        softmax_fused.template online_softmax_with_quant</*FirstTile=*/true,
                                                         /*InfCheck=*/Is_causal>(
            tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
      } else {
        softmax_fused
            .template online_softmax_with_quant_direct_norm_nonfirst</*InfCheck=*/Is_causal>(
                tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
      }

      // --- V: wait, PV GEMM, release ---
      // R19 PV ordering: WG1 first (it finished QK+softmax first via math_order).
      // WG1's PV overlaps with WG0's remaining softmax → TC stays busy.
      if (!is_first_compute) {
        softmax_fused.rescale_o_inplace(tOrO_store);
      }
      consumer_wait(pipeline_v, smem_pipe_read_v);
      copy_v_block(_0{});
      quantize(_0{}, tSrS_local_cv);

      CUTLASS_PRAGMA_UNROLL
      for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
        cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                   make_zip_tensor(v_register_block(v_block), tOrSFVt(_, _, v_block)), tOrO_store);
        if (v_block < size<2>(tOrP) - 1) {
          copy_v_block(v_block + 1);
          quantize(v_block + 1, tSrS_local_cv);
        }
      }
      is_first_compute = false;
      pipeline_v.consumer_release(smem_pipe_read_v);
      ++smem_pipe_read_v;
      // Non-WS refill: thread 0 issues TMA for the next V tile
      tma_refill.refill_v(tile_idx);
    }

    softmax_fused.finalize(tOrO_store);
    return;
  }
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
