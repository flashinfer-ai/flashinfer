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

#include <type_traits>

#include "../primitives/barrier.cuh"
#include "../quantization/fp4_convert.cuh"
#include "../utils/layout.cuh"
#include "../utils/math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
namespace nvfp4_attention {

using namespace cute;

template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd {
  using Element = typename Ktraits::Element;
  using ElementSF = typename Ktraits::ElementSF;
  using ElementDS = typename Ktraits::ElementDS;

  using TileShape_MNK = typename Ktraits::TileShape_MNK;
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
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapeSF = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
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
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
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
      SmemLayoutSFQ{}, make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})), _1{}));

  using TMA_SFKV = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
      SmemLayoutSFK{}(_, _, cute::Int<0>{}),
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})), _1{}));

  using TMA_SFVt = decltype(make_tma_copy<uint16_t>(
      GmemTiledCopySF{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSF{}),
      SmemLayoutSFVt{}(_, _, cute::Int<0>{}),
      make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{}));

  using SmemCopyAtomQ = typename Ktraits::SmemCopyAtomQ;
  using SmemCopyAtomKV = typename Ktraits::SmemCopyAtomKV;
  using SmemCopyAtomSF = typename Ktraits::SmemCopyAtomSF;
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

  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize((SmemLayoutSFQ{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size((SmemLayoutQ{})) * sizeof_bits<Element>::value));

  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFK{})) * cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutDS{})) * cute::sizeof_bits_v<ElementDS>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits<Element>::value));

  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFVt{})) *
                             cute::sizeof_bits_v<ElementSF>) +
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutVt{})) * sizeof_bits<Element>::value));

  struct Arguments {
    Element const* ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element const* ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    ShapeQKV const unpadded_shape_K;
    Element const* ptr_Vt;
    ShapeQKV const shape_Vt;
    StrideQKV const stride_Vt;
    ElementSF const* ptr_SFQ{nullptr};
    ShapeSF const shape_SFQ{};
    ElementSF const* ptr_SFK{nullptr};
    ShapeSF const shape_SFK{};
    ElementSF const* ptr_SFVt{nullptr};
    ShapeSF const shape_SFVt{};
    ElementDS const* ptr_ds;
    ShapeQKV const shape_ds;
    StrideQKV const stride_ds;
    float const softmax_scale_log2;
  };

  struct Params {
    ShapeQKV const shape_Q;
    LayoutSF const layout_SFQ;
    ShapeQKV const shape_K;
    ShapeQKV const unpadded_shape_K;
    LayoutSF const layout_SFK;
    ShapeQKV const shape_Vt;
    LayoutSF const layout_SFVt;
    LayoutDS const layout_DS;
    TMA_Q tma_load_Q;
    TMA_SFQ tma_load_SFQ;
    TMA_KV tma_load_K;
    TMA_SFKV tma_load_SFK;
    TMA_Vt tma_load_Vt;
    TMA_SFVt tma_load_SFVt;
    TMA_DS tma_load_DS;
    float const softmax_scale_log2;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopy{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{});
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_KV tma_load_K = make_tma_copy(GmemTiledCopy{}, mK, SmemLayoutK{}(_, _, _0{}),
                                      select<1, 2>(TileShape_MNK{}), _1{});
    Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, args.stride_Vt);
    TMA_Vt tma_load_Vt =
        make_tma_copy(GmemTiledCopy{}, mVt, SmemLayoutVt{}(_, _, _0{}),
                      make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{});
    auto [Seqlen_Q, Seqlen_K, HeadNum, Batch] = args.shape_ds;
    LayoutDS layout_ds = tile_to_shape(
        SmemLayoutAtomDS{}, make_shape(Seqlen_Q, Seqlen_K, HeadNum, Batch), Step<_2, _1, _3, _4>{});
    Tensor mDS = make_tensor(make_gmem_ptr(args.ptr_ds), layout_ds);
    TMA_DS tma_load_ds =
        make_tma_copy(GmemTiledCopy{}, mDS, SmemLayoutDS{}(_, _, _0{}),
                      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{});
    LayoutSF layout_sfq = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFQ);
    Tensor mSFQ = make_tensor(make_gmem_ptr(args.ptr_SFQ), layout_sfq);
    TMA_SFQ tma_load_sfq = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFQ, SmemLayoutSFQ{},
        make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{})), _1{});
    LayoutSF layout_sfk = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFK);
    Tensor mSFK = make_tensor(make_gmem_ptr(args.ptr_SFK), layout_sfk);
    TMA_SFKV tma_load_sfk = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFK, SmemLayoutSFK{}(_, _, _0{}),
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{})), _1{});
    LayoutSF layout_sfvt = BlkScaledConfig::tile_atom_to_shape_SFVt(args.shape_SFVt);
    Tensor mSFVt = make_tensor(make_gmem_ptr(args.ptr_SFVt), layout_sfvt);
    TMA_SFVt tma_load_sfvt = make_tma_copy<uint16_t>(
        GmemTiledCopySF{}, mSFVt, SmemLayoutSFVt{}(_, _, _0{}),
        make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), _1{});
    return {args.shape_Q, layout_sfq,    args.shape_K, args.unpadded_shape_K,
            layout_sfk,   args.shape_Vt, layout_sfvt,  layout_ds,
            tma_load_Q,   tma_load_sfq,  tma_load_K,   tma_load_sfk,
            tma_load_Vt,  tma_load_sfvt, tma_load_ds,  args.softmax_scale_log2};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFQ.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFK.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFVt.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_DS.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_n_block_max(Params const& mainloop_params, int m_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const seqlen_q = get<0>(mainloop_params.shape_Q);
    int const seqlen_k = get<0>(mainloop_params.shape_K);
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

    auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);

    auto a_tile =
        make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);

    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);

    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);

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

    auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);

    auto a_tile =
        make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);

    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);

    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
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
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto atile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));

    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);

    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    auto btile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<0>{}, Int<1>{})),
                     _));

    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);

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

    int n_block_max = get_n_block_max(mainloop_params, m_block);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
    Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
    Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
    Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});
    Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
    Tensor mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
    Tensor mDS = mainloop_params.tma_load_DS.get_tma_tensor(shape(mainloop_params.layout_DS));
    Tensor mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(shape(mainloop_params.layout_SFQ));
    Tensor mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(shape(mainloop_params.layout_SFK));
    Tensor mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(shape(mainloop_params.layout_SFVt));
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};
    Tensor gQ =
        local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
    Tensor gK =
        local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gVt = local_tile(mVt(_, _, bidh, bidb),
                            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                            make_coord(_0{}, _));
    Tensor gDS = [&] {
      if constexpr (BlockMean) {
        return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}),
                          make_coord(m_block, _));
      } else {
        return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}),
                          make_coord(_0{}, _));
      }
    }();
    Tensor gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}),
                             make_coord(m_block, _0{}));
    Tensor gSFK =
        local_tile(mSFK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    Tensor gSFVt = local_tile(mSFVt(_, _, bidh, bidb),
                              make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                              make_coord(_0{}, _));
    auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = block_tma_q.partition_S(gQ);
    Tensor tQsQ = block_tma_q.partition_D(sQ);
    auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
    Tensor tQgSFQ = block_tma_sfq.partition_S(gSFQ);
    Tensor tQsSFQ = block_tma_sfq.partition_D(sSFQ);
    auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
    Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
    Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
    auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
    Tensor tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
    Tensor tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));
    auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
    Tensor tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
    Tensor tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));
    auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
    Tensor tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
    Tensor tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));
    auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
    Tensor tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
    Tensor tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));
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
           tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
      copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                            mcast_mask_kv),
           tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
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
#pragma unroll 2
      for (; n_block >= 0; --n_block) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             mcast_mask_kv),
             tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                               mcast_mask_kv),
             tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
        copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                              mcast_mask_kv),
             tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
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

  CUTLASS_DEVICE void load_tail(MainloopPipelineQ pipeline_q, MainloopPipeline pipeline_k,
                                MainloopPipeline pipeline_v, PipelineStateQ& smem_pipe_write_q,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      pipeline_q.producer_tail(smem_pipe_write_q);
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  struct NoOpRefill {
    CUTLASS_DEVICE void refill_k(int) {}
    CUTLASS_DEVICE void refill_v(int) {}
  };

  template <typename SharedStorage, typename FrgTensorO, typename SoftmaxFused,
            typename MathOrderBarrier, typename TmaRefill = NoOpRefill>
  CUTLASS_DEVICE void mma(Params const& mainloop_params, MainloopPipelineQ pipeline_q,
                          MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                          PipelineStateQ& smem_pipe_read_q, PipelineState& smem_pipe_read_k,
                          PipelineState& smem_pipe_read_v, FrgTensorO& tOrO_store,
                          SoftmaxFused& softmax_fused, int n_block_count, int thread_idx,
                          int work_idx, int m_block, int wg_id, SharedStorage& shared_storage,
                          MathOrderBarrier& math_order, TmaRefill tma_refill = {}) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockK = get<2>(TileShape_MNK{});
    static constexpr int kBlockMPerWG = Ktraits::kBlockMPerWG;

    Tensor sQ_full = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
    Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
    Tensor sSFQ_full = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
    Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
    Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});

    auto sQ =
        local_tile(sQ_full, make_shape(Int<kBlockMPerWG>{}, Int<kBlockK>{}), make_coord(wg_id, 0));

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    using TiledMmaQK_Full = typename Ktraits::TiledMmaQK_Full;
    TiledMmaQK_Full tiled_mma_qk_full;
    int consumer_thread_idx_full = thread_idx + wg_id * NumMmaThreads;
    auto thread_mma_qk_full = tiled_mma_qk_full.get_thread_slice(consumer_thread_idx_full);

    Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_, _, Int<0>{}));
    Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_, _, Int<0>{}));
    Tensor tOrP = make_tensor_like<Element>(LayoutP{});

    Tensor tSrSFQ = partition_fragment_SFA(sSFQ_full, thread_mma_qk_full);
    Tensor tSrSFK = partition_fragment_SFB(sSFK(_, _, Int<0>{}), thread_mma_qk);
    Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_, _, Int<0>{}), thread_mma_pv);
    Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});
    Tensor tOrSFP_flt = filter_zeros(tOrSFP);

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
    Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(as_position_independent_swizzle_tensor(sVt));
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

    auto tile_shape_mnk = tile_shape(tiled_mma_qk);

    auto tile_shape_mnk_full = tile_shape(tiled_mma_qk_full);
    auto smem_tiled_copy_SFQ = make_tiled_copy_impl(
        SmemCopyAtomSF{}, get_layoutSFA_TV(tiled_mma_qk_full),
        make_shape(size<0>(tile_shape_mnk_full), size<2>(tile_shape_mnk_full)));
    auto smem_thr_copy_SFQ = smem_tiled_copy_SFQ.get_thread_slice(consumer_thread_idx_full);
    Tensor tSsSFQ =
        smem_thr_copy_SFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ_full));
    Tensor tSrSFQ_copy_view = smem_thr_copy_SFQ.retile_D(tSrSFQ);

    auto smem_tiled_copy_SFK =
        make_tiled_copy_impl(SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_qk),
                             make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFK = smem_tiled_copy_SFK.get_thread_slice(thread_idx);
    Tensor tSsSFK = smem_thr_copy_SFK.partition_S(as_position_independent_swizzle_tensor(sSFK));
    Tensor tSrSFK_copy_view = smem_thr_copy_SFK.retile_D(tSrSFK);

    auto smem_tiled_copy_SFV =
        make_tiled_copy_impl(SmemCopyAtomSF{}, get_layoutSFB_TV(tiled_mma_pv),
                             make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFV = smem_tiled_copy_SFV.get_thread_slice(thread_idx);
    Tensor tOsSFVt = smem_thr_copy_SFV.partition_S(as_position_independent_swizzle_tensor(sSFVt));
    Tensor tOrSFVt_copy_view = smem_thr_copy_SFV.retile_D(tOrSFVt);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    int const seqlen_q = get<0>(mainloop_params.shape_Q);
    int const seqlen_k = get<0>(mainloop_params.shape_K);
    int const unpadded_seqlen_k = get<0>(mainloop_params.unpadded_shape_K);
    int const wg_m_offset = wg_id * kBlockMPerWG;

    auto copy_k_block = [&](auto block_id) {
      auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
      auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());
      copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
      copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
    };

    auto copy_v_block = [&](auto block_id) {
      auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
      auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());
      copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
      copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
    };

    auto add_delta_s = [&](auto& acc) {
      auto acc_float4 = recast<float4>(acc);
      int quad_id = (thread_idx % 4) * 2;
      if constexpr (std::is_same_v<ElementDS, float>) {
        auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
        for (int i = 0; i < 4; i++) {
          auto num = quad_id + i * 8;
          float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
          float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
          acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
          acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
          acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
          acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
        }
      } else {
        using ElementDSVec = cutlass::Array<ElementDS, 4>;
        auto tSsDS_stage = recast<ElementDSVec>(sDS(_, _, smem_pipe_read_k.index()));
        cutlass::NumericConverter<float, ElementDS> convert;
        for (int i = 0; i < 4; i++) {
          auto num = quad_id + i * 8;
          ElementDSVec ds0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
          ElementDSVec ds1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
          float4 delta_s_0;
          float4 delta_s_1;
          delta_s_0.x = convert(ds0[0]);
          delta_s_0.y = convert(ds0[1]);
          delta_s_0.z = convert(ds0[2]);
          delta_s_0.w = convert(ds0[3]);
          delta_s_1.x = convert(ds1[0]);
          delta_s_1.y = convert(ds1[1]);
          delta_s_1.z = convert(ds1[2]);
          delta_s_1.w = convert(ds1[3]);
          acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
          acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
          acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
          acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
        }
      }
    };

    Tensor tSrS =
        partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
    Tensor tSrS_converion_view =
        make_tensor(tSrS.data(), nvfp4_attention::convert_to_conversion_layout(tSrS.layout()));
    Tensor AbsMaxP = make_tensor_like<float>(make_layout(
        shape(group<1, 4>(flatten(tSrS_converion_view.layout()(make_coord(_0{}, _), _, _))))));

    auto col_limit_causal = [&](int row, int n_block) {
      return row + wg_m_offset + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
    };

    auto apply_mask = [&](auto& tSrS_local, int n_block_local) {
      int const valid_cols = int(unpadded_seqlen_k - n_block_local * kBlockN);
      if constexpr (!Is_causal) {
        if (valid_cols >= kBlockN) {
          return;
        }
      } else {
        if (valid_cols >= kBlockN && col_limit_causal(0, n_block_local) >= kBlockN) {
          return;
        }
      }

      Tensor cS = cute::make_identity_tensor(make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
      Tensor tScS = thread_mma_qk.partition_C(cS);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS_local); ++i) {
        int const col = nvfp4_attention::qk_acc_col_to_k_col(int(get<1>(tScS(i))));
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

    auto quantize = [&](auto mma_k, auto acc_conversion_view) {
      Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
      Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
      Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
      Tensor SFP_uint32_view = recast<uint32_t>(SFP);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
        uint32_t& tmp = SFP_uint32_view(i / 4);
        nvfp4_attention::packed_float_to_ue4m3(AbsMaxP_stagek(i), AbsMaxP_stagek(i + 1),
                                               AbsMaxP_stagek(i + 2), AbsMaxP_stagek(i + 3), tmp);
      }
      int const quad_id = threadIdx.x & 3;
      uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
      Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
      Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          nvfp4_attention::packed_float_to_e2m1(acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                                                acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                                                tOrP_uint32_view(i, mma_m));
        }
        uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
        uint32_t peer_sfp = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
        if ((quad_id & 1) == 0) {
          tOrSFP_uint32_view(_0{}, mma_m) = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
        } else {
          tOrSFP_uint32_view(_0{}, mma_m) = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
        }
      }
    };

    consumer_wait(pipeline_q, smem_pipe_read_q);
    copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
    pipeline_q.consumer_release(smem_pipe_read_q);
    ++smem_pipe_read_q;

    bool is_first_compute = true;

#pragma unroll 1
    for (int tile_idx = 0; tile_idx < n_block_count; ++tile_idx) {
      int n_block = n_block_count - 1 - tile_idx;

      consumer_wait(pipeline_k, smem_pipe_read_k);

      Tensor tSrS_local =
          partition_fragment_C(tiled_mma_qk, make_shape(Int<kBlockMPerWG>{}, Int<kBlockN>{}));
      Tensor tSrS_local_cv = make_tensor(
          tSrS_local.data(), nvfp4_attention::convert_to_conversion_layout(tSrS_local.layout()));

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrK); ++k_block) {
        copy_k_block(k_block);
      }
      add_delta_s(tSrS_local);
      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;

      if constexpr (!Is_causal) {
        math_order.arrive();
      }

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
        cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                   make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS_local);
      }

      apply_mask(tSrS_local, n_block);
      if (is_first_compute) {
        softmax_fused.template online_softmax_with_quant<true, Is_causal>(
            tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
      } else {
        softmax_fused.template online_softmax_with_quant<false, Is_causal>(
            tSrS_local, AbsMaxP, mainloop_params.softmax_scale_log2);
      }

      auto quantize_score = [&](auto mma_k) { quantize(mma_k, tSrS_local_cv); };

      math_order.wait();
      consumer_wait(pipeline_v, smem_pipe_read_v);
      copy_v_block(_0{});
      quantize_score(_0{});

      if (is_first_compute) {
        CUTLASS_PRAGMA_UNROLL
        for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
          cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                     make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO_store);
          if (v_block < size<2>(tOrP) - 1) {
            copy_v_block(v_block + 1);
            quantize_score(v_block + 1);
          }
        }
        is_first_compute = false;
      } else {
        Tensor tOrO = make_fragment_like(tOrO_store);
        CUTLASS_PRAGMA_UNROLL
        for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
          cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                     make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);
          if (v_block < size<2>(tOrP) - 1) {
            copy_v_block(v_block + 1);
            quantize_score(v_block + 1);
          }
        }
        softmax_fused.rescale_o(tOrO_store, tOrO);
      }

      math_order.arrive();
      pipeline_v.consumer_release(smem_pipe_read_v);
      ++smem_pipe_read_v;

      tma_refill.refill_v(tile_idx);
    }

    softmax_fused.finalize(tOrO_store);
    return;
  }
};

}  // namespace nvfp4_attention
