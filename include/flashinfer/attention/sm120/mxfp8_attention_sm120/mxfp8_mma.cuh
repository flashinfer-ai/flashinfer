/*
 * SM120a MXFP8 block-scaled MMA helpers for FlashInfer.
 *
 * The warp-level block-scaled atom
 *   cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<e4m3, e4m3, float, ue8m0, 32>
 * feeds its scale factors through a "zipped" operand tensor (data + SF). The
 * thread-value partitioning of the SF operands is NOT part of TiledMMA/ThrMMA;
 * upstream CUTLASS only implements it inside the GEMM collective
 * (cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp). A fused attention
 * kernel does not use that collective, so we lift the four partitioning helpers
 * here as free functions, verbatim in behavior, to reuse in our own mainloop.
 *
 * Reference: cutlass @ b46b16d, sm120_blockscaled_mma_tma.hpp:430-523.
 */
#pragma once

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/tensor.hpp>

namespace flashinfer {
namespace sm120_mxfp8 {

using namespace cute;

// (PermM,PermK) SF tensor -> ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
// Mirrors CollectiveMma::thrfrg_SFA.
template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);  // (PermM,PermK)

  auto a_tile =
      make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK))

  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);  // ((ThrV,FrgV),(RestM,RestK))

  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor =
      zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
  return thr_tensor;
}

// (PermN,PermK) SF tensor -> ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
// Mirrors CollectiveMma::thrfrg_SFB.
template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);  // (PermN,PermK)

  auto a_tile =
      make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomN,AtomK),(RestN,RestK))

  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);  // ((ThrV,FrgV),(RestN,RestK))

  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor =
      zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

// Slice an SFA tensor (smem/gmem) to this thread's source view, in the
// thread-value layout the block-scaled atom expects. The returned tensor still
// points at the input memory -- copy() it into partition_fragment_SFA(...).
// Mirrors the body of CollectiveMma::partition_fragment_SFA up to the fragment
// allocation.
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

// Allocate the per-thread SFA register fragment. Mirrors
// CollectiveMma::partition_fragment_SFA.
template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(
      partition_SFA(static_cast<SFATensor&&>(sfatensor), thread_mma));
}

// Allocate the per-thread SFB register fragment. Mirrors
// CollectiveMma::partition_fragment_SFB.
template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(
      partition_SFB(static_cast<SFBTensor&&>(sfbtensor), thread_mma));
}

// Thread-value layout of the SFA operand for a TiledMMA, used to build the
// smem->reg tiled copy (make_tiled_copy_impl) that stages SF into the per-thread
// fragment. Mirrors CollectiveMma::get_layoutSFA_TV (cutlass) / SageAttention's.
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

}  // namespace sm120_mxfp8
}  // namespace flashinfer
