#pragma once
template <bool ZeroInit, bool WaitForCompletion = true, class TiledMMA, class TensorA,
          class TensorB, class TensorC>
__forceinline__ __device__ void issue_wgmma(TiledMMA& tiled_mma, TensorA const& a, TensorB const& b,
                                            TensorC& c) {
  warpgroup_fence_operand(c);
  warpgroup_arrive();
  tiled_mma.accumulate_ = ZeroInit ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(a); ++k_block) {
    cute::gemm(tiled_mma, a(_, _, k_block), b(_, _, k_block), c);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  if constexpr (WaitForCompletion) {
    warpgroup_wait<0>();
    warpgroup_fence_operand(c);
  }
}
