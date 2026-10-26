#pragma once
template <class T, class Fragment, class Plan>
__forceinline__ __device__ void rescale_output_fragment(Fragment& output, Plan& plan, int lane) {
#pragma unroll
  for (int group = 0; group < T::BLOCK_M / 8; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2, n1 = n0 + 1, base = group * 4;
    output(base + 0) *= plan.row_scale[n0];
    output(base + 1) *= plan.row_scale[n1];
    output(base + 2) *= plan.row_scale[n0];
    output(base + 3) *= plan.row_scale[n1];
  }
}
template <class T, class Fragment, class VTile, class P>
__forceinline__ __device__ void issue_pv_gemm(Fragment& output, VTile const& v_tile, P const& sP,
                                              int idx) {
  typename T::TiledMMAPV mma;
  auto thread_mma = mma.get_slice(idx);
  Tensor a = thread_mma.partition_fragment_A(v_tile);
  Tensor b = thread_mma.partition_fragment_B(sP);
  issue_wgmma<false>(mma, a, b, output);
}
__forceinline__ __device__ float raw_exp_weight(float maximum, float sum, float common_max) {
  return sum > 0.0f ? exp2f(maximum - common_max) : 0.0f;
}
__forceinline__ __device__ float2 raw_pair_weights(float local_max, float local_sum, float peer_max,
                                                   float peer_sum) {
  const float maximum = fmaxf(local_max, peer_max);
  const float a = raw_exp_weight(local_max, local_sum, maximum);
  const float b = raw_exp_weight(peer_max, peer_sum, maximum);
  const float denominator = a * local_sum + b * peer_sum;
  const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  return make_float2(a * inv, b * inv);
}
