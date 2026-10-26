// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel.cuh"

namespace flashinfer::mla_swa_sm90 {
// No framework allocation, stream lookup, or host reads of metadata values.
struct Arguments {
  const cutlass::bfloat16_t* q;
  const cutlass::bfloat16_t* kv;
  const int* table;
  const int* lengths;
  cutlass::bfloat16_t* out;
  void* workspace;
  int batch, sq, heads, pages, table_cols, window;
  float scale;
  int64_t q_stride_b, table_stride_b;
  cudaStream_t stream;
};

inline int64_t align256(int64_t n) { return (n + 255) / 256 * 256; }
inline int64_t workspace_bytes(int64_t rows) {
  return align256(rows * 9 * 1024 * sizeof(float)) + align256(rows * 9 * sizeof(float));
}

template <class Config>
cudaError_t launch(const Arguments& a) {
  using namespace cute;
  using namespace sm90::dense_swa;
  using namespace sm90::dense_swa::roles;
  using T = Traits<cutlass::bfloat16_t, Config>;
  using InputT = typename T::InputT;
  const auto shape_q = make_shape(a.sq * a.heads, Int<T::HEAD_DIM_K>{}, a.batch);
  const auto tma_q = make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(const_cast<InputT*>(a.q)),
                  make_layout(shape_q, make_stride(int64_t{1088}, _1{}, a.q_stride_b))),
      typename T::SmemLayoutQ{});
  const auto shape_k = make_shape(Int<T::BLOCK_N>{}, Int<T::HEAD_DIM_K>{}, a.pages);
  const auto tma_k = make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(const_cast<InputT*>(a.kv)),
                  make_layout(shape_k, make_stride(int64_t{1088}, _1{}, int64_t{64 * 1088}))),
      tile_to_shape(
          GMMA::Layout_K_SW128_Atom<InputT>{},
          Layout<Shape<Int<T::BLOCK_N>, Int<T::TILE_K>>, Stride<Int<T::HEAD_DIM_K>, _1>>{}));
  TmaParams<decltype(shape_q), decltype(tma_q), decltype(shape_k), decltype(tma_k)> params{
      shape_q, tma_q, shape_k, tma_k};
  auto kernel = tactic_kernel_pointer<T, decltype(params)>();
  auto status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SHARED_BYTES);
  if (status != cudaSuccess) return status;
  const int64_t rows = int64_t{a.batch} * a.sq * a.heads;
  const int64_t out_b = int64_t{a.sq} * a.heads * 1024;
  const int64_t out_s = int64_t{a.heads} * 1024;
  const int64_t q_s = int64_t{a.heads} * 1088;
  const int tiles = cute::ceil_div(a.sq * a.heads, T::BLOCK_M);
  if constexpr (Config::Backend == mla_swa::Reduce::DSM) {
    auto cluster_status = cutlass::launch_kernel_on_cluster(
        {dim3(a.batch * Config::S, tiles, 1), dim3(T::NUM_THREADS), dim3(Config::S, 1, 1),
         T::SHARED_BYTES, a.stream},
        reinterpret_cast<void const*>(kernel), a.q, a.kv, a.table, a.lengths, a.out,
        static_cast<float*>(nullptr), Config::S, int64_t{0}, a.batch, a.sq, a.heads, a.table_cols,
        a.window, a.scale, a.q_stride_b, q_s, int64_t{1088}, int64_t{1088}, a.table_stride_b, out_b,
        out_s, int64_t{1024}, params);
    if (cluster_status != cutlass::Status::kSuccess) return cudaErrorInvalidConfiguration;
  } else if constexpr (Config::Backend == mla_swa::Reduce::GlobalFP32) {
    auto partial = static_cast<float*>(a.workspace);
    auto lse = reinterpret_cast<float*>(static_cast<char*>(a.workspace) +
                                        align256(rows * Config::S * 1024 * sizeof(float)));
    // Device entry retains its historical InputT* ABI. The split strides are
    // expressed in BF16 elements (x2), while the epilogue stores FP32.
    kernel<<<dim3(a.batch, tiles, Config::S), T::NUM_THREADS, T::SHARED_BYTES, a.stream>>>(
        a.q, a.kv, a.table, a.lengths, reinterpret_cast<InputT*>(partial), lse, Config::S,
        int64_t{1024 * 2}, a.batch, a.sq, a.heads, a.table_cols, a.window, a.scale, a.q_stride_b,
        q_s, int64_t{1088}, int64_t{1088}, a.table_stride_b, out_b * Config::S, out_s * Config::S,
        int64_t{Config::S * 1024}, params);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    dense_swa_roles_split_combine_kernel<InputT, 1024, Config::S>
        <<<rows, 256, 0, a.stream>>>(partial, lse, a.out, rows, Config::S);
  } else {
    kernel<<<dim3(a.batch, tiles), T::NUM_THREADS, T::SHARED_BYTES, a.stream>>>(
        a.q, a.kv, a.table, a.lengths, a.out, a.batch, a.sq, a.heads, a.table_cols, a.window,
        a.scale, a.q_stride_b, q_s, int64_t{1088}, int64_t{1088}, a.table_stride_b, out_b, out_s,
        int64_t{1024}, params);
  }
  return cudaGetLastError();
}
}  // namespace flashinfer::mla_swa_sm90
