/*
 * SM90 NVFP4 MegaMoE -- split (BN128) launcher.
 *
 * Two kernels per call: L1 dequantises the dispatched rows and runs the first
 * GEMM into the intermediate pool; L2 runs the second GEMM and scatters the
 * result back. L1 runs with a 2-CTA cluster, so it needs the extended launch.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "megamoe_sm90_common.cuh"
#include "megamoe_sm90_nvfp4_split_config.inc"

namespace flashinfer::megamoe {
namespace {

namespace cfg = ::flashinfer::megamoe::config;

using detail::make_tma_2d_desc;
using detail::make_tma_sf_desc;

void launch_clustered(void* kernel, int threads, int smem, int cluster, void** args,
                      cudaStream_t stream) {
  TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                    cudaSuccess)
      << "failed to raise dynamic shared memory to " << smem;
  cudaLaunchConfig_t lc{};
  lc.gridDim = dim3(cfg::kNumSMs);
  lc.blockDim = dim3(threads);
  lc.dynamicSmemBytes = smem;
  lc.stream = stream;
  cudaLaunchAttribute attrs[1];
  int n = 0;
  if (cluster > 1) {
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
    n = 1;
  }
  lc.attrs = attrs;
  lc.numAttrs = n;
  TVM_FFI_ICHECK_EQ(cudaLaunchKernelExC(&lc, kernel, args), cudaSuccess)
      << "MegaMoE split kernel launch failed";
}

}  // namespace

void sm90_nvfp4_mega_moe_split_launch(void* y, int* stats, int num_tokens,
                                      const std::vector<int64_t>& sym_buffer_ptrs, int rank_idx,
                                      void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* gs1, const float* gs2, cudaStream_t stream) {
  constexpr int kEPR = cfg::kNumExperts / cfg::kNumRanks;
  const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1};
  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType u8{kDLUInt, 8, 1};

  const auto tm_l1_acts =
      make_tma_2d_desc(l1_acts, fp8, cfg::kHidden, cfg::kNumMaxPoolTokens, cfg::kBlockK,
                       cfg::kBlockM, cfg::kHidden, cfg::kSwizzleActsMode);
  const auto tm_l1_acts_sf = make_tma_sf_desc(l1_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kHidden, cfg::kBlockM, 128);
  const auto tm_l1_weights = make_tma_2d_desc(
      l1_weights, u8, static_cast<int>(l1_weights_inner), kEPR * cfg::kIntermediateHidden * 2,
      cfg::kWeightStoragePerKBlock, cfg::kBlockN, static_cast<int>(l1_weights_stride), 0);
  // The two M warpgroups each store a 64x64 post-SwiGLU tile, so the L1 output
  // box is 64x64 -- not (BLOCK_N/2, BLOCK_M).
  constexpr int kL1OutputStoreBlockM = 64, kL1OutputStoreBlockN = 64;
  const auto tm_l1_output =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens,
                       kL1OutputStoreBlockN, kL1OutputStoreBlockM, cfg::kIntermediateHidden, 0);
  const auto tm_l2_acts =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens, cfg::kBlockK,
                       cfg::kBlockM, cfg::kIntermediateHidden, cfg::kSwizzleActsMode);
  const auto tm_l2_acts_sf = make_tma_sf_desc(l2_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kIntermediateHidden, cfg::kBlockM, 128);
  const auto tm_l2_weights = make_tma_2d_desc(l2_weights, u8, static_cast<int>(l2_weights_inner),
                                              kEPR * cfg::kHidden, cfg::kWeightStoragePerKBlock,
                                              cfg::kBlockN, static_cast<int>(l2_weights_stride), 0);

  const layout::SymBuffer<cfg::kNumRanks> sym_buffer(sym_buffer_ptrs,
                                                     static_cast<uint32_t>(rank_idx));
  const uint32_t tokens = static_cast<uint32_t>(num_tokens);

  void* l1_args[] = {&stats,
                     const_cast<uint32_t*>(&tokens),
                     const_cast<layout::SymBuffer<cfg::kNumRanks>*>(&sym_buffer),
                     const_cast<CUtensorMap*>(&tm_l1_acts),
                     const_cast<CUtensorMap*>(&tm_l1_acts_sf),
                     const_cast<CUtensorMap*>(&tm_l1_weights),
                     const_cast<CUtensorMap*>(&tm_l1_output),
                     const_cast<const float**>(&gs1)};
  launch_clustered(reinterpret_cast<void*>(cfg::kL1Kernel), cfg::kL1NumThreads, cfg::kL1SmemSize,
                   cfg::kL1ClusterSize, l1_args, stream);

  void* l2_args[] = {&y,
                     &stats,
                     const_cast<uint32_t*>(&tokens),
                     const_cast<layout::SymBuffer<cfg::kNumRanks>*>(&sym_buffer),
                     const_cast<CUtensorMap*>(&tm_l2_acts),
                     const_cast<CUtensorMap*>(&tm_l2_acts_sf),
                     const_cast<CUtensorMap*>(&tm_l2_weights),
                     const_cast<const float**>(&gs2)};
  launch_clustered(reinterpret_cast<void*>(cfg::kL2Kernel), cfg::kL2NumThreads, cfg::kL2SmemSize,
                   cfg::kL2ClusterSize, l2_args, stream);
}

}  // namespace flashinfer::megamoe

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace flashinfer::megamoe {

using detail::offset_ptr;

void megamoe_sm90_nvfp4_split(TensorView y, TensorView symm_buffer, Array<int64_t> symm_buffer_ptrs,
                              int64_t rank_idx, TensorView l1_weights, TensorView l2_weights,
                              Array<int64_t> offsets, int64_t num_tokens,
                              Optional<TensorView> cumulative_local_expert_recv_stats,
                              Optional<TensorView> l1_global_scales,
                              Optional<TensorView> l2_global_scales) {
  namespace cfg = ::flashinfer::megamoe::config;
  TVM_FFI_ICHECK_EQ(symm_buffer_ptrs.size(), cfg::kNumRanks);
  TVM_FFI_ICHECK_GE(num_tokens, 1);
  TVM_FFI_ICHECK_LE(num_tokens, cfg::kNumMaxTokensPerRank);

  const std::vector<int64_t> ptrs(symm_buffer_ptrs.begin(), symm_buffer_ptrs.end());
  int* stats = detail::optional_tensor_ptr<int>(cumulative_local_expert_recv_stats);
  const float* gs1 = detail::optional_tensor_ptr<const float>(l1_global_scales);
  const float* gs2 = detail::optional_tensor_ptr<const float>(l2_global_scales);

  sm90_nvfp4_mega_moe_split_launch(
      y.data_ptr(), stats, static_cast<int>(num_tokens), ptrs, static_cast<int>(rank_idx),
      offset_ptr(symm_buffer, offsets[5]), offset_ptr(symm_buffer, offsets[6]),
      offset_ptr(symm_buffer, offsets[7]), offset_ptr(symm_buffer, offsets[8]),
      l1_weights.data_ptr(), l2_weights.data_ptr(), l1_weights.size(2), l1_weights.stride(1),
      l2_weights.size(2), l2_weights.stride(1), gs1, gs2, get_stream(y.device()));
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_split,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_split);
