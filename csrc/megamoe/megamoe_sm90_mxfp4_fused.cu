// MXFP4 fused launch/binding. Weights use contiguous chunk-major decode tiles;
// activation descriptors, buffer ABI and communication primitives are shared.
#include "megamoe_sm90_common.cuh"
#include "megamoe_sm90_mxfp4_config.inc"

namespace flashinfer::megamoe {
namespace cfg = config;

void megamoe_sm90_mxfp4_fused(
    tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
    tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx, tvm::ffi::TensorView l1_weights,
    tvm::ffi::TensorView l2_weights, tvm::ffi::Array<int64_t> offsets, int64_t num_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> cumulative_local_expert_recv_stats,
    tvm::ffi::Optional<tvm::ffi::TensorView> l1_global_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> l2_global_scales) {
  TVM_FFI_ICHECK_EQ(symm_buffer_ptrs.size(), cfg::kNumRanks);
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, cfg::kNumRanks);
  TVM_FFI_ICHECK_GE(num_tokens, 1);
  TVM_FFI_ICHECK_LE(num_tokens, cfg::kNumMaxTokensPerRank);
  const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1}, f32{kDLFloat, 32, 1};
  using detail::make_tma_2d_desc;
  using detail::make_tma_sf_desc;
  using detail::offset_ptr;

  auto tm_l1 = make_tma_2d_desc(offset_ptr(symm_buffer, offsets[5]), fp8, cfg::kHidden,
                                cfg::kNumMaxPoolTokens, 128, cfg::kBlockM, cfg::kHidden, 128);
  auto tm_l1_sf = make_tma_sf_desc(offset_ptr(symm_buffer, offsets[6]), f32,
                                   cfg::kNumPaddedSFPoolTokens, cfg::kHidden, cfg::kBlockM, 128);
  auto tm_out = make_tma_2d_desc(offset_ptr(symm_buffer, offsets[7]), fp8, cfg::kIntermediateHidden,
                                 cfg::kNumMaxPoolTokens, cfg::kBlockN / 2, cfg::kBlockM,
                                 cfg::kIntermediateHidden, 0);
  auto tm_l2 =
      make_tma_2d_desc(offset_ptr(symm_buffer, offsets[7]), fp8, cfg::kIntermediateHidden,
                       cfg::kNumMaxPoolTokens, 128, cfg::kBlockM, cfg::kIntermediateHidden, 128);
  auto tm_l2_sf =
      make_tma_sf_desc(offset_ptr(symm_buffer, offsets[8]), f32, cfg::kNumPaddedSFPoolTokens,
                       cfg::kIntermediateHidden, cfg::kBlockM, cfg::kBlockN / 2);
  const std::vector<int64_t> ptrs(symm_buffer_ptrs.begin(), symm_buffer_ptrs.end());
  layout::SymBuffer<cfg::kNumRanks> sym_buffer(ptrs, static_cast<uint32_t>(rank_idx));
  void* output = y.data_ptr();
  auto* stats = detail::optional_tensor_ptr<int>(cumulative_local_expert_recv_stats);
  const auto* w1 = static_cast<const uint8_t*>(l1_weights.data_ptr());
  const auto* w2 = static_cast<const uint8_t*>(l2_weights.data_ptr());
  const float* gs1 = detail::optional_tensor_ptr<const float>(l1_global_scales);
  const float* gs2 = detail::optional_tensor_ptr<const float>(l2_global_scales);
  uint32_t tokens = static_cast<uint32_t>(num_tokens);
  unsigned long long* stamps = nullptr;
  void* args[] = {&output, &stats,    &tokens, &sym_buffer, &tm_l1, &tm_l1_sf, &tm_out,
                  &tm_l2,  &tm_l2_sf, &w1,     &w2,         &gs1,   &gs2,      &stamps};
  auto* kernel = reinterpret_cast<void*>(cfg::kKernel);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg::kSmemSize),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaLaunchKernel(kernel, dim3(cfg::kNumSMs), dim3(cfg::kNumThreads), args,
                                     cfg::kSmemSize, get_stream(y.device())),
                    cudaSuccess);
}
}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_mxfp4_fused,
                              flashinfer::megamoe::megamoe_sm90_mxfp4_fused);
