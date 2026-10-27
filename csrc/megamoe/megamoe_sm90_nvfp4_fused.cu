/*
 * SM90 NVFP4 MegaMoE -- host launcher.
 *
 * One persistent kernel fuses the cross-rank token dispatch, both FFN GEMMs
 * (NVFP4 weights x FP8 activations), SwiGLU and the combine reduction over an
 * rank-parameterized NVLink symmetric buffer. Everything that shapes shared memory or the
 * task schedule is a template parameter, so this file is compiled once per
 * (shape, plan) pair against the generated config header.
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "megamoe_sm90_common.cuh"
#include "megamoe_sm90_nvfp4_config.inc"

namespace flashinfer::megamoe {
namespace {

namespace cfg = ::flashinfer::megamoe::config;

using detail::make_tma_2d_desc;
using detail::make_tma_sf_desc;

}  // namespace

// Launch one MegaMoE call. Pointers are raw device addresses; `sym_buffer_ptrs`
// holds every rank's base address of the symmetric buffer (NVLink peer mapped).
void sm90_nvfp4_mega_moe_fused_launch(void* y, int* cumulative_local_expert_recv_stats,
                                      int num_tokens, const std::vector<int64_t>& sym_buffer_ptrs,
                                      int rank_idx, void* l1_acts, void* l1_acts_sf, void* l2_acts,
                                      void* l2_acts_sf, void* l1_weights, void* l2_weights,
                                      int64_t l1_acts_stride, int64_t l2_acts_stride,
                                      int64_t l1_weights_inner, int64_t l1_weights_stride,
                                      int64_t l2_weights_inner, int64_t l2_weights_stride,
                                      const float* l1_global_scales, const float* l2_global_scales,
                                      cudaStream_t stream) {
  constexpr int kBlockK = 128;
  constexpr int kSwizzleActs = 128;
  constexpr int kBStoragePerKBlock = 80;
  constexpr int kL1ScaleGranK = 128;
  constexpr int kNumExpertsPerRank = cfg::kNumExperts / cfg::kNumRanks;

  const DLDataType fp8{kDLFloat8_e4m3fn, 8, 1};
  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType u8{kDLUInt, 8, 1};

  const auto tm_l1_acts =
      make_tma_2d_desc(l1_acts, fp8, cfg::kHidden, cfg::kNumMaxPoolTokens, kBlockK, cfg::kBlockM,
                       static_cast<int>(l1_acts_stride), kSwizzleActs);
  const auto tm_l1_acts_sf = make_tma_sf_desc(l1_acts_sf, f32, cfg::kNumPaddedSFPoolTokens,
                                              cfg::kHidden, cfg::kBlockM, kL1ScaleGranK);
  const auto tm_l1_weights =
      make_tma_2d_desc(l1_weights, u8, static_cast<int>(l1_weights_inner),
                       kNumExpertsPerRank * cfg::kIntermediateHidden * 2, kBStoragePerKBlock,
                       cfg::kBlockN, static_cast<int>(l1_weights_stride), 0);
  // L1 writes its SwiGLU output as half-width blocks into the L2 activation pool.
  const auto tm_l1_output =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens,
                       cfg::kBlockN / 2, cfg::kBlockM, static_cast<int>(l2_acts_stride), 0);
  const auto tm_l2_acts =
      make_tma_2d_desc(l2_acts, fp8, cfg::kIntermediateHidden, cfg::kNumMaxPoolTokens, kBlockK,
                       cfg::kBlockM, static_cast<int>(l2_acts_stride), kSwizzleActs);
  const auto tm_l2_acts_sf =
      make_tma_sf_desc(l2_acts_sf, f32, cfg::kNumPaddedSFPoolTokens, cfg::kIntermediateHidden,
                       cfg::kBlockM, cfg::kBlockN / 2);
  const auto tm_l2_weights = make_tma_2d_desc(l2_weights, u8, static_cast<int>(l2_weights_inner),
                                              kNumExpertsPerRank * cfg::kHidden, kBStoragePerKBlock,
                                              cfg::kBlockN, static_cast<int>(l2_weights_stride), 0);

  const layout::SymBuffer<cfg::kNumRanks> sym_buffer(sym_buffer_ptrs,
                                                     static_cast<uint32_t>(rank_idx));

  auto* kernel = reinterpret_cast<void*>(cfg::kKernel);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg::kSmemSize),
      cudaSuccess)
      << "failed to raise the dynamic shared-memory limit to " << cfg::kSmemSize;

  const uint32_t tokens = static_cast<uint32_t>(num_tokens);
  unsigned long long* phase_stamps = nullptr;
  void* args[] = {&y,
                  &cumulative_local_expert_recv_stats,
                  &phase_stamps,
                  const_cast<uint32_t*>(&tokens),
                  const_cast<layout::SymBuffer<cfg::kNumRanks>*>(&sym_buffer),
                  const_cast<CUtensorMap*>(&tm_l1_acts),
                  const_cast<CUtensorMap*>(&tm_l1_acts_sf),
                  const_cast<CUtensorMap*>(&tm_l1_weights),
                  const_cast<CUtensorMap*>(&tm_l1_output),
                  const_cast<CUtensorMap*>(&tm_l2_acts),
                  const_cast<CUtensorMap*>(&tm_l2_acts_sf),
                  const_cast<CUtensorMap*>(&tm_l2_weights),
                  const_cast<const float**>(&l1_global_scales),
                  const_cast<const float**>(&l2_global_scales)};

  // Persistent kernel: exactly one CTA per SM.
  TVM_FFI_ICHECK_EQ(
      cudaLaunchKernel(kernel, dim3(MEGAMOE_NUM_SMS), dim3(384), args, cfg::kSmemSize, stream),
      cudaSuccess)
      << "MegaMoE kernel launch failed";
}

}  // namespace flashinfer::megamoe

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace flashinfer::megamoe {

using detail::offset_ptr;

/*!
 * \brief Byte offsets of every region inside the symmetric buffer.
 *
 * Returns, in order:
 *   [0] total bytes required
 *   [1] x            (fp8, [num_max_tokens_per_rank, hidden])
 *   [2] x_sf         (f32, [num_max_tokens_per_rank, hidden/128])
 *   [3] topk_idx     (i64, [num_max_tokens_per_rank, num_topk])
 *   [4] topk_weights (f32, [num_max_tokens_per_rank, num_topk])
 *   [5] l1_acts      (fp8, [num_max_pool_tokens, hidden])
 *   [6] l1_acts_sf   (f32, [num_padded_sf_pool_tokens, hidden/128], MN-major)
 *   [7] l2_acts      (fp8, [num_max_pool_tokens, intermediate_hidden])
 *   [8] l2_acts_sf   (f32, [num_padded_sf_pool_tokens, intermediate_hidden/64], MN-major)
 *   [9] num_max_pool_tokens
 *  [10] num_padded_sf_pool_tokens
 */
Array<int64_t> megamoe_sm90_nvfp4_symm_buffer_layout() {
  return detail::symm_buffer_layout(cfg::kNumRanks, cfg::kNumExperts, cfg::kNumMaxTokensPerRank,
                                    cfg::kNumTopk, cfg::kHidden, cfg::kIntermediateHidden,
                                    cfg::kNumMaxPoolTokens, cfg::kNumPaddedSFPoolTokens);
}

/*! \brief Plan constants this module was compiled for (for assertions in Python). */
Array<int64_t> megamoe_sm90_nvfp4_plan() {
  return Array<int64_t>(
      {static_cast<int64_t>(cfg::kBlockM), static_cast<int64_t>(cfg::kBlockN),
       static_cast<int64_t>(cfg::kNumStages), static_cast<int64_t>(cfg::kNumExpertsPerWave),
       static_cast<int64_t>(cfg::kPushDispatch), static_cast<int64_t>(cfg::kNoCleanBarrier),
       static_cast<int64_t>(cfg::kUseInterleavedScheduler), static_cast<int64_t>(cfg::kRSSwapAB)});
}

void megamoe_sm90_nvfp4_fused(TensorView y, TensorView symm_buffer, Array<int64_t> symm_buffer_ptrs,
                              int64_t rank_idx, TensorView l1_weights, TensorView l2_weights,
                              Array<int64_t> offsets, int64_t num_tokens,
                              Optional<TensorView> cumulative_local_expert_recv_stats,
                              Optional<TensorView> l1_global_scales,
                              Optional<TensorView> l2_global_scales) {
  TVM_FFI_ICHECK_EQ(symm_buffer_ptrs.size(), cfg::kNumRanks);
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, cfg::kNumRanks);
  TVM_FFI_ICHECK_GE(num_tokens, 1);
  TVM_FFI_ICHECK_LE(num_tokens, cfg::kNumMaxTokensPerRank);

  const std::vector<int64_t> ptrs(symm_buffer_ptrs.begin(), symm_buffer_ptrs.end());

  int* stats = detail::optional_tensor_ptr<int>(cumulative_local_expert_recv_stats);
  const float* gs1 = detail::optional_tensor_ptr<const float>(l1_global_scales);
  const float* gs2 = detail::optional_tensor_ptr<const float>(l2_global_scales);

  sm90_nvfp4_mega_moe_fused_launch(
      y.data_ptr(), stats, static_cast<int>(num_tokens), ptrs, static_cast<int>(rank_idx),
      offset_ptr(symm_buffer, offsets[5]), offset_ptr(symm_buffer, offsets[6]),
      offset_ptr(symm_buffer, offsets[7]), offset_ptr(symm_buffer, offsets[8]),
      l1_weights.data_ptr(), l2_weights.data_ptr(),
      /* l1_acts_stride   */ cfg::kHidden,
      /* l2_acts_stride   */ cfg::kIntermediateHidden,
      /* l1_weights_inner */ l1_weights.size(2), l1_weights.stride(1),
      /* l2_weights_inner */ l2_weights.size(2), l2_weights.stride(1), gs1, gs2,
      get_stream(y.device()));
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_fused,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_symm_buffer_layout,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_symm_buffer_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_plan,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_plan);
