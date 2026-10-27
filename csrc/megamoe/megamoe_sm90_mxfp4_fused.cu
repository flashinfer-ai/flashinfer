// MXFP4 fused launch/binding. Weights use contiguous chunk-major decode tiles;
// activation descriptors, buffer ABI and communication primitives are shared.
// Built with `kInKernelWeightCopy`, the kernel also copies the changed redundant
// expert slots of a replica plan from their resident ranks' weight arenas.
#include "megamoe_sm90_common.cuh"
#include "megamoe_sm90_eplb_common.cuh"
#include "megamoe_sm90_mxfp4_config.inc"

namespace flashinfer::megamoe {
namespace cfg = config;

namespace {

void launch(tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
            tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx,
            tvm::ffi::TensorView l1_weights, tvm::ffi::TensorView l2_weights,
            tvm::ffi::Array<int64_t> offsets, int64_t num_tokens,
            tvm::ffi::Optional<tvm::ffi::TensorView> cumulative_local_expert_recv_stats,
            tvm::ffi::Optional<tvm::ffi::TensorView> l1_global_scales,
            tvm::ffi::Optional<tvm::ffi::TensorView> l2_global_scales,
            const eplb::InKernelCopy<cfg::kNumRanks>& in_kernel_copy) {
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
  void* args[] = {
      &output,     &stats,  &tokens,
      &sym_buffer, &tm_l1,  &tm_l1_sf,
      &tm_out,     &tm_l2,  &tm_l2_sf,
      &w1,         &w2,     &gs1,
      &gs2,        &stamps, const_cast<eplb::InKernelCopy<cfg::kNumRanks>*>(&in_kernel_copy)};
  auto* kernel = reinterpret_cast<void*>(cfg::kKernel);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg::kSmemSize),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaLaunchKernel(kernel, dim3(cfg::kNumSMs), dim3(cfg::kNumThreads), args,
                                     cfg::kSmemSize, get_stream(y.device())),
                    cudaSuccess);
}

}  // namespace

void megamoe_sm90_mxfp4_fused(
    tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
    tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx, tvm::ffi::TensorView l1_weights,
    tvm::ffi::TensorView l2_weights, tvm::ffi::Array<int64_t> offsets, int64_t num_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> cumulative_local_expert_recv_stats,
    tvm::ffi::Optional<tvm::ffi::TensorView> l1_global_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> l2_global_scales) {
  TVM_FFI_ICHECK(!cfg::kInKernelWeightCopy)
      << "this module carries the in-kernel weight copy; call "
         "megamoe_sm90_mxfp4_fused_in_kernel_copy";
  launch(y, symm_buffer, symm_buffer_ptrs, rank_idx, l1_weights, l2_weights, offsets, num_tokens,
         cumulative_local_expert_recv_stats, l1_global_scales, l2_global_scales,
         eplb::InKernelCopy<cfg::kNumRanks>{});
}

// Same launch with the in-kernel weight copy: the redundant expert slots listed
// in `eplb_workspace` are copied from the resident ranks' arenas
// (`arena_ptrs`) by the kernel itself. The weights and global scales are views
// of this rank's arena; `l1_planes` / `l2_planes` are (byte offset, bytes per
// slot) pairs of the arena planes each GEMM reads.
void megamoe_sm90_mxfp4_fused_in_kernel_copy(
    tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
    tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx, tvm::ffi::TensorView l1_weights,
    tvm::ffi::TensorView l2_weights, tvm::ffi::Array<int64_t> offsets, int64_t num_tokens,
    tvm::ffi::Optional<tvm::ffi::TensorView> cumulative_local_expert_recv_stats,
    tvm::ffi::Optional<tvm::ffi::TensorView> l1_global_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> l2_global_scales, tvm::ffi::TensorView eplb_workspace,
    tvm::ffi::TensorView arena, tvm::ffi::Array<int64_t> arena_ptrs, int64_t num_resident_experts,
    tvm::ffi::Array<int64_t> l1_planes, tvm::ffi::Array<int64_t> l2_planes) {
  TVM_FFI_ICHECK(cfg::kInKernelWeightCopy)
      << "this module has no in-kernel weight copy; call megamoe_sm90_mxfp4_fused";
  constexpr int64_t kNumExpertsPerRank = cfg::kNumExperts / cfg::kNumRanks;
  // The kernel's experts per rank are the resident experts plus the redundant slots.
  const int64_t num_redundant_slots = kNumExpertsPerRank - num_resident_experts;
  TVM_FFI_ICHECK(num_resident_experts > 0 and num_redundant_slots > 0 and
                 num_redundant_slots <= eplb::Workspace::kMaxRedundantSlots)
      << "resident experts must leave 1 to " << eplb::Workspace::kMaxRedundantSlots
      << " redundant slots of the kernel's " << kNumExpertsPerRank << " experts per rank";
  const eplb::Workspace layout(cfg::kNumRanks, num_resident_experts * cfg::kNumRanks,
                               num_redundant_slots);
  auto* ws = detail::eplb_workspace_ptr(eplb_workspace, layout);
  eplb::InKernelCopy<cfg::kNumRanks> in_kernel_copy{};
  in_kernel_copy.arena = detail::eplb_sym_buffer<cfg::kNumRanks>(arena, arena_ptrs, rank_idx);
  const int64_t arena_bytes = arena.numel() * get_element_size(arena);
  detail::eplb_append_planes(in_kernel_copy.planes, l1_planes, arena_bytes, kNumExpertsPerRank);
  in_kernel_copy.num_l1_planes = in_kernel_copy.planes.num_planes;
  detail::eplb_append_planes(in_kernel_copy.planes, l2_planes, arena_bytes, kNumExpertsPerRank);
  TVM_FFI_ICHECK(in_kernel_copy.num_l1_planes > 0 and
                 in_kernel_copy.planes.num_planes > in_kernel_copy.num_l1_planes)
      << "each GEMM needs at least one weight plane";
  in_kernel_copy.copy_list = reinterpret_cast<const int2*>(ws + layout.copy_list_offset());
  in_kernel_copy.copy_count = reinterpret_cast<int32_t*>(ws + layout.copy_count_offset());
  in_kernel_copy.copy_done = ws + layout.copy_done_offset();
  in_kernel_copy.num_resident_experts = static_cast<uint32_t>(num_resident_experts);
  in_kernel_copy.num_redundant_slots = static_cast<uint32_t>(num_redundant_slots);
  launch(y, symm_buffer, symm_buffer_ptrs, rank_idx, l1_weights, l2_weights, offsets, num_tokens,
         cumulative_local_expert_recv_stats, l1_global_scales, l2_global_scales, in_kernel_copy);
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_mxfp4_fused,
                              flashinfer::megamoe::megamoe_sm90_mxfp4_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_mxfp4_fused_in_kernel_copy,
                              flashinfer::megamoe::megamoe_sm90_mxfp4_fused_in_kernel_copy);
