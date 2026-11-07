// Replica-plan launchers of the SM90 MXFP4 MegaMoE kernel: the in-kernel
// planner (plan and remap in one launch), the logical-to-physical remap of the
// top-k ids alone and the separate weight copy of the changed redundant expert
// slots. One module per (num_ranks, num_experts, num_redundant_slots).
#include <algorithm>

#include "megamoe_sm90_eplb_common.cuh"
#include "megamoe_sm90_eplb_config.inc"

namespace flashinfer::megamoe {
namespace cfg = config;

namespace {

constexpr eplb::Workspace kLayout(cfg::kNumRanks, cfg::kNumExperts, cfg::kNumRedundantSlots);

// One CTA per `kNumThreads` routing entries, at most one per SM.
dim3 remap_grid(int64_t num_tokens, int64_t num_topk, int64_t num_sms) {
  TVM_FFI_ICHECK_GE(num_sms, 1);
  const int64_t ctas = (num_tokens * num_topk + cfg::kNumThreads - 1) / cfg::kNumThreads;
  return dim3(static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(num_sms, ctas))));
}

}  // namespace

// Plan from the logical routing of every rank and remap it in place. Every
// rank must call this the same number of times; `open_gate_blocks` is the
// smallest allocation that justifies copying weights into an empty slot.
void megamoe_sm90_eplb_plan_and_remap(tvm::ffi::TensorView workspace,
                                      tvm::ffi::Array<int64_t> workspace_ptrs, int64_t rank_idx,
                                      tvm::ffi::TensorView topk_idx, int64_t num_tokens,
                                      int64_t block_m, int64_t open_gate_blocks, int64_t num_sms) {
  TVM_FFI_ICHECK_GE(block_m, 1);
  TVM_FFI_ICHECK_GE(open_gate_blocks, 0);
  (void)detail::eplb_workspace_ptr(workspace, kLayout);  // dtype, size and alignment
  auto sym_ws = detail::eplb_sym_buffer<cfg::kNumRanks>(workspace, workspace_ptrs, rank_idx);
  auto* idx = detail::eplb_topk_idx_ptr(topk_idx, num_tokens);
  uint32_t tokens = static_cast<uint32_t>(num_tokens),
           topk = static_cast<uint32_t>(topk_idx.size(1));
  uint32_t bm = static_cast<uint32_t>(block_m), gate = static_cast<uint32_t>(open_gate_blocks);
  void* args[] = {&sym_ws, &idx, &tokens, &topk, &bm, &gate};
  constexpr uint32_t kSmemBytes = eplb::plan_smem_bytes(
      cfg::kNumRanks, cfg::kNumExperts, cfg::kNumRedundantSlots, cfg::kNumThreads / 32);
  auto* kernel = reinterpret_cast<void*>(
      &eplb::eplb_plan_remap_kernel<cfg::kNumRanks, cfg::kNumExperts, cfg::kNumRedundantSlots,
                                    cfg::kNumThreads>);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      cudaSuccess)
      << "the planner's shared memory does not fit this geometry";
  // At most one CTA per SM, all of them resident at once, as the persistent
  // MegaMoE kernel assumes: the planning CTA waits for every CTA's arrival and
  // the others spin on `plan_ready`, so no CTA may wait for a free SM.
  TVM_FFI_ICHECK_EQ(
      cudaLaunchKernel(kernel, remap_grid(num_tokens, topk, num_sms), dim3(cfg::kNumThreads), args,
                       kSmemBytes, get_stream(topk_idx.device())),
      cudaSuccess);
}

void megamoe_sm90_eplb_remap(tvm::ffi::TensorView workspace, tvm::ffi::TensorView topk_idx,
                             int64_t num_tokens, int64_t block_m, int64_t num_sms) {
  TVM_FFI_ICHECK_GE(block_m, 1);
  auto* ws = detail::eplb_workspace_ptr(workspace, kLayout);
  auto* idx = detail::eplb_topk_idx_ptr(topk_idx, num_tokens);
  uint32_t tokens = static_cast<uint32_t>(num_tokens),
           topk = static_cast<uint32_t>(topk_idx.size(1));
  uint32_t bm = static_cast<uint32_t>(block_m);
  void* args[] = {&ws, &idx, &tokens, &topk, &bm};
  auto* kernel =
      reinterpret_cast<void*>(&eplb::eplb_remap_kernel<cfg::kNumRanks, cfg::kNumExperts,
                                                       cfg::kNumRedundantSlots, cfg::kNumThreads>);
  TVM_FFI_ICHECK_EQ(
      cudaLaunchKernel(kernel, remap_grid(num_tokens, topk, num_sms), dim3(cfg::kNumThreads), args,
                       0, get_stream(topk_idx.device())),
      cudaSuccess);
}

// `planes` lists (byte offset, bytes per slot) of every weight plane of the
// arena; each plane holds `num_experts / num_ranks + num_redundant_slots`
// slots. Copies the slots in the workspace's copy list from their resident
// ranks.
void megamoe_sm90_eplb_copy_weights(tvm::ffi::TensorView workspace, tvm::ffi::TensorView arena,
                                    tvm::ffi::Array<int64_t> arena_ptrs, int64_t rank_idx,
                                    tvm::ffi::Array<int64_t> planes) {
  constexpr uint32_t kNumResidentExperts = cfg::kNumExperts / cfg::kNumRanks;
  constexpr uint32_t kNumSlots = kNumResidentExperts + cfg::kNumRedundantSlots;
  constexpr uint32_t kNumBlocksPerSlot = 32;
  auto* ws = detail::eplb_workspace_ptr(workspace, kLayout);
  auto sym_arena = detail::eplb_sym_buffer<cfg::kNumRanks>(arena, arena_ptrs, rank_idx);
  eplb::CopyPlanes copy_planes{};
  detail::eplb_append_planes(copy_planes, planes, arena.numel() * get_element_size(arena),
                             kNumSlots);
  TVM_FFI_ICHECK_GE(copy_planes.num_planes, 1) << "no weight planes";
  const int2* copy_list = reinterpret_cast<const int2*>(ws + kLayout.copy_list_offset());
  const int32_t* copy_count = reinterpret_cast<const int32_t*>(ws + kLayout.copy_count_offset());
  uint32_t num_resident_experts = kNumResidentExperts;
  void* args[] = {&sym_arena, &copy_planes, &copy_list, &copy_count, &num_resident_experts};
  auto* kernel =
      reinterpret_cast<void*>(&eplb::eplb_copy_weights_kernel<cfg::kNumRanks, cfg::kNumThreads>);
  TVM_FFI_ICHECK_EQ(cudaLaunchKernel(kernel, dim3(cfg::kNumRedundantSlots, kNumBlocksPerSlot),
                                     dim3(cfg::kNumThreads), args, 0, get_stream(arena.device())),
                    cudaSuccess);
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_eplb_plan_and_remap,
                              flashinfer::megamoe::megamoe_sm90_eplb_plan_and_remap);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_eplb_remap,
                              flashinfer::megamoe::megamoe_sm90_eplb_remap);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_eplb_copy_weights,
                              flashinfer::megamoe::megamoe_sm90_eplb_copy_weights);
