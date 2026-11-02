// Query the common buffer layout without compiling a GEMM specialization.
#include <flashinfer/megamoe/eplb_layout.cuh>

#include "megamoe_sm90_common.cuh"

tvm::ffi::Array<int64_t> megamoe_sm90_symm_buffer_layout(int64_t ranks, int64_t experts,
                                                         int64_t capacity, int64_t topk,
                                                         int64_t hidden, int64_t intermediate,
                                                         int64_t pool, int64_t sf_pool) {
  return flashinfer::megamoe::detail::symm_buffer_layout(ranks, experts, capacity, topk, hidden,
                                                         intermediate, pool, sf_pool);
}

// Word offsets of the replica-plan workspace (`eplb::Workspace`), then the
// planner's shared-memory size in bytes:
// [num_words, slot_expert, copies, expert_base, local_counter, copy_list,
//  copy_count, copy_done, remap_done, flags, hist, local_hist, plan_ticket,
//  plan_ready, plan_generation, plan_smem_bytes]
tvm::ffi::Array<int64_t> megamoe_sm90_eplb_workspace_layout(int64_t num_ranks, int64_t num_experts,
                                                            int64_t num_redundant_slots) {
  TVM_FFI_ICHECK(num_ranks > 0 and num_experts > 0 and num_experts % num_ranks == 0)
      << "num_experts must divide across ranks";
  TVM_FFI_ICHECK(num_redundant_slots > 0 and
                 num_redundant_slots <= flashinfer::megamoe::eplb::Workspace::kMaxRedundantSlots)
      << "num_redundant_slots must be in [1, 32]";
  TVM_FFI_ICHECK_LE(num_ranks, flashinfer::megamoe::eplb::Workspace::kMaxRanks)
      << "the in-kernel planner supports at most 32 ranks";
  using flashinfer::megamoe::eplb::Workspace;
  const Workspace ws(num_ranks, num_experts, num_redundant_slots);
  const int64_t smem = flashinfer::megamoe::eplb::plan_smem_bytes(
      num_ranks, num_experts, num_redundant_slots, Workspace::kNumPlanThreads / 32);
  return tvm::ffi::Array<int64_t>(
      {ws.num_words(), ws.slot_expert_offset(), ws.copies_offset(), ws.expert_base_offset(),
       ws.local_counter_offset(), ws.copy_list_offset(), ws.copy_count_offset(),
       ws.copy_done_offset(), ws.remap_done_offset(), ws.flags_offset(), ws.hist_offset(),
       ws.local_hist_offset(), ws.plan_ticket_offset(), ws.plan_ready_offset(),
       ws.plan_generation_offset(), smem});
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_symm_buffer_layout, megamoe_sm90_symm_buffer_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_eplb_workspace_layout,
                              megamoe_sm90_eplb_workspace_layout);
