/*
 * Plan query for the split (BN128) MegaMoE family.
 *
 * Built once per shape and asked for the L1/L2 pipeline depth and shared-memory
 * budget; Python then renders those into the kernel module's config header.
 * Kept as a separate, template-free module so the (slow) kernel build only
 * happens once the plan is known.
 */
#include "megamoe_sm90_nvfp4_split_plan.cuh"

using tvm::ffi::Array;

namespace flashinfer::megamoe {

/*! \brief [l1_stages, l1_smem, l2_stages, l2_smem, experts_per_wave,
 *          num_max_pool_tokens, dispatch_dequant, l2_arrival_counter] */
Array<int64_t> megamoe_sm90_nvfp4_split_plan(int64_t num_sms, int64_t num_ranks,
                                             int64_t num_experts, int64_t num_max_tokens_per_rank,
                                             int64_t num_tokens, int64_t num_topk, int64_t hidden,
                                             int64_t intermediate_hidden,
                                             int64_t num_padded_sf_pool_tokens) {
  const SM90NVFP4MegaMoEInput input{
      static_cast<int>(num_sms),
      static_cast<int>(num_ranks),
      static_cast<int>(num_experts),
      static_cast<int>(num_experts / num_ranks),
      static_cast<int>(num_max_tokens_per_rank),
      static_cast<int>(num_tokens),
      static_cast<int>(num_topk),
      static_cast<int>(hidden),
      static_cast<int>(intermediate_hidden),
      static_cast<int>(num_padded_sf_pool_tokens),
  };
  const auto plan = select_sm90_nvfp4_split_mega_moe(input);
  return Array<int64_t>({
      plan.l1_config.num_stages,
      plan.l1_config.smem_size,
      plan.l2_config.num_stages,
      plan.l2_config.smem_size,
      plan.l1_config.num_experts_per_wave,
      plan.l1_config.num_max_pool_tokens,
      static_cast<int64_t>(plan.dispatch_dequant),
      static_cast<int64_t>(plan.l2_arrival_counter),
  });
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_nvfp4_split_plan,
                              flashinfer::megamoe::megamoe_sm90_nvfp4_split_plan);
