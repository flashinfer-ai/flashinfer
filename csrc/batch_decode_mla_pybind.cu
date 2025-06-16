#include "mla_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchDecodeWithPagedKVCachePlanMLA(at::Tensor float_workspace_buffer,
                                              at::Tensor int_workspace_buffer,
                                              at::Tensor page_locked_int_workspace_buffer,
                                              at::Tensor indptr, int64_t batch_size,
                                              int64_t num_qo_heads, int64_t page_size,
                                              bool enable_cuda_graph);

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q_nope, at::Tensor q_pe, at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache,
    at::Tensor paged_kv_indptr, at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len,
    at::Tensor o, double sm_scale, int64_t window_left, double logits_soft_cap, double rope_scale,
    double rope_theta, std::optional<at::Tensor> maybe_lse, bool enable_pdl);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("plan", BatchDecodeWithPagedKVCachePlanMLA);
  m.def("run", BatchDecodeWithPagedKVCacheRunMLA);
}
