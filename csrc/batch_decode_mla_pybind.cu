#include "mla_config.inc"
#include "pytorch_extension_utils.h"

std::vector<int64_t> BatchDecodeWithPagedKVCachePlanMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, unsigned int batch_size,
    unsigned int num_qo_heads, unsigned int page_size, bool enable_cuda_graph, int64_t cuda_stream);

void BatchDecodeWithPagedKVCacheRunMLA(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    std::vector<int64_t> plan_info_vec, at::Tensor q_nope, at::Tensor q_pe,
    at::Tensor paged_ckv_cache, at::Tensor paged_kpe_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, at::Tensor o, float sm_scale,
    int window_left, float logits_soft_cap, float rope_scale, float rope_theta,
    std::optional<at::Tensor> maybe_lse, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchDecodeWithPagedKVCachePlanMLA);
  m.def("run", &BatchDecodeWithPagedKVCacheRunMLA);
}
