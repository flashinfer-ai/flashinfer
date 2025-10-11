#include "mla_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

Array<int64_t> BatchDecodeWithPagedKVCachePlanMLA(TensorView float_workspace_buffer,
                                                  TensorView int_workspace_buffer,
                                                  TensorView page_locked_int_workspace_buffer,
                                                  TensorView indptr, int64_t batch_size,
                                                  int64_t num_qo_heads, int64_t page_size,
                                                  bool enable_cuda_graph);

void BatchDecodeWithPagedKVCacheRunMLA(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    Array<int64_t> plan_info_vec, TensorView q_nope, TensorView q_pe, TensorView paged_ckv_cache,
    TensorView paged_kpe_cache, TensorView paged_kv_indptr, TensorView paged_kv_indices,
    TensorView paged_kv_last_page_len, TensorView o, double sm_scale, int64_t window_left,
    double logits_soft_cap, double rope_scale, double rope_theta, Optional<TensorView> maybe_lse,
    bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchDecodeWithPagedKVCachePlanMLA);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, BatchDecodeWithPagedKVCacheRunMLA);
