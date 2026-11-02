// Query the common buffer layout without compiling a GEMM specialization.
#include "megamoe_sm90_common.cuh"

tvm::ffi::Array<int64_t> megamoe_sm90_symm_buffer_layout(int64_t ranks, int64_t experts,
                                                         int64_t capacity, int64_t topk,
                                                         int64_t hidden, int64_t intermediate,
                                                         int64_t pool, int64_t sf_pool) {
  return flashinfer::megamoe::detail::symm_buffer_layout(ranks, experts, capacity, topk, hidden,
                                                         intermediate, pool, sf_pool);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm90_symm_buffer_layout, megamoe_sm90_symm_buffer_layout);
