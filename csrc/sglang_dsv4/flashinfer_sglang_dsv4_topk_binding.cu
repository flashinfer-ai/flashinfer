/*
 * TVM-FFI binding for the vendored sglang DeepSeek-V4 ragged top-k.
 */
#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;

void sglang_dsv4_topk_ragged(TensorView scores, TensorView lengths, TensorView out_offsets,
                             TensorView out_indices, bool enable_pdl);
void sglang_dsv4_topk_varlen(TensorView scores, TensorView seq_lens, TensorView out_indices,
                             int64_t next_n, int64_t compress_ratio, TensorView page_table,
                             int64_t page_size, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sglang_dsv4_topk_ragged, sglang_dsv4_topk_ragged);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sglang_dsv4_topk_varlen, sglang_dsv4_topk_varlen);
