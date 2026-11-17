#include <tvm/ffi/tvm_ffi.h>

void C12Run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache, tvm::ffi::TensorView v_cache,
            tvm::ffi::TensorView topk, tvm::ffi::TensorView page_table,
            tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView out);
void F16Run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache, tvm::ffi::TensorView v_cache,
            tvm::ffi::TensorView topk, tvm::ffi::TensorView page_table,
            tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView out);
void DirectKF16Run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache,
                   tvm::ffi::TensorView v_cache, tvm::ffi::TensorView topk,
                   tvm::ffi::TensorView page_table, tvm::ffi::TensorView seq_lens,
                   tvm::ffi::TensorView out);
void DirectKShortB8Run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache,
                       tvm::ffi::TensorView v_cache, tvm::ffi::TensorView topk,
                       tvm::ffi::TensorView page_table, tvm::ffi::TensorView seq_lens,
                       tvm::ffi::TensorView out);
void DirectKShortRun(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache,
                     tvm::ffi::TensorView v_cache, tvm::ffi::TensorView topk,
                     tvm::ffi::TensorView page_table, tvm::ffi::TensorView seq_lens,
                     tvm::ffi::TensorView out);

void Run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache, tvm::ffi::TensorView v_cache,
         tvm::ffi::TensorView topk, tvm::ffi::TensorView page_table, tvm::ffi::TensorView seq_lens,
         tvm::ffi::TensorView out) {
  const int B = static_cast<int>(q.shape()[0]);
  const int pages = static_cast<int>(page_table.shape()[1]);
  // Three visible blocks: every top-k entry is one of them.
  if (pages == 3) {
    if (B <= 2) {
      C12Run(q, k_cache, v_cache, topk, page_table, seq_lens, out);
    } else if (B <= 16) {
      DirectKShortB8Run(q, k_cache, v_cache, topk, page_table, seq_lens, out);
    } else {
      DirectKShortRun(q, k_cache, v_cache, topk, page_table, seq_lens, out);
    }
    return;
  }
  // Full top-16 selection: the direct-K core owns everything but the two
  // smallest batches, where a C16 direct-K donor uses wider output splits.
  if (pages >= 16) {
    if (B <= 2) {
      DirectKShortB8Run(q, k_cache, v_cache, topk, page_table, seq_lens, out);
    } else {
      DirectKF16Run(q, k_cache, v_cache, topk, page_table, seq_lens, out);
    }
    return;
  }
  // 1, 2 or 4..15 visible blocks: the native-F16 core's own split/cluster
  // search covers every remaining (pages, B) with exact K/V dequant.
  F16Run(q, k_cache, v_cache, topk, page_table, seq_lens, out);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(msa_sparse_decode_fp8kv_run, Run);
