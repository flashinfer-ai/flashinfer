// Host utilities shared by the replica-plan launchers: the remap and
// weight-copy module and the MXFP4 fused kernel's in-kernel weight copy.
#pragma once

#include <flashinfer/megamoe/eplb_layout.cuh>
#include <flashinfer/megamoe/sym_buffer.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe::detail {

// The replica-plan workspace of this rank: an INT32 CUDA tensor of at least
// `layout.num_words()` words, 16-byte aligned.
inline uint32_t* eplb_workspace_ptr(const tvm::ffi::TensorView& workspace,
                                    const eplb::Workspace& layout) {
  CHECK_CUDA(workspace);
  CHECK_CONTIGUOUS(workspace);
  const DLDataType i32{kDLInt, 32, 1};
  TVM_FFI_ICHECK_EQ(workspace.dtype(), i32) << "workspace must be INT32";
  TVM_FFI_ICHECK_GE(workspace.numel(), layout.num_words()) << "workspace too small";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16, 0)
      << "workspace must be 16-byte aligned";
  return static_cast<uint32_t*>(workspace.data_ptr());
}

// Logical routing of this rank: INT64 `[>= num_tokens, num_topk]`, contiguous.
inline int64_t* eplb_topk_idx_ptr(const tvm::ffi::TensorView& topk_idx, int64_t num_tokens) {
  CHECK_CUDA(topk_idx);
  CHECK_CONTIGUOUS(topk_idx);
  const DLDataType i64{kDLInt, 64, 1};
  TVM_FFI_ICHECK_EQ(topk_idx.dtype(), i64) << "topk_idx must be INT64";
  TVM_FFI_ICHECK_EQ(topk_idx.ndim(), 2);
  TVM_FFI_ICHECK_GE(num_tokens, 1);
  TVM_FFI_ICHECK_GE(topk_idx.size(0), num_tokens);
  TVM_FFI_ICHECK_GE(topk_idx.size(1), 1);
  return static_cast<int64_t*>(topk_idx.data_ptr());
}

// Append the planes of `specs`, a flat list of (byte offset into the arena,
// bytes per slot) pairs, each holding `num_slots` slots inside `arena_bytes`.
inline void eplb_append_planes(eplb::CopyPlanes& planes, const tvm::ffi::Array<int64_t>& specs,
                               int64_t arena_bytes, int64_t num_slots) {
  TVM_FFI_ICHECK_EQ(specs.size() % 2, 0) << "planes are (offset, bytes per slot) pairs";
  for (size_t i = 0; i < specs.size(); i += 2) {
    const int64_t offset = specs[i], bytes_per_slot = specs[i + 1];
    TVM_FFI_ICHECK_LT(planes.num_planes, eplb::CopyPlanes::kMaxPlanes) << "too many weight planes";
    TVM_FFI_ICHECK(offset >= 0 and offset % 4 == 0 and bytes_per_slot > 0 and
                   bytes_per_slot % 4 == 0)
        << "weight planes must be 4-byte aligned";
    TVM_FFI_ICHECK_LE(offset + bytes_per_slot * num_slots, arena_bytes)
        << "weight plane exceeds the arena";
    const bool vec16 = offset % 16 == 0 and bytes_per_slot % 16 == 0;
    planes.planes[planes.num_planes++] = {static_cast<uint64_t>(offset),
                                          static_cast<uint32_t>(bytes_per_slot), vec16 ? 16u : 4u};
  }
}

// A symmetric buffer of this rank (the replica-plan workspace or the weight
// arena): a CUDA tensor whose peer base addresses are `ptrs` (one per rank,
// this rank's equal to its own), each 16-byte aligned.
template <uint32_t kNumRanks>
inline layout::SymBuffer<kNumRanks> eplb_sym_buffer(const tvm::ffi::TensorView& tensor,
                                                    const tvm::ffi::Array<int64_t>& ptrs,
                                                    int64_t rank_idx) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TVM_FFI_ICHECK_EQ(ptrs.size(), kNumRanks);
  TVM_FFI_ICHECK(rank_idx >= 0 and rank_idx < static_cast<int64_t>(kNumRanks));
  TVM_FFI_ICHECK_EQ(ptrs[rank_idx], reinterpret_cast<int64_t>(tensor.data_ptr()))
      << "ptrs[rank_idx] must be this rank's buffer";
  for (size_t i = 0; i < ptrs.size(); ++i)
    TVM_FFI_ICHECK_EQ(ptrs[i] % 16, 0) << "symmetric buffers must be 16-byte aligned";
  return layout::SymBuffer<kNumRanks>(ptrs, static_cast<uint32_t>(rank_idx));
}

}  // namespace flashinfer::megamoe::detail
