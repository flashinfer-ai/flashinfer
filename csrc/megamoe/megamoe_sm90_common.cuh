// Host utilities shared by all SM90 FP4 MegaMoE launchers and bindings.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <flashinfer/megamoe/layout.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe::detail {

inline CUtensorMapDataType tma_dtype(DLDataType dt) {
  if (dt.code == kDLFloat8_e4m3fn || (dt.code == kDLUInt && dt.bits == 8))
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (dt.code == kDLFloat && dt.bits == 32) return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  TVM_FFI_ICHECK(false) << "unsupported TMA dtype";
  return CU_TENSOR_MAP_DATA_TYPE_UINT8;
}

inline CUtensorMapSwizzle tma_swizzle(int mode) {
  switch (mode) {
    case 0:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  TVM_FFI_ICHECK(false) << "unsupported swizzle mode " << mode;
  return CU_TENSOR_MAP_SWIZZLE_NONE;
}

inline CUtensorMap make_tma_2d_desc(void* ptr, DLDataType dtype, int inner, int outer,
                                    int box_inner, int box_outer, int outer_stride, int swizzle) {
  const int elem_size = dtype.bits * dtype.lanes / 8;
  if (swizzle) box_inner = swizzle / elem_size;
  CUtensorMap tm{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_inner),
                             static_cast<cuuint32_t>(box_outer)};
  const cuuint64_t strides[1] = {
      static_cast<cuuint64_t>(static_cast<int64_t>(outer_stride) * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};
  const auto result =
      cuTensorMapEncodeTiled(&tm, tma_dtype(dtype), 2, ptr, dims, strides, box, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, tma_swizzle(swizzle),
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed: " << result;
  return tm;
}

inline CUtensorMap make_tma_sf_desc(void* ptr, DLDataType dtype, int mn, int k, int block_mn,
                                    int gran_k) {
  const int alignment = 16 / (dtype.bits * dtype.lanes / 8);
  mn = (mn + alignment - 1) / alignment * alignment;
  return make_tma_2d_desc(ptr, dtype, mn, (k + gran_k - 1) / gran_k, block_mn, 1, mn, 0);
}

inline tvm::ffi::Array<int64_t> symm_buffer_layout(int ranks, int experts, int capacity, int topk,
                                                   int hidden, int intermediate, int pool,
                                                   int sf_pool) {
  const layout::MegaMoEBuffers b(nullptr, ranks, experts, capacity, topk, hidden, intermediate,
                                 pool, sf_pool);
  auto off = [](const void* p) { return reinterpret_cast<int64_t>(p); };
  return tvm::ffi::Array<int64_t>(
      {off(b.combine_token_buffer.get_end_ptr()), off(b.input_token_buffer.base),
       off(b.input_sf_buffer.base), off(b.input_topk_idx_buffer.base),
       off(b.input_topk_weights_buffer.base), off(b.l1_token_buffer.base), off(b.l1_sf_buffer.base),
       off(b.l2_token_buffer.base), off(b.l2_sf_buffer.base), pool, sf_pool});
}

template <typename T>
inline T* optional_tensor_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& tensor) {
  return tensor.has_value() ? static_cast<T*>(tensor.value().data_ptr()) : nullptr;
}

inline void* offset_ptr(tvm::ffi::TensorView t, int64_t off) {
  return static_cast<char*>(t.data_ptr()) + off;
}

}  // namespace flashinfer::megamoe::detail
