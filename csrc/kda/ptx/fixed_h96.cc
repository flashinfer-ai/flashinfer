// Static PTX host shim.
// tvm-ffi host launcher for kernel 'kernel_kda_fwd_stream_m128'.
#include <cuda.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

TVM_FFI_EMBED_CUBIN(kda_fwd_stream_m128_84e701abc9);

namespace ptx_host_shim {

using tvm::ffi::TensorView;

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

// Per-launch TMA descriptor table. The caller owns a device buffer with one
// 64-byte-aligned CUtensorMap slot per TMA argument. An upload call (outside
// CUDA Graph capture) encodes every descriptor and copies the table on the
// launch stream; launch calls pass pointers into it and do no host-to-device
// work. The kernels acquire each descriptor with fence.proxy.tensormap before
// use, so a table may be rewritten between launches.
static inline void UploadTmaTable(
    char* table,
    const CUtensorMap* maps,
    size_t count,
    cudaStream_t stream) {
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUresult result = cuStreamIsCapturing(
      reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuStreamIsCapturing for the TMA descriptor table failed: CUresult="
      << static_cast<int>(result);
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "the TMA descriptor table must be uploaded outside CUDA Graph capture";
  result = cuMemcpyHtoDAsync(
      reinterpret_cast<CUdeviceptr>(table), maps, count * sizeof(CUtensorMap),
      reinterpret_cast<CUstream>(stream));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuMemcpyHtoDAsync for the TMA descriptor table failed: CUresult="
      << static_cast<int>(result);
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'q_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'q_tma'";
  uint64_t global_strides[3] = {
      (uint64_t)(((d2 * d1) * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'k_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'k_tma'";
  uint64_t global_strides[3] = {
      (uint64_t)(((d2 * d1) * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'v_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (64, 1, 16) exceeds resolved global dims for 'v_tma'";
  uint64_t global_strides[2] = {
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)(((d1 * d2) * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 1u, 16u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'g_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_g_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'g_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'g_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'g_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 32) exceeds resolved global dims for 'g_tma'";
  uint64_t global_strides[2] = {
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)(((d1 * d2) * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'g_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(8u <= global_dim[0] && 32u <= global_dim[1], ValueError)
      << "TMA box (8, 32) exceeds resolved global dims for 'beta_tma'";
  uint64_t global_strides[1] = {
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'out_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'out_tma'";
  uint64_t global_strides[3] = {
      (uint64_t)(((d2 * d1) * 16) / 8),
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_q, TensorView arg_q_tma, TensorView arg_k, TensorView arg_k_tma, TensorView arg_v, TensorView arg_v_tma, TensorView arg_g, TensorView arg_g_tma, TensorView arg_beta, TensorView arg_beta_tma, TensorView arg_A_log, TensorView arg_dt_bias, TensorView arg_piece_table, TensorView arg_cta_table, TensorView arg_initial_state, TensorView arg_out, TensorView arg_out_tma, TensorView arg_final_state, int64_t arg_num_heads, double arg_scale, double arg_lower_bound, int64_t arg_failure_addr, int64_t arg_generation, double arg_detect_threshold_log2, int64_t arg_handoff_flags_addr, TensorView arg_tma_table, int64_t arg_tma_upload, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_q, "q");
  CheckDtype(arg_q, "q", 4, 16, 1);
  CheckContiguous(arg_q, "q");
  CheckCudaTensor(arg_q_tma, "q_tma");
  CheckDtype(arg_q_tma, "q_tma", 4, 16, 1);
  CheckContiguous(arg_q_tma, "q_tma");
  CheckCudaTensor(arg_k, "k");
  CheckDtype(arg_k, "k", 4, 16, 1);
  CheckContiguous(arg_k, "k");
  CheckCudaTensor(arg_k_tma, "k_tma");
  CheckDtype(arg_k_tma, "k_tma", 4, 16, 1);
  CheckContiguous(arg_k_tma, "k_tma");
  CheckCudaTensor(arg_v, "v");
  CheckDtype(arg_v, "v", 4, 16, 1);
  CheckContiguous(arg_v, "v");
  CheckCudaTensor(arg_v_tma, "v_tma");
  CheckDtype(arg_v_tma, "v_tma", 4, 16, 1);
  CheckContiguous(arg_v_tma, "v_tma");
  CheckCudaTensor(arg_g, "g");
  CheckDtype(arg_g, "g", 4, 16, 1);
  CheckContiguous(arg_g, "g");
  CheckCudaTensor(arg_g_tma, "g_tma");
  CheckDtype(arg_g_tma, "g_tma", 4, 16, 1);
  CheckContiguous(arg_g_tma, "g_tma");
  CheckCudaTensor(arg_beta, "beta");
  CheckDtype(arg_beta, "beta", 4, 16, 1);
  CheckContiguous(arg_beta, "beta");
  CheckCudaTensor(arg_beta_tma, "beta_tma");
  CheckDtype(arg_beta_tma, "beta_tma", 4, 16, 1);
  CheckCudaTensor(arg_A_log, "A_log");
  CheckDtype(arg_A_log, "A_log", 2, 32, 1);
  CheckContiguous(arg_A_log, "A_log");
  CheckCudaTensor(arg_dt_bias, "dt_bias");
  CheckDtype(arg_dt_bias, "dt_bias", 2, 32, 1);
  CheckContiguous(arg_dt_bias, "dt_bias");
  CheckCudaTensor(arg_piece_table, "piece_table");
  CheckDtype(arg_piece_table, "piece_table", 0, 32, 1);
  CheckContiguous(arg_piece_table, "piece_table");
  CheckCudaTensor(arg_cta_table, "cta_table");
  CheckDtype(arg_cta_table, "cta_table", 0, 32, 1);
  CheckContiguous(arg_cta_table, "cta_table");
  CheckCudaTensor(arg_initial_state, "initial_state");
  CheckDtype(arg_initial_state, "initial_state", 2, 32, 1);
  CheckContiguous(arg_initial_state, "initial_state");
  CheckCudaTensor(arg_out, "out");
  CheckDtype(arg_out, "out", 4, 16, 1);
  CheckContiguous(arg_out, "out");
  CheckCudaTensor(arg_out_tma, "out_tma");
  CheckDtype(arg_out_tma, "out_tma", 4, 16, 1);
  CheckContiguous(arg_out_tma, "out_tma");
  CheckCudaTensor(arg_final_state, "final_state");
  CheckDtype(arg_final_state, "final_state", 2, 32, 1);
  CheckContiguous(arg_final_state, "final_state");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_failure_addr >= -9223372036854775808LL && arg_failure_addr <= 9223372036854775807LL, ValueError)
      << "scalar 'failure_addr' value " << arg_failure_addr
      << " is outside i64 range [-9223372036854775808, 9223372036854775807]";
  TVM_FFI_CHECK(arg_generation >= -2147483648LL && arg_generation <= 2147483647LL, ValueError)
      << "scalar 'generation' value " << arg_generation
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_handoff_flags_addr >= -9223372036854775808LL && arg_handoff_flags_addr <= 9223372036854775807LL, ValueError)
      << "scalar 'handoff_flags_addr' value " << arg_handoff_flags_addr
      << " is outside i64 range [-9223372036854775808, 9223372036854775807]";
  CheckSameCudaDevice(arg_q_tma, arg_q, "q_tma", "q");
  CheckSameCudaDevice(arg_k, arg_q, "k", "q");
  CheckSameCudaDevice(arg_k_tma, arg_q, "k_tma", "q");
  CheckSameCudaDevice(arg_v, arg_q, "v", "q");
  CheckSameCudaDevice(arg_v_tma, arg_q, "v_tma", "q");
  CheckSameCudaDevice(arg_g, arg_q, "g", "q");
  CheckSameCudaDevice(arg_g_tma, arg_q, "g_tma", "q");
  CheckSameCudaDevice(arg_beta, arg_q, "beta", "q");
  CheckSameCudaDevice(arg_beta_tma, arg_q, "beta_tma", "q");
  CheckSameCudaDevice(arg_A_log, arg_q, "A_log", "q");
  CheckSameCudaDevice(arg_dt_bias, arg_q, "dt_bias", "q");
  CheckSameCudaDevice(arg_piece_table, arg_q, "piece_table", "q");
  CheckSameCudaDevice(arg_cta_table, arg_q, "cta_table", "q");
  CheckSameCudaDevice(arg_initial_state, arg_q, "initial_state", "q");
  CheckSameCudaDevice(arg_out, arg_q, "out", "q");
  CheckSameCudaDevice(arg_out_tma, arg_q, "out_tma", "q");
  CheckSameCudaDevice(arg_final_state, arg_q, "final_state", "q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_q.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CheckCudaTensor(arg_tma_table, "tma_table");
  CheckDtype(arg_tma_table, "tma_table", 1, 8, 1);
  CheckContiguous(arg_tma_table, "tma_table");
  CheckSameCudaDevice(arg_tma_table, arg_q, "tma_table", "q");
  TVM_FFI_CHECK(arg_tma_table.numel() >= 6 * (int64_t)sizeof(CUtensorMap) &&
                    reinterpret_cast<uintptr_t>(arg_tma_table.data_ptr()) % 64 == 0,
                ValueError)
      << "tma_table must hold 6 64-byte-aligned CUtensorMap slots";
  char* tma_slots = static_cast<char*>(arg_tma_table.data_ptr());
  CUtensorMap tma_maps[6];
  void* p_q = arg_q.data_ptr();
  if (arg_tma_upload) tma_maps[0] = EncodeTma_q_tma(arg_q_tma);
  void* p_q_tma = tma_slots + 0 * sizeof(CUtensorMap);
  void* p_k = arg_k.data_ptr();
  if (arg_tma_upload) tma_maps[1] = EncodeTma_k_tma(arg_k_tma);
  void* p_k_tma = tma_slots + 1 * sizeof(CUtensorMap);
  void* p_v = arg_v.data_ptr();
  if (arg_tma_upload) tma_maps[2] = EncodeTma_v_tma(arg_v_tma);
  void* p_v_tma = tma_slots + 2 * sizeof(CUtensorMap);
  void* p_g = arg_g.data_ptr();
  if (arg_tma_upload) tma_maps[3] = EncodeTma_g_tma(arg_g_tma);
  void* p_g_tma = tma_slots + 3 * sizeof(CUtensorMap);
  void* p_beta = arg_beta.data_ptr();
  if (arg_tma_upload) tma_maps[4] = EncodeTma_beta_tma(arg_beta_tma);
  void* p_beta_tma = tma_slots + 4 * sizeof(CUtensorMap);
  void* p_A_log = arg_A_log.data_ptr();
  void* p_dt_bias = arg_dt_bias.data_ptr();
  void* p_piece_table = arg_piece_table.data_ptr();
  void* p_cta_table = arg_cta_table.data_ptr();
  void* p_initial_state = arg_initial_state.data_ptr();
  void* p_out = arg_out.data_ptr();
  if (arg_tma_upload) tma_maps[5] = EncodeTma_out_tma(arg_out_tma);
  void* p_out_tma = tma_slots + 5 * sizeof(CUtensorMap);
  void* p_final_state = arg_final_state.data_ptr();
  if (arg_tma_upload) {
    UploadTmaTable(tma_slots, tma_maps, 6, stream);
    return;
  }
  int32_t v_num_heads = (int32_t)arg_num_heads;
  float v_scale = (float)arg_scale;
  float v_lower_bound = (float)arg_lower_bound;
  int64_t v_failure_addr = (int64_t)arg_failure_addr;
  int32_t v_generation = (int32_t)arg_generation;
  float v_detect_threshold_log2 = (float)arg_detect_threshold_log2;
  int64_t v_handoff_flags_addr = (int64_t)arg_handoff_flags_addr;
  void* kargs[] = {&p_q, &p_q_tma, &p_k, &p_k_tma, &p_v, &p_v_tma, &p_g, &p_g_tma, &p_beta, &p_beta_tma, &p_A_log, &p_dt_bias, &p_piece_table, &p_cta_table, &p_initial_state, &p_out, &p_out_tma, &p_final_state, &v_num_heads, &v_scale, &v_lower_bound, &v_failure_addr, &v_generation, &v_detect_threshold_log2, &v_handoff_flags_addr};

  static auto kernel = EmbedCubinModule_kda_fwd_stream_m128_84e701abc9::Global()->mod.GetKernelWithMaxDynamicSharedMemory("kernel_kda_fwd_stream_m128", 232448);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(1024u, 1u, 1u);

  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 232448u));
}

}  // namespace ptx_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, ptx_host_shim::Run);
