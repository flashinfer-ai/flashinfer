#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

#define uint8_t cake_dsv4_generated_uint8_t
#define uint16_t cake_dsv4_generated_uint16_t
#define uint32_t cake_dsv4_generated_uint32_t
#define uint64_t cake_dsv4_generated_uint64_t
#define int32_t cake_dsv4_generated_int32_t
#define int16_t cake_dsv4_generated_int16_t
#define CakeTensorMap cake_dsv4_generated_CakeTensorMap
#define CUtensorMap cake_dsv4_generated_CUtensorMap
#include "cake_dsv4_split_reduce.cu"
#undef CUtensorMap
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer::cake_dsv4 {

namespace {

static_assert(sizeof(CUtensorMap) == 128);
static_assert(sizeof(cake_dsv4_generated_CakeTensorMap) == sizeof(CUtensorMap));



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

}  // namespace


void Run_split_reduce(TensorView arg_partial_O, TensorView arg_partial_lse, TensorView arg_O, int64_t arg_num_q_heads, int64_t arg_num_split, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError)
      << "cuda_stream must be non-negative";
  ffi::CUDADeviceGuard device_guard(arg_partial_O.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      static_cast<uintptr_t>(cuda_stream));
  CheckCudaTensor(arg_partial_O, "partial_O");
  CheckDtype(arg_partial_O, "partial_O", 4, 16, 1);
  CheckContiguous(arg_partial_O, "partial_O");
  CheckCudaTensor(arg_partial_lse, "partial_lse");
  CheckDtype(arg_partial_lse, "partial_lse", 2, 32, 1);
  CheckContiguous(arg_partial_lse, "partial_lse");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_split >= -2147483648LL && arg_num_split <= 2147483647LL, ValueError)
      << "scalar 'num_split' value " << arg_num_split
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_partial_lse, arg_partial_O, "partial_lse", "partial_O");
  CheckSameCudaDevice(arg_O, arg_partial_O, "O", "partial_O");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";


  void* p_partial_O = arg_partial_O.data_ptr();
  void* p_partial_lse = arg_partial_lse.data_ptr();
  void* p_O = arg_O.data_ptr();
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_split = (int32_t)arg_num_split;
  void* kargs[] = {&p_partial_O, &p_partial_lse, &p_O, &v_num_q_heads, &v_num_split};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(128u, 1u, 1u);
  cudaError_t status = cudaSuccess;
  status = cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_dsv4_split_reduce), grid, block, kargs,
      1024u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_cake_dsv4_split_reduce launch failed: " << cudaGetErrorString(status);
}


}  // namespace flashinfer::cake_dsv4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_split_reduce, flashinfer::cake_dsv4::Run_split_reduce);
