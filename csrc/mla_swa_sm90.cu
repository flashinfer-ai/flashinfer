// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <flashinfer/attention/mla_swa_sm90/launch.cuh>

#include "tvm_ffi_utils.h"

namespace {
void check_tensor(TensorView t, TensorView q, DLDataType dtype) {
  CHECK_CUDA(t);
  CHECK_DEVICE(t, q);
  CHECK_INPUT_TYPE(t, dtype);
}

bool device_supported(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int major, minor, cluster, smem;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, device),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                    cudaSuccess);
  return major == 9 && minor == 0 && cluster && smem >= 223584;
}

void run(TensorView q, TensorView kv, TensorView workspace, TensorView table, TensorView lengths,
         TensorView out, double scale, int64_t window_left, int64_t tactic) {
  check_tensor(q, q, dl_bfloat16);
  check_tensor(kv, q, dl_bfloat16);
  check_tensor(workspace, q, dl_uint8);
  check_tensor(table, q, dl_int32);
  check_tensor(lengths, q, dl_int32);
  check_tensor(out, q, dl_bfloat16);
  CHECK_DIM(4, q);
  CHECK_DIM(3, kv);
  CHECK_DIM(2, table);
  CHECK_DIM(1, lengths);
  CHECK_DIM(1, workspace);
  CHECK_DIM(4, out);
  const auto b = q.size(0), sq = q.size(1), h = q.size(2);
  TVM_FFI_ICHECK(b == 1 || b == 2 || b == 4 || b == 8 || b == 16);
  TVM_FFI_ICHECK(sq == 1 || sq == 4);
  TVM_FFI_ICHECK_EQ(h, 64);
  TVM_FFI_ICHECK_EQ(q.size(3), 1088);
  TVM_FFI_ICHECK_EQ(kv.size(1), 64);
  TVM_FFI_ICHECK_EQ(kv.size(2), 1088);
  TVM_FFI_ICHECK_GT(kv.size(0), 0);
  TVM_FFI_ICHECK_EQ(q.stride(3), 1);
  TVM_FFI_ICHECK_EQ(q.stride(2), 1088);
  TVM_FFI_ICHECK_EQ(q.stride(1), h * 1088);
  TVM_FFI_ICHECK_GE(q.stride(0), sq * h * 1088);
  TVM_FFI_ICHECK_EQ(kv.stride(2), 1);
  TVM_FFI_ICHECK_EQ(kv.stride(1), 1088);
  TVM_FFI_ICHECK_EQ(kv.stride(0), 64 * 1088);
  TVM_FFI_ICHECK_EQ(table.size(0), b);
  TVM_FFI_ICHECK_GT(table.size(1), 0);
  TVM_FFI_ICHECK_EQ(table.stride(1), 1);
  TVM_FFI_ICHECK_GE(table.stride(0), table.size(1));
  TVM_FFI_ICHECK_EQ(lengths.size(0), b);
  TVM_FFI_ICHECK_EQ(lengths.stride(0), 1);
  TVM_FFI_ICHECK_EQ(workspace.stride(0), 1);
  TVM_FFI_ICHECK_GE(workspace.numel(), flashinfer::mla_swa_sm90::workspace_bytes(b * sq * h));
  TVM_FFI_ICHECK_EQ(out.size(0), b);
  TVM_FFI_ICHECK_EQ(out.size(1), sq);
  TVM_FFI_ICHECK_EQ(out.size(2), h);
  TVM_FFI_ICHECK_EQ(out.size(3), 1024);
  TVM_FFI_ICHECK_EQ(out.stride(3), 1);
  TVM_FFI_ICHECK_EQ(out.stride(2), 1024);
  TVM_FFI_ICHECK_EQ(out.stride(1), h * 1024);
  TVM_FFI_ICHECK_EQ(out.stride(0), sq * h * 1024);
  TVM_FFI_ICHECK_EQ(window_left, 512);
  TVM_FFI_ICHECK(std::isfinite(scale) && scale > 0);
  TVM_FFI_ICHECK(tactic >= 0 && tactic < 6);
  for (auto t : {q, kv, workspace, out}) {
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16, 0);
  }
  TVM_FFI_ICHECK(device_supported(q.device().device_id))
      << "Requires SM90 with cluster launch and 223584 bytes opt-in shared memory";
  ffi::CUDADeviceGuard guard(q.device().device_id);
  using namespace flashinfer::mla_swa_sm90;
  Arguments a{static_cast<const cutlass::bfloat16_t*>(q.data_ptr()),
              static_cast<const cutlass::bfloat16_t*>(kv.data_ptr()),
              static_cast<const int*>(table.data_ptr()),
              static_cast<const int*>(lengths.data_ptr()),
              static_cast<cutlass::bfloat16_t*>(out.data_ptr()),
              workspace.data_ptr(),
              static_cast<int>(b),
              static_cast<int>(sq),
              static_cast<int>(h),
              static_cast<int>(kv.size(0)),
              static_cast<int>(table.size(1)),
              513,
              static_cast<float>(scale),
              q.stride(0),
              table.stride(0),
              get_stream(q.device())};
  cudaError_t status;
  switch (tactic) {
    case 0:
      status = launch<mla_swa::M8S9>(a);
      break;
    case 1:
      status = launch<mla_swa::M8S5>(a);
      break;
    case 2:
      status = launch<mla_swa::M8S3>(a);
      break;
    case 3:
      status = launch<mla_swa::M16S2>(a);
      break;
    case 4:
      status = launch<mla_swa::M32S2>(a);
      break;
    case 5:
      status = launch<mla_swa::M32S1>(a);
      break;
    default:
      status = cudaErrorInvalidValue;
  }
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(device_supported, device_supported);
