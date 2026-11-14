// SPDX-License-Identifier: Apache-2.0
// TVM-FFI bindings for the HC kernels.
#include "kernels.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::hc_sum;
using DType = HC_DTYPE;
constexpr int H = HC_HIDDEN;

static void check(TensorView t, DLDataType dtype, int64_t rows, int64_t width, DLDevice device) {
  CHECK_INPUT(t);
  CHECK_DIM(2, t);
  TVM_FFI_ICHECK_EQ(t.size(0), rows);
  TVM_FFI_ICHECK_EQ(t.size(1), width);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), encode_dlpack_dtype(dtype));
  TVM_FFI_ICHECK_EQ(t.device().device_type, device.device_type);
  TVM_FFI_ICHECK_EQ(t.device().device_id, device.device_id);
}

static DLDataType value_dtype() {
  if constexpr (std::is_same_v<DType, nv_bfloat16>) return dl_bfloat16;
  return dl_float16;
}

template <bool Partials>
void apply_impl(TensorView y, TensorView r, TensorView alpha, TensorView out, TensorView sums) {
  const auto rows = r.size(0);
  const auto device = r.device();
  TVM_FFI_ICHECK_GE(rows, 0);
  TVM_FFI_ICHECK_LT(rows, int64_t(1) << 31);
  check(r, value_dtype(), rows, 4 * H, device);
  check(y, value_dtype(), rows, H, device);
  check(out, value_dtype(), rows, 4 * H, device);
  check(alpha, dl_float32, rows, 4, device);
  if constexpr (Partials) {
    CHECK_INPUT(sums);
    CHECK_DIM(3, sums);
    TVM_FFI_ICHECK_EQ(sums.size(0), rows);
    TVM_FFI_ICHECK_EQ(sums.size(1), 4);
    TVM_FFI_ICHECK_EQ(sums.size(2), 10);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(sums.dtype()), float32_code);
    CHECK_DEVICE(sums, r);
  } else {
    check(sums, dl_float32, rows, 4, device);
  }
  if (!rows) return;
  ffi::CUDADeviceGuard guard(device.device_id);
  ApplyParams<DType> p{static_cast<const DType*>(y.data_ptr()),
                       static_cast<const DType*>(r.data_ptr()),
                       static_cast<const float*>(alpha.data_ptr()),
                       static_cast<DType*>(out.data_ptr()), static_cast<float*>(sums.data_ptr())};
  hc_apply_sum_kernel<H, DType, Partials><<<dim3(rows, 8), 160, 0, get_stream(device)>>>(p);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void apply(TensorView y, TensorView r, TensorView a, TensorView out, TensorView sums) {
  apply_impl<false>(y, r, a, out, sums);
}
void apply_partials(TensorView y, TensorView r, TensorView a, TensorView out, TensorView sums) {
  apply_impl<true>(y, r, a, out, sums);
}
void bootstrap(TensorView r, TensorView sums) {
  const auto rows = r.size(0);
  TVM_FFI_ICHECK_LE(rows, ((int64_t(1) << 31) - 1) / 4);
  check(r, value_dtype(), rows, 4 * H, r.device());
  check(sums, dl_float32, rows, 4, r.device());
  if (!rows) return;
  ffi::CUDADeviceGuard guard(r.device().device_id);
  hc_bootstrap_kernel<H, DType><<<rows * 4, H / 16, 0, get_stream(r.device())>>>(
      static_cast<const DType*>(r.data_ptr()), static_cast<float*>(sums.data_ptr()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
void reduce(TensorView partials, TensorView sums) {
  CHECK_INPUT(partials);
  CHECK_DIM(3, partials);
  TVM_FFI_ICHECK_EQ(partials.size(1), 4);
  TVM_FFI_ICHECK_EQ(partials.size(2), 10);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(partials.dtype()), float32_code);
  const auto rows = partials.size(0);
  TVM_FFI_ICHECK_LT(rows, int64_t(1) << 31);
  check(sums, dl_float32, rows, 4, partials.device());
  if (!rows) return;
  ffi::CUDADeviceGuard guard(partials.device().device_id);
  hc_sum_partials_reduce_kernel<<<rows, 128, 0, get_stream(partials.device())>>>(
      static_cast<const float*>(partials.data_ptr()), static_cast<float*>(sums.data_ptr()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply, apply);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_partials, apply_partials);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bootstrap, bootstrap);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reduce, reduce);
