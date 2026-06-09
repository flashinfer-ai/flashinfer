/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>

#include "flashinfer/gemm/mxfp8_cute_gemm_sm120.h"
#include "tvm_ffi_utils.h"

void CutlassGemmMxfp8GroupwiseScaledCuteSM120(TensorView a, TensorView b, TensorView a_scale,
                                              TensorView b_scale, TensorView out,
                                              int64_t scale_granularity_m,
                                              int64_t scale_granularity_n,
                                              int64_t scale_granularity_k) {
  TVM_FFI_ICHECK_EQ(scale_granularity_m, 1)
      << "scale_granularity_m must be 1; got " << scale_granularity_m;
  TVM_FFI_ICHECK_EQ(scale_granularity_n, 1)
      << "scale_granularity_n must be 1 (kernel exposes block scaling only along K); got "
      << scale_granularity_n;
  TVM_FFI_ICHECK(scale_granularity_k == 32 || scale_granularity_k == 128)
      << "scale_granularity_k must be 32 or 128; got " << scale_granularity_k;

  CHECK_INPUT_AND_TYPE(a, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(b, dl_float8_e4m3fn);
  CHECK_CUDA(a_scale);
  CHECK_INPUT_TYPE(a_scale, dl_int32);
  CHECK_CUDA(b_scale);
  CHECK_INPUT_TYPE(b_scale, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, out);

  int m = static_cast<int>(a.size(0));
  int n = static_cast<int>(b.size(0));
  int k = static_cast<int>(b.size(1));

  TVM_FFI_ICHECK_EQ(a.size(1), k) << "a.size(1) (" << a.size(1) << ") must match b.size(1) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), m);
  TVM_FFI_ICHECK_EQ(out.size(1), n);

  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::Mxfp8CuteGemmSm120Runner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  runner.gemm_mxfp8_nt_groupwise(out.data_ptr(), static_cast<void const*>(a.data_ptr()),
                                 static_cast<void const*>(b.data_ptr()), m, n, k,
                                 reinterpret_cast<float const*>(a_scale.data_ptr()),
                                 reinterpret_cast<float const*>(b_scale.data_ptr()), stream,
                                 static_cast<int>(scale_granularity_k));
}

void CutlassBatchGemmMxfp8GroupwiseScaledCuteSM120(
    TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView out, int64_t lda,
    int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t ldd, int64_t stride_d,
    int64_t scale_granularity_m, int64_t scale_granularity_n, int64_t scale_granularity_k) {
  TVM_FFI_ICHECK_EQ(scale_granularity_m, 1)
      << "scale_granularity_m must be 1; got " << scale_granularity_m;
  TVM_FFI_ICHECK_EQ(scale_granularity_n, 1)
      << "scale_granularity_n must be 1 (kernel exposes block scaling only along K); got "
      << scale_granularity_n;
  TVM_FFI_ICHECK(scale_granularity_k == 32 || scale_granularity_k == 128)
      << "scale_granularity_k must be 32 or 128; got " << scale_granularity_k;

  CHECK_CUDA(a);
  CHECK_INPUT_TYPE(a, dl_float8_e4m3fn);
  CHECK_CUDA(b);
  CHECK_INPUT_TYPE(b, dl_float8_e4m3fn);
  CHECK_CUDA(a_scale);
  CHECK_INPUT_TYPE(a_scale, dl_int32);
  CHECK_CUDA(b_scale);
  CHECK_INPUT_TYPE(b_scale, dl_int32);
  CHECK_CUDA(out);
  CHECK_INPUT_TYPE(out, dl_bfloat16);

  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, out);

  int num_groups = static_cast<int>(a.size(0));
  int m = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(b.size(0), num_groups)
      << "b.size(0) (" << b.size(0) << ") must match a.size(0) (" << num_groups << ")";
  TVM_FFI_ICHECK_EQ(a.size(2), k) << "a.size(2) (" << a.size(2) << ") must match b.size(2) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(out.size(1), m);
  TVM_FFI_ICHECK_EQ(out.size(2), n);

  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  TVM_FFI_ICHECK_EQ(a.stride(2), 1)
      << "a.stride(2) must be 1 (k-major contiguous on last dim); got " << a.stride(2);
  TVM_FFI_ICHECK_EQ(b.stride(2), 1)
      << "b.stride(2) must be 1 (k-major contiguous on last dim); got " << b.stride(2);
  TVM_FFI_ICHECK_EQ(out.stride(2), 1)
      << "out.stride(2) must be 1 (n-major contiguous on last dim); got " << out.stride(2);

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::Mxfp8CuteGemmSm120Runner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  runner.batch_gemm_mxfp8_nt_groupwise(
      const_cast<void*>(static_cast<void const*>(a.data_ptr())), static_cast<int>(lda),
      static_cast<int>(stride_a), const_cast<void*>(static_cast<void const*>(b.data_ptr())),
      static_cast<int>(ldb), static_cast<int>(stride_b), out.data_ptr(), static_cast<int>(ldd),
      static_cast<int>(stride_d), reinterpret_cast<float*>(a_scale.data_ptr()),
      reinterpret_cast<float*>(b_scale.data_ptr()), num_groups, m, n, k, stream,
      static_cast<int>(scale_granularity_k));
}
