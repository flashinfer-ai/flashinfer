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
#include <cstdint>
#include <string>

#include "cute_sm120_mxfp8_groupwise/cute_sm120_fp8_runner.h"
#include "tvm_ffi_utils.h"

void CutlassFP8GroupwiseMoeGEMMSM120(TensorView a, TensorView b, TensorView a_scale,
                                     TensorView b_scale, TensorView m_indptr, TensorView out,
                                     std::string scale_major_mode, int64_t scale_granularity_m,
                                     int64_t scale_granularity_n, int64_t scale_granularity_k) {
  TVM_FFI_ICHECK(scale_major_mode == "MN")
      << "Only scale_major_mode=\"MN\" is supported; got \"" << scale_major_mode << "\"";

  TVM_FFI_ICHECK(scale_granularity_m == 1 && scale_granularity_n == 128 &&
                 scale_granularity_k == 128)
      << "scale_granularity_mnk must be (1, 128, 128); got (" << scale_granularity_m << ", "
      << scale_granularity_n << ", " << scale_granularity_k << ")";

  // a_scale is a zero-padding-mode float SFA: contiguous [Kb, MpE] with per-expert
  // 4-row-aligned offsets; b_scale is a compact float SFB [num_experts, Kb, Nb].
  CHECK_INPUT_AND_TYPE(a, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(b, dl_float8_e4m3fn);
  CHECK_CUDA(a_scale);
  CHECK_INPUT_TYPE(a_scale, dl_float32);
  CHECK_CONTIGUOUS(a_scale);
  CHECK_CUDA(b_scale);
  CHECK_INPUT_TYPE(b_scale, dl_float32);
  CHECK_CONTIGUOUS(b_scale);
  CHECK_INPUT_AND_TYPE(m_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, a_scale);
  CHECK_DEVICE(a, b_scale);
  CHECK_DEVICE(a, m_indptr);
  CHECK_DEVICE(a, out);

  CHECK_DIM(2, a);
  CHECK_DIM(3, b);
  CHECK_DIM(2, a_scale);
  CHECK_DIM(3, b_scale);
  CHECK_DIM(2, out);
  CHECK_DIM(1, m_indptr);

  int total_rows = static_cast<int>(a.size(0));
  int num_experts = static_cast<int>(b.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_GT(num_experts, 0) << "num_experts must be positive; got " << num_experts;
  TVM_FFI_ICHECK_EQ(a.size(1), k) << "a.size(1) (" << a.size(1) << ") must match b.size(2) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), total_rows)
      << "out.size(0) (" << out.size(0) << ") must match a.size(0) (" << total_rows << ")";
  TVM_FFI_ICHECK_EQ(out.size(1), n)
      << "out.size(1) (" << out.size(1) << ") must match b.size(1) (" << n << ")";
  TVM_FFI_ICHECK_EQ(m_indptr.size(0), num_experts + 1)
      << "m_indptr.size(0) (" << m_indptr.size(0) << ") must be num_experts + 1 ("
      << (num_experts + 1) << ")";

  TVM_FFI_ICHECK_GT(k, 0) << "k must be positive; got k=" << k;
  TVM_FFI_ICHECK_GT(n, 0) << "n must be positive; got n=" << n;
  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  auto ceil_div = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  auto compute_padded_offset = [](int64_t offset, int64_t problem_idx) {
    constexpr int64_t kAlignment = 4;
    return (offset + problem_idx * (kAlignment - 1)) / kAlignment * kAlignment;
  };

  int64_t expected_scale_k = ceil_div(k, scale_granularity_k);
  int64_t expected_scale_n = ceil_div(n, scale_granularity_n);
  int64_t expected_a_scale_m = compute_padded_offset(total_rows, num_experts);

  TVM_FFI_ICHECK(a_scale.size(0) == expected_scale_k && a_scale.size(1) == expected_a_scale_m)
      << "a_scale must have zero-padding shape [" << expected_scale_k << ", " << expected_a_scale_m
      << "] (contiguous [Kb, MpE], MpE = (total_rows + 3 * num_experts) / 4"
      << " * 4), got [" << a_scale.size(0) << ", " << a_scale.size(1) << "]";
  TVM_FFI_ICHECK(reinterpret_cast<std::uintptr_t>(a_scale.data_ptr()) % 16 == 0)
      << "a_scale data pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(b_scale.size(0) == num_experts && b_scale.size(1) == expected_scale_k &&
                 b_scale.size(2) == expected_scale_n)
      << "b_scale must have shape [" << num_experts << ", " << expected_scale_k << ", "
      << expected_scale_n << "], got [" << b_scale.size(0) << ", " << b_scale.size(1) << ", "
      << b_scale.size(2) << "]";

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::CuteSm120Fp8GemmRunner<cute::float_e4m3_t, cute::bfloat16_t,
                                                             float, float>
      runner;

  runner.moe_gemm_fp8_nt_groupwise(
      out.data_ptr(), static_cast<void const*>(a.data_ptr()),
      static_cast<void const*>(b.data_ptr()), static_cast<int32_t const*>(m_indptr.data_ptr()),
      num_experts, total_rows, n, k, stream, static_cast<float const*>(a_scale.data_ptr()),
      static_cast<float const*>(b_scale.data_ptr()), static_cast<int>(scale_granularity_m),
      static_cast<int>(scale_granularity_n), static_cast<int>(scale_granularity_k));
}
