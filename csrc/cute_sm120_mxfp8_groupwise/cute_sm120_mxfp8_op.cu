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

#include "cute_sm120_mxfp8_groupwise/cute_sm120_mxfp8_runner.h"
#include "tvm_ffi_utils.h"

void CutlassMXFP8GroupwiseMoeGEMMSM120(TensorView a, TensorView b, TensorView a_scale,
                                       TensorView b_scale, TensorView m_indptr, TensorView out,
                                       std::string scale_major_mode, int64_t scale_granularity_m,
                                       int64_t scale_granularity_n, int64_t scale_granularity_k) {
  TVM_FFI_ICHECK(scale_major_mode == "MN")
      << "Only scale_major_mode=\"MN\" is supported; got \"" << scale_major_mode << "\"";

  TVM_FFI_ICHECK_EQ(scale_granularity_m, 1)
      << "scale_granularity_m must be 1; got " << scale_granularity_m;
  TVM_FFI_ICHECK_EQ(scale_granularity_n, 1)
      << "scale_granularity_n must be 1 (kernel exposes block scaling only along K; "
      << "2D-block B-scale must be broadcast to per-row before calling); got "
      << scale_granularity_n;
  TVM_FFI_ICHECK(scale_granularity_k == 32 || scale_granularity_k == 128)
      << "scale_granularity_k must be 32 or 128; got " << scale_granularity_k;

  // a_scale/b_scale stored as [k_align, m_padded]; .transpose(0,1) caller view is non-contiguous by
  // design.
  CHECK_INPUT_AND_TYPE(a, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(b, dl_float8_e4m3fn);
  CHECK_CUDA(a_scale);
  CHECK_INPUT_TYPE(a_scale, dl_int32);
  CHECK_CUDA(b_scale);
  CHECK_INPUT_TYPE(b_scale, dl_int32);
  CHECK_INPUT_AND_TYPE(m_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  CHECK_DIM(2, a);
  CHECK_DIM(3, b);
  CHECK_DIM(2, out);
  CHECK_DIM(1, m_indptr);

  int total_rows = static_cast<int>(a.size(0));
  int num_experts = static_cast<int>(b.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(a.size(1), k) << "a.size(1) (" << a.size(1) << ") must match b.size(2) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), total_rows)
      << "out.size(0) (" << out.size(0) << ") must match a.size(0) (" << total_rows << ")";
  TVM_FFI_ICHECK_EQ(out.size(1), n)
      << "out.size(1) (" << out.size(1) << ") must match b.size(1) (" << n << ")";
  TVM_FFI_ICHECK_EQ(m_indptr.size(0), num_experts + 1)
      << "m_indptr.size(0) (" << m_indptr.size(0) << ") must be num_experts + 1 ("
      << (num_experts + 1) << ")";

  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::CuteSm120Mxfp8GemmRunner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  // int32-packed UE8M0 reinterpret-cast to float* — matches runner signature convention.
  runner.moe_gemm_mxfp8_nt_groupwise(
      out.data_ptr(), static_cast<void const*>(a.data_ptr()),
      static_cast<void const*>(b.data_ptr()), static_cast<int32_t const*>(m_indptr.data_ptr()),
      num_experts, total_rows, n, k, stream, static_cast<int32_t const*>(a_scale.data_ptr()),
      static_cast<int32_t const*>(b_scale.data_ptr()), static_cast<int>(scale_granularity_k));
}
