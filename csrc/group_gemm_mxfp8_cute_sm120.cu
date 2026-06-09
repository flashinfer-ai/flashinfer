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

#include "flashinfer/gemm/mxfp8_gemm_cute_sm120.h"
#include "tvm_ffi_utils.h"

void CutlassGroupGemmMXFP8GroupwiseScaledSM120ZeroPadding(
    TensorView a, TensorView b, TensorView a_scale, TensorView b_scale, TensorView m_indptr,
    TensorView out, std::string scale_major_mode, int64_t scale_granularity_m,
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

  flashinfer::gemm::mxfp8_cute_sm120::Mxfp8GemmCuteSm120Runner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  // int32-packed UE8M0 reinterpret-cast to float* — matches runner signature convention.
  runner.group_gemm_mxfp8_nt_groupwise_zero_padding(
      out.data_ptr(), static_cast<void const*>(a.data_ptr()),
      static_cast<void const*>(b.data_ptr()), static_cast<int32_t const*>(m_indptr.data_ptr()),
      num_experts, total_rows, n, k, stream, static_cast<int32_t const*>(a_scale.data_ptr()),
      static_cast<int32_t const*>(b_scale.data_ptr()), static_cast<int>(scale_granularity_k));
}

void CutlassGroupGemmMXFP8GroupwiseScaledSM120Main(TensorView a, TensorView b, TensorView a_scale,
                                                   TensorView b_scale, TensorView m_indices,
                                                   TensorView out, bool use_psum_layout,
                                                   int64_t scale_granularity_m,
                                                   int64_t scale_granularity_n,
                                                   int64_t scale_granularity_k) {
  TVM_FFI_ICHECK_EQ(scale_granularity_m, 1)
      << "scale_granularity_m must be 1; got " << scale_granularity_m;
  TVM_FFI_ICHECK_EQ(scale_granularity_n, 1)
      << "scale_granularity_n must be 1 (kernel exposes block scaling only along K; "
      << "2D-block B-scale must be broadcast to per-row before calling); got "
      << scale_granularity_n;
  TVM_FFI_ICHECK(scale_granularity_k == 32 || scale_granularity_k == 128)
      << "scale_granularity_k must be 32 or 128; got " << scale_granularity_k;

  CHECK_INPUT_AND_TYPE(a, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(b, dl_float8_e4m3fn);
  CHECK_CUDA(a_scale);
  CHECK_INPUT_TYPE(a_scale, dl_int32);
  CHECK_CUDA(b_scale);
  CHECK_INPUT_TYPE(b_scale, dl_int32);
  CHECK_INPUT_AND_TYPE(m_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  CHECK_DIM(2, a);
  CHECK_DIM(3, b);
  CHECK_DIM(2, out);
  CHECK_DIM(1, m_indices);

  int m = static_cast<int>(a.size(0));
  int num_groups = static_cast<int>(b.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(a.size(1), k) << "a.size(1) (" << a.size(1) << ") must match b.size(2) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), m)
      << "out.size(0) (" << out.size(0) << ") must match a.size(0) (" << m << ")";
  TVM_FFI_ICHECK_EQ(out.size(1), n)
      << "out.size(1) (" << out.size(1) << ") must match b.size(1) (" << n << ")";
  // m_indices shape contract:
  //   use_psum_layout=false (contiguous): (m,)         per-row group assignment
  //   use_psum_layout=true  (psum_layout): (num_groups,) per-group cumsum aligned m (no leading 0)
  if (use_psum_layout) {
    TVM_FFI_ICHECK_EQ(m_indices.size(0), num_groups)
        << "use_psum_layout=true requires m_indices.size(0) == num_groups (" << num_groups
        << "); got " << m_indices.size(0);
  } else {
    TVM_FFI_ICHECK_EQ(m_indices.size(0), m)
        << "use_psum_layout=false requires m_indices.size(0) == m (" << m << "); got "
        << m_indices.size(0);
  }

  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::Mxfp8GemmCuteSm120Runner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  runner.group_gemm_mxfp8_nt_groupwise_contiguous(
      out.data_ptr(), static_cast<void const*>(a.data_ptr()),
      static_cast<void const*>(b.data_ptr()), static_cast<int32_t const*>(m_indices.data_ptr()),
      num_groups, m, n, k, stream, static_cast<int32_t const*>(a_scale.data_ptr()),
      static_cast<int32_t const*>(b_scale.data_ptr()), static_cast<int>(scale_granularity_k),
      use_psum_layout);
}

void CutlassGroupGemmMXFP8GroupwiseScaledSM120Masked(TensorView a, TensorView b, TensorView a_scale,
                                                     TensorView b_scale, TensorView masked_m,
                                                     TensorView out, int64_t scale_granularity_m,
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
  CHECK_INPUT_AND_TYPE(masked_m, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, out);
  CHECK_DIM(1, masked_m);

  int num_groups = static_cast<int>(a.size(0));
  int max_m = static_cast<int>(a.size(1));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(b.size(0), num_groups)
      << "b.size(0) (" << b.size(0) << ") must match a.size(0) (" << num_groups << ")";
  TVM_FFI_ICHECK_EQ(a.size(2), k) << "a.size(2) (" << a.size(2) << ") must match b.size(2) (" << k
                                  << ")";
  TVM_FFI_ICHECK_EQ(out.size(0), num_groups);
  TVM_FFI_ICHECK_EQ(out.size(1), max_m);
  TVM_FFI_ICHECK_EQ(out.size(2), n);
  TVM_FFI_ICHECK_EQ(masked_m.size(0), num_groups)
      << "masked_m.size(0) (" << masked_m.size(0) << ") must match num_groups (" << num_groups
      << ")";

  TVM_FFI_ICHECK_EQ(k % 16, 0) << "k must be multiple of 16; got k=" << k;
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be multiple of 16; got n=" << n;

  ffi::CUDADeviceGuard device_guard(a.device().device_id);
  auto stream = get_stream(a.device());

  flashinfer::gemm::mxfp8_cute_sm120::Mxfp8GemmCuteSm120Runner<cute::float_e4m3_t, cute::bfloat16_t,
                                                               float, cute::float_ue8m0_t>
      runner;

  runner.group_gemm_mxfp8_nt_groupwise_masked(
      out.data_ptr(), static_cast<void const*>(a.data_ptr()),
      static_cast<void const*>(b.data_ptr()), static_cast<int const*>(masked_m.data_ptr()),
      num_groups, max_m, n, k, stream, static_cast<int32_t const*>(a_scale.data_ptr()),
      static_cast<int32_t const*>(b_scale.data_ptr()), static_cast<int>(scale_granularity_k));
}

void QuantizeMxfp8ForZeroPaddingCuteSM120(TensorView input, TensorView token_offset,
                                          TensorView out_fp8, TensorView out_scale_raw,
                                          int64_t granK) {
  TVM_FFI_ICHECK(granK == 32 || granK == 128) << "granK must be 32 or 128; got " << granK;
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(token_offset, dl_int32);
  CHECK_INPUT_AND_TYPE(out_fp8, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out_scale_raw, dl_int32);

  CHECK_DIM(2, input);
  CHECK_DIM(1, token_offset);
  CHECK_DIM(2, out_fp8);
  CHECK_DIM(2, out_scale_raw);

  int64_t num_experts = token_offset.size(0) - 1;
  int64_t token_num = input.size(0);
  int64_t size_k = input.size(1);

  TVM_FFI_ICHECK_EQ(size_k % 16, 0) << "k must be multiple of 16; got k=" << size_k;
  TVM_FFI_ICHECK_EQ(out_fp8.size(0), token_num);
  TVM_FFI_ICHECK_EQ(out_fp8.size(1), size_k);

  int64_t pack_nk = granK * 4;
  int64_t k_align = (size_k + pack_nk - 1) / pack_nk;
  int64_t m_padded = (token_num + num_experts * 3) / 4 * 4;
  TVM_FFI_ICHECK_EQ(out_scale_raw.size(0), k_align)
      << "out_scale_raw.size(0) must be k_align=" << k_align;
  TVM_FFI_ICHECK_EQ(out_scale_raw.size(1), m_padded)
      << "out_scale_raw.size(1) must be m_padded=" << m_padded;

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  flashinfer::gemm::mxfp8_cute_sm120::quantize_mxfp8_zero_padding(
      out_fp8.data_ptr(), out_scale_raw.data_ptr(), const_cast<void*>(input.data_ptr()),
      const_cast<void*>(static_cast<void const*>(token_offset.data_ptr())), num_experts, token_num,
      size_k, stream, static_cast<int>(granK));
}
