/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <flashinfer/attention/sm120/qk_mxfp8_pv_nvfp4_attention_sm120/api/launcher.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "dlpack/dlpack.h"

using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float8_e4m3fn = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};

#define CHECK_CUDA(x) \
  TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA) << #x " must be a CUDA tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x.ndim(), d) << #x " must be a " #d "D tensor";

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline int64_t get_element_size(ffi::TensorView x) {
  return (x.dtype().bits * x.dtype().lanes) / 8;
}

namespace flashinfer {
namespace qk_mxfp8_pv_nvfp4_attention_sm120 {

namespace {

using FlashFwdParams = ::qk_mxfp8_pv_nvfp4_attention::Flash_fwd_params;

int64_t numel(TensorView x) {
  int64_t n = 1;
  for (int i = 0; i < x.ndim(); ++i) {
    n *= x.size(i);
  }
  return n;
}

void check_same_device(TensorView ref, TensorView x, const char* name) {
  TVM_FFI_ICHECK_EQ(ref.device().device_type, x.device().device_type)
      << name << " must be on the same device as q_fp8";
  TVM_FFI_ICHECK_EQ(ref.device().device_id, x.device().device_id)
      << name << " must be on the same device as q_fp8";
}

int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

void set_params_fprop(FlashFwdParams& params, TensorView q, TensorView k, TensorView v,
                      TensorView q_scale, TensorView k_scale, TensorView v_scale, TensorView out,
                      ffi::Optional<TensorView> maybe_lse, float sm_scale, bool causal,
                      int64_t unpadded_q_len, int64_t unpadded_k_len) {
  params = {};

  const int batch = static_cast<int>(q.size(0));
  const int num_qo_heads = static_cast<int>(q.size(1));
  const int num_kv_heads = static_cast<int>(k.size(1));
  const int seq_len_q = static_cast<int>(q.size(2));
  const int seq_len_k = static_cast<int>(k.size(2));
  const int head_dim = static_cast<int>(q.size(3));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.sfq_ptr = q_scale.data_ptr();
  params.sfk_ptr = k_scale.data_ptr();
  params.sfv_ptr = v_scale.data_ptr();

  params.q_row_stride = q.stride(-2);
  params.k_row_stride = k.stride(-2);
  params.v_row_stride = v.stride(-2) * 2;
  params.q_head_stride = q.stride(-3);
  params.k_head_stride = k.stride(-3);
  params.v_head_stride = v.stride(-3) * 2;
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0) * 2;

  params.sfq_row_stride = q_scale.stride(-2);
  params.sfk_row_stride = k_scale.stride(-2);
  params.sfv_row_stride = v_scale.stride(-2);
  params.sfq_head_stride = q_scale.stride(-3);
  params.sfk_head_stride = k_scale.stride(-3);
  params.sfv_head_stride = v_scale.stride(-3);
  params.sfq_batch_stride = q_scale.stride(0);
  params.sfk_batch_stride = k_scale.stride(0);
  params.sfv_batch_stride = v_scale.stride(0);

  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-2);
  params.o_head_stride = out.stride(-3);
  params.o_batch_stride = out.stride(0);

  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;
  params.p_ptr = nullptr;
  params.softmax_lse_ptr =
      maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;

  params.b = batch;
  params.h = num_qo_heads;
  params.h_k = num_kv_heads;
  params.h_h_k_ratio = num_qo_heads / num_kv_heads;
  params.seqlen_q = seq_len_q;
  params.seqlen_k = seq_len_k;
  params.unpadded_seqlen_q = static_cast<int>(unpadded_q_len);
  params.unpadded_seqlen_k = static_cast<int>(unpadded_k_len);
  params.seqlen_q_rounded = round_multiple(seq_len_q, 128);
  params.seqlen_k_rounded = round_multiple(seq_len_k, 128);
  params.d = head_dim;
  params.d_rounded = head_dim;
  params.head_divmod = cutlass::FastDivmod(num_qo_heads);

  params.scale_softmax = sm_scale;
  params.scale_softmax_log2 = sm_scale * 1.4426950408889634f;
  __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
  __half2 scale_softmax_log2_half2 =
      __halves2half2(scale_softmax_log2_half, scale_softmax_log2_half);
  params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.f;
  params.scale_softmax_rp_dropout = sm_scale;

  params.is_causal = causal;
  params.per_block_mean = false;
  params.seqlen_s = 0;
  params.window_size_left = -1;
  params.window_size_right = causal ? 0 : -1;
  params.is_seqlens_k_cumulative = true;
  params.is_bf16 = out.dtype() == dl_bfloat16;
  params.tile_count_semaphore = nullptr;
}

template <bool ReturnLSE, bool IsBF16>
void run_mha_fwd_dispatch_dtype(FlashFwdParams& params, cudaStream_t stream) {
  using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
  if (params.d == 128) {
    ::qk_mxfp8_pv_nvfp4_attention::run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 128,
                                                OType, float, ReturnLSE>(params, stream);
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported head dimension " << params.d;
  }
}

void run_mha_fwd(FlashFwdParams& params, cudaStream_t stream) {
  if (params.softmax_lse_ptr != nullptr) {
    if (params.is_bf16) {
      run_mha_fwd_dispatch_dtype<true, true>(params, stream);
    } else {
      run_mha_fwd_dispatch_dtype<true, false>(params, stream);
    }
  } else if (params.is_bf16) {
    run_mha_fwd_dispatch_dtype<false, true>(params, stream);
  } else {
    run_mha_fwd_dispatch_dtype<false, false>(params, stream);
  }
}

}  // namespace

void fwd(TensorView q_fp8, TensorView k_fp8, TensorView v_fp4_t, TensorView q_scale,
         TensorView k_scale, TensorView v_scale_t, TensorView out,
         ffi::Optional<TensorView> maybe_lse, double sm_scale, bool causal, int64_t unpadded_q_len,
         int64_t unpadded_k_len) {
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(k_fp8);
  CHECK_INPUT(v_fp4_t);
  CHECK_INPUT(q_scale);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale_t);
  CHECK_INPUT(out);
  if (maybe_lse.has_value()) {
    CHECK_INPUT(maybe_lse.value());
  }

  CHECK_DIM(4, q_fp8);
  CHECK_DIM(4, k_fp8);
  CHECK_DIM(4, v_fp4_t);
  CHECK_DIM(4, q_scale);
  CHECK_DIM(4, k_scale);
  CHECK_DIM(4, v_scale_t);
  CHECK_DIM(4, out);
  if (maybe_lse.has_value()) {
    CHECK_DIM(3, maybe_lse.value());
  }

  TVM_FFI_ICHECK_EQ(q_fp8.dtype(), dl_float8_e4m3fn) << "q_fp8 must be float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(k_fp8.dtype(), dl_float8_e4m3fn) << "k_fp8 must be float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(v_fp4_t.dtype(), dl_uint8) << "v_fp4_t must be uint8 packed FP4";
  TVM_FFI_ICHECK_EQ(q_scale.dtype(), dl_uint8) << "q_scale must be uint8 UE8M0";
  TVM_FFI_ICHECK_EQ(k_scale.dtype(), dl_uint8) << "k_scale must be uint8 UE8M0";
  TVM_FFI_ICHECK_EQ(v_scale_t.dtype(), dl_float8_e4m3fn) << "v_scale_t must be float8_e4m3fn";
  if (maybe_lse.has_value()) {
    TVM_FFI_ICHECK_EQ(maybe_lse.value().dtype(), dl_float32) << "lse must be float32";
  }
  TVM_FFI_ICHECK(out.dtype() == dl_bfloat16 || out.dtype() == dl_float16)
      << "out must be bfloat16 or float16";

  ffi::CUDADeviceGuard device_guard(q_fp8.device().device_id);
  cudaDeviceProp props;
  cudaError_t status = cudaGetDeviceProperties(&props, q_fp8.device().device_id);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaGetDeviceProperties failed: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK(props.major == 12 && (props.minor == 0 || props.minor == 1))
      << "QK MXFP8 / PV NVFP4 attention requires compute capability 12.0 or 12.1";

  const int64_t batch = q_fp8.size(0);
  const int64_t num_qo_heads = q_fp8.size(1);
  const int64_t num_kv_heads = k_fp8.size(1);
  const int64_t seq_len_q = q_fp8.size(2);
  const int64_t seq_len_k = k_fp8.size(2);
  const int64_t head_dim = q_fp8.size(3);

  TVM_FFI_ICHECK_EQ(head_dim, 128) << "head_dim must be 128";
  TVM_FFI_ICHECK_EQ(seq_len_q % 128, 0) << "Q sequence length must be a multiple of 128";
  TVM_FFI_ICHECK_EQ(seq_len_k % 128, 0) << "K/V sequence length must be a multiple of 128";
  TVM_FFI_ICHECK_GT(unpadded_q_len, 0) << "unpadded_q_len must be positive";
  TVM_FFI_ICHECK_LE(unpadded_q_len, seq_len_q)
      << "unpadded_q_len must not exceed the physical Q sequence length";
  TVM_FFI_ICHECK_GT(unpadded_k_len, 0) << "unpadded_k_len must be positive";
  TVM_FFI_ICHECK_LE(unpadded_k_len, seq_len_k)
      << "unpadded_k_len must not exceed the physical K/V sequence length";
  TVM_FFI_ICHECK_GT(num_qo_heads, 0) << "num_qo_heads must be positive";
  TVM_FFI_ICHECK_GT(num_kv_heads, 0) << "num_kv_heads must be positive";
  TVM_FFI_ICHECK_GE(num_qo_heads, num_kv_heads)
      << "num_qo_heads must be greater than or equal to num_kv_heads";
  TVM_FFI_ICHECK_EQ(num_qo_heads % num_kv_heads, 0)
      << "num_qo_heads must be divisible by num_kv_heads";

  TVM_FFI_ICHECK_EQ(k_fp8.size(0), batch);
  TVM_FFI_ICHECK_EQ(k_fp8.size(3), q_fp8.size(3));

  TVM_FFI_ICHECK_EQ(v_fp4_t.size(0), batch);
  TVM_FFI_ICHECK_EQ(v_fp4_t.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(v_fp4_t.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(v_fp4_t.size(3), seq_len_k / 2);

  TVM_FFI_ICHECK_EQ(q_scale.size(0), batch);
  TVM_FFI_ICHECK_EQ(q_scale.size(1), num_qo_heads);
  TVM_FFI_ICHECK_EQ(q_scale.size(2), seq_len_q);
  TVM_FFI_ICHECK_EQ(q_scale.size(3), head_dim / 32);
  TVM_FFI_ICHECK_EQ(k_scale.size(0), batch);
  TVM_FFI_ICHECK_EQ(k_scale.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(k_scale.size(2), seq_len_k);
  TVM_FFI_ICHECK_EQ(k_scale.size(3), head_dim / 32);
  TVM_FFI_ICHECK_EQ(v_scale_t.size(0), batch);
  TVM_FFI_ICHECK_EQ(v_scale_t.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(v_scale_t.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(v_scale_t.size(3), seq_len_k / 16);

  TVM_FFI_ICHECK_EQ(out.size(0), batch);
  TVM_FFI_ICHECK_EQ(out.size(1), num_qo_heads);
  TVM_FFI_ICHECK_EQ(out.size(2), seq_len_q);
  TVM_FFI_ICHECK_EQ(out.size(3), head_dim);
  if (maybe_lse.has_value()) {
    TVM_FFI_ICHECK_EQ(maybe_lse.value().size(0), batch);
    TVM_FFI_ICHECK_EQ(maybe_lse.value().size(1), num_qo_heads);
    TVM_FFI_ICHECK_EQ(maybe_lse.value().size(2), seq_len_q);
  }

  check_same_device(q_fp8, k_fp8, "k_fp8");
  check_same_device(q_fp8, v_fp4_t, "v_fp4_t");
  check_same_device(q_fp8, q_scale, "q_scale");
  check_same_device(q_fp8, k_scale, "k_scale");
  check_same_device(q_fp8, v_scale_t, "v_scale_t");
  check_same_device(q_fp8, out, "out");
  if (maybe_lse.has_value()) {
    check_same_device(q_fp8, maybe_lse.value(), "lse");
  }

  cudaStream_t stream = get_stream(q_fp8.device());

  if (seq_len_q == 0) {
    status = cudaMemsetAsync(out.data_ptr(), 0, numel(out) * get_element_size(out), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaMemsetAsync(out) failed: " << cudaGetErrorString(status);
    if (maybe_lse.has_value()) {
      auto lse = maybe_lse.value();
      status = cudaMemsetAsync(lse.data_ptr(), 0, numel(lse) * get_element_size(lse), stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "cudaMemsetAsync(lse) failed: " << cudaGetErrorString(status);
    }
    return;
  }

  if (maybe_lse.has_value()) {
    // LSE may be allocated uninitialized by the caller. Zero it first so any
    // entries not written by the kernel have a defined value.
    auto lse = maybe_lse.value();
    status = cudaMemsetAsync(lse.data_ptr(), 0, numel(lse) * get_element_size(lse), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaMemsetAsync(lse) failed: " << cudaGetErrorString(status);
  }

  FlashFwdParams params;
  set_params_fprop(params, q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t, out, maybe_lse,
                   static_cast<float>(sm_scale), causal, unpadded_q_len, unpadded_k_len);
  run_mha_fwd(params, stream);
}

}  // namespace qk_mxfp8_pv_nvfp4_attention_sm120
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fwd, flashinfer::qk_mxfp8_pv_nvfp4_attention_sm120::fwd);
