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
 *
 * TVM-FFI launcher for the VibeCUDA Mamba2/SSD combined selective scan.
 * All computation happens in the hand-written mma.sync kernels in
 * include/flashinfer/mamba/vibecuda_ssd_combined.cuh; out/state_in/final_states
 * are caller-owned so no device memory is allocated here.
 */
#include "flashinfer/mamba/vibecuda_ssd_combined.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

using flashinfer::mamba::vibecuda::bf16;
using flashinfer::mamba::vibecuda::fp16;
using flashinfer::mamba::vibecuda::LaunchVibeCudaSsdCombined;

void vibecuda_ssd_combined_fwd(TensorView x, TensorView dt, Optional<TensorView> dt_bias,
                               TensorView a, TensorView b, TensorView c, Optional<TensorView> d,
                               Optional<TensorView> z, Optional<TensorView> initial,
                               Optional<TensorView> seq_idx, TensorView state_in, TensorView out,
                               TensorView final_states, int64_t softplus, double dt_lo,
                               double dt_hi, int64_t d_has_hdim, int64_t varlen,
                               int64_t all_single_host) {
  CHECK_INPUT(x);
  CHECK_INPUT(dt);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(state_in);
  CHECK_INPUT(out);
  CHECK_INPUT(final_states);
  ffi::CUDADeviceGuard device_guard(x.device().device_id);
  CHECK_DIM(4, x);
  CHECK_DIM(4, b);
  CHECK_DIM(4, final_states);
  for (auto tensor : {dt, a, b, c, state_in, out, final_states}) {
    CHECK_DEVICE(x, tensor);
  }
  for (auto tensor : {dt_bias, d, z, initial, seq_idx}) {
    if (tensor.has_value()) {
      CHECK_DEVICE(x, tensor.value());
    }
  }
  TVM_FFI_ICHECK(final_states.dtype() == dl_bfloat16 || final_states.dtype() == dl_float16)
      << "final_states must be bfloat16 or float16";
  TVM_FFI_ICHECK_EQ(state_in.dtype(), dl_bfloat16) << "state_in must be bfloat16";
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16) << "out must be bfloat16";
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_float32) << "A must be float32";
  TVM_FFI_ICHECK_EQ(x.dtype(), dl_bfloat16) << "x must be bfloat16";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_bfloat16) << "b must be bfloat16";
  TVM_FFI_ICHECK_EQ(c.dtype(), dl_bfloat16) << "c must be bfloat16";

  const void* dt_bias_ptr = nullptr;
  if (dt_bias.has_value()) {
    CHECK_INPUT(dt_bias.value());
    TVM_FFI_ICHECK_EQ(dt_bias.value().dtype(), dt.dtype()) << "dt_bias dtype must match dt dtype";
    dt_bias_ptr = dt_bias.value().data_ptr();
  }
  const void* d_ptr = nullptr;
  if (d.has_value()) {
    CHECK_INPUT(d.value());
    TVM_FFI_ICHECK_EQ(d.value().dtype(), dl_bfloat16) << "d must be bfloat16";
    d_ptr = d.value().data_ptr();
  }
  const void* z_ptr = nullptr;
  int z_is_f16 = 0;
  if (z.has_value()) {
    CHECK_INPUT(z.value());
    TVM_FFI_ICHECK(z.value().dtype() == dl_bfloat16 || z.value().dtype() == dl_float16)
        << "z must be bfloat16 or float16";
    z_is_f16 = (z.value().dtype() == dl_float16) ? 1 : 0;
    z_ptr = z.value().data_ptr();
  }
  const void* initial_ptr = nullptr;
  if (initial.has_value()) {
    CHECK_INPUT(initial.value());
    TVM_FFI_ICHECK_EQ(initial.value().dtype(), final_states.dtype())
        << "initial_states dtype must match final_states dtype";
    initial_ptr = initial.value().data_ptr();
  }

  auto x_shape = x.shape();
  const int Bsz = static_cast<int>(x_shape[0]);
  const int L = static_cast<int>(x_shape[1]);
  const int H = static_cast<int>(x_shape[2]);
  const int G = static_cast<int>(b.shape()[2]);
  const int nseg = static_cast<int>(final_states.shape()[0]);
  TVM_FFI_ICHECK(Bsz > 0 && L > 0 && L % 128 == 0 && H > 0 && G > 0 && H % G == 0)
      << "invalid batch, sequence length, or head/group geometry";
  auto check_shape = [](TensorView tensor, std::initializer_list<int64_t> expected) {
    TVM_FFI_ICHECK_EQ(tensor.ndim(), expected.size()) << "invalid tensor rank";
    size_t dim = 0;
    for (auto size : expected) {
      TVM_FFI_ICHECK_EQ(tensor.shape()[dim], size) << "invalid tensor shape";
      ++dim;
    }
  };
  check_shape(x, {Bsz, L, H, 64});
  check_shape(dt, {Bsz, L, H});
  check_shape(a, {H});
  check_shape(b, {Bsz, L, G, 128});
  check_shape(c, {Bsz, L, G, 128});
  check_shape(out, {Bsz, H, 64, L / 128, 128});
  check_shape(final_states, {nseg, H, 64, 128});
  TVM_FFI_ICHECK(nseg > 0 && (varlen ? Bsz == 1 : nseg == Bsz));
  TVM_FFI_ICHECK_EQ(varlen != 0, seq_idx.has_value());
  TVM_FFI_ICHECK_EQ(all_single_host != 0, !varlen && L == 128);
  if (varlen) TVM_FFI_ICHECK(initial.has_value()) << "varlen requires initial_states";
  if (dt_bias.has_value()) check_shape(dt_bias.value(), {H});
  if (d.has_value()) {
    if (d_has_hdim)
      check_shape(d.value(), {H, 64});
    else
      check_shape(d.value(), {H});
  }
  if (z.has_value()) check_shape(z.value(), {Bsz, L, H, 64});
  if (initial.has_value()) check_shape(initial.value(), {nseg, H, 64, 128});
  if (seq_idx.has_value()) check_shape(seq_idx.value(), {Bsz, L});
  const int NT = Bsz * L;
  const int64_t nLCmax = static_cast<int64_t>(NT / 128) + nseg;
  if (!all_single_host) {
    TVM_FFI_ICHECK_GE(state_in.numel(), nLCmax * H * 64 * 128)
        << "state_in scratch too small: need " << nLCmax << " logical chunks, got "
        << state_in.numel() / (H * 64 * 128);
  }

  auto stream = get_stream(x.device());

#define VIBECUDA_SSD_DISPATCH(DT_TYPE_, IDX_TYPE_, ST_TYPE_)                                       \
  do {                                                                                             \
    cudaError_t status = LaunchVibeCudaSsdCombined<DT_TYPE_, IDX_TYPE_, ST_TYPE_>(                 \
        x.data_ptr(), dt.data_ptr(), dt_bias_ptr, a.data_ptr(), b.data_ptr(), c.data_ptr(), d_ptr, \
        z_ptr, z_is_f16, initial_ptr, static_cast<const IDX_TYPE_*>(sid_ptr), state_in.data_ptr(), \
        out.data_ptr(), final_states.data_ptr(), Bsz, L, H, G, nseg, (int)nLCmax, (int)softplus,   \
        dt_lo, dt_hi, (int)d_has_hdim, (int)varlen, all_single_host != 0, stream);                 \
    TVM_FFI_ICHECK(status == cudaSuccess)                                                          \
        << "vibecuda_ssd_combined_fwd failed: " << cudaGetErrorString(status);                     \
    return;                                                                                        \
  } while (0)

  const bool st_is_f16 = (final_states.dtype() == dl_float16);
  const void* sid_ptr = nullptr;
  int64_t sid_code = 0;
  if (seq_idx.has_value()) {
    CHECK_INPUT(seq_idx.value());
    TVM_FFI_ICHECK(seq_idx.value().dtype() == dl_int32 || seq_idx.value().dtype() == dl_int64)
        << "seq_idx must be int32 or int64";
    sid_ptr = seq_idx.value().data_ptr();
    sid_code = encode_dlpack_dtype(seq_idx.value().dtype());
  }
#define VIBECUDA_SSD_DISPATCH_DT(DT_TYPE_)              \
  do {                                                  \
    if (sid_code == int64_code) {                       \
      if (st_is_f16)                                    \
        VIBECUDA_SSD_DISPATCH(DT_TYPE_, int64_t, fp16); \
      else                                              \
        VIBECUDA_SSD_DISPATCH(DT_TYPE_, int64_t, bf16); \
    } else {                                            \
      if (st_is_f16)                                    \
        VIBECUDA_SSD_DISPATCH(DT_TYPE_, int32_t, fp16); \
      else                                              \
        VIBECUDA_SSD_DISPATCH(DT_TYPE_, int32_t, bf16); \
    }                                                   \
  } while (0)

  if (dt.dtype() == dl_float32) {
    VIBECUDA_SSD_DISPATCH_DT(float);
  } else if (dt.dtype() == dl_bfloat16) {
    VIBECUDA_SSD_DISPATCH_DT(bf16);
  } else if (dt.dtype() == dl_float16) {
    VIBECUDA_SSD_DISPATCH_DT(fp16);
  } else {
    TVM_FFI_ICHECK(false) << "unsupported dt dtype for vibecuda_ssd_combined_fwd";
    return;
  }
#undef VIBECUDA_SSD_DISPATCH_DT
#undef VIBECUDA_SSD_DISPATCH
}
