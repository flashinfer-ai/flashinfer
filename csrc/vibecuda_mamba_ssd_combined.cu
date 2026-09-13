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
 * include/flashinfer/mamba/vibecuda_ssd_combined.cuh; workspace/out/
 * final_states are caller-owned so no device memory is allocated here.
 */
#include "flashinfer/mamba/vibecuda_ssd_combined.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

using flashinfer::mamba::vibecuda::CS;
using flashinfer::mamba::vibecuda::LaunchVibeCudaSsdCombined;
using flashinfer::mamba::vibecuda::ND;
using flashinfer::mamba::vibecuda::PD;
using flashinfer::mamba::vibecuda::VibeCudaSsdArgs;

static int _vibecuda_sm_count() {
  static std::unordered_map<int, int> cache;
  int dev = 0;
  cudaGetDevice(&dev);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
  cache[dev] = v > 0 ? v : 148;
  return cache[dev];
}

void vibecuda_ssd_combined_fwd(TensorView x, TensorView dt, Optional<TensorView> dt_bias,
                               TensorView a, TensorView b, TensorView c, Optional<TensorView> d,
                               Optional<TensorView> z, Optional<TensorView> initial,
                               Optional<TensorView> seq_idx,
                               Optional<TensorView> chunk_indices,
                               Optional<TensorView> chunk_offsets,
                               Optional<TensorView> checkpoint_states,
                               Optional<TensorView> checkpoint_tokens,
                               Optional<TensorView> checkpoint_slots, TensorView workspace,
                               TensorView out, TensorView final_states, int64_t nchunk_bound,
                               int64_t do_softplus, double dt_lo, double dt_hi, int64_t unbounded,
                               int64_t d_mode, int64_t varlen, int64_t y_chunk_major) {
  CHECK_INPUT(x);
  CHECK_INPUT(dt);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(workspace);
  CHECK_INPUT(out);
  CHECK_INPUT(final_states);
  TVM_FFI_ICHECK_EQ(a.dtype(), dl_float32) << "A must be float32";
  TVM_FFI_ICHECK_EQ(x.dtype(), dl_bfloat16) << "x must be bfloat16";
  TVM_FFI_ICHECK_EQ(b.dtype(), dl_bfloat16) << "b must be bfloat16";
  TVM_FFI_ICHECK_EQ(c.dtype(), dl_bfloat16) << "c must be bfloat16";
  TVM_FFI_ICHECK(dt.dtype() == dl_float32 || dt.dtype() == dl_bfloat16)
      << "dt must be float32 or bfloat16";
  const bool dt_is32 = (dt.dtype() == dl_float32);

  const void* dt_bias_ptr = nullptr;
  if (dt_bias.has_value()) {
    CHECK_INPUT(dt_bias.value());
    TVM_FFI_ICHECK_EQ(dt_bias.value().dtype(), dt.dtype()) << "dt_bias dtype must match dt dtype";
    dt_bias_ptr = dt_bias.value().data_ptr();
  }
  const void* d_ptr = nullptr;
  int d_is32 = 0;
  if (d.has_value()) {
    CHECK_INPUT(d.value());
    TVM_FFI_ICHECK(d.value().dtype() == dl_bfloat16 || d.value().dtype() == dl_float32)
        << "d must be bfloat16 or float32";
    d_is32 = (d.value().dtype() == dl_float32) ? 1 : 0;
    d_ptr = d.value().data_ptr();
  }
  const void* z_ptr = nullptr;
  int has_z = 0;
  if (z.has_value()) {
    CHECK_INPUT(z.value());
    TVM_FFI_ICHECK_EQ(z.value().dtype(), dl_bfloat16) << "z must be bfloat16";
    has_z = 1;
    z_ptr = z.value().data_ptr();
  }
  const void* initial_ptr = nullptr;
  if (initial.has_value()) {
    CHECK_INPUT(initial.value());
    TVM_FFI_ICHECK_EQ(initial.value().dtype(), final_states.dtype())
        << "initial_states dtype must match final_states dtype";
    initial_ptr = initial.value().data_ptr();
  }
  TVM_FFI_ICHECK(final_states.dtype() == dl_bfloat16 || final_states.dtype() == dl_float16)
      << "final_states dtype must be bfloat16 or float16";

  const void* sid_ptr = nullptr;
  int si_is64 = 0;
  int64_t tseq = 0;
  if (seq_idx.has_value()) {
    CHECK_INPUT(seq_idx.value());
    sid_ptr = seq_idx.value().data_ptr();
    si_is64 = (encode_dlpack_dtype(seq_idx.value().dtype()) == int64_code) ? 1 : 0;
    tseq = static_cast<int64_t>(seq_idx.value().numel());
  }

  // mamba2_metadata part geometry: (physical_chunk, in-chunk offset) of every
  // part start. When provided, the kernel tiles at exactly these boundaries
  // (physical 128-chunk starts, sequence starts, and serving checkpoint
  // splits), which keeps its quantization ladder bitwise-tracked to the CAKE
  // oracle. Both int32 and the same length.
  const void* meta_ci = nullptr;
  const void* meta_co = nullptr;
  int64_t nmeta = 0;
  if (chunk_indices.has_value() && chunk_offsets.has_value()) {
    CHECK_INPUT(chunk_indices.value());
    CHECK_INPUT(chunk_offsets.value());
    TVM_FFI_ICHECK_EQ(chunk_indices.value().dtype(), dl_int32)
        << "chunk_indices must be int32";
    TVM_FFI_ICHECK_EQ(chunk_offsets.value().dtype(), dl_int32)
        << "chunk_offsets must be int32";
    TVM_FFI_ICHECK_EQ(chunk_indices.value().numel(), chunk_offsets.value().numel())
        << "chunk_indices/chunk_offsets length mismatch";
    nmeta = static_cast<int64_t>(chunk_indices.value().numel());
    if (nmeta > 0) {
      meta_ci = chunk_indices.value().data_ptr();
      meta_co = chunk_offsets.value().data_ptr();
    }
  }

  auto x_shape = x.shape();
  const int Bsz = static_cast<int>(x_shape[0]);
  const int L = static_cast<int>(x_shape[1]);
  const int H = static_cast<int>(x_shape[2]);
  const int G = static_cast<int>(b.shape()[2]);
  const int nseq = static_cast<int>(final_states.shape()[0]);

  // Selective checkpoint capture (CAKE contract): one exclusive token boundary
  // per sequence; the kernel writes the part-start state at that boundary into
  // row checkpoint_slots[s] of checkpoint_states. All three or none.
  void* ck_ptr = nullptr;
  const void* ck_tok = nullptr;
  const void* ck_slt = nullptr;
  const int n_ck_args = (checkpoint_states.has_value() ? 1 : 0) +
                        (checkpoint_tokens.has_value() ? 1 : 0) +
                        (checkpoint_slots.has_value() ? 1 : 0);
  TVM_FFI_ICHECK(n_ck_args == 0 || n_ck_args == 3)
      << "checkpoint_states/checkpoint_tokens/checkpoint_slots must be provided together";
  if (n_ck_args == 3) {
    CHECK_INPUT(checkpoint_states.value());
    CHECK_INPUT(checkpoint_tokens.value());
    CHECK_INPUT(checkpoint_slots.value());
    TVM_FFI_ICHECK_EQ(checkpoint_tokens.value().dtype(), dl_int32)
        << "checkpoint_tokens must be int32";
    TVM_FFI_ICHECK_EQ(checkpoint_slots.value().dtype(), dl_int32)
        << "checkpoint_slots must be int32";
    TVM_FFI_ICHECK_EQ(checkpoint_states.value().dtype(), final_states.dtype())
        << "checkpoint_states dtype must match the state dtype";
    TVM_FFI_ICHECK_EQ(checkpoint_tokens.value().numel(), nseq)
        << "checkpoint_tokens must have one entry per sequence";
    TVM_FFI_ICHECK_EQ(checkpoint_slots.value().numel(), nseq)
        << "checkpoint_slots must have one entry per sequence";
    TVM_FFI_ICHECK_EQ(checkpoint_states.value().shape().size(), 4)
        << "checkpoint_states must be [num_checkpoints, nheads, headdim, dstate]";
    ck_ptr = checkpoint_states.value().data_ptr();
    ck_tok = checkpoint_tokens.value().data_ptr();
    ck_slt = checkpoint_slots.value().data_ptr();
  }

  // workspace: chunk_state (+da_last) fp32 for the uniform multi-chunk layout
  const int cps = (L + CS - 1) / CS;
  const bool need_gs = !varlen && cps > 1;
  const int64_t cs_floats = need_gs ? nchunk_bound * H * (int64_t)(PD * ND) : 0;
  const int64_t dal_floats = need_gs ? nchunk_bound * H : 0;
  TVM_FFI_ICHECK_GE(workspace.numel(), cs_floats + dal_floats + 64)
      << "workspace too small: need " << (cs_floats + dal_floats + 64) << " floats, got "
      << workspace.numel();
  TVM_FFI_ICHECK_EQ(workspace.dtype(), dl_float32) << "workspace must be float32";

  VibeCudaSsdArgs args;
  args.x = x.data_ptr();
  args.b = b.data_ptr();
  args.c = c.data_ptr();
  args.z = z_ptr;
  args.d = d_ptr;
  args.dt = dt.data_ptr();
  args.dt_bias = dt_bias_ptr;
  args.a = static_cast<const float*>(a.data_ptr());
  args.initial = initial_ptr;
  args.seq_idx = sid_ptr;
  args.meta_ci = reinterpret_cast<const int*>(meta_ci);
  args.meta_co = reinterpret_cast<const int*>(meta_co);
  args.nmeta = static_cast<int>(nmeta);
  args.ck_states = ck_ptr;
  args.ck_tokens = reinterpret_cast<const int*>(ck_tok);
  args.ck_slots = reinterpret_cast<const int*>(ck_slt);
  args.workspace = static_cast<float*>(workspace.data_ptr());
  args.y = out.data_ptr();
  args.final_states = final_states.data_ptr();
  args.dt_is32 = dt_is32 ? 1 : 0;
  args.st_is_f16 = (final_states.dtype() == dl_float16) ? 1 : 0;
  args.si_is64 = si_is64;
  args.d_mode = static_cast<int>(d_mode);
  args.d_is32 = d_is32;
  args.Bsz = Bsz;
  args.L = L;
  args.heads = H;
  args.groups = G;
  args.nseq = nseq;
  args.nchunk_bound = static_cast<int>(nchunk_bound);
  args.Tseq = varlen ? static_cast<int>(tseq) : 0;
  args.has_z = has_z;
  args.do_softplus = static_cast<int>(do_softplus);
  args.unbounded = static_cast<int>(unbounded);
  args.dt_lo = static_cast<float>(dt_lo);
  args.dt_hi = static_cast<float>(dt_hi);
  args.varlen = static_cast<int>(varlen);
  args.y_chunk_major = static_cast<int>(y_chunk_major);
  args.sm_count = _vibecuda_sm_count();
  args.stream = get_stream(x.device());

  cudaError_t status = LaunchVibeCudaSsdCombined(args);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "vibecuda_ssd_combined_fwd failed: " << cudaGetErrorString(status);
}
