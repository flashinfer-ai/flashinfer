/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "cake_kda_affine_binding_common.cuh"

#if FLASHINFER_CAKE_KDA_AFFINE_ROLE != \
        FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN && \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE != \
        FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAP && \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE != \
        FLASHINFER_CAKE_KDA_AFFINE_ROLE_CORRECTION
#error "Cake KDA affine direct binding requires main, map, or correction role"
#endif

static_assert(FLASHINFER_CAKE_KDA_AFFINE_THREADS == 1024,
              "sealed Cake KDA affine direct roles use 1024 threads");
static_assert(FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES == 227968,
              "sealed Cake KDA affine direct roles use 227968 bytes of shared memory");
#if FLASHINFER_CAKE_KDA_AFFINE_ROLE == \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN
static_assert(FLASHINFER_CAKE_KDA_AFFINE_USE_PDL == 0,
              "sealed Cake KDA affine main launch does not use PDL");
#else
static_assert(FLASHINFER_CAKE_KDA_AFFINE_USE_PDL == 1,
              "sealed Cake KDA affine map and correction launches use PDL");
#endif

namespace flashinfer {
namespace cake_kda {

struct CakeKDAAffineDirectPreparedInputs {
  int32_t device_id;
  int64_t num_sequences;
  cudaStream_t stream;
  TmaPointers tma;
};

inline CakeKDAAffineDirectPreparedInputs
CakeKDAAffinePrepareDirectInputs(
    const TensorView& q, const TensorView& k, const TensorView& v,
    const TensorView& g, const TensorView& beta,
    const TensorView& beta_tma, const TensorView& A_log,
    const TensorView& dt_bias, const TensorView& cu_seqlens,
    const TensorView& seq_order, const TensorView& out,
    const TensorView& descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride, double scale,
    double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA)
      << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckCakeKDATarget(device_id);
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be zero or one";
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= 32)
      << "Cake KDA affine direct roles require 1 <= num_heads <= 32";
  TVM_FFI_ICHECK(std::isfinite(scale) &&
                 std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32";
  TVM_FFI_ICHECK(lower_bound == 0.0)
      << "Cake KDA unbounded-softplus affine roles require lower_bound=0";

  for (const auto& named :
       {std::pair<const TensorView*, const char*>{&q, "q"},
        {&k, "k"}, {&v, "v"}, {&g, "g"}, {&out, "out"}}) {
    CheckCudaTensor(*named.first, named.second, device_id);
    CheckDtype(*named.first, named.second, dl_bfloat16);
  }
  CheckCudaTensor(beta, "beta", device_id);
  CheckDtype(beta, "beta", dl_bfloat16);
  CheckCudaTensor(beta_tma, "beta_tma", device_id);
  CheckDtype(beta_tma, "beta_tma", dl_bfloat16);
  CheckCudaTensor(A_log, "A_log", device_id);
  CheckDtype(A_log, "A_log", dl_float32);
  CheckCudaTensor(dt_bias, "dt_bias", device_id);
  CheckDtype(dt_bias, "dt_bias", dl_float32);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckCudaTensor(seq_order, "seq_order", device_id);
  CheckDtype(seq_order, "seq_order", dl_int32);
  CheckCudaTensor(descriptor_storage, "descriptor_storage", device_id);
  CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8);

  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(0) == 1 &&
                 q.size(2) == num_heads && q.size(3) == kHeadDim)
      << "q must have shape [1, tokens, H, 128]";
  const int64_t token_count = q.size(1);
  TVM_FFI_ICHECK(token_count > 0 && token_count % 32 == 0)
      << "Cake KDA affine direct roles require a positive multiple of 32 tokens";
  for (const auto& named :
       {std::pair<const TensorView*, const char*>{&k, "k"},
        {&v, "v"}, {&g, "g"}, {&out, "out"}}) {
    const TensorView& tensor = *named.first;
    TVM_FFI_ICHECK(tensor.ndim() == 4 && tensor.size(0) == 1 &&
                   tensor.size(1) == token_count &&
                   tensor.size(2) == num_heads &&
                   tensor.size(3) == kHeadDim)
        << named.second << " must match q's [1, tokens, H, 128] shape";
  }
  TVM_FFI_ICHECK(beta.ndim() == 3 && beta.size(0) == 1 &&
                 beta.size(1) == token_count &&
                 beta.size(2) == num_heads)
      << "beta must have shape [1, tokens, H]";
  TVM_FFI_ICHECK(beta_token_stride == beta.stride(1))
      << "beta_token_stride must match beta's physical token stride";
  TVM_FFI_ICHECK(beta_tma.ndim() >= 2)
      << "beta_tma must have at least two dimensions";
  const int64_t padded_heads = RoundUpBetaTmaHeads(num_heads);
  const int64_t beta_tma_heads = beta_tma.size(beta_tma.ndim() - 1);
  TVM_FFI_ICHECK(beta_tma_heads > 0)
      << "beta_tma must have a positive head dimension";
  const bool direct_beta = beta_tma_heads == num_heads && num_heads >= 8;
  TVM_FFI_ICHECK(
      beta_tma.ndim() >= 2 &&
      (direct_beta || beta_tma_heads == padded_heads) &&
      beta_tma.numel() / beta_tma_heads >= token_count)
      << "beta_tma must provide direct aligned rows or round_up(H, 8) padded rows";
  TVM_FFI_ICHECK(beta_tma.data_ptr() != nullptr &&
                 reinterpret_cast<uintptr_t>(beta_tma.data_ptr()) % 16 == 0 &&
                 beta_tma.stride(beta_tma.ndim() - 1) == 1 &&
                 beta_tma.stride(beta_tma.ndim() - 2) *
                         sizeof(__nv_bfloat16) %
                     16 ==
                     0)
      << "beta_tma must satisfy the sealed 16-byte TMA alignment";
  TVM_FFI_ICHECK(A_log.numel() == num_heads)
      << "A_log must contain H elements";
  TVM_FFI_ICHECK(dt_bias.numel() == num_heads * kHeadDim)
      << "dt_bias must contain H * 128 elements";
  constexpr int64_t kMinimumCuSeqlens =
#if FLASHINFER_CAKE_KDA_AFFINE_ROLE == \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN
      3;
#else
      2;
#endif
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 &&
                 cu_seqlens.numel() >= kMinimumCuSeqlens)
      << "affine direct role has too few token parts";
  const int64_t num_sequences = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(seq_order.ndim() == 1 &&
                 seq_order.numel() == num_sequences)
      << "seq_order must contain one entry per affine part";
  TVM_FFI_ICHECK(
      descriptor_storage.numel() >= static_cast<int64_t>(kDescriptorStorageBytes) &&
      reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) %
              kTensorMapAlignment ==
          0)
      << "descriptor_storage must hold six aligned TensorMaps";

  CheckNoPartialOverlapOrExactAlias(beta, "beta", beta_tma, "beta_tma");
  CheckNoOverlap(out, "out", q, "q");
  CheckNoOverlap(out, "out", k, "k");
  CheckNoOverlap(out, "out", v, "v");
  CheckNoOverlap(out, "out", g, "g");
  CheckNoOverlap(out, "out", beta, "beta");
  CheckNoOverlap(out, "out", beta_tma, "beta_tma");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", out, "out");

  const cudaStream_t stream = CakeKDAAffineCheckedStream(cuda_stream);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride,
                         stream);
  const TmaPointers tma = EncodeTmaPointers<128, 32>(
      q, k, v, g, beta_tma, out, descriptor_storage,
      prepare_descriptors, stream);
  return {device_id, num_sequences, stream, tma};
}

struct CakeKDAAffineDirectKernelArgs {
  void* q{};
  void* q_tma{};
  void* k{};
  void* k_tma{};
  void* v{};
  void* v_tma{};
  void* g{};
  void* g_tma{};
  void* beta{};
  void* beta_tma{};
  void* A_log{};
  void* dt_bias{};
  void* cu_seqlens{};
  void* seq_order{};
  void* initial_state{};
  void* out{};
  void* out_tma{};
  void* final_state{};
  int32_t num_heads{};
  int32_t use_initial_state{1};
  int32_t store_final_state{1};
  float scale{};
  float lower_bound{};
  uint64_t state_indices_addr{};
  uint64_t state_checkpoints_addr{};
  uint64_t checkpoint_cu_starts_addr{};
  int64_t beta_token_stride{};
  int64_t state_slot_stride{};
  int32_t use_state_indices{};
  int32_t checkpoint_every_n_tokens{};
  void* cu_chunk_offsets{};
  void* chunk_state{};
  void* state_checkpoint_needed{};
  void* tape_qd{};
  void* tape_kd{};
  void* tape_kr{};
  void* tape_j{};
  void* tape_restore_factor{};
  void* tape_e{};
  void* tape_x{};
  void* tape_r{};
  void* norm_inv_out{};
  void* decay_out{};
  void* beta_active_out{};
  void* initial_state_f32{};
  void* zero_workspace{};
  int32_t zero_words{};
  int32_t num_sequences{};
  void* state_checkpoints_tma{};
  void* final_state_f32{};
};

inline void CakeKDAAffineLaunchDirectM128(
    CakeKDAAffineDirectKernelArgs& args, dim3 grid, int32_t device_id,
    cudaStream_t stream) {
  void* kernel_args[] = {
      &args.q,
      &args.q_tma,
      &args.k,
      &args.k_tma,
      &args.v,
      &args.v_tma,
      &args.g,
      &args.g_tma,
      &args.beta,
      &args.beta_tma,
      &args.A_log,
      &args.dt_bias,
      &args.cu_seqlens,
      &args.seq_order,
      &args.initial_state,
      &args.out,
      &args.out_tma,
      &args.final_state,
      &args.num_heads,
      &args.use_initial_state,
      &args.store_final_state,
      &args.scale,
      &args.lower_bound,
      &args.state_indices_addr,
      &args.state_checkpoints_addr,
      &args.checkpoint_cu_starts_addr,
      &args.beta_token_stride,
      &args.state_slot_stride,
      &args.use_state_indices,
      &args.checkpoint_every_n_tokens,
      &args.cu_chunk_offsets,
      &args.chunk_state,
      &args.state_checkpoint_needed,
      &args.tape_qd,
      &args.tape_kd,
      &args.tape_kr,
      &args.tape_j,
      &args.tape_restore_factor,
      &args.tape_e,
      &args.tape_x,
      &args.tape_r,
      &args.norm_inv_out,
      &args.decay_out,
      &args.beta_active_out,
      &args.initial_state_f32,
      &args.zero_workspace,
      &args.zero_words,
      &args.num_sequences,
      &args.state_checkpoints_tma,
      &args.final_state_f32,
  };
  CakeKDAAffineCheckArgumentCount<50>(kernel_args);
  CakeKDAAffineConfigureAndLaunch(
      reinterpret_cast<const void*>(FLASHINFER_CAKE_KDA_AFFINE_KERNEL),
      grid, device_id, stream, kernel_args,
      "Cake KDA affine direct-M128 launch");
}

inline void RunCakeKDAAffineDirectM128(
    TensorView q, TensorView k, TensorView v, TensorView g,
    TensorView beta, TensorView beta_tma, TensorView A_log,
    TensorView dt_bias, TensorView cu_seqlens, TensorView seq_order,
    TensorView state_indices, TensorView initial_state_bf16,
    TensorView out, TensorView final_state_bf16,
    TensorView initial_state_f32, TensorView final_state_f32,
    TensorView descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride,
    int64_t state_slot_stride, double scale, double lower_bound,
    int64_t grid_x, int64_t grid_y, int64_t grid_z,
    int64_t cuda_stream) {
  const auto prepared = CakeKDAAffinePrepareDirectInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens,
      seq_order, out, descriptor_storage, prepare_descriptors, num_heads,
      beta_token_stride, scale, lower_bound, cuda_stream);
  ffi::CUDADeviceGuard device_guard(prepared.device_id);
  const int64_t compact_state_stride = num_heads * kHeadDim * kHeadDim;
  TVM_FFI_ICHECK(grid_x == prepared.num_sequences * num_heads &&
                 grid_y == 1 && grid_z == 1)
      << "Cake KDA affine direct grid must be [num_sequences * H, 1, 1]";

  CakeKDAAffineDirectKernelArgs args{};
  args.q = q.data_ptr();
  args.q_tma = prepared.tma.q;
  args.k = k.data_ptr();
  args.k_tma = prepared.tma.k;
  args.v = v.data_ptr();
  args.v_tma = prepared.tma.v;
  args.g = g.data_ptr();
  args.g_tma = prepared.tma.g;
  args.beta = beta.data_ptr();
  args.beta_tma = prepared.tma.beta;
  args.A_log = A_log.data_ptr();
  args.dt_bias = dt_bias.data_ptr();
  args.cu_seqlens = cu_seqlens.data_ptr();
  args.seq_order = seq_order.data_ptr();
  args.out = out.data_ptr();
  args.out_tma = prepared.tma.out;
  args.num_heads = CakeKDAAffineCheckedInt32(num_heads, "num_heads");
  args.scale = static_cast<float>(scale);
  args.lower_bound = 0.0f;
  args.beta_token_stride = beta_token_stride;
  args.state_slot_stride = state_slot_stride;
  args.num_sequences = CakeKDAAffineCheckedInt32(
      prepared.num_sequences, "num_sequences");
  // The sealed ABI retains a checkpoint TensorMap slot even though the affine
  // contract disables checkpoints. Supply one valid, unused descriptor.
  args.state_checkpoints_tma = prepared.tma.q;

  const auto check_no_data_overlap = [&](const TensorView& state,
                                         const char* state_name) {
    for (const auto& named :
         {std::pair<const TensorView*, const char*>{&q, "q"},
          {&k, "k"},
          {&v, "v"},
          {&g, "g"},
          {&beta, "beta"},
          {&beta_tma, "beta_tma"},
          {&A_log, "A_log"},
          {&dt_bias, "dt_bias"},
          {&cu_seqlens, "cu_seqlens"},
          {&seq_order, "seq_order"},
          {&out, "out"},
          {&descriptor_storage, "descriptor_storage"}}) {
      CheckNoOverlap(state, state_name, *named.first, named.second);
    }
  };

#if FLASHINFER_CAKE_KDA_AFFINE_ROLE == \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN
  CheckCudaTensor(state_indices, "state_indices", prepared.device_id);
  CheckDtype(state_indices, "state_indices", dl_int32);
  TVM_FFI_ICHECK(state_indices.ndim() == 1 && state_indices.numel() == 1)
      << "Cake KDA affine main requires exactly one external state index";
  CakeKDAAffineCheckBFloat16StatePool(
      initial_state_bf16, "initial_state_bf16", prepared.device_id,
      num_heads, state_slot_stride);
  CakeKDAAffineCheckInactiveTensor(final_state_bf16,
                                   "final_state_bf16",
                                   prepared.device_id, dl_bfloat16);
  CakeKDAAffineCheckInactiveTensor(initial_state_f32,
                                   "initial_state_f32",
                                   prepared.device_id, dl_float32);
  CakeKDAAffineCheckCompactState(
      final_state_f32, "final_state_f32", prepared.device_id, dl_float32,
      prepared.num_sequences, num_heads);
  check_no_data_overlap(state_indices, "state_indices");
  check_no_data_overlap(initial_state_bf16, "initial_state_bf16");
  check_no_data_overlap(final_state_f32, "final_state_f32");
  CheckNoOverlap(initial_state_bf16, "initial_state_bf16", final_state_f32,
                 "final_state_f32");
  args.initial_state = initial_state_bf16.data_ptr();
  args.final_state_f32 = final_state_f32.data_ptr();
  args.state_indices_addr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state_indices.data_ptr()));
  args.use_state_indices = 1;
#elif FLASHINFER_CAKE_KDA_AFFINE_ROLE == \
    FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAP
  CakeKDAAffineCheckInactiveTensor(state_indices, "state_indices",
                                   prepared.device_id, dl_int32);
  TVM_FFI_ICHECK(state_slot_stride == compact_state_stride)
      << "Cake KDA affine map requires compact state_slot_stride";
  CakeKDAAffineCheckCompactState(
      initial_state_bf16, "initial_state_bf16", prepared.device_id,
      dl_bfloat16, prepared.num_sequences, num_heads);
  CakeKDAAffineCheckCompactState(
      final_state_bf16, "final_state_bf16", prepared.device_id,
      dl_bfloat16, prepared.num_sequences, num_heads);
  CakeKDAAffineCheckCompactState(
      initial_state_f32, "initial_state_f32", prepared.device_id,
      dl_float32, prepared.num_sequences + 1, num_heads);
  CakeKDAAffineCheckInactiveTensor(final_state_f32,
                                   "final_state_f32",
                                   prepared.device_id, dl_float32);
  check_no_data_overlap(initial_state_bf16, "initial_state_bf16");
  check_no_data_overlap(final_state_bf16, "final_state_bf16");
  check_no_data_overlap(initial_state_f32, "initial_state_f32");
  CheckNoOverlap(initial_state_bf16, "initial_state_bf16",
                 final_state_bf16, "final_state_bf16");
  CheckNoOverlap(initial_state_bf16, "initial_state_bf16",
                 initial_state_f32, "initial_state_f32");
  CheckNoOverlap(final_state_bf16, "final_state_bf16", initial_state_f32,
                 "initial_state_f32");
  args.initial_state = initial_state_bf16.data_ptr();
  args.final_state = final_state_bf16.data_ptr();
  args.initial_state_f32 = initial_state_f32.data_ptr();
#else
  CakeKDAAffineCheckInactiveTensor(state_indices, "state_indices",
                                   prepared.device_id, dl_int32);
  TVM_FFI_ICHECK(state_slot_stride == compact_state_stride)
      << "Cake KDA affine correction requires compact state_slot_stride";
  CakeKDAAffineCheckInactiveTensor(initial_state_bf16,
                                   "initial_state_bf16",
                                   prepared.device_id, dl_bfloat16);
  CakeKDAAffineCheckInactiveTensor(final_state_bf16,
                                   "final_state_bf16",
                                   prepared.device_id, dl_bfloat16);
  CakeKDAAffineCheckCompactState(
      initial_state_f32, "initial_state_f32", prepared.device_id,
      dl_float32, prepared.num_sequences, num_heads);
  CakeKDAAffineCheckCompactState(
      final_state_f32, "final_state_f32", prepared.device_id, dl_float32,
      prepared.num_sequences, num_heads);
  check_no_data_overlap(initial_state_f32, "initial_state_f32");
  check_no_data_overlap(final_state_f32, "final_state_f32");
  CheckNoOverlap(initial_state_f32, "initial_state_f32", final_state_f32,
                 "final_state_f32");
  args.initial_state_f32 = initial_state_f32.data_ptr();
  args.final_state_f32 = final_state_f32.data_ptr();
#endif

  CakeKDAAffineLaunchDirectM128(
      args, CakeKDAAffineCheckedGrid(grid_x, grid_y, grid_z),
      prepared.device_id, prepared.stream);
}

}  // namespace cake_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run, flashinfer::cake_kda::RunCakeKDAAffineDirectM128);
