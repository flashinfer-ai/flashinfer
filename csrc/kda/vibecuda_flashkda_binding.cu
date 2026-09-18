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

// VibeCUDA FlashKDA prefill TVM-FFI binding TU.
//
// Exposes the VibeCUDA recurrent-KDA prefill entry points used by
// flashinfer.kda_vibecuda (backend="vibecuda"): the direct M128 route (with
// the slab/union regime dispatch), the M64 two-CTA route, the device-planned
// persistent M128 route, the split-seq affine-prefix pipeline for small-BH
// ultra-long fixed layouts, and the stable descending-length sequence order
// helper. Validation stays close to the frozen-path checks but on the
// raw-pointer VibeCUDA launcher ABI; tensor-level layout validation that
// stays stable across variants lives here so each launcher TU only resolves
// kernel pointers.

#include <algorithm>
#include <cstdint>

#include "flashkda_binding_common.cuh"
#include "vibecuda_flashkda_tma.cuh"

// Raw-pointer launchers, one per physical kernel TU.
namespace kda_flash {
void RunM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
             const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
             const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
             int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
             int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream);
void RunM128SplitFull(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                      const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                      const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                      const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                      void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                      int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                      int64_t use_initial_state, double scale, double lower_bound,
                      int64_t beta_tma_rows, int64_t beta_tma_dim1, int64_t num_parts,
                      void* split_state_ptr, void* map_state_ptr, void* carry_ptr,
                      void* split_out_ptr, void* lookback_flags_ptr, void* map_state_bf16_ptr,
                      int64_t ft_slab, int64_t cuda_stream);
void RunM64(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
            const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
            const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
            const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
            void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
            int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
            int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
            int64_t beta_tma_dim1, int64_t cuda_stream);
void RunPersistentM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                       const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                       const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                       const void* seq_order_ptr, void* task_ids_ptr, void* task_offsets_ptr,
                       void* choice_scratch_ptr, const void* initial_state_ptr, void* out_ptr,
                       void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                       int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                       int64_t use_initial_state, int64_t store_final_state, double scale,
                       double lower_bound, int64_t beta_tma_rows, int64_t beta_tma_dim1,
                       int64_t sm_count, int64_t cuda_stream);
}  // namespace kda_flash

namespace kda_flash_slab {
void RunM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
             const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
             const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
             int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
             int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream);
void RunM128SplitFull(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                      const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                      const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                      const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                      void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                      int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                      int64_t use_initial_state, double scale, double lower_bound,
                      int64_t beta_tma_rows, int64_t beta_tma_dim1, int64_t num_parts,
                      void* split_state_ptr, void* map_state_ptr, void* carry_ptr,
                      void* split_out_ptr, void* lookback_flags_ptr, void* map_state_bf16_ptr,
                      int64_t ft_slab, int64_t cuda_stream);
}  // namespace kda_flash_slab

// Head-family launchers: compile-time-H single-kernel per-(sequence, head)
// recurrence built on the evolved slab schedule (see
// vibecuda_flashkda_fam2_binding_h{12,64,96}.cu). RunFam2H12 consumes beta
// through its compile-time direct-read path (the H12 head pitch is not a TMA
// box multiple); the H64/H96 launchers read beta through the TMA box like the
// M128 route. ft_slab is pinned 1 for H12 (slab-only image); the H64/H96
// launchers receive the M128 regime flag.
namespace kda_fam2 {
void RunFam2H12(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
                const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
                void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
                int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
                int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream);
void RunFam2H64(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
                const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
                void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
                int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
                int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream);
void RunFam2H96(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
                const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
                void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
                int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
                int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream);
}  // namespace kda_fam2

// Fused N16 (16-token chunk) launchers for the H12 short/packed classes: the
// prepare kernel factors raw q/k/g/beta into the chain kernels' operands, and
// one of three recurrence-chain schedules (s7/s8/s9, selected on device fill)
// walks the chain. Both launch in a single host call (RunBt16Fused below).
namespace kda_bt16_prepare_fused {
void Run(const void* q_ptr, const void* k_ptr, const void* g_ptr, const void* beta_ptr,
         const void* a_log_ptr, const void* dt_bias_ptr, const void* cu_seqlens_ptr,
         const void* cu_chunks_ptr, const void* chunk_to_seq_ptr, void* ws_qd_ptr, void* ws_kd_ptr,
         void* ws_w_ptr, void* ws_qk_ptr, void* ws_diag_ptr, void* descriptor_storage,
         int64_t prepare_descriptors, int64_t tokens, int64_t total_chunks, int64_t num_heads,
         double gate_lower_bound, int64_t prepare_total_ctas, int64_t cuda_stream,
         int64_t num_seqs);
}  // namespace kda_bt16_prepare_fused
namespace kda_bt16_chain_fn16_s7 {
void RunFull(const void* ws_qd_ptr, const void* ws_kd_ptr, const void* ws_w_ptr,
             const void* ws_qk_ptr, const void* ws_diag_ptr, const void* v_ptr,
             const void* cu_seqlens_ptr, const void* cu_chunks_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage, int64_t prepare_descriptors, int64_t tokens,
             int64_t total_chunks, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream,
             int64_t num_seqs);
}  // namespace kda_bt16_chain_fn16_s7
namespace kda_bt16_chain_fn16_s8 {
void RunFull(const void* ws_qd_ptr, const void* ws_kd_ptr, const void* ws_w_ptr,
             const void* ws_qk_ptr, const void* ws_diag_ptr, const void* v_ptr,
             const void* cu_seqlens_ptr, const void* cu_chunks_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage, int64_t prepare_descriptors, int64_t tokens,
             int64_t total_chunks, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream,
             int64_t num_seqs);
}  // namespace kda_bt16_chain_fn16_s8
namespace kda_bt16_chain_fn16_s9 {
void RunFull(const void* ws_qd_ptr, const void* ws_kd_ptr, const void* ws_w_ptr,
             const void* ws_qk_ptr, const void* ws_diag_ptr, const void* v_ptr,
             const void* cu_seqlens_ptr, const void* cu_chunks_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage, int64_t prepare_descriptors, int64_t tokens,
             int64_t total_chunks, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream,
             int64_t num_seqs);
}  // namespace kda_bt16_chain_fn16_s9

namespace flashinfer {
namespace vibecuda_flashkda {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckDtype;
using flash_kda::CheckFlashKDATarget;
using flash_kda::CheckNoOverlap;

constexpr int64_t kDescriptorStorageBytes = kda_flash::kDescriptorStorageBytes;

// The VibeCUDA M128 prefill carries two physical kernel images: the union
// runtime-regime image and the compile-time slab specialization (the
// combined N=160 UMMA-4 issue replaced by 4a+4b, which shortens tensor-pipe
// residency per chunk on latency-bound shapes but costs ~7% on
// throughput-bound H=1 ultra-long chains). Regime rule: slab on for
// token_count <= 8192 (all short/mid single-seq and packed workloads) or
// tokens <= 65536 with heads >= 4 (small-BH long); off for H=1 ultra-long
// split chains.
inline int64_t VibeCUDASlabRegime(int64_t token_count, int64_t num_heads) {
  return (token_count <= 8192 || (token_count <= 65536 && num_heads >= 4)) ? 1 : 0;
}

// Per-call tensor validation shared by every VibeCUDA prefill entry point.
// Returns the packed token count; callers still dispatch on their own grid
// rules.
int64_t CheckVibeCUDAPrefillInputs(
    const TensorView& q, const TensorView& k, const TensorView& v, const TensorView& g,
    const TensorView& beta, const TensorView& beta_tma, const TensorView& A_log,
    const TensorView& dt_bias, const TensorView& cu_seqlens, const TensorView& seq_order,
    const TensorView& initial_state, const TensorView& out, const TensorView& final_state,
    const TensorView& descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
    int64_t use_initial_state, int64_t store_final_state, double scale, double lower_bound) {
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1, got " << prepare_descriptors;
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range, got " << num_heads;
  TVM_FFI_ICHECK(use_initial_state == 0 || use_initial_state == 1)
      << "use_initial_state must be 0 or 1, got " << use_initial_state;
  TVM_FFI_ICHECK(store_final_state == 0 || store_final_state == 1)
      << "store_final_state must be 0 or 1, got " << store_final_state;
  TVM_FFI_ICHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32, got " << scale;
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && lower_bound < 0.0 &&
                 std::isfinite(static_cast<float>(lower_bound)))
      << "lower_bound must be finite, negative, and representable as float32, got " << lower_bound;

  const int32_t device_id = q.device().device_id;
  CheckCudaTensor(q, "q", device_id);
  CheckCudaTensor(k, "k", device_id);
  CheckCudaTensor(v, "v", device_id);
  CheckCudaTensor(g, "g", device_id);
  CheckCudaTensor(beta, "beta", device_id);
  CheckCudaTensor(beta_tma, "beta_tma", device_id);
  CheckCudaTensor(A_log, "A_log", device_id);
  CheckCudaTensor(dt_bias, "dt_bias", device_id);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckCudaTensor(seq_order, "seq_order", device_id);
  CheckCudaTensor(initial_state, "initial_state", device_id);
  CheckCudaTensor(out, "out", device_id);
  CheckCudaTensor(final_state, "final_state", device_id);
  CheckCudaTensor(descriptor_storage, "descriptor_storage", device_id);

  CheckDtype(q, "q", dl_bfloat16);
  CheckDtype(k, "k", dl_bfloat16);
  CheckDtype(v, "v", dl_bfloat16);
  CheckDtype(g, "g", dl_bfloat16);
  CheckDtype(beta, "beta", dl_bfloat16);
  CheckDtype(beta_tma, "beta_tma", dl_bfloat16);
  CheckDtype(A_log, "A_log", dl_float32);
  CheckDtype(dt_bias, "dt_bias", dl_float32);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckDtype(seq_order, "seq_order", dl_int32);
  CheckDtype(initial_state, "initial_state", dl_bfloat16);
  CheckDtype(out, "out", dl_bfloat16);
  CheckDtype(final_state, "final_state", dl_bfloat16);
  CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8);

  TVM_FFI_ICHECK(descriptor_storage.numel() >= kDescriptorStorageBytes)
      << "descriptor_storage must contain at least " << kDescriptorStorageBytes << " bytes";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % 64 == 0)
      << "descriptor_storage must be 64-byte aligned";

  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(3) == flash_kda::kHeadDim && q.size(2) == num_heads)
      << "q must be [B, T, " << num_heads << ", 128]";
  const int64_t token_count = q.numel() / (num_heads * flash_kda::kHeadDim);
  TVM_FFI_ICHECK(token_count > 0) << "q must contain at least one token";

  for (const auto& named : {std::pair<const TensorView*, const char*>(&k, "k"),
                            std::pair<const TensorView*, const char*>(&v, "v"),
                            std::pair<const TensorView*, const char*>(&g, "g"),
                            std::pair<const TensorView*, const char*>(&out, "out")}) {
    const TensorView& tensor = *named.first;
    TVM_FFI_ICHECK(tensor.numel() == q.numel())
        << named.second << " must match q's flattened [tokens, H, 128] shape";
  }
  TVM_FFI_ICHECK(beta.numel() == token_count * num_heads)
      << "beta must match flattened [tokens, H]";

  TVM_FFI_ICHECK(beta_tma.ndim() == 2) << "beta_tma must be [rows, H]";
  const int64_t beta_tma_dim1 = beta_tma.size(1);
  TVM_FFI_ICHECK(beta_tma_dim1 >= num_heads && beta_tma.numel() % beta_tma_dim1 == 0)
      << "beta_tma must be [rows, H] with H >= num_heads";
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  TVM_FFI_ICHECK(beta_tma_rows >= token_count)
      << "beta_tma must cover all tokens";
  // Padded-beta routes (the descriptor fetches a full 32-token/8-head box)
  // need both invariants; the direct-read beta route (head count not a TMA
  // box multiple) leaves the slot unencoded, so only the row coverage above
  // applies.
  const bool beta_direct_read = (num_heads & 7) != 0 && beta_tma_dim1 == num_heads;
  if (!beta_direct_read) {
    TVM_FFI_ICHECK(beta_tma_dim1 >= flash_kda::kBetaTmaHeadsPerBox &&
                   beta_tma_dim1 % flash_kda::kBetaTmaHeadsPerBox == 0)
        << "padded beta_tma must be [tokens, H] with H padded to a multiple of 8 heads";
    TVM_FFI_ICHECK(beta_tma_rows >= 32)
        << "padded beta_tma must cover all tokens plus the 32-token TMA box";
  }

  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(num_seqs > 0) << "cu_seqlens must describe at least one sequence";
  TVM_FFI_ICHECK(seq_order.numel() == num_seqs) << "seq_order must have one entry per sequence";

  if (use_initial_state != 0) {
    TVM_FFI_ICHECK(initial_state.numel() ==
                   num_seqs * num_heads * flash_kda::kHeadDim * flash_kda::kHeadDim)
        << "initial_state must be [N, H, 128, 128]";
  }

  CheckNoOverlap(out, "out", q, "q");
  CheckNoOverlap(out, "out", k, "k");
  CheckNoOverlap(out, "out", v, "v");
  CheckNoOverlap(out, "out", g, "g");
  CheckNoOverlap(out, "out", beta, "beta");
  CheckNoOverlap(out, "out", initial_state, "initial_state");
  return token_count;
}

// Single-block stable descending sort of segment indices by length, matching
// the packed-prefill host policy: equal lengths keep their original relative
// order. num_seqs is small (tens) in every supported packed layout; one
// thread performs a stable insertion sort.
__global__ void VibeCUDAStableDescLengthSort(const int64_t* __restrict__ cu_seqlens, int num_seqs,
                                             int* __restrict__ order) {
  extern __shared__ int64_t vibecuda_sort_buf[];
  int64_t* lengths = vibecuda_sort_buf;
  int* idx = reinterpret_cast<int*>(vibecuda_sort_buf + num_seqs);
  for (int i = threadIdx.x; i < num_seqs; i += blockDim.x) {
    lengths[i] = cu_seqlens[i + 1] - cu_seqlens[i];
    idx[i] = i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < num_seqs; ++i) {
      const int64_t key_len = lengths[idx[i]];
      const int key_idx = idx[i];
      int j = i - 1;
      while (j >= 0 && lengths[idx[j]] < key_len) {
        idx[j + 1] = idx[j];
        --j;
      }
      idx[j + 1] = key_idx;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < num_seqs; i += blockDim.x) {
    order[i] = idx[i];
  }
}

void SortSeqsInto(TensorView cu_seqlens, TensorView order_out, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(cu_seqlens.device().device_type == kDLCUDA) << "cu_seqlens must be a CUDA tensor";
  const int32_t device_id = cu_seqlens.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckCudaTensor(order_out, "order_out", device_id);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckDtype(order_out, "order_out", dl_int32);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(num_seqs > 0) << "cu_seqlens must describe at least one sequence";
  TVM_FFI_ICHECK(order_out.numel() == num_seqs) << "order_out must have one entry per sequence";
  const int64_t smem = num_seqs * (sizeof(int64_t) + sizeof(int));
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  VibeCUDAStableDescLengthSort<<<1, 1024, static_cast<size_t>(smem), stream>>>(
      reinterpret_cast<const int64_t*>(cu_seqlens.data_ptr()), static_cast<int>(num_seqs),
      reinterpret_cast<int*>(order_out.data_ptr()));
  CheckCuda(cudaGetLastError(), "VibeCUDAStableDescLengthSort launch");
}

void RunM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
             TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
             TensorView seq_order, TensorView initial_state, TensorView out, TensorView final_state,
             TensorView descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
             int64_t use_initial_state, int64_t store_final_state, double scale, double lower_bound,
             int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t token_count = CheckVibeCUDAPrefillInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  const int64_t beta_tma_dim1 = beta_tma.size(1);
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  const int64_t slab = VibeCUDASlabRegime(token_count, num_heads);
  if (slab != 0) {
    kda_flash_slab::RunM128(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                            beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
                            cu_seqlens.data_ptr(), seq_order.data_ptr(), initial_state.data_ptr(),
                            out.data_ptr(), final_state.data_ptr(), descriptor_storage.data_ptr(),
                            token_count, num_seqs, prepare_descriptors, num_heads,
                            use_initial_state, store_final_state, scale, lower_bound, beta_tma_rows,
                            beta_tma_dim1, slab, cuda_stream);
  } else {
    kda_flash::RunM128(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                       beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
                       cu_seqlens.data_ptr(), seq_order.data_ptr(), initial_state.data_ptr(),
                       out.data_ptr(), final_state.data_ptr(), descriptor_storage.data_ptr(),
                       token_count, num_seqs, prepare_descriptors, num_heads, use_initial_state,
                       store_final_state, scale, lower_bound, beta_tma_rows, beta_tma_dim1, slab,
                       cuda_stream);
  }
}

void RunM64(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
            TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
            TensorView seq_order, TensorView initial_state, TensorView out, TensorView final_state,
            TensorView descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
            int64_t use_initial_state, int64_t store_final_state, double scale, double lower_bound,
            int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t token_count = CheckVibeCUDAPrefillInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  const int64_t beta_tma_dim1 = beta_tma.size(1);
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  kda_flash::RunM64(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                    beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
                    cu_seqlens.data_ptr(), seq_order.data_ptr(), initial_state.data_ptr(),
                    out.data_ptr(), final_state.data_ptr(), descriptor_storage.data_ptr(),
                    token_count, num_seqs, prepare_descriptors, num_heads, use_initial_state,
                    store_final_state, scale, lower_bound, beta_tma_rows, beta_tma_dim1,
                    cuda_stream);
}

void RunPersistentM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                       TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                       TensorView cu_seqlens, TensorView seq_order, TensorView task_ids,
                       TensorView task_offsets, TensorView choice_scratch, TensorView initial_state,
                       TensorView out, TensorView final_state, TensorView descriptor_storage,
                       int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                       double scale, double lower_bound, int64_t sm_count, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t token_count = CheckVibeCUDAPrefillInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      /*store_final_state=*/use_initial_state, scale, lower_bound);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  CheckCudaTensor(task_ids, "task_ids", device_id);
  CheckCudaTensor(task_offsets, "task_offsets", device_id);
  CheckCudaTensor(choice_scratch, "choice_scratch", device_id);
  CheckDtype(task_ids, "task_ids", dl_int32);
  CheckDtype(task_offsets, "task_offsets", dl_int32);
  CheckDtype(choice_scratch, "choice_scratch", dl_int32);
  TVM_FFI_ICHECK(sm_count >= 1 && sm_count <= 160)
      << "persistent M128 plan supports 1..160 workers, got " << sm_count;
  TVM_FFI_ICHECK(task_ids.numel() >= num_seqs * num_heads) << "task_ids too small";
  TVM_FFI_ICHECK(task_offsets.numel() >= sm_count + 1) << "task_offsets too small";
  TVM_FFI_ICHECK(choice_scratch.numel() >= num_seqs * num_heads) << "choice_scratch too small";
  const int64_t beta_tma_dim1 = beta_tma.size(1);
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  kda_flash::RunPersistentM128(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
                               beta.data_ptr(), beta_tma.data_ptr(), A_log.data_ptr(),
                               dt_bias.data_ptr(), cu_seqlens.data_ptr(), seq_order.data_ptr(),
                               task_ids.data_ptr(), task_offsets.data_ptr(),
                               choice_scratch.data_ptr(), initial_state.data_ptr(), out.data_ptr(),
                               final_state.data_ptr(), descriptor_storage.data_ptr(), token_count,
                               num_seqs, prepare_descriptors, num_heads, use_initial_state,
                               /*store_final_state=*/use_initial_state, scale, lower_bound,
                               beta_tma_rows, beta_tma_dim1, sm_count, cuda_stream);
}

void RunM128Split(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                  TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
                  TensorView seq_order, TensorView initial_state, TensorView out,
                  TensorView final_state, TensorView descriptor_storage, TensorView split_state,
                  TensorView map_state, TensorView carry, TensorView split_out,
                  TensorView map_state_bf16, TensorView lookback_flags, int64_t prepare_descriptors,
                  int64_t num_heads, int64_t use_initial_state, double scale, double lower_bound,
                  int64_t num_parts, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t token_count = CheckVibeCUDAPrefillInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      /*store_final_state=*/0, scale, lower_bound);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  const int64_t num_tasks = num_seqs * num_heads;
  CheckCudaTensor(split_state, "split_state", device_id);
  CheckCudaTensor(map_state, "map_state", device_id);
  CheckCudaTensor(carry, "carry", device_id);
  CheckCudaTensor(split_out, "split_out", device_id);
  CheckCudaTensor(map_state_bf16, "map_state_bf16", device_id);
  CheckCudaTensor(lookback_flags, "lookback_flags", device_id);
  CheckDtype(split_state, "split_state", dl_float32);
  CheckDtype(map_state, "map_state", dl_float32);
  CheckDtype(carry, "carry", dl_float32);
  CheckDtype(split_out, "split_out", dl_bfloat16);
  CheckDtype(map_state_bf16, "map_state_bf16", dl_bfloat16);
  CheckDtype(lookback_flags, "lookback_flags", dl_int32);
  TVM_FFI_ICHECK(num_parts >= 2) << "RunM128Split requires num_parts >= 2";
  TVM_FFI_ICHECK(split_state.numel() >= num_tasks * num_parts * 16384)
      << "split_state too small for tasks*parts";
  TVM_FFI_ICHECK(map_state.numel() >= num_tasks * num_parts * 16384)
      << "map_state too small for tasks*parts";
  TVM_FFI_ICHECK(map_state_bf16.numel() >= num_tasks * num_parts * 16384)
      << "map_state_bf16 too small for tasks*parts";
  TVM_FFI_ICHECK(carry.numel() >= num_tasks * (num_parts - 1) * 16384)
      << "carry too small for tasks*(parts-1)";
  TVM_FFI_ICHECK(split_out.numel() >= out.numel()) << "split_out too small";
  // The per-part flag channel doubles as the dead-map progress flags and the
  // fused-correction lookback flags; the split kernel binding sizes it at
  // num_tasks * num_parts and clears it before each consuming pass.
  TVM_FFI_ICHECK(lookback_flags.numel() >= num_tasks * num_parts)
      << "lookback_flags too small for tasks*parts";
  CheckNoOverlap(split_out, "split_out", q, "q");
  CheckNoOverlap(split_out, "split_out", k, "k");
  CheckNoOverlap(split_out, "split_out", v, "v");
  CheckNoOverlap(split_out, "split_out", g, "g");
  CheckNoOverlap(out, "out", split_out, "split_out");

  const int64_t beta_tma_dim1 = beta_tma.size(1);
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  const int64_t slab = VibeCUDASlabRegime(token_count, num_heads);
  if (slab != 0) {
    kda_flash_slab::RunM128SplitFull(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), cu_seqlens.data_ptr(),
        seq_order.data_ptr(), initial_state.data_ptr(), out.data_ptr(), final_state.data_ptr(),
        descriptor_storage.data_ptr(), token_count, num_seqs, prepare_descriptors, num_heads,
        use_initial_state, scale, lower_bound, beta_tma_rows, beta_tma_dim1, num_parts,
        split_state.data_ptr(), map_state.data_ptr(), carry.data_ptr(), split_out.data_ptr(),
        lookback_flags.data_ptr(), map_state_bf16.data_ptr(), slab, cuda_stream);
  } else {
    kda_flash::RunM128SplitFull(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), cu_seqlens.data_ptr(),
        seq_order.data_ptr(), initial_state.data_ptr(), out.data_ptr(), final_state.data_ptr(),
        descriptor_storage.data_ptr(), token_count, num_seqs, prepare_descriptors, num_heads,
        use_initial_state, scale, lower_bound, beta_tma_rows, beta_tma_dim1, num_parts,
        split_state.data_ptr(), map_state.data_ptr(), carry.data_ptr(), split_out.data_ptr(),
        lookback_flags.data_ptr(), map_state_bf16.data_ptr(), slab, cuda_stream);
  }
}

}  // namespace vibecuda_flashkda
}  // namespace flashinfer

namespace flashinfer {
namespace vibecuda_flashkda {

// Head-family route (fam2): single-kernel per-(sequence, head) recurrence
// with the head count pinned at compile time. head_family selects the
// namespace launcher: 12 (slab-derived beta direct-read image), 64, and 96
// (generic slab-derived images receiving the M128 regime flag).
void RunFam2(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
             TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
             TensorView seq_order, TensorView initial_state, TensorView out, TensorView final_state,
             TensorView descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
             int64_t use_initial_state, int64_t store_final_state, double scale, double lower_bound,
             int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(num_heads == 12 || num_heads == 64 || num_heads == 96)
      << "RunFam2 requires num_heads in {12, 64, 96}, got " << num_heads;

  const int64_t token_count = CheckVibeCUDAPrefillInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound);
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  const int64_t beta_tma_dim1 = beta_tma.size(1);
  const int64_t beta_tma_rows = beta_tma.numel() / beta_tma_dim1;
  const int64_t ft_slab = (num_heads == 12) ? 1 : VibeCUDASlabRegime(token_count, num_heads);
  auto* run_fam2 = (num_heads == 12)   ? kda_fam2::RunFam2H12
                   : (num_heads == 64) ? kda_fam2::RunFam2H64
                                       : kda_fam2::RunFam2H96;
  run_fam2(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
           beta_tma.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), cu_seqlens.data_ptr(),
           seq_order.data_ptr(), initial_state.data_ptr(), out.data_ptr(), final_state.data_ptr(),
           descriptor_storage.data_ptr(), token_count, num_seqs, prepare_descriptors, num_heads,
           use_initial_state, store_final_state, scale, lower_bound, beta_tma_rows, beta_tma_dim1,
           ft_slab, cuda_stream);
}

// Fused N16 route (H12 short/packed classes): one host call launching the
// factor-preparation kernel followed by the selected recurrence-chain kernel
// (schedule 7/8/9). The factor buffers live in the caller's workspace state.
void RunBt16Fused(TensorView q, TensorView k, TensorView g, TensorView beta, TensorView A_log,
                  TensorView dt_bias, TensorView cu_seqlens, TensorView seq_order, TensorView v,
                  TensorView initial_state, TensorView out, TensorView final_state,
                  TensorView ws_qd, TensorView ws_kd, TensorView ws_w, TensorView ws_qk,
                  TensorView ws_diag, TensorView desc_prepare, TensorView desc_chain,
                  int64_t prep_desc, int64_t chain_desc, int64_t total_chunks, int64_t num_heads,
                  double lower_bound, int64_t prepare_total_ctas, int64_t schedule,
                  int64_t use_initial_state, int64_t store_final_state, double scale,
                  int64_t grid_x, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);
  TVM_FFI_ICHECK(prep_desc == 0 || prep_desc == 1) << "prep_desc must be 0 or 1";
  TVM_FFI_ICHECK(chain_desc == 0 || chain_desc == 1) << "chain_desc must be 0 or 1";
  TVM_FFI_ICHECK(num_heads > 0) << "num_heads must be positive";
  TVM_FFI_ICHECK(total_chunks > 0) << "total_chunks must be positive";
  TVM_FFI_ICHECK(schedule == 7 || schedule == 8 || schedule == 9)
      << "schedule must be 7, 8, or 9, got " << schedule;
  TVM_FFI_ICHECK(q.ndim() == 4 && q.size(3) == flash_kda::kHeadDim && q.size(2) == num_heads)
      << "q must be [B, T, " << num_heads << ", 128]";
  const int64_t tokens = q.numel() / (num_heads * flash_kda::kHeadDim);
  TVM_FFI_ICHECK(tokens > 0) << "q must contain at least one token";
  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(num_seqs > 0) << "cu_seqlens must describe at least one sequence";
  TVM_FFI_ICHECK(seq_order.numel() == num_seqs) << "seq_order must have one entry per sequence";
  CheckCudaTensor(k, "k", device_id);
  CheckCudaTensor(v, "v", device_id);
  CheckCudaTensor(g, "g", device_id);
  CheckCudaTensor(beta, "beta", device_id);
  CheckCudaTensor(out, "out", device_id);
  CheckDtype(k, "k", dl_bfloat16);
  CheckDtype(v, "v", dl_bfloat16);
  CheckDtype(g, "g", dl_bfloat16);
  CheckDtype(beta, "beta", dl_bfloat16);
  CheckDtype(out, "out", dl_bfloat16);
  TVM_FFI_ICHECK(k.numel() == q.numel() && v.numel() == q.numel() && g.numel() == q.numel() &&
                 out.numel() == q.numel())
      << "k/v/g/out must match q's flattened [tokens, H, 128] shape";
  TVM_FFI_ICHECK(beta.numel() == tokens * num_heads) << "beta must match flattened [tokens, H]";
  if (use_initial_state != 0) {
    TVM_FFI_ICHECK(initial_state.numel() ==
                   num_seqs * num_heads * flash_kda::kHeadDim * flash_kda::kHeadDim)
        << "initial_state must be [N, H, 128, 128]";
  }
  CheckNoOverlap(out, "out", q, "q");
  CheckNoOverlap(out, "out", k, "k");
  CheckNoOverlap(out, "out", v, "v");
  CheckNoOverlap(out, "out", g, "g");
  CheckNoOverlap(out, "out", beta, "beta");
  CheckNoOverlap(out, "out", initial_state, "initial_state");

  kda_bt16_prepare_fused::Run(q.data_ptr(), k.data_ptr(), g.data_ptr(), beta.data_ptr(),
                              A_log.data_ptr(), dt_bias.data_ptr(), cu_seqlens.data_ptr(),
                              /*cu_chunks_ptr=*/nullptr, /*chunk_to_seq_ptr=*/nullptr,
                              ws_qd.data_ptr(), ws_kd.data_ptr(), ws_w.data_ptr(),
                              ws_qk.data_ptr(), ws_diag.data_ptr(), desc_prepare.data_ptr(),
                              prep_desc, tokens, total_chunks, num_heads, lower_bound,
                              prepare_total_ctas, cuda_stream, num_seqs);
  auto* run_chain = (schedule == 7)   ? kda_bt16_chain_fn16_s7::RunFull
                    : (schedule == 9) ? kda_bt16_chain_fn16_s9::RunFull
                                      : kda_bt16_chain_fn16_s8::RunFull;
  run_chain(ws_qd.data_ptr(), ws_kd.data_ptr(), ws_w.data_ptr(), ws_qk.data_ptr(),
            ws_diag.data_ptr(), v.data_ptr(), cu_seqlens.data_ptr(),
            /*cu_chunks_ptr=*/nullptr, seq_order.data_ptr(), initial_state.data_ptr(),
            out.data_ptr(), final_state.data_ptr(), desc_chain.data_ptr(), chain_desc, tokens,
            total_chunks, num_heads, use_initial_state, store_final_state, scale, grid_x,
            cuda_stream, num_seqs);
}

}  // namespace vibecuda_flashkda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sort_seqs_into, flashinfer::vibecuda_flashkda::SortSeqsInto);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_m128, flashinfer::vibecuda_flashkda::RunM128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_m64, flashinfer::vibecuda_flashkda::RunM64);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_persistent_m128,
                              flashinfer::vibecuda_flashkda::RunPersistentM128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_m128_split, flashinfer::vibecuda_flashkda::RunM128Split);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_fam2, flashinfer::vibecuda_flashkda::RunFam2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_bt16_fused, flashinfer::vibecuda_flashkda::RunBt16Fused);
