/*
 * Copyright (c) 2023-2026 by FlashInfer team.
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

// Single fused prep kernel for FMHAv2.
//
// Mirrors TensorRT-LLM's computeSeqAndPaddingOffsets pattern: one block per
// launch performs an exclusive-sum scan over per-batch q/kv lengths and stores
// the BMM1/BMM2 scale words into device-resident uint32 buffers.
//
// The scale words are encoded on the HOST in the launchers below with the real
// set_alpha() from fused_multihead_attention_utils.h, using the scale-type
// selection mirrored from fmha_v2_run.cu::set_params — so the prep path cannot
// drift from the legacy host-encoding path. The kernel Data_type is a codegen
// constant of this module (-DFMHA_V2_DATA_TYPE, set by gen_fmha_v2_module; the
// module is layout/dtype-specialized).
//
// The kernel is templated on PAGED. The paged instantiation additionally
// derives the q/kv lengths from (qo_indptr, paged_kv_indptr,
// paged_kv_last_page_len) and scatters paged_kv_indices into a dense
// block_tables[B, max_blocks_per_seq] (zero-padding each row past its last
// page, so callers can reuse an uninitialized buffer) — the lengths feed the
// scan in-register, so the whole paged plan() prep is one kernel launch with
// no seq-lens intermediate in global memory. The ragged instantiation compiles
// all of the paged work out via `if constexpr`.

#include <cuda_runtime.h>
#include <fused_multihead_attention_utils.h>  // Data_type, set_alpha, CudaDevice

#include <cmath>
#include <cub/cub.cuh>

#include "tvm_ffi_utils.h"

#ifndef FMHA_V2_DATA_TYPE
#error "FMHA_V2_DATA_TYPE must be defined by gen_fmha_v2_module (e.g. DATA_TYPE_FP16)"
#endif

namespace ffi = tvm::ffi;

namespace {

constexpr int kThreadsPerBlock = 256;

// Scratch struct shared by the BlockScan and the single-thread tail.
struct PrepShared {
  typename cub::BlockScan<int, kThreadsPerBlock>::TempStorage q_scan;
  typename cub::BlockScan<int, kThreadsPerBlock>::TempStorage kv_scan;
};

// One block strides over the batch. A single block (rather than
// ceil(B/kThreadsPerBlock) blocks) keeps the exclusive-sum scan simple and is
// still far below launch-overhead scale for realistic batch sizes; merging the
// paged derivation in here saves a whole extra kernel launch per plan().
template <bool PAGED>
__global__ void prepare_fmha_v2_inputs_kernel(
    // ragged inputs (PAGED = false)
    int const* __restrict__ seq_lens_q, int const* __restrict__ seq_lens_kv,
    // paged inputs (PAGED = true)
    int const* __restrict__ qo_indptr, int const* __restrict__ paged_kv_indptr,
    int const* __restrict__ paged_kv_last_page_len, int const* __restrict__ paged_kv_indices,
    int* __restrict__ kv_lens_out, int* __restrict__ block_tables, int page_size,
    int max_blocks_per_seq,
    // common
    int batch_size, uint32_t scale_bmm1_word, uint32_t scale_bmm2_word,
    // outputs
    int* __restrict__ cum_seq_lens_q, int* __restrict__ cum_seq_lens_kv,
    uint32_t* __restrict__ scale_bmm1_d, uint32_t* __restrict__ scale_bmm2_d) {
  __shared__ PrepShared smem;
  // Running prefix across iterations of the strided loop below.
  int prefix_q = 0;
  int prefix_kv = 0;
  const int tid = threadIdx.x;

  // One block scans the entire batch in strides of kThreadsPerBlock.
  // The +1 entry of the exclusive-sum (cum[batch_size] = total) is written in
  // the single-thread tail below.
  for (int chunk = 0; chunk < batch_size; chunk += kThreadsPerBlock) {
    int b = chunk + tid;
    int q_len = 0;
    int kv_len = 0;
    if (b < batch_size) {
      if constexpr (PAGED) {
        // Derive lengths from the paged representation and feed them straight
        // into the scan; kv_lens is also materialized for run(), and this
        // sequence's page indices are scattered into its dense block-table
        // row. The row remainder past num_pages is zero-filled so the caller
        // can hand in a reused, uninitialized block_tables buffer.
        q_len = qo_indptr[b + 1] - qo_indptr[b];
        int page_begin = paged_kv_indptr[b];
        int num_pages = paged_kv_indptr[b + 1] - page_begin;
        kv_len = max(num_pages - 1, 0) * page_size + paged_kv_last_page_len[b];
        kv_lens_out[b] = kv_len;
        int row_offset = b * max_blocks_per_seq;
        int j = 0;
        for (; j < num_pages && j < max_blocks_per_seq; ++j) {
          block_tables[row_offset + j] = paged_kv_indices[page_begin + j];
        }
        for (; j < max_blocks_per_seq; ++j) {
          block_tables[row_offset + j] = 0;
        }
      } else {
        q_len = seq_lens_q[b];
        kv_len = seq_lens_kv[b];
      }
    }

    int q_off, kv_off;
    int q_total_in_chunk, kv_total_in_chunk;
    cub::BlockScan<int, kThreadsPerBlock>(smem.q_scan).ExclusiveSum(q_len, q_off, q_total_in_chunk);
    __syncthreads();
    cub::BlockScan<int, kThreadsPerBlock>(smem.kv_scan)
        .ExclusiveSum(kv_len, kv_off, kv_total_in_chunk);
    __syncthreads();

    if (b < batch_size) {
      cum_seq_lens_q[b] = prefix_q + q_off;
      cum_seq_lens_kv[b] = prefix_kv + kv_off;
    }
    prefix_q += q_total_in_chunk;
    prefix_kv += kv_total_in_chunk;
    __syncthreads();
  }

  // Single-thread tail: write the [batch_size] entry of the exclusive-sum and
  // store the scale words (pre-encoded on the host).
  if (tid == 0) {
    cum_seq_lens_q[batch_size] = prefix_q;
    cum_seq_lens_kv[batch_size] = prefix_kv;
    scale_bmm1_d[0] = scale_bmm1_word;
    scale_bmm2_d[0] = scale_bmm2_word;
  }
}

// Host-side scale encoding for the device-resident scale words. Mirrors the
// scale-type selection in fmha_v2_run.cu::set_params (scale_type1/scale_type2;
// acc_type derivation at its call site) so the fused-prep path encodes exactly
// what the legacy host path would.
void encode_scale_words(float scale_bmm1, float scale_bmm2, bool has_alibi, float softcapping_scale,
                        uint32_t& s1_word, uint32_t& s2_word) {
  constexpr Data_type kDataType = FMHA_V2_DATA_TYPE;
  constexpr Data_type kAccType =
      (kDataType == DATA_TYPE_BF16 || kDataType == DATA_TYPE_E4M3) ? DATA_TYPE_FP32 : kDataType;
  Data_type scale_type1 =
      (kDataType == DATA_TYPE_FP16 || kDataType == DATA_TYPE_BF16) ? kAccType : DATA_TYPE_FP32;
  Data_type scale_type2 =
      (kDataType == DATA_TYPE_FP16 || kDataType == DATA_TYPE_BF16) ? kDataType : DATA_TYPE_FP32;
  if (kDataType == DATA_TYPE_E4M3) {
    scale_type1 = kAccType;
    scale_type2 = kAccType;
  }

  // Fuse 1 / softcapping_scale into scale_bmm1, exactly like
  // fmha_v2_run.cu::set_params.
  bool const enable_softcapping = softcapping_scale != 0.f;
  float fused_scale_bmm1 = enable_softcapping ? scale_bmm1 / softcapping_scale : scale_bmm1;

  // Warp-specialized SM90 kernels fuse log2e into scale_bmm1 (the
  // fused_scale_bmm1 branch in fmha_v2_run.cu::set_params); alibi and
  // softcapping cannot use the exp2f fused-scale optimization.
  // determine_launch_params reduces warp_specialization to sm == 90 here:
  // flash_attention is unconditionally on, the force flags default off, and
  // this module's dtype is one of fp16/bf16/e4m3 by codegen.
  CudaDevice device;
  if (device.sm == 90 && !has_alibi && !enable_softcapping) {
    set_alpha(s1_word, fused_scale_bmm1 * float(M_LOG2E), DATA_TYPE_FP32);
  } else {
    set_alpha(s1_word, fused_scale_bmm1, scale_type1);
  }
  set_alpha(s2_word, scale_bmm2, scale_type2);
}

}  // namespace

// Host entry points exposed via TVM-FFI. Two exports, one kernel:
//   prepare       — ragged: per-batch lengths are given.
//   prepare_paged — paged: lengths derived from indptr; also emits kv_lens and
//                   the dense block_tables. One launch covers the whole plan()
//                   prep for the paged wrapper.
//
// seq_lens_q / seq_lens_kv : int32 [B]   (device-resident)
// cum_seq_lens_q / kv     : int32 [B+1] (device-resident output)
// scale_bmm1_d / scale_bmm2_d : uint32 [1] (set_alpha-encoded scalar outputs)
//
// All buffer arguments are required — the wrapper is the only caller and
// always provides them.
//
// Note on scale_bmm1_d layout: this buffer is length 1, not the length-2
// {scale, scale*log2e} pair described in fmha_v2_fused_prep_kernel.md §1.
// The launcher applies the warp-spec log2e fusion before encoding, and
// compute.h:312 and epilogue.h:973 both do a single
// __ldg/reinterpret_cast<float>, so length-1 is the correct contract. Do not
// expand to length-2 without updating those read sites.
void fmha_v2_prepare(ffi::TensorView seq_lens_q, ffi::TensorView seq_lens_kv, int batch_size,
                     double scale_bmm1, double scale_bmm2, bool has_alibi, double softcapping_scale,
                     ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
                     ffi::TensorView scale_bmm1_d, ffi::TensorView scale_bmm2_d) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(seq_lens_q.device()));
  uint32_t s1_word, s2_word;
  encode_scale_words(static_cast<float>(scale_bmm1), static_cast<float>(scale_bmm2), has_alibi,
                     static_cast<float>(softcapping_scale), s1_word, s2_word);

  prepare_fmha_v2_inputs_kernel<false><<<1, kThreadsPerBlock, 0, stream>>>(
      static_cast<int*>(seq_lens_q.data_ptr()), static_cast<int*>(seq_lens_kv.data_ptr()),
      /*qo_indptr=*/nullptr, /*paged_kv_indptr=*/nullptr, /*paged_kv_last_page_len=*/nullptr,
      /*paged_kv_indices=*/nullptr, /*kv_lens_out=*/nullptr, /*block_tables=*/nullptr,
      /*page_size=*/0, /*max_blocks_per_seq=*/0, batch_size, s1_word, s2_word,
      static_cast<int*>(cum_seq_lens_q.data_ptr()), static_cast<int*>(cum_seq_lens_kv.data_ptr()),
      static_cast<uint32_t*>(scale_bmm1_d.data_ptr()),
      static_cast<uint32_t*>(scale_bmm2_d.data_ptr()));
}

// Paged variant: everything fmha_v2_prepare does, plus deriving the q/kv
// lengths from (qo_indptr, paged_kv_indptr, paged_kv_last_page_len), emitting
// kv_lens_out [B] for run(), and scattering paged_kv_indices into
// block_tables_out [B, max_blocks_per_seq] (rows zero-padded past their last
// page; block_tables_out may be an uninitialized reused buffer).
void fmha_v2_prepare_paged(ffi::TensorView qo_indptr, ffi::TensorView paged_kv_indptr,
                           ffi::TensorView paged_kv_last_page_len, ffi::TensorView paged_kv_indices,
                           ffi::TensorView kv_lens_out, ffi::TensorView block_tables_out,
                           int page_size, int max_blocks_per_seq, int batch_size, double scale_bmm1,
                           double scale_bmm2, bool has_alibi, double softcapping_scale,
                           ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
                           ffi::TensorView scale_bmm1_d, ffi::TensorView scale_bmm2_d) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(qo_indptr.device()));
  uint32_t s1_word, s2_word;
  encode_scale_words(static_cast<float>(scale_bmm1), static_cast<float>(scale_bmm2), has_alibi,
                     static_cast<float>(softcapping_scale), s1_word, s2_word);

  prepare_fmha_v2_inputs_kernel<true><<<1, kThreadsPerBlock, 0, stream>>>(
      /*seq_lens_q=*/nullptr, /*seq_lens_kv=*/nullptr, static_cast<int*>(qo_indptr.data_ptr()),
      static_cast<int*>(paged_kv_indptr.data_ptr()),
      static_cast<int*>(paged_kv_last_page_len.data_ptr()),
      static_cast<int*>(paged_kv_indices.data_ptr()), static_cast<int*>(kv_lens_out.data_ptr()),
      static_cast<int*>(block_tables_out.data_ptr()), page_size, max_blocks_per_seq, batch_size,
      s1_word, s2_word, static_cast<int*>(cum_seq_lens_q.data_ptr()),
      static_cast<int*>(cum_seq_lens_kv.data_ptr()),
      static_cast<uint32_t*>(scale_bmm1_d.data_ptr()),
      static_cast<uint32_t*>(scale_bmm2_d.data_ptr()));
}
