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
// launch performs an exclusive-sum scan over per-batch q/kv lengths, zeroes the
// FMHA tile counter, and encodes the BMM1/BMM2 scales into device-resident
// uint32 buffers (matching set_alpha() in fused_multihead_attention_utils.h so
// every existing epilogue that consults scale_bmm{1,2}_d reads identical bits).
//
// The kernel is templated on PAGED. The paged instantiation additionally
// derives the q/kv lengths from (qo_indptr, paged_kv_indptr,
// paged_kv_last_page_len) and scatters paged_kv_indices into a dense
// block_tables[B, max_blocks_per_seq] — the lengths feed the scan in-register,
// so the whole paged plan() prep is one kernel launch with no seq-lens
// intermediate in global memory. The ragged instantiation compiles all of the
// paged work out via `if constexpr`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;
using tvm::ffi::Optional;

// Must match the Data_type enum in csrc/fmha_v2/fused_multihead_attention_utils.h.
// We don't pull in that header here so the prep kernel stays free of FMHA params.
enum FmhaV2DType {
  FMHA_V2_DTYPE_FP16 = 0,
  FMHA_V2_DTYPE_FP32 = 1,
  FMHA_V2_DTYPE_INT32 = 2,
  FMHA_V2_DTYPE_INT8 = 3,
  FMHA_V2_DTYPE_BF16 = 4,
  FMHA_V2_DTYPE_E4M3 = 5,
};

namespace {

// Same encoding as set_alpha() in fused_multihead_attention_utils.h.
__device__ __forceinline__ uint32_t encode_alpha(float norm, int dtype_code) {
  switch (dtype_code) {
    case FMHA_V2_DTYPE_FP16: {
      __half h = __float2half_rn(norm);
      uint16_t u = __half_as_ushort(h);
      return (uint32_t(u) << 16) | uint32_t(u);  // ushort2{u, u} bit-cast
    }
    case FMHA_V2_DTYPE_BF16: {
      __nv_bfloat16 b = __float2bfloat16(norm);
      uint16_t u = __bfloat16_as_ushort(b);
      return (uint32_t(u) << 16) | uint32_t(u);
    }
    case FMHA_V2_DTYPE_INT32: {
      int32_t inorm = static_cast<int32_t>(norm);
      return reinterpret_cast<uint32_t const&>(inorm);
    }
    case FMHA_V2_DTYPE_FP32:
    default:
      return reinterpret_cast<uint32_t const&>(norm);
  }
}

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
    int batch_size, int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, float scale_bmm1,
    float scale_bmm2,
    // outputs
    int* __restrict__ cum_seq_lens_q, int* __restrict__ cum_seq_lens_kv,
    uint32_t* __restrict__ tile_id_counter, uint32_t* __restrict__ scale_bmm1_d,
    uint32_t* __restrict__ scale_bmm2_d) {
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
        // sequence's page indices are scattered into its dense block-table row.
        q_len = qo_indptr[b + 1] - qo_indptr[b];
        int page_begin = paged_kv_indptr[b];
        int num_pages = paged_kv_indptr[b + 1] - page_begin;
        kv_len = max(num_pages - 1, 0) * page_size + paged_kv_last_page_len[b];
        kv_lens_out[b] = kv_len;
        int row_offset = b * max_blocks_per_seq;
        for (int j = 0; j < num_pages && j < max_blocks_per_seq; ++j) {
          block_tables[row_offset + j] = paged_kv_indices[page_begin + j];
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
  // populate the FMHA prep scalars.
  if (tid == 0) {
    cum_seq_lens_q[batch_size] = prefix_q;
    cum_seq_lens_kv[batch_size] = prefix_kv;
    if (tile_id_counter) {
      tile_id_counter[0] = 0u;
    }
    if (scale_bmm1_d) {
      scale_bmm1_d[0] = encode_alpha(scale_bmm1, scale_bmm1_dtype_code);
    }
    if (scale_bmm2_d) {
      scale_bmm2_d[0] = encode_alpha(scale_bmm2, scale_bmm2_dtype_code);
    }
  }
}

struct PrepCommonPtrs {
  int* cum_q;
  int* cum_kv;
  uint32_t* tile;
  uint32_t* s1;
  uint32_t* s2;
};

PrepCommonPtrs unpack_prep_common(Optional<ffi::TensorView> const& cum_seq_lens_q,
                                  Optional<ffi::TensorView> const& cum_seq_lens_kv,
                                  Optional<ffi::TensorView> const& tile_id_counter,
                                  Optional<ffi::TensorView> const& scale_bmm1_d,
                                  Optional<ffi::TensorView> const& scale_bmm2_d) {
  PrepCommonPtrs p;
  p.cum_q =
      cum_seq_lens_q.has_value() ? static_cast<int*>(cum_seq_lens_q.value().data_ptr()) : nullptr;
  p.cum_kv =
      cum_seq_lens_kv.has_value() ? static_cast<int*>(cum_seq_lens_kv.value().data_ptr()) : nullptr;
  p.tile = tile_id_counter.has_value() ? static_cast<uint32_t*>(tile_id_counter.value().data_ptr())
                                       : nullptr;
  p.s1 =
      scale_bmm1_d.has_value() ? static_cast<uint32_t*>(scale_bmm1_d.value().data_ptr()) : nullptr;
  p.s2 =
      scale_bmm2_d.has_value() ? static_cast<uint32_t*>(scale_bmm2_d.value().data_ptr()) : nullptr;
  // cum_seq_lens_{q,kv} must be writable buffers of length B+1: the kernel
  // always scans both arrays.
  TVM_FFI_ICHECK(p.cum_q != nullptr) << "cum_seq_lens_q is required";
  TVM_FFI_ICHECK(p.cum_kv != nullptr) << "cum_seq_lens_kv is required";
  return p;
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
// tile_id_counter         : uint32 [1]  (optional; zeroed if provided)
// scale_bmm1_d / scale_bmm2_d : uint32 [1] (optional; set_alpha-encoded scalars)
//
// Note on scale_bmm1_d layout: this buffer is length 1, not the length-2
// {scale, scale*log2e} pair described in fmha_v2_fused_prep_kernel.md §1.
// The Python caller pre-multiplies log2e on the host for the warp-spec path
// (prefill.py plan()) and encodes a single FP32 word here.  compute.h:312
// and epilogue.h:973 both do a single __ldg/reinterpret_cast<float>, so
// length-1 is the correct contract.  Do not expand to length-2 without
// updating those read sites.
//
// scale_bmm{1,2}_dtype_code follows the FmhaV2DType enum above (same numeric
// values as Data_type in fused_multihead_attention_utils.h). When the caller
// wants the warp-specialized "fused log2e" path for BMM1, it should pass
// scale_bmm1 = fused_scale_bmm1 * M_LOG2E and scale_bmm1_dtype_code = FP32 so
// the encoding matches the host-side branch in fmha_v2_run.cu.
void fmha_v2_prepare(ffi::TensorView seq_lens_q, ffi::TensorView seq_lens_kv, int batch_size,
                     int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, double scale_bmm1,
                     double scale_bmm2, Optional<ffi::TensorView> cum_seq_lens_q,
                     Optional<ffi::TensorView> cum_seq_lens_kv,
                     Optional<ffi::TensorView> tile_id_counter,
                     Optional<ffi::TensorView> scale_bmm1_d,
                     Optional<ffi::TensorView> scale_bmm2_d) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(seq_lens_q.device()));
  PrepCommonPtrs p = unpack_prep_common(cum_seq_lens_q, cum_seq_lens_kv, tile_id_counter,
                                        scale_bmm1_d, scale_bmm2_d);

  prepare_fmha_v2_inputs_kernel<false><<<1, kThreadsPerBlock, 0, stream>>>(
      static_cast<int*>(seq_lens_q.data_ptr()), static_cast<int*>(seq_lens_kv.data_ptr()),
      /*qo_indptr=*/nullptr, /*paged_kv_indptr=*/nullptr, /*paged_kv_last_page_len=*/nullptr,
      /*paged_kv_indices=*/nullptr, /*kv_lens_out=*/nullptr, /*block_tables=*/nullptr,
      /*page_size=*/0, /*max_blocks_per_seq=*/0, batch_size, scale_bmm1_dtype_code,
      scale_bmm2_dtype_code, static_cast<float>(scale_bmm1), static_cast<float>(scale_bmm2),
      p.cum_q, p.cum_kv, p.tile, p.s1, p.s2);
}

// Paged variant: everything fmha_v2_prepare does, plus deriving the q/kv
// lengths from (qo_indptr, paged_kv_indptr, paged_kv_last_page_len), emitting
// kv_lens_out [B] for run(), and scattering paged_kv_indices into
// block_tables_out [B, max_blocks_per_seq].
void fmha_v2_prepare_paged(ffi::TensorView qo_indptr, ffi::TensorView paged_kv_indptr,
                           ffi::TensorView paged_kv_last_page_len, ffi::TensorView paged_kv_indices,
                           ffi::TensorView kv_lens_out, ffi::TensorView block_tables_out,
                           int page_size, int max_blocks_per_seq, int batch_size,
                           int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, double scale_bmm1,
                           double scale_bmm2, Optional<ffi::TensorView> cum_seq_lens_q,
                           Optional<ffi::TensorView> cum_seq_lens_kv,
                           Optional<ffi::TensorView> tile_id_counter,
                           Optional<ffi::TensorView> scale_bmm1_d,
                           Optional<ffi::TensorView> scale_bmm2_d) {
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream(qo_indptr.device()));
  PrepCommonPtrs p = unpack_prep_common(cum_seq_lens_q, cum_seq_lens_kv, tile_id_counter,
                                        scale_bmm1_d, scale_bmm2_d);

  prepare_fmha_v2_inputs_kernel<true><<<1, kThreadsPerBlock, 0, stream>>>(
      /*seq_lens_q=*/nullptr, /*seq_lens_kv=*/nullptr, static_cast<int*>(qo_indptr.data_ptr()),
      static_cast<int*>(paged_kv_indptr.data_ptr()),
      static_cast<int*>(paged_kv_last_page_len.data_ptr()),
      static_cast<int*>(paged_kv_indices.data_ptr()), static_cast<int*>(kv_lens_out.data_ptr()),
      static_cast<int*>(block_tables_out.data_ptr()), page_size, max_blocks_per_seq, batch_size,
      scale_bmm1_dtype_code, scale_bmm2_dtype_code, static_cast<float>(scale_bmm1),
      static_cast<float>(scale_bmm2), p.cum_q, p.cum_kv, p.tile, p.s1, p.s2);
}
