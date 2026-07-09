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

__global__ void prepare_fmha_v2_inputs_kernel(
    int const* __restrict__ seq_lens_q, int const* __restrict__ seq_lens_kv, int batch_size,
    // dtype + flags
    int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, float scale_bmm1, float scale_bmm2,
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
  // the final iteration: when batch_size is exactly aligned, the last lane of
  // the last block writes the total.
  for (int chunk = 0; chunk < batch_size; chunk += kThreadsPerBlock) {
    int b = chunk + tid;
    int q_len = (b < batch_size) ? seq_lens_q[b] : 0;
    int kv_len = (b < batch_size) ? seq_lens_kv[b] : 0;

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

}  // namespace

// Host entry point exposed via TVM-FFI.
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

  int* seq_lens_q_ptr = static_cast<int*>(seq_lens_q.data_ptr());
  int* seq_lens_kv_ptr = static_cast<int*>(seq_lens_kv.data_ptr());
  int* cum_q_ptr =
      cum_seq_lens_q.has_value() ? static_cast<int*>(cum_seq_lens_q.value().data_ptr()) : nullptr;
  int* cum_kv_ptr =
      cum_seq_lens_kv.has_value() ? static_cast<int*>(cum_seq_lens_kv.value().data_ptr()) : nullptr;
  uint32_t* tile_ptr = tile_id_counter.has_value()
                           ? static_cast<uint32_t*>(tile_id_counter.value().data_ptr())
                           : nullptr;
  uint32_t* s1_ptr =
      scale_bmm1_d.has_value() ? static_cast<uint32_t*>(scale_bmm1_d.value().data_ptr()) : nullptr;
  uint32_t* s2_ptr =
      scale_bmm2_d.has_value() ? static_cast<uint32_t*>(scale_bmm2_d.value().data_ptr()) : nullptr;

  // cum_seq_lens_{q,kv} must be writable buffers of length B+1 when their
  // corresponding seq_lens are scanned. The kernel always scans both arrays;
  // if a caller really only wants one, pass a small throwaway buffer for the
  // other. This keeps the kernel branchless.
  TVM_FFI_ICHECK(cum_q_ptr != nullptr) << "cum_seq_lens_q is required";
  TVM_FFI_ICHECK(cum_kv_ptr != nullptr) << "cum_seq_lens_kv is required";

  // Single block; one launch handles any batch size via the strided loop.
  prepare_fmha_v2_inputs_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      seq_lens_q_ptr, seq_lens_kv_ptr, batch_size, scale_bmm1_dtype_code, scale_bmm2_dtype_code,
      static_cast<float>(scale_bmm1), static_cast<float>(scale_bmm2), cum_q_ptr, cum_kv_ptr,
      tile_ptr, s1_ptr, s2_ptr);
}
