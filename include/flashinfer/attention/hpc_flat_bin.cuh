// Experimental HPC-Ops parallel scheduler adapter; Apache-2.0.
// Original parallel assigner is MIT-licensed in hpc_original_assign.cuh.
#pragma once
#include <cub/block/block_scan.cuh>

#include "decode.cuh"
#include "default_decode_params.cuh"
#include "hpc_original_assign.cuh"
#include "variants.cuh"

namespace flashinfer {
__device__ __forceinline__ int HPCPad12(int n) { return (n + 11) / 12 * 12; }

// Storage pages are 64 tokens; SM100 scheduling tiles span 128 tokens.
// Clear only synchronization flags, including when the active map layout changes.
__global__ void HPCPrepare(const int* indptr, const int* tail, int B, int H, int P, int capacity,
                           int* lengths, int* raw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) lengths[i] = (indptr[i + 1] - indptr[i] - 1) * 64 + tail[i];
  int flag_offset = 12 * (1 + (capacity + 1) * P) + HPCPad12(B * H);
  if (i < HPCPad12(P)) raw[flag_offset + i] = 0;
  if (i == 0) {
    raw[2] = H;
    raw[3] = B;
  }
}

// Parallel scan converts original head-major chunk counts to FI's
// batch-major compact partial-state offsets, and CTA task counts to CSR.
__global__ void HPCScan(const int* raw, int B, int H, int P, int* state_ptr, int* cta_ptr,
                        int* meta) {
  using Scan = cub::BlockScan<int, 128>;
  __shared__ typename Scan::TempStorage storage;
  __shared__ int carry;
  int t = threadIdx.x;
  int chunk_offset = 12 * (1 + raw[0] * P);
  if (t == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < B * H; base += 128) {
    int i = base + t, offset, sum;
    int count = i < B * H ? raw[chunk_offset + (i % H) * B + i / H] : 0;
    Scan(storage).ExclusiveSum(count, offset, sum);
    if (i < B * H) state_ptr[i] = carry + offset;
    __syncthreads();
    if (t == 0) carry += sum;
    __syncthreads();
  }
  if (t == 0) {
    state_ptr[B * H] = carry;
    meta[1] = carry;
    carry = 0;
  }
  __syncthreads();
  int count_offset = chunk_offset + HPCPad12(B * H) + HPCPad12(P);
  for (int base = 0; base < P; base += 128) {
    int i = base + t, offset, sum;
    Scan(storage).ExclusiveSum(i < P ? raw[count_offset + i] : 0, offset, sum);
    if (i < P) cta_ptr[i] = carry + offset;
    __syncthreads();
    if (t == 0) carry += sum;
    __syncthreads();
  }
  if (t == 0) {
    cta_ptr[P] = carry;
    meta[0] = raw[0] - 1;
  }
}

// Every CTA adapts its own original 48-byte records in parallel. Execution
// consumes these records, not an independently reconstructed policy.
__global__ void HPCAdapt(const int* raw, const int* state_ptr, const int* cta_ptr, int H,
                         int* tasks, int* requests) {
  int c = blockIdx.x;
  for (int j = threadIdx.x; j < cta_ptr[c + 1] - cta_ptr[c]; j += blockDim.x) {
    const int* src = raw + 12 * (1 + c * raw[0] + j);
    int h = src[0], b = src[1], state = state_ptr[b * H + h] + src[2];
    int* dst = tasks + 5 * (cta_ptr[c] + j);
    dst[0] = b;
    dst[1] = h;
    dst[2] = src[3];
    dst[3] = src[3] + src[4];
    dst[4] = state;
    requests[state] = b;
  }
}

template <uint32_t STAGES, typename Params>
__global__ void HPCFlatBinDecode(Params params, const int32_t* cta_ptr, const int32_t* tasks) {
  extern __shared__ uint8_t smem[];
  for (int i = cta_ptr[blockIdx.x]; i < cta_ptr[blockIdx.x + 1]; ++i) {
    const int32_t* t = tasks + 5 * i;
    BatchDecodeWithPagedKVCacheDevice<PosEncodingMode::kNone, STAGES, 1, 8, 16, 4, 2,
                                      DefaultAttention<false, false, false, false>, Params>(
        params, smem, t[4], t[1], threadIdx.x, threadIdx.y, threadIdx.z, t[2], t[3], t[4] * 4);
    __syncthreads();
  }
}
}  // namespace flashinfer
