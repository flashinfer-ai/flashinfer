// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#ifndef DEVICE_ONLY
#include "tvm_ffi_utils.h"
#endif
#include <stdint.h>

#ifndef DEVICE_ONLY
namespace ffi = tvm::ffi;
#endif

constexpr int kThreads = 256;
constexpr int kItems = 8;
constexpr int kMaxFastBlocks = kThreads * kItems;
constexpr int kTopK = 512;
constexpr int kWidth = 2052;

union SelectStorage {
  uint32_t ids[kTopK];
};

union ScratchStorage {
  uint32_t score_bits[kMaxFastBlocks];
  unsigned long long heap_keys[kTopK];
};

__device__ __forceinline__ int find_request(
    int token, const int* __restrict__ cu, int batch) {
  int lo = 0;
  int hi = batch;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (cu[mid] <= token) lo = mid;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int4 v18_identity_value(int col4, int count) {
  const int c = col4 << 2;
  int4 v;
  v.x = (c     < count) ? c     : -1;
  v.y = (c + 1 < count) ? c + 1 : -1;
  v.z = (c + 2 < count) ? c + 2 : -1;
  v.w = (col4 == kWidth / 4 - 1) ? count
                                  : ((c + 3 < count) ? c + 3 : -1);
  return v;
}

__device__ __forceinline__ void v18_store_cs(int4* p, const int4& v) {
  asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};\n" ::
               "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ float block_score(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ k) {
  float score = 0.0f;
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    float dot = 0.0f;
#pragma unroll 4
    for (int d = 0; d < 128; ++d) {
      dot = fmaf(static_cast<float>(q[h * 128 + d]),
                 static_cast<float>(k[d]), dot);
    }
    score += qsa_relu_exact(dot);
  }
  return score;
}

__device__ __forceinline__ unsigned long long score_key(float score, int index) {
  return (static_cast<unsigned long long>(qsa_score_key_exact(score)) << 32) |
         static_cast<unsigned int>(0xffffffffu - static_cast<unsigned int>(index));
}

__device__ __forceinline__ uint32_t key_index(unsigned long long key) {
  return 0xffffffffu - static_cast<uint32_t>(key);
}

__device__ __forceinline__ void mma_fp8_m16n8k32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t load_u32_or_zero(
    const __nv_fp8_e4m3* p, bool valid) {
  return valid ? *reinterpret_cast<const uint32_t*>(p) : 0u;
}

// One CTA computes 16 query rows against all visible key blocks.  Each warp
// owns an 8-column output tile at a time and uses four K=32 FP8 MMA steps per
// head.  The caller-provided int32 output is legal temporary FP32 storage;
// qsa_indexer_kernel consumes and overwrites every element afterward.
__global__ __launch_bounds__(kThreads, 1) void fp8_score_kernel(
    const __nv_fp8_e4m3* __restrict__ qi,
    const __nv_fp8_e4m3* __restrict__ ck,
    const int* __restrict__ cu,
    const int* __restrict__ cpage_base,
    float* __restrict__ scores,
    int total_tokens,
    int batch) {
  int token_base = static_cast<int>(blockIdx.x) * 16;
  if (token_base >= total_tokens) return;
  int req = find_request(token_base, cu, batch);
  int request_end = cu[req + 1];
  int token_last = min(token_base + 15, request_end - 1);

  int pos_last = token_last - cu[req];
  int max_visible = (pos_last + 1) >> 2;
  if (max_visible <= kTopK) return;
  if (max_visible > kMaxFastBlocks) max_visible = kMaxFastBlocks;

  int warp = static_cast<int>(threadIdx.x) >> 5;
  int lane = static_cast<int>(threadIdx.x) & 31;
  int group = lane >> 2;
  int quad = lane & 3;
  int row0 = group;
  int row1 = group + 8;
  bool row0_valid = token_base + row0 < request_end;
  bool row1_valid = token_base + row1 < request_end;
  const __nv_fp8_e4m3* key_base =
      ck + (static_cast<long long>(cpage_base[req]) * 64) * 128;

  for (int block_base = warp * 8; block_base < max_visible;
       block_base += 8 * 8) {
    int key_col = block_base + group;
    const __nv_fp8_e4m3* key = key_base + static_cast<long long>(key_col) * 128;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int head = 0; head < 4; ++head) {
      float d0 = 0.0f;
      float d1 = 0.0f;
      float d2 = 0.0f;
      float d3 = 0.0f;
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        int k0 = kt * 32 + quad * 4;
        const __nv_fp8_e4m3* q0 =
            qi + static_cast<long long>(token_base + row0) * 512 + head * 128;
        const __nv_fp8_e4m3* q1 =
            qi + static_cast<long long>(token_base + row1) * 512 + head * 128;
        uint32_t a0 = load_u32_or_zero(q0 + k0, row0_valid);
        uint32_t a1 = load_u32_or_zero(q1 + k0, row1_valid);
        uint32_t a2 = load_u32_or_zero(q0 + k0 + 16, row0_valid);
        uint32_t a3 = load_u32_or_zero(q1 + k0 + 16, row1_valid);
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(key + k0);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(key + k0 + 16);
        mma_fp8_m16n8k32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
      }
      sum0 += qsa_relu_exact(d0);
      sum1 += qsa_relu_exact(d1);
      sum2 += qsa_relu_exact(d2);
      sum3 += qsa_relu_exact(d3);
    }

    int col0 = quad * 2;
    if (row0_valid) {
      float* out0 = scores + static_cast<long long>(token_base + row0) * kWidth;
      out0[block_base + col0 + 0] = sum0;
      out0[block_base + col0 + 1] = sum1;
    }
    if (row1_valid) {
      float* out1 = scores + static_cast<long long>(token_base + row1) * kWidth;
      out1[block_base + col0 + 0] = sum2;
      out1[block_base + col0 + 1] = sum3;
    }
  }
}

__device__ __forceinline__ void heap_sift_down(
    unsigned long long* heap, int root, int size) {
  while (true) {
    int child = root * 2 + 1;
    if (child >= size) return;
    if (child + 1 < size && heap[child + 1] < heap[child]) ++child;
    if (heap[root] <= heap[child]) return;
    unsigned long long x = heap[root];
    heap[root] = heap[child];
    heap[child] = x;
    root = child;
  }
}

// Find the bucket containing the descending one-based rank.  Bins are split
// into contiguous per-thread runs so the final reverse walk is local.
__device__ __forceinline__ void find_radix_crossing(
    const int* histogram, int bins, int rank,
    int* warp_prefix, int* result) {
  int tid = static_cast<int>(threadIdx.x);
  int lane = tid & 31;
  int warp = tid >> 5;
  int bins_per_thread = bins / kThreads;
  int base = tid * bins_per_thread;
  int local = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (i < bins_per_thread) local += histogram[base + i];

  int inclusive = local;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) warp_prefix[warp] = inclusive;
  __syncthreads();
  if (tid == 0) {
    int sum = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      int count = warp_prefix[w];
      warp_prefix[w] = sum;
      sum += count;
    }
    warp_prefix[8] = sum;
  }
  __syncthreads();

  int below = warp_prefix[warp] + inclusive - local;
  int above = warp_prefix[8] - below - local;
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    if (i < bins_per_thread) {
      int count = histogram[base + i];
      if (above < rank && rank <= above + count) {
        result[0] = base + i;
        result[1] = above;
      }
      above += count;
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kThreads, 1) void qsa_indexer_kernel(
    const __nv_fp8_e4m3* __restrict__ qi,
    const __nv_fp8_e4m3* __restrict__ ck,
    const int* __restrict__ cu,
    const int* __restrict__ cpage_base,
    int* __restrict__ packed,
    int total_tokens,
    int batch,
    bool scores_ready) {
  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (token >= total_tokens) return;

  // The common W5 shape has two short requests.  Verify that invariant from
  // cu_seqlens on-device; an uneven split with a scored row takes the exact
  // path below.
  if (batch == 2 && total_tokens == 4057) {
    const int split = cu[1];
    if (split <= 2051 && total_tokens - split <= 2051) {
      int4* out4 = reinterpret_cast<int4*>(packed);
      const int g0 = token * (2 * kThreads) + tid;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int g = g0 + k * kThreads;
        const int row = g / (kWidth / 4);
        const int col4 = g - row * (kWidth / 4);
        const int count = row - (row >= split ? split : 0) + 1;
        v18_store_cs(out4 + g, v18_identity_value(col4, count));
      }
      // The remaining one-int4-per-row tail is a disjoint flat interval;
      // assign it to the first 16 CTAs for coalesced stores.
      constexpr int kTailCtas = (4057 + kThreads - 1) / kThreads;
      if (token < kTailCtas) {
        const int g = total_tokens * (2 * kThreads) + token * kThreads + tid;
        if (g >= total_tokens * (2 * kThreads + 1)) return;
        const int row = g / (kWidth / 4);
        const int col4 = g - row * (kWidth / 4);
        const int count = row - (row >= split ? split : 0) + 1;
        v18_store_cs(out4 + g, v18_identity_value(col4, count));
      }
      return;
    }
  }

  int req = find_request(token, cu, batch);
  int pos = token - cu[req];
  int visible = (pos + 1) >> 2;
  int* out = packed + static_cast<long long>(token) * kWidth;

  // For <=512 visible blocks, top-k contains every visible block.  Expansion
  // plus the causal tail is exactly the ascending interval [0, pos].
  if (visible <= kTopK) {
    int4* out4 = reinterpret_cast<int4*>(out);
    for (int v = tid; v < 512; v += kThreads) {
      int j = v << 2;
      out4[v] = make_int4(
          j <= pos ? j : -1,
          j + 1 <= pos ? j + 1 : -1,
          j + 2 <= pos ? j + 2 : -1,
          j + 3 <= pos ? j + 3 : -1);
    }
    if (tid == 0) {
      out4[512] = make_int4(
          2048 <= pos ? 2048 : -1,
          2049 <= pos ? 2049 : -1,
          2050 <= pos ? 2050 : -1,
          pos + 1);
    }
    return;
  }

  const __nv_fp8_e4m3* q = qi + static_cast<long long>(token) * 512;
  const __nv_fp8_e4m3* keys =
      ck + (static_cast<long long>(cpage_base[req]) * 64) * 128;

  __shared__ SelectStorage selected;
  __shared__ ScratchStorage scratch;
  __shared__ int histogram[2048];
  __shared__ int radix_result[2];
  __shared__ int warp_counts[8];
  __shared__ int warp_offsets[9];
  __shared__ int selected_base;
  __shared__ int equal_base;

  if (visible <= kMaxFastBlocks) {
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      int b = tid + i * kThreads;
      if (b < visible) {
        int tile_base = token & ~15;
        bool tensor_score = scores_ready &&
            find_request(tile_base, cu, batch) == req;
        float s = tensor_score
            ? reinterpret_cast<const float*>(packed)[
                  static_cast<long long>(token) * kWidth + b]
            : block_score(q, keys + static_cast<long long>(b) * 128);
        scratch.score_bits[b] = qsa_score_key_exact(s);
      } else {
        scratch.score_bits[b] = 0;
      }
    }
    __syncthreads();

    // Three exact radix rounds locate the 512th FP32 score.  All scores are
    // nonnegative, so their unsigned bit patterns have the same ordering.
    for (int i = tid; i < 1024; i += kThreads) histogram[i] = 0;
    __syncthreads();
    for (int i = 0; i < kItems; ++i) {
      int b = tid + i * kThreads;
      if (b < visible) atomicAdd(&histogram[scratch.score_bits[b] >> 21], 1);
    }
    __syncthreads();
    find_radix_crossing(histogram, 1024, kTopK,
                        warp_offsets, radix_result);
    int high = radix_result[0];
    int rank_high = kTopK - radix_result[1];

    for (int i = tid; i < 2048; i += kThreads) histogram[i] = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      int b = tid + i * kThreads;
      uint32_t bits = scratch.score_bits[b];
      if (b < visible && static_cast<int>(bits >> 21) == high)
        atomicAdd(&histogram[(bits >> 10) & 0x7ffu], 1);
    }
    __syncthreads();
    find_radix_crossing(histogram, 2048, rank_high,
                        warp_offsets, radix_result);
    int middle = radix_result[0];
    int rank_middle = rank_high - radix_result[1];

    for (int i = tid; i < 1024; i += kThreads) histogram[i] = 0;
    __syncthreads();
    uint32_t upper_prefix =
        (static_cast<uint32_t>(high) << 11) |
        static_cast<uint32_t>(middle);
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      int b = tid + i * kThreads;
      uint32_t bits = scratch.score_bits[b];
      if (b < visible && (bits >> 10) == upper_prefix)
        atomicAdd(&histogram[bits & 0x3ffu], 1);
    }
    __syncthreads();
    find_radix_crossing(histogram, 1024, rank_middle,
                        warp_offsets, radix_result);
    uint32_t threshold_score =
        (static_cast<uint32_t>(high) << 21) |
        (static_cast<uint32_t>(middle) << 10) |
        static_cast<uint32_t>(radix_result[0]);
    int equal_needed = rank_middle - radix_result[1];

    // Scan selected flags in block-index order.  Ballots provide warp-local
    // prefix ranks; eight warp counts provide the CTA prefix.  This directly
    // emits ascending ids, eliminating a second key sort.
    if (tid == 0) {
      selected_base = 0;
      equal_base = 0;
    }
    __syncthreads();
    int lane = tid & 31;
    int warp = tid >> 5;
#pragma unroll
    for (int tile = 0; tile < kItems; ++tile) {
      int b = tile * kThreads + tid;
      uint32_t score_bits = scratch.score_bits[b];
      bool equal = b < visible && score_bits == threshold_score;
      unsigned int equal_vote = __ballot_sync(0xffffffffu, equal);
      if (lane == 0) warp_counts[warp] = __popc(equal_vote);
      __syncthreads();
      if (tid == 0) {
        int base = equal_base;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          warp_offsets[w] = base;
          base += warp_counts[w];
        }
        equal_base = base;
      }
      __syncthreads();
      unsigned int lower = (1u << lane) - 1u;
      int equal_rank = warp_offsets[warp] + __popc(equal_vote & lower);
      bool chosen = b < visible &&
          (score_bits > threshold_score || (equal && equal_rank < equal_needed));
      unsigned int vote = __ballot_sync(0xffffffffu, chosen);
      if (lane == 0) warp_counts[warp] = __popc(vote);
      __syncthreads();
      if (tid == 0) {
        int base = selected_base;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          warp_offsets[w] = base;
          base += warp_counts[w];
        }
        selected_base = base;
      }
      __syncthreads();
      if (chosen) {
        int rank = warp_offsets[warp] + __popc(vote & lower);
        int v = b * 4;
        out[rank * 4 + 0] = v + 0;
        out[rank * 4 + 1] = v + 1;
        out[rank * 4 + 2] = v + 2;
        out[rank * 4 + 3] = v + 3;
      }
      __syncthreads();
    }
  } else {
    // Scope-safe fallback for sequences beyond the shipped 2047-block case.
    // One lane maintains the exact 512-entry min-heap; the selected indices
    // are then sorted cooperatively before expansion.
    if (tid == 0) {
      for (int b = 0; b < kTopK; ++b) {
        float s = block_score(q, keys + static_cast<long long>(b) * 128);
        scratch.heap_keys[b] = score_key(s, b);
      }
      for (int root = kTopK / 2 - 1; root >= 0; --root)
        heap_sift_down(scratch.heap_keys, root, kTopK);
      for (int b = kTopK; b < visible; ++b) {
        float s = block_score(q, keys + static_cast<long long>(b) * 128);
        unsigned long long x = score_key(s, b);
        if (x > scratch.heap_keys[0]) {
          scratch.heap_keys[0] = x;
          heap_sift_down(scratch.heap_keys, 0, kTopK);
        }
      }
      for (int i = 0; i < kTopK; ++i)
        selected.ids[i] = key_index(scratch.heap_keys[i]);
    }
    __syncthreads();

    // Scope-only fallback: sort the heap's selected ids before expansion.
    for (int size = 2; size <= kTopK; size <<= 1) {
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
        for (int part = 0; part < 2; ++part) {
          int i = tid + part * kThreads;
          int j = i ^ stride;
          if (j > i) {
            uint32_t a = selected.ids[i];
            uint32_t b = selected.ids[j];
            bool ascending = ((i & size) == 0);
            if ((ascending && a > b) || (!ascending && a < b)) {
              selected.ids[i] = b;
              selected.ids[j] = a;
            }
          }
        }
        __syncthreads();
      }
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      int rank = tid + part * kThreads;
      int v = static_cast<int>(selected.ids[rank]) * 4;
      out[rank * 4 + 0] = v + 0;
      out[rank * 4 + 1] = v + 1;
      out[rank * 4 + 2] = v + 2;
      out[rank * 4 + 3] = v + 3;
    }
  }

  int tail = pos + 1 - visible * 4;
  if (tid < 3) out[2048 + tid] = (tid < tail) ? visible * 4 + tid : -1;
  if (tid == 0) out[kWidth - 1] = 2048 + tail;
}

// When B==1 and every row is in the all-visible-blocks regime, the request
// position is the token id and no metadata lookup or CTA barrier is needed.
__global__ __launch_bounds__(256, 4)
void qsa_trivial_b1_kernel(int4* __restrict__ packed4, int total_tokens) {
  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (token >= total_tokens) return;
  int4* const row = packed4 + static_cast<long long>(token) * (kWidth / 4);
  const int count = token + 1;
  const int col0 = tid * 4;
  const int col1 = col0 + 1024;
  row[tid] = make_int4(col0 < count ? col0 : -1,
                       col0 + 1 < count ? col0 + 1 : -1,
                       col0 + 2 < count ? col0 + 2 : -1,
                       col0 + 3 < count ? col0 + 3 : -1);
  row[tid + 256] = make_int4(col1 < count ? col1 : -1,
                             col1 + 1 < count ? col1 + 1 : -1,
                             col1 + 2 < count ? col1 + 2 : -1,
                             col1 + 3 < count ? col1 + 3 : -1);
  if (tid == 0) {
    row[512] = make_int4(2048 < count ? 2048 : -1,
                         2049 < count ? 2049 : -1,
                         2050 < count ? 2050 : -1,
                         count);
  }
}

constexpr int kGroupedRows = 8;

__global__ __launch_bounds__(256, 4)
void qsa_trivial_grouped_kernel(const int* __restrict__ cu,
                                int4* __restrict__ packed4,
                                int total_tokens, int batch) {
  const int tid = static_cast<int>(threadIdx.x);
  const int token_base = static_cast<int>(blockIdx.x) * kGroupedRows;
  const int rows = min(kGroupedRows, total_tokens - token_base);
  __shared__ int cu_shared[1025];
  __shared__ int positions[kGroupedRows];
  for (int i = tid; i <= batch; i += blockDim.x) cu_shared[i] = cu[i];
  __syncthreads();
  if (tid < rows) {
    const int token = token_base + tid;
    const int request = find_request(token, cu_shared, batch);
    positions[tid] = token - cu_shared[request];
  }
  __syncthreads();

  const int row_int4 = kWidth / 4;
  const int total = rows * row_int4;
  int4* const base = packed4 + static_cast<long long>(token_base) * row_int4;
  for (int i = tid; i < total; i += blockDim.x) {
    const int local_row = i / row_int4;
    const int j = i - local_row * row_int4;
    const int pos = positions[local_row];
    const int col = j * 4;
    base[i] = j < kTopK
        ? make_int4(col <= pos ? col : -1, col + 1 <= pos ? col + 1 : -1,
                    col + 2 <= pos ? col + 2 : -1,
                    col + 3 <= pos ? col + 3 : -1)
        : make_int4(2048 <= pos ? 2048 : -1,
                    2049 <= pos ? 2049 : -1,
                    2050 <= pos ? 2050 : -1, pos + 1);
  }
}

#ifndef DEVICE_ONLY
void Run(ffi::TensorView qi,
         ffi::TensorView ck_cache,
         ffi::TensorView cu_seqlens,
         ffi::TensorView cpage_base,
         ffi::TensorView packed) {
  int total_tokens = static_cast<int>(qi.shape()[0]);
  int batch = static_cast<int>(cu_seqlens.shape()[0]) - 1;
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id));
  auto* q_ptr = static_cast<const __nv_fp8_e4m3*>(qi.data_ptr());
  auto* k_ptr = static_cast<const __nv_fp8_e4m3*>(ck_cache.data_ptr());
  auto* cu_ptr = static_cast<const int*>(cu_seqlens.data_ptr());
  auto* base_ptr = static_cast<const int*>(cpage_base.data_ptr());
  auto* out_ptr = static_cast<int*>(packed.data_ptr());
  if (total_tokens <= 2051) {
    if (batch == 1) {
      qsa_trivial_b1_kernel<<<total_tokens, 256, 0, stream>>>(
          reinterpret_cast<int4*>(out_ptr), total_tokens);
    } else {
      qsa_trivial_grouped_kernel
          <<<(total_tokens + kGroupedRows - 1) / kGroupedRows,
             256, 0, stream>>>(
              cu_ptr, reinterpret_cast<int4*>(out_ptr),
              total_tokens, batch);
    }
    return;
  }
  bool scores_ready = (total_tokens + batch - 1) / batch > 2048;
  if (scores_ready) {
    fp8_score_kernel<<<(total_tokens + 15) / 16, kThreads, 0, stream>>>(
        q_ptr, k_ptr, cu_ptr, base_ptr,
        reinterpret_cast<float*>(out_ptr), total_tokens, batch);
  }
  qsa_indexer_kernel<<<total_tokens, kThreads, 0, stream>>>(
      q_ptr, k_ptr, cu_ptr, base_ptr, out_ptr,
      total_tokens, batch, scores_ready);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif




