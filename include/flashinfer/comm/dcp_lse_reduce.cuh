/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused decode context-parallel all-to-all + LSE-weighted reduce.
//
// Under decode context parallelism the KV cache is sharded across a team. Every
// rank attends the query heads against its own KV shard and produces a partial
// output plus the log-sum-exp of the softmax denominator over that shard. The
// merge is a standard softmax-weighted reduction: gathering the per-rank LSEs
// for one (token, head) into a vector l,
//
//   out = sum_r softmax(l)_r * o_r
//   lse = logsumexp(l)
//
// i.e. the LSEs act as the logits and the softmax is over the *rank* axis, not
// the key axis. It is evaluated in the max-shifted stable form, which is
// identical to softmax(l) because softmax is shift invariant. With base-2 LSE
// (FlashInfer MLA) the exponentials are base 2, giving softmax(l*ln2).
//
// The one deviation from softmax proper: if every shard reports -inf for a head
// (an empty KV range everywhere) softmax would be 0/0, so that case is defined
// to produce a zero output rather than NaN. NaN and +inf inputs fold to -inf
// before the softmax.
//
// Transport. Each rank stores every (row, destination) slice straight into the
// destination's registered symmetric window (ncclGetLsaPointer) using the LL128
// protocol: a row's 8-byte data words are packed 15 per 128-byte line, and the
// line's last word is a per-call flag. A warp stores four lines per 16-byte
// vector store (lane = 8 * line + part); lane part 7 carries data word 14 and
// the flag. Readiness relies on a warp-coalesced 128-byte store becoming visible
// atomically, as NCCL's LL128 protocol does over NVLink: once a receiver sees a
// line's flag, it sees the whole line. The row's LSE travels separately as a
// 16-byte LL line ({lse, flag, 0, flag}; each 8-byte half carries its own flag)
// once per group of four lines. Data and readiness therefore arrive together:
// there is no fence, readiness word or grid-wide barrier on the critical path.
//
// Work split. One warp sends each (row, destination). One warp receives each
// (row, group of four lines): each lane polls its 16-byte part of the line from
// every source together with (lane r) source r's LSE line for the group, merges
// in fixed source order (the result is deterministic) and writes the output.
// Sources are processed in batches of eight; lanes hold the LSEs of sources r
// and r + 32.
//
// Slot reuse. Two slots alternate by call, selected by a device-side epoch that
// every block reads at entry and the last block to finish advances, so CUDA
// graph replays advance it. A peer can run at most one call ahead of a receiver,
// never two. Every line is read by exactly one lane, which zeroes its flag after
// reading, so between calls a slot holds only zero flags; flags are never zero,
// so no stale line can look ready, including after the 32-bit epoch wraps. The
// workspace is zeroed once at creation.
//
// The launch is cooperative only to guarantee that every block is co-resident:
// receivers spin on data that co-resident blocks on the peers send first. All
// ranks must belong to one load/store-accessible (NVLink) domain.

#ifndef FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
#define FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nccl_device.h>

namespace flashinfer {
namespace comm {
namespace dcp {

constexpr int kNumSlots = 2;
constexpr int kMaxRanks = 64;
constexpr int kFusedBlockSize = 128;
constexpr int kWarpsPerBlock = kFusedBlockSize / 32;

// Workspace: [epoch, finished-block counter, ...] metadata, then the payload.
constexpr size_t kPayloadOffset = 128;

// LL128 data lines.
constexpr int kLineBytes = 128;
constexpr int kLineDataWords = 15;  // word 15 of a line is the flag
constexpr int kLinesPerGroup = 4;   // 32 lanes x 16 bytes
constexpr int kLseLineBytes = 16;
// A group index is a lane index when the LSE lines are sent.
constexpr int kMaxGroups = 32;
// Sources a receiving lane keeps in flight at once.
constexpr int kSourceBatch = 8;
// Groups a sending lane loads before storing them.
constexpr int kSendGroups = 4;
// Data-line flags carry the call number in the low word under a nonzero high word.
constexpr uint64_t kFlagTag = 0x4c4c313200000000ull;

inline constexpr int DataWords(int head_dim, int element_size) {
  return head_dim * element_size / 8;
}

inline constexpr int DataLines(int head_dim, int element_size) {
  return (DataWords(head_dim, element_size) + kLineDataWords - 1) / kLineDataWords;
}

inline constexpr int LineGroups(int head_dim, int element_size) {
  return (DataLines(head_dim, element_size) + kLinesPerGroup - 1) / kLinesPerGroup;
}

// Workspace bytes per (slot, source, row): its data lines and one LSE line per group.
inline constexpr size_t RowBytes(int head_dim, int element_size) {
  return static_cast<size_t>(DataLines(head_dim, element_size)) * kLineBytes +
         static_cast<size_t>(LineGroups(head_dim, element_size)) * kLseLineBytes;
}

__device__ __forceinline__ void store_line128(void* dst, uint64_t lo, uint64_t hi) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(dst), "l"(lo), "l"(hi) : "memory");
}

__device__ __forceinline__ ulonglong2 load_line128(const void* src) {
  ulonglong2 v;
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
               : "=l"(v.x), "=l"(v.y)
               : "l"(src)
               : "memory");
  return v;
}

__device__ __forceinline__ void store_zero64(void* dst) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(dst), "l"(0ull) : "memory");
}

__device__ __forceinline__ void store_lse_line(void* dst, uint32_t lse_bits, uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst), "r"(lse_bits),
               "r"(flag), "r"(0u), "r"(flag)
               : "memory");
}

__device__ __forceinline__ uint4 load_lse_line(const void* src) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(src)
               : "memory");
  return v;
}

__device__ __forceinline__ void store_zero128(void* dst) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(dst), "r"(0u) : "memory");
}

__device__ __forceinline__ bool lse_line_ready(const uint4& v, uint32_t flag) {
  return v.y == flag && v.w == flag;
}

// Four fp16/bf16 values in one 8-byte word; accumulation is always fp32.
template <typename T>
struct Word4;

template <>
struct Word4<__nv_bfloat16> {
  __device__ static inline void unpack(uint64_t u, float (&f)[4]) {
    const uint32_t lo = static_cast<uint32_t>(u), hi = static_cast<uint32_t>(u >> 32);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&lo));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&hi));
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
  }
  __device__ static inline uint64_t pack(const float (&f)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(f[0], f[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(f[2], f[3]);
    return static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&a)) |
           (static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&b)) << 32);
  }
};

template <>
struct Word4<__half> {
  __device__ static inline void unpack(uint64_t u, float (&f)[4]) {
    const uint32_t lo = static_cast<uint32_t>(u), hi = static_cast<uint32_t>(u >> 32);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
  }
  __device__ static inline uint64_t pack(const float (&f)[4]) {
    __half2 a = __floats2half2_rn(f[0], f[1]);
    __half2 b = __floats2half2_rn(f[2], f[3]);
    return static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&a)) |
           (static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&b)) << 32);
  }
};

// Sanitise an incoming LSE. A shard that contributed nothing reports -inf; some
// backends emit NaN or +inf instead, and both must fold to "no weight" rather
// than poisoning the whole merge.
__device__ static inline float sanitize_lse(float l) {
  return (isnan(l) || l == INFINITY) ? -INFINITY : l;
}

// partial_o element (token, head, peer, d) is read from
//   partial_o + token * o_s_tok + head * o_s_head + peer * o_s_peer + d   (d contiguous)
// and partial_lse (token, head, peer) at the analogous lse_* strides, so a strided
// view of an attention output [tokens, cp * heads, D] needs no packing. The output
// is contiguous [tokens * heads, head_dim]; row = token * local_heads + head.
//
// Payload layout (bytes from the start of the payload):
//   data lines  [slot][src][max_rows][line]  128 bytes each
//   LSE lines   [slot][src][max_rows][group]  16 bytes each, after both data slots
template <typename T, bool BASE_E>
__global__ void __launch_bounds__(kFusedBlockSize)
    FusedKernel(const T* __restrict__ partial_o, const float* __restrict__ partial_lse,
                int64_t o_s_tok, int64_t o_s_head, int64_t o_s_peer, int64_t lse_s_tok,
                int64_t lse_s_head, int64_t lse_s_peer, unsigned char* __restrict__ workspace,
                ncclWindow_t window, size_t payload_window_offset, T* __restrict__ combined_out,
                int rank, int nranks, int num_tokens, int local_heads, int max_rows, int head_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  const int total_warps = gridDim.x * kWarpsPerBlock;
  const int part = lane & 7;  // 16-byte part of a line
  const int sub = lane >> 3;  // line within a group

  auto* state = reinterpret_cast<uint32_t*>(workspace);
  const uint32_t epoch = *reinterpret_cast<volatile uint32_t*>(&state[0]);
  const uint64_t flag = kFlagTag | static_cast<uint32_t>(epoch + 1u);
  const uint32_t lse_flag = (epoch + 1u) != 0u ? epoch + 1u : 1u;
  const int slot = static_cast<int>(epoch & 1u);

  const int data_words = head_dim * static_cast<int>(sizeof(T)) / 8;
  const int lines = (data_words + kLineDataWords - 1) / kLineDataWords;
  const int groups = (lines + kLinesPerGroup - 1) / kLinesPerGroup;
  const int num_rows = num_tokens * local_heads;
  const size_t row_bytes = static_cast<size_t>(lines) * kLineBytes;
  const size_t lse_row_bytes = static_cast<size_t>(groups) * kLseLineBytes;
  const size_t data_slot_bytes = static_cast<size_t>(nranks) * max_rows * row_bytes;
  const size_t lse_slot_bytes = static_cast<size_t>(nranks) * max_rows * lse_row_bytes;
  const size_t lse_region = kNumSlots * data_slot_bytes;

  // ---- send: one warp per (row, destination); consecutive warps target different peers.
  for (int item = warp; item < num_rows * nranks; item += total_warps) {
    const int dst = item % nranks;
    const int row = item / nranks;
    const int token = row / local_heads;
    const int head = row - token * local_heads;
    const uint64_t* src = reinterpret_cast<const uint64_t*>(partial_o + token * o_s_tok +
                                                            head * o_s_head + dst * o_s_peer);
    const size_t src_row = static_cast<size_t>(rank) * max_rows + row;
    auto* dst_row = reinterpret_cast<unsigned char*>(ncclGetLsaPointer(
        window, payload_window_offset + slot * data_slot_bytes + src_row * row_bytes, dst));
    for (int g0 = 0; g0 < groups; g0 += kSendGroups) {
      uint64_t lo[kSendGroups], hi[kSendGroups];
#pragma unroll
      for (int i = 0; i < kSendGroups; ++i) {
        const int line = (g0 + i) * kLinesPerGroup + sub;
        const int j0 = line * kLineDataWords + 2 * part, j1 = j0 + 1;
        lo[i] = j0 < data_words ? src[j0] : 0ull;
        hi[i] = part == 7 ? flag : (j1 < data_words ? src[j1] : 0ull);
      }
#pragma unroll
      for (int i = 0; i < kSendGroups; ++i) {
        const int line = (g0 + i) * kLinesPerGroup + sub;
        if (line < lines) {
          store_line128(dst_row + static_cast<size_t>(line) * kLineBytes + part * 16, lo[i], hi[i]);
        }
      }
    }
    if (lane < groups) {
      const float lse = partial_lse[token * lse_s_tok + head * lse_s_head + dst * lse_s_peer];
      auto* lse_line = reinterpret_cast<unsigned char*>(
          ncclGetLsaPointer(window,
                            payload_window_offset + lse_region + slot * lse_slot_bytes +
                                src_row * lse_row_bytes + lane * kLseLineBytes,
                            dst));
      store_lse_line(lse_line, __float_as_uint(lse), lse_flag);
    }
  }

  // ---- receive + merge: one warp per (row, group).
  const unsigned char* my_data = workspace + kPayloadOffset + slot * data_slot_bytes;
  const unsigned char* my_lse = workspace + kPayloadOffset + lse_region + slot * lse_slot_bytes;
  const size_t data_src_stride = static_cast<size_t>(max_rows) * row_bytes;
  const size_t lse_src_stride = static_cast<size_t>(max_rows) * lse_row_bytes;
  for (int item = warp; item < num_rows * groups; item += total_warps) {
    const int row = item / groups;
    const int group = item - row * groups;
    const int line = group * kLinesPerGroup + sub;
    const bool has_line = line < lines;
    const unsigned char* my_part = my_data + static_cast<size_t>(row) * row_bytes +
                                   static_cast<size_t>(line) * kLineBytes + part * 16;
    const unsigned char* my_lse_line =
        my_lse + static_cast<size_t>(row) * lse_row_bytes + group * kLseLineBytes;

    // LSE lines of sources lane and lane + 32; an absent source reads as ready.
    const uint4 absent = make_uint4(0u, lse_flag, 0u, lse_flag);
    const unsigned char* lse_line0 = my_lse_line + lane * lse_src_stride;
    const unsigned char* lse_line1 = my_lse_line + (lane + 32) * lse_src_stride;
    uint4 s0 = lane < nranks ? load_lse_line(lse_line0) : absent;
    uint4 s1 = lane + 32 < nranks ? load_lse_line(lse_line1) : absent;

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float e0 = 0.0f, e1 = 0.0f, inv = 0.0f;
    for (int base = 0; base < nranks; base += kSourceBatch) {
      ulonglong2 v[kSourceBatch];
#pragma unroll
      for (int k = 0; k < kSourceBatch; ++k) {
        if (base + k < nranks && has_line)
          v[k] = load_line128(my_part + (base + k) * data_src_stride);
      }
      // Poll this batch's lines (and, with the first batch, the LSE lines) until all
      // have landed. A line's flag lane decides for its whole line.
      for (bool pending = true; pending;) {
        bool mine = false;
        if (base == 0) {
          if (!lse_line_ready(s0, lse_flag)) {
            s0 = load_lse_line(lse_line0);
            mine = true;
          }
          if (!lse_line_ready(s1, lse_flag)) {
            s1 = load_lse_line(lse_line1);
            mine = true;
          }
        }
#pragma unroll
        for (int k = 0; k < kSourceBatch; ++k) {
          if (base + k < nranks) {
            const bool here = !has_line || (part == 7 && v[k].y == flag);
            if (!__shfl_sync(0xffffffffu, here, lane | 7)) {
              v[k] = load_line128(my_part + (base + k) * data_src_stride);
              mine = true;
            }
          }
        }
        pending = __any_sync(0xffffffffu, mine);
      }
      if (base == 0) {
        // Weights: lanes hold the (sanitised) LSEs of sources lane and lane + 32.
        const float l0 = lane < nranks ? sanitize_lse(__uint_as_float(s0.x)) : -INFINITY;
        const float l1 = lane + 32 < nranks ? sanitize_lse(__uint_as_float(s1.x)) : -INFINITY;
        if (lane < nranks) store_zero128(const_cast<unsigned char*>(lse_line0));
        if (lane + 32 < nranks) store_zero128(const_cast<unsigned char*>(lse_line1));
        float m = fmaxf(l0, l1);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (m == -INFINITY) m = 0.0f;
        e0 = lane < nranks ? (BASE_E ? __expf(l0 - m) : exp2f(l0 - m)) : 0.0f;
        e1 = lane + 32 < nranks ? (BASE_E ? __expf(l1 - m) : exp2f(l1 - m)) : 0.0f;
        float denom = e0 + e1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) denom += __shfl_xor_sync(0xffffffffu, denom, o);
        inv = (denom == 0.0f) ? 0.0f : 1.0f / denom;
      }
      // Consume: zero each line's flag, then merge in fixed source order.
#pragma unroll
      for (int k = 0; k < kSourceBatch; ++k) {
        const int r = base + k;
        if (r < nranks) {
          const float w = __shfl_sync(0xffffffffu, r < 32 ? e0 : e1, r & 31) * inv;
          if (has_line) {
            if (part == 7)
              store_zero64(const_cast<unsigned char*>(my_part + r * data_src_stride + 8));
            float x[4], y[4];
            Word4<T>::unpack(v[k].x, x);
            Word4<T>::unpack(v[k].y, y);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              acc0[e] += w * x[e];
              acc1[e] += w * y[e];
            }
          }
        }
      }
    }
    if (has_line) {
      const int j0 = line * kLineDataWords + 2 * part, j1 = j0 + 1;
      uint64_t* out =
          reinterpret_cast<uint64_t*>(combined_out + static_cast<size_t>(row) * head_dim);
      if (j0 < data_words) out[j0] = Word4<T>::pack(acc0);
      if (part != 7 && j1 < data_words) out[j1] = Word4<T>::pack(acc1);
    }
  }

  // The last block to finish advances the epoch. Every block read it at entry,
  // before its __syncthreads below, so no block of this call sees the new value;
  // the next call is stream-ordered after this kernel.
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t finished = atomicAdd(&state[1], 1u);
    if (finished == gridDim.x - 1) {
      state[1] = 0u;
      state[0] = epoch + 1u;
    }
  }
}

}  // namespace dcp
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_DCP_LSE_REDUCE_CUH_
