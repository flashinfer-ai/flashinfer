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

#ifndef FLASHINFER_FUSED_MOE_SM90_PUSH_A2A_CUH_
#define FLASHINFER_FUSED_MOE_SM90_PUSH_A2A_CUH_

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>

// Symmetric-window layout + device round protocol for the SM90 push MegaMoE.
// Each helper documents the invariant it must uphold.

namespace flashinfer {
namespace sm90_push {

constexpr uint64_t kPoolHeadStorageBytes = sizeof(unsigned long long);
constexpr uint64_t kAbortCellStorageBytes = sizeof(uint64_t);
static_assert(kPoolHeadStorageBytes + kAbortCellStorageBytes <= 128,
              "The abort cell must fit in the pool-head alignment padding");

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

#ifndef FLASHINFER_SM90_PUSH_DISABLE_TIMEOUT
#define FLASHINFER_SM90_PUSH_CHECK_TIMEOUT(s) \
  ((globaltimer_ns() - (s)) > (300ull * 1000ull * 1000ull * 1000ull))
#else
#define FLASHINFER_SM90_PUSH_CHECK_TIMEOUT(s) false
#endif

struct SlotMeta {
  int32_t src_token;
  int32_t src_rank_k;  // (k << 16) | src_rank; see pack_rank_k
  float weight;
  int32_t payload_slot;
};
static_assert(sizeof(SlotMeta) == 16, "SlotMeta must be 16B");

__device__ __forceinline__ int32_t pack_rank_k(int src_rank, int k) {
  return static_cast<int32_t>((static_cast<uint32_t>(k) << 16) | static_cast<uint32_t>(src_rank));
}
__device__ __forceinline__ int unpack_src_rank(int32_t v) {
  return static_cast<int>(static_cast<uint32_t>(v) & 0xffffu);
}
__device__ __forceinline__ int unpack_k(int32_t v) {
  return static_cast<int>(static_cast<uint32_t>(v) >> 16);
}

struct PushLayout {
  uint64_t const* peer_bases;  // device int64[ep_size]: per-rank window bases
  int ep_size;
  int rank;
  int num_local_experts;  // E (experts per rank)
  int pool_rows;
  int meta_rows;      // SlotMeta record capacity: capacity_factor*ep*t_cap*top_k
  int bytes_per_row;  // H (fp8 payload) or 2H (bf16 payload)
  int hidden;         // H elements
  int top_k;
  int t_cap;  // max local tokens per rank (combine rows)
  uint64_t pool_offset;
  uint64_t pool_sc_offset;  // 0-sized region in bf16-payload mode
  uint64_t pool_meta_offset;
  uint64_t pool_head_offset;
  uint64_t base_cells_offset;
  uint64_t count_cells_offset;
  uint64_t cdone_cells_offset;
  uint64_t ack_cells_offset;
  uint64_t combine_offset;  // 0-sized region in fp8-combine mode
  uint64_t cfp8_offset;     // 0-sized region in bf16-combine mode
  uint64_t csc_offset;

  __device__ __forceinline__ uint8_t* window(int r) const {
    return reinterpret_cast<uint8_t*>(peer_bases[r]);
  }
  __device__ __forceinline__ uint8_t* pool_row(int r, int slot) const {
    return window(r) + pool_offset + static_cast<uint64_t>(slot) * bytes_per_row;
  }
  __device__ __forceinline__ float* pool_sc_row(int r, int slot) const {
    return reinterpret_cast<float*>(window(r) + pool_sc_offset) +
           static_cast<uint64_t>(slot) * (hidden >> 7);
  }
  __device__ __forceinline__ SlotMeta* pool_meta(int r, int slot) const {
    return reinterpret_cast<SlotMeta*>(window(r) + pool_meta_offset) + slot;
  }
  __device__ __forceinline__ unsigned int* pool_head(int r) const {
    return reinterpret_cast<unsigned int*>(window(r) + pool_head_offset);
  }
  __device__ __forceinline__ unsigned long long* pool_head64(int r) const {
    return reinterpret_cast<unsigned long long*>(window(r) + pool_head_offset);
  }
  __device__ __forceinline__ uint64_t* abort_cell(int r) const {
    return reinterpret_cast<uint64_t*>(window(r) + pool_head_offset + kPoolHeadStorageBytes);
  }
  // protocol cells inside rank r's window, keyed by the WRITING party
  __device__ __forceinline__ uint64_t* base_cell(int r, int s) const {
    return reinterpret_cast<uint64_t*>(window(r) + base_cells_offset) + s;
  }
  __device__ __forceinline__ uint64_t* count_cell(int r, int e, int s) const {
    return reinterpret_cast<uint64_t*>(window(r) + count_cells_offset) +
           static_cast<uint64_t>(e) * ep_size + s;
  }
  __device__ __forceinline__ uint64_t* cdone_cell(int r, int d) const {
    return reinterpret_cast<uint64_t*>(window(r) + cdone_cells_offset) + d;
  }
  __device__ __forceinline__ uint64_t* ack_cell(int r, int d) const {
    return reinterpret_cast<uint64_t*>(window(r) + ack_cells_offset) + d;
  }
  __device__ __forceinline__ __nv_bfloat16* combine_row(int r, int token, int k) const {
    return reinterpret_cast<__nv_bfloat16*>(window(r) + combine_offset) +
           (static_cast<uint64_t>(token) * top_k + k) * hidden;
  }
  // per-route fp8 combine slots: [t_cap][top_k][...] keyed by route k
  __device__ __forceinline__ uint8_t* cfp8_row(int r, int token, int k) const {
    return window(r) + cfp8_offset + (static_cast<uint64_t>(token) * top_k + k) * hidden;
  }
  __device__ __forceinline__ float* csc_row(int r, int token, int k) const {
    return reinterpret_cast<float*>(window(r) + csc_offset) +
           (static_cast<uint64_t>(token) * top_k + k) * (hidden >> 7);
  }
  __device__ __forceinline__ uint8_t* cfp8_row_grouped(int r, int token, int src) const {
    return window(r) + cfp8_offset + (static_cast<uint64_t>(token) * ep_size + src) * hidden;
  }
  __device__ __forceinline__ float* csc_row_grouped(int r, int token, int src) const {
    return reinterpret_cast<float*>(window(r) + csc_offset) +
           (static_cast<uint64_t>(token) * ep_size + src) * (hidden >> 7);
  }
};

// Combine-side packed (M, 4) int32 meta row accessors. Every consumer goes
// through these, so a layout change rewrites only the four bodies.
__device__ __forceinline__ int32_t meta_src_rank(const int32_t* __restrict__ mrow) {
  return mrow[0];
}
__device__ __forceinline__ int32_t meta_src_token(const int32_t* __restrict__ mrow) {
  return mrow[1];
}
__device__ __forceinline__ int32_t meta_route_k(const int32_t* __restrict__ mrow) {
  return mrow[2];
}
__device__ __forceinline__ float meta_weight(const int32_t* __restrict__ mrow) {
  return __int_as_float(mrow[3]);
}

// contiguous expert partitioning: expert eid lives on rank eid / E
__device__ __forceinline__ int expert_owner_rank(int expert_id, int num_local_experts) {
  return expert_id / num_local_experts;
}

// Cells are 8-byte {tag<<32 | value}: one st.release publishes value and tag
// atomically. Tags are free-running uint32; all checks are equality-based,
// so wraparound is safe.
__device__ __forceinline__ uint64_t pack_count_tag(int32_t count, uint32_t tag) {
  return (static_cast<uint64_t>(tag) << 32) | static_cast<uint32_t>(count);
}

// Payload stores must precede the release; payload loads must follow the
// acquire. The PTX orders the memory system; the "memory" clobber stops the
// compiler from reordering ordinary accesses across the asm.
__device__ __forceinline__ void st_release_sys_u64(uint64_t* addr, uint64_t v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
#else
  __threadfence_system();
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
#endif
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(uint64_t const* addr) {
  uint64_t v;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
#else
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
  __threadfence_system();
#endif
  return v;
}

__device__ __forceinline__ void publish_abort_all(PushLayout const& L, uint32_t tag) {
  uint64_t const value = pack_count_tag(L.rank + 1, tag);
  for (int peer = 0; peer < L.ep_size; ++peer) {
    st_release_sys_u64(L.abort_cell(peer), value);
  }
  __threadfence_system();
}

__device__ __forceinline__ int32_t wait_tag_u64(PushLayout const& L, uint64_t const* addr,
                                                uint32_t tag) {
  auto s = globaltimer_ns();
  while (true) {
    uint64_t const abort = ld_acquire_sys_u64(L.abort_cell(L.rank));
    uint32_t const abort_rank_plus_one = static_cast<uint32_t>(abort);
    if (abort_rank_plus_one != 0) {
      uint32_t const abort_tag = static_cast<uint32_t>(abort >> 32);
      int32_t const abort_owner = static_cast<int32_t>(abort_rank_plus_one - 1);
      publish_abort_all(L, abort_tag);
      printf("sm90_push: rank %d aborted round %u while waiting for tag %u at %p\n", abort_owner,
             abort_tag, tag, static_cast<const void*>(addr));
      asm volatile("trap;");
      return -1;
    }
    uint64_t const v = ld_acquire_sys_u64(addr);
    if (static_cast<uint32_t>(v >> 32) == tag) {
      return static_cast<int32_t>(v & 0xffffffffu);
    }
    __nanosleep(64);
    if (FLASHINFER_SM90_PUSH_CHECK_TIMEOUT(s)) {
      publish_abort_all(L, tag);
      printf("sm90_push: tag wait timed out (addr %p want %u have tag %u value %d)\n",
             static_cast<const void*>(addr), tag, static_cast<uint32_t>(v >> 32),
             static_cast<int32_t>(v & 0xffffffffu));
      asm volatile("trap;");
      return -1;
    }
  }
}

}  // namespace sm90_push
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_SM90_PUSH_A2A_CUH_
