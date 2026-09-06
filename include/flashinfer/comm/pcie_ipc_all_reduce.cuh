/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_ALL_REDUCE_CUH_
#define FLASHINFER_COMM_PCIE_IPC_ALL_REDUCE_CUH_

// Custom all-reduce for intra-node PCIe machines without NVLink.
//
// Every peer transfer on such a machine crosses the CPU root complex, where
// all-to-all writes collapse to a fraction of what the same kernel achieves
// when each rank writes to a single destination. The kernels here therefore
// stage their pushes so that at any instant each rank has exactly one outbound
// and one inbound stream, and the 8-rank path keeps a 4+4 island decomposition
// so the scarce cross-socket links carry the minimum traffic. See the PR
// description for the bandwidth measurements this is derived from; they are a
// property of the machine, not of the code.
//
// All state lives in a caller-owned workspace shared over CUDA IPC; see
// compute_workspace_layout() for the byte layout and make_peer_views() for the
// per-region pointers. The caller owns the allocation because tearing it down
// needs a collective barrier between "every rank unmaps its peers" and "every
// rank frees its own slab", which a destructor cannot express.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace flashinfer {
namespace comm {
namespace pcie_ipc {

constexpr int kMaxWorldSize = 8;
constexpr int kSignalPhases = 8;

// Which kernel all_reduce() launches, together with world_size. Values are
// part of the FFI signature, so they are explicit and append-only.
//
// kFlatStaged is accepted at world_size 8 only: at 4 it would name the same
// kernel as kStaged, and at 2 there is no staged-vs-flat distinction.
enum class Variant : int {
  kUnstaged = 0,    // push to every peer at once
  kStaged = 1,      // staged pushes; island-decomposed at world_size 8
  kStagedRing = 2,  // staged pushes in neighbour order; world_size 4 and 8
  kFlatStaged = 3,  // staged pushes without the island decomposition
};

constexpr int kVariantCount = 4;

// Which staging area a kernel uses. The kernels come in two protocol families
// and a region may hold only one of them.
//
// Sentinel kernels poll for +0.0 meaning "not yet written", sanitise real zeros
// out of the payload, and store +0.0 back once a poll succeeds. Barrier kernels
// are content-blind: they publish raw payload and leave it there. Nothing else
// sweeps the workspace -- the host zeroes it once at init and never again.
//
// So a sentinel kernel landing on a barrier kernel's leftovers reads stale
// payload, and its all-gather poll, which watches a single slot, exits on it
// immediately: wrong output, not a hang. The epoch double buffer does not
// substitute for this -- it guarantees the other half is quiescent, not clean.
//
// At world_size 8 that puts the two topology kernels in kBlock and both
// sentinel kernels in kPack.
enum class ScratchRegion : int { kBlock = 0, kPack = 1 };

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
};

template <typename T>
struct PackTraits {
  static constexpr int kPackElems = 16 / sizeof(T);
  using Pack = Vec<T, kPackElems>;
};

template <bool Enabled>
__device__ __forceinline__ void pdl_grid_sync_const() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Enabled) {
    cudaGridDependencySynchronize();
  }
#endif
}

template <bool Enabled>
__device__ __forceinline__ void pdl_grid_release_const() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Enabled) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
#endif
}

__device__ __forceinline__ void store_release_i32(int32_t* addr, int32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(value), "l"(addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(value), "l"(addr));
#endif
}

__device__ __forceinline__ int32_t load_acquire_i32(int32_t* addr) {
  int32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(value) : "l"(addr));
#endif
  return value;
}

// Has the peer reached generation `expected`?
//
// Generations are a free-running counter, so a plain `observed < expected`
// breaks the first time it wraps: the slot still holds the old maximum, the
// new generation has wrapped to the minimum, and the comparison lets the
// barrier through before the peer has arrived. Switching to unsigned does not
// fix it either -- it just moves the break to UINT_MAX -> 0.
//
// Compare on the circle instead: reinterpret the difference as a signed
// distance modulo 2^32. `observed - expected >= 0` is then true exactly when
// the peer is at or past `expected`, for any pair within 2^31 generations of
// each other -- which is always, since a rank advances one generation per
// call and its peers are at most one call behind.
// constexpr so the boundary behaviour can be pinned at compile time, below.
// Those static_asserts are the whole defence against this being "simplified"
// back to a plain `<`: that version is correct for two billion calls and then
// releases every barrier a generation early, which no test anyone would
// actually run is going to catch.
__host__ __device__ __forceinline__ constexpr bool generation_reached(int32_t observed,
                                                                      int32_t expected) {
  return static_cast<int32_t>(static_cast<uint32_t>(observed) - static_cast<uint32_t>(expected)) >=
         0;
}

static_assert(generation_reached(5, 5), "a peer at the expected generation has arrived");
static_assert(generation_reached(6, 5), "a peer past the expected generation has arrived");
static_assert(!generation_reached(4, 5), "a peer one generation behind has not arrived");
// The wrap that motivated this function. Signed `observed < expected` reads
// INT32_MAX < INT32_MIN as false and lets the barrier through.
static_assert(generation_reached(INT32_MIN, INT32_MAX),
              "the generation after INT32_MAX has arrived");
static_assert(!generation_reached(INT32_MAX, INT32_MIN),
              "the generation before the wrap has not arrived");
// Unsigned `<` fixes the pair above but breaks this one, at UINT32_MAX -> 0.
// Only the modular comparison gets both.
static_assert(generation_reached(0, -1), "0 is one generation past -1");
static_assert(!generation_reached(-1, 0), "-1 is one generation before 0");

__device__ __forceinline__ void store_volatile_i32(int32_t* addr, int32_t value) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(value), "l"(addr));
}

__device__ __forceinline__ int32_t load_volatile_i32(int32_t* addr) {
  int32_t value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ uint32_t add_half2_u32(uint32_t a, uint32_t b) {
  auto ah = *reinterpret_cast<half2*>(&a);
  auto bh = *reinterpret_cast<half2*>(&b);
  half2 out = __hadd2(ah, bh);
  return *reinterpret_cast<uint32_t*>(&out);
}

__device__ __forceinline__ uint32_t add_bfloat162_u32(uint32_t a, uint32_t b) {
  auto ah = *reinterpret_cast<__nv_bfloat162*>(&a);
  auto bh = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 out = __hadd2(ah, bh);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <typename T>
__device__ __forceinline__ uint4 packed_add_u4(uint4 a, uint4 b) {
  if constexpr (std::is_same_v<T, half>) {
    a.x = add_half2_u32(a.x, b.x);
    a.y = add_half2_u32(a.y, b.y);
    a.z = add_half2_u32(a.z, b.z);
    a.w = add_half2_u32(a.w, b.w);
  } else {
    static_assert(std::is_same_v<T, nv_bfloat16>);
    a.x = add_bfloat162_u32(a.x, b.x);
    a.y = add_bfloat162_u32(a.y, b.y);
    a.z = add_bfloat162_u32(a.z, b.z);
    a.w = add_bfloat162_u32(a.w, b.w);
  }
  return a;
}

template <typename T>
__device__ __forceinline__ float2 lane_to_float2(uint32_t lane);

template <>
__device__ __forceinline__ float2 lane_to_float2<half>(uint32_t lane) {
  auto value = *reinterpret_cast<half2*>(&lane);
  return __half22float2(value);
}

template <>
__device__ __forceinline__ float2 lane_to_float2<nv_bfloat16>(uint32_t lane) {
  auto value = *reinterpret_cast<__nv_bfloat162*>(&lane);
  return __bfloat1622float2(value);
}

template <typename T>
__device__ __forceinline__ uint32_t float2_to_lane(float2 value);

template <>
__device__ __forceinline__ uint32_t float2_to_lane<half>(float2 value) {
  half2 out = __float22half2_rn(value);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <>
__device__ __forceinline__ uint32_t float2_to_lane<nv_bfloat16>(float2 value) {
  __nv_bfloat162 out = __float22bfloat162_rn(value);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <typename T, int WorldSize>
__device__ __forceinline__ uint4 reduce_u4_fp32(uint4 const (&values)[WorldSize]) {
  float2 acc0 = {0.0f, 0.0f};
  float2 acc1 = {0.0f, 0.0f};
  float2 acc2 = {0.0f, 0.0f};
  float2 acc3 = {0.0f, 0.0f};
#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer) {
    float2 v0 = lane_to_float2<T>(values[peer].x);
    float2 v1 = lane_to_float2<T>(values[peer].y);
    float2 v2 = lane_to_float2<T>(values[peer].z);
    float2 v3 = lane_to_float2<T>(values[peer].w);
    acc0.x += v0.x;
    acc0.y += v0.y;
    acc1.x += v1.x;
    acc1.y += v1.y;
    acc2.x += v2.x;
    acc2.y += v2.y;
    acc3.x += v3.x;
    acc3.y += v3.y;
  }
  uint4 out;
  out.x = float2_to_lane<T>(acc0);
  out.y = float2_to_lane<T>(acc1);
  out.z = float2_to_lane<T>(acc2);
  out.w = float2_to_lane<T>(acc3);
  return out;
}

template <typename T>
struct ZeroBits;

template <>
struct ZeroBits<half> {
  using Raw = uint16_t;
  static constexpr Raw kPos = 0x0000u;
  static constexpr Raw kNeg = 0x8000u;
};

template <>
struct ZeroBits<nv_bfloat16> {
  using Raw = uint16_t;
  static constexpr Raw kPos = 0x0000u;
  static constexpr Raw kNeg = 0x8000u;
};

template <>
struct ZeroBits<float> {
  using Raw = uint32_t;
  static constexpr Raw kPos = 0x00000000u;
  static constexpr Raw kNeg = 0x80000000u;
};

template <typename T>
__device__ __forceinline__ void clear_pos_zero(T& value) {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw* raw = reinterpret_cast<Raw*>(&value);
  if (*raw == Bits::kPos) {
    *raw = Bits::kNeg;
  }
}

template <typename T>
__device__ __forceinline__ bool is_pos_zero(T value) {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw raw = *reinterpret_cast<Raw*>(&value);
  return raw == Bits::kPos;
}

template <typename T>
__device__ __forceinline__ T pos_zero() {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw raw = Bits::kPos;
  return *reinterpret_cast<T*>(&raw);
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack load_pack_volatile(
    typename PackTraits<T>::Pack const* base, int idx) {
  uint4 raw;
  auto const* addr = reinterpret_cast<uint4 const*>(base + idx);
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
               : "l"(addr));
  return *reinterpret_cast<typename PackTraits<T>::Pack*>(&raw);
}

template <typename T>
__device__ __forceinline__ void store_pack_volatile(typename PackTraits<T>::Pack* base, int idx,
                                                    typename PackTraits<T>::Pack value) {
  uint4 raw = *reinterpret_cast<uint4*>(&value);
  auto* addr = reinterpret_cast<uint4*>(base + idx);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(raw.x), "r"(raw.y),
               "r"(raw.z), "r"(raw.w), "l"(addr));
}

template <typename T>
__device__ __forceinline__ void clear_pos_zero_pack(typename PackTraits<T>::Pack& pack) {
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    clear_pos_zero(pack.data[i]);
  }
}

template <typename T>
__device__ __forceinline__ bool has_pos_zero_pack(typename PackTraits<T>::Pack const& pack) {
  bool has_zero = false;
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    has_zero |= is_pos_zero(pack.data[i]);
  }
  return has_zero;
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack zero_pack() {
  typename PackTraits<T>::Pack pack;
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    pack.data[i] = pos_zero<T>();
  }
  return pack;
}

__device__ __forceinline__ uint32_t clear_pos_zero_u16x2(uint32_t raw) {
  uint32_t lo = raw & 0xffffu;
  uint32_t hi = raw & 0xffff0000u;
  if (lo == 0u) {
    lo = 0x8000u;
  }
  if (hi == 0u) {
    hi = 0x80000000u;
  }
  return hi | lo;
}

__device__ __forceinline__ bool has_pos_zero_u16x2(uint32_t raw) {
  return (raw & 0xffffu) == 0u || (raw & 0xffff0000u) == 0u;
}

__device__ __forceinline__ uint4 clear_pos_zero_u4_16(uint4 value) {
  value.x = clear_pos_zero_u16x2(value.x);
  value.y = clear_pos_zero_u16x2(value.y);
  value.z = clear_pos_zero_u16x2(value.z);
  value.w = clear_pos_zero_u16x2(value.w);
  return value;
}

__device__ __forceinline__ bool has_pos_zero_u4_16(uint4 value) {
  return has_pos_zero_u16x2(value.x) || has_pos_zero_u16x2(value.y) ||
         has_pos_zero_u16x2(value.z) || has_pos_zero_u16x2(value.w);
}

__device__ __forceinline__ uint4 load_u4_volatile(uint4 const* base, int idx) {
  uint4 value;
  auto const* addr = base + idx;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(addr));
  return value;
}

__device__ __forceinline__ void store_u4_volatile(uint4* base, int idx, uint4 value) {
  auto* addr = base + idx;
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(value.x), "r"(value.y),
               "r"(value.z), "r"(value.w), "l"(addr));
}

__device__ __forceinline__ int phase_offset(int phase, int block, int peer, int max_blocks,
                                            int world_size) {
  // Epoch slots occupy [0, max_blocks). Barrier slots start after that
  // dedicated prefix so phase 0 can never corrupt an epoch.
  return max_blocks + phase * max_blocks * world_size + block * world_size + peer;
}

__device__ __forceinline__ int flag_offset(int block, int max_blocks, int world_size) {
  return max_blocks + kSignalPhases * max_blocks * world_size + block;
}

// Call-level double-buffer state: the epoch at [0] and the arrival counter at
// [1], placed past the flag region so no existing offset moves.
//
// ONE PAIR PER SCRATCH REGION. The hard constraint runs one way only:
//
// **Kernels writing the same region MUST share a counter.** TP4 alternates
// ipc_rsag_push and ipc_rsag_ring with the payload and both write the block
// region at the same addresses, so private counters would let a ring call on
// half 0 be followed immediately by a push call that also reads 0 -- back to
// back, with nothing in between to drain the first.
//
// Kernels in *different* regions need not be: any intervening collective
// already drains the previous one. They are kept apart anyway, because binding
// the state on the same host line that picks the region makes the two
// impossible to get out of step.
//
// Rank-local, and indexed by nothing, so every CTA of a launch agrees on the
// half regardless of gridDim. Per-block parity cannot: it counts how many times
// *that block* has run, so one change in gridDim desynchronises the block ranges
// permanently and a block picks a half another block is still using.
//
// Consistency across ranks comes from the same SPMD argument that already
// backs the per-block barrier flags: every rank runs the same sequence of
// collectives, so every rank is on the same call parity.
__host__ __device__ __forceinline__ int scratch_state_offset(int max_blocks, int world_size,
                                                             ScratchRegion region) {
  return max_blocks + kSignalPhases * max_blocks * world_size + max_blocks +
         2 * static_cast<int>(region);
}

// Debug only: pin every call to half 0, i.e. the pre-double-buffer behaviour.
// Kept because the cross-island race is invisible without a way to build the
// broken protocol on demand -- it is what proves a repro actually has power,
// and it isolates the cost of the double buffer in a single benchmark session.
// Never define this in a shipping build.
// Debug only: remove the leading CTA barrier from the three signalling helpers.
// The resulting build is incorrect -- a warp can announce "my stage is done"
// while its siblings are still writing. Never define this in a shipping build.
#ifndef FLASHINFER_PCIE_IPC_DEBUG_NO_BARRIER_ENTRY_SYNC
#define FLASHINFER_PCIE_IPC_DEBUG_NO_BARRIER_ENTRY_SYNC 0
#endif

#ifndef FLASHINFER_PCIE_IPC_DEBUG_NO_BLOCK_EPOCH
#define FLASHINFER_PCIE_IPC_DEBUG_NO_BLOCK_EPOCH 0
#endif

// Debug only: restore the per-block epoch parity these kernels used before the
// call-level counter replaced it. This is the negative control for the
// grid-change regression -- without it, a passing test cannot distinguish "the
// fix works" from "the sequence never opened the window". Distinct from
// NO_BLOCK_EPOCH above, which removes double buffering entirely; this one keeps
// two halves and only breaks the agreement about which half a call is on.
// Never define this in a shipping build.
#ifndef FLASHINFER_PCIE_IPC_DEBUG_PER_BLOCK_EPOCH
#define FLASHINFER_PCIE_IPC_DEBUG_PER_BLOCK_EPOCH 0
#endif

// Read this call's epoch and advance it, both at kernel entry.
//
// Advancing at entry rather than at exit is what keeps this affordable. The
// flip only has to follow every block's *read*, not every block's work: the
// state is rank-local (views.self_signal), so its only reader is the next
// kernel on this stream, and that cannot start until this launch has fully
// retired. The last block to arrive therefore knows every peer block has
// already read, and can flip immediately.
//
// Committing at exit instead would put a tail __syncthreads() plus an L2 atomic
// round trip on the block-retirement critical path. It would also be fragile:
// an early return added after the arrival would freeze the counter below
// gridDim.x - 1 and silently pin the epoch.
//
// Electing the last arrival deliberately does not depend on block scheduling
// order or on the whole grid being resident. "Block 0 flips" would need every
// block launched before block 0 reaches this point, which CUDA does not
// promise, and a grid-wide spin barrier deadlocks once gridDim exceeds
// occupancy.
//
// At gridDim.x == 1 there is nothing to elect, so the atomic is skipped; the
// end state is identical (counter 0, epoch flipped).
// Exit half of the PER_BLOCK_EPOCH debug build; compiles to nothing otherwise.
// The flip sits at the exit because that is where the implementation it rebuilds
// put it, and entry-versus-exit changes the cost.
__device__ __forceinline__ void debug_commit_per_block_epoch(int32_t* per_block_slot, int epoch) {
#if FLASHINFER_PCIE_IPC_DEBUG_PER_BLOCK_EPOCH
  // Bare -- no __syncthreads(), matching the implementation this rebuilds.
  if (threadIdx.x == 0) {
    store_volatile_i32(per_block_slot, epoch ^ 1);
  }
#else
  (void)per_block_slot;
  (void)epoch;
#endif
}

__device__ __forceinline__ int advance_scratch_epoch(int32_t* state, int32_t* per_block_slot) {
#if FLASHINFER_PCIE_IPC_DEBUG_NO_BLOCK_EPOCH
  (void)state;
  (void)per_block_slot;
  return 0;
#elif FLASHINFER_PCIE_IPC_DEBUG_PER_BLOCK_EPOCH
  // Read only; the flip is issued at kernel exit by
  // debug_commit_per_block_epoch().
  (void)state;
  return load_volatile_i32(per_block_slot) & 1;
#else
  (void)per_block_slot;
  const int epoch = load_volatile_i32(state) & 1;
  // Every thread must have read before this block announces its arrival.
  __syncthreads();
  if (threadIdx.x == 0) {
    // Device scope, not system scope. This state is rank-local -- its only
    // reader is this rank's next kernel on this stream -- and that reader wants
    // nothing but the value itself, so there is nothing for a release to order.
    // st.release.sys would flush this thread's writes system-wide over PCIe on
    // the entry critical path.
    if (gridDim.x == 1) {
      // Sole CTA: trivially the last arrival, and the counter is already 0.
      store_volatile_i32(state, epoch ^ 1);
    } else if (atomicAdd(state + 1, 1) == static_cast<int32_t>(gridDim.x) - 1) {
      store_volatile_i32(state + 1, 0);
      store_volatile_i32(state, epoch ^ 1);
    }
  }
  return epoch;
#endif
}

__device__ __forceinline__ void block_barrier(uint64_t const* signal_ptrs, int rank, int world_size,
                                              int max_blocks, int phase, int flag) {
  // Publishing a signal means "this CTA's writes for the previous stage are
  // done", so every thread must have finished them before the signalling
  // threads announce it. The call sites use __threadfence_system(), which
  // orders only the *calling* thread's accesses -- it does not wait for the
  // rest of the CTA, and cannot help at all where the previous stage was a load
  // loop. Only a CTA barrier establishes that.
#if !FLASHINFER_PCIE_IPC_DEBUG_NO_BARRIER_ENTRY_SYNC
  __syncthreads();
#endif
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x < world_size) {
    int peer = threadIdx.x;
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_release_i32(peer_signal + phase_offset(phase, block, rank, max_blocks, world_size), flag);
    int32_t* self_slot = self + phase_offset(phase, block, peer, max_blocks, world_size);
    while (!generation_reached(load_acquire_i32(self_slot), flag)) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void block_barrier_mask(uint64_t const* signal_ptrs, int rank,
                                                   int world_size, int max_blocks, int phase,
                                                   int flag, uint32_t participant_mask) {
  // Entry barrier: see block_barrier above.
#if !FLASHINFER_PCIE_IPC_DEBUG_NO_BARRIER_ENTRY_SYNC
  __syncthreads();
#endif
  if ((participant_mask & (1u << rank)) == 0u) {
    __syncthreads();
    return;
  }
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x < world_size) {
    int peer = threadIdx.x;
    if ((participant_mask & (1u << peer)) != 0u) {
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_release_i32(peer_signal + phase_offset(phase, block, rank, max_blocks, world_size),
                        flag);
      int32_t* self_slot = self + phase_offset(phase, block, peer, max_blocks, world_size);
      while (!generation_reached(load_acquire_i32(self_slot), flag)) {
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void island_owner_gather(uint64_t const* signal_ptrs, int rank, int base,
                                                    int owner, int max_blocks, int phase,
                                                    int flag) {
  // Entry barrier: see block_barrier above.
#if !FLASHINFER_PCIE_IPC_DEBUG_NO_BARRIER_ENTRY_SYNC
  __syncthreads();
#endif
  int block = blockIdx.x;
  int32_t* owner_signal = reinterpret_cast<int32_t*>(signal_ptrs[owner]);
  if (threadIdx.x == 0) {
    store_release_i32(owner_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
  }
  if (rank == owner && threadIdx.x < 4) {
    int peer = base + threadIdx.x;
    int32_t* self_slot = owner_signal + phase_offset(phase, block, peer, max_blocks, 8);
    while (!generation_reached(load_acquire_i32(self_slot), flag)) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void owner_pair_barrier(uint64_t const* signal_ptrs, int rank, int owner,
                                                   int cross_owner, int max_blocks, int phase,
                                                   int flag) {
  __syncthreads();
  if (rank == owner && threadIdx.x == 0) {
    int block = blockIdx.x;
    int32_t* cross_signal = reinterpret_cast<int32_t*>(signal_ptrs[cross_owner]);
    store_release_i32(cross_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
    int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
    int32_t* self_slot = self_signal + phase_offset(phase, block, cross_owner, max_blocks, 8);
    while (!generation_reached(load_acquire_i32(self_slot), flag)) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void island_owner_ready(uint64_t const* signal_ptrs, int rank, int base,
                                                   int owner, int max_blocks, int phase, int flag) {
  __syncthreads();
  int block = blockIdx.x;
  if (rank == owner) {
    if (threadIdx.x < 4) {
      int peer = base + threadIdx.x;
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_release_i32(peer_signal + phase_offset(phase, block, owner, max_blocks, 8), flag);
    }
  } else if (threadIdx.x == 0) {
    int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
    int32_t* self_slot = self_signal + phase_offset(phase, block, owner, max_blocks, 8);
    while (!generation_reached(load_acquire_i32(self_slot), flag)) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void island_owner_ack(uint64_t const* signal_ptrs, int rank, int base,
                                                 int owner, int max_blocks, int phase, int flag) {
  __syncthreads();
  int block = blockIdx.x;
  int32_t* owner_signal = reinterpret_cast<int32_t*>(signal_ptrs[owner]);
  if (rank != owner && threadIdx.x == 0) {
    store_release_i32(owner_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
  }
  if (rank == owner && threadIdx.x < 4) {
    int peer = base + threadIdx.x;
    if (peer != owner) {
      int32_t* self_slot = owner_signal + phase_offset(phase, block, peer, max_blocks, 8);
      while (!generation_reached(load_acquire_i32(self_slot), flag)) {
      }
    }
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack add_pack(typename PackTraits<T>::Pack a,
                                                                 typename PackTraits<T>::Pack b) {
  using Traits = PackTraits<T>;
  using Pack = typename Traits::Pack;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 av = *reinterpret_cast<uint4*>(&a);
    uint4 bv = *reinterpret_cast<uint4*>(&b);
    uint4 out = packed_add_u4<T>(av, bv);
    return *reinterpret_cast<Pack*>(&out);
  }

  Pack out;
#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    out.data[i] = from_float<T>(to_float(a.data[i]) + to_float(b.data[i]));
  }
  return out;
}

template <typename T, int WorldSize>
__device__ __forceinline__ typename PackTraits<T>::Pack reduce_loaded_packs(
    typename PackTraits<T>::Pack const (&values)[WorldSize]) {
  using Pack = typename PackTraits<T>::Pack;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 acc = *reinterpret_cast<uint4 const*>(&values[0]);
#pragma unroll
    for (int peer = 1; peer < WorldSize; ++peer) {
      uint4 next = *reinterpret_cast<uint4 const*>(&values[peer]);
      acc = packed_add_u4<T>(acc, next);
    }
    return *reinterpret_cast<Pack*>(&acc);
  } else {
    Pack acc = values[0];
#pragma unroll
    for (int peer = 1; peer < WorldSize; ++peer) {
      acc = add_pack<T>(acc, values[peer]);
    }
    return acc;
  }
}

// Debug-only hook for reproducing the cross-island scratch race.
//
// The hazard needs the SLOW island to still be reading the cross slot while
// the FAST island's next call overwrites it. Delaying a whole kernel launch
// from the host cannot produce that: the pair barrier releases both islands
// together, after which the reader reaches its cross read almost immediately
// while the writer still has several phases to go. The stall has to be here,
// between the pair rendezvous and the cross read.
//
// Enabled only when FLASHINFER_PCIE_IPC_DEBUG_CROSS_STALL_NS is defined, and
// only on the island selected by ..._STALL_ISLAND. Never define these in a
// shipping build.
#ifndef FLASHINFER_PCIE_IPC_DEBUG_CROSS_STALL_NS
#define FLASHINFER_PCIE_IPC_DEBUG_CROSS_STALL_NS 0
#endif
#ifndef FLASHINFER_PCIE_IPC_DEBUG_STALL_ISLAND
#define FLASHINFER_PCIE_IPC_DEBUG_STALL_ISLAND 0
#endif

__device__ __forceinline__ void debug_cross_read_stall(int rank) {
#if FLASHINFER_PCIE_IPC_DEBUG_CROSS_STALL_NS > 0
  const int island = rank < 4 ? 0 : 1;
  if (island == FLASHINFER_PCIE_IPC_DEBUG_STALL_ISLAND) {
    __nanosleep(FLASHINFER_PCIE_IPC_DEBUG_CROSS_STALL_NS);
  }
  __syncthreads();
#else
  (void)rank;
#endif
}

template <typename T>
struct PushOneshotParamData {
  uint64_t tmp_ptrs[kMaxWorldSize];
  uint64_t signal_ptrs[kMaxWorldSize];
  T const* input;
  T* output;
  int32_t* epoch_slots;
  // Call-level double-buffer state for this launch's scratch region, bound
  // host-side so the state and the region cannot disagree. See
  // scratch_state_offset().
  int32_t* scratch_state;
  int num_packs;
  int rank_stride_packs;
  int epoch_stride_packs;
  int rank;
  int max_blocks;
};

template <typename T>
struct IpcTp2RemotePushData {
  uint64_t tmp_ptrs[2];
  T const* input;
  T* output;
  int32_t* epoch_slots;
  // Call-level double-buffer state for this launch's scratch region, bound
  // host-side so the state and the region cannot disagree. See
  // scratch_state_offset().
  int32_t* scratch_state;
  int num_packs;
  // Half-size of the epoch double buffer, in packs. Derived from max_numel,
  // NOT from this call's num_packs: the two epoch halves must sit at fixed
  // addresses. If they moved with the payload, a rank that finished a large
  // collective and flipped its epoch would start writing a small one inside
  // the region a lagging peer is still draining -- corrupting it, or having
  // the peer's reset wipe the just-published data so the poll never ends.
  // Every other v2 kernel already derives its stage offset this way.
  int rank_stride_packs;
  int rank;
};

template <typename T, bool Stream, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_tp2_remote_push_kernel(
    const IpcTp2RemotePushData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = params.rank ^ 1;
  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  int stage_offset = epoch * 2 * params.rank_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    int peer_write_base = stage_offset + params.rank * params.num_packs;
    int local_poll_base = stage_offset + peer * params.num_packs;
    uint4 reset = {0u, 0u, 0u, 0u};
    if constexpr (Stream) {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 local_value = input[idx];
        uint4 publish_value = clear_pos_zero_u4_16(local_value);
        store_u4_volatile(peer_buffer, peer_write_base + idx, publish_value);
        uint4 peer_value;
        while (true) {
          peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_u4_16(peer_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] = packed_add_u4<T>(local_value, peer_value);
        store_u4_volatile(local_buffer, local_poll_base + idx, reset);
      }
    } else {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 value = clear_pos_zero_u4_16(input[idx]);
        store_u4_volatile(peer_buffer, peer_write_base + idx, value);
      }
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 peer_value;
        while (true) {
          peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_u4_16(peer_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] = packed_add_u4<T>(input[idx], peer_value);
        store_u4_volatile(local_buffer, local_poll_base + idx, reset);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    int peer_write_base = stage_offset + params.rank * params.num_packs;
    int local_poll_base = stage_offset + peer * params.num_packs;
    Pack reset = zero_pack<T>();
    if constexpr (Stream) {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack local_value = input[idx];
        Pack publish_value = local_value;
        clear_pos_zero_pack<T>(publish_value);
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx, publish_value);
        Pack peer_value;
        while (true) {
          peer_value = load_pack_volatile<T>(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_pack<T>(peer_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] = add_pack<T>(local_value, peer_value);
        store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
      }
    } else {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack value = input[idx];
        clear_pos_zero_pack<T>(value);
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx, value);
      }
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack peer_value;
        while (true) {
          peer_value = load_pack_volatile<T>(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_pack<T>(peer_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] = add_pack<T>(input[idx], peer_value);
        store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
      }
    }
  }

  debug_commit_per_block_epoch(epoch_slot, epoch);
  pdl_grid_release_const<UsePdl>();
}

// Owner of a pack under the reduce-scatter split. Must agree with the explicit
// chunk ranges the same kernels walk, which give the remainder to the last rank
// -- so `part == 0` (fewer packs than ranks) gives it the whole payload too.
// Writing to one owner and polling another spins forever: no timeout here.
template <int WorldSize>
__device__ __forceinline__ int rsag_owner_for_pack(int idx, int part) {
  int owner = part > 0 ? idx / part : WorldSize - 1;
  return owner < WorldSize ? owner : WorldSize - 1;
}

// Staged (neighbour-ordered) RS/AG push.
//
// ipc_rsag_push_param_kernel below has every rank writing to all WorldSize-1
// peers at the same time. Where every peer transfer crosses the CPU root
// complex that pattern runs far below what the same kernel-issued writes reach
// when a rank has a single outbound destination -- the collective is limited by
// the shape of the traffic, not the amount.
//
// This variant keeps the algorithm and the total bytes identical and only
// reorders the pushes: each push phase is split into WorldSize-1 passes, and in
// pass p every rank writes solely to peer (rank + 1 + p) % WorldSize. That map
// is a permutation, so during a pass every GPU has exactly one outbound stream
// and every GPU is the target of exactly one -- the pattern the fabric
// sustains. A barrier between passes keeps the ranks in the same pass, since
// without it they drift and the passes overlap back into all-to-all.
//
// The cost is 2*(WorldSize-1) barriers per collective, which is why the policy
// only selects this variant once the payload is large enough to pay for them.
// Keep the established phase numbering for this kernel. Epoch slots and
// barrier phases have separate ranges in the signal region (see
// phase_offset), so phase 1 is no longer needed for alias avoidance.
constexpr int kRingRsPhase0 = 1;

template <typename T, int WorldSize, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_rsag_ring_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  // RS uses phases [1, WorldSize-1], AG uses [WorldSize, 2*WorldSize-2].
  static_assert(2 * WorldSize - 2 < kSignalPhases,
                "ring push needs 2*(WorldSize-1) barrier phases");
  pdl_grid_sync_const<UsePdl>();

  int32_t* self_signal = reinterpret_cast<int32_t*>(params.signal_ptrs[params.rank]);
  int flag = static_cast<int32_t>(
      static_cast<uint32_t>(
          load_acquire_i32(self_signal + flag_offset(blockIdx.x, params.max_blocks, WorldSize))) +
      1u);
  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = params.num_packs / WorldSize;
  int my_start = params.rank * part;
  int my_end = (params.rank == WorldSize - 1) ? params.num_packs : my_start + part;
  int my_slot = stage_offset + params.rank * params.rank_stride_packs;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    uint4 reset = {0u, 0u, 0u, 0u};

    // Reduce-scatter, own chunk: stays on this GPU, so it costs no fabric time
    // and does not need a pass of its own.
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      store_u4_volatile(local_buffer, my_slot + idx, clear_pos_zero_u4_16(input[idx]));
    }

    // Reduce-scatter, staged: one destination per pass.
    for (int p = 0; p < WorldSize - 1; ++p) {
      int target = (params.rank + 1 + p) % WorldSize;
      int t_start = target * part;
      int t_end = (target == WorldSize - 1) ? params.num_packs : t_start + part;
      auto* target_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[target]);
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        store_u4_volatile(target_buffer, my_slot + idx, clear_pos_zero_u4_16(input[idx]));
      }
      block_barrier(params.signal_ptrs, params.rank, WorldSize, params.max_blocks,
                    kRingRsPhase0 + p, flag);
    }

    // Owner reduce: every contribution is now in this rank's own buffer, so
    // this phase touches local memory only. The reduced value is stashed back
    // into this rank's own slot for the all-gather passes to re-read.
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      uint4 acc = values[0];
#pragma unroll
      for (int peer = 1; peer < WorldSize; ++peer) {
        acc = packed_add_u4<T>(acc, values[peer]);
      }
      reinterpret_cast<uint4*>(params.output)[idx] = acc;
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        if (peer != params.rank) {
          store_u4_volatile(local_buffer, stage_offset + peer * params.rank_stride_packs + idx,
                            reset);
        }
      }
      store_u4_volatile(local_buffer, my_slot + idx, clear_pos_zero_u4_16(acc));
    }

    // All-gather, staged: same permutation schedule as the reduce-scatter.
    for (int p = 0; p < WorldSize - 1; ++p) {
      int target = (params.rank + 1 + p) % WorldSize;
      auto* target_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[target]);
      for (int idx = my_start + tid; idx < my_end; idx += stride) {
        store_u4_volatile(target_buffer, my_slot + idx,
                          load_u4_volatile(local_buffer, my_slot + idx));
      }
      block_barrier(params.signal_ptrs, params.rank, WorldSize, params.max_blocks, WorldSize + p,
                    flag);
    }

    // Consume the chunks owned by others, then clear this rank's own slot so
    // the sentinel state is clean for the epoch that reuses it.
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      uint4 value;
      while (true) {
        value = load_u4_volatile(local_buffer, offset);
        if (!has_pos_zero_u4_16(value)) {
          break;
        }
      }
      reinterpret_cast<uint4*>(params.output)[idx] = value;
      store_u4_volatile(local_buffer, offset, reset);
    }
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      store_u4_volatile(local_buffer, my_slot + idx, reset);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    Pack reset = zero_pack<T>();

    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
      store_pack_volatile<T>(local_buffer, my_slot + idx, value);
    }
    for (int p = 0; p < WorldSize - 1; ++p) {
      int target = (params.rank + 1 + p) % WorldSize;
      int t_start = target * part;
      int t_end = (target == WorldSize - 1) ? params.num_packs : t_start + part;
      auto* target_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[target]);
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        Pack value = input[idx];
        clear_pos_zero_pack<T>(value);
        store_pack_volatile<T>(target_buffer, my_slot + idx, value);
      }
      block_barrier(params.signal_ptrs, params.rank, WorldSize, params.max_blocks,
                    kRingRsPhase0 + p, flag);
    }
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(params.output)[idx] = acc;
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        if (peer != params.rank) {
          store_pack_volatile<T>(local_buffer, stage_offset + peer * params.rank_stride_packs + idx,
                                 reset);
        }
      }
      Pack publish = acc;
      clear_pos_zero_pack<T>(publish);
      store_pack_volatile<T>(local_buffer, my_slot + idx, publish);
    }
    for (int p = 0; p < WorldSize - 1; ++p) {
      int target = (params.rank + 1 + p) % WorldSize;
      auto* target_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[target]);
      for (int idx = my_start + tid; idx < my_end; idx += stride) {
        store_pack_volatile<T>(target_buffer, my_slot + idx,
                               load_pack_volatile<T>(local_buffer, my_slot + idx));
      }
      block_barrier(params.signal_ptrs, params.rank, WorldSize, params.max_blocks, WorldSize + p,
                    flag);
    }
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      Pack value;
      while (true) {
        value = load_pack_volatile<T>(local_buffer, offset);
        if (!has_pos_zero_pack<T>(value)) {
          break;
        }
      }
      reinterpret_cast<Pack*>(params.output)[idx] = value;
      store_pack_volatile<T>(local_buffer, offset, reset);
    }
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      store_pack_volatile<T>(local_buffer, my_slot + idx, reset);
    }
  }

  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, params.max_blocks, WorldSize), flag);
  }
  debug_commit_per_block_epoch(epoch_slot, epoch);
  pdl_grid_release_const<UsePdl>();
}

template <typename T, int WorldSize, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_rsag_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = params.num_packs / WorldSize;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    uint4 reset = {0u, 0u, 0u, 0u};

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      uint4 value = clear_pos_zero_u4_16(input[idx]);
      store_u4_volatile(owner_buffer, offset, value);
    }

    int start = params.rank * part;
    int end = (params.rank == WorldSize - 1) ? params.num_packs : start + part;
    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    for (int idx = start + tid; idx < end; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      uint4 acc = values[0];
#pragma unroll
      for (int peer = 1; peer < WorldSize; ++peer) {
        acc = packed_add_u4<T>(acc, values[peer]);
      }
      reinterpret_cast<uint4*>(params.output)[idx] = acc;
      uint4 publish = clear_pos_zero_u4_16(acc);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int offset = stage_offset + peer * params.rank_stride_packs + idx;
        store_u4_volatile(local_buffer, offset, reset);
        if (peer != params.rank) {
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, publish);
        }
      }
    }

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      uint4 value;
      while (true) {
        value = load_u4_volatile(local_buffer, offset);
        if (!has_pos_zero_u4_16(value)) {
          break;
        }
      }
      reinterpret_cast<uint4*>(params.output)[idx] = value;
      store_u4_volatile(local_buffer, offset, reset);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    Pack reset = zero_pack<T>();

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
      store_pack_volatile<T>(owner_buffer, offset, value);
    }

    int start = params.rank * part;
    int end = (params.rank == WorldSize - 1) ? params.num_packs : start + part;
    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(params.output)[idx] = acc;
      Pack publish = acc;
      clear_pos_zero_pack<T>(publish);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int offset = stage_offset + peer * params.rank_stride_packs + idx;
        store_pack_volatile<T>(local_buffer, offset, reset);
        if (peer != params.rank) {
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, publish);
        }
      }
    }

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      Pack value;
      while (true) {
        value = load_pack_volatile<T>(local_buffer, offset);
        if (!has_pos_zero_pack<T>(value)) {
          break;
        }
      }
      reinterpret_cast<Pack*>(params.output)[idx] = value;
      store_pack_volatile<T>(local_buffer, offset, reset);
    }
  }
  debug_commit_per_block_epoch(epoch_slot, epoch);
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_topo_rsag8_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = params.num_packs / 4;
  int base = params.rank < 4 ? 0 : 4;
  int cross_base = base ^ 4;
  int local_rank = params.rank - base;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    uint4 reset = {0u, 0u, 0u, 0u};

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int chunk = rsag_owner_for_pack<4>(idx, part);
      int owner = base + chunk;
      auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
      int input_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      uint4 local_value = input[idx];
      uint4 publish_input = clear_pos_zero_u4_16(local_value);
      store_u4_volatile(owner_buffer, input_offset, publish_input);

      if (local_rank == chunk) {
        uint4 values[4];
        while (true) {
          bool waiting = false;
#pragma unroll
          for (int peer_local = 0; peer_local < 4; ++peer_local) {
            int peer = base + peer_local;
            int offset = stage_offset + peer * params.rank_stride_packs + idx;
            values[peer_local] = load_u4_volatile(local_buffer, offset);
            waiting |= has_pos_zero_u4_16(values[peer_local]);
          }
          if (!waiting) {
            break;
          }
        }
        uint4 local_sum = values[0];
#pragma unroll
        for (int peer_local = 1; peer_local < 4; ++peer_local) {
          local_sum = packed_add_u4<T>(local_sum, values[peer_local]);
        }
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          store_u4_volatile(local_buffer, offset, reset);
        }

        int cross_owner = cross_base + chunk;
        auto* cross_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[cross_owner]);
        int cross_write = stage_offset + params.rank * params.rank_stride_packs + idx;
        uint4 publish_sum = clear_pos_zero_u4_16(local_sum);
        store_u4_volatile(cross_buffer, cross_write, publish_sum);

        int cross_read = stage_offset + cross_owner * params.rank_stride_packs + idx;
        uint4 cross_sum;
        while (true) {
          cross_sum = load_u4_volatile(local_buffer, cross_read);
          if (!has_pos_zero_u4_16(cross_sum)) {
            break;
          }
        }
        uint4 final_value = packed_add_u4<T>(local_sum, cross_sum);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
        store_u4_volatile(local_buffer, cross_read, reset);

        uint4 publish_final = clear_pos_zero_u4_16(final_value);
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, publish_final);
        }
      } else {
        int final_offset = stage_offset + owner * params.rank_stride_packs + idx;
        uint4 final_value;
        while (true) {
          final_value = load_u4_volatile(local_buffer, final_offset);
          if (!has_pos_zero_u4_16(final_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
        store_u4_volatile(local_buffer, final_offset, reset);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    Pack reset = zero_pack<T>();

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int chunk = rsag_owner_for_pack<4>(idx, part);
      int owner = base + chunk;
      auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
      int input_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      Pack local_value = input[idx];
      Pack publish_input = local_value;
      clear_pos_zero_pack<T>(publish_input);
      store_pack_volatile<T>(owner_buffer, input_offset, publish_input);

      if (local_rank == chunk) {
        Pack values[4];
        while (true) {
          bool waiting = false;
#pragma unroll
          for (int peer_local = 0; peer_local < 4; ++peer_local) {
            int peer = base + peer_local;
            int offset = stage_offset + peer * params.rank_stride_packs + idx;
            values[peer_local] = load_pack_volatile<T>(local_buffer, offset);
            waiting |= has_pos_zero_pack<T>(values[peer_local]);
          }
          if (!waiting) {
            break;
          }
        }
        Pack local_sum = values[0];
#pragma unroll
        for (int peer_local = 1; peer_local < 4; ++peer_local) {
          local_sum = add_pack<T>(local_sum, values[peer_local]);
        }
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          store_pack_volatile<T>(local_buffer, offset, reset);
        }

        int cross_owner = cross_base + chunk;
        auto* cross_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[cross_owner]);
        int cross_write = stage_offset + params.rank * params.rank_stride_packs + idx;
        Pack publish_sum = local_sum;
        clear_pos_zero_pack<T>(publish_sum);
        store_pack_volatile<T>(cross_buffer, cross_write, publish_sum);

        int cross_read = stage_offset + cross_owner * params.rank_stride_packs + idx;
        Pack cross_sum;
        while (true) {
          cross_sum = load_pack_volatile<T>(local_buffer, cross_read);
          if (!has_pos_zero_pack<T>(cross_sum)) {
            break;
          }
        }
        Pack final_value = add_pack<T>(local_sum, cross_sum);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
        store_pack_volatile<T>(local_buffer, cross_read, reset);

        Pack publish_final = final_value;
        clear_pos_zero_pack<T>(publish_final);
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, publish_final);
        }
      } else {
        int final_offset = stage_offset + owner * params.rank_stride_packs + idx;
        Pack final_value;
        while (true) {
          final_value = load_pack_volatile<T>(local_buffer, final_offset);
          if (!has_pos_zero_pack<T>(final_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
        store_pack_volatile<T>(local_buffer, final_offset, reset);
      }
    }
  }

  debug_commit_per_block_epoch(epoch_slot, epoch);
  pdl_grid_release_const<UsePdl>();
}

// TP8 staged topology RS/AG push.
//
// ipc_topo_rsag8_block_param_kernel below partitions blocks by blockIdx.x & 3,
// so its four block groups push to four different island owners at the same
// instant: intra-island traffic is all-to-all, which is the expensive pattern
// on this fabric.
//
// This variant keeps the topology decomposition exactly as it is (island reduce
// -> owner-pair exchange across SYS -> island gather), because a topology-blind
// ring is slower at every block count. It only re-times the two
// intra-island phases: each is split into three passes, and in pass p a rank
// talks solely to island peer (local + 1 + p) % 4, which is a permutation, so
// each GPU has one outbound stream at a time. The cross-island exchange is
// already one-to-one and is left alone.
//
// Because chunks are now visited in time rather than assigned to block groups,
// blocks no longer have to be a multiple of four and a flat grid-stride loop
// covers each chunk.
//
// Costs six extra island barriers, so the policy only selects this above a
// payload threshold.
template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_topo_rsag8_ring_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  static_assert(kSignalPhases >= 8, "topology ring push needs eight barrier phases");
  pdl_grid_sync_const<UsePdl>();

  int32_t* self_signal = reinterpret_cast<int32_t*>(params.signal_ptrs[params.rank]);
  // Unsigned arithmetic for the bump: signed overflow is UB, and this counter
  // is meant to wrap. generation_reached() reads it back on the circle.
  int flag =
      static_cast<int32_t>(static_cast<uint32_t>(load_acquire_i32(
                               self_signal + flag_offset(blockIdx.x, params.max_blocks, 8))) +
                           1u);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int part = params.num_packs >> 2;
  const int base = params.rank < 4 ? 0 : 4;
  const int local = params.rank & 3;
  const uint32_t island_mask = params.rank < 4 ? 0x0fu : 0xf0u;
  const int cross_owner = params.rank ^ 4;
  // Call-level double buffer. The cross-island payload this rank publishes
  // into its paired owner's slab has no read-complete edge coming back: phase
  // 3 proves the write landed, but nothing stops the paired owner's *next*
  // call from overwriting it while this one is still reading. Alternating
  // halves supplies the missing distance, and two halves are exactly enough --
  // owner_pair_barrier is a two-sided rendezvous, so this rank cannot leave
  // call k until its partner has entered call k, hence cannot reach call k+2
  // (the next use of this half) until the partner has left call k. Anyone
  // making that barrier one-sided silently breaks this.
  int32_t* scratch_state = params.scratch_state;
  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  const int call_epoch = advance_scratch_epoch(scratch_state, epoch_slot);
  const int stage_offset = call_epoch * params.epoch_stride_packs;
  // This rank owns the chunk at its own position in the island.
  const int my_start = local * part;
  const int my_end = (local == 3) ? params.num_packs : my_start + part;
  const int my_slot = stage_offset + params.rank * params.rank_stride_packs;
  const int cross_slot = stage_offset + cross_owner * params.rank_stride_packs;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);

    // Island reduce-scatter. The contribution to this rank's own chunk stays
    // on this GPU, so it costs no fabric time and needs no pass of its own.
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      store_u4_volatile(local_buffer, my_slot + idx, input[idx]);
    }
    for (int p = 0; p < 3; ++p) {
      const int t = (local + 1 + p) & 3;
      const int owner_t = base + t;
      const int t_start = t * part;
      const int t_end = (t == 3) ? params.num_packs : t_start + part;
      auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner_t]);
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        store_u4_volatile(owner_buffer, my_slot + idx, input[idx]);
      }
      __threadfence_system();
      block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, p, flag,
                         island_mask);
    }

    // Island sum, then the one cross-SYS exchange with the paired owner.
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      uint4 v0 = load_u4_volatile(local_buffer,
                                  stage_offset + (base + 0) * params.rank_stride_packs + idx);
      uint4 v1 = load_u4_volatile(local_buffer,
                                  stage_offset + (base + 1) * params.rank_stride_packs + idx);
      uint4 v2 = load_u4_volatile(local_buffer,
                                  stage_offset + (base + 2) * params.rank_stride_packs + idx);
      uint4 v3 = load_u4_volatile(local_buffer,
                                  stage_offset + (base + 3) * params.rank_stride_packs + idx);
      uint4 local_sum = packed_add_u4<T>(packed_add_u4<T>(v0, v1), packed_add_u4<T>(v2, v3));
      // Keep the island sum so the gather phase does not recompute it.
      store_u4_volatile(local_buffer, my_slot + idx, local_sum);
      auto* cross_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[cross_owner]);
      store_u4_volatile(cross_buffer, my_slot + idx, local_sum);
    }
    __threadfence_system();
    owner_pair_barrier(params.signal_ptrs, params.rank, params.rank, cross_owner, params.max_blocks,
                       3, flag);
    debug_cross_read_stall(params.rank);

    // Final value for the owned chunk, written locally.
    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      uint4 mine = load_u4_volatile(local_buffer, my_slot + idx);
      uint4 theirs = load_u4_volatile(local_buffer, cross_slot + idx);
      uint4 final_value = packed_add_u4<T>(mine, theirs);
      reinterpret_cast<uint4*>(params.output)[idx] = final_value;
      store_u4_volatile(local_buffer, my_slot + idx, final_value);
    }
    __threadfence_system();

    // Island all-gather, staged on the same permutation schedule.
    for (int p = 0; p < 3; ++p) {
      const int peer = base + ((local + 1 + p) & 3);
      auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
      for (int idx = my_start + tid; idx < my_end; idx += stride) {
        store_u4_volatile(peer_buffer, my_slot + idx,
                          load_u4_volatile(local_buffer, my_slot + idx));
      }
      __threadfence_system();
      block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, 4 + p, flag,
                         island_mask);
    }

    // Collect the three chunks owned by the other island members.
    for (int p = 0; p < 3; ++p) {
      const int t = (local + 1 + p) & 3;
      const int owner_t = base + t;
      const int t_start = t * part;
      const int t_end = (t == 3) ? params.num_packs : t_start + part;
      const int owner_slot = stage_offset + owner_t * params.rank_stride_packs;
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        reinterpret_cast<uint4*>(params.output)[idx] =
            load_u4_volatile(local_buffer, owner_slot + idx);
      }
    }
    // Hold the island until everyone has finished reading before the next
    // call's reduce-scatter starts writing the same slots. This covers the
    // intra-island reuse only; the cross-island edge is what the epoch double
    // buffer above supplies.
    block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, 7, flag, island_mask);
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);

    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      store_pack_volatile<T>(local_buffer, my_slot + idx, input[idx]);
    }
    for (int p = 0; p < 3; ++p) {
      const int t = (local + 1 + p) & 3;
      const int owner_t = base + t;
      const int t_start = t * part;
      const int t_end = (t == 3) ? params.num_packs : t_start + part;
      auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner_t]);
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        store_pack_volatile<T>(owner_buffer, my_slot + idx, input[idx]);
      }
      __threadfence_system();
      block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, p, flag,
                         island_mask);
    }

    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      Pack values[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[i] = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + i) * params.rank_stride_packs + idx);
      }
      Pack local_sum = reduce_loaded_packs<T, 4>(values);
      store_pack_volatile<T>(local_buffer, my_slot + idx, local_sum);
      auto* cross_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[cross_owner]);
      store_pack_volatile<T>(cross_buffer, my_slot + idx, local_sum);
    }
    __threadfence_system();
    owner_pair_barrier(params.signal_ptrs, params.rank, params.rank, cross_owner, params.max_blocks,
                       3, flag);
    debug_cross_read_stall(params.rank);

    for (int idx = my_start + tid; idx < my_end; idx += stride) {
      Pack pair[2];
      pair[0] = load_pack_volatile<T>(local_buffer, my_slot + idx);
      pair[1] = load_pack_volatile<T>(local_buffer, cross_slot + idx);
      Pack final_value = reduce_loaded_packs<T, 2>(pair);
      reinterpret_cast<Pack*>(params.output)[idx] = final_value;
      store_pack_volatile<T>(local_buffer, my_slot + idx, final_value);
    }
    __threadfence_system();

    for (int p = 0; p < 3; ++p) {
      const int peer = base + ((local + 1 + p) & 3);
      auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
      for (int idx = my_start + tid; idx < my_end; idx += stride) {
        store_pack_volatile<T>(peer_buffer, my_slot + idx,
                               load_pack_volatile<T>(local_buffer, my_slot + idx));
      }
      __threadfence_system();
      block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, 4 + p, flag,
                         island_mask);
    }

    for (int p = 0; p < 3; ++p) {
      const int t = (local + 1 + p) & 3;
      const int owner_t = base + t;
      const int t_start = t * part;
      const int t_end = (t == 3) ? params.num_packs : t_start + part;
      const int owner_slot = stage_offset + owner_t * params.rank_stride_packs;
      for (int idx = t_start + tid; idx < t_end; idx += stride) {
        reinterpret_cast<Pack*>(params.output)[idx] =
            load_pack_volatile<T>(local_buffer, owner_slot + idx);
      }
    }
    block_barrier_mask(params.signal_ptrs, params.rank, 8, params.max_blocks, 7, flag, island_mask);
  }

  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, params.max_blocks, 8), flag);
  }
  // The barrier flag above must stay ahead of the release: a dependent kernel
  // started by the trigger would otherwise observe this call's generation as
  // not yet published. (The epoch is no longer a concern here -- it is
  // committed at entry.)
  debug_commit_per_block_epoch(epoch_slot, call_epoch);
  pdl_grid_release_const<UsePdl>();
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_topo_rsag8_block_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* self_signal = reinterpret_cast<int32_t*>(params.signal_ptrs[params.rank]);
  // Unsigned arithmetic for the bump: signed overflow is UB, and this counter
  // is meant to wrap. generation_reached() reads it back on the circle.
  int flag =
      static_cast<int32_t>(static_cast<uint32_t>(load_acquire_i32(
                               self_signal + flag_offset(blockIdx.x, params.max_blocks, 8))) +
                           1u);

  // The caller's `blocks % 4 == 0` check is what keeps `blocks_per_chunk`
  // non-zero; a zero stride below never advances the grid-stride loops.
  int chunk = blockIdx.x & 3;
  int chunk_block = blockIdx.x >> 2;
  int blocks_per_chunk = gridDim.x >> 2;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;
  int part = params.num_packs >> 2;
  int start = chunk * part;
  int end = (chunk == 3) ? params.num_packs : start + part;
  int base = params.rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;
  // Call-level double buffer, for the same reason as the ring kernel: the
  // phase 4 ack only covers this island, so nothing orders this rank's cross
  // read against the paired owner's next-call cross write. Both TP8 kernels
  // share this counter on purpose -- which one runs changes with the payload,
  // and a counter advanced by only one of them would let a call land on
  // a half two calls old.
  int32_t* scratch_state = params.scratch_state;
  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  const int call_epoch = advance_scratch_epoch(scratch_state, epoch_slot);
  const int stage_offset = call_epoch * params.epoch_stride_packs;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      store_u4_volatile(owner_buffer, offset, input[idx]);
    }
    __threadfence_system();
    island_owner_gather(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 1, flag);

    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 v0 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 0) * params.rank_stride_packs + idx);
        uint4 v1 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 1) * params.rank_stride_packs + idx);
        uint4 v2 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 2) * params.rank_stride_packs + idx);
        uint4 v3 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 3) * params.rank_stride_packs + idx);
        uint4 local_sum = packed_add_u4<T>(packed_add_u4<T>(v0, v1), packed_add_u4<T>(v2, v3));
        auto* cross_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[cross_owner]);
        int cross_write = stage_offset + params.rank * params.rank_stride_packs + idx;
        store_u4_volatile(cross_buffer, cross_write, local_sum);
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    owner_pair_barrier(params.signal_ptrs, params.rank, owner, cross_owner, params.max_blocks, 2,
                       flag);
    debug_cross_read_stall(params.rank);

    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 v0 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 0) * params.rank_stride_packs + idx);
        uint4 v1 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 1) * params.rank_stride_packs + idx);
        uint4 v2 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 2) * params.rank_stride_packs + idx);
        uint4 v3 = load_u4_volatile(local_buffer,
                                    stage_offset + (base + 3) * params.rank_stride_packs + idx);
        uint4 local_sum = packed_add_u4<T>(packed_add_u4<T>(v0, v1), packed_add_u4<T>(v2, v3));
        uint4 cross_sum = load_u4_volatile(
            local_buffer, stage_offset + cross_owner * params.rank_stride_packs + idx);
        uint4 final_value = packed_add_u4<T>(local_sum, cross_sum);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, final_value);
        }
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    island_owner_ready(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 3, flag);

    if (params.rank != owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 final_value =
            load_u4_volatile(local_buffer, stage_offset + owner * params.rank_stride_packs + idx);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
      }
    }
    // WRONG ORDER, and the reason enable_pdl is still refused at the binding:
    // the trigger fires before island_owner_ack and before the barrier flag
    // below, so a dependent kernel can start while this call's phase-4 ack and
    // flag are still being written. Moving the release past both is the fix,
    // but re-enabling PDL needs a per-kernel audit and an SM90 regression, so
    // it is left visible rather than quietly reordered.
    pdl_grid_release_const<UsePdl>();
    island_owner_ack(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 4, flag);
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      store_pack_volatile<T>(owner_buffer, offset, input[idx]);
    }
    __threadfence_system();
    island_owner_gather(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 1, flag);

    auto* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack v0 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 0) * params.rank_stride_packs + idx);
        Pack v1 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 1) * params.rank_stride_packs + idx);
        Pack v2 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 2) * params.rank_stride_packs + idx);
        Pack v3 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 3) * params.rank_stride_packs + idx);
        Pack local_sum = add_pack<T>(add_pack<T>(v0, v1), add_pack<T>(v2, v3));
        auto* cross_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[cross_owner]);
        int cross_write = stage_offset + params.rank * params.rank_stride_packs + idx;
        store_pack_volatile<T>(cross_buffer, cross_write, local_sum);
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    owner_pair_barrier(params.signal_ptrs, params.rank, owner, cross_owner, params.max_blocks, 2,
                       flag);
    debug_cross_read_stall(params.rank);

    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack v0 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 0) * params.rank_stride_packs + idx);
        Pack v1 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 1) * params.rank_stride_packs + idx);
        Pack v2 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 2) * params.rank_stride_packs + idx);
        Pack v3 = load_pack_volatile<T>(local_buffer,
                                        stage_offset + (base + 3) * params.rank_stride_packs + idx);
        Pack local_sum = add_pack<T>(add_pack<T>(v0, v1), add_pack<T>(v2, v3));
        Pack cross_sum = load_pack_volatile<T>(
            local_buffer, stage_offset + cross_owner * params.rank_stride_packs + idx);
        Pack final_value = add_pack<T>(local_sum, cross_sum);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, final_value);
        }
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    island_owner_ready(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 3, flag);

    if (params.rank != owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack final_value = load_pack_volatile<T>(
            local_buffer, stage_offset + owner * params.rank_stride_packs + idx);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
      }
    }
    // WRONG ORDER, and the reason enable_pdl is still refused at the binding:
    // the trigger fires before island_owner_ack and before the barrier flag
    // below, so a dependent kernel can start while this call's phase-4 ack and
    // flag are still being written. Moving the release past both is the fix,
    // but re-enabling PDL needs a per-kernel audit and an SM90 regression, so
    // it is left visible rather than quietly reordered.
    pdl_grid_release_const<UsePdl>();
    island_owner_ack(params.signal_ptrs, params.rank, base, owner, params.max_blocks, 4, flag);
  }

  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, params.max_blocks, 8), flag);
  }
  debug_commit_per_block_epoch(epoch_slot, call_epoch);
}

template <typename T, int WorldSize, bool UsePdl, bool Fp32Reduce>
__global__ __launch_bounds__(1024, 1) void push_oneshot_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot =
      params.epoch_slots + blockIdx.x;  // used only by the PER_BLOCK_EPOCH debug build
  int epoch = advance_scratch_epoch(params.scratch_state, epoch_slot);
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      uint4 value = clear_pos_zero_u4_16(input[idx]);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
        int peer_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
        store_u4_volatile(peer_buffer, peer_offset, value);
      }
    }

    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, peer_offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      uint4 acc;
      if constexpr (Fp32Reduce) {
        acc = reduce_u4_fp32<T, WorldSize>(values);
      } else {
        acc = values[0];
#pragma unroll
        for (int peer = 1; peer < WorldSize; ++peer) {
          acc = packed_add_u4<T>(acc, values[peer]);
        }
      }
      reinterpret_cast<uint4*>(params.output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset = stage_offset + peer * params.rank_stride_packs + idx;
        local_buffer[peer_offset] = reset;
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        Pack* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
        int peer_offset = stage_offset + params.rank * params.rank_stride_packs + idx;
        store_pack_volatile<T>(peer_buffer, peer_offset, value);
      }
    }

    Pack* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    Pack reset = zero_pack<T>();
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, peer_offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(params.output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset = stage_offset + peer * params.rank_stride_packs + idx;
        local_buffer[peer_offset] = reset;
      }
    }
  }

  debug_commit_per_block_epoch(epoch_slot, epoch);
  pdl_grid_release_const<UsePdl>();
}
// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Byte layout of one rank's workspace slab.
//
//   [ epoch slots | barrier phase slots | barrier flags
//     | block-scratch epoch + arrival | pack scratch | block scratch ]
//
// Both scratch regions are sized for world_size ranks x a double-buffered
// epoch, so a rank may start collective N+1 before its peer has drained N.
// The epoch halves sit at fixed offsets derived from max_numel rather than
// from the current payload: if they moved with the payload, a rank that
// finished a large collective and flipped its epoch would start writing a
// small one inside the region a lagging peer is still draining.
struct WorkspaceLayout {
  size_t signal_bytes;
  size_t max_payload_bytes;
  size_t scratch_bytes;  // per scratch region
  size_t total_bytes;
};

inline WorkspaceLayout compute_workspace_layout(int world_size, int64_t max_numel, int elem_size,
                                                int max_blocks) {
  const size_t epoch_slots = static_cast<size_t>(max_blocks);
  const size_t barrier_slots = static_cast<size_t>(kSignalPhases) *
                               static_cast<size_t>(max_blocks) * static_cast<size_t>(world_size);
  const size_t flag_slots = static_cast<size_t>(max_blocks);
  // {epoch, arrival} per scratch region, in ScratchRegion order. Appended at
  // the tail so phase_offset() and flag_offset(), both anchored at the front,
  // are unchanged. See scratch_state_offset().
  const size_t scratch_state_slots = 2 * 2;
  const size_t signal_slots = epoch_slots + barrier_slots + flag_slots + scratch_state_slots;
  auto align128 = [](size_t n) { return (n + 127u) & ~static_cast<size_t>(127u); };
  WorkspaceLayout layout{};
  layout.signal_bytes = align128(sizeof(int32_t) * signal_slots);
  layout.max_payload_bytes =
      align128(static_cast<size_t>(max_numel) * static_cast<size_t>(elem_size));
  layout.scratch_bytes = align128(2 * static_cast<size_t>(world_size) * layout.max_payload_bytes);
  layout.total_bytes = layout.signal_bytes + 2 * layout.scratch_bytes;
  return layout;
}

// Bytes each rank must allocate and share over CUDA IPC.
inline int64_t workspace_size(int world_size, int64_t max_numel, int elem_size, int max_blocks) {
  return static_cast<int64_t>(
      compute_workspace_layout(world_size, max_numel, elem_size, max_blocks).total_bytes);
}

// Per-region device pointers into every rank's slab, as seen by this process.
struct PeerViews {
  uint64_t signal[kMaxWorldSize];
  uint64_t pack[kMaxWorldSize];
  uint64_t block[kMaxWorldSize];
  int32_t* self_signal;
};

// ipc_ptrs[i] must address rank i's slab; ipc_ptrs[rank] is this rank's own.
inline PeerViews make_peer_views(const int64_t* ipc_ptrs, int world_size, int rank,
                                 const WorkspaceLayout& layout) {
  PeerViews views{};
  for (int peer = 0; peer < world_size; ++peer) {
    auto* base = reinterpret_cast<char*>(ipc_ptrs[peer]);
    views.signal[peer] = reinterpret_cast<uint64_t>(base);
    auto* scratch = base + static_cast<ptrdiff_t>(layout.signal_bytes);
    views.pack[peer] = reinterpret_cast<uint64_t>(scratch);
    views.block[peer] = reinterpret_cast<uint64_t>(scratch + layout.scratch_bytes);
  }
  views.self_signal = reinterpret_cast<int32_t*>(ipc_ptrs[rank]);
  return views;
}

template <typename Kernel, typename... Args>
inline cudaError_t launch(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, bool use_pdl,
                          Args const&... args) {
#if CUDART_VERSION >= 12000
  if (use_pdl) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, args...);
  }
#else
  if (use_pdl) return cudaErrorNotSupported;
#endif
  kernel<<<grid, block, 0, stream>>>(args...);
  return cudaGetLastError();
}

// Kernel selection. `variant` is picked by the caller rather than by a
// threshold here, because the crossovers depend on the fabric and are measured
// per machine.
//
//   world  variant       kernel
//     2    kUnstaged     ipc_tp2_remote_push_kernel<false>     (block scratch)
//     2    kStaged       ipc_tp2_remote_push_kernel<true>      (block scratch)
//     4    kUnstaged     push_oneshot_param_kernel             (pack  scratch)
//     4    kStaged       ipc_rsag_push_param_kernel<4>         (block scratch)
//     4    kStagedRing   ipc_rsag_ring_push_param_kernel<4>    (block scratch)
//     8    kUnstaged     ipc_topo_rsag8_push_param_kernel      (pack  scratch)
//     8    kStaged       ipc_topo_rsag8_block_param_kernel     (block scratch)
//     8    kStagedRing   ipc_topo_rsag8_ring_push_param_kernel (block scratch)
//     8    kFlatStaged   ipc_rsag_push_param_kernel<8>         (pack  scratch)
//
// Preconditions the caller must have validated: world_size in {2,4,8}; the
// (world_size, variant) pair appears above; 0 < blocks <= max_blocks;
// 0 < threads <= 1024; numel and max_numel both divisible by the 16-byte pack
// width; numel * elem_size <= max_payload_bytes; and blocks % 4 == 0 for
// (8, kStaged), since that kernel derives its chunk from blockIdx.x & 3.
template <typename T>
cudaError_t all_reduce(const T* input, T* output, int64_t numel, const PeerViews& views, int rank,
                       int world_size, int max_blocks, int64_t max_numel, int blocks, int threads,
                       Variant variant, bool use_pdl, cudaStream_t stream) {
  using Traits = PackTraits<T>;
  const int num_packs = static_cast<int>(numel / Traits::kPackElems);
  const int rank_stride_packs = static_cast<int>(max_numel / Traits::kPackElems);
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 cta(static_cast<unsigned>(threads));

  if (world_size == 2) {
    IpcTp2RemotePushData<T> params{};
    params.tmp_ptrs[0] = views.block[0];
    params.tmp_ptrs[1] = views.block[1];
    params.input = input;
    params.output = output;
    params.epoch_slots = views.self_signal;
    // TP2 stages through views.block under either variant, so it shares the
    // block region's counter -- nominal here, since it is the only TP2 kernel,
    // but the state must follow the region it actually writes.
    params.scratch_state =
        views.self_signal + scratch_state_offset(max_blocks, 2, ScratchRegion::kBlock);
    params.num_packs = num_packs;
    params.rank_stride_packs = rank_stride_packs;
    params.rank = rank;
    const bool staged = variant == Variant::kStaged;
    if (use_pdl) {
      return staged ? launch(ipc_tp2_remote_push_kernel<T, true, true>, grid, cta, stream, true,
                             params)
                    : launch(ipc_tp2_remote_push_kernel<T, false, true>, grid, cta, stream, true,
                             params);
    }
    return staged ? launch(ipc_tp2_remote_push_kernel<T, true, false>, grid, cta, stream, false,
                           params)
                  : launch(ipc_tp2_remote_push_kernel<T, false, false>, grid, cta, stream, false,
                           params);
  }

  PushOneshotParamData<T> params{};
  // Region and counter are chosen together: a kernel reading one region while
  // advancing another's epoch would corrupt both. Which kernels may share a
  // region is a protocol question, not a partitioning one -- see ScratchRegion.
  const ScratchRegion region = (variant == Variant::kUnstaged || variant == Variant::kFlatStaged)
                                   ? ScratchRegion::kPack
                                   : ScratchRegion::kBlock;
  const uint64_t* scratch = region == ScratchRegion::kPack ? views.pack : views.block;
  for (int peer = 0; peer < world_size; ++peer) {
    params.tmp_ptrs[peer] = scratch[peer];
    params.signal_ptrs[peer] = views.signal[peer];
  }
  params.input = input;
  params.output = output;
  params.epoch_slots = views.self_signal;
  params.scratch_state = views.self_signal + scratch_state_offset(max_blocks, world_size, region);
  params.num_packs = num_packs;
  params.rank_stride_packs = rank_stride_packs;
  params.epoch_stride_packs = world_size * rank_stride_packs;
  params.rank = rank;
  params.max_blocks = max_blocks;

#define FI_PCIE_IPC_LAUNCH(KERNEL_EXPR, PDL) launch(KERNEL_EXPR, grid, cta, stream, PDL, params)

#define FI_PCIE_IPC_SELECT(PDL)                                                            \
  do {                                                                                     \
    if (world_size == 8) {                                                                 \
      switch (variant) {                                                                   \
        case Variant::kUnstaged:                                                           \
          return FI_PCIE_IPC_LAUNCH((ipc_topo_rsag8_push_param_kernel<T, PDL>), PDL);      \
        case Variant::kStaged:                                                             \
          return FI_PCIE_IPC_LAUNCH((ipc_topo_rsag8_block_param_kernel<T, PDL>), PDL);     \
        case Variant::kStagedRing:                                                         \
          return FI_PCIE_IPC_LAUNCH((ipc_topo_rsag8_ring_push_param_kernel<T, PDL>), PDL); \
        case Variant::kFlatStaged:                                                         \
          return FI_PCIE_IPC_LAUNCH((ipc_rsag_push_param_kernel<T, 8, PDL>), PDL);         \
      }                                                                                    \
      return cudaErrorInvalidValue;                                                        \
    }                                                                                      \
    switch (variant) {                                                                     \
      case Variant::kUnstaged:                                                             \
        return FI_PCIE_IPC_LAUNCH((push_oneshot_param_kernel<T, 4, PDL, false>), PDL);     \
      case Variant::kStaged:                                                               \
        return FI_PCIE_IPC_LAUNCH((ipc_rsag_push_param_kernel<T, 4, PDL>), PDL);           \
      case Variant::kStagedRing:                                                           \
        return FI_PCIE_IPC_LAUNCH((ipc_rsag_ring_push_param_kernel<T, 4, PDL>), PDL);      \
      default:                                                                             \
        return cudaErrorInvalidValue;                                                      \
    }                                                                                      \
  } while (false)

  if (use_pdl) {
    FI_PCIE_IPC_SELECT(true);
  }
  FI_PCIE_IPC_SELECT(false);

#undef FI_PCIE_IPC_SELECT
#undef FI_PCIE_IPC_LAUNCH
}

}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_PCIE_IPC_ALL_REDUCE_CUH_
