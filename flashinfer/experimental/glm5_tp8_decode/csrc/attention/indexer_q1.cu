/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "launch_util.cuh"
#include "q1_launch.cuh"

namespace flashinfer::glm5_tp8::indexer_q1 {

constexpr int kGrid = 148, kBlock = 384, kConsumers = 256;
constexpr int kDynamicSmem = 229376;
constexpr int kHidden = 6144, kRows = 48, kLines = 8;
constexpr size_t kSlot = 0x8000, kParity = 0x4000, kRecord = 0x80;

__device__ __forceinline__ void group_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void store_record(void* p, uint32_t a, uint32_t b, uint32_t c,
                                             uint32_t tag) {
  const uint64_t lo = uint64_t(a) | (uint64_t(b) << 32);
  const uint64_t hi = uint64_t(c) | (uint64_t(tag) << 32);
  asm volatile(
      "{ .reg .b128 r; mov.b128 r, {%1,%2}; st.relaxed.sys.global.b128 [%0], r; }" ::"l"(p),
      "l"(lo), "l"(hi)
      : "memory");
}
__device__ __forceinline__ void load_record(const void* p, uint64_t& lo, uint64_t& hi) {
  asm volatile("{ .reg .b128 r; ld.relaxed.sys.global.b128 r, [%2]; mov.b128 {%0,%1}, r; }"
               : "=l"(lo), "=l"(hi)
               : "l"(p)
               : "memory");
}
template <class T>
__device__ __forceinline__ T* pointer(const uint64_t* rows, int row, int word) {
  return reinterpret_cast<T*>(rows[row * 16 + word]);
}

// Index-share has no scoring or projection work on rank 0. Eight publisher
// CTAs refresh the q1 LL channel; all 128 output CTAs join the same TP8 exit
// protocol as the MLA ranks. C6 counts monotonically across calls.
__global__ __launch_bounds__(kBlock,
                             1) void indexshare(const __grid_constant__ attention_q1::Q1Args args) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  extern __shared__ __align__(1024) uint8_t storage[];
  const int cta = int(blockIdx.x), tid = int(threadIdx.x);
  auto* generation = reinterpret_cast<uint32_t*>(storage);
  if (tid == 0) {
    *generation = args.launch_ctr[cta] + 1u;
    args.launch_ctr[cta] = *generation;
  }
  __syncthreads();
  if (tid >= kConsumers || cta >= 128) return;
  const uint32_t epoch = *generation;
  const auto* rows = reinterpret_cast<const uint64_t*>(args.context);
  const int cur_pos = *pointer<const int32_t>(rows, 4, 4);
  const auto* ids = pointer<const int32_t>(rows, 6, 0);
  const auto* id_peers = pointer<const uint64_t>(rows, 6, 1);
  const uint32_t ll_tag = uint32_t(rows[6 * 16 + 2]);
  const int described_records = int(uint32_t(rows[6 * 16 + 2] >> 32) / 2);
  uint32_t* published = args.sync_words + 0x60 / 4;
  if (cta < 8) {
    if (cta == 0 && tid == 0) {
      if (!ids || ids != pointer<const int32_t>(rows, 4, 6) ||
          ids != pointer<const int32_t>(rows, 5, 1))
        __trap();
    }
    if (cur_pos >= 2048) {
      for (int record = cta * kConsumers + tid; record < described_records && record < 1024;
           record += 8 * kConsumers) {
        const uint32_t id0 = uint32_t(ids[2 * record]), id1 = uint32_t(ids[2 * record + 1]);
#pragma unroll
        for (int peer = 0; peer < 7; ++peer) {
          store_record(reinterpret_cast<uint8_t*>(id_peers[peer]) + size_t(record) * 16, id0,
                       ll_tag, id1, ll_tag);
        }
      }
    }
    group_sync();
    if (tid == 0)
      asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(published), "r"(32u)
                   : "memory");
  }
  if (tid == 0) {
    uint32_t seen;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(published) : "memory");
    } while (int32_t(seen - 256u * epoch) < 0);
  }
  group_sync();

  auto* own = reinterpret_cast<uint16_t*>(storage + 32);
  auto* received = reinterpret_cast<uint16_t*>(storage + 128);
  const auto* own_source = pointer<const uint16_t>(rows, 7, 0);
  const auto* residual = pointer<const __nv_bfloat16>(rows, 7, 1);
  auto* output = pointer<__nv_bfloat16>(rows, 7, 2);
  if (tid < kRows) own[tid] = own_source[cta * kRows + tid];
  group_sync();
  const int warp = tid >> 5, lane = tid & 31;
  const int peer = (warp & 1) * 4 + (lane & 3);
  const int line = lane >> 2;
  const bool active = warp < 2 && peer < 7;
  const size_t offset = size_t(epoch & 1u) * kParity + size_t(cta) * kRecord + size_t(line) * 16;
  if (active) {
    const uint32_t* payload = reinterpret_cast<const uint32_t*>(own + line * 6);
    store_record(args.peers[peer + 1] + offset, payload[0], payload[1], payload[2], epoch);
  }
  group_sync();
  if (active) {
    uint64_t lo, hi;
    do {
      load_record(args.peers[0] + size_t(peer) * kSlot + offset, lo, hi);
    } while (uint32_t(hi >> 32) != epoch);
    auto* payload = reinterpret_cast<uint32_t*>(received + peer * kRows + line * 6);
    payload[0] = uint32_t(lo);
    payload[1] = uint32_t(lo >> 32);
    payload[2] = uint32_t(hi);
  }
  group_sync();
  if (tid < kRows) {
    float sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
      const auto* source =
          reinterpret_cast<const __nv_bfloat16*>(rank == 0 ? own : received + (rank - 1) * kRows);
      sum = __fadd_rn(sum, __bfloat162float(source[tid]));
    }
    output[cta * kRows + tid] =
        __float2bfloat16_rn(__fadd_rn(sum, __bfloat162float(residual[cta * kRows + tid])));
  }
}
}  // namespace flashinfer::glm5_tp8::indexer_q1

namespace flashinfer::glm5_tp8 {
KernelResources initialize_indexer_q1(int device) {
  using namespace indexer_q1;
  return attention_detail::configure(reinterpret_cast<const void*>(&indexshare), kGrid, kBlock,
                                     kDynamicSmem, device);
}
cudaError_t launch_indexer_q1(const AttentionArguments& p, cudaStream_t stream, bool pdl,
                              bool cooperative) {
  using namespace indexer_q1;
  attention_q1::Q1Args a{};
  a.n_rows = 8;
  a.rank = 0;
  a.context = reinterpret_cast<const uint8_t*>(p.context);
  a.sync_words = reinterpret_cast<uint32_t*>(p.q1_state);
  a.launch_ctr = reinterpret_cast<uint32_t*>(p.q1_state + attention_q1::kQ1Counters);
  a.reset_token = reinterpret_cast<uint32_t*>(p.q1_state + attention_q1::kQ1ResetToken);
  for (int rank = 0; rank < 8; ++rank) a.peers[rank] = p.peer_records[rank];
  return attention_detail::launch(reinterpret_cast<const void*>(&indexshare), a, kGrid, kBlock,
                                  kDynamicSmem, stream, pdl, cooperative);
}
}  // namespace flashinfer::glm5_tp8
