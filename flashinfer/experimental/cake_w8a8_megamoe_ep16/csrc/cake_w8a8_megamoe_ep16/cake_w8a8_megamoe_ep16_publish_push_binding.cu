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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), stage `publish_push`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_publish_push_kernel.cu). The binding
// checks the tensor metadata of the route contract, builds the by-value peer
// tables from the 16 peer addresses of each symmetric buffer, resolves the
// environment stream and performs one launch; the block size and the dynamic
// shared memory are the generated program's constants.
//
// One CTA per staged token: the program quantizes the bf16 row to block-32
// MXFP8, zeroes this rank's combine read window of the term buffer (rows
// [pg_rank * tokens, (pg_rank + 1) * tokens) of `local_terms`), resets the
// routing state (`overflow[0] = 0`, `expert_counts[0:32] = 0`,
// `route_map[0:4096] = -1`), bulk-copies the token's 3232-byte slot
// (row_fp8 | scales | ids | weights) into `push_inbox[pg_rank][token]` of every
// destination rank that owns one of its experts (the 64-byte ids+weights tail
// only elsewhere) and then stores `push_generation[0] + 1` into
// `push_flags[pg_rank][token]` of every destination. The host performs no
// memset between the forward programs and never resets the flags; it zeroes
// the flags and `push_generation` once at allocation and rendezvouses (device
// synchronize, then a process-group barrier) before any rank's first launch.
// The local inbox and flag pointers carry no __restrict__ (peers write them
// while the kernel runs); the by-value peer tables have 32 slots of which
// entries [0, 16) are used.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/shape.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#ifndef CAKE_PEER_POINTER_TABLE_DECLARED
#define CAKE_PEER_POINTER_TABLE_DECLARED
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable {
  T* ptrs[Capacity];
};
#endif

// Complete every by-value peer-table specialization the declaration names
// before the declaration. Required under FlashInfer's --expt-relaxed-constexpr:
// an uninstantiated by-value template parameter type of a __global__
// declaration is not first-class in the device IR of this launcher-only
// translation unit (the module build fails on it even though the TU compiles
// standalone). Keep one sizeof-based assert per specialization named in the
// declaration; the asserts also pin the CUDA ABI (32 slots, 8 bytes each,
// 16-byte aligned).
static_assert(sizeof(CakePeerPointerTable<unsigned int, 32>) == 256 &&
                  alignof(CakePeerPointerTable<unsigned int, 32>) == 16,
              "CakePeerPointerTable<unsigned int, 32> must be 256 bytes, 16-byte aligned");
static_assert(sizeof(CakePeerPointerTable<uint8_t, 32>) == 256 &&
                  alignof(CakePeerPointerTable<uint8_t, 32>) == 16,
              "CakePeerPointerTable<uint8_t, 32> must be 256 bytes, 16-byte aligned");

// Device entry point of the kernel translation unit (manifest declaration).
// clang-format off
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_publish_push(__nv_bfloat16* __restrict__ source_hidden, long long* __restrict__ source_ids, float* __restrict__ source_weights, __nv_bfloat16* __restrict__ local_terms, int* __restrict__ expert_counts, int* __restrict__ route_map, int* __restrict__ overflow, unsigned int* __restrict__ push_generation, int tokens, int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int, 32> pg_flags, uint8_t* push_inbox, CakePeerPointerTable<uint8_t, 32> push_inbox_peers, unsigned int* push_flags, CakePeerPointerTable<unsigned int, 32> push_flags_peers);
// clang-format on

namespace cake_w8a8_megamoe_ep16_publish_push {

constexpr int64_t kWorldSize = 16;
constexpr int64_t kMaxTokensPerRank = 64;
constexpr int64_t kMaxRecvTokens = 1024;  // 16 ranks x 64 tokens
constexpr int64_t kHidden = 3072;
constexpr int64_t kTopK = 8;
constexpr int64_t kLocalExperts = 32;
constexpr int64_t kTotalExpertRows = 4096;  // 32 experts x 128 packed rows
// One inbox slot: row_fp8 [0, 3072) | scales [3072, 96) | ids_i32 [3168, 32) |
// weights_f32 [3200, 32); slot (source_rank, token) at (source_rank * 64 + token) * 3232.
constexpr int64_t kPushSlotBytes = 3232;
// push_generation u32 [2]: [0] forwards this rank completed, [1] local_pack CTA arrivals.
constexpr int64_t kPushGenerationWords = 2;
constexpr int kPeerTableCapacity = 32;
constexpr unsigned kBlockThreads = 128;
constexpr unsigned kDynamicSmemBytes = 3328;  // the staged slot, 128-byte rounded

// Host image of CakePeerPointerTable<T, Capacity>: Capacity 8-byte device
// addresses, 16-byte aligned. Slots at or beyond the world size stay zero.
template <int Capacity>
struct alignas(16) CakePeerPointerTableHost {
  uint64_t ptrs[Capacity];
};
static_assert(sizeof(CakePeerPointerTableHost<kPeerTableCapacity>) == 256,
              "32-entry peer table CUDA ABI must be 256 bytes");
static_assert(alignof(CakePeerPointerTableHost<kPeerTableCapacity>) == 16,
              "peer table CUDA ABI must be 16-byte aligned");

// Copy the exactly-16 non-null peer addresses (slot r = rank r) into a table.
CakePeerPointerTableHost<kPeerTableCapacity> peer_table(const tvm::ffi::Shape& peers,
                                                        const char* name) {
  TVM_FFI_CHECK(static_cast<int64_t>(peers.size()) == kWorldSize, ValueError)
      << "peer table '" << name << "' must carry exactly " << kWorldSize << " entries, got "
      << peers.size();
  CakePeerPointerTableHost<kPeerTableCapacity> table{};
  for (int64_t rank = 0; rank < kWorldSize; ++rank) {
    TVM_FFI_CHECK(peers[rank] != 0, ValueError)
        << "peer table '" << name << "' slot " << rank << " is a null address";
    table.ptrs[rank] = static_cast<uint64_t>(peers[rank]);
  }
  return table;
}

// Symmetric self entry: slot `rank` of a peer table is this rank's own base
// address of the same symmetric buffer, i.e. the local pointer bound beside
// the table (manifest route_contract.push_dispatch.peer_table_rule and
// route_contract.fc2_completion.counter_layout). The programs read or poll
// the buffer through the local pointer while the peers write it through the
// table, so a mismatch is a silent hang rather than a fault; the workspace
// asserts the equality once at allocation and every launch re-checks it.
void check_self_entry(const tvm::ffi::Shape& peers, int64_t rank, const TensorView& local,
                      const char* table_name, const char* local_name) {
  TVM_FFI_CHECK(rank >= 0 && rank < static_cast<int64_t>(peers.size()), ValueError)
      << "peer table '" << table_name << "' has no entry for rank " << rank;
  TVM_FFI_CHECK(static_cast<uint64_t>(peers[rank]) == reinterpret_cast<uint64_t>(local.data_ptr()),
                ValueError)
      << "peer table '" << table_name << "' entry " << rank << " must be this rank's local '"
      << local_name << "' pointer (symmetric self entry)";
}

// Require the exact extents of a route-fixed tensor.
template <int Rank>
void check_extents(const TensorView& t, const int64_t (&extents)[Rank], const char* name) {
  TVM_FFI_CHECK(t.ndim() == Rank, ValueError)
      << name << " must be " << Rank << "-D, got ndim=" << t.ndim();
  for (int axis = 0; axis < Rank; ++axis) {
    TVM_FFI_CHECK(t.size(axis) == extents[axis], ValueError)
        << name << ".size(" << axis << ") must be " << extents[axis] << ", got " << t.size(axis);
  }
}

// Require a 2-D [rows, columns] tensor with at least `rows` rows; rows at or
// beyond `rows` are padding the program never reads.
void check_rows(const TensorView& t, int64_t rows, int64_t columns, const char* name) {
  TVM_FFI_CHECK(t.ndim() == 2, ValueError) << name << " must be 2-D, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.size(1) == columns, ValueError)
      << name << ".size(1) must be " << columns << ", got " << t.size(1);
  TVM_FFI_CHECK(t.size(0) >= rows, ValueError)
      << name << ".size(0) must cover " << rows << " rows, got " << t.size(0);
}

void Run(TensorView source_hidden, TensorView source_ids, TensorView source_weights,
         TensorView local_terms, TensorView expert_counts, TensorView route_map,
         TensorView overflow, TensorView push_generation, int64_t tokens, int64_t pg_world,
         int64_t pg_rank, tvm::ffi::Shape pg_flags, TensorView push_inbox,
         tvm::ffi::Shape push_inbox_peers, TensorView push_flags, tvm::ffi::Shape push_flags_peers,
         int64_t grid_x) {
  check_cuda_tensor(source_hidden, "source_hidden");
  const DLDevice dev = source_hidden.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);

  check_dtype(source_hidden, dl_bfloat16, "source_hidden");
  check_contiguous(source_hidden, "source_hidden");
  check_cuda_tensor(source_ids, "source_ids");
  check_dtype(source_ids, dl_int64, "source_ids");
  check_contiguous(source_ids, "source_ids");
  check_same_device(source_ids, source_hidden, "source_ids", "source_hidden");
  check_cuda_tensor(source_weights, "source_weights");
  check_dtype(source_weights, dl_float32, "source_weights");
  check_contiguous(source_weights, "source_weights");
  check_same_device(source_weights, source_hidden, "source_weights", "source_hidden");
  check_cuda_tensor(local_terms, "local_terms");
  check_dtype(local_terms, dl_bfloat16, "local_terms");
  check_contiguous(local_terms, "local_terms");
  check_same_device(local_terms, source_hidden, "local_terms", "source_hidden");
  check_cuda_tensor(expert_counts, "expert_counts");
  check_dtype(expert_counts, dl_int32, "expert_counts");
  check_contiguous(expert_counts, "expert_counts");
  check_same_device(expert_counts, source_hidden, "expert_counts", "source_hidden");
  check_cuda_tensor(route_map, "route_map");
  check_dtype(route_map, dl_int32, "route_map");
  check_contiguous(route_map, "route_map");
  check_same_device(route_map, source_hidden, "route_map", "source_hidden");
  check_cuda_tensor(overflow, "overflow");
  check_dtype(overflow, dl_int32, "overflow");
  check_contiguous(overflow, "overflow");
  check_same_device(overflow, source_hidden, "overflow", "source_hidden");
  check_cuda_tensor(push_generation, "push_generation");
  check_dtype(push_generation, dl_uint32, "push_generation");
  check_contiguous(push_generation, "push_generation");
  check_same_device(push_generation, source_hidden, "push_generation", "source_hidden");
  check_cuda_tensor(push_inbox, "push_inbox");
  check_dtype(push_inbox, dl_uint8, "push_inbox");
  check_contiguous(push_inbox, "push_inbox");
  check_same_device(push_inbox, source_hidden, "push_inbox", "source_hidden");
  check_cuda_tensor(push_flags, "push_flags");
  check_dtype(push_flags, dl_uint32, "push_flags");
  check_contiguous(push_flags, "push_flags");
  check_same_device(push_flags, source_hidden, "push_flags", "source_hidden");

  TVM_FFI_CHECK(tokens >= 1 && tokens <= kMaxTokensPerRank, ValueError)
      << "tokens must be in [1, " << kMaxTokensPerRank << "], got " << tokens;
  TVM_FFI_CHECK(pg_world == kWorldSize, ValueError)
      << "pg_world must be " << kWorldSize << ", got " << pg_world;
  TVM_FFI_CHECK(pg_rank >= 0 && pg_rank < pg_world, ValueError)
      << "pg_rank must be in [0, " << pg_world << "), got " << pg_rank;
  check_rows(source_hidden, tokens, kHidden, "source_hidden");
  check_rows(source_ids, tokens, kTopK, "source_ids");
  check_rows(source_weights, tokens, kTopK, "source_weights");
  // The combine read window rows [pg_rank * tokens, (pg_rank + 1) * tokens) lie
  // inside the [1024, 8, 3072] term buffer for every admitted rank and bound.
  check_extents(local_terms, {kMaxRecvTokens, kTopK, kHidden}, "local_terms");
  check_extents(expert_counts, {kLocalExperts}, "expert_counts");
  check_extents(route_map, {kTotalExpertRows}, "route_map");
  check_extents(overflow, {int64_t{1}}, "overflow");
  check_extents(push_generation, {kPushGenerationWords}, "push_generation");
  // Every destination's inbox and flag slot (pg_rank * 64 + token) lies inside
  // the symmetric [16, 64, 3232] inbox and [16, 64] flag array.
  check_extents(push_inbox, {kWorldSize, kMaxTokensPerRank, kPushSlotBytes}, "push_inbox");
  check_extents(push_flags, {kWorldSize, kMaxTokensPerRank}, "push_flags");
  CakePeerPointerTableHost<kPeerTableCapacity> v_pg_flags = peer_table(pg_flags, "pg_flags");
  CakePeerPointerTableHost<kPeerTableCapacity> v_push_inbox_peers =
      peer_table(push_inbox_peers, "push_inbox_peers");
  CakePeerPointerTableHost<kPeerTableCapacity> v_push_flags_peers =
      peer_table(push_flags_peers, "push_flags_peers");
  check_self_entry(push_inbox_peers, pg_rank, push_inbox, "push_inbox_peers", "push_inbox");
  check_self_entry(push_flags_peers, pg_rank, push_flags, "push_flags_peers", "push_flags");
  // One block per staged token (grid formula tokens): the program pushes the
  // slot and zeroes the term rows of blockIdx.x without a guard.
  TVM_FFI_CHECK(grid_x == tokens, ValueError)
      << "grid_x must equal tokens = " << tokens << ", got " << grid_x;

  const cudaStream_t stream = get_stream(dev);

  __nv_bfloat16* p_source_hidden = static_cast<__nv_bfloat16*>(source_hidden.data_ptr());
  long long* p_source_ids = static_cast<long long*>(source_ids.data_ptr());
  float* p_source_weights = static_cast<float*>(source_weights.data_ptr());
  __nv_bfloat16* p_local_terms = static_cast<__nv_bfloat16*>(local_terms.data_ptr());
  int* p_expert_counts = static_cast<int*>(expert_counts.data_ptr());
  int* p_route_map = static_cast<int*>(route_map.data_ptr());
  int* p_overflow = static_cast<int*>(overflow.data_ptr());
  unsigned int* p_push_generation = static_cast<unsigned int*>(push_generation.data_ptr());
  int32_t v_tokens = static_cast<int32_t>(tokens);
  int32_t v_pg_world = static_cast<int32_t>(pg_world);
  int32_t v_pg_rank = static_cast<int32_t>(pg_rank);
  uint8_t* p_push_inbox = static_cast<uint8_t*>(push_inbox.data_ptr());
  unsigned int* p_push_flags = static_cast<unsigned int*>(push_flags.data_ptr());
  void* kargs[] = {&p_source_hidden, &p_source_ids,       &p_source_weights, &p_local_terms,
                   &p_expert_counts, &p_route_map,        &p_overflow,       &p_push_generation,
                   &v_tokens,        &v_pg_world,         &v_pg_rank,        &v_pg_flags,
                   &p_push_inbox,    &v_push_inbox_peers, &p_push_flags,     &v_push_flags_peers};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_publish_push), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_publish_push) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_publish_push

TVM_FFI_DLL_EXPORT_TYPED_FUNC(publish_push, cake_w8a8_megamoe_ep16_publish_push::Run);
