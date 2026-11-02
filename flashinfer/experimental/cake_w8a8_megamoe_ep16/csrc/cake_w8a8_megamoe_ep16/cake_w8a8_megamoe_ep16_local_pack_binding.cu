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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), stage `local_pack`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_local_pack_kernel.cu). The binding
// checks the tensor metadata of the route contract, builds the by-value peer
// tables from the 16 peer addresses of each symmetric buffer, sets the
// program's preferred shared-memory carveout once (a function attribute, not a
// launch attribute: this launch itself carries none), resolves the environment
// stream and performs one launch; the block size is the generated program's
// constant and the program uses no dynamic shared memory.
//
// One CTA per (source rank, token) = blockIdx.x: the program waits on its
// LOCAL inbox flag (`push_flags[source_rank][token] == push_generation[0] + 1`)
// with an untimed system-scope acquire poll (a late peer stalls this rank, it
// never traps: the contract of the input barrier this program replaces), reads
// the LOCAL inbox slot through coherent global loads (the local inbox and flag
// pointers carry no __restrict__: peers write them while the kernel runs) and
// packs the routes this rank owns per expert (a route at or beyond
// `active_rows_per_expert` rows of its expert is counted in `overflow`
// instead); the last CTA of the grid advances `push_generation[0]`. No input
// barrier precedes this program. The host binds `active_rows_per_expert = 32`
// and `scale_tile_k128 = 1`, the values the exported FC1 program expects.

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
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_local_pack(float* __restrict__ recv_weights, int* __restrict__ expert_counts, int* __restrict__ route_map, int* __restrict__ inverse_rows, int* __restrict__ overflow, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales_cp, unsigned int* __restrict__ push_generation, int tokens_per_rank, int active_rows_per_expert, int scale_tile_k128, int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int, 32> pg_flags, uint8_t* push_inbox, CakePeerPointerTable<uint8_t, 32> push_inbox_peers, unsigned int* push_flags, CakePeerPointerTable<unsigned int, 32> push_flags_peers);
// clang-format on

namespace cake_w8a8_megamoe_ep16_local_pack {

constexpr int64_t kWorldSize = 16;
constexpr int64_t kLocalExperts = 32;
constexpr int64_t kMaxTokensPerRank = 64;
constexpr int64_t kHidden = 3072;
constexpr int64_t kMaxRowsPerExpert = 128;                      // packed FC1 input rows per expert
constexpr int64_t kTotalExpertRows = 4096;                      // 32 experts x 128 packed rows
constexpr int64_t kMaxRoutes = 8192;                            // 16 ranks x 64 tokens x top-8
constexpr int64_t kFc1InputBytes = kTotalExpertRows * kHidden;  // 12,582,912
constexpr int64_t kFc1InputScaleBytes = 393216;                 // packed scale plane
// One inbox slot: row_fp8 [0, 3072) | scales [3072, 96) | ids_i32 [3168, 32) |
// weights_f32 [3200, 32); slot (source_rank, token) at (source_rank * 64 + token) * 3232.
constexpr int64_t kPushSlotBytes = 3232;
// push_generation u32 [2]: [0] forwards this rank completed, [1] local_pack CTA arrivals.
constexpr int64_t kPushGenerationWords = 2;
constexpr int kPeerTableCapacity = 32;
constexpr unsigned kBlockThreads = 256;
constexpr unsigned kDynamicSmemBytes = 0;
// Manifest launch.preferred_shared_memory_carveout: percent of the unified
// L1/shared-memory capacity the program prefers as shared memory.
constexpr int kPreferredSharedMemoryCarveout = 100;

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

// Require the element count of a route-fixed buffer.
void check_numel(const TensorView& t, int64_t numel, const char* name) {
  TVM_FFI_CHECK(t.numel() == numel, ValueError)
      << name << " must hold " << numel << " elements, got " << t.numel();
}

void Run(TensorView recv_weights, TensorView expert_counts, TensorView route_map,
         TensorView inverse_rows, TensorView overflow, TensorView quantized, TensorView scales_cp,
         TensorView push_generation, int64_t tokens_per_rank, int64_t active_rows_per_expert,
         int64_t scale_tile_k128, int64_t pg_world, int64_t pg_rank, tvm::ffi::Shape pg_flags,
         TensorView push_inbox, tvm::ffi::Shape push_inbox_peers, TensorView push_flags,
         tvm::ffi::Shape push_flags_peers, int64_t grid_x) {
  check_cuda_tensor(recv_weights, "recv_weights");
  const DLDevice dev = recv_weights.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);

  check_dtype(recv_weights, dl_float32, "recv_weights");
  check_contiguous(recv_weights, "recv_weights");
  check_cuda_tensor(expert_counts, "expert_counts");
  check_dtype(expert_counts, dl_int32, "expert_counts");
  check_contiguous(expert_counts, "expert_counts");
  check_same_device(expert_counts, recv_weights, "expert_counts", "recv_weights");
  check_cuda_tensor(route_map, "route_map");
  check_dtype(route_map, dl_int32, "route_map");
  check_contiguous(route_map, "route_map");
  check_same_device(route_map, recv_weights, "route_map", "recv_weights");
  check_cuda_tensor(inverse_rows, "inverse_rows");
  check_dtype(inverse_rows, dl_int32, "inverse_rows");
  check_contiguous(inverse_rows, "inverse_rows");
  check_same_device(inverse_rows, recv_weights, "inverse_rows", "recv_weights");
  check_cuda_tensor(overflow, "overflow");
  check_dtype(overflow, dl_int32, "overflow");
  check_contiguous(overflow, "overflow");
  check_same_device(overflow, recv_weights, "overflow", "recv_weights");
  check_cuda_tensor(quantized, "quantized");
  check_dtype(quantized, dl_uint8, "quantized");
  check_contiguous(quantized, "quantized");
  check_same_device(quantized, recv_weights, "quantized", "recv_weights");
  check_cuda_tensor(scales_cp, "scales_cp");
  check_dtype(scales_cp, dl_uint8, "scales_cp");
  check_contiguous(scales_cp, "scales_cp");
  check_same_device(scales_cp, recv_weights, "scales_cp", "recv_weights");
  check_cuda_tensor(push_generation, "push_generation");
  check_dtype(push_generation, dl_uint32, "push_generation");
  check_contiguous(push_generation, "push_generation");
  check_same_device(push_generation, recv_weights, "push_generation", "recv_weights");
  check_cuda_tensor(push_inbox, "push_inbox");
  check_dtype(push_inbox, dl_uint8, "push_inbox");
  check_contiguous(push_inbox, "push_inbox");
  check_same_device(push_inbox, recv_weights, "push_inbox", "recv_weights");
  check_cuda_tensor(push_flags, "push_flags");
  check_dtype(push_flags, dl_uint32, "push_flags");
  check_contiguous(push_flags, "push_flags");
  check_same_device(push_flags, recv_weights, "push_flags", "recv_weights");

  check_numel(recv_weights, kMaxRoutes, "recv_weights");
  check_numel(expert_counts, kLocalExperts, "expert_counts");
  check_numel(route_map, kTotalExpertRows, "route_map");
  check_numel(inverse_rows, kMaxRoutes, "inverse_rows");
  check_numel(overflow, 1, "overflow");
  check_numel(quantized, kFc1InputBytes, "quantized");
  check_numel(scales_cp, kFc1InputScaleBytes, "scales_cp");
  check_extents(push_generation, {kPushGenerationWords}, "push_generation");
  // The slot and flag of every (source rank, token) lie inside the symmetric
  // [16, 64, 3232] inbox and [16, 64] flag array.
  check_extents(push_inbox, {kWorldSize, kMaxTokensPerRank, kPushSlotBytes}, "push_inbox");
  check_extents(push_flags, {kWorldSize, kMaxTokensPerRank}, "push_flags");
  TVM_FFI_CHECK(tokens_per_rank >= 1 && tokens_per_rank <= kMaxTokensPerRank, ValueError)
      << "tokens_per_rank must be in [1, " << kMaxTokensPerRank << "], got " << tokens_per_rank;
  TVM_FFI_CHECK(active_rows_per_expert >= 1 && active_rows_per_expert <= kMaxRowsPerExpert,
                ValueError)
      << "active_rows_per_expert must be in [1, " << kMaxRowsPerExpert << "], got "
      << active_rows_per_expert;
  TVM_FFI_CHECK(scale_tile_k128 == 0 || scale_tile_k128 == 1, ValueError)
      << "scale_tile_k128 must be 0 or 1, got " << scale_tile_k128;
  TVM_FFI_CHECK(pg_world == kWorldSize, ValueError)
      << "pg_world must be " << kWorldSize << ", got " << pg_world;
  TVM_FFI_CHECK(pg_rank >= 0 && pg_rank < pg_world, ValueError)
      << "pg_rank must be in [0, " << pg_world << "), got " << pg_rank;
  CakePeerPointerTableHost<kPeerTableCapacity> v_pg_flags = peer_table(pg_flags, "pg_flags");
  CakePeerPointerTableHost<kPeerTableCapacity> v_push_inbox_peers =
      peer_table(push_inbox_peers, "push_inbox_peers");
  CakePeerPointerTableHost<kPeerTableCapacity> v_push_flags_peers =
      peer_table(push_flags_peers, "push_flags_peers");
  check_self_entry(push_inbox_peers, pg_rank, push_inbox, "push_inbox_peers", "push_inbox");
  check_self_entry(push_flags_peers, pg_rank, push_flags, "push_flags_peers", "push_flags");
  // One block per received token (grid formula 16 * tokens_per_rank): the
  // program derives the source rank from blockIdx.x without a guard.
  TVM_FFI_CHECK(grid_x == kWorldSize * tokens_per_rank, ValueError)
      << "grid_x must equal 16 * tokens_per_rank = " << kWorldSize * tokens_per_rank << ", got "
      << grid_x;

  const cudaStream_t stream = get_stream(dev);

  // Set the program's preferred shared-memory carveout once (the manifest's
  // function attribute for this program alone); function-local static
  // initialization already runs once and thread-safely.
  static const bool carveout_ready = [] {
    TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_local_pack),
        cudaFuncAttributePreferredSharedMemoryCarveout, kPreferredSharedMemoryCarveout));
    return true;
  }();
  (void)carveout_ready;

  float* p_recv_weights = static_cast<float*>(recv_weights.data_ptr());
  int* p_expert_counts = static_cast<int*>(expert_counts.data_ptr());
  int* p_route_map = static_cast<int*>(route_map.data_ptr());
  int* p_inverse_rows = static_cast<int*>(inverse_rows.data_ptr());
  int* p_overflow = static_cast<int*>(overflow.data_ptr());
  uint8_t* p_quantized = static_cast<uint8_t*>(quantized.data_ptr());
  uint8_t* p_scales_cp = static_cast<uint8_t*>(scales_cp.data_ptr());
  unsigned int* p_push_generation = static_cast<unsigned int*>(push_generation.data_ptr());
  int32_t v_tokens_per_rank = static_cast<int32_t>(tokens_per_rank);
  int32_t v_active_rows_per_expert = static_cast<int32_t>(active_rows_per_expert);
  int32_t v_scale_tile_k128 = static_cast<int32_t>(scale_tile_k128);
  int32_t v_pg_world = static_cast<int32_t>(pg_world);
  int32_t v_pg_rank = static_cast<int32_t>(pg_rank);
  uint8_t* p_push_inbox = static_cast<uint8_t*>(push_inbox.data_ptr());
  unsigned int* p_push_flags = static_cast<unsigned int*>(push_flags.data_ptr());
  void* kargs[] = {&p_recv_weights,    &p_expert_counts,
                   &p_route_map,       &p_inverse_rows,
                   &p_overflow,        &p_quantized,
                   &p_scales_cp,       &p_push_generation,
                   &v_tokens_per_rank, &v_active_rows_per_expert,
                   &v_scale_tile_k128, &v_pg_world,
                   &v_pg_rank,         &v_pg_flags,
                   &p_push_inbox,      &v_push_inbox_peers,
                   &p_push_flags,      &v_push_flags_peers};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_local_pack), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_local_pack) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_local_pack

TVM_FFI_DLL_EXPORT_TYPED_FUNC(local_pack, cake_w8a8_megamoe_ep16_local_pack::Run);
