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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), stage `fc2`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_fc2_kernel.cu): the persistent
// SwapAB FC2 GEMM that pushes each routed output row back to its source rank
// and, when its last CTA has stored its routes, signals the forward's
// completion into every rank's symmetric `fc2_completion` counter (manifest
// route_contract.fc2_completion). The binding checks the tensor metadata of
// the route contract, encodes the four TMA tensor maps on the host and passes
// them by value (A over the k-tile-major w2 payload in 128-row boxes, B over
// the row-major activation rows), builds the by-value peer tables from the 16
// peer addresses, opts the kernel into its dynamic shared memory once,
// resolves the environment stream and performs one launch carrying the
// programmatic-stream-serialization attribute (the fc1 -> fc2 edge of the
// route's programmatic dependent launch chain); the block size and the
// dynamic shared memory are the generated program's constants.
//
// Completion counter (the two trailing device arguments, in the manifest
// order after remote_terms_peers): `fc2_completion` is this rank's symmetric
// u32 [2] (word 0 = done counter, +1 from every rank per forward, monotonic
// and never reset; word 1 = this rank's CTA arrival counter, back to 0 at the
// end of every forward), bound as a plain pointer (no __restrict__: peers add
// into word 0 while the kernel runs), and `fc2_completion_peers` is its
// 32-slot peer table (entries [0, 16) = the 16 ranks' base addresses). The
// host zeroes the counter exactly once at workspace allocation and never
// writes it afterwards. Slot remote_pg_rank of fc2_completion_peers and of
// remote_terms_peers must be the local pointer bound beside the table
// (symmetric self entry): the program counts its CTA arrivals and resets
// word 1 through the local pointer but sends its own +1 through the table,
// and it pushes its own rank's routes through the table while combine reads
// them through the local pointer.

#include <cuda.h>
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
static_assert(sizeof(CakePeerPointerTable<__nv_bfloat16, 32>) == 256 &&
                  alignof(CakePeerPointerTable<__nv_bfloat16, 32>) == 16,
              "CakePeerPointerTable<__nv_bfloat16, 32> must be 256 bytes, 16-byte aligned");

// Device entry point of the kernel translation unit (manifest declaration).
// clang-format off
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_fc2(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ expert_row_offsets, int routes_per_rank, int N, int K_tiles, int grid_m, int32_t remote_pg_world, int32_t remote_pg_rank, CakePeerPointerTable<unsigned int, 32> remote_pg_flags, __nv_bfloat16* __restrict__ remote_terms, CakePeerPointerTable<__nv_bfloat16, 32> remote_terms_peers, unsigned int* fc2_completion, CakePeerPointerTable<unsigned int, 32> fc2_completion_peers);
// clang-format on

namespace cake_w8a8_megamoe_ep16_fc2 {

constexpr int64_t kWorldSize = 16;
constexpr int64_t kLocalExperts = 32;
constexpr int64_t kHidden = 3072;
constexpr int64_t kIntermediate = 5120;
constexpr int64_t kMaxRowsPerExpert = 128;   // packed FC2 input rows per expert
constexpr int64_t kW2ScaleTiles = 960;       // 128x128 scale tiles per expert of w2
constexpr int64_t kFc2InputScaleTiles = 40;  // 128x128 scale tiles per expert of the input
constexpr int64_t kScaleTileRows128 = 16;    // a 128x128 scale tile is [16, 32] bytes
constexpr int64_t kScaleTileColumns = 32;
constexpr int64_t kMaxRecvTokens = 1024;  // 16 ranks x 64 tokens
constexpr int64_t kTopK = 8;
constexpr int64_t kMaxRoutesPerRank = 512;  // 64 tokens x top-8
constexpr int64_t kTotalExpertRows = 4096;  // 32 experts x 128 packed rows
constexpr int64_t kFc2Tiles = 768;          // 32 experts x 24 M128 tiles
constexpr int64_t kK128 = 128;
constexpr cuuint32_t kW2BoxRows = 128;  // k-tile-major w2 payload: one R x 128 B block per box
// fc2_completion u32 [2]: [0] done counter (16 adds per forward), [1] CTA arrivals.
constexpr int64_t kFc2CompletionWords = 2;
constexpr int kPeerTableCapacity = 32;
constexpr unsigned kBlockThreads = 192;
constexpr unsigned kDynamicSmemBytes = 181248;

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

// 4-D tensor map of the k-tile-major uint8 [outer, rows, k] weight payload
// (manifest route_contract.weight_layout_rule: byte (e, r, c) lives at
// (((e * rows / R + r / R) * (k / 128) + c / 128) * R + r % R) * 128 + c % 128,
// i.e. [outer][rows / R][k / 128][R][128 B]), viewed as
// [128, R, k / 128, outer * rows / R] with 128-byte swizzle: global_dim
// {128, R, k / 128, outer * rows / R}, byte strides {128, R * 128, R * k}, box
// {128, R, 1, 1}; one box is one contiguous R x 128-byte block (the manifest's
// A tma record).
CUtensorMap encode_weight(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' needs at least 2 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t k = t.size(t.ndim() - 1);
  const int64_t rows = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(k > 0 && rows > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  TVM_FFI_CHECK(k % kK128 == 0, ValueError)
      << "TMA source '" << name << "' extent " << k << " must divide exactly by " << kK128;
  const int64_t outer = t.numel() / (k * rows);
  TVM_FFI_CHECK((outer * rows) % box_rows == 0, ValueError)
      << "TMA source '" << name << "' row count " << outer * rows
      << " must divide exactly by the box height " << box_rows;
  const cuuint64_t global_dim[4] = {128, static_cast<cuuint64_t>(box_rows),
                                    static_cast<cuuint64_t>(k / kK128),
                                    static_cast<cuuint64_t>(outer * rows / box_rows)};
  const cuuint64_t global_strides[3] = {128, static_cast<cuuint64_t>(box_rows) * 128,
                                        static_cast<cuuint64_t>(box_rows) * k};
  const cuuint32_t box_dim[4] = {128, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

// 4-D tensor map of a row-major uint8 [outer, rows, k] activation operand, viewed
// as [128, rows, k / 128, outer] with 128-byte swizzle; one box covers 128 bytes
// of K for `box_rows` rows.
CUtensorMap encode_operand(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' needs at least 2 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t k = t.size(t.ndim() - 1);
  const int64_t rows = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(k > 0 && rows > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  TVM_FFI_CHECK(k % kK128 == 0, ValueError)
      << "TMA source '" << name << "' extent " << k << " must divide exactly by " << kK128;
  const int64_t outer = t.numel() / (k * rows);
  const cuuint64_t global_dim[4] = {128, static_cast<cuuint64_t>(rows),
                                    static_cast<cuuint64_t>(k / kK128),
                                    static_cast<cuuint64_t>(outer)};
  const cuuint64_t global_strides[3] = {static_cast<cuuint64_t>(k), 128,
                                        static_cast<cuuint64_t>(rows * k)};
  const cuuint32_t box_dim[4] = {128, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

// 4-D tensor map of a row-major uint8 [outer, tiles, rows, 32] scale plane, read as
// [32, rows, tiles, outer] without swizzle; one box covers 32 bytes for `box_rows` rows.
CUtensorMap encode_scales(const TensorView& t, cuuint32_t box_rows, const char* name) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source '" << name << "' needs at least 3 dims, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  const int64_t d1 = t.size(t.ndim() - 1);
  const int64_t d2 = t.size(t.ndim() - 2);
  const int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  const int64_t outer = t.numel() / (d1 * d2 * d3);
  const cuuint64_t global_dim[4] = {static_cast<cuuint64_t>(d1), static_cast<cuuint64_t>(d2),
                                    static_cast<cuuint64_t>(d3), static_cast<cuuint64_t>(outer)};
  const cuuint64_t global_strides[3] = {static_cast<cuuint64_t>(d1),
                                        static_cast<cuuint64_t>(d1 * d2),
                                        static_cast<cuuint64_t>(d1 * d2 * d3)};
  const cuuint32_t box_dim[4] = {32, box_rows, 1, 1};
  const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
  for (int axis = 0; axis < 4; ++axis) {
    TVM_FFI_CHECK(box_dim[axis] <= global_dim[axis], ValueError)
        << "TMA box dim " << axis << " (" << box_dim[axis] << ") exceeds global dim "
        << global_dim[axis] << " for '" << name << "'";
  }
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled(4D, '" << name
      << "') failed: CUresult=" << static_cast<int>(result);
  return tensor_map;
}

void Run(TensorView A, TensorView B, TensorView SFA, TensorView SFB, TensorView C,
         TensorView route_map, TensorView tile_expert, TensorView tile_m_local,
         TensorView expert_row_offsets, int64_t routes_per_rank, int64_t N, int64_t K_tiles,
         int64_t grid_m, int64_t remote_pg_world, int64_t remote_pg_rank,
         tvm::ffi::Shape remote_pg_flags, TensorView remote_terms,
         tvm::ffi::Shape remote_terms_peers, TensorView fc2_completion,
         tvm::ffi::Shape fc2_completion_peers, int64_t grid_x) {
  check_cuda_tensor(A, "A");
  const DLDevice dev = A.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);
  {
    // The encoders need a current driver context on this thread: bind the primary
    // context once per thread per device (the runtime launch binds it itself).
    static thread_local uint64_t bound_devices = 0;
    const uint64_t device_bit = dev.device_id < 64 ? uint64_t{1} << dev.device_id : uint64_t{0};
    if (device_bit == 0 || (bound_devices & device_bit) == 0) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      bound_devices |= device_bit;
    }
  }

  check_dtype(A, dl_uint8, "A");
  check_contiguous(A, "A");
  check_cuda_tensor(B, "B");
  check_dtype(B, dl_uint8, "B");
  check_contiguous(B, "B");
  check_same_device(B, A, "B", "A");
  check_cuda_tensor(SFA, "SFA");
  check_dtype(SFA, dl_uint8, "SFA");
  check_contiguous(SFA, "SFA");
  check_same_device(SFA, A, "SFA", "A");
  check_cuda_tensor(SFB, "SFB");
  check_dtype(SFB, dl_uint8, "SFB");
  check_contiguous(SFB, "SFB");
  check_same_device(SFB, A, "SFB", "A");
  check_cuda_tensor(C, "C");
  check_dtype(C, dl_bfloat16, "C");
  check_contiguous(C, "C");
  check_same_device(C, A, "C", "A");
  check_cuda_tensor(route_map, "route_map");
  check_dtype(route_map, dl_int32, "route_map");
  check_contiguous(route_map, "route_map");
  check_same_device(route_map, A, "route_map", "A");
  check_cuda_tensor(tile_expert, "tile_expert");
  check_dtype(tile_expert, dl_int32, "tile_expert");
  check_contiguous(tile_expert, "tile_expert");
  check_same_device(tile_expert, A, "tile_expert", "A");
  check_cuda_tensor(tile_m_local, "tile_m_local");
  check_dtype(tile_m_local, dl_int32, "tile_m_local");
  check_contiguous(tile_m_local, "tile_m_local");
  check_same_device(tile_m_local, A, "tile_m_local", "A");
  check_cuda_tensor(expert_row_offsets, "expert_row_offsets");
  check_dtype(expert_row_offsets, dl_int32, "expert_row_offsets");
  check_contiguous(expert_row_offsets, "expert_row_offsets");
  check_same_device(expert_row_offsets, A, "expert_row_offsets", "A");
  check_cuda_tensor(remote_terms, "remote_terms");
  check_dtype(remote_terms, dl_bfloat16, "remote_terms");
  check_contiguous(remote_terms, "remote_terms");
  check_same_device(remote_terms, A, "remote_terms", "A");
  check_cuda_tensor(fc2_completion, "fc2_completion");
  check_dtype(fc2_completion, dl_uint32, "fc2_completion");
  check_contiguous(fc2_completion, "fc2_completion");
  check_same_device(fc2_completion, A, "fc2_completion", "A");

  check_extents(A, {kLocalExperts, kHidden, kIntermediate}, "A");
  check_extents(B, {kLocalExperts, kMaxRowsPerExpert, kIntermediate}, "B");
  check_extents(SFA, {kLocalExperts, kW2ScaleTiles, kScaleTileRows128, kScaleTileColumns}, "SFA");
  check_extents(SFB, {kLocalExperts, kFc2InputScaleTiles, kScaleTileRows128, kScaleTileColumns},
                "SFB");
  check_extents(C, {kMaxRecvTokens, kTopK, kHidden}, "C");
  check_numel(route_map, kTotalExpertRows, "route_map");
  check_numel(tile_expert, kFc2Tiles, "tile_expert");
  check_numel(tile_m_local, kFc2Tiles, "tile_m_local");
  check_numel(expert_row_offsets, kLocalExperts, "expert_row_offsets");
  check_extents(remote_terms, {kMaxRecvTokens, kTopK, kHidden}, "remote_terms");
  check_extents(fc2_completion, {kFc2CompletionWords}, "fc2_completion");
  TVM_FFI_CHECK(routes_per_rank >= 1 && routes_per_rank <= kMaxRoutesPerRank, ValueError)
      << "routes_per_rank must be in [1, " << kMaxRoutesPerRank << "], got " << routes_per_rank;
  TVM_FFI_CHECK(N == A.size(1) && N == remote_terms.size(2), ValueError)
      << "N must equal the hidden size A.size(1)=" << A.size(1) << ", got " << N;
  TVM_FFI_CHECK(K_tiles * kK128 == A.size(2), ValueError)
      << "K_tiles * 128 must equal A.size(2)=" << A.size(2) << ", got K_tiles=" << K_tiles;
  TVM_FFI_CHECK(grid_m == tile_expert.numel(), ValueError)
      << "grid_m must equal the tile map length " << tile_expert.numel() << ", got " << grid_m;
  TVM_FFI_CHECK(remote_pg_world == kWorldSize, ValueError)
      << "remote_pg_world must be " << kWorldSize << ", got " << remote_pg_world;
  TVM_FFI_CHECK(remote_pg_rank >= 0 && remote_pg_rank < remote_pg_world, ValueError)
      << "remote_pg_rank must be in [0, " << remote_pg_world << "), got " << remote_pg_rank;
  CakePeerPointerTableHost<kPeerTableCapacity> v_remote_pg_flags =
      peer_table(remote_pg_flags, "remote_pg_flags");
  CakePeerPointerTableHost<kPeerTableCapacity> v_remote_terms_peers =
      peer_table(remote_terms_peers, "remote_terms_peers");
  CakePeerPointerTableHost<kPeerTableCapacity> v_fc2_completion_peers =
      peer_table(fc2_completion_peers, "fc2_completion_peers");
  check_self_entry(remote_terms_peers, remote_pg_rank, remote_terms, "remote_terms_peers",
                   "remote_terms");
  check_self_entry(fc2_completion_peers, remote_pg_rank, fc2_completion, "fc2_completion_peers",
                   "fc2_completion");
  TVM_FFI_CHECK(grid_x > 0 && grid_x <= 2147483647LL, ValueError)
      << "grid_x must be in [1, 2147483647], got " << grid_x;

  const cudaStream_t stream = get_stream(dev);

  // Opt the kernel into its dynamic shared memory once; function-local static
  // initialization already runs once and thread-safely.
  static const bool smem_ready = [] {
    TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_fc2),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicSmemBytes)));
    return true;
  }();
  (void)smem_ready;

  CUtensorMap tma_A = encode_weight(A, kW2BoxRows, "A");
  CUtensorMap tma_B = encode_operand(B, 32, "B");
  CUtensorMap tma_SFA = encode_scales(SFA, 16, "SFA");
  CUtensorMap tma_SFB = encode_scales(SFB, 16, "SFB");
  __nv_bfloat16* p_C = static_cast<__nv_bfloat16*>(C.data_ptr());
  int* p_route_map = static_cast<int*>(route_map.data_ptr());
  int* p_tile_expert = static_cast<int*>(tile_expert.data_ptr());
  int* p_tile_m_local = static_cast<int*>(tile_m_local.data_ptr());
  int* p_expert_row_offsets = static_cast<int*>(expert_row_offsets.data_ptr());
  int32_t v_routes_per_rank = static_cast<int32_t>(routes_per_rank);
  int32_t v_N = static_cast<int32_t>(N);
  int32_t v_K_tiles = static_cast<int32_t>(K_tiles);
  int32_t v_grid_m = static_cast<int32_t>(grid_m);
  int32_t v_remote_pg_world = static_cast<int32_t>(remote_pg_world);
  int32_t v_remote_pg_rank = static_cast<int32_t>(remote_pg_rank);
  __nv_bfloat16* p_remote_terms = static_cast<__nv_bfloat16*>(remote_terms.data_ptr());
  unsigned int* p_fc2_completion = static_cast<unsigned int*>(fc2_completion.data_ptr());
  void* kargs[] = {&tma_A,
                   &tma_B,
                   &tma_SFA,
                   &tma_SFB,
                   &p_C,
                   &p_route_map,
                   &p_tile_expert,
                   &p_tile_m_local,
                   &p_expert_row_offsets,
                   &v_routes_per_rank,
                   &v_N,
                   &v_K_tiles,
                   &v_grid_m,
                   &v_remote_pg_world,
                   &v_remote_pg_rank,
                   &v_remote_pg_flags,
                   &p_remote_terms,
                   &v_remote_terms_peers,
                   &p_fc2_completion,
                   &v_fc2_completion_peers};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  // Programmatic dependent launch (manifest launch.use_pdl): the program
  // prefills its first expert-weight tile through the static tile map, then
  // executes griddepcontrol.wait before it reads fc1's requantized activations
  // and the route tables, so the attribute only overlaps its prologue with the
  // primary's tail. In eager streams and under CUDA-graph capture this launch
  // is the fc1 -> fc2 programmatic edge. The program is also the primary of
  // the fc2 -> combine edge: a CTA executes griddepcontrol.launch_dependents
  // when it starts its last tile, and the combine (launched with the same
  // attribute) waits on the completion counter this program signals.
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_fc2), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_fc2) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_fc2

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fc2, cake_w8a8_megamoe_ep16_fc2::Run);
