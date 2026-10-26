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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), stage `combine`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_combine_kernel.cu): the counted
// source-local combine. Each CTA (one per 1024-column slice of a token)
// executes griddepcontrol.wait, then thread 0 reads `generation =
// push_generation[0]` (the word local_pack's last CTA advanced to g + 1 during
// forward g) and polls this rank's LOCAL `fc2_completion[0]` UNTIMED with
// ld.acquire.sys.global.u32 until it reached 16 * generation (a wrap-safe
// modular compare: a late peer stalls this rank, it never traps), then
// fence.acquire.sys and bar.sync; every thread then reads the eight term rows
// of its token with coherent ld.global.v4.b32 loads through the plain (no
// __restrict__) `local_terms` pointer and accumulates them in FP32 in route
// order (manifest route_contract.fc2_completion.wait_rule; the same order as
// the previous sliced combine, so the output is bit-identical). The binding
// checks the tensor metadata of the route contract, builds the by-value peer
// tables from the 16 peer addresses, resolves the environment stream and
// performs one launch with the generated program's block size, carrying the
// programmatic-stream-serialization attribute (the fc2 -> combine edge of the
// route's programmatic dependent launch chain; manifest launch.use_pdl).
// `fc2_completion` and `fc2_completion_peers` are the same symmetric u32 [2]
// counter and table the fc2 binding receives; the process-group arguments
// (pg_world, pg_rank, pg_flags) are the route's flag pad, bound because the
// program names the process group.

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

// Device entry point of the kernel translation unit (manifest declaration).
// clang-format off
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_combine(__nv_bfloat16* local_terms, __nv_bfloat16* __restrict__ output, unsigned int* __restrict__ push_generation, int tokens_per_rank, int source_rank, int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int, 32> pg_flags, unsigned int* fc2_completion, CakePeerPointerTable<unsigned int, 32> fc2_completion_peers);
// clang-format on

namespace cake_w8a8_megamoe_ep16_combine {

constexpr int64_t kWorldSize = 16;
constexpr int64_t kMaxTokensPerRank = 64;
constexpr int64_t kMaxRecvTokens = 1024;  // 16 ranks x 64 tokens
constexpr int64_t kTopK = 8;
constexpr int64_t kHidden = 3072;
constexpr int64_t kCtasPerToken = 3;  // three 1024-column slices per token
// push_generation u32 [2]: [0] forwards this rank completed, [1] local_pack CTA arrivals.
constexpr int64_t kPushGenerationWords = 2;
// fc2_completion u32 [2]: [0] done counter (16 adds per forward), [1] fc2 CTA arrivals.
constexpr int64_t kFc2CompletionWords = 2;
constexpr int kPeerTableCapacity = 32;
constexpr unsigned kBlockThreads = 128;

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

void Run(TensorView local_terms, TensorView output, TensorView push_generation,
         int64_t tokens_per_rank, int64_t source_rank, int64_t pg_world, int64_t pg_rank,
         tvm::ffi::Shape pg_flags, TensorView fc2_completion, tvm::ffi::Shape fc2_completion_peers,
         int64_t grid_x) {
  check_cuda_tensor(local_terms, "local_terms");
  const DLDevice dev = local_terms.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);

  check_dtype(local_terms, dl_bfloat16, "local_terms");
  check_contiguous(local_terms, "local_terms");
  check_cuda_tensor(output, "output");
  check_dtype(output, dl_bfloat16, "output");
  check_contiguous(output, "output");
  check_same_device(output, local_terms, "output", "local_terms");
  check_cuda_tensor(push_generation, "push_generation");
  check_dtype(push_generation, dl_uint32, "push_generation");
  check_contiguous(push_generation, "push_generation");
  check_same_device(push_generation, local_terms, "push_generation", "local_terms");
  check_cuda_tensor(fc2_completion, "fc2_completion");
  check_dtype(fc2_completion, dl_uint32, "fc2_completion");
  check_contiguous(fc2_completion, "fc2_completion");
  check_same_device(fc2_completion, local_terms, "fc2_completion", "local_terms");

  check_extents(local_terms, {kMaxRecvTokens, kTopK, kHidden}, "local_terms");
  check_extents(push_generation, {kPushGenerationWords}, "push_generation");
  check_extents(fc2_completion, {kFc2CompletionWords}, "fc2_completion");
  TVM_FFI_CHECK(tokens_per_rank >= 1 && tokens_per_rank <= kMaxTokensPerRank, ValueError)
      << "tokens_per_rank must be in [1, " << kMaxTokensPerRank << "], got " << tokens_per_rank;
  TVM_FFI_CHECK(source_rank >= 0 && source_rank < kWorldSize, ValueError)
      << "source_rank must be in [0, " << kWorldSize << "), got " << source_rank;
  TVM_FFI_CHECK(pg_world == kWorldSize, ValueError)
      << "pg_world must be " << kWorldSize << ", got " << pg_world;
  TVM_FFI_CHECK(pg_rank >= 0 && pg_rank < pg_world, ValueError)
      << "pg_rank must be in [0, " << pg_world << "), got " << pg_rank;
  TVM_FFI_CHECK(output.ndim() == 2, ValueError) << "output must be 2-D, got ndim=" << output.ndim();
  TVM_FFI_CHECK(output.size(1) == kHidden, ValueError)
      << "output.size(1) must be " << kHidden << ", got " << output.size(1);
  TVM_FFI_CHECK(output.size(0) >= tokens_per_rank, ValueError)
      << "output.size(0) must cover " << tokens_per_rank << " rows, got " << output.size(0);
  CakePeerPointerTableHost<kPeerTableCapacity> v_pg_flags = peer_table(pg_flags, "pg_flags");
  CakePeerPointerTableHost<kPeerTableCapacity> v_fc2_completion_peers =
      peer_table(fc2_completion_peers, "fc2_completion_peers");
  check_self_entry(fc2_completion_peers, pg_rank, fc2_completion, "fc2_completion_peers",
                   "fc2_completion");
  // Three blocks per token (grid formula 3 * tokens_per_rank), one per
  // 1024-column slice: the program writes the output slice of blockIdx.x
  // without a guard.
  TVM_FFI_CHECK(grid_x == kCtasPerToken * tokens_per_rank, ValueError)
      << "grid_x must equal " << kCtasPerToken
      << " * tokens_per_rank = " << kCtasPerToken * tokens_per_rank << ", got " << grid_x;

  const cudaStream_t stream = get_stream(dev);

  __nv_bfloat16* p_local_terms = static_cast<__nv_bfloat16*>(local_terms.data_ptr());
  __nv_bfloat16* p_output = static_cast<__nv_bfloat16*>(output.data_ptr());
  unsigned int* p_push_generation = static_cast<unsigned int*>(push_generation.data_ptr());
  int32_t v_tokens_per_rank = static_cast<int32_t>(tokens_per_rank);
  int32_t v_source_rank = static_cast<int32_t>(source_rank);
  int32_t v_pg_world = static_cast<int32_t>(pg_world);
  int32_t v_pg_rank = static_cast<int32_t>(pg_rank);
  unsigned int* p_fc2_completion = static_cast<unsigned int*>(fc2_completion.data_ptr());
  void* kargs[] = {
      &p_local_terms, &p_output,  &p_push_generation, &v_tokens_per_rank, &v_source_rank,
      &v_pg_world,    &v_pg_rank, &v_pg_flags,        &p_fc2_completion,  &v_fc2_completion_peers};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  // Programmatic dependent launch (manifest launch.use_pdl): every thread
  // executes griddepcontrol.wait before its first global read, and only then
  // thread 0 polls the local completion counter, so the attribute lets the
  // CTAs become resident beside fc2's tail (fc2 triggers when a CTA starts
  // its last tile) without reading anything early. In eager streams and under
  // CUDA-graph capture this launch is the fc2 -> combine programmatic edge.
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_combine), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_combine) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_combine

TVM_FFI_DLL_EXPORT_TYPED_FUNC(combine, cake_w8a8_megamoe_ep16_combine::Run);
