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
#include <tvm/ffi/container/array.h>

#include <cstdint>

#include "flashinfer/comm/pcie_ipc_all_reduce.cuh"
#include "tvm_ffi_utils.h"

namespace fi = flashinfer::comm::pcie_ipc;

using tvm::ffi::Array;

// Opaque handle, matching the fptr_t convention used by the other custom
// all-reduce bindings in this directory.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace {

// Everything the launcher needs that does not change between calls. The
// workspace itself is owned by the caller (see pcie_ipc_all_reduce.cuh).
struct PcieIpcHandle {
  fi::PeerViews views;
  fi::WorkspaceLayout layout;
  int rank;
  int world_size;
  int max_blocks;
  int64_t max_numel;
  int elem_size;
};

}  // namespace

/*!
 * \brief Bytes each rank must allocate and share over CUDA IPC.
 *
 * The caller passes the result to create_shared_buffer() and hands the
 * resulting pointer array to pcie_ipc_init().
 */
int64_t pcie_ipc_workspace_size(int64_t world_size, int64_t max_numel, int64_t elem_size,
                                int64_t max_blocks) {
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8)
      << "pcie ipc all-reduce supports world_size 2, 4 or 8, got " << world_size;
  TVM_FFI_ICHECK_GT(max_numel, 0) << "max_numel must be positive";
  TVM_FFI_ICHECK_EQ(elem_size, 2)
      << "only 2-byte dtypes (bfloat16, float16) are supported, got elem_size " << elem_size;
  TVM_FFI_ICHECK_GT(max_blocks, 0) << "max_blocks must be positive";
  return fi::workspace_size(static_cast<int>(world_size), max_numel, static_cast<int>(elem_size),
                            static_cast<int>(max_blocks));
}

/*!
 * \brief Bind an already-shared workspace and return an opaque handle.
 *
 * \param ipc_ptrs Peer pointers; entry i must address rank i's slab.
 *
 * The slab is zeroed here because the sentinel protocol reads +0.0 as "not yet
 * written". The caller MUST barrier after this returns and before the first
 * collective: a peer that starts pushing into this slab before we zero it
 * would lose its payload.
 */
fptr_t pcie_ipc_init(Array<fptr_t> ipc_ptrs, int64_t rank, int64_t max_numel, int64_t elem_size,
                     int64_t max_blocks) {
  const int world_size = static_cast<int>(ipc_ptrs.size());
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8)
      << "pcie ipc all-reduce supports world_size 2, 4 or 8, got " << world_size;
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "rank " << rank << " out of range";
  TVM_FFI_ICHECK_EQ(elem_size, 2)
      << "only 2-byte dtypes (bfloat16, float16) are supported, got elem_size " << elem_size;
  TVM_FFI_ICHECK_GT(max_blocks, 0) << "max_blocks must be positive";

  int64_t ptrs[fi::kMaxWorldSize];
  for (int i = 0; i < world_size; ++i) {
    TVM_FFI_ICHECK_NE(ipc_ptrs[i], 0) << "ipc_ptrs[" << i << "] is null";
    ptrs[i] = ipc_ptrs[i];
  }

  auto* handle = new PcieIpcHandle();
  handle->layout = fi::compute_workspace_layout(world_size, max_numel, static_cast<int>(elem_size),
                                                static_cast<int>(max_blocks));
  handle->views = fi::make_peer_views(ptrs, world_size, static_cast<int>(rank), handle->layout);
  handle->rank = static_cast<int>(rank);
  handle->world_size = world_size;
  handle->max_blocks = static_cast<int>(max_blocks);
  handle->max_numel = max_numel;
  handle->elem_size = static_cast<int>(elem_size);

  cudaError_t err = cudaMemset(reinterpret_cast<void*>(ptrs[rank]), 0, handle->layout.total_bytes);
  if (err != cudaSuccess) {
    delete handle;
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "failed to zero the pcie ipc workspace: " << cudaGetErrorString(err);
  }
  return reinterpret_cast<fptr_t>(handle);
}

void pcie_ipc_dispose(fptr_t handle) { delete reinterpret_cast<PcieIpcHandle*>(handle); }

/*!
 * \brief Out-of-place all-reduce over the shared workspace.
 *
 * \param blocks,threads,variant Launch configuration chosen by the caller;
 *        \c variant is a fi::Variant and the (world_size, variant) pairs that
 *        dispatch are listed in pcie_ipc_all_reduce.cuh.
 */
void pcie_ipc_all_reduce(fptr_t handle, TensorView inp, TensorView out, int64_t blocks,
                         int64_t threads, int64_t variant, bool enable_pdl) {
  auto* h = reinterpret_cast<PcieIpcHandle*>(handle);
  ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  auto stream = get_stream(inp.device());

  TVM_FFI_ICHECK(inp.IsContiguous() && out.IsContiguous()) << "input and output must be contiguous";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(inp.dtype()), encode_dlpack_dtype(out.dtype()))
      << "input and output dtype must match";
  TVM_FFI_ICHECK_EQ(inp.numel(), out.numel()) << "input and output must have the same size";

  const int64_t numel = inp.numel();
  const int64_t elem_size = get_element_size(inp);
  TVM_FFI_ICHECK_EQ(elem_size, h->elem_size)
      << "dtype element size " << elem_size << " does not match the workspace's " << h->elem_size;
  TVM_FFI_ICHECK_LE(static_cast<size_t>(numel * elem_size), h->layout.max_payload_bytes)
      << "payload exceeds the workspace capacity";

  const int64_t pack_elems = 16 / elem_size;
  TVM_FFI_ICHECK_EQ(numel % pack_elems, 0)
      << "numel must be divisible by the 16-byte pack width (" << pack_elems << ")";
  TVM_FFI_ICHECK_EQ(h->max_numel % pack_elems, 0)
      << "max_numel must be divisible by the 16-byte pack width";
  TVM_FFI_ICHECK(blocks > 0 && blocks <= h->max_blocks)
      << "blocks must be in (0, " << h->max_blocks << "], got " << blocks;
  TVM_FFI_ICHECK(threads > 0 && threads <= 1024) << "threads must be in (0, 1024], got " << threads;
  // Every barrier signals from threadIdx.x < world_size, so a narrower block
  // leaves some peers with nobody to signal them and the collective hangs.
  TVM_FFI_ICHECK_GE(threads, h->world_size)
      << "threads must be at least world_size (" << h->world_size << "), got " << threads;
  // Refused rather than silently wrong: ipc_topo_rsag8_block_param_kernel
  // triggers launch completion before island_owner_ack and its barrier flag
  // store, so a dependent kernel can start while this call's phase-4 state is
  // still being written. Re-enabling needs that release moved past both stores,
  // an audit of the other six, and an SM90 regression.
  TVM_FFI_ICHECK(!enable_pdl)
      << "enable_pdl is not supported yet: in the TP8 block kernel the launch-completion "
         "trigger precedes the island ack and barrier flag stores";
  TVM_FFI_ICHECK(variant >= 0 && variant < fi::kVariantCount)
      << "variant must be in [0, " << fi::kVariantCount << "), got " << variant;
  const auto algo = static_cast<fi::Variant>(variant);
  // Reject rather than silently alias, so one configuration always names one
  // kernel.
  TVM_FFI_ICHECK(
      !(h->world_size == 2 && algo != fi::Variant::kUnstaged && algo != fi::Variant::kStaged))
      << "world_size 2 accepts only kUnstaged and kStaged, got variant " << variant;
  TVM_FFI_ICHECK(!(algo == fi::Variant::kFlatStaged && h->world_size != 8))
      << "kFlatStaged is world_size 8 only, got " << h->world_size;
  // Only the block-partitioned TP8 kernel needs this: it derives its chunk
  // from blockIdx.x & 3. Every other kernel uses flat grid-stride loops and
  // accepts any block count.
  if (h->world_size == 8 && algo == fi::Variant::kStaged) {
    TVM_FFI_ICHECK_EQ(blocks % 4, 0)
        << "the TP8 topology kernel requires blocks divisible by 4, got " << blocks;
  }

  cudaError_t err = cudaSuccess;
  switch (encode_dlpack_dtype(out.dtype())) {
    case bfloat16_code:
      err = fi::all_reduce<nv_bfloat16>(static_cast<const nv_bfloat16*>(inp.data_ptr()),
                                        static_cast<nv_bfloat16*>(out.data_ptr()), numel, h->views,
                                        h->rank, h->world_size, h->max_blocks, h->max_numel,
                                        static_cast<int>(blocks), static_cast<int>(threads), algo,
                                        enable_pdl, stream);
      break;
    case float16_code:
      err = fi::all_reduce<half>(
          static_cast<const half*>(inp.data_ptr()), static_cast<half*>(out.data_ptr()), numel,
          h->views, h->rank, h->world_size, h->max_blocks, h->max_numel, static_cast<int>(blocks),
          static_cast<int>(threads), algo, enable_pdl, stream);
      break;
    default:
      // The kernel templates carry a generic path, but only the two 2-byte
      // dtypes are instantiated and measured.
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "pcie ipc all-reduce supports bfloat16 and float16 only";
  }
  if (err != cudaSuccess) {
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "pcie ipc all-reduce launch failed: " << cudaGetErrorString(err);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_workspace_size, pcie_ipc_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_init, pcie_ipc_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_dispose, pcie_ipc_dispose);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_all_reduce, pcie_ipc_all_reduce);
