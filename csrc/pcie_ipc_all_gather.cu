/*
 * Copyright (c) 2026 by FlashInfer team.
 * Modifications Copyright (c) 2026 by the pcie_collectives contributors.
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
#include <tvm/ffi/container/array.h>

#include <cstdint>
#include <limits>

#include "flashinfer/comm/pcie_ipc_all_gather.cuh"
#include "tvm_ffi_utils.h"

namespace fi = flashinfer::comm::pcie_ipc::all_gather;
namespace common = flashinfer::comm::pcie_ipc::common;

using tvm::ffi::Array;

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace {

struct PcieIpcAllGatherHandle {
  fi::PeerViews views;
  fi::WorkspaceLayout layout;
  int rank;
  int device;
};

void check_workspace_config(int64_t world_size, int64_t max_numel, int64_t element_size,
                            int64_t max_blocks, bool enable_copy_engine) {
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8)
      << "pcie ipc all-gather supports world_size 2, 4 or 8, got " << world_size;
  TVM_FFI_ICHECK_GT(max_numel, 0) << "max_numel must be positive";
  TVM_FFI_ICHECK(element_size == 2 || element_size == 4)
      << "element_size must be 2 or 4 for bfloat16, float16 or float32, got " << element_size;
  TVM_FFI_ICHECK_EQ(max_numel % (common::kPackBytes / element_size), 0)
      << "max_numel must contain complete " << common::kPackBytes << "-byte packs";
  TVM_FFI_ICHECK(max_blocks > 0 && max_blocks <= common::kMaxBlocks)
      << "max_blocks must be in (0, " << common::kMaxBlocks << "], got " << max_blocks;
  TVM_FFI_ICHECK(!enable_copy_engine || world_size == fi::kCopyEngineWorldSize)
      << "CopyEngine storage requires world_size " << fi::kCopyEngineWorldSize;
}

int tensor_element_size(DLDataType dtype) {
  switch (encode_dlpack_dtype(dtype)) {
    case bfloat16_code:
    case float16_code:
      return 2;
    case float32_code:
      return 4;
    default:
      return 0;
  }
}

}  // namespace

int64_t pcie_ipc_all_gather_workspace_size(int64_t world_size, int64_t max_numel,
                                           int64_t element_size, int64_t max_blocks,
                                           bool enable_copy_engine) {
  check_workspace_config(world_size, max_numel, element_size, max_blocks, enable_copy_engine);
  int64_t bytes = 0;
  TVM_FFI_ICHECK(fi::try_workspace_size(static_cast<int>(world_size), max_numel,
                                        static_cast<int>(element_size),
                                        static_cast<int>(max_blocks), enable_copy_engine, &bytes))
      << "workspace layout overflows size_t or int64";
  return bytes;
}

/*! \brief Bind one already-shared CUDA IPC slab per rank.
 *
 * The caller must execute a collective barrier after every rank returns from
 * init. A peer that starts writing before the local memset completes could
 * otherwise lose either payload or signal state.
 */
fptr_t pcie_ipc_all_gather_init(Array<fptr_t> ipc_ptrs, int64_t rank, int64_t max_numel,
                                int64_t element_size, int64_t max_blocks, bool enable_copy_engine) {
  const int world_size = static_cast<int>(ipc_ptrs.size());
  check_workspace_config(world_size, max_numel, element_size, max_blocks, enable_copy_engine);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "rank " << rank << " is out of range";

  int64_t ptrs[common::kMaxWorldSize];
  for (int peer = 0; peer < world_size; ++peer) {
    TVM_FFI_ICHECK_NE(ipc_ptrs[peer], 0) << "ipc_ptrs[" << peer << "] is null";
    TVM_FFI_ICHECK_EQ(static_cast<uint64_t>(ipc_ptrs[peer]) & 15ULL, 0ULL)
        << "ipc_ptrs[" << peer << "] must be 16-byte aligned";
    ptrs[peer] = ipc_ptrs[peer];
  }

  fi::WorkspaceLayout layout{};
  TVM_FFI_ICHECK(
      fi::try_compute_workspace_layout(world_size, max_numel, static_cast<int>(element_size),
                                       static_cast<int>(max_blocks), enable_copy_engine, &layout))
      << "workspace layout overflows size_t or int64";

  auto* handle = new PcieIpcAllGatherHandle();
  handle->layout = layout;
  handle->views = fi::make_peer_views(ptrs, world_size, handle->layout);
  handle->rank = static_cast<int>(rank);
  cudaError_t error = cudaGetDevice(&handle->device);
  if (error == cudaSuccess) {
    error = cudaMemset(reinterpret_cast<void*>(ptrs[rank]), 0, handle->layout.total_bytes);
  }
  if (error != cudaSuccess) {
    delete handle;
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "failed to initialize the pcie ipc all-gather workspace: " << cudaGetErrorString(error);
  }
  return reinterpret_cast<fptr_t>(handle);
}

void pcie_ipc_all_gather_dispose(fptr_t handle) {
  delete reinterpret_cast<PcieIpcAllGatherHandle*>(handle);
}

void pcie_ipc_all_gather(fptr_t handle, TensorView inp, TensorView out, int64_t blocks,
                         int64_t threads, int64_t variant) {
  TVM_FFI_ICHECK_NE(handle, 0) << "pcie ipc all-gather workspace is closed";
  auto* h = reinterpret_cast<PcieIpcAllGatherHandle*>(handle);

  TVM_FFI_ICHECK_EQ(inp.device().device_type, kDLCUDA) << "input must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(out.device().device_type, kDLCUDA) << "output must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(inp.device().device_id, out.device().device_id)
      << "input and output must be on the same CUDA device";
  TVM_FFI_ICHECK_EQ(inp.device().device_id, h->device)
      << "tensor device differs from the device used to initialize the workspace";
  TVM_FFI_ICHECK(inp.IsContiguous() && out.IsContiguous()) << "input and output must be contiguous";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(inp.dtype()), encode_dlpack_dtype(out.dtype()))
      << "input and output must have the same dtype";
  const int element_size = tensor_element_size(inp.dtype());
  TVM_FFI_ICHECK_NE(element_size, 0)
      << "pcie ipc all-gather supports bfloat16, float16 and float32";
  TVM_FFI_ICHECK_EQ(element_size, h->layout.element_size)
      << "tensor dtype does not match the workspace element size";

  const int64_t numel = inp.numel();
  TVM_FFI_ICHECK_GT(numel, 0) << "input must be non-empty";
  TVM_FFI_ICHECK_LE(numel, h->layout.max_numel) << "input exceeds workspace capacity";
  TVM_FFI_ICHECK_EQ(numel % (common::kPackBytes / element_size), 0)
      << "input must contain complete " << common::kPackBytes << "-byte packs";
  TVM_FFI_ICHECK_LE(numel, std::numeric_limits<int64_t>::max() / h->layout.world_size)
      << "output size overflows int64";
  TVM_FFI_ICHECK_EQ(out.numel(), numel * h->layout.world_size)
      << "output must contain world_size rank-major input shards";

  const uintptr_t input_begin = reinterpret_cast<uintptr_t>(inp.data_ptr());
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(out.data_ptr());
  const size_t input_bytes = static_cast<size_t>(numel) * element_size;
  const size_t output_bytes = input_bytes * h->layout.world_size;
  TVM_FFI_ICHECK_EQ(input_begin & 15ULL, 0ULL) << "input must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(output_begin & 15ULL, 0ULL) << "output must be 16-byte aligned";
  TVM_FFI_ICHECK(input_begin + input_bytes <= output_begin ||
                 output_begin + output_bytes <= input_begin)
      << "input and output must not overlap";

  TVM_FFI_ICHECK(blocks > 0 && blocks <= h->layout.max_blocks)
      << "blocks must be in (0, " << h->layout.max_blocks << "], got " << blocks;
  TVM_FFI_ICHECK(threads >= 32 && threads <= common::kMaxThreads && threads % 32 == 0)
      << "threads must be warp-aligned and in [32, " << common::kMaxThreads << "], got " << threads;
  TVM_FFI_ICHECK(variant >= 0 && variant < fi::kVariantCount)
      << "variant must be in [0, " << fi::kVariantCount << "), got " << variant;

  const auto algorithm = static_cast<fi::Variant>(variant);
  TVM_FFI_ICHECK(!(algorithm == fi::Variant::kFlatPush && h->layout.world_size != 4))
      << "kFlatPush is admitted only for world_size 4";
  TVM_FFI_ICHECK(!(algorithm == fi::Variant::kCopyEngine && h->layout.world_size != 8))
      << "kCopyEngine is admitted only for world_size 8";
  if (algorithm == fi::Variant::kCopyEngine) {
    TVM_FFI_ICHECK_EQ(blocks, 1) << "kCopyEngine requires blocks=1";
    TVM_FFI_ICHECK_EQ(threads, 32) << "kCopyEngine requires threads=32";
    TVM_FFI_ICHECK_NE(h->views.copy_engine_ptrs[h->rank], 0)
        << "CopyEngine staging was not provisioned";
  }

  ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  const cudaStream_t stream = get_stream(inp.device());
  const cudaError_t error =
      fi::all_gather(inp.data_ptr(), out.data_ptr(), numel, h->views, h->rank, h->layout,
                     static_cast<int>(blocks), static_cast<int>(threads), algorithm, stream);
  if (error != cudaSuccess) {
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "pcie ipc all-gather launch failed: " << cudaGetErrorString(error);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_all_gather_workspace_size,
                              pcie_ipc_all_gather_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_all_gather_init, pcie_ipc_all_gather_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_all_gather_dispose, pcie_ipc_all_gather_dispose);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_all_gather, pcie_ipc_all_gather);
