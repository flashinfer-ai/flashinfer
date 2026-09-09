/*
 * Copyright (c) 2026 by FlashInfer team.
 * Copyright (c) 2026 by the pcie_reduce_scatter_flat_safe_static contributors.
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
#include <limits>

#include "flashinfer/comm/pcie_ipc_reduce_scatter.cuh"
#include "tvm_ffi_utils.h"

namespace fi = flashinfer::comm::pcie_ipc::reduce_scatter;
namespace common = flashinfer::comm::pcie_ipc::common;

using tvm::ffi::Array;

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace {

struct PcieIpcReduceScatterHandle {
  fi::PeerViews views;
  fi::WorkspaceLayout layout;
  int rank;
  int device;
};

void check_layout_arguments(int64_t world_size, int64_t max_numel, int64_t element_size,
                            int64_t max_blocks) {
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 8)
      << "pcie ipc reduce-scatter supports world_size 2, 4 or 8, got " << world_size;
  TVM_FFI_ICHECK_GT(max_numel, 0) << "max_numel must be positive";
  TVM_FFI_ICHECK(element_size == 2 || element_size == 4)
      << "element_size must be 2 or 4 for bfloat16, float16 or float32, got " << element_size;
  TVM_FFI_ICHECK_EQ(max_numel % (common::kPackBytes / element_size), 0)
      << "max_numel must contain complete " << common::kPackBytes << "-byte packs";
  TVM_FFI_ICHECK(max_blocks > 0 && max_blocks <= fi::kMaxBlocks)
      << "max_blocks must be in (0, " << fi::kMaxBlocks << "], got " << max_blocks;
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

/*!
 * \brief Number of bytes each rank allocates and shares over CUDA IPC.
 *
 * \param max_numel Maximum number of elements in one rank's output shard.
 * \param max_blocks Maximum actual grid size accepted by the workspace.
 */
int64_t pcie_ipc_reduce_scatter_workspace_size(int64_t world_size, int64_t max_numel,
                                               int64_t element_size, int64_t max_blocks) {
  check_layout_arguments(world_size, max_numel, element_size, max_blocks);
  int64_t bytes = 0;
  TVM_FFI_ICHECK(fi::try_workspace_size(static_cast<int>(world_size), max_numel,
                                        static_cast<int>(element_size),
                                        static_cast<int>(max_blocks), &bytes))
      << "workspace layout overflows size_t or int64";
  return bytes;
}

/*!
 * \brief Bind one shared slab per rank and return a non-owning opaque handle.
 *
 * ipc_ptrs[i] addresses rank i's full slab. This rank's slab is zeroed here;
 * all ranks must rendezvous after init and before launching the first
 * collective so that no peer publication races the memset.
 */
fptr_t pcie_ipc_reduce_scatter_init(Array<fptr_t> ipc_ptrs, int64_t rank, int64_t max_numel,
                                    int64_t element_size, int64_t max_blocks) {
  const int world_size = static_cast<int>(ipc_ptrs.size());
  check_layout_arguments(world_size, max_numel, element_size, max_blocks);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "rank " << rank << " is out of range";

  int64_t ptrs[fi::kMaxWorldSize];
  for (int peer = 0; peer < world_size; ++peer) {
    TVM_FFI_ICHECK_NE(ipc_ptrs[peer], 0) << "ipc_ptrs[" << peer << "] is null";
    TVM_FFI_ICHECK_EQ(static_cast<uint64_t>(ipc_ptrs[peer]) & 15ULL, 0ULL)
        << "ipc_ptrs[" << peer << "] must be 16-byte aligned";
    ptrs[peer] = ipc_ptrs[peer];
  }

  fi::WorkspaceLayout layout{};
  TVM_FFI_ICHECK(fi::try_compute_workspace_layout(
      world_size, max_numel, static_cast<int>(element_size), static_cast<int>(max_blocks), &layout))
      << "workspace layout overflows size_t or int64";

  auto* handle = new PcieIpcReduceScatterHandle();
  handle->layout = layout;
  handle->views = fi::make_peer_views(ptrs, world_size, handle->layout);
  handle->rank = static_cast<int>(rank);
  cudaError_t err = cudaGetDevice(&handle->device);
  if (err != cudaSuccess) {
    delete handle;
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "failed to query the CUDA device: " << cudaGetErrorString(err);
  }

  err = cudaMemset(reinterpret_cast<void*>(ptrs[rank]), 0, handle->layout.total_bytes);
  if (err != cudaSuccess) {
    delete handle;
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "failed to zero the pcie ipc reduce-scatter workspace: " << cudaGetErrorString(err);
  }
  return reinterpret_cast<fptr_t>(handle);
}

void pcie_ipc_reduce_scatter_dispose(fptr_t handle) {
  delete reinterpret_cast<PcieIpcReduceScatterHandle*>(handle);
}

/*!
 * \brief Out-of-place SUM reduce-scatter.
 *
 * blocks and threads are the actual launch dimensions selected by the caller;
 * they are not policy limits and are never silently clamped in the binding.
 */
void pcie_ipc_reduce_scatter(fptr_t handle, TensorView inp, TensorView out, int64_t blocks,
                             int64_t threads, int64_t variant) {
  TVM_FFI_ICHECK_NE(handle, 0) << "pcie ipc reduce-scatter workspace is closed";
  auto* h = reinterpret_cast<PcieIpcReduceScatterHandle*>(handle);

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
      << "pcie ipc reduce-scatter supports bfloat16, float16 and float32";
  TVM_FFI_ICHECK_EQ(element_size, h->layout.element_size)
      << "tensor dtype does not match the workspace element size";

  const int64_t output_numel = out.numel();
  TVM_FFI_ICHECK_GT(output_numel, 0) << "output must be non-empty";
  TVM_FFI_ICHECK_EQ(output_numel % (common::kPackBytes / element_size), 0)
      << "output must contain complete " << common::kPackBytes << "-byte packs";
  TVM_FFI_ICHECK_LE(output_numel, h->layout.max_output_numel)
      << "output exceeds the workspace capacity";
  TVM_FFI_ICHECK_LE(output_numel, std::numeric_limits<int64_t>::max() / h->layout.world_size)
      << "input size overflows int64";
  TVM_FFI_ICHECK_EQ(inp.numel(), output_numel * h->layout.world_size)
      << "input numel must equal world_size * output numel";

  const uintptr_t input_begin = reinterpret_cast<uintptr_t>(inp.data_ptr());
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(out.data_ptr());
  const size_t output_bytes = static_cast<size_t>(output_numel) * element_size;
  const size_t input_bytes = output_bytes * h->layout.world_size;
  TVM_FFI_ICHECK_EQ(input_begin & 15ULL, 0ULL) << "input pointer must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(output_begin & 15ULL, 0ULL) << "output pointer must be 16-byte aligned";
  TVM_FFI_ICHECK(input_begin + input_bytes <= output_begin ||
                 output_begin + output_bytes <= input_begin)
      << "input and output must not overlap";

  TVM_FFI_ICHECK(blocks > 0 && blocks <= h->layout.max_blocks)
      << "blocks must be in (0, " << h->layout.max_blocks << "], got " << blocks;
  TVM_FFI_ICHECK(threads >= 32 && threads <= fi::kMaxThreads && threads % 32 == 0)
      << "threads must be warp-aligned and in [32, " << fi::kMaxThreads << "], got " << threads;
  TVM_FFI_ICHECK(variant >= 0 && variant < fi::kVariantCount)
      << "variant must be in [0, " << fi::kVariantCount << "), got " << variant;

  const auto algorithm = static_cast<fi::Variant>(variant);
  const bool flat_variant =
      algorithm == fi::Variant::kFlatCyclic || algorithm == fi::Variant::kFlatOnePack;
  const bool flat_world_size = h->layout.world_size == 2 || h->layout.world_size == 4;
  const bool topology_variant =
      algorithm == fi::Variant::kTopologyCyclic || algorithm == fi::Variant::kTopologyOnePack;
  TVM_FFI_ICHECK(!flat_variant || flat_world_size) << "flat variants require world_size 2 or 4";
  TVM_FFI_ICHECK(!topology_variant || h->layout.world_size == 8)
      << "4+4 topology variants require world_size 8";

  ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  const cudaStream_t stream = get_stream(inp.device());
  cudaError_t err = cudaSuccess;
  switch (encode_dlpack_dtype(inp.dtype())) {
    case bfloat16_code:
      err = fi::reduce_scatter(static_cast<const nv_bfloat16*>(inp.data_ptr()),
                               static_cast<nv_bfloat16*>(out.data_ptr()), output_numel, h->views,
                               h->rank, h->layout, static_cast<int>(blocks),
                               static_cast<int>(threads), algorithm, stream);
      break;
    case float16_code:
      err = fi::reduce_scatter(static_cast<const half*>(inp.data_ptr()),
                               static_cast<half*>(out.data_ptr()), output_numel, h->views, h->rank,
                               h->layout, static_cast<int>(blocks), static_cast<int>(threads),
                               algorithm, stream);
      break;
    case float32_code:
      err = fi::reduce_scatter(static_cast<const float*>(inp.data_ptr()),
                               static_cast<float*>(out.data_ptr()), output_numel, h->views, h->rank,
                               h->layout, static_cast<int>(blocks), static_cast<int>(threads),
                               algorithm, stream);
      break;
  }
  if (err != cudaSuccess) {
    TVM_FFI_LOG_AND_THROW(RuntimeError)
        << "pcie ipc reduce-scatter launch failed: " << cudaGetErrorString(err);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_reduce_scatter_workspace_size,
                              pcie_ipc_reduce_scatter_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_reduce_scatter_init, pcie_ipc_reduce_scatter_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_reduce_scatter_dispose, pcie_ipc_reduce_scatter_dispose);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_ipc_reduce_scatter, pcie_ipc_reduce_scatter);
