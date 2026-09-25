/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_CE_MEMOP_CUH_
#define FLASHINFER_COMM_PCIE_IPC_CE_MEMOP_CUH_

#include <cuda.h>

#include "pcie_ipc_ce_sm120.cuh"

namespace flashinfer {
namespace comm {
namespace pcie_ipc {

// Per-rank allocation: 2*(world_size-1) cache-line-separated flags, zeroed once.
inline size_t ce_binary_flag_bytes(int world_size) {
  return static_cast<size_t>(2 * (world_size - 1)) * 128u;
}

inline cudaError_t ce_binary_write(cudaStream_t stream, int32_t* flag, uint32_t value) {
  CUresult rc = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream),
                                     reinterpret_cast<CUdeviceptr>(flag), value, 0);
  return rc == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

inline cudaError_t ce_binary_wait_clear(cudaStream_t stream, int32_t* flag) {
  kernel_pcie_ipc_ce_sm120_binary_wait_clear<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(flag));
  return cudaGetLastError();
}

}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer
#endif
