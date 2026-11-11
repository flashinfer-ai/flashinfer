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
// SM120 MegaMoE NIC transport session: the symmetric buffer as one NCCL window on the caller's
// communicator (torch ProcessGroupNCCL, borrowed), the load/store (LSA) peer pointer of every rank
// NCCL maps, and an NCCL device communicator with the GIN resources the kernel uses (the caller's
// context count, sm120::kGinSignals signals per context). Host-only: the kernel module is built per
// GIN backend (megamoe_sm120_mxfp4.cu with FLASHINFER_MEGAMOE_SM120_GIN=1).
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdint>
#include <flashinfer/megamoe/sm120_mxfp4_mega_moe.cuh>
#include <vector>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {
namespace {

inline void check_nccl(ncclResult_t rc, const char* what) {
  TVM_FFI_ICHECK(rc == ncclSuccess) << what << " failed: " << ncclGetErrorString(rc);
}
inline void check_cuda(cudaError_t rc, const char* what) {
  TVM_FFI_ICHECK(rc == cudaSuccess) << what << " failed: " << cudaGetErrorString(rc);
}

struct GinSession {
  ncclComm_t comm = nullptr;
  void* base = nullptr;
  size_t bytes = 0;
  ncclWindow_t win = nullptr;
  ncclDevComm host_dev_comm{};
  ncclDevComm* dev_comm = nullptr;  // device copy, read by the kernel
  bool dev_comm_created = false;
  int gin_type = 0;
  std::vector<int64_t> peer_ptrs;  // LSA mapping per rank, 0 when NCCL cannot map the pair

  void release() {
    if (dev_comm) cudaFree(dev_comm);
    if (dev_comm_created) ncclDevCommDestroy(comm, &host_dev_comm);
    if (win) ncclCommWindowDeregister(comm, win);
    if (base) ncclMemFree(base);
    dev_comm = nullptr, dev_comm_created = false, win = nullptr, base = nullptr;
  }
};

}  // namespace

// Collective over the communicator. Returns a handle for the calls below.
int64_t megamoe_sm120_gin_session_create(int64_t comm_ptr, int64_t rank, int64_t world,
                                         int64_t num_bytes, int64_t gin_contexts) {
  TVM_FFI_ICHECK(comm_ptr != 0) << "null NCCL communicator";
  auto* s = new GinSession();
  s->comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  s->bytes = static_cast<size_t>(num_bytes);
  try {
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    check_nccl(ncclCommQueryProperties(s->comm, &props), "ncclCommQueryProperties");
    TVM_FFI_ICHECK(props.rank == rank && props.nRanks == world)
        << "NCCL communicator is rank " << props.rank << " of " << props.nRanks;
    TVM_FFI_ICHECK(props.deviceApiSupport)
        << "the NCCL communicator does not support the Device API";
    s->gin_type = static_cast<int>(props.ginType);
    check_nccl(ncclMemAlloc(&s->base, s->bytes), "ncclMemAlloc");
    check_nccl(ncclCommWindowRegister(s->comm, s->base, s->bytes, &s->win, NCCL_WIN_COLL_SYMMETRIC),
               "ncclCommWindowRegister");
    check_cuda(cudaMemset(s->base, 0, s->bytes), "cudaMemset");
    s->peer_ptrs.assign(static_cast<size_t>(world), 0);
    for (int p = 0; p < world; ++p) {
      void* ptr = nullptr;
      if (ncclGetPeerDevicePointer(s->win, 0, p, &ptr) == ncclSuccess)
        s->peer_ptrs[p] = reinterpret_cast<int64_t>(ptr);
    }
    // NCCL's own mapping of the local window is another VA of the same memory; the kernel's
    // local accesses all go through the allocation itself
    s->peer_ptrs[rank] = reinterpret_cast<int64_t>(s->base);
    if (s->gin_type != NCCL_GIN_TYPE_NONE) {
      ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      req.ginContextCount = static_cast<int>(gin_contexts);
      req.ginSignalCount = sm120::kGinSignals;
      req.worldGinBarrierCount = 0;
      req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      req.ginStrongSignalsRequired = true;
      req.ginVaSignalsRequired = false;
      check_nccl(ncclDevCommCreate(s->comm, &req, &s->host_dev_comm), "ncclDevCommCreate");
      s->dev_comm_created = true;
      check_cuda(cudaMalloc(&s->dev_comm, sizeof(ncclDevComm)), "cudaMalloc");
      check_cuda(
          cudaMemcpy(s->dev_comm, &s->host_dev_comm, sizeof(ncclDevComm), cudaMemcpyHostToDevice),
          "cudaMemcpy");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  } catch (...) {
    s->release();
    delete s;
    throw;
  }
  return reinterpret_cast<int64_t>(s);
}

// [buffer pointer, ncclWindow_t, device ncclDevComm* (0 without GIN), GIN type, GIN contexts]
tvm::ffi::Array<int64_t> megamoe_sm120_gin_session_info(int64_t handle) {
  auto* s = reinterpret_cast<GinSession*>(handle);
  return tvm::ffi::Array<int64_t>{reinterpret_cast<int64_t>(s->base),
                                  reinterpret_cast<int64_t>(s->win),
                                  reinterpret_cast<int64_t>(s->dev_comm), s->gin_type,
                                  s->dev_comm_created ? s->host_dev_comm.ginContextCount : 0};
}

// LSA-mapped buffer pointer of every rank, 0 where NCCL does not map the pair
tvm::ffi::Array<int64_t> megamoe_sm120_gin_session_peers(int64_t handle) {
  auto* s = reinterpret_cast<GinSession*>(handle);
  return tvm::ffi::Array<int64_t>(s->peer_ptrs.begin(), s->peer_ptrs.end());
}

// Teardown, collective (window deregistration); no kernel may still use the buffer.
void megamoe_sm120_gin_session_close(int64_t handle) {
  auto* s = reinterpret_cast<GinSession*>(handle);
  cudaDeviceSynchronize();
  s->release();
  delete s;
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_gin_session_create,
                              flashinfer::megamoe::megamoe_sm120_gin_session_create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_gin_session_info,
                              flashinfer::megamoe::megamoe_sm120_gin_session_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_gin_session_peers,
                              flashinfer::megamoe::megamoe_sm120_gin_session_peers);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_gin_session_close,
                              flashinfer::megamoe::megamoe_sm120_gin_session_close);
