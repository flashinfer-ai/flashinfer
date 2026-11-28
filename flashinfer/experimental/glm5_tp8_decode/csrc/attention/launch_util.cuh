/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../attention_api.h"

namespace flashinfer::glm5_tp8::attention_detail {

inline void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct DeviceGuard {
  int old;
  explicit DeviceGuard(int device) : old(-1) {
    check(cudaGetDevice(&old), "cudaGetDevice");
    if (old != device) check(cudaSetDevice(device), "cudaSetDevice");
  }
  ~DeviceGuard() {
    if (old >= 0) (void)cudaSetDevice(old);
  }
};

inline KernelResources configure(const void* kernel, int grid, int block, int shared, int device) {
  DeviceGuard guard(device);
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared),
        "attention shared-memory opt-in");
  cudaFuncAttributes attr{};
  check(cudaFuncGetAttributes(&attr, kernel), "attention module warmup");
  int active = 0, sms = 0;
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, block, shared),
        "attention occupancy");
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "attention SM count");
  if (active < 1 || active * sms < grid)
    throw std::runtime_error("attention grid cannot be fully resident");
  return {grid,
          block,
          shared,
          attr.numRegs,
          static_cast<int>(attr.sharedSizeBytes),
          static_cast<int>(attr.localSizeBytes),
          active};
}

template <class Args>
inline cudaError_t launch(const void* kernel, const Args& args, int grid, int block, int shared,
                          cudaStream_t stream, bool pdl, bool cooperative) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(block);
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  cudaLaunchAttribute attributes[2]{};
  int count = 0;
  if (pdl) {
    attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[count++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (cooperative) {
    attributes[count].id = cudaLaunchAttributeCooperative;
    attributes[count++].val.cooperative = 1;
  }
  config.attrs = attributes;
  config.numAttrs = count;
  void* params[] = {const_cast<Args*>(&args)};
  return cudaLaunchKernelExC(&config, kernel, params);
}

}  // namespace flashinfer::glm5_tp8::attention_detail
