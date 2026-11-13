/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include "configs.cuh"
#include "exception.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[1];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 1
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
  int __num_sms = (sms);                          \
  int __num_threads = (threads);                  \
  auto __stream = (stream)
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
#define LAUNCH_KERNEL(config, kernel, ...) \
  CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else
#define LAUNCH_KERNEL(config, kernel, ...)                                           \
  do {                                                                               \
    kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__);                  \
    cudaError_t e = cudaGetLastError();                                              \
    if (e != cudaSuccess) {                                                          \
      EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
      fprintf(stderr, "%s\n", cuda_exception.what());                                \
      throw cuda_exception;                                                          \
    }                                                                                \
  } while (0)
#endif
#endif
