
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/NvInferRuntime.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/workspace.h"
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif  // ENABLE_MULTI_DEVICE

#include <nvml.h>

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

#ifdef ENABLE_MULTI_DEVICE
#define NCCLCHECK_THROW(cmd)                                                                    \
  do {                                                                                          \
    ncclResult_t r = cmd;                                                                       \
    if (TLLM_UNLIKELY(r != ncclSuccess)) {                                                      \
      TLLM_THROW("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    }                                                                                           \
  } while (0)

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap();

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group);

#endif  // ENABLE_MULTI_DEVICE
