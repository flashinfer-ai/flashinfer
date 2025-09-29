/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

#include "../exception.h"
#ifndef _WIN32  // Linux
#include <sys/sysinfo.h>
#endif         // not WIN32
#ifdef _WIN32  // Windows
#include <windows.h>
#undef ERROR  // A Windows header file defines ERROR as 0, but it's used in our logger.h enum.
              // Logging breaks without this undef.
#endif        // WIN32

#include <cassert>
#include <cstdlib>   // for std::getenv
#include <cstring>   // for std::strcmp
#include <iostream>  // for std::cerr
#include <mutex>     // for std::once_flag and std::call_once

#define HOST_DEVICE_FUNC __host__ __device__
#define DEVICE_FUNC __device__

inline void cuErrCheck_(CUresult stat, char const* file, int line) {
  if (stat != CUDA_SUCCESS) {
    char const* msg = nullptr;
    cuGetErrorName(stat, &msg);
    fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
  }
}
#define cuErrCheck(stat)                     \
  {                                          \
    cuErrCheck_((stat), __FILE__, __LINE__); \
  }

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

inline constexpr int kMinHistoryTokensPerBlock = 128;

inline constexpr float kEnableMinBlockFactor = 4.0;
inline constexpr int kTargetWaveFactor = 8;

// For multi-block mode. We reserve workspace for this amount of sub-sequences.
// This should be enough. Huge batch size may result in larger value, but for
// large batch size, multi-block mode is not useful. For llama v2 70b, 6000
// results in ~12MB multi-block workspace, and is enough for > 10 waves.
inline constexpr int kXQA_MAX_NUM_SUB_SEQ = 6000;
inline constexpr int kMaxBeamWidth = 1;

inline int getDevice() {
  int current_dev_id = 0;
  CUDACHECK(cudaGetDevice(&current_dev_id));
  return current_dev_id;
}
inline int getSMVersion() {
  int device{-1};
  CUDACHECK(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDACHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CUDACHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
  if (useUvm) {
    size_t freeSysMem = 0;
    size_t totalSysMem = 0;
#ifndef _WIN32  // Linux
    struct sysinfo info{};

    sysinfo(&info);
    totalSysMem = info.totalram * info.mem_unit;
    freeSysMem = info.freeram * info.mem_unit;
#else   // Windows
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(memInfo);
    GlobalMemoryStatusEx(&memInfo);
    totalSysMem = memInfo.ullTotalPhys;
    freeSysMem = memInfo.ullAvailPhys;
#endif  // WIN32

    //  printf("Using UVM based system memory for KV cache, total memory %0.2f GB, available memory
    //%0.2f GB",
    //            ((double) totalSysMem / 1e9), ((double) freeSysMem / 1e9));
    //        return {freeSysMem, totalSysMem};
  }

  size_t free = 0;
  size_t total = 0;
  CUDACHECK(cudaMemGetInfo(&free, &total));
  //    printf("Using GPU memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
  //        ((double) total / 1e9), ((double) free / 1e9));
  return {free, total};
}

enum class LogLevel {
  LOG_NULL,
  LOG_ERROR,
  LOG_WARNING,
  LOG_INFO,
  LOG_DEBUG,
};

inline LogLevel parseLogLevel(const char* env) {
  if (env == nullptr) {
    return LogLevel::LOG_NULL;  // default
  }

  if (std::strcmp(env, "ERROR") == 0) return LogLevel::LOG_ERROR;
  if (std::strcmp(env, "WARNING") == 0) return LogLevel::LOG_WARNING;
  if (std::strcmp(env, "INFO") == 0) return LogLevel::LOG_INFO;
  if (std::strcmp(env, "DEBUG") == 0) return LogLevel::LOG_DEBUG;

  // Try numeric values as fallback
  if (std::strcmp(env, "1") == 0) return LogLevel::LOG_ERROR;
  if (std::strcmp(env, "2") == 0) return LogLevel::LOG_WARNING;
  if (std::strcmp(env, "3") == 0) return LogLevel::LOG_INFO;
  if (std::strcmp(env, "4") == 0) return LogLevel::LOG_DEBUG;

  return LogLevel::LOG_DEBUG;  // default if unrecognized
}

inline LogLevel getLogLevelFromEnv() {
  static std::once_flag init_flag;
  static LogLevel level = LogLevel::LOG_NULL;

  std::call_once(init_flag, []() {
    const char* env = std::getenv("FLASHINFER_LOG_LEVEL");
    level = parseLogLevel(env);
  });

  return level;
}
// Global log level access
inline LogLevel getCurrentLogLevel() { return getLogLevelFromEnv(); }

// Logging macros
#define IKL_LOG_DEBUG(fmt, ...)                        \
  do {                                                 \
    if (getCurrentLogLevel() >= LogLevel::LOG_DEBUG) { \
      printf("[DEBUG] " fmt "\n", ##__VA_ARGS__);      \
    }                                                  \
  } while (0)

#define IKL_LOG_INFO(fmt, ...)                        \
  do {                                                \
    if (getCurrentLogLevel() >= LogLevel::LOG_INFO) { \
      printf("[INFO] " fmt "\n", ##__VA_ARGS__);      \
    }                                                 \
  } while (0)

#define IKL_LOG_WARNING(fmt, ...)                        \
  do {                                                   \
    if (getCurrentLogLevel() >= LogLevel::LOG_WARNING) { \
      printf("[WARNING] " fmt "\n", ##__VA_ARGS__);      \
    }                                                    \
  } while (0)

#define IKL_LOG_ERROR(fmt, ...)                        \
  do {                                                 \
    if (getCurrentLogLevel() >= LogLevel::LOG_ERROR) { \
      printf("[ERROR] " fmt "\n", ##__VA_ARGS__);      \
    }                                                  \
  } while (0)

// Returns true if the env variable exists and is set to "1"
inline static bool getBoolEnv(char const* name) {
  char const* env = std::getenv(name);
  return env && env[0] == '1' && env[1] == '\0';
}

inline bool getEnvUseTileSizeKv64ForTrtllmGen() {
  static bool const useTileSizeKv64 = getBoolEnv("TRTLLM_GEN_ENABLE_TILE_SIZE_KV64");
  return useTileSizeKv64;
}
template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// For gen kernel IO
enum Data_type {
  DATA_TYPE_FP16,
  DATA_TYPE_BF16,
  DATA_TYPE_FP32,
  DATA_TYPE_INT8,
  DATA_TYPE_INT32,
  DATA_TYPE_E4M3,
  DATA_TYPE_E5M2,
  DATA_TYPE_E2M1,
  DATA_TYPE_UNKNOWN
};

inline constexpr const char* toStr(Data_type dtype) {
  switch (dtype) {
    case DATA_TYPE_FP16:
      return "FP16";
    case DATA_TYPE_BF16:
      return "BF16";
    case DATA_TYPE_FP32:
      return "FP32";
    case DATA_TYPE_INT8:
      return "INT8";
    case DATA_TYPE_INT32:
      return "INT32";
    case DATA_TYPE_E4M3:
      return "E4M3";
    case DATA_TYPE_E5M2:
      return "E5M2";
    case DATA_TYPE_E2M1:
      return "E2M1";
    default:
      return "UNKNOWN";
  }
}

// Type trait to map types to enum values
template <typename T>
struct TypeToDataType {
  static constexpr Data_type value = Data_type::DATA_TYPE_UNKNOWN;
};

// Specialize the trait for specific types
template <>
struct TypeToDataType<__nv_bfloat16> {
  static constexpr Data_type value = Data_type::DATA_TYPE_BF16;
};

template <>
struct TypeToDataType<__half> {
  static constexpr Data_type value = Data_type::DATA_TYPE_FP16;
};

template <>
struct TypeToDataType<uint8_t> {
  static constexpr Data_type value = Data_type::DATA_TYPE_E4M3;
};

template <>
struct TypeToDataType<__nv_fp8_e4m3> {
  static constexpr Data_type value = Data_type::DATA_TYPE_E4M3;
};

static inline size_t get_size_in_bytes(size_t n, Data_type dtype) {
  switch (dtype) {
    case DATA_TYPE_FP32:
      return n * 4;
    case DATA_TYPE_FP16:
      return n * 2;
    case DATA_TYPE_INT32:
      return n * 4;
    case DATA_TYPE_INT8:
      return n;
    case DATA_TYPE_BF16:
      return n * 2;
    case DATA_TYPE_E4M3:
      return n;
    case DATA_TYPE_E5M2:
      return n;
    default:
      FLASHINFER_CHECK(false, "FMHA Data Type is not supported.");
      return 0;
  }
}

static inline size_t get_size_in_bytes(Data_type dtype) { return get_size_in_bytes(1, dtype); }

static inline size_t get_size_in_bits(Data_type dtype) {
  switch (dtype) {
    case DATA_TYPE_FP32:
      return 32;
    case DATA_TYPE_FP16:
      return 16;
    case DATA_TYPE_INT32:
      return 32;
    case DATA_TYPE_INT8:
      return 8;
    case DATA_TYPE_BF16:
      return 16;
    case DATA_TYPE_E2M1:
      return 4;
    case DATA_TYPE_E4M3:
      return 8;
    case DATA_TYPE_E5M2:
      return 8;
    default:
      FLASHINFER_CHECK(false, "FMHA Data Type is not supported.");
      return 0;
  }
}
constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;
constexpr int32_t kSM_100 = 100;
constexpr int32_t kSM_100f = 10100;
constexpr int32_t kSM_103 = 103;
constexpr int32_t kSM_120 = 120;
