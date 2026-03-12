/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * Minimal TensorRT-LLM utilities for standalone compilation
 * These are simplified implementations of TensorRT-LLM utilities
 * NOTE: This file will be replaced with FlashInfer equivalents during integration
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

// Namespace to match TensorRT-LLM
namespace tensorrt_llm {
namespace common {

// packed_as template - maps size to appropriate vector type
template <typename T, int N>
struct packed_as {
    static_assert(N == 1, "packed_as only supports N=1, 2, 4");
    using type = T;
};

template <>
struct packed_as<uint, 1> {
    using type = uint;
};

template <>
struct packed_as<uint, 2> {
    using type = uint2;
};

template <>
struct packed_as<uint, 4> {
    using type = uint4;
};

// Warp reduction sum
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// divUp - computes ceil(a / b) using integer arithmetic
// Based on TensorRT-LLM cudaUtils.h implementation
template <typename T1, typename T2>
__host__ __device__ __forceinline__ size_t divUp(T1 const& a, T2 const& b) {
    auto const tmp_a = static_cast<size_t>(a);
    auto const tmp_b = static_cast<size_t>(b);
    return (tmp_a + tmp_b - 1) / tmp_b;
}

// roundUp - rounds a up to the nearest multiple of b
template <typename T1, typename T2>
__host__ __device__ __forceinline__ size_t roundUp(T1 const& a, T2 const& b) {
    return divUp(a, b) * static_cast<size_t>(b);
}

} // namespace common
} // namespace tensorrt_llm

// Macros for error checking - work on both host and device
#ifdef __CUDA_ARCH__
// Device code path
#define TLLM_CHECK(condition) \
    do { \
        if (!(condition)) { \
            printf("TLLM_CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            __trap(); \
        } \
    } while(0)

#define TLLM_THROW(fmt, ...) \
    do { \
        printf("TLLM_THROW at %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        __trap(); \
    } while(0)
#else
// Host code path
#include <cstdio>
#include <cstdlib>

#define TLLM_CHECK(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "TLLM_CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            abort(); \
        } \
    } while(0)

#define TLLM_THROW(fmt, ...) \
    do { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), fmt, ##__VA_ARGS__); \
        fprintf(stderr, "TLLM_THROW at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        abort(); \
    } while(0)
#endif

