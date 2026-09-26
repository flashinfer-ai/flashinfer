/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 32
#define SMEM_REDUCE_SMEM_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70(__nv_bfloat16* __restrict__ v, float* __restrict__ v_amax_partial, int num_vectors, int num_chunks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* reduce_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_smem_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float acc = 0.0f;
    float values[64];
    #pragma unroll 1
    for (int chunk = bid; chunk < num_chunks; chunk += 512) {
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int vector = (chunk * 4 + u) * 256 + tid;
            if (vector < num_vectors) {
                {
                    const void* _v8p_0 = (const void*)(v + ((long long)vector * 16));
                    uint32_t _v8_0_0[8];
                    asm volatile(
                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 0])[0]), "=f"((&values[u * 16 + 0])[1])
                        : "r"(_v8_0_0[0]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 2])[0]), "=f"((&values[u * 16 + 2])[1])
                        : "r"(_v8_0_0[1]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 4])[0]), "=f"((&values[u * 16 + 4])[1])
                        : "r"(_v8_0_0[2]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 6])[0]), "=f"((&values[u * 16 + 6])[1])
                        : "r"(_v8_0_0[3]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 8])[0]), "=f"((&values[u * 16 + 8])[1])
                        : "r"(_v8_0_0[4]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 10])[0]), "=f"((&values[u * 16 + 10])[1])
                        : "r"(_v8_0_0[5]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 12])[0]), "=f"((&values[u * 16 + 12])[1])
                        : "r"(_v8_0_0[6]));
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&values[u * 16 + 14])[0]), "=f"((&values[u * 16 + 14])[1])
                        : "r"(_v8_0_0[7]));
                }
            } else {
                #pragma unroll
                for (int element = 0; element < 16; element++) {
                    values[u * 16 + element] = 0.0f;
                }
            }
        }
        #pragma unroll
        for (int element_1 = 0; element_1 < 64; element_1++) {
            float _fabs_0 = fabsf(values[element_1]);
            float _max_0 = max_noftz(acc, _fabs_0);
            acc = _max_0;
        }
    }
    float _warp_reduce_0 = acc;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    acc = _warp_reduce_0;
    float _cross_warp_reduce_0;
    if (lane == 0) { reduce_smem[warp] = acc; }
    __syncthreads();
    if (warp == 0) {
        float _br = (lane < 8) ? reduce_smem[lane] : -CUDART_INF_F;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _br = fmaxf(_br, __shfl_xor_sync(0xFFFFFFFF, _br, offset));
        if (lane == 0) { reduce_smem[0] = _br; }
    }
    __syncthreads();
    _cross_warp_reduce_0 = reduce_smem[0];
    float cta_max = _cross_warp_reduce_0;
    if (tid == 0) {
        *(reinterpret_cast<float*>(v_amax_partial + bid) + (0)) = cta_max;
    }
}

} // extern "C"
