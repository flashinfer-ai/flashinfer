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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
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
kernel_cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4(__nv_bfloat16* __restrict__ v, float* __restrict__ v_amax_partial, int num_vectors, int num_chunks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* reduce_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float acc = 0.0f;
    unsigned int words[32];
    #pragma unroll 1
    for (int chunk = bid; chunk < num_chunks; chunk += 512) {
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int vector = (chunk * 4 + u) * 256 + tid;
            if (vector < num_vectors) {
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(words[u * 8 + 0]), "=r"(words[u * 8 + 1]), "=r"(words[u * 8 + 2]), "=r"(words[u * 8 + 3]), "=r"(words[u * 8 + 4]), "=r"(words[u * 8 + 5]), "=r"(words[u * 8 + 6]), "=r"(words[u * 8 + 7]) : "l"((const void*)((const char*)(v + ((long long)vector * 16) + 0) + 0)) : "memory");
                }
            } else {
                #pragma unroll
                for (int word = 0; word < 8; word++) {
                    words[u * 8 + word] = 0;
                }
            }
        }
        #pragma unroll
        for (int word_1 = 0; word_1 < 32; word_1++) {
            float _unpack_bf16_0[2];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((words + word_1)[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((words + word_1)[_pair]) & 0xffff0000u);
            }
            acc = fmaxf(acc, fabsf(_unpack_bf16_0[0]));
            acc = fmaxf(acc, fabsf(_unpack_bf16_0[1]));
        }
    }
    float _warp_reduce_1 = acc;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 = max_noftz(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
    acc = _warp_reduce_1;
    if (lane == 0) {
        reduce_smem[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        float cta_max_partial = ((lane < 8) ? reduce_smem[lane] : -CUDART_INF_F);
        float _warp_reduce_0 = cta_max_partial;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        if (lane == 0) {
            reduce_smem[0] = _warp_reduce_0;
        }
    }
    __syncthreads();
    float cta_max = reduce_smem[0];
    if (tid == 0) {
        *(reinterpret_cast<float*>(v_amax_partial + bid) + (0)) = cta_max;
    }
}

} // extern "C"
